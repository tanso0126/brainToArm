"""Windows에서 한 번에 물체를 탐색하고, 접근하고, 잡고, HOME으로 복귀합니다."""

from pathlib import Path
import argparse
import json
import subprocess
import sys
import time


RELEASE = Path(__file__).resolve().parent
ROOT = RELEASE.parent
LAPTOP = ROOT / "laptop"
ASSET = RELEASE / "assets" / "FastSAM-s.pt"
CAMERA_READY = ROOT / "data" / "runtime" / "windows_camera.json"
RAW_FRAME = ROOT / "data" / "vision" / "wrist_camera_latest_raw.jpg"
STATE_KO = {
    "home-after-grasp": "물체를 잡은 뒤 HOME 복귀 완료",
    "no-target": "물체를 찾지 못함",
    "target-lost": "접근 중 물체 추적을 잃음",
    "grasp-ready": "초음파 정지 거리에서 잡기 준비 완료",
    "grasp-ready-floor": "바닥 정지 높이에서 잡기 준비 완료",
    "safe-reach-exhausted": "안전하게 더 접근할 수 없어 중단",
    "planned": "이동 계획만 계산함",
    "time-limit": "제한 시간 초과",
}
sys.path.insert(0, str(LAPTOP))
sys.path.insert(0, str(RELEASE))

import config  # noqa: E402
from windows_support import (  # noqa: E402
    DirectArmClient,
    find_arm_port,
    open_arm,
)


def start_camera(camera, headless=False):
    command = [
        sys.executable, "-u", str(RELEASE / "windows_camera.py"),
        "--camera", str(camera),
    ]
    if headless:
        command.append("--headless")
    try:
        CAMERA_READY.unlink()
    except FileNotFoundError:
        pass
    process = subprocess.Popen(command, cwd=str(ROOT))
    deadline = time.monotonic() + 45.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "카메라 프로그램이 준비되기 전에 종료되었습니다. "
                f"종료 코드: {process.returncode}\n"
                "CHECK_CAMERA.bat을 실행해 카메라 권한과 화면을 확인하세요.")
        try:
            data = json.loads(CAMERA_READY.read_text(encoding="utf-8"))
            age = time.time() - RAW_FRAME.stat().st_mtime
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.1)
            continue
        if data.get("pid") == process.pid and age <= 1.0:
            return process, data
        time.sleep(0.1)
    process.terminate()
    raise TimeoutError(
        "45초 안에 손목 카메라가 준비되지 않았습니다. 웹캠 USB 연결, "
        "Windows 카메라 권한, 카메라를 사용 중인 다른 앱을 확인하세요.")


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2.0)


def run(camera="auto", port="auto", max_seconds=90.0, headless=False):
    if not ASSET.exists():
        raise RuntimeError(
            f"함께 제공된 AI 모델을 찾을 수 없습니다: {ASSET}\n"
            "저장소를 다시 내려받거나 SETUP_WINDOWS.bat을 실행하세요.")
    config.PLANAR_VISION_MODEL = str(ASSET)
    config.PLANAR_VISION_DEVICE = "cpu"
    publisher = None
    client = None
    try:
        print("[1/4] 손목 카메라를 찾고 있습니다...", flush=True)
        publisher, camera_info = start_camera(camera, headless=headless)
        print(
            f"[2/4] 카메라 {camera_info['cameraIndex']}번이 연결되었습니다. "
            "Arduino Uno를 찾고 있습니다...",
            flush=True,
        )
        arm_port = find_arm_port(port)
        print(f"[로봇팔] {arm_port} 포트를 선택했습니다.", flush=True)
        arm = open_arm(arm_port)
        client = DirectArmClient(arm)

        import realtime_visual_servo as controller
        controller.ArmSessionClient = lambda: client
        print(
            "[3/4] 자동 실행을 시작합니다. 로봇팔 이동 범위에 "
            "손이나 케이블을 넣지 마세요.",
            flush=True,
        )
        result = controller.run(
            execute=True,
            allow_grasp=True,
            max_seconds=float(max_seconds),
        )
        state = result.get("state", "알 수 없음")
        print(
            f"[실행 결과] {STATE_KO.get(state, state)}\n"
            f"[상세 정보] {json.dumps(result, ensure_ascii=False)}",
            flush=True,
        )
        if result.get("state") != "home-after-grasp":
            raise RuntimeError(
                "물체 잡기 또는 HOME 복귀가 끝나기 전에 자동 실행이 "
                f"중단되었습니다. 마지막 상태: "
                f"{STATE_KO.get(state, state)}")
        print(
            "[4/4] 물체를 잡고 HOME 복귀 명령까지 완료했습니다.",
            flush=True,
        )
        return 0
    finally:
        if client is not None:
            client.close()
        stop_process(publisher)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="auto",
                        help="카메라 번호 또는 자동 선택(auto)")
    parser.add_argument("--port", default="auto",
                        help="Arduino COM 포트 또는 자동 선택(auto)")
    parser.add_argument(
        "--max-seconds", type=float, default=90.0,
        help="자동 실행 제한 시간(초), 기본값 90")
    parser.add_argument(
        "--headless", action="store_true",
        help="카메라 미리보기 창을 열지 않음")
    args = parser.parse_args()
    try:
        return run(
            camera=args.camera,
            port=args.port,
            max_seconds=args.max_seconds,
            headless=args.headless,
        )
    except KeyboardInterrupt:
        print("\n[사용자 중단] 키보드 입력으로 실행을 중단했습니다.", flush=True)
        return 130
    except Exception as exc:
        print(
            f"\n[안전 중단] {type(exc).__name__}: {exc}\n"
            "로봇팔 전원을 확인한 뒤 위 오류에 적힌 조치를 수행하세요. "
            "연결 상태는 DIAGNOSE.bat으로 확인할 수 있습니다.",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
