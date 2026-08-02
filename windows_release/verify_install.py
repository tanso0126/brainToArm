"""실물 하드웨어 없이 Windows 설치 결과를 확인합니다."""

from pathlib import Path
import hashlib
import importlib
import sys


RELEASE = Path(__file__).resolve().parent
ROOT = RELEASE.parent
ASSET = RELEASE / "assets" / "FastSAM-s.pt"
EXPECTED_MODEL_SHA256 = (
    "c9f78716a81c7aff0d608ccc73e1b82ab3aaad86005049f6a92106a0be6d0844")


def main():
    required = (
        "numpy", "cv2", "serial", "ultralytics", "scipy", "hid",
        "mujoco", "webview")
    for name in required:
        importlib.import_module(name)
        print(f"[설치 확인] {name} 불러오기: 정상")
    if not ASSET.exists() or ASSET.stat().st_size < 20_000_000:
        raise RuntimeError(
            "함께 제공되어야 할 FastSAM AI 모델이 없거나 파일이 "
            f"불완전합니다: {ASSET}\n저장소를 다시 내려받아 압축을 "
            "완전히 푼 뒤 설치를 다시 실행하세요.")
    digest = hashlib.sha256(ASSET.read_bytes()).hexdigest()
    if digest != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            "FastSAM AI 모델 파일이 손상되었습니다. 저장소를 다시 "
            f"내려받으세요. 확인값: {digest}")
    sys.path.insert(0, str(ROOT / "laptop"))
    sys.path.insert(0, str(RELEASE))
    import config
    config.PLANAR_VISION_MODEL = str(ASSET)
    from vision_segment import FastSAMDetector
    FastSAMDetector()
    for name in (
            "windows_app", "windows_camera", "windows_support",
            "control_service", "control_center"):
        importlib.import_module(name)
    dashboard = ROOT / "dashboard"
    if not (dashboard / "dist" / "client").is_dir():
        raise RuntimeError(
            "통합 GUI 빌드 파일이 없습니다. SETUP_WINDOWS.bat을 다시 "
            "실행하세요.")
    for model in (
            ROOT / "simul" / "models" / "full_task_policy_v1.ts",
            ROOT / "simul" / "models" / "reduced_dof_policy_v1.ts"):
        if not model.is_file():
            raise RuntimeError(f"시뮬레이션 학습 모델이 없습니다: {model}")
    print(f"[설치 확인] FastSAM AI 모델: 정상 ({digest[:12]}...)")
    print("[설치 확인] Windows 실행 모듈: 정상")
    print("[설치 확인] 통합 GUI와 3D 학습 모델: 정상")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
