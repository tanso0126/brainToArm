"""One-command Windows autonomous find, approach, grasp, and HOME."""

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
                f"camera process exited with code {process.returncode}")
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
    raise TimeoutError("wrist camera did not become ready within 45 seconds")


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
            f"bundled AI model is missing: {ASSET}. "
            "Re-download the repository or run SETUP_WINDOWS.bat.")
    config.PLANAR_VISION_MODEL = str(ASSET)
    config.PLANAR_VISION_DEVICE = "cpu"
    publisher = None
    client = None
    try:
        print("[1/4] Finding the wrist camera...", flush=True)
        publisher, camera_info = start_camera(camera, headless=headless)
        print(
            f"[2/4] Camera {camera_info['cameraIndex']} is live. "
            "Finding the Arduino Uno...",
            flush=True,
        )
        arm_port = find_arm_port(port)
        print(f"[arm] selected {arm_port}", flush=True)
        arm = open_arm(arm_port)
        client = DirectArmClient(arm)

        import realtime_visual_servo as controller
        controller.ArmSessionClient = lambda: client
        print(
            "[3/4] Autonomous run started. Keep the workspace clear.",
            flush=True,
        )
        result = controller.run(
            execute=True,
            allow_grasp=True,
            max_seconds=float(max_seconds),
        )
        print("[result] " + json.dumps(result, ensure_ascii=False), flush=True)
        if result.get("state") != "home-after-grasp":
            raise RuntimeError(
                "autonomous run stopped before grasp/HOME: "
                f"{result.get('state', 'unknown')}")
        print("[4/4] Object grasped and HOME command completed.", flush=True)
        return 0
    finally:
        if client is not None:
            client.close()
        stop_process(publisher)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="auto",
                        help="DirectShow camera index, or auto")
    parser.add_argument("--port", default="auto",
                        help="Arduino COM port, or auto")
    parser.add_argument("--max-seconds", type=float, default=90.0)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    try:
        return run(
            camera=args.camera,
            port=args.port,
            max_seconds=args.max_seconds,
            headless=args.headless,
        )
    except KeyboardInterrupt:
        print("\n[STOP] Operator interrupted the run.", flush=True)
        return 130
    except Exception as exc:
        print(
            f"\n[SAFE STOP] {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
