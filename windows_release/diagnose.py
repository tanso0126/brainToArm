"""Read-only Windows hardware discovery report."""

from pathlib import Path
import sys

from serial.tools import list_ports


RELEASE = Path(__file__).resolve().parent
ROOT = RELEASE.parent
sys.path.insert(0, str(ROOT / "laptop"))
sys.path.insert(0, str(RELEASE))

from windows_camera import choose_camera  # noqa: E402
from windows_support import find_arm_port, port_description  # noqa: E402


def main():
    print("=== Serial ports ===")
    ports = list(list_ports.comports())
    if not ports:
        print("(none)")
    for port in ports:
        print(port_description(port))
    try:
        print(f"Selected arm port: {find_arm_port(ports=ports)}")
    except Exception as exc:
        print(f"Arm auto-selection: FAILED - {exc}")

    print("\n=== Cameras ===")
    try:
        index, camera, frame = choose_camera("auto")
        print(
            f"Selected wrist camera: index {index}, "
            f"{frame.shape[1]}x{frame.shape[0]}")
        camera.release()
    except Exception as exc:
        print(f"Camera auto-selection: FAILED - {exc}")

    print("\nThis diagnostic does not open or move the robot arm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
