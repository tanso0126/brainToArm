"""로봇팔을 움직이지 않고 Windows 하드웨어 연결 상태를 확인합니다."""

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
    print("=== 시리얼(COM) 포트 ===")
    ports = list(list_ports.comports())
    if not ports:
        print("(감지된 포트가 없습니다)")
    for port in ports:
        print(port_description(port))
    try:
        print(f"자동 선택한 로봇팔 포트: {find_arm_port(ports=ports)}")
    except Exception as exc:
        print(f"로봇팔 포트 자동 선택: 실패 - {exc}")

    print("\n=== 카메라 ===")
    try:
        index, camera, frame = choose_camera("auto")
        print(
            f"자동 선택한 손목 카메라: 번호 {index}, "
            f"{frame.shape[1]}x{frame.shape[0]}")
        camera.release()
    except Exception as exc:
        print(f"카메라 자동 선택: 실패 - {exc}")

    print("\n이 진단 도구는 로봇팔을 열거나 움직이지 않습니다.")
    print("문제가 있으면 이 창 전체가 보이도록 사진을 찍어 전달하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
