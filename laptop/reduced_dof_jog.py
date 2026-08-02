"""Interactive manual test for the reduced arm; only 2, 3, and 5 can move.

Start ``reduced_dof_session.py serve`` first.  This client never opens the Uno
again, so manual commands do not cause repeated DTR resets.
"""

import config
from reduced_dof import canonicalize_status, command_pose, reduced_home
from reduced_dof_session import client


HELP = """명령:
  j 2 <각도>     2번 어깨 이동
  j 3 <각도>     3번 팔꿈치 이동
  g open|close   5번 집게 열기/닫기
  h              축소 자유도 HOME
  d              초음파 거리
  s              현재 명령 자세
  q              종료

1·4·6번은 고정 축이므로 이 프로그램에서 움직일 수 없습니다.
"""


def main():
    arm = client()
    pose = canonicalize_status(
        arm.request({"command": "status"})["pose"])
    print(f"현재 자세: {pose}")
    print(HELP)
    while True:
        try:
            parts = input("reduced-jog> ").strip().split()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not parts:
            continue
        try:
            command = parts[0].lower()
            if command == "q":
                break
            if command == "s":
                pose = canonicalize_status(
                    arm.request({"command": "status"})["pose"])
                print(f"현재 자세: {pose}")
                continue
            if command == "d":
                result = arm.request({"command": "distance", "samples": 3})
                print(f"초음파 거리: {result.get('distanceMm')} mm")
                continue
            if command == "h":
                target = reduced_home(config.GRIP_OPEN)
            elif command == "g" and len(parts) == 2:
                if parts[1].lower() not in ("open", "close"):
                    raise ValueError("g open 또는 g close를 입력하세요.")
                gripper = (
                    config.GRIP_OPEN if parts[1].lower() == "open"
                    else config.GRIP_CLOSED)
                target = command_pose(
                    pose[config.J_SHOULDER], pose[config.J_ELBOW], gripper)
            elif command == "j" and len(parts) == 3:
                joint = int(parts[1])
                if joint not in (2, 3):
                    raise ValueError("현재 움직일 수 있는 관절은 2번과 3번뿐입니다.")
                target = list(pose)
                target[joint - 1] = int(parts[2])
                target = command_pose(
                    target[config.J_SHOULDER],
                    target[config.J_ELBOW],
                    target[config.J_GRIP],
                )
            else:
                print(HELP)
                continue
            result = arm.request({"command": "move", "pose": target})
            pose = canonicalize_status(result["pose"])
            print(f"완료: {pose}")
        except (TypeError, ValueError, RuntimeError, TimeoutError) as exc:
            print(f"거부됨: {exc}")
    print(f"현재 자세를 유지하고 JOG만 종료합니다: {pose}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
