"""Interactive arm bring-up: jog joints, verify servo calibration, test IK & grip.

Run this right after flashing the firmware and setting ARM_PORT. It confirms the
arm moves the way config.py assumes BEFORE the full loop relies on it — this is
what turns "guessed constants" into "verified constants".

Commands (type + ENTER):
  h              home pose
  j <i> <deg>    jog joint i (1-6) to servo angle deg   e.g.  j 1 120
  g open|close   gripper
  ik <x> <y> [z] move tip to workspace point via IK      e.g.  ik 10 5 2
  s              print last commanded pose
  q              quit

Motor map after removing the old unused servo3:
  1 D13 base yaw       2 D12 shoulder       3 D11 elbow
  4 D10 wrist pitch    5 D9  gripper        6 D8  wrist roll

Checklist while jogging:
  * Does joint 1 (base) rotate the whole arm? Increasing deg -> which way?
    If backwards, flip config.SERVO_DIRECTION[0].
  * At HOME, is each joint at a sensible neutral? If not, adjust SERVO_OFFSET.
  * Run `ik 0 15` etc. and check the tip lands near that spot on the table.
    Consistent offset -> refine link lengths L_* or the servo offsets.
"""
import argparse

import config
from arm_firmware import upload_arm_firmware
from arm_serial import ArmSerial
from policy import Policy


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Safely upload and/or jog the six-servo robot arm")
    parser.add_argument(
        "--upload", action="store_true",
        help="compile/upload arm_controller before connecting (resets/moves arm)")
    args = parser.parse_args(argv)

    port = upload_arm_firmware() if args.upload else None
    arm = ArmSerial(port=port)
    policy = Policy()
    if getattr(arm, "mock", False):
        print("NOTE: ARM_MOCK=True — commands are printed only.")
    pose = None

    try:
        pose = arm.status()
        print(f"현재 자세: {pose}")
        print(__doc__)
        while True:
            cmd = input("jog> ").strip().split()
            if not cmd:
                continue
            c = cmd[0].lower()
            try:
                if c == "q":
                    break
                elif c == "h":
                    next_pose = list(config.HOME_POSE)
                    arm.send_angles(next_pose); arm.wait_done(); pose = next_pose
                elif c == "j" and len(cmd) == 3:
                    i = int(cmd[1]) - 1
                    if not 0 <= i < config.N_JOINTS:
                        raise ValueError(f"joint must be 1..{config.N_JOINTS}")
                    next_pose = list(pose)
                    next_pose[i] = int(cmd[2])
                    arm.send_angles(next_pose); arm.wait_done(); pose = next_pose
                elif c == "g" and len(cmd) == 2:
                    action = cmd[1].lower()
                    if action not in ("open", "close"):
                        raise ValueError("gripper action must be open or close")
                    arm.gripper(open_=(action == "open")); arm.wait_done()
                    pose[config.J_GRIP] = (config.GRIP_OPEN if action == "open"
                                           else config.GRIP_CLOSED)
                elif c == "ik" and len(cmd) in (3, 4):
                    x, y = float(cmd[1]), float(cmd[2])
                    z = float(cmd[3]) if len(cmd) == 4 else 0.0
                    next_pose = policy.target_to_angles((x, y), z=z)
                    print(f"  IK({x},{y},{z}) -> {next_pose}")
                    arm.send_angles(next_pose); arm.wait_done(); pose = next_pose
                elif c == "s":
                    print(f"  pose = {pose}")
                else:
                    print("commands: h | j <i> <deg> | g open|close | ik <x> <y> [z] | s | q")
            except (ValueError, TypeError, RuntimeError, TimeoutError) as exc:
                print(f"error: {exc}")
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        if pose is not None:
            print(f"현재 자세를 유지하고 연결을 닫습니다: {pose}")
        arm.close()


if __name__ == "__main__":
    main()
