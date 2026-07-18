"""Interactive arm bring-up: jog joints, verify servo calibration, test IK & grip.

Run this right after flashing the firmware and setting ARM_PORT. It confirms the
arm moves the way config.py assumes BEFORE the full loop relies on it — this is
what turns "guessed constants" into "verified constants".

Commands (type + ENTER):
  h              home pose
  j <i> <deg>    jog joint i (1-7) to servo angle deg   e.g.  j 1 120
  g open|close   gripper
  ik <x> <y> [z] move tip to workspace point via IK      e.g.  ik 10 5 2
  s              print last commanded pose
  q              quit

Checklist while jogging:
  * Does joint 1 (base) rotate the whole arm? Increasing deg -> which way?
    If backwards, flip config.SERVO_DIRECTION[0].
  * At HOME, is each joint at a sensible neutral? If not, adjust SERVO_OFFSET.
  * Run `ik 0 15` etc. and check the tip lands near that spot on the table.
    Consistent offset -> refine link lengths L_* or the servo offsets.
"""
import config
from arm_serial import ArmSerial
from policy import Policy


def main():
    arm = ArmSerial()
    policy = Policy()
    if getattr(arm, "mock", False):
        print("NOTE: no board detected — running in MOCK (prints commands only).")
    arm.home(); arm.wait_done()
    pose = list(config.HOME_POSE)
    print(__doc__)

    while True:
        try:
            cmd = input("jog> ").strip().split()
        except EOFError:
            break
        if not cmd:
            continue
        c = cmd[0].lower()
        if c == "q":
            break
        elif c == "h":
            pose = list(config.HOME_POSE); arm.send_angles(pose); arm.wait_done()
        elif c == "j" and len(cmd) == 3:
            i = int(cmd[1]) - 1
            if 0 <= i < config.N_JOINTS:
                pose[i] = int(cmd[2])
                arm.send_angles(pose); arm.wait_done()
            else:
                print("joint must be 1..7")
        elif c == "g" and len(cmd) == 2:
            arm.gripper(open_=(cmd[1].lower() == "open")); arm.wait_done()
        elif c == "ik" and len(cmd) >= 3:
            x, y = float(cmd[1]), float(cmd[2])
            z = float(cmd[3]) if len(cmd) > 3 else 0.0
            pose = policy.target_to_angles((x, y), z=z)
            print(f"  IK({x},{y},{z}) -> {pose}")
            arm.send_angles(pose); arm.wait_done()
        elif c == "s":
            print(f"  pose = {pose}")
        else:
            print("commands: h | j <i> <deg> | g open|close | ik <x> <y> [z] | s | q")
    arm.home(); arm.close()


if __name__ == "__main__":
    main()
