"""Inverse kinematics for the arm: workspace point (x,y,z) -> 7 servo commands.

Model: base yaw about Z, then a planar 2-link arm (upper + forearm) in the
vertical plane containing the target, then the hand. This matches the user's
build: servo1 base yaw, servo2 shoulder (1st bend), servo4 elbow (2nd bend),
servo5/6 wrist, servo7 gripper.

Coordinates: workspace origin at the base axis on the table. x,y horizontal
(same frame as the overhead camera), z up. Units cm (config link lengths).

If a target is unreachable the solution is clamped to the nearest reachable
pose rather than throwing, so the loop degrades gracefully.
"""
import math
import config


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def joint_to_servo(i, joint_deg):
    """Map an IK joint angle to a servo.write() value using per-joint calib."""
    cmd = config.SERVO_OFFSET[i] + config.SERVO_DIRECTION[i] * joint_deg
    return int(round(_clamp(cmd, config.SERVO_MIN[i], config.SERVO_MAX[i])))


def solve(x, y, z=0.0, approach_from_above=True):
    """Return a full 7-element servo command list to place the gripper at (x,y,z).

    Joints solved: base yaw, shoulder, elbow. Wrist is set to keep the hand
    roughly level (or pointing down for a top grasp). Gripper left as-is (HOME).
    """
    L1 = config.L_UPPER
    L2 = config.L_FORE
    Lh = config.L_HAND

    # --- base yaw: rotate to face the target in the XY plane ---
    yaw = math.degrees(math.atan2(y, x))            # 0 = +x axis

    # --- reduce to the vertical plane: radial distance and height ---
    r = math.hypot(x, y)
    # if grasping from above, the hand points down: subtract hand length from the
    # height the 2-link arm must reach, and aim the wrist straight for radius r.
    wrist_z = z + (Lh if approach_from_above else 0.0)
    dz = wrist_z - config.L_BASE_HEIGHT             # height of wrist above shoulder
    dr = r                                          # radial reach to wrist

    dist = math.hypot(dr, dz)
    reach = L1 + L2
    if dist > reach:                                # clamp to max reach
        dist = reach - 1e-3
    if dist < abs(L1 - L2):                         # clamp to min reach
        dist = abs(L1 - L2) + 1e-3

    # law of cosines for the 2-link planar arm
    cos_elbow = (L1 * L1 + L2 * L2 - dist * dist) / (2 * L1 * L2)
    elbow_inner = math.degrees(math.acos(_clamp(cos_elbow, -1, 1)))

    cos_sh = (L1 * L1 + dist * dist - L2 * L2) / (2 * L1 * dist)
    sh_a = math.degrees(math.acos(_clamp(cos_sh, -1, 1)))
    sh_b = math.degrees(math.atan2(dz, dr))
    shoulder = sh_a + sh_b                           # from horizontal, up positive
    elbow = elbow_inner - 180                        # 0 = straight arm

    # wrist keeps the hand pointing down for a top grasp: compensate the arm's
    # accumulated angle so the hand stays vertical.
    if approach_from_above:
        wrist = -(shoulder + elbow) - 90
    else:
        wrist = -(shoulder + elbow)

    angles = list(config.HOME_POSE)                  # start from safe pose (servo units)
    angles[config.J_BASE]     = joint_to_servo(config.J_BASE, yaw)
    angles[config.J_SHOULDER] = joint_to_servo(config.J_SHOULDER, shoulder)
    angles[config.J_ELBOW]    = joint_to_servo(config.J_ELBOW, elbow)
    angles[config.J_WRIST]    = joint_to_servo(config.J_WRIST, wrist)
    return angles


def reachable(x, y, z=0.0):
    r = math.hypot(x, y)
    dz = z + config.L_HAND - config.L_BASE_HEIGHT
    dist = math.hypot(r, dz)
    return abs(config.L_UPPER - config.L_FORE) <= dist <= (config.L_UPPER + config.L_FORE)


if __name__ == "__main__":
    for pt in [(12, 4), (8, -3), (-6, 9), (0, 15)]:
        print(pt, "reachable" if reachable(*pt) else "OUT",
              "->", solve(*pt))
