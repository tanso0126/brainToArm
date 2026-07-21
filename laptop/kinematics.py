"""Inverse kinematics for the arm: workspace point (x,y,z) -> 6 servo commands.

Model: base yaw about Z, then a planar 2-link arm (upper + forearm) in the
vertical plane containing the target, then the hand. This matches the user's
build: servo1 base yaw, servo2 shoulder, servo3 elbow, servo4/6 wrist, and
servo5 gripper.

Coordinates: workspace origin at the base axis on the table. x,y horizontal
(same frame as the overhead camera), z up. Units cm (config link lengths).

Unreachable or non-finite targets are rejected. Quietly clamping a physical
target would move the gripper somewhere other than the camera-observed object.
"""
import math
import config


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def joint_to_servo(i, joint_deg):
    """Map an IK joint angle to a servo.write() value using per-joint calib."""
    cmd = config.SERVO_OFFSET[i] + config.SERVO_DIRECTION[i] * joint_deg
    if config.ARM_CALIBRATED and not config.SERVO_MIN[i] <= cmd <= config.SERVO_MAX[i]:
        raise ValueError(
            f"IK requires joint {i + 1} servo command {cmd:.1f}, outside calibrated "
            f"range [{config.SERVO_MIN[i]}, {config.SERVO_MAX[i]}]")
    return int(round(_clamp(cmd, config.SERVO_MIN[i], config.SERVO_MAX[i])))


def solve(x, y, z=0.0, approach_from_above=True):
    """Return a full 6-element servo command list to place the gripper at (x,y,z).

    Joints solved: base yaw, shoulder, elbow. Wrist is set to keep the hand
    roughly level (or pointing down for a top grasp). Gripper left as-is (HOME).
    """
    if not all(math.isfinite(v) for v in (x, y, z)):
        raise ValueError(f"target coordinates must be finite: {(x, y, z)}")
    if not reachable(x, y, z, approach_from_above=approach_from_above):
        raise ValueError(f"target is outside geometric reach: {(x, y, z)}")

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


def reachable(x, y, z=0.0, approach_from_above=True):
    if not all(math.isfinite(v) for v in (x, y, z)):
        return False
    r = math.hypot(x, y)
    hand_z = config.L_HAND if approach_from_above else 0.0
    dz = z + hand_z - config.L_BASE_HEIGHT
    dist = math.hypot(r, dz)
    return abs(config.L_UPPER - config.L_FORE) <= dist <= (config.L_UPPER + config.L_FORE)


if __name__ == "__main__":
    for pt in [(12, 4), (8, -3), (-6, 9), (0, 15)]:
        print(pt, "reachable" if reachable(*pt) else "OUT",
              "->", solve(*pt))
