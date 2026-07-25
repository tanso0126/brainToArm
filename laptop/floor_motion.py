"""Robot-relative floor paths for the fixed planar pick workspace.

Objects may change color, size, and position, but the arm and objects share one
rigid floor.  A physically reproduced reference pose and a measured local image
Jacobian therefore define two parallel curves:

* ``hover`` keeps the open gripper clear of the floor while searching.
* ``grasp`` places the open gripper at the floor pickup level.

Changing elbow moves along the working plane.  Shoulder changes simultaneously
to cancel the measured vertical component.  This is a local hardware model, not
monocular depth estimation and not a scene/background calibration.
"""

import math

import config


LEVEL_SHOULDERS = {
    "hover": config.FLOOR_HOVER_SHOULDER,
    "grasp": config.FLOOR_GRASP_SHOULDER,
}


def _validate_elbow(elbow):
    value = float(elbow)
    if not math.isfinite(value):
        raise ValueError("floor elbow must be finite")
    if not value.is_integer():
        raise ValueError("floor elbow must be an integer servo command")
    value = int(value)
    lower, upper = config.FLOOR_ELBOW_RANGE
    if not lower <= value <= upper:
        raise ValueError(
            f"floor elbow {value} outside calibrated range [{lower}, {upper}]")
    return value


def floor_shoulder(elbow, level="hover"):
    """Return compensated shoulder command for a floor-parallel elbow value."""
    elbow = _validate_elbow(elbow)
    if level not in LEVEL_SHOULDERS:
        raise ValueError(f"unknown floor level {level!r}")
    reference = LEVEL_SHOULDERS[level]
    shoulder = int(round(
        reference + config.FLOOR_SHOULDER_PER_ELBOW
        * (elbow - config.FLOOR_REFERENCE_ELBOW)))
    if not config.SERVO_MIN[config.J_SHOULDER] <= shoulder <= \
            config.SERVO_MAX[config.J_SHOULDER]:
        raise ValueError(
            f"compensated floor shoulder {shoulder} is outside servo limits")
    return shoulder


def floor_pose(elbow=None, level="hover", gripper=None):
    """Build one complete, level wrist-camera pose on the calibrated curve."""
    elbow = (config.FLOOR_REFERENCE_ELBOW if elbow is None
             else _validate_elbow(elbow))
    gripper = config.GRIP_OPEN if gripper is None else int(gripper)
    pose = list(config.HOME_POSE)
    pose[config.J_BASE] = config.FLOOR_BASE_ANGLE
    pose[config.J_SHOULDER] = floor_shoulder(elbow, level)
    pose[config.J_ELBOW] = elbow
    pose[config.J_WRIST] = config.FLOOR_WRIST_PITCH
    pose[config.J_GRIP] = gripper
    pose[config.J_ROLL] = config.FLOOR_WRIST_ROLL
    for joint, value in enumerate(pose):
        if not config.SERVO_MIN[joint] <= value <= config.SERVO_MAX[joint]:
            raise ValueError(
                f"floor pose joint {joint + 1}={value} outside configured limits")
    return pose


def floor_waypoints(start_elbow, target_elbow, level="hover", step=None,
                    gripper=None):
    """Return compensated simultaneous-joint waypoints including the endpoint."""
    start = _validate_elbow(start_elbow)
    target = _validate_elbow(target_elbow)
    step = config.FLOOR_VECTOR_STEP_DEG if step is None else int(step)
    if step <= 0:
        raise ValueError("floor vector step must be positive")
    if start == target:
        return [floor_pose(target, level, gripper)]
    direction = 1 if target > start else -1
    elbows = list(range(start + direction * step, target, direction * step))
    elbows.append(target)
    return [floor_pose(elbow, level, gripper) for elbow in elbows]


def floor_vector(delta_elbow):
    """Expose the continuous local [shoulder, elbow] ground-plane direction."""
    delta = float(delta_elbow)
    if not math.isfinite(delta):
        raise ValueError("floor vector delta must be finite")
    return (
        config.FLOOR_SHOULDER_PER_ELBOW * delta,
        delta,
    )

