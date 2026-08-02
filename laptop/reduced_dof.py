"""Reduced physical arm model: shoulder, elbow, and gripper only.

The historical six-servo stack remains untouched.  This module describes the
current damaged-arm configuration where servo 1 (base), servo 4 (wrist pitch),
and servo 6 (wrist roll) are not actuated.  The wrist is a rigid continuation
of the forearm, so its zero relative angle is represented by the old calibrated
model's servo-4 equivalent of 180 degrees.

The serial protocol still carries six fields for compatibility.  Disabled
fields are fixed placeholders and are never treated as controllable degrees of
freedom.  Only servo 2, servo 3, and servo 5 may change.
"""

from dataclasses import dataclass
import math

import numpy as np

import arm_fk
from arm_safety import PhysicalArmSafety
import config


# Protocol placeholders reported by the reduced firmware.  Keeping these equal
# to the historical HOME values lets the existing strict firmware handshake
# continue to detect stale/wrong firmware without modifying the old stack.
FIXED_BASE_COMMAND_DEG = 90
FIXED_WRIST_COMMAND_DEG = int(config.HOME_POSE[config.J_WRIST])
FIXED_ROLL_COMMAND_DEG = int(config.HOME_POSE[config.J_ROLL])

# One mechanical calibration number.  180 maps to a zero-degree wrist joint in
# arm_fk, i.e. the hand/camera bracket is rigidly straight with the forearm.
# Change only this value if the new physical bracket fixes the wrist at another
# angle; no controller equations or old six-servo files need to be edited.
FIXED_WRIST_GEOMETRY_DEG = 180.0
FIXED_ROLL_GEOMETRY_DEG = 180.0

ACTIVE_MOTION_JOINTS = (config.J_SHOULDER, config.J_ELBOW)
ACTIVE_COMMAND_JOINTS = (*ACTIVE_MOTION_JOINTS, config.J_GRIP)
DISABLED_JOINTS = (config.J_BASE, config.J_WRIST, config.J_ROLL)

# Deliberately narrower than the old raw servo limits until the rigid wrist has
# been measured over the whole workspace.  Collision checking still applies to
# every interpolated state inside this box.
SHOULDER_RANGE = (65, 145)
ELBOW_RANGE = (35, 165)
STREAM_MAX_STEP_DEG = 5.0
IMAGE_Y_MM_PER_PIXEL = 0.045
MAX_IMAGE_CORRECTION_MM = 5.0
TASK_DAMPING = 0.20
MIN_ADVANCE_MM = 3.0
MAX_ADVANCE_MM = 9.0


def command_pose(shoulder, elbow, gripper=config.GRIP_OPEN):
    """Build the only legal six-field command for the reduced firmware."""
    pose = list(config.HOME_POSE)
    pose[config.J_BASE] = FIXED_BASE_COMMAND_DEG
    pose[config.J_SHOULDER] = int(round(shoulder))
    pose[config.J_ELBOW] = int(round(elbow))
    pose[config.J_WRIST] = FIXED_WRIST_COMMAND_DEG
    pose[config.J_GRIP] = int(round(gripper))
    pose[config.J_ROLL] = FIXED_ROLL_COMMAND_DEG
    return validate_command_pose(pose)


def validate_command_pose(pose):
    """Reject any command that tries to use a disabled joint."""
    values = list(pose)
    if len(values) != config.N_JOINTS:
        raise ValueError("축소 자유도 자세에는 서보 명령값 6개가 필요합니다.")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(float(value)) for value in values):
        raise ValueError("축소 자유도 자세에는 유효한 숫자만 사용할 수 있습니다.")
    values = [int(round(value)) for value in values]
    expected = {
        config.J_BASE: FIXED_BASE_COMMAND_DEG,
        config.J_WRIST: FIXED_WRIST_COMMAND_DEG,
        config.J_ROLL: FIXED_ROLL_COMMAND_DEG,
    }
    for joint, fixed in expected.items():
        if values[joint] != fixed:
            raise ValueError(
                f"고정된 {joint + 1}번 축은 {fixed}도 명령만 허용됩니다.")
    if not SHOULDER_RANGE[0] <= values[config.J_SHOULDER] <= SHOULDER_RANGE[1]:
        raise ValueError(
            f"2번 어깨는 {SHOULDER_RANGE[0]}~{SHOULDER_RANGE[1]}도만 허용됩니다.")
    if not ELBOW_RANGE[0] <= values[config.J_ELBOW] <= ELBOW_RANGE[1]:
        raise ValueError(
            f"3번 팔꿈치는 {ELBOW_RANGE[0]}~{ELBOW_RANGE[1]}도만 허용됩니다.")
    if not config.SERVO_MIN[config.J_GRIP] <= values[config.J_GRIP] <= config.SERVO_MAX[config.J_GRIP]:
        raise ValueError("5번 집게 각도가 허용 범위를 벗어났습니다.")
    return values


def canonicalize_status(pose):
    """Use live active angles while discarding meaningless dead-axis values."""
    values = list(pose)
    if len(values) != config.N_JOINTS:
        raise ValueError("로봇 상태에는 서보값 6개가 필요합니다.")
    return command_pose(
        values[config.J_SHOULDER],
        values[config.J_ELBOW],
        values[config.J_GRIP],
    )


def geometry_pose(pose):
    """Translate protocol placeholders into the rigid physical geometry."""
    values = validate_command_pose(pose)
    values[config.J_WRIST] = float(FIXED_WRIST_GEOMETRY_DEG)
    values[config.J_ROLL] = float(FIXED_ROLL_GEOMETRY_DEG)
    return values


def geometry(pose):
    return arm_fk.geometry(geometry_pose(pose))


def fingertip_floor_clearance_mm(pose, table_z_m=0.0):
    return (float(geometry(pose).finger_tip[2]) - float(table_z_m)) * 1000.0


def sensor_axis_xz(pose):
    rotation, _position = arm_fk.sensor_pose(geometry_pose(pose))
    axis = np.asarray((rotation[0, 0], rotation[2, 0]), dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        raise RuntimeError("고정 카메라의 광축을 계산할 수 없습니다.")
    return axis / norm


def camera_state_mm(pose):
    camera = geometry(pose).camera
    return np.asarray((camera[0], camera[2]), dtype=float) * 1000.0


def camera_jacobian(pose):
    """d[camera x mm, camera z mm]/d[servo 2, servo 3]."""
    pose = validate_command_pose(pose)
    columns = []
    ranges = (SHOULDER_RANGE, ELBOW_RANGE)
    for joint, limits in zip(ACTIVE_MOTION_JOINTS, ranges):
        low = list(pose)
        high = list(pose)
        low[joint] = max(limits[0], low[joint] - 1)
        high[joint] = min(limits[1], high[joint] + 1)
        span = float(high[joint] - low[joint])
        if span <= 0:
            columns.append(np.zeros(2, dtype=float))
        else:
            columns.append((camera_state_mm(high) - camera_state_mm(low)) / span)
    return np.column_stack(columns)


class ReducedDofSafety:
    """Run the existing physical collision model on the rigid-wrist geometry."""

    def __init__(self, **kwargs):
        self._physical = PhysicalArmSafety(**kwargs)

    def pose_report(self, pose):
        return self._physical.pose_report(geometry_pose(pose))

    def pose_is_safe(self, pose):
        try:
            return self.pose_report(pose).safe
        except (TypeError, ValueError):
            return False

    def transition_report(self, start, target):
        start = validate_command_pose(start)
        target = validate_command_pose(target)
        return self._physical.transition_report(
            geometry_pose(start), geometry_pose(target))

    def transition_is_safe(self, start, target):
        try:
            return self.transition_report(start, target).safe
        except (TypeError, ValueError):
            return False


@dataclass(frozen=True)
class ReducedStep:
    pose: list[int]
    delta: tuple[float, float]
    desired_camera_mm: tuple[float, float]
    minimum_clearance_mm: float


def resolved_step(pose, vertical_error_px, distance_mm, stop_range_mm,
                  floor_stop_mm, safety=None):
    """One 2x2 damped-Jacobian step using only shoulder and elbow.

    The first component approaches along the fixed camera/sonar axis.  The
    second moves along the image-down direction to keep the target in view.
    No wrist orientation task is requested because that degree of freedom no
    longer exists.
    """
    pose = validate_command_pose(pose)
    safety = safety or ReducedDofSafety()
    distance = (
        float(distance_mm)
        if distance_mm is not None and math.isfinite(float(distance_mm))
        else float(stop_range_mm) + 150.0
    )
    advance = float(np.clip(
        0.08 * (distance - float(stop_range_mm)),
        MIN_ADVANCE_MM,
        MAX_ADVANCE_MM,
    ))
    ray = sensor_axis_xz(pose)
    image_down = np.asarray((ray[1], -ray[0]), dtype=float)
    image_correction = float(np.clip(
        float(vertical_error_px) * IMAGE_Y_MM_PER_PIXEL,
        -MAX_IMAGE_CORRECTION_MM,
        MAX_IMAGE_CORRECTION_MM,
    ))
    desired = advance * ray + image_correction * image_down
    if fingertip_floor_clearance_mm(pose) <= float(floor_stop_mm) + 8.0:
        desired[1] = max(0.0, desired[1])

    jacobian = camera_jacobian(pose)
    normal = jacobian.T @ jacobian + (TASK_DAMPING ** 2) * np.eye(2)
    delta = np.linalg.solve(normal, jacobian.T @ desired)
    largest = float(np.max(np.abs(delta)))
    if largest > STREAM_MAX_STEP_DEG:
        delta *= STREAM_MAX_STEP_DEG / largest

    target = list(pose)
    for joint, value, limits in zip(
            ACTIVE_MOTION_JOINTS, delta, (SHOULDER_RANGE, ELBOW_RANGE)):
        target[joint] = int(np.clip(
            round(target[joint] + value), limits[0], limits[1]))
    target = validate_command_pose(target)
    if target == pose:
        return None
    report = safety.transition_report(pose, target)
    if not report.safe:
        return None
    if fingertip_floor_clearance_mm(target) < float(floor_stop_mm):
        return None
    return ReducedStep(
        target,
        tuple(float(value) for value in delta),
        tuple(float(value) for value in desired),
        float(report.minimum_clearance_mm),
    )


def search_poses(gripper=config.GRIP_OPEN):
    """Deterministic safe-view sweep for the two remaining motion joints."""
    shoulders = (70, 80, 90, 100, 110, 120, 130)
    elbows = (55, 80, 105, 130, 155)
    poses = []
    for row, shoulder in enumerate(shoulders):
        ordered = elbows if row % 2 == 0 else tuple(reversed(elbows))
        for elbow in ordered:
            poses.append(command_pose(shoulder, elbow, gripper))
    return poses


def reduced_home(gripper=config.GRIP_OPEN):
    return command_pose(
        config.HOME_POSE[config.J_SHOULDER],
        config.HOME_POSE[config.J_ELBOW],
        gripper,
    )


def find_lift_pose(pose, safety=None, target_clearance_mm=50.0):
    """Find a short collision-safe lift using only joints 2 and 3."""
    pose = validate_command_pose(pose)
    safety = safety or ReducedDofSafety()
    candidates = []
    for shoulder_delta in range(-5, -61, -5):
        for elbow_delta in (0, -5, 5, -10, 10, -15, 15, -20, 20):
            shoulder = int(np.clip(
                pose[config.J_SHOULDER] + shoulder_delta, *SHOULDER_RANGE))
            elbow = int(np.clip(
                pose[config.J_ELBOW] + elbow_delta, *ELBOW_RANGE))
            candidate = command_pose(shoulder, elbow, pose[config.J_GRIP])
            clearance = fingertip_floor_clearance_mm(candidate)
            if clearance < float(target_clearance_mm):
                continue
            report = safety.transition_report(pose, candidate)
            if not report.safe:
                continue
            movement = abs(shoulder_delta) + abs(elbow_delta)
            candidates.append((movement, -clearance, candidate))
    if not candidates:
        raise RuntimeError(
            "2·3번 모터만으로 물체를 안전하게 들어 올릴 자세를 찾지 못했습니다.")
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def safe_route(start, target, safety=None):
    """Return a direct route or one safe 2-DOF waypoint."""
    start = validate_command_pose(start)
    target = validate_command_pose(target)
    safety = safety or ReducedDofSafety()
    if safety.transition_report(start, target).safe:
        return [target]
    waypoints = []
    for candidate in search_poses(gripper=start[config.J_GRIP]):
        first = safety.transition_report(start, candidate)
        second_target = list(target)
        second = safety.transition_report(candidate, second_target)
        if first.safe and second.safe:
            cost = sum(abs(candidate[j] - start[j]) for j in ACTIVE_MOTION_JOINTS)
            cost += sum(abs(second_target[j] - candidate[j]) for j in ACTIVE_MOTION_JOINTS)
            waypoints.append((cost, candidate, second_target))
    if not waypoints:
        raise RuntimeError("축소 자유도 HOME 경로를 안전하게 계산하지 못했습니다.")
    _cost, waypoint, end = min(waypoints, key=lambda item: item[0])
    return [waypoint, end]
