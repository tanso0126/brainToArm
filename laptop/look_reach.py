"""Closed-loop eye-in-hand approach using shoulder, elbow, and wrist pitch.

The camera is mounted above the gripper and looks approximately perpendicular
to the finger axis.  Consequently, putting an object in the image centre is an
*aim* measurement, not proof that the fingertips have reached it.  This module
keeps the selected object on the camera optical axis with motor 4 while motors
2/3 move the tool a few millimetres along that ray.  Every move is re-observed;
an object must stay the same instance and grow in the image before approach is
allowed to continue.

Nothing moves unless ``--run`` is supplied.  ``--grasp`` additionally permits
the final close; without it the controller stops at a conservative height.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import math
import time

import numpy as np

import arm_fk
import config
from arm_session import ArmSessionClient
from floor_grasp import WristSceneDetector
from floor_servo import (CONTACT_SAMPLE_COUNT, FloorServo, _fresh_frame)
from wrist_search import PlanarSearchSafety


ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "data" / "vision" / "look_reach_latest.jpg"
TABLE_TOUCH_CALIBRATION = (
    ROOT / "data" / "calibration" / "table_touch.json")

# Motor-4's real image response was measured in this exact mounted state:
# 140 -> 150 moved the locked target y from 251 -> 191, i.e. -6 px/servo-degree.
# The calibrated FK maps that move to -1.5 tool-degrees/servo-degree, hence
# +4 px per tool-degree.  This measured sign is more reliable than inferring it
# from a photograph of the mount.
TARGET_PX_PER_TOOL_DEG = 4.0
# At the original [110,87,140] observation pose, the camera sees a floor target
# ahead while the FK finger pitch is 104.2869 deg.  Approximate the optical ray
# as 20 deg downward there.  Closed-loop growth and image aim, not this constant,
# decide whether subsequent motion is accepted.
CAMERA_REFERENCE_TOOL_ANGLE_DEG = 104.28687674418605
CAMERA_REFERENCE_DOWN_DEG = 20.0
AIM_Y_RATIO = 0.46
AIM_X_RATIO = 0.50
MAX_JOINT_STEP_DEG = 3
ADVANCE_MM = 5.0
APPROACH_MIN_TOOL_Z_M = 0.026
GRASP_MIN_TOOL_Z_M = 0.007
MIN_PROGRESS_MM = 0.45
MAX_TRACK_JUMP_PX = 190.0
MIN_TRACK_CONFIDENCE = 0.32
MEASURED_TARGET_PX_PER_WRIST_DEG = -6.0
AIM_ONLY_THRESHOLD_PX = 35.0
VECTOR_START_POSE = [90, 110, 87, 150, 90, 170]
VECTOR_CONTACT_Z_M = 0.008
VECTOR_CALIBRATION_Z_M = 0.100
MAX_TABLE_Z_OFFSET_M = 0.015


def cumulative_tool_angle_deg(pose):
    """Finger-axis pitch in the calibrated x/z plane."""
    return (arm_fk.shoulder_joint_deg(pose[config.J_SHOULDER])
            + arm_fk.elbow_joint_deg(pose[config.J_ELBOW])
            + arm_fk.wrist_pitch_joint_deg(pose[config.J_WRIST]))


def load_table_z_m(path=TABLE_TOUCH_CALIBRATION):
    """Load the repeat-confirmed table height produced by table-touch."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = float(payload["z_table_mm"]) / 1000.0
    except (FileNotFoundError, KeyError, TypeError, ValueError,
            json.JSONDecodeError) as exc:
        raise RuntimeError(
            "valid table-touch calibration missing; run "
            "`PYTHONPATH=laptop python3 laptop/table_touch_calibrate.py --run`"
        ) from exc
    if payload.get("state") != "contact" or not math.isfinite(value):
        raise RuntimeError("table-touch calibration has no confirmed contact")
    if abs(value) > MAX_TABLE_Z_OFFSET_M:
        raise RuntimeError(
            f"table-touch FK offset {value * 1000.0:.1f} mm exceeds "
            f"the ±{MAX_TABLE_Z_OFFSET_M * 1000.0:.0f} mm validation range")
    return value


def optical_axis_xz(pose):
    """Unit camera ray in robot x/z coordinates (z positive upward)."""
    # Increasing motor 4 made the real camera look farther downward while the
    # calibrated FK finger angle decreased.  Therefore camera pitch has the
    # opposite sign from finger pitch for this physical mounting.
    down_deg = (CAMERA_REFERENCE_DOWN_DEG
                + CAMERA_REFERENCE_TOOL_ANGLE_DEG
                - cumulative_tool_angle_deg(pose))
    angle = math.radians(down_deg)
    return np.asarray((math.cos(angle), -math.sin(angle)), dtype=float)


def task_state(pose):
    # The camera is beside the motor-4 pivot.  Counting motor-4's long fingertip
    # arc as camera translation was the first trial's core modelling error: the
    # planner thought it advanced 29 mm while the target size stayed constant.
    camera = arm_fk.wrist_pitch_position(pose)
    return np.asarray((camera[0] * 1000.0, camera[2] * 1000.0,
                       cumulative_tool_angle_deg(pose)), dtype=float)


def task_delta(start, target):
    return task_state(target) - task_state(start)


def constant_x_descent_waypoints(start_pose, contact_z_m=VECTOR_CONTACT_Z_M,
                                 steps=10, advance_mm=-5.0):
    """Solve a coordinated 2/3/4 descent with a small inward advance.

    This is deliberately based on the *tool endpoint*, not camera translation.
    The camera is offset behind the fingers, so its target pixel is expected to
    move during a correct vertical approach.  The terminal pitch is 10.5 deg
    more forward than the starting pitch, matching the real mount's usable
    floor-facing range while leaving motor 4 below its 180-deg over-rotation.
    """
    from scipy.optimize import least_squares

    start = [int(round(value)) for value in start_pose]
    start_tool = arm_fk.tool_position(start)
    start_angle = cumulative_tool_angle_deg(start)
    target_angle = start_angle - 10.5
    # A same-x physical close produced contact but slipped; +10 mm returned the
    # exact empty-jaw endpoint.  Therefore the object's interior is on the -x
    # side of the first contact edge.  Move only 5 mm inward on that measured
    # side, keeping the correction smaller than the object's visible width.
    guess = np.asarray((start[config.J_SHOULDER] + 7.0,
                        start[config.J_ELBOW] + 37.0,
                        start[config.J_WRIST] + 17.0))
    lower = np.asarray((config.SERVO_MIN[config.J_SHOULDER],
                        config.SERVO_MIN[config.J_ELBOW],
                        config.SERVO_MIN[config.J_WRIST]), dtype=float)
    upper = np.asarray((config.SERVO_MAX[config.J_SHOULDER],
                        config.SERVO_MAX[config.J_ELBOW],
                        config.SERVO_MAX[config.J_WRIST]), dtype=float)

    def residual(values):
        pose = list(start)
        pose[config.J_SHOULDER], pose[config.J_ELBOW], pose[config.J_WRIST] = values
        tool = arm_fk.tool_position(pose)
        return np.asarray(((tool[0] - start_tool[0] - advance_mm / 1000.0) * 1000.0,
                           (tool[2] - contact_z_m) * 1000.0,
                           cumulative_tool_angle_deg(pose) - target_angle))

    result = least_squares(residual, np.clip(guess, lower, upper),
                           bounds=(lower, upper), xtol=1e-12, ftol=1e-12,
                           gtol=1e-12, max_nfev=500)
    if not result.success or np.linalg.norm(residual(result.x)) > 1.5:
        raise RuntimeError(f"constant-x endpoint solve failed: {result.message}")
    endpoint = list(start)
    for joint, value in zip((config.J_SHOULDER, config.J_ELBOW, config.J_WRIST),
                            result.x):
        endpoint[joint] = int(round(value))

    waypoints = []
    last_z = float(start_tool[2])
    for index in range(1, int(steps) + 1):
        fraction = index / float(steps)
        pose = [int(round(a + fraction * (b - a)))
                for a, b in zip(start, endpoint)]
        tool = arm_fk.tool_position(pose)
        expected_x = start_tool[0] + fraction * advance_mm / 1000.0
        if abs(float(tool[0] - expected_x)) > 0.006:
            raise RuntimeError("pointing descent deviated more than 6 mm")
        if float(tool[2]) > last_z + 0.001:
            raise RuntimeError("constant-x descent is not height-monotonic")
        last_z = float(tool[2])
        if not waypoints or pose != waypoints[-1]:
            waypoints.append(pose)
    endpoint_z = float(arm_fk.tool_position(waypoints[-1])[2])
    if abs(endpoint_z - float(contact_z_m)) > 0.0031:
        raise RuntimeError(
            "pointing contact endpoint differs from calibrated target height "
            f"by {(endpoint_z - float(contact_z_m)) * 1000.0:.1f} mm")
    return waypoints


def pose_at_height(template, height_m):
    """Keep 3/4 orientation fixed and solve motor 2 for a clear height."""
    pose = list(template)
    candidates = []
    for shoulder in range(config.SERVO_MIN[config.J_SHOULDER],
                          config.SERVO_MAX[config.J_SHOULDER] + 1):
        candidate = list(pose)
        candidate[config.J_SHOULDER] = shoulder
        error = abs(float(arm_fk.tool_position(candidate)[2]) - height_m)
        candidates.append((error, candidate))
    error, result = min(candidates, key=lambda item: item[0])
    if error > 0.006:
        raise RuntimeError("cannot construct clear calibration/lift pose")
    return result


def _blue_target(frame):
    """Competition target-color mode; replaceable by UI click/Hue prototype."""
    import cv2
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (95, 80, 35), (135, 255, 255))
    count, _labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        cx, cy = centers[index]
        if (area >= 300 and 0.38 * frame.shape[1] <= cx <= 0.62 * frame.shape[1]
                and y < frame.shape[0] - 45):
            candidates.append((area, (x, y, width, height), (cx, cy)))
    return max(candidates, key=lambda item: item[0]) if candidates else None


def _calibrate_empty_close(mover, high_pose):
    """Same-run empty endpoint at the final 3/4 orientation."""
    high_open = list(high_pose)
    high_open[config.J_GRIP] = config.GRIP_OPEN
    high_closed = list(high_pose)
    high_closed[config.J_GRIP] = config.GRIP_CLOSED
    mover.slow_move(high_open)
    mover.slow_move(high_closed)
    openings, red_positions = [], []
    for _ in range(CONTACT_SAMPLE_COUNT):
        frame = _fresh_frame(discard=1)
        observation, _ = mover.marker_detector.detect(frame)
        if observation.gripper is not None:
            openings.append(float(observation.gripper.opening_px))
        red_x = mover._red_marker_x(frame)
        if red_x is not None:
            red_positions.append(red_x)
    mover.slow_move(high_open)
    if not openings and not red_positions:
        raise RuntimeError("empty-close calibration could not see either marker")
    if openings:
        mover.empty_closed_opening_px = float(np.median(openings))
        mover.empty_closed_opening_mad_px = float(np.median(np.abs(
            np.asarray(openings) - mover.empty_closed_opening_px)))
    if red_positions:
        mover.empty_closed_red_x_px = float(np.median(red_positions))
        mover.empty_closed_red_x_mad_px = float(np.median(np.abs(
            np.asarray(red_positions) - mover.empty_closed_red_x_px)))
    print(f"[look-reach] empty close opening={mover.empty_closed_opening_px} "
          f"red-x={mover.empty_closed_red_x_px}")


def run_constant_x_grasp(client, execute=False, advance_mm=-5.0,
                         table_z_m=0.0):
    """Aim at the floor object, descend at fixed tool x, close, lift, verify."""
    mover = FloorServo(client, calib=None)
    safety = PlanarSearchSafety()
    start = list(VECTOR_START_POSE)
    current = list(client.request({"command": "status"})["pose"])
    if not safety.transition_is_safe(current, start):
        raise RuntimeError("transition to vector start failed collision model")
    if execute:
        mover.slow_move(start, final_settle=1.0)
    frame = _fresh_frame(discard=2)
    target = _blue_target(frame)
    observation, _ = mover.marker_detector.detect(frame)
    if target is None or observation.gripper is None:
        raise RuntimeError("target or open finger markers are not visible at vector start")
    _area, bbox, center = target
    horizontal_error = float(center[0] - observation.gripper.center[0])
    if abs(horizontal_error) > 0.42 * observation.gripper.opening_px:
        raise RuntimeError(f"target is outside open jaws in x ({horizontal_error:.0f}px)")
    waypoints = constant_x_descent_waypoints(
        start, contact_z_m=float(table_z_m) + VECTOR_CONTACT_Z_M,
        advance_mm=advance_mm)
    for previous, pose in zip([start] + waypoints[:-1], waypoints):
        if not safety.transition_is_safe(previous, pose):
            raise RuntimeError("constant-x waypoint failed collision model")
    endpoint = waypoints[-1]
    high = pose_at_height(
        endpoint, float(table_z_m) + VECTOR_CALIBRATION_Z_M)
    print(f"[look-reach] VECTOR target={tuple(round(v,1) for v in center)} "
          f"bbox={bbox} du={horizontal_error:.1f}px inset={advance_mm:.1f}mm")
    print("[look-reach] VECTOR path "
          + " -> ".join(str(p[1:4]) for p in waypoints))
    for pose in [start] + waypoints:
        tool = arm_fk.tool_position(pose)
        print(f"[look-reach] FK pose234={pose[1:4]} "
              f"tool=({tool[0]*1000:.1f},{tool[2]*1000:.1f})mm")
    if not execute:
        return {"state": "planned", "endpoint": endpoint, "high": high}

    _calibrate_empty_close(mover, high)
    mover.slow_move(start)
    for pose in waypoints:
        mover.slow_move(pose, final_settle=0.35)
    closed_pose = list(endpoint)
    closed_pose[config.J_GRIP] = config.GRIP_CLOSED
    mover.slow_move(closed_pose)
    closed = mover._contact_assessment(allow_single_red=True)
    print(f"[look-reach] close contact={closed.state}: {closed.reason}")
    high_closed = list(high)
    high_closed[config.J_GRIP] = config.GRIP_CLOSED
    mover.slow_move(high_closed)
    lifted = mover._contact_assessment(allow_single_red=True)
    print(f"[look-reach] lifted contact={lifted.state}: {lifted.reason}")
    # At 8 mm the tapes can legitimately leave the frame.  In that case the
    # 100 mm pose is itself the verification probe: floor contact is impossible
    # and its same-orientation empty endpoint was measured seconds earlier.
    # Require a second independent high sample before accepting this path.
    if closed.state == "UNKNOWN" and lifted.contact:
        lifted_confirm = mover._contact_assessment(allow_single_red=True)
        print(f"[look-reach] lifted confirm={lifted_confirm.state}: "
              f"{lifted_confirm.reason}")
        retained = lifted_confirm.contact
        if retained:
            lifted = lifted_confirm
    else:
        retained = mover.contact_retained(closed, lifted)
    if not retained:
        high_open = list(high)
        high_open[config.J_GRIP] = config.GRIP_OPEN
        mover.slow_move(high_open)
        raise RuntimeError("object was not retained through the 100 mm lift")
    print("[look-reach] GRASP CONFIRMED at 100 mm lift")
    return {"state": "retained", "pose": high_closed,
            "close_residual_px": closed.residual_px,
            "lift_residual_px": lifted.residual_px}


def _angle_error_deg(target, actual):
    return (float(target) - float(actual) + 180.0) % 360.0 - 180.0


def plan_resolved_step(pose, vertical_error_px, frame_height,
                       advance_mm=ADVANCE_MM, min_tool_z_m=APPROACH_MIN_TOOL_Z_M,
                       max_joint_step=MAX_JOINT_STEP_DEG):
    """Choose one integer 2/3/4 move by bounded resolved-rate search.

    The task vector is ``[forward x, vertical z, finger pitch]``.  Desired x/z
    travel follows the camera ray.  Vertical pixel error becomes a small pitch
    correction; because the camera is rigid, maintaining that pitch while
    shoulder/elbow move requires motor 4 compensation in the same solve.
    """
    pose = [int(round(value)) for value in pose]
    ray = optical_axis_xz(pose)
    desired_translation = float(advance_mm) * ray
    # The object is known to lie on the same rigid floor as the arm.  Raising
    # the wrist to preserve view must therefore be countered by real 2/3-joint
    # descent.  Keep at least this much downward motion until the tool reaches
    # the guarded floor band; this is the missing term observed physically.
    current_tool_z = float(arm_fk.tool_position(pose)[2])
    remaining_height_mm = max(0.0, (current_tool_z - min_tool_z_m) * 1000.0)
    floor_descent_mm = min(3.0, max(0.5, 0.16 * remaining_height_mm))
    desired_translation[1] = min(desired_translation[1], -floor_descent_mm)
    motion_axis = desired_translation / max(1e-9,
                                            np.linalg.norm(desired_translation))
    del frame_height  # retained in the API so callers state the image geometry
    # Error is actual-target minus desired-target.  A target above the cross has
    # negative error and needs a *larger* FK tool angle (motor 4 lower), exactly
    # matching the measured -6 px/servo-degree response.
    aim_delta = float(np.clip(
        -float(vertical_error_px) / TARGET_PX_PER_TOOL_DEG, -4.5, 4.5))
    desired_angle = cumulative_tool_angle_deg(pose) + aim_delta

    best = None
    joints = (config.J_SHOULDER, config.J_ELBOW, config.J_WRIST)
    for d2 in range(-max_joint_step, max_joint_step + 1):
        for d3 in range(-max_joint_step, max_joint_step + 1):
            for d4 in range(-max_joint_step, max_joint_step + 1):
                if d2 == d3 == d4 == 0:
                    continue
                candidate = list(pose)
                for joint, delta in zip(joints, (d2, d3, d4)):
                    candidate[joint] += delta
                if any(not config.SERVO_MIN[j] <= candidate[j] <= config.SERVO_MAX[j]
                       for j in joints):
                    continue
                tool = arm_fk.tool_position(candidate)
                if tool[2] < float(min_tool_z_m):
                    continue
                delta = task_delta(pose, candidate)
                translation = delta[:2]
                progress = float(np.dot(translation, motion_axis))
                cross = float(translation[0] * motion_axis[1]
                              - translation[1] * motion_axis[0])
                angle_error = _angle_error_deg(desired_angle,
                                               cumulative_tool_angle_deg(candidate))
                translation_error = translation - desired_translation
                # Angle dominates while aim is poor; after it settles, the
                # translation term naturally chooses coordinated 2/3 motion.
                score = (float(np.dot(translation_error, translation_error)) / 9.0
                         + (angle_error / 1.3) ** 2
                         + (cross / 3.0) ** 2
                         + 0.025 * (d2 * d2 + d3 * d3 + d4 * d4))
                if progress < -0.25:
                    score += 30.0 + progress * progress
                key = (score, -progress, abs(d4), abs(d2) + abs(d3))
                if best is None or key < best[0]:
                    best = (key, candidate, progress, cross, angle_error)
    if best is None:
        return None
    _key, candidate, progress, cross, angle_error = best
    return {
        "pose": candidate,
        "progress_mm": progress,
        "cross_mm": cross,
        "angle_error_deg": angle_error,
        "aim_delta_deg": aim_delta,
        "tool_z_m": float(arm_fk.tool_position(candidate)[2]),
    }


def plan_aim_step(pose, vertical_error_px, min_tool_z_m=APPROACH_MIN_TOOL_Z_M):
    """Rotate only motor 4 using the measured mounted-camera pixel response."""
    pose = [int(round(value)) for value in pose]
    desired_pixel_delta = -float(vertical_error_px)
    wrist_delta = int(round(np.clip(
        desired_pixel_delta / MEASURED_TARGET_PX_PER_WRIST_DEG,
        -MAX_JOINT_STEP_DEG, MAX_JOINT_STEP_DEG)))
    if wrist_delta == 0:
        return None
    candidate = list(pose)
    candidate[config.J_WRIST] = int(np.clip(
        candidate[config.J_WRIST] + wrist_delta,
        config.SERVO_MIN[config.J_WRIST], config.SERVO_MAX[config.J_WRIST]))
    if candidate == pose or arm_fk.tool_position(candidate)[2] < min_tool_z_m:
        return None
    delta = task_delta(pose, candidate)
    return {
        "pose": candidate,
        "progress_mm": 0.0,
        "cross_mm": 0.0,
        "angle_error_deg": 0.0,
        "aim_delta_deg": float(delta[2]),
        "tool_z_m": float(arm_fk.tool_position(candidate)[2]),
        "aim_only": True,
    }


@dataclass
class TargetLock:
    center: np.ndarray
    bbox: tuple
    area: float
    confidence: float
    initial_area: float

    @classmethod
    def from_candidate(cls, candidate):
        area = float(candidate.area)
        return cls(np.asarray(candidate.center, dtype=float),
                   tuple(candidate.bbox), area,
                   float(getattr(candidate, "confidence", 1.0)), area)

    def update(self, candidate):
        self.center = np.asarray(candidate.center, dtype=float)
        self.bbox = tuple(candidate.bbox)
        self.area = float(candidate.area)
        self.confidence = float(getattr(candidate, "confidence", 1.0))


def select_initial_target(scene):
    """Select a central, non-gripper object without a color/location template."""
    height, width = scene.frame_shape[:2]
    eligible = []
    for candidate in scene.ranked:
        confidence = float(getattr(candidate, "confidence", 1.0))
        cx, cy = candidate.center
        x, y, box_width, box_height = candidate.bbox
        ratio = float(candidate.area) / float(width * height)
        if (confidence < MIN_TRACK_CONFIDENCE or not 0.001 <= ratio <= 0.12
                or not 0.20 * width <= cx <= 0.80 * width
                or cy >= 0.82 * height):
            continue
        centre_cost = abs(cx / width - AIM_X_RATIO)
        aim_cost = 0.22 * abs(cy / height - AIM_Y_RATIO)
        size_bonus = 0.025 * math.log1p(max(1.0, candidate.area))
        eligible.append((centre_cost + aim_cost - size_bonus, candidate))
    return min(eligible, key=lambda item: item[0])[1] if eligible else None


def match_locked_target(scene, lock):
    """Track the same appearance/position; never jump to a bottom cable/jaw."""
    eligible = []
    old_w, old_h = lock.bbox[2], lock.bbox[3]
    old_aspect = max(old_w, old_h) / max(1.0, min(old_w, old_h))
    for candidate in scene.ranked:
        confidence = float(getattr(candidate, "confidence", 1.0))
        if confidence < MIN_TRACK_CONFIDENCE:
            continue
        distance = float(np.linalg.norm(np.asarray(candidate.center) - lock.center))
        if distance > MAX_TRACK_JUMP_PX:
            continue
        w, h = candidate.bbox[2], candidate.bbox[3]
        aspect = max(w, h) / max(1.0, min(w, h))
        area_ratio = float(candidate.area) / max(1.0, lock.area)
        if not 0.22 <= area_ratio <= 4.5:
            continue
        appearance = abs(math.log(max(1e-6, aspect / old_aspect)))
        score = distance / MAX_TRACK_JUMP_PX + 0.30 * appearance \
            + 0.08 * abs(math.log(area_ratio)) - 0.08 * confidence
        eligible.append((score, candidate))
    return min(eligible, key=lambda item: item[0])[1] if eligible else None


def acquire_initial_target(detector, samples=3):
    """Use several fresh segmentations so a transient object part cannot lock."""
    choices = []
    latest_frame = None
    latest_scene = None
    for _ in range(samples):
        latest_frame = _fresh_frame(discard=1)
        latest_scene, _ = detector.scene(latest_frame)
        selected = select_initial_target(latest_scene)
        if selected is not None:
            choices.append(selected)
    if not choices:
        return latest_frame, latest_scene, None
    # Across frames FastSAM may alternate between a complete object and a nested
    # cap/label.  Same horizontal column plus bbox overlap identifies the group;
    # its largest mask is the stable physical-object lock.
    seed = min(choices, key=lambda item: abs(item.center[0]
                                             - AIM_X_RATIO * latest_frame.shape[1]))
    related = []
    sx, sy, sw, sh = seed.bbox
    for candidate in choices:
        x, y, width, height = candidate.bbox
        horizontal_overlap = max(0, min(sx + sw, x + width) - max(sx, x))
        if (abs(candidate.center[0] - seed.center[0]) <= 90
                and horizontal_overlap > 0.35 * min(sw, width)):
            related.append(candidate)
    return latest_frame, latest_scene, max(
        related or choices, key=lambda item: item.area)


def _draw_preview(frame, lock, step, pose, aim_y, plan=None):
    import cv2
    image = frame.copy()
    x, y, width, height = lock.bbox
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 255), 3)
    aim = (int(round(AIM_X_RATIO * image.shape[1])),
           int(round(aim_y)))
    cv2.drawMarker(image, aim, (255, 255, 255), cv2.MARKER_CROSS, 30, 2)
    cv2.line(image, tuple(np.round(lock.center).astype(int)), aim,
             (0, 255, 255), 2)
    text = f"step={step}  2/3/4={pose[1]}/{pose[2]}/{pose[3]}  area={lock.area:.0f}"
    if plan is not None:
        text += f"  advance={plan['progress_mm']:.1f}mm z={plan['tool_z_m']*1000:.0f}mm"
    cv2.putText(image, text, (22, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (30, 30, 30), 4, cv2.LINE_AA)
    cv2.putText(image, text, (22, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.72, (255, 255, 255), 2, cv2.LINE_AA)
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(PREVIEW), image)


def run_controller(client, max_steps=24, execute=False, allow_grasp=False):
    detector = WristSceneDetector()
    safety = PlanarSearchSafety()
    mover = FloorServo(client, calib=None)
    frame, scene, selected = acquire_initial_target(detector)
    if selected is None:
        raise RuntimeError("no central portable target found")
    lock = TargetLock.from_candidate(selected)
    initial_area = lock.area
    previous_area = lock.area
    # This is the hand-eye offset in image coordinates for the present mount.
    # Holding the feature here keeps the gripper/camera bearing on the target;
    # forcing generic image centre would discard the real mount offset.
    aim_y = float(lock.center[1])
    non_growth = 0
    print(f"[look-reach] LOCK center={np.round(lock.center)} bbox={lock.bbox} "
          f"conf={lock.confidence:.3f} area={lock.area:.0f}")

    for step in range(max_steps):
        pose = list(client.request({"command": "status"})["pose"])
        frame = _fresh_frame(discard=2)
        scene, _ = detector.scene(frame)
        candidate = match_locked_target(scene, lock)
        if candidate is None:
            raise RuntimeError("locked target lost; refusing blind motion")
        lock.update(candidate)
        vertical_error = float(lock.center[1] - aim_y)
        minimum_z = (GRASP_MIN_TOOL_Z_M if allow_grasp
                     else APPROACH_MIN_TOOL_Z_M)
        plan = None
        if abs(vertical_error) > AIM_ONLY_THRESHOLD_PX:
            plan = plan_aim_step(pose, vertical_error, minimum_z)
        if plan is None:
            plan = plan_resolved_step(
                pose, vertical_error, frame.shape[0],
                min_tool_z_m=minimum_z)
        if plan is None:
            raise RuntimeError("no bounded 2/3/4 step satisfies height/servo limits")
        _draw_preview(frame, lock, step, pose, aim_y, plan)
        x, y, width, height = lock.bbox
        bbox_bottom = y + height
        marker_y = (scene.gripper.center[1] if scene.gripper is not None
                    else config.WRIST_GRIPPER_OPEN_PROFILE["center"][1]
                    * frame.shape[0])
        gap = float(marker_y - bbox_bottom)
        growth = lock.area / max(1.0, initial_area)
        print(f"[look-reach] step={step:02d} target=({lock.center[0]:.0f},"
              f"{lock.center[1]:.0f}) gap={gap:.0f}px growth={growth:.2f} "
              f"pose234={pose[1:4]} -> {plan['pose'][1:4]} "
              f"progress={plan['progress_mm']:.2f}mm "
              f"cross={plan['cross_mm']:.2f}mm "
              f"aim-residual={plan['angle_error_deg']:.2f}deg "
              f"z={plan['tool_z_m']*1000:.1f}mm")

        # The coloured finger markers are a direct camera-to-fingertip datum.
        # Close only when the same growing object reaches their near row and is
        # horizontally between the open jaws.  Image centring alone is never a
        # grasp trigger.
        jaw_ready = (scene.gripper is not None
                     and abs(lock.center[0] - scene.gripper.center[0])
                     <= 0.42 * scene.gripper.opening_px
                     and gap <= 95.0
                     # Perspective growth is useful at long range, but the
                     # rigid common floor gives an independent near-field fact:
                     # at <=18 mm tool height the target cannot still be a
                     # far-away object merely projected between the fingers.
                     and (growth >= 1.20
                          or arm_fk.tool_position(pose)[2] <= 0.018))
        if jaw_ready:
            print(f"[look-reach] fingertip geometry reached: gap={gap:.0f}px, "
                  f"growth={growth:.2f}")
            if allow_grasp and execute:
                closed = list(pose)
                closed[config.J_GRIP] = config.GRIP_CLOSED
                mover.slow_move(closed)
                print("[look-reach] GRIPPER CLOSED; retention must be verified separately")
            return {"state": "jaw-ready", "pose": pose, "growth": growth,
                    "gap_px": gap, "preview": str(PREVIEW)}

        if not plan.get("aim_only") and lock.area < previous_area * 0.93:
            non_growth += 1
        elif not plan.get("aim_only"):
            non_growth = max(0, non_growth - 1)
        if non_growth >= 3:
            raise RuntimeError("target repeatedly shrank; approach direction is wrong")
        previous_area = lock.area
        if (not plan.get("aim_only")
                and plan["progress_mm"] < MIN_PROGRESS_MM
                and abs(vertical_error) < 24):
            raise RuntimeError("joint limits prevent further forward progress")
        if not safety.transition_is_safe(pose, plan["pose"]):
            raise RuntimeError("collision model rejected planned transition")
        if not execute:
            print("[look-reach] DRY RUN: first safe step planned; no arm motion")
            return {"state": "planned", "pose": plan["pose"],
                    "preview": str(PREVIEW)}
        mover.slow_move(plan["pose"])
        time.sleep(0.15)

    return {"state": "step-limit", "pose": client.request(
        {"command": "status"})["pose"], "growth": lock.area / initial_area,
        "preview": str(PREVIEW)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="execute safe arm moves")
    parser.add_argument("--grasp", action="store_true",
                        help="allow final low approach and gripper close")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--vector-grasp", action="store_true",
                        help="fixed-tool-x coordinated 2/3/4 grasp and lift")
    parser.add_argument("--vector-inset-mm", type=float, default=-5.0,
                        help="signed endpoint-x correction for vector grasp")
    args = parser.parse_args()
    if args.grasp and not args.run:
        parser.error("--grasp requires --run")
    client = ArmSessionClient()
    table_z_m = (load_table_z_m()
                 if args.vector_grasp and args.run else 0.0)
    result = (run_constant_x_grasp(
                  client, execute=args.run, advance_mm=args.vector_inset_mm,
                  table_z_m=table_z_m)
              if args.vector_grasp else
              run_controller(client, args.max_steps,
                             execute=args.run, allow_grasp=args.grasp))
    print(f"[look-reach] RESULT {result}")


if __name__ == "__main__":
    main()
