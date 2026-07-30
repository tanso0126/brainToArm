"""Closed-loop eye-in-hand approach using shoulder, elbow, and wrist pitch.

The camera is mounted above the gripper and looks approximately perpendicular
to the finger axis.  Consequently, putting an object in the image centre is an
*aim* measurement, not proof that the fingertips have reached it.  This module
keeps the selected object on the camera optical axis with motor 4 while motors
2/3 move the tool a few millimetres along that ray.  Every move is re-observed;
an object must stay the same instance and grow in the image before approach is
allowed to continue.

FastSAM supplies a ranked portable-object list. ``CandidateSelector`` chooses
the nearest reachable item, while ``--reject-count N`` applies N explicit
image-position vetoes and locks the next item. The same selector object exposes
``reject_current(scene, pose)`` for a later ErrP callback.

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
from floor_grasp import CandidateSelector, WristSceneDetector
from floor_servo import (
    BASE_CENTER_RANGE_DEG, CONTACT_SAMPLE_COUNT, FloorServo, _fresh_frame)
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
# FastSAM confidence sags for a correctly detected object once it is close
# to the jaws (partial occlusion, low texture). Geometry - jaw corridor,
# portable-area envelope, solved reach band - plus the fail-closed contact
# and retention proofs are the real discriminators, so keep this low.
MIN_TRACK_CONFIDENCE = 0.22
MEASURED_TARGET_PX_PER_WRIST_DEG = -6.0
AIM_ONLY_THRESHOLD_PX = 35.0
VECTOR_START_POSE = [90, 110, 87, 150, 90, 170]
VECTOR_CONTACT_Z_M = 0.008
VECTOR_CALIBRATION_Z_M = 0.100
MAX_TABLE_Z_OFFSET_M = 0.015
VECTOR_LATERAL_OPENING_FRACTION = 0.42
# Image rows (start view) whose tabletop depth matches the fixed vector reach.
# Measured: the physically closed-and-held eraser had bbox bottom ~614 px; the
# known-good grasp corridor spans roughly this band. A base row above it means
# the object stands beyond the fixed reach (the sagging USB cable crosses the
# corridor there and was once grabbed instead - hence the hard refusal).
VECTOR_DEPTH_CORRIDOR_ROWS = (420.0, 700.0)
# Everything graspable rests on the near table and therefore projects
# into the lower part of this near-horizontal camera view.
TABLE_HORIZON_ROW_RATIO = 0.54


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


REACH_SOLVE_MIN_M = 0.285
REACH_SOLVE_MAX_M = 0.375
REACH_PROBE_MM = 8.0
# The open jaws span ~285 px for ~45 mm, so 25 px is about 4 mm - well
# inside the jaw span for a pencil-sized object. A tighter tolerance only
# forced needless probes that pushed a nearly-aligned object out of view.
REACH_TOL_PX = 25.0
REACH_MAX_ITERS = 10
REACH_MAX_STEP_MM = 18.0
# Camera viewing slices across the verified reach band, far to near.
SEARCH_REACH_SEQUENCE_M = (0.375, 0.350, 0.325, 0.300, 0.288)


def pose_at_reach(template, x_m, z_m):
    """Solve joints 2/3/4 for a tool at (x_m, z_m), preserving tool pitch.

    This is the adaptive counterpart of :func:`pose_at_height`: the arm must
    travel to wherever the object actually is, so forward reach is an argument
    rather than a constant baked into the start pose.
    """
    from scipy.optimize import least_squares

    start = [int(round(value)) for value in template]
    target_angle = cumulative_tool_angle_deg(start)
    lower = np.asarray((config.SERVO_MIN[config.J_SHOULDER],
                        config.SERVO_MIN[config.J_ELBOW],
                        config.SERVO_MIN[config.J_WRIST]), dtype=float)
    upper = np.asarray((config.SERVO_MAX[config.J_SHOULDER],
                        config.SERVO_MAX[config.J_ELBOW],
                        config.SERVO_MAX[config.J_WRIST]), dtype=float)
    guess = np.asarray((float(start[config.J_SHOULDER]),
                        float(start[config.J_ELBOW]),
                        float(start[config.J_WRIST])))

    def residual(values):
        pose = list(start)
        pose[config.J_SHOULDER], pose[config.J_ELBOW], pose[config.J_WRIST] = values
        tool = arm_fk.tool_position(pose)
        return np.asarray(((tool[0] - x_m) * 1000.0,
                           (tool[2] - z_m) * 1000.0,
                           cumulative_tool_angle_deg(pose) - target_angle))

    result = least_squares(residual, np.clip(guess, lower, upper),
                           bounds=(lower, upper), xtol=1e-12, ftol=1e-12,
                           gtol=1e-12, max_nfev=500)
    if not result.success or np.linalg.norm(residual(result.x)) > 2.0:
        raise RuntimeError(
            f"reach solve failed for x={x_m*1000:.0f}mm z={z_m*1000:.0f}mm")
    pose = list(start)
    for joint, value in zip((config.J_SHOULDER, config.J_ELBOW, config.J_WRIST),
                            result.x):
        pose[joint] = int(round(value))
    return pose


def _depth_error_px(scene, target):
    """Object base row minus fingertip marker row at the current view.

    Both lie on the same table plane, so equal rows means the fingertips have
    reached the object's depth. Negative means the object is still farther out.
    """
    gripper = getattr(scene, "gripper", None)
    if gripper is None or target is None:
        return None
    base_row = float(target.bbox[1] + target.bbox[3])
    return base_row - float(gripper.center[1])


def solve_object_reach(mover, detector, target_selector, start_pose,
                       frame_source=_fresh_frame, logger=print,
                       rigid_points=()):
    """Extend/retract forward reach until the fingertips reach the object.

    Measures d(depth error)/d(reach) on the real hardware with one bounded
    probe, then takes clamped Newton steps, re-observing every time. Returns
    ``(pose, tool_x_m)`` at convergence, or ``(None, None)`` when the object
    cannot be reached inside the verified band.
    """
    def reacquire(attempts=3):
        """Re-lock the selected object after a commanded reach change.

        A legitimate reach step moves the object a long way in the image, so a
        single missed segmentation must not abort the approach; only repeated
        failure does.
        """
        for _ in range(attempts):
            frame = frame_source(discard=1)
            scene, _ = detector.scene(frame)
            # Same discrimination as at selection time: own hardware and flat
            # sheet folds must not be able to steal the lock mid-approach.
            scene.ranked = [item for item in scene.ranked
                            if is_graspable_figure(frame, item)
                            and not _is_rigid(item, rigid_points)]
            matched = target_selector.match(scene)
            if matched is not None:
                return scene, matched
        return None, None

    pose = list(start_pose)
    tool = arm_fk.tool_position(pose)
    x_m, z_m = float(tool[0]), float(tool[2])
    gain = None
    previous = None
    reprobes = 0
    for iteration in range(REACH_MAX_ITERS):
        scene, target = reacquire()
        if target is None:
            logger("[look-reach] reach solve: target lost")
            return None, None
        error = _depth_error_px(scene, target)
        if error is None:
            logger("[look-reach] reach solve: finger markers not visible")
            return None, None
        logger(f"[look-reach] reach x={x_m*1000:.0f}mm depth_err={error:+.0f}px")
        if abs(error) <= REACH_TOL_PX:
            return pose, x_m
        if gain is None:
            # One bounded probe measures px per mm of reach in this geometry.
            probe_x = float(np.clip(x_m + REACH_PROBE_MM / 1000.0,
                                    REACH_SOLVE_MIN_M, REACH_SOLVE_MAX_M))
            if abs(probe_x - x_m) < 1e-6:
                probe_x = float(np.clip(x_m - REACH_PROBE_MM / 1000.0,
                                        REACH_SOLVE_MIN_M, REACH_SOLVE_MAX_M))
            probe_pose = pose_at_reach(start_pose, probe_x, z_m)
            mover.slow_move(probe_pose, final_settle=0.6)
            probe_scene, probe_target = reacquire()
            probe_error = _depth_error_px(probe_scene, probe_target)
            if probe_error is None:
                logger("[look-reach] reach solve: lost target during probe")
                return None, None
            delta_mm = (probe_x - x_m) * 1000.0
            gain = (probe_error - error) / delta_mm
            logger(f"[look-reach] measured reach gain {gain:+.2f}px/mm")
            if not np.isfinite(gain) or abs(gain) < 0.15:
                logger("[look-reach] reach response too weak; refusing descent")
                return None, None
            pose, x_m, error = probe_pose, probe_x, probe_error
            if abs(error) <= REACH_TOL_PX:
                return pose, x_m
        step_mm = float(np.clip(-error / gain,
                                -REACH_MAX_STEP_MM, REACH_MAX_STEP_MM))
        new_x = float(np.clip(x_m + step_mm / 1000.0,
                              REACH_SOLVE_MIN_M, REACH_SOLVE_MAX_M))
        if abs(new_x - x_m) < 5e-4:
            logger(f"[look-reach] reach bound reached at x={x_m*1000:.0f}mm "
                   f"with depth_err={error:+.0f}px; object is outside the "
                   "verified reach band")
            return None, None
        if previous is not None and abs(error) > abs(previous) + 6.0:
            # Close to the jaws the row response is nonlinear, so a worsening
            # step means the single measured gain is stale rather than that the
            # object is unreachable. Re-measure it; only repeated failure aborts.
            reprobes += 1
            if reprobes > 2:
                logger("[look-reach] depth error keeps diverging; no descent")
                return None, None
            logger("[look-reach] depth error grew; re-measuring reach gain")
            gain = None
            previous = None
            continue
        previous = error
        pose = pose_at_reach(start_pose, new_x, z_m)
        mover.slow_move(pose, final_settle=0.6)
        x_m = new_x
    logger("[look-reach] reach did not converge within iteration budget")
    return None, None


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
                         table_z_m=0.0, target_selector=None,
                         reject_count=0, detector=None,
                         frame_source=_fresh_frame):
    """Aim at the floor object, descend at fixed tool x, close, lift, verify."""
    detector = detector or WristSceneDetector()
    target_selector = target_selector or LookReachTargetSelector()
    mover = FloorServo(client, calib=None)
    safety = PlanarSearchSafety()
    start = list(VECTOR_START_POSE)
    current = list(client.request({"command": "status"})["pose"])

    # Selection/reject must happen AT the vector start pose: choosing from a
    # different observation pose and then moving makes every candidate pixel
    # jump beyond the identity bound, which previously forced a safe stop
    # ("selected target was not reacquired at vector start"). Move first (the
    # transition is collision-checked), then look, then choose. Dry runs stay
    # motionless and plan from the current view instead.
    if execute:
        if not safety.transition_is_safe(current, start):
            raise RuntimeError(
                "transition to vector start failed collision model")
        mover.slow_move(start, final_settle=1.0)
        current = list(client.request({"command": "status"})["pose"])

    rigid_points = []
    if execute:
        rigid_points = rigid_with_camera_points(
            mover, detector, client, frame_source)
    initial_frame, initial_scene, target = acquire_initial_target(
        detector, target_selector=target_selector,
        reject_count=reject_count, pose=current,
        frame_source=frame_source, rigid_points=rigid_points)
    # Search. One fixed viewing pose only sees one slice of the table, so an
    # object nearer or farther than that slice is simply invisible - which is
    # what "no target" meant every time so far. Sweep the reach band (which
    # sweeps the camera along the table) and stop at the first pose that shows a
    # reachable, background-distinct object. Bounded, collision-checked, gentle.
    if target is None and execute:
        start_tool = arm_fk.tool_position(start)
        for scan_x in SEARCH_REACH_SEQUENCE_M:
            try:
                scan_pose = pose_at_reach(start, scan_x, float(start_tool[2]))
            except RuntimeError:
                continue
            if not safety.transition_is_safe(
                    client.request({"command": "status"})["pose"], scan_pose):
                continue
            print(f"[look-reach] SEARCH at reach {scan_x*1000:.0f}mm")
            mover.slow_move(scan_pose, final_settle=0.7)
            current = list(client.request({"command": "status"})["pose"])
            initial_frame, initial_scene, target = acquire_initial_target(
                detector, target_selector=target_selector,
                reject_count=reject_count, pose=current,
                frame_source=frame_source, rigid_points=rigid_points)
            if target is not None:
                start = scan_pose
                print(f"[look-reach] SEARCH found target at reach "
                      f"{scan_x*1000:.0f}mm")
                break
    if target is None and initial_scene is not None:
        # A candidate that failed ONLY the lateral jaw window may still be
        # reachable through the verified bounded base-yaw centring already
        # exercised by FloorServo (measured Jacobian, ±BASE_CENTER_RANGE_DEG,
        # <=3 deg steps, re-observed). Rejected (vetoed) locations are never
        # centred on; reachability skips are not vetoes.
        # Motor 1 (base yaw) is intentionally unused on this build and does not
        # physically respond, so lateral centring cannot help: it only desyncs
        # the commanded base from reality. Objects must be placed on the arm's
        # sagittal line, which the placement guide shows live.
        fixable = None
        if fixable is not None and not execute:
            print("[look-reach] DRY RUN: candidate at "
                  f"{tuple(round(v) for v in fixable.center)} needs base "
                  "centring; no motion in dry run")
            return {"state": "needs-base-centering", "moved": False}
        if fixable is not None and execute:
            pre_center = np.asarray(fixable.center, dtype=float)
            print(f"[look-reach] BASE CENTERING on candidate at "
                  f"{tuple(round(v) for v in pre_center)}")
            centered_obj, _mid = mover._center_base_lateral(pre_center)
            if centered_obj is None:
                print("[look-reach] base centring failed closed; no motion")
                return {"state": "no-target", "moved": True}
            # Persistent vetoes are image positions; the rotation shifted the
            # whole scene, so translate them by the centred candidate's
            # observed displacement before re-selecting.
            displacement = np.asarray(centered_obj, dtype=float) - pre_center
            vetoes = target_selector.selector.rejected_points
            target_selector.selector.rejected_points = [
                (float(x + displacement[0]), float(y + displacement[1]))
                for x, y in vetoes]
            current = list(client.request({"command": "status"})["pose"])
            initial_frame, initial_scene, target = acquire_initial_target(
                detector, target_selector=target_selector,
                reject_count=0, pose=current, frame_source=frame_source)
    if target is None:
        print("[look-reach] no reachable non-vetoed target; no motion")
        return {"state": "no-target", "moved": bool(execute)}
    start[config.J_BASE] = int(mover.base_angle)

    if execute:
        # Selection already happened at the vector start view; one fresh
        # confirmation frame guards against a transient segmentation.
        frame = frame_source(discard=1)
        scene, observation = detector.scene(frame)
        target = target_selector.match(scene)
        if target is None:
            raise RuntimeError(
                "selected target was not reacquired at vector start")
    else:
        if not safety.transition_is_safe(current, start):
            raise RuntimeError(
                "transition to vector start failed collision model")
        scene, observation = initial_scene, None

    gripper = scene.gripper
    if gripper is None:
        raise RuntimeError("open finger markers are not visible at vector start")
    bbox, center = target.bbox, target.center
    horizontal_error = float(center[0] - gripper.center[0])
    if abs(horizontal_error) > \
            VECTOR_LATERAL_OPENING_FRACTION * gripper.opening_px:
        raise RuntimeError(f"target is outside open jaws in x ({horizontal_error:.0f}px)")
    # Adaptive forward reach: the arm travels to wherever the object actually
    # is. A fixed reach only ever grasped whatever happened to sit at that one
    # spot. The fingertips and the object share the table plane, so equal image
    # rows mean equal depth; the loop below measures px-per-mm on the hardware
    # and drives the reach until the fingertips are at the object.
    if execute:
        solved, solved_x = solve_object_reach(
            mover, detector, target_selector, start,
            frame_source=frame_source, rigid_points=rigid_points)
        if solved is None:
            raise RuntimeError(
                "could not reach the selected object's depth; no descent")
        start = solved
        print(f"[look-reach] reach solved: tool x={solved_x*1000:.0f}mm")
        scene = target = None
        for _ in range(3):
            scene, _ = detector.scene(frame_source(discard=1))
            target = target_selector.match(scene)
            if target is not None:
                break
        if target is None:
            raise RuntimeError("target lost after reach solve")
        gripper = scene.gripper
        if gripper is None:
            raise RuntimeError("finger markers not visible after reach solve")
        bbox, center = target.bbox, target.center
        horizontal_error = float(center[0] - gripper.center[0])
        if abs(horizontal_error) > \
                VECTOR_LATERAL_OPENING_FRACTION * gripper.opening_px:
            raise RuntimeError(
                f"target left the jaw corridor in x ({horizontal_error:.0f}px)")
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
        return {"state": "planned", "endpoint": endpoint, "high": high,
                "target_center": tuple(float(value) for value in center)}

    _calibrate_empty_close(mover, high)
    mover.slow_move(start)
    target = None
    for _ in range(3):
        reacquired_frame = frame_source(discard=2)
        reacquired_scene, _ = detector.scene(reacquired_frame)
        target = target_selector.match(reacquired_scene)
        if target is not None:
            break
    if target is None:
        raise RuntimeError(
            "selected target was not reacquired after empty-close calibration")
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


def candidate_reachability(scene, candidate, pose=None, safety=None):
    """Return ``(reachable, reason)`` before a candidate may be locked.

    ``WristSceneDetector`` has already removed floor/arm/tape masks and ranked
    surviving objects by jaw proximity. This final gate keeps the established
    look/reach image envelope, the vector jaw/base-yaw window, and the physical
    pose interlock. A failed gate is not a human veto.
    """
    height, width = scene.frame_shape[:2]
    confidence = float(getattr(candidate, "confidence", 1.0))
    cx, cy = (float(value) for value in candidate.center)
    _x, _y, box_width, box_height = candidate.bbox
    ratio = float(candidate.area) / float(width * height)
    if confidence < MIN_TRACK_CONFIDENCE:
        return False, f"confidence {confidence:.2f} below {MIN_TRACK_CONFIDENCE:.2f}"
    if not 0.001 <= ratio <= 0.12:
        return False, f"area ratio {ratio:.4f} outside portable-object envelope"
    if not 0.20 * width <= cx <= 0.80 * width:
        return False, f"x={cx:.0f}px outside bounded base-yaw search view"
    # With adaptive reach the object's depth is solved and bounded to the
    # verified 285-375 mm band, so a merely CLOSE object is legitimate. Only
    # reject detections sitting essentially on top of the finger markers.
    if cy >= 0.95 * height:
        return False, f"y={cy:.0f}px sits on the finger markers"
    # Table horizon. This camera is nearly horizontal at the working poses, so
    # the near tabletop projects into the LOWER frame while the room behind it
    # (other desks, cups, monitors) projects high. Without this the search
    # happily locked a coffee cup on a far desk. The object must rest on the
    # near table, so its base row has to be below the horizon.
    base_row = float(candidate.bbox[1] + candidate.bbox[3])
    if base_row < TABLE_HORIZON_ROW_RATIO * height:
        return False, (f"base row {base_row:.0f}px is above the table horizon "
                       f"({TABLE_HORIZON_ROW_RATIO * height:.0f}px)")

    gripper = getattr(scene, "gripper", None)
    if gripper is None:
        diagonal = math.hypot(width, height)
        centre = np.asarray(
            config.WRIST_GRIPPER_OPEN_PROFILE["center"], dtype=float)
        centre *= np.asarray([width, height], dtype=float)
        opening_px = (config.WRIST_GRIPPER_OPEN_PROFILE["opening_ratio"]
                      * diagonal)
    else:
        centre = np.asarray(gripper.center, dtype=float)
        opening_px = float(gripper.opening_px)
    lateral_error = cx - float(centre[0])
    lateral_limit = VECTOR_LATERAL_OPENING_FRACTION * opening_px
    if abs(lateral_error) > lateral_limit:
        return False, (
            f"lateral error {lateral_error:.0f}px exceeds the verified "
            f"jaw/base-yaw window {lateral_limit:.0f}px")

    if pose is not None:
        values = [float(value) for value in pose]
        base_delta = abs(values[config.J_BASE] - config.FLOOR_BASE_ANGLE)
        if base_delta > BASE_CENTER_RANGE_DEG:
            return False, (
                f"base yaw is {base_delta:.0f}deg outside the verified "
                f"±{BASE_CENTER_RANGE_DEG}deg centring range")
        safety = safety or PlanarSearchSafety()
        if not safety.pose_is_safe(values):
            return False, "current pose fails the physical collision interlock"
    return True, "reachable"


class LookReachTargetSelector:
    """Persistent multi-object selector and ErrP-ready reject hook.

    Explicit rejection is delegated to :class:`CandidateSelector`, so vetoes
    are image-position based and survive FastSAM instance/list reshuffles.
    Reachability failures are merely skipped and never added to the veto set.
    """

    def __init__(self, selector=None, reachability=candidate_reachability,
                 logger=print):
        self.selector = selector
        self.reachability = reachability
        self.logger = logger
        self.lock = None
        self.current = None
        self._safety = PlanarSearchSafety()

    def _ensure_selector(self, scene):
        if self.selector is None:
            height, width = scene.frame_shape[:2]
            radius = (config.FLOOR_REJECT_RADIUS_RATIO
                      * math.hypot(width, height))
            self.selector = CandidateSelector(reject_radius_px=radius)
        return self.selector

    def choose(self, scene, pose=None):
        """Choose the nearest ranked, reachable, non-vetoed candidate."""
        selector = self._ensure_selector(scene)
        reachable = []
        for index, candidate in enumerate(scene.ranked):
            allowed, reason = self.reachability(
                scene, candidate, pose=pose, safety=self._safety)
            if allowed:
                reachable.append(candidate)
            elif self.logger is not None:
                self.logger(
                    f"[look-reach] SKIP candidate #{index} "
                    f"center={tuple(round(v, 1) for v in candidate.center)}: "
                    f"{reason}")
        selected = selector.choose(reachable)
        if selected is None:
            self.current = None
            self.lock = None
            return None
        self.current = selected
        self.lock = TargetLock.from_candidate(selected)
        if self.logger is not None:
            self.logger(
                "[look-reach] SELECT "
                f"center={tuple(round(v, 1) for v in selected.center)} "
                f"vetoes={len(selector.rejected_points)}")
        return selected

    def reject_current(self, scene=None, pose=None):
        """Programmatic human/ErrP hook: veto current image position and rechoose."""
        if self.current is None:
            return self.choose(scene, pose=pose) if scene is not None else None
        selector = self._ensure_selector(scene) if scene is not None else self.selector
        if selector is None:
            raise RuntimeError("cannot reject before a scene has initialized selection")
        rejected = self.current
        selector.reject(rejected)
        if self.logger is not None:
            self.logger(
                "[look-reach] REJECT "
                f"center={tuple(round(v, 1) for v in rejected.center)}")
        self.current = None
        self.lock = None
        return self.choose(scene, pose=pose) if scene is not None else None

    def match(self, scene):
        """Reacquire only the lock and carry veto pixels with camera motion.

        Eye-in-hand motion translates the whole local scene. The locked target
        supplies that observed translation, which is applied to rejected image
        positions before the next selection. An explicit reject clears the lock,
        so this translation can never resurrect the just-vetoed target.
        """
        if self.lock is None:
            return None
        candidate = match_locked_target(scene, self.lock)
        if candidate is None:
            return None
        old_center = self.lock.center.copy()
        displacement = np.asarray(candidate.center, dtype=float) - old_center
        selector = self._ensure_selector(scene)
        selector.rejected_points = [
            (float(x + displacement[0]), float(y + displacement[1]))
            for x, y in selector.rejected_points]
        self.lock.update(candidate)
        self.current = candidate
        return candidate


def choose_with_rejections(scene, target_selector, reject_count=0, pose=None):
    """Apply N explicit vetoes before returning the target to prosecute."""
    if reject_count < 0:
        raise ValueError("reject count must be non-negative")
    selected = target_selector.choose(scene, pose=pose)
    for _ in range(int(reject_count)):
        if selected is None:
            break
        selected = target_selector.reject_current(scene, pose=pose)
    return selected


def select_initial_target(scene):
    """Compatibility wrapper: choose the nearest reachable scene candidate."""
    return LookReachTargetSelector(logger=None).choose(scene)


def match_locked_target(scene, lock, selector=None):
    """Track the same appearance/position; never jump to a bottom cable/jaw."""
    eligible = []
    old_w, old_h = lock.bbox[2], lock.bbox[3]
    old_aspect = max(old_w, old_h) / max(1.0, min(old_w, old_h))
    for candidate in scene.ranked:
        confidence = float(getattr(candidate, "confidence", 1.0))
        if confidence < MIN_TRACK_CONFIDENCE:
            continue
        if selector is not None and selector.choose([candidate]) is None:
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


BACKGROUND_RING_PX = 45
BACKGROUND_MIN_SAT_DELTA = 22.0
BACKGROUND_MIN_VAL_DELTA = 28.0


def background_distinctness(frame, candidate):
    """Return (sat_delta, val_delta) of a candidate against its local ring.

    The workspace legitimately contains a large flat white sheet (tissue/paper
    covering the Uno's USB run). Its folds segment as compact blobs and sit
    right in front of the jaws, so they out-rank the real object on proximity
    alone. A graspable object, unlike a fold of the sheet it lies on, differs
    from its immediate surroundings in saturation or brightness. This is a
    figure/ground test, not a colour preset: any object distinct from its own
    background passes, whatever its hue.
    """
    import cv2

    height, width = frame.shape[:2]
    x, y, box_width, box_height = candidate.bbox
    inset_x = max(1, int(box_width * 0.15))
    inset_y = max(1, int(box_height * 0.15))
    ix0, ix1 = max(0, x + inset_x), min(width, x + box_width - inset_x)
    iy0, iy1 = max(0, y + inset_y), min(height, y + box_height - inset_y)
    if ix1 - ix0 < 3 or iy1 - iy0 < 3:
        return 0.0, 0.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    inner = hsv[iy0:iy1, ix0:ix1].reshape(-1, 3)
    ox0, ox1 = max(0, x - BACKGROUND_RING_PX), min(width, x + box_width + BACKGROUND_RING_PX)
    oy0, oy1 = max(0, y - BACKGROUND_RING_PX), min(height, y + box_height + BACKGROUND_RING_PX)
    outer = hsv[oy0:oy1, ox0:ox1].copy()
    outer[max(0, y - oy0):max(0, y - oy0) + box_height,
          max(0, x - ox0):max(0, x - ox0) + box_width] = 0
    ring = outer.reshape(-1, 3)
    ring = ring[ring.any(axis=1)]
    if ring.size == 0:
        return 0.0, 0.0
    sat_delta = abs(float(np.median(inner[:, 1])) - float(np.median(ring[:, 1])))
    val_delta = abs(float(np.median(inner[:, 2])) - float(np.median(ring[:, 2])))
    return sat_delta, val_delta


def is_graspable_figure(frame, candidate):
    """True when the candidate stands out from its own local background."""
    sat_delta, val_delta = background_distinctness(frame, candidate)
    return (sat_delta >= BACKGROUND_MIN_SAT_DELTA
            or val_delta >= BACKGROUND_MIN_VAL_DELTA)


def _laterally_fixable_candidate(scene, target_selector, pose=None,
                                 safety=None):
    """Best non-vetoed candidate whose ONLY gate failure is the lateral window.

    Used to decide bounded base centring. Returns ``None`` when any candidate
    is already reachable (no centring needed) or when nothing fails purely
    laterally. Vetoed image positions are excluded exactly as in selection.
    """
    safety = safety or PlanarSearchSafety()
    ranked = getattr(scene, "ranked", None) or []
    selector = target_selector._ensure_selector(scene)
    fixable = None
    for candidate in ranked:
        if selector._is_rejected(candidate):
            continue
        ok, reason = candidate_reachability(
            scene, candidate, pose=pose, safety=safety)
        if ok:
            return None
        if fixable is None and reason.startswith("lateral error"):
            fixable = candidate
    return fixable


SELECTION_EVIDENCE = ROOT / "data" / "vision" / "selection_evidence.jpg"


def _save_selection_evidence(frame, scene):
    """Write what the controller actually sees at the moment it chooses.

    Every earlier mis-grasp argument was guesswork because nothing recorded the
    decision frame. This makes the selection auditable after the fact.
    """
    import cv2

    image = frame.copy()
    for index, candidate in enumerate(scene.ranked[:8]):
        x, y, width, height = candidate.bbox
        score = sum(background_distinctness(frame, candidate))
        ok, reason = candidate_reachability(scene, candidate)
        colour = (0, 0, 255) if index == 0 and ok else (
            (0, 220, 0) if ok else (150, 150, 150))
        cv2.rectangle(image, (x, y), (x + width, y + height), colour,
                      3 if index == 0 else 2)
        cv2.putText(image, f"{index}:{score:.0f}{'' if ok else ' X'}",
                    (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    colour, 2)
        if not ok:
            cv2.putText(image, reason[:38], (x, min(image.shape[0] - 6,
                                                    y + height + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    gripper = getattr(scene, "gripper", None)
    if gripper is not None:
        cv2.drawMarker(image, tuple(int(round(v)) for v in gripper.center),
                       (0, 255, 255), cv2.MARKER_CROSS, 40, 3)
    cv2.imwrite(str(SELECTION_EVIDENCE), image)


RIGID_PROBE_DEG = 3
RIGID_MAX_SHIFT_PX = 8.0


def rigid_with_camera_points(mover, detector, client, frame_source,
                             probe_deg=RIGID_PROBE_DEG, logger=print):
    """Image points that belong to the ROBOT, not the world.

    The servo loom, cable ties and finger tapes are bolted to the arm, so they
    keep the same image position when the arm moves; every real scene object
    shifts. One tiny wrist nudge therefore separates own-hardware detections
    from graspable objects without any colour, region or position heuristic.
    """
    before_frame = frame_source(discard=1)
    before, _ = detector.scene(before_frame)
    pose = list(client.request({"command": "status"})["pose"])
    probe = list(pose)
    probe[config.J_WRIST] = int(np.clip(
        pose[config.J_WRIST] - probe_deg,
        config.SERVO_MIN[config.J_WRIST], config.SERVO_MAX[config.J_WRIST]))
    if probe[config.J_WRIST] == pose[config.J_WRIST]:
        probe[config.J_WRIST] = int(np.clip(
            pose[config.J_WRIST] + probe_deg,
            config.SERVO_MIN[config.J_WRIST], config.SERVO_MAX[config.J_WRIST]))
    if probe[config.J_WRIST] == pose[config.J_WRIST]:
        return []
    mover.slow_move(probe, final_settle=0.5)
    after, _ = detector.scene(frame_source(discard=1))
    mover.slow_move(pose, final_settle=0.5)
    rigid = []
    for candidate in before.ranked:
        nearest = None
        best = None
        for other in after.ranked:
            distance = math.dist(candidate.center, other.center)
            if best is None or distance < best:
                best, nearest = distance, other
        if nearest is not None and best is not None and best <= RIGID_MAX_SHIFT_PX:
            rigid.append(tuple(float(v) for v in candidate.center))
    if rigid:
        logger(f"[look-reach] {len(rigid)} detection(s) are rigid with the "
               "camera (own hardware); excluded")
    return rigid


def _is_rigid(candidate, rigid_points, radius=18.0):
    return any(math.dist(candidate.center, point) <= radius
               for point in rigid_points)


def acquire_initial_target(detector, samples=3, target_selector=None,
                           reject_count=0, pose=None,
                           frame_source=_fresh_frame, rigid_points=()):
    """Choose once, then follow that lock across several fresh segmentations."""
    target_selector = target_selector or LookReachTargetSelector()
    latest_frame = None
    latest_scene = None
    selected = None
    remaining_rejects = int(reject_count)
    if remaining_rejects < 0:
        raise ValueError("reject count must be non-negative")
    for _ in range(samples):
        latest_frame = frame_source(discard=1)
        latest_scene, _ = detector.scene(latest_frame)
        # Drop folds of the flat sheet covering the workspace: they segment as
        # compact blobs nearest the jaws and would otherwise always out-rank the
        # real object. Only figures distinct from their own background survive.
        latest_scene.ranked = [item for item in latest_scene.ranked
                               if is_graspable_figure(latest_frame, item)
                               and not _is_rigid(item, rigid_points)]
        # Rank by figure/ground distinctness, not jaw proximity. The workspace
        # sheet lies directly in front of the fingers, so proximity ranking put
        # paper first every time; the real object is the one that stands out
        # most from its own background.
        latest_scene.ranked.sort(
            key=lambda item: sum(background_distinctness(latest_frame, item)),
            reverse=True)
        _save_selection_evidence(latest_frame, latest_scene)
        if selected is None:
            selected = target_selector.choose(latest_scene, pose=pose)
            while selected is not None and remaining_rejects:
                selected = target_selector.reject_current(
                    latest_scene, pose=pose)
                remaining_rejects -= 1
        elif selected is not None:
            matched = target_selector.match(latest_scene)
            if matched is not None:
                selected = matched
    return latest_frame, latest_scene, selected


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


def run_controller(client, max_steps=24, execute=False, allow_grasp=False,
                   target_selector=None, reject_count=0, detector=None,
                   frame_source=_fresh_frame):
    detector = detector or WristSceneDetector()
    target_selector = target_selector or LookReachTargetSelector()
    safety = PlanarSearchSafety()
    mover = FloorServo(client, calib=None)
    observation_pose = list(client.request({"command": "status"})["pose"])
    frame, scene, selected = acquire_initial_target(
        detector, target_selector=target_selector,
        reject_count=reject_count, pose=observation_pose,
        frame_source=frame_source)
    if selected is None:
        print("[look-reach] no reachable non-vetoed target; no motion")
        return {"state": "no-target", "moved": False}
    lock = target_selector.lock
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
        frame = frame_source(discard=2)
        scene, _ = detector.scene(frame)
        candidate = target_selector.match(scene)
        if candidate is None:
            raise RuntimeError("locked target lost; refusing blind motion")
        lock = target_selector.lock
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
    parser.add_argument("--vector-inset-mm", type=float, default=0.0,
                        help="signed endpoint-x correction for vector grasp")
    parser.add_argument("--reject-count", type=int, default=0,
                        help="veto this many ranked reachable targets before run")
    args = parser.parse_args()
    if args.grasp and not args.run:
        parser.error("--grasp requires --run")
    if args.reject_count < 0:
        parser.error("--reject-count must be non-negative")
    client = ArmSessionClient()
    target_selector = LookReachTargetSelector()
    table_z_m = (load_table_z_m()
                 if args.vector_grasp and args.run else 0.0)
    result = (run_constant_x_grasp(
                  client, execute=args.run, advance_mm=args.vector_inset_mm,
                  table_z_m=table_z_m, target_selector=target_selector,
                  reject_count=args.reject_count)
              if args.vector_grasp else
              run_controller(client, args.max_steps,
                             execute=args.run, allow_grasp=args.grasp,
                             target_selector=target_selector,
                             reject_count=args.reject_count))
    print(f"[look-reach] RESULT {result}")


if __name__ == "__main__":
    main()
