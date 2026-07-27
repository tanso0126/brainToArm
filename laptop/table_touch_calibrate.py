"""Self-calibrate the real table height with the wrist camera and fingertips.

The eye-in-hand finger tapes are rigid in camera coordinates, so their ordinary
pixel motion cannot reveal a world-space descent.  Instead this routine compares
the table background immediately before/after each 2 mm fixed-pitch command.
Before contact the wrist camera moves and table features flow; after contact the
mechanism/compliance stalls and the flow collapses.  Finger-marker deformation
is logged as corroborating evidence, never used as the sole contact signal.

All joints approach their measurement pose from five servo degrees below to
remove direction-dependent backlash.  Nothing moves without ``--run``.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
import time

import numpy as np
from scipy.optimize import least_squares

import arm_fk
import config
from arm_session import ArmSessionClient
from floor_servo import FloorServo, _fresh_frame
from look_reach import VECTOR_START_POSE, cumulative_tool_angle_deg
from wrist_search import PlanarSearchSafety


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "calibration" / "table_touch.json"
# The current object is centred near x=0.350 m. Touch on the clear, proximal
# table while remaining outside the conservative 287 mm base/hand keep-out.
# This remains configurable at the CLI for a changed setup.
TOUCH_X_M = 0.300
START_Z_M = 0.040
# The collision interlock permits only a small negative FK tolerance. Integer
# servo quantisation puts this request near -1.6 mm, still above that bound.
MIN_COMMAND_Z_M = -0.002
STEP_Z_M = 0.002
SETTLE_S = 1.2
BACKLASH_DEG = 5


def solve_fixed_pitch_pose(x_m, z_m, pitch_deg, guess, template=None):
    """Three exact planar constraints -> motor 2/3/4 commands."""
    template = list(VECTOR_START_POSE if template is None else template)
    lower = np.asarray((config.SERVO_MIN[config.J_SHOULDER],
                        config.SERVO_MIN[config.J_ELBOW],
                        config.SERVO_MIN[config.J_WRIST]), dtype=float)
    upper = np.asarray((config.SERVO_MAX[config.J_SHOULDER],
                        config.SERVO_MAX[config.J_ELBOW],
                        config.SERVO_MAX[config.J_WRIST]), dtype=float)

    def residual(values):
        pose = list(template)
        pose[config.J_SHOULDER], pose[config.J_ELBOW], pose[config.J_WRIST] = values
        tool = arm_fk.tool_position(pose)
        return np.asarray(((tool[0] - x_m) * 1000.0,
                           (tool[2] - z_m) * 1000.0,
                           cumulative_tool_angle_deg(pose) - pitch_deg))

    result = least_squares(
        residual, np.clip(np.asarray(guess, dtype=float), lower, upper),
        bounds=(lower, upper), max_nfev=500,
        xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if not result.success or np.linalg.norm(residual(result.x)) > 1.2:
        raise RuntimeError(f"fixed-pitch IK failed at z={z_m*1000:.1f} mm")
    pose = list(template)
    rounded = [int(round(value)) for value in result.x]
    # Servo commands are integers. Refine the naïve rounding locally so a 2 mm
    # calibration step is not swallowed by up to ~2.3 mm of joint quantization.
    best = None
    for shoulder in range(rounded[0] - 1, rounded[0] + 2):
        for elbow in range(rounded[1] - 1, rounded[1] + 2):
            for wrist in range(rounded[2] - 1, rounded[2] + 2):
                values = (shoulder, elbow, wrist)
                if any(not low <= value <= high for value, low, high in zip(
                        values, lower, upper)):
                    continue
                score = float(np.dot(residual(values), residual(values)))
                if best is None or score < best[0]:
                    best = (score, values)
    if best is None:
        raise RuntimeError("fixed-pitch integer refinement found no command")
    for joint, value in zip((config.J_SHOULDER, config.J_ELBOW, config.J_WRIST),
                            best[1]):
        pose[joint] = int(value)
    return pose


def fixed_pitch_path(x_m=TOUCH_X_M, start_z_m=START_Z_M,
                     minimum_z_m=MIN_COMMAND_Z_M, step_z_m=STEP_Z_M):
    pitch = cumulative_tool_angle_deg(VECTOR_START_POSE)
    guess = np.asarray(VECTOR_START_POSE[1:4], dtype=float)
    path = []
    last_pose = None
    last_actual_z = None
    z = float(start_z_m)
    while z >= minimum_z_m - 1e-9:
        pose = solve_fixed_pitch_pose(x_m, z, pitch, guess)
        guess = np.asarray(pose[1:4], dtype=float)
        actual_z = float(arm_fk.tool_position(pose)[2])
        # Integer servo quantization maps adjacent ideal 2 mm levels to the same
        # command. Never execute/measure a duplicate: it would create a fake
        # zero-flow "contact" sample without requesting physical descent.
        if (pose != last_pose
                and (last_actual_z is None or actual_z < last_actual_z - 0.0007)):
            path.append((z, pose))
            last_pose = pose
            last_actual_z = actual_z
        z -= step_z_m
    return path


def backlash_prepose(target, amount=BACKLASH_DEG):
    """Ensure motors 2/3/4 make their final move in the increasing direction."""
    result = list(target)
    for joint in (config.J_SHOULDER, config.J_ELBOW, config.J_WRIST):
        result[joint] = max(config.SERVO_MIN[joint], target[joint] - amount)
    return result


def median_table_flow(before, after):
    """Robust median table-feature displacement between two wrist frames."""
    import cv2
    gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    height, width = gray_before.shape
    mask = np.zeros_like(gray_before)
    # Table-only middle band: exclude gripper/USB cable at the bottom and room
    # clutter/horizon at the top.  No background reference image is required.
    mask[int(0.12 * height):int(0.72 * height),
         int(0.12 * width):int(0.78 * width)] = 255
    points = cv2.goodFeaturesToTrack(
        gray_before, maxCorners=300, qualityLevel=0.01,
        minDistance=8, mask=mask, blockSize=7)
    if points is None or len(points) < 12:
        return None, 0
    moved, status, _error = cv2.calcOpticalFlowPyrLK(
        gray_before, gray_after, points, None,
        winSize=(31, 31), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    if moved is None:
        return None, 0
    valid = status.reshape(-1).astype(bool)
    displacement = moved.reshape(-1, 2)[valid] - points.reshape(-1, 2)[valid]
    if len(displacement) < 10:
        return None, len(displacement)
    # Reject independently moving outliers (cable/lighting edge) with a median
    # absolute-deviation gate, then report translational flow magnitude.
    centre = np.median(displacement, axis=0)
    residual = np.linalg.norm(displacement - centre, axis=1)
    mad = float(np.median(np.abs(residual - np.median(residual))))
    keep = residual <= max(1.0, np.median(residual) + 3.0 * max(mad, 0.1))
    if np.count_nonzero(keep) < 10:
        return None, int(np.count_nonzero(keep))
    vector = np.median(displacement[keep], axis=0)
    return float(np.linalg.norm(vector)), int(np.count_nonzero(keep))


def gripper_signature(detector, frame):
    observation, _ = detector.detect(frame)
    if observation.gripper is None:
        return None
    gripper = observation.gripper
    return np.asarray((gripper.center[0], gripper.center[1],
                       gripper.opening_px, gripper.angle_deg), dtype=float)


@dataclass
class TouchStep:
    command_z_mm: float
    fk_z_mm: float
    pose234: list
    flow_px: float | None
    flow_points: int
    marker_shift_px: float | None


def move_from_below(mover, safety, target, settle_s=SETTLE_S):
    current = mover.client.request({"command": "status"})["pose"]
    prepose = backlash_prepose(target)
    if not safety.transition_is_safe(current, prepose):
        raise RuntimeError("backlash prepose failed collision model")
    if not safety.transition_is_safe(prepose, target):
        raise RuntimeError("measurement pose failed collision model")
    mover.slow_move(prepose, final_settle=0.25)
    mover.slow_move(target, final_settle=settle_s)


def run_touch_trial(client, execute=False, touch_x_m=TOUCH_X_M):
    path = fixed_pitch_path(x_m=touch_x_m)
    safety = PlanarSearchSafety()
    for (_za, a), (_zb, b) in zip(path, path[1:]):
        if not safety.transition_is_safe(a, backlash_prepose(b)):
            raise RuntimeError("planned touch path has an unsafe prepose")
        if not safety.transition_is_safe(backlash_prepose(b), b):
            raise RuntimeError("planned touch path has an unsafe measurement pose")
    if not execute:
        return {"state": "planned", "poses": len(path),
                "first": path[0], "last": path[-1]}

    mover = FloorServo(client, calib=None)
    records = []
    baseline_flows = []
    contact_z = None
    entered_touch_path = False
    try:
        move_from_below(mover, safety, path[0][1])
        entered_touch_path = True
        previous = _fresh_frame(discard=2)
        baseline_signature = gripper_signature(mover.marker_detector, previous)
        for command_z, pose in path[1:]:
            move_from_below(mover, safety, pose)
            current = _fresh_frame(discard=2)
            flow, points = median_table_flow(previous, current)
            signature = gripper_signature(mover.marker_detector, current)
            marker_shift = (
                None if signature is None or baseline_signature is None
                else float(np.linalg.norm(signature[:2]
                                          - baseline_signature[:2])))
            fk_z_mm = float(arm_fk.tool_position(pose)[2] * 1000.0)
            record = TouchStep(command_z * 1000.0, fk_z_mm, pose[1:4], flow,
                               points, marker_shift)
            records.append(record)
            print(f"[table-touch] z_cmd={command_z*1000:5.1f}mm "
                  f"fk_z={fk_z_mm:5.1f}mm pose234={pose[1:4]} "
                  f"flow={flow} points={points} "
                  f"marker_shift={marker_shift}")
            if flow is not None and len(baseline_flows) < 4:
                baseline_flows.append(flow)
            baseline = (float(np.median(baseline_flows))
                        if len(baseline_flows) >= 3 else None)
            low = (baseline is not None and flow is not None
                   and flow < min(2.0, 0.35 * baseline))
            if low:
                # Never confirm a suspected contact by driving another 2 mm into
                # the table. Retreat one path level, then reproduce exactly the
                # same descent. A texture/flow failure will not consistently
                # recur; a physical motion stall will.
                candidate_index = max(0, len(records) - 2)
                retreat_pose = (path[0][1] if candidate_index == 0
                                else path[candidate_index][1])
                move_from_below(mover, safety, retreat_pose, settle_s=0.5)
                confirm_before = _fresh_frame(discard=2)
                move_from_below(mover, safety, pose)
                confirm_after = _fresh_frame(discard=2)
                confirm_flow, confirm_points = median_table_flow(
                    confirm_before, confirm_after)
                confirm_low = (
                    confirm_flow is not None
                    and confirm_flow < min(2.0, 0.35 * baseline))
                print(f"[table-touch] candidate confirmation "
                      f"flow={confirm_flow} points={confirm_points} "
                      f"confirmed={confirm_low}")
                if confirm_low:
                    contact_z = fk_z_mm
                    print(f"[table-touch] CONTACT at FK z={contact_z:.1f}mm")
                    break
                # Continue relative to the real confirmed candidate pose, not
                # the stale pre-confirmation frame.
                current = confirm_after
            previous = current
    finally:
        # A camera/flow failure near the table must not strand the arm at the
        # lowest command. Retreat to the known-clear first measurement pose.
        if entered_touch_path:
            move_from_below(mover, safety, path[0][1])

    if contact_z is None:
        raise RuntimeError("no repeatable table-contact flow plateau before safety limit")
    return {"state": "contact", "z_table_mm": contact_z,
            "records": [asdict(record) for record in records]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--x-mm", type=float, default=TOUCH_X_M * 1000.0,
                        help="clear sagittal table-touch distance from base")
    args = parser.parse_args()
    result = run_touch_trial(ArmSessionClient(), execute=args.run,
                             touch_x_m=args.x_mm / 1000.0)
    if args.run:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[table-touch] saved {OUTPUT}")
    else:
        print(f"[table-touch] DRY RUN {result}")


if __name__ == "__main__":
    main()
