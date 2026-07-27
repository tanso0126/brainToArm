"""Self-calibrate the real table height with the wrist camera and fingertips.

The eye-in-hand finger tapes are rigid in camera coordinates, so their ordinary
pixel motion cannot reveal a world-space descent.  Instead this routine compares
the table background immediately before/after each fixed-pitch command.
Before contact the wrist camera moves and table features flow; first contact can
momentarily collapse that flow.  The compliant printed fingers may then flex
while the camera resumes moving, so sustained marker deformation—carried across
temporarily missing tape observations—is independent, co-equal contact evidence.

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
from arm_safety import PhysicalArmSafety
from arm_session import ArmSessionClient, DEEPEST_TABLE_TOUCH_Z_M
from floor_servo import FloorServo, _fresh_frame
from look_reach import VECTOR_START_POSE, cumulative_tool_angle_deg


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "calibration" / "table_touch.json"
# The verified clean wood is at x=0.330 m, just proximal to the known-good
# object at x=0.350 m. x=0.300 is forbidden: the sagging USB cable, Uno-box
# edge, and table edge there generated misleading snag/deformation evidence.
TOUCH_X_M = 0.330
START_Z_M = 0.040
# Default remains the old hardware-safe dry limit. At clean x=0.330 m, physical
# contact is now expected around FK z=-12..-6 mm; the operator explicitly opts
# into that deeper probe with ``--min-z-mm -18``.
MIN_COMMAND_Z_M = -0.002
STEP_Z_M = 0.002
FINE_APPROACH_BELOW_Z_M = -0.004
FINE_STEP_Z_M = 0.001
SETTLE_S = 1.2
BACKLASH_DEG = 5
MARKER_CONTACT_SHIFT_PX = 3.0
MARKER_CONTACT_STEPS = 2
MARKER_HARD_STOP_SHIFT_PX = 10.0
MAX_MARKER_PRESS_DEPTH_MM = 4.0


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


def _fine_descent_pose(x_m, z_m, pitch_deg, candidate, previous,
                       previous_z_m):
    """Choose a distinct nearby integer pose for a real ~1 mm descent."""
    best = None
    for shoulder in range(previous[config.J_SHOULDER] - 3,
                          previous[config.J_SHOULDER] + 4):
        for elbow in range(previous[config.J_ELBOW] - 3,
                           previous[config.J_ELBOW] + 4):
            for wrist in range(previous[config.J_WRIST] - 3,
                               previous[config.J_WRIST] + 4):
                values = (shoulder, elbow, wrist)
                if any(not config.SERVO_MIN[joint] <= value
                       <= config.SERVO_MAX[joint]
                       for joint, value in zip(
                           (config.J_SHOULDER, config.J_ELBOW,
                            config.J_WRIST), values)):
                    continue
                pose = list(candidate)
                for joint, value in zip(
                        (config.J_SHOULDER, config.J_ELBOW, config.J_WRIST),
                        values):
                    pose[joint] = value
                tool = arm_fk.tool_position(pose)
                residual = np.asarray((
                    (tool[0] - x_m) * 1000.0,
                    (tool[2] - z_m) * 1000.0,
                    cumulative_tool_angle_deg(pose) - pitch_deg))
                # Preserve the existing x/pitch tolerances and require a real
                # sub-2 mm downward move despite integer servo quantization.
                descent = previous_z_m - float(tool[2])
                if (not 0.00035 <= descent <= 0.0016
                        or abs(residual[0]) > 2.0
                        or abs(residual[1]) > 1.2
                        or abs(residual[2]) > 1.5):
                    continue
                score = float(np.dot(residual, residual))
                if best is None or score < best[0]:
                    best = (score, pose)
    return candidate if best is None else best[1]


def fixed_pitch_path(x_m=TOUCH_X_M, start_z_m=START_Z_M,
                     minimum_z_m=MIN_COMMAND_Z_M, step_z_m=STEP_Z_M):
    pitch = cumulative_tool_angle_deg(VECTOR_START_POSE)
    guess = np.asarray(VECTOR_START_POSE[1:4], dtype=float)
    path = []
    last_pose = None
    last_actual_z = None
    z = float(start_z_m)
    fine_requested = step_z_m <= FINE_STEP_Z_M + 1e-12
    while z >= minimum_z_m - 1e-9:
        pose = solve_fixed_pitch_pose(x_m, z, pitch, guess)
        if (last_pose is not None
                and (fine_requested
                     or z < FINE_APPROACH_BELOW_Z_M - 1e-12)):
            pose = _fine_descent_pose(
                x_m, z, pitch, pose, last_pose, last_actual_z)
        guess = np.asarray(pose[1:4], dtype=float)
        actual_z = float(arm_fk.tool_position(pose)[2])
        # Integer servo quantization can map adjacent requested levels to the same
        # command. Never execute/measure a duplicate: it would create a fake
        # zero-flow "contact" sample without requesting physical descent.
        minimum_descent = (
            0.00035 if (fine_requested
                        or z < FINE_APPROACH_BELOW_Z_M - 1e-12)
            else 0.0007)
        if (pose != last_pose
                and (last_actual_z is None
                     or actual_z < last_actual_z - minimum_descent)):
            path.append((z, pose))
            last_pose = pose
            last_actual_z = actual_z
        z -= (FINE_STEP_Z_M
              if z <= FINE_APPROACH_BELOW_Z_M + 1e-12
              else step_z_m)
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
    if (before is None or after is None or before.shape != after.shape
            or before.ndim != 3 or before.shape[2] != 3):
        return None, 0
    if ((np.issubdtype(before.dtype, np.floating)
         and not np.isfinite(before).all())
            or (np.issubdtype(after.dtype, np.floating)
                and not np.isfinite(after).all())):
        return None, 0
    gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    height, width = gray_before.shape

    def estimate(points):
        if points is None or len(points) < 12:
            return None, 0
        moved, status, _error = cv2.calcOpticalFlowPyrLK(
            gray_before, gray_after, points, None,
            winSize=(31, 31), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      30, 0.01))
        if moved is None or status is None:
            return None, 0
        original = points.reshape(-1, 2)
        moved = moved.reshape(-1, 2)
        valid = status.reshape(-1).astype(bool)
        valid &= np.isfinite(original).all(axis=1)
        valid &= np.isfinite(moved).all(axis=1)
        displacement = moved[valid] - original[valid]
        if len(displacement) < 10:
            return None, len(displacement)
        # Reject independently moving outliers (cable/lighting edge) with a
        # median absolute-deviation gate, then report translational flow.
        centre = np.median(displacement, axis=0)
        residual = np.linalg.norm(displacement - centre, axis=1)
        finite_residual = np.isfinite(residual)
        displacement = displacement[finite_residual]
        residual = residual[finite_residual]
        if len(residual) < 10:
            return None, len(residual)
        residual_median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - residual_median)))
        if not np.isfinite(residual_median) or not np.isfinite(mad):
            return None, 0
        keep = residual <= max(
            1.0, residual_median + 3.0 * max(mad, 0.1))
        kept = int(np.count_nonzero(keep))
        if kept < 10:
            return None, kept
        vector = np.median(displacement[keep], axis=0)
        magnitude = float(np.linalg.norm(vector))
        if not np.isfinite(magnitude):
            return None, kept
        return magnitude, kept

    primary_mask = np.zeros_like(gray_before)
    # Table-only middle band: exclude gripper/USB cable at the bottom and room
    # clutter/horizon at the top.  No background reference image is required.
    primary_mask[int(0.12 * height):int(0.72 * height),
                 int(0.12 * width):int(0.78 * width)] = 255
    primary_points = cv2.goodFeaturesToTrack(
        gray_before, maxCorners=300, qualityLevel=0.01,
        minDistance=8, mask=primary_mask, blockSize=7)
    primary = estimate(primary_points)
    if primary[1] >= 150:
        return primary

    # Close to the wood the original band yielded only 75–105 tracked points.
    # Retry with more table area and a lower corner threshold. The bottom 12%
    # remains excluded so the gripper and cable cannot dominate the estimate.
    expanded_mask = np.zeros_like(gray_before)
    expanded_mask[int(0.05 * height):int(0.88 * height),
                  int(0.04 * width):int(0.96 * width)] = 255
    expanded_points = cv2.goodFeaturesToTrack(
        gray_before, maxCorners=600, qualityLevel=0.003,
        minDistance=5, mask=expanded_mask, blockSize=5)
    expanded = estimate(expanded_points)
    return expanded if expanded[1] > primary[1] else primary


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


@dataclass(frozen=True)
class ContactEvidence:
    kind: str
    contact_path_index: int
    confirmation_path_index: int
    onset_path_index: int


@dataclass
class _PendingMarkerOnset:
    threshold_px: float
    contact_path_index: int
    onset_path_index: int
    onset_z_mm: float


class MarkerEvidenceTracker:
    """Stateful marker onset/strain detector that carries across missing tape."""

    def __init__(self):
        self.last_valid_shift = None
        self.last_valid_path_index = None
        self.pending = None
        self.high_count = 0
        self.high_onset_path_index = None
        self.high_contact_path_index = None
        self.high_onset_z_mm = None

    @staticmethod
    def _valid_shift(record):
        value = record.marker_shift_px
        return (None if value is None or not np.isfinite(value)
                else float(value))

    def observe(self, record, path_index):
        shift = self._valid_shift(record)
        if shift is None:
            # Missing tape neither resets a pending onset nor the two-valid-
            # sample strain stop. The next valid observation continues both.
            return None

        evidence = None
        if self.pending is not None:
            pending = self.pending
            if shift > pending.threshold_px:
                evidence = ContactEvidence(
                    "marker-sustained",
                    pending.contact_path_index,
                    path_index,
                    pending.onset_path_index)
            self.pending = None
        elif self.last_valid_shift is not None:
            threshold = max(
                MARKER_CONTACT_SHIFT_PX,
                self.last_valid_shift + MARKER_CONTACT_SHIFT_PX)
            if shift > threshold:
                self.pending = _PendingMarkerOnset(
                    threshold,
                    (max(0, path_index - 1)
                     if self.last_valid_path_index is None
                     else self.last_valid_path_index),
                    path_index,
                    float(record.command_z_mm))

        if shift > MARKER_HARD_STOP_SHIFT_PX:
            if self.high_count == 0:
                self.high_onset_path_index = path_index
                self.high_contact_path_index = (
                    max(0, path_index - 1)
                    if self.last_valid_path_index is None
                    else self.last_valid_path_index)
                self.high_onset_z_mm = float(record.command_z_mm)
            self.high_count += 1
            if self.high_count >= MARKER_CONTACT_STEPS:
                # The strain guard has priority over ordinary/noisy onset
                # evidence: it terminates descent immediately.
                evidence = ContactEvidence(
                    "marker-safety",
                    self.high_contact_path_index,
                    path_index,
                    self.high_onset_path_index)
        else:
            self.high_count = 0
            self.high_onset_path_index = None
            self.high_contact_path_index = None
            self.high_onset_z_mm = None

        self.last_valid_shift = shift
        self.last_valid_path_index = path_index
        return evidence

    def press_depth_guard(self, record, path_index):
        """Stop at onset+4 mm even if missing marker samples delay confirmation."""
        if self.pending is not None:
            pressed_mm = (
                self.pending.onset_z_mm - float(record.command_z_mm))
            if pressed_mm + 1e-6 >= MAX_MARKER_PRESS_DEPTH_MM:
                return ContactEvidence(
                    "marker-guard",
                    self.pending.contact_path_index,
                    path_index,
                    self.pending.onset_path_index)
        if self.high_count and self.high_onset_path_index is not None:
            # The high-strain sample itself is an onset even when the rolling
            # +3 px detector did not arm.
            pressed_mm = (
                self.high_onset_z_mm - float(record.command_z_mm))
            if pressed_mm + 1e-6 >= MAX_MARKER_PRESS_DEPTH_MM:
                return ContactEvidence(
                    "marker-guard",
                    self.high_contact_path_index,
                    path_index,
                    self.high_onset_path_index)
        return None

    def seed_from_confirmation(self, records, first_path_index):
        """Continue a pass in the replay's marker coordinate system."""
        for offset, record in enumerate(records):
            self.observe(record, first_path_index + offset)


def flow_contact_evidence(record, baseline_flow, current_path_index):
    if (baseline_flow is None or record.flow_px is None
            or not np.isfinite(record.flow_px)
            or record.flow_px >= min(2.0, 0.35 * baseline_flow)):
        return None
    return ContactEvidence(
        "flow-collapse", current_path_index, current_path_index,
        current_path_index)


def move_from_below(mover, safety, target, settle_s=SETTLE_S):
    current = mover.client.request({"command": "status"})["pose"]
    prepose = backlash_prepose(target)
    if not safety.transition_is_safe(current, prepose):
        raise RuntimeError("backlash prepose failed collision model")
    if not safety.transition_is_safe(prepose, target):
        raise RuntimeError("measurement pose failed collision model")
    mover.slow_move(prepose, final_settle=0.25)
    mover.slow_move(target, final_settle=settle_s)


def retreat_to_trial_start(mover, safety, path, current_path_index,
                           settle_s=0.25):
    """Retrace the measured path upward before another backlash approach.

    A direct move from a near-table pose to a low-joint backlash prepose can
    sweep the hand through the base even when both endpoints are valid.  Every
    reverse edge below was already collision checked on the forward plan.
    """
    for index in range(current_path_index - 1, -1, -1):
        target = path[index][1]
        current = mover.client.request({"command": "status"})["pose"]
        if not safety.transition_is_safe(current, target):
            raise RuntimeError(
                f"safe-height retreat failed collision model at path index {index}")
        mover.slow_move(target, final_settle=settle_s)
    return 0


def _replay_for_confirmation(mover, safety, path, current_path_index, evidence):
    """Retreat high, replay the descent, and reproduce one evidence window."""
    retreat_to_trial_start(mover, safety, path, current_path_index)
    baseline_frame = _fresh_frame(discard=2)
    baseline_signature = gripper_signature(
        mover.marker_detector, baseline_frame)

    observe_from = (
        evidence.confirmation_path_index
        if evidence.kind == "flow-collapse"
        else evidence.contact_path_index)
    for index in range(1, observe_from):
        move_from_below(mover, safety, path[index][1], settle_s=0.25)
    previous = _fresh_frame(discard=2)
    confirmation = []
    for index in range(observe_from, evidence.confirmation_path_index + 1):
        move_from_below(mover, safety, path[index][1])
        current = _fresh_frame(discard=2)
        flow, points = median_table_flow(previous, current)
        signature = gripper_signature(mover.marker_detector, current)
        marker_shift = (
            None if signature is None or baseline_signature is None
            else float(np.linalg.norm(signature[:2] - baseline_signature[:2])))
        command_z, pose = path[index]
        confirmation.append(TouchStep(
            command_z * 1000.0,
            float(arm_fk.tool_position(pose)[2] * 1000.0),
            pose[1:4], flow, points, marker_shift))
        previous = current
    return confirmation, previous, baseline_signature


def _evidence_confirmed(evidence, confirmation, baseline_flow):
    if evidence.kind == "flow-collapse":
        flow = confirmation[-1].flow_px if confirmation else None
        return (flow is not None and baseline_flow is not None
                and flow < min(2.0, 0.35 * baseline_flow))
    tracker = MarkerEvidenceTracker()
    observed = []
    for index, record in enumerate(confirmation):
        candidate = tracker.observe(record, index)
        if candidate is not None:
            observed.append(candidate.kind)
        guard = tracker.press_depth_guard(record, index)
        if guard is not None:
            observed.append(guard.kind)
    if evidence.kind == "marker-sustained":
        return "marker-sustained" in observed
    if evidence.kind == "marker-safety":
        return "marker-safety" in observed
    if evidence.kind == "marker-guard":
        return any(kind.startswith("marker-") for kind in observed)
    return False


def _touch_steps_from_payload(payload):
    rows = payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("table-touch replay needs a non-empty records list")
    records = []
    for index, row in enumerate(rows):
        try:
            marker = row.get("marker_shift_px")
            flow = row.get("flow_px")
            record = TouchStep(
                float(row["command_z_mm"]),
                float(row["fk_z_mm"]),
                [int(value) for value in row["pose234"]],
                None if flow is None else float(flow),
                int(row["flow_points"]),
                None if marker is None else float(marker))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid table-touch replay record {index}") from exc
        if (len(record.pose234) != 3
                or not np.isfinite(record.command_z_mm)
                or not np.isfinite(record.fk_z_mm)
                or (record.flow_px is not None
                    and not np.isfinite(record.flow_px))
                or (record.marker_shift_px is not None
                    and not np.isfinite(record.marker_shift_px))):
            raise ValueError(f"non-finite table-touch replay record {index}")
        records.append(record)
    return records


def replay_contact_records(records):
    """Find the first point where the saved pass should have strain-stopped."""
    tracker = MarkerEvidenceTracker()
    for index, record in enumerate(records):
        evidence = tracker.observe(record, index)
        if evidence is not None and evidence.kind == "marker-safety":
            contact = records[evidence.contact_path_index]
            onset = records[evidence.onset_path_index]
            stopped = records[evidence.confirmation_path_index]
            return {
                "state": "contact",
                "z_table_mm": round(float(contact.command_z_mm), 6),
                "fk_z_table_mm": float(contact.fk_z_mm),
                "onset_z_mm": round(float(onset.command_z_mm), 6),
                "would_stop_z_mm": round(float(stopped.command_z_mm), 6),
                "evidence_kind": evidence.kind,
                "source": "offline-replay",
                "records": [asdict(item) for item in records],
            }
    return {
        "state": "no-contact",
        "source": "offline-replay",
        "records": [asdict(item) for item in records],
    }


def replay_touch_file(input_path, output_path=OUTPUT):
    input_path = Path(input_path)
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"could not read table-touch replay {input_path}") from exc
    result = replay_contact_records(_touch_steps_from_payload(payload))
    if result["state"] == "contact":
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_touch_trial(client, execute=False, touch_x_m=TOUCH_X_M,
                    minimum_z_m=MIN_COMMAND_Z_M, start_z_m=START_Z_M,
                    step_z_m=STEP_Z_M):
    minimum_z_m = float(minimum_z_m)
    start_z_m = float(start_z_m)
    step_z_m = float(step_z_m)
    if not DEEPEST_TABLE_TOUCH_Z_M <= minimum_z_m < start_z_m <= START_Z_M:
        raise ValueError(
            "minimum table-touch z must be between "
            f"{DEEPEST_TABLE_TOUCH_Z_M * 1000:.0f} and "
            f"{START_Z_M * 1000:.0f} mm and below its start")
    if step_z_m <= 0:
        raise ValueError("table-touch step must be positive")
    path = fixed_pitch_path(
        x_m=touch_x_m, start_z_m=start_z_m,
        minimum_z_m=minimum_z_m, step_z_m=step_z_m)
    safety = PhysicalArmSafety(table_z_m=minimum_z_m)
    for (_za, a), (_zb, b) in zip(path, path[1:]):
        if not safety.transition_is_safe(a, backlash_prepose(b)):
            raise RuntimeError("planned touch path has an unsafe prepose")
        if not safety.transition_is_safe(backlash_prepose(b), b):
            raise RuntimeError("planned touch path has an unsafe measurement pose")
        if not safety.transition_is_safe(b, a):
            raise RuntimeError("planned touch path has an unsafe retreat edge")
    if not execute:
        return {"state": "planned", "poses": len(path),
                "first": path[0], "last": path[-1]}

    mover = FloorServo(
        client, calib=None, move_command="table_touch_move",
        move_options={"table_z_m": minimum_z_m})
    records = []
    marker_tracker = MarkerEvidenceTracker()
    baseline_flows = []
    contact_z = None
    contact_fk_z = None
    entered_touch_path = False
    current_path_index = 0
    try:
        move_from_below(mover, safety, path[0][1])
        entered_touch_path = True
        previous = _fresh_frame(discard=2)
        baseline_signature = gripper_signature(mover.marker_detector, previous)
        for path_index, (command_z, pose) in enumerate(path[1:], start=1):
            move_from_below(mover, safety, pose)
            current_path_index = path_index
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
            marker_evidence = marker_tracker.observe(
                record, current_path_index)
            marker_guard = marker_tracker.press_depth_guard(
                record, current_path_index)
            evidence = (marker_evidence or marker_guard
                        or flow_contact_evidence(
                            record, baseline, current_path_index))
            if evidence is not None:
                # Confirmation always starts by retracing the known touch path
                # to its clear first pose.  It never asks the collision model
                # for a low-pose -> backlash-prepose transition.
                confirmation, confirm_after, confirm_signature = (
                    _replay_for_confirmation(
                        mover, safety, path, current_path_index, evidence))
                current_path_index = evidence.confirmation_path_index
                confirmed = _evidence_confirmed(
                    evidence, confirmation, baseline)
                details = ", ".join(
                    f"z={step.command_z_mm:.1f} "
                    f"flow={step.flow_px} marker={step.marker_shift_px}"
                    for step in confirmation)
                print(f"[table-touch] {evidence.kind} confirmation "
                      f"{details} confirmed={confirmed}")
                if confirmed:
                    contact_record = path[evidence.contact_path_index]
                    onset_record = path[evidence.onset_path_index]
                    contact_z = round(float(contact_record[0] * 1000.0), 6)
                    contact_fk_z = float(
                        arm_fk.tool_position(contact_record[1])[2] * 1000.0)
                    print(f"[table-touch] CONTACT at command z={contact_z:.1f}mm "
                          f"(FK={contact_fk_z:.1f}mm)")
                    break
                # Continue relative to the real replay pose and its new marker
                # baseline. Do not mix marker shifts from two physical trials.
                current = confirm_after
                baseline_signature = confirm_signature
                marker_tracker = MarkerEvidenceTracker()
                confirmation_first_index = (
                    evidence.confirmation_path_index
                    if evidence.kind == "flow-collapse"
                    else evidence.contact_path_index)
                marker_tracker.seed_from_confirmation(
                    confirmation, confirmation_first_index)
            previous = current
    finally:
        # A camera/flow failure near the table must not strand the arm at the
        # lowest command. Retrace upward; do not form a low-pose prepose edge.
        if entered_touch_path:
            retreat_to_trial_start(
                mover, safety, path, current_path_index, settle_s=0.5)

    if contact_z is None:
        return {"state": "no-contact", "minimum_z_mm": minimum_z_m * 1000.0,
                "records": [asdict(record) for record in records]}
    return {"state": "contact", "z_table_mm": contact_z,
            "fk_z_table_mm": contact_fk_z,
            "onset_z_mm": round(float(onset_record[0] * 1000.0), 6),
            "evidence_kind": evidence.kind,
            "records": [asdict(record) for record in records]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--x-mm", type=float, default=TOUCH_X_M * 1000.0,
                        help="clear sagittal table-touch distance from base")
    parser.add_argument(
        "--min-z-mm", type=float, default=MIN_COMMAND_Z_M * 1000.0,
        help="deepest commanded FK height; below -4 mm uses 1 mm steps")
    parser.add_argument(
        "--replay", type=Path,
        help="reprocess saved records offline and write confirmed calibration")
    parser.add_argument(
        "--confirm-only-around-mm", type=float,
        help="short 1 mm pass from z+10 through at most z-2")
    args = parser.parse_args()
    if args.replay is not None:
        if args.run or args.confirm_only_around_mm is not None:
            parser.error("--replay cannot be combined with hardware modes")
        result = replay_touch_file(args.replay)
        summary = {key: value for key, value in result.items()
                   if key != "records"}
        print(f"[table-touch] REPLAY {summary}")
        if result["state"] == "contact":
            print(f"[table-touch] saved {OUTPUT}")
        return

    start_z_m = START_Z_M
    minimum_z_m = args.min_z_mm / 1000.0
    step_z_m = STEP_Z_M
    if args.confirm_only_around_mm is not None:
        centre_mm = float(args.confirm_only_around_mm)
        start_z_m = (centre_mm + 10.0) / 1000.0
        minimum_z_m = (centre_mm - 2.0) / 1000.0
        step_z_m = FINE_STEP_Z_M
    result = run_touch_trial(ArmSessionClient(), execute=args.run,
                             touch_x_m=args.x_mm / 1000.0,
                             minimum_z_m=minimum_z_m,
                             start_z_m=start_z_m,
                             step_z_m=step_z_m)
    if args.run:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[table-touch] saved {OUTPUT}")
    else:
        print(f"[table-touch] DRY RUN {result}")


if __name__ == "__main__":
    main()
