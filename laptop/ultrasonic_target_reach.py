"""Vision bearing + ultrasonic depth closed-loop approach.

The existing :mod:`look_reach` controller already locks a portable object and
keeps motors 2/3/4 pointing at it.  Its missing observation was distance along
that bearing.  This controller adds exactly that scalar; it does not estimate a
table plane or require an absolute sonar/world transform.

Every physical step is:

1. reacquire the same visual instance and keep it near the camera/sonar axis;
2. take one short ultrasonic observation without blocking far-field motion;
3. plan one adaptive 20/15 mm motor-2/3/4 step with the existing controller;
4. collision-check the complete swept transition;
5. send one firmware-slewed move and immediately repeat.

The sonar face is physically behind the finger contact plane.  With an object
inserted as deeply as the fingers permit, 180 live echoes formed a dominant
stable cluster at 36 mm.  The operational stop therefore retains 10 mm margin
and stops at 46 mm.  The only normal approach stops are that measured sonar
threshold and 10 mm fingertip-to-floor clearance.  Vision keeps steering the
same object but never ends an otherwise valid approach.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import copy

import cv2
import numpy as np

import arm_fk
import config
from arm_safety import PhysicalArmSafety
from arm_session import ArmSessionClient
from decision_signal import DecisionMailbox
from floor_grasp import WristSceneDetector
from floor_servo import FloorServo, _fresh_frame
from look_reach import (
    LookReachTargetSelector,
    acquire_initial_target,
    pose_at_reach,
    plan_resolved_step,
)
from ultrasonic_depth import acquire_profile
from vision_segment import ObjectDetection


ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "data" / "vision" / "ultrasonic_target_reach_latest.jpg"

# Camera and HC-SR04 are mounted as one bracket.  The optical/acoustic boresight
# is near image centre after the operator's physical alignment.
SONAR_AIM_X_RATIO = 0.50
# The sonar sits below the camera. A target on the acoustic axis therefore
# projects below optical image centre at grasp distance. The measured deepest
# insertion frames place it around 65% image height, not 50%.
SONAR_AIM_Y_RATIO = 0.65
MAX_AIM_X_ERROR_PX = 90.0

# The deepest physically inserted object produced a stable 36 mm dominant echo
# cluster (180/180 valid samples, 1 mm MAD). Stop 10 mm before that plane.
MEASURED_DEEPEST_OBJECT_RANGE_MM = 36.0
SONAR_STOP_MARGIN_MM = 10.0
STOP_RANGE_MM = MEASURED_DEEPEST_OBJECT_RANGE_MM + SONAR_STOP_MARGIN_MM
FINGERTIP_FLOOR_STOP_MM = 10.0
FAR_ROW_GAP_PX = 180.0
SONAR_NEAR_ROW_GAP_PX = 110.0
FINAL_JAW_GAP_MIN_PX = -35.0
FINAL_JAW_GAP_MAX_PX = 90.0
FINAL_JAW_HORIZONTAL_FRACTION = 0.30
TRACKING_PX_PER_WRIST_DEG = -6.0
TRACKING_MAX_WRIST_STEP_DEG = 10
PRE_CLOSE_LIFT_MM = 12.0
# One servo-quantized fixed-reach lift from the measured 126 px pre-close view
# raises the fingertip from 20.8 to about 28.2 mm, matching the successful
# object-height band without translating toward the object.
PRE_CLOSE_FINE_LIFT_MM = 3.0
VERIFY_LIFT_MM = 25.0
RETAINED_BOTTOM_RATIO = 0.90
RETAINED_HORIZONTAL_RATIO = 0.20
LOADED_HOME_REASSERT_TEMPLATE = [90, 90, 90, 150, 180, 170]
# This is the physically reproduced start of the successful floor approach.
# Unlike HOME with only motor 4 lowered, its camera ray and open fingers both
# face the forward work surface while retaining generous body clearance.
APPROACH_OBSERVATION_TEMPLATE = [90, 107, 84, 178, 90, 170]
# The reproduced coupled descent reaches 18 mm inward drift at 12 mm fingertip
# height before its clearance lift. The bad independent-wrist branch pulled it
# inward by 46 mm in one step and 166 mm cumulatively. Preserve the measured
# floor-approach family while excluding that fold.
MAX_APPROACH_INWARD_DRIFT_MM = 25.0
SEARCH_WRIST_SEQUENCE = (140, 150, 160, 170, 180)
SEARCH_SELECTION_MIN_WRIST = 170
VIVID_MIN_SATURATION = 70
VIVID_MIN_VALUE = 45
VIVID_MIN_AREA_RATIO = 0.0004
VIVID_MAX_AREA_RATIO = 0.04
VIVID_SEARCH_TOP_RATIO = 0.42
VIVID_SEARCH_BOTTOM_RATIO = 0.90
VIVID_SEARCH_LEFT_RATIO = 0.25
VIVID_SEARCH_RIGHT_RATIO = 0.75
FAR_ADVANCE_MM = 20.0
MID_ADVANCE_MM = 15.0
APPROACH_MAX_JOINT_STEP_DEG = 12
# The explicit fingertip-floor gate below is authoritative. This lower planner
# bound prevents an unrelated tool-centre clamp from ending the approach early.
MIN_TOOL_CENTER_Z_M = 0.010
MAX_STEPS = 24
DEFAULT_DECISION_WAIT_S = 8.0


@dataclass(frozen=True)
class RangeDecision:
    action: str
    reason: str


def approach_stop_decision(distance_mm, fingertip_floor_clearance_mm,
                           stop_range_mm=STOP_RANGE_MM,
                           floor_stop_mm=FINGERTIP_FLOOR_STOP_MM):
    """Return one of the two normal approach stops, otherwise continue."""
    clearance = float(fingertip_floor_clearance_mm)
    distance = float(distance_mm)
    if clearance <= float(floor_stop_mm):
        return RangeDecision(
            "floor",
            f"fingertip-floor clearance {clearance:.1f} mm <= "
            f"{float(floor_stop_mm):.1f} mm")
    if distance <= float(stop_range_mm):
        return RangeDecision(
            "sonar",
            f"sonar {distance:.1f} mm <= {float(stop_range_mm):.1f} mm")
    return RangeDecision(
        "continue",
        f"sonar {distance:.1f} mm; floor clearance {clearance:.1f} mm")


def adaptive_advance_mm(row_gap_px):
    """Make servo-scale moves; never request the ineffective old 5 mm step."""
    gap = float(row_gap_px)
    if gap > FAR_ROW_GAP_PX:
        return FAR_ADVANCE_MM
    return MID_ADVANCE_MM


def tracking_wrist_target(current_wrist, vertical_error_px):
    """Use the measured motor-4 pixel response, not a tiny generic IK term."""
    desired_pixel_delta = -float(vertical_error_px)
    servo_delta = int(round(np.clip(
        desired_pixel_delta / TRACKING_PX_PER_WRIST_DEG,
        -TRACKING_MAX_WRIST_STEP_DEG,
        TRACKING_MAX_WRIST_STEP_DEG,
    )))
    return int(np.clip(
        int(current_wrist) + servo_delta,
        config.SERVO_MIN[config.J_WRIST],
        config.SERVO_MAX[config.J_WRIST],
    ))


def fingertip_floor_clearance_mm(pose, table_z_m=0.0):
    """Physical distal endpoint height over the shared table plane."""
    return (
        float(arm_fk.geometry(pose).finger_tip[2]) - float(table_z_m)
    ) * 1000.0


def transition_fingertip_floor_clearance_mm(
        start_pose, end_pose, table_z_m=0.0, step_deg=0.5):
    """Minimum fingertip clearance over the complete interpolated servo slew."""
    start = np.asarray(start_pose, dtype=float)
    end = np.asarray(end_pose, dtype=float)
    span = float(np.max(np.abs(end - start)))
    samples = max(2, int(np.ceil(span / float(step_deg))) + 1)
    return min(
        fingertip_floor_clearance_mm(pose, table_z_m)
        for pose in np.linspace(start, end, samples)
    )


def _reacquire(detector, selector, attempts=3):
    frame = scene = candidate = None
    for _ in range(attempts):
        frame = _fresh_frame(discard=1)
        scene, _observation = detector.scene(frame)
        candidate = selector.match(scene)
        if candidate is not None:
            return frame, scene, candidate
    return frame, scene, None


def _observe_sonar(client, near):
    """Take one bounded observation; never hold far-field motion for sonar."""
    if near:
        profile = acquire_profile(
            client, samples=12, interval_s=0.035,
            min_valid_fraction=0.60, min_support_fraction=0.55,
            min_cluster_samples=6)
    else:
        profile = acquire_profile(
            client, samples=5, interval_s=0.020,
            min_valid_fraction=0.40, min_support_fraction=0.60,
            min_cluster_samples=3)
    if profile.stable and profile.distance_mm is not None:
        return float(profile.distance_mm), profile
    return float("inf"), profile


def _fast_approach_move(client, target, settle_s=0.18):
    """Use the Uno's own degree-by-degree slew without extra 3° host chunks."""
    return client.request({
        "command": "move",
        "pose": [int(round(value)) for value in target],
        "require_camera": True,
        "settle_s": float(settle_s),
    })


def _draw_preview(frame, scene, candidate, pose, distance_mm, step, decision,
                  target_visible=True):
    import cv2

    image = frame.copy()
    aim = (int(round(SONAR_AIM_X_RATIO * image.shape[1])),
           int(round(SONAR_AIM_Y_RATIO * image.shape[0])))
    cv2.drawMarker(image, aim, (255, 255, 255),
                   cv2.MARKER_CROSS, 34, 2)
    if target_visible:
        x, y, width, height = candidate.bbox
        cv2.rectangle(
            image, (x, y), (x + width, y + height), (0, 220, 255), 3)
        cv2.line(image, tuple(int(round(value)) for value in candidate.center),
                 aim, (0, 220, 255), 2)
    gripper = getattr(scene, "gripper", None)
    if gripper is not None:
        cv2.drawMarker(
            image, tuple(int(round(value)) for value in gripper.center),
            (255, 255, 0), cv2.MARKER_CROSS, 34, 2)
    text = (
        f"step {step}  sonar {distance_mm:.1f} mm  "
        f"pose {pose[1]}/{pose[2]}/{pose[3]}  {decision.action}"
        f"{'' if target_visible else '  LAST LOCK'}")
    cv2.putText(image, text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(image, text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(PREVIEW), image)


def _jaw_metrics(scene, candidate):
    gripper = getattr(scene, "gripper", None)
    if gripper is None:
        return False, "finger markers unavailable", None
    horizontal = abs(float(candidate.center[0]) - float(gripper.center[0]))
    gap = float(gripper.center[1]) - float(
        candidate.bbox[1] + candidate.bbox[3])
    if horizontal > 0.42 * float(gripper.opening_px):
        return (
            False,
            f"object outside open-jaw corridor ({horizontal:.0f}px)",
            gap,
        )
    if gap > 100.0:
        return False, f"object has not reached finger row (gap {gap:.0f}px)", gap
    return True, f"inside jaws; row gap {gap:.0f}px", gap


def _jaw_gate(scene, candidate):
    ready, reason, _gap = _jaw_metrics(scene, candidate)
    return ready, reason


def _final_grasp_gate(scene, candidate):
    """Require the object body, not merely its direction, between the fingers."""
    gripper = getattr(scene, "gripper", None)
    if gripper is None:
        return False, "final finger markers unavailable"
    horizontal = abs(float(candidate.center[0]) - float(gripper.center[0]))
    allowed = FINAL_JAW_HORIZONTAL_FRACTION * float(gripper.opening_px)
    gap = float(gripper.center[1]) - float(
        candidate.bbox[1] + candidate.bbox[3])
    if horizontal > allowed:
        return False, (
            f"final object offset {horizontal:.0f}px > {allowed:.0f}px")
    if not FINAL_JAW_GAP_MIN_PX <= gap <= FINAL_JAW_GAP_MAX_PX:
        return False, (
            f"final jaw-row gap {gap:.0f}px outside "
            f"{FINAL_JAW_GAP_MIN_PX:.0f}..{FINAL_JAW_GAP_MAX_PX:.0f}px")
    return True, (
        f"final grasp aligned: offset {horizontal:.0f}px, gap {gap:.0f}px")


def _preclose_needs_fine_lift(scene, candidate):
    """True only when lateral aim is valid and the object is still too low."""
    gripper = getattr(scene, "gripper", None)
    if gripper is None or candidate is None:
        return False
    horizontal = abs(float(candidate.center[0]) - float(gripper.center[0]))
    allowed = FINAL_JAW_HORIZONTAL_FRACTION * float(gripper.opening_px)
    gap = float(gripper.center[1]) - float(
        candidate.bbox[1] + candidate.bbox[3])
    return horizontal <= allowed and gap > FINAL_JAW_GAP_MAX_PX


def _best_final_grasp_candidate(scene, locked):
    """Use the deepest valid nested mask of the already locked object.

    FastSAM can emit the same two-colour object as a short coloured-body mask
    and as a complete body+cap mask. Tracking favours centre continuity, which
    is useful in motion but can pick the short mask for the final extent test.
    At close time only, inspect horizontally overlapping masks and retain the
    deepest one that independently passes the strict jaw gate.
    """
    if locked is None:
        return None
    lx, _ly, lwidth, _lheight = locked.bbox
    lleft, lright = float(lx), float(lx + lwidth)
    valid = []
    for item in getattr(scene, "ranked", ()):
        ileft = float(item.bbox[0])
        iright = float(item.bbox[0] + item.bbox[2])
        overlap = max(0.0, min(lright, iright) - max(lleft, ileft))
        shared_fraction = overlap / max(
            1.0, min(lright - lleft, iright - ileft))
        if shared_fraction < 0.60:
            continue
        if abs(float(item.center[0]) - float(locked.center[0])) > 60.0:
            continue
        allowed, _reason = _final_grasp_gate(scene, item)
        if allowed:
            valid.append(item)
    if not valid:
        return locked
    return max(valid, key=lambda item: (
        float(item.bbox[1] + item.bbox[3]),
        float(item.area),
    ))


def _vertical_lift_pose(pose, lift_mm):
    """Raise at fixed forward reach and pitch so the fingers do not sweep."""
    geometry = arm_fk.geometry(pose)
    lifted = pose_at_reach(
        pose,
        float(geometry.tool[0]),
        float(geometry.tool[2]) + float(lift_mm) / 1000.0,
    )
    lifted[config.J_GRIP] = int(pose[config.J_GRIP])
    return lifted


def _retained_image_gate(frame, candidate):
    """Held close-ups remain bottom-clipped and centred in the wrist view."""
    if candidate is None:
        return False, "locked object missing after lift"
    height, width = frame.shape[:2]
    bbox_bottom = float(candidate.bbox[1] + candidate.bbox[3])
    bottom_ratio = bbox_bottom / float(height)
    horizontal = abs(float(candidate.center[0]) - 0.5 * float(width))
    if bottom_ratio < RETAINED_BOTTOM_RATIO:
        return False, (
            f"object bottom ratio {bottom_ratio:.2f} < "
            f"{RETAINED_BOTTOM_RATIO:.2f}")
    if horizontal > RETAINED_HORIZONTAL_RATIO * float(width):
        return False, f"lifted object lateral error {horizontal:.0f}px"
    return True, (
        f"retained object bottom={bottom_ratio:.2f}, "
        f"lateral={horizontal:.0f}px")


def _retained_corridor_candidate(frame, scene):
    """Detect a lifted vivid object continuously filling the closed-jaw gap."""
    marker_boxes = sorted(getattr(scene, "marker_boxes", ()),
                          key=lambda box: box[0])
    if len(marker_boxes) < 2:
        return None
    height, width = frame.shape[:2]
    left = int(marker_boxes[0][0] + marker_boxes[0][2])
    right = int(marker_boxes[-1][0])
    top = int(round(0.30 * height))
    if right - left < 15:
        return None
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.asarray((0, VIVID_MIN_SATURATION, VIVID_MIN_VALUE),
                   dtype=np.uint8),
        np.asarray((179, 255, 255), dtype=np.uint8),
    )
    roi = np.zeros_like(mask)
    roi[top:height, left:right] = 255
    mask = cv2.bitwise_and(mask, roi)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    corridor_area = float((right - left) * (height - top))
    candidates = []
    for label in range(1, count):
        x, y, box_width, box_height, area = (
            int(value) for value in stats[label])
        bottom = y + box_height
        if area < 0.12 * corridor_area:
            continue
        if box_height < 0.25 * height or bottom < 0.95 * height:
            continue
        pixels = hsv[:, :, 1][labels == label]
        values = hsv[:, :, 2][labels == label]
        candidates.append(ObjectDetection(
            center=tuple(float(value) for value in centroids[label]),
            bbox=(x, y, box_width, box_height),
            area=float(area),
            confidence=0.95,
            median_saturation=float(np.median(pixels)),
            median_value=float(np.median(values)),
        ))
    return max(candidates, key=lambda item: item.area) if candidates else None


def home_pose_holding(pose):
    """HOME for every joint except the currently loaded gripper servo."""
    home = list(config.HOME_POSE)
    home[config.J_GRIP] = int(pose[config.J_GRIP])
    return home


def grip_hold_pose(pose):
    """Back off from the empty-close endpoint to measured loaded preload."""
    holding = list(pose)
    holding[config.J_GRIP] = int(config.GRIP_HOLD)
    return holding


def loaded_home_reassert_pose(pose):
    """Distinct safe waypoint that forces a physical HOME pulse trajectory."""
    waypoint = list(LOADED_HOME_REASSERT_TEMPLATE)
    waypoint[config.J_GRIP] = int(pose[config.J_GRIP])
    return waypoint


def open_ready_pose(pose):
    """Preserve the observation pose while fully opening the fingers."""
    ready = list(pose)
    ready[config.J_GRIP] = config.GRIP_OPEN
    return ready


def approach_observation_pose(pose):
    """Known safe forward/down observation pose, preserving an open gripper."""
    target = list(APPROACH_OBSERVATION_TEMPLATE)
    target[config.J_GRIP] = int(pose[config.J_GRIP])
    return target


def fingertip_forward_x_mm(pose):
    """Planar distance of the physical finger endpoint from the base axis."""
    return float(arm_fk.geometry(pose).finger_tip[0]) * 1000.0


def approach_stays_forward(start_x_mm, candidate_pose,
                           max_inward_mm=MAX_APPROACH_INWARD_DRIFT_MM):
    """The approach may descend, but must not fold back into the base."""
    return fingertip_forward_x_mm(candidate_pose) >= (
        float(start_x_mm) - float(max_inward_mm))


def _candidate_on_sonar_axis(candidate, frame_width):
    if candidate is None:
        return False
    aim_x = SONAR_AIM_X_RATIO * float(frame_width)
    return abs(float(candidate.center[0]) - aim_x) <= MAX_AIM_X_ERROR_PX


def _axis_scene(scene, frame_width):
    """Shallow scene view containing only fixed-base reachable bearings."""
    filtered = copy.copy(scene)
    filtered.ranked = [
        candidate for candidate in scene.ranked
        if _candidate_on_sonar_axis(candidate, frame_width)
    ]
    return filtered


def _choose_on_axis(scene, selector, pose, frame_width):
    """Choose from every segmented on-axis object, independent of its colour."""
    _clear_target_lock(selector)
    return selector.choose(
        _axis_scene(scene, frame_width), pose=pose)


def select_after_external_decisions(
        scene, selector, pose, frame_width, decisions):
    """Apply external decisions and return the active fixed-base candidate.

    Rejections are stored by the existing persistent selector. When every
    visible candidate has been rejected, the veto stack is reset and selection
    restarts from the first candidate, matching the simulation semantics.
    """
    selectable = _axis_scene(scene, frame_width)
    candidate = selector.current
    if (candidate is None
            or not _candidate_on_sonar_axis(candidate, frame_width)):
        candidate = selector.choose(selectable, pose=pose)
    reset = False
    applied = 0
    for decision in decisions:
        decision = str(decision).lower()
        if decision == "accept":
            break
        if decision != "reject":
            raise ValueError(f"unsupported target decision: {decision}")
        applied += 1
        candidate = selector.reject_current(selectable, pose=pose)
        if candidate is None:
            selector.selector.reset()
            candidate = selector.choose(selectable, pose=pose)
            reset = True
    return {
        "candidate": candidate,
        "rejectionsApplied": applied,
        "cycleReset": reset,
        "candidateCount": len(selectable.ranked),
    }


def _await_target_decision(
        mailbox, cursor, wait_s, scene, selector, pose, frame_width):
    """Wait for one fresh manual/dashboard/ErrP decision for this proposal."""
    if mailbox is None or float(wait_s) <= 0:
        return selector.current, cursor, False
    decision = mailbox.wait_after(cursor, wait_s)
    if decision is None:
        print(
            f"[sonar-reach] decision timeout after {float(wait_s):.1f}s; "
            "continuing with proposed target",
            flush=True,
        )
        return selector.current, cursor, False
    result = select_after_external_decisions(
        scene, selector, pose, frame_width, [decision.decision])
    print(
        f"[sonar-reach] external {decision.decision.upper()} "
        f"seq={decision.sequence} source={decision.source}; "
        f"candidates={result['candidateCount']} "
        f"cycle-reset={result['cycleReset']}",
        flush=True,
    )
    return result["candidate"], decision.sequence, decision.decision == "reject"


def resolve_decision_mailbox(value):
    """Normalize the CLI flag without treating ``False`` as a mailbox."""
    if value is True:
        return DecisionMailbox()
    if value is False:
        return None
    return value


def _clear_target_lock(selector):
    """Discard a background lock while retaining explicit ErrP vetoes."""
    selector.current = None
    selector.lock = None


def _box_overlap_fraction(box, obstacle, padding=12):
    """Fraction of ``box`` covered by a padded finger-marker box."""
    x, y, width, height = box
    ox, oy, owidth, oheight = obstacle
    ox -= padding
    oy -= padding
    owidth += 2 * padding
    oheight += 2 * padding
    left, top = max(x, ox), max(y, oy)
    right = min(x + width, ox + owidth)
    bottom = min(y + height, oy + oheight)
    intersection = max(0, right - left) * max(0, bottom - top)
    return intersection / float(max(1, width * height))


def vivid_table_candidates(frame, marker_boxes=()):
    """Find compact, vividly distinct tabletop objects without a hue preset.

    FastSAM occasionally omits a narrow real object entirely.  This fallback
    finds sufficiently saturated connected components only in the near-table
    search band, removes the coloured finger markers, and ranks surviving
    components by proximity to the camera/sonar axis.  It intentionally does
    not encode blue, yellow, or any other target colour.
    """
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.asarray((0, VIVID_MIN_SATURATION, VIVID_MIN_VALUE), dtype=np.uint8),
        np.asarray((179, 255, 255), dtype=np.uint8),
    )
    roi = np.zeros_like(mask)
    top = int(round(VIVID_SEARCH_TOP_RATIO * height))
    bottom = int(round(VIVID_SEARCH_BOTTOM_RATIO * height))
    left = int(round(VIVID_SEARCH_LEFT_RATIO * width))
    right = int(round(VIVID_SEARCH_RIGHT_RATIO * width))
    roi[top:bottom, left:right] = 255
    mask = cv2.bitwise_and(mask, roi)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    frame_area = float(width * height)
    minimum_area = VIVID_MIN_AREA_RATIO * frame_area
    maximum_area = VIVID_MAX_AREA_RATIO * frame_area
    aim_x = SONAR_AIM_X_RATIO * float(width)
    detections = []
    for label in range(1, count):
        x, y, box_width, box_height, area = (
            int(value) for value in stats[label])
        if not minimum_area <= area <= maximum_area:
            continue
        if box_width < 10 or box_height < 10:
            continue
        if y + box_height >= bottom:
            continue
        bbox = (x, y, box_width, box_height)
        if any(_box_overlap_fraction(bbox, marker) > 0.15
               for marker in marker_boxes):
            continue
        pixels = hsv[:, :, 1][labels == label]
        values = hsv[:, :, 2][labels == label]
        center = tuple(float(value) for value in centroids[label])
        detections.append(ObjectDetection(
            center=center,
            bbox=bbox,
            area=float(area),
            confidence=0.97,
            median_saturation=float(np.median(pixels)),
            median_value=float(np.median(values)),
        ))
    detections.sort(key=lambda item: (
        abs(float(item.center[0]) - aim_x),
        -float(item.center[1]),
        -float(item.area),
    ))
    return detections


class VividFallbackDetector:
    """Add generic colour-connected candidates to every FastSAM scene."""

    def __init__(self, primary):
        self.primary = primary

    def scene(self, frame):
        scene, observation = self.primary.scene(frame)
        vivid = vivid_table_candidates(frame, scene.marker_boxes)
        for candidate in vivid:
            if any(
                    abs(candidate.center[0] - existing.center[0]) < 12
                    and abs(candidate.center[1] - existing.center[1]) < 12
                    for existing in scene.ranked):
                continue
            scene.ranked.append(candidate)
        return scene, observation


def _choose_vivid_on_axis(frame, scene, selector, pose):
    """Prefer a verified on-axis vivid component after a model miss."""
    candidates = [
        item for item in vivid_table_candidates(frame, scene.marker_boxes)
        if _candidate_on_sonar_axis(item, frame.shape[1])
    ]
    if not candidates:
        return None
    _clear_target_lock(selector)
    scene.ranked = candidates + [
        item for item in scene.ranked
        if all(
            abs(item.center[0] - vivid.center[0]) >= 12
            or abs(item.center[1] - vivid.center[1]) >= 12
            for vivid in candidates)
    ]
    return selector.choose(scene, pose=pose)


def _open_and_find_target(
        client, mover, safety, detector, selector, pose, execute):
    """Open at HOME, then lower the wrist camera through a bounded search."""
    ready = open_ready_pose(pose)
    if ready != pose:
        report = safety.transition_report(pose, ready)
        if not report.safe:
            raise RuntimeError("initial open rejected: " + report.explain())
        if execute:
            mover.slow_move(ready, final_settle=0.45)
        pose = ready

    frame = scene = candidate = None
    if not execute:
        frame, scene, candidate = acquire_initial_target(
            detector, target_selector=selector, pose=pose)
        return pose, frame, scene, candidate

    current_wrist = int(pose[config.J_WRIST])
    search_sequence = (current_wrist,) + tuple(
        wrist for wrist in SEARCH_WRIST_SEQUENCE if wrist != current_wrist)
    for wrist in search_sequence:
        search_pose = list(pose)
        search_pose[config.J_WRIST] = int(wrist)
        if search_pose != pose:
            report = safety.transition_report(pose, search_pose)
            if not report.safe:
                continue
            print(
                f"[sonar-reach] SEARCH wrist {pose[config.J_WRIST]}"
                f"->{search_pose[config.J_WRIST]}",
                flush=True,
            )
            _fast_approach_move(client, search_pose, settle_s=0.35)
            pose = search_pose
        if wrist < SEARCH_SELECTION_MIN_WRIST:
            continue
        _clear_target_lock(selector)
        frame, scene, candidate = acquire_initial_target(
            detector, target_selector=selector, pose=pose)
        if _candidate_on_sonar_axis(candidate, frame.shape[1]):
            return pose, frame, scene, candidate
        if candidate is not None:
            print(
                f"[sonar-reach] SEARCH rejected off-axis lock "
                f"x={candidate.center[0]:.0f}px; lowering camera",
                flush=True,
            )
            _clear_target_lock(selector)
        candidate = _choose_on_axis(
            scene, selector, pose, frame.shape[1])
        if _candidate_on_sonar_axis(candidate, frame.shape[1]):
            return pose, frame, scene, candidate
        candidate = _choose_vivid_on_axis(
            frame, scene, selector, pose)
        if _candidate_on_sonar_axis(candidate, frame.shape[1]):
            print(
                f"[sonar-reach] SEARCH vivid fallback locked "
                f"center=({candidate.center[0]:.0f},"
                f"{candidate.center[1]:.0f})",
                flush=True,
            )
            return pose, frame, scene, candidate
    return pose, frame, scene, None


def _enter_forward_observation(
        client, safety, detector, selector, pose, execute):
    """Stage the arm in the reproduced forward approach branch and relock."""
    target = approach_observation_pose(pose)
    report = safety.transition_report(pose, target)
    if not report.safe:
        raise RuntimeError(
            "forward observation transition rejected: " + report.explain())
    if execute and target != pose:
        print(
            f"[sonar-reach] FORWARD observation {pose[1:4]} "
            f"->{target[1:4]}; clearance="
            f"{report.minimum_clearance_mm:.1f}mm",
            flush=True,
        )
        _fast_approach_move(client, target, settle_s=0.50)
    pose = target
    _clear_target_lock(selector)
    frame, scene, candidate = acquire_initial_target(
        detector, target_selector=selector, pose=pose)
    if not _candidate_on_sonar_axis(candidate, frame.shape[1]):
        candidate = _choose_on_axis(
            scene, selector, pose, frame.shape[1])
    if not _candidate_on_sonar_axis(candidate, frame.shape[1]):
        candidate = _choose_vivid_on_axis(
            frame, scene, selector, pose)
    if not _candidate_on_sonar_axis(candidate, frame.shape[1]):
        _clear_target_lock(selector)
        return pose, frame, scene, None
    return pose, frame, scene, candidate


def _return_home_holding(client, mover, safety, pose):
    """Transport a retained object to HOME without ever opening the gripper."""
    if int(pose[config.J_GRIP]) not in (
            int(config.GRIP_CLOSED), int(config.GRIP_HOLD)):
        raise RuntimeError("HOME transport requires a closed/holding gripper")
    waypoint = loaded_home_reassert_pose(pose)
    home = home_pose_holding(pose)
    for start, target, label in (
            (pose, waypoint, "loaded HOME reassert"),
            (waypoint, home, "loaded HOME")):
        report = safety.transition_report(start, target)
        if not report.safe:
            raise RuntimeError(
                f"{label} transition rejected: " + report.explain())
        print(
            f"[sonar-reach] {label} {start} -> {target}; "
            f"clearance={report.minimum_clearance_mm:.1f}mm",
            flush=True,
        )
        mover.slow_move(target, final_settle=0.8)
    return home


def _clearance_grasp_and_verify(
        client, mover, safety, detector, selector, pose, distance_mm):
    """Lift open fingers off the table, close, then verify a retained lift."""
    preclose = _vertical_lift_pose(pose, PRE_CLOSE_LIFT_MM)
    report = safety.transition_report(pose, preclose)
    if not report.safe:
        raise RuntimeError(
            "pre-close clearance lift rejected: " + report.explain())
    mover.slow_move(preclose, final_settle=0.45)

    _frame, scene, candidate = _reacquire(detector, selector, attempts=4)
    if candidate is None:
        return {
            "state": "preclose-target-lost", "pose": preclose,
            "distance_mm": distance_mm,
        }
    candidate = _best_final_grasp_candidate(scene, candidate)
    if _preclose_needs_fine_lift(scene, candidate):
        adjusted = _vertical_lift_pose(preclose, PRE_CLOSE_FINE_LIFT_MM)
        report = safety.transition_report(preclose, adjusted)
        if not report.safe:
            raise RuntimeError(
                "pre-close fine lift rejected: " + report.explain())
        print(
            f"[sonar-reach] pre-close object remains below finger row; "
            f"fixed-reach fine lift {preclose[1:4]}->{adjusted[1:4]}",
            flush=True,
        )
        mover.slow_move(adjusted, final_settle=0.40)
        preclose = adjusted
        _frame, scene, candidate = _reacquire(
            detector, selector, attempts=4)
        if candidate is None:
            return {
                "state": "fine-lift-target-lost", "pose": preclose,
                "distance_mm": distance_mm,
            }
        candidate = _best_final_grasp_candidate(scene, candidate)
    aligned, reason = _final_grasp_gate(scene, candidate)
    print(f"[sonar-reach] after clearance lift: {reason}", flush=True)
    if not aligned:
        return {
            "state": "preclose-alignment-failed", "pose": preclose,
            "distance_mm": distance_mm, "reason": reason,
        }
    reference_center = np.asarray(candidate.center, dtype=float)

    closed = list(preclose)
    closed[config.J_GRIP] = config.GRIP_CLOSED
    report = safety.transition_report(preclose, closed)
    if not report.safe:
        raise RuntimeError("close rejected: " + report.explain())
    mover.slow_move(closed, final_settle=0.55)

    verified = _vertical_lift_pose(closed, VERIFY_LIFT_MM)
    report = safety.transition_report(closed, verified)
    if not report.safe:
        raise RuntimeError(
            "verification lift rejected: " + report.explain())
    mover.slow_move(verified, final_settle=0.65)

    retained_frame, _scene, retained = _reacquire(
        detector, selector, attempts=4)
    if retained is None:
        retained = _retained_corridor_candidate(retained_frame, _scene)
        if retained is not None:
            print(
                "[sonar-reach] retention recovered from continuous "
                "closed-jaw corridor evidence",
                flush=True,
            )
    shift = None
    if retained is not None:
        shift = float(np.linalg.norm(
            np.asarray(retained.center, dtype=float) - reference_center))
    retained_ok, retained_reason = _retained_image_gate(
        retained_frame, retained)
    print(
        f"[sonar-reach] verification lift target shift="
        f"{'missing' if shift is None else f'{shift:.1f}px'} "
        f"retained={retained_ok} ({retained_reason})",
        flush=True,
    )
    transport_pose = verified
    if retained_ok:
        holding = grip_hold_pose(verified)
        report = safety.transition_report(verified, holding)
        if not report.safe:
            raise RuntimeError(
                "loaded gripper backoff rejected: " + report.explain())
        print(
            f"[sonar-reach] loaded gripper backoff "
            f"{config.GRIP_CLOSED}->{config.GRIP_HOLD}deg",
            flush=True,
        )
        mover.slow_move(holding, final_settle=0.45)
        hold_frame = _fresh_frame(discard=2)
        hold_scene, _observation = detector.scene(hold_frame)
        held = _retained_corridor_candidate(hold_frame, hold_scene)
        hold_ok, hold_reason = _retained_image_gate(hold_frame, held)
        print(
            f"[sonar-reach] low-stall hold retained={hold_ok} "
            f"({hold_reason})",
            flush=True,
        )
        if not hold_ok:
            return {
                "state": "hold-backoff-unverified", "pose": holding,
                "distance_mm": distance_mm,
                "retained_reason": hold_reason,
            }
        transport_pose = holding
    final_pose = (
        _return_home_holding(client, mover, safety, transport_pose)
        if retained_ok else verified)
    return {
        "state": "home-with-object" if retained_ok else "closed-unverified",
        "pose": final_pose, "distance_mm": distance_mm,
        "retained_shift_px": shift, "retained_reason": retained_reason,
    }


def run(client=None, execute=False, allow_grasp=False, max_steps=MAX_STEPS,
        detector=None, selector=None, decision_mailbox=None,
        decision_wait_s=DEFAULT_DECISION_WAIT_S):
    client = client or ArmSessionClient()
    detector = detector or WristSceneDetector()
    if not isinstance(detector, VividFallbackDetector):
        detector = VividFallbackDetector(detector)
    selector = selector or LookReachTargetSelector()
    decision_mailbox = resolve_decision_mailbox(decision_mailbox)
    decision_cursor = (
        decision_mailbox.cursor() if decision_mailbox is not None else 0)
    safety = PhysicalArmSafety()
    mover = FloorServo(client, calib=None)
    pose = list(client.request({"command": "status"})["pose"])

    pose, frame, scene, candidate = _open_and_find_target(
        client, mover, safety, detector, selector, pose, execute)
    if candidate is None:
        return {"state": "no-target", "moved": False}
    pose, frame, scene, candidate = _enter_forward_observation(
        client, safety, detector, selector, pose, execute)
    if candidate is None:
        return {
            "state": "target-not-visible-from-forward-observation",
            "pose": pose,
        }
    candidate, decision_cursor, rejected = _await_target_decision(
        decision_mailbox, decision_cursor, decision_wait_s,
        scene, selector, pose, frame.shape[1])
    while rejected:
        if candidate is None:
            return {
                "state": "no-fixed-base-candidate-after-reject",
                "pose": pose,
            }
        print(
            f"[sonar-reach] NEXT target proposed at "
            f"({candidate.center[0]:.0f},{candidate.center[1]:.0f}); "
            "waiting for external decision",
            flush=True,
        )
        candidate, decision_cursor, rejected = _await_target_decision(
            decision_mailbox, decision_cursor, decision_wait_s,
            scene, selector, pose, frame.shape[1])
    if candidate is None:
        return {"state": "no-fixed-base-candidate", "pose": pose}
    jaw_ready, jaw_reason, row_gap = _jaw_metrics(scene, candidate)
    if row_gap is None:
        raise RuntimeError(jaw_reason)
    last_visible_pose = list(pose)
    approach_start_x_mm = fingertip_forward_x_mm(pose)

    for step in range(int(max_steps)):
        frame, observed_scene, observed_candidate = _reacquire(
            detector, selector)
        target_visible = observed_candidate is not None
        if target_visible:
            scene, candidate = observed_scene, observed_candidate
        else:
            pose = list(client.request({"command": "status"})["pose"])
            if not execute:
                return {
                    "state": "target-recovery-needed", "pose": pose,
                    "preview": str(PREVIEW),
                }
            report = safety.transition_report(pose, last_visible_pose)
            if not report.safe:
                raise RuntimeError(
                    "target recovery rejected: " + report.explain())
            print(
                f"[sonar-reach] target left frame; returning "
                f"{pose[1:4]} -> {last_visible_pose[1:4]} for re-aim",
                flush=True,
            )
            _fast_approach_move(client, last_visible_pose, settle_s=0.30)
            continue
        pose = list(client.request({"command": "status"})["pose"])
        last_visible_pose = list(pose)
        aim_x = SONAR_AIM_X_RATIO * frame.shape[1]
        aim_y = SONAR_AIM_Y_RATIO * frame.shape[0]
        x_error = float(candidate.center[0]) - aim_x
        if abs(x_error) > MAX_AIM_X_ERROR_PX:
            raise RuntimeError(
                f"target {x_error:+.0f}px from sonar x axis; "
                "base/lateral alignment required")

        jaw_ready, jaw_reason, row_gap = _jaw_metrics(scene, candidate)
        if row_gap is None:
            raise RuntimeError(jaw_reason)
        sonar_near = float(row_gap) <= SONAR_NEAR_ROW_GAP_PX
        distance, profile = _observe_sonar(client, near=sonar_near)
        floor_clearance = fingertip_floor_clearance_mm(pose)
        decision = approach_stop_decision(distance, floor_clearance)
        _draw_preview(
            frame, scene, candidate, pose, distance, step, decision,
            target_visible=target_visible)
        print(
            f"[sonar-reach] step={step:02d} target="
            f"({candidate.center[0]:.0f},{candidate.center[1]:.0f}) "
            f"visible={target_visible} "
            f"range={distance:.1f}mm sonar-stable={profile.stable} "
            f"sonar-mode={'near' if sonar_near else 'quick'} "
            f"decision={decision.action} "
            f"jaw={jaw_reason}", flush=True)

        if decision.action == "floor":
            if execute and allow_grasp:
                return _clearance_grasp_and_verify(
                    client, mover, safety, detector, selector, pose, distance)
            return {
                "state": "floor-clearance-stop", "pose": pose,
                "distance_mm": distance,
                "fingertip_floor_clearance_mm": floor_clearance,
                "preview": str(PREVIEW),
            }
        if decision.action == "sonar":
            final_aligned, final_reason = _final_grasp_gate(scene, candidate)
            print(f"[sonar-reach] {final_reason}", flush=True)
            if execute and allow_grasp:
                return _clearance_grasp_and_verify(
                    client, mover, safety, detector, selector, pose, distance)
            if not execute or not allow_grasp or not final_aligned:
                return {
                    "state": (
                        "sonar-stop-ready" if final_aligned
                        else "sonar-stop-image-disagrees"),
                    "pose": pose, "distance_mm": distance,
                    "fingertip_floor_clearance_mm": floor_clearance,
                    "preview": str(PREVIEW),
                }
        vertical_error = float(candidate.center[1]) - aim_y
        advance_mm = adaptive_advance_mm(row_gap)
        plan = plan_resolved_step(
            pose, vertical_error, frame.shape[0],
            advance_mm=advance_mm,
            min_tool_z_m=MIN_TOOL_CENTER_Z_M,
            max_joint_step=APPROACH_MAX_JOINT_STEP_DEG)
        if plan is None:
            raise RuntimeError("no bounded 2/3/4 step remains")

        # ``plan_resolved_step`` solves motors 2/3/4 together. Never overwrite
        # motor 4 afterward: on this arm it rotates the entire distal hand, so a
        # camera-only correction would also pull the fingers toward the base.
        next_pose = list(plan["pose"])
        swept_floor_clearance = transition_fingertip_floor_clearance_mm(
            pose, next_pose)
        if swept_floor_clearance <= FINGERTIP_FLOOR_STOP_MM:
            if execute and allow_grasp:
                print(
                    "[sonar-reach] next approach would enter the 10mm floor "
                    "guard; grasping from the current safe pose",
                    flush=True,
                )
                return _clearance_grasp_and_verify(
                    client, mover, safety, detector, selector, pose, distance)
            return {
                "state": "floor-clearance-stop", "pose": pose,
                "distance_mm": distance,
                "fingertip_floor_clearance_mm": swept_floor_clearance,
                "preview": str(PREVIEW),
            }
        if not approach_stays_forward(approach_start_x_mm, next_pose):
            raise RuntimeError(
                "approach rejected: candidate would fold the fingertip "
                f"inward to {fingertip_forward_x_mm(next_pose):.1f}mm "
                f"from forward start {approach_start_x_mm:.1f}mm")
        report = safety.transition_report(pose, next_pose)
        if not report.safe:
            raise RuntimeError(
                "approach rejected: " + report.explain())
        print(
            f"[sonar-reach] pose234 {pose[1:4]} -> {next_pose[1:4]} "
            f"commanded-advance={adaptive_advance_mm(row_gap):.0f}mm "
            f"visual-progress={plan['progress_mm']:.2f}mm "
            f"clearance={report.minimum_clearance_mm:.1f}mm", flush=True)
        if not execute:
            return {
                "state": "planned", "pose": next_pose,
                "distance_mm": distance, "preview": str(PREVIEW)}
        _fast_approach_move(client, next_pose)

    return {
        "state": "step-limit",
        "pose": client.request({"command": "status"})["pose"],
        "preview": str(PREVIEW),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--grasp", action="store_true")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument(
        "--decision-mailbox", action="store_true",
        help="listen for accept/reject decisions from decision_signal.py")
    parser.add_argument(
        "--decision-wait", type=float, default=DEFAULT_DECISION_WAIT_S,
        help="seconds to wait for a fresh decision at each target proposal")
    args = parser.parse_args()
    if args.grasp and not args.run:
        parser.error("--grasp requires --run")
    result = run(
        execute=args.run, allow_grasp=args.grasp, max_steps=args.max_steps,
        decision_mailbox=args.decision_mailbox,
        decision_wait_s=args.decision_wait)
    print(f"[sonar-reach] RESULT {result}")


if __name__ == "__main__":
    main()
