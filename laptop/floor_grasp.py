"""Multi-object floor grasp with an ErrP-ready reject that switches objects.

This is the controller for three shared-autonomy milestones:

  1. Recognize an object with the wrist camera and grasp it on request.
  2. Recognize *several* objects at once, rank them, and grasp the chosen one.
  3. When the human signals "not that one" (reject), drop the current object and
     move on to the next candidate -- no restart, same live loop.

Today the reject signal is a keyboard key in the live window. Tomorrow the
identical ``CandidateSelector.reject`` call is driven by an ErrP decision, so the
robot side does not change when the brain signal is wired in.

Perception is background- and color-independent: FastSAM proposes every visible
instance and small deterministic selectors drop the two finger tapes and the
floor/arm before ranking portable objects. Reaching reuses the physically
reproduced ``floor_motion`` curve; horizontal alignment is a bounded visual
servo along that curve. All physical descent/close/lift stays behind
``config.FLOOR_GRASP_EXECUTE_VERIFIED`` and every failure falls back to OPEN ->
HOVER -> STOP. It never opens the Uno directly: motion goes through the
persistent ``ArmSessionClient`` so the board is never reset mid-task.

Live selection/reject demo (needs ``wrist_vision.py --live`` publishing frames)::

    python3 laptop/floor_grasp.py --live

Add ``--arm`` to also drive a running ``arm_session.py serve`` session; physical
motion additionally requires the execution gate above to be True.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple
import argparse
import math
import time

import config
from vision_segment import ObjectDetection


DEBUG_DIR = Path(__file__).resolve().parents[1] / "data" / "vision"
LATEST_RAW_PATH = DEBUG_DIR / "wrist_camera_latest_raw.jpg"


# ======================================================================
# Pure candidate selection (no camera, no arm -- fully unit testable)
# ======================================================================
def _bbox_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def filter_wrist_candidates(candidates, frame_shape, marker_boxes=(),
                            min_area_ratio=None, max_area_ratio=None,
                            max_aspect=None, marker_iou=None, border_margin=None):
    """Keep portable tabletop objects; drop finger tapes, floor, and arm.

    A candidate is rejected when it touches the frame border (floor/arm spill
    over the edge), when its box area ratio is outside the portable range, when
    it is a long thin sliver, or when it overlaps a detected finger-tape box.
    No background image, color, or absolute size is assumed.
    """
    height, width = frame_shape[:2]
    frame_area = float(width * height)
    min_area_ratio = (config.FLOOR_CAND_MIN_AREA_RATIO
                      if min_area_ratio is None else min_area_ratio)
    max_area_ratio = (config.FLOOR_CAND_MAX_AREA_RATIO
                      if max_area_ratio is None else max_area_ratio)
    max_aspect = (config.FLOOR_CAND_MAX_ASPECT
                  if max_aspect is None else max_aspect)
    marker_iou = (config.FLOOR_CAND_MARKER_IOU
                  if marker_iou is None else marker_iou)
    border_margin = (config.FLOOR_CAND_BORDER_MARGIN_PX
                     if border_margin is None else border_margin)
    valid = []
    for item in candidates:
        x, y, box_width, box_height = item.bbox
        if (x <= border_margin or y <= border_margin
                or x + box_width >= width - border_margin
                or y + box_height >= height - border_margin):
            continue
        ratio = (box_width * box_height) / frame_area
        if not min_area_ratio <= ratio <= max_area_ratio:
            continue
        aspect = max(box_width, box_height) / max(1, min(box_width, box_height))
        if aspect > max_aspect:
            continue
        if any(_bbox_iou(item.bbox, marker_box) > marker_iou
               for marker_box in marker_boxes):
            continue
        valid.append(item)
    return valid


def rank_wrist_candidates(candidates, frame_shape, gripper_center=None):
    """Order surviving objects best-first: nearest the jaws, mild size bonus."""
    height, width = frame_shape[:2]
    diagonal = math.hypot(width, height)
    reference = (tuple(gripper_center) if gripper_center is not None
                 else (width / 2.0, height / 2.0))
    scored = []
    for item in candidates:
        distance = math.hypot(item.center[0] - reference[0],
                              item.center[1] - reference[1])
        score = -distance / diagonal + 0.035 * math.log1p(item.area)
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _score, item in scored]


class CandidateSelector:
    """Pick one object; reject() drops it and exposes the next (step 3).

    Rejection is by image position with a radius, never by a FastSAM segment id:
    the segmenter renumbers instances every frame, so a transient id cannot name
    "the one the human vetoed". A vetoed image location stays vetoed as the ids
    reshuffle. ``reject`` is the shared-autonomy hook an ErrP decision calls.
    """

    def __init__(self, reject_radius_px):
        if not reject_radius_px > 0:
            raise ValueError("reject radius must be positive")
        self.reject_radius_px = float(reject_radius_px)
        self.rejected_points: List[Tuple[float, float]] = []

    def _is_rejected(self, candidate):
        cx, cy = candidate.center
        return any(math.hypot(cx - rx, cy - ry) <= self.reject_radius_px
                   for rx, ry in self.rejected_points)

    def choose(self, ranked_candidates):
        """Return the best-ranked candidate that has not been vetoed."""
        for candidate in ranked_candidates:
            if not self._is_rejected(candidate):
                return candidate
        return None

    def reject(self, candidate):
        """Human said "not that one": veto its location and move on."""
        self.rejected_points.append(tuple(float(v) for v in candidate.center))

    def confirm(self):
        """Accepted (or delivered): forget vetoes; the next goal may differ."""
        self.rejected_points = []

    reset = confirm


# ======================================================================
# Scene perception (FastSAM instances + finger-tape exclusion)
# ======================================================================
@dataclass
class WristScene:
    ranked: List[ObjectDetection]
    gripper: object                       # wrist_vision.GripperObservation | None
    marker_boxes: List[Tuple[int, int, int, int]]
    frame_shape: Tuple[int, int, int]


class WristSceneDetector:
    """FastSAM instance proposals + blue/red finger-tape exclusion."""

    def __init__(self, fastsam=None, wrist_detector=None):
        self._fastsam = fastsam
        self._wrist = wrist_detector

    def _fastsam_model(self):
        if self._fastsam is None:
            from vision_segment import FastSAMDetector
            self._fastsam = FastSAMDetector()
        return self._fastsam

    def _wrist_model(self):
        if self._wrist is None:
            from wrist_vision import WristDetector
            self._wrist = WristDetector()
        return self._wrist

    def scene(self, frame):
        candidates = self._fastsam_model().candidates(frame)
        gripper_observation, _masks = self._wrist_model().detect(frame)
        gripper = gripper_observation.gripper
        marker_boxes = []
        gripper_center = None
        if gripper is not None:
            marker_boxes = [tuple(gripper.blue.bbox), tuple(gripper.red.bbox)]
            gripper_center = gripper.center
        valid = filter_wrist_candidates(candidates, frame.shape, marker_boxes)
        ranked = rank_wrist_candidates(valid, frame.shape, gripper_center)
        return WristScene(ranked, gripper, marker_boxes, frame.shape), \
            gripper_observation


# ======================================================================
# Arm access through the persistent session (never opens the Uno itself)
# ======================================================================
class SessionArmClient:
    """Thin wrapper over ``arm_session.ArmSessionClient`` used by the controller."""

    def __init__(self, client=None):
        if client is None:
            from arm_session import ArmSessionClient
            client = ArmSessionClient()
        self.client = client

    def status(self):
        return self.client.request({"command": "status"})["pose"]

    def move(self, pose, settle_s=0.0, timeout=15.0):
        return self.client.request({
            "command": "move", "pose": [int(v) for v in pose],
            "settle_s": settle_s, "timeout": timeout})["pose"]

    def floor(self, level, elbow, settle_s=0.0, timeout=15.0):
        return self.client.request({
            "command": "floor", "level": level, "elbow": int(elbow),
            "settle_s": settle_s, "timeout": timeout})["pose"]


# ======================================================================
# Fail-closed grasp state machine
# ======================================================================
STATES = (
    "IDLE", "TARGET_SELECTED", "FLOOR_X_ALIGN", "DESCEND_TO_FLOOR",
    "CLOSE_PROBE", "LIFT_TO_HOVER", "VERIFY_LIFT", "HOLD",
    "RECOVER_OPEN_HOVER", "STOP",
)


@dataclass
class GraspResult:
    ok: bool
    state: str
    reason: str
    executed: bool = False              # did any physical descent/close happen
    planned_elbow: Optional[int] = None
    contact: Optional[str] = None       # CONTACT | FREE | UNKNOWN
    trace: List[str] = field(default_factory=list)


class FloorGraspController:
    """Drive one chosen object from hover to grasp with fail-closed recovery.

    ``perceive`` returns a fresh ``(WristScene, wrist_observation)`` after motion
    settles. ``arm`` is a :class:`SessionArmClient` (or a compatible fake).  With
    the execution gate off, the controller runs the full recognize/align/plan
    logic and reports the *planned* motion without commanding physical descent.
    """

    def __init__(self, arm, perceive, baseline=None, elbow=None,
                 execute=None):
        self.arm = arm
        self.perceive = perceive
        self.baseline = baseline
        self.reference_elbow = (config.FLOOR_REFERENCE_ELBOW
                                if elbow is None else int(elbow))
        self.execute = (config.FLOOR_GRASP_EXECUTE_VERIFIED
                        if execute is None else bool(execute))

    # -- helpers -------------------------------------------------------
    def _log(self, trace, message):
        trace.append(message)
        print(f"[floor-grasp] {message}", flush=True)

    def _recover(self, trace, reason):
        """Best-effort OPEN then HOVER; STOP with a clear reason regardless."""
        self._log(trace, f"RECOVER: {reason}")
        if self.execute:
            try:
                from floor_motion import floor_pose
                self.arm.move(
                    floor_pose(self.reference_elbow, "hover",
                               gripper=config.GRIP_OPEN),
                    settle_s=config.FLOOR_SETTLE_S)
                self._log(trace, "recovered to open hover")
            except Exception as exc:            # noqa: BLE001 - must not raise
                self._log(trace, f"WARNING recovery move failed: {exc}")
        return GraspResult(False, "STOP", reason, executed=self.execute,
                           trace=trace)

    def _x_error(self, scene, observation, target):
        """Signed image-x error (target - jaw center), jaw center if seen."""
        if scene.gripper is not None:
            return target.center[0] - scene.gripper.center[0]
        width = scene.frame_shape[1]
        return target.center[0] - width / 2.0

    # -- main routine --------------------------------------------------
    def grasp(self, target):
        """Attempt to grasp ``target`` (an ObjectDetection). Fail closed."""
        trace: List[str] = []
        self._log(trace, f"TARGET_SELECTED center={tuple(round(v,1) for v in target.center)} "
                         f"area={target.area:.0f}")

        # --- FLOOR_X_ALIGN: bounded visual servo along the floor curve ---
        # Re-measure each iteration and nudge elbow so the target drifts toward
        # the jaw center in x. Only the SIGN of the response is used, so an exact
        # px/deg gain is not required -- but the sign itself is still unverified
        # on the wrist camera, which is why execution stays gated.
        elbow = self.reference_elbow
        lower, upper = config.FLOOR_ELBOW_RANGE
        current = target
        aligned = False
        for iteration in range(config.FLOOR_X_ALIGN_MAX_ITERS):
            scene, observation = self.perceive()
            if scene.gripper is None:
                return self._recover(
                    trace, "finger markers not visible during alignment")
            # Re-acquire the same object by nearest-position continuity: FastSAM
            # ids are not stable, so identity is carried by image location.
            current = _nearest(scene.ranked, current) or current
            error = self._x_error(scene, observation, current)
            self._log(trace, f"FLOOR_X_ALIGN iter={iteration} x_error={error:+.1f}px "
                             f"elbow={elbow}")
            if abs(error) <= config.FLOOR_X_ALIGN_TOL_PX:
                aligned = True
                break
            magnitude = config.FLOOR_X_ALIGN_ELBOW_STEP
            direction = config.FLOOR_X_ALIGN_ELBOW_SIGN * (1 if error > 0 else -1)
            proposed = int(elbow + magnitude * direction)
            if not lower <= proposed <= upper:
                return self._recover(
                    trace, f"floor alignment hit elbow limit at {proposed}")
            elbow = proposed
            if self.execute:
                self.arm.floor("hover", elbow, settle_s=config.FLOOR_SETTLE_S)
            else:
                self._log(trace, f"PLANNED hover elbow -> {elbow} (execution gated)")
        if not aligned:
            return self._recover(trace, "floor alignment did not converge")
        self._log(trace, f"aligned at elbow={elbow}")

        if not self.execute:
            reason = ("execution gated (FLOOR_GRASP_EXECUTE_VERIFIED=False): "
                      "planned floor grasp only, no physical descent/close")
            self._log(trace, reason)
            return GraspResult(True, "TARGET_SELECTED", reason, executed=False,
                               planned_elbow=elbow, trace=trace)

        # --- DESCEND_TO_FLOOR: hover -> grasp at the aligned elbow ---
        self._log(trace, f"DESCEND_TO_FLOOR elbow={elbow}")
        self.arm.floor("grasp", elbow, settle_s=config.FLOOR_SETTLE_S)

        # --- CLOSE_PROBE ---
        from floor_motion import floor_pose
        self._log(trace, "CLOSE_PROBE closing gripper")
        self.arm.move(floor_pose(elbow, "grasp", gripper=config.GRIP_CLOSED),
                      settle_s=config.FLOOR_SETTLE_S)
        scene, observation = self.perceive()

        # --- contact verification (fail closed) ---
        if self.baseline is None:
            return self._recover(
                trace, "no empty-jaw baseline; contact cannot be verified")
        assessment = self.baseline.assess(config.GRIP_CLOSED, observation)
        self._log(trace, f"contact={assessment.state}: {assessment.reason}")
        if not assessment.contact:
            return self._recover(
                trace, f"grasp not confirmed (contact={assessment.state})")

        # --- LIFT_TO_HOVER + coherent-motion verification ---
        self._log(trace, "LIFT_TO_HOVER")
        self.arm.move(floor_pose(elbow, "hover", gripper=config.GRIP_CLOSED),
                      settle_s=config.FLOOR_SETTLE_S)
        lifted_scene, _lifted_obs = self.perceive()
        held = _nearest(lifted_scene.ranked, target)
        if held is None:
            return self._recover(
                trace, "target not seen with the gripper after lift")
        self._log(trace, "VERIFY_LIFT: object rose with the gripper")
        return GraspResult(True, "HOLD", "grasp verified and lifted",
                           executed=True, planned_elbow=elbow,
                           contact=assessment.state, trace=trace)


def _nearest(candidates, reference, max_distance=None):
    """Nearest candidate to a reference center (position-based re-acquisition)."""
    if not candidates:
        return None
    best = min(candidates,
               key=lambda item: math.hypot(item.center[0] - reference.center[0],
                                           item.center[1] - reference.center[1]))
    if max_distance is not None:
        distance = math.hypot(best.center[0] - reference.center[0],
                             best.center[1] - reference.center[1])
        if distance > max_distance:
            return None
    return best


# ======================================================================
# Live selection / reject demo (reads the published raw frame; no 2nd camera)
# ======================================================================
def _read_fresh_raw(previous_mtime, timeout=4.0):
    """Return (frame, mtime) once wrist_vision publishes a new raw frame."""
    import cv2
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            mtime = LATEST_RAW_PATH.stat().st_mtime_ns
        except FileNotFoundError:
            time.sleep(0.05)
            continue
        if mtime != previous_mtime:
            frame = cv2.imread(str(LATEST_RAW_PATH))
            if frame is not None:
                return frame, mtime
        time.sleep(0.05)
    return None, previous_mtime


def _annotate_scene(frame, scene, selected, state_text):
    import cv2
    image = frame.copy()
    for index, candidate in enumerate(scene.ranked):
        x, y, box_width, box_height = candidate.bbox
        chosen = selected is not None and candidate is selected
        color = (40, 230, 40) if chosen else (200, 200, 200)
        thickness = 3 if chosen else 1
        cv2.rectangle(image, (x, y), (x + box_width, y + box_height),
                      color, thickness)
        label = f"#{index}" + (" SELECTED" if chosen else "")
        cv2.putText(image, label, (x, max(20, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    if scene.gripper is not None:
        center = tuple(int(round(v)) for v in scene.gripper.center)
        cv2.drawMarker(image, center, (0, 255, 255), cv2.MARKER_CROSS, 30, 3)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 78), (15, 20, 25), -1)
    image = cv2.addWeighted(overlay, 0.78, image, 0.22, 0)
    cv2.putText(image, state_text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (40, 230, 120), 2, cv2.LINE_AA)
    cv2.putText(image, f"objects={len(scene.ranked)}  "
                       "keys: n=reject/next  y=confirm/grasp  q=quit",
                (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (230, 230, 230), 2, cv2.LINE_AA)
    return image


def run_live(use_arm=False):
    """Live recognize/select/reject loop reading wrist_vision's raw frames."""
    import cv2
    detector = WristSceneDetector()
    diagonal = math.hypot(*config.WRIST_FRAME_SIZE)
    selector = CandidateSelector(
        reject_radius_px=config.FLOOR_REJECT_RADIUS_RATIO * diagonal)
    arm = None
    baseline = None
    if use_arm:
        arm = SessionArmClient()
        try:
            from visual_contact import JawBaseline
            baseline = JawBaseline.load()
        except Exception as exc:            # noqa: BLE001
            print(f"[floor-grasp] no jaw baseline ({exc}); lift stays disabled")

    window = "brainToArm floor grasp (n reject, y confirm, q quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    previous_mtime = None
    selected = None
    status = "recognizing objects..."
    print("[floor-grasp] reading wrist_vision raw frames; "
          "n=reject/next  y=confirm/grasp  q=quit", flush=True)
    try:
        while True:
            frame, previous_mtime = _read_fresh_raw(previous_mtime)
            if frame is None:
                print("[floor-grasp] waiting for wrist_vision --live raw frames")
                if cv2.waitKey(200) & 0xFF == ord("q"):
                    break
                continue
            scene, _obs = detector.scene(frame)
            selected = selector.choose(scene.ranked)
            if selected is None:
                status = ("all objects rejected -- confirm(y) to reset"
                          if scene.ranked else "no portable object detected")
            else:
                status = (f"selected object at "
                          f"{tuple(round(v) for v in selected.center)} "
                          f"({len(scene.ranked)} seen)")
            cv2.imshow(window, _annotate_scene(frame, scene, selected, status))
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n") and selected is not None:
                selector.reject(selected)
                print(f"[floor-grasp] REJECT -> next object "
                      f"(vetoed {len(selector.rejected_points)})", flush=True)
            if key == ord("y") and selected is not None:
                print("[floor-grasp] CONFIRM selection", flush=True)
                if arm is not None:
                    def perceive():
                        mtime = None
                        for _ in range(config.FLOOR_SETTLE_DISCARD_FRAMES + 1):
                            new_frame, mtime = _read_fresh_raw(mtime)
                        return detector.scene(new_frame)
                    controller = FloorGraspController(
                        arm, perceive, baseline=baseline)
                    result = controller.grasp(selected)
                    print(f"[floor-grasp] result ok={result.ok} "
                          f"state={result.state}: {result.reason}", flush=True)
                    if result.ok and result.executed:
                        selector.confirm()
                else:
                    print("[floor-grasp] no --arm session; selection confirmed "
                          "(perception/decision only)", flush=True)
                    selector.confirm()
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="live recognize/select/reject loop")
    parser.add_argument("--arm", action="store_true",
                        help="also drive a running arm_session serve session")
    args = parser.parse_args()
    if args.live:
        run_live(use_arm=args.arm)
        return
    parser.error("use --live (optionally with --arm)")


if __name__ == "__main__":
    main()
