"""Eye-in-hand floor servo with a fixed-reach vertical pinch.

The open blue/red finger markers and a size-aware interior pinch line are
aligned at a safe 55 mm hover. Forward reach is then locked: the fingers
descend by shoulder height compensation only, so temporary FastSAM occlusion
cannot switch targets or push the object away. A simulation-trained macro
policy gates descend/close/lift, and the calibrated empty-jaw curve must confirm
physical obstruction both before and after lift.
"""

from pathlib import Path
import argparse
import math
import time

import numpy as np

import config
from floor_calibrate import FloorHomography


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "vision" / "wrist_camera_latest_raw.jpg"

# Servo joints we command at the observation pose: elbow (J2) and wrist_pitch
# (J3 index). Base stays 90, wrist_roll level, gripper open during alignment.
J_SHOULDER, J_ELBOW, J_WRIST = 1, 2, 3

MAX_STEP_DEG = 3          # gentle per-waypoint joint change
STEP_SETTLE_S = 0.25
ALIGN_MAX_ITERS = 26
JAC_PERTURB_DEG = 4       # nudge size for Jacobian measurement
OBJ_JUMP_REJECT_PX = 160  # reject a frame whose object pixel jumps this far

# Search only on the already physically exercised 35 mm floor-hover manifold.
# A floor object that the fixed base can grasp must pass close to the jaw
# centerline as reach increases; decorative/background instances never become
# eligible merely because FastSAM segmented them.
SEARCH_REACHES = (0, 12, 24, 36, 48, 60)
SEARCH_MAX_DU_PX = 55.0
SEARCH_DV_WINDOW_PX = (-360.0, 45.0)
SEARCH_CONFIRM_RADIUS_PX = 80.0
SEARCH_MIN_CONFIDENCE = 0.65
TRACK_MIN_CONFIDENCE = 0.50
GRASP_INSET_RATIO = 0.35   # grasp inside the object, not on its trailing edge
DYNAMIC_CONTACT_MIN_MARGIN_PX = 1.5
DYNAMIC_CONTACT_MAD_MULTIPLIER = 5.0
CONTACT_RETENTION_RATIO = 0.50
CLIPPED_GRASP_DV_WINDOW_PX = (-52.0, -24.0)
CLIPPED_GRASP_VOTES = 2
EMPTY_CALIBRATION_Z = 0.100
HOVER_GRASP_DV_TARGET_PX = 20.0
HOVER_GRASP_DV_TOL_PX = 12.0
BASE_JACOBIAN_STEP_DEG = 2
BASE_CENTER_TOL_PX = 55.0
BASE_CENTER_MAX_STEP_DEG = 3
BASE_CENTER_RANGE_DEG = 10
CONTACT_SAMPLE_COUNT = 5
MIN_SEARCH_TO_PINCH_ADVANCE = 6
CONTACT_PROBE_Z = EMPTY_CALIBRATION_Z


def _fresh_frame(discard=None, timeout=8.0):
    import cv2
    discard = (config.FLOOR_SETTLE_DISCARD_FRAMES + 1) if discard is None else discard
    try:
        prev = RAW.stat().st_mtime_ns
    except FileNotFoundError:
        prev = None
    seen = 0
    deadline = time.monotonic() + timeout
    while seen < discard and time.monotonic() < deadline:
        try:
            m = RAW.stat().st_mtime_ns
        except FileNotFoundError:
            time.sleep(0.05); continue
        if prev is None or m != prev:
            prev = m; seen += 1
        time.sleep(0.05)
    if seen < discard:
        age = (time.time() - RAW.stat().st_mtime
               if RAW.exists() else math.inf)
        raise RuntimeError(
            f"wrist camera did not publish {discard} new frame(s) within "
            f"{timeout:.1f}s (last frame age={age:.1f}s)")
    frame = cv2.imread(str(RAW))
    if frame is None:
        raise RuntimeError("no wrist frame; is the camera publisher running?")
    # The PW315 auto exposure can remain dark after the arm briefly fills the
    # frame. Preserve RAW on disk for evidence, but normalize only genuinely
    # low-light perception frames. A single *linear* B/G/R gain preserves HSV
    # hue/saturation; nonlinear gamma shifted red tape into the yellow target
    # range and is therefore deliberately not used here.
    gray_mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    if 1.0 < gray_mean < 70.0:
        gain = float(np.clip(95.0 / gray_mean, 1.0, 6.0))
        frame = cv2.convertScaleAbs(frame, alpha=gain, beta=0)
    return frame


class FloorServo:
    def __init__(self, client, calib):
        self.client = client
        self.calib = calib
        from floor_grasp import WristSceneDetector
        from wrist_vision import WristDetector
        self.scene_detector = WristSceneDetector()
        self.marker_detector = WristDetector()
        self.locked_target_center = None
        self.shadow = None
        self.empty_closed_opening_px = None
        self.empty_closed_opening_mad_px = None
        self.empty_closed_red_x_px = None
        self.empty_closed_red_x_mad_px = None
        self.last_target_clipped = False
        self.locked_target_clipped = False
        self.base_angle = int(config.FLOOR_BASE_ANGLE)
        self.base_du_per_deg = None

    # -- gentle stepped motion -------------------------------------------------
    def slow_move(self, target, final_settle=None):
        target = [int(round(v)) for v in target]
        current = self.client.request({"command": "status"})["pose"]
        span = max(abs(t - c) for t, c in zip(target, current))
        steps = max(1, int(math.ceil(span / MAX_STEP_DEG)))
        final_settle = config.FLOOR_SETTLE_S if final_settle is None else final_settle
        for i in range(1, steps + 1):
            frac = i / steps
            wp = [int(round(c + (t - c) * frac)) for c, t in zip(current, target)]
            self.client.request({
                "command": "move", "pose": wp,
                "require_camera": True,
                "settle_s": final_settle if i == steps else STEP_SETTLE_S})

    # -- observation -----------------------------------------------------------
    def object_floor(self):
        scene, _ = self.scene_detector.scene(_fresh_frame())
        if not scene.ranked:
            return None
        return np.array(self.calib.pixel_to_floor(scene.ranked[0].center))

    def gripper_floor(self):
        """Floor projection of the finger-marker midpoint at the current pose."""
        obs, _ = self.marker_detector.detect(_fresh_frame())
        if obs.gripper is None:
            return None
        return np.array(self.calib.pixel_to_floor(obs.gripper.center))

    # -- floor-level image servo ----------------------------------------------
    # Correct formulation: at a pose where the gripper is DOWN at floor level,
    # both the fingertips (markers) and the floor object lie on the same plane,
    # so their image pixels can be aligned without any depth ambiguity. The
    # camera is rigid with the gripper, so the marker midpoint is a nearly fixed
    # target pixel; the object is driven onto it by lowering wrist_pitch (which
    # reaches the fingers further along the floor) with the shoulder compensating
    # to hold floor height. Only the SIGN/gain of d(object_pixel)/d(wrist_pitch)
    # is needed, measured live.
    def _level_pose(self, wrist_pitch, z_m, gripper=None, elbow=None):
        """Pose at the given elbow and wrist_pitch, with shoulder chosen (FK) so
        the tool sits at height ``z_m`` above the floor. Both wrist_pitch (aim)
        and elbow (forward reach) extend the fingers along the floor; shoulder
        compensates height. Alignment runs at a safe hover z (no floor scrape);
        only the final grasp uses a near-floor z."""
        import arm_fk
        elbow = config.FLOOR_REFERENCE_ELBOW if elbow is None else int(elbow)
        gripper = config.GRIP_OPEN if gripper is None else gripper
        lo, hi = float(config.SERVO_MIN[J_SHOULDER]), float(config.SERVO_MAX[J_SHOULDER])
        for _ in range(40):
            mid = (lo + hi) / 2
            z = arm_fk.tool_position([
                self.base_angle, mid, elbow, wrist_pitch, 90, 170])[2]
            if z > z_m:
                lo = mid
            else:
                hi = mid
        shoulder = int(round((lo + hi) / 2))
        shoulder = max(config.SERVO_MIN[J_SHOULDER],
                       min(config.SERVO_MAX[J_SHOULDER], shoulder))
        return [self.base_angle, shoulder, elbow, int(wrist_pitch),
                int(gripper), 170]

    def _reach_pose(self, reach, z_m, gripper=None):
        """Single forward-reach scalar -> (elbow, wrist_pitch). Lower wrist_pitch
        first (180->140), then also open the elbow forward (90->78), then finish
        wrist_pitch (140->130). Higher ``reach`` = fingers further along floor."""
        wp, elbow = 180, config.FLOOR_REFERENCE_ELBOW
        r = int(reach)
        wp_first = min(r, 40)           # 180 -> 140
        r -= wp_first
        wp = 180 - wp_first
        el_span = min(r, 12)            # elbow 90 -> 78
        r -= el_span
        elbow = 90 - el_span
        wp = max(130, wp - r)           # remaining lowers wp 140 -> 130
        return self._level_pose(wp, z_m, gripper=gripper, elbow=elbow), wp, elbow

    REACH_MAX = 62                      # 40 (wp) + 12 (elbow) + 10 (wp) degrees

    # Clear the complete <=40 mm simulated object envelope before moving over a
    # candidate. The old 35 mm hover could skim a tall toy and only catch its
    # trailing edge. Verification lifts still higher so floor contact is
    # physically impossible for this envelope.
    HOVER_Z = 0.055
    GRASP_Z = 0.006
    VERIFY_Z = EMPTY_CALIBRATION_Z

    @staticmethod
    def candidate_grasp_point(candidate):
        """Size-aware interior pinch line, inset from the trailing bbox edge."""
        _x, y, _width, height = candidate.bbox
        return np.array((candidate.center[0],
                         y + (1.0 - GRASP_INSET_RATIO) * height), dtype=float)

    @staticmethod
    def _red_marker_x(frame):
        """Return the mounted right/red finger x, even if blue is occluded."""
        import cv2
        from wrist_vision import _combine_hsv_ranges, _valid_blobs

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = frame.shape[:2]
        area = height * width
        red_mask = _combine_hsv_ranges(hsv, config.WRIST_RED_HSV)
        blobs = _valid_blobs(
            red_mask, hsv,
            area * config.WRIST_MARKER_MIN_AREA_RATIO,
            area * config.WRIST_MARKER_MAX_AREA_RATIO,
            config.WRIST_MARKER_MIN_FILL_RATIO,
            config.WRIST_MARKER_MAX_ASPECT,
            allow_bottom=True)
        expected_x = width * (
            config.WRIST_GRIPPER_CLOSED_PROFILE["center"][0]
            + 0.5 * config.WRIST_GRIPPER_CLOSED_PROFILE["opening_ratio"])
        plausible = [
            blob for blob in blobs
            if blob.center[1] >= 0.75 * height
            and 0.45 * width <= blob.center[0] <= 0.72 * width]
        if not plausible:
            return None
        return float(min(
            plausible, key=lambda blob: abs(blob.center[0] - expected_x)
        ).center[0])

    @classmethod
    def search_choice(cls, scene):
        """Return the best currently approachable candidate, or ``None``.

        This is geometry, not an object/location prior: with base yaw fixed, a
        graspable floor object must enter a narrow horizontal band between the
        jaws and approach their row from above during the reach sweep.
        """
        if scene.gripper is None:
            height, width = scene.frame_shape[:2]
            midpoint = np.asarray(
                config.WRIST_GRIPPER_OPEN_PROFILE["center"], dtype=float)
            midpoint *= np.asarray([width, height], dtype=float)
        else:
            midpoint = np.asarray(scene.gripper.center, dtype=float)
        eligible = []
        for candidate in scene.ranked:
            if float(getattr(candidate, "confidence", 1.0)) < SEARCH_MIN_CONFIDENCE:
                continue
            edge = cls.candidate_grasp_point(candidate)
            du, dv = float(edge[0] - midpoint[0]), float(edge[1] - midpoint[1])
            if (abs(du) <= SEARCH_MAX_DU_PX
                    and SEARCH_DV_WINDOW_PX[0] <= dv <= SEARCH_DV_WINDOW_PX[1]):
                # Prefer the object nearest the jaw row, then the more centered
                # one. Candidate ranking/appearance never overrides reachability.
                eligible.append((abs(dv) + 0.35 * abs(du), candidate))
        return min(eligible, key=lambda item: item[0])[1] if eligible else None

    def search_from_home(self):
        """HOME -> verified hover sweep -> two-frame target confirmation.

        Returns ``(reach, candidate)`` and leaves the arm at the target-visible
        hover. No remembered reach, target coordinate, or user correction is
        consumed. Failure reverses the exercised hover route and returns HOME.
        """
        home = list(config.HOME_POSE)
        print(f"[servo] AUTONOMOUS HOME {home}")
        self.slow_move(home)
        self.base_angle = int(home[config.J_BASE])
        self.base_du_per_deg = None

        # This exact HOME->ready->floor transition is the same one used by the
        # persistent session startup. Open before approaching the floor.
        ready = list(home)
        ready[config.J_WRIST] = 180
        ready[config.J_GRIP] = config.GRIP_OPEN
        ready[config.J_ROLL] = config.FLOOR_WRIST_ROLL
        self.slow_move(ready)

        visited = []
        for view, reach in enumerate(SEARCH_REACHES, 1):
            pose = self._reach_pose(reach, self.HOVER_Z,
                                    gripper=config.GRIP_OPEN)[0]
            self.slow_move(pose)
            visited.append(pose)
            search_frame = _fresh_frame()
            import cv2
            cv2.imwrite(str(ROOT / "data" / "vision"
                            / f"autonomous_search_view_{view}.jpg"),
                        search_frame)
            scene, _ = self.scene_detector.scene(search_frame)
            candidate = self.search_choice(scene)
            print(f"[servo] SEARCH view={view}/{len(SEARCH_REACHES)} "
                  f"reach={reach} objects={len(scene.ranked)} "
                  f"eligible={candidate is not None}")
            if candidate is None:
                continue

            # A fresh second frame must contain the same grasp point. This drops
            # transient FastSAM fragments without asking a human to intervene.
            reference = self.candidate_grasp_point(candidate)
            confirm_scene, _ = self.scene_detector.scene(_fresh_frame(discard=2))
            confirmed = self.search_choice(confirm_scene)
            if confirmed is None:
                print("[servo] SEARCH candidate vanished on confirmation")
                continue
            distance = float(np.linalg.norm(
                self.candidate_grasp_point(confirmed) - reference))
            if distance > SEARCH_CONFIRM_RADIUS_PX:
                print(f"[servo] SEARCH identity moved {distance:.0f}px; continue")
                continue

            # Calibrate the empty endpoint at the selected wrist pitch, but lift
            # to 100 mm first. Tool center is not fingertip clearance, so closing
            # at the 55 mm alignment height can still touch a tall object.
            # Gripper flex is orientation-dependent, making HOME/another wrist
            # pitch non-transferable; the high pose preserves wrist pitch while
            # clearing the complete object envelope.
            closed_calibration = self._reach_pose(
                reach, EMPTY_CALIBRATION_Z,
                gripper=config.GRIP_CLOSED)[0]
            open_calibration = self._reach_pose(
                reach, EMPTY_CALIBRATION_Z,
                gripper=config.GRIP_OPEN)[0]
            self.slow_move(open_calibration)
            self.slow_move(closed_calibration)
            empty_openings = []
            empty_red_x = []
            for _ in range(CONTACT_SAMPLE_COUNT):
                empty_frame = _fresh_frame(discard=1)
                empty_observation, _ = self.marker_detector.detect(empty_frame)
                if empty_observation.gripper is not None:
                    empty_openings.append(
                        float(empty_observation.gripper.opening_px))
                red_x = self._red_marker_x(empty_frame)
                if red_x is not None:
                    empty_red_x.append(red_x)
            self.slow_move(open_calibration)
            self.slow_move(pose)  # return open to the target-visible hover
            if not empty_openings and not empty_red_x:
                print("[servo] SEARCH empty-close calibration lost markers")
                continue
            if empty_openings:
                self.empty_closed_opening_px = float(np.median(empty_openings))
                self.empty_closed_opening_mad_px = float(np.median(np.abs(
                    np.asarray(empty_openings) - self.empty_closed_opening_px)))
            if empty_red_x:
                self.empty_closed_red_x_px = float(np.median(empty_red_x))
                self.empty_closed_red_x_mad_px = float(np.median(np.abs(
                    np.asarray(empty_red_x) - self.empty_closed_red_x_px)))

            # Reacquire at the returned, identical hover pose. Do not compare
            # against pixels captured before the 100 mm excursion: an eye-in-
            # hand camera can shift/resize masks substantially during that
            # excursion even though the object never moved.
            post_candidates = []
            for _ in range(5):
                final_scene, _ = self.scene_detector.scene(
                    _fresh_frame(discard=1))
                final_candidate = self.search_choice(final_scene)
                if final_candidate is None:
                    continue
                point = self.candidate_grasp_point(final_candidate)
                if not post_candidates or np.linalg.norm(
                        point - self.candidate_grasp_point(
                            post_candidates[0])) <= SEARCH_CONFIRM_RADIUS_PX:
                    post_candidates.append(final_candidate)
                if len(post_candidates) >= 2:
                    break
            if len(post_candidates) < 2:
                print("[servo] SEARCH post-calibration segmentation sparse; "
                      "retain the two-frame pre-calibration identity and "
                      "require live reacquisition in alignment")
                return reach, confirmed
            final_candidate, verify_candidate = post_candidates[:2]
            final_distance = float(np.linalg.norm(
                self.candidate_grasp_point(verify_candidate)
                - self.candidate_grasp_point(final_candidate)))
            print(f"[servo] SEARCH FOUND reach={reach} "
                  f"center={verify_candidate.center} "
                  f"confirm_delta={final_distance:.1f}px "
                  f"empty-close="
                  f"{self.empty_closed_opening_px if self.empty_closed_opening_px is not None else 'red-only'} "
                  f"red-x={self.empty_closed_red_x_px}")
            return reach, verify_candidate

        print("[servo] SEARCH exhausted verified floor views; return HOME")
        for pose in reversed(visited[:-1]):
            self.slow_move(pose)
        self.slow_move(ready)
        self.slow_move(home)
        return None, None

    def _obj_and_marker_px(self, reference=None):
        """Return (object interior grasp pixel, marker midpoint) at current pose.

        The 35%-inset line avoids both centroid overreach and the weak trailing-
        edge pinch that can let a long object slip during lift.
        """
        frame = _fresh_frame()
        scene, _ = self.scene_detector.scene(frame)
        obs, _ = self.marker_detector.detect(frame)
        trackable = [
            candidate for candidate in scene.ranked
            if float(getattr(candidate, "confidence", 1.0))
            >= TRACK_MIN_CONFIDENCE]
        if not trackable:
            obj = None
            self.last_target_clipped = False
        elif reference is None:
            candidate = trackable[0]
            obj = self.candidate_grasp_point(candidate)
            self.last_target_clipped = (
                candidate.bbox[1] + candidate.bbox[3]
                >= frame.shape[0] - config.FLOOR_CAND_BORDER_MARGIN_PX)
        else:
            candidate = min(
                trackable,
                key=lambda candidate: np.linalg.norm(
                    self.candidate_grasp_point(candidate)
                    - reference))
            obj = self.candidate_grasp_point(candidate)
            self.last_target_clipped = (
                candidate.bbox[1] + candidate.bbox[3]
                >= frame.shape[0] - config.FLOOR_CAND_BORDER_MARGIN_PX)
        if obs.gripper is None:
            height, width = frame.shape[:2]
            mid = np.asarray(
                config.WRIST_GRIPPER_OPEN_PROFILE["center"], dtype=float)
            mid *= np.asarray([width, height], dtype=float)
        else:
            mid = np.array(obs.gripper.center)
        return obj, mid

    def _center_base_lateral(self, reference, max_iters=4):
        """Close the image-x loop with a measured base-yaw Jacobian.

        Shoulder/elbow/wrist motion spans the arm's sagittal plane and cannot
        remove lateral error.  Servo1 is operational, so measure its sign and
        gain in the current camera geometry instead of assuming either.  Every
        correction is bounded and re-observed; a stalled base therefore fails
        closed before the fingers can push the object.
        """
        reference = None if reference is None else np.asarray(reference, dtype=float)
        obj, mid = self._obj_and_marker_px(reference)
        if obj is None or mid is None:
            return None, None
        du = float(obj[0] - mid[0])
        reference = obj
        # With the fixed-base planar arm, a target centered at hover remains on
        # the physical sagittal plane.  Apparent x drift during descent is
        # eye-in-hand parallax, not a new lateral degree of freedom.
        if abs(du) <= BASE_CENTER_TOL_PX:
            return obj, mid

        if self.base_du_per_deg is None:
            low = max(int(config.SERVO_MIN[config.J_BASE]),
                      int(config.FLOOR_BASE_ANGLE) - BASE_CENTER_RANGE_DEG)
            high = min(int(config.SERVO_MAX[config.J_BASE]),
                       int(config.FLOOR_BASE_ANGLE) + BASE_CENTER_RANGE_DEG)
            direction = (1 if self.base_angle + BASE_JACOBIAN_STEP_DEG <= high
                         else -1)
            probe_base = int(np.clip(
                self.base_angle + direction * BASE_JACOBIAN_STEP_DEG, low, high))
            if probe_base == self.base_angle:
                print("[servo] lateral center: no safe base probe remains")
                return None, None
            pose = list(self.client.request({"command": "status"})["pose"])
            pose[config.J_BASE] = probe_base
            self.slow_move(pose)
            self.base_angle = probe_base
            probe_obj, probe_mid = self._obj_and_marker_px(reference)
            if probe_obj is None or probe_mid is None:
                print("[servo] lateral center: target/markers lost during base probe")
                return None, None
            probe_du = float(probe_obj[0] - probe_mid[0])
            jac = (probe_du - du) / (probe_base - (probe_base -
                   direction * BASE_JACOBIAN_STEP_DEG))
            if not np.isfinite(jac) or abs(jac) < 2.0:
                print(f"[servo] lateral center: base response too small "
                      f"({jac:.2f}px/deg); descent forbidden")
                return None, None
            self.base_du_per_deg = float(jac)
            obj, mid, du, reference = probe_obj, probe_mid, probe_du, probe_obj
            print(f"[servo] measured base lateral Jacobian "
                  f"{self.base_du_per_deg:.2f}px/deg")

        low = max(int(config.SERVO_MIN[config.J_BASE]),
                  int(config.FLOOR_BASE_ANGLE) - BASE_CENTER_RANGE_DEG)
        high = min(int(config.SERVO_MAX[config.J_BASE]),
                   int(config.FLOOR_BASE_ANGLE) + BASE_CENTER_RANGE_DEG)
        for _ in range(max_iters):
            if abs(du) <= BASE_CENTER_TOL_PX:
                return obj, mid
            delta = int(round(np.clip(
                -du / self.base_du_per_deg,
                -BASE_CENTER_MAX_STEP_DEG, BASE_CENTER_MAX_STEP_DEG)))
            if delta == 0:
                delta = -1 if du / self.base_du_per_deg > 0 else 1
            new_base = int(np.clip(self.base_angle + delta, low, high))
            if new_base == self.base_angle:
                print(f"[servo] lateral center: base limit with du={du:.0f}px")
                return None, None
            pose = list(self.client.request({"command": "status"})["pose"])
            pose[config.J_BASE] = new_base
            self.slow_move(pose)
            self.base_angle = new_base
            obj, mid = self._obj_and_marker_px(reference)
            if obj is None or mid is None:
                print("[servo] lateral center: target/markers lost after correction")
                return None, None
            du = float(obj[0] - mid[0])
            reference = obj
            print(f"[servo] lateral center base={self.base_angle} du={du:.0f}px")
        return (obj, mid) if abs(du) <= BASE_CENTER_TOL_PX else (None, None)

    def align(self, verbose=True, start_reach=0, selected=None):
        """Drive the object's image row onto the fingertip row at hover, using a
        single forward-reach scalar (wrist_pitch then elbow).

        ``selected`` preserves the choice made by the multi-object/reject UI;
        subsequent frames track the nearest grasp point rather than silently
        falling back to rank zero.
        """
        # The open jaws span ~288px, so an object within ~half that of the
        # fingertip midpoint is already inside the jaws and graspable. Near
        # contact the object detection gets noisy (the gripper occludes it), so
        # we accept the first frame that lands the object within the jaw span and
        # also remember the best reach seen to fall back on.
        # The old 110 px depth threshold only proved that the object was inside
        # the projected jaw span.  It still left the fingertips in front of the
        # object, which encouraged a bulldozing approach.  Center tightly before
        # locking the forward axis and beginning vertical descent.
        ACCEPT_DU = 55.0
        reach = int(np.clip(start_reach, 0, self.REACH_MAX))
        minimum_pinch_reach = min(
            self.REACH_MAX, reach + MIN_SEARCH_TO_PINCH_ADVANCE)
        pose, wp, elbow = self._reach_pose(reach, self.HOVER_Z)
        self.slow_move(pose)
        last_obj = (None if selected is None
                    else self.candidate_grasp_point(selected))
        tracking_initialized = selected is None
        best = None  # (abs(dv), reach)
        clipped_votes = 0
        previous_clipped_dv = None
        for it in range(ALIGN_MAX_ITERS):
            obj, mid = self._obj_and_marker_px(last_obj)
            if mid is None:
                print(f"[servo] it={it}: markers lost; stopping"); break
            if obj is None:
                print(f"[servo] it={it}: object not detected; retrying")
                continue
            if (tracking_initialized and last_obj is not None
                    and np.linalg.norm(obj - last_obj) > OBJ_JUMP_REJECT_PX):
                print(f"[servo] it={it}: object pixel jumped "
                      f"{np.linalg.norm(obj-last_obj):.0f}px -> outlier, ignoring frame")
                continue
            last_obj = obj
            tracking_initialized = True
            du, dv = float(obj[0] - mid[0]), float(obj[1] - mid[1])
            dv_error = dv - HOVER_GRASP_DV_TARGET_PX
            if verbose:
                print(f"[servo] it={it} reach={reach}(wp={wp},el={elbow}) "
                      f"obj_px={np.round(obj)} marker_px={np.round(mid)} "
                      f"du={du:.0f} dv={dv:.0f} "
                      f"depth_error={dv_error:.0f}")
            if best is None or abs(dv_error) < best[0]:
                best = (abs(dv_error), reach)
            if (abs(dv_error) <= HOVER_GRASP_DV_TOL_PX
                    and abs(du) <= ACCEPT_DU
                    and reach >= minimum_pinch_reach):
                centered_obj, centered_mid = self._center_base_lateral(obj)
                if centered_obj is None:
                    print("[servo] lateral centering failed; descent is forbidden")
                    return None
                du = float(centered_obj[0] - centered_mid[0])
                self.locked_target_center = centered_obj.copy()
                self.locked_target_clipped = False
                print(f"[servo] object within the jaw span at hover "
                      f"(du={du:.0f},dv={dv:.0f}) reach={reach} -> descend & close")
                return reach
            clipped_ready = (
                self.last_target_clipped
                and reach >= minimum_pinch_reach
                and abs(du) <= ACCEPT_DU
                and CLIPPED_GRASP_DV_WINDOW_PX[0] <= dv
                <= CLIPPED_GRASP_DV_WINDOW_PX[1])
            if clipped_ready:
                clipped_votes = (clipped_votes + 1
                                 if previous_clipped_dv is not None
                                 and abs(dv - previous_clipped_dv) <= 8.0 else 1)
                previous_clipped_dv = dv
                if clipped_votes >= CLIPPED_GRASP_VOTES:
                    centered_obj, centered_mid = self._center_base_lateral(obj)
                    if centered_obj is None:
                        print("[servo] lateral centering failed; descent is forbidden")
                        return None
                    du = float(centered_obj[0] - centered_mid[0])
                    self.locked_target_center = centered_obj.copy()
                    self.locked_target_clipped = True
                    print(f"[servo] bottom-clipped grasp geometry stable "
                          f"for {clipped_votes} frames (du={du:.0f},dv={dv:.0f}) "
                          f"reach={reach} -> descend & close")
                    return reach
            else:
                clipped_votes = 0
                previous_clipped_dv = None
            if reach >= self.REACH_MAX:
                break
            step = 1 if abs(dv_error) < 120 else MAX_STEP_DEG
            delta = step if dv_error < 0 else -step
            reach = int(min(self.REACH_MAX, max(0, reach + delta)))
            pose, wp, elbow = self._reach_pose(reach, self.HOVER_Z)
            self.slow_move(pose)
        detail = ("no stable target" if best is None
                  else f"best |dv|={best[0]:.0f}px at reach={best[1]}")
        print(f"[servo] strict alignment did not converge ({detail}); "
              "descent is forbidden")
        return None

    def _authorize_macro(self, expected, canonical_pose, *, target_visible):
        """Require the simulation-trained shield and temporal vote on real data."""

        from full_task_adapter import FullTaskShadowController, TaskAction

        if self.shadow is None:
            self.shadow = FullTaskShadowController()
        self.shadow.reset()
        expected = TaskAction(expected)
        for vote in range(8):
            frame = _fresh_frame(discard=1)
            scene, observation = self.scene_detector.scene(frame)
            if (scene.gripper is None
                    and canonical_pose[config.J_GRIP] == config.GRIP_OPEN):
                from types import SimpleNamespace
                height, width = scene.frame_shape[:2]
                center = np.asarray(
                    config.WRIST_GRIPPER_OPEN_PROFILE["center"], dtype=float)
                center *= np.asarray([width, height], dtype=float)
                scene.gripper = SimpleNamespace(
                    center=tuple(center),
                    opening_px=(config.WRIST_GRIPPER_OPEN_PROFILE["opening_ratio"]
                                * math.hypot(width, height)))
            target = None
            policy_candidates = [
                candidate for candidate in scene.ranked
                if float(getattr(candidate, "confidence", 1.0))
                >= TRACK_MIN_CONFIDENCE]
            if target_visible and policy_candidates:
                reference = self.locked_target_center
                candidate = (policy_candidates[0] if reference is None else min(
                    policy_candidates,
                    key=lambda candidate: np.linalg.norm(
                        self.candidate_grasp_point(candidate)
                        - reference)))
                from types import SimpleNamespace
                point = self.candidate_grasp_point(candidate)
                # The size-aware grasp point saturates above the jaw row once
                # the object's lower bbox is clipped. Two-frame clipped
                # convergence already proved the corresponding depth geometry;
                # represent that semantic alignment consistently to the shield.
                if self.locked_target_clipped and scene.gripper is not None:
                    point[1] = scene.gripper.center[1]
                target = SimpleNamespace(center=tuple(point))
            decision = self.shadow.decide(
                scene, observation, canonical_pose, target=target,
                target_locked=self.locked_target_center is not None)
            print(f"[servo] policy vote={vote + 1} action={decision.action.name} "
                  f"model_score={decision.confidence:.3f} expected={expected.name}")
            if decision.action == expected:
                return True
            if decision.action != TaskAction.WAIT:
                print(f"[servo] policy blocked {expected.name}: {decision.reason}")
                return False
        print(f"[servo] policy did not confirm {expected.name} in time")
        return False

    def _recover_open_hover(self, reach, reason):
        print(f"[servo] RECOVER: {reason}")
        self.slow_move(self._reach_pose(
            reach, self.HOVER_Z, gripper=config.GRIP_OPEN)[0])
        return False

    def _contact_assessment(self, *, allow_single_red=False):
        observations = []
        openings = []
        red_positions = []
        for _ in range(CONTACT_SAMPLE_COUNT):
            frame = _fresh_frame(discard=1)
            observation, _ = self.marker_detector.detect(frame)
            observations.append(observation)
            if observation.gripper is not None:
                openings.append(float(observation.gripper.opening_px))
            red_x = self._red_marker_x(frame)
            if red_x is not None:
                red_positions.append(red_x)
        if self.empty_closed_opening_px is not None:
            from visual_contact import ContactAssessment
            threshold = max(
                DYNAMIC_CONTACT_MIN_MARGIN_PX,
                DYNAMIC_CONTACT_MAD_MULTIPLIER
                * float(self.empty_closed_opening_mad_px or 0.0))
            if not openings and not allow_single_red:
                return ContactAssessment(
                    "UNKNOWN", config.GRIP_CLOSED, None,
                    self.empty_closed_opening_px, None,
                    threshold,
                    f"both finger markers missing in {CONTACT_SAMPLE_COUNT} frames")
            if openings:
                observed = float(np.median(openings))
                residual = observed - self.empty_closed_opening_px
                contact = residual > threshold
                return ContactAssessment(
                    "CONTACT" if contact else "FREE", config.GRIP_CLOSED,
                    observed, self.empty_closed_opening_px, residual,
                    threshold,
                    (f"jaw remained {residual:.1f}px wider than same-run empty "
                     f"endpoint (threshold {threshold:.1f}px)"
                     if contact else
                     f"jaw returned to same-run empty endpoint "
                     f"(residual {residual:.1f}px)"))
        if (allow_single_red and self.empty_closed_red_x_px is not None
                and red_positions):
            from visual_contact import ContactAssessment
            red_x = float(np.median(red_positions))
            # Right finger displacement is half the symmetric jaw opening.
            residual = 2.0 * (red_x - self.empty_closed_red_x_px)
            threshold = max(
                DYNAMIC_CONTACT_MIN_MARGIN_PX,
                2.0 * DYNAMIC_CONTACT_MAD_MULTIPLIER
                * float(self.empty_closed_red_x_mad_px or 0.0))
            contact = residual > threshold
            return ContactAssessment(
                "CONTACT" if contact else "FREE", config.GRIP_CLOSED,
                red_x, self.empty_closed_red_x_px, residual, threshold,
                (f"right finger implies {residual:.1f}px symmetric opening "
                 f"residual (threshold {threshold:.1f}px)"))
        from visual_contact import JawBaseline
        baseline = JawBaseline.load()
        valid = next(
            (item for item in reversed(observations)
             if item.gripper is not None), observations[-1])
        return baseline.assess(config.GRIP_CLOSED, valid)

    @staticmethod
    def contact_retained(closed, lifted):
        """Reject an object that obstructed close but slipped out during lift."""
        if not closed.contact or not lifted.contact:
            return False
        if closed.residual_px is None or lifted.residual_px is None:
            return False
        required = max(DYNAMIC_CONTACT_MIN_MARGIN_PX,
                       CONTACT_RETENTION_RATIO * closed.residual_px)
        return lifted.residual_px >= required

    def grasp(self, reach):
        """Fixed-reach vertical pinch, then camera-verified closed-jaw lift.

        Once hover alignment has selected a reach, forward motion is forbidden.
        Only shoulder changes lower the open fingers. Near-field FastSAM output
        is diagnostic and never changes the target or reach after occlusion.
        """
        from full_task_adapter import TaskAction

        if not config.FLOOR_GRASP_EXECUTE_VERIFIED:
            print("[servo] physical grasp gate is disabled; stopping open at hover")
            return False

        if not self._authorize_macro(
                TaskAction.DESCEND,
                self._reach_pose(
                    reach, self.HOVER_Z, gripper=config.GRIP_OPEN)[0],
                target_visible=True):
            return self._recover_open_hover(
                reach, "trained policy did not authorize descent")

        locked_obj, _locked_mid = self._obj_and_marker_px(
            self.locked_target_center)
        for z in (0.030, 0.024, 0.018, 0.012, self.GRASP_Z):
            pose, wp, elbow = self._reach_pose(reach, z, gripper=config.GRIP_OPEN)
            self.slow_move(pose)
            obj, mid = self._obj_and_marker_px(locked_obj)
            if obj is not None and mid is not None:
                if (locked_obj is None
                        or np.linalg.norm(obj - locked_obj) <= OBJ_JUMP_REJECT_PX):
                    du, dv = float(obj[0] - mid[0]), float(obj[1] - mid[1])
                    locked_obj = obj
                    print(f"[servo] vertical descend z={z*1000:.0f}mm "
                          f"LOCKED reach={reach} du={du:.0f} dv={dv:.0f}")
                else:
                    print(f"[servo] vertical descend z={z*1000:.0f}mm: "
                          "near-field mask changed; ignore it and keep locked reach")
            else:
                print(f"[servo] vertical descend z={z*1000:.0f}mm: "
                      "target occluded; keep locked reach")

        if not self._authorize_macro(
                TaskAction.CLOSE,
                self._reach_pose(
                    reach, self.GRASP_Z, gripper=config.GRIP_OPEN)[0],
                target_visible=False):
            return self._recover_open_hover(
                reach, "trained policy did not authorize close")

        # Close at the fixed floor point. A hobby servo has no torque sensor, so
        # the calibrated empty-jaw visual curve is the physical contact sensor.
        self.slow_move(self._reach_pose(reach, self.GRASP_Z, gripper=config.GRIP_CLOSED)[0])
        try:
            closed = self._contact_assessment()
        except Exception as exc:
            return self._recover_open_hover(
                reach, f"contact baseline unavailable/inconsistent: {exc}")
        print(f"[servo] close contact={closed.state}: {closed.reason}")
        if not closed.contact:
            if closed.state != "UNKNOWN":
                return self._recover_open_hover(
                    reach, f"grasp not confirmed after close ({closed.state})")
            # At some wrist/shoulder combinations the closed tapes leave the
            # bottom of the frame at z=6 mm.  Raise the still-closed jaw only
            # far enough to restore sensor visibility, then apply the exact
            # same-run empty-jaw test.  This is a verification probe, not an
            # unverified transport: FREE/UNKNOWN immediately recovers open.
            self.slow_move(self._reach_pose(
                reach, CONTACT_PROBE_Z,
                gripper=config.GRIP_CLOSED)[0])
            try:
                closed = self._contact_assessment(allow_single_red=True)
            except Exception as exc:
                return self._recover_open_hover(
                    reach, f"probe contact check unavailable: {exc}")
            print(f"[servo] probe-lift contact={closed.state}: {closed.reason}")
            if not closed.contact:
                return self._recover_open_hover(
                    reach, "100mm verification probe found no retained object "
                           f"({closed.state})")

        # DESCEND and CLOSE were both admitted by the trained temporal shield.
        # From this point, the same-run physical obstruction is stronger
        # evidence than the policy's orientation-specific static jaw profile;
        # continue the already-started vertical verification lift.

        # Lift without ever opening. The previous code copied OBSERVATION_POSE,
        # whose gripper component is 90=open, and therefore dropped the object
        # before verification. Verify obstruction again at hover instead.
        self.slow_move(self._reach_pose(
            reach, self.VERIFY_Z, gripper=config.GRIP_CLOSED)[0])
        try:
            lifted = self._contact_assessment(allow_single_red=True)
        except Exception as exc:
            return self._recover_open_hover(
                reach, f"post-lift contact check unavailable: {exc}")
        print(f"[servo] lifted contact={lifted.state}: {lifted.reason}")
        if not self.contact_retained(closed, lifted):
            return self._recover_open_hover(
                reach, "object slipped or was not retained through 100mm lift "
                       f"(close residual={closed.residual_px}, "
                       f"lift residual={lifted.residual_px})")
        print("[servo] GRASP CONFIRMED: obstruction retained through 100mm lift")
        return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--align-only", action="store_true",
                   help="run the floor-plane alignment servo and stop (no grasp)")
    p.add_argument("--start-reach", type=int, default=0,
                   help="start near a previously observed safe hover reach")
    p.add_argument("--candidate-index", type=int, default=0,
                   help="ranked object to grasp; 0 is the nearest candidate")
    p.add_argument("--reject-count", type=int, default=0,
                   help="apply this many 'not that one' vetoes before grasping")
    p.add_argument("--autonomous-from-home", action="store_true",
                   help="HOME, search verified floor views, select, align, and grasp")
    _args = p.parse_args()
    if _args.candidate_index < 0 or _args.reject_count < 0:
        p.error("candidate index and reject count must be non-negative")
    if _args.candidate_index and _args.reject_count:
        p.error("use either --candidate-index or --reject-count, not both")
    from arm_session import ArmSessionClient
    from floor_grasp import CandidateSelector
    servo = FloorServo(ArmSessionClient(), FloorHomography.load())
    if _args.autonomous_from_home:
        if (_args.align_only or _args.start_reach or _args.candidate_index
                or _args.reject_count):
            p.error("--autonomous-from-home does not accept manual target/reach options")
        reach, selected = servo.search_from_home()
        if selected is None:
            print("[servo] autonomous search found no reachable object")
            return False
        aligned_reach = servo.align(start_reach=reach, selected=selected)
        if aligned_reach is None:
            print("[servo] autonomous alignment failed; not grasping")
            return False
        return servo.grasp(aligned_reach)

    frame = _fresh_frame(discard=1)
    scene, _ = servo.scene_detector.scene(frame)
    if not scene.ranked:
        print("[servo] no portable object detected; not moving")
        return False
    for index, candidate in enumerate(scene.ranked):
        print(f"[servo] candidate #{index} center={candidate.center} "
              f"bbox={candidate.bbox} area={candidate.area:.0f}")
    if _args.candidate_index >= len(scene.ranked):
        print(f"[servo] candidate #{_args.candidate_index} does not exist; "
              f"only {len(scene.ranked)} detected")
        return False
    selected = scene.ranked[_args.candidate_index]
    if _args.reject_count:
        selector = CandidateSelector(
            reject_radius_px=(config.FLOOR_REJECT_RADIUS_RATIO
                              * math.hypot(frame.shape[1], frame.shape[0])))
        selected = selector.choose(scene.ranked)
        for rejection in range(_args.reject_count):
            if selected is None:
                break
            print(f"[servo] REJECT #{rejection + 1}: center={selected.center}")
            selector.reject(selected)
            selected = selector.choose(scene.ranked)
        if selected is None:
            print("[servo] all detected objects were rejected; not moving")
            return False
    print(f"[servo] SELECTED center={selected.center}")
    wp = servo.align(start_reach=_args.start_reach, selected=selected)
    if wp is None:
        print("[servo] alignment failed; not grasping")
        return False
    if _args.align_only:
        print(f"[servo] align-only: aligned at wp={wp}, stopping before grasp")
        return True
    return servo.grasp(wp)


if __name__ == "__main__":
    main()
