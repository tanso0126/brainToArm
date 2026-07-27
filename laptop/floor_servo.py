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
SEARCH_DV_WINDOW_PX = (-220.0, 45.0)
SEARCH_CONFIRM_RADIUS_PX = 80.0
GRASP_INSET_RATIO = 0.35   # grasp inside the object, not on its trailing edge
DYNAMIC_CONTACT_MARGIN_PX = 10.0
CONTACT_RETENTION_RATIO = 0.50
CLIPPED_GRASP_DV_WINDOW_PX = (-52.0, -24.0)
CLIPPED_GRASP_VOTES = 2
EMPTY_CALIBRATION_Z = 0.100


def _fresh_frame(discard=None, timeout=8.0):
    import cv2
    discard = (config.FLOOR_SETTLE_DISCARD_FRAMES + 1) if discard is None else discard
    prev = None
    seen = 0
    deadline = time.monotonic() + timeout
    while seen < discard and time.monotonic() < deadline:
        try:
            m = RAW.stat().st_mtime_ns
        except FileNotFoundError:
            time.sleep(0.05); continue
        if m != prev:
            prev = m; seen += 1
        time.sleep(0.05)
    frame = cv2.imread(str(RAW))
    if frame is None:
        raise RuntimeError("no wrist frame; is the camera publisher running?")
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
        self.last_target_clipped = False
        self.locked_target_clipped = False

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
            z = arm_fk.tool_position([90, mid, elbow, wrist_pitch, 90, 170])[2]
            if z > z_m:
                lo = mid
            else:
                hi = mid
        shoulder = int(round((lo + hi) / 2))
        shoulder = max(config.SERVO_MIN[J_SHOULDER],
                       min(config.SERVO_MAX[J_SHOULDER], shoulder))
        return [90, shoulder, elbow, int(wrist_pitch), int(gripper), 170]

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
    VERIFY_Z = 0.080

    @staticmethod
    def candidate_grasp_point(candidate):
        """Size-aware interior pinch line, inset from the trailing bbox edge."""
        _x, y, _width, height = candidate.bbox
        return np.array((candidate.center[0],
                         y + (1.0 - GRASP_INSET_RATIO) * height), dtype=float)

    @classmethod
    def search_choice(cls, scene):
        """Return the best currently approachable candidate, or ``None``.

        This is geometry, not an object/location prior: with base yaw fixed, a
        graspable floor object must enter a narrow horizontal band between the
        jaws and approach their row from above during the reach sweep.
        """
        if scene.gripper is None:
            return None
        midpoint = np.asarray(scene.gripper.center, dtype=float)
        eligible = []
        for candidate in scene.ranked:
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
            scene, _ = self.scene_detector.scene(_fresh_frame())
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
            empty_observation, _ = self.marker_detector.detect(_fresh_frame())
            self.slow_move(open_calibration)
            self.slow_move(pose)  # return open to the target-visible hover
            if empty_observation.gripper is None:
                print("[servo] SEARCH empty-close calibration lost markers")
                continue
            self.empty_closed_opening_px = float(
                empty_observation.gripper.opening_px)

            # Closing/opening can perturb segmentation; require the chosen
            # object once more before returning control to alignment.
            final_scene, _ = self.scene_detector.scene(_fresh_frame(discard=2))
            final_candidate = self.search_choice(final_scene)
            if final_candidate is None:
                print("[servo] SEARCH target lost after empty-close calibration")
                continue
            final_distance = float(np.linalg.norm(
                self.candidate_grasp_point(final_candidate) - reference))
            if final_distance > SEARCH_CONFIRM_RADIUS_PX:
                print("[servo] SEARCH target identity changed after calibration")
                continue
            print(f"[servo] SEARCH FOUND reach={reach} "
                  f"center={final_candidate.center} "
                  f"confirm_delta={final_distance:.1f}px "
                  f"empty-close={self.empty_closed_opening_px:.1f}px")
            return reach, final_candidate

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
        if not scene.ranked:
            obj = None
            self.last_target_clipped = False
        elif reference is None:
            candidate = scene.ranked[0]
            obj = self.candidate_grasp_point(candidate)
            self.last_target_clipped = (
                candidate.bbox[1] + candidate.bbox[3]
                >= frame.shape[0] - config.FLOOR_CAND_BORDER_MARGIN_PX)
        else:
            candidate = min(
                scene.ranked,
                key=lambda candidate: np.linalg.norm(
                    self.candidate_grasp_point(candidate)
                    - reference))
            obj = self.candidate_grasp_point(candidate)
            self.last_target_clipped = (
                candidate.bbox[1] + candidate.bbox[3]
                >= frame.shape[0] - config.FLOOR_CAND_BORDER_MARGIN_PX)
        mid = None if obs.gripper is None else np.array(obs.gripper.center)
        return obj, mid

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
        ACCEPT_DV, ACCEPT_DU = 26.0, 55.0
        reach = int(np.clip(start_reach, 0, self.REACH_MAX))
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
            if verbose:
                print(f"[servo] it={it} reach={reach}(wp={wp},el={elbow}) "
                      f"obj_px={np.round(obj)} marker_px={np.round(mid)} "
                      f"du={du:.0f} dv={dv:.0f}")
            if best is None or abs(dv) < best[0]:
                best = (abs(dv), reach)
            if abs(dv) <= ACCEPT_DV and abs(du) <= ACCEPT_DU:
                self.locked_target_center = obj.copy()
                self.locked_target_clipped = False
                print(f"[servo] object within the jaw span at hover "
                      f"(du={du:.0f},dv={dv:.0f}) reach={reach} -> descend & close")
                return reach
            clipped_ready = (
                self.last_target_clipped
                and abs(du) <= ACCEPT_DU
                and CLIPPED_GRASP_DV_WINDOW_PX[0] <= dv
                <= CLIPPED_GRASP_DV_WINDOW_PX[1])
            if clipped_ready:
                clipped_votes = (clipped_votes + 1
                                 if previous_clipped_dv is not None
                                 and abs(dv - previous_clipped_dv) <= 8.0 else 1)
                previous_clipped_dv = dv
                if clipped_votes >= CLIPPED_GRASP_VOTES:
                    self.locked_target_center = obj.copy()
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
            step = 1 if abs(dv) < 120 else MAX_STEP_DEG
            delta = step if dv < 0 else -step
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
            target = None
            if target_visible and scene.ranked:
                reference = self.locked_target_center
                candidate = (scene.ranked[0] if reference is None else min(
                    scene.ranked,
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

    def _contact_assessment(self):
        observation, _ = self.marker_detector.detect(_fresh_frame())
        if self.empty_closed_opening_px is not None:
            from visual_contact import ContactAssessment
            if observation.gripper is None:
                return ContactAssessment(
                    "UNKNOWN", config.GRIP_CLOSED, None,
                    self.empty_closed_opening_px, None,
                    DYNAMIC_CONTACT_MARGIN_PX,
                    "both finger markers are required")
            observed = float(observation.gripper.opening_px)
            residual = observed - self.empty_closed_opening_px
            contact = residual > DYNAMIC_CONTACT_MARGIN_PX
            return ContactAssessment(
                "CONTACT" if contact else "FREE", config.GRIP_CLOSED,
                observed, self.empty_closed_opening_px, residual,
                DYNAMIC_CONTACT_MARGIN_PX,
                (f"jaw remained {residual:.1f}px wider than same-run empty "
                 f"endpoint (threshold {DYNAMIC_CONTACT_MARGIN_PX:.1f}px)"
                 if contact else
                 f"jaw returned to same-run empty endpoint "
                 f"(residual {residual:.1f}px)"))
        from visual_contact import JawBaseline
        baseline = JawBaseline.load()
        return baseline.assess(config.GRIP_CLOSED, observation)

    @staticmethod
    def contact_retained(closed, lifted):
        """Reject an object that obstructed close but slipped out during lift."""
        if not closed.contact or not lifted.contact:
            return False
        if closed.residual_px is None or lifted.residual_px is None:
            return False
        required = max(DYNAMIC_CONTACT_MARGIN_PX,
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
            return self._recover_open_hover(
                reach, f"grasp not confirmed after close ({closed.state})")

        if not self._authorize_macro(
                TaskAction.LIFT,
                self._reach_pose(
                    reach, self.GRASP_Z, gripper=config.GRIP_CLOSED)[0],
                target_visible=False):
            return self._recover_open_hover(
                reach, "trained policy did not authorize lift")

        # Lift without ever opening. The previous code copied OBSERVATION_POSE,
        # whose gripper component is 90=open, and therefore dropped the object
        # before verification. Verify obstruction again at hover instead.
        self.slow_move(self._reach_pose(
            reach, self.VERIFY_Z, gripper=config.GRIP_CLOSED)[0])
        try:
            lifted = self._contact_assessment()
        except Exception as exc:
            return self._recover_open_hover(
                reach, f"post-lift contact check unavailable: {exc}")
        print(f"[servo] lifted contact={lifted.state}: {lifted.reason}")
        if not self.contact_retained(closed, lifted):
            return self._recover_open_hover(
                reach, "object slipped or was not retained through 80mm lift "
                       f"(close residual={closed.residual_px}, "
                       f"lift residual={lifted.residual_px})")
        print("[servo] GRASP CONFIRMED: obstruction retained through 80mm lift")
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
