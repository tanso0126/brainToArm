"""Closed-loop floor-coordinate visual servo (no blind elbow sweeping).

At the fixed observation pose the wrist camera sees BOTH the object and the two
finger-tape markers. The Phase-1 homography maps any pixel to floor(x,y) in one
metric frame, so the object and the gripper's marker-midpoint live in the same
coordinates. We:

  1. Measure a local 2x2 Jacobian J = d(gripper floor xy)/d(elbow, wrist_pitch)
     by nudging each joint a few degrees and re-observing (real hardware, no
     trusted kinematic model).
  2. Drive the gripper's floor projection onto the object with delta =
     J^-1 (object - gripper), applied in small clamped steps and RE-MEASURED each
     iteration, so the gripper tip visually chases the object.
  3. When aligned in the floor plane, descend on the floor curve and close.

This replaces trial-and-error: every motion is computed from the measured
Jacobian and the metric floor error.
"""

from pathlib import Path
import argparse
import math
import time

import numpy as np

import config
from floor_calibrate import FloorHomography, OBSERVATION_POSE


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "vision" / "wrist_camera_latest_raw.jpg"

# Servo joints we command at the observation pose: elbow (J2) and wrist_pitch
# (J3 index). Base stays 90, wrist_roll level, gripper open during alignment.
J_SHOULDER, J_ELBOW, J_WRIST = 1, 2, 3

MAX_STEP_DEG = 3          # gentle per-waypoint joint change
STEP_SETTLE_S = 0.25
ALIGN_TOL_MM = 8.0        # floor-plane convergence tolerance
ALIGN_MAX_ITERS = 26
JAC_PERTURB_DEG = 4       # nudge size for Jacobian measurement
OBJ_JUMP_REJECT_PX = 160  # reject a frame whose object pixel jumps this far


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

    # Alignment hover height (m above floor). Kept low so the hover alignment is
    # close to the grasp height and the fingertip floor point barely shifts on
    # the final descent, while still clearing a short tabletop object.
    HOVER_Z = 0.035
    GRASP_Z = 0.006

    def _obj_and_marker_px(self):
        """Return (object_pixel, marker_midpoint_pixel) at the current pose."""
        frame = _fresh_frame()
        scene, _ = self.scene_detector.scene(frame)
        obs, _ = self.marker_detector.detect(frame)
        obj = None if not scene.ranked else np.array(scene.ranked[0].center)
        mid = None if obs.gripper is None else np.array(obs.gripper.center)
        return obj, mid

    def align(self, verbose=True):
        """Drive the object's image row onto the fingertip row at hover, using a
        single forward-reach scalar (wrist_pitch then elbow)."""
        # The open jaws span ~288px, so an object within ~half that of the
        # fingertip midpoint is already inside the jaws and graspable. Near
        # contact the object detection gets noisy (the gripper occludes it), so
        # we accept the first frame that lands the object within the jaw span and
        # also remember the best reach seen to fall back on.
        ACCEPT_DV, ACCEPT_DU = 110.0, 70.0
        reach = 0
        pose, wp, elbow = self._reach_pose(reach, self.HOVER_Z)
        self.slow_move(pose)
        last_obj = None
        best = None  # (abs(dv), reach)
        for it in range(ALIGN_MAX_ITERS):
            obj, mid = self._obj_and_marker_px()
            if mid is None:
                print(f"[servo] it={it}: markers lost; stopping"); break
            if obj is None:
                print(f"[servo] it={it}: object not detected; retrying")
                continue
            if last_obj is not None and np.linalg.norm(obj - last_obj) > OBJ_JUMP_REJECT_PX:
                print(f"[servo] it={it}: object pixel jumped "
                      f"{np.linalg.norm(obj-last_obj):.0f}px -> outlier, ignoring frame")
                continue
            last_obj = obj
            du, dv = float(obj[0] - mid[0]), float(obj[1] - mid[1])
            if verbose:
                print(f"[servo] it={it} reach={reach}(wp={wp},el={elbow}) "
                      f"obj_px={np.round(obj)} marker_px={np.round(mid)} "
                      f"du={du:.0f} dv={dv:.0f}")
            if best is None or abs(dv) < best[0]:
                best = (abs(dv), reach)
            if abs(dv) <= ACCEPT_DV and abs(du) <= ACCEPT_DU:
                print(f"[servo] object within the jaw span at hover "
                      f"(du={du:.0f},dv={dv:.0f}) reach={reach} -> descend & close")
                return reach
            if reach >= self.REACH_MAX:
                break
            delta = MAX_STEP_DEG if dv < 0 else -MAX_STEP_DEG
            reach = int(min(self.REACH_MAX, max(0, reach + delta)))
            pose, wp, elbow = self._reach_pose(reach, self.HOVER_Z)
            self.slow_move(pose)
        if best is not None and best[0] <= 160:
            print(f"[servo] using best-seen reach={best[1]} (|dv|={best[0]:.0f}px); "
                  "detection got noisy near contact -> attempt grasp there")
            return best[1]
        print("[servo] object never came within the jaw span; "
              "move it a little closer to the base")
        return None

    def grasp(self, reach):
        """Tracked descent: step the tool height down while re-observing and
        nudging reach so the object stays under the fingertips all the way to the
        floor (the fingertip floor point shifts with height, so a fixed reach
        would miss). Then close, lift, and verify by object-disappearance."""
        last_obj = None
        for z in (0.030, 0.024, 0.018, 0.012, self.GRASP_Z):
            pose, wp, elbow = self._reach_pose(reach, z, gripper=config.GRIP_OPEN)
            self.slow_move(pose)
            obj, mid = self._obj_and_marker_px()
            if obj is not None and mid is not None:
                if last_obj is None or np.linalg.norm(obj - last_obj) <= OBJ_JUMP_REJECT_PX:
                    last_obj = obj
                    du, dv = float(obj[0] - mid[0]), float(obj[1] - mid[1])
                    print(f"[servo] descend z={z*1000:.0f}mm reach={reach} "
                          f"du={du:.0f} dv={dv:.0f}")
                    if dv < -40 and reach < self.REACH_MAX:      # object still beyond
                        reach = int(min(self.REACH_MAX, reach + MAX_STEP_DEG))
                    elif dv > 40 and reach > 0:                  # overshot toward base
                        reach = int(max(0, reach - MAX_STEP_DEG))
                else:
                    print(f"[servo] descend z={z*1000:.0f}mm: object occluded, hold reach")
        # close at the floor, lift, verify
        self.slow_move(self._reach_pose(reach, self.GRASP_Z, gripper=config.GRIP_CLOSED)[0])
        self.slow_move(self._reach_pose(reach, self.HOVER_Z, gripper=config.GRIP_CLOSED)[0])
        self.slow_move(list(OBSERVATION_POSE))
        obj = self.object_floor()
        if obj is None:
            print("[servo] GRASP CONFIRMED: object no longer on the floor")
            return True
        print(f"[servo] grasp not confirmed: object still on floor at {np.round(obj,1)}")
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--align-only", action="store_true",
                   help="run the floor-plane alignment servo and stop (no grasp)")
    _args = p.parse_args()
    from arm_session import ArmSessionClient
    servo = FloorServo(ArmSessionClient(), FloorHomography.load())
    wp = servo.align()
    if wp is None:
        print("[servo] alignment failed; not grasping")
        return
    if _args.align_only:
        print(f"[servo] align-only: aligned at wp={wp}, stopping before grasp")
        return
    servo.grasp(wp)


if __name__ == "__main__":
    main()
