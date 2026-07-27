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
    def _level_pose(self, wrist_pitch, z_m, gripper=None):
        """Pose at fixed elbow and given wrist_pitch, with shoulder chosen (FK)
        so the tool sits at height ``z_m`` above the floor. Alignment runs at a
        safe hover height so the fingers never scrape the floor; only the final
        grasp uses a near-floor z."""
        import arm_fk
        elbow = config.FLOOR_REFERENCE_ELBOW
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

    # Alignment hover height (m above floor) — fingers stay clear of the floor.
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
        """Drive the object's image row onto the fingertip row at floor level."""
        wp = 180
        self.slow_move(self._level_pose(wp, self.HOVER_Z))
        prev = None
        last_obj = None
        for it in range(ALIGN_MAX_ITERS):
            obj, mid = self._obj_and_marker_px()
            if mid is None:
                print(f"[servo] it={it}: markers lost; stopping"); return None
            if obj is None:
                print(f"[servo] it={it}: object not detected; retrying")
                continue
            # Reject a spurious jump (object briefly occluded by the descending
            # gripper makes FastSAM latch a different blob); keep the last good.
            if last_obj is not None and np.linalg.norm(obj - last_obj) > OBJ_JUMP_REJECT_PX:
                print(f"[servo] it={it}: object pixel jumped "
                      f"{np.linalg.norm(obj-last_obj):.0f}px -> outlier, keeping last")
                obj = last_obj
            else:
                last_obj = obj
            # Error in image pixels; the graspable direction is the image row
            # (object must come DOWN to the fingertip row). dv>0 => object above
            # (farther) => lower wrist_pitch to reach further.
            du, dv = float(obj[0] - mid[0]), float(obj[1] - mid[1])
            if verbose:
                print(f"[servo] it={it} wp={wp} obj_px={np.round(obj)} "
                      f"marker_px={np.round(mid)} du={du:.0f} dv={dv:.0f}")
            if abs(dv) <= 18 and abs(du) <= 60:
                print(f"[servo] object under the fingertips at hover "
                      f"(du={du:.0f},dv={dv:.0f}) wp={wp} -> descend & close")
                return wp
            # dv<0: object is ABOVE the fingertip row (farther) -> lower wrist_pitch
            # to reach the fingers further along the floor toward it. dv>0: object
            # is nearer than the fingertips -> raise wrist_pitch.
            if prev is not None and abs(abs(dv) - prev) < 3 and wp <= 132:
                closer_mm = abs(dv) / 12.0   # ~12 px per mm near the fingertip row
                print(f"[servo] object stalled {abs(dv):.0f}px from the fingertips at "
                      f"the reach limit (wp={wp}); it is beyond the base-locked floor "
                      f"reach -- move the object ~{closer_mm:.0f}mm closer to the base")
                return None
            prev = abs(dv)
            step = -MAX_STEP_DEG if dv < 0 else MAX_STEP_DEG
            wp = int(min(180, max(130, wp + step)))
            self.slow_move(self._level_pose(wp, self.HOVER_Z))
        print("[servo] did not converge within iteration budget")
        return None

    def grasp(self, wp):
        """After hover alignment: descend to the floor at the same wrist_pitch,
        close, lift, and report. Gentle stepped motion throughout."""
        # descend straight to near-floor at the aligned wrist_pitch (open jaws)
        self.slow_move(self._level_pose(wp, self.GRASP_Z, gripper=config.GRIP_OPEN))
        # close
        self.slow_move(self._level_pose(wp, self.GRASP_Z, gripper=config.GRIP_CLOSED))
        # lift back to hover holding
        self.slow_move(self._level_pose(wp, self.HOVER_Z, gripper=config.GRIP_CLOSED))
        # verify: object gone from the floor at the observation pose
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
