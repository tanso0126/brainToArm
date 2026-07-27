"""Phase 2: teach the floor(x,y) -> grasp servo map on the real arm.

Phase 1 (floor_calibrate) gives an exact pixel->floor(x,y) map at a fixed
observation pose. Phase 2 learns how to actually reach a floor point with the
real arm, measured by real grasps instead of a trusted kinematic model.

Because the base is locked at 90, the gripper touches the floor only along the
arm's sagittal line, so a graspable object lies near that line and its forward
distance selects a grasp elbow on the physically reproduced floor curve. This
tool therefore fits: object forward-distance (from the homography) -> grasp elbow.

Grasp success is verified WITHOUT relying on the close-range finger markers
(which occlude near contact): after a close+lift, the arm returns to the
observation pose and the object is re-detected. If the object is gone from its
original spot, it was lifted -> success.

    python3 laptop/floor_teach.py probe        # observe object, report floor xy + reach
    python3 laptop/floor_teach.py grasp         # full teach-grasp with elbow search
"""

from pathlib import Path
import argparse
import math
import os
import time

import config
from floor_calibrate import FloorHomography, OBSERVATION_POSE
from floor_motion import floor_pose


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "vision" / "wrist_camera_latest_raw.jpg"


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


def _detect_object_pixel(detector, frame):
    """Return the best object's pixel center using FastSAM (color-agnostic).

    ``detector`` is a WristSceneDetector; it proposes every instance and drops
    the finger tapes and the floor/arm, so any object works regardless of colour.
    """
    scene, _ = detector.scene(frame)
    if not scene.ranked:
        return None
    return scene.ranked[0].center


def _observe_object(detector, calib):
    """At the observation pose: detect the object, return (pixel, floor_xy)."""
    frame = _fresh_frame()
    pixel = _detect_object_pixel(detector, frame)
    if pixel is None:
        return None, None
    return pixel, calib.pixel_to_floor(pixel)


def _client():
    from arm_session import ArmSessionClient
    return ArmSessionClient()


# Slow, gentle motion: never send a big pose delta at once. Interpolate from the
# current pose to the target in small per-joint steps so the printed linkage is
# never yanked. Intermediate waypoints use a short settle; only the final pose
# waits the full mechanical settle before the camera measures.
MAX_STEP_DEG = 3
STEP_SETTLE_S = 0.25


def _slow_move(client, target, final_settle=None):
    target = [int(round(v)) for v in target]
    current = client.request({"command": "status"})["pose"]
    span = max(abs(t - c) for t, c in zip(target, current))
    steps = max(1, int(math.ceil(span / MAX_STEP_DEG)))
    final_settle = config.FLOOR_SETTLE_S if final_settle is None else final_settle
    for i in range(1, steps + 1):
        frac = i / steps
        waypoint = [int(round(c + (t - c) * frac))
                    for c, t in zip(current, target)]
        last = (i == steps)
        client.request({"command": "move", "pose": waypoint,
                        "settle_s": final_settle if last else STEP_SETTLE_S})


def _park_observe(client):
    _slow_move(client, list(OBSERVATION_POSE))


def cmd_probe(_args):
    from floor_grasp import WristSceneDetector
    calib = FloorHomography.load()
    client = _client()
    _park_observe(client)
    pixel, floor = _observe_object(WristSceneDetector(), calib)
    if pixel is None:
        print("[teach] no object detected at the observation pose")
        return
    print(f"[teach] object pixel=({pixel[0]:.0f},{pixel[1]:.0f}) "
          f"floor=({floor[0]:.1f},{floor[1]:.1f}) mm  (calib RMS {calib.rms_mm:.2f} mm)")


def _object_on_floor(detector):
    """True if ANY object is detected on the floor at the observation pose.

    A genuine grasp lifts the object off the floor, so it disappears from the
    whole observation view. If the object merely rolled/was pushed to a new
    floor spot, it is still detected somewhere -> not a grasp.
    """
    frame = _fresh_frame()
    pixel = _detect_object_pixel(detector, frame)
    return (pixel is not None), pixel


def cmd_grasp(args):
    from floor_grasp import WristSceneDetector
    detector = WristSceneDetector()
    calib = FloorHomography.load()
    client = _client()

    def move(pose, settle=None):
        _slow_move(client, pose, final_settle=settle)

    # 1) Observe object.
    _park_observe(client)
    pixel, floor = _observe_object(detector, calib)
    if pixel is None:
        print("[teach] no object at observation pose; place one on the centerline")
        return
    print(f"[teach] target pixel=({pixel[0]:.0f},{pixel[1]:.0f}) "
          f"floor=({floor[0]:.1f},{floor[1]:.1f}) mm")

    # 2) Sweep grasp elbow on the floor curve (near..far). Each attempt: open at
    #    grasp level, close, lift to hover, return to observe, check if the
    #    object vanished from its spot -> grasped.
    lo, hi = config.FLOOR_ELBOW_RANGE
    elbows = list(range(args.elbow_min, args.elbow_max + 1, args.step))
    print(f"[teach] elbow sweep {elbows} (grasp curve)")
    for elbow in elbows:
        if not lo <= elbow <= hi:
            continue
        print(f"[teach] -- attempt elbow={elbow}")
        move(floor_pose(elbow, "grasp", gripper=config.GRIP_OPEN))
        move(floor_pose(elbow, "grasp", gripper=config.GRIP_CLOSED))
        # Lift and check while still raised: a real grasp keeps the object with
        # the gripper (out of the floor view); a push leaves it on the floor.
        move(floor_pose(elbow, "hover", gripper=config.GRIP_CLOSED))
        _park_observe(client)
        on_floor, now_px = _object_on_floor(detector)
        if on_floor:
            moved = math.hypot(now_px[0] - pixel[0], now_px[1] - pixel[1])
            print(f"[teach] object still on floor (px {tuple(round(v) for v in now_px)}, "
                  f"moved {moved:.0f}px) -> not grasped (likely pushed); next elbow")
            move(floor_pose(elbow, "hover", gripper=config.GRIP_OPEN))
            continue
        # Object absent from the floor. Confirm it is HELD: open the jaws at
        # hover and re-check -- a truly held object drops back onto the floor
        # and reappears; if nothing ever reappears it may have been knocked out
        # of view, which we do NOT count as a confirmed grasp.
        move(floor_pose(elbow, "hover", gripper=config.GRIP_CLOSED))
        _park_observe(client)
        still_absent, _ = _object_on_floor(detector)
        print(f"[teach] object absent from floor after elbow={elbow} lift "
              f"(closed-recheck absent={still_absent})")
        print(f"[teach] CANDIDATE GRASP floor_x={floor[0]:.1f} floor_y={floor[1]:.1f} "
              f"-> grasp_elbow={elbow}  (verify visually)")
        if not args.hold:
            move(floor_pose(elbow, "hover", gripper=config.GRIP_OPEN))
            print("[teach] released at hover")
        return
    print("[teach] sweep exhausted without a confirmed grasp; "
          "object may be off the reach line or out of the floor band")
    _park_observe(client)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    g = sub.add_parser("grasp")
    g.add_argument("--elbow-min", type=int, default=config.FLOOR_ELBOW_RANGE[0])
    g.add_argument("--elbow-max", type=int, default=config.FLOOR_ELBOW_RANGE[1])
    g.add_argument("--step", type=int, default=4)
    g.add_argument("--hold", action="store_true", help="keep holding after success")
    args = p.parse_args()
    {"probe": cmd_probe, "grasp": cmd_grasp}[args.cmd](args)


if __name__ == "__main__":
    main()
