"""Look-then-move floor calibration for the wrist camera.

The eye-in-hand camera moves with the arm, so a pixel->floor mapping is only
valid at one fixed pose. This module therefore uses a fixed OBSERVATION pose:
the arm parks there, the camera sees the whole tabletop workspace, and a planar
homography maps image pixels to floor coordinates. Because the floor is a plane
and the camera is a pinhole, that mapping is an exact projective homography -- no
camera intrinsics, no arm forward-kinematics model, and no monocular depth guess
are needed. This is what turns the known floor plane into real depth.

Two calibration phases:

  Phase 1 (this file): pixel -> floor(x, y) homography at the observation pose,
    solved from a printed checkerboard (many corners, robust) or, as a fallback,
    from >=4 manually measured object points.

  Phase 2 (floor_grasp, later): floor(x, y) -> grasp servo pose, fitted from real
    grasps whose success is confirmed by the visual jaw-contact signal. That
    measures the *real* arm's reach and IK instead of trusting the sim model.

The homography is stored in a floor frame defined by the checkerboard; the same
frame is reused in Phase 2, so its absolute placement relative to the base does
not matter as long as the board stays put during Phase 1.

Usage::

    # 1) clear the arm path; park at the observation pose and publish a frame
    python3 laptop/floor_calibrate.py observe

    # 2) put a checkerboard flat in the workspace, then solve the homography
    python3 laptop/floor_calibrate.py homography --inner-cols 9 --inner-rows 6 --square-mm 25

    # or, with no checkerboard, add >=4 measured points (object at known floor xy)
    python3 laptop/floor_calibrate.py add-point --x 120 --y 0
    python3 laptop/floor_calibrate.py solve-points
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple
import argparse
import json
import time

import numpy as np

import config


ROOT = Path(__file__).resolve().parents[1]
DEBUG_DIR = ROOT / "data" / "vision"
LATEST_RAW_PATH = DEBUG_DIR / "wrist_camera_latest_raw.jpg"
CALIB_DIR = ROOT / "data" / "calibration"
HOMOGRAPHY_PATH = CALIB_DIR / "wrist_floor_homography.json"
POINTS_PATH = CALIB_DIR / "wrist_floor_points.json"

# Fixed observation pose (six servo degrees). Chosen so the wrist camera views
# the reachable tabletop workspace with the fingers visible at the frame bottom.
OBSERVATION_POSE = [90, 112, 90, 158, 90, 170]


# ======================================================================
# Pure homography math (unit-testable; no camera, no arm)
# ======================================================================
def solve_homography(image_points, floor_points):
    """Return 3x3 H mapping image pixels -> floor coordinates.

    ``image_points`` and ``floor_points`` are matched (N,2) arrays, N>=4. Uses a
    direct linear transform; raises if the correspondences are degenerate.
    """
    image_points = np.asarray(image_points, dtype=np.float64)
    floor_points = np.asarray(floor_points, dtype=np.float64)
    if image_points.shape != floor_points.shape or image_points.ndim != 2 \
            or image_points.shape[1] != 2:
        raise ValueError("image/floor points must be matching (N,2) arrays")
    if len(image_points) < 4:
        raise ValueError("homography needs at least four correspondences")
    rows = []
    for (u, v), (x, y) in zip(image_points, floor_points):
        rows.append([u, v, 1, 0, 0, 0, -x * u, -x * v, -x])
        rows.append([0, 0, 0, u, v, 1, -y * u, -y * v, -y])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    h = vh[-1].reshape(3, 3)
    if abs(h[2, 2]) < 1e-12:
        raise ValueError("degenerate homography (check point spread)")
    h = h / h[2, 2]
    return h


def apply_homography(h, image_point):
    """Map one image pixel (u,v) to floor (x,y) through H."""
    h = np.asarray(h, dtype=np.float64)
    u, v = float(image_point[0]), float(image_point[1])
    denom = h[2, 0] * u + h[2, 1] * v + h[2, 2]
    if abs(denom) < 1e-12:
        raise ValueError("point maps to the plane at infinity")
    x = (h[0, 0] * u + h[0, 1] * v + h[0, 2]) / denom
    y = (h[1, 0] * u + h[1, 1] * v + h[1, 2]) / denom
    return (float(x), float(y))


def reprojection_rms(h, image_points, floor_points):
    """RMS floor-coordinate error of H over the correspondences."""
    errors = []
    for image_point, floor_point in zip(image_points, floor_points):
        x, y = apply_homography(h, image_point)
        errors.append((x - floor_point[0]) ** 2 + (y - floor_point[1]) ** 2)
    return float(np.sqrt(np.mean(errors))) if errors else float("nan")


@dataclass
class FloorHomography:
    matrix: List[List[float]]
    observation_pose: List[int]
    rms_mm: float
    source: str
    created_at: str

    def pixel_to_floor(self, image_point):
        return apply_homography(np.asarray(self.matrix), image_point)

    def save(self, path=HOMOGRAPHY_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path=HOMOGRAPHY_PATH):
        path = Path(path)
        if not path.exists():
            raise RuntimeError(
                f"floor homography missing: {path}. Run floor_calibrate.py first.")
        return cls(**json.loads(path.read_text(encoding="utf-8")))


# ======================================================================
# Camera / arm helpers (hardware side)
# ======================================================================
def _park_at_observation_pose():
    from arm_session import ArmSessionClient
    client = ArmSessionClient()
    print(f"[floor-cal] parking at observation pose {OBSERVATION_POSE}")
    client.request({"command": "move", "pose": OBSERVATION_POSE,
                    "require_camera": True,
                    "settle_s": config.FLOOR_SETTLE_S})
    return client


def _fresh_raw_frame(min_new=None, timeout=8.0):
    import cv2
    min_new = (config.FLOOR_SETTLE_DISCARD_FRAMES + 1) if min_new is None else min_new
    prev = None
    seen = 0
    deadline = time.monotonic() + timeout
    while seen < min_new and time.monotonic() < deadline:
        try:
            mtime = LATEST_RAW_PATH.stat().st_mtime_ns
        except FileNotFoundError:
            time.sleep(0.05)
            continue
        if mtime != prev:
            prev = mtime
            seen += 1
        time.sleep(0.05)
    frame = cv2.imread(str(LATEST_RAW_PATH))
    if frame is None:
        raise RuntimeError("no wrist frame; is wrist_vision/publisher running?")
    return frame


def detect_checkerboard(frame, inner_cols, inner_rows, square_mm):
    """Return (image_points, floor_points) for a flat checkerboard on the floor."""
    import cv2
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, (inner_cols, inner_rows),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        raise RuntimeError(
            "checkerboard not found; check --inner-cols/--inner-rows, lighting, "
            "and that the whole board is flat and fully in view")
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    image_points = corners.reshape(-1, 2)
    floor_points = np.array(
        [[c * square_mm, r * square_mm]
         for r in range(inner_rows) for c in range(inner_cols)],
        dtype=np.float64)
    return image_points, floor_points


# ======================================================================
# CLI
# ======================================================================
def _cmd_observe(_args):
    _park_at_observation_pose()
    frame = _fresh_raw_frame()
    out = DEBUG_DIR / "floor_calib_observation.jpg"
    import cv2
    cv2.imwrite(str(out), frame)
    print(f"[floor-cal] observation frame saved: {out} ({frame.shape[1]}x{frame.shape[0]})")
    print("[floor-cal] now place a checkerboard flat in the workspace and run "
          "`homography`, or use add-point/solve-points.")


def _cmd_homography(args):
    frame = _fresh_raw_frame()
    image_points, floor_points = detect_checkerboard(
        frame, args.inner_cols, args.inner_rows, args.square_mm)
    h = solve_homography(image_points, floor_points)
    rms = reprojection_rms(h, image_points, floor_points)
    calib = FloorHomography(
        matrix=h.tolist(), observation_pose=list(OBSERVATION_POSE),
        rms_mm=rms, source=f"checkerboard {args.inner_cols}x{args.inner_rows}"
        f"@{args.square_mm}mm",
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    calib.save()
    print(f"[floor-cal] homography solved from {len(image_points)} corners, "
          f"reprojection RMS = {rms:.2f} mm")
    print(f"[floor-cal] saved {HOMOGRAPHY_PATH}")


def _load_points():
    if POINTS_PATH.exists():
        return json.loads(POINTS_PATH.read_text(encoding="utf-8"))
    return []


def _save_points(points):
    POINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    POINTS_PATH.write_text(json.dumps(points, indent=2) + "\n", encoding="utf-8")


def _cmd_add_point(args):
    """Detect the yellow object's pixel now; pair it with a measured floor xy."""
    from wrist_vision import WristDetector
    frame = _fresh_raw_frame()
    observation, _ = WristDetector().detect(frame)
    if observation.target is None:
        raise RuntimeError("no object detected; place one compact object in view")
    pixel = [round(v, 1) for v in observation.target.center]
    points = _load_points()
    points.append({"pixel": pixel, "floor": [args.x, args.y]})
    _save_points(points)
    print(f"[floor-cal] point #{len(points)}: pixel={pixel} floor=({args.x},{args.y}) mm")
    if len(points) < 4:
        print(f"[floor-cal] need {4 - len(points)} more before solve-points")


def _cmd_solve_points(_args):
    points = _load_points()
    if len(points) < 4:
        raise RuntimeError(f"need >=4 measured points, have {len(points)}")
    image_points = [p["pixel"] for p in points]
    floor_points = [p["floor"] for p in points]
    h = solve_homography(image_points, floor_points)
    rms = reprojection_rms(h, image_points, floor_points)
    FloorHomography(
        matrix=h.tolist(), observation_pose=list(OBSERVATION_POSE), rms_mm=rms,
        source=f"{len(points)} measured points",
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S")).save()
    print(f"[floor-cal] homography from {len(points)} points, RMS={rms:.2f} mm")
    print(f"[floor-cal] saved {HOMOGRAPHY_PATH}")


def _cmd_test(args):
    calib = FloorHomography.load()
    floor = calib.pixel_to_floor((args.u, args.v))
    print(f"[floor-cal] pixel ({args.u},{args.v}) -> floor "
          f"({floor[0]:.1f}, {floor[1]:.1f}) mm  (RMS {calib.rms_mm:.2f} mm)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("observe")
    hom = sub.add_parser("homography")
    hom.add_argument("--inner-cols", type=int, required=True,
                     help="inner corners per row (squares-1)")
    hom.add_argument("--inner-rows", type=int, required=True,
                     help="inner corners per column (squares-1)")
    hom.add_argument("--square-mm", type=float, required=True)
    ap = sub.add_parser("add-point")
    ap.add_argument("--x", type=float, required=True, help="measured floor x (mm)")
    ap.add_argument("--y", type=float, required=True, help="measured floor y (mm)")
    sub.add_parser("solve-points")
    t = sub.add_parser("test")
    t.add_argument("--u", type=float, required=True)
    t.add_argument("--v", type=float, required=True)
    args = parser.parse_args()
    {"observe": _cmd_observe, "homography": _cmd_homography,
     "add-point": _cmd_add_point, "solve-points": _cmd_solve_points,
     "test": _cmd_test}[args.cmd](args)


if __name__ == "__main__":
    main()
