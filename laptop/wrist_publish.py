"""Headless continuous wrist-frame publisher (no GUI).

Owns the AVFoundation webcam and republishes raw + annotated frames to the same
paths wrist_vision.py uses, so floor_grasp/visual_contact (which read the FILE)
keep working without a cv2 window. Safe to run when no other camera owner exists.
"""
import sys, time
sys.path.insert(0, "laptop")

import config
from wrist_vision import (
    NamedAVFoundationCamera, WristDetector, annotate,
    _atomic_write_jpeg, LATEST_RAW_PATH, LATEST_PREVIEW_PATH)


def main():
    cam = NamedAVFoundationCamera()
    det = WristDetector()
    for _ in range(config.WRIST_CAMERA_WARMUP_FRAMES):
        ok, _f = cam.read()
        if not ok:
            raise RuntimeError("camera stopped during warmup")
    print("[publish] READY", flush=True)
    last = 0.0
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                raise RuntimeError("camera read failed")
            obs, _m = det.detect(frame)
            now = time.monotonic()
            if now - last >= 0.15:
                _atomic_write_jpeg(LATEST_RAW_PATH, frame)
                _atomic_write_jpeg(LATEST_PREVIEW_PATH, annotate(frame, obs))
                last = now
    finally:
        cam.release()


if __name__ == "__main__":
    main()
