"""Headless continuous wrist-frame publisher (no GUI).

Owns the AVFoundation webcam and republishes raw + annotated frames to the same
paths wrist_vision.py uses, so floor_grasp/visual_contact (which read the FILE)
keep working without a cv2 window. Safe to run when no other camera owner exists.

Every ffmpeg pipe read is bounded. If three seconds pass without a complete
frame, the wedged child is killed and a fresh camera child is warmed up. A
failed respawn raises out of ``main`` so launchd/the supervisor sees nonzero.
"""
import sys
import time
sys.path.insert(0, "laptop")

import config
from wrist_vision import (
    NamedAVFoundationCamera, WristDetector, annotate,
    _atomic_write_jpeg, LATEST_RAW_PATH, LATEST_PREVIEW_PATH)


FRAME_STALL_TIMEOUT_S = 3.0
RAW_FRAME_INTERVAL_S = 1.0 / 30.0
PREVIEW_FRAME_INTERVAL_S = 0.10


def _warm_camera(camera, warmup_frames=None, timeout_s=FRAME_STALL_TIMEOUT_S):
    count = (config.WRIST_CAMERA_WARMUP_FRAMES
             if warmup_frames is None else int(warmup_frames))
    for _ in range(count):
        ok, _frame = camera.read(timeout_s=timeout_s)
        if not ok:
            raise RuntimeError("camera stopped during warmup")
    return camera


def _open_warmed_camera(camera_factory=NamedAVFoundationCamera,
                        warmup_frames=None):
    camera = camera_factory()
    try:
        return _warm_camera(camera, warmup_frames=warmup_frames)
    except Exception:
        camera.release()
        raise


def _respawn_after_stall(camera, camera_factory=NamedAVFoundationCamera,
                         warmup_frames=None):
    camera.release()
    print("[publish] camera stalled; respawning ffmpeg", flush=True)
    try:
        replacement = _open_warmed_camera(
            camera_factory, warmup_frames=warmup_frames)
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg camera respawn failed; publisher exiting") from exc
    print("[publish] ffmpeg respawned", flush=True)
    return replacement


def main(camera_factory=NamedAVFoundationCamera):
    cam = _open_warmed_camera(camera_factory)
    det = WristDetector()
    print("[publish] READY", flush=True)
    last_raw = 0.0
    last_preview = 0.0
    try:
        while True:
            ok, frame = cam.read(timeout_s=FRAME_STALL_TIMEOUT_S)
            if not ok:
                cam = _respawn_after_stall(cam, camera_factory)
                continue
            now = time.monotonic()
            if now - last_raw >= RAW_FRAME_INTERVAL_S:
                _atomic_write_jpeg(LATEST_RAW_PATH, frame)
                last_raw = now
            if now - last_preview >= PREVIEW_FRAME_INTERVAL_S:
                obs, _m = det.detect(frame)
                _atomic_write_jpeg(LATEST_PREVIEW_PATH, annotate(frame, obs))
                last_preview = now
    finally:
        cam.release()


if __name__ == "__main__":
    main()
