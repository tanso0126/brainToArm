"""DirectShow wrist-camera auto-selection, publishing, and live preview."""

from pathlib import Path
import argparse
import json
import os
import sys
import time

import cv2


RELEASE = Path(__file__).resolve().parent
ROOT = RELEASE.parent
LAPTOP = ROOT / "laptop"
sys.path.insert(0, str(LAPTOP))

import config  # noqa: E402
from wrist_vision import (  # noqa: E402
    LATEST_PREVIEW_PATH,
    LATEST_RAW_PATH,
    WristDetector,
    _atomic_write_jpeg,
    annotate,
)


READY_FILE = ROOT / "data" / "runtime" / "windows_camera.json"
CONTROL_PREVIEW = (
    ROOT / "data" / "vision" / "realtime_visual_servo_latest.jpg")
WINDOW_NAME = "brainToArm LIVE - Q to stop"


def _backend():
    return cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY


def open_camera(index):
    camera = cv2.VideoCapture(int(index), _backend())
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.WRIST_FRAME_SIZE[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WRIST_FRAME_SIZE[1])
    camera.set(cv2.CAP_PROP_FPS, config.WRIST_CAMERA_FPS)
    if not camera.isOpened():
        camera.release()
        return None
    return camera


def warm_frame(camera, count=12):
    frame = None
    for _ in range(int(count)):
        ok, candidate = camera.read()
        if ok and candidate is not None:
            frame = candidate
    return frame


def gripper_score(frame, detector=None):
    """High score only when the rigid blue-left/red-right finger pair exists."""
    detector = detector or WristDetector()
    observation, _masks = detector.detect(frame)
    if observation.gripper is None:
        return 0.0
    gripper = observation.gripper
    marker_area = gripper.blue.area + gripper.red.area
    return 1000.0 + marker_area / max(1.0, frame.shape[0] * frame.shape[1])


def choose_camera(requested="auto", maximum_index=8):
    if str(requested).lower() != "auto":
        index = int(requested)
        camera = open_camera(index)
        if camera is None:
            raise RuntimeError(f"camera index {index} could not be opened")
        frame = warm_frame(camera)
        if frame is None:
            camera.release()
            raise RuntimeError(f"camera index {index} produced no frames")
        return index, camera, frame

    detector = WristDetector()
    candidates = []
    for index in range(int(maximum_index) + 1):
        camera = open_camera(index)
        if camera is None:
            continue
        frame = warm_frame(camera)
        if frame is not None:
            candidates.append((gripper_score(frame, detector), index))
        camera.release()
    marked = [item for item in candidates if item[0] >= 1000.0]
    if not marked:
        opened = ", ".join(str(index) for _score, index in candidates)
        raise RuntimeError(
            "No camera showed both blue-left and red-right finger tapes. "
            f"Opened indices: {opened or '(none)'}. Mount the wrist camera so "
            "both tapes are visible, or run CHECK_CAMERA.bat with --camera N.")
    _score, index = max(marked)
    camera = open_camera(index)
    frame = warm_frame(camera)
    if frame is None:
        camera.release()
        raise RuntimeError(f"selected camera {index} stopped during reopen")
    return index, camera, frame


def write_ready(index):
    READY_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = READY_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({
            "pid": os.getpid(),
            "cameraIndex": int(index),
            "readyAt": time.time(),
        }),
        encoding="utf-8",
    )
    temporary.replace(READY_FILE)


def _controller_preview(started_at):
    try:
        stat = CONTROL_PREVIEW.stat()
    except FileNotFoundError:
        return None
    if stat.st_mtime < started_at or time.time() - stat.st_mtime > 0.75:
        return None
    return cv2.imread(str(CONTROL_PREVIEW))


def publish(camera_arg="auto", headless=False):
    try:
        READY_FILE.unlink()
    except FileNotFoundError:
        pass
    index, camera, frame = choose_camera(camera_arg)
    detector = WristDetector()
    started_at = time.time()
    write_ready(index)
    print(
        f"[camera] READY index={index} "
        f"size={frame.shape[1]}x{frame.shape[0]}",
        flush=True,
    )
    previous = time.monotonic()
    fps = None
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError(f"camera {index} stopped producing frames")
            now = time.monotonic()
            instant = 1.0 / max(now - previous, 1e-6)
            previous = now
            fps = instant if fps is None else 0.1 * instant + 0.9 * fps
            observation, _masks = detector.detect(frame)
            rendered = annotate(frame, observation, fps=fps)
            _atomic_write_jpeg(LATEST_RAW_PATH, frame)
            _atomic_write_jpeg(LATEST_PREVIEW_PATH, rendered)
            if not headless:
                display = _controller_preview(started_at)
                if display is None:
                    display = rendered
                cv2.putText(
                    display, f"Windows camera {index}",
                    (18, display.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(WINDOW_NAME, display)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    print("[camera] operator closed preview", flush=True)
                    return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()
        try:
            READY_FILE.unlink()
        except FileNotFoundError:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="auto",
                        help="DirectShow camera index, or auto")
    parser.add_argument("--headless", action="store_true",
                        help="publish frames without opening a preview window")
    args = parser.parse_args()
    return publish(args.camera, args.headless)


if __name__ == "__main__":
    raise SystemExit(main())
