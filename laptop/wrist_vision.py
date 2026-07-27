"""Eye-in-hand vision using blue/red tape on the two gripper fingers.

The blue tape is physically on the left finger and red tape on the right. Their
centroids define the jaw axis, their midpoint defines the desired grasp point,
and their separation gives a visual opening measurement. A compact colored
target is detected independently, so alignment is target minus gripper midpoint
rather than target minus a hard-coded image coordinate.

This module never imports or opens the Uno. Its live mode is therefore safe while
the robot is unplugged::

    python3 laptop/wrist_vision.py --live

Left-click the target to lock its hue, right-click to return to automatic color,
S saves an annotated frame, and Q/Escape exits.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import argparse
import math
import os
import select
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

import config


DEBUG_DIR = Path(__file__).resolve().parents[1] / "data" / "vision"
LATEST_PREVIEW_PATH = DEBUG_DIR / "wrist_camera_latest.jpg"
LATEST_RAW_PATH = DEBUG_DIR / "wrist_camera_latest_raw.jpg"


def _atomic_write_jpeg(path, image):
    """Publish a complete JPEG so concurrent readers never see a partial frame."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"could not write camera frame {temporary}")
    temporary.replace(path)


@dataclass(frozen=True)
class ColorBlob:
    center: Tuple[float, float]
    area: float
    bbox: Tuple[int, int, int, int]
    contour: object
    mean_hsv: Tuple[float, float, float]


@dataclass(frozen=True)
class GripperObservation:
    blue: ColorBlob
    red: ColorBlob
    center: Tuple[float, float]
    jaw_axis: Tuple[float, float]
    opening_px: float
    angle_deg: float


@dataclass(frozen=True)
class TargetObservation:
    center: Tuple[float, float]
    area: float
    bbox: Tuple[int, int, int, int]
    contour: object
    hue: float
    angle_deg: Optional[float]


@dataclass(frozen=True)
class FrameQuality:
    clipped_fraction: float
    intensity_std: float
    valid: bool
    reason: str = ""


@dataclass(frozen=True)
class WristObservation:
    gripper: Optional[GripperObservation]
    target: Optional[TargetObservation]
    quality: FrameQuality
    error_xy: Optional[Tuple[float, float]] = None
    error_jaw: Optional[Tuple[float, float]] = None
    distance_px: Optional[float] = None
    orientation_error_deg: Optional[float] = None
    aligned: bool = False


def _normalize_axis_angle(angle):
    """Normalize an unoriented line angle to [-90, 90)."""
    return (float(angle) + 90.0) % 180.0 - 90.0


def _angle_difference(target, reference):
    return _normalize_axis_angle(float(target) - float(reference))


def _combine_hsv_ranges(hsv, ranges):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        mask = cv2.bitwise_or(
            mask,
            cv2.inRange(hsv, np.asarray(lower, dtype=np.uint8),
                        np.asarray(upper, dtype=np.uint8)))
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _blob_from_contour(contour, hsv):
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return None
    center = (moments["m10"] / moments["m00"],
              moments["m01"] / moments["m00"])
    local_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    cv2.drawContours(local_mask, [contour], -1, 255, -1)
    mean = cv2.mean(hsv, mask=local_mask)[:3]
    return ColorBlob(center, float(cv2.contourArea(contour)),
                     cv2.boundingRect(contour), contour,
                     tuple(float(value) for value in mean))


def _valid_blobs(mask, hsv, minimum_area, maximum_area,
                 minimum_fill=0.0, maximum_aspect=None, allow_bottom=False):
    height, width = mask.shape
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not minimum_area <= area <= maximum_area:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if (x <= 1 or y <= 1 or x + box_width >= width - 1
                or (not allow_bottom and y + box_height >= height - 1)):
            continue
        fill = area / max(1, box_width * box_height)
        aspect = max(box_width, box_height) / max(1, min(box_width, box_height))
        if fill < minimum_fill:
            continue
        if maximum_aspect is not None and aspect > maximum_aspect:
            continue
        blob = _blob_from_contour(contour, hsv)
        if blob is not None:
            blobs.append(blob)
    return sorted(blobs, key=lambda item: item.area, reverse=True)[:12]


def frame_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clipped = float(np.mean(gray >= 250))
    spread = float(gray.std())
    reasons = []
    if clipped > config.WRIST_MAX_CLIPPED_FRACTION:
        reasons.append(f"overexposed {clipped:.0%}")
    if spread < config.WRIST_MIN_FRAME_STD:
        reasons.append(f"low contrast std={spread:.1f}")
    return FrameQuality(clipped, spread, not reasons, ", ".join(reasons))


class WristDetector:
    def __init__(self, target_hsv=None):
        self.target_hsv = target_hsv if target_hsv is not None \
            else config.WRIST_TARGET_HSV

    def set_target_from_pixel(self, frame, point, radius=12):
        x, y = (int(round(value)) for value in point)
        height, width = frame.shape[:2]
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        patch = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        saturated = patch[patch[:, :, 1] >= config.WRIST_TARGET_MIN_SATURATION]
        if saturated.size == 0:
            raise ValueError("clicked target patch has too little color saturation")
        hue, saturation, value = np.median(saturated, axis=0)
        self.target_hsv = self._target_ranges(hue, saturation, value)
        return tuple(float(v) for v in (hue, saturation, value))

    @staticmethod
    def _target_ranges(hue, saturation=255, value=255):
        tolerance = config.WRIST_TARGET_HUE_TOLERANCE
        sat_lo = max(config.WRIST_TARGET_MIN_SATURATION,
                     int(saturation) - config.WRIST_TARGET_SATURATION_MARGIN)
        val_lo = max(config.WRIST_TARGET_MIN_VALUE,
                     int(value) - config.WRIST_TARGET_VALUE_MARGIN)
        lo, hi = int(hue) - tolerance, int(hue) + tolerance
        if lo < 0:
            ranges = [([0, sat_lo, val_lo], [hi, 255, 255]),
                      ([180 + lo, sat_lo, val_lo], [179, 255, 255])]
        elif hi > 179:
            ranges = [([lo, sat_lo, val_lo], [179, 255, 255]),
                      ([0, sat_lo, val_lo], [hi - 180, 255, 255])]
        else:
            ranges = [([lo, sat_lo, val_lo], [hi, 255, 255])]
        return ranges

    def set_target_hue(self, hue, saturation=255, value=255):
        """Lock a known object/tape hue without imposing any spatial ROI."""
        if not 0 <= float(hue) <= 179:
            raise ValueError("OpenCV target hue must be in [0, 179]")
        self.target_hsv = self._target_ranges(hue, saturation, value)
        return self.target_hsv

    def clear_target_lock(self):
        self.target_hsv = None

    def _detect_gripper(self, hsv, blue_mask, red_mask):
        height, width = hsv.shape[:2]
        area = height * width
        minimum = area * config.WRIST_MARKER_MIN_AREA_RATIO
        maximum = area * config.WRIST_MARKER_MAX_AREA_RATIO
        marker_args = (minimum, maximum,
                       config.WRIST_MARKER_MIN_FILL_RATIO,
                       config.WRIST_MARKER_MAX_ASPECT)
        # The mounted camera intentionally sees the fingers entering from the
        # bottom. Their tape contours are complete enough for centroids even
        # though the physical fingers continue outside the image.
        blue = _valid_blobs(blue_mask, hsv, *marker_args, allow_bottom=True)
        red = _valid_blobs(red_mask, hsv, *marker_args, allow_bottom=True)
        if not blue or not red:
            return None

        diagonal = math.hypot(width, height)
        min_sep = diagonal * config.WRIST_MARKER_MIN_SEPARATION_RATIO
        max_sep = diagonal * config.WRIST_MARKER_MAX_SEPARATION_RATIO
        closed_profile = config.WRIST_GRIPPER_CLOSED_PROFILE
        open_profile = config.WRIST_GRIPPER_OPEN_PROFILE
        closed_opening = closed_profile["opening_ratio"]
        opening_span = open_profile["opening_ratio"] - closed_opening
        pairs = []
        for left in blue:
            for right in red:
                # The rigid mount always presents blue on image-left and red on
                # image-right. Rejecting an inverted pair also prevents two
                # unrelated scene objects from defining a grasp axis.
                if left.center[0] >= right.center[0]:
                    continue
                dimensions_ok = True
                for marker in (left, right):
                    x, y, width_px, height_px = marker.bbox
                    width_ratio = width_px / width
                    height_ratio = height_px / height
                    bottom_gap_ratio = (height - (y + height_px)) / height
                    if not (
                        config.WRIST_MARKER_MIN_WIDTH_RATIO <= width_ratio
                        <= config.WRIST_MARKER_MAX_WIDTH_RATIO
                        and config.WRIST_MARKER_MIN_HEIGHT_RATIO <= height_ratio
                        <= config.WRIST_MARKER_MAX_HEIGHT_RATIO
                        and bottom_gap_ratio
                        <= config.WRIST_MARKER_MAX_BOTTOM_GAP_RATIO
                    ):
                        dimensions_ok = False
                        break
                if not dimensions_ok:
                    continue
                area_ratio = (max(left.area, right.area)
                              / max(1.0, min(left.area, right.area)))
                if area_ratio > config.WRIST_MARKER_MAX_PAIR_AREA_RATIO:
                    continue
                delta = np.subtract(right.center, left.center)
                separation = float(np.linalg.norm(delta))
                if not min_sep <= separation <= max_sep:
                    continue
                separation_ratio = separation / diagonal
                openness = float(np.clip(
                    (separation_ratio - closed_opening) / opening_span,
                    0.0, 1.0))
                expected_area_ratios = np.asarray([
                    closed_profile["blue_area_ratio"]
                    + openness * (open_profile["blue_area_ratio"]
                                  - closed_profile["blue_area_ratio"]),
                    closed_profile["red_area_ratio"]
                    + openness * (open_profile["red_area_ratio"]
                                  - closed_profile["red_area_ratio"]),
                ])
                actual_area_ratios = np.asarray(
                    [left.area / area, right.area / area])
                area_factors = actual_area_ratios / expected_area_ratios
                min_factor, max_factor = (
                    config.WRIST_MARKER_PROFILE_AREA_FACTOR_RANGE)
                if np.any(area_factors < min_factor) or np.any(
                        area_factors > max_factor):
                    continue
                midpoint = (np.asarray(left.center) + np.asarray(right.center)) / 2
                closed_center = np.asarray(closed_profile["center"], dtype=float)
                open_center = np.asarray(open_profile["center"], dtype=float)
                expected_center = closed_center + openness * (
                    open_center - closed_center)
                expected = expected_center * np.asarray([width, height])
                center_error = np.abs(midpoint - expected) / np.asarray(
                    [width, height], dtype=float)
                tolerance = np.asarray(
                    config.WRIST_GRIPPER_CENTER_TOLERANCE, dtype=float)
                if np.any(center_error > tolerance):
                    continue
                center_cost = float(np.linalg.norm(center_error / tolerance))
                balance_cost = abs(math.log(max(left.area, 1.0)
                                            / max(right.area, 1.0)))
                profile_cost = float(np.sum(np.abs(np.log(area_factors))))
                # Geometry dominates raw area, preventing a large colored object
                # elsewhere in the room from replacing one finger marker.
                score = (math.log1p(left.area + right.area)
                         - 7.0 * center_cost - 0.8 * balance_cost
                         - 0.6 * profile_cost)
                pairs.append((score, left, right, delta, separation, midpoint))
        if not pairs:
            return None
        _score, left, right, delta, separation, midpoint = max(
            pairs, key=lambda item: item[0])
        axis = delta / separation
        angle = _normalize_axis_angle(
            math.degrees(math.atan2(axis[1], axis[0])))
        return GripperObservation(
            left, right, tuple(float(value) for value in midpoint),
            tuple(float(value) for value in axis), separation, angle)

    def _target_mask(self, hsv, blue_mask, red_mask):
        if self.target_hsv:
            mask = _combine_hsv_ranges(hsv, self.target_hsv)
        else:
            lower = np.array([
                0, config.WRIST_TARGET_MIN_SATURATION,
                config.WRIST_TARGET_MIN_VALUE], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, np.array([179, 255, 255], dtype=np.uint8))
            for low_hue, high_hue in config.WRIST_AUTO_TARGET_EXCLUDED_HUES:
                excluded = cv2.inRange(
                    hsv, np.array([low_hue, 0, 0], dtype=np.uint8),
                    np.array([high_hue, 255, 255], dtype=np.uint8))
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(excluded))
            kernel = np.ones((5, 5), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        marker_mask = cv2.bitwise_or(blue_mask, red_mask)
        marker_mask = cv2.dilate(
            marker_mask, np.ones((15, 15), dtype=np.uint8), iterations=1)
        return cv2.bitwise_and(mask, cv2.bitwise_not(marker_mask))

    def _detect_target(self, hsv, mask, gripper):
        height, width = hsv.shape[:2]
        frame_area = height * width
        blobs = _valid_blobs(
            mask, hsv,
            frame_area * config.WRIST_TARGET_MIN_AREA_RATIO,
            frame_area * config.WRIST_TARGET_MAX_AREA_RATIO,
            allow_bottom=True)
        if not blobs:
            return None
        spatially_valid = []
        for blob in blobs:
            x, y, blob_width, blob_height = blob.bbox
            touches_bottom = y + blob_height >= height - 1
            if touches_bottom:
                # A graspable object naturally enters from below between the
                # fingers. Permit that one border case only when both rigid
                # finger markers prove its physical location; colored scene
                # clutter elsewhere along the bottom remains invalid.
                if gripper is None:
                    continue
                finger_x = sorted(
                    [gripper.blue.center[0], gripper.red.center[0]])
                if not (finger_x[0] <= blob.center[0] <= finger_x[1]):
                    continue
                if abs(blob.center[1] - gripper.center[1]) > 0.25 * height:
                    continue
            spatially_valid.append(blob)
        blobs = spatially_valid
        if not blobs:
            return None
        reference = np.asarray(
            gripper.center if gripper is not None else (width / 2, height / 2))
        diagonal = math.hypot(width, height)
        ranked = []
        for blob in blobs:
            distance = float(np.linalg.norm(np.asarray(blob.center) - reference))
            # Prefer the reachable object nearest the jaws, with a mild size
            # bonus. No venue/background image is involved.
            score = -distance / diagonal + 0.035 * math.log1p(blob.area)
            ranked.append((score, blob))
        if not ranked:
            return None
        blob = max(ranked, key=lambda item: item[0])[1]
        rect = cv2.minAreaRect(blob.contour)
        (_cx, _cy), (rect_width, rect_height), angle = rect
        long_side = max(rect_width, rect_height)
        short_side = max(1.0, min(rect_width, rect_height))
        orientation = None
        if long_side / short_side >= 1.35:
            if rect_width < rect_height:
                angle += 90.0
            orientation = _normalize_axis_angle(angle)
        return TargetObservation(
            blob.center, blob.area, blob.bbox, blob.contour,
            blob.mean_hsv[0], orientation)

    def detect(self, frame):
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("wrist frame must be a BGR HxWx3 image")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue_mask = _combine_hsv_ranges(hsv, config.WRIST_BLUE_HSV)
        red_mask = _combine_hsv_ranges(hsv, config.WRIST_RED_HSV)
        height, width = hsv.shape[:2]
        x_min, x_max, y_min, y_max = config.WRIST_MARKER_ROI
        marker_roi = np.zeros((height, width), dtype=np.uint8)
        marker_roi[
            int(round(y_min * height)):int(round(y_max * height)),
            int(round(x_min * width)):int(round(x_max * width)),
        ] = 255
        blue_mask = cv2.bitwise_and(blue_mask, marker_roi)
        red_mask = cv2.bitwise_and(red_mask, marker_roi)
        gripper = self._detect_gripper(hsv, blue_mask, red_mask)
        target_mask = self._target_mask(hsv, blue_mask, red_mask)
        target = self._detect_target(hsv, target_mask, gripper)
        quality = frame_quality(frame)

        if gripper is None or target is None:
            return WristObservation(gripper, target, quality), {
                "blue": blue_mask, "red": red_mask, "target": target_mask}
        error = np.subtract(target.center, gripper.center)
        axis = np.asarray(gripper.jaw_axis)
        normal = np.asarray((-axis[1], axis[0]))
        jaw_error = (float(np.dot(error, axis)), float(np.dot(error, normal)))
        distance = float(np.linalg.norm(error))
        orientation_error = (
            _angle_difference(target.angle_deg, gripper.angle_deg)
            if target.angle_deg is not None else None)
        aligned = quality.valid and distance <= config.WRIST_ALIGN_TOLERANCE_PX
        return WristObservation(
            gripper, target, quality,
            tuple(float(value) for value in error), jaw_error, distance,
            orientation_error, aligned), {
                "blue": blue_mask, "red": red_mask, "target": target_mask}


class NamedAVFoundationCamera:
    """Read a macOS AVFoundation camera by stable device name via ffmpeg."""
    def __init__(self, name=None, frame_size=None, fps=None):
        self.name = name or config.WRIST_CAMERA_NAME
        self.width, self.height = frame_size or config.WRIST_FRAME_SIZE
        self.fps = fps or config.WRIST_CAMERA_FPS
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("named wrist camera on macOS requires ffmpeg")
        command = [
            executable, "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation", "-pixel_format", "uyvy422",
            "-framerate", str(self.fps),
            "-video_size", f"{self.width}x{self.height}",
            "-i", f"{self.name}:none",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
        ]
        self.process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=self.width * self.height * 3 * 2)

    def isOpened(self):
        return self.process.poll() is None and self.process.stdout is not None

    def read(self, timeout_s=None):
        if not self.isOpened():
            return False, None
        needed = self.width * self.height * 3
        data = bytearray(needed)
        view = memoryview(data)
        offset = 0
        deadline = (None if timeout_s is None
                    else time.monotonic() + float(timeout_s))
        while offset < needed:
            if deadline is None:
                count = self.process.stdout.readinto(view[offset:])
                if not count:
                    return False, None
                offset += count
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False, None
            ready, _writable, _errors = select.select(
                [self.process.stdout], [], [], remaining)
            if not ready:
                return False, None
            chunk = os.read(self.process.stdout.fileno(), needed - offset)
            if not chunk:
                return False, None
            view[offset:offset + len(chunk)] = chunk
            offset += len(chunk)
        frame = np.frombuffer(data, dtype=np.uint8).reshape(
            self.height, self.width, 3)
        return True, frame

    def release(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.0)
        if self.process.stdout is not None:
            self.process.stdout.close()


def open_wrist_camera(name=None, index=None):
    if sys.platform == "darwin" and (name or config.WRIST_CAMERA_NAME):
        camera = NamedAVFoundationCamera(name=name)
    else:
        camera = cv2.VideoCapture(
            config.WRIST_CAMERA_INDEX if index is None else int(index))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.WRIST_FRAME_SIZE[0])
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WRIST_FRAME_SIZE[1])
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("could not open wrist camera")
    return camera


def annotate(frame, observation, fps=None):
    image = frame.copy()
    gripper, target = observation.gripper, observation.target
    if gripper is not None:
        for blob, color, label in (
                (gripper.blue, (255, 80, 40), "LEFT / BLUE"),
                (gripper.red, (40, 40, 255), "RIGHT / RED")):
            cv2.drawContours(image, [blob.contour], -1, color, 3)
            point = tuple(int(round(v)) for v in blob.center)
            cv2.circle(image, point, 7, color, -1)
            cv2.putText(image, label, (point[0] + 10, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        blue = tuple(int(round(v)) for v in gripper.blue.center)
        red = tuple(int(round(v)) for v in gripper.red.center)
        center = tuple(int(round(v)) for v in gripper.center)
        cv2.line(image, blue, red, (255, 255, 255), 2)
        cv2.drawMarker(image, center, (0, 255, 255), cv2.MARKER_CROSS, 30, 3)
    if target is not None:
        cv2.drawContours(image, [target.contour], -1, (20, 255, 40), 3)
        point = tuple(int(round(v)) for v in target.center)
        cv2.circle(image, point, 8, (20, 255, 40), -1)
        cv2.putText(image, f"TARGET H={target.hue:.0f}",
                    (point[0] + 10, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (20, 255, 40), 2, cv2.LINE_AA)
    if gripper is not None and target is not None:
        start = tuple(int(round(v)) for v in gripper.center)
        end = tuple(int(round(v)) for v in target.center)
        cv2.arrowedLine(image, start, end, (0, 220, 255), 3, tipLength=0.12)

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 118), (15, 20, 25), -1)
    image = cv2.addWeighted(overlay, 0.78, image, 0.22, 0)
    ready = observation.aligned
    state = "READY / ALIGNED" if ready else "NOT READY"
    color = (40, 230, 40) if ready else (40, 80, 255)
    cv2.putText(image, state, (20, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, color, 2, cv2.LINE_AA)
    marker_text = "markers found" if gripper is not None else "markers missing"
    target_text = "target found" if target is not None else "target missing"
    quality = observation.quality
    cv2.putText(
        image,
        f"{marker_text} | {target_text} | clipped={quality.clipped_fraction:.0%} "
        f"contrast={quality.intensity_std:.1f}" + (f" | {fps:.1f} fps" if fps else ""),
        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        (230, 230, 230), 2, cv2.LINE_AA)
    if not quality.valid:
        cv2.putText(image, "QUALITY FAIL: " + quality.reason, (20, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (40, 80, 255), 2, cv2.LINE_AA)
    elif observation.distance_px is not None:
        orientation = ("n/a" if observation.orientation_error_deg is None
                       else f"{observation.orientation_error_deg:+.1f}deg")
        cv2.putText(
            image,
            f"error x/y={observation.error_xy[0]:+.1f}/"
            f"{observation.error_xy[1]:+.1f}px  distance={observation.distance_px:.1f}px "
            f"orientation={orientation}",
            (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
            (230, 230, 230), 2, cv2.LINE_AA)
    else:
        warning = quality.reason or "show both finger tapes and one colored target"
        cv2.putText(image, warning, (20, 98), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (40, 180, 255), 2, cv2.LINE_AA)
    return image


def observation_summary(observation):
    result = {
        "quality": "ok" if observation.quality.valid else observation.quality.reason,
        "markers": observation.gripper is not None,
        "target": observation.target is not None,
        "aligned": observation.aligned,
    }
    if observation.gripper is not None:
        result.update({
            "gripper_center": tuple(round(v, 1) for v in observation.gripper.center),
            "opening_px": round(observation.gripper.opening_px, 1),
            "gripper_angle_deg": round(observation.gripper.angle_deg, 1),
        })
    if observation.target is not None:
        result["target_center"] = tuple(round(v, 1) for v in observation.target.center)
    if observation.error_xy is not None:
        result["error_xy"] = tuple(round(v, 1) for v in observation.error_xy)
        result["distance_px"] = round(observation.distance_px, 1)
    return result


def run_live(camera_name=None, snapshot=False, seconds=None):
    camera = open_wrist_camera(name=camera_name)
    detector = WristDetector()
    latest = {"frame": None}
    window = "brainToArm wrist camera (Q quit, click target, right-click reset, S save)"

    def mouse(event, x, y, _flags, _param):
        frame = latest["frame"]
        if frame is None:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            try:
                hsv = detector.set_target_from_pixel(frame, (x, y))
                print("[wrist] target hue locked at HSV "
                      + "/".join(f"{value:.0f}" for value in hsv))
            except ValueError as exc:
                print(f"[wrist] target lock rejected: {exc}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            detector.clear_target_lock()
            print("[wrist] target hue lock cleared")

    if not snapshot:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window, mouse)
    started = time.monotonic()
    previous = started
    fps_ema = None
    last_observation = None
    latest_saved = float("-inf")
    try:
        for _ in range(config.WRIST_CAMERA_WARMUP_FRAMES):
            ok, _frame = camera.read()
            if not ok:
                raise RuntimeError(
                    f"camera '{camera_name or config.WRIST_CAMERA_NAME}' stopped during warmup")
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("wrist camera read failed")
            latest["frame"] = frame
            observation, _masks = detector.detect(frame)
            last_observation = observation
            now = time.monotonic()
            instant = 1.0 / max(now - previous, 1e-6)
            previous = now
            fps_ema = instant if fps_ema is None else 0.15 * instant + 0.85 * fps_ema
            rendered = annotate(frame, observation, fps=fps_ema)
            if now - latest_saved >= config.WRIST_LATEST_PREVIEW_INTERVAL_S:
                # Never feed the annotated image back into perception: its
                # yellow grasp cross and colored outlines are valid HSV pixels
                # and can otherwise become self-generated false targets.
                _atomic_write_jpeg(LATEST_RAW_PATH, frame)
                _atomic_write_jpeg(LATEST_PREVIEW_PATH, rendered)
                latest_saved = now
            if snapshot:
                print(f"[wrist] saved {LATEST_PREVIEW_PATH}")
                print(f"[wrist] {observation_summary(observation)}")
                return observation
            cv2.imshow(window, rendered)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                DEBUG_DIR.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                path = DEBUG_DIR / f"wrist_{stamp}.jpg"
                cv2.imwrite(str(path), rendered)
                print(f"[wrist] saved {path}")
            if seconds is not None and now - started >= seconds:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    if last_observation is not None:
        print(f"[wrist] {observation_summary(last_observation)}")
    return last_observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="show live annotated camera (default unless --snapshot)")
    parser.add_argument("--snapshot", action="store_true",
                        help="save one warmed, annotated frame and exit")
    parser.add_argument("--camera-name", default=config.WRIST_CAMERA_NAME,
                        help="stable AVFoundation device name")
    parser.add_argument("--seconds", type=float,
                        help="optional live preview duration")
    args = parser.parse_args()
    run_live(args.camera_name, snapshot=args.snapshot, seconds=args.seconds)


if __name__ == "__main__":
    main()
