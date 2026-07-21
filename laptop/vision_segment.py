"""Background-independent object and gripper perception using FastSAM.

The neural segmenter proposes every visible instance. Small deterministic
selectors then choose the single compact tabletop target and the nearest arm
segment above it. This avoids empty-table snapshots and venue-specific pixels.
"""
from dataclasses import dataclass
from pathlib import Path
import math

import cv2
import numpy as np

import config


@dataclass(frozen=True)
class ObjectDetection:
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: float
    confidence: float = 1.0
    median_saturation: float = 0.0
    median_value: float = 255.0


def _box_area(item):
    return item.bbox[2] * item.bbox[3]


def _box_iou(a, b):
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = _box_area(a) + _box_area(b) - intersection
    return intersection / union if union else 0.0


def estimate_camera_transform(reference, frame):
    """Estimate camera auto-framing change from the arm's fixed base region."""
    if reference is None or frame is None or reference.shape != frame.shape:
        raise ValueError("camera geometry frames must have matching shapes")
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[int(height * 0.52):height, :int(width * 0.58)] = 255
    sift = cv2.SIFT_create(nfeatures=2500)
    key_a, desc_a = sift.detectAndCompute(reference, mask)
    key_b, desc_b = sift.detectAndCompute(frame, mask)
    if desc_a is None or desc_b is None:
        raise RuntimeError("not enough fixed-base features for camera preflight")
    matches = cv2.BFMatcher().knnMatch(desc_a, desc_b, k=2)
    good = [first for first, second in matches
            if first.distance < 0.72 * second.distance]
    if len(good) < 12:
        raise RuntimeError("not enough fixed-base matches for camera preflight")
    source = np.float32([key_a[item.queryIdx].pt for item in good])
    target = np.float32([key_b[item.trainIdx].pt for item in good])
    matrix, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None or inliers is None or int(inliers.sum()) < 10:
        raise RuntimeError("fixed-base camera transform is unreliable")
    a, b = float(matrix[0, 0]), float(matrix[0, 1])
    scale = math.hypot(a, b)
    rotation = math.degrees(math.atan2(-b, a))
    translation = math.hypot(float(matrix[0, 2]), float(matrix[1, 2]))
    return {"scale": scale, "rotation_deg": rotation,
            "translation_px": translation, "inliers": int(inliers.sum())}


def select_pick_target(candidates, frame_shape, previous=None):
    """Select one portable, compact target without coordinates or background."""
    height, width = frame_shape[:2]
    frame_area = width * height
    valid = []
    for item in candidates:
        x, y, box_width, box_height = item.bbox
        ratio = box_width * box_height / frame_area
        aspect = box_width / max(box_height, 1)
        if (x <= 2 or y <= 2 or x + box_width >= width - 2
                or y + box_height >= height - 2):
            continue
        if not config.PLANAR_TARGET_MIN_BOX_RATIO <= ratio <= config.PLANAR_TARGET_MAX_BOX_RATIO:
            continue
        if not config.PLANAR_TARGET_MIN_ASPECT <= aspect <= config.PLANAR_TARGET_MAX_ASPECT:
            continue
        if previous is None and item.center[1] < height * config.PLANAR_TARGET_MIN_Y_RATIO:
            continue
        if previous is None:
            size_bonus = min(ratio / 0.01, 1.0) * 0.10
            score = item.confidence + 0.15 * item.center[1] / height + size_bonus
        else:
            distance = math.dist(item.center, previous.center)
            area_ratio = max(_box_area(item), 1) / max(_box_area(previous), 1)
            similarity = math.exp(-abs(math.log(area_ratio)))
            score = (item.confidence + 0.80 * math.exp(-distance / 250.0)
                     + 0.50 * similarity + 0.25 * _box_iou(item, previous))
        valid.append((score, item))
    if not valid:
        raise RuntimeError("no compact pick target found in the current frame")
    return max(valid, key=lambda pair: pair[0])[1]


def select_gripper(candidates, target, frame_shape):
    """Select the lowest compact arm segment directly above the target."""
    height, width = frame_shape[:2]
    tx, ty, tw, _th = target.bbox
    target_left, target_right = tx - 80, tx + tw + 80
    valid = []
    for item in candidates:
        if item is target or _box_iou(item, target) > 0.20:
            continue
        x, y, box_width, box_height = item.bbox
        right, bottom = x + box_width, y + box_height
        if bottom > ty + 60 or bottom < height * 0.30:
            continue
        if right < target_left or x > target_right:
            continue
        if not 12 <= box_width <= width * 0.16 or not 20 <= box_height <= height * 0.35:
            continue
        # Servo wires in this build are brown/red and were repeatedly lower
        # than the fingers. FastSAM mask pixels separate them cleanly from the
        # white linkage and black low-saturation finger tip.
        if item.median_saturation > 65:
            continue
        horizontal_error = abs(item.center[0] - target.center[0])
        if horizontal_error > 100:
            continue
        gap = max(0, ty - bottom)
        # The fingers are often split into a lower, lower-confidence mask than
        # the white wrist joint. Prefer the physically lowest valid segment;
        # confidence is only a weak tie-breaker.
        score = (bottom / height - 0.10 * horizontal_error / width
                 + 0.05 * item.confidence)
        valid.append((score, item))
    if not valid:
        raise RuntimeError("gripper could not be located above the target")
    return max(valid, key=lambda pair: pair[0])[1]


class FastSAMDetector:
    """Lazy model wrapper; one downloaded 23 MB weight works offline."""

    def __init__(self, model_path=None):
        try:
            from ultralytics import FastSAM
        except ImportError as exc:
            raise RuntimeError(
                "FastSAM is required; run: pip install -r requirements.txt") from exc
        root = Path(__file__).resolve().parents[1]
        self.model_path = Path(model_path or config.PLANAR_VISION_MODEL)
        if not self.model_path.is_absolute():
            self.model_path = root / self.model_path
        self.model = FastSAM(str(self.model_path))

    def candidates(self, frame):
        result = self.model(
            frame,
            device=config.PLANAR_VISION_DEVICE,
            retina_masks=True,
            imgsz=config.PLANAR_VISION_IMAGE_SIZE,
            conf=config.PLANAR_VISION_CONFIDENCE,
            iou=0.9,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        masks = (result.masks.data.cpu().numpy()
                 if result.masks is not None else [None] * len(boxes))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = []
        for xyxy, confidence, mask in zip(boxes, confidences, masks):
            x1, y1, x2, y2 = (float(value) for value in xyxy)
            x, y = int(round(x1)), int(round(y1))
            box_width = max(1, int(round(x2 - x1)))
            box_height = max(1, int(round(y2 - y1)))
            pixels = hsv[mask > 0.5] if mask is not None else np.empty((0, 3))
            median_saturation = float(np.median(pixels[:, 1])) if pixels.size else 0.0
            median_value = float(np.median(pixels[:, 2])) if pixels.size else 255.0
            detections.append(ObjectDetection(
                ((x1 + x2) / 2, (y1 + y2) / 2),
                (x, y, box_width, box_height),
                float(box_width * box_height),
                float(confidence),
                median_saturation,
                median_value,
            ))
        return detections

    def locate(self, frame, previous=None, require_gripper=True):
        candidates = self.candidates(frame)
        target = select_pick_target(candidates, frame.shape, previous=previous)
        gripper = (select_gripper(candidates, target, frame.shape)
                   if require_gripper else None)
        return target, gripper, candidates


def annotate_pair(frame, target, gripper=None, label="target"):
    image = frame.copy()
    for item, color, name in ((target, (0, 255, 0), label),
                              (gripper, (255, 255, 0), "gripper")):
        if item is None:
            continue
        x, y, width, height = item.bbox
        cv2.rectangle(image, (x, y), (x + width, y + height), color, 3)
        cv2.circle(image, tuple(round(value) for value in item.center), 6, color, -1)
        cv2.putText(image, f"{name} {item.confidence:.2f}",
                    (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)
    return image
