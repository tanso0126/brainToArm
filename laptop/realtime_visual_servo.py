"""Continuous eye-in-hand visual servo for motors 2/3/4.

FastSAM is used once to seed an arbitrary tabletop object.  A lightweight HSV
histogram tracker then follows that same object on every new camera frame while
the Uno is still slewing.  New collision-checked joint targets replace the old
firmware target at 10 Hz; the host never waits for ``DONE`` during approach.

The controller deliberately separates three points on the real bracket:

* camera: supplies the target image and keeps it visible;
* ultrasonic sensor: supplies range along the pointing direction;
* gripper: its live red/blue midpoint is the final contact target.

At long range the object centre stays comfortably inside the image.  As range
closes, the desired image row moves toward the live gripper midpoint so the
physical fingers, not the camera centre, travel over the object's centre.  A
floor guard removes downward velocity but still permits bounded forward motion.
Closing is impossible unless both measured sonar range and live centre
alignment pass.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import math
import time

import cv2
import numpy as np

import arm_fk
import config
from arm_session import ArmSessionClient
from arm_safety import PhysicalArmSafety
from floor_grasp import WristSceneDetector
from look_reach import (
    optical_axis_xz,
    task_state,
)
from ultrasonic_target_reach import (
    FINGERTIP_FLOOR_STOP_MM,
    STOP_RANGE_MM,
    VividFallbackDetector,
    fingertip_floor_clearance_mm,
    transition_fingertip_floor_clearance_mm,
)
from wrist_vision import (
    LATEST_RAW_PATH,
    WristDetector,
    _atomic_write_jpeg,
)


ROOT = Path(__file__).resolve().parents[1]
PREVIEW_PATH = ROOT / "data" / "vision" / "realtime_visual_servo_latest.jpg"
CONTROL_HZ = 10.0
SONAR_HZ = 5.0
PREVIEW_HZ = 10.0
TRACKER_MIN_CONFIDENCE = 0.05
TRACKER_MAX_CENTER_JUMP_RATIO = 0.20
TRACKER_SCALE_RANGE = (0.45, 2.20)
TRACKER_INITIAL_SEARCH_MARGIN_PX = 80
FAR_AIM_Y_RATIO = 0.56
NEAR_RANGE_MM = 180.0
GRASP_CENTER_TOLERANCE_PX = 55.0
GRASP_LATERAL_OPENING_FRACTION = 0.25
FLOOR_HOLD_START_MM = 18.0
STREAM_MAX_JOINT_STEP_DEG = 5.0
TASK_ADVANCE_MIN_MM = 2.5
TASK_ADVANCE_MAX_MM = 8.0
TASK_DAMPING = 0.18
MAX_INWARD_CORRECTION_MM = 6.0
TRACK_LOST_TIMEOUT_S = 0.75
SEARCH_WRISTS = (170, 180)
SEED_MIN_AREA_RATIO = 0.0004
SEED_MAX_AREA_RATIO = 0.08
SEED_MAX_AXIS_ERROR_RATIO = 0.16
SEED_FINGER_EXCLUSION_Y_RATIO = 0.72
SEED_FINGER_EXCLUSION_RADIUS_FRACTION = 0.28


@dataclass(frozen=True)
class TrackObservation:
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class GraspReadiness:
    ready: bool
    reason: str
    center_error_px: float


@dataclass(frozen=True)
class RealtimeSeed:
    center: tuple[float, float]
    bbox: tuple[int, int, int, int]
    area: float
    confidence: float
    median_saturation: float


class LatestFrameStream:
    """Consume every newly published frame without reopening the camera."""

    def __init__(self, path=LATEST_RAW_PATH):
        self.path = Path(path)
        self.last_mtime_ns = -1

    def read(self, timeout_s=1.0):
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            try:
                stat = self.path.stat()
            except FileNotFoundError:
                time.sleep(0.005)
                continue
            if stat.st_mtime_ns == self.last_mtime_ns:
                time.sleep(0.003)
                continue
            frame = cv2.imread(str(self.path))
            if frame is None:
                time.sleep(0.003)
                continue
            self.last_mtime_ns = stat.st_mtime_ns
            return frame, stat.st_mtime_ns
        raise TimeoutError("no fresh wrist frame")


class HistogramTargetTracker:
    """CamShift identity tracker seeded by one arbitrary-object box."""

    def __init__(self):
        self.histogram = None
        self.window = None
        self.last_center = None
        self.last_area = None
        self.seed_center = None
        self.seed_size = None
        self.anchor_offset = None
        self.anchor_area = None

    @staticmethod
    def _clamp_box(box, shape, margin=0):
        height, width = shape[:2]
        x, y, box_width, box_height = (int(round(value)) for value in box)
        x0 = max(0, x - int(margin))
        y0 = max(0, y - int(margin))
        x1 = min(width, x + box_width + int(margin))
        y1 = min(height, y + box_height + int(margin))
        return x0, y0, max(2, x1 - x0), max(2, y1 - y0)

    def initialize(self, frame, bbox):
        histogram_box = self._clamp_box(bbox, frame.shape, margin=4)
        x, y, width, height = histogram_box
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        crop = hsv[y:y + height, x:x + width]
        mask = cv2.inRange(
            crop,
            np.asarray((0, 28, 25), dtype=np.uint8),
            np.asarray((179, 255, 255), dtype=np.uint8),
        )
        if int(cv2.countNonZero(mask)) < 30:
            mask[:] = 255
        histogram = cv2.calcHist(
            [crop], [0, 1], mask, [36, 32], [0, 180, 0, 256])
        cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
        self.histogram = histogram
        self.window = self._clamp_box(
            bbox, frame.shape, margin=TRACKER_INITIAL_SEARCH_MARGIN_PX)
        self.last_center = np.asarray(
            (x + 0.5 * width, y + 0.5 * height), dtype=float)
        self.seed_center = self.last_center.copy()
        self.seed_size = np.asarray((width, height), dtype=float)
        nonzero = cv2.findNonZero(mask)
        if nonzero is None:
            anchor_center = self.seed_center.copy()
            anchor_area = float(width * height)
        else:
            anchor_x, anchor_y, anchor_width, anchor_height = (
                cv2.boundingRect(nonzero))
            anchor_center = np.asarray(
                (
                    x + anchor_x + 0.5 * anchor_width,
                    y + anchor_y + 0.5 * anchor_height,
                ),
                dtype=float,
            )
            anchor_area = float(anchor_width * anchor_height)
        self.anchor_offset = self.seed_center - anchor_center
        self.anchor_area = anchor_area
        # The first CamShift update intentionally collapses a wide reacquisition
        # window back onto the physical object, so it has no meaningful scale
        # ratio to the seed box.
        self.last_area = None
        return TrackObservation(
            tuple(self.last_center), histogram_box, 1.0)

    def update(self, frame):
        if self.histogram is None or self.window is None:
            raise RuntimeError("target tracker has not been initialized")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        probability = cv2.calcBackProject(
            [hsv], [0, 1], self.histogram, [0, 180, 0, 256], 1)
        valid = cv2.inRange(
            hsv,
            np.asarray((0, 20, 20), dtype=np.uint8),
            np.asarray((179, 255, 255), dtype=np.uint8),
        )
        probability = cv2.bitwise_and(probability, valid)
        probability = cv2.GaussianBlur(probability, (5, 5), 0)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 1.0)
        _rotated, window = cv2.CamShift(
            probability, tuple(self.window), criteria)
        window = self._clamp_box(window, frame.shape)
        x, y, width, height = window
        center = np.asarray(
            (x + 0.5 * width, y + 0.5 * height), dtype=float)
        area = float(width * height)
        diagonal = math.hypot(frame.shape[1], frame.shape[0])
        center_jump = float(np.linalg.norm(center - self.last_center))
        scale = (
            1.0 if self.last_area is None
            else area / max(1.0, self.last_area)
        )
        roi = probability[y:y + height, x:x + width]
        confidence = (
            float(np.mean(roi)) / 255.0 if roi.size else 0.0)
        if (
            center_jump > TRACKER_MAX_CENTER_JUMP_RATIO * diagonal
            or not TRACKER_SCALE_RANGE[0] <= scale <= TRACKER_SCALE_RANGE[1]
            or confidence < TRACKER_MIN_CONFIDENCE
        ):
            return None
        object_scale = math.sqrt(
            area / max(1.0, float(self.anchor_area)))
        object_center = center + self.anchor_offset * object_scale
        object_size = self.seed_size * object_scale
        object_box = self._clamp_box(
            (
                object_center[0] - 0.5 * object_size[0],
                object_center[1] - 0.5 * object_size[1],
                object_size[0],
                object_size[1],
            ),
            frame.shape,
        )
        self.window = window
        self.last_center = center
        self.last_area = area
        return TrackObservation(
            tuple(float(value) for value in object_center),
            object_box,
            confidence,
        )


def numeric_task_jacobian(pose):
    """d[camera x mm, camera z mm, hand pitch deg]/d[servo 2,3,4]."""
    pose = [int(round(value)) for value in pose]
    joints = (config.J_SHOULDER, config.J_ELBOW, config.J_WRIST)
    columns = []
    for joint in joints:
        low = list(pose)
        high = list(pose)
        low[joint] = max(config.SERVO_MIN[joint], low[joint] - 1)
        high[joint] = min(config.SERVO_MAX[joint], high[joint] + 1)
        span = float(high[joint] - low[joint])
        if span <= 0:
            columns.append(np.zeros(3, dtype=float))
            continue
        columns.append((task_state(high) - task_state(low)) / span)
    return np.column_stack(columns)


def dynamic_aim_y(frame_height, distance_mm, gripper_y):
    """Blend visible far-field aim into the real finger contact row."""
    far = FAR_AIM_Y_RATIO * float(frame_height)
    if distance_mm is None or not math.isfinite(float(distance_mm)):
        return far
    closeness = float(np.clip(
        (NEAR_RANGE_MM - float(distance_mm))
        / max(1.0, NEAR_RANGE_MM - STOP_RANGE_MM),
        0.0,
        1.0,
    ))
    return (1.0 - closeness) * far + closeness * float(gripper_y)


def grasp_readiness(
        target, gripper_center, opening_px, distance_mm,
        floor_clearance_mm):
    if target is None:
        return GraspReadiness(False, "target not tracked", float("inf"))
    center_error = float(np.linalg.norm(
        np.asarray(target.center, dtype=float)
        - np.asarray(gripper_center, dtype=float)))
    lateral = abs(float(target.center[0]) - float(gripper_center[0]))
    if distance_mm is None or not math.isfinite(float(distance_mm)):
        return GraspReadiness(False, "sonar unavailable", center_error)
    if float(distance_mm) > STOP_RANGE_MM:
        return GraspReadiness(
            False,
            f"sonar {float(distance_mm):.1f}mm > {STOP_RANGE_MM:.1f}mm",
            center_error,
        )
    if float(floor_clearance_mm) < FINGERTIP_FLOOR_STOP_MM:
        return GraspReadiness(False, "finger floor clearance exhausted",
                              center_error)
    if lateral > GRASP_LATERAL_OPENING_FRACTION * float(opening_px):
        return GraspReadiness(False, "object centre outside finger width",
                              center_error)
    if center_error > GRASP_CENTER_TOLERANCE_PX:
        return GraspReadiness(False, "object centre not deep inside jaws",
                              center_error)
    return GraspReadiness(True, "range and physical jaw centre aligned",
                          center_error)


def _vivid_frame_seeds(frame, gripper):
    """Recover coloured objects when segmentation misses an overexposed view."""
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.asarray((0, 45, 35), dtype=np.uint8),
        np.asarray((179, 255, 255), dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
    if gripper is not None:
        for blob in (gripper.blue, gripper.red):
            cv2.circle(
                mask,
                tuple(int(round(value)) for value in blob.center),
                int(round(max(blob.bbox[2], blob.bbox[3]) * 0.85)),
                0,
                -1,
            )
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, 8)
    seeds = []
    frame_area = float(width * height)
    for label in range(1, count):
        x, y, box_width, box_height, pixels = (
            int(value) for value in stats[label])
        ratio = float(pixels) / frame_area
        if not SEED_MIN_AREA_RATIO <= ratio <= SEED_MAX_AREA_RATIO:
            continue
        component = labels == label
        saturation = float(np.median(hsv[:, :, 1][component]))
        confidence = float(np.clip(saturation / 180.0, 0.0, 1.0))
        seeds.append(RealtimeSeed(
            center=tuple(float(value) for value in centroids[label]),
            bbox=(x, y, box_width, box_height),
            area=float(pixels),
            confidence=confidence,
            median_saturation=saturation,
        ))
    return seeds


def select_realtime_seed(scene, frame=None):
    """Select a compact object without a pose-dependent fixed horizon row."""
    height, width = scene.frame_shape[:2]
    gripper = getattr(scene, "gripper", None)
    if gripper is not None:
        aim_x = float(gripper.center[0])
        opening_px = float(gripper.opening_px)
    else:
        profile = config.WRIST_GRIPPER_OPEN_PROFILE
        aim_x = float(profile["center"][0]) * width
        opening_px = float(profile["opening_ratio"]) * math.hypot(
            width, height)
    finger_centres = (
        aim_x - 0.5 * opening_px,
        aim_x + 0.5 * opening_px,
    )
    valid = []
    candidates = list(scene.ranked)
    if frame is not None:
        candidates.extend(_vivid_frame_seeds(frame, gripper))
    for candidate in candidates:
        ratio = float(candidate.area) / float(width * height)
        axis_error = abs(float(candidate.center[0]) - aim_x)
        if not SEED_MIN_AREA_RATIO <= ratio <= SEED_MAX_AREA_RATIO:
            continue
        if axis_error > SEED_MAX_AXIS_ERROR_RATIO * width:
            continue
        if (
            float(candidate.center[1])
            >= SEED_FINGER_EXCLUSION_Y_RATIO * height
            and min(
                abs(float(candidate.center[0]) - finger_x)
                for finger_x in finger_centres
            )
            <= SEED_FINGER_EXCLUSION_RADIUS_FRACTION * opening_px
        ):
            continue
        if float(getattr(candidate, "confidence", 1.0)) < 0.20:
            continue
        score = (
            1.6 * float(candidate.center[1]) / height
            - 1.2 * axis_error / width
            + 0.25 * float(
                getattr(candidate, "median_saturation", 0.0)) / 255.0
            + 0.20 * float(getattr(candidate, "confidence", 1.0))
        )
        valid.append((score, candidate))
    return max(valid, key=lambda pair: pair[0])[1] if valid else None


def resolved_velocity_target(
        pose, vertical_error_px, distance_mm, floor_clearance_mm):
    """One damped-Jacobian target for the streaming 2/3/4 controller."""
    pose = [int(round(value)) for value in pose]
    finite_distance = (
        float(distance_mm)
        if distance_mm is not None and math.isfinite(float(distance_mm))
        else NEAR_RANGE_MM + 100.0
    )
    advance = float(np.clip(
        0.08 * (finite_distance - STOP_RANGE_MM),
        TASK_ADVANCE_MIN_MM,
        TASK_ADVANCE_MAX_MM,
    ))
    ray = optical_axis_xz(pose)
    desired_translation = advance * ray
    # The camera can point almost vertically downward at the high search pose.
    # Its optical ray then has a small negative x component even though the
    # physical object remains in front of the base. Never turn that mount angle
    # into an inward command; shoulder/elbow approach is forward by definition.
    desired_translation[0] = max(
        TASK_ADVANCE_MIN_MM, float(desired_translation[0]))
    if float(floor_clearance_mm) <= FLOOR_HOLD_START_MM:
        desired_translation = np.asarray((advance, 0.0), dtype=float)
    # Mounted-camera evidence: decreasing the hand pitch moved a target that
    # was already above the aim row even farther upward.  Image y therefore
    # has the same control sign as this task-space pitch convention.
    desired_pitch = float(np.clip(
        float(vertical_error_px) / 35.0, -2.0, 2.0))
    desired = np.asarray(
        (desired_translation[0], desired_translation[1], desired_pitch),
        dtype=float,
    )
    jacobian = numeric_task_jacobian(pose)
    scale = np.asarray((5.0, 5.0, 1.0), dtype=float)
    weighted_jacobian = jacobian / scale[:, None]
    weighted_desired = desired / scale
    normal = (
        weighted_jacobian.T @ weighted_jacobian
        + (TASK_DAMPING ** 2) * np.eye(3)
    )
    delta = np.linalg.solve(
        normal, weighted_jacobian.T @ weighted_desired)
    largest = float(np.max(np.abs(delta)))
    if largest > STREAM_MAX_JOINT_STEP_DEG:
        delta *= STREAM_MAX_JOINT_STEP_DEG / largest
    target = list(pose)
    for joint, value in zip(
            (config.J_SHOULDER, config.J_ELBOW, config.J_WRIST), delta):
        target[joint] = int(np.clip(
            round(target[joint] + value),
            config.SERVO_MIN[joint],
            config.SERVO_MAX[joint],
        ))
    if target == pose:
        return None
    if (
        transition_fingertip_floor_clearance_mm(pose, target)
        < FINGERTIP_FLOOR_STOP_MM
    ):
        return None
    if (
        float(arm_fk.geometry(target).finger_tip[0])
        < float(arm_fk.geometry(pose).finger_tip[0])
        - MAX_INWARD_CORRECTION_MM / 1000.0
    ):
        return None
    return {
        "pose": target,
        "delta": delta,
        "desired_task": desired,
        "jacobian": jacobian,
    }


def _gripper_geometry(observation, frame):
    if observation.gripper is not None:
        return (
            observation.gripper.center,
            float(observation.gripper.opening_px),
        )
    height, width = frame.shape[:2]
    diagonal = math.hypot(width, height)
    profile = config.WRIST_GRIPPER_OPEN_PROFILE
    return (
        (float(profile["center"][0]) * width,
         float(profile["center"][1]) * height),
        float(profile["opening_ratio"]) * diagonal,
    )


def _draw_preview(frame, target, gripper_center, aim_y, distance_mm,
                  clearance_mm, state):
    image = frame.copy()
    x, y, width, height = target.bbox
    cv2.rectangle(
        image, (x, y), (x + width, y + height), (0, 220, 255), 3)
    cv2.circle(
        image, tuple(int(round(value)) for value in target.center),
        7, (0, 220, 255), -1)
    cv2.drawMarker(
        image,
        tuple(int(round(value)) for value in gripper_center),
        (255, 255, 0), cv2.MARKER_CROSS, 34, 2)
    cv2.line(
        image, (0, int(round(aim_y))),
        (image.shape[1], int(round(aim_y))), (80, 255, 80), 2)
    text = (
        f"{state} range="
        f"{'--' if distance_mm is None else f'{distance_mm:.0f}mm'} "
        f"floor={clearance_mm:.1f}mm")
    cv2.putText(
        image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
        0.72, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(
        image, text, (18, 34), cv2.FONT_HERSHEY_SIMPLEX,
        0.72, (255, 255, 255), 2, cv2.LINE_AA)
    _atomic_write_jpeg(PREVIEW_PATH, image)


def _search_and_seed(
        client, frame_stream, detector, safety, execute):
    pose = list(client.request({"command": "status"})["pose"])
    if not execute:
        frame, _stamp = frame_stream.read(timeout_s=1.0)
        scene, _observation = detector.scene(frame)
        return pose, frame, select_realtime_seed(scene, frame)
    opened = list(pose)
    opened[config.J_GRIP] = config.GRIP_OPEN
    if opened != pose:
        client.request({
            "command": "move", "pose": opened,
            "require_camera": True,
        })
        pose = opened
    # A restarted controller must continue from the live pose. Returning to a
    # canned search pose discards successful approach progress and introduces
    # a large camera jump that can lose an otherwise visible target.
    frame, _stamp = frame_stream.read(timeout_s=1.0)
    scene, _observation = detector.scene(frame)
    candidate = select_realtime_seed(scene, frame)
    if candidate is not None:
        return pose, frame, candidate
    for wrist in SEARCH_WRISTS:
        target_pose = list(pose)
        target_pose[config.J_SHOULDER] = 70
        target_pose[config.J_ELBOW] = 90
        target_pose[config.J_WRIST] = int(wrist)
        report = safety.transition_report(pose, target_pose)
        if not report.safe:
            continue
        client.request({
            "command": "move", "pose": target_pose,
            "require_camera": True,
        })
        pose = target_pose
        frame, _stamp = frame_stream.read(timeout_s=1.0)
        scene, _observation = detector.scene(frame)
        candidate = select_realtime_seed(scene, frame)
        if candidate is not None:
            return pose, frame, candidate
    return pose, frame, None


def run(execute=False, allow_grasp=False, max_seconds=45.0):
    client = ArmSessionClient()
    safety = PhysicalArmSafety()
    frames = LatestFrameStream()
    detector = VividFallbackDetector(WristSceneDetector())
    wrist_detector = WristDetector()
    pose, frame, candidate = _search_and_seed(
        client, frames, detector, safety, execute)
    if candidate is None:
        return {"state": "no-target", "pose": pose}
    tracker = HistogramTargetTracker()
    target = tracker.initialize(frame, candidate.bbox)
    distance = None
    last_sonar = 0.0
    last_control = 0.0
    last_preview = 0.0
    last_seen = time.monotonic()
    deadline = last_seen + float(max_seconds)

    while time.monotonic() < deadline:
        frame, _stamp = frames.read(timeout_s=1.0)
        now = time.monotonic()
        tracked = tracker.update(frame)
        observation, _masks = wrist_detector.detect(frame)
        if tracked is not None:
            target = tracked
            last_seen = now
        elif now - last_seen > TRACK_LOST_TIMEOUT_S:
            return {
                "state": "target-lost",
                "pose": client.request({"command": "status"})["pose"],
                "preview": str(PREVIEW_PATH),
            }
        else:
            continue

        gripper_center, opening_px = _gripper_geometry(observation, frame)
        if now - last_sonar >= 1.0 / SONAR_HZ:
            response = client.request({
                "command": "distance", "samples": 1})
            distance = (
                float(response["distanceMm"])
                if response.get("valid") else None)
            last_sonar = now
        if now - last_control < 1.0 / CONTROL_HZ:
            continue
        pose = list(client.request({"command": "status"})["pose"])
        clearance = fingertip_floor_clearance_mm(pose)
        readiness = grasp_readiness(
            target, gripper_center, opening_px, distance, clearance)
        aim_y = dynamic_aim_y(
            frame.shape[0], distance, gripper_center[1])
        if now - last_preview >= 1.0 / PREVIEW_HZ:
            _draw_preview(
                frame, target, gripper_center, aim_y, distance,
                clearance, "READY" if readiness.ready else "TRACK")
            last_preview = now
        if readiness.ready:
            if not allow_grasp:
                return {
                    "state": "grasp-ready",
                    "pose": pose,
                    "distance_mm": distance,
                    "center_error_px": readiness.center_error_px,
                    "preview": str(PREVIEW_PATH),
                }
            closed = list(pose)
            closed[config.J_GRIP] = config.GRIP_CLOSED
            client.request({
                "command": "move", "pose": closed,
                "require_camera": True,
            })
            return {
                "state": "closed-pending-verification",
                "pose": closed,
                "distance_mm": distance,
                "center_error_px": readiness.center_error_px,
                "preview": str(PREVIEW_PATH),
            }
        plan = resolved_velocity_target(
            pose,
            float(target.center[1]) - float(aim_y),
            distance,
            clearance,
        )
        if plan is None:
            return {
                "state": "safe-reach-exhausted",
                "pose": pose,
                "distance_mm": distance,
                "center_error_px": readiness.center_error_px,
                "reason": readiness.reason,
                "preview": str(PREVIEW_PATH),
            }
        if execute:
            client.request({
                "command": "stream",
                "pose": plan["pose"],
                "require_camera": True,
            })
        else:
            return {
                "state": "planned",
                "pose": plan["pose"],
                "distance_mm": distance,
                "preview": str(PREVIEW_PATH),
            }
        last_control = now
    return {
        "state": "time-limit",
        "pose": client.request({"command": "status"})["pose"],
        "preview": str(PREVIEW_PATH),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--grasp", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=45.0)
    args = parser.parse_args()
    if args.grasp and not args.run:
        parser.error("--grasp requires --run")
    result = run(
        execute=args.run,
        allow_grasp=args.grasp,
        max_seconds=args.max_seconds,
    )
    print(f"[realtime-servo] RESULT {result}", flush=True)


if __name__ == "__main__":
    main()
