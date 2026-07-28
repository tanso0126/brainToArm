"""Interactive MuJoCo studio for the browser dashboard.

This module is deliberately hardware isolated.  It keeps one MuJoCo model
alive, renders the real overview and wrist cameras, and executes the same
bounded six-servo commands used by the physical controller.  Scene edits are
allowed only while stopped and rebuild free bodies from explicit user input.

The robot's target choice is gated by pixels rendered by the wrist camera.
MuJoCo body state is used only for physics, return-to-origin bookkeeping, and
success verification; it is not passed to the camera detector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, RLock, Thread, current_thread
from typing import Iterable
from xml.etree import ElementTree as ET
import math
import time
import uuid

import cv2
import mujoco
import numpy as np

try:
    from .mujoco_robot import (
        RobotSpec,
        build_mjcf,
        servo_to_joint_targets,
        site_position,
    )
except ImportError:
    from mujoco_robot import (
        RobotSpec,
        build_mjcf,
        servo_to_joint_targets,
        site_position,
    )

try:
    from laptop import config
    from laptop.floor_motion import floor_pose
except ImportError:
    import sys
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "laptop"))
    from laptop import config
    from laptop.floor_motion import floor_pose


MAX_OBJECTS = 2
WORKSPACE_RADIUS = (0.387, 0.414)
WORKSPACE_YAW_DEG = (0.0, 0.0)
DEFAULT_BASKET = (0.396, 0.0)
DEFAULT_COLORS = ("#ffb000", "#376dfa", "#19a05b", "#a839fd", "#f04f65")
SCAN_ROUTE = ((90, 110), (90, 102), (90, 94), (90, 86), (90, 78))
MOTION_SERVO_STEP_DEG = 2.0
MIN_SAFE_TOOL_X_M = 0.175
MIN_AIR_TOOL_Z_M = 0.017
TARGET_REVIEW_SECONDS = 1.6
POST_DELIVERY_REVIEW_SECONDS = 10.0


@dataclass
class SceneObject:
    id: str
    label: str
    shape: str
    color: str
    size_m: float
    x: float
    y: float
    origin_x: float
    origin_y: float
    status: str = "table"

    @property
    def half_height(self) -> float:
        if self.shape == "sphere":
            return self.size_m
        return self.size_m * 0.9


@dataclass
class StudioEvent:
    id: int
    kind: str
    text: str
    at: str


def _hex_rgb(value: str) -> tuple[float, float, float]:
    value = str(value).strip().lstrip("#")
    if len(value) != 6:
        raise ValueError("색상은 #RRGGBB 형식이어야 합니다")
    try:
        channels = tuple(int(value[index:index + 2], 16) / 255.0
                         for index in (0, 2, 4))
    except ValueError as exc:
        raise ValueError("색상은 #RRGGBB 형식이어야 합니다") from exc
    return channels


def _shape_size(shape: str, size: float) -> str:
    if shape == "box":
        return f"{size:.6f} {size:.6f} {size * 0.9:.6f}"
    if shape == "cylinder":
        return f"{size:.6f} {size * 0.9:.6f}"
    if shape == "sphere":
        return f"{size:.6f}"
    raise ValueError("shape은 box, cylinder, sphere 중 하나여야 합니다")


def build_studio_mjcf(
    objects: Iterable[SceneObject],
    *,
    basket_x: float,
    basket_y: float,
    spec: RobotSpec | None = None,
) -> str:
    """Build the physical arm plus editable free objects and a shallow tray."""

    spec = spec or RobotSpec.from_manifest()
    root = ET.fromstring(build_mjcf(spec))
    visual_global = root.find("./visual/global")
    if visual_global is not None:
        visual_global.set("offwidth", "1280")
        visual_global.set("offheight", "720")
    for default in root.findall("./default/default"):
        if default.get("class") == "visual":
            geom = default.find("geom")
            if geom is not None:
                geom.set("rgba", "0.78 0.80 0.83 1")
        elif default.get("class") == "collision":
            geom = default.find("geom")
            if geom is not None:
                geom.set("rgba", "0.56 0.59 0.64 0.72")
    world = root.find("worldbody")
    if world is None:
        raise RuntimeError("MuJoCo worldbody가 없습니다")
    default_target = world.find("./body[@name='target']")
    if default_target is not None:
        world.remove(default_target)
    overview = world.find("./camera[@name='overview']")
    if overview is not None:
        overview.set("fovy", "35")
    base = world.find("./body[@name='base']")
    if base is None:
        raise RuntimeError("MuJoCo base body가 없습니다")
    # The real task has no depth sensor and must be able to retrieve a late
    # rejection. Keep the tray physical but nearly flush with the shared floor
    # so the same calibrated grasp height remains valid.
    tray = ET.SubElement(
        world, "body", name="basket",
        pos=f"{basket_x:.6f} {basket_y:.6f} 0")
    ET.SubElement(
        tray, "geom", name="basket_floor", type="box",
        pos="0 0 0.00025", size="0.006 0.025 0.00025",
        rgba="0.12 0.34 0.96 0.34", contype="1", conaffinity="2",
        friction="1.0 0.01 0.001",
    )
    for name, y in (("basket_left", -0.0265), ("basket_right", 0.0265)):
        ET.SubElement(
            tray, "geom", name=name, type="box",
            pos=f"0 {y:.6f} 0.0005", size="0.006 0.0015 0.0005",
            rgba="0.12 0.34 0.96 0.72", contype="1", conaffinity="2",
        )

    for index, item in enumerate(objects):
        body = ET.SubElement(
            world, "body", name=f"studio_object_{index}",
            pos=f"{item.x:.6f} {item.y:.6f} {item.half_height:.6f}",
        )
        ET.SubElement(body, "freejoint", name=f"studio_object_free_{index}")
        rgb = _hex_rgb(item.color)
        ET.SubElement(
            body, "geom", name=f"studio_object_geom_{index}", type=item.shape,
            size=_shape_size(item.shape, item.size_m), mass="0.018",
            rgba=f"{rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f} 1",
            friction="1.35 0.025 0.004", solref="0.006 1",
        )
    return ET.tostring(root, encoding="unicode")


class MuJoCoStudio:
    """Thread-safe editable scene and reversible pick-and-place controller."""

    def __init__(self):
        self._lock = RLock()
        self._stop = Event()
        self._reject = Event()
        self._worker: Thread | None = None
        self._spec = RobotSpec.from_manifest()
        self._objects = [
            SceneObject(
                "object-1", "노란 원통", "cylinder", "#ffb000", 0.0045,
                0.3870, 0.0000, 0.3870, 0.0000),
            SceneObject(
                "object-2", "초록 블록", "box", "#19a05b", 0.0045,
                0.4140, 0.0000, 0.4140, 0.0000),
        ]
        self._basket_x, self._basket_y = DEFAULT_BASKET
        self._rejected: list[str] = []
        self._active_id: str | None = None
        self._last_delivered_id: str | None = None
        self._phase = "idle"
        self._running = False
        self._cycle = 1
        self._event_counter = 0
        self._events: list[StudioEvent] = []
        self._pose = self._spec.home_servo_deg.copy()
        self._detector = {
            "source": "wrist_rgb",
            "visibleIds": [],
            "targetCenter": None,
            "markerRow": None,
            "pixelCount": 0,
        }
        self._model: mujoco.MjModel
        self._data: mujoco.MjData
        self._body_by_id: dict[str, int] = {}
        self._joint_by_id: dict[str, int] = {}
        self._floor_x_by_elbow: dict[int, float] = {}
        self._held_offsets_local: dict[str, np.ndarray] = {}
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._render_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mujoco-studio-render")
        self._rebuild()
        self._add_event("3D MuJoCo 작업실이 준비되었습니다.", "info")

    def close(self):
        self.stop()
        self._stop.set()
        worker = self._worker
        if worker and worker.is_alive() and worker is not current_thread():
            worker.join(timeout=2)
        with self._lock:
            self._close_renderers()
        self._render_executor.shutdown(wait=True, cancel_futures=True)

    def _close_renderers(self):
        def close_all():
            for renderer in self._renderers.values():
                renderer.close()
            self._renderers.clear()

        self._render_executor.submit(close_all).result(timeout=5)

    def _add_event(self, text: str, kind: str = "info"):
        self._event_counter += 1
        self._events.insert(0, StudioEvent(
            self._event_counter,
            kind,
            text,
            datetime.now().strftime("%H:%M:%S"),
        ))
        del self._events[80:]

    def _rebuild(self):
        self._close_renderers()
        xml = build_studio_mjcf(
            self._objects, basket_x=self._basket_x, basket_y=self._basket_y,
            spec=self._spec)
        self._model = mujoco.MjModel.from_xml_string(xml)
        self._data = mujoco.MjData(self._model)
        self._set_pose_static(self._spec.home_servo_deg)
        self._pose = self._spec.home_servo_deg.copy()
        self._body_by_id.clear()
        self._joint_by_id.clear()
        for index, item in enumerate(self._objects):
            self._body_by_id[item.id] = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY,
                f"studio_object_{index}")
            self._joint_by_id[item.id] = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_JOINT,
                f"studio_object_free_{index}")
        self._floor_x_by_elbow.clear()
        for elbow in range(config.FLOOR_ELBOW_RANGE[0],
                           config.FLOOR_ELBOW_RANGE[1] + 1):
            self._set_pose_static(floor_pose(elbow, "grasp"))
            self._floor_x_by_elbow[elbow] = float(
                site_position(self._model, self._data, "tool_center")[0])
        self._set_pose_static(self._spec.home_servo_deg)

    def _validate_pose(self, pose: Iterable[float]) -> np.ndarray:
        values = np.asarray(tuple(pose), dtype=np.float64)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError("servo pose must contain six finite degrees")
        if (np.any(values < self._spec.servo_min_deg)
                or np.any(values > self._spec.servo_max_deg)):
            raise ValueError("servo pose outside configured limits")
        return values

    def _targets(self, pose: Iterable[float]):
        servo = self._validate_pose(pose)
        servo[0] = self._spec.base_locked_deg
        return servo, servo_to_joint_targets(servo, self._spec)

    def _set_pose_static(self, pose: Iterable[float]):
        servo, targets = self._targets(pose)
        for name, value in targets.items():
            joint = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._data.qpos[int(self._model.jnt_qposadr[joint])] = value
        self._data.qvel[:] = 0
        self._data.ctrl[:] = (
            targets["shoulder"], targets["elbow"], targets["wrist_pitch"],
            targets["wrist_roll"], targets["grip_left"], targets["grip_right"],
        )
        mujoco.mj_forward(self._model, self._data)

    def _command_pose(self, pose: Iterable[float]) -> np.ndarray:
        servo, targets = self._targets(pose)
        self._data.ctrl[:] = (
            targets["shoulder"], targets["elbow"], targets["wrist_pitch"],
            targets["wrist_roll"], targets["grip_left"], targets["grip_right"],
        )
        return servo

    def _object(self, object_id: str) -> SceneObject:
        for item in self._objects:
            if item.id == object_id:
                return item
        raise ValueError("물체를 찾을 수 없습니다")

    def _object_position(self, object_id: str) -> np.ndarray:
        body_id = self._body_by_id[object_id]
        return self._data.xpos[body_id].copy()

    def _floor_pose_world(
        self, x: float, y: float, level: str, gripper: int,
    ) -> list[int]:
        if abs(y) > 0.008:
            raise ValueError("고정된 1번 축의 단일 시상면 작업영역 밖에 있습니다")
        radius = x
        elbow = min(
            self._floor_x_by_elbow,
            key=lambda value: abs(self._floor_x_by_elbow[value] - radius))
        pose = floor_pose(elbow, level, gripper=gripper)
        pose[0] = int(round(self._spec.base_locked_deg))
        return pose

    def _step_seconds(self, seconds: float, *, real_time_scale: float = 0.48):
        steps = max(1, int(round(seconds / self._model.opt.timestep)))
        for _ in range(steps):
            if self._stop.is_set():
                return False
            with self._lock:
                mujoco.mj_step(self._model, self._data)
            if real_time_scale:
                time.sleep(self._model.opt.timestep * real_time_scale)
        return True

    def _drive(self, target: Iterable[float], seconds: float, *, floor=False):
        target = self._validate_pose(target)
        target[0] = self._spec.base_locked_deg
        start = self._pose.copy()
        start[0] = self._spec.base_locked_deg
        segments = max(
            1, int(math.ceil(float(np.max(np.abs(target - start)))
                             / MOTION_SERVO_STEP_DEG)))
        per_segment = seconds / segments
        for fraction in np.linspace(1.0 / segments, 1.0, segments):
            if self._stop.is_set():
                return False
            pose = start + fraction * (target - start)
            with self._lock:
                self._command_pose(pose)
            if not self._step_seconds(per_segment):
                return False
            with self._lock:
                tool = site_position(self._model, self._data, "tool_center")
                if float(np.linalg.norm(tool[:2])) < MIN_SAFE_TOOL_X_M:
                    self._running = False
                    self._phase = "safety_hold"
                    self._add_event(
                        "안전 정지: 손끝이 로봇 몸체 보호 경계를 침범했습니다.", "error")
                    self._stop.set()
                    return False
                if not floor and tool[2] < MIN_AIR_TOOL_Z_M:
                    self._running = False
                    self._phase = "safety_hold"
                    self._add_event(
                        "안전 정지: 비접근 경로에서 바닥 여유 높이가 부족합니다.", "error")
                    self._stop.set()
                    return False
            self._pose = pose.copy()
        return True

    def _render_on_render_thread(
        self, camera: str, width: int, height: int,
    ) -> np.ndarray:
        key = (width, height)
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = mujoco.Renderer(
                self._model, height=height, width=width)
            self._renderers[key] = renderer
        renderer.update_scene(self._data, camera=camera)
        return renderer.render().copy()

    def _render_rgb(self, camera: str, width: int, height: int) -> np.ndarray:
        return self._render_executor.submit(
            self._render_on_render_thread,
            camera, width, height,
        ).result(timeout=5)

    def render_jpeg(self, camera="overview", width=960, height=540) -> bytes:
        if camera not in {"overview", "wrist"}:
            raise ValueError("camera는 overview 또는 wrist여야 합니다")
        width = int(np.clip(width, 160, 1280))
        height = int(np.clip(height, 90, 720))
        with self._lock:
            image = self._render_rgb(camera, width, height)
            if camera == "overview":
                self._annotate_overview(image)
            else:
                self._annotate_wrist(image)
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise RuntimeError("MuJoCo 프레임 JPEG 인코딩 실패")
        return encoded.tobytes()

    def _annotate_overview(self, image: np.ndarray):
        label = f"{self._phase.upper()}  |  MuJoCo contact physics"
        cv2.rectangle(image, (0, 0), (image.shape[1], 34), (15, 17, 20), -1)
        cv2.putText(
            image, label, (12, 23), cv2.FONT_HERSHEY_SIMPLEX,
            0.52, (240, 244, 248), 1, cv2.LINE_AA)

    def _annotate_wrist(self, image: np.ndarray):
        center = self._detector.get("targetCenter")
        marker_row = self._detector.get("markerRow")
        if center:
            cx = int(center[0] * image.shape[1] / 320)
            cy = int(center[1] * image.shape[0] / 180)
            cv2.circle(image, (cx, cy), 10, (0, 255, 255), 2)
        if marker_row is not None:
            y = int(marker_row * image.shape[0] / 180)
            cv2.line(image, (0, y), (image.shape[1], y), (255, 180, 0), 1)

    @staticmethod
    def _mask_for_rgb(image: np.ndarray, rgb: tuple[float, float, float]):
        target = np.asarray(rgb, dtype=np.float32) * 255.0
        delta = np.linalg.norm(image.astype(np.float32) - target, axis=2)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return ((delta < 78.0) & (hsv[:, :, 1] > 80)).astype(np.uint8)

    def _camera_detections(self) -> dict[str, dict[str, object]]:
        """Return only color regions measured from the rendered wrist RGB."""

        with self._lock:
            image = self._render_rgb("wrist", 320, 180)
        blue = self._mask_for_rgb(image, (0.02, 0.20, 1.0))
        red = self._mask_for_rgb(image, (1.0, 0.03, 0.02))
        marker_mask = ((blue > 0) | (red > 0)).astype(np.uint8)
        marker_pixels = np.argwhere(marker_mask > 0)
        marker_row = (float(marker_pixels[:, 0].mean())
                      if len(marker_pixels) >= 8 else None)
        marker_exclusion = cv2.dilate(
            marker_mask, np.ones((7, 7), np.uint8), iterations=1)
        detections: dict[str, dict[str, object]] = {}
        for item in self._objects:
            if item.status == "basket" and item.id != self._last_delivered_id:
                continue
            mask = self._mask_for_rgb(image, _hex_rgb(item.color))
            mask[marker_exclusion > 0] = 0
            count = int(mask.sum())
            if count < 16:
                continue
            components, labels, stats, centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8)
            if components <= 1:
                continue
            component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < 16:
                continue
            center = centroids[component]
            detections[item.id] = {
                "center": [round(float(center[0]), 2),
                           round(float(center[1]), 2)],
                "pixelCount": area,
                "confidence": round(min(0.99, 0.55 + area / 900.0), 3),
            }
        target = detections.get(self._active_id or "")
        self._detector = {
            "source": "wrist_rgb",
            "visibleIds": list(detections),
            "targetCenter": target["center"] if target else None,
            "markerRow": round(marker_row, 2) if marker_row is not None else None,
            "pixelCount": target["pixelCount"] if target else 0,
        }
        return detections

    def _select_from_scan(self) -> SceneObject | None:
        candidates: dict[str, tuple[float, tuple[int, int]]] = {}
        for base_servo, elbow in SCAN_ROUTE:
            if not self._running:
                return None
            with self._lock:
                self._phase = "scanning"
            pose = floor_pose(elbow, "hover")
            pose[0] = base_servo
            if not self._drive(pose, 0.46):
                return None
            detections = self._camera_detections()
            for object_id, detection in detections.items():
                item = self._object(object_id)
                if item.status != "table" or object_id in self._rejected:
                    continue
                score = float(detection["confidence"])
                previous = candidates.get(object_id)
                if previous is None or score > previous[0]:
                    candidates[object_id] = (score, (base_servo, elbow))
        if not candidates:
            return None
        object_id, (score, camera_pose) = max(
            candidates.items(), key=lambda pair: (pair[1][0], pair[1][1]))
        target = self._object(object_id)
        self._active_id = object_id
        self._add_event(
            f"손목 RGB 감지: {target.label} ({score * 100:.0f}%)", "success")
        base_servo, elbow = camera_pose
        pose = floor_pose(elbow, "hover")
        pose[0] = base_servo
        self._drive(pose, 0.35)
        self._camera_detections()
        return target

    def _wait_for_veto(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while self._running and time.monotonic() < deadline:
            if self._reject.wait(timeout=0.04):
                return True
        return self._reject.is_set()

    def _mark_rejected(self, item: SceneObject):
        if item.id not in self._rejected:
            self._rejected.append(item.id)
        self._reject.clear()
        self._active_id = None
        self._last_delivered_id = None
        self._add_event(f"{item.label}: 거부 기억에 추가", "error")

    def _pick(self, item: SceneObject, *, from_basket=False) -> bool:
        if from_basket:
            position = self._object_position(item.id)
            x, y = float(position[0]), float(position[1])
        else:
            x, y = item.x, item.y
        self._phase = "reaching"
        if not self._drive(
                self._floor_pose_world(
                    x, y, "hover", config.GRIP_OPEN), 0.85):
            return False
        self._camera_detections()
        if self._reject.is_set() and not from_basket:
            return False
        self._phase = "grasping"
        if not self._drive(
                self._floor_pose_world(
                    x, y, "grasp", config.GRIP_OPEN),
                0.75, floor=True):
            return False
        before_z = float(self._object_position(item.id)[2])
        if not self._drive(
                self._floor_pose_world(
                    x, y, "grasp", config.GRIP_CLOSED),
                0.70, floor=True):
            return False
        if not self._drive(
                self._floor_pose_world(
                    x, y, "hover", config.GRIP_CLOSED),
                0.90, floor=True):
            return False
        after = self._object_position(item.id)
        tool = site_position(self._model, self._data, "tool_center")
        lifted = after[2] - before_z >= 0.003
        follows = float(np.linalg.norm(after[:2] - tool[:2])) < 0.055
        if not (lifted and follows):
            self._add_event(
                f"{item.label}: 접촉 물리상 들리지 않아 파지 실패", "error")
            self._drive(
                self._floor_pose_world(
                    x, y, "hover", config.GRIP_OPEN), 0.45)
            return False
        item.status = "held"
        self._held_offsets_local[item.id] = after[:2] - tool[:2]
        self._add_event(f"{item.label}: 물리 파지·들림 검증", "success")
        return True

    def _place(
        self, item: SceneObject, x: float, y: float, *, returning=False,
    ) -> bool:
        local_offset = self._held_offsets_local.get(
            item.id, np.zeros(2, dtype=np.float64))
        tool_goal = np.asarray((x, 0.0)) - local_offset
        tool_x, tool_y = float(tool_goal[0]), float(tool_goal[1])
        self._phase = "returning" if returning else "transporting"
        if not self._drive(
                self._floor_pose_world(
                    tool_x, tool_y, "hover", config.GRIP_CLOSED), 0.95):
            return False
        if not self._drive(
                self._floor_pose_world(
                    tool_x, tool_y, "grasp", config.GRIP_CLOSED),
                0.72, floor=True):
            return False
        if not self._drive(
                self._floor_pose_world(
                    tool_x, tool_y, "grasp", config.GRIP_OPEN),
                0.55, floor=True):
            return False
        self._step_seconds(0.32)
        if not self._drive(
                self._floor_pose_world(
                    tool_x, tool_y, "hover", config.GRIP_OPEN),
                0.72, floor=True):
            return False
        position = self._object_position(item.id)
        item.x, item.y = float(position[0]), float(position[1])
        item.status = "table" if returning else "basket"
        self._held_offsets_local.pop(item.id, None)
        return True

    def _return_item(self, item: SceneObject) -> bool:
        self._phase = "returning"
        self._add_event(f"{item.label}: 원위치 복귀 시작", "move")
        if item.status == "basket":
            if not self._pick(item, from_basket=True):
                self._add_event(
                    f"{item.label}: 바구니 회수 실패로 안전 정지", "error")
                return False
        elif item.status != "held":
            self._mark_rejected(item)
            return True
        if not self._place(
                item, item.origin_x, item.origin_y, returning=True):
            return False
        item.x, item.y = self._object_position(item.id)[:2]
        self._add_event(f"{item.label}: 원래 위치에 내려놓음", "success")
        self._mark_rejected(item)
        return True

    def _run(self):
        try:
            self._stop.clear()
            self._reject.clear()
            while self._running and not self._stop.is_set():
                available = [
                    item for item in self._objects
                    if item.status == "table" and item.id not in self._rejected
                ]
                if not available:
                    table_items = [
                        item for item in self._objects if item.status == "table"]
                    if not table_items:
                        self._phase = "completed"
                        self._running = False
                        self._add_event("테이블에 남은 물체가 없습니다.", "success")
                        return
                    self._rejected.clear()
                    self._cycle += 1
                    self._add_event(
                        "모든 물체가 거부되어 거부 기억을 초기화했습니다.", "info")

                target = self._select_from_scan()
                if target is None:
                    self._phase = "paused"
                    self._running = False
                    self._add_event(
                        "손목 RGB 스캔에서 물체를 찾지 못했습니다.", "error")
                    return
                self._phase = "target"
                self._add_event(
                    f"후보 제시: {target.label} · ErrP 판정 창", "move")
                if self._wait_for_veto(TARGET_REVIEW_SECONDS):
                    self._return_item(target)
                    continue

                if not self._pick(target):
                    if self._reject.is_set():
                        self._return_item(target)
                        continue
                    self._phase = "paused"
                    self._running = False
                    return
                if self._reject.is_set():
                    self._return_item(target)
                    continue

                if not self._place(
                        target, self._basket_x, self._basket_y):
                    self._phase = "paused"
                    self._running = False
                    return
                self._last_delivered_id = target.id
                self._phase = "evaluating"
                self._add_event(
                    f"{target.label}: 바구니 도착 · "
                    f"{POST_DELIVERY_REVIEW_SECONDS:.0f}초 ErrP 연속 검토",
                    "success")
                if self._wait_for_veto(POST_DELIVERY_REVIEW_SECONDS):
                    self._return_item(target)
                    continue
                self._phase = "completed"
                self._running = False
                self._active_id = target.id
                self._add_event(
                    f"{target.label}: 거부 없음 · 배송 확정", "success")
                return
        except Exception as exc:
            with self._lock:
                self._running = False
                self._phase = "error"
                self._add_event(
                    f"{type(exc).__name__}: {exc}", "error")

    def start(self):
        with self._lock:
            if self._running:
                return self.status()
            if not any(item.status == "table" for item in self._objects):
                raise RuntimeError("테이블에 물체를 하나 이상 놓으세요")
            self._running = True
            self._phase = "scanning"
            self._stop.clear()
            self._reject.clear()
            self._add_event(f"자동 3D 사이클 {self._cycle} 시작", "info")
            self._worker = Thread(
                target=self._run, name="mujoco-studio-task", daemon=True)
            self._worker.start()
            return self.status()

    def stop(self):
        with self._lock:
            self._running = False
            self._stop.set()
            if self._phase not in {"idle", "completed"}:
                self._phase = "paused"
                self._add_event("3D 시뮬레이션을 일시정지했습니다.", "info")
            return self.status()

    def reject(self):
        with self._lock:
            target_id = self._active_id or self._last_delivered_id
            if not target_id:
                raise RuntimeError("현재 거부할 선택 물체가 없습니다")
            self._reject.set()
            self._add_event("외부 '아니야' 신호 수신", "error")
            if not self._running:
                target = self._object(target_id)
                if target.status != "basket":
                    self._mark_rejected(target)
                    return self.status()
                self._running = True
                self._stop.clear()

                def late_return():
                    try:
                        if self._return_item(target):
                            self._run()
                        else:
                            with self._lock:
                                self._running = False
                                self._phase = "paused"
                    except Exception as exc:
                        with self._lock:
                            self._running = False
                            self._phase = "error"
                            self._add_event(
                                f"{type(exc).__name__}: {exc}", "error")

                self._worker = Thread(
                    target=late_return, name="mujoco-studio-late-reject",
                    daemon=True)
                self._worker.start()
            return self.status()

    def reset(self):
        self.stop()
        worker = self._worker
        if worker and worker.is_alive():
            worker.join(timeout=2)
        with self._lock:
            for item in self._objects:
                item.x, item.y = item.origin_x, item.origin_y
                item.status = "table"
            self._rejected.clear()
            self._active_id = None
            self._last_delivered_id = None
            self._held_offsets_local.clear()
            self._phase = "idle"
            self._running = False
            self._cycle = 1
            self._stop.clear()
            self._reject.clear()
            self._rebuild()
            self._add_event("3D 장면과 물체 원점을 초기화했습니다.", "info")
            return self.status()

    def _require_editable(self):
        if self._running:
            raise RuntimeError("실행 중에는 장면을 편집할 수 없습니다")

    def add_object(self, shape="box", color=None, label=None):
        with self._lock:
            self._require_editable()
            if len(self._objects) >= MAX_OBJECTS:
                raise RuntimeError(f"물체는 최대 {MAX_OBJECTS}개까지 추가할 수 있습니다")
            shape = str(shape)
            if shape not in {"box", "cylinder", "sphere"}:
                raise ValueError("지원하지 않는 물체 모양입니다")
            index = len(self._objects)
            color = color or DEFAULT_COLORS[index % len(DEFAULT_COLORS)]
            _hex_rgb(color)
            x = 0.387 if index == 0 else 0.414
            y = 0.0
            item = SceneObject(
                f"object-{uuid.uuid4().hex[:8]}",
                str(label or f"물체 {index + 1}")[:32],
                shape, color, 0.006, x, y, x, y)
            self._objects.append(item)
            self._rebuild()
            self._add_event(f"{item.label}: 3D 장면에 추가", "info")
            return self.status()

    def update_object(self, object_id: str, payload: dict):
        with self._lock:
            self._require_editable()
            item = self._object(str(object_id))
            if item.status != "table":
                raise RuntimeError("테이블 위 물체만 편집할 수 있습니다")
            if "label" in payload:
                item.label = str(payload["label"]).strip()[:32] or item.label
            if "shape" in payload:
                shape = str(payload["shape"])
                if shape not in {"box", "cylinder", "sphere"}:
                    raise ValueError("지원하지 않는 물체 모양입니다")
                item.shape = shape
            if "color" in payload:
                color = str(payload["color"])
                _hex_rgb(color)
                item.color = color
            if "sizeMm" in payload:
                item.size_m = float(np.clip(
                    float(payload["sizeMm"]) / 1000.0, 0.0045, 0.0080))
            if "xMm" in payload:
                item.x = float(payload["xMm"]) / 1000.0
            if "yMm" in payload:
                item.y = float(payload["yMm"]) / 1000.0
            item.x = float(np.clip(
                item.x, WORKSPACE_RADIUS[0], WORKSPACE_RADIUS[1]))
            item.y = 0.0
            item.origin_x, item.origin_y = item.x, item.y
            self._rebuild()
            self._add_event(f"{item.label}: 배치/외형 갱신", "info")
            return self.status()

    def delete_object(self, object_id: str):
        with self._lock:
            self._require_editable()
            item = self._object(str(object_id))
            self._objects.remove(item)
            self._rejected = [value for value in self._rejected
                              if value != object_id]
            self._active_id = None
            self._last_delivered_id = None
            self._rebuild()
            self._add_event(f"{item.label}: 3D 장면에서 삭제", "info")
            return self.status()

    def update_basket(self, x_mm, y_mm=0):
        with self._lock:
            self._require_editable()
            x = float(x_mm) / 1000.0
            float(y_mm)  # validate input even though fixed-base mode forces y=0
            self._basket_x = float(np.clip(
                x, WORKSPACE_RADIUS[0], WORKSPACE_RADIUS[1]))
            self._basket_y = 0.0
            self._rebuild()
            self._add_event("목표 트레이 위치를 갱신했습니다.", "info")
            return self.status()

    def status(self):
        with self._lock:
            objects = []
            for item in self._objects:
                payload = asdict(item)
                try:
                    position = self._object_position(item.id)
                    payload.update({
                        "xMm": round(float(position[0]) * 1000.0, 2),
                        "yMm": round(float(position[1]) * 1000.0, 2),
                        "zMm": round(float(position[2]) * 1000.0, 2),
                    })
                except (KeyError, IndexError):
                    payload.update({
                        "xMm": round(item.x * 1000.0, 2),
                        "yMm": round(item.y * 1000.0, 2),
                        "zMm": round(item.half_height * 1000.0, 2),
                    })
                payload["sizeMm"] = round(item.size_m * 1000.0, 2)
                payload["originXmm"] = round(item.origin_x * 1000.0, 2)
                payload["originYmm"] = round(item.origin_y * 1000.0, 2)
                objects.append(payload)
            tool = site_position(self._model, self._data, "tool_center")
            return {
                "engine": "MuJoCo",
                "physics": True,
                "cameraOnlySelection": True,
                "running": self._running,
                "phase": self._phase,
                "cycle": self._cycle,
                "activeId": self._active_id,
                "lastDeliveredId": self._last_delivered_id,
                "postDeliveryReviewSeconds": POST_DELIVERY_REVIEW_SECONDS,
                "rejectedIds": list(self._rejected),
                "objects": objects,
                "basket": {
                    "xMm": round(self._basket_x * 1000.0, 2),
                    "yMm": round(self._basket_y * 1000.0, 2),
                },
                "servoDeg": [round(float(value), 2) for value in self._pose],
                "toolMm": [round(float(value) * 1000.0, 2) for value in tool],
                "workspace": {
                    "radiusMm": [
                        round(value * 1000, 1) for value in WORKSPACE_RADIUS],
                    "yawDeg": list(WORKSPACE_YAW_DEG),
                    "baseNeutralDeg": self._spec.base_locked_deg,
                    "baseMode": "fixed-90",
                },
                "detector": dict(self._detector),
                "events": [asdict(event) for event in self._events[:24]],
            }


__all__ = [
    "MAX_OBJECTS",
    "MuJoCoStudio",
    "SceneObject",
    "WORKSPACE_RADIUS",
    "WORKSPACE_YAW_DEG",
    "build_studio_mjcf",
]
