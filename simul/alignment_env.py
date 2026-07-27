"""Eye-in-hand floor-alignment environment with a strict deployment boundary.

The actor observes only wrist RGB, commanded servo angles, and its previous
action.  Object/world coordinates are retained privately for reset, reward,
and an expert teacher.  This environment never imports the serial or webcam
modules.
"""

from __future__ import annotations

import colorsys
from typing import Any

import cv2
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

try:
    from .mujoco_robot import (
        RobotSpec,
        load_model,
        place_target,
        set_servo_pose,
        site_position,
    )
except ImportError:
    from mujoco_robot import (
        RobotSpec,
        load_model,
        place_target,
        set_servo_pose,
        site_position,
    )

try:
    from laptop.floor_motion import floor_pose
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laptop"))
    from floor_motion import floor_pose


IMAGE_WIDTH = 128
IMAGE_HEIGHT = 72
ELBOW_MIN = 78
ELBOW_MAX = 110
MAX_ELBOW_STEP = 2
SUCCESS_X_M = 0.0025


def _vivid_rgb(rng: np.random.Generator) -> tuple[float, float, float]:
    hue = float(rng.uniform(0.0, 1.0))
    saturation = float(rng.uniform(0.58, 1.0))
    value = float(rng.uniform(0.58, 1.0))
    return colorsys.hsv_to_rgb(hue, saturation, value)


def _floor_rgb(rng: np.random.Generator) -> tuple[float, float, float]:
    hue = float(rng.uniform(0.0, 1.0))
    saturation = float(rng.uniform(0.0, 0.22))
    value = float(rng.uniform(0.25, 0.95))
    return colorsys.hsv_to_rgb(hue, saturation, value)


class WristAlignmentEnv(gym.Env):
    """Local ground-plane visual servo task for the fixed-base robot."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        *,
        domain_randomization: bool = True,
        image_augmentation: bool = True,
        max_steps: int = 20,
        seed: int | None = None,
    ):
        super().__init__()
        self.domain_randomization = bool(domain_randomization)
        self.image_augmentation = bool(image_augmentation)
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (3, IMAGE_HEIGHT, IMAGE_WIDTH), np.uint8),
            "servo": spaces.Box(-1.0, 1.0, (6,), np.float32),
            "previous_action": spaces.Box(-1.0, 1.0, (1,), np.float32),
        })
        self.action_space = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.robot_spec = RobotSpec.from_manifest()
        self.model = None
        self.data = None
        self.renderer = None
        self.current_elbow = 90
        self.target_elbow = 90
        self.commanded_pose = self.robot_spec.hover_servo_deg.copy()
        self.previous_action = np.zeros(1, dtype=np.float32)
        self.target_xyz = np.zeros(3, dtype=np.float64)
        self.steps = 0
        self.np_random = np.random.default_rng(seed)

    def _rebuild(self):
        if self.model is None:
            self.model, self.data, _ = load_model(self.robot_spec)
            self.renderer = mujoco.Renderer(
                self.model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

    def _randomize_runtime(self) -> float:
        """Change mutable model arrays without recompiling Metal shaders."""

        target_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
        if not self.domain_randomization:
            self.model.geom_type[target_id] = mujoco.mjtGeom.mjGEOM_BOX
            self.model.geom_size[target_id, :3] = (0.020, 0.020, 0.018)
            self.model.geom_rgba[target_id] = (1.0, 0.82, 0.02, 1.0)
            self.model.cam_pos[camera_id] = (0.015, 0.0, 0.075)
            self.model.cam_fovy[camera_id] = 73.0
            return 0.020

        target_size = float(self.np_random.uniform(0.012, 0.028))
        target_shape = str(self.np_random.choice(("box", "cylinder", "sphere")))
        geom_types = {
            "box": mujoco.mjtGeom.mjGEOM_BOX,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
            "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
        }
        self.model.geom_type[target_id] = geom_types[target_shape]
        self.model.geom_size[target_id, :3] = 0
        if target_shape == "box":
            self.model.geom_size[target_id, :3] = (
                target_size, target_size, target_size * 0.9)
        elif target_shape == "cylinder":
            self.model.geom_size[target_id, :2] = (target_size, target_size * 0.9)
        else:
            self.model.geom_size[target_id, 0] = target_size
        self.model.geom_rgba[target_id] = (*_vivid_rgb(self.np_random), 1.0)
        self.model.cam_pos[camera_id] = (
            float(self.np_random.uniform(0.008, 0.023)), 0.0,
            float(self.np_random.uniform(0.066, 0.086)))
        self.model.cam_fovy[camera_id] = float(self.np_random.uniform(64.0, 84.0))
        light_scale = float(self.np_random.uniform(0.65, 1.15))
        nominal_lights = np.asarray(((0.9, 0.9, 0.9), (0.45, 0.48, 0.52)))
        self.model.light_diffuse[:] = np.clip(
            nominal_lights[:self.model.nlight] * light_scale, 0.15, 1.0)
        return target_size

    def _servo_observation(self) -> np.ndarray:
        span = self.robot_spec.servo_max_deg - self.robot_spec.servo_min_deg
        return (2.0 * (self.commanded_pose - self.robot_spec.servo_min_deg) / span - 1.0).astype(
            np.float32)

    def _augment(self, image: np.ndarray) -> np.ndarray:
        if not (self.domain_randomization and self.image_augmentation):
            return image
        value = image.astype(np.float32) / 255.0
        white_balance = self.np_random.uniform(0.78, 1.22, 3).astype(np.float32)
        value *= white_balance[None, None, :]
        value *= float(self.np_random.uniform(0.58, 1.42))
        gamma = float(self.np_random.uniform(0.72, 1.38))
        value = np.power(np.clip(value, 0.0, 1.0), gamma)
        if self.np_random.random() < 0.35:
            kernel = int(self.np_random.choice((3, 5)))
            value = cv2.GaussianBlur(value, (kernel, kernel), 0)
        # Pixel-space installation jitter covers small roll/translation/scale
        # errors without rebuilding the MuJoCo model and leaking Metal shader
        # variants over long training runs.
        angle = float(self.np_random.uniform(-5.0, 5.0))
        scale = float(self.np_random.uniform(0.92, 1.08))
        transform = cv2.getRotationMatrix2D(
            (IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2), angle, scale)
        transform[:, 2] += self.np_random.uniform((-5.0, -3.0), (5.0, 3.0))
        value = cv2.warpAffine(
            value, transform, (IMAGE_WIDTH, IMAGE_HEIGHT),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        noise = self.np_random.normal(
            0.0, self.np_random.uniform(0.0, 0.035), value.shape)
        value = np.clip(value + noise, 0.0, 1.0)
        return (value * 255.0 + 0.5).astype(np.uint8)

    def _observation(self) -> dict[str, np.ndarray]:
        self.renderer.update_scene(self.data, camera="wrist")
        image = self._augment(self.renderer.render().copy())
        return {
            "image": np.transpose(image, (2, 0, 1)),
            "servo": self._servo_observation(),
            "previous_action": self.previous_action.copy(),
        }

    def _tool_x(self) -> float:
        return float(site_position(self.model, self.data, "tool_center")[0])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        options = options or {}
        self._rebuild()
        target_size = self._randomize_runtime()
        self.current_elbow = int(options.get(
            "current_elbow", self.np_random.integers(ELBOW_MIN, ELBOW_MAX + 1)))
        self.target_elbow = int(options.get(
            "target_elbow", self.np_random.integers(ELBOW_MIN, ELBOW_MAX + 1)))
        if not ELBOW_MIN <= self.current_elbow <= ELBOW_MAX:
            raise ValueError("current_elbow outside calibrated floor range")
        if not ELBOW_MIN <= self.target_elbow <= ELBOW_MAX:
            raise ValueError("target_elbow outside calibrated floor range")

        target_pose = floor_pose(self.target_elbow, "hover")
        set_servo_pose(self.model, self.data, target_pose, spec=self.robot_spec)
        target_tool = site_position(self.model, self.data, "tool_center")
        self.target_xyz = np.array((
            target_tool[0],
            float(options.get("target_y", self.np_random.uniform(-0.007, 0.007))),
            target_size,
        ))

        self.commanded_pose = np.asarray(
            floor_pose(self.current_elbow, "hover"), dtype=np.float64)
        set_servo_pose(self.model, self.data, self.commanded_pose, spec=self.robot_spec)
        place_target(self.model, self.data, self.target_xyz)
        self.previous_action[:] = 0
        self.steps = 0
        observation = self._observation()
        return observation, self._info()

    def _info(self) -> dict[str, Any]:
        # Privileged values are diagnostic/training metadata only and are not
        # included in observation_space or the exported actor signature.
        return {
            "current_elbow": self.current_elbow,
            "target_elbow": self.target_elbow,
            "x_error_m": self.target_xyz[0] - self._tool_x(),
            "success": abs(self.target_xyz[0] - self._tool_x()) <= SUCCESS_X_M,
        }

    def expert_action(self) -> np.ndarray:
        if abs(self.target_xyz[0] - self._tool_x()) <= SUCCESS_X_M:
            return np.zeros(1, dtype=np.float32)
        difference = self.target_elbow - self.current_elbow
        return np.asarray(
            [np.clip(difference / MAX_ELBOW_STEP, -1.0, 1.0)], dtype=np.float32)

    def step(self, action):
        value = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1, 1))
        delta = int(round(value * MAX_ELBOW_STEP))
        if delta == 0 and abs(value) >= 0.25:
            delta = 1 if value > 0 else -1
        self.current_elbow = int(np.clip(
            self.current_elbow + delta, ELBOW_MIN, ELBOW_MAX))
        self.commanded_pose = np.asarray(
            floor_pose(self.current_elbow, "hover"), dtype=np.float64)
        set_servo_pose(self.model, self.data, self.commanded_pose, spec=self.robot_spec)
        place_target(self.model, self.data, self.target_xyz)
        self.previous_action[:] = value
        self.steps += 1
        info = self._info()
        terminated = bool(info["success"])
        truncated = self.steps >= self.max_steps and not terminated
        distance = abs(float(info["x_error_m"]))
        reward = 2.0 if terminated else -0.02 - 8.0 * distance
        return self._observation(), reward, terminated, truncated, info

    def render(self):
        observation = self._observation()
        return np.transpose(observation["image"], (1, 2, 0))

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


__all__ = [
    "ELBOW_MAX", "ELBOW_MIN", "IMAGE_HEIGHT", "IMAGE_WIDTH",
    "MAX_ELBOW_STEP", "SUCCESS_X_M", "WristAlignmentEnv",
]
