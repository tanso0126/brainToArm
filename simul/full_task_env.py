"""Transfer-oriented complete wrist-camera floor-pick environment.

Unlike an end-to-end simulator policy that can exploit incorrect synthetic
pixels, this environment trains decisions on quantities available from the real
PW315 pipeline: image quality, selected-candidate continuity, two finger
markers, target-to-jaw image error, jaw opening, lift motion, commanded servo
angles, and the previous action.  The image-y response uses the measured
hardware Jacobian (-12.9 px per +1 elbow degree).

The learned policy chooses safe macro actions.  Existing deterministic code
still converts those actions to bounded floor poses and owns physical safety.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

try:
    from laptop import config
    from laptop.floor_motion import floor_pose
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laptop"))
    import config
    from floor_motion import floor_pose


class TaskAction(IntEnum):
    WAIT = 0
    SEARCH_NEXT = 1
    ALIGN_ELBOW_DOWN = 2
    ALIGN_ELBOW_UP = 3
    DESCEND = 4
    CLOSE = 5
    LIFT = 6
    RECOVER = 7


OBSERVATION_NAMES = (
    "quality_valid",
    "target_visible",
    "markers_visible",
    "selection_continuous",
    "depth_error_normalized",
    "centerline_error_normalized",
    "jaw_opening_normalized",
    "coherent_lift_motion",
    "servo_base",
    "servo_shoulder",
    "servo_elbow",
    "servo_wrist_pitch",
    "servo_gripper",
    "servo_wrist_roll",
    "previous_action_normalized",
)


class FullFloorPickEnv(gym.Env):
    """Search -> align -> descend -> close -> lift with sensor corruption."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        domain_randomization: bool = True,
        max_steps: int = 60,
        seed: int | None = None,
    ):
        super().__init__()
        self.domain_randomization = bool(domain_randomization)
        self.max_steps = int(max_steps)
        self.action_space = spaces.Discrete(len(TaskAction))
        self.observation_space = spaces.Box(
            -1.0, 1.0, (len(OBSERVATION_NAMES),), dtype=np.float32)
        self.np_random = np.random.default_rng(seed)
        self.target_elbow = 90
        self.current_elbow = 90
        self.centerline_error_px = 0.0
        self.visibility_span_deg = 14
        self.jacobian_px_per_deg = float(config.FLOOR_ALIGN_DY_PER_ELBOW)
        self.pose_level = "home"
        self.gripper_open = True
        self.contact = False
        self.holding = False
        self.search_index = 0
        self.steps = 0
        self.previous_action = TaskAction.WAIT
        self.last_observation = np.zeros(len(OBSERVATION_NAMES), np.float32)
        self.last_event = "RESET"
        self.milestones = set()
        self.recoveries = 0

    @property
    def commanded_pose(self) -> list[int]:
        if self.pose_level == "home":
            return list(config.HOME_POSE)
        level = "grasp" if self.pose_level == "grasp" else "hover"
        grip = config.GRIP_OPEN if self.gripper_open else config.GRIP_CLOSED
        return floor_pose(self.current_elbow, level, gripper=grip)

    def _target_is_geometrically_visible(self) -> bool:
        return (self.pose_level in {"hover", "grasp"}
                and abs(self.current_elbow - self.target_elbow)
                <= self.visibility_span_deg)

    def _true_depth_error(self) -> float:
        # Matches FloorGraspController: jaw_y - target_y. Dividing by the
        # negative measured gain yields target-current elbow correction.
        return -self.jacobian_px_per_deg * (
            self.current_elbow - self.target_elbow)

    @staticmethod
    def _normalize_servo(pose) -> np.ndarray:
        minimum = np.asarray(config.SERVO_MIN, dtype=np.float32)
        maximum = np.asarray(config.SERVO_MAX, dtype=np.float32)
        return 2.0 * (np.asarray(pose, dtype=np.float32) - minimum) / (
            maximum - minimum) - 1.0

    def _observe(self) -> np.ndarray:
        randomized = self.domain_randomization
        quality = not (randomized and self.np_random.random() < 0.035)
        markers = not (randomized and self.np_random.random() < 0.045)
        target = self._target_is_geometrically_visible()
        if randomized and self.np_random.random() < 0.055:
            target = False
        continuity = target and not (
            randomized and self.np_random.random() < 0.025)

        depth = self._true_depth_error() if target else 0.0
        lateral = self.centerline_error_px if target else 0.0
        if randomized and target:
            depth += float(self.np_random.normal(0.0, 8.0))
            lateral += float(self.np_random.normal(0.0, 10.0))

        if self.gripper_open:
            jaw = 1.0
        elif self.contact:
            jaw = 0.28
        else:
            jaw = 0.0
        if randomized:
            jaw = float(np.clip(jaw + self.np_random.normal(0.0, 0.04), 0, 1))
        lift = 1.0 if self.holding else 0.0
        if randomized:
            lift = float(np.clip(lift + self.np_random.normal(0.0, 0.035), 0, 1))

        observation = np.concatenate((
            np.asarray((
                1.0 if quality else -1.0,
                1.0 if target else -1.0,
                1.0 if markers else -1.0,
                1.0 if continuity else -1.0,
                np.clip(depth / 420.0, -1.0, 1.0),
                np.clip(lateral / 220.0, -1.0, 1.0),
                2.0 * jaw - 1.0,
                2.0 * lift - 1.0,
            ), dtype=np.float32),
            self._normalize_servo(self.commanded_pose),
            np.asarray((
                2.0 * int(self.previous_action) / (len(TaskAction) - 1) - 1.0,
            ), dtype=np.float32),
        ))
        self.last_observation = observation.astype(np.float32)
        return self.last_observation.copy()

    def _info(self) -> dict[str, Any]:
        return {
            "event": self.last_event,
            "target_elbow": self.target_elbow,
            "current_elbow": self.current_elbow,
            "pose_level": self.pose_level,
            "contact": self.contact,
            "holding": self.holding,
            "true_depth_error_px": self._true_depth_error(),
            "centerline_error_px": self.centerline_error_px,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        options = options or {}
        lower, upper = config.FLOOR_ELBOW_RANGE
        self.target_elbow = int(options.get(
            "target_elbow", self.np_random.integers(lower, upper + 1)))
        self.current_elbow = int(options.get("current_elbow", config.FLOOR_REFERENCE_ELBOW))
        self.centerline_error_px = float(options.get(
            "centerline_error_px", self.np_random.uniform(-95.0, 95.0)))
        self.visibility_span_deg = int(options.get(
            "visibility_span_deg", self.np_random.integers(9, 18)
            if self.domain_randomization else 14))
        self.jacobian_px_per_deg = float(
            self.np_random.uniform(-15.5, -10.5)
            if self.domain_randomization else config.FLOOR_ALIGN_DY_PER_ELBOW)
        self.pose_level = str(options.get("pose_level", "home"))
        self.gripper_open = True
        self.contact = False
        self.holding = False
        self.search_index = 0
        self.steps = 0
        self.previous_action = TaskAction.WAIT
        self.last_event = "RESET"
        self.milestones = set()
        self.recoveries = 0
        return self._observe(), self._info()

    def expert_action(self) -> int:
        quality, target, markers, continuity = self.last_observation[:4] > 0
        if not quality or not markers:
            return int(TaskAction.WAIT)
        if self.pose_level == "grasp":
            if self.gripper_open:
                return int(TaskAction.CLOSE if target and continuity
                           else TaskAction.RECOVER)
            return int(TaskAction.LIFT if self.contact else TaskAction.RECOVER)
        if not target or not continuity:
            return int(TaskAction.SEARCH_NEXT)
        if self.pose_level == "hover":
            error = self._true_depth_error()
            if abs(error) > config.FLOOR_ALIGN_TOL_PX:
                return int(TaskAction.ALIGN_ELBOW_UP
                           if self.target_elbow > self.current_elbow
                           else TaskAction.ALIGN_ELBOW_DOWN)
            return int(TaskAction.DESCEND)
        return int(TaskAction.RECOVER)

    def step(self, action):
        action = TaskAction(int(action))
        before_error = abs(self.current_elbow - self.target_elbow)
        reward = -0.025
        self.last_event = action.name
        terminated = False

        valid_sensors = self.last_observation[0] > 0 and self.last_observation[2] > 0
        target_seen = self.last_observation[1] > 0 and self.last_observation[3] > 0
        lower, upper = config.FLOOR_ELBOW_RANGE

        if action == TaskAction.WAIT:
            # Waiting is correct only for a transient bad frame. At a valid,
            # visible state it creates an infinite stall and is penalized.
            if valid_sensors and (target_seen or self.pose_level != "home"):
                reward -= 0.30
            else:
                reward -= 0.01
        elif action == TaskAction.SEARCH_NEXT:
            if target_seen and valid_sensors:
                reward -= 0.28
                self.last_event += "_UNNECESSARY"
            else:
                self.pose_level = "hover"
                self.gripper_open = True
                # Alternating far/near viewpoints covers the complete calibrated
                # band and changes motors 2/3/4 through floor_pose.
                route = (90, 78, 110, 84, 102, 94)
                self.current_elbow = route[self.search_index % len(route)]
                self.search_index += 1
                if (self._target_is_geometrically_visible()
                        and "found" not in self.milestones):
                    self.milestones.add("found")
                    reward += 0.30
                else:
                    reward -= 0.04
        elif action in (TaskAction.ALIGN_ELBOW_DOWN, TaskAction.ALIGN_ELBOW_UP):
            if self.pose_level != "hover" or not valid_sensors or not target_seen:
                reward -= 3.0
                self.last_event += "_INVALID"
                terminated = True
            elif abs(self._true_depth_error()) <= 0.65 * config.FLOOR_ALIGN_TOL_PX:
                reward -= 0.30
                self.last_event += "_UNNECESSARY"
            else:
                direction = -1 if action == TaskAction.ALIGN_ELBOW_DOWN else 1
                remaining = abs(self.target_elbow - self.current_elbow)
                amount = max(1, min(config.FLOOR_ALIGN_MAX_ELBOW_STEP, remaining))
                self.current_elbow = int(np.clip(
                    self.current_elbow + direction * amount,
                    lower, upper))
                progress = before_error - abs(self.current_elbow - self.target_elbow)
                reward += 0.12 * progress
        elif action == TaskAction.DESCEND:
            aligned = (abs(self._true_depth_error()) <= config.FLOOR_ALIGN_TOL_PX
                       and abs(self.centerline_error_px)
                       <= config.FLOOR_ALIGN_X_CENTERLINE_TOL_PX)
            if self.pose_level == "hover" and valid_sensors and target_seen and aligned:
                self.pose_level = "grasp"
                if "descended" not in self.milestones:
                    self.milestones.add("descended")
                    reward += 0.55
            else:
                reward -= 4.0
                self.last_event += "_BLOCKED"
                terminated = True
        elif action == TaskAction.CLOSE:
            if self.pose_level == "grasp" and self.gripper_open:
                self.gripper_open = False
                self.contact = (
                    abs(self._true_depth_error()) <= config.FLOOR_ALIGN_TOL_PX
                    and abs(self.centerline_error_px) <= 115.0)
                if self.contact and "contact" not in self.milestones:
                    self.milestones.add("contact")
                    reward += 0.9
                elif not self.contact:
                    reward -= 0.7
                self.last_event += "_CONTACT" if self.contact else "_FREE"
            else:
                reward -= 4.0
                self.last_event += "_INVALID"
                terminated = True
        elif action == TaskAction.LIFT:
            if self.pose_level == "grasp" and not self.gripper_open and self.contact:
                self.pose_level = "hover"
                self.holding = True
                reward += 12.0
                terminated = True
                self.last_event += "_SUCCESS"
            else:
                reward -= 5.0
                self.last_event += "_FAILED"
                terminated = True
        elif action == TaskAction.RECOVER:
            self.pose_level = "hover"
            self.gripper_open = True
            self.contact = False
            self.holding = False
            self.recoveries += 1
            reward -= 0.20 + 0.08 * self.recoveries

        self.previous_action = action
        self.steps += 1
        truncated = self.steps >= self.max_steps and not terminated
        observation = self._observe()
        return observation, float(reward), terminated, truncated, self._info()


__all__ = ["FullFloorPickEnv", "OBSERVATION_NAMES", "TaskAction"]
