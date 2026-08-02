"""Transfer-oriented task environment for the rigid-wrist 2-DOF arm.

The policy never sees MuJoCo world coordinates.  Its inputs are quantities the
real wrist-camera/ultrasonic pipeline can provide.  A deterministic controller
still owns the 2x2 Jacobian and collision checks; learning selects task macros.
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Any
import sys

import gymnasium as gym
from gymnasium import spaces
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "laptop") not in sys.path:
    sys.path.insert(0, str(ROOT / "laptop"))

from laptop import config  # noqa: E402
from laptop.reduced_dof import (  # noqa: E402
    ACTIVE_MOTION_JOINTS, camera_state_mm, command_pose,
    fingertip_floor_clearance_mm, reduced_home,
)


class ReducedTaskAction(IntEnum):
    WAIT = 0
    SEARCH_NEXT = 1
    APPROACH = 2
    CLOSE = 3
    LIFT = 4
    RETURN_HOME = 5
    RECOVER = 6
    DONE = 7


OBSERVATION_NAMES = (
    "quality_valid", "target_visible", "markers_visible",
    "selection_continuous", "sonar_valid", "image_vertical_error",
    "image_horizontal_error", "sonar_distance", "floor_clearance",
    "jaw_opening", "coherent_lift", "servo_shoulder", "servo_elbow",
    "servo_gripper", "task_phase", "previous_action",
)

PHASE_SEARCH, PHASE_APPROACH, PHASE_GRASPED, PHASE_LIFTED, PHASE_RETURNED = range(5)

# Safe, near-floor configurations found by the same rigid-wrist collision model
# used on the laptop.  Domain randomization interpolates among these anchors.
GRASP_POSES = (
    # Each anchor and its complete HOME->grasp->lift path keeps at least
    # ~10 mm clearance *after* the 12 mm housing/model safety margin.
    (128, 65), (132, 63), (134, 57), (136, 51),
    (140, 49), (142, 43), (144, 39),
)


class ReducedFloorPickEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, *, domain_randomization=True, max_steps=70, seed=None):
        super().__init__()
        self.domain_randomization = bool(domain_randomization)
        self.max_steps = int(max_steps)
        self.action_space = spaces.Discrete(len(ReducedTaskAction))
        self.observation_space = spaces.Box(
            -1.0, 1.0, (len(OBSERVATION_NAMES),), dtype=np.float32)
        self.np_random = np.random.default_rng(seed)
        self.current_pose = reduced_home()
        self.target_pose = command_pose(*GRASP_POSES[0])
        self.phase = PHASE_SEARCH
        self.previous_action = ReducedTaskAction.WAIT
        self.gripper_open = True
        self.contact = False
        self.holding = False
        self.selection_locked = False
        self.horizontal_error_px = 0.0
        self.distance_scale = 5.4
        self.steps = 0
        self.last_event = "RESET"
        self.last_observation = np.zeros(len(OBSERVATION_NAMES), np.float32)

    @staticmethod
    def _joint_error(pose, target):
        return np.asarray([
            target[config.J_SHOULDER] - pose[config.J_SHOULDER],
            target[config.J_ELBOW] - pose[config.J_ELBOW],
        ], dtype=float)

    def _visible_geometry(self):
        return float(np.max(np.abs(self._joint_error(
            self.current_pose, self.target_pose)))) <= 34.0

    def _distance_mm(self):
        error = self._joint_error(self.current_pose, self.target_pose)
        return 50.0 + self.distance_scale * float(np.linalg.norm(error))

    def _vertical_error_px(self):
        delta = camera_state_mm(self.target_pose) - camera_state_mm(self.current_pose)
        # A bounded image-space proxy; no target/world pose is exposed to policy.
        return float(np.clip(delta[1] / 0.045, -420.0, 420.0))

    @staticmethod
    def _servo_normalized(value, low, high):
        return 2.0 * (float(value) - low) / (high - low) - 1.0

    def _observe(self):
        random = self.domain_randomization
        quality = not (random and self.np_random.random() < 0.035)
        markers = not (random and self.np_random.random() < 0.045)
        visible = self._visible_geometry()
        if random and self.np_random.random() < 0.055:
            visible = False
        if visible:
            self.selection_locked = True
        continuity = self.selection_locked and not (
            random and self.np_random.random() < 0.025)
        sonar_valid = visible and not (random and self.np_random.random() < 0.07)
        vertical = self._vertical_error_px() if visible else 0.0
        horizontal = self.horizontal_error_px if visible else 0.0
        distance = self._distance_mm() if sonar_valid else 400.0
        clearance = fingertip_floor_clearance_mm(self.current_pose)
        if random:
            if visible:
                vertical += float(self.np_random.normal(0, 8.0))
                horizontal += float(self.np_random.normal(0, 9.0))
            if sonar_valid:
                distance += float(self.np_random.normal(0, 7.0))
            clearance += float(self.np_random.normal(0, 3.0))
        jaw = 1.0 if self.gripper_open else (0.34 if self.contact else 0.0)
        lift = 1.0 if self.holding and self.phase >= PHASE_LIFTED else 0.0
        observation = np.asarray((
            1 if quality else -1,
            1 if visible else -1,
            1 if markers else -1,
            1 if continuity else -1,
            1 if sonar_valid else -1,
            np.clip(vertical / 420.0, -1, 1),
            np.clip(horizontal / 220.0, -1, 1),
            np.clip(2.0 * (distance - 20.0) / 380.0 - 1.0, -1, 1),
            np.clip(2.0 * clearance / 160.0 - 1.0, -1, 1),
            2.0 * jaw - 1.0,
            2.0 * lift - 1.0,
            self._servo_normalized(self.current_pose[config.J_SHOULDER], 65, 145),
            self._servo_normalized(self.current_pose[config.J_ELBOW], 35, 165),
            self._servo_normalized(self.current_pose[config.J_GRIP], 90, 180),
            2.0 * self.phase / PHASE_RETURNED - 1.0,
            2.0 * int(self.previous_action) / (len(ReducedTaskAction) - 1) - 1.0,
        ), dtype=np.float32)
        self.last_observation = observation
        return observation.copy()

    def _info(self) -> dict[str, Any]:
        return {
            "event": self.last_event, "phase": self.phase,
            "contact": self.contact, "holding": self.holding,
            "current_pose": tuple(self.current_pose),
            "target_pose": tuple(self.target_pose),
            "distance_mm": self._distance_mm(),
            "floor_clearance_mm": fingertip_floor_clearance_mm(self.current_pose),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        options = options or {}
        anchor = options.get("target_joint")
        if anchor is None:
            anchor = GRASP_POSES[int(self.np_random.integers(0, len(GRASP_POSES)))]
        shoulder = int(np.clip(anchor[0], 65, 145))
        elbow = int(np.clip(anchor[1], 35, 165))
        self.target_pose = command_pose(shoulder, elbow, config.GRIP_OPEN)
        self.current_pose = list(options.get("current_pose", reduced_home()))
        self.phase = int(options.get("phase", PHASE_SEARCH))
        self.previous_action = ReducedTaskAction.WAIT
        self.gripper_open = self.current_pose[config.J_GRIP] < 135
        self.contact = False
        self.holding = False
        self.selection_locked = False
        self.horizontal_error_px = float(options.get(
            "horizontal_error_px", self.np_random.uniform(-65, 65)))
        self.distance_scale = float(
            self.np_random.uniform(4.2, 6.6) if self.domain_randomization else 5.4)
        self.steps = 0
        self.last_event = "RESET"
        return self._observe(), self._info()

    def expert_action(self):
        quality, visible, markers, continuity = self.last_observation[:4] > 0
        if self.phase == PHASE_RETURNED:
            return int(ReducedTaskAction.DONE)
        if not quality or not markers:
            return int(ReducedTaskAction.WAIT)
        if self.phase == PHASE_GRASPED:
            return int(ReducedTaskAction.LIFT if self.contact else ReducedTaskAction.RECOVER)
        if self.phase == PHASE_LIFTED:
            return int(ReducedTaskAction.RETURN_HOME if self.holding else ReducedTaskAction.RECOVER)
        if not visible:
            return int(ReducedTaskAction.SEARCH_NEXT)
        if self.phase in (PHASE_SEARCH, PHASE_APPROACH):
            if np.max(np.abs(self._joint_error(self.current_pose, self.target_pose))) <= 1:
                return int(ReducedTaskAction.CLOSE if continuity else ReducedTaskAction.RECOVER)
            return int(ReducedTaskAction.APPROACH)
        return int(ReducedTaskAction.RECOVER)

    def _move_toward_target(self, max_step):
        target = list(self.current_pose)
        error = self._joint_error(self.current_pose, self.target_pose)
        for joint, delta in zip(ACTIVE_MOTION_JOINTS, error):
            target[joint] += int(np.clip(delta, -max_step, max_step))
        target = command_pose(
            target[config.J_SHOULDER], target[config.J_ELBOW],
            target[config.J_GRIP])
        # Every anchor and the rectangular interpolation envelope is verified
        # separately by MuJoCo/contact tests.  Calling the dense physical
        # collision checker here would make dataset generation thousands of
        # times slower without adding an observation available to the actor.
        self.current_pose = target
        return True

    def step(self, action):
        action = ReducedTaskAction(int(action))
        reward = -0.02
        terminated = False
        self.last_event = action.name
        valid = self.last_observation[0] > 0 and self.last_observation[2] > 0
        visible = self.last_observation[1] > 0

        if action == ReducedTaskAction.WAIT:
            reward += 0.03 if not valid else -0.10
        elif action == ReducedTaskAction.SEARCH_NEXT:
            if self.phase <= PHASE_APPROACH and self.gripper_open:
                before = np.linalg.norm(self._joint_error(self.current_pose, self.target_pose))
                moved = self._move_toward_target(8)
                after = np.linalg.norm(self._joint_error(self.current_pose, self.target_pose))
                reward += 0.08 if moved and after < before else -0.25
                self.phase = PHASE_SEARCH
            else:
                reward -= 0.40
        elif action == ReducedTaskAction.APPROACH:
            if valid and visible and self.phase <= PHASE_APPROACH:
                before = np.linalg.norm(self._joint_error(self.current_pose, self.target_pose))
                moved = self._move_toward_target(5)
                after = np.linalg.norm(self._joint_error(self.current_pose, self.target_pose))
                reward += 0.16 if moved and after < before else -0.35
                self.phase = PHASE_APPROACH
            else:
                reward -= 0.45
        elif action == ReducedTaskAction.CLOSE:
            near = np.max(np.abs(self._joint_error(self.current_pose, self.target_pose))) <= 2
            aligned = abs(self.horizontal_error_px) <= 85
            if near and self.selection_locked and aligned and self.gripper_open:
                self.current_pose = command_pose(
                    self.current_pose[config.J_SHOULDER], self.current_pose[config.J_ELBOW],
                    config.GRIP_CLOSED)
                self.gripper_open = False
                self.contact = True
                self.phase = PHASE_GRASPED
                reward += 1.0
            else:
                self.contact = False
                reward -= 0.9
        elif action == ReducedTaskAction.LIFT:
            if self.phase == PHASE_GRASPED and self.contact:
                self.current_pose = command_pose(80, 90, config.GRIP_CLOSED)
                self.holding = True
                self.phase = PHASE_LIFTED
                reward += 1.4
            else:
                reward -= 0.8
        elif action == ReducedTaskAction.RETURN_HOME:
            if self.phase == PHASE_LIFTED and self.holding:
                self.current_pose = reduced_home(config.GRIP_CLOSED)
                self.phase = PHASE_RETURNED
                reward += 1.6
            else:
                reward -= 0.8
        elif action == ReducedTaskAction.RECOVER:
            self.current_pose = reduced_home(config.GRIP_OPEN)
            self.gripper_open = True
            self.contact = self.holding = self.selection_locked = False
            self.phase = PHASE_SEARCH
            reward -= 0.45
        elif action == ReducedTaskAction.DONE:
            if self.phase == PHASE_RETURNED and self.holding:
                terminated = True
                reward += 3.0
            else:
                reward -= 1.0

        self.steps += 1
        self.previous_action = action
        truncated = self.steps >= self.max_steps and not terminated
        return self._observe(), reward, terminated, truncated, self._info()


__all__ = [
    "GRASP_POSES", "OBSERVATION_NAMES", "PHASE_APPROACH", "PHASE_GRASPED",
    "PHASE_LIFTED", "PHASE_RETURNED", "PHASE_SEARCH", "ReducedFloorPickEnv",
    "ReducedTaskAction",
]
