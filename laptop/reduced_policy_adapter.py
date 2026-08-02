"""Real-sensor adapter for the separately trained reduced-DOF policy.

Importing this module is hardware-free.  It translates existing camera,
ultrasonic, commanded-pose and task-phase values into the exact simulation
training schema; no MuJoCo coordinate or depth buffer can enter inference.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np

import config
from reduced_dof import fingertip_floor_clearance_mm, validate_command_pose

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simul.reduced_dof_task_env import (  # noqa: E402
    OBSERVATION_NAMES, PHASE_APPROACH, PHASE_GRASPED, PHASE_LIFTED,
    PHASE_RETURNED, PHASE_SEARCH, ReducedTaskAction,
)
from simul.reduced_dof_task_policy import (  # noqa: E402
    DEFAULT_REDUCED_MODEL, ReducedTaskPolicyRunner,
)


PHASES = {
    "search": PHASE_SEARCH, "approach": PHASE_APPROACH,
    "grasped": PHASE_GRASPED, "lifted": PHASE_LIFTED,
    "returned": PHASE_RETURNED,
}


@dataclass(frozen=True)
class ReducedPolicyDecision:
    action: ReducedTaskAction
    confidence: float
    probabilities: tuple[float, ...]
    observation: tuple[float, ...]


def _normalize(value, low, high):
    return 2.0 * (float(value) - low) / (high - low) - 1.0


def build_reduced_observation(
    *, pose, target_center=None, gripper_center=None, opening_px=None,
    frame_shape=(720, 1280, 3), quality_valid=True, target_locked=False,
    sonar_distance_mm=None, phase="approach", coherent_lift=False,
    previous_action=ReducedTaskAction.WAIT,
):
    pose = validate_command_pose(pose)
    if phase not in PHASES:
        raise ValueError(f"알 수 없는 축소 정책 단계입니다: {phase}")
    visible = target_center is not None
    markers = gripper_center is not None and opening_px is not None
    vertical = horizontal = 0.0
    if visible and markers:
        vertical = float(target_center[1]) - float(gripper_center[1])
        horizontal = float(target_center[0]) - float(gripper_center[0])
    sonar_valid = sonar_distance_mm is not None and math.isfinite(float(sonar_distance_mm))
    distance = float(sonar_distance_mm) if sonar_valid else 400.0
    diagonal = math.hypot(frame_shape[1], frame_shape[0])
    jaw = (float(np.clip(float(opening_px) / max(diagonal * 0.20, 1.0), 0, 1))
           if markers else 0.0)
    phase_value = PHASES[phase]
    previous_action = ReducedTaskAction(int(previous_action))
    values = np.asarray((
        1 if quality_valid else -1,
        1 if visible else -1,
        1 if markers else -1,
        1 if (visible or target_locked) else -1,
        1 if sonar_valid else -1,
        np.clip(vertical / 420.0, -1, 1),
        np.clip(horizontal / 220.0, -1, 1),
        np.clip(2.0 * (distance - 20.0) / 380.0 - 1.0, -1, 1),
        np.clip(2.0 * fingertip_floor_clearance_mm(pose) / 160.0 - 1.0, -1, 1),
        2.0 * jaw - 1.0,
        1.0 if coherent_lift else -1.0,
        _normalize(pose[config.J_SHOULDER], 65, 145),
        _normalize(pose[config.J_ELBOW], 35, 165),
        _normalize(pose[config.J_GRIP], 90, 180),
        2.0 * phase_value / PHASE_RETURNED - 1.0,
        2.0 * int(previous_action) / (len(ReducedTaskAction) - 1) - 1.0,
    ), dtype=np.float32)
    if values.shape != (len(OBSERVATION_NAMES),):
        raise RuntimeError("축소 정책 입력 규격이 학습 환경과 달라졌습니다.")
    return values


class ReducedPolicyController:
    def __init__(self, model_path=DEFAULT_REDUCED_MODEL):
        self.runner = ReducedTaskPolicyRunner(model_path)
        self.previous_action = ReducedTaskAction.WAIT

    def reset(self):
        self.runner.reset()
        self.previous_action = ReducedTaskAction.WAIT

    def decide(self, **sensor_values):
        observation = build_reduced_observation(
            previous_action=self.previous_action, **sensor_values)
        action, probabilities = self.runner.predict(observation)
        self.previous_action = action
        return ReducedPolicyDecision(
            action, float(probabilities[int(action)]),
            tuple(float(value) for value in probabilities),
            tuple(float(value) for value in observation),
        )


__all__ = [
    "DEFAULT_REDUCED_MODEL", "ReducedPolicyController",
    "ReducedPolicyDecision", "build_reduced_observation",
]
