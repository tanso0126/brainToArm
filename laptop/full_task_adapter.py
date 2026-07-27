"""Real-camera adapter for the simulation-trained full floor-pick policy.

This module is deliberately hardware-free: importing or calling it never opens
the webcam or Uno.  It converts the existing wrist perception objects and the
host's commanded servo pose into the exact 15-value training observation, then
returns a guarded macro recommendation.  The caller remains responsible for
the execution gate and fresh perception between macro actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np

import config

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simul.full_task_env import OBSERVATION_NAMES, TaskAction  # noqa: E402
from simul.full_task_policy import FullTaskPolicyRunner  # noqa: E402


@dataclass(frozen=True)
class MacroDecision:
    action: TaskAction
    confidence: float
    probabilities: tuple[float, ...]
    observation: tuple[float, ...]
    next_pose: tuple[int, ...] | None
    reason: str


def _normalize_servo(pose):
    values = np.asarray(pose, dtype=np.float32)
    if values.shape != (config.N_JOINTS,) or not np.isfinite(values).all():
        raise ValueError("commanded_pose must contain six finite servo values")
    minimum = np.asarray(config.SERVO_MIN, dtype=np.float32)
    maximum = np.asarray(config.SERVO_MAX, dtype=np.float32)
    if np.any(values < minimum) or np.any(values > maximum):
        raise ValueError("commanded_pose is outside configured servo limits")
    return 2.0 * (values - minimum) / (maximum - minimum) - 1.0


def _jaw_opening_normalized(gripper, frame_shape):
    if gripper is None:
        return 0.0
    height, width = frame_shape[:2]
    diagonal = math.hypot(width, height)
    ratio = float(gripper.opening_px) / diagonal
    closed = float(config.WRIST_GRIPPER_CLOSED_PROFILE["opening_ratio"])
    opened = float(config.WRIST_GRIPPER_OPEN_PROFILE["opening_ratio"])
    return float(np.clip((ratio - closed) / (opened - closed), 0.0, 1.0))


def build_policy_observation(
    scene,
    wrist_observation,
    commanded_pose,
    *,
    target=None,
    previous_target_center=None,
    coherent_lift_motion=False,
    target_locked=False,
    previous_action=TaskAction.WAIT,
):
    """Build only from quantities available on the real Mac/camera pipeline."""

    quality = bool(
        wrist_observation is not None
        and getattr(wrist_observation, "quality", None) is not None
        and wrist_observation.quality.valid
    )
    gripper = getattr(scene, "gripper", None)
    markers = gripper is not None
    target_visible = target is not None
    continuity = bool(target_visible or target_locked)
    if target_visible and previous_target_center is not None:
        diagonal = math.hypot(scene.frame_shape[1], scene.frame_shape[0])
        distance = math.hypot(
            target.center[0] - previous_target_center[0],
            target.center[1] - previous_target_center[1],
        )
        continuity = distance <= config.FLOOR_REJECT_RADIUS_RATIO * diagonal

    depth_error = 0.0
    centerline_error = 0.0
    if target_visible and markers:
        depth_error = gripper.center[1] - target.center[1]
        centerline_error = target.center[0] - gripper.center[0]
    jaw = _jaw_opening_normalized(gripper, scene.frame_shape)
    lift = float(np.clip(float(coherent_lift_motion), 0.0, 1.0))
    previous_action = TaskAction(int(previous_action))
    values = np.concatenate((
        np.asarray((
            1.0 if quality else -1.0,
            1.0 if target_visible else -1.0,
            1.0 if markers else -1.0,
            1.0 if continuity else -1.0,
            np.clip(depth_error / 420.0, -1.0, 1.0),
            np.clip(centerline_error / 220.0, -1.0, 1.0),
            2.0 * jaw - 1.0,
            2.0 * lift - 1.0,
        ), dtype=np.float32),
        _normalize_servo(commanded_pose),
        np.asarray((
            2.0 * int(previous_action) / (len(TaskAction) - 1) - 1.0,
        ), dtype=np.float32),
    )).astype(np.float32)
    if values.shape != (len(OBSERVATION_NAMES),):
        raise RuntimeError("policy observation schema drift")
    return values


def macro_next_pose(action, observation, commanded_pose):
    """Translate a policy macro into a bounded pose; SEARCH stays planner-owned."""

    from floor_motion import floor_pose

    action = TaskAction(int(action))
    pose = [int(round(value)) for value in commanded_pose]
    elbow = int(np.clip(
        pose[config.J_ELBOW], *config.FLOOR_ELBOW_RANGE))
    if action == TaskAction.ALIGN_ELBOW_DOWN:
        elbow = max(config.FLOOR_ELBOW_RANGE[0],
                    elbow - config.FLOOR_ALIGN_MAX_ELBOW_STEP)
        return tuple(floor_pose(elbow, "hover", gripper=config.GRIP_OPEN))
    if action == TaskAction.ALIGN_ELBOW_UP:
        elbow = min(config.FLOOR_ELBOW_RANGE[1],
                    elbow + config.FLOOR_ALIGN_MAX_ELBOW_STEP)
        return tuple(floor_pose(elbow, "hover", gripper=config.GRIP_OPEN))
    if action == TaskAction.DESCEND:
        return tuple(floor_pose(elbow, "grasp", gripper=config.GRIP_OPEN))
    if action == TaskAction.CLOSE:
        return tuple(floor_pose(elbow, "grasp", gripper=config.GRIP_CLOSED))
    if action == TaskAction.LIFT:
        return tuple(floor_pose(elbow, "hover", gripper=config.GRIP_CLOSED))
    if action == TaskAction.RECOVER:
        return tuple(floor_pose(elbow, "hover", gripper=config.GRIP_OPEN))
    return None


class FullTaskShadowController:
    """Stateful two-frame guarded policy runner with no hardware side effects."""

    def __init__(self, model_path=None):
        self.runner = (FullTaskPolicyRunner()
                       if model_path is None else FullTaskPolicyRunner(model_path))
        self.previous_action = TaskAction.WAIT
        self.previous_target_center = None

    def reset(self):
        self.runner.reset()
        self.previous_action = TaskAction.WAIT
        self.previous_target_center = None

    def decide(
        self, scene, wrist_observation, commanded_pose, *, target=None,
        coherent_lift_motion=False, target_locked=False,
    ):
        observation = build_policy_observation(
            scene, wrist_observation, commanded_pose,
            target=target,
            previous_target_center=self.previous_target_center,
            coherent_lift_motion=coherent_lift_motion,
            target_locked=target_locked,
            previous_action=self.previous_action,
        )
        action, probabilities = self.runner.predict(
            observation, apply_shield=True, temporal_guard=True)
        if target is not None:
            self.previous_target_center = tuple(float(v) for v in target.center)
        self.previous_action = action
        next_pose = macro_next_pose(action, observation, commanded_pose)
        reasons = {
            TaskAction.WAIT: "wait for another verified camera frame",
            TaskAction.SEARCH_NEXT: "target absent; request next safe search pose",
            TaskAction.ALIGN_ELBOW_DOWN: "move backward on the calibrated floor curve",
            TaskAction.ALIGN_ELBOW_UP: "move forward on the calibrated floor curve",
            TaskAction.DESCEND: "two-frame alignment verified; descend open",
            TaskAction.CLOSE: "two-frame grasp pose verified; close",
            TaskAction.LIFT: "two-frame visual contact verified; lift closed",
            TaskAction.RECOVER: "verification failed; open and return to hover",
        }
        return MacroDecision(
            action=action,
            confidence=float(probabilities[int(action)]),
            probabilities=tuple(float(value) for value in probabilities),
            observation=tuple(float(value) for value in observation),
            next_pose=next_pose,
            reason=reasons[action],
        )


__all__ = [
    "FullTaskShadowController", "MacroDecision", "build_policy_observation",
    "macro_next_pose",
]
