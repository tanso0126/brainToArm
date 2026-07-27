"""TorchScript complete-task macro policy and hardware-free runner."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from .full_task_env import OBSERVATION_NAMES, TaskAction
except ImportError:
    from full_task_env import OBSERVATION_NAMES, TaskAction

try:
    from laptop import config
    from laptop.floor_motion import floor_shoulder
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laptop"))
    import config
    from floor_motion import floor_shoulder


HERE = Path(__file__).resolve().parent
DEFAULT_FULL_TASK_MODEL = HERE / "models" / "full_task_policy_v1.ts"


class FullTaskNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(OBSERVATION_NAMES), 192), nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(192, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, len(TaskAction)),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation.to(dtype=torch.float32))


class FullTaskPolicyRunner:
    """Pure inference; it selects a macro but never opens serial or camera."""

    def __init__(self, model_path=DEFAULT_FULL_TASK_MODEL):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.model = torch.jit.load(str(self.model_path), map_location="cpu").eval()
        self.temporal_guard = TemporalTaskGuard()

    def reset(self):
        self.temporal_guard.reset()

    def predict(
        self, observation, *, apply_shield=True, temporal_guard=False,
    ) -> tuple[TaskAction, np.ndarray]:
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (len(OBSERVATION_NAMES),):
            raise ValueError(
                f"full-task observation must have {len(OBSERVATION_NAMES)} values")
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(values[None]))[0]
            probabilities = torch.softmax(logits, dim=0).numpy()
        raw_action = TaskAction(int(np.argmax(probabilities)))
        action = shield_action(values, raw_action) if apply_shield else raw_action
        if temporal_guard:
            action = self.temporal_guard.filter(action)
        return action, probabilities


class TemporalTaskGuard:
    """Demand two fresh observations before irreversible macro transitions."""

    guarded_actions = frozenset((
        TaskAction.DESCEND, TaskAction.CLOSE, TaskAction.LIFT,
    ))

    def __init__(self, votes_required=2):
        self.votes_required = int(votes_required)
        if self.votes_required < 1:
            raise ValueError("votes_required must be positive")
        self.reset()

    def reset(self):
        self.pending = None
        self.votes = 0

    def filter(self, proposed):
        action = TaskAction(int(proposed))
        # A dropped/overexposed frame must pause the vote, not erase two valid
        # confirmations already collected for the same stable pose.
        if action == TaskAction.WAIT and self.pending is not None:
            return action
        if action not in self.guarded_actions:
            self.reset()
            return action
        if action == self.pending:
            self.votes += 1
        else:
            self.pending = action
            self.votes = 1
        if self.votes < self.votes_required:
            return TaskAction.WAIT
        self.reset()
        return action


def _decode_servo(observation: np.ndarray) -> np.ndarray:
    normalized = observation[8:14]
    minimum = np.asarray(config.SERVO_MIN, dtype=np.float32)
    maximum = np.asarray(config.SERVO_MAX, dtype=np.float32)
    return minimum + (normalized + 1.0) * 0.5 * (maximum - minimum)


def shield_action(observation, proposed: TaskAction | int) -> TaskAction:
    """Restrict the learned macro to the next camera-verified safe action.

    This function uses no simulator state. Every input is derived from the real
    camera pipeline or the host's already-commanded servo vector.
    """

    values = np.asarray(observation, dtype=np.float32)
    proposed = TaskAction(int(proposed))
    quality = values[0] > 0
    target = values[1] > 0 and values[3] > 0
    markers = values[2] > 0
    depth_error = float(values[4] * 420.0)
    lateral_error = float(values[5] * 220.0)
    jaw_opening = float((values[6] + 1.0) * 0.5)
    coherent_lift = values[7] > 0.55
    servo = _decode_servo(values)
    elbow = int(round(float(servo[config.J_ELBOW])))
    elbow = int(np.clip(elbow, *config.FLOOR_ELBOW_RANGE))
    hover_shoulder = floor_shoulder(elbow, "hover")
    grasp_shoulder = floor_shoulder(elbow, "grasp")
    at_grasp = abs(servo[config.J_SHOULDER] - grasp_shoulder) < \
        abs(servo[config.J_SHOULDER] - hover_shoulder)
    gripper_command_open = servo[config.J_GRIP] < (
        config.GRIP_OPEN + config.GRIP_CLOSED) / 2.0

    if coherent_lift:
        return TaskAction.WAIT
    if not quality or not markers:
        return TaskAction.WAIT
    if at_grasp:
        if gripper_command_open:
            return TaskAction.CLOSE if target else TaskAction.RECOVER
        if jaw_opening > 0.12:
            return TaskAction.LIFT
        return TaskAction.RECOVER
    if not target:
        return TaskAction.SEARCH_NEXT
    if not at_grasp:
        # Use a stricter entry threshold than the physical controller's final
        # tolerance so ±8 px perception noise cannot trigger premature descent.
        aligned = (abs(depth_error) <= 0.65 * config.FLOOR_ALIGN_TOL_PX
                   and abs(lateral_error)
                   <= config.FLOOR_ALIGN_X_CENTERLINE_TOL_PX)
        if aligned:
            return TaskAction.DESCEND
        correction = depth_error / config.FLOOR_ALIGN_DY_PER_ELBOW
        return (TaskAction.ALIGN_ELBOW_UP if correction > 0
                else TaskAction.ALIGN_ELBOW_DOWN)
    return proposed


def export_torchscript(model: FullTaskNetwork, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model.to("cpu").eval())
    scripted.save(str(path))
    return path


__all__ = [
    "DEFAULT_FULL_TASK_MODEL", "FullTaskNetwork", "FullTaskPolicyRunner",
    "TemporalTaskGuard",
    "export_torchscript", "shield_action",
]
