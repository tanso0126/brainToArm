"""Learned high-level policy for the rigid-wrist reduced arm."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .reduced_dof_task_env import (
    OBSERVATION_NAMES, PHASE_APPROACH, PHASE_GRASPED, PHASE_LIFTED,
    PHASE_RETURNED, ReducedTaskAction,
)


HERE = Path(__file__).resolve().parent
DEFAULT_REDUCED_MODEL = HERE / "models" / "reduced_dof_policy_v1.ts"


class ReducedTaskNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(OBSERVATION_NAMES), 128), nn.SiLU(),
            nn.Linear(128, 96), nn.SiLU(),
            nn.Linear(96, 48), nn.SiLU(),
            nn.Linear(48, len(ReducedTaskAction)),
        )

    def forward(self, observation):
        return self.network(observation.to(dtype=torch.float32))


def _decode_phase(observation):
    return int(round((float(observation[14]) + 1.0) * 0.5 * PHASE_RETURNED))


def shield_action(observation, proposed):
    """Use real-observable state to block impossible learned transitions."""
    values = np.asarray(observation, dtype=np.float32)
    proposed = ReducedTaskAction(int(proposed))
    quality, visible, markers, continuity = values[:4] > 0
    phase = _decode_phase(values)
    if phase == PHASE_RETURNED:
        return ReducedTaskAction.DONE
    if not quality or not markers:
        return ReducedTaskAction.WAIT
    if phase == PHASE_GRASPED:
        return ReducedTaskAction.LIFT if values[9] > -0.75 else ReducedTaskAction.RECOVER
    if phase == PHASE_LIFTED:
        return ReducedTaskAction.RETURN_HOME if values[10] > 0.55 else ReducedTaskAction.RECOVER
    if not visible:
        return ReducedTaskAction.SEARCH_NEXT
    if phase <= PHASE_APPROACH:
        distance_mm = 20.0 + (float(values[7]) + 1.0) * 0.5 * 380.0
        near = distance_mm <= 62.0 and abs(float(values[6]) * 220.0) <= 85.0
        return (ReducedTaskAction.CLOSE if near and continuity
                else ReducedTaskAction.APPROACH)
    return proposed


class ReducedTemporalGuard:
    guarded = frozenset((ReducedTaskAction.CLOSE, ReducedTaskAction.LIFT))

    def __init__(self, votes_required=2):
        self.votes_required = int(votes_required)
        self.reset()

    def reset(self):
        self.pending = None
        self.votes = 0

    def filter(self, action):
        action = ReducedTaskAction(int(action))
        if action == ReducedTaskAction.WAIT and self.pending is not None:
            return action
        if action not in self.guarded:
            self.reset()
            return action
        if action == self.pending:
            self.votes += 1
        else:
            self.pending, self.votes = action, 1
        if self.votes < self.votes_required:
            return ReducedTaskAction.WAIT
        self.reset()
        return action


class ReducedTaskPolicyRunner:
    def __init__(self, model_path=DEFAULT_REDUCED_MODEL):
        self.model_path = Path(model_path)
        self.model = torch.jit.load(str(self.model_path), map_location="cpu").eval()
        self.guard = ReducedTemporalGuard()

    def reset(self):
        self.guard.reset()

    def predict(self, observation, *, apply_shield=True, temporal_guard=True):
        values = np.asarray(observation, dtype=np.float32)
        if values.shape != (len(OBSERVATION_NAMES),):
            raise ValueError(f"축소 정책 입력은 {len(OBSERVATION_NAMES)}개여야 합니다.")
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(values[None]))[0]
            probabilities = torch.softmax(logits, dim=0).numpy()
        action = ReducedTaskAction(int(np.argmax(probabilities)))
        if apply_shield:
            action = shield_action(values, action)
        if temporal_guard:
            action = self.guard.filter(action)
        return action, probabilities


def export_torchscript(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(model.cpu().eval()).save(str(path))
    return path


__all__ = [
    "DEFAULT_REDUCED_MODEL", "ReducedTaskNetwork", "ReducedTaskPolicyRunner",
    "ReducedTemporalGuard", "export_torchscript", "shield_action",
]
