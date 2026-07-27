"""Small deployable RGB alignment actor and a plan-only inference wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch import nn

try:
    from .alignment_env import IMAGE_HEIGHT, IMAGE_WIDTH, MAX_ELBOW_STEP
    from .mujoco_robot import RobotSpec
except ImportError:
    from alignment_env import IMAGE_HEIGHT, IMAGE_WIDTH, MAX_ELBOW_STEP
    from mujoco_robot import RobotSpec


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = HERE / "models" / "alignment_policy_v1.ts"


class AlignmentPolicy(nn.Module):
    """RGB + command-state actor with no simulator-only input."""

    def __init__(self):
        super().__init__()
        self.visual = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(),
        )
        self.trunk = nn.Sequential(
            # 72x128 through four stride-2 convolutions -> 5x8. Keeping this
            # grid (instead of global pooling) preserves target left/right
            # position and is compatible with Apple MPS.
            nn.Linear(64 * 5 * 8 + 7, 192), nn.SiLU(), nn.Dropout(0.08),
            nn.Linear(192, 64), nn.SiLU(),
        )
        self.action_head = nn.Sequential(nn.Linear(64, 1), nn.Tanh())
        self.aligned_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(
        self,
        image: torch.Tensor,
        servo: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        image = image.to(dtype=torch.float32) / 255.0
        features = self.visual(image)
        state = torch.cat((servo.to(dtype=torch.float32),
                           previous_action.to(dtype=torch.float32)), dim=1)
        hidden = self.trunk(torch.cat((features, state), dim=1))
        # Column 0: normalized elbow motion. Column 1: probability that the
        # target is already inside the calibrated alignment tolerance.
        return torch.cat((self.action_head(hidden), self.aligned_head(hidden)), dim=1)


def normalize_servo(pose: Iterable[float], spec: RobotSpec | None = None) -> np.ndarray:
    spec = spec or RobotSpec.from_manifest()
    values = spec.validate_servo_pose(pose)
    span = spec.servo_max_deg - spec.servo_min_deg
    return (2.0 * (values - spec.servo_min_deg) / span - 1.0).astype(np.float32)


class AlignmentPolicyRunner:
    """Pure inference. It returns a suggestion and never commands the Uno."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.model = torch.jit.load(str(self.model_path), map_location="cpu")
        self.model.eval()
        self.spec = RobotSpec.from_manifest()

    @staticmethod
    def prepare_image(frame_rgb: np.ndarray) -> np.ndarray:
        frame = np.asarray(frame_rgb)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be HxWx3 RGB")
        resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT),
                             interpolation=cv2.INTER_AREA)
        return np.transpose(resized.astype(np.uint8), (2, 0, 1))

    def predict(
        self,
        frame_rgb: np.ndarray,
        commanded_pose: Iterable[float],
        previous_action: float = 0.0,
    ) -> tuple[float, float]:
        image = torch.from_numpy(self.prepare_image(frame_rgb)[None])
        servo = torch.from_numpy(normalize_servo(commanded_pose, self.spec)[None])
        previous = torch.tensor([[float(previous_action)]], dtype=torch.float32)
        with torch.inference_mode():
            output = self.model(image, servo, previous)[0]
        action = float(np.clip(output[0].item(), -1.0, 1.0))
        aligned_probability = float(np.clip(output[1].item(), 0.0, 1.0))
        return action, aligned_probability

    def recommend_elbow_delta(
        self,
        frame_rgb: np.ndarray,
        commanded_pose: Iterable[float],
        previous_action: float = 0.0,
        deadband: float = 0.25,
        aligned_threshold: float = 0.65,
        geometry_aligned: bool = False,
    ) -> tuple[int, float, float]:
        action, aligned_probability = self.predict(
            frame_rgb, commanded_pose, previous_action)
        delta = int(round(action * MAX_ELBOW_STEP))
        # The learned probability is never allowed to stop motion by itself.
        # The caller must independently confirm target/jaw geometry from the
        # existing candidate detector before the two signals vote to stop.
        if (geometry_aligned and aligned_probability >= aligned_threshold) \
                or abs(action) < deadband:
            delta = 0
        elif delta == 0:
            delta = 1 if action > 0 else -1
        return delta, action, aligned_probability


def export_torchscript(model: AlignmentPolicy, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    scripted = torch.jit.script(model)
    scripted.save(str(path))
    return path


__all__ = [
    "AlignmentPolicy", "AlignmentPolicyRunner", "DEFAULT_MODEL_PATH", "export_torchscript",
    "normalize_servo",
]
