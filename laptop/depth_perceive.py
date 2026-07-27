"""General object 3D localization from one moving RGB camera.

Unlike a floor-plane homography (which assumes the object sits on one known
plane and breaks on a shelf), this estimates per-pixel metric depth with a
monocular model, so an object at any height/surface is localized. Monocular
metric models have an unreliable absolute scale, so the scale is recalibrated
every frame against a rigid reference whose true camera distance is known: the
blue/red finger-tape markers are mounted on the wrist a fixed distance from the
lens, so the model's depth at the marker pixels fixes the metric scale (a
one-shot metric-depth alignment, per Metric-Depth-Alignment for RGB grasping).

Output is the object position in the CAMERA frame. A separate hand/eye step
places it in the arm base frame for IK. Perception here is FK-chain independent.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math

import numpy as np


# Reference camera->marker distance (m), from the mounted-camera CAD geometry:
# camera at ~(0.015,0,0.075) in the wrist frame, tape markers at ~(0.074,+/-jaw,
# 0.007). At the open jaw this is ~0.10 m. Used only to fix the depth scale, so
# a coarse value already removes most of the model's absolute-scale error.
REFERENCE_MARKER_DISTANCE_M = 0.10
WRIST_FOVY_DEG = 73.0  # PW315 vertical FoV (nominal; refine with calibration)


@dataclass
class DepthObject:
    center_px: Tuple[float, float]
    xyz_cam_m: Tuple[float, float, float]   # object in camera frame (metres)
    raw_depth_m: float
    scale: float                            # applied metric scale factor
    reference: str


class DepthPerceiver:
    """Monocular metric depth + marker-referenced scale -> object 3D (cam frame)."""

    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
                 device="cpu", frame_size=(1280, 720), fovy_deg=WRIST_FOVY_DEG):
        self.model_id = model_id
        self.device = device
        self.width, self.height = frame_size
        self.fovy = math.radians(fovy_deg)
        # Pinhole intrinsics from the vertical FoV (square pixels, centre principal).
        self.fy = (self.height / 2.0) / math.tan(self.fovy / 2.0)
        self.fx = self.fy
        self.cx, self.cy = self.width / 2.0, self.height / 2.0
        self._proc = None
        self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            self._proc = AutoImageProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
            self._model.to(self.device).eval()

    def depth_map(self, frame_bgr) -> np.ndarray:
        """Return raw model metric depth (metres, unscaled) at full frame size."""
        import cv2
        import torch
        from PIL import Image
        self._load()
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inp = self._proc(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            pred = self._model(**inp).predicted_depth
        depth = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(self.height, self.width),
            mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        return depth

    def _scale_from_markers(self, depth, gripper) -> Optional[float]:
        """Metric scale = true marker distance / model depth at the markers."""
        if gripper is None:
            return None
        samples = []
        for blob in (gripper.blue, gripper.red):
            x, y = (int(round(v)) for v in blob.center)
            if 0 <= y < self.height and 0 <= x < self.width:
                samples.append(float(depth[y, x]))
        samples = [s for s in samples if s > 1e-6]
        if not samples:
            return None
        return REFERENCE_MARKER_DISTANCE_M / float(np.median(samples))

    def _backproject(self, u, v, depth_m):
        """Pixel + metric depth -> 3D in camera frame (x right, y down, z fwd)."""
        z = float(depth_m)
        x = (u - self.cx) / self.fx * z
        y = (v - self.cy) / self.fy * z
        return (x, y, z)

    def locate(self, frame_bgr, target_center, gripper, patch=6) -> DepthObject:
        """Localize the object at ``target_center`` in the camera frame."""
        depth = self.depth_map(frame_bgr)
        u, v = (float(c) for c in target_center)
        x0, x1 = max(0, int(u - patch)), min(self.width, int(u + patch + 1))
        y0, y1 = max(0, int(v - patch)), min(self.height, int(v + patch + 1))
        raw = float(np.median(depth[y0:y1, x0:x1]))
        scale = self._scale_from_markers(depth, gripper)
        reference = "markers"
        if scale is None or not math.isfinite(scale) or scale <= 0:
            scale = 1.0
            reference = "none(uncalibrated)"
        metric = raw * scale
        return DepthObject(
            center_px=(u, v), xyz_cam_m=self._backproject(u, v, metric),
            raw_depth_m=raw, scale=float(scale), reference=reference)
