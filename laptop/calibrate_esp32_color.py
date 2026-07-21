"""Build a fixed OV2640 spatial color map from a neutral gray/white scene."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from capture_esp32_camera import (
    DEFAULT_COLOR_CALIBRATION,
    apply_spatial_gain,
    capture,
    correct_neutral_background,
    find_port,
)


def build_gain_grid(image, columns=20, rows=15, blur_sigma=35.0):
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    blurred = np.dstack([
        cv2.GaussianBlur(pixels[:, :, channel], (0, 0), blur_sigma)
        for channel in range(3)
    ])
    neutral = blurred.mean(axis=2, keepdims=True)
    full_gain = np.clip(neutral / np.maximum(blurred, 1.0), 0.82, 1.18)
    return cv2.resize(full_gain, (columns, rows),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def calibrate(port, output, preview):
    with TemporaryDirectory() as temporary:
        source = Path(temporary) / "neutral-source.jpg"
        capture(port, source, color_correct=False, calibration_path=None)
        image = Image.open(source).convert("RGB")
        gain_grid = build_gain_grid(image)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, gain_grid=gain_grid,
                        width=image.width, height=image.height)

    corrected = apply_spatial_gain(image, gain_grid)
    corrected, gains, neutral_count = correct_neutral_background(corrected)
    preview.parent.mkdir(parents=True, exist_ok=True)
    corrected.save(preview, quality=95)
    print(f"[calibration] grid={gain_grid.shape[1]}x{gain_grid.shape[0]} "
          f"range={gain_grid.min():.3f}..{gain_grid.max():.3f} -> {output}")
    print(f"[calibration] preview global gain="
          f"({gains[0]:.3f},{gains[1]:.3f},{gains[2]:.3f}) "
          f"neutral_pixels={neutral_count} -> {preview}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_COLOR_CALIBRATION)
    parser.add_argument("--preview", type=Path,
                        default=Path("data/vision/esp32_color_calibration_preview.jpg"))
    args = parser.parse_args()
    calibrate(find_port(args.port), args.output, args.preview)


if __name__ == "__main__":
    main()
