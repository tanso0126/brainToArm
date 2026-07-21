"""Capture and validate one OV2640 frame over the ESP32 CP2102 USB link."""

import argparse
import glob
from io import BytesIO
import time
from pathlib import Path

import numpy as np
import serial
from PIL import Image

DEFAULT_COLOR_CALIBRATION = Path("data/calibration/esp32_color_calibration.npz")


def apply_spatial_gain(image, gain_grid):
    """Apply a smooth sensor-coordinate RGB gain grid to an image."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = pixels.shape[:2]
    if gain_grid.ndim != 3 or gain_grid.shape[2] != 3:
        raise ValueError("color calibration gain_grid must have shape (rows, cols, 3)")
    channels = []
    for channel in range(3):
        gain_image = Image.fromarray(gain_grid[:, :, channel].astype(np.float32), mode="F")
        gain_image = gain_image.resize((width, height), Image.Resampling.BICUBIC)
        channels.append(np.asarray(gain_image, dtype=np.float32))
    full_gain = np.dstack(channels)
    return Image.fromarray(np.clip(pixels * full_gain, 0, 255).astype(np.uint8), mode="RGB")


def load_spatial_gain(path):
    with np.load(path, allow_pickle=False) as calibration:
        return calibration["gain_grid"].astype(np.float32)


def correct_neutral_background(image, gain_grid=None):
    """Remove small per-frame RGB drift using sufficiently gray background pixels.

    The gripper/object colors stay out of the estimate because pixels with more
    than 35 levels of channel spread are excluded. Gains are deliberately
    bounded to +/-15% so a scene without enough gray cannot be over-corrected.
    """
    if gain_grid is not None:
        image = apply_spatial_gain(image, gain_grid)
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    channel_max = pixels.max(axis=2)
    channel_min = pixels.min(axis=2)
    luminance = pixels.mean(axis=2)
    neutral = ((channel_max - channel_min) < 35) & (luminance > 35) & (luminance < 220)
    neutral_count = int(neutral.sum())
    if neutral_count < pixels.shape[0] * pixels.shape[1] * 0.10:
        return image.convert("RGB"), (1.0, 1.0, 1.0), neutral_count

    medians = np.median(pixels[neutral], axis=0)
    target = float(medians.mean())
    gains = np.clip(target / np.maximum(medians, 1.0), 0.85, 1.15)
    corrected = np.clip(pixels * gains, 0, 255).astype(np.uint8)
    return Image.fromarray(corrected, mode="RGB"), tuple(float(v) for v in gains), neutral_count


def find_port(requested):
    if requested != "auto":
        return requested
    candidates = sorted(glob.glob("/dev/cu.usbserial-*")
                        + glob.glob("/dev/cu.usbmodem*"))
    if len(candidates) != 1:
        raise RuntimeError(
            "expected one ESP32 serial port; use --port: " + ", ".join(candidates))
    return candidates[0]


def read_line_until(connection, prefixes, timeout):
    deadline = time.monotonic() + timeout
    observed = []
    while time.monotonic() < deadline:
        # Bound a corrupt/no-newline device burst so a wiring fault cannot make
        # the diagnostic print megabytes of binary-looking serial noise.
        raw = connection.read_until(b"\n", 512)
        if not raw:
            continue
        if b"\x00" in raw or (len(raw) == 512 and not raw.endswith(b"\n")):
            continue
        line = raw.decode(errors="replace").strip()
        matched = any(line.startswith(prefix) for prefix in prefixes)
        if line and matched:
            observed.append(line)
            print(f"[esp32] {line}")
        if matched:
            return line
    raise TimeoutError("ESP32 response timeout; observed: " + " | ".join(observed[-8:]))


def read_exact(connection, size, timeout):
    deadline = time.monotonic() + timeout
    chunks = bytearray()
    while len(chunks) < size and time.monotonic() < deadline:
        chunk = connection.read(size - len(chunks))
        if chunk:
            chunks.extend(chunk)
    if len(chunks) != size:
        raise TimeoutError(f"frame truncated: expected {size}, received {len(chunks)}")
    return bytes(chunks)


def decode_frame(width, height, pixel_format, frame):
    if pixel_format == "4":
        if not (frame.startswith(b"\xff\xd8") and frame.endswith(b"\xff\xd9")):
            raise ValueError("frame does not have valid JPEG start/end markers")
        image = Image.open(BytesIO(frame)).convert("RGB")
        image.load()
        return image
    if pixel_format == "0":
        expected = int(width) * int(height) * 2
        if len(frame) != expected:
            raise ValueError(f"RGB565 length mismatch: expected {expected}, got {len(frame)}")
        rgb = bytearray(int(width) * int(height) * 3)
        for source in range(0, len(frame), 2):
            value = (frame[source] << 8) | frame[source + 1]
            target = (source // 2) * 3
            rgb[target] = ((value >> 11) & 0x1F) * 255 // 31
            rgb[target + 1] = ((value >> 5) & 0x3F) * 255 // 63
            rgb[target + 2] = (value & 0x1F) * 255 // 31
        return Image.frombytes("RGB", (int(width), int(height)), bytes(rgb))
    raise ValueError(f"unsupported camera pixel format: {pixel_format}")


def validate_frame_quality(image, neutral_count=None):
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    pixel_count = pixels.shape[0] * pixels.shape[1]
    white_fraction = float(np.all(pixels > 248, axis=2).mean())
    black_fraction = float(np.all(pixels < 7, axis=2).mean())
    if white_fraction > 0.80 or black_fraction > 0.80:
        raise ValueError(
            f"implausibly clipped frame: white={white_fraction:.1%}, "
            f"black={black_fraction:.1%}")
    if neutral_count is not None and neutral_count < pixel_count * 0.10:
        raise ValueError(
            f"neutral calibrated background missing: {neutral_count}/{pixel_count} pixels")

    values = pixels.astype(np.float32)
    horizontal = np.mean(np.abs(values[:, 1:] - values[:, :-1]), axis=2)
    vertical = np.mean(np.abs(values[1:] - values[:-1]), axis=2)
    block_x = np.array([x for x in range(horizontal.shape[1]) if (x + 1) % 8 == 0])
    inner_x = np.array([x for x in range(horizontal.shape[1]) if (x + 1) % 8 != 0])
    block_y = np.array([y for y in range(vertical.shape[0]) if (y + 1) % 8 == 0])
    inner_y = np.array([y for y in range(vertical.shape[0]) if (y + 1) % 8 != 0])
    boundary = (horizontal[:, block_x].mean() + vertical[block_y, :].mean()) / 2
    interior = (horizontal[:, inner_x].mean() + vertical[inner_y, :].mean()) / 2
    block_ratio = float(boundary / max(interior, 0.1))
    if block_ratio > 1.75:
        raise ValueError(f"JPEG block discontinuity ratio too high: {block_ratio:.2f}")


def capture(port, output, color_correct=True, calibration_path=DEFAULT_COLOR_CALIBRATION):
    gain_grid = None
    if color_correct and calibration_path is not None and calibration_path.exists():
        gain_grid = load_spatial_gain(calibration_path)
    gains = (1.0, 1.0, 1.0)
    neutral_count = None

    with serial.Serial(port, 115200, timeout=0.15) as connection:
        # Do not toggle EN/DTR: camera XCLK shares boot strap GPIO0 on this
        # wiring. Query the already running sketch instead of forcing a reset.
        ready = None
        for _ in range(3):
            connection.reset_input_buffer()
            connection.write(b"STATUS\n")
            try:
                ready = read_line_until(
                    connection, ("CAMERA_READY", "CAMERA_ERROR"), timeout=3.0)
                break
            except TimeoutError:
                continue
        if ready is None:
            raise TimeoutError("ESP32 did not answer STATUS after three attempts")
        if ready.startswith("CAMERA_ERROR"):
            raise RuntimeError(f"OV2640 initialization failed: {ready}")

        image = None
        last_error = None
        for attempt in range(1, 9):
            connection.reset_input_buffer()
            connection.write(b"CAPTURE\n")
            header = read_line_until(
                connection, ("FRAME ", "CAPTURE_ERROR"), timeout=12.0)
            if header == "CAPTURE_ERROR":
                last_error = RuntimeError("OV2640 initialized but frame capture failed")
                continue
            _, width, height, byte_count, pixel_format = header.split()
            frame = read_exact(connection, int(byte_count), timeout=8.0)
            try:
                candidate = decode_frame(width, height, pixel_format, frame)
                candidate_neutral_count = None
                candidate_gains = (1.0, 1.0, 1.0)
                if color_correct:
                    candidate, candidate_gains, candidate_neutral_count = \
                        correct_neutral_background(candidate, gain_grid)
                    if gain_grid is not None and (
                            min(candidate_gains) < 0.90 or max(candidate_gains) > 1.10):
                        raise ValueError(
                            "sensor white balance not settled: gain="
                            f"({candidate_gains[0]:.3f},"
                            f"{candidate_gains[1]:.3f},"
                            f"{candidate_gains[2]:.3f})")
                validate_frame_quality(
                    candidate,
                    candidate_neutral_count if gain_grid is not None else None)
                image = candidate
                gains = candidate_gains
                neutral_count = candidate_neutral_count
                break
            except (OSError, ValueError) as error:
                last_error = error
                print(f"[camera] dropping corrupt frame {attempt}/8: {error}")
        if image is None:
            raise RuntimeError(f"eight consecutive camera frames failed: {last_error}")

    output.parent.mkdir(parents=True, exist_ok=True)
    if color_correct:
        print("[camera] neutral correction "
              f"gain=({gains[0]:.3f},{gains[1]:.3f},{gains[2]:.3f}) "
              f"pixels={neutral_count} spatial={'yes' if gain_grid is not None else 'no'}")
    save_options = {"quality": 95} if output.suffix.lower() in (".jpg", ".jpeg") else {}
    image.save(output, **save_options)
    print(f"[camera] valid frame {width}x{height}, format={pixel_format}, "
          f"{len(frame)} bytes -> {output}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="auto")
    parser.add_argument("--output", type=Path,
                        default=Path("data/vision/esp32_camera_test.png"))
    parser.add_argument("--no-color-correct", action="store_true",
                        help="save decoded sensor colors without gray-background correction")
    parser.add_argument("--calibration", type=Path,
                        default=DEFAULT_COLOR_CALIBRATION,
                        help="sensor-coordinate color gain map (.npz)")
    args = parser.parse_args()
    capture(find_port(args.port), args.output,
            color_correct=not args.no_color_correct,
            calibration_path=args.calibration)


if __name__ == "__main__":
    main()
