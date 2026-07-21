"""Capture and validate one OV2640 JPEG over the ESP32 CP2102 USB link."""

import argparse
import glob
import time
from pathlib import Path

import serial


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
        if line:
            observed.append(line)
            print(f"[esp32] {line}")
        if any(line.startswith(prefix) for prefix in prefixes):
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
        raise TimeoutError(f"JPEG truncated: expected {size}, received {len(chunks)}")
    return bytes(chunks)


def capture(port, output):
    with serial.Serial(port, 115200, timeout=0.15) as connection:
        # Do not toggle EN/DTR: camera XCLK shares boot strap GPIO0 on this
        # wiring. Query the already running sketch instead of forcing a reset.
        connection.reset_input_buffer()
        connection.write(b"?\n")
        ready = read_line_until(
            connection, ("CAMERA_READY", "CAMERA_ERROR"), timeout=8.0)
        if ready.startswith("CAMERA_ERROR"):
            raise RuntimeError(f"OV2640 initialization failed: {ready}")

        connection.reset_input_buffer()
        connection.write(b"C\n")
        header = read_line_until(
            connection, ("FRAME ", "CAPTURE_ERROR"), timeout=4.0)
        if header == "CAPTURE_ERROR":
            raise RuntimeError("OV2640 initialized but frame capture failed")
        _, width, height, byte_count, pixel_format = header.split()
        jpeg = read_exact(connection, int(byte_count), timeout=8.0)

    if not (jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")):
        raise ValueError("frame does not have valid JPEG start/end markers")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(jpeg)
    print(f"[camera] valid JPEG {width}x{height}, format={pixel_format}, "
          f"{len(jpeg)} bytes -> {output}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="auto")
    parser.add_argument("--output", type=Path,
                        default=Path("data/vision/esp32_camera_test.jpg"))
    args = parser.parse_args()
    capture(find_port(args.port), args.output)


if __name__ == "__main__":
    main()
