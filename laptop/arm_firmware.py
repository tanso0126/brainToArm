"""Compile and upload the robot-arm Uno firmware from the laptop.

The upload is explicit because flashing resets the Uno and therefore commands
the physical servos to the pose in ``home_pose.h``. Normal arm connections use
the firmware protocol to verify that the already-flashed pose matches first.
"""

import os
from pathlib import Path
import shutil
import subprocess

import config
from arm_serial import _serial_candidates


FQBN = "arduino:avr:uno"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKETCH_DIR = PROJECT_ROOT / "firmware" / "arm_controller"
MAC_APP_CLI = Path(
    "/Applications/Arduino IDE.app/Contents/Resources/app/lib/backend/"
    "resources/arduino-cli")


def find_arduino_cli():
    """Find an explicit override, PATH install, or Arduino IDE bundled CLI."""
    override = os.environ.get("ARDUINO_CLI")
    candidates = [Path(override).expanduser()] if override else []
    path_cli = shutil.which("arduino-cli")
    if path_cli:
        candidates.append(Path(path_cli))
    candidates.append(MAC_APP_CLI)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(
        "arduino-cli was not found. Install Arduino IDE or set ARDUINO_CLI "
        "to the executable path.")


def resolve_upload_port(port=None):
    """Resolve exactly one Uno serial port without ever selecting the ESP32."""
    requested = port or config.ARM_PORT
    if requested and requested != "auto":
        return requested
    candidates = _serial_candidates()
    if not candidates:
        raise RuntimeError("connect the Uno: no non-excluded serial board was found")
    if len(candidates) != 1:
        raise RuntimeError(
            "multiple possible Uno ports found; set ARM_PORT explicitly: "
            + ", ".join(candidates))
    return candidates[0]


def upload_arm_firmware(port=None):
    """Compile and upload the current sketch, returning the selected port."""
    cli = find_arduino_cli()
    selected_port = resolve_upload_port(port)
    print(f"[firmware] compile {SKETCH_DIR} ({FQBN})")
    subprocess.run(
        [str(cli), "compile", "--fqbn", FQBN, str(SKETCH_DIR)],
        check=True)
    print(f"[firmware] upload {selected_port}")
    subprocess.run(
        [str(cli), "upload", "--port", selected_port,
         "--fqbn", FQBN, str(SKETCH_DIR)],
        check=True)
    print(f"[firmware] uploaded HOME_POSE={config.HOME_POSE}")
    return selected_port
