"""Windows COM-port adapter for the proven real-time arm controller.

The macOS runtime owns the Uno through a Unix-domain socket. Windows does not
need that extra process: this adapter owns the COM port in the same process as
the autonomous controller and exposes the same small ``request`` contract.
"""

from pathlib import Path
import os
import time

from serial.tools import list_ports

import config
from arm_safety import PhysicalArmSafety
import arm_serial


ROOT = Path(__file__).resolve().parents[1]
WRIST_RAW_FRAME = ROOT / "data" / "vision" / "wrist_camera_latest_raw.jpg"
REALTIME_MAX_TARGET_DELTA_DEG = 8


def port_description(port):
    fields = (
        getattr(port, "device", ""),
        getattr(port, "description", ""),
        getattr(port, "manufacturer", ""),
        getattr(port, "hwid", ""),
    )
    return " | ".join(str(value or "") for value in fields)


def _port_score(port):
    text = port_description(port).lower()
    vid = getattr(port, "vid", None)
    if "cp210" in text or "esp32" in text or vid == 0x10C4:
        return -100
    score = 0
    if "arduino uno" in text:
        score += 100
    elif "arduino" in text:
        score += 70
    if "ch340" in text or "usb-serial" in text or vid == 0x1A86:
        score += 60
    if vid == 0x2341:
        score += 80
    if str(getattr(port, "device", "")).upper().startswith("COM"):
        score += 5
    return score


def find_arm_port(preferred=None, ports=None):
    """Select the Uno/CH340 while explicitly excluding the ESP32 CP210x."""
    if preferred and str(preferred).lower() != "auto":
        return str(preferred).upper()
    ports = list(list_ports.comports() if ports is None else ports)
    ranked = sorted(
        ((_port_score(port), port) for port in ports),
        key=lambda pair: pair[0],
        reverse=True,
    )
    usable = [(score, port) for score, port in ranked if score > 0]
    if not usable:
        listing = "\n  ".join(port_description(port) for port in ports)
        raise RuntimeError(
            "Arduino Uno/CH340 COM port was not found. Connect the Uno, close "
            "Arduino Serial Monitor, and retry. Detected ports:\n  "
            + (listing or "(none)"))
    best_score = usable[0][0]
    best = [port for score, port in usable if score == best_score]
    if len(best) != 1:
        listing = ", ".join(str(port.device) for port in best)
        raise RuntimeError(
            "Multiple likely arm ports were found: "
            f"{listing}. Run with --port COMx.")
    return str(best[0].device)


def open_arm(port):
    """Open ArmSerial without passing POSIX-only ``exclusive`` on Windows."""
    original_serial = arm_serial.serial.Serial
    if os.name == "nt":
        def windows_serial(*args, **kwargs):
            kwargs.pop("exclusive", None)
            return original_serial(*args, **kwargs)
        arm_serial.serial.Serial = windows_serial
    try:
        return arm_serial.ArmSerial(port=port, mock=False)
    finally:
        arm_serial.serial.Serial = original_serial


class DirectArmClient:
    """In-process equivalent of ``ArmSessionClient`` for Windows."""

    def __init__(self, arm, safety=None, camera_path=WRIST_RAW_FRAME):
        self.arm = arm
        self.safety = safety or PhysicalArmSafety()
        self.camera_path = Path(camera_path)

    def _assert_camera_live(self):
        try:
            age = time.time() - self.camera_path.stat().st_mtime
        except FileNotFoundError as exc:
            raise RuntimeError(
                "wrist camera is not publishing frames") from exc
        if age < -1.0 or age > config.WRIST_CAMERA_MAX_FRAME_AGE_S:
            raise RuntimeError(
                f"wrist camera frame is stale ({age:.1f}s old)")

    @staticmethod
    def _pose(payload):
        pose = payload.get("pose")
        if not isinstance(pose, list) or len(pose) != config.N_JOINTS:
            raise ValueError("pose must contain six joint angles")
        return [int(value) for value in pose]

    def _checked_move(self, target, require_camera=True, wait=True,
                      timeout=15.0, settle_s=0.0):
        if require_camera:
            self._assert_camera_live()
        current = self.arm.status()
        report = self.safety.transition_report(current, target)
        if not report.safe:
            raise RuntimeError(
                "motion rejected before COM write: " + report.explain())
        self.arm.send_angles(target)
        if wait:
            self.arm.wait_done(timeout=float(timeout))
            if settle_s:
                time.sleep(float(settle_s))
            return self.arm.status()
        return current

    def request(self, payload, timeout=30.0):
        del timeout
        command = payload.get("command")
        if command == "ping":
            return {"ok": bool(self.arm.ping())}
        if command == "status":
            return {"ok": True, "pose": self.arm.status()}
        if command == "distance":
            samples = int(payload.get("samples", config.ULTRASONIC_SAMPLES))
            distance = self.arm.ultrasonic_distance_mm(samples=samples)
            return {
                "ok": True,
                "distanceMm": distance,
                "valid": distance is not None,
                "samples": samples,
            }
        if command == "move":
            target = self._pose(payload)
            pose = self._checked_move(
                target,
                require_camera=bool(payload.get("require_camera", False)),
                wait=True,
                timeout=float(payload.get("timeout", 15.0)),
                settle_s=float(payload.get("settle_s", 0.0)),
            )
            return {"ok": True, "pose": pose}
        if command == "stream":
            target = self._pose(payload)
            current = self.arm.status()
            largest = max(
                abs(goal - actual)
                for goal, actual in zip(target, current))
            if largest > REALTIME_MAX_TARGET_DELTA_DEG:
                raise RuntimeError(
                    f"stream step {largest}deg exceeds "
                    f"{REALTIME_MAX_TARGET_DELTA_DEG}deg")
            pose = self._checked_move(
                target,
                require_camera=bool(payload.get("require_camera", True)),
                wait=False,
            )
            return {
                "ok": True,
                "pose": pose,
                "target": target,
                "streaming": True,
            }
        raise ValueError(f"unsupported Windows arm command: {command!r}")

    def close(self):
        self.arm.close()
