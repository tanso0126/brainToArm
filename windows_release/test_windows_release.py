import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest import mock

import cv2
import numpy as np

import config
from windows_camera import gripper_score
from windows_support import (
    DirectArmClient,
    Wrist3DofDirectArmClient,
    find_arm_port,
)
from control_service import ControlCenterService


RELEASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = RELEASE_DIR.parent


class FakeArm:
    def __init__(self):
        self.pose = list(config.HOME_POSE)
        self.sent = []
        self.closed = False

    def status(self):
        return list(self.pose)

    def send_angles(self, pose):
        self.pose = list(pose)
        self.sent.append(list(pose))

    def wait_done(self, timeout):
        self.timeout = timeout

    def ultrasonic_distance_mm(self, samples):
        return 42.0

    def ping(self):
        return True

    def close(self):
        self.closed = True

    def stop_motion(self):
        self.stopped = True
        return True


class WindowsReleaseTests(unittest.TestCase):
    def test_batch_launchers_are_ascii_crlf(self):
        for path in RELEASE_DIR.glob("*.bat"):
            payload = path.read_bytes()
            self.assertTrue(payload.isascii(), path.name)
            self.assertNotIn(b"\n", payload.replace(b"\r\n", b""))

    def test_powershell_launchers_have_utf8_bom(self):
        for name in (
                "setup_windows.ps1", "open_firmware.ps1",
                "launch_tool.ps1", "start_control_center.ps1"):
            payload = (RELEASE_DIR / name).read_bytes()
            self.assertTrue(payload.startswith(b"\xef\xbb\xbf"), name)

    def test_firmware_launcher_target_is_shipped(self):
        sketch = ROOT_DIR / "firmware" / "arm_controller" / "arm_controller.ino"
        self.assertTrue(sketch.is_file())

    def test_embedded_windows_ui_is_built(self):
        ui = RELEASE_DIR / "assets" / "ui"
        index = (ui / "index.html").read_text(encoding="utf-8")
        self.assertIn("brainToArm 통합 운영실", index)
        self.assertTrue((ui / "eeg-field.png").is_file())
        self.assertTrue(any((ui / "assets").glob("*.js")))
        self.assertTrue(any((ui / "assets").glob("*.css")))

    def test_single_installer_build_files_exist(self):
        self.assertTrue((RELEASE_DIR / "brainToArm.spec").is_file())
        self.assertTrue((RELEASE_DIR / "brainToArm.iss").is_file())
        workflow = ROOT_DIR / ".github" / "workflows" / "build-windows-app.yml"
        self.assertIn(
            "brainToArm-Windows-Setup", workflow.read_text(encoding="utf-8"))

    def test_control_center_uses_embedded_ui_and_frozen_camera_worker(self):
        center = (RELEASE_DIR / "control_center.py").read_text(encoding="utf-8")
        service = (RELEASE_DIR / "control_service.py").read_text(encoding="utf-8")
        self.assertIn('UI_ROOT = RELEASE / "assets" / "ui"', center)
        self.assertNotIn("npm.cmd", center)
        self.assertIn('"--camera-worker"', service)
        self.assertIn("def open_firmware", service)
        self.assertIn('"/api/control/firmware/open"', center)

    def test_port_selection_prefers_uno_and_rejects_esp32(self):
        ports = [
            SimpleNamespace(
                device="COM3", description="Silicon Labs CP210x",
                manufacturer="Silicon Labs", hwid="VID:PID=10C4:EA60",
                vid=0x10C4),
            SimpleNamespace(
                device="COM5", description="USB-SERIAL CH340",
                manufacturer="wch.cn", hwid="VID:PID=1A86:7523",
                vid=0x1A86),
        ]

        self.assertEqual(find_arm_port(ports=ports), "COM5")

    def test_explicit_port_is_normalized(self):
        self.assertEqual(find_arm_port("com7", ports=[]), "COM7")

    def test_direct_client_moves_without_unix_socket(self):
        arm = FakeArm()
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame.jpg"
            frame.write_bytes(b"live")
            os.utime(frame, (time.time(), time.time()))
            safety = SimpleNamespace(
                transition_report=lambda _start, _target:
                SimpleNamespace(safe=True, explain=lambda: "safe"))
            client = DirectArmClient(
                arm, safety=safety, camera_path=frame)
            target = list(config.HOME_POSE)
            target[config.J_WRIST] = min(
                config.SERVO_MAX[config.J_WRIST],
                target[config.J_WRIST] + 5)

            result = client.request({
                "command": "move", "pose": target,
                "require_camera": True,
            })

        self.assertEqual(result["pose"], target)
        self.assertEqual(arm.sent, [target])

    def test_stale_camera_blocks_motion(self):
        arm = FakeArm()
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame.jpg"
            frame.write_bytes(b"stale")
            old = time.time() - 30
            os.utime(frame, (old, old))
            client = DirectArmClient(
                arm,
                safety=mock.Mock(),
                camera_path=frame,
            )
            with self.assertRaisesRegex(RuntimeError, "갱신되지 않았습니다"):
                client.request({
                    "command": "move",
                    "pose": list(config.HOME_POSE),
                    "require_camera": True,
                })
        self.assertEqual(arm.sent, [])

    def test_direct_client_exposes_firmware_emergency_hold(self):
        arm = FakeArm()
        client = DirectArmClient(arm)

        response = client.request({"command": "stop"})

        self.assertTrue(response["ok"])
        self.assertTrue(arm.stopped)

    def test_control_center_defaults_to_repaired_wrist_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            service = ControlCenterService(
                settings_path=Path(directory) / "settings.json")
            status = service.status()

        self.assertEqual(status["arm"]["mode"], "wrist-3dof")
        self.assertEqual(status["arm"]["activeServos"], [2, 3, 4, 5])
        self.assertEqual(status["arm"]["fixedServos"], [1, 6])
        self.assertFalse(status["camera"]["running"])
        self.assertFalse(status["arm"]["connected"])

    def test_wrist_mode_never_changes_base_or_roll(self):
        arm = FakeArm()
        arm.pose = [87, 70, 90, 140, 90, 23]
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame.jpg"
            frame.write_bytes(b"live")
            client = Wrist3DofDirectArmClient(arm, camera_path=frame)
            client.safety = SimpleNamespace(
                transition_report=lambda _start, _target:
                SimpleNamespace(safe=True, explain=lambda: "safe"))

            response = client.request({
                "command": "move",
                "pose": [0, 80, 100, 170, 180, 180],
            })

        self.assertEqual(response["pose"], [87, 80, 100, 170, 180, 23])
        self.assertEqual(arm.sent[-1][0], 87)
        self.assertEqual(arm.sent[-1][5], 23)

    def test_control_center_persists_validated_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            service = ControlCenterService(settings_path=path)
            saved = service.update_settings({
                "camera": "2", "armPort": "com7",
                "candidateReviewSeconds": 99,
            })
            restored = ControlCenterService(settings_path=path)

        self.assertEqual(saved["armPort"], "COM7")
        self.assertEqual(saved["candidateReviewSeconds"], 15.0)
        self.assertEqual(restored.status()["settings"]["camera"], "2")

    def test_firmware_has_immediate_stop_command(self):
        source = (ROOT_DIR / "firmware" / "arm_controller"
                  / "arm_controller.ino").read_text(encoding="utf-8")
        self.assertIn("case 'X':", source)
        self.assertIn('Serial.println("STOPPED")', source)

    def test_camera_score_requires_real_marker_pair(self):
        blank = np.full((720, 1280, 3), 240, dtype=np.uint8)
        self.assertEqual(gripper_score(blank), 0.0)

    def test_camera_score_accepts_blue_left_red_right_fingers(self):
        frame = np.full((720, 1280, 3), 240, dtype=np.uint8)
        cv2.rectangle(frame, (450, 620), (520, 710), (255, 0, 0), -1)
        cv2.rectangle(frame, (750, 620), (820, 710), (0, 0, 255), -1)

        self.assertGreaterEqual(gripper_score(frame), 1000.0)


if __name__ == "__main__":
    unittest.main()
