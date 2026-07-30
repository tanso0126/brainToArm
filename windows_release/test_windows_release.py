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
from windows_support import DirectArmClient, find_arm_port


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


class WindowsReleaseTests(unittest.TestCase):
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
