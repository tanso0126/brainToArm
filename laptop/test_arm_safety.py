import tempfile
import time
import unittest
from pathlib import Path

import arm_fk
import config
from arm_safety import PhysicalArmSafety
from arm_session import ArmSessionServer


HOME = [90, 70, 90, 140, 170, 170]
HOVER = [90, 124, 90, 180, 90, 170]
GRASP = [90, 142, 90, 180, 90, 170]
VECTOR = [90, 110, 87, 150, 90, 170]
FAILED_PREPOSE = [90, 98, 168, 158, 90, 170]
FAILED_MEASUREMENT = [90, 104, 171, 163, 90, 170]


class FakeArm:
    def __init__(self, pose=HOVER):
        self.pose = list(pose)
        self.moves = []

    def status(self):
        return list(self.pose)

    def send_angles(self, pose):
        self.pose = list(pose)
        self.moves.append(list(pose))

    @staticmethod
    def wait_done(timeout=15.0):
        return True

    @staticmethod
    def ultrasonic_distance_mm(samples=None):
        return 237.0


class PhysicalArmSafetyTests(unittest.TestCase):
    def setUp(self):
        self.safety = PhysicalArmSafety()

    def test_authoritative_fk_is_real_scale_not_legacy_guess(self):
        chain = arm_fk.geometry(HOVER)
        upper_length = ((chain.elbow - chain.shoulder) ** 2).sum() ** 0.5
        self.assertAlmostEqual(upper_length, 0.241767, places=6)
        self.assertGreater(upper_length, 2.0 * config.L_UPPER / 100.0)

    def test_physically_exercised_reference_poses_remain_available(self):
        for pose in (HOME, HOVER, GRASP, VECTOR):
            with self.subTest(pose=pose):
                self.assertTrue(self.safety.pose_report(pose).safe)
        self.assertTrue(self.safety.transition_report(HOME, HOVER).safe)
        self.assertTrue(self.safety.transition_report(HOVER, GRASP).safe)

    def test_body_hook_incident_poses_are_rejected(self):
        for pose in (FAILED_PREPOSE, FAILED_MEASUREMENT):
            with self.subTest(pose=pose):
                report = self.safety.pose_report(pose)
                self.assertFalse(report.safe)
                self.assertIn("base-housing", report.explain())
        self.assertFalse(
            self.safety.transition_report(HOVER, FAILED_PREPOSE).safe)

    def test_server_interlock_prevents_serial_write(self):
        arm = FakeArm(HOVER)
        server = ArmSessionServer("/tmp/brainToArm-safety-test.sock", arm=arm)
        with self.assertRaisesRegex(RuntimeError, "rejected before serial write"):
            server.handle({"command": "move", "pose": FAILED_PREPOSE})
        self.assertEqual(arm.moves, [])
        self.assertEqual(arm.pose, HOVER)

    def test_table_touch_command_moves_only_the_bounded_table_plane(self):
        deep_prepose = [90, 137, 83, 152, 90, 170]
        deep_target = [90, 142, 88, 157, 90, 170]
        arm = FakeArm(deep_prepose)
        server = ArmSessionServer("/tmp/brainToArm-touch-test.sock", arm=arm)

        with self.assertRaisesRegex(RuntimeError, "rejected before serial write"):
            server.handle({"command": "move", "pose": deep_target})
        self.assertEqual(arm.moves, [])

        response = server.handle({
            "command": "table_touch_move",
            "pose": deep_target,
            "table_z_m": -0.018,
        })
        self.assertEqual(response["pose"], deep_target)
        self.assertEqual(arm.moves, [deep_target])

        with self.assertRaisesRegex(ValueError, "between -20 and 0 mm"):
            server.handle({
                "command": "table_touch_move",
                "pose": deep_target,
                "table_z_m": -0.021,
            })

        # Moving the table plane cannot bypass the independent base envelope.
        arm.pose = list(HOVER)
        with self.assertRaisesRegex(RuntimeError, "base-housing"):
            server.handle({
                "command": "table_touch_move",
                "pose": FAILED_PREPOSE,
                "table_z_m": -0.018,
            })

    def test_check_command_reports_without_moving(self):
        arm = FakeArm(HOVER)
        server = ArmSessionServer("/tmp/brainToArm-check-test.sock", arm=arm)
        report = server.handle({"command": "check", "pose": FAILED_PREPOSE})
        self.assertFalse(report["safe"])
        self.assertIn("base-housing", report["explanation"])
        self.assertEqual(arm.moves, [])

    def test_distance_command_reads_sensor_without_moving(self):
        arm = FakeArm(HOVER)
        server = ArmSessionServer("/tmp/brainToArm-distance-test.sock", arm=arm)
        response = server.handle({"command": "distance", "samples": 3})
        self.assertEqual(response["distanceMm"], 237.0)
        self.assertTrue(response["valid"])
        self.assertEqual(response["triggerPin"], 7)
        self.assertEqual(response["echoPin"], 6)
        self.assertEqual(arm.moves, [])

    def test_autonomous_motion_rejects_stale_camera(self):
        arm = FakeArm(HOVER)
        server = ArmSessionServer("/tmp/brainToArm-camera-test.sock", arm=arm)
        with tempfile.TemporaryDirectory() as directory:
            frame = Path(directory) / "frame.jpg"
            frame.touch()
            stale = time.time() - config.WRIST_CAMERA_MAX_FRAME_AGE_S - 5.0
            import os
            os.utime(frame, (stale, stale))
            with self.assertRaisesRegex(RuntimeError, "frame is stale"):
                server._assert_camera_live(frame)
        self.assertEqual(arm.moves, [])


if __name__ == "__main__":
    unittest.main()
