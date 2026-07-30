import unittest
from types import SimpleNamespace

import config
from arm_session import ArmSessionServer, REALTIME_MAX_TARGET_DELTA_DEG


class FakeArm:
    def __init__(self, pose=None):
        self.pose = list(pose or config.HOME_POSE)
        self.targets = []

    def status(self):
        return list(self.pose)

    def send_angles(self, target):
        self.targets.append(list(target))
        return "OK"


class SafeTransitions:
    @staticmethod
    def transition_report(_start, _target):
        return SimpleNamespace(
            safe=True,
            minimum_clearance_mm=42.0,
            explain=lambda: "safe",
        )


class RealtimeArmSessionTests(unittest.TestCase):
    def test_stream_replaces_target_without_waiting_for_done(self):
        arm = FakeArm()
        server = ArmSessionServer(arm=arm, safety=SafeTransitions())
        target = list(config.HOME_POSE)
        target[config.J_WRIST] += 3

        response = server.handle({
            "command": "stream",
            "pose": target,
            "require_camera": False,
        })

        self.assertTrue(response["streaming"])
        self.assertEqual(response["target"], target)
        self.assertEqual(arm.targets, [target])

    def test_stream_rejects_a_large_discontinuous_jump(self):
        arm = FakeArm()
        server = ArmSessionServer(arm=arm, safety=SafeTransitions())
        target = list(config.HOME_POSE)
        target[config.J_SHOULDER] += REALTIME_MAX_TARGET_DELTA_DEG + 1

        with self.assertRaisesRegex(RuntimeError, "largest live delta"):
            server.handle({
                "command": "stream",
                "pose": target,
                "require_camera": False,
            })

        self.assertEqual(arm.targets, [])


if __name__ == "__main__":
    unittest.main()
