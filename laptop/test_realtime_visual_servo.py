import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from realtime_visual_servo import (
    HistogramTargetTracker,
    dynamic_aim_y,
    grasp_readiness,
    numeric_task_jacobian,
    resolved_velocity_target,
)


class RealtimeVisualServoTests(unittest.TestCase):
    def test_histogram_tracker_follows_a_translated_arbitrary_colour(self):
        first = np.full((240, 320, 3), 235, dtype=np.uint8)
        second = first.copy()
        cv2.rectangle(first, (120, 80), (160, 160), (180, 60, 120), -1)
        cv2.rectangle(second, (132, 86), (172, 166), (180, 60, 120), -1)
        tracker = HistogramTargetTracker()
        tracker.initialize(first, (120, 80, 40, 80))

        tracked = tracker.update(second)

        self.assertIsNotNone(tracked)
        self.assertAlmostEqual(tracked.center[0], 152.0, delta=8.0)
        self.assertAlmostEqual(tracked.center[1], 126.0, delta=8.0)

    def test_dynamic_aim_moves_from_image_to_live_gripper_row(self):
        self.assertAlmostEqual(dynamic_aim_y(720, 300, 690), 403.2)
        self.assertAlmostEqual(dynamic_aim_y(720, 46, 690), 690.0)

    def test_grasp_requires_both_sonar_and_deep_centre_alignment(self):
        target = SimpleNamespace(center=(610.0, 680.0))

        far = grasp_readiness(target, (600.0, 690.0), 280.0, 96.5, 15.0)
        ready = grasp_readiness(target, (600.0, 690.0), 280.0, 46.0, 15.0)

        self.assertFalse(far.ready)
        self.assertIn("sonar", far.reason)
        self.assertTrue(ready.ready)

    def test_numeric_jacobian_and_resolved_target_are_bounded(self):
        pose = [90, 107, 84, 178, 90, 170]

        jacobian = numeric_task_jacobian(pose)
        plan = resolved_velocity_target(
            pose, vertical_error_px=-40,
            distance_mm=125.0, floor_clearance_mm=62.0)

        self.assertEqual(jacobian.shape, (3, 3))
        self.assertTrue(np.isfinite(jacobian).all())
        self.assertIsNotNone(plan)
        self.assertGreater(plan["desired_task"][0], 0.0)
        self.assertLessEqual(
            max(abs(a - b) for a, b in zip(plan["pose"], pose)), 5)

    def test_floor_hold_never_plans_below_the_ten_mm_guard(self):
        pose = [90, 120, 73, 175, 90, 170]

        plan = resolved_velocity_target(
            pose, vertical_error_px=-120,
            distance_mm=96.5, floor_clearance_mm=12.0)

        if plan is not None:
            from ultrasonic_target_reach import (
                transition_fingertip_floor_clearance_mm,
            )
            self.assertGreaterEqual(
                transition_fingertip_floor_clearance_mm(
                    pose, plan["pose"]),
                10.0,
            )


if __name__ == "__main__":
    unittest.main()
