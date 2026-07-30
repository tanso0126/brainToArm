import unittest
from types import SimpleNamespace

import cv2
import numpy as np

import arm_fk
from realtime_visual_servo import (
    HistogramTargetTracker,
    dynamic_aim_y,
    grasp_readiness,
    numeric_task_jacobian,
    resolved_velocity_target,
    select_realtime_seed,
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

    def test_first_temporal_frame_reacquires_beyond_seed_box(self):
        first = np.full((240, 320, 3), 235, dtype=np.uint8)
        shifted = first.copy()
        cv2.rectangle(first, (80, 80), (110, 150), (180, 60, 120), -1)
        cv2.rectangle(
            shifted, (140, 85), (170, 155), (180, 60, 120), -1)
        tracker = HistogramTargetTracker()
        tracker.initialize(first, (80, 80, 30, 70))

        tracked = tracker.update(shifted)

        self.assertIsNotNone(tracked)
        self.assertAlmostEqual(tracked.center[0], 155.0, delta=10.0)

    def test_coloured_feature_preserves_whole_object_centre(self):
        frame = np.full((240, 320, 3), 245, dtype=np.uint8)
        # FastSAM's whole object box contains a small coloured feature and a
        # mostly overexposed body, matching the physical ruler/eraser view.
        cv2.rectangle(frame, (100, 50), (200, 190), (250, 250, 250), -1)
        cv2.rectangle(frame, (160, 70), (185, 110), (180, 60, 120), -1)
        tracker = HistogramTargetTracker()
        tracker.initialize(frame, (100, 50, 100, 140))

        tracked = tracker.update(frame)

        self.assertIsNotNone(tracked)
        self.assertAlmostEqual(tracked.center[0], 150.0, delta=8.0)
        self.assertAlmostEqual(tracked.center[1], 120.0, delta=8.0)
        self.assertGreater(tracked.bbox[2], 80)
        self.assertGreater(tracked.bbox[3], 110)

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

    def test_realtime_seed_does_not_use_a_fixed_image_horizon(self):
        near_real_object = SimpleNamespace(
            center=(600.0, 284.0), area=4200.0,
            confidence=0.85, median_saturation=70.0)
        higher_background = SimpleNamespace(
            center=(592.0, 169.0), area=4200.0,
            confidence=0.85, median_saturation=70.0)
        scene = SimpleNamespace(
            ranked=[higher_background, near_real_object],
            frame_shape=(720, 1280, 3),
            gripper=None,
        )

        self.assertIs(select_realtime_seed(scene), near_real_object)

    def test_realtime_seed_excludes_expected_bottom_finger_lobes(self):
        real_object = SimpleNamespace(
            center=(600.0, 676.0), area=6200.0,
            confidence=0.80, median_saturation=70.0)
        blue_finger = SimpleNamespace(
            center=(503.0, 640.0), area=6200.0,
            confidence=0.90, median_saturation=6.0)
        scene = SimpleNamespace(
            ranked=[blue_finger, real_object],
            frame_shape=(720, 1280, 3),
            gripper=None,
        )

        self.assertIs(select_realtime_seed(scene), real_object)

    def test_realtime_seed_recovers_vivid_object_on_white_frame(self):
        frame = np.full((720, 1280, 3), 255, dtype=np.uint8)
        cv2.rectangle(frame, (530, 20), (650, 180), (220, 50, 150), -1)
        scene = SimpleNamespace(
            ranked=[],
            frame_shape=frame.shape,
            gripper=None,
        )

        selected = select_realtime_seed(scene, frame)

        self.assertIsNotNone(selected)
        self.assertGreater(selected.area, 10000)
        self.assertLess(abs(selected.center[0] - 590), 5)

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

    def test_measured_high_target_pose_can_continue_forward(self):
        pose = [90, 77, 90, 163, 90, 180]

        plan = resolved_velocity_target(
            pose, vertical_error_px=-300,
            distance_mm=72.0, floor_clearance_mm=231.0)

        self.assertIsNotNone(plan)
        self.assertGreater(
            arm_fk.geometry(plan["pose"]).finger_tip[0],
            arm_fk.geometry(pose).finger_tip[0],
        )

    def test_near_aligned_pose_allows_small_inward_pitch_correction(self):
        pose = [90, 106, 55, 180, 90, 180]

        plan = resolved_velocity_target(
            pose, vertical_error_px=46,
            distance_mm=98.0, floor_clearance_mm=126.0)

        self.assertIsNotNone(plan)
        inward_mm = 1000.0 * (
            arm_fk.geometry(pose).finger_tip[0]
            - arm_fk.geometry(plan["pose"]).finger_tip[0])
        self.assertLessEqual(inward_mm, 6.0)


if __name__ == "__main__":
    unittest.main()
