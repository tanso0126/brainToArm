import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from ultrasonic_target_reach import (
    _final_grasp_gate,
    _retained_image_gate,
    _candidate_on_sonar_axis,
    adaptive_advance_mm,
    approach_observation_pose,
    approach_stays_forward,
    approach_stop_decision,
    fingertip_floor_clearance_mm,
    home_pose_holding,
    loaded_home_reassert_pose,
    open_ready_pose,
    tracking_wrist_target,
    transition_fingertip_floor_clearance_mm,
    vivid_table_candidates,
)


class ApproachStopTests(unittest.TestCase):
    def test_only_two_normal_stop_conditions(self):
        self.assertEqual(
            approach_stop_decision(46.1, 10.1).action, "continue")
        self.assertEqual(
            approach_stop_decision(46.0, 40.0).action, "sonar")
        self.assertEqual(
            approach_stop_decision(100.0, 10.0).action, "floor")

    def test_floor_stop_has_priority_when_both_apply(self):
        self.assertEqual(
            approach_stop_decision(40.0, 8.0).action, "floor")

    def test_adaptive_step_never_falls_below_ten_mm(self):
        self.assertEqual(adaptive_advance_mm(220), 20.0)
        self.assertEqual(adaptive_advance_mm(150), 15.0)
        self.assertEqual(adaptive_advance_mm(110), 15.0)

    def test_fingertip_clearance_uses_distal_model_point(self):
        pose = [90, 107, 84, 178, 90, 170]
        self.assertAlmostEqual(
            fingertip_floor_clearance_mm(pose), 67.0, delta=1.0)

    def test_swept_clearance_includes_endpoints(self):
        high = [90, 107, 84, 178, 90, 170]
        low = [90, 111, 82, 180, 90, 170]
        swept = transition_fingertip_floor_clearance_mm(high, low)
        self.assertLessEqual(swept, fingertip_floor_clearance_mm(high))
        self.assertLessEqual(swept, fingertip_floor_clearance_mm(low) + 0.1)

    def test_measured_last_descent_crosses_floor_guard(self):
        safe_last = [90, 120, 73, 175, 90, 170]
        rejected_next = [90, 132, 66, 173, 90, 170]

        self.assertGreater(fingertip_floor_clearance_mm(safe_last), 10.0)
        self.assertLess(
            transition_fingertip_floor_clearance_mm(
                safe_last, rejected_next),
            10.0,
        )

    def test_final_close_requires_object_between_physical_fingers(self):
        scene = SimpleNamespace(gripper=SimpleNamespace(
            center=(640.0, 690.0), opening_px=300.0))
        aligned = SimpleNamespace(
            center=(650.0, 500.0), bbox=(610, 400, 80, 240))
        overshot = SimpleNamespace(
            center=(650.0, 680.0), bbox=(610, 540, 80, 220))
        measured_perfect = SimpleNamespace(
            center=(650.0, 600.0), bbox=(610, 470, 80, 245))

        self.assertTrue(_final_grasp_gate(scene, aligned)[0])
        self.assertFalse(_final_grasp_gate(scene, overshot)[0])
        self.assertTrue(_final_grasp_gate(scene, measured_perfect)[0])

    def test_motor_four_uses_measured_pixel_error_with_large_headroom(self):
        self.assertEqual(tracking_wrist_target(178, -60), 168)
        self.assertEqual(tracking_wrist_target(168, 30), 173)
        self.assertEqual(tracking_wrist_target(178, 0), 178)

    def test_retention_uses_bottom_clipped_locked_object(self):
        import numpy as np

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        held = SimpleNamespace(
            center=(614.0, 528.0), bbox=(566, 335, 97, 385))
        table = SimpleNamespace(
            center=(614.0, 430.0), bbox=(566, 300, 97, 260))

        self.assertTrue(_retained_image_gate(frame, held)[0])
        self.assertFalse(_retained_image_gate(frame, table)[0])

    def test_loaded_home_preserves_only_closed_gripper_value(self):
        held = [90, 111, 70, 177, 180, 170]
        self.assertEqual(
            home_pose_holding(held), [90, 70, 90, 140, 180, 170])
        self.assertEqual(
            loaded_home_reassert_pose(held),
            [90, 90, 90, 150, 180, 170])

    def test_full_cycle_opens_without_changing_home_observation_pose(self):
        home = [90, 70, 90, 140, 170, 170]
        self.assertEqual(
            open_ready_pose(home), [90, 70, 90, 140, 90, 170])

    def test_forward_observation_uses_reproduced_approach_branch(self):
        home_open = [90, 70, 90, 180, 90, 170]
        self.assertEqual(
            approach_observation_pose(home_open),
            [90, 107, 84, 178, 90, 170])

    def test_approach_cannot_fold_back_toward_body(self):
        forward = [90, 107, 84, 178, 90, 170]
        coupled = [90, 111, 80, 177, 90, 170]
        floor_approach = [90, 120, 73, 175, 90, 170]
        wrist_overwrite = [90, 111, 80, 168, 90, 170]
        known_close = [90, 124, 66, 179, 90, 170]
        folded = [90, 80, 131, 144, 90, 170]

        from ultrasonic_target_reach import fingertip_forward_x_mm
        start_x = fingertip_forward_x_mm(forward)
        self.assertTrue(approach_stays_forward(start_x, coupled))
        self.assertTrue(approach_stays_forward(start_x, floor_approach))
        self.assertFalse(approach_stays_forward(start_x, wrist_overwrite))
        self.assertTrue(approach_stays_forward(start_x, known_close))
        self.assertFalse(approach_stays_forward(start_x, folded))

    def test_search_rejects_home_background_outside_sonar_axis(self):
        near_axis = SimpleNamespace(center=(600.0, 450.0))
        background = SimpleNamespace(center=(546.0, 450.0))
        self.assertTrue(_candidate_on_sonar_axis(near_axis, 1280))
        self.assertFalse(_candidate_on_sonar_axis(background, 1280))

    def test_vivid_fallback_is_hue_independent_and_axis_ranked(self):
        frame = np.full((720, 1280, 3), 215, dtype=np.uint8)
        cv2.rectangle(frame, (625, 495), (650, 595), (255, 0, 0), -1)
        cv2.rectangle(frame, (430, 470), (520, 570), (0, 255, 255), -1)

        candidates = vivid_table_candidates(frame)

        self.assertGreaterEqual(len(candidates), 2)
        self.assertAlmostEqual(candidates[0].center[0], 637.5, delta=2.0)
        self.assertAlmostEqual(candidates[0].center[1], 545.0, delta=2.0)

    def test_vivid_fallback_excludes_finger_marker(self):
        frame = np.full((720, 1280, 3), 215, dtype=np.uint8)
        cv2.rectangle(frame, (625, 495), (650, 595), (0, 255, 255), -1)
        cv2.rectangle(frame, (700, 560), (760, 640), (255, 0, 0), -1)

        candidates = vivid_table_candidates(
            frame, marker_boxes=[(695, 555, 70, 90)])

        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0].center[0], 637.5, delta=2.0)

if __name__ == "__main__":
    unittest.main()
