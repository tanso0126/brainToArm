import unittest
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

import config
from ultrasonic_target_reach import (
    _best_final_grasp_candidate,
    _final_grasp_gate,
    _preclose_needs_fine_lift,
    _retained_image_gate,
    _retained_corridor_candidate,
    _candidate_on_sonar_axis,
    adaptive_advance_mm,
    approach_observation_pose,
    approach_stays_forward,
    approach_stop_decision,
    fingertip_floor_clearance_mm,
    grip_hold_pose,
    home_pose_holding,
    loaded_home_reassert_pose,
    open_ready_pose,
    select_after_external_decisions,
    tracking_wrist_target,
    transition_fingertip_floor_clearance_mm,
    vivid_table_candidates,
)
from look_reach import LookReachTargetSelector
from decision_signal import DecisionMailbox


class ApproachStopTests(unittest.TestCase):
    def test_decision_mailbox_ignores_stale_signal(self):
        with TemporaryDirectory() as directory:
            mailbox = DecisionMailbox(Path(directory) / "decision.json")
            old = mailbox.emit("reject", source="test")
            cursor = mailbox.cursor()

            self.assertEqual(cursor, old.sequence)
            self.assertIsNone(mailbox.wait_after(cursor, timeout_s=0))

            fresh = mailbox.emit("accept", source="test")
            self.assertEqual(
                mailbox.wait_after(cursor, timeout_s=0), fresh)

    def test_external_reject_cycles_multiple_objects_and_resets(self):
        candidates = [
            SimpleNamespace(
                center=(640.0, 350.0), bbox=(610, 310, 60, 80),
                area=4800.0, confidence=0.95),
            SimpleNamespace(
                center=(650.0, 570.0), bbox=(620, 530, 60, 80),
                area=4800.0, confidence=0.95),
        ]
        scene = SimpleNamespace(
            ranked=candidates,
            frame_shape=(720, 1280, 3),
            gripper=SimpleNamespace(center=(640.0, 680.0), opening_px=320.0),
        )
        selector = LookReachTargetSelector(
            reachability=lambda *_args, **_kwargs: (True, "ok"),
            logger=None,
        )
        first = selector.choose(scene, pose=[90, 70, 90, 140, 90, 170])

        switched = select_after_external_decisions(
            scene, selector, [90, 70, 90, 140, 90, 170],
            1280, ["reject"])
        reset = select_after_external_decisions(
            scene, selector, [90, 70, 90, 140, 90, 170],
            1280, ["reject"])

        self.assertIs(first, candidates[0])
        self.assertIs(switched["candidate"], candidates[1])
        self.assertFalse(switched["cycleReset"])
        self.assertIs(reset["candidate"], candidates[0])
        self.assertTrue(reset["cycleReset"])

    def test_fixed_base_selection_ignores_lateral_candidate(self):
        lateral = SimpleNamespace(
            center=(400.0, 500.0), bbox=(370, 460, 60, 80),
            area=4800.0, confidence=0.95)
        axial = SimpleNamespace(
            center=(640.0, 570.0), bbox=(610, 530, 60, 80),
            area=4800.0, confidence=0.95)
        scene = SimpleNamespace(
            ranked=[lateral, axial],
            frame_shape=(720, 1280, 3),
            gripper=SimpleNamespace(center=(640.0, 680.0), opening_px=320.0),
        )
        selector = LookReachTargetSelector(
            reachability=lambda *_args, **_kwargs: (True, "ok"),
            logger=None,
        )

        result = select_after_external_decisions(
            scene, selector, [90, 70, 90, 140, 90, 170],
            1280, [])

        self.assertEqual(result["candidateCount"], 1)
        self.assertIs(result["candidate"], axial)

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

    def test_preclose_fine_lift_only_corrects_vertical_gap(self):
        scene = SimpleNamespace(gripper=SimpleNamespace(
            center=(620.0, 696.0), opening_px=290.0))
        measured_far = SimpleNamespace(
            center=(610.0, 445.0), bbox=(580, 315, 60, 255))
        laterally_wrong = SimpleNamespace(
            center=(800.0, 445.0), bbox=(770, 315, 60, 255))
        ready = SimpleNamespace(
            center=(610.0, 480.0), bbox=(580, 350, 60, 285))

        self.assertTrue(_preclose_needs_fine_lift(scene, measured_far))
        self.assertFalse(_preclose_needs_fine_lift(
            scene, laterally_wrong))
        self.assertFalse(_preclose_needs_fine_lift(scene, ready))

    def test_final_gate_uses_complete_nested_mask_of_same_object(self):
        partial = SimpleNamespace(
            center=(586.0, 420.0), bbox=(535, 310, 102, 246),
            area=18000.0)
        complete = SimpleNamespace(
            center=(584.0, 450.0), bbox=(532, 183, 105, 537),
            area=56000.0)
        unrelated = SimpleNamespace(
            center=(760.0, 450.0), bbox=(720, 183, 80, 537),
            area=43000.0)
        scene = SimpleNamespace(
            gripper=SimpleNamespace(
                center=(625.0, 694.0), opening_px=291.0),
            ranked=[partial, complete, unrelated],
        )

        selected = _best_final_grasp_candidate(scene, partial)

        self.assertIs(selected, complete)
        self.assertTrue(_final_grasp_gate(scene, selected)[0])

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

    def test_retention_recovers_vivid_object_inside_closed_jaws(self):
        frame = np.full((720, 1280, 3), 220, dtype=np.uint8)
        cv2.rectangle(frame, (610, 365), (650, 719), (255, 0, 0), -1)
        scene = SimpleNamespace(
            marker_boxes=[(485, 518, 125, 202), (669, 612, 99, 108)])

        candidate = _retained_corridor_candidate(frame, scene)

        self.assertIsNotNone(candidate)
        self.assertTrue(_retained_image_gate(frame, candidate)[0])

    def test_retention_corridor_rejects_short_background_blob(self):
        frame = np.full((720, 1280, 3), 220, dtype=np.uint8)
        cv2.rectangle(frame, (610, 640), (650, 719), (255, 0, 0), -1)
        scene = SimpleNamespace(
            marker_boxes=[(485, 518, 125, 202), (669, 612, 99, 108)])

        self.assertIsNone(_retained_corridor_candidate(frame, scene))

    def test_loaded_home_preserves_only_closed_gripper_value(self):
        held = [90, 111, 70, 177, 180, 170]
        self.assertEqual(
            home_pose_holding(held), [90, 70, 90, 140, 180, 170])
        self.assertEqual(
            loaded_home_reassert_pose(held),
            [90, 90, 90, 150, 180, 170])

    def test_loaded_hold_reduces_stall_without_opening(self):
        closed = [90, 108, 78, 172, 180, 170]
        holding = grip_hold_pose(closed)

        self.assertEqual(holding, [90, 108, 78, 172, 158, 170])
        self.assertGreater(holding[4], config.GRIP_OPEN)
        self.assertLess(holding[4], config.GRIP_CLOSED)
        self.assertEqual(
            home_pose_holding(holding), [90, 70, 90, 140, 158, 170])
        self.assertEqual(
            loaded_home_reassert_pose(holding),
            [90, 90, 90, 150, 158, 170])

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
