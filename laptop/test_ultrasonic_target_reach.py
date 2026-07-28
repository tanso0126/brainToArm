import unittest

from ultrasonic_target_reach import (
    adaptive_advance_mm,
    fused_progress_decision,
    range_progress_decision,
    update_range_progress,
)


class RangeProgressTests(unittest.TestCase):
    def test_monotonic_approach_continues(self):
        self.assertEqual(
            range_progress_decision([159.0, 156.0]).action, "continue")

    def test_near_standoff_wins(self):
        decision = range_progress_decision([90.0, 77.0])
        self.assertEqual(decision.action, "near")

    def test_large_range_increase_stops(self):
        decision = range_progress_decision([130.0, 139.0])
        self.assertEqual(decision.action, "stop")
        self.assertIn("increased", decision.reason)

    def test_long_window_without_depth_progress_stops(self):
        decision = range_progress_decision(
            [130.0, 129.7, 129.0, 129.5, 129.0])
        self.assertEqual(decision.action, "stop")
        self.assertIn("too small", decision.reason)

    def test_local_jitter_does_not_override_good_long_trend(self):
        decision = range_progress_decision(
            [152.0, 148.8, 141.2, 144.2, 142.0])
        self.assertEqual(decision.action, "continue")

    def test_aim_only_rotation_resets_range_baseline(self):
        history, decision = update_range_progress(
            [142.0], 151.8, previous_was_approach=False)
        self.assertEqual(history, [151.8])
        self.assertEqual(decision.action, "continue")

    def test_adaptive_step_is_servo_scale_until_final_band(self):
        self.assertEqual(adaptive_advance_mm(220), 15.0)
        self.assertEqual(adaptive_advance_mm(150), 10.0)
        self.assertEqual(adaptive_advance_mm(110), 5.0)

    def test_visual_progress_overrides_ambiguous_sonar_plateau(self):
        decision = fused_progress_decision(
            [134.5, 132.5, 132.0, 132.0, 132.0],
            [183.0, 180.0, 163.0, 134.0, 105.0],
            jaw_ready=False,
        )
        self.assertEqual(decision.action, "continue")
        self.assertIn("jaw-row gap fell", decision.reason)

    def test_jaw_ready_stops_at_positive_sensor_distance(self):
        decision = fused_progress_decision(
            [134.5, 132.5, 132.0, 132.0, 132.0],
            [183.0, 180.0, 163.0, 134.0, 72.0],
            jaw_ready=True,
        )
        self.assertEqual(decision.action, "ready")

    def test_near_sonar_without_jaw_alignment_stops(self):
        decision = fused_progress_decision(
            [90.0, 77.0], [180.0, 170.0], jaw_ready=False)
        self.assertEqual(decision.action, "stop")
        self.assertIn("not inside", decision.reason)


if __name__ == "__main__":
    unittest.main()
