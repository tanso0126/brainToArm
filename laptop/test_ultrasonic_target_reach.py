import unittest

from ultrasonic_target_reach import (
    adaptive_advance_mm,
    approach_stop_decision,
    fingertip_floor_clearance_mm,
    transition_fingertip_floor_clearance_mm,
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
        self.assertEqual(adaptive_advance_mm(220), 15.0)
        self.assertEqual(adaptive_advance_mm(150), 10.0)
        self.assertEqual(adaptive_advance_mm(110), 10.0)

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


if __name__ == "__main__":
    unittest.main()
