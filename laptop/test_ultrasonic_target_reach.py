import unittest
from types import SimpleNamespace

from ultrasonic_target_reach import (
    _final_grasp_gate,
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

    def test_final_close_requires_object_between_physical_fingers(self):
        scene = SimpleNamespace(gripper=SimpleNamespace(
            center=(640.0, 690.0), opening_px=300.0))
        aligned = SimpleNamespace(
            center=(650.0, 500.0), bbox=(610, 400, 80, 240))
        overshot = SimpleNamespace(
            center=(650.0, 650.0), bbox=(610, 520, 80, 190))

        self.assertTrue(_final_grasp_gate(scene, aligned)[0])
        self.assertFalse(_final_grasp_gate(scene, overshot)[0])

if __name__ == "__main__":
    unittest.main()
