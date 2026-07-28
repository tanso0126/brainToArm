import unittest

from ultrasonic_target_reach import range_progress_decision


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

    def test_two_steps_without_depth_progress_stop(self):
        decision = range_progress_decision([130.0, 129.7, 129.0])
        self.assertEqual(decision.action, "stop")
        self.assertIn("too small", decision.reason)


if __name__ == "__main__":
    unittest.main()
