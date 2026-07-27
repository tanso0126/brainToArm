import unittest

import cv2
import numpy as np

import arm_fk
from look_reach import cumulative_tool_angle_deg
from table_touch_calibrate import (TOUCH_X_M, backlash_prepose,
                                   fixed_pitch_path, median_table_flow,
                                   run_touch_trial)
from wrist_search import PlanarSearchSafety


class TableTouchTests(unittest.TestCase):
    def test_fixed_pitch_path_holds_x_pitch_and_descends(self):
        path = fixed_pitch_path(minimum_z_m=0.010)
        pitch = cumulative_tool_angle_deg(path[0][1])
        heights = []
        for command_z, pose in path:
            tool = arm_fk.tool_position(pose)
            self.assertLess(abs(tool[0] - TOUCH_X_M), 0.002)
            # Integer servo commands quantize some requested 2 mm levels to
            # ~2.2 mm; runtime records the resulting FK z, not the ideal label.
            self.assertLess(abs(tool[2] - command_z), 0.0025)
            self.assertLess(abs(cumulative_tool_angle_deg(pose) - pitch), 1.5)
            heights.append(tool[2])
        self.assertTrue(all(a > b for a, b in zip(heights, heights[1:])))

    def test_backlash_prepose_finishes_from_below(self):
        target = [90, 120, 100, 160, 90, 170]
        prepose = backlash_prepose(target)
        self.assertEqual(prepose[1:4], [115, 95, 155])

    def test_default_dry_run_is_collision_checked_and_hardware_free(self):
        result = run_touch_trial(None, execute=False)
        self.assertEqual(result["state"], "planned")
        path = fixed_pitch_path()
        safety = PlanarSearchSafety()
        for (_za, a), (_zb, b) in zip(path, path[1:]):
            self.assertTrue(
                safety.transition_is_safe(a, backlash_prepose(b)))
            self.assertTrue(
                safety.transition_is_safe(backlash_prepose(b), b))

    def test_background_flow_recovers_known_translation(self):
        rng = np.random.default_rng(3)
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        for x, y in rng.integers([180, 120], [900, 500], size=(180, 2)):
            cv2.circle(image, (int(x), int(y)), 2, (255, 255, 255), -1)
        matrix = np.float32([[1, 0, 3.0], [0, 1, -4.0]])
        shifted = cv2.warpAffine(image, matrix, (1280, 720))
        flow, points = median_table_flow(image, shifted)
        self.assertGreater(points, 50)
        self.assertAlmostEqual(flow, 5.0, delta=0.35)


if __name__ == "__main__":
    unittest.main()
