import unittest
from unittest.mock import patch

import cv2
import numpy as np

import arm_fk
from look_reach import cumulative_tool_angle_deg
from table_touch_calibrate import (
    TOUCH_X_M, backlash_prepose, fixed_pitch_path, median_table_flow,
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

    def test_physical_marker_sequence_confirms_contact_after_safe_retry(self):
        """Regression for the 2026-07-27 x=300 mm physical touch log."""
        path = fixed_pitch_path()
        hover = [90, 124, 90, 180, 90, 170]

        class FakeClient:
            def __init__(self):
                self.pose = list(hover)

            def request(self, payload):
                self.assert_status(payload)
                return {"pose": list(self.pose)}

            @staticmethod
            def assert_status(payload):
                if payload != {"command": "status"}:
                    raise AssertionError(f"unexpected arm request: {payload}")

        class FakeMover:
            def __init__(self, client):
                self.client = client
                self.marker_detector = object()
                self.moves = []

            def slow_move(self, target, final_settle=None):
                self.moves.append(list(target))
                self.client.pose = list(target)

        client = FakeClient()
        mover = FakeMover(client)

        # First descent is the exact reported sequence. The z=4 flow collapse
        # does not repeat (3.97), after which marker deformation is sustained
        # at z=0 and z=-2. The final three values repeat the onset boundary.
        flows = iter([
            7.71, 2.69, 9.56, 4.04, 9.46, 4.41, 4.17, 24.11,
            3.99, 0.93,
            3.97,
            10.25, 11.17, 14.71,
            10.25, 11.17, 14.71,
        ])
        shifts = iter([
            0.0,                         # initial trial baseline
            0.0, 0.0, 0.0, 0.0, 0.0, 1.93, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,                    # flow-confirm baseline + z=4
            0.0, 4.95, 6.00,             # z=2, z=0, z=-2
            0.0, 0.0, 4.95, 6.00,        # confirm baseline + onset boundary
        ])

        def fake_flow(_before, _after):
            return next(flows), 100

        def fake_signature(_detector, _frame):
            return np.asarray((next(shifts), 0.0, 300.0, 0.0))

        with (patch("table_touch_calibrate.FloorServo",
                    return_value=mover),
              patch("table_touch_calibrate._fresh_frame",
                    side_effect=range(1000)),
              patch("table_touch_calibrate.median_table_flow",
                    side_effect=fake_flow),
              patch("table_touch_calibrate.gripper_signature",
                    side_effect=fake_signature)):
            result = run_touch_trial(client, execute=True)

        self.assertEqual(result["state"], "contact")
        self.assertGreaterEqual(result["z_table_mm"], 2.0)
        self.assertLessEqual(result["z_table_mm"], 6.0)

        # Reproduce the exact unsafe edge from hardware, then prove the retry
        # did not attempt it. After the first z=4 target, retreat poses climb
        # above 30 mm before the next backlash prepose is commanded.
        safety = PlanarSearchSafety()
        low = [90, 120, 136, 163, 90, 170]
        unsafe_prepose = [90, 101, 140, 154, 90, 170]
        self.assertFalse(safety.transition_is_safe(low, unsafe_prepose))

        first_candidate = mover.moves.index(path[10][1])
        preposes = {tuple(backlash_prepose(pose)) for _z, pose in path}
        next_prepose = next(
            index for index in range(first_candidate + 1, len(mover.moves))
            if tuple(mover.moves[index]) in preposes)
        retry_lift = mover.moves[first_candidate + 1:next_prepose]
        self.assertTrue(retry_lift)
        self.assertGreaterEqual(
            max(arm_fk.tool_position(pose)[2] for pose in retry_lift), 0.030)
        self.assertNotEqual(mover.moves[first_candidate + 1], unsafe_prepose)


if __name__ == "__main__":
    unittest.main()
