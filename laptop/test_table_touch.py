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


HOVER = [90, 124, 90, 180, 90, 170]


class FakeClient:
    def __init__(self):
        self.pose = list(HOVER)

    def request(self, payload):
        if payload != {"command": "status"}:
            raise AssertionError(f"unexpected arm request: {payload}")
        return {"pose": list(self.pose)}


class FakeMover:
    def __init__(self, client):
        self.client = client
        self.marker_detector = object()
        self.moves = []

    def slow_move(self, target, final_settle=None):
        self.moves.append(list(target))
        self.client.pose = list(target)


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

    def test_deep_path_uses_distinct_fine_steps_below_minus_4(self):
        path = fixed_pitch_path(x_m=0.330, minimum_z_m=-0.018)
        fine = [(z, arm_fk.tool_position(pose)[2])
                for z, pose in path if z < -0.0045]
        self.assertEqual(
            [round(z * 1000.0) for z, _actual in fine],
            list(range(-5, -19, -1)))
        descents = [(before - after) * 1000.0
                    for (_za, before), (_zb, after)
                    in zip(fine, fine[1:])]
        self.assertTrue(all(0.35 <= amount <= 1.6 for amount in descents))

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

    def test_sparse_primary_roi_expands_near_wood(self):
        rng = np.random.default_rng(8)
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Features live in the expanded side bands, outside the old x ROI.
        for low_x, high_x in ((60, 145), (1020, 1215)):
            for x, y in rng.integers(
                    [low_x, 80], [high_x, 610], size=(150, 2)):
                cv2.circle(image, (int(x), int(y)), 2, (180, 180, 180), -1)
        matrix = np.float32([[1, 0, 2.0], [0, 1, -3.0]])
        shifted = cv2.warpAffine(image, matrix, (1280, 720))
        flow, points = median_table_flow(image, shifted)
        self.assertGreaterEqual(points, 150)
        self.assertTrue(np.isfinite(flow))
        self.assertAlmostEqual(flow, np.hypot(2.0, 3.0), delta=0.4)
        invalid = image.astype(np.float32)
        invalid[100, 100, 0] = np.nan
        self.assertEqual(median_table_flow(invalid, invalid), (None, 0))

    def _run_mock(self, path, flow_results, signatures, minimum_z_m):
        client = FakeClient()
        mover = FakeMover(client)
        flows = iter(flow_results)
        signature_values = None if signatures is None else iter(signatures)

        def fake_flow(_before, _after):
            return next(flows)

        def fake_signature(_detector, _frame):
            if signature_values is None:
                return None
            shift = next(signature_values)
            return np.asarray((shift, 0.0, 300.0, 0.0))

        with (patch("table_touch_calibrate.fixed_pitch_path",
                    return_value=path),
              patch("table_touch_calibrate.FloorServo",
                    return_value=mover),
              patch("table_touch_calibrate._fresh_frame",
                    side_effect=range(1000)),
              patch("table_touch_calibrate.median_table_flow",
                    side_effect=fake_flow),
              patch("table_touch_calibrate.gripper_signature",
                    side_effect=fake_signature)):
            result = run_touch_trial(
                client, execute=True, touch_x_m=0.330,
                minimum_z_m=minimum_z_m)
        return result, mover

    def test_x330_free_descent_to_minus_2_reports_no_contact(self):
        """The clean-wood hardware run stays free through the old safety floor."""
        path = fixed_pitch_path(x_m=0.330, minimum_z_m=-0.002)
        count = len(path) - 1
        points = [262] * count
        points[-2:] = [75, 105]
        flows = [(4.0 + 0.2 * (index % 3), points[index])
                 for index in range(count)]

        result, _mover = self._run_mock(
            path, flows, signatures=None, minimum_z_m=-0.002)

        self.assertEqual(result["state"], "no-contact")
        self.assertNotIn("z_table_mm", result)
        self.assertLess(min(record["flow_points"]
                            for record in result["records"][-4:]), 150)

    def test_deep_x330_marker_onset_confirms_near_minus_9(self):
        """Expected table contact is around FK/command z=-12..-6 mm."""
        path = fixed_pitch_path(x_m=0.330, minimum_z_m=-0.018)
        onset_index = min(
            range(len(path)), key=lambda index: abs(path[index][0] + 0.010))
        confirmation_index = onset_index + 1
        contact_index = onset_index - 1
        self.assertAlmostEqual(path[onset_index][0], -0.010, places=6)

        # Primary descent through the second loaded marker sample.
        primary_shifts = []
        for path_index in range(1, confirmation_index + 1):
            if path_index == onset_index:
                primary_shifts.append(4.95)
            elif path_index == confirmation_index:
                primary_shifts.append(6.00)
            else:
                primary_shifts.append(0.0)
        signatures = (
            [0.0] + primary_shifts
            + [0.0, 0.0, 4.95, 6.00])  # replay baseline + -9/-10/-11
        flow_count = confirmation_index + 3
        flows = [(5.0, 220)] * flow_count

        result, mover = self._run_mock(
            path, flows, signatures, minimum_z_m=-0.018)

        self.assertEqual(result["state"], "contact")
        self.assertGreaterEqual(result["z_table_mm"], -12.0)
        self.assertLessEqual(result["z_table_mm"], -6.0)
        self.assertAlmostEqual(
            result["z_table_mm"], path[contact_index][0] * 1000.0,
            places=5)

        # Confirmation still retreats above 30 mm before any new prepose.
        first_candidate = mover.moves.index(path[confirmation_index][1])
        preposes = {tuple(backlash_prepose(pose)) for _z, pose in path}
        next_prepose = next(
            index for index in range(first_candidate + 1, len(mover.moves))
            if tuple(mover.moves[index]) in preposes)
        retry_lift = mover.moves[first_candidate + 1:next_prepose]
        self.assertTrue(retry_lift)
        self.assertGreaterEqual(
            max(arm_fk.tool_position(pose)[2] for pose in retry_lift), 0.030)


if __name__ == "__main__":
    unittest.main()
