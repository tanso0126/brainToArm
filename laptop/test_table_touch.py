import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import arm_fk
from look_reach import cumulative_tool_angle_deg
from table_touch_calibrate import (
    ContactEvidence, MarkerEvidenceTracker, TOUCH_X_M, TouchStep,
    _evidence_confirmed, backlash_prepose, fixed_pitch_path,
    median_table_flow, replay_contact_records, replay_touch_file,
    run_touch_trial)
from wrist_search import PlanarSearchSafety


HOVER = [90, 124, 90, 180, 90, 170]

# Exact records saved by the 2026-07-27 x=330, min-z=-18 physical run.
PACKET4_X330_ROWS = [
    (38.0, 36.636915370546305, [111, 100, 153], 40.40512466430664, 458, 2.3356169766979504),
    (33.99999999999999, 35.92900067662397, [111, 101, 154], 36.65341567993164, 224, 5.206452819227016),
    (31.999999999999993, 31.1897606582386, [112, 100, 154], 15.464066505432129, 537, 4.969906081554333),
    (27.99999999999999, 26.997932383194506, [113, 98, 154], 44.999961853027344, 491, 0.48204046673691037),
    (25.99999999999999, 26.099924562651033, [114, 97, 154], 12.92220401763916, 494, 0.7507349358341595),
    (23.999999999999986, 24.659011945118422, [115, 97, 154], 69.48113250732422, 433, 6.949400995441196),
    (21.999999999999986, 22.317046852888073, [117, 96, 154], 52.54043960571289, 515, 7.299422925289587),
    (19.999999999999982, 19.64234431551097, [118, 98, 155], 75.08283996582031, 464, 4.307094507382382),
    (17.999999999999982, 17.825367023556527, [120, 96, 155], 52.43257522583008, 479, 7.053155310299574),
    (15.999999999999979, 16.384435302845667, [121, 96, 155], 10.70329761505127, 479, 7.852008897776173),
    (13.999999999999979, 14.031804815667876, [123, 95, 155], 49.70773696899414, 484, 0.41763313051481116),
    (11.999999999999979, 12.197393645513921, [125, 93, 155], 34.26490020751953, 531, 0.24696731861602955),
    (9.999999999999979, 10.756543317881533, [126, 93, 155], 52.40773010253906, 511, 0.5584385427730162),
    (7.99999999999998, 8.395714745833589, [128, 92, 155], 7.680413246154785, 558, 0.5926678392348571),
    (5.99999999999998, 6.272785013541865, [129, 93, 156], 53.51464080810547, 450, 0.7119244015280883),
    (3.999999999999979, 3.9041166821312254, [131, 92, 156], 17.195125579833984, 548, 1.1367490357078707),
    (1.9999999999999791, 1.5336829321843382, [133, 91, 156], 8.602005958557129, 538, 1.6146616452386897),
    (-2.0816681711721685e-14, 0.601568542215436, [134, 90, 156], 0.7869609594345093, 172, 1.907897668061771),
    (-2.000000000000021, -1.7737149848140732, [136, 89, 156], 44.73220443725586, 176, 4.020087652268031),
    (-4.000000000000021, -4.150369504515639, [138, 88, 156], 43.73705291748047, 183, 7.130199538487831),
    (-5.000000000000021, -5.089822065826935, [139, 87, 156], 12.851743698120117, 217, 7.839674515699842),
    (-6.000000000000021, -6.030183784420426, [140, 86, 156], 19.033138275146484, 265, 8.418150060649706),
    (-7.000000000000021, -7.470738482252493, [141, 86, 156], 27.07853889465332, 228, 9.715869176713129),
    (-8.000000000000021, -8.414111435660399, [142, 85, 156], 23.565011978149414, 206, 11.123920323215588),
    (-9.000000000000021, -9.358334813844793, [143, 84, 156], 24.306453704833984, 230, 13.27746953374287),
    (-10.000000000000023, -9.852629350992432, [143, 85, 156], 9.281723022460938, 266, 13.205111196833734),
    (-11.000000000000023, -11.015049012634337, [143, 87, 157], 12.144193649291992, 291, 15.111783038250156),
    (-12.000000000000025, -11.960783369214688, [144, 86, 157], 17.04318618774414, 257, 16.09099128703909),
    (-13.000000000000025, -13.400035913769232, [145, 86, 157], 11.640069961547852, 223, 17.223508843347798),
    (-14.000000000000027, -14.348630615748023, [146, 85, 157], 11.001470565795898, 519, 17.53520855467083),
    (-15.000000000000027, -15.29799601922162, [147, 84, 157], 1.7141190767288208, 448, 17.535343626450683),
    (-16.00000000000003, -16.248112554524948, [148, 83, 157], 17.715232849121094, 554, 17.336119970712637),
    (-17.00000000000003, -17.689148421238894, [149, 83, 157], 9.121460914611816, 513, 18.917603518281858),
    (-18.00000000000003, -18.641968746862165, [150, 82, 157], 21.939687728881836, 573, None),
]
PACKET4_X330_RECORDS = [
    TouchStep(z, fk, pose, flow, points, marker)
    for z, fk, pose, flow, points, marker in PACKET4_X330_ROWS
]


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

    def test_packet4_exact_replay_detects_contact_and_strain_stop(self):
        result = replay_contact_records(PACKET4_X330_RECORDS)
        self.assertEqual(result["state"], "contact")
        self.assertEqual(result["evidence_kind"], "marker-safety")
        self.assertGreaterEqual(result["onset_z_mm"], -10.0)
        self.assertLessEqual(result["onset_z_mm"], -6.0)
        self.assertGreaterEqual(result["would_stop_z_mm"], -13.0)
        self.assertEqual(result["would_stop_z_mm"], -9.0)
        self.assertEqual(result["z_table_mm"], -7.0)
        self.assertEqual(len(result["records"]), len(PACKET4_X330_ROWS))

    def test_replay_file_writes_confirmed_calibration(self):
        payload = {
            "state": "no-contact",
            "minimum_z_mm": -18.0,
            "records": [
                {
                    "command_z_mm": row.command_z_mm,
                    "fk_z_mm": row.fk_z_mm,
                    "pose234": row.pose234,
                    "flow_px": row.flow_px,
                    "flow_points": row.flow_points,
                    "marker_shift_px": row.marker_shift_px,
                }
                for row in PACKET4_X330_RECORDS
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "physical.json"
            output = Path(directory) / "table_touch.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            result = replay_touch_file(source, output)
            saved = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(result["state"], "contact")
        self.assertEqual(saved["state"], "contact")
        self.assertEqual(saved["onset_z_mm"], -8.0)

    def test_missing_marker_samples_do_not_disarm_sustained_onset(self):
        tracker = MarkerEvidenceTracker()
        records = [
            TouchStep(-6.0, -6.0, [1, 2, 3], 5.0, 200, 1.0),
            TouchStep(-7.0, -7.0, [1, 2, 3], 5.0, 200, None),
            TouchStep(-8.0, -8.0, [1, 2, 3], 5.0, 200, 5.0),
            TouchStep(-9.0, -9.0, [1, 2, 3], 5.0, 200, None),
            TouchStep(-10.0, -10.0, [1, 2, 3], 5.0, 200, 4.8),
        ]
        evidence = None
        for index, record in enumerate(records):
            evidence = tracker.observe(record, index) or evidence
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.kind, "marker-sustained")
        self.assertEqual(evidence.onset_path_index, 2)
        self.assertEqual(evidence.confirmation_path_index, 4)

    def test_missing_second_strain_sample_stops_at_onset_plus_4(self):
        tracker = MarkerEvidenceTracker()
        records = [
            TouchStep(-8.0, -8.0, [1, 2, 3], 5.0, 200, 11.0),
            TouchStep(-9.0, -9.0, [1, 2, 3], 5.0, 200, None),
            TouchStep(-10.0, -10.0, [1, 2, 3], 5.0, 200, None),
            TouchStep(-11.0, -11.0, [1, 2, 3], 5.0, 200, None),
            TouchStep(-12.0, -12.0, [1, 2, 3], 5.0, 200, None),
        ]
        guard = None
        for index, record in enumerate(records):
            tracker.observe(record, index)
            guard = tracker.press_depth_guard(record, index) or guard
        self.assertIsNotNone(guard)
        self.assertEqual(guard.kind, "marker-guard")
        self.assertEqual(records[guard.confirmation_path_index].command_z_mm,
                         -12.0)

    def test_confirmation_cannot_switch_evidence_family(self):
        marker_trigger = ContactEvidence(
            "marker-sustained", 0, 2, 1)
        flow_only_replay = [
            TouchStep(-7.0, -7.0, [1, 2, 3], 0.5, 200, 0.0),
            TouchStep(-8.0, -8.0, [1, 2, 3], 0.5, 200, 1.0),
            TouchStep(-9.0, -9.0, [1, 2, 3], 0.5, 200, 1.0),
        ]
        self.assertFalse(
            _evidence_confirmed(marker_trigger, flow_only_replay, 10.0))

        flow_trigger = ContactEvidence("flow-collapse", 0, 0, 0)
        marker_only_replay = [
            TouchStep(-7.0, -7.0, [1, 2, 3], 9.0, 200, 0.0),
            TouchStep(-8.0, -8.0, [1, 2, 3], 9.0, 200, 11.0),
            TouchStep(-9.0, -9.0, [1, 2, 3], 9.0, 200, 13.0),
        ]
        self.assertFalse(
            _evidence_confirmed(flow_trigger, marker_only_replay, 10.0))

    def test_confirm_only_window_is_z_plus_10_through_z_minus_2(self):
        path = fixed_pitch_path(
            x_m=0.330, start_z_m=0.003, minimum_z_m=-0.009,
            step_z_m=0.001)
        commands = [round(z * 1000.0) for z, _pose in path]
        self.assertEqual(commands, list(range(3, -10, -1)))

    def test_backlash_prepose_finishes_from_below(self):
        target = [90, 120, 100, 160, 90, 170]
        prepose = backlash_prepose(target)
        self.assertEqual(prepose[1:4], [115, 95, 155])

    def test_legacy_touch_path_is_blocked_by_physical_finger_geometry(self):
        with self.assertRaisesRegex(RuntimeError, "unsafe prepose"):
            run_touch_trial(None, execute=False)

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
        """The old virtual-tool path cannot bypass the new finger endpoint."""
        path = fixed_pitch_path(x_m=0.330, minimum_z_m=-0.002)
        count = len(path) - 1
        points = [262] * count
        points[-2:] = [75, 105]
        flows = [(4.0 + 0.2 * (index % 3), points[index])
                 for index in range(count)]

        with self.assertRaisesRegex(RuntimeError, "unsafe prepose"):
            self._run_mock(
                path, flows, signatures=None, minimum_z_m=-0.002)

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

        with self.assertRaisesRegex(RuntimeError, "unsafe prepose"):
            self._run_mock(
                path, flows, signatures, minimum_z_m=-0.018)


if __name__ == "__main__":
    unittest.main()
