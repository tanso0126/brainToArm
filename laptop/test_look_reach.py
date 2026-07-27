"""Pure regression tests for the eye-in-hand resolved-rate controller."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import arm_fk
import config
from look_reach import (TargetLock, cumulative_tool_angle_deg,
                        constant_x_descent_waypoints,
                        load_table_z_m,
                        match_locked_target, optical_axis_xz,
                        plan_aim_step, plan_resolved_step,
                        select_initial_target, task_delta)


def candidate(center, bbox, area, confidence):
    return SimpleNamespace(center=center, bbox=bbox, area=area,
                           confidence=confidence)


class LookReachTests(unittest.TestCase):
    def test_resolved_step_advances_and_aims(self):
        pose = [90, 110, 87, 140, 90, 170]
        plan = plan_resolved_step(pose, -100, 720)
        self.assertIsNotNone(plan)
        delta = task_delta(pose, plan["pose"])
        self.assertGreater(np.dot(delta[:2], optical_axis_xz(pose)), 0.0)
        self.assertGreater(cumulative_tool_angle_deg(plan["pose"]),
                        cumulative_tool_angle_deg(pose))
        self.assertGreaterEqual(arm_fk.tool_position(plan["pose"])[2], 0.026)
        self.assertTrue(all(abs(plan["pose"][j] - pose[j]) <= 3
                            for j in (config.J_SHOULDER,
                                      config.J_ELBOW, config.J_WRIST)))

    def test_centered_aim_keeps_pitch_while_advancing(self):
        pose = [90, 110, 87, 150, 90, 170]
        plan = plan_resolved_step(pose, 0, 720)
        self.assertGreater(plan["progress_mm"], 0.4)
        self.assertLess(abs(cumulative_tool_angle_deg(plan["pose"])
                            - cumulative_tool_angle_deg(pose)), 1.1)
        self.assertLess(task_delta(pose, plan["pose"])[1], 0.0)

    def test_wrist_look_rotation_is_not_fake_camera_translation(self):
        pose = [90, 110, 92, 150, 90, 170]
        rotated = list(pose)
        rotated[config.J_WRIST] -= 3
        delta = task_delta(pose, rotated)
        self.assertLess(np.linalg.norm(delta[:2]), 1e-9)
        self.assertGreater(delta[2], 4.0)

    def test_measured_aim_probe_moves_target_down(self):
        pose = [90, 110, 92, 150, 90, 170]
        plan = plan_aim_step(pose, -140)
        self.assertEqual(plan["pose"][config.J_WRIST], 147)
        self.assertEqual(plan["pose"][config.J_SHOULDER], pose[config.J_SHOULDER])
        self.assertEqual(plan["pose"][config.J_ELBOW], pose[config.J_ELBOW])

    def test_constant_x_descent_reaches_floor_without_horizontal_drift(self):
        start = [90, 110, 87, 150, 90, 170]
        waypoints = constant_x_descent_waypoints(start)
        start_tool = arm_fk.tool_position(start)
        heights = []
        for index, pose in enumerate(waypoints, 1):
            tool = arm_fk.tool_position(pose)
            expected_x = start_tool[0] - index / len(waypoints) * 0.005
            self.assertLess(abs(tool[0] - expected_x), 0.0061)
            heights.append(tool[2])
        self.assertTrue(all(a >= b - 0.001 for a, b in zip(heights, heights[1:])))
        self.assertLess(abs(heights[-1] - 0.008), 0.0031)
        self.assertLess(arm_fk.tool_position(waypoints[-1])[0],
                        start_tool[0] - 0.003)

    def test_table_touch_height_offsets_contact_endpoint(self):
        start = [90, 110, 87, 150, 90, 170]
        waypoints = constant_x_descent_waypoints(start, contact_z_m=0.011)
        self.assertAlmostEqual(
            float(arm_fk.tool_position(waypoints[-1])[2]), 0.011,
            delta=0.0031)

    def test_repeat_confirmed_table_height_loads_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table_touch.json"
            path.write_text(json.dumps({
                "state": "contact", "z_table_mm": 2.5,
            }), encoding="utf-8")
            self.assertAlmostEqual(load_table_z_m(path), 0.0025)
            path.write_text(json.dumps({
                "state": "planned", "z_table_mm": 2.5,
            }), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "confirmed contact"):
                load_table_z_m(path)

    def test_selection_prefers_real_central_object_over_jaw(self):
        actual = candidate((607, 252), (560, 100, 93, 304), 28272, 0.51)
        jaw = candidate((625, 690), (524, 661, 201, 59), 11859, 0.40)
        distractor = candidate((853, 85), (825, 51, 56, 69), 3864, 0.77)
        scene = SimpleNamespace(ranked=[jaw, actual, distractor],
                                frame_shape=(720, 1280, 3))
        self.assertIs(select_initial_target(scene), actual)

    def test_lock_rejects_bottom_identity_switch(self):
        original = candidate((607, 252), (560, 100, 93, 304), 28272, 0.51)
        moved = candidate((612, 280), (558, 100, 108, 360), 38880, 0.48)
        bottom_false = candidate((625, 690), (524, 661, 201, 59), 11859, 0.80)
        lock = TargetLock.from_candidate(original)
        scene = SimpleNamespace(ranked=[bottom_false, moved])
        self.assertIs(match_locked_target(scene, lock), moved)


if __name__ == "__main__":
    unittest.main()
