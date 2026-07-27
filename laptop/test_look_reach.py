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
                        LookReachTargetSelector,
                        choose_with_rejections,
                        constant_x_descent_waypoints,
                        load_table_z_m,
                        match_locked_target, optical_axis_xz,
                        plan_aim_step, plan_resolved_step,
                        run_controller, select_initial_target, task_delta)


def candidate(center, bbox, area, confidence):
    return SimpleNamespace(center=center, bbox=bbox, area=area,
                           confidence=confidence)


def multi_object_scene(candidates):
    gripper = SimpleNamespace(center=(640.0, 650.0), opening_px=300.0)
    return SimpleNamespace(ranked=list(candidates), gripper=gripper,
                           frame_shape=(720, 1280, 3))


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

    def test_two_candidates_choose_nearest_ranked_reachable_object(self):
        near = candidate((630, 450), (600, 410, 60, 80), 4800, 0.92)
        other = candidate((675, 350), (645, 310, 60, 80), 4800, 0.90)
        selection = LookReachTargetSelector(logger=None)
        selected = selection.choose(multi_object_scene([near, other]))
        self.assertIs(selected, near)
        np.testing.assert_allclose(selection.lock.center, near.center)

    def test_reject_one_selects_other_and_lock_follows_it(self):
        near = candidate((630, 450), (600, 410, 60, 80), 4800, 0.92)
        other = candidate((675, 350), (645, 310, 60, 80), 4800, 0.90)
        selection = LookReachTargetSelector(logger=None)
        scene = multi_object_scene([near, other])
        selected = choose_with_rejections(
            scene, selection, reject_count=1)
        self.assertIs(selected, other)
        np.testing.assert_allclose(selection.lock.center, other.center)

        # The eye-in-hand camera moved: both objects shifted by roughly +108 px
        # vertically. The selected target must still match, and its displacement
        # must carry the old veto into the new image coordinates.
        moved_other = candidate(
            (681, 458), (651, 418, 60, 80), 4800, 0.88)
        moved_near = candidate(
            (635, 555), (605, 515, 60, 80), 4800, 0.91)
        matched = selection.match(
            multi_object_scene([moved_near, moved_other]))
        self.assertIs(matched, moved_other)
        np.testing.assert_allclose(selection.lock.center, moved_other.center)
        self.assertIs(
            selection.selector.choose([moved_near, moved_other]),
            moved_other)

    def test_reject_all_returns_safe_stop_without_motion(self):
        near = candidate((630, 450), (600, 410, 60, 80), 4800, 0.92)
        other = candidate((675, 350), (645, 310, 60, 80), 4800, 0.90)
        scene = multi_object_scene([near, other])
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        class StaticDetector:
            @staticmethod
            def scene(_frame):
                return scene, SimpleNamespace(gripper=scene.gripper)

        class NoMotionClient:
            def __init__(self):
                self.requests = []

            def request(self, payload):
                self.requests.append(payload)
                if payload == {"command": "status"}:
                    return {"pose": [90, 124, 90, 180, 90, 170]}
                raise AssertionError("safe no-target stop must not request motion")

        client = NoMotionClient()
        result = run_controller(
            client, detector=StaticDetector(),
            frame_source=lambda discard=1: frame,
            target_selector=LookReachTargetSelector(logger=None),
            reject_count=2)
        self.assertEqual(result, {"state": "no-target", "moved": False})
        self.assertEqual(client.requests, [{"command": "status"}])

    def test_position_veto_survives_candidate_list_reshuffle(self):
        near = candidate((630, 450), (600, 410, 60, 80), 4800, 0.92)
        other = candidate((675, 350), (645, 310, 60, 80), 4800, 0.90)
        selection = LookReachTargetSelector(logger=None)
        first_scene = multi_object_scene([near, other])
        self.assertIs(selection.choose(first_scene), near)
        selection.reject_current()

        renumbered_near = candidate(
            (636, 455), (606, 415, 60, 80), 4800, 0.95)
        renumbered_other = candidate(
            (680, 355), (650, 315, 60, 80), 4800, 0.87)
        reshuffled = multi_object_scene([renumbered_near, renumbered_other])
        self.assertIs(selection.choose(reshuffled), renumbered_other)

    def test_unreachable_candidate_is_logged_and_not_vetoed(self):
        unreachable = candidate(
            (900, 430), (870, 390, 60, 80), 4800, 0.96)
        reachable = candidate(
            (630, 450), (600, 410, 60, 80), 4800, 0.90)
        messages = []
        selection = LookReachTargetSelector(logger=messages.append)
        selected = selection.choose(
            multi_object_scene([unreachable, reachable]))
        self.assertIs(selected, reachable)
        self.assertEqual(selection.selector.rejected_points, [])
        self.assertTrue(any("SKIP candidate #0" in item
                            and "base-yaw window" in item
                            for item in messages))


if __name__ == "__main__":
    unittest.main()
