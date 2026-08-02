import unittest
from pathlib import Path

import numpy as np

import config
from reduced_dof import (
    ACTIVE_MOTION_JOINTS,
    FIXED_BASE_COMMAND_DEG,
    FIXED_ROLL_COMMAND_DEG,
    FIXED_WRIST_COMMAND_DEG,
    FIXED_WRIST_GEOMETRY_DEG,
    ReducedDofSafety,
    camera_jacobian,
    command_pose,
    find_lift_pose,
    geometry_pose,
    reduced_home,
    resolved_step,
    safe_route,
    search_poses,
    validate_command_pose,
)
from reduced_dof_firmware import SKETCH_DIR
from reduced_dof_visual_servo import close_lift_home


ROOT = Path(__file__).resolve().parents[1]


class ReducedDofTests(unittest.TestCase):
    def test_command_exposes_only_shoulder_elbow_and_gripper(self):
        pose = command_pose(104, 83, 180)

        self.assertEqual(pose[config.J_BASE], FIXED_BASE_COMMAND_DEG)
        self.assertEqual(pose[config.J_SHOULDER], 104)
        self.assertEqual(pose[config.J_ELBOW], 83)
        self.assertEqual(pose[config.J_WRIST], FIXED_WRIST_COMMAND_DEG)
        self.assertEqual(pose[config.J_GRIP], 180)
        self.assertEqual(pose[config.J_ROLL], FIXED_ROLL_COMMAND_DEG)

    def test_disabled_joint_command_is_rejected(self):
        for joint in (config.J_BASE, config.J_WRIST, config.J_ROLL):
            with self.subTest(joint=joint + 1):
                pose = reduced_home()
                pose[joint] -= 1
                with self.assertRaisesRegex(ValueError, "고정된"):
                    validate_command_pose(pose)

    def test_rigid_wrist_geometry_is_independent_of_protocol_placeholder(self):
        pose = command_pose(95, 90)

        physical = geometry_pose(pose)

        self.assertEqual(physical[config.J_WRIST], FIXED_WRIST_GEOMETRY_DEG)
        self.assertNotEqual(
            physical[config.J_WRIST], pose[config.J_WRIST])

    def test_camera_jacobian_has_exactly_two_motion_columns(self):
        jacobian = camera_jacobian(command_pose(95, 90))

        self.assertEqual(jacobian.shape, (2, 2))
        self.assertTrue(np.isfinite(jacobian).all())
        self.assertEqual(np.linalg.matrix_rank(jacobian), 2)

    def test_resolved_step_never_changes_failed_axes_or_gripper(self):
        start = command_pose(95, 90, 90)
        step = resolved_step(
            start,
            vertical_error_px=-40,
            distance_mm=160.0,
            stop_range_mm=46.0,
            floor_stop_mm=10.0,
        )

        self.assertIsNotNone(step)
        changed = {
            index for index, (before, after) in enumerate(
                zip(start, step.pose)) if before != after
        }
        self.assertTrue(changed)
        self.assertLessEqual(changed, set(ACTIVE_MOTION_JOINTS))
        self.assertEqual(step.pose[config.J_GRIP], start[config.J_GRIP])
        self.assertTrue(ReducedDofSafety().transition_report(
            start, step.pose).safe)

    def test_search_and_home_keep_all_disabled_fields_fixed(self):
        for pose in [reduced_home(), *search_poses()]:
            self.assertEqual(pose[config.J_BASE], FIXED_BASE_COMMAND_DEG)
            self.assertEqual(pose[config.J_WRIST], FIXED_WRIST_COMMAND_DEG)
            self.assertEqual(pose[config.J_ROLL], FIXED_ROLL_COMMAND_DEG)

    def test_known_near_floor_pose_can_lift_and_route_home(self):
        safety = ReducedDofSafety()
        start = command_pose(113, 92, config.GRIP_CLOSED)

        lift = find_lift_pose(start, safety=safety)
        route = safe_route(
            lift, reduced_home(config.GRIP_CLOSED), safety=safety)

        self.assertEqual(lift[config.J_GRIP], config.GRIP_CLOSED)
        self.assertTrue(safety.transition_report(start, lift).safe)
        self.assertGreaterEqual(len(route), 1)
        self.assertEqual(route[-1], reduced_home(config.GRIP_CLOSED))
        for pose in [lift, *route]:
            self.assertEqual(pose[config.J_WRIST], FIXED_WRIST_COMMAND_DEG)

    def test_vent_housing_incident_pose_is_permanently_rejected(self):
        """Regression for the photographed 2026-08-02 motor-damage incident."""
        safety = ReducedDofSafety()
        home = reduced_home(170)
        incident = command_pose(95, 165, 170)

        pose_report = safety.pose_report(incident)
        transition_report = safety.transition_report(home, incident)

        self.assertFalse(pose_report.safe)
        self.assertIn("base-housing", pose_report.explain())
        self.assertFalse(transition_report.safe)
        self.assertIn("base-housing", transition_report.explain())

    def test_close_lift_home_keeps_full_gripper_clamp(self):
        class RecordingConnection:
            def __init__(self):
                self.poses = []

            def request(self, payload):
                self.poses.append(list(payload["pose"]))
                return {"ok": True, "pose": list(payload["pose"])}

        connection = RecordingConnection()
        start = command_pose(113, 92, config.GRIP_OPEN)

        result = close_lift_home(connection, ReducedDofSafety(), start)

        self.assertGreaterEqual(len(connection.poses), 3)
        self.assertTrue(all(
            pose[config.J_GRIP] == config.GRIP_CLOSED
            for pose in connection.poses
        ))
        self.assertEqual(result, reduced_home(config.GRIP_CLOSED))

    def test_reduced_firmware_attaches_only_2_3_and_5(self):
        source = (SKETCH_DIR / "arm_controller_reduced.ino").read_text(
            encoding="utf-8")

        self.assertEqual(SKETCH_DIR, ROOT / "firmware" / "arm_controller_reduced")
        self.assertIn(
            "ACTIVE_LOGICAL[ACTIVE_COUNT] = {1, 2, 4}", source)
        self.assertIn("ACTIVE_PINS[ACTIVE_COUNT] = {12, 11, 9}", source)
        self.assertNotIn("Servo servos[", source)
        self.assertNotIn("attach(13", source)
        self.assertNotIn("attach(10", source)
        self.assertNotIn("attach(8", source)


if __name__ == "__main__":
    unittest.main()
