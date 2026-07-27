"""Hardware-free regression tests for the physical-shape MuJoCo model."""

import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "laptop"))

from floor_motion import floor_pose  # noqa: E402
from simul.mujoco_robot import (  # noqa: E402
    load_model,
    model_summary,
    place_target_below_tool,
    render_rgb,
    servo_to_joint_targets,
    set_servo_pose,
    site_position,
)
from simul.alignment_env import WristAlignmentEnv  # noqa: E402


class MuJoCoRobotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data, cls.spec = load_model()

    def test_motor_one_is_physically_absent_and_locked(self):
        summary = model_summary(self.model)
        self.assertNotIn("base", summary["joints"])
        self.assertNotIn("base_yaw", summary["joints"])
        bad = self.spec.home_servo_deg.copy()
        bad[0] = 91
        with self.assertRaisesRegex(ValueError, "base is locked"):
            servo_to_joint_targets(bad, self.spec)

    def test_real_servo_limits_are_the_public_boundary(self):
        for index in range(6):
            pose = self.spec.home_servo_deg.copy()
            pose[0] = self.spec.base_locked_deg
            pose[index] = self.spec.servo_min_deg[index]
            if index == 0:
                pose[index] = self.spec.base_locked_deg
            servo_to_joint_targets(pose, self.spec)
            pose[index] = self.spec.servo_max_deg[index]
            if index == 0:
                pose[index] = self.spec.base_locked_deg
            servo_to_joint_targets(pose, self.spec)
        bad = self.spec.home_servo_deg.copy()
        bad[1] = self.spec.servo_max_deg[1] + 1
        with self.assertRaisesRegex(ValueError, "outside limits"):
            servo_to_joint_targets(bad, self.spec)

    def test_gripper_direction_matches_physical_arm(self):
        opened = self.spec.hover_servo_deg.copy()
        closed = opened.copy()
        opened[4] = 90
        closed[4] = 180
        open_target = servo_to_joint_targets(opened, self.spec)
        closed_target = servo_to_joint_targets(closed, self.spec)
        self.assertGreater(open_target["grip_left"], closed_target["grip_left"])
        self.assertAlmostEqual(open_target["grip_left"], open_target["grip_right"])

    def test_floor_curve_stays_level_and_moves_forward(self):
        for level, expected_z in (("hover", 0.041), ("grasp", 0.008)):
            points = []
            for elbow in (78, 90, 110):
                pose = floor_pose(elbow, level)
                set_servo_pose(self.model, self.data, pose, spec=self.spec)
                points.append(site_position(self.model, self.data, "tool_center"))
            z_values = [point[2] for point in points]
            x_values = [point[0] for point in points]
            self.assertLess(max(z_values) - min(z_values), 0.004)
            self.assertGreater(max(x_values) - min(x_values), 0.015)
            self.assertAlmostEqual(points[1][2], expected_z, delta=0.003)

    def test_wrist_render_contains_real_marker_order_and_target(self):
        set_servo_pose(
            self.model, self.data, self.spec.hover_servo_deg, spec=self.spec)
        place_target_below_tool(self.model, self.data)
        image = render_rgb(self.model, self.data, width=128, height=72)
        self.assertEqual(image.shape, (72, 128, 3))
        self.assertGreater(float(image.std()), 10)
        blue = ((image[:, :, 2] > 150) & (image[:, :, 0] < 100)
                & (image[:, :, 1] < 150))
        red = ((image[:, :, 0] > 150) & (image[:, :, 1] < 100)
               & (image[:, :, 2] < 100))
        yellow = ((image[:, :, 0] > 150) & (image[:, :, 1] > 100)
                  & (image[:, :, 2] < 100))
        self.assertGreater(int(blue.sum()), 30)
        self.assertGreater(int(red.sum()), 25)
        self.assertGreater(int(yellow.sum()), 30)
        self.assertLess(float(np.where(blue)[1].mean()),
                        float(np.where(red)[1].mean()))

    def test_dynamic_command_uses_six_actuators_without_hardware(self):
        self.assertEqual(self.model.nu, 6)
        self.assertEqual(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist"),
            1,
        )

    def test_alignment_environment_hides_privileged_state(self):
        env = WristAlignmentEnv(
            domain_randomization=False, image_augmentation=False, max_steps=20)
        try:
            observation, info = env.reset(seed=12, options={
                "current_elbow": 78, "target_elbow": 100, "target_y": 0})
            self.assertEqual(
                set(observation), {"image", "servo", "previous_action"})
            self.assertNotIn("target_elbow", observation)
            self.assertIn("target_elbow", info)
            for _ in range(20):
                observation, _, terminated, truncated, info = env.step(
                    env.expert_action())
                if terminated or truncated:
                    break
            self.assertTrue(info["success"])
            self.assertLessEqual(env.steps, 12)
        finally:
            env.close()

if __name__ == "__main__":
    unittest.main(verbosity=2)
