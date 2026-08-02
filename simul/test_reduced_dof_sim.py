"""Regression tests for the separate rigid-wrist simulation and policy."""

import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "laptop"))

import config  # noqa: E402
from reduced_dof import (  # noqa: E402
    ReducedDofSafety, command_pose, geometry, reduced_home,
)
from reduced_policy_adapter import build_reduced_observation  # noqa: E402
from simul.reduced_dof_robot import (  # noqa: E402
    ACTIVE_ACTUATORS, load_reduced_model, model_summary, set_reduced_pose,
    site_position,
)
from simul.reduced_dof_task_env import (  # noqa: E402
    GRASP_POSES, OBSERVATION_NAMES, ReducedFloorPickEnv,
)
from simul.reduced_dof_task_policy import (  # noqa: E402
    DEFAULT_REDUCED_MODEL, ReducedTaskPolicyRunner,
)


class ReducedSimulationTests(unittest.TestCase):
    def test_model_has_no_dead_axis_joint_or_actuator(self):
        model, _data = load_reduced_model()
        summary = model_summary(model)
        self.assertEqual(summary["nu"], 4)
        self.assertEqual(tuple(summary["actuators"]), ACTIVE_ACTUATORS)
        self.assertNotIn("wrist_pitch", summary["joints"])
        self.assertNotIn("wrist_roll", summary["joints"])
        self.assertNotIn("base", summary["joints"])

    def test_vent_housing_is_on_same_side_as_arm(self):
        model, _data = load_reduced_model()
        geom = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "base_collision")
        self.assertGreater(model.geom_pos[geom, 0], 0.0)
        self.assertAlmostEqual(model.geom_pos[geom, 0], 0.1025, places=6)
        self.assertAlmostEqual(model.geom_size[geom, 0], 0.1945, places=6)

    def test_incident_pose_contacts_forward_housing_in_mujoco(self):
        model, data = load_reduced_model()
        set_reduced_pose(model, data, command_pose(95, 165, 170))
        names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            for contact in data.contact
            for geom in (int(contact.geom1), int(contact.geom2))
        }
        self.assertIn("base_collision", names)
        self.assertTrue(
            {"wrist_collision", "palm_collision"}.intersection(names))

    def test_all_training_grasps_keep_post_margin_housing_clearance(self):
        safety = ReducedDofSafety()
        home = reduced_home(config.GRIP_CLOSED)
        lift = command_pose(80, 90, config.GRIP_CLOSED)
        for shoulder, elbow in GRASP_POSES:
            with self.subTest(shoulder=shoulder, elbow=elbow):
                grasp = command_pose(
                    shoulder, elbow, config.GRIP_CLOSED)
                approach = safety.transition_report(home, grasp)
                retreat = safety.transition_report(grasp, lift)
                self.assertTrue(approach.safe, approach.explain())
                self.assertTrue(retreat.safe, retreat.explain())
                self.assertGreaterEqual(
                    min(approach.minimum_clearance_mm,
                        retreat.minimum_clearance_mm),
                    10.0,
                )

    def test_sim_sites_match_real_reduced_fk(self):
        model, data = load_reduced_model()
        for shoulder, elbow in ((70, 90), (110, 110), (130, 65), (145, 40)):
            pose = command_pose(shoulder, elbow)
            set_reduced_pose(model, data, pose)
            expected = geometry(pose)
            np.testing.assert_allclose(
                site_position(model, data, "tool_center"), expected.tool,
                atol=1e-9)
            np.testing.assert_allclose(
                site_position(model, data, "finger_tip"), expected.finger_tip,
                atol=1e-9)

    def test_only_active_commands_change_simulated_state(self):
        model, data = load_reduced_model()
        first = command_pose(90, 100, config.GRIP_OPEN)
        second = command_pose(120, 70, config.GRIP_CLOSED)
        set_reduced_pose(model, data, first)
        before = data.qpos.copy()
        set_reduced_pose(model, data, second)
        self.assertFalse(np.allclose(before[:4], data.qpos[:4]))
        with self.assertRaises(ValueError):
            invalid = list(second)
            invalid[config.J_WRIST] += 1
            set_reduced_pose(model, data, invalid)

    def test_expert_completes_randomized_tasks(self):
        for seed in range(100):
            env = ReducedFloorPickEnv(domain_randomization=True, seed=seed)
            observation, _ = env.reset(seed=seed)
            for _ in range(env.max_steps):
                observation, _, terminated, truncated, info = env.step(
                    env.expert_action())
                if terminated or truncated:
                    break
            self.assertTrue(terminated, (seed, info))
            self.assertTrue(info["holding"])
            self.assertEqual(observation.shape, (len(OBSERVATION_NAMES),))

    def test_real_adapter_matches_training_schema(self):
        observation = build_reduced_observation(
            pose=command_pose(110, 110), target_center=(650, 500),
            gripper_center=(640, 510), opening_px=260,
            sonar_distance_mm=80, phase="approach")
        self.assertEqual(observation.shape, (len(OBSERVATION_NAMES),))
        self.assertTrue(np.isfinite(observation).all())

    @unittest.skipUnless(DEFAULT_REDUCED_MODEL.is_file(), "trained artifact not built")
    def test_trained_runner_completes_held_out_tasks(self):
        runner = ReducedTaskPolicyRunner(DEFAULT_REDUCED_MODEL)
        successes = 0
        for seed in range(100, 200):
            env = ReducedFloorPickEnv(domain_randomization=True, seed=seed)
            observation, _ = env.reset(seed=seed)
            runner.reset()
            for _ in range(env.max_steps):
                action, _ = runner.predict(observation)
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            successes += int(terminated and info["holding"])
        self.assertGreaterEqual(successes, 95)


if __name__ == "__main__":
    unittest.main(verbosity=2)
