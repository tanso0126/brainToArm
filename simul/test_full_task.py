"""Hardware-free regression tests for the complete sim-to-real policy."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "laptop"))

import config  # noqa: E402
from floor_motion import floor_pose  # noqa: E402
from full_task_adapter import (  # noqa: E402
    FullTaskShadowController, build_policy_observation)
from simul.evaluate_full_task_physics import PhysicsTaskEvaluator  # noqa: E402
from simul.full_task_env import (  # noqa: E402
    FullFloorPickEnv, OBSERVATION_NAMES, TaskAction)
from simul.full_task_policy import (  # noqa: E402
    DEFAULT_FULL_TASK_MODEL, FullTaskPolicyRunner)


class FullTaskTests(unittest.TestCase):
    def test_expert_completes_the_entire_deterministic_task(self):
        env = FullFloorPickEnv(domain_randomization=False)
        try:
            observation, _ = env.reset(seed=7, options={
                "target_elbow": 107, "centerline_error_px": 45})
            for _ in range(env.max_steps):
                observation, _, terminated, truncated, info = env.step(
                    env.expert_action())
                if terminated or truncated:
                    break
            self.assertTrue(terminated)
            self.assertTrue(info["holding"])
            self.assertEqual(len(observation), len(OBSERVATION_NAMES))
        finally:
            env.close()

    def test_release_runner_is_guarded_and_hardware_free(self):
        runner = FullTaskPolicyRunner(DEFAULT_FULL_TASK_MODEL)
        env = FullFloorPickEnv(domain_randomization=False)
        try:
            observation, _ = env.reset(seed=3, options={
                "target_elbow": 90, "current_elbow": 90,
                "pose_level": "hover"})
            first, _ = runner.predict(
                observation, apply_shield=True, temporal_guard=True)
            second, _ = runner.predict(
                observation, apply_shield=True, temporal_guard=True)
            self.assertEqual(first, TaskAction.WAIT)
            self.assertEqual(second, TaskAction.DESCEND)
        finally:
            env.close()

    def test_real_adapter_matches_the_training_schema(self):
        target = SimpleNamespace(center=(640.0, 500.0))
        gripper = SimpleNamespace(center=(640.0, 505.0), opening_px=280.0)
        scene = SimpleNamespace(
            gripper=gripper, frame_shape=(720, 1280, 3), ranked=[target])
        wrist = SimpleNamespace(quality=SimpleNamespace(valid=True))
        pose = floor_pose(90, "hover")
        observation = build_policy_observation(
            scene, wrist, pose, target=target)
        self.assertEqual(observation.shape, (len(OBSERVATION_NAMES),))
        self.assertTrue(np.isfinite(observation).all())

        controller = FullTaskShadowController(DEFAULT_FULL_TASK_MODEL)
        first = controller.decide(scene, wrist, pose, target=target)
        second = controller.decide(scene, wrist, pose, target=target)
        self.assertEqual(first.action, TaskAction.WAIT)
        self.assertEqual(second.action, TaskAction.DESCEND)
        self.assertEqual(
            second.next_pose,
            tuple(floor_pose(90, "grasp", gripper=config.GRIP_OPEN)))

    def test_locked_target_can_close_through_grasp_occlusion(self):
        """The eye-in-hand target vanishes between the fingers at contact."""
        gripper = SimpleNamespace(center=(640.0, 700.0), opening_px=280.0)
        scene = SimpleNamespace(
            gripper=gripper, frame_shape=(720, 1280, 3), ranked=[])
        wrist = SimpleNamespace(quality=SimpleNamespace(valid=True))
        pose = floor_pose(90, "grasp", gripper=config.GRIP_OPEN)
        controller = FullTaskShadowController(DEFAULT_FULL_TASK_MODEL)
        first = controller.decide(
            scene, wrist, pose, target=None, target_locked=True)
        second = controller.decide(
            scene, wrist, pose, target=None, target_locked=True)
        self.assertEqual(first.action, TaskAction.WAIT)
        self.assertEqual(second.action, TaskAction.CLOSE)

        markerless_scene = SimpleNamespace(
            gripper=None, frame_shape=(720, 1280, 3), ranked=[])

        # Real fixed-reach poses use wrist pitch !=180; shoulder 130 looks
        # closer to the legacy hover curve but the current FK puts the tool at
        # ~6 mm and must therefore authorize the same guarded close.
        real_grasp_pose = [90, 130, 90, 156, config.GRIP_OPEN, 170]
        controller.reset()
        first = controller.decide(
            markerless_scene, wrist, real_grasp_pose,
            target=None, target_locked=True)
        second = controller.decide(
            markerless_scene, wrist, real_grasp_pose,
            target=None, target_locked=True)
        self.assertEqual(first.action, TaskAction.WAIT)
        self.assertEqual(second.action, TaskAction.CLOSE)

        controller.reset()
        blocked = controller.decide(
            scene, wrist, pose, target=None, target_locked=False)
        self.assertNotEqual(blocked.action, TaskAction.CLOSE)

        # At the final open grasp pose the object can cover one tape as well as
        # its own mask. The pre-descent lock permits close, but only contact
        # verification (which requires both markers) may subsequently lift.
        controller.reset()
        first = controller.decide(
            markerless_scene, wrist, pose, target=None, target_locked=True)
        second = controller.decide(
            markerless_scene, wrist, pose, target=None, target_locked=True)
        self.assertEqual(first.action, TaskAction.WAIT)
        self.assertEqual(second.action, TaskAction.CLOSE)

    def test_contact_physics_not_symbolic_state_decides_success(self):
        evaluator = PhysicsTaskEvaluator(DEFAULT_FULL_TASK_MODEL, seed=20260729)
        try:
            results = [evaluator.run_episode(20260729 + index)
                       for index in range(30)]
        finally:
            evaluator.close()
        self.assertTrue(all(result.symbolic_success for result in results))
        self.assertGreaterEqual(
            sum(result.success for result in results) / len(results), 0.90)
        for result in results:
            if result.success:
                self.assertFalse(result.floor_contact)
                self.assertTrue(result.follows_tool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
