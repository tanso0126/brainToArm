"""Run the learned/shielded full task against MuJoCo contact dynamics.

The symbolic training environment decides from camera-derived observations.
This evaluator independently requires the simulated object body to be pinched
by the two jaws and physically rise with the arm. A symbolic HOLD alone cannot
pass this evaluation.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json

import cv2
import mujoco
import numpy as np

try:
    from .full_task_env import FullFloorPickEnv, TaskAction
    from .full_task_policy import FullTaskPolicyRunner
    from .mujoco_robot import (
        command_servo_pose, load_model, place_target, set_servo_pose, site_position)
except ImportError:
    from full_task_env import FullFloorPickEnv, TaskAction
    from full_task_policy import FullTaskPolicyRunner
    from mujoco_robot import (
        command_servo_pose, load_model, place_target, set_servo_pose, site_position)

try:
    from laptop import config
    from laptop.floor_motion import floor_pose
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laptop"))
    import config
    from floor_motion import floor_pose


HERE = Path(__file__).resolve().parent


@dataclass
class PhysicsResult:
    success: bool
    symbolic_success: bool
    target_elbow: int
    target_size_m: float
    target_shape: str
    initial_z_m: float
    final_z_m: float
    lift_m: float
    bottom_clearance_m: float
    floor_contact: bool
    follows_tool: bool
    attempts: int
    steps: int
    trace: list[str]


class PhysicsTaskEvaluator:
    def __init__(self, policy_path, *, seed=0):
        self.runner = FullTaskPolicyRunner(policy_path)
        self.model, self.data, self.spec = load_model()
        self.env = FullFloorPickEnv(domain_randomization=True, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.target_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.target_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        self.floor_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # Robot-relative floor lookup is the simulator equivalent of the real
        # controller re-detecting an object after a failed lift.  It is used by
        # recovery/reset code only and is never included in the policy input.
        lookup_model, lookup_data, lookup_spec = load_model()
        self.floor_x_by_elbow = {}
        for elbow in range(config.FLOOR_ELBOW_RANGE[0],
                           config.FLOOR_ELBOW_RANGE[1] + 1):
            set_servo_pose(
                lookup_model, lookup_data, floor_pose(elbow, "grasp"),
                spec=lookup_spec)
            self.floor_x_by_elbow[elbow] = float(site_position(
                lookup_model, lookup_data, "tool_center")[0])

    def close(self):
        self.env.close()

    def _set_target_shape(self, shape, size):
        kinds = {
            "box": mujoco.mjtGeom.mjGEOM_BOX,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
            "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
        }
        self.model.geom_type[self.target_geom] = kinds[shape]
        self.model.geom_size[self.target_geom, :3] = 0
        if shape == "box":
            self.model.geom_size[self.target_geom, :3] = (size, size, size * 0.9)
        elif shape == "cylinder":
            self.model.geom_size[self.target_geom, :2] = (size, size * 0.9)
        else:
            self.model.geom_size[self.target_geom, 0] = size

    def _drive(self, pose, seconds):
        command_servo_pose(self.model, self.data, pose, spec=self.spec)
        for _ in range(max(1, int(round(seconds / self.model.opt.timestep)))):
            mujoco.mj_step(self.model, self.data)

    def _drive_transition(self, start_pose, end_pose, seconds, max_step_deg=3.0):
        """Follow bounded servo waypoints like the real Uno, without teleporting."""

        start = np.asarray(start_pose, dtype=np.float64)
        end = np.asarray(end_pose, dtype=np.float64)
        segments = max(1, int(np.ceil(np.max(np.abs(end - start)) / max_step_deg)))
        for fraction in np.linspace(1.0 / segments, 1.0, segments):
            pose = start + fraction * (end - start)
            self._drive(pose, seconds / segments)

    def _target_touches_floor(self):
        """Use MuJoCo contacts, not an arbitrary center-height threshold."""

        pair = {self.floor_geom, self.target_geom}
        return any(
            {int(contact.geom1), int(contact.geom2)} == pair
            for contact in self.data.contact
        )

    @staticmethod
    def _target_half_height(shape, size):
        return size if shape == "sphere" else size * 0.9

    def _nearest_floor_elbow(self):
        target_x = float(self.data.xpos[self.target_body, 0])
        return min(
            self.floor_x_by_elbow,
            key=lambda elbow: abs(self.floor_x_by_elbow[elbow] - target_x))

    def run_episode(self, seed, save_frames=False, max_attempts=3):
        rng = np.random.default_rng(seed)
        target_elbow = int(rng.integers(*(
            config.FLOOR_ELBOW_RANGE[0], config.FLOOR_ELBOW_RANGE[1] + 1)))
        centerline = float(rng.uniform(-70.0, 70.0))
        size = float(rng.uniform(0.014, 0.022))
        shape = str(rng.choice(("box", "cylinder", "sphere")))
        target_y = centerline / 70.0 * 0.0045

        mujoco.mj_resetData(self.model, self.data)
        self._set_target_shape(shape, size)
        set_servo_pose(
            self.model, self.data,
            floor_pose(target_elbow, "grasp", gripper=config.GRIP_OPEN),
            spec=self.spec)
        target_x = float(site_position(
            self.model, self.data, "tool_center")[0])
        set_servo_pose(
            self.model, self.data, self.spec.home_servo_deg, spec=self.spec)
        object_z = self._target_half_height(shape, size)
        place_target(self.model, self.data, (target_x, target_y, object_z))
        self._drive(self.spec.home_servo_deg, 0.25)
        physical_pose = np.asarray(self.spec.home_servo_deg, dtype=np.float64)
        initial_z = float(self.data.xpos[self.target_body, 2])

        trace = []
        frames = []
        symbolic_success = False
        success = False
        total_steps = 0
        attempts = 0
        durations = {
            TaskAction.SEARCH_NEXT: 0.80,
            TaskAction.ALIGN_ELBOW_DOWN: 0.60,
            TaskAction.ALIGN_ELBOW_UP: 0.60,
            TaskAction.DESCEND: 1.50,
            TaskAction.CLOSE: 1.50,
            TaskAction.LIFT: 1.90,
            TaskAction.RECOVER: 1.20,
        }
        for attempt in range(1, max_attempts + 1):
            attempts = attempt
            if attempt > 1:
                # Camera-verified lift failure: open at hover, let the object
                # settle wherever physics left it, then reacquire on the floor.
                recovery_elbow = self._nearest_floor_elbow()
                recovery_pose = np.asarray(floor_pose(
                    recovery_elbow, "hover", gripper=config.GRIP_OPEN),
                    dtype=np.float64)
                self._drive_transition(physical_pose, recovery_pose, 1.5)
                self._drive(recovery_pose, 0.5)
                physical_pose = recovery_pose
                target_elbow = self._nearest_floor_elbow()
                target_y = float(self.data.xpos[self.target_body, 1])
                centerline = float(np.clip(target_y / 0.0045 * 70.0, -110, 110))
                options = {
                    "target_elbow": target_elbow,
                    "current_elbow": recovery_elbow,
                    "centerline_error_px": centerline,
                    "pose_level": "hover",
                }
                trace.append(f"A{attempt}:REACQUIRE:e{target_elbow}")
            else:
                options = {
                    "target_elbow": target_elbow,
                    "centerline_error_px": centerline,
                }
            observation, _ = self.env.reset(
                seed=seed + 100_000 * (attempt - 1), options=options)
            self.runner.reset()
            symbolic_success = False
            for step in range(self.env.max_steps):
                action, _probabilities = self.runner.predict(
                    observation, apply_shield=True, temporal_guard=True)
                observation, _, terminated, truncated, info = self.env.step(action)
                total_steps += 1
                trace.append(
                    f"A{attempt}.{step + 1}:{action.name}:"
                    f"{info['event']}:e{info['current_elbow']}")
                if action in durations:
                    next_pose = np.asarray(
                        self.env.commanded_pose, dtype=np.float64)
                    self._drive_transition(
                        physical_pose, next_pose, durations[action])
                    physical_pose = next_pose
                if save_frames:
                    frames.append(self._render_overview(trace[-1]))
                if terminated or truncated:
                    symbolic_success = bool(terminated and info["holding"])
                    break

            final_z = float(self.data.xpos[self.target_body, 2])
            lift = final_z - initial_z
            tool = site_position(self.model, self.data, "tool_center")
            target = self.data.xpos[self.target_body]
            follows_tool = np.linalg.norm(target[:2] - tool[:2]) < 0.055
            floor_contact = self._target_touches_floor()
            bottom_clearance = final_z - self._target_half_height(shape, size)
            # A real pick means that contact physics has separated the object
            # from the floor and the object follows the hand.  This cannot be
            # satisfied by the symbolic task state or an object tipped in place.
            success = bool(
                symbolic_success
                and lift >= 0.003
                and bottom_clearance >= 0.002
                and not floor_contact
                and follows_tool
            )
            if success:
                break
        if save_frames and frames:
            self._write_sheet(frames, seed)
        return PhysicsResult(
            success, symbolic_success, target_elbow, size, shape,
            initial_z, final_z, lift, bottom_clearance, floor_contact,
            bool(follows_tool), attempts, total_steps, trace)

    def _render_overview(self, label):
        renderer = mujoco.Renderer(self.model, height=270, width=480)
        try:
            renderer.update_scene(self.data, camera="overview")
            image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        finally:
            renderer.close()
        cv2.rectangle(image, (0, 0), (479, 25), (8, 10, 14), -1)
        cv2.putText(image, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.44, (240, 245, 250), 1, cv2.LINE_AA)
        return image

    @staticmethod
    def _write_sheet(frames, seed):
        selected = frames[:8]
        while len(selected) < 8:
            selected.append(np.zeros_like(selected[0]))
        sheet = np.vstack((np.hstack(selected[:4]), np.hstack(selected[4:8])))
        output = HERE / "generated" / f"full_task_physics_{seed}.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output), sheet)


def evaluate(policy_path, episodes, seed):
    evaluator = PhysicsTaskEvaluator(policy_path, seed=seed)
    results = []
    try:
        for episode in range(episodes):
            results.append(evaluator.run_episode(
                seed + episode, save_frames=(episode == 0)))
    finally:
        evaluator.close()
    successes = sum(result.success for result in results)
    nominal = [result for result in results if result.target_size_m <= 0.020]
    edge_stress = [result for result in results if result.target_size_m > 0.020]

    def success_rate(group):
        return (sum(result.success for result in group) / len(group)
                if group else None)

    payload = {
        "episodes": episodes,
        "physics_successes": successes,
        "physics_success_rate": successes / episodes,
        "symbolic_success_rate": sum(r.symbolic_success for r in results) / episodes,
        "mean_physical_lift_m": float(np.mean([r.lift_m for r in results])),
        "single_attempt_success_rate": sum(
            r.success and r.attempts == 1 for r in results) / episodes,
        "recovered_after_retry": sum(
            r.success and r.attempts > 1 for r in results),
        "nominal_max_object_width_m": 0.040,
        "nominal_episodes": len(nominal),
        "nominal_physics_success_rate": success_rate(nominal),
        "edge_stress_episodes": len(edge_stress),
        "edge_stress_physics_success_rate": success_rate(edge_stress),
        "failures": [asdict(result) for result in results if not result.success][:20],
    }
    print(json.dumps(payload, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path,
                        default=HERE / "generated" / "full_task_policy_v1.ts")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()
    evaluate(args.policy, args.episodes, args.seed)


if __name__ == "__main__":
    main()
