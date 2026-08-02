"""Contact-physics evaluation of the trained rigid-wrist reduced policy."""

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "laptop") not in sys.path:
    sys.path.insert(0, str(ROOT / "laptop"))

from laptop.reduced_dof import reduced_home  # noqa: E402
from .reduced_dof_robot import (  # noqa: E402
    command_reduced_pose, load_reduced_model, place_target,
    set_reduced_pose, site_position,
)
from .reduced_dof_task_env import (  # noqa: E402
    ReducedFloorPickEnv, ReducedTaskAction,
)
from .reduced_dof_task_policy import (  # noqa: E402
    DEFAULT_REDUCED_MODEL, ReducedTaskPolicyRunner,
)


@dataclass(frozen=True)
class ReducedPhysicsResult:
    success: bool
    symbolic_success: bool
    shape: str
    half_size_m: float
    lift_m: float
    follows_hand: bool
    floor_contact: bool
    steps: int


class ReducedPhysicsEvaluator:
    def __init__(self, model_path=DEFAULT_REDUCED_MODEL):
        self.model, self.data = load_reduced_model()
        self.env = ReducedFloorPickEnv(domain_randomization=True)
        self.runner = ReducedTaskPolicyRunner(model_path)
        self.target_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.target_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        self.floor_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def _shape(self, shape, size):
        kinds = {"box": mujoco.mjtGeom.mjGEOM_BOX,
                 "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
                 "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}
        self.model.geom_type[self.target_geom] = kinds[shape]
        self.model.geom_size[self.target_geom, :3] = 0
        if shape == "box":
            self.model.geom_size[self.target_geom, :3] = (size, size, size)
        elif shape == "cylinder":
            self.model.geom_size[self.target_geom, :2] = (size, size)
        else:
            self.model.geom_size[self.target_geom, 0] = size

    def _drive(self, start, end, seconds=0.6):
        start, end = np.asarray(start, float), np.asarray(end, float)
        segments = max(1, int(np.ceil(np.max(np.abs(end - start)) / 3.0)))
        for fraction in np.linspace(1.0 / segments, 1.0, segments):
            command_reduced_pose(self.model, self.data,
                                 start + fraction * (end - start))
            ticks = max(1, int(seconds / segments / self.model.opt.timestep))
            for _ in range(ticks):
                mujoco.mj_step(self.model, self.data)

    def _floor_contact(self):
        pair = {self.floor_geom, self.target_geom}
        return any({int(c.geom1), int(c.geom2)} == pair for c in self.data.contact)

    def run_episode(self, seed):
        rng = np.random.default_rng(seed)
        observation, _ = self.env.reset(seed=seed)
        target_pose = list(self.env.target_pose)
        shape = str(rng.choice(("box", "cylinder", "sphere")))
        size = float(rng.uniform(0.012, 0.019))
        lateral = float(rng.uniform(-0.0045, 0.0045))

        mujoco.mj_resetData(self.model, self.data)
        self._shape(shape, size)
        set_reduced_pose(self.model, self.data, target_pose)
        # The real finger pads extend past the historical CAD ``tool_center``.
        # Floor objects are pinched at that distal row, not under the palm.
        grasp = site_position(self.model, self.data, "finger_tip")
        set_reduced_pose(self.model, self.data, reduced_home())
        place_target(self.model, self.data, (grasp[0], grasp[1] + lateral, size))
        current = reduced_home()
        initial_z = float(self.data.xpos[self.target_body, 2])
        self.runner.reset()
        symbolic_success = False
        steps = 0
        for steps in range(1, self.env.max_steps + 1):
            action, _ = self.runner.predict(observation)
            observation, _, terminated, truncated, info = self.env.step(action)
            next_pose = list(self.env.current_pose)
            if next_pose != current:
                duration = 1.0 if action in (
                    ReducedTaskAction.CLOSE, ReducedTaskAction.LIFT,
                    ReducedTaskAction.RETURN_HOME) else 0.35
                self._drive(current, next_pose, duration)
                current = next_pose
            if terminated or truncated:
                symbolic_success = bool(terminated and info["holding"])
                break
        final_z = float(self.data.xpos[self.target_body, 2])
        hand = site_position(self.model, self.data, "finger_tip")
        target = self.data.xpos[self.target_body]
        follows = bool(np.linalg.norm(target[:2] - hand[:2]) < 0.060)
        floor_contact = self._floor_contact()
        lift = final_z - initial_z
        success = bool(symbolic_success and lift >= 0.030 and follows and not floor_contact)
        return ReducedPhysicsResult(
            success, symbolic_success, shape, size, lift, follows,
            floor_contact, steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--model", type=Path, default=DEFAULT_REDUCED_MODEL)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "models" /
                        "reduced_dof_policy_v1.physics.json")
    args = parser.parse_args()
    evaluator = ReducedPhysicsEvaluator(args.model)
    results = [evaluator.run_episode(args.seed + i) for i in range(args.episodes)]
    payload = {
        "episodes": args.episodes,
        "contact_physics_success_rate": sum(r.success for r in results) / len(results),
        "symbolic_success_rate": sum(r.symbolic_success for r in results) / len(results),
        "mean_lift_m": float(np.mean([r.lift_m for r in results])),
        "failure_count": sum(not r.success for r in results),
        "by_shape": {
            shape: {
                "episodes": sum(r.shape == shape for r in results),
                "success_rate": (sum(r.success and r.shape == shape for r in results) /
                                 max(1, sum(r.shape == shape for r in results))),
            } for shape in ("box", "cylinder", "sphere")
        },
        "failures": [asdict(r) for r in results if not r.success][:30],
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
