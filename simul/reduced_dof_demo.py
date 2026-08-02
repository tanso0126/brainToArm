"""Watch one trained reduced-arm pick and HOME return in MuJoCo."""

from pathlib import Path
import argparse
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "laptop") not in sys.path:
    sys.path.insert(0, str(ROOT / "laptop"))

from laptop.reduced_dof import reduced_home  # noqa: E402
from .reduced_dof_robot import (  # noqa: E402
    command_reduced_pose, load_reduced_model, place_target,
    set_reduced_pose, site_position,
)
from .reduced_dof_task_env import ReducedFloorPickEnv  # noqa: E402
from .reduced_dof_task_policy import (  # noqa: E402
    DEFAULT_REDUCED_MODEL, ReducedTaskPolicyRunner,
)


def run(seed=20260802, headless=False):
    model, data = load_reduced_model()
    env = ReducedFloorPickEnv(domain_randomization=False)
    runner = ReducedTaskPolicyRunner(DEFAULT_REDUCED_MODEL)
    observation, _ = env.reset(seed=seed)
    set_reduced_pose(model, data, env.target_pose)
    grasp = site_position(model, data, "finger_tip")
    set_reduced_pose(model, data, reduced_home())
    place_target(model, data, (grasp[0], grasp[1], 0.016))
    current = reduced_home()
    viewer = None if headless else mujoco.viewer.launch_passive(model, data)
    try:
        for _ in range(env.max_steps):
            action, probabilities = runner.predict(observation)
            observation, _, terminated, truncated, info = env.step(action)
            target = np.asarray(env.current_pose, float)
            start = np.asarray(current, float)
            segments = max(1, int(np.ceil(np.max(np.abs(target - start)) / 2.0)))
            for fraction in np.linspace(1.0 / segments, 1.0, segments):
                command_reduced_pose(model, data, start + fraction * (target - start))
                for _tick in range(4):
                    mujoco.mj_step(model, data)
                    if viewer is not None:
                        viewer.sync()
                        time.sleep(model.opt.timestep)
            current = target.tolist()
            print(f"{action.name:12s} confidence={probabilities[int(action)]:.3f} pose={tuple(int(v) for v in current)}")
            if terminated or truncated:
                print(f"완료: holding={info['holding']} phase={info['phase']}")
                if viewer is not None:
                    time.sleep(2.0)
                return bool(terminated and info["holding"])
    finally:
        if viewer is not None:
            viewer.close()
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if run(args.seed, args.headless) else 1)


if __name__ == "__main__":
    main()
