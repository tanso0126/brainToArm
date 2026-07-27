# Full floor-pick policy v1

## Delivered scope

`models/full_task_policy_v1.ts` is the promoted complete-task macro policy. It
owns the decision sequence `search -> align -> descend -> close -> lift ->
recover`; deterministic safety code still converts each macro into bounded
`floor_pose` commands and requires fresh camera evidence. The older
`alignment_policy_v1.ts` is retained only for reproducibility and is superseded
for integration.

The artifact SHA-256 is
`e4451c8bc64399a8b7382d50874a262a0c205eddccc263fe33c9c379abf40323`.

## Transfer boundary

The network receives exactly 15 values that the real Mac pipeline has:

1. frame quality, target visibility, both-marker visibility, and target
   continuity;
2. target-to-jaw depth and centerline image errors;
3. normalized visual jaw opening and coherent lift motion;
4. all six already-commanded servo angles and the previous macro action.

True target pose, simulated depth, contacts, shape, size, target elbow, reward,
and MuJoCo state never enter the actor. The depth response is randomized around
the measured physical Jacobian `-12.9 px/degree`, rather than trusting a
synthetic camera gain. `laptop/full_task_adapter.py` builds the identical vector
from `WristScene`, `WristObservation`, and the host's commanded pose without
opening a camera or serial port.

## Training

DAgger-style behavior cloning collected 120,000 expert/noisy states plus three
40,000-state policy-disturbance rounds (240,000 total). A 15->192->128->64->8
MLP reached 98.3715% held-out action accuracy. Raw randomized closed-loop
success was 95.15% over 2,000 episodes. The camera/pose safety shield raised it
to 99.65% over 10,000 episodes.

Descent, close, and lift additionally require the same guarded decision from two
fresh frames. On a later disjoint 10,000-seed evaluation this completed 9,997
episodes (99.97%, mean 12.0048 macro frames). The three failures stopped or
timed out; the shield never converts missing quality/markers into permission.
Deterministic evaluation completed 1,000/1,000.

## Independent contact-physics evaluation

`evaluate_full_task_physics.py` runs the learned/shielded decisions but judges
success from the free MuJoCo body, not the symbolic environment. A pass requires
all of the following: the object rises at least 3 mm, its bottom clears the
floor, no target-floor contact remains, and its XY position follows the tool.
The evaluator uses bounded servo trajectories, box/cylinder/sphere targets,
14--22 mm half-size, random reachable depth and lateral error, sensor dropouts,
and up to two camera-style reacquisition retries.

Final disjoint 2,000-seed result:

| Set | Result |
|---|---:|
| symbolic task completion | 2,000/2,000 (100%) |
| physical lift, all 28--44 mm objects | 1,960/2,000 (98.0%) |
| first-attempt physical lift | 1,953/2,000 (97.65%) |
| normal envelope, width <=40 mm | 1,488/1,488 (100%) |
| 40--44 mm edge-size stress | 472/512 (92.1875%) |

All remaining failures were oversized edge-stress objects. They are reported as
failures, not relabelled by lowering the lift threshold. The 40 mm normal
envelope is therefore the current simulated jaw-capacity contract; it becomes a
real limit only after measuring the physical fingers.

## Safety and reality boundary

These numbers prove repeatable behavior in the corrected simulator, not a real
robot success rate. `FLOOR_GRASP_EXECUTE_VERIFIED` remains `False`. Reality
promotion is a shadow comparison using the adapter, followed by a cleared-workspace
test that verifies marker visibility, contact against the empty-jaw baseline,
and coherent post-lift motion. No learned output may bypass those checks.

## Reproduce

```bash
python3 simul/prepare_assets.py --extract
python3 -m unittest simul.test_mujoco_robot simul.test_full_task -v
python3 simul/train_full_task.py
python3 simul/evaluate_full_task_physics.py --policy simul/models/full_task_policy_v1.ts --episodes 2000 --seed 20260729
PYTHONPATH=laptop python3 laptop/test_pipeline.py
```

Training output stays in ignored `simul/generated/`. Only reviewed artifacts,
their exact metrics, and source needed to reproduce them are tracked.
