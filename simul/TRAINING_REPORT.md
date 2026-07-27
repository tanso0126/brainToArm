# Full floor-pick policy v1

## Delivered scope

`models/full_task_policy_v1.ts` is the promoted complete-task macro policy. It
owns the decision sequence `search -> align -> descend -> close -> lift ->
recover`; deterministic safety code still converts each macro into bounded
`floor_pose` commands and requires fresh camera evidence. The older
`alignment_policy_v1.ts` is retained only for reproducibility and is superseded
for integration.

The artifact SHA-256 is
`b4a5cf2b976b7571bf38b2b7e96d30d5159d00186e12d93569fe91cdfa4772b7`.

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
MLP reached 98.2882% held-out action accuracy. Raw randomized closed-loop
success was 95.55% over 2,000 episodes. The two-frame camera/pose safety guard
completed 9,998/10,000 randomized episodes (99.98%, mean 11.4678 macro frames)
and 1,000/1,000 deterministic episodes.

Training randomization now explicitly hides the target mask through most grasp
poses, matching the real eye-in-hand occlusion. A selection locked by verified
hover alignment may remain continuous through that occlusion; a missing mask
without such a lock still cannot authorize close.

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
| physical lift, all 28--44 mm objects | 1,959/2,000 (97.95%) |
| first-attempt physical lift | 1,952/2,000 (97.6%) |
| normal envelope, width <=40 mm | 1,488/1,488 (100%) |
| 40--44 mm edge-size stress | 471/512 (91.9922%) |

All remaining failures were oversized edge-stress objects. They are reported as
failures, not relabelled by lowering the lift threshold. The 40 mm normal
envelope is therefore the current simulated jaw-capacity contract; it becomes a
real limit only after measuring the physical fingers.

## Safety and reality boundary

These percentages are simulation-only and are not presented as a real robot
success rate. The guarded model was nevertheless used as a macro gate during
the real 2026-07-27 trial; the fixed-reach controller then achieved one physical
close-and-lift success with visual jaw contact retained after lift. See
`docs/PHYSICAL_GRASP_VALIDATION.md`. No learned output bypasses the real marker,
alignment, or jaw-contact checks.

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
