# Wrist alignment policy v1

## Result

`models/alignment_policy_v1.ts` is the first reproducible sim-trained policy for
local floor alignment. Its SHA-256 is
`becbb150a282299707b0b4f7c122ad4091cf259bc60d8ea4b8f29fc36fc1d7d6`.

The final model trained on 20,000 domain-randomized frames for 16 epochs. The
held-out frame set reached 0.0928 normalized action MAE and 97.80% direction
accuracy. The separate aligned head reached 93.43% classification accuracy,
96.09% stop recall, and 7.29% false-stop rate at threshold 0.65.

On a later disjoint 500-seed closed-loop simulation evaluation, 499 episodes
aligned within 20 commands (99.8%) with a mean of 4.664 commands. These are
**simulation metrics, not a claim of 99.8% real-robot success**.

## What was randomized

- target color across vivid HSV colors, target size 12–28 mm, and box/cylinder/
  sphere geometry;
- wrist-camera field of view and local mounting position;
- light intensity, per-channel white balance, brightness, gamma, sensor noise,
  blur, small image rotation, translation, and scale;
- arbitrary initial and target positions over the physically calibrated local
  elbow range 78..110 degrees.

Metal shader variants are not recompiled during resets. One MuJoCo model is
loaded and mutable camera/geometry arrays plus pixel-space installation jitter
are randomized at runtime, preventing the resource leak found in the first
training attempt.

## Strict deployment boundary

The TorchScript signature accepts only RGB, the six commanded servo values, and
the previous action. True object pose, desired elbow, depth, collision contacts,
segmentation, and reward state never enter the actor.

The actor has two outputs: elbow motion and `aligned_probability`. The learned
aligned output has a measured 7.29% false-stop rate, so it must never authorize
descent or stopping alone. Deployment requires both:

1. the existing marker/candidate geometry says the selected target is aligned;
2. `aligned_probability >= 0.65`.

If the learned action falls inside its deadband while geometry is not aligned,
the existing centroid-sign controller should take the next bounded step; it
must not reinterpret a zero recommendation as grasp permission.

## Claude integration contract

Use `AlignmentPolicyRunner()` in the `ALIGN` portion of `floor_grasp.py`, after
FastSAM has selected a candidate and while the arm remains on the hover curve.
Pass the current complete six-servo command and previous normalized action. Clamp
the resulting elbow change through `floor_pose`; never write raw angles or open
a new serial connection.

For v1, invoke the actor only when exactly one portable candidate is in view.
The training scene contains one goal and has no goal-conditioning input, so with
several visible candidates the actor cannot know which FastSAM candidate the
human selected. The existing candidate-specific centroid controller remains the
correct path for multi-object scenes until a target-conditioned v2 is trained.

Keep `FLOOR_GRASP_EXECUTE_VERIFIED=False`. First run shadow mode on real frames,
log learned direction/probability beside the existing controller, and verify the
sign over multiple poses. Only then may the already documented physical gate be
changed. The policy covers local alignment only; deterministic code still owns
search, descent, close, visual contact, coherent lift, recovery, and STOP.

## Reproduce

```bash
python3 simul/prepare_assets.py --extract
python3 simul/test_mujoco_robot.py
python3 simul/train_alignment.py --samples 20000 --epochs 16 --eval-episodes 200
```

Training outputs go to ignored `simul/generated/`. A candidate is promoted to
`simul/models/` only after extended evaluation and a recorded hash.
