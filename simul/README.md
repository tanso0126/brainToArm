# brainToArm sim-to-real workspace

This directory contains the user-supplied printable robot sources and the
simulation/training pipeline built from them.

## Source geometry

- `Robotic+Arm+with+Servo+&+Arduino.3mf`: Bambu Studio print project with 22
  mesh objects, embedded photos, millimeter units, and model metadata.
- `Robotic+Arm+with+Servo+&+Arduino.zip`: 20 named STL source parts.
- `model_manifest.json`: immutable hashes, measured mesh envelopes, current
  six-servo semantics, camera constraints, and the role assigned to each part.

The 3MF arranges parts for printing; it does **not** contain assembly joint
frames or servo pivots. Simulation therefore uses the original meshes for visual
identity and measured envelopes, while explicit calibrated kinematic frames and
simple convex collision shapes define dynamics. Print-plate transforms must not
be mistaken for assembly transforms.

Verify and extract the meshes needed by MuJoCo:

```bash
python3 simul/prepare_assets.py
python3 simul/prepare_assets.py --extract
```

Generated meshes, datasets, checkpoints, videos, and run logs are deliberately
ignored by Git. The source archives and manifest remain tracked, so every
generated artifact is reproducible.

## Transfer contract

The eventual actor may observe only:

- RGB from one gripper-mounted camera, randomized around the PW315 model;
- commanded servo angles and its previous action, which the real host already
  knows without adding a sensor.

Simulator-only object pose, depth, contacts, and segmentation labels may be used
by rewards or a teacher, but must never enter the deployed actor observation.
Motor 1 is locked at 90 degrees for the current planar task. The remaining
servo commands use the same minima/maxima, 90/180 gripper convention, 170-degree
level wrist roll, and floor reference poses as the real controller.

## MuJoCo robot model

`mujoco_robot.py` builds the assembled kinematic model. It does not guess joint
pivots from the print-bed layout: named STL parts are non-colliding visuals,
while capsules and boxes define stable collision geometry. Its public control
boundary is always the physical six-value servo vector. There is no base joint,
so a policy cannot accidentally learn a motion that failed motor 1 cannot make.

The shoulder mapping is intentionally piecewise because the printed linkage is
not a direct one-degree servo-to-joint mechanism. The near-floor portion is fit
to both physically reproduced levels and the measured `-6/11` shoulder/elbow
vector. Across elbow 78..110 degrees, each curve varies by under 2.5 mm in
height while translating the tool by over 15 mm. This is a calibrated local
model, not a claim that every unmeasured joint pose is already exact.

The gripper camera renders only RGB. At the 170-degree level wrist it reproduces
the physical tape convention (blue left, red right), includes both fingers and
the space between them, and never renders simulator sites, depth, masks, or
object coordinates into the actor input.

Run the headless visual and numeric checks without opening the webcam or Uno:

```bash
python3 simul/smoke_mujoco.py
python3 simul/test_mujoco_robot.py
```

The smoke sheet is written to ignored
`simul/generated/mujoco_smoke.png`. Expected tool-center heights are about 41 mm
at the hover reference and 8 mm at the grasp reference.
