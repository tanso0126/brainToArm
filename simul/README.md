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
