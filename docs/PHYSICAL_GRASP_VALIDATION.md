# Physical floor-grasp validation — 2026-07-27

This record separates the real-arm result from simulation and mock tests.

## Verified setup

- Camera: PW315 wrist camera, 1280×720 live publisher.
- Arm: Uno persistent session on `/dev/cu.usbserial-110`.
- Base: fixed at 90°; wrist roll: 170°; gripper: 90° open / 180° closed.
- Test object: the toy car visible between the blue-left and red-right finger
  markers.
- Initial open hover: `[90, 115, 90, 143, 90, 170]`.

## Executed result

1. FastSAM produced one consolidated physical candidate despite nested masks and
   lower-frame clipping.
2. The selected approach-side edge was `(636, 720)` px and the finger midpoint
   was `(614, 699)` px: `du=21`, `dv=21` at hover.
3. Forward reach was locked at 37 for the whole descent. The tool-height targets
   were 30, 24, 18, 12, and 6 mm; the final observed alignment remained
   `du=10`, `dv=23` instead of chasing an occluded replacement mask.
4. The guarded policy authorized descend, close, and lift on fresh frames.
5. At gripper command 180°, the empty-jaw baseline expected 89.25 px. The
   observed jaw opening remained about 292 px immediately after close (203 px
   residual) and about 285 px after lift (196 px residual), both far above the
   7 px contact threshold.
6. The arm lifted closed to `[90, 115, 90, 143, 180, 170]`; the second independent
   jaw assessment remained `CONTACT`. This is physical obstruction retained
   through a vertical lift, not merely an object centered in the image.
7. The object was then lowered closed to the same fixed-reach floor point,
   released, and the arm returned to open hover.

Result: **goal 1 has a real close-and-lift success**. Goals 2 and 3 have the same
physical controller connected to candidate selection and reject/next, but still
require a live scene containing at least two reachable objects for separate
physical demonstrations. Software-only results are not counted as those two
physical validations.

## Root cause fixed

`laptop/arm_fk.py` still used an older pre-contact simulation joint map. At
`[90, 109, 90, 143, …]` it therefore believed the fingers were at floor height
while the current model put them roughly 57 mm high. The runtime joint map now
matches `simul/mujoco_robot.py` to numerical precision over representative
poses. With wrist pitch 143°, shoulder 115→138° lowers the modeled tool from
34.7 to 5.4 mm. This explicitly compensates for the height gained while aiming
the wrist upward.

## Reproduce

```bash
# List candidates, select nearest, align, and physically grasp.
PYTHONPATH=laptop:. python3 laptop/floor_servo.py --start-reach 37

# Select a different ranked object directly.
PYTHONPATH=laptop:. python3 laptop/floor_servo.py --candidate-index 1

# Apply one "not that one" veto, then grasp the next candidate.
PYTHONPATH=laptop:. python3 laptop/floor_servo.py --reject-count 1
```

The final two commands must be run only when two separately segmented reachable
objects are visible. Missing candidates stop without moving the arm.
