# Autonomous-from-HOME grasp validation — 2026-07-27

This report records failures as failures. The requirement is one command from
the literal firmware HOME pose, with no remembered reach, target coordinate, or
human correction after start.

## Command

```bash
PYTHONPATH=laptop:. python3 laptop/floor_servo.py --autonomous-from-home
```

The command performs HOME → open ready → verified floor-hover search → two-frame
candidate confirmation → strict alignment → fixed-reach descent → guarded close
→ contact check → 80 mm lift → retention check.

## Observed trials

1. The first autonomous trial found the object at reach 0, aligned itself to
   reach 9, closed, and reported contact. At 35 mm it still reported a 15.9 px
   residual. A diagnostic 70 mm lift showed the car on the table and the empty
   gripper at a 16.8 px residual: the object had slipped and the old 7 px
   absolute threshold produced a false success. **Failure.**
2. The controller was changed to align a 35%-inset interior pinch line at a
   55 mm hover, lift to 80 mm, and require at least 50% of the initial close
   obstruction after lift. Strict alignment also replaced the old 160 px
   best-seen fallback. One run reached `dv=-27` but was interrupted before any
   descent; no success is claimed.
3. A run converged on repeated bottom-clipped geometry but the policy correctly
   blocked descent because its raw depth meaning had not yet been normalized.
   **Safe stop.**
4. After the geometry/policy meanings were unified, a run reached the floor but
   lost tape markers. Close did not regain the markers, contact became UNKNOWN,
   and recovery opened the arm. **Safe failure.**
5. A later run exposed lateral drift (`du=-42→-129`) during descent. Contact was
   UNKNOWN and no lift was permitted. The object was displaced outside the
   current wrist view during these failed trials. **Failure.**

## Improvements retained

- Literal HOME is part of the autonomous command.
- Search uses only the physically exercised 55 mm floor-hover manifold and
  selects candidates entering the fixed-base jaw corridor in two fresh frames.
- FastSAM background fragments cannot trigger descent merely by being ranked.
- No best-seen alignment fallback remains; exact or stable clipped geometry is
  required.
- A target and tape may be occluded at the final open grasp pose only after a
  verified pre-descent lock. Post-close marker loss remains UNKNOWN and blocks
  lift.
- Contact uses a same-run empty endpoint at the selected wrist pitch. The empty
  calibration pose is 100 mm high so the calibration close cannot touch the
  target.
- An 80 mm lift must retain at least half of the initial obstruction. The weak
  slipped-object residual that fooled the old threshold now fails.

## Current conclusion

The autonomous search and fail-closed behavior are working, but a general
unassisted physical grasp is **not yet validated**. The next physical trial needs
the object placed at an arbitrary reachable starting location because the
previous object was displaced out of view. No target coordinate or reach should
be provided after placement.
