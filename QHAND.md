# brainToArm quick handoff — simulation + EEG demo

Updated: 2026-07-28 (KST)

## 1. Immediate objective

The physical arm is intentionally out of scope for the next demo. The deliverable
is a safe local simulation in which a participant can:

1. place several objects and a destination basket;
2. watch a wrist-camera-style RGB view detect and choose an object;
3. watch the simulated arm grasp, carry, and place it;
4. send a late or early “wrong object” signal;
5. see the robot retrieve the object if necessary, return it to its exact origin,
   remember the rejection, and try the next object;
6. replace the manual signal with the PolyG-I ErrP detector without changing the
   task state machine.

## 2. One-command launch

From the repository root:

```bash
python3 laptop/eeg_dashboard.py
```

This starts:

- the local API at `http://127.0.0.1:8765`;
- the UI at `http://localhost:3000`;
- the existing EEG monitor;
- the new simulation studio, which is the default tab.

The simulation itself runs even when the EEG API/device is offline. A separate
UI launch is:

```bash
cd dashboard
npm run dev
```

## 3. Simulation Studio

Main file: `dashboard/app/SimulationLab.tsx`

### Direct manipulation

- Drag any table object to a new location while stopped.
- Drag the dashed blue basket.
- Add circles, blocks, or capsules.
- Select an object to rename it, change its color, or change its size.
- Delete the selected object.
- `Space`: start/pause.
- `X`: send the current rejection signal.

### Perception path

The wrist-camera panel is not just a decorative crop. It:

1. renders the table from the current simulated eye-in-hand pose;
2. re-reads the rendered RGB pixel buffer;
3. segments learned object colors;
4. builds bounding boxes/confidence values;
5. passes only those detections to the selection loop.

The selection loop performs a three-heading camera sweep and cannot choose an
object that did not appear in the pixel-derived detection list. Blue/red bars at
the image bottom reproduce the physical finger-tape convention.

### Reversible task state machine

The normal state sequence is:

```text
SCANNING
  -> TARGET_PRESENTED / ErrP window
  -> REACHING
  -> GRASPING
  -> TRANSPORTING
  -> BASKET_DROP / ErrP window
  -> COMPLETED
```

A rejection can arrive during any of those states.

- Before grasp: abandon the reach and return home.
- While holding: carry the object back to its saved origin and release.
- After basket drop or even after `COMPLETED`: move back to the basket, pick the
  delivered object up, return it to its saved origin, and release.
- Add the object ID to the rejection set and continue with the next detected
  candidate.
- Never select an ID in the current rejection set.
- If every table object is rejected, clear the set, increment the cycle counter,
  and begin again from the first camera-detected candidate.

The right-side action trace is the observable audit log for these transitions.

## 4. External signal modes

The simulation’s “거부 신호” panel has three sources.

### Manual

Press the red button or `X`. This is the reliable no-device demo path.

### Mock ErrP

The same button is labeled “가상 ErrP 보내기”. It exercises the identical
rollback/rejection logic while presenting it as a simulated brain event.

### PolyG-I

The browser uses the existing native macOS HID stack:

- VID `0x0F1F`
- PID `0x0010`
- D1WD10 1,024-byte HID reports
- 32 rows/report, 16 physical channels, EEG CH1–CH8
- 256 Hz (`sample selector 8`)

Preparation:

1. Connect PolyG-I exactly as in the prior verified setup.
2. Run `python3 laptop/eeg_dashboard.py`.
3. Open **EEG 실시간 모니터** and press **측정 시작**.
4. Keep still/resting for at least 8 seconds.
5. Return to **시뮬레이션 작업실**.
6. Select **PolyG-I** under `거부 신호`.
7. Press **최근 8초로 ErrP 보정**.
8. Start the simulation.

At both `SIM_TARGET_PRESENTED` and `SIM_BASKET_DROP`, the API opens an onset-
locked epoch:

```text
0.2 s pre-onset baseline + 0.8 s post-onset response
```

It requires at least 80% of the expected 256 Hz samples. Missing/short epochs do
not become a rejection. The existing `ErrPDetector` returns probability and the
configured threshold (`0.5`) decides whether the same rollback state machine is
triggered.

New local endpoints:

```text
POST /api/errp/calibrate  {"seconds": 8}
POST /api/errp/check      {"marker": "SIM_TARGET_PRESENTED"}
POST /api/errp/check      {"marker": "SIM_BASKET_DROP"}
```

If CSV recording is active, the decision marker is attached to the next sample
row.

## 5. Design

The simulation UI follows `docs/DESIGN.md`:

- one saturated primary orange (`#ff5d00`);
- grayscale surfaces and low-contrast hairlines;
- flat cards rather than decorative texture/gradient;
- short Korean action labels;
- 8–20 px rounded scale and pill statuses;
- explicit error red, informative blue, and success green.

The existing EEG tab remains available in the same shell. Its waveform still
uses fixed shared scales and buffered `requestAnimationFrame` rendering.

## 6. Important honesty boundary

The new human-facing studio is a browser kinematic/interaction simulator. It
uses a rendered wrist-camera pixel pipeline and deterministic grasp/transport
state, but it is **not claiming MuJoCo rigid-body contact physics**.

The repository’s separate `simul/` MuJoCo environment remains the physics and
policy-evaluation layer. It currently models one primary target and a fixed
planar base. Connecting the editable multi-object UI directly to live MuJoCo
free bodies is future work; the current studio is intended for tomorrow’s
human/EEG shared-autonomy protocol demonstration and state-machine validation.

## 7. Physical-arm safety status

Do not infer that this simulation re-enables physical execution.

- Physical motion is still separate.
- The persistent Uno session has a swept-volume collision interlock.
- Autonomous movement requires a fresh wrist-camera frame.
- The previous body-hook incident path is a regression test and is rejected.
- Motor 1 was last physically observed non-responsive; see `PATCH_NOTES.md`.

## 8. Verification performed

```text
dashboard npm build: pass
dashboard SSR tests: 2/2 pass
dashboard ESLint: pass after warning cleanup
eeg_dashboard.py py_compile: pass
synthetic 8-second ErrP baseline calibration: pass (2,046 samples)
laptop/test_pipeline.py: all pass
```

No real robot-arm serial port was opened and no physical motion was commanded for
this simulation/EEG patch.

