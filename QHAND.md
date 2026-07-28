# brainToArm quick handoff — MuJoCo 3D + EEG demo

Updated: 2026-07-28 (KST)

## 1. What the demo is now

The earlier browser-only 2D simulator has been removed. The default dashboard
workspace is a client for a live MuJoCo engine under `simul/`.

The participant can:

1. create and arrange several 3D objects and a destination tray;
2. see the original robot STL assembly in a MuJoCo overview camera;
3. see the robot's eye-in-hand RGB camera, including the blue-left/red-right
   finger markers;
4. run camera-gated search, contact-physics grasp, lift, carry, and release;
5. send “wrong object” before grasp, during movement, after drop, or after the
   task says complete;
6. watch a late rejection physically re-grasp the delivered free body, return it
   near its saved origin, remember the rejection, and choose another object;
7. use the same rejection entry point from a manual button, mock ErrP, or the
   PolyG-I onset-locked detector.

No physical serial port or webcam is opened by this simulation.

## 2. Start

From the repository root:

```bash
python3 laptop/eeg_dashboard.py
```

This starts:

- UI: `http://localhost:3000`
- local API: `http://127.0.0.1:8765`
- the EEG monitor
- the MuJoCo studio API, loaded lazily on first simulation request

Separate processes for debugging:

```bash
python3 laptop/eeg_dashboard.py --api-only --no-browser
cd dashboard && npm run dev -- --port 3000
```

## 3. Files and boundaries

| File | Responsibility |
|---|---|
| `simul/studio.py` | editable MuJoCo scene, RGB detection, bounded motion, reversible physics task |
| `simul/mujoco_robot.py` | original fixed-base training/evaluation robot and calibrated planar mapping |
| `dashboard/app/SimulationLab.tsx` | 3D/wrist streams, scene controls, state/servo/event UI, ErrP bridge |
| `laptop/eeg_dashboard.py` | local HTTP ownership of EEG and MuJoCo services |
| `simul/test_studio.py` | scene, rendering, delivery, late rejection, and next-object regression tests |

The original `mujoco_robot.py`, trained TorchScript policy, and its reported
fixed-base evaluation remain unchanged. `studio.py` builds a separate runtime
MJCF from it and adds:

- editable box/cylinder/sphere free bodies;
- a nearly floor-flush physical destination tray;
- the same six MuJoCo controls as the fixed-base robot;
- a single sagittal workspace line: depth 387–414 mm, lateral position 0.

Servo 1 has no MuJoCo joint or actuator. Every pose forcibly reports and
commands 90°, matching the currently non-responsive real base motor. Search
uses only coordinated servo 2/3/4 movement.

## 4. What the screen means

### MuJoCo 3D overview

This is a live render of the physics state, not a CSS/Canvas arm drawing.
Objects can be pushed, pinched, lifted, dropped, and retrieved through MuJoCo
contact.

### Wrist RGB camera

The selection gate renders a 320×180 wrist image and finds connected color
regions. Marker-colored pixels around the known blue/red finger tapes are
excluded before candidate scoring. An object cannot be selected unless its
pixels appeared in a scan frame.

The detector does use the editable object's RGB color to associate the blob with
the task identity. It does not receive depth, object pose, contact, or
segmentation buffers. MuJoCo pose is still legitimately used by the physics
engine, origin bookkeeping, delivery planning, and success verification.

### Scene placement map

The small top-down SVG is only an authoring tool. Select a point or the blue
tray and drag it left/right; the backend projects the request onto the reachable
fixed-base depth interval and rebuilds the paused MuJoCo scene. It is
deliberately not used for robot perception or task success.

### Success

“Grasped” requires both:

- the free body rose at least 3 mm;
- its XY position follows the gripper tool within 55 mm.

A symbolic `held=true` alone cannot pass. Placement is also actual open/release
physics. A measured local object-to-tool offset compensates the destination
command so return error stays bounded instead of silently teleporting the body.

## 5. Reversible state machine

```text
SCANNING
  -> TARGET / ErrP window
  -> REACHING
  -> GRASPING (contact + lift verification)
  -> TRANSPORTING
  -> EVALUATING / ErrP window
  -> COMPLETED
```

`X` or the red button sends the same reject event at any time.

- Before grasp: remember the rejection and scan another camera-visible object.
- While held: return and release at the saved origin.
- After tray drop or `COMPLETED`: go back to the tray, close on the free body,
  lift it, return it, release it, then scan the next non-rejected object.
- Rejected IDs cannot be chosen within the cycle.
- If every eligible table object is rejected, clear the rejection memory,
  increment the cycle, and search again.

## 6. Scene editing

Editing is disabled while the physics task is running.

- Add: sphere, box, cylinder
- Edit: name, shape, RGB color, half-size 4.5–8.0 mm
- Place: depth 387–414 mm on the fixed y=0 line
- Move tray: the same bounded line
- Delete: selected table object
- Reset: return every object to its authored origin and clear task memory

The bounds are ceilings from the current simulated floor curve, not a claim
that every real point is calibrated.

## 7. PolyG-I

Hardware path:

- VID `0x0F1F`, PID `0x0010`
- D1WD10 HID reports
- EEG CH1–CH8 at 256 Hz

Preparation:

1. Connect PolyG-I and run the one-command launcher.
2. Stay in **시뮬레이션 작업실**; its embedded EEG card now shows all eight
   live channels beside the 3D scene.
3. Press **측정 시작** in that card.
4. Verify CH1, CH2, CH3, CH4, and CH8 all say `신호 있음`, then remain still
   and relaxed for eight seconds.
5. Choose **PolyG-I** and press **최근 8초로 ErrP + TAR 통합 보정**.
6. Start the 3D task and look at the selected object/robot action. Do not try to
   manufacture a generic blink or button-press signal; an ErrP trial is the
   time-locked response to perceiving the shown decision as wrong.

At `SIM_TARGET_PRESENTED` and `SIM_BASKET_DROP`, the API evaluates:

```text
0.2 s pre-onset baseline + 0.8 s post-onset response
```

At least 80% of the expected 256 Hz samples are required. A missing or short
epoch fails closed and does not become a rejection. If CSV recording is active,
the onset marker is written into the sample stream.

All eight channels are acquired and displayed. **ErrP uses CH8 only**, according
to the project's electrode mapping. Continuous cognitive load is calculated
separately every second from a two-second Welch window: mean theta power
(4–8 Hz) on CH1–CH4 divided by alpha power (8–13 Hz) on CH8. The session value
is `(current TAR - rest TAR) / rest TAR`, then smoothed by an EMA.

The web simulation now uses the same `AutonomyAllocator` as the standalone robot
orchestrator. Higher relative TAR increases robot weight, the ErrP action
threshold, and the action-application stride. Lower TAR shifts weight toward the
human and applies every checkpoint. CH8 ErrP probability is still calculated at
every action checkpoint; a skipped checkpoint means observation-only, not that
the EEG was ignored. A very strong ErrP remains an immediate override.

The default `baseline` backend is not a trained ErrP classifier. The eight-second
button measures CH8 resting noise σ and the CH1–4/CH8 resting TAR. For each
decision it:

1. filters CH8 to 1–10 Hz;
2. subtracts the 0.2-second pre-onset mean;
3. searches the first 0.6 seconds after onset for the strongest sustained
   150 ms negative deflection;
4. divides that magnitude by resting σ;
5. maps the z-score to probability; the current 50% threshold corresponds to
   approximately z=3.3.

The embedded panel shows TAR, rest-relative change, robot/human weights, action
stride, CH8 probability, adaptive threshold, and whether the latest decision was
applied or only observed. Flat/saturated required channels, insufficient samples,
or an uncalibrated session produces an explicit error instead of a silent
non-detection. Reliable participant use still requires collecting labeled
correct/error trials and training the `model` backend; the baseline heuristic is
a diagnostic/demo detector and must not be reported as validated subject
accuracy.

The calibration button uses its own full eight-second quality window, not the
short two-second channel badge. It becomes enabled only when enough samples are
present and the complete CH1·2·3·4·8 window is clean. If it stays disabled, stop
acquisition, lower PGA (start with ×1.00), reconnect the indicated channels plus
reference/ground, restart, and wait for a new clean eight-second window. Do not
bypass saturation: it cannot produce defensible ErrP or TAR baselines.

## 8. Local endpoints

```text
GET  /api/simulation/status
GET  /api/simulation/frame?camera=overview|wrist&width=...&height=...
POST /api/simulation/start
POST /api/simulation/stop
POST /api/simulation/reset
POST /api/simulation/reject
POST /api/simulation/objects/add
POST /api/simulation/objects/update
POST /api/simulation/objects/delete
POST /api/simulation/basket/update

POST /api/errp/calibrate
POST /api/errp/check
```

## 9. Verification

```bash
PYTHONPATH=laptop:. python3 -m unittest simul.test_studio -v
PYTHONPATH=laptop:. python3 -m unittest \
  simul.test_mujoco_robot simul.test_full_task simul.test_studio -v
cd dashboard && npm run lint && npm test
python3 -m py_compile simul/studio.py laptop/eeg_dashboard.py
```

The studio regression suite covers:

- original STL arm plus overview/wrist JPEG rendering;
- absent servo-1 yaw joint, six MuJoCo controls, and an invariant 90° base command;
- scene edit clamping;
- physical delivery;
- rejection after completion;
- physical tray retrieval and return;
- rejection memory and a different next delivery.

## 10. Honest status

- This is now a real MuJoCo rigid-body/contact studio, not the removed 2D mock.
- Its success is simulation evidence, not a real-arm success rate.
- The current demo intentionally cannot yaw. Multiple objects and the tray must
  share the narrow calibrated front/back lane, just as on the current real arm.
- Restoring servo 1 later requires a separate calibration and patch; simulation
  will not silently enable it ahead of the hardware.
- RGB candidate gating is exercised, but the complete autonomous transport
  planner still uses known authored origin/tray coordinates. That task metadata
  is not claimed to be inferred from monocular RGB.
- The physical arm remains separately safety-gated; this patch sends no Uno
  commands.
