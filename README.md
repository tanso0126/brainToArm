# brainToArm — a shared-autonomy robot arm corrected by brain waves (EEG)

> **One-line summary:** a robot arm picks up objects and delivers them on its own
> using an overhead camera; a human wearing an 8-channel EEG headset does nothing
> but *watch*, and when the arm reaches for the **wrong** object the human's brain
> automatically fires an error signal that we detect and use to make the arm
> change its mind. The human never steers — they only veto.

This document is the complete, self-contained description of the project. If you
are a person or an AI seeing this repo for the first time, reading it top to
bottom tells you what the project is, why it is built this way, exactly what
every file does, what already works, what still needs real hardware, and the
precise steps to bring it up. No other context is required.

> ### ✅ PolyG-I input works natively on this Mac
>
> The device has now been examined on real hardware. Findings that override
> several statements later in this file:
>
> - The PolyG-I on hand is **USB HID** (VID `0x0F1F`, PID `0x0010`), **not** a
>   virtual COM port. **"Path A" (`EEG_SOURCE="serial"`) can never work for it** —
>   LAXTHA's COM-port variant is a different product ID (`0x002A`).
> - It **opens natively on macOS** via `hidapi`, so **"Path B" (the Windows
>   `LXSMWD12.dll` bridge) is unnecessary**.
> - The blocker is resolved. Static analysis of TeleScan's installed
>   `LXSM-D1WD10.dll`, the official D1WD10 manual, and physical A/B captures
>   recovered and verified the complete initialization sequence, decoder, and
>   ADC-input voltage coefficient. `python3 laptop/eeg_detect.py` starts the
>   device and `EEG_SOURCE="hid"` feeds its samples into `EEGBridge`.
> - This HID stream is **not LXSDF**. Each 1,024-byte report holds 32 time rows ×
>   16 physical channels; channels 1–8 are EEG. Selector 8 produces the specified
>   and measured 256 Hz stream.
> - TeleScan under CrossOver **cannot** reach the device (Wine filters
>   vendor-defined HID); proven, not a configuration issue.
>
> Full evidence, commands, and the corrected earlier findings are recorded in
> [`docs/EEG_DEVICE_COMMS.md`](docs/EEG_DEVICE_COMMS.md).

> ### ✅ Base rotation restored; calibrated planar pick remains available
>
> Servo1 base yaw is operational again and manual jog/3-D control may command
> it through 0–180°. The previously verified side-camera controller remains a
> deliberately fixed-plane mode: it locks servo1 at 90°,
> re-detects the object before every attempt, corrects
> shoulder/elbow in image pixels, rotates wrist pitch to 180° and the physically
> level wrist roll to 170°, opens the
> gripper at 90°, descends before closing it at 180°, verifies lift, transports,
> releases, and verifies displacement. The complete sequence was physically
> demonstrated on 2026-07-20: the white object was picked up, moved right, and
> placed back on the table.
>
> With the saved clean background and unchanged camera, run
> `python3 laptop/planar_pick.py --detect-only`, then
> `python3 laptop/planar_pick.py --run`. If the camera moves, remove the object
> once and run `python3 laptop/planar_pick.py --learn-background` first.

> ### ✅ Wired ESP32 + OV2640 frame capture works
>
> With the robot arm disconnected, the loose OV2640 was physically verified on
> the classic no-PSRAM ESP32 at `/dev/cu.usbserial-0001`. The physical harness
> has D5/D7 crossed; the firmware compensates for it. The current reliable mode
> is hardware JPEG at 160x120 over USB serial with spatial color calibration,
> bounded per-frame neutral correction, full JPEG decoding, and block-damage
> rejection. Run:
>
> ```bash
> # Run once while the camera sees an empty neutral gray/white workspace:
> python3 laptop/calibrate_esp32_color.py
>
> # Normal corrected capture:
> python3 laptop/capture_esp32_camera.py
> ```
>
> The ESP32 must have `firmware/esp32_camera_diagnostic` flashed with ESP32
> Arduino core 2.0.17. This is a verified USB frame source, not yet Wi-Fi video.
> The lower resolution is deliberate: 320x240 worked but long jumper wires
> caused intermittent JPEG block corruption; 160x120 plus the quality gate
> completed 20/20 clean output requests in the physical stability test.

> ### ✅ Named wrist webcam and colored-finger perception are implemented
>
> The primary eye-in-hand source is now the USB **AVerMedia PW315**, opened by
> its stable AVFoundation device name so iPhone Continuity Camera and the MacBook
> front camera cannot silently replace it. Blue tape on the physical left finger
> and red tape on the right finger define the live gripper midpoint, opening, and
> jaw angle. A separate colored-object contour produces image x/y and orientation
> error relative to that measured midpoint. The detector does not use a venue
> background image. With the Uno unplugged, run:
>
> ```bash
> python3 laptop/wrist_vision.py --live
> ```
>
> Left-click the intended object to lock its hue, right-click to return to
> automatic colored-object selection, `S` saves an annotated frame, and
> `Q`/Escape exits. `READY / ALIGNED` is impossible unless both tapes, a target,
> and an exposure/contrast quality gate all pass.

> ### 🧪 Complete MuJoCo floor-pick policy
>
> The user-supplied 3MF/STL arm is now assembled in a hardware-isolated MuJoCo
> model under `simul/`. For the current wrist-camera floor task it deliberately
> has no base-yaw joint, uses the real six-command limits and 90°/180° gripper
> direction, reproduces the measured hover/grasp floor curve, and renders a
> blue-left/red-right eye-in-hand view.
>
> The promoted TorchScript policy now covers search, selected-target alignment,
> descent, close, lift, and recovery. It was trained on 240,000 disturbed states
> using only real-available wrist-camera features, commanded angles, and the
> previous action; simulator pose/depth/contact never enters inference. A
> deterministic shield and two-frame vote guard every irreversible transition.
>
> Independent MuJoCo contact evaluation physically lifted 1,960/2,000 randomized
> 28–44 mm box/cylinder/sphere targets, including 1,488/1,488 at the current
> <=40 mm normal simulated jaw envelope. These are **simulation results, not a
> claim of real-hardware success**. The hardware-free adapter in
> `laptop/full_task_adapter.py` is ready for shadow comparison while the physical
> execution gate remains off. See
> [`simul/TRAINING_REPORT.md`](simul/TRAINING_REPORT.md) before integration.

---

## Table of contents

1. [The idea and why it's interesting](#1-the-idea-and-why-its-interesting)
2. [How it works, conceptually](#2-how-it-works-conceptually)
3. [The hardware](#3-the-hardware)
4. [System architecture](#4-system-architecture)
5. [The full task, step by step](#5-the-full-task-step-by-step)
6. [Repository layout — every file explained](#6-repository-layout--every-file-explained)
7. [Key technical decisions (and the traps they avoid)](#7-key-technical-decisions-and-the-traps-they-avoid)
8. [Current status: what works, what's assumed](#8-current-status-what-works-what-is-assumed)
9. [Run it right now with zero hardware](#9-run-it-right-now-with-zero-hardware)
10. [Bring up the real hardware — exact steps](#10-bring-up-the-real-hardware--exact-steps)
11. [config.py — the single tuning surface](#11-configpy--the-single-tuning-surface)
12. [Testing and verification](#12-testing-and-verification)
13. [Glossary](#13-glossary)
14. [Sources and prior art](#14-sources-and-prior-art)

---

## 1. The idea and why it's interesting

Robots and AI are good at *executing* a task but bad at *knowing which task the
human actually wants*. A camera can see three objects on a table perfectly well,
but it cannot see the intention inside the human's head — "I wanted the *small*
nail, not the big one." That intention is roughly **one bit** of information at
the moment of ambiguity.

Classic solutions try to infer intent (gaze, gestures, voice) — all indirect and
error-prone. This project takes the intent **directly from the brain**, but in a
clever, low-effort way:

- We do **not** ask the human to consciously "think left/right" to command the
  arm (that's called motor imagery — slow, needs weeks of training, unreliable on
  cheap hardware).
- Instead we exploit a reflex: when a human watches a machine make a mistake,
  their brain **involuntarily** produces a characteristic electrical response
  called an **Error-Related Potential (ErrP)** about a third of a second later.
  No training, no conscious effort. The human just watches and silently judges.

So the division of labor is:

- **The AI does ~99% of the work:** find objects, plan a reach, grasp, deliver.
- **The brain contributes the ~1% the AI cannot get:** "no, not that one." A veto.

This is called **shared autonomy**. The whole project is a working demonstration
of using an involuntary brain signal as the correction channel.

**Concrete demo scenario:** on the table sit a big nail and a small nail (plus
maybe a screw). The human is told "get me the small nail." The arm, not knowing
which is wanted, reaches for the big nail first. The human's brain fires an ErrP.
The system detects it, the arm abandons the big nail, moves to the small nail,
gets no ErrP, and completes the pick-and-place.

---

## 2. How it works, conceptually

Two completely different kinds of information flow into the laptop:

| Source | Answers the question | Example |
|--------|----------------------|---------|
| **Overhead camera** | *What is where?* (the state of the world) | "there are 3 blobs at these table positions; the arm tip is here" |
| **EEG headset** | *What does the human want?* (the goal/reward) | "the brain just objected to the action that is happening right now" |

The camera cannot answer the second question no matter how good it is, which is
exactly why the EEG is not redundant.

The loop, in one sentence: **the arm commits to an object where the human can see
it, we read the brain for ~0.8 s, and if an ErrP appears we treat it as "wrong,
pick another," otherwise we finish the grasp and delivery.**

---

## 3. The hardware

- **Robot arm:** 6 servo motors (all *angle* servos — position-controlled, not
  continuous-rotation). Physical model:
  [MakerWorld "Robotic Arm with Servo / Arduino"](https://makerworld.com/ko/models/1134925-robotic-arm-with-servo-arduino).
  Driven by an **Arduino**. Motor map, from the base upward:

  | Servo | Arduino pin | Joint | Role |
  |-------|-------------|-------|------|
  | servo1 | 13 | base yaw | rotate the whole arm about the vertical (Z) axis |
  | servo2 | 12 | shoulder | first bend |
  | servo3 | 11 | elbow | second bend |
  | servo4 | 10 | wrist pitch | verified safe range 130°–180° |
  | servo5 | 9 | gripper | 90° open, 180° closed |
  | servo6 | 8 | wrist roll | rotate gripper orientation |

  The former unused servo3 slot was removed completely. Elbow and wrist pitch
  moved to 3/D11 and 4/D10; the observed harness has the gripper on 5/D9 and
  wrist roll on 6/D8. Arduino pin 7 is unused.

- **EEG device:** [LAXTHA PolyG-I](https://www.laxtha.com/ProductView.asp?Model=PolyG-I),
  an 8-channel EEG amplifier (part of a 16-channel polygraph). Connects by USB.
  This PID `0x0010` model uses vendor-defined **USB HID** with fixed raw blocks;
  it does not expose a COM port or LXSDF framing. The vendor's official app
  (TeleScan) is Windows-only, so the Mac implementation reproduces its HID
  initialization and decoder directly (see §7).

- **Camera:** any cheap overhead camera — a **phone used as a webcam** is ideal
  (a laptop cam can't easily point straight down). 720p is plenty. Precision does
  **not** come from the camera; it comes from visual feedback (see §7).

- **Laptop:** macOS. Runs everything. One machine — no router, no second PC.

---

## 4. System architecture

Everything runs on one laptop, connected to the arm and the EEG over USB. There
is no network, no router, no UDP.

```
  ┌─────────────────┐  USB HID (raw blocks)  ┌──────────────────────────────┐
  │  PolyG-I  EEG    │───────────────────────▶│           LAPTOP             │
  │  (8 channels)    │                        │                              │
  └─────────────────┘                        │  eeg_bridge ─▶ errp          │
                                              │       │          │           │
  ┌─────────────────┐  USB (video)           │       ▼          ▼           │
  │ overhead camera  │───────────────────────▶│   vision ─▶ orchestrator     │
  │ (phone/laptop)   │                        │              │   ▲           │
  └─────────────────┘                        │              ▼   │           │
                                              │           policy (IK)        │
                                              │              │               │
                                              └──────────────┼───────────────┘
                                                             │ USB serial
                                                             ▼  (angle commands)
                                                    ┌──────────────────┐
                                                    │     Arduino       │
                                                    │  6 servo motors   │
                                                    └──────────────────┘
```

- **Arduino** is "dumb": it receives target joint angles over serial and moves
  the servos there smoothly. No intelligence.
- **Laptop** is the brain: camera vision, target selection, inverse kinematics,
  EEG acquisition + ErrP detection, and the shared-autonomy loop.
- **EEG** contributes only the veto signal.

**Compatibility path:** `windows_eeg_server.py` remains for a different supported
Windows/LXSDF device if one is introduced later. It is neither needed nor used
for this PID `0x0010` PolyG-I; its native HID protocol is already working.

---

## 5. The full task, step by step

This is exactly what `orchestrator.py` does per object (function
`do_pick_and_place`), inside a loop that clears the whole table (`run_trial`):

```
 1. DETECT      camera finds candidate objects + the arm tip (workspace cm)
 2. SELECT      policy picks the best remaining target (naive: nearest first)
 3. COMMIT      arm moves to HOVER above that object, where the human can see it
                → the instant of commit is TIMESTAMPED (action onset)
 4. VETO WINDOW read the EEG for [onset-0.2s , onset+0.8s], run the ErrP detector
                   • ErrP detected  → this is the wrong object:
                                       drop it, return home, wait, reselect (→2)
                   • no ErrP        → the human is fine with it, continue
5. ALIGN       visual servoing: measure tip→object error from the camera and
                nudge until the tip is over the object; preserve that corrected
                command through descent (failure to see/converge aborts the grasp)
 6. GRASP       descend, close the claw, lift
 7. VERIFY      is the object gone from its old spot? (camera check, no force
                sensor needed) — if not, reopen and retry
 8. TRANSPORT   carry to the delivery zone
 9. PLACE       descend, release, retract
10. HOME        return, then repeat for the next object until the table is clear
```

With **real EEG** there is no keyboard at all — the brain drives step 4 directly.
In the **mock** demo (no headset), you press `y`/`n` to stand in for the brain,
and pressing `y` injects a realistic synthetic ErrP so the *real* detector code
still runs end-to-end.

---

## 6. Repository layout — every file explained

```
brainToArm/
├── README.md                      ← this document
├── requirements.txt               ← Python deps (pip install -r)
├── dashboard/                     ← local real-time EEG web interface (React)
├── example.cpp                    ← the ORIGINAL hardcoded servo demo (obsolete;
│                                    kept only as a reference to the motor map)
├── firmware/
│   ├── arm_controller/
│   │   └── arm_controller.ino     ← Arduino firmware (flash this)
│   └── esp32_camera_diagnostic/   ← wired OV2640 USB frame/signal proof
└── laptop/                        ← everything that runs on the laptop (Python)
    ├── config.py                  ← ALL tunable constants in one place
    ├── orchestrator.py            ← the main shared-autonomy loop + preflight
    │
    ├── arm_serial.py              ← laptop → Arduino serial link
    ├── kinematics.py              ← inverse kinematics (x,y,z → servo angles)
    ├── planar_pick.py             ← verified fixed-base camera pick/place loop
    ├── policy.py                  ← which object to pick + reach planning
    │
    ├── eeg_bridge.py              ← live EEG acquisition → timestamped buffer
    ├── eeg_dashboard.py           ← PolyG-I live UI API + one-command launcher
    ├── polyg_hid.py               ← native PolyG-I start/stop + block decoder
    ├── lxsdf.py                   ← LXSDF T2A packet parser/encoder
    ├── eeg_detect.py              ← live PolyG-I HID bench diagnostic
    ├── windows_eeg_server.py      ← legacy/alternate-device DLL → TCP bridge
    ├── errp.py                    ← ErrP detector (the brain-signal classifier)
    ├── cognitive_load.py          ← continuous TAR + adaptive autonomy allocator
    ├── record_errp.py             ← collect labeled ErrP training data
    ├── errp_train.py              ← train the ErrP model from that data
    │
    ├── vision.py                  ← camera → objects + arm tip (markerless)
    ├── wrist_vision.py            ← named wrist webcam + tape/target geometry
    ├── wrist_search.py            ← collision-checked motor2/3/4 view planner
    ├── calibrate_workspace.py     ← click 4 points → pixel↔cm mapping
    ├── camera_calibrate.py        ← optional lens (fisheye) calibration
    ├── calibrate_esp32_color.py   ← neutral-scene OV2640 spatial color map
    ├── capture_esp32_camera.py    ← capture/validate one wired OV2640 frame
    │
    ├── sim.py                     ← toy world so the mock runs coherently
    ├── validate.py                ← static config sanity checks
    ├── arm_jog.py                 ← interactive arm bring-up / calibration
    └── test_pipeline.py           ← hardware-free unit tests
```

### Firmware

- **`firmware/arm_controller/arm_controller.ino`** — Runs on the Arduino. Opens
  serial at 115200 baud and accepts newline-terminated ASCII commands:
  - `A a1 a2 a3 a4 a5 a6` — set target angles for servos 1–6; a value
    of `-1` means "leave that joint's target unchanged." Replies `OK`.
  - `P` → `PONG` (ping), `S` → current angles.
  - When all joints reach their targets it prints `DONE` once.
  Motion is **slew-limited** (a max degrees-per-tick) so the arm moves smoothly
  and never snaps. All the intelligence is on the laptop; this file only moves
  servos and reports back.

### The orchestration and arm control

- **`config.py`** — The single source of truth for every hardware- and
  task-dependent number: serial ports, arm link lengths, servo calibration,
  pick/place heights, the delivery location, EEG source/rate/channel map, ADC
  scaling, ErrP parameters, camera calibration, detection method. **When real
  hardware doesn't match an assumption, you edit a constant here — never code.**
  Fully commented with how to confirm each value. See §11.

- **`orchestrator.py`** — The main program. Wires everything together and runs
  the loop from §5. Key functions:
  - `preflight()` — before running, checks the arm acks, the EEG is actually
    streaming samples, the camera sees objects, and the config is self-consistent
    (via `validate.py`). Prints GO / NO-GO; refuses to run on a dead subsystem
    (override with `--force`).
  - `read_veto()` — uses the timestamp captured immediately before the visible
    reach begins, waits, cuts the ErrP epoch by that timestamp, and refuses to
    decide from an incomplete epoch.
  - `do_pick_and_place()` — one object, the full sequence (steps 3–10).
  - `run_trial()` — repeatedly clear the table until empty or `max_objects`.
  Run with `--auto` to remove the human (arm accepts everything).

- **`arm_serial.py`** — `ArmSerial` class: opens the Arduino port (auto-detects
  `/dev/cu.usbmodem*` etc. only when the candidate is unambiguous), sends validated angle commands, requires `OK` and
  `DONE`, drives the gripper, and homes. Mock mode is selected explicitly with
  `ARM_MOCK=True`; when a real arm is requested, a missing port, bad reply, or
  motion timeout stops the run instead of pretending the command succeeded.

- **`planar_pick.py`** — The physically verified side-camera path. Although base
  yaw is operational again, this calibration deliberately holds it at 90°. It
  uses the MacBook side camera directly and detects the current tabletop
  object against a clean background, maps its pixel x-coordinate through the
  measured local elbow Jacobian, and executes the physically verified sequence:
  fully open, pitch 180° / level roll 170°, descend in 2° steps, close, lift verification,
  short transport, release, and displacement verification. The base is locked;
  a missing object, stale camera background, failed lift, or failed
  displacement stops the sequence rather than continuing blindly.

- **`kinematics.py`** — Inverse kinematics. `solve(x, y, z)` turns a workspace
  point (centimeters, origin at the arm base) into the 6 servo commands: base
  yaw to face the target, then a 2-link (upper arm + forearm) law-of-cosines
  solution for shoulder and elbow, wrist kept pointing down for a top grasp.
  Uses the link lengths and servo calibration from `config.py`. Unreachable
  points are rejected rather than silently mapped to a different pose.
  `reachable(x,y,z)` reports feasibility.

- **`policy.py`** — `Policy` class, two jobs kept separate:
  1. **Target selection** (the part the brain corrects): rank candidate objects,
     naive prior = nearest first (so it *will* sometimes pick wrong — that's the
     point). `reject()` records a vetoed **position** (not a detection id, because
     markerless detection renumbers objects every frame). Vetoes last for one
     object-selection cycle; persistent spatial preference is disabled unless the
     application supplies a stable task meaning and explicitly enables it.
  2. **Reach planning:** `target_to_angles()` calls the IK. This is the seam where
     a trained reinforcement-learning reacher could later replace IK without
     touching anything else.

### EEG acquisition and the brain signal

- **`eeg_bridge.py`** — `EEGBridge` runs a background thread that fills a
  **timestamped** ring buffer of EEG samples. Batched transport
  reads are reconstructed at the configured device sample interval instead of
  assigning one arrival timestamp to every packet. Four
  interchangeable sources selected by `config.EEG_SOURCE`:
  - `"hid"` — the connected PID `0x0010` PolyG-I. Native macOS `hidapi`, the
    D1WD10 initialization sequence, and 1,024-byte/16-channel block decoder.
  - `"mock"` — synthetic 8-channel theta/alpha signal + noise, with an
    injectable ErrP-like burst for testing. Goes through the **real** LXSDF
    encode+parse path so nothing downstream is special-cased.
  - `"serial"` — compatibility path for LAXTHA CDC variants, not PID `0x0010`.
  - `"tcp"` — compatibility path for a Windows DLL bridge.
  Provides `mark_onset()` (timestamp the action) and `wait_and_epoch(onset)`
  (block until the response has developed, then return the time-aligned epoch) —
  this timestamp-based epoching is what keeps ErrP correctly aligned on real
  hardware despite thread jitter.

- **`polyg_hid.py`** — The verified real-device path. It enforces exact-one-device
  discovery, sends STOP → 16 physical channels → 256 Hz → EEG-group PGA → START,
  checks every HID write, removes the marking bit, decodes offset-binary words,
  converts them to ADC-input mV, and sends STOP during cleanup after any error.

- **`lxsdf.py`** — The compatibility/mock LXSDF T2A parser, written from LAXTHA's official
  spec. Packets start with sync bytes `0xFF 0xFE`, an 8-byte header (including a
  packet counter), then 2 bytes per channel (high, low). `LXSDFParser.feed()`
  takes raw bytes from any transport, resynchronizes on the sync bytes,
  **auto-detects the channel count**, counts dropped packets, and yields decoded
  samples. `build_packet()` encodes packets (used by the mock and tests, so the
  exact same parser is exercised with and without hardware).

- **`eeg_detect.py`** — Bench tool. It finds PID `0x0010`, starts a bounded live
  acquisition, reports report cadence and all eight channel ranges, then stops
  cleanly. `--port` retains an explicit CDC compatibility probe.

- **`windows_eeg_server.py`** — Legacy/alternate-device path. Runs on Windows, calls the LAXTHA
  `LXSMWD12.dll`, and forwards the raw LXSDF bytes to the laptop over TCP. The
  ctypes function names follow LAXTHA's documented API pattern; confirm them
  against the LXSMWD12 developer manual if that compatibility path is needed.

- **`errp.py`** — `ErrPDetector`: the brain-signal classifier. Given all eight
  EEG channels around an action onset, it returns `p_error` ∈ [0,1]
  (probability the human judged
  the action wrong). Pipeline: **band-pass 1–10 Hz** → baseline-correct →
  spatially average the configured eight channels → find the
  strongest **sustained negative deflection** in the post-onset region (a sliding
  ~150 ms window over roughly the first 0.6 s, robust to person-to-person latency
  differences). Two backends:
  - `"baseline"` — zero-training heuristic; scores the deflection as a **z-score
    against the person's own resting EEG noise**, so it works without knowing the
    device's exact microvolt scaling. Runs on day one.
  - `"model"` — a trained scikit-learn classifier for higher accuracy.
  Saved models include sampling, channel, band, and epoch metadata; a mismatched
  runtime configuration is rejected instead of producing an invalid prediction.
  `update_baseline()` calibrates the resting noise; `fit()`/`save()` train.

- **`cognitive_load.py`** — continuous mental-workload and adaptive-autonomy
  path. Every second it computes a Welch PSD on the latest two-second window,
  averages theta power (4–8 Hz) from CH1–CH4, divides it by alpha power
  (8–13 Hz) from CH8, and normalizes TAR against the participant's eight-second
  session rest baseline: `(current TAR - rest TAR) / rest TAR`. An EMA prevents
  one noisy window from changing authority abruptly, and a ±10% rest deadband
  prevents baseline jitter from changing checkpoint density. Higher normalized TAR
  raises robot weight and the ErrP veto threshold and spaces out application
  checkpoints; lower TAR gives the human/ErrP more weight. ErrP probability is
  still calculated at every action event, and a very strong response remains an
  override. Invalid/flat PSD windows fail toward minimum robot authority.

- **`record_errp.py`** — Collects training data. A goal is fixed for the session,
  then the script drives the arm through
  deliberately correct and deliberately wrong actions while the subject watches;
  since *we* chose right vs wrong, each epoch is **auto-labeled** (no button
  pressing). Saves one CSV per epoch plus a `labels.csv`; repeated sessions append
  unused epoch numbers and never overwrite earlier trials.

- **`errp_train.py`** — Validates a recorded folder, reports stratified
  cross-validated balanced accuracy when the dataset is large enough, fits the
  final `ErrPDetector` model, and saves `errp_model.pkl`. Then set
  `ERRP_BACKEND="model"` in config to use it live.

### Vision

- **`vision.py`** — `Vision` turns camera frames into `Detection` objects
  (label + workspace x,y) and locates the arm tip, all in centimeters via a
  pixel→world homography. **Markerless by default** (`OBJECT_METHOD="bgsub"`):
  snapshot the empty table once (`learn_background()`), then objects and the arm
  are foreground; arm-tip candidates are the contour points farthest from the
  base, disambiguated by proximity to the expected target during servoing.
  Also supports `"yolo"` (semantic detection, needs `ultralytics`), `"hsv"`
  (color blobs), and `"aruco"` (printed markers) if you prefer. `location_clear()`
  implements the markerless grasp verification. Falls back to a fixed mock scene
  (from `sim.py`) when `CAM_MOCK=True`, so the loop runs with no camera.

- **`calibrate_workspace.py`** — The one camera step you can't skip: click 4
  points of known real-world centimeters in the live image; it computes and
  prints the pixel↔cm homography to paste into `config.py`. Four points define the
  transform but cannot independently measure between-point accuracy; use 6+ if
  you want a meaningful fit-consistency check.

- **`camera_calibrate.py`** — Optional. Removes lens/fisheye distortion (only
  matters if a cheap wide lens bends straight lines near the edges), using a
  printed checkerboard.

- **`wrist_vision.py`** — Robot-independent eye-in-hand perception. On macOS it
  asks AVFoundation for `AVerMedia PW315` by name through ffmpeg, avoiding
  unstable numeric indices. Compact blue-left/red-right tape contours define a
  gripper pose; their midpoint is the grasp point and their connecting line is
  the jaw axis. A separate colored target yields pixel displacement in both
  image and jaw coordinates plus an optional elongated-object roll error.
  Pair geometry, contour compactness, expected lower-center placement, exposure,
  and contrast are all gated. It never imports or opens `ArmSerial`, so live
  framing and hue setup cannot move the robot.

- **`wrist_search.py`** — Motion layer for a target outside the current frame.
  It coordinates shoulder, elbow, and wrist pitch (motors 2/3/4) to generate
  diverse camera positions and directions. Planar forward kinematics rejects
  endpoints or complete firmware-slew paths that violate table, base-column, or
  non-adjacent-link clearance. Search stops on the first full-frame target and
  failure retraces the verified route. Physical `--run` remains locked until
  servo zero/direction, link lengths, and the search model are explicitly marked
  calibrated; `--target-hue` selects a known object color without a spatial crop.

### Support and testing

- **`sim.py`** — A toy world used only in mock mode. Models the systematic error
  a cheap camera + rough IK would produce (a fixed position bias + noise) so the
  visual-servoing loop demonstrably **converges** without hardware, and tracks
  which objects have been picked so the mock scene shrinks coherently.

- **`validate.py`** — Static config checks (servo array lengths, home pose within
  limits, EEG channel-map bounds, ErrP band below Nyquist, place location
  reachable, height ordering, valid enum choices). Catches setup mistakes before
  they cause a confusing mid-run failure. Run standalone or via preflight.

- **`arm_jog.py`** — Interactive bring-up console. Jog individual joints, run IK
  to a point, open/close the gripper — used to verify and fix the servo
  direction/offset and link-length constants against the real arm.

- **`test_pipeline.py`** — Hardware-free unit tests: LXSDF encode→parse roundtrip
  / resync / drop counting, IK output in range and aiming correctly, ErrP
  separating error vs clean epochs, and a full mock grasp+place removing an
  object. Run this any time; everything should pass.

---

## 7. Key technical decisions (and the traps they avoid)

- **Involuntary ErrP, not conscious motor imagery.** Reading a reflex needs no
  user training and almost no cognitive load; asking the user to "think a
  command" is slow, tiring, and unreliable on an 8-channel consumer device.

- **The task must stay ambiguous.** If detection could perfectly identify "the
  small nail," you'd just hardcode the choice and the brain would be pointless.
  The demo deliberately presents a genuine tie (big vs small nail) so the veto
  carries real information. Detection therefore reports **positions** and leaves
  the **choice** open.

- **We read the HID device ourselves, not through TeleScan.** The installed
  D1WD10 DLL and official manual revealed the exact commands (`01 01 00` starts;
  `01 00 00` stops), 16-channel layout, offset-binary conversion, marking bit,
  and ADC voltage coefficient. Native `hidapi` reproduces that behavior on
  macOS; no Windows, CrossOver, screen scraping, or serial emulation is involved.

- **No router / no UDP.** A previous student's rig needed a router only to
  synchronize three separate PCs over UDP. We run everything on one machine, so
  it's all local serial + (at most) a localhost socket.

- **A cheap camera is fine because precision comes from feedback.** Visual
  servoing closes the loop: the arm watches its own tip approach the object and
  corrects until the error is small, cancelling both camera- and IK-calibration
  error. The corrected command is retained for descent; failure to observe or
  converge never falls through to a blind grasp. A 720p phone cam over a 30 cm
  table already resolves ~0.5 mm/pixel.

- **Markerless, zero props.** Background subtraction needs only one snapshot of
  the empty table — no stickers, no printed markers, no special backdrop.

- **Grasp verified visually, no force sensor.** After lifting, the arm retracts
  and checks whether the object's spot is now empty; if not, it retries.

- **Timestamp-based ErrP epoching.** The action onset is timestamped and the EEG
  epoch is cut by time, not by sample count, so the brain response stays aligned
  regardless of thread timing jitter — the thing that most often breaks ErrP
  detection in practice.

- **Scale-invariant ErrP decision.** The heuristic scores the deflection as a
  z-score against the person's own resting EEG noise, so it works before you've
  calibrated the device's absolute microvolt scaling.

---

## 8. Current status: what works, what is assumed

**Everything is implemented and runs today in mock (no hardware).** The full
shared-autonomy loop — detect, select, hover, ErrP veto, visual-servo align,
grasp, verify, transport, place, repeat until the table is clear — executes end
to end, and all unit tests pass. The data pipeline (record → train → load model)
is verified. All Python modules compile.

The parts that are written against **documented assumptions** and can only be
*confirmed* (not written) once the physical hardware is present:

| Area | Assumed value / behavior | How to confirm |
|------|--------------------------|----------------|
| EEG transport | **RESOLVED:** native USB HID, VID `0x0F1F` PID `0x0010` | `eeg_detect.py` passes on this Mac |
| EEG start/stop | **RESOLVED:** D1WD10 vendor sequence | bounded captures repeatedly start, stream, and stop |
| EEG framing | **RESOLVED:** 1,024-byte block, 32 rows × 16 physical channels; first 8 are EEG | decoder matches DLL, official manual, and hardware |
| EEG rate / channel map | **RESOLVED transport:** selector 8 = 256 Hz; EEG source is physical channels 1–8 | confirm electrode montage before setting `EEG_CONFIG_VERIFIED=True` |
| Arm geometry | link lengths `L_*`, servo offsets/directions | `arm_jog.py` |
| Camera mapping | 4-point pixel↔cm homography | `calibrate_workspace.py` |
| ErrP model | trained classifier (baseline heuristic works meanwhile) | `record_errp.py` + `errp_train.py` |

If every assumption happens to match, the system runs unmodified. Realistically
you will tune a handful of `config.py` constants using the helper tools above —
and only constants, never code.

---

### Live PolyG-I monitor

The local dashboard is the recommended EEG bring-up surface before connecting
the robot arm. It owns the HID device in one Python process and shows all eight
EEG channels in filtered ADC-input mV, measured stream rate, fixed-scale PSD,
exact window statistics, ADC rail warnings, CSV recording, and event markers.
The default smooth renderer buffers only the display by 0.45 seconds and advances
the waveform at the browser refresh rate; choose the 0.08-second low-latency mode
when immediate feedback matters more than perfectly even motion. Neither mode
interpolates or changes recorded samples.

```bash
pip install -r requirements.txt
cd dashboard && npm install && cd ..       # first run only
python3 laptop/eeg_dashboard.py
```

The launcher opens `http://localhost:3000` and serves the device API only on
`127.0.0.1:8765`. Press **측정 시작** in the browser. `Ctrl-C` shuts down the
stream and UI deterministically. CSV files are kept under the ignored local
`recordings/` directory and can be downloaded from the interface.

Every channel uses the same user-selected, fixed Y-axis and a visible 0 mV line;
there is no moving auto-scale or per-window recentering. The live path applies a
stateful 60 Hz notch and 4th-order 0.5–45 Hz band-pass. The D1WD10 coefficient
supports accurate **ADC-input mV**, but not electrode-input μV: the fixed analog
front-end gain is not published or independently calibrated. RMS, peak-to-peak,
DC offset and clipping are measurements, but they are not electrode impedance or
clinical interpretations.

---

## 9. Run it right now with zero hardware

The checked-in source is now `EEG_SOURCE="hid"` for the connected device. For a
hardware-free demo, temporarily set it to `"mock"` in `laptop/config.py`, then:

```bash
cd brainToArm
python3 laptop/test_pipeline.py         # unit tests — everything should pass
python3 laptop/orchestrator.py --auto   # full loop, no human: clears the table
python3 laptop/orchestrator.py          # full loop; press y/N to play the brain
```

In the interactive run, pressing `y` (veto) injects a realistic synthetic ErrP;
the **real** detector flags it (P≈0.8–1.0) and the arm reselects, while accepted
targets read P≈0.05–0.2. It picks and "delivers" each object in turn.

(Only `pyserial`/`numpy`/etc. are needed for hardware; the mock path runs on a
bare Python 3.9+ interpreter. `pip install -r requirements.txt` for the rest.)

---

## 10. Bring up the real hardware — exact steps

Do these in order; each step verifies the one assumption it owns, so by the time
you run the loop nothing is guessed.

```bash
pip install -r requirements.txt
```

**1 — Arm.** The connected Uno/CH340 is auto-discovered while the ESP32 camera's
stable `/dev/cu.usbserial-0001` CP2102 port is explicitly excluded. The Uno has
`firmware/arm_controller/arm_controller.ino` flashed, responds to `PONG`,
reports all six angles, and completed the
bounded base test `90° → 95° → 90°`. Measure the arm's link lengths with a ruler
into `L_BASE_HEIGHT / L_UPPER / L_FORE / L_HAND`. Then:
```bash
python3 laptop/arm_jog.py --upload # after changing home_pose.h: compile, flash,
                                  # verify the Uno, then enter the jog console
python3 laptop/arm_jog.py        # later runs verify the flashed HOME first;
                                 # reads and preserves the current pose; use h
                                 # explicitly when a home move is wanted
                                 # jog joints; fix SERVO_DIRECTION/OFFSET if a
                                 # joint runs backwards or neutral is off.
                                 # try `ik 0 15` and check the tip goes there.
```
`ARM_MOCK=False` now selects the physical board. `ARM_CALIBRATED=False` still
blocks the autonomous preflight. Set it to `True` only after geometry, offsets,
directions, and mechanical safe limits are confirmed.

The single source of truth for the power-on and `h` home pose is
`firmware/arm_controller/home_pose.h`. Change only the number on the named
`ARM_HOME_SERVO_n` line, connect the Uno, and run
`python3 laptop/arm_jog.py --upload`. The upload resets the Uno and moves the
physical arm to the new pose, so clear its workspace first. Every real
`ArmSerial` connection—including autonomous and recording code—queries the HOME
pose compiled into the Uno before accepting commands. Missing query support or
a mismatch stops the run and names the upload command; stale firmware can no
longer continue silently.

**2 — EEG.** Plug in the PolyG-I and:
```bash
python3 laptop/eeg_detect.py --seconds 5
python3 laptop/eeg_bridge.py
```
Both commands must show live 8-channel data. `EEG_SOURCE="hid"` is already set
for this unit. ErrP uses CH1–CH8. Continuous load uses theta from CH1–CH4 and
alpha from CH8, so verify those physical electrode assignments, signal response,
and the sustained rate with the headset mounted, then set
`EEG_CONFIG_VERIFIED=True`. At trial start remain still during the printed
eight-second resting-TAR calibration. The gate intentionally
distinguishes “USB input works” from “this participant's montage is trustworthy.”

**3 — Camera.** Mount a phone overhead looking straight down. Set `CAM_MOCK=False`.
```bash
python3 laptop/calibrate_workspace.py   # click 4 known cm points -> homography
```
Paste the printed points into `config.py`. (Optionally run `camera_calibrate.py`
if the lens visibly bends straight lines.) The calibration tool also prints the
required `CAM_CALIBRATED=True` confirmation flag.

For the mounted USB wrist webcam, keep the arm unplugged and run:

```bash
python3 laptop/wrist_vision.py --snapshot  # named AVerMedia proof + one annotated frame
python3 laptop/wrist_vision.py --live      # placement, tape, target, exposure UI
python3 laptop/wrist_search.py             # inspect gates/joints; moves nothing
```

The mounted view may show the fingers entering through the bottom edge; those
contours are intentionally accepted. Global white-table clipping is reported but
does not block alignment while the colored tape/object pixels remain usable. A
near-total (`>92%`) clipped frame or low-contrast frame still fails closed. The
automatic target mode ignores skin/red and blue hues so a nearby hand and the
finger tapes are not selected as the object; left-click hue lock is available
when an explicit target color is needed. Target location is never restricted to
a center/lower ROI: the full frame remains searchable so the motion layer can
search new collision-checked 2/3/4 viewpoints whenever the chosen color is absent. The
live view is perception-only. Physical image-Jacobian and approach-depth
calibration are still required after reconnecting the Uno before autonomous
motion is enabled.

The current competition target is fixed to **vivid yellow** in `config.py` using
OpenCV HSV `H=22..38, S=140..255, V=100..255`. This explicit range is searched
over the entire frame and avoids the red/blue finger tapes and ordinary wood/skin
tones. Clicking in the preview can still replace it for a later target color.

For the alternate wired OV2640 hardware proof, flash
`firmware/esp32_camera_diagnostic` and run:

```bash
python3 laptop/calibrate_esp32_color.py  # empty neutral scene; repeat after remount/lighting change
python3 laptop/capture_esp32_camera.py
```

A passing result currently reports `CAMERA_READY pid=0x26 psram=0 controls=0`
followed by `FRAME 160 120 ... 4` and writes a color-corrected image under
`data/vision/`. Pixel format 4 is OV2640 hardware JPEG. The Mac drops frames
that fail full decoding, clipping/background checks, or the measured JPEG
8-pixel block-discontinuity limit. This proves the camera and frame bus; it does
not yet replace the calibrated overhead-camera source in `vision.py`.

**4 — ErrP model.** The baseline heuristic already works, but for best accuracy:
```bash
python3 laptop/record_errp.py data/errp --trials 40   # arm does right/wrong;
                                                       # subject just watches
python3 laptop/errp_train.py data/errp                # -> errp_model.pkl
```
Set `ERRP_BACKEND="model"` in `config.py`.

**5 — Run.**
```bash
python3 laptop/orchestrator.py
```
Preflight self-checks every subsystem and prints GO/NO-GO. With real EEG there is
no keyboard — the brain drives the veto directly.

---

## 11. config.py — the single tuning surface

Grouped constants (see the file for full comments):

- **Arm serial:** `ARM_PORT`, `ARM_PORT_EXCLUDE`, `ARM_BAUD`, `ARM_MOCK`, `ARM_CALIBRATED`, `N_JOINTS`, `SERVO_PINS`, `JOINT_NAMES`, `HOME_POSE`, joint index
  names, `GRIP_OPEN/CLOSED`.
- **Arm geometry & calibration:** `L_BASE_HEIGHT/L_UPPER/L_FORE/L_HAND`,
  `SERVO_OFFSET/DIRECTION/MIN/MAX`.
- **Pick-and-place:** `Z_APPROACH/GRASP/LIFT/PLACE`, `PLACE_LOCATION`,
  `PLACE_ZONES`, `GRASP_VERIFY`, `GRASP_RETRIES`.
- **EEG:** `EEG_SOURCE`, `EEG_HID_*`, `EEG_CHANNEL_MAP`, `EEG_CHANNELS`,
  measured `EEG_FS`, relative scaling, plus legacy `EEG_PORT/BAUD`/`EEG_TCP_*`.
- **ErrP:** `ERRP_BACKEND`, `ERRP_MODEL_PATH`, `ERRP_CHANNELS`, `ERRP_USE_CAR`,
  `ERRP_WINDOW_S`, `ERRP_BASELINE_S`, `ERRP_BAND`, `ERRP_THRESHOLD`.
- **Cognitive load/autonomy:** `COG_THETA_CHANNELS`, `COG_ALPHA_CHANNELS`,
  `COG_*_BAND`, window/rest/update/EMA controls, `AUTONOMY_ROBOT_*`, adaptive
  threshold gain, application stride, and strong-ErrP override threshold.
- **Vision:** `CAM_INDEX`, `CAM_MOCK`, `OBJECT_METHOD` (`bgsub`/`yolo`/`hsv`/
  `aruco`), `BGSUB_THRESH`, `OBJECT_MIN_AREA`, `ARM_MIN_AREA`, `YOLO_*`,
  aruco ids, `OBJECT_HSV`, `CAM_CALIB_IMAGE_PTS/WORLD_PTS`, `CAM_MATRIX/DIST`.
- **Wrist vision:** stable `WRIST_CAMERA_NAME`, frame/FPS settings, blue/red HSV
  ranges, marker pair geometry, expected gripper center, colored-target bounds,
  alignment tolerance, and exposure/contrast gates.

**Rule:** hardware doesn't match? Change a constant here, not code.

---

## 12. Testing and verification

- **`python3 laptop/test_pipeline.py`** — must pass: PolyG-I command/decoder,
  LXSDF compatibility, IK, all-channel ErrP, TAR/autonomy direction, and full
  mock pick-and-place tests.
- **`python3 laptop/eeg_detect.py --seconds 5`** — physical HID start, live
  8-channel range/cadence report, and deterministic stop.
- **`python3 laptop/eeg_dashboard.py`** — local live waveform, spectrum,
  recording, marker, pause/resume, and device-status interface.
- **`cd dashboard && npm run build`** — production build check for the interface.
- **`python3 laptop/validate.py`** — currently reports the intentional
  `EEG_CONFIG_VERIFIED=False` live-montage gate; it prints `config OK` only after
  the electrode/rate validation described above (or in mock mode).
- **`python3 laptop/orchestrator.py --auto`** — full mock run after selecting
  `EEG_SOURCE="mock"`; the active HID configuration fails closed at preflight
  until its montage gate is cleared.
- **`python3 laptop/wrist_vision.py --snapshot`** — opens the AVerMedia by name,
  warms exposure, saves an annotated frame, and prints marker/target/quality
  state without opening the robot serial port.
- **`python3 laptop/wrist_search.py`** — reports the 2/3/4 search and calibration
  gates without hardware. Add `--run --target-hue H` only after the Uno is
  connected and the forward-kinematics safety model is physically verified.

---

## 13. Glossary

- **EEG (electroencephalography):** measuring the brain's electrical activity
  with electrodes on the scalp.
- **BCI (brain–computer interface):** a system that uses brain signals as input.
- **ErrP (Error-Related Potential):** an involuntary EEG response, ~250–450 ms
  after a person perceives an error, strongest over fronto-central scalp sites.
  This project's core signal.
- **TAR (theta/alpha ratio):** here, mean 4–8 Hz power from CH1–CH4 divided by
  8–13 Hz power from CH8. Its session-rest-relative change drives autonomy
  allocation; it is not treated as a clinical diagnosis.
- **Shared autonomy:** human and autonomous system jointly control a task; here
  the AI does the work and the human contributes veto corrections.
- **Motor imagery:** consciously imagining a movement to produce a detectable EEG
  pattern — the *opposite* approach to ErrP; deliberately **not** used here.
- **Inverse kinematics (IK):** computing joint angles that place the end effector
  at a desired position.
- **Visual servoing:** closed-loop control that drives the arm using visual error
  between the tip and the target.
- **Homography:** a projective mapping between two planes; here camera pixels ↔
  table centimeters.
- **CAR (Common Average Reference):** subtracting the average across all
  electrodes to cancel shared noise.
- **LXSDF:** LAXTHA's serial packet format for streaming multi-channel biosignal
  data.
- **Slew limiting:** capping how fast a servo moves per control tick for smooth
  motion.

---

## 14. Sources and prior art

- [Evaluation of alpha/theta and theta/alpha workload indexes](https://pmc.ncbi.nlm.nih.gov/articles/PMC9149374/)
- [Reliability of EEG mental-workload indexes across electrode/preprocessing configurations](https://pmc.ncbi.nlm.nih.gov/articles/PMC9920504/)
- [ErrP in a virtual pick-and-place shared-control experiment](https://pubmed.ncbi.nlm.nih.gov/38083754/)

- Device / protocol: [PolyG-I](https://www.laxtha.com/ProductView.asp?Model=PolyG-I) ·
  [LXSDF spec repo](https://github.com/LAXTHA/LXSDF) ·
  [LXSDF T2A packet layout](http://laxtha.net/packet-lxsdf-t2a/) ·
  [TeleScan](https://github.com/LAXTHA/TeleScan) ·
  [LAXTHA Windows DLL API pattern](http://laxtha.net/api-for-windows-ubpulse-h3/)
- Arm model: [MakerWorld Robotic Arm with Servo / Arduino](https://makerworld.com/ko/models/1134925-robotic-arm-with-servo-arduino)
- Prior rig this replaced (multi-PC UDP/router): [automated_EEG_experiments](https://github.com/eeuunn/automated_EEG_experiments)
- Concept prior art — using ErrP to correct a robot in real time: MIT CSAIL
  (DelPreto & Rus), "brain-controlled robot correction."

---

*Everything above reflects the code as it stands. The system is complete and
runs today in simulation; connecting the real arm, EEG, and camera is a matter of
confirming the assumptions in §8 and §10 using the provided helper tools.*
