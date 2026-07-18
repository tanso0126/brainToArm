"""Shared-autonomy main loop — the point of the whole project.

  vision  : where are the arm tip and candidate objects (markerless)
  policy  : pick a target, plan the reach (IK; RL-ready)
  arm     : move the servos
  eeg/errp: while the arm commits to a target, watch for the human's brain
            saying "wrong one" — and if it does, veto and reselect.

Full task per accepted target: hover -> veto window -> visual-servo align ->
grasp -> lift -> verify pickup -> transport -> place -> home. See run_trial.

Autonomy does 99% (find, reach, grasp). The brain contributes ~1 bit at the
moment of ambiguity: no. That bit is what pure AI + camera can't get, because
"which object did the human actually want" lives only in the human's head.

Run with everything mocked (no hardware, no headset):
    python orchestrator.py
It will pick the "big nail", you (as the mock brain) veto, it moves to the next.

Go live by flipping config: EEG_SOURCE='serial'|'tcp', and construct
Vision(mock=False), ArmSerial() with a real board. The loop code is unchanged.
"""
import time
import sys
import math

import config
import sim
from arm_serial import ArmSerial
from eeg_bridge import EEGBridge
from errp import ErrPDetector
from vision import Vision
from policy import Policy

# how long after committing to a target we watch the brain for a veto
VETO_WATCH_S = config.ERRP_WINDOW_S
# refractory pause after a veto so the prior ErrP clears before the next action
REFRACTORY_S = 0.7
# keyboard stands in for the brain ONLY when the EEG is mocked. With real EEG the
# brain drives directly — no keyboard, nothing to wear/press beyond the headset.
KEYBOARD_BRAIN = (config.EEG_SOURCE == "mock") and ("--auto" not in sys.argv)


def move_to(arm, policy, world_xy, z=0.0):
    """Open-loop IK move to a workspace point at height z. Keeps the mock world
    in sync so the mock camera sees a consistent tip."""
    angles = policy.target_to_angles(world_xy, z=z)
    arm.send_angles(angles)
    if getattr(arm, "mock", False):
        sim.WORLD.set_command(world_xy)
    arm.wait_done()


def servo_to_object(arm, vision, policy, target, z):
    """Closed-loop visual servoing at height z: measure tip->object error from the
    camera and nudge until the tip lines up over the object. This is why a cheap
    phone/laptop camera (and rough IK) is fine — precision comes from feedback,
    not optics. Systematic bias is cancelled iteratively."""
    if not config.SERVO_ENABLE:
        move_to(arm, policy, (target.x, target.y), z)
        return (target.x, target.y), None, True
    corr = [0.0, 0.0]
    err = None
    command_xy = (target.x, target.y)
    for it in range(config.SERVO_MAX_ITERS):
        command_xy = (target.x + corr[0], target.y + corr[1])
        move_to(arm, policy, command_xy, z)
        tip = vision.arm_tip(expected_xy=(target.x, target.y))
        if tip is None:                       # tip not seen this frame; retry
            continue
        ex, ey = target.x - tip[0], target.y - tip[1]
        err = math.hypot(ex, ey)
        print(f"    [servo] iter {it}: tip=({tip[0]:.1f},{tip[1]:.1f}) err={err:.2f}cm")
        if err <= config.SERVO_TOL_CM:
            return command_xy, err, True
        corr[0] += config.SERVO_GAIN * ex
        corr[1] += config.SERVO_GAIN * ey
    return command_xy, err, False


def grasp_object(arm, vision, policy, target, grasp_xy=None):
    """Descend, close, lift, and verify the object was actually picked up.
    Retries the grasp if verification says the object is still on the table."""
    grasp_xy = grasp_xy or (target.x, target.y)
    for attempt in range(config.GRASP_RETRIES + 1):
        move_to(arm, policy, grasp_xy, config.Z_GRASP)               # descend
        arm.gripper(open_=False); arm.wait_done()                    # close claw
        mock_grasped = True
        if getattr(arm, "mock", False):
            idx = sim.WORLD.index_of(target.label, target.x, target.y)
            mock_grasped = sim.WORLD.grasp(
                idx, (target.x, target.y), tolerance_cm=1.0)
        move_to(arm, policy, grasp_xy, config.Z_LIFT)                # lift
        if getattr(arm, "mock", False):
            sim.WORLD.lifted()                                       # object leaves table
        if not config.GRASP_VERIFY:
            return mock_grasped
        # retract sideways a bit so the arm isn't over the spot during the check
        move_to(arm, policy, (target.x * 0.5, target.y * 0.5), config.Z_LIFT)
        if vision.location_clear(target):
            print(f"    [grasp] verified — '{target.label}' picked up")
            return True
        print(f"    [grasp] FAILED (object still there), retry {attempt+1}")
        arm.gripper(open_=True); arm.wait_done()
        if getattr(arm, "mock", False):
            sim.WORLD.release()
    return False


def place_object(arm, policy, place_xy):
    """Transport and release at the delivery zone. Placing needs no visual servo
    (empty target spot) — open-loop IK is enough."""
    move_to(arm, policy, place_xy, config.Z_LIFT)     # transport at height
    move_to(arm, policy, place_xy, config.Z_PLACE)    # descend
    arm.gripper(open_=True); arm.wait_done()          # release
    if getattr(arm, "mock", False):
        sim.WORLD.release()
    move_to(arm, policy, place_xy, config.Z_LIFT)     # clear


def human_vetoes_mock(det):
    ans = input(f"    [brain?] arm is going for '{det.label}'. veto? (y/N) ").strip().lower()
    return ans == "y"


def read_veto(eeg, errp, onset):
    """Onset-locked veto read. Given the timestamp at which the arm began its
    visible commitment, let
    the response develop, then epoch [onset-baseline, onset+window] by TIMESTAMP
    (not sample count) so a real ErrP stays aligned regardless of thread jitter.
    Returns (is_veto, p_error)."""
    window = eeg.wait_and_epoch(onset)
    expected = int((config.ERRP_BASELINE_S + config.ERRP_WINDOW_S) * config.EEG_FS)
    minimum = int(expected * config.EEG_MIN_EPOCH_FRACTION)
    if len(window) < minimum:
        detail = f"; source error: {eeg.last_error}" if eeg.last_error else ""
        raise RuntimeError(
            f"incomplete EEG epoch ({len(window)}/{expected} samples){detail}")
    p_err = errp.p_error(window)
    return p_err >= errp.threshold, p_err


def do_pick_and_place(arm, eeg, errp, vision, policy, target):
    """One object, full sequence. Returns "done" | "veto" | "fail"."""
    # commit: hover above the object so the human SEES the intent (ErrP needs a
    # clearly perceived wrong action, so we pause at hover during the veto read).
    print(f"[act] committing to '{target.label}' at ({target.x:.1f},{target.y:.1f})")
    arm.gripper(open_=True); arm.wait_done()

    # The visible reach is the event the observer judges. Timestamp immediately
    # before motion begins; timestamping after the blocking move misses the ErrP.
    mock_veto = KEYBOARD_BRAIN and human_vetoes_mock(target)
    onset = eeg.mark_onset()
    if mock_veto:
        eeg.mark_error(VETO_WATCH_S, onset_t=onset)
    move_to(arm, policy, (target.x, target.y), config.Z_APPROACH)

    veto, p_err = read_veto(eeg, errp, onset)
    print(f"    [errp] P(error)={p_err:.2f} (thr={errp.threshold})")
    if veto:
        print(f"    -> BRAIN SAYS NO. veto '{target.label}', reselect.")
        policy.reject(target)
        arm.home(); arm.wait_done()
        time.sleep(REFRACTORY_S)          # let the ErrP clear before next action
        return "veto"

    print(f"    -> accepted '{target.label}'. pick-and-place.")
    grasp_xy, err, aligned = servo_to_object(
        arm, vision, policy, target, config.Z_APPROACH)
    if err is not None:
        print(f"    [servo] aligned to {err:.2f}cm")
    if not aligned:
        detail = "tip was not visible" if err is None else f"final error {err:.2f}cm"
        print(f"    [servo] FAILED ({detail}); refusing to descend.")
        arm.home(); arm.wait_done()
        return "fail"
    if not grasp_object(arm, vision, policy, target, grasp_xy=grasp_xy):
        print(f"    [grasp] gave up on '{target.label}'.")
        arm.home(); arm.wait_done()
        return "fail"
    place_object(arm, policy, config.PLACE_LOCATION)
    print(f"    [place] delivered '{target.label}' to {config.PLACE_LOCATION}")
    policy.confirm(target)
    arm.home(); arm.wait_done()
    return "done"


def run_trial(arm, eeg, errp, vision, policy, max_objects=None):
    """Continuously clear the table: re-detect, pick the best remaining target,
    veto-or-place, repeat until nothing's left (or max_objects delivered)."""
    policy.reset_trial()
    resting = eeg.snapshot(1.0)                # calibrate on a calm moment
    minimum = int(config.EEG_FS * config.EEG_MIN_EPOCH_FRACTION)
    if len(resting) < minimum:
        raise RuntimeError(
            f"not enough resting EEG for calibration ({len(resting)}/{config.EEG_FS})")
    errp.update_baseline(resting)

    delivered = 0
    while True:
        detections = vision.detect()
        if not detections:
            print("[trial] table clear.")
            break
        arm_xy = vision.arm_tip() or (0.0, 0.0)
        print(f"[scene] {len(detections)} object(s): " +
              ", ".join(d.label for d in detections))
        target = policy.choose(detections, arm_xy)
        if target is None:
            if policy.unreachable:
                labels = ", ".join(d.label for d in policy.unreachable)
                print(f"[trial] no eligible target; unreachable: {labels}. Stop.")
            else:
                print("[trial] every remaining object was vetoed. Stop.")
            break

        result = do_pick_and_place(arm, eeg, errp, vision, policy, target)
        if result == "done":
            delivered += 1
            # A veto answers "not for the current goal", not "never touch this
            # location". The next pick in a clear-table run is a new selection.
            policy.reset_selection()
            if max_objects and delivered >= max_objects:
                break
        elif result == "fail":
            return False
        # "veto" -> loop; the spot remains rejected for this selection cycle
    print(f"[trial] delivered {delivered} object(s).")
    return True


def setup_scene(vision):
    """Markerless background-subtraction needs one empty-table snapshot. Mock
    skips this. Two keypresses, no props."""
    if getattr(vision, "mock", False) or config.OBJECT_METHOD == "aruco":
        return
    input("[setup] clear the table (arm parked), then press Enter... ")
    vision.learn_background()
    input("[setup] place the objects, then press Enter... ")


def preflight(arm, eeg, vision):
    """Confirm every subsystem is actually live before running, so a bad cable
    or wrong config fails loudly HERE instead of mid-task. Returns True if go."""
    ok = True
    print("\n[preflight] checking subsystems...")

    # static config sanity
    import validate
    errs, warns = validate.validate()
    for w in warns:
        print(f"  config : warn — {w}")
    for e in errs:
        print(f"  config : ERROR — {e}")
    if errs:
        ok = False
    elif not warns:
        print("  config : OK")

    # arm serial
    if getattr(arm, "mock", False):
        print("  arm    : MOCK (ARM_MOCK=True)")
        if config.EEG_SOURCE != "mock":
            print("           ! real EEG with an explicitly mocked arm")
    else:
        arm.ping()
        r = arm.send_angles(config.HOME_POSE)
        arm.wait_done()
        print(f"  arm    : OK ({arm.port}) ping=PONG ack={r!r}")

    # eeg stream actually producing samples
    time.sleep(1.2)
    n = len(eeg.snapshot(1.0))
    if eeg.last_error is not None:
        print(f"  eeg    : FAIL — source stopped: {eeg.last_error}")
        ok = False
    elif n == 0:
        print("  eeg    : FAIL — no samples. Run eeg_detect.py; check EEG_SOURCE/port/baud.")
        ok = False
    else:
        exp = int(0.6 * config.EEG_FS)
        tag = "OK" if n >= exp else "LOW"
        print(f"  eeg    : {tag} ({n} samples/s, source={config.EEG_SOURCE}, "
              f"parser_ch={eeg.parser.total_channels})")
        if tag == "LOW":
            print("           ! fewer samples than EEG_FS suggests — check EEG_FS/baud")

    # camera / vision
    if getattr(vision, "mock", False):
        print("  vision : MOCK (CAM_MOCK=True)")
    else:
        try:
            dets = vision.detect()
            print(f"  vision : OK (method={config.OBJECT_METHOD}, {len(dets)} object(s) seen)")
            if not dets:
                print("           ! no objects detected — check lighting / background snapshot")
        except Exception as e:
            print(f"  vision : FAIL — {e}")
            ok = False

    print(f"[preflight] {'GO' if ok else 'NO-GO'}\n")
    return ok


def main():
    print("=== brainToArm shared-autonomy loop ===")
    force = "--force" in sys.argv
    arm = eeg = vision = None
    try:
        arm = ArmSerial()
        eeg = EEGBridge().start()
        errp = ErrPDetector()          # backend from config.ERRP_BACKEND
        vision = Vision()              # mock/real from config.CAM_MOCK
        policy = Policy()

        arm.home(); arm.wait_done()
        setup_scene(vision)
        if not preflight(arm, eeg, vision) and not force:
            print("Aborting. Fix the above, or pass --force to run anyway.")
            return False
        ok = run_trial(arm, eeg, errp, vision, policy)
        print(f"[run] {'complete' if ok else 'aborted'}")
        return ok
    except KeyboardInterrupt:
        print("\n[run] interrupted by user")
        return False
    except (RuntimeError, ValueError, TimeoutError, OSError) as exc:
        print(f"[run] aborted: {type(exc).__name__}: {exc}")
        return False
    finally:
        if eeg is not None:
            eeg.stop()
        if arm is not None:
            try:
                arm.home()
                arm.wait_done()
            except Exception as exc:
                print(f"[arm] failed to return home during shutdown: {exc}")
            arm.close()
        if vision is not None and hasattr(vision, "close"):
            vision.close()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
