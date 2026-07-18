"""Record labeled ErrP epochs so errp_train.py can fit a real classifier.

The catch with ErrP: you need examples of the brain reacting to CORRECT actions
and to WRONG actions. We generate both on purpose. The arm points at an object;
sometimes it's the object we told the subject to want (correct), sometimes not
(error). The subject just watches and silently judges. We know the ground truth
(we chose it), so labeling is automatic — no button pressing.

Onset-locked exactly like the live loop: the instant the arm commits to pointing
is timestamped, and the epoch [onset-baseline, onset+window] is cut by that time.

Each trial writes one epoch CSV (samples x EEG_CHANNELS, microvolts) plus a row
in labels.csv (file,label). Feed the folder to errp_train.py.

Usage:
    python record_errp.py data/errp --trials 40
Then:
    python errp_train.py data/errp     # -> errp_model.pkl
"""
import os
import csv
import time
import random
import argparse
import glob
import math
import re

import config
from arm_serial import ArmSerial
from eeg_bridge import EEGBridge
from policy import Policy
from vision import Vision
from orchestrator import preflight


def save_epoch(folder, idx, window):
    fname = f"epoch_{idx:04d}.csv"
    with open(os.path.join(folder, fname), "x", newline="") as f:
        w = csv.writer(f)
        for row in window:
            w.writerow([f"{v:.4f}" for v in row])
    return fname


def next_epoch_index(folder):
    """Choose an unused monotonic index across repeated recording sessions."""
    indices = []
    for path in glob.glob(os.path.join(folder, "epoch_*.csv")):
        match = re.fullmatch(r"epoch_(\d+)\.csv", os.path.basename(path))
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=-1) + 1


def setup_scene(vision):
    if vision.mock or config.OBJECT_METHOD == "aruco":
        return
    input("[setup] clear the table (arm parked), then press Enter... ")
    vision.learn_background()
    input("[setup] place at least two objects, then press Enter... ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--error-rate", type=float, default=0.5,
                    help="fraction of trials that are deliberately WRONG")
    ap.add_argument("--goal-index", type=int, default=0,
                    help="index in the initial detection list that the subject should want")
    ap.add_argument("--max-wait-s", type=float, default=60.0,
                    help="abort if a usable >=2-object scene is absent this long")
    args = ap.parse_args()
    if args.trials <= 0:
        ap.error("--trials must be > 0")
    if not (0 <= args.error_rate <= 1):
        ap.error("--error-rate must be in [0, 1]")
    if args.goal_index < 0 or args.max_wait_s <= 0:
        ap.error("--goal-index must be >= 0 and --max-wait-s must be > 0")
    os.makedirs(args.folder, exist_ok=True)

    arm = eeg = vision = lf = None
    try:
        arm = ArmSerial()
        eeg = EEGBridge().start()
        vision = Vision()
        policy = Policy()
        arm.home(); arm.wait_done()
        setup_scene(vision)
        if not preflight(arm, eeg, vision):
            raise RuntimeError("recording preflight failed")

        initial = vision.detect()
        if len(initial) < 2:
            raise RuntimeError("recording needs at least two detected objects")
        if args.goal_index >= len(initial):
            raise ValueError(
                f"--goal-index {args.goal_index} but only {len(initial)} objects were detected")
        goal = initial[args.goal_index]
        goal_xy = (goal.x, goal.y)

        labels_path = os.path.join(args.folder, "labels.csv")
        new_file = not os.path.exists(labels_path)
        lf = open(labels_path, "a", newline="")
        writer = csv.writer(lf)
        if new_file:
            writer.writerow(["file", "label"])
        epoch_idx = next_epoch_index(args.folder)

        print(f"[record] goal='{goal.label}' near ({goal.x:.1f},{goal.y:.1f}).")
        print(f"[record] {args.trials} trials into {args.folder}/. Memorize the goal; "
              "watch the arm and silently judge its target.\n")

        recorded = 0
        missing_since = None
        while recorded < args.trials:
            objs = vision.detect()
            if len(objs) < 2:
                print("[record] need >=2 objects in view; rearrange and continue.")
                missing_since = missing_since or time.monotonic()
                if time.monotonic() - missing_since >= args.max_wait_s:
                    raise TimeoutError("usable recording scene did not return in time")
                time.sleep(1.0)
                continue
            missing_since = None

            wanted = min(objs, key=lambda d: math.hypot(d.x - goal_xy[0], d.y - goal_xy[1]))
            alternatives = [d for d in objs if d is not wanted]
            is_error = random.random() < args.error_rate
            target = random.choice(alternatives) if is_error else wanted
            label = 1 if is_error else 0

            print(f"  trial {recorded+1}/{args.trials}: goal='{wanted.label}', "
                  f"arm targets '{target.label}'  ({'WRONG' if is_error else 'correct'})")

            # commit + onset-lock, same as the live loop
            arm.gripper(open_=True); arm.wait_done()
            angles = policy.target_to_angles((target.x, target.y), config.Z_APPROACH)
            onset = eeg.mark_onset()
            if getattr(arm, "mock", False) and is_error:
                eeg.mark_error(config.ERRP_WINDOW_S, onset_t=onset)
            arm.send_angles(angles)
            if getattr(arm, "mock", False):
                import sim
                sim.WORLD.set_command((target.x, target.y))
            window = eeg.wait_and_epoch(onset)
            arm.wait_done()
            expected = int((config.ERRP_BASELINE_S + config.ERRP_WINDOW_S) * config.EEG_FS)
            if len(window) < int(expected * config.EEG_MIN_EPOCH_FRACTION):
                raise RuntimeError(f"incomplete recording epoch ({len(window)}/{expected})")

            fname = save_epoch(args.folder, epoch_idx, window)
            writer.writerow([fname, label]); lf.flush()
            epoch_idx += 1
            recorded += 1

            arm.home(); arm.wait_done()
            time.sleep(0.8)                         # inter-trial rest
    finally:
        if lf is not None:
            lf.close()
        if eeg is not None:
            eeg.stop()
        if arm is not None:
            try:
                arm.home(); arm.wait_done()
            except Exception as exc:
                print(f"[arm] failed to return home during recorder shutdown: {exc}")
            arm.close()
        if vision is not None and hasattr(vision, "close"):
            vision.close()
    print(f"\n[record] done. Train with:  python errp_train.py {args.folder}")


if __name__ == "__main__":
    main()
