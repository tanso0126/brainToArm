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
import sys
import csv
import time
import random
import argparse

import config
from arm_serial import ArmSerial
from eeg_bridge import EEGBridge
from policy import Policy
from vision import Vision


def save_epoch(folder, idx, window):
    fname = f"epoch_{idx:04d}.csv"
    with open(os.path.join(folder, fname), "w", newline="") as f:
        w = csv.writer(f)
        for row in window:
            w.writerow([f"{v:.4f}" for v in row])
    return fname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--error-rate", type=float, default=0.5,
                    help="fraction of trials that are deliberately WRONG")
    args = ap.parse_args()
    os.makedirs(args.folder, exist_ok=True)

    arm = ArmSerial()
    eeg = EEGBridge().start()
    vision = Vision()
    policy = Policy()
    arm.home(); arm.wait_done()
    time.sleep(1.0)

    labels_path = os.path.join(args.folder, "labels.csv")
    new_file = not os.path.exists(labels_path)
    lf = open(labels_path, "a", newline="")
    writer = csv.writer(lf)
    if new_file:
        writer.writerow(["file", "label"])

    print(f"[record] {args.trials} trials into {args.folder}/. "
          "Watch the arm; silently judge if it targets the RIGHT object.\n")

    try:
        for i in range(args.trials):
            objs = vision.detect()
            if len(objs) < 2:
                print("[record] need >=2 objects in view; rearrange and continue.")
                time.sleep(2.0); continue

            wanted = objs[0]                       # we declare object 0 the goal
            is_error = random.random() < args.error_rate
            target = random.choice(objs[1:]) if is_error else wanted
            label = 1 if is_error else 0

            print(f"  trial {i+1}/{args.trials}: goal='{wanted.label}', "
                  f"arm targets '{target.label}'  ({'WRONG' if is_error else 'correct'})")

            # commit + onset-lock, same as the live loop
            arm.gripper(open_=True)
            angles = policy.target_to_angles((target.x, target.y), config.Z_APPROACH)
            onset = eeg.mark_onset()
            arm.send_angles(angles)
            if getattr(arm, "mock", False):
                import sim
                sim.WORLD.set_command((target.x, target.y))
                if is_error:
                    eeg.mark_error(config.ERRP_WINDOW_S)   # mock brain reacts
            window = eeg.wait_and_epoch(onset)

            fname = save_epoch(args.folder, i, window)
            writer.writerow([fname, label]); lf.flush()

            arm.home(); arm.wait_done()
            time.sleep(0.8)                         # inter-trial rest
    finally:
        lf.close()
        eeg.stop()
        arm.home(); arm.close()
        if hasattr(vision, "close"):
            vision.close()
    print(f"\n[record] done. Train with:  python errp_train.py {args.folder}")


if __name__ == "__main__":
    main()
