"""Train the ErrP classifier from labeled epochs, then save errp_model.pkl.

How to collect data (once hardware works):
  Run record_errp.py-style sessions where the arm deliberately does correct and
  wrong reaches while the subject watches. Each action produces one epoch (the
  EEG window right after action onset) + a label (1 = wrong/error, 0 = correct).

Storage format (simple, portable): one CSV per epoch in a folder, plus a
labels.csv mapping filename -> label. Each epoch CSV is samples x channels (uV),
one row per sample, EEG_CHANNELS columns.

    data/errp/
        labels.csv          # columns: file,label
        epoch_0001.csv
        epoch_0002.csv
        ...

Usage:
    python errp_train.py data/errp
"""
import sys
import os
import csv

import config
from errp import ErrPDetector


def load_epoch(path):
    window = []
    with open(path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            window.append([float(v) for v in row])
    return window


def load_dataset(folder):
    labels_path = os.path.join(folder, "labels.csv")
    windows, labels = [], []
    with open(labels_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            fpath = os.path.join(folder, r["file"])
            windows.append(load_epoch(fpath))
            labels.append(int(r["label"]))
    return windows, labels


def main():
    if len(sys.argv) < 2:
        print("usage: python errp_train.py <data_folder>")
        sys.exit(1)
    folder = sys.argv[1]
    windows, labels = load_dataset(folder)
    n_pos = sum(labels)
    print(f"loaded {len(windows)} epochs ({n_pos} error / {len(labels) - n_pos} ok)")

    det = ErrPDetector(backend="baseline")
    det.fit(windows, labels)
    det.save(config.ERRP_MODEL_PATH)
    print(f"saved model -> {config.ERRP_MODEL_PATH}")

    # quick train-set sanity check
    correct = sum(1 for w, y in zip(windows, labels)
                  if int(det.p_error(w) >= 0.5) == y)
    print(f"train accuracy: {correct}/{len(labels)} = {correct/len(labels):.2f}")
    print("Set ERRP_BACKEND='model' in config.py to use it live.")


if __name__ == "__main__":
    main()
