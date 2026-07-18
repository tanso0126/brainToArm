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
import os
import csv
import argparse

import config
from errp import ErrPDetector


def load_epoch(path):
    window = []
    with open(path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            window.append([float(v) for v in row])
    if not window:
        raise ValueError(f"empty epoch: {path}")
    widths = {len(row) for row in window}
    if widths != {config.EEG_CHANNELS}:
        raise ValueError(
            f"{path} has channel widths {sorted(widths)}, expected {config.EEG_CHANNELS}")
    return window


def load_dataset(folder):
    labels_path = os.path.join(folder, "labels.csv")
    windows, labels = [], []
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"missing labels file: {labels_path}")
    with open(labels_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = r["file"]
            if os.path.basename(name) != name:
                raise ValueError(f"labels.csv contains a non-local filename: {name}")
            label = int(r["label"])
            if label not in (0, 1):
                raise ValueError(f"invalid label {label} for {name}")
            fpath = os.path.join(folder, name)
            windows.append(load_epoch(fpath))
            labels.append(label)
    if not windows:
        raise ValueError(f"no labeled epochs in {labels_path}")
    return windows, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--output", default=config.ERRP_MODEL_PATH)
    args = ap.parse_args()
    folder = args.folder
    windows, labels = load_dataset(folder)
    n_pos = sum(labels)
    print(f"loaded {len(windows)} epochs ({n_pos} error / {len(labels) - n_pos} ok)")
    if n_pos == 0 or n_pos == len(labels):
        raise ValueError("training requires both correct and error epochs")

    det = ErrPDetector(backend="baseline")
    min_class = min(n_pos, len(labels) - n_pos)
    if min_class >= 2:
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        X = [det._features(w) for w in windows]
        folds = min(5, min_class)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        scores = cross_val_score(det.make_model(), X, labels, cv=cv, scoring="balanced_accuracy")
        print(f"{folds}-fold balanced accuracy: {scores.mean():.2f} ± {scores.std():.2f}")
    else:
        print("warning: too few samples per class for cross-validation")

    det.fit(windows, labels)
    det.save(args.output)
    print(f"saved model -> {args.output}")

    # quick train-set sanity check
    correct = sum(1 for w, y in zip(windows, labels)
                  if int(det.p_error(w) >= 0.5) == y)
    print(f"fit-set accuracy (diagnostic only): {correct}/{len(labels)} = {correct/len(labels):.2f}")
    print("Set ERRP_BACKEND='model' in config.py to use it live.")


if __name__ == "__main__":
    main()
