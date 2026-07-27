"""Train and evaluate the wrist-RGB local alignment policy.

Training is behavior cloning from a simulator-privileged teacher. The exported
actor signature contains only RGB, normalized commanded servo angles, and the
previous action, matching the real deployment contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .alignment_env import WristAlignmentEnv
    from .alignment_policy import AlignmentPolicy, export_torchscript
except ImportError:
    from alignment_env import WristAlignmentEnv
    from alignment_policy import AlignmentPolicy, export_torchscript


HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"


@dataclass
class TrainingReport:
    seed: int
    samples: int
    epochs: int
    validation_mae: float
    validation_direction_accuracy: float
    validation_stop_accuracy: float
    validation_stop_recall: float
    validation_false_stop_rate: float
    randomized_rollout_success_rate: float
    randomized_rollout_mean_steps: float
    device: str
    elapsed_seconds: float
    actor_inputs: tuple[str, ...] = (
        "wrist_rgb_128x72", "commanded_servo_angles", "previous_action")


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collect(samples: int, seed: int):
    env = WristAlignmentEnv(domain_randomization=True, image_augmentation=True,
                            max_steps=20, seed=seed)
    rng = np.random.default_rng(seed)
    images, servos, previous, labels = [], [], [], []
    def training_reset(first_seed=None):
        # A moving expert episode contributes several frames but an aligned
        # episode terminates in one. Sampling aligned resets more often keeps
        # actual frame-level stop examples near 15–20% instead of ~3%.
        if rng.random() < 0.55:
            current = int(rng.integers(78, 111))
            target = int(np.clip(current + rng.integers(-3, 4), 78, 110))
            return env.reset(seed=first_seed, options={
                "current_elbow": current, "target_elbow": target})
        return env.reset(seed=first_seed)

    try:
        observation, _ = training_reset(seed)
        while len(labels) < samples:
            expert = env.expert_action()
            images.append(observation["image"])
            servos.append(observation["servo"])
            previous.append(observation["previous_action"])
            labels.append(expert)
            # DAgger-like disturbance exposes the actor to recovery states
            # rather than only the exact expert trajectory.
            behavior = (expert if rng.random() > 0.23
                        else rng.uniform(-1.0, 1.0, 1).astype(np.float32))
            observation, _, terminated, truncated, _ = env.step(behavior)
            if terminated or truncated:
                observation, _ = training_reset()
    finally:
        env.close()
    return (
        np.asarray(images, dtype=np.uint8),
        np.asarray(servos, dtype=np.float32),
        np.asarray(previous, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
    )


def metrics(prediction: torch.Tensor, label: torch.Tensor):
    action = prediction[:, :1]
    aligned_probability = prediction[:, 1]
    mae = float(torch.mean(torch.abs(action - label)).item())
    active = torch.abs(label[:, 0]) >= 0.25
    if bool(active.any()):
        direction = torch.sign(action[active, 0]) == torch.sign(label[active, 0])
        accuracy = float(direction.float().mean().item())
    else:
        accuracy = 1.0
    stopped = torch.abs(label[:, 0]) < 0.25
    stop_accuracy = float(
        ((aligned_probability >= 0.65) == stopped).float().mean().item())
    predicted_stop = aligned_probability >= 0.65
    stop_recall = float(predicted_stop[stopped].float().mean().item()) \
        if bool(stopped.any()) else 1.0
    moving = ~stopped
    false_stop_rate = float(predicted_stop[moving].float().mean().item()) \
        if bool(moving.any()) else 0.0
    return mae, accuracy, stop_accuracy, stop_recall, false_stop_rate


def train_model(images, servos, previous, labels, *, epochs, batch_size, seed, device):
    count = len(labels)
    generator = np.random.default_rng(seed + 1)
    order = generator.permutation(count)
    split = max(1, int(count * 0.85))
    train_idx, valid_idx = order[:split], order[split:]

    def dataset(indices):
        return TensorDataset(
            torch.from_numpy(images[indices]), torch.from_numpy(servos[indices]),
            torch.from_numpy(previous[indices]), torch.from_numpy(labels[indices]))

    train_loader = DataLoader(dataset(train_idx), batch_size=batch_size,
                              shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset(valid_idx), batch_size=batch_size,
                              shuffle=False, num_workers=0)
    model = AlignmentPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.SmoothL1Loss(beta=0.20)
    best_state = None
    best_score = float("inf")

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for image, servo, prev, label in train_loader:
            image, servo = image.to(device), servo.to(device)
            prev, label = prev.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(image, servo, prev)
            action_prediction = prediction[:, :1]
            aligned_probability = prediction[:, 1]
            regression = loss_fn(action_prediction, label)
            # A wrong direction is much more costly than a slightly imperfect
            # magnitude because it moves the wrist away from the target.
            direction_penalty = torch.relu(-action_prediction * label).mean()
            aligned_target = (torch.abs(label[:, 0]) < 0.25).to(torch.float32)
            aligned_loss = nn.functional.binary_cross_entropy(
                aligned_probability, aligned_target,
                weight=1.0 + 3.0 * aligned_target)
            loss = regression + 0.35 * direction_penalty + 0.55 * aligned_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            running += float(loss.item()) * len(label)
        scheduler.step()

        model.eval()
        predictions, targets = [], []
        with torch.inference_mode():
            for image, servo, prev, label in valid_loader:
                predictions.append(model(
                    image.to(device), servo.to(device), prev.to(device)).cpu())
                targets.append(label)
        prediction = torch.cat(predictions)
        target = torch.cat(targets)
        mae, accuracy, stop_accuracy, stop_recall, false_stop_rate = metrics(
            prediction, target)
        selection_score = mae + 0.20 * (1.0 - stop_recall) + 0.30 * false_stop_rate
        if selection_score < best_score:
            best_score = selection_score
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        print(
            f"epoch={epoch + 1:02d}/{epochs} "
            f"train_loss={running / len(train_idx):.4f} "
            f"val_mae={mae:.4f} direction={accuracy:.3f} "
            f"stop={stop_accuracy:.3f} recall={stop_recall:.3f} "
            f"false_stop={false_stop_rate:.3f}")

    model.load_state_dict(best_state)
    model.to("cpu").eval()
    with torch.inference_mode():
        prediction = model(
            torch.from_numpy(images[valid_idx]), torch.from_numpy(servos[valid_idx]),
            torch.from_numpy(previous[valid_idx]))
    return model, metrics(prediction, torch.from_numpy(labels[valid_idx]))


def evaluate_rollouts(model: AlignmentPolicy, episodes: int, seed: int):
    env = WristAlignmentEnv(domain_randomization=True, image_augmentation=True,
                            max_steps=20, seed=seed)
    successes = 0
    step_counts = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            for step in range(env.max_steps):
                with torch.inference_mode():
                    output = model(
                        torch.from_numpy(observation["image"][None]),
                        torch.from_numpy(observation["servo"][None]),
                        torch.from_numpy(observation["previous_action"][None]),
                    ).numpy()[0]
                    # Deployment uses the existing jaw/target geometry gate as
                    # an independent stop vote. Rollout alignment is measured
                    # by the environment geometry, so the learned stop head is
                    # reported separately and never stops a moving rollout.
                    action = output[:1]
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated:
                    successes += 1
                    step_counts.append(step + 1)
                    break
                if truncated:
                    step_counts.append(env.max_steps)
                    break
    finally:
        env.close()
    return successes / episodes, float(np.mean(step_counts))


def run(args):
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print(f"collecting {args.samples} randomized frames")
    arrays = collect(args.samples, args.seed)
    print(f"training on {device}")
    model, (mae, accuracy, stop_accuracy, stop_recall, false_stop_rate) = train_model(
        *arrays, epochs=args.epochs, batch_size=args.batch_size,
        seed=args.seed, device=device)
    success_rate, mean_steps = evaluate_rollouts(
        model, args.eval_episodes, args.seed + 10_000)

    output = Path(args.output)
    export_torchscript(model, output)
    report = TrainingReport(
        seed=args.seed, samples=args.samples, epochs=args.epochs,
        validation_mae=mae, validation_direction_accuracy=accuracy,
        validation_stop_accuracy=stop_accuracy,
        validation_stop_recall=stop_recall,
        validation_false_stop_rate=false_stop_rate,
        randomized_rollout_success_rate=success_rate,
        randomized_rollout_mean_steps=mean_steps,
        device=str(device), elapsed_seconds=time.monotonic() - started,
    )
    report_path = output.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(asdict(report), indent=2))
    print(f"POLICY -> {output}")
    print(f"METRICS -> {report_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path,
                        default=GENERATED / "alignment_policy.ts")
    args = parser.parse_args()
    if args.samples < 100 or args.epochs < 1 or args.eval_episodes < 1:
        parser.error("samples>=100, epochs>=1, eval-episodes>=1 required")
    run(args)


if __name__ == "__main__":
    main()
