"""DAgger-style training for the complete safe floor-pick macro policy."""

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
    from .full_task_env import FullFloorPickEnv, TaskAction
    from .full_task_policy import (
        FullTaskNetwork, TemporalTaskGuard, export_torchscript, shield_action)
except ImportError:
    from full_task_env import FullFloorPickEnv, TaskAction
    from full_task_policy import (
        FullTaskNetwork, TemporalTaskGuard, export_torchscript, shield_action)


HERE = Path(__file__).resolve().parent
GENERATED = HERE / "generated"


@dataclass
class FullTaskReport:
    seed: int
    samples: int
    dagger_rounds: int
    validation_accuracy: float
    randomized_episodes: int
    randomized_success_rate: float
    randomized_mean_steps: float
    deterministic_episodes: int
    deterministic_success_rate: float
    elapsed_seconds: float


def collect(count, seed, model=None, policy_fraction=0.0):
    env = FullFloorPickEnv(domain_randomization=True, seed=seed)
    rng = np.random.default_rng(seed)
    observations = []
    labels = []
    observation, _ = env.reset(seed=seed)
    try:
        while len(labels) < count:
            expert = env.expert_action()
            observations.append(observation)
            labels.append(expert)
            draw = rng.random()
            if model is not None and draw < policy_fraction:
                with torch.inference_mode():
                    behavior = int(torch.argmax(
                        model(torch.from_numpy(observation[None])), dim=1).item())
            elif draw < policy_fraction + 0.18:
                behavior = int(rng.integers(0, len(TaskAction)))
            else:
                behavior = expert
            observation, _, terminated, truncated, _ = env.step(behavior)
            if terminated or truncated:
                observation, _ = env.reset()
    finally:
        env.close()
    return np.asarray(observations, np.float32), np.asarray(labels, np.int64)


def fit(model, observations, labels, *, epochs, batch_size, seed, device):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    split = int(len(labels) * 0.88)
    train_index, valid_index = order[:split], order[split:]
    counts = np.bincount(labels[train_index], minlength=len(TaskAction)).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts, 1.0)
    class_weights /= class_weights.mean()
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights).to(device))
    loader = DataLoader(TensorDataset(
        torch.from_numpy(observations[train_index]),
        torch.from_numpy(labels[train_index])), batch_size=batch_size,
        shuffle=True, num_workers=0)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    best = None
    best_accuracy = -1.0
    for epoch in range(epochs):
        model.train()
        for batch_observation, batch_label in loader:
            batch_observation = batch_observation.to(device)
            batch_label = batch_label.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_observation), batch_label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = torch.argmax(model(
                torch.from_numpy(observations[valid_index]).to(device)), dim=1).cpu().numpy()
        accuracy = float(np.mean(prediction == labels[valid_index]))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best = {name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()}
        print(f"epoch={epoch + 1:02d}/{epochs} validation_accuracy={accuracy:.5f}")
    model.load_state_dict(best)
    return model.to("cpu").eval(), best_accuracy


def evaluate(model, episodes, seed, randomized=True, shielded=True):
    env = FullFloorPickEnv(domain_randomization=randomized, seed=seed)
    successes = 0
    steps = []
    failures = {}
    temporal_guard = TemporalTaskGuard()
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            temporal_guard.reset()
            for step in range(env.max_steps):
                with torch.inference_mode():
                    action = int(torch.argmax(
                        model(torch.from_numpy(observation[None])), dim=1).item())
                if shielded:
                    action = int(shield_action(observation, action))
                    action = int(temporal_guard.filter(action))
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    success = terminated and info["holding"]
                    successes += int(success)
                    steps.append(step + 1)
                    if not success:
                        failures[info["event"]] = failures.get(info["event"], 0) + 1
                    break
    finally:
        env.close()
    return successes / episodes, float(np.mean(steps)), failures


def choose_device(value):
    if value != "auto":
        return torch.device(value)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def run(args):
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    observations, labels = collect(args.samples, args.seed)
    model, accuracy = fit(
        FullTaskNetwork(), observations, labels, epochs=args.epochs,
        batch_size=args.batch_size, seed=args.seed, device=device)

    for round_index in range(args.dagger_rounds):
        new_observations, new_labels = collect(
            args.dagger_samples, args.seed + 10_000 * (round_index + 1),
            model=model, policy_fraction=0.62)
        observations = np.concatenate((observations, new_observations))
        labels = np.concatenate((labels, new_labels))
        model, accuracy = fit(
            model, observations, labels, epochs=max(4, args.epochs // 2),
            batch_size=args.batch_size,
            seed=args.seed + round_index + 1, device=device)

    raw_rate, raw_steps, raw_failures = evaluate(
        model, min(args.eval_episodes, 2000), args.seed + 90_000,
        randomized=True, shielded=False)
    random_rate, mean_steps, failures = evaluate(
        model, args.eval_episodes, args.seed + 100_000, randomized=True)
    deterministic_episodes = min(args.eval_episodes, 1000)
    deterministic_rate, _det_steps, deterministic_failures = evaluate(
        model, deterministic_episodes, args.seed + 200_000, randomized=False)
    output = export_torchscript(model, args.output)
    report = FullTaskReport(
        seed=args.seed, samples=len(labels), dagger_rounds=args.dagger_rounds,
        validation_accuracy=accuracy,
        randomized_episodes=args.eval_episodes,
        randomized_success_rate=random_rate,
        randomized_mean_steps=mean_steps,
        deterministic_episodes=deterministic_episodes,
        deterministic_success_rate=deterministic_rate,
        elapsed_seconds=time.monotonic() - started)
    payload = asdict(report)
    payload["raw_unshielded_randomized_success_rate"] = raw_rate
    payload["raw_unshielded_randomized_mean_steps"] = raw_steps
    payload["raw_unshielded_failure_events"] = raw_failures
    payload["randomized_failure_events"] = failures
    payload["deterministic_failure_events"] = deterministic_failures
    metrics_path = Path(output).with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"POLICY -> {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=120_000)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-samples", type=int, default=40_000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path,
                        default=GENERATED / "full_task_policy_v1.ts")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
