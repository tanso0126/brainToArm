"""Train and evaluate the separate rigid-wrist reduced-arm macro policy."""

from pathlib import Path
import argparse
import hashlib
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .reduced_dof_task_env import ReducedFloorPickEnv, ReducedTaskAction
    from .reduced_dof_task_policy import (
        ReducedTaskNetwork, ReducedTemporalGuard, export_torchscript,
        shield_action,
    )
except ImportError:
    from reduced_dof_task_env import ReducedFloorPickEnv, ReducedTaskAction
    from reduced_dof_task_policy import (
        ReducedTaskNetwork, ReducedTemporalGuard, export_torchscript,
        shield_action,
    )


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "models" / "reduced_dof_policy_v1.ts"


def collect(count, seed, model=None, policy_fraction=0.0):
    env = ReducedFloorPickEnv(domain_randomization=True, seed=seed)
    rng = np.random.default_rng(seed)
    x, y = [], []
    observation, _ = env.reset(seed=seed)
    while len(y) < count:
        expert = env.expert_action()
        x.append(observation)
        y.append(expert)
        draw = rng.random()
        if model is not None and draw < policy_fraction:
            with torch.inference_mode():
                behavior = int(torch.argmax(
                    model(torch.from_numpy(observation[None])), dim=1).item())
        elif draw < policy_fraction + 0.20:
            behavior = int(rng.integers(0, len(ReducedTaskAction)))
        else:
            behavior = expert
        observation, _, terminated, truncated, _ = env.step(behavior)
        if terminated or truncated:
            observation, _ = env.reset()
    return np.asarray(x, np.float32), np.asarray(y, np.int64)


def fit(model, observations, labels, *, epochs, batch_size, seed, device):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))
    split = int(len(labels) * 0.88)
    train, valid = order[:split], order[split:]
    counts = np.bincount(labels[train], minlength=len(ReducedTaskAction)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1)
    weights /= weights.mean()
    loader = DataLoader(TensorDataset(
        torch.from_numpy(observations[train]), torch.from_numpy(labels[train])),
        batch_size=batch_size, shuffle=True, num_workers=0)
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    best, best_accuracy = None, -1.0
    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            prediction = torch.argmax(
                model(torch.from_numpy(observations[valid]).to(device)), dim=1
            ).cpu().numpy()
        accuracy = float(np.mean(prediction == labels[valid]))
        print(f"epoch={epoch + 1:02d}/{epochs} validation_accuracy={accuracy:.5f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best = {name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()}
    model.load_state_dict(best)
    return model.cpu().eval(), best_accuracy


def evaluate(model, episodes, seed, *, randomized=True, shielded=True):
    env = ReducedFloorPickEnv(domain_randomization=randomized, seed=seed)
    guard = ReducedTemporalGuard()
    successes, steps, failures = 0, [], {}
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        guard.reset()
        for step in range(env.max_steps):
            with torch.inference_mode():
                action = int(torch.argmax(
                    model(torch.from_numpy(observation[None])), dim=1).item())
            if shielded:
                action = int(guard.filter(shield_action(observation, action)))
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                success = bool(terminated and info["holding"])
                successes += int(success)
                steps.append(step + 1)
                if not success:
                    failures[info["event"]] = failures.get(info["event"], 0) + 1
                break
    return successes / episodes, float(np.mean(steps)), failures


def _device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def run(args):
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    observations, labels = collect(args.samples, args.seed)
    model, accuracy = fit(
        ReducedTaskNetwork(), observations, labels, epochs=args.epochs,
        batch_size=args.batch_size, seed=args.seed, device=_device(args.device))
    for round_index in range(args.dagger_rounds):
        new_x, new_y = collect(
            args.dagger_samples, args.seed + 10_000 * (round_index + 1),
            model=model, policy_fraction=0.62)
        observations = np.concatenate((observations, new_x))
        labels = np.concatenate((labels, new_y))
        model, accuracy = fit(
            model, observations, labels, epochs=max(4, args.epochs // 2),
            batch_size=args.batch_size, seed=args.seed + round_index + 1,
            device=_device(args.device))

    raw_rate, raw_steps, raw_failures = evaluate(
        model, min(args.eval_episodes, 2000), args.seed + 80_000,
        randomized=True, shielded=False)
    shield_rate, mean_steps, failures = evaluate(
        model, args.eval_episodes, args.seed + 100_000,
        randomized=True, shielded=True)
    deterministic_count = min(args.eval_episodes, 1000)
    deterministic_rate, deterministic_steps, deterministic_failures = evaluate(
        model, deterministic_count, args.seed + 200_000,
        randomized=False, shielded=True)
    output = export_torchscript(model, args.output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    payload = {
        "policy": "rigid_wrist_reduced_dof_macro_v1",
        "algorithm": "DAgger-style imitation learning with a deterministic safety shield",
        "controllable_motors": [2, 3, 5],
        "fixed_motors": [1, 4, 6],
        "observations": int(observations.shape[1]),
        "actions": len(ReducedTaskAction),
        "samples": int(len(labels)),
        "dagger_rounds": args.dagger_rounds,
        "validation_accuracy": accuracy,
        "randomized_episodes": args.eval_episodes,
        "randomized_success_rate": shield_rate,
        "randomized_mean_steps": mean_steps,
        "raw_unshielded_success_rate": raw_rate,
        "raw_unshielded_mean_steps": raw_steps,
        "raw_unshielded_failures": raw_failures,
        "shielded_failures": failures,
        "deterministic_episodes": deterministic_count,
        "deterministic_success_rate": deterministic_rate,
        "deterministic_mean_steps": deterministic_steps,
        "deterministic_failures": deterministic_failures,
        "sha256": digest,
        "elapsed_seconds": time.monotonic() - started,
        "physical_validation": "required after measuring the fixed wrist angle",
    }
    metrics = output.with_suffix(".metrics.json")
    metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"POLICY -> {output}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-samples", type=int, default=30_000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
