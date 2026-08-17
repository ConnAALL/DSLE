#!/usr/bin/env python3
"""Train or evaluate a small PPO baseline against the public DSLE API.

The policy uses the 84x84 grayscale Nature CNN from the paper.  This file is
deliberately self-contained: it is example agent code, not part of ``dsle``.

Run inside a DSLE NVIDIA container, for example::

    python examples/ppo.py train --boss asylum_demon --instance dsr-1
    python examples/ppo.py eval --boss asylum_demon --instance dsr-1 \
        --checkpoint runs/ppo_asylum_demon_seed42.pt

``--seed`` controls only agent initialization and sampling.  Dark Souls runs
in a separate process and its internal randomness cannot be seeded by DSLE.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import dsle

try:  # Support both ``python examples/ppo.py`` and module imports.
    from .common import (
        append_jsonl,
        atomic_output,
        default_output_dir,
        prepare_output_file,
        prepare_output_target,
    )
except ImportError:  # pragma: no cover - exercised by direct script execution
    from common import (
        append_jsonl,
        atomic_output,
        default_output_dir,
        prepare_output_file,
        prepare_output_target,
    )

IMAGE_SIZE = (84, 84)
TOTAL_TIMESTEPS = 100_000
ROLLOUT_STEPS = 128
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 32
LEARNING_RATE = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEFFICIENT = 0.1


def _layer_init(layer: nn.Module, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)  # type: ignore[attr-defined]
    nn.init.constant_(layer.bias, bias)  # type: ignore[attr-defined]
    return layer


class PPOAgent(nn.Module):
    """Nature CNN with independent categorical-policy and value heads."""

    def __init__(self, n_actions: int):
        super().__init__()
        self.encoder = nn.Sequential(
            _layer_init(nn.Conv2d(1, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            _layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.policy = _layer_init(nn.Linear(512, n_actions), std=0.01)
        self.value = _layer_init(nn.Linear(512, 1), std=1.0)

    def _features(self, observation: torch.Tensor) -> torch.Tensor:
        return self.encoder(observation.float() / 255.0)

    def value_of(self, observation: torch.Tensor) -> torch.Tensor:
        return self.value(self._features(observation)).squeeze(-1)

    def action_and_value(
        self,
        observation: torch.Tensor,
        action: torch.Tensor | None = None,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._features(observation)
        distribution = Categorical(logits=self.policy(features))
        if action is None:
            action = distribution.probs.argmax(dim=-1) if deterministic else distribution.sample()
        return (
            action,
            distribution.log_prob(action),
            distribution.entropy(),
            self.value(features).squeeze(-1),
        )


def resize_observation(observation: np.ndarray) -> torch.Tensor:
    """Convert a DSLE grayscale frame to a CPU ``(1, 84, 84)`` uint8 tensor."""

    array = np.asarray(observation)
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError(
            "PPO expects dsle.make(..., obs_mode='grayscale') observations with "
            f"shape (1, H, W), received {array.shape}"
        )
    frame = torch.from_numpy(np.ascontiguousarray(array)).float().unsqueeze(0)
    resized = F.interpolate(frame, size=IMAGE_SIZE, mode="area").squeeze(0)
    return resized.round().clamp_(0, 255).to(torch.uint8)


def time_limit_bootstrap_reward(
    reward: float,
    *,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
    final_value: float,
    gamma: float,
) -> float:
    """Bootstrap a valid final observation at a Gymnasium time limit.

    GAE still cuts the trajectory at the reset boundary. Folding the final
    state's value into the last reward preserves the time-limit target without
    leaking advantages into the next episode. Runtime-error truncations do not
    bootstrap because their returned frame can be stale.
    """

    if not terminated and truncated and info.get("truncated_reason") == "max_steps":
        return float(reward) + float(gamma) * float(final_value)
    return float(reward)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(name)


def _seed_agent(seed: int) -> np.random.Generator:
    """Seed agent-side randomness without claiming to seed the game process."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def _make_env(args: argparse.Namespace) -> Any:
    kwargs: dict[str, Any] = {
        "instance": args.instance,
        "difficulty": args.difficulty,
        "obs_mode": "grayscale",
        "max_steps": args.max_steps,
    }
    if args.asset_dir is not None:
        kwargs["asset_dir"] = args.asset_dir
    if args.instance_config is not None:
        kwargs["instance_config"] = args.instance_config
    return dsle.make(args.boss, **kwargs)


def _checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return Path(args.checkpoint)
    return Path(args.output_dir) / f"ppo_{args.boss}_seed{args.seed}.pt"


def _metrics_path(args: argparse.Namespace, phase: str) -> Path:
    explicit = getattr(args, "metrics_output", None)
    if explicit is not None:
        return Path(explicit)
    output_dir = Path(getattr(args, "output_dir", default_output_dir()))
    return output_dir / f"ppo_{args.boss}_seed{args.seed}_{phase}.jsonl"


def _save_checkpoint(
    path: Path,
    agent: PPOAgent,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    updates: int,
    n_actions: int,
    args: argparse.Namespace,
) -> None:
    """Atomically save enough state for evaluation and future inspection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "handler"
    }
    with atomic_output(path) as stream:
        torch.save(
            {
                "algorithm": "ppo",
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "updates": updates,
                "n_actions": n_actions,
                "config": config,
            },
            stream,
        )


def _load_agent(path: Path, n_actions: int, device: torch.device) -> PPOAgent:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("algorithm") != "ppo":
        raise ValueError(f"{path} is not a PPO checkpoint")
    saved_actions = int(checkpoint.get("n_actions", n_actions))
    if saved_actions != n_actions:
        raise ValueError(
            f"Checkpoint uses {saved_actions} actions, but this environment exposes {n_actions}"
        )
    agent = PPOAgent(n_actions).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    return agent


def _validate_train_args(args: argparse.Namespace) -> None:
    positive = {
        "total_timesteps": args.total_timesteps,
        "rollout_steps": args.rollout_steps,
        "epochs": args.epochs,
        "minibatch_size": args.minibatch_size,
        "checkpoint_interval": args.checkpoint_interval,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"These options must be positive: {', '.join(invalid)}")
    floating = {
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "clip_coefficient": args.clip_coefficient,
        "entropy_coefficient": args.entropy_coefficient,
        "value_coefficient": args.value_coefficient,
    }
    nonfinite = [name for name, value in floating.items() if not math.isfinite(value)]
    if nonfinite:
        raise ValueError(f"These options must be finite: {', '.join(nonfinite)}")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("--gamma must be in (0, 1]")
    if not 0.0 <= args.gae_lambda <= 1.0:
        raise ValueError("--gae-lambda must be in [0, 1]")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be positive")
    if args.clip_coefficient < 0.0:
        raise ValueError("--clip-coefficient must be non-negative")
    if args.entropy_coefficient < 0.0:
        raise ValueError("--entropy-coefficient must be non-negative")
    if args.value_coefficient < 0.0:
        raise ValueError("--value-coefficient must be non-negative")


def _validate_environment_args(args: argparse.Namespace) -> None:
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")


def train(args: argparse.Namespace) -> Path:
    """Train PPO for exactly ``total_timesteps`` environment transitions."""

    _validate_train_args(args)
    _validate_environment_args(args)
    checkpoint_path = prepare_output_target(_checkpoint_path(args))
    metrics_path = _metrics_path(args, "train").expanduser().resolve()
    if metrics_path == checkpoint_path:
        raise ValueError("--metrics-output and --checkpoint must be different files")
    metrics_path = prepare_output_file(metrics_path)
    rng = _seed_agent(args.seed)
    device = _device(args.device)
    env = None

    try:
        env = _make_env(args)
        n_actions = int(env.action_space.n)
        agent = PPOAgent(n_actions).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        observation, _ = env.reset()
        next_observation = resize_observation(observation)
        next_done = False
        global_step = 0
        updates = 0
        episode_return = 0.0
        episode_length = 0
        episodes = 0
        started = time.monotonic()

        print(
            f"PPO train: boss={args.boss} instance={args.instance} "
            f"device={device} checkpoint={checkpoint_path}"
        )

        while global_step < args.total_timesteps:
            learning_rate_fraction = 1.0 - global_step / args.total_timesteps
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = args.learning_rate * learning_rate_fraction
            rollout_size = min(args.rollout_steps, args.total_timesteps - global_step)
            observations = torch.empty((rollout_size, 1, *IMAGE_SIZE), dtype=torch.uint8)
            actions = torch.empty(rollout_size, dtype=torch.long)
            log_probabilities = torch.empty(rollout_size)
            rewards = torch.empty(rollout_size)
            dones = torch.empty(rollout_size)
            values = torch.empty(rollout_size)

            for index in range(rollout_size):
                observations[index] = next_observation
                dones[index] = float(next_done)
                with torch.no_grad():
                    action, log_probability, _, value = agent.action_and_value(
                        next_observation.unsqueeze(0).to(device)
                    )

                action_number = int(action.item())
                actions[index] = action_number
                log_probabilities[index] = float(log_probability.item())
                values[index] = float(value.item())

                observation, reward, terminated, truncated, info = env.step(action_number)
                resized_successor = resize_observation(observation)
                final_value = 0.0
                if not terminated and truncated and info.get("truncated_reason") == "max_steps":
                    with torch.no_grad():
                        final_value = float(
                            agent.value_of(resized_successor.unsqueeze(0).to(device)).item()
                        )
                rewards[index] = time_limit_bootstrap_reward(
                    float(reward),
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    final_value=final_value,
                    gamma=args.gamma,
                )
                global_step += 1
                episode_return += float(reward)
                episode_length += 1
                next_done = bool(terminated or truncated)

                if next_done:
                    episodes += 1
                    won = bool(info.get("win", False))
                    append_jsonl(
                        metrics_path,
                        {
                            "algorithm": "ppo",
                            "phase": "train",
                            "boss": args.boss,
                            "instance": args.instance,
                            "seed": args.seed,
                            "episode": episodes,
                            "global_step": global_step,
                            "return": episode_return,
                            "length": episode_length,
                            "win": won,
                            "terminated_reason": info.get("terminated_reason"),
                            "truncated_reason": info.get("truncated_reason"),
                        },
                    )
                    print(
                        f"episode={episodes} step={global_step} return={episode_return:.3f} "
                        f"length={episode_length} victory={won}"
                    )
                    if global_step < args.total_timesteps:
                        observation, _ = env.reset()
                    episode_return = 0.0
                    episode_length = 0
                next_observation = resize_observation(observation)

            with torch.no_grad():
                bootstrap_value = float(
                    agent.value_of(next_observation.unsqueeze(0).to(device)).item()
                )

            advantages = torch.empty(rollout_size)
            last_advantage = 0.0
            for index in reversed(range(rollout_size)):
                if index == rollout_size - 1:
                    next_nonterminal = 1.0 - float(next_done)
                    next_value = bootstrap_value
                else:
                    next_nonterminal = 1.0 - float(dones[index + 1])
                    next_value = float(values[index + 1])
                delta = (
                    float(rewards[index])
                    + args.gamma * next_value * next_nonterminal
                    - float(values[index])
                )
                last_advantage = (
                    delta + args.gamma * args.gae_lambda * next_nonterminal * last_advantage
                )
                advantages[index] = last_advantage
            returns = advantages + values

            batch_indices = np.arange(rollout_size)
            for _ in range(args.epochs):
                rng.shuffle(batch_indices)
                for start in range(0, rollout_size, args.minibatch_size):
                    selection = torch.as_tensor(
                        batch_indices[start : start + args.minibatch_size], dtype=torch.long
                    )
                    batch_observations = observations[selection].to(device)
                    batch_actions = actions[selection].to(device)
                    old_log_probabilities = log_probabilities[selection].to(device)
                    batch_advantages = advantages[selection].to(device)
                    batch_returns = returns[selection].to(device)
                    if batch_advantages.numel() > 1:
                        batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                            batch_advantages.std(unbiased=False) + 1e-8
                        )

                    _, new_log_probabilities, entropy, new_values = agent.action_and_value(
                        batch_observations, batch_actions
                    )
                    ratio = (new_log_probabilities - old_log_probabilities).exp()
                    policy_loss = torch.maximum(
                        -batch_advantages * ratio,
                        -batch_advantages
                        * ratio.clamp(1.0 - args.clip_coefficient, 1.0 + args.clip_coefficient),
                    ).mean()
                    value_loss = 0.5 * (new_values - batch_returns).square().mean()
                    loss = (
                        policy_loss
                        + args.value_coefficient * value_loss
                        - args.entropy_coefficient * entropy.mean()
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            updates += 1
            if updates % args.checkpoint_interval == 0:
                _save_checkpoint(
                    checkpoint_path,
                    agent,
                    optimizer,
                    global_step=global_step,
                    updates=updates,
                    n_actions=n_actions,
                    args=args,
                )

            elapsed = max(time.monotonic() - started, 1e-9)
            print(
                f"update={updates} step={global_step}/{args.total_timesteps} "
                f"steps_per_second={global_step / elapsed:.2f}"
            )

        _save_checkpoint(
            checkpoint_path,
            agent,
            optimizer,
            global_step=global_step,
            updates=updates,
            n_actions=n_actions,
            args=args,
        )
        return checkpoint_path
    finally:
        if env is not None:
            env.close()


def evaluate(args: argparse.Namespace) -> None:
    """Run deterministic policy episodes from a saved checkpoint."""

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    _validate_environment_args(args)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PPO checkpoint does not exist: {checkpoint_path}")
    metrics_path = _metrics_path(args, "eval").expanduser().resolve()
    if metrics_path == checkpoint_path:
        raise ValueError("--metrics-output and --checkpoint must be different files")
    metrics_path = prepare_output_file(metrics_path)
    _seed_agent(args.seed)
    device = _device(args.device)
    env = None

    try:
        env = _make_env(args)
        agent = _load_agent(checkpoint_path, int(env.action_space.n), device)
        agent.eval()
        returns: list[float] = []
        victories = 0

        for episode in range(1, args.episodes + 1):
            observation, _ = env.reset()
            episode_return = 0.0
            episode_length = 0
            done = False
            final_info: dict[str, Any] = {}
            while not done:
                resized = resize_observation(observation).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, _, _, _ = agent.action_and_value(resized, deterministic=True)
                observation, reward, terminated, truncated, final_info = env.step(
                    int(action.item())
                )
                episode_return += float(reward)
                episode_length += 1
                done = bool(terminated or truncated)

            won = bool(final_info.get("win", False))
            victories += int(won)
            returns.append(episode_return)
            append_jsonl(
                metrics_path,
                {
                    "algorithm": "ppo",
                    "phase": "evaluate",
                    "boss": args.boss,
                    "instance": args.instance,
                    "seed": args.seed,
                    "episode": episode,
                    "return": episode_return,
                    "length": episode_length,
                    "win": won,
                    "terminated_reason": final_info.get("terminated_reason"),
                    "truncated_reason": final_info.get("truncated_reason"),
                },
            )
            print(f"episode={episode} return={episode_return:.3f} victory={won}")

        print(
            f"episodes={args.episodes} mean_return={np.mean(returns):.3f} "
            f"victory_rate={victories / args.episodes:.3f}"
        )
    finally:
        if env is not None:
            env.close()


def _add_environment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--boss", required=True, help="Boss id from dsle.list_bosses()")
    parser.add_argument("--instance", default="dsr-1", help="Configured game instance")
    parser.add_argument("--difficulty", choices=("standard", "boosted"), default="standard")
    parser.add_argument("--max-steps", type=int, default=7_200)
    parser.add_argument("--asset-dir", type=Path)
    parser.add_argument("--instance-config", type=Path)
    parser.add_argument("--seed", type=int, default=42, help="Agent seed; does not seed the game")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    output_dir = default_output_dir()

    train_parser = commands.add_parser("train", help="Train a PPO policy")
    _add_environment_arguments(train_parser)
    train_parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    train_parser.add_argument("--rollout-steps", type=int, default=ROLLOUT_STEPS)
    train_parser.add_argument("--epochs", type=int, default=UPDATE_EPOCHS)
    train_parser.add_argument("--minibatch-size", type=int, default=MINIBATCH_SIZE)
    train_parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    train_parser.add_argument("--gamma", type=float, default=GAMMA)
    train_parser.add_argument("--gae-lambda", type=float, default=GAE_LAMBDA)
    train_parser.add_argument("--clip-coefficient", type=float, default=CLIP_COEFFICIENT)
    train_parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    train_parser.add_argument("--value-coefficient", type=float, default=0.5)
    train_parser.add_argument("--max-grad-norm", type=float, default=0.5)
    train_parser.add_argument("--checkpoint", type=Path)
    train_parser.add_argument(
        "--checkpoint-interval", type=int, default=10, help="Save every N updates"
    )
    train_parser.add_argument("--output-dir", type=Path, default=output_dir)
    train_parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Episode JSONL path (defaults below --output-dir)",
    )
    train_parser.set_defaults(handler=train)

    eval_parser = commands.add_parser("eval", help="Evaluate a PPO checkpoint")
    _add_environment_arguments(eval_parser)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--episodes", type=int, default=100)
    eval_parser.add_argument("--output-dir", type=Path, default=output_dir)
    eval_parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Episode JSONL path (defaults below --output-dir)",
    )
    eval_parser.set_defaults(handler=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = args.handler(args)
    if isinstance(result, Path):
        print(f"checkpoint={result}")


if __name__ == "__main__":
    main()
