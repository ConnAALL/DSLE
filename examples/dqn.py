#!/usr/bin/env python3
"""Train or evaluate a small DQN baseline against the public DSLE API.

The Q-network uses the 84x84 grayscale Nature CNN from the paper.  This file
is deliberately self-contained: it is example agent code, not part of
``dsle``.

Run inside a DSLE NVIDIA container, for example::

    python examples/dqn.py train --boss asylum_demon --instance dsr-1
    python examples/dqn.py eval --boss asylum_demon --instance dsr-1 \
        --checkpoint runs/dqn_asylum_demon_seed42.pt

``--seed`` controls only agent initialization, exploration, and replay
sampling.  Dark Souls runs separately and its internal RNG is not seedable.
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

import dsle

try:  # Support both ``python examples/dqn.py`` and module imports.
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
REPLAY_SIZE = 100_000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
LEARNING_STARTS = 10_000
TRAIN_FREQUENCY = 4
TARGET_UPDATE_FREQUENCY = 1_000


def _layer_init(layer: nn.Module, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)  # type: ignore[attr-defined]
    nn.init.constant_(layer.bias, bias)  # type: ignore[attr-defined]
    return layer


class QNetwork(nn.Module):
    """Nature CNN mapping one 84x84 grayscale observation to action values."""

    def __init__(self, n_actions: int):
        super().__init__()
        self.network = nn.Sequential(
            _layer_init(nn.Conv2d(1, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            _layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            _layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
            _layer_init(nn.Linear(512, n_actions), std=0.01),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation.float() / 255.0)


def resize_observation(observation: np.ndarray) -> np.ndarray:
    """Convert a DSLE grayscale frame to a ``(1, 84, 84)`` uint8 array."""

    array = np.asarray(observation)
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError(
            "DQN expects dsle.make(..., obs_mode='grayscale') observations with "
            f"shape (1, H, W), received {array.shape}"
        )
    frame = torch.from_numpy(np.ascontiguousarray(array)).float().unsqueeze(0)
    resized = F.interpolate(frame, size=IMAGE_SIZE, mode="area").squeeze(0)
    return resized.round().clamp_(0, 255).to(torch.uint8).numpy()


class ReplayBuffer:
    """Fixed-size circular replay memory with compact uint8 image storage."""

    def __init__(self, capacity: int):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Replay capacity must be a positive integer")
        self.capacity = capacity
        self._observations: np.ndarray | None = None
        self._next_observations: np.ndarray | None = None
        self._actions = np.empty(capacity, dtype=np.int64)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._terminals = np.empty(capacity, dtype=np.float32)
        self._position = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        terminal: bool,
    ) -> None:
        if self._observations is None:
            shape = (self.capacity, *observation.shape)
            self._observations = np.empty(shape, dtype=np.uint8)
            self._next_observations = np.empty(shape, dtype=np.uint8)
        if observation.shape != self._observations.shape[1:]:
            raise ValueError("Replay observations changed shape during training")

        next_observations = self._next_observations
        if next_observations is None:
            raise RuntimeError("Replay buffer storage was not initialized")
        self._observations[self._position] = observation
        next_observations[self._position] = next_observation
        self._actions[self._position] = action
        self._rewards[self._position] = reward
        self._terminals[self._position] = float(terminal)
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Replay batch size must be a positive integer")
        if batch_size > self._size:
            raise ValueError(f"Cannot sample {batch_size} transitions from {self._size}")
        observations = self._observations
        next_observations = self._next_observations
        if observations is None or next_observations is None:
            raise RuntimeError("Replay buffer has no initialized transitions")
        indices = rng.integers(0, self._size, size=batch_size)
        return (
            torch.as_tensor(observations[indices], device=device),
            torch.as_tensor(self._actions[indices], device=device),
            torch.as_tensor(self._rewards[indices], device=device),
            torch.as_tensor(next_observations[indices], device=device),
            torch.as_tensor(self._terminals[indices], device=device),
        )


def epsilon_at_step(
    step: int,
    duration: int,
    start: float = EPSILON_START,
    end: float = EPSILON_END,
) -> float:
    """Linearly anneal epsilon and clamp it at the requested endpoint."""

    if duration <= 0:
        return end
    if step <= 0:
        return start
    if step >= duration:
        return end
    fraction = step / duration
    return start + fraction * (end - start)


def terminal_for_bootstrap(
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> bool:
    """Return whether a transition must suppress value bootstrapping.

    Gymnasium time-limit truncations retain a valid final observation, so DQN
    should bootstrap through them. Runtime failures are conservatively treated
    as terminal because their final observation may only be the last known
    frame rather than a real successor state.
    """

    if terminated:
        return True
    if not truncated:
        return False
    return info.get("truncated_reason") != "max_steps"


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
    return Path(args.output_dir) / f"dqn_{args.boss}_seed{args.seed}.pt"


def _metrics_path(args: argparse.Namespace, phase: str) -> Path:
    explicit = getattr(args, "metrics_output", None)
    if explicit is not None:
        return Path(explicit)
    output_dir = Path(getattr(args, "output_dir", default_output_dir()))
    return output_dir / f"dqn_{args.boss}_seed{args.seed}_{phase}.jsonl"


def _save_checkpoint(
    path: Path,
    q_network: QNetwork,
    target_network: QNetwork,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    n_actions: int,
    args: argparse.Namespace,
) -> None:
    """Save model state without embedding the potentially multi-gigabyte replay memory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "handler"
    }
    with atomic_output(path) as stream:
        torch.save(
            {
                "algorithm": "dqn",
                "model_state_dict": q_network.state_dict(),
                "target_state_dict": target_network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
                "n_actions": n_actions,
                "config": config,
            },
            stream,
        )


def _load_network(path: Path, n_actions: int, device: torch.device) -> QNetwork:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("algorithm") != "dqn":
        raise ValueError(f"{path} is not a DQN checkpoint")
    saved_actions = int(checkpoint.get("n_actions", n_actions))
    if saved_actions != n_actions:
        raise ValueError(
            f"Checkpoint uses {saved_actions} actions, but this environment exposes {n_actions}"
        )
    network = QNetwork(n_actions).to(device)
    network.load_state_dict(checkpoint["model_state_dict"])
    return network


def _validate_train_args(args: argparse.Namespace) -> None:
    positive = {
        "total_timesteps": args.total_timesteps,
        "replay_size": args.replay_size,
        "batch_size": args.batch_size,
        "learning_starts": args.learning_starts,
        "train_frequency": args.train_frequency,
        "target_update_frequency": args.target_update_frequency,
        "checkpoint_interval": args.checkpoint_interval,
        "log_interval": args.log_interval,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"These options must be positive: {', '.join(invalid)}")
    if args.replay_size < args.batch_size:
        raise ValueError("--replay-size must be at least --batch-size")
    floating = {
        "gamma": args.gamma,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "exploration_fraction": args.exploration_fraction,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
    }
    nonfinite = [name for name, value in floating.items() if not math.isfinite(value)]
    if nonfinite:
        raise ValueError(f"These options must be finite: {', '.join(nonfinite)}")
    if not 0.0 < args.gamma <= 1.0:
        raise ValueError("--gamma must be in (0, 1]")
    if not 0.0 <= args.epsilon_end <= args.epsilon_start <= 1.0:
        raise ValueError("Epsilon values must satisfy 0 <= end <= start <= 1")
    if not 0.0 <= args.exploration_fraction <= 1.0:
        raise ValueError("--exploration-fraction must be in [0, 1]")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.max_grad_norm <= 0.0:
        raise ValueError("--max-grad-norm must be positive")


def _validate_environment_args(args: argparse.Namespace) -> None:
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")


def _greedy_action(network: QNetwork, observation: np.ndarray, device: torch.device) -> int:
    tensor = torch.as_tensor(observation, device=device).unsqueeze(0)
    with torch.no_grad():
        return int(network(tensor).argmax(dim=1).item())


def train(args: argparse.Namespace) -> Path:
    """Train DQN for exactly ``total_timesteps`` environment transitions."""

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
        q_network = QNetwork(n_actions).to(device)
        target_network = QNetwork(n_actions).to(device)
        target_network.load_state_dict(q_network.state_dict())
        target_network.eval()
        optimizer = torch.optim.Adam(q_network.parameters(), lr=args.learning_rate)
        replay = ReplayBuffer(args.replay_size)

        raw_observation, _ = env.reset()
        observation = resize_observation(raw_observation)
        episode_return = 0.0
        episode_length = 0
        episodes = 0
        exploration_steps = max(1, int(args.total_timesteps * args.exploration_fraction))
        started = time.monotonic()

        print(
            f"DQN train: boss={args.boss} instance={args.instance} "
            f"device={device} checkpoint={checkpoint_path}"
        )

        for global_step in range(1, args.total_timesteps + 1):
            epsilon = epsilon_at_step(
                global_step - 1,
                exploration_steps,
                start=args.epsilon_start,
                end=args.epsilon_end,
            )
            if rng.random() < epsilon:
                action = int(rng.integers(n_actions))
            else:
                action = _greedy_action(q_network, observation, device)

            raw_next_observation, reward, terminated, truncated, info = env.step(action)
            next_observation = resize_observation(raw_next_observation)
            done = bool(terminated or truncated)
            replay.add(
                observation,
                action,
                float(reward),
                next_observation,
                terminal_for_bootstrap(terminated, truncated, info),
            )

            episode_return += float(reward)
            episode_length += 1
            if done:
                episodes += 1
                won = bool(info.get("win", False))
                append_jsonl(
                    metrics_path,
                    {
                        "algorithm": "dqn",
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
                    raw_observation, _ = env.reset()
                    observation = resize_observation(raw_observation)
                episode_return = 0.0
                episode_length = 0
            else:
                observation = next_observation

            ready = global_step >= args.learning_starts and len(replay) >= args.batch_size
            if ready and global_step % args.train_frequency == 0:
                (
                    batch_observations,
                    batch_actions,
                    batch_rewards,
                    batch_next_observations,
                    batch_terminals,
                ) = replay.sample(args.batch_size, rng, device)
                with torch.no_grad():
                    next_values = target_network(batch_next_observations).max(dim=1).values
                    targets = batch_rewards + args.gamma * (1.0 - batch_terminals) * next_values
                predictions = (
                    q_network(batch_observations).gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                )
                loss = F.mse_loss(predictions, targets)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_network.parameters(), args.max_grad_norm)
                optimizer.step()

            if ready and global_step % args.target_update_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

            if global_step % args.checkpoint_interval == 0:
                _save_checkpoint(
                    checkpoint_path,
                    q_network,
                    target_network,
                    optimizer,
                    global_step=global_step,
                    n_actions=n_actions,
                    args=args,
                )

            if global_step % args.log_interval == 0 or global_step == args.total_timesteps:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"step={global_step}/{args.total_timesteps} epsilon={epsilon:.4f} "
                    f"replay={len(replay)} steps_per_second={global_step / elapsed:.2f}"
                )

        _save_checkpoint(
            checkpoint_path,
            q_network,
            target_network,
            optimizer,
            global_step=args.total_timesteps,
            n_actions=n_actions,
            args=args,
        )
        return checkpoint_path
    finally:
        if env is not None:
            env.close()


def evaluate(args: argparse.Namespace) -> None:
    """Run greedy policy episodes from a saved checkpoint."""

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    _validate_environment_args(args)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"DQN checkpoint does not exist: {checkpoint_path}")
    metrics_path = _metrics_path(args, "eval").expanduser().resolve()
    if metrics_path == checkpoint_path:
        raise ValueError("--metrics-output and --checkpoint must be different files")
    metrics_path = prepare_output_file(metrics_path)
    _seed_agent(args.seed)
    device = _device(args.device)
    env = None

    try:
        env = _make_env(args)
        network = _load_network(checkpoint_path, int(env.action_space.n), device)
        network.eval()
        returns: list[float] = []
        victories = 0

        for episode in range(1, args.episodes + 1):
            raw_observation, _ = env.reset()
            observation = resize_observation(raw_observation)
            episode_return = 0.0
            episode_length = 0
            done = False
            final_info: dict[str, Any] = {}
            while not done:
                action = _greedy_action(network, observation, device)
                raw_observation, reward, terminated, truncated, final_info = env.step(action)
                observation = resize_observation(raw_observation)
                episode_return += float(reward)
                episode_length += 1
                done = bool(terminated or truncated)

            won = bool(final_info.get("win", False))
            victories += int(won)
            returns.append(episode_return)
            append_jsonl(
                metrics_path,
                {
                    "algorithm": "dqn",
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

    train_parser = commands.add_parser("train", help="Train a DQN policy")
    _add_environment_arguments(train_parser)
    train_parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    train_parser.add_argument("--replay-size", type=int, default=REPLAY_SIZE)
    train_parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    train_parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    train_parser.add_argument("--gamma", type=float, default=GAMMA)
    train_parser.add_argument("--epsilon-start", type=float, default=EPSILON_START)
    train_parser.add_argument("--epsilon-end", type=float, default=EPSILON_END)
    train_parser.add_argument(
        "--exploration-fraction",
        type=float,
        default=0.1,
        help="Fraction of training over which epsilon is annealed",
    )
    train_parser.add_argument("--learning-starts", type=int, default=LEARNING_STARTS)
    train_parser.add_argument("--train-frequency", type=int, default=TRAIN_FREQUENCY)
    train_parser.add_argument(
        "--target-update-frequency", type=int, default=TARGET_UPDATE_FREQUENCY
    )
    train_parser.add_argument("--max-grad-norm", type=float, default=10.0)
    train_parser.add_argument("--checkpoint", type=Path)
    train_parser.add_argument("--checkpoint-interval", type=int, default=10_000)
    train_parser.add_argument("--log-interval", type=int, default=1_000)
    train_parser.add_argument("--output-dir", type=Path, default=output_dir)
    train_parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Episode JSONL path (defaults below --output-dir)",
    )
    train_parser.set_defaults(handler=train)

    eval_parser = commands.add_parser("eval", help="Evaluate a DQN checkpoint")
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
