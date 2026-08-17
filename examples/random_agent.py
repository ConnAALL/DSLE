#!/usr/bin/env python3
"""Uniform-random DSLE reference baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # Support both ``python examples/random_agent.py`` and module imports.
    from .common import default_output_dir, prepare_output_file, run_episodes
except ImportError:  # pragma: no cover - exercised by direct script execution
    from common import default_output_dir, prepare_output_file, run_episodes

import dsle


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space

    def reset(self) -> None:
        pass

    def act(self, _observation, _info) -> int:
        return int(self.action_space.sample())


def validate_args(args: argparse.Namespace) -> None:
    """Reject invalid values before allocating a live game environment."""

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")


def run(args: argparse.Namespace):
    validate_args(args)
    output = prepare_output_file(args.output)
    env = dsle.make(
        args.boss,
        instance=args.instance,
        difficulty=args.difficulty,
        max_steps=args.max_steps,
    )
    try:
        action_space = env.action_space
        action_space.seed(args.seed)
        policy = RandomPolicy(action_space)
    except BaseException:
        env.close()
        raise
    return run_episodes(
        env,
        policy,
        args.episodes,
        output,
        metadata={
            "algorithm": "random",
            "phase": "evaluate",
            "boss": args.boss,
            "instance": args.instance,
            "seed": args.seed,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boss", default="asylum_demon", choices=dsle.list_bosses())
    parser.add_argument("--instance", default="dsr-1")
    parser.add_argument("--difficulty", default="standard", choices=("standard", "boosted"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=7200)
    parser.add_argument(
        "--seed", type=int, default=42, help="Seeds action sampling, not game dynamics"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_dir() / "random.jsonl",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
