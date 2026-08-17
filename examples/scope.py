#!/usr/bin/env python3
"""Train or evaluate the SCOPE visual policy with CMA-ES.

This baseline intentionally lives outside ``src/dsle``. It consumes only the
same public pixel observations and action integers available to any user agent.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from numbers import Integral
from pathlib import Path

import numpy as np

import dsle

try:  # Support both ``python examples/scope.py`` and module imports.
    from .common import (
        append_jsonl,
        atomic_output,
        default_output_dir,
        prepare_output_directory,
        prepare_output_file,
    )
except ImportError:  # pragma: no cover - exercised by direct script execution
    from common import (
        append_jsonl,
        atomic_output,
        default_output_dir,
        prepare_output_directory,
        prepare_output_file,
    )

MAX_K = 600  # DSLE's visual contract is 600x800 before policy preprocessing.


def chromosome_size(k: int, actions: int = 14) -> int:
    if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
        raise ValueError("k must be a positive integer")
    if isinstance(actions, bool) or not isinstance(actions, Integral) or actions <= 0:
        raise ValueError("actions must be a positive integer")
    return int(k) + int(k) * int(actions) + int(actions)


def parse_instances(value: str) -> tuple[str, ...]:
    """Normalize and validate the live-game instance pool."""

    instances = tuple(part.strip() for part in value.split(",") if part.strip())
    if not instances:
        raise ValueError("--instances must contain at least one instance name")
    seen: set[str] = set()
    duplicates: set[str] = set()
    for instance in instances:
        if instance in seen:
            duplicates.add(instance)
        seen.add(instance)
    if duplicates:
        names = ", ".join(sorted(duplicates))
        raise ValueError(
            f"--instances contains duplicates ({names}); parallel workers must use "
            "different live game instances"
        )
    return instances


class ScopePolicy:
    def __init__(self, chromosome: np.ndarray, *, k: int = 100, percentile: float = 90.0):
        expected = chromosome_size(k)
        if k > MAX_K:
            raise ValueError(f"k cannot exceed {MAX_K} for DSLE's 600x800 observations")
        percentile = float(percentile)
        if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
            raise ValueError("percentile must be finite and in [0, 100]")
        flat = np.asarray(chromosome, dtype=np.float64).reshape(-1)
        if flat.size != expected:
            raise ValueError(f"SCOPE needs {expected} weights for k={k}, got {flat.size}")
        if not np.all(np.isfinite(flat)):
            raise ValueError("SCOPE weights must all be finite")
        self.k = int(k)
        self.percentile = percentile
        self.left = flat[:k].reshape(1, k)
        self.right = flat[k : k + k * 14].reshape(k, 14)
        self.bias = flat[-14:]

    def logits(self, observation: np.ndarray) -> np.ndarray:
        array = np.asarray(observation)
        if array.ndim != 3 or array.shape[0] != 1:
            raise ValueError(
                f"SCOPE requires one-channel CHW observations, got shape {array.shape}"
            )
        frame = array[0]
        if self.k > min(frame.shape):
            raise ValueError(
                f"k={self.k} exceeds the observation's smallest spatial dimension "
                f"({min(frame.shape)})"
            )
        try:
            from scipy.fft import dctn
        except ImportError as exc:
            raise RuntimeError(
                "SCOPE requires scipy; install `pip install -e '.[examples]'`"
            ) from exc
        transformed = dctn(frame, type=2, norm="ortho")[: self.k, : self.k].copy()
        threshold = np.percentile(np.abs(transformed), self.percentile)
        transformed[np.abs(transformed) < threshold] = 0.0
        return (self.left @ transformed @ self.right).reshape(-1) + self.bias

    def act(self, observation: np.ndarray) -> int:
        return int(np.argmax(self.logits(observation)))


def summarize_episode(
    info: dict[str, object], *, episode_return: float, instance: str
) -> dict[str, object]:
    """Return the SCOPE fitness and common episode metrics.

    Keeping this calculation beside the serial baseline lets other example-only
    launchers reuse the same optimizer objective without moving algorithm code
    into :mod:`dsle`.
    """

    player_hp = info.get("player_hp")
    player_max = info.get("player_hp_max")
    boss_hp = info.get("boss_hp")
    boss_max = info.get("boss_hp_max")
    player_fraction = (
        float(player_hp / max(1, player_max))
        if isinstance(player_hp, int) and isinstance(player_max, int)
        else 0.0
    )
    boss_fraction = (
        float(boss_hp / max(1, boss_max))
        if isinstance(boss_hp, int) and isinstance(boss_max, int)
        else 1.0
    )
    fitness = 100.0 * (player_fraction + 1.0 - boss_fraction)
    return {
        "fitness": fitness,
        "return": float(episode_return),
        "win": bool(info.get("win", False)),
        "length": int(info.get("step", 0)),
        "terminated_reason": info.get("terminated_reason"),
        "truncated_reason": info.get("truncated_reason"),
        "instance": instance,
    }


def evaluate_once(
    weights: np.ndarray, args: argparse.Namespace, instance: str
) -> dict[str, object]:
    # Validate the policy before allocating or mutating a live game backend.
    policy = ScopePolicy(weights, k=args.k, percentile=args.percentile)
    env = dsle.make(
        args.boss,
        instance=instance,
        difficulty=args.difficulty,
        obs_mode="grayscale",
        max_steps=args.max_steps,
    )
    episode_return = 0.0
    try:
        observation, info = env.reset()
        while True:
            observation, reward, terminated, truncated, info = env.step(policy.act(observation))
            episode_return += float(reward)
            if terminated or truncated:
                break
    finally:
        env.close()
    return summarize_episode(info, episode_return=episode_return, instance=instance)


def evaluate_population(
    solutions: list[np.ndarray], args: argparse.Namespace
) -> list[dict[str, object]]:
    instances = parse_instances(args.instances)
    results: list[dict[str, object]] = []
    for start in range(0, len(solutions), len(instances)):
        batch = solutions[start : start + len(instances)]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = [
                executor.submit(evaluate_once, weights, args, instances[index])
                for index, weights in enumerate(batch)
            ]
            results.extend(future.result() for future in futures)
    return results


def validate_args(args: argparse.Namespace, *, training: bool) -> None:
    """Reject invalid CLI values before allocating a live game backend."""

    chromosome_size(args.k)
    if args.k > MAX_K:
        raise ValueError(f"--k cannot exceed {MAX_K} for DSLE's 600x800 observations")
    percentile = float(args.percentile)
    if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("--percentile must be finite and in [0, 100]")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    parse_instances(args.instances)
    if training:
        sigma = float(args.sigma)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("--sigma must be finite and positive")
        if args.generations <= 0:
            raise ValueError("--generations must be positive")
        if args.population < 0 or args.population == 1:
            raise ValueError("--population must be 0 (CMA-ES default) or at least 2")
    else:
        if args.episodes <= 0:
            raise ValueError("--episodes must be positive")
        if not args.weights:
            raise ValueError("evaluate mode requires --weights")


def train(args: argparse.Namespace) -> None:
    validate_args(args, training=True)
    try:
        import cma
    except ImportError as exc:
        raise RuntimeError(
            "Training SCOPE requires cma; install `pip install -e '.[examples]'`"
        ) from exc
    output = prepare_output_directory(args.output)
    metrics_output = prepare_output_file(output / "metrics.jsonl")
    options: dict[str, object] = {"seed": args.seed}
    if args.population > 0:
        options["popsize"] = args.population
    strategy = cma.CMAEvolutionStrategy(
        np.zeros(chromosome_size(args.k), dtype=np.float64), args.sigma, options
    )
    best_fitness = float("-inf")
    best_weights: np.ndarray | None = None
    for generation in range(1, args.generations + 1):
        solutions = [np.asarray(item) for item in strategy.ask()]
        results = evaluate_population(solutions, args)
        fitnesses = [float(result["fitness"]) for result in results]
        strategy.tell(solutions, [-value for value in fitnesses])
        index = int(np.argmax(fitnesses))
        if fitnesses[index] > best_fitness:
            best_fitness = fitnesses[index]
            best_weights = solutions[index].copy()
            with atomic_output(output / "best.npy") as stream:
                np.save(stream, best_weights, allow_pickle=False)
        row = {
            "algorithm": "scope",
            "phase": "train",
            "boss": args.boss,
            "generation": generation,
            "best": max(fitnesses),
            "mean": float(np.mean(fitnesses)),
            "wins": sum(bool(result["win"]) for result in results),
            "evaluations": len(results),
        }
        append_jsonl(metrics_output, row)
        print(json.dumps(row), flush=True)
    if best_weights is None:
        raise RuntimeError("CMA-ES returned no candidates")


def evaluate(args: argparse.Namespace) -> None:
    validate_args(args, training=False)
    output = prepare_output_directory(args.output)
    metrics_output = prepare_output_file(output / "metrics.jsonl")
    weights = np.load(args.weights, allow_pickle=False)
    instance = parse_instances(args.instances)[0]
    rows = [evaluate_once(weights, args, instance) for _ in range(args.episodes)]
    for episode, row in enumerate(rows, start=1):
        append_jsonl(
            metrics_output,
            {
                "algorithm": "scope",
                "phase": "evaluate",
                "boss": args.boss,
                "episode": episode,
                **row,
            },
        )
    wins = sum(bool(row["win"]) for row in rows)
    print(
        json.dumps(
            {"episodes": len(rows), "wins": wins, "win_rate": wins / len(rows), "rows": rows},
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--boss", default="asylum_demon", choices=dsle.list_bosses())
    parser.add_argument("--difficulty", default="standard", choices=("standard", "boosted"))
    parser.add_argument("--instances", default="dsr-1", help="Comma-separated instance pool")
    parser.add_argument("--max-steps", type=int, default=7200)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument(
        "--population", type=int, default=0, help="0 uses CMA-ES's default (25 at k=100)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_dir() / "scope",
        help="Artifact directory for best.npy and metrics.jsonl",
    )
    parser.add_argument("--weights", help="best.npy for evaluate mode")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    elif not args.weights:
        parser.error("evaluate mode requires --weights")
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
