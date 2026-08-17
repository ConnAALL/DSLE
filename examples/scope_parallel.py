#!/usr/bin/env python3
"""Train SCOPE with CMA-ES on an API-managed pool of DSLE instances.

This is a host-side example. It starts one owned prebuilt container, bulk-starts
``dsr-1`` through ``dsr-N``, and reuses one environment proxy per instance while
evaluating each CMA-ES population in batches. The SCOPE implementation remains
in :mod:`examples.scope`; no baseline algorithm is part of :mod:`dsle`.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import time
import warnings
import zipfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

import dsle
from dsle.container import ContainerSession, default_output_directory

try:  # Support both ``python examples/scope_parallel.py`` and module imports.
    from .common import (
        append_jsonl,
        atomic_output,
        prepare_output_directory,
        prepare_output_file,
    )
    from .scope import MAX_K, ScopePolicy, chromosome_size, summarize_episode
except ImportError:  # pragma: no cover - exercised by direct script execution
    from common import (
        append_jsonl,
        atomic_output,
        prepare_output_directory,
        prepare_output_file,
    )
    from scope import MAX_K, ScopePolicy, chromosome_size, summarize_episode


class Environment(Protocol):
    """The small Gymnasium surface used by one population worker."""

    def reset(self) -> tuple[np.ndarray, dict[str, object]]: ...

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]: ...

    def close(self) -> None: ...


class EvolutionStrategy(Protocol):
    """CMA-ES methods needed by the training loop."""

    countiter: int
    mean: np.ndarray
    _dsle_resume_generation: int

    def ask(self) -> Sequence[Sequence[float] | np.ndarray]: ...

    def tell(self, solutions: Sequence[np.ndarray], values: Sequence[float]) -> None: ...


@dataclass(frozen=True)
class Worker:
    instance: str
    environment: Environment


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    generation_metrics: Path
    individual_metrics: Path
    checkpoints: Path
    latest_checkpoint: Path
    best_weights: Path
    latest_mean: Path
    state: Path


Evaluator = Callable[[np.ndarray, argparse.Namespace, Worker], dict[str, object]]
_CHECKPOINT_FORMAT = 1
_MAX_CHECKPOINT_BYTES = 2 << 20
_CHECKPOINT_MEMBERS = {
    "format_version.npy",
    "generation.npy",
    "dimension.npy",
    "mean.npy",
    "sigma.npy",
}


@dataclass(frozen=True)
class ResumeState:
    """Data-only state used to restart CMA-ES without deserializing code."""

    generation: int
    mean: np.ndarray
    sigma: float


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if not value else Path(value).expanduser()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boss", default="asylum_demon", choices=dsle.list_bosses())
    parser.add_argument("--difficulty", default="standard", choices=("standard", "boosted"))
    parser.add_argument(
        "--game-dir",
        type=Path,
        default=_environment_path("DSLE_GAME_DIR"),
        help="Host Dark Souls Remastered directory (default: DSLE_GAME_DIR)",
    )
    parser.add_argument(
        "--image",
        default=os.environ.get("DSLE_IMAGE", "dsle-runtime:0.1.0"),
        help="Existing local runtime image; this example never builds or pulls it",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_directory("scope-parallel"),
        help="Persistent host DSLE session directory",
    )
    parser.add_argument("--container-name", help="Optional owned-container name")
    parser.add_argument("--num-instances", type=int, default=2)
    parser.add_argument("--instance-mode", choices=("headless", "headless-vnc"), default="headless")
    parser.add_argument("--max-steps", type=int, default=7200)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument(
        "--generations",
        type=int,
        default=40,
        help="Generations to run now (additional generations when resuming)",
    )
    parser.add_argument(
        "--population", type=int, default=0, help="0 uses CMA-ES's population default"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Data-only .npz restart checkpoint written by this example",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Reject bad inputs before Docker or a game process is allocated."""

    chromosome_size(args.k)
    if args.k > MAX_K:
        raise ValueError(f"--k cannot exceed {MAX_K} for DSLE's 600x800 observations")
    percentile = float(args.percentile)
    if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError("--percentile must be finite and in [0, 100]")
    sigma = float(args.sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("--sigma must be finite and positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.generations <= 0:
        raise ValueError("--generations must be positive")
    if args.population < 0 or args.population == 1:
        raise ValueError("--population must be 0 (CMA-ES default) or at least 2")
    if (
        isinstance(args.num_instances, bool)
        or not isinstance(args.num_instances, int)
        or not 1 <= args.num_instances <= 30
    ):
        raise ValueError("--num-instances must be an integer in [1, 30]")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    if args.game_dir is None:
        raise ValueError("--game-dir is required when DSLE_GAME_DIR is unset")
    if not isinstance(args.image, str) or not args.image.strip():
        raise ValueError("--image must be non-empty")
    if args.resume_checkpoint is not None and not args.resume_checkpoint.is_file():
        raise FileNotFoundError(f"Optimizer checkpoint does not exist: {args.resume_checkpoint}")


def _new_strategy(
    args: argparse.Namespace,
    *,
    initial_mean: np.ndarray | None = None,
    sigma: float | None = None,
) -> EvolutionStrategy:
    try:
        import cma
    except ImportError as exc:
        raise RuntimeError(
            "Parallel SCOPE training requires cma; install `pip install -e '.[examples]'`"
        ) from exc
    options: dict[str, object] = {"seed": args.seed}
    if args.population > 0:
        options["popsize"] = args.population
    mean = (
        np.zeros(chromosome_size(args.k), dtype=np.float64)
        if initial_mean is None
        else np.asarray(initial_mean, dtype=np.float64).reshape(-1)
    )
    if mean.size != chromosome_size(args.k) or not np.all(np.isfinite(mean)):
        raise ValueError("Initial CMA-ES mean has the wrong size or contains non-finite values")
    selected_sigma = float(args.sigma if sigma is None else sigma)
    if not np.isfinite(selected_sigma) or selected_sigma <= 0.0:
        raise ValueError("Initial CMA-ES sigma must be finite and positive")
    return cma.CMAEvolutionStrategy(
        mean,
        selected_sigma,
        options,
    )


def _load_resume_state(path: Path, *, expected_weights: int) -> ResumeState:
    """Load one bounded numeric checkpoint without Python object deserialization."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError(f"Could not safely open resume checkpoint {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or not 0 < metadata.st_size <= _MAX_CHECKPOINT_BYTES
        ):
            raise ValueError(
                f"Resume checkpoint must be one regular file no larger than "
                f"{_MAX_CHECKPOINT_BYTES} bytes: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            with zipfile.ZipFile(stream) as archive:
                members = archive.infolist()
                names = [member.filename for member in members]
                if len(names) != len(set(names)) or set(names) != _CHECKPOINT_MEMBERS:
                    raise ValueError("Resume checkpoint has missing, duplicate, or unknown fields")
                if (
                    any(
                        member.compress_type != zipfile.ZIP_STORED
                        or member.flag_bits & 0x1
                        or member.file_size > _MAX_CHECKPOINT_BYTES
                        for member in members
                    )
                    or sum(member.file_size for member in members) > _MAX_CHECKPOINT_BYTES
                ):
                    raise ValueError("Resume checkpoint contains unsafe or oversized archive data")
            stream.seek(0)
            with np.load(stream, allow_pickle=False) as archive:
                format_array = np.asarray(archive["format_version"])
                generation_array = np.asarray(archive["generation"])
                dimension_array = np.asarray(archive["dimension"])
                sigma_array = np.asarray(archive["sigma"])
                mean = np.asarray(archive["mean"], dtype=np.float64)
    except (OSError, EOFError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"Could not load resume checkpoint {path}: {exc}") from exc
    finally:
        os.close(descriptor)

    integer_fields = {
        "format_version": format_array,
        "generation": generation_array,
        "dimension": dimension_array,
    }
    if any(value.shape != () or value.dtype.kind not in "iu" for value in integer_fields.values()):
        raise ValueError("Resume checkpoint integer metadata is malformed")
    format_version = int(format_array)
    generation = int(generation_array)
    dimension = int(dimension_array)
    if format_version != _CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported SCOPE checkpoint format: {format_version}")
    if generation < 0:
        raise ValueError("Resume checkpoint generation cannot be negative")
    if dimension != expected_weights or mean.shape != (expected_weights,):
        raise ValueError(
            f"Resume checkpoint contains {dimension} weights; expected {expected_weights}"
        )
    if not np.all(np.isfinite(mean)):
        raise ValueError("Resume checkpoint mean must contain only finite values")
    if sigma_array.shape != () or sigma_array.dtype.kind not in "iuf":
        raise ValueError("Resume checkpoint sigma is malformed")
    sigma = float(sigma_array)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("Resume checkpoint sigma must be finite and positive")
    return ResumeState(generation=generation, mean=mean, sigma=sigma)


def _strategy(args: argparse.Namespace) -> EvolutionStrategy:
    if args.resume_checkpoint is None:
        strategy = _new_strategy(args)
    else:
        state = _load_resume_state(
            args.resume_checkpoint,
            expected_weights=chromosome_size(args.k),
        )
        strategy = _new_strategy(args, initial_mean=state.mean, sigma=state.sigma)
        strategy._dsle_resume_generation = state.generation
    dimension = getattr(strategy, "N", chromosome_size(args.k))
    if int(dimension) != chromosome_size(args.k):
        raise ValueError(
            f"Optimizer dimension {dimension} does not match --k={args.k} "
            f"({chromosome_size(args.k)} weights)"
        )
    return strategy


def evaluate_individual(
    weights: np.ndarray, args: argparse.Namespace, worker: Worker
) -> dict[str, object]:
    """Reset one worker and run exactly one population individual to termination."""

    policy = ScopePolicy(weights, k=args.k, percentile=args.percentile)
    observation, info = worker.environment.reset()
    episode_return = 0.0
    steps = 0
    while True:
        action = policy.act(observation)
        observation, reward, terminated, truncated, info = worker.environment.step(action)
        episode_return += float(reward)
        steps += 1
        if terminated or truncated:
            break
    result = summarize_episode(
        info,
        episode_return=episode_return,
        instance=worker.instance,
    )
    result["length"] = steps
    return result


def _validated_result(result: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(result)
    for field in ("fitness", "return"):
        value = normalized.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"Population evaluator returned non-numeric {field}: {value!r}")
        if not np.isfinite(float(value)):
            raise RuntimeError(f"Population evaluator returned non-finite {field}: {value!r}")
        normalized[field] = float(value)
    return normalized


def evaluate_population(
    solutions: Sequence[np.ndarray],
    workers: Sequence[Worker],
    args: argparse.Namespace,
    *,
    evaluator: Evaluator = evaluate_individual,
    executor: Executor | None = None,
) -> list[dict[str, object]]:
    """Evaluate solutions in ordered batches no larger than the instance pool."""

    if not workers:
        raise ValueError("At least one environment worker is required")
    if len({worker.instance for worker in workers}) != len(workers):
        raise ValueError("Environment workers must use unique instances")

    def evaluate_batches(pool: Executor) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for start in range(0, len(solutions), len(workers)):
            batch = solutions[start : start + len(workers)]
            futures = [
                pool.submit(evaluator, np.asarray(weights), args, workers[index])
                for index, weights in enumerate(batch)
            ]
            results.extend(_validated_result(future.result()) for future in futures)
        return results

    if executor is not None:
        return evaluate_batches(executor)
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        return evaluate_batches(pool)


def _prepare_artifacts(output_dir: Path) -> ArtifactPaths:
    root = prepare_output_directory(output_dir / "results" / "scope-parallel")
    checkpoints = prepare_output_directory(root / "checkpoints")
    return ArtifactPaths(
        root=root,
        generation_metrics=prepare_output_file(root / "generations.jsonl"),
        individual_metrics=prepare_output_file(root / "individuals.jsonl"),
        checkpoints=checkpoints,
        latest_checkpoint=root / "scope-state-latest.npz",
        best_weights=root / "best.npy",
        latest_mean=root / "mean-latest.npy",
        state=root / "state.json",
    )


def _save_checkpoint(
    strategy: EvolutionStrategy,
    paths: ArtifactPaths,
    *,
    generation: int,
    checkpoint_every: int,
    fallback_sigma: float,
) -> None:
    mean = np.asarray(getattr(strategy, "mean", None), dtype=np.float64).reshape(-1)
    if mean.size == 0 or not np.all(np.isfinite(mean)):
        raise RuntimeError("CMA-ES returned an empty or non-finite mean")
    sigma = float(getattr(strategy, "sigma", fallback_sigma))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise RuntimeError("CMA-ES returned a non-finite or non-positive sigma")

    def write_checkpoint(path: Path) -> None:
        with atomic_output(path) as stream:
            np.savez(
                stream,
                format_version=np.asarray(_CHECKPOINT_FORMAT, dtype=np.uint16),
                generation=np.asarray(generation, dtype=np.int64),
                dimension=np.asarray(mean.size, dtype=np.int64),
                mean=mean,
                sigma=np.asarray(sigma, dtype=np.float64),
            )

    write_checkpoint(paths.latest_checkpoint)
    if generation % checkpoint_every == 0:
        write_checkpoint(paths.checkpoints / f"scope-state-generation-{generation:06d}.npz")
    with atomic_output(paths.latest_mean) as stream:
        np.save(stream, mean, allow_pickle=False)


def _write_run_config(args: argparse.Namespace, paths: ArtifactPaths) -> None:
    payload = {
        "algorithm": "scope",
        "boss": args.boss,
        "difficulty": args.difficulty,
        "game_dir": str(Path(args.game_dir).expanduser().resolve()),
        "image": args.image,
        "num_instances": args.num_instances,
        "instance_mode": args.instance_mode,
        "max_steps": args.max_steps,
        "k": args.k,
        "percentile": args.percentile,
        "sigma": args.sigma,
        "generations_this_run": args.generations,
        "population": args.population,
        "seed": args.seed,
        "resume_checkpoint": (
            None if args.resume_checkpoint is None else str(args.resume_checkpoint.resolve())
        ),
    }
    with atomic_output(paths.root / "run-config.json") as stream:
        stream.write((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _load_persisted_best(
    paths: ArtifactPaths,
    *,
    expected_weights: int,
) -> tuple[float, np.ndarray | None]:
    """Restore the best candidate when continuing in an existing output directory."""

    state_exists = paths.state.is_file()
    weights_exist = paths.best_weights.is_file()
    if not state_exists and not weights_exist:
        return float("-inf"), None
    if state_exists != weights_exist:
        raise RuntimeError("Existing parallel SCOPE output has incomplete best-candidate artifacts")
    try:
        state = json.loads(paths.state.read_text(encoding="utf-8"))
        fitness = state["best_fitness"]
        weights = np.asarray(
            np.load(paths.best_weights, allow_pickle=False),
            dtype=np.float64,
        ).reshape(-1)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not restore the persisted best candidate: {exc}") from exc
    if isinstance(fitness, bool) or not isinstance(fitness, (int, float)):
        raise RuntimeError("Persisted best_fitness must be numeric")
    best_fitness = float(fitness)
    if not np.isfinite(best_fitness):
        raise RuntimeError("Persisted best_fitness must be finite")
    if weights.size != expected_weights or not np.all(np.isfinite(weights)):
        raise RuntimeError(
            "Persisted best.npy does not contain the expected finite SCOPE chromosome"
        )
    return best_fitness, weights


def optimize(
    strategy: EvolutionStrategy,
    workers: Sequence[Worker],
    args: argparse.Namespace,
    paths: ArtifactPaths,
    *,
    evaluator: Evaluator = evaluate_individual,
) -> dict[str, object]:
    """Run CMA-ES generations against an already-started environment pool."""

    expected = chromosome_size(args.k)
    best_fitness, best_weights = _load_persisted_best(
        paths,
        expected_weights=expected,
    )
    first_generation = (
        int(getattr(strategy, "_dsle_resume_generation", getattr(strategy, "countiter", 0))) + 1
    )
    last_generation = first_generation + args.generations - 1
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        for generation in range(first_generation, last_generation + 1):
            started = time.monotonic()
            solutions = [np.asarray(item, dtype=np.float64).reshape(-1) for item in strategy.ask()]
            if not solutions:
                raise RuntimeError("CMA-ES returned an empty population")
            if any(solution.size != expected for solution in solutions):
                sizes = sorted({solution.size for solution in solutions})
                raise RuntimeError(f"CMA-ES returned chromosome sizes {sizes}; expected {expected}")
            results = evaluate_population(
                solutions,
                workers,
                args,
                evaluator=evaluator,
                executor=executor,
            )
            fitnesses = [float(result["fitness"]) for result in results]
            returns = [float(result["return"]) for result in results]
            strategy.tell(solutions, [-fitness for fitness in fitnesses])

            best_index = int(np.argmax(fitnesses))
            if fitnesses[best_index] > best_fitness:
                best_fitness = fitnesses[best_index]
                best_weights = solutions[best_index].copy()
                with atomic_output(paths.best_weights) as stream:
                    np.save(stream, best_weights, allow_pickle=False)

            for index, result in enumerate(results, start=1):
                append_jsonl(
                    paths.individual_metrics,
                    {
                        "algorithm": "scope",
                        "phase": "train",
                        "boss": args.boss,
                        "generation": generation,
                        "individual": index,
                        **result,
                    },
                )
            row: dict[str, object] = {
                "algorithm": "scope",
                "phase": "train",
                "boss": args.boss,
                "generation": generation,
                "best_fitness": max(fitnesses),
                "mean_fitness": float(np.mean(fitnesses)),
                "best_return": max(returns),
                "mean_return": float(np.mean(returns)),
                "wins": sum(bool(result.get("win", False)) for result in results),
                "evaluations": len(results),
                "instances": len(workers),
                "elapsed_seconds": time.monotonic() - started,
            }
            append_jsonl(paths.generation_metrics, row)
            _save_checkpoint(
                strategy,
                paths,
                generation=generation,
                checkpoint_every=args.checkpoint_every,
                fallback_sigma=args.sigma,
            )
            with atomic_output(paths.state) as stream:
                stream.write(
                    (
                        json.dumps(
                            {
                                "generation": generation,
                                "best_fitness": best_fitness,
                                "best_weights": paths.best_weights.name,
                                "latest_checkpoint": paths.latest_checkpoint.name,
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n"
                    ).encode("utf-8")
                )
            print(json.dumps(row), flush=True)
    if best_weights is None:
        raise RuntimeError("CMA-ES returned no candidates")
    return {
        "first_generation": first_generation,
        "last_generation": last_generation,
        "best_fitness": best_fitness,
        "best_weights": paths.best_weights,
        "resume_checkpoint": paths.latest_checkpoint,
        "artifacts": paths.root,
    }


def train(
    args: argparse.Namespace,
    *,
    session_factory: Callable[..., Any] = ContainerSession,
    strategy: EvolutionStrategy | None = None,
    evaluator: Evaluator = evaluate_individual,
) -> dict[str, object]:
    """Own the container, worker pool, optimizer, and persistent artifacts."""

    validate_args(args)
    selected_strategy = _strategy(args) if strategy is None else strategy
    instance_numbers = range(1, args.num_instances + 1)
    vnc_ports = (
        tuple(5900 + index for index in instance_numbers)
        if args.instance_mode == "headless-vnc"
        else ()
    )
    session = session_factory(
        image=args.image,
        game_dir=args.game_dir,
        output_dir=args.output_dir,
        name=args.container_name,
        vnc_ports=vnc_ports,
    )
    workers: list[Worker] = []
    failed = False
    try:
        session.start()
        session.start_instances(args.num_instances, mode=args.instance_mode)
        paths = _prepare_artifacts(Path(session.output_dir))
        _write_run_config(args, paths)
        for index in instance_numbers:
            instance = f"dsr-{index}"
            environment = session.make(
                args.boss,
                instance=instance,
                start_instance=False,
                difficulty=args.difficulty,
                obs_mode="grayscale",
                max_steps=args.max_steps,
            )
            workers.append(Worker(instance=instance, environment=environment))
        return optimize(
            selected_strategy,
            workers,
            args,
            paths,
            evaluator=evaluator,
        )
    except BaseException:
        failed = True
        raise
    finally:
        for worker in reversed(workers):
            with suppress(Exception):
                worker.environment.close()
        if failed:
            try:
                session.close()
            except Exception as exc:
                warnings.warn(
                    f"Failed to remove the owned DSLE container during cleanup: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            session.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = train(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in summary.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
