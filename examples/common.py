"""Shared evaluation loop used only by the baseline examples."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, cast

import numpy as np


class Policy(Protocol):
    def reset(self) -> None: ...

    def act(self, observation: np.ndarray, info: dict[str, object]) -> int: ...


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    reward: float
    steps: int
    win: bool
    terminated_reason: str | None
    truncated_reason: str | None
    elapsed_seconds: float


def default_output_dir() -> Path:
    """Return the shared example-output directory.

    ``DSLE_OUTPUT_DIR`` makes the examples container-friendly while retaining
    ``./runs`` as a predictable local default.
    """

    configured = os.environ.get("DSLE_OUTPUT_DIR") or "runs"
    return Path(configured).expanduser().resolve()


def prepare_output_target(output: str | Path) -> Path:
    """Resolve an output file and create its parent before a live env starts."""

    requested = Path(output).expanduser()
    destination = requested.parent.resolve() / requested.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        metadata = destination.lstat()
    except FileNotFoundError:
        pass
    else:
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
        ):
            raise ValueError(f"Output path must be one non-symlink regular file: {destination}")
    return destination


def prepare_output_directory(output: str | Path) -> Path:
    """Resolve and create an output directory before a live env starts."""

    destination = Path(output).expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Output path is not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def prepare_output_file(output: str | Path) -> Path:
    """Preflight an append-only output file and return its resolved path."""

    destination = prepare_output_target(output)
    descriptor = _open_append_file(destination)
    os.close(descriptor)
    return destination


def append_jsonl(output: str | Path, payload: Mapping[str, object]) -> None:
    """Append one JSON object to a previously prepared metrics file."""

    encoded = (json.dumps(dict(payload)) + "\n").encode("utf-8")
    descriptor = _open_append_file(Path(output))
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("Could not append DSLE example metrics")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_append_file(path: Path) -> int:
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError(f"Output path must be one non-symlink regular file: {path}")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


@contextmanager
def atomic_output(output: str | Path) -> Iterator[BinaryIO]:
    """Yield a private temporary file and atomically replace one output target."""

    destination = prepare_output_target(output)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            yield cast(BinaryIO, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary is None:  # Defensive invariant; NamedTemporaryFile always has a name.
            raise RuntimeError("Could not allocate a temporary output file")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_episodes(
    env,
    policy: Policy,
    episodes: int,
    output: str | Path | None = None,
    *,
    metadata: Mapping[str, object] | None = None,
) -> list[EpisodeResult]:
    """Evaluate a policy and optionally append one JSON object per episode."""

    results: list[EpisodeResult] = []
    try:
        if episodes <= 0:
            raise ValueError("episodes must be positive")
        destination = None if output is None else prepare_output_file(output)
        common_fields = dict(metadata or {})
        for episode in range(1, episodes + 1):
            policy.reset()
            observation, info = env.reset()
            episode_return = 0.0
            steps = 0
            started = time.monotonic()
            while True:
                action = policy.act(observation, info)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            result = EpisodeResult(
                episode=episode,
                reward=episode_return,
                steps=steps,
                win=bool(info.get("win", False)),
                terminated_reason=info.get("terminated_reason"),
                truncated_reason=info.get("truncated_reason"),
                elapsed_seconds=time.monotonic() - started,
            )
            results.append(result)
            if destination is not None:
                append_jsonl(
                    destination,
                    {
                        **common_fields,
                        "episode": result.episode,
                        "return": result.reward,
                        "length": result.steps,
                        "win": result.win,
                        "terminated_reason": result.terminated_reason,
                        "truncated_reason": result.truncated_reason,
                        "elapsed_seconds": result.elapsed_seconds,
                    },
                )
            print(
                f"episode={episode} reward={episode_return:.3f} steps={steps} "
                f"win={int(result.win)} reason={result.terminated_reason or result.truncated_reason}",
                flush=True,
            )
    finally:
        env.close()
    return results
