"""Executable fingerprint gate for the memory-instrumented DSR v1.04 runtime."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from dsle.exceptions import RuntimeUnavailableError

GAME_EXECUTABLE = "DarkSoulsRemastered.exe"
REQUIRED_GAME_DIRECTORIES = ("chr", "event", "map", "mtd", "param", "script")
ALLOW_UNVERIFIED_ENV = "DSLE_ALLOW_UNVERIFIED_GAME"

# The benchmark's supplied DSR v1.04 executable. Memory offsets and writes in
# memory.py are validated against this exact binary before a live instance starts.
SUPPORTED_EXECUTABLES = {
    "a45aaa36dd2f6cc151670a639ea5547043cf38ea79ff4178b963c6ed71f98d7b": (
        "Dark Souls: Remastered 1.04 benchmark build",
        50_286_344,
    ),
}


@dataclass(frozen=True)
class GameBuild:
    """Identity and verification status of one supplied executable."""

    path: Path
    sha256: str
    size: int
    name: str
    verified: bool


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file without loading the executable into memory at once."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def allow_unverified_from_environment() -> bool:
    """Return the explicit unsafe override, rejecting ambiguous values."""

    value = os.environ.get(ALLOW_UNVERIFIED_ENV, "0").strip().casefold()
    if value in {"0", "false", "no", "off", ""}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise RuntimeUnavailableError(f"{ALLOW_UNVERIFIED_ENV} must be a boolean value, got {value!r}")


def verify_game_executable(
    executable: str | Path,
    *,
    allow_unverified: bool | None = None,
) -> GameBuild:
    """Require a known v1.04 executable before enabling process-memory writes."""

    path = Path(executable).expanduser().resolve(strict=False)
    if not path.is_file():
        raise RuntimeUnavailableError(f"Game executable does not exist: {path}")
    if path.name.casefold() != GAME_EXECUTABLE.casefold():
        raise RuntimeUnavailableError(
            f"Expected {GAME_EXECUTABLE!r}, received executable {path.name!r}"
        )
    size = path.stat().st_size
    with path.open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise RuntimeUnavailableError(f"Game executable is not a Windows PE file: {path}")
    digest = sha256_file(path)
    supported = SUPPORTED_EXECUTABLES.get(digest)
    if supported is not None:
        name, expected_size = supported
        if size != expected_size:
            raise RuntimeUnavailableError(
                f"Known game hash has unexpected size {size}; expected {expected_size}: {path}"
            )
        return GameBuild(path, digest, size, name, True)

    if allow_unverified is not None and not isinstance(allow_unverified, bool):
        raise ValueError("allow_unverified must be a boolean or None")
    unsafe = allow_unverified_from_environment() if allow_unverified is None else allow_unverified
    if not unsafe:
        raise RuntimeUnavailableError(
            "Unsupported DarkSoulsRemastered.exe fingerprint. DSLE's memory offsets and writes "
            f"target the verified 1.04 benchmark build; detected sha256={digest}, size={size}. "
            f"Use the supported executable or explicitly set {ALLOW_UNVERIFIED_ENV}=1 at your "
            "own risk after verifying that its memory layout is identical."
        )
    return GameBuild(path, digest, size, "unverified override", False)


__all__ = [
    "ALLOW_UNVERIFIED_ENV",
    "GAME_EXECUTABLE",
    "REQUIRED_GAME_DIRECTORIES",
    "SUPPORTED_EXECUTABLES",
    "GameBuild",
    "allow_unverified_from_environment",
    "sha256_file",
    "verify_game_executable",
]
