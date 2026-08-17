"""Finalize persistent container output for the calling host user."""

from __future__ import annotations

import argparse
import os
import stat
from collections.abc import Sequence
from pathlib import Path

from dsle.exceptions import ConfigurationError

MARKER = ".dsle-output"
DURABLE_DIRECTORIES = (
    "config",
    "dxvk-cache",
    "logs",
    "pulse",
    "recordings",
    "results",
    "rpc",
    "run",
    "wineprefixes",
    "xdg",
)


def _assign(path: Path, uid: int, gid: int, *, directory: bool) -> None:
    metadata = path.lstat()
    expected_kind = stat.S_ISDIR if directory else stat.S_ISREG
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not expected_kind(metadata.st_mode)
        or (not directory and metadata.st_nlink != 1)
    ):
        return
    flags = os.O_RDONLY
    if directory and hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not expected_kind(opened.st_mode):
            raise ConfigurationError(f"DSLE output path changed during finalization: {path}")
        os.fchown(descriptor, uid, gid)
        required = 0o700 if directory else 0o600
        os.fchmod(descriptor, stat.S_IMODE(opened.st_mode) | required)
    finally:
        os.close(descriptor)


def finalize_output(state_dir: str | Path, uid: int, gid: int) -> None:
    """Make known DSLE output trees host-owned without following symlinks."""

    if uid < 0 or gid < 0:
        raise ConfigurationError("Host uid and gid cannot be negative")
    root = Path(state_dir).resolve(strict=True)
    marker = root / MARKER
    marker_metadata = marker.lstat()
    if (
        stat.S_ISLNK(marker_metadata.st_mode)
        or not stat.S_ISREG(marker_metadata.st_mode)
        or marker.read_bytes() != b"dsle-output\n"
    ):
        raise ConfigurationError(f"Invalid DSLE output marker: {marker}")

    for name in DURABLE_DIRECTORIES:
        top = root / name
        try:
            metadata = top.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ConfigurationError(f"Unsafe reserved DSLE output path: {top}")
        for current, directories, files in os.walk(top, topdown=False, followlinks=False):
            current_path = Path(current)
            for filename in files:
                _assign(current_path / filename, uid, gid, directory=False)
            for dirname in directories:
                candidate = current_path / dirname
                if not candidate.is_symlink():
                    _assign(candidate, uid, gid, directory=True)
            _assign(current_path, uid, gid, directory=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--gid", type=int, required=True)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    if os.environ.get("DSLE_CONTAINER") != "1":
        raise SystemExit("dsle.runtime.output may only run inside the DSLE container")
    args = build_parser().parse_args(arguments)
    finalize_output(args.state_dir, args.uid, args.gid)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised in the NVIDIA image
    raise SystemExit(main())
