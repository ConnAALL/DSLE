"""Preflight diagnostics for DSLE hosts and NVIDIA runtime containers."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from dsle._version import __version__
from dsle.exceptions import RuntimeUnavailableError
from dsle.runtime.version import REQUIRED_GAME_DIRECTORIES, verify_game_executable


@dataclass(frozen=True)
class Check:
    """One diagnostic result."""

    name: str
    status: str
    detail: str


def _run(command: Sequence[str], timeout: float = 10.0) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, f"{type(exc).__name__}: {exc}"
    return completed.returncode, completed.stdout.strip()


def _command_check(name: str, command: Sequence[str]) -> Check:
    executable = command[0]
    if shutil.which(executable) is None:
        return Check(name, "fail", f"{executable} is not installed")
    code, output = _run(command)
    detail = output.splitlines()[0] if output else f"exit code {code}"
    return Check(name, "pass" if code == 0 else "fail", detail)


def _game_checks(game_dir: Path | None, required: bool) -> list[Check]:
    if game_dir is None:
        status = "fail" if required else "warn"
        return [Check("game", status, "set --game-dir or DSLE_GAME_DIR")]

    checks: list[Check] = []
    if not game_dir.is_dir():
        return [Check("game", "fail", f"directory does not exist: {game_dir}")]

    executable = game_dir / "DarkSoulsRemastered.exe"
    checks.append(
        Check(
            "game executable",
            "pass" if executable.is_file() else "fail",
            str(executable),
        )
    )
    if executable.is_file():
        try:
            build = verify_game_executable(executable)
            checks.append(
                Check(
                    "game build",
                    "pass" if build.verified else "warn",
                    f"{build.name}; sha256={build.sha256}",
                )
            )
        except RuntimeUnavailableError as exc:
            checks.append(Check("game build", "fail", str(exc)))
    missing = [name for name in REQUIRED_GAME_DIRECTORIES if not (game_dir / name).is_dir()]
    checks.append(
        Check(
            "game assets",
            "pass" if not missing else "fail",
            "core asset directories present" if not missing else f"missing: {', '.join(missing)}",
        )
    )
    return checks


def _mount_is_read_only(path: Path) -> bool | None:
    """Return whether the exact Linux mount point is read-only, if discoverable."""
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        return None
    resolved = str(path.resolve())
    for line in mountinfo.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        mount_point = fields[4].replace("\\040", " ")
        if mount_point == resolved:
            return "ro" in fields[5].split(",")
    return None


def _linux_security_checks() -> list[Check]:
    checks: list[Check] = []
    status_path = Path("/proc/self/status")
    status = (
        status_path.read_text(encoding="utf-8", errors="replace") if status_path.is_file() else ""
    )
    fields = {}
    for line in status.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key] = value.strip()

    try:
        effective = int(fields.get("CapEff", "0"), 16)
    except ValueError:
        effective = 0
    has_ptrace = bool(effective & (1 << 19))
    checks.append(
        Check(
            "SYS_PTRACE capability",
            "pass" if has_ptrace else "fail",
            "effective" if has_ptrace else "missing from CapEff",
        )
    )

    seccomp = fields.get("Seccomp")
    checks.append(
        Check(
            "seccomp profile",
            "pass" if seccomp == "0" else "fail",
            "unconfined" if seccomp == "0" else f"Seccomp={seccomp or 'unknown'}",
        )
    )

    apparmor_path = Path("/proc/self/attr/current")
    apparmor = (
        apparmor_path.read_text(encoding="utf-8", errors="replace").strip()
        if apparmor_path.is_file()
        else "unknown"
    )
    apparmor_unconfined = (
        apparmor in {"unconfined", "docker-default (enforce)"}
        and apparmor != "docker-default (enforce)"
    )
    checks.append(
        Check(
            "AppArmor profile",
            "pass" if apparmor_unconfined else "fail",
            apparmor,
        )
    )
    return checks


def _host_checks() -> list[Check]:
    checks = [
        _command_check("Docker CLI", ("docker", "--version")),
        _command_check("Docker Compose", ("docker", "compose", "version")),
        _command_check("Docker daemon", ("docker", "info", "--format", "{{.ServerVersion}}")),
        _command_check(
            "NVIDIA GPU",
            ("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
        ),
    ]
    code, output = _run(("docker", "info", "--format", "{{json .Runtimes}}"))
    has_runtime = code == 0 and '"nvidia"' in output
    checks.append(
        Check(
            "NVIDIA Container Runtime",
            "pass" if has_runtime else "fail",
            "registered with Docker" if has_runtime else "runtime named 'nvidia' was not found",
        )
    )
    return checks


def _resource_checks(path: Path) -> list[Check]:
    checks: list[Check] = []
    architecture = platform.machine().lower()
    supported_architecture = architecture in {"amd64", "x86_64"}
    checks.append(
        Check(
            "CPU architecture",
            "pass" if supported_architecture else "fail",
            architecture,
        )
    )

    memory_path = Path("/proc/meminfo")
    memory_gib: float | None = None
    if memory_path.is_file():
        for line in memory_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                memory_gib = int(line.split()[1]) / 1024 / 1024
                break
    checks.append(
        Check(
            "system memory",
            "pass" if memory_gib is not None and memory_gib >= 8 else "warn",
            f"{memory_gib:.1f} GiB total" if memory_gib is not None else "could not read MemTotal",
        )
    )

    usage = shutil.disk_usage(path)
    free_gib = usage.free / 1024**3
    checks.append(
        Check(
            "free disk",
            "pass" if free_gib >= 30 else "warn",
            f"{free_gib:.1f} GiB available at {path}",
        )
    )
    return checks


def _container_checks() -> list[Check]:
    checks = [
        _command_check("Wine", ("wine", "--version")),
        _command_check(
            "NVIDIA GPU",
            ("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
        ),
        _command_check("Vulkan", ("vulkaninfo", "--summary")),
        _command_check("Xorg", ("Xorg", "-version")),
        _command_check("VNC", ("x11vnc", "-version")),
    ]
    state_dir = Path(os.environ.get("DSLE_STATE_DIR", "/var/lib/dsle"))
    state_ok = state_dir.is_dir() and os.access(state_dir, os.W_OK | os.X_OK)
    checks.append(
        Check(
            "state volume",
            "pass" if state_ok else "fail",
            f"writable: {state_dir}" if state_ok else f"not writable: {state_dir}",
        )
    )
    pulse_server = os.environ.get("PULSE_SERVER", "")
    pulse_path = (
        Path(pulse_server.removeprefix("unix:")) if pulse_server.startswith("unix:") else None
    )
    pulse_error = "PULSE_SERVER must name a Unix socket"
    if pulse_path is not None:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(1.0)
        try:
            probe.connect(os.fspath(pulse_path))
        except OSError as exc:
            pulse_error = f"cannot connect to {pulse_path}: {exc}"
        else:
            pulse_error = ""
        finally:
            probe.close()
    checks.append(
        Check(
            "PulseAudio null sink",
            "pass" if not pulse_error else "fail",
            pulse_error if pulse_error else f"listening at {pulse_path}",
        )
    )
    game_dir = Path(os.environ.get("DSLE_GAME_DIR", "/opt/dsle/game"))
    read_only = _mount_is_read_only(game_dir)
    checks.append(
        Check(
            "game mount mode",
            "pass" if read_only is True else ("fail" if read_only is False else "warn"),
            (
                "read-only bind mount"
                if read_only is True
                else (
                    "game directory is mounted writable"
                    if read_only is False
                    else "could not determine the game mount mode"
                )
            ),
        )
    )
    checks.extend(_linux_security_checks())
    return checks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsle-doctor",
        description="Validate a DSLE host or runtime container.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--host", action="store_true", help="check host Docker and NVIDIA support")
    mode.add_argument(
        "--container", action="store_true", help="check the running Wine/DXVK container"
    )
    parser.add_argument(
        "--game-dir",
        type=Path,
        help="Dark Souls: Remastered directory (default: recognized folder in the CWD)",
    )
    parser.add_argument(
        "--require-game", action="store_true", help="treat a missing game as a failure"
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output", help="emit machine-readable JSON"
    )
    parser.add_argument("--quiet", action="store_true", help="print output only when a check fails")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    in_container = bool(
        args.container or (not args.host and os.environ.get("DSLE_CONTAINER") == "1")
    )
    configured_game = args.game_dir or (
        Path(os.environ["DSLE_GAME_DIR"]) if os.environ.get("DSLE_GAME_DIR") else None
    )
    if configured_game is None:
        configured_game = next(
            (
                candidate
                for name in ("Dark.Souls.Remastered.v1.04", "game")
                if (candidate := Path.cwd() / name).is_dir()
            ),
            None,
        )

    checks = [
        Check(
            "operating system",
            "pass" if platform.system() == "Linux" else "fail",
            f"{platform.system()} {platform.release()}",
        ),
        Check(
            "Python",
            "pass" if sys.version_info >= (3, 10) else "fail",
            platform.python_version(),
        ),
    ]
    resource_path = (
        configured_game if configured_game is not None and configured_game.exists() else Path.cwd()
    )
    checks.extend(_resource_checks(resource_path))
    checks.extend(_game_checks(configured_game, required=bool(args.require_game or in_container)))
    checks.extend(_container_checks() if in_container else _host_checks())

    ok = all(check.status != "fail" for check in checks)
    if args.json_output:
        print(
            json.dumps(
                {
                    "ok": ok,
                    "mode": "container" if in_container else "host",
                    "checks": [asdict(check) for check in checks],
                },
                indent=2,
            )
        )
    elif not args.quiet or not ok:
        for check in checks:
            print(f"{check.status.upper():4}  {check.name}: {check.detail}")
        print("DSLE preflight passed." if ok else "DSLE preflight failed.")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
