"""Host-side management for a prebuilt, local-only DSLE runtime container."""

from __future__ import annotations

import atexit
import math
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
import time
import warnings
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import SyncVectorEnv, VectorEnv, VectorWrapper
from platformdirs import user_data_path

from dsle.exceptions import ConfigurationError, RuntimeUnavailableError
from dsle.progress import LifecycleLogger
from dsle.remote import RemoteDarkSoulsEnv, RPCClient, ping
from dsle.runtime.version import (
    GAME_EXECUTABLE,
    REQUIRED_GAME_DIRECTORIES,
    verify_game_executable,
)

_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}")
_CONTAINER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_INSTANCE_NAME = re.compile(r"dsr-([1-9]|[12][0-9]|30)")
_IMAGE_REFERENCE = re.compile(r"[^\s\x00-\x1f\x7f]+")
_CONTAINER_GAME_DIR = Path("/opt/dsle/game")
_CONTAINER_STATE_DIR = Path("/var/lib/dsle")
_CONTAINER_RPC_DIR = _CONTAINER_STATE_DIR / "rpc"
_OUTPUT_MARKER = ".dsle-output"
_OUTPUT_MARKER_CONTENT = b"dsle-output\n"
_OUTPUT_LOCK = ".container-session.lock"
_OUTPUT_DIRECTORIES = {
    "config": 0o2770,
    "dxvk-cache": 0o2770,
    "logs": 0o2770,
    "pulse": 0o700,
    "recordings": 0o2770,
    "results": 0o2770,
    "rpc": 0o700,
    "run": 0o700,
    "wineprefixes": 0o2770,
    "xdg": 0o700,
}
_ACTIVE_SESSIONS: set[ContainerSession] = set()


def _close_active_sessions() -> None:
    """Best-effort cleanup for normally exiting host processes."""

    current_pid = os.getpid()
    for session in tuple(_ACTIVE_SESSIONS):
        if session._owner_pid != current_pid:
            # ``fork()`` copies the registry, but the child must never stop a
            # container owned by the still-running parent process.
            _ACTIVE_SESSIONS.discard(session)
            continue
        with suppress(Exception):
            session.close()


atexit.register(_close_active_sessions)


def validate_game_directory(path: str | Path) -> Path:
    """Resolve an externally supplied DSR installation without copying it."""

    try:
        directory = Path(path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ConfigurationError(f"Game directory does not exist: {path}") from exc
    if not directory.is_dir():
        raise ConfigurationError(f"Game path is not a directory: {directory}")
    executable = directory / GAME_EXECUTABLE
    try:
        resolved_executable = executable.resolve(strict=True)
        resolved_executable.relative_to(directory)
    except (OSError, ValueError) as exc:
        raise ConfigurationError(
            f"{GAME_EXECUTABLE} is missing or escapes the game directory: {executable}"
        ) from exc
    if not resolved_executable.is_file() or resolved_executable.stat().st_size <= 0:
        raise ConfigurationError(f"{GAME_EXECUTABLE} is not a non-empty file: {executable}")
    missing: list[str] = []
    for name in REQUIRED_GAME_DIRECTORIES:
        candidate = directory / name
        try:
            resolved_candidate = candidate.resolve(strict=True)
            resolved_candidate.relative_to(directory)
        except (OSError, ValueError):
            missing.append(name)
            continue
        if not resolved_candidate.is_dir():
            missing.append(name)
    if missing:
        raise ConfigurationError(
            f"Game installation is missing required directories: {', '.join(missing)}"
        )
    verify_game_executable(resolved_executable, allow_unverified=False)
    if "," in os.fspath(directory):
        raise ConfigurationError("Game directory cannot contain ',' when used as a Docker mount")
    return directory


def _open_control_file(path: Path, expected: bytes) -> None:
    """Create or validate one host-owned, non-linked DSLE control file."""

    flags = os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    created = False
    try:
        descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        created = True
    except FileExistsError:
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ConfigurationError(f"Unsafe DSLE control file {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigurationError(f"Could not create DSLE control file {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ConfigurationError(f"DSLE control path must be one regular file: {path}")
        if metadata.st_uid != os.getuid():
            raise ConfigurationError(f"DSLE control file is not owned by this user: {path}")
        if created:
            if expected:
                os.write(descriptor, expected)
                os.fsync(descriptor)
        else:
            os.lseek(descriptor, 0, os.SEEK_SET)
            actual = os.read(descriptor, max(len(expected) + 1, 1))
            if actual != expected:
                raise ConfigurationError(f"DSLE control file has invalid content: {path}")
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


def _prepare_output_layout(directory: Path) -> None:
    marker = directory / _OUTPUT_MARKER
    try:
        marker.lstat()
    except FileNotFoundError:
        try:
            first_entry = next(directory.iterdir())
        except StopIteration:
            pass
        else:
            raise ConfigurationError(
                f"Refusing non-empty uninitialized output directory {directory}; "
                f"choose an empty directory or one containing a valid {_OUTPUT_MARKER} marker "
                f"(first existing entry: {first_entry.name})"
            )
    _open_control_file(marker, _OUTPUT_MARKER_CONTENT)

    for name, mode in _OUTPUT_DIRECTORIES.items():
        candidate = directory / name
        with suppress(FileExistsError):
            candidate.mkdir(mode=mode)
        try:
            metadata = candidate.lstat()
        except OSError as exc:
            raise ConfigurationError(
                f"Could not validate DSLE output path {candidate}: {exc}"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ConfigurationError(f"Reserved DSLE output path must be a directory: {candidate}")
        if metadata.st_uid != os.getuid():
            raise ConfigurationError(f"Reserved DSLE output path is not user-owned: {candidate}")
        candidate.chmod(mode)

    _open_control_file(directory / _OUTPUT_LOCK, b"")


def _validate_output_directory(path: str | Path, game_dir: Path) -> Path:
    directory = Path(path).expanduser().resolve(strict=False)
    if directory == Path(directory.anchor):
        raise ConfigurationError("Output directory cannot be a filesystem root")
    try:
        directory.relative_to(game_dir)
    except ValueError:
        pass
    else:
        raise ConfigurationError(
            "Writable output directory cannot be inside the read-only game mount"
        )
    try:
        game_dir.relative_to(directory)
    except ValueError:
        pass
    else:
        raise ConfigurationError("Output directory cannot contain the game installation")
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise ConfigurationError(f"Output path is not a directory: {directory}")
    if "," in os.fspath(directory):
        raise ConfigurationError("Output directory cannot contain ',' when used as a Docker mount")
    _prepare_output_layout(directory)
    return directory


class ContainerSession:
    """Own exactly one ephemeral container started from an existing local image.

    ``start`` first resolves the image with ``docker image inspect`` and then
    runs that immutable image id with ``--pull=never``.  It never calls build,
    pull, or copies game content into an image.
    """

    # Some lifecycle safety tests construct a deliberately partial object with
    # ``object.__new__``. Keep those emergency paths quiet and functional.
    _lifecycle = LifecycleLogger(False)

    def __init__(
        self,
        *,
        image: str,
        game_dir: str | Path,
        output_dir: str | Path,
        name: str | None = None,
        docker_binary: str | Path | None = None,
        startup_timeout: float = 120.0,
        stop_timeout: float = 45.0,
        request_timeout: float = 180.0,
        vnc_ports: Sequence[int] = (),
        vnc_host_ports: Mapping[int, int] | None = None,
        verbose: bool = True,
    ):
        self._lifecycle = LifecycleLogger(verbose)
        if not isinstance(image, str) or not image or not _IMAGE_REFERENCE.fullmatch(image):
            raise ValueError("image must be a non-empty Docker image reference without whitespace")
        if image.startswith("-"):
            raise ValueError("image references cannot begin with '-'")
        selected_timeouts: list[float] = []
        for value in (startup_timeout, stop_timeout, request_timeout):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("Container and request timeouts must be finite positive numbers")
            selected = float(value)
            if not math.isfinite(selected) or selected <= 0:
                raise ValueError("Container and request timeouts must be finite positive numbers")
            selected_timeouts.append(selected)
        if isinstance(vnc_ports, (str, bytes)):
            raise ValueError("vnc_ports must be a sequence of unique ports in [5901, 5930]")
        try:
            selected_vnc_ports = tuple(vnc_ports)
        except TypeError as exc:
            raise ValueError(
                "vnc_ports must be a sequence of unique ports in [5901, 5930]"
            ) from exc
        if any(
            isinstance(port, bool) or not isinstance(port, int) or not 5901 <= port <= 5930
            for port in selected_vnc_ports
        ) or len(set(selected_vnc_ports)) != len(selected_vnc_ports):
            raise ValueError("vnc_ports must be a sequence of unique ports in [5901, 5930]")
        if vnc_host_ports is None:
            selected_vnc_host_ports: dict[int, int] = {}
        elif not isinstance(vnc_host_ports, Mapping):
            raise ValueError("vnc_host_ports must map published VNC ports to unique host ports")
        else:
            selected_vnc_host_ports = dict(vnc_host_ports)
        if any(
            isinstance(container_port, bool)
            or not isinstance(container_port, int)
            or container_port not in selected_vnc_ports
            or isinstance(host_port, bool)
            or not isinstance(host_port, int)
            or not 1 <= host_port <= 65535
            for container_port, host_port in selected_vnc_host_ports.items()
        ) or len(set(selected_vnc_host_ports.values())) != len(selected_vnc_host_ports):
            raise ValueError(
                "vnc_host_ports must map selected VNC ports to unique host ports in [1, 65535]"
            )

        self.image = image
        self.startup_timeout, self.stop_timeout, self.request_timeout = selected_timeouts
        self.vnc_container_ports = selected_vnc_ports
        self.vnc_host_ports = selected_vnc_host_ports
        self.vnc_ports: dict[int, int] = {}
        self._owner_pid = os.getpid()
        self.session_id = secrets.token_hex(16)
        selected_name = f"dsle-{os.getpid()}-{self.session_id[:12]}" if name is None else name
        if not isinstance(selected_name, str) or not _CONTAINER_NAME.fullmatch(selected_name):
            raise ValueError("Container name must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
        self.name = selected_name

        self._lifecycle.check("Checking host dependency: Docker CLI")
        if docker_binary is None:
            discovered = shutil.which("docker")
            if discovered is None:
                raise RuntimeUnavailableError("Docker CLI is not installed or not on PATH")
            self.docker_binary = discovered
        else:
            try:
                self.docker_binary = os.fsdecode(os.fspath(docker_binary))
            except TypeError as exc:
                raise ValueError("docker_binary must be a non-empty path") from exc
            if not self.docker_binary or "\x00" in self.docker_binary:
                raise ValueError("docker_binary must be a non-empty path")
        self._lifecycle.ready(f"Docker CLI found: {self.docker_binary}")

        self._lifecycle.check("Validating the attached Dark Souls: Remastered installation")
        self.game_dir = validate_game_directory(game_dir)
        self._lifecycle.ready(f"Game installation verified: {self.game_dir}")
        self._lifecycle.check("Preparing the persistent output directory")
        self.output_dir = _validate_output_directory(output_dir, self.game_dir)
        self._lifecycle.ready(f"Persistent output ready: {self.output_dir}")

        transport_id = self.session_id[:16]
        try:
            self._transport_dir = Path(
                tempfile.mkdtemp(
                    prefix=f"dsle-rpc-{os.getpid()}-",
                    dir=tempfile.gettempdir(),
                )
            ).resolve(strict=True)
            self._transport_dir.chmod(0o700)
        except OSError as exc:
            raise RuntimeUnavailableError(
                f"Could not create a private RPC directory: {exc}"
            ) from exc
        self.socket_path = self._transport_dir / f"rpc-{transport_id}.sock"
        self.token_path = self._transport_dir / f"rpc-{self.session_id}.token"
        self._cidfile_path = self._transport_dir / "container.cid"
        self._container_socket = _CONTAINER_RPC_DIR / self.socket_path.name
        self._container_token = _CONTAINER_RPC_DIR / self.token_path.name
        if len(os.fsencode(self.socket_path)) >= 104:
            self._transport_dir.rmdir()
            raise ConfigurationError(f"Temporary Unix socket path is too long: {self.socket_path}")
        self._token: str | None = None
        self._image_id: str | None = None
        self._container_id: str | None = None
        self._container_may_exist = False
        self._environments: list[RemoteDarkSoulsEnv] = []
        self._started_instances = False
        self._closed = False
        _ACTIVE_SESSIONS.add(self)
        self._lifecycle.debug(f"Prepared managed session {self.session_id} as {self.name}")

    @property
    def container_id(self) -> str | None:
        return self._container_id

    @property
    def running(self) -> bool:
        return self._container_id is not None and self._owned_container_exists()

    def _require_owner_process(self) -> None:
        if os.getpid() != self._owner_pid:
            raise RuntimeError(
                "ContainerSession lifecycle cannot be used from a forked child process"
            )

    def _run(
        self,
        arguments: list[str],
        *,
        timeout: float,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                [self.docker_binary, *arguments],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeUnavailableError(
                f"Docker command failed to execute: {arguments[0] if arguments else 'docker'}: {exc}"
            ) from exc
        if check and result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no error output"
            raise RuntimeUnavailableError(
                f"Docker {' '.join(arguments[:2])} failed with exit {result.returncode}: {detail}"
            )
        return result

    def _inspect_local_image(self) -> str:
        result = self._run(
            ["image", "inspect", "--format", "{{.Id}}", self.image],
            timeout=20.0,
        )
        image_id = result.stdout.strip()
        if not _IMAGE_ID.fullmatch(image_id):
            raise RuntimeUnavailableError(
                f"Docker returned an invalid image id for local image {self.image!r}: {image_id!r}"
            )
        return image_id

    def _write_token(self) -> str:
        token = secrets.token_hex(32)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.token_path, flags, 0o600)
        except OSError as exc:
            raise RuntimeUnavailableError(
                f"Could not create private RPC token {self.token_path}: {exc}"
            ) from exc
        try:
            os.write(descriptor, token.encode("ascii") + b"\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(self.token_path, 0o600)
        return token

    def _run_arguments(self, image_id: str) -> list[str]:
        game_mount = f"type=bind,source={self.game_dir},target={_CONTAINER_GAME_DIR},readonly"
        state_mount = f"type=bind,source={self.output_dir},target={_CONTAINER_STATE_DIR}"
        rpc_mount = f"type=bind,source={self._transport_dir},target={_CONTAINER_RPC_DIR}"
        environment = {
            "DSLE_ASSET_DIR": "/opt/dsle/assets",
            "DSLE_CONTAINER": "1",
            "DSLE_GAME_DIR": str(_CONTAINER_GAME_DIR),
            "DSLE_INSTANCE_CONFIG": "/var/lib/dsle/config/instances.json",
            "DSLE_OUTPUT_DIR": "/var/lib/dsle/results",
            "DSLE_RUNTIME_SOCKET": str(self._container_socket),
            "DSLE_RUNTIME_TOKEN_FILE": str(self._container_token),
            "DSLE_STATE_DIR": str(_CONTAINER_STATE_DIR),
            "DSLE_HOST_GID": str(os.getgid()),
            "DSLE_HOST_UID": str(os.getuid()),
            "DXVK_STATE_CACHE_PATH": "/var/lib/dsle/dxvk-cache",
            "NVIDIA_DRIVER_CAPABILITIES": "graphics,utility,compute,display",
            "NVIDIA_VISIBLE_DEVICES": "all",
            "PULSE_SERVER": "unix:/var/lib/dsle/pulse/native",
            "WINEPREFIX_TEMPLATE": "/opt/dsle/wine-template",
            "XDG_RUNTIME_DIR": "/var/lib/dsle/xdg/container",
        }
        arguments = [
            "run",
            "--detach",
            "--rm",
            "--pull=never",
            "--init",
            "--name",
            self.name,
            "--cidfile",
            os.fspath(self._cidfile_path),
            "--label",
            "io.dsle.managed=true",
            "--label",
            f"io.dsle.session={self.session_id}",
            "--gpus",
            "all",
            "--runtime",
            "nvidia",
            "--cap-add",
            "SYS_PTRACE",
            "--security-opt",
            "seccomp=unconfined",
            "--security-opt",
            "apparmor=unconfined",
            "--shm-size",
            "2g",
            "--stop-timeout",
            str(max(1, int(self.stop_timeout))),
            "--mount",
            game_mount,
            "--mount",
            state_mount,
            "--mount",
            rpc_mount,
        ]
        for key, value in environment.items():
            arguments.extend(("--env", f"{key}={value}"))
        for port in self.vnc_container_ports:
            host_port = self.vnc_host_ports.get(port)
            publication = (
                f"127.0.0.1::{port}/tcp"
                if host_port is None
                else f"127.0.0.1:{host_port}:{port}/tcp"
            )
            arguments.extend(("--publish", publication))
        arguments.extend(
            (
                image_id,
                "python3",
                "-m",
                "dsle.runtime.server",
                "--socket",
                str(self._container_socket),
                "--token-file",
                str(self._container_token),
                "--state-dir",
                str(_CONTAINER_STATE_DIR),
            )
        )
        return arguments

    def _inspect_identity(self) -> tuple[str, str, str] | None:
        if self._container_id is None:
            return None
        result = self._run(
            [
                "container",
                "inspect",
                "--format",
                '{{.Id}}|{{.Name}}|{{index .Config.Labels "io.dsle.session"}}',
                self._container_id,
            ],
            timeout=10.0,
            check=False,
        )
        if result.returncode != 0:
            if not self._container_exists_by_id(self._container_id):
                return None
            detail = result.stderr.strip() or result.stdout.strip() or "no error output"
            raise RuntimeUnavailableError(
                f"Docker could not inspect owned container {self._container_id}: {detail}"
            )
        parts = result.stdout.strip().split("|")
        if len(parts) != 3:
            raise RuntimeUnavailableError("Docker returned malformed container identity data")
        return parts[0], parts[1].removeprefix("/"), parts[2]

    def _container_exists_by_id(self, container_id: str) -> bool:
        result = self._run(
            [
                "container",
                "ls",
                "--all",
                "--quiet",
                "--no-trunc",
                "--filter",
                f"id={container_id}",
            ],
            timeout=10.0,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no error output"
            raise RuntimeUnavailableError(
                f"Docker could not verify whether container {container_id} exists: {detail}"
            )
        identifiers = {line.strip() for line in result.stdout.splitlines() if line.strip()}
        if any(not _CONTAINER_ID.fullmatch(identifier) for identifier in identifiers):
            raise RuntimeUnavailableError("Docker returned malformed container listing data")
        return container_id in identifiers

    def _recover_container_id(self) -> None:
        """Recover an owned ID after a timed-out detached ``docker run``."""

        if self._container_id is not None:
            return
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                candidate = self._cidfile_path.read_text(encoding="ascii").strip()
            except (FileNotFoundError, OSError, UnicodeError):
                candidate = ""
            if _CONTAINER_ID.fullmatch(candidate):
                self._container_id = candidate
                self._container_may_exist = False
                return
            time.sleep(0.05)

        result = self._run(
            [
                "container",
                "inspect",
                "--format",
                '{{.Id}}|{{.Name}}|{{index .Config.Labels "io.dsle.session"}}',
                self.name,
            ],
            timeout=10.0,
            check=False,
        )
        if result.returncode != 0:
            listing = self._run(
                [
                    "container",
                    "ls",
                    "--all",
                    "--quiet",
                    "--no-trunc",
                    "--filter",
                    f"name=^/{self.name}$",
                ],
                timeout=10.0,
                check=False,
            )
            if listing.returncode != 0:
                detail = listing.stderr.strip() or result.stderr.strip() or "no error output"
                raise RuntimeUnavailableError(
                    f"Docker could not recover pending container {self.name}: {detail}"
                )
            identifiers = [line.strip() for line in listing.stdout.splitlines() if line.strip()]
            if identifiers:
                raise RuntimeUnavailableError(
                    f"Docker listed pending container {self.name} but its identity is unavailable"
                )
            self._container_may_exist = False
            return
        parts = result.stdout.strip().split("|")
        if (
            len(parts) != 3
            or _CONTAINER_ID.fullmatch(parts[0]) is None
            or not hmac_compare(parts[1].removeprefix("/"), self.name)
        ):
            # A successful but malformed/partial inspect cannot prove whether
            # the timed-out `docker run` created a container. Retain the
            # pending flag so callers never silently abandon possible state.
            raise RuntimeUnavailableError(
                f"Docker returned malformed pending-container identity for {self.name}"
            )
        if hmac_compare(parts[2], self.session_id):
            self._container_id = parts[0]
            self._container_may_exist = False
            return
        # A differently labelled container occupying the exact name proves
        # that this session's pending `docker run` did not create one.
        self._container_may_exist = False

    def _owned_container_exists(self) -> bool:
        identity = self._inspect_identity()
        if identity is None or self._container_id is None:
            return False
        container_id, name, session_label = identity
        return (
            hmac_compare(container_id, self._container_id)
            and hmac_compare(name, self.name)
            and hmac_compare(session_label, self.session_id)
        )

    def _require_owned_container(self) -> None:
        identity = self._inspect_identity()
        expected = (self._container_id, self.name, self.session_id)
        if identity != expected:
            raise RuntimeUnavailableError(
                f"Refusing to manage container with mismatched identity: expected {expected}, got {identity}"
            )

    def _wait_until_ready(self) -> None:
        token = self._token
        if token is None:
            raise RuntimeError("ContainerSession has no RPC token; call start() first")
        deadline = time.monotonic() + self.startup_timeout
        self._lifecycle.check("Waiting for the container RPC server")
        last_error = "runtime socket has not appeared"
        next_container_probe = 0.0
        while time.monotonic() < deadline:
            if self.socket_path.exists():
                try:
                    ping(self.socket_path, token=token, timeout=min(2.0, self.startup_timeout))
                except Exception as exc:
                    last_error = str(exc)
                else:
                    self._verify_runtime_health()
                    self._lifecycle.ready("Container RPC server is healthy")
                    return
            now = time.monotonic()
            if now >= next_container_probe:
                if not self._owned_container_exists():
                    raise RuntimeUnavailableError(
                        f"DSLE container exited before its RPC server was ready: {last_error}"
                    )
                next_container_probe = now + 1.0
            time.sleep(0.05)
        raise RuntimeUnavailableError(
            f"DSLE RPC server did not become ready within {self.startup_timeout:g}s: {last_error}"
        )

    def _verify_runtime_health(self) -> None:
        """Require the image's full NVIDIA/Wine/game doctor after RPC readiness."""

        if self._container_id is None:
            raise RuntimeError("ContainerSession has not been started")
        self._lifecycle.check("Checking NVIDIA, Wine, dependencies, and read-only game mount")
        self._require_owned_container()
        self._run(
            [
                "container",
                "exec",
                self._container_id,
                "dsle-doctor",
                "--container",
                "--require-game",
                "--quiet",
            ],
            timeout=min(max(self.startup_timeout, 30.0), 120.0),
        )
        self._lifecycle.ready("Container dependency and GPU checks passed")

    def _resolve_vnc_ports(self) -> None:
        if self._container_id is None:
            raise RuntimeError("ContainerSession has not been started")
        resolved: dict[int, int] = {}
        for container_port in self.vnc_container_ports:
            result = self._run(
                ["container", "port", self._container_id, f"{container_port}/tcp"],
                timeout=10.0,
            )
            endpoints = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if len(endpoints) != 1 or not endpoints[0].startswith("127.0.0.1:"):
                raise RuntimeUnavailableError(
                    f"Docker returned malformed VNC mapping for port {container_port}: {endpoints}"
                )
            host_port_text = endpoints[0].rsplit(":", 1)[-1]
            if not host_port_text.isdigit() or not 1 <= int(host_port_text) <= 65535:
                raise RuntimeUnavailableError(
                    f"Docker returned invalid VNC host port: {endpoints[0]!r}"
                )
            resolved[container_port] = int(host_port_text)
        self.vnc_ports = resolved
        if resolved:
            mappings = ", ".join(
                f"127.0.0.1:{host}->:{container}" for container, host in sorted(resolved.items())
            )
            self._lifecycle.info(f"VNC ports published: {mappings}")

    def start(self) -> ContainerSession:
        self._require_owner_process()
        if self._closed:
            raise RuntimeError("Cannot restart a closed ContainerSession")
        if self._container_id is not None:
            self._require_owned_container()
            self._lifecycle.ready(f"Container is already running: {self.name}")
            return self
        try:
            self._lifecycle.check(f"Resolving local prebuilt image: {self.image}")
            self._image_id = self._inspect_local_image()
            self._lifecycle.ready(f"Using immutable image: {self._image_id[:19]}")
            self._token = self._write_token()
            self._lifecycle.debug(
                f"Game mount={self.game_dir} (read-only); output={self.output_dir} (persistent)"
            )
            try:
                self._container_may_exist = True
                self._lifecycle.start(f"Starting managed container: {self.name}")
                result = self._run(
                    self._run_arguments(self._image_id),
                    # A detached run is normally immediate, but the daemon can
                    # serialize it behind a large image export.  Reuse the
                    # caller's startup budget so killing the CLI at 30 seconds
                    # cannot leave a late-created container outside our ID
                    # recovery window.
                    timeout=max(30.0, self.startup_timeout),
                )
            except Exception:
                self._recover_container_id()
                raise
            container_id = result.stdout.strip()
            if not _CONTAINER_ID.fullmatch(container_id):
                self._recover_container_id()
                raise RuntimeUnavailableError(
                    f"Docker run returned an invalid container id: {container_id!r}"
                )
            self._container_id = container_id
            self._container_may_exist = False
            try:
                cidfile_id = self._cidfile_path.read_text(encoding="ascii").strip()
            except FileNotFoundError:
                cidfile_id = container_id
            if cidfile_id != container_id:
                raise RuntimeUnavailableError(
                    "Docker stdout and --cidfile reported different container identities"
                )
            self._require_owned_container()
            self._lifecycle.ready(f"Container running: {self.name} ({self._container_id[:12]})")
            self._resolve_vnc_ports()
            self._wait_until_ready()
            self._lifecycle.done(f"Managed container ready: {self.name}")
            return self
        except Exception as exc:
            self._lifecycle.error(f"Container startup failed: {exc}")
            try:
                self._stop_owned_container()
            finally:
                if self._container_id is None and not self._container_may_exist:
                    self._cleanup_transport_files()
                    self._closed = True
                    _ACTIVE_SESSIONS.discard(self)
            raise

    def ping(self, timeout: float = 5.0) -> dict[str, Any]:
        if self._container_id is None or self._token is None:
            raise RuntimeError("ContainerSession has not been started")
        self._require_owned_container()
        return ping(self.socket_path, token=self._token, timeout=timeout)

    def start_instance(self, name: str, *, mode: str = "headless") -> dict[str, Any]:
        self._require_owner_process()
        if self._container_id is None or self._token is None:
            raise RuntimeError("ContainerSession has not been started")
        self._require_owned_container()
        client = RPCClient(
            self.socket_path,
            self._token,
            request_timeout=max(self.request_timeout, 120.0),
        )
        self._lifecycle.start(f"Starting game instance {name} in {mode} mode")
        try:
            result, array = client.request("start_instance", {"name": name, "mode": mode})
            if array is not None:
                raise RuntimeUnavailableError("start_instance unexpectedly returned an ndarray")
            status = result.get("status")
            if not isinstance(status, dict):
                raise RuntimeUnavailableError("start_instance returned malformed status")
            self._started_instances = True
            self._lifecycle.ready(f"Game running: {name}")
            self._lifecycle.debug(f"{name} status: {status}")
            return status
        finally:
            client.close()

    def start_instances(
        self,
        count: int,
        *,
        mode: str = "headless",
    ) -> dict[str, dict[str, Any]]:
        """Start ``dsr-1`` through ``dsr-N`` in this shared container."""

        self._require_owner_process()
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 30:
            raise ValueError("count must be an integer in [1, 30]")
        if self._container_id is None or self._token is None:
            raise RuntimeError("ContainerSession has not been started")
        self._require_owned_container()
        client = RPCClient(
            self.socket_path,
            self._token,
            request_timeout=max(self.request_timeout, 120.0),
        )
        names = ", ".join(f"dsr-{index}" for index in range(1, count + 1))
        self._lifecycle.start(f"Starting {count} game instance(s): {names}")
        try:
            result, array = client.request(
                "start_instances",
                {"count": count, "mode": mode},
            )
            if array is not None:
                raise RuntimeUnavailableError("start_instances unexpectedly returned an ndarray")
            statuses = result.get("statuses")
            if not isinstance(statuses, dict):
                raise RuntimeUnavailableError("start_instances returned malformed statuses")
            expected = {f"dsr-{index}" for index in range(1, count + 1)}
            if set(statuses) != expected or any(
                not isinstance(status, dict)
                or status.get("status") != "ready"
                or status.get("running") is not True
                for status in statuses.values()
            ):
                raise RuntimeUnavailableError(
                    f"start_instances did not return a ready pool: {statuses}"
                )
            self._started_instances = True
            for name in sorted(statuses):
                self._lifecycle.ready(f"Game running: {name}")
            self._lifecycle.done(f"All {count} game instance(s) are ready")
            return statuses
        finally:
            client.close()

    def make(
        self,
        boss: str,
        *,
        start_instance: bool = True,
        instance_mode: str = "headless",
        **env_kwargs: Any,
    ) -> RemoteDarkSoulsEnv:
        self._require_owner_process()
        if self._container_id is None or self._token is None:
            raise RuntimeError("ContainerSession has not been started")
        self._require_owned_container()
        instance = str(env_kwargs.get("instance", "dsr-1"))
        self._lifecycle.start(f"Creating environment proxy: boss={boss}, instance={instance}")
        if start_instance:
            self._lifecycle.start(f"Starting game instance {instance} in {instance_mode} mode")
        environment = RemoteDarkSoulsEnv(
            self.socket_path,
            boss,
            token=self._token,
            request_timeout=self.request_timeout,
            start_instance=start_instance,
            instance_mode=instance_mode,
            **env_kwargs,
        )
        self._environments.append(environment)
        self._started_instances = self._started_instances or start_instance
        if start_instance:
            self._lifecycle.ready(f"Game running: {instance}")
        self._lifecycle.ready(f"Environment proxy ready: boss={boss}, instance={instance}")
        return environment

    make_env = make

    def _close_remote_state(self) -> bool:
        """Close server state and report whether the container still exists.

        A managed container may have exited independently after a failed game
        operation.  Proven absence is already the desired lifecycle outcome;
        it must not turn a cleanup path into a second, masking exception.
        """

        if self._container_id is None or self._token is None:
            return False
        identity = self._inspect_identity()
        if identity is None:
            return False
        expected = (self._container_id, self.name, self.session_id)
        if identity != expected:
            raise RuntimeUnavailableError(
                f"Refusing to manage container with mismatched identity: "
                f"expected {expected}, got {identity}"
            )
        if not self._environments and not self._started_instances:
            return True
        client = RPCClient(
            self.socket_path,
            self._token,
            connect_timeout=2.0,
            request_timeout=min(60.0, self.request_timeout),
        )
        try:
            result, array = client.request("close_all")
            if array is not None or result.get("closed") is not True:
                raise RuntimeUnavailableError("close_all returned a malformed response")
        finally:
            client.close()
        return True

    def _finalize_output_ownership(self) -> None:
        if self._container_id is None:
            return
        self._require_owned_container()
        self._run(
            [
                "container",
                "exec",
                self._container_id,
                "python3",
                "-m",
                "dsle.runtime.output",
                "--state-dir",
                str(_CONTAINER_STATE_DIR),
                "--uid",
                str(os.getuid()),
                "--gid",
                str(os.getgid()),
            ],
            timeout=30.0,
        )

    def _stop_owned_container(self) -> None:
        if self._container_id is None and self._container_may_exist:
            self._recover_container_id()
        if self._container_id is None:
            return
        container_id = self._container_id
        identity = self._inspect_identity()
        if identity is None:
            self._container_id = None
            self._container_may_exist = False
            return
        expected = (container_id, self.name, self.session_id)
        if identity != expected:
            raise RuntimeUnavailableError(
                f"Refusing to stop container with mismatched identity: expected {expected}, got {identity}"
            )
        stop_arguments = [
            "container",
            "stop",
            "--time",
            str(max(1, int(self.stop_timeout))),
            container_id,
        ]
        try:
            stop_result = self._run(
                stop_arguments,
                timeout=self.stop_timeout + 10.0,
                check=False,
            )
        except RuntimeUnavailableError:
            if not self._container_exists_by_id(container_id):
                self._container_id = None
                self._container_may_exist = False
                return
            raise
        if stop_result.returncode != 0:
            if not self._container_exists_by_id(container_id):
                self._container_id = None
                self._container_may_exist = False
                return
            detail = stop_result.stderr.strip() or stop_result.stdout.strip() or "no error output"
            raise RuntimeUnavailableError(
                f"Docker container stop failed with exit {stop_result.returncode}: {detail}"
            )
        removal_deadline = time.monotonic() + min(10.0, self.stop_timeout)
        while time.monotonic() < removal_deadline:
            if not self._container_exists_by_id(container_id):
                self._container_id = None
                self._container_may_exist = False
                return
            time.sleep(0.05)
        self._require_owned_container()
        self._run(["container", "rm", container_id], timeout=15.0)
        if self._container_exists_by_id(container_id):
            raise RuntimeUnavailableError(
                f"Owned container {container_id} stopped but Docker did not remove it"
            )
        self._container_id = None
        self._container_may_exist = False

    def _cleanup_transport_files(self) -> None:
        for path, expected_kind in (
            (self.socket_path, "socket"),
            (self.token_path, "file"),
            (self._cidfile_path, "file"),
        ):
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                continue
            matches = (
                stat.S_ISSOCK(metadata.st_mode)
                if expected_kind == "socket"
                else stat.S_ISREG(metadata.st_mode)
            )
            if stat.S_ISLNK(metadata.st_mode) or matches:
                path.unlink()
        try:
            self._transport_dir.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            # Never recursively remove unexpected files from the private
            # directory. Retaining it is safer and preserves diagnostics.
            pass

    def close(self) -> None:
        if os.getpid() != self._owner_pid:
            _ACTIVE_SESSIONS.discard(self)
            return
        if self._closed:
            _ACTIVE_SESSIONS.discard(self)
            return
        self._lifecycle.stop(f"Closing managed session: {self.name}")
        first_error: Exception | None = None
        container_available = False
        try:
            if self._started_instances:
                self._lifecycle.stop("Stopping game instances")
            container_available = self._close_remote_state()
        except Exception as exc:
            first_error = first_error or exc
        for environment in reversed(self._environments):
            environment._disconnect()
        self._environments.clear()
        self._started_instances = False
        if container_available:
            try:
                self._lifecycle.check("Finalizing persistent output ownership")
                self._finalize_output_ownership()
            except Exception as exc:
                first_error = first_error or exc
        try:
            self._lifecycle.stop("Stopping and removing the owned container")
            self._stop_owned_container()
        except Exception as exc:
            first_error = first_error or exc
        finally:
            if self._container_id is None and not self._container_may_exist:
                self._cleanup_transport_files()
            self._closed = self._container_id is None and not self._container_may_exist
            if self._closed:
                _ACTIVE_SESSIONS.discard(self)
        if first_error is not None:
            self._lifecycle.error(f"Managed session cleanup failed: {first_error}")
            raise first_error
        self._lifecycle.done(f"Container removed; output preserved at {self.output_dir}")

    def __enter__(self) -> ContainerSession:
        return self.start()

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


def hmac_compare(left: str, right: str) -> bool:
    """Use constant-time equality for opaque Docker identity labels."""

    return secrets.compare_digest(left, right)


class ManagedDarkSoulsEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """A one-environment convenience wrapper that owns its container session."""

    def __init__(self, environment: RemoteDarkSoulsEnv, session: ContainerSession):
        super().__init__(environment)
        self._remote_environment = environment
        self._container_session = session
        self.output_dir = session.output_dir
        self.vnc_ports = dict(getattr(session, "vnc_ports", {}))
        self._managed_closed = False

    def close(self) -> None:
        if self._managed_closed:
            return
        self._container_session.close()
        self._managed_closed = True

    def set_player_hp(self, hp: int) -> dict[str, Any]:
        """Set current player HP through the managed runtime proxy."""

        return self._remote_environment.set_player_hp(hp)

    def set_boss_hp(self, boss_number: int, hp: int) -> dict[str, Any]:
        """Set one ``bossN`` HP, or use ``(-1, 0)``, through the runtime proxy."""

        return self._remote_environment.set_boss_hp(boss_number, hp)

    def observe(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Passively refresh state without injecting a policy action."""

        return self._remote_environment.observe()

    def return_to_menu(self, *, timeout_s: float = 30.0) -> None:
        """Return the owned game instance to a verified title menu."""

        self._remote_environment.return_to_menu(timeout_s=timeout_s)

    def __enter__(self) -> ManagedDarkSoulsEnv:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


class ManagedVectorEnv(VectorWrapper):
    """An N-instance Gymnasium vector environment owning one container."""

    def __init__(self, environment: VectorEnv, session: ContainerSession):
        super().__init__(environment)
        self._container_session = session
        self.output_dir = session.output_dir
        self.vnc_ports = dict(getattr(session, "vnc_ports", {}))
        self._managed_closed = False

    def close(self, **kwargs: Any) -> None:
        if self._managed_closed:
            return
        first_error: Exception | None = None
        try:
            self._container_session.close()
        except Exception as exc:
            first_error = first_error or exc
        try:
            self.env.close(**kwargs)
        except Exception as exc:
            first_error = first_error or exc
        if self._container_session.container_id is None:
            self._managed_closed = True
        if first_error is not None:
            raise first_error

    def __enter__(self) -> ManagedVectorEnv:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


def default_output_directory(label: str = "run") -> Path:
    """Return a unique persistent host directory for one managed run."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_label = re.sub(r"[^a-z0-9]+", "-", str(label).strip().lower()).strip("-") or "run"
    return (
        Path(user_data_path("dsle"))
        / "runs"
        / (f"{timestamp}-{safe_label}-{os.getpid()}-{secrets.token_hex(4)}")
    )


def _select_game_directory(game_dir: str | Path | None) -> str | Path | None:
    """Use an explicit setting, then a recognized game folder in the CWD."""

    selected = game_dir or os.environ.get("DSLE_GAME_DIR")
    if selected:
        return selected
    for name in ("Dark.Souls.Remastered.v1.04", "game"):
        local_game = Path.cwd() / name
        if local_game.is_dir():
            return local_game
    return None


def make_managed(
    boss: str,
    *,
    game_dir: str | Path | None = None,
    image: str | None = None,
    output_dir: str | Path | None = None,
    container_name: str | None = None,
    num_instances: int = 1,
    start_instance: bool = True,
    instance_mode: str = "headless",
    verbose: bool = True,
    **env_kwargs: Any,
) -> ManagedDarkSoulsEnv | ManagedVectorEnv:
    """Start one owned prebuilt-container session and return its Gym proxy."""

    if not isinstance(verbose, bool):
        raise ValueError("verbose must be a boolean")
    selected_game = _select_game_directory(game_dir)
    if selected_game is None:
        raise ConfigurationError("game_dir is required for managed container mode")
    if (
        isinstance(num_instances, bool)
        or not isinstance(num_instances, int)
        or not 1 <= num_instances <= 30
    ):
        raise ValueError("num_instances must be an integer in [1, 30]")
    if instance_mode == "gui":
        raise ValueError(
            "Managed containers cannot expose a host GUI; use headless, headless-vnc, "
            "or the manual container workflow"
        )
    if instance_mode not in {"headless", "headless-vnc"}:
        raise ValueError("instance_mode must be headless or headless-vnc in managed mode")
    if not start_instance:
        raise ValueError(
            "Managed mode creates a fresh container and must start its game instance; "
            "use runtime='external' to attach without container ownership"
        )
    requested_instance = env_kwargs.get("instance", "dsr-1")
    if num_instances > 1:
        if requested_instance != "dsr-1":
            raise ValueError(
                "num_instances > 1 uses dsr-1 through dsr-N; remove the instance option"
            )
        instance_numbers = tuple(range(1, num_instances + 1))
        env_kwargs.pop("instance", None)
    else:
        if (
            not isinstance(requested_instance, str)
            or (match := _INSTANCE_NAME.fullmatch(requested_instance)) is None
        ):
            raise ValueError("Managed mode instance must be dsr-1 through dsr-30")
        instance_numbers = (int(match.group(1)),)
    selected_output = output_dir or default_output_directory(boss)
    selected_image = image or os.environ.get("DSLE_IMAGE", "dsle-runtime:0.1.0")
    session = ContainerSession(
        image=selected_image,
        game_dir=selected_game,
        output_dir=selected_output,
        name=container_name,
        vnc_ports=(
            tuple(5900 + number for number in instance_numbers)
            if instance_mode == "headless-vnc"
            else ()
        ),
        verbose=verbose,
    )
    try:
        session.start()
        if num_instances > 1:
            session.start_instances(num_instances, mode=instance_mode)

            def factory(instance_name: str):
                return lambda: session.make(
                    boss,
                    instance=instance_name,
                    start_instance=False,
                    verbose=verbose,
                    **env_kwargs,
                )

            vector = SyncVectorEnv(
                [factory(f"dsr-{index}") for index in range(1, num_instances + 1)]
            )
            return ManagedVectorEnv(vector, session)
        environment = session.make(
            boss,
            start_instance=True,
            instance_mode=instance_mode,
            verbose=verbose,
            **env_kwargs,
        )
        return ManagedDarkSoulsEnv(environment, session)
    except Exception:
        try:
            session.close()
        except Exception as cleanup_error:
            warnings.warn(
                f"Managed environment creation also failed to clean up: {cleanup_error}",
                RuntimeWarning,
                stacklevel=2,
            )
        # Preserve the construction/start failure as the public exception.
        raise


__all__ = [
    "ContainerSession",
    "ManagedDarkSoulsEnv",
    "ManagedVectorEnv",
    "default_output_directory",
    "make_managed",
    "validate_game_directory",
]
