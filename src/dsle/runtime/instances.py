"""Load and generate configuration for isolated live-game instances."""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from platformdirs import user_config_path, user_data_path

from dsle.exceptions import AssetError, ConfigurationError, RuntimeUnavailableError
from dsle.models import InstanceConfig
from dsle.runtime.version import verify_game_executable

INSTANCE_CONFIG_ENV = "DSLE_INSTANCE_CONFIG"
INSTANCE_SCHEMA_VERSION = 1
INSTANCE_READINESS_TIMEOUT_S = 90.0
_INSTANCE_NAME = re.compile(r"dsr-([1-9]|[12][0-9]|30)")
_INSTANCE_LAUNCH_LOCK = ".instances-launch.lock"


def validate_instance_name(value: object, *, field: str = "instance name") -> str:
    """Return one canonical managed-instance name or reject unsafe path-like input."""

    if not isinstance(value, str) or _INSTANCE_NAME.fullmatch(value) is None:
        raise ConfigurationError(f"{field} must be dsr-1 through dsr-30")
    return value


def default_instance_config_path() -> Path:
    """Return the host-appropriate default instance configuration path."""

    return Path(user_config_path("dsle")) / "instances.json"


def resolve_instance_config_path(path: str | Path | None = None) -> Path:
    """Resolve an explicit path, then ``DSLE_INSTANCE_CONFIG``, then the user config dir."""

    selected = path if path is not None else os.environ.get(INSTANCE_CONFIG_ENV)
    if selected is None:
        return default_instance_config_path()
    return Path(selected).expanduser().resolve(strict=False)


def _mapping(value: Any, field: str, source: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{source}: '{field}' must be an object")
    return dict(value)


def _required_text(raw: Mapping[str, Any], field: str, source: Path) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{source}: '{field}' must be a non-empty string")
    return value.strip()


def _required_int(raw: Mapping[str, Any], field: str, source: Path) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{source}: '{field}' must be an integer")
    return value


def _path(value: Any, field: str, source: Path, *, optional: bool = False) -> Path | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        qualifier = "a string or null" if optional else "a non-empty string"
        raise ConfigurationError(f"{source}: '{field}' must be {qualifier}")
    return Path(value).expanduser().resolve(strict=False)


def _parse_instance(name: str, value: Any, source: Path) -> InstanceConfig:
    raw = _mapping(value, f"instances.{name}", source)
    display_num = _required_int(raw, "display_num", source)
    if display_num < 0:
        raise ConfigurationError(f"{source}: instances.{name}.display_num cannot be negative")
    expected_display = f":{display_num}"
    display = str(raw.get("display", expected_display)).strip()
    if display != expected_display:
        raise ConfigurationError(
            f"{source}: instances.{name}.display must be {expected_display!r} "
            f"when display_num is {display_num}"
        )
    vnc_port = _required_int(raw, "vnc_port", source)
    if not 1 <= vnc_port <= 65535:
        raise ConfigurationError(f"{source}: instances.{name}.vnc_port must be in [1, 65535]")
    return InstanceConfig(
        name=name,
        display=display,
        display_num=display_num,
        desktop_name=_required_text(raw, "desktop_name", source),
        wineprefix=_path(raw.get("wineprefix"), "wineprefix", source),  # type: ignore[arg-type]
        vnc_port=vnc_port,
        xdg_runtime_dir=_path(  # type: ignore[arg-type]
            raw.get("xdg_runtime_dir"), "xdg_runtime_dir", source
        ),
        save_dir=_path(raw.get("save_dir"), "save_dir", source, optional=True),
    )


def load_instances(path: str | Path | None = None) -> dict[str, InstanceConfig]:
    """Load and validate every instance from a schema-versioned JSON document."""

    source = resolve_instance_config_path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(
            f"Instance configuration does not exist: {source}. "
            f"Pass instance_config=... or set {INSTANCE_CONFIG_ENV}."
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Could not read instance configuration {source}: {exc}") from exc
    raw = _mapping(document, "document", source)
    if raw.get("schema_version") != INSTANCE_SCHEMA_VERSION:
        raise ConfigurationError(f"{source}: schema_version must be {INSTANCE_SCHEMA_VERSION}")
    entries = _mapping(raw.get("instances"), "instances", source)
    if not entries:
        raise ConfigurationError(f"{source}: 'instances' cannot be empty")
    parsed: dict[str, InstanceConfig] = {}
    for name, value in entries.items():
        instance_name = validate_instance_name(name, field=f"{source}: instance name")
        parsed[instance_name] = _parse_instance(instance_name, value, source)
    displays = [instance.display_num for instance in parsed.values()]
    ports = [instance.vnc_port for instance in parsed.values()]
    prefixes = [instance.wineprefix for instance in parsed.values()]
    runtime_directories = [instance.xdg_runtime_dir for instance in parsed.values()]
    save_directories = [
        instance.save_dir for instance in parsed.values() if instance.save_dir is not None
    ]
    if len(set(displays)) != len(displays):
        raise ConfigurationError(f"{source}: display_num values must be unique")
    if len(set(ports)) != len(ports):
        raise ConfigurationError(f"{source}: vnc_port values must be unique")
    if len(set(prefixes)) != len(prefixes):
        raise ConfigurationError(f"{source}: wineprefix values must be unique")
    if len(set(runtime_directories)) != len(runtime_directories):
        raise ConfigurationError(f"{source}: xdg_runtime_dir values must be unique")
    if len(set(save_directories)) != len(save_directories):
        raise ConfigurationError(f"{source}: non-null save_dir values must be unique")
    return parsed


def resolve_instance(name: str, path: str | Path | None = None) -> InstanceConfig:
    """Resolve one named instance and report the available names on failure."""

    selected_name = validate_instance_name(name)
    instances = load_instances(path)
    try:
        return instances[selected_name]
    except KeyError as exc:
        available = ", ".join(sorted(instances))
        raise ConfigurationError(
            f"Unknown live-game instance {selected_name!r}. Available instances: {available}"
        ) from exc


def generate_instances(
    count: int,
    *,
    wineprefix_root: str | Path | None = None,
    display_start: int = 90,
    vnc_port_start: int = 5901,
    name_prefix: str = "dsr-",
) -> dict[str, InstanceConfig]:
    """Create deterministic instance records without starting Wine or the game."""

    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 30:
        raise ValueError("count must be an integer in [1, 30]")
    if isinstance(display_start, bool) or not isinstance(display_start, int) or display_start < 0:
        raise ValueError("display_start must be a non-negative integer")
    if name_prefix != "dsr-":
        raise ValueError("name_prefix must be 'dsr-' so instance names remain canonical")
    if isinstance(vnc_port_start, bool) or not isinstance(vnc_port_start, int):
        raise ValueError("vnc_port_start must be an integer")
    if not 1 <= vnc_port_start <= 65535 or vnc_port_start + count - 1 > 65535:
        raise ValueError("Generated VNC ports must stay in [1, 65535]")
    prefix_root = (
        Path(wineprefix_root).expanduser().resolve(strict=False)
        if wineprefix_root is not None
        else Path(user_data_path("dsle")) / "wineprefixes"
    )
    runtime_root = Path(tempfile.gettempdir()) / "dsle-runtime"
    generated: dict[str, InstanceConfig] = {}
    for offset in range(count):
        number = offset + 1
        name = f"{name_prefix}{number}"
        display_num = display_start + offset
        generated[name] = InstanceConfig(
            name=name,
            display=f":{display_num}",
            display_num=display_num,
            desktop_name=f"DSR_{number}",
            wineprefix=prefix_root / name,
            vnc_port=vnc_port_start + offset,
            xdg_runtime_dir=runtime_root / name,
            save_dir=None,
        )
    return generated


def instance_document(
    instances: Mapping[str, InstanceConfig], *, resolution: str | None = None
) -> dict[str, Any]:
    """Convert typed instances to the stable on-disk JSON representation."""

    if not instances:
        raise ConfigurationError("instances cannot be empty")
    for name, instance in instances.items():
        safe_name = validate_instance_name(name, field="instance name")
        if instance.name != safe_name:
            raise ConfigurationError(
                f"Instance mapping key {safe_name!r} does not match record name {instance.name!r}"
            )
    normalized_resolution = None
    if resolution is not None:
        _width, _height, normalized_resolution = _resolution({"resolution": resolution})
    state_roots = {instance.wineprefix.parent.parent for instance in instances.values()}
    document: dict[str, Any] = {
        "schema_version": INSTANCE_SCHEMA_VERSION,
        "instances": {
            name: {
                "display": instance.display,
                "display_num": instance.display_num,
                "desktop_name": instance.desktop_name,
                "wineprefix": str(instance.wineprefix),
                "vnc_port": instance.vnc_port,
                "xdg_runtime_dir": str(instance.xdg_runtime_dir),
                "save_dir": None if instance.save_dir is None else str(instance.save_dir),
                **(
                    {"resolution": normalized_resolution}
                    if normalized_resolution is not None
                    else {}
                ),
            }
            for name, instance in sorted(instances.items())
        },
    }
    if len(state_roots) == 1:
        document["state_dir"] = str(next(iter(state_roots)))
    return document


def write_instance_config(
    path: str | Path,
    instances: Mapping[str, InstanceConfig],
    *,
    overwrite: bool = False,
    resolution: str | None = None,
) -> Path:
    """Atomically write a host-readable config, refusing accidental overwrite by default."""

    destination = Path(path).expanduser().resolve(strict=False)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing instance configuration: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(instance_document(instances, resolution=resolution), indent=2) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary is None:  # Defensive invariant; NamedTemporaryFile always has a name.
            raise RuntimeUnavailableError("Could not allocate temporary instance configuration")
        os.replace(temporary, destination)
        # Docker user-namespace remapping can make the container-created file appear
        # as owned by ``nobody`` on the host. Keep it private from other users while
        # allowing the shared state-directory group to discover VNC endpoints.
        destination.chmod(0o640)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return destination


# Live lifecycle ------------------------------------------------------------
#
# A lightweight supervisor is deliberately kept in this module so the public
# CLI can synchronously start and stop instances without Docker-in-Docker or
# legacy shell scripts. Each supervisor owns only one configured instance.


def _load_document(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = resolve_instance_config_path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Instance configuration does not exist: {source}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Could not read instance configuration {source}: {exc}") from exc
    document = _mapping(value, "document", source)
    if document.get("schema_version") != INSTANCE_SCHEMA_VERSION:
        raise ConfigurationError(f"{source}: schema_version must be {INSTANCE_SCHEMA_VERSION}")
    _mapping(document.get("instances"), "instances", source)
    return source, document


def _selected_instances(
    config_path: str | Path, names: Sequence[str] | None
) -> tuple[Path, dict[str, Any], dict[str, InstanceConfig]]:
    source, document = _load_document(config_path)
    instances = load_instances(source)
    if names is None:
        selected_names = tuple(sorted(instances))
    else:
        if isinstance(names, (str, bytes)):
            raise ConfigurationError("names must be a sequence of instance names, not one string")
        selected_names = tuple(
            dict.fromkeys(validate_instance_name(name, field="instance name") for name in names)
        )
        if not selected_names:
            raise ConfigurationError("names must contain at least one instance name")
    unknown = sorted(set(selected_names) - set(instances))
    if unknown:
        raise ConfigurationError(
            f"Unknown instance(s): {', '.join(unknown)}. Available: {', '.join(sorted(instances))}"
        )
    return source, document, {name: instances[name] for name in selected_names}


def _state_directory(document: Mapping[str, Any], config_path: Path) -> Path:
    configured = document.get("state_dir") or os.environ.get("DSLE_STATE_DIR")
    if configured is not None:
        return Path(str(configured)).expanduser().resolve(strict=False)
    # Keeping state beside an explicit config is deterministic and works in containers.
    return config_path.parent.parent.resolve(strict=False)


@contextmanager
def _instance_launch_lock(state_dir: Path) -> Iterator[None]:
    """Serialize launches that share one durable runtime state directory."""

    run_directory = state_dir / "run"
    run_directory.mkdir(parents=True, exist_ok=True)
    lock_path = run_directory / _INSTANCE_LAUNCH_LOCK
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise RuntimeUnavailableError(
            f"Could not open instance launch lock {lock_path}: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeUnavailableError(
                f"Instance launch lock must be one regular file: {lock_path}"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            raise RuntimeUnavailableError(
                f"Could not acquire instance launch lock {lock_path}: {exc}"
            ) from exc
        try:
            yield
        finally:
            with suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _metadata_path(state_dir: Path, name: str) -> Path:
    safe_name = validate_instance_name(name)
    return state_dir / "run" / f"{safe_name}.json"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(dict(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if temporary is None:  # Defensive invariant; NamedTemporaryFile always has a name.
            raise RuntimeUnavailableError("Could not allocate temporary supervisor metadata")
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _read_metadata(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "error": f"Could not read {path}: {exc}"}
    return value if isinstance(value, dict) else {"status": "invalid", "error": "not an object"}


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _supervisor_arguments_match(
    arguments: Sequence[str], name: str, config_path: str | Path
) -> bool:
    """Require a supervisor command to match both its instance and exact config."""

    try:
        module_index = arguments.index("-m") + 1
        name_index = arguments.index("--name") + 1
        config_index = arguments.index("--config") + 1
    except (ValueError, IndexError):
        return False
    if max(module_index, name_index, config_index) >= len(arguments):
        return False
    try:
        expected_name = validate_instance_name(name)
    except ConfigurationError:
        return False
    expected_config = resolve_instance_config_path(config_path)
    actual_config = resolve_instance_config_path(arguments[config_index])
    return (
        arguments[module_index] == "dsle.runtime.instances"
        and "--supervise" in arguments
        and arguments[name_index] == expected_name
        and actual_config == expected_config
    )


def _is_instance_supervisor(pid: int, name: str, config_path: str | Path) -> bool:
    if not _pid_exists(pid):
        return False
    try:
        arguments = tuple(
            item.decode("utf-8", errors="replace")
            for item in (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
            if item
        )
    except OSError:
        return False
    return _supervisor_arguments_match(arguments, name, config_path)


def _public_status(
    name: str, metadata: dict[str, Any] | None, config_path: str | Path
) -> dict[str, Any]:
    if metadata is None:
        return {"name": name, "status": "not_started", "running": False}
    status = str(metadata.get("status", "unknown"))
    pid_value = metadata.get("supervisor_pid")
    pid = pid_value if isinstance(pid_value, int) else -1
    running = _is_instance_supervisor(pid, name, config_path)
    if status in {"starting", "ready"} and not running:
        status = "exited"
    result = dict(metadata)
    result.update({"name": name, "status": status, "running": running})
    return result


def instance_status(
    *, config_path: str | Path, names: Sequence[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Return durable lifecycle state for selected configured instances."""

    source, document, selected = _selected_instances(config_path, names)
    state_dir = _state_directory(document, source)
    return {
        name: _public_status(name, _read_metadata(_metadata_path(state_dir, name)), source)
        for name in selected
    }


def start_instances(
    *,
    config_path: str | Path,
    names: Sequence[str] | None = None,
    mode: str = "headless-vnc",
) -> dict[str, dict[str, Any]]:
    """Start one supervisor per instance and wait until each reports ready or failed."""

    if mode not in {"headless", "headless-vnc", "gui"}:
        raise ConfigurationError("mode must be headless, headless-vnc, or gui")
    source, document, selected = _selected_instances(config_path, names)
    state_dir = _state_directory(document, source)
    with _instance_launch_lock(state_dir):
        return _start_instances_locked(
            source=source,
            state_dir=state_dir,
            selected=selected,
            mode=mode,
        )


def _start_instances_locked(
    *,
    source: Path,
    state_dir: Path,
    selected: Mapping[str, InstanceConfig],
    mode: str,
) -> dict[str, dict[str, Any]]:
    """Launch selected instances while the state directory's process lock is held."""

    (state_dir / "logs").mkdir(parents=True, exist_ok=True)
    (state_dir / "run").mkdir(parents=True, exist_ok=True)
    current_statuses = {
        name: _public_status(name, _read_metadata(_metadata_path(state_dir, name)), source)
        for name in selected
    }
    for name, current in current_statuses.items():
        if current["running"] and current.get("mode") != mode:
            raise ConfigurationError(
                f"Instance {name!r} is already running in mode {current.get('mode')!r}; "
                f"stop it before requesting mode {mode!r}"
            )

    launched: dict[str, subprocess.Popen[bytes]] = {}
    for name in selected:
        if current_statuses[name]["running"]:
            continue
        # Invalidate a durable failed/stopped record before launching. Without
        # this parent-side write, the readiness loop can observe stale failure
        # metadata before the new supervisor has scheduled its first write and
        # tear down a replacement process that is actually starting normally.
        _atomic_json(
            _metadata_path(state_dir, name),
            {
                "name": name,
                "status": "starting",
                "mode": mode,
                "launch_requested_at": time.time(),
            },
        )
        log_path = state_dir / "logs" / f"{name}-supervisor.log"
        with log_path.open("ab", buffering=0) as log_handle:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "dsle.runtime.instances",
                    "--supervise",
                    "--config",
                    str(source),
                    "--name",
                    name,
                    "--mode",
                    mode,
                ],
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        launched[name] = process

    pending = set(launched)
    readiness_timeout_s = INSTANCE_READINESS_TIMEOUT_S
    deadline = time.monotonic() + readiness_timeout_s
    while pending and time.monotonic() < deadline:
        for name in tuple(pending):
            metadata = _read_metadata(_metadata_path(state_dir, name))
            status = None if metadata is None else metadata.get("status")
            process = launched[name]
            if status in {"ready", "failed"} or process.poll() is not None:
                pending.remove(name)
        if pending:
            time.sleep(0.1)
    timeout_error = f"Supervisor did not report ready within {readiness_timeout_s:g} seconds"
    for name in sorted(pending):
        _terminate_child(launched[name], timeout_s=3.0)
        path = _metadata_path(state_dir, name)
        metadata = _read_metadata(path) or {"name": name}
        metadata.update(
            {
                "status": "failed",
                "failed_at": time.time(),
                "error": timeout_error,
            }
        )
        _atomic_json(path, metadata)

    result = instance_status(config_path=source, names=tuple(selected))
    failures = {
        name: status
        for name, status in result.items()
        if status.get("status") != "ready" or status.get("running") is not True
    }
    if failures and launched:
        original = {name: dict(status) for name, status in result.items()}
        try:
            rolled_back = stop_instances(config_path=source, names=tuple(launched))
        except Exception as exc:
            for name in launched:
                result[name]["rollback_error"] = f"{type(exc).__name__}: {exc}"
        else:
            for name, status in rolled_back.items():
                result[name] = dict(status)
                result[name]["status"] = "failed" if name in failures else "rolled_back"
                if name in failures:
                    result[name]["error"] = original[name].get(
                        "error", "Instance did not become ready"
                    )
                else:
                    result[name]["error"] = "Rolled back after another instance failed"
    return result


def _terminate_process_group(pid: int, timeout_s: float = 8.0) -> None:
    if not _pid_exists(pid):
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout_s
    while _pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _pid_exists(pid):
        with suppress(ProcessLookupError):
            os.killpg(pid, signal.SIGKILL)


def stop_instances(
    *, config_path: str | Path, names: Sequence[str] | None = None
) -> dict[str, dict[str, Any]]:
    """Stop only the exact supervisors and child process groups recorded for each instance."""

    source, document, selected = _selected_instances(config_path, names)
    state_dir = _state_directory(document, source)
    for name in selected:
        path = _metadata_path(state_dir, name)
        metadata = _read_metadata(path)
        if metadata is None:
            continue
        supervisor_value = metadata.get("supervisor_pid")
        supervisor_pid = supervisor_value if isinstance(supervisor_value, int) else -1
        controlled = _is_instance_supervisor(supervisor_pid, name, source)
        if controlled:
            with suppress(ProcessLookupError):
                os.kill(supervisor_pid, signal.SIGTERM)
            deadline = time.monotonic() + 15.0
            while (
                _is_instance_supervisor(supervisor_pid, name, source)
                and time.monotonic() < deadline
            ):
                time.sleep(0.1)
        if controlled:
            for field in ("wine_pid", "vnc_pid", "xorg_pid"):
                value = metadata.get(field)
                if isinstance(value, int):
                    _terminate_process_group(value, timeout_s=3.0)
            if _is_instance_supervisor(supervisor_pid, name, source):
                with suppress(ProcessLookupError):
                    os.kill(supervisor_pid, signal.SIGKILL)
        final = _read_metadata(path) or metadata
        final.update(
            {
                "name": name,
                "status": "stopped",
                "stopped_at": time.time(),
            }
        )
        _atomic_json(path, final)
    return instance_status(config_path=source, names=tuple(selected))


def _command(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeUnavailableError(f"Required runtime command is not installed: {name}")
    return path


def _resolution(raw: Mapping[str, Any]) -> tuple[int, int, str]:
    value = str(
        raw.get("resolution")
        or raw.get("desktop_res")
        or os.environ.get("DSLE_DESKTOP_RES", "800x600")
    ).lower()
    pieces = value.split("x", 1)
    if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
        raise ConfigurationError(f"Invalid instance resolution: {value!r}")
    width, height = (int(piece) for piece in pieces)
    if not 320 <= width <= 3840 or not 240 <= height <= 2160:
        raise ConfigurationError(f"Instance resolution is outside 320x240..3840x2160: {value}")
    return width, height, value


def _xorg_template() -> Path:
    configured = os.environ.get("DSLE_XORG_TEMPLATE")
    candidates: tuple[Path, ...]
    if configured:
        candidates = (Path(configured).expanduser().resolve(strict=False),)
    else:
        project_root = Path(__file__).resolve().parents[3]
        candidates = (
            project_root / "xorg" / "nvidia.conf.template",
            project_root / "docker" / "xorg" / "nvidia.conf.template",
        )
    for path in candidates:
        if path.is_file():
            return path
    raise RuntimeUnavailableError(
        f"NVIDIA Xorg template is missing; checked: {', '.join(str(path) for path in candidates)}"
    )


def _wait_for_display(display: str, process: subprocess.Popen[bytes], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    xdpyinfo = _command("xdpyinfo")
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeUnavailableError(
                f"Xorg for {display} exited with code {process.returncode}"
            )
        check = subprocess.run(
            [xdpyinfo, "-display", display],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2.0,
        )
        if check.returncode == 0:
            return
        time.sleep(0.1)
    raise RuntimeUnavailableError(f"Xorg display {display} was not ready within {timeout_s:g}s")


def _ensure_wineprefix(prefix: Path) -> None:
    if prefix.is_symlink():
        raise RuntimeUnavailableError(f"Wine prefix path cannot be a symbolic link: {prefix}")
    if prefix.is_dir():
        _claim_runtime_tree(prefix)
        return
    if prefix.exists():
        raise RuntimeUnavailableError(f"Wine prefix path is not a directory: {prefix}")
    template = (
        Path(
            os.environ.get(
                "WINEPREFIX_TEMPLATE", str(Path(user_data_path("dsle")) / "wine-template")
            )
        )
        .expanduser()
        .resolve(strict=False)
    )
    if not template.is_dir():
        raise RuntimeUnavailableError(
            f"Wine prefix template is missing: {template}. Set WINEPREFIX_TEMPLATE."
        )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template, prefix, symlinks=True)


def _claim_runtime_tree(root: Path) -> None:
    """Make one trusted runtime tree usable by the current container user.

    Graceful container shutdown recursively returns persistent output to the
    calling host UID. A later container runs Wine as root, and Wine refuses a
    prefix owned by another UID. Reclaim only the configured prefix/XDG tree;
    never follow the intentional ``dosdevices`` symlinks in a Wine prefix.
    """

    if root.is_symlink() or not root.is_dir():
        raise RuntimeUnavailableError(f"Runtime path must be one real directory: {root}")
    uid, gid = os.geteuid(), os.getegid()
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in (*directories, *files):
            path = current_path / name
            try:
                os.chown(path, uid, gid, follow_symlinks=False)
            except OSError as exc:
                raise RuntimeUnavailableError(
                    f"Could not reclaim persistent runtime path {path}: {exc}"
                ) from exc
        try:
            os.chown(current_path, uid, gid, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeUnavailableError(
                f"Could not reclaim persistent runtime directory {current_path}: {exc}"
            ) from exc


def _bootstrap_save(instance: InstanceConfig, deadline: float) -> Path:
    from dsle.runtime.saves import (
        ACTIVE_SAVE_NAME,
        SaveManager,
        discover_save_directory,
        resolve_scenario_directory,
    )

    asset_dir = (
        Path(os.environ.get("DSLE_ASSET_DIR", str(Path(__file__).resolve().parents[3] / "assets")))
        .expanduser()
        .resolve(strict=False)
    )
    last_error = "save directory has not appeared"
    while time.monotonic() < deadline:
        try:
            save_dir = instance.save_dir or discover_save_directory(instance.wineprefix)
            if not (save_dir / ACTIVE_SAVE_NAME).is_file():
                SaveManager(resolve_scenario_directory(asset_dir), save_dir).load("_beginning.sl2")
            return save_dir
        except (RuntimeUnavailableError, AssetError, FileNotFoundError) as exc:
            last_error = str(exc)
            time.sleep(0.25)
    raise RuntimeUnavailableError(f"Could not initialize the active save: {last_error}")


def _terminate_child(process: subprocess.Popen[bytes], timeout_s: float = 3.0) -> None:
    """Terminate and reap one child process group owned by this supervisor."""

    if process.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=timeout_s)


def _configure_wine_audio(environment: Mapping[str, str]) -> None:
    """Force Wine's PulseAudio driver before starting FAudio-based DSR audio.

    Wine can otherwise select ALSA in a headless container. Dark Souls then
    reaches its splash screen with no usable voice and crashes while FAudio
    destroys a null mastering voice. The registry write is intentionally
    repeated for reusable output prefixes created by earlier runtime images.
    """

    result = subprocess.run(
        [
            _command("wine"),
            "reg",
            "add",
            r"HKCU\Software\Wine\Drivers",
            "/v",
            "Audio",
            "/t",
            "REG_SZ",
            "/d",
            "pulse",
            "/f",
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env=dict(environment),
        check=False,
        timeout=15.0,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no error output"
        raise RuntimeUnavailableError(
            f"Could not configure Wine PulseAudio driver (exit {result.returncode}): {detail}"
        )


def _supervise(config_path: Path, name: str, mode: str) -> int:
    source, document, selected = _selected_instances(config_path, (name,))
    instance = selected[name]
    raw_instances = _mapping(document.get("instances"), "instances", source)
    raw = _mapping(raw_instances[name], f"instances.{name}", source)
    state_dir = _state_directory(document, source)
    run_dir = state_dir / "run"
    log_dir = state_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = _metadata_path(state_dir, name)
    metadata: dict[str, Any] = {
        "name": name,
        "status": "starting",
        "supervisor_pid": os.getpid(),
        "display": instance.display,
        "vnc_port": instance.vnc_port,
        "mode": mode,
        "started_at": time.time(),
    }
    _atomic_json(metadata_path, metadata)
    children: list[subprocess.Popen[bytes]] = []
    stop_requested = False
    child_environment: dict[str, str] | None = None

    def request_stop(_signal_number: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        if mode not in {"headless", "headless-vnc", "gui"}:
            raise ConfigurationError(f"Unsupported instance mode: {mode}")
        _ensure_wineprefix(instance.wineprefix)
        instance.xdg_runtime_dir.mkdir(parents=True, exist_ok=True)
        _claim_runtime_tree(instance.xdg_runtime_dir)
        instance.xdg_runtime_dir.chmod(0o700)
        width, height, resolution = _resolution(raw)
        game_dir = (
            Path(str(document.get("game_dir") or os.environ.get("DSLE_GAME_DIR", "")))
            .expanduser()
            .resolve(strict=False)
        )
        game_executable = game_dir / "DarkSoulsRemastered.exe"
        if not game_executable.is_file():
            raise RuntimeUnavailableError(
                f"DarkSoulsRemastered.exe is missing from DSLE_GAME_DIR: {game_executable}"
            )
        verify_game_executable(game_executable)

        child_environment = os.environ.copy()
        child_environment.update(
            {
                "DISPLAY": instance.display,
                "WINEPREFIX": str(instance.wineprefix),
                "WINEARCH": "win64",
                "XDG_RUNTIME_DIR": str(instance.xdg_runtime_dir),
                "DSLE_DESKTOP_RES": resolution,
            }
        )

        if mode != "gui":
            template = _xorg_template().read_text(encoding="utf-8")
            xorg_config = run_dir / f"xorg-{name}.conf"
            xorg_config.write_text(
                template.replace("@WIDTH@", str(width))
                .replace("@HEIGHT@", str(height))
                .replace("@DEPTH@", "24"),
                encoding="utf-8",
            )
            xorg_log = log_dir / f"xorg-{name}.log"
            xorg = subprocess.Popen(
                [
                    _command("Xorg"),
                    instance.display,
                    "-noreset",
                    "-nolisten",
                    "tcp",
                    "-logfile",
                    str(xorg_log),
                    "-config",
                    str(xorg_config),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=child_environment,
                start_new_session=True,
                close_fds=True,
            )
            children.append(xorg)
            metadata["xorg_pid"] = xorg.pid
            _atomic_json(metadata_path, metadata)
            _wait_for_display(instance.display, xorg, 15.0)
        else:
            probe = subprocess.run(
                [_command("xdpyinfo"), "-display", instance.display],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=3.0,
            )
            if probe.returncode != 0:
                raise RuntimeUnavailableError(
                    f"GUI mode requires an existing X11 display: {instance.display}"
                )

        _configure_wine_audio(child_environment)

        if mode == "headless-vnc":
            vnc_log = (log_dir / f"vnc-{name}.log").open("ab", buffering=0)
            try:
                vnc = subprocess.Popen(
                    [
                        _command("x11vnc"),
                        "-display",
                        instance.display,
                        "-rfbport",
                        str(instance.vnc_port),
                        "-forever",
                        "-shared",
                        "-xkb",
                        "-noxdamage",
                        "-nopw",
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=vnc_log,
                    stderr=subprocess.STDOUT,
                    env=child_environment,
                    start_new_session=True,
                    close_fds=True,
                )
            finally:
                vnc_log.close()
            children.append(vnc)
            metadata["vnc_pid"] = vnc.pid

        wine_log = (log_dir / f"wine-{name}.log").open("ab", buffering=0)
        try:
            wine = subprocess.Popen(
                [
                    _command("wine"),
                    "explorer",
                    f"/desktop={instance.desktop_name},{resolution}",
                    game_executable.name,
                ],
                cwd=game_dir,
                stdin=subprocess.DEVNULL,
                stdout=wine_log,
                stderr=subprocess.STDOUT,
                env=child_environment,
                start_new_session=True,
                close_fds=True,
            )
        finally:
            wine_log.close()
        children.append(wine)
        metadata["wine_pid"] = wine.pid
        _atomic_json(metadata_path, metadata)

        from dsle.runtime.memory import ProcessLocator

        ready_deadline = time.monotonic() + 35.0
        while time.monotonic() < ready_deadline:
            try:
                ProcessLocator(instance.wineprefix).locate()
                break
            except RuntimeUnavailableError as exc:
                if wine.poll() is not None:
                    raise RuntimeUnavailableError(
                        f"Wine launcher exited before the game was ready (code {wine.returncode})"
                    ) from exc
                time.sleep(0.25)
        else:
            raise RuntimeUnavailableError("DarkSoulsRemastered.exe did not start within 35 seconds")
        save_dir = _bootstrap_save(instance, time.monotonic() + 30.0)
        metadata.update(
            {
                "status": "ready",
                "ready_at": time.time(),
                "save_dir": str(save_dir),
                "resolution": resolution,
            }
        )
        _atomic_json(metadata_path, metadata)

        locator = ProcessLocator(instance.wineprefix)
        while not stop_requested:
            try:
                locator.locate()
            except RuntimeUnavailableError:
                if wine.poll() is not None:
                    break
            time.sleep(0.5)
        metadata.update(
            {
                "status": "stopped" if stop_requested else "exited",
                "stopped_at": time.time(),
            }
        )
        _atomic_json(metadata_path, metadata)
        return 0 if stop_requested else 1
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "failed_at": time.time(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_json(metadata_path, metadata)
        return 1
    finally:
        for child in reversed(children):
            _terminate_child(child, timeout_s=3.0)
        if child_environment is not None and shutil.which("wineserver"):
            subprocess.run(
                [str(shutil.which("wineserver")), "-k"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=child_environment,
                check=False,
                timeout=10.0,
            )


def _supervisor_main(arguments: Sequence[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", required=True)
    parsed = parser.parse_args(arguments)
    if not parsed.supervise:
        parser.error("this module entry point is reserved for --supervise")
    return _supervise(parsed.config, parsed.name, parsed.mode)


if __name__ == "__main__":  # pragma: no cover - exercised inside the runtime container
    raise SystemExit(_supervisor_main(sys.argv[1:]))
