"""Authenticated in-container RPC server for the authoritative DSLE environment."""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import re
import signal
import socket
import socketserver
import stat
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import replace
from numbers import Real
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np

import dsle
from dsle.exceptions import ConfigurationError, RuntimeUnavailableError
from dsle.models import InstanceConfig, RewardConfig
from dsle.progress import EventSink
from dsle.remote import (
    PROTOCOL,
    ProtocolError,
    default_socket_path,
    default_token_path,
    ping,
    read_auth_token,
    receive_frame,
    send_frame,
    tokens_equal,
)
from dsle.runtime.instances import (
    generate_instances,
    load_instances,
    stop_instances,
    validate_instance_name,
    write_instance_config,
)
from dsle.runtime.instances import (
    start_instances as start_runtime_instances,
)

REQUEST_TIMEOUT = 300.0
PREAUTH_REQUEST_TIMEOUT = 5.0
MAX_CONNECTIONS = 64
_AUTO_INSTANCE = re.compile(r"dsr-([1-9]|[12][0-9]|30)")
_AUTO_INSTANCE_COUNT = 30
_MANAGED_DOCUMENT_FIELDS = frozenset({"schema_version", "instances", "state_dir"})
_MANAGED_INSTANCE_FIELDS = frozenset(
    {
        "display",
        "display_num",
        "desktop_name",
        "wineprefix",
        "vnc_port",
        "xdg_runtime_dir",
        "save_dir",
        "resolution",
    }
)
_ENVIRONMENT_KWARGS = {
    "instance",
    "difficulty",
    "obs_mode",
    "action_ms",
    "action_repeat",
    "max_steps",
    "auto_lock_on",
    "lock_on_interval",
    "reward",
    "render_mode",
    "verbose",
}


def _local_environment(boss: str, **kwargs: Any) -> gym.Env[Any, Any]:
    """Force the authoritative in-process implementation and prevent RPC recursion."""

    return cast(gym.Env[Any, Any], dsle.make(boss, runtime="local", **kwargs))


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{name} must be a JSON object")
    return dict(value)


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{name} must be a non-empty string")
    return value.strip()


def _space_description(space: gym.Space[Any]) -> dict[str, Any]:
    if isinstance(space, gym.spaces.Discrete):
        return {"kind": "discrete", "n": int(space.n), "start": int(space.start)}
    if isinstance(space, gym.spaces.Box):
        low = np.asarray(space.low)
        high = np.asarray(space.high)
        dtype = space.dtype
        if dtype is None:
            raise RuntimeUnavailableError("RPC Box spaces must declare a dtype")
        if low.size and (not np.all(low == low.flat[0]) or not np.all(high == high.flat[0])):
            raise RuntimeUnavailableError("RPC supports only uniformly bounded Box spaces")
        return {
            "kind": "box",
            "shape": list(space.shape),
            "dtype": dtype.str,
            "low": 0 if low.size == 0 else low.flat[0].item(),
            "high": 0 if high.size == 0 else high.flat[0].item(),
        }
    raise RuntimeUnavailableError(f"RPC does not support {type(space).__name__} spaces")


def _validate_under(path: Path, root: Path, description: str) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConfigurationError(f"{description} must be inside DSLE_STATE_DIR ({root})") from exc
    return resolved


class EnvironmentState:
    """Own environments and instance supervisors inside one container namespace."""

    def __init__(
        self,
        *,
        state_dir: str | Path,
        env_factory: Callable[..., gym.Env[Any, Any]] = _local_environment,
        allow_instance_start: bool = True,
    ):
        self.state_dir = Path(state_dir).expanduser().resolve(strict=False)
        configured = os.environ.get(
            "DSLE_INSTANCE_CONFIG", str(self.state_dir / "config" / "instances.json")
        )
        self.instance_config = _validate_under(
            Path(configured), self.state_dir, "DSLE_INSTANCE_CONFIG"
        )
        self.env_factory = env_factory
        self.allow_instance_start = allow_instance_start
        self._environments: dict[str, gym.Env[Any, Any]] = {}
        self._environment_locks: dict[str, threading.Lock] = {}
        self._environment_instances: dict[str, str] = {}
        self._reserved_instances: set[str] = set()
        self._started_instances: set[str] = set()
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._started_at = time.time()
        self.shutdown_callback: Callable[[], None] | None = None

    def _acquire_environment(
        self, payload: Mapping[str, Any]
    ) -> tuple[gym.Env[Any, Any], threading.Lock]:
        """Resolve and lock an environment before lifecycle state can remove it."""

        env_id = _required_text(payload.get("env_id"), "payload.env_id")
        with self._lock:
            try:
                environment = self._environments[env_id]
                lock = self._environment_locks[env_id]
            except KeyError as exc:
                raise ProtocolError(f"Unknown or closed environment id: {env_id}") from exc
            lock.acquire()
            return environment, lock

    def _managed_instance_pool(self) -> dict[str, InstanceConfig]:
        """Return the canonical pool used by API-owned persistent state."""

        generated = generate_instances(
            _AUTO_INSTANCE_COUNT,
            wineprefix_root=self.state_dir / "wineprefixes",
            display_start=90,
            vnc_port_start=5901,
        )
        return {
            name: replace(instance, xdg_runtime_dir=self.state_dir / "xdg" / name)
            for name, instance in generated.items()
        }

    def _expand_managed_instance_config(
        self,
        instances: Mapping[str, InstanceConfig],
        document: Mapping[str, Any],
    ) -> None:
        """Expand a canonical persisted subset without replacing durable instance data."""

        unexpected_document_fields = sorted(set(document) - _MANAGED_DOCUMENT_FIELDS)
        raw_entries = document.get("instances")
        if not isinstance(raw_entries, Mapping):
            raise ConfigurationError(
                f"Cannot automatically expand malformed instance config {self.instance_config}"
            )
        unexpected_entry_fields: dict[str, list[str]] = {}
        resolutions: set[str | None] = set()
        for name, raw_entry in raw_entries.items():
            if not isinstance(name, str) or not isinstance(raw_entry, Mapping):
                raise ConfigurationError(
                    f"Cannot automatically expand malformed instance config {self.instance_config}"
                )
            unexpected = sorted(set(raw_entry) - _MANAGED_INSTANCE_FIELDS)
            if unexpected:
                unexpected_entry_fields[name] = unexpected
            resolution = raw_entry.get("resolution")
            if resolution is not None and not isinstance(resolution, str):
                raise ConfigurationError(
                    f"Cannot automatically expand instance config {self.instance_config}: "
                    f"instances.{name}.resolution must be a string or null"
                )
            resolutions.add(resolution)

        expected = self._managed_instance_pool()
        incompatible = sorted(
            name
            for name, instance in instances.items()
            if replace(instance, save_dir=None) != expected[name]
        )
        reasons: list[str] = []
        configured_state = document.get("state_dir")
        if (
            configured_state is not None
            and Path(str(configured_state)).expanduser().resolve(strict=False) != self.state_dir
        ):
            reasons.append("state_dir is not the managed output root")
        if unexpected_document_fields:
            reasons.append("unsupported document fields: " + ", ".join(unexpected_document_fields))
        if unexpected_entry_fields:
            details = "; ".join(
                f"{name}: {', '.join(fields)}"
                for name, fields in sorted(unexpected_entry_fields.items())
            )
            reasons.append(f"unsupported instance fields: {details}")
        if len(resolutions) > 1:
            reasons.append("existing instances use different resolutions")
        if incompatible:
            reasons.append("noncanonical managed instances: " + ", ".join(incompatible))
        if reasons:
            raise ConfigurationError(
                f"Cannot automatically expand instance config {self.instance_config}; "
                + "; ".join(reasons)
            )

        expanded = dict(expected)
        expanded.update(instances)
        resolution = next(iter(resolutions), None)
        write_instance_config(
            self.instance_config,
            expanded,
            overwrite=True,
            resolution=resolution,
        )

    def _ensure_configured_instance(self, name: str) -> None:
        name = validate_instance_name(name, field="instance name")
        if self.instance_config.is_file():
            instances = load_instances(self.instance_config)
            try:
                loaded_document = json.loads(self.instance_config.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigurationError(
                    f"Could not validate instance config {self.instance_config}: {exc}"
                ) from exc
            if not isinstance(loaded_document, Mapping):
                raise ConfigurationError(
                    f"Could not validate instance config {self.instance_config}: document must be an object"
                )
            document = dict(loaded_document)
            configured_state = document.get("state_dir")
            if configured_state is not None:
                _validate_under(Path(str(configured_state)), self.state_dir, "instance state_dir")
            configured_game = document.get("game_dir")
            if configured_game is not None:
                expected_game = Path(os.environ.get("DSLE_GAME_DIR", "/opt/dsle/game")).resolve(
                    strict=False
                )
                actual_game = Path(str(configured_game)).expanduser().resolve(strict=False)
                if actual_game != expected_game:
                    raise ConfigurationError(
                        f"Instance game_dir must be the mounted DSLE_GAME_DIR ({expected_game})"
                    )
            for instance in instances.values():
                _validate_under(
                    instance.wineprefix,
                    self.state_dir,
                    f"instance {instance.name} wineprefix",
                )
                _validate_under(
                    instance.xdg_runtime_dir,
                    self.state_dir,
                    f"instance {instance.name} xdg_runtime_dir",
                )
                if instance.save_dir is not None:
                    _validate_under(
                        instance.save_dir,
                        self.state_dir,
                        f"instance {instance.name} save_dir",
                    )
            if name not in instances:
                self._expand_managed_instance_config(instances, document)
            return
        match = _AUTO_INSTANCE.fullmatch(name)
        if match is None:
            raise ConfigurationError(
                "Automatic instance configuration only accepts dsr-1 through dsr-30"
            )
        # A fresh API-managed state directory gets the complete deterministic
        # pool.  This makes later session.make(..., instance="dsr-N") calls and
        # server restarts work without rewriting an existing user config.
        instances = self._managed_instance_pool()
        write_instance_config(self.instance_config, instances)

    def start_instance(self, name: str, mode: str) -> dict[str, Any]:
        if not self.allow_instance_start:
            raise RuntimeUnavailableError(
                "This server was started with instance lifecycle disabled"
            )
        if mode not in {"headless", "headless-vnc", "gui"}:
            raise ProtocolError("instance_mode must be headless, headless-vnc, or gui")
        name = validate_instance_name(name, field="instance name")
        with self._lifecycle_lock, self._lock:
            self._ensure_configured_instance(name)
            status = start_runtime_instances(
                config_path=self.instance_config,
                names=(name,),
                mode=mode,
            )[name]
            if status.get("status") != "ready" or status.get("running") is not True:
                rollback_error = self._rollback_running_instances((name,), {name: status})
                detail = status.get("error", status)
                if rollback_error is not None:
                    detail = f"{detail}; cleanup also failed: {rollback_error}"
                raise RuntimeUnavailableError(f"Instance {name!r} did not become ready: {detail}")
            self._started_instances.add(name)
            return status

    def start_instance_pool(self, count: int, mode: str) -> dict[str, dict[str, Any]]:
        """Start a numbered pool with one parallel runtime lifecycle call."""

        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 30:
            raise ProtocolError("count must be an integer in [1, 30]")
        if mode not in {"headless", "headless-vnc", "gui"}:
            raise ProtocolError("instance_mode must be headless, headless-vnc, or gui")
        if not self.allow_instance_start:
            raise RuntimeUnavailableError(
                "This server was started with instance lifecycle disabled"
            )
        names = tuple(f"dsr-{index}" for index in range(1, count + 1))
        with self._lifecycle_lock, self._lock:
            self._ensure_configured_instance(names[-1])
            statuses = start_runtime_instances(
                config_path=self.instance_config,
                names=names,
                mode=mode,
            )
            failures = {
                name: status
                for name, status in statuses.items()
                if status.get("status") != "ready" or status.get("running") is not True
            }
            if failures:
                rollback_error = self._rollback_running_instances(names, statuses)
                detail = f"Instance pool did not become ready: {failures}"
                if rollback_error is not None:
                    detail = f"{detail}; cleanup also failed: {rollback_error}"
                raise RuntimeUnavailableError(detail)
            self._started_instances.update(names)
            return statuses

    def _rollback_running_instances(
        self,
        names: Sequence[str],
        statuses: Mapping[str, Mapping[str, Any]],
    ) -> str | None:
        """Stop failed-start remnants and retain any survivors for shutdown retry."""

        running = {
            name for name in names if name in statuses and statuses[name].get("running") is True
        }
        if not running:
            return None
        self._started_instances.update(running)
        try:
            stopped = stop_instances(
                config_path=self.instance_config,
                names=tuple(sorted(running)),
            )
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        survivors = (running - set(stopped)) | {
            name for name, status in stopped.items() if status.get("running") is True
        }
        self._started_instances.difference_update(running - survivors)
        if survivors:
            return "instances remained running: " + ", ".join(sorted(survivors))
        return None

    def _create(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], None]:
        with self._lifecycle_lock:
            return self._create_locked(payload)

    def _create_locked(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], None]:
        boss = _required_text(payload.get("boss"), "payload.boss")
        kwargs = _mapping(payload.get("kwargs", {}), "payload.kwargs")
        forbidden = sorted(set(kwargs) - _ENVIRONMENT_KWARGS)
        if forbidden:
            raise ProtocolError(f"Unsupported environment option(s): {', '.join(forbidden)}")
        instance = validate_instance_name(
            _required_text(kwargs.get("instance", "dsr-1"), "kwargs.instance"),
            field="kwargs.instance",
        )
        kwargs["instance"] = instance
        start_requested = payload.get("start_instance", False)
        if not isinstance(start_requested, bool):
            raise ProtocolError("start_instance must be a boolean")
        mode = _required_text(payload.get("instance_mode", "headless"), "instance_mode")

        reward = kwargs.get("reward")
        if reward is not None:
            reward_values = _mapping(reward, "reward")
            allowed_reward = set(RewardConfig.__dataclass_fields__)
            unknown = sorted(set(reward_values) - allowed_reward)
            if unknown:
                raise ProtocolError(f"Unknown reward field(s): {', '.join(unknown)}")
            kwargs["reward"] = RewardConfig(**reward_values)
        # Reject an unknown task before an expensive game-process launch.
        dsle.load_boss(boss)
        with self._lock:
            occupied = instance in self._reserved_instances or instance in (
                self._environment_instances.values()
            )
            if occupied:
                raise ConfigurationError(
                    f"Instance {instance!r} already belongs to an open environment"
                )
            self._reserved_instances.add(instance)

        environment: gym.Env[Any, Any] | None = None
        try:
            if start_requested:
                self.start_instance(instance, mode)
            kwargs["asset_dir"] = os.environ.get("DSLE_ASSET_DIR", "/opt/dsle/assets")
            kwargs["instance_config"] = self.instance_config
            environment = self.env_factory(boss, **kwargs)
            action_space = _space_description(environment.action_space)
            observation_space = _space_description(environment.observation_space)
            if not isinstance(environment.metadata, Mapping):
                raise RuntimeUnavailableError("Environment metadata must be a mapping")
            metadata = dict(environment.metadata)
            env_id = os.urandom(16).hex()
            with self._lock:
                self._environments[env_id] = environment
                self._environment_locks[env_id] = threading.Lock()
                self._environment_instances[env_id] = instance
                self._reserved_instances.discard(instance)
        except Exception:
            with self._lock:
                self._reserved_instances.discard(instance)
            if environment is not None:
                with suppress(Exception):
                    environment.close()
            raise
        return (
            {
                "env_id": env_id,
                "action_space": action_space,
                "observation_space": observation_space,
                "metadata": metadata,
            },
            None,
        )

    @contextmanager
    def _environment_call(
        self,
        payload: Mapping[str, Any],
        event_sink: EventSink | None,
    ) -> Iterator[gym.Env[Any, Any]]:
        """Lock one environment and temporarily connect its verbose event stream."""

        environment, lock = self._acquire_environment(payload)
        setter = getattr(environment.unwrapped, "set_event_sink", None)
        try:
            if callable(setter):
                setter(event_sink)
            yield environment
        finally:
            try:
                if callable(setter):
                    setter(None)
            finally:
                lock.release()

    def _reset(
        self,
        payload: Mapping[str, Any],
        *,
        event_sink: EventSink | None = None,
    ) -> tuple[dict[str, Any], np.ndarray]:
        seed = payload.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ProtocolError("reset seed must be an integer or null")
        options = payload.get("options")
        if options is not None:
            options = _mapping(options, "reset options")
        with self._environment_call(payload, event_sink) as environment:
            observation, info = environment.reset(seed=seed, options=options)
        observation_array = np.asarray(observation)
        if not environment.observation_space.contains(observation_array):
            raise RuntimeUnavailableError("Environment reset returned an out-of-space observation")
        if not isinstance(info, Mapping):
            raise RuntimeUnavailableError("Environment reset info must be a mapping")
        return {"info": dict(info)}, observation_array

    def _step(
        self,
        payload: Mapping[str, Any],
        *,
        event_sink: EventSink | None = None,
    ) -> tuple[dict[str, Any], np.ndarray]:
        action = payload.get("action")
        if isinstance(action, bool) or not isinstance(action, int):
            raise ProtocolError("step action must be an integer")
        with self._environment_call(payload, event_sink) as environment:
            observation, reward, terminated, truncated, info = environment.step(action)
        observation_array = np.asarray(observation)
        if not environment.observation_space.contains(observation_array):
            raise RuntimeUnavailableError("Environment step returned an out-of-space observation")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, Real)
            or not math.isfinite(float(reward))
        ):
            raise RuntimeUnavailableError("Environment step reward must be a finite number")
        if not isinstance(terminated, bool) or not isinstance(truncated, bool):
            raise RuntimeUnavailableError("Environment terminal flags must be booleans")
        if not isinstance(info, Mapping):
            raise RuntimeUnavailableError("Environment step info must be a mapping")
        return (
            {
                "reward": float(reward),
                "terminated": terminated,
                "truncated": truncated,
                "info": dict(info),
            },
            observation_array,
        )

    def _observe(
        self,
        payload: Mapping[str, Any],
        *,
        event_sink: EventSink | None = None,
    ) -> tuple[dict[str, Any], np.ndarray]:
        with self._environment_call(payload, event_sink) as environment:
            observer = getattr(environment.unwrapped, "observe", None)
            if not callable(observer):
                raise RuntimeUnavailableError("Environment does not support passive observation")
            observation, info = observer()
        observation_array = np.asarray(observation)
        if not environment.observation_space.contains(observation_array):
            raise RuntimeUnavailableError(
                "Environment observe returned an out-of-space observation"
            )
        if not isinstance(info, Mapping):
            raise RuntimeUnavailableError("Environment observe info must be a mapping")
        return {"info": dict(info)}, observation_array

    def _return_to_menu(
        self,
        payload: Mapping[str, Any],
        *,
        event_sink: EventSink | None = None,
    ) -> tuple[dict[str, Any], None]:
        timeout_s = payload.get("timeout_s", 30.0)
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, Real)
            or not math.isfinite(float(timeout_s))
            or float(timeout_s) <= 0
        ):
            raise ProtocolError("return_to_menu timeout_s must be a finite positive number")
        with self._environment_call(payload, event_sink) as environment:
            returner = getattr(environment.unwrapped, "return_to_menu", None)
            if not callable(returner):
                raise RuntimeUnavailableError("Environment does not support returning to the menu")
            returner(timeout_s=float(timeout_s))
        return {"at_menu": True}, None

    def _set_player_hp(
        self,
        payload: Mapping[str, Any],
        *,
        event_sink: EventSink | None = None,
    ) -> tuple[dict[str, Any], None]:
        hp = payload.get("hp")
        if isinstance(hp, bool) or not isinstance(hp, int):
            raise ProtocolError("player hp must be an integer")
        with self._environment_call(payload, event_sink) as environment:
            writer = getattr(environment.unwrapped, "set_player_hp", None)
            if not callable(writer):
                raise RuntimeUnavailableError("Environment does not support player HP writes")
            info = writer(hp)
        if not isinstance(info, Mapping):
            raise RuntimeUnavailableError("Player HP write returned malformed info")
        return {"info": dict(info)}, None

    def _set_boss_hp(
        self,
        payload: Mapping[str, Any],
        *,
        event_sink: EventSink | None = None,
    ) -> tuple[dict[str, Any], None]:
        boss_number = payload.get("boss_number")
        hp = payload.get("hp")
        if isinstance(boss_number, bool) or not isinstance(boss_number, int):
            raise ProtocolError("boss_number must be an integer")
        if isinstance(hp, bool) or not isinstance(hp, int):
            raise ProtocolError("boss hp must be an integer")
        with self._environment_call(payload, event_sink) as environment:
            writer = getattr(environment.unwrapped, "set_boss_hp", None)
            if not callable(writer):
                raise RuntimeUnavailableError("Environment does not support boss HP writes")
            info = writer(boss_number, hp)
        if not isinstance(info, Mapping):
            raise RuntimeUnavailableError("Boss HP write returned malformed info")
        return {"info": dict(info)}, None

    def _render(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], np.ndarray | None]:
        environment, lock = self._acquire_environment(payload)
        try:
            frame = environment.render()
        finally:
            lock.release()
        if frame is None:
            return {"rendered": False}, None
        frame_array = np.asarray(frame)
        if frame_array.ndim != 3 or frame_array.shape[-1] != 3 or frame_array.dtype != np.uint8:
            raise RuntimeUnavailableError("Environment render must return an HxWx3 uint8 RGB frame")
        return {"rendered": True}, frame_array

    def _close(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], None]:
        with self._lifecycle_lock:
            return self._close_locked(payload)

    def _close_locked(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], None]:
        env_id = _required_text(payload.get("env_id"), "payload.env_id")
        with self._lock:
            environment = self._environments.pop(env_id, None)
            lock = self._environment_locks.pop(env_id, None)
            instance = self._environment_instances.pop(env_id, None)
            if instance is not None:
                self._reserved_instances.add(instance)
        if environment is not None:
            if lock is None or instance is None:
                with suppress(Exception):
                    environment.close()
                if instance is not None:
                    with self._lock:
                        self._reserved_instances.discard(instance)
                raise RuntimeUnavailableError("Environment lifecycle state is inconsistent")
            try:
                with lock:
                    environment.close()
            finally:
                with self._lock:
                    self._reserved_instances.discard(instance)
        return {"closed": environment is not None}, None

    def dispatch(
        self,
        operation: str,
        payload: Mapping[str, Any],
        *,
        event_sink: EventSink | None = None,
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        """Dispatch only the fixed operation whitelist."""

        if operation == "ping":
            return (
                {
                    "status": "ready",
                    "protocol": PROTOCOL,
                    "server_pid": os.getpid(),
                    "dsle_version": dsle.__version__,
                    "uptime_seconds": max(0.0, time.time() - self._started_at),
                },
                None,
            )
        if operation == "create":
            return self._create(payload)
        if operation == "reset":
            return self._reset(payload, event_sink=event_sink)
        if operation == "step":
            return self._step(payload, event_sink=event_sink)
        if operation == "observe":
            return self._observe(payload, event_sink=event_sink)
        if operation == "return_to_menu":
            return self._return_to_menu(payload, event_sink=event_sink)
        if operation == "set_player_hp":
            return self._set_player_hp(payload, event_sink=event_sink)
        if operation == "set_boss_hp":
            return self._set_boss_hp(payload, event_sink=event_sink)
        if operation == "render":
            return self._render(payload)
        if operation == "close":
            return self._close(payload)
        if operation == "close_all":
            self.close_all()
            return {"closed": True}, None
        if operation == "start_instance":
            name = _required_text(payload.get("name"), "payload.name")
            mode = _required_text(payload.get("mode", "headless"), "payload.mode")
            return {"status": self.start_instance(name, mode)}, None
        if operation == "start_instances":
            count = payload.get("count")
            if isinstance(count, bool) or not isinstance(count, int):
                raise ProtocolError("payload.count must be an integer")
            mode = _required_text(payload.get("mode", "headless"), "payload.mode")
            return {"statuses": self.start_instance_pool(count, mode)}, None
        if operation == "shutdown":
            if self.shutdown_callback is not None:
                self.shutdown_callback()
            return {"shutting_down": True}, None
        raise ProtocolError(f"Unknown RPC operation: {operation!r}")

    def close_all(self) -> None:
        with self._lifecycle_lock:
            with self._lock:
                environments = [
                    (
                        env_id,
                        environment,
                        self._environment_locks[env_id],
                        self._environment_instances[env_id],
                    )
                    for env_id, environment in self._environments.items()
                ]
                closing_instances = {instance for _id, _env, _lock, instance in environments}
                self._reserved_instances.update(closing_instances)
                self._environments.clear()
                self._environment_locks.clear()
                self._environment_instances.clear()
                started = tuple(sorted(self._started_instances))
                self._started_instances.clear()
            failures: list[str] = []
            for _env_id, environment, lock, _instance in environments:
                lock.acquire()
                try:
                    environment.close()
                except Exception as exc:
                    failures.append(f"environment close failed: {type(exc).__name__}: {exc}")
                finally:
                    lock.release()
            with self._lock:
                self._reserved_instances.difference_update(closing_instances)
            if started:
                if not self.instance_config.is_file():
                    with self._lock:
                        self._started_instances.update(started)
                    failures.append(
                        f"instance config disappeared before stopping: {self.instance_config}"
                    )
                else:
                    try:
                        statuses = stop_instances(config_path=self.instance_config, names=started)
                    except Exception as exc:
                        with self._lock:
                            self._started_instances.update(started)
                        failures.append(f"instance stop failed: {type(exc).__name__}: {exc}")
                    else:
                        still_running = (set(started) - set(statuses)) | {
                            name
                            for name, status in statuses.items()
                            if status.get("running") is True
                        }
                        if still_running:
                            with self._lock:
                                self._started_instances.update(still_running)
                            failures.append(
                                "instances remained running: " + ", ".join(sorted(still_running))
                            )
            if failures:
                raise RuntimeUnavailableError("; ".join(failures))


class _ThreadedUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    request_queue_size = MAX_CONNECTIONS

    def __init__(self, socket_path: Path, token: str, state: EnvironmentState):
        self.auth_token = token
        self.environment_state = state
        self._connection_slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        super().__init__(os.fspath(socket_path), _RequestHandler, bind_and_activate=True)

    def process_request(
        self,
        request: socket.socket | tuple[bytes, socket.socket],
        client_address: object,
    ) -> None:
        if not self._connection_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._connection_slots.release()
            raise

    def process_request_thread(
        self,
        request: socket.socket | tuple[bytes, socket.socket],
        client_address: object,
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._connection_slots.release()


class _RequestHandler(socketserver.BaseRequestHandler):
    request: socket.socket

    def handle(self) -> None:
        server = self.server
        if not isinstance(server, _ThreadedUnixServer):
            raise RuntimeError("DSLE request handler received an incompatible server")
        authenticated = False
        while True:
            try:
                request, request_array = receive_frame(
                    self.request,
                    timeout=REQUEST_TIMEOUT if authenticated else PREAUTH_REQUEST_TIMEOUT,
                )
            except EOFError:
                return
            except Exception as exc:
                self._send_error(None, exc)
                return

            request_id = request.get("id")
            if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 1:
                self._send_error(None, ProtocolError("RPC request id must be a positive integer"))
                return
            if request.get("protocol") != PROTOCOL:
                self._send_error(
                    request_id, ProtocolError("RPC protocol is missing or incompatible")
                )
                return
            if not tokens_equal(request.get("token"), server.auth_token):
                self._send_error(request_id, PermissionError("RPC authentication failed"))
                return
            authenticated = True
            try:
                if request_array is not None:
                    raise ProtocolError("RPC requests cannot contain ndarray payloads")
                operation = _required_text(request.get("op"), "op")
                payload = _mapping(request.get("payload", {}), "payload")

                def forward_event(
                    label: str,
                    message: str,
                    *,
                    current_request_id: int = request_id,
                ) -> None:
                    send_frame(
                        self.request,
                        {
                            "protocol": PROTOCOL,
                            "id": current_request_id,
                            "progress": {"label": label, "message": message},
                        },
                        timeout=REQUEST_TIMEOUT,
                    )

                result, response_array = server.environment_state.dispatch(
                    operation,
                    payload,
                    event_sink=forward_event,
                )
                send_frame(
                    self.request,
                    {
                        "protocol": PROTOCOL,
                        "id": request_id,
                        "ok": True,
                        "result": result,
                    },
                    response_array,
                    timeout=REQUEST_TIMEOUT,
                )
            except Exception as exc:
                self._send_error(request_id, exc)

    def _send_error(self, request_id: int | None, error: Exception) -> None:
        with suppress(Exception):
            send_frame(
                self.request,
                {
                    "protocol": PROTOCOL,
                    "id": request_id,
                    "ok": False,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                },
                timeout=10.0,
            )


def _prepare_socket_path(path: Path) -> None:
    """Remove only a proven-stale socket; never replace a live or non-socket path."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISSOCK(metadata.st_mode):
        raise RuntimeUnavailableError(f"Refusing to replace non-socket runtime path: {path}")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.5)
    try:
        probe.connect(os.fspath(path))
    except OSError as exc:
        if exc.errno != errno.ECONNREFUSED:
            raise RuntimeUnavailableError(
                f"Could not safely classify existing runtime socket {path}: {exc}"
            ) from exc
    else:
        raise RuntimeUnavailableError(f"A runtime server is already listening on {path}")
    finally:
        probe.close()
    path.unlink()


def serve(
    socket_path: str | Path | None = None,
    *,
    token_file: str | Path | None = None,
    state_dir: str | Path | None = None,
    env_factory: Callable[..., gym.Env[Any, Any]] = _local_environment,
    allow_instance_start: bool = True,
) -> None:
    """Serve authenticated RPC requests until SIGTERM, SIGINT, or shutdown."""

    root = Path(state_dir or os.environ.get("DSLE_STATE_DIR", "/var/lib/dsle")).resolve(
        strict=False
    )
    selected_socket = _validate_under(
        Path(socket_path or default_socket_path()), root, "DSLE runtime socket"
    )
    selected_token = _validate_under(
        Path(token_file or default_token_path()), root, "DSLE runtime token file"
    )
    token = read_auth_token(selected_token)
    selected_socket.parent.mkdir(parents=True, exist_ok=True)
    selected_socket.parent.chmod(0o700)
    _prepare_socket_path(selected_socket)

    state = EnvironmentState(
        state_dir=root,
        env_factory=env_factory,
        allow_instance_start=allow_instance_start,
    )
    server = _ThreadedUnixServer(selected_socket, token, state)
    try:
        socket_uid = int(os.environ.get("DSLE_HOST_UID", str(os.getuid())))
        socket_gid = int(os.environ.get("DSLE_HOST_GID", str(os.getgid())))
    except ValueError as exc:
        server.server_close()
        raise ConfigurationError("DSLE_HOST_UID and DSLE_HOST_GID must be integers") from exc
    if socket_uid < 0 or socket_gid < 0:
        server.server_close()
        raise ConfigurationError("DSLE_HOST_UID and DSLE_HOST_GID cannot be negative")
    try:
        os.chown(selected_socket, socket_uid, socket_gid)
    except PermissionError as exc:
        server.server_close()
        raise RuntimeUnavailableError(
            f"Could not assign runtime socket to {socket_uid}:{socket_gid}: {exc}"
        ) from exc
    os.chmod(selected_socket, 0o600)

    def request_shutdown() -> None:
        threading.Thread(target=server.shutdown, name="dsle-rpc-shutdown", daemon=True).start()

    state.shutdown_callback = request_shutdown
    previous_handlers: dict[signal.Signals, Any] = {}
    if threading.current_thread() is threading.main_thread():
        for signal_number in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, lambda *_args: request_shutdown())
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
        with suppress(Exception):
            state.close_all()
        try:
            from dsle.runtime.output import finalize_output

            finalize_output(
                root,
                int(os.environ.get("DSLE_HOST_UID", str(os.getuid()))),
                int(os.environ.get("DSLE_HOST_GID", str(os.getgid()))),
            )
        except Exception as exc:
            print(
                f"DSLE server warning: could not finalize output ownership: {exc}",
                file=sys.stderr,
            )
        try:
            metadata = selected_socket.lstat()
            if stat.S_ISSOCK(metadata.st_mode):
                selected_socket.unlink()
        except FileNotFoundError:
            pass
        for signal_number, handler in previous_handlers.items():
            signal.signal(signal_number, handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", type=Path, default=default_socket_path())
    parser.add_argument("--token-file", type=Path, default=default_token_path())
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path(os.environ.get("DSLE_STATE_DIR", "/var/lib/dsle")),
    )
    parser.add_argument("--ping", action="store_true", help="Probe a running authenticated server")
    parser.add_argument("--timeout", type=float, default=5.0, help="Ping timeout in seconds")
    parser.add_argument(
        "--no-instance-start",
        action="store_true",
        help="Disable instance lifecycle operations",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(arguments)
    if args.ping:
        try:
            result = ping(
                args.socket,
                token_file=args.token_file,
                timeout=args.timeout,
            )
        except Exception as exc:
            print(f"DSLE server is unavailable: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(result, sort_keys=True))
        return 0
    try:
        serve(
            args.socket,
            token_file=args.token_file,
            state_dir=args.state_dir,
            allow_instance_start=not args.no_instance_start,
        )
    except Exception as exc:
        print(f"DSLE server failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by container health checks
    raise SystemExit(main())
