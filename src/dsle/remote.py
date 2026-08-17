"""Authenticated Unix-socket transport and Gymnasium proxy for DSLE.

The protocol intentionally has no code-serialization feature.  Each frame is
an eight-byte network-order JSON length, a UTF-8 JSON object, and optionally
one contiguous ndarray as raw bytes described by the JSON header.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import socket
import stat
import struct
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from numbers import Real
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dsle._version import __version__
from dsle.exceptions import (
    AssetError,
    ConfigurationError,
    ResetError,
    RuntimeCommunicationError,
    RuntimeUnavailableError,
)
from dsle.progress import LifecycleLogger

PROTOCOL = "dsle-rpc-v1"
DEFAULT_SOCKET_PATH = Path("/var/lib/dsle/run/dsle.sock")
DEFAULT_TOKEN_PATH = Path("/var/lib/dsle/run/dsle.token")
MAX_JSON_BYTES = 1 << 20
MAX_JSON_DEPTH = 100
MAX_ARRAY_BYTES = 64 << 20
MAX_TOKEN_BYTES = 4096
_LENGTH = struct.Struct("!Q")


class ProtocolError(RuntimeCommunicationError):
    """A peer sent a malformed or incompatible RPC frame."""


class RemoteError(RuntimeCommunicationError):
    """An authenticated DSLE server rejected or failed an operation."""

    def __init__(self, remote_type: str, message: str):
        self.remote_type = remote_type
        self.remote_message = message
        super().__init__(f"Remote {remote_type}: {message}")


_REMOTE_EXCEPTION_TYPES: dict[str, type[Exception]] = {
    "AssetError": AssetError,
    "ConfigurationError": ConfigurationError,
    "ResetError": ResetError,
    "RuntimeCommunicationError": RuntimeCommunicationError,
    "RuntimeUnavailableError": RuntimeUnavailableError,
    "TimeoutError": TimeoutError,
    "ValueError": ValueError,
}


def _raise_remote_error(remote_type: str, message: str) -> None:
    """Restore stable public errors while wrapping unknown server failures."""

    exception_type = _REMOTE_EXCEPTION_TYPES.get(remote_type)
    if exception_type is not None:
        raise exception_type(message)
    raise RemoteError(remote_type, message)


def default_socket_path() -> Path:
    return Path(os.environ.get("DSLE_RUNTIME_SOCKET", DEFAULT_SOCKET_PATH)).expanduser()


def default_token_path() -> Path:
    return Path(os.environ.get("DSLE_RUNTIME_TOKEN_FILE", DEFAULT_TOKEN_PATH)).expanduser()


def read_auth_token(path: str | Path) -> str:
    """Read an ASCII bearer token from a private, non-symlink regular file."""

    source = Path(path).expanduser()
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise RuntimeCommunicationError(
            f"RPC token must be a safely readable non-symlink file ({source}): {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeCommunicationError(
                f"RPC token file must be one non-symlink regular file: {source}"
            )
        # The container server normally runs as root while reading a token
        # created by the unprivileged host user through a private bind mount.
        if os.geteuid() != 0 and metadata.st_uid != os.geteuid():
            raise RuntimeCommunicationError(f"RPC token file must be owned by this user: {source}")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise RuntimeCommunicationError(
                f"RPC token file must not be accessible by group/other: {source}"
            )
        if metadata.st_size > MAX_TOKEN_BYTES:
            raise RuntimeCommunicationError(
                f"RPC token file exceeds the {MAX_TOKEN_BYTES}-byte limit: {source}"
            )
        encoded = os.read(descriptor, MAX_TOKEN_BYTES + 1)
    except OSError as exc:
        raise RuntimeCommunicationError(f"Could not read RPC token file {source}: {exc}") from exc
    finally:
        os.close(descriptor)
    try:
        token = encoded.strip().decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeCommunicationError(
            f"Could not read ASCII RPC token from {source}: {exc}"
        ) from exc
    if len(token) < 32:
        raise RuntimeCommunicationError(
            "RPC authentication token must contain at least 32 characters"
        )
    return token


def tokens_equal(provided: object, expected: str) -> bool:
    """Compare request authentication without leaking a useful timing signal."""

    if not isinstance(provided, str):
        return False
    try:
        return hmac.compare_digest(provided, expected)
    except TypeError:
        # ``compare_digest`` rejects non-ASCII ``str`` input. Authentication
        # failures must remain ordinary rejections, not handler exceptions.
        return False


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _validate_json_depth(encoded: bytes) -> None:
    """Reject documents whose container nesting exceeds the protocol limit."""

    depth = 0
    in_string = False
    escaped = False
    for byte in encoded:
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
        elif byte == ord('"'):
            in_string = True
        elif byte in (ord("["), ord("{")):
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise ValueError(f"JSON nesting depth exceeds {MAX_JSON_DEPTH}")
        elif byte in (ord("]"), ord("}")):
            depth -= 1


def _positive_timeout(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive number")
    selected = float(value)
    if not math.isfinite(selected) or selected <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return selected


def _deadline(timeout: float) -> float:
    return time.monotonic() + _positive_timeout(timeout, "timeout")


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("RPC operation timed out")
    return remaining


def _recv_exact(connection: socket.socket, size: int, deadline: float) -> bytes:
    chunks: list[bytes] = []
    received = 0
    while received < size:
        connection.settimeout(_remaining(deadline))
        chunk = connection.recv(size - received)
        if not chunk:
            if received == 0:
                raise EOFError("RPC peer closed the connection")
            raise ProtocolError("RPC peer closed in the middle of a frame")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def send_frame(
    connection: socket.socket,
    document: Mapping[str, Any],
    array: np.ndarray | None = None,
    *,
    timeout: float,
) -> None:
    """Send one bounded JSON/raw-array frame."""

    selected_timeout = _positive_timeout(timeout, "timeout")
    if not isinstance(document, Mapping):
        raise ProtocolError("RPC document must be a mapping")
    header = dict(document)
    if "array" in header:
        raise ProtocolError("'array' is reserved protocol metadata")
    payload = b""
    if array is not None:
        contiguous = np.ascontiguousarray(array)
        if contiguous.dtype.hasobject or contiguous.dtype.kind not in "biuf":
            raise ProtocolError(f"Unsupported ndarray dtype: {contiguous.dtype}")
        if contiguous.nbytes > MAX_ARRAY_BYTES:
            raise ProtocolError(
                f"ndarray payload is {contiguous.nbytes} bytes; limit is {MAX_ARRAY_BYTES}"
            )
        header["array"] = {
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "nbytes": contiguous.nbytes,
        }
        payload = contiguous.tobytes(order="C")
    try:
        encoded = json.dumps(
            header,
            default=_json_default,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        _validate_json_depth(encoded)
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ProtocolError(f"RPC document is not bounded JSON data: {exc}") from exc
    if not encoded or len(encoded) > MAX_JSON_BYTES:
        raise ProtocolError(f"JSON header length must be in [1, {MAX_JSON_BYTES}]")
    connection.settimeout(selected_timeout)
    connection.sendall(_LENGTH.pack(len(encoded)) + encoded + payload)


def receive_frame(
    connection: socket.socket,
    *,
    timeout: float,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Receive and validate one bounded JSON/raw-array frame."""

    deadline = _deadline(timeout)
    header_length = _LENGTH.unpack(_recv_exact(connection, _LENGTH.size, deadline))[0]
    if not 0 < header_length <= MAX_JSON_BYTES:
        raise ProtocolError(f"JSON header length {header_length} is outside [1, {MAX_JSON_BYTES}]")
    encoded = _recv_exact(connection, header_length, deadline)
    try:
        _validate_json_depth(encoded)
        document = json.loads(encoded, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ProtocolError(f"Invalid JSON header: {exc}") from exc
    if not isinstance(document, dict):
        raise ProtocolError("RPC JSON header must be an object")

    metadata = document.pop("array", None)
    if metadata is None:
        return document, None
    if not isinstance(metadata, dict):
        raise ProtocolError("RPC array metadata must be an object")
    dtype_text = metadata.get("dtype")
    shape_value = metadata.get("shape")
    nbytes = metadata.get("nbytes")
    if not isinstance(dtype_text, str):
        raise ProtocolError("RPC array dtype must be a string")
    try:
        dtype = np.dtype(dtype_text)
    except TypeError as exc:
        raise ProtocolError(f"Invalid RPC array dtype: {dtype_text!r}") from exc
    if dtype.hasobject or dtype.kind not in "biuf" or dtype.itemsize > 8:
        raise ProtocolError(f"Unsupported RPC array dtype: {dtype}")
    if (
        not isinstance(shape_value, list)
        or len(shape_value) > 4
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape_value
        )
    ):
        raise ProtocolError("RPC array shape must contain up to four non-negative integers")
    shape = tuple(shape_value)
    expected_nbytes = math.prod(shape) * dtype.itemsize
    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes != expected_nbytes:
        raise ProtocolError("RPC array byte count does not match its dtype and shape")
    if nbytes > MAX_ARRAY_BYTES:
        raise ProtocolError(f"RPC array length {nbytes} exceeds {MAX_ARRAY_BYTES}")
    payload = _recv_exact(connection, nbytes, deadline)
    return document, np.frombuffer(payload, dtype=dtype).reshape(shape).copy()


class RPCClient:
    """One authenticated, persistent connection to a DSLE runtime server."""

    def __init__(
        self,
        socket_path: str | Path,
        token: str,
        *,
        connect_timeout: float = 10.0,
        request_timeout: float = 180.0,
    ):
        if not isinstance(token, str) or len(token) < 32:
            raise ValueError("RPC authentication token must contain at least 32 characters")
        try:
            token_bytes = token.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("RPC authentication token must contain only ASCII") from exc
        if len(token_bytes) > MAX_TOKEN_BYTES:
            raise ValueError(f"RPC authentication token cannot exceed {MAX_TOKEN_BYTES} bytes")
        self.socket_path = Path(socket_path).expanduser()
        if len(os.fsencode(self.socket_path)) >= 104:
            raise ValueError(
                f"Unix socket path is too long ({self.socket_path}); choose a shorter state directory"
            )
        self.token = token
        self.connect_timeout = _positive_timeout(connect_timeout, "connect_timeout")
        self.request_timeout = _positive_timeout(request_timeout, "request_timeout")
        self._connection: socket.socket | None = None
        self._request_id = 0
        self._lock = threading.Lock()

    def connect(self) -> None:
        if self._connection is not None:
            return
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.connect_timeout)
        try:
            connection.connect(os.fspath(self.socket_path))
        except (OSError, TimeoutError) as exc:
            connection.close()
            raise RuntimeCommunicationError(
                f"Could not connect to DSLE runtime socket {self.socket_path}: {exc}"
            ) from exc
        self._connection = connection

    def request(
        self,
        operation: str,
        payload: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
        event_handler: Callable[[str, str], None] | None = None,
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        if not isinstance(operation, str) or not operation:
            raise ValueError("operation must be a non-empty string")
        if payload is not None and not isinstance(payload, Mapping):
            raise ValueError("payload must be a mapping or None")
        if event_handler is not None and not callable(event_handler):
            raise TypeError("event_handler must be callable or None")
        request_timeout = (
            self.request_timeout if timeout is None else _positive_timeout(timeout, "timeout")
        )
        with self._lock:
            self.connect()
            connection = self._connection
            if connection is None:
                raise RuntimeCommunicationError("RPC connection was not established")
            self._request_id += 1
            request_id = self._request_id
            request = {
                "protocol": PROTOCOL,
                "id": request_id,
                "token": self.token,
                "op": operation,
                "payload": dict(payload or {}),
            }
            try:
                response_deadline = _deadline(request_timeout)
                send_frame(connection, request, timeout=_remaining(response_deadline))
                while True:
                    response, array = receive_frame(
                        connection,
                        timeout=_remaining(response_deadline),
                    )
                    if response.get("protocol") != PROTOCOL:
                        raise ProtocolError("RPC response protocol is missing or incompatible")
                    if response.get("id") != request_id:
                        raise ProtocolError("RPC response id does not match the request")
                    progress = response.get("progress")
                    if progress is None:
                        break
                    if array is not None or not isinstance(progress, Mapping):
                        raise ProtocolError("RPC progress event is malformed")
                    label = progress.get("label")
                    message = progress.get("message")
                    if (
                        not isinstance(label, str)
                        or not 1 <= len(label) <= 32
                        or not isinstance(message, str)
                        or len(message) > 8192
                    ):
                        raise ProtocolError("RPC progress event fields are malformed")
                    if event_handler is not None:
                        event_handler(label, message)
            except (OSError, EOFError, TimeoutError, ProtocolError) as exc:
                self.close()
                if isinstance(exc, RuntimeCommunicationError):
                    raise
                raise RuntimeCommunicationError(
                    f"RPC {operation!r} failed through {self.socket_path}: {exc}"
                ) from exc
            try:
                if response.get("protocol") != PROTOCOL:
                    raise ProtocolError("RPC response protocol is missing or incompatible")
                if response.get("id") != request_id:
                    raise ProtocolError("RPC response id does not match the request")
                if response.get("ok") is not True:
                    error = response.get("error")
                    if not isinstance(error, dict):
                        raise ProtocolError("RPC error response is malformed")
                    _raise_remote_error(
                        str(error.get("type", "Error")),
                        str(error.get("message", "")),
                    )
                result = response.get("result", {})
                if not isinstance(result, dict):
                    raise ProtocolError("RPC result must be an object")
            except ProtocolError:
                # A malformed response makes stream alignment and peer identity
                # untrustworthy.  Force the next request to establish a new socket.
                self.close()
                raise
            return result, array

    def close(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            with suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)
            connection.close()


def ping(
    socket_path: str | Path | None = None,
    *,
    token: str | None = None,
    token_file: str | Path | None = None,
    timeout: float = 2.0,
) -> dict[str, Any]:
    """Authenticate and return server identity, or raise on stale/unready sockets."""

    selected_socket = default_socket_path() if socket_path is None else Path(socket_path)
    if token is None:
        token = read_auth_token(default_token_path() if token_file is None else token_file)
    client = RPCClient(
        selected_socket,
        token,
        connect_timeout=timeout,
        request_timeout=timeout,
    )
    try:
        result, array = client.request("ping", timeout=timeout)
        if array is not None:
            raise ProtocolError("Ping unexpectedly returned an ndarray")
        if result.get("status") != "ready":
            raise ProtocolError("DSLE server did not report ready status")
        server_version = result.get("dsle_version")
        if server_version != __version__:
            raise ProtocolError(
                "DSLE host/runtime version mismatch: "
                f"host={__version__!r}, runtime={server_version!r}"
            )
        return result
    finally:
        client.close()


def _space_from_description(description: object) -> gym.Space[Any]:
    if not isinstance(description, Mapping):
        raise ProtocolError("Remote Gymnasium space description must be an object")
    kind = description.get("kind")
    if kind == "discrete":
        size = description.get("n")
        start = description.get("start", 0)
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or isinstance(start, bool)
            or not isinstance(start, int)
        ):
            raise ProtocolError("Remote Discrete space bounds are malformed")
        return spaces.Discrete(size, start=start)
    if kind == "box":
        shape_value = description.get("shape")
        if (
            not isinstance(shape_value, list)
            or len(shape_value) > 4
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in shape_value
            )
        ):
            raise ProtocolError("Remote Box shape is malformed")
        try:
            dtype = np.dtype(description.get("dtype"))
        except TypeError as exc:
            raise ProtocolError("Remote Box dtype is malformed") from exc
        if dtype.hasobject or dtype.kind not in "biuf" or dtype.itemsize > 8:
            raise ProtocolError(f"Remote Box dtype is unsupported: {dtype}")
        low = description.get("low")
        high = description.get("high")
        if (
            isinstance(low, bool)
            or not isinstance(low, (int, float))
            or not math.isfinite(float(low))
            or isinstance(high, bool)
            or not isinstance(high, (int, float))
            or not math.isfinite(float(high))
            or float(low) > float(high)
        ):
            raise ProtocolError("Remote Box bounds are malformed")
        return spaces.Box(
            low=low,
            high=high,
            shape=tuple(shape_value),
            dtype=cast(Any, dtype),
        )
    raise ProtocolError(f"Unsupported remote Gymnasium space kind: {kind!r}")


class RemoteDarkSoulsEnv(gym.Env[np.ndarray, int]):
    """Gymnasium proxy whose authoritative environment lives in the container."""

    def __init__(
        self,
        socket_path: str | Path,
        boss: str,
        *,
        token: str | None = None,
        token_file: str | Path | None = None,
        connect_timeout: float = 10.0,
        request_timeout: float = 180.0,
        start_instance: bool = False,
        instance_mode: str = "headless",
        **env_kwargs: Any,
    ):
        super().__init__()
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 4}
        if not isinstance(boss, str) or not boss.strip():
            raise ValueError("boss must be a non-empty string")
        if not isinstance(start_instance, bool):
            raise ValueError("start_instance must be a boolean")
        verbose = env_kwargs.get("verbose", True)
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean")
        self._lifecycle = LifecycleLogger(verbose)
        if not isinstance(instance_mode, str) or instance_mode not in {
            "gui",
            "headless",
            "headless-vnc",
        }:
            raise ValueError("instance_mode must be 'gui', 'headless', or 'headless-vnc'")
        if token is None:
            token = read_auth_token(default_token_path() if token_file is None else token_file)
        self._client = RPCClient(
            socket_path,
            token,
            connect_timeout=connect_timeout,
            request_timeout=request_timeout,
        )
        self._env_id: str | None = None
        self.render_mode = env_kwargs.get("render_mode")
        try:
            identity, identity_array = self._client.request("ping")
            if identity_array is not None:
                raise ProtocolError("Ping unexpectedly returned an ndarray")
            if identity.get("status") != "ready" or identity.get("dsle_version") != __version__:
                raise ProtocolError(
                    "DSLE host/runtime version mismatch or unready server: "
                    f"host={__version__!r}, runtime={identity.get('dsle_version')!r}"
                )
            result, array = self._client.request(
                "create",
                {
                    "boss": boss,
                    "kwargs": env_kwargs,
                    "start_instance": start_instance,
                    "instance_mode": instance_mode,
                },
            )
            if array is not None:
                raise ProtocolError("Create unexpectedly returned an ndarray")
            env_id = result.get("env_id")
            if not isinstance(env_id, str) or not env_id:
                raise ProtocolError("Create response omitted env_id")
            self._env_id = env_id
            self.action_space = _space_from_description(result.get("action_space"))
            self.observation_space = _space_from_description(result.get("observation_space"))
            remote_metadata = result.get("metadata")
            if isinstance(remote_metadata, dict):
                self.metadata = remote_metadata
        except Exception:
            self._client.close()
            raise

    def _request(
        self, operation: str, payload: Mapping[str, Any] | None = None
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        if self._env_id is None:
            raise RuntimeError("Remote environment is closed")
        full_payload = {"env_id": self._env_id, **dict(payload or {})}
        lifecycle = getattr(self, "_lifecycle", None)
        if lifecycle is None:
            return self._client.request(operation, full_payload)
        return self._client.request(
            operation,
            full_payload,
            event_handler=lifecycle.event,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options is not None and not isinstance(options, Mapping):
            raise ValueError("reset options must be a mapping or None")
        result, observation = self._request("reset", {"seed": seed, "options": options})
        if observation is None:
            raise ProtocolError("Reset response omitted its observation")
        if not self.observation_space.contains(observation):
            raise ProtocolError(f"Reset returned an observation outside {self.observation_space}")
        info = result.get("info")
        if not isinstance(info, dict):
            raise ProtocolError("Reset response info is malformed")
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise ValueError(f"Action must be an integer inside {self.action_space}")
        selected_action = int(action)
        if not self.action_space.contains(selected_action):
            raise ValueError(f"Action is outside {self.action_space}")
        result, observation = self._request("step", {"action": selected_action})
        if observation is None:
            raise ProtocolError("Step response omitted its observation")
        if not self.observation_space.contains(observation):
            raise ProtocolError(f"Step returned an observation outside {self.observation_space}")
        info = result.get("info")
        if not isinstance(info, dict):
            raise ProtocolError("Step response info is malformed")
        reward = result.get("reward")
        terminated = result.get("terminated")
        truncated = result.get("truncated")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
        ):
            raise ProtocolError("Step response reward must be a finite number")
        if not isinstance(terminated, bool) or not isinstance(truncated, bool):
            raise ProtocolError("Step response terminal flags must be booleans")
        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    def observe(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Passively refresh the remote game state without injecting an action."""

        result, observation = self._request("observe")
        if observation is None:
            raise ProtocolError("Observe response omitted its observation")
        if not self.observation_space.contains(observation):
            raise ProtocolError(f"Observe returned an observation outside {self.observation_space}")
        info = result.get("info")
        if not isinstance(info, dict):
            raise ProtocolError("Observe response info is malformed")
        return observation, info

    def return_to_menu(self, *, timeout_s: float = 30.0) -> None:
        """Return the remote game to a verified title menu."""

        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ValueError("timeout_s must be a finite positive number")
        selected_timeout = float(timeout_s)
        if not math.isfinite(selected_timeout) or selected_timeout <= 0:
            raise ValueError("timeout_s must be a finite positive number")
        result, array = self._request("return_to_menu", {"timeout_s": selected_timeout})
        if array is not None or result.get("at_menu") is not True:
            raise ProtocolError("Return-to-menu response is malformed")

    @staticmethod
    def _nonnegative_integer(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        selected = int(value)
        if selected < 0:
            raise ValueError(f"{name} must be non-negative")
        return selected

    def set_player_hp(self, hp: int) -> dict[str, Any]:
        """Set current player HP in the active remote episode."""

        selected = self._nonnegative_integer(hp, "player HP")
        result, array = self._request("set_player_hp", {"hp": selected})
        if array is not None:
            raise ProtocolError("Player HP write unexpectedly returned an ndarray")
        info = result.get("info")
        if not isinstance(info, dict):
            raise ProtocolError("Player HP write response info is malformed")
        return info

    def set_boss_hp(self, boss_number: int, hp: int) -> dict[str, Any]:
        """Set one-based ``bossN`` HP, or use ``(-1, 0)`` to zero every boss HP."""

        if isinstance(boss_number, bool) or not isinstance(boss_number, (int, np.integer)):
            raise ValueError("boss_number must be an integer")
        selected_number = int(boss_number)
        selected_hp = self._nonnegative_integer(hp, "boss HP")
        if selected_number == -1 and selected_hp != 0:
            raise ValueError("boss_number=-1 is only valid with hp=0")
        if selected_number != -1 and selected_number < 1:
            raise ValueError("boss_number must be at least 1, or -1 with hp=0")
        result, array = self._request(
            "set_boss_hp",
            {"boss_number": selected_number, "hp": selected_hp},
        )
        if array is not None:
            raise ProtocolError("Boss HP write unexpectedly returned an ndarray")
        info = result.get("info")
        if not isinstance(info, dict):
            raise ProtocolError("Boss HP write response info is malformed")
        return info

    def render(self) -> np.ndarray | None:
        result, frame = self._request("render")
        rendered = result.get("rendered")
        if not isinstance(rendered, bool):
            raise ProtocolError("Render response omitted its boolean rendered flag")
        if rendered is False:
            if frame is not None:
                raise ProtocolError("Render response returned a frame while declaring none")
            return None
        if frame is None:
            raise ProtocolError("Render response declared a frame but omitted it")
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ProtocolError(
                f"Render returned an invalid RGB frame: {frame.shape}, {frame.dtype}"
            )
        return frame

    def close(self) -> None:
        env_id = self._env_id
        try:
            if env_id is not None:
                result, array = self._client.request("close", {"env_id": env_id})
                if array is not None or result.get("closed") is not True:
                    raise ProtocolError("Close response is malformed")
                self._env_id = None
        finally:
            self._client.close()

    def _disconnect(self) -> None:
        """Close only the host transport when its owned server is going away."""

        self._env_id = None
        self._client.close()
