from __future__ import annotations

import json
import socket
import stat
import struct
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pytest

import dsle.remote as remote_module
import dsle.runtime.server as server_module
from dsle.exceptions import ConfigurationError, RuntimeCommunicationError, RuntimeUnavailableError
from dsle.remote import (
    PROTOCOL,
    ProtocolError,
    RemoteDarkSoulsEnv,
    RemoteError,
    RPCClient,
    _raise_remote_error,
    _space_from_description,
    ping,
    read_auth_token,
    receive_frame,
    send_frame,
    tokens_equal,
)
from dsle.runtime.instances import generate_instances, load_instances, write_instance_config
from dsle.runtime.server import EnvironmentState, _prepare_socket_path, serve


class FakeEnvironment(gym.Env[np.ndarray, int]):
    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, boss: str, **kwargs):
        self.boss = boss
        self.kwargs = kwargs
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(0, 255, (1, 2, 3), np.uint8)
        self.closed = False
        self.steps = 0
        self.observations = 0
        self.event_sink = None
        self.health_writes: list[tuple[object, ...]] = []
        self.menu_timeouts: list[float] = []

    def set_event_sink(self, event_sink):
        self.event_sink = event_sink

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        if self.event_sink is not None:
            self.event_sink("SETUP", "dsr-1 1/1 completed fake_setup")
        return np.arange(6, dtype=np.uint8).reshape(1, 2, 3), {"seed": seed, "options": options}

    def step(self, action):
        self.steps += 1
        if self.event_sink is not None:
            self.event_sink(
                "COMBAT",
                f"dsr-1 step={self.steps} player_hp=500/500 boss1_hp=900/1000 "
                "boss_defeated=no player_dead=no",
            )
        return (
            np.full((1, 2, 3), action, dtype=np.uint8),
            1.25,
            self.steps >= 2,
            False,
            {"action": action, "win": self.steps >= 2},
        )

    def observe(self):
        self.observations += 1
        return np.full((1, 2, 3), 9, dtype=np.uint8), {
            "boss_state_valid": True,
            "boss_defeated": False,
            "player_dead": False,
        }

    def return_to_menu(self, *, timeout_s=30.0):
        self.menu_timeouts.append(timeout_s)

    def render(self):
        return np.full((2, 3, 3), 7, dtype=np.uint8)

    def set_player_hp(self, hp):
        self.health_writes.append(("player", hp))
        if self.event_sink is not None:
            self.event_sink("WRITE", f"dsr-1 setting player_hp={hp}")
        return {"player_hp": hp}

    def set_boss_hp(self, boss_number, hp):
        self.health_writes.append(("boss", boss_number, hp))
        if self.event_sink is not None:
            self.event_sink("WRITE", f"dsr-1 setting boss{boss_number}_hp={hp}")
        return {"boss_number": boss_number, "boss_hp": hp}

    def close(self):
        self.closed = True


@pytest.fixture
def runtime_server(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    token_path = run_dir / "token"
    token_path.write_text("a" * 64, encoding="ascii")
    token_path.chmod(0o600)
    socket_path = run_dir / "runtime.sock"
    config_path = tmp_path / "config" / "instances.json"
    monkeypatch.setenv("DSLE_INSTANCE_CONFIG", str(config_path))
    created: list[FakeEnvironment] = []
    failures: list[BaseException] = []

    def factory(boss, **kwargs):
        environment = FakeEnvironment(boss, **kwargs)
        created.append(environment)
        return environment

    def run_server():
        try:
            serve(
                socket_path,
                token_file=token_path,
                state_dir=tmp_path,
                env_factory=factory,
                allow_instance_start=False,
            )
        except BaseException as exc:  # surfaced in fixture teardown
            failures.append(exc)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    deadline = time.monotonic() + 3.0
    while not socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert socket_path.exists(), failures

    yield socket_path, token_path, created

    if thread.is_alive():
        client = RPCClient(socket_path, "a" * 64, request_timeout=2.0)
        try:
            client.request("shutdown")
        finally:
            client.close()
    thread.join(timeout=3.0)
    assert not thread.is_alive()
    assert not failures


def test_frame_round_trip_uses_json_and_raw_ndarray() -> None:
    sender, receiver = socket.socketpair()
    try:
        expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        send_frame(sender, {"kind": "observation"}, expected, timeout=1.0)
        document, actual = receive_frame(receiver, timeout=1.0)
    finally:
        sender.close()
        receiver.close()
    assert document == {"kind": "observation"}
    assert actual is not None
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous


def test_frame_rejects_inconsistent_array_metadata() -> None:
    sender, receiver = socket.socketpair()
    header = json.dumps({"array": {"dtype": "|u1", "shape": [2, 3], "nbytes": 5}}).encode("utf-8")
    try:
        sender.sendall(struct.pack("!Q", len(header)) + header)
        with pytest.raises(ProtocolError, match="byte count"):
            receive_frame(receiver, timeout=1.0)
    finally:
        sender.close()
        receiver.close()


def test_frame_rejects_nonfinite_json_constants() -> None:
    sender, receiver = socket.socketpair()
    header = b'{"value":NaN}'
    try:
        sender.sendall(struct.pack("!Q", len(header)) + header)
        with pytest.raises(ProtocolError, match="Non-finite"):
            receive_frame(receiver, timeout=1.0)
    finally:
        sender.close()
        receiver.close()


def test_frame_rejects_excessive_json_nesting_as_a_protocol_error() -> None:
    sender, receiver = socket.socketpair()
    nested_arrays = remote_module.MAX_JSON_DEPTH
    header = b'{"nested":' + (b"[" * nested_arrays) + b"0" + (b"]" * nested_arrays) + b"}"
    try:
        sender.sendall(struct.pack("!Q", len(header)) + header)
        with pytest.raises(ProtocolError, match="nesting depth"):
            receive_frame(receiver, timeout=1.0)
    finally:
        sender.close()
        receiver.close()


def test_frame_accepts_json_at_nesting_limit_and_delimiters_in_strings() -> None:
    sender, receiver = socket.socketpair()
    nested_arrays = remote_module.MAX_JSON_DEPTH - 1
    leaf = b'"brackets in a string: [{ and escaped quote: \\""'
    header = b'{"nested":' + (b"[" * nested_arrays) + leaf + (b"]" * nested_arrays) + b"}"
    try:
        sender.sendall(struct.pack("!Q", len(header)) + header)
        document, array = receive_frame(receiver, timeout=1.0)
    finally:
        sender.close()
        receiver.close()

    value = document["nested"]
    for _ in range(nested_arrays):
        value = value[0]
    assert value == 'brackets in a string: [{ and escaped quote: "'
    assert array is None


def test_send_frame_rejects_excessive_json_nesting() -> None:
    sender, receiver = socket.socketpair()
    nested: object = 0
    for _ in range(remote_module.MAX_JSON_DEPTH):
        nested = [nested]
    try:
        with pytest.raises(ProtocolError, match="nesting depth"):
            send_frame(sender, {"nested": nested}, timeout=1.0)
    finally:
        sender.close()
        receiver.close()


@pytest.mark.parametrize("document", [{"bad": {1, 2}}, {"bad": float("nan")}, []])
def test_send_frame_normalizes_non_json_data_to_protocol_error(document) -> None:
    sender, receiver = socket.socketpair()
    try:
        with pytest.raises(ProtocolError, match=r"bounded JSON|must be a mapping"):
            send_frame(sender, document, timeout=1.0)
    finally:
        sender.close()
        receiver.close()


@pytest.mark.parametrize(
    "description",
    [
        None,
        [],
        {"kind": "discrete", "n": True},
        {"kind": "discrete", "n": 0},
        {"kind": "box", "shape": [True], "dtype": "u1", "low": 0, "high": 1},
        {"kind": "box", "shape": [1], "dtype": "O", "low": 0, "high": 1},
        {"kind": "box", "shape": [1], "dtype": "u1", "low": 2, "high": 1},
        {"kind": "box", "shape": [1], "dtype": "u1", "low": 0, "high": float("inf")},
    ],
)
def test_remote_space_description_rejects_malformed_values(description: object) -> None:
    with pytest.raises(ProtocolError):
        _space_from_description(description)


def test_token_file_must_be_private_and_not_a_symlink(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_text("a" * 64, encoding="ascii")
    token.chmod(0o644)
    with pytest.raises(RuntimeCommunicationError, match="group/other"):
        read_auth_token(token)

    token.chmod(0o600)
    link = tmp_path / "token-link"
    link.symlink_to(token)
    with pytest.raises(RuntimeCommunicationError, match="non-symlink"):
        read_auth_token(link)


def test_token_file_has_a_bounded_size(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_bytes(b"a" * (remote_module.MAX_TOKEN_BYTES + 1))
    token.chmod(0o600)
    with pytest.raises(RuntimeCommunicationError, match="byte limit"):
        read_auth_token(token)


def test_direct_rpc_tokens_are_ascii_bounded_and_nonascii_auth_is_rejected() -> None:
    with pytest.raises(ValueError, match="ASCII"):
        RPCClient("/tmp/dsle-test.sock", "é" * 32)
    with pytest.raises(ValueError, match="cannot exceed"):
        RPCClient("/tmp/dsle-test.sock", "a" * (remote_module.MAX_TOKEN_BYTES + 1))
    assert not tokens_equal("é" * 32, "a" * 64)


def test_rpc_request_rejects_non_mapping_payload_before_connect() -> None:
    client = RPCClient("/tmp/dsle-test.sock", "a" * 64)
    with pytest.raises(ValueError, match="payload must be a mapping"):
        client.request("ping", [])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "options",
    [
        {"start_instance": "false"},
        {"start_instance": 1},
        {"instance_mode": "interactive"},
        {"instance_mode": None},
    ],
)
def test_remote_environment_rejects_invalid_instance_launch_options(options) -> None:
    with pytest.raises(ValueError, match=r"start_instance|instance_mode"):
        RemoteDarkSoulsEnv(
            "/tmp/does-not-connect.sock",
            "asylum_demon",
            token="a" * 64,
            **options,
        )


@pytest.mark.parametrize("value", [0, -1, True, False, float("nan"), float("inf")])
def test_rpc_client_rejects_invalid_timeouts(value: object) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        RPCClient("/tmp/dsle-test.sock", "a" * 64, connect_timeout=value)
    with pytest.raises(ValueError, match="finite positive"):
        RPCClient("/tmp/dsle-test.sock", "a" * 64, request_timeout=value)


def test_authenticated_remote_environment_round_trip(runtime_server, capsys) -> None:
    socket_path, token_path, created = runtime_server
    assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(socket_path.parent.stat().st_mode) == 0o700
    identity = ping(socket_path, token_file=token_path, timeout=1.0)
    assert identity["status"] == "ready"
    assert identity["protocol"] == PROTOCOL

    environment = RemoteDarkSoulsEnv(
        socket_path,
        "asylum_demon",
        token_file=token_path,
        render_mode="rgb_array",
    )
    assert environment.action_space == gym.spaces.Discrete(3)
    assert environment.observation_space.shape == (1, 2, 3)

    observation, info = environment.reset(seed=17, options={"difficulty": "standard"})
    np.testing.assert_array_equal(observation, np.arange(6, dtype=np.uint8).reshape(1, 2, 3))
    assert info == {"seed": 17, "options": {"difficulty": "standard"}}

    observation, reward, terminated, truncated, info = environment.step(2)
    np.testing.assert_array_equal(observation, np.full((1, 2, 3), 2, dtype=np.uint8))
    assert (reward, terminated, truncated) == (1.25, False, False)
    assert info == {"action": 2, "win": False}
    passive, passive_info = environment.observe()
    np.testing.assert_array_equal(passive, np.full((1, 2, 3), 9, dtype=np.uint8))
    assert passive_info["player_dead"] is False
    environment.return_to_menu(timeout_s=12.5)
    assert environment.set_player_hp(400) == {"player_hp": 400}
    assert environment.set_boss_hp(2, 700) == {"boss_number": 2, "boss_hp": 700}
    assert environment.set_boss_hp(-1, 0) == {"boss_number": -1, "boss_hp": 0}
    np.testing.assert_array_equal(environment.render(), np.full((2, 3, 3), 7, dtype=np.uint8))

    environment.close()
    environment.close()
    assert created[0].closed
    assert created[0].health_writes == [
        ("player", 400),
        ("boss", 2, 700),
        ("boss", -1, 0),
    ]
    assert created[0].observations == 1
    assert created[0].menu_timeouts == [12.5]
    telemetry = capsys.readouterr().err
    assert "[SETUP] dsr-1 1/1 completed fake_setup" in telemetry
    assert "[COMBAT] dsr-1 step=1" in telemetry
    assert "[WRITE] dsr-1 setting player_hp=400" in telemetry
    assert "[WRITE] dsr-1 setting boss2_hp=700" in telemetry
    assert "[WRITE] dsr-1 setting boss-1_hp=0" in telemetry


def test_remote_health_writers_reject_invalid_values_before_rpc(runtime_server) -> None:
    socket_path, token_path, created = runtime_server
    environment = RemoteDarkSoulsEnv(socket_path, "asylum_demon", token_file=token_path)
    try:
        for value in (True, 1.5, "1", -1):
            with pytest.raises(ValueError, match=r"integer|non-negative"):
                environment.set_player_hp(value)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="at least 1"):
            environment.set_boss_hp(0, 100)
        with pytest.raises(ValueError, match="only valid with hp=0"):
            environment.set_boss_hp(-1, 100)
        with pytest.raises(ValueError, match="integer"):
            environment.set_boss_hp(True, 100)
        assert created[0].health_writes == []
    finally:
        environment.close()


def test_remote_verbose_false_suppresses_forwarded_runtime_events(runtime_server, capsys) -> None:
    socket_path, token_path, _created = runtime_server
    environment = RemoteDarkSoulsEnv(
        socket_path,
        "asylum_demon",
        token_file=token_path,
        verbose=False,
    )
    try:
        environment.reset()
        environment.step(1)
    finally:
        environment.close()

    assert capsys.readouterr().err == ""


def test_remote_step_rejects_coercible_nonintegers_before_rpc(runtime_server) -> None:
    socket_path, token_path, created = runtime_server
    environment = RemoteDarkSoulsEnv(socket_path, "asylum_demon", token_file=token_path)
    try:
        environment.reset()
        for value in (True, False, 1.0, 1.9, "1"):
            with pytest.raises(ValueError, match="integer"):
                environment.step(value)
        assert created[0].steps == 0
        environment.step(np.int64(1))
        assert created[0].steps == 1
    finally:
        environment.close()


def test_remote_reset_rejects_non_mapping_options_before_rpc(runtime_server) -> None:
    socket_path, token_path, _created = runtime_server
    environment = RemoteDarkSoulsEnv(socket_path, "asylum_demon", token_file=token_path)
    try:
        with pytest.raises(ValueError, match="mapping or None"):
            environment.reset(options=[])  # type: ignore[arg-type]
    finally:
        environment.close()


def test_remote_close_retains_id_after_transport_failure_for_retry() -> None:
    class FlakyClient:
        def __init__(self):
            self.requests = 0
            self.closes = 0

        def request(self, operation, payload):
            assert operation == "close"
            assert payload == {"env_id": "environment-id"}
            self.requests += 1
            if self.requests == 1:
                raise RuntimeCommunicationError("response was lost")
            return {"closed": self.requests >= 3}, None

        def close(self):
            self.closes += 1

    environment = RemoteDarkSoulsEnv.__new__(RemoteDarkSoulsEnv)
    environment._client = FlakyClient()
    environment._env_id = "environment-id"

    with pytest.raises(RuntimeCommunicationError, match="response was lost"):
        environment.close()
    assert environment._env_id == "environment-id"
    with pytest.raises(ProtocolError, match="Close response"):
        environment.close()
    assert environment._env_id == "environment-id"
    environment.close()
    assert environment._env_id is None
    assert environment._client.requests == 3
    assert environment._client.closes == 3


def test_remote_render_rejects_non_rgb_frames() -> None:
    class Client:
        def request(self, operation, payload):
            assert operation == "render"
            assert payload == {"env_id": "environment-id"}
            return {"rendered": True}, np.zeros((2, 3), dtype=np.uint8)

    environment = RemoteDarkSoulsEnv.__new__(RemoteDarkSoulsEnv)
    environment._client = Client()
    environment._env_id = "environment-id"
    with pytest.raises(ProtocolError, match="invalid RGB"):
        environment.render()


@pytest.mark.parametrize(
    ("result", "frame", "message"),
    [
        ({}, None, "boolean rendered"),
        ({"rendered": 1}, None, "boolean rendered"),
        ({"rendered": False}, np.zeros((2, 3, 3), dtype=np.uint8), "declaring none"),
        ({"rendered": True}, None, "omitted"),
    ],
)
def test_remote_render_rejects_inconsistent_response_metadata(result, frame, message) -> None:
    class Client:
        def request(self, operation, payload):
            return result, frame

    environment = RemoteDarkSoulsEnv.__new__(RemoteDarkSoulsEnv)
    environment._client = Client()
    environment._env_id = "environment-id"
    with pytest.raises(ProtocolError, match=message):
        environment.render()


def test_wrong_token_cannot_ping_or_invoke_operations(runtime_server) -> None:
    socket_path, _token_path, created = runtime_server
    with pytest.raises(RemoteError, match="authentication failed"):
        ping(socket_path, token="b" * 64, timeout=1.0)
    assert created == []


def test_stable_remote_errors_map_back_to_public_exception_types() -> None:
    with pytest.raises(ConfigurationError, match="bad boss config"):
        _raise_remote_error("ConfigurationError", "bad boss config")
    with pytest.raises(RemoteError, match="Remote KeyError"):
        _raise_remote_error("KeyError", "private implementation failure")


def test_ping_rejects_a_differently_versioned_runtime(monkeypatch) -> None:
    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def request(self, *_args, **_kwargs):
            return {"status": "ready", "dsle_version": "incompatible-release"}, None

        def close(self):
            pass

    monkeypatch.setattr(remote_module, "RPCClient", Client)
    with pytest.raises(ProtocolError, match="version mismatch"):
        ping("/tmp/not-used.sock", token="a" * 64)


def test_server_rejects_non_whitelisted_operation(runtime_server) -> None:
    socket_path, token_path, _created = runtime_server
    client = RPCClient(socket_path, read_auth_token(token_path), request_timeout=1.0)
    try:
        with pytest.raises(RemoteError, match="Unknown RPC operation"):
            client.request("exec")
    finally:
        client.close()


def test_malformed_response_invalidates_persistent_connection(tmp_path: Path) -> None:
    socket_path = tmp_path / "malformed.sock"
    ready = threading.Event()

    def malformed_server() -> None:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(socket_path))
            listener.listen(1)
            ready.set()
            connection, _address = listener.accept()
            with connection:
                request, _array = receive_frame(connection, timeout=1.0)
                send_frame(
                    connection,
                    {
                        "protocol": "incompatible-protocol",
                        "id": request["id"],
                        "ok": True,
                        "result": {},
                    },
                    timeout=1.0,
                )
        finally:
            listener.close()

    thread = threading.Thread(target=malformed_server, daemon=True)
    thread.start()
    assert ready.wait(timeout=1.0)
    client = RPCClient(socket_path, "a" * 64, request_timeout=1.0)
    with pytest.raises(ProtocolError, match="incompatible"):
        client.request("ping")
    assert client._connection is None
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_only_proven_stale_socket_is_replaced(tmp_path: Path) -> None:
    stale_path = tmp_path / "stale.sock"
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(stale_path))
    stale.close()
    _prepare_socket_path(stale_path)
    assert not stale_path.exists()

    live_path = tmp_path / "live.sock"
    live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    live.bind(str(live_path))
    live.listen()
    try:
        with pytest.raises(RuntimeUnavailableError, match="already listening"):
            _prepare_socket_path(live_path)
        assert live_path.exists()
    finally:
        live.close()


def test_default_server_factory_forces_local_runtime(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DSLE_INSTANCE_CONFIG", str(tmp_path / "config" / "instances.json"))
    calls = []

    def local_make(boss, **kwargs):
        calls.append((boss, kwargs))
        return FakeEnvironment(boss, **kwargs)

    monkeypatch.setattr(server_module.dsle, "make", local_make)
    state = EnvironmentState(state_dir=tmp_path, allow_instance_start=False)
    result, _array = state.dispatch(
        "create",
        {
            "boss": "asylum_demon",
            "kwargs": {"instance": "dsr-1"},
            "start_instance": False,
        },
    )
    try:
        assert calls[0][1]["runtime"] == "local"
        assert result["env_id"]
    finally:
        state.close_all()


def test_fresh_server_provisions_restart_safe_full_managed_pool(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config" / "instances.json"
    monkeypatch.setenv("DSLE_INSTANCE_CONFIG", str(config))
    state = EnvironmentState(state_dir=tmp_path, allow_instance_start=False)
    state._ensure_configured_instance("dsr-2")

    instances = load_instances(config)
    assert set(instances) == {f"dsr-{index}" for index in range(1, 31)}
    assert instances["dsr-1"].xdg_runtime_dir == (tmp_path / "xdg" / "dsr-1").resolve()
    assert instances["dsr-30"].wineprefix == (tmp_path / "wineprefixes" / "dsr-30").resolve()

    # A new server process must accept the generated file rather than reject
    # its own XDG paths or lose the rest of the managed pool.
    restarted = EnvironmentState(state_dir=tmp_path, allow_instance_start=False)
    restarted._ensure_configured_instance("dsr-30")


def test_persisted_managed_subset_expands_for_a_larger_pool_without_data_loss(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config" / "instances.json"
    monkeypatch.setenv("DSLE_INSTANCE_CONFIG", str(config))
    stale = generate_instances(
        1,
        wineprefix_root=tmp_path / "wineprefixes",
        display_start=90,
        vnc_port_start=5901,
    )
    save_dir = tmp_path / "saves" / "dsr-1"
    stale = {
        name: replace(
            instance,
            xdg_runtime_dir=tmp_path / "xdg" / name,
            save_dir=save_dir,
        )
        for name, instance in stale.items()
    }
    write_instance_config(config, stale, resolution="1280x720")
    prefix_marker = stale["dsr-1"].wineprefix / "persisted-prefix-data"
    prefix_marker.parent.mkdir(parents=True)
    prefix_marker.write_text("keep", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def start_pool(**kwargs):
        calls.append(kwargs)
        return {
            name: {"name": name, "status": "ready", "running": True} for name in kwargs["names"]
        }

    monkeypatch.setattr(server_module, "start_runtime_instances", start_pool)
    state = EnvironmentState(state_dir=tmp_path, allow_instance_start=True)

    statuses = state.start_instance_pool(2, "headless")

    assert tuple(statuses) == ("dsr-1", "dsr-2")
    assert calls[0]["names"] == ("dsr-1", "dsr-2")
    expanded = load_instances(config)
    assert set(expanded) == {f"dsr-{index}" for index in range(1, 31)}
    assert expanded["dsr-1"] == stale["dsr-1"]
    assert prefix_marker.read_text(encoding="utf-8") == "keep"
    document = json.loads(config.read_text(encoding="utf-8"))
    assert {entry.get("resolution") for entry in document["instances"].values()} == {"1280x720"}


def test_smaller_reused_pool_keeps_expanded_config_and_selects_only_requested_instances(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config" / "instances.json"
    monkeypatch.setenv("DSLE_INSTANCE_CONFIG", str(config))
    initial = EnvironmentState(state_dir=tmp_path, allow_instance_start=False)
    initial._ensure_configured_instance("dsr-3")
    original_config = config.read_bytes()
    calls: list[dict[str, Any]] = []

    def start_pool(**kwargs):
        calls.append(kwargs)
        return {
            name: {"name": name, "status": "ready", "running": True} for name in kwargs["names"]
        }

    monkeypatch.setattr(server_module, "start_runtime_instances", start_pool)
    restarted = EnvironmentState(state_dir=tmp_path, allow_instance_start=True)

    statuses = restarted.start_instance_pool(1, "headless")

    assert tuple(statuses) == ("dsr-1",)
    assert calls[0]["names"] == ("dsr-1",)
    assert config.read_bytes() == original_config
    assert len(load_instances(config)) == 30


def test_noncanonical_persisted_subset_is_never_overwritten_during_expansion(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config" / "instances.json"
    monkeypatch.setenv("DSLE_INSTANCE_CONFIG", str(config))
    custom = generate_instances(
        1,
        wineprefix_root=tmp_path / "wineprefixes",
        display_start=120,
        vnc_port_start=6201,
    )
    custom = {
        name: replace(instance, xdg_runtime_dir=tmp_path / "xdg" / name)
        for name, instance in custom.items()
    }
    write_instance_config(config, custom)
    original_config = config.read_bytes()
    state = EnvironmentState(state_dir=tmp_path, allow_instance_start=True)

    with pytest.raises(ConfigurationError, match="noncanonical managed instances: dsr-1"):
        state._ensure_configured_instance("dsr-2")

    assert config.read_bytes() == original_config
    assert load_instances(config) == custom


def test_server_starts_numbered_pool_with_one_parallel_runtime_call(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []

    def start_pool(**kwargs):
        calls.append(kwargs)
        return {
            name: {"name": name, "status": "ready", "running": True} for name in kwargs["names"]
        }

    monkeypatch.setattr(server_module, "start_runtime_instances", start_pool)
    state = EnvironmentState(state_dir=tmp_path, allow_instance_start=True)
    statuses = state.start_instance_pool(3, "headless")
    assert tuple(statuses) == ("dsr-1", "dsr-2", "dsr-3")
    assert len(calls) == 1
    assert calls[0]["names"] == ("dsr-1", "dsr-2", "dsr-3")
    assert set(state._started_instances) == {"dsr-1", "dsr-2", "dsr-3"}


def test_failed_pool_start_rolls_back_running_members_immediately(
    tmp_path: Path, monkeypatch
) -> None:
    stopped: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        server_module,
        "start_runtime_instances",
        lambda **_kwargs: {
            "dsr-1": {"name": "dsr-1", "status": "ready", "running": True},
            "dsr-2": {
                "name": "dsr-2",
                "status": "failed",
                "running": False,
                "error": "game failed",
            },
        },
    )

    def stop_pool(*, config_path, names):
        assert Path(config_path).is_file()
        stopped.append(tuple(names))
        return {name: {"name": name, "status": "stopped", "running": False} for name in names}

    monkeypatch.setattr(server_module, "stop_instances", stop_pool)
    state = EnvironmentState(state_dir=tmp_path, allow_instance_start=True)

    with pytest.raises(RuntimeUnavailableError, match="did not become ready"):
        state.start_instance_pool(2, "headless")

    assert stopped == [("dsr-1",)]
    assert state._started_instances == set()


def test_close_all_reports_stop_failure_and_retains_instances_for_retry(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "config" / "instances.json"
    config.parent.mkdir()
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("DSLE_INSTANCE_CONFIG", str(config))
    state = EnvironmentState(state_dir=tmp_path, allow_instance_start=True)
    state._started_instances.update(("dsr-1", "dsr-2"))
    monkeypatch.setattr(
        server_module,
        "stop_instances",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("stop failed")),
    )

    with pytest.raises(RuntimeUnavailableError, match="instance stop failed"):
        state.close_all()
    assert state._started_instances == {"dsr-1", "dsr-2"}


def test_server_leases_each_live_instance_to_only_one_environment(tmp_path: Path) -> None:
    created: list[FakeEnvironment] = []

    def factory(boss, **kwargs):
        environment = FakeEnvironment(boss, **kwargs)
        created.append(environment)
        return environment

    state = EnvironmentState(
        state_dir=tmp_path,
        env_factory=factory,
        allow_instance_start=False,
    )
    payload = {
        "boss": "asylum_demon",
        "kwargs": {"instance": "dsr-1"},
        "start_instance": False,
        "instance_mode": "headless",
    }
    first, _array = state.dispatch("create", payload)
    with pytest.raises(ConfigurationError, match="already belongs"):
        state.dispatch("create", payload)

    state.dispatch("close", {"env_id": first["env_id"]})
    second, _array = state.dispatch("create", payload)
    state.dispatch("close", {"env_id": second["env_id"]})
    assert len(created) == 2
    assert all(environment.closed for environment in created)


def test_server_rolls_back_environment_with_an_unsupported_public_contract(tmp_path: Path) -> None:
    created: list[FakeEnvironment] = []

    def factory(boss, **kwargs):
        environment = FakeEnvironment(boss, **kwargs)
        environment.observation_space = gym.spaces.Text(8)
        created.append(environment)
        return environment

    state = EnvironmentState(state_dir=tmp_path, env_factory=factory, allow_instance_start=False)
    with pytest.raises(RuntimeUnavailableError, match="does not support Text"):
        state.dispatch(
            "create",
            {
                "boss": "asylum_demon",
                "kwargs": {"instance": "dsr-1"},
                "start_instance": False,
            },
        )

    assert created[0].closed
    assert not state._environments and not state._reserved_instances


@pytest.mark.parametrize(
    ("operation", "fault", "match"),
    [
        ("reset", "reset_observation", "out-of-space observation"),
        ("reset", "reset_info", "info must be a mapping"),
        ("step", "step_observation", "out-of-space observation"),
        ("step", "step_reward", "finite number"),
        ("step", "step_terminal", "flags must be booleans"),
        ("step", "step_info", "info must be a mapping"),
        ("render", "render_frame", "HxWx3 uint8"),
    ],
)
def test_server_rejects_malformed_environment_results(
    tmp_path: Path, operation: str, fault: str, match: str
) -> None:
    class MalformedEnvironment(FakeEnvironment):
        def reset(self, *, seed=None, options=None):
            observation, info = super().reset(seed=seed, options=options)
            if fault == "reset_observation":
                observation = np.zeros((9,), dtype=np.uint8)
            if fault == "reset_info":
                info = []
            return observation, info

        def step(self, action):
            observation, reward, terminated, truncated, info = super().step(action)
            if fault == "step_observation":
                observation = np.zeros((9,), dtype=np.uint8)
            if fault == "step_reward":
                reward = float("nan")
            if fault == "step_terminal":
                terminated = 1
            if fault == "step_info":
                info = []
            return observation, reward, terminated, truncated, info

        def render(self):
            if fault == "render_frame":
                return np.zeros((2, 3, 3), dtype=np.float32)
            return super().render()

    state = EnvironmentState(
        state_dir=tmp_path,
        env_factory=lambda boss, **kwargs: MalformedEnvironment(boss, **kwargs),
        allow_instance_start=False,
    )
    created, _array = state.dispatch(
        "create",
        {
            "boss": "asylum_demon",
            "kwargs": {"instance": "dsr-1"},
            "start_instance": False,
        },
    )
    payload: dict[str, object] = {"env_id": created["env_id"]}
    if operation == "step":
        payload["action"] = 0
    try:
        with pytest.raises(RuntimeUnavailableError, match=match):
            state.dispatch(operation, payload)
    finally:
        state.close_all()


def test_server_rejects_noncanonical_instance_before_environment_creation(
    tmp_path: Path,
) -> None:
    created = []
    state = EnvironmentState(
        state_dir=tmp_path,
        env_factory=lambda *_args, **_kwargs: created.append(True),
        allow_instance_start=False,
    )

    with pytest.raises(ConfigurationError, match="dsr-1 through dsr-30"):
        state.dispatch(
            "create",
            {
                "boss": "asylum_demon",
                "kwargs": {"instance": "../config/instances"},
                "start_instance": False,
            },
        )

    assert created == []


def test_close_all_waits_for_an_active_step_before_closing(tmp_path: Path) -> None:
    step_started = threading.Event()
    release_step = threading.Event()
    environment_closed = threading.Event()
    failures: list[BaseException] = []

    class BlockingEnvironment(FakeEnvironment):
        def step(self, action):
            step_started.set()
            if not release_step.wait(timeout=5.0):
                raise TimeoutError("test step was not released")
            assert not self.closed
            return super().step(action)

        def close(self):
            self.closed = True
            environment_closed.set()

    state = EnvironmentState(
        state_dir=tmp_path,
        env_factory=lambda boss, **kwargs: BlockingEnvironment(boss, **kwargs),
        allow_instance_start=False,
    )
    created, _array = state.dispatch(
        "create",
        {
            "boss": "asylum_demon",
            "kwargs": {"instance": "dsr-1"},
            "start_instance": False,
        },
    )

    def run_step() -> None:
        try:
            state.dispatch("step", {"env_id": created["env_id"], "action": 0})
        except BaseException as exc:
            failures.append(exc)

    def close_everything() -> None:
        try:
            state.close_all()
        except BaseException as exc:
            failures.append(exc)

    step_thread = threading.Thread(target=run_step)
    close_thread = threading.Thread(target=close_everything)
    step_thread.start()
    assert step_started.wait(timeout=2.0)
    close_thread.start()
    assert not environment_closed.wait(timeout=0.05)
    release_step.set()
    step_thread.join(timeout=2.0)
    close_thread.join(timeout=2.0)

    assert not step_thread.is_alive()
    assert not close_thread.is_alive()
    assert failures == []
    assert environment_closed.is_set()


def test_existing_instance_config_cannot_escape_state_bind(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config" / "instances.json"
    instances = generate_instances(1, wineprefix_root=tmp_path.parent / "outside-prefixes")
    write_instance_config(config, instances)
    document = json.loads(config.read_text(encoding="utf-8"))
    document["state_dir"] = str(tmp_path)
    config.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setenv("DSLE_INSTANCE_CONFIG", str(config))
    state = EnvironmentState(state_dir=tmp_path, allow_instance_start=False)
    with pytest.raises(ConfigurationError, match="wineprefix"):
        state._ensure_configured_instance("dsr-1")
