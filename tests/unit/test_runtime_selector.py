from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pytest

import dsle.container as container_module
import dsle.env as env_module
import dsle.remote as remote_module
from dsle.models import RewardConfig
from tests.fakes import ScriptedBackend


def test_auto_uses_authoritative_local_env_for_injected_backend(monkeypatch) -> None:
    backend = ScriptedBackend()
    sentinel = object()
    calls = []
    monkeypatch.delenv("DSLE_CONTAINER", raising=False)
    monkeypatch.setattr(
        env_module,
        "DarkSoulsEnv",
        lambda boss, **kwargs: calls.append((boss, kwargs)) or sentinel,
    )
    environment = env_module.make("asylum_demon", backend=backend, action_ms=0)
    assert environment is sentinel
    assert calls == [("asylum_demon", {"backend": backend, "action_ms": 0, "verbose": True})]


def test_auto_uses_local_inside_container(tmp_path: Path, monkeypatch) -> None:
    sentinel = object()
    calls = []
    monkeypatch.setenv("DSLE_CONTAINER", "1")
    monkeypatch.setattr(
        env_module,
        "DarkSoulsEnv",
        lambda boss, **kwargs: calls.append((boss, kwargs)) or sentinel,
    )
    assert env_module.make("asylum_demon", max_steps=10) is sentinel
    assert calls == [("asylum_demon", {"max_steps": 10, "verbose": True})]


def test_auto_host_uses_owned_managed_helper(monkeypatch) -> None:
    sentinel = object()
    calls = []
    monkeypatch.delenv("DSLE_CONTAINER", raising=False)
    monkeypatch.setattr(
        container_module,
        "make_managed",
        lambda boss, **kwargs: calls.append((boss, kwargs)) or sentinel,
    )

    result = env_module.make(
        "capra_demon",
        game_dir="/games/dsr",
        output_dir="/state/dsle",
        image="local-dsle:test",
        instance="dsr-2",
    )
    assert result is sentinel
    assert calls == [
        (
            "capra_demon",
            {
                "game_dir": "/games/dsr",
                "output_dir": "/state/dsle",
                "image": "local-dsle:test",
                "container_name": None,
                "num_instances": 1,
                "start_instance": True,
                "instance_mode": "headless",
                "verbose": True,
                "instance": "dsr-2",
            },
        )
    ]


def test_auto_backend_none_does_not_force_local_host_mode(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.delenv("DSLE_CONTAINER", raising=False)
    monkeypatch.setattr(container_module, "make_managed", lambda *_args, **_kwargs: sentinel)
    assert (
        env_module.make(
            "asylum_demon",
            game_dir="/games/dsr",
            backend=None,
        )
        is sentinel
    )


def test_explicit_managed_overrides_container_marker(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setenv("DSLE_CONTAINER", "1")
    monkeypatch.setattr(container_module, "make_managed", lambda *_args, **_kwargs: sentinel)
    assert (
        env_module.make(
            "asylum_demon",
            runtime="managed",
            game_dir="/games/dsr",
        )
        is sentinel
    )


def test_external_uses_socket_proxy_and_serializes_reward(monkeypatch) -> None:
    sentinel = object()
    calls = []
    monkeypatch.setattr(
        remote_module,
        "RemoteDarkSoulsEnv",
        lambda socket, boss, **kwargs: calls.append((socket, boss, kwargs)) or sentinel,
    )
    result = env_module.make(
        "asylum_demon",
        runtime="external",
        socket_path="/state/run/dsle.sock",
        token_file="/state/run/dsle.token",
        reward=RewardConfig(win_bonus=25.0),
        instance="dsr-3",
    )
    assert result is sentinel
    socket, boss, kwargs = calls[0]
    assert socket == "/state/run/dsle.sock"
    assert boss == "asylum_demon"
    assert kwargs["token_file"] == "/state/run/dsle.token"
    assert kwargs["start_instance"] is True
    assert kwargs["instance"] == "dsr-3"
    assert kwargs["reward"]["win_bonus"] == 25.0


def test_external_can_bulk_start_a_non_owning_vector(monkeypatch) -> None:
    created: list[str] = []
    rpc_calls: list[tuple[str, dict[str, object]]] = []

    class Proxy(gym.Env[np.ndarray, int]):
        metadata: ClassVar[dict[str, Any]] = {}

        def __init__(self, instance: str):
            created.append(instance)
            self.action_space = gym.spaces.Discrete(2)
            self.observation_space = gym.spaces.Box(0, 1, (1,), np.uint8)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1, np.uint8), {}

        def step(self, action):
            return np.zeros(1, np.uint8), 0.0, False, False, {}

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def request(self, operation, payload):
            rpc_calls.append((operation, payload))
            return (
                {
                    "statuses": {
                        f"dsr-{index}": {"status": "ready", "running": True}
                        for index in range(1, 3)
                    }
                },
                None,
            )

        def close(self):
            pass

    monkeypatch.setattr(remote_module, "read_auth_token", lambda _path: "a" * 64)
    monkeypatch.setattr(remote_module, "RPCClient", Client)
    monkeypatch.setattr(
        remote_module,
        "RemoteDarkSoulsEnv",
        lambda _socket, _boss, **kwargs: Proxy(kwargs["instance"]),
    )
    environments = env_module.make(
        "asylum_demon",
        runtime="external",
        socket_path="/state/run/dsle.sock",
        token_file="/state/run/dsle.token",
        num_instances=2,
    )
    assert environments.num_envs == 2
    assert created == ["dsr-1", "dsr-2"]
    assert rpc_calls == [("start_instances", {"count": 2, "mode": "headless"})]
    environments.close()


def test_external_can_attach_to_existing_pool_without_starting_or_owning(monkeypatch) -> None:
    created: list[tuple[str, bool]] = []

    class Proxy(gym.Env[np.ndarray, int]):
        metadata: ClassVar[dict[str, Any]] = {}

        def __init__(self, instance: str, start_instance: bool):
            created.append((instance, start_instance))
            self.action_space = gym.spaces.Discrete(2)
            self.observation_space = gym.spaces.Box(0, 1, (1,), np.uint8)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(1, np.uint8), {}

        def step(self, action):
            return np.zeros(1, np.uint8), 0.0, False, False, {}

    class UnexpectedClient:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("attach-only mode must not issue lifecycle RPCs")

    monkeypatch.setattr(remote_module, "RPCClient", UnexpectedClient)
    monkeypatch.setattr(
        remote_module,
        "RemoteDarkSoulsEnv",
        lambda _socket, _boss, **kwargs: Proxy(kwargs["instance"], kwargs["start_instance"]),
    )

    environments = env_module.make(
        "asylum_demon",
        runtime="external",
        socket_path="/state/run/dsle.sock",
        token_file="/state/run/dsle.token",
        num_instances=2,
        start_instance=False,
    )

    assert environments.num_envs == 2
    assert created == [("dsr-1", False), ("dsr-2", False)]
    environments.close()


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("connect_timeout", True),
        ("connect_timeout", "3"),
        ("request_timeout", float("nan")),
        ("request_timeout", 0.0),
    ],
)
def test_external_vector_rejects_invalid_rpc_timeouts(monkeypatch, option, value) -> None:
    monkeypatch.setattr(remote_module, "read_auth_token", lambda _path: "a" * 64)

    with pytest.raises(ValueError, match="finite positive number"):
        env_module.make(
            "asylum_demon",
            runtime="external",
            socket_path="/state/run/dsle.sock",
            token_file="/state/run/dsle.token",
            num_instances=2,
            **{option: value},
        )


@pytest.mark.parametrize("runtime", ["managed", "external"])
def test_remote_modes_reject_in_process_backends(runtime: str) -> None:
    with pytest.raises(ValueError, match="backend"):
        env_module.make(
            "asylum_demon",
            runtime=runtime,
            backend=ScriptedBackend(),
        )


def test_mode_specific_options_and_local_lifecycle_are_rejected() -> None:
    with pytest.raises(ValueError, match="Local runtime"):
        env_module.make("asylum_demon", runtime="local", socket_path="/tmp/dsle.sock")
    with pytest.raises(ValueError, match="does not own instance lifecycle"):
        env_module.make("asylum_demon", runtime="local", start_instance=True)
    with pytest.raises(ValueError, match="Managed runtime"):
        env_module.make(
            "asylum_demon",
            runtime="managed",
            socket_path="/tmp/dsle.sock",
        )
    with pytest.raises(ValueError, match="External runtime"):
        env_module.make(
            "asylum_demon",
            runtime="external",
            game_dir="/games/dsr",
        )


def test_invalid_runtime_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="auto, managed, external, or local"):
        env_module.make("asylum_demon", runtime="magic")
    with pytest.raises(ValueError, match="boolean or None"):
        env_module.make("asylum_demon", start_instance="yes")
    with pytest.raises(ValueError, match="instance_mode"):
        env_module.make("asylum_demon", instance_mode="remote-desktop")
    with pytest.raises(ValueError, match="does not manage instance_mode"):
        env_module.make("asylum_demon", runtime="local", instance_mode="gui")
    with pytest.raises(ValueError, match="verbose must be a boolean"):
        env_module.make("asylum_demon", verbose="yes")
