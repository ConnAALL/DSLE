from __future__ import annotations

import numpy as np
import pytest

from dsle.suites import DSLE_5, resolve_suite
from dsle.vector import make_vector_env
from tests.fakes import ScriptedBackend


def test_named_and_full_suites_resolve_without_agent_code() -> None:
    assert resolve_suite("dsle5") == DSLE_5
    assert len(resolve_suite("full")) == 22
    with pytest.raises(KeyError, match="Available"):
        resolve_suite("missing")


def test_sync_vector_env_maps_unique_instance_names() -> None:
    created: list[str] = []

    def backend_factory(instance, _assets, _config):
        created.append(instance)
        return ScriptedBackend()

    env = make_vector_env(
        "asylum_demon",
        2,
        asynchronous=False,
        backend_factory=backend_factory,
        action_ms=0,
        max_steps=1,
    )
    try:
        observations, info = env.reset()
        assert observations.shape == (2, 1, 600, 800)
        assert set(info["instance"].tolist()) == {"dsr-1", "dsr-2"}
        next_observations, rewards, terminated, truncated, _ = env.step(
            np.asarray([0, 1], dtype=np.int64)
        )
        assert next_observations.shape == observations.shape
        assert rewards.shape == terminated.shape == truncated.shape == (2,)
        assert truncated.all()
        assert created == ["dsr-1", "dsr-2"]
    finally:
        env.close()


@pytest.mark.parametrize(
    ("num_envs", "instances"),
    [
        (0, None),
        (31, None),
        (True, None),
        (1.5, None),
        (2, ("dsr-1",)),
        (2, ("same", "same")),
        (2, "dsr-1"),
        (2, 5),
        (2, []),
        (2, ("dsr-1", "")),
        (2, ("dsr-1", " dsr-2")),
        (2, ("dsr-1", "dsr-31")),
    ],
)
def test_vector_env_rejects_invalid_instance_layout(num_envs, instances) -> None:
    with pytest.raises(ValueError):
        make_vector_env("asylum_demon", num_envs, instances=instances)


def test_vector_env_rejects_ambiguous_lifecycle_options() -> None:
    with pytest.raises(ValueError, match="asynchronous"):
        make_vector_env("asylum_demon", 1, asynchronous=1)
    with pytest.raises(ValueError, match="instances"):
        make_vector_env("asylum_demon", 1, instance="dsr-1")
    with pytest.raises(ValueError, match="cannot share"):
        make_vector_env("asylum_demon", 1, backend=ScriptedBackend())
