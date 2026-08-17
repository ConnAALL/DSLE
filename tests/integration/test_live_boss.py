from __future__ import annotations

import os

import pytest

import dsle

RUN_LIVE = os.environ.get("DSLE_RUN_LIVE_TESTS") == "1"
RUN_BOSS_SWEEP = os.environ.get("DSLE_RUN_BOSS_SWEEP") == "1"
RUN_BOOSTED_BOSS_SWEEP = os.environ.get("DSLE_RUN_BOOSTED_BOSS_SWEEP") == "1"
TEST_RUNTIME = os.environ.get("DSLE_TEST_RUNTIME", "auto")


def make_live_environment(boss: str, **kwargs):
    if TEST_RUNTIME not in {"auto", "managed", "external", "local"}:
        raise ValueError("DSLE_TEST_RUNTIME must be auto, managed, external, or local")
    selected = dict(kwargs)
    selected["runtime"] = TEST_RUNTIME
    if TEST_RUNTIME == "external":
        selected["start_instance"] = False
    return dsle.make(boss, **selected)


def assert_clean_transition(terminated: bool, truncated: bool, info: dict) -> None:
    """Reject backend/cleanup failures even though Gymnasium reports them as truncations."""

    assert "runtime_error" not in info
    assert "cleanup_error" not in info
    if terminated:
        assert not truncated
        assert info["terminated_reason"] in {"boss_defeated", "player_dead"}
        assert info["truncated_reason"] is None
    elif truncated:
        assert info["terminated_reason"] is None
        assert info["truncated_reason"] == "max_steps"
    else:
        assert info["terminated_reason"] is None
        assert info["truncated_reason"] is None


@pytest.mark.game
@pytest.mark.boss
@pytest.mark.slow
@pytest.mark.skipif(not RUN_LIVE, reason="set DSLE_RUN_LIVE_TESTS=1 inside a started runtime")
def test_asylum_demon_reset_and_short_episode() -> None:
    env = make_live_environment("asylum_demon", instance="dsr-1", max_steps=8)
    try:
        observation, info = env.reset()
        assert observation.shape == (1, 600, 800)
        assert info["player_state_valid"] and info["boss_state_valid"]
        done = False
        for action in range(8):
            observation, reward, terminated, truncated, info = env.step(action % 14)
            assert env.observation_space.contains(observation)
            assert isinstance(reward, float)
            assert_clean_transition(terminated, truncated, info)
            done = terminated or truncated
            if done:
                break
        assert done
    finally:
        env.close()


@pytest.mark.game
@pytest.mark.boss
@pytest.mark.slow
@pytest.mark.skipif(
    not (RUN_LIVE and RUN_BOSS_SWEEP),
    reason="set DSLE_RUN_LIVE_TESTS=1 and DSLE_RUN_BOSS_SWEEP=1 inside the runtime",
)
@pytest.mark.parametrize("boss", dsle.list_bosses())
def test_every_configured_boss_resets_and_steps_once(boss: str) -> None:
    """Release-gate sweep: every advertised task must reach a real transition."""

    env = make_live_environment(boss, instance="dsr-1", max_steps=1)
    try:
        observation, info = env.reset()
        assert observation.shape == (1, 600, 800)
        assert info["boss"] == boss
        assert info["player_state_valid"] and info["boss_state_valid"]
        observation, reward, terminated, truncated, info = env.step(0)
        assert env.observation_space.contains(observation)
        assert isinstance(reward, float)
        assert terminated or truncated
        assert_clean_transition(terminated, truncated, info)
    finally:
        env.close()


@pytest.mark.game
@pytest.mark.boss
@pytest.mark.slow
@pytest.mark.skipif(
    not (RUN_LIVE and RUN_BOOSTED_BOSS_SWEEP),
    reason=("set DSLE_RUN_LIVE_TESTS=1 and DSLE_RUN_BOOSTED_BOSS_SWEEP=1 inside the runtime"),
)
@pytest.mark.parametrize("boss", dsle.list_bosses(include_experimental=False))
def test_every_configured_boosted_boss_resets_and_steps_once(boss: str) -> None:
    """Optional release gate for every advertised boosted scenario save."""

    env = make_live_environment(
        boss,
        instance="dsr-1",
        difficulty="boosted",
        max_steps=1,
    )
    try:
        observation, info = env.reset(options={"difficulty": "boosted"})
        assert observation.shape == (1, 600, 800)
        assert info["boss"] == boss
        assert info["difficulty"] == "boosted"
        assert info["player_state_valid"] and info["boss_state_valid"]
        observation, reward, terminated, truncated, info = env.step(0)
        assert env.observation_space.contains(observation)
        assert isinstance(reward, float)
        assert terminated or truncated
        assert_clean_transition(terminated, truncated, info)
    finally:
        env.close()
