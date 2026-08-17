from __future__ import annotations

import pytest

from dsle.models import RewardConfig
from dsle.rewards import transition_reward
from tests.fakes import state


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "one"])
def test_reward_override_requires_finite_numbers(value) -> None:
    with pytest.raises(ValueError, match="boss_damage"):
        RewardConfig(boss_damage=value)


def test_paper_reward_equation() -> None:
    previous = state(player_hp=500, player_hp_max=500, boss_hp=1000, boss_hp_max=1000)
    current = state(player_hp=400, player_hp_max=500, boss_hp=750, boss_hp_max=1000)
    result = transition_reward(previous, current, RewardConfig())
    assert result.boss_damage == 250
    assert result.player_damage == 100
    assert result.total == pytest.approx(-0.001 + 0.25 - 0.25 * 0.2)


def test_terminal_bonuses_match_paper() -> None:
    normal = state()
    win = transition_reward(normal, normal, RewardConfig(), terminated_reason="boss_defeated")
    death = transition_reward(normal, normal, RewardConfig(), terminated_reason="player_dead")
    assert win.total == pytest.approx(99.999)
    assert death.total == pytest.approx(-10.001)


def test_invalid_sensor_read_is_not_interpreted_as_damage() -> None:
    previous = state()
    missing = state(
        player_hp=None,
        player_hp_max=None,
        boss_hp=None,
        boss_hp_max=None,
        player_state_valid=False,
        boss_state_valid=False,
    )
    result = transition_reward(previous, missing, RewardConfig())
    assert result.boss_damage == 0
    assert result.player_damage == 0
    assert result.total == pytest.approx(-0.001)


def test_health_increases_never_create_negative_damage_reward() -> None:
    previous = state(player_hp=100, boss_hp=100)
    healed = state(player_hp=500, boss_hp=1000)
    result = transition_reward(previous, healed, RewardConfig())
    assert result.boss_damage == result.player_damage == 0
