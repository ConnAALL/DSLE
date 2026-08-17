"""Reward and transition-delta calculations shared by every agent example."""

from __future__ import annotations

from dataclasses import dataclass

from dsle.models import GameState, RewardConfig


@dataclass(frozen=True)
class RewardBreakdown:
    """Auditable components of a DSLE transition reward."""

    boss_damage: int
    player_damage: int
    boss_damage_reward: float
    player_damage_reward: float
    step_reward: float
    terminal_reward: float

    @property
    def total(self) -> float:
        return float(
            self.boss_damage_reward
            + self.player_damage_reward
            + self.step_reward
            + self.terminal_reward
        )


def transition_reward(
    previous: GameState,
    current: GameState,
    config: RewardConfig,
    *,
    terminated_reason: str | None = None,
    truncated_reason: str | None = None,
) -> RewardBreakdown:
    """Compute the paper-specified reward without treating missing reads as damage."""

    boss_damage = 0
    if (
        previous.boss_state_valid
        and current.boss_state_valid
        and previous.boss_hp is not None
        and current.boss_hp is not None
    ):
        boss_damage = max(0, previous.boss_hp - current.boss_hp)

    player_damage = 0
    if (
        previous.player_state_valid
        and current.player_state_valid
        and previous.player_hp is not None
        and current.player_hp is not None
    ):
        player_damage = max(0, previous.player_hp - current.player_hp)

    boss_max = max(1, previous.boss_hp_max or current.boss_hp_max or 1)
    player_max = max(1, previous.player_hp_max or current.player_hp_max or 1)
    terminal_reward = 0.0
    if terminated_reason == "boss_defeated":
        terminal_reward += config.win_bonus
    elif terminated_reason == "player_dead":
        terminal_reward += config.death_penalty
    if truncated_reason == "max_steps":
        terminal_reward += config.timeout_penalty
    elif truncated_reason == "runtime_error":
        terminal_reward += config.runtime_error_penalty

    return RewardBreakdown(
        boss_damage=boss_damage,
        player_damage=player_damage,
        boss_damage_reward=config.boss_damage * (boss_damage / boss_max),
        player_damage_reward=config.player_damage * (player_damage / player_max),
        step_reward=config.step_penalty,
        terminal_reward=terminal_reward,
    )
