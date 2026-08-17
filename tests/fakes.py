"""Deterministic backend doubles for environment and wrapper tests."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import replace

import numpy as np

from dsle.models import BossConfig, EpisodeOutcome, GameState, RuntimeObservation
from dsle.runtime import GameBackend


def state(
    *,
    player_hp: int | None = 500,
    player_hp_max: int | None = 500,
    boss_hp: int | None = 1000,
    boss_hp_max: int | None = 1000,
    boss_hps: tuple[int | None, ...] | None = None,
    boss_hp_maxes: tuple[int | None, ...] | None = None,
    death_count: int | None = 0,
    locked_on: bool | None = True,
    boss_defeated: bool = False,
    player_dead: bool = False,
    player_state_valid: bool = True,
    boss_state_valid: bool = True,
    player_x: float | None = 1.25,
    player_y: float | None = 2.5,
    player_z: float | None = -3.75,
) -> GameState:
    return GameState(
        player_hp=player_hp,
        player_hp_max=player_hp_max,
        boss_hp=boss_hp,
        boss_hp_max=boss_hp_max,
        death_count=death_count,
        locked_on=locked_on,
        boss_defeated=boss_defeated,
        player_dead=player_dead,
        player_state_valid=player_state_valid,
        boss_state_valid=boss_state_valid,
        boss_hps=(boss_hp,) if boss_hps is None else boss_hps,
        boss_hp_maxes=(boss_hp_max,) if boss_hp_maxes is None else boss_hp_maxes,
        player_x=player_x,
        player_y=player_y,
        player_z=player_z,
    )


class ScriptedBackend(GameBackend):
    def __init__(
        self,
        states: list[GameState | Exception] | None = None,
        *,
        initial: GameState | None = None,
        frame: np.ndarray | None = None,
        finish_error: Exception | None = None,
    ):
        self.initial = initial or state()
        self.states = deque(states or [])
        self.frame = (
            np.asarray(frame, dtype=np.uint8)
            if frame is not None
            else np.full((600, 800, 3), (10, 20, 30), dtype=np.uint8)
        )
        self.finish_error = finish_error
        self.actions: list[tuple[object, float]] = []
        self.resets: list[tuple[str, str]] = []
        self.outcomes: list[EpisodeOutcome] = []
        self.health_writes: list[tuple[object, ...]] = []
        self.menu_timeouts: list[float] = []
        self._written_state: GameState | None = None
        self.closed = False

    def _observation(self, item: GameState) -> RuntimeObservation:
        return RuntimeObservation(self.frame.copy(), item, time.monotonic())

    def reset(self, boss: BossConfig, save_state: str) -> RuntimeObservation:
        self.resets.append((boss.boss_id, save_state))
        self._written_state = None
        return self._observation(self.initial)

    def perform(self, action, hold_s: float) -> None:
        self.actions.append((action, hold_s))

    def observe(self, boss: BossConfig) -> RuntimeObservation:
        item = self.states.popleft() if self.states else self._written_state or self.initial
        if isinstance(item, Exception):
            raise item
        return self._observation(item)

    def set_player_hp(self, boss: BossConfig, hp: int) -> RuntimeObservation:
        current = self._written_state or self.initial
        self._written_state = replace(current, player_hp=hp, player_dead=hp == 0)
        self.health_writes.append(("player", hp))
        return self._observation(self._written_state)

    def set_boss_hp(
        self,
        boss: BossConfig,
        boss_number: int,
        hp: int,
    ) -> RuntimeObservation:
        current = self._written_state or self.initial
        values = list(current.boss_hps or (current.boss_hp,))
        aggregate: int | None
        if boss_number == -1:
            values = [hp] * len(values)
            aggregate = hp
        else:
            values[boss_number - 1] = hp
            aggregate = hp if boss_number == 1 else current.boss_hp
        self._written_state = replace(current, boss_hp=aggregate, boss_hps=tuple(values))
        self.health_writes.append(("boss", boss_number, hp))
        return self._observation(self._written_state)

    def finish(self, boss: BossConfig, outcome: EpisodeOutcome) -> None:
        self.outcomes.append(outcome)
        if self.finish_error is not None:
            raise self.finish_error

    def return_to_menu(self, *, timeout_s: float = 30.0) -> None:
        self.menu_timeouts.append(timeout_s)

    def close(self) -> None:
        self.closed = True
