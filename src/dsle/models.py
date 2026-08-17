"""Typed data shared by the environment, configuration, and runtime layers."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np


def _immutable_value(value: Any) -> Any:
    """Recursively freeze configuration containers."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _immutable_value(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_immutable_value(item) for item in value)
    return value


def immutable_mapping(value: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """Return a recursively immutable mapping suitable for frozen dataclasses."""

    return _immutable_value(value or {})


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the paper's shaped transition reward."""

    boss_damage: float = 1.0
    player_damage: float = -0.25
    step_penalty: float = -0.001
    win_bonus: float = 100.0
    death_penalty: float = -10.0
    timeout_penalty: float = 0.0
    runtime_error_penalty: float = -10.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"RewardConfig.{name} must be a finite number")


@dataclass(frozen=True)
class EpisodeLimits:
    """Episode and orchestration limits."""

    max_steps: int = 7200
    setup_timeout_s: float = 120.0
    cleanup_timeout_s: float = 60.0
    minimum_victory_step: int = 1
    victory_confirmations: int = 1


@dataclass(frozen=True)
class ReadinessConfig:
    """How reset detects that control can be handed to the policy."""

    mode: str = "traverse_light"
    template: str | None = "traverseLight"
    threshold: float = 0.9
    timeout_s: float = 60.0
    advance_key: str = "w"
    advance_hold_s: float = 0.05
    poll_s: float = 0.3
    interact_after_ready: bool = True


@dataclass(frozen=True)
class MemoryConfig:
    """Boss-specific HP source and process-memory pointer chains.

    ``defeated_flag`` mode intentionally has no HP chains and exposes binary
    health: one before the event flag is set and zero afterward.
    """

    hp_chains: tuple[tuple[int, ...], ...]
    hp_mode: str
    defeated_offsets: tuple[int, ...]
    defeated_bit: int


@dataclass(frozen=True)
class VictoryConfig:
    """Independent signals that can establish a boss victory."""

    hp_zero_is_victory: bool
    templates: tuple[str, ...]
    template_threshold: float = 0.8


@dataclass(frozen=True)
class RuntimeOperation:
    """One validated declarative reset or cleanup operation."""

    op: str
    params: Mapping[str, Any] = field(default_factory=immutable_mapping)
    when: tuple[str, ...] = ("always",)


@dataclass(frozen=True)
class BossMetadata:
    """Descriptive benchmark metadata; it never changes runtime behavior."""

    tier: str
    tags: tuple[str, ...]
    description: str
    expected_hp: int | None = None


@dataclass(frozen=True)
class BossConfig:
    """Complete validated description of one boss task."""

    boss_id: str
    display_name: str
    aliases: tuple[str, ...]
    availability: str
    save_states: Mapping[str, str]
    memory: MemoryConfig
    victory: VictoryConfig
    readiness: ReadinessConfig
    reward: RewardConfig
    limits: EpisodeLimits
    setup: tuple[RuntimeOperation, ...]
    cleanup: tuple[RuntimeOperation, ...]
    metadata: BossMetadata
    source: Path

    def save_state(self, difficulty: str) -> str:
        """Return the configured save filename for a difficulty profile."""

        try:
            return self.save_states[difficulty]
        except KeyError as exc:
            choices = ", ".join(sorted(self.save_states))
            raise ValueError(
                f"Boss '{self.boss_id}' has no '{difficulty}' save state; choose one of: {choices}"
            ) from exc


@dataclass(frozen=True)
class EnvironmentDefaults:
    """Global environment defaults loaded from ``defaults.yaml``."""

    frame_height: int
    frame_width: int
    action_ms: int
    auto_lock_on: bool
    lock_on_interval: int
    monitor_index: int
    template_threshold: float
    reward: RewardConfig
    limits: EpisodeLimits
    readiness: ReadinessConfig
    setup_presets: Mapping[str, tuple[RuntimeOperation, ...]]
    cleanup_presets: Mapping[str, tuple[RuntimeOperation, ...]]


@dataclass(frozen=True)
class InstanceConfig:
    """Isolation parameters for one live game process."""

    name: str
    display: str
    display_num: int
    desktop_name: str
    wineprefix: Path
    vnc_port: int
    xdg_runtime_dir: Path
    save_dir: Path | None = None


@dataclass(frozen=True)
class GameState:
    """Diagnostic state read from the game process after an action."""

    player_hp: int | None
    player_hp_max: int | None
    boss_hp: int | None
    boss_hp_max: int | None
    death_count: int | None
    locked_on: bool | None
    boss_defeated: bool
    player_dead: bool
    player_state_valid: bool = True
    boss_state_valid: bool = True
    boss_hps: tuple[int | None, ...] = ()
    boss_hp_maxes: tuple[int | None, ...] = ()
    player_x: float | None = None
    player_y: float | None = None
    player_z: float | None = None

    @property
    def player_location(self) -> dict[str, float | None]:
        """Return the diagnostic player position with stable axis names."""

        return {"x": self.player_x, "y": self.player_y, "z": self.player_z}

    @property
    def player_hp_fraction(self) -> float:
        if not self.player_state_valid or self.player_hp is None or self.player_hp_max is None:
            return 0.0
        return float(np.clip(self.player_hp / max(1, self.player_hp_max), 0.0, 1.0))

    @property
    def boss_hp_fraction(self) -> float:
        if not self.boss_state_valid or self.boss_hp is None or self.boss_hp_max is None:
            return 0.0
        return float(np.clip(self.boss_hp / max(1, self.boss_hp_max), 0.0, 1.0))


@dataclass(frozen=True)
class RuntimeObservation:
    """An RGB frame and matching process state from a runtime backend."""

    frame: np.ndarray
    state: GameState
    timestamp: float


@dataclass(frozen=True)
class EpisodeOutcome:
    """Terminal context given to backend cleanup operations."""

    win: bool
    terminated: bool
    truncated: bool
    reason: str
    steps: int
