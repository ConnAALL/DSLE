"""Backend boundary between the Gymnasium API and a live game process."""

from __future__ import annotations

from abc import ABC, abstractmethod

from dsle.actions import ActionSpec
from dsle.models import BossConfig, EpisodeOutcome, RuntimeObservation


class GameBackend(ABC):
    """Minimal interface implemented by live and test backends.

    The environment owns the backend. A backend may hold X11, process-memory,
    capture, and save-file resources, but none of those concerns leak into the
    Gymnasium state machine.
    """

    @abstractmethod
    def reset(self, boss: BossConfig, save_state: str) -> RuntimeObservation:
        """Load a scenario and return the first policy observation."""

    @abstractmethod
    def perform(self, action: ActionSpec, hold_s: float) -> None:
        """Inject one action and block until its configured hold completes."""

    @abstractmethod
    def observe(self, boss: BossConfig) -> RuntimeObservation:
        """Capture a frame and matching diagnostic state."""

    @abstractmethod
    def set_player_hp(self, boss: BossConfig, hp: int) -> RuntimeObservation:
        """Write current player HP and return the resulting observation."""

    @abstractmethod
    def set_boss_hp(
        self,
        boss: BossConfig,
        boss_number: int,
        hp: int,
    ) -> RuntimeObservation:
        """Write one boss HP, or every HP for ``(-1, 0)``, and return the readback."""

    @abstractmethod
    def finish(self, boss: BossConfig, outcome: EpisodeOutcome) -> None:
        """Run post-episode operations after a terminal or truncated transition."""

    def return_to_menu(self, *, timeout_s: float = 30.0) -> None:
        """Return to a verified title menu when the backend supports manual control."""

        raise NotImplementedError("This backend cannot return directly to the title menu")

    @abstractmethod
    def close(self) -> None:
        """Release every backend-owned resource. Calls must be idempotent."""

    def __enter__(self) -> GameBackend:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
