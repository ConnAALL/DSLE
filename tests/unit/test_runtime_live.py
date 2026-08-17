from __future__ import annotations

import threading

import numpy as np
import pytest

from dsle.config import ConfigRepository
from dsle.runtime.live import LiveGameBackend, _memory_reports_boss_defeated
from dsle.runtime.memory import BossMemory, PlayerMemory, ReadResult


def boss_memory(*, hp: int, defeated: bool) -> BossMemory:
    return BossMemory(
        hp=ReadResult.success(hp),
        defeated=ReadResult.success(defeated),
    )


def test_event_flag_boss_uses_defeated_flag() -> None:
    boss = ConfigRepository().get("seath_the_scaleless")

    assert not _memory_reports_boss_defeated(boss, boss_memory(hp=2643, defeated=False))
    assert _memory_reports_boss_defeated(boss, boss_memory(hp=1200, defeated=True))


def test_live_state_never_promotes_post_boss_template_to_victory() -> None:
    boss = ConfigRepository().get("seath_the_scaleless")

    class Memory:
        @staticmethod
        def read_player() -> PlayerMemory:
            return PlayerMemory(
                hp=ReadResult.success(594),
                hp_max=ReadResult.success(594),
                death_count=ReadResult.success(0),
                locked_on=ReadResult.success(True),
                x=ReadResult.success(3.16),
                y=ReadResult.success(-21.4),
                z=ReadResult.success(182.72),
            )

        @staticmethod
        def read_boss(_config) -> BossMemory:
            return boss_memory(hp=2643, defeated=False)

    class Templates:
        @staticmethod
        def match_any(*_args, **_kwargs):
            raise AssertionError("post-boss templates must not run during state reads")

    backend = object.__new__(LiveGameBackend)
    backend._memory = Memory()
    backend._templates = Templates()
    backend._diagnostics = {}
    backend._death_count_start = 0
    backend._boss_hp_max = 2643

    state = backend._read_state(boss, np.zeros((600, 800, 3), dtype=np.uint8))

    assert state.boss_hp == 2643
    assert not state.boss_defeated
    assert state.player_location == {"x": 3.16, "y": -21.4, "z": 182.72}


def test_live_state_exposes_each_boss_hp_chain_with_independent_initial_maximums() -> None:
    boss = ConfigRepository().get("ornstein_and_smough")

    class Memory:
        @staticmethod
        def read_player() -> PlayerMemory:
            return PlayerMemory(
                hp=ReadResult.success(594),
                hp_max=ReadResult.success(594),
                death_count=ReadResult.success(0),
                locked_on=ReadResult.success(True),
            )

        @staticmethod
        def read_boss(_config) -> BossMemory:
            return BossMemory(
                hp=ReadResult.success(1642),
                defeated=ReadResult.success(False),
                hp_values=(ReadResult.success(1642), ReadResult.success(2981)),
            )

    backend = object.__new__(LiveGameBackend)
    backend._memory = Memory()
    backend._diagnostics = {}
    backend._death_count_start = 0
    backend._boss_hp_max = 1642
    backend._boss_hp_maxes = ()

    state = backend._read_state(boss, np.zeros((600, 800, 3), dtype=np.uint8))

    assert state.boss_hp == 1642
    assert state.boss_hps == (1642, 2981)
    assert state.boss_hp_maxes == (1642, 2981)


@pytest.mark.parametrize(
    "boss_id",
    [
        "ceaseless_discharge",
        "demon_firesage",
        "moonlight_butterfly",
        "stray_demon",
    ],
)
def test_hp_zero_boss_ignores_stale_event_flag(boss_id: str) -> None:
    boss = ConfigRepository().get(boss_id)

    assert not _memory_reports_boss_defeated(boss, boss_memory(hp=1, defeated=True))
    assert _memory_reports_boss_defeated(boss, boss_memory(hp=0, defeated=False))


def test_wait_until_ready_normalizes_validated_mode_text() -> None:
    backend = object.__new__(LiveGameBackend)
    boss = ConfigRepository().get("asylum_demon")

    backend.wait_until_ready(boss, {"mode": " none "}, timeout_s=1.0)


def test_repeated_action_normalizes_validated_action_text() -> None:
    action, hold_s = LiveGameBackend._configured_action(" move_forward ", 0.125)

    assert action.name == "move_forward"
    assert hold_s == 0.125


def test_return_to_menu_attaches_and_verifies_title_state() -> None:
    calls: list[object] = []

    class Memory:
        @staticmethod
        def attach() -> None:
            calls.append("attach")

    backend = object.__new__(LiveGameBackend)
    backend._lock = threading.RLock()
    backend._closed = False
    backend._memory = Memory()
    backend._episode_ready = True
    backend._boss = object()
    backend._save_state = "asylum_demon.sl2"
    backend.ensure_menu = lambda *, timeout_s, threshold: calls.append(
        ("ensure_menu", timeout_s, threshold)
    )

    backend.return_to_menu(timeout_s=12.0, threshold=0.9)

    assert calls == ["attach", ("ensure_menu", 12.0, 0.9)]
    assert backend._episode_ready is False
    assert backend._boss is None
    assert backend._save_state == ""


def test_ensure_menu_prefers_valid_player_memory_over_false_splash_match() -> None:
    calls: list[str] = []

    class Memory:
        player_reads = 0

        def read_player(self) -> PlayerMemory:
            self.player_reads += 1
            valid = self.player_reads == 1
            return PlayerMemory(
                hp=ReadResult(value=594 if valid else None, valid=valid),
                hp_max=ReadResult(value=594 if valid else None, valid=valid),
                death_count=ReadResult.success(0),
                locked_on=ReadResult.success(False),
            )

        @staticmethod
        def return_to_title() -> None:
            calls.append("return_to_title")

    template_scans: list[str] = []
    backend = object.__new__(LiveGameBackend)
    backend._memory = Memory()
    backend._menu_state = None
    backend._focus = lambda: object()
    backend._frame = lambda: np.zeros((600, 800, 3), dtype=np.uint8)

    def scan(_frame, _threshold):
        template_scans.append("scan")
        # Before the player-memory read, imitate the false splash match seen
        # against a real dark gameplay frame. Afterwards, imitate the menu.
        return "company1" if backend._memory.player_reads == 0 else "menuOptionsContinue"

    backend._scan_menu = scan
    backend._sleep_with_deadline = lambda *_args, **_kwargs: None

    backend.ensure_menu(timeout_s=5.0, threshold=0.8)

    assert calls == ["return_to_title"]
    assert template_scans == ["scan"]
    assert backend._menu_state == "continue"
