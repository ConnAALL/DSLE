"""Live-game implementation of :class:`~dsle.runtime.base.GameBackend`."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from platformdirs import user_data_path

from dsle.actions import ACTION_SPECS, ActionSpec
from dsle.exceptions import (
    ResetError,
    RuntimeCommunicationError,
    RuntimeUnavailableError,
)
from dsle.models import (
    BossConfig,
    EpisodeOutcome,
    GameState,
    RuntimeObservation,
)
from dsle.progress import EventSink, LifecycleLogger
from dsle.runtime.base import GameBackend
from dsle.runtime.capture import MSSCapture
from dsle.runtime.instances import resolve_instance
from dsle.runtime.memory import BossMemory, DSRMemory, PlayerMemory, ReadResult
from dsle.runtime.operations import OperationContext, OperationExecutor
from dsle.runtime.saves import SaveManager, discover_save_directory, resolve_scenario_directory
from dsle.runtime.templates import TemplateCatalog
from dsle.runtime.version import GAME_EXECUTABLE, verify_game_executable
from dsle.runtime.x11 import X11Input

ASSET_DIR_ENV = "DSLE_ASSET_DIR"
MONITOR_INDEX_ENV = "DSLE_MONITOR_INDEX"

_ACTION_BY_NAME = {spec.name: spec for spec in ACTION_SPECS}
_MENU_ACTIONS: dict[str, tuple[str, ...] | None] = {
    "company1": ("esc",),
    "company2": ("esc",),
    "company3": ("esc",),
    "gameServer": ("right", "e"),
    "lastTime": ("e",),
    "menuOptionsNew": None,
    "menuOptionsContinue": None,
    "offlineMode": ("e",),
    "pressAny": ("e",),
    "privacyPolicy": None,
}


def default_asset_dir() -> Path:
    """Resolve runtime assets without depending on the caller's working directory."""

    configured = os.environ.get(ASSET_DIR_ENV)
    if configured:
        return Path(configured).expanduser().resolve(strict=False)
    checkout_assets = Path(__file__).resolve().parents[3] / "assets"
    if checkout_assets.is_dir():
        return checkout_assets
    return Path(user_data_path("dsle")) / "assets"


def _monitor_index(default: int = 0) -> int:
    value = os.environ.get(MONITOR_INDEX_ENV, str(default))
    try:
        index = int(value)
    except ValueError as exc:
        raise RuntimeUnavailableError(
            f"{MONITOR_INDEX_ENV} must be an integer, got {value!r}"
        ) from exc
    if index < 0:
        raise RuntimeUnavailableError(f"{MONITOR_INDEX_ENV} cannot be negative")
    return index


def _result_value(result: ReadResult[Any]) -> Any | None:
    return result.value if result.valid else None


def _memory_reports_boss_defeated(boss: BossConfig, memory: BossMemory) -> bool:
    """Apply darkSCOPE's boss-specific memory termination semantics.

    Four bosses have unreliable event flags in the ground-truth runner.  For
    those encounters a valid zero HP reading replaces the flag instead of
    being combined with it.  Victory templates are deliberately absent here:
    darkSCOPE uses them only to wait for a post-win screen during cleanup.
    """

    if boss.victory.hp_zero_is_victory:
        return bool(memory.hp.valid and memory.hp.value == 0)
    return bool(memory.defeated.valid and memory.defeated.value)


class LiveGameBackend(GameBackend):
    """Coordinate X11, capture, saves, templates, and v1.04 process memory."""

    def __init__(
        self,
        instance: str,
        *,
        asset_dir: str | Path | None = None,
        instance_config: str | Path | None = None,
        monitor_index: int = 0,
        verbose: bool = True,
        lifecycle: LifecycleLogger | None = None,
    ):
        game_dir = os.environ.get("DSLE_GAME_DIR")
        if not game_dir:
            raise RuntimeUnavailableError(
                "DSLE_GAME_DIR is required by the live backend so the v1.04 executable can be "
                "verified before process-memory writes are enabled"
            )
        self.game_build = verify_game_executable(Path(game_dir) / GAME_EXECUTABLE)
        self.instance = resolve_instance(instance, instance_config)
        self._lifecycle = lifecycle or LifecycleLogger(verbose)
        self.monitor_index = _monitor_index(monitor_index)
        self.asset_dir = (
            default_asset_dir()
            if asset_dir is None
            else Path(asset_dir).expanduser().resolve(strict=False)
        )
        self._templates = TemplateCatalog(self.asset_dir)
        self._memory = DSRMemory(self.instance.wineprefix)
        self._capture: MSSCapture | None = None
        self._input: X11Input | None = None
        self._save_manager: SaveManager | None = None
        self._executor = OperationExecutor(self, lifecycle=self._lifecycle)
        self._boss: BossConfig | None = None
        self._save_state = ""
        self._boss_hp_max: int | None = None
        self._boss_hp_maxes: tuple[int | None, ...] = ()
        self._death_count_start: int | None = None
        self._menu_state: str | None = None
        self._episode_ready = False
        self._closed = False
        self._diagnostics: dict[str, str] = {}
        self._lock = threading.RLock()

    @property
    def memory_diagnostics(self) -> Mapping[str, str]:
        """Expose reasons for the most recent invalid memory fields."""

        return MappingProxyType(dict(self._diagnostics))

    def set_event_sink(self, event_sink: EventSink | None) -> None:
        """Forward live setup, cleanup, and combat events during one RPC request."""

        self._lifecycle.set_event_sink(event_sink)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeCommunicationError("LiveGameBackend is closed")

    def _input_device(self) -> X11Input:
        self._require_open()
        if self._input is None:
            self._input = X11Input(self.instance.display)
        return self._input

    def _capture_device(self) -> MSSCapture:
        self._require_open()
        if self._capture is None:
            self._capture = MSSCapture(
                self.instance.display,
                monitor_index=self.monitor_index,
            )
        return self._capture

    def _focus(self) -> X11Input:
        controller = self._input_device()
        controller.focus_window(self.instance.desktop_name, allow_root_fallback=True)
        return controller

    def _frame(self) -> np.ndarray:
        return self._capture_device().capture()

    def _match(self, frame: np.ndarray, name: str, threshold: float) -> bool:
        matched, _result = self._templates.match(frame, name, threshold=threshold)
        return bool(matched)

    def _save_device(self) -> SaveManager:
        if self._save_manager is None:
            scenario_dir = resolve_scenario_directory(self.asset_dir)
            active_dir = self.instance.save_dir
            if active_dir is None:
                active_dir = discover_save_directory(self.instance.wineprefix)
            self._save_manager = SaveManager(scenario_dir, active_dir)
        return self._save_manager

    @staticmethod
    def _sleep_with_deadline(seconds: float, deadline: float, label: str) -> None:
        remaining = deadline - time.monotonic()
        if seconds > remaining:
            raise TimeoutError(
                f"{label} needs {seconds:.3f}s but only {max(0.0, remaining):.3f}s remain"
            )
        if seconds > 0:
            time.sleep(seconds)

    def reset(self, boss: BossConfig, save_state: str) -> RuntimeObservation:
        """Run the boss's declarative setup and return a validated initial observation."""

        with self._lock:
            self._require_open()
            self._episode_ready = False
            self._boss = boss
            self._save_state = str(save_state)
            self._boss_hp_max = None
            self._boss_hp_maxes = ()
            self._death_count_start = None
            self._menu_state = None
            self._diagnostics.clear()
            try:
                self._memory.attach()
                self._executor.execute(
                    boss.setup,
                    OperationContext(
                        boss=boss,
                        save_state=self._save_state,
                        phase="setup",
                        instance=self.instance.name,
                    ),
                    timeout_s=boss.limits.setup_timeout_s,
                )
                # Reloading a save keeps the PID but can invalidate every dynamic pointer.
                self._memory.attach()
                initial = self._observe(boss)
                if not initial.state.player_state_valid:
                    raise RuntimeCommunicationError(
                        f"Player memory is invalid after setup: {dict(self._diagnostics)}"
                    )
                if initial.state.boss_hp is None or not initial.state.boss_state_valid:
                    raise RuntimeCommunicationError(
                        f"Boss memory is invalid after setup: {dict(self._diagnostics)}"
                    )
                if initial.state.boss_hp <= 0:
                    raise RuntimeCommunicationError(
                        f"Boss HP is {initial.state.boss_hp} after setup; the encounter is not ready"
                    )
                self._death_count_start = initial.state.death_count
                # The ground-truth runner normalizes by HP observed at reset.
                # Metadata is descriptive and must never change reward dynamics.
                self._boss_hp_max = max(1, initial.state.boss_hp)
                state = self._read_state(boss, initial.frame)
                self._episode_ready = True
                return RuntimeObservation(
                    frame=initial.frame,
                    state=state,
                    timestamp=time.monotonic(),
                )
            except ResetError:
                raise
            except Exception as exc:
                raise ResetError(
                    f"Could not prepare {boss.boss_id!r} on {self.instance.name}: {exc}"
                ) from exc

    def perform(self, action: ActionSpec, hold_s: float) -> None:
        """Focus the configured Wine desktop and inject one policy action."""

        with self._lock:
            self._require_open()
            if not self._episode_ready:
                raise RuntimeCommunicationError("No boss episode is ready; call reset() first")
            self._focus().perform(action, hold_s)

    @staticmethod
    def _invalid_diagnostics(player: PlayerMemory, boss: BossMemory) -> dict[str, str]:
        results: dict[str, ReadResult[Any]] = {
            "player_hp": player.hp,
            "player_hp_max": player.hp_max,
            "death_count": player.death_count,
            "locked_on": player.locked_on,
            "boss_hp": boss.hp,
            "boss_defeated": boss.defeated,
        }
        return {
            name: result.error or "invalid read"
            for name, result in results.items()
            if not result.valid
        }

    def _read_state(self, boss_config: BossConfig, frame: np.ndarray) -> GameState:
        player = self._memory.read_player()
        boss = self._memory.read_boss(boss_config.memory)
        self._diagnostics = self._invalid_diagnostics(player, boss)

        player_hp = _result_value(player.hp)
        player_hp_max = _result_value(player.hp_max)
        death_count = _result_value(player.death_count)
        locked_on = _result_value(player.locked_on)
        player_x = _result_value(player.x)
        player_y = _result_value(player.y)
        player_z = _result_value(player.z)
        boss_hp = _result_value(boss.hp)
        hp_reads = boss.hp_values or (boss.hp,)
        boss_hps = tuple(
            None if not result.valid or result.value is None else int(result.value)
            for result in hp_reads
        )
        stored_maxes = list(getattr(self, "_boss_hp_maxes", ()))
        if len(stored_maxes) != len(boss_hps):
            stored_maxes = [None] * len(boss_hps)
        for index, hp_value in enumerate(boss_hps):
            if (
                hp_value is not None
                and hp_value > 0
                and (stored_maxes[index] is None or hp_value > stored_maxes[index])
            ):
                stored_maxes[index] = hp_value
        self._boss_hp_maxes = tuple(stored_maxes)
        boss_defeated = _memory_reports_boss_defeated(boss_config, boss)
        player_dead = bool(player.hp.valid and player_hp is not None and player_hp <= 0)
        if (
            player.death_count.valid
            and death_count is not None
            and self._death_count_start is not None
            and death_count > self._death_count_start
        ):
            player_dead = True
        player_valid = player.hp.valid and player.hp_max.valid
        boss_valid = boss.hp.valid or boss_defeated
        observed_max = int(boss_hp) if boss_hp is not None and boss_hp > 0 else 0
        boss_hp_max = self._boss_hp_max or max(1, observed_max)
        return GameState(
            player_hp=None if player_hp is None else int(player_hp),
            player_hp_max=None if player_hp_max is None else int(player_hp_max),
            boss_hp=None if boss_hp is None else int(boss_hp),
            boss_hp_max=boss_hp_max,
            death_count=None if death_count is None else int(death_count),
            locked_on=None if locked_on is None else bool(locked_on),
            boss_defeated=boss_defeated,
            player_dead=player_dead,
            player_state_valid=player_valid,
            boss_state_valid=boss_valid,
            boss_hps=boss_hps,
            boss_hp_maxes=self._boss_hp_maxes,
            player_x=None if player_x is None else float(player_x),
            player_y=None if player_y is None else float(player_y),
            player_z=None if player_z is None else float(player_z),
        )

    def _observe(self, boss: BossConfig) -> RuntimeObservation:
        frame = self._frame()
        state = self._read_state(boss, frame)
        return RuntimeObservation(frame=frame, state=state, timestamp=time.monotonic())

    def observe(self, boss: BossConfig) -> RuntimeObservation:
        """Capture RGB pixels and read validity-safe process state."""

        with self._lock:
            self._require_open()
            if not self._episode_ready:
                raise RuntimeCommunicationError("No boss episode is ready; call reset() first")
            if self._boss is not None and boss.boss_id != self._boss.boss_id:
                raise RuntimeCommunicationError(
                    f"Backend is running {self._boss.boss_id!r}, not {boss.boss_id!r}"
                )
            return self._observe(boss)

    def set_player_hp(self, boss: BossConfig, hp: int) -> RuntimeObservation:
        """Write player HP for the active episode and return an authoritative readback."""

        with self._lock:
            self._require_open()
            if not self._episode_ready:
                raise RuntimeCommunicationError("No boss episode is ready; call reset() first")
            if self._boss is None or boss.boss_id != self._boss.boss_id:
                raise RuntimeCommunicationError(
                    f"Backend is running {self._boss.boss_id if self._boss else None!r}, "
                    f"not {boss.boss_id!r}"
                )
            self._memory.write_player_hp(hp)
            return self._observe(boss)

    def set_boss_hp(
        self,
        boss: BossConfig,
        boss_number: int,
        hp: int,
    ) -> RuntimeObservation:
        """Write one or all boss HP locations and return an authoritative readback."""

        with self._lock:
            self._require_open()
            if not self._episode_ready:
                raise RuntimeCommunicationError("No boss episode is ready; call reset() first")
            if self._boss is None or boss.boss_id != self._boss.boss_id:
                raise RuntimeCommunicationError(
                    f"Backend is running {self._boss.boss_id if self._boss else None!r}, "
                    f"not {boss.boss_id!r}"
                )
            self._memory.write_boss_hp(boss.memory, boss_number, hp)
            return self._observe(boss)

    def finish(self, boss: BossConfig, outcome: EpisodeOutcome) -> None:
        """Execute outcome-conditioned cleanup operations."""

        with self._lock:
            self._require_open()
            try:
                self._executor.execute(
                    boss.cleanup,
                    OperationContext(
                        boss=boss,
                        save_state=self._save_state,
                        outcome=outcome,
                        phase="cleanup",
                        instance=self.instance.name,
                    ),
                    timeout_s=boss.limits.cleanup_timeout_s,
                )
            finally:
                self._episode_ready = False

    def return_to_menu(self, *, timeout_s: float = 30.0, threshold: float = 0.8) -> None:
        """Return the attached game to a verified title menu for manual development."""

        with self._lock:
            self._require_open()
            self._memory.attach()
            self.ensure_menu(timeout_s=timeout_s, threshold=threshold)
            self._episode_ready = False
            self._boss = None
            self._save_state = ""

    def close(self) -> None:
        """Close all live handles without changing container lifecycle."""

        with self._lock:
            if self._closed:
                return
            self._memory.close()
            if self._input is not None:
                self._input.close()
                self._input = None
            self._capture = None
            self._episode_ready = False
            self._closed = True

    # Primitive operations below intentionally mirror the names accepted by config.loader.

    def _scan_menu(self, frame: np.ndarray, threshold: float) -> str | None:
        for name in _MENU_ACTIONS:
            if self._match(frame, name, threshold):
                return name
        return None

    def _accept_privacy(self, deadline: float, threshold: float) -> None:
        controller = self._focus()
        decision_threshold = max(0.9, threshold)
        while time.monotonic() < deadline:
            frame = self._frame()
            if self._match(frame, "privacyPolicyDecision", decision_threshold):
                controller.tap_sequence(("left", "e"), hold_s=0.05, interval_s=0.1)
                return
            controller.tap_key("Next", 0.05)
            self._sleep_with_deadline(0.06, deadline, "privacy policy navigation")
        raise TimeoutError("Timed out while scrolling the privacy policy")

    def ensure_menu(self, *, timeout_s: float, threshold: float) -> None:
        """Reach a main-menu Continue/New screen using templates and v1.04 menu memory."""

        deadline = time.monotonic() + timeout_s
        controller = self._focus()
        # Player memory is a stronger in-game signal than a full-frame template
        # match. Dark gameplay can otherwise resemble one of the black splash
        # screens closely enough to create a false positive.
        player = self._memory.read_player()
        if player.hp.valid:
            self._memory.return_to_title()
            self._sleep_with_deadline(1.0, deadline, "return to title")
        else:
            frame = self._frame()
            visible = self._scan_menu(frame, threshold)
            if visible in {"menuOptionsContinue", "menuOptionsNew"}:
                self._menu_state = "continue" if visible.endswith("Continue") else "new"
                return

        unmatched = 0
        while time.monotonic() < deadline:
            # Repeat the memory request if the game ignored it during a frame
            # transition. Once the player pointer disappears, title-screen
            # navigation can safely rely on templates.
            player = self._memory.read_player()
            if player.hp.valid:
                self._memory.return_to_title()
                unmatched = 0
                self._sleep_with_deadline(0.5, deadline, "return to title")
                continue
            frame = self._frame()
            visible = self._scan_menu(frame, threshold)
            if visible == "menuOptionsContinue":
                self._menu_state = "continue"
                return
            if visible == "menuOptionsNew":
                self._menu_state = "new"
                return
            if visible == "privacyPolicy":
                self._accept_privacy(deadline, threshold)
                unmatched = 0
            elif visible is not None:
                keys = _MENU_ACTIONS[visible]
                if keys:
                    controller.tap_sequence(keys, hold_s=0.05, interval_s=0.1)
                unmatched = 0
            else:
                unmatched += 1
                if unmatched % 8 == 0:
                    fallback = "esc" if (unmatched // 8) % 2 == 0 else "e"
                    controller.tap_key(fallback, 0.05)
            self._sleep_with_deadline(0.3, deadline, "main-menu navigation")
        raise TimeoutError("Timed out before reaching the Dark Souls main menu")

    def load_save(
        self,
        save_state: str,
        *,
        interact: bool,
        settle_before_s: float,
        settle_after_s: float,
        timeout_s: float,
    ) -> None:
        """Atomically swap the active save and enter it from the main menu."""

        deadline = time.monotonic() + timeout_s
        self._save_device().load(save_state)
        self._sleep_with_deadline(settle_before_s, deadline, "save-state swap")
        if self._menu_state == "new":
            # The menu was rendered before a save existed; force it to rescan the new active slot.
            self._memory.return_to_title()
            self._sleep_with_deadline(1.0, deadline, "main-menu save refresh")
            self.ensure_menu(timeout_s=max(0.001, deadline - time.monotonic()), threshold=0.8)
            if self._menu_state != "continue":
                raise RuntimeCommunicationError(
                    "The main menu still offers New Game after installing the scenario save"
                )
        if interact:
            self.tap_key("e", hold_s=0.05)
        self._sleep_with_deadline(settle_after_s, deadline, "save-state load")

    def wait_until_ready(
        self, boss: BossConfig, params: Mapping[str, Any], *, timeout_s: float
    ) -> None:
        """Wait for, or walk toward, the configured encounter-ready template."""

        mode = str(params.get("mode", boss.readiness.mode)).strip()
        if mode == "none":
            return
        template = str(params.get("template", boss.readiness.template or "")).strip()
        if not template:
            raise ValueError("wait_until_ready requires a template for this readiness mode")
        threshold = float(params.get("threshold", boss.readiness.threshold))
        poll_s = float(params.get("poll_s", boss.readiness.poll_s))
        advance_key = str(params.get("advance_key", boss.readiness.advance_key))
        advance_hold_s = float(params.get("advance_hold_s", boss.readiness.advance_hold_s))
        interact = bool(params.get("interact_after_ready", boss.readiness.interact_after_ready))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._match(self._frame(), template, threshold):
                if interact:
                    self.tap_key("e", hold_s=0.05)
                return
            if mode == "traverse_light":
                self.tap_key(advance_key, hold_s=advance_hold_s)
            elif mode != "template":
                raise ValueError(f"Unsupported readiness mode: {mode!r}")
            self._sleep_with_deadline(poll_s, deadline, "encounter readiness")
        raise TimeoutError(f"Timed out waiting for ready template {template!r}")

    def set_flag(self, name: str, enabled: bool) -> None:
        self._memory.set_flag(name, enabled)

    def teleport_player(self, *, x: float | None, y: float | None, z: float | None) -> None:
        self._memory.teleport(x=x, y=y, z=z)

    def tap_key(self, key: str, *, hold_s: float) -> None:
        self._focus().tap_key(key, hold_s)

    def hold_key(self, key: str, *, duration_s: float) -> None:
        self._focus().hold_key_for(key, duration_s)

    def walk_until_template(self, params: Mapping[str, Any], *, timeout_s: float) -> None:
        template = str(params.get("template", "")).strip()
        if not template:
            raise ValueError("walk_until_template requires 'template'")
        key = str(params.get("key", "w"))
        hold_s = float(params.get("hold_s", 0.15))
        poll_s = float(params.get("poll_s", 0.2))
        threshold = float(params.get("threshold", 0.9))
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.tap_key(key, hold_s=hold_s)
            if self._match(self._frame(), template, threshold):
                return
            self._sleep_with_deadline(poll_s, deadline, f"walk toward {template}")
        raise TimeoutError(f"Timed out walking toward template {template!r}")

    @staticmethod
    def _configured_action(entry: Any, default_hold_s: float) -> tuple[ActionSpec, float]:
        if isinstance(entry, str):
            name = entry.strip()
            hold_s = default_hold_s
        elif isinstance(entry, Mapping):
            name = str(entry.get("name", entry.get("action", ""))).strip()
            hold_s = float(entry.get("hold_s", default_hold_s))
        else:
            raise ValueError("repeat_actions entries must be action names or mappings")
        try:
            spec = _ACTION_BY_NAME[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown repeated action {name!r}; choose from {sorted(_ACTION_BY_NAME)}"
            ) from exc
        if hold_s < 0:
            raise ValueError("repeat_actions hold_s values cannot be negative")
        return spec, hold_s

    def repeat_actions(
        self,
        actions: Sequence[Any],
        *,
        duration_s: float,
        hold_s: float,
        interval_s: float,
        simultaneous: bool,
    ) -> None:
        """Repeat named policy actions, optionally combining them into one input chord."""

        configured = tuple(self._configured_action(entry, hold_s) for entry in actions)
        controller = self._focus()
        deadline = time.monotonic() + duration_s
        if simultaneous:
            keys = tuple(dict.fromkeys(key for spec, _ in configured for key in spec.keys))
            mouse_buttons = {spec.mouse for spec, _ in configured if spec.mouse is not None}
            if len(mouse_buttons) > 1:
                raise ValueError("A simultaneous action cannot press both mouse buttons")
            mouse = next(iter(mouse_buttons), None)
            combined = ActionSpec(name="simultaneous_setup_action", keys=keys, mouse=mouse)
            cycle_hold = max((entry_hold for _spec, entry_hold in configured), default=hold_s)
            while time.monotonic() < deadline:
                controller.perform(combined, min(cycle_hold, deadline - time.monotonic()))
                if interval_s and time.monotonic() < deadline:
                    time.sleep(min(interval_s, deadline - time.monotonic()))
            return
        while time.monotonic() < deadline:
            for spec, entry_hold in configured:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                controller.perform(spec, min(entry_hold, remaining))
            if interval_s and time.monotonic() < deadline:
                time.sleep(min(interval_s, deadline - time.monotonic()))

    def wait_for_victory(self, boss: BossConfig, *, timeout_s: float, threshold: float) -> None:
        """Wait for any configured post-boss reward or victory template."""

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            name, _result = self._templates.match_any(
                self._frame(), boss.victory.templates, threshold=threshold
            )
            if name is not None:
                return
            self._sleep_with_deadline(0.2, deadline, "victory screen")
        raise TimeoutError(f"Timed out waiting for a victory screen for {boss.boss_id!r}")

    def return_to_title(self) -> None:
        self._memory.return_to_title()
