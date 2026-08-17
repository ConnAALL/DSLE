"""The public Gymnasium environment implementation for DSLE."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec

from dsle.actions import ACTION_SPECS, LOCK_ON, action_spec
from dsle.config import ConfigRepository
from dsle.exceptions import ResetError, RuntimeCommunicationError, RuntimeUnavailableError
from dsle.models import (
    BossConfig,
    EpisodeOutcome,
    GameState,
    RewardConfig,
    RuntimeObservation,
)
from dsle.progress import EventSink, LifecycleLogger
from dsle.rewards import RewardBreakdown, transition_reward
from dsle.runtime.base import GameBackend

ObservationMode = Literal["grayscale", "rgb", "state_only"]
BackendFactory = Callable[[str, Path | None, Path | None], GameBackend]


def _integer_option(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    selected = int(value)
    if selected < minimum:
        comparison = "non-negative" if minimum == 0 else f"at least {minimum}"
        raise ValueError(f"{name} must be {comparison}")
    return selected


def _default_backend_factory(
    instance: str,
    asset_dir: Path | None,
    instance_config: Path | None,
    *,
    monitor_index: int,
    verbose: bool,
    lifecycle: LifecycleLogger,
) -> GameBackend:
    try:
        from dsle.runtime.live import LiveGameBackend
    except ImportError as exc:  # pragma: no cover - depends on optional runtime extra
        raise RuntimeUnavailableError(
            "The live backend needs DSLE's runtime dependencies. Install with `pip install -e '.[runtime]'`."
        ) from exc
    return LiveGameBackend(
        instance,
        asset_dir=asset_dir,
        instance_config=instance_config,
        monitor_index=monitor_index,
        verbose=verbose,
        lifecycle=lifecycle,
    )


class DarkSoulsEnv(gym.Env[np.ndarray, int]):
    """One Dark Souls: Remastered boss encounter.

    Agent-side seeds control Gymnasium sampling only. The external game process
    has unseeded internal AI and animation randomness, so equal seeds and action
    sequences do not make trajectories deterministic.
    """

    def __init__(
        self,
        boss: str,
        *,
        instance: str = "dsr-1",
        difficulty: str = "standard",
        obs_mode: ObservationMode = "grayscale",
        action_ms: int | None = None,
        action_repeat: int = 1,
        max_steps: int | None = None,
        auto_lock_on: bool | None = None,
        lock_on_interval: int | None = None,
        reward: RewardConfig | None = None,
        render_mode: Literal["rgb_array"] | None = None,
        config_dir: str | Path | None = None,
        asset_dir: str | Path | None = None,
        instance_config: str | Path | None = None,
        backend: GameBackend | None = None,
        backend_factory: BackendFactory | None = None,
        verbose: bool = True,
    ):
        super().__init__()
        self._lifecycle = LifecycleLogger(verbose)
        self.verbose = verbose
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 4}
        self._lifecycle.check(f"Loading boss configuration: {boss}")
        repository = ConfigRepository(config_dir)
        self.boss: BossConfig = repository.get(boss)
        self.spec = EnvSpec(
            id=f"dsle/{self.boss.boss_id}",
            entry_point="dsle.env:DarkSoulsEnv",
            max_episode_steps=None,
            nondeterministic=True,
            kwargs={"boss": self.boss.boss_id},
        )
        if not isinstance(instance, str) or not instance.strip():
            raise ValueError("instance must be a non-empty string")
        if not isinstance(difficulty, str) or not difficulty.strip():
            raise ValueError("difficulty must be a non-empty string")
        if auto_lock_on is not None and not isinstance(auto_lock_on, bool):
            raise ValueError("auto_lock_on must be a boolean or None")
        if reward is not None and not isinstance(reward, RewardConfig):
            raise TypeError("reward must be a RewardConfig or None")
        self.instance = instance.strip()
        self.difficulty = difficulty.strip()
        self.obs_mode = obs_mode
        self.render_mode = render_mode
        self.action_ms = (
            repository.defaults.action_ms
            if action_ms is None
            else _integer_option(action_ms, "action_ms", minimum=0)
        )
        self.action_repeat = _integer_option(action_repeat, "action_repeat", minimum=1)
        self.max_steps = (
            self.boss.limits.max_steps
            if max_steps is None
            else _integer_option(max_steps, "max_steps", minimum=1)
        )
        self.auto_lock_on = (
            repository.defaults.auto_lock_on if auto_lock_on is None else auto_lock_on
        )
        self.lock_on_interval = (
            repository.defaults.lock_on_interval
            if lock_on_interval is None
            else _integer_option(lock_on_interval, "lock_on_interval", minimum=1)
        )
        self.reward_config = self.boss.reward if reward is None else reward
        self.frame_height = repository.defaults.frame_height
        self.frame_width = repository.defaults.frame_width

        if obs_mode not in {"grayscale", "rgb", "state_only"}:
            raise ValueError(f"Unknown obs_mode {obs_mode!r}")
        if render_mode not in {None, "rgb_array"}:
            raise ValueError("render_mode must be None or 'rgb_array'")
        self.boss.save_state(self.difficulty)  # validate before allocating a backend

        self.action_space: spaces.Discrete = spaces.Discrete(len(ACTION_SPECS))
        if obs_mode == "grayscale":
            self.observation_space = spaces.Box(
                0, 255, (1, self.frame_height, self.frame_width), np.uint8
            )
        elif obs_mode == "rgb":
            self.observation_space = spaces.Box(
                0, 255, (3, self.frame_height, self.frame_width), np.uint8
            )
        else:
            self.observation_space = spaces.Box(0.0, 1.0, (2,), np.float32)

        if backend is not None and backend_factory is not None:
            raise ValueError("Pass backend or backend_factory, not both")
        assets = None if asset_dir is None else Path(asset_dir).expanduser().resolve()
        instances = (
            None if instance_config is None else Path(instance_config).expanduser().resolve()
        )
        if backend is not None:
            self._backend = backend
        elif backend_factory is not None:
            self._backend = backend_factory(self.instance, assets, instances)
        else:
            self._backend = _default_backend_factory(
                self.instance,
                assets,
                instances,
                monitor_index=repository.defaults.monitor_index,
                verbose=verbose,
                lifecycle=self._lifecycle,
            )

        self._last_runtime_observation: RuntimeObservation | None = None
        self._last_observation: np.ndarray | None = None
        self._step_count = 0
        self._victory_streak = 0
        self._episode_active = False
        self._episode_start = 0.0
        self._closed = False
        self._lifecycle.ready(
            f"Environment configured: boss={self.boss.boss_id}, "
            f"difficulty={self.difficulty}, instance={self.instance}"
        )

    def set_event_sink(self, event_sink: EventSink | None) -> None:
        """Forward verbose events while a remote API request is in progress."""

        self._lifecycle.set_event_sink(event_sink)
        backend_setter = getattr(self._backend, "set_event_sink", None)
        if callable(backend_setter):
            backend_setter(event_sink)

    @staticmethod
    def _health(value: int | None, maximum: int | None, *, valid: bool) -> str:
        if not valid or value is None:
            return "unavailable"
        if maximum is None:
            return str(value)
        return f"{value}/{maximum}"

    def _report_combat_state(self, state: GameState) -> None:
        """Report the policy-visible health and terminal signals for this instance."""

        player = self._health(
            state.player_hp,
            state.player_hp_max,
            valid=state.player_state_valid,
        )
        boss_hps = state.boss_hps or (state.boss_hp,)
        boss_hp_maxes = state.boss_hp_maxes or (state.boss_hp_max,)
        boss_health = " ".join(
            f"boss{index + 1}_hp="
            + self._health(
                hp,
                boss_hp_maxes[index] if index < len(boss_hp_maxes) else None,
                valid=state.boss_state_valid,
            )
            for index, hp in enumerate(boss_hps)
        )
        self._lifecycle.event(
            "COMBAT",
            f"{self.instance} step={self._step_count} player_hp={player} "
            f"player_location={state.player_location} {boss_health} "
            f"boss_defeated={'yes' if state.boss_defeated else 'no'} "
            f"player_dead={'yes' if state.player_dead else 'no'}",
        )

    def _policy_observation(self, runtime: RuntimeObservation) -> np.ndarray:
        if self.obs_mode == "state_only":
            return np.asarray(
                [runtime.state.player_hp_fraction, runtime.state.boss_hp_fraction],
                dtype=np.float32,
            )
        frame = np.asarray(runtime.frame)
        expected = (self.frame_height, self.frame_width, 3)
        if frame.shape != expected:
            raise RuntimeCommunicationError(
                f"Runtime returned frame {frame.shape}; expected RGB {expected}"
            )
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if self.obs_mode == "rgb":
            return np.transpose(frame, (2, 0, 1)).copy()
        # Integer approximation of ITU-R BT.601, applied to RGB input.
        gray = (
            frame[..., 0].astype(np.uint16) * 77
            + frame[..., 1].astype(np.uint16) * 150
            + frame[..., 2].astype(np.uint16) * 29
        ) >> 8
        return gray.astype(np.uint8, copy=False)[None, ...]

    def _info(
        self,
        state: GameState,
        breakdown: RewardBreakdown | None = None,
        *,
        terminated_reason: str | None = None,
        truncated_reason: str | None = None,
    ) -> dict[str, Any]:
        boss_damage = 0 if breakdown is None else breakdown.boss_damage
        player_damage = 0 if breakdown is None else breakdown.player_damage
        return {
            "boss": self.boss.boss_id,
            "instance": self.instance,
            "difficulty": self.difficulty,
            "step": self._step_count,
            "player_hp": state.player_hp,
            "player_hp_max": state.player_hp_max,
            "player_location": state.player_location,
            "boss_hp": state.boss_hp,
            "boss_hp_max": state.boss_hp_max,
            "boss_hps": list(state.boss_hps or (state.boss_hp,)),
            "boss_hp_maxes": list(state.boss_hp_maxes or (state.boss_hp_max,)),
            "death_count": state.death_count,
            "locked_on": state.locked_on,
            "player_state_valid": state.player_state_valid,
            "boss_state_valid": state.boss_state_valid,
            "boss_damage_dealt": boss_damage,
            "player_damage_taken": player_damage,
            "boss_defeated": state.boss_defeated,
            "player_dead": state.player_dead,
            "win": terminated_reason == "boss_defeated",
            "terminated_reason": terminated_reason,
            "truncated_reason": truncated_reason,
            "elapsed_seconds": max(0.0, time.monotonic() - self._episode_start),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self._closed:
            raise RuntimeError("Cannot reset a closed environment")
        if options is not None and not isinstance(options, Mapping):
            raise ValueError("reset options must be a mapping or None")
        selected_options = dict(options or {})
        unknown_options = sorted(set(selected_options) - {"difficulty"})
        if unknown_options:
            raise ValueError(f"Unknown reset option(s): {', '.join(unknown_options)}")
        difficulty = selected_options.get("difficulty", self.difficulty)
        if not isinstance(difficulty, str) or not difficulty.strip():
            raise ValueError("reset difficulty must be a non-empty string")
        difficulty = difficulty.strip()
        save_state = self.boss.save_state(difficulty)
        self._lifecycle.start(f"Loading {self.boss.boss_id} ({difficulty}) on {self.instance}")
        try:
            runtime = self._backend.reset(self.boss, save_state)
            self._validate_initial_state(runtime.state)
            # The ground-truth runner enters every encounter locked on.  Only
            # an explicit False triggers input; an invalid lock-on read (None)
            # must never be converted into a speculative key press.
            if self.auto_lock_on and runtime.state.locked_on is False:
                self._backend.perform(LOCK_ON, 0.1)
                runtime = self._backend.observe(self.boss)
                self._validate_initial_state(runtime.state)
            observation = self._policy_observation(runtime)
        except Exception as exc:
            self._episode_active = False
            self._lifecycle.error(f"Could not start {self.boss.boss_id} on {self.instance}: {exc}")
            if isinstance(exc, ResetError):
                raise
            raise ResetError(
                f"Failed to reset '{self.boss.boss_id}' on {self.instance}: {exc}"
            ) from exc

        self.difficulty = difficulty
        self._last_runtime_observation = runtime
        self._last_observation = observation
        self._step_count = 0
        self._victory_streak = 0
        self._episode_active = True
        self._episode_start = time.monotonic()
        self._lifecycle.ready(
            f"Fight running: {self.boss.boss_id} ({difficulty}) on {self.instance}"
        )
        self._report_combat_state(runtime.state)
        return observation, self._info(runtime.state)

    @staticmethod
    def _validate_initial_state(initial: GameState) -> None:
        if not initial.player_state_valid:
            raise ResetError("Initial player state is invalid")
        if initial.player_hp is None or initial.player_hp_max is None:
            raise ResetError("Initial player HP and maximum HP must be available")
        if initial.player_hp <= 0 or initial.player_hp_max <= 0 or initial.player_dead:
            raise ResetError("Initial player must be alive with positive HP and maximum HP")
        if not initial.boss_state_valid:
            raise ResetError("Initial boss state is invalid")
        if initial.boss_defeated:
            raise ResetError("Initial boss state is already defeated")
        if initial.boss_hp is None or initial.boss_hp_max is None:
            raise ResetError("Initial boss HP and maximum HP must be available")
        if initial.boss_hp <= 0 or initial.boss_hp_max <= 0:
            raise ResetError("Initial boss HP and maximum HP must be positive")

    def _terminal_reason(self, state: GameState) -> str | None:
        victory_signal = state.boss_state_valid and state.boss_defeated
        self._victory_streak = self._victory_streak + 1 if victory_signal else 0
        if (
            self._step_count >= self.boss.limits.minimum_victory_step
            and self._victory_streak >= self.boss.limits.victory_confirmations
        ):
            return "boss_defeated"
        # darkSCOPE records a victory when the boss and player die on the same
        # transition; keep that ordering once the victory signal is verified.
        if state.player_dead:
            return "player_dead"
        return None

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if (
            not self._episode_active
            or self._last_runtime_observation is None
            or self._last_observation is None
        ):
            raise RuntimeError(
                "reset() must be called before step(), and after every completed episode"
            )
        selected_action = action_spec(action)

        previous = self._last_runtime_observation
        current = previous
        observation = self._last_observation
        terminated_reason: str | None = None
        truncated_reason: str | None = None
        runtime_error: str | None = None
        repeated_rewards: list[RewardBreakdown] = []
        repeats = min(self.action_repeat, self.max_steps - self._step_count)
        try:
            for _ in range(repeats):
                if (
                    self.auto_lock_on
                    and self._step_count > 0
                    and self._step_count % self.lock_on_interval == 0
                    and previous.state.locked_on is False
                ):
                    self._backend.perform(LOCK_ON, self.action_ms / 1000.0)
                self._backend.perform(selected_action, self.action_ms / 1000.0)
                self._step_count += 1
                candidate = self._backend.observe(self.boss)
                # Treat observation conversion as part of runtime I/O. A bad
                # frame must follow the same diagnostic-truncation path as a
                # failed capture, rather than escaping after an action has
                # already advanced the external game.
                candidate_observation = self._policy_observation(candidate)
                candidate_reward = transition_reward(
                    previous.state,
                    candidate.state,
                    self.reward_config,
                )
                current = candidate
                observation = candidate_observation
                repeated_rewards.append(candidate_reward)
                terminated_reason = self._terminal_reason(current.state)
                if terminated_reason is not None:
                    break
                previous = current
        except Exception as exc:
            truncated_reason = "runtime_error"
            runtime_error = f"{type(exc).__name__}: {exc}"

        if truncated_reason is not None:
            terminated_reason = None
        if (
            terminated_reason is None
            and truncated_reason is None
            and self._step_count >= self.max_steps
        ):
            truncated_reason = "max_steps"
        # Action repeat is a convenience over multiple real transitions, so it
        # must not alter reward semantics: accumulate every damage delta and
        # every per-step penalty, then apply the terminal component once.
        if not repeated_rewards:
            repeated_rewards.append(
                transition_reward(
                    self._last_runtime_observation.state,
                    current.state,
                    self.reward_config,
                )
            )
        terminal = transition_reward(
            current.state,
            current.state,
            self.reward_config,
            terminated_reason=terminated_reason,
            truncated_reason=truncated_reason,
        )
        breakdown = RewardBreakdown(
            boss_damage=sum(item.boss_damage for item in repeated_rewards),
            player_damage=sum(item.player_damage for item in repeated_rewards),
            boss_damage_reward=sum(item.boss_damage_reward for item in repeated_rewards),
            player_damage_reward=sum(item.player_damage_reward for item in repeated_rewards),
            step_reward=sum(item.step_reward for item in repeated_rewards),
            terminal_reward=terminal.terminal_reward,
        )
        terminated = terminated_reason is not None
        truncated = truncated_reason is not None
        info = self._info(
            current.state,
            breakdown,
            terminated_reason=terminated_reason,
            truncated_reason=truncated_reason,
        )
        if runtime_error:
            info["runtime_error"] = runtime_error

        self._report_combat_state(current.state)

        self._last_runtime_observation = current
        self._last_observation = observation
        if terminated or truncated:
            self._episode_active = False
            outcome = EpisodeOutcome(
                win=terminated_reason == "boss_defeated",
                terminated=terminated,
                truncated=truncated,
                reason=terminated_reason or truncated_reason or "unknown",
                steps=self._step_count,
            )
            try:
                self._backend.finish(self.boss, outcome)
            except Exception as exc:  # cleanup must not invalidate a completed transition
                info["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            self._lifecycle.done(
                f"Episode finished on {self.instance}: {outcome.reason} "
                f"after {outcome.steps} step(s)"
            )

        return observation, breakdown.total, terminated, truncated, info

    def observe(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Passively refresh observation and state without injecting a policy action.

        This manual-control extension does not advance Gymnasium's step counter or
        automatically finish the episode. It is intended for monitoring, recording,
        and other human-facing tools; learning agents should continue to use ``step``.
        """

        if (
            not self._episode_active
            or self._last_runtime_observation is None
            or self._last_observation is None
        ):
            raise RuntimeError("reset() must be called before observe()")
        runtime = self._backend.observe(self.boss)
        observation = self._policy_observation(runtime)
        self._last_runtime_observation = runtime
        self._last_observation = observation
        self._report_combat_state(runtime.state)
        return observation, self._info(runtime.state)

    def return_to_menu(self, *, timeout_s: float = 30.0) -> None:
        """End manual control and return the live game to a verified title menu."""

        if self._closed:
            raise RuntimeError("Cannot return to the menu from a closed environment")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, Real):
            raise ValueError("timeout_s must be a finite positive number")
        selected_timeout = float(timeout_s)
        if not math.isfinite(selected_timeout) or selected_timeout <= 0:
            raise ValueError("timeout_s must be a finite positive number")
        self._lifecycle.start(f"Returning {self.instance} to the title menu")
        self._backend.return_to_menu(timeout_s=selected_timeout)
        self._episode_active = False
        self._lifecycle.ready(f"{self.instance} is at the title menu")

    def _active_state_for_write(self) -> GameState:
        if not self._episode_active or self._last_runtime_observation is None:
            raise RuntimeError("reset() must be called before writing health")
        return self._last_runtime_observation.state

    @staticmethod
    def _validated_health(value: object, label: str, maximum: int) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{label} must be an integer")
        selected = int(value)
        if not 0 <= selected <= maximum:
            raise ValueError(f"{label} must be in [0, {maximum}]")
        return selected

    def _accept_health_write(self, runtime: RuntimeObservation) -> dict[str, Any]:
        observation = self._policy_observation(runtime)
        self._last_runtime_observation = runtime
        self._last_observation = observation
        self._report_combat_state(runtime.state)
        return self._info(runtime.state)

    def set_player_hp(self, hp: int) -> dict[str, Any]:
        """Set current player HP during an active episode and return refreshed info."""

        state = self._active_state_for_write()
        if not state.player_state_valid or state.player_hp_max is None:
            raise RuntimeCommunicationError("Player HP is not currently writable")
        selected = self._validated_health(hp, "player HP", state.player_hp_max)
        self._lifecycle.event("WRITE", f"{self.instance} setting player_hp={selected}")
        runtime = self._backend.set_player_hp(self.boss, selected)
        return self._accept_health_write(runtime)

    def set_boss_hp(self, boss_number: int, hp: int) -> dict[str, Any]:
        """Set one-based ``bossN`` HP, or use ``(-1, 0)`` to zero every boss HP."""

        state = self._active_state_for_write()
        if isinstance(boss_number, bool) or not isinstance(boss_number, Integral):
            raise ValueError("boss_number must be an integer")
        selected_number = int(boss_number)
        if isinstance(hp, bool) or not isinstance(hp, Integral):
            raise ValueError("boss HP must be an integer")
        if self.boss.memory.hp_mode == "defeated_flag":
            raise RuntimeCommunicationError(
                "Boss HP is derived from the defeated flag and is read-only"
            )
        count = len(self.boss.memory.hp_chains)
        if selected_number == -1:
            if isinstance(hp, bool) or not isinstance(hp, Integral) or int(hp) != 0:
                raise ValueError("boss_number=-1 is only valid with hp=0")
            health_values = state.boss_hps or (state.boss_hp,)
            if len(health_values) != count or any(value is None for value in health_values):
                raise RuntimeCommunicationError(
                    "All configured boss HP locations must be readable before a kill-all write"
                )
            self._lifecycle.event("WRITE", f"{self.instance} setting all_boss_hp=0")
            runtime = self._backend.set_boss_hp(self.boss, -1, 0)
            return self._accept_health_write(runtime)
        if not 1 <= selected_number <= count:
            raise ValueError(f"boss_number must be in [1, {count}], or -1 with hp=0")
        health_values = state.boss_hps or (state.boss_hp,)
        maximums = state.boss_hp_maxes or (state.boss_hp_max,)
        if selected_number > len(health_values) or selected_number > len(maximums):
            raise RuntimeCommunicationError(f"boss{selected_number} health metadata is unavailable")
        current = health_values[selected_number - 1]
        maximum = maximums[selected_number - 1]
        if isinstance(hp, bool) or not isinstance(hp, Integral):
            raise ValueError(f"boss{selected_number} HP must be an integer")
        requested_hp = int(hp)
        if current is None or (maximum is None and requested_hp != 0):
            raise RuntimeCommunicationError(f"boss{selected_number} HP is not currently writable")
        selected = self._validated_health(
            requested_hp,
            f"boss{selected_number} HP",
            0 if maximum is None else maximum,
        )
        self._lifecycle.event(
            "WRITE",
            f"{self.instance} setting boss{selected_number}_hp={selected}",
        )
        runtime = self._backend.set_boss_hp(self.boss, selected_number, selected)
        return self._accept_health_write(runtime)

    def render(self) -> np.ndarray | None:
        if self.render_mode != "rgb_array" or self._last_runtime_observation is None:
            return None
        frame = np.asarray(self._last_runtime_observation.frame)
        expected = (self.frame_height, self.frame_width, 3)
        if frame.shape != expected:
            raise RuntimeCommunicationError(
                f"Runtime returned render frame {frame.shape}; expected RGB {expected}"
            )
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame.copy()

    def close(self) -> None:
        if self._closed:
            return
        self._backend.close()
        self._episode_active = False
        self._closed = True
        self._lifecycle.done(f"Environment closed: {self.instance}")


def make(
    boss: str,
    *,
    runtime: Literal["auto", "managed", "external", "local"] = "auto",
    game_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    image: str | None = None,
    container_name: str | None = None,
    num_instances: int = 1,
    socket_path: str | Path | None = None,
    token_file: str | Path | None = None,
    start_instance: bool | None = None,
    instance_mode: str = "headless",
    verbose: bool = True,
    **kwargs: Any,
) -> gym.Env[np.ndarray, int] | gym.vector.VectorEnv:
    """Create a local, managed-container, or external-server DSLE environment.

    ``auto`` keeps dependency-injected tests and in-container calls local. On
    a normal host it starts an owned prebuilt container. ``num_instances > 1``
    returns a synchronous vector environment over ``dsr-1..dsr-N``. Managed
    returns own their container: closing them stops and removes it while
    preserving the host ``output_dir`` and prebuilt image. External and local
    returns do not own a container. Managed mode checks the current working
    directory for a recognized game folder when no game path is configured.
    """

    if not isinstance(runtime, str) or runtime not in {"auto", "managed", "external", "local"}:
        raise ValueError("runtime must be auto, managed, external, or local")
    if start_instance is not None and not isinstance(start_instance, bool):
        raise ValueError("start_instance must be a boolean or None")
    if not isinstance(verbose, bool):
        raise ValueError("verbose must be a boolean")
    if instance_mode not in {"headless", "headless-vnc", "gui"}:
        raise ValueError("instance_mode must be headless, headless-vnc, or gui")
    if (
        isinstance(num_instances, bool)
        or not isinstance(num_instances, int)
        or not 1 <= num_instances <= 30
    ):
        raise ValueError("num_instances must be an integer in [1, 30]")
    has_backend = kwargs.get("backend") is not None or kwargs.get("backend_factory") is not None
    if kwargs.get("backend") is None:
        kwargs.pop("backend", None)
    if kwargs.get("backend_factory") is None:
        kwargs.pop("backend_factory", None)
    selected_runtime = runtime
    if selected_runtime == "auto":
        selected_runtime = (
            "local" if os.environ.get("DSLE_CONTAINER") == "1" or has_backend else "managed"
        )
    lifecycle = LifecycleLogger(verbose)
    lifecycle.info(f"Runtime selected: {selected_runtime}; boss={boss}; instances={num_instances}")
    should_start = selected_runtime != "local" if start_instance is None else start_instance

    managed_options = {
        "game_dir": game_dir,
        "output_dir": output_dir,
        "image": image,
        "container_name": container_name,
    }
    external_options = {"socket_path": socket_path, "token_file": token_file}

    if selected_runtime == "local":
        if instance_mode != "headless":
            raise ValueError("Local runtime does not manage instance_mode")
        invalid = [
            name
            for name, value in {**managed_options, **external_options}.items()
            if value is not None
        ]
        if invalid:
            raise ValueError(
                f"Local runtime does not accept selector option(s): {', '.join(invalid)}"
            )
        if should_start:
            raise ValueError(
                "Local runtime does not own instance lifecycle; start it with dsle-instances first"
            )
        if num_instances == 1:
            return DarkSoulsEnv(boss, verbose=verbose, **kwargs)
        if has_backend:
            raise ValueError("num_instances > 1 cannot share an injected backend")
        if "instance" in kwargs and kwargs["instance"] != "dsr-1":
            raise ValueError(
                "num_instances > 1 uses dsr-1 through dsr-N; remove the instance option"
            )
        kwargs.pop("instance", None)
        from dsle.vector import make_vector_env

        return make_vector_env(
            boss,
            num_instances,
            asynchronous=False,
            verbose=verbose,
            **kwargs,
        )

    if has_backend:
        raise ValueError(
            f"backend/backend_factory cannot be used with runtime={selected_runtime!r}"
        )
    unsafe_remote_paths = sorted(
        name
        for name in ("config_dir", "asset_dir", "instance_config")
        if kwargs.get(name) is not None
    )
    if unsafe_remote_paths:
        raise ValueError(
            "Container RPC owns runtime paths; remove option(s): " + ", ".join(unsafe_remote_paths)
        )
    reward = kwargs.get("reward")
    if isinstance(reward, RewardConfig):
        kwargs = dict(kwargs)
        kwargs["reward"] = {
            name: getattr(reward, name) for name in RewardConfig.__dataclass_fields__
        }

    if selected_runtime == "managed":
        invalid = [name for name, value in external_options.items() if value is not None]
        if invalid:
            raise ValueError(
                f"Managed runtime does not accept selector option(s): {', '.join(invalid)}"
            )
        from dsle.container import make_managed

        return make_managed(
            boss,
            game_dir=game_dir,
            output_dir=output_dir,
            image=image,
            container_name=container_name,
            num_instances=num_instances,
            start_instance=should_start,
            instance_mode=instance_mode,
            verbose=verbose,
            **kwargs,
        )

    invalid = [name for name, value in managed_options.items() if value is not None]
    if invalid:
        raise ValueError(
            f"External runtime does not accept selector option(s): {', '.join(invalid)}"
        )
    from dsle.remote import (
        RemoteDarkSoulsEnv,
        RPCClient,
        default_socket_path,
        default_token_path,
        read_auth_token,
    )

    selected_socket = default_socket_path() if socket_path is None else socket_path
    if num_instances == 1:
        return RemoteDarkSoulsEnv(
            selected_socket,
            boss,
            token_file=token_file,
            start_instance=should_start,
            instance_mode=instance_mode,
            verbose=verbose,
            **kwargs,
        )

    requested_instance = kwargs.pop("instance", "dsr-1")
    if requested_instance != "dsr-1":
        raise ValueError("num_instances > 1 uses dsr-1 through dsr-N; remove the instance option")
    if should_start:
        selected_token_file = default_token_path() if token_file is None else token_file
        token = read_auth_token(selected_token_file)
        client = RPCClient(
            selected_socket,
            token,
            connect_timeout=kwargs.get("connect_timeout", 10.0),
            request_timeout=kwargs.get("request_timeout", 180.0),
        )
        try:
            result, array = client.request(
                "start_instances",
                {"count": num_instances, "mode": instance_mode},
            )
            statuses = result.get("statuses")
            expected = {f"dsr-{index}" for index in range(1, num_instances + 1)}
            if (
                array is not None
                or not isinstance(statuses, dict)
                or set(statuses) != expected
                or any(
                    not isinstance(status, dict)
                    or status.get("status") != "ready"
                    or status.get("running") is not True
                    for status in statuses.values()
                )
            ):
                raise RuntimeUnavailableError("External server returned a malformed instance pool")
        finally:
            client.close()

    def factory(instance_name: str):
        return lambda: RemoteDarkSoulsEnv(
            selected_socket,
            boss,
            token_file=token_file,
            start_instance=False,
            instance_mode=instance_mode,
            instance=instance_name,
            verbose=verbose,
            **kwargs,
        )

    return gym.vector.SyncVectorEnv(
        [factory(f"dsr-{index}") for index in range(1, num_instances + 1)]
    )
