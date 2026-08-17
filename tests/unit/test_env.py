from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import dsle
import dsle.env as env_module
from dsle.actions import LOCK_ON, Action
from dsle.exceptions import ResetError, RuntimeCommunicationError
from tests.fakes import ScriptedBackend, state


def make_env(backend: ScriptedBackend, **kwargs):
    kwargs.setdefault("action_ms", 0)
    return dsle.make("asylum_demon", backend=backend, **kwargs)


def test_verbose_lifecycle_is_default_and_can_be_disabled(capsys) -> None:
    visible = make_env(ScriptedBackend())
    visible.reset()
    visible.close()
    output = capsys.readouterr().err
    assert "[CHECK]" in output
    assert "Fight running: asylum_demon" in output
    assert "[COMBAT] dsr-1 step=0" in output
    assert "player_hp=500/500" in output
    assert "player_location={'x': 1.25, 'y': 2.5, 'z': -3.75}" in output
    assert "boss1_hp=1000/1000" in output
    assert "boss_defeated=no" in output
    assert "Environment closed: dsr-1" in output

    quiet = make_env(ScriptedBackend(), verbose=False)
    quiet.reset()
    quiet.close()
    assert capsys.readouterr().err == ""


def test_combat_telemetry_and_info_report_every_configured_boss_health(capsys) -> None:
    initial = state(
        boss_hp=1500,
        boss_hp_max=1500,
        boss_hps=(1500, 2200),
        boss_hp_maxes=(1500, 2200),
    )
    environment = make_env(ScriptedBackend(initial=initial))

    _observation, info = environment.reset()
    environment.close()

    output = capsys.readouterr().err
    assert "boss1_hp=1500/1500" in output
    assert "boss2_hp=2200/2200" in output
    assert info["boss_hps"] == [1500, 2200]
    assert info["boss_hp_maxes"] == [1500, 2200]
    assert info["player_location"] == {"x": 1.25, "y": 2.5, "z": -3.75}


def test_health_writers_refresh_state_and_support_numbered_bosses(capsys) -> None:
    backend = ScriptedBackend(
        initial=state(
            boss_hp=1642,
            boss_hp_max=1642,
            boss_hps=(1642, 2981),
            boss_hp_maxes=(1642, 2981),
        )
    )
    environment = dsle.make(
        "ornstein_and_smough",
        backend=backend,
        action_ms=0,
    )
    environment.reset()

    player_info = environment.set_player_hp(250)
    boss_info = environment.set_boss_hp(2, 1200)
    all_zero_info = environment.set_boss_hp(-1, 0)
    environment.close()

    assert player_info["player_hp"] == 250
    assert boss_info["player_hp"] == 250
    assert boss_info["boss_hps"] == [1642, 1200]
    assert all_zero_info["boss_hps"] == [0, 0]
    assert backend.health_writes == [
        ("player", 250),
        ("boss", 2, 1200),
        ("boss", -1, 0),
    ]
    output = capsys.readouterr().err
    assert "[WRITE] dsr-1 setting player_hp=250" in output
    assert "[WRITE] dsr-1 setting boss2_hp=1200" in output
    assert "[WRITE] dsr-1 setting all_boss_hp=0" in output
    assert "boss2_hp=1200/2981" in output


def test_health_writers_require_active_readable_bounded_targets() -> None:
    backend = ScriptedBackend(
        initial=state(
            boss_hp=1642,
            boss_hp_max=1642,
            boss_hps=(1642, 2981),
            boss_hp_maxes=(1642, 2981),
        )
    )
    environment = dsle.make(
        "ornstein_and_smough",
        backend=backend,
        action_ms=0,
        verbose=False,
    )
    with pytest.raises(RuntimeError, match="reset"):
        environment.set_player_hp(100)

    environment.reset()
    with pytest.raises(ValueError, match=r"\[0, 500\]"):
        environment.set_player_hp(501)
    with pytest.raises(ValueError, match="integer"):
        environment.set_player_hp(True)
    with pytest.raises(ValueError, match=r"\[1, 2\].*-1"):
        environment.set_boss_hp(3, 100)
    with pytest.raises(ValueError, match="only valid with hp=0"):
        environment.set_boss_hp(-1, 1)
    with pytest.raises(ValueError, match=r"\[0, 2981\]"):
        environment.set_boss_hp(2, 2982)
    environment.close()


def test_flag_backed_boss_health_is_read_only() -> None:
    backend = ScriptedBackend(initial=state(boss_hp=1, boss_hp_max=1))
    environment = dsle.make(
        "bed_of_chaos",
        backend=backend,
        action_ms=0,
        verbose=False,
    )
    environment.reset()

    with pytest.raises(RuntimeCommunicationError, match=r"defeated flag.*read-only"):
        environment.set_boss_hp(1, 0)
    with pytest.raises(RuntimeCommunicationError, match=r"defeated flag.*read-only"):
        environment.set_boss_hp(-1, 0)
    assert backend.health_writes == []
    environment.close()


@pytest.mark.parametrize(
    ("mode", "shape", "dtype"),
    [
        ("grayscale", (1, 600, 800), np.uint8),
        ("rgb", (3, 600, 800), np.uint8),
        ("state_only", (2,), np.float32),
    ],
)
def test_observation_contract(mode, shape, dtype) -> None:
    env = make_env(ScriptedBackend(), obs_mode=mode)
    observation, info = env.reset(seed=7)
    assert observation.shape == shape
    assert observation.dtype == dtype
    assert env.observation_space.contains(observation)
    assert info["boss"] == "asylum_demon"
    env.close()


def test_passive_observe_does_not_inject_actions_and_menu_return_ends_manual_control() -> None:
    backend = ScriptedBackend([state(player_hp=321, boss_hp=765)])
    environment = make_env(backend, obs_mode="state_only", verbose=False)
    environment.reset()

    observation, info = environment.observe()

    assert environment.observation_space.contains(observation)
    assert info["player_hp"] == 321
    assert info["boss_hp"] == 765
    assert backend.actions == []
    assert backend.outcomes == []
    environment.return_to_menu(timeout_s=12.5)
    assert backend.menu_timeouts == [12.5]
    with pytest.raises(RuntimeError, match="reset"):
        environment.observe()
    environment.close()


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
def test_return_to_menu_rejects_invalid_timeouts(timeout: object) -> None:
    environment = make_env(ScriptedBackend(), verbose=False)
    with pytest.raises(ValueError, match="finite positive"):
        environment.return_to_menu(timeout_s=timeout)  # type: ignore[arg-type]
    environment.close()


def test_declarative_monitor_index_reaches_default_live_backend(monkeypatch) -> None:
    calls = []

    def factory(instance, assets, config, *, monitor_index, verbose, lifecycle):
        calls.append((instance, assets, config, monitor_index, verbose, lifecycle.verbose))
        return ScriptedBackend()

    monkeypatch.setattr(env_module, "_default_backend_factory", factory)
    environment = env_module.DarkSoulsEnv("asylum_demon", action_ms=0)
    assert calls == [("dsr-1", None, None, 0, True, True)]
    environment.close()


def test_rgb_render_is_hwc_rgb_copy() -> None:
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    frame[..., 0] = 255
    env = make_env(ScriptedBackend(frame=frame), render_mode="rgb_array")
    env.reset()
    rendered = env.render()
    assert rendered is not None and rendered.shape == (600, 800, 3)
    assert tuple(rendered[0, 0]) == (255, 0, 0)
    rendered[:] = 0
    assert tuple(env.render()[0, 0]) == (255, 0, 0)
    env.close()


def test_rgb_render_normalizes_dtype_and_validates_state_only_frames() -> None:
    floating = np.full((600, 800, 3), 300.0, dtype=np.float32)
    floating_backend = ScriptedBackend()
    floating_backend.frame = floating
    env = make_env(floating_backend, render_mode="rgb_array")
    env.reset()
    rendered = env.render()
    assert rendered is not None
    assert rendered.dtype == np.uint8
    assert rendered.max() == 255
    env.close()

    malformed = make_env(
        ScriptedBackend(frame=np.zeros((2, 3), dtype=np.uint8)),
        obs_mode="state_only",
        render_mode="rgb_array",
    )
    malformed.reset()
    with pytest.raises(Exception, match="render frame"):
        malformed.render()
    malformed.close()


def test_player_death_terminates_immediately() -> None:
    backend = ScriptedBackend([state(player_hp=0, player_dead=True)])
    env = make_env(backend)
    env.reset()
    _, reward, terminated, truncated, info = env.step(Action.LIGHT_ATTACK)
    assert terminated and not truncated
    assert info["terminated_reason"] == "player_dead"
    assert reward < -10
    assert backend.outcomes[0].reason == "player_dead"


def test_invalid_victory_read_never_wins() -> None:
    invalid = state(
        boss_hp=None,
        boss_hp_max=None,
        boss_defeated=True,
        boss_state_valid=False,
    )
    backend = ScriptedBackend([invalid], initial=state())
    env = make_env(backend)
    env.reset()
    _, _, terminated, truncated, info = env.step(Action.LIGHT_ATTACK)
    assert not terminated and not truncated
    assert not info["win"]


def test_valid_first_transition_boss_kill_wins_immediately() -> None:
    defeated = state(boss_hp=0, boss_defeated=True)
    backend = ScriptedBackend([defeated])
    env = make_env(backend)
    env.reset()
    _, reward, terminated, truncated, info = env.step(Action.LIGHT_ATTACK)
    assert terminated and not truncated and info["win"]
    assert reward > 99


@pytest.mark.parametrize(
    "initial",
    [
        state(boss_state_valid=False),
        state(boss_hp=None),
        state(boss_hp_max=None),
        state(boss_hp=0),
        state(boss_hp_max=0),
        state(boss_defeated=True),
    ],
)
def test_reset_rejects_unusable_initial_boss_state(initial) -> None:
    backend = ScriptedBackend(initial=initial)
    env = make_env(backend)
    with pytest.raises(ResetError, match=r"Initial boss state|Initial boss HP"):
        env.reset()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


@pytest.mark.parametrize(
    "initial",
    [
        state(player_state_valid=False),
        state(player_hp=None),
        state(player_hp_max=None),
        state(player_hp=0),
        state(player_hp_max=0),
        state(player_dead=True),
    ],
)
def test_reset_rejects_unusable_initial_player_state(initial) -> None:
    env = make_env(ScriptedBackend(initial=initial))
    with pytest.raises(ResetError, match=r"Initial player"):
        env.reset()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)


def test_missing_boss_pointer_never_looks_like_a_win() -> None:
    missing = state(
        boss_hp=None,
        boss_hp_max=None,
        boss_defeated=False,
        boss_state_valid=False,
    )
    backend = ScriptedBackend([missing] * 3)
    env = make_env(backend, max_steps=3)
    env.reset()
    for _index in range(3):
        _, _, terminated, truncated, info = env.step(0)
    assert not terminated and truncated
    assert info["truncated_reason"] == "max_steps"
    assert not info["win"]


def test_action_repeat_stops_at_episode_limit() -> None:
    backend = ScriptedBackend()
    env = make_env(backend, action_repeat=4, max_steps=3)
    env.reset()
    _, reward, terminated, truncated, info = env.step(0)
    assert not terminated and truncated and info["step"] == 3
    assert len(backend.actions) == 3
    assert reward == pytest.approx(-0.003)


def test_runtime_fault_becomes_a_diagnostic_truncation() -> None:
    backend = ScriptedBackend([OSError("process vanished")])
    env = make_env(backend)
    initial, _ = env.reset()
    observation, reward, terminated, truncated, info = env.step(0)
    assert not terminated and truncated
    assert info["truncated_reason"] == "runtime_error"
    assert "process vanished" in info["runtime_error"]
    assert np.array_equal(observation, initial)
    assert reward < -10


def test_malformed_step_frame_truncates_and_finishes_once() -> None:
    backend = ScriptedBackend()
    env = make_env(backend)
    initial, _ = env.reset()
    backend.frame = np.zeros((10, 10, 3), dtype=np.uint8)

    observation, reward, terminated, truncated, info = env.step(0)

    assert not terminated and truncated
    assert info["truncated_reason"] == "runtime_error"
    assert "expected RGB" in info["runtime_error"]
    assert np.array_equal(observation, initial)
    assert reward < -10
    assert len(backend.actions) == 1
    assert len(backend.outcomes) == 1
    assert backend.outcomes[0].reason == "runtime_error"
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    assert len(backend.actions) == 1
    assert len(backend.outcomes) == 1


def test_cleanup_error_is_reported_without_losing_transition() -> None:
    backend = ScriptedBackend(
        [state(player_hp=0, player_dead=True)], finish_error=TimeoutError("menu stuck")
    )
    env = make_env(backend)
    env.reset()
    _, _, terminated, _, info = env.step(0)
    assert terminated
    assert info["cleanup_error"] == "TimeoutError: menu stuck"


def test_automatic_lock_on_is_not_a_policy_action() -> None:
    unlocked = state(locked_on=False)
    backend = ScriptedBackend([unlocked] * 3, initial=unlocked)
    env = make_env(backend, lock_on_interval=1)
    env.reset()
    env.step(Action.MOVE_FORWARD)
    env.step(Action.MOVE_FORWARD)
    actions = [entry[0] for entry in backend.actions]
    assert actions.count(LOCK_ON) == 2  # reset acquisition plus one lost-lock repair
    assert backend.actions[0] == (LOCK_ON, 0.1)


def test_invalid_lock_on_sensor_never_causes_speculative_input() -> None:
    unknown_lock = state(locked_on=None)
    backend = ScriptedBackend([unknown_lock] * 3, initial=unknown_lock)
    env = make_env(backend, lock_on_interval=1)
    env.reset()
    env.step(Action.MOVE_FORWARD)
    env.step(Action.MOVE_FORWARD)
    assert LOCK_ON not in [entry[0] for entry in backend.actions]


def test_invalid_actions_and_step_before_reset_are_rejected() -> None:
    backend = ScriptedBackend()
    env = make_env(backend)
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset()
    for value in (-1, 14, True, False, 1.0, 1.9, "1"):
        with pytest.raises(ValueError):
            env.step(value)
    assert backend.actions == []
    assert env.step(np.int64(1))[3] is False


def test_gymnasium_checker_accepts_public_state_machine() -> None:
    backend = ScriptedBackend()
    env = make_env(backend, max_steps=3)
    # check_env warns that an external game cannot be deterministically seeded;
    # skip render checks because render_mode is fixed at construction.
    check_env(env, skip_render_check=True, skip_close_check=True)


def test_can_enable_stricter_victory_guard_for_targeted_experiment() -> None:
    defeated = state(boss_hp=0, boss_defeated=True)
    backend = ScriptedBackend([defeated, defeated])
    env = make_env(backend)
    env.boss = replace(
        env.boss,
        limits=replace(env.boss.limits, minimum_victory_step=2, victory_confirmations=2),
    )
    env.reset()
    assert env.step(0)[2] is False
    assert env.step(0)[2] is True


def test_verified_boss_death_wins_a_simultaneous_trade() -> None:
    traded = state(player_hp=0, boss_hp=0, player_dead=True, boss_defeated=True)
    backend = ScriptedBackend([traded])
    env = make_env(backend)
    env.reset()
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert info["win"] and info["terminated_reason"] == "boss_defeated"
    assert reward > 99


@pytest.mark.parametrize(
    "kwargs",
    [
        {"obs_mode": "depth"},
        {"render_mode": "human"},
        {"action_ms": -1},
        {"action_repeat": 0},
        {"action_repeat": True},
        {"action_repeat": 1.5},
        {"max_steps": 0},
        {"max_steps": True},
        {"lock_on_interval": 0},
        {"lock_on_interval": 1.5},
        {"action_ms": False},
        {"auto_lock_on": "false"},
        {"instance": ""},
        {"difficulty": "missing"},
        {"difficulty": None},
    ],
)
def test_constructor_rejects_invalid_public_options(kwargs) -> None:
    with pytest.raises(ValueError):
        make_env(ScriptedBackend(), **kwargs)


def test_constructor_rejects_non_reward_config() -> None:
    with pytest.raises(TypeError, match="RewardConfig"):
        make_env(ScriptedBackend(), reward={"win_bonus": 1.0})


def test_constructor_rejects_two_backend_sources() -> None:
    with pytest.raises(ValueError, match="backend or backend_factory"):
        dsle.make(
            "asylum_demon",
            backend=ScriptedBackend(),
            backend_factory=lambda *_args: ScriptedBackend(),
        )


def test_reset_supports_difficulty_option_and_rejects_after_close() -> None:
    backend = ScriptedBackend()
    env = make_env(backend)
    env.reset(options={"difficulty": "boosted"})
    assert backend.resets[-1] == ("asylum_demon", "asylum_demon_boosted.sl2")
    assert env.difficulty == "boosted"
    env.close()
    env.close()
    assert backend.closed
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


@pytest.mark.parametrize(
    "options",
    [
        {"difficulty": 1},
        {"difficulty": ""},
        {"unknown": True},
        [("difficulty", "standard")],
    ],
)
def test_reset_rejects_malformed_options_without_touching_backend(options) -> None:
    backend = ScriptedBackend()
    env = make_env(backend)
    with pytest.raises(ValueError):
        env.reset(options=options)
    assert backend.resets == []


def test_backend_reset_failure_is_wrapped_with_boss_and_instance() -> None:
    class FailingBackend(ScriptedBackend):
        def reset(self, boss, save_state):
            raise OSError("save unavailable")

    env = make_env(FailingBackend())
    with pytest.raises(Exception, match=r"asylum_demon.*dsr-1.*save unavailable"):
        env.reset()


def test_bad_runtime_frame_is_rejected() -> None:
    env = make_env(ScriptedBackend(frame=np.zeros((10, 10, 3), dtype=np.uint8)))
    with pytest.raises(Exception, match="expected RGB"):
        env.reset()


def test_render_without_rgb_mode_or_reset_returns_none() -> None:
    env = make_env(ScriptedBackend())
    assert env.render() is None
    env.reset()
    assert env.render() is None
    env.close()


def test_step_after_terminal_requires_an_explicit_reset() -> None:
    env = make_env(ScriptedBackend([state(player_hp=0, player_dead=True)]))
    env.reset()
    assert env.step(0)[2]
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
