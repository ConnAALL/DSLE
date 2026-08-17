from __future__ import annotations

import json
from argparse import Namespace

import numpy as np
import pytest

from dsle import Action
from examples import expert_agent, random_agent, scope
from examples.common import default_output_dir, run_episodes
from examples.expert_agent import ExpertPolicy


class _Policy:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def act(self, _observation, _info) -> int:
        return 0


class _ExplodingPolicy(_Policy):
    def act(self, _observation, _info) -> int:
        raise RuntimeError("policy failed")


class _OneStepEnv:
    def __init__(self) -> None:
        self.close_calls = 0

    def reset(self):
        return np.zeros((1, 2, 2), dtype=np.uint8), {}

    def step(self, _action):
        return (
            np.zeros((1, 2, 2), dtype=np.uint8),
            2.5,
            True,
            False,
            {"win": True, "terminated_reason": "boss_defeated"},
        )

    def close(self) -> None:
        self.close_calls += 1


class _ActionSpace:
    def __init__(self, *, seed_error: Exception | None = None) -> None:
        self.seed_error = seed_error
        self.seed_value: int | None = None

    def seed(self, value: int) -> None:
        if self.seed_error is not None:
            raise self.seed_error
        self.seed_value = value

    def sample(self) -> int:
        return 11


class _AllocatedEnv(_OneStepEnv):
    def __init__(self, action_space: _ActionSpace | None = None) -> None:
        super().__init__()
        self.action_space = action_space or _ActionSpace()


def _random_args(tmp_path, **overrides) -> Namespace:
    values = {
        "boss": "asylum_demon",
        "instance": "dsr-1",
        "difficulty": "standard",
        "episodes": 1,
        "max_steps": 10,
        "seed": 42,
        "output": tmp_path / "random.jsonl",
    }
    values.update(overrides)
    return Namespace(**values)


def _expert_args(tmp_path, **overrides) -> Namespace:
    values = {
        "boss": "asylum_demon",
        "instance": "dsr-1",
        "difficulty": "standard",
        "episodes": 1,
        "max_steps": 10,
        "heal_threshold": 0.4,
        "heal_cooldown_seconds": 0.0,
        "max_heals": 5,
        "output": tmp_path / "expert.jsonl",
    }
    values.update(overrides)
    return Namespace(**values)


def test_default_output_dir_uses_environment_or_local_runs(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DSLE_OUTPUT_DIR", raising=False)
    assert default_output_dir() == (tmp_path / "runs").resolve()

    configured = tmp_path / "artifacts"
    monkeypatch.setenv("DSLE_OUTPUT_DIR", str(configured))
    assert default_output_dir() == configured.resolve()


def test_run_episodes_writes_standard_jsonl_and_closes(tmp_path) -> None:
    env = _OneStepEnv()
    policy = _Policy()
    output = tmp_path / "nested" / "episodes.jsonl"

    results = run_episodes(
        env,
        policy,
        1,
        output,
        metadata={"algorithm": "test", "phase": "evaluate"},
    )

    assert env.close_calls == 1
    assert policy.resets == 1
    assert results[0].reward == 2.5
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row == {
        "algorithm": "test",
        "phase": "evaluate",
        "episode": 1,
        "return": 2.5,
        "length": 1,
        "win": True,
        "terminated_reason": "boss_defeated",
        "truncated_reason": None,
        "elapsed_seconds": pytest.approx(row["elapsed_seconds"]),
    }


@pytest.mark.parametrize("episodes", [0, -1])
def test_run_episodes_closes_when_validation_fails(episodes) -> None:
    env = _OneStepEnv()
    with pytest.raises(ValueError, match="episodes"):
        run_episodes(env, _Policy(), episodes)
    assert env.close_calls == 1


def test_run_episodes_closes_when_output_preflight_fails(tmp_path) -> None:
    env = _OneStepEnv()
    output_directory = tmp_path / "not-a-file"
    output_directory.mkdir()

    with pytest.raises(ValueError, match="regular file"):
        run_episodes(env, _Policy(), 1, output_directory)
    assert env.close_calls == 1


def test_run_episodes_closes_when_policy_fails() -> None:
    env = _OneStepEnv()
    with pytest.raises(RuntimeError, match="policy failed"):
        run_episodes(env, _ExplodingPolicy(), 1)
    assert env.close_calls == 1


@pytest.mark.parametrize(
    ("module", "filename"),
    [(random_agent, "random.jsonl"), (expert_agent, "expert.jsonl")],
)
def test_simple_baseline_defaults_use_dsle_output_dir(
    module, filename, monkeypatch, tmp_path
) -> None:
    output_dir = tmp_path / "artifacts"
    monkeypatch.setenv("DSLE_OUTPUT_DIR", str(output_dir))
    args = module.build_parser().parse_args([])
    assert args.output == output_dir.resolve() / filename


def test_random_policy_uses_the_environment_action_space() -> None:
    action_space = _ActionSpace()
    policy = random_agent.RandomPolicy(action_space)
    assert policy.act(None, {}) == 11


@pytest.mark.parametrize(
    "overrides",
    [{"episodes": 0}, {"max_steps": 0}],
)
def test_random_rejects_bad_arguments_before_env_allocation(
    monkeypatch, tmp_path, overrides
) -> None:
    allocated = False

    def make(*_args, **_kwargs):
        nonlocal allocated
        allocated = True
        return _AllocatedEnv()

    monkeypatch.setattr(random_agent.dsle, "make", make)
    with pytest.raises(ValueError):
        random_agent.run(_random_args(tmp_path, **overrides))
    assert not allocated


def test_random_preflights_output_before_env_allocation(monkeypatch, tmp_path) -> None:
    allocated = False
    output_directory = tmp_path / "directory"
    output_directory.mkdir()

    def make(*_args, **_kwargs):
        nonlocal allocated
        allocated = True
        return _AllocatedEnv()

    monkeypatch.setattr(random_agent.dsle, "make", make)
    with pytest.raises(ValueError, match="regular file"):
        random_agent.run(_random_args(tmp_path, output=output_directory))
    assert not allocated


def test_random_closes_env_when_action_space_seeding_fails(monkeypatch, tmp_path) -> None:
    env = _AllocatedEnv(_ActionSpace(seed_error=RuntimeError("seed failed")))
    monkeypatch.setattr(random_agent.dsle, "make", lambda *_args, **_kwargs: env)

    with pytest.raises(RuntimeError, match="seed failed"):
        random_agent.run(_random_args(tmp_path))
    assert env.close_calls == 1


def test_expert_confirms_immediate_heal_before_cooldown() -> None:
    sleeps: list[float] = []
    policy = ExpertPolicy(sleeper=sleeps.append)

    first = policy.act(None, {"player_hp": 300, "player_hp_max": 1000})
    second = policy.act(None, {"player_hp": 650, "player_hp_max": 1000})

    assert first == int(Action.HEAL)
    assert second == int(Action.LIGHT_ATTACK)
    assert policy.heals_attempted == 1
    assert policy.heals_committed == 1
    assert sleeps == []


def test_expert_waits_once_then_confirms_delayed_heal() -> None:
    sleeps: list[float] = []
    policy = ExpertPolicy(heal_cooldown_s=1.25, sleeper=sleeps.append)

    assert policy.act(None, {"player_hp": 300, "player_hp_max": 1000}) == int(Action.HEAL)
    assert policy.act(None, {"player_hp": 300, "player_hp_max": 1000}) != int(Action.HEAL)
    assert policy.act(None, {"player_hp": 600, "player_hp_max": 1000}) != int(Action.HEAL)

    assert sleeps == [1.25]
    assert policy.heals_attempted == 1
    assert policy.heals_committed == 1


def test_expert_retries_an_unconfirmed_heal() -> None:
    policy = ExpertPolicy(heal_cooldown_s=0.0, sleeper=lambda _seconds: None)
    low_hp = {"player_hp": 300, "player_hp_max": 1000}

    assert policy.act(None, low_hp) == int(Action.HEAL)
    assert policy.act(None, low_hp) != int(Action.HEAL)
    assert policy.act(None, low_hp) == int(Action.HEAL)
    assert policy.heals_attempted == 2
    assert policy.heals_committed == 0


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("heal_threshold", float("nan")),
        ("heal_threshold", float("inf")),
        ("heal_cooldown_s", float("nan")),
        ("heal_cooldown_s", float("inf")),
    ],
)
def test_expert_rejects_nonfinite_configuration(keyword, value) -> None:
    with pytest.raises(ValueError, match="finite"):
        ExpertPolicy(**{keyword: value})


def test_expert_validates_policy_before_env_allocation(monkeypatch, tmp_path) -> None:
    allocated = False

    def make(*_args, **_kwargs):
        nonlocal allocated
        allocated = True
        return _AllocatedEnv()

    monkeypatch.setattr(expert_agent.dsle, "make", make)
    with pytest.raises(ValueError, match="heal_threshold"):
        expert_agent.run(_expert_args(tmp_path, heal_threshold=float("nan")))
    assert not allocated


def test_scope_evaluation_appends_the_standard_metrics_schema(monkeypatch, tmp_path) -> None:
    weights = tmp_path / "weights.npy"
    np.save(weights, np.zeros(scope.chromosome_size(1)))
    args = Namespace(
        boss="asylum_demon",
        instances="dsr-1",
        difficulty="standard",
        max_steps=10,
        k=1,
        percentile=90.0,
        episodes=1,
        weights=weights,
        output=tmp_path / "scope",
    )
    monkeypatch.setattr(
        scope,
        "evaluate_once",
        lambda *_args: {
            "fitness": 125.0,
            "return": 3.5,
            "win": True,
            "length": 12,
            "terminated_reason": "boss_defeated",
            "truncated_reason": None,
            "instance": "dsr-1",
        },
    )

    scope.evaluate(args)

    row = json.loads((args.output / "metrics.jsonl").read_text(encoding="utf-8"))
    assert row["algorithm"] == "scope"
    assert row["phase"] == "evaluate"
    assert row["return"] == 3.5
    assert row["length"] == 12
