from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="deep-RL example tests require the examples extra")

from examples import dqn, ppo  # noqa: E402


@pytest.mark.parametrize(
    "field",
    [
        "gamma",
        "epsilon_start",
        "epsilon_end",
        "exploration_fraction",
        "learning_rate",
        "max_grad_norm",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_dqn_rejects_every_nonfinite_hyperparameter(field, value, tmp_path) -> None:
    args = dqn.build_parser().parse_args(
        ["train", "--boss", "asylum_demon", "--output-dir", str(tmp_path)]
    )
    setattr(args, field, value)

    with pytest.raises(ValueError, match="finite"):
        dqn._validate_train_args(args)


@pytest.mark.parametrize(
    "field",
    [
        "gamma",
        "gae_lambda",
        "learning_rate",
        "max_grad_norm",
        "clip_coefficient",
        "entropy_coefficient",
        "value_coefficient",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_ppo_rejects_every_nonfinite_hyperparameter(field, value, tmp_path) -> None:
    args = ppo.build_parser().parse_args(
        ["train", "--boss", "asylum_demon", "--output-dir", str(tmp_path)]
    )
    setattr(args, field, value)

    with pytest.raises(ValueError, match="finite"):
        ppo._validate_train_args(args)


@pytest.mark.parametrize("module", [dqn, ppo])
def test_deep_rl_defaults_use_dsle_output_dir(module, monkeypatch, tmp_path) -> None:
    output_dir = tmp_path / module.__name__.split(".")[-1]
    monkeypatch.setenv("DSLE_OUTPUT_DIR", str(output_dir))

    args = module.build_parser().parse_args(["train", "--boss", "asylum_demon"])

    assert args.output_dir == output_dir.resolve()
    assert module._checkpoint_path(args).parent == output_dir.resolve()
    assert module._metrics_path(args, "train").parent == output_dir.resolve()


@pytest.mark.parametrize("module", [dqn, ppo])
def test_eval_rejects_missing_checkpoint_before_env_allocation(
    module, monkeypatch, tmp_path
) -> None:
    args = module.build_parser().parse_args(
        [
            "eval",
            "--boss",
            "asylum_demon",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--output-dir",
            str(tmp_path),
        ]
    )
    allocated = False

    def make(_args):
        nonlocal allocated
        allocated = True

    monkeypatch.setattr(module, "_make_env", make)
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        module.evaluate(args)
    assert not allocated


@pytest.mark.parametrize("module", [dqn, ppo])
def test_train_rejects_invalid_values_before_env_allocation(module, monkeypatch, tmp_path) -> None:
    args = module.build_parser().parse_args(
        ["train", "--boss", "asylum_demon", "--output-dir", str(tmp_path)]
    )
    args.learning_rate = float("nan")
    allocated = False

    def make(_args):
        nonlocal allocated
        allocated = True

    monkeypatch.setattr(module, "_make_env", make)
    with pytest.raises(ValueError, match="finite"):
        module.train(args)
    assert not allocated


@pytest.mark.parametrize("module", [dqn, ppo])
def test_train_rejects_checkpoint_metrics_collision_before_env_allocation(
    module, monkeypatch, tmp_path
) -> None:
    collision = tmp_path / "same-file"
    args = module.build_parser().parse_args(
        [
            "train",
            "--boss",
            "asylum_demon",
            "--checkpoint",
            str(collision),
            "--metrics-output",
            str(collision),
        ]
    )
    allocated = False

    def make(_args):
        nonlocal allocated
        allocated = True

    monkeypatch.setattr(module, "_make_env", make)
    with pytest.raises(ValueError, match="different files"):
        module.train(args)
    assert not allocated


def test_dqn_bootstrap_semantics_and_network_shape() -> None:
    assert dqn.terminal_for_bootstrap(True, False, {})
    assert not dqn.terminal_for_bootstrap(False, True, {"truncated_reason": "max_steps"})
    assert dqn.terminal_for_bootstrap(False, True, {"truncated_reason": "runtime_error"})

    network = dqn.QNetwork(14)
    values = network(torch.zeros((2, 1, 84, 84), dtype=torch.uint8))
    assert values.shape == (2, 14)


def test_ppo_time_limit_semantics_and_network_shape() -> None:
    assert ppo.time_limit_bootstrap_reward(
        2.0,
        terminated=False,
        truncated=True,
        info={"truncated_reason": "max_steps"},
        final_value=3.0,
        gamma=0.5,
    ) == pytest.approx(3.5)
    assert ppo.time_limit_bootstrap_reward(
        2.0,
        terminated=False,
        truncated=True,
        info={"truncated_reason": "runtime_error"},
        final_value=3.0,
        gamma=0.5,
    ) == pytest.approx(2.0)

    agent = ppo.PPOAgent(14)
    action, log_probability, entropy, value = agent.action_and_value(
        torch.zeros((2, 1, 84, 84), dtype=torch.uint8), deterministic=True
    )
    assert action.shape == (2,)
    assert log_probability.shape == (2,)
    assert entropy.shape == (2,)
    assert value.shape == (2,)


@pytest.mark.parametrize("module", [dqn, ppo])
def test_eval_closes_env_when_checkpoint_loading_fails(module, monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "invalid.pt"
    checkpoint.write_text("not a torch checkpoint", encoding="utf-8")
    args = module.build_parser().parse_args(
        [
            "eval",
            "--boss",
            "asylum_demon",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "metrics"),
        ]
    )

    class ActionSpace:
        n = 14

    class Env:
        action_space = ActionSpace()

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    env = Env()
    monkeypatch.setattr(module, "_make_env", lambda _args: env)
    loader_name = "_load_network" if module is dqn else "_load_agent"

    def fail_to_load(*_args, **_kwargs):
        raise RuntimeError("invalid checkpoint")

    monkeypatch.setattr(module, loader_name, fail_to_load)

    with pytest.raises(RuntimeError, match="invalid checkpoint"):
        module.evaluate(args)
    assert env.closed


@pytest.mark.parametrize("module", [dqn, ppo])
def test_metrics_path_can_be_overridden(module, tmp_path) -> None:
    explicit = tmp_path / "custom.jsonl"
    args = module.build_parser().parse_args(
        [
            "train",
            "--boss",
            "asylum_demon",
            "--metrics-output",
            str(explicit),
        ]
    )
    assert module._metrics_path(args, "train") == Path(explicit)


class _TerminalTrainingEnv:
    class ActionSpace:
        n = 14

    action_space = ActionSpace()

    def __init__(self) -> None:
        self.reset_calls = 0
        self.step_calls = 0
        self.closed = False

    def reset(self):
        self.reset_calls += 1
        return np.zeros((1, 84, 84), dtype=np.uint8), {}

    def step(self, _action):
        self.step_calls += 1
        return (
            np.zeros((1, 84, 84), dtype=np.uint8),
            1.0,
            True,
            False,
            {"win": True, "terminated_reason": "boss_defeated"},
        )

    def close(self):
        self.closed = True


def test_dqn_does_not_reset_after_its_final_training_transition(monkeypatch, tmp_path) -> None:
    args = dqn.build_parser().parse_args(
        [
            "train",
            "--boss",
            "asylum_demon",
            "--total-timesteps",
            "1",
            "--replay-size",
            "1",
            "--batch-size",
            "1",
            "--learning-starts",
            "2",
            "--checkpoint-interval",
            "2",
            "--log-interval",
            "1",
            "--output-dir",
            str(tmp_path),
            "--device",
            "cpu",
        ]
    )
    environment = _TerminalTrainingEnv()
    monkeypatch.setattr(dqn, "_make_env", lambda _args: environment)

    checkpoint = dqn.train(args)

    assert checkpoint.is_file()
    assert environment.reset_calls == 1
    assert environment.step_calls == 1
    assert environment.closed


def test_ppo_does_not_reset_after_its_final_training_transition(monkeypatch, tmp_path) -> None:
    args = ppo.build_parser().parse_args(
        [
            "train",
            "--boss",
            "asylum_demon",
            "--total-timesteps",
            "1",
            "--rollout-steps",
            "1",
            "--epochs",
            "1",
            "--minibatch-size",
            "1",
            "--checkpoint-interval",
            "2",
            "--output-dir",
            str(tmp_path),
            "--device",
            "cpu",
        ]
    )
    environment = _TerminalTrainingEnv()
    monkeypatch.setattr(ppo, "_make_env", lambda _args: environment)

    checkpoint = ppo.train(args)

    assert checkpoint.is_file()
    assert environment.reset_calls == 1
    assert environment.step_calls == 1
    assert environment.closed
