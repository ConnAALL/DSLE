from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO

import pytest
from rich.console import Console

from dsle.config import load_boss
from dsle.exceptions import ConfigurationError
from dsle.models import EpisodeOutcome, RuntimeOperation, immutable_mapping
from dsle.progress import LifecycleLogger
from dsle.runtime.operations import OperationContext, OperationExecutor, matches_condition


@dataclass
class Target:
    calls: list[tuple[str, object]] = field(default_factory=list)

    def ensure_menu(self, **kwargs):
        self.calls.append(("ensure_menu", kwargs))

    def load_save(self, save_state, **kwargs):
        self.calls.append(("load_save", (save_state, kwargs)))

    def wait_until_ready(self, boss, params, **kwargs):
        self.calls.append(("wait_until_ready", (boss.boss_id, dict(params), kwargs)))

    def set_flag(self, name, enabled):
        self.calls.append(("set_flag", (name, enabled)))

    def teleport_player(self, **kwargs):
        self.calls.append(("teleport_player", kwargs))

    def tap_key(self, key, **kwargs):
        self.calls.append(("tap_key", (key, kwargs)))

    def hold_key(self, key, **kwargs):
        self.calls.append(("hold_key", (key, kwargs)))

    def walk_until_template(self, params, **kwargs):
        self.calls.append(("walk_until_template", (dict(params), kwargs)))

    def repeat_actions(self, actions, **kwargs):
        self.calls.append(("repeat_actions", (list(actions), kwargs)))

    def wait_for_victory(self, boss, **kwargs):
        self.calls.append(("wait_for_victory", (boss.boss_id, kwargs)))

    def return_to_title(self):
        self.calls.append(("return_to_title", None))


def operation(op: str, *, when=("always",), **params) -> RuntimeOperation:
    return RuntimeOperation(op=op, params=immutable_mapping(params), when=when)


def test_operation_conditions_are_explicit() -> None:
    win = EpisodeOutcome(True, True, False, "boss_defeated", 100)
    loss = EpisodeOutcome(False, True, False, "player_dead", 50)
    runtime_error = EpisodeOutcome(False, False, True, "runtime_error", 2)
    assert matches_condition(("win",), win)
    assert not matches_condition(("win",), loss)
    assert matches_condition(("loss",), loss)
    assert matches_condition(("terminated",), win)
    assert matches_condition(("truncated",), runtime_error)
    assert matches_condition(("runtime_error",), runtime_error)
    assert not matches_condition(("win",), None)


def test_executor_dispatches_validated_operations_and_skips_wrong_condition() -> None:
    target = Target()
    boss = load_boss("gwyn_lord_of_cinder")
    outcome = EpisodeOutcome(True, True, False, "boss_defeated", 100)
    operations = (
        operation("load_save", interact=False),
        operation("set_flag", flag="no_dead", enable=True),
        operation("teleport_player", x=1.0, y=None, z=-2.0),
        operation(
            "repeat_actions",
            actions=[
                {"name": "move_forward", "simultaneous": True},
                {"name": "light_attack", "simultaneous": True},
            ],
            duration_s=1.0,
        ),
        operation("return_to_title", when=("loss",)),
    )
    OperationExecutor(target, clock=lambda: 0.0, sleeper=lambda _seconds: None).execute(
        operations,
        OperationContext(boss=boss, save_state="gwyn_lord_of_cinder.sl2", outcome=outcome),
        timeout_s=10.0,
    )
    assert [name for name, _ in target.calls] == [
        "load_save",
        "set_flag",
        "teleport_player",
        "repeat_actions",
    ]
    assert target.calls[0][1][0] == "gwyn_lord_of_cinder.sl2"


def test_executor_reports_each_applicable_operation_with_instance_and_phase() -> None:
    stream = StringIO()
    lifecycle = LifecycleLogger(
        console=Console(file=stream, color_system=None, highlight=False, width=160)
    )
    target = Target()
    boss = load_boss("asylum_demon")
    operations = (
        operation("set_flag", flag="no_damage", enable=True),
        operation("return_to_title", when=("win",)),
    )

    OperationExecutor(
        target,
        clock=lambda: 0.0,
        sleeper=lambda _seconds: None,
        lifecycle=lifecycle,
    ).execute(
        operations,
        OperationContext(
            boss=boss,
            save_state="asylum_demon.sl2",
            phase="setup",
            instance="dsr-3",
        ),
        timeout_s=10.0,
    )

    assert stream.getvalue().splitlines() == [
        "[SETUP] dsr-3 1/1 starting set_flag (flag='no_damage', enable=True)",
        "[SETUP] dsr-3 1/1 completed set_flag",
    ]


def test_executor_enforces_one_sequence_deadline() -> None:
    now = iter((0.0, 0.1, 2.0))
    target = Target()
    boss = load_boss("asylum_demon")
    executor = OperationExecutor(target, clock=lambda: next(now), sleeper=lambda _seconds: None)
    with pytest.raises(TimeoutError, match="timed out"):
        executor.execute(
            (operation("return_to_title"), operation("return_to_title")),
            OperationContext(boss=boss, save_state="asylum_demon.sl2"),
            timeout_s=1.0,
        )


def test_executor_rejects_malformed_repeat_actions() -> None:
    target = Target()
    boss = load_boss("asylum_demon")
    with pytest.raises(ConfigurationError, match="non-empty"):
        OperationExecutor(target, clock=lambda: 0.0).execute(
            (operation("repeat_actions", actions=[]),),
            OperationContext(boss=boss, save_state="asylum_demon.sl2"),
            timeout_s=1.0,
        )


@pytest.mark.parametrize(
    ("operation_value", "field"),
    [
        (operation("load_save", interact="false"), "interact"),
        (operation("set_flag", flag="no_dead", enable=1), "enable"),
        (
            operation(
                "repeat_actions",
                actions=["move_forward"],
                duration_s=0.1,
                simultaneous="false",
            ),
            "simultaneous",
        ),
    ],
)
def test_executor_never_coerces_nonboolean_operation_values(
    operation_value: RuntimeOperation, field: str
) -> None:
    with pytest.raises(ConfigurationError, match=field):
        OperationExecutor(Target(), clock=lambda: 0.0).execute(
            (operation_value,),
            OperationContext(boss=load_boss("asylum_demon"), save_state="asylum_demon.sl2"),
            timeout_s=1.0,
        )


@pytest.mark.parametrize("timeout", [True, 0.0, float("nan"), float("inf")])
def test_executor_rejects_invalid_sequence_timeouts(timeout: object) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        OperationExecutor(Target()).execute(
            (),
            OperationContext(boss=load_boss("asylum_demon"), save_state="asylum_demon.sl2"),
            timeout_s=timeout,
        )


@pytest.mark.parametrize("operation_name", ["walk_until_template", "wait_for_victory"])
def test_executor_can_continue_after_an_explicitly_recoverable_timeout(
    operation_name: str,
) -> None:
    class TimeoutTarget(Target):
        def walk_until_template(self, params, **kwargs):
            super().walk_until_template(params, **kwargs)
            raise TimeoutError("prompt did not appear")

        def wait_for_victory(self, boss, **kwargs):
            super().wait_for_victory(boss, **kwargs)
            raise TimeoutError("victory screen did not appear")

    target = TimeoutTarget()
    boss = load_boss("asylum_demon")
    params = (
        {"template": "talk", "continue_on_timeout": True}
        if operation_name == "walk_until_template"
        else {"continue_on_timeout": True}
    )
    with pytest.warns(RuntimeWarning, match=rf"{operation_name}.*timed out"):
        OperationExecutor(target, clock=lambda: 0.0).execute(
            (operation(operation_name, **params), operation("return_to_title")),
            OperationContext(boss=boss, save_state="asylum_demon.sl2"),
            timeout_s=60.0,
        )

    assert [name for name, _ in target.calls] == [operation_name, "return_to_title"]


def test_executor_propagates_timeout_without_explicit_recovery() -> None:
    class TimeoutTarget(Target):
        def walk_until_template(self, params, **kwargs):
            raise TimeoutError("prompt did not appear")

    target = TimeoutTarget()
    boss = load_boss("asylum_demon")
    with pytest.raises(TimeoutError, match="prompt did not appear"):
        OperationExecutor(target, clock=lambda: 0.0).execute(
            (operation("walk_until_template", template="talk"),),
            OperationContext(boss=boss, save_state="asylum_demon.sl2"),
            timeout_s=60.0,
        )
