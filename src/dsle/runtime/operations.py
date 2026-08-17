"""Deadline-aware execution of validated declarative runtime operations."""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from dsle.exceptions import ConfigurationError
from dsle.models import BossConfig, EpisodeOutcome, RuntimeOperation
from dsle.progress import LifecycleLogger


class OperationTarget(Protocol):
    """Primitive live operations consumed by :class:`OperationExecutor`."""

    def ensure_menu(self, *, timeout_s: float, threshold: float) -> None: ...

    def load_save(
        self,
        save_state: str,
        *,
        interact: bool,
        settle_before_s: float,
        settle_after_s: float,
        timeout_s: float,
    ) -> None: ...

    def wait_until_ready(
        self, boss: BossConfig, params: Mapping[str, Any], *, timeout_s: float
    ) -> None: ...

    def set_flag(self, name: str, enabled: bool) -> None: ...

    def teleport_player(self, *, x: float | None, y: float | None, z: float | None) -> None: ...

    def tap_key(self, key: str, *, hold_s: float) -> None: ...

    def hold_key(self, key: str, *, duration_s: float) -> None: ...

    def walk_until_template(self, params: Mapping[str, Any], *, timeout_s: float) -> None: ...

    def repeat_actions(
        self,
        actions: Sequence[Any],
        *,
        duration_s: float,
        hold_s: float,
        interval_s: float,
        simultaneous: bool,
    ) -> None: ...

    def wait_for_victory(self, boss: BossConfig, *, timeout_s: float, threshold: float) -> None: ...

    def return_to_title(self) -> None: ...


@dataclass(frozen=True)
class OperationContext:
    """Values supplied by the episode rather than repeated in every operation."""

    boss: BossConfig
    save_state: str
    outcome: EpisodeOutcome | None = None
    default_action_hold_s: float = 0.1
    phase: str = "setup"
    instance: str = "runtime"


def _display_value(value: Any) -> str:
    """Render validated operation parameters without Rich markup interpretation."""

    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{key}={_display_value(item)}" for key, item in value.items()) + "}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "[" + ", ".join(_display_value(item) for item in value) + "]"
    return repr(value) if isinstance(value, str) else str(value)


def describe_operation(operation: RuntimeOperation) -> str:
    """Return a concise public description of one declarative operation."""

    if not operation.params:
        return operation.op
    parameters = ", ".join(
        f"{name}={_display_value(value)}" for name, value in operation.params.items()
    )
    return f"{operation.op} ({parameters})"


def matches_condition(when: tuple[str, ...], outcome: EpisodeOutcome | None) -> bool:
    """Evaluate an operation's condition against terminal episode context."""

    for condition in when:
        if condition == "always":
            return True
        if outcome is None:
            continue
        if condition == "win" and outcome.win:
            return True
        if condition == "loss" and not outcome.win:
            return True
        if condition == "terminated" and outcome.terminated:
            return True
        if condition == "truncated" and outcome.truncated:
            return True
        if condition == "runtime_error" and outcome.reason == "runtime_error":
            return True
    return False


def _text(params: Mapping[str, Any], field: str) -> str:
    value = params.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Runtime operation requires non-empty '{field}'")
    return value.strip()


def _boolean(params: Mapping[str, Any], field: str, default: bool) -> bool:
    value = params.get(field, default)
    if not isinstance(value, bool):
        raise ConfigurationError(f"Runtime operation '{field}' must be a boolean")
    return value


def _number(
    params: Mapping[str, Any],
    field: str,
    default: float | None = None,
    *,
    minimum: float = 0.0,
) -> float:
    value = params.get(field, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"Runtime operation '{field}' must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ConfigurationError(
            f"Runtime operation '{field}' must be finite and at least {minimum}"
        )
    return result


def _optional_coordinate(params: Mapping[str, Any], field: str) -> float | None:
    if params.get(field) is None:
        return None
    return _number(params, field, minimum=-float("inf"))


class OperationExecutor:
    """Dispatch declarative operations with one shared monotonic deadline."""

    def __init__(
        self,
        target: OperationTarget,
        *,
        clock: Any = time.monotonic,
        sleeper: Any = time.sleep,
        lifecycle: LifecycleLogger | None = None,
    ):
        self.target = target
        self._clock = clock
        self._sleep = sleeper
        self._lifecycle = lifecycle or LifecycleLogger(False)

    def execute(
        self,
        operations: tuple[RuntimeOperation, ...],
        context: OperationContext,
        *,
        timeout_s: float,
    ) -> None:
        """Execute applicable operations in order or raise at the shared deadline."""

        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Operation sequence timeout_s must be finite and positive")
        deadline = self._clock() + timeout_s
        selected = tuple(
            operation
            for operation in operations
            if matches_condition(operation.when, context.outcome)
        )
        phase = context.phase.strip().upper() or "OPERATION"
        instance = context.instance.strip() or "runtime"
        for index, operation in enumerate(selected, start=1):
            description = describe_operation(operation)
            prefix = f"{instance} {index}/{len(selected)}"
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise TimeoutError(
                    f"Runtime operation sequence timed out before #{index - 1} ({operation.op})"
                )
            self._lifecycle.event(phase, f"{prefix} starting {description}")
            try:
                self._execute_one(operation, context, remaining=remaining)
            except TimeoutError as exc:
                if operation.params.get("continue_on_timeout") is True:
                    self._lifecycle.event(
                        "WARN",
                        f"{prefix} continuing after optional {operation.op} timeout: {exc}",
                    )
                    warnings.warn(
                        f"Optional runtime operation #{index - 1} ({operation.op}) timed out: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                raise
            self._lifecycle.event(phase, f"{prefix} completed {operation.op}")

    @staticmethod
    def _bounded_timeout(params: Mapping[str, Any], default: float, remaining: float) -> float:
        configured = _number(params, "timeout_s", default, minimum=0.001)
        return min(configured, remaining)

    def _execute_one(
        self, operation: RuntimeOperation, context: OperationContext, *, remaining: float
    ) -> None:
        params = operation.params
        name = operation.op
        if name == "ensure_menu":
            self.target.ensure_menu(
                timeout_s=self._bounded_timeout(params, remaining, remaining),
                threshold=_number(params, "threshold", 0.8, minimum=0.001),
            )
        elif name == "load_save":
            self.target.load_save(
                str(params.get("save_state", context.save_state)),
                interact=_boolean(params, "interact", True),
                settle_before_s=_number(params, "settle_before_s", 0.5),
                settle_after_s=_number(params, "settle_after_s", 1.0),
                timeout_s=self._bounded_timeout(params, remaining, remaining),
            )
        elif name == "wait_until_ready":
            default = context.boss.readiness.timeout_s
            self.target.wait_until_ready(
                context.boss,
                params,
                timeout_s=self._bounded_timeout(params, default, remaining),
            )
        elif name == "sleep":
            seconds = _number(params, "seconds", 0.0)
            if seconds > remaining:
                raise TimeoutError(
                    f"sleep operation needs {seconds:.3f}s but only {remaining:.3f}s remain"
                )
            self._sleep(seconds)
        elif name == "set_flag":
            self.target.set_flag(_text(params, "flag"), _boolean(params, "enable", True))
        elif name == "teleport_player":
            self.target.teleport_player(
                x=_optional_coordinate(params, "x"),
                y=_optional_coordinate(params, "y"),
                z=_optional_coordinate(params, "z"),
            )
        elif name == "tap_key":
            self.target.tap_key(_text(params, "key"), hold_s=_number(params, "hold_s", 0.05))
        elif name == "hold_key":
            duration = _number(
                params, "duration_s", params.get("seconds", context.default_action_hold_s)
            )
            if duration > remaining:
                raise TimeoutError(
                    f"hold_key needs {duration:.3f}s but only {remaining:.3f}s remain"
                )
            self.target.hold_key(_text(params, "key"), duration_s=duration)
        elif name == "walk_until_template":
            self.target.walk_until_template(
                params,
                timeout_s=self._bounded_timeout(
                    params, context.boss.readiness.timeout_s, remaining
                ),
            )
        elif name == "repeat_actions":
            actions = params.get("actions")
            if (
                not isinstance(actions, Sequence)
                or isinstance(actions, (str, bytes))
                or not actions
            ):
                raise ConfigurationError("repeat_actions requires a non-empty 'actions' list")
            duration = _number(params, "duration_s", remaining)
            if duration > remaining:
                raise TimeoutError(
                    f"repeat_actions needs {duration:.3f}s but only {remaining:.3f}s remain"
                )
            self.target.repeat_actions(
                actions,
                duration_s=duration,
                hold_s=_number(params, "hold_s", context.default_action_hold_s),
                interval_s=_number(params, "interval_s", params.get("sleep_s", 0.03)),
                simultaneous=_boolean(params, "simultaneous", False),
            )
        elif name == "wait_for_victory":
            self.target.wait_for_victory(
                context.boss,
                timeout_s=self._bounded_timeout(
                    params, context.boss.limits.cleanup_timeout_s, remaining
                ),
                threshold=_number(
                    params,
                    "threshold",
                    context.boss.victory.template_threshold,
                    minimum=0.001,
                ),
            )
        elif name == "return_to_title":
            self.target.return_to_title()
        else:  # Config loading rejects this, but programmatically built configs should fail clearly.
            raise ConfigurationError(f"Unsupported runtime operation: {name!r}")


__all__ = [
    "OperationContext",
    "OperationExecutor",
    "OperationTarget",
    "describe_operation",
    "matches_condition",
]
