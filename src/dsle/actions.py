"""The single, stable 14-action policy interface used by every boss."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral
from typing import Literal

MouseButton = Literal["left", "right"]


class Action(IntEnum):
    """Integer values exposed through :class:`gymnasium.spaces.Discrete`."""

    MOVE_FORWARD = 0
    MOVE_LEFT = 1
    MOVE_BACKWARD = 2
    MOVE_RIGHT = 3
    LIGHT_ATTACK = 4
    STRONG_ATTACK = 5
    HEAL = 6
    BACKSTEP = 7
    ROLL_FORWARD = 8
    ROLL_LEFT = 9
    ROLL_BACKWARD = 10
    ROLL_RIGHT = 11
    LEFT_CLICK = 12
    RIGHT_CLICK = 13


@dataclass(frozen=True)
class ActionSpec:
    """Keyboard keys and optional mouse button held together for one action."""

    name: str
    keys: tuple[str, ...] = ()
    mouse: MouseButton | None = None


ACTION_SPECS: tuple[ActionSpec, ...] = (
    ActionSpec("move_forward", ("w",)),
    ActionSpec("move_left", ("a",)),
    ActionSpec("move_backward", ("s",)),
    ActionSpec("move_right", ("d",)),
    ActionSpec("light_attack", mouse="left"),
    ActionSpec("strong_attack", ("Shift_L",), mouse="left"),
    ActionSpec("heal", ("r",)),
    ActionSpec("backstep", ("space",)),
    ActionSpec("roll_forward", ("w", "space")),
    ActionSpec("roll_left", ("a", "space")),
    ActionSpec("roll_backward", ("s", "space")),
    ActionSpec("roll_right", ("d", "space")),
    ActionSpec("left_click", mouse="left"),
    ActionSpec("right_click", mouse="right"),
)

LOCK_ON = ActionSpec("lock_on", ("q",))
INTERACT = ActionSpec("interact", ("e",))


def action_spec(action: int | Action) -> ActionSpec:
    """Resolve one integral policy action without silently coercing user input."""

    if isinstance(action, bool) or not isinstance(action, Integral):
        raise ValueError(f"Action must be an integer, got {action!r}")
    index = int(action)
    if not 0 <= index < len(ACTION_SPECS):
        raise ValueError(f"Action index must be in [0, {len(ACTION_SPECS) - 1}], got {action!r}")
    return ACTION_SPECS[index]


def action_names() -> tuple[str, ...]:
    """Return policy action names in integer order."""

    return tuple(spec.name for spec in ACTION_SPECS)
