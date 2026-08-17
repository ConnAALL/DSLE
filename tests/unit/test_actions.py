from __future__ import annotations

import numpy as np
import pytest

from dsle.actions import ACTION_SPECS, Action, action_names, action_spec

EXPECTED = (
    "move_forward",
    "move_left",
    "move_backward",
    "move_right",
    "light_attack",
    "strong_attack",
    "heal",
    "backstep",
    "roll_forward",
    "roll_left",
    "roll_backward",
    "roll_right",
    "left_click",
    "right_click",
)


def test_action_contract_is_explicit_and_stable() -> None:
    assert len(ACTION_SPECS) == 14
    assert tuple(int(action) for action in Action) == tuple(range(14))
    assert action_names() == EXPECTED


@pytest.mark.parametrize("value", [-1, 14, 999, True, False, 1.0, 1.9, "1"])
def test_action_lookup_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        action_spec(value)


def test_action_lookup_accepts_numpy_integers() -> None:
    assert action_spec(np.int64(2)) is ACTION_SPECS[2]


def test_roll_is_a_simultaneous_direction_space_combo() -> None:
    assert action_spec(Action.ROLL_FORWARD).keys == ("w", "space")
    assert action_spec(Action.STRONG_ATTACK).keys == ("Shift_L",)
    assert action_spec(Action.STRONG_ATTACK).mouse == "left"
