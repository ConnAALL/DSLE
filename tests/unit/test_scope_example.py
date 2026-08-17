from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest

from examples.scope import (
    ScopePolicy,
    chromosome_size,
    evaluate_population,
    parse_instances,
    validate_args,
)


def arguments(**overrides) -> Namespace:
    values = {
        "instances": "dsr-1",
        "k": 100,
        "percentile": 90.0,
        "max_steps": 7200,
        "sigma": 0.5,
        "generations": 40,
        "population": 0,
        "episodes": 100,
        "weights": "best.npy",
    }
    values.update(overrides)
    return Namespace(**values)


def test_instance_pool_rejects_duplicates_before_parallel_evaluation() -> None:
    args = arguments(instances="dsr-1, dsr-1")
    with pytest.raises(ValueError, match=r"duplicates.*dsr-1"):
        evaluate_population([np.zeros(chromosome_size(1))], args)
    assert parse_instances(" dsr-1, dsr-2 ") == ("dsr-1", "dsr-2")


@pytest.mark.parametrize(
    ("overrides", "training", "message"),
    [
        ({"k": 0}, True, "k must"),
        ({"k": 601}, True, "cannot exceed"),
        ({"percentile": float("nan")}, True, "percentile"),
        ({"sigma": 0.0}, True, "sigma"),
        ({"generations": 0}, True, "generations"),
        ({"population": 1}, True, "population"),
        ({"episodes": 0}, False, "episodes"),
        ({"max_steps": 0}, False, "max-steps"),
        ({"weights": None}, False, "weights"),
    ],
)
def test_cli_parameter_validation(overrides, training, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_args(arguments(**overrides), training=training)


def test_scope_policy_rejects_nonfinite_weights_and_bad_observations() -> None:
    weights = np.zeros(chromosome_size(3))
    weights[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ScopePolicy(weights, k=3)

    policy = ScopePolicy(np.zeros(chromosome_size(3)), k=3)
    with pytest.raises(ValueError, match="one-channel CHW"):
        policy.logits(np.zeros((3, 10, 10), dtype=np.uint8))
    with pytest.raises(ValueError, match="smallest spatial dimension"):
        policy.logits(np.zeros((1, 2, 10), dtype=np.uint8))
