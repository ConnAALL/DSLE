from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

import examples.scope_parallel as parallel
from examples.scope import chromosome_size


class FakeEnvironment:
    def __init__(self) -> None:
        self.reset_count = 0
        self.step_index = 0
        self.close_count = 0

    def reset(self):
        self.reset_count += 1
        self.step_index = 0
        return np.zeros((1, 2, 2), dtype=np.uint8), {}

    def step(self, action):
        self.step_index += 1
        terminal = self.step_index == 2
        info = (
            {
                "player_hp": 50,
                "player_hp_max": 100,
                "boss_hp": 25,
                "boss_hp_max": 100,
                "win": True,
                "step": 2,
            }
            if terminal
            else {}
        )
        reward = 2.75 if terminal else 1.25
        return np.zeros((1, 2, 2), dtype=np.uint8), reward, terminal, False, info

    def close(self) -> None:
        self.close_count += 1


class FakeStrategy:
    N = chromosome_size(1)

    def __init__(self, population: int = 3) -> None:
        self.countiter = 0
        self.mean = np.zeros(self.N)
        self.solutions = [np.full(self.N, index, dtype=float) for index in range(population)]
        self.told_values: list[float] = []
        self.tell_count = 0

    def ask(self):
        return self.solutions

    def tell(self, solutions, values) -> None:
        self.told_values = list(values)
        self.countiter += 1
        self.tell_count += 1


class FakeSession:
    created: ClassVar[list[FakeSession]] = []

    def __init__(self, **options) -> None:
        self.options = options
        self.output_dir = Path(options["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started = 0
        self.instance_calls: list[tuple[int, str]] = []
        self.make_calls: list[tuple[str, dict[str, object]]] = []
        self.environments: list[FakeEnvironment] = []
        self.close_count = 0
        self.created.append(self)

    def start(self) -> None:
        self.started += 1

    def start_instances(self, count: int, *, mode: str) -> None:
        self.instance_calls.append((count, mode))

    def make(self, boss: str, **options):
        self.make_calls.append((boss, options))
        environment = FakeEnvironment()
        self.environments.append(environment)
        return environment

    def close(self) -> None:
        self.close_count += 1


def arguments(tmp_path: Path, **overrides) -> Namespace:
    values = {
        "boss": "asylum_demon",
        "difficulty": "standard",
        "game_dir": tmp_path / "game",
        "image": "dsle-runtime:test",
        "output_dir": tmp_path / "output",
        "container_name": None,
        "num_instances": 2,
        "instance_mode": "headless",
        "max_steps": 10,
        "k": 1,
        "percentile": 90.0,
        "sigma": 0.5,
        "generations": 1,
        "population": 3,
        "seed": 42,
        "checkpoint_every": 1,
        "resume_checkpoint": None,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    ("population", "expected_instances"),
    [(1, ["dsr-1"]), (5, ["dsr-1", "dsr-2", "dsr-1", "dsr-2", "dsr-1"])],
)
def test_population_batches_smaller_and_larger_than_pool(
    tmp_path: Path, population: int, expected_instances: list[str]
) -> None:
    workers = [
        parallel.Worker("dsr-1", FakeEnvironment()),
        parallel.Worker("dsr-2", FakeEnvironment()),
    ]

    def evaluator(weights, args, worker):
        marker = float(weights[0])
        return {"fitness": marker, "return": marker, "instance": worker.instance}

    solutions = [np.array([index], dtype=float) for index in range(population)]
    results = parallel.evaluate_population(
        solutions, workers, arguments(tmp_path), evaluator=evaluator
    )
    assert [result["instance"] for result in results] == expected_instances
    assert [result["fitness"] for result in results] == list(map(float, range(population)))


def test_individual_evaluation_resets_and_collects_return(tmp_path, monkeypatch) -> None:
    class FakePolicy:
        def __init__(self, weights, **options) -> None:
            pass

        def act(self, observation) -> int:
            return 0

    monkeypatch.setattr(parallel, "ScopePolicy", FakePolicy)
    environment = FakeEnvironment()
    worker = parallel.Worker("dsr-1", environment)
    weights = np.zeros(chromosome_size(1))

    first = parallel.evaluate_individual(weights, arguments(tmp_path), worker)
    second = parallel.evaluate_individual(weights, arguments(tmp_path), worker)

    assert environment.reset_count == 2
    assert first == second
    assert first["return"] == 4.0
    assert first["fitness"] == 125.0
    assert first["length"] == 2


def test_examples_extra_cma_constructs_and_samples_with_current_numpy(tmp_path) -> None:
    pytest.importorskip("cma", reason="SCOPE optimizer test requires the examples extra")

    strategy = parallel._new_strategy(arguments(tmp_path, population=4))
    solutions = strategy.ask()

    assert len(solutions) == 4
    assert all(np.asarray(solution).size == chromosome_size(1) for solution in solutions)
    strategy.tell(solutions, [0.0, 1.0, 2.0, 3.0])
    assert len(strategy.ask()) == 4


def test_training_owns_one_pool_and_persists_artifacts(tmp_path) -> None:
    FakeSession.created.clear()
    strategy = FakeStrategy()

    def evaluator(weights, args, worker):
        marker = float(weights[0])
        return {
            "fitness": marker + 1.0,
            "return": marker + 10.0,
            "win": marker == 2.0,
            "instance": worker.instance,
        }

    summary = parallel.train(
        arguments(tmp_path, generations=2),
        session_factory=FakeSession,
        strategy=strategy,
        evaluator=evaluator,
    )

    assert len(FakeSession.created) == 1
    session = FakeSession.created[0]
    assert session.started == 1
    assert session.instance_calls == [(2, "headless")]
    assert [call[1]["instance"] for call in session.make_calls] == ["dsr-1", "dsr-2"]
    assert all(call[1]["start_instance"] is False for call in session.make_calls)
    assert all(environment.close_count == 1 for environment in session.environments)
    assert session.close_count == 1
    assert strategy.told_values == [-1.0, -2.0, -3.0]
    assert strategy.tell_count == 2

    artifacts = Path(summary["artifacts"])
    assert np.load(artifacts / "best.npy")[0] == 2.0
    assert (artifacts / "scope-state-latest.npz").is_file()
    assert (artifacts / "checkpoints" / "scope-state-generation-000002.npz").is_file()
    individuals = [
        json.loads(line) for line in (artifacts / "individuals.jsonl").read_text().splitlines()
    ]
    assert [row["instance"] for row in individuals] == [
        "dsr-1",
        "dsr-2",
        "dsr-1",
        "dsr-1",
        "dsr-2",
        "dsr-1",
    ]


def test_training_closes_container_after_evaluation_failure(tmp_path) -> None:
    FakeSession.created.clear()

    def evaluator(weights, args, worker):
        raise RuntimeError("candidate failed")

    with pytest.raises(RuntimeError, match="candidate failed"):
        parallel.train(
            arguments(tmp_path),
            session_factory=FakeSession,
            strategy=FakeStrategy(),
            evaluator=evaluator,
        )

    session = FakeSession.created[0]
    assert all(environment.close_count == 1 for environment in session.environments)
    assert session.close_count == 1


def test_resume_preserves_a_historically_better_candidate(tmp_path) -> None:
    FakeSession.created.clear()
    args = arguments(tmp_path)

    def first_evaluator(weights, args, worker):
        marker = float(weights[0])
        return {"fitness": marker + 1.0, "return": marker, "instance": worker.instance}

    first = parallel.train(
        args,
        session_factory=FakeSession,
        strategy=FakeStrategy(),
        evaluator=first_evaluator,
    )
    artifacts = Path(first["artifacts"])
    original_best = np.load(artifacts / "best.npy", allow_pickle=False).copy()

    resumed_strategy = FakeStrategy()
    resumed_strategy.countiter = 1

    def worse_evaluator(weights, args, worker):
        marker = float(weights[0])
        return {"fitness": marker - 100.0, "return": marker, "instance": worker.instance}

    resumed = parallel.train(
        args,
        session_factory=FakeSession,
        strategy=resumed_strategy,
        evaluator=worse_evaluator,
    )

    np.testing.assert_array_equal(
        np.load(artifacts / "best.npy", allow_pickle=False),
        original_best,
    )
    state = json.loads((artifacts / "state.json").read_text(encoding="utf-8"))
    assert state["best_fitness"] == 3.0
    assert resumed["best_fitness"] == 3.0


def test_resume_checkpoint_is_bounded_data_only_state(tmp_path, monkeypatch) -> None:
    paths = parallel._prepare_artifacts(tmp_path)
    strategy = FakeStrategy()
    strategy.mean = np.arange(strategy.N, dtype=np.float64)
    parallel._save_checkpoint(
        strategy,
        paths,
        generation=7,
        checkpoint_every=1,
        fallback_sigma=0.25,
    )

    state = parallel._load_resume_state(
        paths.latest_checkpoint,
        expected_weights=strategy.N,
    )
    assert state.generation == 7
    assert state.sigma == pytest.approx(0.25)
    np.testing.assert_array_equal(state.mean, strategy.mean)

    captured = {}

    def new_strategy(args, *, initial_mean=None, sigma=None):
        captured["mean"] = initial_mean
        captured["sigma"] = sigma
        return FakeStrategy()

    monkeypatch.setattr(parallel, "_new_strategy", new_strategy)
    resumed = parallel._strategy(arguments(tmp_path, resume_checkpoint=paths.latest_checkpoint))
    assert resumed._dsle_resume_generation == 7
    np.testing.assert_array_equal(captured["mean"], strategy.mean)
    assert captured["sigma"] == pytest.approx(0.25)


def test_resume_rejects_legacy_or_malformed_object_checkpoint(tmp_path) -> None:
    legacy = tmp_path / "legacy.pkl"
    legacy.write_bytes(b"not-a-data-only-numpy-archive")
    with pytest.raises(RuntimeError, match="resume checkpoint"):
        parallel._load_resume_state(legacy, expected_weights=chromosome_size(1))

    compressed = tmp_path / "compressed.npz"
    np.savez_compressed(
        compressed,
        format_version=np.asarray(1),
        generation=np.asarray(1),
        dimension=np.asarray(chromosome_size(1)),
        mean=np.zeros(chromosome_size(1)),
        sigma=np.asarray(0.5),
    )
    with pytest.raises(RuntimeError, match="unsafe or oversized"):
        parallel._load_resume_state(compressed, expected_weights=chromosome_size(1))


@pytest.mark.parametrize("num_instances", [0, 31, True, 1.5])
def test_invalid_pool_size_is_rejected_before_session_creation(tmp_path, num_instances) -> None:
    def unexpected_session(**options):
        raise AssertionError("session should not be allocated")

    with pytest.raises(ValueError, match="num-instances"):
        parallel.train(
            arguments(tmp_path, num_instances=num_instances),
            session_factory=unexpected_session,
            strategy=FakeStrategy(),
        )
