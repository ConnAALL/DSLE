from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
import pytest

import dsle
from dsle.wrappers import (
    ClipReward,
    FrameStack,
    RecordBossEpisode,
    ResizeObservation,
    ScaleReward,
    make_atari_style,
)
from tests.fakes import ScriptedBackend, state


def test_resize_and_frame_stack_contract() -> None:
    base = dsle.make("asylum_demon", backend=ScriptedBackend(), action_ms=0)
    env = FrameStack(ResizeObservation(base, (84, 84)), count=4)
    observation, _ = env.reset()
    assert observation.shape == (4, 84, 84)
    assert observation.dtype == np.uint8
    assert env.observation_space.contains(observation)
    next_observation, *_ = env.step(0)
    assert next_observation.shape == (4, 84, 84)
    env.close()


def test_clip_reward() -> None:
    backend = ScriptedBackend([state(player_hp=0, player_dead=True)])
    env = ClipReward(dsle.make("asylum_demon", backend=backend, action_ms=0))
    env.reset()
    assert env.step(0)[1] == -1.0
    assert env.reward(-0.25) == -0.25
    assert env.reward(0.25) == 0.25
    assert env.reward(12.0) == 1.0


@pytest.mark.parametrize("shape", [(True, 84), (84.0, 84), (84, 0)])
def test_resize_rejects_noninteger_or_nonpositive_shapes(shape) -> None:
    base = dsle.make("asylum_demon", backend=ScriptedBackend(), action_ms=0)
    try:
        with pytest.raises(ValueError, match="positive integers"):
            ResizeObservation(base, shape)
    finally:
        base.close()


@pytest.mark.parametrize("count", [True, 0, 1.5])
def test_frame_stack_rejects_noninteger_or_nonpositive_counts(count) -> None:
    base = dsle.make("asylum_demon", backend=ScriptedBackend(), action_ms=0)
    try:
        with pytest.raises(ValueError, match="positive integer"):
            FrameStack(base, count=count)
    finally:
        base.close()


@pytest.mark.parametrize("scale", [True, "1", float("nan"), float("inf"), -float("inf")])
def test_scale_reward_rejects_nonfinite_scale(scale) -> None:
    base = dsle.make("asylum_demon", backend=ScriptedBackend(), action_ms=0)
    try:
        with pytest.raises(ValueError, match="finite"):
            ScaleReward(base, scale)
    finally:
        base.close()


def test_atari_style_rejects_vector_environment_and_closes_it(monkeypatch) -> None:
    backends: list[ScriptedBackend] = []

    def factory():
        backend = ScriptedBackend()
        backends.append(backend)
        return dsle.make("asylum_demon", runtime="local", backend=backend, action_ms=0)

    vector = gym.vector.SyncVectorEnv([factory, factory])
    monkeypatch.setattr("dsle.env.make", lambda *_args, **_kwargs: vector)
    with pytest.raises(ValueError, match="one environment"):
        make_atari_style("asylum_demon")
    assert all(backend.closed for backend in backends)


def test_atari_style_closes_single_environment_when_wrapping_fails() -> None:
    backend = ScriptedBackend()
    with pytest.raises(TypeError, match="channel-first"):
        make_atari_style("asylum_demon", backend=backend, obs_mode="state_only", action_ms=0)
    assert backend.closed


def test_episode_recorder_requires_rgb_array_rendering(tmp_path) -> None:
    base = dsle.make("asylum_demon", backend=ScriptedBackend(), action_ms=0)
    with pytest.raises(ValueError, match="render_mode='rgb_array'"):
        RecordBossEpisode(base, tmp_path)
    base.close()


def test_episode_recorder_writes_frames_and_summary(tmp_path) -> None:
    backend = ScriptedBackend([state(player_hp=0, player_dead=True)])
    base = dsle.make("asylum_demon", backend=backend, action_ms=0, render_mode="rgb_array")
    env = RecordBossEpisode(base, tmp_path, chunk_frames=1)
    env.reset()
    env.step(0)
    summary_path = tmp_path / "episode-000001.json"
    chunk_paths = sorted(tmp_path.glob("episode-000001-frames-*.npz"))
    assert summary_path.is_file()
    assert len(chunk_paths) == 2
    assert [np.load(path)["frames"].shape for path in chunk_paths] == [
        (1, 600, 800, 3),
        (1, 600, 800, 3),
    ]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "complete"
    assert summary["frames"] == 2
    assert [chunk["file"] for chunk in summary["chunks"]] == [path.name for path in chunk_paths]


def test_episode_recorder_flushes_chunks_before_episode_end(tmp_path) -> None:
    backend = ScriptedBackend([state(), state(player_hp=0, player_dead=True)])
    base = dsle.make("asylum_demon", backend=backend, action_ms=0, render_mode="rgb_array")
    env = RecordBossEpisode(base, tmp_path, chunk_frames=2)
    env.reset()
    env.step(0)
    chunks = sorted(tmp_path.glob("episode-000001-frames-*.npz"))
    assert len(chunks) == 1
    assert np.load(chunks[0])["frames"].shape == (2, 600, 800, 3)
    assert json.loads((tmp_path / "episode-000001.json").read_text())["status"] == "recording"
    env.step(0)


def test_episode_recorder_selects_episode_at_reset(tmp_path) -> None:
    backend = ScriptedBackend(
        [state(player_hp=0, player_dead=True), state(player_hp=0, player_dead=True)]
    )
    base = dsle.make("asylum_demon", backend=backend, action_ms=0, render_mode="rgb_array")
    env = RecordBossEpisode(base, tmp_path, every=2)
    env.reset()
    env.step(0)
    assert not list(tmp_path.iterdir())
    env.reset()
    assert (tmp_path / "episode-000002.json").is_file()
    env.step(0)


def test_episode_recorder_does_not_overwrite_existing_recordings(tmp_path) -> None:
    (tmp_path / "episode-000001.json").write_text("{}\n", encoding="utf-8")
    backend = ScriptedBackend([state(player_hp=0, player_dead=True)])
    base = dsle.make("asylum_demon", backend=backend, action_ms=0, render_mode="rgb_array")
    env = RecordBossEpisode(base, tmp_path)
    env.reset()
    env.step(0)
    assert (tmp_path / "episode-000001.json").read_text(encoding="utf-8") == "{}\n"
    assert (tmp_path / "episode-000002.json").is_file()


def test_episode_reservation_skips_large_intervals_without_linear_work(tmp_path) -> None:
    base = dsle.make(
        "asylum_demon",
        backend=ScriptedBackend(),
        action_ms=0,
        render_mode="rgb_array",
    )
    env = RecordBossEpisode(base, tmp_path, every=1_000_000_000)
    try:
        assert env._reserve_episode(1) == 1_000_000_000
        assert (tmp_path / "episode-1000000000.json").is_file()
    finally:
        env.close()


def test_episode_summary_update_refuses_a_replacement_symlink(tmp_path) -> None:
    backend = ScriptedBackend([state(player_hp=0, player_dead=True)])
    base = dsle.make(
        "asylum_demon",
        backend=backend,
        action_ms=0,
        render_mode="rgb_array",
    )
    env = RecordBossEpisode(base, tmp_path, chunk_frames=1)
    env.reset()
    summary = tmp_path / "episode-000001.json"
    victim = tmp_path / "victim.json"
    victim.write_text("keep\n", encoding="utf-8")
    summary.unlink()
    summary.symlink_to(victim)
    try:
        with pytest.raises(OSError):
            env.step(0)
        assert victim.read_text(encoding="utf-8") == "keep\n"
    finally:
        summary.unlink()
        base.close()
