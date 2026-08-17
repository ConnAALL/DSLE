"""Small algorithm-agnostic preprocessing wrappers for DSLE observations."""

from __future__ import annotations

import json
import math
import os
import re
import stat
from collections import deque
from numbers import Real
from pathlib import Path
from typing import Any, SupportsFloat, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _resize_chw(observation: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a CHW image with OpenCV when available and a safe fallback otherwise."""

    array = np.asarray(observation)
    if array.ndim != 3:
        raise ValueError(f"ResizeObservation expects a CHW image, got {array.shape}")
    out_h, out_w = shape
    try:
        import cv2

        hwc = np.transpose(array, (1, 2, 0))
        resized = cv2.resize(hwc, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if resized.ndim == 2:
            resized = resized[..., None]
        return np.transpose(resized, (2, 0, 1)).astype(array.dtype, copy=False)
    except ImportError:
        # Nearest-neighbor fallback keeps the core wrapper usable without the live-runtime extra.
        rows = np.linspace(0, array.shape[1] - 1, out_h).round().astype(np.intp)
        cols = np.linspace(0, array.shape[2] - 1, out_w).round().astype(np.intp)
        return array[:, rows][:, :, cols]


class ResizeObservation(gym.ObservationWrapper):
    """Resize a channel-first visual observation to ``(height, width)``."""

    def __init__(self, env: gym.Env, shape: tuple[int, int] = (84, 84)):
        super().__init__(env)
        if (
            len(shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
            or any(value <= 0 for value in shape)
        ):
            raise ValueError("shape must contain two positive integers")
        if (
            not isinstance(env.observation_space, spaces.Box)
            or len(env.observation_space.shape) != 3
        ):
            raise TypeError("ResizeObservation requires a channel-first Box image observation")
        self.shape = (int(shape[0]), int(shape[1]))
        channels = env.observation_space.shape[0]
        dtype = env.observation_space.dtype
        if dtype is None:
            raise TypeError("ResizeObservation requires an observation dtype")
        self.observation_space = spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=(channels, *self.shape),
            dtype=cast(Any, dtype.type),
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return _resize_chw(observation, self.shape)


class FrameStack(gym.Wrapper):
    """Concatenate the latest ``count`` CHW observations along the channel axis."""

    def __init__(self, env: gym.Env, count: int = 4):
        super().__init__(env)
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer")
        if (
            not isinstance(env.observation_space, spaces.Box)
            or len(env.observation_space.shape) != 3
        ):
            raise TypeError("FrameStack requires a channel-first Box image observation")
        self.count = int(count)
        self._frames: deque[np.ndarray] = deque(maxlen=self.count)
        low = np.concatenate([env.observation_space.low] * self.count, axis=0)
        high = np.concatenate([env.observation_space.high] * self.count, axis=0)
        dtype = env.observation_space.dtype
        if dtype is None:
            raise TypeError("FrameStack requires an observation dtype")
        self.observation_space = spaces.Box(low=low, high=high, dtype=cast(Any, dtype.type))

    def _stack(self) -> np.ndarray:
        return np.concatenate(tuple(self._frames), axis=0)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = self.env.reset(**kwargs)
        self._frames.clear()
        self._frames.extend(np.asarray(observation).copy() for _ in range(self.count))
        return self._stack(), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(np.asarray(observation).copy())
        return self._stack(), float(reward), terminated, truncated, info


class ClipReward(gym.RewardWrapper):
    """Clamp rewards to the closed interval ``[-1, 1]``."""

    def reward(self, reward: SupportsFloat) -> float:
        return max(-1.0, min(1.0, float(reward)))


class ScaleReward(gym.RewardWrapper):
    """Multiply every reward by a fixed scale."""

    def __init__(self, env: gym.Env, scale: float):
        super().__init__(env)
        if isinstance(scale, bool) or not isinstance(scale, Real):
            raise ValueError("scale must be a finite number")
        self.scale: float = float(scale)
        if not math.isfinite(self.scale):
            raise ValueError("scale must be finite")

    def reward(self, reward: SupportsFloat) -> float:
        return float(reward) * self.scale


class RecordBossEpisode(gym.Wrapper):
    """Write chunked RGB frames plus episode metadata without a video codec.

    A recorded episode produces a JSON summary and one or more compressed frame
    chunks named ``episode-NNNNNN-frames-NNNNNN.npz``. Chunking bounds recorder
    memory independently of episode length. The wrapped environment must have
    been created with ``render_mode="rgb_array"``.
    """

    _EPISODE_NAME = re.compile(r"^episode-(\d+)(?:[.-]|$)")

    def __init__(
        self,
        env: gym.Env,
        output_dir: str | Path,
        *,
        every: int = 1,
        chunk_frames: int = 16,
    ):
        super().__init__(env)
        if isinstance(every, bool) or not isinstance(every, int) or every <= 0:
            raise ValueError("every must be positive")
        if isinstance(chunk_frames, bool) or not isinstance(chunk_frames, int) or chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        if getattr(env, "render_mode", None) != "rgb_array":
            raise ValueError(
                "RecordBossEpisode requires an environment configured with render_mode='rgb_array'"
            )
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.every = int(every)
        self.chunk_frames = int(chunk_frames)
        self._episode = self._highest_existing_episode()
        self._episode_id: int | None = None
        self._recording = False
        self._active = False
        self._frame_buffer: list[np.ndarray] = []
        self._frame_count = 0
        self._chunks: list[dict[str, int | str]] = []
        self._return = 0.0

    def _highest_existing_episode(self) -> int:
        highest = 0
        for path in self.output_dir.iterdir():
            match = self._EPISODE_NAME.match(path.name)
            if match is not None:
                highest = max(highest, int(match.group(1)))
        return highest

    def _summary_path(self, episode: int) -> Path:
        return self.output_dir / f"episode-{episode:06d}.json"

    def _reserve_episode(self, candidate: int) -> int:
        """Claim a recording number without overwriting another recorder."""

        while True:
            remainder = candidate % self.every
            if remainder:
                candidate += self.every - remainder
            summary_path = self._summary_path(candidate)
            try:
                with summary_path.open("x", encoding="utf-8") as stream:
                    json.dump(
                        {
                            "episode": candidate,
                            "status": "recording",
                            "frames": 0,
                            "chunks": [],
                        },
                        stream,
                        indent=2,
                    )
                    stream.write("\n")
            except FileExistsError:
                candidate += self.every
                continue
            return candidate

    @staticmethod
    def _write_reserved_summary(path: Path, summary: dict[str, Any]) -> None:
        """Update a claimed summary without following a replacement link."""

        encoded = (json.dumps(summary, indent=2, default=str) + "\n").encode("utf-8")
        flags = os.O_WRONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise RuntimeError(f"Reserved recording summary became unsafe: {path}")
            os.ftruncate(descriptor, 0)
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError(f"Could not update recording summary: {path}")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _flush_frames(self) -> None:
        if not self._frame_buffer:
            return
        if self._episode_id is None:
            raise RuntimeError("recorder has frames without an active episode")
        chunk_number = len(self._chunks) + 1
        path = self.output_dir / (f"episode-{self._episode_id:06d}-frames-{chunk_number:06d}.npz")
        frames = np.stack(self._frame_buffer)
        with path.open("xb") as stream:
            np.savez_compressed(stream, frames=frames)
        self._chunks.append({"file": path.name, "frames": int(frames.shape[0])})
        self._frame_buffer.clear()

    def _capture(self) -> None:
        if not self._recording:
            return
        frame = self.env.render()
        if frame is None:
            raise RuntimeError("rgb_array rendering returned no frame")
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise RuntimeError(f"rgb_array rendering returned an invalid shape: {array.shape}")
        array = np.clip(array, 0, 255).astype(np.uint8) if array.dtype != np.uint8 else array.copy()
        self._frame_buffer.append(array)
        self._frame_count += 1
        if len(self._frame_buffer) >= self.chunk_frames:
            self._flush_frames()

    def _finish_recording(
        self,
        *,
        status: str,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        if not self._recording:
            return
        if self._episode_id is None:
            raise RuntimeError("recorder is active without an episode number")
        self._flush_frames()
        summary = {
            "episode": self._episode_id,
            "status": status,
            "return": self._return,
            "frames": self._frame_count,
            "chunks": self._chunks,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
        }
        self._write_reserved_summary(self._summary_path(self._episode_id), summary)
        self._recording = False
        self._frame_buffer.clear()

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        if self._active and self._recording:
            self._finish_recording(
                status="interrupted",
                terminated=False,
                truncated=False,
                info={"reason": "reset_before_episode_end"},
            )
        observation, info = self.env.reset(**kwargs)
        self._episode = max(self._episode + 1, self._highest_existing_episode() + 1)
        self._episode_id = self._episode
        self._recording = self._episode % self.every == 0
        if self._recording:
            self._episode_id = self._reserve_episode(self._episode)
            self._episode = self._episode_id
        self._active = True
        self._frame_buffer = []
        self._frame_count = 0
        self._chunks = []
        self._return = 0.0
        self._capture()
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._return += float(reward)
        self._capture()
        if terminated or truncated:
            self._finish_recording(
                status="complete",
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            self._active = False
        return observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        try:
            if self._active and self._recording:
                self._finish_recording(
                    status="interrupted",
                    terminated=False,
                    truncated=False,
                    info={"reason": "closed_before_episode_end"},
                )
        finally:
            super().close()


def make_atari_style(boss: str, **env_kwargs: Any) -> gym.Env:
    """Create an 84x84, four-frame, reward-clipped visual DSLE environment."""

    from dsle.env import make

    env_kwargs.setdefault("obs_mode", "grayscale")
    env = make(boss, **env_kwargs)
    if isinstance(env, gym.vector.VectorEnv):
        env.close()
        raise ValueError("make_atari_style supports one environment; set num_instances=1")
    try:
        return ClipReward(FrameStack(ResizeObservation(env, (84, 84)), count=4))
    except Exception:
        env.close()
        raise


__all__ = [
    "ClipReward",
    "FrameStack",
    "RecordBossEpisode",
    "ResizeObservation",
    "ScaleReward",
    "make_atari_style",
]
