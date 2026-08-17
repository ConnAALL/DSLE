#!/usr/bin/env python3
"""Record every supported regular boss as one continuous single-instance video."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import dsle
from dsle.container import ContainerSession, default_output_directory
from dsle.exceptions import DSLEError
from dsle.progress import LifecycleLogger
from dsle.vnc import ReadOnlyVNCClient, VNCEndpoint, VNCFrame

VNC_CONTAINER_PORT = 5901
DEFAULT_FIGHT_SECONDS = 10.0
DEFAULT_MENU_SECONDS = 3.0
DEFAULT_POLL_SECONDS = 0.25
DEFAULT_VIDEO_FPS = 30.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, value: object) -> None:
    """Replace a JSON manifest atomically so interrupted showcases retain progress."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bgr_frame(frame: VNCFrame) -> np.ndarray:
    expected = frame.width * frame.height * 4
    if len(frame.bgrx) != expected:
        raise ValueError(f"VNC frame has {len(frame.bgrx)} bytes; expected {expected}")
    pixels = np.frombuffer(frame.bgrx, dtype=np.uint8).reshape(frame.height, frame.width, 4)
    return pixels[:, :, :3].copy()


class VNCVideoRecorder:
    """Record a passive VNC feed as a constant-frame-rate MP4 without sending input."""

    def __init__(
        self,
        endpoint: VNCEndpoint,
        output: Path,
        *,
        fps: float = DEFAULT_VIDEO_FPS,
        codec: str = "mp4v",
        socket_timeout_s: float = 3.0,
        startup_timeout_s: float = 30.0,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a finite positive number")
        if len(codec) != 4 or not codec.isascii():
            raise ValueError("codec must contain exactly four ASCII characters")
        if socket_timeout_s <= 0 or startup_timeout_s <= 0:
            raise ValueError("VNC timeouts must be positive")
        self.endpoint = endpoint
        self.output = output
        self.fps = float(fps)
        self.codec = codec
        self.socket_timeout_s = float(socket_timeout_s)
        self.startup_timeout_s = float(startup_timeout_s)
        self._monotonic = monotonic
        self._client = ReadOnlyVNCClient(endpoint, timeout=self.socket_timeout_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._writer: Any = None
        self._failure: BaseException | None = None
        self._started_at: float | None = None
        self._last_frame: np.ndarray | None = None
        self.frames_written = 0
        self.width = 0
        self.height = 0

    def _capture_initial_frame(self) -> np.ndarray:
        deadline = self._monotonic() + self.startup_timeout_s
        last_error: BaseException | None = None
        while self._monotonic() < deadline:
            try:
                return _bgr_frame(self._client.capture())
            except Exception as exc:
                last_error = exc
                time.sleep(0.2)
        raise TimeoutError(
            f"VNC feed {self.endpoint.host}:{self.endpoint.port} did not produce a frame "
            f"within {self.startup_timeout_s:g}s: {last_error}"
        )

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("VNC recorder is already started")
        if self.output.exists():
            raise FileExistsError(f"Refusing to overwrite existing video: {self.output}")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "Video recording requires OpenCV; install DSLE with `pip install -e '.[runtime]'`"
            ) from exc

        opencv: Any = cv2
        self.output.parent.mkdir(parents=True, exist_ok=True)
        initial = self._capture_initial_frame()
        self.height, self.width = initial.shape[:2]
        fourcc = opencv.VideoWriter_fourcc(*self.codec)
        writer = opencv.VideoWriter(
            os.fspath(self.output),
            fourcc,
            self.fps,
            (self.width, self.height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"OpenCV could not open {self.output} with codec {self.codec!r}")
        self._writer = writer
        self._last_frame = initial
        self._started_at = self._monotonic()
        self._write_until(self._started_at)
        self._thread = threading.Thread(
            target=self._record,
            name="dsle-vnc-video-recorder",
            daemon=True,
        )
        self._thread.start()

    def _write_until(self, timestamp: float) -> None:
        if self._writer is None or self._last_frame is None or self._started_at is None:
            return
        target_count = max(1, math.ceil(max(0.0, timestamp - self._started_at) * self.fps))
        while self.frames_written < target_count:
            self._writer.write(self._last_frame)
            self.frames_written += 1

    def _record(self) -> None:
        assert self._started_at is not None
        period = 1.0 / self.fps
        next_capture = self._started_at + period
        failure_started: float | None = None
        try:
            while not self._stop.is_set():
                if self._stop.wait(max(0.0, next_capture - self._monotonic())):
                    break
                try:
                    captured = _bgr_frame(self._client.capture())
                    if captured.shape != (self.height, self.width, 3):
                        raise RuntimeError(
                            f"VNC resolution changed from {self.width}x{self.height} to "
                            f"{captured.shape[1]}x{captured.shape[0]} during recording"
                        )
                    now = self._monotonic()
                    self._write_until(now)
                    self._last_frame = captured
                    failure_started = None
                except Exception:
                    now = self._monotonic()
                    failure_started = now if failure_started is None else failure_started
                    if now - failure_started >= 10.0:
                        raise
                next_capture = max(next_capture + period, self._monotonic())
        except BaseException as exc:
            self._failure = exc
            self._stop.set()

    @property
    def elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, self._monotonic() - self._started_at)

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError(f"VNC video recorder failed: {self._failure}") from self._failure

    def stop(self) -> None:
        if self._thread is None:
            self._client.close()
            return
        self._stop.set()
        self._thread.join(timeout=self.socket_timeout_s + 2.0)
        if self._thread.is_alive():
            self._client.close()
            self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            self._failure = self._failure or TimeoutError("VNC recorder thread did not stop")
        self._write_until(self._monotonic())
        self._client.close()
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self.raise_if_failed()
        if not self.output.is_file() or self.output.stat().st_size <= 0:
            raise RuntimeError(f"Video encoder did not produce a non-empty file: {self.output}")


@dataclass(frozen=True)
class FightResult:
    reason: str
    elapsed_seconds: float
    samples: int
    info: dict[str, Any]


def _terminal_reason(info: dict[str, Any]) -> str | None:
    # Match the environment's simultaneous-death rule: a defeated boss wins.
    if info.get("boss_state_valid") is True and info.get("boss_defeated") is True:
        return "boss_defeated"
    if info.get("player_dead") is True:
        return "player_dead"
    return None


def monitor_fight(
    environment: Any,
    initial_info: dict[str, Any],
    *,
    duration_s: float,
    poll_s: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    heartbeat: Callable[[], None] | None = None,
) -> FightResult:
    """Passively monitor until victory, player death, or the wall-clock limit."""

    started = monotonic()
    deadline = started + duration_s
    samples = 0
    info = dict(initial_info)
    while True:
        reason = _terminal_reason(info)
        if reason is not None:
            return FightResult(reason, monotonic() - started, samples, info)
        if heartbeat is not None:
            heartbeat()
        now = monotonic()
        if now >= deadline:
            return FightResult("time_limit", now - started, samples, info)
        sleep(min(poll_s, deadline - now))
        _observation, observed_info = environment.observe()
        info = dict(observed_info)
        samples += 1


def record_boss_segment(
    environment: Any,
    boss: str,
    recorder: VNCVideoRecorder,
    *,
    fight_seconds: float,
    menu_seconds: float,
    poll_seconds: float,
    menu_timeout: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Load, monitor, return to menu, and mark one cut-friendly video segment."""

    setup_started_s = recorder.elapsed_seconds
    _observation, initial_info = environment.reset(options={"difficulty": "standard"})
    fight_started_s = recorder.elapsed_seconds
    result = monitor_fight(
        environment,
        initial_info,
        duration_s=fight_seconds,
        poll_s=poll_seconds,
        monotonic=monotonic,
        sleep=sleep,
        heartbeat=recorder.raise_if_failed,
    )
    fight_ended_s = fight_started_s + result.elapsed_seconds
    environment.return_to_menu(timeout_s=menu_timeout)
    menu_reached_s = recorder.elapsed_seconds
    sleep(menu_seconds)
    menu_pause_ended_s = recorder.elapsed_seconds
    recorder.raise_if_failed()
    return {
        "boss": boss,
        "difficulty": "standard",
        "fight_elapsed_seconds": result.elapsed_seconds,
        "fight_ended_video_s": fight_ended_s,
        "fight_started_video_s": fight_started_s,
        "final_state": {
            "boss_defeated": result.info.get("boss_defeated"),
            "boss_hp": result.info.get("boss_hp"),
            "boss_hps": result.info.get("boss_hps"),
            "player_dead": result.info.get("player_dead"),
            "player_hp": result.info.get("player_hp"),
        },
        "menu_pause_ended_video_s": menu_pause_ended_s,
        "menu_reached_video_s": menu_reached_s,
        "reason": result.reason,
        "samples": result.samples,
        "setup_started_video_s": setup_started_s,
        "status": "passed",
    }


def _select_game_directory(value: Path | None) -> Path:
    configured = value or (
        Path(os.environ["DSLE_GAME_DIR"]) if os.environ.get("DSLE_GAME_DIR") else None
    )
    if configured is not None:
        return configured.expanduser()
    for name in ("Dark.Souls.Remastered.v1.04", "game"):
        candidate = Path.cwd() / name
        if candidate.is_dir():
            return candidate
    raise ValueError(
        "Dark Souls: Remastered was not found in the current directory; pass --game-dir PATH"
    )


def _select_bosses(value: str | None) -> tuple[str, ...]:
    supported = dsle.list_bosses(include_experimental=False)
    if value is None:
        return supported
    selected = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not selected:
        raise ValueError("--bosses must contain at least one boss id")
    unknown = tuple(name for name in selected if name not in supported)
    if unknown:
        raise ValueError(
            f"Unsupported boss(es): {', '.join(unknown)}. Available: {', '.join(supported)}"
        )
    return selected


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "fight_seconds",
        "menu_seconds",
        "poll_seconds",
        "fps",
        "menu_timeout",
        "startup_timeout",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be a finite positive number")
    if len(args.codec) != 4 or not args.codec.isascii() or not args.codec.isprintable():
        raise ValueError("--codec must contain exactly four printable ASCII characters")
    if (
        not args.video_name
        or Path(args.video_name).name != args.video_name
        or args.video_name in {".", ".."}
    ):
        raise ValueError("--video-name must be a file name without directory components")


def run(args: argparse.Namespace) -> tuple[Path, Path, list[dict[str, Any]]]:
    validate_args(args)
    bosses = _select_bosses(args.bosses)
    game_dir = _select_game_directory(args.game_dir)
    output_dir = (
        args.output_dir.expanduser().resolve(strict=False)
        if args.output_dir is not None
        else default_output_directory("boss-showcase")
    )
    lifecycle = LifecycleLogger(not args.quiet)
    artifact_dir = output_dir / "results" / "boss-showcase"
    if artifact_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing showcase: {artifact_dir}")
    session = ContainerSession(
        image=args.image,
        game_dir=game_dir,
        output_dir=output_dir,
        vnc_ports=(VNC_CONTAINER_PORT,),
        verbose=not args.quiet,
        startup_timeout=args.startup_timeout,
        request_timeout=max(180.0, args.menu_timeout + 30.0),
    )
    recorder: VNCVideoRecorder | None = None
    segments: list[dict[str, Any]] = []
    artifact_dir = session.output_dir / "results" / "boss-showcase"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    video_path = artifact_dir / args.video_name
    manifest_path = artifact_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "bosses": list(bosses),
        "codec": args.codec,
        "created_at": _utc_now(),
        "difficulty": "standard",
        "fight_limit_seconds": args.fight_seconds,
        "fps": args.fps,
        "image": args.image,
        "menu_pause_seconds": args.menu_seconds,
        "poll_seconds": args.poll_seconds,
        "segments": segments,
        "status": "starting",
        "video": video_path.name,
    }
    _atomic_json(manifest_path, manifest)
    try:
        session.start()
        status = session.start_instance("dsr-1", mode="headless-vnc")
        if status.get("status") != "ready" or status.get("running") is not True:
            raise RuntimeError(f"dsr-1 did not become ready: {status}")
        host_port = session.vnc_ports.get(VNC_CONTAINER_PORT)
        if host_port is None:
            raise RuntimeError("The managed container did not publish its VNC port")
        recorder = VNCVideoRecorder(
            VNCEndpoint("dsr-1", "127.0.0.1", host_port),
            video_path,
            fps=args.fps,
            codec=args.codec,
        )
        lifecycle.start(f"Recording one continuous VNC video: {video_path}")
        recorder.start()
        manifest["status"] = "recording"
        manifest["vnc_host_port"] = host_port
        _atomic_json(manifest_path, manifest)

        for index, boss in enumerate(bosses, start=1):
            lifecycle.start(f"Showcase boss {index}/{len(bosses)}: {boss}")
            environment = None
            try:
                environment = session.make(
                    boss,
                    instance="dsr-1",
                    start_instance=False,
                    difficulty="standard",
                    obs_mode="state_only",
                    action_ms=0,
                    verbose=not args.quiet,
                )
                segment = record_boss_segment(
                    environment,
                    boss,
                    recorder,
                    fight_seconds=args.fight_seconds,
                    menu_seconds=args.menu_seconds,
                    poll_seconds=args.poll_seconds,
                    menu_timeout=args.menu_timeout,
                )
                segments.append(segment)
                lifecycle.ready(
                    f"{boss}: {segment['reason']}; menu pause recorded through "
                    f"{segment['menu_pause_ended_video_s']:.2f}s"
                )
            except Exception as exc:
                segments.append(
                    {
                        "boss": boss,
                        "difficulty": "standard",
                        "error": f"{type(exc).__name__}: {exc}",
                        "failed_video_s": recorder.elapsed_seconds,
                        "status": "failed",
                    }
                )
                lifecycle.error(f"{boss} failed: {exc}")
                if environment is not None:
                    try:
                        environment.return_to_menu(timeout_s=args.menu_timeout)
                        time.sleep(args.menu_seconds)
                    except Exception as recovery_error:
                        segments[-1]["recovery_error"] = (
                            f"{type(recovery_error).__name__}: {recovery_error}"
                        )
                # A lost recorder invalidates the entire continuous artifact;
                # continuing would only produce a misleading partial showcase.
                recorder.raise_if_failed()
                if args.fail_fast:
                    raise
            finally:
                if environment is not None:
                    environment.close()
                manifest["segments"] = segments
                _atomic_json(manifest_path, manifest)

        failures = [segment for segment in segments if segment["status"] != "passed"]
        manifest["status"] = "complete" if not failures else "complete_with_failures"
        manifest["completed_at"] = _utc_now()
        manifest["failed_bosses"] = [segment["boss"] for segment in failures]
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        recorder_error: BaseException | None = None
        if recorder is not None:
            try:
                recorder.stop()
                manifest["video_frames"] = recorder.frames_written
                manifest["video_height"] = recorder.height
                manifest["video_seconds"] = recorder.elapsed_seconds
                manifest["video_width"] = recorder.width
            except BaseException as exc:
                recorder_error = exc
                manifest["recorder_error"] = f"{type(exc).__name__}: {exc}"
                manifest["status"] = "failed"
        try:
            session.close()
        finally:
            manifest["segments"] = segments
            manifest["updated_at"] = _utc_now()
            _atomic_json(manifest_path, manifest)
        if recorder_error is not None and sys.exc_info()[0] is None:
            raise recorder_error
    return video_path, manifest_path, segments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game-dir",
        type=Path,
        help="game installation; defaults to DSLE_GAME_DIR or a recognized local folder",
    )
    parser.add_argument("--image", default=os.environ.get("DSLE_IMAGE", "dsle-runtime:0.1.0"))
    parser.add_argument("--output-dir", type=Path, help="new or initialized persistent output")
    parser.add_argument(
        "--bosses",
        help="optional comma-separated supported boss subset; defaults to every supported boss",
    )
    parser.add_argument("--fight-seconds", type=float, default=DEFAULT_FIGHT_SECONDS)
    parser.add_argument("--menu-seconds", type=float, default=DEFAULT_MENU_SECONDS)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--menu-timeout", type=float, default=60.0)
    parser.add_argument("--fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--codec", default="mp4v", help="four-character OpenCV codec")
    parser.add_argument("--video-name", default="boss-showcase.mp4")
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        video, manifest, segments = run(build_parser().parse_args(argv))
    except (DSLEError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"boss-showcase: {exc}", file=sys.stderr)
        return 2
    failed = [segment["boss"] for segment in segments if segment["status"] != "passed"]
    print(f"Video: {video}")
    print(f"Cut manifest: {manifest}")
    if failed:
        print(f"Failed bosses: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
