from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

import dsle
from dsle.container import ContainerSession, default_output_directory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_SCREENSHOT_SWEEP = os.environ.get("DSLE_RUN_BOSS_SCREENSHOT_SWEEP") == "1"
SUPPORTED_BOSSES = dsle.list_bosses(include_experimental=False)
SCREENSHOT_CAPTURE_DELAY_S = 10.0
VNC_CONTAINER_PORT = 5901

pytestmark = [
    pytest.mark.docker,
    pytest.mark.game,
    pytest.mark.boss,
    pytest.mark.slow,
    pytest.mark.skipif(
        not RUN_SCREENSHOT_SWEEP,
        reason="set DSLE_RUN_BOSS_SCREENSHOT_SWEEP=1 to run the regular-save screenshot sweep",
    ),
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _wait_until_screenshot_capture(
    reset_completed_monotonic: float,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Wait until exactly ten seconds after reset completed.

    The injected clock and sleeper keep this timing contract regression-testable
    without making the ordinary unit suite wait for the live-game delay.
    """

    deadline = float(reset_completed_monotonic) + SCREENSHOT_CAPTURE_DELAY_S
    remaining = deadline - monotonic()
    while remaining > 0:
        sleep(remaining)
        remaining = deadline - monotonic()
    return monotonic() - float(reset_completed_monotonic)


def _capture_fresh_frame_after_delay(
    environment: Any,
    reset_completed_monotonic: float,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[np.ndarray, tuple[Any, ...], float]:
    """Wait, refresh the cached observation with one zero-time step, then render it."""

    elapsed = _wait_until_screenshot_capture(
        reset_completed_monotonic,
        monotonic=monotonic,
        sleep=sleep,
    )
    transition = environment.step(0)
    frame = environment.render()
    return frame, transition, elapsed


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_lossless_rgb_png(path: Path, frame: np.ndarray) -> dict[str, Any]:
    rgb = np.asarray(frame)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise AssertionError(f"Expected HxWx3 uint8 RGB frame, got {rgb.shape} {rgb.dtype}")
    if int(rgb.min()) == int(rgb.max()):
        raise AssertionError("Refusing to save a uniform frame; the game display was not visible")

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite screenshot from another run: {path}")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    encoded, payload = cv2.imencode(
        ".png",
        bgr,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    if not encoded:
        raise AssertionError(f"OpenCV could not encode {path.name} as PNG")
    png = payload.tobytes()
    with path.open("xb") as stream:
        stream.write(png)

    decoded_bgr = cv2.imread(os.fspath(path), cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise AssertionError(f"OpenCV could not reopen the written screenshot: {path}")
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)
    if not np.array_equal(decoded_rgb, rgb):
        raise AssertionError(f"PNG round trip changed pixels for {path}")

    height, width, channels = rgb.shape
    return {
        "bytes": len(png),
        "channels": channels,
        "dtype": str(rgb.dtype),
        "height": height,
        "maximum": int(rgb.max()),
        "mean": float(rgb.mean()),
        "minimum": int(rgb.min()),
        "sha256": hashlib.sha256(png).hexdigest(),
        "width": width,
    }


@dataclass
class ScreenshotSweep:
    session: ContainerSession
    output_dir: Path
    screenshot_dir: Path
    image: str
    game_dir: Path
    vnc_enabled: bool = False
    vnc_host_port: int | None = None
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    restart_count: int = 0

    @property
    def manifest_path(self) -> Path:
        return self.screenshot_dir / "manifest.json"

    def record(self, boss: str, result: dict[str, Any]) -> None:
        self.records[boss] = result
        _atomic_json(self.screenshot_dir / f"{boss}-standard.json", result)
        self.write_manifest()

    def write_manifest(self) -> None:
        passed = sorted(
            boss for boss, result in self.records.items() if result.get("status") == "passed"
        )
        failed = sorted(
            boss for boss, result in self.records.items() if result.get("status") == "failed"
        )
        missing = sorted(set(SUPPORTED_BOSSES) - set(self.records))
        expected_screenshots = {f"{boss}-standard.png" for boss in SUPPORTED_BOSSES}
        actual_screenshots = {
            path.name for path in self.screenshot_dir.glob("*.png") if path.is_file()
        }
        _atomic_json(
            self.manifest_path,
            {
                "difficulty": "standard",
                "expected_bosses": list(SUPPORTED_BOSSES),
                "expected_count": len(SUPPORTED_BOSSES),
                "failed": failed,
                "image": self.image,
                "missing": missing,
                "missing_screenshots": sorted(expected_screenshots - actual_screenshots),
                "passed": passed,
                "records": [self.records[boss] for boss in sorted(self.records)],
                "restart_count": self.restart_count,
                "screenshot_capture_delay_s": SCREENSHOT_CAPTURE_DELAY_S,
                "screenshot_count": len(actual_screenshots),
                "unexpected_screenshots": sorted(actual_screenshots - expected_screenshots),
                "updated_at": _utc_now(),
            },
        )

    def restart_runtime(self) -> None:
        """Replace a tainted game process without losing host artifacts."""

        self.session.close()
        replacement = ContainerSession(
            image=self.image,
            game_dir=self.game_dir,
            output_dir=self.output_dir,
            startup_timeout=180,
            request_timeout=300,
            vnc_ports=(VNC_CONTAINER_PORT,) if self.vnc_enabled else (),
            vnc_host_ports=(
                {VNC_CONTAINER_PORT: self.vnc_host_port}
                if self.vnc_enabled and self.vnc_host_port is not None
                else None
            ),
        )
        try:
            replacement.start()
            viewer = _start_sweep_instance(replacement, vnc_enabled=self.vnc_enabled)
            if viewer is not None:
                _announce_vnc(viewer, replacement=True)
        except Exception:
            replacement.close()
            raise
        self.session = replacement
        self.restart_count += 1
        self.write_manifest()


def _selected_output_directory() -> Path:
    configured = os.environ.get("DSLE_BOSS_SWEEP_OUTPUT_DIR")
    if configured:
        selected = Path(configured).expanduser().resolve(strict=False)
        if selected.exists() and any(selected.iterdir()):
            pytest.fail(
                "DSLE_BOSS_SWEEP_OUTPUT_DIR must be new or empty so screenshots are never "
                f"overwritten: {selected}"
            )
        return selected
    return default_output_directory("regular-boss-screenshot-sweep")


def _selected_vnc_host_port() -> int:
    value = os.environ.get("DSLE_BOSS_SWEEP_VNC_HOST_PORT", "5901")
    if not value.isascii() or not value.isdecimal():
        pytest.fail("DSLE_BOSS_SWEEP_VNC_HOST_PORT must be an integer in [1, 65535]")
    port = int(value)
    if not 1 <= port <= 65535:
        pytest.fail("DSLE_BOSS_SWEEP_VNC_HOST_PORT must be an integer in [1, 65535]")
    return port


def _start_sweep_instance(
    session: ContainerSession,
    *,
    vnc_enabled: bool,
) -> str | None:
    mode = "headless-vnc" if vnc_enabled else "headless"
    status = session.start_instance("dsr-1", mode=mode)
    if status.get("status") != "ready" or status.get("running") is not True:
        raise AssertionError(f"dsr-1 was not ready in {mode} mode: {status}")
    if not vnc_enabled:
        return None
    host_port = session.vnc_ports.get(VNC_CONTAINER_PORT)
    if isinstance(host_port, bool) or not isinstance(host_port, int) or not 1 <= host_port <= 65535:
        raise AssertionError(f"Missing VNC host mapping for port {VNC_CONTAINER_PORT}")
    return f"127.0.0.1::{host_port}"


def _announce_vnc(viewer: str, *, replacement: bool = False) -> None:
    qualifier = "replacement runtime" if replacement else "serial boss sweep"
    print(
        f"\nDSLE VNC ({qualifier}): vncviewer {viewer}\n"
        "The same endpoint shows dsr-1 while standard boss saves load one by one.\n",
        file=sys.__stdout__,
        flush=True,
    )


@pytest.fixture(scope="module")
def screenshot_sweep() -> Iterator[ScreenshotSweep]:
    if os.environ.get("PYTEST_XDIST_WORKER"):
        pytest.fail("The live boss screenshot sweep must run serially; do not use pytest-xdist")
    game_value = os.environ.get("DSLE_GAME_DIR")
    assert game_value, "DSLE_GAME_DIR must point to a legally obtained game installation"
    game_dir = Path(game_value).expanduser().resolve(strict=True)
    image = os.environ.get("DSLE_DOCKER_TEST_IMAGE", "dsle-runtime:0.1.0")
    vnc_enabled = os.environ.get("DSLE_BOSS_SWEEP_VNC") == "1"
    vnc_host_port = _selected_vnc_host_port() if vnc_enabled else None
    output_dir = _selected_output_directory()
    session = ContainerSession(
        image=image,
        game_dir=game_value,
        output_dir=output_dir,
        startup_timeout=180,
        request_timeout=300,
        vnc_ports=(VNC_CONTAINER_PORT,) if vnc_enabled else (),
        vnc_host_ports=(
            {VNC_CONTAINER_PORT: vnc_host_port}
            if vnc_enabled and vnc_host_port is not None
            else None
        ),
    )
    sweep: ScreenshotSweep | None = None
    try:
        session.start()
        viewer = _start_sweep_instance(session, vnc_enabled=vnc_enabled)
        if viewer is not None:
            _announce_vnc(viewer)
        screenshot_dir = session.output_dir / "results" / "boss-screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=False)
        sweep = ScreenshotSweep(
            session=session,
            output_dir=session.output_dir,
            screenshot_dir=screenshot_dir,
            image=image,
            game_dir=game_dir,
            vnc_enabled=vnc_enabled,
            vnc_host_port=vnc_host_port,
        )
        sweep.write_manifest()
        print(f"\nDSLE regular-boss screenshots: {screenshot_dir}", flush=True)
        yield sweep
    finally:
        if sweep is not None:
            sweep.write_manifest()
            sweep.session.close()
        else:
            session.close()


def _capture_supported_standard_save_and_write_screenshot(
    screenshot_sweep: ScreenshotSweep,
    boss: str,
    attempt: int,
) -> None:
    config = dsle.load_boss(boss)
    save_state = config.save_state("standard")
    assert save_state == f"{boss}.sl2"
    assert not save_state.endswith("_boosted.sl2")

    screenshot_path = screenshot_sweep.screenshot_dir / f"{boss}-standard.png"
    started_at = _utc_now()
    reset_completed_at: str | None = None
    reset_completed_monotonic: float | None = None
    capture_elapsed_after_reset_s: float | None = None
    captured_at: str | None = None
    info: dict[str, Any] | None = None
    final_info: dict[str, Any] | None = None
    png: dict[str, Any] | None = None
    environment = None
    try:
        environment = screenshot_sweep.session.make(
            boss,
            instance="dsr-1",
            start_instance=False,
            difficulty="standard",
            render_mode="rgb_array",
            action_ms=0,
            max_steps=1,
        )
        observation, info = environment.reset(options={"difficulty": "standard"})
        reset_completed_at = _utc_now()
        reset_completed_monotonic = time.monotonic()
        assert environment.observation_space.contains(observation)
        frame, transition, capture_elapsed_after_reset_s = _capture_fresh_frame_after_delay(
            environment,
            reset_completed_monotonic,
        )
        captured_at = _utc_now()
        next_observation, reward, terminated, truncated, final_info = transition
        assert capture_elapsed_after_reset_s >= SCREENSHOT_CAPTURE_DELAY_S
        assert frame is not None
        assert frame.shape == (600, 800, 3)
        assert frame.dtype == np.uint8
        png = _save_lossless_rgb_png(screenshot_path, frame)

        # The zero-duration transition above refreshes render()'s cached reset
        # observation and guarantees outcome-conditioned cleanup, even when a
        # loaded save exposes a bad state such as zero HP.
        assert environment.observation_space.contains(next_observation)
        assert isinstance(reward, float) and np.isfinite(reward)
        assert terminated or truncated
        assert "runtime_error" not in final_info
        assert "cleanup_error" not in final_info
        environment.close()
        environment = None

        assert info["boss"] == boss
        assert info["difficulty"] == "standard"
        assert info["player_state_valid"] is True
        assert info["boss_state_valid"] is True
        assert isinstance(info["player_hp"], int) and info["player_hp"] > 0
        assert isinstance(info["player_hp_max"], int) and info["player_hp_max"] > 0
        assert isinstance(info["boss_hp"], int) and info["boss_hp"] > 0
        assert isinstance(info["boss_hp_max"], int) and info["boss_hp_max"] > 0
        assert info["boss_defeated"] is False
        assert info["player_dead"] is False

        screenshot_sweep.record(
            boss,
            {
                "attempt": attempt,
                "boss": boss,
                "capture_elapsed_after_reset_s": capture_elapsed_after_reset_s,
                "captured_at": captured_at,
                "capture_state": {
                    "boss_defeated": final_info.get("boss_defeated"),
                    "boss_hp": final_info.get("boss_hp"),
                    "boss_hp_max": final_info.get("boss_hp_max"),
                    "boss_state_valid": final_info.get("boss_state_valid"),
                    "player_dead": final_info.get("player_dead"),
                    "player_hp": final_info.get("player_hp"),
                    "player_hp_max": final_info.get("player_hp_max"),
                    "player_state_valid": final_info.get("player_state_valid"),
                },
                "display_name": config.display_name,
                "difficulty": "standard",
                "finished": {
                    "terminated": terminated,
                    "terminated_reason": final_info.get("terminated_reason"),
                    "truncated": truncated,
                    "truncated_reason": final_info.get("truncated_reason"),
                },
                "initial_state": {
                    "boss_hp": info["boss_hp"],
                    "boss_hp_max": info["boss_hp_max"],
                    "player_hp": info["player_hp"],
                    "player_hp_max": info["player_hp_max"],
                },
                "png": png,
                "reset_completed_at": reset_completed_at,
                "save_state": save_state,
                "screenshot": screenshot_path.name,
                "screenshot_capture_delay_s": SCREENSHOT_CAPTURE_DELAY_S,
                "started_at": started_at,
                "status": "passed",
            },
        )
    except Exception as exc:
        close_error = None
        failure_capture = None
        failure_capture_error = None
        if environment is not None and png is None:
            try:
                failure_path = (
                    screenshot_sweep.screenshot_dir
                    / "failures"
                    / f"{boss}-standard-reset-failure.png"
                )
                failure_frame = environment.render()
                if failure_frame is None:
                    raise AssertionError("The environment had no renderable failure frame")
                failure_capture = {
                    "png": _save_lossless_rgb_png(failure_path, failure_frame),
                    "screenshot": str(failure_path.relative_to(screenshot_sweep.screenshot_dir)),
                }
            except Exception as capture_exc:
                failure_capture_error = f"{type(capture_exc).__name__}: {capture_exc}"
        if environment is not None:
            try:
                environment.close()
            except Exception as cleanup_exc:  # retain both the primary and close failures
                close_error = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
        result = {
            "attempt": attempt,
            "boss": boss,
            "capture_elapsed_after_reset_s": capture_elapsed_after_reset_s,
            "capture_state": (
                None
                if final_info is None
                else {
                    "boss_defeated": final_info.get("boss_defeated"),
                    "boss_hp": final_info.get("boss_hp"),
                    "boss_hp_max": final_info.get("boss_hp_max"),
                    "boss_state_valid": final_info.get("boss_state_valid"),
                    "player_dead": final_info.get("player_dead"),
                    "player_hp": final_info.get("player_hp"),
                    "player_hp_max": final_info.get("player_hp_max"),
                    "player_state_valid": final_info.get("player_state_valid"),
                    "terminated_reason": final_info.get("terminated_reason"),
                    "truncated_reason": final_info.get("truncated_reason"),
                }
            ),
            "captured_at": captured_at,
            "close_error": close_error,
            "difficulty": "standard",
            "error": f"{type(exc).__name__}: {exc}",
            "failure_capture": failure_capture,
            "failure_capture_error": failure_capture_error,
            "initial_state": (
                None
                if info is None
                else {
                    "boss_hp": info.get("boss_hp"),
                    "boss_hp_max": info.get("boss_hp_max"),
                    "boss_state_valid": info.get("boss_state_valid"),
                    "player_hp": info.get("player_hp"),
                    "player_hp_max": info.get("player_hp_max"),
                    "player_state_valid": info.get("player_state_valid"),
                }
            ),
            "png": png,
            "reset_completed_at": reset_completed_at,
            "save_state": save_state,
            "screenshot": screenshot_path.name if screenshot_path.exists() else None,
            "screenshot_capture_delay_s": SCREENSHOT_CAPTURE_DELAY_S,
            "started_at": started_at,
            "status": "failed",
            "traceback": traceback.format_exc(),
        }
        screenshot_sweep.record(boss, result)
        raise


def _capture_with_one_fresh_runtime_retry(
    screenshot_sweep: ScreenshotSweep,
    boss: str,
    *,
    run_attempt: Callable[[ScreenshotSweep, str, int], None] = (
        _capture_supported_standard_save_and_write_screenshot
    ),
) -> None:
    """Retry one pre-screenshot failure after replacing the game runtime."""

    screenshot_path = screenshot_sweep.screenshot_dir / f"{boss}-standard.png"
    for attempt in (1, 2):
        try:
            run_attempt(screenshot_sweep, boss, attempt)
        except Exception:
            can_retry = attempt == 1 and not screenshot_path.exists()
            if can_retry or boss != SUPPORTED_BOSSES[-1]:
                try:
                    screenshot_sweep.restart_runtime()
                except Exception as restart_exc:
                    result = screenshot_sweep.records.get(boss)
                    if result is not None:
                        result["restart_error"] = f"{type(restart_exc).__name__}: {restart_exc}"
                        screenshot_sweep.record(boss, result)
                    raise
            if can_retry:
                continue
            raise
        else:
            return
    raise AssertionError(f"Retry loop ended without a result for {boss}")


@pytest.mark.parametrize("boss", SUPPORTED_BOSSES)
def test_every_supported_standard_save_loads_and_writes_screenshot(
    screenshot_sweep: ScreenshotSweep,
    boss: str,
) -> None:
    _capture_with_one_fresh_runtime_retry(screenshot_sweep, boss)


def test_screenshot_manifest_covers_every_supported_boss(
    screenshot_sweep: ScreenshotSweep,
) -> None:
    screenshot_sweep.write_manifest()
    expected = set(SUPPORTED_BOSSES)
    assert set(screenshot_sweep.records) == expected
    failures = {
        boss: result.get("error")
        for boss, result in screenshot_sweep.records.items()
        if result.get("status") != "passed"
    }
    assert not failures, f"Boss screenshot failures: {failures}"
    expected_screenshots = {f"{boss}-standard.png" for boss in expected}
    assert {
        path.name for path in screenshot_sweep.screenshot_dir.glob("*.png") if path.is_file()
    } == expected_screenshots
    assert screenshot_sweep.manifest_path.is_file()
    manifest = json.loads(screenshot_sweep.manifest_path.read_text(encoding="utf-8"))
    assert manifest["screenshot_count"] == len(expected_screenshots)
    assert manifest["missing_screenshots"] == []
    assert manifest["unexpected_screenshots"] == []
