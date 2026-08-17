from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import dsle
import tests.integration.test_boss_screenshot_sweep as sweep_module
from tests.integration.test_boss_screenshot_sweep import (
    SCREENSHOT_CAPTURE_DELAY_S,
    SUPPORTED_BOSSES,
    ScreenshotSweep,
    _capture_fresh_frame_after_delay,
    _capture_with_one_fresh_runtime_retry,
    _save_lossless_rgb_png,
    _selected_vnc_host_port,
    _start_sweep_instance,
    _wait_until_screenshot_capture,
)


def test_screenshot_png_is_lossless_and_never_overwritten(tmp_path: Path) -> None:
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    frame[..., 0] = np.arange(16, dtype=np.uint8)
    frame[..., 1] = np.arange(12, dtype=np.uint8)[:, None]
    frame[..., 2] = 201
    screenshot = tmp_path / "asylum_demon-standard.png"

    metadata = _save_lossless_rgb_png(screenshot, frame)

    assert metadata["width"] == 16
    assert metadata["height"] == 12
    assert metadata["channels"] == 3
    assert metadata["dtype"] == "uint8"
    assert metadata["bytes"] == screenshot.stat().st_size
    assert metadata["sha256"] == hashlib.sha256(screenshot.read_bytes()).hexdigest()
    decoded = cv2.cvtColor(cv2.imread(str(screenshot)), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(decoded, frame)

    with np.testing.assert_raises(FileExistsError):
        _save_lossless_rgb_png(screenshot, frame)


def test_sweep_targets_only_supported_regular_save_files() -> None:
    assert dsle.list_bosses(include_experimental=False) == SUPPORTED_BOSSES
    assert len(SUPPORTED_BOSSES) == 22
    assert {dsle.load_boss(boss).save_state("standard") for boss in SUPPORTED_BOSSES} == {
        f"{boss}.sl2" for boss in SUPPORTED_BOSSES
    }
    assert all(
        not dsle.load_boss(boss).save_state("standard").endswith("_boosted.sl2")
        for boss in SUPPORTED_BOSSES
    )


def test_vnc_sweep_reuses_dsr_1_and_returns_tigervnc_endpoint() -> None:
    class Session:
        def __init__(self) -> None:
            self.vnc_ports = {5901: 32781}
            self.calls: list[tuple[str, str]] = []

        def start_instance(self, name: str, *, mode: str):
            self.calls.append((name, mode))
            return {"status": "ready", "running": True}

    session = Session()

    viewer = _start_sweep_instance(session, vnc_enabled=True)  # type: ignore[arg-type]

    assert session.calls == [("dsr-1", "headless-vnc")]
    assert viewer == "127.0.0.1::32781"


def test_non_vnc_sweep_starts_headless_without_an_endpoint() -> None:
    class Session:
        def __init__(self) -> None:
            self.vnc_ports: dict[int, int] = {}
            self.calls: list[tuple[str, str]] = []

        def start_instance(self, name: str, *, mode: str):
            self.calls.append((name, mode))
            return {"status": "ready", "running": True}

    session = Session()

    viewer = _start_sweep_instance(session, vnc_enabled=False)  # type: ignore[arg-type]

    assert session.calls == [("dsr-1", "headless")]
    assert viewer is None


def test_vnc_sweep_uses_one_validated_stable_host_port(monkeypatch) -> None:
    monkeypatch.setenv("DSLE_BOSS_SWEEP_VNC_HOST_PORT", "6901")
    assert _selected_vnc_host_port() == 6901


@pytest.mark.parametrize("value", ["", "0", "65536", "5901.0", " 5901", "5901 "])
def test_vnc_sweep_rejects_invalid_host_port(monkeypatch, value: str) -> None:
    monkeypatch.setenv("DSLE_BOSS_SWEEP_VNC_HOST_PORT", value)
    with pytest.raises(pytest.fail.Exception, match=r"\[1, 65535\]"):
        _selected_vnc_host_port()


def test_screenshot_capture_waits_ten_seconds_without_real_sleep() -> None:
    now = [37.5]
    sleep_calls: list[float] = []

    def monotonic() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        now[0] += seconds

    elapsed = _wait_until_screenshot_capture(
        now[0],
        monotonic=monotonic,
        sleep=sleep,
    )

    assert SCREENSHOT_CAPTURE_DELAY_S == 10.0
    assert sleep_calls == [10.0]
    assert elapsed == 10.0


def test_delayed_capture_steps_before_rendering_fresh_frame() -> None:
    now = [100.0]
    calls: list[object] = []
    fresh_frame = np.full((2, 3, 3), 77, dtype=np.uint8)
    transition = (fresh_frame, 0.0, False, True, {"fresh": True})

    class Environment:
        def step(self, action: int):
            calls.append(("step", action, now[0]))
            return transition

        def render(self):
            calls.append(("render", now[0]))
            return fresh_frame

    def monotonic() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        calls.append(("sleep", seconds))
        now[0] += seconds

    frame, actual_transition, elapsed = _capture_fresh_frame_after_delay(
        Environment(),
        now[0],
        monotonic=monotonic,
        sleep=sleep,
    )

    assert calls == [("sleep", 10.0), ("step", 0, 110.0), ("render", 110.0)]
    assert frame is fresh_frame
    assert actual_transition is transition
    assert elapsed == 10.0


def test_pre_capture_failure_retries_once_on_a_fresh_runtime(tmp_path: Path) -> None:
    attempts: list[int] = []

    class Sweep:
        def __init__(self) -> None:
            self.screenshot_dir = tmp_path
            self.records: dict[str, dict[str, object]] = {}
            self.restarts = 0

        def restart_runtime(self) -> None:
            self.restarts += 1

    sweep = Sweep()

    def run_attempt(_sweep, _boss: str, attempt: int) -> None:
        attempts.append(attempt)
        if attempt == 1:
            raise RuntimeError("transient reset failure")

    _capture_with_one_fresh_runtime_retry(
        sweep,  # type: ignore[arg-type]
        SUPPORTED_BOSSES[0],
        run_attempt=run_attempt,
    )

    assert attempts == [1, 2]
    assert sweep.restarts == 1


def test_failure_after_png_is_not_retried_or_overwritten(tmp_path: Path) -> None:
    attempts: list[int] = []

    class Sweep:
        def __init__(self) -> None:
            self.screenshot_dir = tmp_path
            self.records: dict[str, dict[str, object]] = {}
            self.restarts = 0

        def restart_runtime(self) -> None:
            self.restarts += 1

    sweep = Sweep()
    boss = SUPPORTED_BOSSES[-1]

    def run_attempt(_sweep, _boss: str, attempt: int) -> None:
        attempts.append(attempt)
        (tmp_path / f"{boss}-standard.png").write_bytes(b"preserve me")
        raise RuntimeError("post-capture validation failure")

    with np.testing.assert_raises_regex(RuntimeError, "post-capture"):
        _capture_with_one_fresh_runtime_retry(
            sweep,  # type: ignore[arg-type]
            boss,
            run_attempt=run_attempt,
        )

    assert attempts == [1]
    assert sweep.restarts == 0
    assert (tmp_path / f"{boss}-standard.png").read_bytes() == b"preserve me"


def test_manifest_reports_passed_failed_and_missing_bosses(tmp_path: Path) -> None:
    screenshots = tmp_path / "results" / "boss-screenshots"
    screenshots.mkdir(parents=True)
    sweep = ScreenshotSweep(
        session=object(),  # type: ignore[arg-type]
        output_dir=tmp_path,
        screenshot_dir=screenshots,
        image="dsle-runtime:test",
        game_dir=tmp_path / "game",
    )
    passed_boss, failed_boss = SUPPORTED_BOSSES[:2]
    (screenshots / f"{passed_boss}-standard.png").write_bytes(b"expected screenshot")
    (screenshots / "unconfigured-standard.png").write_bytes(b"unexpected screenshot")

    sweep.record(passed_boss, {"boss": passed_boss, "status": "passed"})
    sweep.record(
        failed_boss,
        {"boss": failed_boss, "status": "failed", "error": "reset failed"},
    )

    manifest = json.loads(sweep.manifest_path.read_text(encoding="utf-8"))
    assert manifest["expected_count"] == 22
    assert manifest["screenshot_capture_delay_s"] == 10.0
    assert manifest["passed"] == [passed_boss]
    assert manifest["failed"] == [failed_boss]
    assert set(manifest["missing"]) == set(SUPPORTED_BOSSES[2:])
    assert manifest["screenshot_count"] == 2
    assert set(manifest["missing_screenshots"]) == {
        f"{boss}-standard.png" for boss in SUPPORTED_BOSSES[1:]
    }
    assert manifest["unexpected_screenshots"] == ["unconfigured-standard.png"]
    assert {record["boss"] for record in manifest["records"]} == {
        passed_boss,
        failed_boss,
    }


def test_failed_case_can_replace_its_runtime_without_losing_output(
    tmp_path: Path, monkeypatch
) -> None:
    sessions = []

    class Session:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            self.started = False
            sessions.append(self)

        def start(self):
            self.started = True
            return self

        def start_instance(self, name, *, mode):
            assert (name, mode) == ("dsr-1", "headless")
            return {"status": "ready", "running": True}

        def close(self):
            self.closed = True

    original = Session()
    sweep = ScreenshotSweep(
        session=original,  # type: ignore[arg-type]
        output_dir=tmp_path,
        screenshot_dir=tmp_path / "results" / "boss-screenshots",
        image="dsle-runtime:test",
        game_dir=tmp_path / "game",
    )
    monkeypatch.setattr(sweep_module, "ContainerSession", Session)

    sweep.restart_runtime()

    assert original.closed
    assert sweep.session is sessions[-1]
    assert sessions[-1].started
    assert sessions[-1].kwargs["output_dir"] == tmp_path
    assert sweep.restart_count == 1
    manifest = json.loads(sweep.manifest_path.read_text(encoding="utf-8"))
    assert manifest["restart_count"] == 1
