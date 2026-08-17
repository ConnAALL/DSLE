from __future__ import annotations

import time
from argparse import Namespace
from types import SimpleNamespace

import numpy as np
import pytest

from dsle.vnc import VNCFrame
from examples import boss_showcase


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class Recorder:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.heartbeats = 0

    @property
    def elapsed_seconds(self) -> float:
        return self.clock.now

    def raise_if_failed(self) -> None:
        self.heartbeats += 1


class Environment:
    def __init__(self, observations: list[dict[str, object]]) -> None:
        self.observations = list(observations)
        self.menu_calls: list[float] = []

    @staticmethod
    def reset(*, options):
        assert options == {"difficulty": "standard"}
        return np.zeros((2,), dtype=np.float32), {
            "boss_state_valid": True,
            "boss_defeated": False,
            "boss_hp": 1000,
            "boss_hps": [1000],
            "player_dead": False,
            "player_hp": 500,
        }

    def observe(self):
        info = self.observations.pop(0)
        return np.zeros((2,), dtype=np.float32), info

    def return_to_menu(self, *, timeout_s: float) -> None:
        self.menu_calls.append(timeout_s)


def healthy(**changes: object) -> dict[str, object]:
    return {
        "boss_state_valid": True,
        "boss_defeated": False,
        "boss_hp": 1000,
        "boss_hps": [1000],
        "player_dead": False,
        "player_hp": 500,
        **changes,
    }


def test_segment_stops_on_death_then_records_three_seconds_at_menu() -> None:
    clock = Clock()
    recorder = Recorder(clock)
    environment = Environment([healthy(), healthy(player_dead=True, player_hp=0)])

    result = boss_showcase.record_boss_segment(
        environment,
        "asylum_demon",
        recorder,  # type: ignore[arg-type]
        fight_seconds=10.0,
        menu_seconds=3.0,
        poll_seconds=0.5,
        menu_timeout=60.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result["reason"] == "player_dead"
    assert result["fight_elapsed_seconds"] == 1.0
    assert result["fight_ended_video_s"] == 1.0
    assert result["menu_reached_video_s"] == 1.0
    assert result["menu_pause_ended_video_s"] == 4.0
    assert environment.menu_calls == [60.0]
    assert recorder.heartbeats == 3


def test_segment_observes_at_ten_second_boundary_then_returns_to_menu() -> None:
    clock = Clock()
    recorder = Recorder(clock)
    environment = Environment([healthy() for _ in range(4)])

    result = boss_showcase.record_boss_segment(
        environment,
        "taurus_demon",
        recorder,  # type: ignore[arg-type]
        fight_seconds=10.0,
        menu_seconds=3.0,
        poll_seconds=3.0,
        menu_timeout=20.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result["reason"] == "time_limit"
    assert result["samples"] == 4
    assert result["fight_elapsed_seconds"] == 10.0
    assert result["menu_reached_video_s"] == 10.0
    assert result["menu_pause_ended_video_s"] == 13.0
    assert environment.menu_calls == [20.0]


def test_boss_victory_takes_precedence_over_simultaneous_player_death() -> None:
    clock = Clock()
    result = boss_showcase.monitor_fight(
        Environment([]),
        healthy(boss_defeated=True, player_dead=True, boss_hp=0, player_hp=0),
        duration_s=10.0,
        poll_s=0.5,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.reason == "boss_defeated"
    assert result.samples == 0


def test_vnc_bgrx_conversion_preserves_bgr_channels() -> None:
    frame = VNCFrame(
        width=2,
        height=1,
        bgrx=bytes((10, 20, 30, 0, 40, 50, 60, 0)),
        server_name="dsr-1",
    )

    converted = boss_showcase._bgr_frame(frame)

    assert converted.shape == (1, 2, 3)
    assert converted.dtype == np.uint8
    assert converted.tolist() == [[[10, 20, 30], [40, 50, 60]]]


def test_vnc_recorder_writes_a_readable_constant_frame_rate_video(tmp_path, monkeypatch) -> None:
    cv2 = pytest.importorskip("cv2")

    class Client:
        def __init__(self, endpoint, *, timeout):
            assert endpoint.port == 5901
            assert timeout > 0
            self.value = 0

        def capture(self):
            self.value = (self.value + 20) % 256
            pixel = bytes((self.value, 40, 80, 0))
            return VNCFrame(16, 16, pixel * (16 * 16), "dsr-1")

        def close(self):
            pass

    monkeypatch.setattr(boss_showcase, "ReadOnlyVNCClient", Client)
    output = tmp_path / "showcase.mp4"
    recorder = boss_showcase.VNCVideoRecorder(
        SimpleNamespace(name="dsr-1", host="127.0.0.1", port=5901),  # type: ignore[arg-type]
        output,
        fps=10.0,
        socket_timeout_s=0.1,
    )

    recorder.start()
    time.sleep(0.35)
    recorder.stop()

    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 16
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 16
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) >= 3
        readable, frame = capture.read()
        assert readable and frame.shape == (16, 16, 3)
    finally:
        capture.release()


def test_supported_boss_selection_defaults_to_regular_scenarios() -> None:
    bosses = boss_showcase._select_bosses(None)

    assert len(bosses) == 22
    assert "asylum_demon" in bosses
    assert "bed_of_chaos" in bosses
    assert "centipede_demon" in bosses
    assert boss_showcase._select_bosses("asylum_demon,centipede_demon,bed_of_chaos") == (
        "asylum_demon",
        "centipede_demon",
        "bed_of_chaos",
    )


def test_showcase_defaults_to_thirty_fps() -> None:
    assert boss_showcase.build_parser().parse_args([]).fps == 30.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fight_seconds", 0.0),
        ("menu_seconds", -1.0),
        ("poll_seconds", float("nan")),
        ("fps", float("inf")),
        ("menu_timeout", True),
    ],
)
def test_argument_validation_rejects_nonpositive_or_nonfinite_timings(
    field: str, value: object
) -> None:
    values = {
        "fight_seconds": 10.0,
        "menu_seconds": 3.0,
        "poll_seconds": 0.25,
        "fps": 15.0,
        "menu_timeout": 60.0,
        "startup_timeout": 180.0,
        "codec": "mp4v",
        "video_name": "boss-showcase.mp4",
    }
    values[field] = value

    with pytest.raises(ValueError, match="finite positive"):
        boss_showcase.validate_args(Namespace(**values))


def test_recorder_validates_video_settings_before_connecting() -> None:
    endpoint = SimpleNamespace(name="dsr-1", host="127.0.0.1", port=5901)
    with pytest.raises(ValueError, match="fps"):
        boss_showcase.VNCVideoRecorder(endpoint, SimpleNamespace(), fps=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="four ASCII"):
        boss_showcase.VNCVideoRecorder(  # type: ignore[arg-type]
            endpoint,
            SimpleNamespace(),
            codec="bad",
        )
