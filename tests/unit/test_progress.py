from __future__ import annotations

from io import StringIO

from rich.console import Console

from dsle.progress import LifecycleLogger


def test_lifecycle_logger_writes_structured_rich_events() -> None:
    stream = StringIO()
    logger = LifecycleLogger(
        console=Console(file=stream, color_system=None, highlight=False, width=120)
    )

    logger.check("Checking dependencies")
    logger.debug("Container id: abc123")
    logger.ready("Game running: dsr-1")

    assert stream.getvalue().splitlines() == [
        "[CHECK] Checking dependencies",
        "[DEBUG] Container id: abc123",
        "[READY] Game running: dsr-1",
    ]


def test_lifecycle_logger_can_be_disabled() -> None:
    stream = StringIO()
    logger = LifecycleLogger(
        False,
        console=Console(file=stream, color_system=None, highlight=False),
    )

    logger.info("hidden")
    logger.error("also hidden")

    assert stream.getvalue() == ""


def test_lifecycle_logger_forwards_structured_events_and_can_detach_sink() -> None:
    stream = StringIO()
    events: list[tuple[str, str]] = []
    logger = LifecycleLogger(
        console=Console(file=stream, color_system=None, highlight=False),
        event_sink=lambda label, message: events.append((label, message)),
    )

    logger.event("setup", "dsr-2 1/3 starting load_save")
    logger.set_event_sink(None)
    logger.event("combat", "dsr-2 step=0 player_hp=500/500")

    assert events == [("SETUP", "dsr-2 1/3 starting load_save")]
    assert "[SETUP]" in stream.getvalue()
    assert "[COMBAT]" in stream.getvalue()
