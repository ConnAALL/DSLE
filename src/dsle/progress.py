"""Small Rich-based lifecycle messages for public DSLE workflows."""

from __future__ import annotations

from collections.abc import Callable

from rich.console import Console
from rich.text import Text

_STYLES = {
    "CHECK": "bold cyan",
    "CLEANUP": "bold yellow",
    "COMBAT": "bold blue",
    "DEBUG": "dim cyan",
    "DONE": "bold green",
    "ERROR": "bold red",
    "INFO": "bold blue",
    "READY": "bold green",
    "SETUP": "bold magenta",
    "START": "bold magenta",
    "STOP": "bold yellow",
    "WARN": "bold yellow",
    "WRITE": "bold cyan",
}

EventSink = Callable[[str, str], None]


class LifecycleLogger:
    """Write concise, structured lifecycle events when verbosity is enabled."""

    def __init__(
        self,
        verbose: bool = True,
        *,
        console: Console | None = None,
        event_sink: EventSink | None = None,
    ):
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean")
        if event_sink is not None and not callable(event_sink):
            raise TypeError("event_sink must be callable or None")
        self.verbose = verbose
        self._console = console or Console(stderr=True, highlight=False, soft_wrap=True)
        self._event_sink = event_sink

    def set_event_sink(self, event_sink: EventSink | None) -> None:
        """Set the temporary transport used to forward runtime events to a client."""

        if event_sink is not None and not callable(event_sink):
            raise TypeError("event_sink must be callable or None")
        self._event_sink = event_sink

    def event(self, label: str, message: str) -> None:
        """Print one tagged event without interpreting user-controlled markup."""

        if not self.verbose:
            return
        normalized = label.strip().upper()
        rendered_message = str(message)
        if self._event_sink is not None:
            self._event_sink(normalized, rendered_message)
        line = Text.assemble(
            (f"[{normalized}]", _STYLES.get(normalized, "bold")),
            " ",
            rendered_message,
        )
        self._console.print(line)

    def check(self, message: str) -> None:
        self.event("CHECK", message)

    def debug(self, message: str) -> None:
        self.event("DEBUG", message)

    def done(self, message: str) -> None:
        self.event("DONE", message)

    def error(self, message: str) -> None:
        self.event("ERROR", message)

    def info(self, message: str) -> None:
        self.event("INFO", message)

    def ready(self, message: str) -> None:
        self.event("READY", message)

    def start(self, message: str) -> None:
        self.event("START", message)

    def stop(self, message: str) -> None:
        self.event("STOP", message)


__all__ = ["EventSink", "LifecycleLogger"]
