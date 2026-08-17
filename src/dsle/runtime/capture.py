"""Thread-safe, per-display X11 capture using MSS."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import numpy as np

from dsle.exceptions import RuntimeCommunicationError, RuntimeUnavailableError

_DISPLAY_LOCKS: dict[str, threading.RLock] = {}
_DISPLAY_LOCKS_GUARD = threading.Lock()


def _display_lock(display: str) -> threading.RLock:
    with _DISPLAY_LOCKS_GUARD:
        return _DISPLAY_LOCKS.setdefault(display, threading.RLock())


def bgra_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert an MSS BGRA/BGR array to contiguous RGB uint8."""

    array = np.asarray(frame, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError(f"Expected an MSS HxWx3/4 frame, got {array.shape}")
    return np.ascontiguousarray(array[..., 2::-1])


def _mss_factory() -> Callable[..., Any]:
    try:
        from mss import mss
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "Screen capture needs the optional runtime dependencies. "
            "Install DSLE with `pip install -e '.[runtime]'`."
        ) from exc
    return mss


class MSSCapture:
    """Capture one X11 display without mutating the process-global ``DISPLAY``."""

    def __init__(
        self,
        display: str,
        monitor_index: int = 0,
        *,
        factory: Callable[..., Any] | None = None,
    ):
        normalized = str(display).strip()
        if not normalized:
            raise ValueError("display must be a non-empty X11 display string")
        if monitor_index < 0:
            raise ValueError("monitor_index cannot be negative")
        self.display = normalized
        self.monitor_index = int(monitor_index)
        self._factory = factory or _mss_factory()
        self._lock = _display_lock(normalized)

    def capture(self) -> np.ndarray:
        """Return one RGB frame from this instance's configured monitor."""

        try:
            with self._lock, self._factory(display=self.display) as session:
                monitors = session.monitors
                if not 0 <= self.monitor_index < len(monitors):
                    raise RuntimeCommunicationError(
                        f"Display {self.display} exposes {len(monitors)} MSS monitor entries; "
                        f"monitor_index={self.monitor_index} is invalid"
                    )
                shot = session.grab(monitors[self.monitor_index])
                return bgra_to_rgb(np.asarray(shot))
        except RuntimeCommunicationError:
            raise
        except Exception as exc:
            raise RuntimeCommunicationError(
                f"Could not capture X11 display {self.display}: {exc}"
            ) from exc
