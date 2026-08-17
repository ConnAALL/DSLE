"""Focused XTEST keyboard and mouse injection for one X11 game display."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from dsle.actions import ActionSpec
from dsle.exceptions import RuntimeCommunicationError, RuntimeUnavailableError

KEY_ALIASES = {
    "esc": "Escape",
    "escape": "Escape",
    "enter": "Return",
    "return": "Return",
    "space": "space",
    "backspace": "BackSpace",
    "tab": "Tab",
    "left": "Left",
    "right": "Right",
    "up": "Up",
    "down": "Down",
    "pagedown": "Next",
    "page_down": "Next",
    "next": "Next",
}


def normalize_key_name(key: str) -> str:
    """Normalize human-friendly key aliases to X11 keysym names."""

    value = str(key).strip()
    if not value:
        raise ValueError("Key name cannot be empty")
    return KEY_ALIASES.get(value.lower(), value)


def _xlib() -> tuple[Any, Any, Any, Any]:
    try:
        from Xlib import XK, X
        from Xlib import display as xdisplay
        from Xlib.ext import xtest
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "X11 input needs the optional runtime dependencies. "
            "Install DSLE with `pip install -e '.[runtime]'`."
        ) from exc
    return X, XK, xdisplay, xtest


def _decode_title(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip("\x00 ") or None
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace").strip("\x00 ") or None
    if hasattr(value, "tobytes"):
        return bytes(value.tobytes()).decode("utf-8", errors="replace").strip("\x00 ") or None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class FoundWindow:
    """A located X11 window."""

    window_id: int
    title: str | None


class X11Input:
    """A reusable, serialized X11 connection for one game instance."""

    def __init__(self, display: str):
        normalized = str(display).strip()
        if not normalized:
            raise ValueError("display must be a non-empty X11 display string")
        self.display = normalized
        self._X, self._XK, xdisplay, self._xtest = _xlib()
        try:
            self._connection = xdisplay.Display(normalized)
            if not self._connection.has_extension("XTEST"):
                self._connection.close()
                raise RuntimeUnavailableError(
                    f"XTEST is not available on display {normalized}; synthetic input is required"
                )
            self._root = self._connection.screen().root
        except RuntimeUnavailableError:
            raise
        except Exception as exc:
            raise RuntimeUnavailableError(
                f"Could not connect to X11 display {normalized}: {exc}"
            ) from exc
        self._closed = False
        self._lock = threading.RLock()

    def close(self) -> None:
        """Close the X11 connection; repeated calls are harmless."""

        with self._lock:
            if self._closed:
                return
            self._connection.close()
            self._closed = True

    def __enter__(self) -> X11Input:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeCommunicationError(f"X11 input for {self.display} is closed")

    def _window_title(self, window: Any) -> str | None:
        try:
            title = _decode_title(window.get_wm_name())
            if title:
                return title
        except Exception:
            pass
        try:
            atom = self._connection.intern_atom("_NET_WM_NAME")
            prop = window.get_full_property(atom, self._X.AnyPropertyType)
            return None if prop is None else _decode_title(prop.value)
        except Exception:
            return None

    def _preferred_wine_child(self, window: Any) -> FoundWindow | None:
        try:
            children = window.query_tree().children
        except Exception:
            return None
        for child in children:
            title = self._window_title(child)
            if title and not any(
                marker in title for marker in ("Wine Desktop", "Default IME", "DXGI")
            ):
                return FoundWindow(int(child.id), title)
        return None

    def find_window(self, title: str) -> FoundWindow | None:
        """Depth-first search for an exact or substring title match."""

        target = str(title).strip()
        if not target:
            raise ValueError("Window title cannot be empty")
        with self._lock:
            self._require_open()
            stack = [self._root]
            seen: set[int] = set()
            while stack:
                window = stack.pop()
                try:
                    window_id = int(window.id)
                except Exception:
                    continue
                if window_id in seen:
                    continue
                seen.add(window_id)
                actual = self._window_title(window)
                if actual and (actual == target or target in actual):
                    if "Wine Desktop" in actual:
                        preferred = self._preferred_wine_child(window)
                        if preferred is not None:
                            return preferred
                    return FoundWindow(window_id, actual)
                try:
                    stack.extend(window.query_tree().children)
                except Exception:
                    continue
            return None

    def focus_window(self, title: str, *, allow_root_fallback: bool = True) -> FoundWindow | None:
        """Raise and focus a matching window, optionally falling back to the root window."""

        with self._lock:
            self._require_open()
            found = self.find_window(title)
            if found is None:
                if not allow_root_fallback:
                    raise RuntimeCommunicationError(
                        f"No window containing {title!r} exists on display {self.display}"
                    )
                target = self._root
            else:
                target = self._connection.create_resource_object("window", found.window_id)
                with suppress(Exception):
                    target.raise_window()
            try:
                target.set_input_focus(self._X.RevertToParent, self._X.CurrentTime)
                self._connection.sync()
            except Exception as exc:
                raise RuntimeCommunicationError(
                    f"Could not focus window {title!r} on {self.display}: {exc}"
                ) from exc
            return found

    def _keycode(self, key: str) -> int:
        normalized = normalize_key_name(key)
        keysym = self._XK.string_to_keysym(normalized)
        if keysym == 0 and len(normalized) == 1:
            keysym = self._XK.string_to_keysym(normalized.lower())
        if keysym == 0:
            raise ValueError(f"Unknown X11 key name: {key!r}")
        keycode = int(self._connection.keysym_to_keycode(keysym))
        if keycode == 0:
            raise RuntimeCommunicationError(f"Display {self.display} has no keycode for {key!r}")
        return keycode

    def _key_event(self, event_type: int, key: str) -> None:
        self._xtest.fake_input(self._connection, event_type, self._keycode(key))

    def _button_event(self, event_type: int, button: int) -> None:
        self._xtest.fake_input(self._connection, event_type, button)

    def perform(self, spec: ActionSpec, hold_s: float) -> None:
        """Hold all keys and the optional mouse button as one atomic action."""

        duration = float(hold_s)
        if duration < 0:
            raise ValueError("hold_s cannot be negative")
        button = None if spec.mouse is None else {"left": 1, "right": 3}[spec.mouse]
        pressed: list[str] = []
        button_pressed = False
        with self._lock:
            self._require_open()
            try:
                for key in spec.keys:
                    self._key_event(self._X.KeyPress, key)
                    pressed.append(key)
                if button is not None:
                    self._button_event(self._X.ButtonPress, button)
                    button_pressed = True
                self._connection.sync()
                if duration > 0:
                    time.sleep(duration)
            except Exception as exc:
                if isinstance(exc, (ValueError, RuntimeCommunicationError)):
                    raise
                raise RuntimeCommunicationError(
                    f"Could not inject action {spec.name!r} on {self.display}: {exc}"
                ) from exc
            finally:
                if button_pressed and button is not None:
                    self._button_event(self._X.ButtonRelease, button)
                for key in reversed(pressed):
                    self._key_event(self._X.KeyRelease, key)
                if button_pressed or pressed:
                    self._connection.sync()

    def tap_key(self, key: str, hold_s: float = 0.05) -> None:
        """Tap one key."""

        self.perform(ActionSpec(name=f"tap_{key}", keys=(key,)), hold_s)

    def tap_sequence(
        self,
        keys: Iterable[str],
        *,
        hold_s: float = 0.05,
        interval_s: float = 0.1,
    ) -> None:
        """Tap keys in order with a delay between taps."""

        sequence = tuple(str(key) for key in keys)
        if interval_s < 0:
            raise ValueError("interval_s cannot be negative")
        for index, key in enumerate(sequence):
            self.tap_key(key, hold_s)
            if interval_s and index + 1 < len(sequence):
                time.sleep(interval_s)

    def hold_key_for(self, key: str, duration_s: float) -> None:
        """Hold one key for a fixed duration."""

        self.perform(ActionSpec(name=f"hold_{key}", keys=(key,)), duration_s)

    def move_pointer(self, x: int, y: int) -> None:
        """Move the pointer in root-window coordinates."""

        with self._lock:
            self._require_open()
            self._root.warp_pointer(int(x), int(y))
            self._connection.sync()
