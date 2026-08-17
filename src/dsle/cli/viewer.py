"""Passive Tk CCTV viewer for one to thirty DSLE VNC feeds."""

from __future__ import annotations

import argparse
import math
import os
import queue
import re
import sys
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

from dsle._version import __version__
from dsle.exceptions import ConfigurationError, RuntimeUnavailableError
from dsle.vnc import ReadOnlyVNCClient, VNCEndpoint, VNCFrame, build_vnc_endpoints

_ENDPOINT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def _bounded_float(value: str, *, field: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise argparse.ArgumentTypeError(f"{field} must be in [{minimum:g}, {maximum:g}]")
    return parsed


def _period(value: str) -> float:
    return _bounded_float(value, field="sampling period", minimum=0.1, maximum=60.0)


def _timeout(value: str) -> float:
    return _bounded_float(value, field="connection timeout", minimum=0.1, maximum=30.0)


def _bounded_int(value: str, *, field: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{field} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise argparse.ArgumentTypeError(f"{field} must be in [{minimum}, {maximum}]")
    return parsed


def _port(value: str) -> int:
    return _bounded_int(value, field="port", minimum=1, maximum=65535)


def _columns(value: str) -> int:
    return _bounded_int(value, field="columns", minimum=1, maximum=30)


def _workers(value: str) -> int:
    return _bounded_int(value, field="workers", minimum=1, maximum=30)


def _tile_dimension(value: str) -> int:
    return _bounded_int(value, field="tile dimension", minimum=80, maximum=1920)


def _selected_names(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise ConfigurationError("--instances must name at least one instance")
    return names


def _explicit_endpoints(values: Sequence[str]) -> tuple[VNCEndpoint, ...]:
    endpoints: list[VNCEndpoint] = []
    for value in values:
        name, separator, address = value.partition("=")
        host, port_separator, port_text = address.rpartition(":")
        if (
            not separator
            or not _ENDPOINT_NAME.fullmatch(name)
            or not port_separator
            or not host.strip()
            or not port_text.isdigit()
            or not 1 <= int(port_text) <= 65535
        ):
            raise ConfigurationError(f"invalid endpoint {value!r}; expected NAME=HOST:PORT")
        endpoints.append(VNCEndpoint(name=name, host=host.strip(), port=int(port_text)))
    if len(set(endpoint.name for endpoint in endpoints)) != len(endpoints):
        raise ConfigurationError("explicit endpoint names must be unique")
    if len(endpoints) > 30:
        raise ConfigurationError("at most 30 viewer endpoints are supported")
    return tuple(endpoints)


def _resolve_endpoints(args: argparse.Namespace) -> tuple[VNCEndpoint, ...]:
    if args.endpoint:
        if args.config is not None or args.instances is not None:
            raise ConfigurationError("--endpoint cannot be combined with --config or --instances")
        return _explicit_endpoints(args.endpoint)
    configured = args.config or os.environ.get("DSLE_INSTANCE_CONFIG")
    if configured is None:
        raise ConfigurationError(
            "provide --config PATH, set DSLE_INSTANCE_CONFIG, or pass --endpoint NAME=HOST:PORT"
        )
    endpoints = build_vnc_endpoints(
        configured,
        host=args.host,
        host_port_start=args.base_port,
        selected_names=_selected_names(args.instances),
    )
    if len(endpoints) > 30:
        raise ConfigurationError("at most 30 viewer endpoints are supported")
    return endpoints


class _CCTVWindow:
    """Tk widgets and asynchronous sampling; all game connections are passive."""

    def __init__(
        self,
        root: Any,
        tk: Any,
        image_module: Any,
        image_tk_module: Any,
        endpoints: tuple[VNCEndpoint, ...],
        *,
        period: float,
        timeout: float,
        columns: int,
        workers: int,
        tile_width: int,
        tile_height: int,
        title: str,
    ):
        self.root = root
        self.tk = tk
        self.image_module = image_module
        self.image_tk_module = image_tk_module
        self.endpoints = endpoints
        self.period = period
        self.timeout = timeout
        self.columns = min(columns, len(endpoints))
        self.workers = min(workers, len(endpoints))
        self.tile_width, self.tile_height = self._fit_tiles(tile_width, tile_height)
        self.closed = False
        self.paused = False
        self.pending: set[str] = set()
        self.results: queue.SimpleQueue[tuple[str, VNCFrame | None, str | None]] = (
            queue.SimpleQueue()
        )
        self.clients = {
            endpoint.name: ReadOnlyVNCClient(endpoint, timeout=timeout) for endpoint in endpoints
        }
        self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="dsle-cctv")
        self.image_labels: dict[str, Any] = {}
        self.status_labels: dict[str, Any] = {}
        self.images: dict[str, Any] = {}
        self.period_variable = tk.StringVar(value=f"{period:g}")
        self.summary_variable = tk.StringVar()
        self.root.title(title)
        self.root.configure(background="#080c12")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._build_widgets()
        self._update_summary()
        self.root.after(0, self._refresh)
        self.root.after(40, self._drain_results)

    def _fit_tiles(self, requested_width: int, requested_height: int) -> tuple[int, int]:
        rows = math.ceil(len(self.endpoints) / self.columns)
        available_width = max(320, self.root.winfo_screenwidth() - 80)
        available_height = max(240, self.root.winfo_screenheight() - 150)
        width = min(requested_width, max(80, available_width // self.columns - 14))
        height = min(requested_height, max(60, available_height // rows - 58))
        return width, height

    def _build_widgets(self) -> None:
        toolbar = self.tk.Frame(self.root, background="#111827", padx=10, pady=8)
        toolbar.grid(row=0, column=0, columnspan=self.columns, sticky="ew")
        self.tk.Label(
            toolbar,
            text="DSLE Instance Viewer",
            background="#111827",
            foreground="#f9fafb",
            font=("TkDefaultFont", 11, "bold"),
        ).pack(side="left", padx=(0, 18))
        self.tk.Label(
            toolbar,
            text="Sampling period (seconds):",
            background="#111827",
            foreground="#d1d5db",
        ).pack(side="left")
        period_entry = self.tk.Entry(
            toolbar, textvariable=self.period_variable, width=7, justify="center"
        )
        period_entry.pack(side="left", padx=6)
        period_entry.bind("<Return>", lambda _event: self._apply_period())
        self.tk.Button(toolbar, text="Apply", command=self._apply_period).pack(side="left")
        self.pause_button = self.tk.Button(toolbar, text="Pause", command=self._toggle_pause)
        self.pause_button.pack(side="left", padx=(8, 0))
        self.tk.Label(
            toolbar,
            textvariable=self.summary_variable,
            background="#111827",
            foreground="#93c5fd",
        ).pack(side="right")

        for index, endpoint in enumerate(self.endpoints):
            row, column = divmod(index, self.columns)
            panel = self.tk.Frame(
                self.root,
                background="#111827",
                highlightbackground="#374151",
                highlightthickness=1,
                padx=4,
                pady=4,
            )
            panel.grid(row=row + 1, column=column, padx=4, pady=4, sticky="nsew")
            self.root.grid_columnconfigure(column, weight=1)
            self.root.grid_rowconfigure(row + 1, weight=1)
            self.tk.Label(
                panel,
                text=endpoint.name,
                background="#111827",
                foreground="#f9fafb",
                font=("TkDefaultFont", 10, "bold"),
            ).pack(fill="x")
            image_label = self.tk.Label(
                panel,
                text="Connecting…",
                width=max(12, self.tile_width // 9),
                height=max(4, self.tile_height // 18),
                background="#030712",
                foreground="#9ca3af",
            )
            image_label.pack(expand=True, fill="both")
            status_label = self.tk.Label(
                panel,
                text=f"{endpoint.host}:{endpoint.port}",
                anchor="w",
                background="#111827",
                foreground="#9ca3af",
            )
            status_label.pack(fill="x")
            self.image_labels[endpoint.name] = image_label
            self.status_labels[endpoint.name] = status_label

    def _apply_period(self) -> None:
        try:
            self.period = _period(self.period_variable.get())
        except argparse.ArgumentTypeError:
            self.period_variable.set(f"{self.period:g}")
            self.summary_variable.set("Sampling period must be between 0.1 and 60 seconds")
            return
        self.period_variable.set(f"{self.period:g}")
        self._update_summary()

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_button.configure(text="Resume" if self.paused else "Pause")
        self._update_summary()

    def _update_summary(self) -> None:
        state = "paused" if self.paused else f"every {self.period:g}s"
        self.summary_variable.set(f"{len(self.endpoints)} feeds · {state}")

    def _complete(self, name: str, future: Future[VNCFrame]) -> None:
        try:
            self.results.put((name, future.result(), None))
        except Exception as exc:
            self.results.put((name, None, f"{type(exc).__name__}: {exc}"))

    def _refresh(self) -> None:
        if self.closed:
            return
        if not self.paused:
            for endpoint in self.endpoints:
                if endpoint.name in self.pending:
                    continue
                self.pending.add(endpoint.name)
                future = self.executor.submit(self.clients[endpoint.name].capture)
                future.add_done_callback(partial(self._complete, endpoint.name))
        self.root.after(max(100, round(self.period * 1000)), self._refresh)

    def _display_frame(self, name: str, frame: VNCFrame) -> None:
        image = self.image_module.frombytes(
            "RGB", (frame.width, frame.height), frame.bgrx, "raw", "BGRX"
        )
        image.thumbnail((self.tile_width, self.tile_height), self.image_module.Resampling.BILINEAR)
        photo = self.image_tk_module.PhotoImage(image=image)
        self.images[name] = photo
        self.image_labels[name].configure(image=photo, text="", width=0, height=0)
        endpoint = self.clients[name].endpoint
        timestamp = time.strftime("%H:%M:%S")
        self.status_labels[name].configure(
            text=f"{endpoint.host}:{endpoint.port} · {timestamp}", foreground="#86efac"
        )

    def _drain_results(self) -> None:
        if self.closed:
            return
        while True:
            try:
                name, frame, error = self.results.get_nowait()
            except queue.Empty:
                break
            self.pending.discard(name)
            if frame is not None:
                self._display_frame(name, frame)
            else:
                short_error = (error or "unavailable").replace("\n", " ")[:180]
                self.status_labels[name].configure(text=short_error, foreground="#fca5a5")
        self.root.after(40, self._drain_results)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for client in self.clients.values():
            client.close()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def run_viewer(args: argparse.Namespace) -> int:
    """Resolve endpoints, import optional GUI dependencies, and run Tk."""

    endpoints = _resolve_endpoints(args)
    try:
        import tkinter as tk

        from PIL import Image, ImageTk  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "the CCTV viewer needs the optional viewer dependencies; run "
            "`python -m pip install '.[viewer]'` and, on Ubuntu, "
            "`sudo apt install python3-tk`"
        ) from exc
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeUnavailableError(
            f"could not open the CCTV window; check the host DISPLAY: {exc}"
        ) from exc
    columns = args.columns or math.ceil(math.sqrt(len(endpoints)))
    workers = args.workers or min(12, len(endpoints))
    _CCTVWindow(
        root,
        tk,
        Image,
        ImageTk,
        endpoints,
        period=args.period,
        timeout=args.timeout,
        columns=columns,
        workers=workers,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        title=args.title,
    )
    root.mainloop()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsle-viewer",
        description=(
            "Passively sample up to 30 DSLE VNC feeds in one lightweight CCTV-style window."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", type=Path, help="host path to instances.json")
    parser.add_argument("--host", default="127.0.0.1", help="VNC host for configured instances")
    parser.add_argument(
        "--base-port",
        type=_port,
        default=5901,
        help="host port corresponding to the first configured instance (default: 5901)",
    )
    parser.add_argument("--instances", help="comma-separated configured instance names")
    parser.add_argument(
        "--endpoint",
        action="append",
        default=[],
        metavar="NAME=HOST:PORT",
        help="explicit feed; repeat for managed containers with dynamic host ports",
    )
    parser.add_argument(
        "--period",
        type=_period,
        default=1.0,
        help="seconds between samples for each feed (default: 1.0)",
    )
    parser.add_argument("--timeout", type=_timeout, default=3.0)
    parser.add_argument("--columns", type=_columns, help="grid columns (default: automatic)")
    parser.add_argument("--workers", type=_workers, help="parallel captures (default: up to 12)")
    parser.add_argument("--tile-width", type=_tile_dimension, default=320)
    parser.add_argument("--tile-height", type=_tile_dimension, default=240)
    parser.add_argument("--title", default="DSLE Instance Viewer")
    parser.set_defaults(handler=run_viewer)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ConfigurationError, RuntimeUnavailableError, ValueError, OSError) as exc:
        print(f"dsle-viewer: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
