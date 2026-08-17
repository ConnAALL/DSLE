"""Configure and control isolated game instances inside the runtime container."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any

from dsle._version import __version__
from dsle.exceptions import ConfigurationError, RuntimeUnavailableError

DEFAULT_CONFIG = Path(os.environ.get("DSLE_INSTANCE_CONFIG", "/var/lib/dsle/config/instances.json"))
DEFAULT_STATE_DIR = Path(os.environ.get("DSLE_STATE_DIR", "/var/lib/dsle"))
RUNTIME_MODULE = "dsle.runtime.instances"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _resolution(value: str) -> str:
    pieces = value.lower().split("x", 1)
    if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
        raise argparse.ArgumentTypeError("must look like 800x600")
    width, height = (int(piece) for piece in pieces)
    if not 320 <= width <= 3840 or not 240 <= height <= 2160:
        raise argparse.ArgumentTypeError("must be between 320x240 and 3840x2160")
    return f"{width}x{height}"


def _names(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    if not names:
        raise ConfigurationError("--instances must name at least one instance")
    return names


def _load_instances(path: Path) -> dict[str, Any]:
    loader = _runtime_function("load_instances")
    loaded = loader(path)
    if not isinstance(loaded, dict):
        raise RuntimeUnavailableError(f"`{RUNTIME_MODULE}.load_instances` returned a non-mapping")
    return loaded


def _list_rows(path: Path, instances: dict[str, Any]) -> list[dict[str, Any]]:
    """Include persisted per-instance resolution in human and JSON listings."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        configured = document["instances"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ConfigurationError(f"Could not read instance resolutions from {path}: {exc}") from exc
    if not isinstance(configured, dict):
        raise ConfigurationError(f"{path}: 'instances' must be an object")

    fallback = os.environ.get("DSLE_DESKTOP_RES", "800x600")
    rows: list[dict[str, Any]] = []
    for name, instance in instances.items():
        raw = configured.get(name)
        if not isinstance(raw, dict):
            raise ConfigurationError(f"{path}: instances.{name} must be an object")
        try:
            resolution = _resolution(str(raw.get("resolution") or fallback))
        except argparse.ArgumentTypeError as exc:
            raise ConfigurationError(f"{path}: instances.{name}.resolution {exc}") from exc
        rows.append({**asdict(instance), "resolution": resolution})
    return rows


def _validate_selected(instances: dict[str, Any], names: tuple[str, ...] | None) -> None:
    if names is None:
        return
    known = set(instances)
    missing = [name for name in names if name not in known]
    if missing:
        raise ConfigurationError(
            f"unknown instance(s): {', '.join(missing)}; available: {', '.join(sorted(known))}"
        )


def _runtime_function(name: str) -> Callable[..., Any]:
    try:
        module = importlib.import_module(RUNTIME_MODULE)
    except ImportError as exc:
        raise RuntimeUnavailableError(
            f"live instance control needs optional module `{RUNTIME_MODULE}`; "
            f"configuration/listing still work without it ({exc})"
        ) from exc
    function = getattr(module, name, None)
    if not callable(function):
        raise RuntimeUnavailableError(
            f"`{RUNTIME_MODULE}` does not provide required function `{name}`"
        )
    return function


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _print_result(result: Any, json_output: bool) -> None:
    normalized = _jsonable(result)
    if json_output or isinstance(normalized, (dict, list)):
        print(json.dumps(normalized, indent=2, sort_keys=True))
    elif normalized is not None:
        print(normalized)


def _unready_instances(result: Any) -> list[str]:
    """Return instance names that did not reach the live ready state."""

    if not isinstance(result, Mapping):
        return ["runtime returned no per-instance status"]
    if not result:
        return ["runtime returned an empty instance status"]
    failures: list[str] = []
    for name, value in result.items():
        if not isinstance(value, Mapping):
            failures.append(f"{name}=invalid")
            continue
        status = str(value.get("status", "unknown"))
        if status != "ready" or value.get("running") is not True:
            failures.append(f"{name}={status}")
    return failures


def _configure(args: argparse.Namespace) -> int:
    count = args.count
    if count > 30:
        raise ConfigurationError("at most 30 instances are supported by the default VNC port range")
    state_dir = args.state_dir.expanduser().resolve(strict=False)
    generate = _runtime_function("generate_instances")
    write = _runtime_function("write_instance_config")
    instances = generate(
        count,
        wineprefix_root=state_dir / "wineprefixes",
        display_start=args.base_display,
        vnc_port_start=args.base_vnc_port,
    )
    instances = {
        name: replace(instance, xdg_runtime_dir=state_dir / "xdg" / name)
        for name, instance in instances.items()
    }
    destination = write(
        args.config,
        instances,
        resolution=args.resolution,
        overwrite=args.force,
    )
    print(f"wrote {count} instance(s) to {destination}")
    return 0


def _list(args: argparse.Namespace) -> int:
    instances = _load_instances(args.config)
    rows = _list_rows(args.config, instances)
    if args.json_output:
        _print_result(rows, True)
        return 0
    print("NAME\tDISPLAY\tVNC\tRESOLUTION\tWINEPREFIX")
    for row in rows:
        print(
            f"{row['name']}\t{row['display']}\t{row['vnc_port']}\t"
            f"{row['resolution']}\t{row['wineprefix']}"
        )
    return 0


def _control(args: argparse.Namespace, function_name: str) -> int:
    instances = _load_instances(args.config)
    selected = None if args.all else _names(args.instances)
    _validate_selected(instances, selected)
    function = _runtime_function(function_name)
    kwargs: dict[str, Any] = {"config_path": args.config, "names": selected}
    if function_name == "start_instances":
        kwargs["mode"] = args.mode
    result = function(**kwargs)
    _print_result(result, args.json_output)
    if function_name == "start_instances":
        failures = _unready_instances(result)
        if failures:
            print(
                "dsle-instances: instance start did not reach ready: " + ", ".join(failures),
                file=sys.stderr,
            )
            return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsle-instances",
        description="Configure and control DSLE game instances.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    configure = subparsers.add_parser("configure", help="write deterministic isolation settings")
    configure.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    configure.add_argument("--count", type=_positive_int, default=1)
    configure.add_argument("--base-display", type=int, default=90)
    configure.add_argument("--base-vnc-port", type=int, default=5901)
    configure.add_argument("--resolution", type=_resolution, default="800x600")
    configure.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    configure.add_argument("--force", action="store_true")
    configure.set_defaults(handler=_configure)

    listing = subparsers.add_parser("list", help="show configured isolation settings")
    listing.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    listing.add_argument("--json", action="store_true", dest="json_output")
    listing.set_defaults(handler=_list)

    for command, function_name in (
        ("start", "start_instances"),
        ("status", "instance_status"),
        ("stop", "stop_instances"),
    ):
        control = subparsers.add_parser(command, help=f"{command} configured game processes")
        control.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        selection = control.add_mutually_exclusive_group(required=True)
        selection.add_argument(
            "--all", action="store_true", help="select every configured instance"
        )
        selection.add_argument("--instances", help="comma-separated instance names")
        control.add_argument("--json", action="store_true", dest="json_output")
        if command == "start":
            control.add_argument(
                "--mode",
                choices=("headless", "headless-vnc", "gui"),
                default="headless-vnc",
            )
        control.set_defaults(
            handler=lambda args, selected_function=function_name: _control(args, selected_function)
        )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (ConfigurationError, FileExistsError, RuntimeUnavailableError, ValueError) as exc:
        print(f"dsle-instances: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
