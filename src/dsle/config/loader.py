"""Strict loader for defaults and one-YAML-per-boss task definitions."""

from __future__ import annotations

import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from copy import deepcopy
from difflib import get_close_matches
from importlib.resources import as_file, files
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from dsle.actions import ACTION_SPECS
from dsle.exceptions import ConfigurationError
from dsle.models import (
    BossConfig,
    BossMetadata,
    EnvironmentDefaults,
    EpisodeLimits,
    MemoryConfig,
    ReadinessConfig,
    RewardConfig,
    RuntimeOperation,
    VictoryConfig,
    immutable_mapping,
)

SUPPORTED_OPERATIONS = {
    "ensure_menu",
    "load_save",
    "wait_until_ready",
    "sleep",
    "set_flag",
    "teleport_player",
    "tap_key",
    "hold_key",
    "walk_until_template",
    "repeat_actions",
    "wait_for_victory",
    "return_to_title",
}
VALID_WHEN = {"always", "win", "loss", "terminated", "truncated", "runtime_error"}
_BOSS_ID = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")
_SAFE_SAVE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_ACTION_NAMES = {action.name for action in ACTION_SPECS}
_MISSING = object()
_MAX_YAML_BYTES = 1 << 20


def _unknown_keys(
    raw: Mapping[str, Any],
    allowed: set[str],
    field: str,
    source: Path,
) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigurationError(f"{source}: '{field}' has unknown key(s): {', '.join(unknown)}")


def _bool(
    raw: Mapping[str, Any],
    name: str,
    field: str,
    source: Path,
    *,
    default: object = _MISSING,
) -> bool:
    value = raw.get(name, default)
    if value is _MISSING or not isinstance(value, bool):
        raise ConfigurationError(f"{source}: '{field}.{name}' must be a boolean")
    return value


def _integer(
    raw: Mapping[str, Any],
    name: str,
    field: str,
    source: Path,
    *,
    default: object = _MISSING,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = raw.get(name, default)
    if value is _MISSING or isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{source}: '{field}.{name}' must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigurationError(f"{source}: '{field}.{name}' must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigurationError(f"{source}: '{field}.{name}' must be at most {maximum}")
    return value


def _number(
    raw: Mapping[str, Any],
    name: str,
    field: str,
    source: Path,
    *,
    default: object = _MISSING,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    value = raw.get(name, default)
    if value is _MISSING or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{source}: '{field}.{name}' must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigurationError(f"{source}: '{field}.{name}' must be finite")
    if minimum is not None and (result < minimum or (strict_minimum and result == minimum)):
        comparison = "greater than" if strict_minimum else "at least"
        raise ConfigurationError(f"{source}: '{field}.{name}' must be {comparison} {minimum:g}")
    if maximum is not None and result > maximum:
        raise ConfigurationError(f"{source}: '{field}.{name}' must be at most {maximum:g}")
    return result


def _text(
    raw: Mapping[str, Any],
    name: str,
    field: str,
    source: Path,
    *,
    default: object = _MISSING,
) -> str:
    value = raw.get(name, default)
    if value is _MISSING or not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{source}: '{field}.{name}' must be non-empty text")
    return value.strip()


def _safe_save_name(value: object, field: str, source: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{source}: '{field}' must be a non-empty save filename")
    name = value.strip()
    candidate = Path(name)
    if (
        candidate.name != name
        or candidate.suffix.casefold() != ".sl2"
        or _SAFE_SAVE_NAME.fullmatch(name) is None
    ):
        raise ConfigurationError(f"{source}: '{field}' must be one relative .sl2 filename")
    return name


def _mapping(value: Any, field: str, source: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{source}: '{field}' must be a mapping")
    return dict(value)


def _sequence(value: Any, field: str, source: Path) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigurationError(f"{source}: '{field}' must be a list")
    return list(value)


def _read_yaml(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise OSError("configuration source is not a regular file")
            if metadata.st_size > _MAX_YAML_BYTES:
                raise OSError(f"configuration source exceeds {_MAX_YAML_BYTES} bytes")
            chunks: list[bytes] = []
            remaining = _MAX_YAML_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, min(64 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            encoded = b"".join(chunks)
            if len(encoded) > _MAX_YAML_BYTES:
                raise OSError(f"configuration source exceeds {_MAX_YAML_BYTES} bytes")
        finally:
            os.close(descriptor)
        raw = yaml.safe_load(encoded.decode("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError, RecursionError) as exc:
        raise ConfigurationError(f"Could not load {path}: {exc}") from exc
    return _mapping(raw, "document", path)


def _reward(raw: Mapping[str, Any], field: str, source: Path) -> RewardConfig:
    defaults = RewardConfig()
    allowed = set(defaults.__dict__)
    _unknown_keys(raw, allowed, field, source)
    return RewardConfig(
        **{
            name: _number(raw, name, field, source, default=default)
            for name, default in defaults.__dict__.items()
        }
    )


def _limits(raw: Mapping[str, Any], field: str, source: Path) -> EpisodeLimits:
    _unknown_keys(raw, set(EpisodeLimits().__dict__), field, source)
    cfg = EpisodeLimits(
        max_steps=_integer(raw, "max_steps", field, source, default=7200, minimum=1),
        setup_timeout_s=_number(
            raw,
            "setup_timeout_s",
            field,
            source,
            default=120.0,
            minimum=0.0,
            strict_minimum=True,
        ),
        cleanup_timeout_s=_number(
            raw,
            "cleanup_timeout_s",
            field,
            source,
            default=60.0,
            minimum=0.0,
            strict_minimum=True,
        ),
        minimum_victory_step=_integer(
            raw,
            "minimum_victory_step",
            field,
            source,
            default=1,
            minimum=1,
        ),
        victory_confirmations=_integer(
            raw,
            "victory_confirmations",
            field,
            source,
            default=1,
            minimum=1,
        ),
    )
    return cfg


def _readiness(raw: Mapping[str, Any], field: str, source: Path) -> ReadinessConfig:
    _unknown_keys(raw, set(ReadinessConfig().__dict__), field, source)
    template_value = raw.get("template", "traverseLight")
    if template_value is not None and (
        not isinstance(template_value, str) or not template_value.strip()
    ):
        raise ConfigurationError(f"{source}: '{field}.template' must be text or null")
    cfg = ReadinessConfig(
        mode=_text(raw, "mode", field, source, default="traverse_light"),
        template=None if template_value is None else template_value.strip(),
        threshold=_number(
            raw,
            "threshold",
            field,
            source,
            default=0.9,
            minimum=0.0,
            maximum=1.0,
            strict_minimum=True,
        ),
        timeout_s=_number(
            raw,
            "timeout_s",
            field,
            source,
            default=60.0,
            minimum=0.0,
            strict_minimum=True,
        ),
        advance_key=_text(raw, "advance_key", field, source, default="w"),
        advance_hold_s=_number(raw, "advance_hold_s", field, source, default=0.05, minimum=0.0),
        poll_s=_number(
            raw,
            "poll_s",
            field,
            source,
            default=0.3,
            minimum=0.0,
            strict_minimum=True,
        ),
        interact_after_ready=_bool(raw, "interact_after_ready", field, source, default=True),
    )
    if cfg.mode not in {"none", "template", "traverse_light"}:
        raise ConfigurationError(f"{source}: unsupported {field}.mode: {cfg.mode}")
    if cfg.mode != "none" and cfg.template is None:
        raise ConfigurationError(f"{source}: '{field}.template' is required for {cfg.mode}")
    return cfg


_OPERATION_KEYS: dict[str, set[str]] = {
    "ensure_menu": {"timeout_s", "threshold"},
    "load_save": {
        "save_state",
        "interact",
        "settle_before_s",
        "settle_after_s",
        "timeout_s",
    },
    "wait_until_ready": {
        "mode",
        "template",
        "threshold",
        "timeout_s",
        "poll_s",
        "advance_key",
        "advance_hold_s",
        "interact_after_ready",
    },
    "sleep": {"seconds"},
    "set_flag": {"flag", "enable"},
    "teleport_player": {"x", "y", "z"},
    "tap_key": {"key", "hold_s"},
    "hold_key": {"key", "duration_s", "seconds"},
    "walk_until_template": {
        "template",
        "timeout_s",
        "continue_on_timeout",
        "key",
        "hold_s",
        "poll_s",
        "threshold",
    },
    "repeat_actions": {
        "actions",
        "duration_s",
        "hold_s",
        "interval_s",
        "sleep_s",
        "simultaneous",
    },
    "wait_for_victory": {"timeout_s", "threshold", "continue_on_timeout"},
    "return_to_title": set(),
}


def _optional_number(
    raw: Mapping[str, Any],
    name: str,
    field: str,
    source: Path,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> None:
    if name in raw:
        _number(
            raw,
            name,
            field,
            source,
            minimum=minimum,
            maximum=maximum,
            strict_minimum=strict_minimum,
        )


def _optional_text(raw: Mapping[str, Any], name: str, field: str, source: Path) -> None:
    if name in raw:
        _text(raw, name, field, source)


def _optional_bool(raw: Mapping[str, Any], name: str, field: str, source: Path) -> None:
    if name in raw:
        _bool(raw, name, field, source)


def _validate_thresholds_and_timeouts(raw: Mapping[str, Any], field: str, source: Path) -> None:
    _optional_number(
        raw,
        "threshold",
        field,
        source,
        minimum=0.0,
        maximum=1.0,
        strict_minimum=True,
    )
    _optional_number(raw, "timeout_s", field, source, minimum=0.0, strict_minimum=True)


def _validate_repeat_actions(raw: Mapping[str, Any], field: str, source: Path) -> None:
    actions = _sequence(raw.get("actions"), f"{field}.actions", source)
    if not actions:
        raise ConfigurationError(f"{source}: '{field}.actions' cannot be empty")
    for index, action in enumerate(actions):
        action_field = f"{field}.actions[{index}]"
        if isinstance(action, str):
            name = action.strip()
            if not name:
                raise ConfigurationError(f"{source}: '{action_field}' cannot be empty")
        else:
            entry = _mapping(action, action_field, source)
            _unknown_keys(entry, {"name", "action", "hold_s"}, action_field, source)
            if ("name" in entry) == ("action" in entry):
                raise ConfigurationError(
                    f"{source}: '{action_field}' requires exactly one of name/action"
                )
            name = _text(
                entry,
                "name" if "name" in entry else "action",
                action_field,
                source,
            )
            _optional_number(entry, "hold_s", action_field, source, minimum=0.0)
        if name not in _ACTION_NAMES:
            raise ConfigurationError(f"{source}: '{action_field}' uses unknown action {name!r}")


def _validate_operation(
    op: str,
    raw: Mapping[str, Any],
    field: str,
    source: Path,
) -> None:
    _unknown_keys(raw, _OPERATION_KEYS[op], field, source)
    _validate_thresholds_and_timeouts(raw, field, source)

    if op == "load_save":
        if "save_state" in raw:
            _safe_save_name(raw["save_state"], f"{field}.save_state", source)
        _optional_bool(raw, "interact", field, source)
        _optional_number(raw, "settle_before_s", field, source, minimum=0.0)
        _optional_number(raw, "settle_after_s", field, source, minimum=0.0)
    elif op == "wait_until_ready":
        if "mode" in raw and _text(raw, "mode", field, source) not in {
            "none",
            "template",
            "traverse_light",
        }:
            raise ConfigurationError(f"{source}: '{field}.mode' is unsupported")
        _optional_text(raw, "template", field, source)
        _optional_text(raw, "advance_key", field, source)
        _optional_number(raw, "poll_s", field, source, minimum=0.0, strict_minimum=True)
        _optional_number(raw, "advance_hold_s", field, source, minimum=0.0)
        _optional_bool(raw, "interact_after_ready", field, source)
    elif op == "sleep":
        _number(raw, "seconds", field, source, minimum=0.0)
    elif op == "set_flag":
        flag = _text(raw, "flag", field, source)
        if flag not in {"no_dead", "no_damage"}:
            raise ConfigurationError(f"{source}: '{field}.flag' is unsupported: {flag}")
        _optional_bool(raw, "enable", field, source)
    elif op == "teleport_player":
        if not any(raw.get(axis) is not None for axis in ("x", "y", "z")):
            raise ConfigurationError(f"{source}: '{field}' requires x, y, or z")
        for axis in ("x", "y", "z"):
            if raw.get(axis) is not None:
                _number(raw, axis, field, source)
    elif op == "tap_key":
        _text(raw, "key", field, source)
        _optional_number(raw, "hold_s", field, source, minimum=0.0)
    elif op == "hold_key":
        _text(raw, "key", field, source)
        if "duration_s" in raw and "seconds" in raw:
            raise ConfigurationError(f"{source}: '{field}' cannot combine duration_s and seconds")
        if "duration_s" in raw:
            _number(raw, "duration_s", field, source, minimum=0.0)
        if "seconds" in raw:
            _number(raw, "seconds", field, source, minimum=0.0)
    elif op == "walk_until_template":
        _text(raw, "template", field, source)
        _optional_bool(raw, "continue_on_timeout", field, source)
        _optional_text(raw, "key", field, source)
        _optional_number(raw, "hold_s", field, source, minimum=0.0)
        _optional_number(raw, "poll_s", field, source, minimum=0.0, strict_minimum=True)
    elif op == "repeat_actions":
        _validate_repeat_actions(raw, field, source)
        _optional_number(raw, "duration_s", field, source, minimum=0.0, strict_minimum=True)
        _optional_number(raw, "hold_s", field, source, minimum=0.0)
        if "interval_s" in raw and "sleep_s" in raw:
            raise ConfigurationError(f"{source}: '{field}' cannot combine interval_s and sleep_s")
        _optional_number(raw, "interval_s", field, source, minimum=0.0)
        _optional_number(raw, "sleep_s", field, source, minimum=0.0)
        _optional_bool(raw, "simultaneous", field, source)
    elif op == "wait_for_victory":
        _optional_bool(raw, "continue_on_timeout", field, source)


def _operations(raw: Any, field: str, source: Path) -> tuple[RuntimeOperation, ...]:
    operations: list[RuntimeOperation] = []
    for index, item in enumerate(_sequence(raw, field, source)):
        entry = _mapping(item, f"{field}[{index}]", source)
        op = str(entry.pop("op", "")).strip()
        if op not in SUPPORTED_OPERATIONS:
            raise ConfigurationError(
                f"{source}: {field}[{index}] uses unsupported operation {op!r}"
            )
        raw_when = entry.pop("when", "always")
        when = (
            (str(raw_when),)
            if isinstance(raw_when, str)
            else tuple(str(x) for x in _sequence(raw_when, f"{field}[{index}].when", source))
        )
        unknown_when = set(when) - VALID_WHEN
        if unknown_when:
            raise ConfigurationError(
                f"{source}: invalid operation condition(s): {sorted(unknown_when)}"
            )
        if not when or len(set(when)) != len(when):
            raise ConfigurationError(
                f"{source}: {field}[{index}].when must be non-empty and unique"
            )
        _validate_operation(op, entry, f"{field}[{index}]", source)
        operations.append(RuntimeOperation(op=op, params=immutable_mapping(entry), when=when))
    return tuple(operations)


def _operation_section(
    raw: Any,
    field: str,
    source: Path,
    presets: Mapping[str, tuple[RuntimeOperation, ...]],
) -> tuple[RuntimeOperation, ...]:
    if isinstance(raw, str):
        try:
            return presets[raw]
        except KeyError as exc:
            raise ConfigurationError(f"{source}: unknown {field} preset {raw!r}") from exc
    section = _mapping(raw, field, source)
    _unknown_keys(section, {"preset", "prepend", "append", "replace"}, field, source)
    preset_value = section.get("preset", "")
    if not isinstance(preset_value, str):
        raise ConfigurationError(f"{source}: '{field}.preset' must be text")
    preset_name = preset_value.strip()
    base = list(presets.get(preset_name, ()))
    if preset_name and preset_name not in presets:
        raise ConfigurationError(f"{source}: unknown {field} preset {preset_name!r}")
    prepend = _operations(section.get("prepend", []), f"{field}.prepend", source)
    append = _operations(section.get("append", []), f"{field}.append", source)
    replace = section.get("replace")
    if replace is not None:
        if preset_name or prepend or append:
            raise ConfigurationError(
                f"{source}: {field}.replace cannot be combined with preset/prepend/append"
            )
        return _operations(replace, f"{field}.replace", source)
    return (*prepend, *base, *append)


def _resource_config_dir() -> Path:
    resource = files("dsle.config")
    with as_file(resource) as path:
        return Path(path)


class ConfigRepository:
    """Loads all task definitions from one explicit configuration directory."""

    def __init__(self, root: str | Path | None = None):
        self.root = (
            Path(root).expanduser().resolve() if root is not None else _resource_config_dir()
        )
        self.defaults = self._load_defaults()
        self._bosses, self._aliases = self._load_all_bosses()

    def _load_defaults(self) -> EnvironmentDefaults:
        source = self.root / "defaults.yaml"
        raw = _read_yaml(source)
        _unknown_keys(
            raw,
            {
                "schema_version",
                "environment",
                "observation",
                "reward",
                "limits",
                "readiness",
                "setup_presets",
                "cleanup_presets",
            },
            "document",
            source,
        )
        if _integer(raw, "schema_version", "document", source) != 1:
            raise ConfigurationError(f"{source}: schema_version must be 1")
        environment = _mapping(raw.get("environment", {}), "environment", source)
        observation = _mapping(raw.get("observation", {}), "observation", source)
        _unknown_keys(
            environment,
            {"action_ms", "auto_lock_on", "lock_on_interval"},
            "environment",
            source,
        )
        _unknown_keys(
            observation,
            {"height", "width", "monitor_index", "template_threshold"},
            "observation",
            source,
        )
        reward = _reward(_mapping(raw.get("reward", {}), "reward", source), "reward", source)
        limits = _limits(_mapping(raw.get("limits", {}), "limits", source), "limits", source)
        readiness = _readiness(
            _mapping(raw.get("readiness", {}), "readiness", source),
            "readiness",
            source,
        )
        setup_raw = _mapping(raw.get("setup_presets", {}), "setup_presets", source)
        cleanup_raw = _mapping(raw.get("cleanup_presets", {}), "cleanup_presets", source)
        setup_presets = {
            _text({"name": name}, "name", "setup_presets", source): _operations(
                value, f"setup_presets.{name}", source
            )
            for name, value in setup_raw.items()
        }
        cleanup_presets = {
            _text({"name": name}, "name", "cleanup_presets", source): _operations(
                value, f"cleanup_presets.{name}", source
            )
            for name, value in cleanup_raw.items()
        }
        defaults = EnvironmentDefaults(
            frame_height=_integer(
                observation, "height", "observation", source, default=600, minimum=1
            ),
            frame_width=_integer(
                observation, "width", "observation", source, default=800, minimum=1
            ),
            action_ms=_integer(
                environment, "action_ms", "environment", source, default=250, minimum=0
            ),
            auto_lock_on=_bool(environment, "auto_lock_on", "environment", source, default=True),
            lock_on_interval=_integer(
                environment,
                "lock_on_interval",
                "environment",
                source,
                default=1,
                minimum=1,
            ),
            monitor_index=_integer(
                observation,
                "monitor_index",
                "observation",
                source,
                default=0,
                minimum=0,
            ),
            template_threshold=_number(
                observation,
                "template_threshold",
                "observation",
                source,
                default=0.9,
                minimum=0.0,
                maximum=1.0,
                strict_minimum=True,
            ),
            reward=reward,
            limits=limits,
            readiness=readiness,
            setup_presets=MappingProxyType(setup_presets),
            cleanup_presets=MappingProxyType(cleanup_presets),
        )
        return defaults

    def _load_all_bosses(self) -> tuple[dict[str, BossConfig], dict[str, str]]:
        paths = sorted((self.root / "bosses").glob("*.yaml"))
        if not paths:
            raise ConfigurationError(f"No boss YAML files found under {self.root / 'bosses'}")
        bosses: dict[str, BossConfig] = {}
        aliases: dict[str, str] = {}
        for path in paths:
            boss = self._load_boss(path)
            if boss.boss_id in bosses:
                raise ConfigurationError(f"Duplicate boss id {boss.boss_id!r}: {path}")
            bosses[boss.boss_id] = boss
            for alias in (boss.boss_id, boss.display_name, *boss.aliases):
                key = alias.strip().lower()
                if key in aliases and aliases[key] != boss.boss_id:
                    raise ConfigurationError(f"Alias {alias!r} refers to multiple bosses")
                aliases[key] = boss.boss_id
        return bosses, aliases

    def _load_boss(self, source: Path) -> BossConfig:
        raw = _read_yaml(source)
        _unknown_keys(
            raw,
            {
                "schema_version",
                "id",
                "display_name",
                "aliases",
                "availability",
                "save_states",
                "memory",
                "victory",
                "reward",
                "limits",
                "readiness",
                "setup",
                "cleanup",
                "metadata",
            },
            "document",
            source,
        )
        if _integer(raw, "schema_version", "document", source) != 1:
            raise ConfigurationError(f"{source}: schema_version must be 1")
        boss_id = _text(raw, "id", "document", source)
        if not _BOSS_ID.fullmatch(boss_id) or source.stem != boss_id:
            raise ConfigurationError(f"{source}: id must be non-empty and match its filename")
        aliases_raw = _sequence(raw.get("aliases", []), "aliases", source)
        if any(not isinstance(value, str) or not value.strip() for value in aliases_raw):
            raise ConfigurationError(f"{source}: aliases must contain non-empty text")
        aliases = tuple(value.strip() for value in aliases_raw)
        if len(set(alias.casefold() for alias in aliases)) != len(aliases):
            raise ConfigurationError(f"{source}: aliases must be unique")
        save_states = _mapping(raw.get("save_states", {}), "save_states", source)
        if "standard" not in save_states:
            raise ConfigurationError(f"{source}: save_states must include 'standard'")
        normalized_saves: dict[str, str] = {}
        for key, value in save_states.items():
            if not isinstance(key, str) or not key.strip() or key != key.strip():
                raise ConfigurationError(
                    f"{source}: save_states keys must be non-empty trimmed text"
                )
            normalized_saves[key] = _safe_save_name(value, f"save_states.{key}", source)
        save_states = normalized_saves

        memory_raw = _mapping(raw.get("memory", {}), "memory", source)
        _unknown_keys(
            memory_raw,
            {"hp_chains", "hp_mode", "defeated_flag"},
            "memory",
            source,
        )
        chains = tuple(
            tuple(
                _integer(
                    {"offset": offset},
                    "offset",
                    "memory.hp_chains[]",
                    source,
                    minimum=0,
                )
                for offset in _sequence(chain, "memory.hp_chains[]", source)
            )
            for chain in _sequence(memory_raw.get("hp_chains", []), "memory.hp_chains", source)
        )
        defeated = _mapping(memory_raw.get("defeated_flag", {}), "memory.defeated_flag", source)
        _unknown_keys(defeated, {"offsets", "bit"}, "memory.defeated_flag", source)
        hp_mode = _text(memory_raw, "hp_mode", "memory", source, default="first_valid")
        if hp_mode not in {"first_valid", "sum", "defeated_flag"}:
            raise ConfigurationError(
                f"{source}: memory.hp_mode must be first_valid, sum, or defeated_flag"
            )
        if hp_mode == "defeated_flag" and chains:
            raise ConfigurationError(
                f"{source}: defeated_flag HP mode requires memory.hp_chains to be empty"
            )
        if hp_mode != "defeated_flag" and (not chains or any(not chain for chain in chains)):
            raise ConfigurationError(f"{source}: at least one non-empty HP chain is required")
        memory = MemoryConfig(
            hp_chains=chains,
            hp_mode=hp_mode,
            defeated_offsets=tuple(
                _integer(
                    {"offset": offset},
                    "offset",
                    "memory.defeated_flag.offsets",
                    source,
                    minimum=0,
                )
                for offset in _sequence(
                    defeated.get("offsets", []),
                    "memory.defeated_flag.offsets",
                    source,
                )
            ),
            defeated_bit=_integer(
                defeated,
                "bit",
                "memory.defeated_flag",
                source,
                minimum=0,
                maximum=7,
            ),
        )
        if not memory.defeated_offsets:
            raise ConfigurationError(
                f"{source}: defeated flag requires offsets and a bit in [0, 7]"
            )

        victory_raw = _mapping(raw.get("victory", {}), "victory", source)
        _unknown_keys(
            victory_raw,
            {"hp_zero_is_victory", "templates", "template_threshold"},
            "victory",
            source,
        )
        templates_raw = _sequence(victory_raw.get("templates", []), "victory.templates", source)
        if any(not isinstance(value, str) or not value.strip() for value in templates_raw):
            raise ConfigurationError(f"{source}: victory.templates must contain non-empty text")
        victory = VictoryConfig(
            hp_zero_is_victory=_bool(
                victory_raw,
                "hp_zero_is_victory",
                "victory",
                source,
                default=False,
            ),
            templates=tuple(value.strip() for value in templates_raw),
            template_threshold=_number(
                victory_raw,
                "template_threshold",
                "victory",
                source,
                default=self.defaults.template_threshold,
                minimum=0.0,
                maximum=1.0,
                strict_minimum=True,
            ),
        )
        if not victory.templates:
            raise ConfigurationError(f"{source}: victory.templates cannot be empty")

        reward_raw = deepcopy(self.defaults.reward.__dict__)
        reward_raw.update(_mapping(raw.get("reward", {}), "reward", source))
        limit_raw = deepcopy(self.defaults.limits.__dict__)
        limit_raw.update(_mapping(raw.get("limits", {}), "limits", source))
        ready_raw = deepcopy(self.defaults.readiness.__dict__)
        ready_raw.update(_mapping(raw.get("readiness", {}), "readiness", source))

        metadata_raw = _mapping(raw.get("metadata", {}), "metadata", source)
        _unknown_keys(
            metadata_raw,
            {"tier", "tags", "description", "expected_hp"},
            "metadata",
            source,
        )
        tags_raw = _sequence(metadata_raw.get("tags", []), "metadata.tags", source)
        if any(not isinstance(value, str) or not value.strip() for value in tags_raw):
            raise ConfigurationError(f"{source}: metadata.tags must contain non-empty text")
        expected_hp = metadata_raw.get("expected_hp")
        if expected_hp is not None:
            expected_hp = _integer(
                metadata_raw,
                "expected_hp",
                "metadata",
                source,
                minimum=1,
            )
        metadata = BossMetadata(
            tier=_text(metadata_raw, "tier", "metadata", source, default="unknown"),
            tags=tuple(value.strip() for value in tags_raw),
            description=_text(metadata_raw, "description", "metadata", source),
            expected_hp=expected_hp,
        )

        availability = _text(raw, "availability", "document", source, default="supported")
        if availability not in {"supported", "experimental"}:
            raise ConfigurationError(f"{source}: availability must be supported or experimental")

        display_name = raw.get("display_name", boss_id.replace("_", " ").title())
        if not isinstance(display_name, str) or not display_name.strip():
            raise ConfigurationError(f"{source}: display_name must be non-empty text")

        return BossConfig(
            boss_id=boss_id,
            display_name=display_name.strip(),
            aliases=aliases,
            availability=availability,
            save_states=MappingProxyType(save_states),
            memory=memory,
            victory=victory,
            readiness=_readiness(ready_raw, "readiness", source),
            reward=_reward(reward_raw, "reward", source),
            limits=_limits(limit_raw, "limits", source),
            setup=_operation_section(
                raw.get("setup", "fog_gate"), "setup", source, self.defaults.setup_presets
            ),
            cleanup=_operation_section(
                raw.get("cleanup", "default"), "cleanup", source, self.defaults.cleanup_presets
            ),
            metadata=metadata,
            source=source,
        )

    def list_bosses(self, *, include_experimental: bool = False) -> tuple[str, ...]:
        return tuple(
            boss_id
            for boss_id, boss in sorted(self._bosses.items())
            if include_experimental or boss.availability == "supported"
        )

    def get(self, name: str) -> BossConfig:
        key = str(name).strip().lower()
        try:
            return self._bosses[self._aliases[key]]
        except KeyError as exc:
            available = ", ".join(self.list_bosses())
            close = get_close_matches(key, tuple(self._aliases), n=1, cutoff=0.55)
            hint = f" Did you mean {self._aliases[close[0]]!r}?" if close else ""
            raise KeyError(f"Unknown boss {name!r}.{hint} Available bosses: {available}") from exc


_DEFAULT_REPOSITORY: ConfigRepository | None = None


def _repository() -> ConfigRepository:
    global _DEFAULT_REPOSITORY
    if _DEFAULT_REPOSITORY is None:
        _DEFAULT_REPOSITORY = ConfigRepository()
    return _DEFAULT_REPOSITORY


def load_defaults() -> EnvironmentDefaults:
    return _repository().defaults


def load_boss(name: str) -> BossConfig:
    return _repository().get(name)


def list_bosses(*, include_experimental: bool = False) -> tuple[str, ...]:
    return _repository().list_bosses(include_experimental=include_experimental)
