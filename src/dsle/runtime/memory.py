"""Validity-safe process-memory access for Dark Souls Remastered v1.04."""

from __future__ import annotations

import math
import os
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar, cast

from dsle.exceptions import RuntimeCommunicationError, RuntimeUnavailableError
from dsle.models import MemoryConfig

T = TypeVar("T")


class MemoryAccessError(RuntimeCommunicationError):
    """A concrete process-memory read, pointer resolution, or write failed."""


@dataclass(frozen=True)
class ReadResult(Generic[T]):
    """A memory value whose invalid state cannot be confused with numeric zero."""

    value: T | None
    valid: bool
    error: str | None = None

    @classmethod
    def success(cls, value: T) -> ReadResult[T]:
        return cls(value=value, valid=True)

    @classmethod
    def invalid(cls, error: str) -> ReadResult[T]:
        return cls(value=None, valid=False, error=error)

    def require(self, label: str) -> T:
        """Return the value or raise with its original invalidity reason."""

        if not self.valid or self.value is None:
            raise MemoryAccessError(f"{label} is unavailable: {self.error or 'invalid read'}")
        return self.value


@dataclass(frozen=True)
class MemoryProfile:
    """Static offsets tied to one exact game executable release."""

    version: str
    executable: str
    basex_rva: int
    baseb_rva: int
    boss_rva: int
    flags_rva: int
    lockon_rva: int
    menu_rva: int
    player_struct_offset: int
    player_hp_offset: int
    player_hp_max_offset: int
    death_count_offset: int
    lockon_offsets: tuple[int, ...]
    title_menu_offsets: tuple[int, ...]
    no_dead_offsets: tuple[int, ...]
    no_dead_bit: int
    no_damage_offsets: tuple[int, ...]
    no_damage_bit: int
    player_x_offsets: tuple[int, ...]
    player_y_offsets: tuple[int, ...]
    player_z_offsets: tuple[int, ...]
    player_angle_offsets: tuple[int, ...]
    player_warp_latch_offsets: tuple[int, ...]
    player_warp_x_offsets: tuple[int, ...]
    player_warp_y_offsets: tuple[int, ...]
    player_warp_z_offsets: tuple[int, ...]
    player_warp_angle_offsets: tuple[int, ...]


DSR_1_04 = MemoryProfile(
    version="1.04",
    executable="DarkSoulsRemastered.exe",
    basex_rva=0x1C77E50,
    baseb_rva=0x1C8A530,
    boss_rva=0x01C79AD0,
    flags_rva=0x1C7C5F0,
    lockon_rva=0x01C696C8,
    menu_rva=0x1C88D98,
    player_struct_offset=0x68,
    player_hp_offset=0x3E8,
    player_hp_max_offset=0x3EC,
    death_count_offset=0x98,
    lockon_offsets=(0xCE0,),
    title_menu_offsets=(0x24C,),
    no_dead_offsets=(0x68, 0x524),
    no_dead_bit=5,
    no_damage_offsets=(0x68, 0x524),
    no_damage_bit=6,
    player_x_offsets=(0x68, 0x18, 0x28, 0x50, 0x20, 0x120),
    player_z_offsets=(0x68, 0x18, 0x28, 0x50, 0x20, 0x124),
    player_y_offsets=(0x68, 0x18, 0x28, 0x50, 0x20, 0x128),
    player_angle_offsets=(0x68, 0x68, 0x28, 0x4),
    player_warp_latch_offsets=(0x68, 0x68, 0x108),
    player_warp_x_offsets=(0x68, 0x68, 0x110),
    player_warp_z_offsets=(0x68, 0x68, 0x114),
    player_warp_y_offsets=(0x68, 0x68, 0x118),
    player_warp_angle_offsets=(0x68, 0x68, 0x124),
)


@dataclass(frozen=True)
class LocatedProcess:
    """A matching game process and its relocated executable base."""

    pid: int
    module_base: int


def _normalize_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _environment(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in path.read_bytes().split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        values[key.decode("utf-8", errors="replace")] = value.decode("utf-8", errors="replace")
    return values


def module_base_from_maps(maps_text: str, executable: str) -> int:
    """Compute a module image base from matching ``/proc/<pid>/maps`` lines."""

    needle = executable.casefold()
    bases: list[int] = []
    for line in maps_text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or needle not in fields[5].casefold():
            continue
        try:
            start = int(fields[0].split("-", 1)[0], 16)
            file_offset = int(fields[2], 16)
        except (IndexError, ValueError):
            continue
        bases.append(start - file_offset)
    if not bases:
        raise ValueError(f"{executable!r} is not mapped")
    return min(bases)


class ProcessLocator:
    """Locate the one DSR process belonging to an exact Wine prefix."""

    def __init__(
        self,
        wineprefix: str | Path,
        *,
        executable: str = DSR_1_04.executable,
        proc_root: str | Path = "/proc",
    ):
        self.wineprefix = _normalize_path(wineprefix)
        self.executable = executable
        self.proc_root = Path(proc_root)

    def locate(self) -> LocatedProcess:
        """Return an unambiguous process match or a diagnostic error."""

        matches: list[LocatedProcess] = []
        inaccessible_candidates: list[int] = []
        needle = self.executable.casefold()
        try:
            entries = tuple(self.proc_root.iterdir())
        except OSError as exc:
            raise RuntimeUnavailableError(f"Could not inspect {self.proc_root}: {exc}") from exc
        for entry in entries:
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                command = (entry / "cmdline").read_bytes().decode("utf-8", errors="replace")
                if needle not in command.casefold():
                    continue
                environment = _environment(entry / "environ")
                prefix = environment.get("WINEPREFIX")
                if prefix is None or _normalize_path(prefix) != self.wineprefix:
                    continue
                maps_text = (entry / "maps").read_text(encoding="utf-8", errors="replace")
                base = module_base_from_maps(maps_text, self.executable)
                matches.append(LocatedProcess(pid=pid, module_base=base))
            except (FileNotFoundError, ProcessLookupError):
                continue  # The process exited while /proc was being enumerated.
            except PermissionError:
                inaccessible_candidates.append(pid)
            except (OSError, ValueError):
                continue
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            pids = ", ".join(str(match.pid) for match in matches)
            raise RuntimeUnavailableError(
                f"Multiple {self.executable} processes use Wine prefix {self.wineprefix}: {pids}"
            )
        permission_hint = (
            f" Matching candidates without permission: {inaccessible_candidates}."
            if inaccessible_candidates
            else ""
        )
        raise RuntimeUnavailableError(
            f"No running {self.executable} process uses Wine prefix {self.wineprefix}."
            f"{permission_hint}"
        )


@dataclass(frozen=True)
class PlayerMemory:
    """Independent validity results for player state fields."""

    hp: ReadResult[int]
    hp_max: ReadResult[int]
    death_count: ReadResult[int]
    locked_on: ReadResult[bool]
    x: ReadResult[float] = field(
        default_factory=lambda: ReadResult.invalid("player x coordinate was not read")
    )
    y: ReadResult[float] = field(
        default_factory=lambda: ReadResult.invalid("player y coordinate was not read")
    )
    z: ReadResult[float] = field(
        default_factory=lambda: ReadResult.invalid("player z coordinate was not read")
    )


@dataclass(frozen=True)
class BossMemory:
    """Independent validity results for boss HP and defeated flag."""

    hp: ReadResult[int]
    defeated: ReadResult[bool]
    hp_values: tuple[ReadResult[int], ...] = ()


_FORMATS: dict[str, struct.Struct] = {
    "u64": struct.Struct("<Q"),
    "u8": struct.Struct("<B"),
    "i32": struct.Struct("<i"),
    "f32": struct.Struct("<f"),
}


class DSRMemory:
    """Attached process memory using offsets explicitly scoped to DSR v1.04."""

    def __init__(
        self,
        wineprefix: str | Path,
        *,
        profile: MemoryProfile = DSR_1_04,
        locator: ProcessLocator | None = None,
    ):
        self.wineprefix = _normalize_path(wineprefix)
        self.profile = profile
        self.locator = locator or ProcessLocator(self.wineprefix, executable=profile.executable)
        self._fd: int | None = None
        self._process: LocatedProcess | None = None
        self._lock = threading.RLock()

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    @property
    def attached(self) -> bool:
        return self._fd is not None and self._process is not None

    def attach(self) -> None:
        """Locate the configured instance and open its memory read/write handle."""

        located = self.locator.locate()
        path = Path("/proc") / str(located.pid) / "mem"
        try:
            descriptor = os.open(path, os.O_RDWR | getattr(os, "O_CLOEXEC", 0))
        except OSError as exc:
            raise RuntimeUnavailableError(
                f"Could not open {path} for DSR {self.profile.version}: {exc}. "
                "The container needs ptrace permission for its own Wine process."
            ) from exc
        with self._lock:
            old = self._fd
            self._fd = descriptor
            self._process = located
            if old is not None:
                os.close(old)

    def close(self) -> None:
        """Close the process handle; repeated calls are harmless."""

        with self._lock:
            descriptor = self._fd
            self._fd = None
            self._process = None
            if descriptor is not None:
                os.close(descriptor)

    def __enter__(self) -> DSRMemory:
        self.attach()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def _descriptor(self) -> int:
        if self._fd is None:
            raise RuntimeCommunicationError("DSR process memory is not attached")
        return self._fd

    def static_address(self, rva: int) -> int:
        if self._process is None:
            raise RuntimeCommunicationError("DSR process memory is not attached")
        return self._process.module_base + int(rva)

    @staticmethod
    def _validate_address(address: int) -> None:
        if isinstance(address, bool) or not isinstance(address, int) or address <= 0:
            raise MemoryAccessError(f"Invalid process-memory address: {address!r}")

    def read(self, address: int, kind: str) -> int | float:
        """Read one typed value, requiring an exact byte count."""

        self._validate_address(address)
        try:
            codec = _FORMATS[kind]
        except KeyError as exc:
            raise ValueError(f"Unsupported memory type {kind!r}") from exc
        with self._lock:
            try:
                data = os.pread(self._descriptor(), codec.size, address)
            except OSError as exc:
                raise MemoryAccessError(f"Read {kind} at 0x{address:X} failed: {exc}") from exc
        if len(data) != codec.size:
            raise MemoryAccessError(
                f"Short read for {kind} at 0x{address:X}: expected {codec.size}, got {len(data)}"
            )
        return cast(int | float, codec.unpack(data)[0])

    def try_read(self, address: int, kind: str, *, label: str) -> ReadResult[int | float]:
        """Read one value while preserving any expected invalidity as data."""

        try:
            return ReadResult.success(self.read(address, kind))
        except MemoryAccessError as exc:
            return ReadResult.invalid(f"{label}: {exc}")

    def write(self, address: int, kind: str, value: int | float) -> None:
        """Write one typed value, raising on partial or invalid writes."""

        self._validate_address(address)
        try:
            codec = _FORMATS[kind]
            data = codec.pack(value)
        except KeyError as exc:
            raise ValueError(f"Unsupported memory type {kind!r}") from exc
        except struct.error as exc:
            raise ValueError(f"Value {value!r} is invalid for memory type {kind}") from exc
        with self._lock:
            try:
                written = os.pwrite(self._descriptor(), data, address)
            except OSError as exc:
                raise MemoryAccessError(f"Write {kind} at 0x{address:X} failed: {exc}") from exc
        if written != len(data):
            raise MemoryAccessError(
                f"Short write for {kind} at 0x{address:X}: expected {len(data)}, got {written}"
            )

    def _try_static_pointer(self, rva: int, label: str) -> ReadResult[int]:
        result = self.try_read(self.static_address(rva), "u64", label=label)
        if not result.valid:
            return ReadResult.invalid(result.error or f"{label}: invalid read")
        pointer = int(result.value or 0)
        if pointer == 0:
            return ReadResult.invalid(f"{label}: null pointer")
        return ReadResult.success(pointer)

    def resolve_chain(self, root: int, offsets: tuple[int, ...], *, label: str) -> int:
        """Resolve a pointer chain and return its final value address."""

        self._validate_address(root)
        if not offsets:
            raise ValueError(f"{label}: pointer chain cannot be empty")
        current = root
        for index, offset in enumerate(offsets[:-1]):
            pointer_address = current + int(offset)
            pointer = int(self.read(pointer_address, "u64"))
            if pointer == 0:
                raise MemoryAccessError(
                    f"{label}: null pointer at chain index {index} (0x{pointer_address:X})"
                )
            current = pointer
        final = current + int(offsets[-1])
        self._validate_address(final)
        return final

    def try_chain(
        self,
        root: int,
        offsets: tuple[int, ...],
        kind: str,
        *,
        label: str,
    ) -> ReadResult[int | float]:
        """Resolve and read a chain, retaining the precise invalidity reason."""

        try:
            address = self.resolve_chain(root, offsets, label=label)
            return self.try_read(address, kind, label=label)
        except (MemoryAccessError, ValueError) as exc:
            return ReadResult.invalid(str(exc))

    @staticmethod
    def _bounded_int(
        result: ReadResult[int | float], label: str, *, minimum: int, maximum: int
    ) -> ReadResult[int]:
        if not result.valid or result.value is None:
            return ReadResult.invalid(result.error or f"{label}: invalid read")
        value = int(result.value)
        if not minimum <= value <= maximum:
            return ReadResult.invalid(
                f"{label}: implausible value {value} outside [{minimum}, {maximum}]"
            )
        return ReadResult.success(value)

    @staticmethod
    def _coordinate(result: ReadResult[int | float], label: str) -> ReadResult[float]:
        """Validate one finite, plausible world-space coordinate."""

        if not result.valid or result.value is None:
            return ReadResult.invalid(result.error or f"{label}: invalid read")
        value = float(result.value)
        if not math.isfinite(value) or abs(value) > 10_000_000:
            return ReadResult.invalid(f"{label}: implausible value {value!r}")
        return ReadResult.success(value)

    def read_player(self) -> PlayerMemory:
        """Read player health, location, death count, and lock-on independently."""

        basex = self._try_static_pointer(self.profile.basex_rva, "player root")
        if basex.valid:
            struct_pointer = self.try_read(
                basex.require("player root") + self.profile.player_struct_offset,
                "u64",
                label="player structure",
            )
        else:
            struct_pointer = ReadResult.invalid(basex.error or "player root is invalid")
        if struct_pointer.valid and int(struct_pointer.value or 0) != 0:
            player_struct = int(struct_pointer.value or 0)
            hp = self._bounded_int(
                self.try_read(
                    player_struct + self.profile.player_hp_offset, "i32", label="player HP"
                ),
                "player HP",
                minimum=0,
                maximum=10_000_000,
            )
            hp_max = self._bounded_int(
                self.try_read(
                    player_struct + self.profile.player_hp_max_offset,
                    "i32",
                    label="player maximum HP",
                ),
                "player maximum HP",
                minimum=1,
                maximum=10_000_000,
            )
        else:
            reason = struct_pointer.error or "player structure: null pointer"
            hp = ReadResult.invalid(reason)
            hp_max = ReadResult.invalid(reason)

        if basex.valid:
            player_root = basex.require("player root")
            x = self._coordinate(
                self.try_chain(
                    player_root,
                    self.profile.player_x_offsets,
                    "f32",
                    label="player x coordinate",
                ),
                "player x coordinate",
            )
            y = self._coordinate(
                self.try_chain(
                    player_root,
                    self.profile.player_y_offsets,
                    "f32",
                    label="player y coordinate",
                ),
                "player y coordinate",
            )
            z = self._coordinate(
                self.try_chain(
                    player_root,
                    self.profile.player_z_offsets,
                    "f32",
                    label="player z coordinate",
                ),
                "player z coordinate",
            )
        else:
            reason = basex.error or "player root is invalid"
            x = ReadResult.invalid(reason)
            y = ReadResult.invalid(reason)
            z = ReadResult.invalid(reason)

        baseb = self._try_static_pointer(self.profile.baseb_rva, "game-data root")
        if baseb.valid:
            death_count = self._bounded_int(
                self.try_read(
                    baseb.require("game-data root") + self.profile.death_count_offset,
                    "i32",
                    label="death count",
                ),
                "death count",
                minimum=0,
                maximum=10_000_000,
            )
        else:
            death_count = ReadResult.invalid(baseb.error or "game-data root is invalid")

        lock_root = self._try_static_pointer(self.profile.lockon_rva, "lock-on root")
        if lock_root.valid:
            lock_value = self.try_chain(
                lock_root.require("lock-on root"),
                self.profile.lockon_offsets,
                "u8",
                label="lock-on state",
            )
            locked_on = (
                ReadResult.success(int(lock_value.value or 0) == 1)
                if lock_value.valid
                else ReadResult.invalid(lock_value.error or "lock-on state is invalid")
            )
        else:
            locked_on = ReadResult.invalid(lock_root.error or "lock-on root is invalid")
        return PlayerMemory(
            hp=hp,
            hp_max=hp_max,
            death_count=death_count,
            locked_on=locked_on,
            x=x,
            y=y,
            z=z,
        )

    @staticmethod
    def _health_value(value: object, label: str, *, maximum: int = 10_000_000) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{label} must be an integer")
        if not 0 <= value <= maximum:
            raise ValueError(f"{label} must be in [0, {maximum}]")
        return value

    def write_player_hp(self, hp: int) -> None:
        """Write validated current player HP through the live player structure."""

        basex = self._try_static_pointer(self.profile.basex_rva, "player root")
        player_root = basex.require("player root")
        struct_pointer = self.try_read(
            player_root + self.profile.player_struct_offset,
            "u64",
            label="player structure",
        )
        player_struct = int(struct_pointer.require("player structure"))
        if player_struct == 0:
            raise MemoryAccessError("player structure: null pointer")
        maximum = self._bounded_int(
            self.try_read(
                player_struct + self.profile.player_hp_max_offset,
                "i32",
                label="player maximum HP",
            ),
            "player maximum HP",
            minimum=1,
            maximum=10_000_000,
        ).require("player maximum HP")
        selected = self._health_value(hp, "player HP", maximum=maximum)
        self.write(player_struct + self.profile.player_hp_offset, "i32", selected)

    def read_boss(self, config: MemoryConfig) -> BossMemory:
        """Read boss health and the defeated flag without false zeroes."""

        hp_reads: list[ReadResult[int]] = []
        if config.hp_chains:
            boss_root = self._try_static_pointer(self.profile.boss_rva, "boss root")
            if boss_root.valid:
                root = boss_root.require("boss root")
                for index, chain in enumerate(config.hp_chains):
                    hp_reads.append(
                        self._bounded_int(
                            self.try_chain(
                                root,
                                chain,
                                "i32",
                                label=f"boss HP chain {index}",
                            ),
                            f"boss HP chain {index}",
                            minimum=0,
                            maximum=10_000_000,
                        )
                    )
            else:
                error = boss_root.error or "boss root is invalid"
                hp_reads.extend(ReadResult.invalid(error) for _chain in config.hp_chains)

        valid_hp = [
            result.value for result in hp_reads if result.valid and result.value is not None
        ]
        if valid_hp:
            hp_value = valid_hp[0] if config.hp_mode == "first_valid" else sum(valid_hp)
            hp = ReadResult.success(int(hp_value))
        else:
            errors = "; ".join(result.error or "invalid chain" for result in hp_reads)
            hp = ReadResult.invalid(errors or "no boss HP chain produced a valid value")

        flags_root = self._try_static_pointer(self.profile.flags_rva, "event-flags root")
        if flags_root.valid:
            flag_value = self.try_chain(
                flags_root.require("event-flags root"),
                config.defeated_offsets,
                "u8",
                label="boss defeated flag",
            )
            defeated = (
                ReadResult.success(bool(int(flag_value.value or 0) & (1 << config.defeated_bit)))
                if flag_value.valid
                else ReadResult.invalid(flag_value.error or "boss defeated flag is invalid")
            )
        else:
            defeated = ReadResult.invalid(flags_root.error or "event-flags root is invalid")

        if config.hp_mode == "defeated_flag":
            if defeated.valid:
                hp = ReadResult.success(0 if defeated.value else 1)
            else:
                hp = ReadResult.invalid(
                    defeated.error or "boss defeated flag is invalid for binary health"
                )
            hp_reads = [hp]
        return BossMemory(hp=hp, defeated=defeated, hp_values=tuple(hp_reads))

    def write_boss_hp(self, config: MemoryConfig, boss_number: int, hp: int) -> None:
        """Write one boss HP chain, or every chain for the explicit ``(-1, 0)`` sentinel."""

        if isinstance(boss_number, bool) or not isinstance(boss_number, int):
            raise ValueError("boss_number must be an integer")
        if config.hp_mode == "defeated_flag":
            raise RuntimeUnavailableError(
                "Boss HP is derived from the defeated flag and has no writable HP address"
            )
        write_all = boss_number == -1
        if write_all and hp != 0:
            raise ValueError("boss_number=-1 is only valid with hp=0")
        if not write_all and not 1 <= boss_number <= len(config.hp_chains):
            raise ValueError(f"boss_number must be in [1, {len(config.hp_chains)}], or -1")
        selected = self._health_value(
            hp,
            "all boss HP" if write_all else f"boss{boss_number} HP",
        )
        selected_numbers = range(1, len(config.hp_chains) + 1) if write_all else (boss_number,)
        with self._lock:
            root = self._try_static_pointer(self.profile.boss_rva, "boss root").require("boss root")
            addresses: list[int] = []
            for number in selected_numbers:
                label = f"boss HP chain {number - 1}"
                address = self.resolve_chain(root, config.hp_chains[number - 1], label=label)
                # Validate every dynamic target before performing the first
                # multi-target write, avoiding partial writes on stale chains.
                self._bounded_int(
                    self.try_read(address, "i32", label=label),
                    label,
                    minimum=0,
                    maximum=10_000_000,
                ).require(label)
                addresses.append(address)
            for address in addresses:
                self.write(address, "i32", selected)

    def _set_chain_bit(self, offsets: tuple[int, ...], bit: int, enabled: bool) -> None:
        root = self._try_static_pointer(self.profile.basex_rva, "player root").require(
            "player root"
        )
        address = self.resolve_chain(root, offsets, label="player runtime flag")
        current = int(self.read(address, "u8"))
        updated = current | (1 << bit) if enabled else current & ~(1 << bit)
        self.write(address, "u8", updated)

    def set_flag(self, name: str, enabled: bool) -> None:
        """Set one supported player runtime flag."""

        flags = {
            "no_dead": (self.profile.no_dead_offsets, self.profile.no_dead_bit),
            "no_damage": (self.profile.no_damage_offsets, self.profile.no_damage_bit),
        }
        try:
            offsets, bit = flags[name]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported DSR runtime flag {name!r}; choose from {sorted(flags)}"
            ) from exc
        self._set_chain_bit(offsets, bit, bool(enabled))

    def teleport(
        self,
        *,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
    ) -> None:
        """Request an engine-managed warp while preserving omitted axes and facing."""

        if x is None and y is None and z is None:
            raise ValueError("teleport requires at least one of x, y, or z")
        with self._lock:
            root = self._try_static_pointer(self.profile.basex_rva, "player root").require(
                "player root"
            )
            current = {
                "x": self._coordinate(
                    self.try_chain(
                        root,
                        self.profile.player_x_offsets,
                        "f32",
                        label="player x coordinate",
                    ),
                    "player x coordinate",
                ).require("player x coordinate"),
                "y": self._coordinate(
                    self.try_chain(
                        root,
                        self.profile.player_y_offsets,
                        "f32",
                        label="player y coordinate",
                    ),
                    "player y coordinate",
                ).require("player y coordinate"),
                "z": self._coordinate(
                    self.try_chain(
                        root,
                        self.profile.player_z_offsets,
                        "f32",
                        label="player z coordinate",
                    ),
                    "player z coordinate",
                ).require("player z coordinate"),
            }
            angle = self._coordinate(
                self.try_chain(
                    root,
                    self.profile.player_angle_offsets,
                    "f32",
                    label="player facing angle",
                ),
                "player facing angle",
            ).require("player facing angle")
            destination = {
                "x": current["x"] if x is None else float(x),
                "y": current["y"] if y is None else float(y),
                "z": current["z"] if z is None else float(z),
            }
            warp_addresses = {
                "x": self.resolve_chain(
                    root, self.profile.player_warp_x_offsets, label="player warp x coordinate"
                ),
                "y": self.resolve_chain(
                    root, self.profile.player_warp_y_offsets, label="player warp y coordinate"
                ),
                "z": self.resolve_chain(
                    root, self.profile.player_warp_z_offsets, label="player warp z coordinate"
                ),
                "angle": self.resolve_chain(
                    root, self.profile.player_warp_angle_offsets, label="player warp angle"
                ),
                "latch": self.resolve_chain(
                    root, self.profile.player_warp_latch_offsets, label="player warp latch"
                ),
            }
            # DSR stores its native position triplet as X, Z, Y. Write the complete
            # request before setting the latch so the game applies it atomically.
            self.write(warp_addresses["x"], "f32", destination["x"])
            self.write(warp_addresses["z"], "f32", destination["z"])
            self.write(warp_addresses["y"], "f32", destination["y"])
            self.write(warp_addresses["angle"], "f32", angle)
            self.write(warp_addresses["latch"], "i32", 1)

    def return_to_title(self) -> None:
        """Request a return to the title menu through the v1.04 menu field."""

        root = self._try_static_pointer(self.profile.menu_rva, "menu root").require("menu root")
        address = self.resolve_chain(
            root, self.profile.title_menu_offsets, label="title-menu command"
        )
        self.write(address, "i32", 1)
