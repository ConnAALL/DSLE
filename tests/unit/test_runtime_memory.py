from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from dsle.exceptions import RuntimeUnavailableError
from dsle.models import MemoryConfig
from dsle.runtime.memory import (
    DSR_1_04,
    DSRMemory,
    LocatedProcess,
    MemoryAccessError,
    ProcessLocator,
    ReadResult,
    module_base_from_maps,
)


def test_invalid_read_is_distinct_from_a_valid_zero() -> None:
    assert ReadResult.success(0).valid
    missing = ReadResult[int].invalid("pointer missing")
    assert not missing.valid and missing.value is None
    with pytest.raises(MemoryAccessError, match="pointer missing"):
        missing.require("boss HP")


def test_module_base_accounts_for_file_offsets() -> None:
    maps = "\n".join(
        [
            "140000000-140001000 r--p 00000000 08:01 1 /game/DarkSoulsRemastered.exe",
            "140101000-140201000 r-xp 00101000 08:01 1 /game/DarkSoulsRemastered.exe",
            "7f000000-7f001000 r--p 00000000 00:00 0 /other/library.so",
        ]
    )
    assert module_base_from_maps(maps, "darksoulsremastered.exe") == 0x140000000


def test_process_locator_requires_the_exact_wineprefix(tmp_path: Path) -> None:
    prefix = tmp_path / "prefix-one"
    other = tmp_path / "prefix-two"
    prefix.mkdir()
    other.mkdir()
    proc = tmp_path / "proc"
    for pid, selected in ((101, prefix), (202, other)):
        entry = proc / str(pid)
        entry.mkdir(parents=True)
        (entry / "cmdline").write_bytes(b"wine\0DarkSoulsRemastered.exe\0")
        (entry / "environ").write_bytes(f"WINEPREFIX={selected}\0".encode())
        (entry / "maps").write_text(
            "140000000-140001000 r--p 00000000 08:01 1 /game/DarkSoulsRemastered.exe\n",
            encoding="utf-8",
        )

    located = ProcessLocator(prefix, proc_root=proc).locate()
    assert located == LocatedProcess(pid=101, module_base=0x140000000)


def attached_memory(path: Path, *, profile=DSR_1_04) -> DSRMemory:
    path.write_bytes(b"\0" * 4096)
    memory = DSRMemory(path.parent / "prefix", profile=profile)
    memory._fd = os.open(path, os.O_RDWR)  # test-only attachment to a regular byte store
    memory._process = LocatedProcess(pid=os.getpid(), module_base=0x100)
    return memory


def test_typed_reads_writes_and_pointer_resolution(tmp_path: Path) -> None:
    memory = attached_memory(tmp_path / "memory.bin")
    try:
        memory.write(32, "u64", 64)
        memory.write(68, "i32", 321)
        assert memory.resolve_chain(32, (0, 4), label="test chain") == 68
        assert memory.read(68, "i32") == 321
        assert memory.try_read(68, "i32", label="value") == ReadResult.success(321)
    finally:
        memory.close()


def test_boss_memory_reads_hp_and_defeated_flag_independently(tmp_path: Path) -> None:
    profile = replace(DSR_1_04, boss_rva=0x10, flags_rva=0x20)
    memory = attached_memory(tmp_path / "boss-memory.bin", profile=profile)
    config = MemoryConfig(
        hp_chains=((8,),),
        hp_mode="first_valid",
        defeated_offsets=(4,),
        defeated_bit=3,
    )
    try:
        memory.write(0x110, "u64", 0x200)
        memory.write(0x208, "i32", 900)
        memory.write(0x120, "u64", 0x300)
        memory.write(0x304, "u8", 1 << 3)
        result = memory.read_boss(config)
        assert result.hp == ReadResult.success(900)
        assert result.hp_values == (ReadResult.success(900),)
        assert result.defeated == ReadResult.success(True)

        memory.write(0x110, "u64", 0)
        missing = memory.read_boss(config)
        assert not missing.hp.valid
        assert missing.hp.value is None
        assert missing.defeated == ReadResult.success(True)
    finally:
        memory.close()


def test_boss_memory_preserves_every_hp_chain_while_keeping_configured_aggregate(
    tmp_path: Path,
) -> None:
    profile = replace(DSR_1_04, boss_rva=0x10, flags_rva=0x20)
    memory = attached_memory(tmp_path / "multi-boss-memory.bin", profile=profile)
    config = MemoryConfig(
        hp_chains=((8,), (12,)),
        hp_mode="first_valid",
        defeated_offsets=(4,),
        defeated_bit=3,
    )
    try:
        memory.write(0x110, "u64", 0x200)
        memory.write(0x208, "i32", 900)
        memory.write(0x20C, "i32", 700)
        memory.write(0x120, "u64", 0x300)
        memory.write(0x304, "u8", 0)

        first = memory.read_boss(config)
        assert first.hp == ReadResult.success(900)
        assert first.hp_values == (ReadResult.success(900), ReadResult.success(700))

        combined = memory.read_boss(replace(config, hp_mode="sum"))
        assert combined.hp == ReadResult.success(1600)
        assert combined.hp_values == first.hp_values

        memory.write_boss_hp(config, 2, 350)
        assert memory.read(0x20C, "i32") == 350
        with pytest.raises(MemoryAccessError):
            memory.write_boss_hp(replace(config, hp_chains=((8,), (5000,))), -1, 0)
        assert memory.read(0x208, "i32") == 900

        memory.write_boss_hp(config, -1, 0)
        assert memory.read(0x208, "i32") == 0
        assert memory.read(0x20C, "i32") == 0
        with pytest.raises(ValueError, match="only valid with hp=0"):
            memory.write_boss_hp(config, -1, 1)
        with pytest.raises(ValueError, match=r"\[1, 2\].*-1"):
            memory.write_boss_hp(config, 3, 1)
        with pytest.raises(ValueError, match=r"\[0, 10000000\]"):
            memory.write_boss_hp(config, 1, -1)
    finally:
        memory.close()


def test_flag_backed_boss_health_is_binary_and_does_not_require_a_boss_pointer(
    tmp_path: Path,
) -> None:
    profile = replace(DSR_1_04, boss_rva=0x5000, flags_rva=0x20)
    memory = attached_memory(tmp_path / "flag-backed-boss-memory.bin", profile=profile)
    config = MemoryConfig(
        hp_chains=(),
        hp_mode="defeated_flag",
        defeated_offsets=(0, 2),
        defeated_bit=5,
    )
    try:
        memory.write(0x120, "u64", 0x300)
        memory.write(0x300, "u64", 0x380)
        memory.write(0x382, "u8", 0)

        alive = memory.read_boss(config)
        assert alive.hp == ReadResult.success(1)
        assert alive.hp_values == (ReadResult.success(1),)
        assert alive.defeated == ReadResult.success(False)

        memory.write(0x382, "u8", 1 << 5)
        defeated = memory.read_boss(config)
        assert defeated.hp == ReadResult.success(0)
        assert defeated.hp_values == (ReadResult.success(0),)
        assert defeated.defeated == ReadResult.success(True)

        with pytest.raises(RuntimeUnavailableError, match="no writable HP address"):
            memory.write_boss_hp(config, 1, 0)
    finally:
        memory.close()


def test_player_hp_writer_resolves_live_structure_and_enforces_current_maximum(
    tmp_path: Path,
) -> None:
    profile = replace(
        DSR_1_04,
        basex_rva=0x10,
        player_struct_offset=8,
        player_hp_offset=12,
        player_hp_max_offset=16,
    )
    memory = attached_memory(tmp_path / "player-health-memory.bin", profile=profile)
    try:
        memory.write(0x110, "u64", 0x200)
        memory.write(0x208, "u64", 0x300)
        memory.write(0x30C, "i32", 500)
        memory.write(0x310, "i32", 750)

        memory.write_player_hp(600)
        assert memory.read(0x30C, "i32") == 600
        with pytest.raises(ValueError, match=r"\[0, 750\]"):
            memory.write_player_hp(751)
        with pytest.raises(ValueError, match="integer"):
            memory.write_player_hp(True)
    finally:
        memory.close()


def test_teleport_uses_warp_latch_and_preserves_omitted_coordinates(tmp_path: Path) -> None:
    profile = replace(
        DSR_1_04,
        basex_rva=0x10,
        player_x_offsets=(8,),
        player_y_offsets=(12,),
        player_z_offsets=(16,),
        player_angle_offsets=(20,),
        player_warp_latch_offsets=(24,),
        player_warp_x_offsets=(28,),
        player_warp_z_offsets=(32,),
        player_warp_y_offsets=(36,),
        player_warp_angle_offsets=(40,),
    )
    memory = attached_memory(tmp_path / "teleport-memory.bin", profile=profile)
    try:
        memory.write(0x110, "u64", 0x200)
        memory.write(0x208, "f32", 1.0)
        memory.write(0x20C, "f32", 2.0)
        memory.write(0x210, "f32", 3.0)
        memory.write(0x214, "f32", 0.75)
        player = memory.read_player()
        assert player.x == ReadResult.success(1.0)
        assert player.y == ReadResult.success(2.0)
        assert player.z == ReadResult.success(3.0)
        memory.teleport(x=11.5, z=-7.25)
        assert memory.read(0x21C, "f32") == pytest.approx(11.5)
        assert memory.read(0x220, "f32") == pytest.approx(-7.25)
        assert memory.read(0x224, "f32") == pytest.approx(2.0)
        assert memory.read(0x228, "f32") == pytest.approx(0.75)
        assert memory.read(0x218, "i32") == 1
        # The game consumes the warp request; the unit-test byte store does not.
        assert memory.read(0x208, "f32") == pytest.approx(1.0)
        assert memory.read(0x20C, "f32") == pytest.approx(2.0)
        assert memory.read(0x210, "f32") == pytest.approx(3.0)
    finally:
        memory.close()
