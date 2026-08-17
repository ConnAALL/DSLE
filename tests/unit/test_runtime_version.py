from __future__ import annotations

from pathlib import Path

import pytest

from dsle.exceptions import RuntimeUnavailableError
from dsle.runtime.version import (
    ALLOW_UNVERIFIED_ENV,
    SUPPORTED_EXECUTABLES,
    sha256_file,
    verify_game_executable,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
GAME_EXECUTABLE = WORKSPACE_ROOT / "Dark.Souls.Remastered.v1.04" / "DarkSoulsRemastered.exe"


def test_supplied_v104_executable_matches_the_memory_profile_allowlist() -> None:
    if not GAME_EXECUTABLE.is_file():
        pytest.skip("Commercial game checkout is not present")
    build = verify_game_executable(GAME_EXECUTABLE)
    assert build.verified
    assert build.sha256 in SUPPORTED_EXECUTABLES
    assert build.size == 50_286_344


def test_unknown_executable_is_rejected_before_memory_access(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "DarkSoulsRemastered.exe"
    executable.write_bytes(b"MZ" + b"unknown-layout")
    monkeypatch.delenv(ALLOW_UNVERIFIED_ENV, raising=False)
    with pytest.raises(RuntimeUnavailableError, match=r"Unsupported.*fingerprint"):
        verify_game_executable(executable)


def test_unverified_override_is_explicit_and_reported(tmp_path: Path) -> None:
    executable = tmp_path / "DarkSoulsRemastered.exe"
    executable.write_bytes(b"MZ" + b"deliberate-test-build")
    build = verify_game_executable(executable, allow_unverified=True)
    assert not build.verified
    assert build.name == "unverified override"
    assert build.sha256 == sha256_file(executable)


@pytest.mark.parametrize("value", ["false", 0, 1, [], object()])
def test_unverified_override_rejects_boolean_coercion(tmp_path: Path, value: object) -> None:
    executable = tmp_path / "DarkSoulsRemastered.exe"
    executable.write_bytes(b"MZ" + b"unknown")
    with pytest.raises(ValueError, match="boolean or None"):
        verify_game_executable(executable, allow_unverified=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5, "1024"])
def test_hash_chunk_size_must_be_a_positive_integer(tmp_path: Path, chunk_size: object) -> None:
    executable = tmp_path / "payload.bin"
    executable.write_bytes(b"payload")
    with pytest.raises(ValueError, match="positive integer"):
        sha256_file(executable, chunk_size=chunk_size)  # type: ignore[arg-type]


def test_invalid_override_value_is_rejected(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "DarkSoulsRemastered.exe"
    executable.write_bytes(b"MZ" + b"unknown")
    monkeypatch.setenv(ALLOW_UNVERIFIED_ENV, "perhaps")
    with pytest.raises(RuntimeUnavailableError, match="boolean"):
        verify_game_executable(executable)


def test_non_pe_and_wrong_filename_are_rejected(tmp_path: Path) -> None:
    wrong_name = tmp_path / "game.exe"
    wrong_name.write_bytes(b"MZpayload")
    with pytest.raises(RuntimeUnavailableError, match="Expected"):
        verify_game_executable(wrong_name)

    not_pe = tmp_path / "DarkSoulsRemastered.exe"
    not_pe.write_bytes(b"not a PE")
    with pytest.raises(RuntimeUnavailableError, match="Windows PE"):
        verify_game_executable(not_pe)
