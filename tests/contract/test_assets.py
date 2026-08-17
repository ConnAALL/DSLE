from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pytest

from dsle.config import ConfigRepository
from dsle.runtime.templates import TemplateCatalog
from dsle.runtime.version import REQUIRED_GAME_DIRECTORIES, verify_game_executable

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def game_dir() -> Path:
    configured = os.environ.get("DSLE_GAME_DIR")
    if configured:
        return Path(configured).expanduser()
    for name in ("Dark.Souls.Remastered.v1.04", "game"):
        candidate = PROJECT_ROOT / name
        if candidate.is_dir():
            return candidate
    return PROJECT_ROOT / "Dark.Souls.Remastered.v1.04"


def require_available_game() -> Path:
    root = game_dir()
    if root.is_dir():
        return root
    if os.environ.get("DSLE_GAME_DIR"):
        pytest.fail(f"DSLE_GAME_DIR does not point to a directory: {root}")
    pytest.skip("No local commercial game checkout; set DSLE_GAME_DIR to validate one")


def test_legal_user_supplied_game_is_present_and_plausible() -> None:
    root = require_available_game()
    executable = root / "DarkSoulsRemastered.exe"
    assert root.is_dir(), f"Set DSLE_GAME_DIR to the legally obtained game directory: {root}"
    assert executable.is_file(), f"Missing game executable: {executable}"
    assert executable.stat().st_size > 40_000_000
    assert executable.read_bytes()[:2] == b"MZ"
    assert verify_game_executable(executable).verified
    for directory in REQUIRED_GAME_DIRECTORIES:
        assert (root / directory).is_dir(), f"Game installation is missing {directory}/"


def test_every_configured_scenario_has_one_canonical_flat_save() -> None:
    repository = ConfigRepository()
    scenario_dir = PROJECT_ROOT / "assets" / "scenarios"
    expected = {
        boss.save_state(difficulty)
        for boss_id in repository.list_bosses(include_experimental=False)
        for boss in (repository.get(boss_id),)
        for difficulty in ("standard", "boosted")
    }
    for filename in expected:
        path = scenario_dir / filename
        assert path.is_file(), f"Missing scenario save: {path}"
        assert path.stat().st_size == 4_326_608
        with path.open("rb") as stream:
            assert stream.read(4) == b"BND4", f"Scenario save has an invalid header: {path}"


def test_binary_asset_checksum_inventory_is_complete_and_valid() -> None:
    manifest = PROJECT_ROOT / "assets" / "checksums.sha256"
    records: dict[str, str] = {}
    for line in manifest.read_text(encoding="ascii").splitlines():
        if not line or line.startswith("#"):
            continue
        digest, separator, relative_name = line.partition("  ")
        assert separator and re.fullmatch(r"[0-9a-f]{64}", digest), line
        assert relative_name not in records, f"Duplicate checksum entry: {relative_name}"
        records[relative_name] = digest

    expected = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for root in (PROJECT_ROOT / "assets" / "scenarios", PROJECT_ROOT / "assets" / "templates")
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".sl2", ".png"}
    }
    assert set(records) == expected

    for relative_name, expected_digest in records.items():
        asset = PROJECT_ROOT / relative_name
        digest = hashlib.sha256()
        with asset.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        assert digest.hexdigest() == expected_digest, f"Corrupt binary asset: {relative_name}"


def test_standard_and_boosted_saves_are_distinct() -> None:
    repository = ConfigRepository()
    scenario_dir = PROJECT_ROOT / "assets" / "scenarios"
    for boss_id in repository.list_bosses(include_experimental=False):
        boss = repository.get(boss_id)
        standard = scenario_dir / boss.save_state("standard")
        boosted = scenario_dir / boss.save_state("boosted")
        assert standard.read_bytes() != boosted.read_bytes(), (
            f"{boss_id} standard and boosted saves unexpectedly contain identical data"
        )


def test_every_declared_template_resolves_from_the_canonical_catalog() -> None:
    repository = ConfigRepository()
    catalog = TemplateCatalog(PROJECT_ROOT / "assets")
    names = {
        template
        for boss_id in repository.list_bosses()
        for template in repository.get(boss_id).victory.templates
    }
    names.add(repository.defaults.readiness.template)
    for boss_id in repository.list_bosses():
        boss = repository.get(boss_id)
        if boss.readiness.template:
            names.add(boss.readiness.template)
        for operation in (*boss.setup, *boss.cleanup):
            template = operation.params.get("template")
            if isinstance(template, str):
                names.add(template)
    for name in names:
        assert name is not None
        assert catalog.resolve(name).is_file(), f"Missing declared template: {name}"


def test_local_game_folders_are_excluded_from_version_and_build_contexts() -> None:
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert "Dark.Souls.Remastered.v1.04/" in gitignore
    assert "game/" in gitignore
    assert next(line for line in dockerignore if line and not line.startswith("#")) == "**"
    assert "**/DarkSoulsRemastered.exe" in dockerignore
