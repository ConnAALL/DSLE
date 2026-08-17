from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from dsle.exceptions import AssetError, RuntimeCommunicationError, RuntimeUnavailableError
from dsle.runtime.capture import MSSCapture, bgra_to_rgb
from dsle.runtime.saves import (
    ACTIVE_SAVE_NAME,
    SaveManager,
    discover_save_directory,
    normalize_save_name,
    resolve_scenario_directory,
)
from dsle.runtime.templates import TemplateCatalog, best_match, matches, to_grayscale

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_bgra_capture_conversion_is_contiguous_rgb() -> None:
    bgra = np.asarray([[[1, 2, 3, 255], [4, 5, 6, 128]]], dtype=np.uint8)
    rgb = bgra_to_rgb(bgra)
    assert rgb.tolist() == [[[3, 2, 1], [6, 5, 4]]]
    assert rgb.flags.c_contiguous


def test_mss_capture_uses_the_requested_display_and_monitor() -> None:
    calls: list[str] = []

    class Session:
        monitors: ClassVar[list[dict[str, int]]] = [{"left": 0}, {"left": 10}]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def grab(self, monitor):
            assert monitor == self.monitors[1]
            return np.asarray([[[10, 20, 30, 255]]], dtype=np.uint8)

    def factory(*, display: str):
        calls.append(display)
        return Session()

    capture = MSSCapture(":91", monitor_index=1, factory=factory)
    assert capture.capture().tolist() == [[[30, 20, 10]]]
    assert calls == [":91"]


def test_mss_capture_rejects_a_missing_monitor() -> None:
    class Session:
        monitors: ClassVar[list[dict[str, object]]] = [{}]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    capture = MSSCapture(":92", monitor_index=1, factory=lambda **_kwargs: Session())
    with pytest.raises(RuntimeCommunicationError, match="monitor_index"):
        capture.capture()


def test_template_match_finds_exact_patch_and_respects_roi() -> None:
    template = np.asarray(
        [
            [0, 30, 80, 10],
            [20, 255, 40, 70],
            [90, 60, 5, 120],
        ],
        dtype=np.uint8,
    )
    source = np.zeros((20, 24), dtype=np.uint8)
    source[8:11, 12:16] = template

    result = best_match(source, template)
    assert result.top_left == (12, 8)
    assert result.bottom_right == (16, 11)
    assert result.score == pytest.approx(1.0, abs=1e-6)

    matched, roi_result = matches(source, template, threshold=0.99, roi=(10, 6, 10, 10))
    assert matched
    assert roi_result.top_left == (12, 8)


def test_grayscale_conversion_uses_rgb_order() -> None:
    rgb = np.asarray([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    assert to_grayscale(rgb).tolist() == [[76, 149, 28]]


def test_real_template_catalog_resolves_and_decodes_assets() -> None:
    catalog = TemplateCatalog(PROJECT_ROOT / "assets")
    path = catalog.resolve("traverseLight")
    assert path.name == "traverseLight.png"
    image = catalog.load("traverseLight")
    assert image.ndim == 2 and image.dtype == np.uint8 and image.size > 0
    assert image.flags.writeable is False
    assert catalog.load("traverseLight") is image


def test_asset_resolvers_reject_symlink_escapes(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    templates = assets / "templates"
    scenarios = assets / "scenarios"
    outside = tmp_path / "outside"
    templates.mkdir(parents=True)
    scenarios.mkdir()
    outside.mkdir()

    outside_template = outside / "secret.png"
    outside_template.write_bytes(b"not-an-asset")
    (templates / "secret.png").symlink_to(outside_template)
    with pytest.raises(AssetError, match="not found"):
        TemplateCatalog(assets).resolve("secret")

    outside_save = outside / "outside.sl2"
    outside_save.write_bytes(b"not-a-curated-save")
    (scenarios / "outside.sl2").symlink_to(outside_save)
    manager = SaveManager(scenarios, tmp_path / "active")
    assert manager.available() == ()
    with pytest.raises(AssetError, match="does not exist"):
        manager.resolve("outside")


def test_scenario_discovery_rejects_directory_symlink_escape(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    outside = tmp_path / "outside"
    assets.mkdir()
    outside.mkdir()
    (outside / "boss.sl2").write_bytes(b"save")
    (assets / "scenarios").symlink_to(outside, target_is_directory=True)

    with pytest.raises(AssetError, match=r"No \.sl2"):
        resolve_scenario_directory(assets)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("asylum_demon", "asylum_demon.sl2"),
        ("boss-1.sl2", "boss-1.sl2"),
        ("_beginning.sl2", "_beginning.sl2"),
    ],
)
def test_save_name_normalization(raw: str, expected: str) -> None:
    assert normalize_save_name(raw) == expected


@pytest.mark.parametrize("raw", ["", "../outside", "/absolute.sl2", "space name"])
def test_save_name_rejects_path_traversal(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_save_name(raw)


def test_save_manager_atomically_copies_without_changing_scenario(tmp_path: Path) -> None:
    scenarios = tmp_path / "scenarios"
    active = tmp_path / "active"
    scenarios.mkdir()
    active.mkdir()
    source = scenarios / "asylum_demon.sl2"
    source.write_bytes(b"curated-save")

    manager = SaveManager(scenarios, active)
    destination = manager.load("asylum_demon")
    assert destination == active / ACTIVE_SAVE_NAME
    assert destination.read_bytes() == b"curated-save"
    assert source.read_bytes() == b"curated-save"
    assert manager.available() == ("asylum_demon.sl2",)


def test_scenario_and_instance_save_directory_discovery(tmp_path: Path, monkeypatch) -> None:
    asset_root = tmp_path / "assets"
    scenarios = asset_root / "scenarios"
    scenarios.mkdir(parents=True)
    (scenarios / "one.sl2").write_bytes(b"save")
    assert resolve_scenario_directory(asset_root) == scenarios

    monkeypatch.delenv("DSLE_SAVE_DIR", raising=False)
    account = (
        tmp_path
        / "prefix"
        / "drive_c"
        / "users"
        / "runner"
        / "Documents"
        / "NBGI"
        / "DARK SOULS REMASTERED"
        / "123456"
    )
    account.mkdir(parents=True)
    assert discover_save_directory(tmp_path / "prefix") == account


def test_save_discovery_refuses_ambiguous_accounts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DSLE_SAVE_DIR", raising=False)
    root = (
        tmp_path
        / "prefix"
        / "drive_c"
        / "users"
        / "runner"
        / "Documents"
        / "NBGI"
        / "DARK SOULS REMASTERED"
    )
    (root / "111").mkdir(parents=True)
    (root / "222").mkdir()
    with pytest.raises(RuntimeUnavailableError, match="Multiple"):
        discover_save_directory(tmp_path / "prefix")
