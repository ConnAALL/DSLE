from __future__ import annotations

import importlib.util
import platform
import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("module", ["gymnasium", "numpy", "platformdirs", "rich", "yaml"])
def test_core_dependency_is_importable(module: str) -> None:
    assert importlib.util.find_spec(module) is not None


@pytest.mark.parametrize("module", ["cv2", "mss", "Xlib"])
def test_live_runtime_dependency_is_importable(module: str) -> None:
    assert importlib.util.find_spec(module) is not None


def test_host_platform_and_container_client() -> None:
    assert platform.system() == "Linux"
    assert shutil.which("docker"), "Docker CLI is required for the supported NVIDIA runtime"


def test_scope_extra_keeps_its_numpy_two_compatible_cma_floor() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"cma>=3.4"' in pyproject
