from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_viewer_is_an_optional_install_and_public_command() -> None:
    project = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'viewer = [\n  "Pillow>=10,<13",\n]' in project
    assert 'dsle-viewer = "dsle.cli.viewer:main"' in project


def test_viewer_launcher_is_executable_and_documents_passive_sampling() -> None:
    launcher = PROJECT_ROOT / "scripts" / "view-instances.sh"
    assert launcher.stat().st_mode & 0o111
    result = subprocess.run(
        [str(launcher), "--help"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert "CCTV-style grid" in result.stdout
    assert "--period SECONDS" in result.stdout
    assert "--dev" in result.stdout
    payload = launcher.read_text(encoding="utf-8")
    assert "python3-tk" in payload
    assert "never focuses a game window or sends input" in payload


def test_readme_documents_thirty_feed_viewer_without_core_dependency() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "## Multi-instance CCTV viewer" in readme
    assert "./scripts/view-instances.sh --period 1.0" in readme
    assert "python -m pip install '.[viewer]'" in readme
    assert "keyboard, mouse, focus, reset, or environment-control commands" in readme
