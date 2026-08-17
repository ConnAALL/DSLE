from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from dsle.exceptions import ConfigurationError
from dsle.runtime.output import finalize_output


def test_finalize_output_makes_nested_data_host_writable_without_following_links(
    tmp_path: Path,
) -> None:
    root = tmp_path / "output"
    results = root / "results" / "nested"
    results.mkdir(parents=True)
    (root / ".dsle-output").write_text("dsle-output\n", encoding="ascii")
    checkpoint = results / "checkpoint.bin"
    checkpoint.write_bytes(b"weights")
    checkpoint.chmod(0o400)

    victim = tmp_path / "outside.txt"
    victim.write_text("outside", encoding="utf-8")
    victim.chmod(0o400)
    (root / "results" / "outside-link").symlink_to(victim)

    hardlink = results / "outside-hardlink"
    os.link(victim, hardlink)
    original_victim_mode = stat.S_IMODE(victim.stat().st_mode)
    results.chmod(0o500)

    finalize_output(root, os.getuid(), os.getgid())

    assert checkpoint.stat().st_uid == os.getuid()
    assert stat.S_IMODE(checkpoint.stat().st_mode) & 0o600 == 0o600
    assert stat.S_IMODE(results.stat().st_mode) & 0o700 == 0o700
    assert (root / "results" / "outside-link").is_symlink()
    assert victim.read_text(encoding="utf-8") == "outside"
    assert stat.S_IMODE(victim.stat().st_mode) == original_victim_mode


def test_finalize_output_requires_the_exact_marker(tmp_path: Path) -> None:
    root = tmp_path / "output"
    root.mkdir()
    (root / ".dsle-output").write_text("not-dsle\n", encoding="ascii")
    with pytest.raises(ConfigurationError, match="Invalid DSLE output marker"):
        finalize_output(root, os.getuid(), os.getgid())
