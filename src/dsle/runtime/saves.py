"""Resolve and atomically swap Dark Souls: Remastered save states."""

from __future__ import annotations

import os
import re
import shutil
import stat
import tempfile
from pathlib import Path

from dsle.exceptions import AssetError, RuntimeUnavailableError

ACTIVE_SAVE_NAME = "DRAKS0005.sl2"
SAVE_SUFFIX = ".sl2"
SAVE_DIR_ENV = "DSLE_SAVE_DIR"
_SAFE_SAVE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_NUMERIC_ACCOUNT = re.compile(r"^[0-9]+$")


def _available_scenarios(directory: Path) -> tuple[Path, ...]:
    """Return save entries whose resolved files remain inside ``directory``."""

    if not directory.is_dir():
        return ()
    available: list[Path] = []
    for candidate in sorted(directory.glob(f"*{SAVE_SUFFIX}")):
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(directory)
        except (OSError, ValueError):
            continue
        if resolved.is_file():
            available.append(candidate)
    return tuple(available)


def normalize_save_name(name: str) -> str:
    """Return a safe save-state filename, adding ``.sl2`` when omitted."""

    value = str(name).strip()
    if not value or not _SAFE_SAVE_NAME.fullmatch(value):
        raise ValueError(
            "Save-state names may contain only letters, numbers, dots, underscores, and hyphens"
        )
    if not value.lower().endswith(SAVE_SUFFIX):
        value = f"{value}{SAVE_SUFFIX}"
    return value


def resolve_scenario_directory(asset_dir: str | Path) -> Path:
    """Find the save-state collection in an asset bundle."""

    root = Path(asset_dir).expanduser().resolve(strict=False)
    candidates = (root / "scenarios", root / "saves", root / "dsr_save_files", root)
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved.is_dir() and _available_scenarios(resolved):
            return resolved
    checked = ", ".join(str(path) for path in candidates)
    raise AssetError(f"No .sl2 save states were found. Checked: {checked}")


def _candidate_save_roots(wineprefix: Path) -> tuple[Path, ...]:
    users = wineprefix / "drive_c" / "users"
    if not users.is_dir():
        return ()
    return tuple(
        user / "Documents" / "NBGI" / "DARK SOULS REMASTERED"
        for user in sorted(users.iterdir())
        if user.is_dir()
    )


def discover_save_directory(wineprefix: str | Path) -> Path:
    """Discover the numeric account directory created beneath a Wine prefix."""

    configured = os.environ.get(SAVE_DIR_ENV)
    if configured:
        selected = Path(configured).expanduser().resolve(strict=False)
        if not selected.is_dir():
            raise RuntimeUnavailableError(
                f"{SAVE_DIR_ENV} points to a directory that does not exist: {selected}"
            )
        return selected

    prefix = Path(wineprefix).expanduser().resolve(strict=False)
    matches: list[Path] = []
    roots = _candidate_save_roots(prefix)
    for root in roots:
        try:
            resolved_root = root.resolve(strict=True)
            resolved_root.relative_to(prefix)
        except (OSError, ValueError):
            continue
        for path in sorted(resolved_root.iterdir()):
            if not _NUMERIC_ACCOUNT.fullmatch(path.name):
                continue
            try:
                resolved_path = path.resolve(strict=True)
                resolved_path.relative_to(prefix)
            except (OSError, ValueError):
                continue
            if resolved_path.is_dir():
                matches.append(resolved_path)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        choices = ", ".join(str(path) for path in matches)
        raise RuntimeUnavailableError(
            f"Multiple DSR account save directories were found: {choices}. "
            "Set save_dir in the instance configuration."
        )
    checked = ", ".join(str(path) for path in roots) or f"{prefix}/drive_c/users/*"
    raise RuntimeUnavailableError(
        "Could not discover a DSR account save directory. Start the game once or set "
        f"save_dir in the instance configuration. Checked: {checked}"
    )


def atomic_copy(source: Path, destination: Path) -> None:
    """Copy a file into place without exposing a partial active save."""

    if not destination.parent.is_dir():
        raise FileNotFoundError(destination.parent)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        source_descriptor = os.open(source, flags)
    except OSError as exc:
        raise FileNotFoundError(source) from exc
    temporary: Path | None = None
    try:
        with (
            os.fdopen(source_descriptor, "rb") as source_handle,
            tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as destination_handle,
        ):
            source_metadata = os.fstat(source_handle.fileno())
            if not stat.S_ISREG(source_metadata.st_mode):
                raise FileNotFoundError(source)
            temporary = Path(destination_handle.name)
            shutil.copyfileobj(source_handle, destination_handle)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
        if temporary is None:  # Defensive invariant; NamedTemporaryFile always has a name.
            raise RuntimeError("Could not allocate a temporary save file")
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


class SaveManager:
    """Swap curated scenario saves into one live instance's active slot."""

    def __init__(self, scenario_dir: str | Path, active_save_dir: str | Path):
        self.scenario_dir = Path(scenario_dir).expanduser().resolve(strict=False)
        self.active_save_dir = Path(active_save_dir).expanduser().resolve(strict=False)

    @property
    def active_save(self) -> Path:
        return self.active_save_dir / ACTIVE_SAVE_NAME

    def available(self) -> tuple[str, ...]:
        """Return available scenario filenames in deterministic order."""

        return tuple(path.name for path in _available_scenarios(self.scenario_dir))

    def resolve(self, name: str) -> Path:
        """Resolve one scenario without allowing traversal outside the asset directory."""

        filename = normalize_save_name(name)
        source = self.scenario_dir / filename
        try:
            resolved = source.resolve(strict=True)
            resolved.relative_to(self.scenario_dir)
        except (OSError, ValueError) as exc:
            available = ", ".join(self.available()) or "(none)"
            raise AssetError(
                f"Save state {filename!r} does not exist in {self.scenario_dir}. "
                f"Available: {available}"
            ) from exc
        if not resolved.is_file():
            raise AssetError(f"Save state is not a regular file: {resolved}")
        return resolved

    def load(self, name: str) -> Path:
        """Atomically replace ``DRAKS0005.sl2`` with the selected scenario."""

        if not self.active_save_dir.is_dir():
            raise RuntimeUnavailableError(
                f"Live DSR save directory does not exist: {self.active_save_dir}"
            )
        source = self.resolve(name)
        atomic_copy(source, self.active_save)
        return self.active_save
