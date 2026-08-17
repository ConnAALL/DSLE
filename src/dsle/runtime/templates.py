"""OpenCV template loading and matching for runtime orchestration."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from dsle.exceptions import AssetError, RuntimeUnavailableError

ColorOrder = Literal["rgb", "bgr"]


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeUnavailableError(
            "Template matching needs the optional runtime dependencies. "
            "Install DSLE with `pip install -e '.[runtime]'`."
        ) from exc
    return cv2


@dataclass(frozen=True)
class MatchResult:
    """The strongest normalized template match in a source image."""

    score: float
    top_left: tuple[int, int]
    size: tuple[int, int]

    @property
    def bottom_right(self) -> tuple[int, int]:
        x, y = self.top_left
        width, height = self.size
        return x + width, y + height


def to_grayscale(image: np.ndarray, *, color_order: ColorOrder = "rgb") -> np.ndarray:
    """Convert an RGB/BGR/RGBA/BGRA image to deterministic uint8 grayscale."""

    array = np.asarray(image)
    if array.ndim == 2:
        return np.clip(array, 0, 255).astype(np.uint8, copy=False)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError(f"Expected an HxW or HxWx3/4 image, got {array.shape}")
    if color_order not in {"rgb", "bgr"}:
        raise ValueError(f"Unsupported color order: {color_order!r}")
    channels = array[..., :3].astype(np.uint16, copy=False)
    if color_order == "bgr":
        channels = channels[..., ::-1]
    gray = (channels[..., 0] * 77 + channels[..., 1] * 150 + channels[..., 2] * 29) >> 8
    return gray.astype(np.uint8, copy=False)


def _roi_view(
    source: np.ndarray, roi: tuple[int, int, int, int] | None
) -> tuple[np.ndarray, tuple[int, int]]:
    if roi is None:
        return source, (0, 0)
    x, y, width, height = (int(value) for value in roi)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid ROI {roi}; expected non-negative x/y and positive width/height")
    if x + width > source.shape[1] or y + height > source.shape[0]:
        raise ValueError(f"ROI {roi} exceeds source bounds {source.shape[1]}x{source.shape[0]}")
    return source[y : y + height, x : x + width], (x, y)


def best_match(
    source: np.ndarray,
    template: np.ndarray,
    *,
    roi: tuple[int, int, int, int] | None = None,
) -> MatchResult:
    """Return the strongest ``TM_CCORR_NORMED`` match."""

    source_gray = to_grayscale(source)
    template_gray = to_grayscale(template)
    view, (offset_x, offset_y) = _roi_view(source_gray, roi)
    if template_gray.shape[0] > view.shape[0] or template_gray.shape[1] > view.shape[1]:
        raise ValueError(
            f"Template {template_gray.shape[1]}x{template_gray.shape[0]} is larger than "
            f"source region {view.shape[1]}x{view.shape[0]}"
        )
    cv2 = _cv2()
    scores = cv2.matchTemplate(view, template_gray, cv2.TM_CCORR_NORMED)
    _min_score, max_score, _min_location, max_location = cv2.minMaxLoc(scores)
    return MatchResult(
        score=float(max_score),
        top_left=(int(max_location[0] + offset_x), int(max_location[1] + offset_y)),
        size=(int(template_gray.shape[1]), int(template_gray.shape[0])),
    )


def matches(
    source: np.ndarray,
    template: np.ndarray,
    *,
    threshold: float = 0.9,
    roi: tuple[int, int, int, int] | None = None,
) -> tuple[bool, MatchResult]:
    """Return whether the best normalized match meets ``threshold``."""

    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be finite and in (0, 1]")
    result = best_match(source, template, roi=roi)
    return result.score >= threshold, result


class TemplateCatalog:
    """Resolve templates from an explicit asset bundle and cache decoded images."""

    _SUBDIRECTORIES = (
        "introScreen",
        "duringGame",
        "pauseReturnMenu",
        "bosses",
    )

    def __init__(self, asset_dir: str | Path):
        self.asset_dir = Path(asset_dir).expanduser().resolve(strict=False)
        self._cache: dict[Path, np.ndarray] = {}
        self._lock = threading.RLock()

    @property
    def roots(self) -> tuple[Path, ...]:
        """Return supported template roots in precedence order."""

        candidates = (
            self.asset_dir / "templates",
            self.asset_dir / "menu_navigation" / "templates",
            self.asset_dir,
        )
        return tuple(dict.fromkeys(candidates))

    @staticmethod
    def _filename(name: str) -> str:
        value = str(name).strip()
        if not value:
            raise ValueError("Template name cannot be empty")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Template name must stay inside the asset bundle: {name!r}")
        return value if value.lower().endswith(".png") else f"{value}.png"

    def candidate_paths(self, name: str) -> tuple[Path, ...]:
        """Return deterministic candidate locations for a logical template name."""

        filename = self._filename(name)
        relative = Path(filename)
        stem = relative.stem
        candidates: list[Path] = []
        for root in self.roots:
            candidates.append(root / relative)
            for subdirectory in self._SUBDIRECTORIES:
                candidates.append(root / subdirectory / relative.name)
            for suffix in ("_post", "_victory"):
                if stem.endswith(suffix):
                    boss = stem[: -len(suffix)]
                    candidates.append(root / "bosses" / boss / relative.name)
        return tuple(dict.fromkeys(candidates))

    def resolve(self, name: str) -> Path:
        """Resolve a named PNG without following an escape from the asset bundle."""

        candidates = self.candidate_paths(name)
        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(self.asset_dir)
            except (OSError, ValueError):
                continue
            if resolved.is_file():
                return resolved
        checked = ", ".join(str(path) for path in candidates)
        raise AssetError(f"Template {name!r} was not found. Checked: {checked}")

    def load(self, name: str) -> np.ndarray:
        """Load one template as grayscale, retaining an immutable cached copy."""

        path = self.resolve(name)
        with self._lock:
            cached = self._cache.get(path)
            if cached is not None:
                return cached
            cv2 = _cv2()
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise AssetError(f"OpenCV could not decode template: {path}")
            decoded = np.ascontiguousarray(image, dtype=np.uint8)
            decoded.setflags(write=False)
            self._cache[path] = decoded
            return decoded

    def match(
        self,
        frame_rgb: np.ndarray,
        name: str,
        *,
        threshold: float = 0.9,
        roi: tuple[int, int, int, int] | None = None,
    ) -> tuple[bool, MatchResult]:
        """Match one catalog template against an RGB frame."""

        return matches(frame_rgb, self.load(name), threshold=threshold, roi=roi)

    def match_any(
        self,
        frame_rgb: np.ndarray,
        names: tuple[str, ...] | list[str],
        *,
        threshold: float,
    ) -> tuple[str | None, MatchResult | None]:
        """Return the first matching logical name and its result."""

        gray = to_grayscale(frame_rgb)
        for name in names:
            matched, result = matches(gray, self.load(name), threshold=threshold)
            if matched:
                return name, result
        return None, None
