"""Step 5: project settings with explicit "unknown" values.

Why this exists
---------------
Some facts (image scale in nm/pixel, exact acquisition times) are not known yet.
Instead of inventing "standard" numbers, the settings file marks them as
"unknown". Later steps work in pixels and chronological rank until the
supervisor's real values are entered in config/project_config.ini.

Three levels of certainty
-------------------------
* unknown   : nothing is assumed; conversions are refused.
* estimated : derived from what the supervisor told us in an email (for example
              "each image is one micrometre") but not checked against raw data.
              Outputs built on it are labelled with the status.
* confirmed : checked against raw data or stated explicitly as a measured value.

Safety rules enforced here
--------------------------
* An unknown scale cannot be used to convert pixels to nm (it raises an error).
* A scale or time marked "estimated" or "confirmed" must name its source.
* The run label "final" is refused unless both scale and time are confirmed.
* Nothing in this module reads AFM height from JPG brightness.
"""

from __future__ import annotations

import argparse
import configparser
import math
import sys
from dataclasses import dataclass
from pathlib import Path

from afm_pipeline.inventory import PROJECT_ROOT, InventoryError
from afm_pipeline.order import ORDER_STATUS

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "project_config.ini"
_STATUSES = {"unknown", "estimated", "confirmed"}
_LABELS = {"preliminary", "final"}


class ConfigError(InventoryError):
    """The settings file is missing, malformed, or contradicts itself."""


@dataclass(frozen=True)
class ScaleConfig:
    status: str
    nm_per_pixel: float | None
    source: str
    frame_size_nm: float | None = None  # estimated: whole JPG frame = this many nm

    @property
    def known(self) -> bool:
        return self.status == "confirmed"

    @property
    def usable(self) -> bool:
        return self.status in ("estimated", "confirmed")


@dataclass(frozen=True)
class TimeConfig:
    status: str
    source: str
    frame_interval_s: float | None = None  # time one scan takes
    total_duration_s: float | None = None  # length of the whole imaging session

    @property
    def known(self) -> bool:
        return self.status == "confirmed"

    @property
    def usable(self) -> bool:
        return (
            self.status in ("estimated", "confirmed")
            and self.frame_interval_s is not None
            and self.total_duration_s is not None
        )


@dataclass(frozen=True)
class SegmentationConfig:
    """Settings for the region mask (Step 6). Intensity-based, NOT height.

    No region is thrown away because it is small. Regions are listed from
    min_region_area_px upward and labelled small / medium / large for filtering.
    """

    smoothing_sigma_px: float = 1.0
    threshold_k: float = 6.0  # "strong" regions: this many noise-sigmas above background
    weak_threshold_k: float = 3.5  # "faint" regions: dimmer pieces that never reach the strong level
    min_region_area_px: int = 10
    small_area_px: int = 80  # regions below this area are labelled "small"
    large_area_px: int = 320  # regions at or above this area are labelled "large"


@dataclass(frozen=True)
class ProjectConfig:
    expected_image_count: int
    run_label: str
    scale: ScaleConfig
    time: TimeConfig
    segmentation: SegmentationConfig = SegmentationConfig()


def _status(cp: configparser.ConfigParser, section: str) -> str:
    value = cp.get(section, "status", fallback="").strip().lower()
    if value not in _STATUSES:
        raise ConfigError(
            f"[{section}] status must be 'unknown', 'estimated' or 'confirmed', got '{value}'."
        )
    return value


def _optional(cp: configparser.ConfigParser, section: str, key: str) -> str:
    return cp.get(section, key, fallback="").strip()


def _positive_float(raw: str, section: str, key: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"[{section}] {key} is not a number: '{raw}'.") from exc
    if not math.isfinite(value) or value <= 0:
        raise ConfigError(f"[{section}] {key} must be a positive number, got {raw}.")
    return value


def _parse_scale(cp: configparser.ConfigParser) -> ScaleConfig:
    status = _status(cp, "scale")
    raw = _optional(cp, "scale", "nm_per_pixel")
    frame = _optional(cp, "scale", "frame_size_nm")
    source = _optional(cp, "scale", "source")
    if status == "unknown":
        if raw or frame:
            raise ConfigError(
                "[scale] has a value but status is 'unknown'.\n"
                "Set status = estimated or confirmed and add a source, or remove the value."
            )
        return ScaleConfig("unknown", None, "")
    if not source:
        raise ConfigError(f"[scale] is '{status}' but has no source (who gave this value?).")
    if status == "estimated":
        if raw:
            raise ConfigError(
                "[scale] an estimated scale uses frame_size_nm, not nm_per_pixel.\n"
                "Use status = confirmed if nm_per_pixel was measured."
            )
        if not frame:
            raise ConfigError("[scale] status is 'estimated' but frame_size_nm is missing.")
        return ScaleConfig("estimated", None, source, _positive_float(frame, "scale", "frame_size_nm"))
    if not raw:
        raise ConfigError("[scale] status is 'confirmed' but nm_per_pixel is missing.")
    return ScaleConfig("confirmed", _positive_float(raw, "scale", "nm_per_pixel"), source)


def _parse_time(cp: configparser.ConfigParser) -> TimeConfig:
    status = _status(cp, "time")
    source = _optional(cp, "time", "source")
    interval = _optional(cp, "time", "frame_interval_s")
    total = _optional(cp, "time", "total_duration_s")
    if status == "unknown":
        if interval or total:
            raise ConfigError(
                "[time] has a value but status is 'unknown'.\n"
                "Set status = estimated or confirmed and add a source, or remove the value."
            )
        return TimeConfig("unknown", "")
    if not source:
        raise ConfigError(f"[time] is '{status}' but has no source (which files / who said so?).")
    if status == "estimated" and not (interval and total):
        raise ConfigError("[time] status is 'estimated' but frame_interval_s or total_duration_s is missing.")
    interval_v = _positive_float(interval, "time", "frame_interval_s") if interval else None
    total_v = _positive_float(total, "time", "total_duration_s") if total else None
    return TimeConfig(status, source, interval_v, total_v)


def _parse_segmentation(cp: configparser.ConfigParser) -> SegmentationConfig:
    d = SegmentationConfig()
    if not cp.has_section("segmentation"):
        return d
    try:
        sigma = cp.getfloat("segmentation", "smoothing_sigma_px", fallback=d.smoothing_sigma_px)
        k = cp.getfloat("segmentation", "threshold_k", fallback=d.threshold_k)
        kw = cp.getfloat("segmentation", "weak_threshold_k", fallback=d.weak_threshold_k)
        min_area = cp.getint("segmentation", "min_region_area_px", fallback=d.min_region_area_px)
        small = cp.getint("segmentation", "small_area_px", fallback=d.small_area_px)
        large = cp.getint("segmentation", "large_area_px", fallback=d.large_area_px)
    except ValueError as exc:
        raise ConfigError(f"[segmentation] has a value that is not a valid number: {exc}") from exc
    if not (math.isfinite(sigma) and 0 <= sigma <= 10):
        raise ConfigError("[segmentation] smoothing_sigma_px must be between 0 and 10.")
    if not (math.isfinite(k) and k > 0):
        raise ConfigError("[segmentation] threshold_k must be a positive number.")
    if not (math.isfinite(kw) and 0 < kw <= k):
        raise ConfigError("[segmentation] weak_threshold_k must be positive and not larger than threshold_k.")
    if min_area < 1:
        raise ConfigError("[segmentation] min_region_area_px must be at least 1.")
    if not (min_area <= small < large):
        raise ConfigError(
            "[segmentation] areas must satisfy min_region_area_px <= small_area_px < large_area_px."
        )
    return SegmentationConfig(sigma, k, kw, min_area, small, large)


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Settings file not found: {path}")
    cp = configparser.ConfigParser(interpolation=None)
    try:
        cp.read_string(path.read_text(encoding="utf-8"))
    except configparser.Error as exc:
        raise ConfigError(f"Could not parse {path}: {exc}") from exc

    for section in ("project", "run", "scale", "time"):
        if not cp.has_section(section):
            raise ConfigError(f"Missing [{section}] section in {path}")

    try:
        count = cp.getint("project", "expected_image_count")
    except (ValueError, configparser.Error) as exc:
        raise ConfigError(f"[project] expected_image_count must be a whole number: {exc}") from exc
    if count < 1:
        raise ConfigError("[project] expected_image_count must be at least 1.")

    label = cp.get("run", "label", fallback="").strip().lower()
    if label not in _LABELS:
        raise ConfigError(f"[run] label must be 'preliminary' or 'final', got '{label}'.")

    scale = _parse_scale(cp)
    time = _parse_time(cp)
    if label == "final" and not (scale.known and time.known):
        raise ConfigError(
            "[run] label 'final' is not allowed while scale or time is still 'unknown'.\n"
            "Keep label = preliminary until the supervisor's values are entered."
        )
    return ProjectConfig(count, label, scale, time, _parse_segmentation(cp))


def pixels_to_nm(value_px: float, config: ProjectConfig) -> float:
    """Convert a length in pixels to nm. Refuses to guess if the scale is unknown."""
    if not config.scale.known or config.scale.nm_per_pixel is None:
        raise ConfigError(
            "Cannot convert pixels to nm: the image scale is still 'unknown'.\n"
            "Enter the supervisor's value in config/project_config.ini first."
        )
    return value_px * config.scale.nm_per_pixel


def nm_per_pixel_for_image(config: ProjectConfig, width_px: int, height_px: int) -> float:
    """Scale (nm per pixel) for one image.

    confirmed: the measured nm_per_pixel.
    estimated: the whole JPG frame is assumed to be frame_size_nm wide, so the scale
               is frame_size_nm divided by the mean of width and height in pixels.
    unknown  : refuses to guess.
    """
    scale = config.scale
    if scale.status == "confirmed" and scale.nm_per_pixel is not None:
        return scale.nm_per_pixel
    if scale.status == "estimated" and scale.frame_size_nm is not None:
        if width_px < 1 or height_px < 1:
            raise ConfigError("Image size must be positive to estimate the scale.")
        return scale.frame_size_nm / ((width_px + height_px) / 2)
    raise ConfigError(
        "Cannot convert pixels to nm: the image scale is still 'unknown'.\n"
        "Enter the supervisor's value in config/project_config.ini first."
    )


def time_bounds_s(rank: int, n_images: int, time_cfg: TimeConfig) -> tuple[float, float] | tuple[None, None]:
    """Earliest and latest possible scan START time (seconds) of image number `rank`.

    Images are kept in chronological order, each scan takes frame_interval_s, and the
    whole session lasts total_duration_s. Discarded scans can only add delay, so:
      earliest = (rank - 1) * interval            (no earlier frame was discarded)
      latest   = total - (n - rank + 1) * interval (all later frames still have to fit)
    This is a range, not a measured time. Returns (None, None) if time is not usable.
    """
    if not time_cfg.usable:
        return None, None
    interval, total = time_cfg.frame_interval_s, time_cfg.total_duration_s
    if not (1 <= rank <= n_images):
        raise ConfigError(f"rank {rank} is outside 1..{n_images}.")
    if n_images * interval > total:
        raise ConfigError(
            f"{n_images} images x {interval} s = {n_images * interval} s does not fit into "
            f"total_duration_s = {total} s. Check the [time] values."
        )
    return (rank - 1) * interval, total - (n_images - rank + 1) * interval


def describe(config: ProjectConfig) -> list[str]:
    lines = [
        f"Run label        : {config.run_label}",
        f"Expected images  : {config.expected_image_count}",
    ]
    if config.scale.known:
        lines.append(
            f"Scale            : CONFIRMED {config.scale.nm_per_pixel} nm/pixel "
            f"(source: {config.scale.source})"
        )
    elif config.scale.usable:
        lines.append(
            f"Scale            : ESTIMATED, whole image = {config.scale.frame_size_nm:g} nm "
            f"(about {config.scale.frame_size_nm / 277:.2f} nm/pixel for a 277 px image); "
            "not verified against raw data"
        )
    else:
        lines.append("Scale            : UNKNOWN -> all lengths stay in pixels")
    if config.time.known:
        lines.append(f"Acquisition time : CONFIRMED (source: {config.time.source})")
    elif config.time.usable:
        lines.append(
            f"Acquisition time : ESTIMATED range only ({config.time.frame_interval_s:g} s per scan, "
            f"{config.time.total_duration_s:g} s session); exact times unknown"
        )
    else:
        lines.append("Acquisition time : UNKNOWN -> only chronological_rank is available")
    lines.append(f"Image order      : {ORDER_STATUS}")
    seg = config.segmentation
    lines.append(
        f"Mask settings    : smoothing {seg.smoothing_sigma_px} px, strong {seg.threshold_k} x noise, "
        f"faint {seg.weak_threshold_k} x noise (intensity-based, preliminary)"
    )
    lines.append(
        f"Region listing   : every region from {seg.min_region_area_px} px upward is kept "
        f"(small < {seg.small_area_px}, large >= {seg.large_area_px} px)"
    )
    lines.append("AFM height (z)   : NOT AVAILABLE from JPG files; needs the original AFM data")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show the current project settings and what is still unknown.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except InventoryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("\n".join(describe(config)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
