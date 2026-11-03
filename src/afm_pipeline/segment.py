"""Step 6: find every bright region in each image and keep each one separately.

PRELIMINARY, pixel units, JPG intensity only.

What a "region" is
------------------
A region is a connected group of pixels that stands clearly above its own
row/background level. Every region gets its own number (region_id) in each image
and its own row in regions.csv. NOTHING is deleted for being small.

Two tiers
---------
* strong: pixels more than threshold_k noise-sigmas above background. These give
  tight outlines.
* faint : dimmer pieces that never reach the strong level but still stand out
  (above weak_threshold_k noise-sigmas). Faint regions never touch a strong one
  (a dim edge next to a strong region simply belongs to the weak halo and is not
  listed separately).

Size labels (small / medium / large) are only sorting and colour hints.

Linked groups
-------------
Two strong regions can be separate at the strong level yet still joined by a
dimmer stretch (above the faint level). Such regions share the same
linked_group_id. A gap between regions with the SAME group id may just be a dim
stretch of one molecule, while regions in DIFFERENT groups are separated by real
background. This is a hint for later analysis, not a verdict.

Nanometres and time
-------------------
If the settings file gives a usable scale, lengths and areas are also written in nm
(columns *_nm, area_nm2) together with the scale status ("estimated" until the
supervisor confirms it). If it gives a usable time estimate, every image gets an
earliest and a latest possible scan start time, never a single invented value.

What this module does NOT do
----------------------------
* It does not measure AFM height.
* It does not decide which regions are DNA, fragments, enzyme or debris.
  That needs the supervisor's criteria and labelled examples.
* It never modifies an original image; all results go under outputs/.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

from afm_pipeline.config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    ProjectConfig,
    SegmentationConfig,
    load_config,
    nm_per_pixel_for_image,
    time_bounds_s,
)
from afm_pipeline.inventory import (  # noqa: E402
    DEFAULT_DATA_DIR,
    PROJECT_ROOT,
    InventoryError,
    find_image_files,
)
from afm_pipeline.order import build_provisional_order  # noqa: E402

MIN_NOISE_FLOOR = 0.5  # intensity levels; avoids a zero threshold on perfectly flat images
NOTE = "intensity-based region in pixel units; not AFM height; not a classification"

# preview outline colours (RGB)
COLOR_LARGE = (0, 255, 255)  # cyan
COLOR_MEDIUM = (255, 255, 0)  # yellow
COLOR_SMALL = (255, 0, 255)  # magenta
COLOR_FAINT = (0, 255, 0)  # green

SUMMARY_COLUMNS = [
    "chronological_rank",
    "order_status",
    "filename",
    "read_error",
    "image_width_px",
    "image_height_px",
    "nm_per_pixel_used",
    "scale_status",
    "time_earliest_s",
    "time_latest_s",
    "time_status",
    "noise_sigma",
    "strong_threshold",
    "weak_threshold",
    "n_regions",
    "n_linked_groups",
    "n_strong",
    "n_faint",
    "n_large",
    "n_medium",
    "n_small",
    "largest_region_px",
    "mask_px",
    "mask_fraction_pct",
    "n_below_min_area",
    "run_label",
    "note",
]

REGION_COLUMNS = [
    "chronological_rank",
    "filename",
    "region_id",
    "tier",
    "size_class",
    "linked_group_id",
    "area_px",
    "centroid_x_px",
    "centroid_y_px",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "peak_contrast_sigma",
    "mean_contrast_sigma",
    "approx_major_axis_px",
    "approx_minor_axis_px",
    "touches_border",
    "nm_per_pixel_used",
    "scale_status",
    "area_nm2",
    "centroid_x_nm",
    "centroid_y_nm",
    "approx_major_axis_nm",
    "approx_minor_axis_nm",
    "time_earliest_s",
    "time_latest_s",
    "time_status",
    "run_label",
    "note",
]


# ----------------------------------------------------------------- image math
def intensity_from_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def gaussian_kernel(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian_smooth(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.asarray(image, dtype=np.float64).copy()
    kernel = gaussian_kernel(sigma)
    radius = len(kernel) // 2
    out = np.asarray(image, dtype=np.float64)
    for axis in (0, 1):
        n = out.shape[axis]
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="reflect" if radius < n else "edge")
        acc = np.zeros_like(out)
        for i, weight in enumerate(kernel):
            index = [slice(None), slice(None)]
            index[axis] = slice(i, i + n)
            acc += weight * padded[tuple(index)]
        out = acc
    return out


def remove_row_offsets(intensity: np.ndarray) -> np.ndarray:
    flat = intensity - np.median(intensity, axis=1, keepdims=True)
    return flat - np.median(flat)


def robust_noise(values: np.ndarray) -> float:
    centre = np.median(values)
    return 1.4826 * float(np.median(np.abs(values - centre)))


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 8-connected True regions. Returns (labels, count); background is 0."""
    height, width = mask.shape
    grid = mask.tolist()
    labels = [[0] * width for _ in range(height)]
    count = 0
    ys, xs = np.nonzero(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if labels[y0][x0]:
            continue
        count += 1
        labels[y0][x0] = count
        stack = [(y0, x0)]
        while stack:
            y, x = stack.pop()
            for yy in (y - 1, y, y + 1):
                if yy < 0 or yy >= height:
                    continue
                for xx in (x - 1, x, x + 1):
                    if 0 <= xx < width and grid[yy][xx] and not labels[yy][xx]:
                        labels[yy][xx] = count
                        stack.append((yy, xx))
    return np.array(labels, dtype=np.int32).reshape(height, width), count


def _group_pixels(labels: np.ndarray) -> dict[int, np.ndarray]:
    """Map each label id to the flat indices of its pixels (fast, sort-based)."""
    flat = labels.ravel()
    idx = np.flatnonzero(flat)
    if idx.size == 0:
        return {}
    lbl = flat[idx]
    order = np.argsort(lbl, kind="stable")
    idx, lbl = idx[order], lbl[order]
    cuts = np.flatnonzero(np.diff(lbl)) + 1
    return {int(l[0]): g for g, l in zip(np.split(idx, cuts), np.split(lbl, cuts))}


# ------------------------------------------------------------- region finding
@dataclass
class RegionResult:
    labels: np.ndarray  # int32; 0 = background; ids 1..N, largest area first
    regions: list[dict]
    noise_sigma: float
    strong_threshold: float
    weak_threshold: float
    n_below_min_area: int

    @property
    def mask(self) -> np.ndarray:
        return self.labels > 0


def _size_class(area: int, settings: SegmentationConfig) -> str:
    if area < settings.small_area_px:
        return "small"
    if area < settings.large_area_px:
        return "medium"
    return "large"


def _region_stats(idx: np.ndarray, smooth: np.ndarray, noise: float, tier: str,
                  settings: SegmentationConfig) -> dict:
    height, width = smooth.shape
    ys, xs = idx // width, idx % width
    values = smooth.ravel()[idx] / noise
    area = int(idx.size)
    if area >= 2:
        eig = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys]).astype(float)))
        major, minor = 4 * math.sqrt(max(eig[1], 0.0)), 4 * math.sqrt(max(eig[0], 0.0))
    else:
        major = minor = float("nan")
    return {
        "tier": tier,
        "size_class": _size_class(area, settings),
        "area_px": area,
        "centroid_x_px": round(float(xs.mean()), 2),
        "centroid_y_px": round(float(ys.mean()), 2),
        "bbox_x0": int(xs.min()),
        "bbox_y0": int(ys.min()),
        "bbox_x1": int(xs.max()),
        "bbox_y1": int(ys.max()),
        "peak_contrast_sigma": round(float(values.max()), 2),
        "mean_contrast_sigma": round(float(values.mean()), 2),
        "approx_major_axis_px": round(major, 1),
        "approx_minor_axis_px": round(minor, 1),
        "touches_border": bool(ys.min() == 0 or xs.min() == 0 or ys.max() == height - 1 or xs.max() == width - 1),
    }


def extract_regions(smooth: np.ndarray, noise: float, settings: SegmentationConfig) -> RegionResult:
    strong_thr = settings.threshold_k * noise
    weak_thr = settings.weak_threshold_k * noise
    strong = smooth > strong_thr
    weak = smooth > weak_thr

    lab_s, _ = label_components(strong)
    lab_w, n_w = label_components(weak)
    has_strong = np.zeros(n_w + 1, dtype=bool)
    has_strong[lab_w[strong]] = True

    lab_w_flat = lab_w.ravel()
    candidates = [("strong", idx, int(lab_w_flat[idx[0]])) for idx in _group_pixels(lab_s).values()]
    candidates += [("faint", idx, lid) for lid, idx in _group_pixels(lab_w).items() if not has_strong[lid]]

    listed = [c for c in candidates if c[1].size >= settings.min_region_area_px]
    n_below = len(candidates) - len(listed)
    listed.sort(key=lambda item: -item[1].size)

    labels = np.zeros(smooth.size, dtype=np.int32)
    regions = []
    group_numbers: dict[int, int] = {}
    for region_id, (tier, idx, weak_id) in enumerate(listed, start=1):
        labels[idx] = region_id
        stats = _region_stats(idx, smooth, noise, tier, settings)
        stats["region_id"] = region_id
        stats["linked_group_id"] = group_numbers.setdefault(weak_id, len(group_numbers) + 1)
        regions.append(stats)
    return RegionResult(labels.reshape(smooth.shape), regions, noise, strong_thr, weak_thr, n_below)


def segment_array(rgb: np.ndarray, settings: SegmentationConfig) -> RegionResult:
    intensity = intensity_from_rgb(rgb)
    flat = remove_row_offsets(intensity)
    smooth = gaussian_smooth(flat, settings.smoothing_sigma_px)
    noise = max(robust_noise(smooth), MIN_NOISE_FLOOR)
    return extract_regions(smooth, noise, settings)


def load_rgb(path: Path) -> np.ndarray | None:
    """RGB array copy for analysis; the file itself is only read."""
    try:
        with Image.open(path) as img:
            img.load()
            return np.asarray(img.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def outline_from_labels(labels: np.ndarray) -> np.ndarray:
    """Pixels of each region that touch a different label (including background)."""
    p = np.pad(labels, 1, mode="constant", constant_values=0)
    c = p[1:-1, 1:-1]
    differs = (c != p[:-2, 1:-1]) | (c != p[2:, 1:-1]) | (c != p[1:-1, :-2]) | (c != p[1:-1, 2:])
    return (c > 0) & differs


def _color_lookup(regions: list[dict]) -> np.ndarray:
    table = np.zeros((len(regions) + 1, 3), dtype=np.uint8)
    for r in regions:
        if r["tier"] == "faint":
            color = COLOR_FAINT
        else:
            color = {"large": COLOR_LARGE, "medium": COLOR_MEDIUM, "small": COLOR_SMALL}[r["size_class"]]
        table[r["region_id"]] = color
    return table


# --------------------------------------------------------------------- outputs
def _preview_sheet(items: list, out_path: Path, columns: int, run_label: str) -> None:
    n = len(items)
    rows = max(1, math.ceil(n / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.0, rows * 2.3 + 1.0), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, arr, result) in zip(axes.ravel(), items):
        if arr is None:
            ax.text(0.5, 0.5, "unreadable", ha="center", va="center", color="red", fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            shown = arr.copy()
            edge = outline_from_labels(result.labels)
            shown[edge] = _color_lookup(result.regions)[result.labels[edge]]
            ax.imshow(shown, interpolation="nearest")
        ax.set_title(name, fontsize=8, pad=2)
    fig.suptitle(
        f"{run_label.upper()} regions, PROVISIONAL chronological order. "
        "Outline colour: cyan = large, yellow = medium, magenta = small, green = faint (all kept).\n"
        "Intensity-based, NOT AFM height; regions are not yet classified as DNA / fragment / debris.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_segmentation(
    data_dir: Path | str,
    out_dir: Path | str,
    settings: SegmentationConfig,
    run_label: str = "preliminary",
    columns: int = 10,
    config: ProjectConfig | None = None,
) -> dict:
    out_dir = Path(out_dir).resolve()
    data_resolved = Path(data_dir).resolve()
    if out_dir == data_resolved or data_resolved in out_dir.parents:
        raise InventoryError(
            f"Refusing to write results inside the data folder ({data_resolved}).\n"
            "Original images must stay untouched; choose an output folder in outputs/."
        )
    if columns < 1:
        raise InventoryError("--columns must be at least 1.")

    images, _ = find_image_files(data_dir)
    by_name = {p.name: p for p in images}
    order = build_provisional_order(list(by_name))
    masks_dir, labels_dir = out_dir / "masks", out_dir / "labels"
    masks_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    n_images = len(order)
    scale_status = config.scale.status if config is not None and config.scale.usable else ""
    time_status = config.time.status if config is not None and config.time.usable else ""

    summary_rows, region_rows, preview_items = [], [], []
    for rank, name in zip(order["chronological_rank"], order["filename"]):
        rank = int(rank)
        row = {c: pd.NA for c in SUMMARY_COLUMNS}
        row.update(chronological_rank=rank, order_status="provisional", filename=name,
                   run_label=run_label, note=NOTE)
        t_early, t_late = (None, None)
        if time_status:
            t_early, t_late = time_bounds_s(rank, n_images, config.time)
            row.update(time_earliest_s=t_early, time_latest_s=t_late, time_status=time_status)
        arr = load_rgb(by_name[name])
        if arr is None:
            row["read_error"] = "image could not be read"
            preview_items.append((name, None, None))
            summary_rows.append(row)
            continue

        height_px, width_px = arr.shape[:2]
        row.update(image_width_px=width_px, image_height_px=height_px)
        nmpp = None
        if scale_status:
            nmpp = nm_per_pixel_for_image(config, width_px, height_px)
            row.update(nm_per_pixel_used=round(nmpp, 4), scale_status=scale_status)

        result = segment_array(arr, settings)
        stem = Path(name).stem
        Image.fromarray((result.mask * 255).astype(np.uint8)).save(masks_dir / f"{stem}_mask.png")
        Image.fromarray(result.labels.astype(np.uint16)).save(labels_dir / f"{stem}_labels.png")

        regs = result.regions
        mask_px = int(result.mask.sum())
        row.update(
            noise_sigma=round(result.noise_sigma, 4),
            strong_threshold=round(result.strong_threshold, 4),
            weak_threshold=round(result.weak_threshold, 4),
            n_regions=len(regs),
            n_linked_groups=len({r["linked_group_id"] for r in regs}),
            n_strong=sum(r["tier"] == "strong" for r in regs),
            n_faint=sum(r["tier"] == "faint" for r in regs),
            n_large=sum(r["size_class"] == "large" for r in regs),
            n_medium=sum(r["size_class"] == "medium" for r in regs),
            n_small=sum(r["size_class"] == "small" for r in regs),
            largest_region_px=max((r["area_px"] for r in regs), default=0),
            mask_px=mask_px,
            mask_fraction_pct=round(100 * mask_px / result.mask.size, 3),
            n_below_min_area=result.n_below_min_area,
        )
        summary_rows.append(row)
        for r in regs:
            extra = {"time_earliest_s": t_early, "time_latest_s": t_late, "time_status": time_status or None}
            if nmpp is not None:
                extra.update(
                    nm_per_pixel_used=round(nmpp, 4),
                    scale_status=scale_status,
                    area_nm2=round(r["area_px"] * nmpp**2, 1),
                    centroid_x_nm=round(r["centroid_x_px"] * nmpp, 1),
                    centroid_y_nm=round(r["centroid_y_px"] * nmpp, 1),
                    approx_major_axis_nm=round(r["approx_major_axis_px"] * nmpp, 1),
                    approx_minor_axis_nm=round(r["approx_minor_axis_px"] * nmpp, 1),
                )
            region_rows.append({"chronological_rank": rank, "filename": name, **r, **extra,
                                "run_label": run_label, "note": NOTE})
        preview_items.append((name, arr, result))

    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    regions = pd.DataFrame(region_rows, columns=REGION_COLUMNS)
    summary_path = out_dir / "segmentation_summary.csv"
    regions_path = out_dir / "regions.csv"
    preview_path = out_dir / "segmentation_preview.png"
    summary.to_csv(summary_path, index=False)
    regions.to_csv(regions_path, index=False)
    _preview_sheet(preview_items, preview_path, columns, run_label)
    return {
        "summary": summary, "regions": regions,
        "summary_path": summary_path, "regions_path": regions_path,
        "preview_path": preview_path, "masks_dir": masks_dir, "labels_dir": labels_dir,
    }


# ------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find and number every bright region (PRELIMINARY).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--columns", type=int, default=10)
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        out_dir = args.out_dir or (PROJECT_ROOT / "outputs" / f"segmentation_{config.run_label}")
        res = run_segmentation(
            args.data_dir, out_dir, config.segmentation, config.run_label, args.columns, config
        )
    except InventoryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    summary, regions = res["summary"], res["regions"]
    ok = summary[summary["read_error"].isna()]
    print(f"Images processed : {len(summary)} ({len(ok)} readable)")
    if len(ok):
        print(f"Regions listed   : {len(regions)} in total, {int(ok['n_regions'].min())} - {int(ok['n_regions'].max())} per image")
        print(f"  by tier        : strong {int(ok['n_strong'].sum())}, faint {int(ok['n_faint'].sum())}")
        print(f"  by size        : large {int(ok['n_large'].sum())}, medium {int(ok['n_medium'].sum())}, small {int(ok['n_small'].sum())}")
        print(f"Noise sigma      : {ok['noise_sigma'].astype(float).min():.2f} - {ok['noise_sigma'].astype(float).max():.2f} (intensity levels)")
        print(f"Mask coverage    : {ok['mask_fraction_pct'].astype(float).min():.1f}% - {ok['mask_fraction_pct'].astype(float).max():.1f}% of each image")
    if len(ok) and ok["scale_status"].notna().any():
        lo, hi = ok["nm_per_pixel_used"].astype(float).min(), ok["nm_per_pixel_used"].astype(float).max()
        print(f"Scale used       : {lo:.2f} - {hi:.2f} nm/pixel ({config.scale.status}, not verified against raw data)")
    else:
        print("Scale used       : none (unknown); lengths are in pixels only")
    if len(summary) and summary["time_status"].notna().any():
        print(f"Time ranges      : each image has an earliest and latest scan start ({config.time.status}); exact times unknown")
    warnings = []
    if len(ok) != len(summary):
        warnings.append("unreadable: " + ", ".join(summary[summary["read_error"].notna()]["filename"]))
    empty = ok[ok["n_regions"].astype(int) == 0]
    if len(empty):
        warnings.append("no region found in: " + ", ".join(empty["filename"]))
    heavy = ok[ok["mask_fraction_pct"].astype(float) > 25]
    if len(heavy):
        warnings.append("regions cover over 25% of the image (check): " + ", ".join(heavy["filename"]))
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  ! {w}")
    print("\nSTATUS: PRELIMINARY. Regions are based on JPG intensity in pixel units.")
    print("        Not AFM height; not yet classified as DNA / fragment / debris.")
    print(f"\nRegion table : {res['regions_path']}")
    print(f"Image summary: {res['summary_path']}")
    print(f"Preview      : {res['preview_path']}")
    print(f"Masks / label images: {res['masks_dir']} , {res['labels_dir']}")
    return 1 if len(ok) != len(summary) else 0


if __name__ == "__main__":
    sys.exit(main())
