"""Step 2: read-only inventory of the AFM JPG images.

What this module does
---------------------
* Finds the JPG files in a data folder (macOS metadata files are ignored).
* Records filename, dimensions, channel count and basic pixel statistics.
* Writes everything to a CSV file in a separate output folder.

What this module deliberately does NOT do
-----------------------------------------
* It never modifies, renames, moves or re-saves an original image.
* It never decides the chronological order. Files are listed in a fixed,
  reproducible order (plain sorting of the filename) ONLY so that runs are
  repeatable. That order has no scientific meaning. The column
  ``chronological_rank`` is left empty until the real order is confirmed.
* It never converts pixel brightness into AFM height. The ``pixel_*``
  columns describe JPG intensity values (0-255) and nothing else.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

VALID_SUFFIXES = {".jpg", ".jpeg"}
EXPECTED_COUNT = 74

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "image_inventory.csv"

CSV_COLUMNS = [
    "listing_index",
    "chronological_rank",
    "filename",
    "file_size_bytes",
    "sha256",
    "format",
    "mode",
    "width_px",
    "height_px",
    "channels",
    "dtype",
    "channels_identical",
    "pixel_min",
    "pixel_max",
    "pixel_mean",
    "pixel_std",
    "pixel_median",
    "pixel_mean_ch1",
    "pixel_mean_ch2",
    "pixel_mean_ch3",
    "read_error",
]


class InventoryError(Exception):
    """A problem the user can fix (missing folder, no images, bad output path)."""


def find_image_files(data_dir: Path | str) -> tuple[list[Path], list[Path]]:
    """Return (image_files, ignored_visible_files).

    Hidden files (names starting with '.') such as .DS_Store, ._1.jpg and
    .gitkeep are silently ignored. Visible files that are not JPGs are not
    used but are reported so nothing disappears without a trace.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise InventoryError(
            f"Data folder not found: {data_dir}\n"
            "Create it and copy your JPG files into it (copy, do not move)."
        )
    if not data_dir.is_dir():
        raise InventoryError(f"Data path is not a folder: {data_dir}")

    images: list[Path] = []
    ignored: list[Path] = []
    for path in data_dir.iterdir():
        if path.name.startswith("."):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in VALID_SUFFIXES:
            images.append(path)
        else:
            ignored.append(path)

    if not images:
        raise InventoryError(
            f"No .jpg/.jpeg files found in: {data_dir}\n"
            "Copy the AFM images into this folder and run again."
        )

    # Fixed order for reproducibility only. NOT chronological order.
    images.sort(key=lambda p: p.name)
    ignored.sort(key=lambda p: p.name)
    return images, ignored


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> dict:
    """Read one image (read-only) and return one CSV row as a dict."""
    row: dict = {col: pd.NA for col in CSV_COLUMNS}
    row["filename"] = path.name
    row["file_size_bytes"] = path.stat().st_size
    row["sha256"] = sha256_of_file(path)

    try:
        with Image.open(path) as img:  # opened read-only; file is never written
            img.load()
            row["format"] = img.format
            row["mode"] = img.mode
            row["width_px"], row["height_px"] = img.size
            arr = np.asarray(img)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        row["read_error"] = f"{type(exc).__name__}: {exc}"
        return row

    channels = 1 if arr.ndim == 2 else arr.shape[2]
    row["channels"] = channels
    row["dtype"] = str(arr.dtype)
    row["pixel_min"] = float(arr.min())
    row["pixel_max"] = float(arr.max())
    row["pixel_mean"] = float(arr.mean())
    row["pixel_std"] = float(arr.std())
    row["pixel_median"] = float(np.median(arr))

    if channels >= 3:
        row["channels_identical"] = bool(
            np.array_equal(arr[..., 0], arr[..., 1])
            and np.array_equal(arr[..., 0], arr[..., 2])
        )
        for i in range(3):
            row[f"pixel_mean_ch{i + 1}"] = float(arr[..., i].mean())
    else:
        row["channels_identical"] = True  # single channel: nothing to compare
        row["pixel_mean_ch1"] = float(arr.mean())
    return row


def build_inventory(data_dir: Path | str) -> tuple[pd.DataFrame, list[Path]]:
    images, ignored = find_image_files(data_dir)
    rows = []
    for index, path in enumerate(images, start=1):
        row = inspect_image(path)
        row["listing_index"] = index  # reproducible listing only, NOT time order
        rows.append(row)
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df["chronological_rank"] = pd.array([pd.NA] * len(df), dtype="Int64")
    return df, ignored


def write_inventory_csv(df: pd.DataFrame, out_path: Path | str, data_dir: Path | str) -> Path:
    out_path = Path(out_path).resolve()
    data_dir = Path(data_dir).resolve()
    if out_path == data_dir or data_dir in out_path.parents:
        raise InventoryError(
            f"Refusing to write the CSV inside the data folder ({data_dir}).\n"
            "Original images must stay untouched; choose an output path in outputs/."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def summarize(df: pd.DataFrame, ignored: list[Path], expected_count: int | None) -> list[str]:
    """Human-readable summary lines, including warnings."""
    lines = [f"Images found: {len(df)}"]
    warnings: list[str] = []

    if expected_count is not None and len(df) != expected_count:
        warnings.append(f"Expected {expected_count} images but found {len(df)}.")

    failed = df[df["read_error"].notna()]
    if len(failed):
        warnings.append(f"{len(failed)} image(s) could not be read:")
        for _, r in failed.iterrows():
            warnings.append(f"  - {r['filename']}: {r['read_error']}")

    ok = df[df["read_error"].isna()]
    if len(ok):
        lines.append(
            f"Width range : {int(ok['width_px'].min())}-{int(ok['width_px'].max())} px"
        )
        lines.append(
            f"Height range: {int(ok['height_px'].min())}-{int(ok['height_px'].max())} px"
        )
        lines.append(f"Channel counts: {sorted(ok['channels'].astype(int).unique().tolist())}")
        modes = sorted(ok["mode"].astype(str).unique().tolist())
        lines.append(f"Color modes: {modes}")

    dup = df[df["sha256"].duplicated(keep=False)]
    if len(dup):
        warnings.append(
            "Identical files (same sha256): "
            + ", ".join(sorted(dup["filename"].astype(str)))
        )

    if ignored:
        warnings.append(
            "Ignored non-JPG files: " + ", ".join(p.name for p in ignored)
        )

    lines.append("")
    lines.append("NOTE: 'listing_index' is only a reproducible listing order,")
    lines.append("      NOT the chronological order. 'chronological_rank' is empty.")
    lines.append("NOTE: pixel_* columns are JPG intensity values (0-255),")
    lines.append("      NOT AFM height measurements.")
    if warnings:
        lines.append("")
        lines.append("WARNINGS:")
        lines.extend(f"  ! {w}" if not w.startswith("  ") else w for w in warnings)
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a CSV inventory of AFM JPG images.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=EXPECTED_COUNT,
        help="Warn if a different number of images is found (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    try:
        df, ignored = build_inventory(args.data_dir)
        out = write_inventory_csv(df, args.output, args.data_dir)
    except InventoryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("\n".join(summarize(df, ignored, args.expected_count)))
    print(f"\nCSV written to: {out}")
    return 1 if df["read_error"].notna().any() else 0


if __name__ == "__main__":
    sys.exit(main())
