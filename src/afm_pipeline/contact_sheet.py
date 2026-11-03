"""Step 3: contact sheet showing every image with its filename.

The sheet is a VIEWING aid only:
* Originals are opened read-only and never re-saved or modified.
* Images appear in the reproducible listing order (plain filename sorting).
  That is NOT the chronological order, and the sheet says so in its title.
* No pixel value is converted to an AFM height.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to file only; no window needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

from afm_pipeline.inventory import (  # noqa: E402
    DEFAULT_DATA_DIR,
    EXPECTED_COUNT,
    PROJECT_ROOT,
    InventoryError,
    find_image_files,
)

DEFAULT_SHEET = PROJECT_ROOT / "outputs" / "contact_sheet.png"


def _load_for_display(path: Path) -> np.ndarray | None:
    """Return an RGB array copy for drawing, or None if the file is unreadable."""
    try:
        with Image.open(path) as img:  # read-only
            img.load()
            return np.asarray(img.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def build_contact_sheet(
    data_dir: Path | str,
    out_path: Path | str,
    columns: int = 10,
    cell_inches: float = 2.0,
    dpi: int = 150,
) -> tuple[Path, int, list[str]]:
    """Draw the sheet. Returns (output_path, n_images, unreadable_filenames)."""
    if columns < 1:
        raise InventoryError("--columns must be at least 1.")

    out_path = Path(out_path).resolve()
    data_resolved = Path(data_dir).resolve()
    if out_path == data_resolved or data_resolved in out_path.parents:
        raise InventoryError(
            f"Refusing to write the contact sheet inside the data folder ({data_resolved}).\n"
            "Original images must stay untouched; choose an output path in outputs/."
        )

    images, _ = find_image_files(data_dir)
    n = len(images)
    rows = math.ceil(n / columns)

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * cell_inches, rows * (cell_inches + 0.3) + 0.8),
        squeeze=False,
    )
    unreadable: list[str] = []
    for ax in axes.ravel():
        ax.axis("off")

    for ax, path in zip(axes.ravel(), images):
        arr = _load_for_display(path)
        if arr is None:
            unreadable.append(path.name)
            ax.text(0.5, 0.5, "unreadable", ha="center", va="center", color="red", fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            ax.imshow(arr, interpolation="nearest")
        ax.set_title(path.name, fontsize=8, pad=2)

    fig.suptitle(
        f"AFM images: {n} files, shown in listing order (sorted by filename). "
        "NOT chronological order.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path, n, unreadable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a contact sheet of the AFM JPG images.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_SHEET)
    parser.add_argument("--columns", type=int, default=10)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_COUNT)
    args = parser.parse_args(argv)

    try:
        out, n, unreadable = build_contact_sheet(args.data_dir, args.output, args.columns)
    except InventoryError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Images drawn: {n}")
    if args.expected_count is not None and n != args.expected_count:
        print(f"WARNING: expected {args.expected_count} images but found {n}.")
    if unreadable:
        print("WARNING: could not read: " + ", ".join(unreadable))
    print("NOTE: images are in listing order, NOT chronological order.")
    print(f"Contact sheet written to: {out}")
    return 1 if unreadable else 0


if __name__ == "__main__":
    sys.exit(main())
