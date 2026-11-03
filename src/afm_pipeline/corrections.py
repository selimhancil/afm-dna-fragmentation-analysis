"""Manual correction of DNA masks and evaluation against manual annotations.

Two simple tools, both optional and both leaving the original images untouched:

1. corrections CSV (annotations/corrections.csv), coordinates in ORIGINAL image pixels:
       filename,action,x,y,radius_px
       F1.jpg,erase,120,80,6      <- remove a false detection (a disk is cleared)
       F1.jpg,add,200,150,3       <- add DNA the software missed (a disk is painted)
   Several rows paint a stroke. Corrected images are flagged mask_source = "corrected".

2. manual annotation masks (annotations/<image stem>.png): white = DNA, same size as the
   original image. evaluate() compares them with the automatic mask (Dice, IoU,
   precision, recall) so the accuracy is reported as measured numbers, not asserted.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

ACTIONS = {"erase", "add"}


class CorrectionError(ValueError):
    """The corrections file is malformed."""


def load_corrections(path: Path | str) -> dict[str, list[dict]]:
    path = Path(path)
    if not path.is_file():
        return {}
    out: dict[str, list[dict]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        need = {"filename", "action", "x", "y", "radius_px"}
        if not reader.fieldnames or not need <= set(reader.fieldnames):
            raise CorrectionError(f"{path} needs columns: {', '.join(sorted(need))}")
        for n, row in enumerate(reader, start=2):
            action = row["action"].strip().lower()
            if action not in ACTIONS:
                raise CorrectionError(f"{path} line {n}: action must be 'erase' or 'add', got '{action}'")
            try:
                x, y, r = float(row["x"]), float(row["y"]), float(row["radius_px"])
            except ValueError as exc:
                raise CorrectionError(f"{path} line {n}: x, y, radius_px must be numbers") from exc
            if r <= 0:
                raise CorrectionError(f"{path} line {n}: radius_px must be positive")
            out.setdefault(row["filename"].strip(), []).append({"action": action, "x": x, "y": y, "r": r})
    return out


def apply_corrections(mask: np.ndarray, rows: list[dict], crop_box: tuple[int, int, int, int]) -> np.ndarray:
    """Apply erase/add disks (original-image coordinates) to a cropped mask."""
    top, _, left, _ = crop_box
    h, w = mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    out = mask.copy()
    for row in rows:
        disk = (xx - (row["x"] - left)) ** 2 + (yy - (row["y"] - top)) ** 2 <= row["r"] ** 2
        out = out & ~disk if row["action"] == "erase" else out | disk
    return out


def load_annotation(path: Path, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = crop_box
    arr = np.asarray(Image.open(path).convert("L")) > 127
    return arr[top:bottom, left:right]


def evaluate(auto: np.ndarray, manual: np.ndarray) -> dict:
    """Pixel-wise agreement between an automatic and a manual mask."""
    if auto.shape != manual.shape:
        raise CorrectionError(f"mask shapes differ: {auto.shape} vs {manual.shape}")
    a, m = auto.astype(bool), manual.astype(bool)
    tp = int((a & m).sum())
    fp = int((a & ~m).sum())
    fn = int((~a & m).sum())
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    return {
        "dice": round(dice, 4), "iou": round(iou, 4),
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "auto_px": int(a.sum()), "manual_px": int(m.sum()),
    }
