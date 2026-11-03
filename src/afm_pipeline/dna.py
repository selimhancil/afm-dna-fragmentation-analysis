"""Phase 2: preprocessing and DNA-level segmentation (JPG intensity only, pixel units).

Pipeline for one image
----------------------
1. crop_borders      remove white screenshot borders / edge ticks (auto-detected)
2. preprocess        luminance -> remove scan-line offsets -> remove slow background
                     -> light Gaussian denoise -> robust noise estimate (MAD)
3. dna mask          hysteresis threshold (low/high, in noise-sigmas) -> closing that
                     bridges the beaded gaps of a DNA strand -> manual corrections
4. skeleton          thin mask, count end points / branch points / length per object
5. classification    each object is a CANDIDATE only:
                       closed_loop_candidate  - encloses a hole of at least min_loop_hole_px
                       open_candidate         - no enclosed hole, long enough
                       small_object           - short; kept and listed, never discarded
6. bright features   compact spots much brighter than the DNA itself

Nothing here measures AFM height, and nothing decides what is DNA, polymer, enzyme or
debris. "closed_loop_candidate" only means the mask has a hole in it.
"""
from __future__ import annotations

import configparser
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "project_config.ini"


class DnaConfigError(ValueError):
    """The [dna] settings are invalid."""


@dataclass(frozen=True)
class DnaSettings:
    border_white_level: int = 200        # a pixel counts as "white border" above this (all channels)
    border_fraction: float = 0.25        # row/column is border if this share of it is white
    border_margin_px: int = 2            # extra pixels trimmed after the border
    background_sigma_px: float = 25.0    # scale of the slow background that is removed
    denoise_sigma_px: float = 1.0        # Gaussian blur before thresholding
    low_k: float = 3.5                   # hysteresis low threshold, in noise-sigmas
    high_k: float = 6.0                  # hysteresis high threshold, in noise-sigmas
    bridge_radius_px: int = 2            # closing radius that joins beads of one strand
    min_object_area_px: int = 10         # smaller specks are not listed
    min_dna_length_px: float = 40.0      # skeleton length separating open_candidate / small_object
    min_loop_hole_px: int = 2000         # largest hole must be this big for closed_loop_candidate (plasmid-ring scale)
    min_hole_px: int = 150               # smaller holes (knots, tail loops) are only counted in n_holes
    max_loop_bridge_px: int = 10         # largest gap-bridging radius tried when asking "closes if bridged?"
    edge_fraction: float = 0.4           # outline at this fraction of the strand's own ridge level
    bright_rel_level: float = 1.5        # bright feature peak must reach this multiple of the ridge level
    bright_min_area_px: int = 6
    bright_min_radius_factor: float = 1.5  # spot must be this much thicker than the strand


def load_dna_settings(path: Path | str = DEFAULT_CONFIG_PATH) -> DnaSettings:
    """Read the optional [dna] section; missing keys keep their defaults."""
    d = DnaSettings()
    cp = configparser.ConfigParser(interpolation=None)
    try:
        cp.read_string(Path(path).read_text(encoding="utf-8"))
    except (OSError, configparser.Error) as exc:
        raise DnaConfigError(f"Could not read {path}: {exc}") from exc
    if not cp.has_section("dna"):
        return d
    values = {}
    for key, default in asdict(d).items():
        if cp.has_option("dna", key):
            try:
                values[key] = type(default)(cp.get("dna", key))
            except ValueError as exc:
                raise DnaConfigError(f"[dna] {key} is not a valid number: {exc}") from exc
    s = DnaSettings(**values)
    _validate(s)
    return s


def _validate(s: DnaSettings) -> None:
    if not (0 < s.low_k <= s.high_k):
        raise DnaConfigError("[dna] low_k must be positive and not larger than high_k.")
    if not (0 <= s.denoise_sigma_px <= 10):
        raise DnaConfigError("[dna] denoise_sigma_px must be between 0 and 10.")
    if s.background_sigma_px <= 0:
        raise DnaConfigError("[dna] background_sigma_px must be positive.")
    if s.bridge_radius_px < 0:
        raise DnaConfigError("[dna] bridge_radius_px cannot be negative.")
    if min(s.min_object_area_px, s.min_loop_hole_px, s.min_hole_px, s.bright_min_area_px) < 1:
        raise DnaConfigError("[dna] area limits must be at least 1.")
    if s.min_dna_length_px <= 0:
        raise DnaConfigError("[dna] min_dna_length_px must be positive.")
    if not (0 < s.border_fraction <= 1):
        raise DnaConfigError("[dna] border_fraction must be in (0, 1].")


# ---------------------------------------------------------------- preprocessing

def crop_borders(rgb: np.ndarray, s: DnaSettings) -> tuple[int, int, int, int]:
    """Return (top, bottom, left, right) crop box (bottom/right exclusive).

    Rows/columns at the image edge that are mostly white are screenshot borders, not sample.
    Trimming works inward from each side and stops at the first non-border line.
    """
    h, w = rgb.shape[:2]
    white = rgb.min(axis=2) > s.border_white_level
    row_frac = white.mean(axis=1)
    col_frac = white.mean(axis=0)
    top, bottom, left, right = 0, h, 0, w
    while top < h and row_frac[top] > s.border_fraction:
        top += 1
    while bottom > top and row_frac[bottom - 1] > s.border_fraction:
        bottom -= 1
    while left < w and col_frac[left] > s.border_fraction:
        left += 1
    while right > left and col_frac[right - 1] > s.border_fraction:
        right -= 1
    m = s.border_margin_px
    top, left = top + m, left + m
    bottom, right = bottom - m, right - m
    if bottom - top < 32 or right - left < 32:
        raise ValueError("Border detection left less than 32 px of image; check the image.")
    return top, bottom, left, right


@dataclass
class Preprocessed:
    crop_box: tuple[int, int, int, int]
    rgb: np.ndarray          # cropped colour image (uint8), for display only
    flat: np.ndarray         # flattened, denoised intensity (arbitrary units)
    noise: float             # robust noise of `flat`
    z: np.ndarray            # flat / noise


def _robust_sigma(values: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(values - np.median(values))))


def preprocess(rgb: np.ndarray, s: DnaSettings) -> Preprocessed:
    box = crop_borders(rgb, s)
    t, b, l, r = box
    crop = rgb[t:b, l:r]
    lum = (0.299 * crop[..., 0] + 0.587 * crop[..., 1] + 0.114 * crop[..., 2]).astype(np.float64)
    lum = lum - np.median(lum, axis=1, keepdims=True)            # scan-line offsets
    flat = lum - ndi.gaussian_filter(lum, s.background_sigma_px, mode="reflect")
    flat = flat - np.median(flat)
    # Second pass: estimate the background again WITHOUT the bright objects, so the DNA
    # itself does not pull the background up and leave dark halos around it.
    noise0 = _robust_sigma(flat)
    keep = ndi.binary_dilation(flat > 3 * noise0, iterations=3)
    weight = (~keep).astype(np.float64)
    num = ndi.gaussian_filter(lum * weight, s.background_sigma_px, mode="reflect")
    den = ndi.gaussian_filter(weight, s.background_sigma_px, mode="reflect")
    background = num / np.maximum(den, 1e-6)
    flat = lum - background
    flat = flat - np.median(flat[~keep]) if (~keep).any() else flat - np.median(flat)
    if s.denoise_sigma_px > 0:
        flat = ndi.gaussian_filter(flat, s.denoise_sigma_px, mode="reflect")
    noise = _robust_sigma(flat)
    if noise <= 0:
        raise ValueError("Image has no measurable noise level (flat image?).")
    return Preprocessed(box, crop, flat, noise, flat / noise)


# ---------------------------------------------------------------- mask + skeleton

def ridge_level(z: np.ndarray, mask: np.ndarray, s: DnaSettings) -> float | None:
    """Typical intensity along the centre line of the long objects of `mask` (noise units)."""
    skel = morphology.skeletonize(mask)
    labels, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0 or not skel.any():
        return None
    lengths = ndi.sum(skel, labels, index=np.arange(1, n + 1))
    long_ids = [i + 1 for i, v in enumerate(lengths) if v >= s.min_dna_length_px]
    sel = skel & np.isin(labels, long_ids) if long_ids else skel
    return float(np.median(z[sel])) if sel.any() else None


def dna_mask(z: np.ndarray, s: DnaSettings) -> np.ndarray:
    """Hysteresis mask. The outline is placed at `edge_fraction` of the strand's own ridge
    level, so frames with a low noise estimate do not get a fatter mask than noisy ones."""
    mask = filters.apply_hysteresis_threshold(z, s.low_k, s.high_k)
    level = ridge_level(z, mask, s)
    if level is not None:
        low = min(max(s.low_k, s.edge_fraction * level), s.high_k)
        mask = filters.apply_hysteresis_threshold(z, low, s.high_k)
    if s.bridge_radius_px > 0:
        mask = morphology.closing(mask, morphology.disk(s.bridge_radius_px))
    return mask


def skeleton_length(skel: np.ndarray) -> float:
    """Length of a 1-px skeleton in pixels: orthogonal links 1, diagonal links sqrt(2)."""
    sk = skel.astype(bool)
    orth = (sk[:, 1:] & sk[:, :-1]).sum() + (sk[1:, :] & sk[:-1, :]).sum()
    diag = (sk[1:, 1:] & sk[:-1, :-1]).sum() + (sk[1:, :-1] & sk[:-1, 1:]).sum()
    return float(orth + math.sqrt(2) * diag)


def neighbour_count(skel: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), dtype=int)
    k[1, 1] = 0
    return ndi.convolve(skel.astype(int), k, mode="constant") * skel


@dataclass
class DnaResult:
    mask: np.ndarray
    labels: np.ndarray            # object label image (0 = background)
    skeleton: np.ndarray
    endpoints: np.ndarray
    branchpoints: np.ndarray
    objects: list[dict]
    bright_labels: np.ndarray
    bright: list[dict]


def _hole_areas(obj_mask: np.ndarray) -> np.ndarray:
    """Areas (px) of all enclosed holes of a mask, largest first."""
    filled = ndi.binary_fill_holes(obj_mask)
    holes, n = ndi.label(filled & ~obj_mask)
    if n == 0:
        return np.zeros(0, dtype=int)
    return np.sort(np.bincount(holes.ravel())[1:])[::-1]


def _enclosed_holes(obj_mask: np.ndarray, min_area: int) -> int:
    return int((_hole_areas(obj_mask) >= min_area).sum())


def loop_bridge_radius(obj_mask: np.ndarray, s: DnaSettings) -> int | None:
    """Smallest closing radius (px) at which this object encloses a loop, or None.

    A ring with a small gap is not a closed loop in the mask, but would become one if the
    gap were bridged. The radius tells how big the gap is. It is a sensitivity measure,
    not a claim that the gap is imaging noise.
    """
    ys, xs = np.nonzero(obj_mask)
    pad = s.max_loop_bridge_px + 2
    y0, y1 = max(ys.min() - pad, 0), ys.max() + pad + 1
    x0, x1 = max(xs.min() - pad, 0), xs.max() + pad + 1
    sub = obj_mask[y0:y1, x0:x1]
    for r in range(0, s.max_loop_bridge_px + 1):
        closed = morphology.closing(sub, morphology.disk(r)) if r else sub
        if _enclosed_holes(closed, s.min_loop_hole_px) >= 1:
            return r
    return None


def analyse(pre: Preprocessed, s: DnaSettings, mask: np.ndarray | None = None) -> DnaResult:
    """Skeletonise the mask, classify objects, find bright features."""
    if mask is None:
        mask = dna_mask(pre.z, s)
    labels, n = ndi.label(mask.astype(bool), structure=np.ones((3, 3)))
    areas = np.bincount(labels.ravel(), minlength=n + 1)
    small = areas < s.min_object_area_px
    small[0] = False
    labels[small[labels]] = 0
    mask = labels > 0
    labels, n = ndi.label(mask, structure=np.ones((3, 3)))
    skel = morphology.skeletonize(mask)
    nb = neighbour_count(skel)
    endpoints = skel & (nb == 1)
    branchpoints = skel & (nb >= 3)
    edt = ndi.distance_transform_edt(mask)

    objects = []
    for rp in measure.regionprops(labels, intensity_image=pre.z):
        sel = labels == rp.label
        sk = skel & sel
        length = skeleton_length(sk)
        areas = _hole_areas(sel)
        holes = int((areas >= s.min_hole_px).sum())
        largest = int(areas[0]) if len(areas) else 0
        if largest >= s.min_loop_hole_px:
            kind = "closed_loop_candidate"
        elif length >= s.min_dna_length_px:
            kind = "open_candidate"
        else:
            kind = "small_object"
        cy, cx = rp.centroid
        bh, bw = int(rp.bbox[2] - rp.bbox[0]), int(rp.bbox[3] - rp.bbox[1])
        streak = bool(bh <= 4 and bw >= 12)   # thin horizontal line: probably a scan-line artifact
        objects.append({
            "object_id": rp.label, "kind": kind, "scan_line_suspect": streak,
            "area_px": int(rp.area), "skeleton_length_px": round(length, 2),
            "n_endpoints": int((endpoints & sel).sum()),
            "n_branchpoints": int((branchpoints & sel).sum()),
            "n_holes": holes, "largest_hole_px": largest,
            "loop_bridge_radius_px": (0 if largest >= s.min_loop_hole_px else loop_bridge_radius(sel, s)),
            "mean_width_px": round(float(2 * edt[sk].mean()), 2) if sk.any() else 0.0,
            "mean_z": round(float(rp.intensity_mean), 2),
            "centroid_x_px": round(cx, 1), "centroid_y_px": round(cy, 1),
            "bbox_w_px": bw, "bbox_h_px": bh,
            "touches_border": bool(rp.bbox[0] == 0 or rp.bbox[1] == 0
                                   or rp.bbox[2] == labels.shape[0] or rp.bbox[3] == labels.shape[1]),
        })

    bright_labels, bright = _bright_features(pre, s, mask, skel, labels, edt)
    return DnaResult(mask, labels, skel, endpoints, branchpoints, objects, bright_labels, bright)


def _bright_features(pre, s, mask, skel, labels, edt):
    empty = np.zeros(mask.shape, dtype=np.int32)
    if mask.sum() < 20:
        return empty, []
    level = ridge_level(pre.z, mask, s)
    if level is None or level <= 0:
        return empty, []
    thr = s.bright_rel_level * level
    cand = (pre.z > thr) & mask
    blobs, n = ndi.label(cand, structure=np.ones((3, 3)))
    strand_radius = float(np.median(edt[skel])) if skel.any() else 1.0
    dist_to_skel = ndi.distance_transform_edt(~skel)
    out, keep = [], np.zeros(mask.shape, dtype=np.int32)
    next_id = 1
    for rp in measure.regionprops(blobs, intensity_image=pre.z):
        if rp.area < s.bright_min_area_px:
            continue
        sel = blobs == rp.label
        radius = float(edt[sel].max())
        if radius < s.bright_min_radius_factor * strand_radius:
            continue  # as thin as the strand: just a brighter stretch of DNA
        cy, cx = rp.centroid
        owner = int(np.bincount(labels[sel]).argmax())
        keep[sel] = next_id
        out.append({
            "feature_id": next_id, "label": "bright feature (candidate bound particle)",
            "area_px": int(rp.area), "peak_z": round(float(rp.intensity_max), 2),
            "relative_to_ridge_level": round(float(rp.intensity_max) / max(level, 1e-6), 2),
            "radius_px": round(radius, 2),
            "centroid_x_px": round(cx, 1), "centroid_y_px": round(cy, 1),
            "dist_to_skeleton_px": round(float(dist_to_skel[int(cy), int(cx)]), 2),
            "object_id": owner,
        })
        next_id += 1
    return keep, out
