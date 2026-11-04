"""Phase 6b: stronger tracking of bright features along the DNA.

Weak points of Phase 6: 32 of 66 tracks were seen in a single frame (a detection that flickers
above/below a fixed threshold looks like many short tracks). This module adds:

  1. hysteresis tracking: strong detections (Phase 2 rule) are seeds; a track is extended to
     neighbouring frames wherever a weaker local peak (>= `weak_rel` x DNA ridge level) is found
     near the predicted place and on the DNA. Extended detections are flagged `recovered`.
  2. merging of track fragments that continue each other after a short gap;
  3. a confidence level per track (high / medium / low);
  4. spacing between features measured ALONG the DNA skeleton (shortest path), frame by frame.
     This is independent of any global registration and is the cleanest test of whether
     features slide relative to each other on the DNA;
  5. a picture strip of every long track (crop of the JPG in sampled frames) for visual checking;
  6. a before/after table.

Pixels and JPG intensity only. Features stay "bright features (candidate bound particles)":
no identification as enzyme or dendrimer.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric

from . import config as cfgmod
from .dna import load_dna_settings, preprocess, ridge_level
from .inventory import DEFAULT_DATA_DIR, PROJECT_ROOT
from .phase4 import cumulative_shifts
from .phase6 import (FeatureSettings, add_contour_position, reference_contour, summarise_tracks)
from .segment import load_rgb

DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"


@dataclass
class HysteresisSettings:
    weak_rel: float = 1.25           # weak peak must reach this multiple of the DNA ridge level
    search_radius_px: float = 6.0    # where to look around the predicted place
    max_miss: int = 2                # consecutive frames without a peak before a track stops
    on_dna_px: float = 4.0           # a recovered peak must be this close to the DNA mask
    min_separation_px: float = 5.0   # keep away from a detection of another track
    merge_gap_frames: int = 3        # fragments this close in time ...
    merge_radius_px: float = 8.0     # ... and in space are one track
    long_track: int = 6              # detections for a track to be analysed further
    patch_half_px: int = 20          # half width of the picture-strip crops
    snap_px: float = 8.0             # a feature farther than this from the skeleton is not used for spacing
    spacing_min_frames: int = 5
    spacing_change_px: float = 8.0   # total change that counts as "spacing changes"
    spacing_alpha: float = 0.01
    spacing_rel_change: float = 0.10  # ... and at least this fraction of the mean spacing (length-estimate noise ~5%)
    flip_jump_px: float = 30.0       # one frame-to-frame jump this large = the shortest path switched route (ring opened/closed)


# ------------------------------------------------------------------ peak search

def find_peak(z: np.ndarray, level: float, center: tuple[float, float], radius: float, weak_rel: float,
              near_dna: np.ndarray | None = None) -> dict | None:
    """Best local peak of `z` around `center` (y, x in frame pixels).

    Accepts the strongest pixel within `radius` if it reaches weak_rel x level, is a local maximum
    (not on the border of the search window) and lies on the DNA (`near_dna` mask)."""
    h, w = z.shape
    cy, cx = int(round(center[0])), int(round(center[1]))
    r = int(np.ceil(radius))
    y0, y1, x0, x1 = max(cy - r, 0), min(cy + r + 1, h), max(cx - r, 0), min(cx + r + 1, w)
    if y1 - y0 < 3 or x1 - x0 < 3:
        return None
    win = z[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius ** 2
    if not inside.any():
        return None
    masked = np.where(inside, win, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(masked)), masked.shape)
    peak = float(win[iy, ix])
    py, px = y0 + iy, x0 + ix
    if peak < weak_rel * level:
        return None
    ring = z[max(py - 1, 0):py + 2, max(px - 1, 0):px + 2]
    if peak < float(ring.max()):
        return None
    on_edge = (iy in (0, win.shape[0] - 1)) or (ix in (0, win.shape[1] - 1))
    if on_edge and (y1 - y0) >= 2 * r and (x1 - x0) >= 2 * r:
        return None                                    # still climbing at the window border
    if near_dna is not None and not near_dna[py, px]:
        return None
    # area: connected pixels above 70% of the peak, within the window
    region = (z[y0:y1, x0:x1] >= 0.7 * peak)
    lab, _ = ndi.label(region, structure=np.ones((3, 3)))
    area = int((lab == lab[iy, ix]).sum()) if lab[iy, ix] else 1
    return {"y": float(py), "x": float(px), "peak_z": peak, "area_px": max(area, 1),
            "relative_to_ridge_level": peak / level, "radius_px": float(np.sqrt(max(area, 1) / np.pi))}


# ------------------------------------------------------------------ frames cache

def load_frame_cache(data_dir: Path, outputs: Path, frames: pd.DataFrame, dna) -> dict:
    p2 = outputs / "phase2"
    cache = {}
    for row in frames.sort_values("rank").itertuples():
        pre = preprocess(load_rgb(data_dir / row.filename), dna)
        stem = Path(row.filename).stem
        mask = np.asarray(Image.open(p2 / "masks" / f"{int(row.rank):02d}_{stem}_mask.png").convert("L")) > 127
        level = ridge_level(pre.z, mask, dna)
        skel = np.asarray(Image.open(p2 / "skeletons" / f"{int(row.rank):02d}_{stem}_skeleton.png").convert("L")) > 127
        cache[int(row.rank)] = {"z": pre.z, "rgb": pre.rgb, "level": level, "mask": mask, "skel": skel,
                                "crop_top": int(row.crop_top), "crop_left": int(row.crop_left)}
    return cache


def to_frame(cache_r: dict, shifts_r: tuple[float, float], y_common: float, x_common: float) -> tuple[float, float]:
    return y_common - cache_r["crop_top"] - shifts_r[0], x_common - cache_r["crop_left"] - shifts_r[1]


def to_common(cache_r: dict, shifts_r: tuple[float, float], y: float, x: float) -> tuple[float, float]:
    return y + cache_r["crop_top"] + shifts_r[0], x + cache_r["crop_left"] + shifts_r[1]


# ------------------------------------------------------------------ hysteresis tracking

def extend_tracks(det: pd.DataFrame, cache: dict, shifts: dict, h: HysteresisSettings) -> pd.DataFrame:
    """Add weak ('recovered') detections to tracks, forward and backward in time."""
    det = det.copy()
    det["recovered"] = False
    ranks = sorted(cache)
    occupied: dict[int, list[tuple[float, float, int]]] = {r: [] for r in ranks}
    for r in det.itertuples():
        occupied[int(r.rank)].append((r.y_common, r.x_common, int(r.track_id)))
    near = {r: ndi.binary_dilation(c["mask"], iterations=int(np.ceil(h.on_dna_px))) for r, c in cache.items()}
    new_rows = []
    for tid, g in det.groupby("track_id"):
        g = g.sort_values("rank")
        for direction in (+1, -1):
            seed = g.iloc[-1] if direction > 0 else g.iloc[0]
            y_c, x_c = float(seed["y_common"]), float(seed["x_common"])
            rank, miss = int(seed["rank"]), 0
            while True:
                rank += direction
                if rank not in cache:
                    break
                c = cache[rank]
                if not c["level"]:
                    miss += 1
                    if miss > h.max_miss:
                        break
                    continue
                fy, fx = to_frame(c, shifts.get(rank, (0.0, 0.0)), y_c, x_c)
                already = [o for o in occupied[rank] if o[2] == tid]
                if already:                              # this track already has a detection here
                    y_c, x_c = already[0][0], already[0][1]
                    miss = 0
                    continue
                pk = find_peak(c["z"], c["level"], (fy, fx), h.search_radius_px, h.weak_rel, near[rank])
                if pk is not None:
                    yc, xc = to_common(c, shifts.get(rank, (0.0, 0.0)), pk["y"], pk["x"])
                    if any(o[2] != tid and np.hypot(o[0] - yc, o[1] - xc) < h.min_separation_px
                           for o in occupied[rank]):
                        pk = None
                if pk is None:
                    miss += 1
                    if miss > h.max_miss:
                        break
                    continue
                miss = 0
                y_c, x_c = yc, xc
                occupied[rank].append((yc, xc, tid))
                new_rows.append({"track_id": tid, "rank": rank, "filename": "", "feature_id": -1,
                                 "x_common": xc, "y_common": yc, "area_px": pk["area_px"],
                                 "relative_to_ridge_level": round(pk["relative_to_ridge_level"], 2),
                                 "radius_px": round(pk["radius_px"], 2), "peak_z": round(pk["peak_z"], 2),
                                 "gap_before": 0, "registration_reliable": True, "recovered": True,
                                 "label": "bright feature (candidate bound particle)"})
    if new_rows:
        add = pd.DataFrame(new_rows)
        fn = det.drop_duplicates("rank").set_index("rank")["filename"].to_dict()
        add["filename"] = add["rank"].map(fn)
        det = pd.concat([det, add], ignore_index=True)
    return det.sort_values(["track_id", "rank"]).reset_index(drop=True)


def null_extension(det: pd.DataFrame, cache: dict, shifts: dict, h: HysteresisSettings,
                   n_fake: int = 200, seed: int = 1) -> dict:
    """Control: start fake tracks at random DNA skeleton points (>=15 px from every real detection)
    and extend them with the same rule. If weak peaks were everywhere on the DNA, fake tracks would
    also grow long. Returns recovered-detection counts of fake and real tracks."""
    rng = np.random.default_rng(seed)
    ranks = sorted(cache)
    real = {r: det[det["rank"] == r][["y_common", "x_common"]].to_numpy() for r in ranks}
    rows = []
    tid = 100000
    tries = 0
    while len(rows) < n_fake and tries < 50 * n_fake:
        tries += 1
        r = int(rng.choice(ranks))
        c = cache[r]
        ys, xs = np.nonzero(c["skel"])
        if len(ys) == 0:
            continue
        i = int(rng.integers(len(ys)))
        yc, xc = to_common(c, shifts.get(r, (0.0, 0.0)), float(ys[i]), float(xs[i]))
        if len(real[r]) and np.min(np.hypot(real[r][:, 0] - yc, real[r][:, 1] - xc)) < 15:
            continue
        rows.append({"track_id": tid, "rank": r, "filename": "", "feature_id": -1, "x_common": xc, "y_common": yc,
                     "area_px": 1, "relative_to_ridge_level": 0.0, "radius_px": 0.0, "peak_z": 0.0, "gap_before": 0,
                     "registration_reliable": True, "label": "null"})
        tid += 1
    fake = pd.DataFrame(rows)
    fake["recovered"] = False
    ext = extend_tracks(fake, cache, shifts, h)
    n_rec = ext[ext["recovered"]].groupby("track_id").size().reindex(fake["track_id"]).fillna(0)
    real_rec = det[det["recovered"]].groupby("track_id").size()
    strong = det[~det["recovered"]].groupby("track_id").size()
    real_rec = real_rec.reindex(strong.index).fillna(0)
    return {"n_fake": int(len(fake)), "fake_mean_recovered": float(n_rec.mean()),
            "fake_share_with_6plus": float((n_rec >= h.long_track - 1).mean()),
            "real_mean_recovered": float(real_rec.mean()),
            "real_share_with_6plus": float((real_rec >= h.long_track - 1).mean())}


def merge_fragments(det: pd.DataFrame, h: HysteresisSettings) -> tuple[pd.DataFrame, int]:
    """Relabel a track as the continuation of an earlier one when it starts shortly after that one
    ends, at nearly the same place."""
    det = det.copy()
    merged = 0
    while True:
        spans = det.groupby("track_id").agg(first=("rank", "min"), last=("rank", "max"))
        ends = {t: det[(det.track_id == t) & (det["rank"] == spans.loc[t, "last"])].iloc[0] for t in spans.index}
        starts = {t: det[(det.track_id == t) & (det["rank"] == spans.loc[t, "first"])].iloc[0] for t in spans.index}
        best = None
        for a in spans.index:
            for b in spans.index:
                if a == b:
                    continue
                gap = spans.loc[b, "first"] - spans.loc[a, "last"]
                if not (1 <= gap <= h.merge_gap_frames):
                    continue
                d = float(np.hypot(ends[a]["y_common"] - starts[b]["y_common"], ends[a]["x_common"] - starts[b]["x_common"]))
                if d <= h.merge_radius_px and (best is None or d < best[0]):
                    best = (d, a, b)
        if best is None:
            return det, merged
        det.loc[det.track_id == best[2], "track_id"] = best[1]
        merged += 1


def track_confidence(trk: pd.DataFrame, det: pd.DataFrame, h: HysteresisSettings) -> pd.DataFrame:
    strong = det[~det["recovered"]].groupby("track_id").size()
    trk = trk.copy()
    trk["n_strong_detections"] = trk["track_id"].map(strong).fillna(0).astype(int)
    trk["n_recovered_detections"] = trk["n_detections"] - trk["n_strong_detections"]

    def conf(r):
        if r.n_detections >= h.long_track and r.n_strong_detections >= 3 and not r.crosses_unreliable_registration:
            return "high"
        if r.n_detections >= 3 and r.n_strong_detections >= 1:
            return "medium"
        return "low"

    trk["confidence"] = [conf(r) for r in trk.itertuples()]
    return trk


# ------------------------------------------------------------------ spacing along the DNA

def geodesic_between(skel: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> float | None:
    """Shortest path length (px) along skeleton pixels from a to b; None if not connected."""
    if not (skel[a] and skel[b]):
        return None
    cost = np.where(skel, 1.0, np.inf)
    mcp = MCP_Geometric(cost, fully_connected=True)
    costs, _ = mcp.find_costs([a], [b])
    v = float(costs[b])
    return None if not np.isfinite(v) else v


def snap_to_skeleton(skel: np.ndarray, y: float, x: float, max_px: float) -> tuple[int, int] | None:
    ys, xs = np.nonzero(skel)
    if len(ys) == 0:
        return None
    d, i = cKDTree(np.column_stack([ys, xs])).query([y, x])
    return (int(ys[i]), int(xs[i])) if d <= max_px else None


def classify_pair(frames: np.ndarray, d: np.ndarray, h: HysteresisSettings) -> dict:
    """Constant / changing spacing / path_flip.

    path_flip: some frame-to-frame jump exceeds flip_jump_px. On a ring or a loop the shortest path
    between two features can switch to the other way round when the loop opens or closes, which changes
    the spacing by hundreds of px without any feature moving. Such pairs are NOT counted as sliding.
    A genuine change must exceed both an absolute and a relative threshold (the pixel-chain length
    estimate itself varies by a few percent)."""
    if len(frames) < h.spacing_min_frames:
        return {"n_frames": len(frames), "verdict": "too_few_frames"}
    order = np.argsort(frames)
    frames, d = frames[order], d[order]
    lr = stats.linregress(frames, d)
    span = float(frames.max() - frames.min())
    total = float(lr.slope * span)
    jumps = np.abs(np.diff(d))
    max_jump = float(jumps.max()) if len(jumps) else 0.0
    need = max(h.spacing_change_px, h.spacing_rel_change * float(d.mean()))
    if max_jump > h.flip_jump_px:
        verdict = "path_flip"
    elif lr.pvalue < h.spacing_alpha and abs(total) >= need:
        verdict = "spacing_changes"
    else:
        verdict = "spacing_constant"
    return {"n_frames": len(frames), "mean_spacing_px": round(float(d.mean()), 1), "sd_px": round(float(d.std()), 2),
            "slope_px_per_frame": round(float(lr.slope), 3), "total_change_px": round(total, 1),
            "required_change_px": round(need, 1), "max_jump_px": round(max_jump, 1),
            "p_value": round(float(lr.pvalue), 4), "step_to_step_jitter_px": round(float(np.median(jumps)), 2),
            "verdict": verdict}


def combine_verdicts(sk: str, rf: str) -> tuple[str, str]:
    """Conservative combination of the two spacing measures.

    spacing_changes only when the reference-contour measure says so and the skeleton measure does not say
    'constant'; a disagreement stays visible as conflicting_measures instead of being resolved by a coin toss."""
    unknown = {"too_few_frames"}
    if rf in unknown and sk in unknown:
        return "undetermined", "no measure has enough frames"
    if rf in unknown:
        return ("undetermined", "skeleton path flipped, no reference-contour measure") if sk == "path_flip" else (sk, "skeleton_path only")
    if rf == "spacing_changes":
        if sk == "spacing_constant":
            return "conflicting_measures", "reference contour says change, skeleton path says constant"
        return "spacing_changes", "reference contour change, not contradicted by the skeleton path"
    # rf == spacing_constant
    if sk == "spacing_changes":
        return "conflicting_measures", "skeleton path says change, reference contour says constant"
    return "spacing_constant", "reference contour constant, skeleton path constant or flipped"


def pair_spacing(det: pd.DataFrame, trk: pd.DataFrame, cache: dict, shifts: dict, h: HysteresisSettings,
                 aligned_ranks: set | None = None):
    """Spacing between two tracked features, two ways:
      * along each frame's own skeleton (shortest path): independent of registration, but the route can
        switch on a ring (path_flip);
      * along the reference contour (difference of the contour positions s): route-consistent, but only in
        frames aligned to the reference (`aligned_ranks`).
    The combined verdict uses the reference-contour measure when it has enough frames, otherwise the skeleton one."""
    ids = [int(t) for t in trk.loc[trk["n_detections"] >= h.long_track, "track_id"]]
    pos = {(int(r.track_id), int(r.rank)): (r.y_common, r.x_common) for r in det.itertuples()}
    spos = {}
    if aligned_ranks is not None and "s_px" in det:
        for r in det.itertuples():
            if int(r.rank) in aligned_ranks and not bool(r.off_reference_contour) and np.isfinite(r.s_px):
                spos[(int(r.track_id), int(r.rank))] = float(r.s_px)
    series, summary = [], []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            common = sorted(set(r for (t, r) in pos if t == a) & set(r for (t, r) in pos if t == b))
            rows = []
            for r in common:
                c = cache[r]
                sh = shifts.get(r, (0.0, 0.0))
                pa = snap_to_skeleton(c["skel"], *to_frame(c, sh, *pos[(a, r)]), h.snap_px)
                pb = snap_to_skeleton(c["skel"], *to_frame(c, sh, *pos[(b, r)]), h.snap_px)
                if pa is None or pb is None:
                    continue
                g = geodesic_between(c["skel"], pa, pb)
                if g is not None:
                    rows.append((r, g))
            ref = [(r, abs(spos[(a, r)] - spos[(b, r)])) for r in common if (a, r) in spos and (b, r) in spos]
            if not rows and not ref:
                continue
            res = {"track_a": a, "track_b": b}
            if rows:
                fr = np.array([x[0] for x in rows], dtype=float)
                dd = np.array([x[1] for x in rows], dtype=float)
                for r, g in rows:
                    series.append({"track_a": a, "track_b": b, "rank": r, "spacing_along_dna_px": round(g, 1)})
                sk = classify_pair(fr, dd, h)
            else:
                sk = {"n_frames": 0, "verdict": "too_few_frames"}
            rf = classify_pair(np.array([x[0] for x in ref], dtype=float), np.array([x[1] for x in ref], dtype=float), h) if ref else {"n_frames": 0, "verdict": "too_few_frames"}
            res.update(sk)
            res["verdict_skeleton_path"] = sk["verdict"]
            res["n_frames_reference_contour"] = rf["n_frames"]
            res["verdict_reference_contour"] = rf["verdict"]
            res["total_change_reference_px"] = rf.get("total_change_px", np.nan)
            res["verdict"], res["verdict_reason"] = combine_verdicts(sk["verdict"], rf["verdict"])
            summary.append(res)
    return pd.DataFrame(series), pd.DataFrame(summary)


# ------------------------------------------------------------------ figures

def montage(det: pd.DataFrame, trk: pd.DataFrame, cache: dict, shifts: dict, h: HysteresisSettings, out: Path,
            max_tracks: int = 14, n_cols: int = 10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = trk[(trk["n_detections"] >= h.long_track)].sort_values(["confidence", "n_detections"],
                                                                  ascending=[True, False]).head(max_tracks)
    if pick.empty:
        return
    half = h.patch_half_px
    fig, axs = plt.subplots(len(pick), n_cols, figsize=(1.35 * n_cols, 1.45 * len(pick)), squeeze=False)
    for ri, t in enumerate(pick.itertuples()):
        g = det[det["track_id"] == t.track_id].sort_values("rank")
        sel = g.iloc[np.unique(np.linspace(0, len(g) - 1, n_cols).round().astype(int))]
        for ci in range(n_cols):
            ax = axs[ri, ci]
            ax.axis("off")
            if ci >= len(sel):
                continue
            d = sel.iloc[ci]
            c = cache[int(d["rank"])]
            fy, fx = to_frame(c, shifts.get(int(d["rank"]), (0.0, 0.0)), d["y_common"], d["x_common"])
            img = np.zeros((2 * half + 1, 2 * half + 1, 3), dtype=np.uint8)
            rgb = c["rgb"]
            y0, x0 = int(round(fy)) - half, int(round(fx)) - half
            ys, xs = max(y0, 0), max(x0, 0)
            ye, xe = min(y0 + 2 * half + 1, rgb.shape[0]), min(x0 + 2 * half + 1, rgb.shape[1])
            if ye > ys and xe > xs:
                img[ys - y0:ye - y0, xs - x0:xe - x0] = rgb[ys:ye, xs:xe]
            ax.imshow(img)
            ax.add_patch(plt.Circle((half, half), 9, fill=False, lw=0.9,
                                    ec="cyan" if not d["recovered"] else "yellow",
                                    ls="-" if not d["recovered"] else "--"))
            ax.set_title(f"{int(d['rank'])}", fontsize=6, pad=1)
        axs[ri, 0].text(-0.15, 0.5, f"track {t.track_id}\n{t.confidence}", transform=axs[ri, 0].transAxes,
                        ha="right", va="center", fontsize=6)
    fig.suptitle("Bright-feature tracks, sampled frames (cyan = strong detection, yellow dashed = weak/recovered)",
                 fontsize=8, y=0.995)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_track_montage.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def spacing_figure(series: pd.DataFrame, summary: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if series.empty:
        return
    ok = summary[summary["n_frames"] >= 5]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in ok.itertuples():
        g = series[(series.track_a == r.track_a) & (series.track_b == r.track_b)].sort_values("rank")
        ax.plot(g["rank"], g["spacing_along_dna_px"], marker="o", ms=2.5, lw=1,
                color={"spacing_changes": "#d9534f", "conflicting_measures": "#e0a030", "undetermined": "#aaaaaa"}.get(r.verdict, "#2255c4"), alpha=0.8)
    ax.plot([], [], color="#2255c4", label="spacing constant")
    ax.plot([], [], color="#d9534f", label="spacing changes")
    ax.plot([], [], color="#e0a030", label="measures disagree")
    ax.plot([], [], color="#aaaaaa", label="undetermined")
    ax.set_xlabel("frame index (provisional order)")
    ax.set_ylabel("distance between two features along the DNA (px)")
    ax.set_title("Feature-to-feature spacing along the DNA skeleton", fontsize=9, loc="left")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_pair_spacing.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ summary

def comparison_table(trk_before: pd.DataFrame, trk_after: pd.DataFrame, h: HysteresisSettings) -> pd.DataFrame:
    def stats_of(t, name):
        return {"version": name, "n_tracks": len(t), "tracks_with_1_detection": int((t["n_detections"] == 1).sum()),
                "tracks_with_6plus_detections": int((t["n_detections"] >= h.long_track).sum()),
                "median_detections_per_track": float(t["n_detections"].median()),
                "total_detections": int(t["n_detections"].sum())}
    return pd.DataFrame([stats_of(trk_before, "phase6 (strong detections only)"),
                         stats_of(trk_after, "phase6b (hysteresis + merging)")])


def summary_text(cmp: pd.DataFrame, trk: pd.DataFrame, spacing: pd.DataFrame, n_merged: int,
                 null: dict | None = None) -> str:
    b, a = cmp.iloc[0], cmp.iloc[1]
    conf = trk["confidence"].value_counts().to_dict()
    lines = [
        "## Başlık 1: parlak özellik takibi (güçlendirilmiş, Türkçe özet)",
        "",
        f"- Güçlü tespitlerle iz sayısı {int(b.n_tracks)} idi, bunun {int(b.tracks_with_1_detection)} tanesi tek karedeydi. "
        f"Zayıf ama tutarlı tepeleri de izleyen histerezisli takip ve parça birleştirme ({n_merged} birleştirme) sonrası "
        f"iz sayısı {int(a.n_tracks)}, tek karelik iz {int(a.tracks_with_1_detection)}, en az {6} kare süren iz "
        f"{int(b.tracks_with_6plus_detections)} → {int(a.tracks_with_6plus_detections)}.",
        f"- İz güveni: yüksek {conf.get('high', 0)}, orta {conf.get('medium', 0)}, düşük {conf.get('low', 0)}. "
        "Zayıf tespitle uzatılan görünümler 'recovered' diye işaretli; resim şeridinde sarı kesikli daire.",
    ]
    if len(spacing):
        ok = spacing[spacing["n_frames"] >= 5]
        n_c = int((ok["verdict"] == "spacing_changes").sum())
        n_f = int(ok["verdict"].isin(["conflicting_measures", "undetermined"]).sum())
        n_k = len(ok) - n_c - n_f
        lines += [
            f"- Özellikler arası DNA boyunca mesafe: {len(ok)} çift en az 5 karede birlikte görünüyor; "
            f"{n_k} çiftte mesafe sabit, {n_c} çiftte iki ölçü de değişim gösteriyor, {n_f} çiftte ölçüler çelişiyor veya belirsiz. "
            "İki ölçü birleştirildi: hizalı karelerde (≈1–22) referans kontur üzerindeki konum farkı (rota sabit), yeterli kare yoksa her karenin kendi iskeletindeki en kısa yol. "
            "Yol değişimi: halka biçimli molekülde halka açılıp kapanınca en kısa yol ters yönden dolanır ve mesafe yüzlerce piksel zıplar; "
            "bu özelliklerin kayması değildir, kayma sayılmadı. Ölçüler çelişirse karar verilmez, çift 'çelişkili' kalır. Gerçek değişim için hem en az 8 px hem ortalama mesafenin %10'u kadar değişim ve "
            "doğrusal eğim p<0.01 gerekir. Bu ölçü genel hizalamadan bağımsızdır."]
    if null:
        lines += [f"- Kontrol: DNA üzerinde rastgele {null['n_fake']} noktadan başlatılan sahte izler aynı kuralla "
                  f"ortalama {null['fake_mean_recovered']:.1f} zayıf tepe topladı; gerçek izler ortalama "
                  f"{null['real_mean_recovered']:.1f}. En az 5 zayıf tepe toplayan iz oranı: sahte "
                  f"%{100 * null['fake_share_with_6plus']:.0f}, gerçek %{100 * null['real_share_with_6plus']:.0f}. "
                  "Fark büyükse zayıf tepeler DNA'nın her yerinde değil, gerçek özelliklerin çevresindedir."]
    lines += ["- Özelliklerin kimliği hâlâ bilinmiyor; hepsi 'parlak özellik (aday bağlı parçacık)'. "
              "Her izin fotoğraf şeridi fig_track_montage'da, gözle kontrol için."]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ run

def run(outputs: Path, data_dir: Path, config: Path = cfgmod.DEFAULT_CONFIG_PATH,
        h: HysteresisSettings | None = None) -> dict:
    h = h or HysteresisSettings()
    p2, p6 = outputs / "phase2", outputs / "phase6"
    for f in (p2 / "frames.csv", p6 / "feature_detections.csv", p6 / "feature_tracks.csv",
              p6 / "contour_alignment.csv", outputs / "phase3" / "registration.csv"):
        if not f.is_file():
            raise FileNotFoundError(f"{f} not found. Run phases 2 to 6 first (run_all.sh).")
    frames = pd.read_csv(p2 / "frames.csv").sort_values("rank")
    reg = pd.read_csv(outputs / "phase3" / "registration.csv")
    shifts = cumulative_shifts(reg)
    reliable = {int(r.rank_to): bool(r.reliable) for r in reg.itertuples()}
    dna = load_dna_settings(config)
    cache = load_frame_cache(data_dir, outputs, frames, dna)
    det0 = pd.read_csv(p6 / "feature_detections.csv")
    trk0 = pd.read_csv(p6 / "feature_tracks.csv")

    det = extend_tracks(det0, cache, shifts, h)
    null = null_extension(det, cache, shifts, h)
    det, n_merged = merge_fragments(det, h)
    fs = FeatureSettings()
    contour = reference_contour(p2, frames, 1)
    delta = pd.read_csv(p6 / "contour_alignment.csv")
    det = add_contour_position(det, contour, fs, delta)
    last_rank = int(frames["rank"].max())
    trk = summarise_tracks(det, last_rank, reliable, fs)
    trk = track_confidence(trk, det, h)
    cmp = comparison_table(trk0, trk, h)
    aligned = set(int(r) for r, v in zip(delta['rank'], delta['contour_matched_fraction_aligned'] >= 0.90) if v)
    series, summ = pair_spacing(det, trk, cache, shifts, h, aligned)

    out = outputs / "phase6b"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["track_id", "rank", "filename", "recovered", "x_common", "y_common", "s_px", "dist_to_ref_contour_px",
            "off_reference_contour", "area_px", "relative_to_ridge_level", "radius_px", "peak_z", "label"]
    det[cols].round(2).to_csv(out / "feature_detections_v2.csv", index=False)
    trk.to_csv(out / "feature_tracks_v2.csv", index=False)
    cmp.to_csv(out / "track_comparison.csv", index=False)
    series.to_csv(out / "pair_spacing_series.csv", index=False)
    summ.to_csv(out / "pair_spacing_summary.csv", index=False)
    (out / "summary_tr.md").write_text(summary_text(cmp, trk, summ, n_merged, null), encoding="utf-8")
    (out / "parameters.json").write_text(json.dumps({
        "hysteresis_settings": asdict(h), "merged_fragments": n_merged, "null_control": null,
        "notes": "features unidentified; recovered detections are weak peaks following a strong seed; "
                 "spacing is shortest path along the skeleton; pixel units only"}, indent=2), encoding="utf-8")
    montage(det, trk, cache, shifts, h, out)
    spacing_figure(series, summ, out)
    return {"null": null, "detections": det, "tracks": trk, "comparison": cmp, "spacing": summ, "n_merged": n_merged}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 6b: stronger bright-feature tracking.")
    p.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--config", type=Path, default=cfgmod.DEFAULT_CONFIG_PATH)
    a = p.parse_args(argv)
    try:
        res = run(a.outputs, a.data_dir, a.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(res["comparison"].to_string(index=False))
    print("null:", res["null"])
    print("confidence:", res["tracks"]["confidence"].value_counts().to_dict(), " merged fragments:", res["n_merged"])
    print(res["tracks"]["motion_class"].value_counts().to_string())
    sp = res["spacing"]
    if len(sp):
        print(sp["verdict"].value_counts().to_string())
    print(f"Results: {a.outputs / 'phase6b'}")
    print("Features are unidentified; recovered detections are weak peaks; nothing here confirms a mechanism.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
