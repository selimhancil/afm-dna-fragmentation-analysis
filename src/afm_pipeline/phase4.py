"""Phase 4: quantitative measurements, bright-feature analysis, time series, sensitivity.

Everything is in pixels (JPG intensity). nm columns exist only as ESTIMATES that carry the
scale status; time is the frame index (provisional order) plus an earliest/latest range.

Measurements per frame (frame_measurements.csv)
-----------------------------------------------
n_objects_tracked          objects with skeleton >= track_min_length_px (listed pieces)
largest_object_px          skeleton length of the longest piece
total_skeleton_px          skeleton length of all tracked pieces
continuity_fraction        largest piece / total  (1.0 = one single piece; falls as it breaks up)
n_closed_loop_candidates   pieces whose mask encloses a ring-sized hole
n_loop_if_gaps_bridged     pieces that would become a loop if a small gap were bridged
n_bright_features, mean_bright_area_px, bright features kept at the same place frame to frame
split events that reach this frame (candidates only)

Sensitivity analysis (sensitivity.csv, sensitivity_summary.csv)
---------------------------------------------------------------
No minimum fragment length is established, so the same measurements are repeated over a grid
of settings (edge threshold, bridging radius, minimum piece length). A trend is called robust
only if its direction (Spearman correlation with frame index) is the same in every setting.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config as cfgmod
from .dna import DnaSettings, analyse, dna_mask, load_dna_settings, preprocess
from .inventory import DEFAULT_DATA_DIR, PROJECT_ROOT, find_image_files
from .order import build_provisional_order
from .segment import load_rgb
from .tracking import TrackSettings, load_track_settings

DEFAULT_OUT = PROJECT_ROOT / "outputs" / "phase4"
GRID = {"low_k": [3.0, 3.5, 4.5], "bridge_radius_px": [1, 2, 3], "min_len": [20, 40, 60, 100]}


# ------------------------------------------------------------------ per-frame table

def cumulative_shifts(registration: pd.DataFrame) -> dict[int, tuple[float, float]]:
    """Shift (dy, dx) that moves each frame onto frame 1, from the pairwise registrations."""
    out = {int(registration["rank_from"].iloc[0]): (0.0, 0.0)}
    dy = dx = 0.0
    for r in registration.itertuples():
        dy, dx = dy + r.dy, dx + r.dx
        out[int(r.rank_to)] = (dy, dx)
    return out


def bright_persistence(bright: pd.DataFrame, frames: pd.DataFrame, registration: pd.DataFrame,
                       radius_px: float = 6.0) -> pd.DataFrame:
    """Share of each frame's bright features found again (within radius) in the previous frame,
    after registration. Positions are put in common (frame 1) coordinates first."""
    shifts = cumulative_shifts(registration)
    crop = frames.set_index("rank")[["crop_top", "crop_left"]]
    common = {}
    for rank, g in bright.groupby("rank"):
        dy, dx = shifts.get(int(rank), (0.0, 0.0))
        top, left = crop.loc[rank, "crop_top"], crop.loc[rank, "crop_left"]
        common[int(rank)] = np.column_stack([g["centroid_y_px"] + top + dy, g["centroid_x_px"] + left + dx])
    rows = []
    for rank in sorted(frames["rank"]):
        cur, prev = common.get(rank), common.get(rank - 1)
        if cur is None or len(cur) == 0 or prev is None or len(prev) == 0:
            rows.append({"rank": rank, "bright_kept_from_previous": np.nan})
            continue
        d = np.sqrt(((cur[:, None, :] - prev[None, :, :]) ** 2).sum(-1)).min(axis=1)
        rows.append({"rank": rank, "bright_kept_from_previous": round(float((d <= radius_px).mean()), 3)})
    return pd.DataFrame(rows)


def frame_measurements(frames: pd.DataFrame, objects: pd.DataFrame, bright: pd.DataFrame,
                       events: pd.DataFrame, registration: pd.DataFrame, ts: TrackSettings) -> pd.DataFrame:
    tracked = objects[objects["skeleton_length_px"] >= ts.track_min_length_px]
    rows = []
    splits = events[events["event"] == "split_candidate"] if len(events) else events
    for f in frames.sort_values("rank").itertuples():
        o = tracked[tracked["rank"] == f.rank]
        b = bright[bright["rank"] == f.rank] if len(bright) else bright
        total = float(o["skeleton_length_px"].sum())
        largest = float(o["skeleton_length_px"].max()) if len(o) else 0.0
        n_split = int((splits["rank_to"] == f.rank).sum()) if len(splits) else 0
        n_pers = int(((splits["rank_to"] == f.rank) & (splits["persistence"] == "persistent")).sum()) if len(splits) else 0
        rows.append({
            "rank": f.rank, "filename": f.filename,
            "time_earliest_s": f.time_earliest_s, "time_latest_s": f.time_latest_s, "time_status": f.time_status,
            "n_objects_tracked": len(o), "largest_object_px": round(largest, 1),
            "total_skeleton_px": round(total, 1),
            "continuity_fraction": round(largest / total, 3) if total else np.nan,
            "n_closed_loop_candidates": int((o["kind"] == "closed_loop_candidate").sum()),
            "n_loop_if_gaps_bridged": int(f.n_loop_if_gaps_bridged),
            "n_bright_features": len(b),
            "mean_bright_area_px": round(float(b["area_px"].mean()), 1) if len(b) else np.nan,
            "split_candidates_reaching_frame": n_split, "persistent_split_candidates": n_pers,
            "nm_per_pixel_used": f.nm_per_pixel_used, "scale_status": f.scale_status,
            "largest_object_nm_estimated": (None if pd.isna(f.nm_per_pixel_used)
                                            else round(largest * f.nm_per_pixel_used, 1)),
        })
    out = pd.DataFrame(rows)
    out["split_candidates_cumulative"] = out["split_candidates_reaching_frame"].cumsum()
    out["persistent_splits_cumulative"] = out["persistent_split_candidates"].cumsum()
    kept = bright_persistence(bright, frames, registration) if len(bright) else \
        pd.DataFrame({"rank": frames["rank"], "bright_kept_from_previous": np.nan})
    return out.merge(kept, on="rank", how="left")


def event_bright_association(events: pd.DataFrame, bright: pd.DataFrame) -> pd.DataFrame:
    """Distance from each split candidate's gap to the nearest bright feature (same frame, px)."""
    if events.empty or "event_x_px" not in events:
        return pd.DataFrame()
    rows = []
    for e in events[events["event"] == "split_candidate"].itertuples():
        g = bright[bright["rank"] == e.rank_from]
        if len(g) == 0 or pd.isna(e.event_x_px):
            dist = np.nan
        else:
            dist = float(np.hypot(g["centroid_x_px"] - e.event_x_px, g["centroid_y_px"] - e.event_y_px).min())
        rows.append({"rank_from": e.rank_from, "rank_to": e.rank_to, "persistence": e.persistence,
                     "min_gap_px": e.min_gap_px, "nearest_bright_feature_px": None if np.isnan(dist) else round(dist, 1),
                     "note": "association only; says nothing about what the bright feature is"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ sensitivity

def sensitivity(data_dir: Path, base: DnaSettings, grid: dict = GRID, progress=None) -> pd.DataFrame:
    files, _ = find_image_files(data_dir)
    order = build_provisional_order([f.name for f in files])
    by_name = {f.name: f for f in files}
    combos = list(itertools.product(grid["low_k"], grid["bridge_radius_px"]))
    rows = []
    for row in order.itertuples():
        rgb = load_rgb(by_name[row.filename])
        pre = preprocess(rgb, base)
        for low_k, bridge in combos:
            s = replace(base, low_k=low_k, bridge_radius_px=bridge, max_loop_bridge_px=0, min_dna_length_px=20.0)
            res = analyse(pre, s, dna_mask(pre.z, s))
            lens = np.array([o["skeleton_length_px"] for o in res.objects])
            for min_len in grid["min_len"]:
                keep = lens >= min_len
                total = float(lens[keep].sum())
                rows.append({
                    "setting": f"low_k={low_k};bridge={bridge};min_len={min_len}",
                    "low_k": low_k, "bridge_radius_px": bridge, "min_len_px": min_len,
                    "rank": int(row.chronological_rank), "filename": row.filename,
                    "n_pieces": int(keep.sum()),
                    "largest_px": round(float(lens[keep].max()), 1) if keep.any() else 0.0,
                    "total_px": round(total, 1),
                    "continuity_fraction": round(float(lens[keep].max()) / total, 3) if keep.any() and total else np.nan,
                    "n_closed_loops": sum(1 for o, k in zip(res.objects, keep)
                                          if k and o["kind"] == "closed_loop_candidate"),
                })
        if progress:
            progress(row.chronological_rank)
    return pd.DataFrame(rows)


def sensitivity_summary(sens: pd.DataFrame) -> pd.DataFrame:
    """Trend direction of each measurement vs frame index, for every setting."""
    out = []
    for setting, g in sens.groupby("setting"):
        row = {"setting": setting}
        for col in ("n_pieces", "largest_px", "continuity_fraction", "n_closed_loops"):
            ok = g[["rank", col]].dropna()
            rho = spearmanr(ok["rank"], ok[col])[0] if ok[col].nunique() > 1 else np.nan
            row[f"spearman_{col}"] = None if pd.isna(rho) else round(float(rho), 3)
        out.append(row)
    return pd.DataFrame(out)


def robustness_statement(summary: pd.DataFrame) -> dict:
    """For each measurement: do ALL settings agree on the direction of the trend?"""
    res = {}
    for col in [c for c in summary.columns if c.startswith("spearman_")]:
        v = summary[col].dropna()
        if v.empty:
            res[col] = "not computable"
        elif (v < -0.3).all():
            res[col] = f"robust decrease (rho {v.min():.2f} to {v.max():.2f} over {len(v)} settings)"
        elif (v > 0.3).all():
            res[col] = f"robust increase (rho {v.min():.2f} to {v.max():.2f} over {len(v)} settings)"
        else:
            res[col] = f"NOT robust (rho {v.min():.2f} to {v.max():.2f} over {len(v)} settings)"
    return res


# ------------------------------------------------------------------ runner

def run(data_dir: Path, phase2_dir: Path, phase3_dir: Path, out_dir: Path, dna: DnaSettings,
        ts: TrackSettings, do_sensitivity: bool = True) -> dict:
    for d in (phase2_dir, phase3_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"{d} not found. Run the earlier phases first.")
    frames = pd.read_csv(phase2_dir / "frames.csv")
    objects = pd.read_csv(phase2_dir / "objects.csv")
    bright = pd.read_csv(phase2_dir / "bright_features.csv")
    events = pd.read_csv(phase3_dir / "events.csv")
    registration = pd.read_csv(phase3_dir / "registration.csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    fm = frame_measurements(frames, objects, bright, events, registration, ts)
    fm.to_csv(out_dir / "frame_measurements.csv", index=False)
    assoc = event_bright_association(events, bright)
    assoc.to_csv(out_dir / "event_bright_association.csv", index=False)
    result = {"frame_measurements": fm, "association": assoc}
    if do_sensitivity:
        sens = sensitivity(data_dir, dna)
        sens.to_csv(out_dir / "sensitivity.csv", index=False)
        summ = sensitivity_summary(sens)
        summ.to_csv(out_dir / "sensitivity_summary.csv", index=False)
        stmt = robustness_statement(summ)
        (out_dir / "robustness.json").write_text(json.dumps(stmt, indent=2), encoding="utf-8")
        result.update({"sensitivity": sens, "summary": summ, "robustness": stmt})
    (out_dir / "parameters.json").write_text(json.dumps(
        {"dna": asdict(dna), "tracking": asdict(ts), "sensitivity_grid": GRID,
         "notes": "pixel units; nm only as estimate; time = frame index (provisional order)"},
        indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 4: measurements, time series data, sensitivity analysis.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--phase2-dir", type=Path, default=PROJECT_ROOT / "outputs" / "phase2")
    p.add_argument("--phase3-dir", type=Path, default=PROJECT_ROOT / "outputs" / "phase3")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--config", type=Path, default=cfgmod.DEFAULT_CONFIG_PATH)
    p.add_argument("--skip-sensitivity", action="store_true")
    a = p.parse_args(argv)
    try:
        out = run(a.data_dir, a.phase2_dir, a.phase3_dir, a.out_dir, load_dna_settings(a.config),
                  load_track_settings(a.config), not a.skip_sensitivity)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    fm = out["frame_measurements"]
    print(f"Frames measured: {len(fm)}")
    print(f"Largest piece: {fm.largest_object_px.iloc[0]:.0f} px (first) -> {fm.largest_object_px.iloc[-1]:.0f} px (last)")
    for k, v in out.get("robustness", {}).items():
        print(f"  {k.replace('spearman_', '')}: {v}")
    print(f"Results: {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
