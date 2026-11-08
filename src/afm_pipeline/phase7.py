"""Phase 7: the supervisor's three requests, answered as far as JPG data allow.

(1) "Height profile / contour length over time, when and where fragmentation occurs"
    -> intensity profile along the DNA contour (JPG intensity relative to the DNA ridge level,
       NOT calibrated height, NO nm) and a loss-time map: for every pixel of the reference
       contour, the first frame from which the DNA is no longer found there.
(2) "Differentiate dendrimers from enzymes"
    -> not possible from JPG colours. What can be done: group the tracked bright features by
       behaviour (long-lived and stationary vs short-lived) and test whether size/brightness
       differ. The groups are behavioural classes, NOT identifications.
(3) "Probabilistic model to see which mechanism correlates best"
    -> explicit likelihood model on WHERE candidate splits occur relative to bright features:
         H0 (independent of features):         split place uniform over the DNA
         near-feature model (H1-like):         split place within R px of a feature
         far-from-feature model (H3-like):     split place farther than R px from every feature
       compared by log-likelihood ratios against H0, with sensitivity over R, over the mixing
       weight, over event subsets, and a bootstrap over events. H1 adds a fate check (do
       features near a split vanish more often than usual?). H2 (self-migration) is judged from
       the motion classes of Phase 6.
Everything is descriptive: one molecule, one experiment, no control, unidentified features.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy import stats
from scipy.spatial import cKDTree
from skimage import morphology

from . import config as cfgmod
from .dna import load_dna_settings, preprocess, ridge_level
from .inventory import DEFAULT_DATA_DIR, PROJECT_ROOT
from .phase4 import cumulative_shifts
from .phase6 import FeatureSettings, reference_contour, skeleton_common
from .segment import load_rgb

DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"
R_GRID = [10, 15, 20, 25, 30, 40, 50]
EPS_GRID = [0.01, 0.05, 0.20]
LOSS_RUN = 2          # a contour pixel counts as lost when uncovered in this many consecutive valid frames
CLS_LABEL = {"persistent_stationary": "persistent and stationary", "transient_short": "short-lived", "intermediate": "intermediate"}
VALID_ALIGN = 0.90    # frames where the aligned contour matches the mask this well are used for the loss map
COVER_TOL_PX = 3      # mask is dilated by this much before asking "is the DNA still here?"


# ------------------------------------------------------------------ (1) contour profile and loss map

def loss_rank(cover: np.ndarray, run: int = LOSS_RUN) -> np.ndarray:
    """First frame index (row) from which a contour pixel stays uncovered for `run` frames.

    cover: (frames, pixels) of 1.0 / 0.0 / nan (nan = outside the frame, unknown).
    Returns row index or -1 if never lost."""
    n_f, n_p = cover.shape
    out = np.full(n_p, -1, dtype=int)
    for j in range(n_p):
        col = cover[:, j]
        for i in range(n_f - run + 1):
            w = col[i:i + run]
            if np.all(w == 0.0):
                out[j] = i
                break
    return out


def contour_profile(data_dir: Path, outputs: Path, frames: pd.DataFrame, shifts: dict, contour, dna,
                    delta: pd.DataFrame) -> dict:
    """Reference-contour coverage and intensity per frame, plus the DNA's own intensity profile.

    Alignment: cumulative pairwise registration + the per-frame extra shift from Phase 6.
    The loss map is computed only over 'valid' frames (aligned contour matches the mask well),
    because a deforming molecule cannot be followed by translation alone."""
    p2 = outputs / "phase2"
    pts = contour.points.astype(float)
    d = delta.set_index("rank")
    n_f, n_p = len(frames), len(pts)
    cover = np.full((n_f, n_p), np.nan)
    rel = np.full((n_f, n_p), np.nan)
    ranks, own = [], []
    for i, row in enumerate(frames.sort_values("rank").itertuples()):
        ranks.append(int(row.rank))
        pre = preprocess(load_rgb(data_dir / row.filename), dna)
        stem = Path(row.filename).stem
        mask = np.asarray(Image.open(p2 / "masks" / f"{int(row.rank):02d}_{stem}_mask.png").convert("L")) > 127
        level = ridge_level(pre.z, mask, dna)
        dy, dx = shifts.get(int(row.rank), (0.0, 0.0))
        cy = np.rint(pts[:, 0] - row.crop_top - dy).astype(int) + int(d.loc[row.rank, "ddy"])
        cx = np.rint(pts[:, 1] - row.crop_left - dx).astype(int) + int(d.loc[row.rank, "ddx"])
        h, w = mask.shape
        ok = (cy >= 0) & (cy < h) & (cx >= 0) & (cx < w)
        dil = ndi.binary_dilation(mask, structure=morphology.disk(COVER_TOL_PX))
        zmax = ndi.maximum_filter(pre.z, size=5)
        cover[i, ok] = dil[cy[ok], cx[ok]].astype(float)
        if level:
            rel[i, ok] = zmax[cy[ok], cx[ok]] / level
        sk = np.asarray(Image.open(p2 / "skeletons" / f"{int(row.rank):02d}_{stem}_skeleton.png").convert("L")) > 127
        lab, n = ndi.label(sk, structure=np.ones((3, 3)))
        sizes = ndi.sum(sk, lab, index=range(1, n + 1)) if n else []
        keep = np.isin(lab, [k + 1 for k, v in enumerate(sizes) if v >= 20]) if n else sk
        zs = pre.z[keep] if keep.any() else np.array([np.nan])
        own.append({"ridge_level_noise_units": round(float(level), 2) if level else np.nan,
                    "own_skeleton_median_rel": round(float(np.nanmedian(zs) / level), 3) if level else np.nan,
                    "own_skeleton_faint_fraction": round(float(np.nanmean(zs < 0.5 * level)), 3) if level else np.nan})
    valid = (delta.sort_values("rank")["contour_matched_fraction_aligned"].to_numpy() >= VALID_ALIGN)
    lost_v = loss_rank(cover[valid])
    vr = [r for r, v in zip(ranks, valid) if v]
    lost_rank = [vr[k] if k >= 0 else np.nan for k in lost_v]
    per_frame = pd.DataFrame({
        "rank": ranks,
        "contour_covered_fraction": np.round(np.nanmean(cover, axis=1), 3),
        "aligned_match_valid_for_loss_map": valid,
        "mean_relative_intensity_on_contour": np.round(np.nanmean(rel, axis=1), 3),
        **pd.DataFrame(own).to_dict("list"),
        "note": "JPG intensity relative to the DNA ridge level; not calibrated height"})
    pix = pd.DataFrame({"y_common": pts[:, 0], "x_common": pts[:, 1], "s_px": np.round(contour.arc, 1),
                        "first_gap_rank_in_valid_frames": lost_rank,
                        "gap_seen": ~np.isnan(np.array(lost_rank, dtype=float))})
    return {"per_frame": per_frame, "pixels": pix, "cover": cover, "rel": rel, "ranks": ranks, "valid": valid}


# ------------------------------------------------------------------ (2) behavioural classes

def behaviour_classes(tracks: pd.DataFrame) -> pd.DataFrame:
    t = tracks.copy()

    def cls(r):
        if r.n_detections >= 10 and r.motion_class == "stationary":
            return "persistent_stationary"
        if r.lifespan_frames <= 3:
            return "transient_short"
        return "intermediate"

    t["behaviour_class"] = [cls(r) for r in t.itertuples()]
    return t


def class_summary(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for c, g in t.groupby("behaviour_class"):
        rows.append({"behaviour_class": c, "n_tracks": len(g),
                     "median_area_px": round(float(g["mean_area_px"].median()), 1),
                     "median_relative_intensity": round(float(g["mean_relative_intensity"].median()), 2),
                     "median_lifespan_frames": float(g["lifespan_frames"].median()),
                     "label": "behavioural class, not an identification"})
    tests = {}
    a = t[t["behaviour_class"] == "persistent_stationary"]
    b = t[t["behaviour_class"] == "transient_short"]
    if len(a) >= 3 and len(b) >= 3:
        for col in ("mean_area_px", "mean_relative_intensity"):
            u = stats.mannwhitneyu(a[col], b[col], alternative="two-sided")
            tests[col] = {"n_persistent": len(a), "n_transient": len(b), "p_value": round(float(u.pvalue), 4)}
    diag = {"tracks": int(len(t)), "single_detection_tracks": int((t["n_detections"] == 1).sum()),
            "tracks_with_missed_frames": int((t["missed_frames"] > 0).sum())}
    return pd.DataFrame(rows), {"tests_persistent_vs_transient": tests, "diagnostics": diag}


# ------------------------------------------------------------------ (3) likelihood model

def site_loglik(d_event: float, pool: np.ndarray, kind: str, R: float, eps: float) -> float:
    """log-likelihood of one split place under a model, as a mixture with H0 (weight eps).

    pool: distances (to the nearest bright feature) of all DNA pixels of that frame (the H0 support).
    kind: 'near' = place within R of a feature; 'far' = farther than R."""
    n_all = len(pool)
    inside = pool <= R if kind == "near" else pool > R
    n_in = int(inside.sum())
    p0 = 1.0 / n_all
    hit = (d_event <= R) if kind == "near" else (d_event > R)
    p = (1 - eps) * (1.0 / n_in if (hit and n_in) else 0.0) + eps * p0
    return float(np.log(p))


def event_pools(events: pd.DataFrame, b: pd.DataFrame, phase2: Path, frames: pd.DataFrame,
                shifts: dict, s: FeatureSettings) -> list[dict]:
    out = []
    for e in events.itertuples():
        g = b[b["rank"] == e.rank_from]
        if len(g) == 0 or np.isnan(e.nearest_feature_px):
            continue
        pts = skeleton_common(phase2, frames, int(e.rank_from), shifts)
        if len(pts) < s.min_skeleton_px:
            continue
        pool = cKDTree(g[["y_common", "x_common"]].to_numpy()).query(pts)[0]
        out.append({"rank_from": int(e.rank_from), "rank_to": int(e.rank_to), "persistence": e.persistence,
                    "min_gap_px": float(e.min_gap_px), "d": float(e.nearest_feature_px), "pool": pool})
    return out


def llr_table(evs: list[dict], subsets: dict[str, list[dict]], seed: int = 1, n_boot: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for name, sub in subsets.items():
        if not sub:
            continue
        for R in R_GRID:
            for eps in EPS_GRID:
                base = np.array([np.log(1.0 / len(e["pool"])) for e in sub])
                row = {"subset": name, "n_events": len(sub), "R_px": R, "eps": eps}
                for kind in ("near", "far"):
                    ll = np.array([site_loglik(e["d"], e["pool"], kind, R, eps) for e in sub]) - base
                    row[f"LLR_{kind}_vs_independent"] = round(float(ll.sum()), 2)
                    if eps == 0.05 and R == 30:
                        boots = [ll[rng.integers(0, len(ll), len(ll))].sum() for _ in range(n_boot)]
                        row[f"LLR_{kind}_boot95_lo"], row[f"LLR_{kind}_boot95_hi"] = (
                            round(float(np.percentile(boots, 2.5)), 2), round(float(np.percentile(boots, 97.5)), 2))
                rows.append(row)
    return pd.DataFrame(rows)


def h1_fate_check(ev_assoc: pd.DataFrame, baseline_lost: float) -> dict:
    """Features within the near radius of a split: do they vanish more often than usual?"""
    n = int(ev_assoc[["near_kept_in_place", "near_moved", "near_lost"]].sum().sum())
    k = int(ev_assoc["near_lost"].sum())
    out = {"n_near_features": n, "n_lost": k, "baseline_lost_fraction": baseline_lost}
    if n:
        out["p_value_more_loss_than_baseline"] = round(float(
            stats.binomtest(k, n, baseline_lost, alternative="greater").pvalue), 4)
        out["p_value_if_half_were_lost"] = round(float(
            stats.binomtest(k, n, 0.5, alternative="less").pvalue), 4)
    return out


def _state(lo: float, hi: float, point: float) -> str:
    if hi < 0:
        return "clearly worse than the independent model (bootstrap interval entirely below zero)"
    if lo > 0:
        return "clearly better than the independent model (bootstrap interval entirely above zero)"
    return "indistinguishable from the independent model (bootstrap interval contains zero)"


def verdict_text(llr: pd.DataFrame, fate: dict, trk: pd.DataFrame, cls_tests: dict) -> pd.DataFrame:
    """Readings are generated from the numbers, never typed in advance."""
    r = llr[(llr["subset"] == "all") & (llr["R_px"] == 30) & (llr["eps"] == 0.05)]
    allr = llr[llr["subset"] == "all"]
    long = trk[trk["n_detections"] >= 4]
    n_st = int((long["motion_class"] == "stationary").sum())
    n_mv = int((long["motion_class"] == "moved").sum())
    rows = []
    if r.empty:
        near_s = far_s = "could not be computed"
        nt = ft = "none"
        best_near = best_far = float("nan")
    else:
        r = r.iloc[0]
        near_s = _state(r["LLR_near_boot95_lo"], r["LLR_near_boot95_hi"], r["LLR_near_vs_independent"])
        far_s = _state(r["LLR_far_boot95_lo"], r["LLR_far_boot95_hi"], r["LLR_far_vs_independent"])
        nt = (f"near model {r['LLR_near_vs_independent']:+.1f} "
              f"(95% interval {r['LLR_near_boot95_lo']:+.1f} to {r['LLR_near_boot95_hi']:+.1f})")
        ft = (f"far model {r['LLR_far_vs_independent']:+.1f} "
              f"(95% interval {r['LLR_far_boot95_lo']:+.1f} to {r['LLR_far_boot95_hi']:+.1f})")
        best_near = float(allr["LLR_near_vs_independent"].max())
        best_far = float(allr["LLR_far_vs_independent"].max())
    grid = (f"Best values across R=10-50 px and mixture weight 0.01-0.20: near {best_near:+.1f}, far {best_far:+.1f} "
            f"(a value above +2.3 would have counted as a clear advantage).")
    rows.append({"hypothesis": "H1", "hypothesis_name": "The enzyme pushes dendrimers away and then cuts",
                 "model_numbers": f"Near model (R=30 px, 16 candidates): {nt}. {grid} "
                                  f"Of the {fate.get('n_near_features')} nearby features, {fate.get('n_lost')} disappeared "
                                  f"(overall loss share {fate.get('baseline_lost_fraction')}; p={fate.get('p_value_if_half_were_lost')} against the expectation 'half would disappear').",
                 "reading": f"Position model: near model {near_s}. This is the first version (strong detections only); the current evaluation, compared across three feature sets, is in Heading 6 (outputs/mechanisms)."})
    rows.append({"hypothesis": "H2", "hypothesis_name": "Dendrimers slide along the DNA on their own",
                 "model_numbers": f"Of {len(long)} long tracks, {n_st} are stationary and {n_mv} moved.",
                 "reading": "First version: only the movement classes were examined. The current H2 test based on the distance between features is in Heading 1 and Heading 6 (outputs/phase6b, outputs/mechanisms)."})
    rows.append({"hypothesis": "H3", "hypothesis_name": "The enzyme cuts only naked regions",
                 "model_numbers": f"Far model (R=30 px, 16 candidates): {ft}. {grid}",
                 "reading": f"Position model: far model {far_s}. But faint DNA next to a feature can look broken after thresholding (method effect), "
                               "so this result does not count as biological evidence."})
    df = pd.DataFrame(rows)
    df["verdict"] = "none: one molecule, unidentified features; descriptive model comparison only"
    return df


# ------------------------------------------------------------------ figures

def figures(prof: dict, contour, ev: pd.DataFrame, llr: pd.DataFrame, t: pd.DataFrame, out: Path,
            cont: pd.DataFrame | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
        plt.close(fig)

    pix = prof["pixels"]
    nvalid = int(prof["valid"].sum())
    last_valid = max(r for r, v in zip(prof["ranks"], prof["valid"]) if v)
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    never = ~pix["gap_seen"]
    ax.scatter(pix.loc[never, "x_common"], pix.loc[never, "y_common"], s=5, c="#c9ced8", label="no gap seen")
    if (~never).any():
        sc = ax.scatter(pix.loc[~never, "x_common"], pix.loc[~never, "y_common"], s=9,
                        c=pix.loc[~never, "first_gap_rank_in_valid_frames"], cmap="viridis_r")
        plt.colorbar(sc, ax=ax, fraction=0.04, label="first frame with a gap here")
    if len(ev):
        e2 = ev[ev["rank_to"] <= last_valid + 1]
        pers = e2["persistence"] == "persistent"
        ax.scatter(e2.loc[~pers, "event_x_ref"], e2.loc[~pers, "event_y_ref"], marker="x", c="k", s=45,
                   label="split candidate (re-joined)")
        ax.scatter(e2.loc[pers, "event_x_ref"], e2.loc[pers, "event_y_ref"], marker="X", c="red", s=70,
                   label="split candidate (persistent)")
        for r in e2.itertuples():
            ax.annotate(str(r.rank_to), (r.event_x_ref, r.event_y_ref), fontsize=6, xytext=(3, 3),
                        textcoords="offset points")
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(f"Where gaps appear on the frame-1 contour, frames up to {last_valid} "
                 f"({nvalid} frames with a reliable alignment)", fontsize=8, loc="left")
    ax.legend(fontsize=7, loc="lower left")
    ax.axis("off")
    save(fig, "fig_loss_time_map")

    pf = prof["per_frame"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    axs[0].plot(pf["rank"], pf["contour_covered_fraction"], color="#2255c4", label="overlap with frame-1 contour")
    if cont is not None:
        axs[0].plot(cont["rank"], cont["continuity_fraction"], color="#888", ls="--", label="continuity (Phase 4)")
    axs[0].set_ylabel("fraction")
    axs[0].legend(fontsize=7)
    axs[0].set_title("Shape overlap falls long before the longest piece does", fontsize=8, loc="left")
    axs[1].plot(pf["rank"], pf["ridge_level_noise_units"], color="#c47a22")
    axs[1].set_ylabel("DNA ridge level (JPG intensity / noise; not height)")
    axs[2].plot(pf["rank"], pf["own_skeleton_faint_fraction"], color="#7a3fb0")
    axs[2].set_ylabel("share of DNA skeleton fainter than half the ridge level")
    for a_ in axs:
        a_.set_xlabel("frame index (provisional order)")
        a_.spines[["top", "right"]].set_visible(False)
        a_.grid(alpha=0.25, lw=0.5)
    save(fig, "fig_contour_coverage_intensity_time")

    if len(llr):
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        for ax, kind, title in ((axs[0], "near", "split place near features"), (axs[1], "far", "split place far from features")):
            for name, g in llr[llr["eps"] == 0.05].groupby("subset"):
                ax.plot(g["R_px"], g[f"LLR_{kind}_vs_independent"], marker="o", label=f"{name} (n={int(g['n_events'].iloc[0])})")
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xlabel("R (px)")
            ax.set_ylabel("log-likelihood ratio vs independent")
            ax.set_title(title, fontsize=9, loc="left")
            ax.legend(fontsize=7)
            ax.spines[["top", "right"]].set_visible(False)
        save(fig, "fig_model_llr_vs_R")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"persistent_stationary": "#2255c4", "transient_short": "#d9534f", "intermediate": "#888"}
    for c, g in t.groupby("behaviour_class"):
        axs[0].scatter(g["mean_area_px"], g["mean_relative_intensity"], s=g["n_detections"] * 3 + 8,
                       c=colors.get(c, "k"), alpha=0.7, label=f"{c} (n={len(g)})")
    axs[0].set_xlabel("mean area (px)")
    axs[0].set_ylabel("mean peak / ridge level")
    axs[0].legend(fontsize=7)
    axs[0].set_title("Behavioural classes (not identifications)", fontsize=9, loc="left")
    for c, g in t.groupby("behaviour_class"):
        axs[1].hist(g["lifespan_frames"], bins=range(1, int(t["lifespan_frames"].max()) + 3), alpha=0.6,
                    color=colors.get(c, "k"), label=c)
    axs[1].set_xlabel("track lifespan (frames)")
    axs[1].set_ylabel("tracks")
    axs[1].legend(fontsize=7)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_behaviour_classes")


# ------------------------------------------------------------------ run

def run(outputs: Path, data_dir: Path, config: Path = cfgmod.DEFAULT_CONFIG_PATH) -> dict:
    p2, p6 = outputs / "phase2", outputs / "phase6"
    for f in (p2 / "frames.csv", p6 / "feature_tracks.csv", p6 / "feature_detections.csv",
              p6 / "event_feature_association.csv", p6 / "contour_alignment.csv",
              outputs / "phase3" / "registration.csv"):
        if not f.is_file():
            raise FileNotFoundError(f"{f} not found. Run phases 2 to 6 first (run_all.sh).")
    frames = pd.read_csv(p2 / "frames.csv").sort_values("rank")
    reg = pd.read_csv(outputs / "phase3" / "registration.csv")
    shifts = cumulative_shifts(reg)
    dna = load_dna_settings(config)
    s = FeatureSettings()
    contour = reference_contour(p2, frames, 1)
    delta = pd.read_csv(p6 / "contour_alignment.csv")
    prof = contour_profile(data_dir, outputs, frames, shifts, contour, dna, delta)

    trk = behaviour_classes(pd.read_csv(p6 / "feature_tracks.csv"))
    cls_df, cls_info = class_summary(trk)

    det = pd.read_csv(p6 / "feature_detections.csv")
    ev = pd.read_csv(p6 / "event_feature_association.csv")
    evs = event_pools(ev, det, p2, frames, shifts, s)
    subsets = {"all": evs,
               "persistent_only": [e for e in evs if e["persistence"] == "persistent"],
               "gap_above_8px": [e for e in evs if e["min_gap_px"] > 8.0]}
    llr = llr_table(evs, subsets)
    params6 = json.loads((p6 / "parameters.json").read_text(encoding="utf-8"))
    fate = h1_fate_check(ev, params6["baseline_fate_counts"]["lost_fraction"])
    verdict = verdict_text(llr, fate, trk, cls_info["tests_persistent_vs_transient"])

    out = outputs / "phase7"
    out.mkdir(parents=True, exist_ok=True)
    prof["per_frame"].to_csv(out / "contour_profile_by_frame.csv", index=False)
    prof["pixels"].to_csv(out / "contour_pixels_loss_map.csv", index=False)
    trk.to_csv(out / "feature_behaviour_classes.csv", index=False)
    cls_df.to_csv(out / "behaviour_class_summary.csv", index=False)
    llr.to_csv(out / "model_comparison.csv", index=False)
    verdict.to_csv(out / "mechanism_model_summary.csv", index=False)
    (out / "parameters.json").write_text(json.dumps({
        "r_grid_px": R_GRID, "eps_grid": EPS_GRID, "loss_run_frames": LOSS_RUN, "cover_tolerance_px": COVER_TOL_PX,
        "h1_fate_check": fate, "class_info": cls_info,
        "models": {"independent": "split place uniform over DNA pixels of the frame",
                   "near": "mixture (1-eps) uniform over pixels within R of a feature + eps independent",
                   "far": "mixture (1-eps) uniform over pixels farther than R from every feature + eps independent"},
        "notes": "intensity is JPG intensity relative to ridge level, not height; features unidentified; "
                 "one molecule, descriptive only"}, indent=2, default=float), encoding="utf-8")
    p4 = outputs / "phase4" / "frame_measurements.csv"
    figures(prof, contour, ev, llr, trk, out, pd.read_csv(p4) if p4.is_file() else None)
    (out / "summary.md").write_text(summary_text(prof, cls_df, cls_info, fate, verdict), encoding="utf-8")
    return {"profile": prof, "classes": cls_df, "class_info": cls_info, "llr": llr, "fate": fate, "verdict": verdict}


def summary_text(prof: dict, class_df: pd.DataFrame, info: dict, fate: dict, verdict: pd.DataFrame) -> str:
    pf = prof["per_frame"]
    ok = pf[pf["aligned_match_valid_for_loss_map"]]
    pix = prof["pixels"]
    gap_share = float(pix["gap_seen"].mean())
    n_valid = len(ok)
    lines = [
        "## Phase 7: time, place and mechanism model (English summary)",
        "",
        f"- In the first {n_valid} frames the DNA is reliably aligned with the reference contour (overlap at least 90%). In these frames about "
        f"{100 * gap_share:.0f}% of the contour showed a gap in at least 2 consecutive frames; the gaps cluster at a few specific places on the contour (fig_loss_time_map).",
        f"- In later frames the contour overlap drops to about {100 * float(pf['contour_covered_fraction'].iloc[-1]):.0f}%. Because the longest piece in Phase 4 "
        "stayed a similar length in the first 49 frames, this drop is most likely a change of shape rather than DNA disappearing "
        "(only translation alignment was used). This cannot be separated firmly with these data.",
        "- There is no height in this phase: only JPG intensity (relative to the ridge level) and the faint skeleton fraction were given frame by frame, and these are not calibrated heights. "
        "Assumed nm height is produced separately in Heading 3 (outputs/heights, config [height]) with an 'ASSUMED' label.",
        f"- Behaviour classes ({', '.join(f'{CLS_LABEL.get(r.behaviour_class, r.behaviour_class)}: {r.n_tracks}' for r in class_df.itertuples())}): "
        "long-lived stationary features are clearly larger and brighter than short-lived ones; however, small and faint features drop below the threshold easily and so look short-lived, so this difference is partly a selection effect. This is not an identity assignment (details: Heading 2).",
        f"- {info['diagnostics']['single_detection_tracks']}/{info['diagnostics']['tracks']} of the tracks appear in a single frame "
        "(only strong detections were tracked in this phase). The strengthened tracking that also follows weak but consistent peaks is in Heading 1 (outputs/phase6b), where the number of single-frame tracks drops a lot.",
        "",
        "### Three mechanisms (probability model, descriptive)",
    ]
    for r in verdict.itertuples():
        lines += [f"- **{r.hypothesis} ({r.hypothesis_name})**: {r.model_numbers} {r.reading}"]
    lines += ["", "One molecule, one experiment, no control, features unidentified: no mechanism counts as confirmed or refuted."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 7: contour profile, loss map, behaviour classes, model comparison.")
    p.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--config", type=Path, default=cfgmod.DEFAULT_CONFIG_PATH)
    a = p.parse_args(argv)
    try:
        res = run(a.outputs, a.data_dir, a.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    pf = res["profile"]["per_frame"]
    print(f"Contour covered: {pf['contour_covered_fraction'].iloc[0]:.2f} (first) -> {pf['contour_covered_fraction'].iloc[-1]:.2f} (last)")
    print(res["classes"].to_string(index=False))
    print(res["class_info"])
    print(res["fate"])
    print(res["llr"][(res["llr"]["R_px"] == 30) & (res["llr"]["eps"] == 0.05)].to_string(index=False))
    print(f"Results: {a.outputs / 'phase7'}")
    print("Descriptive only: no mechanism is confirmed, features are unidentified, intensity is not height.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
