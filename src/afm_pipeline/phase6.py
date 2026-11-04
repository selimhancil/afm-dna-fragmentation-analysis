"""Phase 6: bright-feature tracking along the DNA contour and mechanism-oriented summary.

Question from the supervisor: do the bright features (dendrimers and/or enzymes, identity NOT
known) move, grow, fade or vanish, and does that relate to where candidate cuts appear?
Three mechanisms were proposed:
  H1  the enzyme actively pushes dendrimers away and then cuts,
  H2  dendrimers migrate along the DNA on their own,
  H3  the enzyme only cleaves the remaining naked segments.

What this module does (pixels and JPG intensity only, never nm heights):
  * puts every bright feature of every frame into common (frame 1) coordinates,
  * links features frame to frame (Hungarian matching, small gap closing) into tracks,
  * gives each detection a position s along a reference contour (shortest path along the
    skeleton of frame 1's longest piece, from its topmost point),
  * summarises each track (lifetime, motion, size, relative intensity, fate),
  * compares the place of each candidate split with random places on the DNA (null),
  * writes a table of what each mechanism would predict and what was observed.
Nothing here identifies a feature as enzyme or dendrimer, and nothing confirms a cut.
With one molecule in one experiment, all statistics are descriptive.
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
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.graph import MCP_Geometric

from .inventory import PROJECT_ROOT
from .phase4 import cumulative_shifts

DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"


@dataclass
class FeatureSettings:
    link_radius_px: float = 8.0          # max move between consecutive frames (after registration)
    gap_link_radius_px: float = 10.0     # max move when a feature was missed for 1+ frames
    max_gap_frames: int = 2              # frames a feature may be missed and still be re-linked
    area_ratio_max: float = 3.0          # reject links whose area differs by more than this factor
    stationary_px: float = 5.0           # max distance from median position to be called stationary
    moved_px: float = 10.0               # net displacement above this (with enough detections) = moved
    min_track_len: int = 4               # detections needed to classify motion
    near_event_px: float = 30.0          # features within this distance of a split gap are "near"
    window_frames: int = 3               # frames after an event in which the fate is judged
    keep_radius_px: float = 6.0          # same place = within this many px
    off_contour_px: float = 12.0         # farther than this from the reference contour = off contour
    min_skeleton_px: int = 20            # skeleton pieces shorter than this are ignored for the null
    n_null: int = 5000                   # random draws for the null comparison
    seed: int = 1


# ------------------------------------------------------------------ loading

def load_inputs(outputs: Path) -> dict:
    p2, p3 = outputs / "phase2", outputs / "phase3"
    need = [p2 / "frames.csv", p2 / "bright_features.csv", p3 / "registration.csv", p3 / "events.csv"]
    for f in need:
        if not f.is_file():
            raise FileNotFoundError(f"{f} not found. Run phases 2 and 3 first (run_all.sh).")
    return {"frames": pd.read_csv(p2 / "frames.csv").sort_values("rank"),
            "bright": pd.read_csv(p2 / "bright_features.csv"),
            "registration": pd.read_csv(p3 / "registration.csv"),
            "events": pd.read_csv(p3 / "events.csv"), "phase2": p2}


def common_positions(bright: pd.DataFrame, frames: pd.DataFrame, registration: pd.DataFrame) -> pd.DataFrame:
    """Add y_common/x_common: position in frame-1 canvas coordinates."""
    shifts = cumulative_shifts(registration)
    crop = frames.set_index("rank")[["crop_top", "crop_left"]]
    b = bright.copy()
    top = b["rank"].map(crop["crop_top"]).to_numpy(float)
    left = b["rank"].map(crop["crop_left"]).to_numpy(float)
    dy = b["rank"].map(lambda r: shifts.get(int(r), (0.0, 0.0))[0]).to_numpy(float)
    dx = b["rank"].map(lambda r: shifts.get(int(r), (0.0, 0.0))[1]).to_numpy(float)
    b["y_common"] = b["centroid_y_px"].to_numpy(float) + top + dy
    b["x_common"] = b["centroid_x_px"].to_numpy(float) + left + dx
    return b


# ------------------------------------------------------------------ linking

def link_features(b: pd.DataFrame, s: FeatureSettings) -> pd.DataFrame:
    """Assign track_id to every detection (frame-by-frame Hungarian matching with gap closing)."""
    b = b.sort_values(["rank", "feature_id"]).reset_index(drop=True)
    b["track_id"] = 0
    b["gap_before"] = 0
    tracks: dict[int, dict] = {}     # track_id -> last detection info
    next_id = 1
    for rank, g in b.groupby("rank", sort=True):
        idx = g.index.to_list()
        active = [t for t, info in tracks.items() if 0 < rank - info["rank"] <= s.max_gap_frames + 1]
        assigned = set()
        if active and idx:
            cost = np.full((len(active), len(idx)), 1e6)
            for i, t in enumerate(active):
                info = tracks[t]
                gap = rank - info["rank"] - 1
                radius = s.link_radius_px if gap == 0 else s.gap_link_radius_px
                for j, k in enumerate(idx):
                    d = float(np.hypot(b.at[k, "y_common"] - info["y"], b.at[k, "x_common"] - info["x"]))
                    ratio = max(b.at[k, "area_px"], info["area"]) / max(min(b.at[k, "area_px"], info["area"]), 1)
                    if d <= radius and ratio <= s.area_ratio_max:
                        cost[i, j] = d + 0.5 * gap
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] < 1e5:
                    t, k = active[i], idx[j]
                    b.at[k, "track_id"] = t
                    b.at[k, "gap_before"] = rank - tracks[t]["rank"] - 1
                    assigned.add(k)
        for k in idx:
            if k not in assigned:
                b.at[k, "track_id"] = next_id
                next_id += 1
            t = int(b.at[k, "track_id"])
            tracks[t] = {"rank": rank, "y": b.at[k, "y_common"], "x": b.at[k, "x_common"],
                         "area": b.at[k, "area_px"]}
    return b


# ------------------------------------------------------------------ reference contour

def skeleton_common(phase2: Path, frames: pd.DataFrame, rank: int, shifts: dict) -> np.ndarray:
    """Skeleton pixels of one frame as an (n, 2) array of (y, x) in common coordinates."""
    row = frames[frames["rank"] == rank].iloc[0]
    path = phase2 / "skeletons" / f"{int(rank):02d}_{Path(row.filename).stem}_skeleton.png"
    sk = np.asarray(Image.open(path).convert("L")) > 127
    ys, xs = np.nonzero(sk)
    dy, dx = shifts.get(int(rank), (0.0, 0.0))
    return np.column_stack([ys + row.crop_top + dy, xs + row.crop_left + dx])


def _largest_piece(skel_img: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(skel_img, structure=np.ones((3, 3)))
    if n == 0:
        return skel_img
    sizes = ndi.sum(skel_img, lab, index=range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


@dataclass
class Contour:
    points: np.ndarray      # (n, 2) integer canvas coordinates (y, x)
    arc: np.ndarray         # (n,) shortest-path distance from the anchor along the skeleton (px)
    tree: cKDTree

    def project(self, y: float, x: float) -> tuple[float, float]:
        """(s_px, distance to contour px) of a point."""
        d, i = self.tree.query([y, x])
        return float(self.arc[i]), float(d)


def reference_contour(phase2: Path, frames: pd.DataFrame, rank: int = 1) -> Contour:
    row = frames[frames["rank"] == rank].iloc[0]
    path = phase2 / "skeletons" / f"{int(rank):02d}_{Path(row.filename).stem}_skeleton.png"
    sk = _largest_piece(np.asarray(Image.open(path).convert("L")) > 127)
    ys, xs = np.nonzero(sk)
    anchor = int(np.argmin(ys))                              # topmost skeleton pixel
    cost = np.where(sk, 1.0, np.inf)
    arc_map, _ = MCP_Geometric(cost, fully_connected=True).find_costs([(ys[anchor], xs[anchor])])
    pts = np.column_stack([ys + row.crop_top, xs + row.crop_left])
    return Contour(pts, arc_map[ys, xs], cKDTree(pts))


def add_contour_position(b: pd.DataFrame, contour: Contour, s: FeatureSettings,
                         delta: pd.DataFrame | None = None) -> pd.DataFrame:
    """Position along the reference contour. `delta` (per-frame ddy, ddx from refine_alignment)
    moves a feature from cumulative-registration coordinates into the reference contour frame."""
    b = b.copy()
    d = delta.set_index("rank") if delta is not None else None
    ddy = b["rank"].map(d["ddy"]).to_numpy(float) if d is not None else np.zeros(len(b))
    ddx = b["rank"].map(d["ddx"]).to_numpy(float) if d is not None else np.zeros(len(b))
    b["y_ref"] = b["y_common"].to_numpy(float) - ddy
    b["x_ref"] = b["x_common"].to_numpy(float) - ddx
    res = [contour.project(y, x) for y, x in zip(b["y_ref"], b["x_ref"])]
    b["s_px"] = [round(r[0], 1) for r in res]
    b["dist_to_ref_contour_px"] = [round(r[1], 1) for r in res]
    b["off_reference_contour"] = b["dist_to_ref_contour_px"] > s.off_contour_px
    return b


def refine_alignment(phase2: Path, frames: pd.DataFrame, shifts: dict, contour: Contour,
                     window: int = 25, tol: int = 2) -> pd.DataFrame:
    """Per-frame extra shift (ddy, ddx) that best lays the frame-1 contour onto the DNA mask of the frame.

    The cumulative pairwise registration drifts (the molecule moves up to ~40 px over the series),
    so each frame is also aligned directly to the reference contour by maximising how many contour
    pixels fall on its (slightly dilated) mask. Pure translation: a deforming molecule stays partly
    unmatched, which is reported as low coverage, not hidden."""
    from skimage import morphology
    pts = contour.points.astype(int)
    rows = []
    for fr in frames.sort_values("rank").itertuples():
        dy, dx = shifts.get(int(fr.rank), (0.0, 0.0))
        path = phase2 / "masks" / f"{int(fr.rank):02d}_{Path(fr.filename).stem}_mask.png"
        mask = np.asarray(Image.open(path).convert("L")) > 127
        dil = ndi.binary_dilation(mask, structure=morphology.disk(tol))
        h, w = dil.shape
        cy0 = np.rint(pts[:, 0] - fr.crop_top - dy).astype(int)
        cx0 = np.rint(pts[:, 1] - fr.crop_left - dx).astype(int)

        def score(ddy, ddx):
            yy, xx = cy0 + ddy, cx0 + ddx
            ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            return float(dil[yy[ok], xx[ok]].sum()) / len(pts)
        best = (score(0, 0) + 1e-9, 0, 0)
        s0 = best[0]
        for ddy in range(-window, window + 1):
            for ddx in range(-window, window + 1):
                sc = score(ddy, ddx) - 0.0005 * (abs(ddy) + abs(ddx))
                if sc > best[0]:
                    best = (sc, ddy, ddx)
        rows.append({"rank": int(fr.rank), "ddy": best[1], "ddx": best[2],
                     "contour_matched_fraction_at_zero": round(s0, 3),
                     "contour_matched_fraction_aligned": round(score(best[1], best[2]), 3)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ track summary

def summarise_tracks(b: pd.DataFrame, last_rank: int, reliable: dict[int, bool], s: FeatureSettings) -> pd.DataFrame:
    rows = []
    for t, g in b.groupby("track_id"):
        g = g.sort_values("rank")
        y, x = g["y_common"].to_numpy(), g["x_common"].to_numpy()
        my, mx = np.median(y), np.median(x)
        spread = float(np.hypot(y - my, x - mx).max())
        net = float(np.hypot(y[-1] - y[0], x[-1] - x[0]))
        n = len(g)
        if n < s.min_track_len:
            motion = "short_track_uncertain"
        elif spread <= s.stationary_px:
            motion = "stationary"
        elif net >= s.moved_px:
            motion = "moved"
        else:
            motion = "wandering_uncertain"
        first, last = int(g["rank"].iloc[0]), int(g["rank"].iloc[-1])
        crosses_bad = any(not reliable.get(r, True) for r in range(first + 1, last + 1))
        rows.append({
            "track_id": int(t), "first_rank": first, "last_rank": last, "n_detections": n,
            "lifespan_frames": last - first + 1, "missed_frames": last - first + 1 - n,
            "motion_class": motion, "max_dist_from_median_px": round(spread, 1),
            "net_displacement_px": round(net, 1),
            "s_first_px": float(g["s_px"].iloc[0]), "s_last_px": float(g["s_px"].iloc[-1]),
            "net_drift_along_contour_px": round(float(g["s_px"].iloc[-1] - g["s_px"].iloc[0]), 1),
            "mean_area_px": round(float(g["area_px"].mean()), 1),
            "area_change_px": round(float(g["area_px"].iloc[-1] - g["area_px"].iloc[0]), 1),
            "mean_relative_intensity": round(float(g["relative_to_ridge_level"].mean()), 2),
            "intensity_change": round(float(g["relative_to_ridge_level"].iloc[-1]
                                            - g["relative_to_ridge_level"].iloc[0]), 2),
            "fraction_off_contour": round(float(g["off_reference_contour"].mean()), 2),
            "fate": "present_to_last_frame" if last == last_rank else f"not_seen_after_rank_{last}",
            "crosses_unreliable_registration": bool(crosses_bad),
            "label": "bright feature (candidate bound particle)"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ events vs features

def _present_near(b: pd.DataFrame, ranks: range, y: float, x: float, radius: float) -> bool:
    g = b[b["rank"].isin(list(ranks))]
    if g.empty:
        return False
    return bool((np.hypot(g["y_common"] - y, g["x_common"] - x) <= radius).any())


def fate_after(b: pd.DataFrame, rank: int, y: float, x: float, s: FeatureSettings) -> str:
    """Fate of a feature seen at (y, x) in `rank`: look at the next window of frames."""
    after = b[(b["rank"] > rank) & (b["rank"] <= rank + s.window_frames)]
    if after.empty:
        return "unknown_end_of_series"
    d = np.hypot(after["y_common"] - y, after["x_common"] - x)
    if (d <= s.keep_radius_px).any():
        return "kept_in_place"
    if (d <= s.gap_link_radius_px + s.moved_px).any():
        return "moved"
    return "lost"


def baseline_loss_rate(b: pd.DataFrame, last_rank: int, s: FeatureSettings) -> dict:
    """Share of all detections with each fate (reference level for the near-event numbers)."""
    counts = {"kept_in_place": 0, "moved": 0, "lost": 0}
    for r in b.itertuples():
        if r.rank > last_rank - s.window_frames:
            continue
        counts[fate_after(b, r.rank, r.y_common, r.x_common, s)] += 1
    n = sum(counts.values())
    return {**counts, "n": n, "lost_fraction": round(counts["lost"] / n, 3) if n else np.nan}


def event_feature_table(events: pd.DataFrame, b: pd.DataFrame, frames: pd.DataFrame, shifts: dict,
                        contour: Contour, s: FeatureSettings, delta: pd.DataFrame | None = None) -> pd.DataFrame:
    crop = frames.set_index("rank")[["crop_top", "crop_left"]]
    rows = []
    sp = events[events["event"] == "split_candidate"]
    for e in sp.itertuples():
        a = int(e.rank_from)
        if pd.isna(e.event_x_px):
            continue
        dy, dx = shifts.get(a, (0.0, 0.0))
        y = float(e.event_y_px) + crop.loc[a, "crop_top"] + dy
        x = float(e.event_x_px) + crop.loc[a, "crop_left"] + dx
        dd = delta.set_index("rank").loc[a] if delta is not None else None
        y_ref = y - (float(dd["ddy"]) if dd is not None else 0.0)
        x_ref = x - (float(dd["ddx"]) if dd is not None else 0.0)
        s_ev, d_ev = contour.project(y_ref, x_ref)
        g = b[b["rank"] == a]
        dist = np.hypot(g["y_common"] - y, g["x_common"] - x) if len(g) else pd.Series(dtype=float)
        near = g[dist <= s.near_event_px]
        fates = [fate_after(b, a, r.y_common, r.x_common, s) for r in near.itertuples()]
        rows.append({
            "rank_from": a, "rank_to": int(e.rank_to), "persistence": e.persistence,
            "min_gap_px": e.min_gap_px, "event_y_common": round(y, 1), "event_x_common": round(x, 1),
            "event_y_ref": round(y_ref, 1), "event_x_ref": round(x_ref, 1),
            "event_s_px": round(s_ev, 1), "event_dist_to_ref_contour_px": round(d_ev, 1),
            "n_features_in_frame": len(g),
            "nearest_feature_px": round(float(dist.min()), 1) if len(g) else np.nan,
            "n_features_near": len(near),
            "near_kept_in_place": fates.count("kept_in_place"), "near_moved": fates.count("moved"),
            "near_lost": fates.count("lost"), "near_unknown": fates.count("unknown_end_of_series"),
            "note": "association only; candidate split, not a confirmed cleavage"})
    return pd.DataFrame(rows)


def null_comparison(events: pd.DataFrame, b: pd.DataFrame, phase2: Path, frames: pd.DataFrame,
                    shifts: dict, s: FeatureSettings) -> dict:
    """Is the nearest-feature distance at split gaps smaller/larger than at random contour points?

    For every split candidate, random points are drawn from the DNA skeleton pieces (>= min
    length) of the same frame; distance to the nearest bright feature of that frame is the null.
    Permutation-style: mean distance over events vs mean over one random point per event."""
    rng = np.random.default_rng(s.seed)
    obs, pools = [], []
    for e in events.itertuples():
        g = b[b["rank"] == e.rank_from]
        if len(g) == 0 or np.isnan(e.nearest_feature_px):
            continue
        pts = skeleton_common(phase2, frames, int(e.rank_from), shifts)
        lab_ok = len(pts) >= s.min_skeleton_px
        if not lab_ok:
            continue
        tree = cKDTree(g[["y_common", "x_common"]].to_numpy())
        d_null = tree.query(pts)[0]
        obs.append(float(e.nearest_feature_px))
        pools.append(d_null)
    if not obs:
        return {"n_events": 0}
    obs = np.array(obs)
    sims = np.array([np.mean([rng.choice(p) for p in pools]) for _ in range(s.n_null)])
    percentiles = [float((p <= o).mean()) for p, o in zip(pools, obs)]
    return {"n_events": int(len(obs)), "observed_mean_distance_px": round(float(obs.mean()), 1),
            "null_mean_distance_px": round(float(sims.mean()), 1),
            "p_closer_than_random": round(float((sims <= obs.mean()).mean()), 4),
            "p_farther_than_random": round(float((sims >= obs.mean()).mean()), 4),
            "mean_event_percentile": round(float(np.mean(percentiles)), 3),
            "null_means": sims, "observed": obs, "percentiles": percentiles}


# ------------------------------------------------------------------ mechanisms table

def mechanism_table(trk: pd.DataFrame, ev: pd.DataFrame, null_all: dict, null_pers: dict,
                    base: dict, s: FeatureSettings) -> pd.DataFrame:
    long = trk[trk["n_detections"] >= s.min_track_len]
    n_long = len(long)
    n_stat = int((long["motion_class"] == "stationary").sum())
    n_move = int((long["motion_class"] == "moved").sum())
    near_total = int(ev[["near_kept_in_place", "near_moved", "near_lost"]].sum().sum()) if len(ev) else 0
    near_lost = int(ev["near_lost"].sum()) if len(ev) else 0
    near_moved = int(ev["near_moved"].sum()) if len(ev) else 0
    ev_with_near = int((ev["n_features_near"] > 0).sum()) if len(ev) else 0
    p_all = null_all.get("n_events", 0)
    rows = [
        {"hypothesis": "H1", "hypothesis_tr": "Enzim dendrimerleri iterek keser",
         "expected_if_true_tr": "Kopma yerinin yakınındaki parlak özellikler olaydan sonra kaybolur ya da yer değiştirir; "
                                "bu oran diğer yerlerdeki kayıp oranından belirgin yüksektir.",
         "observed": f"Kopma adaylarının {ev_with_near}/{len(ev)} tanesinin {s.near_event_px:.0f} px çevresinde özellik var. "
                     f"Bu {near_total} özellikten {near_lost} kayboldu, {near_moved} yer değiştirdi. "
                     f"Genel kayıp oranı {base.get('lost_fraction')} (n={base.get('n')}).",
         "limitation_tr": "Çoğu aday kopma kısa sürede geri birleşiyor (görüntü hatası olabilir). Özelliklerin ne olduğu bilinmiyor."},
        {"hypothesis": "H2", "hypothesis_tr": "Dendrimerler DNA üzerinde kendiliğinden kayar",
         "expected_if_true_tr": "Uzun süre görünen özelliklerin birçoğu DNA boyunca net bir yer değiştirme gösterir "
                                "ve bu kopma olaylarından bağımsızdır.",
         "observed": f"{n_long} uzun izden {n_stat} tanesi sabit, {n_move} tanesi yer değiştirmiş "
                     f"(sabit: konum medyanından en fazla {s.stationary_px:.0f} px sapma).",
         "limitation_tr": "Hizalama ve eşleştirme hatası küçük kaymaları yapay üretebilir. Konum yalnızca piksel cinsinden."},
        {"hypothesis": "H3", "hypothesis_tr": "Enzim yalnızca çıplak (korunmayan) bölgeleri keser",
         "expected_if_true_tr": "Kopma yerleri rastgele DNA noktalarından parlak özelliklere daha uzaktır; "
                                "özellikler kesikten sonra yerinde kalır.",
         "observed": (f"Tüm adaylarda (n={p_all}) en yakın özelliğe uzaklık ortalaması {null_all.get('observed_mean_distance_px')} px, "
                      f"rastgele DNA noktaları için {null_all.get('null_mean_distance_px')} px "
                      f"(rastgeleden uzak olma p={null_all.get('p_farther_than_random')}, yakın olma p={null_all.get('p_closer_than_random')}). "
                      f"Kalıcı 4 adayda: gözlenen {null_pers.get('observed_mean_distance_px')} px, rastgele {null_pers.get('null_mean_distance_px')} px "
                      f"(uzak olma p={null_pers.get('p_farther_than_random')})."),
         "limitation_tr": "Tek molekül, olaylar birbirinden bağımsız değil. Parlak özelliğin yanındaki soluk DNA parçaları eşiklemede "
                          "kolayca kopuk görünür (Faz 4'te halka çağrısının üst parlak özelliğin yanındaki küçük boşluğa bağlı olduğunu görmüştük). "
                          "Bu yüzden 'özelliğe yakın' sonucu yöntem etkisi olabilir, biyolojik bulgu sayılmaz. p değerleri yalnızca tanımlayıcıdır."}]
    df = pd.DataFrame(rows)
    df["verdict"] = "none: this data cannot separate the mechanisms; descriptive numbers only"
    return df


# ------------------------------------------------------------------ figures

def figures(b: pd.DataFrame, trk: pd.DataFrame, ev: pd.DataFrame, null_all: dict, out: Path, last_rank: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
        plt.close(fig)

    longids = set(trk.loc[trk["n_detections"] >= 4, "track_id"])
    d_all = b[b["track_id"].isin(longids)]
    d = d_all[~d_all["off_reference_contour"]]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    cmap = plt.get_cmap("tab20")
    for t, g in d.groupby("track_id"):
        ax.scatter(g["s_px"], g["rank"], s=g["area_px"] * 0.6, color=cmap(t % 20), alpha=0.7, lw=0)
    if len(ev):
        pers = ev["persistence"] == "persistent"
        ax.scatter(ev.loc[~pers, "event_s_px"], ev.loc[~pers, "rank_to"], marker="x", c="k", s=50, label="split candidate (re-joined)")
        ax.scatter(ev.loc[pers, "event_s_px"], ev.loc[pers, "rank_to"], marker="X", c="red", s=70, label="split candidate (persistent)")
    ax.invert_yaxis()
    ax.set_xlabel("position along reference contour, s (px; shortest path from topmost point of frame 1)")
    ax.set_ylabel("frame index (provisional order)")
    ax.set_title(f"Bright features along the reference contour (dot size ~ area; tracks of 4+ detections; "
                 f"{len(d_all) - len(d)} of {len(d_all)} detections off the contour not drawn)", fontsize=8, loc="left")
    ax.legend(fontsize=7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_feature_kymograph")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for t, g in d.groupby("track_id"):
        g = g.sort_values("rank")
        axs[0].plot(g["rank"], g["area_px"], color=cmap(t % 20), lw=1, marker="o", ms=2)
        axs[1].plot(g["rank"], g["relative_to_ridge_level"], color=cmap(t % 20), lw=1, marker="o", ms=2)
    axs[0].set_ylabel("area (px)")
    axs[1].set_ylabel("peak / DNA ridge level (JPG intensity ratio)")
    for a in axs:
        a.set_xlabel("frame index (provisional order)")
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.25, lw=0.5)
    axs[0].set_title("Size of each tracked feature", fontsize=9, loc="left")
    axs[1].set_title("Relative brightness of each tracked feature", fontsize=9, loc="left")
    save(fig, "fig_feature_size_brightness_time")

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].scatter(trk["mean_area_px"], trk["mean_relative_intensity"], s=trk["n_detections"] * 4 + 8,
                   c=trk["n_detections"], cmap="viridis", alpha=0.8)
    axs[0].set_xlabel("mean area (px)")
    axs[0].set_ylabel("mean peak / ridge level")
    axs[0].set_title("One dot per track (colour/size = number of detections). Unlabelled groups: no identification.",
                     fontsize=8, loc="left")
    axs[1].hist(trk["lifespan_frames"], bins=range(1, int(trk["lifespan_frames"].max()) + 3), color="#4a7be0")
    axs[1].set_xlabel("track lifespan (frames)")
    axs[1].set_ylabel("number of tracks")
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    save(fig, "fig_feature_size_brightness_groups")

    if null_all.get("n_events"):
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.hist(null_all["null_means"], bins=40, color="#b8c4de")
        ax.axvline(null_all["observed_mean_distance_px"], color="red", lw=2, label="observed (split candidates)")
        ax.set_xlabel("mean distance to nearest bright feature (px)")
        ax.set_ylabel("random draws")
        ax.set_title("Split places vs random DNA places (descriptive, one molecule)", fontsize=9, loc="left")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        save(fig, "fig_event_vs_random")


# ------------------------------------------------------------------ run

def run(outputs: Path, s: FeatureSettings) -> dict:
    inp = load_inputs(outputs)
    frames, reg, events = inp["frames"], inp["registration"], inp["events"]
    last_rank = int(frames["rank"].max())
    shifts = cumulative_shifts(reg)
    reliable = {int(r.rank_to): bool(r.reliable) for r in reg.itertuples()}
    b = common_positions(inp["bright"], frames, reg)
    b = link_features(b, s)
    contour = reference_contour(inp["phase2"], frames, 1)
    delta = refine_alignment(inp["phase2"], frames, shifts, contour)
    b = add_contour_position(b, contour, s, delta)
    b["registration_reliable"] = b["rank"].map(lambda r: reliable.get(int(r), True))
    trk = summarise_tracks(b, last_rank, reliable, s)
    ev = event_feature_table(events, b, frames, shifts, contour, s, delta)
    base = baseline_loss_rate(b, last_rank, s)
    null_all = null_comparison(ev, b, inp["phase2"], frames, shifts, s) if len(ev) else {"n_events": 0}
    pers = ev[ev["persistence"] == "persistent"] if len(ev) else ev
    null_pers = null_comparison(pers, b, inp["phase2"], frames, shifts, s) if len(pers) else {"n_events": 0}
    mech = mechanism_table(trk, ev, null_all, null_pers, base, s)
    out = outputs / "phase6"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["track_id", "rank", "filename", "feature_id", "x_common", "y_common", "x_ref", "y_ref", "s_px",
            "dist_to_ref_contour_px", "off_reference_contour", "area_px", "relative_to_ridge_level",
            "radius_px", "peak_z", "gap_before", "registration_reliable", "label"]
    b[cols].round(2).to_csv(out / "feature_detections.csv", index=False)
    delta.to_csv(out / "contour_alignment.csv", index=False)
    trk.to_csv(out / "feature_tracks.csv", index=False)
    ev.to_csv(out / "event_feature_association.csv", index=False)
    mech.to_csv(out / "mechanism_summary.csv", index=False)
    summary = {k: v for k, v in null_all.items() if not isinstance(v, (np.ndarray, list))}
    summary_p = {k: v for k, v in null_pers.items() if not isinstance(v, (np.ndarray, list))}
    (out / "parameters.json").write_text(json.dumps({
        "settings": asdict(s), "baseline_fate_counts": base,
        "null_all_events": summary, "null_persistent_events": summary_p,
        "reference_contour": "longest skeleton piece of frame 1; s = shortest path from its topmost pixel",
        "notes": "features are unidentified bright features; no mechanism is confirmed; "
                 "all statistics descriptive (one molecule, one experiment)"}, indent=2, default=float), encoding="utf-8")
    figures(b, trk, ev, null_all, out, last_rank)
    return {"detections": b, "tracks": trk, "events": ev, "mechanisms": mech, "baseline": base,
            "null_all": null_all, "null_persistent": null_pers}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 6: bright-feature tracking and mechanism summary.")
    p.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    a = p.parse_args(argv)
    try:
        res = run(a.outputs, FeatureSettings())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    t = res["tracks"]
    print(f"Detections: {len(res['detections'])}   tracks: {len(t)}   tracks with 4+ detections: "
          f"{int((t['n_detections'] >= 4).sum())}")
    print(t["motion_class"].value_counts().to_string())
    print(res["mechanisms"][["hypothesis", "observed"]].to_string(index=False))
    print(f"Results: {a.outputs / 'phase6'}")
    print("Features are unidentified; nothing here confirms a mechanism or a cut.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
