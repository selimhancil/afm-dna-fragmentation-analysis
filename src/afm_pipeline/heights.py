"""Heading 3: height profile of the DNA and of the bright features, in ASSUMED nm.

The JPGs are colour renderings of AFM height. The colour-to-height mapping is not known, so heights
here come from an explicit, changeable assumption in config/project_config.ini [height]:
the luminance range of every frame (low_percentile .. high_percentile) is linear in height and spans
z_range_nm (10 nm, from the supervisor's email). Every output is labelled ASSUMED while status != confirmed.

Height above the local substrate:  h_nm = flat_luminance * k,  k = z_range_nm / (P_high - P_low) of the
raw luminance of the frame. A plausibility check compares the DNA ridge height with the usual 0.5-2 nm.
Changing z_range_nm rescales all heights linearly; shapes, ratios and timing do not change.
"""
from __future__ import annotations

import argparse
import configparser
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config as cfgmod
from .dna import load_dna_settings, preprocess, ridge_level
from .inventory import DEFAULT_DATA_DIR, PROJECT_ROOT
from .phase4 import cumulative_shifts
from .phase6 import reference_contour
from .phase7 import contour_profile
from .segment import load_rgb

DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"


@dataclass
class HeightConfig:
    status: str = "unknown"
    z_range_nm: float | None = None
    low_percentile: float = 0.5
    high_percentile: float = 99.95
    source: str = ""
    plausible_low_nm: float = 0.5
    plausible_high_nm: float = 2.0
    anchor_dna_height_nm: float | None = None   # optional: rescale so the median DNA ridge equals this

    @property
    def usable(self) -> bool:
        return self.status in ("assumed", "confirmed") and self.z_range_nm is not None

    @property
    def label(self) -> str:
        return "measured (confirmed)" if self.status == "confirmed" else "ASSUMED (not measured)"


def load_height_config(path: Path | str = cfgmod.DEFAULT_CONFIG_PATH) -> HeightConfig:
    cp = configparser.ConfigParser(inline_comment_prefixes=None)
    cp.read(path, encoding="utf-8")
    if not cp.has_section("height"):
        return HeightConfig()
    g = lambda k, d="": cp.get("height", k, fallback=d).strip()
    status = g("status", "unknown").lower()
    if status not in ("unknown", "assumed", "confirmed"):
        raise cfgmod.ConfigError("[height] status must be unknown, assumed or confirmed.")
    if status == "unknown":
        return HeightConfig()
    try:
        z = float(g("z_range_nm"))
        lo, hi = float(g("low_percentile", "0.5")), float(g("high_percentile", "99.95"))
        pl, ph = float(g("plausible_dna_height_nm_low", "0.5")), float(g("plausible_dna_height_nm_high", "2.0"))
        anchor = float(g("anchor_dna_height_nm")) if g("anchor_dna_height_nm") else None
    except ValueError as exc:
        raise cfgmod.ConfigError(f"[height] a number is not valid: {exc}") from exc
    if z <= 0 or not (0 <= lo < hi <= 100):
        raise cfgmod.ConfigError("[height] z_range_nm must be positive and 0 <= low_percentile < high_percentile <= 100.")
    if not g("source"):
        raise cfgmod.ConfigError("[height] a value is given but 'source' is empty (who or what says so?).")
    if anchor is not None and anchor <= 0:
        raise cfgmod.ConfigError("[height] anchor_dna_height_nm must be positive.")
    return HeightConfig(status, z, lo, hi, g("source"), pl, ph, anchor)


def nm_per_noise_unit(rgb_crop: np.ndarray, noise: float, h: HeightConfig) -> float:
    """Height (nm) of one z unit (= one noise sigma) for this frame."""
    lum = 0.299 * rgb_crop[..., 0] + 0.587 * rgb_crop[..., 1] + 0.114 * rgb_crop[..., 2]
    lo, hi = np.percentile(lum, [h.low_percentile, h.high_percentile])
    if hi <= lo:
        raise ValueError("Frame has no luminance range.")
    return float(noise * h.z_range_nm / (hi - lo))


def plausibility(ridge_nm: np.ndarray, h: HeightConfig) -> dict:
    med = float(np.nanmedian(ridge_nm))
    inside = h.plausible_low_nm <= med <= h.plausible_high_nm
    return {"median_ridge_nm": round(med, 2), "plausible_range_nm": [h.plausible_low_nm, h.plausible_high_nm],
            "within_plausible_range": bool(inside),
            "reading": ("DNA height is within the expected range; the linear mapping looks plausible (consistency, not proof)"
                        if inside else
                        "DNA height is OUTSIDE the expected range; the linear mapping may be wrong and the heights are unreliable")}


def feature_heights(det: pd.DataFrame, trk: pd.DataFrame, k_by_rank: dict, noise_by_rank: dict,
                    ridge_nm_by_rank: dict) -> pd.DataFrame:
    """Peak height of each bright feature in assumed nm, plus height above the DNA ridge."""
    s = det[~det["recovered"]].copy()
    s["peak_nm"] = s["peak_z"] * s["rank"].map(k_by_rank)
    s["ridge_nm"] = s["rank"].map(ridge_nm_by_rank)
    s["above_ridge_nm"] = s["peak_nm"] - s["ridge_nm"]
    g = s.groupby("track_id")
    out = pd.DataFrame({"n_strong": g.size(), "median_peak_nm": g["peak_nm"].median(),
                        "min_peak_nm": g["peak_nm"].min(), "max_peak_nm": g["peak_nm"].max(),
                        "median_above_ridge_nm": g["above_ridge_nm"].median()}).reset_index()
    return out.merge(trk[["track_id", "motion_class", "confidence", "first_rank", "last_rank"]], on="track_id", how="left").round(2)


def contour_height_map(rel: np.ndarray, level: np.ndarray, k: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(frames x contour points) height in nm: rel * level * k. Invalid-alignment frames are NaN."""
    m = rel * (level * k)[:, None]
    m[~valid] = np.nan
    return m


def figures(frame_df: pd.DataFrame, fh: pd.DataFrame, hmap: np.ndarray, ranks: list, arc: np.ndarray,
            hc: HeightConfig, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tag = f"{hc.label}: linear colour mapping over {hc.z_range_nm:g} nm"
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    ax = axs[0]
    valid = ~np.all(np.isnan(hmap), axis=1)
    if valid.any():
        vr = np.array(ranks)[valid]
        im = ax.imshow(hmap[valid].T, aspect="auto", origin="lower", cmap="magma",
                       extent=[vr.min() - 0.5, vr.max() + 0.5, float(arc.min()), float(arc.max())],
                       vmin=0, vmax=float(np.nanpercentile(hmap, 99)))
        fig.colorbar(im, ax=ax, label="height (nm, ASSUMED)")
    ax.set_xlabel("frame index (provisional order)")
    ax.set_ylabel("position along reference DNA contour (px)")
    ax.set_title("Height along the DNA over time", fontsize=9, loc="left")
    ax = axs[1]
    ax.plot(frame_df["rank"], frame_df["ridge_height_nm"], marker="o", ms=2.5, lw=1, color="#2255c4", label="DNA ridge (typical)")
    ax.axhspan(hc.plausible_low_nm, hc.plausible_high_nm, color="green", alpha=0.12, label="usual dsDNA range")
    ax.set_xlabel("frame index (provisional order)")
    ax.set_ylabel("height (nm, ASSUMED)")
    ax.set_title("DNA ridge height per frame", fontsize=9, loc="left")
    ax.legend(fontsize=7)
    ax = axs[2]
    if len(fh):
        f = fh.sort_values("n_strong", ascending=False).head(25).sort_values("median_peak_nm")
        ax.barh(range(len(f)), f["median_peak_nm"], color="#d9534f", alpha=0.85)
        ax.axvline(float(frame_df["ridge_height_nm"].median()), color="#2255c4", ls="--", lw=1, label="DNA ridge")
        ax.set_yticks(range(len(f)))
        ax.set_yticklabels([f"track {t}" for t in f["track_id"]], fontsize=6)
        ax.legend(fontsize=7)
    ax.set_xlabel("median peak height (nm, ASSUMED)")
    ax.set_title("25 best-observed bright features", fontsize=9, loc="left")
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(tag, fontsize=9, color="#a33", y=1.02)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_height_profile.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def summary_text(hc: HeightConfig, plaus: dict, frame_df: pd.DataFrame, fh: pd.DataFrame) -> str:
    L = ["## Heading 3: height profile (English summary)", "",
         f"- **STATUS: {hc.label}.** The JPG is a colour image; the colour-to-height mapping is unknown. Temporary assumption: each frame's "
         f"dark-to-light colour range is linear and spans {hc.z_range_nm:g} nm in total (the Z range in the supervisor's email). "
         "These values are not measurements; change them in the `[height]` section of `config/project_config.ini` "
         "(the code does not change). If z_range_nm is changed, all heights scale proportionally; the figures and timing do not change.",
         f"- Typical DNA ridge height (assumed): median {plaus['median_ridge_nm']} nm; usual dsDNA range "
         f"{plaus['plausible_range_nm'][0]}–{plaus['plausible_range_nm'][1]} nm. Consistency check: {plaus['reading']}."]
    if len(frame_df):
        a, b = frame_df.iloc[0], frame_df.iloc[-1]
        L.append(f"- Ridge height is {a.ridge_height_nm:.2f} nm in the first frame and {b.ridge_height_nm:.2f} nm in the last "
                 f"(median across frames {frame_df['ridge_height_nm'].median():.2f}, lowest {frame_df['ridge_height_nm'].min():.2f}, "
                 f"highest {frame_df['ridge_height_nm'].max():.2f}). The colour scale is also normalised per frame; "
                 "in reality each frame may have been recorded with its own z range.")
    if len(fh):
        L.append(f"- Bright features ({len(fh)} tracks): median peak height {fh['median_peak_nm'].median():.1f} nm "
                 f"(lowest {fh['median_peak_nm'].min():.1f}, highest {fh['median_peak_nm'].max():.1f}); "
                 f"median {fh['median_above_ridge_nm'].median():.1f} nm above the DNA ridge. "
                 "These values cannot be used to identify the feature (assumed scale, tip broadening and colour mapping).")
    L += ["- When the supervisor provides the real raw AFM files or the colour bar, set `status = confirmed` in the `[height]` section "
          "and all heights are regenerated with real values."]
    return "\n".join(L) + "\n"


def run(outputs: Path, data_dir: Path, config: Path = cfgmod.DEFAULT_CONFIG_PATH) -> dict:
    hc = load_height_config(config)
    if not hc.usable:
        raise ValueError("[height] status is 'unknown' in the config: no heights are produced. "
                         "Set status = assumed (with z_range_nm) or confirmed.")
    p2, p6, p6b = outputs / "phase2", outputs / "phase6", outputs / "phase6b"
    for f in (p2 / "frames.csv", p6 / "contour_alignment.csv", outputs / "phase3" / "registration.csv",
              p6b / "feature_detections_v2.csv", p6b / "feature_tracks_v2.csv"):
        if not f.is_file():
            raise FileNotFoundError(f"{f} not found. Run phases 2 to 6b first (run_all.sh).")
    frames = pd.read_csv(p2 / "frames.csv").sort_values("rank")
    shifts = cumulative_shifts(pd.read_csv(outputs / "phase3" / "registration.csv"))
    dna = load_dna_settings(config)
    contour = reference_contour(p2, frames, 1)
    delta = pd.read_csv(p6 / "contour_alignment.csv")
    prof = contour_profile(data_dir, outputs, frames, shifts, contour, dna, delta)

    k_by, noise_by, ridge_by, level_by = {}, {}, {}, {}
    for row in frames.itertuples():
        pre = preprocess(load_rgb(data_dir / row.filename), dna)
        stem = Path(row.filename).stem
        mask = np.asarray(Image.open(p2 / "masks" / f"{int(row.rank):02d}_{stem}_mask.png").convert("L")) > 127
        level = ridge_level(pre.z, mask, dna)
        k = nm_per_noise_unit(pre.rgb, pre.noise, hc)
        k_by[int(row.rank)], noise_by[int(row.rank)] = k, pre.noise
        level_by[int(row.rank)] = float(level) if level else np.nan
        ridge_by[int(row.rank)] = float(level * k) if level else np.nan
    scale_factor = 1.0
    if hc.anchor_dna_height_nm is not None:
        scale_factor = hc.anchor_dna_height_nm / float(np.nanmedian(list(ridge_by.values())))
        k_by = {r: v * scale_factor for r, v in k_by.items()}
        ridge_by = {r: v * scale_factor for r, v in ridge_by.items()}
    rows = []
    for row in frames.itertuples():
        pre_rank = int(row.rank)
        k = k_by[pre_rank]
        level = level_by[pre_rank]
        rows.append({"rank": int(row.rank), "nm_per_noise_unit": round(k, 4), "noise_sigma_nm": round(k, 3),
                     "ridge_level_noise_units": round(float(level), 2) if level else np.nan,
                     "ridge_height_nm": round(ridge_by[int(row.rank)], 3)})
    frame_df = pd.DataFrame(rows)
    pf = prof["per_frame"].set_index("rank")
    frame_df["mean_contour_height_nm"] = [round(float(pf.loc[r, "mean_relative_intensity_on_contour"]) * ridge_by[r], 3)
                                          for r in frame_df["rank"]]
    frame_df["contour_covered_fraction"] = frame_df["rank"].map(pf["contour_covered_fraction"])
    frame_df["aligned_valid"] = frame_df["rank"].map(pf["aligned_match_valid_for_loss_map"])
    frame_df["height_status"] = hc.label

    det = pd.read_csv(p6b / "feature_detections_v2.csv")
    trk = pd.read_csv(p6b / "feature_tracks_v2.csv")
    fh = feature_heights(det, trk, k_by, noise_by, ridge_by)
    fh["height_status"] = hc.label

    ranks = prof["ranks"]
    lvl = np.array([frame_df.set_index("rank").loc[r, "ridge_level_noise_units"] for r in ranks], float)
    kk = np.array([k_by[r] for r in ranks], float)
    hmap = contour_height_map(prof["rel"], lvl, kk, np.asarray(prof["valid"], bool))
    plaus = plausibility(frame_df["ridge_height_nm"].to_numpy(), hc)
    if hc.anchor_dna_height_nm is not None:
        plaus["reading"] = (f"DNA height was set directly to {hc.anchor_dna_height_nm:g} nm (anchor); "
                            "the check is meaningless in this case, only ratios and changes in height can be interpreted")
        plaus["anchored"] = True
        plaus["within_plausible_range"] = None   # circular when anchored
        plaus["scale_factor_vs_z_range_mapping"] = round(scale_factor, 3)

    out = outputs / "heights"
    out.mkdir(parents=True, exist_ok=True)
    frame_df.to_csv(out / "dna_height_by_frame.csv", index=False)
    fh.to_csv(out / "feature_heights.csv", index=False)
    bins = np.arange(0, float(contour.arc.max()) + 10, 10)
    idx = np.digitize(contour.arc, bins)
    prof_rows = []
    for fi, r in enumerate(ranks):
        for b in np.unique(idx):
            v = hmap[fi, idx == b]
            if np.isfinite(v).any():
                prof_rows.append({"rank": r, "s_bin_start_px": float(bins[b - 1]), "mean_height_nm": round(float(np.nanmean(v)), 3),
                                  "height_status": hc.label})
    pd.DataFrame(prof_rows).to_csv(out / "height_profile_along_contour.csv", index=False)
    (out / "summary.md").write_text(summary_text(hc, plaus, frame_df, fh), encoding="utf-8")
    (out / "parameters.json").write_text(json.dumps({"height_config": asdict(hc), "label": hc.label, "plausibility": plaus,
                                                     "notes": "ASSUMED linear colour mapping; not measured height"}, indent=2), encoding="utf-8")
    figures(frame_df, fh, hmap, ranks, contour.arc, hc, out)
    return {"config": hc, "plausibility": plaus, "frames": frame_df, "features": fh}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Heading 3: height profile in assumed nm.")
    p.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--config", type=Path, default=cfgmod.DEFAULT_CONFIG_PATH)
    a = p.parse_args(argv)
    try:
        r = run(a.outputs, a.data_dir, a.config)
    except (FileNotFoundError, ValueError, cfgmod.ConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(r["config"].label, r["plausibility"])
    print(r["frames"][["ridge_height_nm", "mean_contour_height_nm"]].describe().round(2).to_string())
    print(r["features"][["median_peak_nm", "median_above_ridge_nm"]].describe().round(2).to_string())
    print(f"Results: {a.outputs / 'heights'}")
    print("Heights are ASSUMED (linear colour mapping), not measured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
