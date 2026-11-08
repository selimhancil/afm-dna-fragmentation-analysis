"""Heading 4: DNA contour length over time, with a measured bias of the length estimator.

Length of a 1-px skeleton by summing links (orthogonal 1, diagonal sqrt 2) is NOT the true curve length:
digital curves are biased. The bias is measured here on synthetic curves of known length that are
rasterised, blurred, thresholded and skeletonised the way the real images are, and every real length is
reported both raw and bias-corrected (with the spread of the bias as uncertainty).

Scale cross-check: if the largest piece of frame 1 were the whole pBR322 (4363 bp, 0.34 nm/bp = 1483 nm),
the implied nm/pixel is compared with the estimated scale from the config. A mismatch is reported, never hidden.
Lengths in nm are only as good as the scale (still estimated).
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
from skimage import draw, morphology

from . import config as cfgmod
from .dna import skeleton_length
from .inventory import PROJECT_ROOT

DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"
PLASMID_BP = 4363
BP_NM = 0.34


@dataclass
class LengthSettings:
    min_piece_px: int = 20
    n_synthetic: int = 60
    strand_width_px: float = 3.0
    blur_px: float = 1.0
    seed: int = 1


def _curve(rng: np.random.Generator, n: int = 400) -> np.ndarray:
    """A smooth random open curve (points, in px) with gentle curvature, 1 px point spacing target."""
    ang = np.cumsum(rng.normal(0, 0.06, n)) + rng.uniform(0, 2 * np.pi)
    steps = np.column_stack([np.sin(ang), np.cos(ang)])
    pts = np.cumsum(steps, axis=0)
    return pts - pts.min(axis=0) + 20


def synthetic_bias(s: LengthSettings) -> pd.DataFrame:
    """Raw skeleton length / true length for rasterised synthetic curves."""
    rng = np.random.default_rng(s.seed)
    rows = []
    for i in range(s.n_synthetic):
        pts = _curve(rng, int(rng.integers(150, 600)))
        true = float(np.sum(np.hypot(*np.diff(pts, axis=0).T)))
        h, w = int(pts[:, 0].max()) + 40, int(pts[:, 1].max()) + 40
        img = np.zeros((h, w))
        for (y0, x0), (y1, x1) in zip(pts[:-1], pts[1:]):
            rr, cc = draw.line(int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)))
            img[rr, cc] = 1.0
        img = ndi.gaussian_filter(img, s.strand_width_px / 2.0)
        mask = img > 0.5 * img.max() * 0.6
        sk = morphology.skeletonize(mask)
        lab, n = ndi.label(sk, structure=np.ones((3, 3)))
        if n == 0:
            continue
        sizes = ndi.sum(sk, lab, index=range(1, n + 1))
        main = lab == (int(np.argmax(sizes)) + 1)
        raw = skeleton_length(main)
        rows.append({"true_px": round(true, 1), "raw_skeleton_px": round(raw, 1), "ratio": raw / true})
    return pd.DataFrame(rows)


def bias_summary(b: pd.DataFrame) -> dict:
    r = b["ratio"].to_numpy()
    lo, hi = np.percentile(r, [2.5, 97.5])
    return {"n_curves": int(len(r)), "mean_ratio": float(r.mean()), "sd_ratio": float(r.std(ddof=1)),
            "ratio_95_low": float(lo), "ratio_95_high": float(hi),
            "reading": ("the pixel-chain length %s the true length by %.1f%% on average"
                        % ("overestimates" if r.mean() > 1 else "underestimates", abs(r.mean() - 1) * 100))}


def pieces(skel: np.ndarray, min_px: int, small: list | None = None) -> list[float]:
    """Lengths (px, raw chain) of skeleton pieces with at least min_px pixels, longest first.
    Shorter pieces are appended to `small` (if given) so that the length not counted is known."""
    lab, n = ndi.label(skel, structure=np.ones((3, 3)))
    out = []
    for k in range(1, n + 1):
        sub = lab == k
        if sub.sum() >= min_px:
            out.append(skeleton_length(sub))
        elif small is not None:
            small.append(skeleton_length(sub))
    return sorted(out, reverse=True)


def length_table(outputs: Path, frames: pd.DataFrame, bias: dict, s: LengthSettings,
                 scale_by_rank: dict, anchor_nm_per_px: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, frag = [], []
    for row in frames.sort_values("rank").itertuples():
        stem = Path(row.filename).stem
        sk = np.asarray(Image.open(outputs / "phase2" / "skeletons" / f"{int(row.rank):02d}_{stem}_skeleton.png").convert("L")) > 127
        sm: list = []
        p = pieces(sk, s.min_piece_px, sm)
        total = float(sum(p))
        largest = p[0] if p else 0.0
        k = scale_by_rank[int(row.rank)]
        c = 1.0 / bias["mean_ratio"]
        rows.append({"rank": int(row.rank), "time_earliest_s": row.time_earliest_s, "time_latest_s": row.time_latest_s,
                     "n_pieces": len(p), "n_pieces_below_min_not_counted": len(sm),
                     "length_below_min_not_counted_px": round(float(sum(sm)), 1), "total_raw_px": round(total, 1), "largest_raw_px": round(largest, 1),
                     "total_corrected_px": round(total * c, 1), "largest_corrected_px": round(largest * c, 1),
                     "total_nm_estimated_scale": round(total * c * k, 0), "largest_nm_estimated_scale": round(largest * c * k, 0),
                     "total_nm_plasmid_anchor": round(total * c * anchor_nm_per_px, 0) if anchor_nm_per_px else np.nan,
                     "nm_per_pixel_used": round(k, 3)})
        for i, v in enumerate(p, 1):
            frag.append({"rank": int(row.rank), "piece": i, "length_raw_px": round(v, 1), "length_corrected_px": round(v * c, 1),
                         "length_nm_estimated_scale": round(v * c * k, 0)})
    return pd.DataFrame(rows), pd.DataFrame(frag)


def scale_check(first_largest_corrected_px: float, first_total_corrected_px: float, nm_per_px: float) -> dict:
    full_nm = PLASMID_BP * BP_NM
    out = {"plasmid_full_contour_nm": round(full_nm, 0), "estimated_nm_per_px": round(nm_per_px, 3),
           "frame1_largest_nm_at_estimated_scale": round(first_largest_corrected_px * nm_per_px, 0),
           "frame1_total_nm_at_estimated_scale": round(first_total_corrected_px * nm_per_px, 0)}
    out["implied_nm_per_px_if_largest_is_whole_plasmid"] = round(full_nm / first_largest_corrected_px, 3)
    out["ratio_largest_to_plasmid"] = round(first_largest_corrected_px * nm_per_px / full_nm, 2)
    out["ratio_total_to_plasmid"] = round(first_total_corrected_px * nm_per_px / full_nm, 2)
    return out


def trend(df: pd.DataFrame, col: str, n_boot: int = 1000, seed: int = 1) -> dict:
    x, y = df["rank"].to_numpy(float), df[col].to_numpy(float)
    ts = stats.theilslopes(y, x)
    rng = np.random.default_rng(seed)
    sl = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        sl.append(stats.theilslopes(y[i], x[i])[0])
    rho = stats.spearmanr(x, y)
    return {"theil_sen_slope_per_frame": float(ts[0]), "ci_low": float(np.percentile(sl, 2.5)),
            "ci_high": float(np.percentile(sl, 97.5)), "spearman_rho": float(rho.statistic), "p": float(rho.pvalue)}


def figures(df: pd.DataFrame, frag: pd.DataFrame, sc: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"wspace": 0.45})
    ax = axs[0]
    ax.plot(df["rank"], df["total_corrected_px"], marker="o", ms=2.5, lw=1, color="#2255c4", label="total DNA in view")
    ax.plot(df["rank"], df["largest_corrected_px"], marker="o", ms=2.5, lw=1, color="#d9534f", label="largest piece")
    ax.set_xlabel("frame index (provisional order)")
    ax.set_ylabel("contour length (px, bias-corrected)")
    ax2 = ax.secondary_yaxis("right", functions=(lambda v: v * sc["estimated_nm_per_px"], lambda v: v / sc["estimated_nm_per_px"]))
    ax2.set_ylabel("nm (estimated scale)")
    ax.axhline(sc["plasmid_full_contour_nm"] / sc["estimated_nm_per_px"], color="grey", ls="--", lw=1)
    ax.text(df["rank"].max(), sc["plasmid_full_contour_nm"] / sc["estimated_nm_per_px"], " 1483 nm (whole pBR322)", va="bottom", ha="right", fontsize=7, color="grey")
    ax.legend(fontsize=7)
    ax.set_title("Contour length over time", fontsize=9, loc="left")
    ax = axs[1]
    picks = [(int(df["rank"].min()), "#2255c4"), (int(df["rank"].median()), "#e0a030"), (int(df["rank"].max()), "#d9534f")]
    for i, (r, c) in enumerate(picks):
        v = frag.loc[frag["rank"] == r, "length_corrected_px"].to_numpy()
        ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(v)), v, s=22, color=c, alpha=0.85)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"first (frame {picks[0][0]})", f"middle (frame {picks[1][0]})", f"last (frame {picks[2][0]})"], fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("piece length (px, log scale)")
    ax.set_title("Fragment lengths at three moments", fontsize=9, loc="left")
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.25, lw=0.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_contour_length_time.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def summary_text(bias: dict, sc: dict, df: pd.DataFrame, tr_total: dict, tr_largest: dict) -> str:
    a, z = df.iloc[0], df.iloc[-1]
    rho_u = stats.spearmanr(df["rank"], df["length_below_min_not_counted_px"])
    L = ["## Heading 4: contour length over time (English summary)", "",
         f"- Measurement-method bias (with synthetic curves, {bias['n_curves']} curves): {bias['reading']} "
         f"(95% interval {bias['ratio_95_low']:.3f}–{bias['ratio_95_high']:.3f}). All lengths are corrected for this bias and the raw values are also provided.",
         f"- In the first frame the longest piece (corrected) is {a.largest_corrected_px:.0f} px and the total DNA {a.total_corrected_px:.0f} px; "
         f"in the last frame the longest is {z.largest_corrected_px:.0f} px and the total {z.total_corrected_px:.0f} px; number of pieces {int(a.n_pieces)} → {int(z.n_pieces)}.",
         f"- Trend (Theil–Sen, bootstrap 95%): total length {tr_total['theil_sen_slope_per_frame']:.2f} px per frame "
         f"[{tr_total['ci_low']:.2f}, {tr_total['ci_high']:.2f}]; longest piece {tr_largest['theil_sen_slope_per_frame']:.2f} px "
         f"[{tr_largest['ci_low']:.2f}, {tr_largest['ci_high']:.2f}]. The time order is provisional; this slope is a 'per frame' value.",
         "- **SCALE CONSISTENCY CHECK:** at the estimated scale (%.2f nm/pixel) the longest piece in the first frame is %.0f nm and the total DNA %.0f nm; "
         "the whole pBR322 can only be %.0f nm (4363 bp × 0.34 nm). The longest piece is %.2f times the plasmid and the total %.2f times."
         % (sc["estimated_nm_per_px"], sc["frame1_largest_nm_at_estimated_scale"], sc["frame1_total_nm_at_estimated_scale"],
            sc["plasmid_full_contour_nm"], sc["ratio_largest_to_plasmid"], sc["ratio_total_to_plasmid"])]
    if sc["ratio_largest_to_plasmid"] > 1.15:
        L.append("  This is too long for a single pBR322. Possible reasons: (1) the image scale is smaller than the 1000 nm estimate "
                 f"(≈{sc['implied_nm_per_px_if_largest_is_whole_plasmid']} nm/pixel if the longest piece is the whole plasmid), "
                 "(2) several molecules were segmented as one piece, (3) the strand is not straight but coiled and the length is overcounted. "
                 "A firm scale needs the raw file or the real pixel size from the supervisor; the nm columns are therefore only a rough estimate.")
    elif sc["ratio_largest_to_plasmid"] < 0.85:
        L.append("  The longest piece is shorter than the whole plasmid: either it is already fragmented or the image contains only part of the plasmid.")
    else:
        L.append("  The longest piece looks consistent with the full plasmid length.")
    L.append(f"- Small pieces not counted (shorter than 20 pixels): on average {df['n_pieces_below_min_not_counted'].mean():.0f} pieces per frame, "
             f"{df['length_below_min_not_counted_px'].mean():.0f} px in total (at most {df['length_below_min_not_counted_px'].max():.0f} per frame). "
             f"Trend over time: Spearman ρ = {rho_u.statistic:.2f} (p = {rho_u.pvalue:.2f}). "
             "If their number and total do not increase systematically with time, they count as a constant noise floor and do not distort the trend; "
             "real short pieces (20–49 px) are already counted.")
    L.append("- Results in pixels (longest piece, total length, number of pieces) are independent of the scale and are the main finding; nm is only an estimate.")
    return "\n".join(L) + "\n"


def run(outputs: Path, config: Path = cfgmod.DEFAULT_CONFIG_PATH, s: LengthSettings | None = None) -> dict:
    s = s or LengthSettings()
    p2 = outputs / "phase2"
    p4 = outputs / "phase4" / "frame_measurements.csv"
    for f in (p2 / "frames.csv", p4):
        if not f.is_file():
            raise FileNotFoundError(f"{f} not found. Run phases 2 to 4 first (run_all.sh).")
    frames = pd.read_csv(p2 / "frames.csv").sort_values("rank")
    scale_by_rank = {int(r.rank): float(r.nm_per_pixel_used) for r in frames.itertuples()}
    synth = synthetic_bias(s)
    bias = bias_summary(synth)
    # first pass without anchor, to get frame-1 numbers for the cross-check
    df0, _ = length_table(outputs, frames, bias, s, scale_by_rank, None)
    sc = scale_check(float(df0.iloc[0]["largest_corrected_px"]), float(df0.iloc[0]["total_corrected_px"]),
                     scale_by_rank[int(frames.iloc[0]["rank"])])
    anchor = sc["implied_nm_per_px_if_largest_is_whole_plasmid"]
    df, frag = length_table(outputs, frames, bias, s, scale_by_rank, anchor)
    tr_total, tr_largest = trend(df, "total_corrected_px"), trend(df, "largest_corrected_px")
    out = outputs / "lengths"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "length_by_frame.csv", index=False)
    frag.to_csv(out / "fragment_lengths.csv", index=False)
    synth.round(4).to_csv(out / "length_estimator_bias.csv", index=False)
    (out / "summary.md").write_text(summary_text(bias, sc, df, tr_total, tr_largest), encoding="utf-8")
    (out / "parameters.json").write_text(json.dumps({"settings": asdict(s), "bias": bias, "scale_check": sc,
                                                     "trend_total": tr_total, "trend_largest": tr_largest,
                                                     "notes": "nm columns depend on a still-estimated scale; plasmid-anchor column assumes the largest piece of frame 1 is the whole plasmid (a hypothesis, not a fact)"},
                                                    indent=2), encoding="utf-8")
    figures(df, frag, sc, out)
    return {"bias": bias, "scale_check": sc, "table": df, "trend_total": tr_total, "trend_largest": tr_largest}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Heading 4: contour length over time.")
    p.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    p.add_argument("--config", type=Path, default=cfgmod.DEFAULT_CONFIG_PATH)
    a = p.parse_args(argv)
    try:
        r = run(a.outputs, a.config)
    except (FileNotFoundError, ValueError, cfgmod.ConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(r["bias"])
    print(r["scale_check"])
    print(r["trend_total"])
    print(r["trend_largest"])
    print(f"Results: {a.outputs / 'lengths'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
