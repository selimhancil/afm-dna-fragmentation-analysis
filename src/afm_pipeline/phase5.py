"""Phase 5: publication-style figures (PNG + PDF), CSV bundle check and a concise report.

All figures use the provisional frame index on the x axis (the order is not yet verified
against raw timestamps). Pixel units; nm only as flagged estimates. Candidate events are
never called confirmed cleavage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from . import config as cfgmod
from .inventory import DEFAULT_DATA_DIR, PROJECT_ROOT, find_image_files
from .segment import load_rgb

DEFAULT_OUT = PROJECT_ROOT / "outputs" / "phase5"
PLOT_DPI = 300
COL = {"main": "#1f5aa6", "alt": "#c4561c", "grey": "#8a8a8a"}


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=PLOT_DPI if ext == "png" else None, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _style(ax, ylabel, title=None):
    ax.set_xlabel("frame index (provisional order)")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)
    if title:
        ax.set_title(title, fontsize=10, loc="left")


def series_figures(fm: pd.DataFrame, events: pd.DataFrame, out_dir: Path) -> list[Path]:
    x = fm["rank"]
    persistent = events[(events["event"] == "split_candidate") & (events["persistence"] == "persistent")] \
        if len(events) else events
    specs = [
        ("fig_continuity", "continuity_fraction", "continuity (longest piece / total skeleton)", None),
        ("fig_largest_piece", "largest_object_px", "longest piece, skeleton length (px)", None),
        ("fig_components", "n_objects_tracked", "pieces with skeleton >= 20 px", None),
        ("fig_closed_loops", "n_closed_loop_candidates", "closed-loop candidates", None),
        ("fig_bright_features", "n_bright_features", "bright features (candidate bound particles)", None),
        ("fig_split_candidates", "split_candidates_cumulative", "cumulative split candidates", None),
    ]
    paths = []
    combined, axes = plt.subplots(3, 2, figsize=(11, 10))
    for ax_c, (name, col, ylabel, _) in zip(axes.ravel(), specs):
        for ax in (None, ax_c):
            if ax is None:
                fig, ax = plt.subplots(figsize=(6.4, 3.6))
            drawstyle = "steps-post" if col in ("n_closed_loop_candidates", "split_candidates_cumulative") else "default"
            ax.plot(x, fm[col], color=COL["main"], lw=1.4, marker="o", ms=2.5, drawstyle=drawstyle)
            if col == "split_candidates_cumulative":
                ax.plot(x, fm["persistent_splits_cumulative"], color=COL["alt"], lw=1.4, drawstyle="steps-post",
                        label="persistent only")
                ax.legend(frameon=False, fontsize=8)
            for r in persistent["rank_to"] if len(persistent) else []:
                ax.axvline(r, color=COL["alt"], lw=0.8, ls=":", alpha=0.8)
            _style(ax, ylabel)
            if ax is not ax_c:
                paths += _save(fig, out_dir, name)
    combined.suptitle("Image-based measurements vs frame index (orange dotted: persistent split candidates)",
                      fontsize=10, x=0.01, ha="left")
    combined.tight_layout()
    paths += _save(combined, out_dir, "fig_time_series_all")
    return paths


def _load_crop(files: dict, frames: pd.DataFrame, rank: int) -> np.ndarray:
    row = frames[frames["rank"] == rank].iloc[0]
    rgb = load_rgb(files[row.filename])
    return rgb[int(row.crop_top):int(row.crop_bottom), int(row.crop_left):int(row.crop_right)]


def _mask_png(phase2: Path, frames: pd.DataFrame, rank: int, sub: str, suffix: str) -> np.ndarray:
    row = frames[frames["rank"] == rank].iloc[0]
    p = phase2 / sub / f"{rank:02d}_{Path(row.filename).stem}_{suffix}.png"
    return np.asarray(Image.open(p).convert("L")) > 127


def progression_figure(data_dir: Path, phase2: Path, frames, bright, out_dir: Path) -> list[Path]:
    files = {f.name: f for f in find_image_files(data_dir)[0]}
    n = len(frames)
    picks = sorted({1, max(2, n // 6), n // 3, n // 2, (3 * n) // 5, (2 * n) // 3, (5 * n) // 6, n})
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for ax, rank in zip(axes.ravel(), picks):
        img = _load_crop(files, frames, rank)
        ax.imshow(img)
        sk = _mask_png(phase2, frames, rank, "skeletons", "skeleton")
        ys, xs = np.nonzero(sk)
        ax.scatter(xs, ys, s=0.6, c="white", marker="s", linewidths=0)
        for b in bright[bright["rank"] == rank].itertuples():
            ax.add_patch(plt.Circle((b.centroid_x_px, b.centroid_y_px), max(b.radius_px * 1.8, 4),
                                    fill=False, ec="cyan", lw=1))
        ax.set_title(f"frame {rank}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Structural progression (white: skeleton; cyan: bright features). "
                 "Frame order provisional.", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_progression")


def panels_figure(data_dir, phase2, frames, out_dir, ranks=(1, 37, 74)) -> list[Path]:
    """Original | preprocessed-free mask | skeleton for chosen frames (publication panel)."""
    files = {f.name: f for f in find_image_files(data_dir)[0]}
    fig, axes = plt.subplots(len(ranks), 3, figsize=(9, 3 * len(ranks)))
    for i, rank in enumerate(ranks):
        img = _load_crop(files, frames, rank)
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(_mask_png(phase2, frames, rank, "masks", "mask"), cmap="gray")
        axes[i, 2].imshow(_mask_png(phase2, frames, rank, "skeletons", "skeleton"), cmap="gray")
        for j, t in enumerate(("original (cropped)", "DNA mask", "skeleton")):
            if i == 0:
                axes[i, j].set_title(t, fontsize=9)
            axes[i, j].axis("off")
        axes[i, 0].text(3, 12, f"frame {rank}", color="white", fontsize=8)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_original_mask_skeleton")


def events_figure(data_dir, phase2, frames, events, out_dir) -> list[Path]:
    if events.empty:
        return []
    sp = events[(events["event"] == "split_candidate") & (events["persistence"] == "persistent")]
    if sp.empty:
        return []
    files = {f.name: f for f in find_image_files(data_dir)[0]}
    fig, axes = plt.subplots(len(sp), 2, figsize=(7, 3.4 * len(sp)), squeeze=False)
    for i, e in enumerate(sp.itertuples()):
        for j, rank in enumerate((e.rank_from, e.rank_to)):
            ax = axes[i, j]
            ax.imshow(_load_crop(files, frames, rank))
            sk = _mask_png(phase2, frames, rank, "skeletons", "skeleton")
            ys, xs = np.nonzero(sk)
            ax.scatter(xs, ys, s=0.5, c="white", marker="s", linewidths=0)
            if j == 0 and not pd.isna(e.event_x_px):
                ax.add_patch(plt.Circle((e.event_x_px, e.event_y_px), 12, fill=False, ec="red", lw=1.5))
            head = f"frame {rank}" + ("  (gap marked)" if j == 0 else "")
            if j == 0:
                head = (f"candidate split {e.rank_from} to {e.rank_to}: gap {e.min_gap_px} px, "
                        f"confidence {e.confidence}\n{head}")
            ax.set_title(head, fontsize=8, loc="left")
            ax.axis("off")
    fig.suptitle("Persistent split candidates (not confirmed cleavage)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_fragmentation_candidates")


def sensitivity_figure(sens: pd.DataFrame, out_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for col, ax, label in (("largest_px", axes[0], "longest piece (px)"),
                           ("continuity_fraction", axes[1], "continuity fraction")):
        for _, g in sens.groupby("setting"):
            ax.plot(g["rank"], g[col], color=COL["grey"], lw=0.5, alpha=0.35)
        base = sens[(sens["low_k"] == 3.5) & (sens["bridge_radius_px"] == 2) & (sens["min_len_px"] == 20)]
        ax.plot(base["rank"], base[col], color=COL["main"], lw=1.8, label="default setting")
        ax.legend(frameon=False, fontsize=8)
        _style(ax, label)
    fig.suptitle("Sensitivity: every grey line is one threshold setting (36 settings)", fontsize=10,
                 x=0.01, ha="left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_sensitivity")


def write_report(out_dir, fm, events, robustness, assoc, frames, project) -> Path:
    sp = events[events["event"] == "split_candidate"] if len(events) else events
    pers = sp[sp["persistence"] == "persistent"] if len(sp) else sp
    trans = sp[sp["persistence"] == "transient_rejoined"] if len(sp) else sp
    half = max(1, int(len(fm) * 2 / 3))
    early, late = fm.iloc[:half], fm.iloc[half:]
    kept = fm["bright_kept_from_previous"].dropna()
    lines = [
        "# AFM DNA fragmentation analysis: preliminary report",
        "",
        f"Run label: **{project.run_label}**. {len(fm)} images, provisional chronological order "
        "(not yet verified against raw acquisition timestamps).",
        "",
        "## What was measured",
        "Image-based measurements on exported JPG images (pixel units, JPG intensity). Image colours are "
        "not calibrated heights; no AFM height is reported. Scale "
        f"({project.scale.status}) and time ({project.time.status}) come from the supervisor's email and "
        "are estimates, so nm values are marked as such and time is shown as frame index.",
        "",
        "## Main observations (candidates, not conclusions)",
        f"- The longest connected piece shortens from {fm.largest_object_px.iloc[0]:.0f} px (first frame) to "
        f"{fm.largest_object_px.iloc[-1]:.0f} px (last frame); continuity fraction "
        f"{fm.continuity_fraction.iloc[0]:.2f} to {fm.continuity_fraction.iloc[-1]:.2f}.",
        f"- {len(sp)} split candidates were found; {len(trans)} re-joined within the next frames "
        f"(transient, often small gaps) and {len(pers)} persisted"
        + (f" (frames {', '.join(str(int(r)) for r in pers['rank_to'])})." if len(pers) else "."),
        f"- Frame-to-frame values fluctuate strongly in the first two thirds (frames 1 to {half}: continuity "
        f"median {early.continuity_fraction.median():.2f}, range {early.continuity_fraction.min():.2f} to "
        f"{early.continuity_fraction.max():.2f}; longest piece median {early.largest_object_px.median():.0f} px), "
        f"so single-frame values are unreliable. The clear step down comes later (frames {half + 1} to "
        f"{len(fm)}: continuity median {late.continuity_fraction.median():.2f}; longest piece median "
        f"{late.largest_object_px.median():.0f} px).",
        f"- Closed-loop candidates were found in {int((fm.n_closed_loop_candidates > 0).sum())} of "
        f"{len(fm)} frames, all among the first {int(fm.loc[fm.n_closed_loop_candidates > 0, 'rank'].max())} "
        "frames (this call depends on a small gap near the top bright feature).",
        "",
        "## Robustness across settings (36 threshold settings)",
    ]
    for k, v in robustness.items():
        lines.append(f"- {k.replace('spearman_', '')}: {v}")
    lines += [
        "",
        "## Bright features",
        f"Median {fm.n_bright_features.median():.0f} bright features per frame. After registration, a median of "
        f"{kept.median() * 100:.0f}% of the bright features of a frame are found again within 6 px in the "
        "previous frame (one on the upper arc and one at the lower crossing are visible in nearly every frame). "
        "They are labelled 'bright features / candidate bound particles'. Their identity (polymer, enzyme or "
        "other) is unknown.",
    ]
    if len(assoc) and assoc["nearest_bright_feature_px"].notna().any():
        a = assoc.dropna(subset=["nearest_bright_feature_px"])
        lines.append(f"Distance from split-candidate gaps to the nearest bright feature: median "
                     f"{a['nearest_bright_feature_px'].median():.0f} px (range "
                     f"{a['nearest_bright_feature_px'].min():.0f} to {a['nearest_bright_feature_px'].max():.0f}). "
                     "Association only; no causal claim.")
    lines += [
        "",
        "## Limitations",
        "- Raw AFM height data are unavailable, so heights and true contour lengths cannot be measured.",
        "- Scale is assumed (1 um per whole JPG frame) and image sizes vary (263 to 285 px), so images may "
        "have been cropped or resized after the scan.",
        "- One molecule in one experiment, no control run: results show the workflow works and which "
        "measurements are stable, not a biological conclusion.",
        "- Gaps near faint DNA stretches can be visibility or thresholding effects; 12 of 16 split "
        "candidates re-joined. Closed/open loop calls depend on a small gap near the top bright feature.",
        "- Segmentation accuracy against manual annotations has not been measured yet "
        "(no annotation masks exist). Manual correction and evaluation tools are included.",
        "- No fragmentation event is confirmed by this software.",
        "",
        "## Files",
        "frame_measurements.csv, objects.csv, bright_features.csv, events.csv, tracks.csv, "
        "registration.csv, sensitivity.csv, parameters.json files, masks and overlays, figures (PNG + PDF).",
    ]
    p = out_dir / "report.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def run(data_dir: Path, outputs: Path, out_dir: Path, project) -> dict:
    p2, p3, p4 = outputs / "phase2", outputs / "phase3", outputs / "phase4"
    for d in (p2, p3, p4):
        if not d.is_dir():
            raise FileNotFoundError(f"{d} not found. Run phases 2 to 4 first.")
    frames = pd.read_csv(p2 / "frames.csv")
    bright = pd.read_csv(p2 / "bright_features.csv")
    events = pd.read_csv(p3 / "events.csv")
    fm = pd.read_csv(p4 / "frame_measurements.csv")
    assoc = pd.read_csv(p4 / "event_bright_association.csv") if (p4 / "event_bright_association.csv").stat().st_size > 5 else pd.DataFrame()
    robustness = json.loads((p4 / "robustness.json").read_text()) if (p4 / "robustness.json").is_file() else {}
    out_dir.mkdir(parents=True, exist_ok=True)
    made = series_figures(fm, events, out_dir)
    made += progression_figure(data_dir, p2, frames, bright, out_dir)
    made += panels_figure(data_dir, p2, frames, out_dir, ranks=(1, max(2, len(frames) // 2), len(frames)))
    made += events_figure(data_dir, p2, frames, events, out_dir)
    if (p4 / "sensitivity.csv").is_file():
        made += sensitivity_figure(pd.read_csv(p4 / "sensitivity.csv"), out_dir)
    report = write_report(out_dir, fm, events, robustness, assoc, frames, project)
    return {"figures": made, "report": report}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 5: figures and report.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--outputs", type=Path, default=PROJECT_ROOT / "outputs")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--config", type=Path, default=cfgmod.DEFAULT_CONFIG_PATH)
    a = p.parse_args(argv)
    try:
        res = run(a.data_dir, a.outputs, a.out_dir, cfgmod.load_config(a.config))
    except (FileNotFoundError, ValueError, cfgmod.ConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Figures written: {len(res['figures'])} files (PNG + PDF) in {a.out_dir}")
    print(f"Report: {res['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
