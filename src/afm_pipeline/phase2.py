"""Phase 2 runner: preprocessing + DNA segmentation for all images, with review overlays.

Outputs (outputs/phase2/):
    overlays/NN_<name>.png     4 panels: original | preprocessed | mask + skeleton | features
    masks/, skeletons/         binary PNGs (cropped frame)
    objects.csv                one row per object per image (CANDIDATE classes only)
    bright_features.csv        one row per bright feature
    frames.csv                 one row per image
    evaluation.csv             only if manual annotations exist (annotations/<stem>.png)
    parameters.json            every setting used (for reproducibility)
    review_begin_middle_end.png/.pdf   multi-panel figure for manual review
Everything is in pixel units on JPG intensity; nm columns use the ESTIMATED scale and say so.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from . import config as cfgmod
from .corrections import apply_corrections, evaluate, load_annotation, load_corrections
from .dna import DnaSettings, analyse, dna_mask, load_dna_settings, preprocess
from .inventory import DEFAULT_DATA_DIR, PROJECT_ROOT, find_image_files
from .order import build_provisional_order
from .segment import load_rgb

KIND_COLOUR = {"closed_loop_candidate": "#00e5ff", "open_candidate": "#ffd600", "small_object": "#ff4081"}
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "phase2"
ANNOTATION_DIR = PROJECT_ROOT / "annotations"


def _outline(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi
    return mask & ~ndi.binary_erosion(mask)


def draw_panels(axes, rgb_name: str, pre, res, title: str) -> None:
    axes[0].imshow(pre.rgb)
    axes[0].set_title(title + "\noriginal (cropped)", fontsize=8, loc="left")
    axes[1].imshow(pre.z, cmap="gray", vmin=-3, vmax=12)
    axes[1].set_title("preprocessed (noise units)", fontsize=8)
    ov = np.ones(res.mask.shape + (3,)) * 0.08
    ov[res.mask] = [0.35, 0.35, 0.35]
    for obj in res.objects:
        sel = res.labels == obj["object_id"]
        col = np.array(matplotlib.colors.to_rgb(KIND_COLOUR[obj["kind"]]))
        ov[_outline(sel)] = col
    ov[res.skeleton] = [1, 1, 1]
    ov[res.endpoints] = [1, 0, 0]
    ov[res.branchpoints] = [0.2, 0.4, 1]
    axes[2].imshow(ov)
    axes[2].set_title("mask outline (cyan loop / yellow open / pink small),\nskeleton white, ends red, branches blue", fontsize=7)
    axes[3].imshow(pre.rgb)
    for f in res.bright:
        axes[3].add_patch(plt.Circle((f["centroid_x_px"], f["centroid_y_px"]), max(f["radius_px"] * 1.8, 4),
                                     fill=False, ec="cyan", lw=1.2))
    axes[3].set_title(f"bright features: {len(res.bright)}", fontsize=8)
    for a in axes:
        a.axis("off")


def run(data_dir: Path, out_dir: Path, settings: DnaSettings, project: cfgmod.ProjectConfig,
        corrections_path: Path | None = None, annotation_dir: Path | None = None) -> dict:
    files, _ = find_image_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No JPG images in {data_dir}")
    order = build_provisional_order([f.name for f in files])
    by_name = {f.name: f for f in files}
    for sub in ("overlays", "masks", "skeletons"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    corrections = load_corrections(corrections_path) if corrections_path else {}
    n = len(order)
    frames, objects, bright, evals, cache = [], [], [], [], {}

    for row in order.itertuples():
        name, rank = row.filename, int(row.chronological_rank)
        rgb = load_rgb(by_name[name])
        if rgb is None:
            raise ValueError(f"Could not read image: {name}")
        pre = preprocess(rgb, settings)
        auto = dna_mask(pre.z, settings)
        mask, source = auto, "automatic"
        if name in corrections:
            mask, source = apply_corrections(auto, corrections[name], pre.crop_box), "corrected"
        res = analyse(pre, settings, mask)
        stem = Path(name).stem
        h, w = rgb.shape[:2]
        npp = None
        if project.scale.usable:
            npp = cfgmod.nm_per_pixel_for_image(project, w, h)
        t_lo, t_hi = cfgmod.time_bounds_s(rank, n, project.time)
        kinds = [o["kind"] for o in res.objects]
        total_len = sum(o["skeleton_length_px"] for o in res.objects)
        frames.append({
            "rank": rank, "filename": name, "width_px": w, "height_px": h,
            "crop_top": pre.crop_box[0], "crop_bottom": pre.crop_box[1],
            "crop_left": pre.crop_box[2], "crop_right": pre.crop_box[3],
            "noise_level": round(pre.noise, 3), "mask_source": source,
            "n_objects": len(kinds),
            "n_closed_loop_candidates": kinds.count("closed_loop_candidate"),
            "n_open_candidates": kinds.count("open_candidate"),
            "n_small_objects": kinds.count("small_object"),
            "n_loop_if_gaps_bridged": sum(1 for o in res.objects
                                          if o["kind"] != "closed_loop_candidate"
                                          and o["loop_bridge_radius_px"] is not None
                                          and not pd.isna(o["loop_bridge_radius_px"])),
            "n_scan_line_suspects": sum(o["scan_line_suspect"] for o in res.objects),
            "mask_area_px": int(res.mask.sum()),
            "skeleton_length_px_total": round(total_len, 1),
            "largest_object_skeleton_px": max([o["skeleton_length_px"] for o in res.objects], default=0.0),
            "n_bright_features": len(res.bright),
            "nm_per_pixel_used": None if npp is None else round(npp, 4),
            "scale_status": project.scale.status,
            "skeleton_length_nm_estimated": None if npp is None else round(total_len * npp, 1),
            "time_earliest_s": t_lo, "time_latest_s": t_hi, "time_status": project.time.status,
        })
        for o in res.objects:
            objects.append({"rank": rank, "filename": name, **o})
        for f in res.bright:
            bright.append({"rank": rank, "filename": name, **f})
        Image.fromarray((res.mask * 255).astype(np.uint8)).save(out_dir / "masks" / f"{rank:02d}_{stem}_mask.png")
        Image.fromarray((res.skeleton * 255).astype(np.uint8)).save(out_dir / "skeletons" / f"{rank:02d}_{stem}_skeleton.png")
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
        draw_panels(axes, name, pre, res, f"#{rank:02d}  {name}   ({source} mask; provisional order)")
        fig.tight_layout()
        fig.savefig(out_dir / "overlays" / f"{rank:02d}_{stem}.png", dpi=110)
        plt.close(fig)
        cache[rank] = (name, pre, res)

        if annotation_dir is not None:
            ann = Path(annotation_dir) / f"{stem}.png"
            if ann.is_file():
                score = evaluate(auto, load_annotation(ann, pre.crop_box))
                evals.append({"rank": rank, "filename": name, **score})

    pd.DataFrame(frames).to_csv(out_dir / "frames.csv", index=False)
    pd.DataFrame(objects).to_csv(out_dir / "objects.csv", index=False)
    pd.DataFrame(bright).to_csv(out_dir / "bright_features.csv", index=False)
    if evals:
        pd.DataFrame(evals).to_csv(out_dir / "evaluation.csv", index=False)
    params = {
        "run_label": project.run_label,
        "dna_settings": asdict(settings),
        "scale": {"status": project.scale.status, "frame_size_nm": project.scale.frame_size_nm,
                  "source": project.scale.source},
        "time": {"status": project.time.status, "frame_interval_s": project.time.frame_interval_s,
                 "total_duration_s": project.time.total_duration_s, "source": project.time.source},
        "notes": "pixel units on JPG intensity; not AFM height; classes are candidates only",
    }
    (out_dir / "parameters.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    _review_figure(cache, out_dir)
    return {"frames": pd.DataFrame(frames), "objects": pd.DataFrame(objects), "n": n,
            "evaluated": len(evals)}


def _review_figure(cache: dict, out_dir: Path) -> None:
    ranks = sorted(cache)
    picks = sorted({ranks[0], ranks[len(ranks) // 4], ranks[len(ranks) // 2],
                    ranks[(3 * len(ranks)) // 4], ranks[-1]})
    fig, axes = plt.subplots(len(picks), 4, figsize=(14, 3.6 * len(picks)))
    for i, rk in enumerate(picks):
        name, pre, res = cache[rk]
        draw_panels(axes[i], name, pre, res, f"#{rk:02d}  {name}")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"review_begin_middle_end.{ext}", dpi=200 if ext == "png" else None)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 2: preprocessing and DNA segmentation with overlays.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--config", type=Path, default=cfgmod.DEFAULT_CONFIG_PATH)
    p.add_argument("--corrections", type=Path, default=ANNOTATION_DIR / "corrections.csv")
    p.add_argument("--annotations", type=Path, default=ANNOTATION_DIR)
    a = p.parse_args(argv)
    try:
        project = cfgmod.load_config(a.config)
        settings = load_dna_settings(a.config)
        out = run(a.data_dir, a.out_dir, settings, project, a.corrections, a.annotations)
    except (cfgmod.ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    fr = out["frames"]
    print(f"Images processed: {out['n']}   (run label: {project.run_label})")
    print(f"Objects per image: median {fr.n_objects.median():.0f}; closed-loop candidates in "
          f"{(fr.n_closed_loop_candidates > 0).sum()} images; bright features median {fr.n_bright_features.median():.0f}")
    print(f"Manual annotations evaluated: {out['evaluated']}")
    print(f"Results: {a.out_dir}")
    print("Candidate classes only: nothing here confirms DNA identity or cleavage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
