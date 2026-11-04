"""Phase 8: when does the big fragmentation step happen? Change-point analysis and literature context.

Supervisor's question: "pinpoint exactly when ... fragmentation events occur".
The longest-piece length and the continuity fraction fall from ~550 px to ~170 px, but the series
is noisy (frame to frame the longest piece can jump back). This module asks, with explicit
statistics, where a single step is best placed and how well it is determined:

  * step model (one change point) vs constant, linear, two-step and logistic models, by BIC
  * block-bootstrap interval for the change point (blocks keep neighbouring-frame correlation)
  * the same change point recomputed for each of the 36 Phase 4 threshold settings
  * the frame converted to an ESTIMATED time range (scan time 31 s; the exact times are unknown)
  * a side-by-side with values from the literature table (knowledge/evidence.csv), which are
    UNVERIFIED and refer to time after enzyme addition, a time that is not known for our series.

Pixels and frame index only. Descriptive: one molecule, one experiment, no control.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .inventory import PROJECT_ROOT

DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"
DEFAULT_KNOWLEDGE = PROJECT_ROOT / "knowledge"
SERIES = {"largest_object_px": "largest_px", "continuity_fraction": "continuity_fraction"}
MIN_SEG = 3            # frames needed on each side of a step
BLOCK = 5              # bootstrap block length (frames)


# ------------------------------------------------------------------ models

def step_fit(y: np.ndarray, min_seg: int = MIN_SEG) -> dict:
    """Best single step: index k (0-based) = first element of the second level."""
    n = len(y)
    best = None
    for k in range(min_seg, n - min_seg + 1):
        a, b = y[:k], y[k:]
        sse = float(((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum())
        if best is None or sse < best["sse"]:
            best = {"k": k, "sse": sse, "before": float(a.mean()), "after": float(b.mean())}
    return best


def two_step_fit(y: np.ndarray, min_seg: int = MIN_SEG) -> dict:
    n = len(y)
    best = None
    for k1 in range(min_seg, n - 2 * min_seg + 1):
        for k2 in range(k1 + min_seg, n - min_seg + 1):
            parts = (y[:k1], y[k1:k2], y[k2:])
            sse = float(sum(((p - p.mean()) ** 2).sum() for p in parts))
            if best is None or sse < best["sse"]:
                best = {"k1": k1, "k2": k2, "sse": sse}
    return best


def _logistic(x, top, bottom, x0, w):
    return bottom + (top - bottom) / (1.0 + np.exp((x - x0) / w))


def logistic_fit(x: np.ndarray, y: np.ndarray) -> dict | None:
    try:
        p0 = [float(y[:5].mean()), float(y[-5:].mean()), float(np.median(x)), max(len(x) / 20, 1.0)]
        popt, _ = curve_fit(_logistic, x, y, p0=p0, maxfev=20000)
        sse = float(((y - _logistic(x, *popt)) ** 2).sum())
        return {"sse": sse, "x0": float(popt[2]), "width": float(abs(popt[3]))}
    except (RuntimeError, ValueError):
        return None


def bic(n: int, sse: float, n_params: int) -> float:
    return float(n * np.log(max(sse, 1e-12) / n) + n_params * np.log(n))


def compare_models(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    n = len(y)
    rows = [{"model": "constant", "n_params": 1, "sse": float(((y - y.mean()) ** 2).sum())}]
    slope, icpt = np.polyfit(x, y, 1)
    rows.append({"model": "linear", "n_params": 2, "sse": float(((y - (slope * x + icpt)) ** 2).sum())})
    s1 = step_fit(y)
    rows.append({"model": "one_step", "n_params": 3, "sse": s1["sse"]})
    s2 = two_step_fit(y)
    rows.append({"model": "two_steps", "n_params": 5, "sse": s2["sse"]})
    lg = logistic_fit(x, y)
    if lg:
        rows.append({"model": "logistic", "n_params": 4, "sse": lg["sse"]})
    df = pd.DataFrame(rows)
    df["bic"] = [round(bic(n, r.sse, r.n_params), 1) for r in df.itertuples()]
    df["delta_bic_vs_best"] = (df["bic"] - df["bic"].min()).round(1)
    return df.sort_values("bic").reset_index(drop=True)


def block_bootstrap_step(y: np.ndarray, n_boot: int = 1000, block: int = BLOCK, seed: int = 1) -> np.ndarray:
    """Change-point indices from residual block bootstrap around the fitted step."""
    rng = np.random.default_rng(seed)
    fit = step_fit(y)
    n = len(y)
    fitted = np.where(np.arange(n) < fit["k"], fit["before"], fit["after"])
    res = y - fitted
    out = []
    n_blocks = int(np.ceil(n / block))
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        r = np.concatenate([res[s:s + block] for s in starts])[:n]
        out.append(step_fit(fitted + r)["k"])
    return np.array(out)


# ------------------------------------------------------------------ analysis

def frame_to_time(frames: pd.DataFrame, rank: float) -> tuple[float, float]:
    """Earliest and latest possible acquisition time (s) of a frame (estimated, see config)."""
    r = frames.set_index("rank")
    rank = int(min(max(round(rank), r.index.min()), r.index.max()))
    return float(r.loc[rank, "time_earliest_s"]), float(r.loc[rank, "time_latest_s"])


def analyse_series(name: str, ranks: np.ndarray, y: np.ndarray, frames: pd.DataFrame, n_boot: int = 1000) -> dict:
    fit = step_fit(y)
    cp_rank = int(ranks[fit["k"]])
    boots = block_bootstrap_step(y, n_boot=n_boot)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ci = (int(ranks[int(lo)]), int(ranks[int(hi)]))
    t_lo = frame_to_time(frames, ci[0])[0]
    t_hi = frame_to_time(frames, ci[1])[1]
    models = compare_models(ranks.astype(float), y)
    t2 = two_step_fit(y)
    lv = [float(y[a:b].mean()) for a, b in ((0, t2["k1"]), (t2["k1"], t2["k2"]), (t2["k2"], len(y)))]
    bic_of = lambda m: float(models.loc[models["model"] == m, "bic"].iloc[0])
    return {"series": name, "two_step_rank_1": int(ranks[t2["k1"]]), "two_step_rank_2": int(ranks[t2["k2"]]),
            "two_step_levels": [round(v, 3) for v in lv],
            "delta_bic_two_steps_vs_one_step": round(bic_of("one_step") - bic_of("two_steps"), 1), "change_point_rank": cp_rank, "ci95_rank_low": ci[0], "ci95_rank_high": ci[1],
            "level_before": round(fit["before"], 3), "level_after": round(fit["after"], 3),
            "best_model_by_bic": models.iloc[0]["model"],
            "delta_bic_one_step_vs_constant": float(models.loc[models["model"] == "one_step", "bic"].iloc[0]
                                                    - models.loc[models["model"] == "constant", "bic"].iloc[0]),
            "time_range_min_estimated": f"{t_lo / 60:.0f} to {t_hi / 60:.0f}",
            "time_status": "estimated (scan 31 s; exact times unknown)", "models": models}


def sensitivity_change_points(sens: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (setting, g) in sens.groupby("setting"):
        g = g.sort_values("rank")
        for col, label in (("largest_px", "largest_object_px"), ("continuity_fraction", "continuity_fraction")):
            y = g[col].to_numpy(float)
            if np.isnan(y).any() or len(y) < 2 * MIN_SEG + 1:
                continue
            k = step_fit(y)["k"]
            rows.append({"setting": setting, "series": label, "change_point_rank": int(g["rank"].iloc[k])})
    return pd.DataFrame(rows)


def literature_table(evidence_csv: Path, our: dict, persistent_ranks: list[int]) -> pd.DataFrame:
    if not evidence_csv.is_file():
        return pd.DataFrame()
    ev = pd.read_csv(evidence_csv).set_index("evidence_id")
    want = [("E006", "çıplak DNA: bozunma başlangıcı"), ("E007", "çıplak DNA: tam parçalanma"),
            ("E010", "G4:DNA 0.5:1 ve 1:1: bozunma başlangıcı"), ("E008", "ömür (çıplak / 0.5:1)"),
            ("E009", "ömür (1:1)")]
    rows = []
    for eid, label in want:
        if eid in ev.index:
            r = ev.loc[eid]
            rows.append({"source": "Abdelhady 2003 (kitaplık, DOĞRULANMADI)", "evidence_id": eid, "what_tr": label,
                         "value": f"{r['value']} {r['unit']}", "quote": r["quote"], "verified": r["verified"]})
    rows.append({"source": "bizim seri (tahmini)", "evidence_id": "", "what_tr": "büyük basamak (en uzun parça)",
                 "value": f"kare {our['change_point_rank']} (aralık {our['ci95_rank_low']}-{our['ci95_rank_high']}), "
                          f"tahmini {our['time_range_min_estimated']} dk",
                 "quote": f"kalıcı kopma adayı kareleri: {persistent_ranks}",
                 "verified": "tahmin; DNase I ekleme zamanı bilinmiyor"})
    return pd.DataFrame(rows)


def summary_text(res: dict, sens_sum: dict, persistent_ranks: list[int]) -> str:
    a = res["largest_object_px"]
    b = res["continuity_fraction"]
    lines = [
        "## Faz 8: kopma ne zaman? (kırılma noktası analizi, Türkçe özet)",
        "",
        f"- En uzun parça serisinde en iyi tek basamak **kare {a['change_point_rank']}** civarında "
        f"(%95 bootstrap aralığı kare {a['ci95_rank_low']} ile {a['ci95_rank_high']}); düzey {a['level_before']:.0f} px'ten {a['level_after']:.0f} px'e. "
        f"Süreklilik oranı için kare {b['change_point_rank']} (aralık {b['ci95_rank_low']}-{b['ci95_rank_high']}).",
        f"- İki basamaklı model: kare {a['two_step_rank_1']} ve kare {a['two_step_rank_2']} civarında basamak, düzeyler "
        f"{a['two_step_levels'][0]:.0f} → {a['two_step_levels'][1]:.0f} → {a['two_step_levels'][2]:.0f} px. İki basamağın tek basamağa göre "
        f"BIC üstünlüğü {a['delta_bic_two_steps_vs_one_step']:.1f} ({'belirgin değil, ikisi ayırt edilemiyor' if a['delta_bic_two_steps_vs_one_step'] < 6 else 'belirgin'}); "
        f"tek basamağın sabit modele göre BIC farkı {a['delta_bic_one_step_vs_constant']:.0f} (negatif = basamak çok daha iyi).",
        f"- 36 eşik ayarında aynı kırılma noktası: ortanca kare {sens_sum['median']:.0f}, aralık {sens_sum['min']}-{sens_sum['max']}; "
        f"ayarların %{100 * sens_sum['share_within_5']:.0f}'i ana sonuca ±5 kare içinde.",
        f"- Tahmini zaman aralığı: {a['time_range_min_estimated']} dakika (31 sn/tarama varsayımı; tekrarlanan taramalar yüzünden gerçek zaman bilinmiyor).",
        f"- Kalıcı kopma adayı kareleri: {persistent_ranks}.",
        "- Literatür karşılaştırması (Abdelhady 2003, kitaplıkta doğrulanmamış): o çalışmada süre enzim eklendikten sonra ölçülüyor; "
        "bizde enzim ekleme zamanı bilinmediği için bu yalnızca büyüklük sırası kıyasıdır, sonuç sayılmaz.",
        "- Seri gürültülü: tek karedeki değerlere güvenilmez (ilk 49 karede en uzun parça defalarca düşüp yeniden yükseliyor). "
        "Basamak, bu dalgalanmanın üzerindeki kalıcı düşüşü yakalıyor.",
    ]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ figure

def figure(m: pd.DataFrame, res: dict, sens_cp: pd.DataFrame, persistent_ranks: list[int], out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    for ax, (col, label) in zip(axs[:2], (("largest_object_px", "longest piece (skeleton px)"),
                                          ("continuity_fraction", "continuity fraction"))):
        r = res[col]
        ax.plot(m["rank"], m[col], color="#2255c4", lw=1, marker="o", ms=2.5)
        ax.hlines(r["level_before"], m["rank"].min(), r["change_point_rank"], color="#d9534f", lw=2)
        ax.hlines(r["level_after"], r["change_point_rank"], m["rank"].max(), color="#d9534f", lw=2)
        ax.axvspan(r["ci95_rank_low"], r["ci95_rank_high"], color="#d9534f", alpha=0.15,
                   label=f"95% bootstrap interval, step at {r['change_point_rank']}")
        for pr in persistent_ranks:
            ax.axvline(pr, color="k", ls=":", lw=0.8)
        ax.set_xlabel("frame index (provisional order)")
        ax.set_ylabel(label)
        ax.legend(fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25, lw=0.5)
    ax = axs[2]
    if len(sens_cp):
        for series, color in (("largest_object_px", "#2255c4"), ("continuity_fraction", "#c47a22")):
            g = sens_cp[sens_cp["series"] == series]
            ax.hist(g["change_point_rank"], bins=range(int(sens_cp["change_point_rank"].min()),
                                                      int(sens_cp["change_point_rank"].max()) + 3), alpha=0.6,
                    color=color, label=f"{series} (36 settings)")
    ax.set_xlabel("change-point frame")
    ax.set_ylabel("threshold settings")
    ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_changepoint.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ run

def run(outputs: Path, knowledge: Path, n_boot: int = 1000) -> dict:
    m_path = outputs / "phase4" / "frame_measurements.csv"
    s_path = outputs / "phase4" / "sensitivity.csv"
    for f in (m_path, s_path, outputs / "phase2" / "frames.csv", outputs / "phase3" / "events.csv"):
        if not f.is_file():
            raise FileNotFoundError(f"{f} not found. Run phases 2 to 4 first (run_all.sh).")
    m = pd.read_csv(m_path).sort_values("rank")
    frames = pd.read_csv(outputs / "phase2" / "frames.csv")
    events = pd.read_csv(outputs / "phase3" / "events.csv")
    persistent = sorted(int(r) for r in events[(events["event"] == "split_candidate")
                                               & (events["persistence"] == "persistent")]["rank_to"])
    ranks = m["rank"].to_numpy()
    res = {c: analyse_series(c, ranks, m[c].to_numpy(float), frames, n_boot) for c in SERIES}
    sens_cp = sensitivity_change_points(pd.read_csv(s_path))
    main_cp = res["largest_object_px"]["change_point_rank"]
    g = sens_cp[sens_cp["series"] == "largest_object_px"]["change_point_rank"]
    sens_sum = {"median": float(g.median()), "min": int(g.min()), "max": int(g.max()),
                "share_within_5": float((np.abs(g - main_cp) <= 5).mean()), "n_settings": int(len(g))}
    lit = literature_table(knowledge / "evidence.csv", res["largest_object_px"], persistent)
    out = outputs / "phase8"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: (str(v) if isinstance(v, list) else v) for k, v in r.items() if k != "models"}
                  for r in res.values()]).to_csv(
        out / "changepoint.csv", index=False)
    pd.concat([r["models"].assign(series=c) for c, r in res.items()]).to_csv(out / "model_comparison_changepoint.csv", index=False)
    sens_cp.to_csv(out / "changepoint_sensitivity.csv", index=False)
    lit.to_csv(out / "literature_comparison.csv", index=False)
    (out / "summary_tr.md").write_text(summary_text(res, sens_sum, persistent), encoding="utf-8")
    (out / "parameters.json").write_text(json.dumps({
        "min_segment_frames": MIN_SEG, "bootstrap_block_frames": BLOCK, "n_bootstrap": n_boot,
        "sensitivity_summary": sens_sum, "persistent_split_ranks": persistent,
        "notes": "frame order provisional; time estimated; literature values unverified; descriptive only"},
        indent=2), encoding="utf-8")
    figure(m, res, sens_cp, persistent, out)
    return {"series": res, "sensitivity": sens_sum, "literature": lit, "persistent": persistent}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 8: change-point analysis of the fragmentation series.")
    p.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    p.add_argument("--knowledge", type=Path, default=DEFAULT_KNOWLEDGE)
    a = p.parse_args(argv)
    try:
        res = run(a.outputs, a.knowledge)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for name, r in res["series"].items():
        print(f"{name}: step at frame {r['change_point_rank']} (95% {r['ci95_rank_low']}-{r['ci95_rank_high']}), "
              f"{r['level_before']} -> {r['level_after']}, best model {r['best_model_by_bic']}, "
              f"estimated {r['time_range_min_estimated']} min")
    print("36-setting sensitivity:", res["sensitivity"])
    print(f"Results: {a.outputs / 'phase8'}")
    print("Descriptive only: order provisional, time estimated, literature values unverified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
