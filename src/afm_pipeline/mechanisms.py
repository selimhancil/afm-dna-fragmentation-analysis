"""Heading 6: which of the three mechanisms fits best? Probabilistic comparison, descriptive only.

H1  the enzyme pushes dendrimers away and then cuts   (splits happen NEAR bright features, which then leave)
H2  dendrimers migrate along the DNA by themselves    (features slide relative to each other)
H3  the enzyme cuts only the remaining naked segments (splits happen FAR from bright features)

Evidence, each recomputed with the strengthened tracks of heading 1:
  * position model on WHERE candidate splits occur relative to bright features (independent / near(R) / far(R)
    mixtures), repeated for three feature sets so that a result that depends on the tracking is visible;
  * Bayes factors: the marginal likelihood averaged over a grid of R and mixing weights, and posterior
    probabilities with equal prior (the prior on R is a choice and is stated);
  * H1 fate check: do features within the near radius of a split disappear more often than usual?
  * H2 spacing check: feature-to-feature spacing (heading 1), conservative combination of two measures.
One molecule, one experiment, no control, unidentified features: no mechanism is confirmed or rejected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

from .inventory import PROJECT_ROOT
from .phase4 import cumulative_shifts
from .phase6 import FeatureSettings, baseline_loss_rate, fate_after, skeleton_common
from .phase7 import EPS_GRID, R_GRID, h1_fate_check, llr_table

DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"
FEATURE_SETS = ("strong_only", "strong_and_weak", "trusted_tracks")
BF_AGAINST = 1.0 / 3.0   # Bayes factor below this in EVERY feature set counts as evidence against a model
SET_TR = {"strong_only": "yalnızca güçlü tespitler (eski)", "strong_and_weak": "güçlü + zayıf tespitler (Başlık 1)",
          "trusted_tracks": "yalnızca güvenilir izler (yüksek/orta güven)"}


def feature_sets(det_strong: pd.DataFrame, det_v2: pd.DataFrame, trk_v2: pd.DataFrame) -> dict[str, pd.DataFrame]:
    good = set(trk_v2.loc[trk_v2["confidence"].isin(["high", "medium"]), "track_id"])
    cols = ["rank", "y_common", "x_common"]
    return {"strong_only": det_strong[cols].copy(),
            "strong_and_weak": det_v2[cols].copy(),
            "trusted_tracks": det_v2[det_v2["track_id"].isin(good)][cols].copy()}


def build_events(ev: pd.DataFrame, feats: pd.DataFrame, phase2: Path, frames: pd.DataFrame, shifts: dict) -> list[dict]:
    """One dict per split candidate: distance to the nearest feature in the frame and the null pool
    (distance to the nearest feature of every DNA pixel of that frame)."""
    out = []
    for e in ev.itertuples():
        g = feats[feats["rank"] == e.rank_from]
        if g.empty:
            continue
        tree = cKDTree(g[["y_common", "x_common"]].to_numpy())
        d = float(tree.query([e.event_y_common, e.event_x_common])[0])
        pts = skeleton_common(phase2, frames, int(e.rank_from), shifts)
        if len(pts) < 20:
            continue
        out.append({"rank_from": int(e.rank_from), "rank_to": int(e.rank_to), "persistence": e.persistence,
                    "min_gap_px": float(e.min_gap_px), "d": d, "pool": tree.query(pts)[0]})
    return out


def bayes_factors(llr: pd.DataFrame, subset: str = "all") -> dict:
    """Marginal likelihood ratio against the independent model, averaging exp(LLR) over the (R, eps) grid
    (uniform prior on the grid), and posterior probabilities with equal prior on the three models."""
    t = llr[llr["subset"] == subset]
    if t.empty:
        return {}
    bf_near = float(np.mean(np.exp(t["LLR_near_vs_independent"].to_numpy(float))))
    bf_far = float(np.mean(np.exp(t["LLR_far_vs_independent"].to_numpy(float))))
    z = 1.0 + bf_near + bf_far
    return {"subset": subset, "n_events": int(t["n_events"].iloc[0]), "BF_near_vs_independent": bf_near,
            "BF_far_vs_independent": bf_far, "P_independent": 1.0 / z, "P_near": bf_near / z, "P_far": bf_far / z}


def fate_frame(ev: pd.DataFrame, det: pd.DataFrame, s: FeatureSettings) -> pd.DataFrame:
    rows = []
    for e in ev.itertuples():
        g = det[det["rank"] == e.rank_from]
        d = np.hypot(g["y_common"] - e.event_y_common, g["x_common"] - e.event_x_common) if len(g) else pd.Series(dtype=float)
        near = g[d <= s.near_event_px]
        fates = [fate_after(det, int(e.rank_from), r.y_common, r.x_common, s) for r in near.itertuples()]
        rows.append({"near_kept_in_place": fates.count("kept_in_place"), "near_moved": fates.count("moved"),
                     "near_lost": fates.count("lost"), "near_unknown": fates.count("unknown_end_of_series")})
    return pd.DataFrame(rows)


def h2_summary(pairs: pd.DataFrame, trk: pd.DataFrame) -> dict:
    ok = pairs[pairs["n_frames"] >= 5]
    dec = ok[ok["verdict"].isin(["spacing_constant", "spacing_changes"])]
    n_ch = int((dec["verdict"] == "spacing_changes").sum())
    out = {"pairs_total": int(len(ok)), "pairs_decided": int(len(dec)), "pairs_constant": int((dec["verdict"] == "spacing_constant").sum()),
           "pairs_changing": n_ch, "pairs_conflicting_or_undetermined": int(len(ok) - len(dec))}
    if len(dec):
        out["changing_fraction_decided"] = round(n_ch / len(dec), 3)
        out["p_more_changes_than_1pct_false_alarm"] = float(stats.binomtest(n_ch, len(dec), 0.01, alternative="greater").pvalue)
    inv = pd.concat([dec.loc[dec["verdict"] == "spacing_changes", "track_a"], dec.loc[dec["verdict"] == "spacing_changes", "track_b"]])
    out["tracks_in_changing_pairs"] = {int(k): int(v) for k, v in inv.value_counts().items()}
    long = trk[(trk["n_detections"] >= 6) & trk["confidence"].isin(["high", "medium"])]
    out["trusted_long_tracks"] = int(len(long))
    out["trusted_long_moved"] = int((long["motion_class"] == "moved").sum())
    out["trusted_long_stationary"] = int((long["motion_class"] == "stationary").sum())
    return out


def _level(lo: float, hi: float) -> str:
    if hi < 0:
        return "worse"
    if lo > 0:
        return "better"
    return "indistinguishable"


def evidence_table(res: dict, bf: dict, fate: dict, h2: dict) -> pd.DataFrame:
    """Per hypothesis: evidence items with direction. Texts are built from the numbers."""
    rows = []
    near = {k: res[k] for k in FEATURE_SETS if k in res}
    levels_near = {k: _level(v["near_lo"], v["near_hi"]) for k, v in near.items()}
    levels_far = {k: _level(v["far_lo"], v["far_hi"]) for k, v in near.items()}
    same_near = len(set(levels_near.values())) == 1
    same_far = len(set(levels_far.values())) == 1
    bf_near_all = [bf[k]["BF_near_vs_independent"] for k in near]
    bf_far_all = [bf[k]["BF_far_vs_independent"] for k in near]
    strong_against_near = same_near and "worse" in levels_near.values() and max(bf_near_all) < BF_AGAINST
    strong_against_far = same_far and "worse" in levels_far.values() and max(bf_far_all) < BF_AGAINST
    rows.append({"hypothesis": "H1", "evidence": "konum (yakın model, R=30 px)",
                 "numbers": "; ".join(f"{SET_TR[k]}: LLR {v['near_llr']:+.1f} [{v['near_lo']:+.1f}, {v['near_hi']:+.1f}]" for k, v in near.items()),
                 "reading": ("tüm özellik kümelerinde bağımsız modelden " + {"worse": "kötü", "better": "iyi", "indistinguishable": "ayırt edilemiyor"}[next(iter(levels_near.values()))]
                             if same_near else "sonuç özellik kümesine göre değişiyor (izleme yöntemine duyarlı)"),
                 "direction": "against" if strong_against_near else ("for" if (same_near and "better" in levels_near.values() and min(bf_near_all) > 3.0) else "neutral")})
    rows.append({"hypothesis": "H1", "evidence": "konum, R ve karışım ağırlığı üzerinden ortalama (Bayes faktörü yakın/bağımsız)",
                 "numbers": "; ".join(f"{SET_TR[k]}: BF {bf[k]['BF_near_vs_independent']:.2g}" for k in near),
                 "reading": ("tüm kümelerde bağımsız modeli tercih ediyor (BF < 1/3)" if max(bf_near_all) < BF_AGAINST
                             else "izleme yöntemine göre değişiyor; bağımsız modeli açıkça tercih etmiyor"),
                 "direction": "against" if max(bf_near_all) < BF_AGAINST else ("for" if min(bf_near_all) > 3.0 else "neutral")})
    n, k = fate.get("n_near_features"), fate.get("n_lost")
    base = fate.get("baseline_lost_fraction")
    more = fate.get("p_value_more_loss_than_baseline")
    rows.append({"hypothesis": "H1", "evidence": "kopma yerine yakın özelliklerin kaderi",
                 "numbers": f"yakındaki {n} özellikten {k} kayboldu; genel kayıp oranı {base}; daha çok kayıp testi p={more}",
                 "reading": "yakındaki özellikler olağandan çok kaybolmuyor" if (more is None or more >= 0.05) else "yakındaki özellikler olağandan çok kaybolmuş",
                 "direction": "against" if (more is None or more >= 0.05) else "for"})
    rows.append({"hypothesis": "H2", "evidence": "özellikler arası mesafe (Başlık 1)",
                 "numbers": f"karar verilen {h2.get('pairs_decided')} çiftten {h2.get('pairs_constant')} sabit, {h2.get('pairs_changing')} değişiyor; "
                            f"{h2.get('pairs_conflicting_or_undetermined')} çift belirsiz/çelişkili",
                 "reading": ("çiftlerin çoğunda mesafe sabit; kayma yaygın değil" if h2.get("changing_fraction_decided", 0) < 0.2
                             else "çiftlerin önemli bölümünde mesafe değişiyor; kayma olasılığı var"),
                 "direction": ("for" if h2.get("changing_fraction_decided", 0) >= 0.2 else
                               ("neutral" if h2.get("p_more_changes_than_1pct_false_alarm", 1) < 0.05 else "against"))})
    if h2.get("p_more_changes_than_1pct_false_alarm", 1) < 0.05 and h2.get("pairs_changing", 0):
        top = sorted(h2["tracks_in_changing_pairs"].items(), key=lambda kv: -kv[1])[:3]
        rows.append({"hypothesis": "H2", "evidence": "mesafesi değişen çiftler (hareketli aday)",
                     "numbers": f"{h2['pairs_changing']}/{h2['pairs_decided']} çift (%{100 * h2['changing_fraction_decided']:.1f}); 1% yanlış alarm beklentisinden fazla (p={h2['p_more_changes_than_1pct_false_alarm']:.3f}); "
                                f"en sık görünen izler: " + ", ".join(f"iz {t} ({n} çift)" for t, n in top),
                     "reading": "az sayıda çiftte mesafe şansın açıklayacağından fazla değişiyor; tek bir özellik (en sık görünen iz) hareketli olabilir. Çiftler bağımsız olmadığı için p iyimser",
                     "direction": "neutral"})
    rows.append({"hypothesis": "H2", "evidence": "güvenilir uzun izlerin hareket sınıfları",
                 "numbers": f"{h2.get('trusted_long_tracks')} güvenilir uzun izden {h2.get('trusted_long_stationary')} sabit, {h2.get('trusted_long_moved')} yer değiştirmiş",
                 "reading": "çoğu özellik sabit", "direction": "against" if h2.get("trusted_long_moved", 0) < 0.25 * max(h2.get("trusted_long_tracks", 1), 1) else "neutral"})
    rows.append({"hypothesis": "H3", "evidence": "konum (uzak model, R=30 px)",
                 "numbers": "; ".join(f"{SET_TR[k]}: LLR {v['far_llr']:+.1f} [{v['far_lo']:+.1f}, {v['far_hi']:+.1f}]" for k, v in near.items()),
                 "reading": ("tüm özellik kümelerinde bağımsız modelden " + {"worse": "kötü", "better": "iyi", "indistinguishable": "ayırt edilemiyor"}[next(iter(levels_far.values()))]
                             if same_far else "sonuç özellik kümesine göre değişiyor (izleme yöntemine duyarlı)")
                            + "; ancak özelliğin yanındaki soluk DNA eşiklemede kopuk görünebilir, bu yöntem etkisi olabilir",
                 "direction": "against" if strong_against_far else ("for" if (same_far and "better" in levels_far.values() and min(bf_far_all) > 3.0) else "neutral")})
    rows.append({"hypothesis": "H3", "evidence": "konum, R ve karışım ağırlığı üzerinden ortalama (Bayes faktörü uzak/bağımsız)",
                 "numbers": "; ".join(f"{SET_TR[k]}: BF {bf[k]['BF_far_vs_independent']:.2g}" for k in near),
                 "reading": ("tüm kümelerde bağımsız modeli tercih ediyor (BF < 1/3)" if max(bf_far_all) < BF_AGAINST
                             else "izleme yöntemine göre değişiyor (eski güçlü-tespit kümesinde bağımsız modelden ayırt edilemiyor); sonuç izlemeye duyarlı"),
                 "direction": "against" if max(bf_far_all) < BF_AGAINST else ("for" if min(bf_far_all) > 3.0 else "neutral")})
    b = bf.get("strong_and_weak", {})
    if b:
        rows.append({"hypothesis": "all", "evidence": "Bayes faktörleri ve olasılıklar (güçlü + zayıf özellikler, tüm kopma adayları)",
                     "numbers": f"BF yakın/bağımsız {b['BF_near_vs_independent']:.2g}; BF uzak/bağımsız {b['BF_far_vs_independent']:.2g}; "
                                f"olasılıklar (eşit önsel): bağımsız {b['P_independent']:.2f}, yakın {b['P_near']:.2f}, uzak {b['P_far']:.2f}",
                     "reading": "önsel dağılım R ve karışım ağırlığı ızgarası üzerinde düzgün; olasılıklar bu seçime bağlıdır", "direction": "neutral"})
    return pd.DataFrame(rows)


def final_text(ev_tab: pd.DataFrame, res: dict, bf: dict, h2: dict) -> str:
    L = ["## Başlık 6: üç mekanizma modeli (Türkçe özet, tanımlayıcı)", ""]
    for h, name in (("H1", "Enzim dendrimerleri iterek keser"), ("H2", "Dendrimerler DNA üzerinde kendiliğinden kayar"),
                    ("H3", "Enzim yalnızca çıplak bölgeleri keser")):
        sub = ev_tab[ev_tab["hypothesis"] == h]
        dirs = list(sub["direction"])
        if dirs and all(d == "for" for d in dirs):
            v = "veri bu hipotezle uyumlu"
        elif dirs and all(d == "against" for d in dirs):
            v = "veri bu hipotezi desteklemiyor (kesin çürütme değil)"
        elif "for" in dirs and "against" in dirs:
            v = "kanıtlar çelişiyor"
        else:
            v = "veri belirsiz (kanıtların hepsi aynı yönde değil ya da yöntem etkisi var)"
        L.append(f"- **{h} ({name}): {v}.**")
        for r in sub.itertuples():
            L.append(f"  - {r.evidence}: {r.numbers}. Okuma: {r.reading}.")
    b = bf.get("strong_and_weak")
    if b:
        L.append(f"- Olasılık özeti (güçlü + zayıf özellikler, {b['n_events']} kopma adayı, eşit önsel): bağımsız {b['P_independent']:.2f}, yakın {b['P_near']:.2f}, uzak {b['P_far']:.2f}. "
                 "Bu sayılar R ve karışım ağırlığı için seçilen düzgün önsele bağlıdır; mutlak olasılık olarak okunmamalıdır.")
    L.append("- Bu veride hiçbir mekanizma doğrulanmış ya da kesin çürütülmüş değildir: tek molekül, tek deney, kontrol yok, özelliklerin kimliği bilinmiyor, "
             f"kopma adayı sayısı az ({res['strong_and_weak']['n_events']}), kalıcı aday ise yalnızca 4.")
    L.append("- Bu soruyu gerçekten çözecek veri: aynı koşulda enzimsiz ve polimersiz kontroller, enzimi ya da polimeri etiketli deney, ham yükseklik verisi, birden çok molekül.")
    return "\n".join(L) + "\n"


def figure(res: dict, bf: dict, llr_all: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"wspace": 0.35})
    ax = axs[0]
    names = [k for k in FEATURE_SETS if k in res]
    for i, k in enumerate(names):
        v = res[k]
        for j, (kind, c) in enumerate((("near", "#2255c4"), ("far", "#d9534f"))):
            ax.errorbar(i + (j - 0.5) * 0.25, v[f"{kind}_llr"], yerr=[[v[f"{kind}_llr"] - v[f"{kind}_lo"]], [v[f"{kind}_hi"] - v[f"{kind}_llr"]]],
                        fmt="o", color=c, capsize=3, label=("H1-like: near" if kind == "near" else "H3-like: far") if i == 0 else None)
    ax.axhline(0, color="grey", lw=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("log-likelihood ratio vs independent\n(R = 30 px, bootstrap 95%)")
    ax.set_title("Position model by feature set", fontsize=9, loc="left")
    ax.legend(fontsize=7)
    ax = axs[1]
    for k, c in zip(names, ("#888888", "#2255c4", "#2a9d8f")):
        d = llr_all[k]
        ax.plot(d["R_px"], d["LLR_near_vs_independent"], marker="o", ms=3, color=c, ls="-", label=f"{k} near")
        ax.plot(d["R_px"], d["LLR_far_vs_independent"], marker="s", ms=3, color=c, ls="--", label=f"{k} far")
    ax.axhline(0, color="grey", lw=1)
    ax.axhline(2.3, color="green", lw=0.8, ls=":")
    ax.set_xlabel("radius R (px)")
    ax.set_ylabel("LLR (mixing weight 0.05)")
    ax.set_title("Sensitivity to R (dotted: +2.3 = clear advantage)", fontsize=9, loc="left")
    ax.legend(fontsize=6, ncol=2)
    ax = axs[2]
    left = np.zeros(len(names))
    for key, col, lab in (("P_independent", "#bbbbbb", "independent"), ("P_near", "#2255c4", "near (H1-like)"), ("P_far", "#d9534f", "far (H3-like)")):
        vals = np.array([bf[k][key] for k in names])
        ax.barh(range(len(names)), vals, left=left, color=col, label=lab)
        left += vals
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("posterior probability (equal prior, uniform on R and mixing weight)", fontsize=8)
    ax.set_title("Model probabilities (depend on the prior)", fontsize=9, loc="left")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_mechanisms.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def run(outputs: Path, seed: int = 1) -> dict:
    p2, p6, p6b = outputs / "phase2", outputs / "phase6", outputs / "phase6b"
    need = [p2 / "frames.csv", p6 / "feature_detections.csv", p6 / "event_feature_association.csv", p6 / "parameters.json",
            p6b / "feature_detections_v2.csv", p6b / "feature_tracks_v2.csv", p6b / "pair_spacing_summary.csv",
            outputs / "phase3" / "registration.csv"]
    for f in need:
        if not f.is_file():
            raise FileNotFoundError(f"{f} not found. Run the earlier phases first (run_all.sh).")
    frames = pd.read_csv(p2 / "frames.csv").sort_values("rank")
    shifts = cumulative_shifts(pd.read_csv(outputs / "phase3" / "registration.csv"))
    ev = pd.read_csv(p6 / "event_feature_association.csv")
    det_s = pd.read_csv(p6 / "feature_detections.csv")
    det_v2 = pd.read_csv(p6b / "feature_detections_v2.csv")
    trk_v2 = pd.read_csv(p6b / "feature_tracks_v2.csv")
    pairs = pd.read_csv(p6b / "pair_spacing_summary.csv")
    sets = feature_sets(det_s, det_v2, trk_v2)
    s = FeatureSettings()
    last = int(frames["rank"].max())

    res, bf, llr_all, fate_rows = {}, {}, {}, []
    fate_main = {}
    for name, feats in sets.items():
        evs = build_events(ev, feats, p2, frames, shifts)
        subsets = {"all": evs, "persistent_only": [e for e in evs if e["persistence"] == "persistent"],
                   "gap_above_8px": [e for e in evs if e["min_gap_px"] > 8.0]}
        llr = llr_table(evs, subsets, seed=seed)
        llr["feature_set"] = name
        llr_all[name] = llr[(llr["subset"] == "all") & (llr["eps"] == 0.05)].copy()
        row = llr[(llr["subset"] == "all") & (llr["R_px"] == 30) & (llr["eps"] == 0.05)].iloc[0]
        res[name] = {"n_events": int(row["n_events"]), "near_llr": float(row["LLR_near_vs_independent"]),
                     "near_lo": float(row["LLR_near_boot95_lo"]), "near_hi": float(row["LLR_near_boot95_hi"]),
                     "far_llr": float(row["LLR_far_vs_independent"]), "far_lo": float(row["LLR_far_boot95_lo"]),
                     "far_hi": float(row["LLR_far_boot95_hi"]),
                     "best_near_over_grid": float(llr[llr["subset"] == "all"]["LLR_near_vs_independent"].max()),
                     "best_far_over_grid": float(llr[llr["subset"] == "all"]["LLR_far_vs_independent"].max())}
        bf[name] = {**bayes_factors(llr, "all"), "persistent_only": bayes_factors(llr, "persistent_only")}
        fa = fate_frame(ev, feats.assign(), s)
        base = baseline_loss_rate(feats, last, s)
        fate = h1_fate_check(fa, base["lost_fraction"])
        fate_rows.append({"feature_set": name, **fate})
        if name == "strong_and_weak":
            fate_main = fate
        res[name]["fate"] = fate
        fate_rows[-1]["baseline_n"] = base["n"]
        res[name]["_llr"] = llr
    h2 = h2_summary(pairs, trk_v2)
    ev_tab = evidence_table(res, bf, fate_main, h2)
    out = outputs / "mechanisms"
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([res[k]["_llr"] for k in FEATURE_SETS]).to_csv(out / "model_comparison_by_feature_set.csv", index=False)
    pd.DataFrame([{"feature_set": k, **{kk: vv for kk, vv in bf[k].items() if kk != "persistent_only"}} for k in FEATURE_SETS]).to_csv(out / "bayes_factors.csv", index=False)
    pd.DataFrame(fate_rows).to_csv(out / "fate_check_by_feature_set.csv", index=False)
    pd.DataFrame([{"key": k, "value": json.dumps(v) if isinstance(v, dict) else v} for k, v in h2.items()]).to_csv(out / "h2_spacing_summary.csv", index=False)
    ev_tab.to_csv(out / "hypothesis_evidence.csv", index=False)
    clean = {k: {kk: vv for kk, vv in v.items() if kk != "_llr"} for k, v in res.items()}
    (out / "parameters.json").write_text(json.dumps({"R_grid": R_GRID, "eps_grid": EPS_GRID, "results": clean, "bayes": bf, "h2": h2,
                                                     "prior": "equal prior on independent/near/far; uniform on the R x eps grid",
                                                     "notes": "descriptive model comparison; no mechanism confirmed"}, indent=2, default=float), encoding="utf-8")
    (out / "summary_tr.md").write_text(final_text(ev_tab, clean, bf, h2), encoding="utf-8")
    figure(clean, bf, llr_all, out)
    return {"results": clean, "bayes": bf, "h2": h2, "evidence": ev_tab}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Heading 6: three-mechanism comparison.")
    p.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    a = p.parse_args(argv)
    try:
        r = run(a.outputs)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for k, v in r["results"].items():
        print(k, {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "fate"})
    for k, v in r["bayes"].items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk not in ("persistent_only",)})
    print(r["h2"])
    print(r["evidence"][["hypothesis", "evidence", "direction"]].to_string(index=False))
    print(f"Results: {a.outputs / 'mechanisms'}")
    print("Descriptive only: no mechanism is confirmed or rejected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
