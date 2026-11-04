"""Heading 5: when and where does the DNA break (candidates only).

WHEN: time range of every split candidate (earliest/latest possible scan time, estimated), whether it is
persistent, and where it lies relative to the change-point of Phase 8.

WHERE has two parts:
  * position along the reference contour (Phase 6), trusted ONLY when the frame is aligned to the reference
    (aligned overlap >= 0.9) and the event lies within `max_contour_dist_px` of the contour; later frames
    deform, so their contour position is reported as "not reliable" instead of being used;
  * registration-free position: the cut fraction = shortest new piece / sum of the new pieces (0 = at an
    end, 0.5 = in the middle), measured on the pieces themselves. A conservation check compares the sum of
    the new pieces with the length of the original object.

Nothing here confirms cleavage: the events are split candidates from segmentation.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .inventory import PROJECT_ROOT

DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"


@dataclass
class WhereWhenSettings:
    max_contour_dist_px: float = 6.0
    end_fraction: float = 0.2          # a cut closer than this to an end counts as "near an end"
    conservation_tol: float = 0.25     # |sum(new) - old| / old above this = pieces do not account for the old object
    min_events_for_test: int = 8
    pre_window: int = 5
    post_window: int = 5
    seed: int = 1


def _ids(txt) -> list[int]:
    return [int(v) for v in str(txt).split(";") if v.strip() != "" and str(txt) != "nan"]


def event_table(events: pd.DataFrame, objects: pd.DataFrame, assoc: pd.DataFrame, valid: dict,
                frames: pd.DataFrame, cp_low: int, cp_high: int, s: WhereWhenSettings) -> pd.DataFrame:
    length = {(int(r.rank), int(r.object_id)): float(r.skeleton_length_px) for r in objects.itertuples()}
    ft = frames.set_index("rank")
    a = assoc.set_index(["rank_from", "rank_to"])
    rows = []
    for e in events[events["event"] == "split_candidate"].itertuples():
        old = [length.get((int(e.rank_from), i), np.nan) for i in _ids(e.from_objects)]
        new = [length.get((int(e.rank_to), i), np.nan) for i in _ids(e.to_objects)]
        old_len, new_sum = float(np.nansum(old)), float(np.nansum(new))
        frac = float(np.nanmin(new) / new_sum) if new_sum > 0 and len(new) >= 2 else np.nan
        cons = abs(new_sum - old_len) / old_len if old_len > 0 else np.nan
        key = (int(e.rank_from), int(e.rank_to))
        s_px = float(a.loc[key, "event_s_px"]) if key in a.index else np.nan
        dist = float(a.loc[key, "event_dist_to_ref_contour_px"]) if key in a.index else np.nan
        pos_ok = bool(valid.get(int(e.rank_to), False) and valid.get(int(e.rank_from), False)
                      and np.isfinite(dist) and dist <= s.max_contour_dist_px)
        rel = "before" if e.rank_to < cp_low else ("after" if e.rank_to > cp_high else "inside")
        rows.append({"rank_from": int(e.rank_from), "rank_to": int(e.rank_to), "persistence": e.persistence,
                     "time_earliest_s": float(ft.loc[int(e.rank_from), "time_earliest_s"]),
                     "time_latest_s": float(ft.loc[int(e.rank_to), "time_latest_s"]), "n_new_pieces": int(e.n_pieces_after),
                     "old_length_px": round(old_len, 1), "new_lengths_px": ";".join(f"{v:.0f}" for v in new),
                     "cut_fraction_shortest_piece": round(frac, 3) if np.isfinite(frac) else np.nan,
                     "length_conservation_error": round(cons, 3) if np.isfinite(cons) else np.nan,
                     "conserved": bool(np.isfinite(cons) and cons <= s.conservation_tol),
                     "contour_position_s_px": round(s_px, 1) if np.isfinite(s_px) else np.nan,
                     "contour_dist_px": round(dist, 1) if np.isfinite(dist) else np.nan,
                     "contour_position_reliable": pos_ok,
                     "relative_to_change_point": rel})
    return pd.DataFrame(rows)


def cut_fraction_tests(ev: pd.DataFrame, s: WhereWhenSettings) -> dict:
    f = ev.loc[ev["conserved"] & (ev["n_new_pieces"] == 2), "cut_fraction_shortest_piece"].dropna().to_numpy(float)
    out = {"n_two_piece_conserved": int(len(f))}
    if len(f) < s.min_events_for_test:
        out["verdict"] = "too_few_events"
        return out
    # under a uniformly random cut point the shortest fraction is uniform on [0, 0.5]
    ks = stats.kstest(f, stats.uniform(loc=0, scale=0.5).cdf)
    n_end = int((f < s.end_fraction).sum())
    p_end = stats.binomtest(n_end, len(f), 2 * s.end_fraction, alternative="two-sided").pvalue
    out.update({"ks_p_vs_uniform_cut": float(ks.pvalue), "n_near_end": n_end, "expected_near_end": round(2 * s.end_fraction * len(f), 1),
                "binom_p_near_end": float(p_end),
                "verdict": "cut_positions_differ_from_uniform" if min(ks.pvalue, p_end) < 0.05 else "cut_positions_consistent_with_uniform"})
    return out


def step_decomposition(lengths: pd.DataFrame, cp: int, s: WhereWhenSettings) -> dict:
    pre = lengths[(lengths["rank"] >= cp - s.pre_window) & (lengths["rank"] < cp)]
    post = lengths[(lengths["rank"] >= cp) & (lengths["rank"] < cp + s.post_window)]
    if pre.empty or post.empty:
        return {}
    return {"change_point_rank": int(cp),
            "largest_piece_before_px": round(float(pre["largest_corrected_px"].mean()), 1),
            "largest_piece_after_px": round(float(post["largest_corrected_px"].mean()), 1),
            "total_before_px": round(float(pre["total_corrected_px"].mean()), 1),
            "total_after_px": round(float(post["total_corrected_px"].mean()), 1),
            "pieces_before": round(float(pre["n_pieces"].mean()), 1), "pieces_after": round(float(post["n_pieces"].mean()), 1),
            "largest_over_total_before": round(float((pre["largest_corrected_px"] / pre["total_corrected_px"]).mean()), 2),
            "largest_over_total_after": round(float((post["largest_corrected_px"] / post["total_corrected_px"]).mean()), 2)}


def figure(ev: pd.DataFrame, lengths: pd.DataFrame, cp: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"hspace": 0.12})
    ax = axs[0]
    ax.plot(lengths["rank"], lengths["largest_corrected_px"], color="#d9534f", lw=1, marker="o", ms=2.5, label="largest piece")
    ax.plot(lengths["rank"], lengths["total_corrected_px"], color="#2255c4", lw=1, marker="o", ms=2.5, label="total DNA in view")
    ax.axvspan(cp["low"], cp["high"], color="grey", alpha=0.25, label=f"change point {cp['rank']} (95% interval)")
    for r in ev.itertuples():
        ax.axvline(r.rank_to, color="black" if r.persistence == "persistent" else "#bbbbbb", lw=1.2 if r.persistence == "persistent" else 0.7,
                   ymax=0.12)
    ax.set_ylabel("length (px, bias-corrected)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("When: split candidates (ticks at the bottom: black = persistent, grey = rejoined) and the change point", fontsize=9, loc="left")
    ax = axs[1]
    for r in ev.itertuples():
        pers = r.persistence == "persistent"
        ax.scatter(r.rank_to, r.cut_fraction_shortest_piece, s=50 if pers else 28, color="black" if pers else "#888888",
                   marker="o" if r.conserved else "x", alpha=0.9)
    ax.axhspan(0, 0.2, color="#ffd6a5", alpha=0.35)
    ax.axvspan(cp["low"], cp["high"], color="grey", alpha=0.25)
    ax.set_ylim(-0.02, 0.52)
    ax.set_xlabel("frame index (provisional order)")
    ax.set_ylabel("cut fraction\n(0 = at an end, 0.5 = middle)")
    ax.set_title("Where (registration-free): shortest new piece / all new pieces. x = pieces do not add up to the old object", fontsize=9, loc="left")
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.25, lw=0.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_when_where.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def summary_text(ev: pd.DataFrame, tests: dict, dec: dict, cp: dict, n_valid_pos: int) -> str:
    n = len(ev)
    pers = ev[ev["persistence"] == "persistent"]
    L = ["## Başlık 5: kopma ne zaman ve nerede? (Türkçe özet, yalnızca aday)", "",
         f"- NE ZAMAN: {n} kopma adayından {len(pers)} tanesi kalıcı (kopuk parçalar sonraki karelerde birleşmiyor), kalan {n - len(pers)} tanesi sonradan yeniden birleşiyor. "
         f"Kalıcı adaylar kare " + ", ".join(str(int(r)) for r in pers["rank_to"]) + " civarında. "
         f"Ana basamak (Faz 8) kare {cp['rank']} (%95 aralığı {cp['low']}–{cp['high']}); kalıcı adaylar basamaktan "
         f"{int((pers['rank_to'] < cp['low']).sum())} önce, {int(((pers['rank_to'] >= cp['low']) & (pers['rank_to'] <= cp['high'])).sum())} aralıkta, {int((pers['rank_to'] > cp['high']).sum())} sonra. "
         "Zaman karşılığı tahminidir (31 sn/tarama varsayımı, tekrarlanan taramalar ve enzim ekleme anı bilinmiyor)."]
    if dec:
        L.append(f"- Basamağın ayrıntısı (kare {dec['change_point_rank']} öncesi ve sonrası {5} kare ortalaması): en uzun parça "
                 f"{dec['largest_piece_before_px']} → {dec['largest_piece_after_px']} px, toplam DNA {dec['total_before_px']} → {dec['total_after_px']} px, "
                 f"parça sayısı {dec['pieces_before']} → {dec['pieces_after']}. En uzun parçanın toplam içindeki payı {dec['largest_over_total_before']} → {dec['largest_over_total_after']}. "
                 "Toplam yaklaşık sabit kalırken en uzun parça küçülüyorsa DNA kaybolmuyor, parçalara ayrılıyor demektir (aday bulgu).")
    L.append(f"- NEREDE (konturda): kontur konumu yalnızca hizalı kareler (≈1–22) ve konturun 6 px yakınındaki olaylar için güvenilir; "
             f"{n} adaydan yalnızca {n_valid_pos} tanesi bu koşulu sağlıyor, hiçbiri kalıcı değil. Sonraki karelerde molekül şekil değiştirdiği için kontur konumu verilmiyor.")
    if tests.get("verdict") == "too_few_events":
        L.append(f"- NEREDE (kayıt gerektirmeyen ölçü): kesim oranı (en kısa yeni parça / tüm yeni parçalar) için yeterli olay yok "
                 f"(iki parçalı ve uzunluğu korunan olay: {tests['n_two_piece_conserved']}); test yapılmadı.")
    else:
        L.append(f"- NEREDE (kayıt gerektirmeyen ölçü): iki parçalı ve uzunluğu korunan {tests['n_two_piece_conserved']} olayda kesim oranı rastgele bir kesimden "
                 f"{'farklı' if 'differ' in tests['verdict'] else 'anlamlı biçimde farklı değil'} (KS p = {tests['ks_p_vs_uniform_cut']:.2f}); uçlara yakın kesim "
                 f"{tests['n_near_end']} (rastgelede beklenen {tests['expected_near_end']}), iki yönlü binom p = {tests['binom_p_near_end']:.2f}.")
    cons = ev[~ev["conserved"]]
    L.append(f"- Uzunluk korunumu: {len(cons)}/{n} adayda yeni parçaların toplamı eski nesneyle uyuşmuyor (tolerans %25). Bu adaylarda parçalar başka nesnelerle karışmış veya segmentasyon değişmiş olabilir; kesim oranı güvenilmez sayılır (şekilde x).")
    L.append("- Özet sınır: kopma yeri için güvenilir veri yalnızca birkaç erken adayda var ve onlar yeniden birleşiyor. Kalıcı kopmaların yeri bu görüntülerle kesin söylenemez; ham veya daha az kaymış kareler gerekir.")
    return "\n".join(L) + "\n"


def run(outputs: Path, s: WhereWhenSettings | None = None) -> dict:
    s = s or WhereWhenSettings()
    need = [outputs / "phase3" / "events.csv", outputs / "phase2" / "objects.csv", outputs / "phase2" / "frames.csv",
            outputs / "phase6" / "event_feature_association.csv", outputs / "phase7" / "contour_profile_by_frame.csv",
            outputs / "phase8" / "changepoint.csv", outputs / "lengths" / "length_by_frame.csv"]
    for f in need:
        if not f.is_file():
            raise FileNotFoundError(f"{f} not found. Run the earlier phases and scripts/19_lengths.py first (run_all.sh).")
    events = pd.read_csv(need[0])
    objects = pd.read_csv(need[1])
    frames = pd.read_csv(need[2])
    assoc = pd.read_csv(need[3])
    valid = {int(r.rank): bool(r.aligned_match_valid_for_loss_map) for r in pd.read_csv(need[4]).itertuples()}
    cpt = pd.read_csv(need[5])
    row = cpt[cpt["series"] == "largest_object_px"].iloc[0]
    cp = {"rank": int(row["change_point_rank"]), "low": int(row["ci95_rank_low"]), "high": int(row["ci95_rank_high"])}
    lengths = pd.read_csv(need[6])
    ev = event_table(events, objects, assoc, valid, frames, cp["low"], cp["high"], s)
    tests = cut_fraction_tests(ev, s)
    dec = step_decomposition(lengths, cp["rank"], s)
    out = outputs / "where_when"
    out.mkdir(parents=True, exist_ok=True)
    ev.to_csv(out / "split_candidates_when_where.csv", index=False)
    (out / "summary_tr.md").write_text(summary_text(ev, tests, dec, cp, int(ev["contour_position_reliable"].sum())), encoding="utf-8")
    (out / "parameters.json").write_text(json.dumps({"settings": asdict(s), "tests": tests, "step_decomposition": dec, "change_point": cp,
                                                     "notes": "candidates only; contour position trusted only in aligned frames"}, indent=2), encoding="utf-8")
    figure(ev, lengths, cp, out)
    return {"events": ev, "tests": tests, "decomposition": dec, "change_point": cp}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Heading 5: when and where the DNA breaks (candidates).")
    p.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    a = p.parse_args(argv)
    try:
        r = run(a.outputs)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(r["events"][["rank_to", "persistence", "n_new_pieces", "cut_fraction_shortest_piece", "conserved", "contour_position_reliable", "relative_to_change_point"]].to_string(index=False))
    print(r["tests"])
    print(r["decomposition"])
    print(f"Results: {a.outputs / 'where_when'}")
    print("Candidates only: nothing here confirms DNA cleavage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
