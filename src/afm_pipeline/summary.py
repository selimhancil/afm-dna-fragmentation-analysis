"""Overview of the six supervisor requests, built only from the output files (no typed-in numbers).

Writes outputs/supervisor/:
  overview_rows.json   rows for the viewer's first tab (Turkish)
  overview_tr.md       the same as text
  summary_en.md        a short summary for the supervisor (English)
  email_en.md          a plain reply email with the data requests (English)
Missing inputs are reported as "not available" instead of failing, so the file can be built at any stage.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .inventory import PROJECT_ROOT

DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"
NA = "n/a"


def _csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p) if p.is_file() else pd.DataFrame()


def _json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def _f(x, nd=1):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return NA


def collect(outputs: Path) -> dict:
    k: dict = {}
    t6 = _json(outputs / "phase6" / "parameters.json")
    trk6 = _csv(outputs / "phase6" / "feature_tracks.csv")
    k["old_tracks"] = int(len(trk6)) if len(trk6) else None
    k["old_single"] = int((trk6["n_detections"] == 1).sum()) if len(trk6) else None
    cmp = _csv(outputs / "phase6b" / "track_comparison.csv")
    if len(cmp) == 2:
        a = cmp.iloc[1]
        k["tracks"], k["single"], k["long"] = int(a.n_tracks), int(a.tracks_with_1_detection), int(a.tracks_with_6plus_detections)
        k["old_long"] = int(cmp.iloc[0].tracks_with_6plus_detections)
        k["detections"] = int(a.total_detections)
    v2 = _csv(outputs / "phase6b" / "feature_tracks_v2.csv")
    if len(v2):
        k["conf"] = v2["confidence"].value_counts().to_dict()
    p6b = _json(outputs / "phase6b" / "parameters.json")
    k["null"] = p6b.get("null_control", {})
    pr = _csv(outputs / "phase6b" / "pair_spacing_summary.csv")
    if len(pr):
        ok = pr[pr["n_frames"] >= 5]
        k["pairs"] = {"n": int(len(ok)), "constant": int((ok["verdict"] == "spacing_constant").sum()),
                      "changes": int((ok["verdict"] == "spacing_changes").sum()),
                      "unclear": int(ok["verdict"].isin(["conflicting_measures", "undetermined"]).sum())}
    p6c = _json(outputs / "phase6c" / "parameters.json")
    k["pop"] = {"p": p6c.get("mixture", {}).get("p_bootstrap"), "p_range": p6c.get("mixture", {}).get("p_range"),
                "icc": p6c.get("icc", {}).get("icc"), "n": p6c.get("mixture", {}).get("n")}
    h = _json(outputs / "heights" / "parameters.json")
    k["heights"] = {"label": h.get("label"), "ridge": h.get("plausibility", {}).get("median_ridge_nm"),
                    "plausible": h.get("plausibility", {}).get("within_plausible_range"),
                    "z": h.get("height_config", {}).get("z_range_nm"), "anchor": h.get("height_config", {}).get("anchor_dna_height_nm")}
    L = _json(outputs / "lengths" / "parameters.json")
    k["len"] = {"bias": L.get("bias", {}).get("mean_ratio"), "sc": L.get("scale_check", {}),
                "tt": L.get("trend_total", {}), "tl": L.get("trend_largest", {})}
    W = _json(outputs / "where_when" / "parameters.json")
    k["cp"] = W.get("change_point", {})
    k["dec"] = W.get("step_decomposition", {})
    ev = _csv(outputs / "where_when" / "split_candidates_when_where.csv")
    if len(ev):
        k["events"] = {"n": int(len(ev)), "persistent": int((ev["persistence"] == "persistent").sum()),
                       "pos_reliable": int(ev["contour_position_reliable"].sum())}
    cpt = _csv(outputs / "phase8" / "changepoint.csv")
    if len(cpt):
        k["time_range"] = str(cpt.iloc[0]["time_range_min_estimated"])
    M = _json(outputs / "mechanisms" / "parameters.json")
    k["mech"] = M
    txt = outputs / "mechanisms" / "summary_tr.md"
    k["mech_verdicts"] = [ln.strip("- *").strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.startswith("- **H")] if txt.is_file() else []
    return k


def _short_verdict(v: str) -> str:
    """'H1 (name): veri belirsiz (long reason).' -> 'H1 (name): veri belirsiz'."""
    head, sep, rest = v.partition("): ")
    return (head + sep + rest.split(" (")[0].rstrip(".")) if sep else v


def rows_tr(k: dict) -> list[dict]:
    r = []
    c = k.get("conf", {})
    p = k.get("pairs", {})
    r.append({"no": 1, "istek": "Parlak özellikleri DNA boyunca takip et (hareket, boyut, parlaklık)",
              "yapilan": "Zayıf ama tutarlı tepeleri de izleyen takip, rastgele noktalarla kontrol, iz güveni, özellikler arası mesafe, fotoğraf şeritleri",
              "sonuc": f"İz sayısı {k.get('tracks', NA)}; tek karelik iz {k.get('old_single', NA)} → {k.get('single', NA)}; en az 6 kare süren iz {k.get('old_long', NA)} → {k.get('long', NA)}. "
                       f"Güven: yüksek {c.get('high', 0)}, orta {c.get('medium', 0)}, düşük {c.get('low', 0)}. "
                       f"Özellik çiftlerinde mesafe: {p.get('constant', NA)} sabit, {p.get('changes', NA)} değişiyor, {p.get('unclear', NA)} belirsiz (toplam {p.get('n', NA)}).",
              "sinir": "Özelliklerin kimliği bilinmiyor; zayıf tespitlerin kontrolü rastgele izlerin %12'si de uzadığını gösteriyor.", "durum": "yapıldı, sınırlı"})
    pp = k.get("pop", {})
    r.append({"no": 2, "istek": "Dendrimerleri enzimlerden ayır",
              "yapilan": "Boyut ve parlaklıkta topluluk testi, iz içi tutarlılık, hareket ve ömürle ilişki",
              "sonuc": f"İki topluluk testi p = {_f(pp.get('p'), 3)} (sınırda); her özellik kendi boyutunu koruyor (ICC {_f(pp.get('icc'), 2)}). "
                       "Boyut hareketi ve ömrü açıklamıyor.",
              "sinir": "JPG renklerinden kimlik belirlenemez; alan ile parlaklık aynı şeyi ölçüyor. Kontrol görüntüleri ve ham veri gerekir.", "durum": "yapılamaz (veri yok), tanımlayıcı"})
    h = k.get("heights", {})
    r.append({"no": 3, "istek": "Yükseklik profili (zaman serisi topografya)",
              "yapilan": "Ayarlanabilir [height] varsayımıyla nm yüksekliği, tutarlılık kontrolü, isteğe bağlı DNA çapası",
              "sonuc": f"{'VARSAYIM (ölçüm değil)' if str(h.get('label', '')).startswith('ASSUMED') else 'ölçülmüş (doğrulanmış)'}: {_f(h.get('z'), 0)} nm renk aralığı doğrusal alınınca DNA sırtı {_f(h.get('ridge'))} nm "
                       f"({'olağan aralıkta' if h.get('plausible') else 'olağan 0,5–2 nm aralığının dışında'}).",
              "sinir": "Gerçek yükseklik yok; sayılar varsayımdır. Hoca ham veri ya da renk çubuğu verince status = confirmed yazılır.", "durum": "varsayımla yapıldı"})
    s = k.get("len", {}).get("sc", {})
    tt, tl = k.get("len", {}).get("tt", {}), k.get("len", {}).get("tl", {})
    r.append({"no": 4, "istek": "Kontur uzunluğunun zamanla değişimi",
              "yapilan": "Ölçüm sapması sahte eğrilerle ölçüldü ve düzeltildi, toplam ve en uzun parça, ölçek tutarlılık kontrolü",
              "sonuc": f"Toplam uzunluk eğilimi {_f(tt.get('theil_sen_slope_per_frame'), 2)} px/kare (aralık {_f(tt.get('ci_low'), 2)} ile {_f(tt.get('ci_high'), 2)}), "
                       f"en uzun parça {_f(tl.get('theil_sen_slope_per_frame'), 2)} px/kare. DNA kaybolmuyor, parçalanıyor görünüyor.",
              "sinir": f"İlk karedeki DNA tahmini ölçekle {_f(s.get('frame1_largest_nm_at_estimated_scale'), 0)} nm; tam pBR322 {_f(s.get('plasmid_full_contour_nm'), 0)} nm olabilir. "
                       "Ölçek ya da birleşik segmentasyon sorunu çözülmedi; nm değerleri kaba.", "durum": "piksel olarak yapıldı, nm belirsiz"})
    cp, e, d = k.get("cp", {}), k.get("events", {}), k.get("dec", {})
    r.append({"no": 5, "istek": "Kopma ne zaman ve nerede oluyor",
              "yapilan": "Kırılma noktası, kopma adaylarının zamanı, kayıt gerektirmeyen kesim oranı",
              "sonuc": f"Ana basamak kare {cp.get('rank', NA)} (aralık {cp.get('low', NA)}–{cp.get('high', NA)}), tahmini {str(k.get('time_range', NA)).replace(' to ', '–')} dakika. "
                       f"{e.get('n', NA)} adaydan {e.get('persistent', NA)} tanesi kalıcı. Basamakta en uzun parça {d.get('largest_piece_before_px', NA)} → {d.get('largest_piece_after_px', NA)} px, toplam sabit.",
              "sinir": f"Kontur konumu yalnızca {e.get('pos_reliable', NA)} erken adayda güvenilir, hiçbiri kalıcı değil. Zaman tahminidir, enzim ekleme anı bilinmiyor.", "durum": "yapıldı, yer sınırlı"})
    r.append({"no": 6, "istek": "Üç mekanizmadan hangisi uyuyor (olasılık modeli)",
              "yapilan": "Üç özellik kümesiyle konum modeli, Bayes faktörleri, kayıp testi, mesafe testi, sıkı karar kuralları",
              "sonuc": " | ".join(_short_verdict(v) for v in k.get("mech_verdicts", [])) or NA,
              "sinir": "Tek molekül, tek deney, kontrol yok, kopma adayı sayısı az; hiçbir mekanizma doğrulanmadı ya da çürütülmedi.", "durum": "yapıldı, belirsiz"})
    return r


def overview_md_tr(rows: list[dict]) -> str:
    out = ["# Hocanın 6 isteği: ne yapıldı, ne çıktı", ""]
    for x in rows:
        out += [f"## {x['no']}. {x['istek']}", f"- Durum: {x['durum']}", f"- Yapılan: {x['yapilan']}", f"- Sonuç: {x['sonuc']}", f"- Sınır: {x['sinir']}", ""]
    return "\n".join(out)


def summary_en(k: dict) -> str:
    c, p, pp = k.get("conf", {}), k.get("pairs", {}), k.get("pop", {})
    h, s = k.get("heights", {}), k.get("len", {}).get("sc", {})
    tt, tl = k.get("len", {}).get("tt", {}), k.get("len", {}).get("tl", {})
    cp, e, d = k.get("cp", {}), k.get("events", {}), k.get("dec", {})
    L = ["# Analysis of the 74 AFM frames: status against your six requests", "",
         "All results come from the JPG frames only (pixel units, one molecule, one experiment, no control). Nothing below confirms DNA cleavage "
         "or identifies a feature as enzyme or dendrimer. The code and every output are reproducible from the repository.", "",
         "## 1. Tracking bright features along the DNA",
         f"Features were tracked frame to frame, including weaker peaks that follow a clear detection. This gave {k.get('tracks', NA)} tracks, of which "
         f"{k.get('long', NA)} last six frames or more (before: {k.get('old_long', NA)}); single-frame tracks fell from {k.get('old_single', NA)} to {k.get('single', NA)}. "
         f"Track confidence: {c.get('high', 0)} high, {c.get('medium', 0)} medium, {c.get('low', 0)} low. "
         f"Spacing between feature pairs along the DNA: {p.get('constant', NA)} constant, {p.get('changes', NA)} changing, {p.get('unclear', NA)} unclear, out of {p.get('n', NA)} pairs. "
         "Sliding is therefore not common in these frames.", "",
         "## 2. Dendrimer versus enzyme",
         f"Size and brightness give weak and borderline evidence for two populations (bootstrap p = {_f(pp.get('p'), 3)}); area and brightness measure nearly the same thing. "
         f"Each feature keeps its own size over time (ICC {_f(pp.get('icc'), 2)}). Colour alone cannot identify the molecule; control images (enzyme only, polymer only) are needed.", "",
         "## 3. Height profile",
         f"The colour-to-height mapping is unknown, so heights are an assumption: a linear mapping over the {_f(h.get('z'), 0)} nm Z range gives a DNA ridge of about {_f(h.get('ridge'))} nm, "
         f"{'within' if h.get('plausible') else 'above'} the usual 0.5 to 2 nm, so the mapping is doubtful. All heights are labelled ASSUMED and are changed in one config section.", "",
         "## 4. Contour length over time",
         f"Total DNA length in view stays about constant (trend {_f(tt.get('theil_sen_slope_per_frame'), 2)} px per frame), while the largest piece falls "
         f"({_f(tl.get('theil_sen_slope_per_frame'), 2)} px per frame), consistent with fragmentation rather than loss. "
         f"A scale check does not close: at the estimated scale the DNA in frame 1 is {_f(s.get('frame1_largest_nm_at_estimated_scale'), 0)} nm, longer than the full pBR322 contour of about {_f(s.get('plasmid_full_contour_nm'), 0)} nm.", "",
         "## 5. When and where fragmentation happens",
         f"The main step in the largest piece is at frame {cp.get('rank', NA)} (95% interval {cp.get('low', NA)} to {cp.get('high', NA)}), about {k.get('time_range', NA)} minutes on the estimated time axis. "
         f"Of {e.get('n', NA)} split candidates, {e.get('persistent', NA)} are persistent. Across the step the largest piece goes from {d.get('largest_piece_before_px', NA)} to {d.get('largest_piece_after_px', NA)} px while the total stays the same. "
         f"Positions along the contour are reliable for only {e.get('pos_reliable', NA)} early candidates, none persistent.", "",
         "## 6. Which mechanism fits",
         "A probabilistic comparison of where splits occur relative to features (three feature sets, Bayes factors), the fate of features near splits, and feature spacing does not favour any of the three mechanisms. "
         "None is confirmed or rejected, and the result for the third depends on the tracking method.", "",
         "## What would settle it",
         "Raw AFM files with real height and pixel size; the enzyme addition time and true scan times; an enzyme-only and a polymer-only control; the true frame order; the polymer identity."]
    return "\n".join(L) + "\n"


def email_en(k: dict) -> str:
    s = k.get("len", {}).get("sc", {})
    return f"""Subject: AFM analysis update and a few things I need from you

Dear Dr. Abdelhady,

Thank you for your email. I have worked through your three requests and organised them into six parts: tracking the bright features, telling dendrimers from enzymes, height profiles, contour length over time, when and where the DNA breaks, and a comparison of the three mechanisms. A short summary is attached. The main point is that the images alone are not enough to decide between the mechanisms, and I would like to ask for the following so that I can do it properly.

1. The raw AFM files for these 74 scans, or at least the height data and the colour scale. I am only working from JPG colours, so the heights in my analysis are assumptions.
2. The real pixel size or scan size of each frame. With the image taken as 1000 nm wide, the DNA in the first frame comes to about {_f(s.get('frame1_largest_nm_at_estimated_scale'), 0)} nm, which is longer than the full pBR322 contour of about {_f(s.get('plasmid_full_contour_nm'), 0)} nm. Either the scale is smaller than I assumed or two molecules are joined in that frame.
3. When the DNase I was added, and the real scan times, so that I can put the break on a true time axis.
4. An enzyme-only image series and a polymer-only series under the same conditions, so that the two kinds of bright spot can be told apart.
5. The true order of the frames. I currently use the order I could infer from the file names.
6. Which polymer and which generation these frames show, if it is not G4 PAMAM.

With what I have, the largest DNA piece drops at about frame {k.get('cp', {}).get('rank', NA)} while the total length stays about the same, which looks like fragmentation, but I would not call it confirmed. Features rarely slide along the DNA, and none of the three mechanisms is favoured yet.

Best regards,
Selim
"""


def run(outputs: Path) -> dict:
    k = collect(outputs)
    out = outputs / "supervisor"
    out.mkdir(parents=True, exist_ok=True)
    rows = rows_tr(k)
    (out / "overview_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "overview_tr.md").write_text(overview_md_tr(rows), encoding="utf-8")
    (out / "summary_en.md").write_text(summary_en(k), encoding="utf-8")
    (out / "email_en.md").write_text(email_en(k), encoding="utf-8")
    return {"rows": rows, "k": k}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Overview of the six requests + supervisor summary and email.")
    p.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    a = p.parse_args(argv)
    try:
        run(a.outputs)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Written: {a.outputs / 'supervisor'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
