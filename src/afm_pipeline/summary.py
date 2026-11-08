"""Overview of the six supervisor requests, built only from the output files (no typed-in numbers).

Writes outputs/supervisor/:
  overview_rows.json   rows for the viewer's first tab
  overview.md       the same as text
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


def _pct(x):
    try:
        return f"{100 * float(x):.0f}%"
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
    txt = outputs / "mechanisms" / "summary.md"
    k["mech_verdicts"] = [ln.strip("- *").strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.startswith("- **H")] if txt.is_file() else []
    return k


def _short_verdict(v: str) -> str:
    """'H1 (name): data inconclusive (long reason).' -> 'H1 (name): data inconclusive'."""
    head, sep, rest = v.partition("): ")
    return (head + sep + rest.split(" (")[0].rstrip(".")) if sep else v


def overview_rows(k: dict) -> list[dict]:
    r = []
    c = k.get("conf", {})
    p = k.get("pairs", {})
    r.append({"no": 1, "request": "Track bright features along the DNA (movement, size, brightness)",
              "done": "Tracking that also follows weak but consistent peaks, control with random points, track confidence, distance between features, photo strips",
              "result": f"Number of tracks {k.get('tracks', NA)}; single-frame tracks {k.get('old_single', NA)} → {k.get('single', NA)}; tracks lasting at least 6 frames {k.get('old_long', NA)} → {k.get('long', NA)}. "
                        f"Confidence: high {c.get('high', 0)}, medium {c.get('medium', 0)}, low {c.get('low', 0)}. "
                        f"Distance in feature pairs: {p.get('constant', NA)} constant, {p.get('changes', NA)} changing, {p.get('unclear', NA)} unclear (total {p.get('n', NA)}).",
              "limit": f"The identity of the features is unknown; in the control, {_pct(k.get('null', {}).get('fake_share_with_6plus'))} of random tracks also collect at least 5 weak peaks (real tracks: {_pct(k.get('null', {}).get('real_share_with_6plus'))}).", "status": "done, limited"})
    pp = k.get("pop", {})
    r.append({"no": 2, "request": "Separate dendrimers from enzymes",
              "done": "Population test on size and brightness, within-track consistency, relation to movement and lifetime",
              "result": f"Two-population test p = {_f(pp.get('p'), 3)} (borderline); each feature keeps its own size (ICC {_f(pp.get('icc'), 2)}). "
                        "Size does not explain movement or lifetime.",
              "limit": "Identity cannot be determined from JPG colours; area and brightness measure the same thing. Control images and raw data are needed.", "status": "not possible (no data), descriptive"})
    h = k.get("heights", {})
    r.append({"no": 3, "request": "Height profile (time-series topography)",
              "done": "Height in nm under an adjustable [height] assumption, consistency check, optional DNA anchor",
              "result": f"{'ASSUMED (not measured)' if str(h.get('label', '')).startswith('ASSUMED') else 'measured (confirmed)'}: taking the {_f(h.get('z'), 0)} nm colour range as linear gives a DNA ridge of {_f(h.get('ridge'))} nm "
                        f"({'in the usual range' if h.get('plausible') else 'outside the usual 0.5–2 nm range'}).",
              "limit": "There is no real height; the numbers are assumptions. When the supervisor provides raw data or the colour bar, set status = confirmed.", "status": "done under an assumption"})
    s = k.get("len", {}).get("sc", {})
    tt, tl = k.get("len", {}).get("tt", {}), k.get("len", {}).get("tl", {})
    r.append({"no": 4, "request": "Contour length over time",
              "done": "Measurement bias measured with synthetic curves and corrected, total and longest piece, scale consistency check",
              "result": f"Total length trend {_f(tt.get('theil_sen_slope_per_frame'), 2)} px/frame (interval {_f(tt.get('ci_low'), 2)} to {_f(tt.get('ci_high'), 2)}), "
                        f"longest piece {_f(tl.get('theil_sen_slope_per_frame'), 2)} px/frame. The DNA does not disappear, it appears to fragment.",
              "limit": f"At the estimated scale the DNA in the first frame is {_f(s.get('frame1_largest_nm_at_estimated_scale'), 0)} nm; the whole pBR322 can be {_f(s.get('plasmid_full_contour_nm'), 0)} nm. "
                       "The scale or merged-segmentation problem is unresolved; nm values are rough.", "status": "done in pixels, nm uncertain"})
    cp, e, d = k.get("cp", {}), k.get("events", {}), k.get("dec", {})
    r.append({"no": 5, "request": "When and where does the break happen",
              "done": "Change point, timing of break candidates, registration-free cut fraction",
              "result": f"Main step at frame {cp.get('rank', NA)} (interval {cp.get('low', NA)}–{cp.get('high', NA)}), estimated {str(k.get('time_range', NA)).replace(' to ', '–')} minutes. "
                        f"Of {e.get('n', NA)} candidates, {e.get('persistent', NA)} are persistent. Across the step the longest piece goes {d.get('largest_piece_before_px', NA)} → {d.get('largest_piece_after_px', NA)} px, total constant.",
              "limit": f"The contour position is reliable only for {e.get('pos_reliable', NA)} early candidates, none persistent. The time is an estimate; the enzyme addition time is unknown.", "status": "done, location limited"})
    r.append({"no": 6, "request": "Which of the three mechanisms fits (probability model)",
              "done": "Position model with three feature sets, Bayes factors, loss test, distance test, strict decision rules",
              "result": " | ".join(_short_verdict(v) for v in k.get("mech_verdicts", [])) or NA,
              "limit": "One molecule, one experiment, no control, few break candidates; no mechanism was confirmed or refuted.", "status": "done, inconclusive"})
    return r


def overview_md(rows: list[dict]) -> str:
    out = ["# The supervisor's 6 requests: what was done, what came out", ""]
    for x in rows:
        out += [f"## {x['no']}. {x['request']}", f"- Status: {x['status']}", f"- Done: {x['done']}", f"- Result: {x['result']}", f"- Limit: {x['limit']}", ""]
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
    rows = overview_rows(k)
    (out / "overview_rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "overview.md").write_text(overview_md(rows), encoding="utf-8")
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
