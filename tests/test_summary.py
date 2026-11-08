import json

import pandas as pd

from afm_pipeline import summary as S


def test_short_verdict_strips_reason():
    v = "H1 (The enzyme pushes dendrimers away): data inconclusive (the evidence does not all point the same way)."
    assert S._short_verdict(v) == "H1 (The enzyme pushes dendrimers away): data inconclusive"


def test_collect_and_texts_work_on_empty_outputs(tmp_path):
    k = S.collect(tmp_path)
    rows = S.overview_rows(k)
    assert len(rows) == 6 and all(r["request"] and r["status"] for r in rows)
    assert "n/a" in S.summary_en(k) and "Subject:" in S.email_en(k)


def test_run_writes_files_and_uses_numbers(tmp_path):
    (tmp_path / "phase6b").mkdir()
    pd.DataFrame({"version": ["a", "b"], "n_tracks": [66, 63], "tracks_with_1_detection": [32, 6],
                  "tracks_with_6plus_detections": [16, 36], "median_detections_per_track": [2, 6], "total_detections": [247, 498]}).to_csv(
        tmp_path / "phase6b" / "track_comparison.csv", index=False)
    r = S.run(tmp_path)
    assert (tmp_path / "supervisor" / "summary_en.md").is_file()
    assert "36" in r["rows"][0]["result"] and "63" in r["rows"][0]["result"]
    assert json.loads((tmp_path / "supervisor" / "overview_rows.json").read_text(encoding="utf-8"))[0]["no"] == 1


def test_no_forbidden_typography_in_email(tmp_path):
    t = S.email_en(S.collect(tmp_path))
    assert "—" not in t and "“" not in t and "’" not in t
