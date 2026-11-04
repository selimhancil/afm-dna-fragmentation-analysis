import numpy as np
import pandas as pd

from afm_pipeline import mechanisms as M


def _llr(near, far):
    return pd.DataFrame({"subset": "all", "n_events": 10, "LLR_near_vs_independent": near, "LLR_far_vs_independent": far})


def test_bayes_factors_probabilities_sum_to_one():
    b = M.bayes_factors(_llr([0.0, 0.0], [-2.0, -2.0]))
    assert abs(b["P_independent"] + b["P_near"] + b["P_far"] - 1) < 1e-12
    assert abs(b["BF_near_vs_independent"] - 1.0) < 1e-12 and abs(b["BF_far_vs_independent"] - np.exp(-2)) < 1e-12


def test_bayes_factors_empty_subset():
    assert M.bayes_factors(_llr([0.0], [0.0]), "persistent_only") == {}


def test_feature_sets_split_by_confidence():
    s = pd.DataFrame({"rank": [1], "y_common": [1.0], "x_common": [1.0]})
    v2 = pd.DataFrame({"track_id": [1, 2], "rank": [1, 1], "y_common": [1.0, 5.0], "x_common": [1.0, 5.0]})
    trk = pd.DataFrame({"track_id": [1, 2], "confidence": ["high", "low"]})
    f = M.feature_sets(s, v2, trk)
    assert len(f["strong_only"]) == 1 and len(f["strong_and_weak"]) == 2 and len(f["trusted_tracks"]) == 1


def test_h2_summary_counts_and_false_alarm_test():
    pairs = pd.DataFrame({"track_a": [1, 1, 2, 3], "track_b": [2, 3, 3, 4], "n_frames": [10, 10, 10, 10],
                          "verdict": ["spacing_constant", "spacing_changes", "conflicting_measures", "spacing_constant"]})
    trk = pd.DataFrame({"track_id": [1, 2], "n_detections": [10, 10], "confidence": ["high", "high"], "motion_class": ["stationary", "moved"]})
    h = M.h2_summary(pairs, trk)
    assert h["pairs_decided"] == 3 and h["pairs_changing"] == 1 and h["pairs_conflicting_or_undetermined"] == 1
    assert h["trusted_long_moved"] == 1 and h["tracks_in_changing_pairs"] == {1: 1, 3: 1}


def _res(near, far):
    d = {}
    for k in M.FEATURE_SETS:
        d[k] = {"n_events": 10, "near_llr": near[0], "near_lo": near[1], "near_hi": near[2],
                "far_llr": far[0], "far_lo": far[1], "far_hi": far[2]}
    return d


def _bf(n, f):
    return {k: {"BF_near_vs_independent": n, "BF_far_vs_independent": f, "P_independent": 0.5, "P_near": 0.25, "P_far": 0.25, "n_events": 10}
            for k in M.FEATURE_SETS}


def test_evidence_against_needs_ci_and_bayes_factor_in_every_set():
    h2 = {"pairs_decided": 10, "pairs_constant": 10, "pairs_changing": 0, "pairs_conflicting_or_undetermined": 0,
          "changing_fraction_decided": 0.0, "p_more_changes_than_1pct_false_alarm": 1.0, "trusted_long_tracks": 10,
          "trusted_long_moved": 0, "trusted_long_stationary": 10, "tracks_in_changing_pairs": {}}
    fate = {"n_near_features": 5, "n_lost": 0, "baseline_lost_fraction": 0.1, "p_value_more_loss_than_baseline": 1.0}
    t = M.evidence_table(_res((-5, -9, -1), (-5, -9, -1)), _bf(0.01, 0.01), fate, h2)
    assert t[(t.hypothesis == "H3") & t.evidence.str.contains("R=30")].iloc[0].direction == "against"
    bf = _bf(0.01, 0.01)
    bf["strong_only"]["BF_far_vs_independent"] = 0.88
    t2 = M.evidence_table(_res((-5, -9, -1), (-5, -9, -1)), bf, fate, h2)
    assert t2[(t2.hypothesis == "H3") & t2.evidence.str.contains("R=30")].iloc[0].direction == "neutral"


def test_final_text_requires_unanimity():
    h2 = {"pairs_decided": 10, "pairs_constant": 10, "pairs_changing": 0, "pairs_conflicting_or_undetermined": 0,
          "changing_fraction_decided": 0.0, "p_more_changes_than_1pct_false_alarm": 1.0, "trusted_long_tracks": 10,
          "trusted_long_moved": 0, "trusted_long_stationary": 10, "tracks_in_changing_pairs": {}}
    fate = {"n_near_features": 5, "n_lost": 0, "baseline_lost_fraction": 0.1, "p_value_more_loss_than_baseline": 1.0}
    res, bf = _res((-5, -9, -1), (-5, -9, -1)), _bf(0.01, 0.01)
    t = M.evidence_table(res, bf, fate, h2)
    txt = M.final_text(t, res, bf, h2)
    assert "H3" in txt and "desteklemiyor" in txt
