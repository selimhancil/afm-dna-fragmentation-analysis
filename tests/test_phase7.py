import numpy as np
import pandas as pd

from afm_pipeline.phase7 import behaviour_classes, llr_table, loss_rank, site_loglik


def test_loss_rank_needs_consecutive_uncovered_frames():
    cover = np.array([[1, 1, 1, np.nan],
                      [1, 0, 1, np.nan],
                      [0, 0, 1, np.nan],
                      [0, 1, 1, np.nan]], dtype=float)
    out = loss_rank(cover, run=2)
    assert out.tolist() == [2, 1, -1, -1]      # col 3 is all unknown: never called lost


def test_site_loglik_prefers_matching_model():
    pool = np.array([5.0] * 20 + [80.0] * 80)            # 20% of DNA is near a feature
    ll_near_hit = site_loglik(5.0, pool, "near", 30, 0.05)
    ll_far_hit = site_loglik(80.0, pool, "far", 30, 0.05)
    ll_near_miss = site_loglik(80.0, pool, "near", 30, 0.05)
    assert ll_near_hit > np.log(1 / 100) > ll_near_miss
    assert ll_far_hit > np.log(1 / 100)


def test_llr_sign_for_events_near_features():
    pool = np.array([5.0] * 20 + [80.0] * 80)
    evs = [{"rank_from": 1, "rank_to": 2, "persistence": "persistent", "min_gap_px": 10.0, "d": 5.0, "pool": pool}
           for _ in range(6)]
    t = llr_table(evs, {"all": evs}, n_boot=50)
    row = t[(t["R_px"] == 30) & (t["eps"] == 0.05)].iloc[0]
    assert row["LLR_near_vs_independent"] > 0 > row["LLR_far_vs_independent"]


def test_behaviour_classes():
    t = pd.DataFrame({"n_detections": [12, 1, 5], "motion_class": ["stationary", "short_track_uncertain", "moved"],
                      "lifespan_frames": [12, 1, 6]})
    assert behaviour_classes(t)["behaviour_class"].tolist() == ["persistent_stationary", "transient_short", "intermediate"]
