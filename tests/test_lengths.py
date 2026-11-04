import numpy as np
import pandas as pd

from afm_pipeline.lengths import (LengthSettings, bias_summary, pieces, scale_check, synthetic_bias, trend)


def test_pieces_counts_and_small_bookkeeping():
    sk = np.zeros((20, 60), bool)
    sk[5, 2:32] = True            # 30 px
    sk[12, 40:45] = True          # 5 px (below min)
    small: list = []
    p = pieces(sk, 20, small)
    assert len(p) == 1 and abs(p[0] - 29) < 1e-6 and len(small) == 1


def test_synthetic_bias_is_small_and_positive():
    b = synthetic_bias(LengthSettings(n_synthetic=8, seed=3))
    s = bias_summary(b)
    assert 1.0 < s["mean_ratio"] < 1.15 and s["sd_ratio"] < 0.05


def test_scale_check_ratio():
    sc = scale_check(500.0, 500.0, 3.0)
    assert abs(sc["ratio_largest_to_plasmid"] - 1500 / 1483) < 0.01
    assert abs(sc["implied_nm_per_px_if_largest_is_whole_plasmid"] - 1483 / 500) < 0.01


def test_trend_detects_decline():
    df = pd.DataFrame({"rank": np.arange(30), "v": 100 - 2 * np.arange(30) + np.random.default_rng(0).normal(0, 1, 30)})
    t = trend(df, "v", n_boot=200)
    assert t["ci_high"] < -1.5 and t["spearman_rho"] < -0.9
