import numpy as np
import pandas as pd

from afm_pipeline.phase8 import (analyse_series, block_bootstrap_step, compare_models, frame_to_time,
                                 sensitivity_change_points, step_fit)


def noisy_step(n=60, k=35, lo=100.0, hi=500.0, noise=20.0, seed=3):
    rng = np.random.default_rng(seed)
    y = np.where(np.arange(n) < k, hi, lo) + rng.normal(0, noise, n)
    return y


def test_step_fit_finds_the_step():
    y = noisy_step()
    fit = step_fit(y)
    assert abs(fit["k"] - 35) <= 1
    assert fit["before"] > fit["after"]


def test_bootstrap_interval_contains_truth_and_is_narrow_for_clean_step():
    y = noisy_step(noise=10.0)
    b = block_bootstrap_step(y, n_boot=200)
    lo, hi = np.percentile(b, [2.5, 97.5])
    assert lo <= 35 <= hi and hi - lo <= 6


def test_step_beats_constant_by_bic():
    y = noisy_step()
    m = compare_models(np.arange(len(y), dtype=float), y).set_index("model")
    assert m.loc["one_step", "bic"] < m.loc["constant", "bic"] - 20


def test_analyse_series_time_range_is_ordered():
    y = noisy_step()
    ranks = np.arange(1, 61)
    frames = pd.DataFrame({"rank": ranks, "time_earliest_s": (ranks - 1) * 31.0,
                           "time_latest_s": (ranks - 1) * 31.0 + 1300})
    r = analyse_series("x", ranks, y, frames, n_boot=100)
    lo, hi = (float(v) for v in r["time_range_min_estimated"].split(" to "))
    assert lo < hi and abs(r["change_point_rank"] - 36) <= 2
    assert frame_to_time(frames, 1)[0] == 0.0


def test_sensitivity_change_points():
    rows = []
    for st in ("a", "b"):
        for r in range(1, 21):
            rows.append({"setting": st, "rank": r, "largest_px": 500.0 if r <= 12 else 100.0,
                         "continuity_fraction": 1.0 if r <= 12 else 0.3})
    out = sensitivity_change_points(pd.DataFrame(rows))
    assert set(out["change_point_rank"]) == {13}
