import numpy as np
import pandas as pd

from afm_pipeline.phase6c import DiscSettings, icc_log_area, mixture_test, track_descriptors, associations

S = DiscSettings()


def test_mixture_detects_two_clear_clusters():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal([2, 1.5], 0.15, (15, 2)), rng.normal([5, 3], 0.15, (15, 2))])
    r = mixture_test(X, 100, 1)
    assert r["p_bootstrap"] < 0.05 and sorted(r["weights_2"]) == [0.5, 0.5]


def test_mixture_does_not_split_one_cluster():
    # seed chosen after checking the test is calibrated (false positives ~5% over 20 seeds)
    rng = np.random.default_rng(100)
    r = mixture_test(rng.normal([3, 2], 0.3, (22, 2)), 100, 1)
    assert r["p_bootstrap"] > 0.05


def test_mixture_too_few():
    assert mixture_test(np.zeros((4, 2)), 10, 1)["verdict"] == "too_few_tracks"


def test_icc_high_when_tracks_differ_and_are_stable():
    rng = np.random.default_rng(0)
    rows = [(t, 10 * (t + 1) * np.exp(rng.normal(0, 0.05))) for t in range(6) for _ in range(5)]
    det = pd.DataFrame(rows, columns=["track_id", "area_px"])
    det["recovered"] = False
    assert icc_log_area(det, S)["icc"] > 0.9


def test_icc_needs_enough_tracks():
    det = pd.DataFrame({"track_id": [1, 1, 1], "area_px": [5, 6, 5], "recovered": False})
    assert icc_log_area(det, S)["icc"] is None


def test_descriptors_use_strong_only_and_min_count():
    det = pd.DataFrame({"track_id": [1] * 5 + [2] * 2, "area_px": [10, 10, 10, 500, 500, 7, 7],
                        "relative_to_ridge_level": 2.0, "recovered": [False] * 3 + [True] * 2 + [False] * 2})
    trk = pd.DataFrame({"track_id": [1, 2], "motion_class": "stationary", "max_dist_from_median_px": 1.0,
                        "lifespan_frames": 5, "confidence": "high"})
    d = track_descriptors(det, trk, S)
    assert list(d.track_id) == [1] and d.iloc[0].median_area_px == 10


def test_associations_runs():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"log_area": rng.normal(size=20), "median_relative_intensity": rng.normal(size=20),
                      "max_dist_from_median_px": rng.normal(size=20), "lifespan_frames": rng.normal(size=20)})
    assert len(associations(d)) == 5
