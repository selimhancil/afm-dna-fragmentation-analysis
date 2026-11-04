import numpy as np
import pandas as pd

from afm_pipeline.phase6b import (combine_verdicts, HysteresisSettings, classify_pair, find_peak, geodesic_between,
                                  merge_fragments, snap_to_skeleton, track_confidence)

H = HysteresisSettings()


def _blob(shape=(40, 40), c=(20, 20), amp=3.0, s=2.0):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return amp * np.exp(-((yy - c[0]) ** 2 + (xx - c[1]) ** 2) / (2 * s ** 2))


def test_find_peak_finds_weak_blob():
    z = _blob(amp=1.6)
    pk = find_peak(z, 1.0, (22, 19), 6, 1.25)
    assert pk and abs(pk["y"] - 20) <= 1 and abs(pk["x"] - 20) <= 1


def test_find_peak_rejects_too_weak_or_far():
    assert find_peak(_blob(amp=1.1), 1.0, (20, 20), 6, 1.25) is None
    assert find_peak(_blob(amp=3.0), 1.0, (5, 5), 4, 1.25) is None


def test_find_peak_requires_dna():
    z = _blob(amp=2.0)
    assert find_peak(z, 1.0, (20, 20), 6, 1.25, np.zeros(z.shape, bool)) is None


def test_geodesic_on_line_and_disconnected():
    sk = np.zeros((10, 30), bool)
    sk[5, 2:25] = True
    assert abs(geodesic_between(sk, (5, 2), (5, 22)) - 20) < 1e-6
    sk[5, 12] = False
    assert geodesic_between(sk, (5, 2), (5, 22)) is None


def test_snap_to_skeleton():
    sk = np.zeros((10, 30), bool)
    sk[5, 2:25] = True
    assert snap_to_skeleton(sk, 7, 10, 8) == (5, 10)
    assert snap_to_skeleton(sk, 9, 10, 2) is None


def test_classify_pair_constant_vs_changing():
    f = np.arange(10, dtype=float)
    rng = np.random.default_rng(0)
    const = classify_pair(f, 30 + rng.normal(0, 0.5, 10), H)
    chg = classify_pair(f, 30 + 3 * f + rng.normal(0, 0.5, 10), H)
    assert const["verdict"] == "spacing_constant"
    assert chg["verdict"] == "spacing_changes"
    assert classify_pair(f[:3], f[:3], H)["verdict"] == "too_few_frames"


def test_merge_fragments_joins_close_gap_only():
    rows = [(1, 1, 10, 10), (1, 2, 10, 11), (2, 4, 11, 11), (2, 5, 11, 12), (3, 4, 80, 80)]
    det = pd.DataFrame(rows, columns=["track_id", "rank", "y_common", "x_common"])
    out, n = merge_fragments(det, H)
    assert n == 1
    assert out.loc[out.track_id == 2].empty and set(out.track_id) == {1, 3}


def test_track_confidence_levels():
    trk = pd.DataFrame({"track_id": [1, 2, 3], "n_detections": [10, 4, 1],
                        "crosses_unreliable_registration": [False, False, False]})
    det = pd.DataFrame({"track_id": [1] * 10 + [2] * 4 + [3], "recovered": [False] * 5 + [True] * 5 + [False, True, True, True, False]})
    out = track_confidence(trk, det, H)
    assert list(out["confidence"]) == ["high", "medium", "low"]


def test_classify_pair_path_flip_and_relative_threshold():
    f = np.arange(10, dtype=float)
    d = np.array([180, 181, 179, 180, 90, 91, 90, 92, 91, 90], float)      # route switched once, big jump
    assert classify_pair(f, d, H)["verdict"] == "path_flip"
    rng = np.random.default_rng(1)
    big = 180 + 1.0 * f + rng.normal(0, 0.3, 10)                           # +9 px on 180 px: under the 10% rule
    assert classify_pair(f, big, H)["verdict"] == "spacing_constant"


def test_combine_verdicts():
    assert combine_verdicts("spacing_constant", "spacing_constant")[0] == "spacing_constant"
    assert combine_verdicts("path_flip", "spacing_constant")[0] == "spacing_constant"
    assert combine_verdicts("path_flip", "spacing_changes")[0] == "spacing_changes"
    assert combine_verdicts("spacing_constant", "spacing_changes")[0] == "conflicting_measures"
    assert combine_verdicts("spacing_changes", "spacing_constant")[0] == "conflicting_measures"
    assert combine_verdicts("path_flip", "too_few_frames")[0] == "undetermined"
    assert combine_verdicts("spacing_constant", "too_few_frames")[0] == "spacing_constant"
    assert combine_verdicts("too_few_frames", "too_few_frames")[0] == "undetermined"
