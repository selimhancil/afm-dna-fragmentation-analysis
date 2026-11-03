import json

import numpy as np
import pandas as pd
import pytest

from afm_pipeline.phase4 import (bright_persistence, cumulative_shifts, event_bright_association,
                                 robustness_statement, sensitivity_summary)


def test_cumulative_shifts_add_up():
    reg = pd.DataFrame({"rank_from": [1, 2, 3], "rank_to": [2, 3, 4], "dy": [1.0, 2.0, -1.0],
                        "dx": [0.0, 1.0, 1.0]})
    s = cumulative_shifts(reg)
    assert s[1] == (0.0, 0.0) and s[2] == (1.0, 0.0) and s[4] == (2.0, 2.0)


def test_bright_persistence_counts_features_found_again():
    frames = pd.DataFrame({"rank": [1, 2], "crop_top": [0, 0], "crop_left": [0, 0]})
    bright = pd.DataFrame({"rank": [1, 1, 2, 2], "centroid_y_px": [10, 50, 11, 90],
                           "centroid_x_px": [10, 50, 10, 90]})
    reg = pd.DataFrame({"rank_from": [1], "rank_to": [2], "dy": [0.0], "dx": [0.0]})
    out = bright_persistence(bright, frames, reg).set_index("rank")
    assert np.isnan(out.loc[1, "bright_kept_from_previous"])
    assert out.loc[2, "bright_kept_from_previous"] == 0.5


def test_bright_persistence_uses_registration_shift():
    frames = pd.DataFrame({"rank": [1, 2], "crop_top": [0, 0], "crop_left": [0, 0]})
    bright = pd.DataFrame({"rank": [1, 2], "centroid_y_px": [10, 20], "centroid_x_px": [10, 10]})
    reg = pd.DataFrame({"rank_from": [1], "rank_to": [2], "dy": [-10.0], "dx": [0.0]})
    assert bright_persistence(bright, frames, reg).set_index("rank").loc[2, "bright_kept_from_previous"] == 1.0


def test_event_bright_association_distance():
    ev = pd.DataFrame({"event": ["split_candidate"], "rank_from": [1], "rank_to": [2], "persistence": ["persistent"],
                       "min_gap_px": [5.0], "event_x_px": [10.0], "event_y_px": [10.0]})
    br = pd.DataFrame({"rank": [1, 1], "centroid_x_px": [13.0, 100.0], "centroid_y_px": [14.0, 100.0]})
    assert event_bright_association(ev, br).iloc[0]["nearest_bright_feature_px"] == 5.0


def test_robustness_statement_requires_agreement_in_all_settings():
    ok = pd.DataFrame({"spearman_x": [-0.5, -0.7, -0.9]})
    mixed = pd.DataFrame({"spearman_x": [-0.5, 0.1, -0.9]})
    assert robustness_statement(ok)["spearman_x"].startswith("robust decrease")
    assert robustness_statement(mixed)["spearman_x"].startswith("NOT robust")


def test_sensitivity_summary_direction():
    rows = [{"setting": "a", "rank": r, "n_pieces": r, "largest_px": 100 - r,
             "continuity_fraction": 1 - r / 100, "n_closed_loops": 0} for r in range(1, 20)]
    s = sensitivity_summary(pd.DataFrame(rows)).iloc[0]
    assert s["spearman_n_pieces"] == 1.0 and s["spearman_largest_px"] == -1.0
    assert s["spearman_n_closed_loops"] is None
