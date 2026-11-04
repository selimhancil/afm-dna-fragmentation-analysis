import numpy as np
import pandas as pd

from afm_pipeline.where_when import (WhereWhenSettings, _ids, cut_fraction_tests, event_table, step_decomposition)

S = WhereWhenSettings()


def test_ids_parse():
    assert _ids("10;17") == [10, 17] and _ids(float("nan")) == [] and _ids("5") == [5]


def _inputs():
    events = pd.DataFrame({"event": ["split_candidate", "merge_candidate"], "rank_from": [2, 2], "rank_to": [3, 3],
                           "from_objects": ["12", "1"], "to_objects": ["10;17", "2"], "n_pieces_after": [2, 1],
                           "persistence": ["persistent", None]})
    objects = pd.DataFrame({"rank": [2, 3, 3], "object_id": [12, 10, 17], "skeleton_length_px": [100.0, 30.0, 70.0]})
    assoc = pd.DataFrame({"rank_from": [2], "rank_to": [3], "event_s_px": [40.0], "event_dist_to_ref_contour_px": [2.0]})
    frames = pd.DataFrame({"rank": [2, 3], "time_earliest_s": [31.0, 62.0], "time_latest_s": [900.0, 931.0]})
    return events, objects, assoc, frames


def test_event_table_cut_fraction_and_reliability():
    events, objects, assoc, frames = _inputs()
    ev = event_table(events, objects, assoc, {2: True, 3: True}, frames, 49, 52, S)
    assert len(ev) == 1                                  # only split candidates
    r = ev.iloc[0]
    assert abs(r.cut_fraction_shortest_piece - 0.3) < 1e-9 and r.conserved and r.contour_position_reliable
    assert r.relative_to_change_point == "before"
    bad = event_table(events, objects, assoc, {2: True, 3: False}, frames, 49, 52, S).iloc[0]
    assert not bad.contour_position_reliable


def test_conservation_flag():
    events, objects, assoc, frames = _inputs()
    objects.loc[objects.object_id == 17, "skeleton_length_px"] = 5.0
    assert not event_table(events, objects, assoc, {2: True, 3: True}, frames, 49, 52, S).iloc[0].conserved


def test_cut_fraction_tests_too_few_and_uniform():
    ev = pd.DataFrame({"conserved": True, "n_new_pieces": 2, "cut_fraction_shortest_piece": [0.1, 0.3]})
    assert cut_fraction_tests(ev, S)["verdict"] == "too_few_events"
    f = np.random.default_rng(0).uniform(0, 0.5, 40)
    ev = pd.DataFrame({"conserved": True, "n_new_pieces": 2, "cut_fraction_shortest_piece": f})
    assert cut_fraction_tests(ev, S)["verdict"] == "cut_positions_consistent_with_uniform"
    ev2 = pd.DataFrame({"conserved": True, "n_new_pieces": 2, "cut_fraction_shortest_piece": np.full(20, 0.02)})
    assert cut_fraction_tests(ev2, S)["verdict"] == "cut_positions_differ_from_uniform"


def test_step_decomposition():
    L = pd.DataFrame({"rank": np.arange(1, 21), "largest_corrected_px": [400.0] * 10 + [200.0] * 10,
                      "total_corrected_px": 500.0, "n_pieces": [2] * 10 + [4] * 10})
    d = step_decomposition(L, 11, S)
    assert d["largest_piece_before_px"] == 400.0 and d["largest_piece_after_px"] == 200.0 and d["pieces_after"] == 4.0
