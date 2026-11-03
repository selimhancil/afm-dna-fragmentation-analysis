import numpy as np
import pandas as pd

from afm_pipeline.phase6 import (FeatureSettings, fate_after, link_features, summarise_tracks)


def det(rank, y, x, area=40, fid=1):
    return {"rank": rank, "feature_id": fid, "y_common": y, "x_common": x, "area_px": area,
            "relative_to_ridge_level": 2.0, "s_px": x, "off_reference_contour": False}


def test_stationary_feature_is_one_track_and_new_one_starts_far_away():
    rows = [det(r, 50 + (r % 2), 60) for r in range(1, 7)] + [det(3, 150, 150, fid=2)]
    b = link_features(pd.DataFrame(rows), FeatureSettings())
    assert b[b["x_common"] == 60]["track_id"].nunique() == 1
    assert b[b["x_common"] == 150]["track_id"].iloc[0] != b[b["x_common"] == 60]["track_id"].iloc[0]


def test_gap_is_closed_but_long_gap_is_not():
    s = FeatureSettings()
    b = link_features(pd.DataFrame([det(1, 10, 10), det(3, 11, 10), det(7, 10, 10)]), s)
    assert b["track_id"].iloc[0] == b["track_id"].iloc[1]          # 1 missed frame: linked
    assert b["track_id"].iloc[2] != b["track_id"].iloc[0]          # 3 missed frames: new track
    assert b["gap_before"].iloc[1] == 1


def test_big_area_change_blocks_link():
    b = link_features(pd.DataFrame([det(1, 10, 10, area=20), det(2, 10, 10, area=200)]), FeatureSettings())
    assert b["track_id"].nunique() == 2


def test_motion_classes():
    s = FeatureSettings()
    still = [det(r, 20, 30) for r in range(1, 7)]
    move = [det(r, 20, 30 + 4 * r) for r in range(1, 7)]
    for d in still:
        d["track_id"] = 1
    for d in move:
        d["track_id"] = 2
    t = summarise_tracks(pd.DataFrame(still + move), 6, {}, s).set_index("track_id")
    assert t.loc[1, "motion_class"] == "stationary" and t.loc[2, "motion_class"] == "moved"
    assert t.loc[1, "fate"] == "present_to_last_frame"
    assert (t["label"] == "bright feature (candidate bound particle)").all()


def test_fate_after():
    s = FeatureSettings()
    b = pd.DataFrame([det(2, 10, 10), det(3, 11, 10), det(2, 80, 80, fid=2), det(3, 90, 140, fid=3), det(4, 90, 140, fid=3)])
    assert fate_after(b, 1, 10, 10, s) == "kept_in_place"
    assert fate_after(b, 2, 80, 80, s) == "lost"
    assert fate_after(b, 4, 10, 10, s) == "unknown_end_of_series"
