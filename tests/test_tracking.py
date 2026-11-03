import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from afm_pipeline.tracking import (Frame, TrackSettings, TrackingConfigError, load_phase2,
                                   load_track_settings, min_gap_px, register_pair, track)

S = TrackSettings()
SIZE = 140


def make_frame(rank, rects, shift=(0, 0)):
    """Frame with filled rectangles (y0, y1, x0, x1); each is one object."""
    mask = np.zeros((SIZE, SIZE), bool)
    for y0, y1, x0, x1 in rects:
        mask[y0 + shift[0]:y1 + shift[0], x0 + shift[1]:x1 + shift[1]] = True
    labels, n = ndi.label(mask, structure=np.ones((3, 3)))
    objs = pd.DataFrame({"skeleton_length_px": [float(np.sum(labels == i) / 6) for i in range(1, n + 1)]},
                        index=pd.Index(range(1, n + 1), name="object_id"))
    return Frame(rank, f"f{rank}.jpg", labels, objs, 0, 0)


LONG = (60, 66, 20, 120)                 # one long strand
HALVES = [(60, 66, 20, 65), (60, 66, 78, 120)]   # same strand with a 13 px gap


def kinds(res):
    ev = res["events"]
    return [] if ev.empty else list(ev["event"])


def test_identical_frames_continue_without_events():
    res = track([make_frame(1, [LONG]), make_frame(2, [LONG])], S)
    assert kinds(res) == []
    assert res["tracks"]["track_id"].nunique() == 1


def test_shifted_frame_is_registered_and_continues():
    res = track([make_frame(1, [LONG]), make_frame(2, [LONG], shift=(4, -3))], S)
    reg = res["registration"].iloc[0]
    assert (reg["dy"], reg["dx"]) == (-4.0, 3.0) and reg["reliable"]
    assert kinds(res) == []


def test_split_is_a_candidate_with_gap_and_never_confirmed():
    frames = [make_frame(1, [LONG])] + [make_frame(k, HALVES) for k in range(2, 8)]
    ev = track(frames, S)["events"]
    split = ev[ev["event"] == "split_candidate"].iloc[0]
    assert split["min_gap_px"] == pytest.approx(13, abs=1.5)
    assert split["persistence"] == "persistent"
    assert "confirmed cleavage" in split["note"] and "not a confirmed" in split["note"]


def test_small_gap_split_is_flagged_as_possible_artifact():
    close = [(60, 66, 20, 68), (60, 66, 72, 120)]           # 4 px gap
    ev = track([make_frame(1, [LONG]), make_frame(2, close)], S)["events"]
    assert "gap_small_possible_artifact" in ev.iloc[0]["flags"]


def test_split_that_rejoins_is_transient():
    frames = [make_frame(1, [LONG]), make_frame(2, HALVES), make_frame(3, [LONG]),
              make_frame(4, [LONG]), make_frame(5, [LONG])]
    ev = track(frames, S)["events"]
    split = ev[ev["event"] == "split_candidate"].iloc[0]
    assert split["persistence"] == "transient_rejoined"
    assert "merge_candidate" in set(ev["event"])


def test_split_near_end_of_series_is_unresolved():
    ev = track([make_frame(1, [LONG]), make_frame(2, HALVES)], S)["events"]
    assert ev[ev["event"] == "split_candidate"].iloc[0]["persistence"] == "unresolved_end_of_series"


def test_unrelated_frames_give_lost_and_new_object():
    a = make_frame(1, [(10, 16, 10, 60)])
    b = make_frame(2, [(120, 126, 80, 130)])
    kinds_ = set(kinds(track([a, b], S)))
    assert {"object_lost", "new_object"} <= kinds_


def test_short_pieces_after_split_do_not_make_a_split_candidate():
    tiny = [(60, 66, 20, 112), (60, 63, 118, 120)]           # second piece far shorter than 20 px
    ev = track([make_frame(1, [LONG]), make_frame(2, tiny)], S)["events"]
    assert "split_candidate" not in set(ev["event"]) if not ev.empty else True


def test_register_pair_reports_low_quality_for_unrelated_shapes():
    shape = (SIZE + 40, SIZE + 40)
    reg = register_pair(make_frame(1, [(10, 16, 10, 60)]), make_frame(2, [(120, 126, 80, 130)]), shape, S)
    assert not reg["reliable"]


def test_min_gap_px():
    labels = np.zeros((20, 40), int)
    labels[5, 2:10] = 1
    labels[5, 16:30] = 2
    assert min_gap_px(labels, [1, 2]) == 7.0
    assert min_gap_px(labels, [1]) is None


def test_track_needs_two_frames():
    with pytest.raises(ValueError):
        track([make_frame(1, [LONG])], S)


def test_settings_defaults_and_validation(tmp_path):
    p = tmp_path / "c.ini"
    p.write_text("[project]\nx=1\n", encoding="utf-8")
    assert load_track_settings(p) == TrackSettings()
    p.write_text("[tracking]\nmin_link_fraction = 0\n", encoding="utf-8")
    with pytest.raises(TrackingConfigError):
        load_track_settings(p)
    p.write_text("[tracking]\ngap_small_px = abc\n", encoding="utf-8")
    with pytest.raises(TrackingConfigError):
        load_track_settings(p)


def test_missing_phase2_results_give_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="06_phase2"):
        load_phase2(tmp_path)


def test_shipped_config_tracking_section_is_valid():
    assert load_track_settings().track_min_length_px > 0
