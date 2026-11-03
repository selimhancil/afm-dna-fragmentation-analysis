import numpy as np
import pytest
from PIL import Image

from afm_pipeline.corrections import (CorrectionError, apply_corrections, evaluate, load_corrections)
from afm_pipeline.dna import (DnaConfigError, DnaSettings, analyse, crop_borders, dna_mask,
                              load_dna_settings, preprocess, skeleton_length)

S = DnaSettings()


def synthetic(ring=True, gap=0, size=160, seed=0):
    """Brown noisy background with a drawn ring (optionally with a gap) and a bright spot."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 60.0) + rng.normal(0, 3, (size, size))
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.hypot(yy - size / 2, xx - size / 2)
    band = (abs(r - 45) < 2.5)
    if gap:
        ang = np.degrees(np.arctan2(yy - size / 2, xx - size / 2))
        band &= ~(abs(ang) < gap)
    img[band] += 90
    rgb = np.stack([img, img * 0.6, img * 0.1], axis=2).clip(0, 255).astype(np.uint8)
    return rgb


def test_crop_borders_removes_white_frame():
    rgb = synthetic()
    rgb[:, -6:] = 255
    rgb[:3] = 255
    t, b, l, r = crop_borders(rgb, S)
    assert t >= 3 and r <= rgb.shape[1] - 6 and l == S.border_margin_px


def test_crop_borders_on_clean_image_only_trims_margin():
    rgb = synthetic()
    assert crop_borders(rgb, S) == (2, 158, 2, 158)


def test_crop_borders_refuses_all_white():
    with pytest.raises(ValueError):
        crop_borders(np.full((100, 100, 3), 255, np.uint8), S)


def test_closed_ring_is_a_closed_loop_candidate():
    pre = preprocess(synthetic(ring=True), S)
    res = analyse(pre, S)
    kinds = [o["kind"] for o in res.objects]
    assert "closed_loop_candidate" in kinds


def test_ring_with_gap_is_open_and_reports_bridge_radius():
    pre = preprocess(synthetic(gap=4), S)
    res = analyse(pre, S)
    big = max(res.objects, key=lambda o: o["skeleton_length_px"])
    assert big["kind"] == "open_candidate"
    assert big["n_endpoints"] >= 2


def test_small_objects_are_kept_not_dropped():
    rgb = synthetic()
    rgb[20:25, 20:25] = [200, 120, 20]  # a 5x5 speck
    res = analyse(preprocess(rgb, S), S)
    assert any(o["kind"] == "small_object" for o in res.objects)


def test_bright_feature_found_and_labelled_as_candidate():
    rgb = synthetic()
    yy, xx = np.mgrid[0:160, 0:160]
    rgb[np.hypot(yy - 35, xx - 80) < 6] = [255, 255, 200]
    res = analyse(preprocess(rgb, S), S)
    assert res.bright
    assert "candidate bound particle" in res.bright[0]["label"]
    assert "enzyme" not in res.bright[0]["label"].lower()


def test_blank_image_gives_clear_error():
    with pytest.raises(ValueError, match="noise"):
        preprocess(np.full((100, 100, 3), 90, np.uint8), S)


def test_skeleton_length_straight_and_diagonal():
    sk = np.zeros((10, 10), bool)
    sk[5, 1:9] = True
    assert skeleton_length(sk) == 7
    d = np.eye(10, dtype=bool)
    assert skeleton_length(d) == pytest.approx(9 * 2 ** 0.5)


def test_analysis_does_not_change_input_array():
    rgb = synthetic()
    before = rgb.copy()
    analyse(preprocess(rgb, S), S)
    assert np.array_equal(rgb, before)


def test_results_are_deterministic():
    a = analyse(preprocess(synthetic(), S), S)
    b = analyse(preprocess(synthetic(), S), S)
    assert a.objects == b.objects


def test_settings_defaults_when_section_missing(tmp_path):
    p = tmp_path / "c.ini"
    p.write_text("[project]\nexpected_image_count=1\n", encoding="utf-8")
    assert load_dna_settings(p) == DnaSettings()


def test_settings_read_and_validate(tmp_path):
    p = tmp_path / "c.ini"
    p.write_text("[dna]\nlow_k = 3\nhigh_k = 5\nbridge_radius_px = 1\n", encoding="utf-8")
    s = load_dna_settings(p)
    assert (s.low_k, s.high_k, s.bridge_radius_px) == (3.0, 5.0, 1)
    p.write_text("[dna]\nlow_k = 9\nhigh_k = 5\n", encoding="utf-8")
    with pytest.raises(DnaConfigError):
        load_dna_settings(p)
    p.write_text("[dna]\nlow_k = abc\n", encoding="utf-8")
    with pytest.raises(DnaConfigError):
        load_dna_settings(p)


def test_shipped_config_dna_section_is_valid():
    assert load_dna_settings().low_k > 0


# ---- corrections / evaluation

def test_corrections_erase_and_add(tmp_path):
    f = tmp_path / "c.csv"
    f.write_text("filename,action,x,y,radius_px\nA.jpg,erase,12,12,3\nA.jpg,add,30,30,2\n", encoding="utf-8")
    rows = load_corrections(f)["A.jpg"]
    mask = np.zeros((40, 40), bool)
    mask[10:15, 10:15] = True
    out = apply_corrections(mask, rows, (0, 40, 0, 40))
    assert not out[12, 12] and out[30, 30]


def test_corrections_use_original_coordinates_after_crop():
    mask = np.zeros((20, 20), bool)
    out = apply_corrections(mask, [{"action": "add", "x": 12, "y": 14, "r": 1}], (4, 24, 2, 22))
    assert out[10, 10]  # (x-left, y-top) = (10, 10)


def test_corrections_bad_files(tmp_path):
    f = tmp_path / "c.csv"
    f.write_text("filename,action,x,y,radius_px\nA.jpg,paint,1,1,1\n", encoding="utf-8")
    with pytest.raises(CorrectionError, match="action"):
        load_corrections(f)
    f.write_text("filename,action,x,y,radius_px\nA.jpg,add,1,1,-2\n", encoding="utf-8")
    with pytest.raises(CorrectionError, match="positive"):
        load_corrections(f)
    f.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(CorrectionError, match="columns"):
        load_corrections(f)
    assert load_corrections(tmp_path / "missing.csv") == {}


def test_evaluate_perfect_and_partial():
    m = np.zeros((10, 10), bool)
    m[2:6, 2:6] = True
    assert evaluate(m, m)["dice"] == 1.0
    half = m.copy()
    half[:, 4:] = False
    sc = evaluate(half, m)
    assert sc["precision"] == 1.0 and sc["recall"] == pytest.approx(0.5)
    with pytest.raises(CorrectionError):
        evaluate(m, np.zeros((5, 5), bool))


def test_phase2_end_to_end(tmp_path):
    from afm_pipeline.config import load_config
    from afm_pipeline.phase2 import run
    data = tmp_path / "raw"
    data.mkdir()
    Image.fromarray(synthetic(seed=1)).save(data / "1.jpg", quality=95)
    Image.fromarray(synthetic(gap=5, seed=2)).save(data / "11.jpg", quality=95)
    out = run(data, tmp_path / "out", S, load_config())
    assert out["n"] == 2
    assert (tmp_path / "out" / "frames.csv").is_file()
    assert (tmp_path / "out" / "review_begin_middle_end.pdf").is_file()
    assert (tmp_path / "out" / "parameters.json").is_file()
    assert sorted(p.name for p in data.iterdir()) == ["1.jpg", "11.jpg"]  # originals untouched


def test_prune_spurs_removes_short_side_branch_but_keeps_free_ends():
    from afm_pipeline.dna import prune_spurs
    sk = np.zeros((20, 30), bool)
    sk[10, 2:28] = True          # main strand with two free ends
    sk[6:10, 15] = True          # 4-px spur
    out = prune_spurs(sk, 6)
    assert not out[6:9, 15].any()
    assert out[10, 2] and out[10, 27]                      # free ends kept
    assert out.sum() <= sk.sum() - 3                       # spur shortened to at most its root pixel


def test_prune_spurs_keeps_long_branch():
    from afm_pipeline.dna import prune_spurs
    sk = np.zeros((30, 30), bool)
    sk[20, 2:28] = True
    sk[5:20, 15] = True          # 15-px branch
    assert prune_spurs(sk, 6)[5, 15]
