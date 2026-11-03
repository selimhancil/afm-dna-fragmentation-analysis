import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from afm_pipeline.config import SegmentationConfig
from afm_pipeline.inventory import InventoryError
from afm_pipeline.segment import (
    extract_regions,
    gaussian_smooth,
    label_components,
    main,
    outline_from_labels,
    run_segmentation,
    segment_array,
)

SETTINGS = SegmentationConfig()  # strong 6, faint 3.5, list from 10 px, small<80, large>=320


def synth(row_offsets=True, seed=0, size=120):
    """Dark noisy background, optional scan-line offsets, one bright ring, one tiny speck."""
    rng = np.random.default_rng(seed)
    img = 40 + rng.normal(0, 3, (size, size))
    if row_offsets:
        img += rng.normal(0, 8, (size, 1))
    yy, xx = np.mgrid[:size, :size]
    dist = np.hypot(yy - size / 2, xx - size / 2)
    img[(dist > 33) & (dist < 37)] += 120
    img[5:7, 5:7] += 200
    gray = np.clip(img, 0, 255).astype(np.uint8)
    return np.stack([gray] * 3, axis=-1)


def save_jpeg(path: Path, arr):
    Image.fromarray(arr).save(path, "JPEG", quality=95)


def blank(size=60):
    return np.zeros((size, size))


# ------------------------------------------------------------------ basic math
def test_smoothing_keeps_constant_image_constant():
    assert np.allclose(gaussian_smooth(np.full((30, 30), 7.0), 1.5), 7.0)


def test_smoothing_with_zero_sigma_changes_nothing():
    img = np.random.default_rng(1).random((20, 20))
    assert np.array_equal(gaussian_smooth(img, 0), img)


def test_components_use_eight_connectivity():
    diagonal = np.zeros((5, 5), bool)
    diagonal[1, 1] = diagonal[2, 2] = True
    assert label_components(diagonal)[1] == 1
    apart = np.zeros((5, 5), bool)
    apart[0, 0] = apart[4, 4] = True
    assert label_components(apart)[1] == 2


def test_components_empty_mask():
    labels, n = label_components(np.zeros((6, 6), bool))
    assert n == 0 and labels.sum() == 0


def test_outline_separates_touching_regions():
    labels = np.zeros((10, 10), np.int32)
    labels[2:8, 1:5] = 1
    labels[2:8, 5:9] = 2
    edge = outline_from_labels(labels)
    assert edge[4, 4] and edge[4, 5]  # the shared border is outlined for both regions
    assert not edge[4, 2]  # interior pixel


# --------------------------------------------------- every piece kept separately
def test_each_separate_piece_gets_its_own_region():
    smooth = blank()
    smooth[5:15, 5:15] = 10  # large strong piece (100 px)
    smooth[30:34, 30:50] = 10  # elongated piece (80 px)
    smooth[50:53, 50:53] = 10  # tiny piece (9 px) -> below the listing limit
    res = extract_regions(smooth, 1.0, SETTINGS)
    assert [r["area_px"] for r in res.regions] == [100, 80]
    assert [r["region_id"] for r in res.regions] == [1, 2]
    assert res.n_below_min_area == 1
    assert set(np.unique(res.labels).tolist()) == {0, 1, 2}


def test_small_regions_are_kept_and_labelled_not_dropped():
    smooth = blank()
    smooth[5:9, 5:9] = 10  # 16 px: small but above the 10 px listing limit
    res = extract_regions(smooth, 1.0, SETTINGS)
    assert len(res.regions) == 1 and res.regions[0]["size_class"] == "small"


def test_size_classes():
    smooth = blank(100)
    smooth[2:12, 2:12] = 10  # 100 px -> medium
    smooth[20:40, 20:40] = 10  # 400 px -> large
    smooth[60:64, 60:64] = 10  # 16 px -> small
    classes = {r["area_px"]: r["size_class"] for r in extract_regions(smooth, 1.0, SETTINGS).regions}
    assert classes == {400: "large", 100: "medium", 16: "small"}


def test_faint_piece_is_found_as_its_own_tier():
    smooth = blank()
    smooth[5:15, 5:15] = 10  # strong (peak 10 sigma)
    smooth[30:40, 30:40] = 5  # dim: above 3.5 but below 6 sigma
    res = extract_regions(smooth, 1.0, SETTINGS)
    assert sorted(r["tier"] for r in res.regions) == ["faint", "strong"]
    faint = next(r for r in res.regions if r["tier"] == "faint")
    assert 3.5 < faint["peak_contrast_sigma"] < 6


def test_dim_halo_next_to_strong_region_is_not_a_separate_faint_region():
    smooth = blank()
    smooth[20:40, 20:40] = 5  # dim halo
    smooth[25:35, 25:35] = 10  # strong core inside it
    res = extract_regions(smooth, 1.0, SETTINGS)
    assert len(res.regions) == 1 and res.regions[0]["tier"] == "strong"
    assert res.regions[0]["area_px"] == 100  # tight outline at the strong level


def test_regions_joined_by_a_dim_stretch_share_a_linked_group():
    smooth = blank()
    smooth[20:30, 10:20] = 10  # strong piece A
    smooth[20:30, 30:40] = 10  # strong piece B
    smooth[22:28, 20:30] = 5  # dim bridge between them (above faint, below strong)
    smooth[45:55, 10:20] = 10  # strong piece C, far from the others
    res = extract_regions(smooth, 1.0, SETTINGS)
    groups = {(r["bbox_x0"], r["bbox_y0"]): r["linked_group_id"] for r in res.regions}
    assert len(res.regions) == 3
    assert groups[(10, 20)] == groups[(30, 20)]  # A and B are linked through the bridge
    assert groups[(10, 45)] != groups[(10, 20)]  # C is separate


def test_region_measurements():
    smooth = blank()
    smooth[10:13, 10:40] = 8  # 3 x 30 px bar, touching nothing
    r = extract_regions(smooth, 1.0, SETTINGS).regions[0]
    assert r["area_px"] == 90
    assert (r["bbox_x0"], r["bbox_x1"], r["bbox_y0"], r["bbox_y1"]) == (10, 39, 10, 12)
    assert r["approx_major_axis_px"] > 3 * r["approx_minor_axis_px"]
    assert r["touches_border"] is False and r["mean_contrast_sigma"] == 8.0


def test_border_touching_is_flagged():
    smooth = blank()
    smooth[0:10, 0:10] = 10
    assert extract_regions(smooth, 1.0, SETTINGS).regions[0]["touches_border"] is True


# ------------------------------------------------------------- whole-image runs
def test_ring_is_one_large_strong_region():
    res = segment_array(synth(), SETTINGS)
    large = [r for r in res.regions if r["size_class"] == "large"]
    assert len(large) == 1 and large[0]["tier"] == "strong"
    assert 800 < large[0]["area_px"] < 2200


def test_scan_line_offsets_do_not_change_the_result_much():
    a = segment_array(synth(row_offsets=True), SETTINGS)
    b = segment_array(synth(row_offsets=False), SETTINGS)
    la = max(r["area_px"] for r in a.regions)
    lb = max(r["area_px"] for r in b.regions)
    assert 0.7 < la / lb < 1.3


def test_pure_noise_gives_no_regions():
    rng = np.random.default_rng(3)
    gray = np.clip(rng.normal(40, 3, (100, 100)), 0, 255).astype(np.uint8)
    res = segment_array(np.stack([gray] * 3, axis=-1), SETTINGS)
    assert res.regions == [] and res.mask.sum() == 0


def test_flat_image_does_not_crash():
    res = segment_array(np.full((50, 50, 3), 40, np.uint8), SETTINGS)
    assert res.mask.sum() == 0


# ------------------------------------------------------------------ file output
@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    save_jpeg(d / "1.jpg", synth(seed=1))
    save_jpeg(d / "11.jpg", synth(seed=2))
    save_jpeg(d / "A1.jpg", synth(seed=3))
    (d / "A11.jpg").write_bytes(b"not a jpeg")
    return d


def test_run_writes_all_outputs_and_keeps_originals(data_dir, tmp_path):
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data_dir.iterdir()}
    res = run_segmentation(data_dir, tmp_path / "outputs" / "seg", SETTINGS)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data_dir.iterdir()}
    assert before == after
    for key in ("summary_path", "regions_path", "preview_path"):
        assert res[key].exists()
    assert len(res["summary"]) == 4
    assert sorted(p.name for p in res["masks_dir"].iterdir()) == ["11_mask.png", "1_mask.png", "A1_mask.png"]
    assert sorted(p.name for p in res["labels_dir"].iterdir()) == ["11_labels.png", "1_labels.png", "A1_labels.png"]


def test_label_image_matches_region_table(data_dir, tmp_path):
    res = run_segmentation(data_dir, tmp_path / "seg", SETTINGS)
    with Image.open(res["labels_dir"] / "1_labels.png") as im:
        labels = np.asarray(im)
    table = res["regions"]
    mine = table[table["filename"] == "1.jpg"]
    assert labels.shape == (120, 120)
    assert int(labels.max()) == len(mine)
    for _, r in mine.iterrows():
        assert int((labels == r["region_id"]).sum()) == r["area_px"]


def test_unreadable_image_is_recorded_not_fatal(data_dir, tmp_path):
    res = run_segmentation(data_dir, tmp_path / "seg", SETTINGS)
    s = res["summary"]
    bad = s[s["filename"] == "A11.jpg"].iloc[0]
    assert "could not be read" in bad["read_error"]
    assert s[s["filename"] != "A11.jpg"]["read_error"].isna().all()


def test_rows_follow_provisional_order_and_are_labelled(data_dir, tmp_path):
    res = run_segmentation(data_dir, tmp_path / "seg", SETTINGS)
    s = res["summary"]
    assert s["filename"].tolist() == ["1.jpg", "11.jpg", "A1.jpg", "A11.jpg"]
    assert (s["order_status"] == "provisional").all() and (s["run_label"] == "preliminary").all()
    assert res["regions"]["note"].str.contains("not AFM height").all()


def test_refuses_to_write_inside_data_folder(data_dir):
    with pytest.raises(InventoryError, match="Refusing to write"):
        run_segmentation(data_dir, data_dir / "seg", SETTINGS)


def test_main_returns_error_code_when_an_image_is_unreadable(data_dir, tmp_path, capsys):
    code = main(["--data-dir", str(data_dir), "--out-dir", str(tmp_path / "seg")])
    assert code == 1
    text = capsys.readouterr().out
    assert "unreadable" in text and "Not AFM height" in text


def test_main_ok_with_readable_images(data_dir, tmp_path, capsys):
    (data_dir / "A11.jpg").unlink()
    code = main(["--data-dir", str(data_dir), "--out-dir", str(tmp_path / "seg")])
    assert code == 0
    assert "PRELIMINARY" in capsys.readouterr().out


def test_main_error_for_missing_folder(tmp_path, capsys):
    code = main(["--data-dir", str(tmp_path / "missing"), "--out-dir", str(tmp_path / "seg")])
    assert code == 2
    assert "ERROR" in capsys.readouterr().err


# ------------------------------------------------------- nanometres and time ranges
from afm_pipeline.config import load_config  # noqa: E402


def _cfg(tmp_path, scale, time):
    path = tmp_path / "cfg.ini"
    path.write_text(
        "[project]\nexpected_image_count = 3\n[run]\nlabel = preliminary\n"
        f"[scale]\n{scale}\n[time]\n{time}\n",
        encoding="utf-8",
    )
    return load_config(path)


EST_SCALE = "status = estimated\nframe_size_nm = 1000\nsource = email"
EST_TIME = "status = estimated\nframe_interval_s = 31\ntotal_duration_s = 3600\nsource = email"
UNKNOWN = "status = unknown"


def test_without_config_nm_and_time_columns_stay_empty(data_dir, tmp_path):
    res = run_segmentation(data_dir, tmp_path / "seg", SETTINGS)
    r = res["regions"]
    for col in ("area_nm2", "approx_major_axis_nm", "nm_per_pixel_used", "time_earliest_s"):
        assert r[col].isna().all()
    assert res["summary"]["scale_status"].isna().all()


def test_estimated_scale_fills_nm_columns_consistently(data_dir, tmp_path):
    (data_dir / "A11.jpg").unlink()
    cfg = _cfg(tmp_path, EST_SCALE, UNKNOWN)
    res = run_segmentation(data_dir, tmp_path / "seg", SETTINGS, config=cfg)
    r, s = res["regions"], res["summary"]
    assert (r["scale_status"] == "estimated").all()
    assert (s["image_width_px"] == 120).all() and (s["image_height_px"] == 120).all()
    nmpp = 1000 / 120
    assert np.allclose(r["nm_per_pixel_used"], round(nmpp, 4))
    assert np.allclose(r["area_nm2"], r["area_px"] * nmpp**2, rtol=1e-3, atol=0.1)
    assert np.allclose(r["approx_major_axis_nm"].astype(float), r["approx_major_axis_px"] * nmpp, rtol=1e-3, atol=0.1)
    assert r["time_earliest_s"].isna().all()  # time was unknown in this config


def test_estimated_time_gives_ranges_that_follow_the_provisional_rank(data_dir, tmp_path):
    (data_dir / "A11.jpg").unlink()
    cfg = _cfg(tmp_path, UNKNOWN, EST_TIME)
    res = run_segmentation(data_dir, tmp_path / "seg", SETTINGS, config=cfg)
    s = res["summary"].set_index("chronological_rank")
    assert s.loc[1, "time_earliest_s"] == 0 and s.loc[1, "time_latest_s"] == 3600 - 3 * 31
    assert s.loc[3, "time_earliest_s"] == 62 and s.loc[3, "time_latest_s"] == 3600 - 31
    assert (s["time_status"] == "estimated").all()
    r = res["regions"]
    first = r[r["chronological_rank"] == 1].iloc[0]
    assert first["time_earliest_s"] == 0 and first["time_status"] == "estimated"
    assert r["area_nm2"].isna().all()  # scale was unknown in this config


def test_unknown_config_adds_nothing(data_dir, tmp_path):
    (data_dir / "A11.jpg").unlink()
    cfg = _cfg(tmp_path, UNKNOWN, UNKNOWN)
    res = run_segmentation(data_dir, tmp_path / "seg", SETTINGS, config=cfg)
    assert res["regions"]["area_nm2"].isna().all()
    assert res["summary"]["time_earliest_s"].isna().all()


def test_main_reports_scale_and_time_status_from_the_shipped_config(data_dir, tmp_path, capsys):
    (data_dir / "A11.jpg").unlink()
    code = main(["--data-dir", str(data_dir), "--out-dir", str(tmp_path / "seg")])
    text = capsys.readouterr().out
    assert code == 0
    assert "Scale used" in text and "estimated" in text and "not verified against raw data" in text
    assert "earliest and latest" in text
