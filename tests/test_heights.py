import numpy as np
import pandas as pd
import pytest

from afm_pipeline import config as cfgmod
from afm_pipeline.heights import (HeightConfig, contour_height_map, feature_heights, load_height_config,
                                  nm_per_noise_unit, plausibility)


def _write(tmp_path, body):
    p = tmp_path / "c.ini"
    p.write_text("[height]\n" + body, encoding="utf-8")
    return p


def test_default_config_is_assumed_and_usable():
    h = load_height_config()
    assert h.status == "assumed" and h.usable and h.z_range_nm == 10 and "ASSUMED" in h.label


def test_unknown_or_missing_section_is_unusable(tmp_path):
    assert not load_height_config(_write(tmp_path, "status = unknown\n")).usable
    p = tmp_path / "e.ini"
    p.write_text("[project]\nx=1\n")
    assert not load_height_config(p).usable


def test_config_requires_source_and_valid_numbers(tmp_path):
    with pytest.raises(cfgmod.ConfigError):
        load_height_config(_write(tmp_path, "status = assumed\nz_range_nm = 10\nsource =\n"))
    with pytest.raises(cfgmod.ConfigError):
        load_height_config(_write(tmp_path, "status = assumed\nz_range_nm = -1\nsource = x\n"))
    with pytest.raises(cfgmod.ConfigError):
        load_height_config(_write(tmp_path, "status = assumed\nz_range_nm = 10\nsource = x\nanchor_dna_height_nm = 0\n"))


def test_confirmed_label_and_anchor_parse(tmp_path):
    h = load_height_config(_write(tmp_path, "status = confirmed\nz_range_nm = 8\nsource = raw data\nanchor_dna_height_nm = 1.2\n"))
    assert h.label.startswith("measured") and h.anchor_dna_height_nm == 1.2


def test_nm_per_noise_unit_linear_in_range():
    h = HeightConfig("assumed", 10.0, 0.0, 100.0, "t")
    rgb = np.zeros((10, 10, 3))
    rgb[..., :] = np.linspace(0, 200, 100).reshape(10, 10, 1)       # luminance 0..200
    k = nm_per_noise_unit(rgb, noise=2.0, h=h)
    assert abs(k - 2.0 * 10 / 200) < 1e-6


def test_plausibility_flags_outside_range():
    h = HeightConfig("assumed", 10.0, source="t")
    assert plausibility(np.array([1.0, 1.2]), h)["within_plausible_range"]
    assert not plausibility(np.array([3.8, 4.0]), h)["within_plausible_range"]


def test_feature_heights_use_strong_only():
    det = pd.DataFrame({"track_id": [1, 1, 1], "rank": [1, 2, 3], "peak_z": [10.0, 10.0, 99.0],
                        "recovered": [False, False, True]})
    trk = pd.DataFrame({"track_id": [1], "motion_class": "stationary", "confidence": "high", "first_rank": 1, "last_rank": 3})
    out = feature_heights(det, trk, {1: 0.1, 2: 0.1, 3: 0.1}, {}, {1: 0.4, 2: 0.4, 3: 0.4})
    assert out.iloc[0].median_peak_nm == 1.0 and out.iloc[0].median_above_ridge_nm == 0.6 and out.iloc[0].n_strong == 2


def test_contour_height_map_masks_invalid_frames():
    m = contour_height_map(np.ones((2, 3)), np.array([2.0, 2.0]), np.array([0.5, 0.5]), np.array([True, False]))
    assert np.allclose(m[0], 1.0) and np.isnan(m[1]).all()
