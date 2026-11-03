from pathlib import Path

import pytest

from afm_pipeline.config import (
    ConfigError,
    describe,
    load_config,
    main,
    nm_per_pixel_for_image,
    pixels_to_nm,
    time_bounds_s,
)

BASE = """
[project]
expected_image_count = 74

[run]
label = {label}

[scale]
{scale}

[time]
{time}
"""

UNKNOWN = "status = unknown"


def write_cfg(tmp_path: Path, label="preliminary", scale=UNKNOWN, time=UNKNOWN) -> Path:
    path = tmp_path / "cfg.ini"
    path.write_text(BASE.format(label=label, scale=scale, time=time), encoding="utf-8")
    return path


CONFIRMED_SCALE = "status = confirmed\nnm_per_pixel = 2.5\nsource = Dr. Abdelhady email"
CONFIRMED_TIME = "status = confirmed\nsource = raw AFM files"


def test_shipped_config_file_loads():
    cfg = load_config()
    assert cfg.expected_image_count >= 1
    assert cfg.run_label in {"preliminary", "final"}


def test_unknown_values_are_reported_as_unknown(tmp_path):
    cfg = load_config(write_cfg(tmp_path))
    assert not cfg.scale.known and cfg.scale.nm_per_pixel is None
    assert not cfg.time.known
    text = "\n".join(describe(cfg))
    assert "UNKNOWN" in text
    assert "NOT AVAILABLE from JPG" in text


def test_unknown_scale_refuses_conversion(tmp_path):
    cfg = load_config(write_cfg(tmp_path))
    with pytest.raises(ConfigError, match="still 'unknown'"):
        pixels_to_nm(10, cfg)


def test_confirmed_scale_converts(tmp_path):
    cfg = load_config(write_cfg(tmp_path, scale=CONFIRMED_SCALE))
    assert cfg.scale.known
    assert pixels_to_nm(10, cfg) == 25.0


def test_confirmed_scale_needs_source(tmp_path):
    bad = "status = confirmed\nnm_per_pixel = 2.5"
    with pytest.raises(ConfigError, match="no source"):
        load_config(write_cfg(tmp_path, scale=bad))


def test_confirmed_scale_must_be_positive_number(tmp_path):
    for value in ["0", "-3", "abc", "nan"]:
        bad = f"status = confirmed\nnm_per_pixel = {value}\nsource = x"
        with pytest.raises(ConfigError):
            load_config(write_cfg(tmp_path, scale=bad))


def test_value_with_unknown_status_is_rejected(tmp_path):
    bad = "status = unknown\nnm_per_pixel = 2.5"
    with pytest.raises(ConfigError, match="status is 'unknown'"):
        load_config(write_cfg(tmp_path, scale=bad))


def test_invalid_status_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="'unknown', 'estimated' or 'confirmed'"):
        load_config(write_cfg(tmp_path, scale="status = maybe"))


def test_confirmed_time_needs_source(tmp_path):
    with pytest.raises(ConfigError, match="no source"):
        load_config(write_cfg(tmp_path, time="status = confirmed"))


def test_final_label_requires_scale_and_time(tmp_path):
    with pytest.raises(ConfigError, match="not allowed"):
        load_config(write_cfg(tmp_path, label="final"))
    with pytest.raises(ConfigError, match="not allowed"):
        load_config(write_cfg(tmp_path, label="final", scale=CONFIRMED_SCALE))
    cfg = load_config(write_cfg(tmp_path, label="final", scale=CONFIRMED_SCALE, time=CONFIRMED_TIME))
    assert cfg.run_label == "final"


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.ini")


def test_malformed_file_gives_clear_error(tmp_path):
    path = tmp_path / "bad.ini"
    path.write_text("this is not an ini file", encoding="utf-8")
    with pytest.raises(ConfigError, match="Could not parse"):
        load_config(path)


def test_missing_section_gives_clear_error(tmp_path):
    path = tmp_path / "cfg.ini"
    path.write_text("[project]\nexpected_image_count = 74\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Missing"):
        load_config(path)


def test_bad_image_count_is_rejected(tmp_path):
    path = write_cfg(tmp_path)
    path.write_text(path.read_text().replace("74", "zero"), encoding="utf-8")
    with pytest.raises(ConfigError, match="whole number"):
        load_config(path)


def test_main_prints_report(tmp_path, capsys):
    code = main(["--config", str(write_cfg(tmp_path))])
    assert code == 0
    assert "Scale" in capsys.readouterr().out


def test_main_error_code_for_missing_file(tmp_path, capsys):
    code = main(["--config", str(tmp_path / "missing.ini")])
    assert code == 2
    assert "ERROR" in capsys.readouterr().err


def test_segmentation_defaults_when_section_missing(tmp_path):
    seg = load_config(write_cfg(tmp_path)).segmentation
    assert seg.threshold_k == 6.0 and seg.weak_threshold_k == 3.5
    assert (seg.min_region_area_px, seg.small_area_px, seg.large_area_px) == (10, 80, 320)


def test_segmentation_values_are_read(tmp_path):
    path = write_cfg(tmp_path)
    path.write_text(
        path.read_text()
        + "\n[segmentation]\nsmoothing_sigma_px = 2\nthreshold_k = 5.5\nweak_threshold_k = 3\n"
        "min_region_area_px = 8\nsmall_area_px = 50\nlarge_area_px = 400\n",
        encoding="utf-8",
    )
    seg = load_config(path).segmentation
    assert (seg.smoothing_sigma_px, seg.threshold_k, seg.weak_threshold_k) == (2.0, 5.5, 3.0)
    assert (seg.min_region_area_px, seg.small_area_px, seg.large_area_px) == (8, 50, 400)


def test_segmentation_invalid_values_are_rejected(tmp_path):
    bad_bodies = [
        "threshold_k = 0",
        "threshold_k = abc",
        "weak_threshold_k = 7",  # weak must not exceed strong
        "min_region_area_px = 0",
        "smoothing_sigma_px = -1",
        "small_area_px = 500\nlarge_area_px = 300",
        "min_region_area_px = 100\nsmall_area_px = 80",
    ]
    for body in bad_bodies:
        path = write_cfg(tmp_path)
        path.write_text(path.read_text() + "\n[segmentation]\n" + body + "\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path)


def test_describe_mentions_mask_settings(tmp_path):
    text = "\n".join(describe(load_config(write_cfg(tmp_path))))
    assert "Mask settings" in text and "Region listing" in text


ESTIMATED_SCALE = "status = estimated\nframe_size_nm = 1000\nsource = supervisor email"
ESTIMATED_TIME = "status = estimated\nframe_interval_s = 31\ntotal_duration_s = 3600\nsource = supervisor email"


def test_shipped_config_uses_estimates_from_the_supervisor_email():
    cfg = load_config()
    assert cfg.scale.status == "estimated" and cfg.scale.frame_size_nm == 1000
    assert cfg.time.status == "estimated"
    assert (cfg.time.frame_interval_s, cfg.time.total_duration_s) == (31, 3600)
    assert cfg.run_label == "preliminary"


def test_estimated_scale_is_parsed_and_usable_but_not_confirmed(tmp_path):
    cfg = load_config(write_cfg(tmp_path, scale=ESTIMATED_SCALE))
    assert cfg.scale.usable and not cfg.scale.known
    assert "ESTIMATED" in "\n".join(describe(cfg))


def test_estimated_scale_needs_frame_size_and_source(tmp_path):
    with pytest.raises(ConfigError, match="frame_size_nm is missing"):
        load_config(write_cfg(tmp_path, scale="status = estimated\nsource = x"))
    with pytest.raises(ConfigError, match="no source"):
        load_config(write_cfg(tmp_path, scale="status = estimated\nframe_size_nm = 1000"))


def test_estimated_scale_must_not_carry_nm_per_pixel(tmp_path):
    bad = "status = estimated\nframe_size_nm = 1000\nnm_per_pixel = 3.6\nsource = x"
    with pytest.raises(ConfigError, match="uses frame_size_nm"):
        load_config(write_cfg(tmp_path, scale=bad))


def test_estimated_values_cannot_be_used_for_a_final_run(tmp_path):
    with pytest.raises(ConfigError, match="not allowed"):
        load_config(write_cfg(tmp_path, label="final", scale=ESTIMATED_SCALE, time=ESTIMATED_TIME))


def test_nm_per_pixel_for_image_uses_mean_of_width_and_height(tmp_path):
    cfg = load_config(write_cfg(tmp_path, scale=ESTIMATED_SCALE))
    assert nm_per_pixel_for_image(cfg, 277, 275) == pytest.approx(1000 / 276)
    assert nm_per_pixel_for_image(cfg, 250, 250) == pytest.approx(4.0)


def test_nm_per_pixel_for_image_prefers_confirmed_value(tmp_path):
    cfg = load_config(write_cfg(tmp_path, scale=CONFIRMED_SCALE))
    assert nm_per_pixel_for_image(cfg, 277, 275) == 2.5


def test_nm_per_pixel_for_image_refuses_when_unknown(tmp_path):
    cfg = load_config(write_cfg(tmp_path))
    with pytest.raises(ConfigError, match="still 'unknown'"):
        nm_per_pixel_for_image(cfg, 277, 275)


def test_time_bounds_for_first_and_last_image(tmp_path):
    cfg = load_config(write_cfg(tmp_path, time=ESTIMATED_TIME))
    assert time_bounds_s(1, 74, cfg.time) == (0, 3600 - 74 * 31)  # 0 s .. 1306 s
    assert time_bounds_s(74, 74, cfg.time) == (73 * 31, 3600 - 31)  # 2263 s .. 3569 s


def test_time_bounds_are_ordered_and_never_cross(tmp_path):
    cfg = load_config(write_cfg(tmp_path, time=ESTIMATED_TIME))
    bounds = [time_bounds_s(k, 74, cfg.time) for k in range(1, 75)]
    assert all(lo <= hi for lo, hi in bounds)
    assert all(bounds[i][0] + 31 == bounds[i + 1][0] for i in range(73))
    assert all(bounds[i][1] + 31 == bounds[i + 1][1] for i in range(73))


def test_time_bounds_unknown_gives_none(tmp_path):
    cfg = load_config(write_cfg(tmp_path))
    assert time_bounds_s(5, 74, cfg.time) == (None, None)


def test_time_bounds_reject_impossible_numbers(tmp_path):
    cfg = load_config(write_cfg(tmp_path, time=ESTIMATED_TIME))
    with pytest.raises(ConfigError, match="outside"):
        time_bounds_s(75, 74, cfg.time)
    with pytest.raises(ConfigError, match="does not fit"):
        time_bounds_s(1, 200, cfg.time)  # 200 x 31 s > 3600 s


def test_estimated_time_needs_interval_total_and_source(tmp_path):
    with pytest.raises(ConfigError, match="missing"):
        load_config(write_cfg(tmp_path, time="status = estimated\nsource = x"))
    with pytest.raises(ConfigError, match="no source"):
        load_config(write_cfg(tmp_path, time="status = estimated\nframe_interval_s = 31\ntotal_duration_s = 3600"))


def test_unknown_time_with_values_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="status is 'unknown'"):
        load_config(write_cfg(tmp_path, time="status = unknown\nframe_interval_s = 31"))
