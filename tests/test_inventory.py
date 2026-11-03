import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from afm_pipeline.inventory import (
    InventoryError,
    build_inventory,
    find_image_files,
    main,
    write_inventory_csv,
)


def make_gray(path: Path, width=280, height=275, seed=0):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (height, width), dtype=np.uint8), "L").save(path, "JPEG")


def make_rgb(path: Path, width=272, height=272, seed=1):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8), "RGB").save(path, "JPEG")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    make_gray(d / "1.jpg", seed=1)
    make_gray(d / "11.jpg", seed=2)
    make_rgb(d / "A1.jpg", seed=3)
    make_rgb(d / "B11.JPG", seed=4)
    return d


def test_ignores_macos_and_hidden_files(data_dir):
    (data_dir / ".DS_Store").write_bytes(b"junk")
    (data_dir / "._1.jpg").write_bytes(b"junk")
    (data_dir / ".gitkeep").write_text("")
    images, _ = find_image_files(data_dir)
    assert sorted(p.name for p in images) == ["1.jpg", "11.jpg", "A1.jpg", "B11.JPG"]


def test_reports_visible_non_jpg_files(data_dir):
    (data_dir / "notes.txt").write_text("hello")
    images, ignored = find_image_files(data_dir)
    assert len(images) == 4
    assert [p.name for p in ignored] == ["notes.txt"]


def test_missing_folder_gives_clear_error(tmp_path):
    with pytest.raises(InventoryError, match="Data folder not found"):
        find_image_files(tmp_path / "nope")


def test_empty_folder_gives_clear_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / ".DS_Store").write_bytes(b"x")
    with pytest.raises(InventoryError, match="No .jpg/.jpeg files"):
        find_image_files(empty)


def test_dimensions_and_channels(data_dir):
    df, _ = build_inventory(data_dir)
    row = df.set_index("filename")
    assert (row.loc["1.jpg", "width_px"], row.loc["1.jpg", "height_px"]) == (280, 275)
    assert row.loc["1.jpg", "channels"] == 1
    assert row.loc["A1.jpg", "channels"] == 3
    assert row.loc["B11.JPG", "width_px"] == 272


def test_pixel_statistics_are_sane(data_dir):
    df, _ = build_inventory(data_dir)
    assert (df["pixel_min"] >= 0).all() and (df["pixel_max"] <= 255).all()
    assert (df["pixel_min"] <= df["pixel_mean"]).all()
    assert (df["pixel_mean"] <= df["pixel_max"]).all()


def test_chronological_rank_is_left_empty(data_dir):
    df, _ = build_inventory(data_dir)
    assert df["chronological_rank"].isna().all()
    assert df["listing_index"].tolist() == [1, 2, 3, 4]


def test_no_height_column_is_invented(data_dir):
    df, _ = build_inventory(data_dir)
    forbidden = ("nm", "height_nm", "z_height", "afm")
    assert not any(c in forbidden for c in df.columns)
    assert all(c.startswith("pixel_") or "height" not in c or c == "height_px" for c in df.columns)


def test_originals_are_not_modified(data_dir, tmp_path):
    before = {p.name: sha(p) for p in data_dir.iterdir()}
    df, _ = build_inventory(data_dir)
    write_inventory_csv(df, tmp_path / "out" / "inv.csv", data_dir)
    after = {p.name: sha(p) for p in data_dir.iterdir()}
    assert before == after
    assert sorted(before) == sorted(after)  # no files added or removed


def test_corrupt_image_is_recorded_not_fatal(data_dir):
    (data_dir / "broken.jpg").write_bytes(b"this is not a jpeg")
    df, _ = build_inventory(data_dir)
    bad = df[df["filename"] == "broken.jpg"].iloc[0]
    assert "Error" in bad["read_error"]
    assert df[df["filename"] != "broken.jpg"]["read_error"].isna().all()


def test_refuses_to_write_csv_inside_data_folder(data_dir):
    df, _ = build_inventory(data_dir)
    with pytest.raises(InventoryError, match="Refusing to write"):
        write_inventory_csv(df, data_dir / "inventory.csv", data_dir)


def test_main_end_to_end(data_dir, tmp_path, capsys):
    out = tmp_path / "outputs" / "inv.csv"
    code = main(["--data-dir", str(data_dir), "--output", str(out), "--expected-count", "4"])
    assert code == 0
    assert out.exists()
    text = capsys.readouterr().out
    assert "NOT the chronological order" in text
    assert "NOT AFM height" in text


def test_main_returns_error_code_for_missing_folder(tmp_path, capsys):
    code = main(["--data-dir", str(tmp_path / "missing"), "--output", str(tmp_path / "o.csv")])
    assert code == 2
    assert "ERROR" in capsys.readouterr().err


def test_main_warns_on_wrong_count(data_dir, tmp_path, capsys):
    code = main(["--data-dir", str(data_dir), "--output", str(tmp_path / "o.csv"), "--expected-count", "74"])
    assert code == 0
    assert "Expected 74 images but found 4" in capsys.readouterr().out
