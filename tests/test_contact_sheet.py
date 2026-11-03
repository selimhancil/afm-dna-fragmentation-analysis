import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from afm_pipeline.contact_sheet import build_contact_sheet, main
from afm_pipeline.inventory import InventoryError


def make_rgb(path: Path, seed=0, width=277, height=275):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8), "RGB").save(path, "JPEG")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    for i, name in enumerate(["1.jpg", "11.jpg", "A1.jpg", "B11.jpg", "C1.jpg"]):
        make_rgb(d / name, seed=i)
    return d


def test_creates_png_with_all_images(data_dir, tmp_path):
    out, n, unreadable = build_contact_sheet(data_dir, tmp_path / "out" / "sheet.png", columns=3)
    assert n == 5 and unreadable == []
    assert out.exists()
    with Image.open(out) as sheet:
        assert sheet.format == "PNG"
        assert sheet.width > 500 and sheet.height > 300


def test_originals_unchanged_and_nothing_added(data_dir, tmp_path):
    before = {p.name: sha(p) for p in data_dir.iterdir()}
    build_contact_sheet(data_dir, tmp_path / "sheet.png")
    after = {p.name: sha(p) for p in data_dir.iterdir()}
    assert before == after


def test_corrupt_image_is_marked_not_fatal(data_dir, tmp_path):
    (data_dir / "broken.jpg").write_bytes(b"not a jpeg")
    out, n, unreadable = build_contact_sheet(data_dir, tmp_path / "sheet.png")
    assert n == 6
    assert unreadable == ["broken.jpg"]
    assert out.exists()


def test_refuses_to_write_inside_data_folder(data_dir):
    with pytest.raises(InventoryError, match="Refusing to write"):
        build_contact_sheet(data_dir, data_dir / "sheet.png")


def test_invalid_columns_gives_clear_error(data_dir, tmp_path):
    with pytest.raises(InventoryError, match="--columns"):
        build_contact_sheet(data_dir, tmp_path / "s.png", columns=0)


def test_missing_folder_gives_clear_error(tmp_path):
    with pytest.raises(InventoryError, match="Data folder not found"):
        build_contact_sheet(tmp_path / "nope", tmp_path / "s.png")


def test_main_end_to_end(data_dir, tmp_path, capsys):
    out = tmp_path / "sheet.png"
    code = main(["--data-dir", str(data_dir), "--output", str(out), "--expected-count", "5"])
    assert code == 0 and out.exists()
    text = capsys.readouterr().out
    assert "NOT chronological order" in text


def test_main_error_code_for_missing_folder(tmp_path, capsys):
    code = main(["--data-dir", str(tmp_path / "missing"), "--output", str(tmp_path / "s.png")])
    assert code == 2
    assert "ERROR" in capsys.readouterr().err
