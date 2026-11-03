import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from afm_pipeline.inventory import InventoryError
from afm_pipeline.order import (
    OrderError,
    build_provisional_order,
    main,
    parse_filename,
    write_order_csv,
)


def test_group_then_number_order():
    names = ["B1.jpg", "1.jpg", "A11.jpg", "A1.jpg", "11.jpg", "B11.jpg"]
    df = build_provisional_order(names)
    assert df["filename"].tolist() == ["1.jpg", "11.jpg", "A1.jpg", "A11.jpg", "B1.jpg", "B11.jpg"]
    assert df["chronological_rank"].tolist() == [1, 2, 3, 4, 5, 6]


def test_numeric_value_is_used_not_alphabetical():
    # Alphabetical sorting would put A10 before A2. The rule uses the numeric value.
    df = build_provisional_order(["A10.jpg", "A2.jpg"])
    assert df["filename"].tolist() == ["A2.jpg", "A10.jpg"]


def test_plain_numbers_come_before_letters():
    df = build_provisional_order(["A1.jpg", "111.jpg"])
    assert df["filename"].tolist() == ["111.jpg", "A1.jpg"]


def test_order_is_marked_provisional_and_has_no_time_axis():
    df = build_provisional_order(["1.jpg", "11.jpg"])
    assert df["order_status"].str.startswith("provisional").all()
    assert df["acquisition_time_s"].isna().all()
    assert df["time_gap_to_previous_s"].isna().all()
    assert df["order_source"].str.contains("Abdelhady").all()


def test_unexpected_filename_raises_clear_error():
    with pytest.raises(OrderError, match="does not follow the described pattern"):
        parse_filename("final_scan.jpg")


def test_lowercase_or_multi_letter_prefix_is_rejected():
    for bad in ["a1.jpg", "AB1.jpg", "A.jpg"]:
        with pytest.raises(OrderError):
            parse_filename(bad)


def test_same_position_twice_is_rejected():
    with pytest.raises(OrderError, match="same position"):
        build_provisional_order(["1.jpg", "01.jpg"])


def test_refuses_to_write_inside_data_folder(tmp_path):
    data = tmp_path / "raw"
    data.mkdir()
    df = build_provisional_order(["1.jpg"])
    with pytest.raises(InventoryError, match="Refusing to write"):
        write_order_csv(df, data / "order.csv", data)


def _make_images(folder: Path, names):
    folder.mkdir()
    for i, name in enumerate(names):
        rng = np.random.default_rng(i)
        Image.fromarray(rng.integers(0, 256, (50, 50, 3), dtype=np.uint8), "RGB").save(folder / name, "JPEG")


def test_main_end_to_end_and_originals_untouched(tmp_path, capsys):
    data = tmp_path / "raw"
    _make_images(data, ["B1.jpg", "1.jpg", "A1.jpg", "A11.jpg"])
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()}
    out = tmp_path / "outputs" / "order.csv"
    code = main(["--data-dir", str(data), "--output", str(out)])
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data.iterdir()}
    assert code == 0 and out.exists()
    assert before == after
    text = capsys.readouterr().out
    assert "PROVISIONAL" in text
    assert "NOT uniform" in text


def test_main_error_code_for_bad_filename(tmp_path, capsys):
    data = tmp_path / "raw"
    _make_images(data, ["1.jpg", "notes_final.jpg"])
    code = main(["--data-dir", str(data), "--output", str(tmp_path / "o.csv")])
    assert code == 2
    assert "ERROR" in capsys.readouterr().err
