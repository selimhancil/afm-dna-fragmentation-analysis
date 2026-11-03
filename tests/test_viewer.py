from afm_pipeline.viewer import build


def test_build_with_empty_project(tmp_path):
    (tmp_path / "outputs").mkdir()
    p = build(tmp_path)
    assert p.exists() and "AFM DNA" in p.read_text(encoding="utf-8")
