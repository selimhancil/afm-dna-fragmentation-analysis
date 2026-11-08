import json

from afm_pipeline import environment as env


def test_collect_has_core_packages():
    e = env.collect()
    assert e["packages"]["numpy"] != "not installed"
    assert e["python"].count(".") >= 1


def test_compare_reports_only_differences():
    a = {"python": "3.12", "machine": "arm64", "packages": {"numpy": "2.5.3", "scipy": "1.18.1"}}
    b = {"python": "3.12", "machine": "arm64", "packages": {"numpy": "2.5.4", "scipy": "1.18.1"}}
    assert env.compare(a, b) == ["numpy: 2.5.3 vs 2.5.4"]
    assert env.compare(a, a) == []


def test_main_writes_files(tmp_path, monkeypatch):
    monkeypatch.setattr(env, "__file__", str(tmp_path / "src" / "afm_pipeline" / "environment.py"))
    assert env.main() == 0
    data = json.loads((tmp_path / "outputs" / "environment" / "environment.json").read_text())
    assert "packages" in data
