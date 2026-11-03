from pathlib import Path
from afm_pipeline import knowledge as k

ROOT = Path(__file__).resolve().parents[1] / "knowledge"


def test_real_tables_are_valid_and_unverified():
    lib = k.load_library(ROOT / "library.csv")
    ev = k.load_evidence(ROOT / "evidence.csv")
    assert k.validate(lib, ev) == []
    assert all(r["verified"] == "unverified" for r in ev)


def test_validate_catches_problems():
    lib = [{c: "x" for c in k.LIBRARY_COLS}]
    lib[0]["access"] = "open"
    ev = [{c: "" for c in k.EVIDENCE_COLS}]
    ev[0].update(evidence_id="E1", source_id="nope", verified="maybe")
    problems = k.validate(lib, ev)
    assert any("unknown source_id" in p for p in problems)
    assert any("empty quote" in p for p in problems)
    assert any("bad verified" in p for p in problems)


def test_parse_openalex():
    payload = {"results": [{"id": "https://openalex.org/W1", "title": "T", "doi": "https://doi.org/10.1/x",
               "publication_year": 2020, "authorships": [{"author": {"display_name": "A B"}}],
               "open_access": {"is_oa": True, "oa_url": "http://u"},
               "primary_location": {"source": {"display_name": "J"}}}]}
    row = k.parse_openalex(payload)[0]
    assert row["source_id"] == "W1" and row["doi"] == "10.1/x" and row["access"] == "open"
    assert row["read_status"] == "listed_only"
