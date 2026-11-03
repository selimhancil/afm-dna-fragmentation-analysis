"""Literature library and evidence table.

Rules (kept in code so they cannot be forgotten):
- Every evidence row must point to a source in library.csv and carry a quote.
- Values are copied only from text that was actually read. Nothing is filled from memory.
- A new row is always 'unverified'. Only a person changes it.
"""
import csv
from pathlib import Path

LIBRARY_COLS = ["source_id", "title", "authors", "year", "journal", "doi", "url",
                "language", "access", "read_status", "note"]
EVIDENCE_COLS = ["evidence_id", "source_id", "topic", "value", "unit", "quote",
                 "context_note", "verified"]
VERIFIED_STATES = {"unverified", "user_verified", "supervisor_verified"}
ACCESS_STATES = {"open", "paywalled", "unknown"}


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_library(path):
    return _read(path)


def load_evidence(path):
    return _read(path)


def validate(library, evidence):
    """Return a list of problem strings (empty list = fine)."""
    problems = []
    ids = [r["source_id"] for r in library]
    if len(ids) != len(set(ids)):
        problems.append("duplicate source_id in library")
    for r in library:
        if set(r) != set(LIBRARY_COLS):
            problems.append(f"library columns wrong for {r.get('source_id')}")
        if r.get("access") not in ACCESS_STATES:
            problems.append(f"bad access value for {r['source_id']}")
        if not (r.get("doi") or r.get("url")):
            problems.append(f"{r['source_id']}: needs a DOI or a URL")
    known = set(ids)
    eids = [r["evidence_id"] for r in evidence]
    if len(eids) != len(set(eids)):
        problems.append("duplicate evidence_id")
    for r in evidence:
        if set(r) != set(EVIDENCE_COLS):
            problems.append(f"evidence columns wrong for {r.get('evidence_id')}")
        if r["source_id"] not in known:
            problems.append(f"{r['evidence_id']}: unknown source_id {r['source_id']}")
        if not r["quote"].strip():
            problems.append(f"{r['evidence_id']}: empty quote")
        if r["verified"] not in VERIFIED_STATES:
            problems.append(f"{r['evidence_id']}: bad verified value")
    return problems


def summary(library, evidence):
    n_ver = sum(r["verified"] != "unverified" for r in evidence)
    return {"sources": len(library), "evidence_rows": len(evidence), "verified_rows": n_ver}


def parse_openalex(payload):
    """Turn an OpenAlex /works JSON payload into library rows (read_status = 'listed_only')."""
    rows = []
    for w in payload.get("results", []):
        oa = w.get("open_access") or {}
        authors = "; ".join(a["author"]["display_name"] for a in w.get("authorships", [])
                            if a.get("author"))
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name", "")
        rows.append({
            "source_id": (w.get("id") or "").rsplit("/", 1)[-1],
            "title": w.get("title") or "", "authors": authors,
            "year": w.get("publication_year") or "", "journal": src, "doi": doi,
            "url": oa.get("oa_url") or w.get("id") or "",
            "language": w.get("language") or "",
            "access": "open" if oa.get("is_oa") else "unknown",
            "read_status": "listed_only", "note": ""})
    return rows
