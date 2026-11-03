"""Run on the Mac (needs internet). Lists papers from OpenAlex into knowledge/candidates.csv.
Usage: python scripts/11_collect_openalex.py "PAMAM dendrimer DNA AFM" [more queries...]
Only lists papers (title, authors, DOI, open-access link). Reading and evidence stay manual."""
import csv, json, sys, urllib.parse, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afm_pipeline import knowledge as k

rows = {}
for q in sys.argv[1:]:
    url = "https://api.openalex.org/works?per-page=25&search=" + urllib.parse.quote(q)
    with urllib.request.urlopen(url, timeout=30) as resp:
        for r in k.parse_openalex(json.load(resp)):
            rows[r["source_id"]] = r
out = Path(__file__).resolve().parents[1] / "knowledge" / "candidates.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=k.LIBRARY_COLS)
    w.writeheader(); w.writerows(rows.values())
print(len(rows), "papers ->", out)
