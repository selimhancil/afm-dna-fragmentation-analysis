"""Check the literature library and evidence table (no network needed)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afm_pipeline import knowledge as k

root = Path(__file__).resolve().parents[1] / "knowledge"
lib = k.load_library(root / "library.csv")
ev = k.load_evidence(root / "evidence.csv")
bad = k.validate(lib, ev)
print(k.summary(lib, ev))
for p in bad:
    print("PROBLEM:", p)
sys.exit(1 if bad else 0)
