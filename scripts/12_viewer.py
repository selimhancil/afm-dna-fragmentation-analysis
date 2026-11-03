"""Build the viewer page: python scripts/12_viewer.py  (then open outputs/viewer/index.html)"""
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from afm_pipeline.viewer import build
print("Viewer:", build(root))
