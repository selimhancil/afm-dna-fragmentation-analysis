"""Phase 7: contour profile, loss map, behaviour classes, model comparison. Run: python scripts/14_phase7.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afm_pipeline.phase7 import main  # noqa: E402

raise SystemExit(main())
