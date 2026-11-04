"""Phase 8: change-point analysis. Run: python scripts/15_phase8.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afm_pipeline.phase8 import main  # noqa: E402

raise SystemExit(main())
