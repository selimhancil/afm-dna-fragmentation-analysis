"""Phase 5: figures + report. Run: python scripts/09_phase5.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afm_pipeline.phase5 import main  # noqa: E402

raise SystemExit(main())
