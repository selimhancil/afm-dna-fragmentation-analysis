"""Phase 4: measurements + sensitivity analysis. Run: python scripts/08_phase4.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afm_pipeline.phase4 import main  # noqa: E402

raise SystemExit(main())
