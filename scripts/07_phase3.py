"""Phase 3: tracking + candidate fragmentation events. Run: python scripts/07_phase3.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afm_pipeline.phase3 import main  # noqa: E402

raise SystemExit(main())
