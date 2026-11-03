"""Phase 6: bright-feature tracking + mechanism summary. Run: python scripts/13_phase6.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afm_pipeline.phase6 import main  # noqa: E402

raise SystemExit(main())
