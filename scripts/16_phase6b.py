"""Phase 6b: stronger bright-feature tracking. Run: python scripts/16_phase6b.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afm_pipeline.phase6b import main  # noqa: E402

raise SystemExit(main())
