"""Phase 2: preprocessing + DNA segmentation with overlays. Run: python scripts/06_phase2.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afm_pipeline.phase2 import main  # noqa: E402

raise SystemExit(main())
