"""Heading 2: dendrimer vs enzyme, descriptive populations. Run: python scripts/17_phase6c.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afm_pipeline.phase6c import main  # noqa: E402

raise SystemExit(main())
