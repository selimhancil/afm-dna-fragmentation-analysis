"""Heading 3: height profile in ASSUMED nm. Run: python scripts/18_heights.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afm_pipeline.heights import main  # noqa: E402

raise SystemExit(main())
