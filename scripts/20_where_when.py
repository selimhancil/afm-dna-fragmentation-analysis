"""Heading 5: when and where the DNA breaks (candidates). Run: python scripts/20_where_when.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afm_pipeline.where_when import main  # noqa: E402

raise SystemExit(main())
