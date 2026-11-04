"""Heading 6: three-mechanism comparison. Run: python scripts/21_mechanisms.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afm_pipeline.mechanisms import main  # noqa: E402

raise SystemExit(main())
