"""Overview of the six requests + supervisor summary and email. Run: python scripts/22_summary.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afm_pipeline.summary import main  # noqa: E402

raise SystemExit(main())
