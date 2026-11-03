"""Write the provisional chronological order:  python scripts/03_order.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afm_pipeline.order import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
