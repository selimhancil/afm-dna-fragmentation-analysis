"""Show project settings and open questions:  python scripts/04_show_config.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afm_pipeline.config import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
