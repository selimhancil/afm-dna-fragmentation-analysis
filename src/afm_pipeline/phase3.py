"""Phase 3 runner: tracking and candidate fragmentation events from the Phase 2 results."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import config as cfgmod
from .inventory import PROJECT_ROOT
from .tracking import (TrackingConfigError, load_phase2, load_track_settings, track)

DEFAULT_PHASE2 = PROJECT_ROOT / "outputs" / "phase2"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "phase3"


def run(phase2_dir: Path, out_dir: Path, settings) -> dict:
    frames = load_phase2(phase2_dir)
    res = track(frames, settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("registration", "links", "events", "tracks"):
        res[name].to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "parameters.json").write_text(json.dumps(
        {"tracking_settings": asdict(settings),
         "notes": "frame order is provisional; events are candidates, never confirmed cleavage"},
        indent=2), encoding="utf-8")
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase 3: tracking and candidate fragmentation events.")
    p.add_argument("--phase2-dir", type=Path, default=DEFAULT_PHASE2)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--config", type=Path, default=cfgmod.DEFAULT_CONFIG_PATH)
    a = p.parse_args(argv)
    try:
        res = run(a.phase2_dir, a.out_dir, load_track_settings(a.config))
    except (TrackingConfigError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    ev = res["events"]
    print(f"Frame pairs: {len(res['registration'])}   unreliable registrations: "
          f"{int((~res['registration']['reliable']).sum())}")
    if len(ev):
        print(ev["event"].value_counts().to_string())
        sp = ev[ev["event"] == "split_candidate"]
        print(f"Split candidates: {len(sp)}  (persistent: {int((sp['persistence'] == 'persistent').sum())})")
    print(f"Results: {a.out_dir}")
    print("Events are candidates only: nothing here confirms DNA cleavage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
