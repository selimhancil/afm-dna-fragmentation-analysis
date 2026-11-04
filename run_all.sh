#!/bin/bash
# Runs the whole pipeline in order. Usage: bash run_all.sh
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
python scripts/06_phase2.py      # preprocessing + DNA segmentation + overlays
python scripts/07_phase3.py      # registration, tracking, candidate split events
python scripts/08_phase4.py      # measurements + sensitivity analysis (about 3 minutes)
python scripts/09_phase5.py      # figures (PNG + PDF) and report
python scripts/13_phase6.py      # bright-feature tracking + mechanism summary
python scripts/16_phase6b.py     # heading 1: hysteresis tracking, confidence, spacing along DNA, photo strips
python scripts/17_phase6c.py     # heading 2: feature populations (descriptive, no identification)
python scripts/18_heights.py     # heading 3: height profile in ASSUMED nm (see [height] in the config)
python scripts/19_lengths.py     # heading 4: contour length over time with estimator bias and scale check
python scripts/14_phase7.py      # contour profile, gap map, behaviour classes, model comparison
python scripts/15_phase8.py      # change-point analysis of the fragmentation series
python scripts/12_viewer.py     # viewer page
echo "DONE. Results are in outputs/ (report: outputs/phase5/report.md)"
