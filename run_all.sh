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
python scripts/14_phase7.py      # contour profile, gap map, behaviour classes, model comparison
python scripts/15_phase8.py      # change-point analysis of the fragmentation series
python scripts/12_viewer.py     # viewer page
echo "DONE. Results are in outputs/ (report: outputs/phase5/report.md)"
