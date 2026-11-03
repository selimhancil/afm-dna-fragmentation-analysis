#!/bin/bash
# Double-click on the Mac: builds the viewer page and opens it in the browser.
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null
python scripts/12_viewer.py && open outputs/viewer/index.html
