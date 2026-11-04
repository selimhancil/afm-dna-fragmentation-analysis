# AFM DNA Fragmentation Analysis

Reproducible image-analysis pipeline for time-lapse AFM images of **pBR322 plasmid DNA (4,363 bp)**
cleaved by **DNase I** in the presence of **G4 PAMAM dendrimer**.

> Status: **preliminary, work in progress.** All results are in pixel units on exported JPG images.
> Nothing here is a confirmed biological conclusion.

## Data and honest limits

- Input: 74 exported JPG images (about 263-285 px per side). Raw AFM height data is **not** available.
- Image colours are **not** calibrated heights. No nanometre height is derived from them.
- Scale (1 µm per frame) and timing (31 s per scan, ~1 h session) come from the supervisor's email and are
  stored in `config/project_config.ini` as **estimated**, not confirmed. Exact scan times are unknown, so
  each image gets an earliest/latest possible time, not a single value.
- Chronological order is **provisional** (plain numbers first, then A, B, C...; numeric within a group).
- Bright features are called "bright features" / "candidate bound particles". They are not
  identified as enzyme or dendrimer.
- No fragmentation event is called confirmed by the software.

## Layout

```
config/project_config.ini   scale, time, segmentation settings (status: unknown / estimated / confirmed)
src/afm_pipeline/           inventory, contact_sheet, order, config, segment, dna, corrections, phase2, tracking, phase3, phase4, phase5, phase6, phase7, phase8, knowledge, viewer
annotations/                optional manual corrections.csv and annotation masks
scripts/                    thin runners: 01_inventory ... 05_segment
tests/                      pytest suite
data/raw/                   put the 74 JPGs here (not tracked by git)
outputs/                    generated results (not tracked by git)
```

## Setup (macOS)

```bash
bash setup.sh          # creates .venv, installs requirements, runs tests
source .venv/bin/activate
python scripts/01_inventory.py
python scripts/02_contact_sheet.py
python scripts/03_order.py
python scripts/04_show_config.py
python scripts/05_segment.py
bash run_all.sh        # phases 2 to 5 in one go
python -m pytest -q
```

Original images are opened read-only and never modified.

## Roadmap

1. Inspect images, sequence and quality (done for inventory/order/contact sheet)
2. Preprocessing and DNA segmentation with overlays (done, under review: `python scripts/06_phase2.py`)
3. Tracking and candidate fragmentation events (done: `python scripts/07_phase3.py`)
4. Quantification and time series (done: `python scripts/08_phase4.py`)
5. Figures (PNG + PDF), CSV/JSON export, report (done: `python scripts/09_phase5.py`)
6. Bright-feature tracking along the DNA contour, event-feature association, mechanism table H1/H2/H3 (done: `python scripts/13_phase6.py`)
7. Contour intensity profile and gap map, behavioural classes of bright features, likelihood-model comparison of the three mechanisms (done: `python scripts/14_phase7.py`)
8. Change-point analysis (when does the large step happen, 95% interval, 36-setting robustness, literature context) (done: `python scripts/15_phase8.py`)

Planned validation: diagnostic overlays, manual correction, comparison with manual annotations,
threshold sensitivity analysis.

## Viewer (app-like)

Double-click `AFM Viewer.command` (macOS) or run `python scripts/12_viewer.py`, then open `outputs/viewer/index.html`.
Tabs: frames (slider, play, layers, per-frame numbers, time-series chart), candidate events, literature table, figures, report.
