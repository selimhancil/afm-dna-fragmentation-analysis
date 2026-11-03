"""Phase 3: frame-to-frame registration, object tracking and candidate fragmentation events.

Works on the Phase 2 outputs (masks + objects.csv). Pixel units, provisional frame order.

What is tracked
---------------
Objects with a skeleton of at least `track_min_length_px`. Shorter objects stay in the
Phase 2 tables but are not linked (too little shape to match reliably).

How frames are aligned
----------------------
The mask of frame t+1 is shifted onto frame t by normalised cross-correlation of blurred
masks, searched only within +-max_shift_px (translation only). The shift and the
correlation quality (0..1) are written to registration.csv; a low quality marks the pair
as uncertain.

How objects are linked
----------------------
Object i (frame t) and object j (frame t+1) are linked when they overlap after alignment
(with a small dilation to tolerate movement). The links form groups:
    1 -> 1   continuation          (confidence = mean overlap fraction)
    1 -> N   split candidate       (a possible fragmentation event)
    N -> 1   merge candidate       (pieces re-joined, or touching molecules)
    0 -> 1   new object / 1 -> 0 object lost
    N -> M   complex (flagged uncertain)

What a "split candidate" means
------------------------------
Only that one object became several in the next frame. It is NOT a confirmed cleavage:
  * `min_gap_px` is the smallest distance between the new pieces; a small gap may be a
    thresholding or visibility artifact (flag `gap_small_possible_artifact`).
  * `persistence` is "persistent" if the pieces do not re-join within
    `persistence_frames` later frames, "transient_rejoined" if they do, and
    "unresolved_end_of_series" if the series ends first.
Confirmed fragmentation is never reported by this module.
"""
from __future__ import annotations

import configparser
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.signal import fftconvolve

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "project_config.ini"


class TrackingConfigError(ValueError):
    """The [tracking] settings are invalid."""


@dataclass(frozen=True)
class TrackSettings:
    track_min_length_px: float = 20.0    # objects shorter than this are not linked
    link_dilate_px: int = 3              # tolerance for small movement between frames
    min_link_fraction: float = 0.30      # overlap share needed to link two objects
    min_overlap_px: int = 15
    max_shift_px: float = 15.0           # larger registration shifts are flagged uncertain
    min_registration_quality: float = 0.5  # correlation below this: registration uncertain
    uncertain_confidence: float = 0.5
    gap_small_px: float = 8.0            # pieces closer than this: possible artifact
    persistence_frames: int = 3
    min_fragment_length_px: float = 20.0  # smaller new pieces do not make a split candidate


def load_track_settings(path: Path | str = DEFAULT_CONFIG_PATH) -> TrackSettings:
    d = TrackSettings()
    cp = configparser.ConfigParser(interpolation=None)
    try:
        cp.read_string(Path(path).read_text(encoding="utf-8"))
    except (OSError, configparser.Error) as exc:
        raise TrackingConfigError(f"Could not read {path}: {exc}") from exc
    if not cp.has_section("tracking"):
        return d
    values = {}
    for key, default in asdict(d).items():
        if cp.has_option("tracking", key):
            try:
                values[key] = type(default)(cp.get("tracking", key))
            except ValueError as exc:
                raise TrackingConfigError(f"[tracking] {key} is not a valid number: {exc}") from exc
    s = TrackSettings(**values)
    if not (0 < s.min_link_fraction <= 1) or s.link_dilate_px < 0 or s.persistence_frames < 1:
        raise TrackingConfigError("[tracking] min_link_fraction must be in (0,1], link_dilate_px >= 0, "
                                  "persistence_frames >= 1.")
    if s.track_min_length_px <= 0 or s.min_overlap_px < 1:
        raise TrackingConfigError("[tracking] track_min_length_px and min_overlap_px must be positive.")
    return s


# ------------------------------------------------------------------ data loading

@dataclass
class Frame:
    rank: int
    filename: str
    labels: np.ndarray                 # object label image (cropped frame)
    objects: pd.DataFrame              # Phase 2 objects of this frame (indexed by object_id)
    crop_top: int
    crop_left: int


def load_phase2(out_dir: Path) -> list[Frame]:
    out_dir = Path(out_dir)
    frames_csv, objects_csv = out_dir / "frames.csv", out_dir / "objects.csv"
    if not frames_csv.is_file() or not objects_csv.is_file():
        raise FileNotFoundError(f"Phase 2 results not found in {out_dir}. Run scripts/06_phase2.py first.")
    frames = pd.read_csv(frames_csv)
    objects = pd.read_csv(objects_csv)
    result = []
    for row in frames.sort_values("rank").itertuples():
        stem = Path(row.filename).stem
        mask_path = out_dir / "masks" / f"{int(row.rank):02d}_{stem}_mask.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask {mask_path}")
        mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        labels, _ = ndi.label(mask, structure=np.ones((3, 3)))   # same rule as Phase 2
        obj = objects[objects["rank"] == row.rank].set_index("object_id")
        result.append(Frame(int(row.rank), row.filename, labels, obj, int(row.crop_top), int(row.crop_left)))
    return result


# ------------------------------------------------------------------ registration

def _canvas(frame: Frame, shape: tuple[int, int], s: TrackSettings, long_only=True) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.float64)
    keep = frame.objects.index[frame.objects["skeleton_length_px"] >= s.track_min_length_px] \
        if long_only else frame.objects.index
    m = np.isin(frame.labels, list(keep))
    h, w = m.shape
    canvas[frame.crop_top:frame.crop_top + h, frame.crop_left:frame.crop_left + w] = m
    return ndi.gaussian_filter(canvas, 3.0)


def register_pair(a: Frame, b: Frame, shape: tuple[int, int], s: TrackSettings) -> dict:
    """Integer shift (dy, dx) that moves frame b onto frame a, with a quality score 0..1.

    Normalised cross-correlation of blurred long-object masks, searched within
    +-max_shift_px. quality = correlation peak / sqrt(energy_a * energy_b).
    """
    ca, cb = _canvas(a, shape, s), _canvas(b, shape, s)
    norm = float(np.sqrt((ca ** 2).sum() * (cb ** 2).sum()))
    if norm == 0:
        return {"dy": 0.0, "dx": 0.0, "quality": 0.0, "reliable": False}
    corr = fftconvolve(ca, cb[::-1, ::-1], mode="full") / norm
    h, w = shape
    r = int(s.max_shift_px)
    window = corr[h - 1 - r:h + r, w - 1 - r:w + r]
    iy, ix = np.unravel_index(int(np.argmax(window)), window.shape)
    dy, dx = float(iy - r), float(ix - r)
    quality = float(window[iy, ix])
    return {"dy": dy, "dx": dx, "quality": round(quality, 4),
            "reliable": bool(quality >= s.min_registration_quality)}


def _align_labels(b: Frame, a: Frame, reg: dict, shape: tuple[int, int]) -> np.ndarray:
    """Labels of frame b, shifted onto frame a's cropped coordinates."""
    canvas = np.zeros(shape, dtype=b.labels.dtype)
    h, w = b.labels.shape
    canvas[b.crop_top:b.crop_top + h, b.crop_left:b.crop_left + w] = b.labels
    moved = ndi.shift(canvas, (reg["dy"], reg["dx"]), order=0, mode="constant", cval=0)
    ah, aw = a.labels.shape
    return moved[a.crop_top:a.crop_top + ah, a.crop_left:a.crop_left + aw]


# ------------------------------------------------------------------ linking

def _tracked_ids(frame: Frame, s: TrackSettings) -> list[int]:
    return [int(i) for i in frame.objects.index[frame.objects["skeleton_length_px"] >= s.track_min_length_px]]


def link_pair(a: Frame, b_labels_on_a: np.ndarray, b: Frame, s: TrackSettings) -> list[dict]:
    """All overlap links between tracked objects of a and b (b already aligned to a)."""
    foot = np.ones((2 * s.link_dilate_px + 1,) * 2, dtype=bool)
    b_ids = _tracked_ids(b, s)
    links = []
    for i in _tracked_ids(a, s):
        ai = a.labels == i
        ai_d = ndi.binary_dilation(ai, structure=foot) if s.link_dilate_px else ai
        for j in b_ids:
            bj = b_labels_on_a == j
            n_bj = int(bj.sum())
            if n_bj == 0:
                continue
            inter = int((ai_d & bj).sum())
            if inter < s.min_overlap_px:
                continue
            bj_d = ndi.binary_dilation(bj, structure=foot) if s.link_dilate_px else bj
            frac_a = int((bj_d & ai).sum()) / int(ai.sum())   # share of old object covered by new
            frac_b = inter / n_bj                              # share of new object covered by old
            if max(frac_a, frac_b) >= s.min_link_fraction:
                links.append({"from_object": i, "to_object": j, "overlap_px": inter,
                              "frac_of_from": round(frac_a, 3), "frac_of_to": round(frac_b, 3)})
    return links


def _groups(links: list[dict], from_ids: list[int], to_ids: list[int]) -> list[tuple[list[int], list[int]]]:
    """Connected groups of the bipartite link graph (including unlinked singles)."""
    parent = {("a", i): ("a", i) for i in from_ids} | {("b", j): ("b", j) for j in to_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for l in links:
        parent[find(("a", l["from_object"]))] = find(("b", l["to_object"]))
    groups: dict = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    return [([n[1] for n in g if n[0] == "a"], [n[1] for n in g if n[0] == "b"]) for g in groups.values()]


def min_gap_info(labels: np.ndarray, ids: list[int]) -> tuple[float, tuple[float, float]] | None:
    """Smallest distance (px) between any two of the given objects, and where it is.

    The location (y, x) is the midpoint of the closest pair of pixels.
    """
    pts = [np.argwhere(labels == i) for i in ids]
    best, where = None, None
    for x in range(len(pts)):
        tree = cKDTree(pts[x])
        for y in range(x + 1, len(pts)):
            dist, idx = tree.query(pts[y], k=1)
            k = int(np.argmin(dist))
            if best is None or dist[k] < best:
                best = float(dist[k])
                where = tuple((pts[x][idx[k]] + pts[y][k]) / 2.0)
    return None if best is None else (round(best, 2), (round(where[0], 1), round(where[1], 1)))


def min_gap_px(labels: np.ndarray, ids: list[int]) -> float | None:
    """Smallest distance (px) between any two of the given objects."""
    info = min_gap_info(labels, ids)
    return None if info is None else info[0]


def track(frames: list[Frame], s: TrackSettings) -> dict:
    """Run registration + linking over the whole series. Returns DataFrames."""
    if len(frames) < 2:
        raise ValueError("Tracking needs at least two frames.")
    shape = (max(f.crop_top + f.labels.shape[0] for f in frames) + 40,
             max(f.crop_left + f.labels.shape[1] for f in frames) + 40)
    reg_rows, link_rows, event_rows, node_track = [], [], [], {}
    next_track = 1
    # tracks: (rank, object_id) -> track_id
    for obj_id in _tracked_ids(frames[0], s):
        node_track[(frames[0].rank, obj_id)] = next_track
        next_track += 1
    parents: dict[int, list[int]] = {}

    for a, b in zip(frames[:-1], frames[1:]):
        reg = register_pair(a, b, shape, s)
        reg_rows.append({"rank_from": a.rank, "rank_to": b.rank, **reg})
        b_on_a = _align_labels(b, a, reg, shape)
        links = link_pair(a, b_on_a, b, s)
        from_ids, to_ids = _tracked_ids(a, s), _tracked_ids(b, s)
        for l in links:
            link_rows.append({"rank_from": a.rank, "rank_to": b.rank, **l})
        for src, dst in _groups(links, from_ids, to_ids):
            conf_base = 1.0 if reg["reliable"] else 0.5
            if len(src) == 1 and len(dst) == 1:
                lk = next(x for x in links if x["from_object"] == src[0] and x["to_object"] == dst[0])
                conf = round(conf_base * (lk["frac_of_from"] + lk["frac_of_to"]) / 2, 3)
                node_track[(b.rank, dst[0])] = node_track[(a.rank, src[0])]
                if conf < s.uncertain_confidence:
                    event_rows.append(_event(a, b, "uncertain_continuation", src, dst, conf, reg, None, []))
            elif len(src) == 1 and len(dst) >= 2:
                lens = [float(b.objects.loc[j, "skeleton_length_px"]) for j in dst]
                big = [j for j, ln in zip(dst, lens) if ln >= s.min_fragment_length_px]
                conf = round(conf_base * min(1.0, np.mean([x["frac_of_to"] for x in links
                                                          if x["from_object"] == src[0]])), 3)
                for j in dst:
                    node_track[(b.rank, j)] = next_track
                    parents[next_track] = [node_track[(a.rank, src[0])]]
                    next_track += 1
                if len(big) >= 2:
                    info = min_gap_info(b_on_a, big)
                    gap = None if info is None else info[0]
                    loc = None if info is None else info[1]
                    flags = []
                    if gap is not None and gap <= s.gap_small_px:
                        flags.append("gap_small_possible_artifact")
                    ev = _event(a, b, "split_candidate", src, dst, conf, reg, gap, flags)
                    ev["event_y_px"], ev["event_x_px"] = (None, None) if loc is None else loc
                    event_rows.append(ev)
            elif len(src) >= 2 and len(dst) == 1:
                conf = round(conf_base * 0.8, 3)
                node_track[(b.rank, dst[0])] = next_track
                parents[next_track] = [node_track[(a.rank, i)] for i in src]
                next_track += 1
                event_rows.append(_event(a, b, "merge_candidate", src, dst, conf, reg, None, []))
            elif len(src) >= 2 and len(dst) >= 2:
                for j in dst:
                    node_track[(b.rank, j)] = next_track
                    parents[next_track] = [node_track[(a.rank, i)] for i in src]
                    next_track += 1
                event_rows.append(_event(a, b, "complex_change", src, dst, 0.3, reg, None, ["uncertain"]))
            elif len(src) == 1 and len(dst) == 0:
                event_rows.append(_event(a, b, "object_lost", src, dst, round(conf_base, 3), reg, None, []))
            elif len(src) == 0 and len(dst) == 1:
                node_track[(b.rank, dst[0])] = next_track
                next_track += 1
                event_rows.append(_event(a, b, "new_object", src, dst, round(conf_base, 3), reg, None, []))

    events = pd.DataFrame(event_rows)
    events = _persistence(events, frames, s) if len(events) else events
    tracks = pd.DataFrame([
        {"rank": r, "filename": next(f.filename for f in frames if f.rank == r), "object_id": o, "track_id": t}
        for (r, o), t in sorted(node_track.items())])
    if len(tracks):
        tracks["parent_tracks"] = tracks["track_id"].map(lambda t: ";".join(map(str, parents.get(t, []))))
    return {"registration": pd.DataFrame(reg_rows), "links": pd.DataFrame(link_rows),
            "events": events, "tracks": tracks}


def _event(a, b, kind, src, dst, conf, reg, gap, flags):
    flags = list(flags)
    if not reg["reliable"]:
        flags.append("registration_uncertain")
    if conf < 0.5 and "uncertain" not in flags:
        flags.append("uncertain")
    return {
        "rank_from": a.rank, "rank_to": b.rank, "filename_from": a.filename, "filename_to": b.filename,
        "event": kind, "from_objects": ";".join(map(str, src)), "to_objects": ";".join(map(str, dst)),
        "n_pieces_after": len(dst), "min_gap_px": gap, "confidence": conf,
        "registration_shift_px": round(float(np.hypot(reg["dy"], reg["dx"])), 2),
        "flags": ";".join(flags), "persistence": "",
        "note": "candidate only; not a confirmed cleavage",
    }


def _persistence(events: pd.DataFrame, frames: list[Frame], s: TrackSettings) -> pd.DataFrame:
    """Mark each split candidate persistent / transient_rejoined / unresolved_end_of_series."""
    last = frames[-1].rank
    merges = events[events["event"].isin(["merge_candidate", "complex_change"])]
    out = []
    for row in events.itertuples():
        if row.event != "split_candidate":
            out.append("")
            continue
        window = merges[(merges["rank_from"] >= row.rank_to) &
                        (merges["rank_to"] <= row.rank_to + s.persistence_frames)]
        if len(window):
            out.append("transient_rejoined")
        elif row.rank_to + s.persistence_frames > last:
            out.append("unresolved_end_of_series")
        else:
            out.append("persistent")
    events = events.copy()
    events["persistence"] = out
    return events
