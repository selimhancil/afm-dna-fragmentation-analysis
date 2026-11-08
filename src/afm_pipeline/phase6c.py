"""Heading 2: can the bright features be told apart (dendrimer vs enzyme)?

Honest framing: nothing in the JPG labels a feature. What can be tested from size, brightness,
mobility and lifetime of each tracked feature is:
  (a) do features fall into more than one population? (1 vs 2 component mixture, parametric
      bootstrap likelihood-ratio test on log area and relative brightness);
  (b) are features individually consistent over time (between-track spread vs within-track spread,
      intraclass correlation), i.e. are they discrete objects with their own size?
  (c) do size/brightness relate to mobility or lifetime?
Only strong detections (Phase 2 rule) are used for size/brightness: recovered weak peaks are measured with
a different threshold and would bias area.
The result is a descriptive grouping ("population A / B"), never an identification.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .inventory import PROJECT_ROOT

DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"


@dataclass
class DiscSettings:
    min_strong: int = 3
    n_boot: int = 300
    seed: int = 1
    alpha: float = 0.05
    icc_min_tracks: int = 5


class _Gmm:
    """Minimal full-covariance Gaussian mixture fitted by EM (numpy only)."""

    def __init__(self, k, seed=1, n_init=5, iters=200, reg=1e-4):
        self.k, self.seed, self.n_init, self.iters, self.reg = k, seed, n_init, iters, reg

    @staticmethod
    def _logpdf(X, mu, cov):
        d = X.shape[1]
        diff = X - mu
        sol = np.linalg.solve(cov, diff.T).T
        _, logdet = np.linalg.slogdet(cov)
        return -0.5 * (np.sum(diff * sol, axis=1) + logdet + d * np.log(2 * np.pi))

    def _comp_logs(self, X):
        return np.column_stack([np.log(w) + self._logpdf(X, m, c)
                                for w, m, c in zip(self.weights_, self.means_, self.covs_)])

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        n, d = X.shape
        best = None
        for _ in range(self.n_init):
            if self.k == 1:
                resp = np.ones((n, 1))
            else:
                centers = X[rng.choice(n, self.k, replace=False)]
                lab = np.argmin(((X[:, None, :] - centers[None]) ** 2).sum(-1), axis=1)
                resp = np.eye(self.k)[lab]
            for _ in range(self.iters):
                nk = resp.sum(0) + 1e-9
                self.weights_ = nk / n
                self.means_ = (resp.T @ X) / nk[:, None]
                self.covs_ = [((resp[:, j, None] * (X - self.means_[j])).T @ (X - self.means_[j])) / nk[j]
                              + self.reg * np.eye(d) for j in range(self.k)]
                lg = self._comp_logs(X)
                mx = lg.max(1, keepdims=True)
                new = np.exp(lg - mx)
                new /= new.sum(1, keepdims=True)
                if np.abs(new - resp).max() < 1e-7:
                    resp = new
                    break
                resp = new
            ll = self.score(X) * n
            if best is None or ll > best[0]:
                best = (ll, self.weights_.copy(), self.means_.copy(), [c.copy() for c in self.covs_])
        _, self.weights_, self.means_, self.covs_ = best
        return self

    def score(self, X):
        lg = self._comp_logs(X)
        mx = lg.max(1)
        return float(np.mean(mx + np.log(np.exp(lg - mx[:, None]).sum(1))))

    def predict(self, X):
        return np.argmax(self._comp_logs(X), axis=1)

    def bic(self, X):
        d = X.shape[1]
        npar = self.k * d + self.k * d * (d + 1) / 2 + (self.k - 1)
        return float(-2 * self.score(X) * len(X) + npar * np.log(len(X)))

    def sample(self, n, rng):
        lab = rng.choice(self.k, n, p=self.weights_)
        return np.vstack([rng.multivariate_normal(self.means_[j], self.covs_[j]) for j in lab])


def track_descriptors(det: pd.DataFrame, trk: pd.DataFrame, s: DiscSettings) -> pd.DataFrame:
    strong = det[~det["recovered"]]
    g = strong.groupby("track_id")
    d = pd.DataFrame({"n_strong": g.size(),
                      "median_area_px": g["area_px"].median(),
                      "median_relative_intensity": g["relative_to_ridge_level"].median(),
                      "area_within_sd_log": g["area_px"].apply(lambda v: float(np.std(np.log(v), ddof=1)) if len(v) > 2 else np.nan)})
    d = d[d["n_strong"] >= s.min_strong].reset_index()
    cols = ["track_id", "motion_class", "max_dist_from_median_px", "lifespan_frames", "confidence"]
    d = d.merge(trk[cols], on="track_id", how="left")
    d["log_area"] = np.log(d["median_area_px"])
    return d


def mixture_test(X: np.ndarray, n_boot: int, seed: int) -> dict:
    """1 vs 2 Gaussian components; parametric bootstrap of the likelihood-ratio statistic."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float)
    if len(X) < 8:
        return {"verdict": "too_few_tracks", "n": int(len(X))}

    def fit(Z, k):
        return _Gmm(k, seed=seed).fit(Z)

    g1, g2 = fit(X, 1), fit(X, 2)
    lr = 2 * (g2.score(X) - g1.score(X)) * len(X)
    null = []
    for _ in range(n_boot):
        Z = g1.sample(len(X), rng)
        h1, h2 = fit(Z, 1), fit(Z, 2)
        null.append(2 * (h2.score(Z) - h1.score(Z)) * len(Z))
    p = float((1 + np.sum(np.array(null) >= lr)) / (n_boot + 1))
    return {"n": int(len(X)), "bic_1": float(g1.bic(X)), "bic_2": float(g2.bic(X)), "lr": float(lr),
            "p_bootstrap": p, "weights_2": [round(float(w), 2) for w in g2.weights_],
            "means_2": [[round(float(v), 2) for v in m] for m in g2.means_],
            "labels": g2.predict(X).tolist()}


def icc_log_area(det: pd.DataFrame, s: DiscSettings) -> dict:
    """One-way ICC of log area over strong detections of tracks with >=3 detections."""
    strong = det[~det["recovered"]]
    groups = [np.log(v.to_numpy()) for _, v in strong.groupby("track_id")["area_px"] if len(v) >= 3]
    if len(groups) < s.icc_min_tracks:
        return {"icc": None, "n_tracks": len(groups)}
    k = np.mean([len(x) for x in groups])
    allv = np.concatenate(groups)
    gm = allv.mean()
    n = len(groups)
    msb = sum(len(x) * (x.mean() - gm) ** 2 for x in groups) / (n - 1)
    msw = sum(((x - x.mean()) ** 2).sum() for x in groups) / (len(allv) - n)
    icc = (msb - msw) / (msb + (k - 1) * msw)
    return {"icc": float(icc), "n_tracks": n, "between_sd_log": float(np.sqrt(max((msb - msw) / k, 0))),
            "within_sd_log": float(np.sqrt(msw))}


def associations(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in [("log_area", "max_dist_from_median_px"), ("median_relative_intensity", "max_dist_from_median_px"),
                 ("log_area", "lifespan_frames"), ("median_relative_intensity", "lifespan_frames"),
                 ("log_area", "median_relative_intensity")]:
        ok = d[[a, b]].dropna()
        if len(ok) >= 8:
            r = stats.spearmanr(ok[a], ok[b])
            rows.append({"x": a, "y": b, "n": len(ok), "spearman_rho": round(float(r.statistic), 2),
                         "p_value": round(float(r.pvalue), 4)})
    return pd.DataFrame(rows)


def group_summary(d: pd.DataFrame, labels: list[int] | None) -> pd.DataFrame:
    d = d.copy()
    if labels is None:
        d["population"] = "A"
    else:
        order = np.argsort(d.groupby(np.array(labels))["median_area_px"].median().to_numpy())
        names = {int(o): "AB"[i] for i, o in enumerate(order)}
        d["population"] = [names[int(l)] for l in labels]
    out = d.groupby("population").agg(n_tracks=("track_id", "size"), median_area_px=("median_area_px", "median"),
                                      median_relative_intensity=("median_relative_intensity", "median"),
                                      median_lifespan_frames=("lifespan_frames", "median"),
                                      n_stationary=("motion_class", lambda v: int((v == "stationary").sum())),
                                      n_moved=("motion_class", lambda v: int((v == "moved").sum()))).reset_index()
    return out.round(2), d


def summary_text(mix: dict, icc: dict, assoc: pd.DataFrame, groups: pd.DataFrame, n_tracks: int) -> str:
    L = ["## Heading 2: dendrimer vs enzyme discrimination (English summary)", ""]
    L.append(f"- Number of tracks in the analysis: {n_tracks} (tracks with at least 3 strong detections).")
    if "p_bootstrap" in mix:
        pv = mix["p_bootstrap"]
        two = pv < 0.05
        verdict = ("two populations explain the data significantly better than one" if two else
                   "BORDERLINE: evidence for two populations is weak, no firm statement (neither one population nor two can be excluded)"
                   if pv < 0.10 else "no justification for two populations, one population is enough")
        L.append(f"- Do size and brightness split into two separate populations? Bootstrap likelihood-ratio test p = {pv:.3f} "
                 f"(with different random seeds {mix.get('p_range', [pv, pv])[0]}–{mix.get('p_range', [pv, pv])[1]}) → {verdict}.")
        if two:
            L.append("  Populations (A small, B large): " + "; ".join(
                f"{r.population}: {int(r.n_tracks)} tracks, median area {r.median_area_px:.0f} px, brightness {r.median_relative_intensity:.2f}×"
                for r in groups.itertuples()) + ".")
    if icc.get("icc") is not None:
        L.append(f"- Is each feature consistent within itself? Intraclass correlation (ICC) = {icc['icc']:.2f} "
                 f"({icc['n_tracks']} tracks; between-track SD {icc['between_sd_log']:.2f}, within-track SD {icc['within_sd_log']:.2f}, log area). "
                 "Close to 1 means each feature has its own fixed size.")
    ai = assoc[(assoc.x == "log_area") & (assoc.y == "median_relative_intensity")]
    if len(ai) and ai.iloc[0].spearman_rho > 0.8:
        L.append(f"- Caution: area and brightness measure almost the same thing (ρ = {ai.iloc[0].spearman_rho}); "
                 "thresholding makes a large feature look both wide and bright. The two populations are therefore effectively a single axis (size).")
    L.append("- Tracks should not be treated as independent particles: the same physical particle can be a separate track in different frame ranges "
             "and n is small; the p value may therefore be optimistic.")
    for r in assoc.itertuples():
        if r.p_value < 0.05 and not (r.x == "log_area" and r.y == "median_relative_intensity"):
            L.append(f"- Significant association: Spearman ρ = {r.spearman_rho} between {r.x} and {r.y} (n={r.n}, p={r.p_value}).")
    L += ["- This split is only descriptive: population A/B does not say which is the enzyme and which is the dendrimer. "
          "G4 PAMAM (~14 kDa) and DNase I (~31 kDa) typically look like blobs of similar size in AFM; "
          "the apparent width is set by tip shape and JPG brightness, and there is no real height.",
          "- Needed to identify them: raw AFM files (height in nm), control images with enzyme only and polymer only, "
          "or a labelled experiment (for example with the enzyme tagged)."]
    return "\n".join(L) + "\n"


def figure(d: pd.DataFrame, det: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axs[0]
    pal = {"stationary": "#2255c4", "moved": "#d9534f", "short_track_uncertain": "#888888", "wandering_uncertain": "#e0a030"}
    for m, g in d.groupby("motion_class"):
        ax.scatter(g["median_area_px"], g["median_relative_intensity"], s=30 + 6 * g["n_strong"], alpha=0.8,
                   color=pal.get(m, "#444"), label=m, edgecolor="white", lw=0.5)
    if "population" in d:
        for p_, g in d.groupby("population"):
            ax.annotate(p_, (g["median_area_px"].median(), g["median_relative_intensity"].median()), fontsize=12, weight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("median feature area (px, log scale)")
    ax.set_ylabel("brightness relative to DNA ridge")
    ax.set_title("One point per tracked feature", fontsize=9, loc="left")
    ax.legend(fontsize=7, title="motion class", title_fontsize=7)
    ax = axs[1]
    strong = det[~det["recovered"]]
    keep = d.sort_values("n_strong", ascending=False).head(14)["track_id"]
    for t in keep:
        g = strong[strong.track_id == t].sort_values("rank")
        ax.plot(g["rank"], g["area_px"], marker="o", ms=2.5, lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("frame index (provisional order)")
    ax.set_ylabel("area (px, log scale)")
    ax.set_title("Area of the 14 best-observed features over time", fontsize=9, loc="left")
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.25, lw=0.5)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_feature_populations.{ext}", dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def run(outputs: Path, s: DiscSettings | None = None) -> dict:
    s = s or DiscSettings()
    p6b = outputs / "phase6b"
    for f in (p6b / "feature_detections_v2.csv", p6b / "feature_tracks_v2.csv"):
        if not f.is_file():
            raise FileNotFoundError(f"{f} not found. Run phase 6b first (run_all.sh).")
    det = pd.read_csv(p6b / "feature_detections_v2.csv")
    trk = pd.read_csv(p6b / "feature_tracks_v2.csv")
    d = track_descriptors(det, trk, s)
    X = d[["log_area", "median_relative_intensity"]].to_numpy()
    runs = [mixture_test(X, s.n_boot, s.seed + i) for i in range(3)]
    mix = dict(runs[0])
    if "p_bootstrap" in mix:
        ps = [r["p_bootstrap"] for r in runs]
        mix["p_bootstrap"], mix["p_range"] = float(np.median(ps)), [round(min(ps), 3), round(max(ps), 3)]
    use2 = "p_bootstrap" in mix and mix["p_bootstrap"] < s.alpha
    groups, d = group_summary(d, mix["labels"] if use2 else None)
    icc = icc_log_area(det, s)
    assoc = associations(d)
    out = outputs / "phase6c"
    out.mkdir(parents=True, exist_ok=True)
    d.round(3).to_csv(out / "feature_descriptors.csv", index=False)
    groups.to_csv(out / "population_summary.csv", index=False)
    assoc.to_csv(out / "associations.csv", index=False)
    (out / "summary.md").write_text(summary_text(mix, icc, assoc, groups, len(d)), encoding="utf-8")
    (out / "parameters.json").write_text(json.dumps({
        "settings": asdict(s), "mixture": {k: v for k, v in mix.items() if k != "labels"}, "icc": icc,
        "notes": "descriptive grouping only; strong detections only; no identification as enzyme or dendrimer"},
        indent=2), encoding="utf-8")
    figure(d, det, out)
    return {"descriptors": d, "mixture": mix, "icc": icc, "assoc": assoc, "groups": groups}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Heading 2: dendrimer vs enzyme, descriptive populations.")
    p.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    a = p.parse_args(argv)
    try:
        r = run(a.outputs)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print({k: v for k, v in r["mixture"].items() if k != "labels"})
    print(r["icc"])
    print(r["groups"].to_string(index=False))
    print(r["assoc"].to_string(index=False))
    print(f"Results: {a.outputs / 'phase6c'}")
    print("Descriptive grouping only: nothing here identifies a feature as enzyme or dendrimer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
