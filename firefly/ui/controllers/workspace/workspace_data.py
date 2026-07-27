"""Qt-free data + statistics backend for the merged Analysis workspace.

The Analysis tab (formerly the separate *Compare* and *Results* tabs) is a live
cross-condition comparison surface: the user drops 2–6 *conditions*, each a set
of FIREFLY analysis-output run folders, and the figure / headline metrics /
group statistics / pairwise significance recompute instantly as folders and
settings change.

This module is the engine underneath that — pure Python (no Qt, no matplotlib),
so it is fully unit-testable:

* :func:`load_run` reads one analysis-output run folder into a :class:`RunData`
  (per-folder *replicate* scalars from ``*_summary_metrics.json`` + lazy
  per-track distributions from ``*_diffusion_summary.csv`` and friends).
* :data:`METRICS` is the comparison-metric registry; each metric knows how to
  pull a per-folder scalar (the replicate value used for the bars + the
  statistical test) and, where available, a per-track distribution (used only
  for the violin / ECDF *picture*).
* :func:`pairwise_stats` runs the real cross-condition statistics — scipy tests,
  Cliff's δ effect size, and multiple-comparison correction.

**Statistical unit.** Following the real FIREFLY comparison engine (and avoiding
pseudoreplication), the unit of analysis is the *run folder*: tests compare the
arrays of per-folder scalar values (n = active folders per condition), not the
pooled per-track values.  Pooled per-track distributions exist only to draw a
richer figure.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Callable, Optional

import numpy as np
import pandas as pd

# ── canonical motion-class palette (dark theme — never recolour) ──────────
# Mirrors firefly.analysis.fa_constants; hard-coded so this stays importable
# without dragging in the plotting stack.
MOTION_CLASSES = ["Immobile", "Confined", "Brownian", "Directed", "Unknown"]
MOTION_COLORS = {
    "Immobile": "#e05252", "Confined": "#f5a623", "Brownian": "#4a90d9",
    "Directed": "#7ed321", "Unknown": "#aaaaaa",
}

# condition swatch palette (matches the design tokens).  12 distinct hues so a
# group × time-point design (one condition per cell — e.g. 3 groups × 3 time
# points = 9) gets its own swatch up to the cap before the palette repeats.
GROUP_COLORS = ["#58a6ff", "#f78166", "#56d364", "#27c0e8", "#f6a623", "#a371f7",
                "#e05252", "#ec6cb9", "#d2a8ff", "#39c5cf", "#e3b341", "#7ee787"]
# Raised 6 → 12 (matches the legacy Compare cap): conditions live in a scrollable
# rail, so this is just a runaway guard, not a layout limit.
MAX_CONDITIONS = 12


# ── one loaded run folder ─────────────────────────────────────────────────
class RunData:
    """A single analysis-output run folder.

    Holds the cheap per-run summary (read eagerly) and lazily reads the larger
    per-track CSVs only when a distribution is actually requested.  Every
    accessor degrades to ``None`` / empty rather than raising, so a malformed or
    partial run never breaks the live recompute.
    """

    def __init__(self, folder: str, stem: str, extras_dir: str, summary: dict):
        self.folder = folder
        self.stem = stem
        self.extras_dir = extras_dir
        self.summary = summary or {}
        self.n_tracks = int(self.summary.get("n_tracks") or 0)
        self.n_locs = int(self.summary.get("n_locs") or 0)
        fi = self.summary.get("fi_s")
        self.fi_s = float(fi) if fi else None
        self._cache: dict = {}

    # -- identity ----------------------------------------------------------
    @property
    def id(self) -> str:
        return self.stem

    @property
    def n_label(self) -> str:
        """Compact track count, e.g. ``4.2k`` / ``980``."""
        n = self.n_tracks
        return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

    @property
    def qc_level(self) -> str:
        """``ok`` / ``warn`` / ``error`` for the folder chip's QC dot.

        ``error`` = no usable tracks (an effectively empty run); ``warn`` = the
        pipeline raised at least one QC warning flag; otherwise ``ok``.
        """
        if self.n_tracks <= 0:
            return "error"
        flags = (self.summary.get("qc") or {}).get("flags") or []
        for f in flags:
            if isinstance(f, dict) and f.get("level") == "warn":
                return "warn"
        return "ok"

    @property
    def qc_messages(self) -> list[str]:
        flags = (self.summary.get("qc") or {}).get("flags") or []
        return [f.get("msg", "") for f in flags if isinstance(f, dict)]

    # -- lazy CSV access ---------------------------------------------------
    def _read_csv(self, suffix: str) -> Optional[pd.DataFrame]:
        key = ("csv", suffix)
        if key in self._cache:
            return self._cache[key]
        path = os.path.join(self.extras_dir, f"{self.stem}{suffix}")
        df = None
        if os.path.isfile(path):
            try:
                df = pd.read_csv(path)
            except Exception:
                df = None
        self._cache[key] = df
        return df

    def _read_json(self, suffix: str) -> Optional[dict]:
        key = ("json", suffix)
        if key in self._cache:
            return self._cache[key]
        path = os.path.join(self.extras_dir, f"{self.stem}{suffix}")
        data = None
        if os.path.isfile(path):
            try:
                with open(path) as fh:
                    data = json.load(fh)
            except Exception:
                data = None
        self._cache[key] = data
        return data

    def diff(self) -> Optional[pd.DataFrame]:
        return self._read_csv("_diffusion_summary.csv")

    def _diff_col(self, col: str, *, positive: bool = False) -> Optional[np.ndarray]:
        df = self.diff()
        if df is None or col not in df.columns:
            return None
        arr = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if positive:
            arr = arr[arr > 0]
        return arr if arr.size else None


# ── metric registry ───────────────────────────────────────────────────────
class Metric:
    """One comparison metric: how to read a per-folder scalar and (optionally)
    a per-track distribution, plus presentation hints."""

    def __init__(self, id, label, unit, dp, axis, cat, *,
                 scalar: Callable[[RunData], Optional[float]],
                 dist: Optional[Callable[[RunData], Optional[np.ndarray]]] = None,
                 approx: bool = False, log_default: bool = False):
        self.id = id
        self.label = label
        self.unit = unit
        self.dp = dp
        self.axis = axis
        self.cat = cat
        self._scalar = scalar
        self._dist = dist
        self.approx = approx          # derived/approximate source → badge it
        self.log_default = log_default

    def scalar(self, run: RunData) -> Optional[float]:
        try:
            v = self._scalar(run)
        except Exception:
            return None
        if v is None:
            return None
        v = float(v)
        return v if np.isfinite(v) else None

    def dist(self, run: RunData) -> Optional[np.ndarray]:
        if self._dist is None:
            return None
        try:
            arr = self._dist(run)
        except Exception:
            return None
        if arr is None:
            return None
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        return arr if arr.size else None

    def fmt(self, v: Optional[float]) -> str:
        if v is None or not np.isfinite(v):
            return "—"
        if self.dp == 0:
            return f"{round(v):,}"
        return f"{v:.{self.dp}f}"


def _summary(run: RunData, key: str) -> Optional[float]:
    v = run.summary.get(key)
    return float(v) if v is not None else None


def _mobile_pct(run: RunData) -> Optional[float]:
    v = run.summary.get("mobile_fraction")
    return float(v) * 100.0 if v is not None else None


def _motion_frac(run: RunData, cls: str) -> Optional[float]:
    counts = run.summary.get("motion_counts") or {}
    total = sum(int(c) for c in counts.values()) if counts else 0
    if not total:
        return None
    return 100.0 * int(counts.get(cls, 0)) / total


def _track_len_median(run: RunData) -> Optional[float]:
    qc = run.summary.get("qc") or {}
    v = qc.get("median_track_length")
    return float(v) if v is not None else None


def _track_len_dist(run: RunData) -> Optional[np.ndarray]:
    tr = run._read_csv("_trajectories.csv")
    if tr is None or "particle" not in tr.columns:
        return None
    lens = tr.groupby("particle").size().to_numpy(dtype=float)
    return lens if lens.size else None


def _msd_at_1s(run: RunData) -> Optional[float]:
    emsd = run._read_csv("_ensemble_msd.csv")
    if emsd is None or {"lag_frame", "msd_um2"} - set(emsd.columns):
        return None
    fi = run.fi_s
    if not fi:
        return None
    lag_s = emsd["lag_frame"].to_numpy(dtype=float) * fi
    msd = emsd["msd_um2"].to_numpy(dtype=float)
    ok = np.isfinite(lag_s) & np.isfinite(msd)
    if ok.sum() < 2:
        return None
    return float(np.interp(1.0, lag_s[ok], msd[ok]))


def _speed_measured_scalar(run: RunData) -> Optional[float]:
    """Measured step speed = per-track mean step distance ÷ Δt (median over
    tracks).  The straight-up geometric speed, no diffusion-model assumption."""
    v = run._diff_col("mean_step_um", positive=True)
    fi = run.fi_s
    if v is None or not len(v) or not fi:
        return None
    return float(np.median(v) / float(fi))


def _speed_measured_dist(run: RunData) -> Optional[np.ndarray]:
    v = run._diff_col("mean_step_um", positive=True)
    fi = run.fi_s
    if v is None or not len(v) or not fi:
        return None
    return v / float(fi)


def _col_median(run: RunData, col: str) -> Optional[float]:
    """Median of one per-track diffusion_summary column (for a scalar readout)."""
    v = run._diff_col(col, positive=True)
    return float(np.median(v)) if v is not None and len(v) else None


def _fluor_dist(run: RunData) -> Optional[np.ndarray]:
    """Per-TRACK fluorescence intensity: the mean of a track's localisation
    intensities (``mass``), i.e. Σ intensities in the track ÷ its number of
    localisations — one value per trajectory.  It's post-preprocessing integrated
    intensity that scales with the detection threshold, so it's only meaningful
    COMPARED across dishes analysed with identical settings."""
    tr = run._read_csv("_trajectories.csv")
    cols = getattr(tr, "columns", [])
    if tr is None or "mass" not in cols or "particle" not in cols:
        return None
    m = pd.to_numeric(tr["mass"], errors="coerce")
    per_track = tr.assign(_m=m).groupby("particle")["_m"].mean().to_numpy(float)
    per_track = per_track[np.isfinite(per_track) & (per_track > 0)]
    return per_track if len(per_track) else None


def _fluor_scalar(run: RunData) -> Optional[float]:
    v = _fluor_dist(run)
    return float(np.median(v)) if v is not None else None


def _duration_dist(run: RunData) -> Optional[np.ndarray]:
    """Per-track duration (s) = (localisations − 1) × frame interval."""
    lens = _track_len_dist(run)
    fi = run.fi_s
    if lens is None or not fi:
        return None
    dur = (lens - 1.0) * float(fi)
    return dur if dur.size else None


def _duration_scalar(run: RunData) -> Optional[float]:
    v = _duration_dist(run)
    return float(np.median(v)) if v is not None and len(v) else None


def _nlocs_scalar(run: RunData) -> Optional[float]:
    """Number of localisations contained in all qualifying trajectories (the
    trajectory table's row count)."""
    tr = run._read_csv("_trajectories.csv")
    return float(len(tr)) if tr is not None else None


def _angle_cos(run: RunData) -> Optional[np.ndarray]:
    """Per-step |cos θ| from the saved turning angles (column ``turning_angle_deg``
    — the analysis writes raw per-step angles in degrees, fa_palmtracer.py)."""
    ta = run._read_csv("_turning_angles.csv")
    if ta is None or "turning_angle_deg" not in ta.columns:
        return None
    deg = pd.to_numeric(ta["turning_angle_deg"], errors="coerce").to_numpy(float)
    deg = deg[np.isfinite(deg)]
    if not deg.size:
        return None
    return np.abs(np.cos(np.radians(deg)))


def _angle_scalar(run: RunData) -> Optional[float]:
    c = _angle_cos(run)
    return float(c.mean()) if c is not None and c.size else None


def _angle_dist(run: RunData) -> Optional[np.ndarray]:
    return _angle_cos(run)


def _tracks_for_vacf(run: RunData):
    """The per-track table compute_vacf / compute_van_hove need (particle, frame,
    x, y).  fa_compare computes VACF/α₂ per replicate from these raw tracks with
    px=1/dt=1 (dimensionless ratios), NOT from a saved scalar — so the headline
    has to do the same to match the figure."""
    trk = run._read_csv("_trajectories.csv")
    if trk is None or not {"particle", "frame", "x", "y"} <= set(trk.columns):
        return None
    return trk


def _nongauss_a2(run: RunData) -> Optional[float]:
    """Van-Hove non-Gaussian α₂ per replicate (matches fa_compare's per-folder
    scalar).  Memoised — the trajectory pass is non-trivial."""
    if ("a2",) in run._cache:
        return run._cache[("a2",)]
    val = None
    trk = _tracks_for_vacf(run)
    if trk is not None:
        try:
            from firefly.analysis.fa_diffusion import compute_van_hove
            vh = compute_van_hove(trk, 1.0)
            if vh and np.isfinite(vh.get("non_gaussian_alpha2", np.nan)):
                val = float(vh["non_gaussian_alpha2"])
        except Exception:
            val = None
    if val is None:                       # fall back to a saved summary key
        val = _summary(run, "nongauss_alpha2")
    run._cache[("a2",)] = val
    return val


def _vacf_persistence(run: RunData) -> Optional[float]:
    """VACF lag-1 directional persistence per replicate (matches fa_compare's
    per-folder scalar — the dots in the Persistence panel).  Memoised."""
    if ("vacf",) in run._cache:
        return run._cache[("vacf",)]
    val = None
    trk = _tracks_for_vacf(run)
    if trk is not None:
        try:
            from firefly.analysis.fa_diffusion import compute_vacf
            vc = compute_vacf(trk, 1.0, 1.0)
            if vc and np.isfinite(vc.get("persistence", np.nan)):
                val = float(vc["persistence"])
        except Exception:
            val = None
    if val is None:                       # fall back to a saved summary key
        val = _summary(run, "vacf_persistence")
    run._cache[("vacf",)] = val
    return val


def _dwell_dist(run: RunData) -> Optional[np.ndarray]:
    dw = run._read_csv("_dwell_times.csv")
    if dw is None:
        return None
    col = next((c for c in ("tau_s", "dwell_s", "tau") if c in dw.columns), None)
    if col is None:
        return None
    arr = pd.to_numeric(dw[col], errors="coerce").to_numpy(float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    return arr if arr.size else None


def _conf_scalar(run: RunData) -> Optional[float]:
    arr = run._diff_col("radius_of_gyration_um", positive=True)
    return float(np.median(arr)) * 1000.0 if arr is not None else None


def _conf_dist(run: RunData) -> Optional[np.ndarray]:
    arr = run._diff_col("radius_of_gyration_um", positive=True)
    return arr * 1000.0 if arr is not None else None


def _track_count(run: RunData) -> Optional[float]:
    """Number of tracks in the run — the per-folder scalar behind the Track-count
    comparison (one dot per folder)."""
    return float(run.n_tracks) if run.n_tracks else None


def _msd_auc(run: RunData) -> Optional[float]:
    """Area under the ensemble-MSD curve (µm²·s) — trapezoidal over lag-time."""
    emsd = run._read_csv("_ensemble_msd.csv")
    if emsd is None or {"lag_frame", "msd_um2"} - set(emsd.columns):
        return None
    fi = run.fi_s
    if not fi:
        return None
    lag_s = emsd["lag_frame"].to_numpy(dtype=float) * fi
    msd = emsd["msd_um2"].to_numpy(dtype=float)
    ok = np.isfinite(lag_s) & np.isfinite(msd)
    if ok.sum() < 2:
        return None
    _trap = getattr(np, "trapezoid", None) or np.trapz   # np.trapz removed in numpy 2.0
    return float(_trap(msd[ok], lag_s[ok]))


def _jdd_median_jump(run: RunData) -> Optional[float]:
    """Median single-frame jump distance (µm) from the saved JDD fit."""
    j = run._read_json("_jdd.json")
    if not isinstance(j, dict):
        return None
    jumps = j.get("jumps")
    if jumps is None:
        return None
    arr = np.asarray(jumps, dtype=float)
    arr = arr[np.isfinite(arr) & (arr >= 0)]
    return float(np.median(arr)) if arr.size else None


# ── 17-panel per-condition publication report ─────────────────────────────
# Each condition's "All panels" view pools its active run folders into one set
# of report panels, grouped by category.  kind:
#   'metric' — pooled per-track distribution of METRIC_BY_ID[ref]
#   'col'    — pooled per-track distribution of a diffusion_summary column
#   'motion' — stacked motion-class fractions
#   'msd'    — ensemble MSD curves (one per folder)
#   'raster' — representative saved image (art = figure-folder filename hint);
#              degrades to a placeholder if the run didn't export it.
PANEL_CATS = ["Imaging", "Tracking", "Diffusion", "Population"]
# The All-panels view mirrors the single-run analysis figure EXACTLY — the 17
# fa_figure.make_figure panels (A–Q) and nothing else.  Two render routes:
#   `letter`       → the make_figure panel, group-AVERAGED (pooled over the
#                    condition's folders), exact matplotlib look.  (kind "mfig")
#   raster + panel_letter → a SPATIAL map tied to one field of view; can't be
#                    averaged → shown per-replicate from the saved per-panel PNG
#                    (`art` is the fallback artifact when per-panel PNGs weren't
#                    exported).
PANELS = [
    {"name": "Max projection",                "cat": "Imaging",    "kind": "raster", "panel_letter": "A", "art": "_sptpalm_figure"},
    {"name": "Trajectories",                  "cat": "Tracking",   "kind": "raster", "panel_letter": "B", "art": "_sptpalm_figure"},
    {"name": "Trajectories by D",             "cat": "Tracking",   "kind": "raster", "panel_letter": "C", "art": "_sptpalm_figure"},
    {"name": "MSD curves",                    "cat": "Diffusion",  "kind": "mfig", "letter": "D"},
    {"name": "Diffusion coefficient distribution", "cat": "Diffusion", "kind": "mfig", "letter": "E"},
    {"name": "Motion classification",         "cat": "Population",  "kind": "mfig", "letter": "F"},
    {"name": "Anomalous exponent α",          "cat": "Diffusion",  "kind": "mfig", "letter": "G"},
    {"name": "Position density map",          "cat": "Imaging",    "kind": "raster", "panel_letter": "H", "art": "_superres"},
    {"name": "Turning-angle distribution",    "cat": "Tracking",   "kind": "mfig", "letter": "I"},
    {"name": "Mobile fraction over time",     "cat": "Population",  "kind": "mfig", "letter": "J"},
    {"name": "Jump-distance distribution",    "cat": "Diffusion",  "kind": "mfig", "letter": "K"},
    {"name": "Cluster map",                   "cat": "Imaging",    "kind": "raster", "panel_letter": "L", "art": "_superres"},
    {"name": "Dwell-time distribution",       "cat": "Population",  "kind": "mfig", "letter": "M"},
    {"name": "Moment-scaling spectrum",       "cat": "Diffusion",  "kind": "mfig", "letter": "N"},
    {"name": "Radial distribution",           "cat": "Tracking",   "kind": "mfig", "letter": "O"},
    {"name": "van Hove displacements",        "cat": "Population",  "kind": "mfig", "letter": "P"},
    {"name": "Velocity autocorrelation",      "cat": "Population",  "kind": "mfig", "letter": "Q"},
]

# gallery index → fa_figure letter (derived from the `letter` fields above).
MFIG_LETTER = {i: p["letter"] for i, p in enumerate(PANELS) if p.get("letter")}


def pooled_column(runs, col, *, positive=False):
    """Concatenate one diffusion_summary column across runs (active folders)."""
    chunks = [r._diff_col(col, positive=positive) for r in runs]
    chunks = [c for c in chunks if c is not None and len(c)]
    return np.concatenate(chunks) if chunks else None


def find_artifact(run, hint):
    """Locate a representative saved image in a run folder's figures/ dir whose
    name contains *hint* (e.g. '_superres', '_sptpalm_figure'); else None."""
    for d in (os.path.join(run.folder, "figures"),
              os.path.join(run.folder, "figures", "panels"),
              run.folder):
        if not os.path.isdir(d):
            continue
        hits = sorted(glob.glob(os.path.join(d, f"*{hint}*.png")))
        if hits:
            return hits[0]
    return None


METRICS: list[Metric] = [
    Metric("D", "Diffusion D", "µm²/s", 3, "log₁₀ D (µm²/s)", "Diffusion",
           scalar=lambda r: _summary(r, "median_d"),
           dist=lambda r: r._diff_col("D", positive=True), log_default=True),
    Metric("a", "Anomalous α", "", 2, "anomalous exponent α", "Diffusion",
           scalar=lambda r: _summary(r, "median_alpha"),
           dist=lambda r: r._diff_col("alpha")),
    Metric("mob", "Mobile fraction", "%", 0, "mobile fraction (%)", "Population",
           scalar=_mobile_pct),
    Metric("motion", "Motion classes", "%", 0, "Brownian fraction (%)", "Population",
           scalar=lambda r: _motion_frac(r, "Brownian")),
    Metric("len", "Track length", "frames", 0, "track length (frames)", "Tracking",
           scalar=_track_len_median, dist=_track_len_dist),
    Metric("msd", "MSD @1s", "µm²", 3, "MSD @1s (µm²)", "Diffusion",
           scalar=_msd_at_1s),
    Metric("step", "Step distance", "µm", 3, "step distance (µm)", "Tracking",
           scalar=lambda r: _col_median(r, "mean_step_um"),
           dist=lambda r: r._diff_col("mean_step_um", positive=True)),
    Metric("speed", "Step speed", "µm/s", 3, "step speed (µm/s)", "Tracking",
           scalar=_speed_measured_scalar, dist=_speed_measured_dist),
    Metric("rg", "Radius of gyration", "µm", 3, "R_g (µm)", "Tracking",
           scalar=lambda r: _col_median(r, "radius_of_gyration_um"),
           dist=lambda r: r._diff_col("radius_of_gyration_um", positive=True)),
    Metric("netdisp", "Net displacement", "µm", 3, "net displacement (µm)", "Tracking",
           scalar=lambda r: _col_median(r, "net_displacement_um"),
           dist=lambda r: r._diff_col("net_displacement_um", positive=True)),
    Metric("path", "Path length", "µm", 3, "path length (µm)", "Tracking",
           scalar=lambda r: _col_median(r, "path_length_um"),
           dist=lambda r: r._diff_col("path_length_um", positive=True)),
    Metric("dir", "Directionality ratio", "", 3, "net ÷ path", "Tracking",
           scalar=lambda r: _col_median(r, "directionality_ratio"),
           dist=lambda r: r._diff_col("directionality_ratio")),
    Metric("dur", "Track duration", "s", 3, "track duration (s)", "Tracking",
           scalar=_duration_scalar, dist=_duration_dist),
    Metric("nlocs", "Localisations", "", 0, "localisations (n)", "Tracking",
           scalar=_nlocs_scalar),
    Metric("fluor", "Fluorescence intensity", "a.u.", 0, "fluorescence (a.u.)", "Imaging",
           scalar=_fluor_scalar, dist=_fluor_dist, approx=True),
    Metric("angle", "Turning angle", "", 2, "mean |cos θ|", "Tracking",
           scalar=_angle_scalar, dist=_angle_dist),
    Metric("a2", "Non-Gaussian α₂", "", 2, "α₂", "Diffusion",
           scalar=_nongauss_a2),
    Metric("conf", "Confinement R", "nm", 0, "R_conf (nm)", "Population",
           scalar=_conf_scalar, dist=_conf_dist),
    Metric("dwell", "Dwell time", "s", 2, "dwell time (s)", "Population",
           scalar=lambda r: _summary(r, "dwell_tau_s"), dist=_dwell_dist),
    # scalars that back comparison panels which previously had no stats cards
    Metric("count", "Track count", "", 0, "tracks (n)", "Tracking",
           scalar=_track_count),
    Metric("auc", "MSD AUC", "µm²·s", 3, "MSD AUC (µm²·s)", "Diffusion",
           scalar=_msd_auc),
    Metric("jdd", "Jump distance", "µm", 3, "median jump (µm)", "Diffusion",
           scalar=_jdd_median_jump),
    # the radial distribution summarises the same turning-angle directionality
    Metric("radial", "Radial persistence", "", 2, "mean |cos θ|", "Tracking",
           scalar=_angle_scalar, dist=_angle_dist),
    # VACF lag-1 directional persistence (computed from tracks like fa_compare)
    Metric("vacf", "VACF persistence", "", 3, "VACF persistence (lag 1)", "Population",
           scalar=_vacf_persistence),
]
METRIC_BY_ID = {m.id: m for m in METRICS}


# ── run-folder loading ─────────────────────────────────────────────────────
def _resolve_extras(folder: str) -> Optional[tuple[str, str]]:
    """Find the (extras_dir, stem) for a run folder.

    A FIREFLY run writes its machine-readable sidecars under
    ``<run>/firefly_extras/`` (older layouts used ``<run>/data/``).  We look in
    the folder itself and those two subdirs for a ``*_summary_metrics.json`` or,
    failing that, a ``*_diffusion_summary.csv`` to derive the stem.
    """
    candidates = [folder,
                  os.path.join(folder, "firefly_extras"),
                  os.path.join(folder, "data")]
    for d in candidates:
        if not os.path.isdir(d):
            continue
        hits = sorted(glob.glob(os.path.join(d, "*_summary_metrics.json")))
        if hits:
            stem = os.path.basename(hits[0])[:-len("_summary_metrics.json")]
            return d, stem
    for d in candidates:
        if not os.path.isdir(d):
            continue
        hits = sorted(glob.glob(os.path.join(d, "*_diffusion_summary.csv")))
        if hits:
            stem = os.path.basename(hits[0])[:-len("_diffusion_summary.csv")]
            return d, stem
    return None


def load_run(folder: str) -> Optional[RunData]:
    """Load one analysis-output run folder, or ``None`` if it isn't one."""
    resolved = _resolve_extras(folder)
    if resolved is None:
        return None
    extras_dir, stem = resolved
    summary: dict = {}
    sm_path = os.path.join(extras_dir, f"{stem}_summary_metrics.json")
    if os.path.isfile(sm_path):
        try:
            with open(sm_path) as fh:
                summary = json.load(fh)
        except Exception:
            summary = {}
    run = RunData(folder, stem, extras_dir, summary)
    # If there were no summary metrics, fall back to counting tracks from the
    # per-track table so the chip still shows a sensible n / QC.
    if not summary:
        df = run.diff()
        run.n_tracks = int(len(df)) if df is not None else 0
    return run


def is_run_folder(folder: str) -> bool:
    return _resolve_extras(folder) is not None


# ── external localisation files → on-the-fly replicates ─────────────────────
# The Analysis tab compares FIREFLY run folders.  A raw localisation table from
# another tool (palmTRACER / ThunderSTORM / Picasso / TrackMate) carries no
# diffusion metrics yet, so to be pooled as a replicate it must be ANALYSED
# first.  We run the real pipeline (`_run_one_analysis`) on it — using the
# caller's current sidebar settings, so its D/α are computed the SAME way as the
# FIREFLY replicates it sits beside — into a cached run folder that then loads
# as an ordinary RunData.  Analyse-once: a byte-identical file + settings reuse
# the cache instead of re-running.
_EXTERNAL_LOC_EXTS = (".csv", ".txt", ".tsv")


def is_external_loc_file(path: str) -> bool:
    """True when *path* is a localisation-table FILE (an external tool's export)
    rather than a FIREFLY run folder — i.e. something we must analyse before it
    can be a replicate."""
    return (bool(path) and os.path.isfile(path)
            and str(path).lower().endswith(_EXTERNAL_LOC_EXTS))


class _ExternalImportStub:
    """The handful of ImportController attributes `build_params` reads, for a
    bare file.  Calibration is left to the file's embedded metadata (palmTRACER
    encodes pixel size / frame interval; others fall back to FIREFLY defaults)."""
    def __init__(self, path: str):
        self.filePath = path
        self.outDir = None
        self.isCsv = True
        self.overridePx = False
        self.pixelSize = 0.0
        self.overrideFi = False
        self.frameInterval = 0.0


def _external_run_dir(path: str, cache_root: str, sig: str) -> str:
    """Deterministic cache folder for analysing *path* under *cache_root*, keyed
    by the file's identity + mtime + a signature of the analysis settings, so an
    unchanged file+settings reuse the cached run and a changed one re-runs."""
    import hashlib
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    key = hashlib.blake2b(
        f"{os.path.abspath(path)}|{mtime}|{sig}".encode("utf-8"),
        digest_size=10).hexdigest()
    stem = os.path.splitext(os.path.basename(path))[0]
    return os.path.join(cache_root, f"{stem}__{key}")


def analyse_external_file(path: str, settings, *, cache_root: str,
                          log: "Callable[[str], None] | None" = None,
                          cancel=None) -> Optional[str]:
    """Analyse an external localisation file into a FIREFLY run folder and return
    its path (or ``None`` on failure / no tracks).

    Reuses a cached analysis when the file + settings are unchanged.  Runs the
    real ``_run_one_analysis`` in-thread (the caller does this off the GUI
    thread), writing outputs flat into the cache folder (``wrap_in_stem_folder``
    off) so the returned path loads directly with :func:`load_run`.
    """
    import queue as _queue
    import threading as _threading
    _log = log or (lambda _m: None)
    try:
        from firefly.ui.controllers.params.params_builder import build_params
        p = build_params(settings, _ExternalImportStub(path), fpath=path,
                         out_dir=None)
    except Exception as exc:
        _log(f"  Couldn't build analysis parameters for "
             f"{os.path.basename(path)}: {exc}")
        return None

    # Signature = the params that affect the NUMBERS (exclude per-run paths), so
    # the cache key changes iff the analysis would produce different results.
    _volatile = {"file", "out_dir", "stem_override", "widget_state",
                 "wrap_in_stem_folder"}
    try:
        sig = json.dumps({k: v for k, v in sorted(p.items())
                          if k not in _volatile}, default=str, sort_keys=True)
    except Exception:
        sig = repr(sorted((k, str(v)) for k, v in p.items()
                          if k not in _volatile))
    run_dir = _external_run_dir(path, cache_root, sig)
    stem = os.path.splitext(os.path.basename(path))[0]
    cached = os.path.join(run_dir, "firefly_extras",
                          f"{stem}_diffusion_summary.csv")
    if os.path.isfile(cached):
        _log(f"  Reusing cached analysis of {os.path.basename(path)}.")
        return run_dir

    p["out_dir"] = run_dir
    p["wrap_in_stem_folder"] = False       # land outputs flat in run_dir
    p["skip_figure"] = True                # the tab needs the data sidecars, not
                                           # the per-file PNG — and this avoids a
                                           # matplotlib pass on a background thread
    try:
        os.makedirs(run_dir, exist_ok=True)
    except OSError as exc:
        _log(f"  Couldn't create analysis cache folder: {exc}")
        return None

    from firefly import firefly_worker
    cancel = cancel if cancel is not None else _threading.Event()
    _log(f"  Analysing {os.path.basename(path)} …")
    try:
        payload = firefly_worker._run_one_analysis(
            p, _queue.Queue(maxsize=100000), cancel,
            lambda m: _log(m), lambda _pct, _m: None)
    except firefly_worker._NoTracks:
        _log(f"  {os.path.basename(path)} produced no trajectories.")
        return None
    except BaseException as exc:                       # noqa: BLE001
        if type(exc).__name__ in ("_Cancelled", "_Stopped"):
            return None
        _log(f"  Analysis of {os.path.basename(path)} failed: {exc}")
        return None
    out = (payload or {}).get("out_dir")
    return out if (out and os.path.isdir(out) and is_run_folder(out)) else None


# ── statistics ─────────────────────────────────────────────────────────────
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's δ effect size in [-1, 1].  P(x>y) − P(x<y) over all pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return 0.0
    diff = np.sign(x[:, None] - y[None, :])
    return float(diff.sum() / (x.size * y.size))


def effect_magnitude(d: float) -> str:
    a = abs(d)
    if a < 0.147:
        return "negligible"
    if a < 0.33:
        return "small"
    if a < 0.474:
        return "medium"
    return "large"


def stars(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# statistical-test labels → scipy callables.  Each returns a two-sided p-value
# or NaN when the test can't run (too few / degenerate / mismatched-length).
def _p_two_sample(test: str, x: np.ndarray, y: np.ndarray, paired: bool) -> float:
    from scipy import stats
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    y = np.asarray(y, dtype=float); y = y[np.isfinite(y)]
    if x.size < 2 or y.size < 2:
        return float("nan")
    try:
        if test == "Mann–Whitney U":
            return float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
        if test == "Kolmogorov–Smirnov":
            return float(stats.ks_2samp(x, y).pvalue)
        if test == "Welch's t-test":
            return float(stats.ttest_ind(x, y, equal_var=False).pvalue)
        if test == "Kruskal–Wallis":
            return float(stats.kruskal(x, y).pvalue)
        if test == "Wilcoxon signed-rank":
            n = min(x.size, y.size)
            if n < 2 or np.allclose(x[:n], y[:n]):
                return float("nan")
            return float(stats.wilcoxon(x[:n], y[:n]).pvalue)
        if test == "Paired t-test":
            n = min(x.size, y.size)
            if n < 2:
                return float("nan")
            return float(stats.ttest_rel(x[:n], y[:n]).pvalue)
        # default
        return float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")


def _correct(pvals: list[float], correction: str) -> list[float]:
    """Apply a multiple-comparison correction, skipping NaN (underpowered)
    pairs so they don't poison the family.  Returns adjusted p-values aligned
    with the input (NaN stays NaN)."""
    out = list(pvals)
    finite_idx = [i for i, p in enumerate(pvals) if p is not None and np.isfinite(p)]
    if not finite_idx or correction == "None":
        return out
    finite_p = [pvals[i] for i in finite_idx]
    try:
        from statsmodels.stats.multitest import multipletests
        method = {"Bonferroni": "bonferroni",
                  "Benjamini–Hochberg (FDR)": "fdr_bh"}.get(correction, "bonferroni")
        adj = multipletests(finite_p, method=method)[1]
    except Exception:
        # graceful fallback: Bonferroni by hand
        k = len(finite_p)
        adj = [min(1.0, p * k) for p in finite_p]
    for slot, val in zip(finite_idx, adj):
        out[slot] = float(val)
    return out


def pairwise_stats(groups: list[dict], metric: Metric, *, test: str,
                   correction: str, alpha: float, paired: bool = False) -> list[dict]:
    """Every condition pair's significance for ``metric``.

    ``groups`` is a list of ``{"label", "color", "phase", "values": ndarray}``
    where ``values`` are the per-folder replicate scalars.  Returns one dict per
    pair with raw + corrected p, Cliff's δ, magnitude, stars and ``sig``.
    """
    n = len(groups)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    raw = []
    rows = []
    for i, j in pairs:
        x = np.asarray(groups[i]["values"], dtype=float)
        y = np.asarray(groups[j]["values"], dtype=float)
        p = _p_two_sample(test, x, y, paired)
        d = cliffs_delta(x, y)
        raw.append(p)
        rows.append({"a": groups[i], "b": groups[j], "p_raw": p, "delta": d})
    adj = _correct(raw, correction)
    for row, p_adj in zip(rows, adj):
        row["p"] = p_adj
        row["sig"] = bool(p_adj is not None and np.isfinite(p_adj) and p_adj < alpha)
        row["stars"] = stars(p_adj)
        row["magnitude"] = effect_magnitude(row["delta"])
    return rows


def engine_pairwise_stats(groups: list[dict], stats_config: dict):
    """Real-engine N-group stats on the per-folder replicate scalars, via
    ``fa_circular._stat_test_n`` — the same code path the full report uses, so the
    live numbers match the report.  Returns ``(rows, omnibus)``: ``rows`` mirror
    :func:`pairwise_stats` (the significance UI is unchanged) and additionally
    carry the engine's effect sizes; ``omnibus`` is the overall-test dict (Welch /
    one-way ANOVA or Kruskal for 3+ groups) or ``None``."""
    from firefly.analysis import fa_circular as fc
    from firefly.analysis import fa_stats_config as fsc
    arrays = [np.asarray(g["values"], dtype=float) for g in groups]
    labels = [g.get("label", f"Group {k + 1}") for k, g in enumerate(groups)]
    sc = fsc.normalize_stats_config(stats_config or {})
    alpha = float(sc.get("alpha", 0.05))
    try:
        omnibus, pw = fc._stat_test_n(arrays, labels, sc)
    except Exception:
        return [], None, []
    # Correct the family of p-values, but leave self-correcting post-hoc rows
    # (Games–Howell / Tukey) untouched so they aren't double-corrected.
    free = [k for k, r in enumerate(pw) if not r.get("self_corrected")]
    adj = fsc.correct_pvalues([pw[k].get("p") for k in free], sc.get("correction", "holm"))
    pcorr = {k: adj[m] for m, k in enumerate(free)}
    rows = []
    for k, r in enumerate(pw):
        p_use = pcorr.get(k, r.get("p"))
        d = r.get("cliffs_delta")
        d = float(d) if (d is not None and np.isfinite(d)) else 0.0
        rows.append({
            "a": groups[r["i"]], "b": groups[r["j"]],
            "p_raw": r.get("p"), "p": p_use, "delta": d,
            "magnitude": effect_magnitude(d),
            "sig": bool(p_use is not None and np.isfinite(p_use) and p_use < alpha),
            "stars": fsc.stars_for(p_use, alpha),
            "test": r.get("test", ""), "note": r.get("note", ""),
            "hedges_g": r.get("hedges_g"),
            "hedges_g_ci": (r.get("hedges_g_ci_low"), r.get("hedges_g_ci_high")),
            "cliffs_ci": (r.get("cliffs_delta_ci_low"), r.get("cliffs_delta_ci_high")),
            "rank_biserial": r.get("rank_biserial"),
            "tost_equivalent": r.get("tost_equivalent"), "tost_p": r.get("tost_p"),
        })
    return rows, omnibus, pw    # pw = raw engine pairwise (for verdict text)


def es_magnitude(es: Optional[float]) -> str:
    """Cohen magnitude word for an η²/ε² omnibus effect size."""
    if es is None or not np.isfinite(es):
        return ""
    return ("large" if es >= 0.14 else "medium" if es >= 0.06
            else "small" if es >= 0.01 else "negligible")


# ── comparison-report figure panels (keys match fa_compare's panel_order) ───────
COMPARE_PANELS = [
    ("msd", "Ensemble MSD"), ("auc", "MSD AUC"), ("fluor", "Fluorescence"),
    ("logd_dist", "log₁₀(D) dist."),
    ("mob_immob", "Mobile / immobile"), ("motion_classes", "Motion classes"),
    ("track_length", "Track length"), ("rg", "Radius of gyration"),
    ("netdisp", "Net displacement"), ("path", "Path length"),
    ("step", "Step distance"), ("speed", "Step speed"),
    ("dir", "Directionality"), ("dur", "Track duration"),
    ("track_count", "Track count"), ("nlocs", "Localisations"),
    ("jdd", "Jump distance"), ("dwell_cdf", "Dwell-time CDF"),
    ("turning_angles", "Turning angles"), ("radial_dist", "Radial dist."),
    ("van_hove", "Van Hove"), ("vacf", "VACF"),
]
COMPARE_PANEL_PRESETS = {
    "Essential": ("msd", "logd_dist", "motion_classes", "track_count"),
    "Diffusion": ("msd", "auc", "logd_dist", "jdd", "mob_immob"),
    "Dynamics": ("turning_angles", "radial_dist", "van_hove", "vacf", "dwell_cdf"),
}
# the engine's default set is every panel EXCEPT track_count
DEFAULT_COMPARE_PANELS = {k for k, _ in COMPARE_PANELS} - {"track_count"}
LOGD_STYLES = [("overlaid", "Overlaid"), ("ridgeline", "Ridgeline"),
               ("violin", "Violin"), ("faceted", "Faceted")]

# Live metric → the export figure panel it corresponds to (so the live preview
# can render the EXACT engine panel).  Metrics with no entry have no dedicated
# export panel (they're scalar-only comparisons) and keep the bespoke preview.
METRIC_PANEL = {
    "D": "logd_dist", "mob": "mob_immob", "motion": "motion_classes",
    "len": "track_length", "msd": "msd", "angle": "turning_angles",
    "dwell": "dwell_cdf", "a2": "van_hove", "fluor": "fluor", "rg": "rg",
    "netdisp": "netdisp", "path": "path", "dir": "dir", "dur": "dur",
    "nlocs": "nlocs", "step": "step", "speed": "speed",
}

# The scroller IS the comparison-figure panels (key, chip label, scalar metric for
# the stats cards or "" when the panel is visualisation-only).  Order = the
# engine's panel_order so the live scroller mirrors the exported figure.
COMPARE_PANEL_TABS = [
    ("msd", "Mean square displacement", "msd"),
    ("auc", "Mean square displacement AUC", "auc"),
    ("fluor", "Fluorescence intensity", "fluor"),
    ("logd_dist", "Diffusion D", "D"),
    ("mob_immob", "Mobile fraction", "mob"),
    ("motion_classes", "Motion classes", "motion"),
    ("track_length", "Track length", "len"),
    ("rg", "Radius of gyration", "rg"),
    ("netdisp", "Net displacement", "netdisp"),
    ("path", "Path length", "path"),
    ("step", "Step distance", "step"),
    ("speed", "Step speed", "speed"),
    ("dir", "Directionality ratio", "dir"),
    ("dur", "Track duration", "dur"),
    ("track_count", "Track count", "count"),
    ("nlocs", "Localisations", "nlocs"),
    ("jdd", "Jump distance", "jdd"),
    ("dwell_cdf", "Dwell time", "dwell"),
    ("turning_angles", "Turning angle", "angle"),
    ("radial_dist", "Radial distribution", "radial"),
    ("van_hove", "Non-Gaussian α₂", "a2"),
    ("vacf", "Persistence (VACF)", "vacf"),
]
PANEL_METRIC = {k: m for k, _l, m in COMPARE_PANEL_TABS if m}
PANEL_LABEL = {k: lbl for k, lbl, _m in COMPARE_PANEL_TABS}
PANEL_KEYS = [k for k, _l, _m in COMPARE_PANEL_TABS]


def recommend_config(n_groups: int, paired: bool, multi_folder: bool) -> dict:
    """Propose a sensible test / correction / normalisation from the data shape
    (mirrors the prototype's ``recommendCfg``)."""
    n_pairs = n_groups * (n_groups - 1) // 2
    if paired:
        test = "Wilcoxon signed-rank"
        why = "Wilcoxon signed-rank · paired by timepoint"
    else:
        test = "Kruskal–Wallis" if n_groups > 2 else "Mann–Whitney U"
        why = (f"Kruskal–Wallis for {n_groups} groups" if n_groups > 2
               else "Mann–Whitney U for 2 groups")
    correction = ("Benjamini–Hochberg (FDR)" if n_pairs >= 4
                  else "Bonferroni" if n_pairs > 1 else "None")
    why += " · " + ("no correction (single pair)" if correction == "None"
                    else f"{correction.split(' ')[0]} over {n_pairs} pairs")
    return {
        "cfg": {"test": test, "correction": correction,
                "alpha": "0.05", "err": "95% CI", "plot": "Violin", "logX": True},
        "why": why,
    }
