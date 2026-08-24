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
import re
from typing import Callable, Optional

import numpy as np
import pandas as pd

from firefly.analysis.fa_constants import (DEFAULT_FRAME_INTERVAL_S,
                                           DEFAULT_PIXEL_SIZE_UM,
                                           MOBILE_D_THRESHOLD_DEFAULT)

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


def _normalise_step_definition(raw, schema: int) -> str:
    """Canonical compatibility token for a persisted step definition."""
    value = str(raw or "").strip().lower()
    if int(schema) < 2:
        return "adjacent_observation"
    if (not value
            or "unit_frame" in value
            or "one frame" in value
            or "single_frame" in value):
        return "single_frame"
    return value


# ── one loaded run folder ─────────────────────────────────────────────────
class RunData:
    """A single analysis-output run folder.

    Holds the cheap per-run summary (read eagerly) and lazily reads the larger
    per-track CSVs only when a distribution is actually requested.  Every
    accessor degrades to ``None`` / empty rather than raising, so a malformed or
    partial run never breaks the live recompute.
    """

    # D (µm²/s) splitting mobile from immobile, for metrics recomputed live.
    # A CLASS attribute so changing the Preferences value re-reads every loaded
    # run without rebuilding them; the controller sets it alongside the value it
    # hands the engine, keeping the stats card and the drawn panel on one number.
    mobile_d: float = MOBILE_D_THRESHOLD_DEFAULT

    def __init__(self, folder: str, stem: str, extras_dir: str, summary: dict):
        self.folder = folder
        self.stem = stem
        self.extras_dir = extras_dir
        self.summary = summary or {}
        # A run can be scientifically complete even when the optional summary
        # sidecar failed to save (the worker deliberately treats that export as
        # non-fatal).  Keep the params sidecar as a second metadata authority so
        # a schema-2 run never silently becomes "legacy schema 1" merely because
        # ``summary_metrics.json`` is missing or corrupt.
        self.params: dict = {}
        try:
            params_path = os.path.join(extras_dir, f"{stem}_params.json")
            if os.path.isfile(params_path):
                with open(params_path) as fh:
                    loaded = json.load(fh)
                if isinstance(loaded, dict):
                    self.params = loaded
        except Exception:
            self.params = {}

        def _metadata(key, default=None):
            value = self.summary.get(key)
            if value is None or value == "":
                value = self.params.get(key)
            return default if value is None or value == "" else value

        self.n_tracks = int(self.summary.get("n_tracks") or 0)
        self.n_locs = int(self.summary.get("n_locs") or 0)
        fi = self.summary.get("fi_s")
        if not fi:
            fi = self.params.get("frame_interval_s")
        self.fi_s = float(fi) if fi else None
        self._cache: dict = {}
        try:
            self.metrics_schema_version = int(
                _metadata("metrics_schema_version", 1))
        except (TypeError, ValueError):
            self.metrics_schema_version = 1
        # Runs predating the explicit contract used the contiguous-observation
        # estimator.  Treat a missing token as that legacy policy rather than
        # silently pretending old numbers used the new all-pairs default.
        self.gap_policy = str(
            _metadata("gap_policy")
            or ("contiguous" if self.metrics_schema_version < 2
                else "all_pairs"))
        self.metric_contract = str(
            _metadata("metric_contract")
            or f"firefly_metrics_schema_{self.metrics_schema_version}")
        self.step_definition = _normalise_step_definition(
            _metadata("step_definition"),
            self.metrics_schema_version)
        # Retain the remaining persisted contract/provenance fields for callers
        # that need to explain a partial run.  Summary values win; params fill
        # only absent fields, matching the four compatibility keys above.
        self.link_definition = str(_metadata("link_definition", ""))
        self.duration_definition = str(_metadata("duration_definition", ""))
        self.observed_time_definition = str(
            _metadata("observed_time_definition", ""))
        self.metric_contract_note = str(_metadata("metric_contract_note", ""))
        self.effective_calibration = _metadata("effective_calibration", {})
        self.embedded_calibration = _metadata("embedded_calibration", {})

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
    """Mobile fraction (%), recomputed from the per-track D at the CURRENT
    threshold rather than read from the summary.

    ``mobile_fraction`` is frozen into summary_metrics.json by the worker at
    process time, so a run analysed under one threshold kept reporting that
    value forever — while the Mobile/Immobile panel drawn right above the stats
    card was recomputed live by the engine.  The two could disagree on the same
    run.  Recomputing here puts both on one number and makes a threshold change
    apply retroactively, with no reprocessing.

    The stored value is still the fallback for runs with no per-track table
    (palmTRACER caches, partial outputs)."""
    d = run._diff_col("D", positive=True)
    if d is not None and d.size:
        return 100.0 * float((d >= float(run.mobile_d)).mean())
    v = run.summary.get("mobile_fraction")
    return float(v) * 100.0 if v is not None else None


def _motion_frac(run: RunData, cls: str) -> Optional[float]:
    counts = run.summary.get("motion_counts") or {}
    # Unknown/below-resolution tracks are shown separately and are not part of
    # an alpha-derived motion-class denominator.
    classified = ("Immobile", "Confined", "Brownian", "Directed")
    total = sum(int(counts.get(name, 0)) for name in classified)
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
    v = run._diff_col("mean_step_um")
    fi = run.fi_s
    if v is None or not len(v) or not fi:
        return None
    return float(np.median(v) / float(fi))


def _speed_measured_dist(run: RunData) -> Optional[np.ndarray]:
    v = run._diff_col("mean_step_um")
    fi = run.fi_s
    if v is None or not len(v) or not fi:
        return None
    return v / float(fi)


def _col_median(run: RunData, col: str) -> Optional[float]:
    """Median of one per-track diffusion_summary column (for a scalar readout)."""
    # Zero is a valid physical value for linear geometry (a static step, a
    # return to the origin, or zero radius of gyration).  Positivity filtering
    # belongs only at log-D call sites.
    v = run._diff_col(col)
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
    """Per-track elapsed duration: (last frame − first frame) × Δt."""
    # Schema-2 runs persist the canonical value directly.
    persisted = run._diff_col("track_duration_s")
    if persisted is not None:
        return persisted
    tr = run._read_csv("_trajectories.csv")
    fi = run.fi_s
    if (tr is None or not fi
            or {"particle", "frame"} - set(getattr(tr, "columns", []))):
        return None
    fr = pd.to_numeric(tr["frame"], errors="coerce")
    tmp = tr.assign(_frame=fr).dropna(subset=["particle", "_frame"])
    if tmp.empty:
        return None
    span = (tmp.groupby("particle")["_frame"].max()
            - tmp.groupby("particle")["_frame"].min())
    dur = span.to_numpy(dtype=float) * float(fi)
    return dur if dur.size else None


def _link_speed_scalar(run: RunData) -> Optional[float]:
    return _col_median(run, "mean_link_speed_um_s")


def _link_speed_dist(run: RunData) -> Optional[np.ndarray]:
    return run._diff_col("mean_link_speed_um_s")


# Metrics whose numerical definition changed between the legacy implicit
# schema and metrics schema 2.  Motion class and VACF belong here too: both are
# temporal inferences derived from the same gap-aware trajectory semantics, not
# source-stable labels.  The controller/report engine uses these keys to prevent
# incompatible values from being silently pooled.
_GAP_CONTRACT_METRICS = {
    "D", "a", "mob", "motion", "msd", "mss", "auc", "vacf",
}
_STEP_CONTRACT_METRICS = {"step", "speed", "linkstep", "linkspeed"}


def metric_contract_key(run: RunData, metric_id: str):
    """Return the compatibility key for one run/metric, or ``None`` when the
    metric is stable across the schema transition."""
    if metric_id in _GAP_CONTRACT_METRICS:
        return ("gap", int(run.metrics_schema_version), str(run.gap_policy),
                str(run.metric_contract))
    if metric_id in _STEP_CONTRACT_METRICS:
        return ("step", int(run.metrics_schema_version),
                str(run.step_definition), str(run.metric_contract))
    return None


def metric_contract_issue(runs, metric_id: str) -> str:
    """Explain an incompatible mixed-run metric selection.

    Stable metrics return ``""``.  A legacy-only cohort is allowed and labelled
    by :func:`metric_contract_label`; only differing keys suppress pooling.
    """
    keyed = [(r, metric_contract_key(r, metric_id)) for r in runs]
    keys = {k for _r, k in keyed if k is not None}
    if len(keys) <= 1:
        return ""
    if metric_id in _STEP_CONTRACT_METRICS:
        details = ", ".join(sorted({
            f"schema {r.metrics_schema_version}/{r.step_definition}"
            for r, k in keyed if k is not None
        }))
    else:
        details = ", ".join(sorted({
            f"schema {r.metrics_schema_version}/{r.gap_policy}"
            for r, k in keyed if k is not None
        }))
    return (f"Incompatible metric definitions ({details}). Re-analyse legacy "
            f"runs or select one contract before pooling this metric.")


def metric_contract_label(runs, metric_id: str) -> str:
    """Short badge text for a homogeneous legacy metric cohort."""
    keys = [metric_contract_key(r, metric_id) for r in runs]
    keys = [k for k in keys if k is not None]
    if keys and all(getattr(r, "metrics_schema_version", 1) < 2 for r in runs):
        return "legacy definition"
    return ""


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
# The All-panels view mirrors the single-run analysis figure EXACTLY — the 19
# fa_figure.make_figure panels (A–S) and nothing else.  Two render routes:
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
    {"name": "Track length",                  "cat": "Tracking",   "kind": "mfig", "letter": "R"},
    {"name": "Total tracks",                  "cat": "Tracking",   "kind": "mfig", "letter": "S"},
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
           dist=lambda r: r._diff_col("mean_step_um")),
    Metric("speed", "Step speed", "µm/s", 3, "step speed (µm/s)", "Tracking",
           scalar=_speed_measured_scalar, dist=_speed_measured_dist),
    Metric("linkstep", "Observed-link distance", "µm", 3,
           "observed-link distance (µm)", "Tracking",
           scalar=lambda r: _col_median(r, "mean_link_displacement_um"),
           dist=lambda r: r._diff_col("mean_link_displacement_um")),
    Metric("linkspeed", "Observed-link speed", "µm/s", 3,
           "observed-link speed (µm/s)", "Tracking",
           scalar=_link_speed_scalar, dist=_link_speed_dist),
    Metric("rg", "Radius of gyration", "µm", 3, "R_g (µm)", "Tracking",
           scalar=lambda r: _col_median(r, "radius_of_gyration_um"),
           dist=lambda r: r._diff_col("radius_of_gyration_um")),
    Metric("netdisp", "Net displacement", "µm", 3, "net displacement (µm)", "Tracking",
           scalar=lambda r: _col_median(r, "net_displacement_um"),
           dist=lambda r: r._diff_col("net_displacement_um")),
    Metric("path", "Path length", "µm", 3, "path length (µm)", "Tracking",
           scalar=lambda r: _col_median(r, "path_length_um"),
           dist=lambda r: r._diff_col("path_length_um")),
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


def _is_palmtracer_dir(folder: str) -> bool:
    """True when *folder* (or its ``data/`` subdir) holds raw palmTRACER output.

    A raw palmTRACER folder carries no FIREFLY sidecars, so ``_resolve_extras``
    cannot see it even though ``load_summary_from_folder`` reads it natively.
    Probing for it here is what lets a never-before-opened ``.PT`` folder be
    accepted instead of flagged as invalid.  Cheap: one directory listing.
    """
    try:
        from firefly.analysis.fa_palmtracer import _is_palmtracer_folder
    except Exception:
        return False
    try:
        if _is_palmtracer_folder(folder):
            return True
        data = os.path.join(folder, "data")
        return os.path.isdir(data) and _is_palmtracer_folder(data)
    except Exception:
        return False


# Per-track columns a CURRENT derivation writes.  A palmTRACER cache written by
# an older FIREFLY lacks them, which silently blanks the graphs that read them.
_CURRENT_TRACK_COLUMNS = ("path_length_um", "net_displacement_um",
                          "directionality_ratio", "mean_step_um",
                          "track_duration_s")


def _palmtracer_cache_is_stale(folder: str, extras_dir: str, stem: str) -> bool:
    """True when *folder* is palmTRACER output whose cached per-track table
    predates the current metric set.

    Only palmTRACER folders qualify: their source files sit right beside the
    cache, so re-deriving is cheap and lossless.  A FIREFLY run folder is never
    touched — its raw movie may be long gone, and silently recomputing someone's
    recorded output would be worse than showing what was actually saved.
    """
    if not _is_palmtracer_dir(folder):
        return False
    path = os.path.join(extras_dir, f"{stem}_diffusion_summary.csv")
    if not os.path.isfile(path):
        return True
    try:
        cols = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    if not set(_CURRENT_TRACK_COLUMNS).issubset(cols):
        return True
    # A cache built under a different minimum track length describes a different
    # track set, so reusing it would silently mix filtering policies.
    try:
        from firefly.analysis.fa_palmtracer import PALMTRACER_MIN_TRACK_LEN
        pj = os.path.join(extras_dir, f"{stem}_params.json")
        if not os.path.isfile(pj):
            return True
        with open(pj) as fh:
            cached = json.load(fh).get("min_track_len")
        return int(cached or 0) != int(PALMTRACER_MIN_TRACK_LEN)
    except Exception:
        return False


# ── experiment naming convention → condition / animal / side ────────────────
# Acquisition folders are named like
#     "N=2 MB543B-Sx1A-mEos3.2_CrimsonVenus 31July Propofol"
# and the recordings inside like
#     "Fly-1-16k Frames-LSide.czi"
# The DRUG is whatever trails the date in an ancestor folder; a bare date means
# control.  It matters that we search ANCESTORS: a run folder is named from the
# recording's stem ("Fly-1-16k Frames-LSide"), which carries the animal and side
# but never the condition.
_DATE_RE = re.compile(
    r"\b(\d{1,2})\s*"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b"
    r"(?P<trailing>.*)$", re.IGNORECASE)
_GENOTYPE_RE = re.compile(r"\bMB\d+[A-Z]\b", re.IGNORECASE)
_ANIMAL_RE = re.compile(r"\bfly[\s_-]*(\d+)\b", re.IGNORECASE)
_SIDE_RE = re.compile(r"\b([LR])[\s_-]*side\b", re.IGNORECASE)

CONTROL_LABEL = "Control"


def parse_experiment_path(path: str) -> dict:
    """Pull condition / genotype / animal / side out of an acquisition path.

    Returns a dict with ``condition`` (never empty — a bare date means
    :data:`CONTROL_LABEL`), plus ``genotype``, ``animal``, ``side``, ``date``
    and ``matched`` (False when no date token was found anywhere, i.e. the
    convention did not apply and the caller should not guess).
    """
    out = {"condition": "", "genotype": "", "animal": "", "side": "",
           "date": "", "matched": False}
    if not path:
        return out
    parts = [p for p in str(path).replace("\\", "/").split("/") if p]
    # Ancestors carry genotype + date + drug; the leaf carries animal + side.
    for part in parts:
        m = _DATE_RE.search(part)
        if m and not out["matched"]:
            out["date"] = f"{m.group(1)}{m.group(2).title()}"
            trailing = (m.group("trailing") or "").strip(" _-")
            out["condition"] = trailing or CONTROL_LABEL
            out["matched"] = True
        g = _GENOTYPE_RE.search(part)
        if g and not out["genotype"]:
            out["genotype"] = g.group(0).upper()
    leaf = os.path.splitext(parts[-1])[0] if parts else ""
    for src in (leaf, " ".join(parts)):
        a = _ANIMAL_RE.search(src)
        if a and not out["animal"]:
            out["animal"] = f"Fly-{int(a.group(1))}"
        s = _SIDE_RE.search(src)
        if s and not out["side"]:
            out["side"] = f"{s.group(1).upper()}Side"
    return out


def _run_calibration(extras_dir: str, stem: str):
    """(pixel_size_um, frame_interval_s) for a run folder, or (None, None)."""
    for name, keys in ((f"{stem}_summary_metrics.json", ("px_um", "fi_s")),
                       (f"{stem}_params.json",
                        ("pixel_size_um", "frame_interval_s"))):
        path = os.path.join(extras_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception:
            continue
        px, fi = data.get(keys[0]), data.get(keys[1])
        if px and fi:
            return float(px), float(fi)
    return None, None


def _run_metrics_schema(extras_dir: str, stem: str) -> int:
    """Persisted metrics schema for a run, defaulting to legacy schema 1.

    Summary metadata is authoritative when present; the params sidecar is the
    recovery source for otherwise-valid partial runs whose summary export failed.
    This mirrors :class:`RunData`'s compatibility lookup.
    """
    for name in (f"{stem}_summary_metrics.json", f"{stem}_params.json"):
        path = os.path.join(extras_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                data = json.load(fh)
            raw = data.get("metrics_schema_version")
            if raw is not None and raw != "":
                return int(raw)
        except Exception:
            continue
    return 1


def _backfill_track_geometry(extras_dir: str, stem: str) -> bool:
    """Add the newer per-track geometry columns to an older run's saved table.

    These quantities (path length, net displacement, directionality, step,
    duration) depend ONLY on the cached trajectories, which every run folder
    keeps — so a run analysed before they existed can gain them without the raw
    movie and without re-fitting anything.  Step semantics follow the persisted
    contract: schema 1 uses every adjacent observed link; schema 2 uses only
    exactly-one-frame links.  Purely additive: D, alpha, motion and every other
    recorded value are left exactly as they were saved, because recomputing those
    WOULD change published output.

    Returns True when the table was rewritten.  The formulas mirror
    ``fa_diffusion._msd_and_fit_one`` and are pinned to it by a test.
    """
    diff_path = os.path.join(extras_dir, f"{stem}_diffusion_summary.csv")
    traj_path = os.path.join(extras_dir, f"{stem}_trajectories.csv")
    if not (os.path.isfile(diff_path) and os.path.isfile(traj_path)):
        return False
    px, fi = _run_calibration(extras_dir, stem)
    if not px or not fi:
        return False
    metrics_schema = _run_metrics_schema(extras_dir, stem)
    try:
        diff = pd.read_csv(diff_path)
        tr = pd.read_csv(traj_path)
    except Exception:
        return False
    if "particle" not in diff.columns or {"particle", "frame", "x", "y"} - set(tr.columns):
        return False

    rows = []
    tr = tr.sort_values(["particle", "frame"], kind="stable")
    for pid, g in tr.groupby("particle", sort=False):
        xy = g[["x", "y"]].to_numpy(dtype=float) * float(px)
        frames = g["frame"].to_numpy(dtype=float)
        n = len(xy)
        if n < 2:
            # One localisation is not a trajectory — undefined, not zero.
            rows.append((pid, np.nan, np.nan, np.nan, np.nan,
                         0.0 if n else np.nan))
            continue
        steps = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
        path = float(steps.sum())
        net = float(np.sqrt(np.sum((xy[-1] - xy[0]) ** 2)))
        unit = np.diff(frames) == 1
        # ``mean_step_um`` changed meaning in schema 2.  Backfilling the current
        # one-frame value into a schema-1 run while labelling it
        # ``adjacent_observation`` makes two nominally compatible legacy runs
        # numerically incompatible on gapped tracks.  Reconstruct the definition
        # the run's contract actually promises.
        mean_step = (float(steps.mean()) if metrics_schema < 2
                     else float(steps[unit].mean()) if unit.any() else np.nan)
        rows.append((
            pid, path, net,
            float(net / path) if path > 0 else 0.0,
            mean_step,
            float(frames[-1] - frames[0]) * float(fi)))
    geo = pd.DataFrame(rows, columns=[
        "particle", "path_length_um", "net_displacement_um",
        "directionality_ratio", "mean_step_um", "track_duration_s"])
    add = [c for c in geo.columns if c != "particle" and c not in diff.columns]
    if not add:
        return False
    merged = diff.merge(geo[["particle"] + add], on="particle", how="left")
    try:
        merged.to_csv(diff_path, index=False)
    except Exception:
        return False
    return True


def load_run(folder: str) -> Optional[RunData]:
    """Load one analysis-output run folder, or ``None`` if it isn't one."""
    resolved = _resolve_extras(folder)
    # Refresh a palmTRACER cache that predates the current metrics, so the
    # geometry/time graphs are populated instead of silently empty.
    if resolved is not None and _palmtracer_cache_is_stale(folder, *resolved):
        try:
            # Must go through load_summary_from_palmtracer: the generic loader
            # would find the existing firefly_extras and hand back the very
            # stale cache we are trying to replace.  cache=True rewrites it.
            from firefly.analysis.fa_palmtracer import (
                load_summary_from_palmtracer, _is_palmtracer_folder)
            if _is_palmtracer_folder(folder):
                load_summary_from_palmtracer(folder, cache=True)
                resolved = _resolve_extras(folder) or resolved
        except Exception:
            pass                            # keep the old cache rather than fail
    if resolved is None and _is_palmtracer_dir(folder):
        # Raw palmTRACER: let the analysis loader derive FIREFLY's metrics.  It
        # caches them into <folder>/firefly_extras (cache=True), after which the
        # normal sidecar path resolves — so this conversion happens once.
        try:
            from firefly.analysis.fa_palmtracer import load_summary_from_folder
            load_summary_from_folder(folder)
        except Exception:
            return None
        resolved = _resolve_extras(folder)
    # A FIREFLY run predating the newer geometry columns can still gain them:
    # they come from the cached trajectories, not the raw movie.  Additive only.
    if resolved is not None and not _is_palmtracer_dir(folder):
        try:
            _backfill_track_geometry(*resolved)
        except Exception:
            pass                            # show what was saved rather than fail
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
    """True for anything :func:`load_run` can turn into a replicate — FIREFLY's
    own output AND a raw palmTRACER folder.  The two must agree: a folder the
    loader can read but this gate rejects shows up as an invalid (red) chip."""
    return _resolve_extras(folder) is not None or _is_palmtracer_dir(folder)


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
_EXTERNAL_CACHE_SCHEMA = 2


def is_external_loc_file(path: str) -> bool:
    """True when *path* is a localisation-table FILE (an external tool's export)
    rather than a FIREFLY run folder — i.e. something we must analyse before it
    can be a replicate."""
    return (bool(path) and os.path.isfile(path)
            and str(path).lower().endswith(_EXTERNAL_LOC_EXTS))


class _ExternalImportStub:
    """The handful of ImportController attributes `build_params` reads, for a
    bare external-localisation file.

    External tables have no image-container calibration path.  Mirror the
    Import tab's CSV branch by exposing the *current sidebar* calibration so
    the on-the-fly workspace replicate is numerically identical to an ordinary
    external-table run.  ``override*`` stays false: ``build_params`` already
    applies these values for CSV inputs even without an image-style override.
    """
    def __init__(self, path: str, settings):
        self.filePath = path
        self.outDir = None
        self.isCsv = True
        self.overridePx = False
        self.pixelSize = float(settings.get_float(
            "analysis/pixel_size", DEFAULT_PIXEL_SIZE_UM))
        self.overrideFi = False
        self.frameInterval = float(settings.get_float(
            "analysis/frame_interval", DEFAULT_FRAME_INTERVAL_S))


def _external_run_dir(path: str, cache_root: str, sig: str) -> str:
    """Deterministic cache folder for analysing *path* under *cache_root*, keyed
    by the source-content digest + a signature of every number-affecting
    analysis setting. Path/mtime alone is insufficient: a file can be replaced
    in place while retaining its timestamp."""
    import hashlib
    try:
        digest = hashlib.blake2b(digest_size=20)
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        content_digest = digest.hexdigest()
    except OSError:
        content_digest = "unreadable"
    key = hashlib.blake2b(
        f"{content_digest}|{sig}".encode("utf-8"),
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
        p = build_params(settings, _ExternalImportStub(path, settings), fpath=path,
                         out_dir=None)
    except Exception as exc:
        _log(f"  Couldn't build analysis parameters for "
             f"{os.path.basename(path)}: {exc}")
        return None

    # Signature = the params that affect the NUMBERS (exclude per-run paths), so
    # the cache key changes iff the analysis would produce different results.
    _volatile = {"file", "out_dir", "stem_override", "widget_state",
                 "wrap_in_stem_folder"}
    signature_params = {k: v for k, v in sorted(p.items())
                        if k not in _volatile}
    # Bump this only when the interpretation of an external table changes.
    # It prevents a cache made under an older calibration policy from being
    # treated as reproducible merely because its other parameters match.
    signature_payload = {"external_cache_schema": _EXTERNAL_CACHE_SCHEMA,
                         "params": signature_params}
    try:
        sig = json.dumps(signature_payload, default=str, sort_keys=True)
    except Exception:
        sig = repr(("external_cache_schema", _EXTERNAL_CACHE_SCHEMA,
                    sorted((k, str(v)) for k, v in signature_params.items())))
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
    ("linkstep", "Observed-link distance"), ("linkspeed", "Observed-link speed"),
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
DEFAULT_COMPARE_PANELS = {
    k for k, _ in COMPARE_PANELS
} - {"track_count", "linkstep", "linkspeed"}
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
    "linkstep": "linkstep", "linkspeed": "linkspeed",
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
    ("linkstep", "Observed-link distance", "linkstep"),
    ("linkspeed", "Observed-link speed", "linkspeed"),
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
