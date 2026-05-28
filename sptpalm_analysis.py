#!/usr/bin/env python3
import multiprocessing
import sys
import os

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  RELEASE CHECKLIST                                                        ║
# ║  Every release MUST bump this string so the in-app update checker         ║
# ║  (app_qt._UpdateCheckThread → __version__) doesn't keep nagging users     ║
# ║  who already installed the latest binary.  The check compares this        ║
# ║  string against the latest GitHub tag — if they don't match, the nag      ║
# ║  fires.  Always touch this line in the same commit as the `git tag`.     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
__version__ = "2.6.18"

# Fix macOS multiprocessing crashes — must be set before any other imports
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

"""
FIREFLY — Fluorescence Inference & Reconstruction Engine  (OPTIMISED)
=======================================================================
Framework for Localization Yields.  Supports .czi (Zeiss native) and
.tif / .tiff files.
Pixel size and frame interval are read automatically from CZI metadata.

Speed optimisations vs the original version:
  - Background subtraction:  rolling_ball (~4800 ms/frame)
                           -> uniform_filter (~3 ms/frame)  [~1700x faster]
  - Preprocessing:           serial -> parallel across all CPU cores
  - Localisation:            single core -> all CPU cores
  - Memory:                  entire stack in RAM -> chunked processing
  - Progress:                silent -> live progress bars

Usage
-----
  # Typical usage — everything auto-detected from CZI:
  python sptpalm_analysis.py my_experiment.czi

  # With output folder:
  python sptpalm_analysis.py my_experiment.czi --output-dir C:\\results

  # Override metadata if needed:
  python sptpalm_analysis.py my_experiment.czi --pixel-size 0.104 --frame-interval 0.05

  # Limit CPU cores (default: all available):
  python sptpalm_analysis.py my_experiment.czi --workers 4

  # Use legacy rolling-ball background (slower but more accurate for uneven illumination):
  python sptpalm_analysis.py my_experiment.czi --bg-method rolling_ball

All options:
  --pixel-size       um per pixel (auto from CZI metadata)
  --frame-interval   seconds per frame (auto from CZI metadata)
  --diameter         PSF diameter in pixels, must be odd (default: 7)
  --minmass          Min integrated brightness (auto if omitted)
  --search-range     Max displacement between frames in px (default: 5)
  --memory           Frames a particle may vanish and reappear (default: 3)
  --min-track-length Discard tracks shorter than this (default: 5)
  --max-lagtime      MSD lag time points (default: 20)
  --bg-method        Background method: uniform_filter (fast) or rolling_ball (default: uniform_filter)
  --bg-radius        Background radius in pixels (default: 50)
  --workers          CPU cores to use (default: all)
  --chunk-size       Frames per processing chunk, reduce if RAM is low (default: 500)
  --channel          Channel index for multi-channel CZI (default: 0)
  --output-dir       Where to save results (default: same folder as input)
"""

import argparse
import multiprocessing
import os
import sys
import time
import warnings
import xml.etree.ElementTree as ET
warnings.filterwarnings("ignore")

# ── BLAS / OpenBLAS / MKL threading policy ─────────────────────────────────────
# Cap internal BLAS threads to 1.  We use ThreadPoolExecutor for preprocessing
# (one Python thread per frame, all calling scipy.ndimage which uses BLAS).
# Without this cap, we get N² threads (Python pool × BLAS pool) on N cores,
# which deadlocks Windows frozen apps before the first preview frame is sent.
#
# Per-frame numpy/scipy operations on small (256×256) images are too fast to
# benefit from BLAS threading anyway — chunk-level Python threading wins.
# This MUST be set before numpy is imported to take effect.
for _blas_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_blas_env, "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import trackpy as tp
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:
    # Fallback no-op context manager if threadpoolctl unavailable
    from contextlib import contextmanager as _cm
    @_cm
    def _threadpool_limits(limits=None, user_api=None):
        yield
from scipy.ndimage import uniform_filter, gaussian_filter, gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import correlate as _correlate2d
from scipy.stats import gaussian_kde
from skimage import filters, exposure
from tqdm import tqdm

# On Windows with console=False (PyInstaller GUI build), sys.stderr is None.
# tqdm writes to sys.stderr by default and crashes with AttributeError.
# Use sys.stdout instead — the GUI redirects stdout to its log panel, so
# tqdm progress lines will appear there in real time.
import io as _io

# Shared constants + leaf helpers now live in fa_constants and are
# re-exported here so existing `sptpalm_analysis.N_CPUS` / `_Cancelled` /
# `_tqdm` / `_dim_size` call sites keep working unchanged.
from fa_constants import N_CPUS, _Cancelled, _tqdm, _dim_size


tp.quiet()


# ══════════════════════════════════════════════════════════════════════════════
#  CZI LOADING + METADATA
# ══════════════════════════════════════════════════════════════════════════════

# CZI/TIF/external loaders now live in fa_loaders; re-exported here so
# existing `sptpalm_analysis.load_file(...)` etc. call sites keep working.
from fa_loaders import (
    load_file, load_czi, load_tif, load_external_locs, load_projection_fast,
    _parse_czi_metadata, _parse_ome_metadata, _find_czi_series,
    _find_tif_series, _load_single_czi, _load_single_tif,
    _probe_tif_shape_and_count, _tif_series_nat_key, _autodetect_csv_preset,
    HAS_AICS, HAS_CZIFILE, HAS_TIFFFILE,
)















# Memory / temp-stack management lives in fa_memory; re-exported here so
# existing call sites (and firefly_worker's cleanup_temp_stack_paths
# import) keep working unchanged.
from fa_memory import (
    set_temp_stack_dir, _resolve_temp_stack_dir, _register_temp_stack_path,
    cleanup_temp_stack_paths, _cleanup_temp_stack_paths,
    _alloc_or_memmap_stack, _user_ram_reserve_gb,
)








# ── External-localisations loader ─────────────────────────────────────────────
# Schema for a "preset" that maps an external tool's CSV columns to FIREFLY's
# canonical {frame, x, y, mass}.  Frame offset is added to the source values
# (-1 for 1-indexed tools); units lets us convert nm → px on the fly.








# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING  (fast path + parallel)
# ══════════════════════════════════════════════════════════════════════════════

# preprocess / drift / roi now live in fa_preprocess, fa_drift, fa_roi;
# re-exported here so existing call sites keep working.
from fa_preprocess import (_preprocess_fast, _preprocess_rolling,
                           preprocess_stack, auto_threshold)
from fa_drift import correct_drift
from fa_roi import (build_roi_mask_mean, build_roi_mask_perframe,
                    build_roi_mask, build_roi_mask_advanced, apply_roi_mask)








# ══════════════════════════════════════════════════════════════════════════════
#  ROI  —  simple intensity threshold
# ══════════════════════════════════════════════════════════════════════════════









# ──────────────────────────────────────────────────────────────────────────────
#  build_roi_mask_advanced — single source of truth for the GUI preview AND
#  the firefly_worker analysis path, so the green mask the user tunes in the
#  ROI viewer is *identical* to the mask actually applied during analysis.
#
#  Pipeline:
#      projection ─►  fine + coarse Gaussian blur (DoG bg subtraction)
#                  ►  normalise to [0, 1]
#                  ►  intensity threshold (manual or auto: Li/Otsu/…)
#                  ►  morphological opening   (kill 1-2 px speckle bridges)
#                  ►  morphological closing   (fill speckle-gap interior)
#                  ►  remove_small_holes      (merge interior pockets)
#                  ►  remove_small_objects    (drop sub-cell fragments)
#                  ►  keep top-N components   (hard cap on background islands)
#
#  All numeric defaults match the GUI preview (see _RoiViewer._refresh_roi_mask_
#  overlay in app_qt.py).  If you change a default here, update the docstring
#  hint in the GUI's "Background scale σ" tooltip too.
# ──────────────────────────────────────────────────────────────────────────────



# ══════════════════════════════════════════════════════════════════════════════
#  DRIFT CORRECTION
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
#  LOCALISATION  (parallel + chunked)
# ══════════════════════════════════════════════════════════════════════════════

# fa_localize extracted; re-exported here so call sites keep working.
from fa_localize import (
    _ram_strategy, _adaptive_chunk_and_workers,
    _fast_preprocess_and_localise, preprocess_and_localise_adaptive,
    preprocess_and_localise_stream, _localise_chunk, _localise_chunk_mp,
    _localise_chunk_mmap_mp, LocaliserBackend, _emit_trackpy_chunk_preview,
    TrackpyBackend, TorchBackend, list_available_backends, _resolve_backend,
    localise_particles, _BACKEND_REGISTRY,
)
















# ══════════════════════════════════════════════════════════════════════════════
#  LOCALISER BACKENDS
# ══════════════════════════════════════════════════════════════════════════════
#
# A backend takes a *preprocessed* stack (T × Y × X, float32) and returns a
# DataFrame with at least the columns: x, y, frame, mass.  Preprocessing
# (background subtraction, bandpass) is handled separately so the fast / stream
# RAM strategies in this file stay backend-agnostic.
#
# Registration model: subclass LocaliserBackend, set `.name`, implement
# `.is_available()` (classmethod) and `.localise(stack, **params)`, then append
# to _BACKEND_REGISTRY in the preference order used by `backend="auto"`.
#
# Phase A1: only TrackpyBackend exists (refactor — no behaviour change).
# Phase A2: TorchBackend (CPU) lands here.
# Phase A3: device selection (MPS / CUDA) inside TorchBackend.









# Order matters: `backend="auto"` resolves to the first available entry.
# TorchBackend stays AFTER TrackpyBackend so "auto" picks trackpy (the
# peer-reviewed reference) by default; users opt into the GPU path by
# selecting a "torch*" backend in the GUI.  FIREFLY ships exactly these
# two detection engines — both Crocker-Grier-family centroid detectors.








# ══════════════════════════════════════════════════════════════════════════════
#  LINKING
# ══════════════════════════════════════════════════════════════════════════════

# fa_linking extracted; re-exported here.
from fa_linking import (
    link_trajectories, _link_via_trackpy,
)




# ══════════════════════════════════════════════════════════════════════════════
#  MSD + DIFFUSION  (custom parallel — replaces slow tp.imsd)
# ══════════════════════════════════════════════════════════════════════════════

# fa_diffusion extracted; re-exported here.
from fa_diffusion import (
    msd_linear, classify_motion, _msd_and_fit_one, compute_msd_and_fit,
    compute_jdd, compute_turning_angles, compute_mobile_fraction_over_time,
    compute_dwell_times, compute_mss, _msd_auc, _mob_immob_ratio,
    _motion_fractions, _track_lengths, ALPHA_THRESHOLDS_DEFAULT,
    MOBILE_D_THRESHOLD_DEFAULT,
)


# Default alpha-exponent thresholds for the four-class motion classifier.
# Conventional sptPALM values: 0.5 / 0.9 / 1.1.  These are now the *defaults*
# but every public function that classifies motion accepts a thresholds=
# triple so users can tune the boundaries to their lab's convention.

# Default D cutoff for splitting Mobile / Immobile populations (µm²/s).
# 0.05 is the conventional membrane-protein threshold used throughout the
# sptPALM literature; tracks with D ≥ this value are considered Mobile.
# Defined here at the top so functions defined later in the file can use
# it as a default argument (Python evaluates defaults at definition time).








# ══════════════════════════════════════════════════════════════════════════════
#  JUMP DISTANCE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
#  TURNING ANGLES
# ══════════════════════════════════════════════════════════════════════════════

def compute_circular_statistics(angles_deg):
    """Full circular-statistics summary for an array of angles, in
    degrees, on the interval (-180°, +180°] (signed turning-angle
    convention used by `compute_turning_angles`).

    Returns a dict whose keys are the same statistic names MATLAB's
    CircStat toolbox (Berens 2009) uses, so a supervisor familiar with
    that toolbox can map results 1:1.  All angles in the output are in
    DEGREES; rates / dispersions in their natural units.

    Computed statistics
    -------------------
    n                            : sample size
    mean_direction_deg           : μ = atan2(S, C), in (-180°, +180°]
    mean_resultant_length        : R̄ in [0, 1]    (1 = perfect alignment)
    circular_variance            : 1 - R̄          ("S" in Fisher 1993)
    circular_std_deg             : √(-2·ln R̄)·(180/π)
    angular_deviation_deg        : √(2·(1 - R̄))·(180/π)  ("s₀" in Fisher)
    median_deg                   : circular median
    concentration_kappa          : von Mises κ via standard piecewise
                                   approximation (Best & Fisher 1981)
    rayleigh_z                   : n·R̄²   (test statistic for uniformity)
    rayleigh_p                   : Wilkie-Mardia approximation (good
                                   to ~5e-4 for n ≥ 10)
    v_test_z, v_test_p           : V-test against μ₀ = 0° (tests for a
                                   preferred mean direction at "straight
                                   ahead")
    circular_skewness            : b̄ / (1 - R̄)^1.5
    circular_kurtosis            : (ā - R̄⁴) / (1 - R̄)²
    ci95_lower_deg, ci95_upper_deg
                                 : approximate 95% CI for μ (Fisher 1993
                                   §4.4.4, large-sample normal approx)

    References
    ----------
    Mardia & Jupp 2000, "Directional Statistics".
    Fisher 1993, "Statistical Analysis of Circular Data".
    Berens 2009, "CircStat: A MATLAB Toolbox for Circular Statistics",
    J. Stat. Soft. 31(10).
    """
    a = np.asarray(angles_deg, dtype=float).ravel()
    a = a[np.isfinite(a)]
    n = int(a.size)
    out = {"n": n}
    if n < 2:
        # Nothing meaningful with <2 points.  Fill the schema with NaN
        # so downstream CSV consumers see the same columns regardless.
        for k in ("mean_direction_deg", "mean_resultant_length",
                 "circular_variance", "circular_std_deg",
                 "angular_deviation_deg", "median_deg",
                 "concentration_kappa", "rayleigh_z", "rayleigh_p",
                 "v_test_z", "v_test_p", "circular_skewness",
                 "circular_kurtosis", "ci95_lower_deg", "ci95_upper_deg"):
            out[k] = float("nan")
        return out

    rad = np.radians(a)
    C = float(np.mean(np.cos(rad)))
    S = float(np.mean(np.sin(rad)))
    R_bar = float(np.hypot(C, S))           # mean resultant length
    mu_rad = float(np.arctan2(S, C))         # mean direction (radians)
    mu_deg = float(np.degrees(mu_rad))
    # Standard CircStat convention: report direction on (-180°, +180°]
    if mu_deg <= -180.0: mu_deg += 360.0
    if mu_deg >   180.0: mu_deg -= 360.0

    # Dispersion measures
    circ_var = 1.0 - R_bar
    # √(-2·ln R̄) is undefined at R̄=0 (uniform), gigantic for tiny R̄.
    # Clamp to avoid log(0) screaming; report NaN for R̄ ≤ 0 instead.
    if R_bar > 0:
        circ_std_deg = float(np.degrees(np.sqrt(-2.0 * np.log(R_bar))))
    else:
        circ_std_deg = float("nan")
    ang_dev_deg = float(np.degrees(np.sqrt(2.0 * max(circ_var, 0.0))))

    # Circular median: angle θ̃ minimising Σ (π − |π − |θᵢ − θ̃||).
    # Evaluating the objective at every datum is O(n²) in time AND
    # memory if we do it with broadcasting (the 50k × 50k float64
    # array alone is 20 GB).  We instead:
    #   * cap CANDIDATES at 3000 (random subsample of the data)
    #   * cap SUMMAND points at 8000 (random subsample of the data)
    # which gives 24 million ops + ~190 MB temporary — fast enough,
    # and the median estimate from a 8000-point subsample is accurate
    # to a couple of degrees, well below other sources of noise here.
    _rng = np.random.default_rng(0)
    if n > 3000:
        cand = rad[_rng.choice(n, size=3000, replace=False)]
    else:
        cand = rad
    if n > 8000:
        ref = rad[_rng.choice(n, size=8000, replace=False)]
    else:
        ref = rad
    diff = np.abs(cand[:, None] - ref[None, :])
    diff = np.minimum(diff, 2.0 * np.pi - diff)        # circular distance
    obj = diff.sum(axis=1)
    median_rad = float(cand[int(np.argmin(obj))])
    median_deg = float(np.degrees(median_rad))
    if median_deg <= -180.0: median_deg += 360.0
    if median_deg >   180.0: median_deg -= 360.0

    # Concentration κ — Best & Fisher 1981 piecewise approximation,
    # with a small-n bias correction (Fisher 1993 eq. 4.41).
    if R_bar < 0.53:
        kappa = 2.0 * R_bar + R_bar ** 3 + 5.0 * R_bar ** 5 / 6.0
    elif R_bar < 0.85:
        kappa = -0.4 + 1.39 * R_bar + 0.43 / max(1.0 - R_bar, 1e-12)
    else:
        denom = max(R_bar ** 3 - 4.0 * R_bar ** 2 + 3.0 * R_bar, 1e-12)
        kappa = 1.0 / denom
    if n < 15:
        if kappa < 2.0:
            kappa = max(kappa - 2.0 / (n * kappa), 0.0)
        else:
            kappa = ((n - 1.0) ** 3) * kappa / (n ** 3 + n)

    # Rayleigh test for uniformity (Wilkie 1983 / Mardia & Jupp eq. 6.3.5).
    # We compute in LOG space so the result doesn't underflow to 0
    # when n is large (e.g. n=240k with R̄=0.08 → z≈1500 → exp(-z)
    # rounds to 0 in float64, which the user sees as a spurious
    # "p = 0").  The leading term is exp(-z); we still apply the
    # Mardia higher-order correction multiplicatively in log-space.
    R_total = n * R_bar
    z_ray = R_total ** 2 / n
    correction = (1.0 + (2.0 * z_ray - z_ray ** 2) / (4.0 * n)
                  - (24.0 * z_ray - 132.0 * z_ray ** 2
                     + 76.0 * z_ray ** 3 - 9.0 * z_ray ** 4)
                    / (288.0 * n ** 2))
    if correction <= 0:
        correction = 1.0   # higher-order correction overshot; ignore.
    log_p_ray = -z_ray + np.log(correction)
    # If log p < ~-700, exp underflows.  Convert to a tiny positive
    # number that survives float64 (1e-300) so downstream callers see
    # "very small" rather than zero, and formatters can render it as
    # "<1e-300".
    if log_p_ray < -700.0:
        p_ray = 1e-300
    else:
        p_ray = float(np.exp(log_p_ray))
    p_ray = float(np.clip(p_ray, 0.0, 1.0))

    # V-test against μ₀ = 0° ("are tracks preferentially going
    # straight ahead?").  V = R̄·cos(μ − μ₀); z = V·√(2n); one-tailed.
    mu0 = 0.0
    V = R_bar * np.cos(mu_rad - mu0)
    z_v = V * np.sqrt(2.0 * n)
    # One-tailed p via the standard normal survival function.  Use
    # scipy's norm.sf where available (numerically stable to ~p≈1e-300);
    # fall back to a math.erf-based computation otherwise, and floor at
    # 1e-300 so a huge z doesn't round to exactly 0.
    try:
        from scipy.stats import norm as _norm
        p_v = float(_norm.sf(z_v))
    except Exception:
        from math import erf
        p_v = float(0.5 * (1.0 - erf(z_v / np.sqrt(2.0))))
    if p_v == 0.0:
        p_v = 1e-300    # underflow sentinel
    p_v = float(np.clip(p_v, 0.0, 1.0))

    # Circular skewness and kurtosis (Mardia & Jupp §2.3).
    # b̄ = (1/n) Σ sin(2(θᵢ − μ))   ;   ā = (1/n) Σ cos(2(θᵢ − μ))
    b_bar = float(np.mean(np.sin(2.0 * (rad - mu_rad))))
    a_bar = float(np.mean(np.cos(2.0 * (rad - mu_rad))))
    sigma = max(1.0 - R_bar, 1e-12)
    skew = b_bar / (sigma ** 1.5)
    kurt = (a_bar - R_bar ** 4) / (sigma ** 2)

    # 95% CI for μ — large-sample normal approximation (Fisher 1993
    # eq. 4.46).  Only meaningful when R̄ is appreciable AND n ≥ ~15;
    # report NaN when the approximation breaks down.
    if R_bar >= 0.4 and n >= 15:
        sd_mu = np.sqrt((1.0 - a_bar) / (2.0 * n * R_bar ** 2))
        half = float(np.degrees(1.959964 * sd_mu))   # 1.96 σ
        lo = mu_deg - half
        hi = mu_deg + half
        # Keep both endpoints on (-180°, +180°] without wrapping the
        # interval ordering — supervisor will read this from the CSV.
        ci_lo, ci_hi = lo, hi
    else:
        ci_lo = float("nan")
        ci_hi = float("nan")

    out.update({
        "mean_direction_deg":     mu_deg,
        "mean_resultant_length":  R_bar,
        "circular_variance":      circ_var,
        "circular_std_deg":       circ_std_deg,
        "angular_deviation_deg":  ang_dev_deg,
        "median_deg":             median_deg,
        "concentration_kappa":    float(kappa),
        "rayleigh_z":             float(z_ray),
        "rayleigh_p":             p_ray,
        "v_test_z":               float(z_v),
        "v_test_p":               p_v,
        "circular_skewness":      float(skew),
        "circular_kurtosis":      float(kurt),
        "ci95_lower_deg":         float(ci_lo),
        "ci95_upper_deg":         float(ci_hi),
    })
    return out


_THEME_REQUIRED_KEYS = (
    "BG", "PNL", "TXT", "MUT", "GRD", "ACC",
    "HDR_BG", "HDR_TXT", "ZEBRA", "FONT", "ARROW",
    # legacy keys consumed by `compare_groups` & `_write_pdf_report`
    "BAR_FILL", "SIG",
)


def _theme_palette(theme: str) -> dict:
    """Return a colour palette matching the master figure theme.
    Centralised so the master figure, the circular-statistics PDF, and
    the comparison PDF all read from the same source of truth.

    The returned dict is GUARANTEED to contain every key in
    `_THEME_REQUIRED_KEYS` — if any caller starts using a new key,
    add it to the tuple and to every branch below, and the
    `_validate_palette` check at the bottom will catch a regression at
    module-import time rather than at PDF-render time.
    """
    t = (theme or "Dark").strip()
    if t == "Light":
        pal = {"BG":   "#ffffff", "PNL":  "#f6f8fa",
               "TXT":  "#24292f", "MUT":  "#57606a",
               "GRD":  "#d0d7de", "ACC":  "#0969da",
               "HDR_BG":"#1f2937", "HDR_TXT":"#ffffff",
               "ZEBRA":"#f3f4f6", "FONT": "sans-serif",
               "ARROW":"#d93636",
               "BAR_FILL":"#0969da", "SIG":"#d93636"}
    elif t == "Publication":
        pal = {"BG":   "#ffffff", "PNL":  "#ffffff",
               "TXT":  "#000000", "MUT":  "#444444",
               "GRD":  "#cccccc", "ACC":  "#333333",
               "HDR_BG":"#000000", "HDR_TXT":"#ffffff",
               "ZEBRA":"#f2f2f2", "FONT": "serif",
               "ARROW":"#000000",
               "BAR_FILL":"#333333", "SIG":"#000000"}
    elif t == "AMOLED":
        # Pure-black backgrounds for OLED displays.  Mirrors Dark
        # otherwise so the figures are recognisable as the same FIREFLY
        # output.  PNL nudged to #0a0a0a so card-style panels still
        # read as cards against the BG.
        pal = {"BG":   "#000000", "PNL":  "#0a0a0a",
               "TXT":  "#e6edf3", "MUT":  "#9da7b1",
               "GRD":  "#30363d", "ACC":  "#58a6ff",
               "HDR_BG":"#141414", "HDR_TXT":"#e6edf3",
               "ZEBRA":"#050505", "FONT": "monospace",
               "ARROW":"#ff7b72",
               "BAR_FILL":"#58a6ff", "SIG":"#ff7b72"}
    else:
        # Dark (default).
        pal = {"BG":   "#0d1117", "PNL":  "#161b22",
               "TXT":  "#e6edf3", "MUT":  "#9da7b1",
               "GRD":  "#30363d", "ACC":  "#58a6ff",
               "HDR_BG":"#21262d", "HDR_TXT":"#e6edf3",
               "ZEBRA":"#1c2128", "FONT": "monospace",
               "ARROW":"#ff7b72",
               "BAR_FILL":"#58a6ff", "SIG":"#ff7b72"}
    # Belt-and-braces: if a caller (or future edit) ever accesses a key
    # we forgot to include, return a sensible TXT fallback rather than
    # crashing with a KeyError mid-render.  We do this via a small
    # dict subclass so `pal[<missing>]` works as if `pal.get(<missing>,
    # pal["TXT"])` were called.
    class _PalDict(dict):
        __slots__ = ()
        def __missing__(self, key):
            return self.get("TXT", "#000000")
    return _PalDict(pal)


def save_circular_statistics_pdf(angles_deg, stats, *, pdf_path,
                                  file_label="", fig_theme="Dark",
                                  circ_lin_result=None):
    """Render a single-page A4-portrait PDF report summarising the
    circular statistics in `stats` (as produced by
    `compute_circular_statistics`) alongside a small polar histogram of
    the underlying angle distribution.

    Designed to be supervisor-facing: stat names match MATLAB CircStat,
    each value is annotated with a one-line plain-English meaning, and
    the polar plot orients 0° at the top with positive angles sweeping
    counter-clockwise (the convention `compute_turning_angles` uses).

    Parameters
    ----------
    angles_deg : 1-D array of turning angles in degrees (signed, on
                 (-180°, +180°]).  Used only for the polar histogram.
    stats      : dict returned by `compute_circular_statistics`.
    pdf_path   : where to write the PDF.
    file_label : appears in the page header (typically the analysis stem).
    fig_theme  : "Dark" | "Light" | "Publication" — palette to match the
                 master figure renderer.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pal = _theme_palette(fig_theme)

    a = np.asarray(angles_deg, dtype=float).ravel()
    a = a[np.isfinite(a)]

    # Helper: render NaN as an em-dash so the PDF doesn't look broken
    # when a stat couldn't be computed (small-n or R̄ ≈ 0 cases).
    # Also collapse the 1e-300 underflow sentinel produced by the
    # log-space p-value computations into a human-readable "<1e-300"
    # — otherwise the supervisor sees "1e-300" and wonders why so
    # many tests give exactly that value.
    def _fmt(x, prec=4):
        try:
            if x is None: return "—"
            xf = float(x)
            if np.isnan(xf): return "—"
            if xf > 0.0 and xf <= 1e-300:
                return "<1e-300"
            return f"{xf:.{prec}g}"
        except Exception:
            return str(x)

    # One-line plain-English gloss per statistic.  Order matches the
    # CSV column order so the table reads top-to-bottom like the CSV.
    rows = [
        ("n",                          "Sample size", "count",
         f"{int(stats.get('n', 0)):,}"),
        ("mean_direction_deg",         "Mean direction μ", "deg",
         _fmt(stats.get("mean_direction_deg"), 4)),
        ("mean_resultant_length",      "Mean resultant length R̄  (0 = uniform, 1 = aligned)",
         "—",
         _fmt(stats.get("mean_resultant_length"), 4)),
        ("circular_variance",          "Circular variance  1 − R̄", "—",
         _fmt(stats.get("circular_variance"), 4)),
        ("circular_std_deg",           "Circular standard deviation  √(−2·ln R̄)",
         "deg",
         _fmt(stats.get("circular_std_deg"), 4)),
        ("angular_deviation_deg",      "Angular deviation  s₀ = √(2·(1−R̄))",
         "deg",
         _fmt(stats.get("angular_deviation_deg"), 4)),
        ("median_deg",                 "Circular median", "deg",
         _fmt(stats.get("median_deg"), 4)),
        ("concentration_kappa",        "Von Mises concentration κ  (Best & Fisher 1981)",
         "—",
         _fmt(stats.get("concentration_kappa"), 4)),
        ("rayleigh_z",                 "Rayleigh test statistic  z = n·R̄²", "—",
         _fmt(stats.get("rayleigh_z"), 4)),
        ("rayleigh_p",                 "Rayleigh test p-value  (uniformity)", "—",
         _fmt(stats.get("rayleigh_p"), 3)),
        ("v_test_z",                   "V-test statistic against μ₀ = 0°", "—",
         _fmt(stats.get("v_test_z"), 4)),
        ("v_test_p",                   "V-test p-value  (preferred direction)", "—",
         _fmt(stats.get("v_test_p"), 3)),
        ("circular_skewness",          "Circular skewness  (Mardia & Jupp §2.3)", "—",
         _fmt(stats.get("circular_skewness"), 4)),
        ("circular_kurtosis",          "Circular kurtosis  (Mardia & Jupp §2.3)", "—",
         _fmt(stats.get("circular_kurtosis"), 4)),
        ("ci95_lower_deg",             "95% CI lower bound for μ  (large-sample)", "deg",
         _fmt(stats.get("ci95_lower_deg"), 4)),
        ("ci95_upper_deg",             "95% CI upper bound for μ  (large-sample)", "deg",
         _fmt(stats.get("ci95_upper_deg"), 4)),
    ]
    # Circ-lin correlation rows — optional; only present when the
    # caller passed a `circ_lin_result` (computed from per-track
    # (mean_angle, D) pairs).  Three rows: r, χ²(2), p, n.  Treated
    # as a single stats block so it can be excluded silently when
    # the caller has no D data (e.g. external-CSV input path).
    if circ_lin_result:
        rows.extend([
            ("circ_lin_angle_vs_D_r",
             "Circ-lin correlation r — turning bias vs D", "—",
             _fmt(circ_lin_result.get("r"), 4)),
            ("circ_lin_angle_vs_D_chi2",
             "Circ-lin χ²(2) test statistic  (n·r²)", "—",
             _fmt(circ_lin_result.get("test_stat"), 4)),
            ("circ_lin_angle_vs_D_p",
             "Circ-lin correlation p-value", "—",
             _fmt(circ_lin_result.get("p"), 3)),
            ("circ_lin_angle_vs_D_n",
             "Circ-lin sample size  (tracks with ≥ 3 frames + D)",
             "count",
             f"{int(circ_lin_result.get('n', 0)):,}"
             if circ_lin_result.get("n") is not None else "—"),
        ])

    # ── rcParams snapshot ──────────────────────────────────────────────
    # plt.rcParams persists across figures in the same process — the
    # master figure renderer might have left things on the Dark palette
    # (text.color = #e6edf3 etc.).  Snapshot then force everything to
    # OUR palette so we can't accidentally pick up someone else's
    # colours.  Restored at the end.
    _rc_keys = ("text.color", "axes.labelcolor", "axes.edgecolor",
                "xtick.color", "ytick.color", "axes.facecolor",
                "axes.titlecolor", "figure.facecolor", "grid.color",
                "font.family")
    _rc_save = {k: plt.rcParams.get(k) for k in _rc_keys}
    plt.rcParams.update({
        "text.color":       pal["TXT"],
        "axes.labelcolor":  pal["TXT"],
        "axes.edgecolor":   pal["GRD"],
        "xtick.color":      pal["TXT"],
        "ytick.color":      pal["TXT"],
        "axes.facecolor":   pal["PNL"],
        "axes.titlecolor":  pal["TXT"],
        "figure.facecolor": pal["BG"],
        "grid.color":       pal["GRD"],
        "font.family":      pal["FONT"],
    })

    try:
        # ── Layout (A4 portrait, all coords in figure-fraction) ─────────
        #
        # Vertical bands, top → bottom:
        #   y 0.94 – 0.98  : header bar (title + n)
        #   y 0.89 – 0.93  : file label
        #   y 0.61 – 0.86  : polar  |  interpretation banner
        #   y 0.54 – 0.58  : "Statistics" section title
        #   y 0.12 – 0.52  : statistics table
        #   y 0.06 – 0.10  : sign-convention footer (3 short lines)
        #   y 0.02 – 0.04  : references footer
        #
        # The earlier layout placed the Statistics title with
        # `transform=ax_tbl.transAxes` at y=1.04 which sits at about
        # figure-y 0.53 — directly underneath the polar's "±180°" tick.
        # Moving it to its own fig.text at a fixed y resolves the overlap.
        # The footer used to be at y=0.04 which collided with the
        # table's bottom row at y=0.05; both footers now live below
        # y=0.10 with the table topping at y=0.52.
        fig = plt.figure(figsize=(8.27, 11.69), facecolor=pal["BG"])

        # Header (full width)
        ax_hdr = fig.add_axes([0.07, 0.94, 0.86, 0.04])
        ax_hdr.axis("off")
        title = "Circular Statistics Report"
        ax_hdr.text(0.0, 0.5, title, fontsize=16, fontweight="bold",
                    va="center", ha="left", color=pal["TXT"])
        n_val = int(stats.get("n", 0))
        ax_hdr.text(1.0, 0.5,
                    f"n = {n_val:,} turning angles",
                    fontsize=11, color=pal["MUT"], va="center", ha="right")
        if file_label:
            # File label on its own dedicated row so it can't fight the
            # polar plot's "0°" tick label below.
            fig.text(0.07, 0.91, file_label, fontsize=10,
                     color=pal["MUT"], va="top", ha="left",
                     family=pal["FONT"])

        # Polar histogram (left side of middle band).
        # Convention matched to the master figure's Radial-Distribution
        # panel (see sax "O" in make_figure): 0° at the top, positive
        # angles sweep CLOCKWISE so they appear on the right hemisphere.
        # Signed angles on (-180°, +180°] are first wrapped to [0, 2π)
        # before histogramming — matplotlib's polar bar() silently drops
        # bars at negative theta when set_theta_direction(-1) is active.
        ax_polar = fig.add_axes([0.08, 0.61, 0.36, 0.25], projection="polar")
        ax_polar.set_facecolor(pal["PNL"])
        if a.size >= 10:
            nbins = 36
            angles_rad = np.mod(np.deg2rad(a), 2.0 * np.pi)
            bins  = np.linspace(0.0, 2.0 * np.pi, nbins + 1)
            counts, edges = np.histogram(angles_rad, bins=bins)
            widths  = np.diff(edges)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax_polar.set_theta_zero_location("N")
            ax_polar.set_theta_direction(-1)  # CW positive — match master fig
            ax_polar.bar(centers, counts, width=widths * 0.95,
                         align="center", color=pal["ACC"],
                         edgecolor=pal["PNL"], linewidth=0.4, alpha=0.92)
            mu = stats.get("mean_direction_deg")
            if mu is not None and not (isinstance(mu, float) and np.isnan(mu)):
                r_max = float(counts.max()) if counts.size else 1.0
                # Wrap signed μ into [0, 2π) so the arrow lands at the
                # same place the bar histogram does.
                mu_rad = np.mod(np.deg2rad(mu), 2.0 * np.pi)
                ax_polar.annotate("",
                    xy=(mu_rad, r_max * 0.95),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->",
                                    color=pal["ARROW"], lw=2.0))
            # Show signed-angle labels at positive-angle slot positions
            # so the visual reads "+45° upper-right, -45° upper-left",
            # exactly like the master figure.
            ax_polar.set_xticks(np.deg2rad(
                [0, 45, 90, 135, 180, 225, 270, 315]))
            ax_polar.set_xticklabels(
                ["0°", "+45°", "+90°", "+135°", "±180°",
                 "−135°", "−90°", "−45°"], fontsize=8)
            ax_polar.set_yticklabels([])
            ax_polar.tick_params(colors=pal["TXT"], labelsize=8)
            ax_polar.grid(True, ls=":", alpha=0.4)
            # NB: deliberately no `set_title` here — matplotlib places
            # the polar title above the axes box (offset by `pad`), and
            # at this layout that overlaps the file-label rendered in
            # the header area.  The page header already identifies the
            # report, and the footer covers the sign convention, so a
            # title on the polar would be redundant anyway.
        else:
            ax_polar.axis("off")
            ax_polar.text(0.5, 0.5, "Too few angles for histogram",
                          transform=ax_polar.transAxes,
                          ha="center", va="center", color=pal["MUT"],
                          fontsize=10)

        # Interpretation banner (right side of middle band)
        ax_intr = fig.add_axes([0.48, 0.61, 0.46, 0.25])
        ax_intr.axis("off")
        R = stats.get("mean_resultant_length")
        p = stats.get("rayleigh_p")
        if R is None or (isinstance(R, float) and np.isnan(R)):
            interp = "Distribution: insufficient data."
        elif R < 0.10:
            interp = ("Distribution is consistent with uniform circular "
                      "scatter — no preferred turning direction is "
                      "evident.  Typical of free 2-D diffusion.")
        elif R < 0.30:
            interp = ("Weak directional bias.  Most steps are close "
                      "to uniform, but a slight tendency toward "
                      f"{stats.get('mean_direction_deg', 0):.0f}° is "
                      "present.")
        elif R < 0.60:
            interp = ("Moderate directional bias toward "
                      f"{stats.get('mean_direction_deg', 0):.0f}°.  "
                      "Consider whether this reflects biology (e.g. "
                      "transport along a cytoskeletal track) or an "
                      "artefact (uncorrected drift, anisotropic ROI).")
        else:
            interp = ("Strong directional bias toward "
                      f"{stats.get('mean_direction_deg', 0):.0f}°.  "
                      "Verify the drift correction and ROI geometry "
                      "before biological interpretation.")
        if p is not None and not (isinstance(p, float) and np.isnan(p)):
            if p < 0.001:
                verdict = ("Rayleigh test strongly rejects uniformity "
                           f"(p = {p:.3g}).")
            elif p < 0.05:
                verdict = ("Rayleigh test rejects uniformity at α = "
                           f"0.05 (p = {p:.3g}).")
            else:
                verdict = ("Rayleigh test does NOT reject uniformity "
                           f"(p = {p:.3g}).")
            interp = interp + "\n\n" + verdict
        ax_intr.text(0.0, 1.0, "Interpretation",
                     fontsize=12, fontweight="bold", va="top",
                     color=pal["TXT"])
        ax_intr.text(0.0, 0.9, interp, fontsize=10, va="top",
                     wrap=True, color=pal["TXT"])

        # Section title — placed in FIGURE coords so its vertical
        # position is decoupled from the table's bbox and can't
        # collide with the polar's bottom ticks above.
        fig.text(0.07, 0.555, "Statistics  (MATLAB CircStat conventions)",
                 fontsize=12, fontweight="bold", va="bottom",
                 ha="left", color=pal["TXT"])
        # Statistics table — pinned with a clear gap above (title) and
        # below (footer block).  Bottom edge y=0.12 leaves room for two
        # footer lines without collision.
        ax_tbl = fig.add_axes([0.07, 0.12, 0.88, 0.40])
        ax_tbl.axis("off")

        cell_text, row_labels = [], []
        for key, gloss, unit, val in rows:
            unit_s = "" if unit in ("", "—") else f"  ({unit})"
            cell_text.append([f"{gloss}", f"{val}{unit_s}"])
            row_labels.append(key)
        tbl = ax_tbl.table(cellText=cell_text,
                           rowLabels=row_labels,
                           colLabels=["Description", "Value"],
                           cellLoc="left", rowLoc="left",
                           colLoc="left",
                           colWidths=[0.62, 0.28],
                           bbox=[0.20, 0.0, 0.80, 1.0])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.0)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_linewidth(0.5)
            cell.set_edgecolor(pal["GRD"])
            if r == 0:                       # column header row
                cell.set_facecolor(pal["HDR_BG"])
                cell.set_text_props(color=pal["HDR_TXT"], fontweight="bold")
            else:
                # Zebra-stripe data rows.  Use theme PNL for the
                # "darker" stripes and ZEBRA for the lighter ones.
                cell.set_facecolor(pal["ZEBRA"] if r % 2 == 0 else pal["PNL"])
                if c == -1:                  # row-label column
                    cell.set_text_props(family="monospace", fontsize=8.0,
                                        color=pal["MUT"])
                else:
                    cell.set_text_props(color=pal["TXT"])

        # Footer — explicit short lines instead of `wrap=True`, because
        # matplotlib's fig.text wrap only kicks in when the text would
        # exceed a containing artist's width, NOT the figure width, so
        # long strings just run off the right edge of the PDF (which is
        # what was happening to the References line).  Breaking into
        # pre-wrapped lines side-steps that entirely.
        _foot_kw = dict(fontsize=7, color=pal["MUT"], ha="left",
                        va="bottom", family=pal["FONT"])
        sign_lines = [
            "Sign convention: turning angles are SIGNED on (−180°, +180°].",
            "0° = straight ahead.  +θ = left turn (CCW).  −θ = right "
            "turn (CW).  ±180° = full reversal.",
            "Unsigned 0–360° equivalent: u = θ if θ ≥ 0, else θ + 360 "
            "(so −90° ≡ 270°, +90° ≡ 90°).",
        ]
        ref_lines = [
            "References:",
            "  Mardia & Jupp 2000 — Directional Statistics.",
            "  Fisher 1993 — Statistical Analysis of Circular Data.",
            "  Berens 2009 — CircStat: A MATLAB Toolbox for Circular "
            "Statistics, J. Stat. Soft. 31(10).",
        ]
        y = 0.095
        for line in sign_lines:
            fig.text(0.07, y, line, **_foot_kw)
            y -= 0.014
        y -= 0.006
        for line in ref_lines:
            fig.text(0.07, y, line, **_foot_kw)
            y -= 0.014

        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, facecolor=pal["BG"])
        plt.close(fig)
    finally:
        # Restore rcParams so we don't bleed our palette into whatever
        # plot the caller draws next.
        plt.rcParams.update(_rc_save)


def _circ_watson_williams(samples_deg):
    """k-sample Watson-Williams F-test for equality of mean directions
    across k≥2 circular samples (Mardia & Jupp 2000 §6.4.2).  This is
    the circular analogue of one-way ANOVA: H₀ = all groups share a
    common mean direction.

    Parameters
    ----------
    samples_deg : list of 1-D angle arrays (degrees, any range)

    Returns
    -------
    None if fewer than 2 valid samples, else dict with:
      F, df1, df2, p           — test statistic + degrees of freedom + p
      kappa_pooled, R_bar_pooled
      valid                     — True iff κ̂ ≥ 2 and R̄ ≥ 0.45 (the test
                                  assumes concentrated von Mises samples;
                                  flag the result if not).
      n_per_group, n_total, k
    """
    rad = [np.radians(np.asarray(s, dtype=float).ravel())
           for s in samples_deg]
    rad = [r[np.isfinite(r)] for r in rad]
    rad = [r for r in rad if r.size >= 2]
    k = len(rad)
    if k < 2:
        return None
    n_per = [int(r.size) for r in rad]
    N = int(sum(n_per))
    Ci = np.array([float(np.cos(r).sum()) for r in rad])
    Si = np.array([float(np.sin(r).sum()) for r in rad])
    Ri = np.hypot(Ci, Si)
    Cp = float(Ci.sum()); Sp = float(Si.sum())
    Rp = float(np.hypot(Cp, Sp))
    R_bar = Rp / N
    # Pooled concentration (Best & Fisher 1981).
    if R_bar < 0.53:
        kappa = 2.0 * R_bar + R_bar ** 3 + 5.0 * R_bar ** 5 / 6.0
    elif R_bar < 0.85:
        kappa = -0.4 + 1.39 * R_bar + 0.43 / max(1.0 - R_bar, 1e-12)
    else:
        denom = max(R_bar ** 3 - 4.0 * R_bar ** 2 + 3.0 * R_bar, 1e-12)
        kappa = 1.0 / denom
    # Stephens 1972 K correction (≈1 when κ is large; sharper at low κ).
    K = 1.0 + 3.0 / (8.0 * kappa) if kappa > 0 else 1.0
    sumR = float(Ri.sum())
    denom_f = (k - 1) * (N - sumR)
    if denom_f <= 0:
        return None
    F = K * (N - k) * (sumR - Rp) / denom_f
    df1, df2 = int(k - 1), int(N - k)
    try:
        from scipy.stats import f as _f_dist
        # Use logsf → exp so we get a meaningful tiny p instead of a
        # rounded-to-zero float when F is huge (which is normal with
        # 100k+ angles per group).  logsf returns log(1 - cdf) with
        # log-space stability.
        log_p = float(_f_dist.logsf(F, df1, df2))
        p = 1e-300 if log_p < -700.0 else float(np.exp(log_p))
    except Exception:
        p = float("nan")
    return {
        "F": float(F), "df1": df1, "df2": df2, "p": p,
        "kappa_pooled": float(kappa),
        "R_bar_pooled": float(R_bar),
        "valid": bool(kappa >= 2.0 and R_bar >= 0.45),
        "n_per_group": n_per, "n_total": N, "k": int(k),
    }


def _circ_mardia_watson_wheeler(samples_deg):
    """Mardia-Watson-Wheeler (uniform-scores) non-parametric k-sample
    test for equal CIRCULAR DISTRIBUTIONS across k≥2 groups (Mardia &
    Jupp 2000 §7.6.1).  Unlike Watson-Williams it makes no assumption
    about concentration, so it's the safe fallback when κ < 2 or when
    you suspect groups differ in spread rather than only in mean
    direction.

    Returns None if fewer than 2 valid samples, else dict with:
      W, df, p, n_per_group, n_total, k
    """
    rad = [np.radians(np.asarray(s, dtype=float).ravel())
           for s in samples_deg]
    rad = [r[np.isfinite(r)] for r in rad]
    rad = [r for r in rad if r.size >= 1]
    k = len(rad)
    if k < 2:
        return None
    pooled = np.concatenate(rad)
    N = int(pooled.size)
    try:
        from scipy.stats import rankdata, chi2
    except Exception:
        return None
    ranks = rankdata(pooled, method="average")
    # Convert ranks → uniform circular scores in [0, 2π).
    beta = 2.0 * np.pi * ranks / N
    # Sample-wise C/S sums, then W = 2 · Σ (C² + S²) / n_j.
    W_stat = 0.0
    cursor = 0
    for r in rad:
        n_j = int(r.size)
        end = cursor + n_j
        b = beta[cursor:end]
        Cj = float(np.cos(b).sum())
        Sj = float(np.sin(b).sum())
        W_stat += (Cj * Cj + Sj * Sj) / n_j
        cursor = end
    W = 2.0 * W_stat
    df = int(2 * (k - 1))
    try:
        # logsf for numerical stability — chi2.sf(3.4e3, 2) underflows
        # to 0.0 in float64 but chi2.logsf returns the actual log p.
        log_p = float(chi2.logsf(W, df))
        p = 1e-300 if log_p < -700.0 else float(np.exp(log_p))
    except Exception:
        p = float("nan")
    return {
        "W": float(W), "df": df, "p": p,
        "n_per_group": [int(r.size) for r in rad],
        "n_total": N, "k": int(k),
    }


def _circ_wallraff_ktest(samples_deg):
    """Wallraff k-sample test for equality of circular concentrations.

    H₀ = all samples share the same concentration κ.  Implementation
    follows Mardia & Jupp (2000) §7.5.5: convert each angle to its
    deviation from its own sample's mean direction (mapped to [0, π]),
    then run a rank-sum test on those deviations across groups.

    For k = 2 we use the Mann-Whitney U test; for k > 2 we use the
    Kruskal-Wallis H test.  Returns None if fewer than 2 valid samples.

    Returned dict:
      H or U   : test statistic (key name depends on k)
      df       : degrees of freedom (Kruskal-Wallis only)
      p        : p-value
      n_per_group, n_total, k
    """
    rad = [np.radians(np.asarray(s, dtype=float).ravel())
           for s in samples_deg]
    rad = [r[np.isfinite(r)] for r in rad]
    rad = [r for r in rad if r.size >= 2]
    k = len(rad)
    if k < 2:
        return None
    # Per-sample angular deviation from its OWN mean direction,
    # mapped to [0, π] (the circular distance).
    deviations = []
    for r in rad:
        mu = np.arctan2(np.sin(r).mean(), np.cos(r).mean())
        d  = np.abs(r - mu)
        d  = np.minimum(d, 2.0 * np.pi - d)
        deviations.append(d)
    n_per = [int(d.size) for d in deviations]
    try:
        if k == 2:
            from scipy.stats import mannwhitneyu
            stat, p = mannwhitneyu(deviations[0], deviations[1],
                                   alternative="two-sided")
            return {
                "U": float(stat), "p": float(p), "k": 2,
                "n_per_group": n_per, "n_total": int(sum(n_per)),
            }
        else:
            from scipy.stats import kruskal
            stat, p = kruskal(*deviations)
            return {
                "H": float(stat), "df": int(k - 1),
                "p": float(p), "k": int(k),
                "n_per_group": n_per, "n_total": int(sum(n_per)),
            }
    except Exception:
        return None


def _circ_kuiper_two_sample(a_deg, b_deg):
    """Kuiper two-sample test for equality of circular distributions.

    Non-parametric, distribution-free analogue of the Kolmogorov-Smirnov
    test, adapted for circular data.  Sensitive to differences anywhere
    in the distribution (not just shifts in mean), and unlike the KS
    statistic the Kuiper statistic V = D⁺ + D⁻ is invariant to the
    choice of origin on the circle — a property that matters because
    "where you put 0°" is arbitrary for circular data.

    Returns None if either sample is < 2 elements, else dict:
      V       : Kuiper statistic
      p       : asymptotic p-value (Stephens 1965 series approximation)
      n1, n2  : sample sizes
    """
    a = np.sort(np.mod(np.radians(np.asarray(a_deg, dtype=float).ravel()),
                       2.0 * np.pi))
    b = np.sort(np.mod(np.radians(np.asarray(b_deg, dtype=float).ravel()),
                       2.0 * np.pi))
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    n1, n2 = int(a.size), int(b.size)
    if n1 < 2 or n2 < 2:
        return None

    # Empirical CDFs evaluated at every observation in the combined
    # sample.  V = max(F1 - F2) + max(F2 - F1).
    combined = np.sort(np.concatenate([a, b]))
    F1 = np.searchsorted(a, combined, side="right") / n1
    F2 = np.searchsorted(b, combined, side="right") / n2
    D_plus  = float((F1 - F2).max())
    D_minus = float((F2 - F1).max())
    V = D_plus + D_minus

    # Stephens (1965) asymptotic p-value: λ = (√n_eff + 0.155 + 0.24/√n_eff)·V.
    n_eff = n1 * n2 / (n1 + n2)
    lam = (np.sqrt(n_eff) + 0.155 + 0.24 / np.sqrt(n_eff)) * V
    if lam <= 0:
        p = 1.0
    else:
        # Convergent series in j; cap at j=100 (terms decay
        # exponentially in j²).
        s_terms = 0.0
        l2 = lam * lam
        for j in range(1, 101):
            j2 = j * j
            term = 2.0 * (4.0 * j2 * l2 - 1.0) * np.exp(-2.0 * j2 * l2)
            s_terms += term
            if abs(term) < 1e-18:
                break
        p = float(np.clip(s_terms, 0.0, 1.0))
    if p > 0.0 and p <= 1e-300:
        p = 1e-300
    return {
        "V": float(V), "p": float(p),
        "n1": n1, "n2": n2,
    }


def _circ_lin_correlation(theta_deg, x):
    """Circular-linear correlation (Mardia 1976; Mardia & Jupp 2000
    §6.5.1).

    Tests whether a circular variable θ is associated with a linear
    variable x.  Compute the three Pearson correlations
        r_xc = corr(x, cos θ),  r_xs = corr(x, sin θ),  r_cs = corr(cos θ, sin θ)
    and combine them into the circular-linear coefficient

        R² = (r_xc² + r_xs² − 2·r_xc·r_xs·r_cs) / (1 − r_cs²)

    R ∈ [0, 1] (analogous to a Pearson |r|).  Under H₀ of independence
    and large n, n·R² ~ χ²(2), giving a usable p-value.

    Returns None if n < 3 or the data are degenerate; else dict:
      r, r2          : coefficient and its square
      test_stat      : n · r²
      df, p          : χ²(2) p-value
      n              : effective sample size after finite-mask
    """
    theta = np.asarray(theta_deg, dtype=float).ravel()
    x     = np.asarray(x,         dtype=float).ravel()
    if theta.size != x.size:
        return None
    mask = np.isfinite(theta) & np.isfinite(x)
    theta = theta[mask]; x = x[mask]
    n = int(theta.size)
    if n < 3:
        return None
    rad = np.radians(theta)
    c = np.cos(rad); s = np.sin(rad)
    # Need non-zero variance in x AND in c/s for the correlations to
    # exist.  If all angles are identical (or all x identical), bail.
    if np.std(x) == 0 or np.std(c) == 0 or np.std(s) == 0:
        return None
    rxc = float(np.corrcoef(x, c)[0, 1])
    rxs = float(np.corrcoef(x, s)[0, 1])
    rcs = float(np.corrcoef(c, s)[0, 1])
    denom = 1.0 - rcs ** 2
    if abs(denom) < 1e-12:
        return None
    r2 = (rxc ** 2 + rxs ** 2 - 2.0 * rxc * rxs * rcs) / denom
    r2 = float(np.clip(r2, 0.0, 1.0))
    test_stat = n * r2
    try:
        from scipy.stats import chi2
        log_p = float(chi2.logsf(test_stat, 2))
        p = 1e-300 if log_p < -700.0 else float(np.exp(log_p))
    except Exception:
        p = float("nan")
    return {
        "r": float(np.sqrt(r2)), "r2": r2,
        "test_stat": float(test_stat), "df": 2,
        "p": float(p), "n": n,
    }


def compute_per_track_mean_angle(tracks):
    """For each track in `tracks` with ≥ 3 localisations, compute the
    circular mean of its signed turning angles (degrees on
    (-180°, +180°]).  Returns a list of (particle_id, mean_angle_deg).

    Used to build (angle, D) pairs for the circular-linear correlation
    between a track's turning bias and its diffusion coefficient.
    """
    if len(tracks) < 3:
        return []
    srt = (tracks.reset_index(drop=True)
                 .sort_values(["particle", "frame"], kind="stable"))
    pid_arr = srt["particle"].to_numpy()
    xy_arr  = srt[["x", "y"]].to_numpy()
    steps = np.diff(xy_arr, axis=0)
    same_step = (pid_arr[1:] == pid_arr[:-1])
    if len(steps) < 2:
        return []
    v1 = steps[:-1]; v2 = steps[1:]
    both_in_track = same_step[:-1] & same_step[1:]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot   = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    valid = both_in_track & (norm1 > 0) & (norm2 > 0)
    if not valid.any():
        return []
    angles = np.arctan2(cross[valid], dot[valid])    # radians
    # The middle row of each (i, i+1, i+2) triple is pid_arr[i+1].
    pid_at_turn = pid_arr[1:-1][valid]
    # Bucket angles by particle and compute the circular mean.
    out = []
    for pid in np.unique(pid_at_turn):
        sel = (pid_at_turn == pid)
        rad = angles[sel]
        if rad.size == 0:
            continue
        mu = np.degrees(np.arctan2(np.sin(rad).mean(),
                                   np.cos(rad).mean()))
        out.append((int(pid), float(mu)))
    return out


def _watson_williams_mu_per_replicate(mu_lists_per_group):
    """Watson-Williams F-test on per-replicate mean directions.

    Treats each replicate's mean direction μ_ij as a single circular
    observation (not the underlying angles).  This is the supervisor-
    facing way to compare directionality between groups: the n is
    the number of REPLICATES, not the number of pooled localisations,
    so the test isn't inflated by huge per-file angle counts.

    Parameters
    ----------
    mu_lists_per_group : list aligned with the groups, each entry is
        a 1-D array/list of per-replicate mean directions in DEGREES
        (signed, (-180°, +180°]).

    Returns dict matching the shape `_circ_watson_williams` already
    uses (F, df1, df2, p, valid, ...), or None if fewer than 2 groups
    have ≥ 2 replicates each.
    """
    # _circ_watson_williams already does k-sample WW on a list of
    # angle arrays — pass the per-replicate μ values in as samples.
    samples = [np.asarray(arr, dtype=float).ravel()
               for arr in mu_lists_per_group]
    samples = [a[np.isfinite(a)] for a in samples]
    if sum(1 for a in samples if a.size >= 2) < 2:
        return None
    return _circ_watson_williams(samples)


def compute_circular_comparison_tests(groups, *, track_angle_d_pairs=None,
                                       per_replicate_angles=None):
    """Run all the standard 'do these circular samples differ?' tests on
    a list of labelled groups.

    Parameters
    ----------
    groups : list of (label, angles_deg_array)
        One entry per comparison group; the array is the pooled
        turning angles across all replicates in that group.
    track_angle_d_pairs : optional list aligned with `groups`
        Each element is a 2-tuple of arrays (per_track_mean_angle_deg,
        per_track_D_um2_s).  Used to compute the per-group circular-
        linear correlation between a track's average turning bias and
        its diffusion coefficient.  Pass None to skip the correlation.

    Returns
    -------
    dict with keys:
      omnibus_ww   : Watson-Williams F-test (equal mean directions)
      omnibus_mww  : Mardia-Watson-Wheeler W-test (equal distributions)
      omnibus_wallraff
                   : Wallraff k-sample test (equal concentrations);
                     directly addresses "is one group more tightly
                     clustered than the other?".
      pairwise     : list, one entry per (i, j) with i<j, each with
                     keys label_a, label_b, ww, mww, wallraff, kuiper
                     (Kuiper two-sample test for equal distributions).
      circ_lin_per_group
                   : list aligned with `groups`, dict per group with
                     keys label and result (the _circ_lin_correlation
                     dict, or None if not enough data).  Only populated
                     when track_angle_d_pairs is provided.
    """
    labels = [g[0] for g in groups]
    samples = [g[1] for g in groups]
    out = {
        "omnibus_ww":       _circ_watson_williams(samples),
        "omnibus_mww":      _circ_mardia_watson_wheeler(samples),
        "omnibus_wallraff": _circ_wallraff_ktest(samples),
        "pairwise": [],
        "circ_lin_per_group": [],
        # Per-replicate tests: see `per_replicate_angles` arg below.
        # Populated when the caller provides per-replicate angle arrays;
        # otherwise None so consumers can detect "not computed".
        "per_replicate_kappa_test": None,
        "per_replicate_rbar_test":  None,
        "per_replicate_mu_ww":      None,
        "per_replicate_scalars":    None,
    }
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            out["pairwise"].append({
                "label_a": labels[i],
                "label_b": labels[j],
                "ww":       _circ_watson_williams([samples[i], samples[j]]),
                "mww":      _circ_mardia_watson_wheeler(
                    [samples[i], samples[j]]),
                "wallraff": _circ_wallraff_ktest(
                    [samples[i], samples[j]]),
                "kuiper":   _circ_kuiper_two_sample(samples[i],
                                                    samples[j]),
            })
    if track_angle_d_pairs is not None:
        for label, pair in zip(labels, track_angle_d_pairs):
            theta, x = pair
            out["circ_lin_per_group"].append({
                "label":  label,
                "result": _circ_lin_correlation(theta, x),
            })

    # ── Per-replicate tests ──────────────────────────────────────────
    # Treats each replicate as ONE data point (its own κ, R̄, μ),
    # producing a defensible Welch's t-test on κ + R̄ (linear scalars)
    # and a Watson-Williams F-test on μ (a circular quantity).  This
    # is the right framing when the user has e.g. 5 vs 3 movies and
    # wants stats that respect the biological replicate count, not the
    # inflated angle-count produced by pooling.
    if per_replicate_angles is not None:
        per_kappa  = []      # list-per-group of replicate κ values
        per_rbar   = []      #          "          "        R̄
        per_mu     = []      #          "          "        μ (deg)
        per_n_reps = []
        scalars_per_group = []
        for label in labels:
            arrs = per_replicate_angles.get(label, [])
            kappas, rbars, mus = [], [], []
            for arr in arrs:
                a = np.asarray(arr, dtype=float).ravel()
                a = a[np.isfinite(a)]
                if a.size < 2:
                    continue
                cs = compute_circular_statistics(a)
                if cs is None:
                    continue
                k_val  = cs.get("concentration_kappa")
                r_val  = cs.get("mean_resultant_length")
                mu_val = cs.get("mean_direction_deg")
                if k_val is not None and np.isfinite(k_val):
                    kappas.append(float(k_val))
                if r_val is not None and np.isfinite(r_val):
                    rbars.append(float(r_val))
                if mu_val is not None and np.isfinite(mu_val):
                    mus.append(float(mu_val))
            per_kappa.append(np.asarray(kappas, dtype=float))
            per_rbar.append(np.asarray(rbars, dtype=float))
            per_mu.append(np.asarray(mus, dtype=float))
            per_n_reps.append(len(kappas))
            scalars_per_group.append({
                "label": label, "n_replicates": len(kappas),
                "kappa": list(kappas), "rbar": list(rbars),
                "mu_deg": list(mus),
            })
        out["per_replicate_scalars"] = scalars_per_group

        # _stat_test_n returns (omnibus_dict, pairwise_list).
        # Welch's t for 2 groups, ANOVA for N>2 (auto-selected).
        if sum(1 for arr in per_kappa if arr.size >= 1) >= 2:
            try:
                om_k, pw_k = _stat_test_n(per_kappa, labels)
                out["per_replicate_kappa_test"] = {
                    "omnibus": om_k, "pairwise": pw_k}
            except Exception:
                pass
            try:
                om_r, pw_r = _stat_test_n(per_rbar, labels)
                out["per_replicate_rbar_test"] = {
                    "omnibus": om_r, "pairwise": pw_r}
            except Exception:
                pass
        out["per_replicate_mu_ww"] = _watson_williams_mu_per_replicate(per_mu)

    return out


def _p_stars(p):
    """Three-tier significance markers used in the comparison PDF."""
    try:
        if p is None: return ""
        pf = float(p)
        if np.isnan(pf): return ""
        if pf < 0.001: return "***"
        if pf < 0.01:  return "**"
        if pf < 0.05:  return "*"
        return "ns"
    except Exception:
        return ""


def save_comparison_circular_statistics(groups_angles, *,
                                         csv_path=None, pdf_path=None,
                                         fig_theme="Dark",
                                         track_angle_d_pairs=None,
                                         per_replicate_angles=None):
    """Pool turning angles per group, compute circular statistics for
    each group, write a combined CSV (one row per group) and a multi-
    page themed PDF (one page per group + a comparative summary page).

    Parameters
    ----------
    groups_angles : list of (label, angles_deg_array, color)
        One entry per comparison group.  `angles_deg_array` is the
        concatenation of every replicate's turning angles within the
        group; `color` is the group's display colour (used to tint the
        polar histograms so PDF and master figure agree visually).
    csv_path : str or None
        If given, write a long-form CSV with columns `group`, `n`,
        `mean_direction_deg`, … (all keys from compute_circular_statistics).
    pdf_path : str or None
        If given, write the multi-page PDF.
    fig_theme : str
        "Dark" | "Light" | "Publication".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pal = _theme_palette(fig_theme)

    # ── Per-group stats ────────────────────────────────────────────────
    rows = []
    per_group_stats = []
    for label, angles, color in groups_angles:
        a = np.asarray(angles, dtype=float).ravel()
        a = a[np.isfinite(a)]
        stats = compute_circular_statistics(a)
        per_group_stats.append((label, a, color, stats))
        row = {"group": label}
        row.update(stats)
        rows.append(row)

    # ── Between-group tests ────────────────────────────────────────────
    # Watson-Williams (parametric, tests equal mean directions; assumes
    # κ ≥ 2) plus Mardia-Watson-Wheeler (non-parametric, tests equal
    # distributions; valid at any κ).  Both are reported so the
    # supervisor can pick the appropriate one for their data — and so
    # disagreement between them (one significant, the other not) is
    # visible rather than hidden.
    test_groups = [(g[0], np.asarray(g[1], dtype=float).ravel())
                   for g in groups_angles]
    test_groups = [(lbl, a[np.isfinite(a)]) for lbl, a in test_groups]
    comp_tests = compute_circular_comparison_tests(
        test_groups,
        track_angle_d_pairs=track_angle_d_pairs,
        per_replicate_angles=per_replicate_angles)

    # ── CSV ────────────────────────────────────────────────────────────
    # The single CSV grows by one row per pairwise test (with a `kind`
    # column distinguishing per-group rows from test rows) so a
    # downstream consumer (Excel / R / pandas) can read all the
    # comparison output from one file.
    if csv_path is not None:
        try:
            per_group_rows = [{"kind": "group", **r} for r in rows]
            test_rows = []
            ow = comp_tests.get("omnibus_ww") or {}
            om = comp_tests.get("omnibus_mww") or {}
            if ow:
                test_rows.append({
                    "kind": "test", "test": "Watson-Williams (omnibus)",
                    "label_a": "all", "label_b": "all",
                    "statistic_F": ow.get("F"),
                    "df1": ow.get("df1"), "df2": ow.get("df2"),
                    "p_value": ow.get("p"),
                    "kappa_pooled": ow.get("kappa_pooled"),
                    "valid_assumptions": ow.get("valid"),
                })
            if om:
                test_rows.append({
                    "kind": "test", "test": "Mardia-Watson-Wheeler (omnibus)",
                    "label_a": "all", "label_b": "all",
                    "statistic_W": om.get("W"),
                    "df": om.get("df"),
                    "p_value": om.get("p"),
                })
            ok = comp_tests.get("omnibus_wallraff") or {}
            if ok:
                test_rows.append({
                    "kind": "test",
                    "test": "Wallraff κ-test (omnibus)",
                    "label_a": "all", "label_b": "all",
                    "statistic_H": ok.get("H"),
                    "statistic_U": ok.get("U"),
                    "df": ok.get("df"),
                    "p_value": ok.get("p"),
                })
            for pw in comp_tests.get("pairwise", []):
                ww  = pw.get("ww")  or {}
                mww = pw.get("mww") or {}
                wal = pw.get("wallraff") or {}
                kup = pw.get("kuiper")   or {}
                if ww:
                    test_rows.append({
                        "kind": "test",
                        "test": "Watson-Williams (pairwise)",
                        "label_a": pw["label_a"],
                        "label_b": pw["label_b"],
                        "statistic_F": ww.get("F"),
                        "df1": ww.get("df1"), "df2": ww.get("df2"),
                        "p_value": ww.get("p"),
                        "kappa_pooled": ww.get("kappa_pooled"),
                        "valid_assumptions": ww.get("valid"),
                    })
                if mww:
                    test_rows.append({
                        "kind": "test",
                        "test": "Mardia-Watson-Wheeler (pairwise)",
                        "label_a": pw["label_a"],
                        "label_b": pw["label_b"],
                        "statistic_W": mww.get("W"),
                        "df": mww.get("df"),
                        "p_value": mww.get("p"),
                    })
                if wal:
                    test_rows.append({
                        "kind": "test",
                        "test": "Wallraff κ-test (pairwise)",
                        "label_a": pw["label_a"],
                        "label_b": pw["label_b"],
                        "statistic_U": wal.get("U"),
                        "p_value": wal.get("p"),
                    })
                if kup:
                    test_rows.append({
                        "kind": "test",
                        "test": "Kuiper two-sample",
                        "label_a": pw["label_a"],
                        "label_b": pw["label_b"],
                        "statistic_V": kup.get("V"),
                        "p_value": kup.get("p"),
                    })
            # Circular-linear correlation (per-track mean angle vs D)
            # is a PER-GROUP descriptive measure, not a between-group
            # test — one row per group with r, r², n, p.
            for cl in comp_tests.get("circ_lin_per_group", []):
                res = cl.get("result")
                if not res:
                    continue
                test_rows.append({
                    "kind": "correlation",
                    "test": "Circ-lin: per-track mean angle vs D",
                    "label_a": cl.get("label"),
                    "label_b": "",
                    "r": res.get("r"),
                    "r_squared": res.get("r2"),
                    "statistic_chi2": res.get("test_stat"),
                    "df": res.get("df"),
                    "p_value": res.get("p"),
                    "n": res.get("n"),
                })

            # ── Per-replicate (n=replicates) tests ─────────────────
            # One row per replicate listing the scalars used as data
            # points; then between-group rows for the κ and R̄ Welch
            # / ANOVA tests and the Watson-Williams F-test on μ.
            scalars = comp_tests.get("per_replicate_scalars") or []
            for grp in scalars:
                lbl = grp.get("label", "?")
                ks  = grp.get("kappa") or []
                rs  = grp.get("rbar")  or []
                ms  = grp.get("mu_deg") or []
                # Pad to common length so each replicate gets one row.
                n_rep = max(len(ks), len(rs), len(ms))
                for i in range(n_rep):
                    test_rows.append({
                        "kind": "per_replicate_scalar",
                        "test": "per-replicate κ/R̄/μ",
                        "label_a": lbl,
                        "label_b": f"replicate_{i + 1}",
                        "kappa":   ks[i] if i < len(ks) else None,
                        "rbar":    rs[i] if i < len(rs) else None,
                        "mu_deg":  ms[i] if i < len(ms) else None,
                    })

            def _flatten_per_rep_test(slot, label):
                t = comp_tests.get(slot)
                if not t:
                    return
                om = t.get("omnibus") or {}
                if om:
                    test_rows.append({
                        "kind": "per_replicate_test",
                        "test": f"{label} (omnibus, per-replicate)",
                        "label_a": "all", "label_b": "all",
                        "statistic": om.get("p") and om.get("test"),
                        "p_value": om.get("p"),
                    })
                for pw in (t.get("pairwise") or []):
                    test_rows.append({
                        "kind": "per_replicate_test",
                        "test": f"{label} (pairwise, per-replicate)",
                        "label_a": pw.get("label_i"),
                        "label_b": pw.get("label_j"),
                        "n_a": pw.get("n_i"), "n_b": pw.get("n_j"),
                        "mean_a": pw.get("mean_i"),
                        "mean_b": pw.get("mean_j"),
                        "sem_a": pw.get("sem_i"),
                        "sem_b": pw.get("sem_j"),
                        "statistic": pw.get("test"),
                        "p_value": pw.get("p"),
                    })

            _flatten_per_rep_test("per_replicate_kappa_test", "Welch κ")
            _flatten_per_rep_test("per_replicate_rbar_test",  "Welch R̄")

            mu_ww = comp_tests.get("per_replicate_mu_ww")
            if mu_ww is not None:
                test_rows.append({
                    "kind": "per_replicate_test",
                    "test": "Watson-Williams μ (per-replicate)",
                    "label_a": "all", "label_b": "all",
                    "statistic_F": mu_ww.get("F"),
                    "df1": mu_ww.get("df1"), "df2": mu_ww.get("df2"),
                    "p_value": mu_ww.get("p"),
                })

            df = pd.DataFrame(per_group_rows + test_rows)
            df.to_csv(csv_path, index=False)
        except Exception as exc:
            print(f"  comparison-circstats CSV failed: {exc}")

    # ── PDF ────────────────────────────────────────────────────────────
    if pdf_path is None:
        return per_group_stats

    _rc_keys = ("text.color", "axes.labelcolor", "axes.edgecolor",
                "xtick.color", "ytick.color", "axes.facecolor",
                "axes.titlecolor", "figure.facecolor", "grid.color",
                "font.family")
    _rc_save = {k: plt.rcParams.get(k) for k in _rc_keys}
    plt.rcParams.update({
        "text.color":       pal["TXT"],
        "axes.labelcolor":  pal["TXT"],
        "axes.edgecolor":   pal["GRD"],
        "xtick.color":      pal["TXT"],
        "ytick.color":      pal["TXT"],
        "axes.facecolor":   pal["PNL"],
        "axes.titlecolor":  pal["TXT"],
        "figure.facecolor": pal["BG"],
        "grid.color":       pal["GRD"],
        "font.family":      pal["FONT"],
    })

    def _fmt(x, prec=4):
        try:
            if x is None: return "—"
            xf = float(x)
            if np.isnan(xf): return "—"
            if xf > 0.0 and xf <= 1e-300:
                return "<1e-300"
            return f"{xf:.{prec}g}"
        except Exception:
            return str(x)

    try:
        with PdfPages(pdf_path) as pdf:
            # ── Page 1: comparison summary ─────────────────────────────
            # Landscape A4.  Layout, top → bottom:
            #   y 0.93 – 0.97  header bar
            #   y 0.58 – 0.88  row of polar histograms (one per group)
            #   y 0.51 – 0.55  "Summary" title
            #   y 0.36 – 0.50  per-group summary table
            #   y 0.30 – 0.34  "Between-group tests" title
            #   y 0.13 – 0.29  comparison-tests table
            #   y 0.02 – 0.10  footer block (sign convention + refs)
            fig = plt.figure(figsize=(11.69, 8.27), facecolor=pal["BG"])
            ax_hdr = fig.add_axes([0.05, 0.93, 0.90, 0.04])
            ax_hdr.axis("off")
            ax_hdr.text(0.0, 0.5, "Comparison: Circular Statistics",
                        fontsize=18, fontweight="bold", va="center",
                        ha="left", color=pal["TXT"])
            ax_hdr.text(1.0, 0.5,
                        f"{len(per_group_stats)} groups",
                        fontsize=11, color=pal["MUT"],
                        va="center", ha="right")

            # Grid of polar histograms — auto-wraps to multiple rows
            # when n_groups > 5 so plots don't get sliver-thin.  Each
            # cell is divided VERTICALLY into a label strip (top) and
            # the polar plot itself (below); doing it this way means
            # the group name + n count can never collide with the
            # polar's 0° tick label, regardless of how thick that tick
            # label is at any given font size.
            #
            #   1 ≤ n ≤ 5  →  1 row of n cols  (cell height 0.30, y 0.55–0.88)
            #   6 ≤ n ≤ 10 →  2 rows of ≤ 5 cols  (cell height ~0.16)
            #   n ≥ 11     →  3 rows; outer caller may also paginate.
            #
            # Within each cell:
            #   top 22%  → label band  (group name + "n = N")
            #   bottom 78% → polar plot
            #
            # When in multi-row mode the per-polar font sizes shrink so
            # the tick labels stay readable in a smaller plot.
            n_g  = len(per_group_stats)
            # Polar band height tuned so the polar plot + its top label
            # band (group name + n) sit comfortably above the
            # per-group summary title at y=0.555.  polar_bot=0.58
            # leaves a 0.025 gap to that title.
            polar_top, polar_bot = 0.88, 0.58
            if n_g <= 5:
                n_cols, n_rows = n_g, 1
            elif n_g <= 10:
                n_cols, n_rows = 5, 2
            else:
                # Cap at 12 polars/page; the table-pagination below
                # handles "lots of groups" by giving each batch its
                # own summary page.  For now assume ≤ 12 on page 1.
                n_cols = 6
                n_rows = (min(n_g, 12) + n_cols - 1) // n_cols
            row_h    = (polar_top - polar_bot) / n_rows
            cell_w   = 0.86 / n_cols
            left     = 0.07
            # Tick / label fontsizes shrink when polars get small.
            tick_fs  = 7 if n_cols <= 4 else 6
            lbl_fs   = 10 if n_cols <= 4 else 8
            n_fs     = 8  if n_cols <= 4 else 7
            # Bottom-margin reserves space for the polar's ±180° tick
            # label, which matplotlib renders OUTSIDE the axes box just
            # below the polar circle.  Needs to be large enough that
            # the tick label can't reach down into the Per-group title
            # at y=0.535 below (single-row case) or into the next-row's
            # group label (multi-row case).
            bottom_margin = 0.040 if n_rows == 1 else 0.045
            label_band_frac = 0.22 if n_rows == 1 else 0.28
            for i, (label, a, color, stats) in enumerate(per_group_stats):
                if i >= n_rows * n_cols:
                    break    # truncate at the page's polar capacity
                row = i // n_cols
                col = i % n_cols
                cell_y = polar_top - (row + 1) * row_h
                label_band_h = label_band_frac * row_h
                polar_band_h = row_h - label_band_h - bottom_margin
                ax = fig.add_axes(
                    [left + col * cell_w + 0.015,
                     cell_y + bottom_margin,
                     cell_w - 0.03, polar_band_h],
                    projection="polar")
                ax.set_facecolor(pal["PNL"])
                if a.size >= 10:
                    # Match the master figure's polar convention:
                    # 0° at top, CW positive, signed labels on slot
                    # positions, [0, 2π) wrap for bar rendering.
                    nbins = 36
                    angles_rad = np.mod(np.deg2rad(a), 2.0 * np.pi)
                    bins  = np.linspace(0.0, 2.0 * np.pi, nbins + 1)
                    counts, edges = np.histogram(angles_rad, bins=bins)
                    widths  = np.diff(edges)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    ax.set_theta_zero_location("N")
                    ax.set_theta_direction(-1)
                    bar_col = color or pal["ACC"]
                    ax.bar(centers, counts, width=widths * 0.95,
                           align="center", color=bar_col,
                           edgecolor=pal["PNL"], linewidth=0.4,
                           alpha=0.92)
                    mu = stats.get("mean_direction_deg")
                    if mu is not None and not (
                            isinstance(mu, float) and np.isnan(mu)):
                        r_max = float(counts.max()) if counts.size else 1.0
                        mu_rad = np.mod(np.deg2rad(mu), 2.0 * np.pi)
                        ax.annotate("",
                            xy=(mu_rad, r_max * 0.95),
                            xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->",
                                            color=pal["ARROW"], lw=2.0))
                    ax.set_xticks(np.deg2rad(
                        [0, 45, 90, 135, 180, 225, 270, 315]))
                    ax.set_xticklabels(
                        ["0°", "+45°", "+90°", "+135°", "±180°",
                         "−135°", "−90°", "−45°"], fontsize=tick_fs)
                    ax.set_yticklabels([])
                    ax.tick_params(colors=pal["TXT"], labelsize=tick_fs)
                    ax.grid(True, ls=":", alpha=0.4)
                    # Labels live ABOVE the cell.  Raising the label
                    # block above `cell_y + row_h` (rather than just
                    # inside the top of the cell) creates a clean gap
                    # between the "n = …" line and the polar's 0° tick
                    # label, which renders just outside the polar
                    # circle at the top of the axes box.
                    label_x = (left + col * cell_w + 0.015
                               + (cell_w - 0.03) / 2.0)
                    label_top  = cell_y + row_h + 0.020
                    line2_top  = label_top - 0.018
                    fig.text(label_x, label_top, label,
                             fontsize=lbl_fs, fontweight="bold",
                             ha="center", va="top", color=pal["TXT"])
                    fig.text(label_x, line2_top,
                             f"n = {int(stats.get('n', 0)):,}",
                             fontsize=n_fs, ha="center", va="top",
                             color=pal["MUT"])
                else:
                    ax.axis("off")
                    label_x = (left + col * cell_w + 0.015
                               + (cell_w - 0.03) / 2.0)
                    label_top = cell_y + row_h + 0.020
                    fig.text(label_x, label_top,
                             f"{label}\ntoo few angles",
                             fontsize=n_fs, ha="center", va="top",
                             color=pal["MUT"])

            # Section title placed in FIGURE coords.  Sits below the
            # polar band's bottom (y=0.58) with a generous gap so the
            # polar's ±180° tick label can't reach down into it.
            fig.text(0.05, 0.535,
                     "Per-group summary  (MATLAB CircStat conventions)",
                     fontsize=11, fontweight="bold", va="bottom",
                     ha="left", color=pal["TXT"])
            # Combined summary table — one row per group, columns =
            # the most informative stats for an at-a-glance comparison.
            ax_tbl = fig.add_axes([0.05, 0.43, 0.90, 0.10])
            ax_tbl.axis("off")
            cols = ["group", "n", "mean_direction_deg",
                    "mean_resultant_length", "circular_std_deg",
                    "concentration_kappa", "rayleigh_p", "v_test_p"]
            col_labels = ["Group", "n", "μ (°)", "R̄", "σ_circ (°)",
                          "κ", "Rayleigh p", "V-test p"]
            cell = []
            for r in rows:
                cell.append([
                    str(r["group"]),
                    f"{int(r['n']):,}",
                    _fmt(r["mean_direction_deg"], 4),
                    _fmt(r["mean_resultant_length"], 4),
                    _fmt(r["circular_std_deg"], 4),
                    _fmt(r["concentration_kappa"], 4),
                    _fmt(r["rayleigh_p"], 3),
                    _fmt(r["v_test_p"], 3),
                ])
            tbl = ax_tbl.table(cellText=cell, colLabels=col_labels,
                               cellLoc="left", colLoc="left",
                               bbox=[0.0, 0.0, 1.0, 1.0])
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9.0)
            for (rr, cc), c_obj in tbl.get_celld().items():
                c_obj.set_linewidth(0.5)
                c_obj.set_edgecolor(pal["GRD"])
                if rr == 0:
                    c_obj.set_facecolor(pal["HDR_BG"])
                    c_obj.set_text_props(color=pal["HDR_TXT"],
                                         fontweight="bold")
                else:
                    c_obj.set_facecolor(
                        pal["ZEBRA"] if rr % 2 == 0 else pal["PNL"])
                    c_obj.set_text_props(color=pal["TXT"])

            # ── Between-group tests section ────────────────────────
            #
            # Layout (y-coords):
            #   0.34 : section title
            #   0.27 – 0.33 : plain-English explanation of each test
            #   0.13 – 0.26 : results table (Test  Statistic  p  sig)
            #   0.02 – 0.10 : footer
            #
            # The previous version had a 5th "Note" column for H₀
            # descriptions which overflowed the page; pulling that
            # description out into a separate explanatory paragraph
            # both fixes the overflow AND makes the tests intelligible
            # to a reader who isn't already a circular-statistics
            # expert (supervisor's request).
            fig.text(0.05, 0.395,
                     "Between-group tests — does the turning-angle "
                     "distribution differ between groups?",
                     fontsize=11, fontweight="bold", va="bottom",
                     ha="left", color=pal["TXT"])

            # Plain-English explanation block — 3 compact lines so the
            # tests table below still has room.  Each line covers one
            # test (or the significance convention).  Italicised
            # caveats appear at the end of each line, not on their own
            # row.
            txt_kw = dict(fontsize=8.0, color=pal["TXT"], ha="left",
                          va="top", family=pal["FONT"])
            explain_block = [
                "Watson-Williams F-test (circular ANOVA): tests "
                "EQUAL MEAN DIRECTIONS.  Assumes κ ≥ 2 — rows tagged "
                "\"κ<2\" violate this, prefer M-W-W or Kuiper.",
                "Mardia-Watson-Wheeler & Kuiper (non-parametric): "
                "test EQUAL FULL DISTRIBUTIONS (any change in mean, "
                "spread, or shape).  Safe at any κ.",
                "Wallraff κ-test: tests EQUAL CONCENTRATIONS — answers "
                "\"is one group MORE TIGHTLY clustered than the other?\".",
                "Per-replicate (n = #replicates): Welch's t-test on κ "
                "and R̄, plus Watson-Williams F on per-replicate μ — "
                "respects biological n, not pooled n.",
                "Circ-lin angle vs D (per group): tests whether each "
                "track's mean turning angle correlates with its "
                "diffusion coefficient.  r ∈ [0, 1].",
                "Significant p (< 0.05, stars) rejects H₀ — i.e. "
                "groups DO differ (or angle DOES correlate with D).",
            ]
            yE = 0.395
            for line in explain_block:
                fig.text(0.05, yE, line, **txt_kw)
                yE -= 0.012
            # Build comparison-test rows in priority order:
            #   1. Omnibus Watson-Williams
            #   2. Omnibus Mardia-Watson-Wheeler
            #   3. Pairwise WW / MWW (one row per test per pair)
            # _fmt_p collapses underflow-sentinel p (1e-300) to
            # "<1e-300" so the supervisor doesn't see a literal
            # "1e-300" repeated across rows and assume there's a bug.
            def _fmt_p(p):
                if p is None: return "—"
                pf = float(p)
                if np.isnan(pf): return "—"
                if pf > 0.0 and pf <= 1e-300:
                    return "<1e-300"
                return f"{pf:.3g}"

            omnibus_rows = []
            ow  = comp_tests.get("omnibus_ww")
            om  = comp_tests.get("omnibus_mww")
            owk = comp_tests.get("omnibus_wallraff")
            if ow is not None:
                tag = "" if ow.get("valid", False) else "  (κ<2, caution)"
                omnibus_rows.append([
                    f"Watson-Williams · all groups{tag}",
                    f"F({ow['df1']}, {ow['df2']}) = {ow['F']:.3g}",
                    _fmt_p(ow["p"]),
                    _p_stars(ow["p"]),
                ])
            if om is not None:
                omnibus_rows.append([
                    "Mardia-Watson-Wheeler · all groups",
                    f"W({om['df']}) = {om['W']:.3g}",
                    _fmt_p(om["p"]),
                    _p_stars(om["p"]),
                ])
            if owk is not None:
                # k=2 → Mann-Whitney U; k>2 → Kruskal-Wallis H.
                if "H" in owk:
                    stat_str = f"H({owk['df']}) = {owk['H']:.3g}"
                else:
                    stat_str = f"U = {owk.get('U', 0):.3g}"
                omnibus_rows.append([
                    "Wallraff κ-test · all groups",
                    stat_str,
                    _fmt_p(owk["p"]),
                    _p_stars(owk["p"]),
                ])

            pairwise_rows = []
            for pw in comp_tests.get("pairwise", []):
                ww  = pw.get("ww")
                mww = pw.get("mww")
                wal = pw.get("wallraff")
                kup = pw.get("kuiper")
                pair = f"{pw['label_a']}  vs  {pw['label_b']}"
                if ww is not None:
                    tag = "" if ww.get("valid", False) else "  (κ<2)"
                    pairwise_rows.append([
                        f"Watson-Williams · {pair}{tag}",
                        f"F({ww['df1']}, {ww['df2']}) = {ww['F']:.3g}",
                        _fmt_p(ww["p"]),
                        _p_stars(ww["p"]),
                    ])
                if mww is not None:
                    pairwise_rows.append([
                        f"Mardia-Watson-Wheeler · {pair}",
                        f"W({mww['df']}) = {mww['W']:.3g}",
                        _fmt_p(mww["p"]),
                        _p_stars(mww["p"]),
                    ])
                if wal is not None:
                    pairwise_rows.append([
                        f"Wallraff κ-test · {pair}",
                        f"U = {wal.get('U', 0):.3g}",
                        _fmt_p(wal["p"]),
                        _p_stars(wal["p"]),
                    ])
                if kup is not None:
                    pairwise_rows.append([
                        f"Kuiper 2-sample · {pair}",
                        f"V = {kup['V']:.4g}",
                        _fmt_p(kup["p"]),
                        _p_stars(kup["p"]),
                    ])

            # Per-group circular-linear correlation rows.  These are
            # descriptive stats (one per group), not between-group
            # tests, but they live in the same table because they share
            # the same "name · stat · p · sig" template.
            corr_rows = []
            for cl in comp_tests.get("circ_lin_per_group", []):
                res = cl.get("result")
                grp = cl.get("label", "?")
                if not res:
                    corr_rows.append([
                        f"Circ-lin angle vs D · {grp}",
                        "n < 3", "—", "",
                    ])
                    continue
                corr_rows.append([
                    f"Circ-lin angle vs D · {grp}",
                    (f"r = {res['r']:.3g}  "
                     f"(χ²({res['df']}) = {res['test_stat']:.3g})"),
                    _fmt_p(res["p"]),
                    _p_stars(res["p"]),
                ])

            # ── Per-replicate test rows ─────────────────────────────
            # n = number of biological replicates, not pooled angles.
            # Each row reads "Welch κ · all groups" or "Welch κ · A vs
            # B" plus the t/F statistic and the p-value with stars.
            per_rep_rows = []

            def _push_per_rep(slot, label):
                t = comp_tests.get(slot)
                if not t:
                    return
                om = t.get("omnibus") or {}
                if om and om.get("p") is not None:
                    test_name = om.get("test", "")
                    per_rep_rows.append([
                        f"{label} · all groups  ({test_name})",
                        "(see CSV for full stats)",
                        _fmt_p(om["p"]),
                        _p_stars(om["p"]),
                    ])
                for pw in (t.get("pairwise") or []):
                    if pw.get("p") is None:
                        continue
                    pair = (f"{pw.get('label_i', '?')}  vs  "
                            f"{pw.get('label_j', '?')}")
                    n_i = pw.get("n_i", 0)
                    n_j = pw.get("n_j", 0)
                    per_rep_rows.append([
                        f"{label} · {pair}  ({pw.get('test', '')})",
                        f"n = {n_i} vs {n_j}",
                        _fmt_p(pw["p"]),
                        _p_stars(pw["p"]),
                    ])

            _push_per_rep("per_replicate_kappa_test", "Welch κ (per-replicate)")
            _push_per_rep("per_replicate_rbar_test",  "Welch R̄ (per-replicate)")

            mu_ww = comp_tests.get("per_replicate_mu_ww")
            if mu_ww is not None and mu_ww.get("p") is not None:
                tag = "" if mu_ww.get("valid", False) else "  (κ<2)"
                per_rep_rows.append([
                    f"Watson-Williams μ · all groups (per-replicate){tag}",
                    f"F({mu_ww['df1']}, {mu_ww['df2']}) = {mu_ww['F']:.3g}",
                    _fmt_p(mu_ww["p"]),
                    _p_stars(mu_ww["p"]),
                ])

            # Page-1 tests table: omnibus + circ-lin correlations +
            # per-replicate tests + as many pairwise rows as fit.  We
            # always put the per-group correlations and per-replicate
            # tests on page 1 (they're tiny, ~k rows each) and let the
            # pooled pairwise tests be the ones that paginate.
            PAGE1_TESTS_CAP = 14        # omnibus + corr + per-rep + pairwise
            CONT_PAGE_CAP   = 24        # ~24 rows on a continuation page

            fixed_rows = omnibus_rows + corr_rows + per_rep_rows
            page1_pairwise_cap = max(
                PAGE1_TESTS_CAP - len(fixed_rows), 0)
            page1_tests   = fixed_rows + pairwise_rows[:page1_pairwise_cap]
            overflow_pairs = pairwise_rows[page1_pairwise_cap:]

            if not page1_tests:
                page1_tests = [["Insufficient data", "—", "—", "—"]]

            def _render_tests_table(host_fig, rect, cells, pal):
                """Render a 4-column tests table into the given fig+rect."""
                ax = host_fig.add_axes(rect); ax.axis("off")
                tbl = ax.table(
                    cellText=cells,
                    colLabels=["Test  ·  Comparison", "Statistic",
                               "p-value", "sig"],
                    cellLoc="left", colLoc="left",
                    colWidths=[0.55, 0.25, 0.13, 0.07],
                    bbox=[0.0, 0.0, 1.0, 1.0])
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(8.5)
                for (rr, cc), c_obj in tbl.get_celld().items():
                    c_obj.set_linewidth(0.5)
                    c_obj.set_edgecolor(pal["GRD"])
                    if rr == 0:
                        c_obj.set_facecolor(pal["HDR_BG"])
                        c_obj.set_text_props(color=pal["HDR_TXT"],
                                             fontweight="bold")
                    else:
                        c_obj.set_facecolor(
                            pal["ZEBRA"] if rr % 2 == 0 else pal["PNL"])
                        c_obj.set_text_props(color=pal["TXT"])

            _render_tests_table(fig, [0.05, 0.14, 0.90, 0.165],
                                page1_tests, pal)

            # ── Footer block ──────────────────────────────────────
            # Pre-wrapped lines instead of one long sign-convention
            # string: matplotlib's fig.text doesn't wrap against the
            # figure margins, so the long "Sign convention…" line was
            # being cut off at the right edge.  Same wrapping pattern
            # the per-file PDF footer uses (search `sign_lines = [`).
            def _render_footer(host_fig, pal, *, top_y=0.105):
                _foot_kw2 = dict(fontsize=7, color=pal["MUT"], ha="left",
                                 va="bottom", family=pal["FONT"])
                foot_lines = [
                    "Sign convention: turning angles are SIGNED on "
                    "(−180°, +180°].",
                    "0° = straight ahead.  +θ = left turn (CCW).  "
                    "−θ = right turn (CW).  ±180° = full reversal.",
                    "Plots use clockwise-positive direction so +θ "
                    "labels appear on the right hemisphere.",
                    "Significance markers: *** p<0.001,  ** p<0.01,  "
                    "* p<0.05,  ns = not significant.",
                    "References: Mardia & Jupp 2000 §6.4.2, §7.6.1; "
                    "Fisher 1993; Berens 2009 (CircStat).",
                ]
                y2 = top_y
                for line in foot_lines:
                    host_fig.text(0.05, y2, line, **_foot_kw2)
                    y2 -= 0.014

            _render_footer(fig, pal)
            pdf.savefig(fig, facecolor=pal["BG"])
            plt.close(fig)

            # ── Continuation pages for overflow pairwise tests ──────
            # When the pairwise count is large (e.g. 6+ groups → 15+
            # pairs × 2 tests = 30+ rows), we paginate the remainder
            # onto fresh landscape pages so nothing gets squashed off
            # the bottom of page 1.
            if overflow_pairs:
                page_num = 2
                total_cont_pages = (len(overflow_pairs)
                                    + CONT_PAGE_CAP - 1) // CONT_PAGE_CAP
                for chunk_start in range(0, len(overflow_pairs),
                                         CONT_PAGE_CAP):
                    chunk = overflow_pairs[chunk_start:
                                           chunk_start + CONT_PAGE_CAP]
                    fig_c = plt.figure(figsize=(11.69, 8.27),
                                       facecolor=pal["BG"])
                    ax_h = fig_c.add_axes([0.05, 0.93, 0.90, 0.04])
                    ax_h.axis("off")
                    ax_h.text(0.0, 0.5,
                              "Comparison: Circular Statistics  —  "
                              f"pairwise tests (page {page_num - 1} of "
                              f"{total_cont_pages})",
                              fontsize=14, fontweight="bold",
                              va="center", ha="left", color=pal["TXT"])
                    # Big tests-table area on a continuation page.
                    _render_tests_table(fig_c,
                                        [0.05, 0.13, 0.90, 0.75],
                                        chunk, pal)
                    _render_footer(fig_c, pal)
                    pdf.savefig(fig_c, facecolor=pal["BG"])
                    plt.close(fig_c)
                    page_num += 1

            # ── Pages 2..N+1: per-group full report ───────────────────
            for label, a, color, stats in per_group_stats:
                # Reuse the single-file renderer by writing to a
                # temp page object isn't supported directly — instead,
                # we mirror its layout here in a fresh figure so the
                # per-group pages all live in ONE multi-page PDF.
                _write_single_group_page(pdf, a, stats, label, pal, color)
    finally:
        plt.rcParams.update(_rc_save)

    return per_group_stats


def _write_single_group_page(pdf, angles_deg, stats, label, pal,
                              group_color=None):
    """Render one A4-portrait page mirroring save_circular_statistics_pdf
    into an open PdfPages stream.  Used by save_comparison_circular_
    statistics so the per-group full reports all live inside the same
    multi-page comparison PDF.
    """
    import matplotlib.pyplot as plt

    a = np.asarray(angles_deg, dtype=float).ravel()
    a = a[np.isfinite(a)]

    def _fmt(x, prec=4):
        try:
            if x is None: return "—"
            xf = float(x)
            if np.isnan(xf): return "—"
            if xf > 0.0 and xf <= 1e-300:
                return "<1e-300"
            return f"{xf:.{prec}g}"
        except Exception:
            return str(x)

    rows = [
        ("n", "Sample size", "count", f"{int(stats.get('n', 0)):,}"),
        ("mean_direction_deg", "Mean direction μ", "deg",
         _fmt(stats.get("mean_direction_deg"), 4)),
        ("mean_resultant_length",
         "Mean resultant length R̄  (0 = uniform, 1 = aligned)", "—",
         _fmt(stats.get("mean_resultant_length"), 4)),
        ("circular_variance", "Circular variance  1 − R̄", "—",
         _fmt(stats.get("circular_variance"), 4)),
        ("circular_std_deg",
         "Circular standard deviation  √(−2·ln R̄)", "deg",
         _fmt(stats.get("circular_std_deg"), 4)),
        ("angular_deviation_deg",
         "Angular deviation  s₀ = √(2·(1−R̄))", "deg",
         _fmt(stats.get("angular_deviation_deg"), 4)),
        ("median_deg", "Circular median", "deg",
         _fmt(stats.get("median_deg"), 4)),
        ("concentration_kappa",
         "Von Mises concentration κ  (Best & Fisher 1981)", "—",
         _fmt(stats.get("concentration_kappa"), 4)),
        ("rayleigh_z", "Rayleigh test statistic  z = n·R̄²", "—",
         _fmt(stats.get("rayleigh_z"), 4)),
        ("rayleigh_p", "Rayleigh test p-value  (uniformity)", "—",
         _fmt(stats.get("rayleigh_p"), 3)),
        ("v_test_z", "V-test statistic against μ₀ = 0°", "—",
         _fmt(stats.get("v_test_z"), 4)),
        ("v_test_p", "V-test p-value  (preferred direction)", "—",
         _fmt(stats.get("v_test_p"), 3)),
        ("circular_skewness",
         "Circular skewness  (Mardia & Jupp §2.3)", "—",
         _fmt(stats.get("circular_skewness"), 4)),
        ("circular_kurtosis",
         "Circular kurtosis  (Mardia & Jupp §2.3)", "—",
         _fmt(stats.get("circular_kurtosis"), 4)),
        ("ci95_lower_deg",
         "95% CI lower bound for μ  (large-sample)", "deg",
         _fmt(stats.get("ci95_lower_deg"), 4)),
        ("ci95_upper_deg",
         "95% CI upper bound for μ  (large-sample)", "deg",
         _fmt(stats.get("ci95_upper_deg"), 4)),
    ]

    # Layout mirrors save_circular_statistics_pdf — same coord bands so
    # the per-group page in a comparison PDF reads like a per-file PDF.
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=pal["BG"])
    ax_hdr = fig.add_axes([0.07, 0.94, 0.86, 0.04])
    ax_hdr.axis("off")
    ax_hdr.text(0.0, 0.5, f"Circular Statistics — {label}",
                fontsize=15, fontweight="bold", va="center",
                ha="left", color=pal["TXT"])
    ax_hdr.text(1.0, 0.5,
                f"n = {int(stats.get('n', 0)):,} turning angles",
                fontsize=11, color=pal["MUT"], va="center", ha="right")

    ax_polar = fig.add_axes([0.08, 0.61, 0.36, 0.25], projection="polar")
    ax_polar.set_facecolor(pal["PNL"])
    if a.size >= 10:
        # Same convention as the master figure: 0° top, CW positive,
        # signed labels on slot positions, [0, 2π) wrap for bars.
        nbins = 36
        angles_rad = np.mod(np.deg2rad(a), 2.0 * np.pi)
        bins  = np.linspace(0.0, 2.0 * np.pi, nbins + 1)
        counts, edges = np.histogram(angles_rad, bins=bins)
        widths = np.diff(edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax_polar.set_theta_zero_location("N")
        ax_polar.set_theta_direction(-1)
        ax_polar.bar(centers, counts, width=widths * 0.95, align="center",
                     color=group_color or pal["ACC"],
                     edgecolor=pal["PNL"], linewidth=0.4, alpha=0.92)
        mu = stats.get("mean_direction_deg")
        if mu is not None and not (isinstance(mu, float) and np.isnan(mu)):
            r_max = float(counts.max()) if counts.size else 1.0
            mu_rad = np.mod(np.deg2rad(mu), 2.0 * np.pi)
            ax_polar.annotate("",
                xy=(mu_rad, r_max * 0.95), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=pal["ARROW"],
                                lw=2.0))
        ax_polar.set_xticks(np.deg2rad(
            [0, 45, 90, 135, 180, 225, 270, 315]))
        ax_polar.set_xticklabels(
            ["0°", "+45°", "+90°", "+135°", "±180°",
             "−135°", "−90°", "−45°"], fontsize=8)
        ax_polar.set_yticklabels([])
        ax_polar.tick_params(colors=pal["TXT"], labelsize=8)
        ax_polar.grid(True, ls=":", alpha=0.4)
        # Title intentionally omitted — see save_circular_statistics_pdf
        # for the rationale (header + footer already cover it).
    else:
        ax_polar.axis("off")
        ax_polar.text(0.5, 0.5, "Too few angles for histogram",
                      transform=ax_polar.transAxes,
                      ha="center", va="center", color=pal["MUT"],
                      fontsize=10)

    # Compact "Top stats" box for the right side.
    ax_top = fig.add_axes([0.48, 0.61, 0.46, 0.25]); ax_top.axis("off")
    R = stats.get("mean_resultant_length")
    p = stats.get("rayleigh_p")
    lines = [
        f"Mean direction μ:        {_fmt(stats.get('mean_direction_deg'), 4)}°",
        f"Resultant length R̄:      {_fmt(stats.get('mean_resultant_length'), 4)}",
        f"Concentration κ:         {_fmt(stats.get('concentration_kappa'), 4)}",
        f"Rayleigh p (uniformity): {_fmt(stats.get('rayleigh_p'), 3)}",
        f"V-test p (μ₀ = 0°):      {_fmt(stats.get('v_test_p'), 3)}",
    ]
    ax_top.text(0.0, 1.0, "Headline stats", fontsize=12,
                fontweight="bold", va="top", color=pal["TXT"])
    ax_top.text(0.0, 0.88, "\n".join(lines), fontsize=10, va="top",
                family="monospace", color=pal["TXT"])

    # Section title in figure coords so it can't collide with the polar
    # plot's bottom tick labels above.
    fig.text(0.07, 0.555, "Statistics  (MATLAB CircStat conventions)",
             fontsize=12, fontweight="bold", va="bottom",
             ha="left", color=pal["TXT"])
    ax_tbl = fig.add_axes([0.07, 0.12, 0.88, 0.40]); ax_tbl.axis("off")

    cell_text, row_labels = [], []
    for key, gloss, unit, val in rows:
        unit_s = "" if unit in ("", "—") else f"  ({unit})"
        cell_text.append([f"{gloss}", f"{val}{unit_s}"])
        row_labels.append(key)
    tbl = ax_tbl.table(cellText=cell_text, rowLabels=row_labels,
                       colLabels=["Description", "Value"],
                       cellLoc="left", rowLoc="left", colLoc="left",
                       colWidths=[0.62, 0.28],
                       bbox=[0.20, 0.0, 0.80, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.0)
    for (rr, cc), c_obj in tbl.get_celld().items():
        c_obj.set_linewidth(0.5)
        c_obj.set_edgecolor(pal["GRD"])
        if rr == 0:
            c_obj.set_facecolor(pal["HDR_BG"])
            c_obj.set_text_props(color=pal["HDR_TXT"], fontweight="bold")
        else:
            c_obj.set_facecolor(
                pal["ZEBRA"] if rr % 2 == 0 else pal["PNL"])
            if cc == -1:
                c_obj.set_text_props(family="monospace", fontsize=8.0,
                                     color=pal["MUT"])
            else:
                c_obj.set_text_props(color=pal["TXT"])

    _foot_kw = dict(fontsize=7, color=pal["MUT"], ha="left",
                    va="bottom", family=pal["FONT"])
    sign_lines = [
        "Sign convention: turning angles SIGNED on (−180°, +180°].",
        "0° = straight.  +θ = left turn (CCW).  −θ = right turn (CW).  "
        "±180° = reversal.",
        "Unsigned 0–360° equivalent: u = θ if θ ≥ 0, else θ + 360 "
        "(so −90° ≡ 270°, +90° ≡ 90°).",
    ]
    ref_lines = [
        "References: Mardia & Jupp 2000; Fisher 1993; "
        "Berens 2009 (CircStat).",
    ]
    y = 0.095
    for line in sign_lines:
        fig.text(0.07, y, line, **_foot_kw); y -= 0.014
    y -= 0.006
    for line in ref_lines:
        fig.text(0.07, y, line, **_foot_kw); y -= 0.014
    pdf.savefig(fig, facecolor=pal["BG"])
    plt.close(fig)




# ══════════════════════════════════════════════════════════════════════════════
#  MOBILE FRACTION OVER TIME
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTER ANALYSIS  (DBSCAN)
# ══════════════════════════════════════════════════════════════════════════════

# fa_clustering extracted; re-exported here.
from fa_clustering import (
    compute_clusters,
)


# ══════════════════════════════════════════════════════════════════════════════
#  DWELL TIME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
#  MOMENT SCALING SPECTRUM  (MSS)
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ══════════════════════════════════════════════════════════════════════════════

MC   = {"Immobile":"#e05252","Confined":"#f5a623","Brownian":"#4a90d9",
        "Directed":"#7ed321","Unknown":"#aaaaaa"}
MORD = ["Immobile","Confined","Brownian","Directed"]


def _draw_track(grp, color, ax, lw=0.8, alpha=0.6):
    """Draw one track with a tail-to-head alpha fade.

    Old implementation called ax.plot() once per segment (N-1 calls
    per track).  For 2000 tracks × 50 frames = ~100 000 plot calls,
    figure rendering became the bottleneck of the whole save phase.

    LineCollection batches all segments of a single track into one
    artist with a per-segment alpha array — same visual result,
    ~30× faster on dense track sets.
    """
    xy = grp[["x", "y"]].values
    n = len(xy)
    if n < 2:
        return
    # Build segment endpoints: shape (N-1, 2, 2) — i.e. for each
    # segment, [start_xy, end_xy].
    import numpy as _np
    from matplotlib.collections import LineCollection as _LC
    segs = _np.stack([xy[:-1], xy[1:]], axis=1)
    # Per-segment alpha ramp from 0.2 → `alpha`.
    alphas = _np.linspace(0.2, alpha, max(n - 1, 1))
    # Pre-multiply RGBA so each segment carries its own alpha through
    # the LineCollection.  Accept either a hex string or RGB tuple.
    try:
        import matplotlib.colors as _mc
        r, g, b = _mc.to_rgb(color)
    except Exception:
        r, g, b = 0.5, 0.5, 0.5
    colors = _np.column_stack(
        [_np.full(len(alphas), r),
         _np.full(len(alphas), g),
         _np.full(len(alphas), b),
         alphas])
    lc = _LC(segs, colors=colors, linewidths=lw,
             capstyle="round", antialiased=True)
    ax.add_collection(lc)


def make_figure(stack, tracks, imsd_df, emsd_df, diff_df,
                pixel_size, frame_interval, output_path=None, roi_mask=None,
                fig_theme="Dark", proj_cmap="Inferno", jdd=None,
                turning_angles=None, mobile_frac_df=None,
                cluster_labels=None, cluster_locs=None,
                dwell_df=None, dwell_tau=None, return_pdf_bytes=False,
                want_panels=None):
    # want_panels controls the per-panel PNG export, which is expensive:
    # each panel is produced by a full-figure savefig() cropped to that
    # panel's bbox, so rendering all 15 panels means ~15 full rasterisations
    # of the whole figure.  Callers that don't need per-panel PNGs should
    # pass an empty collection to skip the loop entirely.
    #   * None            → render every panel (back-compat default)
    #   * set()/[]        → render no panels (just the combined figure)
    #   * {"A","C", ...}  → render only those panels
    print("  Rendering figure ...")

    # ── Theme palettes ─────────────────────────────────────────────────────────
    if fig_theme == "Light":
        BG, PNL   = "#ffffff", "#f6f8fa"
        TXT, GRD  = "#24292f", "#d0d7de"
        ACC       = "#0969da"
        _kde_col  = "#000000"
        _traj_bg  = "Greys"
        _pie_text = "#ffffff"
        _font     = "sans-serif"
    elif fig_theme == "Publication":
        BG, PNL   = "#ffffff", "#ffffff"
        TXT, GRD  = "#000000", "#cccccc"
        ACC       = "#333333"
        _kde_col  = "#000000"
        _traj_bg  = "Greys"
        _pie_text = "#ffffff"
        _font     = "serif"
    elif fig_theme == "AMOLED":
        # Pure-black BG variant of Dark.
        BG, PNL   = "#000000", "#0a0a0a"
        TXT, GRD  = "#e6edf3", "#30363d"
        ACC       = "#58a6ff"
        _kde_col  = "white"
        _traj_bg  = "Greys_r"
        _pie_text = "#000000"
        _font     = "monospace"
    else:                                    # Dark (default)
        BG, PNL   = "#0d1117", "#161b22"
        TXT, GRD  = "#e6edf3", "#30363d"
        ACC       = "#58a6ff"
        _kde_col  = "white"
        _traj_bg  = "Greys_r"
        _pie_text = "#0d1117"
        _font     = "monospace"

    # ── Projection colourmap ───────────────────────────────────────────────────
    _cmap_map = {
        "Inferno": "inferno",
        "Hot":     "hot",
        "Viridis": "viridis",
        "Plasma":  "plasma",
        "Greys":   "Greys" if fig_theme in ("Light", "Publication") else "Greys_r",   # Dark + AMOLED → Greys_r
    }
    _pcmap = _cmap_map.get(proj_cmap, "inferno")

    plt.rcParams.update({
        "text.color":       TXT, "axes.labelcolor": TXT,
        "xtick.color":      TXT, "ytick.color":     TXT,
        "axes.edgecolor":   GRD, "axes.facecolor":  PNL,
        "grid.color":       GRD, "grid.alpha":      0.4,
        "font.family":      _font})

    _has_jdd = jdd is not None
    # Grid expanded from 5 to 6 rows in v1.0.64 to fit the new Radial
    # Distribution polar panel.
    fig = plt.figure(figsize=(20, 38), facecolor=BG)
    gs  = GridSpec(6, 3, figure=fig, hspace=0.45, wspace=0.32,
                   left=0.06, right=0.97, top=0.95, bottom=0.035)

    _panels          = []   # (letter, axes) collected for per-panel export
    _letter_artists  = []   # text objects for letter labels (hidden for panel renders)

    def sax(ax, ltr, ttl):
        ax.set_facecolor(PNL)
        for sp in ax.spines.values(): sp.set_edgecolor(GRD)
        ax.set_title(f"  {ttl}", loc="left", fontsize=11,
                     color=TXT, pad=8, fontweight="bold")
        txt = ax.text(-0.04, 1.06, ltr, transform=ax.transAxes, fontsize=14,
                      color=ACC, fontweight="bold", va="top", ha="right")
        _panels.append((ltr, ax))
        _letter_artists.append(txt)

    # Use up to 200 evenly-spaced frames for the max projection to save memory
    idx  = np.linspace(0, len(stack)-1, min(200, len(stack)), dtype=int)
    proj = stack[idx].max(axis=0)
    from skimage import exposure as _exp
    proj_eq = _exp.equalize_adapthist(
        (proj / proj.max()).astype(np.float32), clip_limit=0.03)
    mcol = diff_df.set_index("particle")["motion"].to_dict()

    # A — max projection
    ax = fig.add_subplot(gs[0,0])
    ax.imshow(proj_eq, cmap=_pcmap, origin="lower", aspect="equal")
    bp = 5/pixel_size; y0,x0 = proj.shape[0]*.05, proj.shape[1]*.05
    ax.plot([x0,x0+bp],[y0,y0],"-",color="white",lw=3)
    ax.text(x0+bp/2,y0+proj.shape[0]*.025,"5 um",
            ha="center",va="bottom",color="white",fontsize=8)
    ax.set_xlabel(f"X  ({pixel_size} um/px)",fontsize=9)
    ax.set_ylabel("Y (px)",fontsize=9)
    if roi_mask is not None:
        ax.contour(roi_mask.astype(float), levels=[0.5],
                   colors=["#58a6ff"], linewidths=[1.2], alpha=0.8)
        ax.text(0.02, 0.02, f"ROI", transform=ax.transAxes,
                color="#58a6ff", fontsize=8, va="bottom")
    sax(ax,"A","Max Projection")

    # B — trajectory map coloured by motion type (subsample if very many tracks)
    ax = fig.add_subplot(gs[0,1])
    ax.imshow(proj_eq,cmap=_traj_bg,origin="lower",aspect="equal",alpha=0.35)
    all_pids  = list(tracks["particle"].unique())
    draw_pids = set(np.random.default_rng(42).choice(
        all_pids, min(2000, len(all_pids)), replace=False))
    n_drawn = 0
    for pid, grp in (tracks[tracks["particle"].isin(draw_pids)]
                     .reset_index(drop=True).sort_values("frame")
                     .groupby("particle")):
        _draw_track(grp, MC.get(mcol.get(pid,"Unknown"),"#aaa"), ax)
        n_drawn += 1
    els = [Line2D([0],[0],color=MC[m],lw=2,label=m)
           for m in MORD if m in mcol.values()]
    ax.legend(handles=els,fontsize=8,loc="upper right",
              framealpha=0.7,facecolor=PNL,edgecolor=GRD,labelcolor=TXT)
    ax.set_xlim(0,proj.shape[1]); ax.set_ylim(0,proj.shape[0])
    ax.set_xlabel("X (px)",fontsize=9); ax.set_ylabel("Y (px)",fontsize=9)
    shown = f"{n_drawn:,}" + (f" of {len(all_pids):,}" if n_drawn < len(all_pids) else "")
    sax(ax,"B",f"Trajectories  (n={shown})")

    # C — trajectories coloured by D value
    ax = fig.add_subplot(gs[0,2])
    ax.imshow(proj_eq, cmap=_traj_bg, origin="lower", aspect="equal", alpha=0.35)
    d_map = diff_df.set_index("particle")["D"].to_dict()
    d_vals_valid = [v for v in d_map.values() if v is not None and np.isfinite(v) and v > 0]
    if d_vals_valid:
        log_d_vals = np.log10(d_vals_valid)
        _p5  = np.percentile(log_d_vals, 5)
        _p95 = np.percentile(log_d_vals, 95)
        _cmap_d = plt.cm.plasma
        _norm_d = plt.Normalize(vmin=_p5, vmax=_p95)
        _sm_d   = plt.cm.ScalarMappable(cmap=_cmap_d, norm=_norm_d)
        _sm_d.set_array([])
        draw_pids_c = set(np.random.default_rng(43).choice(
            all_pids, min(2000, len(all_pids)), replace=False))
        for pid, grp in (tracks[tracks["particle"].isin(draw_pids_c)]
                         .reset_index(drop=True).sort_values("frame")
                         .groupby("particle")):
            D_val = d_map.get(pid)
            if D_val is not None and np.isfinite(D_val) and D_val > 0:
                col = _cmap_d(_norm_d(np.log10(D_val)))
            else:
                col = "#555555"
            _draw_track(grp, col, ax)
        cb = plt.colorbar(_sm_d, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("log10(D)  [µm²/s]", fontsize=8, color=TXT)
        cb.ax.yaxis.set_tick_params(color=TXT)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TXT, fontsize=7)
    ax.set_xlim(0, proj.shape[1]); ax.set_ylim(0, proj.shape[0])
    ax.set_xlabel("X (px)", fontsize=9); ax.set_ylabel("Y (px)", fontsize=9)
    sax(ax, "C", "Trajectories by D value")

    # D — MSD curves
    ax = fig.add_subplot(gs[1,0])
    lt  = emsd_df.index.values * frame_interval
    rng = np.random.default_rng(42)
    for pid in rng.choice(list(imsd_df.columns), min(200,len(imsd_df.columns)), replace=False):
        v  = imsd_df[pid].values
        t  = imsd_df.index.values * frame_interval
        ok = np.isfinite(v) & (v > 0)
        if ok.sum() >= 2:
            ax.plot(t[ok],v[ok],"-",color="#8b949e",lw=0.4,alpha=0.3)
    ax.plot(lt,emsd_df.values,"-o",color=ACC,lw=2.5,ms=4,zorder=5,
            label="Ensemble MSD")
    try:
        t6,m6 = lt[:6], emsd_df.values[:6].ravel()
        ok6   = np.isfinite(m6) & (m6>0)
        po,_  = curve_fit(msd_linear,t6[ok6],m6[ok6],p0=[0.01,0],maxfev=2000)
        te    = np.linspace(t6[0],lt[-1],200)
        ax.plot(te,msd_linear(te,*po),"--",color="#f78166",lw=2,
                label=f"Fit D={po[0]:.4f} um2/s")
    except Exception: pass
    ax.set_xlabel("Lag time (s)",fontsize=9)
    ax.set_ylabel("MSD (um2)",fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True,which="both",ls=":",alpha=0.3)
    ax.legend(fontsize=8,framealpha=0.6,facecolor=PNL,edgecolor=GRD,labelcolor=TXT)
    sax(ax,"D","MSD Curves")

    # E — D distribution
    ax = fig.add_subplot(gs[1,1])
    dv = diff_df["D"].dropna()
    dv = dv[(dv>0) & (dv<dv.quantile(0.995))]
    if len(dv) > 5:
        ld   = np.log10(dv)
        bins = np.linspace(ld.min(), ld.max(), 40)
        for m in MORD:
            sub = diff_df[(diff_df["motion"]==m) & (diff_df["D"]>0)]
            if len(sub):
                ax.hist(np.log10(sub["D"].clip(1e-6)),bins=bins,
                        color=MC[m],alpha=0.7,label=m,edgecolor="none")
        if len(ld) > 10:
            kde = gaussian_kde(ld)
            xk  = np.linspace(ld.min(), ld.max(), 300)
            ax.plot(xk, kde(xk)*len(dv)*(bins[1]-bins[0]),
                    "-",color=_kde_col,lw=2)
        ax.axvline(np.log10(dv.median()),color=ACC,ls="--",lw=1.5,
                   label=f"Median={dv.median():.4f}")
        ax.set_xlabel("log10(D)  [um2/s]",fontsize=9)
        ax.set_ylabel("Count",fontsize=9)
        ax.legend(fontsize=8,framealpha=0.6,facecolor=PNL,edgecolor=GRD,labelcolor=TXT)
    ax.grid(True,ls=":",alpha=0.3)
    sax(ax,"E","Diffusion Coefficient Distribution")

    # F — pie chart
    ax = fig.add_subplot(gs[1,2])
    mc_ = diff_df["motion"].value_counts()
    lbl = [m for m in MORD if m in mc_]
    sz  = [mc_[m] for m in lbl]
    co  = [MC[m] for m in lbl]
    _,_,ats = ax.pie(sz,labels=lbl,colors=co,autopct="%1.1f%%",startangle=140,
                      textprops={"color":TXT,"fontsize":9},
                      wedgeprops={"edgecolor":PNL,"linewidth":2})
    for at in ats: at.set_fontsize(8); at.set_color(_pie_text)
    sax(ax,"F","Motion Classification")

    # G — alpha distribution
    ax = fig.add_subplot(gs[2,0])
    av = diff_df["alpha"].dropna()
    av = av[(av>-1) & (av<4)]
    if len(av) > 5:
        ba = np.linspace(av.min(), av.max(), 40)
        for m in MORD:
            sub = diff_df[(diff_df["motion"]==m) & diff_df["alpha"].notna()]
            if len(sub):
                ax.hist(sub["alpha"].clip(-1,4),bins=ba,
                        color=MC[m],alpha=0.7,label=m,edgecolor="none")
        for xv,lb,ls in [(0.5,"a=0.5",":"),(1.0,"a=1 Brownian","--"),(2.0,"a=2 directed",":")]:
            ax.axvline(xv,color=GRD,ls=ls,lw=1.2,label=lb)
        ax.set_xlabel("Anomalous exponent alpha",fontsize=9)
        ax.set_ylabel("Count",fontsize=9)
        ax.legend(fontsize=7,framealpha=0.6,facecolor=PNL,edgecolor=GRD,labelcolor=TXT)
    ax.grid(True,ls=":",alpha=0.3)
    sax(ax,"G","Anomalous Exponent Alpha Distribution")

    # H — Position Density Heatmap
    ax = fig.add_subplot(gs[2, 1])
    try:
        x_um = tracks["x"].values * pixel_size
        y_um = tracks["y"].values * pixel_size
        h, xe, ye = np.histogram2d(x_um, y_um, bins=120)
        from scipy.ndimage import gaussian_filter as _gf
        h_sm = _gf(h, sigma=1.5)
        ax.imshow(h_sm.T, origin="lower", cmap="hot",
                  extent=[xe[0], xe[-1], ye[0], ye[-1]],
                  aspect="equal", interpolation="bilinear")
        ax.set_xlabel("X  (µm)", fontsize=9)
        ax.set_ylabel("Y  (µm)", fontsize=9)
        if roi_mask is not None:
            H_px, W_px = roi_mask.shape
            ax.contour(
                np.linspace(0, W_px * pixel_size, W_px),
                np.linspace(0, H_px * pixel_size, H_px),
                roi_mask.astype(float), levels=[0.5],
                colors=["#58a6ff"], linewidths=[1.0], alpha=0.7)
    except Exception:
        pass
    sax(ax, "H", "Position Density Map")

    # I — Turning Angle Distribution
    # Plotted as a single LINE following the count of each |angle| bin,
    # using UNSIGNED magnitudes (|θ|) so the x-axis runs 0°–180°.
    # 0° = continued straight; 180° = full reversal; 90° = right-angle
    # deflection; the radial-distribution panel (O) shows the rotational
    # direction (sign) separately.
    ax = fig.add_subplot(gs[2, 2])
    if turning_angles is None or len(turning_angles) < 10:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=12)
    else:
        ta_unsigned = np.abs(np.asarray(turning_angles, dtype=float))
        _ta_bins = np.linspace(0, 180, 37)            # 5° bins
        _ta_centres = 0.5 * (_ta_bins[:-1] + _ta_bins[1:])
        _ta_counts, _ = np.histogram(ta_unsigned, bins=_ta_bins)
        # Normalise to relative frequency so the shape is comparable across
        # runs (and consistent with the Compare-mode panel).  Total track
        # count is already reported in the suptitle / Summary tab.
        _ta_freq = (_ta_counts / _ta_counts.sum()
                    if _ta_counts.sum() else _ta_counts)
        ax.plot(_ta_centres, _ta_freq, "-o",
                color=ACC, lw=2, ms=3, alpha=0.95)
        # Uniform-distribution reference line (1/N_bins)
        ax.axhline(1.0 / len(_ta_centres),
                   color=GRD, lw=0.6, ls=":", label="uniform")
        # Reference verticals: 90° (right-angle), 180° (full reversal)
        ax.axvline(90,  color=GRD, lw=0.8, ls="--")
        ax.axvline(180, color=GRD, lw=0.6, ls=":")
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xlabel("|Turning angle|  (°)", fontsize=9)
        ax.set_ylabel("Relative frequency", fontsize=9)
        ax.grid(True, ls=":", alpha=0.3)
        ax.legend(fontsize=7, frameon=False, loc="best")
    sax(ax, "I", "Turning Angle Distribution")

    # J — Mobile Fraction Over Time
    ax = fig.add_subplot(gs[3, 0])
    if mobile_frac_df is None or len(mobile_frac_df) < 2:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=12)
    else:
        ts  = mobile_frac_df["time_s"].values
        mf  = mobile_frac_df["mobile_fraction"].values * 100
        ax.plot(ts, mf, "o-", color=ACC, lw=2, ms=5)
        ax.fill_between(ts, 0, mf, alpha=0.2, color=ACC)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Mobile fraction (%)", fontsize=9)
        ax.grid(True, ls=":", alpha=0.3)
    sax(ax, "J", "Mobile Fraction Over Time")

    # K — Jump Distance Distribution (spans cols 1–2)
    ax = fig.add_subplot(gs[3, 1:])
    if _has_jdd:
        _jdd_colors = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff"]

        r_max_plot = np.percentile(jdd["jumps"], 99.5)
        bins = np.linspace(0, r_max_plot, 60)
        ax.hist(jdd["jumps"], bins=bins, density=True,
                color="#8b949e", alpha=0.45, edgecolor="none",
                label=f"Observed  (n={jdd['n_jumps']:,})")

        _comp_labels = ["Slow", "Medium", "Fast"]
        for k, (pdf_k, D_k, f_k) in enumerate(
                zip(jdd["pdfs"], jdd["D_values"], jdd["fractions"])):
            lbl = (f"{_comp_labels[k]}  D={D_k:.4f} µm²/s  "
                   f"({f_k*100:.1f}%)")
            ax.plot(jdd["r_range"], pdf_k,
                    color=_jdd_colors[k], lw=2, label=lbl)

        ax.plot(jdd["r_range"], jdd["pdf_total"],
                color=TXT, lw=2.5, ls="--", label="Total fit")
        ax.set_xlabel("Jump distance  (µm)", fontsize=9)
        ax.set_ylabel("Probability density", fontsize=9)
        ax.set_xlim(0, r_max_plot)
        ax.set_ylim(bottom=0)
        ax.grid(True, ls=":", alpha=0.3)
        ax.legend(fontsize=8, framealpha=0.6,
                  facecolor=PNL, edgecolor=GRD, labelcolor=TXT,
                  loc="upper right")
        sax(ax, "K",
            f"Jump Distance Distribution  "
            f"({jdd['n_components']}-population fit  |  "
            f"{jdd['n_jumps']:,} jumps)")
    else:
        ax.text(0.5, 0.5, "JDD not computed", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=12)
        sax(ax, "K", "Jump Distance Distribution")

    # L — Cluster Map
    ax = fig.add_subplot(gs[4, 0])
    if cluster_labels is not None and cluster_locs is not None and len(cluster_locs) > 0:
        xy_um = cluster_locs  # already in µm, subsampled to match labels
        noise = cluster_labels == -1
        if noise.any():
            ax.scatter(xy_um[noise, 0], xy_um[noise, 1],
                       s=0.5, c="#444", alpha=0.3, linewidths=0, rasterized=True)
        clustered = ~noise
        if clustered.any():
            n_c = max(cluster_labels.max() + 1, 1)
            cmap_c = plt.cm.get_cmap("tab20", n_c)
            ax.scatter(xy_um[clustered, 0], xy_um[clustered, 1],
                       s=1.5, c=cluster_labels[clustered], cmap=cmap_c,
                       alpha=0.7, linewidths=0, rasterized=True,
                       vmin=0, vmax=n_c - 1)
        ax.set_xlabel("X  (µm)", fontsize=9)
        ax.set_ylabel("Y  (µm)", fontsize=9)
        n_shown = int(cluster_labels.max()) + 1 if cluster_labels.max() >= 0 else 0
        ax.text(0.02, 0.98, f"n={n_shown} clusters",
                transform=ax.transAxes, fontsize=8, color=TXT, va="top")
    else:
        ax.text(0.5, 0.5, "Cluster analysis\nnot computed",
                transform=ax.transAxes, ha="center", va="center", color=TXT, fontsize=10)
    sax(ax, "L", "Cluster Map  (DBSCAN)")

    # M — Dwell Time Distribution
    ax = fig.add_subplot(gs[4, 1])
    if dwell_df is not None and len(dwell_df) >= 5:
        dt_vals = dwell_df["dwell_time_s"].values
        ax.hist(dt_vals, bins=30, color=ACC, alpha=0.75, edgecolor="none", density=True)
        if np.isfinite(dwell_tau):
            t_fit = np.linspace(0, dt_vals.max(), 200)
            ax.plot(t_fit, (1/dwell_tau) * np.exp(-t_fit / dwell_tau),
                    "--", color="#f78166", lw=2,
                    label=f"τ = {dwell_tau:.2f} s")
            ax.legend(fontsize=8, framealpha=0.6, facecolor=PNL,
                      edgecolor=GRD, labelcolor=TXT)
        ax.set_xlabel("Dwell time  (s)", fontsize=9)
        ax.set_ylabel("Probability density", fontsize=9)
        ax.grid(True, ls=":", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Insufficient data\n(need confined/immobile tracks)",
                transform=ax.transAxes, ha="center", va="center", color=TXT, fontsize=10)
    sax(ax, "M", "Dwell Time Distribution")

    # N — MSS Slope Distribution
    ax = fig.add_subplot(gs[4, 2])
    if "mss_slope" in diff_df.columns and diff_df["mss_slope"].notna().sum() >= 5:
        ms = diff_df["mss_slope"].dropna()
        ms = ms[ms.between(-0.5, 1.5)]
        bins = np.linspace(ms.min(), ms.max(), 40)
        for m in MORD:
            sub = diff_df[(diff_df["motion"] == m) & diff_df["mss_slope"].notna()]
            sub = sub[sub["mss_slope"].between(-0.5, 1.5)]
            if len(sub):
                ax.hist(sub["mss_slope"], bins=bins, color=MC[m],
                        alpha=0.7, label=m, edgecolor="none")
        for xv, lb, ls_ in [(0.25, "Confined", ":"), (0.5, "Brownian", "--"), (0.75, "Directed", ":")]:
            ax.axvline(xv, color=GRD, ls=ls_, lw=1.2, label=lb)
        ax.set_xlabel("MSS slope  (ν)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=7, framealpha=0.6, facecolor=PNL, edgecolor=GRD, labelcolor=TXT)
        ax.grid(True, ls=":", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "MSS not computed\n(tracks too short)",
                transform=ax.transAxes, ha="center", va="center", color=TXT, fontsize=10)
    sax(ax, "N", "Moment Scaling Spectrum  (MSS slope)")

    # O — Radial Distribution of turning angles (polar)
    # A polar histogram of signed turning angles, oriented so 0° (straight
    # ahead) is at the top and positive angles sweep CLOCKWISE around to the
    # right (i.e. right hemisphere = positive turns, left hemisphere =
    # negative turns).  The bars radiate outward; their angular position is
    # the turning direction, their height the relative frequency.  Uniform
    # circle = Brownian motion; lobe at 0° = directional persistence; lobe
    # at ±180° = back-tracking / confinement.
    # Placed at the centre column of row 5 so it sits visually balanced
    # rather than pinned to a corner.
    ax = fig.add_subplot(gs[5, 1], projection="polar")
    if turning_angles is None or len(turning_angles) < 10:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ta_arr = np.asarray(turning_angles, dtype=float)
        is_signed = bool(np.any(ta_arr < -1e-3))
        print(f"  Radial-dist input: n={len(ta_arr):,}  "
              f"signed={is_signed}  "
              f"pos={int((ta_arr>0).sum()):,}  neg={int((ta_arr<0).sum()):,}  "
              f"min={ta_arr.min():.1f}°  max={ta_arr.max():.1f}°")
        if not is_signed:
            ta_arr = np.concatenate([ta_arr, -ta_arr])
        # CRITICAL: matplotlib polar's ax.bar() does NOT render correctly
        # when theta values are in (-π, +π].  Half the bars (the side with
        # negative theta after applying set_theta_direction) silently fail
        # to draw, producing only a half-circle of bars.
        # Empirical fix: shift the angles to [0, 2π) before histogramming.
        # The xticks are then placed at positive-only angles too, but
        # *labelled* with the signed values the user expects.
        angles_rad = np.mod(np.deg2rad(ta_arr), 2 * np.pi)
        n_bins = 36
        bins   = np.linspace(0, 2 * np.pi, n_bins + 1)
        counts, edges = np.histogram(angles_rad, bins=bins, density=True)
        theta = 0.5 * (edges[:-1] + edges[1:])
        width = bins[1] - bins[0]
        ax.bar(theta, counts, width=width * 0.95, bottom=0.0,
               color=ACC, alpha=0.75, edgecolor=GRD, linewidth=0.5)
        ax.set_theta_zero_location("N")     # 0° at the top
        ax.set_theta_direction(-1)          # clockwise positive (right = +)
        # xticks at 0°, 45°, ..., 315° (positive only); labels show signed
        # equivalents so the reader still sees "-45°" on the left, etc.
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(["0°", "+45°", "+90°", "+135°", "±180°",
                            "−135°", "−90°", "−45°"], fontsize=8)
        # Hide the radial-axis numeric labels.
        ax.set_yticklabels([])
        ax.tick_params(axis="y", which="both", left=False)
        ax.grid(True, ls=":", alpha=0.4)
    sax(ax, "O", "Radial Distribution  (signed turning angles)")

    md = diff_df["D"].dropna().median()
    ma = diff_df["alpha"].dropna().median()
    fig.suptitle(
        f"FIREFLY Analysis  |  {diff_df.shape[0]:,} trajectories  |  "
        f"Median D = {md:.4f} um2/s  |  Median alpha = {ma:.2f}",
        fontsize=13,color=TXT,y=0.97,fontweight="bold")

    import io as _io
    from matplotlib.transforms import Bbox as _Bbox

    from PIL import Image as _PILImage

    # Render individual panels WITHOUT letter labels.  Each panel costs a
    # full-figure savefig(), so only do this for the panels actually
    # requested — and skip the whole block (and its two extra draws) when
    # none are wanted.
    panel_images = {}
    _render_panels = (want_panels is None) or bool(want_panels)
    if _render_panels:
        for _txt in _letter_artists:
            _txt.set_visible(False)
        fig.canvas.draw()
        _renderer = fig.canvas.get_renderer()
        _pad_px   = fig.dpi * 0.12
        for _ltr, _pax in _panels:
            if want_panels is not None and _ltr not in want_panels:
                continue
            _bbox = _pax.get_tightbbox(_renderer)
            if _bbox is None:
                continue
            _bbox_pad = _Bbox([[_bbox.x0 - _pad_px, _bbox.y0 - _pad_px],
                                [_bbox.x1 + _pad_px, _bbox.y1 + _pad_px]])
            _bbox_in  = _bbox_pad.transformed(fig.dpi_scale_trans.inverted())
            _pbuf = _io.BytesIO()
            fig.savefig(_pbuf, format="png", dpi=150, bbox_inches=_bbox_in,
                        facecolor=fig.get_facecolor())
            _pbuf.seek(0)
            panel_images[_ltr] = _PILImage.open(_pbuf).copy()
            _pbuf.close()

        # Restore letter labels for the combined figure
        for _txt in _letter_artists:
            _txt.set_visible(True)
        fig.canvas.draw()
    _buf = _io.BytesIO()
    fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    _buf.seek(0)
    combined_pil = _PILImage.open(_buf).copy()
    _buf.close()

    # Save to disk only if output_path explicitly provided (CLI / legacy callers)
    if output_path:
        combined_pil.save(output_path, dpi=(150, 150))
        print(f"  Figure -> {output_path}")
        _pdf = os.path.splitext(output_path)[0] + ".pdf"
        fig.savefig(_pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Figure (PDF) -> {_pdf}")

    pdf_bytes = None
    if return_pdf_bytes:
        try:
            _pdfbuf = _io.BytesIO()
            fig.savefig(_pdfbuf, format="pdf", bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            pdf_bytes = _pdfbuf.getvalue()
            _pdfbuf.close()
        except Exception as _exc:
            print(f"  WARN: PDF render failed: {_exc}")

    plt.close(fig)
    print("  Figure rendered.")
    return {
        "combined":     combined_pil,
        "panels":       panel_images,
        "panel_titles": {ltr: ax.get_title().strip() for ltr, ax in _panels},
        "pdf_bytes":    pdf_bytes,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="FIREFLY — Fluorescence Inference & Reconstruction Engine "
                    "(CZI / TIF, optimised)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("--pixel-size",       type=float, default=None)
    p.add_argument("--frame-interval",   type=float, default=None)
    p.add_argument("--diameter",         type=int,   default=7)
    p.add_argument("--minmass",          type=float, default=None)
    p.add_argument("--search-range",     type=float, default=5)
    p.add_argument("--memory",           type=int,   default=3)
    p.add_argument("--min-track-length", type=int,   default=5)
    p.add_argument("--max-lagtime",      type=int,   default=20)
    p.add_argument("--bg-method",        default="uniform_filter",
                   choices=["uniform_filter","rolling_ball"])
    p.add_argument("--bg-radius",        type=float, default=50)
    p.add_argument("--workers",          type=int,   default=N_CPUS)
    p.add_argument("--chunk-size",       type=int,   default=500)
    p.add_argument("--channel",          type=int,   default=0)
    p.add_argument("--output-dir",       default=None)
    p.add_argument("--roi-threshold",      type=float, default=None,
                   help="Manual intensity threshold for ROI mask on [0,1]. "
                        "If omitted with --roi-auto, threshold is determined "
                        "automatically. Omit both to process the full frame.")
    p.add_argument("--roi-auto",           action="store_true", default=False,
                   help="Automatically determine ROI threshold from the data. "
                        "Uses --roi-auto-method to select the algorithm.")
    p.add_argument("--roi-auto-method",    default="auto",
                   choices=["auto", "otsu", "li", "triangle"],
                   help="Algorithm for automatic ROI thresholding. "
                        "auto     = picks best method for sptPALM (default). "
                        "otsu     = maximises inter-class variance. "
                        "li       = minimises cross-entropy (sparse cells). "
                        "triangle = best for large dark backgrounds.")
    p.add_argument("--roi-mode",           default="mean",
                   choices=["mean", "perframe"],
                   help="ROI masking mode. "
                        "mean     = one mask from mean projection (default). "
                        "perframe = separate mask computed per frame.")
    return p.parse_args()


def main():
    args    = parse_args()
    t_start = time.perf_counter()

    if not os.path.isfile(args.input):
        sys.exit(f"ERROR: File not found: {args.input}")

    stem    = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "="*67)
    print("  sptPALM Analysis Pipeline  --  Zeiss Elyra  --  By Jacob Levers")
    print("="*67)
    print(f"  CPU cores available : {N_CPUS}  |  Using: {args.workers}")

    # 1 — Load
    print("\n[1/6] Loading file")
    stack, meta_px, meta_fi = load_file(args.input, channel=args.channel)
    n_frames = len(stack)

    pixel_size = args.pixel_size or meta_px
    if pixel_size is None:
        print("  WARNING: Pixel size not in metadata. Using 0.104 um/px.")
        print("  (Override with --pixel-size)")
        pixel_size = 0.104
    else:
        src = "command line" if args.pixel_size else "CZI metadata"
        print(f"  Pixel size     : {pixel_size} um/px  [{src}]")

    frame_interval = args.frame_interval or meta_fi
    if frame_interval is None:
        print("  WARNING: Frame interval not in metadata. Using 0.05 s.")
        print("  (Override with --frame-interval)")
        frame_interval = 0.05
    else:
        src = "command line" if args.frame_interval else "CZI metadata"
        print(f"  Frame interval : {frame_interval} s/frame  [{src}]")

    print(f"  Total frames   : {n_frames:,}")
    print(f"  Output dir     : {out_dir}")

    # 2 — Preprocess
    print("\n[2/6] Preprocessing")
    stack_pp = preprocess_stack(stack, bg_radius=args.bg_radius,
                                bg_method=args.bg_method,
                                workers=args.workers)

    if args.minmass is None:
        sample = stack_pp[min(5, n_frames-1)]
        _peak = float(np.percentile(sample, 99))
        # See _fast_preprocess_and_localise for the rationale on the d²/8 factor.
        args.minmass = float(_peak * (args.diameter ** 2) / 8.0)
        print(f"  Auto minmass: {args.minmass:.4f}  "
              f"(from 99th-pct peak {_peak:.4f} × d²/8)")

    # 2b — ROI mask (optional)
    roi_mask = None
    use_roi  = (args.roi_threshold is not None) or args.roi_auto
    if use_roi:
        manual_thresh = args.roi_threshold  # None = auto
        auto_method   = args.roi_auto_method if args.roi_auto else None
        if manual_thresh is not None:
            mode_str = f"threshold={manual_thresh}, mode={args.roi_mode}"
        else:
            mode_str = f"auto-threshold ({args.roi_auto_method}), mode={args.roi_mode}"
        print(f"\n[2b/6] Building ROI mask  ({mode_str})")
        roi_preview = os.path.join(out_dir, f"{stem}_roi_mask.png")
        roi_mask = build_roi_mask(
            stack_pp,
            threshold=manual_thresh,
            mode=args.roi_mode,
            threshold_method=args.roi_auto_method if args.roi_auto else "auto",
            save_path=roi_preview)
    else:
        print("  ROI: disabled  "
              "(use --roi-auto for automatic, or --roi-threshold 0.15 for manual)")

    # 3 — Localise
    print("\n[3/6] Localisation")
    locs = localise_particles(stack_pp, diameter=args.diameter,
                              minmass=args.minmass,
                              workers=args.workers,
                              chunk_size=args.chunk_size)
    if len(locs) == 0:
        sys.exit("ERROR: No particles found. Try adding --minmass 0.05")

    if roi_mask is not None:
        locs = apply_roi_mask(locs, roi_mask)
        if len(locs) == 0:
            sys.exit("ERROR: No localisations inside ROI. "
                     "Lower --roi-threshold or remove it.")

    # 4 — Link
    print("\n[4/6] Linking trajectories")
    tracks = link_trajectories(locs, search_range=args.search_range,
                               memory=args.memory,
                               min_len=args.min_track_length)
    if tracks["particle"].nunique() == 0:
        sys.exit("ERROR: No trajectories found. Lower --min-track-length.")

    # 5 — MSD + diffusion (single parallel pass — no tp.imsd)
    print("\n[5/6] MSD & diffusion fitting")
    imsd_df, emsd_df, diff_df = compute_msd_and_fit(
        tracks, pixel_size, frame_interval,
        max_lagtime=args.max_lagtime, workers=args.workers)

    # 5b — JDD
    print("\n[5b/6] Jump Distance Distribution")
    jdd = compute_jdd(tracks, pixel_size, frame_interval, n_components=2)
    if jdd:
        print(f"  Jumps: {jdd['n_jumps']:,}")
        for k, (D, f) in enumerate(zip(jdd["D_values"], jdd["fractions"])):
            print(f"  Population {k+1}: D={D:.4f} um2/s  fraction={f*100:.1f}%")
    else:
        print("  Too few jumps to fit JDD.")

    # 6 — Save
    print("\n[6/6] Saving outputs")
    for df, suffix in [(locs,"localisations"), (tracks,"trajectories"),
                       (diff_df,"diffusion_summary")]:
        path = os.path.join(out_dir, f"{stem}_{suffix}.csv")
        df.to_csv(path, index=False)
        print(f"  {suffix:<25} -> {path}")

    emsd_out  = emsd_df.to_frame("msd_um2").reset_index(names="lag_frame")
    emsd_path = os.path.join(out_dir, f"{stem}_ensemble_msd.csv")
    emsd_out.to_csv(emsd_path, index=False)
    print(f"  ensemble_msd              -> {emsd_path}")

    fig_path = os.path.join(out_dir, f"{stem}_sptpalm_figure.png")
    make_figure(stack, tracks, imsd_df, emsd_df, diff_df,
                pixel_size, frame_interval, fig_path,
                roi_mask=roi_mask, jdd=jdd,
                turning_angles=None, mobile_frac_df=None,
                cluster_labels=None, cluster_locs=None,
                dwell_df=None, dwell_tau=None)

    # Summary
    total = time.perf_counter() - t_start
    print("\n" + "="*67)
    print("  RESULTS SUMMARY")
    print("="*67)
    print(f"  Raw localisations : {len(locs):>8,}")
    print(f"  Final trajectories: {tracks['particle'].nunique():>8,}")
    mc_ = diff_df["motion"].value_counts()
    for m in MORD:
        cnt = mc_.get(m, 0)
        print(f"    {m:<12}  {cnt:>6,}  ({100*cnt/max(len(diff_df),1):.1f}%)")
    print(f"\n  Median D  : {diff_df['D'].median():.5f} um2/s")
    print(f"  Mean D    : {diff_df['D'].mean():.5f} um2/s")
    print(f"  Median a  : {diff_df['alpha'].median():.3f}")
    print(f"\n  Total time: {total:.1f}s  ({total/60:.1f} min)")
    print("="*67)
    print(f"\n  Done! Results in: {out_dir}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARISON  —  group A vs group B over multiple analysis output folders
# ══════════════════════════════════════════════════════════════════════════════
#
# A "group" is a list of analysis output folders.  For each folder we re-load
# the per-experiment summary (MSD curve, D values, motion classes, etc.) and
# compute scalar metrics (AUC, mob/immob ratio, mean track length).  We then
# render a multi-panel figure overlaying the two groups, with scatter dots
# per replicate and t-test significance stars on bar charts when n≥2 each.
# Layout matches the lab's Pre/Post style: MSD curve overlay, AUC bar chart,
# LogD frequency distribution, mobile/immobile ratio bar chart, motion class
# fractions, track length distribution, JDD, dwell time CDF, turning angles.

def _find_stem(data_dir):
    """Find the experiment stem from filenames like {stem}_params.json or
    {stem}_diffusion_summary.csv inside an analysis output folder's data/ dir."""
    for f in sorted(os.listdir(data_dir)):
        if f.endswith("_params.json"):
            return f[:-len("_params.json")]
    for f in sorted(os.listdir(data_dir)):
        if f.endswith("_diffusion_summary.csv"):
            return f[:-len("_diffusion_summary.csv")]
    raise FileNotFoundError(f"No analysis CSVs found in {data_dir}")


def _is_palmtracer_folder(folder):
    """Return True if `folder` contains raw PALM-Tracer output."""
    try:
        names = os.listdir(folder)
    except OSError:
        return False
    # PALM-Tracer files have no stem prefix (e.g. 'locPALMTracer.txt')
    has_loc = any(n.lower() == "locpalmtracer.txt" or n.lower() == "locpalmtracer.csv"
                  for n in names)
    has_trc = any(n.lower() == "trcpalmtracer.txt" or n.lower() == "trcpalmtracer.csv"
                  for n in names)
    return has_loc and has_trc


def _read_palmtracer_table(path, header_lines):
    """Read a PALM-Tracer file (tab- or comma-separated), skipping comment /
    metadata rows.  `header_lines` is the number of non-data leading rows."""
    # PALM-Tracer's reference files are TSV; FIREFLY-emitted ones are CSV.
    # Sniff the separator from the first data line.
    with open(path, "r") as fh:
        for _ in range(header_lines):
            fh.readline()
        first = fh.readline()
    sep = "\t" if "\t" in first and first.count("\t") >= first.count(",") else ","
    return pd.read_csv(path, sep=sep, header=None, comment="#",
                       skiprows=header_lines, engine="python")


def load_summary_from_palmtracer(folder):
    """
    Read a raw PALM-Tracer output folder and return the same dict shape as
    `load_summary_from_folder` so the Compare tab can treat it identically.

    PALM-Tracer does not store FIREFLY-specific quantities (alpha, motion
    class, dwell times, turning angles, JDD, mobile fraction, Rg) — these are
    re-derived on the fly from the imported trajectories using the same
    pipeline functions FIREFLY normally runs.
    """
    # ── Locate the six PALM-Tracer files (tab or csv) ────────────────────
    def _pick(*candidates):
        for c in candidates:
            p = os.path.join(folder, c)
            if os.path.isfile(p):
                return p
        return None

    loc_path = _pick("locPALMTracer.txt", "locPALMTracer.csv")
    trc_path = _pick("trcPALMTracer.txt", "trcPALMTracer.csv")
    d_path   = _pick("trcPALMTracer-AllROI-D.txt", "trcPALMTracer-AllROI-D.csv",
                     "trcPALMTracer-1-D.txt",     "trcPALMTracer-1-D.csv")
    msd_path = _pick("trcPALMTracer-AllROI-MSD.txt", "trcPALMTracer-AllROI-MSD.csv",
                     "trcPALMTracer-1-MSD.txt",     "trcPALMTracer-1-MSD.csv")

    if not (loc_path and trc_path):
        raise FileNotFoundError(f"PALM-Tracer files not found in {folder}")

    # ── Parse loc / trc metadata header (line 2 contains values) ─────────
    pixel_size_um    = 0.106
    frame_interval_s = 0.02
    width = height = n_frames = 0
    try:
        with open(loc_path, "r") as fh:
            _hdr_names  = fh.readline().rstrip("\n").replace(",", "\t").split("\t")
            _hdr_values = fh.readline().rstrip("\n").replace(",", "\t").split("\t")
        meta = {k.strip(): v.strip() for k, v in zip(_hdr_names, _hdr_values)}
        pixel_size_um    = float(meta.get("Pixel_Size(um)", pixel_size_um))
        frame_interval_s = float(meta.get("Frame_Duration(s)", frame_interval_s))
        width    = int(float(meta.get("Width",  0) or 0))
        height   = int(float(meta.get("Height", 0) or 0))
        n_frames = int(float(meta.get("nb_Planes", 0) or 0))
    except Exception:
        pass

    # ── Localisations ────────────────────────────────────────────────────
    # Header rows in loc/trc files: metadata-names, metadata-values, column-names
    loc_df = _read_palmtracer_table(loc_path, header_lines=3)
    loc_df.columns = ["id", "Plane", "Index", "Channel", "Integrated_Intensity",
                      "CentroidX_px", "CentroidY_px", "SigmaX_px", "SigmaY_px",
                      "Angle_rad", "MSE_Gauss", "CentroidZ_um", "MSE_Z_um",
                      "Pair_Distance_px"][:loc_df.shape[1]]
    locs = pd.DataFrame({
        "x":     loc_df["CentroidX_px"].astype(float).values,
        "y":     loc_df["CentroidY_px"].astype(float).values,
        "frame": (loc_df["Plane"].astype(int).values - 1),   # 1-based → 0-based
        "mass":  loc_df["Integrated_Intensity"].astype(float).values,
    })

    # ── Trajectories ─────────────────────────────────────────────────────
    trc_df = _read_palmtracer_table(trc_path, header_lines=3)
    trc_df.columns = ["Track", "Plane", "CentroidX_px", "CentroidY_px",
                      "CentroidZ_um", "Integrated_Intensity", "id",
                      "Pair_Distance_px"][:trc_df.shape[1]]
    tracks = pd.DataFrame({
        "particle": trc_df["Track"].astype(int).values,
        "frame":    trc_df["Plane"].astype(int).values - 1,
        "x":        trc_df["CentroidX_px"].astype(float).values,
        "y":        trc_df["CentroidY_px"].astype(float).values,
        "mass":     trc_df["Integrated_Intensity"].astype(float).values,
    }).sort_values(["particle", "frame"]).reset_index(drop=True)

    # ── Re-derive D, alpha, motion via FIREFLY's own pipeline ────────────
    # This guarantees the Compare tab sees the same column names and
    # identical statistics it would for a native FIREFLY run.
    imsd_df, emsd_series, diff_df = compute_msd_and_fit(
        tracks, pixel_size_um, frame_interval_s, max_lagtime=20, n_fit=5)

    emsd_df = (emsd_series.to_frame("msd_um2")
                          .reset_index(names="lag_frame"))

    # FIREFLY-only metrics — re-derive on the fly
    try:
        jdd = compute_jdd(tracks, pixel_size_um, frame_interval_s)
    except Exception:
        jdd = None
    try:
        dwell_df, _ = compute_dwell_times(tracks, diff_df, frame_interval_s)
    except Exception:
        dwell_df = None
    try:
        ta_deg = compute_turning_angles(tracks)
    except Exception:
        ta_deg = None
    try:
        mobile_frac_df = compute_mobile_fraction_over_time(
            tracks, diff_df, frame_interval_s)
    except Exception:
        mobile_frac_df = None

    stem = os.path.basename(folder.rstrip(os.sep)) or "palmtracer_run"
    if stem.lower().endswith(".pt"):
        stem = stem[:-3]

    # ── Cache the recomputed FIREFLY-only metrics next to the PALM-Tracer
    # files so re-opening this folder in the Compare tab is instant.  The
    # cache lives in <folder>/firefly_extras/ and uses FIREFLY's native
    # CSV/JSON schema.
    try:
        import json as _json
        extras_dir = os.path.join(folder, "firefly_extras")
        os.makedirs(extras_dir, exist_ok=True)
        diff_df.to_csv(
            os.path.join(extras_dir, f"{stem}_diffusion_summary.csv"), index=False)
        tracks.to_csv(
            os.path.join(extras_dir, f"{stem}_trajectories.csv"), index=False)
        locs.to_csv(
            os.path.join(extras_dir, f"{stem}_localisations.csv"), index=False)
        emsd_df.to_csv(
            os.path.join(extras_dir, f"{stem}_ensemble_msd.csv"), index=False)
        with open(os.path.join(extras_dir, f"{stem}_params.json"), "w") as _fp:
            _json.dump({
                "stem":             stem,
                "pixel_size_um":    pixel_size_um,
                "frame_interval_s": frame_interval_s,
                "n_localisations":  int(len(locs)),
                "n_tracks":         int(diff_df.shape[0]),
                "n_frames":         int(n_frames),
                "width":            width,
                "height":           height,
                "source":           "palmtracer (re-derived)",
            }, _fp, indent=2)
        if jdd:
            with open(os.path.join(extras_dir, f"{stem}_jdd.json"), "w") as _fp:
                _json.dump(_to_jsonable(jdd) if "_to_jsonable" in globals() else jdd,
                           _fp, indent=2, default=str)
        if dwell_df is not None and len(dwell_df):
            dwell_df.to_csv(
                os.path.join(extras_dir, f"{stem}_dwell_times.csv"), index=False)
        if ta_deg is not None and len(ta_deg):
            pd.DataFrame({"turning_angle_deg": ta_deg}).to_csv(
                os.path.join(extras_dir, f"{stem}_turning_angles.csv"), index=False)
        if mobile_frac_df is not None and len(mobile_frac_df):
            mobile_frac_df.to_csv(
                os.path.join(extras_dir, f"{stem}_mobile_fraction.csv"), index=False)
    except Exception:
        # Caching is best-effort — never fail the load over a write error
        pass

    return {
        "folder":     folder,
        "stem":       stem,
        "data_dir":   folder,
        "source":     "palmtracer",
        "params": {
            "stem":             stem,
            "pixel_size_um":    pixel_size_um,
            "frame_interval_s": frame_interval_s,
            "n_localisations":  int(len(locs)),
            "n_tracks":         int(diff_df.shape[0]),
            "n_frames":         int(n_frames),
            "width":            width,
            "height":           height,
        },
        "ensemble_msd":          emsd_df,
        "diffusion":             diff_df,
        "tracks":                tracks,
        "jdd":                   jdd,
        "dwell_times":           dwell_df,
        "turning_angles":        ta_deg if ta_deg is not None else None,
        "turning_angles_signed": True,
    }


def load_summary_from_folder(folder):
    """Load all per-experiment summary data from one analysis output folder.

    Accepts any of:
      <run_dir>/                       (containing firefly_extras/ and data/)
      <run_dir>/firefly_extras/        (the FIREFLY-extras directory itself)
      <palm_tracer_folder>/            (auto-detected, re-derived on load)
      <run_dir>/data/                  (PALM-Tracer CSVs from a FIREFLY run)
    """
    import json

    # ── Resolve which directory holds the FIREFLY-native CSVs ────────────
    # 1) <folder>/firefly_extras  (folder is the run dir)
    if os.path.isdir(os.path.join(folder, "firefly_extras")):
        data_dir = os.path.join(folder, "firefly_extras")
    # 2) folder is itself the firefly_extras dir
    elif os.path.basename(folder.rstrip(os.sep)) == "firefly_extras":
        data_dir = folder
    # 3) folder is a PALM-Tracer folder (raw or FIREFLY-emitted CSV mirrors)
    elif _is_palmtracer_folder(folder):
        return load_summary_from_palmtracer(folder)
    # 4) folder is a run dir whose `data/` holds PALM-Tracer CSVs
    elif (os.path.isdir(os.path.join(folder, "data"))
          and _is_palmtracer_folder(os.path.join(folder, "data"))):
        return load_summary_from_palmtracer(os.path.join(folder, "data"))
    else:
        raise FileNotFoundError(
            f"No firefly_extras/ directory and no PALM-Tracer files in {folder}")

    stem = _find_stem(data_dir)
    s = {"folder": folder, "stem": stem, "data_dir": data_dir}

    # Params (frame interval, pixel size, ...)
    params_path = os.path.join(data_dir, f"{stem}_params.json")
    if os.path.isfile(params_path):
        with open(params_path) as f:
            s["params"] = json.load(f)
    else:
        s["params"] = {"pixel_size_um": 0.104, "frame_interval_s": 0.05}

    # Ensemble MSD
    msd_path = os.path.join(data_dir, f"{stem}_ensemble_msd.csv")
    if os.path.isfile(msd_path):
        s["ensemble_msd"] = pd.read_csv(msd_path)
    else:
        s["ensemble_msd"] = None

    # Diffusion summary (per-track D, alpha, motion_class)
    diff_path = os.path.join(data_dir, f"{stem}_diffusion_summary.csv")
    if os.path.isfile(diff_path):
        s["diffusion"] = pd.read_csv(diff_path)
    else:
        s["diffusion"] = None

    # Trajectories (for track length distribution)
    tr_path = os.path.join(data_dir, f"{stem}_trajectories.csv")
    if os.path.isfile(tr_path):
        s["tracks"] = pd.read_csv(tr_path)
    else:
        s["tracks"] = None

    # JDD
    jdd_path = os.path.join(data_dir, f"{stem}_jdd.json")
    if os.path.isfile(jdd_path):
        with open(jdd_path) as f:
            s["jdd"] = json.load(f)
    else:
        s["jdd"] = None

    # Dwell times
    dwell_path = os.path.join(data_dir, f"{stem}_dwell_times.csv")
    if os.path.isfile(dwell_path):
        s["dwell_times"] = pd.read_csv(dwell_path)
    else:
        s["dwell_times"] = None

    # Turning angles — signed degrees (-180..+180°)
    ta_path = os.path.join(data_dir, f"{stem}_turning_angles.csv")
    if os.path.isfile(ta_path):
        _ta_df = pd.read_csv(ta_path)
        s["turning_angles"]        = _ta_df["turning_angle_deg"].values
        s["turning_angles_signed"] = True
    else:
        s["turning_angles"]        = None
        s["turning_angles_signed"] = False

    return s


def save_palmtracer_csvs(out_dir, stem, locs, tracks, diff_df, imsd_df,
                         pixel_size_um, frame_interval_s,
                         width=None, height=None, n_frames=None,
                         mobile_D_threshold=None):
    """
    Emit PALM-Tracer-compatible CSV files alongside FIREFLY's native outputs.

    Files written (all comma-separated, written into `out_dir`):
        <stem>_locPALMTracer.csv              (one row per localisation)
        <stem>_trcPALMTracer.csv              (one row per trajectory plane)
        <stem>_trcPALMTracer-1-D.csv          (per-track D, MSD(0), MSE, LogD)
        <stem>_trcPALMTracer-1-MSD.csv        (per-track MSD curve, jagged)
        <stem>_trcPALMTracer-AllROI-D.csv     (per-track D summary)
        <stem>_trcPALMTracer-AllROI-MSD.csv   (per-track MSD curve, jagged)

    Column ordering, naming and unit conventions follow PALM-Tracer
    (Bordeaux Imaging Center).  ROI is hard-coded to 1 (FIREFLY does not
    sub-ROI tracks).  Fields FIREFLY does not measure (SigmaX/Y, Angle,
    MSE(Gauss), CentroidZ, MSE_Z, Pair_Distance) are filled with the
    PALM-Tracer "unused" sentinels (-1 or 0).
    """
    import csv as _csv
    import numpy as _np
    import pandas as _pd
    import os as _os

    if mobile_D_threshold is None:
        mobile_D_threshold = MOBILE_D_THRESHOLD_DEFAULT

    width    = int(width)    if width    is not None else 0
    height   = int(height)   if height   is not None else 0
    n_frames = int(n_frames) if n_frames is not None else int(
        max(locs["frame"].max() + 1, tracks["frame"].max() + 1))

    print(f"  PALM-Tracer: {len(locs):,} locs, {len(diff_df):,} tracks, "
          f"imsd_df shape {imsd_df.shape if imsd_df is not None else None}")

    # ── 1. locPALMTracer.csv ─────────────────────────────────────────────
    n_loc = len(locs)
    loc_path = _os.path.join(out_dir, f"{stem}_locPALMTracer.csv")
    with open(loc_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["Width", "Height", "nb_Planes", "nb_Points",
                    "Pixel_Size(um)", "Frame_Duration(s)",
                    "Gaussian_Fit", "Spectral"])
        w.writerow([width, height, n_frames, n_loc,
                    pixel_size_um, frame_interval_s, "None", "False"])
        w.writerow(["id", "Plane", "Index", "Channel",
                    "Integrated_Intensity",
                    "CentroidX(px)", "CentroidY(px)",
                    "SigmaX(px)", "SigmaY(px)", "Angle(rad)", "MSE(Gauss)",
                    "CentroidZ(um)", "MSE_Z(um)", "Pair_Distance(px)"])
        frames_l = locs["frame"].values
        xs       = locs["x"].values
        ys       = locs["y"].values
        mass     = (locs["mass"].values if "mass" in locs.columns
                    else _np.zeros(n_loc))
        for i in range(n_loc):
            w.writerow([i + 1, int(frames_l[i]) + 1, i + 1, -1,
                        float(mass[i]),
                        float(xs[i]), float(ys[i]),
                        0.0, 0.0, 0.0, 0.0,
                        -1.0, -1.0, 0.0])

    # ── 2. trcPALMTracer.csv ─────────────────────────────────────────────
    tr_path = _os.path.join(out_dir, f"{stem}_trcPALMTracer.csv")
    # Re-number particles 1..n in PALM-Tracer style
    pid_order  = (diff_df["particle"].values if "particle" in diff_df.columns
                  else sorted(tracks["particle"].unique()))
    pid_to_new = {int(p): i + 1 for i, p in enumerate(pid_order)}
    n_tracks   = len(pid_to_new)

    with open(tr_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["Width", "Height", "nb_Planes", "nb_Tracks",
                    "Pixel_Size(um)", "Frame_Duration(s)",
                    "Gaussian_Fit", "Spectral"])
        w.writerow([width, height, n_frames, n_tracks,
                    pixel_size_um, frame_interval_s, "None", "False"])
        w.writerow(["Track", "Plane", "CentroidX(px)", "CentroidY(px)",
                    "CentroidZ(um)", "Integrated_Intensity", "id",
                    "Pair_Distance(px)"])
        # trackpy.link sets `frame` as the index AND keeps it as a column —
        # pandas refuses to disambiguate in sort_values, so drop the index first.
        tr_sorted = tracks.reset_index(drop=True).sort_values(["particle", "frame"])
        pids      = tr_sorted["particle"].values
        frames_t  = tr_sorted["frame"].values
        xs_t      = tr_sorted["x"].values
        ys_t      = tr_sorted["y"].values
        mass_t    = (tr_sorted["mass"].values if "mass" in tr_sorted.columns
                     else _np.zeros(len(tr_sorted)))
        for k in range(len(tr_sorted)):
            new_id = pid_to_new.get(int(pids[k]))
            if new_id is None:
                continue
            w.writerow([new_id, int(frames_t[k]) + 1,
                        float(xs_t[k]), float(ys_t[k]),
                        -1, float(mass_t[k]), k + 1, 0])

    print(f"  PALM-Tracer: wrote loc + trc; starting D files")

    # ── 3 & 5. D files ───────────────────────────────────────────────────
    D_arr     = diff_df["D"].values
    msd0_arr  = (diff_df["MSD0"].values if "MSD0" in diff_df.columns
                 else _np.zeros(len(diff_df)))
    mse_arr   = (diff_df["MSE"].values  if "MSE"  in diff_df.columns
                 else _np.zeros(len(diff_df)))
    logD_arr  = _np.where(D_arr > 0, _np.log10(_np.where(D_arr > 0, D_arr, 1)),
                          _np.nan)
    mobile_n  = int(_np.sum(D_arr > mobile_D_threshold))
    immob_n   = int(_np.sum(D_arr <= mobile_D_threshold))
    mob_ratio = (mobile_n / immob_n) if immob_n else _np.nan

    d1_path = _os.path.join(out_dir, f"{stem}_trcPALMTracer-1-D.csv")
    with open(d1_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow([f"#Diffusion Coef in um2/s; Linear fit performed on the "
                    f"first points of trajectories"])
        w.writerow([f"#Pixel size= {pixel_size_um}um ; Frame rate= "
                    f"{frame_interval_s}sec"])
        w.writerow(["ROI", "Trace", "D(um2/s)", "MSD(0)", "MSE",
                    "LogD", "Mobile/Immobile", "Tracks"])
        for i, pid in enumerate(pid_order):
            new_id = pid_to_new[int(pid)]
            row = [1, new_id,
                   float(D_arr[i]) if _np.isfinite(D_arr[i]) else "",
                   float(msd0_arr[i]) if _np.isfinite(msd0_arr[i]) else "",
                   float(mse_arr[i]) if _np.isfinite(mse_arr[i]) else "",
                   float(logD_arr[i]) if _np.isfinite(logD_arr[i]) else "",
                   "", ""]
            if i == 0:
                row[6] = mob_ratio if _np.isfinite(mob_ratio) else ""
                row[7] = n_tracks
            w.writerow(row)

    dA_path = _os.path.join(out_dir, f"{stem}_trcPALMTracer-AllROI-D.csv")
    with open(dA_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow([f"#Diffusion Coef in um2/s; Linear fit performed on the "
                    f"first points of trajectories"])
        w.writerow([f"#Pixel size= {pixel_size_um}um ; Frame rate= "
                    f"{frame_interval_s}sec"])
        w.writerow(["ROI", "Trace", "D(um2/s)", "MSD(0)", "MSE"])
        for i, pid in enumerate(pid_order):
            new_id = pid_to_new[int(pid)]
            w.writerow([1, new_id,
                        float(D_arr[i]) if _np.isfinite(D_arr[i]) else "",
                        float(msd0_arr[i]) if _np.isfinite(msd0_arr[i]) else "",
                        float(mse_arr[i]) if _np.isfinite(mse_arr[i]) else ""])

    print(f"  PALM-Tracer: wrote D files; starting MSD files")

    # ── 4 & 6. MSD files (jagged: one column per surviving lag) ──────────
    def _write_msd(path):
        with open(path, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["#MSD(DeltaT) in um2"])
            w.writerow([f"#Pixel size= {pixel_size_um}um ; Frame rate= "
                        f"{frame_interval_s}sec"])
            for pid in pid_order:
                if int(pid) not in imsd_df.columns and pid not in imsd_df.columns:
                    continue
                col = imsd_df[pid] if pid in imsd_df.columns else imsd_df[int(pid)]
                vals = col.values
                finite_idx = _np.where(_np.isfinite(vals))[0]
                if len(finite_idx) == 0:
                    continue
                last = finite_idx[-1] + 1
                row = [1, pid_to_new[int(pid)]]
                row.extend(float(v) if _np.isfinite(v) else ""
                           for v in vals[:last])
                w.writerow(row)

    _write_msd(_os.path.join(out_dir, f"{stem}_trcPALMTracer-1-MSD.csv"))
    _write_msd(_os.path.join(out_dir, f"{stem}_trcPALMTracer-AllROI-MSD.csv"))
    print(f"  PALM-Tracer: all 6 files written successfully")

    return {
        "loc":           loc_path,
        "trc":           tr_path,
        "D_1":           d1_path,
        "D_AllROI":      dA_path,
        "MSD_1":         _os.path.join(out_dir, f"{stem}_trcPALMTracer-1-MSD.csv"),
        "MSD_AllROI":    _os.path.join(out_dir, f"{stem}_trcPALMTracer-AllROI-MSD.csv"),
    }










def _stat_test(a, b):
    """Two-sample test on per-experiment scalars.  Welch's t by default,
    Mann-Whitney as fallback for non-normal data.  Returns (p, label)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return (np.nan, "")
    try:
        from scipy.stats import ttest_ind, mannwhitneyu, shapiro
        normal = True
        for arr in (a, b):
            if 3 <= len(arr) <= 5000:
                try:
                    if shapiro(arr).pvalue < 0.05:
                        normal = False
                        break
                except Exception:
                    pass
        if normal:
            p = ttest_ind(a, b, equal_var=False).pvalue
        else:
            p = mannwhitneyu(a, b, alternative="two-sided").pvalue
        if not np.isfinite(p):
            return (np.nan, "")
        if p < 0.001: stars = "***"
        elif p < 0.01: stars = "**"
        elif p < 0.05: stars = "*"
        else: stars = "ns"
        return (float(p), stars)
    except Exception:
        return (np.nan, "")


# NOTE: the canonical `_theme_palette` definition lives near
# `compute_circular_statistics` above.  Earlier there was a SECOND
# definition here with a smaller set of keys (BG/PNL/TXT/GRD/BAR_FILL/
# SIG/FONT only); because Python rebinds `_theme_palette` at module
# parse time, the second definition silently won and every call to
# `_theme_palette` returned a dict missing MUT/ACC/HDR_BG/HDR_TXT/
# ZEBRA/ARROW.  That manifested as `KeyError('MUT')` from
# `save_circular_statistics_pdf` (which uses those keys).  The
# canonical version above now also exports the BAR_FILL/SIG keys
# `compare_groups` consumes here, so the duplicate is safe to remove.


def _stat_test_n(arrays, labels):
    """Statistical test across N≥2 groups.

    Returns
    -------
    omnibus : dict with keys {"test", "p", "stars"} or None if n<2 each
    pairwise : list of dicts with keys
        {"i", "j", "label_i", "label_j", "test", "p", "stars",
         "n_i", "n_j", "mean_i", "mean_j", "sem_i", "sem_j"}
    """
    arrs = [np.asarray(a, dtype=float)[np.isfinite(np.asarray(a, dtype=float))]
            for a in arrays]
    valid_idx = [i for i, a in enumerate(arrs) if len(a) >= 2]

    omnibus = None
    pairwise = []

    def _star(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    if len(valid_idx) < 2:
        # Still record per-pair "ns" rows for stats CSV completeness
        for i in range(len(arrs)):
            for j in range(i + 1, len(arrs)):
                pairwise.append({
                    "i": i, "j": j,
                    "label_i": labels[i], "label_j": labels[j],
                    "test": "n<2", "p": np.nan, "stars": "",
                    "n_i": int(len(arrs[i])), "n_j": int(len(arrs[j])),
                    "mean_i": float(arrs[i].mean()) if len(arrs[i]) else np.nan,
                    "mean_j": float(arrs[j].mean()) if len(arrs[j]) else np.nan,
                    "sem_i": (float(arrs[i].std(ddof=1) / np.sqrt(len(arrs[i])))
                              if len(arrs[i]) > 1 else np.nan),
                    "sem_j": (float(arrs[j].std(ddof=1) / np.sqrt(len(arrs[j])))
                              if len(arrs[j]) > 1 else np.nan),
                })
        return omnibus, pairwise

    # Omnibus test
    try:
        from scipy.stats import f_oneway, kruskal, shapiro
        valid_arrs = [arrs[i] for i in valid_idx]

        normal = True
        for a in valid_arrs:
            if 3 <= len(a) <= 5000:
                try:
                    if shapiro(a).pvalue < 0.05:
                        normal = False
                        break
                except Exception:
                    pass

        if len(valid_arrs) == 2:
            from scipy.stats import ttest_ind, mannwhitneyu
            if normal:
                p = ttest_ind(*valid_arrs, equal_var=False).pvalue
                test_name = "Welch's t-test"
            else:
                p = mannwhitneyu(*valid_arrs, alternative="two-sided").pvalue
                test_name = "Mann-Whitney U"
        else:
            if normal:
                p = f_oneway(*valid_arrs).pvalue
                test_name = "One-way ANOVA"
            else:
                p = kruskal(*valid_arrs).pvalue
                test_name = "Kruskal-Wallis"
        if np.isfinite(p):
            omnibus = {"test": test_name, "p": float(p), "stars": _star(p)}
    except Exception:
        pass

    # Pairwise comparisons
    try:
        from scipy.stats import ttest_ind, mannwhitneyu, shapiro
        for i in range(len(arrs)):
            for j in range(i + 1, len(arrs)):
                a, b = arrs[i], arrs[j]
                if len(a) < 2 or len(b) < 2:
                    p = np.nan
                    test_name = "n<2"
                else:
                    is_normal = True
                    for arr in (a, b):
                        if 3 <= len(arr) <= 5000:
                            try:
                                if shapiro(arr).pvalue < 0.05:
                                    is_normal = False
                                    break
                            except Exception:
                                pass
                    if is_normal:
                        p = ttest_ind(a, b, equal_var=False).pvalue
                        test_name = "Welch's t-test"
                    else:
                        p = mannwhitneyu(a, b, alternative="two-sided").pvalue
                        test_name = "Mann-Whitney U"
                pairwise.append({
                    "i": i, "j": j,
                    "label_i": labels[i], "label_j": labels[j],
                    "test": test_name,
                    "p": float(p) if np.isfinite(p) else np.nan,
                    "stars": _star(p) if np.isfinite(p) else "",
                    "n_i": int(len(a)), "n_j": int(len(b)),
                    "mean_i": float(a.mean()) if len(a) else np.nan,
                    "mean_j": float(b.mean()) if len(b) else np.nan,
                    "sem_i": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.nan,
                    "sem_j": float(b.std(ddof=1) / np.sqrt(len(b))) if len(b) > 1 else np.nan,
                })
    except Exception:
        pass

    return omnibus, pairwise


def _bar_with_dots_n(ax, data_per_group, labels, colors, palette,
                     ylabel="", record_stats=None, metric_name=""):
    """Bar chart with mean ± SEM and individual replicate dots, generalised
    to N groups.

    For 2 groups: shows pairwise stars on a bracket (matches lab style).
    For 3+ groups: shows omnibus ANOVA / Kruskal p-value as a panel
    annotation; full pairwise comparisons go to record_stats[metric_name]."""
    fill = palette["BAR_FILL"]
    sig_col = palette["SIG"]

    arrs = [np.asarray(d, dtype=float) for d in data_per_group]
    arrs = [a[np.isfinite(a)] for a in arrs]
    n = len(arrs)
    means = [float(a.mean()) if len(a) else 0.0 for a in arrs]
    sems  = [float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
             for a in arrs]
    x = np.arange(n)
    ax.bar(x, means, yerr=sems, capsize=4,
           color=[fill] * n,
           edgecolor=colors, linewidth=1.5,
           ecolor=sig_col)
    rng = np.random.default_rng(0)
    for i, a in enumerate(arrs):
        if len(a):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(a)), a,
                       color=colors[i], s=18, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15 if n > 3 else 0)
    ax.set_ylabel(ylabel)

    # Stats
    omnibus, pairwise = _stat_test_n(arrs, labels)
    if record_stats is not None and metric_name:
        record_stats[metric_name] = {"omnibus": omnibus, "pairwise": pairwise}

    # Annotation
    top_data = max([a.max() if len(a) else 0 for a in arrs] + [max(means) * 1.2 if max(means) > 0 else 1])
    if n == 2 and pairwise:
        pair = pairwise[0]
        if pair["stars"] and np.isfinite(pair["p"]):
            top = top_data * 1.05
            ax.plot([0, 0, 1, 1], [top, top * 1.03, top * 1.03, top],
                    color=sig_col, lw=0.8)
            # Numeric p plus stars, e.g. "p = 0.003  **"
            p_str = (f"p = {pair['p']:.2e}" if pair['p'] < 0.001
                     else f"p = {pair['p']:.3f}")
            label = f"{p_str}  {pair['stars']}"
            ax.text(0.5, top * 1.05, label, ha="center", va="bottom",
                    fontsize=9, color=sig_col)
            # Make room above the bracket for the longer label
            ax.set_ylim(0, top * 1.30)
    elif n > 2 and omnibus:
        # Show test name + omnibus p + stars in the upper-left corner.
        # Numeric format adapts to magnitude: scientific < 0.001, fixed otherwise.
        p_val = omnibus['p']
        p_str = (f"p = {p_val:.2e}" if p_val < 0.001
                 else f"p = {p_val:.3f}")
        text = f"{omnibus['test']}\n{p_str}   {omnibus['stars']}"
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color=sig_col,
                bbox=dict(facecolor=palette["PNL"], edgecolor="none",
                          alpha=0.7, pad=3))


def compare_groups(groups,
                   output_dir=None, output_stem="comparison",
                   panels=None, theme="Dark",
                   pdf_report=True,
                   mobile_d_threshold=MOBILE_D_THRESHOLD_DEFAULT,
                   progress_cb=None):
    """Compare N≥2 groups of analysis output folders and render a multi-panel
    figure, summary CSV, statistics CSV and combined PDF report.

    Parameters
    ----------
    groups : list[dict]
        [{"folders": [path, ...], "label": "Pre", "color": "#000000"}, ...]
    output_dir : str or None
        Where to save the figure / CSVs / PDF report.  If None, nothing is
        saved to disk and only the figure is returned.
    panels : set[str] or None
        Subset of panels to render.  Default: all of {"msd", "auc",
        "logd_dist", "mob_immob", "motion_classes", "track_length",
        "jdd", "dwell_cdf", "turning_angles"}.
    theme : str
        Figure theme — "Dark" (default), "Light" or "Publication".
    pdf_report : bool
        If True (default) and output_dir is given, also write a multi-page
        PDF report bundling the figure, parameters, folder lists and stats.
    progress_cb : callable or None
        Optional callback(done:int, total:int, msg:str) for UI progress.

    Returns
    -------
    fig         : matplotlib.figure.Figure
    summary_df  : pandas.DataFrame  — per-replicate scalar metrics
    stats       : dict[str, dict]   — per-metric omnibus + pairwise tests
    """
    import matplotlib.pyplot as plt

    if len(groups) < 2:
        raise ValueError(f"Need at least 2 groups; got {len(groups)}")

    if panels is None:
        panels = {"msd", "auc", "logd_dist", "mob_immob", "motion_classes",
                  "track_length", "jdd", "dwell_cdf", "turning_angles",
                  "radial_dist"}

    n_groups = len(groups)
    labels   = [g.get("label", f"Group {i+1}") for i, g in enumerate(groups)]
    colors   = [g.get("color", "#3b6ed8")     for g in groups]
    folder_lists = [list(g["folders"]) for g in groups]

    # ── Load summaries for all groups ─────────────────────────────────────────
    all_summaries = [[] for _ in groups]
    total = sum(len(f) for f in folder_lists)
    done = 0
    for gi, folders in enumerate(folder_lists):
        for f in folders:
            if progress_cb:
                progress_cb(done, total, f"Loading: {os.path.basename(f)}")
            try:
                all_summaries[gi].append(load_summary_from_folder(f))
            except Exception as e:
                print(f"  Skipping {f}: {e}")
            done += 1

    empty_groups = [labels[i] for i, ss in enumerate(all_summaries) if len(ss) == 0]
    if empty_groups:
        raise RuntimeError(
            "Need at least one valid folder per group; these are empty: "
            + ", ".join(empty_groups))

    if progress_cb:
        progress_cb(total, total, "Computing scalars and rendering...")

    # ── Compute per-folder scalars (one row per replicate) ────────────────────
    summary_rows = []
    def _row(group_label, summary):
        p = summary["params"]
        fi = float(p.get("frame_interval_s", 0.05))
        d = summary["diffusion"]
        return {
            "group":            group_label,
            "folder":           summary["folder"],
            "stem":             summary["stem"],
            "n_tracks":         len(d) if d is not None else 0,
            "auc_msd":          _msd_auc(summary["ensemble_msd"], fi),
            "mob_immob_ratio":  _mob_immob_ratio(d, mobile_d_threshold),
            "median_D":         float(d["D"].median()) if d is not None and "D" in d.columns else np.nan,
            "median_alpha":     float(d["alpha"].median()) if d is not None and "alpha" in d.columns else np.nan,
            "mean_track_length_s": float(_track_lengths(summary["tracks"], fi).mean())
                                   if summary["tracks"] is not None else np.nan,
        }
    for gi, summaries in enumerate(all_summaries):
        for s in summaries:
            summary_rows.append(_row(labels[gi], s))
    summary_df = pd.DataFrame(summary_rows)

    # Per-metric statistics dict — populated as panels render
    stats_records = {}

    # ── Render the figure ────────────────────────────────────────────────────
    panel_order = ["msd", "auc", "logd_dist", "mob_immob",
                   "motion_classes", "track_length",
                   "jdd", "dwell_cdf", "turning_angles", "radial_dist"]
    enabled = [p for p in panel_order if p in panels]
    n_plots = len(enabled)
    if n_plots == 0:
        raise RuntimeError("No panels enabled")
    print(f"  Compare: rendering {n_plots} panel(s): {enabled}")
    if "radial_dist" not in panels:
        print(f"  Compare: 'radial_dist' NOT in requested panels — "
              f"check the 'Radial distribution (polar)' tickbox in the "
              f"Compare tab to include it.")
    ncols = 3 if n_plots > 4 else 2
    nrows = (n_plots + ncols - 1) // ncols

    pal = _theme_palette(theme)
    plt.rcParams.update({
        "text.color":      pal["TXT"], "axes.labelcolor": pal["TXT"],
        "xtick.color":     pal["TXT"], "ytick.color":     pal["TXT"],
        "axes.titlecolor": pal["TXT"],
        "axes.edgecolor":  pal["GRD"], "axes.facecolor":  pal["PNL"],
        "figure.facecolor": pal["BG"], "figure.edgecolor": pal["BG"],
        "savefig.facecolor": pal["BG"], "savefig.edgecolor": pal["BG"],
        "grid.color":      pal["GRD"], "grid.alpha": 0.4,
        "font.family":     pal["FONT"],
        "legend.facecolor": pal["PNL"], "legend.edgecolor": pal["GRD"],
    })

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.6),
                             facecolor=pal["BG"])
    axes = np.array(axes).reshape(-1)
    for ax in axes[n_plots:]:
        ax.axis("off")

    panel_idx = 0
    def _next_ax():
        nonlocal panel_idx
        ax = axes[panel_idx]; panel_idx += 1
        return ax

    def _zip_groups():
        """Iterator: (label, summaries, color) for each group."""
        for i in range(n_groups):
            yield labels[i], all_summaries[i], colors[i]

    # ── 1. MSD overlay ────────────────────────────────────────────────────────
    if "msd" in panels:
        ax = _next_ax()
        for grp_label, summaries, color in _zip_groups():
            curves = []
            tref = None
            for s in summaries:
                e = s["ensemble_msd"]
                if e is None: continue
                fi = float(s["params"].get("frame_interval_s", 0.05))
                t = e["lag_frame"].values * fi
                y = e["msd_um2"].values
                order = np.argsort(t)
                t, y = t[order], y[order]
                if tref is None:
                    tref = t
                if len(t) != len(tref) or not np.allclose(t, tref):
                    y = np.interp(tref, t, y)
                curves.append(y)
            if not curves:
                continue
            arr = np.vstack(curves)
            mean = arr.mean(axis=0)
            sem = arr.std(axis=0, ddof=1) / np.sqrt(len(curves)) if len(curves) > 1 else None
            ax.plot(tref, mean, "-o", color=color, label=grp_label, ms=4, lw=1.5)
            if sem is not None:
                ax.fill_between(tref, mean - sem, mean + sem, color=color, alpha=0.15)
        ax.set_xlabel("Time delta (s)")
        ax.set_ylabel("MSD (µm²)")
        ax.set_title("Mean Square Displacement")
        ax.legend(frameon=False, loc="best")

    # ── 2. AUC bar chart ──────────────────────────────────────────────────────
    if "auc" in panels:
        ax = _next_ax()
        data = [summary_df.loc[summary_df["group"] == lbl, "auc_msd"].values
                for lbl in labels]
        _bar_with_dots_n(ax, data, labels, colors, pal,
                         ylabel="AUC (µm²·s)",
                         record_stats=stats_records, metric_name="auc_msd")
        ax.set_title("Area Under the Curve")

    # ── 3. LogD frequency distribution ────────────────────────────────────────
    if "logd_dist" in panels:
        ax = _next_ax()
        bins = np.linspace(-5, 1, 31)
        for grp_label, summaries, color in _zip_groups():
            all_logD = []
            for s in summaries:
                d = s["diffusion"]
                if d is None or "D" not in d.columns: continue
                vals = d["D"].values
                vals = vals[vals > 0]
                if len(vals): all_logD.append(np.log10(vals))
            if not all_logD: continue
            pooled = np.concatenate(all_logD)
            counts, edges = np.histogram(pooled, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            frac = counts / counts.sum() if counts.sum() else counts
            ax.plot(centers, frac, "-o", color=color, label=grp_label, ms=4, lw=1.2)
        ax.axvline(np.log10(mobile_d_threshold), color=pal["GRD"], ls="--", lw=0.8,
                   label=f"D = {mobile_d_threshold} µm²/s")
        ax.set_xlabel("log₁₀ D  (µm²/s)")
        ax.set_ylabel("Relative frequency")
        ax.set_title("LogD Frequency Distribution")
        ax.legend(frameon=False, loc="best")

    # ── 4. Mobile/Immobile ratio bar ──────────────────────────────────────────
    if "mob_immob" in panels:
        ax = _next_ax()
        data = [summary_df.loc[summary_df["group"] == lbl, "mob_immob_ratio"].values
                for lbl in labels]
        _bar_with_dots_n(ax, data, labels, colors, pal,
                         ylabel="Mobile/Immobile ratio",
                         record_stats=stats_records, metric_name="mob_immob_ratio")
        ax.set_title("Mobile/Immobile Ratio")

    # ── 5. Motion class fractions (grouped bars, N groups) ────────────────────
    if "motion_classes" in panels:
        ax = _next_ax()
        classes = ["Immobile", "Confined", "Brownian", "Directed"]
        def _fracs(summaries):
            rows = []
            for s in summaries:
                f = _motion_fractions(s["diffusion"])
                rows.append([f.get(c, 0.0) for c in classes])
            return np.array(rows) if rows else np.zeros((0, len(classes)))
        per_group = [_fracs(ss) for ss in all_summaries]
        x = np.arange(len(classes))
        # Group-bar width: total slot ~0.8, divided across N groups
        slot = 0.8
        w = slot / n_groups
        rng = np.random.default_rng(1)
        for gi, (grp_label, color, fracs) in enumerate(zip(labels, colors, per_group)):
            if not len(fracs): continue
            x_off = (gi - (n_groups - 1) / 2) * w
            ax.bar(x + x_off, fracs.mean(axis=0), w * 0.9,
                   yerr=fracs.std(axis=0, ddof=1)/np.sqrt(len(fracs)) if len(fracs) > 1 else None,
                   color=pal["BAR_FILL"], edgecolor=color, linewidth=1.5,
                   ecolor=pal["SIG"], capsize=3, label=grp_label)
            for ci in range(len(classes)):
                ax.scatter(np.full(len(fracs), x[ci] + x_off)
                           + rng.uniform(-w*0.25, w*0.25, len(fracs)),
                           fracs[:, ci], color=color, s=12, zorder=3)
        # Per-class stats
        for ci, cname in enumerate(classes):
            arrs = [fracs[:, ci] if len(fracs) else np.array([]) for fracs in per_group]
            omn, pw = _stat_test_n(arrs, labels)
            stats_records[f"motion_frac_{cname}"] = {"omnibus": omn, "pairwise": pw}
        ax.set_xticks(x); ax.set_xticklabels(classes, rotation=15)
        ax.set_ylabel("Fraction of tracks")
        ax.set_title("Motion Class Fractions")
        ax.legend(frameon=False, loc="best", fontsize=8)

    # ── 6. Track length distribution (CDF, x clipped at 99th %ile) ────────────
    if "track_length" in panels:
        ax = _next_ax()
        pooled_per_group = {}
        for grp_label, summaries, _ in _zip_groups():
            arrs = []
            for s in summaries:
                fi = float(s["params"].get("frame_interval_s", 0.05))
                tl = _track_lengths(s["tracks"], fi)
                if len(tl):
                    arrs.append(tl)
            if arrs:
                pooled_per_group[grp_label] = np.concatenate(arrs)
        combined = (np.concatenate(list(pooled_per_group.values()))
                    if pooled_per_group else np.array([]))
        x_clip = float(np.percentile(combined, 99)) if len(combined) else None
        for grp_label, color in zip(labels, colors):
            p = pooled_per_group.get(grp_label)
            if p is None or len(p) == 0: continue
            x_sorted = np.sort(p)
            y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
            ax.plot(x_sorted, y, color=color, lw=1.5, label=grp_label)
        if pooled_per_group:
            if x_clip and x_clip > 0:
                ax.set_xlim(0, x_clip)
                ax.set_title("Track Length Distribution  (x clipped at 99th %ile)")
            else:
                ax.set_title("Track Length Distribution")
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Track length (s)")
            ax.set_ylabel("Cumulative fraction")
            ax.legend(frameon=False, loc="best")
        else:
            ax.text(0.5, 0.5, "No track-length data",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Track Length Distribution")
        # Stats: mean track length (per-replicate)
        arrs = [summary_df.loc[summary_df["group"] == lbl, "mean_track_length_s"].values
                for lbl in labels]
        omn, pw = _stat_test_n(arrs, labels)
        stats_records["mean_track_length_s"] = {"omnibus": omn, "pairwise": pw}

    # ── 7. JDD: per-population D + fraction (N groups) ────────────────────────
    if "jdd" in panels:
        ax = _next_ax()
        any_data = False
        max_pop_overall = 0
        # Spread groups across ±0.18 around each population index
        if n_groups > 1:
            offsets = np.linspace(-0.18, 0.18, n_groups)
        else:
            offsets = np.array([0.0])
        for gi, (grp_label, summaries, color) in enumerate(_zip_groups()):
            label_done = False
            for s in summaries:
                jd = s.get("jdd")
                if not jd or "D_values" not in jd: continue
                D = np.asarray(jd["D_values"], dtype=float)
                f = np.asarray(jd.get("fractions", np.ones_like(D)), dtype=float)
                if D.size == 0: continue
                any_data = True
                max_pop_overall = max(max_pop_overall, len(D))
                sizes = 25 + 175 * np.clip(f, 0, 1)
                xs = np.arange(len(D)) + offsets[gi]
                ax.scatter(xs, D, s=sizes, color=color,
                           alpha=0.55, edgecolor=color,
                           label=(grp_label if not label_done else None))
                label_done = True
        if any_data:
            tick_labels = ["Immobile", "Mobile", "Fast"][:max_pop_overall]
            if max_pop_overall == 1: tick_labels = ["All"]
            ax.set_xticks(np.arange(max_pop_overall))
            ax.set_xticklabels(tick_labels)
            ax.set_xlim(-0.5, max_pop_overall - 0.5)
            ax.set_ylabel("D (µm²/s, log)")
            ax.set_yscale("log")
            ax.set_title("JDD: per-population D  (marker size ∝ population fraction)")
            ax.legend(frameon=False, loc="best")
        else:
            ax.text(0.5, 0.5, "No JDD data\n(re-run analysis to generate)",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Jump Distance Distribution")

    # ── 8. Dwell time CDF (N groups) ──────────────────────────────────────────
    if "dwell_cdf" in panels:
        ax = _next_ax()
        any_data = False
        for grp_label, summaries, color in _zip_groups():
            pooled = []
            for s in summaries:
                d = s.get("dwell_times")
                if d is None or len(d) == 0: continue
                col = next((c for c in ("dwell_time_s", "dwell_s",
                                        "dwell_time", "dwell", "tau_s")
                            if c in d.columns), None)
                if col is None: continue
                pooled.extend(d[col].values)
            if not pooled: continue
            any_data = True
            arr = np.sort(np.asarray(pooled, dtype=float))
            arr = arr[arr > 0]
            if len(arr) == 0: continue
            y = 1 - np.arange(1, len(arr) + 1) / len(arr)
            ax.plot(arr, y, color=color, lw=1.5, label=grp_label)
        if any_data:
            ax.set_xlabel("Dwell time (s)")
            ax.set_ylabel("Survival fraction")
            ax.set_title("Dwell Time Survival")
            ax.set_yscale("log")
            ax.legend(frameon=False, loc="best")
        else:
            ax.text(0.5, 0.5, "No dwell-time data\n(re-run analysis to generate)",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Dwell Time Survival")

    # ── 9. Turning angle distribution (N groups, unsigned |angle|) ────────────
    # Single line per group, plotting the count of each |θ| bin on
    # the same 0°–180° x-axis.  Sign / rotational direction is handled
    # separately by the Radial Distribution panel.
    if "turning_angles" in panels:
        ax = _next_ax()
        any_data = False
        bins = np.linspace(0, 180, 37)                 # 5° bins
        centers = 0.5 * (bins[:-1] + bins[1:])
        pooled_per_group = []
        for grp_label, summaries, color in _zip_groups():
            pooled = []
            for s in summaries:
                ta = s.get("turning_angles")
                if ta is None or len(ta) == 0: continue
                pooled.extend(np.abs(np.asarray(ta).ravel()))
            pooled_per_group.append((grp_label, color, pooled))
        for grp_label, color, pooled in pooled_per_group:
            if not pooled: continue
            any_data = True
            counts, _ = np.histogram(pooled, bins=bins)
            frac = counts / counts.sum() if counts.sum() else counts
            ax.plot(centers, frac, "-o", color=color, lw=1.5, ms=3, label=grp_label)
        if any_data:
            ax.set_xlabel("|Turning angle|  (°)")
            ax.set_ylabel("Relative frequency")
            ax.set_xlim(0, 180)
            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_title("Turning Angle Distribution")
            ax.legend(frameon=False, loc="best")
        else:
            ax.text(0.5, 0.5, "No turning-angle data\n(re-run analysis to generate)",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Turning Angle Distribution")

    # ── 10. Radial distribution (polar, signed turning angles) ────────────────
    # Polar histogram showing the angular distribution of step-to-step
    # turning angles.  Each group is plotted as a separate set of bars
    # offset around each bin centre.
    #
    # Implementation note: we replace the auto-created cartesian axis with
    # a polar one at the SAME SubplotSpec (not via fig.add_axes with raw
    # bounds), so that the polar axis remains a managed gridspec member.
    # If we used add_axes(bounds), tight_layout would later reposition the
    # other (gridspec-managed) subplots but leave the polar in its original
    # location, causing visible overlap.
    if "radial_dist" in panels:
        old_ax = axes[panel_idx]
        ss = old_ax.get_subplotspec()
        old_ax.remove()
        ax = fig.add_subplot(ss, projection="polar")
        axes[panel_idx] = ax
        panel_idx += 1

        any_data = False
        n_bins = 36
        # matplotlib polar bar() only renders correctly when theta ∈ [0, 2π);
        # shift the data accordingly.  The xticks are placed at positive-only
        # angles but labelled with their signed equivalents.
        bin_edges   = np.linspace(0, 2 * np.pi, n_bins + 1)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bar_width   = (bin_edges[1] - bin_edges[0]) * 0.95

        # First pass: get raw counts per group per bin.
        counts_per_group = []     # list of (group_idx, counts_array)
        for gi in range(n_groups):
            pooled = []
            for s in all_summaries[gi]:
                ta = s.get("turning_angles")
                if ta is None or len(ta) == 0: continue
                pooled.extend(np.asarray(ta).ravel())
            if not pooled:
                counts_per_group.append((gi, np.zeros(n_bins)))
                continue
            arr = np.asarray(pooled, dtype=float)
            if not np.any(arr < -1e-3):
                arr = np.concatenate([arr, -arr])
            angles_rad = np.mod(np.deg2rad(arr), 2 * np.pi)
            counts, _ = np.histogram(angles_rad, bins=bin_edges)
            counts_per_group.append((gi, counts.astype(float)))
            if counts.sum() > 0:
                any_data = True

        if any_data:
            # ── Normalise each group to ITS OWN total ─────────────────────
            # Otherwise a group with more total angles automatically draws
            # bigger bars everywhere — a sample-size artefact, not a real
            # shape difference.  After dividing by the per-group total, each
            # group's values sum to 1.0 across the full circle, so the bars
            # compare distribution SHAPE.
            # Bars from different groups are offset around each bin centre
            # for easy side-by-side comparison.
            per_bar_width = bar_width / max(1, n_groups) * 0.95
            for gi, counts in counts_per_group:
                total = counts.sum()
                if total <= 0:
                    continue
                normalised = counts / total
                offset = (gi - (n_groups - 1) / 2) * per_bar_width
                ax.bar(bin_centres + offset, normalised,
                       width=per_bar_width, bottom=0.0,
                       color=colors[gi], alpha=0.85,
                       edgecolor=pal["GRD"], linewidth=0.3,
                       label=labels[gi])

        if any_data:
            # Conventional orientation: 0° at top (straight ahead),
            # right hemisphere = positive turns, left hemisphere = negative.
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            # Positive-only xticks; labelled with signed equivalents.
            ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
            ax.set_xticklabels(["0°", "+45°", "+90°", "+135°", "±180°",
                                "−135°", "−90°", "−45°"], fontsize=7)
            # Hide the radial-axis numeric labels — bar length is
            # interpreted comparatively, not in absolute density units.
            ax.set_yticklabels([])
            ax.tick_params(axis="y", which="both", left=False)
            ax.set_title("Radial Distribution  (each group normalised to "
                         "its own total)", pad=14, fontsize=9)
            ax.legend(loc="upper right", bbox_to_anchor=(1.20, 1.10),
                      frameon=False, fontsize=8)
            ax.grid(True, ls=":", alpha=0.4)
        else:
            ax.text(0.5, 0.5, "No turning-angle data",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Radial Distribution")

    # ── Suptitle: Group A (n=…) vs Group B (n=…) [vs Group C …] ───────────────
    parts = [f"{labels[i]}  (n={len(all_summaries[i])})" for i in range(n_groups)]
    fig.suptitle("   vs   ".join(parts),
                 fontsize=12, fontweight="bold", color=pal["TXT"])
    for ax in axes[:n_plots]:
        ax.set_facecolor(pal["PNL"])
        for spine in ax.spines.values():
            spine.set_edgecolor(pal["GRD"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # ── Build statistics dataframe (per metric × pairwise) ────────────────────
    # Bonferroni correction across pairwise comparisons WITHIN each metric:
    # multiplies the raw p-value by the number of pairs (capped at 1.0).
    # The omnibus row gets the raw p-value only — it's a single test.
    stats_rows = []
    for metric, rec in stats_records.items():
        omn = rec.get("omnibus")
        if omn:
            stars = omn["stars"]
            stars_bonf = stars  # omnibus needs no correction
            stats_rows.append({
                "metric": metric, "comparison": "omnibus",
                "test": omn["test"],
                "p_value": omn["p"], "stars": stars,
                "p_value_bonferroni": omn["p"], "stars_bonferroni": stars_bonf,
                "n_a": "", "n_b": "", "mean_a": "", "mean_b": "",
                "sem_a": "", "sem_b": "", "label_a": "all groups", "label_b": "",
            })
        pairs = rec.get("pairwise", [])
        n_pairs = max(1, len(pairs))
        for pw in pairs:
            p = pw["p"]
            if np.isfinite(p):
                p_bonf = min(1.0, p * n_pairs)
                if   p_bonf < 0.001: stars_bonf = "***"
                elif p_bonf < 0.01:  stars_bonf = "**"
                elif p_bonf < 0.05:  stars_bonf = "*"
                else:                stars_bonf = "ns"
            else:
                p_bonf = np.nan
                stars_bonf = ""
            stats_rows.append({
                "metric": metric, "comparison": f"{pw['label_i']} vs {pw['label_j']}",
                "test": pw["test"],
                "p_value": pw["p"], "stars": pw["stars"],
                "p_value_bonferroni": p_bonf, "stars_bonferroni": stars_bonf,
                "n_a": pw["n_i"], "n_b": pw["n_j"],
                "mean_a": pw["mean_i"], "mean_b": pw["mean_j"],
                "sem_a": pw["sem_i"], "sem_b": pw["sem_j"],
                "label_a": pw["label_i"], "label_b": pw["label_j"],
            })
    stats_df = pd.DataFrame(stats_rows)

    # ── Save outputs ──────────────────────────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        png_path  = os.path.join(output_dir, f"{output_stem}.png")
        pdf_path  = os.path.join(output_dir, f"{output_stem}.pdf")
        csv_path  = os.path.join(output_dir, f"{output_stem}_summary.csv")
        stats_csv = os.path.join(output_dir, f"{output_stem}_stats.csv")
        fig.savefig(png_path, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        summary_df.to_csv(csv_path, index=False)
        if len(stats_df):
            stats_df.to_csv(stats_csv, index=False)
        print(f"  Saved: {png_path}")
        print(f"  Saved: {pdf_path}")
        print(f"  Saved: {csv_path}")
        if len(stats_df):
            print(f"  Saved: {stats_csv}")

        # ── Combined PDF report (figure + parameters + folders + stats) ──────
        if pdf_report:
            report_path = os.path.join(output_dir, f"{output_stem}_report.pdf")
            try:
                _write_pdf_report(report_path, fig, groups, all_summaries,
                                  labels, colors, summary_df, stats_df,
                                  panels=panels, theme=theme, palette=pal)
                print(f"  Saved: {report_path}")
            except Exception as exc:
                print(f"  PDF report skipped ({type(exc).__name__}: {exc})")

        # ── Per-comparison circular-statistics CSV + PDF ────────────────────
        # Pool angles per group (across all replicates), compute the full
        # CircStat suite for each group, and emit:
        #   * {stem}_circular_statistics.csv  — one row per group
        #   * {stem}_circular_statistics.pdf  — themed multi-page PDF
        #     (page 1 = summary grid + comparison table; pages 2..N+1 =
        #     per-group detail mirroring the per-file report).
        try:
            groups_angles_pooled = []
            # Per-track (mean_angle_deg, D) pairs per group, used for
            # the circular-linear correlation between a track's
            # average turning bias and its diffusion coefficient.
            # One list of pairs per group; each list pools across
            # the group's replicates.
            track_angle_d_pairs = []
            # Per-replicate angle arrays — one list of arrays per group.
            # Used to compute per-replicate κ, R̄, μ for the Welch t-test
            # and per-replicate Watson-Williams F-test (treats each
            # replicate as one data point, the statistically defensible
            # framing for n=5 vs n=3 designs).
            per_replicate_angles = {}
            for label, ss, color in zip(labels, all_summaries, colors):
                pooled = []
                t_angles_g = []
                t_D_g      = []
                rep_angle_arrays = []
                for s in ss:
                    ta = s.get("turning_angles")
                    if ta is not None:
                        arr = np.asarray(ta, dtype=float).ravel()
                        if arr.size:
                            pooled.append(arr)
                            rep_angle_arrays.append(arr)
                    tracks = s.get("tracks")
                    diff_df = s.get("diffusion")
                    if tracks is None or diff_df is None:
                        continue
                    if "D" not in diff_df.columns:
                        continue
                    try:
                        pairs = compute_per_track_mean_angle(tracks)
                        if not pairs:
                            continue
                        d_map = dict(zip(diff_df["particle"].astype(int),
                                         diff_df["D"].astype(float)))
                        for pid, mu_deg in pairs:
                            d_val = d_map.get(int(pid))
                            if d_val is None or not np.isfinite(d_val):
                                continue
                            t_angles_g.append(float(mu_deg))
                            t_D_g.append(float(d_val))
                    except Exception:
                        continue
                pooled_arr = (np.concatenate(pooled)
                              if pooled else np.array([], dtype=float))
                groups_angles_pooled.append((label, pooled_arr, color))
                track_angle_d_pairs.append(
                    (np.asarray(t_angles_g, dtype=float),
                     np.asarray(t_D_g,      dtype=float)))
                per_replicate_angles[label] = rep_angle_arrays
            cs_csv = os.path.join(
                output_dir, f"{output_stem}_circular_statistics.csv")
            cs_pdf = os.path.join(
                output_dir, f"{output_stem}_circular_statistics.pdf")
            save_comparison_circular_statistics(
                groups_angles_pooled,
                csv_path=cs_csv, pdf_path=cs_pdf,
                fig_theme=theme,
                track_angle_d_pairs=track_angle_d_pairs,
                per_replicate_angles=per_replicate_angles)
            print(f"  Saved: {cs_csv}")
            print(f"  Saved: {cs_pdf}")
        except Exception as exc:
            print(f"  Comparison circular-stats skipped "
                  f"({type(exc).__name__}: {exc})")

    return fig, summary_df, stats_records


def _write_pdf_report(path, fig, groups, all_summaries, labels, colors,
                      summary_df, stats_df, panels, theme, palette):
    """Multi-page PDF: cover + figure, parameters & folders, statistics."""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    pal = palette
    with PdfPages(path) as pdf:
        # ── Page 1: the comparison figure itself ──────────────────────────────
        pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches="tight")

        # ── Page 2: cover / parameters ────────────────────────────────────────
        page2 = plt.figure(figsize=(8.5, 11), facecolor=pal["BG"])
        page2.text(0.5, 0.96, "sptPALM Comparison Report",
                   ha="center", fontsize=18, fontweight="bold", color=pal["TXT"])

        meta_lines = [
            f"Theme:              {theme}",
            f"Panels rendered:    {', '.join(sorted(panels))}",
            f"Number of groups:   {len(groups)}",
            "",
            "Groups:",
        ]
        for i, g in enumerate(groups):
            meta_lines.append(
                f"  • {labels[i]}   "
                f"(n={len(all_summaries[i])} folder(s), "
                f"colour {colors[i]})")
        meta_lines.append("")
        meta_lines.append("Folders:")
        for i in range(len(groups)):
            meta_lines.append(f"  [{labels[i]}]")
            for f in groups[i]["folders"]:
                meta_lines.append(f"    {f}")
            meta_lines.append("")

        page2.text(0.06, 0.92, "\n".join(meta_lines),
                   ha="left", va="top", fontsize=9, family="monospace",
                   color=pal["TXT"])
        pdf.savefig(page2, facecolor=pal["BG"], bbox_inches="tight")
        plt.close(page2)

        # ── Page 3: per-replicate scalar summary table ────────────────────────
        if len(summary_df):
            page3 = plt.figure(figsize=(11, 8.5), facecolor=pal["BG"])
            page3.text(0.5, 0.96, "Per-replicate scalar metrics",
                       ha="center", fontsize=14, fontweight="bold",
                       color=pal["TXT"])
            ax = page3.add_axes([0.04, 0.04, 0.92, 0.86])
            ax.axis("off")
            disp = summary_df.copy()
            for c in disp.select_dtypes(include="float").columns:
                disp[c] = disp[c].apply(
                    lambda x: f"{x:.4g}" if np.isfinite(x) else "")
            disp["folder"] = disp["folder"].apply(
                lambda p: "..." + p[-40:] if isinstance(p, str) and len(p) > 43 else p)
            tbl = ax.table(cellText=disp.values.tolist(),
                           colLabels=list(disp.columns), loc="center",
                           cellLoc="left")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.scale(1, 1.2)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor(pal["GRD"])
                cell.set_text_props(color=pal["TXT"])
                cell.set_facecolor(pal["PNL"] if r > 0 else pal["BG"])
                if r == 0:
                    cell.set_text_props(weight="bold", color=pal["TXT"])
            pdf.savefig(page3, facecolor=pal["BG"], bbox_inches="tight")
            plt.close(page3)

        # ── Page 4: statistical tests ─────────────────────────────────────────
        if len(stats_df):
            page4 = plt.figure(figsize=(11, 8.5), facecolor=pal["BG"])
            page4.text(0.5, 0.96, "Statistical tests",
                       ha="center", fontsize=14, fontweight="bold",
                       color=pal["TXT"])
            ax = page4.add_axes([0.03, 0.04, 0.94, 0.86])
            ax.axis("off")
            disp = stats_df.copy()
            for c in ("p_value", "mean_a", "mean_b", "sem_a", "sem_b"):
                if c in disp.columns:
                    disp[c] = disp[c].apply(
                        lambda x: f"{x:.4g}" if isinstance(x, (int, float)) and np.isfinite(x) else x)
            tbl = ax.table(cellText=disp.values.tolist(),
                           colLabels=list(disp.columns), loc="center",
                           cellLoc="left")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.scale(1, 1.2)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor(pal["GRD"])
                cell.set_text_props(color=pal["TXT"])
                cell.set_facecolor(pal["PNL"] if r > 0 else pal["BG"])
                if r == 0:
                    cell.set_text_props(weight="bold", color=pal["TXT"])
            pdf.savefig(page4, facecolor=pal["BG"], bbox_inches="tight")
            plt.close(page4)


if __name__ == "__main__":
    main()
