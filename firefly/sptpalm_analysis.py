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
__version__ = "2.76.4"
# v2.76.4 — packaging: drop bundled `*.tests.*` trees (~2,300 modules across
#   pandas/scipy/sklearn/statsmodels/numpy) from the onefile via a no-tests
#   collect_submodules filter — smaller .exe + faster build, no runtime change.
# v2.76.3 — CRITICAL frozen-build fix: bundle scipy's vendored scipy._external
#   (array_api_compat/.numpy.fft etc.) so the analysis worker no longer dies on
#   `import sptpalm_analysis` with ModuleNotFoundError the instant a run starts;
#   + durable worker crash logging (firefly_worker.log/faulthandler) + super-res
#   `np`→`_np` NameError fix.  See sptpalm.spec / CHANGELOG.
# v2.76.0 — napari REMOVED: the interactive viewers (Visualise tab + ROI editor)
#   are now bespoke Qt-only widgets (QGraphicsView/QImage/QPainter + numpy) —
#   no napari, no pyqtgraph, no vispy.  Drops napari + its dependency tree and
#   moves the stack to numpy 2 / Python 3.13.
# v2.75.0 — interactive Track explorer on the Visualise tab: filter the loaded
#   trajectories by D / α / motion-class / length in a sortable table, click a
#   row to centre the viewer + populate the inspector, and export the filtered
#   subset to CSV.
# v2.74.0 — super-resolution reconstruction: every run saves a *_superres.png
#   (Gaussian/histogram render of the localisation cloud), plus an interactive
#   live-tunable layer on the Visualise tab (overlaid on the raw image).
# v2.73.0 — hardening: regression tests for the figure-defaults code; QC flags
#   surfacing the DBSCAN sub-sample / skipped-ROI / dense-field auto-threshold
#   caveats; figure-preview polish (dead proj-cmap trigger removed, shared grid).
# v2.72.1 — new FIREFLY app icon (Windows .ico + macOS .icns + runtime PNG).
# v2.72.0 — Figure-defaults reorg (sub-tabs), panel pickers for both figures,
#   single-sample combined figure is now panel-selectable, real-data preview;
#   FIX: reflow var `_pos` clashed with the van Hove panel → crashed real runs.
# v2.71.0 — Compare figure: quick-glance per-group summary band (trajectory
#   count, median D, median α) at the top; the redundant bottom legend is
#   removed (the band is now the colour/number/n key).
# v2.70.0 — review remediation: CRITICAL drift-correction sign fix (RCC was
#   DOUBLING drift, not removing it; now locked by a synthetic-drift sign test);
#   JDD now subtracts the MSD localisation-error offset so D_JDD agrees with the
#   offset-corrected MSD D; dwell-time τ uses a right-censored exponential MLE;
#   turning-angle & MSS now use frame-contiguous steps (no gap mis-counting);
#   Prism CSV honours underpowered-blanking + configured α; two-way ANOVA no
#   longer drops metrics on cross-group cell-name collisions; SA-linker cycle
#   guard; bounded gap-closing matrix; uint16 preprocess underflow guard.
# v2.69.3 — gaussian-mle / radial-symmetry refiners run their numerics on CPU
#   when the device is MPS (Apple GPUs silently mis-compute the linalg/conv ops,
#   intermittently mis-localising spots); detection stays GPU-accelerated.
# v2.69.2 — linker-dispatch audit fixes: SA linker no longer crashes from the GUI
#   (merge/split kwargs leaked into link_trajectories_sa); nn is canonical
#   frame-to-frame (max_gap=1); unified DEFAULT_LINKER="kalman" forward default;
#   feature-penalty relabelled FIREFLY-specific; MPS bandpass phantom fix;
#   doc-vs-code cleanups; tests/test_linker_dispatch.py.
# v2.69.1 — disambiguate the Simple LAP linker label → "Jaqaman LAP — TrackMate
#   (simple)" (vs "(merge/split)"); settings-migration keeps saved prefs.
# v2.68.0 — TrackMate & palmTRACER linkers (NN, Simple/Full LAP, simulated
#   annealing) via a linker registry; opt-in auto search-range; simplified
#   GPU backend dropdown; Torch CG auto-threshold decoupled from trackpy.
# v2.67.1 — fix: palmTRACER motion classification was blanked when D was taken
# from palmTRACER's native -D file (use_native); now FIREFLY's alpha/motion are
# kept (native D only overrides the D/MSD family), and a blanked cache self-heals
# on load.  No effect on FIREFLY-localised runs.
# TAG: an annotated `v2.67.0` tag is created on the release-prep commit (the one
# that adds the CHANGELOG v2.67.0 section).  The in-app updater compares this
# string against the latest *GitHub* tag, so it will not offer an update until
# that tag is PUSHED / a GitHub release is published — the remaining manual
# release step.  If this branch is squash-merged into main, re-create the tag on
# the resulting commit so it lands on main's history.  (R3-3)
# v2.67.0 — hostile-review remediation. NOTE: this release changes some
# scientific OUTPUTS vs 2.66.x, so a re-analysis of old data may differ:
#   • mobile fraction is now (finite, positive D) >= threshold (was D > threshold
#     over all rows incl. failed NaN fits) — matches the panel; None for an
#     all-immobile dataset (was 0.0).
#   • short (3-lag) tracks with an unmeasurable anomalous exponent are now
#     "Unknown" (were sometimes mislabelled Directed/Immobile).
#   • a 3-component JDD that yields a negative population falls back to 2.
#   • assumed pixel-size/frame-interval defaults are now unified at 0.106 µm /
#     0.02 s everywhere (post-process was 0.03 s, Compare/CLI 0.05 s).
# Plus many robustness/GUI/security fixes that don't change numbers.

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
from firefly.analysis.fa_constants import (N_CPUS, _Cancelled, _tqdm, _dim_size,
                                           safe_process_workers,
                                           DEFAULT_PIXEL_SIZE_UM,
                                           DEFAULT_FRAME_INTERVAL_S)


tp.quiet()


# ══════════════════════════════════════════════════════════════════════════════
#  CZI LOADING + METADATA
# ══════════════════════════════════════════════════════════════════════════════

# CZI/TIF/external loaders now live in fa_loaders; re-exported here so
# existing `sptpalm_analysis.load_file(...)` etc. call sites keep working.
from firefly.analysis.fa_loaders import (
    load_file, load_czi, load_tif, load_external_locs, load_projection_fast,
    _parse_czi_metadata, _parse_ome_metadata, _find_czi_series,
    _find_tif_series, _load_single_czi, _load_single_tif,
    _probe_tif_shape_and_count, _tif_series_nat_key, _autodetect_csv_preset,
    HAS_AICS, HAS_CZIFILE, HAS_TIFFFILE,
)















# Memory / temp-stack management lives in fa_memory; re-exported here so
# existing call sites (and firefly_worker's cleanup_temp_stack_paths
# import) keep working unchanged.
from firefly.analysis.fa_memory import (
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
from firefly.analysis.fa_preprocess import (_preprocess_fast, _preprocess_rolling,
                           preprocess_stack, auto_threshold)
from firefly.analysis.fa_drift import correct_drift
from firefly.analysis.fa_roi import (build_roi_mask_mean, build_roi_mask_perframe,
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
from firefly.analysis.fa_localize import (
    _ram_strategy, _adaptive_chunk_and_workers,
    _fast_preprocess_and_localise, preprocess_and_localise_adaptive,
    preprocess_and_localise_stream, _localise_chunk, _localise_chunk_mp,
    _localise_chunk_mmap_mp, LocaliserBackend, _emit_trackpy_chunk_preview,
    TrackpyBackend, TorchBackend, AtrousWaveletBackend, list_available_backends,
    _resolve_backend, localise_particles, _BACKEND_REGISTRY,
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
from firefly.analysis.fa_linking import (
    link_trajectories, _link_via_trackpy,
)




# ══════════════════════════════════════════════════════════════════════════════
#  MSD + DIFFUSION  (custom parallel — replaces slow tp.imsd)
# ══════════════════════════════════════════════════════════════════════════════

# fa_diffusion extracted; re-exported here.
from firefly.analysis.fa_diffusion import (
    msd_linear, classify_motion, _msd_and_fit_one, compute_msd_and_fit,
    compute_jdd, compute_turning_angles, compute_van_hove, compute_vacf,
    compute_mobile_fraction_over_time,
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

# fa_circular extracted; re-exported here.
from firefly.analysis.fa_circular import (
    compute_circular_statistics, _circ_watson_williams,
    _circ_mardia_watson_wheeler, _circ_wallraff_ktest,
    _circ_kuiper_two_sample, _circ_lin_correlation,
    compute_per_track_mean_angle, _watson_williams_mu_per_replicate,
    compute_circular_comparison_tests, _p_stars,
    save_circular_statistics_pdf, save_comparison_circular_statistics,
    _write_single_group_page,
)


# fa_theme extracted; re-exported here.
from firefly.analysis.fa_theme import (
    _theme_palette, _THEME_REQUIRED_KEYS,
)






























# ══════════════════════════════════════════════════════════════════════════════
#  MOBILE FRACTION OVER TIME
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTER ANALYSIS  (DBSCAN)
# ══════════════════════════════════════════════════════════════════════════════

# fa_clustering extracted; re-exported here.
from firefly.analysis.fa_clustering import (
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

# fa_figure extracted; re-exported here.
from firefly.analysis.fa_figure import (
    _draw_track, make_figure, MC, MORD,
)






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
    p.add_argument("--aggregate", action="store_true", default=False,
                   help="Treat INPUT as a folder: collect every "
                        "firefly_extras/<stem>_summary_metrics.json beneath it "
                        "into one CSV (one row per run, condition inferred from "
                        "the parent folder) and exit.  Combine with "
                        "--output-dir to choose where the CSV is written.")
    return p.parse_args()


def main():
    args    = parse_args()
    t_start = time.perf_counter()

    # ── Aggregate mode: fold a tree of per-run summaries into one CSV ────
    if args.aggregate:
        if not os.path.isdir(args.input):
            sys.exit(f"ERROR: --aggregate expects a folder, got: {args.input}")
        df = aggregate_run_summaries(args.input)
        if df.empty:
            sys.exit("No *_summary_metrics.json files found under: "
                     f"{args.input}")
        out = args.output_dir or os.path.join(args.input, "run_summaries.csv")
        if os.path.isdir(out):
            out = os.path.join(out, "run_summaries.csv")
        df.to_csv(out, index=False)
        groups = ", ".join(sorted(map(str, df["group"].unique())))
        print(f"Aggregated {len(df)} run(s) across [{groups}] -> {out}")
        return

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
        print(f"  WARNING: Pixel size not in metadata. Using "
              f"{DEFAULT_PIXEL_SIZE_UM} um/px.")
        print("  (Override with --pixel-size)")
        pixel_size = DEFAULT_PIXEL_SIZE_UM
    else:
        src = "command line" if args.pixel_size else "CZI metadata"
        print(f"  Pixel size     : {pixel_size} um/px  [{src}]")

    frame_interval = args.frame_interval or meta_fi
    if frame_interval is None:
        print(f"  WARNING: Frame interval not in metadata. Using "
              f"{DEFAULT_FRAME_INTERVAL_S} s.")
        print("  (Override with --frame-interval)")
        frame_interval = DEFAULT_FRAME_INTERVAL_S
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

# fa_palmtracer extracted; re-exported here.
from firefly.analysis.fa_palmtracer import (
    _find_stem, _is_palmtracer_folder, _read_palmtracer_table,
    load_summary_from_palmtracer, load_summary_from_folder,
    save_palmtracer_csvs, aggregate_run_summaries,
)




















# fa_compare extracted; re-exported here.
from firefly.analysis.fa_compare import (
    _stat_test, _stat_test_n, _bar_with_dots_n, compare_groups,
    _write_pdf_report,
)


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










if __name__ == "__main__":
    main()
