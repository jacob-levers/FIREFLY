"""Shared constants and leaf helpers for the FIREFLY analysis pipeline.

Extracted from sptpalm_analysis.py as the first step of modularising that
file.  These have no dependencies on the rest of the pipeline, so every
other extracted module (loaders, localize, diffusion, …) can import from
here without risking an import cycle.  sptpalm_analysis re-exports these
names so existing `import sptpalm_analysis as s; s.N_CPUS` call sites keep
working unchanged.
"""
from __future__ import annotations

import io as _io
import multiprocessing
import os
import sys

from tqdm import tqdm

# ── Canonical calibration defaults ───────────────────────────────────────────
# Diffusion coefficients scale as D ∝ px²/Δt, so the pixel size and frame
# interval assumed when a file has neither an override nor embedded metadata
# MUST be identical everywhere.  They had silently drifted — post-process used
# Δt = 0.03 s and the comparison reader used 0.05 s, while the worker/GUI used
# 0.02 s — which rescaled D and the MSD axis between code paths.  Every default
# site now references these so they can never diverge again.
DEFAULT_PIXEL_SIZE_UM = 0.106
DEFAULT_FRAME_INTERVAL_S = 0.02

# D (µm²/s) separating immobile from mobile.  Constals et al. (2015),
# Neuron 85:787-803, use the resolution-floor criterion
#
#     D_threshold = resolution² / (4 × n_frames × frame_interval).
#
# Their 80 nm resolution, four-frame criterion, and 50 ms interval give the
# paper's 0.008 µm²/s boundary.  Applying the same criterion at FIREFLY's
# default 20 ms interval gives 0.020 µm²/s.  This is an acquisition-dependent
# default, not a universal physical constant; users must justify their assay's
# resolution and observation-time assumptions.  The published adult-Drosophila
# sptPALM method reports 0.021 µm²/s and its built-in preset retains that value.
# Lives here, not in fa_diffusion, so the worker can reach it without importing
# numpy/scipy/trackpy at module scope.  fa_diffusion re-exports it.
_MOBILE_D_REFERENCE_RESOLUTION_UM = 0.080
_MOBILE_D_REFERENCE_N_FRAMES = 4
MOBILE_D_THRESHOLD_DEFAULT = (
    _MOBILE_D_REFERENCE_RESOLUTION_UM ** 2
    / (4 * _MOBILE_D_REFERENCE_N_FRAMES * DEFAULT_FRAME_INTERVAL_S)
)

# Worker-count default for the parallel localisation / MSD passes.
N_CPUS = multiprocessing.cpu_count()


def _cpu_core_budget() -> int:
    """Per-process CPU core ceiling for the parallel paths (loaders, detection,
    BLAS thread-pools).

    HYPERFLY runs several files concurrently, each in its own worker process; it
    sets ``FIREFLY_CPU_CORE_BUDGET`` to that file's slice of the machine so the
    nested pools don't each grab all the cores (N files × all cores = massive
    oversubscription — e.g. 11 files × 128 ≈ 1400 TIF-decode threads).  Unset →
    all cores, i.e. the original single-file behaviour.
    """
    try:
        b = int(os.environ.get("FIREFLY_CPU_CORE_BUDGET", "0"))
    except (ValueError, TypeError):
        b = 0
    return b if b > 0 else int(N_CPUS)


def safe_process_workers(n: int) -> int:
    """Clamp a requested worker count to what ``concurrent.futures.
    ProcessPoolExecutor`` actually allows.

    On Windows, ProcessPoolExecutor is hard-capped at **61** workers: its
    implementation waits on every worker's sentinel handle via
    ``_winapi.WaitForMultipleObjects``, which can track at most 64 handles
    (minus a few reserved), so ``max_workers > 61`` raises
    ``ValueError: max_workers must be <= 61`` at pool construction — before any
    work runs.  This bites many-core Windows boxes (e.g. a 128-core EPYC) where
    the default worker count is ``os.cpu_count()``.

    Only ProcessPoolExecutor has this limit — ``ThreadPoolExecutor`` and
    ``multiprocessing.Pool`` are exempt, so callers using those should NOT
    route through this helper.
    """
    n = max(1, int(n))
    return min(n, 61) if sys.platform.startswith("win") else n


class _Cancelled(Exception):
    """Raised inside loaders/pipeline when a stop_event fires mid-run."""
    pass


# Bar-free progress format: the GUI log panel is plain text, so the animated
# `|████░░░░|` glyph is just noise that re-renders on every tick.  Keep the
# numbers that actually inform — label, percentage, count, and rate — and drop
# the bar.  e.g.  "  Preprocessing: 100% (16000/16000) 1820 fr/s"
_BAR_FORMAT = "{desc}: {percentage:3.0f}% ({n_fmt}/{total_fmt}) {rate_fmt}"


def _tqdm(*args, **kwargs):
    """tqdm wrapper that writes to stdout (captured by the GUI log panel).
    Falls back to a no-op StringIO if stdout is somehow invalid."""
    out = sys.stdout if (sys.stdout is not None) else _io.StringIO()
    kwargs.setdefault("file", out)
    # Disable ANSI colour codes — the log panel is plain text.
    kwargs.setdefault("colour", None)
    # Drop the fake `████` bar, keep the percentage / count / rate readout.
    kwargs.setdefault("bar_format", _BAR_FORMAT)
    return tqdm(*args, **kwargs)


def _dim_size(v, default=1):
    """aicspylibczi returns dims as int or (start, size) tuple."""
    if isinstance(v, tuple):
        return int(v[1])
    return int(v) if v is not None else default


# ── Canonical motion-class colours ────────────────────────────────────────────
# ONE source of truth so EVERY view draws the same motion class in the same
# colour — the single-run figure, the group-comparison figure, AND the napari
# overlay.  They had drifted apart (the comparison figure used a different
# scheme from the viewer + single-run figure), which read as a mislabel.
# Standardised on the VIEWER's scheme: Immobile = red, Confined = orange,
# Brownian = blue, Directed = green, Unknown = grey.
MOTION_CLASS_ORDER  = ["Immobile", "Confined", "Brownian", "Directed"]

# Per-theme motion-class palettes.  The DATA colours must suit the figure
# background, and the Publication theme must be colour-blind safe (it is the one
# people put in papers).  Keep Immobile=warm-red, Confined=orange, Brownian=blue,
# Directed=green semantics everywhere so a class reads the same across themes.
MOTION_CLASS_COLORS_BY_THEME = {
    # Dark — unchanged from the historical palette so existing Dark exports stay
    # pixel-identical.  AMOLED reuses it (it's just a blacker Dark).
    "Dark": {
        "Immobile": "#e05252", "Confined": "#f5a623",
        "Brownian": "#4a90d9", "Directed": "#7ed321", "Unknown": "#aaaaaa",
    },
    "AMOLED": {
        "Immobile": "#e05252", "Confined": "#f5a623",
        "Brownian": "#4a90d9", "Directed": "#7ed321", "Unknown": "#aaaaaa",
    },
    # Light — deeper, more-saturated hues so fills/lines don't wash out on white.
    "Light": {
        "Immobile": "#d1242f", "Confined": "#bc4c00",
        "Brownian": "#0969da", "Directed": "#1a7f37", "Unknown": "#6e7781",
    },
    # Publication — Okabe-Ito colour-blind-safe palette (deuteranopia /
    # protanopia / tritanopia distinguishable, and separable in grayscale print).
    "Publication": {
        "Immobile": "#d55e00",   # vermillion
        "Confined": "#e69f00",   # orange
        "Brownian": "#0072b2",   # blue
        "Directed": "#009e73",   # bluish-green
        "Unknown":  "#999999",   # neutral grey
    },
}

# Legacy alias — equals the Dark palette, so existing importers (napari overlay,
# console summaries) are byte-for-byte unchanged.
MOTION_CLASS_COLORS = MOTION_CLASS_COLORS_BY_THEME["Dark"]


def motion_class_colors(theme="Dark"):
    """Motion-class colour dict for a figure theme (Immobile/Confined/Brownian/
    Directed/Unknown).  Falls back to the Dark palette for unknown theme names."""
    return MOTION_CLASS_COLORS_BY_THEME.get(
        (theme or "Dark").strip(), MOTION_CLASS_COLORS_BY_THEME["Dark"])


def label_text_color(hexcol):
    """Black or white label text — whichever has the higher WCAG contrast
    against the given fill colour.

    Used for on-segment labels in stacked bars / waffle-style charts so the
    text stays legible on every segment.  Robust across themes including the
    Publication colour-blind palette, where a naive luminance cut mislabels
    amber.  Shared by the single-run figure and the comparison figure so a
    segment label is coloured the same way everywhere.
    """
    h = str(hexcol).lstrip("#")
    if len(h) < 6:
        return "#ffffff"
    try:
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    except ValueError:
        return "#ffffff"

    def _lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    L = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
    # contrast(black) = (L+0.05)/0.05 ; contrast(white) = 1.05/(L+0.05)
    return "#101010" if (L + 0.05) / 0.05 >= 1.05 / (L + 0.05) else "#ffffff"
