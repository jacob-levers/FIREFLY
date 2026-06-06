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
import sys

from tqdm import tqdm

# Worker-count default for the parallel localisation / MSD passes.
N_CPUS = multiprocessing.cpu_count()


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
