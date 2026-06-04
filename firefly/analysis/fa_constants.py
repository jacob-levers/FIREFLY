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
MOTION_CLASS_COLORS = {
    "Immobile": "#e05252",   # red
    "Confined": "#f5a623",   # orange
    "Brownian": "#4a90d9",   # blue
    "Directed": "#7ed321",   # green
    "Unknown":  "#aaaaaa",   # grey
}
