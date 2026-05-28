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


def _tqdm(*args, **kwargs):
    """tqdm wrapper that writes to stdout (captured by the GUI log panel).
    Falls back to a no-op StringIO if stdout is somehow invalid."""
    out = sys.stdout if (sys.stdout is not None) else _io.StringIO()
    kwargs.setdefault("file", out)
    # Disable ANSI colour codes — the log panel is plain text.
    kwargs.setdefault("colour", None)
    return tqdm(*args, **kwargs)


def _dim_size(v, default=1):
    """aicspylibczi returns dims as int or (start, size) tuple."""
    if isinstance(v, tuple):
        return int(v[1])
    return int(v) if v is not None else default
