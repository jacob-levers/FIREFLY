"""Atomic file writes.

A direct ``df.to_csv(path)`` / ``open(path, "w")`` leaves a *truncated* file at
the final path if the write is interrupted (disk full, crash, kill) — and a
downstream loader will happily half-read it.  These helpers write to a sibling
``<path>.tmp`` first, then ``os.replace`` it onto the final path: ``os.replace``
is atomic on the same filesystem, so the final path only ever holds the previous
complete file or the new complete file, never a partial one.

Stdlib only (no third-party imports) so it stays trivially bundle-safe.
"""
from __future__ import annotations

import os
from contextlib import contextmanager


def atomic_to_csv(df, path, **to_csv_kwargs):
    """``df.to_csv(path, **kwargs)`` written atomically (tmp + ``os.replace``).

    Forwards ``to_csv_kwargs`` unchanged (so the caller keeps its ``index=`` /
    ``columns=`` / etc.).  On any failure the temp file is removed and the
    exception re-raised — matching the raise-on-error semantics of a plain
    ``to_csv`` while guaranteeing the final path is never a truncated file.
    Returns ``path``.
    """
    tmp = f"{path}.tmp"
    try:
        df.to_csv(tmp, **to_csv_kwargs)
        os.replace(tmp, path)
    except BaseException:
        _cleanup(tmp)
        raise
    return path


@contextmanager
def atomic_write(path, mode="w", **open_kwargs):
    """Context manager: ``with atomic_write(path, "w", newline="") as fh:`` —
    write through ``fh`` (e.g. a ``csv.writer``), and on clean exit the temp file
    is ``os.replace``-d onto ``path``.  On exception the temp file is removed and
    the original ``path`` is left untouched.
    """
    tmp = f"{path}.tmp"
    try:
        with open(tmp, mode, **open_kwargs) as fh:
            yield fh
        os.replace(tmp, path)
    except BaseException:
        _cleanup(tmp)
        raise


def _cleanup(tmp):
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except Exception:
        pass
