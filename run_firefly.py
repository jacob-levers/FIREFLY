#!/usr/bin/env python3
"""FIREFLY launch entry point.

Thin wrapper around the app's ``main`` so the repo root stays clean while the
application lives in the ``firefly/`` package.  Kept as the spawn "main module":
multiprocessing (start method = spawn) re-imports this file in each worker child,
so the ``freeze_support()`` + ``__main__`` guard below must remain here to stop
the GUI from relaunching recursively in those children.

Front-end selection: the default is the mature Widgets app; set ``FIREFLY_UI=qml``
(env) to launch the new QML / Qt Quick front-end instead.  Both ship in the
frozen bundle; the default flips to QML once side-by-side parity is signed off.
The UI module is imported LAZILY inside ``_select_main`` so a spawned worker
child (which re-imports this file but never runs ``main``) doesn't load Qt.
"""
import os
import sys

# Frozen windowed builds (PyInstaller console=False) set sys.stdout/stderr to
# None.  multiprocessing writes worker tracebacks to sys.stderr during pool
# teardown; with stderr=None that raises a confusing secondary
#   AttributeError: 'NoneType' object has no attribute 'write'
# which masks the real error (e.g. a BrokenPipeError when a pool worker
# outlives a parent that died).  Give every process level — parent and every
# spawn child, since this module is re-imported as __main__ in each — a real
# writable stream so teardown fails quietly.  Genuine crash reporting goes
# through crash_reporter's excepthook + log files, so routing these to
# os.devnull discards only noise.
if getattr(sys, "frozen", False):
    _devnull = open(os.devnull, "w")
    if sys.stdout is None:
        sys.stdout = _devnull
    if sys.stderr is None:
        sys.stderr = _devnull

import multiprocessing  # noqa: E402  (after the stream guard, intentionally)


def _select_main():
    """Return the chosen front-end's ``main`` (imported lazily so worker children
    don't pull in Qt)."""
    ui = os.environ.get("FIREFLY_UI", "").strip().lower()
    if ui in ("qml", "quick", "new"):
        from firefly.ui.app_qml import main as _m
    else:
        from firefly.ui.app_qt import main as _m
    return _m


if __name__ == "__main__":
    multiprocessing.freeze_support()
    _select_main()()
