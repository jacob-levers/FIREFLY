#!/usr/bin/env python3
"""FIREFLY launch entry point.

Thin wrapper around ``firefly.ui.app_qt.main`` so the repo root stays clean
while the application lives in the ``firefly/`` package.  Kept as the spawn
"main module": multiprocessing (start method = spawn) re-imports this file in
each worker child, so the ``freeze_support()`` + ``__main__`` guard below must
remain here to stop the GUI from relaunching recursively in those children.
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

from firefly.ui.app_qt import main  # noqa: E402

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
