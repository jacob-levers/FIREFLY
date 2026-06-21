#!/usr/bin/env python3
"""FIREFLY launch entry point.

Thin wrapper around the app's ``main`` so the repo root stays clean while the
application lives in the ``firefly/`` package.  Kept as the spawn "main module":
multiprocessing (start method = spawn) re-imports this file in each worker child,
so the ``freeze_support()`` + ``__main__`` guard below must remain here to stop
the GUI from relaunching recursively in those children.

The front-end is the QML / Qt Quick app (``firefly.ui.app_qml``).  It's imported
LAZILY inside the ``__main__`` guard so a spawned worker child (which re-imports
this file but never runs ``main``) doesn't load Qt.
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


if __name__ == "__main__":
    multiprocessing.freeze_support()
    # CI/packaging smoke runs the frozen app on a GPU-less runner where the Qt
    # Quick scene graph can't create a hardware (D3D/Metal) context and hangs at
    # first paint.  When the smoke handshake is active (SPTPALM_READY_MARKER set)
    # fall back to the software renderer so the window still shows + repaints.
    # Real machines (with a GPU) never set the marker, so they keep hardware.
    if os.environ.get("SPTPALM_READY_MARKER") and not os.environ.get("QT_QUICK_BACKEND"):
        os.environ["QT_QUICK_BACKEND"] = "software"
    from firefly.ui.app_qml import main   # lazy: worker children never reach here
    main()
