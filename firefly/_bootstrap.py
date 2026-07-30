"""Safe GUI launch helpers shared by source and wheel entry modules.

The launcher must remain a real ``__main__`` module because multiprocessing's
``spawn`` mode re-imports it in worker children.  Keep Qt imports inside
``main`` so those children never construct a GUI.
"""
from __future__ import annotations

import multiprocessing
import os
import sys


def prepare_process_streams() -> None:
    """Give frozen parent and worker processes usable stdio streams.

    Windowed PyInstaller applications set stdout/stderr to ``None``.  This
    needs to run at module import time in both launch modules, before a spawned
    worker has a chance to emit its own traceback during teardown.
    """
    if not getattr(sys, "frozen", False):
        return
    devnull = open(os.devnull, "w")
    if sys.stdout is None:
        sys.stdout = devnull
    if sys.stderr is None:
        sys.stderr = devnull


def main() -> int:
    """Launch FIREFLY from a protected multiprocessing entry boundary."""
    prepare_process_streams()
    multiprocessing.freeze_support()

    if os.environ.get("FIREFLY_SELFTEST"):
        from firefly._selftest import run_selftest
        return int(run_selftest(os.environ["FIREFLY_SELFTEST"]))

    # CI and frozen-package smoke tests run on GPU-less runners.  The software
    # renderer is opt-in and leaves normal hardware-accelerated launches alone.
    if (os.environ.get("SPTPALM_READY_MARKER")
            and not os.environ.get("QT_QUICK_BACKEND")):
        os.environ["QT_QUICK_BACKEND"] = "software"

    from firefly.ui.app_qml import main as app_main
    result = app_main()
    return int(result) if result is not None else 0
