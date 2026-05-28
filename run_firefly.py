#!/usr/bin/env python3
"""FIREFLY launch entry point.

Thin wrapper around ``firefly.ui.app_qt.main`` so the repo root stays clean
while the application lives in the ``firefly/`` package.  Kept as the spawn
"main module": multiprocessing (start method = spawn) re-imports this file in
each worker child, so the ``freeze_support()`` + ``__main__`` guard below must
remain here to stop the GUI from relaunching recursively in those children.
"""
import multiprocessing

from firefly.ui.app_qt import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
