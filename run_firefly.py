#!/usr/bin/env python3
"""FIREFLY source/PyInstaller spawn entry point.

Installed wheels use ``python -m firefly``.  Keep this thin root module for
PyInstaller and multiprocessing spawn, which must re-import a real main module
without loading Qt in worker children.
"""
from firefly._bootstrap import main, prepare_process_streams


prepare_process_streams()

if __name__ == "__main__":
    raise SystemExit(main())
