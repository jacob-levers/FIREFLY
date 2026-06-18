#!/usr/bin/env python3
"""FIREFLY launch entry point — QML / Qt Quick front-end (Phase 2 UI rewrite).

Sibling of ``run_firefly.py`` that boots the new QML UI instead of the legacy
Widgets app, sharing the same analysis core. Both stay runnable until parity;
the default launcher switches to this one then. The spawn re-import guard +
``freeze_support()`` must remain here (multiprocessing re-imports this module as
``__main__`` in each worker child) once the worker is wired up.
"""
import os
import sys

if getattr(sys, "frozen", False):
    _devnull = open(os.devnull, "w")
    if sys.stdout is None:
        sys.stdout = _devnull
    if sys.stderr is None:
        sys.stderr = _devnull

import multiprocessing  # noqa: E402

from firefly.ui.app_qml import main  # noqa: E402

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
