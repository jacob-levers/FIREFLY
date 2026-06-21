"""Read FIREFLY's ``__version__`` WITHOUT importing the analysis core.

``firefly.sptpalm_analysis`` does a module-level ``import matplotlib.pyplot``,
which on a FROZEN app's first run builds matplotlib's font cache — scanning the
whole system font set, which can block startup for minutes (the packaging smoke
test then times out at the blank window).  The version is only a short string,
so parse it straight from the bundled ``sptpalm_analysis.py`` file instead of
importing the module.  (The spec bundles that file as data, so it's readable in
the frozen app.)
"""
from __future__ import annotations

import os
import re
import sys


def app_version() -> str:
    """``__version__`` from sptpalm_analysis.py, read without importing it."""
    try:
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            base = os.path.join(sys._MEIPASS, "firefly")
        else:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "sptpalm_analysis.py")
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                m = re.match(r'\s*__version__\s*=\s*["\']([^"\']+)["\']', line)
                if m:
                    return m.group(1)
    except Exception:
        pass
    return "dev"
