"""Pytest bootstrap — make the project modules importable from tests/ and
force UTF-8 stdout so the analysis code's non-ASCII progress prints don't
raise UnicodeEncodeError on a cp1252 Windows console (the app itself
redirects stdout, so this only matters under a bare pytest console)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
