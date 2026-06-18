"""RoiStore — per-file manual-polygon ROI store (Phase 6c).

Mirrors the Widgets ``self._roi_polygons`` dict: maps an input file's absolute
path to its drawn polygon(s) (each a list of ``(y, x)`` vertices).  Shared
between RoiController (writes when the user draws) and params_builder (reads into
the ``roi_polygon`` key — sent regardless of ROI mode, matching
``_build_params_for_file``).  Plain dict-like; no Qt.
"""
from __future__ import annotations

import os


class RoiStore:
    def __init__(self):
        self._by_file: dict = {}

    @staticmethod
    def _key(path):
        return os.path.abspath(path) if path else ""

    def get(self, path):
        """Polygons for a file as ``[[(y, x), …], …]`` or None."""
        return self._by_file.get(self._key(path))

    def set(self, path, polygons):
        k = self._key(path)
        if polygons:
            self._by_file[k] = [[(float(y), float(x)) for y, x in poly]
                                for poly in polygons]
        else:
            self._by_file.pop(k, None)

    def has(self, path):
        return bool(self._by_file.get(self._key(path)))

    def clear(self, path):
        self._by_file.pop(self._key(path), None)
