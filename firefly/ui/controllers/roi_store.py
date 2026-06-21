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


class RoiOverrideStore:
    """Per-file ROI-settings override (abspath → spec dict of the analysis/roi_*
    values: ``roi_mode``/``roi_auto_method``/``roi_threshold``/``roi_mask_mode``/
    ``roi_bg_sigma`` in their settings-label form).

    The left sidebar holds the DEFAULT ROI applied to every file; a file with an
    entry here overrides that default for THAT file only (set from the Preview &
    ROI viewer).  Read by ``params_builder`` at run time.  Session-only, like
    ``RoiStore``.
    """

    def __init__(self):
        self._by_file: dict = {}

    @staticmethod
    def _key(path):
        return os.path.abspath(path) if path else ""

    def get(self, path):
        return self._by_file.get(self._key(path))

    def set(self, path, spec):
        k = self._key(path)
        if spec:
            self._by_file[k] = dict(spec)
        else:
            self._by_file.pop(k, None)

    def has(self, path):
        return self._key(path) in self._by_file

    def clear(self, path):
        self._by_file.pop(self._key(path), None)
