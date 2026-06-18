"""ImportController — backs the QML Import tab.

Owns the input file + output folder + calibration (pixel size / frame interval,
each with a metadata-override flag), a cheap format/frame-count probe, and the
native file/folder pickers. State persists through SettingsController under the
same QSettings keys the Widgets app uses, so the two stay in sync. Mirrors the
Widgets fields: e_file / e_outdir / c_override_px / s_pixel_size /
c_override_fi / s_frame_interval.
"""
from __future__ import annotations

import os

from PySide6 import QtWidgets
from PySide6.QtCore import QObject, Property, Signal, Slot

from firefly.analysis.fa_constants import (DEFAULT_PIXEL_SIZE_UM,
                                           DEFAULT_FRAME_INTERVAL_S)

_IMAGE_FILTER = ("Images or localisations (*.czi *.tif *.tiff *.csv *.txt *.tsv);;"
                 "Image stacks (*.czi *.tif *.tiff);;"
                 "Localisations (*.csv *.txt *.tsv);;All files (*)")


class ImportController(QObject):
    filePathChanged = Signal()
    outDirChanged = Signal()
    pixelSizeChanged = Signal()
    frameIntervalChanged = Signal()
    overridePxChanged = Signal()
    overrideFiChanged = Signal()
    probeChanged = Signal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._s = settings
        self._file = settings.get_str("analysis/file", "")
        self._outdir = settings.get_str("analysis/outdir", "")
        self._override_px = settings.get_bool("analysis/override_px", False)
        self._pixel = settings.get_float("analysis/pixel_size", DEFAULT_PIXEL_SIZE_UM)
        self._override_fi = settings.get_bool("analysis/override_fi", False)
        self._fi = settings.get_float("analysis/frame_interval", DEFAULT_FRAME_INTERVAL_S)
        self._fmt = ""
        self._frames = 0
        self._is_csv = False
        if self._file:
            self._probe()

    # ── input file ───────────────────────────────────────────────────────
    @Property(str, notify=filePathChanged)
    def filePath(self):
        return self._file

    @filePath.setter
    def filePath(self, v):
        if v != self._file:
            self._file = v
            self._s.set("analysis/file", v)
            self.filePathChanged.emit()
            self._probe()

    @Property(str, notify=filePathChanged)
    def fileName(self):
        return os.path.basename(self._file) if self._file else ""

    @Property(str, notify=outDirChanged)
    def outDir(self):
        return self._outdir

    @outDir.setter
    def outDir(self, v):
        if v != self._outdir:
            self._outdir = v
            self._s.set("analysis/outdir", v)
            self.outDirChanged.emit()

    # ── calibration ──────────────────────────────────────────────────────
    @Property(bool, notify=overridePxChanged)
    def overridePx(self):
        return self._override_px

    @overridePx.setter
    def overridePx(self, v):
        v = bool(v)
        if v != self._override_px:
            self._override_px = v
            self._s.set("analysis/override_px", v)
            self.overridePxChanged.emit()

    @Property(float, notify=pixelSizeChanged)
    def pixelSize(self):
        return self._pixel

    @pixelSize.setter
    def pixelSize(self, v):
        v = float(v)
        if v != self._pixel:
            self._pixel = v
            self._s.set("analysis/pixel_size", v)
            self.pixelSizeChanged.emit()

    @Property(bool, notify=overrideFiChanged)
    def overrideFi(self):
        return self._override_fi

    @overrideFi.setter
    def overrideFi(self, v):
        v = bool(v)
        if v != self._override_fi:
            self._override_fi = v
            self._s.set("analysis/override_fi", v)
            self.overrideFiChanged.emit()

    @Property(float, notify=frameIntervalChanged)
    def frameInterval(self):
        return self._fi

    @frameInterval.setter
    def frameInterval(self, v):
        v = float(v)
        if v != self._fi:
            self._fi = v
            self._s.set("analysis/frame_interval", v)
            self.frameIntervalChanged.emit()

    # ── probe (cheap format + frame count) ───────────────────────────────
    @Property(str, notify=probeChanged)
    def fileFormat(self):
        return self._fmt

    @Property(int, notify=probeChanged)
    def frameCount(self):
        return self._frames

    @Property(bool, notify=probeChanged)
    def isCsv(self):
        return self._is_csv

    @Property(bool, notify=filePathChanged)
    def hasFile(self):
        return bool(self._file) and os.path.isfile(self._file)

    def _probe(self):
        self._fmt, self._frames, self._is_csv = "", 0, False
        p = self._file
        if p and os.path.isfile(p):
            ext = os.path.splitext(p)[1].lower()
            if ext in (".csv", ".txt", ".tsv"):
                self._is_csv, self._fmt = True, "localisations"
            elif ext in (".tif", ".tiff"):
                self._fmt = "TIFF"
                try:
                    import tifffile
                    with tifffile.TiffFile(p) as t:
                        self._frames = len(t.pages)
                except Exception:
                    pass
            elif ext == ".czi":
                self._fmt = "CZI"
                try:
                    from aicspylibczi import CziFile
                    czi = CziFile(p)
                    dims = czi.dims               # e.g. "TCYX"
                    if "T" in dims:
                        self._frames = int(czi.size[dims.index("T")])
                except Exception:
                    pass
        self.probeChanged.emit()

    # ── native pickers (called from QML) ─────────────────────────────────
    @Slot()
    def browseFile(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select input file",
            self._outdir or os.path.expanduser("~"), _IMAGE_FILTER)
        if path:
            self.filePath = path
            if not self._outdir:
                self.outDir = os.path.dirname(path)

    @Slot()
    def browseOutDir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select output folder",
            self._outdir or os.path.expanduser("~"))
        if path:
            self.outDir = path
