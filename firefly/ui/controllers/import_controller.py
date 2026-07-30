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
from PySide6.QtCore import QObject, Property, QUrl, Signal, Slot

from firefly.analysis.fa_constants import (DEFAULT_PIXEL_SIZE_UM,
                                           DEFAULT_FRAME_INTERVAL_S)

_IMAGE_FILTER = ("Images or localisations (*.czi *.tif *.tiff *.csv *.txt *.tsv);;"
                 "Image stacks (*.czi *.tif *.tiff);;"
                 "Localisations (*.csv *.txt *.tsv);;All files (*)")


def _human_size(n: int) -> str:
    """Compact human-readable byte size (e.g. 1.4 GB)."""
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return (f"{int(f)} {unit}" if unit == "B" else f"{f:.1f} {unit}")
        f /= 1024
    return f"{f:.1f} TB"


# Preview colormaps live in preview_loader (shared with the ROI editor).
from firefly.ui.controllers.params.preview_loader import (        # noqa: E402
    PREVIEW_CMAPS as _PREVIEW_CMAPS, render_projection as _project_to_qimage)


class ImportController(QObject):
    filePathChanged = Signal()
    outDirChanged = Signal()
    pixelSizeChanged = Signal()
    frameIntervalChanged = Signal()
    overridePxChanged = Signal()
    overrideFiChanged = Signal()
    probeChanged = Signal()
    seriesFilesChanged = Signal()
    thumbChanged = Signal()
    previewCmapChanged = Signal()
    csvPresetChanged = Signal()
    bgImagePathChanged = Signal()
    batchModeChanged = Signal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self._s = settings
        self._batch_mode = False        # Import tab: single (False) vs batch (True);
                                        # shared so the landing card + ROI viewer agree
        # Clean slate on every launch — do NOT restore the last input file /
        # output dir.  The app should open with nothing loaded (parity with the
        # batch queue, which also starts empty), so a stale recording from the
        # previous session never silently becomes the next run's input.  The
        # QSettings keys are still WRITTEN by the setters (harmless), just not
        # read back at startup.
        self._file = ""
        self._outdir = ""
        # Output tracks the recording's folder until the user picks one explicitly.
        self._output_explicit = False
        self._fmt = ""
        self._frames = 0
        self._is_csv = False
        self._size_str = ""
        self._read_error = ""             # non-empty → file couldn't be read (corrupt?)
        # When the picked file is part of a multi-file series, single analysis
        # auto-combines the siblings (load_file with files=None).  Surface the
        # exact list the run will load.  Probed synchronously on file-pick — it's
        # only the siblings of ONE recording (a handful of metadata reads), and a
        # background thread here would capture `self` and could destroy the
        # controller (+ its QTimer) from the worker thread on teardown.
        self._series_files = []           # [{name, path, frames}], primary first
        # External-CSV options (shown only for localisation-table input): the
        # source-format preset (persisted) and an optional background image (kept
        # per-session — it's tied to the current recording, so a new file clears it).
        self._bg_image = ""
        self._thumb = None                # QImage preview of the recording
        self._thumb_token = 0
        self._proj = None                 # cached max-projection array (recolour without re-read)
        # Pixel size / frame interval read from the picked file's embedded
        # metadata (None when the file has none, e.g. a CSV).  Used to fill the
        # sidebar's Imaging-metadata fields so they show the file's real values.
        self._meta_px = None
        self._meta_fi = None
        # The preview colour is owned by the ROI editor's dropdown; we just read
        # the shared 'ui/preview_cmap' key so the thumbnail matches it.
        self._preview_cmap = self._current_cmap()
        # Re-fill the metadata fields when the user unticks an Override in the
        # sidebar (they should revert to the file's detected value).
        try:
            self._s.changed.connect(self._on_settings_changed)
        except Exception:
            pass
        if self._file:
            self._probe()

    def _on_settings_changed(self, key):
        if key == "analysis/override_px" and not self.overridePx:
            self._apply_metadata(px=True, fi=False)
        elif key == "analysis/override_fi" and not self.overrideFi:
            self._apply_metadata(px=False, fi=True)

    def _current_cmap(self):
        c = self._s.get_str("ui/preview_cmap", "Grayscale")
        return c if c in _PREVIEW_CMAPS else "Grayscale"

    # ── single / batch mode (shared with the landing card + ROI viewer) ──
    @Property(bool, notify=batchModeChanged)
    def batchMode(self):
        return self._batch_mode

    @Slot(bool)
    def setBatchMode(self, on):
        if bool(on) != self._batch_mode:
            self._batch_mode = bool(on)
            self.batchModeChanged.emit()

    # ── input file ───────────────────────────────────────────────────────
    @Property(str, notify=filePathChanged)
    def filePath(self):
        return self._file

    @filePath.setter
    def filePath(self, v):
        if v != self._file:
            self._file = v
            self._s.set("analysis/file", v)
            if self._bg_image:                  # bg image is tied to the old file
                self._bg_image = ""
                self.bgImagePathChanged.emit()
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
    # Owned by the parameter sidebar (Imaging-metadata section); there's no
    # calibration UI on the Import tab.  These properties are live views over
    # the shared QSettings keys, so a sidebar edit is exactly what
    # params_builder and VisualiseController read back.
    @Property(bool, notify=overridePxChanged)
    def overridePx(self):
        return self._s.get_bool("analysis/override_px", False)

    @overridePx.setter
    def overridePx(self, v):
        v = bool(v)
        if v != self.overridePx:
            self._s.set("analysis/override_px", v)
            self.overridePxChanged.emit()

    @Property(float, notify=pixelSizeChanged)
    def pixelSize(self):
        return self._s.get_float("analysis/pixel_size", DEFAULT_PIXEL_SIZE_UM)

    @pixelSize.setter
    def pixelSize(self, v):
        v = float(v)
        if v != self.pixelSize:
            self._s.set("analysis/pixel_size", v)
            self.pixelSizeChanged.emit()

    @Property(bool, notify=overrideFiChanged)
    def overrideFi(self):
        return self._s.get_bool("analysis/override_fi", False)

    @overrideFi.setter
    def overrideFi(self, v):
        v = bool(v)
        if v != self.overrideFi:
            self._s.set("analysis/override_fi", v)
            self.overrideFiChanged.emit()

    @Property(float, notify=frameIntervalChanged)
    def frameInterval(self):
        return self._s.get_float("analysis/frame_interval", DEFAULT_FRAME_INTERVAL_S)

    @frameInterval.setter
    def frameInterval(self, v):
        v = float(v)
        if v != self.frameInterval:
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

    # External-CSV source format (one of fa_loaders._CSV_PRESETS keys, or "auto").
    # Auto-detect works for most exports; the picker is the override for when it
    # can't (and is consumed by params_builder → load_external_locs).
    @Property(str, notify=csvPresetChanged)
    def csvPreset(self):
        return self._s.get_str("import/csv_preset", "auto")

    @csvPreset.setter
    def csvPreset(self, v):
        v = str(v) or "auto"
        if v != self.csvPreset:
            self._s.set("import/csv_preset", v)
            self.csvPresetChanged.emit()

    # Optional background image for a CSV-only run (gives the figure a real
    # max-projection instead of a blank canvas).  Per-session, not persisted.
    @Property(str, notify=bgImagePathChanged)
    def bgImagePath(self):
        return self._bg_image

    @bgImagePath.setter
    def bgImagePath(self, v):
        v = str(v or "")
        if v != self._bg_image:
            self._bg_image = v
            self.bgImagePathChanged.emit()

    @Property(str, notify=probeChanged)
    def fileSize(self):
        return self._size_str

    @Property(bool, notify=filePathChanged)
    def hasFile(self):
        return bool(self._file) and os.path.isfile(self._file)

    # Non-empty when the selected recording couldn't be read (corrupt / truncated
    # / wrong contents) — surfaced up-front so the user isn't told "no preview for
    # this type" or left to discover it only when the run fails.
    @Property(str, notify=probeChanged)
    def readError(self):
        return self._read_error

    @Property(bool, notify=probeChanged)
    def hasReadError(self):
        return bool(self._read_error)

    # ── output ───────────────────────────────────────────────────────────
    # The output folder auto-follows the recording's directory; it only stops
    # tracking once the user picks an output explicitly (Browse).
    @Property(bool, notify=outDirChanged)
    def outputExplicit(self):
        return self._output_explicit

    def _follow_output_to(self, path):
        """Point output at the recording's folder, unless explicitly overridden."""
        if not self._output_explicit and path:
            self.outDir = os.path.dirname(path)

    @Slot(str)
    def dropFile(self, url):
        """Accept a file dropped onto the Import tab (file:// URL or path)."""
        path = QUrl(url).toLocalFile() if url.startswith("file:") else url
        if path and os.path.isfile(path):
            self.filePath = path
            self._follow_output_to(path)

    # ── live preview thumbnail (one representative frame of the recording) ─
    @Property(int, notify=thumbChanged)
    def thumbToken(self):
        return self._thumb_token

    @Property(bool, notify=thumbChanged)
    def hasThumb(self):
        return self._thumb is not None and not self._thumb.isNull()

    def thumb_image(self):
        """Current preview QImage (read by the 'importthumb' image provider)."""
        return self._thumb

    # ── preview colormap ──────────────────────────────────────────────────
    # The colour is chosen in the ROI editor and shared via 'ui/preview_cmap';
    # the Import thumbnail just reflects it.
    @Property(str, notify=previewCmapChanged)
    def previewCmap(self):
        return self._preview_cmap

    @Slot()
    def refreshPreviewColour(self):
        """Re-render the cached projection after the ROI editor changed the
        shared colour (wired to RoiController.cmapChanged)."""
        self._apply_cmap()
        self.previewCmapChanged.emit()

    def _probe(self):
        self._fmt, self._frames, self._is_csv, self._size_str = "", 0, False, ""
        self._read_error = ""
        self._meta_px = self._meta_fi = None
        p = self._file
        if p and os.path.isfile(p):
            try:    self._size_str = _human_size(os.path.getsize(p))
            except Exception: self._size_str = ""
            ext = os.path.splitext(p)[1].lower()
            if ext in (".csv", ".txt", ".tsv"):
                self._is_csv, self._fmt = True, "localisations"
            elif ext in (".tif", ".tiff"):
                self._fmt = "TIFF"
                try:
                    import tifffile
                    with tifffile.TiffFile(p) as t:
                        self._frames = len(t.pages)
                        try:
                            from firefly.analysis.fa_loaders import _parse_ome_metadata
                            self._meta_px, self._meta_fi = _parse_ome_metadata(t)
                        except Exception:
                            pass
                    # a TIFF that opens but has no image frames is unusable
                    if self._frames <= 0:
                        self._read_error = ("This TIFF has no image frames — "
                                            "it may be corrupt or incomplete.")
                except Exception:
                    # opening/parsing failed: don't mislabel a corrupt file as an
                    # unsupported type — say plainly it couldn't be read
                    self._read_error = ("Couldn't read this file — it may be "
                                        "corrupt or incomplete.")
            elif ext == ".czi":
                self._fmt = "CZI"
                try:
                    from aicspylibczi import CziFile
                    czi = CziFile(p)
                    dims = czi.dims               # e.g. "TCYX"
                    if "T" in dims:
                        self._frames = int(czi.size[dims.index("T")])
                    try:
                        from firefly.analysis.fa_loaders import _parse_czi_metadata
                        m = _parse_czi_metadata(czi.meta if hasattr(czi, "meta")
                                                else None)
                        self._meta_px = m.get("pixel_size_um")
                        self._meta_fi = m.get("frame_interval_s")
                    except Exception:
                        pass
                except Exception:
                    self._read_error = ("Couldn't read this file — it may be "
                                        "corrupt or incomplete.")
            else:
                shown = ext or "(no extension)"
                self._read_error = (
                    f"Unsupported input type {shown}. FIREFLY supports "
                    ".czi, .tif, .tiff, .csv, .txt and .tsv files.")
        # Fill the sidebar's Imaging-metadata fields from the file (image inputs
        # only; CSVs carry no metadata) unless the user is overriding.
        self._apply_metadata()
        self.probeChanged.emit()
        self._probe_series(p, self._is_csv)
        self._render_thumb()

    def _apply_metadata(self, px=True, fi=True):
        """Fill pixel size / frame interval from the picked file's embedded
        metadata (falling back to the built-in default when the file has none) —
        matching exactly what the run will use when not overriding.  Skipped for
        CSV input (no image metadata) and for a field the user is overriding."""
        # A rejected drop must not reset the user's calibration to defaults.
        # Keep the helper permissive for its existing direct/test use before a
        # probe has assigned a format.
        if self._is_csv or (self._read_error and self._fmt not in ("TIFF", "CZI")):
            return
        if px and not self.overridePx:
            self.pixelSize = float(self._meta_px or DEFAULT_PIXEL_SIZE_UM)
        if fi and not self.overrideFi:
            self.frameInterval = float(self._meta_fi or DEFAULT_FRAME_INTERVAL_S)

    # Detected-from-file values (None when the file carries no metadata), so the
    # UI can label the fields "from file" vs "default/manual" if it wants to.
    @Property(float, notify=probeChanged)
    def metaPixelSize(self):
        return float(self._meta_px) if self._meta_px else 0.0

    @Property(float, notify=probeChanged)
    def metaFrameInterval(self):
        return float(self._meta_fi) if self._meta_fi else 0.0

    # ── multi-file series (single analysis auto-combines siblings) ────────
    def _probe_series(self, path, is_csv):
        """Discover the sibling files single analysis will auto-combine (the same
        ``_find_tif_series`` / ``_find_czi_series`` the run uses) + their frame
        counts, and publish them for the Import tab.  Only a genuine multi-file
        series (>1 file) is shown."""
        rows = []
        ext = os.path.splitext(path or "")[1].lower()
        if (not is_csv and ext in (".tif", ".tiff", ".czi")
                and path and os.path.isfile(path)):
            try:
                from firefly.analysis.fa_loaders import (_find_tif_series,
                                                         _find_czi_series)
                from firefly.ui.controllers.params.preview_loader import quick_frame_count
                files = (_find_czi_series(path) if ext == ".czi"
                         else _find_tif_series(path))
                if files and len(files) > 1:
                    rows = []
                    for f in files:
                        # series files are always images → 0 frames means the
                        # part couldn't be read (corrupt / truncated)
                        n = int(quick_frame_count(f))
                        rows.append({"name": os.path.basename(f), "path": f,
                                     "frames": n, "unreadable": n <= 0})
            except Exception:
                rows = []
        if rows or self._series_files:
            self._series_files = rows
            self.seriesFilesChanged.emit()
        # A corrupt PART dooms the combined run (load_file reads every sibling),
        # so surface it like the picked-file read error — Alert + Start disabled —
        # unless the picked file already flagged one.
        if any(r.get("unreadable") for r in rows) and not self._read_error:
            self._read_error = ("One or more files in this recording can't be read "
                                "— it may be corrupt or incomplete.")
            self.probeChanged.emit()

    @Property("QVariantList", notify=seriesFilesChanged)
    def seriesFiles(self):
        return list(self._series_files)

    @Property(int, notify=seriesFilesChanged)
    def seriesCount(self):
        return len(self._series_files)

    @Property(int, notify=seriesFilesChanged)
    def seriesFrameTotal(self):
        return int(sum(r.get("frames", 0) for r in self._series_files))

    @Property(int, notify=seriesFilesChanged)
    def seriesUnreadableCount(self):
        return sum(1 for r in self._series_files if r.get("unreadable"))

    def _render_thumb(self):
        """Compute the recording's max-intensity projection (a cheap sampled
        projection — never loads the whole stack) and render it with the current
        colormap."""
        self._proj = None
        if self._file and not self._is_csv:
            try:
                from firefly.ui.controllers.params.preview_loader import sampled_max_projection
                self._proj = sampled_max_projection(self._file)
            except Exception:
                self._proj = None
        self._apply_cmap()

    def _apply_cmap(self):
        """(Re)render the cached projection with the shared colour + emit."""
        self._preview_cmap = self._current_cmap()      # always reflect the latest pick
        self._thumb = None
        if self._proj is not None:
            try:
                img = _project_to_qimage(self._proj, self._preview_cmap)
                self._thumb = img if (img is not None and not img.isNull()) else None
            except Exception:
                self._thumb = None
        self._thumb_token += 1
        self.thumbChanged.emit()

    # ── native pickers (called from QML) ─────────────────────────────────
    @Slot()
    def browseFile(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select input file",
            self._outdir or os.path.expanduser("~"), _IMAGE_FILTER)
        if path:
            self.filePath = path
            self._follow_output_to(path)

    @Slot()
    def browseBgImage(self):
        """Pick an optional background image-stack for a CSV-only run."""
        start = (self._outdir or os.path.dirname(self._file)
                 or os.path.expanduser("~"))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select background image (optional)", start,
            "Image stacks (*.czi *.tif *.tiff);;All files (*)")
        if path:
            self.bgImagePath = path

    @Slot()
    def browseOutDir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select output folder",
            self._outdir or os.path.expanduser("~"))
        if path:
            self._output_explicit = True
            self._s.set("analysis/output_explicit", True)
            self.outDir = path
