"""RoiController — QML bridge for the Preview & ROI viewer (Phase 4 → batch).

Owns a HEADLESS polygon model (list of ``(y, x)`` vertex lists + an optional open
draft) plus the per-file ROI settings shown in the viewer.  The left sidebar
holds the DEFAULT ROI applied to every file; the viewer edits a PER-FILE override
(RoiOverrideStore) that replaces that default for one file only — the edits are
transient until ``commit`` (Save ROI), which stores the override (if it differs
from the default) and the polygon (RoiStore).  The threshold-mask + raw-frame
previews are produced by CALLING the analysis core (never modifying it).
The public convention is ``(y, x)`` end-to-end.
"""
from __future__ import annotations

from PySide6.QtCore import Property, QObject, Signal, Slot

# label → analysis mask-mode "mode_hint" for the projection / mask builder
_MASK_MODE_HINT = {"Max": "max", "Mean": "mean", "Sum": "sum",
                   "Blink density": "blink"}


def _green_mask_qimage(mask):
    """bool ROI mask → translucent-lime RGBA QImage (transparent where False),
    matching the legacy RoiEditor overlay colour."""
    import numpy as np
    from PySide6.QtGui import QImage
    m = np.asarray(mask).astype(bool)
    h, w = m.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[m] = (51, 255, 77, 96)
    rgba = np.ascontiguousarray(rgba)
    return QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()


class RoiController(QObject):
    polygonsChanged = Signal()
    draftChanged = Signal()
    frameChanged = Signal(int)
    editingChanged = Signal()
    imageChanged = Signal()
    statusMessage = Signal(str)
    cmapChanged = Signal()
    roiSettingsChanged = Signal()       # mode / method / threshold / mask mode / bg σ
    maskChanged = Signal()              # threshold-mask preview
    viewChanged = Signal()              # proj ↔ raw + frame scrub

    def __init__(self, store=None, settings=None, override_store=None, parent=None):
        super().__init__(parent)
        self._editor = None
        self._polys: list = []          # list[list[(y, x)]]
        self._draft: list = []          # open polygon being drawn
        # ── per-file editing ──────────────────────────────────────────────
        self._store = store             # per-file polygon store
        self._ovr = override_store      # per-file roi-settings override store
        self._s = settings
        self._file = ""
        self._image = None              # QImage currently displayed (proj or raw)
        self._proj = None               # cached MAX projection (display in proj view)
        self._mask_proj = None          # projection used for the mask (per mask mode)
        self._mask_proj_mode = ""
        self._raw_frame = None          # cached raw frame (display in raw view)
        self._img_w = 0
        self._img_h = 0
        self._img_token = 0
        self._editing = False
        # ── view (proj ↔ raw) + scrub ─────────────────────────────────────
        self._view_mode = "proj"        # "proj" | "raw"
        self._n_frames = 0
        self._frame_idx = 0
        # ── per-file ROI settings (transient until commit) ────────────────
        self._roi_mode = "Auto threshold"
        self._auto_method = "Li"
        self._threshold = 0.08
        self._mask_mode = "Max"
        self._bg_sigma = 25.0
        self._mask = None               # green RGBA overlay QImage
        self._mask_token = 0
        self._mask_fraction = 0.0
        from firefly.ui.controllers.params.preview_loader import PREVIEW_CMAPS
        self._cmap = "Grayscale"
        if settings is not None:
            c = settings.get_str("ui/preview_cmap", "Grayscale")
            if c in PREVIEW_CMAPS:
                self._cmap = c
        # load the global defaults so an un-overridden file opens showing them
        self._apply_spec(self._default_spec())

    # ── default / effective ROI spec ──────────────────────────────────────
    def _default_spec(self):
        g = self._s
        if g is None:
            return {"roi_mode": self._roi_mode, "roi_auto_method": self._auto_method,
                    "roi_threshold": self._threshold, "roi_mask_mode": self._mask_mode,
                    "roi_bg_sigma": self._bg_sigma}
        return {
            "roi_mode":        g.get_str("analysis/roi_mode", "Auto threshold"),
            "roi_auto_method": g.get_str("analysis/roi_auto_method", "Li"),
            "roi_threshold":   g.get_float("analysis/roi_threshold", 0.08),
            "roi_mask_mode":   g.get_str("analysis/roi_mask_mode", "Max"),
            "roi_bg_sigma":    g.get_float("analysis/roi_bg_sigma", 25.0),
        }

    def _apply_spec(self, spec):
        self._roi_mode    = spec.get("roi_mode", self._roi_mode)
        self._auto_method = spec.get("roi_auto_method", self._auto_method)
        self._threshold   = float(spec.get("roi_threshold", self._threshold))
        self._mask_mode   = spec.get("roi_mask_mode", self._mask_mode)
        self._bg_sigma    = float(spec.get("roi_bg_sigma", self._bg_sigma))

    def _current_spec(self):
        return {"roi_mode": self._roi_mode, "roi_auto_method": self._auto_method,
                "roi_threshold": self._threshold, "roi_mask_mode": self._mask_mode,
                "roi_bg_sigma": self._bg_sigma}

    @staticmethod
    def _spec_differs(a, b):
        if a["roi_mode"] != b["roi_mode"]:               return True
        if a["roi_auto_method"] != b["roi_auto_method"]: return True
        if a["roi_mask_mode"] != b["roi_mask_mode"]:     return True
        if abs(float(a["roi_threshold"]) - float(b["roi_threshold"])) > 1e-6: return True
        if abs(float(a["roi_bg_sigma"]) - float(b["roi_bg_sigma"])) > 1e-6:   return True
        return False

    # ── editor lifecycle (legacy widget island) ──────────────────────────
    def ensureEditor(self):
        if self._editor is not None:
            return self._editor
        from firefly.ui.roi_editor import RoiEditor
        ed = RoiEditor()
        ed.polygonsChanged.connect(self._on_editor_changed)
        ed.frameChanged.connect(self.frameChanged)
        if self._polys:
            ed.set_polygons(self._polys)
        self._editor = ed
        return ed

    def editorWidget(self):
        return self.ensureEditor()

    def _on_editor_changed(self):
        if self._editor is not None:
            self._polys = [[(float(y), float(x)) for y, x in poly]
                           for poly in self._editor.polygons()]
            self.polygonsChanged.emit()

    def _push_to_editor(self):
        if self._editor is not None:
            self._editor.set_polygons(self._polys)

    # ── headless polygon model ───────────────────────────────────────────
    @Slot(float, float)
    def addVertex(self, y: float, x: float):
        self._draft.append((float(y), float(x)))
        self.draftChanged.emit()

    @Slot(result=bool)
    def closeDraft(self) -> bool:
        """Commit the open draft as a polygon (needs ≥3 vertices)."""
        if len(self._draft) < 3:
            return False
        self._polys.append(list(self._draft))
        self._draft = []
        self._push_to_editor()
        self.draftChanged.emit()
        self.polygonsChanged.emit()
        return True

    @Slot()
    def cancelDraft(self):
        if self._draft:
            self._draft = []
            self.draftChanged.emit()

    @Slot(int)
    def deletePolygon(self, idx: int):
        if 0 <= idx < len(self._polys):
            del self._polys[idx]
            self._push_to_editor()
            self.polygonsChanged.emit()

    @Slot(int, int)
    def deleteVertex(self, poly_idx: int, vert_idx: int):
        if 0 <= poly_idx < len(self._polys):
            poly = self._polys[poly_idx]
            if 0 <= vert_idx < len(poly):
                del poly[vert_idx]
                if len(poly) < 3:
                    del self._polys[poly_idx]
                self._push_to_editor()
                self.polygonsChanged.emit()

    @Slot(int, int, float, float)
    def moveVertex(self, poly_idx: int, vert_idx: int, y: float, x: float):
        if 0 <= poly_idx < len(self._polys):
            poly = self._polys[poly_idx]
            if 0 <= vert_idx < len(poly):
                poly[vert_idx] = (float(y), float(x))
                self._push_to_editor()
                self.polygonsChanged.emit()

    @Slot()
    def clearPolygons(self):
        self._polys = []
        self._draft = []
        if self._editor is not None:
            self._editor.clear_polygons()
        self.polygonsChanged.emit()
        self.draftChanged.emit()

    @Slot("QVariantList")
    def setPolygons(self, polys):
        self._polys = [[(float(p[0]), float(p[1])) for p in poly] for poly in polys]
        self._push_to_editor()
        self.polygonsChanged.emit()

    @Slot(result="QVariantList")
    def getPolygons(self):
        return [[list(v) for v in poly] for poly in self._polys]

    @Property("QVariantList", notify=polygonsChanged)
    def polygons(self):
        return [[list(v) for v in poly] for poly in self._polys]

    @Property(int, notify=polygonsChanged)
    def polygonCount(self):
        return len(self._polys)

    @Property(int, notify=draftChanged)
    def draftLength(self):
        return len(self._draft)

    @Property("QVariantList", notify=draftChanged)
    def draftPoints(self):
        return [[y, x] for y, x in self._draft]

    @Property(bool, notify=draftChanged)
    def canClose(self):
        return len(self._draft) >= 3

    # ── image (display) ───────────────────────────────────────────────────
    def roi_image(self):
        """The currently displayed QImage (proj or raw frame) — read by the provider."""
        return self._image

    @Property(bool, notify=editingChanged)
    def editing(self):
        return self._editing

    @Property(int, notify=imageChanged)
    def imageToken(self):
        return self._img_token

    @Property(int, notify=imageChanged)
    def imageWidth(self):
        return self._img_w

    @Property(int, notify=imageChanged)
    def imageHeight(self):
        return self._img_h

    @Property(bool, notify=imageChanged)
    def hasImage(self):
        return self._image is not None and not self._image.isNull()

    @Property(str, notify=editingChanged)
    def fileName(self):
        import os
        return os.path.basename(self._file) if self._file else ""

    @Slot(str)
    def editFile(self, path):
        """Open the viewer over ``path``'s projection, loading the file's ROI
        override (or the global default) + any stored polygon."""
        self._file = path or ""
        # effective spec = per-file override, else the global sidebar default
        spec = (self._ovr.get(self._file) if self._ovr else None) or self._default_spec()
        self._apply_spec(spec)
        self._view_mode = "proj"
        self._frame_idx = 0
        self._raw_frame = None
        self._mask_proj = None
        self._mask_proj_mode = ""
        self._load_background(self._file)     # sets self._proj + n_frames, renders proj
        self._recompute_mask()
        self._draft = []
        existing = self._store.get(self._file) if self._store else None
        self._polys = [[(float(y), float(x)) for y, x in poly]
                       for poly in (existing or [])]
        self._editing = True
        self.roiSettingsChanged.emit()
        self.viewChanged.emit()
        self.polygonsChanged.emit()
        self.draftChanged.emit()
        self.editingChanged.emit()

    def _load_background(self, path):
        # A sampled max-intensity projection (never a full stack load — that froze
        # the GUI on multi-GB recordings). (Y, X) layout matches load_file's.
        from firefly.ui.controllers.params.preview_loader import (
            sampled_projection, quick_frame_count)
        self._proj = None
        self._n_frames = 0
        try:
            import os
            if not (path and os.path.isfile(path)):
                self._render_display()
                return
            self.statusMessage.emit(f"Loading {os.path.basename(path)}…")
            self._proj = sampled_projection(path, "max")
            self._n_frames = quick_frame_count(path)
            if self._proj is None:
                self.statusMessage.emit("Couldn't load image.")
        except Exception as exc:
            self.statusMessage.emit(f"Couldn't load image: {exc}")
        self._render_display()

    def _render_display(self):
        """(Re)render the active source (max projection or the current raw frame)
        with the current colormap into the display image + notify."""
        from firefly.ui.controllers.params.preview_loader import render_projection
        src = self._raw_frame if self._view_mode == "raw" else self._proj
        self._image = None
        self._img_w = self._img_h = 0
        if src is not None:
            try:
                img = render_projection(src, self._cmap)
                if img is not None and not img.isNull():
                    self._image = img
                    self._img_w, self._img_h = img.width(), img.height()
            except Exception:
                self._image = None
        self._img_token += 1
        self.imageChanged.emit()

    # ── view: max projection ↔ raw frame + scrub ──────────────────────────
    @Property("QStringList", constant=True)
    def viewModes(self):
        return ["Max projection", "Raw frames"]

    @Property(str, notify=viewChanged)
    def viewMode(self):
        return self._view_mode

    @Slot(str)
    def setViewMode(self, mode):
        m = "raw" if str(mode).lower().startswith("raw") else "proj"
        if m == self._view_mode:
            return
        self._view_mode = m
        if m == "raw" and self._raw_frame is None:
            self._load_frame(self._frame_idx)
        self._render_display()
        self._recompute_mask()        # mask follows the displayed source (raw ↔ proj)
        self.viewChanged.emit()

    @Property(int, notify=viewChanged)
    def nFrames(self):
        return self._n_frames

    @Property(int, notify=viewChanged)
    def frameIndex(self):
        return self._frame_idx

    @Property(str, notify=viewChanged)
    def frameLabel(self):
        if self._view_mode == "raw" and self._n_frames > 0:
            return f"frame {self._frame_idx + 1} / {self._n_frames}"
        return "max projection"

    @Slot(int)
    def setFrame(self, i):
        i = int(i)
        if self._n_frames > 0:
            i = max(0, min(i, self._n_frames - 1))
        if i == self._frame_idx and self._raw_frame is not None:
            return
        self._frame_idx = i
        if self._view_mode == "raw":
            self._load_frame(i)
            self._render_display()
        self.frameChanged.emit(i)
        self.viewChanged.emit()

    def _load_frame(self, i):
        from firefly.ui.controllers.params.preview_loader import sampled_frame
        try:
            self._raw_frame = sampled_frame(self._file, i)
        except Exception:
            self._raw_frame = None

    # ── colormap ──────────────────────────────────────────────────────────
    @Property("QStringList", constant=True)
    def cmaps(self):
        from firefly.ui.controllers.params.preview_loader import PREVIEW_CMAPS
        return list(PREVIEW_CMAPS)

    @Property(str, notify=cmapChanged)
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, v):
        from firefly.ui.controllers.params.preview_loader import PREVIEW_CMAPS
        v = str(v)
        if v != self._cmap and v in PREVIEW_CMAPS:
            self._cmap = v
            if self._s is not None:
                self._s.set("ui/preview_cmap", v)
            self.cmapChanged.emit()
            self._render_display()

    # ── per-file ROI settings (transient; mirror the sidebar ROI menu) ────
    @Property("QStringList", constant=True)
    def roiModes(self):
        return ["None", "Auto threshold", "Manual threshold", "Manual polygon",
                "Sister TIFF", "ImageJ ROI"]

    @Property("QStringList", constant=True)
    def autoMethods(self):
        return ["Li", "Otsu", "Triangle", "Mean"]

    @Property("QStringList", constant=True)
    def maskModes(self):
        return ["Max", "Blink density", "Mean", "Sum"]

    @Property(str, notify=roiSettingsChanged)
    def roiMode(self):
        return self._roi_mode

    @roiMode.setter
    def roiMode(self, v):
        v = str(v)
        if v == self._roi_mode:
            return
        self._roi_mode = v
        self.roiSettingsChanged.emit()
        self._recompute_mask()

    @Property(str, notify=roiSettingsChanged)
    def autoMethod(self):
        return self._auto_method

    @autoMethod.setter
    def autoMethod(self, v):
        v = str(v)
        if v == self._auto_method:
            return
        self._auto_method = v
        self.roiSettingsChanged.emit()
        self._recompute_mask()

    @Property(str, notify=roiSettingsChanged)
    def maskMode(self):
        return self._mask_mode

    @maskMode.setter
    def maskMode(self, v):
        v = str(v)
        if v == self._mask_mode:
            return
        self._mask_mode = v
        self._mask_proj = None          # force the mask projection to rebuild
        self._mask_proj_mode = ""
        self.roiSettingsChanged.emit()
        self._recompute_mask()

    @Property(float, notify=roiSettingsChanged)
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, v):
        v = float(v)
        if abs(v - self._threshold) < 1e-9:
            return
        self._threshold = v
        self.roiSettingsChanged.emit()
        # mask NOT rebuilt here — QML debounces refreshMask (DoG+morphology is heavy)

    @Property(float, notify=roiSettingsChanged)
    def bgSigma(self):
        return self._bg_sigma

    @bgSigma.setter
    def bgSigma(self, v):
        v = float(v)
        if abs(v - self._bg_sigma) < 1e-9:
            return
        self._bg_sigma = v
        self.roiSettingsChanged.emit()
        # debounced (refreshMask) like threshold

    @Slot()
    def refreshMask(self):
        """Rebuild the threshold-mask preview (debounced from the sliders)."""
        self._recompute_mask()

    # ── threshold-mask preview ────────────────────────────────────────────
    @Property(int, notify=maskChanged)
    def maskToken(self):
        return self._mask_token

    @Property(bool, notify=maskChanged)
    def hasMask(self):
        return self._mask is not None and not self._mask.isNull()

    @Property(float, notify=maskChanged)
    def maskFraction(self):
        return self._mask_fraction

    def roi_mask_image(self):
        return self._mask

    def _mask_projection(self):
        """The projection the mask is thresholded on, per the mask mode (Max ≈
        the display projection; Mean/Sum re-reduced; Blink falls back to Max)."""
        hint = _MASK_MODE_HINT.get(self._mask_mode, "max")
        if hint in ("max", "blink"):
            return self._proj
        if self._mask_proj is not None and self._mask_proj_mode == hint:
            return self._mask_proj
        from firefly.ui.controllers.params.preview_loader import sampled_projection
        self._mask_proj = sampled_projection(self._file, hint)
        self._mask_proj_mode = hint
        return self._mask_proj if self._mask_proj is not None else self._proj

    def _mask_source(self):
        """The image the threshold mask is computed on: the displayed RAW frame
        when scrubbing (so the mask updates per frame — what the threshold catches
        in this frame), else the mask-mode projection (the ROI the run builds)."""
        if self._view_mode == "raw" and self._raw_frame is not None:
            return self._raw_frame
        return self._mask_projection()

    def _recompute_mask(self):
        """(Re)build the green threshold-mask overlay for the auto/manual modes
        by calling the analysis core's mask builder on the mask projection."""
        self._mask = None
        self._mask_fraction = 0.0
        from firefly.ui.controllers.params.params_builder import ROI_MODE_MAP
        mode = ROI_MODE_MAP.get(self._roi_mode, "none")
        proj = self._mask_source()
        if proj is not None and mode in ("auto", "manual"):
            try:
                from firefly.analysis.fa_roi import build_roi_mask_advanced
                thr = None if mode == "auto" else float(self._threshold)
                method = (self._auto_method or "Li").lower()
                hint = _MASK_MODE_HINT.get(self._mask_mode, "max")
                mask, info = build_roi_mask_advanced(
                    proj, threshold=thr, threshold_method=method,
                    bg_sigma=float(self._bg_sigma), mode_hint=hint)
                if mask is not None:
                    self._mask = _green_mask_qimage(mask)
                    self._mask_fraction = float(info.get("fraction", 0.0))
            except Exception as exc:
                self._mask = None
                self.statusMessage.emit(f"ROI mask preview failed: {exc}")
        self._mask_token += 1
        self.maskChanged.emit()

    # ── per-file override indicator ───────────────────────────────────────
    @Slot(str, result=bool)
    def fileHasOverride(self, path):
        return bool(self._ovr and self._ovr.has(path))

    @Slot(str, result=bool)
    def fileHasRoi(self, path):
        return bool((self._store and self._store.has(path))
                    or (self._ovr and self._ovr.has(path)))

    # ── commit / cancel ───────────────────────────────────────────────────
    @Slot()
    def commit(self):
        """Save the per-file ROI: the polygon (RoiStore) + the settings override
        (RoiOverrideStore) when they differ from the sidebar default; if they
        match the default and there's no polygon, the override is cleared."""
        from firefly.ui.controllers.params.params_builder import ROI_MODE_MAP
        if self._store is not None:
            self._store.set(self._file, self._polys)
        if self._ovr is not None:
            spec = self._current_spec()
            is_poly = ROI_MODE_MAP.get(self._roi_mode) == "polygon"
            custom = (self._spec_differs(spec, self._default_spec())
                      or (is_poly and bool(self._polys)))
            if custom:
                self._ovr.set(self._file, spec)
            else:
                self._ovr.clear(self._file)
        self._editing = False
        self.editingChanged.emit()
        self.statusMessage.emit(
            f"ROI saved for {self.fileName}" if self._file else "ROI saved")

    @Slot()
    def cancel(self):
        """Discard edits and close — revert transient settings + polygon."""
        existing = self._store.get(self._file) if self._store else None
        self._polys = [[(float(y), float(x)) for y, x in poly]
                       for poly in (existing or [])]
        self._draft = []
        spec = (self._ovr.get(self._file) if self._ovr else None) or self._default_spec()
        self._apply_spec(spec)
        self._editing = False
        self.roiSettingsChanged.emit()
        self.polygonsChanged.emit()
        self.draftChanged.emit()
        self.editingChanged.emit()
