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
    detectChanged = Signal()            # detection on/off + minmass
    spotsChanged = Signal()             # detected-spot overlay

    def __init__(self, store=None, settings=None, override_store=None, parent=None):
        super().__init__(parent)
        self._editor = None
        self._polys: list = []          # list[list[(y, x)]]
        self._draft: list = []          # open polygon being drawn
        # ── per-file editing ──────────────────────────────────────────────
        self._store = store             # per-file polygon store
        self._ovr = override_store      # per-file roi-settings override store
        self._s = settings
        self._batch_mode = False        # single: viewer edits mirror into the
                                        # sidebar default; batch: per-file override
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
        # ── sister-TIFF ROI preview (mode "Sister TIFF") ──────────────────
        self._sister_path = ""          # detected companion ROI image
        self._sister_status = ""        # provenance ("name · Li threshold …") or reason
        # ── detection-threshold (minmass) preview ─────────────────────────
        self._detect_on = False
        self._minmass = (float(settings.get_float("analysis/minmass", 1.0))
                         if settings else 1.0)
        self._spots = None              # green detected-spot overlay QImage
        self._spots_token = 0
        self._spot_count = 0
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
        if self._s is not None:               # pick up the latest sidebar minmass
            self._minmass = float(self._s.get_float("analysis/minmass", self._minmass))
        self._spots = None
        self._spots_token += 1
        if self._detect_on:
            self._recompute_spots()
        self.detectChanged.emit()
        self.spotsChanged.emit()
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
        if self._detect_on:
            self._recompute_spots()   # detections follow the displayed source too
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
            # spots follow the frame too, but DEBOUNCED from QML (refreshSpots) —
            # locate() per scrub tick would lag, like the mask.
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

    @Slot(bool)
    def setBatchMode(self, on):
        """Single vs batch.  Single mode mirrors viewer ROI edits into the
        global sidebar default (the run uses that default for the one file);
        batch mode keeps them as a per-file override that must not move the
        shared default.  Driven by ImportController.batchMode."""
        self._batch_mode = bool(on)

    def _push_default(self, key, value):
        """Mirror a viewer ROI edit into the global sidebar setting — single
        mode only.  In batch the viewer is a per-file override (RoiOverrideStore)
        and must NOT touch the shared default."""
        if self._batch_mode or self._s is None:
            return
        self._s.set(key, value)

    @Property(str, notify=roiSettingsChanged)
    def roiMode(self):
        return self._roi_mode

    @roiMode.setter
    def roiMode(self, v):
        v = str(v)
        if v == self._roi_mode:
            return
        self._roi_mode = v
        self._push_default("analysis/roi_mode", v)
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
        self._push_default("analysis/roi_auto_method", v)
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
        self._push_default("analysis/roi_mask_mode", v)
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
        self._push_default("analysis/roi_threshold", v)
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
        self._push_default("analysis/roi_bg_sigma", v)
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

    # ── sister-TIFF ROI status (drives the viewer caption in Sister mode) ──
    @Property(bool, notify=maskChanged)
    def sisterFound(self):
        return bool(self._sister_path)

    @Property(str, notify=maskChanged)
    def sisterName(self):
        import os
        return os.path.basename(self._sister_path) if self._sister_path else ""

    @Property(str, notify=maskChanged)
    def sisterStatus(self):
        return self._sister_status

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
        """(Re)build the green ROI overlay.  Threshold modes call the analysis
        core's mask builder on the mask projection; Sister-TIFF mode loads +
        thresholds the companion image via the SAME fa_roi helper the run uses,
        so the preview is exactly what the analysis includes."""
        self._mask = None
        self._mask_fraction = 0.0
        self._sister_path = ""
        self._sister_status = ""
        from firefly.ui.controllers.params.params_builder import ROI_MODE_MAP
        mode = ROI_MODE_MAP.get(self._roi_mode, "none")
        if mode == "sister":
            self._recompute_sister_mask()
            self._mask_token += 1
            self.maskChanged.emit()
            return
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

    def _recompute_sister_mask(self):
        """Preview the Sister-TIFF ROI: locate the companion image and build its
        mask with the SAME fa_roi helpers the analysis run uses, so the green
        overlay is exactly the region FIREFLY will keep.  Sets self._mask +
        self._sister_path/_sister_status (the latter drives the viewer caption)."""
        if not self._file:
            return
        suffix = (self._s.get_str("analysis/roi_sister_suffix", "_green")
                  if self._s else "_green")
        from firefly.analysis.fa_roi import (find_sister_roi_path,
                                             build_sister_roi_mask)
        path = find_sister_roi_path(self._file, suffix)
        if not path:
            self._sister_status = (f"No sister image (…{suffix}.tif) "
                                   f"beside this file")
            return
        self._sister_path = path
        target = self._proj.shape if self._proj is not None else None
        try:
            mask, note = build_sister_roi_mask(path, target_shape=target)
        except Exception as exc:
            self._sister_status = f"Sister ROI failed: {exc}"
            return
        if mask is not None:
            self._mask = _green_mask_qimage(mask)
            self._mask_fraction = float(mask.mean())
        self._sister_status = note      # provenance on success, reason on skip

    # ── detection-threshold (minmass) preview ─────────────────────────────
    # Runs the SAME detect path the analysis uses (preprocess + trackpy.locate)
    # on the displayed frame so the user can see which spots a given minmass
    # catches.  The slider writes through to the sidebar analysis/minmass (and
    # turns auto-minmass off) so what you preview is what the run detects.
    @Property(bool, notify=detectChanged)
    def detectEnabled(self):
        return self._detect_on

    @detectEnabled.setter
    def detectEnabled(self, on):
        on = bool(on)
        if on == self._detect_on:
            return
        self._detect_on = on
        self.detectChanged.emit()
        self._recompute_spots()

    @Property(float, notify=detectChanged)
    def detectMinmass(self):
        return self._minmass

    @detectMinmass.setter
    def detectMinmass(self, v):
        v = max(0.0, float(v))
        if abs(v - self._minmass) < 1e-9:
            return
        self._minmass = v
        if self._s is not None:                  # what you preview is what runs
            self._s.set("analysis/minmass", v)
            self._s.set("analysis/auto_minmass", False)
        self.detectChanged.emit()
        # spots NOT rebuilt here — QML debounces refreshSpots (locate is heavy)

    @Property(int, notify=spotsChanged)
    def spotToken(self):
        return self._spots_token

    @Property(bool, notify=spotsChanged)
    def hasSpots(self):
        return self._spots is not None and not self._spots.isNull()

    @Property(int, notify=spotsChanged)
    def spotCount(self):
        return self._spot_count

    def roi_spots_image(self):
        return self._spots

    @Slot()
    def refreshSpots(self):
        self._recompute_spots()

    def _spots_qimage(self, h, w, xs, ys):
        from PySide6.QtCore import QPointF, Qt as _Qt
        from PySide6.QtGui import QImage, QPainter, QPen, QColor
        img = QImage(int(w), int(h), QImage.Format.Format_ARGB32)
        img.fill(0)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(57, 255, 110)); pen.setWidthF(1.3)   # match live-detection green
        p.setPen(pen); p.setBrush(_Qt.NoBrush)
        for x, y in zip(xs, ys):
            p.drawEllipse(QPointF(float(x), float(y)), 4.0, 4.0)
        p.end()
        return img

    def _recompute_spots(self):
        """Detect spots on the displayed frame at the current minmass + render
        them as a green-circle overlay (run-matching preprocess + trackpy)."""
        self._spots = None
        self._spot_count = 0
        if self._detect_on:
            src = self._mask_source()
            if src is not None:
                try:
                    import numpy as np
                    import trackpy as tp
                    from firefly.analysis.fa_preprocess import preprocess_stack
                    from firefly.ui.controllers.params.params_builder import BG_METHOD_MAP
                    g = self._s
                    diameter = int(g.get_float("analysis/diameter", 7)) if g else 7
                    if diameter % 2 == 0:
                        diameter += 1
                    bg_radius = int(g.get_float("analysis/bg_radius", 10)) if g else 10
                    bg_method = (BG_METHOD_MAP.get(g.get_str("analysis/bg_method",
                                 "Uniform Filter"), "uniform_filter") if g else "uniform_filter")
                    frame = np.asarray(src, dtype=np.float32)
                    pp = preprocess_stack(frame[None], bg_radius=bg_radius,
                                          bg_method=bg_method, workers=1, quiet=True)[0]
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        f = tp.locate(pp, diameter, minmass=float(self._minmass),
                                      percentile=64)
                    if f is not None and len(f):
                        self._spot_count = int(len(f))
                        self._spots = self._spots_qimage(
                            frame.shape[0], frame.shape[1],
                            f["x"].to_numpy(), f["y"].to_numpy())
                except Exception as exc:
                    self._spots = None
                    self.statusMessage.emit(f"Detection preview failed: {exc}")
        self._spots_token += 1
        self.spotsChanged.emit()

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
