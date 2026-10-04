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

import threading
from collections import OrderedDict

from PySide6.QtCore import Property, QObject, Signal, Slot

# Detections cached per (file, frame, detector settings, threshold): moving
# the threshold back to a value already seen shows its spots at once.
_DETECTION_CACHE_SIZE = 64

# The overlay also marks, in red, what the threshold excludes: spots the same
# detector finds at this fraction of the threshold but not at the threshold —
# the near misses.  Measured on MB112C frames at the recommended threshold, ¾
# gives about 1–5 red per detected spot; ½ gave 3–40, nearly all noise, which
# buried the frame.
_EXCLUDED_FRACTION = 0.75



class _PreviewImport:
    """The ImportController attributes ``build_params`` reads, for the file the
    viewer has open — enough to build a run's own detection parameters."""
    def __init__(self, path, settings):
        from firefly.analysis.fa_constants import (DEFAULT_FRAME_INTERVAL_S,
                                                   DEFAULT_PIXEL_SIZE_UM)
        self.filePath, self.outDir, self.isCsv = path, None, False
        self.overridePx = self.overrideFi = False
        self.pixelSize = float(settings.get_float("analysis/pixel_size", DEFAULT_PIXEL_SIZE_UM))
        self.frameInterval = float(settings.get_float("analysis/frame_interval",
                                                      DEFAULT_FRAME_INTERVAL_S))


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
    splitChanged = Signal()             # "analyse each ROI separately" flag + labels
    maskChanged = Signal()              # threshold-mask preview
    viewChanged = Signal()              # proj ↔ raw + frame scrub
    detectChanged = Signal()            # detection on/off + minmass
    previewInvalidated = Signal()
    spotsChanged = Signal()             # detected-spot overlay
    _spotsDetected = Signal(object, object)   # (key, rows | error) worker → GUI
    _recommendReady = Signal(object, object)  # (request, (value, note) | error) worker → GUI
    recommendChanged = Signal()         # "Set to recommended" busy / note
    brushChanged = Signal()             # brush tool / radius / painted preview
    panelChanged = Signal()             # which screen of the viewer is showing

    def __init__(self, store=None, settings=None, override_store=None, parent=None):
        super().__init__(parent)
        self._polys: list = []          # list[list[(y, x)]]
        self._draft: list = []          # open polygon being drawn
        # ── which screen of the viewer is showing ──────────────────────────
        # Drawing a region and picking a detection threshold are separate jobs
        # with separate entry points (editFile / editDetection).  One modal
        # holding both meant every visit scrolled past the other job's controls.
        self._panel = "roi"             # "roi" | "detect"
        # ── brush editing ─────────────────────────────────────────────────
        # A painted region is still stored as POLYGONS: the mask below is only
        # the editing buffer, converted back on every stroke end (exact — see
        # fa_roi.mask_to_polygons).  Nothing downstream learns a new ROI type.
        self._tool = "polygon"          # "polygon" | "brush" | "eraser"
        self._brush_radius = 6.0        # image pixels
        self._brush_mask = None         # np.bool_ (H, W) while a brush session is live
        self._brush_img = None          # QImage preview of _brush_mask
        self._brush_token = 0
        self._brush_undo: list = []     # mask snapshots, newest last
        self._brush_holes_warned = False
        # ── per-file editing ──────────────────────────────────────────────
        self._store = store             # per-file polygon store
        self._ovr = override_store      # per-file roi-settings override store
        self._s = settings
        self._run_scoped = False        # True while editing a FINISHED run's ROI:
                                        # suppresses every write-through into the
                                        # sidebar defaults (see _push_default)
        self._run_dir = ""              # the run being edited, when run-scoped
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
        self._view_mode = "proj"        # "proj" | "raw" | "green"
        self._n_frames = 0
        self._frame_idx = 0
        # ── companion "green" image as a third view (see what the ROI covers) ─
        self._green_path = ""           # sister image beside this file, if any
        self._green_img = None          # lazily loaded, on the stack's pixel grid
        self._green_note = ""           # provenance / why it couldn't be shown
        # ── per-file ROI settings (transient until commit) ────────────────
        self._roi_mode = "Auto threshold"
        self._auto_method = "Li"
        self._threshold = 0.08
        self._mask_mode = "Max"
        self._bg_sigma = 25.0
        # ── multiple-ROI → individual replicates (per file) ───────────────
        self._split_replicates = False  # analyse each drawn ROI as its own output
        self._roi_labels: list = []     # optional per-ROI names (parallel to _polys)
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
        self._spot_rows = None
        self._spot_summary = "Preview off"
        self._spot_inspection = ""
        self._spots_stale = True
        # Threshold dragging: detections run off the GUI thread, newest request
        # wins, and each one is cached (see previewThreshold).
        self._det_cache = OrderedDict()
        self._det_wanted = None         # the detection the overlay should show
        self._det_pending = False
        self._det_req = None
        self._det_lock = threading.Lock()
        self._det_thread = None
        self._det_low_wanted = None     # its partner at _EXCLUDED_FRACTION (red spots)
        self._shown = None              # (kept, low) detections behind the overlay
        # "Set to recommended": worked out off the GUI thread for the open file.
        self._rec_busy = False
        self._rec_note = ""
        self._rec_thread = None
        # Candidate masses from a minmass=0 detection on the displayed frame,
        # cached against the settings that change them.  Re-thresholding this
        # array is instant, so the histogram and the consequence readouts can
        # follow the slider without re-running the detector.
        self._cand_masses = None
        self._cand_key = None
        self._cand_n_frames = 0
        self._cand_per_frame = None
        self._cand_floor = None         # (floor, status) — see _candidate_masses
        self._minmass_per_file = False   # write the threshold to THIS file only
        # The palmTRACER-style detector's threshold — same per-file mechanism.
        self._wavelet = (float(settings.get_float("analysis/wavelet_threshold", 250.0))
                         if settings is not None else 250.0)
        self._wavelet_per_file = False
        from firefly.ui.controllers.params.preview_loader import PREVIEW_CMAPS
        self._cmap = "Grayscale"
        if settings is not None:
            c = settings.get_str("ui/preview_cmap", "Grayscale")
            if c in PREVIEW_CMAPS:
                self._cmap = c
        # load the global defaults so an un-overridden file opens showing them
        self._apply_spec(self._default_spec())
        if settings is not None and hasattr(settings, "changed"):
            settings.changed.connect(self._spot_settings_changed)
        self._spotsDetected.connect(self._on_spots_detected)
        self._recommendReady.connect(self._on_recommended)
        # An ROI edit does not change what was DETECTED, only which candidates
        # fall inside it — so re-label the cached spots instead of throwing them
        # away.  Discarding them made the overlay empty on every brush stroke
        # and every vertex move, which is precisely when you are looking at it.
        self.polygonsChanged.connect(self._reclassify_spots)
        self.roiSettingsChanged.connect(self._reclassify_spots)


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
        # Split-replicates + labels are per-FILE only (never in the global default).
        self._split_replicates = bool(spec.get("roi_split_replicates", False))
        self._roi_labels = list(spec.get("roi_labels") or [])
        if spec.get("minmass") is not None:
            self._minmass_per_file = True
            self._minmass = float(spec["minmass"])
        else:
            self._minmass_per_file = False
        if spec.get("wavelet_threshold") is not None:
            self._wavelet_per_file = True
            self._wavelet = float(spec["wavelet_threshold"])
        else:
            self._wavelet_per_file = False

    def _current_spec(self):
        spec = {"roi_mode": self._roi_mode, "roi_auto_method": self._auto_method,
                "roi_threshold": self._threshold, "roi_mask_mode": self._mask_mode,
                "roi_bg_sigma": self._bg_sigma,
                "roi_split_replicates": self._split_replicates,
                "roi_labels": list(self._roi_labels)}
        # Only present when the user asked for it: an absent key means this file
        # inherits the sidebar threshold, which is what most files should do.
        if self._minmass_per_file:
            spec["minmass"] = float(self._minmass)
            spec["auto_minmass"] = False
        if self._wavelet_per_file:
            spec["wavelet_threshold"] = float(self._wavelet)
        return spec

    @staticmethod
    def _spec_differs(a, b):
        if a["roi_mode"] != b["roi_mode"]:               return True
        if a["roi_auto_method"] != b["roi_auto_method"]: return True
        if a["roi_mask_mode"] != b["roi_mask_mode"]:     return True
        if abs(float(a["roi_threshold"]) - float(b["roi_threshold"])) > 1e-6: return True
        if abs(float(a["roi_bg_sigma"]) - float(b["roi_bg_sigma"])) > 1e-6:   return True
        return False

    # ── headless polygon model ───────────────────────────────────────────
    @Slot(float, float)
    def addVertex(self, y: float, x: float):
        self._draft.append((float(y), float(x)))
        self.draftChanged.emit()

    @Slot(result=bool)
    def closeDraft(self) -> bool:
        """Commit the open draft as a polygon (needs ≥3 vertices).

        Vertices may be drawn OFF the image (to comfortably enclose samples that
        sit against an edge); on close the shape is clipped to the image
        rectangle, so any part drawn past an edge is run along that edge.
        """
        if len(self._draft) < 3:
            return False
        poly = self._clip_poly_to_image(self._draft)
        if len(poly) < 3:                     # drawn entirely outside the image
            return False
        self._polys.append([(float(y), float(x)) for y, x in poly])
        self._draft = []
        self.draftChanged.emit()
        self.polygonsChanged.emit()
        return True

    def _clip_poly_to_image(self, poly):
        """Clip a polygon to the image rectangle (x∈[0,W], y∈[0,H]) with the
        Sutherland–Hodgman algorithm, so an ROI drawn past an edge follows that
        edge instead of extending into empty space.  Points are ``(y, x)``.  A
        polygon already inside the image is returned unchanged; if no image is
        loaded (dimensions unknown) the polygon passes through untouched."""
        W = float(self._img_w); H = float(self._img_h)
        if W <= 0 or H <= 0 or len(poly) < 3:
            return [(float(y), float(x)) for y, x in poly]

        def _lerp(a, b, t):
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

        def _clip(pts, inside, isect):
            out = []
            for i in range(len(pts)):
                a, b = pts[i - 1], pts[i]      # edge a→b (i-1 wraps to last)
                a_in, b_in = inside(a), inside(b)
                if b_in:
                    if not a_in:
                        out.append(isect(a, b))
                    out.append(b)
                elif a_in:
                    out.append(isect(a, b))
                # isect is only called across the boundary, so the crossed
                # coordinate differs → its denominator below is never zero.
            return out

        pts = [(float(y), float(x)) for y, x in poly]
        pts = _clip(pts, lambda p: p[1] >= 0.0,                     # x ≥ 0
                    lambda a, b: _lerp(a, b, (0.0 - a[1]) / (b[1] - a[1])))
        if len(pts) >= 3:
            pts = _clip(pts, lambda p: p[1] <= W,                  # x ≤ W
                        lambda a, b: _lerp(a, b, (W - a[1]) / (b[1] - a[1])))
        if len(pts) >= 3:
            pts = _clip(pts, lambda p: p[0] >= 0.0,                 # y ≥ 0
                        lambda a, b: _lerp(a, b, (0.0 - a[0]) / (b[0] - a[0])))
        if len(pts) >= 3:
            pts = _clip(pts, lambda p: p[0] <= H,                  # y ≤ H
                        lambda a, b: _lerp(a, b, (H - a[0]) / (b[0] - a[0])))
        return pts

    @Slot()
    def cancelDraft(self):
        if self._draft:
            self._draft = []
            self.draftChanged.emit()

    @Slot(int)
    def deletePolygon(self, idx: int):
        if 0 <= idx < len(self._polys):
            del self._polys[idx]
            if idx < len(self._roi_labels):
                del self._roi_labels[idx]        # keep labels aligned to polygons
            self.polygonsChanged.emit()
            self.splitChanged.emit()

    @Slot(int, int)
    def deleteVertex(self, poly_idx: int, vert_idx: int):
        if 0 <= poly_idx < len(self._polys):
            poly = self._polys[poly_idx]
            if 0 <= vert_idx < len(poly):
                del poly[vert_idx]
                if len(poly) < 3:
                    del self._polys[poly_idx]
                self.polygonsChanged.emit()

    @Slot(int, int, float, float)
    def moveVertex(self, poly_idx: int, vert_idx: int, y: float, x: float):
        if 0 <= poly_idx < len(self._polys):
            poly = self._polys[poly_idx]
            if 0 <= vert_idx < len(poly):
                poly[vert_idx] = (float(y), float(x))
                self.polygonsChanged.emit()

    @Slot()
    def clearPolygons(self):
        self._polys = []
        self._draft = []
        self._roi_labels = []
        self.polygonsChanged.emit()
        self.draftChanged.emit()
        self.splitChanged.emit()

    @Slot("QVariantList")
    def setPolygons(self, polys):
        self._polys = [[(float(p[0]), float(p[1])) for p in poly] for poly in polys]
        self.polygonsChanged.emit()

    # ── per-file detection threshold ─────────────────────────────────────
    @Property(bool, notify=detectChanged)
    def minmassPerFile(self):
        return self._minmass_per_file

    @Slot(bool)
    def setMinmassPerFile(self, on):
        """Route the threshold to THIS file instead of the sidebar default.

        Off (the default) keeps the historic behaviour: the slider writes the
        global ``analysis/minmass`` that every file in a batch uses.  On, the
        value is saved with this file's ROI override on Save and only that file
        is thresholded with it — and the global setting is left exactly as it
        was, so turning this on for one recording cannot quietly move every
        other file in the queue.
        """
        on = bool(on)
        if on == self._minmass_per_file or self._run_scoped:
            return
        self._minmass_per_file = on
        if not on and self._ovr is not None and self._file:
            spec = self._ovr.get(self._file)
            if spec and "minmass" in spec:
                spec = dict(spec)
                spec.pop("minmass", None); spec.pop("auto_minmass", None)
                self._ovr.set(self._file, spec)
        self.detectChanged.emit()

    # ── manual-threshold guidance ────────────────────────────────────────
    _PROFILE_FRAMES = 16        # frames pooled for the mass histogram
    _HARVEST_CAP = 120          # per-frame candidate cap, matching the auto-picker

    def _detect_key(self):
        g = self._s
        if g is None:
            return None
        return (self._file, int(self._frame_idx),
                int(round(g.get_float("analysis/diameter", 7))),
                int(round(g.get_float("analysis/bg_radius", 10))),
                g.get_str("analysis/bg_method", "Uniform Filter"),
                g.get_str("analysis/backend", "Auto"),
                int(round(g.get_float("analysis/channel", 0))),
                float(self._wavelet))

    def _candidate_masses(self):
        """Candidate masses with the threshold at ZERO, pooled over a spread of
        frames, cached.  Returns ``(masses, n_frames)``.

        Pooled deliberately: on ONE frame the noise and signal modes are not
        separable (0.16-0.18 dex on real MB543B data, against the 0.5 the valley
        finder needs), so the noise-floor marker — the single most useful thing
        on the panel — never appears.  Sampling across the recording also makes
        the per-frame numbers an average rather than one arbitrary frame's.
        This mirrors what the auto-picker does with its contiguous windows.
        """
        import numpy as np
        key = self._detect_key()
        if key is not None and key == self._cand_key and self._cand_masses is not None:
            return self._cand_masses, self._cand_n_frames
        if self._s is None or not self._file:
            return None, 0
        try:
            from firefly.analysis.fa_detection_preview import preview_detections
            from firefly.ui.controllers.params.preview_loader import detection_frame
            from firefly.ui.controllers.params.params_builder import (
                BG_METHOD_MAP, BACKEND_LABEL_TO_VALUE)
            g = self._s
            opts = dict(
                diameter=int(round(g.get_float("analysis/diameter", 7))),
                minmass=0.0,
                bg_radius=int(round(g.get_float("analysis/bg_radius", 10))),
                bg_method=BG_METHOD_MAP.get(
                    g.get_str("analysis/bg_method", "Uniform Filter"),
                    "uniform_filter"),
                backend=BACKEND_LABEL_TO_VALUE.get(
                    g.get_str("analysis/backend", "Auto"), "auto"),
                min_cnr=0.0, roi_mask=None, roi_known=False,
                **self._detector_kwargs())
            channel = int(round(g.get_float("analysis/channel", 0)))
            total = int(self._n_frames or 0)
            if total > 1:
                idx = np.unique(np.linspace(0, total - 1,
                                            min(self._PROFILE_FRAMES, total)).astype(int))
            else:
                idx = np.array([int(self._frame_idx)])
            pooled, capped, used = [], [], 0
            for i in idx:
                frame = (self._raw_frame if (int(i) == int(self._frame_idx)
                                             and self._raw_frame is not None)
                         else detection_frame(self._file, int(i), channel))
                if frame is None:
                    continue
                rows, _ = preview_detections(frame, **opts)
                mm = np.asarray(rows["mass"].to_numpy(), dtype=float)
                pooled.append(mm)
                capped.append(mm)      # per-frame; estimate_noise_floor applies the cap
                used += 1
            if not used:
                raise ValueError("no frames could be read for the profile")
            self._cand_masses = np.concatenate(pooled) if pooled else np.array([])
            self._cand_per_frame = capped
            self._cand_n_frames = used
            # The floor is a property of the HARVEST, not of the threshold, so
            # it is fitted here — once — rather than on every massProfile call.
            # It costs three GaussianMixture fits (~15 ms, measured as ~100% of
            # massProfile's cost), and massProfile runs on every slider move and
            # every toggle, which is what made the panel feel sticky.
            from firefly.analysis.fa_localize import estimate_noise_floor
            self._cand_floor = estimate_noise_floor(capped, cap=self._HARVEST_CAP)
            self._cand_key = key
        except Exception as exc:
            self._cand_masses = None
            self._cand_key = None
            self._cand_n_frames = 0
            self._cand_floor = None
            self.statusMessage.emit(f"Threshold guidance unavailable: {exc}")
            return None, 0
        return self._cand_masses, self._cand_n_frames

    @Slot(result="QVariantMap")
    def massProfile(self):
        """Histogram + markers + the cost of the current threshold, for the
        guidance panel.  Safe to call on every slider move."""
        from firefly.analysis.fa_localize import threshold_guidance
        masses, n_frames = self._candidate_masses()
        if masses is None or not len(masses):
            return {"n_candidates": 0, "edges": [], "counts": [],
                    "warning": "Open a recording to profile its detection threshold."}
        floor, status = self._cand_floor or (None, "")
        out = threshold_guidance(masses, float(self._minmass), n_frames=n_frames,
                                 noise_floor=floor, floor_status=status)
        out["n_frames_sampled"] = int(n_frames)
        return out

    def _is_palmtracer_detector(self):
        if self._s is None:
            return False
        from firefly.ui.controllers.params.params_builder import BACKEND_LABEL_TO_VALUE
        return BACKEND_LABEL_TO_VALUE.get(self._s.get_str("analysis/backend", "Auto"), "auto") == "palmtracer"

    def _detector_kwargs(self):
        """Extra detector options for the preview: the palmTRACER-style detector
        thresholds on its own wavelet value, not minmass."""
        if not self._is_palmtracer_detector():
            return {}
        return {"wavelet_threshold": float(self._wavelet)}

    @Property(float, notify=detectChanged)
    def waveletThreshold(self):
        return self._wavelet

    @waveletThreshold.setter
    def waveletThreshold(self, v):
        v = max(0.0, float(v))
        if abs(v - self._wavelet) < 1e-9:
            return
        self._wavelet = v
        self.invalidateMassProfile()
        self._invalidate_spots()
        self._persist_threshold()
        self.detectChanged.emit()

    @Property(bool, notify=detectChanged)
    def thresholdPerFile(self):
        """Per-file for whichever threshold the selected detector uses."""
        return self._wavelet_per_file if self._is_palmtracer_detector() else self._minmass_per_file

    @Slot(bool)
    def setThresholdPerFile(self, on):
        if not self._is_palmtracer_detector():
            self.setMinmassPerFile(on)
            return
        on = bool(on)
        if on == self._wavelet_per_file or self._run_scoped:
            return
        self._wavelet_per_file = on
        if not on:
            # the file goes back to the shared value — show and use that
            if self._s is not None:
                self._wavelet = float(self._s.get_float("analysis/wavelet_threshold", self._wavelet))
            if self._ovr is not None and self._file:
                spec = self._ovr.get(self._file)
                if spec and "wavelet_threshold" in spec:
                    spec = dict(spec); spec.pop("wavelet_threshold", None)
                    self._ovr.set(self._file, spec)
            self.invalidateMassProfile()
            self._invalidate_spots()
        self.detectChanged.emit()

    @Property(bool, notify=detectChanged)
    def minmassApplies(self):
        """False for the palmTRACER-style detector, which ignores minmass — the
        threshold slider would otherwise move and change nothing."""
        return not self._is_palmtracer_detector()

    @Slot()
    def invalidateMassProfile(self):
        self._cand_masses = None
        self._cand_key = None
        self._cand_floor = None

    # ── brush / eraser ───────────────────────────────────────────────────
    # Design note: the brush is an INPUT METHOD for the existing polygon ROI,
    # not a new ROI kind.  Strokes accumulate in a boolean mask, and each stroke
    # end retraces that mask into `_polys`.  So `roi_mode` stays "Manual
    # polygon", params.json still carries `roi_polygon`, and multi-ROI replicate
    # fan-out, the post-hoc shrink guard and the saved mask PNG all keep working
    # untouched.  The alternative — shipping a raster mask through the worker —
    # would have needed new plumbing in every one of those places.

    @Property(str, notify=brushChanged)
    def tool(self):
        return self._tool

    @Slot(str)
    def setTool(self, name):
        name = str(name).lower()
        if name not in ("polygon", "brush", "eraser") or name == self._tool:
            return
        was_brush = self._tool in ("brush", "eraser")
        self._tool = name
        if name in ("brush", "eraser"):
            self.cancelDraft()              # an open polygon draft has no meaning here
            if not was_brush:
                self._begin_brush_session()
        else:
            self._end_brush_session()
        self.brushChanged.emit()

    @Property(float, notify=brushChanged)
    def brushRadius(self):
        return self._brush_radius

    @brushRadius.setter
    def brushRadius(self, r):
        r = max(0.5, min(float(r), 200.0))
        if abs(r - self._brush_radius) > 1e-9:
            self._brush_radius = r
            self.brushChanged.emit()

    @Property(bool, notify=brushChanged)
    def brushActive(self):
        return self._tool in ("brush", "eraser")

    @Property(bool, notify=brushChanged)
    def hasBrushPreview(self):
        return self._brush_img is not None and not self._brush_img.isNull()

    @Property(int, notify=brushChanged)
    def brushToken(self):
        return self._brush_token

    @Property(bool, notify=brushChanged)
    def canUndoStroke(self):
        return bool(self._brush_undo)

    def roi_brush_image(self):
        return self._brush_img

    def _begin_brush_session(self):
        """Seed the paint buffer from the polygons already drawn, so the brush
        EXTENDS an existing ROI instead of starting from blank."""
        H, W = int(self._img_h), int(self._img_w)
        if H <= 0 or W <= 0:
            self._brush_mask = None
            self.statusMessage.emit("Load an image before painting a region.")
            return
        from firefly.analysis.fa_roi import polygons_to_mask
        self._brush_mask = polygons_to_mask(self._polys, (H, W))
        self._brush_undo = []
        self._brush_holes_warned = False
        self._render_brush_preview()

    def _end_brush_session(self):
        self._brush_mask = None
        self._brush_img = None
        self._brush_undo = []
        self._brush_token += 1

    def _restart_brush(self):
        """The ROI just changed files (or reverted): the paint buffer, its
        preview and its undo history belonged to the previous one.  Left in
        place, the previous file's region stayed drawn over the next and the
        next stroke extended it.  Re-seed from the polygons now loaded."""
        self._end_brush_session()
        if self._tool in ("brush", "eraser"):
            self._begin_brush_session()
        self.brushChanged.emit()

    def _render_brush_preview(self):
        """Translucent green fill of the painted mask, for the overlay."""
        import numpy as np
        from PySide6.QtGui import QImage
        m = self._brush_mask
        if m is None or not m.any():
            self._brush_img = None
            self._brush_token += 1
            return
        h, w = m.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[m] = (86, 211, 100, 90)
        rgba = np.ascontiguousarray(rgba)
        self._brush_img = QImage(rgba.data, w, h, 4 * w,
                                 QImage.Format.Format_RGBA8888).copy()
        self._brush_token += 1

    @Slot()
    def beginStroke(self):
        """Snapshot for undo.  One entry per STROKE, not per painted dab."""
        if self._brush_mask is None:
            self._begin_brush_session()
        if self._brush_mask is not None:
            self._brush_undo.append(self._brush_mask.copy())
            del self._brush_undo[:-30]      # bound the history
            self.brushChanged.emit()

    @Slot(float, float)
    @Slot(float, float, float, float)
    def paintAt(self, y, x, y_prev=None, x_prev=None):
        """Stamp the brush at ``(y, x)`` in IMAGE pixels.

        Given the previous point too, the segment between them is filled, so a
        fast drag paints a continuous stroke instead of a dotted line.
        """
        import numpy as np
        if self._brush_mask is None:
            self._begin_brush_session()
        m = self._brush_mask
        if m is None:
            return
        H, W = m.shape
        r = float(self._brush_radius)
        pts = [(float(y), float(x))]
        if y_prev is not None and x_prev is not None:
            dy, dx = float(y) - float(y_prev), float(x) - float(x_prev)
            dist = float(np.hypot(dy, dx))
            if dist > 0:
                # step under a radius so consecutive dabs always overlap
                n = int(dist / max(r * 0.5, 0.5)) + 1
                pts = [(float(y_prev) + dy * k / n, float(x_prev) + dx * k / n)
                       for k in range(n + 1)]
        erase = self._tool == "eraser"
        rad = int(np.ceil(r))
        for py, px in pts:
            iy, ix = int(round(py)), int(round(px))
            y0, y1 = max(0, iy - rad), min(H, iy + rad + 1)
            x0, x1 = max(0, ix - rad), min(W, ix + rad + 1)
            if y0 >= y1 or x0 >= x1:
                continue
            yy = np.arange(y0, y1)[:, None] - py
            xx = np.arange(x0, x1)[None, :] - px
            disc = (yy * yy + xx * xx) <= r * r
            if erase:
                m[y0:y1, x0:x1] &= ~disc
            else:
                m[y0:y1, x0:x1] |= disc
        self._render_brush_preview()
        self.brushChanged.emit()

    @Slot()
    def endStroke(self):
        """Retrace the painted mask into polygons — the analysis ROI."""
        if self._brush_mask is None:
            return
        from firefly.analysis.fa_roi import count_mask_holes, mask_to_polygons
        holes = count_mask_holes(self._brush_mask)
        # A tolerance of 0 is exact but yields ~1400 vertices on a hand-painted
        # region; 0.5 keeps ~99% of the area at a quarter of the vertices, which
        # is what makes the shape editable and params.json small.
        polys = mask_to_polygons(self._brush_mask, simplify_tol=0.5)
        self._polys = [[(float(pt[0]), float(pt[1])) for pt in poly]
                       for poly in polys]
        del self._roi_labels[len(self._polys):]
        self.polygonsChanged.emit()
        self.splitChanged.emit()
        if holes and not self._brush_holes_warned:
            self._brush_holes_warned = True
            self.statusMessage.emit(
                f"Filled {holes} enclosed gap(s): an ROI is the union of its "
                f"regions, so a hole inside one cannot be analysed as excluded. "
                f"Erase from an edge instead.")

    @Slot()
    def undoStroke(self):
        if not self._brush_undo:
            return
        self._brush_mask = self._brush_undo.pop()
        self._render_brush_preview()
        self.brushChanged.emit()
        self.endStroke()

    @Slot()
    def clearBrush(self):
        if self._brush_mask is None:
            return
        self.beginStroke()
        self._brush_mask[:] = False
        self._render_brush_preview()
        self.brushChanged.emit()
        self.endStroke()

    @Slot(result="QVariantList")
    def getPolygons(self):
        return [[list(v) for v in poly] for poly in self._polys]

    @Property("QVariantList", notify=polygonsChanged)
    def polygons(self):
        return [[list(v) for v in poly] for poly in self._polys]

    @Property(int, notify=polygonsChanged)
    def polygonCount(self):
        return len(self._polys)

    # ── multiple ROIs → individual replicates ─────────────────────────────
    @Property(bool, notify=splitChanged)
    def splitReplicates(self):
        return self._split_replicates

    @splitReplicates.setter
    def splitReplicates(self, v):
        v = bool(v)
        if v != self._split_replicates:
            self._split_replicates = v
            self.splitChanged.emit()

    @Property("QStringList", notify=splitChanged)
    def roiLabels(self):
        # padded to the polygon count so QML can bind one label field per ROI
        return [(self._roi_labels[i] if i < len(self._roi_labels) else "")
                for i in range(len(self._polys))]

    @Slot(int, str)
    def setRoiLabel(self, idx, text):
        if idx < 0:
            return
        while len(self._roi_labels) <= idx:
            self._roi_labels.append("")
        self._roi_labels[idx] = str(text).strip()
        self.splitChanged.emit()

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

    @Property(str, notify=panelChanged)
    def panel(self):
        return self._panel

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
        """Open the ROI panel over ``path``'s projection, loading the file's ROI
        override (or the global default) + any stored polygon."""
        self._open(path, "roi")

    @Slot(str)
    def editDetection(self, path):
        """Open the detection-threshold panel over ``path``.

        Same viewer, same file, different screen — and the spot overlay comes up
        already on, because watching the dots answer the slider is the entire
        reason to be here; arriving with it off puts a click between the user and
        the evidence.
        """
        self._open(path, "detect", detect=True)

    def _open(self, path, panel, detect=None):
        self._file = path or ""
        self._rec_note = ""
        self.recommendChanged.emit()
        self._panel = panel
        if detect is not None:
            self._detect_on = bool(detect)
        # effective spec = per-file override, else the global sidebar default
        spec = (self._ovr.get(self._file) if self._ovr else None) or self._default_spec()
        self._apply_spec(spec)
        self._view_mode = "proj"
        self._frame_idx = 0
        self._raw_frame = None
        self._mask_proj = None
        self._mask_proj_mode = ""
        self._detect_green(self._file)        # offer the companion image as a view
        self._load_background(self._file)     # sets self._proj + n_frames, renders proj
        self._recompute_mask()
        # Pick up the latest SIDEBAR thresholds — unless this file carries its
        # own.  Re-reading the sidebar unconditionally (as this used to) showed a
        # reopened file at the wrong threshold, and the next Save then wrote the
        # sidebar's value over the file's own.
        if self._s is not None:
            if not self._minmass_per_file:
                self._minmass = float(self._s.get_float("analysis/minmass", self._minmass))
            if not self._wavelet_per_file:
                self._wavelet = float(self._s.get_float("analysis/wavelet_threshold", self._wavelet))
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
        self._restart_brush()
        self._editing = True
        self.roiSettingsChanged.emit()
        self.viewChanged.emit()
        self.polygonsChanged.emit()
        self.draftChanged.emit()
        self.editingChanged.emit()
        self.splitChanged.emit()
        self.panelChanged.emit()
        # Detection runs on acquired frames, so _recompute_spots draws nothing
        # over a projection — it posts "Select Raw frames" instead.  The ROI
        # panel wants the projection (every frame's signal at once, the right
        # canvas for tracing a neuron); the threshold panel wants a real frame,
        # or it opens with the overlay on, no dots, and a nag.
        if panel == "detect" and self._n_frames > 1:
            self.setViewMode("raw")

    @Slot(str, result=bool)
    def editRun(self, run_dir):
        """Open the editor over a COMPLETED run so its ROI can be added to or
        edited, then re-applied by ``firefly_worker.run_postproc``.

        Differs from :meth:`editFile` in four ways, each of which matters:

        * **Background.**  Prefer the original movie (the run manifest records
          its path), but fall back to rebuilding a localisation-density image
          from the run's own saved localisations at the recorded field size.
          The movies live on removable drives, so the fallback is the common
          case, not the exotic one — without it the feature is unavailable
          whenever the drive is unplugged.
        * **Forces Manual polygon mode.**  ``editFile`` inherits
          ``analysis/roi_mode`` from settings, which for the default "Auto
          threshold" leaves the drawing canvas hidden entirely.
        * **Seeds the run's saved polygon** so an existing ROI can be edited
          rather than only replaced.
        * **Run-scoped**: no write-through to the sidebar defaults.

        Keyed on the run directory, not the movie path, so a post-hoc ROI can
        never collide with the pre-run polygon stored for the same movie.
        """
        import os
        run_dir = str(run_dir or "")
        if not run_dir or not os.path.isdir(run_dir):
            self.statusMessage.emit("That run folder is not available.")
            return False

        self._run_scoped = True
        self._run_dir = run_dir
        self._file = run_dir                      # store key + header caption
        self._apply_spec(self._default_spec())
        self._roi_mode = "Manual polygon"         # set directly: the setter
                                                  # would push to settings
        self._view_mode = "proj"
        self._frame_idx = 0
        self._raw_frame = None
        self._mask_proj = None
        self._mask_proj_mode = ""
        self._green_path = ""                     # no companion image for a run
        self._detect_on = False                   # no live detection over a run
        # A finished run's detections are already written, so a threshold panel
        # here would offer a control that changes nothing.
        self._panel = "roi"
        self._spots = None
        self._spots_token += 1

        meta = self._run_meta(run_dir)
        movie = meta.get("movie") or ""
        if movie and os.path.isfile(movie):
            self._load_background(movie)          # the real thing when we have it
        else:
            self._load_run_locs_background(meta)
        self._n_frames = 0                        # hide the frame scrubber

        self._draft = []
        self._polys = [[(float(y), float(x)) for y, x in poly]
                       for poly in (meta.get("polygons") or [])]
        self._restart_brush()
        self._editing = True
        for sig in (self.roiSettingsChanged, self.viewChanged, self.polygonsChanged,
                    self.draftChanged, self.detectChanged, self.spotsChanged,
                    self.editingChanged, self.splitChanged, self.panelChanged):
            sig.emit()
        return True

    def _run_meta(self, run_dir):
        """{movie, polygons, width, height, stem, extras} read from a run folder.

        Everything is best-effort: a run produced before the polygon was
        persisted simply has no polygons to seed, and one whose manifest is
        missing just loses the movie shortcut.
        """
        import json
        import os
        out = {"movie": "", "polygons": None, "width": 0, "height": 0,
               "stem": "", "extras": ""}
        extras = os.path.join(run_dir, "firefly_extras")
        if not os.path.isdir(extras):
            return out
        out["extras"] = extras
        try:
            locs = [f for f in os.listdir(extras)
                    if f.endswith("_localisations.csv") and not f.startswith("._")]
            if locs:
                out["stem"] = locs[0][:-len("_localisations.csv")]
        except OSError:
            return out
        stem = out["stem"]
        try:
            with open(os.path.join(extras, f"{stem}_params.json"),
                      encoding="utf-8") as fh:
                params = json.load(fh) or {}
            out["width"] = int(params.get("width") or 0)
            out["height"] = int(params.get("height") or 0)
            out["polygons"] = params.get("roi_polygon") or None
        except Exception:
            pass
        for cand in (os.path.join(run_dir, f"{stem}_run_manifest.json"),
                     os.path.join(extras, f"{stem}_run_manifest.json")):
            try:
                with open(cand, encoding="utf-8") as fh:
                    out["movie"] = ((json.load(fh) or {})
                                    .get("input", {}).get("path") or "")
                break
            except Exception:
                continue
        return out

    def _load_run_locs_background(self, meta):
        """Rebuild the drawing canvas from the run's saved localisations.

        Reuses the worker's own ``_make_loc_histogram_proj``, which already
        prefers the run's recorded width/height — so the rebuilt image sits on
        exactly the movie's pixel grid and polygons drawn on it stay valid for
        the analysis.  (A canvas on any other grid would be silently rejected by
        the worker's ROI extent check.)
        """
        import os
        self._proj = None
        self._n_frames = 0
        try:
            import pandas as pd
            from firefly.firefly_worker import _make_loc_histogram_proj
            csv = os.path.join(meta.get("extras") or "",
                               f"{meta.get('stem')}_localisations.csv")
            if not os.path.isfile(csv):
                self.statusMessage.emit(
                    "This run has no saved localisations to draw on.")
                self._render_display()
                return
            locs = pd.read_csv(csv)
            proj = _make_loc_histogram_proj(
                locs, {"width": meta.get("width") or 0,
                       "height": meta.get("height") or 0})
            self._proj = proj[0] if getattr(proj, "ndim", 0) == 3 else proj
            self.statusMessage.emit(
                "Source movie not found — drawing on the run's own "
                "localisation density.")
        except Exception as exc:
            self.statusMessage.emit(f"Couldn't rebuild the run image: {exc}")
        self._render_display()

    @Slot(result="QVariantList")
    def runPolygons(self):
        """The polygons drawn over a run, for dispatch to run_postproc.

        Deliberately separate from :meth:`commit`, which writes the Import-tab
        override store — a run-scoped ROI has no business touching that.
        """
        return [[[float(y), float(x)] for y, x in poly] for poly in self._polys]

    @Property(bool, notify=editingChanged)
    def runScoped(self):
        return self._run_scoped

    @Property(str, notify=editingChanged)
    def runDir(self):
        return self._run_dir

    @Slot()
    def closeRun(self):
        """Leave a run-scoped edit without writing anything anywhere."""
        self._run_scoped = False
        self._run_dir = ""
        self._editing = False
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
        if self._view_mode == "raw":
            src = self._raw_frame
        elif self._view_mode == "green":
            src = self._green_img
        else:
            src = self._proj
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

    # ── view: max projection ↔ raw frame ↔ companion green image + scrub ───
    @Property("QStringList", notify=viewChanged)
    def viewModes(self):
        modes = ["Max projection", "Raw frames"]
        if self._green_path:
            modes.append("Green image")
        return modes

    @Property(str, notify=viewChanged)
    def viewMode(self):
        return self._view_mode

    @Slot(str)
    def setViewMode(self, mode):
        low = str(mode).lower()
        if low.startswith("raw"):
            m = "raw"
        elif low.startswith("green") and self._green_path:
            m = "green"
        else:
            m = "proj"
        if m == self._view_mode:
            return
        self._view_mode = m
        if m == "raw" and self._raw_frame is None:
            self._load_frame(self._frame_idx)
        elif m == "green" and self._green_img is None:
            self._load_green()
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
        if self._view_mode == "green":
            return self._green_note or self.greenName
        return "max projection"

    @Slot(int)
    def setFrame(self, i):
        i = int(i)
        if self._n_frames > 0:
            i = max(0, min(i, self._n_frames - 1))
        if i == self._frame_idx and self._raw_frame is not None:
            return
        self._invalidate_spots()
        self._frame_idx = i
        if self._view_mode == "raw":
            self._load_frame(i)
            self._render_display()
            # spots follow the frame too, but DEBOUNCED from QML (refreshSpots) —
            # locate() per scrub tick would lag, like the mask.
        self.frameChanged.emit(i)
        self.viewChanged.emit()

    def _load_frame(self, i):
        try:
            from firefly.ui.controllers.params.preview_loader import detection_frame
            channel = int(round(self._s.get_float("analysis/channel", 0))) if self._s else 0
            self._raw_frame = detection_frame(self._file, i, channel)
        except Exception:
            self._raw_frame = None

    # ── companion "green" image view ──────────────────────────────────────
    # Some recordings ship a companion widefield/marker image (…_green.tif)
    # beside them.  Offering it as a third view lets you SEE the cell the ROI
    # threshold is meant to cover — and in Sister-TIFF mode the green overlay
    # is built from this very image, so mask + backdrop are the same picture.
    def _sister_suffix(self):
        return (self._s.get_str("analysis/roi_sister_suffix", "_green")
                if self._s else "_green")

    def _detect_green(self, path):
        """Note whether a companion image exists (cheap path probe only — the
        pixels are loaded lazily, the first time the view is selected)."""
        self._green_path = ""
        self._green_img = None
        self._green_note = ""
        if not path:
            return
        try:
            from firefly.analysis.fa_roi import find_sister_roi_path
            self._green_path = find_sister_roi_path(path, self._sister_suffix()) or ""
        except Exception:
            self._green_path = ""

    def _load_green(self):
        """Load the companion image onto the stack's pixel grid, so the mask
        overlay and any drawn polygons land where they do in the other views."""
        self._green_img = None
        self._green_note = ""
        if not self._green_path:
            return
        try:
            from firefly.analysis.fa_roi import load_sister_image
            target = self._proj.shape if self._proj is not None else None
            arr, note = load_sister_image(self._green_path, target)
        except Exception as exc:
            self._green_note = f"Couldn't load companion image: {exc}"
            self.statusMessage.emit(self._green_note)
            return
        if arr is None:
            self._green_note = note
            self.statusMessage.emit(note)
            return
        self._green_img = arr
        self._green_note = f"{self.greenName}{note}"

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
        and must NOT touch the shared default.  Nor may a RUN-SCOPED edit: that
        run is already finished, and rewriting the sidebar from it would change
        the parameters of the NEXT analysis on the strength of a post-hoc ROI."""
        if self._batch_mode or self._run_scoped or self._s is None:
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
    @Property(str, notify=maskChanged)
    def sisterStatus(self):
        return self._sister_status

    # ── companion "green" image view ──────────────────────────────────────
    @Property(bool, notify=viewChanged)
    def hasGreenImage(self):
        return bool(self._green_path)

    @Property(str, notify=viewChanged)
    def greenName(self):
        import os
        return os.path.basename(self._green_path) if self._green_path else ""

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
    # Preview the production localizer on raw acquired frames. Acceptance
    # after contrast/known ROI filtering is distinct from track retention.
    @Property(bool, notify=detectChanged)
    def detectEnabled(self):
        return self._detect_on

    @detectEnabled.setter
    def detectEnabled(self, on):
        on = bool(on)
        if on and self._run_scoped:
            self.statusMessage.emit("Open the raw input to preview detection; completed-run settings are historical.")
            return
        if on == self._detect_on:
            return
        self._detect_on = on
        if on and not self._run_scoped and self._s is not None:
            self._s.set("analysis/auto_minmass", False)
        self._invalidate_spots()
        switched = on and self._view_mode != "raw"
        if switched:
            self.setViewMode("raw")
        self.detectChanged.emit()
        if not switched:
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
        self._invalidate_spots()
        self._persist_threshold()
        self.detectChanged.emit()
        # spots NOT rebuilt here — QML debounces refreshSpots (locate is heavy)

    def _persist_threshold(self):
        """Write the panel's threshold where it belongs.  Run-scoped: the
        threshold of a COMPLETED run is history — previewing over it must not
        rewrite the sidebar.  Per-file: it belongs to this file's override,
        saved on commit; writing the sidebar would change every other file.
        Otherwise it is the shared sidebar value (minmass also turns Auto
        minmass off).  Only values that changed are written: each write makes
        the whole parameter sidebar re-evaluate."""
        g = self._s
        if g is None or self._run_scoped:
            return
        if self._is_palmtracer_detector():
            if (not self._wavelet_per_file
                    and abs(g.get_float("analysis/wavelet_threshold", -1.0) - self._wavelet) > 1e-9):
                g.set("analysis/wavelet_threshold", self._wavelet)
        elif not self._minmass_per_file:
            if abs(g.get_float("analysis/minmass", -1.0) - self._minmass) > 1e-9:
                g.set("analysis/minmass", self._minmass)
            if g.get_bool("analysis/auto_minmass", False):
                g.set("analysis/auto_minmass", False)

    # ── threshold dragging ───────────────────────────────────────────────
    @Slot(float)
    def previewThreshold(self, v):
        """The slider is being dragged: move the preview's threshold only.
        Nothing is written until commitThreshold (writing the setting on every
        step made the whole sidebar re-evaluate per pixel of drag), the last
        overlay stays up, and the new one is detected off the GUI thread."""
        v = max(0.0, float(v))
        if self._is_palmtracer_detector():
            self._wavelet = v
        else:
            self._minmass = v
        self.detectChanged.emit()
        self._request_spots_async()

    @Slot(float)
    def commitThreshold(self, v):
        """The slider was released / a value typed: keep it, write it where it
        belongs, and show its detection at once (usually already cached)."""
        v = max(0.0, float(v))
        if self._is_palmtracer_detector():
            self._wavelet = v
            self.invalidateMassProfile()
        else:
            self._minmass = v
        self._persist_threshold()
        self.detectChanged.emit()
        if self._detect_on:
            self._recompute_spots()

    # ── recommended threshold ────────────────────────────────────────────
    @Property(bool, notify=recommendChanged)
    def recommending(self):
        return self._rec_busy

    @Property(str, notify=recommendChanged)
    def recommendNote(self):
        return self._rec_note

    @Slot()
    def recommendThreshold(self):
        """Move the threshold to FIREFLY's recommendation for the open
        recording, then commit it like a typed value.  palmTRACER-style
        detector: 4.4 × the recording's wavelet noise.  Every other detector:
        the minmass an Auto-threshold run would pick.  Worked out off the GUI
        thread — the minmass search links a few hundred frames."""
        if self._rec_busy or not self._file or self._run_scoped or self._n_frames <= 0:
            return
        g = self._s
        req = dict(file=self._file, n=int(self._n_frames),
                   channel=int(round(g.get_float("analysis/channel", 0))) if g else 0,
                   wavelet=self._is_palmtracer_detector(), params=None)
        if not req["wavelet"]:
            try:
                from firefly.ui.controllers.params.params_builder import build_params
                req["params"] = build_params(g, _PreviewImport(self._file, g), fpath=self._file,
                                             override_store=self._ovr)
            except Exception as exc:
                self._rec_note = f"No recommendation: {exc}"
                self.recommendChanged.emit()
                return
        self._rec_busy = True
        self._rec_note = "Working out the recommended threshold…"
        self.recommendChanged.emit()
        self._rec_thread = threading.Thread(target=self._recommend_worker, args=(req,),
                                            daemon=True, name="FIREFLY-RecommendThreshold")
        self._rec_thread.start()

    def _recommend_worker(self, req):
        try:
            from firefly.analysis import fa_localize as fl
            from firefly.ui.controllers.params.preview_loader import DetectionFrames
            stack = DetectionFrames(req["file"], req["n"], req["channel"])
            if req["wavelet"]:
                t, info = fl.recommend_wavelet_threshold(stack)
                note = (f"Recommended {t:g}: {info['factor']:g} × this recording's wavelet "
                        f"noise (σ = {info['sigma']:.1f} counts, {info['n_frames']} frames).")
            else:
                t, diag = fl.estimate_minmass_for_run(stack, req["params"], log_cb=lambda _m: None)
                method = str((diag or {}).get("method") or "")
                note = (f"Recommended {t:.3g}: the minmass an Auto-threshold run picks for "
                        f"this recording"
                        + (" (fallback estimate — the linking test was inconclusive)."
                           if method.startswith("static_fallback") else "."))
            self._recommendReady.emit(req, (float(t), note))
        except Exception as exc:                  # reported on the GUI thread
            self._recommendReady.emit(req, exc)

    def _on_recommended(self, req, result):
        self._rec_busy = False
        if req["file"] != self._file or req["wavelet"] != self._is_palmtracer_detector():
            self._rec_note = ""                   # another file / detector by now
        elif isinstance(result, Exception):
            self._rec_note = f"No recommendation: {result}"
        else:
            value, self._rec_note = result
            self.commitThreshold(value)
        self.recommendChanged.emit()

    @Property(bool, notify=spotsChanged)
    def spotsUpdating(self):
        """A detection for the current threshold is running."""
        return self._det_pending

    def _det_options(self, scale=1.0):
        """``(key, options)`` of the detection the panel currently asks for —
        the production localiser's own arguments (fa_detection_preview) — with
        the threshold multiplied by ``scale``."""
        from firefly.ui.controllers.params.params_builder import BG_METHOD_MAP, BACKEND_LABEL_TO_VALUE
        g = self._s
        extra = self._detector_kwargs()
        minmass = float(self._minmass)
        if "wavelet_threshold" in extra:
            extra = {"wavelet_threshold": extra["wavelet_threshold"] * scale}
        else:
            minmass *= scale
        opts = dict(
            diameter=int(round(g.get_float("analysis/diameter", 7))) if g else 7,
            minmass=minmass,
            bg_radius=int(round(g.get_float("analysis/bg_radius", 10))) if g else 10,
            bg_method=(BG_METHOD_MAP.get(g.get_str("analysis/bg_method", "Uniform Filter"),
                                         "uniform_filter") if g else "uniform_filter"),
            backend=(BACKEND_LABEL_TO_VALUE.get(g.get_str("analysis/backend", "Auto"), "auto")
                     if g else "auto"),
            **extra)
        channel = int(round(g.get_float("analysis/channel", 0))) if g else 0
        key = (self._file, int(self._frame_idx), channel, opts["diameter"], opts["bg_radius"],
               opts["bg_method"], opts["backend"], round(opts["minmass"], 9),
               round(float(opts.get("wavelet_threshold", -1.0)), 9))
        return key, opts

    def _view_dets(self):
        """The detections the overlay needs: at the threshold (green) and at
        _EXCLUDED_FRACTION of it, whose extra spots are what the threshold
        excludes (red).  ``None`` for the second when the threshold is 0."""
        key, opts = self._det_options()
        low = self._det_options(_EXCLUDED_FRACTION)
        return (key, opts), (low if low[0] != key else None)

    def _want(self, dets):
        (key, _opts), low = dets
        self._det_wanted = key               # a late result for another view cannot replace it
        self._det_low_wanted = low[0] if low else None

    def _show_cached(self):
        """Show the wanted view if its detections are cached; True if shown."""
        rows = self._det_cache.get(self._det_wanted)
        if rows is None:
            return False
        low = None
        if self._det_low_wanted is not None:
            low = self._det_cache.get(self._det_low_wanted)
            if low is None:
                return False
            self._det_cache.move_to_end(self._det_low_wanted)
        self._det_cache.move_to_end(self._det_wanted)
        self._det_pending = False
        self._apply_rows(rows, low)
        return True

    def _cache_put(self, key, rows):
        self._det_cache[key] = rows
        self._det_cache.move_to_end(key)
        while len(self._det_cache) > _DETECTION_CACHE_SIZE:
            self._det_cache.popitem(last=False)

    def _request_spots_async(self):
        if not self._detect_on or self._view_mode != "raw" or self._raw_frame is None:
            return
        dets = self._view_dets()
        self._want(dets)
        if self._show_cached():                  # seen before: show it at once
            return
        need = [d for d in dets if d is not None and d[0] not in self._det_cache]
        self._det_pending = True
        self.spotsChanged.emit()
        with self._det_lock:
            self._det_req = (self._raw_frame, need)
            if self._det_thread is None or not self._det_thread.is_alive():
                self._det_thread = threading.Thread(target=self._det_loop, daemon=True,
                                                    name="FIREFLY-SpotPreview")
                self._det_thread.start()

    def _det_loop(self):
        """Worker: detect the newest request; requests superseded while one
        runs are dropped.  Exits when there is nothing left to do."""
        from firefly.analysis import fa_detection_preview as fdp
        while True:
            with self._det_lock:
                req, self._det_req = self._det_req, None
                if req is None:
                    self._det_thread = None
                    return
            frame, need = req
            for key, opts in need:
                try:
                    rows, _ = fdp.preview_detections(frame, roi_mask=None, roi_known=False, **opts)
                    self._spotsDetected.emit(key, rows)
                except Exception as exc:          # reported on the GUI thread
                    self._spotsDetected.emit(key, exc)
                with self._det_lock:
                    if self._det_req is not None:   # superseded: start on the newest
                        break

    def _on_spots_detected(self, key, rows):
        if isinstance(rows, Exception):
            if key == self._det_wanted:
                self._det_pending = False
                self._spot_summary = f"Preview unavailable: {rows}"
                self.spotsChanged.emit()
            elif key == self._det_low_wanted:    # no red spots; the green ones stand
                self._det_low_wanted = None
                self._show_cached()
            return
        self._cache_put(key, rows)
        if key in (self._det_wanted, self._det_low_wanted):   # still the view on screen
            self._show_cached()

    def dispose(self):
        """Let running preview work finish and stop its threads."""
        with self._det_lock:
            self._det_req = None
        for th in (self._det_thread, self._rec_thread):
            if th is not None:
                th.join(timeout=10)

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

    def _spots_qimage(self, h, w, xs, ys, decisions=None):
        from PySide6.QtCore import QPointF, Qt as _Qt
        from PySide6.QtGui import QImage, QPainter, QPen, QColor
        img = QImage(int(w), int(h), QImage.Format.Format_ARGB32)
        img.fill(0)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(QColor(57, 255, 110)); pen.setWidthF(1.3)   # match live-detection green
        p.setPen(pen); p.setBrush(_Qt.NoBrush)
        colors = {"passes_detection": "#39ff6e", "roi_unchecked": "#55c8ff",
                  "outside_roi": "#b47cff", "low_contrast": "#ffb347",
                  "contrast_unavailable": "#ffb347", "below_threshold": "#ff2a2a"}
        halo = QPen(QColor(0, 0, 0, 200)); halo.setWidthF(3.2)
        for i, (x, y) in enumerate(zip(xs, ys)):
            c = QPointF(float(x), float(y))
            if decisions is not None:
                if decisions[i] == "below_threshold":
                    # red sits on red in Inferno/Hot: a dark halo keeps it legible
                    p.setPen(halo); p.drawEllipse(c, 4.0, 4.0)
                pen.setColor(QColor(colors[decisions[i]])); p.setPen(pen)
            p.drawEllipse(c, 4.0, 4.0)
        p.end()
        return img

    @Property(bool, notify=spotsChanged)
    def spotsStale(self):
        return self._spots_stale

    @Property(str, notify=spotsChanged)
    def spotSummary(self):
        return self._spot_summary

    @Property(str, notify=spotsChanged)
    def spotInspection(self):
        return self._spot_inspection

    def _invalidate_spots(self, *args, schedule=True):
        self._spots_stale = True
        self._spots = None
        self._spot_rows = None
        self._shown = None
        self._spot_count = 0
        self._spot_inspection = ""
        if schedule:
            self.previewInvalidated.emit()
        self._spot_summary = "Preview outdated — refresh to see current settings" if self._detect_on else "Preview off"
        self._spots_token += 1
        self.spotsChanged.emit()

    def _compose_spot_summary(self, summary, roi_note=""):
        """One wording for both the detect and the re-label paths, so the panel
        cannot say different things about the same overlay."""
        g = self._s
        backend = summary.get("backend") or "auto"
        thr = (f"wavelet threshold {self._wavelet:g}" if self._is_palmtracer_detector()
               else f"minmass {self._minmass:g}")
        text = (f"{backend} · frame {self._frame_idx + 1} · {thr}\n"
                f"{summary['candidates']} detected · "
                f"{summary['contrast_rejected']} contrast rejected · "
                f"{summary['outside_roi']} outside ROI · {summary['passed']} pass "
                + ("detection + ROI." if summary.get("roi_known")
                   else f"detection. ROI NOT evaluated: {roi_note}.")
                + (f" {summary['below_threshold']} excluded by the threshold (red: found at "
                   f"{_EXCLUDED_FRACTION:g}× it)." if summary.get("below_threshold") else "")
                + " Final track retention is not evaluated.")
        if g and g.get_bool("analysis/auto_minmass", False):
            text += " Auto minmass is ON: the run will choose a different threshold."
        return text

    def _reclassify_spots(self, *args):
        """Re-label the cached candidates against the current ROI — no detector.

        Falls back to a full invalidation only when there is nothing cached to
        re-label, or when the ROI cannot be resolved (an unclosed polygon, a
        missing sister image), because then the labels really are unknown.
        """
        if not self._detect_on or self._shown is None or self._raw_frame is None:
            self._invalidate_spots()
            return
        try:
            self._apply_rows(*self._shown)
        except Exception:
            self._invalidate_spots()

    def _apply_rows(self, rows, low=None):
        """Show detected candidates: label them against the current ROI and
        contrast cutoff (cheap — no detector) and redraw the overlay.  ``low``
        is the same detector at _EXCLUDED_FRACTION of the threshold; its spots
        that the threshold drops are drawn red."""
        import pandas as pd
        from firefly.analysis.fa_detection_preview import below_threshold, classify_candidates
        kept = rows
        g = self._s
        cnr = float(g.get_float("analysis/min_cnr", 0)) if g else 0.
        try:
            mask, known = self._detection_roi(self._raw_frame.shape)
            roi_note = ""
        except ValueError as exc:
            mask, known, roi_note = None, False, f"{exc}; fix/save the ROI before running"
        rows, summary = classify_candidates(
            rows, min_cnr=cnr, roi_mask=mask, roi_known=known,
            shape=self._raw_frame.shape,
            backend=(g.get_str("analysis/backend", "Auto") if g else "Auto"))
        shown = rows
        if low is not None and len(low):
            diam = float(g.get_float("analysis/diameter", 7)) if g else 7.0
            ex = below_threshold(kept, low, radius=max(2.0, diam / 2.0))
            if len(ex):
                ex, _ = classify_candidates(ex, roi_mask=mask, roi_known=known,
                                            shape=self._raw_frame.shape)
                ex["decision"] = "below_threshold"
                summary["below_threshold"] = len(ex)
                shown = pd.concat([ex, rows], ignore_index=True)   # green drawn on top
        self._shown = (kept, low)
        self._spot_rows = shown
        self._spot_count = summary["passed"]
        self._spots = self._spots_qimage(*self._raw_frame.shape,
                                         shown.x, shown.y, shown.decision.tolist())
        self._spots_stale = False
        self._spot_summary = self._compose_spot_summary(summary, roi_note)
        self._spots_token += 1
        self.spotsChanged.emit()

    def _spot_settings_changed(self, key):
        key = str(key)
        if self._s is not None and key in ("analysis/wavelet_threshold", "analysis/minmass"):
            cur = self._wavelet if key == "analysis/wavelet_threshold" else self._minmass
            if abs(self._s.get_float(key, cur) - cur) < 1e-9:
                return              # the panel's own value coming back: nothing changed
        if key == "analysis/auto_minmass":
            self.spotsChanged.emit()            # only the summary's note changes
            return
        if str(key).startswith("analysis/"):
            if key == "analysis/minmass" and self._s is not None and not self._minmass_per_file:
                self._minmass = float(self._s.get_float(key, self._minmass))
                self.detectChanged.emit()
            if key == "analysis/wavelet_threshold" and self._s is not None and not self._wavelet_per_file:
                self._wavelet = float(self._s.get_float(key, self._wavelet))
            if key in ("analysis/backend", "analysis/wavelet_threshold"):
                self.invalidateMassProfile()
                self.detectChanged.emit()           # minmassApplies may have flipped
            if key == "analysis/channel":
                self._raw_frame = None
                if self._view_mode == "raw":
                    self._load_frame(self._frame_idx)
                    self._render_display()
            self._invalidate_spots()

    @Slot(float, float)
    def inspectSpot(self, y, x):
        if self._spots_stale or self._spot_rows is None or not len(self._spot_rows):
            return
        import numpy as np
        rows = self._spot_rows
        distance = (rows.x-float(x))**2 + (rows.y-float(y))**2
        i = int(np.argmin(distance.to_numpy()))
        if distance.iloc[i] > 36:
            self._spot_inspection = ("No candidate within 6 pixels. Spots that need less than "
                                     f"{_EXCLUDED_FRACTION:g}× the threshold are not displayed.")
        else:
            r = rows.iloc[i]
            cnr = f"{r.raw_cnr:.3g}" if np.isfinite(r.raw_cnr) else "unavailable (edge/zero noise)"
            names = {"passes_detection":"passes detection + ROI", "roi_unchecked":"passes detection; ROI not evaluated",
                     "outside_roi":"rejected: outside ROI", "low_contrast":"rejected: raw contrast",
                     "contrast_unavailable":"rejected: raw contrast unavailable",
                     "below_threshold": f"excluded by the threshold (found at {_EXCLUDED_FRACTION:g}× it)"}
            roi = "not evaluated" if r.inside_roi is None else ("inside" if r.inside_roi else "outside")
            self._spot_inspection = (f"ROI={roi} · x={r.x:.2f}, y={r.y:.2f} · mass={r.mass:.4g} · raw CNR={cnr} · "
                                     + names[r.decision] + ". Track retention not evaluated.")
        self.spotsChanged.emit()

    def _detection_roi(self, shape):
        """Only promise ROI acceptance where the exact production mask is available."""
        import numpy as np
        from pathlib import Path
        from firefly.analysis.fa_roi import (build_sister_roi_mask, find_sister_roi_path,
            find_sibling_imagej_roi, load_roi_polygons_any)
        polys = self._polys
        mode = self._roi_mode
        if mode == "ImageJ ROI" and not polys:
            path = find_sibling_imagej_roi(str(Path(self._file).parent), Path(self._file).stem)
            if not path: raise ValueError("Requested ImageJ ROI is missing")
            polys = load_roi_polygons_any(path)
        if polys or mode == "Manual polygon":
            from skimage.draw import polygon2mask
            if not polys: raise ValueError("Draw and close the requested polygon ROI first")
            mask = np.zeros(shape, dtype=bool)
            for poly in polys:
                points = np.asarray(poly, dtype=float)
                if (points.min() < -1 or points[:,0].max() > shape[0]+1 or points[:,1].max() > shape[1]+1):
                    raise ValueError("Polygon extends beyond the camera frame")
                mask |= polygon2mask(shape, points)
            return mask, True
        if mode == "Sister TIFF":
            suffix = self._s.get_str("analysis/roi_sister_suffix", "_green") if self._s else "_green"
            path = find_sister_roi_path(self._file, suffix)
            if not path: raise ValueError("Requested sister ROI is missing")
            mask, note = build_sister_roi_mask(path, target_shape=shape)
            if mask is None: raise ValueError(note)
            return mask, True
        return None, mode == "None"

    def _recompute_spots(self):
        self._invalidate_spots(schedule=False)
        if not self._detect_on:
            return
        if self._view_mode != "raw":
            self._spot_summary = "Select Raw frames: projections/companion images are not detection inputs."
            self.spotsChanged.emit()
            return
        try:
            from firefly.analysis.fa_detection_preview import preview_detections
            from firefly.ui.controllers.params.params_builder import BG_METHOD_MAP, BACKEND_LABEL_TO_VALUE
            if self._raw_frame is None:
                self._load_frame(self._frame_idx)
                self._render_display()
            if self._raw_frame is None:
                raise ValueError("Cannot read the exact raw plane for this file/layout")
            dets = self._view_dets()
            self._want(dets)                     # a late drag result cannot replace it
            self._det_pending = False
            (key, opts), low = dets
            if key not in self._det_cache:
                rows, _ = preview_detections(self._raw_frame, roi_mask=None, roi_known=False,
                                             **opts)
                self._cache_put(key, rows)
            if low is not None and low[0] not in self._det_cache:
                try:
                    rows, _ = preview_detections(self._raw_frame, roi_mask=None,
                                                 roi_known=False, **low[1])
                    self._cache_put(low[0], rows)
                except Exception:                # no red spots; the green ones stand
                    self._det_low_wanted = None
            self._show_cached()
            return
        except Exception as exc:
            self._spot_summary = f"Preview unavailable: {exc}"
            self.statusMessage.emit(self._spot_summary)
        self._spots_token += 1
        self.spotsChanged.emit()

    # ── per-file override indicator ───────────────────────────────────────
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
                      or (is_poly and bool(self._polys))
                      or self._split_replicates or any(self._roi_labels)
                      or spec.get("minmass") is not None
                      or spec.get("wavelet_threshold") is not None)
            if custom:
                self._ovr.set(self._file, spec)
            else:
                self._ovr.clear(self._file)
        self._editing = False
        self._run_scoped = False        # never leak run scope into the next edit
        self._run_dir = ""
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
        self._restart_brush()
        spec = (self._ovr.get(self._file) if self._ovr else None) or self._default_spec()
        self._apply_spec(spec)
        self._editing = False
        self._run_scoped = False        # never leak run scope into the next edit
        self._run_dir = ""
        self.roiSettingsChanged.emit()
        self.polygonsChanged.emit()
        self.draftChanged.emit()
        self.editingChanged.emit()
