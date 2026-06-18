"""RoiController — QML bridge for polygon ROI editing (Phase 4).

Owns a HEADLESS polygon model (list of ``(y, x)`` vertex lists + an optional open
draft) so the QML layer can build/inspect ROIs without a live widget, and lazily
builds the bespoke RoiEditor island when embedded.  The model adds the per-item
delete verbs the RoiEditor widget lacks (delete_polygon / delete_vertex).  The
public convention is ``(y, x)`` end-to-end (matching RoiEditor.polygons /
set_polygons); the flip to ``(x, y)`` stays inside the RoiEditor renderer.
"""
from __future__ import annotations

from PySide6.QtCore import Property, QObject, Signal, Slot


class RoiController(QObject):
    polygonsChanged = Signal()
    draftChanged = Signal()
    frameChanged = Signal(int)
    editingChanged = Signal()
    imageChanged = Signal()
    statusMessage = Signal(str)

    def __init__(self, store=None, parent=None):
        super().__init__(parent)
        self._editor = None
        self._polys: list = []          # list[list[(y, x)]]
        self._draft: list = []          # open polygon being drawn
        # ── per-file manual-polygon editing (Phase 6c) ───────────────────
        self._store = store
        self._file = ""
        self._image = None              # QImage max-projection background
        self._img_w = 0
        self._img_h = 0
        self._img_token = 0
        self._editing = False

    # ── editor lifecycle ─────────────────────────────────────────────────
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
        # Pull the editor's committed polygons back into the headless model.
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
                # A polygon with <3 vertices is degenerate — drop it.
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

    # ── properties ───────────────────────────────────────────────────────
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

    # ── per-file editing over the input image (Phase 6c) ──────────────────
    def roi_image(self):
        """The current max-projection background QImage (read by the provider)."""
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
        """Open the manual-polygon editor over ``path``'s max-projection, loading
        any polygon already stored for that file."""
        self._file = path or ""
        self._load_background(self._file)
        self._draft = []
        existing = self._store.get(self._file) if self._store else None
        self._polys = [[(float(y), float(x)) for y, x in poly]
                       for poly in (existing or [])]
        self._editing = True
        self.polygonsChanged.emit()
        self.draftChanged.emit()
        self.editingChanged.emit()

    def _load_background(self, path):
        from firefly.ui.controllers.live_frame_provider import render_frame
        self._image = None
        self._img_w = self._img_h = 0
        try:
            import os
            if not (path and os.path.isfile(path)):
                return
            from firefly.sptpalm_analysis import load_file
            self.statusMessage.emit(f"Loading {os.path.basename(path)}…")
            stack, _, _ = load_file(path, channel=0)
            import numpy as np
            arr = np.asarray(stack)
            maxproj = arr.max(axis=0) if arr.ndim == 3 else arr
            img = render_frame(maxproj.astype("float32"))
            if not img.isNull():
                self._image = img
                self._img_w, self._img_h = img.width(), img.height()
        except Exception as exc:
            self.statusMessage.emit(f"Couldn't load image: {exc}")
        finally:
            self._img_token += 1
            self.imageChanged.emit()

    @Slot()
    def commit(self):
        """Store the drawn polygons under the current file and close the editor."""
        if self._store is not None:
            self._store.set(self._file, self._polys)
        self._editing = False
        self.editingChanged.emit()
        self.statusMessage.emit(
            f"{len(self._polys)} ROI polygon(s) saved" if self._polys
            else "ROI cleared")

    @Slot()
    def cancel(self):
        """Discard edits and close — revert to whatever was stored."""
        existing = self._store.get(self._file) if self._store else None
        self._polys = [[(float(y), float(x)) for y, x in poly]
                       for poly in (existing or [])]
        self._draft = []
        self._editing = False
        self.polygonsChanged.emit()
        self.draftChanged.emit()
        self.editingChanged.emit()

    @Slot(str, result=bool)
    def fileHasRoi(self, path):
        return bool(self._store and self._store.has(path))
