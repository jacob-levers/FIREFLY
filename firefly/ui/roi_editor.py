"""RoiEditor — FIREFLY's bespoke ROI drawing/preview widget.

A Qt-only (QGraphicsView) replacement for the napari-based ROI editors.  No
napari, no pyqtgraph — PySide6 + numpy only.  Supports everything the old
embedded-napari ROI viewers needed:

* an image **stack** (or a single 2-D background) + a frame slider,
* a swappable base image (raw ↔ bandpass-**filtered** view),
* a toggleable **max-projection** anatomy overlay (inferno, additive),
* a toggleable auto/manual-threshold **ROI mask** overlay (lime),
* a **detections** scatter (trackpy preview, coloured per point),
* interactive **polygon editing**: click to add vertices, right-click /
  double-click / Esc to close, drag vertex handles to reshape, plus
  programmatic ``set_polygons`` / ``add_polygons`` / ``clear_polygons``.

Polygons are stored and returned as lists of ``(y, x)`` pixel vertices —
directly consumable by ``skimage.draw.polygon2mask`` and the ``fa_roi``
parsers, exactly like the napari Shapes layer was.
"""
from __future__ import annotations

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QPointF, QRectF, Signal

from firefly.ui.viewer import (_gray_qimage, _lut_qimage, _robust_levels,
                               _PointsItem, _AdditivePixmapItem, _qcolor)

_EDGE = QtGui.QColor("#58a6ff")
_FILL = QtGui.QColor(88, 166, 255, 46)


def _mask_qimage(mask, color=(51, 255, 77), alpha=90):
    """bool mask → RGBA QImage (transparent where 0, ``color`` where 1)."""
    m = np.asarray(mask).astype(bool)
    h, w = m.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[m] = (color[0], color[1], color[2], alpha)
    rgba = np.ascontiguousarray(rgba)
    return QtGui.QImage(rgba.data, w, h, 4 * w,
                        QtGui.QImage.Format.Format_RGBA8888).copy()


class _Handle(QtWidgets.QGraphicsEllipseItem):
    """A draggable vertex handle drawn at constant on-screen size."""
    _R = 5.0

    def __init__(self, polygon):
        super().__init__(-self._R, -self._R, 2 * self._R, 2 * self._R)
        self._poly = polygon
        self.setBrush(QtGui.QBrush(QtGui.QColor("#ffffff")))
        self.setPen(QtGui.QPen(_EDGE, 1.5))
        self.setZValue(30)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges,
            True)
        self.setFlag(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations,
            True)

    def itemChange(self, change, value):
        if (change == QtWidgets.QGraphicsItem.GraphicsItemChange
                .ItemScenePositionHasChanged):
            self._poly.rebuild()
        return super().itemChange(change, value)


class _EditablePolygon:
    """A closed polygon: a filled path item + one draggable handle per vertex."""
    def __init__(self, editor, pts_xy):
        self._editor = editor
        self._scene = editor._scene
        self.path_item = QtWidgets.QGraphicsPathItem()
        self.path_item.setPen(QtGui.QPen(_EDGE, 0))   # cosmetic width below
        pen = QtGui.QPen(_EDGE)
        pen.setWidthF(2.0)
        pen.setCosmetic(True)
        self.path_item.setPen(pen)
        self.path_item.setBrush(QtGui.QBrush(_FILL))
        self.path_item.setZValue(20)
        self._scene.addItem(self.path_item)
        self.handles: list[_Handle] = []
        for x, y in pts_xy:
            h = _Handle(self)
            h.setPos(float(x), float(y))
            self._scene.addItem(h)
            self.handles.append(h)
        self.rebuild()

    def rebuild(self):
        if len(self.handles) < 2:
            self.path_item.setPath(QtGui.QPainterPath())
            return
        path = QtGui.QPainterPath()
        p0 = self.handles[0].scenePos()
        path.moveTo(p0)
        for h in self.handles[1:]:
            path.lineTo(h.scenePos())
        path.closeSubpath()
        self.path_item.setPath(path)

    def vertices_yx(self):
        return [(float(h.scenePos().y()), float(h.scenePos().x()))
                for h in self.handles]

    def remove(self):
        for h in self.handles:
            self._scene.removeItem(h)
        self.handles.clear()
        self._scene.removeItem(self.path_item)


class _RoiView(QtWidgets.QGraphicsView):
    """View with wheel-zoom, middle-drag pan, and polygon-drawing clicks."""
    def __init__(self, scene, editor):
        super().__init__(scene)
        self._editor = editor
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing
                            | QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._panning = False
        self._last = None
        self._drag_handle = False

    def wheelEvent(self, ev):
        f = 1.25 if ev.angleDelta().y() > 0 else 1.0 / 1.25
        self.scale(f, f)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._last = ev.position().toPoint()
            ev.accept()
            return
        if ev.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(ev.position().toPoint())
            if isinstance(item, _Handle):
                self._drag_handle = True
                super().mousePressEvent(ev)     # let the handle drag
                return
            self._editor._add_draft_vertex(
                self.mapToScene(ev.position().toPoint()))
            ev.accept()
            return
        if ev.button() == Qt.MouseButton.RightButton:
            self._editor._finish_draft()
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._panning and self._last is not None:
            p = ev.position().toPoint()
            d = p - self._last
            self._last = p
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - d.y())
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            ev.accept()
            return
        if ev.button() == Qt.MouseButton.LeftButton and self._drag_handle:
            self._drag_handle = False
            super().mouseReleaseEvent(ev)
            self._editor._emit_changed()        # a vertex was dragged
            return
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        self._editor._finish_draft()
        ev.accept()

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Escape:
            self._editor._finish_draft()
            ev.accept()
            return
        super().keyPressEvent(ev)


class RoiEditor(QtWidgets.QWidget):
    """Bespoke ROI editor.  See the module docstring."""

    polygonsChanged = Signal()
    frameChanged = Signal(int)

    def __init__(self, parent=None, *, background="#0d1117"):
        super().__init__(parent)
        self._stack = None
        self._levels = (0.0, 1.0)
        self._base_cmap = "gray"
        self._base_item: QtWidgets.QGraphicsPixmapItem | None = None
        self._maxproj_item: _AdditivePixmapItem | None = None
        self._mask_item: QtWidgets.QGraphicsPixmapItem | None = None
        self._points_item: _PointsItem | None = None
        self._polys: list[_EditablePolygon] = []
        self._draft: list[QPointF] = []
        self._draft_item: QtWidgets.QGraphicsPathItem | None = None
        self._draft_marks: list[QtWidgets.QGraphicsEllipseItem] = []

        self._scene = QtWidgets.QGraphicsScene(self)
        self._scene.setBackgroundBrush(QtGui.QColor(background))
        self._view = _RoiView(self._scene, self)

        self._slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider)
        self._frame_lbl = QtWidgets.QLabel("—")
        self._frame_lbl.setMinimumWidth(72)
        self._frame_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 2)
        bar.addWidget(QtWidgets.QLabel("Frame"))
        bar.addWidget(self._slider, 1)
        bar.addWidget(self._frame_lbl)
        self._bar = QtWidgets.QWidget()
        self._bar.setLayout(bar)
        self._bar.setVisible(False)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._view, 1)
        lay.addWidget(self._bar)

    # ── base image / stack ───────────────────────────────────────────────
    def set_stack(self, stack, *, cmap="gray", reset=True):
        """Show a ``(T,Y,X)`` / ``(Y,X)`` array as the base image + slider."""
        arr = np.asarray(stack)
        if arr.ndim == 2:
            arr = arr[None, ...]
        self._stack = arr
        self._base_cmap = cmap
        self._levels = _robust_levels(arr[arr.shape[0] // 2])
        h, w = arr.shape[1], arr.shape[2]
        self._scene.setSceneRect(0, 0, w, h)
        n = arr.shape[0]
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(0)
        self._slider.setEnabled(n > 1)
        self._slider.blockSignals(False)
        self._bar.setVisible(n > 1)
        self._show_frame(0)
        if reset:
            self.reset_view()

    def swap_base_stack(self, stack):
        """Replace the base pixels (e.g. raw↔filtered) keeping frame +
        polygons + overlays.  Must match the current stack's frame count."""
        arr = np.asarray(stack)
        if arr.ndim == 2:
            arr = arr[None, ...]
        self._stack = arr
        self._levels = _robust_levels(arr[arr.shape[0] // 2])
        self._show_frame(self.current_frame)

    @property
    def current_frame(self) -> int:
        return int(self._slider.value())

    def _on_slider(self, i):
        self._show_frame(int(i))
        self.frameChanged.emit(int(i))

    def _show_frame(self, i):
        if self._stack is None:
            return
        i = int(max(0, min(self._stack.shape[0] - 1, i)))
        if self._base_cmap == "gray":
            qimg = _gray_qimage(self._stack[i], self._levels)
        else:
            qimg = _lut_qimage(self._stack[i], self._base_cmap, self._levels)
        pix = QtGui.QPixmap.fromImage(qimg)
        if self._base_item is None:
            self._base_item = self._scene.addPixmap(pix)
            self._base_item.setZValue(0)
        else:
            self._base_item.setPixmap(pix)
        self._frame_lbl.setText(f"{i + 1}/{self._stack.shape[0]}")

    # ── max-projection overlay ───────────────────────────────────────────
    def set_max_projection(self, arr):
        self.clear_max_projection()
        if arr is None:
            return
        a = np.asarray(arr, float)
        item = _AdditivePixmapItem(QtGui.QPixmap.fromImage(
            _lut_qimage(a, "inferno", _robust_levels(a))))
        item.setOpacity(0.85)
        item.setZValue(2)
        item.setVisible(False)
        self._scene.addItem(item)
        self._maxproj_item = item

    def set_max_projection_visible(self, on):
        if self._maxproj_item is not None:
            self._maxproj_item.setVisible(bool(on))

    def clear_max_projection(self):
        if self._maxproj_item is not None:
            self._scene.removeItem(self._maxproj_item)
            self._maxproj_item = None

    # ── ROI mask overlay ─────────────────────────────────────────────────
    def set_mask(self, mask):
        self.clear_mask()
        if mask is None:
            return
        item = self._scene.addPixmap(QtGui.QPixmap.fromImage(_mask_qimage(mask)))
        item.setZValue(3)
        self._mask_item = item

    def clear_mask(self):
        if self._mask_item is not None:
            self._scene.removeItem(self._mask_item)
            self._mask_item = None

    # ── detections scatter ───────────────────────────────────────────────
    def set_detections(self, ys, xs, *, brushes=None, size=6):
        self.clear_detections()
        x = np.asarray(xs, float).ravel()
        y = np.asarray(ys, float).ravel()
        if x.size == 0:
            return
        if brushes is None:
            brushes = [QtGui.QColor(0, 255, 255, 230)] * x.size
        item = _PointsItem(x, y, brushes, size)
        item.setZValue(10)
        self._scene.addItem(item)
        self._points_item = item

    def clear_detections(self):
        if self._points_item is not None:
            self._scene.removeItem(self._points_item)
            self._points_item = None

    # ── polygons ─────────────────────────────────────────────────────────
    def _add_draft_vertex(self, scene_pt: QPointF):
        self._draft.append(scene_pt)
        if self._draft_item is None:
            self._draft_item = QtWidgets.QGraphicsPathItem()
            pen = QtGui.QPen(_EDGE)
            pen.setWidthF(2.0)
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._draft_item.setPen(pen)
            self._draft_item.setZValue(25)
            self._scene.addItem(self._draft_item)
        path = QtGui.QPainterPath()
        path.moveTo(self._draft[0])
        for p in self._draft[1:]:
            path.lineTo(p)
        self._draft_item.setPath(path)
        m = self._scene.addEllipse(-4, -4, 8, 8,
                                   QtGui.QPen(_EDGE, 1.0),
                                   QtGui.QBrush(QtGui.QColor("#ffffff")))
        m.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag
                  .ItemIgnoresTransformations, True)
        m.setPos(scene_pt)
        m.setZValue(26)
        self._draft_marks.append(m)

    def _finish_draft(self):
        pts = self._draft
        self._clear_draft_items()
        self._draft = []
        if len(pts) >= 3:
            self._polys.append(_EditablePolygon(
                self, [(p.x(), p.y()) for p in pts]))
            self._emit_changed()

    def _clear_draft_items(self):
        if self._draft_item is not None:
            self._scene.removeItem(self._draft_item)
            self._draft_item = None
        for m in self._draft_marks:
            self._scene.removeItem(m)
        self._draft_marks = []

    def polygons(self):
        """Return all finished polygons as lists of ``(y, x)`` vertices."""
        return [p.vertices_yx() for p in self._polys]

    def set_polygons(self, polys):
        self.clear_polygons(emit=False)
        self.add_polygons(polys, emit=False)

    def add_polygons(self, polys, *, emit=True):
        for poly in (polys or []):
            pts = [(float(x), float(y)) for y, x in np.asarray(poly)]
            if len(pts) >= 3:
                self._polys.append(_EditablePolygon(self, pts))
        if emit:
            self._emit_changed()

    def clear_polygons(self, *, emit=True):
        self._clear_draft_items()
        self._draft = []
        for p in self._polys:
            p.remove()
        self._polys = []
        if emit:
            self._emit_changed()

    def _emit_changed(self):
        self.polygonsChanged.emit()

    # ── camera ───────────────────────────────────────────────────────────
    def reset_view(self):
        if self._base_item is not None:
            self._view.fitInView(self._base_item.sceneBoundingRect(),
                                 Qt.AspectRatioMode.KeepAspectRatio)
