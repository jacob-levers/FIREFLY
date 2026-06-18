"""FireflyViewer — FIREFLY's own 2-D image/tracks/points viewer.

A fully **bespoke** viewer built only on Qt (``QGraphicsView`` / ``QImage`` /
``QPainter``) and numpy — no napari, no pyqtgraph, no vispy.  PySide6 and numpy
are already hard dependencies, so this adds **nothing** to the dependency tree.

It is a plain ``QWidget`` exposing a narrow, UI-policy-free API:

* a raw image **stack** + a Qt frame slider (time scrubbing),
* per-motion-class **track** polylines with independent visibility,
* a **points** overlay (DBSCAN clusters) with click-to-pick,
* an additive **super-resolution** image overlay,
* camera helpers (``center_on`` / ``reset_view``) + scroll-wheel zoom / drag-pan.

Coordinates are image-pixel ``(y, x)`` throughout.  The scene maps ``x`` →
horizontal and ``y`` → vertical (downwards), which already matches image
display, so no axis flip is needed.  Rendering and picking are pure numpy/Qt so
the same logic can drive a future QML viewer.
"""
from __future__ import annotations

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QPointF, QRectF, Signal


def _qcolor(c) -> QtGui.QColor:
    """Coerce a hex string / QColor / RGBA tuple to a QColor."""
    if isinstance(c, QtGui.QColor):
        return c
    if isinstance(c, (tuple, list, np.ndarray)):
        arr = list(c)
        if len(arr) >= 3 and max(arr[:3]) <= 1.0:
            rgb = [int(round(v * 255)) for v in arr[:3]]
        else:
            rgb = [int(v) for v in arr[:3]]
        col = QtGui.QColor(*rgb)
        if len(arr) >= 4:
            col.setAlpha(int(arr[3] * 255) if arr[3] <= 1 else int(arr[3]))
        return col
    return QtGui.QColor(str(c))


def _robust_levels(arr, lo=0.5, hi=99.8):
    """Percentile contrast limits, NaN-tolerant, with a safe fallback."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (0.0, 1.0)
    lo_v, hi_v = np.percentile(a, [lo, hi])
    if hi_v <= lo_v:
        hi_v = lo_v + 1.0
    return (float(lo_v), float(hi_v))


def _gray_qimage(frame, levels):
    """numpy (H,W) → 8-bit grayscale QImage stretched to ``levels``."""
    lo, hi = levels
    a = np.asarray(frame, dtype=float)
    a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    u8 = np.ascontiguousarray((a * 255).astype(np.uint8))
    h, w = u8.shape
    img = QtGui.QImage(u8.data, w, h, w,
                       QtGui.QImage.Format.Format_Grayscale8)
    return img.copy()   # copy so the QImage owns its buffer


def _lut_qimage(arr, cmap_name, levels):
    """numpy (H,W) → RGBA QImage coloured by a matplotlib colormap."""
    import matplotlib
    lo, hi = levels
    a = np.asarray(arr, dtype=float)
    a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    cmap = matplotlib.colormaps[cmap_name]
    rgba = np.ascontiguousarray((cmap(a) * 255).astype(np.uint8))   # (H,W,4)
    h, w = a.shape
    img = QtGui.QImage(rgba.data, w, h, 4 * w,
                       QtGui.QImage.Format.Format_RGBA8888)
    return img.copy()


class _AdditivePixmapItem(QtWidgets.QGraphicsPixmapItem):
    """A pixmap item that blends additively (Plus) over what's beneath it —
    the bespoke equivalent of napari's "additive" image blending, used for the
    super-resolution overlay so it glows over the raw frame."""
    def paint(self, painter, option, widget=None):
        painter.setCompositionMode(
            QtGui.QPainter.CompositionMode.CompositionMode_Plus)
        super().paint(painter, option, widget)


class _PointsItem(QtWidgets.QGraphicsItem):
    """Paints many coloured dots in a single item.  Points are grouped by
    colour so each colour is one native ``drawPoints`` call — fast even for the
    full localisation set."""
    def __init__(self, xs, ys, brushes, size):
        super().__init__()
        self._size = max(1, int(size))
        groups: dict[int, list] = {}
        for x, y, b in zip(xs, ys, brushes):
            col = _qcolor(b)
            key = col.rgba()
            groups.setdefault(key, [QtGui.QColor.fromRgba(key), []])[1].append(
                QPointF(float(x), float(y)))
        self._groups = [(col, QtGui.QPolygonF(pts))
                        for col, pts in groups.values()]
        if len(xs):
            pad = self._size
            self._rect = QRectF(float(np.min(xs)) - pad, float(np.min(ys)) - pad,
                                float(np.ptp(xs)) + 2 * pad,
                                float(np.ptp(ys)) + 2 * pad)
        else:
            self._rect = QRectF()

    def boundingRect(self):
        return self._rect

    def paint(self, painter, option, widget=None):
        for col, poly in self._groups:
            pen = QtGui.QPen(col)
            pen.setWidth(self._size)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setCosmetic(True)        # constant on-screen size while zooming
            painter.setPen(pen)
            painter.drawPoints(poly)


class _SceneView(QtWidgets.QGraphicsView):
    """QGraphicsView with wheel-zoom (anchored under the cursor), left-drag pan,
    and a click signal (a press+release that didn't drag) in scene coords."""
    sceneClicked = Signal(QPointF)

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing
                            | QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setMouseTracking(True)
        self._panning = False
        self._last = None
        self._press = None
        self._moved = 0.0

    def wheelEvent(self, ev):
        f = 1.25 if ev.angleDelta().y() > 0 else 1.0 / 1.25
        self.scale(f, f)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._press = ev.position().toPoint()
            self._last = self._press
            self._moved = 0.0
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._panning and self._last is not None:
            p = ev.position().toPoint()
            d = p - self._last
            self._last = p
            self._moved += abs(d.x()) + abs(d.y())
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - d.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - d.y())
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self._panning:
            self._panning = False
            if self._moved < 5.0 and self._press is not None:
                self.sceneClicked.emit(self.mapToScene(self._press))
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


class FireflyViewer(QtWidgets.QWidget):
    """A self-contained, dependency-free Qt viewer.  See the module docstring."""

    trackClicked = Signal(int)        # particle id of the nearest track
    clusterClicked = Signal(int)      # cluster id of the nearest point (-1 = noise)
    frameChanged = Signal(int)        # current stack frame index

    def __init__(self, parent=None, *, background="#0d1117"):
        super().__init__(parent)

        self._stack = None
        self._levels = (0.0, 1.0)
        self._track_items: dict[str, QtWidgets.QGraphicsPathItem] = {}
        self._track_pick: dict[str, tuple] = {}
        self._class_visible: dict[str, bool] = {}
        self._point_item: _PointsItem | None = None
        self._point_xy = None
        self._point_ids = None
        self._sr_item: _AdditivePixmapItem | None = None
        self._img_item: QtWidgets.QGraphicsPixmapItem | None = None
        self._track_frames: dict[str, np.ndarray] = {}   # per-vertex frame, ⟂ pick
        self._head_item: _PointsItem | None = None        # current-frame markers
        self._n_time = 0                                   # length of the time axis
        self._max_track_frame = -1

        self._scene = QtWidgets.QGraphicsScene(self)
        self._scene.setBackgroundBrush(QtGui.QColor(background))
        self._view = _SceneView(self._scene, self)
        self._view.sceneClicked.connect(self._on_scene_clicked)

        # ── playback bar: play/pause + scrub slider + fps ─────────────────
        self._play_btn = QtWidgets.QToolButton()
        self._play_btn.setText("▶")
        self._play_btn.setCheckable(True)
        self._play_btn.setToolTip("Play / pause the sequence (space)")
        self._play_btn.toggled.connect(self._on_play_toggled)

        self._slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider)
        self._frame_lbl = QtWidgets.QLabel("—")
        self._frame_lbl.setMinimumWidth(72)
        self._frame_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._fps_spin = QtWidgets.QSpinBox()
        self._fps_spin.setRange(1, 60)
        self._fps_spin.setValue(7)
        self._fps_spin.setSuffix(" fps")
        self._fps_spin.setToolTip("Playback frame rate")
        self._fps_spin.valueChanged.connect(self._on_fps_changed)

        self._play_timer = QtCore.QTimer(self)
        self._play_timer.timeout.connect(self._advance)

        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 2)
        bar.addWidget(self._play_btn)
        bar.addWidget(self._slider, 1)
        bar.addWidget(self._frame_lbl)
        bar.addWidget(self._fps_spin)
        self._bar_widget = QtWidgets.QWidget()
        self._bar_widget.setLayout(bar)
        self._bar_widget.setVisible(False)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._view, 1)
        lay.addWidget(self._bar_widget)

    # ─────────────────────────────────────────────────────────────────────
    # Image stack
    # ─────────────────────────────────────────────────────────────────────
    def set_stack(self, stack, *, autorange=True):
        """Display a ``(T, Y, X)`` (or ``(Y, X)``) array as the base image."""
        arr = np.asarray(stack)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim != 3:
            raise ValueError(f"stack must be 2-D or 3-D, got {arr.ndim}-D")
        self._stack = arr
        self._levels = _robust_levels(arr[arr.shape[0] // 2])
        h, w = arr.shape[1], arr.shape[2]
        self._scene.setSceneRect(0, 0, w, h)
        self._update_time_axis(reset=True)
        self._show_time(0)
        if autorange:
            self.reset_view()

    def clear_stack(self):
        if self._img_item is not None:
            self._scene.removeItem(self._img_item)
            self._img_item = None
        self._stack = None
        self._update_time_axis()

    # ── time axis (stack frames ∪ track frames) + playback ────────────────
    def _stack_len(self) -> int:
        return 0 if self._stack is None else int(self._stack.shape[0])

    def _update_time_axis(self, *, reset=False):
        """Recompute the time-axis length from the loaded stack AND tracks,
        and (re)size the scrub slider.  The bar shows whenever there's more
        than one time step from EITHER source — so the timeline works even
        when only tracks (no raw movie) are loaded."""
        n = max(self._stack_len(), self._max_track_frame + 1)
        self._n_time = n
        cur = self._slider.value()
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(0 if reset else min(cur, max(0, n - 1)))
        self._slider.setEnabled(n > 1)
        self._slider.blockSignals(False)
        self._play_btn.setEnabled(n > 1)
        self._fps_spin.setEnabled(n > 1)
        self._bar_widget.setVisible(n > 1)
        if n <= 1 and self._play_btn.isChecked():
            self._play_btn.setChecked(False)

    @property
    def n_frames(self) -> int:
        return int(self._n_time)

    @property
    def current_frame(self) -> int:
        return int(self._slider.value())

    @current_frame.setter
    def current_frame(self, i: int):
        if self._n_time <= 0:
            return
        self._slider.setValue(int(max(0, min(self._n_time - 1, i))))

    def _on_slider(self, i: int):
        self._show_time(int(i))
        self.frameChanged.emit(int(i))

    def _show_time(self, t: int):
        t = int(max(0, min(max(0, self._n_time - 1), t)))
        if self._stack is not None:
            j = min(t, self._stack_len() - 1)
            pix = QtGui.QPixmap.fromImage(
                _gray_qimage(self._stack[j], self._levels))
            if self._img_item is None:
                self._img_item = self._scene.addPixmap(pix)
                self._img_item.setZValue(0)
            else:
                self._img_item.setPixmap(pix)
        self._refresh_heads(t)
        self._frame_lbl.setText(f"{t + 1}/{max(1, self._n_time)}")

    def _on_play_toggled(self, on: bool):
        if on and self._n_time > 1:
            self._play_timer.start(max(16, 1000 // int(self._fps_spin.value())))
            self._play_btn.setText("❚❚")
        else:
            self._play_timer.stop()
            self._play_btn.setText("▶")

    def _on_fps_changed(self, fps: int):
        if self._play_timer.isActive():
            self._play_timer.start(max(16, 1000 // int(fps)))

    def _advance(self):
        if self._n_time <= 1:
            return
        self._slider.setValue((self._slider.value() + 1) % self._n_time)

    def _refresh_heads(self, t: int):
        """Draw a bright marker at each visible track's position at frame t,
        so scrubbing / playback animates the particles over the static tracks."""
        if self._head_item is not None:
            self._scene.removeItem(self._head_item)
            self._head_item = None
        xs, ys = [], []
        for cls, (xy, _pid) in self._track_pick.items():
            if not self._class_visible.get(cls, True):
                continue
            fr = self._track_frames.get(cls)
            if fr is None or not len(fr):
                continue
            m = fr == t
            if m.any():
                xs.append(xy[m, 0])
                ys.append(xy[m, 1])
        if not xs:
            return
        X = np.concatenate(xs)
        Y = np.concatenate(ys)
        item = _PointsItem(X, Y, [QtGui.QColor(255, 255, 255, 235)] * len(X), 7)
        item.setZValue(15)
        self._scene.addItem(item)
        self._head_item = item

    # ─────────────────────────────────────────────────────────────────────
    # Tracks
    # ─────────────────────────────────────────────────────────────────────
    def _add_track_item(self, cls, trajs, color, width):
        """trajs: list of (pid, yx_array, frames_array_or_None)."""
        path = QtGui.QPainterPath()
        pick_xy, pick_pid, pick_fr = [], [], []
        have_frames = True
        for pid, traj, frames in trajs:
            t = np.asarray(traj, dtype=float)
            if t.ndim != 2 or t.shape[0] < 2:
                continue
            y, x = t[:, 0], t[:, 1]
            path.moveTo(float(x[0]), float(y[0]))
            for k in range(1, len(x)):
                path.lineTo(float(x[k]), float(y[k]))
            pick_xy.append(np.column_stack([x, y]))
            pick_pid.append(np.full(len(x), int(pid), np.int64))
            if frames is None:
                have_frames = False
            else:
                pick_fr.append(np.asarray(frames))
        if path.isEmpty():
            return
        item = QtWidgets.QGraphicsPathItem(path)
        pen = QtGui.QPen(_qcolor(color))
        pen.setWidthF(float(width))
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(Qt.BrushStyle.NoBrush)
        item.setZValue(10)
        self._scene.addItem(item)
        self._track_items[cls] = item
        self._class_visible[cls] = True
        self._track_pick[cls] = (np.concatenate(pick_xy),
                                 np.concatenate(pick_pid))
        if have_frames and pick_fr:
            self._track_frames[cls] = np.concatenate(pick_fr)

    def set_tracks(self, tracks_by_class: dict, colors: dict, *, width=1.5):
        """Render per-class trajectories.  ``tracks_by_class``:
        ``{class: [traj, ...]}`` with each ``traj`` an ``(N, 2)`` array of
        ``(y, x)``; the trajectory index is used as the pick id."""
        self.clear_tracks()
        for cls, trajs in tracks_by_class.items():
            self._add_track_item(
                cls, [(i, tr, None) for i, tr in enumerate(trajs)],
                colors.get(cls, colors.get("Unknown", "#888")), width)
        self._update_time_axis()

    def set_tracks_from_df(self, df, motion_map: dict, colors: dict, *,
                           min_len: int = 1, width=1.5):
        """Build per-class trajectories from a tracks DataFrame
        (particle/frame/x/y) + a ``{particle: motion}`` map.  Returns
        ``{class: {particle_ids}}``.  Pick arrays carry the real particle id."""
        out: dict[str, set] = {}
        self.clear_tracks()
        if df is None or not len(df):
            return out
        lens = df.groupby("particle").size()
        sub = df.sort_values(["particle", "frame"], kind="mergesort")
        by_class: dict[str, list] = {}
        for pid, g in sub.groupby("particle"):
            if int(lens.get(pid, 0)) < max(2, min_len):
                continue
            cls = str(motion_map.get(int(pid), "Unknown"))
            by_class.setdefault(cls, []).append(
                (int(pid), np.column_stack([g["y"].to_numpy(float),
                                            g["x"].to_numpy(float)]),
                 g["frame"].to_numpy()))
        for cls in _order_classes(by_class.keys()):
            trajs = by_class.get(cls)
            if not trajs:
                continue
            self._add_track_item(cls, trajs,
                                 colors.get(cls, colors.get("Unknown", "#888")),
                                 width)
            out[cls] = set(int(pid) for pid, _, _ in trajs)
        # Tracks define their own time axis (frame range), so the timeline +
        # playback work even when no raw movie is loaded.
        try:
            self._max_track_frame = max(
                (int(fr.max()) for fr in self._track_frames.values() if len(fr)),
                default=-1)
        except Exception:
            self._max_track_frame = -1
        self._update_time_axis()
        self._refresh_heads(self.current_frame)
        return out

    def set_class_visible(self, cls: str, visible: bool):
        item = self._track_items.get(cls)
        if item is not None:
            item.setVisible(bool(visible))
            self._class_visible[cls] = bool(visible)
            self._refresh_heads(self.current_frame)

    def class_names(self) -> list:
        return list(self._track_items.keys())

    def clear_tracks(self):
        for item in self._track_items.values():
            try:
                self._scene.removeItem(item)
            except Exception:
                pass
        self._track_items.clear()
        self._track_pick.clear()
        self._class_visible.clear()
        self._track_frames.clear()
        self._max_track_frame = -1
        if self._head_item is not None:
            try:
                self._scene.removeItem(self._head_item)
            except Exception:
                pass
            self._head_item = None
        self._update_time_axis()

    def recolor_tracks(self, colors: dict, *, width=1.5):
        for cls, item in self._track_items.items():
            pen = QtGui.QPen(_qcolor(colors.get(cls, colors.get("Unknown",
                                                                "#888"))))
            pen.setWidthF(float(width))
            pen.setCosmetic(True)
            item.setPen(pen)

    # ─────────────────────────────────────────────────────────────────────
    # Points (DBSCAN clusters)
    # ─────────────────────────────────────────────────────────────────────
    def set_points(self, ys, xs, *, ids=None, brushes=None, size=4):
        """Scatter overlay in pixel coords; ``brushes`` a per-point colour
        list; ``ids`` parallels the points for click resolution."""
        self.clear_points()
        x = np.asarray(xs, float).ravel()
        y = np.asarray(ys, float).ravel()
        if x.size == 0:
            return
        if brushes is None:
            brushes = [QtGui.QColor(120, 180, 255, 200)] * x.size
        item = _PointsItem(x, y, brushes, size)
        item.setZValue(20)
        self._scene.addItem(item)
        self._point_item = item
        self._point_xy = np.column_stack([x, y])
        self._point_ids = (np.asarray(ids).ravel() if ids is not None
                           else np.arange(x.size))

    def clear_points(self):
        if self._point_item is not None:
            try:
                self._scene.removeItem(self._point_item)
            except Exception:
                pass
        self._point_item = None
        self._point_xy = None
        self._point_ids = None

    # ─────────────────────────────────────────────────────────────────────
    # Super-resolution overlay
    # ─────────────────────────────────────────────────────────────────────
    def set_superres(self, img, *, scale=1.0, translate=(0.0, 0.0),
                     cmap="inferno", opacity=0.9):
        """Add a super-res image as an additive overlay.  ``scale`` is raw px
        per super-res px; ``translate`` is the ``(ty, tx)`` origin offset."""
        self.clear_superres()
        arr = np.asarray(img, float)
        lo, hi = _robust_levels(arr, 0, 99.5)
        qimg = _lut_qimage(arr, cmap, (lo, hi))
        item = _AdditivePixmapItem(QtGui.QPixmap.fromImage(qimg))
        item.setOpacity(float(opacity))
        ty, tx = translate
        tr = QtGui.QTransform()
        tr.translate(float(tx), float(ty))
        tr.scale(float(scale), float(scale))
        item.setTransform(tr)
        item.setZValue(5)
        self._scene.addItem(item)
        self._sr_item = item

    def clear_superres(self):
        if self._sr_item is not None:
            try:
                self._scene.removeItem(self._sr_item)
            except Exception:
                pass
        self._sr_item = None

    @property
    def has_superres(self) -> bool:
        return self._sr_item is not None

    # ─────────────────────────────────────────────────────────────────────
    # Camera
    # ─────────────────────────────────────────────────────────────────────
    def center_on(self, y, x, *, span=None):
        """Recentre on ``(y, x)``; if ``span`` is given, fit that square."""
        if span:
            self._view.fitInView(QRectF(x - span / 2, y - span / 2, span, span),
                                 Qt.AspectRatioMode.KeepAspectRatio)
        else:
            self._view.centerOn(float(x), float(y))

    def reset_view(self):
        rect = (self._img_item.sceneBoundingRect() if self._img_item is not None
                else self._scene.itemsBoundingRect())
        if rect.isValid() and rect.width() > 0 and rect.height() > 0:
            self._view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def visible_rect(self) -> QRectF:
        """The scene rectangle currently visible in the viewport."""
        return self._view.mapToScene(
            self._view.viewport().rect()).boundingRect()

    # ─────────────────────────────────────────────────────────────────────
    # Click resolution (points take priority over tracks)
    # ─────────────────────────────────────────────────────────────────────
    def _data_tol(self, px=8.0) -> float:
        m = abs(self._view.transform().m11())
        return float(px) / m if m > 1e-9 else 8.0

    def pick_at(self, y, x, *, tol=None):
        """Resolve a data-space click at ``(y, x)``.  Returns
        ``("cluster", id)`` (points take priority) or ``("track", pid)`` for
        the nearest visible match within ``tol``, else ``None``."""
        if tol is None:
            tol = self._data_tol()
        px, py = float(x), float(y)

        if self._point_xy is not None and len(self._point_xy):
            d = np.hypot(self._point_xy[:, 0] - px, self._point_xy[:, 1] - py)
            j = int(np.argmin(d))
            if d[j] <= tol:
                return ("cluster", int(self._point_ids[j]))

        best_pid, best_d = None, np.inf
        for cls, (xy, pid) in self._track_pick.items():
            if not self._class_visible.get(cls, True) or not len(xy):
                continue
            d = np.hypot(xy[:, 0] - px, xy[:, 1] - py)
            j = int(np.argmin(d))
            if d[j] < best_d:
                best_d, best_pid = d[j], int(pid[j])
        if best_pid is not None and best_d <= tol:
            return ("track", best_pid)
        return None

    def _on_scene_clicked(self, scene_pt: QPointF):
        hit = self.pick_at(scene_pt.y(), scene_pt.x())
        if hit is None:
            return
        kind, ident = hit
        if kind == "cluster":
            self.clusterClicked.emit(int(ident))
        else:
            self.trackClicked.emit(int(ident))


def _order_classes(present) -> list:
    """Known motion order first, then any extras seen in the data."""
    order = ["Immobile", "Confined", "Brownian", "Directed", "Unknown"]
    present = list(present)
    return ([c for c in order if c in present]
            + [c for c in present if c not in order])
