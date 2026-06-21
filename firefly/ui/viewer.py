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
from PySide6.QtCore import Qt, QPointF, QRectF, QLineF, Signal

# Pure rendering/picking logic shared with the in-scene FireflyPaintedItem
# (Phase 8) — no behaviour fork, one place to test.
from firefly.ui._viewer_core import (
    _qcolor, _robust_levels, _gray_qimage, _lut_qimage, _order_classes,
    build_class_model, tail_window, df_to_class_trajs, group_points_by_color,
    head_xy_for_frame, pick_at as _pick_at_core)


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
        self._groups, self._rect = group_points_by_color(xs, ys, brushes, self._size)

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


class _HeadItem(QtWidgets.QGraphicsItem):
    """A reusable single-colour scatter of current-frame position markers,
    updated IN PLACE each playback tick (one item, never added/removed) so the
    scene never churns during playback."""
    def __init__(self, *, color=None, size=7):
        super().__init__()
        self._poly = QtGui.QPolygonF()
        self._rect = QRectF()
        self._pen = QtGui.QPen(color or QtGui.QColor(255, 255, 255, 235))
        self._pen.setWidth(int(size))
        self._pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self._pen.setCosmetic(True)
        self.setZValue(15)

    def set_points(self, xs, ys):
        self.prepareGeometryChange()
        self._poly = QtGui.QPolygonF(
            [QPointF(float(x), float(y)) for x, y in zip(xs, ys)])
        if len(xs):
            pad = 8.0
            self._rect = QRectF(float(np.min(xs)) - pad, float(np.min(ys)) - pad,
                                float(np.ptp(xs)) + 2 * pad,
                                float(np.ptp(ys)) + 2 * pad)
        else:
            self._rect = QRectF()
        self.update()

    def count(self) -> int:
        return self._poly.count()

    def boundingRect(self):
        return self._rect

    def paint(self, painter, option, widget=None):
        if self._poly.count():
            painter.setPen(self._pen)
            painter.drawPoints(self._poly)


class _TailItem(QtWidgets.QGraphicsItem):
    """A motion class's track **tail**.

    Holds every track segment for the class, pre-sorted by frame, and each tick
    paints only the segments whose frame lies in ``(t - tail, t]`` — so the view
    shows the trajectories *near the current frame*, the way napari's Tracks
    layer does, instead of every full track at once.  Window selection is a
    contiguous ``searchsorted`` slice, so it's O(log n) + the (small) draw.
    """
    def __init__(self, lines, frames, color, width, rect):
        super().__init__()
        self._lines = lines              # list[QLineF], sorted by frame
        self._frames = frames            # np.ndarray[int], sorted, ⟂ lines
        self._rect = rect
        self._lo, self._hi = 0, 0
        self._pen = QtGui.QPen(_qcolor(color))
        self._pen.setWidthF(float(width))
        self._pen.setCosmetic(True)
        self.setZValue(10)

    def set_color(self, color, width):
        self._pen = QtGui.QPen(_qcolor(color))
        self._pen.setWidthF(float(width))
        self._pen.setCosmetic(True)
        self.update()

    def set_width(self, width):
        self._pen.setWidthF(float(width))
        self.update()

    def set_window(self, t, tail, head=0):
        """Show segments with frame in ``(t - tail, t + head]`` — ``tail``
        frames of history behind the playhead and ``head`` frames ahead."""
        lo, hi = tail_window(self._frames, t, tail, head)
        if (lo, hi) != (self._lo, self._hi):
            self._lo, self._hi = lo, hi
            self.update()

    def boundingRect(self):
        return self._rect

    def paint(self, painter, option, widget=None):
        if self._hi > self._lo:
            # AA off for the tail strokes: antialiasing thousands of cosmetic
            # line segments is pathologically slow in Qt's raster engine (tens
            # of ms); off it's a few ms and 1px lines still read fine.  Restore
            # the hint afterwards so the markers/points above stay crisp.
            aa = QtGui.QPainter.RenderHint.Antialiasing
            was = bool(painter.renderHints() & aa)
            painter.setRenderHint(aa, False)
            painter.setPen(self._pen)
            painter.drawLines(self._lines[self._lo:self._hi])
            painter.setRenderHint(aa, was)


class _SceneView(QtWidgets.QGraphicsView):
    """QGraphicsView with wheel-zoom (anchored under the cursor), left-drag pan,
    and a click signal (a press+release that didn't drag) in scene coords."""
    sceneClicked = Signal(QPointF)

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        # Antialiasing for the vector overlays, but NO SmoothPixmapTransform so
        # the image (and any cached pixmaps) blit fast during playback —
        # nearest-neighbour is also the more honest scaling for raw pixels.
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(
            QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setMouseTracking(True)
        # Accept keyboard focus so spacebar-play works once the viewer is
        # clicked (the parent FireflyViewer.keyPressEvent handles Space).
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._panning = False
        self._last = None
        self._press = None
        self._moved = 0.0

    def wheelEvent(self, ev):
        # Zoom proportional to the scroll delta, not a fixed step per event: a
        # trackpad emits many small wheel events, so a fixed 1.25×/event rocketed
        # the view in/out.  ~10% per mouse-wheel notch (delta 120); clamped so a
        # fling can't jump wildly.
        delta = ev.angleDelta().y()
        if not delta:
            return
        f = pow(1.0008, max(-700, min(700, delta)))
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
    _maxprojReady = Signal()          # full max projection finished (off-thread)

    def __init__(self, parent=None, *, background=None):
        super().__init__(parent)
        if background is None:
            # match the active theme's page canvas (the colour behind the tabs)
            # so the viewer doesn't read as a different shade.
            try:
                from firefly.ui.ui_theme import _THEMES, _pick_startup_theme
                background = _THEMES.get(_pick_startup_theme(), _THEMES["Dark"])["BG"]
            except Exception:
                background = "#090b0f"

        self._stack = None
        self._levels = (0.0, 1.0)
        self._frame_pix_cache: dict[int, QtGui.QPixmap] = {}
        self._pix_cache_cap = 256                          # frames; sized in set_stack
        self._track_items: dict[str, QtWidgets.QGraphicsPathItem] = {}
        self._track_pick: dict[str, tuple] = {}
        self._class_visible: dict[str, bool] = {}
        self._point_item: _PointsItem | None = None
        self._point_xy = None
        self._point_ids = None
        self._img_item: QtWidgets.QGraphicsPixmapItem | None = None
        # selectable background sources for the base layer
        self._maxproj_full = None             # full max projection (computed off-thread)
        self._maxproj_quick = None            # instant strided-sample preview
        self._maxproj_computing = False
        self._maxprojReady.connect(self._on_maxproj_ready)
        self._sr = None                       # (img, scale, (ty, tx)) or None
        self._bg_mode = "Raw movie"
        self._track_frames: dict[str, np.ndarray] = {}   # per-vertex frame, ⟂ pick
        self._n_time = 0                                   # length of the time axis
        self._max_track_frame = -1

        self._scene = QtWidgets.QGraphicsScene(self)
        self._scene.setBackgroundBrush(QtGui.QColor(background))
        # We pick via numpy (pick_at), not the scene's spatial index, so turn the
        # index OFF — otherwise it rebuilds on every item add/remove and the
        # marker churn stutters playback badly.
        self._scene.setItemIndexMethod(
            QtWidgets.QGraphicsScene.ItemIndexMethod.NoIndex)
        # One persistent marker item, updated in place each tick.
        self._head_item = _HeadItem()
        self._head_item.setVisible(False)
        self._scene.addItem(self._head_item)
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
        self._fps_spin.setValue(60)
        self._fps_spin.setSuffix(" fps")
        self._fps_spin.setToolTip("Playback frame rate")
        self._fps_spin.valueChanged.connect(self._on_fps_changed)

        # ── track-display controls (width / tail / head) ─────────────────
        self._width_spin = QtWidgets.QDoubleSpinBox()
        self._width_spin.setRange(0.5, 12.0)
        self._width_spin.setSingleStep(0.5)
        self._width_spin.setValue(1.5)
        self._width_spin.setPrefix("w ")
        self._width_spin.setMaximumWidth(80)
        self._width_spin.setToolTip("Track line width (px).")
        self._width_spin.valueChanged.connect(self._on_width_changed)

        self._tail_spin = QtWidgets.QSpinBox()
        self._tail_spin.setRange(1, 100000)
        self._tail_spin.setValue(30)
        self._tail_spin.setPrefix("tail ")
        self._tail_spin.setMaximumWidth(96)
        self._tail_spin.setToolTip(
            "Tail length: frames of trajectory history shown BEHIND the current\n"
            "frame.  Small = clean, fast playback; raise it toward the clip\n"
            "length to approach whole-track view.")
        self._tail_spin.valueChanged.connect(self._on_tail_changed)

        self._head_spin = QtWidgets.QSpinBox()
        self._head_spin.setRange(0, 100000)
        self._head_spin.setValue(0)
        self._head_spin.setPrefix("head ")
        self._head_spin.setMaximumWidth(96)
        self._head_spin.setToolTip(
            "Head length: frames of trajectory shown AHEAD of the current frame\n"
            "(where each particle is going).  0 = none.")
        self._head_spin.valueChanged.connect(self._on_tail_changed)

        self._play_timer = QtCore.QTimer(self)
        self._play_timer.timeout.connect(self._advance)

        # Row 1 — playback; Row 2 — track display.
        row1 = QtWidgets.QHBoxLayout()
        row1.setContentsMargins(4, 2, 4, 0)
        row1.addWidget(self._play_btn)
        row1.addWidget(self._slider, 1)
        row1.addWidget(self._frame_lbl)
        row1.addWidget(self._fps_spin)
        row2 = QtWidgets.QHBoxLayout()
        row2.setContentsMargins(4, 0, 4, 2)
        row2.setSpacing(8)
        self._bg_combo = QtWidgets.QComboBox()
        self._bg_combo.setToolTip(
            "Background layer shown beneath the tracks:\n"
            "Raw movie (per-frame) · Max projection · Super-resolution · Off.")
        self._bg_combo.currentTextChanged.connect(self._on_bg_changed)
        self._bg_combo.setVisible(False)

        row2.addWidget(QtWidgets.QLabel("Tracks:"))
        row2.addWidget(self._width_spin)
        row2.addWidget(self._tail_spin)
        row2.addWidget(self._head_spin)
        row2.addStretch(1)
        row2.addWidget(QtWidgets.QLabel("Background:"))
        row2.addWidget(self._bg_combo)
        # Track-display + background row — hidden here; surfaced in the QML
        # Visualise SETTINGS rail instead (the spins/combo still drive the view).
        self._row2_widget = QtWidgets.QWidget()
        self._row2_widget.setLayout(row2)
        self._row2_widget.setVisible(False)
        barv = QtWidgets.QVBoxLayout()
        barv.setContentsMargins(0, 0, 0, 0)
        barv.setSpacing(0)
        barv.addLayout(row1)
        barv.addWidget(self._row2_widget)
        self._bar_widget = QtWidgets.QWidget()
        self._bar_widget.setLayout(barv)
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
        self._frame_pix_cache.clear()
        h, w = arr.shape[1], arr.shape[2]
        # Bound the frame-pixmap cache to ~250 MB regardless of frame size, so a
        # large movie can't OOM (a 2048² frame → ~16 MB pixmap → ~16 frames).
        self._pix_cache_cap = max(8, int(2.5e8 / (h * w * 4 + 1)))
        self._maxproj_full = None; self._maxproj_quick = None; self._maxproj_computing = False
        self._scene.setSceneRect(0, 0, w, h)
        self._update_time_axis(reset=True)
        self._update_bg_options()           # registers Raw/Max + applies the bg
        if autorange:
            self.reset_view()

    def clear_stack(self):
        if self._img_item is not None:
            self._scene.removeItem(self._img_item)
            self._img_item = None
        self._stack = None
        self._maxproj_full = None; self._maxproj_quick = None; self._maxproj_computing = False
        self._update_bg_options()
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
        # The transport is now driven by the QML scrubber (HudOverlay / VisualiseTab
        # via VisualiseController); keep these native widgets alive as the backing
        # state + play timer, but never show the dated native bar.
        self._bar_widget.setVisible(False)
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
        # Only the Raw-movie background animates per frame; Max projection /
        # Super-resolution / Off are static, set once by _apply_background.
        if self._bg_mode == "Raw movie" and self._stack is not None:
            j = min(t, self._stack_len() - 1)
            pix = self._frame_pix_cache.get(j)
            if pix is None:
                pix = QtGui.QPixmap.fromImage(
                    _gray_qimage(self._stack[j], self._levels))
                if len(self._frame_pix_cache) >= self._pix_cache_cap:
                    self._frame_pix_cache.clear()
                self._frame_pix_cache[j] = pix
            self._set_base_pixmap(pix, None)
        self._apply_tail_windows(t)
        self._refresh_heads(t)
        self._frame_lbl.setText(f"{t + 1}/{max(1, self._n_time)}")

    # ── selectable background layer ───────────────────────────────────────
    def _set_base_pixmap(self, pix, transform):
        if self._img_item is None:
            self._img_item = self._scene.addPixmap(pix)
            self._img_item.setZValue(0)
        else:
            self._img_item.setPixmap(pix)
        self._img_item.setVisible(True)
        self._img_item.setTransform(transform or QtGui.QTransform())

    def _maxproj_array(self):
        # A full max projection over a 10⁴-frame movie reads the whole stack and
        # stalls the UI. Show an instant strided-SAMPLE preview, compute the full
        # projection on a background thread, and swap it in when ready (an SMLM
        # max needs every frame — blinks land at different pixels each frame).
        if self._stack is None:
            return None
        if self._maxproj_full is not None:
            return self._maxproj_full
        self._ensure_maxproj_full()
        if self._maxproj_quick is None:
            n = self._stack.shape[0]
            stride = max(1, n // 256)
            sample = self._stack[::stride] if stride > 1 else self._stack
            self._maxproj_quick = sample.max(axis=0)
        return self._maxproj_quick

    def _ensure_maxproj_full(self):
        """Compute the full max projection once, off the GUI thread (or inline
        for a small stack where the sample already IS the full)."""
        if (self._maxproj_full is not None or self._maxproj_computing
                or self._stack is None):
            return
        if self._stack.shape[0] <= 256:
            self._maxproj_full = self._stack.max(axis=0)
            return
        self._maxproj_computing = True
        stack = self._stack
        import threading

        def _work():
            try:    mp = stack.max(axis=0)
            except Exception: mp = None
            self._maxproj_full = mp
            self._maxproj_computing = False
            try:    self._maxprojReady.emit()       # GUI-thread refresh (queued)
            except Exception: pass
        threading.Thread(target=_work, daemon=True).start()

    def _on_maxproj_ready(self):
        if self._bg_mode == "Max projection":
            self._apply_background()                 # swap the preview for the full proj

    def _update_bg_options(self):
        """Rebuild the Background combo from what's available + apply it."""
        opts = []
        if self._stack is not None:
            opts += ["Raw movie", "Max projection"]
        if self._sr is not None:
            opts.append("Super-resolution")
        opts.append("Off")
        if self._bg_mode not in opts:
            self._bg_mode = opts[0]
        self._bg_combo.blockSignals(True)
        self._bg_combo.clear()
        self._bg_combo.addItems(opts)
        self._bg_combo.setCurrentText(self._bg_mode)
        self._bg_combo.blockSignals(False)
        self._bg_combo.setVisible(len(opts) > 1)
        self._apply_background()

    def _on_bg_changed(self, text: str):
        if text:
            self._bg_mode = text
            self._apply_background()

    def _apply_background(self):
        mode = self._bg_mode
        if mode == "Max projection":
            mp = self._maxproj_array()
            if mp is not None:
                self._set_base_pixmap(
                    QtGui.QPixmap.fromImage(_gray_qimage(mp, _robust_levels(mp))),
                    None)
        elif mode == "Super-resolution" and self._sr is not None:
            img, scale, (ty, tx) = self._sr
            tr = QtGui.QTransform()
            tr.translate(tx, ty)
            tr.scale(scale, scale)
            self._set_base_pixmap(
                QtGui.QPixmap.fromImage(
                    _lut_qimage(img, "inferno", _robust_levels(img, 0, 99.5))),
                tr)
        elif mode == "Off":
            if self._img_item is not None:
                self._img_item.setVisible(False)
        # "Raw movie" base is drawn per-frame by _show_time.
        self._show_time(self.current_frame)

    def _apply_tail_windows(self, t: int):
        tail = int(self._tail_spin.value())
        head = int(self._head_spin.value())
        for item in self._track_items.values():
            item.set_window(t, tail, head)

    def _on_tail_changed(self, _v=None):
        self._apply_tail_windows(self.current_frame)

    def _on_width_changed(self, _v=None):
        w = float(self._width_spin.value())
        for item in self._track_items.values():
            item.set_width(w)

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
        """Move the bright markers to each visible track's position at frame t,
        so scrubbing / playback animates the particles over the static tracks.
        Updates the ONE persistent head item in place (no add/remove churn)."""
        if self._head_item is None:
            return
        xs, ys = head_xy_for_frame(self._track_pick, self._track_frames,
                                   self._class_visible, t)
        if len(xs):
            self._head_item.set_points(xs, ys)
            self._head_item.setVisible(True)
        else:
            self._head_item.set_points([], [])
            self._head_item.setVisible(False)

    # ─────────────────────────────────────────────────────────────────────
    # Tracks
    # ─────────────────────────────────────────────────────────────────────
    def _add_track_item(self, cls, trajs, color, width):
        """trajs: list of (pid, yx_array, frames_array_or_None).  Builds a
        frame-sorted set of segments rendered as a time-windowed tail."""
        model = build_class_model(trajs)
        if model is None:
            return
        item = _TailItem(model["lines"], model["frames"], color, width,
                         model["rect"])
        self._scene.addItem(item)
        self._track_items[cls] = item
        self._class_visible[cls] = True
        self._track_pick[cls] = (model["pick_xy"], model["pick_pid"])
        self._track_frames[cls] = model["pick_fr"]

    def set_tracks(self, tracks_by_class: dict, colors: dict, *, width=1.5):
        """Render per-class trajectories.  ``tracks_by_class``:
        ``{class: [traj, ...]}`` with each ``traj`` an ``(N, 2)`` array of
        ``(y, x)``; the trajectory index is used as the pick id."""
        self.clear_tracks()
        for cls, trajs in tracks_by_class.items():
            self._add_track_item(
                cls, [(i, tr, None) for i, tr in enumerate(trajs)],
                colors.get(cls, colors.get("Unknown", "#888")),
                self._width_spin.value())
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
        by_class = df_to_class_trajs(df, motion_map, min_len)
        for cls in _order_classes(by_class.keys()):
            trajs = by_class.get(cls)
            if not trajs:
                continue
            self._add_track_item(cls, trajs,
                                 colors.get(cls, colors.get("Unknown", "#888")),
                                 self._width_spin.value())
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
        # Tracks-only load: park the playhead at the LAST frame so the tail
        # view isn't empty on arrival (a movie load starts at frame 0 instead).
        if self._stack is None and self._n_time > 1:
            self._slider.blockSignals(True)
            self._slider.setValue(self._n_time - 1)
            self._slider.blockSignals(False)
        self._show_time(self.current_frame)
        # Keep the Background selector populated even on a tracks-only load (no
        # stack) so the control is always available — "Off" until a movie/super-
        # res layer is added, then Raw movie / Max projection appear.
        self._update_bg_options()
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
            self._head_item.set_points([], [])
            self._head_item.setVisible(False)
        self._update_time_axis()

    def recolor_tracks(self, colors: dict, *, width=None):
        w = self._width_spin.value() if width is None else width
        for cls, item in self._track_items.items():
            item.set_color(colors.get(cls, colors.get("Unknown", "#888")), w)

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
        # Cache the (static) cluster scatter so it isn't re-drawn point-by-point
        # on every playback frame — up to ~200k points otherwise re-paint each
        # tick when the markers above force a full-field redraw.
        item.setCacheMode(
            QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)
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
    # Super-resolution (a selectable background layer)
    # ─────────────────────────────────────────────────────────────────────
    def set_superres(self, img, *, scale=1.0, translate=(0.0, 0.0),
                     cmap="inferno", opacity=0.9):
        """Store a super-res reconstruction and show it as the background.
        ``scale`` is raw px per super-res px; ``translate`` is ``(ty, tx)``.
        Adds a "Super-resolution" entry to the Background selector."""
        self._sr = (np.asarray(img, float), float(scale),
                    (float(translate[0]), float(translate[1])))
        self._bg_mode = "Super-resolution"
        self._update_bg_options()

    def clear_superres(self):
        self._sr = None
        if self._bg_mode == "Super-resolution":
            self._bg_mode = "Raw movie" if self._stack is not None else "Off"
        self._update_bg_options()

    @property
    def has_superres(self) -> bool:
        return self._sr is not None

    # ─────────────────────────────────────────────────────────────────────
    # Headless transport façade (Phase 4)
    # ─────────────────────────────────────────────────────────────────────
    # Thin property wrappers over the existing control-bar widgets so a QML
    # VisualiseController can drive the viewer without reaching into private
    # widgets.  The spins/combos stay the canonical store — every setter routes
    # through the SAME slot the user's interaction does (_on_tail_changed /
    # _on_width_changed / _on_fps_changed / _on_play_toggled / _on_bg_changed),
    # so behaviour is identical whether driven from QML or the native bar.
    @property
    def tail(self) -> int:
        return int(self._tail_spin.value())

    @tail.setter
    def tail(self, v: int):
        self._tail_spin.setValue(int(v))

    @property
    def head(self) -> int:
        return int(self._head_spin.value())

    @head.setter
    def head(self, v: int):
        self._head_spin.setValue(int(v))

    @property
    def track_width(self) -> float:
        return float(self._width_spin.value())

    @track_width.setter
    def track_width(self, v: float):
        self._width_spin.setValue(float(v))

    @property
    def fps(self) -> int:
        return int(self._fps_spin.value())

    @fps.setter
    def fps(self, v: int):
        self._fps_spin.setValue(int(v))

    @property
    def playing(self) -> bool:
        return bool(self._play_btn.isChecked())

    @playing.setter
    def playing(self, on: bool):
        self._play_btn.setChecked(bool(on))

    @property
    def background_mode(self) -> str:
        return str(self._bg_mode)

    @background_mode.setter
    def background_mode(self, mode: str):
        # Only an available layer can be selected — setCurrentText is a no-op
        # for an unknown mode, which is the correct (can't-select-missing)
        # behaviour.  Routes through _on_bg_changed → _apply_background.
        self._bg_combo.setCurrentText(str(mode))

    def background_options(self) -> list:
        """Current Background-selector layers (Raw movie / Max projection /
        Super-resolution / Off), in order."""
        return [self._bg_combo.itemText(i)
                for i in range(self._bg_combo.count())]

    def keyPressEvent(self, ev):
        # Spacebar toggles playback when the viewer island has focus (the
        # _SceneView grabs focus; an unhandled space bubbles up to here).
        # Scoped to the focused widget — NOT an app/window QShortcut — so it
        # never steals space from a QML text field elsewhere.
        if ev.key() == Qt.Key.Key_Space:
            self._play_btn.toggle()
            ev.accept()
            return
        super().keyPressEvent(ev)

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
        rect = (self._img_item.sceneBoundingRect()
                if self._img_item is not None and self._img_item.isVisible()
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
        return _pick_at_core(self._point_xy, self._point_ids, self._track_pick,
                             self._class_visible, y, x, tol,
                             track_frames=self._track_frames,
                             t=self.current_frame, tail=self.tail, head=self.head)

    def _on_scene_clicked(self, scene_pt: QPointF):
        hit = self.pick_at(scene_pt.y(), scene_pt.x())
        if hit is None:
            return
        kind, ident = hit
        if kind == "cluster":
            self.clusterClicked.emit(int(ident))
        else:
            self.trackClicked.emit(int(ident))
