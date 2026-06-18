"""FireflyView — the FIREFLY 2-D viewer as an in-scene QQuickPaintedItem (Phase 8).

An Option-A re-author of the bespoke ``FireflyViewer`` (a native QGraphicsView
embedded under QML via Option B + EmbedController) as a normal QML item: no
native surface, no two-QQuickWidget compositing, no geometry-sync, no modal-
occlusion bridge.  It reuses the shared pure logic in :mod:`_viewer_core` (so it
never forks the native viewer's rendering/picking) and re-authors only the
QGraphicsScene plumbing as a single ``paint()`` + a manual ``(scale, offx, offy)``
camera, with wheel-zoom-under-cursor / drag-pan / click-pick / spacebar-play.

Exposes the SAME public method surface as FireflyViewer (set_stack /
set_tracks_from_df / set_class_visible / set_points / set_superres / current_frame
/ tail / head / track_width / fps / playing / background_mode / pick_at +
trackClicked/clusterClicked/frameChanged), so VisualiseController drives it
unchanged.  Coordinates are image-pixel ``(y, x)`` throughout.
"""
from __future__ import annotations

import numpy as np

from PySide6 import QtGui
from PySide6.QtCore import Qt, QPointF, QRectF, QTimer, Property, Signal
from PySide6.QtGui import QPolygonF, QTransform
from PySide6.QtQuick import QQuickPaintedItem
from PySide6.QtQml import QmlElement

from firefly.ui import _viewer_core as core

QML_IMPORT_NAME = "Firefly"
QML_IMPORT_MAJOR_VERSION = 1


@QmlElement
class FireflyView(QQuickPaintedItem):
    trackClicked = Signal(int)
    clusterClicked = Signal(int)
    frameChanged = Signal(int)
    playingChanged = Signal()
    nFramesChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptedMouseButtons(Qt.MouseButton.AllButtons)
        self.setFlag(QQuickPaintedItem.Flag.ItemAcceptsInputMethod, False)
        self.setFillColor(QtGui.QColor("#0d1117"))

        # ── background / image state ─────────────────────────────────────
        self._stack = None
        self._levels = (0.0, 1.0)
        self._frame_cache: dict = {}
        self._cache_cap = 64
        self._maxproj = None
        self._sr = None                  # (img, scale, (ty, tx))
        self._bg_mode = "Off"
        self._base_img = None            # QImage
        self._base_tr = QTransform()     # SR translate·scale (identity otherwise)

        # ── tracks / points / heads ──────────────────────────────────────
        self._tail_recs: dict = {}       # cls → {lines, frames, lo, hi, pen, rect}
        self._track_pick: dict = {}
        self._track_frames: dict = {}
        self._class_visible: dict = {}
        self._max_track_frame = -1
        self._point_groups: list = []
        self._point_xy = None
        self._point_ids = None
        self._point_size = 4
        self._head_poly = QPolygonF()
        self._head_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 235))
        self._head_pen.setWidth(7)
        self._head_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self._head_pen.setCosmetic(True)

        # ── transport state ──────────────────────────────────────────────
        self._cur = 0
        self._n_time = 0
        self._tail = 30
        self._head = 0
        self._width = 1.5
        self._fps = 7
        self._playing = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._advance)

        # ── camera ───────────────────────────────────────────────────────
        self._scale = 1.0
        self._offx = 0.0
        self._offy = 0.0
        self._fitted = False
        self._panning = False
        self._press = None
        self._last = None
        self._moved = 0.0

    # ═════════════════════════════════════════════════════════════════════
    #  paint
    # ═════════════════════════════════════════════════════════════════════
    def paint(self, p: QtGui.QPainter):
        p.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, False)
        p.save()
        p.translate(self._offx, self._offy)
        p.scale(self._scale, self._scale)

        if self._bg_mode != "Off" and self._base_img is not None:
            if self._bg_mode == "Super-resolution":
                p.save()
                p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Plus)
                p.setTransform(self._base_tr, True)
                p.drawImage(0, 0, self._base_img)
                p.restore()
            else:
                p.drawImage(0, 0, self._base_img)

        # tails — AA OFF for the whole block (perf), restore after
        aa = QtGui.QPainter.RenderHint.Antialiasing
        was = bool(p.renderHints() & aa)
        p.setRenderHint(aa, False)
        for cls in core._order_classes(self._tail_recs.keys()):
            if not self._class_visible.get(cls, True):
                continue
            rec = self._tail_recs[cls]
            if rec["hi"] > rec["lo"]:
                p.setPen(rec["pen"])
                p.drawLines(rec["lines"][rec["lo"]:rec["hi"]])
        p.setRenderHint(aa, was)

        if self._head_poly.count():
            p.setPen(self._head_pen)
            p.drawPoints(self._head_poly)

        for col, poly in self._point_groups:
            pen = QtGui.QPen(col)
            pen.setWidth(self._point_size)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.drawPoints(poly)
        p.restore()

    def geometryChange(self, new, old):
        super().geometryChange(new, old)
        if not self._fitted and new.width() > 1 and self._has_content():
            self.reset_view()

    # ═════════════════════════════════════════════════════════════════════
    #  background / stack
    # ═════════════════════════════════════════════════════════════════════
    def set_stack(self, stack, *, autorange=True):
        self._stack = np.asarray(stack)
        if self._stack.ndim == 2:
            self._stack = self._stack[None]
        mid = self._stack[len(self._stack) // 2]
        self._levels = core._robust_levels(mid)
        self._frame_cache.clear()
        self._maxproj = None
        if self._bg_mode == "Off":
            self._bg_mode = "Raw movie"
        self._rebuild_time_axis()
        self._fitted = False
        self._refresh_base()
        if autorange:
            self.reset_view()
        self.update()

    def clear_stack(self):
        self._stack = None
        self._maxproj = None
        self._frame_cache.clear()
        if self._bg_mode in ("Raw movie", "Max projection"):
            self._bg_mode = "Super-resolution" if self._sr is not None else "Off"
        self._rebuild_time_axis()
        self._refresh_base()
        self.update()

    def _stack_len(self):
        return 0 if self._stack is None else len(self._stack)

    def _frame_image(self, j):
        img = self._frame_cache.get(j)
        if img is None:
            img = core._gray_qimage(self._stack[j], self._levels)
            if len(self._frame_cache) >= self._cache_cap:
                self._frame_cache.clear()
            self._frame_cache[j] = img
        return img

    def _refresh_base(self):
        self._base_tr = QTransform()
        if self._bg_mode == "Raw movie" and self._stack is not None:
            j = min(self._cur, self._stack_len() - 1)
            self._base_img = self._frame_image(j)
        elif self._bg_mode == "Max projection" and self._stack is not None:
            if self._maxproj is None:
                self._maxproj = self._stack.max(axis=0)
            self._base_img = core._gray_qimage(self._maxproj,
                                               core._robust_levels(self._maxproj))
        elif self._bg_mode == "Super-resolution" and self._sr is not None:
            img, scale, (ty, tx) = self._sr
            self._base_img = core._lut_qimage(img, "inferno",
                                              core._robust_levels(img, 0, 99.5))
            tr = QTransform(); tr.translate(tx, ty); tr.scale(scale, scale)
            self._base_tr = tr
        else:
            self._base_img = None

    def background_options(self):
        opts = []
        if self._stack is not None:
            opts += ["Raw movie", "Max projection"]
        if self._sr is not None:
            opts.append("Super-resolution")
        opts.append("Off")
        return opts

    @property
    def background_mode(self):
        return self._bg_mode

    @background_mode.setter
    def background_mode(self, mode):
        if mode in self.background_options():
            self._bg_mode = mode
            self._refresh_base()
            self.update()

    def set_superres(self, img, *, scale=1.0, translate=(0.0, 0.0), cmap="inferno", opacity=0.9):
        self._sr = (np.asarray(img, float), float(scale),
                    (float(translate[0]), float(translate[1])))
        self._bg_mode = "Super-resolution"
        self._refresh_base()
        self.update()

    def clear_superres(self):
        self._sr = None
        if self._bg_mode == "Super-resolution":
            self._bg_mode = "Raw movie" if self._stack is not None else "Off"
        self._refresh_base()
        self.update()

    @property
    def has_superres(self):
        return self._sr is not None

    # ═════════════════════════════════════════════════════════════════════
    #  tracks
    # ═════════════════════════════════════════════════════════════════════
    def _add_class(self, cls, trajs, color, width):
        model = core.build_class_model(trajs)
        if model is None:
            return
        pen = QtGui.QPen(core._qcolor(color))
        pen.setWidthF(float(width))
        pen.setCosmetic(True)
        self._tail_recs[cls] = {"lines": model["lines"], "frames": model["frames"],
                                "lo": 0, "hi": 0, "pen": pen, "rect": model["rect"]}
        self._class_visible[cls] = True
        self._track_pick[cls] = (model["pick_xy"], model["pick_pid"])
        self._track_frames[cls] = model["pick_fr"]

    def set_tracks(self, tracks_by_class, colors, *, width=1.5):
        self.clear_tracks()
        for cls, trajs in tracks_by_class.items():
            self._add_class(cls, [(i, tr, None) for i, tr in enumerate(trajs)],
                            colors.get(cls, colors.get("Unknown", "#888")), self._width)
        self._rebuild_time_axis()
        self._apply_windows()
        self.update()

    def set_tracks_from_df(self, df, motion_map, colors, *, min_len=1, width=1.5):
        out = {}
        self.clear_tracks()
        by_class = core.df_to_class_trajs(df, motion_map, min_len)
        for cls in core._order_classes(by_class.keys()):
            trajs = by_class.get(cls)
            if not trajs:
                continue
            self._add_class(cls, trajs, colors.get(cls, colors.get("Unknown", "#888")),
                            self._width)
            out[cls] = set(int(pid) for pid, _, _ in trajs)
        self._max_track_frame = max(
            (int(fr.max()) for fr in self._track_frames.values() if len(fr)), default=-1)
        self._rebuild_time_axis()
        if self._stack is None and self._n_time > 1:
            self._cur = self._n_time - 1
        self._fitted = False
        self._apply_windows()
        self._refresh_heads()
        if self._has_content() and self.width() > 1:
            self.reset_view()
        self.update()
        self.frameChanged.emit(self._cur)
        return out

    def set_class_visible(self, cls, visible):
        if cls in self._tail_recs:
            self._class_visible[cls] = bool(visible)
            self._refresh_heads()
            self.update()

    def class_names(self):
        return list(self._tail_recs.keys())

    def clear_tracks(self):
        self._tail_recs.clear()
        self._track_pick.clear()
        self._class_visible.clear()
        self._track_frames.clear()
        self._max_track_frame = -1
        self._head_poly = QPolygonF()
        self._rebuild_time_axis()

    def recolor_tracks(self, colors, *, width=None):
        w = self._width if width is None else width
        for cls, rec in self._tail_recs.items():
            pen = QtGui.QPen(core._qcolor(colors.get(cls, colors.get("Unknown", "#888"))))
            pen.setWidthF(float(w))
            pen.setCosmetic(True)
            rec["pen"] = pen
        self.update()

    def _apply_windows(self):
        for rec in self._tail_recs.values():
            rec["lo"], rec["hi"] = core.tail_window(rec["frames"], self._cur,
                                                    self._tail, self._head)

    def _refresh_heads(self):
        xs, ys = core.head_xy_for_frame(self._track_pick, self._track_frames,
                                        self._class_visible, self._cur)
        self._head_poly = QPolygonF([QPointF(float(x), float(y))
                                     for x, y in zip(xs, ys)]) if len(xs) else QPolygonF()

    # ═════════════════════════════════════════════════════════════════════
    #  points
    # ═════════════════════════════════════════════════════════════════════
    def set_points(self, ys, xs, *, ids=None, brushes=None, size=4):
        self.clear_points()
        x = np.asarray(xs, float).ravel()
        y = np.asarray(ys, float).ravel()
        if x.size == 0:
            return
        if brushes is None:
            brushes = [QtGui.QColor(120, 180, 255, 200)] * x.size
        self._point_size = max(1, int(size))
        self._point_groups, _ = core.group_points_by_color(x, y, brushes, self._point_size)
        self._point_xy = np.column_stack([x, y])
        self._point_ids = (np.asarray(ids).ravel() if ids is not None else np.arange(x.size))
        self.update()

    def clear_points(self):
        self._point_groups = []
        self._point_xy = None
        self._point_ids = None
        self.update()

    # ═════════════════════════════════════════════════════════════════════
    #  transport
    # ═════════════════════════════════════════════════════════════════════
    def _rebuild_time_axis(self):
        n = max(self._stack_len(), self._max_track_frame + 1)
        if n != self._n_time:
            self._n_time = n
            self.nFramesChanged.emit()
        self._cur = max(0, min(self._cur, max(0, n - 1)))

    @property
    def n_frames(self):
        return int(self._n_time)

    @property
    def current_frame(self):
        return int(self._cur)

    @current_frame.setter
    def current_frame(self, i):
        if self._n_time <= 0:
            return
        i = int(max(0, min(self._n_time - 1, i)))
        if i == self._cur:
            return
        self._cur = i
        if self._bg_mode == "Raw movie":
            self._refresh_base()
        self._apply_windows()
        self._refresh_heads()
        self.update()
        self.frameChanged.emit(self._cur)

    def _advance(self):
        if self._n_time > 1:
            self.current_frame = (self._cur + 1) % self._n_time

    @property
    def tail(self):
        return self._tail

    @tail.setter
    def tail(self, v):
        self._tail = int(v); self._apply_windows(); self.update()

    @property
    def head(self):
        return self._head

    @head.setter
    def head(self, v):
        self._head = int(v); self._apply_windows(); self.update()

    @property
    def track_width(self):
        return self._width

    @track_width.setter
    def track_width(self, v):
        self._width = float(v)
        for rec in self._tail_recs.values():
            rec["pen"].setWidthF(self._width)
        self.update()

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, v):
        self._fps = int(v)
        if self._play_timer.isActive():
            self._play_timer.start(max(16, 1000 // self._fps))

    @property
    def playing(self):
        return self._playing

    @playing.setter
    def playing(self, on):
        on = bool(on) and self._n_time > 1
        if on == self._playing:
            return
        self._playing = on
        if on:
            self._play_timer.start(max(16, 1000 // self._fps))
        else:
            self._play_timer.stop()
        self.playingChanged.emit()

    # ═════════════════════════════════════════════════════════════════════
    #  camera
    # ═════════════════════════════════════════════════════════════════════
    def _has_content(self):
        return self._stack is not None or bool(self._tail_recs) or self._sr is not None

    def _content_rect(self) -> QRectF:
        if self._stack is not None:
            h, w = self._stack.shape[1], self._stack.shape[2]
            return QRectF(0, 0, w, h)
        if self._tail_recs:
            r = QRectF()
            for rec in self._tail_recs.values():
                r = rec["rect"] if r.isNull() else r.united(rec["rect"])
            return r
        if self._sr is not None:
            img, scale, (ty, tx) = self._sr
            h, w = np.asarray(img).shape[:2]
            return QRectF(tx, ty, w * scale, h * scale)
        return QRectF()

    def reset_view(self):
        r = self._content_rect()
        w, h = self.width(), self.height()
        if r.width() <= 0 or r.height() <= 0 or w < 1 or h < 1:
            return
        self._scale = min(w / r.width(), h / r.height()) * 0.98
        self._offx = w / 2 - (r.x() + r.width() / 2) * self._scale
        self._offy = h / 2 - (r.y() + r.height() / 2) * self._scale
        self._fitted = True
        self.update()

    def center_on(self, y, x, *, span=None):
        w, h = self.width(), self.height()
        if span:
            self._scale = min(w / span, h / span)
        self._offx = w / 2 - float(x) * self._scale
        self._offy = h / 2 - float(y) * self._scale
        self.update()

    def visible_rect(self) -> QRectF:
        if self._scale <= 1e-9:
            return QRectF()
        x = -self._offx / self._scale
        y = -self._offy / self._scale
        return QRectF(x, y, self.width() / self._scale, self.height() / self._scale)

    def _to_data(self, wx, wy):
        return (wy - self._offy) / self._scale, (wx - self._offx) / self._scale  # (y, x)

    def _data_tol(self, px=8.0):
        return float(px) / self._scale if self._scale > 1e-9 else 8.0

    def pick_at(self, y, x, *, tol=None):
        if tol is None:
            tol = self._data_tol()
        return core.pick_at(self._point_xy, self._point_ids, self._track_pick,
                            self._class_visible, y, x, tol)

    # ═════════════════════════════════════════════════════════════════════
    #  interaction
    # ═════════════════════════════════════════════════════════════════════
    def wheelEvent(self, ev):
        f = 1.25 if ev.angleDelta().y() > 0 else 1.0 / 1.25
        mx, my = ev.position().x(), ev.position().y()
        dx = (mx - self._offx) / self._scale
        dy = (my - self._offy) / self._scale
        self._scale *= f
        self._offx = mx - dx * self._scale
        self._offy = my - dy * self._scale
        ev.accept()
        self.update()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._panning = True
            self._press = ev.position()
            self._last = ev.position()
            self._moved = 0.0
            self.forceActiveFocus()
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._panning and self._last is not None:
            d = ev.position() - self._last
            self._last = ev.position()
            self._moved += abs(d.x()) + abs(d.y())
            self._offx += d.x()
            self._offy += d.y()
            ev.accept()
            self.update()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self._panning:
            self._panning = False
            if self._moved < 5.0 and self._press is not None:
                y, x = self._to_data(self._press.x(), self._press.y())
                hit = self.pick_at(y, x)
                if hit is not None:
                    kind, ident = hit
                    (self.clusterClicked if kind == "cluster" else self.trackClicked).emit(int(ident))
            ev.accept()
            return
        super().mouseReleaseEvent(ev)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key.Key_Space:
            self.playing = not self._playing
            ev.accept()
            return
        super().keyPressEvent(ev)
