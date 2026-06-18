"""FireflyViewer — a pyqtgraph-based 2-D image/tracks/points viewer.

This is FIREFLY's in-house replacement for the embedded napari viewer.  It is a
plain ``QWidget`` (no napari, no vispy, no magicgui) wrapping a single
pyqtgraph ``ViewBox`` with:

* a raw image **stack** + a Qt frame slider (time scrubbing),
* per-motion-class **track** polylines with independent visibility,
* a **points** overlay (DBSCAN clusters) with click-to-pick,
* an additive **super-resolution** image overlay,
* camera helpers (``center_on`` / ``reset_view``).

Design notes
------------
* Coordinates are image-pixel ``(y, x)`` throughout (the same convention the
  analysis core and the old napari layers used).  The ViewBox plots ``x`` →
  horizontal, ``y`` → vertical with the **y-axis inverted** so row 0 is at the
  top, matching image display.  ``imageAxisOrder='row-major'`` makes
  ``ImageItem`` accept ``img[y, x]`` directly.
* Tracks are drawn as **complete polylines** (one pyqtgraph item per motion
  class).  We deliberately drop napari's animated head/tail-over-time
  behaviour: showing whole trajectories is clearer for sptPALM, and click
  resolution becomes a simple spatial nearest-vertex test.
* The widget owns its visibility state (``set_class_visible``) because, unlike
  napari, there is no built-in layer-list UI — the Visualise tab supplies the
  checkboxes.

The public surface is intentionally narrow and UI-policy-free so a future QML
phase can re-skin around it.
"""
from __future__ import annotations

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Signal

import pyqtgraph as pg

# Image arrays are indexed [y, x]; tell pyqtgraph so ImageItem maps them right.
pg.setConfigOption("imageAxisOrder", "row-major")
pg.setConfigOption("antialias", True)


def _qcolor(c) -> QtGui.QColor:
    """Coerce a hex string / QColor / RGBA tuple to a QColor."""
    if isinstance(c, QtGui.QColor):
        return c
    if isinstance(c, (tuple, list, np.ndarray)):
        arr = list(c)
        if len(arr) >= 3 and max(arr) <= 1.0:
            arr = [int(round(v * 255)) for v in arr]
        col = QtGui.QColor(*[int(v) for v in arr[:3]])
        if len(arr) >= 4:
            col.setAlpha(int(arr[3] if arr[3] > 1 else round(arr[3] * 255)))
        return col
    return QtGui.QColor(str(c))


def _robust_levels(arr: np.ndarray, lo=0.5, hi=99.8):
    """Percentile contrast limits, NaN-tolerant, with a safe fallback."""
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (0.0, 1.0)
    lo_v, hi_v = np.percentile(a, [lo, hi])
    if hi_v <= lo_v:
        hi_v = lo_v + 1.0
    return (float(lo_v), float(hi_v))


class FireflyViewer(QtWidgets.QWidget):
    """A self-contained pyqtgraph viewer.  See the module docstring."""

    trackClicked = Signal(int)        # particle id of the nearest track
    clusterClicked = Signal(int)      # cluster id of the nearest point (-1 = noise)
    frameChanged = Signal(int)        # current stack frame index

    def __init__(self, parent=None, *, background="#0d1117"):
        super().__init__(parent)

        self._stack: np.ndarray | None = None      # (T, Y, X)
        self._levels = (0.0, 1.0)
        self._track_items: dict[str, pg.PlotCurveItem] = {}
        self._track_pick: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._class_visible: dict[str, bool] = {}
        self._point_item: pg.ScatterPlotItem | None = None
        self._point_xy: np.ndarray | None = None    # (N, 2) as (x, y)
        self._point_ids: np.ndarray | None = None
        self._sr_item: pg.ImageItem | None = None

        # ── graphics ──────────────────────────────────────────────────────
        self._glw = pg.GraphicsLayoutWidget()
        self._glw.setBackground(background)
        self._plot = self._glw.addPlot()
        self._plot.hideAxis("left")
        self._plot.hideAxis("bottom")
        self._vb = self._plot.getViewBox()
        self._vb.setAspectLocked(True)
        self._vb.invertY(True)
        self._vb.setMenuEnabled(False)

        self._img_item = pg.ImageItem()
        self._vb.addItem(self._img_item)

        # ── frame slider ──────────────────────────────────────────────────
        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider)
        self._frame_lbl = QtWidgets.QLabel("—")
        self._frame_lbl.setMinimumWidth(72)
        self._frame_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 2)
        bar.addWidget(QtWidgets.QLabel("Frame"))
        bar.addWidget(self._slider, 1)
        bar.addWidget(self._frame_lbl)
        self._bar_widget = QtWidgets.QWidget()
        self._bar_widget.setLayout(bar)
        self._bar_widget.setVisible(False)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._glw, 1)
        lay.addWidget(self._bar_widget)

        self._plot.scene().sigMouseClicked.connect(self._on_scene_clicked)

    # ─────────────────────────────────────────────────────────────────────
    # Image stack
    # ─────────────────────────────────────────────────────────────────────
    def set_stack(self, stack, *, cmap="gray", autorange=True):
        """Display a ``(T, Y, X)`` (or ``(Y, X)``) array as the base image."""
        arr = np.asarray(stack)
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim != 3:
            raise ValueError(f"stack must be 2-D or 3-D, got {arr.ndim}-D")
        self._stack = arr
        # Global contrast from a mid frame keeps brightness stable while scrubbing.
        self._levels = _robust_levels(arr[arr.shape[0] // 2])
        try:
            self._img_item.setLookupTable(
                pg.colormap.get(cmap).getLookupTable(0.0, 1.0, 256)
                if cmap != "gray" else None)
        except Exception:
            pass
        n = arr.shape[0]
        self._slider.blockSignals(True)
        self._slider.setMaximum(max(0, n - 1))
        self._slider.setValue(0)
        self._slider.setEnabled(n > 1)
        self._slider.blockSignals(False)
        self._bar_widget.setVisible(n > 1)
        self._show_frame(0)
        if autorange:
            self.reset_view()

    def clear_stack(self):
        self._stack = None
        self._img_item.clear()
        self._slider.setMaximum(0)
        self._bar_widget.setVisible(False)

    @property
    def n_frames(self) -> int:
        return 0 if self._stack is None else int(self._stack.shape[0])

    @property
    def current_frame(self) -> int:
        return int(self._slider.value())

    @current_frame.setter
    def current_frame(self, i: int):
        if self._stack is None:
            return
        i = int(max(0, min(self.n_frames - 1, i)))
        self._slider.setValue(i)   # triggers _on_slider → _show_frame

    def _on_slider(self, i: int):
        self._show_frame(int(i))
        self.frameChanged.emit(int(i))

    def _show_frame(self, i: int):
        if self._stack is None:
            return
        i = int(max(0, min(self.n_frames - 1, i)))
        self._img_item.setImage(self._stack[i], autoLevels=False,
                                levels=self._levels)
        self._frame_lbl.setText(f"{i + 1}/{self.n_frames}")

    # ─────────────────────────────────────────────────────────────────────
    # Tracks (one polyline item per motion class)
    # ─────────────────────────────────────────────────────────────────────
    def set_tracks(self, tracks_by_class: dict, colors: dict, *, width=1.5):
        """Render per-class trajectories.

        ``tracks_by_class``: ``{class_name: [traj, ...]}`` where each ``traj``
        is an ``(N, 2)`` array of ``(y, x)`` vertices (a single trajectory).
        ``colors``: ``{class_name: hex/QColor}``.
        """
        self.clear_tracks()
        for cls, trajs in tracks_by_class.items():
            xs_parts, ys_parts, connect_parts = [], [], []
            pick_xy, pick_pid = [], []
            for idx, traj in enumerate(trajs):
                t = np.asarray(traj, dtype=float)
                if t.ndim != 2 or t.shape[0] < 2:
                    continue
                y = t[:, 0]
                x = t[:, 1]
                xs_parts.append(x)
                ys_parts.append(y)
                con = np.ones(len(x), dtype=np.int32)
                con[-1] = 0          # break between trajectories
                connect_parts.append(con)
                pick_xy.append(np.column_stack([x, y]))
                # pid is carried on the traj via a parallel id list, see below
                pick_pid.append(np.full(len(x), idx, dtype=np.int64))
            if not xs_parts:
                continue
            xs = np.concatenate(xs_parts)
            ys = np.concatenate(ys_parts)
            con = np.concatenate(connect_parts)
            pen = pg.mkPen(_qcolor(colors.get(cls, "#888888")), width=width)
            item = pg.PlotCurveItem(x=xs, y=ys, connect=con, pen=pen)
            item.setCompositionMode(
                QtGui.QPainter.CompositionMode.CompositionMode_Plus)
            self._vb.addItem(item)
            self._track_items[cls] = item
            self._class_visible[cls] = True
            self._track_pick[cls] = (np.concatenate(pick_xy),
                                     np.concatenate(pick_pid))

    def set_tracks_from_df(self, df, motion_map: dict, colors: dict, *,
                           min_len: int = 1, width=1.5):
        """Convenience: build per-class trajectories straight from a tracks
        DataFrame (columns particle/frame/x/y) + a ``{particle: motion}`` map.

        Returns ``{class: set(particle_ids)}`` for the caller's bookkeeping.
        The pick arrays carry the **real particle id** so clicks resolve to it.
        """
        out_pids: dict[str, set] = {}
        self.clear_tracks()
        if df is None or not len(df):
            return out_pids
        lens = df.groupby("particle").size()
        for cls in _iter_classes(motion_map, df):
            pids = [p for p in df["particle"].unique()
                    if motion_map.get(int(p), "Unknown") == cls
                    and int(lens.get(p, 0)) >= max(2, min_len)]
            if not pids:
                continue
            xs_parts, ys_parts, con_parts, pxy, ppid = [], [], [], [], []
            sub = df[df["particle"].isin(pids)].sort_values(
                ["particle", "frame"], kind="mergesort")
            for pid, g in sub.groupby("particle"):
                x = g["x"].to_numpy(float)
                y = g["y"].to_numpy(float)
                if len(x) < 2:
                    continue
                xs_parts.append(x); ys_parts.append(y)
                con = np.ones(len(x), np.int32); con[-1] = 0
                con_parts.append(con)
                pxy.append(np.column_stack([x, y]))
                ppid.append(np.full(len(x), int(pid), np.int64))
            if not xs_parts:
                continue
            pen = pg.mkPen(_qcolor(colors.get(cls, colors.get("Unknown", "#888"))),
                           width=width)
            item = pg.PlotCurveItem(x=np.concatenate(xs_parts),
                                    y=np.concatenate(ys_parts),
                                    connect=np.concatenate(con_parts), pen=pen)
            item.setCompositionMode(
                QtGui.QPainter.CompositionMode.CompositionMode_Plus)
            self._vb.addItem(item)
            self._track_items[cls] = item
            self._class_visible[cls] = True
            self._track_pick[cls] = (np.concatenate(pxy), np.concatenate(ppid))
            out_pids[cls] = set(int(p) for p in pids)
        return out_pids

    def set_class_visible(self, cls: str, visible: bool):
        item = self._track_items.get(cls)
        if item is not None:
            item.setVisible(bool(visible))
            self._class_visible[cls] = bool(visible)

    def class_names(self) -> list:
        return list(self._track_items.keys())

    def clear_tracks(self):
        for item in self._track_items.values():
            try:
                self._vb.removeItem(item)
            except Exception:
                pass
        self._track_items.clear()
        self._track_pick.clear()
        self._class_visible.clear()

    def recolor_tracks(self, colors: dict, *, width=1.5):
        """Recolour existing per-class items in place (no rebuild)."""
        for cls, item in self._track_items.items():
            item.setPen(pg.mkPen(
                _qcolor(colors.get(cls, colors.get("Unknown", "#888"))),
                width=width))

    # ─────────────────────────────────────────────────────────────────────
    # Points (DBSCAN clusters)
    # ─────────────────────────────────────────────────────────────────────
    def set_points(self, ys, xs, *, ids=None, brushes=None, size=4):
        """Scatter overlay.  ``ys``/``xs`` are pixel coords; ``brushes`` a
        per-point colour list (hex/QColor/RGBA); ``ids`` parallels the points
        for click resolution."""
        self.clear_points()
        x = np.asarray(xs, float).ravel()
        y = np.asarray(ys, float).ravel()
        if x.size == 0:
            return
        if brushes is None:
            spot_brushes = pg.mkBrush(120, 180, 255, 200)
        else:
            spot_brushes = [pg.mkBrush(_qcolor(b)) for b in brushes]
        self._point_item = pg.ScatterPlotItem(
            x=x, y=y, size=size, pen=None, brush=spot_brushes)
        self._vb.addItem(self._point_item)
        self._point_xy = np.column_stack([x, y])
        self._point_ids = (np.asarray(ids).ravel() if ids is not None
                           else np.arange(x.size))

    def clear_points(self):
        if self._point_item is not None:
            try:
                self._vb.removeItem(self._point_item)
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
        """Add a super-res image as an additive overlay.

        ``scale`` is data-units (raw px) per super-res px; ``translate`` is the
        ``(ty, tx)`` offset of the super-res origin in raw pixels.
        """
        self.clear_superres()
        arr = np.asarray(img, float)
        item = pg.ImageItem(arr)
        try:
            item.setLookupTable(pg.colormap.get(cmap).getLookupTable(0, 1, 256))
        except Exception:
            pass
        lo, hi = _robust_levels(arr, 0, 99.5)
        item.setLevels((lo, hi))
        item.setOpacity(float(opacity))
        item.setCompositionMode(
            QtGui.QPainter.CompositionMode.CompositionMode_Plus)
        tr = QtGui.QTransform()
        ty, tx = translate
        tr.translate(float(tx), float(ty))
        tr.scale(float(scale), float(scale))
        item.setTransform(tr)
        self._vb.addItem(item)
        self._sr_item = item

    def clear_superres(self):
        if self._sr_item is not None:
            try:
                self._vb.removeItem(self._sr_item)
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
        """Recentre on ``(y, x)`` keeping the current zoom span (or ``span``)."""
        (xr, yr) = self._vb.viewRange()
        w = float(span) if span else (xr[1] - xr[0])
        h = float(span) if span else (yr[1] - yr[0])
        self._vb.setRange(xRange=(x - w / 2, x + w / 2),
                          yRange=(y - h / 2, y + h / 2), padding=0)

    def reset_view(self):
        self._vb.autoRange()

    # ─────────────────────────────────────────────────────────────────────
    # Click resolution (points take priority over tracks)
    # ─────────────────────────────────────────────────────────────────────
    def _data_tol(self, px=8.0) -> float:
        try:
            psx, psy = self._vb.viewPixelSize()
            return float(px) * max(abs(psx), abs(psy))
        except Exception:
            return 8.0

    def pick_at(self, y, x, *, tol=None):
        """Resolve a data-space click at ``(y, x)``.

        Returns ``("cluster", id)`` if a points overlay is shown and the
        nearest point is within ``tol`` (points take priority), else
        ``("track", particle_id)`` for the nearest visible track vertex within
        ``tol``, else ``None``.  ``tol`` defaults to ~8 screen px in data units.
        """
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

    def _on_scene_clicked(self, ev):
        try:
            if ev.button() != QtCore.Qt.MouseButton.LeftButton or ev.double():
                return
            pt = self._vb.mapSceneToView(ev.scenePos())
            hit = self.pick_at(pt.y(), pt.x())
        except Exception:
            return
        if hit is None:
            return
        kind, ident = hit
        if kind == "cluster":
            self.clusterClicked.emit(int(ident))
        else:
            self.trackClicked.emit(int(ident))


def _iter_classes(motion_map: dict, df) -> list:
    """Motion-class iteration order: known order first, then any extras seen."""
    order = ["Immobile", "Confined", "Brownian", "Directed", "Unknown"]
    seen = set(str(m) for m in motion_map.values()) or {"Unknown"}
    extras = [c for c in seen if c not in order]
    return [c for c in order if c in seen or c == "Unknown"] + extras
