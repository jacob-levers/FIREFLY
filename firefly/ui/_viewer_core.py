"""Pure (scene-graph-free) rendering + picking logic for the FIREFLY viewer.

Extracted from ``viewer.py`` so the native ``FireflyViewer`` (QGraphicsView) AND
the in-scene ``FireflyPaintedItem`` (QQuickPaintedItem, Phase 8) share ONE
implementation — no behaviour fork.  Everything here is numpy + Qt primitives
(QImage / QColor / QLineF / QPolygonF); nothing touches a QGraphicsScene/View.
Coordinates are image-pixel ``(y, x)`` throughout.
"""
from __future__ import annotations

import numpy as np

from PySide6 import QtGui
from PySide6.QtCore import QPointF, QRectF, QLineF


# ── colour + image compositors ───────────────────────────────────────────────
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
    img = QtGui.QImage(u8.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
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
    img = QtGui.QImage(rgba.data, w, h, 4 * w, QtGui.QImage.Format.Format_RGBA8888)
    return img.copy()


def _order_classes(present) -> list:
    """Known motion order first, then any extras seen in the data."""
    order = ["Immobile", "Confined", "Brownian", "Directed", "Unknown"]
    present = list(present)
    return ([c for c in order if c in present]
            + [c for c in present if c not in order])


# ── track model build + tail window ──────────────────────────────────────────
def build_class_model(trajs):
    """Build the per-class render+pick model from ``trajs`` =
    ``list[(pid, yx_array, frames_or_None)]``.  Returns a dict
    ``{lines, frames, pick_xy, pick_pid, pick_fr, rect}`` (lines pre-sorted by
    frame so a contiguous ``tail_window`` slice is the visible tail), or None if
    no segments.  Identical to the old ``_add_track_item`` body."""
    lines, seg_frames = [], []
    pick_xy, pick_pid, pick_fr = [], [], []
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    for pid, traj, frames in trajs:
        t = np.asarray(traj, dtype=float)
        if t.ndim != 2 or t.shape[0] < 2:
            continue
        y, x = t[:, 0], t[:, 1]
        fr = (np.arange(len(x)) if frames is None else np.asarray(frames))
        for k in range(1, len(x)):
            lines.append(QLineF(float(x[k - 1]), float(y[k - 1]),
                                float(x[k]), float(y[k])))
            seg_frames.append(int(fr[k]))
        pick_xy.append(np.column_stack([x, y]))
        pick_pid.append(np.full(len(x), int(pid), np.int64))
        pick_fr.append(fr)
        xmin = min(xmin, float(x.min())); xmax = max(xmax, float(x.max()))
        ymin = min(ymin, float(y.min())); ymax = max(ymax, float(y.max()))
    if not lines:
        return None
    seg_frames = np.asarray(seg_frames)
    order = np.argsort(seg_frames, kind="mergesort")
    lines = [lines[i] for i in order]
    seg_frames = seg_frames[order]
    pad = 4.0
    rect = QRectF(xmin - pad, ymin - pad,
                  (xmax - xmin) + 2 * pad, (ymax - ymin) + 2 * pad)
    return {"lines": lines, "frames": seg_frames,
            "pick_xy": np.concatenate(pick_xy),
            "pick_pid": np.concatenate(pick_pid),
            "pick_fr": np.concatenate(pick_fr), "rect": rect}


def tail_window(frames, t, tail, head=0):
    """``(lo, hi)`` index slice of frame-sorted ``frames`` for the visible tail
    ``(t - tail, t + head]`` — the O(log n) searchsorted window."""
    f = frames
    lo = 0 if (tail is None or tail <= 0) else int(np.searchsorted(f, t - tail + 1, side="left"))
    hi = int(np.searchsorted(f, t + int(head) + 1, side="left"))
    return lo, hi


def df_to_class_trajs(df, motion_map, min_len=1):
    """Group a tracks DataFrame (particle/frame/x/y) + ``{particle: motion}`` into
    ``{class: [(pid, yx_array, frames), …]}`` (filtered by ``min_len``)."""
    by_class: dict = {}
    if df is None or not len(df):
        return by_class
    lens = df.groupby("particle").size()
    sub = df.sort_values(["particle", "frame"], kind="mergesort")
    for pid, g in sub.groupby("particle"):
        if int(lens.get(pid, 0)) < max(2, min_len):
            continue
        cls = str(motion_map.get(int(pid), "Unknown"))
        by_class.setdefault(cls, []).append(
            (int(pid), np.column_stack([g["y"].to_numpy(float),
                                        g["x"].to_numpy(float)]),
             g["frame"].to_numpy()))
    return by_class


# ── points + heads ───────────────────────────────────────────────────────────
def group_points_by_color(xs, ys, brushes, size):
    """Bucket points by colour so each colour is ONE drawPoints call.  Returns
    ``([(QColor, QPolygonF), …], bbox_QRectF)``."""
    groups: dict = {}
    for x, y, b in zip(xs, ys, brushes):
        col = _qcolor(b)
        key = col.rgba()
        groups.setdefault(key, [QtGui.QColor.fromRgba(key), []])[1].append(
            QPointF(float(x), float(y)))
    out = [(col, QtGui.QPolygonF(pts)) for col, pts in groups.values()]
    if len(xs):
        pad = max(1, int(size))
        rect = QRectF(float(np.min(xs)) - pad, float(np.min(ys)) - pad,
                      float(np.ptp(xs)) + 2 * pad, float(np.ptp(ys)) + 2 * pad)
    else:
        rect = QRectF()
    return out, rect


def head_xy_for_frame(track_pick, track_frames, class_visible, t):
    """The (xs, ys) of every visible track's vertex AT frame ``t`` — the bright
    current-frame markers."""
    xs, ys = [], []
    for cls, (xy, _pid) in track_pick.items():
        if not class_visible.get(cls, True):
            continue
        fr = track_frames.get(cls)
        if fr is None or not len(fr):
            continue
        m = fr == t
        if m.any():
            xs.append(xy[m, 0])
            ys.append(xy[m, 1])
    if xs:
        return np.concatenate(xs), np.concatenate(ys)
    return [], []


# ── picking (points take priority over tracks) ───────────────────────────────
def pick_at(point_xy, point_ids, track_pick, class_visible, y, x, tol):
    """Resolve a data-space click. ``("cluster", id)`` (points priority) or
    ``("track", pid)`` for the nearest visible match within ``tol``, else None."""
    px, py = float(x), float(y)
    if point_xy is not None and len(point_xy):
        d = np.hypot(point_xy[:, 0] - px, point_xy[:, 1] - py)
        j = int(np.argmin(d))
        if d[j] <= tol:
            return ("cluster", int(point_ids[j]))
    best_pid, best_d = None, np.inf
    for cls, (xy, pid) in track_pick.items():
        if not class_visible.get(cls, True) or not len(xy):
            continue
        d = np.hypot(xy[:, 0] - px, xy[:, 1] - py)
        j = int(np.argmin(d))
        if d[j] < best_d:
            best_d, best_pid = d[j], int(pid[j])
    if best_pid is not None and best_d <= tol:
        return ("track", best_pid)
    return None
