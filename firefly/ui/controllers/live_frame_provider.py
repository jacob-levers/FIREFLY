"""LiveFrameProvider — feeds the QML Analysis cockpit's live detection view.

The worker streams ``PREVIEW_FRAME`` messages (a raw float32 frame blob + the
detection x/y for that frame).  AnalysisController renders each into a QImage via
:func:`render_frame` and bumps a token; the QML ``Image`` rebinds
``source: "image://liveframe/<token>"`` and this provider returns the current
image.  Mirrors the Widgets ``_LiveFrameView`` (robust-percentile grayscale with
accent detection markers), backend-independent so it works under the software
scene-graph used in headless tests.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtGui import QImage
from PySide6.QtQuick import QQuickImageProvider


def render_frame(arr, xs=None, ys=None,
                 marker_rgb=(88, 166, 255)) -> QImage:
    """Render a float32 frame to an RGB QImage with robust contrast + detection
    markers.  ``xs``/``ys`` are detection centres in pixel coords (optional)."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim != 2 or a.size == 0:
        return QImage()
    h, w = a.shape
    # Robust 1–99.5 percentile stretch → 8-bit, so a few hot pixels don't crush
    # the dynamic range (matches the viewer's _robust_levels intent).
    finite = a[np.isfinite(a)]
    if finite.size:
        lo, hi = np.percentile(finite, (1.0, 99.5))
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1.0
    g8 = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    g8 = (g8 * 255.0).astype(np.uint8)
    rgb = np.repeat(g8[:, :, None], 3, axis=2)        # HxWx3 grayscale

    # Detection markers: a 3x3 accent square at each (x, y).
    if xs is not None and ys is not None and len(xs):
        mr, mg, mb = marker_rgb
        xi = np.rint(np.asarray(xs, dtype=np.float64)).astype(int)
        yi = np.rint(np.asarray(ys, dtype=np.float64)).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        xi, yi = xi[ok], yi[ok]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cx = np.clip(xi + dx, 0, w - 1)
                cy = np.clip(yi + dy, 0, h - 1)
                rgb[cy, cx, 0] = mr
                rgb[cy, cx, 1] = mg
                rgb[cy, cx, 2] = mb

    rgb = np.ascontiguousarray(rgb)
    # Copy so the QImage owns its buffer (the numpy array goes out of scope).
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class LiveFrameProvider(QQuickImageProvider):
    """Serves AnalysisController's current live frame.  The image id is just a
    cache-busting token (the frame index); the actual image is whatever the
    controller last rendered."""

    def __init__(self, controller):
        super().__init__(QQuickImageProvider.ImageType.Image)
        self._ctrl = controller

    def requestImage(self, image_id, size, requested):  # noqa: N802 (Qt API)
        img = getattr(self._ctrl, "_live_image", None)
        if img is None or img.isNull():
            img = QImage(2, 2, QImage.Format.Format_RGB888)
            img.fill(0)
        if size is not None:
            size.setWidth(img.width())
            size.setHeight(img.height())
        return img
