"""FigureImageProvider — serves a controller's current figure PNG to QML.

The Results / Compare figures are static PNGs on disk.  A direct ``file://`` URL
would be cached by QML across re-runs that reuse the same path (stem), so the
figure wouldn't refresh.  Instead the controller exposes a cache-busting token
property; the QML ``Image`` binds ``source: "image://<id>/<token>"`` and this
provider reads the controller's current ``figure_path`` off disk each request —
exactly the pattern LiveFrameProvider uses for the Analysis live frame.
"""
from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtGui import QImage
from PySide6.QtQuick import QQuickImageProvider


class FigureImageProvider(QQuickImageProvider):
    def __init__(self, controller):
        super().__init__(QQuickImageProvider.ImageType.Image)
        self._ctrl = controller

    def requestImage(self, image_id, size, requested):  # noqa: N802 (Qt API)
        path = ""
        try:
            path = self._ctrl.figure_path() or ""
        except Exception:
            path = ""
        img = QImage(path) if path else QImage()
        if img.isNull():
            img = QImage(2, 2, QImage.Format.Format_RGB888)
            img.fill(0)
        if size is not None:
            size.setWidth(img.width())
            size.setHeight(img.height())
        return img
