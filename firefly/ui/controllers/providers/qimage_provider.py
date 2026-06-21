"""QImageProvider — serves a QImage returned by a getter callable to QML.

A tiny generic provider for in-memory images (the ROI max-projection
background).  The QML ``Image`` binds ``source: "image://<id>/<token>"`` where the
token cache-busts; this reads the current image from ``getter()`` each request.
"""
from __future__ import annotations

from PySide6.QtGui import QImage
from PySide6.QtQuick import QQuickImageProvider


class QImageProvider(QQuickImageProvider):
    def __init__(self, getter):
        super().__init__(QQuickImageProvider.ImageType.Image)
        self._getter = getter

    def requestImage(self, image_id, size, requested):  # noqa: N802 (Qt API)
        img = None
        try:
            img = self._getter()
        except Exception:
            img = None
        if img is None or img.isNull():
            img = QImage(2, 2, QImage.Format.Format_RGB888)
            img.fill(0)
        if size is not None:
            size.setWidth(img.width())
            size.setHeight(img.height())
        return img
