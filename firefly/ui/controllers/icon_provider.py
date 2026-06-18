"""IconImageProvider — themed Lucide icons as raster images for QML.

QML asks for ``image://icon/<name>/<rrggbb>`` and gets the Lucide stroke SVG
rendered + tinted to that colour at the requested size. Raster output works in
every Qt scene-graph backend (incl. the software backend used in headless
tests), unlike shader effects (ColorOverlay/MultiEffect). Results are cached.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QByteArray, QSize
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtQuick import QQuickImageProvider
from PySide6.QtSvg import QSvgRenderer


class IconImageProvider(QQuickImageProvider):
    def __init__(self, icons_dir: str):
        super().__init__(QQuickImageProvider.ImageType.Image)
        self._dir = icons_dir
        self._cache: dict = {}

    def requestImage(self, image_id: str, size: QSize, requested: QSize) -> QImage:
        name, _, color = image_id.partition("/")
        w = requested.width() if requested.width() > 0 else 48
        h = requested.height() if requested.height() > 0 else 48
        key = (name, color, w, h)
        cached = self._cache.get(key)
        if cached is not None:
            size.setWidth(cached.width())
            size.setHeight(cached.height())
            return cached

        img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(0)
        path = os.path.join(self._dir, name + ".svg")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                svg = fh.read()
            tint = QColor("#" + color.lstrip("#")) if color else QColor("#e6edf3")
            hexrgb = tint.name()                       # "#rrggbb" (drops alpha)
            svg = svg.replace("currentColor", hexrgb)
            renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
            p = QPainter(img)
            renderer.render(p)
            p.end()
        except Exception:
            pass   # missing/bad icon → transparent square (graceful)

        self._cache[key] = img
        size.setWidth(w)
        size.setHeight(h)
        return img
