"""Image provider for the Analysis gallery (publication-figure panels).

Serves two kinds of request from one provider:

* ``image://workspacepanel/hero/<token>`` — the large, selected panel
  (rendered asynchronously by the controller and stashed in ``panel_image``).
* ``image://workspacepanel/thumb/<cond>/<panel>/<rev>`` — a small thumbnail of
  panel ``<panel>`` for condition ``<cond>``, rendered on demand (and cached) by
  ``controller.render_panel_for``.  The QML ``Image`` uses ``asynchronous: true``
  so the 17 thumbnails render off the GUI thread and pop in as they finish.
"""
from __future__ import annotations

from PySide6.QtGui import QImage
from PySide6.QtQuick import QQuickImageProvider


class WorkspacePanelProvider(QQuickImageProvider):
    def __init__(self, controller):
        super().__init__(QQuickImageProvider.ImageType.Image)
        self._ctrl = controller

    def requestImage(self, image_id, size, requested):  # noqa: N802 (Qt API)
        img = None
        try:
            parts = image_id.split("/")
            if parts and parts[0] == "thumb" and len(parts) >= 3:
                cond = int(parts[1])
                panel = int(parts[2])
                w = requested.width() if requested.width() > 0 else 130
                h = requested.height() if requested.height() > 0 else 66
                img = self._ctrl.render_panel_for(cond, panel, w, h)
            else:                                   # "hero/<token>" (or anything else)
                img = self._ctrl.panel_image()
        except Exception:
            img = None
        if img is None or img.isNull():
            img = QImage(2, 2, QImage.Format.Format_RGB888)
            img.fill(0)
        if size is not None:
            size.setWidth(img.width())
            size.setHeight(img.height())
        return img
