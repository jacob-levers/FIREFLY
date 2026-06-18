"""EmbedController — hosts the native viewer/ROI island under the QML chrome.

Option-B composition (validated by the Phase-0 spike): the app is Widget-rooted
(a ``stage`` QWidget holds the chrome QQuickWidget); the bespoke FireflyViewer /
RoiEditor are NATIVE QWidget siblings of the chrome, and a native sibling always
composites ABOVE a QQuickWidget's texture on macOS/Qt 6 — which is exactly how the
viewer shows through an invisible QML placeholder ("anchor") with no QML
cooperation.  This controller is the bridge:

  * QML's ``viewerAnchor`` Item pushes its scene rect to :meth:`setAnchorRect`
    (coalesced), and we ``setGeometry`` the active island to match + emit
    :attr:`anchorChanged` so a transparent HUD overlay can re-anchor.
  * exactly one island is shown at a time (``activeIsland`` ∈ none|viewer|roi),
  * a modal-open bridge hides the always-on-top island so QML dialogs/popups
    aren't occluded,
  * tab/page changes show/hide the island (deferred, since a freshly-laid-out
    anchor reads (0,0,0,0) until the QQuickWidget lays out).

Geometry is in logical px on both sides (mapToItem ↔ setGeometry), which match on
Retina with the chrome pinned at the stage origin — no devicePixelRatio maths.
"""
from __future__ import annotations

from PySide6.QtCore import Property, QObject, QRect, QRectF, QTimer, Signal, Slot


class EmbedController(QObject):
    activeIslandChanged = Signal()
    anchorChanged = Signal(QRectF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._viewer = None
        self._roi = None
        self._hud = None               # transparent HUD QQuickWidget (L3)
        self._active = "none"          # none | viewer | roi
        self._modal = False
        self._rect = QRect()           # last-applied geometry (dedupe)
        self._anchor = QRectF()        # last anchor rect (logical px)
        self._viewer_shown = False     # first-show reset_view guard

    def setIslands(self, viewer=None, roi=None, hud=None):
        """Register the native island widgets + the transparent HUD overlay
        (called once from app_qml after the controllers are built)."""
        if viewer is not None:
            self._viewer = viewer
        if roi is not None:
            self._roi = roi
        if hud is not None:
            self._hud = hud

    def _widget(self, name=None):
        name = name or self._active
        return self._viewer if name == "viewer" else self._roi if name == "roi" else None

    # ── geometry sync ────────────────────────────────────────────────────
    @Slot(float, float, float, float)
    def setAnchorRect(self, x: float, y: float, w: float, h: float):
        self._anchor = QRectF(x, y, w, h)
        rect = QRect(round(x), round(y), round(w), round(h))
        if rect == self._rect:
            return                      # dedupe identical rects → no setGeometry churn
        self._rect = rect
        self._apply_geometry()
        self.anchorChanged.emit(self._anchor)

    def _apply_geometry(self):
        if self._modal or self._active == "none":
            return
        w = self._widget()
        if w is not None and not self._rect.isEmpty():
            w.setGeometry(self._rect)
        # The HUD overlay tracks the viewer island exactly (only over the
        # viewer, not the ROI editor).
        if self._hud is not None and self._active == "viewer" and not self._rect.isEmpty():
            self._hud.setGeometry(self._rect)
            self._hud.raise_()

    @Property("QRectF", notify=anchorChanged)
    def anchorRect(self):
        return self._anchor

    # ── single-island management ─────────────────────────────────────────
    @Property(str, notify=activeIslandChanged)
    def activeIsland(self):
        return self._active

    @Slot(str)
    def showIsland(self, name: str):
        if name not in ("none", "viewer", "roi"):
            return
        self._active = name
        # hide the inactive island, show + raise the active one
        for n in ("viewer", "roi"):
            w = self._widget(n)
            if w is None:
                continue
            if n == name and not self._modal:
                w.setGeometry(self._rect)
                w.show()
                w.raise_()
                try:    w.setFocus()
                except Exception: pass
            else:
                w.hide()
        # HUD overlays the viewer only.
        if self._hud is not None:
            if name == "viewer" and not self._modal:
                self._hud.setGeometry(self._rect)
                self._hud.show()
                self._hud.raise_()
            else:
                self._hud.hide()
        # First time the viewer is shown at a real size, re-fit (fitInView is a
        # no-op before the widget is laid out, so defer one tick).
        if name == "viewer" and self._viewer is not None and not self._viewer_shown:
            self._viewer_shown = True
            QTimer.singleShot(0, self._reset_viewer)
        self.activeIslandChanged.emit()

    def _reset_viewer(self):
        try:    self._viewer.reset_view()
        except Exception: pass

    @Slot()
    def hideIslands(self):
        for n in ("viewer", "roi"):
            w = self._widget(n)
            if w is not None:
                w.hide()
        if self._hud is not None:
            self._hud.hide()

    @Slot(bool)
    def setModalOpen(self, on: bool):
        """Hide the always-on-top island while a QML modal/popup is up so it
        isn't occluded; restore on close."""
        self._modal = bool(on)
        if self._modal:
            self.hideIslands()
        elif self._active != "none":
            w = self._widget()
            if w is not None:
                w.setGeometry(self._rect)
                w.show()
                w.raise_()
            if self._hud is not None and self._active == "viewer":
                self._hud.setGeometry(self._rect)
                self._hud.show()
                self._hud.raise_()

    # ── tab / page visibility ────────────────────────────────────────────
    @Slot(int, str)
    def onLocationChanged(self, tab: int, page: str):
        """Show the viewer island only on the Visualise tab of the main page;
        hide it everywhere else (synchronously, to avoid island bleed during a
        tab transition)."""
        if page == "main" and tab == VISUALISE_TAB:
            self.showIsland("viewer" if self._active == "none" else self._active)
        else:
            self.hideIslands()


VISUALISE_TAB = 4
