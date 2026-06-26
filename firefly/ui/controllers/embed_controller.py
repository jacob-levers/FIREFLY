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
from PySide6.QtGui import QPainterPath, QRegion

from firefly.ui.controllers.app_controller import TABS

# Corner radius for the rounded viewer island (matched by the HUD border + the
# QML panel cards so the viewer reads as a floating widget).
_VIEWER_RADIUS = 14


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
        self._on_visualise = False     # currently on the Visualise tab/page
        self._viewer_content = False   # viewer actually has data to draw

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
            if self._active == "viewer":
                self._round_viewer(w)
        # The HUD overlay tracks the viewer island exactly (only over the
        # viewer, not the ROI editor).
        if self._hud is not None and self._active == "viewer" and not self._rect.isEmpty():
            self._hud.setGeometry(self._rect)
            self._hud.raise_()

    def _round_viewer(self, w):
        """Mask the viewer island to a rounded rect so it composites as a
        floating card (the HUD draws the matching border at the same rect)."""
        try:
            r = self._rect
            if r.isEmpty():
                w.clearMask()
                return
            path = QPainterPath()
            path.addRoundedRect(0.0, 0.0, float(r.width()), float(r.height()),
                                float(_VIEWER_RADIUS), float(_VIEWER_RADIUS))
            w.setMask(QRegion(path.toFillPolygon().toPolygon()))
        except Exception:
            pass

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
                if name == "viewer":
                    self._round_viewer(w)
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
        isn't occluded; restore on close — but only to where it BELONGS.  A blind
        re-show let the Visualise viewer 'escape' onto whatever tab you'd switched
        to before closing the modal (e.g. open Preferences on Visualise, close it
        on Import → the viewer reappeared over Import)."""
        self._modal = bool(on)
        if self._modal:
            self.hideIslands()
        elif self._active == "viewer":
            self._sync_viewer_visibility()   # tab + content gated → no escape
        elif self._active == "roi":
            self.showIsland("roi")           # ROI editor is its own modal overlay

    # ── tab / page visibility ────────────────────────────────────────────
    @Slot(int, str)
    def onLocationChanged(self, tab: int, page: str):
        """Track whether we're on the Visualise tab of the main page.  The viewer
        island shows only there AND only once it has content (see
        :meth:`setViewerContent`) — an empty Visualise tab shows the placeholder
        rather than a blank floating card."""
        self._on_visualise = (page == "main" and tab == VISUALISE_TAB)
        self._sync_viewer_visibility()

    @Slot(bool)
    def setViewerContent(self, on: bool):
        """Mark whether the viewer has something to draw (a run / tracks / stack)
        so the island isn't shown as an empty card before anything is loaded."""
        self._viewer_content = bool(on)
        self._sync_viewer_visibility()

    def _sync_viewer_visibility(self):
        if self._on_visualise and self._viewer_content:
            self.showIsland("viewer" if self._active in ("none", "viewer") else self._active)
        else:
            self.hideIslands()


# Derived from the single source of truth so a tab reorder/rename can't silently
# strand the island (it was hardcoded to 4 and broke when Compare+Results merged
# into "Analysis", shifting Visualise from index 4 → 3).
try:
    VISUALISE_TAB = TABS.index("Visualise")
except ValueError:  # pragma: no cover — tab set changed unexpectedly
    VISUALISE_TAB = 3
