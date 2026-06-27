"""AppController — top-level navigation state for the QML shell.

Owns which page is showing (landing vs the main tabbed UI) and the active tab.
The QML header/tab-bar binds to these; tiles/pills call the slots.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Property, Signal, Slot, Qt
from PySide6.QtGui import QGuiApplication

# Tab order mirrors the Widgets app (ui_constants TAB_*); HYPER-FLY is the live
# parallel-batch dashboard (populated only during a HYPER-FLY batch run).
TABS = ["Import", "Process", "Analysis", "Visualise", "HYPER-FLY"]


class AppController(QObject):
    pageChanged = Signal()
    tabChanged = Signal()
    appActiveChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._page = "landing"      # "landing" | "main"
        self._tab = 0
        # Whether FIREFLY is the frontmost application.  The landing folds this
        # into its animation gate so the field pauses while the app is in the
        # background (battery).  Default True and let the signal drive only the
        # pause/resume transitions — the controller is built before the window
        # is shown, so reading applicationState() here would read Inactive and
        # start the field dead.  Connecting a *bound-method* slot ties the
        # connection to this QObject, so Qt auto-disconnects it on teardown —
        # a late state change can never call into a half-destroyed controller.
        self._app_active = True
        app = QGuiApplication.instance()
        if app is not None:
            app.applicationStateChanged.connect(self._on_app_state)

    @Property(str, notify=pageChanged)
    def page(self):
        return self._page

    @Property(int, notify=tabChanged)
    def currentTab(self):
        return self._tab

    @Property(bool, notify=appActiveChanged)
    def appActive(self):
        return self._app_active

    @Property("QStringList", constant=True)
    def tabs(self):
        return list(TABS)

    @Property("QVariantList", constant=True)
    def recentUpdates(self):
        """The landing's "Recent updates" timeline, parsed from the bundled
        CHANGELOG (newest 3) so it tracks releases automatically."""
        from firefly.ui.changelog import recent_updates
        return recent_updates(3)

    @Slot(int)
    def enterMain(self, tab: int):
        if 0 <= tab < len(TABS) and tab != self._tab:
            self._tab = tab
            self.tabChanged.emit()
        if self._page != "main":
            self._page = "main"
            self.pageChanged.emit()

    @Slot()
    def goLanding(self):
        if self._page != "landing":
            self._page = "landing"
            self.pageChanged.emit()

    @Slot(int)
    def setTab(self, tab: int):
        if 0 <= tab < len(TABS) and tab != self._tab:
            self._tab = tab
            self.tabChanged.emit()

    def _on_app_state(self, state):
        """Frontmost ⇄ background transitions (QGuiApplication signal)."""
        active = state == Qt.ApplicationState.ApplicationActive
        if active != self._app_active:
            self._app_active = active
            self.appActiveChanged.emit()
