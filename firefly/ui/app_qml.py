"""FIREFLY — QML / Qt Quick front-end entry point (Phase 2 UI rewrite).

Built alongside the legacy Widgets app (`firefly.ui.app_qt`); the two share the
analysis core, controllers wrap the backend, and the default launcher switches
to this once parity is reached. Architecture: Widget-rooted (`QMainWindow`) so
the bespoke QWidget canvases (FireflyViewer / RoiEditor) can compose as islands;
the UI itself is authored in QML and surfaced via `QQuickWidget`.
"""
from __future__ import annotations

import multiprocessing
import os
import sys

# macOS + multiprocessing: spawn is the only safe context for the analysis
# worker (clean interpreter for MPS/CUDA, no Qt/Metal claim) — same rationale as
# the Widgets app.  The Widgets entry sets this when `app_qt` is imported; the
# QML path doesn't import `app_qt`, so set it here before any run is spawned.
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # already set

from PySide6 import QtWidgets
from PySide6.QtCore import QUrl, Qt, QEvent, QObject
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtQuickControls2 import QQuickStyle

# The Basic style is the customisable one — our components restyle Controls
# (TextField/ComboBox/…) via the design tokens, which the native style forbids.
# Must be set before any Quick item is created, so do it at import time.
QQuickStyle.setStyle("Basic")

from firefly.ui.controllers.theme_controller import ThemeController
from firefly.ui.controllers.app_controller import AppController
from firefly.ui.controllers.settings_controller import SettingsController
from firefly.ui.controllers.import_controller import ImportController
from firefly.ui.controllers.analysis_controller import AnalysisController
from firefly.ui.controllers.visualise_controller import VisualiseController
from firefly.ui.controllers.roi_controller import RoiController
from firefly.ui.controllers.embed_controller import EmbedController
from firefly.ui.controllers.results_controller import ResultsController
from firefly.ui.controllers.compare_controller import CompareController
from firefly.ui.controllers.sidebar_controller import SidebarController
from firefly.ui.controllers.roi_store import RoiStore
from firefly.ui.controllers.icon_provider import IconImageProvider
from firefly.ui.controllers.live_frame_provider import LiveFrameProvider
from firefly.ui.controllers.figure_image_provider import FigureImageProvider
from firefly.ui.controllers.qimage_provider import QImageProvider
from firefly.sptpalm_analysis import __version__


class _StageResizer(QObject):
    """Keeps the chrome QQuickWidget filling the layout-less ``stage`` central
    widget on every resize.  The chrome is pinned at the stage origin so QML
    scene coords == stage coords (no offset maths for the native island)."""
    def __init__(self, stage, chrome):
        super().__init__(stage)
        self._stage = stage
        self._chrome = chrome

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.Resize:
            self._chrome.setGeometry(self._stage.rect())
        return False

_QML_DIR = os.path.join(os.path.dirname(__file__), "qml")
_ICONS_DIR = os.path.join(_QML_DIR, "assets", "icons")


def build_main_window(app: QtWidgets.QApplication):
    """Construct the QML shell hosted in a QMainWindow. Returns (window, ctx)
    where ctx keeps the controllers alive (they must outlive the QML engine)."""
    theme = ThemeController()
    appc = AppController()
    settings = SettingsController()
    importc = ImportController(settings)
    roi_store = RoiStore()
    analysis = AnalysisController(settings, importc, roi_store=roi_store)
    visualise = VisualiseController(settings, importc)
    roi = RoiController(roi_store)
    embed = EmbedController()
    results = ResultsController()
    comparec = CompareController(settings, results=results)
    sidebar = SidebarController(settings, importc)

    win = QtWidgets.QMainWindow()
    win.setWindowTitle("FIREFLY")

    # Layout-less stage central widget hosts three composited layers: the chrome
    # QQuickWidget (L1), the native FireflyViewer island (L2, raised over an
    # invisible QML anchor on the Visualise tab), and a transparent HUD
    # QQuickWidget (L3, raised above the island).  See EmbedController.
    stage = QtWidgets.QWidget()
    win.setCentralWidget(stage)

    # ── L1: chrome ───────────────────────────────────────────────────────
    qw = QQuickWidget(stage)
    qw.engine().addImageProvider("icon", IconImageProvider(_ICONS_DIR))
    qw.engine().addImageProvider("liveframe", LiveFrameProvider(analysis))
    qw.engine().addImageProvider("resultfig", FigureImageProvider(results))
    qw.engine().addImageProvider("comparefig", FigureImageProvider(comparec))
    qw.engine().addImageProvider("roibg", QImageProvider(roi.roi_image))
    ctx = qw.rootContext()
    ctx.setContextProperty("Theme", theme)
    ctx.setContextProperty("App", appc)
    ctx.setContextProperty("Settings", settings)
    ctx.setContextProperty("Import", importc)
    ctx.setContextProperty("Analysis", analysis)
    ctx.setContextProperty("Vis", visualise)
    ctx.setContextProperty("Roi", roi)
    ctx.setContextProperty("Embed", embed)
    ctx.setContextProperty("Results", results)
    ctx.setContextProperty("Compare", comparec)
    ctx.setContextProperty("Sidebar", sidebar)
    ctx.setContextProperty("appVersion", __version__)
    qw.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
    qw.setSource(QUrl.fromLocalFile(os.path.join(_QML_DIR, "Main.qml")))
    qw.setGeometry(0, 0, 1100, 760)
    for e in qw.errors():
        print(f"[FIREFLY-QML] {e.toString()}", file=sys.stderr)

    # ── L3: transparent HUD overlay (glass pill + track inspector) ───────
    hud = QQuickWidget(stage)
    hud.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    hud.setClearColor(Qt.GlobalColor.transparent)
    hud.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop, True)
    hud.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    hud.engine().addImageProvider("icon", IconImageProvider(_ICONS_DIR))
    hctx = hud.rootContext()
    hctx.setContextProperty("Theme", theme)
    hctx.setContextProperty("Vis", visualise)
    hctx.setContextProperty("Embed", embed)
    hud.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
    hud.setSource(QUrl.fromLocalFile(os.path.join(_QML_DIR, "HudOverlay.qml")))
    for e in hud.errors():
        print(f"[FIREFLY-QML] {e.toString()}", file=sys.stderr)
    hud.hide()

    # ── L2: native viewer island (built eagerly so the embed has a target) ─
    viewer_w = visualise.viewerWidget()
    viewer_w.setParent(stage)
    viewer_w.hide()
    embed.setIslands(viewer=viewer_w, hud=hud)

    resizer = _StageResizer(stage, qw)
    stage.installEventFilter(resizer)

    # Tab / page changes drive the island's show/hide (Visualise tab only).
    def _loc(*_):
        embed.onLocationChanged(appc.currentTab, appc.page)
    appc.tabChanged.connect(_loc)
    appc.pageChanged.connect(_loc)

    # A finished comparison loads its snapshot into the Results tab + jumps there.
    def _on_results_ready(rj):
        results.loadFromFile(rj)
        appc.setTab(3)
    comparec.resultsReady.connect(_on_results_ready)

    win.resize(1100, 760)
    # Keep controllers + widgets referenced on the window so Python doesn't GC
    # them while QML still binds to them (and the islands while they're hidden).
    win._firefly_ctx = (theme, appc, settings, importc, analysis, visualise,
                        roi, embed, results, comparec, sidebar, qw, hud,
                        viewer_w, resizer)
    return win, qw


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("FIREFLY")
    app.setOrganizationName("jacoblevers")   # must match the Widgets app for QSettings
    win, qw = build_main_window(app)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
