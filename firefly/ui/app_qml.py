"""FIREFLY — QML / Qt Quick front-end entry point.

The sole front-end; controllers wrap the shared analysis core. Architecture:
Widget-rooted (`QMainWindow`) so the bespoke QWidget canvases (FireflyViewer /
RoiEditor) can compose as islands; the UI itself is authored in QML and surfaced
via `QQuickWidget`.
"""
from __future__ import annotations

import collections
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
from PySide6.QtCore import QUrl, Qt, QEvent, QObject, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtQuickControls2 import QQuickStyle

from firefly import crash_reporter

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
from firefly.ui.controllers.workspace.workspace_controller import AnalysisWorkspaceController
from firefly.ui.controllers.params.sidebar_controller import SidebarController
from firefly.ui.controllers.preset_controller import PresetController
from firefly.ui.controllers.batch_controller import BatchController
from firefly.ui.controllers.hyperfly_controller import (HyperflyController,
                                                        HfWorkerFrameProvider)
from firefly.ui.controllers.updates_controller import UpdatesController
from firefly.ui.controllers.roi_store import RoiStore, RoiOverrideStore
from firefly.ui.controllers.providers.icon_provider import IconImageProvider
from firefly.ui.controllers.providers.live_frame_provider import LiveFrameProvider
from firefly.ui.controllers.providers.figure_image_provider import FigureImageProvider
from firefly.ui.controllers.providers.qimage_provider import QImageProvider
from firefly.ui.controllers.workspace.workspace_panel_provider import WorkspacePanelProvider
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


class _CrashUI(QObject):
    """Marshals a crash-report notification onto the GUI thread.  The crash
    reporter's ``on_crash`` callback can fire from any thread (a worker's
    ``threading.excepthook``); emitting ``requested`` over a queued connection
    hands the path to ``_show`` on the main thread, mirroring the Widgets app's
    invokeMethod-queued crash dialog."""
    requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.requested.connect(self._show, Qt.ConnectionType.QueuedConnection)

    @Slot(str)
    def _show(self, path: str):
        try:
            box = QtWidgets.QMessageBox()
            box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            box.setWindowTitle("FIREFLY — Unexpected error")
            box.setText("FIREFLY hit an unexpected error and saved a crash report.")
            box.setInformativeText(path)
            open_btn = box.addButton("Open folder",
                                     QtWidgets.QMessageBox.ButtonRole.ActionRole)
            box.addButton(QtWidgets.QMessageBox.StandardButton.Close)
            box.exec()
            if box.clickedButton() is open_btn:
                QDesktopServices.openUrl(
                    QUrl.fromLocalFile(os.path.dirname(path)))
        except Exception:
            pass


# Resolve the QML tree both in dev (next to this file) and in a frozen
# PyInstaller build (extracted under sys._MEIPASS/firefly/ui/qml — see
# sptpalm.spec, which bundles the whole dir).  Without the frozen branch a
# packaged app looks beside a path that doesn't exist → Main.qml never loads.
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _QML_DIR = os.path.join(sys._MEIPASS, "firefly", "ui", "qml")
else:
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
    roi_override = RoiOverrideStore()     # per-file ROI-settings overrides (viewer)
    analysis = AnalysisController(settings, importc, roi_store=roi_store,
                                  override_store=roi_override)
    visualise = VisualiseController(settings, importc)
    roi = RoiController(roi_store, settings, override_store=roi_override)
    embed = EmbedController()
    results = ResultsController()
    comparec = CompareController(settings, results=results)
    # merged live Compare + Results workspace (the new "Analysis" tab); the old
    # run cockpit keeps its AnalysisController but is now exposed as "Process".
    workspace = AnalysisWorkspaceController(settings)
    sidebar = SidebarController(settings, importc)
    presets = PresetController(sidebar)
    hyperfly = HyperflyController()       # live parallel-batch (HYPER-FLY) dashboard
    batchc = BatchController(settings, importc, roi_store=roi_store,
                             override_store=roi_override, hyperfly=hyperfly)
    updates = UpdatesController()

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
    qw.engine().addImageProvider("roimask", QImageProvider(roi.roi_mask_image))
    qw.engine().addImageProvider("hfworker", HfWorkerFrameProvider(hyperfly))
    qw.engine().addImageProvider("importthumb", QImageProvider(importc.thumb_image))
    qw.engine().addImageProvider("workspacefig", QImageProvider(workspace.figure_image))
    qw.engine().addImageProvider("workspacepanel", WorkspacePanelProvider(workspace))
    ctx = qw.rootContext()
    ctx.setContextProperty("Theme", theme)
    ctx.setContextProperty("App", appc)
    ctx.setContextProperty("Settings", settings)
    ctx.setContextProperty("Import", importc)
    ctx.setContextProperty("Process", analysis)     # run cockpit (was "Analysis")
    ctx.setContextProperty("Analysis", workspace)    # merged Compare+Results workspace
    ctx.setContextProperty("Vis", visualise)
    ctx.setContextProperty("Roi", roi)
    ctx.setContextProperty("Embed", embed)
    ctx.setContextProperty("Results", results)
    ctx.setContextProperty("Compare", comparec)
    ctx.setContextProperty("Sidebar", sidebar)
    ctx.setContextProperty("Preset", presets)
    ctx.setContextProperty("Batch", batchc)
    ctx.setContextProperty("Hyperfly", hyperfly)
    ctx.setContextProperty("Updates", updates)
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

    # The viewer island only appears once the Visualise viewer has content, so an
    # empty Visualise tab shows the placeholder instead of a blank floating card.
    def _vis_content(*_):
        embed.setViewerContent(visualise.hasRun)
    visualise.dataChanged.connect(_vis_content)

    # The preview colour is chosen in the ROI editor; reflect it in the Import
    # tab's max-projection thumbnail (both read the shared 'ui/preview_cmap').
    roi.cmapChanged.connect(importc.refreshPreviewColour)

    # A finished comparison loads its snapshot into the Results tab + jumps there.
    def _on_results_ready(rj):
        results.loadFromFile(rj)
        appc.setTab(3)
    comparec.resultsReady.connect(_on_results_ready)

    # ── Crash reporter ───────────────────────────────────────────────────
    # The Widgets app installs this; the QML entrypoint never did, so uncaught
    # exceptions (incl. on worker threads) left no report.  Feed the reporter a
    # rolling log tail + a snapshot of the current run, and surface the saved
    # report path via a dialog marshalled to the GUI thread.
    crash_ui = _CrashUI()
    log_tail: collections.deque = collections.deque(maxlen=400)
    analysis.logLine.connect(lambda s: log_tail.extend(str(s).splitlines()))

    def _log_provider(n: int = 120) -> str:
        return "\n".join(list(log_tail)[-n:])

    def _state_provider() -> dict:
        try:
            from shiboken6 import isValid as _is_valid
            if not _is_valid(win):
                return {"<state>": "main window already destroyed"}
        except Exception:
            pass
        try:
            return {
                "UI":               "PySide6 QML",
                "Current file":     importc.filePath,
                "Output folder":    importc.outDir or "(default)",
                "Pixel size":       importc.pixelSize,
                "Frame interval":   importc.frameInterval,
                "Running":          bool(getattr(analysis, "running", False)),
            }
        except Exception as e:                       # never raise from the hook
            return {"<state error>": repr(e)}

    crash_reporter.set_log_provider(_log_provider)
    crash_reporter.set_app_state_provider(_state_provider)
    crash_reporter.install_global_handlers(on_crash=crash_ui.requested.emit)

    win.resize(1100, 760)
    # Keep controllers + widgets referenced on the window so Python doesn't GC
    # them while QML still binds to them (and the islands while they're hidden).
    win._firefly_ctx = (theme, appc, settings, importc, analysis, visualise,
                        roi, embed, results, comparec, sidebar, presets, batchc,
                        updates, qw, hud, viewer_w, resizer, crash_ui)
    return win, qw


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("FIREFLY")
    app.setOrganizationName("jacoblevers")   # must match the Widgets app for QSettings
    win, qw = build_main_window(app)
    win.show()

    # CI/frozen smoke-test marker (mirrors the Widgets app): write the file the
    # packaging smoke waits on once the QML root has rendered, so a "blank frozen
    # window" (missing Qt Quick plugins) is caught — the marker only lands when
    # the scene graph actually painted.
    marker_path = os.environ.get("SPTPALM_READY_MARKER")
    if marker_path:
        try:    qw.repaint()
        except Exception: pass
        try:
            with open(marker_path, "w") as f:
                f.write("ready\n")
        except Exception:
            pass

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
