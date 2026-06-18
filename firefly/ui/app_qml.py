"""FIREFLY — QML / Qt Quick front-end entry point (Phase 2 UI rewrite).

Built alongside the legacy Widgets app (`firefly.ui.app_qt`); the two share the
analysis core, controllers wrap the backend, and the default launcher switches
to this once parity is reached. Architecture: Widget-rooted (`QMainWindow`) so
the bespoke QWidget canvases (FireflyViewer / RoiEditor) can compose as islands;
the UI itself is authored in QML and surfaced via `QQuickWidget`.
"""
from __future__ import annotations

import os
import sys

from PySide6 import QtWidgets
from PySide6.QtCore import QUrl
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
from firefly.ui.controllers.icon_provider import IconImageProvider
from firefly.sptpalm_analysis import __version__

_QML_DIR = os.path.join(os.path.dirname(__file__), "qml")
_ICONS_DIR = os.path.join(_QML_DIR, "assets", "icons")


def build_main_window(app: QtWidgets.QApplication):
    """Construct the QML shell hosted in a QMainWindow. Returns (window, ctx)
    where ctx keeps the controllers alive (they must outlive the QML engine)."""
    theme = ThemeController()
    appc = AppController()
    settings = SettingsController()
    importc = ImportController(settings)

    win = QtWidgets.QMainWindow()
    win.setWindowTitle("FIREFLY")

    qw = QQuickWidget()
    qw.engine().addImageProvider("icon", IconImageProvider(_ICONS_DIR))
    ctx = qw.rootContext()
    ctx.setContextProperty("Theme", theme)
    ctx.setContextProperty("App", appc)
    ctx.setContextProperty("Settings", settings)
    ctx.setContextProperty("Import", importc)
    ctx.setContextProperty("appVersion", __version__)
    qw.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
    qw.setSource(QUrl.fromLocalFile(os.path.join(_QML_DIR, "Main.qml")))
    errs = qw.errors()
    if errs:
        for e in errs:
            print(f"[FIREFLY-QML] {e.toString()}", file=sys.stderr)

    win.setCentralWidget(qw)
    win.resize(1100, 760)
    # Keep controllers + the quick widget referenced on the window so Python
    # doesn't GC them while QML still binds to them.
    win._firefly_ctx = (theme, appc, settings, importc, qw)
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
