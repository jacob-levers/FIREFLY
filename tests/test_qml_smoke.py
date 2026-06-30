"""Headless smoke tests for the QML UI shell (Phase 2 rewrite).

Loads the real `Main.qml` through the app's `build_main_window` under the
offscreen platform + software scene-graph and asserts it loads cleanly, plus
unit tests for the Theme/App controllers (plain QObjects — no QML needed).
Skipped in the Qt-less CI image, like the other Qt tests.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

import pytest                                          # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("PySide6.QtQuickWidgets")
from PySide6 import QtWidgets                           # noqa: E402
from PySide6.QtQuickWidgets import QQuickWidget         # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_shell_loads_without_qml_errors():
    from firefly.ui.app_qml import build_main_window
    win, qw = build_main_window(_app)
    assert qw.status() == QQuickWidget.Status.Ready
    assert qw.errors() == []
    assert qw.rootObject() is not None
    # composes + rasterises (chrome renders)
    win.resize(900, 600)
    win.show()
    _app.processEvents()
    assert not win.grab().isNull()
    win.hide()


def test_theme_controller_tokens_and_live_switch():
    from firefly.ui.controllers.theme_controller import ThemeController
    t = ThemeController()
    pal = t.palette
    assert {"BG", "PANEL", "ACC", "TXT", "DANGER"} <= set(pal)
    assert t.scale["radiusLg"] == 6 and t.scale["sidebarWidth"] == 380
    start = t.name
    other = next(n for n in t.themes if n != start)
    seen = []
    t.changed.connect(lambda: seen.append(1))
    t.setTheme(other)
    assert t.name == other and seen == [1]
    assert t.palette["BG"] == _theme_bg(other)


def _theme_bg(name):
    from firefly.ui.ui_theme import _THEMES
    return _THEMES[name]["BG"]


def test_icon_provider_renders_tinted():
    from PySide6.QtCore import QSize
    from firefly.ui import app_qml
    from firefly.ui.controllers.providers.icon_provider import IconImageProvider
    p = IconImageProvider(app_qml._ICONS_DIR)
    img = p.requestImage("scan-search/58a6ff", QSize(), QSize(48, 48))
    assert not img.isNull() and img.width() == 48
    # the stroke renders (some opaque pixels) and isn't pure black (it's tinted)
    opaque = [img.pixelColor(x, y) for x in range(0, 48, 3)
              for y in range(0, 48, 3) if img.pixelColor(x, y).alpha() > 40]
    assert opaque, "icon rendered empty"
    assert any(c.blue() > c.red() for c in opaque), "icon not tinted to accent"


def test_import_controller_probe_and_calibration(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    import numpy as np
    from firefly.ui.controllers.import_controller import ImportController

    class FakeSettings:           # avoid touching the user's real QSettings
        def __init__(self): self.d = {}
        def get_str(self, k, d=""): return str(self.d.get(k, d))
        def get_float(self, k, d=0.0):
            try: return float(self.d.get(k, d))
            except Exception: return d
        def get_bool(self, k, d=False): return bool(self.d.get(k, d))
        def set(self, k, v): self.d[k] = v

    fs = FakeSettings()
    ic = ImportController(fs)
    assert not ic.hasFile and ic.fileFormat == ""

    p = tmp_path / "m.tif"
    tifffile.imwrite(str(p), np.zeros((42, 16, 16), dtype=np.uint16))
    ic.filePath = str(p)
    assert ic.hasFile and ic.fileFormat == "TIFF" and ic.frameCount == 42
    assert ic.fileName == "m.tif"
    assert fs.d["analysis/file"] == str(p)

    ic.overridePx = True
    ic.pixelSize = 0.108
    assert fs.d["analysis/override_px"] is True
    assert fs.d["analysis/pixel_size"] == 0.108

    csv = tmp_path / "locs.csv"
    csv.write_text("x,y,frame\n1,2,0\n")
    ic.filePath = str(csv)
    assert ic.isCsv and ic.fileFormat == "localisations"


def test_analysis_tab_qml_loads_without_errors():
    # The Analysis cockpit only instantiates when routed to tab 1, so load it
    # directly through the configured engine to surface any QML binding errors.
    import os as _os
    from PySide6.QtCore import QUrl
    from PySide6.QtQml import QQmlComponent
    from firefly.ui.app_qml import build_main_window, _QML_DIR
    win, qw = build_main_window(_app)
    comp = QQmlComponent(
        qw.engine(),
        QUrl.fromLocalFile(_os.path.join(_QML_DIR, "tabs", "AnalysisTab.qml")))
    obj = comp.create(qw.rootContext())
    assert comp.errors() == [], comp.errorString()
    assert obj is not None


def test_visualise_tab_qml_loads_without_errors():
    import os as _os
    from PySide6.QtCore import QUrl
    from PySide6.QtQml import QQmlComponent
    from firefly.ui.app_qml import build_main_window, _QML_DIR
    win, qw = build_main_window(_app)
    for rel in (("tabs", "ImportTab.qml"), ("tabs", "VisualiseTab.qml"),
                ("tabs", "ProcessTab.qml"), ("tabs", "HyperflyTab.qml"),
                ("HudOverlay.qml",), ("RoiOverlay.qml",),
                ("components", "ParameterSidebar.qml"), ("PreferencesDialog.qml",)):
        comp = QQmlComponent(qw.engine(),
                             QUrl.fromLocalFile(_os.path.join(_QML_DIR, *rel)))
        obj = comp.create(qw.rootContext())
        assert comp.errors() == [], comp.errorString()
        assert obj is not None


def test_app_controller_navigation():
    from firefly.ui.controllers.app_controller import AppController
    a = AppController()
    assert a.page == "landing" and a.currentTab == 0
    assert a.tabs[:2] == ["Import", "Process"]
    a.enterMain(2)
    assert a.page == "main" and a.currentTab == 2
    a.setTab(3)
    assert a.currentTab == 3
    a.goLanding()
    assert a.page == "landing"
