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


def test_app_controller_navigation():
    from firefly.ui.controllers.app_controller import AppController
    a = AppController()
    assert a.page == "landing" and a.currentTab == 0
    assert a.tabs[:2] == ["Import", "Analysis"]
    a.enterMain(2)
    assert a.page == "main" and a.currentTab == 2
    a.setTab(4)
    assert a.currentTab == 4
    a.goLanding()
    assert a.page == "landing"
