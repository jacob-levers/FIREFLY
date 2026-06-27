"""AppController — top-level navigation + the app-focus gate for the landing.

The landing's molecule-field animation pauses while FIREFLY is backgrounded
(battery).  That is driven by `appActive`, which tracks
QGuiApplication.applicationStateChanged.  These tests exercise the property's
default and its active⇄inactive transitions without needing a real window.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from firefly.ui.controllers.app_controller import AppController

_app = QApplication.instance() or QApplication([])


def _spy(controller):
    seen = {"n": 0}
    controller.appActiveChanged.connect(lambda: seen.__setitem__("n", seen["n"] + 1))
    return seen


def test_app_active_defaults_true():
    # The controller is built before the window is shown, so it must NOT read
    # applicationState() (which would be Inactive then) — it defaults to running.
    c = AppController()
    assert c.appActive is True


def test_background_pauses_then_focus_resumes():
    c = AppController()
    spy = _spy(c)

    c._on_app_state(Qt.ApplicationState.ApplicationInactive)
    assert c.appActive is False and spy["n"] == 1

    # same state again must not re-emit (binding churn / wasted repaints)
    c._on_app_state(Qt.ApplicationState.ApplicationInactive)
    assert c.appActive is False and spy["n"] == 1

    c._on_app_state(Qt.ApplicationState.ApplicationActive)
    assert c.appActive is True and spy["n"] == 2


def test_suspended_hidden_states_count_as_inactive():
    c = AppController()
    for state in (Qt.ApplicationState.ApplicationSuspended,
                  Qt.ApplicationState.ApplicationHidden,
                  Qt.ApplicationState.ApplicationInactive):
        c._app_active = True
        c._on_app_state(state)
        assert c.appActive is False, state


def test_teardown_after_gc_does_not_crash():
    # A bound-method slot ties the applicationStateChanged connection to the
    # controller, so Qt auto-disconnects on destruction — a later state change
    # must never call into a torn-down controller.
    c = AppController()
    del c
    import gc
    gc.collect()
    for _ in range(10):
        _app.processEvents()
    _app.setActiveWindow(None)  # touch the app; no dangling slot should fire


def test_navigation_basics():
    c = AppController()
    assert c.page == "landing"
    c.enterMain(2)
    assert c.page == "main" and c.currentTab == 2
    c.goLanding()
    assert c.page == "landing"
    c.setTab(4)
    assert c.currentTab == 4
