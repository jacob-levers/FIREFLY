"""Regression tests for _ResultsPanel failure styling (R2-7, backs #17/#18).

A failed or empty run must NOT be painted in success-green.
"""
import pytest

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                       # noqa: E402
from firefly.ui.ui_theme import _apply_firefly_theme, _THEME   # noqa: E402
from firefly.ui import ui_widgets                   # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
_apply_firefly_theme(_app)


def _headline_colour(panel):
    return panel._headline.styleSheet().lower()


def test_show_results_severity_changes_headline_colour():
    p = ui_widgets._ResultsPanel("idle")
    p.show_results("ok", "", severity="success")
    success = _headline_colour(p)
    p.show_results("all failed", "", severity="danger")
    danger = _headline_colour(p)
    p.show_results("partial", "", severity="warn")
    warn = _headline_colour(p)
    # A failed/empty run is styled differently from a successful one (#17/#18).
    assert _THEME["DANGER"].lower() in danger
    assert danger != success
    assert _THEME["WARN"].lower() in warn


def test_show_warning_adds_a_banner():
    p = ui_widgets._ResultsPanel("idle")
    p.show_stats({})                 # empty summary → clears the flags container
    p.show_warning("No trajectories were produced", severity="warn")
    assert p._flags_layout.count() == 1
    banner = p._flags_layout.itemAt(0).widget()
    # Assert it's actually an alert banner, not just "some widget got added".
    assert isinstance(banner, ui_widgets._AlertBanner)
