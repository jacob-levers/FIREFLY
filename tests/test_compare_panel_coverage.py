"""Every comparison graph you can browse is also exported.

The scroller, the Preferences picker and the engine's own ``panel_order`` are
three separate lists of the same keys, and the default export set was a fourth.
The default used to hold three panels back (track_count, linkstep, linkspeed),
so a graph could be on screen in the tab and absent from the report with
nothing saying so.
"""
import re

import pytest
from PySide6.QtWidgets import QApplication

from firefly.ui.controllers.workspace import workspace_data as wd
from firefly.ui.controllers.workspace.workspace_controller import \
    AnalysisWorkspaceController

ENGINE_SRC = "firefly/analysis/fa_compare.py"


@pytest.fixture(scope="module", autouse=True)
def _app():
    yield QApplication.instance() or QApplication([])


class _FakeSettings:
    """In-memory settings — SettingsController hardcodes the real QSettings
    domain, so a headless test must never touch it."""

    def __init__(self, d=None):
        self._d = dict(d or {})

    def get(self, k, default=""):
        return self._d.get(k, default)

    def get_str(self, k, default=""):
        return self._d.get(k, default)

    def get_bool(self, k, default=False):
        return self._d.get(k, default)

    def setValue(self, k, v):
        self._d[k] = v

    def sync(self):
        pass


def _engine_panel_order():
    src = open(ENGINE_SRC, encoding="utf-8").read()
    m = re.search(r"panel_order = warning_panels \+ \[(.*?)\]", src, re.S)
    assert m, "fa_compare no longer declares panel_order the same way"
    return re.findall(r'"([a-z_0-9]+)"', m.group(1))


# ── the four lists must agree ────────────────────────────────────────────────
def test_the_engine_offers_nothing_the_ui_hides():
    engine = _engine_panel_order()
    ui = [k for k, _ in wd.COMPARE_PANELS]
    assert [k for k in engine if k not in ui] == [], \
        "the engine can render a panel the picker never offers"
    assert [k for k in ui if k not in engine] == [], \
        "the picker offers a panel the engine cannot render"
    assert engine == ui, "the scroller order should mirror the exported figure"


def test_the_scroller_and_the_picker_hold_the_same_panels():
    assert set(wd.PANEL_KEYS) == {k for k, _ in wd.COMPARE_PANELS}


def test_every_browsable_panel_is_exported_by_default():
    """The whole point: nothing you can look at is silently left out."""
    assert wd.DEFAULT_COMPARE_PANELS == set(wd.PANEL_KEYS)
    assert len(wd.DEFAULT_COMPARE_PANELS) == len(wd.COMPARE_PANELS)


def test_a_fresh_controller_ticks_everything():
    c = AnalysisWorkspaceController(settings=None)
    rows = c.comparePanels
    assert len(rows) == len(wd.COMPARE_PANELS)
    assert all(r["on"] for r in rows), \
        [r["key"] for r in rows if not r["on"]]


# ── the stored preference has to follow the default forward ──────────────────
def test_the_old_default_is_upgraded_not_treated_as_a_choice():
    """The panel set is persisted as a side effect of switching the selected
    graph, so an existing user has the OLD default saved whether or not they
    ever opened the picker.  Without this they would keep 21 panels forever."""
    stored = ",".join(sorted(wd.LEGACY_DEFAULT_COMPARE_PANELS))
    c = AnalysisWorkspaceController(
        settings=_FakeSettings({"figures/compare_panels": stored}))
    assert c._panels == set(wd.DEFAULT_COMPARE_PANELS)


def test_a_real_customisation_is_left_alone():
    c = AnalysisWorkspaceController(
        settings=_FakeSettings({"figures/compare_panels": "msd,logd_dist"}))
    assert c._panels == {"msd", "logd_dist"}


def test_the_legacy_set_is_exactly_the_three_that_were_held_back():
    assert wd.DEFAULT_COMPARE_PANELS - wd.LEGACY_DEFAULT_COMPARE_PANELS == {
        "track_count", "linkstep", "linkspeed"}


def test_an_unknown_stored_key_is_dropped():
    c = AnalysisWorkspaceController(
        settings=_FakeSettings({"figures/compare_panels": "msd,not_a_panel"}))
    assert c._panels == {"msd"}
