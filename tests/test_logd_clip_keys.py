"""The Analysis tab must clamp LogD with the SAME range as the runs.

The Log-D clip setting moved to log₁₀D entries (`analysis/dcoeff_clip_logmin`
/ `logmax`), and params_builder converts them for every run.  The Analysis
workspace kept reading the retired linear keys (`analysis/dcoeff_clip_min` /
`max`) — which nothing writes any more — so it always clamped at the defaults
(1e-5…10 µm²/s) whatever the sidebar said.  Its redraw trigger watched the same
dead keys, so changing the range never refreshed that tab either.  The runs and
the live comparison disagreed and nothing said so.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication              # noqa: E402

_app = QApplication.instance() or QApplication([])

from test_compare_panel_coverage import _FakeSettings    # noqa: E402
from firefly.ui.controllers.workspace.workspace_controller import (  # noqa: E402
    AnalysisWorkspaceController)


def _ctrl(**values):
    return AnalysisWorkspaceController(settings=_FakeSettings(values))


def test_the_analysis_tab_uses_the_sidebar_clip_range():
    c = _ctrl(**{"analysis/dcoeff_clip_logmin": -3.0,
                 "analysis/dcoeff_clip_logmax": 0.0})
    lo, hi = c._dcoeff_clip()
    assert lo == pytest.approx(1e-3)
    assert hi == pytest.approx(1.0)


def test_it_matches_what_a_run_is_given():
    """Same settings in, same numbers out — the run and the tab must agree."""
    from firefly.ui.controllers.params.params_builder import build_params

    class _Imp:
        filePath = "/tmp/a.czi"; outDir = "/tmp/o"; isCsv = False
        overridePx = False; pixelSize = 0.1; overrideFi = False; frameInterval = 0.02

    class _S(_FakeSettings):
        def get_float(self, k, d=0.0): return float(self._d.get(k, d))

    vals = {"analysis/dcoeff_clip_logmin": -4.5, "analysis/dcoeff_clip_logmax": 0.5}
    run = build_params(_S(vals), _Imp(), fpath="/tmp/a.czi")
    tab = _ctrl(**vals)._dcoeff_clip()
    assert tab == pytest.approx((run["dcoeff_clip_min"], run["dcoeff_clip_max"]))


def test_defaults_are_unchanged():
    assert _ctrl()._dcoeff_clip() == pytest.approx((1e-5, 10.0))


def test_changing_the_clip_range_redraws_the_tab():
    c = _ctrl()
    before = c._engfig_rev
    c._on_figpref_changed("analysis/dcoeff_clip_logmin")
    assert c._engfig_rev > before, "the tab would keep showing the old clamp"
