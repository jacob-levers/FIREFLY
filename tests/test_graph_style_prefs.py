"""Preferences → Graph styles: one style for every comparison graph, per-graph
overrides, and controls that do what they say.

Fifteen rows offered the same Box + points / Violin + points / Bar choice, one
per graph, and read as one setting repeated.  Now one "Comparison graphs" style
(``figures/group_style``) applies to all of them, and a graph is set apart only
under "Customise individual graphs" (``figures/style_<panel>``, empty = same as
all).  "Trajectory length distribution" offered Density / Box while the graph —
in the export and the live tab — was always a cumulative curve and never read
the setting; it now offers Cumulative (the default, unchanged) or Density, and
both are drawn.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import hashlib
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from firefly.analysis.fa_compare import compute_report, render_report


@pytest.fixture(scope="module")
def rd(tmp_path_factory):
    from test_per_recording_curves import _uneven
    return compute_report(_uneven(tmp_path_factory.mktemp("d")))


def _png(rd, panel, **kw):
    fig, _s, _st = render_report(rd, panels={panel}, panel_only=True, pdf_report=False,
                                 theme="Light", **kw)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=40)
    ylabels = [a.get_ylabel() for a in fig.axes if a.get_ylabel()]
    plt.close(fig)
    return hashlib.md5(buf.getvalue()).hexdigest(), ylabels


# ── the engine ──────────────────────────────────────────────────────────────
def test_trajectory_length_cumulative_is_the_unchanged_default(rd):
    default, ylab = _png(rd, "track_length")
    assert _png(rd, "track_length", length_plot_style="cdf")[0] == default
    assert "Cumulative fraction" in ylab
    # a stored value from the old control ("box") still draws the default
    assert _png(rd, "track_length", length_plot_style="box")[0] == default


def test_trajectory_length_density_is_drawn(rd):
    dens, ylab = _png(rd, "track_length", length_plot_style="density")
    assert dens != _png(rd, "track_length")[0]
    assert any("density" in y.lower() for y in ylab), ylab


# ── the Analysis tab's settings → the engine ────────────────────────────────
@pytest.fixture
def controller():
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from firefly.ui.controllers.workspace.workspace_controller import AnalysisWorkspaceController

    class FakeSettings(QObject):
        changed = Signal(str)
        def __init__(self): super().__init__(); self._d = {}
        def get(self, k, d=None): return self._d.get(k, d)
        def getStr(self, k, d=""): return str(self._d.get(k, d))
        def getBool(self, k, d=False): return bool(self._d.get(k, d))
        def setValue(self, k, v): self._d[str(k)] = v; self.changed.emit(str(k))
        def sync(self): pass

    return AnalysisWorkspaceController(settings=FakeSettings())


def test_one_style_reaches_every_graph_that_is_not_set_apart(controller, rd):
    s = controller._settings
    s.setValue("figures/group_style", "violin")
    s.setValue("figures/style_rg", "bar")               # one graph set apart
    s.setValue("figures/style_path", "")                # "Same as all"
    kw = controller._render_report_kwargs({"rg", "netdisp", "path"})
    assert kw["group_style"] == "violin"
    assert kw["panel_styles"] == {"rg": "bar"}
    for panel, want in (("netdisp", "violin"), ("path", "violin"), ("rg", "bar")):
        got = _png(rd, panel, group_style=kw["group_style"], panel_styles=kw["panel_styles"])[0]
        assert got == _png(rd, panel, panel_styles={panel: want})[0], panel


def test_the_length_style_reaches_the_engine(controller):
    assert controller._render_report_kwargs({"track_length"})["length_plot_style"] == "cdf"
    controller._settings.setValue("figures/length_style", "density")
    assert controller._render_report_kwargs({"track_length"})["length_plot_style"] == "density"


def test_the_single_condition_preview_draws_both_length_styles(tmp_path):
    pytest.importorskip("PySide6")
    from firefly.ui.controllers.workspace import workspace_data as wd, workspace_figures as wf
    from test_per_recording_curves import _live_groups
    seen = []
    real = wf._qimage_from_figure
    wf._qimage_from_figure = lambda fig: seen.append([a.get_ylabel() for a in fig.axes]) or "img"
    try:
        groups = _live_groups(tmp_path, "len")[:1]
        for style in ("cdf", "density"):
            wf.render_metric(groups, wd.METRIC_BY_ID["len"], length_style=style)
    finally:
        wf._qimage_from_figure = real
    assert any("Cumulative fraction" in y for y in seen[0])
    assert any("density" in y.lower() for y in seen[1])


# ── the Preferences dialog ──────────────────────────────────────────────────
def _find(item, name):
    for ch in item.childItems():
        if ch.objectName() == name:
            return ch
        r = _find(ch, name)
        if r is not None:
            return r


def _model(item):
    m = item.property("model")
    return m.toVariant() if hasattr(m, "toVariant") else list(m)


def test_the_dialog_offers_one_style_and_hides_the_overrides(qml_window):
    from test_qml_smoke import _app
    win, qw = qml_window
    root = qw.rootObject()
    prefs = next(c for c in root.findChildren(type(root))
                 if c.property("section") is not None and c.property("opened") is not None)
    prefs.setProperty("section", "figures"); prefs.setProperty("opened", True)
    win.resize(1400, 950); win.show(); _app.processEvents()

    every = _find(root, "graphStyleAll")
    assert every is not None, "no single style for the comparison graphs"
    assert _model(every) == ["Box + points", "Violin + points", "Bar"]

    toggle = _find(root, "customiseGraphs")
    rows = _find(root, "graphOverrides")
    assert toggle is not None and rows is not None
    assert rows.property("visible") is False, "the per-graph overrides should start folded away"

    one = _find(root, "graphStyle_rg")
    assert _model(one)[0] == "Same as all graphs"

    length = _find(root, "lengthStyleSelect")
    assert _model(length) == ["Cumulative", "Density"]
    auc = _find(root, "aucStyleSelect")
    assert all("needs timepoints" in l for l in _model(auc) if l.startswith(("Paired", "Δ")))
    prefs.setProperty("opened", False); _app.processEvents()


from test_qml_smoke import qml_window  # noqa: E402,F401
