"""Live Analysis-tab 'Grouped by timepoint' wiring (workspace_figures).

`render_metric(..., group_style="grouped", grouped_data=...)` must reshape the
per-condition-NAME × timepoint payload into the ``{name: {phase: values}}`` map
the shared renderer expects, and carry the timepoint order + colors through.

Pure-Python: the Qt QImage step and the matplotlib drawing are stubbed so the
test asserts the *structure* handed to the renderer, no Qt/display needed.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np

from firefly.ui.controllers.workspace import workspace_figures as wf
from firefly.analysis import fa_group_figures as gf


class _M:  # minimal metric stand-in
    id = "D"; label = "D"; unit = "µm²/s"; axis = "D (µm²/s)"


def _capture(monkeypatch):
    seen = {}

    def fake_draw(fig, ss, order, values, **kw):
        seen["order"] = order; seen["values"] = values; seen["kw"] = kw

    monkeypatch.setattr(gf, "draw_group_comparison", fake_draw)
    monkeypatch.setattr(wf, "_qimage_from_figure", lambda fig: "IMG")
    return seen


def test_grouped_uses_name_by_timepoint(monkeypatch):
    seen = _capture(monkeypatch)
    grouped = {
        "names": ["Control", "KCl"],
        "phases": ["pre", "post"],
        "data": {
            "Control": {"pre": np.array([0.10, 0.12, 0.09]),
                        "post": np.array([0.20, 0.22])},
            "KCl":     {"pre": np.array([0.11, 0.13]),
                        "post": np.array([0.30, 0.28, 0.31])},
        },
        "colors": {"Control": "#58a6ff", "KCl": "#f78166"},
    }
    out = wf.render_metric([], _M(), plot="Box", group_style="grouped",
                           grouped_data=grouped, width_px=600, height_px=360, dpi=100)
    assert out == "IMG"
    assert seen["order"] == ["Control", "KCl"]
    assert set(seen["values"]["Control"]) == {"pre", "post"}
    assert list(seen["values"]["KCl"]["post"]) == [0.30, 0.28, 0.31]
    assert seen["kw"]["tp_order"] == ["pre", "post"]
    assert seen["kw"]["group_colors"]["KCl"] == "#f78166"
    # KW label is computed across all pooled per-name values
    assert "Kruskal" in seen["kw"]["stat_label"]


def test_grouped_without_payload_falls_back_to_one_series(monkeypatch):
    # No grouped_data → one series per group (the legacy single-box behaviour).
    seen = _capture(monkeypatch)
    groups = [
        {"label": "A", "color": "#58a6ff", "values": np.array([0.1, 0.2, 0.15])},
        {"label": "B", "color": "#f78166", "values": np.array([0.3, 0.25])},
    ]
    wf.render_metric(groups, _M(), plot="Box", group_style="grouped",
                     grouped_data=None, width_px=600, height_px=360, dpi=100)
    assert seen["order"] == ["A", "B"]
    assert list(seen["values"]["A"]) == [""]          # single unnamed timepoint
    assert seen["kw"]["tp_order"] is None
