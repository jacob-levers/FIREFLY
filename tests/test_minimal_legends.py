"""Minimal figures have no legends — all of them.

Minimal mode removed legends with ``ax.get_legend().remove()``.  The overlaid
MSD panel draws its "Group" key with ``ax.legend(...)`` and then
``ax.add_artist(...)`` (how a second key is kept on one axes), and for such a
legend ``remove()`` only detaches the added copy: the axes still drew it.  The
overlaid MSD panel therefore kept its key in minimal exports and in the
Analysis tab's preview.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from matplotlib.legend import Legend

from firefly.analysis.fa_compare import compute_report, render_report


def _legends(fig):
    fig.canvas.draw()
    return [lg for lg in fig.findobj(Legend) if lg.get_visible()]


@pytest.fixture(scope="module")
def rd(tmp_path_factory):
    from test_per_recording_curves import _uneven
    return compute_report(_uneven(tmp_path_factory.mktemp("d")))


@pytest.mark.parametrize("msd_style", ["overlaid", "mean_faceted", "individual"])
@pytest.mark.parametrize("panel_only", [True, False])
def test_a_minimal_msd_panel_has_no_legend(rd, msd_style, panel_only):
    fig, _s, _st = render_report(rd, panels={"msd"}, panel_only=panel_only, pdf_report=False,
                                 minimal=True, msd_plot_style=msd_style)
    assert _legends(fig) == [], msd_style
    plt.close(fig)


def test_a_minimal_full_report_has_no_legend_anywhere(rd):
    fig, _s, _st = render_report(rd, pdf_report=False, minimal=True, msd_plot_style="overlaid")
    assert _legends(fig) == []
    plt.close(fig)


def test_the_overlaid_msd_keeps_its_key_when_not_minimal(rd):
    """Out of minimal mode a lone exported panel still needs its colour key."""
    fig, _s, _st = render_report(rd, panels={"msd"}, panel_only=True, pdf_report=False,
                                 minimal=False, msd_plot_style="overlaid")
    assert _legends(fig)
    plt.close(fig)


def test_the_analysis_tab_preview_drops_it_too():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets
    _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from firefly.ui.controllers.workspace import workspace_figures as wf
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot([0, 1], [0, 1], label="A")
    key = ax.legend(title="Group")
    ax.add_artist(key)                  # the overlaid MSD panel's pattern
    wf._RENDER.minimal = True
    try:
        wf._qimage_from_figure(fig)
    finally:
        wf._RENDER.minimal = False
    assert _legends(fig) == []
    plt.close(fig)


# ── …and no space kept for a key that is gone ───────────────────────────────
@pytest.mark.parametrize("panel", ["motion_classes", "states_occupancy", "states_d"])
def test_a_minimal_panel_gives_its_key_strip_back(rd, panel):
    """Bar panels whose key sits beside the bars widen their x-range for it.
    Minimal drops the key, and the strip stayed behind as empty space on the
    right of the panel."""
    def bars_ax(minimal):
        fig, _s, _st = render_report(rd, panels={panel}, panel_only=True, pdf_report=False,
                                     minimal=minimal)
        ax = max((a for a in fig.axes if a.patches), key=lambda a: len(a.patches))
        right = max(p.get_x() + p.get_width() for p in ax.patches if p.get_width() < 5)
        out = (ax.get_xlim()[1] - right, len(_legends(fig)))
        plt.close(fig)
        return out
    gap_full, keys_full = bars_ax(False)
    gap_min, keys_min = bars_ax(True)
    assert keys_full and not keys_min
    assert gap_full > 1.0, "the full panel keeps room for its key"
    assert gap_min < 0.5, f"{gap_min:.2f} data units of empty space right of the bars"
