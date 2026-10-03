"""Preferences → Motion-class palette → Colour-blind safe recolours the motion
classes everywhere, not just in the Visualise viewer.

It used to say "Exported figures follow the figure theme — pick Publication
there for colour-blind", so the comparison report, the live Analysis tab and
each run's figure kept red/orange/blue/green unless the whole figure theme was
switched.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

from firefly.analysis.fa_compare import compare_groups
from firefly.analysis.fa_constants import MOTION_CLASS_ORDER, motion_class_colors

OKABE_ITO = {"Immobile": "#d55e00", "Confined": "#f0e442", "Brownian": "#0072b2",
             "Directed": "#009e73", "Unknown": "#999999"}


@pytest.mark.parametrize("theme", ["Dark", "Light", "Publication", "AMOLED"])
def test_the_option_gives_okabe_ito_whatever_the_theme(theme):
    assert motion_class_colors(theme, colourblind=True) == OKABE_ITO
    assert motion_class_colors("Dark") != OKABE_ITO                # and is off by default


def _bar_colours(fig):
    return {to_hex(p.get_facecolor()) for ax in fig.axes for p in ax.patches
            if p.get_height() > 0 and p.get_width() < 1}


@pytest.mark.parametrize("cb", [False, True])
def test_the_report_motion_panel_follows_the_option(tmp_path, cb):
    from test_workspace_data import make_run_folder
    groups = [{"label": f"G{g}", "color": "#888888",
               "folders": [make_run_folder(str(tmp_path), f"g{g}_{k}", seed=10 * g + k) for k in range(2)]}
              for g in range(2)]
    fig, _s, _st = compare_groups(groups, output_dir=None, panels={"motion_classes"},
                                  pdf_report=False, theme="Dark", motion_colourblind=cb)
    want = {motion_class_colors("Dark", colourblind=cb)[c] for c in MOTION_CLASS_ORDER}
    assert want <= _bar_colours(fig)
    plt.close(fig)


def test_the_live_motion_graph_follows_the_option(monkeypatch):
    pytest.importorskip("PySide6")
    from firefly.ui.controllers.workspace import workspace_data as wd, workspace_figures as wf
    seen = []
    monkeypatch.setattr(wf, "_qimage_from_figure", lambda fig: seen.append(fig))
    groups = [{"label": "A", "color": "#888888",
               "motion_counts": {"Immobile": 5, "Confined": 3, "Brownian": 2, "Directed": 1}}]
    motion = wd.METRIC_BY_ID["motion"]
    wf.render_metric(groups, motion, motion_colourblind=True)
    wf.render_metric(groups, motion)
    cb, std = (_bar_colours(f) for f in seen)
    assert {OKABE_ITO[c] for c in ("Immobile", "Confined", "Brownian", "Directed")} <= cb
    assert not cb & {wd.MOTION_COLORS[c] for c in ("Immobile", "Brownian")}
    assert wd.MOTION_COLORS["Brownian"] in std and OKABE_ITO["Brownian"] not in std
    assert not getattr(wf._RENDER, "colourblind", False)            # reset after the render


def test_the_live_palette_mirrors_the_analysis_core():
    from firefly.ui.controllers.workspace import workspace_data as wd
    assert wd.motion_colors(True) == motion_class_colors("Dark", colourblind=True)
    assert wd.motion_colors(False) == motion_class_colors("Dark")


def test_runs_pick_the_option_up_from_settings():
    """Each run's own figure (fa_figure.make_figure) gets it via the params."""
    from firefly.ui.controllers.params.params_builder import build_params
    from firefly.ui.controllers.roi_store import RoiOverrideStore, RoiStore
    from test_per_file_wavelet import _Import, _Settings

    def p(**s):
        return build_params(_Settings(s), _Import(), fpath="/tmp/a.tif",
                            roi_store=RoiStore(), override_store=RoiOverrideStore())
    assert p(**{"visualise/motion_colours": "Colour-blind safe"})["motion_colourblind"] is True
    assert p()["motion_colourblind"] is False


def test_the_preferences_text_no_longer_says_exports_ignore_it():
    import os
    qml = open(os.path.join(os.path.dirname(__file__), "..", "firefly", "ui", "qml",
                            "PreferencesDialog.qml"), encoding="utf-8").read()
    assert "Exported figures follow the figure theme" not in qml
