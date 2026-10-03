"""Comparison graphs: only the p-value, above the data; and a minimal mode with no
in-figure legend, title or group-summary band (Preferences → Figures).

The label used to be a red block of test name, p, stars, effect size and
correction drawn over the points (three-group panels put it inside the axes at
top-left).  The rest of that is in the statistics CSV and report.  Minimal mode
is for thesis figures whose legend lives in the caption.
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from firefly.analysis.fa_compare import compare_groups

P_ONLY = re.compile(r"^(p = (\d\.\d{3}|\d\.\d{2}e-\d+)|p < 1e-300)$")


def _groups(tmp_path, n):
    from test_workspace_data import make_run_folder
    centres = [0.05, 0.4, 0.15][:n]
    out = []
    for gi, c in enumerate(centres):
        out.append({"label": f"G{gi}", "color": ["#000000", "#d55e00", "#0072b2"][gi],
                    "folders": [make_run_folder(str(tmp_path), f"g{gi}_{k}", seed=10 * gi + k, d_centre=c)
                                for k in range(3)]})
    return out


def _auc_axis(fig):
    # by the y label: minimal mode clears every title
    return next(a for a in fig.axes if a.get_ylabel().startswith("AUC"))


def _p_texts(ax):
    return [t for t in ax.texts if t.get_text().startswith(("p = ", "p < "))]


@pytest.mark.parametrize("n,pairs", [(2, 1), (3, 3)])
def test_every_pair_gets_a_bracket_labelled_with_its_p_value_alone(tmp_path, n, pairs):
    fig, _s, _st = compare_groups(_groups(tmp_path, n), output_dir=None, panels={"auc"},
                                  pdf_report=False, auc_plot_style="box_points")
    ax = _auc_axis(fig)
    labels = [t.get_text() for t in ax.texts if t.get_text().strip()]
    assert len(labels) == pairs and all(P_ONLY.match(t) for t in labels), labels
    brackets = [l for l in ax.lines if len(l.get_xdata()) == 4]
    assert len(brackets) == pairs
    spans = sorted(tuple(sorted({l.get_xdata()[0], l.get_xdata()[-1]})) for l in brackets)
    assert spans == sorted({(float(i), float(j)) for i in range(n) for j in range(i + 1, n)})
    plt.close(fig)


def test_replicate_dots_take_their_group_colour(tmp_path):
    from matplotlib.colors import to_rgba
    groups = _groups(tmp_path, 3)
    fig, _s, _st = compare_groups(groups, output_dir=None, panels={"auc"},
                                  pdf_report=False, auc_plot_style="box_points")
    dots = [c for c in _auc_axis(fig).collections if len(c.get_offsets()) == 3]
    assert len(dots) == 3
    for c, g in zip(dots, groups):
        assert np.allclose(c.get_facecolor()[:, :3], to_rgba(g["color"])[:3])
        assert np.allclose(c.get_sizes(), 22)                       # smaller than before (34)
    plt.close(fig)


def test_a_pair_with_an_untestable_group_gets_no_bracket(tmp_path):
    """Propofol + Aniline was one fly: no test against it is possible."""
    from test_workspace_data import make_run_folder
    groups = _groups(tmp_path, 2) + [{"label": "Single", "color": "#0072b2",
                                       "folders": [make_run_folder(str(tmp_path), "one", seed=99, d_centre=0.2)]}]
    fig, _s, _st = compare_groups(groups, output_dir=None, panels={"auc"},
                                  pdf_report=False, auc_plot_style="box_points")
    brackets = [l for l in _auc_axis(fig).lines if len(l.get_xdata()) == 4]
    assert [tuple(sorted({l.get_xdata()[0], l.get_xdata()[-1]})) for l in brackets] == [(0.0, 1.0)]
    plt.close(fig)


def test_with_many_groups_only_significant_pairs_are_bracketed():
    from firefly.analysis.fa_figure_common import select_bracket_pairs
    pairs = [(0, 1, 0.01), (0, 2, 0.30), (1, 2, float("nan")), (3, 4, 0.04)]
    assert select_bracket_pairs(pairs, 4) == [(0, 1, 0.01), (0, 2, 0.30), (3, 4, 0.04)]
    assert select_bracket_pairs(pairs, 5) == [(0, 1, 0.01), (3, 4, 0.04)]


def test_brackets_that_would_overlap_are_stacked():
    from firefly.analysis.fa_figure_common import draw_pvalue_brackets
    fig, ax = plt.subplots(); ax.set_ylim(0, 1)
    texts = draw_pvalue_brackets(ax, [(0, 1, "p = 0.010"), (1, 2, "p = 0.200"), (0, 2, "p = 0.040")],
                                 color="k", data_top=1.0)
    ys = sorted(t.get_position()[1] for t in texts)
    assert ys[0] < ys[1] < ys[2]                        # three levels: touching spans don't share one
    assert ax.get_ylim()[1] > ys[2]
    plt.close(fig)


@pytest.mark.parametrize("n", [2, 3])
def test_the_p_value_sits_above_the_data(tmp_path, n):
    fig, _s, _st = compare_groups(_groups(tmp_path, n), output_dir=None, panels={"auc"},
                                  pdf_report=False, auc_plot_style="box_points")
    ax = _auc_axis(fig)
    fig.canvas.draw()
    points = np.concatenate([c.get_offsets()[:, 1] for c in ax.collections if len(c.get_offsets())])
    highest_point_px = ax.transData.transform((0, points.max()))[1]
    for txt in _p_texts(ax):
        assert txt.get_window_extent().y0 > highest_point_px
    plt.close(fig)


def test_minimal_mode_has_no_legend_titles_or_band(tmp_path):
    """No panel titles either: in a thesis figure they go in the caption."""
    fig, _s, _st = compare_groups(_groups(tmp_path, 3), output_dir=None, minimal=True,
                                  panels={"auc", "logd_dist", "motion_classes", "dwell_cdf"},
                                  pdf_report=False, auc_plot_style="box_points")
    assert not (fig._suptitle and fig._suptitle.get_text())
    assert not any(a.get_title(loc=l) for a in fig.axes for l in ("center", "left", "right"))
    assert fig.texts == []                                    # no group-summary band
    assert all(a.get_legend() is None for a in fig.axes) and not fig.legends
    marks = [t.get_text() for t in _auc_axis(fig).texts if t.get_text().strip()]
    assert marks and set(marks) <= {"*", "**", "***", "n.s."}  # stars, as the lab marks them
    plt.close(fig)


def test_the_default_figure_keeps_its_title_and_band(tmp_path):
    fig, _s, _st = compare_groups(_groups(tmp_path, 3), output_dir=None,
                                  panels={"auc"}, pdf_report=False, auc_plot_style="box_points")
    assert fig._suptitle is not None and fig._suptitle.get_text()
    assert _auc_axis(fig).get_title() == "Area under the MSD curve"
    assert len(fig.texts) == 1 + 3                            # the title + one band entry per group
    plt.close(fig)


# ── the live Analysis tab's quick preview ───────────────────────────────────
def test_the_live_preview_p_value_comes_from_the_report_engine():
    """It ran its own Kruskal–Wallis; with only 'p = …' on show, that would put
    a different p-value on the same comparison than the report and stats cards."""
    from firefly.analysis import fa_circular as fc, fa_stats_config as fsc
    from firefly.ui.controllers.workspace.workspace_figures import _engine_p_label
    from scipy.stats import kruskal
    a = np.array([0.050, 0.061, 0.055, 0.058, 0.052]); b = np.array([0.071, 0.069, 0.083, 0.064, 0.090])
    cfg = fsc.normalize_stats_config({})
    _om, pw = fc._stat_test_n([a, b], ["A", "B"], cfg)
    want = pw[0]["p"]
    assert abs(want - kruskal(a, b).pvalue) > 1e-4               # the two tests disagree here
    from firefly.analysis.fa_compare import format_p
    assert _engine_p_label([a, b], ["A", "B"], {}) == format_p(want)


def test_an_underflowing_p_value_is_not_printed_as_zero():
    from firefly.analysis.fa_compare import format_p
    assert format_p(0.0) == "p < 1e-300"
    assert format_p(0.0312) == "p = 0.031" and format_p(4.2e-5) == "p = 4.20e-05"
    assert format_p(float("nan")) is None


def test_the_live_preview_brackets_each_pair_in_group_colours():
    from matplotlib.colors import to_rgba
    from matplotlib.figure import Figure
    from firefly.analysis import fa_group_figures as gf
    fig = Figure()
    vals = {"A": {"": np.array([1.0, 1.2, 1.1])}, "B": {"": np.array([2.0, 2.1, 1.9])},
            "C": {"": np.array([1.5, 1.6, 1.4])}}
    cols = {"A": "#000000", "B": "#d55e00", "C": "#0072b2"}
    ax = gf.draw_group_comparison(fig, fig.add_gridspec(1, 1)[0], ["A", "B", "C"], vals,
                                  pairs=[(0, 1, "p = 0.001"), (1, 2, "p = 0.020"), (0, 2, "p = 0.030")],
                                  group_colors=cols, theme={"bg": "#fff", "fg": "#222", "grid": "#eee",
                                                            "spine": "#ccc", "muted": "#888"})
    assert sorted(t.get_text() for t in ax.texts) == ["p = 0.001", "p = 0.020", "p = 0.030"]
    dots = [c for c in ax.collections if len(c.get_offsets()) == 3]
    for c, g in zip(dots, "ABC"):
        assert np.allclose(c.get_facecolor()[:, :3], to_rgba(cols[g])[:3])


def test_live_minimal_mode_strips_legends_and_titles():
    pytest.importorskip("PySide6")
    from matplotlib.figure import Figure
    from firefly.ui.controllers.workspace import workspace_figures as wf
    fig = Figure(); ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], label="Control"); ax.legend()
    ax.set_title("Median D"); ax.set_title("n = 3", loc="right"); fig.suptitle("Report")
    wf._RENDER.minimal = True
    try:
        wf._qimage_from_figure(fig)
    finally:
        wf._RENDER.minimal = False
    assert ax.get_legend() is None
    assert not (ax.get_title() or ax.get_title(loc="right") or fig._suptitle.get_text())


def test_the_preferences_toggle_exists_and_resets_off():
    import os
    qml = open(os.path.join(os.path.dirname(__file__), "..", "firefly", "ui", "qml",
                            "PreferencesDialog.qml"), encoding="utf-8").read()
    assert 'Settings.getBool("figures/minimal", false)' in qml
    assert 'Settings.setValue("figures/minimal", c)' in qml
    assert 'Settings.setValue("figures/minimal", false)' in qml       # Restore defaults


def test_across_metric_correction_relabels_every_bracket(tmp_path):
    """The post-pass that swaps in across-metric-corrected p-values used to
    expect one label per panel; each bracketed pair now has its own."""
    groups = _groups(tmp_path, 3)
    plain, _s, st_plain = compare_groups(groups, output_dir=None, panels={"auc", "mob_immob"},
                                         pdf_report=False, auc_plot_style="box_points")
    corr, _s, st_corr = compare_groups(groups, output_dir=None, panels={"auc", "mob_immob"},
                                       pdf_report=False, auc_plot_style="box_points",
                                       stats_config={"across_metric_correction": True})
    a = sorted(t.get_text() for t in _auc_axis(plain).texts)
    b = sorted(t.get_text() for t in _auc_axis(corr).texts)
    assert len(a) == len(b) == 3 and all(P_ONLY.match(t) for t in b)
    plt.close(plain); plt.close(corr)
