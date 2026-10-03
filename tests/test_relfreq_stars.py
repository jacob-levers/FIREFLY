"""Three things the van Swinderen lab's figures do that FIREFLY now offers.

* The diffusion-coefficient distribution as a relative-frequency histogram:
  the fraction of each recording's trajectories in 0.1-log₁₀-unit bins from
  −5 to 1 (Hines & van Swinderen 2021 Fig. 1J; Bademosi et al. 2017 Fig. 2e),
  averaged across recordings ± SEM.  An option beside the density curves.
* Units on the D axis: "Log₁₀ diffusion coefficient (µm²/s)".
* Minimal figures mark significance with stars (* / ** / *** / n.s.) as the
  lab's figures do; the full figure keeps the p-value.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from firefly.analysis.fa_compare import compare_groups

D_AXIS = "Log₁₀ diffusion coefficient (µm²/s)"


def _groups(tmp_path):
    from test_per_recording_curves import _uneven
    return _uneven(tmp_path)


def _logd(folder):
    stem = os.path.basename(folder)
    d = pd.read_csv(os.path.join(folder, "firefly_extras", f"{stem}_diffusion_summary.csv"))["D"].to_numpy()
    return np.clip(np.log10(d[d > 0]), -5, 1)


def _d_axis(fig):
    return next(a for a in fig.axes if a.get_xlabel() == D_AXIS)


# ── relative frequency ──────────────────────────────────────────────────────
@pytest.mark.parametrize("weighting", ["recording", "tracks"])
def test_relative_frequency_is_the_fraction_of_trajectories_per_bin(tmp_path, weighting):
    groups = _groups(tmp_path)
    fig, _s, _st = compare_groups(groups, output_dir=None, panels={"logd_dist"}, pdf_report=False,
                                  logd_plot_style="relfreq", curve_weighting=weighting)
    ax = _d_axis(fig)
    assert ax.get_ylabel() == "Relative frequency (fractions)"
    line = next(l for l in ax.lines if l.get_label() == "A")
    x, y = line.get_xdata(), line.get_ydata()
    assert np.allclose(x, np.round(np.arange(-5, 1.0001, 0.1), 6))     # the lab's bins
    edges = np.concatenate([x - 0.05, [x[-1] + 0.05]])
    fr = [np.histogram(_logd(f), bins=edges)[0] / len(_logd(f)) for f in groups[0]["folders"]]
    want = (np.mean(fr, axis=0) if weighting == "recording"
            else np.histogram(np.concatenate([_logd(f) for f in groups[0]["folders"]]),
                              bins=edges)[0] / sum(len(_logd(f)) for f in groups[0]["folders"]))
    assert np.allclose(y, want) and y.sum() == pytest.approx(1.0)
    has_bars = any(c.get_label().startswith("_") and len(c.get_segments()) == len(x)
                   for c in ax.collections if hasattr(c, "get_segments"))
    assert has_bars == (weighting == "recording")                      # SEM error bars
    plt.close(fig)


def test_the_preferences_offer_relative_frequency():
    qml = open(os.path.join(os.path.dirname(__file__), "..", "firefly", "ui", "qml",
                            "PreferencesDialog.qml"), encoding="utf-8").read()
    assert '"Relative frequency"' in qml and '"relfreq"' in qml


def test_the_live_tab_draws_relative_frequency(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from firefly.ui.controllers.workspace import workspace_data as wd, workspace_figures as wf
    from test_per_recording_curves import _live_groups
    seen = []
    monkeypatch.setattr(wf, "_qimage_from_figure", lambda fig: seen.append(fig) or "image")
    wf.render_metric(_live_groups(tmp_path), wd.METRIC_BY_ID["D"], logd_style="relfreq")
    ax = seen[0].axes[0]
    assert ax.get_ylabel() == "Relative frequency (fractions)" and ax.get_xlabel() == D_AXIS


# ── the D axis carries its units ────────────────────────────────────────────
@pytest.mark.parametrize("style", ["overlaid", "faceted", "ridgeline", "violin", "relfreq"])
def test_every_d_distribution_style_labels_its_units(tmp_path, style):
    fig, _s, _st = compare_groups(_groups(tmp_path), output_dir=None, panels={"logd_dist"},
                                  pdf_report=False, logd_plot_style=style)
    labels = [a.get_xlabel() for a in fig.axes] + [a.get_ylabel() for a in fig.axes]
    assert D_AXIS in labels
    plt.close(fig)


# ── stars in minimal mode ───────────────────────────────────────────────────
def test_star_thresholds():
    from firefly.analysis.fa_compare import significance_label
    assert significance_label(0.2, "stars") == "n.s."
    assert significance_label(0.03, "stars") == "*"
    assert significance_label(0.004, "stars") == "**"
    assert significance_label(2e-5, "stars") == "***"
    assert significance_label(0.03, "p") == "p = 0.030"
    assert significance_label(0.03, "stars", alpha=0.01) == "n.s."    # follows the α setting


def _auc_texts(fig):
    ax = next(a for a in fig.axes if a.get_ylabel().startswith("AUC"))
    return sorted(t.get_text() for t in ax.texts if t.get_text().strip())


def test_minimal_mode_brackets_carry_stars_and_the_full_figure_p(tmp_path):
    from test_compare_pvalue_minimal import _groups as three
    groups = three(tmp_path, 3)
    full, _s, _st = compare_groups(groups, output_dir=None, panels={"auc"}, pdf_report=False,
                                   auc_plot_style="box_points")
    mini, _s, _st = compare_groups(groups, output_dir=None, panels={"auc"}, pdf_report=False,
                                   auc_plot_style="box_points", minimal=True)
    assert all(t.startswith("p ") for t in _auc_texts(full))
    stars = _auc_texts(mini)
    assert len(stars) == 3 and set(stars) <= {"*", "**", "***", "n.s."}
    plt.close(full); plt.close(mini)


def test_the_live_minimal_graph_uses_stars(monkeypatch):
    pytest.importorskip("PySide6")
    from firefly.ui.controllers.workspace import workspace_data as wd, workspace_figures as wf
    seen = []
    monkeypatch.setattr(wf, "_qimage_from_figure", lambda fig: seen.append(fig) or "image")
    groups = [{"label": "A", "color": "#0072b2", "values": np.array([1.0, 1.1, 0.9, 1.05])},
              {"label": "B", "color": "#d55e00", "values": np.array([2.0, 2.1, 1.9, 2.05])}]
    wf.render_metric(groups, wd.METRIC_BY_ID["a"], minimal=True)
    wf.render_metric(groups, wd.METRIC_BY_ID["a"])
    mini, full = ([t.get_text() for t in f.axes[0].texts if t.get_text().strip()] for f in seen)
    assert mini and set(mini) <= {"*", "**", "***", "n.s."}
    assert full and all(t.startswith("p ") for t in full)


@pytest.mark.parametrize("minimal", [False, True])
def test_the_live_graph_keeps_its_chosen_style(monkeypatch, minimal):
    """The star/p switch must not clobber the graph style (violin stayed a box)."""
    pytest.importorskip("PySide6")
    from matplotlib.collections import PolyCollection
    from firefly.ui.controllers.workspace import workspace_data as wd, workspace_figures as wf
    seen = []
    monkeypatch.setattr(wf, "_qimage_from_figure", lambda fig: seen.append(fig) or "image")
    groups = [{"label": "A", "color": "#0072b2", "values": np.array([1.0, 1.1, 0.9, 1.05])},
              {"label": "B", "color": "#d55e00", "values": np.array([2.0, 2.1, 1.9, 2.05])}]
    wf.render_metric(groups, wd.METRIC_BY_ID["a"], group_style="violin", minimal=minimal)
    assert any(isinstance(c, PolyCollection) for c in seen[0].axes[0].collections)   # violin bodies


def _relfreq_axis(cells):
    from firefly.analysis.fa_compare import _render_logd_relfreq, _theme_palette
    fig, ax = plt.subplots()
    _render_logd_relfreq(ax, [("A", "#0072b2", cells, [])], -1.68, _theme_palette("Light"), 0.021)
    return fig, ax


def test_the_axis_spans_where_the_trajectories_are():
    """Not a stretch of the panel flat at zero: the axis starts at the whole
    log unit below the last bin holding ≥ 1% of the peak (on MB112C, −4)."""
    rng = np.random.default_rng(0)
    cells = [np.clip(rng.normal(-1.1, 0.6, 3000), -5, 1) for _ in range(3)]
    fig, ax = _relfreq_axis(cells)
    y = next(l for l in ax.lines if l.get_label() == "A").get_ydata()
    x = next(l for l in ax.lines if l.get_label() == "A").get_xdata()
    first = x[np.argmax(y >= 0.01 * y.max())]                 # ≈ −2.9 for this spread
    lo, hi = ax.get_xlim()
    assert lo == pytest.approx(np.floor(first) - 0.1) and lo > -4.5
    assert hi == pytest.approx(1.1)
    plt.close(fig)


def test_a_pile_of_clipped_trajectories_keeps_the_floor_in_view():
    rng = np.random.default_rng(1)
    v = np.clip(rng.normal(-1.1, 0.6, 3000), -5, 1)
    v[:150] = -5.0                                   # 5% immobile, clipped at the floor
    fig, ax = _relfreq_axis([v, v.copy()])
    assert ax.get_xlim()[0] < -5.0
    plt.close(fig)
