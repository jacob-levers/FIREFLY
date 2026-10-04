"""The overlaid MSD style carries error bars.

"Group overlaid" drew only each condition's mean curve; error appeared only in
the faceted style.  The lab's figures plot MSD as mean ± s.e.m. between
recordings, so the overlaid curves now carry error bars too — the same
spread as the faceted style (between recordings, by the Analysis tab's Error
setting: SD / SEM / 95% CI).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.container import ErrorbarContainer

from firefly.analysis.fa_group_figures import dispersion, draw_msd

T = np.array([0.02, 0.04, 0.06, 0.08])
A = np.array([[0.010, 0.020, 0.028, 0.035],
              [0.014, 0.025, 0.034, 0.040],
              [0.012, 0.021, 0.030, 0.039]])
B = np.array([[0.008, 0.015, 0.020, 0.024],
              [0.011, 0.018, 0.024, 0.028]])


def _bars(data, err="SEM", groups=("A", "B")):
    fig = plt.figure()
    draw_msd(fig, fig.add_gridspec(1, 1)[0], list(groups), data, T, style="overlaid", err=err)
    ax = fig.axes[0]
    out = {}
    for c in ax.containers:
        if isinstance(c, ErrorbarContainer) and c.has_yerr:
            segs = c.lines[2][0].get_segments()
            out[c.lines[0].get_color()] = np.array([s[1, 1] - s[0, 1] for s in segs]) / 2
    plt.close(fig)
    return out


@pytest.mark.parametrize("err", ["SEM", "SD", "95% CI"])
def test_each_condition_gets_its_spread_between_recordings(err):
    bars = _bars({"A": {"": A}, "B": {"": B}}, err=err)
    halves = sorted(bars.values(), key=lambda v: -v[-1])
    want = sorted([dispersion(A, err), dispersion(B, err)], key=lambda v: -v[-1])
    assert len(halves) == 2
    for got, exp in zip(halves, want):
        assert np.allclose(got, exp)


def test_one_recording_has_no_spread_to_show():
    bars = _bars({"A": {"": A[:1]}}, groups=("A",))
    assert all(np.allclose(v, 0) for v in bars.values())


def test_the_exported_overlaid_panel_has_them(tmp_path):
    from test_per_recording_curves import _uneven
    from firefly.analysis.fa_compare import compute_report, render_report
    rd = compute_report(_uneven(tmp_path))
    fig, _s, _st = render_report(rd, panels={"msd"}, panel_only=True, pdf_report=False,
                                 msd_plot_style="overlaid", msd_err="SEM")
    n = sum(isinstance(c, ErrorbarContainer) for a in fig.axes for c in a.containers)
    assert n == 2
    plt.close(fig)


def test_the_bars_of_different_conditions_do_not_sit_on_each_other():
    """At one time lag both conditions' bars were drawn at the same x, so the
    second hid the first.  Each is nudged by a small fraction of a lag."""
    fig = plt.figure()
    draw_msd(fig, fig.add_gridspec(1, 1)[0], ["A", "B"], {"A": {"": A}, "B": {"": B}}, T,
             style="overlaid", err="SEM")
    ax = fig.axes[0]
    xs = [c.lines[2][0].get_segments()[0][0, 0] for c in ax.containers
          if isinstance(c, ErrorbarContainer)]
    plt.close(fig)
    dt = T[1] - T[0]
    assert abs(xs[0] - xs[1]) > 0.05 * dt                 # apart …
    assert all(abs(x - T[0]) < 0.25 * dt for x in xs)     # … but still on their lag


# ── …or none, by choice ─────────────────────────────────────────────────────
def test_overlaid_without_error_bars_is_an_option():
    fig = plt.figure()
    draw_msd(fig, fig.add_gridspec(1, 1)[0], ["A", "B"], {"A": {"": A}, "B": {"": B}}, T,
             style="overlaid_plain", err="SEM")
    ax = fig.axes[0]
    assert not any(isinstance(c, ErrorbarContainer) and c.has_yerr for c in ax.containers)
    # the curves sit on their lags (nothing to keep apart without bars)
    assert all(np.allclose(l.get_xdata(), T) for l in ax.lines if len(l.get_xdata()) == len(T))
    assert len([l for l in ax.lines if len(l.get_xdata()) == len(T)]) == 2
    plt.close(fig)


def test_the_plain_style_reaches_the_export_and_keeps_the_overlaid_shape(tmp_path):
    from test_per_recording_curves import _uneven
    from firefly.analysis.fa_compare import compute_report, panel_span, render_report
    assert panel_span("msd", 6, msd_style="overlaid_plain") == panel_span("msd", 6, msd_style="overlaid")
    rd = compute_report(_uneven(tmp_path))
    fig, _s, _st = render_report(rd, panels={"msd"}, panel_only=True, pdf_report=False,
                                 msd_plot_style="overlaid_plain")
    axes = [a for a in fig.axes if a.lines]
    assert len(axes) == 1, "fell back to the faceted style"
    assert not any(isinstance(c, ErrorbarContainer) and c.has_yerr for c in axes[0].containers)
    plt.close(fig)


def test_the_preferences_offer_it():
    import os
    qml = open(os.path.join(os.path.dirname(__file__), "..", "firefly", "ui", "qml",
                            "PreferencesDialog.qml"), encoding="utf-8").read()
    assert '"Group overlaid, no error bars"' in qml and '"overlaid_plain"' in qml
