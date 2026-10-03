"""A group with one recording is drawn as its dot, not as a box.

A box of one value has zero height: the box, the whiskers and the median all
collapse onto one flat line across the slot, which reads as a rendering
glitch.  (Propofol + Aniline, n = 1, in the MB112C comparison.)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest


def _flat_lines(ax):
    """Horizontal line segments wider than a dot's jitter — box edges, caps, medians."""
    out = []
    for ln in ax.lines:
        x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if len(x) >= 2 and np.all(y == y[0]) and np.ptp(x) > 0.2:
            out.append((float(np.mean(x)), float(y[0])))
    return out


def test_the_report_draws_a_lone_replicate_as_a_dot():
    from firefly.analysis.fa_compare import _bar_with_dots_n, _theme_palette
    fig, ax = plt.subplots()
    _bar_with_dots_n(ax, [np.array([1.0, 1.2, 1.1]), np.array([2.0])], ["A", "B"],
                     ["#0072b2", "#d55e00"], _theme_palette("Light"))
    assert not [x for x, _y in _flat_lines(ax) if abs(x - 1) < 0.5]     # nothing flat at B
    assert any(len(c.get_offsets()) == 1 for c in ax.collections)       # B's dot
    assert [x for x, _y in _flat_lines(ax) if abs(x) < 0.5]             # A keeps its box
    plt.close(fig)


def test_the_live_graph_draws_a_lone_replicate_as_a_dot():
    from matplotlib.figure import Figure
    from firefly.analysis import fa_group_figures as gf
    fig = Figure()
    ax = gf.draw_group_comparison(fig, fig.add_gridspec(1, 1)[0], ["A", "B"],
                                  {"A": {"": np.array([1.0, 1.2, 1.1])}, "B": {"": np.array([2.0])}})
    assert not [x for x, _y in _flat_lines(ax) if abs(x - 1) < 0.5]
    assert any(len(c.get_offsets()) == 1 for c in ax.collections)
