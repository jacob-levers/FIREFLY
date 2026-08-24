"""Track Length / Total Tracks — the two sampling panels (R and S).

Every per-track number in the figure (D, alpha, the MSS slope) is a fit to a
handful of points, so how many points each track carried — and how many tracks
there are to average over — is part of reading the rest of the figure.  Each
panel is a single bar stating one number, meant to be read straight off.

The invariants pinned here are the ones that actually drift.  The figure's
layout, the params default that says which panels to draw, the Analysis tab's
gallery and the group-averaging allow-list are four separate lists of the same
letters — a panel added to one and missed in another silently disappears from
that surface.  And the UI crops a panel out of the saved combined PNG using the
grid's geometry, so growing the grid must not move the panels already in it,
nor break runs analysed against the previous grid.
"""
import inspect
import re

import numpy as np
import pandas as pd
import pytest

from firefly.analysis import fa_figure as ff


def _drawn_letters():
    """The panel letters make_figure actually draws.  Read from the source: the
    layout table is built inside the function against a live GridSpec, and a
    panel only really exists once it has been handed to ``sax``."""
    body = inspect.getsource(ff.make_figure)
    return set(re.findall(r'sax\(\s*ax\s*,\s*"([A-Z])"', body))


# ── the four lists of letters must agree ─────────────────────────────────────
def test_the_figure_draws_track_length_and_duration():
    assert {"R", "S"} <= _drawn_letters()


def test_every_panel_list_carries_the_same_letters():
    from firefly.ui.controllers.params.params_builder import FIG_PANELS_ALL
    from firefly.ui.controllers.workspace import workspace_data as wd
    from firefly.ui.controllers.workspace import workspace_group_figures as gpf

    drawn = _drawn_letters()
    assert set(FIG_PANELS_ALL) == drawn, "params default is out of step with the figure"
    assert len(FIG_PANELS_ALL) == len(set(FIG_PANELS_ALL))

    gallery = {p.get("letter") or p.get("panel_letter") for p in wd.PANELS}
    assert gallery == drawn, "the All-panels gallery is out of step with the figure"
    assert len(wd.PANELS) == len(drawn)

    # the group-averaging allow-list is a subset (spatial panels can't pool)
    assert gpf.AVERAGEABLE_LETTERS <= drawn
    assert gpf.SPATIAL_LETTERS <= drawn
    assert not (gpf.AVERAGEABLE_LETTERS & gpf.SPATIAL_LETTERS)
    assert {"R", "S"} <= gpf.AVERAGEABLE_LETTERS


def test_the_gallery_names_the_two_new_panels():
    from firefly.ui.controllers.workspace import workspace_data as wd
    by_letter = {p.get("letter"): p for p in wd.PANELS if p.get("letter")}
    assert "track length" in by_letter["R"]["name"].lower()
    assert "tracks" in by_letter["S"]["name"].lower()
    assert by_letter["R"]["kind"] == by_letter["S"]["kind"] == "mfig"


# ── growing the grid must not disturb the panels already in it ───────────────
def test_the_six_row_geometry_is_unchanged():
    """Runs analysed before the grid grew are still browsable, and the UI crops
    their panels out of the saved PNG — so the OLD geometry must still be
    reproduced exactly, not approximated."""
    from firefly.ui.controllers.workspace import workspace_figures as wf
    cells, aspect, gain, pivot = wf._COMBINED_GEOM[6]
    assert aspect == pytest.approx(20.0 / 38.0)
    assert gain == pytest.approx(0.08, abs=5e-4)      # historical constants
    assert pivot == pytest.approx(0.573, abs=5e-4)
    # the historical cell for the cluster map (row 4, col 0)
    assert cells["L"] == pytest.approx((0.06, 0.195818, 0.31, 0.306727), abs=1e-5)


def test_the_new_row_is_appended_below_the_old_ones():
    """Every pre-existing panel keeps its column and its size on the page; the
    figure grew by exactly one row's worth of inches."""
    from firefly.ui.controllers.workspace import workspace_figures as wf
    six, seven = wf._COMBINED_GEOM[6][0], wf._COMBINED_GEOM[7][0]
    assert set(six) == set(seven)
    for l in six:
        x0, yb, x1, yt = six[l]
        n0, nb, n1, nt = seven[l]
        assert (x0, x1) == pytest.approx((n0, n1))                 # columns unmoved
        assert (yt - yb) * 38.0 == pytest.approx((nt - nb) * ff.GRID_H_IN, rel=0.01)
    assert ff.GRID_H_IN == pytest.approx(38.0 / 6.0 * 7)


def test_grid_margins_reproduce_the_historical_values_at_the_old_height():
    assert ff.grid_margins(38.0) == pytest.approx(
        {"left": 0.06, "right": 0.97, "top": 0.95, "bottom": 0.035}, abs=1e-9)


# ── the panels themselves ────────────────────────────────────────────────────
def _run(*, columns=True, frame_interval=0.02, seed=5, want={"R", "S"}):
    rng = np.random.default_rng(seed)
    rows, trows, pid = [], [], 0
    for m in ("Immobile", "Confined", "Brownian", "Directed"):
        for _ in range(40):
            L = int(np.clip(rng.gamma(3.0, 4.0) + 3, 3, 90))
            r = dict(particle=pid, D=0.05, alpha=1.0, motion=m, mss_slope=0.5)
            if columns:
                r["n_observations"] = L
                r["track_duration_s"] = (L - 1) * frame_interval
            rows.append(r)
            for k in range(L):
                trows.append(dict(particle=pid, frame=k,
                                  x=rng.normal(30, 8), y=rng.normal(30, 8)))
            pid += 1
    imsd = pd.DataFrame(np.abs(rng.normal(0.05, 0.01, (20, 20))),
                        index=np.arange(1, 21))
    emsd = pd.Series(np.linspace(0.01, 0.3, 20), index=np.arange(1, 21))
    diff = pd.DataFrame(rows)
    _run.last_diff = diff          # what the panels were actually handed
    return ff.make_figure(rng.random((4, 48, 48)).astype(np.float32),
                          pd.DataFrame(trows), imsd, emsd, diff,
                          0.106, frame_interval, want_panels=want)


def test_both_panels_render_and_are_titled():
    out = _run()
    assert out["panel_titles"]["R"] == "Track Length"
    assert out["panel_titles"]["S"] == "Total Tracks"
    for l in ("R", "S"):
        assert out["panels"][l].size[0] > 100 and out["panels"][l].size[1] > 100


def test_a_run_without_the_per_track_columns_still_renders():
    """palmTRACER caches and older runs have no n_observations /
    track_duration_s column — the panel must say so, not raise."""
    out = _run(columns=False)
    assert set(out["panels"]) == {"R", "S"}


def _bars_by_ylabel(want={"R", "S"}):
    """Run the figure and return {y-axis label: (bar height, yerr)} for the bars
    drawn.  Keyed off the axes rather than call order, because several other
    panels draw bars too."""
    from matplotlib.axes import Axes
    drawn, real = [], Axes.bar

    def spy(self, x, height, *a, **kw):
        res = real(self, x, height, *a, **kw)
        yerr = kw.get("yerr")
        drawn.append((self, float(np.ravel(height)[0]),
                      float(np.ravel(yerr)[0]) if yerr is not None else None))
        return res

    try:
        Axes.bar = spy
        _run(want=want)
    finally:
        Axes.bar = real
    return {ax.get_ylabel(): (h, e) for ax, h, e in drawn if ax.get_ylabel()}


def test_the_length_box_is_built_from_every_track():
    """The box summarises the real per-track population — not a subsample and
    not a pre-trimmed one, so the quartiles and whiskers are the true ones.

    The outlier MARKERS are off (thousands of tracks draw a solid streak that
    reads as individual samples), but that is a drawing choice: nothing may be
    dropped from the data the box is computed from."""
    from matplotlib.axes import Axes
    seen, real = [], Axes.boxplot

    def spy(self, x, *a, **kw):
        seen.append((np.asarray(x[0], dtype=float), kw))
        return real(self, x, *a, **kw)

    try:
        Axes.boxplot = spy
        _run(want={"R"})
    finally:
        Axes.boxplot = real
    assert len(seen) == 1
    drawn, kw = seen[0]
    expected = _run.last_diff["n_observations"].to_numpy(dtype=float)
    assert np.array_equal(np.sort(drawn), np.sort(expected))
    assert kw.get("showmeans") is True, (
        "the average was the original ask — the box must still mark the mean")
    assert kw.get("showfliers") is False, (
        "outlier markers pile into a streak that reads as individual samples")


def test_hiding_the_outlier_markers_still_reports_the_longest_track():
    """A track lasting thousands of frames is a stuck or aggregated emitter, and
    it is the reason to distrust the mean.  Not drawing its marker is fine; not
    reporting it at all would be hiding the finding."""
    from matplotlib.axes import Axes
    seen, real = [], Axes.text

    def spy(self, x, y, txt, *a, **kw):
        seen.append(str(txt))
        return real(self, x, y, txt, *a, **kw)

    try:
        Axes.text = spy
        _run(want={"R"})
    finally:
        Axes.text = real
    longest = int(_run.last_diff["n_observations"].max())
    assert any("longest" in t.lower() and f"{longest:,}" in t for t in seen)


def test_the_length_axis_leaves_the_mean_visible():
    """With the outlier markers hidden the axis is framed on the whiskers — but
    on a heavily skewed population the mean sits ABOVE the upper whisker, and a
    mean marker drawn off the top of the panel is worse than no marker."""
    from matplotlib.axes import Axes
    lims, real = [], Axes.set_ylim

    def spy(self, *a, **kw):
        if a and isinstance(a[0], (int, float)) and len(a) > 1:
            lims.append((float(a[0]), float(a[1])))
        return real(self, *a, **kw)

    # a population whose mean is dragged past the whiskers by one huge track
    heavy = pd.DataFrame({
        "particle": range(200), "D": 0.05, "alpha": 1.0, "motion": "Immobile",
        "n_observations": [10] * 199 + [20000],
        "track_duration_s": [0.18] * 199 + [400.0]})
    imsd = pd.DataFrame(np.zeros((5, 1)), index=np.arange(1, 6))
    emsd = pd.Series(np.linspace(0.01, 0.1, 5), index=np.arange(1, 6))
    tracks = pd.DataFrame({"particle": [0, 0], "frame": [0, 1],
                           "x": [1.0, 2.0], "y": [1.0, 2.0]})
    try:
        Axes.set_ylim = spy
        ff.make_figure(np.zeros((2, 8, 8), np.float32), tracks, imsd, emsd,
                       heavy, 0.106, 0.02, want_panels={"R"})
    finally:
        Axes.set_ylim = real
    mean = float(heavy["n_observations"].mean())
    assert any(lo <= mean <= hi for lo, hi in lims), \
        f"mean {mean:.0f} fell outside every axis range drawn: {lims}"


def test_the_length_panel_shows_the_mean_and_median_apart():
    """Track length is right-skewed: on real recordings the mean runs 1.5-2x the
    median, dragged up by a few stuck emitters.  Both numbers must be on the
    panel, or the reader takes one of them for 'the' track length."""
    from matplotlib.axes import Axes
    seen, real = [], Axes.legend

    def spy(self, *a, **kw):
        for h in kw.get("handles", ()):
            seen.append(str(h.get_label()))
        return real(self, *a, **kw)

    try:
        Axes.legend = spy
        _run(want={"R"})
    finally:
        Axes.legend = real
    joined = " ".join(seen).lower()
    assert "median" in joined and "mean" in joined


def test_the_count_bar_is_the_number_of_trajectories():
    """S is a count, so it stays a bar — a total has no spread to box."""
    bars = _bars_by_ylabel()
    height, err = bars["Trajectories"]
    assert height == pytest.approx(float(len(_run.last_diff)))
    assert err is None, "a total count has no standard error"


def test_a_run_with_no_tracks_at_all_does_not_raise():
    """An empty result is a real outcome (a failed detection pass) — the panels
    must say so rather than blowing up the whole figure render."""
    empty = pd.DataFrame(columns=["particle", "D", "alpha", "motion",
                                  "n_observations", "track_duration_s"])
    imsd = pd.DataFrame(np.zeros((5, 1)), index=np.arange(1, 6))
    emsd = pd.Series(np.linspace(0.01, 0.1, 5), index=np.arange(1, 6))
    tracks = pd.DataFrame({"particle": [0, 0], "frame": [0, 1],
                           "x": [1.0, 2.0], "y": [1.0, 2.0]})
    out = ff.make_figure(np.zeros((2, 8, 8), np.float32), tracks, imsd, emsd,
                         empty, 0.106, 0.02, want_panels={"R", "S"})
    assert set(out["panels"]) == {"R", "S"}
