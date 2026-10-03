"""Distribution curves weight each recording equally, mean ± SEM, by default.

Pooling every track let the recordings with the most tracks set a group's
curve: on MB112C one Propofol recording (a quarter of the group's dwells)
supplied 26 of its 35 dwells over 10 s, which read as a Propofol effect.  The
van Swinderen lab's sptPALM papers average per-recording curves and show the
s.e.m. across recordings (n = recordings).  Pooled tracks
stay available (Preferences → Figures → Distribution curves).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PolyCollection

from firefly.analysis.fa_compare import _logd_kde_or_none, compare_groups
from firefly.analysis.fa_figure_common import group_curve, recording_cells


# ── the rule ────────────────────────────────────────────────────────────────
def test_each_recording_counts_once():
    big, small = np.zeros(900), np.ones(100)
    frac_one = lambda v: np.array([np.mean(v == 1)])
    mean, sem = group_curve([big, small], frac_one, per_recording=True)
    assert mean[0] == pytest.approx(0.5) and sem[0] == pytest.approx(0.5)
    pooled, none = group_curve([big, small], frac_one, per_recording=False)
    assert pooled[0] == pytest.approx(0.1) and none is None


def test_one_recording_has_no_band_and_one_array_is_one_recording():
    f = lambda v: np.array([v.mean()])
    assert group_curve([np.arange(5.0)], f)[1] is None
    assert len(recording_cells(np.arange(5.0))) == 1
    assert recording_cells([np.array([np.nan]), None, np.arange(3.0)])[0].tolist() == [0, 1, 2]


# ── the report ──────────────────────────────────────────────────────────────
def _with_tracks(folder, n, p_end, seed):
    """Add a trajectories table: n tracks, lengths ~ 3 + geometric(p_end)."""
    rng = np.random.default_rng(seed)
    stem = os.path.basename(folder)
    rows, f0 = [], 0
    for pid in range(n):
        L = 3 + rng.geometric(p_end)
        rows += [(pid, f0 + k, rng.random(), rng.random()) for k in range(L)]
        f0 += 1
    pd.DataFrame(rows, columns=["particle", "frame", "x", "y"]).to_csv(
        os.path.join(folder, "firefly_extras", f"{stem}_trajectories.csv"), index=False)
    return folder


def _uneven(tmp_path):
    """Group A: one big slow recording with long tracks + one small fast one."""
    from test_workspace_data import make_run_folder
    a = [_with_tracks(make_run_folder(str(tmp_path), "a_big", seed=1, n_tracks=3000, d_centre=0.01), 3000, 0.08, 1),
         _with_tracks(make_run_folder(str(tmp_path), "a_small", seed=2, n_tracks=300, d_centre=0.3), 300, 0.5, 2)]
    b = [_with_tracks(make_run_folder(str(tmp_path), f"b{k}", seed=5 + k, n_tracks=600, d_centre=0.08), 600, 0.2, 5 + k)
         for k in range(2)]
    return [{"label": "A", "color": "#0072b2", "folders": a},
            {"label": "B", "color": "#d55e00", "folders": b}]


def _logd(folder):
    stem = os.path.basename(folder)
    d = pd.read_csv(os.path.join(folder, "firefly_extras", f"{stem}_diffusion_summary.csv"))["D"].to_numpy()
    return np.clip(np.log10(d[d > 0]), -5, 1)


def _line(ax, label):
    return next(l for l in ax.lines if l.get_label() == label)


@pytest.mark.parametrize("weighting", ["recording", "tracks"])
def test_the_log_d_curve(tmp_path, weighting):
    groups = _uneven(tmp_path)
    kw = {} if weighting == "recording" else {"curve_weighting": "tracks"}   # default = recording
    fig, _s, _st = compare_groups(groups, output_dir=None, panels={"logd_dist"}, pdf_report=False,
                                  logd_plot_style="overlaid", **kw)
    ax = next(a for a in fig.axes if a.get_xlabel() == "Log₁₀ diffusion coefficient (µm²/s)")
    line = _line(ax, "A")
    xk = line.get_xdata()
    curves = [_logd_kde_or_none(_logd(f), xk) for f in groups[0]["folders"]]
    want = (np.mean(curves, axis=0) if weighting == "recording"
            else _logd_kde_or_none(np.concatenate([_logd(f) for f in groups[0]["folders"]]), xk))
    assert np.allclose(line.get_ydata(), want, rtol=1e-6, atol=1e-9)
    bands = [c for c in ax.collections if isinstance(c, PolyCollection)]
    if weighting == "recording":            # a SEM band each, never below zero density
        assert len(bands) == 2
        assert all(np.asarray(b.get_paths()[0].vertices)[:, 1].min() >= 0 for b in bands)
        # B's two recordings agree, so at its peak the band is narrow — a band
        # around the curve, not the old fill down to the axis
        yb = _line(ax, "B").get_ydata()
        vb = np.asarray(bands[1].get_paths()[0].vertices)
        at_peak = vb[np.isclose(vb[:, 0], xk[np.argmax(yb)]), 1]
        assert at_peak.min() > 0.5 * yb.max()
    plt.close(fig)


def test_the_other_curves_average_recordings_with_a_band(tmp_path):
    fig, _s, _st = compare_groups(_uneven(tmp_path), output_dir=None, pdf_report=False,
                                  panels={"track_length", "turning_angles"})
    for xlabel in ("Trajectory length (s)", "|Turning angle| (°)"):
        ax = next(a for a in fig.axes if a.get_xlabel() == xlabel)
        assert sum(isinstance(c, PolyCollection) for c in ax.collections) == 2, xlabel
    plt.close(fig)


def test_every_log_d_style_draws_both_ways(tmp_path):
    groups = _uneven(tmp_path)
    for style in ("faceted", "ridgeline", "overlaid", "violin"):
        for w in ("recording", "tracks"):
            fig, _s, _st = compare_groups(groups, output_dir=None, panels={"logd_dist"},
                                          pdf_report=False, logd_plot_style=style, curve_weighting=w)
            assert any(ax.lines or ax.collections for ax in fig.axes), (style, w)
            plt.close(fig)


# ── the live Analysis tab ───────────────────────────────────────────────────
def _live_groups(tmp_path, metric="D"):
    from firefly.ui.controllers.workspace import workspace_data as wd
    out = []
    for g in _uneven(tmp_path):
        runs = [wd.load_run(f) for f in g["folders"]]
        m = wd.METRIC_BY_ID[metric]
        dists = [m.dist(r) for r in runs]
        out.append({"label": g["label"], "color": g["color"], "dists": dists,
                    "dist": np.concatenate(dists), "values": np.array([m.scalar(r) for r in runs])})
    return out


@pytest.mark.parametrize("metric", ["D", "len"])
def test_the_live_curves_follow_the_setting(tmp_path, monkeypatch, metric):
    pytest.importorskip("PySide6")
    from firefly.ui.controllers.workspace import workspace_data as wd, workspace_figures as wf
    seen = []
    monkeypatch.setattr(wf, "_qimage_from_figure", lambda fig: seen.append(fig) or "image")
    groups = _live_groups(tmp_path, metric)
    m = wd.METRIC_BY_ID[metric]
    wf.render_metric(groups, m, logd_style="overlaid", length_style="density")
    wf.render_metric(groups, m, logd_style="overlaid", length_style="density", curve_weighting="tracks")
    per_rec, pooled = seen
    bands = lambda fig: sum(isinstance(c, PolyCollection) for a in fig.axes for c in a.collections)
    lines = lambda fig: [l.get_ydata() for a in fig.axes for l in a.lines if l.get_label() == "A"]
    assert bands(per_rec) >= 2
    assert not np.allclose(lines(per_rec)[0], lines(pooled)[0])


def test_the_controller_reads_the_setting():
    pytest.importorskip("PySide6")
    from types import SimpleNamespace
    from firefly.ui.controllers.workspace.workspace_controller import AnalysisWorkspaceController as WC
    fake = lambda v: SimpleNamespace(_settings=SimpleNamespace(getStr=lambda k, d=None: v if v else d))
    assert WC._curve_weighting(fake(None)) == "recording"            # default
    assert WC._curve_weighting(fake("tracks")) == "tracks"
    assert WC._curve_weighting(fake("nonsense")) == "recording"


def test_the_preference_exists_and_resets_to_per_recording():
    qml = open(os.path.join(os.path.dirname(__file__), "..", "firefly", "ui", "qml",
                            "PreferencesDialog.qml"), encoding="utf-8").read()
    assert 'Settings.getStr("figures/curve_weighting", "recording")' in qml
    assert 'Settings.setValue("figures/curve_weighting", "recording")' in qml
