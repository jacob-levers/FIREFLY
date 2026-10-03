"""JDD fits stuck at a parameter limit, and where the static offset comes from.

On the MB112C fly recordings every run's JDD fit sat on its bounds (D = 1e-6 and
100 µm²/s, fractions 0.99 / 0.01) and the comparison panel drew those as
populations.  The worker had passed the median MSD intercept as the "known static
offset" — compute_jdd's own docstring says not to, since that intercept carries
the plateau of confined tracks and motion blur — and it exceeded the median
squared jump, so no positive D could fit.  Without it the same data fit cleanly
(R² 0.94–0.98 → 0.9997).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from firefly.analysis.fa_diffusion import (compute_jdd, jdd_fit_at_limit,
                                           jdd_static_offset_um2)

PX, DT = 0.1, 0.02


def _two_pop_tracks(n_tracks=400, n_frames=12, D=(0.03, 0.25), slow_frac=0.4,
                    sigma_um=0.02, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_tracks):
        d = D[0] if rng.random() < slow_frac else D[1]
        xy = np.cumsum(rng.normal(0, np.sqrt(2 * d * DT), (n_frames, 2)), axis=0)
        xy += rng.normal(0, sigma_um, (n_frames, 2))                # static error
        rows += [(p, f, x / PX, y / PX) for f, (x, y) in enumerate(xy)]
    return pd.DataFrame(rows, columns=["particle", "frame", "x", "y"])


def _quiet_jdd(*a, **k):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        return compute_jdd(*a, **k)


# ── the fit says when it failed ─────────────────────────────────────────────
def test_an_offset_bigger_than_the_jumps_is_reported_as_a_failed_fit():
    tr = _two_pop_tracks()
    probe = _quiet_jdd(tr, PX, DT, loc_offset_um2=None)
    too_big = 2.0 * float(np.median(probe["jumps"] ** 2))
    r = _quiet_jdd(tr, PX, DT, loc_offset_um2=too_big)
    assert r["fit_status"] == "at_limit"
    assert jdd_fit_at_limit(r)


def test_a_good_fit_is_marked_ok():
    r = _quiet_jdd(_two_pop_tracks(), PX, DT, loc_offset_um2=None)
    assert r["fit_status"] == "ok"
    assert not jdd_fit_at_limit(r)
    assert 0.01 < r["D_values"][0] < r["D_values"][1] < 1.0


def test_stuck_fits_saved_by_older_runs_are_recognised():
    """Runs analysed before this fix have no fit_status; the values give it away."""
    assert jdd_fit_at_limit({"D_values": [1.0000000000000002e-06, 99.99999999994854],
                             "fractions": [0.945, 0.055], "n_components": 2})
    assert jdd_fit_at_limit({"D_values": [1.0000000008381731e-06, 1.0000000230358566e-05],
                             "fractions": [0.99, 0.01], "n_components": 2})
    assert not jdd_fit_at_limit({"D_values": [0.0416, 0.251], "fractions": [0.41, 0.59],
                                 "n_components": 2})


# ── where the offset comes from ─────────────────────────────────────────────
def test_the_offset_comes_from_localisation_precision():
    tr = _two_pop_tracks(n_tracks=20).assign(loc_sigma_x_nm=20.0, loc_sigma_y_nm=20.0)
    assert jdd_static_offset_um2(tr) == pytest.approx(4 * 0.020 ** 2)


@pytest.mark.parametrize("cols", [{}, {"loc_sigma_x_nm": np.nan, "loc_sigma_y_nm": np.nan}])
def test_no_precision_means_no_offset(cols):
    """Apparent D, labelled as such — not an MSD intercept standing in for it."""
    tr = _two_pop_tracks(n_tracks=20).assign(**cols)
    assert jdd_static_offset_um2(tr) is None


# ── the comparison panel ────────────────────────────────────────────────────
def test_the_comparison_panel_does_not_plot_a_stuck_fit(tmp_path):
    from firefly.analysis.fa_compare import compare_groups
    from test_workspace_data import make_run_folder

    stuck = {"D_values": [1e-06, 100.0], "fractions": [0.95, 0.05], "n_components": 2}
    good = {"D_values": [0.04, 0.25], "fractions": [0.4, 0.6], "n_components": 2}

    def run(stem, jd, seed):
        f = make_run_folder(str(tmp_path), stem, seed=seed)
        with open(os.path.join(f, "firefly_extras", f"{stem}_jdd.json"), "w") as fh:
            json.dump(jd, fh)
        return f
    groups = [{"label": "A", "color": "#000000", "folders": [run("a0", stuck, 1), run("a1", stuck, 2)]},
              {"label": "B", "color": "#d55e00", "folders": [run("b0", good, 3), run("b1", good, 4)]}]
    fig, _s, _st = compare_groups(groups, output_dir=None, panels={"jdd"}, pdf_report=False)
    ax = next(a for a in fig.axes if "JDD" in (a.get_title() or ""))
    ys = np.concatenate([c.get_offsets()[:, 1] for c in ax.collections if len(c.get_offsets())])
    assert ys.size == 4                                   # B's two fits × two populations
    assert np.all((ys > 1e-5) & (ys < 10))
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "2" in texts and "limit" in texts              # says how many were left out, and why
    plt.close(fig)
