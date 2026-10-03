"""Anomalous-exponent fits that stop on a bound are not α estimates.

On the MB112C fly recordings (median track ≈ 10 points, α fitted from 4 MSD
points with K, α and an offset free, α ∈ [0, 2]) 31% of finite α sat at
α ≥ 1.999 and 29% at 0.001–0.01 — the optimiser's edges, not molecules.  Those
fits were classified Directed / Immobile; simulated purely Brownian tracks came
out 45% "Directed".  They now carry alpha_fit_status "at_limit", α = NaN and no
motion class.  The displacement rule (an MSD that never rises above its own
floor → Immobile) is unchanged: it does not use α.
"""
import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from firefly.analysis.fa_diffusion import ALPHA_LIMIT_TOL, compute_msd_and_fit

PX, DT, SIG = 0.1, 0.02, 0.025


def _fit(tracks):
    with contextlib.redirect_stdout(io.StringIO()):
        _imsd, _emsd, diff = compute_msd_and_fit(tracks, PX, DT, max_lagtime=10, n_fit=4, workers=1)
    return diff


def _brownian(n_tracks=800, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_tracks):
        n = 8 + rng.geometric(0.25) - 1                 # ≥ 8 points, median ≈ 10
        D = rng.choice([0.03, 0.25])
        xy = np.cumsum(rng.normal(0, np.sqrt(2 * D * DT), (n, 2)), axis=0) + rng.normal(0, SIG, (n, 2))
        rows += [(p, f, x / PX, y / PX) for f, (x, y) in enumerate(xy)]
    return pd.DataFrame(rows, columns=["particle", "frame", "x", "y"])


@pytest.fixture(scope="module")
def brownian_fit():
    return _fit(_brownian())


def test_no_reported_alpha_sits_on_a_limit(brownian_fit):
    """Nothing is reported at the lower bound; at the upper bound only fits whose
    best value genuinely is 2 survive (see the ballistic test) — a handful, where
    ~20% of these tracks sat there before."""
    a = brownian_fit["alpha"].to_numpy()
    assert np.isfinite(a).sum()
    assert not np.any(a[np.isfinite(a)] < ALPHA_LIMIT_TOL)
    assert np.mean(a > 2.0 - ALPHA_LIMIT_TOL) < 0.02


def test_a_fit_stuck_at_a_limit_gets_no_motion_class(brownian_fit):
    stuck = brownian_fit[brownian_fit["alpha_fit_status"] == "at_limit"]
    assert len(stuck) > 0.1 * len(brownian_fit)          # it is common at these lengths
    assert set(stuck["motion"]) == {"Unknown"}
    assert stuck["alpha"].isna().all()


def test_brownian_tracks_are_no_longer_mostly_directed(brownian_fit):
    """45% of these purely Brownian tracks were 'Directed' before."""
    assert (brownian_fit["motion"] == "Directed").mean() < 0.30


def test_a_truly_ballistic_track_keeps_its_alpha_of_2():
    """Constant velocity: the MSD rises exactly as t², so the best fit IS α = 2.
    That is a real exponent — refit with the bound lifted, it settles at 2 — not
    a fit stopped by the bound, so it stays Directed."""
    t = np.arange(12)
    xy = np.c_[0.02 * t, np.zeros_like(t, dtype=float)] / PX
    row = _fit(pd.DataFrame({"particle": 0, "frame": t, "x": xy[:, 0], "y": xy[:, 1]})).iloc[0]
    assert row["alpha_fit_status"] == "descriptive_fit"
    assert row["alpha"] == pytest.approx(2.0, abs=1e-3) and row["motion"] == "Directed"


def test_stuck_fits_at_the_upper_bound_wanted_to_go_past_it(brownian_fit):
    """The rule behind 'at_limit' at α = 2: lifting the bound sends the exponent past 2."""
    from firefly.analysis.fa_diffusion import _alpha_wants_past_two, msd_anomalous
    from scipy.optimize import curve_fit
    t = DT * np.arange(1, 5)
    m = np.array([0.0040, 0.0045, 0.0090, 0.0200])          # convex beyond t²
    popt, _ = curve_fit(msd_anomalous, t, m, p0=[0.01, 1.0, 0.002],
                        bounds=([0, 0, -np.inf], [np.inf, 2.0, np.inf]))
    assert popt[1] == pytest.approx(2.0, abs=ALPHA_LIMIT_TOL)
    assert _alpha_wants_past_two(t, m, popt)


def test_a_track_that_never_moves_is_still_immobile_by_displacement():
    rng = np.random.default_rng(5)
    n = 15
    xy = rng.normal(0, SIG, (n, 2)) / PX
    diff = _fit(pd.DataFrame({"particle": 0, "frame": np.arange(n), "x": xy[:, 0], "y": xy[:, 1]}))
    assert diff.iloc[0]["motion"] == "Immobile"
