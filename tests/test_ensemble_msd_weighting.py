"""The ensemble MSD must be a mean over DISPLACEMENT PAIRS, not over tracks.

The unweighted per-track mean this replaced gave a one-pair track the same
weight as a twenty-pair one.  Because the pair count per track falls with lag
at a rate set by that track's own length and gaps, the effective weighting
drifted with lag and deformed the CURVE'S SHAPE: every MB543B recording, in
both conditions, showed the ensemble MSD going DOWN from lag 4 to lag 5 — a
step that reads as confinement and that propagates into `_msd_auc` and the
comparison's MSD panel.  Nothing raised; the curve was simply the wrong shape.
"""
import numpy as np
import pandas as pd
import pytest

from firefly.analysis.fa_diffusion import compute_msd_and_fit


def _fit(tracks, max_lagtime=6, n_fit=4):
    return compute_msd_and_fit(tracks, 1.0, 1.0, max_lagtime=max_lagtime,
                               n_fit=n_fit, workers=1)


def test_one_short_track_cannot_outweigh_a_long_one():
    """Hand-checkable: 11 zero-displacement pairs and one 100 um^2 pair give a
    lag-1 ensemble of 100/12, not the 50 an equal-weight track mean returns."""
    rows = [(0, f, 0.0, 0.0) for f in range(12)]          # still, 11 pairs at lag 1
    rows += [(1, 0, 0.0, 0.0), (1, 1, 10.0, 0.0)]         # one 10 um jump, 1 pair
    _imsd, emsd, _diff = _fit(pd.DataFrame(
        rows, columns=["particle", "frame", "x", "y"]))

    assert float(emsd.loc[1]) == pytest.approx(100.0 / 12.0, rel=1e-9)
    assert float(emsd.loc[1]) < 50.0, "short track still weighted like a long one"


def test_brownian_ensemble_is_straight_when_track_lengths_differ():
    """The artefact needs only a spread of track lengths — no gaps, no drug.
    A pure Brownian population must give a straight MSD whatever the mix, so a
    lag-dependent weighting shows up as curvature."""
    rng = np.random.default_rng(7)
    D, dt = 0.05, 1.0
    rows = []
    for pid, n in enumerate([6] * 400 + [9] * 200 + [30] * 40):
        step = rng.normal(0.0, np.sqrt(2 * D * dt), (n, 2))
        xy = step.cumsum(axis=0)
        rows.append(pd.DataFrame({"particle": pid, "frame": np.arange(n),
                                  "x": xy[:, 0], "y": xy[:, 1]}))
    _imsd, emsd, _diff = _fit(pd.concat(rows, ignore_index=True), max_lagtime=5)

    m = emsd.to_numpy()
    assert np.all(np.diff(m) > 0), f"ensemble MSD is not monotonic: {m}"
    # MSD = 4*D*t for 2-D Brownian motion; curvature must be ~0, and in
    # particular the lag-4 -> lag-5 step must not collapse.
    expected = 4 * D * dt * np.arange(1, len(m) + 1)
    assert np.allclose(m, expected, rtol=0.12), f"{m} vs {expected}"
    assert abs(np.diff(m, 2)).max() < 0.1 * m[0], f"curvature in a straight MSD: {m}"


def test_gapped_tracks_do_not_bend_the_curve():
    """Blink-stitched tracks contribute pairs at some lags and not others; with
    pair weighting that changes precision, not shape."""
    rng = np.random.default_rng(11)
    D, dt = 0.05, 1.0
    rows = []
    for pid in range(500):
        frames = np.arange(14)
        if pid % 2:                       # drop a blink of up to 3 frames
            drop = rng.integers(3, 8)
            frames = frames[(frames < drop) | (frames > drop + rng.integers(1, 4))]
        step = rng.normal(0.0, np.sqrt(2 * D * dt), (len(frames), 2))
        # build positions on the real time axis so gaps mean real elapsed time
        xy = np.zeros((len(frames), 2))
        for k in range(1, len(frames)):
            gap = frames[k] - frames[k - 1]
            xy[k] = xy[k - 1] + step[k] * np.sqrt(gap)
        rows.append(pd.DataFrame({"particle": pid, "frame": frames,
                                  "x": xy[:, 0], "y": xy[:, 1]}))
    _imsd, emsd, _diff = _fit(pd.concat(rows, ignore_index=True), max_lagtime=6)

    m = emsd.to_numpy()
    assert np.all(np.diff(m) > 0), f"gap-stitched ensemble MSD dips: {m}"


def test_every_lag_reports_the_pairs_behind_it():
    """The weights come from the per-track pair counts; keep them in the
    contract so a future refactor can't silently drop back to a track mean."""
    from firefly.analysis.fa_diffusion import _msd_and_fit_one

    xy = np.column_stack((np.arange(10.0), np.zeros(10)))
    out = _msd_and_fit_one(xy, np.arange(10), 0, np.arange(1, 5) * 1.0, 4, 3)
    assert len(out) == 4, "pair counts dropped from the return contract"
    n_pairs = out[3]
    # a gapless 10-point track has 10-lag pairs at each lag
    np.testing.assert_array_equal(n_pairs, [9, 8, 7, 6])
