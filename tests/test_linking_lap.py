"""Tests for the LAP / Kalman alternative linkers (Qt-free: numpy/scipy/pandas)."""
import numpy as np
import pandas as pd
import pytest

from firefly.analysis.fa_linking_lap import (
    link_trajectories_lap, link_trajectories_kalman)
from firefly.analysis.fa_linking import link_trajectories


def _three_static_tracks():
    rows = []
    for _pid, (cx, cy) in enumerate([(10, 10), (10, 60), (60, 30)]):
        for f in range(10):
            rows.append((f, float(cx), float(cy)))
    return pd.DataFrame(rows, columns=["frame", "x", "y"])


def _crossing():
    """Two fast tracks crossing at frame 10 (A→right, B→left)."""
    rows = []
    for t in range(20):
        rows.append((t, 10 + 2.0 * t, 30.0, 0))
        rows.append((t, 50 - 2.0 * t, 30.0, 1))
    return pd.DataFrame(rows, columns=["frame", "x", "y", "particle"])


def test_lap_recovers_separated_tracks():
    out = link_trajectories_lap(_three_static_tracks(), search_range=5,
                                max_gap=3, min_len=2)
    assert out["particle"].nunique() == 3


def test_lap_gap_closing_horizon():
    f = list(range(0, 6)) + list(range(10, 16))           # gap on frames 6..9
    df = pd.DataFrame({"frame": f, "x": [20.0] * 12, "y": [20.0] * 12})
    assert link_trajectories_lap(df, 5, max_gap=12, min_len=2)["particle"].nunique() == 1
    assert link_trajectories_lap(df, 5, max_gap=2, min_len=2)["particle"].nunique() == 2


def test_kalman_preserves_identity_through_crossing():
    gt = _crossing()
    est = link_trajectories_kalman(gt[["x", "y", "frame"]].copy(),
                                   search_range=5, max_gap=4, min_len=3)
    assert est["particle"].nunique() == 2
    # No identity swap: each recovered track must be a straight (monotonic-x) line.
    for _pid, g in est.groupby("particle"):
        dx = np.diff(g.sort_values("frame")["x"].to_numpy())
        assert np.all(dx > 0) or np.all(dx < 0), "identity swapped at crossing"


def test_dispatch_via_link_trajectories():
    gt = _crossing()[["x", "y", "frame"]]
    for lk in ("lap", "kalman"):
        out = link_trajectories(gt.copy(), search_range=5, memory=4,
                                min_len=3, linker=lk)
        assert "particle" in out.columns
        assert out["particle"].nunique() == 2


def test_empty_input_safe():
    empty = pd.DataFrame(columns=["x", "y", "frame"])
    assert "particle" in link_trajectories_lap(empty).columns
    assert "particle" in link_trajectories_kalman(empty).columns


def test_determinism():
    gt = _crossing()[["x", "y", "frame"]]
    a = link_trajectories_kalman(gt.copy(), search_range=5, max_gap=4, min_len=3)
    b = link_trajectories_kalman(gt.copy(), search_range=5, max_gap=4, min_len=3)
    assert a["particle"].to_numpy().tolist() == b["particle"].to_numpy().tolist()
