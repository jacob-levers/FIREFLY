"""Tests for the simulated-annealing (palmTRACER-style) linker."""
import numpy as np
import pandas as pd

from firefly.analysis.fa_linking_sa import link_trajectories_sa


def _three_static_tracks(n=12):
    rows = []
    for cx, cy in [(10, 10), (10, 60), (60, 30)]:
        for f in range(n):
            rows.append((f, float(cx), float(cy)))
    return pd.DataFrame(rows, columns=["frame", "x", "y"])


def _two_moving(n=15):
    """Two separated tracks both moving in x (no crossing / coincidence) — the
    regime SA is meant for; perfectly-coincident crossings are degenerate for a
    displacement-only energy and don't occur in real localisation data."""
    rows = []
    for t in range(n):
        rows.append((t, 10 + 1.5 * t, 20.0))
        rows.append((t, 10 + 1.5 * t, 45.0))
    return pd.DataFrame(rows, columns=["frame", "x", "y"])


def test_sa_recovers_separated_tracks():
    out = link_trajectories_sa(_three_static_tracks(), search_range=3,
                               max_gap=2, min_len=3, seed=0)
    assert out["particle"].nunique() == 3
    # every original point linked into a length-12 track
    assert len(out) == 36


def test_sa_recovers_two_moving_tracks():
    out = link_trajectories_sa(_two_moving(), search_range=5, max_gap=2,
                               min_len=3, seed=0)
    assert out["particle"].nunique() == 2
    assert len(out) == 30                       # all points linked, none dropped


def test_sa_gap_closing():
    f = list(range(0, 6)) + list(range(10, 16))           # gap on frames 6..9
    df = pd.DataFrame({"frame": f, "x": [20.0] * 12, "y": [20.0] * 12})
    # gap = 10 - 5 = 5: closes when max_gap >= 5, not when max_gap < 5.
    assert link_trajectories_sa(df, 5, max_gap=6, min_len=2, seed=0
                                )["particle"].nunique() == 1
    assert link_trajectories_sa(df, 5, max_gap=3, min_len=2, seed=0
                                )["particle"].nunique() == 2


def test_sa_determinism():
    df = _two_moving()
    a = link_trajectories_sa(df.copy(), search_range=5, max_gap=2, min_len=3, seed=0)
    b = link_trajectories_sa(df.copy(), search_range=5, max_gap=2, min_len=3, seed=0)
    assert a["particle"].to_numpy().tolist() == b["particle"].to_numpy().tolist()
    assert a["x"].to_numpy().tolist() == b["x"].to_numpy().tolist()


def test_sa_empty_input_safe():
    assert "particle" in link_trajectories_sa(
        pd.DataFrame(columns=["x", "y", "frame"])).columns


def test_sa_dense_cluster_completes_and_is_valid():
    """A dense cluster drives many SWAP moves.  The linker must terminate and
    return a VALID partition — no frame repeated within a track.  A cycle
    introduced by a bad swap would either hang the chain trace or surface as a
    repeated node; the swap guard + visited-set trace prevent both."""
    rng = np.random.default_rng(0)
    rows = []
    for f in range(12):
        for _ in range(8):                       # 8 tightly-packed points/frame
            rows.append((f, float(rng.uniform(20, 26)), float(rng.uniform(20, 26))))
    df = pd.DataFrame(rows, columns=["frame", "x", "y"])
    out = link_trajectories_sa(df, search_range=5, max_gap=2, min_len=2, seed=0)
    assert "particle" in out.columns
    for pid, g in out.groupby("particle"):
        fr = g["frame"].to_numpy()
        assert len(np.unique(fr)) == len(fr), f"track {pid} repeats a frame (cycle?)"
