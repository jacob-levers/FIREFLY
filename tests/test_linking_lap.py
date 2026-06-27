"""Tests for the LAP / Kalman alternative linkers (Qt-free: numpy/scipy/pandas)."""
import numpy as np
import pandas as pd
import pytest

from firefly.analysis.fa_linking_lap import (
    link_trajectories_lap, link_trajectories_kalman, link_trajectories_nn,
    link_trajectories_simple_lap)
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


def test_kalman_gap_closing_horizon():
    # Frames 6..9 are entirely empty, so the gap (5 -> 10) spans 5 frames.  The
    # coast must be gated by frame NUMBER, not by present-frame iteration count;
    # otherwise the lone next-present-frame iteration reads the gap as 1 and the
    # tracker bridges it regardless of max_gap.  Mirrors the NN/LAP horizon tests.
    f = list(range(0, 6)) + list(range(10, 16))
    df = pd.DataFrame({"frame": f, "x": [20.0] * 12, "y": [20.0] * 12})
    assert link_trajectories_kalman(df, 5, max_gap=12, min_len=2)["particle"].nunique() == 1
    assert link_trajectories_kalman(df, 5, max_gap=2, min_len=2)["particle"].nunique() == 2


def test_kalman_rejects_impossible_empty_frame_jump():
    # Detections at frames 0,1,2 then nothing until frame 100.  With max_gap=2 the
    # 98-frame void must NOT be bridged into one particle (would otherwise inflate
    # track length and corrupt the downstream MSD/diffusion estimate).
    df = pd.DataFrame({"frame": [0, 1, 2, 100], "x": [0.0, 1.0, 2.0, 100.0],
                       "y": [0.0, 0.0, 0.0, 0.0]})
    out = link_trajectories_kalman(df, search_range=5, max_gap=2, min_len=2)
    assert int(out["frame"].max()) == 2
    for _pid, g in out.groupby("particle"):
        assert 100 not in set(g["frame"].astype(int))


def test_kalman_coasts_within_gap_with_correct_prediction():
    # A particle moving at v=1px/frame blinks off for frames 3,4 and reappears at
    # frame 5 exactly where constant velocity predicts (x=5).  With max_gap>=3 it
    # must re-link as ONE track — proving the predict step advances by the true
    # frame gap (dt=3), not a single step (which would predict x=3 and miss it).
    df = pd.DataFrame({"frame": [0, 1, 2, 5], "x": [0.0, 1.0, 2.0, 5.0],
                       "y": [0.0, 0.0, 0.0, 0.0]})
    out = link_trajectories_kalman(df, search_range=2.0, max_gap=3, min_len=2)
    assert out["particle"].nunique() == 1
    assert set(out["frame"].astype(int)) == {0, 1, 2, 5}


def test_simple_lap_alias_matches_lap():
    df = _three_static_tracks()
    a = link_trajectories_simple_lap(df, search_range=5, max_gap=3, min_len=2)
    b = link_trajectories_lap(df, search_range=5, max_gap=3, min_len=2)
    assert a["particle"].to_numpy().tolist() == b["particle"].to_numpy().tolist()


def test_nn_recovers_separated_tracks():
    out = link_trajectories_nn(_three_static_tracks(), search_range=5,
                               max_gap=3, min_len=2)
    assert out["particle"].nunique() == 3


def test_nn_gap_closing_horizon():
    f = list(range(0, 6)) + list(range(10, 16))           # gap on frames 6..9
    df = pd.DataFrame({"frame": f, "x": [20.0] * 12, "y": [20.0] * 12})
    assert link_trajectories_nn(df, 5, max_gap=12, min_len=2)["particle"].nunique() == 1
    assert link_trajectories_nn(df, 5, max_gap=2, min_len=2)["particle"].nunique() == 2


def test_nn_determinism():
    gt = _crossing()[["x", "y", "frame"]]
    a = link_trajectories_nn(gt.copy(), search_range=5, max_gap=4, min_len=3)
    b = link_trajectories_nn(gt.copy(), search_range=5, max_gap=4, min_len=3)
    assert a["particle"].to_numpy().tolist() == b["particle"].to_numpy().tolist()


def test_nn_empty_input_safe():
    assert "particle" in link_trajectories_nn(
        pd.DataFrame(columns=["x", "y", "frame"])).columns


def _merge_pair():
    """Segment B ongoing (frames 0–10) + segment A ending at frame 5 whose end
    sits next to B's ongoing point at frame 6 (a merge, not a gap-close)."""
    rows = [(f, 20.0, 20.0) for f in range(11)] + \
           [(f, 22.0, 20.0) for f in range(6)]
    return pd.DataFrame(rows, columns=["frame", "x", "y"])


def _split_pair():
    """Segment A ongoing (frames 0–10) + child C starting at frame 6 next to A's
    ongoing point at frame 5 (a split, not a gap-close)."""
    rows = [(f, 20.0, 20.0) for f in range(11)] + \
           [(f, 22.0, 20.0) for f in range(6, 11)]
    return pd.DataFrame(rows, columns=["frame", "x", "y"])


def _feature_tie():
    """Main track (frames 0–3) then two frame-4 candidates: one CLOSER but with
    a very different mass, one FARTHER but mass-matched.  Distance-only links the
    closer; the feature penalty flips to the mass-matched (farther) one."""
    rows = [(f, 20.0, 20.0, 100.0) for f in range(4)]
    rows.append((4, 20.7, 20.0, 1.0))     # closer (d²=0.49), wrong mass
    rows.append((4, 21.0, 20.0, 100.0))   # farther (d²=1.0), matching mass
    return pd.DataFrame(rows, columns=["frame", "x", "y", "mass"])


def test_full_lap_off_matches_simple_lap():
    """With every flag off, the full LAP is byte-identical to the Simple LAP."""
    for df in (_three_static_tracks(), _crossing()[["x", "y", "frame"]]):
        a = link_trajectories_lap(df.copy(), search_range=5, max_gap=3, min_len=2)
        b = link_trajectories_lap(df.copy(), search_range=5, max_gap=3, min_len=2,
                                  allow_merging=False, allow_splitting=False,
                                  feature_penalty=False)
        assert a["particle"].to_numpy().tolist() == b["particle"].to_numpy().tolist()


def test_full_lap_merge_unions_segments():
    df = _merge_pair()
    off = link_trajectories_lap(df, search_range=5, max_gap=3, min_len=2)
    on = link_trajectories_lap(df, search_range=5, max_gap=3, min_len=2,
                               allow_merging=True)
    assert off["particle"].nunique() == 2
    assert on["particle"].nunique() == 1


def test_full_lap_split_unions_segments():
    df = _split_pair()
    off = link_trajectories_lap(df, search_range=5, max_gap=3, min_len=2)
    on = link_trajectories_lap(df, search_range=5, max_gap=3, min_len=2,
                               allow_splitting=True)
    assert off["particle"].nunique() == 2
    assert on["particle"].nunique() == 1


def test_full_lap_feature_penalty_breaks_tie():
    off = link_trajectories_lap(_feature_tie(), search_range=5, max_gap=2,
                                min_len=2, feature_penalty=False)
    on = link_trajectories_lap(_feature_tie(), search_range=5, max_gap=2,
                               min_len=2, feature_penalty=True,
                               feature_cols=("mass",), penalty_weight=1.0)
    # distance-only links the closer (wrong-mass) point; the penalty flips to the
    # mass-matched one.
    assert abs(float(off[off["frame"] == 4]["x"].iloc[0]) - 20.7) < 0.1
    assert abs(float(on[on["frame"] == 4]["x"].iloc[0]) - 21.0) < 0.1


def test_dispatch_via_link_trajectories():
    gt = _crossing()[["x", "y", "frame"]]
    for lk in ("lap", "kalman", "simple_lap", "full_lap", "nn"):
        out = link_trajectories(gt.copy(), search_range=5, memory=4,
                                min_len=3, linker=lk)
        assert "particle" in out.columns
        assert out["particle"].nunique() == 2
    # SA dispatches too; it is displacement-only so on a perfectly-coincident
    # synthetic crossing it may over-split — just verify it runs and tracks.
    out = link_trajectories(gt.copy(), search_range=5, memory=4, min_len=3,
                            linker="sa")
    assert "particle" in out.columns and out["particle"].nunique() >= 2


def test_empty_input_safe():
    empty = pd.DataFrame(columns=["x", "y", "frame"])
    assert "particle" in link_trajectories_lap(empty).columns
    assert "particle" in link_trajectories_kalman(empty).columns


def test_determinism():
    gt = _crossing()[["x", "y", "frame"]]
    a = link_trajectories_kalman(gt.copy(), search_range=5, max_gap=4, min_len=3)
    b = link_trajectories_kalman(gt.copy(), search_range=5, max_gap=4, min_len=3)
    assert a["particle"].to_numpy().tolist() == b["particle"].to_numpy().tolist()
