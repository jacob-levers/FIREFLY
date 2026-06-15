"""Tests for the auto search-range estimator (fa_linking_auto)."""
import numpy as np
import pandas as pd
import pytest

from firefly.analysis.fa_linking_auto import estimate_link_params


def _walks(n_tracks=24, n_frames=30, step=1.5, H=120, W=120, seed=0):
    """Well-separated random walks with a known per-frame step scale."""
    rng = np.random.default_rng(seed)
    starts = rng.uniform(12, H - 12, (n_tracks, 2))
    rows = []
    for p, (x0, y0) in enumerate(starts):
        x, y = float(x0), float(y0)
        for f in range(n_frames):
            x += rng.normal(0, step); y += rng.normal(0, step)
            rows.append((f, x, y))
    return pd.DataFrame(rows, columns=["frame", "x", "y"])


def test_estimate_link_params_picks_sane_range():
    """For ~1.5 px steps in sparse data the estimate should land a few px (cover
    the motion) and well below the inter-spot spacing — never the grid floor or a
    huge value."""
    sr, diag = estimate_link_params(_walks(step=1.5), memory=3, min_len=3)
    assert 2.0 <= sr <= 9.0, (sr, diag)
    assert diag["method"] == "meanlen_plateau"
    assert sr < diag["spacing_px"]                 # below the cross-link scale


def test_estimate_link_params_scales_with_motion():
    """Faster motion → larger estimated search range."""
    slow, _ = estimate_link_params(_walks(step=0.8, seed=1), memory=3, min_len=3)
    fast, _ = estimate_link_params(_walks(step=3.0, seed=1), memory=3, min_len=3)
    assert fast > slow


def test_estimate_link_params_deterministic():
    df = _walks(step=2.0, seed=2)
    a, _ = estimate_link_params(df, memory=3, min_len=3)
    b, _ = estimate_link_params(df, memory=3, min_len=3)
    assert a == b


def test_estimate_link_params_edge_cases():
    assert estimate_link_params(pd.DataFrame(columns=["x", "y", "frame"]))[0] == 5.0
    # too few localisations → safe default, no crash
    tiny = pd.DataFrame({"frame": [0, 1], "x": [1.0, 1.1], "y": [1.0, 1.1]})
    assert estimate_link_params(tiny)[0] == 5.0
    with pytest.raises(ValueError):
        estimate_link_params(pd.DataFrame({"x": [1.0], "y": [1.0]}))  # no 'frame'
