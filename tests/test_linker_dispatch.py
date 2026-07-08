"""Linker dispatch / registry regression tests.

Covers the audit fixes:
  * B1 — every registry linker must run with the GUI's default ``link_params``
    (which always carries ``allow_merging``/``allow_splitting`` booleans); the SA
    adapter used to forward those into ``link_trajectories_sa`` and crash.
  * B2 — the ``nn`` linker is canonical TrackMate Nearest-neighbour: strictly
    frame-to-frame, it must NOT bridge a one-frame gap.
  * B3 — the forward default linker is ``fa_enums.DEFAULT_LINKER`` ("trackpy"),
    matching the GUI/README.
  * B4 — the feature-penalty multiplier is the FIREFLY-specific form
    ``P = 1 + Σ w·|a−b|/(a+b)`` (no ×3 TrackMate factor).
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from firefly.analysis import fa_enums
from firefly.analysis.fa_enums import Linker
from firefly.analysis.fa_linking import link_trajectories


def _two_track_locs():
    """Two well-separated particles drifting slowly over frames 0–5."""
    rows = []
    for f in range(6):
        rows.append({"frame": f, "x": 5.0 + 0.1 * f, "y": 5.0 + 0.1 * f,
                     "mass": 100.0})
        rows.append({"frame": f, "x": 25.0 - 0.1 * f, "y": 25.0 + 0.1 * f,
                     "mass": 120.0})
    return pd.DataFrame(rows)


# ── B1: every linker survives the GUI's default link_params ───────────────────
# The GUI injects these for EVERY linker (ui_mixin_build.py); the SA adapter must
# not choke on the Full-LAP-only keys.
_GUI_DEFAULT_LINK_PARAMS = {"allow_merging": False, "allow_splitting": False}


@pytest.mark.parametrize("token", [m.value for m in Linker])
def test_every_linker_runs_with_gui_default_link_params(token):
    locs = _two_track_locs()
    out = link_trajectories(
        locs, search_range=5, memory=3, min_len=2,
        linker=token, link_params=dict(_GUI_DEFAULT_LINK_PARAMS))
    assert "particle" in out.columns
    # A clean two-track field should yield at least one trajectory for any linker.
    assert out["particle"].nunique() >= 1


def test_sa_linker_does_not_crash_on_merge_split_keys():
    """Direct regression for the exact TypeError the audit found."""
    locs = _two_track_locs()
    out = link_trajectories(
        locs, search_range=5, memory=3, min_len=2, linker="sa",
        link_params={"allow_merging": False, "allow_splitting": False,
                     "C_merge": 1.0, "C_split": 1.0})
    assert "particle" in out.columns


# ── B2: nn is strictly frame-to-frame (no gap bridging) ───────────────────────
def test_nn_does_not_bridge_a_one_frame_gap():
    # One spatial location present at frames 0,1,(gap 2),3,4.
    locs = pd.DataFrame({
        "frame": [0, 1, 3, 4],
        "x": [5.0, 5.0, 5.0, 5.0],
        "y": [5.0, 5.0, 5.0, 5.0],
        "mass": [100.0, 100.0, 100.0, 100.0],
    })
    out = link_trajectories(locs, search_range=2, memory=3, min_len=2,
                            linker="nn")
    # Canonical NN ignores `memory`: the frame-2 gap is NOT bridged, so the
    # detections split into two separate 2-point tracks.
    assert out["particle"].nunique() == 2
    for _pid, grp in out.groupby("particle"):
        frames = set(grp["frame"])
        assert not (frames & {0, 1} and frames & {3, 4})  # no track spans the gap


def test_lap_does_bridge_the_gap_for_contrast():
    """The gap-closing linkers SHOULD bridge the same gap — confirms the nn
    result above is a real behavioural difference, not an artefact of the data."""
    locs = pd.DataFrame({
        "frame": [0, 1, 3, 4],
        "x": [5.0, 5.0, 5.0, 5.0],
        "y": [5.0, 5.0, 5.0, 5.0],
        "mass": [100.0, 100.0, 100.0, 100.0],
    })
    out = link_trajectories(locs, search_range=2, memory=3, min_len=2,
                            linker="simple_lap")
    assert out["particle"].nunique() == 1


# ── B3: forward default linker is trackpy (matches GUI/README) ────────────────
def test_default_linker_constant_is_trackpy():
    assert fa_enums.DEFAULT_LINKER == "trackpy"
    assert fa_enums.DEFAULT_LINKER == Linker.TRACKPY.value


def test_link_trajectories_forward_default_is_default_linker():
    default = inspect.signature(link_trajectories).parameters["linker"].default
    assert default == fa_enums.DEFAULT_LINKER == "trackpy"


def test_default_linker_resolves_to_trackpy_adapter():
    from firefly.analysis.fa_linking_registry import _resolve_linker
    assert _resolve_linker(fa_enums.DEFAULT_LINKER).name == "trackpy"


# ── B4: feature-penalty value is the FIREFLY form (no ×3) ─────────────────────
def test_feature_penalty_value_is_firefly_form():
    from firefly.analysis.fa_linking_lap import _feature_penalty
    fi = np.array([[100.0]])
    fj = np.array([[300.0]])
    P = _feature_penalty(fi, fj, weight=1.0)
    # term = |100-300|/(100+300) = 0.5 ; P = 1 + 1.0*0.5 = 1.5 (NOT 1 + 3*0.5).
    assert P is not None
    assert P.shape == (1, 1)
    assert P[0, 0] == pytest.approx(1.5)


def test_feature_penalty_disabled_returns_none():
    from firefly.analysis.fa_linking_lap import _feature_penalty
    assert _feature_penalty(np.array([[1.0]]), np.array([[2.0]]), weight=0) is None
