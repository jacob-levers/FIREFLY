"""Validation: the SA (palmTRACER-style) linker on real palmTRACER localisations
should reproduce palmTRACER's own trajectories about as well as the other
linkers.  Path-gated — skips unless the challenge dataset is present.

The data lives outside the repo (challenge_data/, an untracked download); when it
is available this exercises the full pipeline (PALM-Tracer loc + trc loaders →
SA linker → ISBI tracking metrics).  The threshold is intentionally lenient: the
SA energy constants are not yet calibrated, so the contract here is "SA is no
worse than the nearest-neighbour seed it starts from", not a tuned target.
"""
import io
import contextlib
import os

import pytest

_LOC = os.path.join("challenge_data", "stack", "data", "stack_locPALMTracer.csv")
_TRC = os.path.join("challenge_data", "stack", "data", "stack_trcPALMTracer.csv")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(_LOC) and os.path.exists(_TRC)),
    reason="palmTRACER challenge dataset not present (challenge_data/)")


def _load_window(n_frames=40):
    from firefly.analysis.fa_loaders import load_external_locs
    with contextlib.redirect_stdout(io.StringIO()):
        loc = load_external_locs(_LOC, preset="PALM-Tracer", pixel_size_um=0.1)
        trc = load_external_locs(_TRC, preset="PALM-Tracer", pixel_size_um=0.1)
    loc = loc[loc["frame"] < n_frames].reset_index(drop=True)
    gt = trc[trc["frame"] < n_frames]
    gt = gt.groupby("particle").filter(lambda g: len(g) >= 5).reset_index(drop=True)
    return loc, gt


def test_palmtracer_loaders_emit_expected_columns():
    loc, gt = _load_window()
    assert {"x", "y", "frame", "mass"} <= set(loc.columns)
    assert "particle" not in loc.columns                 # a loc file has no Track
    assert "particle" in gt.columns and gt["particle"].nunique() > 0


def test_sa_reproduces_palmtracer_tracks_and_not_worse_than_nn():
    from firefly.analysis.fa_linking import link_trajectories
    from firefly.bench.metrics import tracking_isbi
    loc, gt = _load_window()
    with contextlib.redirect_stdout(io.StringIO()):
        sa = link_trajectories(loc.copy(), search_range=2, memory=5, min_len=5,
                               linker="sa", link_params={"seed": 0})
        nn = link_trajectories(loc.copy(), search_range=2, memory=5, min_len=5,
                               linker="nn")
    sa_a = tracking_isbi(sa, gt, gate_px=2.0, pixel_size_um=0.1)["alpha"]
    nn_a = tracking_isbi(nn, gt, gate_px=2.0, pixel_size_um=0.1)["alpha"]
    assert sa_a >= 0.5, f"SA alpha {sa_a:.3f} below lenient floor"
    # SA seeds from NN and keeps the best config seen, so it must not regress
    # meaningfully below NN.
    assert sa_a >= nn_a - 0.02, f"SA {sa_a:.3f} worse than NN {nn_a:.3f}"
