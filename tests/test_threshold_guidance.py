"""Evidence for choosing a detection threshold by hand.

Manual minmass previously offered one number — a spot count on one frame — so
the choice was guesswork. These cover the guidance that replaces it, and in
particular the refusal to draw a noise floor the data does not support: on real
MB543B recordings the valley estimate moved between ~0.27 and ~0.47 depending
only on which frames were sampled, at 16, 48 AND 96 frames. A line that moves
that much is worse than no line, because the user aims at it.
"""
import numpy as np
import pytest

from firefly.analysis.fa_localize import estimate_noise_floor, threshold_guidance


def _frames(n_frames=12, n_noise=300, n_signal=60, seed=0):
    """Frames shaped like real data: a low-mass noise flood outnumbering a
    brighter real-spot mode, so the per-frame harvest cap still reaches into
    the noise."""
    rng = np.random.default_rng(seed)
    return [np.r_[10 ** rng.normal(-1.0, 0.12, n_noise),
                  10 ** rng.normal(0.0, 0.18, n_signal)] for _ in range(n_frames)]


# ── the floor, and its refusal ───────────────────────────────────────────────
def test_a_reproducible_valley_is_reported():
    floor, status = estimate_noise_floor(_frames())
    assert floor is not None and "agreed" in status
    assert 0.02 < floor < 1.0, floor


def test_a_unimodal_distribution_yields_no_floor():
    rng = np.random.default_rng(2)
    floor, status = estimate_noise_floor([10 ** rng.normal(0, 0.3, 400)
                                          for _ in range(12)])
    assert floor is None and "no separable noise mode" in status


def test_a_floor_that_moves_between_halves_is_refused():
    """The real failure mode: alternating frames support two different valleys.
    Reporting either one would be a line the user aims at that isn't there."""
    rng = np.random.default_rng(5)
    low = [np.r_[10 ** rng.normal(-1.3, 0.10, 300),
                 10 ** rng.normal(-0.2, 0.15, 60)] for _ in range(6)]
    high = [np.r_[10 ** rng.normal(-0.5, 0.10, 300),
                  10 ** rng.normal(0.7, 0.15, 60)] for _ in range(6)]
    interleaved = [f for pair in zip(low, high) for f in pair]

    floor, status = estimate_noise_floor(interleaved)
    assert floor is None
    assert "no single floor" in status
    assert "sample more frames" not in status, (
        "must not advise collecting more frames — measured on real data, "
        "16/48/96 frames all split the same way")


def test_one_frame_cannot_establish_a_floor():
    floor, status = estimate_noise_floor([np.array([0.1, 0.5, 1.0])])
    assert floor is None and "too few frames" in status


# ── the consequences of a chosen threshold ───────────────────────────────────
def test_counts_and_fractions_describe_the_chosen_threshold():
    masses = np.concatenate(_frames())
    g = threshold_guidance(masses, 0.5, n_frames=12)

    assert g["n_candidates"] == masses.size
    assert g["n_kept"] == int((masses >= 0.5).sum())
    assert g["kept_fraction"] == pytest.approx(g["n_kept"] / masses.size)
    assert g["per_frame_kept"] == pytest.approx(g["n_kept"] / 12)
    assert len(g["counts"]) == len(g["edges"]) - 1, "histogram edges/counts mismatch"


def test_sitting_below_a_supported_floor_is_flagged():
    frames = _frames()
    floor, _ = estimate_noise_floor(frames)
    g = threshold_guidance(np.concatenate(frames), floor * 0.4, n_frames=len(frames),
                           noise_floor=floor)
    assert g["below_noise_floor"]
    assert "noise" in g["warning"] and "spurious" in g["warning"]


def test_no_floor_means_no_below_floor_claim():
    """With no trustworthy floor the panel must not imply one."""
    frames = _frames()
    g = threshold_guidance(np.concatenate(frames), 0.001, n_frames=len(frames),
                           noise_floor=None, floor_status="no separable noise mode")
    assert g["noise_floor"] is None
    assert not g["below_noise_floor"]


def test_a_dense_result_is_flagged_because_precision_degrades():
    """The artefact that drove the whole MB543B reanalysis: density inflates
    every displacement metric through localisation precision."""
    frames = _frames(n_frames=4, n_noise=400, n_signal=200)
    g = threshold_guidance(np.concatenate(frames), 1e-6, n_frames=4)
    assert g["per_frame_kept"] > 100
    assert "precision" in g["warning"]


def test_an_empty_candidate_set_does_not_crash():
    g = threshold_guidance(np.array([]), 0.45, n_frames=3)
    assert g["n_candidates"] == 0 and g["counts"] == [] and g["warning"]


def test_a_threshold_above_everything_says_so():
    g = threshold_guidance(np.concatenate(_frames()), 1e6, n_frames=12)
    assert g["n_kept"] == 0 and "Nothing survives" in g["warning"]
