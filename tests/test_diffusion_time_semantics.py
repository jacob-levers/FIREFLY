"""Regression tests for timestamp-aware trajectory analysis semantics.

These intentionally use small deterministic tracks: a missing observation must
change a timestamp-lag calculation, not silently be treated as a row offset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from firefly.analysis.fa_diffusion import (
    _mob_immob_ratio,
    _motion_fractions,
    compute_jdd,
    compute_msd_and_fit,
    compute_mss,
    compute_turning_angles,
    compute_vacf,
    compute_van_hove,
    track_elapsed_durations,
)
from firefly.analysis.fa_enums import GapPolicy


def _tracks(frames, x, *, particle=0):
    return pd.DataFrame({
        "particle": np.full(len(frames), particle, dtype=int),
        "frame": np.asarray(frames, dtype=int),
        "x": np.asarray(x, dtype=float),
        "y": np.zeros(len(frames), dtype=float),
    })


def test_msd_uses_actual_frame_lags_and_contiguous_is_explicit_compatibility():
    """A missing frame must not turn a 2-frame displacement into lag one."""
    tr = _tracks([0, 2, 3], [0.0, 2.0, 3.0])

    all_pairs, _, _ = compute_msd_and_fit(
        tr, 1.0, 1.0, max_lagtime=3, n_fit=3, workers=1)
    contiguous, _, _ = compute_msd_and_fit(
        tr, 1.0, 1.0, max_lagtime=3, n_fit=3, workers=1,
        gap_policy=GapPolicy.CONTIGUOUS)

    assert all_pairs.loc[1, 0] == pytest.approx(1.0)  # frame 2 -> 3
    assert all_pairs.loc[2, 0] == pytest.approx(4.0)  # frame 0 -> 2
    assert all_pairs.loc[3, 0] == pytest.approx(9.0)  # frame 0 -> 3
    assert contiguous.loc[1, 0] == pytest.approx(1.0)
    assert np.isnan(contiguous.loc[2, 0])
    assert np.isnan(contiguous.loc[3, 0])


def test_timestamp_lag_can_exceed_observation_count():
    """Sparse tracks retain a valid long physical lag (frame 0 -> frame 10)."""
    tr = _tracks([0, 10], [0.0, 10.0])
    imsd, _, _ = compute_msd_and_fit(
        tr, 1.0, 1.0, max_lagtime=10, n_fit=5, workers=1)
    assert imsd.loc[10, 0] == pytest.approx(100.0)


def test_empty_input_keeps_the_documented_empty_msd_contract():
    imsd, emsd, diff = compute_msd_and_fit(
        pd.DataFrame(), 1.0, 1.0, max_lagtime=3, n_fit=3, workers=1)
    assert imsd.shape == (3, 0)
    assert emsd.isna().all()
    assert diff.empty
    assert {"fit_status", "mean_link_speed_um_s", "track_duration_s"} <= set(diff.columns)


def test_tracks_are_canonicalized_and_branched_exports_are_rejected():
    """Direct analysis is deterministic for shuffled rows and refuses branches."""
    shuffled = _tracks([2, 0, 1], [3.0, 0.0, 1.0])
    _, _, diff = compute_msd_and_fit(
        shuffled, 1.0, 1.0, max_lagtime=3, n_fit=3, workers=1)
    row = diff.iloc[0]
    assert row["path_length_um"] == pytest.approx(3.0)
    assert row["net_displacement_um"] == pytest.approx(3.0)
    assert row["mean_step_um"] == pytest.approx(1.5)

    branched = _tracks([0, 0, 1], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="branched TrackMate export"):
        compute_msd_and_fit(branched, 1.0, 1.0, workers=1)


def test_gapped_links_and_unit_frame_steps_have_distinct_metrics_and_units():
    """Links span observations; ``mean_step`` is restricted to Δframe == 1."""
    tr = _tracks([0, 2, 3], [0.0, 10.0, 11.0])
    _, _, diff = compute_msd_and_fit(
        tr, 1.0, 0.5, max_lagtime=3, n_fit=3, workers=1)
    row = diff.iloc[0]

    assert row["path_length_um"] == pytest.approx(11.0)
    assert row["mean_link_displacement_um"] == pytest.approx(5.5)
    # 10 µm / (2 * 0.5 s) and 1 µm / (1 * 0.5 s), averaged.
    assert row["mean_link_speed_um_s"] == pytest.approx(6.0)
    assert row["mean_step_um"] == pytest.approx(1.0)
    assert row["n_single_frame_steps"] == 1
    assert row["track_duration_s"] == pytest.approx(1.5)
    assert row["n_observations"] == 3


def test_exact_zero_msd_is_below_resolution_but_linear_geometry_stays_zero():
    tr = _tracks(range(5), [0.0] * 5)
    _, _, diff = compute_msd_and_fit(
        tr, 1.0, 1.0, max_lagtime=5, n_fit=5, workers=1)
    row = diff.iloc[0]

    assert np.isnan(row["D"])
    assert np.isnan(row["alpha"])
    assert row["fit_status"] == "below_resolution"
    assert row["motion"] == "Unknown"
    assert row["MSE"] == pytest.approx(0.0)
    assert row["path_length_um"] == pytest.approx(0.0)
    assert row["mean_link_displacement_um"] == pytest.approx(0.0)
    assert row["mean_link_speed_um_s"] == pytest.approx(0.0)
    assert row["mean_step_um"] == pytest.approx(0.0)
    assert row["net_displacement_um"] == pytest.approx(0.0)
    assert row["directionality_ratio"] == pytest.approx(0.0)

    # A historic/manual D=0 must also stay out of the mobility/log-D contract.
    ratio = _mob_immob_ratio(pd.DataFrame({"D": [0.0, 0.01, 0.10, np.nan]}))
    assert ratio == pytest.approx(1.0)
    fractions = _motion_fractions(pd.DataFrame({
        "motion": ["Unknown", "Brownian", "Confined"],
    }))
    assert fractions == {"Brownian": 0.5, "Confined": 0.5}


def test_below_resolution_checks_all_valid_msd_bins_not_only_fit_window():
    """A later blink-spanning displacement invalidates an otherwise zero prefix."""
    tr = _tracks([0, 1, 2, 3, 4, 5, 20], [0.0] * 6 + [1.0])
    imsd, _, diff = compute_msd_and_fit(
        tr, 1.0, 1.0, max_lagtime=20, n_fit=5, workers=1)
    row = diff.iloc[0]

    assert imsd.loc[1, 0] == pytest.approx(0.0)
    assert imsd.loc[5, 0] == pytest.approx(0.0)
    assert imsd.loc[15, 0] == pytest.approx(1.0)
    assert imsd.loc[20, 0] == pytest.approx(1.0)
    assert row["fit_status"] == "insufficient_lags"
    assert np.isnan(row["D"])


def test_offset_dominated_nonzero_track_has_its_own_status_and_finite_D():
    """Static-error-floor fits are not the same as an exactly zero MSD curve."""
    x = [0.0, 0.01, -0.01, 0.01, -0.01, 0.01,
         -0.01, 0.01, -0.01, 0.01, -0.01, 0.01]
    _, _, diff = compute_msd_and_fit(
        _tracks(range(len(x)), x), 1.0, 1.0, max_lagtime=10, n_fit=5, workers=1)
    row = diff.iloc[0]

    assert row["fit_status"] == "offset_dominated"
    assert np.isfinite(row["D"]) and row["D"] > 0
    assert np.isnan(row["alpha"])
    assert row["motion"] == "Immobile"


def test_singleton_track_has_finite_zero_directionality():
    _, _, diff = compute_msd_and_fit(
        _tracks([4], [2.5]), 1.0, 1.0, max_lagtime=3, n_fit=3, workers=1)
    assert diff.loc[0, "directionality_ratio"] == pytest.approx(0.0)


def test_mss_shares_the_timestamp_gap_policy():
    """The frame-0 gap changes all-pairs moments but not contiguous moments."""
    frames = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    tr = _tracks(frames, [0.0, 20.0, 21.0, 22.0, 23.0,
                          24.0, 25.0, 26.0, 27.0, 28.0])
    all_pairs = compute_mss(tr, 1.0, 1.0, max_lagtime=5)
    contiguous = compute_mss(
        tr, 1.0, 1.0, max_lagtime=5, gap_policy=GapPolicy.CONTIGUOUS)

    assert len(all_pairs) == len(contiguous) == 1
    assert contiguous.loc[0, "mss_slope"] == pytest.approx(1.0)
    assert all_pairs.loc[0, "mss_slope"] > contiguous.loc[0, "mss_slope"]


def test_contiguous_mss_uses_the_longest_uninterrupted_frame_span():
    """Six consecutive frames support four usable contiguous MSS lags."""
    contiguous = compute_mss(
        _tracks(range(6), np.arange(6, dtype=float)), 1.0, 1.0,
        max_lagtime=5, gap_policy=GapPolicy.CONTIGUOUS)
    assert len(contiguous) == 1
    assert contiguous.loc[0, "mss_slope"] == pytest.approx(1.0)


def test_gapless_mss_is_identical_under_both_policies():
    """Gap policy cannot change pair sets or lag horizon without a gap."""
    x = [0.0, 0.7, 1.1, 2.8, 3.0, 5.2, 5.7, 7.9, 9.1, 9.4]
    tr = _tracks(range(len(x)), x)
    all_pairs = compute_mss(
        tr, 1.0, 0.25, max_lagtime=8, gap_policy=GapPolicy.ALL_PAIRS)
    contiguous = compute_mss(
        tr, 1.0, 0.25, max_lagtime=8, gap_policy=GapPolicy.CONTIGUOUS)
    pd.testing.assert_frame_equal(all_pairs, contiguous)


def test_gapless_msd_matches_trackpy_reference():
    """The timestamp estimator reduces to Trackpy's standard gapless IMSD."""
    import trackpy as tp

    tr = pd.concat([
        _tracks(range(6), [0, 1, 2, 3, 4, 5], particle=0),
        _tracks(range(6), [0, 0.5, 1, 2, 3, 5], particle=1),
    ], ignore_index=True)
    ours, _, _ = compute_msd_and_fit(
        tr, 1.0, 1.0, max_lagtime=4, n_fit=4, workers=1)
    reference = tp.imsd(tr, mpp=1.0, fps=1.0, max_lagtime=4)
    np.testing.assert_allclose(ours.to_numpy(), reference.to_numpy())


def test_remaining_public_track_summaries_reject_duplicate_timestamps():
    from firefly.analysis.fa_diffusion import (
        compute_dwell_times,
        compute_mobile_fraction_over_time,
    )

    branched = _tracks([0, 0, 1], [0.0, 1.0, 2.0])
    diff = pd.DataFrame({
        "particle": [0], "D": [0.1], "motion": ["Confined"],
    })
    with pytest.raises(ValueError, match="branched TrackMate export"):
        track_elapsed_durations(branched, 1.0)
    with pytest.raises(ValueError, match="branched TrackMate export"):
        compute_mobile_fraction_over_time(branched, diff, 1.0)
    with pytest.raises(ValueError, match="branched TrackMate export"):
        compute_dwell_times(branched, diff, 1.0)


def test_vacf_lags_velocity_start_timestamps_not_velocity_row_offsets():
    rows = []
    for particle in range(20):
        # Two real unit velocities begin at frames 0 and 3; there is no valid
        # lag-1 or lag-2 velocity pair across the missing frames.
        rows.extend((particle, frame, x, 0.0)
                    for frame, x in zip([0, 1, 3, 4], [0.0, 1.0, 10.0, 11.0]))
    tr = pd.DataFrame(rows, columns=["particle", "frame", "x", "y"])

    vacf = compute_vacf(tr, frame_interval_s=1.0, pixel_size_um=1.0, max_lag=3)
    assert vacf is not None
    assert vacf["n_velocities"] == 40
    assert vacf["n_pairs"].tolist() == [40, 0, 0, 20]
    assert vacf["vacf"][0] == pytest.approx(1.0)
    assert np.isnan(vacf["vacf"][1])
    assert np.isnan(vacf["vacf"][2])
    assert vacf["vacf"][3] == pytest.approx(1.0)
    assert np.isnan(vacf["persistence"])


def test_vacf_includes_two_localisation_tracks_in_its_zero_lag_ensemble():
    rows = []
    for particle in range(20):
        rows.extend([(particle, 0, 0.0, 0.0), (particle, 1, 1.0, 0.0)])
    tr = pd.DataFrame(rows, columns=["particle", "frame", "x", "y"])

    vacf = compute_vacf(tr, frame_interval_s=1.0, pixel_size_um=1.0, max_lag=2)
    assert vacf is not None
    assert vacf["n_velocities"] == 20
    assert vacf["n_pairs"].tolist() == [20, 0, 0]
    assert vacf["vacf"][0] == pytest.approx(1.0)


def test_public_temporal_helpers_share_canonical_duplicate_rejection():
    branched = _tracks([0, 0, 1], [0.0, 1.0, 2.0])
    for call in (
        lambda: compute_jdd(branched, 1.0, 1.0),
        lambda: compute_van_hove(branched, 1.0),
        lambda: compute_turning_angles(branched),
    ):
        with pytest.raises(ValueError, match="branched TrackMate export"):
            call()


def test_turning_angles_uses_canonical_frame_order():
    shuffled = pd.DataFrame({
        "particle": [0, 0, 0],
        "frame": [2, 0, 1],
        "x": [1.0, 0.0, 1.0],
        "y": [1.0, 0.0, 0.0],
    })
    assert compute_turning_angles(shuffled) == pytest.approx([90.0])


def test_elapsed_duration_uses_frame_span_not_observation_count():
    tr = pd.concat([
        _tracks([0, 2, 5], [0.0, 1.0, 2.0], particle=7),
        _tracks([3], [0.0], particle=9),
    ], ignore_index=True)
    assert np.array_equal(np.sort(track_elapsed_durations(tr, 0.5)), [0.0, 2.5])


def test_trackpy_chunk_offsets_follow_the_actual_array_split(monkeypatch):
    """1001 frames split into 3 chunks begins at 0, 334, 668 — never 0,500,1000."""
    from firefly.analysis import fa_localize_backends as backends

    def fake_localise_chunk(chunk, diameter, minmass, percentile, frame_offset):
        frames = np.arange(len(chunk), dtype=int) + frame_offset
        return pd.DataFrame({
            "frame": frames,
            "x": np.zeros(len(frames)),
            "y": np.zeros(len(frames)),
            "mass": np.ones(len(frames)),
        })

    monkeypatch.setattr(backends, "_localise_chunk", fake_localise_chunk)
    monkeypatch.delenv("FIREFLY_FORCE_MP", raising=False)
    result = backends.TrackpyBackend().localise(
        np.zeros((1001, 1, 1), dtype=np.float32),
        diameter=7, minmass=0.1, percentile=64, workers=1, chunk_size=500)

    assert np.array_equal(result["frame"].to_numpy(), np.arange(1001))


def test_worker_prelinked_canonicalization_uses_the_same_branch_error():
    from firefly.firefly_worker import _canonicalize_tracks

    ordered = _canonicalize_tracks(_tracks([2, 0, 1], [3.0, 0.0, 1.0]))
    assert ordered["frame"].tolist() == [0, 1, 2]

    with pytest.raises(ValueError, match="branched TrackMate export"):
        _canonicalize_tracks(_tracks([0, 0, 1], [0.0, 1.0, 2.0]))


# ── the three time-like quantities are documented and genuinely distinct ──────
def test_glossary_disambiguates_the_three_time_quantities():
    """duration (span), observed sampling time (count), dwell (occupancy, +1)
    read almost identically — the glossary must state each formula so a value
    can't be mistaken for another."""
    from firefly.analysis.fa_stats_config import glossary_def
    dur = glossary_def("track duration").lower()
    obs = glossary_def("observed sampling time").lower()
    dwell = glossary_def("dwell time").lower()
    assert dur and obs and dwell
    assert "last frame" in dur and "first frame" in dur      # span, no +1
    assert "+ 1" not in dur
    assert "localisation count" in obs                        # count-based
    assert "+ 1" in dwell and "occupied" in dwell             # occupancy, +1


def test_dwell_is_exactly_one_frame_longer_than_duration():
    """Locks the deliberate off-by-one the glossary explains: dwell counts
    frames OCCUPIED, duration counts the interval SPANNED."""
    import numpy as np, pandas as pd
    from firefly.analysis.fa_diffusion import compute_msd_and_fit, compute_dwell_times
    dt, n = 0.02, 12
    rng = np.random.default_rng(0)
    # Jitter-dominated → classified Immobile, which is the confined/immobile
    # population dwell times are computed for (a perfectly static track is
    # below-resolution and carries no dwell row).  Gapless, so both formulas
    # are unambiguous.
    tr = pd.DataFrame({"particle": 0, "frame": np.arange(n),
                       "x": rng.normal(0, 0.3, n), "y": rng.normal(0, 0.3, n)})
    _i, _e, diff = compute_msd_and_fit(tr, 1.0, dt, max_lagtime=5, n_fit=4, workers=1)
    duration = float(diff.iloc[0]["track_duration_s"])
    assert duration == pytest.approx((n - 1) * dt)            # span = 11 × dt
    dw, _tau = compute_dwell_times(tr, diff, dt)
    assert dw is not None and len(dw), "no dwell row to compare against"
    total = float(dw.iloc[0]["dwell_time_total_s"])
    assert total == pytest.approx(n * dt)                      # occupancy = 12 × dt
    assert total - duration == pytest.approx(dt)               # exactly one frame
