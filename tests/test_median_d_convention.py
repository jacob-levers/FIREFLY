"""Median D must count the molecules that did not move.

A track whose MSD does not rise gets a non-positive linear slope and D = NaN
(`fit_status == "nonpositive_slope"`).  FIREFLY's mobile fraction already counts
those as immobile (`mobility_masks`) — they are about half the immobile
population, not failures.  Median D did not: pandas' `.median()` and the compare
engine's `positive=True` filter both skipped them, so the median described only
the tracks that moved and came out high.  On a PC12 recording analysed alongside
PALMTracer that was 0.128 µm²/s against 0.088 with them counted — the single
largest reason the two programs' medians disagreed.

PALMTracer pins every D ≤ 1e-5 µm²/s at 1e-5 and keeps it; median D now does the
same.  Genuinely unmeasurable rows (below resolution, too few lags) stay out,
exactly as in `mobility_masks`, so the two metrics share one population.
"""
import numpy as np
import pandas as pd
import pytest

from firefly.analysis.fa_diffusion import D_FLOOR, median_d, mobility_masks


def _df(rows):
    return pd.DataFrame(rows, columns=["D", "fit_status"])


def test_non_moving_tracks_count_and_rank_lowest():
    df = _df([(0.1, "fit"), (0.2, "fit"), (0.3, "fit"),
              (np.nan, "nonpositive_slope"), (np.nan, "nonpositive_slope")])
    # old behaviour: median of the three that moved = 0.2
    assert df["D"].median() == pytest.approx(0.2)
    # counted at the floor: [1e-5, 1e-5, 0.1, 0.2, 0.3] → 0.1
    assert median_d(df) == pytest.approx(0.1)


def test_unmeasurable_rows_stay_out_exactly_as_in_the_mobile_fraction():
    df = _df([(0.1, "fit"), (0.3, "fit"),
              (np.nan, "below_resolution"), (np.nan, "insufficient_lags")])
    assert median_d(df) == pytest.approx(0.2)
    mobile, immobile = mobility_masks(df, 0.03)
    assert int(mobile.sum() + immobile.sum()) == 2


def test_it_uses_the_same_population_as_the_mobile_fraction():
    rng = np.random.default_rng(0)
    status = rng.choice(["fit", "nonpositive_slope", "below_resolution", "insufficient_lags"],
                        size=400, p=[0.7, 0.15, 0.05, 0.1])
    d = np.where(status == "fit", 10 ** rng.uniform(-4, 0, 400), np.nan)
    df = _df(list(zip(d, status)))
    mobile, immobile = mobility_masks(df, 0.03)
    counted = mobile | immobile
    expected = np.median(np.maximum(np.where(status == "nonpositive_slope", D_FLOOR, d)[counted], D_FLOOR))
    assert median_d(df) == pytest.approx(expected)


def test_positive_values_below_the_floor_are_pinned_like_palmtracer():
    df = _df([(1e-7, "fit"), (1e-6, "fit"), (0.5, "fit")])
    assert median_d(df) == pytest.approx(D_FLOOR)


def test_a_table_without_fit_status_falls_back_to_measured_values():
    """Older tables cannot say WHY a D is missing, so nothing is assumed."""
    df = pd.DataFrame({"D": [0.1, np.nan, 0.3]})
    assert median_d(df) == pytest.approx(0.2)


def test_nothing_measurable_is_nan_not_an_error():
    assert np.isnan(median_d(_df([(np.nan, "insufficient_lags")])))
    assert np.isnan(median_d(None))
    assert np.isnan(median_d(pd.DataFrame({"x": [1]})))


def test_the_compare_engine_reports_the_same_median():
    """The comparison report's per-replicate median_D must be this helper, not
    a separate positive-only filter."""
    from firefly.analysis import fa_compare
    df = _df([(0.1, "fit"), (0.2, "fit"), (0.3, "fit"),
              (np.nan, "nonpositive_slope"), (np.nan, "nonpositive_slope")])
    assert fa_compare._replicate_median_d(df) == pytest.approx(0.1)
