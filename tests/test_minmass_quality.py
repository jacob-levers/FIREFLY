"""Fast regressions for the Quality-first minmass/detection-QC contract."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from firefly.analysis.fa_minmass_quality import (
    filter_candidates_to_roi,
    full_detection_diagnostics,
    make_spatial_null,
    sampled_detection_diagnostics,
    select_quality_threshold,
)


def _row(t, null, *, n_good=20, n_surv=100, good_fraction=0.5):
    return {
        "t": float(t), "N_good": int(n_good), "n_surv": int(n_surv),
        "good_fraction": float(good_fraction), "spurious_rate": 0.0,
        "null_good_fraction_upper": float(null),
        "good_detection_yield": float(n_surv * good_fraction),
    }


def test_quality_selector_never_goes_below_locked_floor():
    sweep = [_row(0.10, 0.30), _row(0.20, 0.08),
             _row(0.30, 0.07), _row(0.40, 0.05)]
    picked = select_quality_threshold(
        sweep, quality_floor=0.20, max_null_fraction=0.10)
    assert picked.status == "valid"
    assert picked.threshold == 0.20

    raised_floor = select_quality_threshold(
        sweep, quality_floor=0.25, max_null_fraction=0.10)
    assert raised_floor.status == "valid"
    assert raised_floor.threshold == 0.30
    assert raised_floor.threshold >= 0.25


def test_quality_selector_marks_an_unstable_single_point_unresolved():
    sweep = [_row(0.20, 0.20), _row(0.30, 0.08),
             _row(0.40, 0.20), _row(0.50, 0.20)]
    picked = select_quality_threshold(
        sweep, quality_floor=0.20, max_null_fraction=0.10)
    assert picked.status == "unresolved"
    assert picked.reason == "no_stable_quality_plateau"
    assert picked.threshold >= 0.20


def test_outside_roi_candidates_cannot_enter_quality_population():
    roi = np.zeros((10, 10), dtype=bool)
    roi[:, :5] = True
    inside = pd.DataFrame({
        "x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0],
        "frame": [0, 0, 1], "mass": [0.5, 0.6, 0.7],
    })
    outside = pd.DataFrame({
        "x": np.full(1000, 8.0), "y": np.full(1000, 8.0),
        "frame": np.arange(1000) % 2, "mass": np.full(1000, 99.0),
    })
    a = filter_candidates_to_roi(inside, roi)
    b = filter_candidates_to_roi(pd.concat([inside, outside]), roi)
    pd.testing.assert_frame_equal(a, b)


def test_spatial_null_is_deterministic_and_preserves_frames_masses_and_roi():
    roi = np.zeros((20, 30), dtype=bool)
    roi[2:18, 3:27] = True
    h = pd.DataFrame({
        "x": np.tile([5.0, 10.0, 20.0], 20),
        "y": np.tile([5.0, 12.0, 15.0], 20),
        "frame": np.repeat(np.arange(20), 3),
        "mass": np.tile([0.4, 0.7, 1.0], 20),
        "window_id": np.zeros(60, dtype=int),
    })
    n1 = make_spatial_null(
        h, frame_shape=roi.shape, roi_mask=roi, search_range=3, seed=42)
    n2 = make_spatial_null(
        h, frame_shape=roi.shape, roi_mask=roi, search_range=3, seed=42)
    pd.testing.assert_frame_equal(n1, n2)
    np.testing.assert_array_equal(n1["frame"], h["frame"])
    np.testing.assert_array_equal(n1["mass"], h["mass"])
    xi = np.floor(n1["x"].to_numpy() + 0.5).astype(int)
    yi = np.floor(n1["y"].to_numpy() + 0.5).astype(int)
    assert roi[yi, xi].all()
    assert not np.allclose(n1[["x", "y"]], h[["x", "y"]])


def test_sample_density_uses_physical_roi_area_and_scales_with_pixel_size():
    h = pd.DataFrame({
        "mass": np.ones(40),
        "window_id": np.repeat([0, 1], 20),
        "frame": np.tile(np.repeat(np.arange(10), 2), 2),
    })
    d1 = sampled_detection_diagnostics(
        h, threshold=0.5, windows=[(0, 10), (20, 30)],
        roi_area_pixels=100, pixel_size_um=0.1)
    d2 = sampled_detection_diagnostics(
        h, threshold=0.5, windows=[(0, 10), (20, 30)],
        roi_area_pixels=100, pixel_size_um=0.2)
    assert d1["sample_mean_per_frame"] == 2.0
    assert d2["sample_density_per_um2_frame"] == (
        d1["sample_density_per_um2_frame"] / 4.0)


def test_full_run_density_includes_zero_frames_and_reports_temporal_shift():
    # 4, 0, 4, 0 detections over a 2 µm² ROI -> 1 loc/(µm²·frame).
    locs = pd.DataFrame({
        "frame": [0] * 4 + [2] * 4,
        "x": [1, 2, 3, 4, 1, 2, 3, 4],
        "y": [1] * 8,
    })
    roi = np.ones((10, 20), dtype=bool)
    d = full_detection_diagnostics(
        locs, n_frames=4, frame_shape=roi.shape, pixel_size_um=0.1,
        roi_mask=roi, search_range_px=1.0, temporal_bins=2)
    assert d["full_run_mean_per_frame"] == 2.0
    assert d["full_run_zero_frame_fraction"] == 0.5
    assert d["full_run_density_per_um2_frame"] == 1.0
    assert d["temporal_bin_mean_per_frame"] == [2.0, 2.0]

    drifting = pd.DataFrame({
        "frame": np.repeat(np.arange(8), [8, 8, 8, 8, 1, 1, 1, 1]),
        "x": 1.0, "y": 1.0,
    })
    q = full_detection_diagnostics(
        drifting, n_frames=8, frame_shape=roi.shape, pixel_size_um=0.1,
        roi_mask=roi, search_range_px=1.0, temporal_bins=2)
    assert q["temporal_last_first_ratio"] == 0.125
    assert "temporal_density_shift" in q["qc_codes"]


def test_full_run_local_ambiguity_not_global_count_certifies_linkability():
    # Frame 0 points each see two frame-1 successors inside r=1 px.
    locs = pd.DataFrame({
        "frame": [0, 0, 1, 1, 1, 1],
        "x": [5.0, 15.0, 4.8, 5.2, 14.8, 15.2],
        "y": [5.0, 15.0, 5.0, 5.0, 15.0, 15.0],
    })
    d = full_detection_diagnostics(
        locs, n_frames=2, frame_shape=(20, 20), pixel_size_um=0.1,
        roi_mask=None, search_range_px=1.0)
    assert d["next_frame_ambiguous_successor_fraction"] == 1.0
    assert "high_local_assignment_ambiguity" in d["qc_codes"]


def test_quality_estimator_uses_exact_roi_population_and_no_quota_cap(monkeypatch):
    """Exercise the estimator branch, not only its pure selector helper."""
    import firefly.analysis.fa_localize as localize

    stack = np.zeros((100, 20, 20), dtype=np.float32)
    roi = np.zeros((20, 20), dtype=bool)
    roi[:, :10] = True
    inside_n = 400
    inside = pd.DataFrame({
        "x": np.resize(np.arange(1, 9, dtype=float), inside_n),
        "y": np.resize(np.arange(1, 19, dtype=float), inside_n),
        "frame": np.arange(inside_n) % 100,
        "mass": np.linspace(0.16, 1.0, inside_n),
        "window_id": 0,
    })
    outside = pd.DataFrame({
        "x": 15.0,
        "y": 10.0,
        "frame": np.arange(1200) % 100,
        "mass": 99.0,
        "window_id": 0,
    })
    harvested = pd.concat([inside, outside], ignore_index=True)

    class _Backend:
        name = "torch"

    monkeypatch.setattr(localize, "_resolve_backend", lambda _name: _Backend())
    monkeypatch.setattr(localize, "_contiguous_windows", lambda _n: [(0, 100)])
    monkeypatch.setattr(
        localize, "_harvest_windows",
        lambda *args, **kwargs: (harvested.copy(), None))
    monkeypatch.setattr(
        localize, "_noise_floor_valley", lambda _m: (0.12, 1.0, 0.5))

    def _null(h, **kwargs):
        out = h.copy()
        out["_is_null"] = True
        return out

    def _sweep(h, grid, *_args):
        null = "_is_null" in h.columns
        return [{
            "t": float(t), "n_surv": 300, "N_good": 20,
            "good_fraction": 0.05 if null else 0.50,
            "spurious_rate": 0.0, "median_ep": float("nan"),
        } for t in grid]

    monkeypatch.setattr(localize, "make_spatial_null", _null)
    monkeypatch.setattr(localize, "_sweep_thresholds", _sweep)
    mm, diag = localize.estimate_minmass(
        stack, backend="torch", mode="quality_first", quality_floor=0.16,
        roi_mask=roi, pixel_size_um=0.1, max_false_track_rate=0.10,
        quality_null_replicates=2, workers=1, log_cb=lambda _m: None)

    assert mm >= 0.16
    assert diag["quality_status"] == "valid"
    assert diag["n_candidates_raw_full_frame"] == len(harvested)
    assert diag["n_candidates_raw_roi"] == inside_n
    assert "harvest_density_cap" not in diag
    assert "density_target" not in diag


def test_quality_estimator_error_retains_floor_and_is_invalid(monkeypatch):
    import firefly.analysis.fa_localize as localize

    class _Backend:
        name = "torch"

    monkeypatch.setattr(localize, "_resolve_backend", lambda _name: _Backend())
    monkeypatch.setattr(localize, "_contiguous_windows", lambda _n: [(0, 2)])

    def _fail(*_args, **_kwargs):
        raise RuntimeError("synthetic calibration failure")

    monkeypatch.setattr(localize, "_harvest_windows", _fail)
    mm, diag = localize.estimate_minmass(
        np.zeros((2, 8, 8), dtype=np.float32), mode="quality_first",
        quality_floor=0.16, pixel_size_um=0.1, log_cb=lambda _m: None)
    assert mm == pytest.approx(0.16)
    assert diag["quality_status"] == "invalid"
    assert diag["quality_reason"] == "estimator_error:RuntimeError"
    assert diag["method"].startswith("quality_first_invalid:")
