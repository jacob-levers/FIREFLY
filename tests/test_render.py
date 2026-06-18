"""Tests for the super-resolution reconstruction renderer (Qt-free)."""
import numpy as np

from firefly.analysis.fa_render import render_superres, _MAX_EDGE


def test_output_dims_match_field_and_sr():
    # 100x100 px field at 0.1 µm/px = 10 000 nm; at 20 nm/px → 500x500.
    img = render_superres([10, 20], [10, 20], 0.1, sr_nm=20.0, blur_nm=0,
                          field_px=(100, 100))
    assert img.shape == (500, 500)
    assert img.dtype == np.float32


def test_peak_where_points_cluster():
    # All localisations at px (50, 50) → bright spot at the matching SR pixel.
    n = 200
    x = np.full(n, 50.0); y = np.full(n, 50.0)
    img = render_superres(x, y, 0.1, sr_nm=20.0, blur_nm=20.0,
                          field_px=(100, 100))
    iy, ix = np.unravel_index(int(np.argmax(img)), img.shape)
    # (50 px * 0.1 µm * 1000 nm) / 20 nm = 250
    assert abs(iy - 250) <= 2 and abs(ix - 250) <= 2
    assert img.max() > 0


def test_raw_histogram_counts_when_no_blur():
    img = render_superres([10.0, 10.0, 10.0], [10.0, 10.0, 10.0], 0.1,
                          sr_nm=50.0, blur_nm=0, field_px=(50, 50))
    assert img.sum() == 3.0            # 3 localisations, all in one bin
    assert img.max() == 3.0


def test_deterministic():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 60, 500); y = rng.uniform(0, 60, 500)
    a = render_superres(x, y, 0.106, sr_nm=15.0, blur_nm=15.0)
    b = render_superres(x, y, 0.106, sr_nm=15.0, blur_nm=15.0)
    assert np.array_equal(a, b)


def test_empty_input_no_crash():
    img = render_superres([], [], 0.1, sr_nm=20.0)
    assert img.shape == (1, 1) and img.sum() == 0.0


def test_size_guard_caps_huge_field():
    # A 4000x4000 px field at 0.1 µm/px and 1 nm/px would be 400k px/edge;
    # the guard must bump sr_nm so neither edge exceeds _MAX_EDGE.
    img = render_superres([0, 4000], [0, 4000], 0.1, sr_nm=1.0, blur_nm=0,
                          field_px=(4000, 4000))
    assert max(img.shape) <= _MAX_EDGE
