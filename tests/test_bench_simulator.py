"""Tests for the benchmark simulator (Qt-free; numpy + scipy + pandas)."""
import numpy as np
import pytest

from firefly.bench.config import SimConfig, DiffusionPopulation
from firefly.bench.simulator import simulate


def _cfg(**kw):
    base = dict(seed=1, n_frames=20, height=64, width=64, n_emitters=6,
                photons_per_emitter=1500, bg_photons=8,
                populations=(DiffusionPopulation("brownian", 1.0, D_um2_s=0.05),))
    base.update(kw)
    return SimConfig(**base)


def test_determinism_same_seed_byte_identical():
    a = simulate(_cfg(seed=42))
    b = simulate(_cfg(seed=42))
    assert np.array_equal(a.stack, b.stack)
    assert a.gt_locs.equals(b.gt_locs)


def test_different_seed_differs():
    a = simulate(_cfg(seed=1))
    b = simulate(_cfg(seed=2))
    assert not np.array_equal(a.stack, b.stack)


def test_schema_and_dtypes():
    r = simulate(_cfg())
    assert r.stack.dtype == np.float32 and r.stack.shape == (20, 64, 64)
    assert list(r.gt_locs.columns) == ["frame", "x", "y", "photons",
                                       "emitter_id", "population"]
    assert list(r.gt_tracks.columns) == ["frame", "x", "y", "particle", "population"]
    assert r.gt_locs["frame"].dtype.kind == "i"
    assert r.gt_locs["x"].dtype.kind == "f"
    assert r.meta["pixel_size_um"] == r.meta["pixel_size_um"]   # present


def test_subpixel_centroid_accuracy():
    # one bright emitter, no noise → image centroid must land on the GT position
    cfg = SimConfig(seed=2, n_frames=1, height=40, width=40, n_emitters=1,
                    photons_per_emitter=50000, bg_photons=0, read_noise_e=0,
                    camera_offset=0, photon_cv=0.0, k_on=1.0, k_off=0.0,
                    bleach_prob=0.0,
                    populations=(DiffusionPopulation("immobile", 1.0),))
    r = simulate(cfg)
    f = r.stack[0].astype(float)
    ys, xs = np.mgrid[0:40, 0:40]
    cx = (f * xs).sum() / f.sum()
    cy = (f * ys).sum() / f.sum()
    gx, gy = float(r.gt_locs.x.iloc[0]), float(r.gt_locs.y.iloc[0])
    assert np.hypot(cx - gx, cy - gy) < 0.05


def test_photon_conservation():
    cfg = SimConfig(seed=3, n_frames=1, height=40, width=40, n_emitters=1,
                    photons_per_emitter=40000, bg_photons=0, read_noise_e=0,
                    camera_offset=0, qe=0.9, photon_cv=0.0, k_on=1.0, k_off=0.0,
                    bleach_prob=0.0,
                    populations=(DiffusionPopulation("immobile", 1.0),))
    r = simulate(cfg)
    # rendered photo-electrons ≈ qe × emitted photons
    assert 0.85 < r.stack[0].astype(float).sum() / 40000 < 0.95


def test_blinking_produces_gaps():
    cfg = _cfg(seed=4, n_frames=60, n_emitters=8, k_on=0.2, k_off=0.4,
               bleach_prob=0.0)
    r = simulate(cfg)
    gappy = False
    for _eid, g in r.gt_locs.groupby("emitter_id"):
        fr = np.sort(g["frame"].to_numpy())
        if len(fr) >= 2 and np.any(np.diff(fr) > 1):
            gappy = True
            break
    assert gappy, "expected at least one emitter with a non-contiguous frame run"


def test_brightness_monotonic_in_photons():
    dim = simulate(_cfg(seed=5, photons_per_emitter=300, photon_cv=0.0))
    bright = simulate(_cfg(seed=5, photons_per_emitter=5000, photon_cv=0.0))
    assert bright.stack.max() > dim.stack.max()
