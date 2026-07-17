"""The vectorised effect-size bootstrap helpers (Phase-C speedup) must compute the
EXACT same per-resample statistic as the scalar estimators — only the resampling
draw pattern differs (which legitimately shifts the stochastic CI bounds).  These
tests pin the formula equivalence so the CIs can't silently drift on a future edit.
"""
import numpy as np

from firefly.analysis import fa_circular as fc


def test_pooled_d_batch_matches_scalar_row_by_row():
    rng = np.random.default_rng(1)
    a = rng.normal(1.0, 0.7, 8)
    b = rng.normal(1.5, 0.5, 6)
    sa = a[rng.integers(0, 8, (400, 8))]
    sb = b[rng.integers(0, 6, (400, 6))]
    batch = fc._pooled_d_batch(sa, sb)
    scalar = np.array([fc._cohens_d_pooled(sa[i], sb[i]) for i in range(400)], dtype=float)
    ok = np.isfinite(batch)
    assert np.array_equal(np.isfinite(batch), np.isfinite(scalar))
    assert np.allclose(batch[ok], scalar[ok], atol=1e-12, rtol=0)


def test_cliffs_delta_batch_matches_scalar_row_by_row():
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 1.0, 9)
    b = rng.normal(0.4, 1.2, 11)
    sa = a[rng.integers(0, 9, (300, 9))]
    sb = b[rng.integers(0, 11, (300, 11))]
    batch = fc._cliffs_delta_batch(sa, sb)
    scalar = np.array([fc._cliffs_delta(sa[i], sb[i]) for i in range(300)], dtype=float)
    assert np.allclose(batch, scalar, atol=1e-12, rtol=0)


def test_cliffs_delta_batch_chunking_is_transparent():
    # A na·nb large enough to force multiple memory chunks must give the same
    # answer as a single-shot computation.
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, 40); b = rng.normal(0.2, 1, 40)
    sa = a[rng.integers(0, 40, (5000, 40))]
    sb = b[rng.integers(0, 40, (5000, 40))]
    batch = fc._cliffs_delta_batch(sa, sb)
    scalar = np.array([fc._cliffs_delta(sa[i], sb[i]) for i in range(sa.shape[0])])
    assert np.allclose(batch, scalar, atol=1e-12, rtol=0)


def test_hedges_g_point_estimate_is_deterministic_and_ci_ordered():
    a = np.array([1.0, 1.2, 0.9, 1.1, 1.05])
    b = np.array([1.6, 1.8, 1.5, 1.7, 1.55])
    g, lo, hi = fc._hedges_g_ci(a, b)
    # g is analytic (not bootstrapped) → exact + reproducible
    g2, _, _ = fc._hedges_g_ci(a, b)
    assert g == g2 and g is not None
    assert lo is not None and hi is not None and lo <= g <= hi
