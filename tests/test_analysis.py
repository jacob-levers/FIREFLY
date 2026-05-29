"""Numerical regression tests for the FIREFLY analysis pipeline.

These exercise the real localisation + diffusion code on synthetic data
with known ground truth, so a refactor that silently changes the numbers
gets caught.  No image/CSV fixtures or GUI are required.
"""
import numpy as np
import pandas as pd
import pytest

from firefly import sptpalm_analysis as s


# ── helpers ────────────────────────────────────────────────────────────────
def _synthetic_brownian_tracks(n_tracks=300, n_frames=60, sigma_px=2.0, seed=0):
    """Pure 2D Brownian tracks: each frame adds a Gaussian step of std
    `sigma_px` pixels.  Returns a (particle, frame, x, y) DataFrame."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in range(n_tracks):
        x = np.cumsum(rng.normal(0, sigma_px, n_frames))
        y = np.cumsum(rng.normal(0, sigma_px, n_frames))
        for f in range(n_frames):
            rows.append((pid, f, x[f], y[f]))
    return pd.DataFrame(rows, columns=["particle", "frame", "x", "y"])


def _synthetic_spot_frame(truth, H=96, W=96, amp=400.0, sigma=1.3, bg=10.0):
    yy, xx = np.mgrid[0:H, 0:W]
    frame = np.full((H, W), bg, dtype=np.float32)
    for cx, cy in truth:
        frame = frame + amp * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    return frame.astype(np.float32)


def _match_to_truth(locs, truth, tol_px):
    """Nearest-neighbour match of detected (x, y) to ground-truth spots.
    Returns the list of per-spot errors (px); asserts each spot matched."""
    errs = []
    for cx, cy in truth:
        d = np.hypot(locs["x"].to_numpy() - cx, locs["y"].to_numpy() - cy)
        j = int(np.argmin(d))
        assert d[j] <= tol_px, f"spot ({cx},{cy}) unmatched; nearest {d[j]:.2f}px"
        errs.append(float(d[j]))
    return errs


# ── diffusion ───────────────────────────────────────────────────────────────
def test_brownian_diffusion_recovers_D_and_alpha():
    """MSD fit recovers the known D (=sigma_um**2/2dt) and alpha~1 for pure
    Brownian motion."""
    px, dt, sigma_px = 0.1, 0.05, 2.0
    D_true = (sigma_px * px) ** 2 / (2 * dt)
    tracks = _synthetic_brownian_tracks(sigma_px=sigma_px)
    _imsd, _emsd, diff = s.compute_msd_and_fit(
        tracks, px, dt, max_lagtime=20, n_fit=5, workers=1)
    ratio = diff["D"].median() / D_true
    assert 0.8 <= ratio <= 1.2, f"D off: median/true = {ratio:.3f}"
    assert 0.85 <= diff["alpha"].median() <= 1.15, \
        f"alpha not ~1: {diff['alpha'].median():.3f}"


# ── localisation ─────────────────────────────────────────────────────────────
def test_trackpy_localisation_finds_and_locates_spots():
    truth = [(20.5, 30.2), (60.1, 45.7), (75.0, 70.0)]
    stack = np.stack([_synthetic_spot_frame(truth)] * 2)
    locs = s.localise_particles(stack, diameter=7, minmass=50, backend="trackpy")
    f0 = locs[locs["frame"] == 0]
    assert len(f0) == len(truth)
    errs = _match_to_truth(f0, truth, tol_px=1.0)
    assert max(errs) < 0.5, f"sub-pixel localisation drifted: {errs}"


@pytest.mark.parametrize("backend", ["torch", "torch-cpu"])
def test_torch_agrees_with_trackpy(backend):
    """The two detection engines must agree on the same synthetic spots —
    this is the calibration invariant the README advertises."""
    if backend not in s.list_available_backends():
        pytest.skip(f"{backend} backend unavailable")
    truth = [(20.5, 30.2), (60.1, 45.7), (75.0, 70.0)]
    stack = np.stack([_synthetic_spot_frame(truth)] * 2)
    tp = s.localise_particles(stack, diameter=7, minmass=50, backend="trackpy")
    th = s.localise_particles(stack, diameter=7, minmass=50, backend=backend)
    tp0 = tp[tp["frame"] == 0].reset_index(drop=True)
    th0 = th[th["frame"] == 0].reset_index(drop=True)
    assert abs(len(tp0) - len(th0)) <= 1, \
        f"spot-count mismatch trackpy={len(tp0)} {backend}={len(th0)}"
    # Every trackpy spot should have a close match in the torch output.
    # Tolerance 0.2 px: the engines agree to ~0.01 px on these clean
    # synthetic spots, and to ~0.05 px (5 nm) median on real sptPALM data.
    # 0.2 px keeps ~20x headroom over the synthetic value so the gate
    # isn't flaky across platforms / numpy builds, while still being tight
    # enough to catch a genuine calibration regression (the old 1.0 px
    # tolerance would have passed a 10x degradation silently).
    for _, r in tp0.iterrows():
        d = np.hypot(th0["x"] - r["x"], th0["y"] - r["y"]).min()
        assert d <= 0.2, f"engine disagreement: {d:.3f}px > 0.2px tolerance"


# ── memory-safe loader allocation ────────────────────────────────────────────
def test_alloc_stack_uses_ram_when_it_fits():
    a = s._alloc_or_memmap_stack((4, 8, 8))
    assert isinstance(a, np.ndarray) and not isinstance(a, np.memmap)
    assert a.shape == (4, 8, 8) and a.dtype == np.float32


def test_alloc_stack_falls_back_to_memmap_when_too_big(monkeypatch):
    # The allocator lives in fa_memory and resolves the RAM reserve there,
    # so patch fa_memory's copy to force the "won't fit in RAM" branch.
    from firefly.analysis import fa_memory
    monkeypatch.setattr(fa_memory, "_user_ram_reserve_gb", lambda: 1e9)
    arr = fa_memory._alloc_or_memmap_stack((4, 8, 8))
    try:
        assert isinstance(arr, np.memmap)
        arr[0] = 3.0
        assert float(arr[0, 0, 0]) == 3.0
    finally:
        del arr
        fa_memory.cleanup_temp_stack_paths()


# ── regression: pre-existing ROI/empty-locs crash (fixed) ───────────────────
def test_link_trajectories_handles_empty_locs():
    """Empty localisations must not crash the trackpy linker (it used to raise
    a cryptic IndexError on coords_from_df)."""
    empty = pd.DataFrame(columns=["x", "y", "frame", "mass"])
    out = s.link_trajectories(empty, search_range=5, memory=3, min_len=5)
    assert len(out) == 0
    assert "particle" in out.columns


def test_roi_mask_not_wiped_for_small_structure():
    """A compact bright structure far smaller than the legacy 8000-px object
    floor must still yield a non-empty ROI mask — the bug that dropped every
    localisation and crashed linking."""
    yy, xx = np.mgrid[0:128, 0:128]
    proj = (5.0 * np.exp(-(((xx - 64) ** 2 + (yy - 64) ** 2) / (2 * 8.0 ** 2)))
            ).astype(np.float32)
    mask, info = s.build_roi_mask_advanced(
        proj, threshold=None, threshold_method="li", bg_sigma=25.0,
        mode_hint="max")
    assert mask.any(), "ROI mask wrongly empty for a real small structure"
    assert 0.0 < float(mask.mean()) < 1.0
