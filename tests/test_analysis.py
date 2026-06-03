"""Numerical regression tests for the FIREFLY analysis pipeline.

These exercise the real localisation + diffusion code on synthetic data
with known ground truth, so a refactor that silently changes the numbers
gets caught.  No image/CSV fixtures or GUI are required.
"""
import os

import numpy as np
import pandas as pd
import pytest

from firefly import sptpalm_analysis as s
from firefly.analysis import fa_circular as fc
from firefly.analysis import fa_diffusion as fd


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


def test_alpha_offset_aware_unbiased_under_localisation_error():
    """A genuinely Brownian (slow) population observed WITH localisation error
    has an MSD floor (≈4·sigma²) that flattens the first lags.  A naive log-log
    slope reads alpha < 1 (mislabelling it confined/immobile); the offset-aware
    joint fit recovers alpha ≈ 1.  This test demonstrates the bias AND the fix."""
    px, dt = 0.1, 0.05
    sigma_px = 0.5            # slow diffusion -> small signal at short lags
    loc_noise_px = 0.5        # static localisation error -> MSD offset
    tracks = _synthetic_brownian_tracks(
        n_tracks=400, n_frames=60, sigma_px=sigma_px, seed=7).copy()
    rng = np.random.default_rng(11)
    tracks["x"] = tracks["x"].to_numpy() + rng.normal(0, loc_noise_px, len(tracks))
    tracks["y"] = tracks["y"].to_numpy() + rng.normal(0, loc_noise_px, len(tracks))

    _imsd, emsd, diff = fd.compute_msd_and_fit(
        tracks, px, dt, max_lagtime=20, n_fit=5, workers=1)

    # The OLD estimator (plain log-log slope of the ensemble MSD) is biased low.
    t5 = np.arange(1, 6) * dt
    m5 = np.asarray(emsd)[:5]
    old_alpha = float(np.polyfit(np.log(t5), np.log(m5), 1)[0])
    assert old_alpha < 0.8, f"expected biased-low old alpha, got {old_alpha:.3f}"

    # The NEW offset-aware estimator recovers ~1.
    assert diff["alpha"].median() > 0.85, \
        f"offset-aware alpha still biased: {diff['alpha'].median():.3f}"
    # offset (MSD0) is recovered as a positive localisation-error floor.
    assert diff["MSD0"].median() > 0


def test_hedges_g_ci_and_small_n_guard():
    """Hedges' g returns a finite value bracketed by its bootstrap CI; the
    n<3-replicate guard blanks the significance stars and flags the comparison."""
    a = np.array([0.10, 0.12, 0.11, 0.13])
    b = np.array([0.30, 0.31, 0.29, 0.32])
    g, lo, hi = fc._hedges_g_ci(a, b)
    assert g is not None and np.isfinite(g)
    assert lo <= g <= hi

    _om, pw = fc._stat_test_n([a, b], ["A", "B"])
    assert pw[0]["hedges_g"] is not None
    assert pw[0]["stars"] in ("*", "**", "***")   # n=4, real difference

    # n=2 per group: test is not interpretable -> no stars, flagged underpowered.
    _om2, pw2 = fc._stat_test_n(
        [np.array([0.1, 0.2]), np.array([0.9, 1.0])], ["A", "B"])
    assert pw2[0]["stars"] == ""
    assert "underpowered" in pw2[0]["note"]


def test_circular_comparison_csv_is_split_and_clean(tmp_path):
    """The comparison circular CSV is split into three clean single-table files
    (per-group / per-replicate / tests), each with a consistent header."""
    rng = np.random.default_rng(0)

    def reps(mean_deg, k=3, n=200):
        return [np.degrees(rng.vonmises(np.radians(mean_deg), 2.0, n))
                for _ in range(k)]

    a_reps, b_reps = reps(0.0), reps(90.0)
    groups_angles = [("A", np.concatenate(a_reps), "#ff0000"),
                     ("B", np.concatenate(b_reps), "#00ff00")]
    per_rep = {"A": a_reps, "B": b_reps}

    csv = tmp_path / "X_circular_statistics.csv"
    fc.save_comparison_circular_statistics(
        groups_angles, csv_path=str(csv), pdf_path=None,
        per_replicate_angles=per_rep)

    base = str(tmp_path / "X")
    pg = pd.read_csv(base + "_circular_per_group.csv")
    rep = pd.read_csv(base + "_circular_per_replicate.csv")
    tst = pd.read_csv(base + "_circular_tests.csv")

    # The old monolithic file should NOT be written.
    assert not os.path.exists(str(csv))

    assert {"group", "n_replicates"}.issubset(pg.columns) and len(pg) == 2
    assert list(rep.columns) == ["group", "replicate", "kappa", "rbar", "mu_deg"]
    assert len(rep) == 6                       # 2 groups × 3 replicates
    assert {"metric", "scope", "p_value"}.issubset(tst.columns)
    assert tst["hedges_g"].notna().any()       # effect size populated


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
def test_alloc_stack_uses_ram_when_it_fits(monkeypatch):
    # Pin the RAM reserve to 0 so a ~1 KB array unconditionally "fits in RAM".
    # Without this the allocator decides via (free_RAM - reserve), which can dip
    # to ~0 on a memory-pressured machine and defensively memmap even a tiny
    # array — making this test flaky.  We're testing the fits-in-RAM branch, not
    # the host's spare memory.
    from firefly.analysis import fa_memory
    monkeypatch.setattr(fa_memory, "_user_ram_reserve_gb", lambda: 0.0)
    a = fa_memory._alloc_or_memmap_stack((4, 8, 8))
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


# ── auto-threshold (estimate_minmass) ────────────────────────────────────────
def _bimodal_spot_stack(n_bright=8, n_dim=40, amp_bright=500.0, amp_dim=70.0,
                        H=160, W=160, F=24, seed=0):
    """Frames with a bright real-spot population and a dim spurious-spot
    population on a noisy background — a clean bimodal mass distribution."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    frames = []
    for _ in range(F):
        im = rng.poisson(30, (H, W)).astype(np.float32)
        for amp, k in ((amp_bright, n_bright), (amp_dim, n_dim)):
            for _ in range(k):
                cx, cy = rng.uniform(8, W - 8), rng.uniform(8, H - 8)
                im = im + amp * np.exp(
                    -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.3 ** 2)))
        frames.append(im.astype(np.float32))
    return np.stack(frames)


def _quiet_estimate(stack, **kw):
    import io, contextlib, logging
    logging.getLogger("trackpy").setLevel(logging.ERROR)
    from firefly.analysis import fa_localize as L
    with contextlib.redirect_stdout(io.StringIO()):
        return L.estimate_minmass(stack, backend="trackpy", workers=2, **kw)


def test_estimate_minmass_lands_between_noise_and_signal():
    """The chosen minmass must sit between the dim-noise and bright-signal
    mass clusters, recovering ~the bright-spot count."""
    import io, contextlib, logging
    from firefly.analysis import fa_localize as L
    from firefly.analysis.fa_preprocess import preprocess_stack
    stack = _bimodal_spot_stack()
    # link_min_len huge → nothing qualifies as a track → the linkability sweep
    # is inconclusive and we exercise the STATIC estimator under test.
    mm, diag = _quiet_estimate(stack, diameter=7, sensitivity="balanced",
                               frame_sample=20, link_min_len=999)
    assert "gmm" in (diag.get("static_method") or diag["method"]), diag
    lo, hi = diag["gmm_means"]               # log10 component means
    assert lo < np.log10(mm) < hi, "cutoff not between the two clusters"
    # Apply it: should keep ~8 bright spots/frame, drop the 40 dim ones.
    logging.getLogger("trackpy").setLevel(logging.ERROR)
    with contextlib.redirect_stdout(io.StringIO()):
        pp = preprocess_stack(stack, workers=2)
        kept = L.localise_particles(pp, diameter=7, minmass=mm,
                                    percentile=64, backend="trackpy", workers=2)
    per_frame = len(kept) / len(stack)
    assert 5 <= per_frame <= 12, f"expected ~8 bright spots/frame, got {per_frame:.1f}"


def test_estimate_minmass_sensitivity_ordering():
    stack = _bimodal_spot_stack(seed=1)
    strict = _quiet_estimate(stack, diameter=7, sensitivity="strict", frame_sample=20, link_min_len=999)[0]
    bal    = _quiet_estimate(stack, diameter=7, sensitivity="balanced", frame_sample=20, link_min_len=999)[0]
    lenient = _quiet_estimate(stack, diameter=7, sensitivity="lenient", frame_sample=20, link_min_len=999)[0]
    assert strict >= bal >= lenient, (strict, bal, lenient)


def test_estimate_minmass_continuous_uses_quantile():
    """Dense data with a single continuous mass distribution (no noise/signal
    valley — the typical PC12 case) must use the calibrated mass-quantile cut,
    NOT fall back to a histogram-shape method, and the cutoff must sit inside
    the bulk of the candidate masses."""
    rng = np.random.default_rng(3)
    H = W = 160
    yy, xx = np.mgrid[0:H, 0:W]
    frames = []
    for _ in range(24):
        im = rng.poisson(30, (H, W)).astype(np.float32)
        for _ in range(60):                      # many similar-brightness spots
            cx, cy = rng.uniform(8, W - 8), rng.uniform(8, H - 8)
            amp = rng.uniform(200, 600)          # broad, unimodal range
            im = im + amp * np.exp(
                -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.3 ** 2)))
        frames.append(im.astype(np.float32))
    stack = np.stack(frames)
    mm, diag = _quiet_estimate(stack, diameter=7, sensitivity="balanced",
                               frame_sample=20, link_min_len=999)
    assert "mass_quantile" in (diag.get("static_method") or diag["method"]), diag
    lo, hi = np.percentile(diag["_log_masses"], [5, 95])
    assert lo < np.log10(mm) < hi, "quantile cut outside the candidate bulk"
    strict = _quiet_estimate(stack, diameter=7, sensitivity="strict", frame_sample=20, link_min_len=999)[0]
    lenient = _quiet_estimate(stack, diameter=7, sensitivity="lenient", frame_sample=20, link_min_len=999)[0]
    assert strict >= mm >= lenient


def test_estimate_minmass_noise_only_falls_back():
    """A noise-only stack (no real spots) must not crash and should return a
    finite, clamped threshold via a fallback path."""
    rng = np.random.default_rng(2)
    stack = rng.poisson(30, (20, 128, 128)).astype(np.float32)
    mm, diag = _quiet_estimate(stack, diameter=7, sensitivity="balanced",
                               frame_sample=20)
    assert np.isfinite(mm) and mm >= 0.05
    assert diag["method"] is not None


# ── linkability-optimised auto-threshold (the primary engine) ────────────────
def _linkable_spot_stack(n_real=10, n_blip=55, F=44, H=160, W=160,
                         amp_real=520.0, amp_blip=(120.0, 320.0), step=1.4,
                         noise=30.0, sigma=1.3, seed=0):
    """`n_real` emitters Brownian-walking across F CONSECUTIVE frames (they
    persist → link into long tracks) plus `n_blip` random per-frame blips
    (new positions every frame → cannot link).  The blip brightnesses PARTIALLY
    OVERLAP the real spots, so a fixed mass-quantile cut admits many blips —
    only temporal linkability cleanly separates them."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    pos = rng.uniform(18, W - 18, size=(n_real, 2))
    frames = []
    for _ in range(F):
        im = rng.poisson(noise, (H, W)).astype(np.float32)
        pos = np.clip(pos + rng.normal(0, step, size=pos.shape), 8, W - 8)
        for cx, cy in pos:
            im += amp_real * np.exp(
                -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
        for _ in range(n_blip):
            bx, by = rng.uniform(8, W - 8), rng.uniform(8, H - 8)
            a = rng.uniform(*amp_blip)
            im += a * np.exp(
                -(((xx - bx) ** 2 + (yy - by) ** 2) / (2 * sigma ** 2)))
        frames.append(im.astype(np.float32))
    return np.stack(frames)


def _count_tracks(stack, mm, diameter=7, search_range=5, memory=1, L=4):
    """Detect at `mm` and link the full stack; return (good, spurious) track
    counts (≥L frames vs ≤2 frames)."""
    import io, contextlib, logging
    from firefly.analysis.fa_preprocess import preprocess_stack
    from firefly.analysis import fa_localize as Lz
    from firefly.analysis.fa_linking import link_trajectories
    logging.getLogger("trackpy").setLevel(logging.ERROR)
    with contextlib.redirect_stdout(io.StringIO()):
        pp = preprocess_stack(stack, workers=2)
        locs = Lz.localise_particles(pp, diameter=diameter, minmass=mm,
                                     percentile=64, backend="trackpy", workers=2)
        linked = link_trajectories(locs, search_range=search_range,
                                   memory=memory, min_len=1)
    lens = linked.groupby("particle")["frame"].count()
    return int((lens >= L).sum()), int((lens <= 2).sum())


def test_estimate_minmass_linkability_path_selected():
    """On a stack with persistent (linkable) real spots and non-linkable blips
    of OVERLAPPING brightness, the estimator must use the linkability sweep
    (not the static fallback) and recover roughly the real-spot count."""
    stack = _linkable_spot_stack(seed=0)
    mm, diag = _quiet_estimate(stack, diameter=7, sensitivity="balanced",
                               search_range=5, memory=1, link_min_len=4)
    assert diag["method"] == "linkability", diag.get("method")
    assert "sweep" in diag and len(diag["sweep"]) >= 3
    assert np.isfinite(mm) and mm >= diag["noise_floor"]
    assert diag.get("n_good", 0) >= 5, diag.get("n_good")


def test_estimate_minmass_linkability_beats_quantile_precision():
    """The linkability cut must be at least as PURE as the static p30 quantile
    cut — fewer spurious (≤2-frame) tracks at comparable real-track recall —
    on data where the two mass populations overlap."""
    from firefly.analysis.fa_localize import _static_minmass
    stack = _linkable_spot_stack(seed=1)
    mm_link, diag = _quiet_estimate(stack, diameter=7, sensitivity="balanced",
                                    search_range=5, memory=1, link_min_len=4)
    assert diag["method"] == "linkability", diag.get("method")
    masses = 10.0 ** np.asarray(diag["_log_masses"], dtype=float)
    mm_quant = _static_minmass(masses, "balanced", {}, lambda m: None)
    g_link, s_link = _count_tracks(stack, mm_link)
    g_quant, s_quant = _count_tracks(stack, mm_quant)
    # Linkability keeps the real tracks but admits fewer spurious fragments.
    assert g_link >= max(5, int(0.7 * g_quant)), (g_link, g_quant)
    assert s_link <= s_quant, (s_link, s_quant)


def test_pick_linkability_falls_back_when_no_linkage():
    """Nothing links (N_good ≈ 0 at every threshold) → the sweep is
    inconclusive and the picker returns None so the caller uses the static
    estimator."""
    from firefly.analysis.fa_localize import _pick_linkability_threshold
    sweep = [dict(t=float(t), n_surv=100, N_good=0, good_fraction=0.0,
                  spurious_rate=1.0, median_ep=float("nan"))
             for t in np.geomspace(1, 50, 12)]
    pick, info = _pick_linkability_threshold(sweep, "balanced", None, 1.0)
    assert pick is None and info["reason"] == "no_linkage", info


def test_pick_linkability_falls_back_when_no_spurious_population():
    """good_fraction is high at every threshold (immobile-dominated / no
    suppressible spurious population) → linkability cannot beat a static cut,
    so the picker defers."""
    from firefly.analysis.fa_localize import _pick_linkability_threshold
    sweep = [dict(t=float(t), n_surv=100, N_good=10, good_fraction=0.95,
                  spurious_rate=0.03, median_ep=0.1)
             for t in np.geomspace(1, 50, 12)]
    pick, info = _pick_linkability_threshold(sweep, "balanced", None, 1.0)
    assert pick is None and info["reason"] == "no_spurious_population", info


def test_pick_linkability_picks_purity_recall_balance():
    """On a realistic sweep (spurious rate falls and good_fraction rises as the
    threshold climbs, then real tracks start dropping) the picker lands at the
    F1 balance — not the highest-N_good point inflated by fragmentation."""
    from firefly.analysis.fa_localize import _pick_linkability_threshold
    rows = [  # t, n_surv, N_good, good_fraction, spurious_rate
        (1.0, 2000, 13, 0.25, 0.67),
        (1.6, 1300, 14, 0.33, 0.64),
        (2.1,  750, 13, 0.58, 0.42),
        (2.6,  500, 13, 0.87, 0.13),   # ← the balance point
        (3.1,  420, 12, 0.93, 0.06),
        (3.5,  360, 26, 0.94, 0.06),   # N_good inflated by fragmentation
        (3.8,  270, 34, 0.84, 0.14),   # …and spurious creeps back up
    ]
    sweep = [dict(t=t, n_surv=ns, N_good=ng, good_fraction=gf,
                  spurious_rate=sr, median_ep=0.1)
             for (t, ns, ng, gf, sr) in rows]
    pick, info = _pick_linkability_threshold(sweep, "balanced", None, 0.5)
    assert info["rule"] == "f1_purity_recall"
    assert 2.0 <= pick <= 3.2, (pick, info)        # not the fragmented tail


def test_estimate_minmass_max_false_track_rate_override():
    """With a max-false-track-rate ceiling set, the chosen point's measured
    spurious rate must respect it (override path), still finite and ≥ floor."""
    stack = _linkable_spot_stack(seed=2)
    mm, diag = _quiet_estimate(stack, diameter=7, sensitivity="balanced",
                               search_range=5, memory=1, link_min_len=4,
                               max_false_track_rate=0.10)
    assert diag["method"] == "linkability", diag.get("method")
    info = diag.get("link_info") or {}
    assert info.get("rule") == "max_false_track_rate", info
    assert np.isfinite(mm) and mm >= diag["noise_floor"]


# ── two-factor (group × time point) mixed ANOVA ──────────────────────────────
def _paired_twofactor_df(interaction=True, seed=0):
    """Synthetic paired 2-group × 2-time-point cell-level scalars.  When
    `interaction` is True, group 'B' shifts only at the 'Post' time point —
    a clean group×time interaction the mixed ANOVA must recover."""
    rng = np.random.default_rng(seed)
    rows = []
    for grp, base in [("A", 0.040), ("B", 0.040)]:
        for ci in range(8):
            cell = f"{grp}_c{ci}"
            for tp in ["Pre", "Post"]:
                val = base + rng.normal(0, 0.003)
                if interaction and grp == "B" and tp == "Post":
                    val -= 0.012
                rows.append({"group": grp, "timepoint": tp, "cell": cell,
                             "auc_msd": val})
    return pd.DataFrame(rows)


def test_twoway_recovers_planted_interaction():
    tw = pytest.importorskip("firefly.analysis.fa_twoway")
    if not tw.HAVE_PINGOUIN:
        pytest.skip("pingouin not installed")
    # interaction present → significant
    df = _paired_twofactor_df(interaction=True)
    res, _msg = tw.compute_twoway_anova(df, ["auc_msd"])
    inter = res[(res["section"] == "anova") & (res["effect"] == "Interaction")]
    p = float(inter["p_GG"].iloc[0])
    assert p < 0.05, f"planted interaction not detected (p={p:.4f})"
    # no interaction → not significant
    df0 = _paired_twofactor_df(interaction=False, seed=1)
    res0, _ = tw.compute_twoway_anova(df0, ["auc_msd"])
    inter0 = res0[(res0["section"] == "anova") & (res0["effect"] == "Interaction")]
    assert float(inter0["p_GG"].iloc[0]) > 0.05, "false-positive interaction"


def test_twoway_headline_extracts_interaction_and_group():
    """_twoway_headline pulls the interaction (group×time) and group-main-effect
    p-values out of the ANOVA table for display on the interaction panel."""
    tw = pytest.importorskip("firefly.analysis.fa_twoway")
    if not tw.HAVE_PINGOUIN:
        pytest.skip("pingouin not installed")
    from firefly.analysis import fa_compare as fc
    df = _paired_twofactor_df(interaction=True)
    res, _ = tw.compute_twoway_anova(df, ["auc_msd"])
    hl = fc._twoway_headline(res, "auc_msd")
    assert hl is not None
    assert hl["interaction_p"] is not None and hl["interaction_p"] < 0.05
    assert hl["interaction_stars"] in ("*", "**", "***")
    assert hl["group_p"] is not None
    # graceful on missing inputs
    assert fc._twoway_headline(None, "auc_msd") is None
    assert fc._twoway_headline(res, "no_such_metric") is None


def test_twoway_validate_pairing_drops_unmatched():
    tw = pytest.importorskip("firefly.analysis.fa_twoway")
    df = _paired_twofactor_df(interaction=False)
    # remove one cell's 'Post' row → it should be listwise-dropped
    df = df[~((df["cell"] == "A_c0") & (df["timepoint"] == "Post"))]
    clean, warn, dropped = tw.validate_pairing(df)
    assert warn is not None and "A_c0" in warn
    assert ("A", "A_c0") not in set(zip(clean["group"], clean["cell"]))
    # every surviving cell appears at both time points
    counts = clean.groupby(["group", "cell"])["timepoint"].nunique()
    assert (counts == 2).all()


def test_twoway_subject_key_strips_timepoint():
    tw = pytest.importorskip("firefly.analysis.fa_twoway")
    key, matched = tw.derive_subject_key(
        "20250319_PC12 P11_Syntaxin_DMSO_D1_Post", ["Pre", "Post"])
    assert matched and key == "20250319_PC12 P11_Syntaxin_DMSO_D1"
    # no token present → unmatched, stem returned unchanged
    key2, matched2 = tw.derive_subject_key("cellX_only", ["Pre", "Post"])
    assert key2 == "cellX_only" and matched2 is False


def test_twoway_needs_two_timepoints():
    tw = pytest.importorskip("firefly.analysis.fa_twoway")
    if not tw.HAVE_PINGOUIN:
        pytest.skip("pingouin not installed")
    df = _paired_twofactor_df()
    df = df[df["timepoint"] == "Pre"]            # only one time point
    res, msg = tw.compute_twoway_anova(df, ["auc_msd"])
    assert res is None and "time point" in msg


# ── loader robustness: macOS AppleDouble sidecars on exFAT drives ────────────
def test_find_stem_ignores_appledouble_sidecars(tmp_path):
    """On exFAT/SMB volumes macOS writes a binary `._<name>` sidecar next to
    every file.  `._<stem>_params.json` sorts before the real file and used
    to be picked as the stem (then json.load choked on the binary blob).
    _find_stem must skip dotfiles and return the real stem."""
    from firefly.analysis import fa_palmtracer as fp
    stem = "20250319_Cell_D1_Pre"
    # real params file
    (tmp_path / f"{stem}_params.json").write_text('{"pixel_size_um": 0.106}')
    # AppleDouble sidecar: binary, sorts first, 0xb0 byte at pos 37
    (tmp_path / f"._{stem}_params.json").write_bytes(
        b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        \x00\x02\x00\x00\x00"
        b"\xb0\x00\x00\x00\x02")
    assert fp._find_stem(str(tmp_path)) == stem
    assert "._" not in fp._find_stem(str(tmp_path))


# ── paired (within-group, across-time) stats for interaction plots ───────────
def test_paired_test_and_effect_size():
    from firefly.analysis.fa_circular import _paired_test, _paired_hedges_g
    rng = np.random.default_rng(0)
    pre = rng.normal(2.5, 0.4, 12)
    post = pre - 0.5 + rng.normal(0, 0.1, 12)     # consistent paired drop
    p, stars = _paired_test(pre, post)
    assert p < 0.05 and stars in ("*", "**", "***")
    g = _paired_hedges_g(pre, post)
    assert g is not None and g < 0            # post < pre → negative effect
    # no real change → not significant
    post0 = pre + rng.normal(0, 0.05, 12)
    p0, _ = _paired_test(pre, post0)
    assert p0 > 0.05
    # degenerate guards
    assert _paired_hedges_g([1.0], [2.0]) is None
    assert not np.isfinite(_paired_test([1.0, 1.0], [1.0, 1.0])[0])


def test_alpha_unidentifiable_for_immobile_no_boundary_wall():
    """Regression: a jitter-dominated (immobile) track has a flat MSD, so the
    anomalous-exponent fit can't identify alpha — curve_fit used to park it at
    the upper bound, producing an unphysical 'wall' in the alpha histogram.
    Such tracks must now report alpha = NaN and classify as 'Immobile', and NO
    track may sit pinned at the alpha bound."""
    rng = np.random.default_rng(0)
    px, dt, sigma_loc = 0.1, 0.1, 0.3
    rows = []
    # 50 immobile molecules: fixed position + per-frame localisation jitter.
    for pid in range(50):
        cx, cy = rng.uniform(20, 200, 2)
        for f in range(15):
            rows.append((pid, f, cx + rng.normal(0, sigma_loc),
                         cy + rng.normal(0, sigma_loc), 100.0))
    # 30 mobile Brownian molecules.
    for pid in range(50, 80):
        x = np.cumsum(rng.normal(0, 1.5, 15)); y = np.cumsum(rng.normal(0, 1.5, 15))
        for f in range(15):
            rows.append((pid, f, x[f] + rng.normal(0, sigma_loc),
                         y[f] + rng.normal(0, sigma_loc), 100.0))
    tracks = pd.DataFrame(rows, columns=["particle", "frame", "x", "y", "mass"])
    _imsd, _emsd, diff = s.compute_msd_and_fit(
        tracks, px, dt, max_lagtime=10, n_fit=5, workers=1)

    a = diff["alpha"]
    # No unphysical wall: nothing pinned at/above the (now 2.0) bound, and
    # never above 2.0.
    assert (a.dropna() < 1.99).all(), "alpha still pinned at the upper bound"
    # Immobile molecules: many alpha become NaN and are classified Immobile.
    assert a.isna().sum() > 0
    assert (diff["motion"] == "Immobile").sum() > 0
    # Mobile molecules retain a finite, physical alpha (~1 for Brownian).
    assert a.notna().sum() > 0
    assert 0.0 <= a.dropna().median() <= 1.6


def test_tif_series_streams_into_combined_stack(tmp_path):
    """Multi-file TIF series must load by STREAMING each source into the
    combined stack in chunks (so the whole source file is never held in RAM,
    which used to inflate the peak and demote big series to disk memmap).  The
    streamed result must equal a plain concatenation."""
    tifffile = pytest.importorskip("tifffile")
    from firefly.analysis import fa_loaders as L
    parts, paths = [], []
    for fi in range(3):
        arr = np.zeros((6, 8, 8), np.uint16)
        for fr in range(6):
            arr[fr] = fi * 100 + fr            # unique per (file, frame)
        name = "mov.tif" if fi == 0 else f"mov-file{fi+1:03d}.tif"
        p = str(tmp_path / name)
        tifffile.imwrite(p, arr)
        paths.append(p); parts.append(arr)
    expected = np.concatenate(parts).astype(np.float32)
    combined, _px, _fi = L.load_tif(paths[0], files=paths)
    combined = np.asarray(combined)
    assert combined.shape == (18, 8, 8) and combined.dtype == np.float32
    assert np.array_equal(combined, expected)


def test_loc_precision_from_msd_offset():
    """sigma_loc = sqrt(MSD0/4) recovers the known localisation precision from
    the fitted MSD offset.  Best-determined for immobile/slow tracks where the
    offset dominates — use jitter-only molecules with 30 nm precision."""
    rng = np.random.default_rng(1)
    px, dt, sig_px = 0.1, 0.05, 0.3          # 0.3 px = 30 nm true precision
    rows = []
    for pid in range(300):
        cx, cy = rng.uniform(20, 200, 2)
        for f in range(40):
            rows.append((pid, f, cx + rng.normal(0, sig_px),
                         cy + rng.normal(0, sig_px), 100.0))
    tracks = pd.DataFrame(rows, columns=["particle", "frame", "x", "y", "mass"])
    _i, _e, diff = s.compute_msd_and_fit(tracks, px, dt, max_lagtime=10,
                                         n_fit=5, workers=1)
    assert "loc_sigma_nm" in diff.columns
    assert diff["loc_sigma_nm"].notna().all()
    med = diff["loc_sigma_nm"].median()
    assert 22.0 <= med <= 40.0, f"loc precision off: {med:.1f} nm (true 30)"


def test_van_hove_non_gaussian_parameter():
    """Van Hove alpha2 ~ 0 for a homogeneous Brownian ensemble and clearly
    positive for a heterogeneous mix (mobile + immobile) — the population-
    heterogeneity signal the per-track averages miss."""
    rng = np.random.default_rng(0)

    def brownian(n, sigma_step, start_pid=0):
        rows = []
        for pid in range(start_pid, start_pid + n):
            x = np.cumsum(rng.normal(0, sigma_step, 40))
            y = np.cumsum(rng.normal(0, sigma_step, 40))
            for f in range(40):
                rows.append((pid, f, x[f], y[f], 100.0))
        return rows

    cols = ["particle", "frame", "x", "y", "mass"]
    homo = pd.DataFrame(brownian(300, 2.0), columns=cols)
    vh_homo = fd.compute_van_hove(homo, 0.1)
    assert vh_homo is not None
    assert abs(vh_homo["non_gaussian_alpha2"]) < 0.15
    assert vh_homo["n_displacements"] > 1000

    rows = brownian(150, 2.0)
    for pid in range(150, 300):              # add a near-immobile population
        cx, cy = rng.uniform(0, 50, 2)
        for f in range(40):
            rows.append((pid, f, cx + rng.normal(0, 0.1),
                         cy + rng.normal(0, 0.1), 100.0))
    hetero = pd.DataFrame(rows, columns=cols)
    vh_het = fd.compute_van_hove(hetero, 0.1)
    assert vh_het["non_gaussian_alpha2"] > 0.3


def test_jdd_recovers_two_populations():
    """Jump-distance-distribution fit recovers two diffusion coefficients from
    a 2-population mixture (slow + fast Brownian)."""
    rng = np.random.default_rng(3)
    px, dt = 0.1, 0.05
    D1 = (1.0 * px) ** 2 / (2 * dt)          # slow  -> 0.1 µm²/s
    D2 = (4.0 * px) ** 2 / (2 * dt)          # fast  -> 1.6 µm²/s
    rows = []
    for pid, sig in [(p, 1.0) for p in range(150)] + \
                    [(p, 4.0) for p in range(150, 300)]:
        x = np.cumsum(rng.normal(0, sig, 50)); y = np.cumsum(rng.normal(0, sig, 50))
        for f in range(50):
            rows.append((pid, f, x[f], y[f]))
    tracks = pd.DataFrame(rows, columns=["particle", "frame", "x", "y"])
    jdd = s.compute_jdd(tracks, px, dt, n_components=2)
    assert jdd is not None
    Ds = sorted(jdd["D_values"])
    assert 0.5 * D1 <= Ds[0] <= 2.0 * D1, f"slow D off: {Ds[0]:.3f} (true {D1:.3f})"
    assert 0.5 * D2 <= Ds[1] <= 2.0 * D2, f"fast D off: {Ds[1]:.3f} (true {D2:.3f})"


def test_mss_slope_brownian_near_half():
    """The moment-scaling-spectrum slope is ~0.5 for normal (Brownian)
    diffusion (1.0 = ballistic, <0.5 = subdiffusive)."""
    tracks = _synthetic_brownian_tracks(n_tracks=200, n_frames=60, sigma_px=2.0)
    mss = s.compute_mss(tracks, 0.1, 0.05, max_lagtime=10)
    assert mss is not None and len(mss) > 0
    assert 0.35 <= float(mss["mss_slope"].median()) <= 0.65


def test_turning_angles_symmetric_for_brownian():
    """Brownian motion has no directional bias: signed turning angles are
    symmetric about 0 (mean ~0) and span the full (-180, 180]."""
    tracks = _synthetic_brownian_tracks(n_tracks=150, n_frames=40, sigma_px=2.0)
    ang = s.compute_turning_angles(tracks)
    ang = np.asarray(ang, float)
    assert ang.size > 500
    assert ang.min() >= -180.001 and ang.max() <= 180.001
    assert abs(float(np.mean(ang))) < 10.0          # no net turning bias


def test_vacf_brownian_vs_directed():
    """Velocity autocorrelation: ~0 at lag 1 for Brownian (uncorrelated steps),
    strongly positive for directed motion (a persistent drift)."""
    rng = np.random.default_rng(11)
    px, dt = 0.1, 0.05
    # Brownian ensemble
    brn = _synthetic_brownian_tracks(n_tracks=200, n_frames=40, sigma_px=2.0)
    vb = s.compute_vacf(brn, dt, px, max_lag=8)
    assert vb is not None
    assert abs(vb["vacf"][0] - 1.0) < 1e-9          # normalised
    assert abs(vb["persistence"]) < 0.15            # uncorrelated steps

    # Directed: constant drift + small noise -> highly persistent velocity
    rows = []
    for pid in range(120):
        ang = rng.uniform(0, 2 * np.pi)
        vx, vy = 3.0 * np.cos(ang), 3.0 * np.sin(ang)
        x = y = 0.0
        for f in range(40):
            x += vx + rng.normal(0, 0.4); y += vy + rng.normal(0, 0.4)
            rows.append((pid, f, x, y))
    drv = pd.DataFrame(rows, columns=["particle", "frame", "x", "y"])
    vd = s.compute_vacf(drv, dt, px, max_lag=8)
    assert vd is not None
    assert vd["persistence"] > 0.6                  # strong directional memory


def test_msd_fit_rejects_bad_calibration():
    """compute_msd_and_fit raises a clear ValueError on non-positive pixel
    size / frame interval instead of silently returning garbage D/alpha."""
    tracks = _synthetic_brownian_tracks(n_tracks=5, n_frames=20, sigma_px=2.0)
    import pytest
    with pytest.raises(ValueError, match="pixel_size"):
        s.compute_msd_and_fit(tracks, 0.0, 0.05)
    with pytest.raises(ValueError, match="pixel_size"):
        s.compute_msd_and_fit(tracks, float("nan"), 0.05)
    with pytest.raises(ValueError, match="frame_interval"):
        s.compute_msd_and_fit(tracks, 0.1, -1.0)
    with pytest.raises(ValueError, match="max_lagtime"):
        s.compute_msd_and_fit(tracks, 0.1, 0.05, max_lagtime=0)


def test_alpha2_and_persistence_are_scale_time_invariant():
    """alpha2 (moment ratio) and VACF persistence (normalised) are
    dimensionless — invariant to pixel size and frame interval.  This is what
    lets Compare compute them per replicate with px=1/dt=1 from pixel-unit
    tracks and still get the calibrated value."""
    tracks = _synthetic_brownian_tracks(n_tracks=150, n_frames=40, sigma_px=2.0)
    a_unit = s.compute_van_hove(tracks, 1.0)["non_gaussian_alpha2"]
    a_cal  = s.compute_van_hove(tracks, 0.137)["non_gaussian_alpha2"]
    assert abs(a_unit - a_cal) < 1e-9

    p_unit = s.compute_vacf(tracks, 1.0, 1.0)["persistence"]
    p_cal  = s.compute_vacf(tracks, 0.042, 0.137)["persistence"]
    assert abs(p_unit - p_cal) < 1e-9


# ── CZI metadata parsing (regression for the Element/FrameTime fix) ──────────
def _czi_xml(frametime=None, timespan_ms=None, px_x_m=None):
    """Build minimal ZEN-style CZI metadata XML."""
    scaling = ""
    if px_x_m is not None:
        scaling = (f"<Scaling><Items>"
                   f'<Distance Id="X"><Value>{px_x_m}</Value></Distance>'
                   f'<Distance Id="Y"><Value>{px_x_m}</Value></Distance>'
                   f"</Items></Scaling>")
    ft = f"<FrameTime>{frametime}</FrameTime>" if frametime is not None else ""
    ts = ("<TimeSpan><Value>%s</Value><DefaultUnitFormat>ms</DefaultUnitFormat>"
          "</TimeSpan>" % timespan_ms) if timespan_ms is not None else ""
    return f"<ImageDocument><Metadata>{scaling}{ft}{ts}</Metadata></ImageDocument>"


def test_czi_metadata_frametime_and_pixel_size():
    from firefly.analysis.fa_loaders import _parse_czi_metadata
    m = _parse_czi_metadata(_czi_xml(frametime=0.05, px_x_m=1.04e-7))
    assert abs(m["pixel_size_um"] - 0.104) < 1e-6
    assert abs(m["frame_interval_s"] - 0.05) < 1e-9


def test_czi_metadata_accepts_element_and_bytes():
    """The aicspylibczi path passes an ElementTree Element, not a string —
    this was the bug fixed this session."""
    import xml.etree.ElementTree as ET
    from firefly.analysis.fa_loaders import _parse_czi_metadata
    xml = _czi_xml(frametime=0.02, px_x_m=1.6e-7)
    el = ET.fromstring(xml)
    m_el = _parse_czi_metadata(el)                 # Element input
    m_by = _parse_czi_metadata(xml.encode("utf-8"))  # bytes input
    for m in (m_el, m_by):
        assert abs(m["pixel_size_um"] - 0.16) < 1e-6
        assert abs(m["frame_interval_s"] - 0.02) < 1e-9


def test_czi_metadata_timespan_ms_fallback():
    """With no <FrameTime>, a ms <TimeSpan> is converted to seconds."""
    from firefly.analysis.fa_loaders import _parse_czi_metadata
    m = _parse_czi_metadata(_czi_xml(timespan_ms=30.0, px_x_m=1.0e-7))
    assert abs(m["frame_interval_s"] - 0.03) < 1e-9


def test_czi_metadata_handles_garbage():
    from firefly.analysis.fa_loaders import _parse_czi_metadata
    assert _parse_czi_metadata(None) == {"pixel_size_um": None, "frame_interval_s": None}
    assert _parse_czi_metadata("not xml at all <<<") == {"pixel_size_um": None, "frame_interval_s": None}


def test_aggregate_run_summaries(tmp_path):
    """aggregate_run_summaries globs per-run summary_metrics.json into one row
    per run, infers the condition from the parent folder, and flattens the
    nested motion_counts / qc blocks."""
    import json
    def _write(group, stem, **kw):
        d = tmp_path / group / stem / "firefly_extras"
        d.mkdir(parents=True)
        payload = {"stem": stem, "n_tracks": kw["nt"], "median_d": kw["md"],
                   "nongauss_alpha2": kw["a2"], "vacf_persistence": kw["p"],
                   "motion_counts": {"Immobile": kw["imm"], "Brownian": kw["brn"]},
                   "qc": {"link_ratio": 0.4, "median_track_length": 12.0,
                          "flags": [{"level": "warn", "msg": "x"}]}}
        (d / f"{stem}_summary_metrics.json").write_text(json.dumps(payload))
    _write("Control",    "B1R1", nt=900, md=0.05, a2=0.2, p=0.0,  imm=400, brn=500)
    _write("Isoflurane", "B2R1", nt=850, md=0.02, a2=0.7, p=-0.1, imm=700, brn=150)

    df = s.aggregate_run_summaries(str(tmp_path))
    assert len(df) == 2
    assert set(df["group"]) == {"Control", "Isoflurane"}
    assert {"n_tracks", "median_d", "nongauss_alpha2", "vacf_persistence",
            "motion_Immobile", "motion_Brownian", "link_ratio",
            "n_qc_flags"} <= set(df.columns)
    iso = df.set_index("group").loc["Isoflurane"]
    assert iso["nongauss_alpha2"] == 0.7 and iso["n_qc_flags"] == 1
    # empty root -> empty frame, no crash
    assert len(s.aggregate_run_summaries(str(tmp_path / "Control" / "B1R1" / "nope"))) == 0


def test_jdd_goodness_of_fit_prefers_correct_model():
    """JDD now reports R²/RMSE/AIC/BIC.  For a genuine 2-population mixture the
    2-component model should fit better (higher R²) and be preferred by BIC
    (lower) over the 1-component model."""
    rng = np.random.default_rng(7)
    px, dt = 0.1, 0.05
    rows = []
    for pid, sig in [(p, 1.0) for p in range(160)] + \
                    [(p, 4.5) for p in range(160, 320)]:
        x = np.cumsum(rng.normal(0, sig, 40)); y = np.cumsum(rng.normal(0, sig, 40))
        for f in range(40):
            rows.append((pid, f, x[f], y[f]))
    tracks = pd.DataFrame(rows, columns=["particle", "frame", "x", "y"])
    j1 = s.compute_jdd(tracks, px, dt, n_components=1)
    j2 = s.compute_jdd(tracks, px, dt, n_components=2)
    assert j1 is not None and j2 is not None
    for j in (j1, j2):
        assert set(["r_squared", "rmse", "aic", "bic", "n_params"]) <= set(j)
    assert j2["r_squared"] > j1["r_squared"]      # 2 fits the mixture better
    assert j2["bic"] < j1["bic"]                  # and BIC prefers it
    assert 0.0 <= j2["r_squared"] <= 1.0


def _write_run_folder(root, group, stem, n_tracks=25, n_frames=30, sigma_px=2.0,
                      seed=0):
    """Write a minimal FIREFLY run folder under root/group/stem/firefly_extras."""
    import json as _json
    rng = np.random.default_rng(seed)
    d = os.path.join(root, group, stem, "firefly_extras")
    os.makedirs(d)
    rows, drows = [], []
    for pid in range(n_tracks):
        t = np.cumsum(rng.normal(0, sigma_px, (n_frames, 2)), axis=0)
        for f in range(n_frames):
            rows.append((pid, f, t[f, 0], t[f, 1]))
        D = float(abs(rng.normal(0.05, 0.02)))
        drows.append((pid, D, float(rng.uniform(0.7, 1.1)),
                      "Brownian" if D > 0.04 else "Immobile"))
    pd.DataFrame(rows, columns=["particle", "frame", "x", "y"]).to_csv(
        os.path.join(d, f"{stem}_trajectories.csv"), index=False)
    pd.DataFrame(drows, columns=["particle", "D", "alpha", "motion"]).to_csv(
        os.path.join(d, f"{stem}_diffusion_summary.csv"), index=False)
    pd.DataFrame({"lag_frame": np.arange(1, 11),
                  "msd_um2": np.linspace(0.01, 0.1, 10)}).to_csv(
        os.path.join(d, f"{stem}_ensemble_msd.csv"), index=False)
    with open(os.path.join(d, f"{stem}_params.json"), "w") as fp:
        _json.dump({"pixel_size_um": 0.1, "frame_interval_s": 0.05}, fp)
    return os.path.join(root, group, stem)


def test_compare_groups_reports_alpha2_and_persistence_stats(tmp_path):
    """compare_groups runs end-to-end on synthetic run folders and the returned
    stats include the new per-replicate metrics (alpha2, VACF persistence)."""
    import matplotlib; matplotlib.use("Agg")
    from firefly.analysis.fa_compare import compare_groups
    root = str(tmp_path)
    g1 = [_write_run_folder(root, "Control", f"C{i}", sigma_px=2.0, seed=i)
          for i in range(3)]
    g2 = [_write_run_folder(root, "Iso", f"I{i}", sigma_px=3.5, seed=10 + i)
          for i in range(3)]
    groups = [{"folders": g1, "label": "Control", "color": "#4a90d9"},
              {"folders": g2, "label": "Iso", "color": "#e05252"}]
    fig, summary_df, stats = compare_groups(
        groups, output_dir=str(tmp_path / "out"), pdf_report=False)
    # per-replicate scalars present
    assert {"nongauss_alpha2", "vacf_persistence"} <= set(summary_df.columns)
    assert len(summary_df) == 6
    # and they were tested across groups
    assert "nongauss_alpha2" in stats and "vacf_persistence" in stats
    assert "omnibus" in stats["nongauss_alpha2"]


def test_jdd_and_mss_reject_bad_calibration():
    """compute_jdd / compute_mss validate calibration like compute_msd_and_fit
    (no silent garbage on zero/NaN pixel size or frame interval)."""
    import pytest
    tracks = _synthetic_brownian_tracks(n_tracks=20, n_frames=30, sigma_px=2.0)
    with pytest.raises(ValueError, match="pixel_size"):
        s.compute_jdd(tracks, 0.0, 0.05)
    with pytest.raises(ValueError, match="frame_interval"):
        s.compute_jdd(tracks, 0.1, float("nan"))
    with pytest.raises(ValueError, match="pixel_size"):
        s.compute_mss(tracks, -1.0, 0.05)
    with pytest.raises(ValueError, match="frame_interval"):
        s.compute_mss(tracks, 0.1, 0.0)


# ── Auto-threshold (estimate_minmass) knee floor ─────────────────────────────
def _noisy_blink_stack(n_frames=18, H=192, W=192, n_spots=120, noise=30.0,
                       amp=(10, 60), sigma=1.0, seed=0):
    """Dense, low-contrast frames whose blink brightness overlaps the noise —
    the regime where the mass distribution is unimodal and the auto-threshold
    falls back to the mass-quantile branch."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    stack = np.zeros((n_frames, H, W), dtype=np.float32)
    for f in range(n_frames):
        img = rng.normal(100, noise, (H, W)).astype(np.float32)
        for _ in range(n_spots):
            cx, cy = rng.uniform(5, W - 5), rng.uniform(5, H - 5)
            img += rng.uniform(*amp) * np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
        stack[f] = img
    return stack


def test_auto_threshold_knee_floor_engages_on_noise_dominated_data():
    """On dense/noise-dominated data the mass-quantile cut sits in the noise;
    the knee floor must raise it to the noise/signal knee (the fix for the
    'auto threshold detects everything' failure)."""
    from firefly.analysis.fa_localize import estimate_minmass
    import pytest
    stack = _noisy_blink_stack(seed=1)
    # link_min_len huge → force the static estimator (the knee-floor logic).
    mm, diag = estimate_minmass(stack, diameter=7, backend="trackpy",
                                sensitivity="balanced", bg_radius=10,
                                workers=2, log_cb=lambda m: None,
                                link_min_len=999)
    assert "quantile" in (diag.get("static_method") or diag["method"])
    assert diag["knee"] is not None
    assert diag.get("knee_floor_applied") is True     # floor engaged
    # cut was raised exactly to the knee, and is strictly above the raw p30
    assert mm == pytest.approx(10 ** diag["knee"], rel=1e-6)


def test_auto_threshold_knee_floor_is_noop_on_clean_bimodal_data():
    """When a clean noise/signal valley exists (GMM), the chosen cut already
    sits above the knee, so the floor must NOT fire — no regression."""
    from firefly.analysis.fa_localize import estimate_minmass
    # Bright, well-separated blinks → clean bimodal mass distribution.
    stack = _noisy_blink_stack(n_spots=20, noise=12.0, amp=(120, 260),
                               sigma=1.4, seed=0)
    mm, diag = estimate_minmass(stack, diameter=7, backend="trackpy",
                                sensitivity="balanced", bg_radius=10,
                                workers=2, log_cb=lambda m: None,
                                link_min_len=999)
    assert "gmm" in (diag.get("static_method") or diag["method"])
    assert not diag.get("knee_floor_applied")


def test_mss_computes_for_short_tracks():
    """MSS now uses tracks down to 10 frames (>=4 usable lags) instead of
    demanding max_lagtime+2=12 — so short single-molecule (sptPALM) tracks
    contribute a slope instead of an empty 'tracks too short' panel."""
    ten = _synthetic_brownian_tracks(n_tracks=150, n_frames=10, sigma_px=2.0)
    nine = _synthetic_brownian_tracks(n_tracks=150, n_frames=9, sigma_px=2.0)
    mss10 = s.compute_mss(ten, 0.1, 0.05)
    mss9  = s.compute_mss(nine, 0.1, 0.05)
    assert len(mss10) == 150            # 10-frame tracks now qualify
    assert len(mss9) == 0               # 9 frames -> only 3 lags -> still skipped
    assert 0.3 <= float(mss10["mss_slope"].median()) <= 0.65   # ~0.5 Brownian


def test_mss_handles_frame_as_index_and_column():
    """Regression: trackpy.link leaves 'frame' as both an index level and a
    column; compute_mss must not raise the pandas 'ambiguous' ValueError
    (it did after the MSS perf rewrite dropped reset_index)."""
    tracks = _synthetic_brownian_tracks(n_tracks=60, n_frames=15, sigma_px=2.0)
    tracks = tracks.set_index("frame", drop=False)   # mimic real linker output
    assert "frame" in tracks.columns and tracks.index.name == "frame"
    mss = s.compute_mss(tracks, 0.1, 0.05)
    assert len(mss) == 60                            # computed, did not crash
    assert 0.3 <= float(mss["mss_slope"].median()) <= 0.65
