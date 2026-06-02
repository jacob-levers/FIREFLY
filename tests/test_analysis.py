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
    mm, diag = _quiet_estimate(stack, diameter=7, sensitivity="balanced",
                               frame_sample=20)
    assert diag["method"].startswith("gmm"), diag["method"]
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
    strict = _quiet_estimate(stack, diameter=7, sensitivity="strict", frame_sample=20)[0]
    bal    = _quiet_estimate(stack, diameter=7, sensitivity="balanced", frame_sample=20)[0]
    lenient = _quiet_estimate(stack, diameter=7, sensitivity="lenient", frame_sample=20)[0]
    assert strict >= bal >= lenient, (strict, bal, lenient)


def test_estimate_minmass_noise_only_falls_back():
    """A noise-only stack (no real spots) must not crash and should return a
    finite, clamped threshold via a fallback path."""
    rng = np.random.default_rng(2)
    stack = rng.poisson(30, (20, 128, 128)).astype(np.float32)
    mm, diag = _quiet_estimate(stack, diameter=7, sensitivity="balanced",
                               frame_sample=20)
    assert np.isfinite(mm) and mm >= 0.05
    assert diag["method"] is not None


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
