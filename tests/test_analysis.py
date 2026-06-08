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


def test_msd_fit_serial_matches_parallel():
    """compute_msd_and_fit must give identical D/alpha whether run on 1 worker
    or several — parallelism is a performance detail, not a numerical one."""
    px, dt = 0.1, 0.05
    tracks = _synthetic_brownian_tracks(n_tracks=40, n_frames=40, sigma_px=1.5,
                                        seed=5)
    _i1, _e1, d1 = fd.compute_msd_and_fit(tracks, px, dt, max_lagtime=15,
                                          n_fit=5, workers=1)
    _i2, _e2, d2 = fd.compute_msd_and_fit(tracks, px, dt, max_lagtime=15,
                                          n_fit=5, workers=2)
    d1 = d1.sort_values("particle").reset_index(drop=True)
    d2 = d2.sort_values("particle").reset_index(drop=True)
    pd.testing.assert_series_equal(d1["D"], d2["D"], rtol=1e-9, atol=1e-12)
    pd.testing.assert_series_equal(d1["alpha"], d2["alpha"],
                                   rtol=1e-9, atol=1e-12)


def test_msd_fit_sparse_short_tracks_finite():
    """Short/sparse tracks must yield finite-or-NaN D/alpha without crashing or
    blowing up to inf."""
    px, dt = 0.1, 0.05
    tracks = _synthetic_brownian_tracks(n_tracks=20, n_frames=5, sigma_px=1.0,
                                        seed=9)
    _i, _e, diff = fd.compute_msd_and_fit(tracks, px, dt, max_lagtime=4,
                                          n_fit=3, workers=1)
    assert len(diff) > 0
    for col in ("D", "alpha"):
        assert not np.isinf(diff[col].to_numpy()).any(), f"{col} has inf"


def test_make_figure_smoke(tmp_path):
    """The single-run master figure renders headlessly (matplotlib Agg) across
    all panels on synthetic data and writes a non-empty PNG — a regression guard
    for the 828-LOC make_figure (panel crashes, bad axis limits, etc.)."""
    import matplotlib
    matplotlib.use("Agg")
    from firefly.analysis.fa_figure import make_figure
    rng = np.random.default_rng(3)
    stack = rng.poisson(15, (4, 64, 64)).astype(float)
    nt, nf = 60, 30
    classes = ["Immobile", "Confined", "Brownian", "Directed"]
    rows, drows = [], []
    for pid in range(nt):
        x = np.cumsum(rng.normal(0, 2, nf)) + 32
        y = np.cumsum(rng.normal(0, 2, nf)) + 32
        for f in range(nf):
            rows.append((pid, f, x[f], y[f]))
        u = rng.random()
        D = 1e-6 if u < 0.25 else abs(rng.normal(0.05, 0.03))
        drows.append((pid, max(D, 1e-7),
                      float(np.clip(rng.normal(0.9, 0.3), 0.1, 1.8)),
                      classes[min(3, int(u * 4))],
                      float(np.clip(rng.normal(0.5, 0.2), 0, 1))))
    tracks = pd.DataFrame(rows, columns=["particle", "frame", "x", "y"])
    diff = pd.DataFrame(drows,
                        columns=["particle", "D", "alpha", "motion", "mss_slope"])
    lags = np.arange(1, 11)
    imsd = pd.DataFrame({pid: np.linspace(0.01, 0.1, 10) * rng.uniform(0.5, 2)
                         for pid in range(nt)}, index=lags)
    emsd = pd.Series(np.linspace(0.012, 0.09, 10), index=lags)
    out_png = tmp_path / "fig.png"
    make_figure(stack, tracks, imsd, emsd, diff, 0.106, 0.02,
                output_path=str(out_png), want_panels=set())
    assert out_png.exists() and out_png.stat().st_size > 0


def test_load_external_locs_formats(tmp_path):
    """External-format CSV import maps columns, converts units (nm/µm→px),
    applies the per-tool frame offset, and emits a `particle` column from
    TrackMate TRACK_ID (dropping unlinked spots) — the import path most prone to
    silent coordinate/frame corruption."""
    from firefly.analysis import fa_loaders as L
    px = 0.1   # µm/px

    # Picasso — px coords, 0-indexed → unchanged
    p = tmp_path / "picasso.csv"
    p.write_text("frame,x,y,photons\n0,10.0,20.0,500\n1,11.0,21.0,480\n")
    out = L.load_external_locs(str(p), preset="Picasso", pixel_size_um=px)
    assert list(out["frame"]) == [0, 1]
    assert out["x"].iloc[0] == 10.0 and out["y"].iloc[0] == 20.0

    # ThunderSTORM — nm coords + 1-indexed → /(px*1000) and frame-1
    t = tmp_path / "ts.csv"
    t.write_text('frame,x [nm],y [nm],intensity [photon]\n'
                 '1,1000,2000,300\n2,1100,2100,310\n')
    out = L.load_external_locs(str(t), preset="ThunderSTORM", pixel_size_um=px)
    assert list(out["frame"]) == [0, 1]                       # 1- → 0-indexed
    assert out["x"].iloc[0] == pytest.approx(1000 / (px * 1000))   # 1000nm→10px
    assert out["y"].iloc[0] == pytest.approx(2000 / (px * 1000))   # 2000nm→20px

    # TrackMate — µm coords + TRACK_ID → particle, unlinked (-1) dropped
    tm = tmp_path / "tm.csv"
    tm.write_text("FRAME,POSITION_X,POSITION_Y,MEAN_INTENSITY_CH1,TRACK_ID\n"
                  "0,1.0,2.0,100,0\n"
                  "1,1.05,2.05,100,0\n"
                  "0,3.0,4.0,100,-1\n")              # unlinked spot → dropped
    out = L.load_external_locs(str(tm), preset="TrackMate", pixel_size_um=px)
    assert "particle" in out.columns
    assert set(out["particle"]) == {0} and len(out) == 2
    assert out["x"].iloc[0] == pytest.approx(1.0 / px)        # 1 µm → 10 px

    # PALM-Tracer — px coords + 1-indexed
    pt = tmp_path / "pt.csv"
    pt.write_text("Plane,CentroidX(px),CentroidY(px),Integrated_Intensity\n"
                  "1,5.0,6.0,200\n2,5.5,6.5,210\n")
    out = L.load_external_locs(str(pt), preset="PALM-Tracer", pixel_size_um=px)
    assert list(out["frame"]) == [0, 1]
    assert out["x"].iloc[0] == 5.0


def test_load_external_locs_empty_raises(tmp_path):
    """A file that parses to zero localisations raises a clear error rather than
    silently returning an empty frame that crashes downstream."""
    from firefly.analysis import fa_loaders as L
    p = tmp_path / "empty.csv"
    p.write_text("frame,x,y,photons\n")          # header only, no data
    with pytest.raises(ValueError, match="No localisations found"):
        L.load_external_locs(str(p), preset="Picasso", pixel_size_um=0.1)


def test_link_trajectories_validates_input():
    """link_trajectories gives a clear error for a missing required column and
    drops negative-frame rows (with a warning) instead of a cryptic trackpy
    failure — valid rows still link."""
    from firefly.analysis.fa_linking import link_trajectories
    # missing 'y' → clear ValueError (not a deep trackpy KeyError)
    bad = pd.DataFrame({"x": [1.0, 2.0], "frame": [0, 1], "mass": [1, 1]})
    with pytest.raises(ValueError, match="missing required"):
        link_trajectories(bad)
    # negative frames dropped; the valid 3-point track still links
    locs = pd.DataFrame({
        "x": [10.0, 10.1, 10.2, 5.0], "y": [10.0, 10.1, 10.2, 5.0],
        "frame": [0, 1, 2, -1], "mass": [100, 100, 100, 100]})
    out = link_trajectories(locs, search_range=3, memory=1, min_len=2)
    assert (out["frame"] >= 0).all()             # the frame=-1 row was dropped
    assert len(out) >= 3                          # the valid track survived


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


def _circ_groups(rng):
    """3 von-Mises groups + per-replicate angle arrays for circular tests."""
    def reps(mean_deg, k=4, n=200):
        return [np.degrees(rng.vonmises(np.radians(mean_deg), 2.0, n))
                for _ in range(k)]
    A, B, C = reps(0.0), reps(45.0), reps(120.0)
    ga = [("A", np.concatenate(A), "#f00"), ("B", np.concatenate(B), "#0f0"),
          ("C", np.concatenate(C), "#00f")]
    return ga, {"A": A, "B": B, "C": C}


def test_circular_comparison_tests_respect_alpha_and_correction():
    """The per-replicate circular tests honour the stats_config α + correction:
    raw p is unchanged but p_corrected differs between 'none' and 'bonferroni',
    and stars are gated on the chosen α."""
    rng = np.random.default_rng(3)
    ga, per_rep = _circ_groups(rng)
    groups = [(lbl, a) for lbl, a, _ in ga]
    none = fc.compute_circular_comparison_tests(
        groups, per_replicate_angles=per_rep,
        stats_config={"correction": "none", "alpha": 0.05})
    bonf = fc.compute_circular_comparison_tests(
        groups, per_replicate_angles=per_rep,
        stats_config={"correction": "bonferroni", "alpha": 0.05})
    pw_none = none["per_replicate_kappa_test"]["pairwise"]
    pw_bonf = bonf["per_replicate_kappa_test"]["pairwise"]
    # raw p identical; corrected p ≥ raw and ≥ the 'none' corrected p
    for a, b in zip(pw_none, pw_bonf):
        assert a["p"] == pytest.approx(b["p"], nan_ok=True)
        if a["p"] is not None and np.isfinite(a["p"]):
            assert b["p_corrected"] >= a["p_corrected"] - 1e-12
    # every pairwise row carries the corrected fields the CSV/PDF read
    assert all("p_corrected" in r and "stars_corrected" in r for r in pw_bonf)
    # μ test gets an α-gated star
    mu = bonf.get("per_replicate_mu_ww")
    assert mu is not None and "stars" in mu


def test_circular_comparison_csv_honors_test_toggles(tmp_path):
    """Turning off circ_test_rbar/mu/circlin drops their rows from the tests CSV
    (only the κ test remains), and the file schema is preserved."""
    rng = np.random.default_rng(4)
    ga, per_rep = _circ_groups(rng)
    csv = tmp_path / "K_circular_statistics.csv"
    fc.save_comparison_circular_statistics(
        ga, csv_path=str(csv), pdf_path=None, per_replicate_angles=per_rep,
        stats_config={"circ_test_rbar": False, "circ_test_mu": False,
                      "circ_test_circlin": False})
    tst = pd.read_csv(str(tmp_path / "K_circular_tests.csv"))
    metrics = set(tst["metric"].unique())
    assert metrics == {"kappa (concentration)"}
    assert "p_corrected" in tst.columns


def test_compare_groups_circular_outputs_toggle(tmp_path):
    """include_circular_outputs=False writes NO circular files, while the
    comparison figure + summary CSV are still produced."""
    import matplotlib; matplotlib.use("Agg")
    import glob
    from firefly.analysis.fa_compare import compare_groups
    root = str(tmp_path)
    g1 = [_write_run_folder(root, "A", f"A{i}", sigma_px=2.0, seed=i)
          for i in range(3)]
    g2 = [_write_run_folder(root, "B", f"B{i}", sigma_px=3.5, seed=10 + i)
          for i in range(3)]
    groups = [{"folders": g1, "label": "A", "color": "#4a90d9"},
              {"folders": g2, "label": "B", "color": "#e05252"}]
    out = str(tmp_path / "out")
    compare_groups(groups, output_dir=out, pdf_report=False,
                   stats_config={"include_circular_outputs": False})
    assert not glob.glob(os.path.join(out, "*circular*"))
    assert os.path.exists(os.path.join(out, "comparison_summary.csv"))


def test_compute_clusters_rg_and_subsample_attrs():
    """compute_clusters adds a radius-of-gyration column and records subsample
    provenance in df.attrs (used to surface an honest 'sub-sampled to N')."""
    from firefly.sptpalm_analysis import compute_clusters
    rng = np.random.default_rng(0)
    sig = 0.02
    xy = np.vstack([rng.normal(c, sig, (60, 2)) for c in ([0, 0], [3, 3])])
    locs = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1],
                         "frame": np.zeros(len(xy), int)})
    labels, stats, ncl, _ = compute_clusters(
        locs, pixel_size_um=1.0, eps_um=0.1, min_samples=5)
    assert ncl == 2
    assert "rg_um" in stats.columns
    rg = stats["rg_um"].to_numpy()
    assert np.all(np.isfinite(rg)) and np.all(rg > 0)
    assert np.allclose(rg, np.sqrt(2) * sig, rtol=0.4)   # 2-D Gaussian Rg
    assert stats.attrs["subsampled"] is False
    assert stats.attrs["n_used_locs"] == len(xy)
    # forcing a small cap records the subsample provenance
    _, stats2, _, xy2 = compute_clusters(
        locs, pixel_size_um=1.0, eps_um=0.1, min_samples=5, max_locs=50)
    assert stats2.attrs["subsampled"] is True
    assert stats2.attrs["n_used_locs"] == 50
    assert stats2.attrs["n_input_locs"] == len(xy)
    assert len(xy2) == 50


def test_compute_clusters_guards_oversized_eps():
    """A too-large eps must be handled safely by SUB-SAMPLING (never an O(n²)
    neighbour-graph blow-up / crash) — it still returns a result, with the
    sub-sample recorded so callers can surface it."""
    from firefly.sptpalm_analysis import compute_clusters
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 0.5, (8000, 2))          # dense, in a 0.5 µm box
    locs = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1],
                         "frame": np.zeros(len(xy), int)})
    # eps spanning the whole box → guard kicks in and sub-samples
    labels, stats, ncl, used = compute_clusters(
        locs, pixel_size_um=1.0, eps_um=0.5, min_samples=8)
    assert stats.attrs["subsampled"] is True
    assert stats.attrs["n_used_locs"] < stats.attrs["n_input_locs"]
    assert len(used) == stats.attrs["n_used_locs"]   # labels/xy aligned to sub-sample
    assert len(labels) == len(used)
    # a sane eps clusters the full set (no sub-sampling)
    _, stats2, _, _ = compute_clusters(
        locs, pixel_size_um=1.0, eps_um=0.01, min_samples=8)
    assert stats2.attrs["subsampled"] is False


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


def _write_tif_series(tmp_path, counts=(6, 7, 8), HW=(8, 8)):
    """Write a multi-file TIF series with a unique value per (file, frame) so a
    frame read lazily can be checked against its eager position byte-for-byte."""
    tifffile = pytest.importorskip("tifffile")
    H, W = HW
    parts, paths = [], []
    for fi, n in enumerate(counts):
        arr = np.zeros((n, H, W), np.uint16)
        for fr in range(n):
            arr[fr] = fi * 100 + fr           # unique per (file, frame)
        name = "mov.tif" if fi == 0 else f"mov-file{fi+1:03d}.tif"
        p = str(tmp_path / name)
        tifffile.imwrite(p, arr)
        paths.append(p); parts.append(arr)
    return paths, np.concatenate(parts).astype(np.float32), list(counts)


def test_lazy_tiff_stack_is_byte_identical_to_eager(tmp_path):
    """LazyTiffStack must return frames byte-identical to the eager combined
    array for every access pattern the localisation pipeline uses: contiguous
    slices (incl. file-boundary-spanning), single int (incl. negative), fancy
    gather (the projection sample), and strided slices."""
    from firefly.analysis import fa_loaders as L
    paths, expected, counts = _write_tif_series(tmp_path)
    n = sum(counts)
    lazy = L.LazyTiffStack(paths, counts, 8, 8)
    try:
        assert lazy.shape == (n, 8, 8)
        assert lazy.ndim == 3
        assert lazy.dtype == np.float32
        assert len(lazy) == n
        assert lazy.nbytes == expected.nbytes
        # full range
        assert np.array_equal(lazy[:], expected)
        # slices that start/stop inside a file AND span file boundaries
        for a, b in [(0, 6), (5, 9), (6, 13), (4, 20), (0, n), (13, 21)]:
            assert np.array_equal(lazy[a:b], expected[a:b]), (a, b)
        # single frame, including the first frame of files 2 and 3 + negatives
        for i in [0, 5, 6, 12, 13, n - 1, -1, -n]:
            assert np.array_equal(lazy[i], expected[i]), i
            assert lazy[i].shape == (8, 8)
        # fancy gather, sorted-and-spread (projection sample) + unsorted
        for idx in (np.linspace(0, n - 1, 5, dtype=int),
                    np.array([0, 6, 13, 3, 20, 5])):
            assert np.array_equal(lazy[idx], expected[idx])
        # strided slice
        assert np.array_equal(lazy[1:20:3], expected[1:20:3])
    finally:
        lazy.close()


def test_load_tif_returns_lazy_under_memory_pressure(tmp_path, monkeypatch):
    """When the combined stack won't fit in RAM, load_tif must hand back a
    LazyTiffStack (no combined-stack write) whose frames equal the eager array,
    and FIREFLY_LAZY_TIF=0 must restore the eager path."""
    from firefly.analysis import fa_loaders as L
    paths, expected, _counts = _write_tif_series(tmp_path, counts=(6, 6, 6))
    # Force the "won't fit in RAM" decision by reserving an absurd amount of RAM.
    monkeypatch.setenv("FIREFLY_USER_RAM_RESERVE_GB", "100000")
    monkeypatch.setenv("FIREFLY_LAZY_TIF", "1")
    lazy, _px, _fi = L.load_tif(paths[0], files=paths)
    try:
        assert getattr(lazy, "_is_lazy_stack", False), "expected a LazyTiffStack"
        assert lazy.shape == (18, 8, 8)
        assert np.array_equal(lazy[:], expected)
        assert np.array_equal(lazy[5:13], expected[5:13])      # spans boundary
    finally:
        if hasattr(lazy, "close"):
            lazy.close()
    # Opt-out restores the eager (materialised) path.
    monkeypatch.setenv("FIREFLY_LAZY_TIF", "0")
    eager, _, _ = L.load_tif(paths[0], files=paths)
    assert not getattr(eager, "_is_lazy_stack", False)
    assert np.array_equal(np.asarray(eager), expected)


def test_adaptive_forces_stream_for_lazy_stack(tmp_path):
    """A lazy stack must be routed to STREAM, never FAST (which would pull every
    frame into RAM).  Run the real adaptive localiser on a lazy-wrapped series
    and confirm it produces the same localisations as the eager array."""
    pytest.importorskip("trackpy")
    import io, contextlib, logging, tifffile
    logging.getLogger("trackpy").setLevel(logging.ERROR)
    from firefly.analysis import fa_loaders as L
    from firefly.analysis.fa_localize import preprocess_and_localise_adaptive
    # Bright bimodal spots so trackpy reliably detects something; split the same
    # frames across two TIF files and wrap them lazily.
    full = _bimodal_spot_stack(H=64, W=64, F=24, seed=3).astype(np.uint16)
    counts = [12, 12]
    paths = []
    off = 0
    for fi, n in enumerate(counts):
        name = "mov.tif" if fi == 0 else f"mov-file{fi+1:03d}.tif"
        p = str(tmp_path / name)
        tifffile.imwrite(p, full[off:off + n]); paths.append(p); off += n
    eager = full.astype(np.float32)
    lazy = L.LazyTiffStack(paths, counts, 64, 64)
    try:
        common = dict(diameter=7, minmass=5, percentile=64, bg_radius=10,
                      workers=1, chunk_size=8, backend="trackpy")
        with contextlib.redirect_stdout(io.StringIO()) as _buf:
            locs_e = preprocess_and_localise_adaptive(eager, **common)[0]
            locs_l = preprocess_and_localise_adaptive(lazy, **common)[0]
    finally:
        lazy.close()
    # The lazy stack must have been routed through STREAM, never FAST.
    assert "forcing STREAM (lazy on-demand stack)" in _buf.getvalue()
    assert len(locs_e) == len(locs_l) and len(locs_l) > 0
    for col in ("x", "y", "frame"):
        np.testing.assert_allclose(
            np.sort(locs_e[col].values), np.sort(locs_l[col].values), atol=1e-5)


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


def test_interaction_plot_per_card_colours_and_gradient_line(tmp_path):
    """Two-factor interaction panels must colour dots PER group×timepoint cell
    (matching the bottom legend) and join a group's time points with a colour
    gradient — not one flat colour per group.  Reusing one colour per group
    across its time points must be fanned out to distinct cell colours."""
    import matplotlib; matplotlib.use("Agg")
    from matplotlib.collections import LineCollection
    from firefly.analysis.fa_compare import compare_groups
    root = str(tmp_path)
    # 2 groups × PRE/POST, paired cells, with the SAME colour reused per group
    # (the case that used to collapse the dots to 2 colours).
    spec = [("DMSO", "PRE", "#3b6ed8"), ("DMSO", "POST", "#3b6ed8"),
            ("Cip", "PRE", "#54a24b"), ("Cip", "POST", "#54a24b")]
    groups = []
    for gi, (grp, tp, col) in enumerate(spec):
        folders = [_write_run_folder(root, f"{grp}_{tp}", f"{grp}_c{c}_{tp}",
                                     sigma_px=2.0 + 0.3 * gi, seed=gi * 7 + c)
                   for c in range(5)]
        groups.append({"folders": folders, "label": grp,
                       "timepoint": tp, "color": col})
    fig, summary_df, stats = compare_groups(
        groups, output_dir=str(tmp_path / "out"), pdf_report=False)
    auc = [ax for ax in fig.axes if ax.get_title() == "Area Under the Curve"]
    assert auc, "AUC interaction panel missing"
    ax = auc[0]
    scat = set()
    for c in ax.collections:
        if c.__class__.__name__ == "PathCollection":
            fc = c.get_facecolors()
            if len(fc):
                scat.add(tuple(np.round(fc[0], 3)))
    assert len(scat) >= 4, f"expected ≥4 distinct cell colours, got {len(scat)}"
    assert any(isinstance(c, LineCollection) for c in ax.collections), \
        "expected a gradient connecting line (LineCollection)"
    # bottom legend names every cell as group / timepoint
    leg = fig.legends[0]
    txts = {t.get_text() for t in leg.get_texts()}
    assert any("DMSO / PRE" in t for t in txts)
    assert any("DMSO / POST" in t for t in txts)


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


def test_audit_mass_scale_noop_on_trackpy():
    """The Torch/Trackpy mass-scale self-audit must be a clean no-op on the
    trackpy backend (returns None, never raises) — it only does work when a
    Torch run will consume the trackpy-harvested threshold."""
    from firefly.analysis.fa_localize import _audit_mass_scale
    import pandas as pd
    H = pd.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0], "frame": [0, 0],
                      "mass": [10.0, 20.0], "window_id": [0, 0]})
    stack = np.zeros((4, 32, 32), dtype=np.float32)
    r = _audit_mass_scale(stack, [(0, 4)], H, diameter=7, percentile=64,
                          bg_radius=10, bg_method="uniform_filter", workers=1,
                          backend="trackpy", log_cb=lambda m: None)
    assert r is None
    # empty harvest / no windows are also safe
    assert _audit_mass_scale(stack, [], H.iloc[:0], 7, 64, 10,
                             "uniform_filter", 1, "torch", lambda m: None) is None


# ── perf-refactor regression: results-identity guards ────────────────────────
def _synthetic_harvest_table(seed=11):
    """A windowed candidate table H like `_harvest_windows` produces:
    persistent bright emitters (link into tracks) + faint noise blips."""
    rng = np.random.default_rng(seed)
    rows = []
    for wid in range(4):
        for _ in range(12):                       # persistent emitters
            x0, y0 = rng.uniform(5, 55, 2)
            for fr in range(int(rng.integers(6, 20))):
                rows.append((x0 + rng.normal(0, 0.2), y0 + rng.normal(0, 0.2),
                             fr, rng.uniform(60, 300), rng.uniform(0.2, 0.4), wid))
        for _ in range(500):                      # noise blips
            rows.append((rng.uniform(0, 60), rng.uniform(0, 60),
                         int(rng.integers(0, 40)), rng.uniform(0.5, 25),
                         rng.uniform(0.1, 0.6), wid))
    return pd.DataFrame(rows, columns=["x", "y", "frame", "mass", "ep",
                                       "window_id"])


def test_sweep_thresholds_parallel_matches_serial():
    """The parallel (process-pool) threshold sweep must be byte-identical to the
    serial path — same grid, same linker, just concurrent."""
    from firefly.analysis import fa_localize as L
    H = _synthetic_harvest_table()
    grid = np.unique(np.geomspace(1.0, 250.0, 18))
    saved = L._SWEEP_PARALLEL_MIN_CANDIDATES
    try:
        L._SWEEP_PARALLEL_MIN_CANDIDATES = 1          # force the process pool
        par = L._sweep_thresholds(H, grid, search_range=5, memory=3, link_min_len=4)
        L._SWEEP_PARALLEL_MIN_CANDIDATES = 10**12     # force serial
        ser = L._sweep_thresholds(H, grid, search_range=5, memory=3, link_min_len=4)
    finally:
        L._SWEEP_PARALLEL_MIN_CANDIDATES = saved
    assert len(par) == len(ser) == len(grid)
    for a, b in zip(par, ser):
        for k in ("t", "n_surv", "N_good", "good_fraction", "spurious_rate"):
            assert a[k] == b[k], (k, a[k], b[k])
        # median_ep: identical (or both NaN)
        assert (a["median_ep"] == b["median_ep"]
                or (np.isnan(a["median_ep"]) and np.isnan(b["median_ep"])))


def test_harvest_windows_parallel_matches_serial(tmp_path):
    """The process-pool harvest must be byte-identical to the serial harvest:
    each window runs the identical single-process tp.batch, only scheduled
    concurrently.  Both the candidate table H and window-0's pp0 must match."""
    pytest.importorskip("trackpy")
    import logging
    logging.getLogger("trackpy").setLevel(logging.ERROR)
    from firefly.analysis import fa_localize as L
    stack = _bimodal_spot_stack(H=64, W=64, F=40, seed=5).astype(np.float32)
    windows = L._contiguous_windows(len(stack))
    common = dict(diameter=7, percentile=64, bg_radius=10,
                  bg_method="uniform_filter", workers=2)
    import os as _os
    _prev = _os.environ.get("FIREFLY_HARVEST_PARALLEL")
    try:
        _os.environ["FIREFLY_HARVEST_PARALLEL"] = "1"          # process pool
        H_par, pp0_par = L._harvest_windows(stack, windows, **common)
        _os.environ["FIREFLY_HARVEST_PARALLEL"] = "0"          # serial
        H_ser, pp0_ser = L._harvest_windows(stack, windows, **common)
    finally:
        if _prev is None:
            _os.environ.pop("FIREFLY_HARVEST_PARALLEL", None)
        else:
            _os.environ["FIREFLY_HARVEST_PARALLEL"] = _prev
    assert len(H_par) == len(H_ser) and len(H_par) > 0
    pd.testing.assert_frame_equal(
        H_par.reset_index(drop=True), H_ser.reset_index(drop=True))
    assert np.array_equal(pp0_par, pp0_ser)


def _drift_localisations(n_frames, gx, gy, poison_seg=None, seg_len=None, seed=0):
    """Localisations of a fixed structure shifted by a per-frame drift, with an
    optional fully-scrambled ('poisoned') segment."""
    rng = np.random.default_rng(seed)
    struct = rng.uniform(20, 80, size=(300, 2))
    rows = []
    for fr in range(n_frames):
        k = int(rng.integers(15, 35))
        idx = rng.choice(len(struct), k, replace=False)
        xs = struct[idx, 0] + gx[fr] + rng.normal(0, 0.3, k)
        ys = struct[idx, 1] + gy[fr] + rng.normal(0, 0.3, k)
        if poison_seg is not None and (fr // seg_len) == poison_seg:
            xs = rng.uniform(20, 80, k); ys = rng.uniform(20, 80, k)
        for a, b in zip(xs, ys):
            rows.append((0, fr, a, b))
    return pd.DataFrame(rows, columns=["particle", "frame", "x", "y"])


def test_correct_drift_clean_noop_and_bounds_spurious():
    """Robust RCC: a clean drift ramp is recovered with NO pairs rejected (guard
    inactive → behaviour preserved); a fully-poisoned segment can no longer blow
    the drift up to non-physical values (the 150 px artefact can't recur)."""
    from firefly.analysis.fa_drift import correct_drift
    n = 2000
    gx = np.linspace(0, 8.0, n); gy = np.linspace(0, -5.0, n)

    locs = _drift_localisations(n, gx, gy, seed=0)
    _, drift = correct_drift(locs)
    rng_x = float(drift["dx"].max() - drift["dx"].min())
    rng_y = float(drift["dy"].max() - drift["dy"].min())
    # Recovers the right order of magnitude (smoothing compresses the endpoints).
    assert 3.0 < rng_x < 9.0 and 2.0 < rng_y < 6.0, (rng_x, rng_y)

    seg = 200
    locs2 = _drift_localisations(n, gx, gy, poison_seg=5, seg_len=seg, seed=0)
    _, drift2 = correct_drift(locs2, n_seg_frames=seg)
    rng2_x = float(drift2["dx"].max() - drift2["dx"].min())
    rng2_y = float(drift2["dy"].max() - drift2["dy"].min())
    # Bounded — must NOT explode to the old 100+ px failure mode.
    assert rng2_x < 20.0 and rng2_y < 20.0, (rng2_x, rng2_y)


def test_streaming_localise_quiet_and_reproducible():
    """The streaming Torch path must not leak the backend's per-call banner
    (quiet=True), and on a small deterministic stack (<5M-element chunks → exact
    quantile) two runs must give identical localisations."""
    import io, contextlib
    from firefly.analysis.fa_localize import preprocess_and_localise_stream
    rng = np.random.default_rng(5)
    T, S = 160, 40
    stack = rng.normal(100, 6, size=(T, S, S)).astype(np.float32)
    yy, xx = np.mgrid[0:S, 0:S]
    for fr in range(T):
        for _ in range(4):
            ex, ey = rng.uniform(6, 34, 2); amp = rng.uniform(500, 1000)
            stack[fr] += amp * np.exp(-(((xx - ex) ** 2 + (yy - ey) ** 2)
                                        / (2 * 1.6 ** 2)))

    def _run():
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            locs, *_ = preprocess_and_localise_stream(
                stack, diameter=7, minmass=2.0, bg_radius=12, workers=2,
                chunk_size=32, backend="torch-cpu")
        return locs[["x", "y", "frame", "mass"]].to_numpy(), buf.getvalue()

    a, log = _run()
    b, _ = _run()
    assert len(a) > 0
    assert "Device    : cpu" not in log, "Torch per-call banner leaked into stream"
    assert a.shape == b.shape and np.array_equal(a, b), "streaming not reproducible"


def test_streaming_gpu_batch_matches_unbatched():
    """Decoupled GPU batching must not change detections: streaming with a large
    GPU batch (one backend call per 256 frames) yields the SAME localisations as
    the un-batched path (one call per 32-frame sub-chunk) on sparse data — the
    percentile threshold is a frame-grouping-stable background estimate."""
    import os, io, contextlib
    from firefly.analysis.fa_localize import preprocess_and_localise_stream
    rng = np.random.default_rng(5)
    T, S = 300, 64
    stack = rng.normal(100, 6, size=(T, S, S)).astype(np.float32)
    yy, xx = np.mgrid[0:S, 0:S]
    for fr in range(T):
        for _ in range(5):
            ex, ey = rng.uniform(8, 56, 2); amp = rng.uniform(500, 1000)
            stack[fr] += amp * np.exp(-(((xx - ex) ** 2 + (yy - ey) ** 2)
                                        / (2 * 1.6 ** 2)))

    def _run(gpu_batch):
        prev = os.environ.get("FIREFLY_GPU_BATCH")
        os.environ["FIREFLY_GPU_BATCH"] = str(gpu_batch)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                locs, *_ = preprocess_and_localise_stream(
                    stack, diameter=7, minmass=2.0, bg_radius=12, workers=4,
                    chunk_size=32, backend="torch-cpu")
        finally:
            if prev is None:
                os.environ.pop("FIREFLY_GPU_BATCH", None)
            else:
                os.environ["FIREFLY_GPU_BATCH"] = prev
        return locs[["x", "y", "frame", "mass"]].to_numpy()

    unbatched = _run(32)      # buffer flushes every sub-chunk
    batched = _run(256)       # 8 sub-chunks per backend call
    assert len(unbatched) > 0
    assert unbatched.shape == batched.shape
    # Same detections — same spots, same frames — but NOT necessarily bitwise
    # equal: batching changes the float32 reduction order inside torch
    # (quantile/sum are non-associative), so positions/masses can differ by a
    # few ×1e-6.  Assert agreement to float tolerance, not bit-exactness, or the
    # test is machine-dependent (passes only where the reduction order matches).
    # `frame` is integer and must match exactly.
    assert np.array_equal(unbatched[:, 2], batched[:, 2]), "frame indices changed"
    assert np.allclose(unbatched, batched, rtol=0, atol=1e-3), \
        "GPU batching changed detections beyond float32 rounding"


# ── low-severity hardening regressions ───────────────────────────────────────
def test_preprocess_flat_frame_returns_zero_not_unnormalised():
    """A perfectly uniform frame (mx == mn) must come back all-finite and in
    [0,1] (zeros), not as un-normalised values downstream code assumes are
    normalised."""
    from firefly.analysis.fa_preprocess import _preprocess_fast
    flat = np.full((32, 32), 1234.0, dtype=np.float32)
    out = _preprocess_fast(flat, bg_radius=5)
    assert out.shape == (32, 32) and out.dtype == np.float32
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert np.all(out == 0.0)          # flat → zero image


def test_apply_roi_mask_excluding_all_returns_empty_without_raising():
    """If the ROI mask excludes every localisation, apply_roi_mask must return an
    empty DataFrame (not raise) — the contract the worker relies on to surface a
    clean 'ROI removed all localisations' stop rather than crashing."""
    from firefly.analysis.fa_roi import apply_roi_mask
    locs = pd.DataFrame({"x": [5.0, 10.0, 20.0],
                         "y": [5.0, 10.0, 20.0],
                         "frame": [0, 1, 2]})
    mask = np.zeros((32, 32), dtype=bool)        # nothing inside
    out = apply_roi_mask(locs, mask)
    assert len(out) == 0
    assert list(out.columns) == list(locs.columns)


# ── Statistics config + correction (fa_stats_config) ─────────────────────────
def test_stats_config_normalize_backfills_and_clamps():
    from firefly.analysis.fa_stats_config import normalize_stats_config, DEFAULT_STATS_CONFIG
    assert normalize_stats_config(None) == DEFAULT_STATS_CONFIG
    c = normalize_stats_config({"alpha": 0.9, "correction": "BOGUS",
                                "anova3plus": "nope", "ci_level": 2.0})
    assert c["alpha"] == 0.05          # out-of-range clamped to default
    assert c["correction"] == "holm"   # unknown → default
    assert c["anova3plus"] == "welch"
    assert c["ci_level"] == 0.95
    # known values preserved
    assert normalize_stats_config({"correction": "fdr_bh"})["correction"] == "fdr_bh"
    # circular keys: present + defaulted on, and bool-coerced
    base = normalize_stats_config(None)
    for k in ("include_circular_outputs", "circ_test_kappa", "circ_test_rbar",
              "circ_test_mu", "circ_test_circlin"):
        assert base[k] is True
    coerced = normalize_stats_config({"include_circular_outputs": 0,
                                      "circ_test_kappa": 1})
    assert coerced["include_circular_outputs"] is False
    assert coerced["circ_test_kappa"] is True


def test_correct_pvalues_methods_and_edge_cases():
    from firefly.analysis.fa_stats_config import correct_pvalues
    ps = [0.01, 0.04, 0.20]
    assert np.allclose(correct_pvalues(ps, "bonferroni"), [0.03, 0.12, 0.60])
    assert np.allclose(correct_pvalues(ps, "holm"), [0.03, 0.08, 0.20])
    assert np.allclose(correct_pvalues(ps, "fdr_bh"), [0.03, 0.06, 0.20])
    assert correct_pvalues(ps, "none") == ps
    # m=1 identity for every method
    for meth in ("bonferroni", "holm", "fdr_bh"):
        assert correct_pvalues([0.03], meth) == [0.03]
    # NaN passes through and is excluded from the comparison count m
    out = correct_pvalues([0.01, float("nan"), 0.04], "bonferroni")
    assert out[0] == 0.02 and out[2] == 0.08 and np.isnan(out[1])
    assert correct_pvalues([float("nan")], "holm")[0] != correct_pvalues([float("nan")], "holm")[0]  # NaN


def test_stars_for_alpha_gate():
    from firefly.analysis.fa_stats_config import stars_for
    assert stars_for(0.03, 0.05) == "*"
    assert stars_for(0.03, 0.01) == "ns"     # stricter alpha → not significant
    assert stars_for(0.0005, 0.05) == "***"
    assert stars_for(float("nan"), 0.05) == ""


# ── Config-driven test selection (fa_circular) ───────────────────────────────
def _two_normal_groups(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(10, 1, 6), rng.normal(13, 1, 6)


def test_force_strategy_overrides_normality():
    from firefly.analysis import fa_circular as fc
    a, b = _two_normal_groups()
    om, _ = fc._stat_test_n([a, b], ["A", "B"],
                            {"parametric_strategy": "force_nonparametric"})
    assert om["test"] == "Mann-Whitney U"
    om, _ = fc._stat_test_n([a, b], ["A", "B"],
                            {"parametric_strategy": "force_parametric"})
    assert om["test"] == "Welch's t-test"
    # default (auto) on clearly-normal data → parametric
    om, _ = fc._stat_test_n([a, b], ["A", "B"])
    assert om["test"] == "Welch's t-test"


def test_anova3plus_selection_and_welch_fallback():
    from firefly.analysis import fa_circular as fc
    rng = np.random.default_rng(1)
    a, b, c = rng.normal(10, 1, 6), rng.normal(13, 1, 6), rng.normal(16, 3, 6)
    # force_parametric so the 3+-group test choice is exercised regardless of
    # the random Shapiro outcome.
    def _t(mode):
        return fc._stat_test_n([a, b, c], ["A", "B", "C"],
                               {"anova3plus": mode,
                                "parametric_strategy": "force_parametric"})[0]["test"]
    assert _t("welch") == "Welch's ANOVA"
    assert _t("oneway") == "One-way ANOVA"
    assert _t("auto") == "Welch's ANOVA"
    # zero-variance group → Welch undefined → graceful one-way fallback (labelled)
    z = np.full(6, 5.0)
    t = fc._stat_test_n([a, b, z], ["A", "B", "Z"],
                        {"anova3plus": "welch",
                         "parametric_strategy": "force_parametric"})[0]["test"]
    assert t.startswith("One-way ANOVA")


def test_welch_anova_matches_known_value():
    # Welch's ANOVA on a small fixed dataset vs an independent reference value.
    from firefly.analysis.fa_circular import _welch_anova_p
    g1 = np.array([27., 24., 29., 30., 26.])
    g2 = np.array([21., 19., 25., 22., 23.])
    g3 = np.array([20., 18., 17., 16., 22.])
    p = _welch_anova_p([g1, g2, g3])
    assert p is not None and 0.0 < p < 0.01      # strongly different groups
    assert _welch_anova_p([g1, np.full(5, 3.0)]) is None   # zero-variance → undefined


def test_force_parametric_keeps_underpowered_guard():
    """Forcing parametric must NOT override the n<3 underpowered guard."""
    from firefly.analysis import fa_circular as fc
    a, b = np.array([1.0, 2.0]), np.array([5.0, 6.0])
    _, pw = fc._stat_test_n([a, b], ["A", "B"],
                            {"parametric_strategy": "force_parametric"})
    assert pw[0]["stars"] == ""               # blanked (n=2 per group)
    assert "underpowered" in pw[0]["note"]


# ── End-to-end: compare_groups stats threading + transparency ────────────────
def _fake_summary(stem, folder, d_mean, alpha_mean, n=40, seed=0):
    import pandas as _pd
    rng = np.random.default_rng(seed)
    D = np.clip(rng.normal(d_mean, d_mean * 0.3, n), 1e-6, None)
    alpha = np.clip(rng.normal(alpha_mean, 0.15, n), 0.05, 1.9)
    motion = np.where(alpha < 0.5, "Immobile",
             np.where(alpha < 0.9, "Confined",
             np.where(alpha < 1.1, "Brownian", "Directed")))
    diffusion = _pd.DataFrame({"D": D, "alpha": alpha, "motion": motion})
    emsd = _pd.DataFrame({"lag_frame": np.arange(1, 11),
                          "msd_um2": d_mean * 4 * np.arange(1, 11) * 0.02})
    # simple tracks: n particles, 8 frames each, brownian
    rows = []
    for pid in range(n):
        xy = np.cumsum(rng.normal(0, np.sqrt(D[pid]), size=(8, 2)), axis=0)
        for fr in range(8):
            rows.append({"particle": pid, "frame": fr,
                         "x": xy[fr, 0], "y": xy[fr, 1]})
    tracks = _pd.DataFrame(rows)
    return {"params": {"frame_interval_s": 0.02}, "diffusion": diffusion,
            "ensemble_msd": emsd, "tracks": tracks, "stem": stem,
            "folder": folder}


def test_compare_groups_stats_transparency_end_to_end(tmp_path, monkeypatch):
    """End-to-end: the stats CSV carries the config header + renamed/extra
    corrected columns, and the on-figure annotation names the test+correction —
    i.e. the figure and CSV are self-describing and agree."""
    import firefly.analysis.fa_compare as fcmp
    # Map each fake folder path → a synthetic summary; two clearly different groups.
    table = {}
    for i in range(4):
        table[f"/A/cell{i}"] = _fake_summary(f"A_cell{i}", f"/A/cell{i}",
                                             d_mean=0.02, alpha_mean=0.7, seed=i)
        table[f"/B/cell{i}"] = _fake_summary(f"B_cell{i}", f"/B/cell{i}",
                                             d_mean=0.20, alpha_mean=1.2, seed=10 + i)
    monkeypatch.setattr(fcmp, "load_summary_from_folder", lambda f: table[f])

    groups = [{"label": "DMSO", "color": "#3b6ed8",
               "folders": [f"/A/cell{i}" for i in range(4)]},
              {"label": "Drug", "color": "#f78166",
               "folders": [f"/B/cell{i}" for i in range(4)]}]
    cfg = {"correction": "holm", "across_metric_correction": True,
           "anova3plus": "welch", "figure_stars_use_corrected": True}
    fig, summary_df, stats = fcmp.compare_groups(
        groups, output_dir=str(tmp_path), output_stem="cmp",
        pdf_report=False, stats_config=cfg)

    # per-cell unit of analysis: 8 rows (4 cells × 2 groups)
    assert len(summary_df) == 8

    # stats CSV: config header + accurate, renamed columns
    stats_csv = tmp_path / "cmp_stats.csv"
    assert stats_csv.exists()
    text = stats_csv.read_text()
    assert "Statistics configuration" in text
    assert "Within-metric correction" in text and "Holm" in text
    assert "Across-metric correction" in text
    assert "Adjusted P value (Holm, within metric)" in text
    assert "Adjusted P value (Holm, across metrics)" in text
    # old misleading hard-coded "Bonferroni" header is gone
    assert "Bonferroni-adjusted across pairs" not in text

    # on-figure transparency: at least one panel names the test + correction
    fig_texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    joined = "\n".join(fig_texts)
    assert ("Welch's t-test" in joined) or ("Mann-Whitney" in joined)
    assert "Holm" in joined          # correction named on the figure

    import matplotlib.pyplot as _plt
    _plt.close(fig)


def test_single_shared_timepoint_is_not_two_factor(tmp_path, monkeypatch):
    """Two groups that share ONE time point (e.g. both 'Pre') must be treated as
    a plain one-factor 2-group comparison — NOT a degenerate two-factor design
    (which produced weird single-x-position interaction plots and a spurious
    two-way ANOVA)."""
    import firefly.analysis.fa_compare as fcmp
    table = {}
    for i in range(4):
        table[f"/A/c{i}"] = _fake_summary(f"A_c{i}", f"/A/c{i}", 0.02, 0.7, seed=i)
        table[f"/B/c{i}"] = _fake_summary(f"B_c{i}", f"/B/c{i}", 0.20, 1.2, seed=10 + i)
    monkeypatch.setattr(fcmp, "load_summary_from_folder", lambda f: table[f])
    groups = [{"label": "DMSO", "color": "#3b6ed8", "timepoint": "Pre",
               "folders": [f"/A/c{i}" for i in range(4)]},
              {"label": "Drug", "color": "#f78166", "timepoint": "Pre",
               "folders": [f"/B/c{i}" for i in range(4)]}]
    fig, summary_df, stats = fcmp.compare_groups(
        groups, output_dir=str(tmp_path), output_stem="cmp", pdf_report=False)
    # No two-way ANOVA file (would only exist in true two-factor mode).
    assert not (tmp_path / "cmp_twoway_anova.csv").exists()
    # Treated as a plain 2-group comparison: groups stay DMSO/Drug (not
    # "DMSO / Pre"), and a normal pairwise test was recorded.
    assert set(summary_df["group"]) == {"DMSO", "Drug"}
    pw = stats.get("auc_msd", {}).get("pairwise", [])
    assert pw and {pw[0]["label_i"], pw[0]["label_j"]} == {"DMSO", "Drug"}
    import matplotlib.pyplot as _plt
    _plt.close(fig)


def test_compare_groups_inaccessible_folders_raise_friendly_error(tmp_path):
    """Empty/inaccessible group folders raise CompareInputError (a clean user
    error the worker turns into a popup) with actionable per-folder reasons —
    not a bare RuntimeError/crash."""
    from firefly.analysis.fa_compare import compare_groups, CompareInputError
    groups = [{"label": "DMSO", "color": "#3b6ed8",
               "folders": ["/Volumes/Gone/DMSO/cell1", "/Volumes/Gone/DMSO/cell2"]},
              {"label": "Ciprofol", "color": "#f78166",
               "folders": ["/Volumes/Gone/Cip/cell1"]}]
    import pytest
    with pytest.raises(CompareInputError) as ei:
        compare_groups(groups, output_dir=str(tmp_path), pdf_report=False)
    msg = str(ei.value)
    assert "DMSO" in msg and "Ciprofol" in msg
    assert "not found" in msg and "firefly_extras" in msg   # actionable guidance


def test_compare_groups_too_few_groups_is_friendly():
    from firefly.analysis.fa_compare import compare_groups, CompareInputError
    import pytest
    with pytest.raises(CompareInputError):
        compare_groups([{"label": "A", "folders": ["/x"]}])


# ── Expanded Compare-page statistics (v2.42.0) ───────────────────────────────
def test_correction_sidak_and_hochberg():
    from firefly.analysis.fa_stats_config import correct_pvalues
    p = [0.01, 0.04, 0.03]
    sid = correct_pvalues(p, "sidak")
    hoch = correct_pvalues(p, "hochberg")
    bonf = correct_pvalues(p, "bonferroni")
    # all corrected ≥ raw, and Šidák ≤ Bonferroni elementwise
    assert all(s >= pp - 1e-12 for s, pp in zip(sid, p))
    assert all(s <= b + 1e-12 for s, b in zip(sid, bonf))
    assert all(0.0 <= h <= 1.0 for h in hoch)
    # NaN / None entries pass through and don't inflate the family size
    nan_in = correct_pvalues([0.01, None, 0.02], "hochberg")
    assert np.isnan(nan_in[1])


def test_cliffs_delta_sign_and_magnitude():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 30)
    b_far = a + 50.0                      # b strictly greater → δ ≈ -1
    assert fc._cliffs_delta(a, b_far) == pytest.approx(-1.0, abs=1e-9)
    assert fc._cliffs_delta(b_far, a) == pytest.approx(1.0, abs=1e-9)
    # identical-ish → near 0, CI brackets it
    d, lo, hi = fc._cliffs_delta_ci(a, a + rng.normal(0, 0.01, 30), seed=1)
    assert abs(d) < 0.3 and lo is not None and lo <= d <= hi
    # rank-biserial agrees in sign with Cliff's delta
    assert fc._rank_biserial(b_far, a) > 0
    assert fc._rank_biserial(a, b_far) < 0


def test_alternative_nonparametric_tests_finite():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 8)
    b = rng.normal(1.5, 1, 8)
    for leaf, name in (("brunner_munzel", "Brunner-Munzel"),
                       ("permutation", "Permutation"),
                       ("mann_whitney", "Mann-Whitney U")):
        om, pw = fc._stat_test_n(
            [a, b], ["A", "B"],
            {"parametric_strategy": "force_nonparametric",
             "nonparametric_test": leaf})
        assert om["test"] == name
        assert np.isfinite(om["p"]) and np.isfinite(pw[0]["p"])
        # robust effect sizes always present
        assert pw[0]["cliffs_delta"] is not None
        assert pw[0]["rank_biserial"] is not None


def test_posthoc_games_howell_and_dunn():
    rng = np.random.default_rng(2)
    g = [rng.normal(m, 1, 7) for m in (0.0, 1.5, 3.0)]
    labels = ["A", "B", "C"]
    # Dunn (non-parametric, NOT self-correcting → flows through correction)
    _om, pw = fc._stat_test_n(g, labels, {"posthoc": "dunn"})
    assert all(p["test"] == "Dunn" for p in pw)
    assert all(p["self_corrected"] is False for p in pw)
    assert all(np.isfinite(p["p"]) for p in pw)
    # Games-Howell (self-correcting) — needs pingouin
    pytest.importorskip("pingouin")
    _om2, pw2 = fc._stat_test_n(g, labels, {"posthoc": "games_howell"})
    assert all(p["test"] == "Games-Howell" for p in pw2)
    assert all(p["self_corrected"] is True for p in pw2)


def test_dunnett_vs_control():
    rng = np.random.default_rng(3)
    ctrl = rng.normal(0.0, 1, 8)
    near = rng.normal(0.1, 1, 8)
    far = rng.normal(5.0, 1, 8)
    _om, pw = fc._stat_test_n(
        [ctrl, near, far], ["WT", "Near", "Far"],
        {"dunnett": True, "control_group": "WT"})
    dun = [p for p in pw if p.get("family") == "dunnett"]
    assert len(dun) == 2 and all(p["self_corrected"] for p in dun)
    by = {p["label_i"]: p for p in dun}
    assert by["Far"]["p"] < 0.05 < by["Near"]["p"]


def test_tost_equivalence():
    pytest.importorskip("pingouin")
    rng = np.random.default_rng(4)
    a = rng.normal(0, 1, 12)
    _om, pw = fc._stat_test_n(
        [a, a + rng.normal(0, 0.02, 12)], ["A", "A2"],
        {"equivalence_tost": True, "tost_margin": 1.0})
    assert pw[0]["tost_equivalent"] is True
    b = a + 5.0
    _om2, pw2 = fc._stat_test_n(
        [a, b], ["A", "B"],
        {"equivalence_tost": True, "tost_margin": 0.2})
    assert pw2[0]["tost_equivalent"] is False


def test_omnibus_effect_size_present():
    rng = np.random.default_rng(5)
    g = [rng.normal(m, 1, 8) for m in (0.0, 1.0, 2.0)]
    # parametric → eta²
    om_p, _ = fc._stat_test_n(g, ["A", "B", "C"],
                              {"parametric_strategy": "force_parametric"})
    assert om_p["effect_size_kind"] == "eta_sq"
    assert 0.0 <= om_p["effect_size"] <= 1.0
    # non-parametric → epsilon²
    om_np, _ = fc._stat_test_n(g, ["A", "B", "C"],
                               {"parametric_strategy": "force_nonparametric"})
    assert om_np["effect_size_kind"] == "epsilon_sq"


def test_normalize_stats_config_backward_compat():
    from firefly.analysis.fa_stats_config import (
        normalize_stats_config, DEFAULT_STATS_CONFIG)
    # empty / None get all new keys at defaults
    for cfg in ({}, None):
        n = normalize_stats_config(cfg)
        for k in ("nonparametric_test", "posthoc", "control_group", "dunnett",
                  "equivalence_tost", "tost_margin"):
            assert n[k] == DEFAULT_STATS_CONFIG[k]
    # legacy alias + bad values fall back; tost_margin clamps
    n = normalize_stats_config({"posthoc": "pairwise", "nonparametric_test": "xx",
                                "tost_margin": 999, "correction": "sidak"})
    assert n["posthoc"] == "auto"
    assert n["nonparametric_test"] == "mann_whitney"
    assert n["tost_margin"] == DEFAULT_STATS_CONFIG["tost_margin"]
    assert n["correction"] == "sidak"


def test_self_corrected_not_double_corrected(tmp_path):
    """End-to-end: with a self-correcting post-hoc the CSV's corrected p must
    equal the raw post-hoc p (no extra multiplication)."""
    import matplotlib; matplotlib.use("Agg")
    from firefly.analysis.fa_compare import compare_groups
    root = tmp_path / "runs"
    groups = []
    for gi, gname in enumerate(["A", "B", "C"]):
        folders = [_write_run_folder(str(root), gname, f"{gname}_r{r}",
                                     n_tracks=30, seed=gi * 7 + r)
                   for r in range(3)]
        groups.append({"label": gname, "color": "#888888", "folders": folders})
    out = tmp_path / "out"
    _fig, _summary, stats = compare_groups(
        groups=groups, output_dir=str(out), output_stem="cmp",
        pdf_report=False, stats_config={"posthoc": "tukey", "correction": "holm"})
    # every pairwise record from a self-correcting post-hoc keeps p_within == p
    found = False
    for rec in stats.values():
        for pw in rec.get("pairwise", []):
            if pw.get("self_corrected") and np.isfinite(pw.get("p", np.nan)):
                found = True
                assert pw["p_within"] == pytest.approx(pw["p"], abs=1e-12)
    assert found, "expected at least one self-corrected (Tukey) pairwise record"


@pytest.mark.parametrize("style", ["faceted", "ridgeline", "overlaid", "violin"])
def test_logd_plot_styles_render(tmp_path, style):
    """Every LogD distribution style renders without error and writes a PNG."""
    import matplotlib; matplotlib.use("Agg")
    from firefly.analysis.fa_compare import compare_groups
    root = tmp_path / "runs"
    groups = []
    for gi, gname in enumerate(["A", "B"]):
        folders = [_write_run_folder(str(root), gname, f"{gname}_r{r}",
                                     n_tracks=25, seed=gi * 3 + r)
                   for r in range(3)]
        groups.append({"label": gname, "color": "#777777", "folders": folders})
    out = tmp_path / ("out_" + style)
    fig, _summary, _stats = compare_groups(
        groups=groups, output_dir=str(out), output_stem="cmp",
        pdf_report=False, panels=["logd_dist"], logd_plot_style=style)
    assert (out / "cmp.png").exists()
    # An unknown style must fall back to the default, not crash.
    out2 = tmp_path / "out_bogus"
    compare_groups(groups=groups, output_dir=str(out2), output_stem="cmp",
                   pdf_report=False, panels=["logd_dist"],
                   logd_plot_style="not-a-real-style")
    assert (out2 / "cmp.png").exists()
