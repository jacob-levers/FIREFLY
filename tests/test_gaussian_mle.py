"""Numerical + wiring tests for the Gaussian-MLE and radial-symmetry refinement
backends.

Both share TorchBackend's detection (bandpass → percentile → max-pool) and swap
only the sub-pixel refiner via the ``_refine_peaks`` hook, so these tests check:
  * registration / enum / resolvability (the wiring),
  * detection is UNCHANGED vs torch (same spot set on a clean frame),
  * the refiners are sub-pixel-accurate and self-consistent (mass/size/ecc, the
    minmass filter, batch stability, the CPU-serial guard),
  * on a noisy ensemble neither refiner is materially WORSE than centroid, and
    the MLE is essentially exact on noiseless spots.

No GUI / image fixtures required.  Mirrors the à trous tests in test_analysis.py.
"""
import numpy as np
import pytest

from firefly import sptpalm_analysis as s


# ── helpers (mirror test_analysis.py) ────────────────────────────────────────
def _spot_frame(truth, H=96, W=96, amp=400.0, sigma=1.3, bg=10.0):
    yy, xx = np.mgrid[0:H, 0:W]
    frame = np.full((H, W), bg, dtype=np.float32)
    for cx, cy in truth:
        frame = frame + amp * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    return frame.astype(np.float32)


def _match_to_truth(locs, truth, tol_px):
    errs = []
    for cx, cy in truth:
        d = np.hypot(locs["x"].to_numpy() - cx, locs["y"].to_numpy() - cy)
        j = int(np.argmin(d))
        assert d[j] <= tol_px, f"spot ({cx},{cy}) unmatched; nearest {d[j]:.2f}px"
        errs.append(float(d[j]))
    return errs


def _have(name):
    return name in s.list_available_backends()


ENGINES = ["gaussian-mle", "radial-symmetry"]


# ── registration / enum ──────────────────────────────────────────────────────
@pytest.mark.parametrize("name,cls_name", [
    ("gaussian-mle", "GaussianMleBackend"),
    ("radial-symmetry", "RadialSymmetryBackend"),
])
def test_engine_registered_and_resolvable(name, cls_name):
    """Each refiner backend is registered, listed, and resolves to a TorchBackend
    subclass (so the GPU-slot gating in backend_uses_gpu still applies)."""
    assert name in s.list_available_backends()
    from firefly.analysis import fa_localize as L
    from firefly.analysis import fa_localize_backends as B
    impl = L._resolve_backend(name)
    assert type(impl).__name__ == cls_name
    assert isinstance(impl, B.TorchBackend)
    assert type(impl) in L._BACKEND_REGISTRY


def test_engine_backend_enums():
    from firefly.analysis.fa_enums import Backend
    assert Backend.parse("gaussian-mle") is Backend.GAUSSIAN_MLE
    assert Backend.parse("radial-symmetry") is Backend.RADIAL_SYMMETRY
    assert Backend.parse("gaussian-mle:0") is Backend.GAUSSIAN_MLE   # suffix tolerant
    # Auto-device (like atrous/torch), NOT an explicit GPU pin.
    assert not Backend.GAUSSIAN_MLE.is_explicit_gpu
    assert not Backend.RADIAL_SYMMETRY.is_explicit_gpu


# ── accuracy on clean data ───────────────────────────────────────────────────
@pytest.mark.parametrize("name", ENGINES)
def test_engine_finds_known_spots(name):
    """Detects known spot positions to sub-pixel accuracy on a clean frame."""
    pytest.importorskip("torch")
    if not _have(name):
        pytest.skip(f"{name} unavailable")
    truth = [(30.2, 20.5), (45.7, 60.1), (70.0, 75.0)]
    stack = np.stack([_spot_frame(truth), _spot_frame(truth)])
    df = s.localise_particles(stack, diameter=7, minmass=20,
                              percentile=64, backend=name)
    f0 = df[df.frame == 0]
    assert len(f0) == len(truth), f"expected {len(truth)} spots, got {len(f0)}"
    assert max(_match_to_truth(f0, truth, tol_px=0.5)) < 0.5


@pytest.mark.parametrize("name", ENGINES)
def test_engine_detection_matches_torch(name):
    """Detection is UNCHANGED vs torch — both refiners reuse TorchBackend's
    detection, so the spot SET (count + positions ≤0.5 px) is the same; only the
    sub-pixel polish differs."""
    pytest.importorskip("torch")
    if not (_have(name) and _have("torch")):
        pytest.skip("torch backend unavailable")
    truth = [(30.2, 20.5), (45.7, 60.1), (70.0, 75.0), (50.0, 40.0)]
    stack = np.stack([_spot_frame(truth), _spot_frame(truth)])
    tch = s.localise_particles(stack, diameter=7, minmass=20,
                               percentile=64, backend="torch")
    eng = s.localise_particles(stack, diameter=7, minmass=20,
                               percentile=64, backend=name)
    t0, e0 = tch[tch.frame == 0], eng[eng.frame == 0]
    assert abs(len(t0) - len(e0)) <= 1
    for _, r in t0.iterrows():
        d = np.hypot(e0["x"].to_numpy() - r.x, e0["y"].to_numpy() - r.y)
        assert d.min() <= 0.5


def test_mle_exact_on_noiseless_spots():
    """The Gaussian MLE fits the exact generative model, so on NOISELESS spots it
    recovers the centre essentially exactly (≤0.05 px) — at least as well as
    centroid-of-mass."""
    pytest.importorskip("torch")
    if not (_have("gaussian-mle") and _have("torch")):
        pytest.skip("torch backend unavailable")
    truth = [(30.27, 20.53), (45.71, 60.14), (70.0, 75.0)]
    stack = np.stack([_spot_frame(truth), _spot_frame(truth)])
    mle = s.localise_particles(stack, diameter=7, minmass=20,
                               percentile=64, backend="gaussian-mle")
    cen = s.localise_particles(stack, diameter=7, minmass=20,
                               percentile=64, backend="torch")
    e_mle = _match_to_truth(mle[mle.frame == 0], truth, tol_px=0.5)
    e_cen = _match_to_truth(cen[cen.frame == 0], truth, tol_px=0.5)
    assert max(e_mle) < 0.05
    # MLE is no worse than centroid on the clean model.
    assert np.mean(e_mle) <= np.mean(e_cen) + 1e-3


# ── accuracy on a noisy ensemble ─────────────────────────────────────────────
@pytest.mark.parametrize("name", ENGINES)
def test_engine_not_worse_than_centroid_on_noise(name):
    """On a Poisson-noisy ensemble the refiner must not be materially WORSE than
    centroid-of-mass (the per-spot fallback guarantees this) — we allow a 15 %
    RMSE margin for ensemble noise.  Isolated well-bandpassed spots leave little
    headroom over centroid; the real gains are at low SNR / on real PSFs."""
    pytest.importorskip("torch")
    if not (_have(name) and _have("torch")):
        pytest.skip("torch backend unavailable")
    rng = np.random.default_rng(0)
    H = W = 13
    frames, tx, ty = [], [], []
    for _ in range(120):
        cx = 6 + rng.uniform(-0.5, 0.5)
        cy = 6 + rng.uniform(-0.5, 0.5)
        clean = _spot_frame([(cx, cy)], H=H, W=W, amp=60.0, sigma=1.3, bg=5.0)
        frames.append(rng.poisson(clean).astype(np.float32))
        tx.append(cx); ty.append(cy)
    stack = np.stack(frames)
    tx = np.array(tx); ty = np.array(ty)

    def rmse(backend):
        df = s.localise_particles(stack, diameter=7, minmass=1.0,
                                  percentile=50, backend=backend)
        e = []
        for i in range(len(stack)):
            r = df[df.frame == i]
            if len(r):
                d = np.hypot(r.x.to_numpy() - tx[i], r.y.to_numpy() - ty[i])
                e.append(d.min())
        return float(np.sqrt(np.mean(np.square(e)))), len(e)

    r_cen, n_cen = rmse("torch")
    r_eng, n_eng = rmse(name)
    assert n_eng >= 0.9 * n_cen            # detection essentially unchanged
    assert r_eng <= 1.15 * r_cen, (
        f"{name} RMSE {r_eng:.4f} >> centroid {r_cen:.4f}")


# ── self-consistency: characterise / minmass / batch / CPU-serial ────────────
@pytest.mark.parametrize("name", ENGINES)
def test_engine_characterize_columns(name):
    """characterize=True yields size/ecc columns (the PSF-quality gate the
    auto-threshold harvest feeds), inherited from the centroid seed."""
    pytest.importorskip("torch")
    if not _have(name):
        pytest.skip(f"{name} unavailable")
    from firefly.analysis import fa_localize as L
    impl = L._resolve_backend(name)
    truth = [(30.2, 20.5), (45.7, 60.1)]
    stack = np.stack([_spot_frame(truth)])
    df = impl.localise(stack, diameter=7, minmass=20, percentile=64,
                       quiet=True, characterize=True)
    assert {"size", "ecc"} <= set(df.columns)
    assert np.isfinite(df["size"]).all() and np.isfinite(df["ecc"]).all()


@pytest.mark.parametrize("name", ENGINES)
def test_engine_minmass_filters(name):
    """A high minmass drops spots (mass stays on the trackpy scale)."""
    pytest.importorskip("torch")
    if not _have(name):
        pytest.skip(f"{name} unavailable")
    truth = [(30.2, 20.5), (45.7, 60.1), (70.0, 75.0)]
    stack = np.stack([_spot_frame(truth)])
    lo = s.localise_particles(stack, diameter=7, minmass=5,
                              percentile=64, backend=name)
    hi = s.localise_particles(stack, diameter=7, minmass=10_000,
                              percentile=64, backend=name)
    assert len(lo) == len(truth)
    assert len(hi) < len(lo)


def test_engine_auto_threshold_runs():
    """estimate_minmass works with the new backends (the harvest uses this
    backend's own localise(characterize=True))."""
    pytest.importorskip("torch")
    if not _have("gaussian-mle"):
        pytest.skip("gaussian-mle unavailable")
    from firefly.analysis.fa_localize import estimate_minmass
    truth = [(30.2, 20.5), (45.7, 60.1), (70.0, 75.0)]
    stack = np.stack([_spot_frame(truth) for _ in range(6)])
    mm = estimate_minmass(stack, diameter=7, backend="gaussian-mle")
    mm = mm[0] if isinstance(mm, tuple) else float(mm)
    assert np.isfinite(mm) and mm >= 0.0


@pytest.mark.parametrize("name", ENGINES)
def test_engine_batch_stable(name):
    """Refinement is batch-size-stable: same spot count across chunk sizes."""
    pytest.importorskip("torch")
    if not _have(name):
        pytest.skip(f"{name} unavailable")
    truth = [(30.2, 20.5), (45.7, 60.1), (70.0, 75.0)]
    stack = np.stack([_spot_frame(truth) for _ in range(6)])
    n1 = len(s.localise_particles(stack, diameter=7, minmass=20,
                                  percentile=64, backend=name, chunk_size=6))
    n2 = len(s.localise_particles(stack, diameter=7, minmass=20,
                                  percentile=64, backend=name, chunk_size=2))
    assert n1 == n2, f"batch-unstable: {n1} vs {n2}"


@pytest.mark.parametrize("name", ENGINES)
def test_engine_forces_cpu_serial(name):
    """The CPU multi-process worker hard-codes TorchBackend (centroid), so the
    refiner backends must force the serial CPU path (return None) so their
    override actually runs; a small CPU run still detects correctly."""
    pytest.importorskip("torch")
    if not _have(name):
        pytest.skip(f"{name} unavailable")
    from firefly.analysis import fa_localize as L
    impl = L._resolve_backend(name)
    assert impl._localise_cpu_parallel() is None
    truth = [(30.2, 20.5), (45.7, 60.1)]
    stack = np.stack([_spot_frame(truth) for _ in range(3)])
    inst = L._resolve_backend(name); inst._forced_device = "cpu"
    df = inst.localise(stack, diameter=7, minmass=20, percentile=64,
                       quiet=True, chunk_size=2)
    assert len(df[df.frame == 0]) == len(truth)


def test_mle_emits_per_spot_loc_sigma():
    """GaussianMleBackend attaches a per-spot Fisher-info CRLB precision
    (loc_sigma_x_px / loc_sigma_y_px), finite & positive on fitted spots, in a
    physically sane band once converted to nm."""
    pytest.importorskip("torch")
    if not _have("gaussian-mle"):
        pytest.skip("gaussian-mle unavailable")
    import contextlib, io
    from firefly.analysis import fa_localize as L
    rng = np.random.default_rng(0)
    truth = [(30.4, 40.2), (60.1, 25.7), (45.0, 70.3)]
    # realistic SNR + Poisson noise so the CRLB lands in a believable range
    frames = []
    for _ in range(8):
        fr = _spot_frame(truth, amp=120.0, sigma=1.3, bg=8.0)
        frames.append(rng.poisson(np.clip(fr, 0, None)).astype(np.float32))
    stack = np.stack(frames)
    impl = L._resolve_backend("gaussian-mle"); impl._forced_device = "cpu"
    with contextlib.redirect_stdout(io.StringIO()):
        df = impl.localise(stack, diameter=7, minmass=20, percentile=64, quiet=True)
    assert {"loc_sigma_x_px", "loc_sigma_y_px"} <= set(df.columns)
    sx = df["loc_sigma_x_px"].to_numpy(float)
    assert np.isfinite(sx).mean() > 0.7            # most fits yield a CRLB
    fin = sx[np.isfinite(sx)]
    assert (fin > 0).all()
    nm = float(np.median(fin)) * 100.0             # px → nm at px≈100 nm
    assert 1.0 <= nm <= 200.0, f"median σ_x = {nm:.1f} nm out of band"


def test_non_mle_engine_has_no_loc_sigma_px():
    """The radial-symmetry refiner has no Fisher Hessian, so it must NOT emit the
    loc_sigma_*_px columns (the worker fills NaN / camera-CRLB for it instead)."""
    pytest.importorskip("torch")
    if not _have("radial-symmetry"):
        pytest.skip("radial-symmetry unavailable")
    import contextlib, io
    from firefly.analysis import fa_localize as L
    stack = np.stack([_spot_frame([(30.2, 20.5), (45.7, 60.1)]) for _ in range(3)])
    inst = L._resolve_backend("radial-symmetry"); inst._forced_device = "cpu"
    with contextlib.redirect_stdout(io.StringIO()):
        df = inst.localise(stack, diameter=7, minmass=20, percentile=64, quiet=True)
    assert "loc_sigma_x_px" not in df.columns


# ── MPS-safety: refiner numerics run on CPU on Apple GPUs ─────────────────────
# MPS intermittently returns silently-wrong results for the linalg/conv ops the
# Gaussian-MLE and radial-symmetry refiners use, mis-localising spots on Apple
# GPUs while passing on CPU.  TorchBackend._refine_off_mps forces just the
# (cheap) refinement onto CPU on MPS; detection stays GPU-accelerated.
def test_refine_off_mps_passthrough_for_non_mps():
    """The wrapper is a no-op for CPU/CUDA: it invokes the impl on the given
    device and returns its 6-tuple unchanged (only MPS pays the CPU round-trip)."""
    torch = pytest.importorskip("torch")
    from firefly.analysis import fa_localize as L
    inst = L._resolve_backend("gaussian-mle")
    seen = {}

    def fake_impl(signal, t_ix, y_ix, x_ix, diameter, *, device, dtype, raw,
                  characterize):
        seen["device"] = device
        z = torch.zeros(2)
        return z, z, z, z, z, {"size": z}

    cpu = torch.device("cpu")
    out = inst._refine_off_mps(
        fake_impl, torch.zeros(1, 1, 4, 4),
        torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long), 7,
        device=cpu, dtype=torch.float32, raw=None, characterize=True)
    assert seen["device"] == cpu           # ran on the given device, no redirect
    assert len(out) == 6                    # 6-tuple contract preserved


@pytest.mark.parametrize("name", ENGINES)
def test_refiner_runs_on_cpu_under_mps(name, monkeypatch):
    """On MPS the refiner impl must be invoked with a CPU device (the fix); the
    results are then moved back to the GPU device for the caller."""
    torch = pytest.importorskip("torch")
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")
    if not _have(name):
        pytest.skip(f"{name} unavailable")
    import contextlib, io
    from firefly.analysis import fa_localize as L
    inst = L._resolve_backend(name)
    inst._forced_device = "mps"; inst._validated_device = "mps"
    seen = []
    orig = inst._refine_peaks_impl

    def spy(*a, **k):
        seen.append(getattr(k.get("device"), "type", str(k.get("device"))))
        return orig(*a, **k)

    monkeypatch.setattr(inst, "_refine_peaks_impl", spy)
    stack = np.stack([_spot_frame([(30.2, 20.5), (45.7, 60.1)])])
    with contextlib.redirect_stdout(io.StringIO()):
        inst.localise(stack, diameter=7, minmass=20, percentile=64, quiet=True)
    assert seen and all(d == "cpu" for d in seen), seen


@pytest.mark.parametrize("name", ENGINES)
def test_engine_mps_matches_cpu(name):
    """End-to-end guard: on MPS the engine's localisations match the pure-CPU
    result (because refinement runs on CPU in both)."""
    torch = pytest.importorskip("torch")
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        pytest.skip("MPS not available")
    if not _have(name):
        pytest.skip(f"{name} unavailable")
    import contextlib, io
    from firefly.analysis import fa_localize as L
    truth = [(30.27, 20.53), (45.71, 60.14), (70.0, 75.0)]
    stack = np.stack([_spot_frame(truth), _spot_frame(truth)])

    def run(dev):
        inst = L._resolve_backend(name)
        inst._forced_device = dev; inst._validated_device = dev
        with contextlib.redirect_stdout(io.StringIO()):
            df = inst.localise(stack, diameter=7, minmass=20, percentile=64,
                               quiet=True)
        return df[df.frame == 0].sort_values("x").reset_index(drop=True)

    cpu, mps = run("cpu"), run("mps")
    assert len(cpu) == len(mps) == len(truth)
    assert np.abs(cpu["x"].to_numpy() - mps["x"].to_numpy()).max() < 1e-3
    assert np.abs(cpu["y"].to_numpy() - mps["y"].to_numpy()).max() < 1e-3
