"""palmTRACER-style wavelet detection.

Reverse-engineered from palmTRACER's own output on 11 recordings (12,150 spots on
660 frames): 99.4 % of its spots reproduced at identical positions and identical
"Integrated_Intensity", and 99.7 % of these detections are palmTRACER's.  The
recipe, from Izeddin et al. 2012 as palmTRACER configures it:

  * B3-spline à trous on the RAW frame; detection uses the second wavelet plane
    W2 = A1 − A2 (nearest-pixel edges);
  * an ABSOLUTE threshold on W2 — the number palmTRACER calls "Wavelet threshold";
  * 8-connected regions above it, split by watershed between local maxima;
  * regions of fewer than 5 pixels are dropped;
  * position = W2-weighted centroid; mass = sum of W2 above the threshold.

The threshold being absolute is the whole point and the trap: FIREFLY normally
hands detectors background-subtracted frames rescaled so the brightest pixel is
1, on which "250" means nothing.  So this detector declares that it needs raw
frames, and the pipeline must honour that.
"""
import numpy as np
import pytest
from scipy import ndimage as ndi

from firefly.analysis.fa_localize_backends import PalmTracerWaveletBackend, palmtracer_w2


def _spot(shape, cy, cx, amp, sigma=1.2):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))


def _frame(spots, shape=(48, 48), bg=1000.0, noise=0.0, seed=0):
    img = np.full(shape, bg, float)
    for cy, cx, a in spots:
        img += _spot(shape, cy, cx, a)
    if noise:
        img += np.random.default_rng(seed).normal(0, noise, shape)
    return img.astype(np.float32)


def _detect(frames, T=250.0, **kw):
    stack = np.asarray(frames, np.float32)
    if stack.ndim == 2:
        stack = stack[None]
    return PalmTracerWaveletBackend().localise(stack, wavelet_threshold=T, quiet=True, **kw)


def _region_px(frame, T):
    """Size of the largest above-threshold region — to place T at a boundary."""
    lab, n = ndi.label(palmtracer_w2(frame) > T, structure=np.ones((3, 3)))
    return 0 if not n else int(np.bincount(lab.ravel())[1:].max())


# ── what it reports ─────────────────────────────────────────────────────────
def test_one_spot_gives_one_detection_at_its_centre():
    d = _detect(_frame([(20, 25, 2000)]))
    assert len(d) == 1
    assert d.x.iloc[0] == pytest.approx(25.0, abs=1e-6)
    assert d.y.iloc[0] == pytest.approx(20.0, abs=1e-6)
    off = _detect(_frame([(20.3, 25.6, 2000)]))
    assert len(off) == 1
    assert abs(off.x.iloc[0] - 25.6) < 0.2 and abs(off.y.iloc[0] - 20.3) < 0.2


def test_mass_is_the_wavelet_signal_above_threshold():
    """palmTRACER's Integrated_Intensity, exactly."""
    f = _frame([(20, 25, 2000)])
    w2 = palmtracer_w2(f)
    d = _detect(f, T=250.0)
    assert d.mass.iloc[0] == pytest.approx(float(w2[w2 > 250.0].sum()), rel=1e-6)


def test_a_constant_offset_changes_nothing():
    """W2 removes any constant — the camera offset and a flat background drop out."""
    f = _frame([(14, 15, 2000), (30, 33, 1500)], noise=20)
    a, b = _detect(f), _detect(f + 5000.0)
    assert len(a) == len(b) == 2
    np.testing.assert_allclose(a[["x", "y", "mass"]].to_numpy(), b[["x", "y", "mass"]].to_numpy(), rtol=1e-5)


def test_the_threshold_is_in_absolute_units():
    """A per-frame-normalised detector cannot tell a brighter movie from a dimmer
    one; this one must — that is what palmTRACER's single number means."""
    dim, bright = _frame([(14, 15, 900)]), _frame([(14, 15, 2000)])
    peak_dim, peak_bright = palmtracer_w2(dim).max(), palmtracer_w2(bright).max()
    T = (peak_dim + peak_bright) / 2
    assert len(_detect(dim, T)) == 0
    assert len(_detect(bright, T)) == 1
    assert len(_detect(dim * 3.0, T)) == 1, "scaling the raw counts must change what passes"


def test_regions_smaller_than_five_pixels_are_dropped():
    # centred on a pixel CORNER, so the region above threshold grows through
    # exactly 4 pixels (on-pixel spots jump 1 → 5 → 9 and never test the edge)
    f = _frame([(20.5, 25.5, 2000)])
    w2 = palmtracer_w2(f)
    sizes = {T: _region_px(f, T) for T in np.linspace(10, w2.max() - 1, 600)}
    T4 = max(T for T, n in sizes.items() if n == 4)
    T5 = max(T for T, n in sizes.items() if n >= 5)
    assert len(_detect(f, T4)) == 0, "a 4-pixel region must not be reported"
    assert len(_detect(f, T5)) == 1


def test_touching_spots_are_split():
    f = _frame([(20, 20, 2000), (20, 24, 2000)])
    assert _region_px(f, 250.0) > 0
    lab, n = ndi.label(palmtracer_w2(f) > 250.0, structure=np.ones((3, 3)))
    assert n == 1, "the two spots must form ONE region above threshold for this test to mean anything"
    d = _detect(f)
    assert len(d) == 2
    assert sorted(np.round(d.x.to_numpy())) == [20, 24]


def test_frames_keep_their_index():
    stack = np.stack([_frame([]), _frame([(20, 25, 2000)]), _frame([])])
    d = _detect(stack)
    assert d.frame.tolist() == [1]


def test_an_empty_frame_returns_the_standard_columns():
    d = _detect(_frame([]))
    assert len(d) == 0
    assert {"x", "y", "frame", "mass"} <= set(d.columns)


# ── wiring ──────────────────────────────────────────────────────────────────
def test_the_pipeline_hands_it_raw_frames():
    """The dispatcher normally detects on rescaled frames whose maximum is 1, on
    which W2 can never exceed 250: if raw frames did not reach this detector it
    would find nothing at all."""
    from firefly.analysis.fa_localize import preprocess_and_localise_adaptive
    rng = np.random.default_rng(3)
    stack = np.stack([_frame([(14 + i % 3, 15, 2000), (30, 33 - i % 2, 1500)], noise=15, seed=i)
                      for i in range(12)])
    direct = _detect(stack)
    locs, mean_proj, max_proj, blink_proj, _ = preprocess_and_localise_adaptive(
        stack, diameter=7, minmass=0.0, bg_radius=10, workers=1, chunk_size=5,
        backend="palmtracer", wavelet_threshold=250.0)
    assert len(direct) == 24
    assert len(locs) == len(direct)
    a = locs.sort_values(["frame", "x"])[["frame", "x", "y", "mass"]].to_numpy()
    b = direct.sort_values(["frame", "x"])[["frame", "x", "y", "mass"]].to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-5)
    assert mean_proj.shape == max_proj.shape == stack.shape[1:]


def test_it_is_registered_and_never_auto_selected():
    from firefly.analysis.fa_enums import Backend
    from firefly.analysis.fa_localize import _resolve_backend, backend_uses_gpu
    assert Backend.parse("palmtracer") is Backend.PALMTRACER
    assert isinstance(_resolve_backend("palmtracer"), PalmTracerWaveletBackend)
    assert not backend_uses_gpu("palmtracer")
    assert not isinstance(_resolve_backend("auto"), PalmTracerWaveletBackend)


def test_the_sidebar_choice_and_threshold_reach_the_run():
    from firefly.ui.controllers.params.params_builder import BACKEND_LABEL_TO_VALUE, build_params
    label = next(k for k, v in BACKEND_LABEL_TO_VALUE.items() if v == "palmtracer")

    class S(dict):
        def get_str(self, k, d=""): return dict.get(self, k, d)
        def get_float(self, k, d=0.0): return float(dict.get(self, k, d))
        def get_bool(self, k, d=False): return bool(dict.get(self, k, d))
        def get(self, k, d=None): return dict.get(self, k, d)

    class Imp:
        filePath = "/tmp/a.tif"; outDir = "/tmp/o"; isCsv = False
        overridePx = False; pixelSize = 0.1; overrideFi = False; frameInterval = 0.02

    p = build_params(S({"analysis/backend": label, "analysis/wavelet_threshold": 320.0}), Imp(), fpath="/tmp/a.tif")
    assert p["backend"] == "palmtracer"
    assert p["wavelet_threshold"] == pytest.approx(320.0)


def test_the_new_setting_is_in_the_schema():
    from firefly.ui.controllers.params.sidebar_schema import BY_KEY
    f = BY_KEY["analysis/wavelet_threshold"]
    assert f["default"] == pytest.approx(250.0)


# ── the real pipeline, end to end ───────────────────────────────────────────
@pytest.mark.slow
def test_a_full_run_detects_exactly_what_the_detector_does(tmp_path):
    """Auto-minmass ON is deliberate: it must be skipped for this detector (its
    search runs a different detector and picks a number this one never uses),
    and the run record must say which threshold actually applied."""
    import glob, json
    import tifffile
    from test_worker_pipeline_parity import _synthetic_movie, _params, _run
    mov = _synthetic_movie(tmp_path / "m.tif", n_frames=40)
    stack = tifffile.imread(mov).astype(np.float32)
    T = 0.3 * float(palmtracer_w2(stack[0]).max())
    expected = PalmTracerWaveletBackend().localise(stack, wavelet_threshold=T, quiet=True)
    payload = _run(_params(mov, str(tmp_path / "out"), backend="palmtracer",
                           auto_minmass=True, wavelet_threshold=T, skip_figure=True))
    assert int(payload["n_locs"]) == len(expected) > 0
    pj = json.load(open(glob.glob(f"{payload['out_dir']}/firefly_extras/*_params.json")[0]))
    assert pj["wavelet_threshold"] == pytest.approx(T)
    assert pj["minmass_method"] == "palmtracer wavelet threshold"
    assert pj["auto_minmass"] is False, "auto-minmass was skipped; the record must not claim it ran"
    man = json.load(open(glob.glob(f"{payload['out_dir']}/*_run_manifest.json")[0]))
    assert man.get("resolved_minmass") is None, "no minmass was resolved for this detector"


# ── the ROI viewer's live preview ───────────────────────────────────────────
def test_the_detection_preview_uses_raw_frames_and_the_wavelet_threshold():
    from firefly.analysis.fa_detection_preview import preview_detections
    f = _frame([(14, 15, 2000), (30, 33, 1500), (36, 10, 700)], noise=15)
    T = float(palmtracer_w2(_frame([(20, 20, 1100)])).max())     # passes the two bright spots only
    direct = _detect(f, T)
    rows, summary = preview_detections(f, diameter=7, minmass=0.0, bg_radius=10,
                                       backend="palmtracer", wavelet_threshold=T)
    assert len(direct) == 2
    assert summary["candidates"] == len(direct)
