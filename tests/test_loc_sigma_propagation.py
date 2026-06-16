"""Per-localisation precision (loc_sigma_x_nm / loc_sigma_y_nm):
  * the worker helper that attaches it from each backend's native estimate,
  * its survival through EVERY registry linker (kalman/nn rebuild rows but must
    keep all input columns — a regression lock),
  * its per-track aggregate (loc_sigma_meas_nm) in the diffusion summary,
    distinct from the MSD-offset loc_sigma_nm.
"""
import numpy as np
import pandas as pd
import pytest

from firefly import sptpalm_analysis as s


def _synthetic_locs(n_tracks=6, n_frames=12, seed=0):
    """Well-separated, slowly-drifting tracks (unambiguous linking) that carry
    per-spot loc_sigma_x_nm / loc_sigma_y_nm columns."""
    rng = np.random.default_rng(seed)
    rows = []
    starts = [(15 + 18 * (i % 4), 15 + 18 * (i // 4)) for i in range(n_tracks)]
    for x0, y0 in starts:
        for f in range(n_frames):
            x0 += rng.normal(0, 0.25)
            y0 += rng.normal(0, 0.25)
            rows.append((f, x0, y0, 100.0,
                         12.0 + rng.normal(0, 1.0), 14.0 + rng.normal(0, 1.0)))
    return pd.DataFrame(rows, columns=["frame", "x", "y", "mass",
                                       "loc_sigma_x_nm", "loc_sigma_y_nm"])


def test_attach_loc_sigma_sources():
    """_attach_loc_sigma yields nm precision from each source — MLE Fisher px cols
    (× px, gain-corrected), trackpy ep, camera CRLB — and NaN otherwise, with the
    columns ALWAYS present."""
    import firefly.firefly_worker as fw
    from firefly.bench.metrics import crlb_sigma_nm
    px = 0.1

    mle = pd.DataFrame({"x": [1.0], "y": [1.0], "frame": [0], "mass": [100.0],
                        "loc_sigma_x_px": [0.02], "loc_sigma_y_px": [0.03]})
    o = fw._attach_loc_sigma(mle, p={"diameter": 7}, px=px, backend="gaussian-mle")
    assert "loc_sigma_x_px" not in o.columns                  # consumed
    assert np.isclose(o["loc_sigma_x_nm"].iloc[0], 2.0)        # 0.02 px × 100 nm
    og = fw._attach_loc_sigma(mle, p={"diameter": 7, "camera_gain": 2.0,
                                      "camera_qe": 0.9}, px=px, backend="gaussian-mle")
    assert np.isclose(og["loc_sigma_x_nm"].iloc[0], 2.0 * np.sqrt(0.9 / 2.0))

    ep = pd.DataFrame({"x": [1.0], "y": [1.0], "frame": [0], "mass": [100.0],
                       "ep": [0.15]})
    oe = fw._attach_loc_sigma(ep, p={"diameter": 7}, px=px, backend="trackpy")
    assert np.isclose(oe["loc_sigma_x_nm"].iloc[0], 15.0)

    crlb = pd.DataFrame({"x": [1.0], "y": [1.0], "frame": [0], "mass": [1000.0]})
    oc = fw._attach_loc_sigma(crlb, p={"diameter": 7, "camera_gain": 1.0,
                                       "camera_bg_photons": 5.0}, px=px, backend="torch")
    assert np.isclose(oc["loc_sigma_x_nm"].iloc[0],
                      crlb_sigma_nm(1000.0, 5.0, 1.75, px), rtol=1e-6)

    none = fw._attach_loc_sigma(crlb, p={"diameter": 7}, px=px, backend="torch")
    assert {"loc_sigma_x_nm", "loc_sigma_y_nm"} <= set(none.columns)
    assert none["loc_sigma_x_nm"].isna().all()


@pytest.mark.parametrize("linker",
                         ["trackpy", "kalman", "simple_lap", "full_lap", "nn", "sa"])
def test_loc_sigma_survives_each_linker(linker):
    """Every registry linker carries loc_sigma_*_nm through into the trajectory
    table (kalman/nn rebuild rows but preserve all input columns)."""
    pytest.importorskip("trackpy")
    from firefly.analysis.fa_linking_registry import _resolve_linker
    locs = _synthetic_locs()
    tracks = _resolve_linker(linker).link(
        locs, search_range=5, memory=2, min_len=3, max_len=None, params={})
    assert len(tracks) > 0, f"{linker}: nothing linked"
    assert {"loc_sigma_x_nm", "loc_sigma_y_nm"} <= set(tracks.columns), \
        f"{linker} dropped the per-localisation precision columns"
    assert tracks["loc_sigma_x_nm"].notna().all()


def test_diffusion_summary_has_loc_sigma_meas():
    """compute_msd_and_fit adds a per-track loc_sigma_meas_nm aggregate (the
    measured per-spot precision), distinct from the MSD-offset loc_sigma_nm."""
    pytest.importorskip("trackpy")
    from firefly.analysis.fa_linking_registry import _resolve_linker
    locs = _synthetic_locs(n_tracks=8, n_frames=40, seed=1)
    tracks = _resolve_linker("trackpy").link(
        locs, search_range=5, memory=2, min_len=10, max_len=None, params={})
    _i, _e, diff = s.compute_msd_and_fit(tracks, 0.1, 0.02, max_lagtime=10,
                                         n_fit=5, workers=1)
    assert "loc_sigma_meas_nm" in diff.columns
    assert diff["loc_sigma_meas_nm"].notna().any()
    # hypot(12,14)/√2 ≈ 13 nm — a wide sanity band, not a tight check
    assert 5.0 <= float(diff["loc_sigma_meas_nm"].median()) <= 40.0
    assert "loc_sigma_nm" in diff.columns          # the offset estimate still present
