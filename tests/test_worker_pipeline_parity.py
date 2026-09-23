"""End-to-end parity net for the analysis core (`_run_one_analysis`).

This is the safety net for restructuring that function (localise-once + per-ROI
loop): it runs the REAL pipeline on a small synthetic movie of diffusing spots
and freezes the numbers it produces.  Any refactor that changes single-ROI
behaviour turns this red.

Deliberately tolerant on absolute values (localisation/linking depend on the
exact trackpy build); what it pins is that the pipeline (a) runs end to end,
(b) writes its outputs, and (c) is REPRODUCIBLE — two runs of the same input
give identical numbers.  A behaviour-preserving refactor keeps that identity.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import queue
import threading

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("trackpy")


# ── synthetic movie: diffusing 2-D Gaussian spots ────────────────────────────
def _synthetic_movie(path, *, n_frames=60, size=96, n_spots=14, seed=0,
                     sigma=1.4, amp=900.0, bg=100.0, noise=8.0, step=1.1):
    """Write a small TIFF stack of Brownian spots — enough signal for trackpy to
    localise and link, small enough to run in a few seconds."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    pos = rng.uniform(12, size - 12, size=(n_spots, 2))       # (y, x)
    frames = np.empty((n_frames, size, size), dtype=np.uint16)
    for f in range(n_frames):
        img = np.full((size, size), bg, dtype=float)
        for s in range(n_spots):
            y, x = pos[s]
            img += amp * np.exp(-(((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma ** 2)))
        img += rng.normal(0.0, noise, img.shape)
        frames[f] = np.clip(img, 0, 65535).astype(np.uint16)
        pos += rng.normal(0.0, step, pos.shape)               # Brownian step
        pos = np.clip(pos, 6, size - 6)
    tifffile.imwrite(str(path), frames)
    return str(path)


def _params(fpath, out_dir, **over):
    """A minimal-but-complete params dict (the worker .get()s most keys)."""
    p = {
        "file": fpath, "out_dir": out_dir,
        "pixel_size": 0.106, "frame_interval": 0.02, "channel": 0,
        "bg_method": "uniform_filter", "bg_radius": 10,
        "diameter": 7, "auto_minmass": True, "minmass": 1.0,
        "minmass_sensitivity": "balanced",
        "search_range": 5, "auto_search_range": False, "memory": 1,
        "min_track_len": 5, "workers": 1, "chunk_size": 500,
        "max_track_len": None, "max_lagtime": 10, "n_fit": 4,
        "roi_polygon": None,
        "alpha_thresholds": (0.4, 0.9, 1.4), "mobile_d_threshold": 0.05,
        "jdd_components": 2, "linker": "trackpy", "backend": "auto",
        "roi_mode": "none", "drift_correct": False,
        "fig_save_pdf": False, "fig_per_panel": False, "fig_dpi": 80,
        "include_circular_outputs": False,
    }
    p.update(over)
    return p


def _run(params):
    """Drive the real pipeline; return its done-payload."""
    from firefly import firefly_worker
    q = queue.Queue(maxsize=100000)
    cancel = threading.Event()
    return firefly_worker._run_one_analysis(
        params, q, cancel, lambda m: None, lambda pct, m: None)


def _digest(out_dir, payload):
    """Numeric fingerprint of a run — what a refactor must not change."""
    import pandas as pd
    extras = os.path.join(payload["out_dir"], "firefly_extras")
    stem = payload["stem"]
    d = {"n_locs": int(payload.get("n_locs") or 0),
         "n_tracks": int(payload.get("n_tracks") or 0)}
    dif = os.path.join(extras, f"{stem}_diffusion_summary.csv")
    if os.path.isfile(dif):
        df = pd.read_csv(dif)
        for col in ("D", "alpha"):
            if col in df.columns and len(df):
                d[f"median_{col}"] = round(float(df[col].median()), 9)
        d["n_diff_rows"] = int(len(df))
    return d


@pytest.mark.slow
def test_pipeline_runs_and_is_reproducible(tmp_path):
    """The real pipeline completes on a synthetic movie and produces IDENTICAL
    numbers when re-run — the invariant a behaviour-preserving refactor keeps."""
    mov = _synthetic_movie(tmp_path / "cells.tif")

    o1 = str(tmp_path / "out1"); os.makedirs(o1, exist_ok=True)
    pay1 = _run(_params(mov, o1))
    assert pay1 and pay1.get("out_dir") and os.path.isdir(pay1["out_dir"])
    d1 = _digest(o1, pay1)
    assert d1["n_locs"] > 0, "no localisations — synthetic movie too weak"
    assert d1["n_tracks"] > 0, "no tracks — linking produced nothing"
    summary = pay1["summary"]
    assert summary["mean_observed_time_s"] is not None
    assert summary["mean_track_duration_s"] is not None
    assert summary["mean_track_length_s"] == summary["mean_observed_time_s"]

    o2 = str(tmp_path / "out2"); os.makedirs(o2, exist_ok=True)
    pay2 = _run(_params(mov, o2))
    d2 = _digest(o2, pay2)
    assert d1 == d2, f"pipeline is not reproducible:\n  {d1}\n  {d2}"


@pytest.mark.slow
def test_polygon_roi_restricts_the_analysis(tmp_path):
    """A polygon ROI must actually shrink the analysed set — the property the
    per-ROI loop relies on (each cell sees only its own localisations)."""
    mov = _synthetic_movie(tmp_path / "cells.tif")

    o_full = str(tmp_path / "full"); os.makedirs(o_full, exist_ok=True)
    full = _digest(o_full, _run(_params(mov, o_full)))

    # left half only
    half = [[(0.0, 0.0), (0.0, 47.0), (95.0, 47.0), (95.0, 0.0)]]
    o_roi = str(tmp_path / "roi"); os.makedirs(o_roi, exist_ok=True)
    roi = _digest(o_roi, _run(_params(mov, o_roi, roi_mode="polygon",
                                      roi_polygon=half)))
    assert roi["n_locs"] < full["n_locs"], "ROI did not restrict the localisations"
    assert roi["n_locs"] > 0, "ROI removed everything — polygon/frame mismatch"


def test_requested_polygon_cannot_silently_fall_back(tmp_path):
    mov = _synthetic_movie(tmp_path / "cells.tif", n_frames=10)
    with pytest.raises(ValueError, match="requested polygon ROI"):
        _run(_params(mov, str(tmp_path / "missing_roi"),
                     roi_mode="polygon", roi_polygon=None,
                     auto_minmass=False, minmass=.45, backend="trackpy"))


def test_raw_gate_and_new_metric_contract_survive_pipeline_exports(tmp_path):
    import json
    import pandas as pd
    from pathlib import Path
    mov = _synthetic_movie(tmp_path / "cells.tif", n_frames=25)
    result = _run(_params(mov, str(tmp_path / "gated"),
                          auto_minmass=False, minmass=.45, min_cnr=3.,
                          backend="trackpy", skip_figure=True))
    folder = Path(result['out_dir'])
    manifest = json.loads((folder / (result['stem']+'_run_manifest.json')).read_text())
    assert manifest['metrics_schema_version'] == 3
    assert manifest['parameters']['min_cnr'] == 3
    assert 'n_localisations_excluded_by_cnr' in manifest['parameters']
    loc = pd.read_csv(folder/'firefly_extras'/(result['stem']+'_localisations.csv'))
    assert len(loc) > 0 and loc.raw_cnr.ge(3).all()
    diff = pd.read_csv(folder/'firefly_extras'/(result['stem']+'_diffusion_summary.csv'))
    assert {'D_linear_raw','K_alpha','alpha_fit_status'} <= set(diff)
    assert diff.loc_sigma_nm.isna().all()


def test_drift_quality_and_reference_provenance_are_exported(tmp_path):
    import json
    import pandas as pd
    from pathlib import Path
    mov = _synthetic_movie(tmp_path/'stable.tif', n_frames=40, step=0)
    result = _run(_params(mov, str(tmp_path/'drift'), backend='trackpy',
        auto_minmass=False, minmass=.45, drift_correct=True, drift_segment=10,
        drift_adaptive=False, drift_min_locs=20, skip_figure=True))
    folder=Path(result['out_dir']); stem=result['stem']
    extras=folder/'firefly_extras'
    d=json.loads((extras/(stem+'_drift_diagnostics.json')).read_text())
    assert d['status']=='applied' and d['reference']['mode']=='Analysis region'
    assert d['n_accepted_pairs'] >= d['n_segments']
    assert (folder/'figures'/(stem+'_drift_diagnostic.png')).is_file()
    drift=pd.read_csv(extras/(stem+'_drift.csv'))
    assert {'status','support_pairs','extrapolated'} <= set(drift)
    manifest=json.loads((folder/(stem+'_run_manifest.json')).read_text())
    assert manifest['parameters']['drift_diagnostics']['status']=='applied'


def test_unsupported_drift_can_stop_pipeline(tmp_path):
    mov=_synthetic_movie(tmp_path/'sparse.tif', n_frames=10)
    with pytest.raises(ValueError,match='Drift correction unsupported'):
        _run(_params(mov,str(tmp_path/'stopped'),backend='trackpy',auto_minmass=False,
            minmass=.45,drift_correct=True,drift_min_locs=100000,drift_failure_policy='Stop run',
            skip_figure=True))
