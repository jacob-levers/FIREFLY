"""Comparison panels + per-replicate scalars for the track metrics.

Covers Fluorescence (per-track mean intensity), Radius of gyration, and the
track-geometry metrics: Net displacement (first→last), Path length,
Directionality ratio, Track duration, and Number of localisations — plus the
per-track geometry computed in fa_diffusion.  Each panel's per-replicate value
is computed the SAME way as its metric so panel and metric agree.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import json
import numpy as np
import pytest

pytest.importorskip("pandas")
import pandas as pd                                     # noqa: E402

from firefly.analysis import fa_compare as fc


def _run(root, stem, *, masses, seed=0):
    """A minimal run folder.  ``masses`` are per-TRACK constant intensities: each
    track gets 2 localisations at that mass, so its mean intensity == the mass."""
    extras = os.path.join(root, stem, "firefly_extras")
    os.makedirs(extras, exist_ok=True)
    with open(os.path.join(extras, f"{stem}_params.json"), "w") as fh:
        json.dump({"pixel_size_um": 0.1, "frame_interval_s": 0.05}, fh)
    pd.DataFrame({"lag_frame": [1, 2, 3], "msd_um2": [0.1, 0.2, 0.3]}).to_csv(
        os.path.join(extras, f"{stem}_ensemble_msd.csv"), index=False)
    rng = np.random.default_rng(seed)
    n = len(masses)
    pd.DataFrame({"particle": np.arange(n), "D": rng.random(n) * 0.5 + 0.01,
                  "alpha": rng.random(n) + 0.5,
                  "radius_of_gyration_um": rng.random(n) * 0.2 + 0.05,
                  "net_displacement_um": rng.random(n) + 0.1,
                  "path_length_um": rng.random(n) * 2 + 1.0,
                  "mean_step_um": rng.random(n) * 0.3 + 0.05,
                  "directionality_ratio": rng.random(n) * 0.5 + 0.25,
                  "motion": ["Brownian"] * n}).to_csv(
        os.path.join(extras, f"{stem}_diffusion_summary.csv"), index=False)
    rows = []
    for i, m in enumerate(masses):                      # 2 locs per track @ mass
        rows.append((i, 0, 0.0, 0.0, float(m)))
        rows.append((i, 1, 1.0, 0.0, float(m)))
    pd.DataFrame(rows, columns=["particle", "frame", "x", "y", "mass"]).to_csv(
        os.path.join(extras, f"{stem}_trajectories.csv"), index=False)
    return os.path.join(root, stem)


# ── fluorescence = median over tracks of each track's MEAN intensity ─────────
def test_spot_intensity_is_median_of_per_track_means(tmp_path):
    # each track's 2 locs share a mass, so its mean == that mass; median over
    # finite, >0 per-track means (100,200,300,400) → 250
    run = _run(str(tmp_path), "cellA", masses=[100, 200, 300, 400, 0, -5])
    from firefly.analysis.fa_palmtracer import load_summary_from_folder
    summary = load_summary_from_folder(run)
    assert fc._spot_intensity(summary) == 250.0


def test_spot_intensity_nan_without_tracks(tmp_path):
    assert np.isnan(fc._spot_intensity({}))             # no tracks → NaN, no crash


# ── the panel is registered and produces a per-replicate column + figure ─────
def test_fluor_panel_registered_next_to_auc():
    from firefly.ui.controllers.workspace import workspace_data as wd
    keys = [k for k, _ in wd.COMPARE_PANELS]
    assert "fluor" in keys
    assert keys.index("fluor") == keys.index("auc") + 1     # sits next to MSD AUC
    tabs = [k for k, _l, _m in wd.COMPARE_PANEL_TABS]
    assert "fluor" in tabs and wd.PANEL_METRIC["fluor"] == "fluor"
    assert "fluor" in wd.DEFAULT_COMPARE_PANELS


def test_compute_report_has_spot_intensity_and_panel_renders(tmp_path):
    groups = [
        {"label": "Ctrl", "color": "#3b6ed8", "folders": [
            _run(str(tmp_path), "c0", masses=[100, 200, 300], seed=1),
            _run(str(tmp_path), "c1", masses=[110, 210, 310], seed=2)]},
        {"label": "Drug", "color": "#d8683b", "folders": [
            _run(str(tmp_path), "d0", masses=[500, 600, 700], seed=3),
            _run(str(tmp_path), "d1", masses=[520, 620, 720], seed=4)]},
    ]
    rd = fc.compute_report(groups)
    sdf = rd.summary_df
    assert "spot_intensity" in sdf.columns
    # Ctrl medians ~200/210, Drug ~600/620 → Drug clearly higher
    ctrl = sdf.loc[sdf["group"] == "Ctrl", "spot_intensity"].to_numpy(float)
    drug = sdf.loc[sdf["group"] == "Drug", "spot_intensity"].to_numpy(float)
    assert np.nanmedian(drug) > np.nanmedian(ctrl)
    assert set(np.round(ctrl)) == {200.0, 210.0}

    # the panel renders without error and records stats under 'spot_intensity'
    fig, _sdf, stats = fc.render_report(rd, panels={"fluor"})
    assert fig is not None
    assert "spot_intensity" in stats
    import matplotlib.pyplot as plt
    plt.close(fig)


# ── radius of gyration: same pattern, per-track column already loaded ─────────
def test_rg_panel_registered_after_track_length():
    from firefly.ui.controllers.workspace import workspace_data as wd
    keys = [k for k, _ in wd.COMPARE_PANELS]
    assert "rg" in keys
    assert keys.index("rg") == keys.index("track_length") + 1
    tabs = [k for k, _l, _m in wd.COMPARE_PANEL_TABS]
    assert "rg" in tabs and wd.PANEL_METRIC["rg"] == "rg"
    assert "rg" in wd.DEFAULT_COMPARE_PANELS


def test_compute_report_has_radius_of_gyration_and_panel_renders(tmp_path):
    groups = [
        {"label": "Ctrl", "color": "#3b6ed8", "folders": [
            _run(str(tmp_path), "c0", masses=[100, 200, 300], seed=1),
            _run(str(tmp_path), "c1", masses=[110, 210, 310], seed=2)]},
        {"label": "Drug", "color": "#d8683b", "folders": [
            _run(str(tmp_path), "d0", masses=[500, 600, 700], seed=3),
            _run(str(tmp_path), "d1", masses=[520, 620, 720], seed=4)]},
    ]
    rd = fc.compute_report(groups)
    assert "radius_of_gyration" in rd.summary_df.columns
    vals = rd.summary_df["radius_of_gyration"].to_numpy(float)
    assert np.isfinite(vals).all() and (vals > 0).all()   # per-replicate median R_g
    fig, _sdf, stats = fc.render_report(rd, panels={"rg"})
    assert fig is not None and "radius_of_gyration" in stats
    import matplotlib.pyplot as plt
    plt.close(fig)


# ── per-track path geometry (fa_diffusion) — the user's exact definitions ─────
def test_track_geometry_net_path_directionality():
    """net displacement = straight line first→last; path length = Σ step
    distances; directionality = net ÷ path.  Track (0,0)→(3,0)→(3,4):
    steps 3 and 4 → path 7; first→last (0,0)→(3,4) → net 5; ratio 5/7."""
    from firefly.analysis.fa_diffusion import compute_msd_and_fit
    tr = pd.DataFrame({"particle": [0, 0, 0], "frame": [0, 1, 2],
                       "x": [0.0, 3.0, 3.0], "y": [0.0, 0.0, 4.0]})
    _imsd, _emsd, diff = compute_msd_and_fit(tr, pixel_size=1.0,
                                             frame_interval=1.0, workers=1)
    row = diff.iloc[0]
    assert abs(float(row["path_length_um"]) - 7.0) < 1e-9
    assert abs(float(row["net_displacement_um"]) - 5.0) < 1e-9
    assert abs(float(row["directionality_ratio"]) - 5.0 / 7.0) < 1e-9
    # measured step distance = mean of the two steps (3, 4) = 3.5 — NOT derived from D
    assert abs(float(row["mean_step_um"]) - 3.5) < 1e-9


def test_step_metrics_are_measured_not_derived_from_D(tmp_path):
    """Step distance / Step speed must read the MEASURED per-step column, not the
    old √(2·D·Δt) approximation."""
    from firefly.ui.controllers.workspace import workspace_data as wd
    extras = tmp_path / "run" / "firefly_extras"; extras.mkdir(parents=True)
    (extras / "run_summary_metrics.json").write_text(json.dumps(
        {"fi_s": 0.5, "px_um": 0.1, "stem": "run"}))   # fi_s → run.fi_s
    pd.DataFrame({"particle": [0, 1], "D": [0.1, 0.2], "alpha": [1.0, 1.0],
                  "mean_step_um": [0.2, 0.4], "motion": ["Brownian"]*2}).to_csv(
        extras / "run_diffusion_summary.csv", index=False)
    run = wd.load_run(str(tmp_path / "run"))
    step = next(m for m in wd.METRICS if m.id == "step")
    speed = next(m for m in wd.METRICS if m.id == "speed")
    assert step.scalar(run) == pytest.approx(0.3)        # median(0.2, 0.4)
    assert speed.scalar(run) == pytest.approx(0.3 / 0.5)  # median step ÷ Δt
    assert step.approx is False and speed.approx is False


def test_net_displacement_metric_uses_first_to_last_not_centroid(tmp_path):
    """Regression for the mislabel: the 'Net displacement' metric must read the
    first→last column, NOT mean_radial_displacement_um."""
    from firefly.ui.controllers.workspace import workspace_data as wd
    extras = tmp_path / "run" / "firefly_extras"; extras.mkdir(parents=True)
    (extras / "run_params.json").write_text(json.dumps(
        {"pixel_size_um": 0.1, "frame_interval_s": 0.05}))
    pd.DataFrame({"particle": [0, 1], "D": [0.1, 0.2], "alpha": [1.0, 1.0],
                  "net_displacement_um": [5.0, 5.0],           # what we want
                  "mean_radial_displacement_um": [99.0, 99.0],  # the old (wrong) col
                  "motion": ["Brownian", "Brownian"]}).to_csv(
        extras / "run_diffusion_summary.csv", index=False)
    run = wd.load_run(str(tmp_path / "run"))
    metric = next(m for m in wd.METRICS if m.id == "netdisp")
    assert metric.scalar(run) == 5.0                     # first→last, not 99


def test_new_scalar_metrics_and_panels_registered():
    from firefly.ui.controllers.workspace import workspace_data as wd
    ids = {m.id for m in wd.METRICS}
    assert {"netdisp", "path", "step", "speed", "linkstep", "linkspeed",
            "dir", "dur", "nlocs"} <= ids
    keys = {k for k, _ in wd.COMPARE_PANELS}
    assert {"netdisp", "path", "step", "speed", "linkstep", "linkspeed",
            "dir", "dur", "nlocs"} <= keys
    for k in ("netdisp", "path", "step", "speed", "linkstep", "linkspeed",
              "dir", "dur", "nlocs"):
        assert wd.PANEL_METRIC[k] == k


def test_compute_report_has_new_track_metrics_and_panels_render(tmp_path):
    groups = [
        {"label": "A", "color": "#3b6ed8", "folders": [
            _run(str(tmp_path), "a0", masses=[100, 200], seed=1),
            _run(str(tmp_path), "a1", masses=[110, 210], seed=2)]},
        {"label": "B", "color": "#d8683b", "folders": [
            _run(str(tmp_path), "b0", masses=[300, 400], seed=3),
            _run(str(tmp_path), "b1", masses=[320, 420], seed=4)]},
    ]
    rd = fc.compute_report(groups)
    for col in ("net_displacement", "path_length", "step_distance", "step_speed",
                "directionality", "track_duration", "n_localisations"):
        assert col in rd.summary_df.columns, col
    # 2 locs per track → duration = (2-1)*0.05 = 0.05 s; 2 tracks → 4 locs
    assert np.allclose(rd.summary_df["track_duration"].to_numpy(float), 0.05)
    assert set(rd.summary_df["n_localisations"].to_numpy(float)) == {4.0}
    fig, _sdf, stats = fc.render_report(
        rd, panels={"netdisp", "path", "step", "speed", "dir", "dur", "nlocs"})
    assert fig is not None
    for col in ("net_displacement", "path_length", "step_distance", "step_speed",
                "directionality", "track_duration", "n_localisations"):
        assert col in stats, col
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_single_run_zero_only_histogram_bins_are_strictly_increasing():
    from firefly.analysis.fa_figure import _safe_linear_bins

    bins = _safe_linear_bins(np.zeros(12), 40, nonnegative=True)
    assert np.all(np.diff(bins) > 0)
    assert bins[0] == 0.0
    assert bins[-1] > 0.0
