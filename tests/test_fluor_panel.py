"""Fluorescence (spot-intensity) comparison panel + per-replicate scalar.

The Analysis tab gained a Fluorescence panel next to MSD AUC.  Its per-replicate
value is the median spot intensity (the localisations' `mass` column = palmTRACER's
Integrated_Intensity), computed the SAME way as the existing 'Spot intensity'
metric so the panel and the metric agree.
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
    """A minimal run folder that carries a localisations table with `mass`."""
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
                  "motion": ["Brownian"] * n}).to_csv(
        os.path.join(extras, f"{stem}_diffusion_summary.csv"), index=False)
    pd.DataFrame({"frame": np.arange(n), "x": rng.random(n), "y": rng.random(n),
                  "mass": np.asarray(masses, float)}).to_csv(
        os.path.join(extras, f"{stem}_localisations.csv"), index=False)
    return os.path.join(root, stem)


# ── the per-replicate scalar matches the 'Spot intensity' metric definition ──
def test_spot_intensity_is_median_localisation_mass(tmp_path):
    run = _run(str(tmp_path), "cellA", masses=[100, 200, 300, 400, 0, -5])
    from firefly.analysis.fa_palmtracer import load_summary_from_folder
    summary = load_summary_from_folder(run)
    # median over finite, >0 masses (100,200,300,400) → 250
    assert fc._spot_intensity(summary) == 250.0


def test_spot_intensity_nan_without_localisations(tmp_path):
    # a summary with no data_dir/stem → NaN, no crash
    assert np.isnan(fc._spot_intensity({}))


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
