"""MSD AUC — the area under the ensemble-MSD curve.

An ensemble MSD legitimately carries NaN at its longest lags: the curve is a
nanmean across tracks and ``compute_msd_and_fit`` fills only
``where=valid_counts > 0``, so a lag no track is long enough to reach stays NaN.
That is an ordinary recording, not a broken one.

The AUC integrator used to run straight through those NaNs and return NaN for
the *whole* run.  The effect was confined to one panel and was silent: the MSD
panel drew every replicate, and the AUC panel beside it dropped the affected
ones — n=4 per condition rendering as n=2, with the stats degrading to "n<3
replicates - underpowered, not interpretable".

Worse, the quantity had two implementations — the analysis core's and the
Analysis tab's own copy — which disagreed on exactly this case, so the live card
showed an AUC that the exported panel had thrown away.
"""
import numpy as np
import pandas as pd
import pytest

from firefly.analysis.fa_diffusion import _msd_auc


def _curve(n_lags=10, d=0.05, alpha=0.85, dt=0.02, nan_tail=0):
    lags = np.arange(1, n_lags + 1)
    msd = 4 * d * (lags * dt) ** alpha
    if nan_tail:
        msd[-nan_tail:] = np.nan
    return pd.DataFrame({"lag_frame": lags, "msd_um2": msd})


# ── the defect ───────────────────────────────────────────────────────────────
def test_an_unpopulated_lag_does_not_void_the_whole_curve():
    """The regression: two NaN lags at the tail used to return NaN outright."""
    auc = _msd_auc(_curve(nan_tail=2), 0.02)
    assert np.isfinite(auc), "a NaN lag voided the entire run's AUC"
    assert auc > 0


def test_it_integrates_exactly_the_lags_that_have_data():
    """Not 'NaN treated as zero' — the missing lags are skipped, so the result
    equals the AUC of the curve that remains."""
    full = _curve(n_lags=10)
    trimmed = full.iloc[:8].copy()               # the same curve, tail removed
    holed = _curve(n_lags=10, nan_tail=2)
    assert _msd_auc(holed, 0.02) == pytest.approx(_msd_auc(trimmed, 0.02))
    # and that is strictly less than integrating all ten lags
    assert _msd_auc(holed, 0.02) < _msd_auc(full, 0.02)


def test_a_hole_in_the_middle_is_bridged_not_zeroed():
    c = _curve(n_lags=10)
    c.loc[c["lag_frame"] == 5, "msd_um2"] = np.nan
    auc = _msd_auc(c, 0.02)
    assert np.isfinite(auc)
    # trapezoid across the gap sits between dropping the span and keeping it
    assert 0 < auc < _msd_auc(_curve(n_lags=10), 0.02) * 1.05


def test_a_clean_curve_is_unchanged():
    """The fix must not move any existing number: with no NaN present the
    result is the plain trapezoid it always was."""
    c = _curve(n_lags=10)
    t = c["lag_frame"].to_numpy(float) * 0.02
    y = c["msd_um2"].to_numpy(float)
    assert _msd_auc(c, 0.02) == pytest.approx(float(np.trapezoid(y, t)))


# ── the honest failures still fail ───────────────────────────────────────────
def test_an_all_nan_curve_is_still_nan():
    assert np.isnan(_msd_auc(_curve(n_lags=10, nan_tail=10), 0.02))


def test_a_single_usable_lag_cannot_make_an_area():
    assert np.isnan(_msd_auc(_curve(n_lags=10, nan_tail=9), 0.02))


def test_an_empty_curve_is_nan():
    assert np.isnan(_msd_auc(pd.DataFrame({"lag_frame": [], "msd_um2": []}), 0.02))
    assert np.isnan(_msd_auc(None, 0.02))


def test_unsorted_lags_are_ordered_before_integrating():
    c = _curve(n_lags=10).sample(frac=1.0, random_state=0)
    assert _msd_auc(c, 0.02) == pytest.approx(_msd_auc(_curve(n_lags=10), 0.02))


# ── one implementation, not two ──────────────────────────────────────────────
def test_the_live_card_and_the_engine_use_the_same_integrator(tmp_path):
    """They were separate copies that disagreed on the NaN case, so the tab
    could show an AUC the exported panel had already dropped."""
    import os
    from firefly.ui.controllers.workspace.workspace_data import RunData
    from firefly.ui.controllers.workspace import workspace_data as wd

    extras = tmp_path / "extras"
    extras.mkdir()
    curve = _curve(n_lags=10, nan_tail=2)
    curve.to_csv(extras / "rec_ensemble_msd.csv", index=False)
    run = RunData(str(tmp_path), "rec", str(extras), {"fi_s": 0.02})

    live = wd._msd_auc(run)
    engine = _msd_auc(curve, 0.02)
    assert live is not None, "the live card dropped a run the engine keeps"
    assert live == pytest.approx(engine)


def test_the_live_card_reports_nothing_when_the_engine_reports_nan(tmp_path):
    from firefly.ui.controllers.workspace.workspace_data import RunData
    from firefly.ui.controllers.workspace import workspace_data as wd

    extras = tmp_path / "extras"
    extras.mkdir()
    _curve(n_lags=10, nan_tail=10).to_csv(extras / "rec_ensemble_msd.csv",
                                          index=False)
    run = RunData(str(tmp_path), "rec", str(extras), {"fi_s": 0.02})
    assert wd._msd_auc(run) is None


# ── the panel keeps its replicates ───────────────────────────────────────────
def test_the_comparison_keeps_every_replicate_in_the_auc_panel(tmp_path):
    """End to end: the MSD panel and the AUC panel must agree on how many
    replicates each condition has."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from test_workspace_data import make_run_folder
    from firefly.analysis.fa_compare import compare_groups

    groups = []
    for gi, (name, d) in enumerate((("Control", 0.05), ("Treated", 0.09))):
        folders = []
        for k in range(4):
            f = make_run_folder(str(tmp_path), f"{name}{k}",
                                seed=gi * 20 + k, d_centre=d)
            # half the replicates have unpopulated tail lags — the normal case
            _curve(n_lags=10, d=d, nan_tail=2 if k % 2 == 0 else 0).to_csv(
                os.path.join(f, "firefly_extras", f"{name}{k}_ensemble_msd.csv"),
                index=False)
            folders.append(f)
        groups.append({"folders": folders, "label": name,
                       "color": "#58a6ff", "timepoint": ""})

    _fig, summary, _stats = compare_groups(groups, output_dir=None,
                                           panels={"msd", "auc"},
                                           pdf_report=False)
    assert summary["auc_msd"].notna().all(), (
        "the AUC panel lost replicates the MSD panel kept:\n"
        + summary[["group", "auc_msd"]].to_string(index=False))
    assert (summary.groupby("group")["auc_msd"].count() == 4).all()
