"""The comparison's mobility panel shows the mobile fraction, not mobile/immobile.

The ratio is unbounded and undefined when a recording has no immobile track
(the golden fixture holds replicates at 119, 599 and NaN), so one recording can
stretch the axis and another drops out of the test.  The fraction mobile /
(mobile + immobile) is bounded 0–1 and always defined when any track is.

The live stats card (``workspace_data._mobile_pct``) quotes the same quantity in
%, so it has to use the same tracks: it used to count only D > 0, dropping the
tracks whose MSD slope came out non-positive, which ``mobility_masks`` counts
as immobile.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from firefly.analysis.fa_compare import compare_groups
from firefly.analysis.fa_diffusion import _mobile_fraction


def _groups(tmp_path):
    from test_workspace_data import make_run_folder
    return [{"label": f"G{gi}", "color": c,
             "folders": [make_run_folder(str(tmp_path), f"g{gi}_{k}", seed=10 * gi + k, d_centre=dc)
                         for k in range(3)]}
            for gi, (c, dc) in enumerate([("#0072b2", 0.02), ("#d55e00", 0.2)])]


def _diff(d, status=None):
    df = pd.DataFrame({"D": np.asarray(d, float)})
    if status is not None:
        df["fit_status"] = status
    return df


def test_the_fraction_counts_a_non_positive_slope_as_immobile():
    d = _diff([0.5, 0.2, 0.001, np.nan, np.nan],
              ["ok", "ok", "ok", "nonpositive_slope", "too_short"])
    assert _mobile_fraction(d, 0.01) == pytest.approx(2 / 4)


def test_no_immobile_track_is_a_fraction_of_one_not_a_gap():
    assert _mobile_fraction(_diff([0.5, 0.9]), 0.01) == 1.0
    assert np.isnan(_mobile_fraction(_diff([np.nan]), 0.01))


def test_the_panel_plots_the_fraction_and_tests_it(tmp_path):
    fig, sdf, stats = compare_groups(_groups(tmp_path), output_dir=None, panels={"mob_immob"},
                                     pdf_report=False, group_style="box_points")
    assert "mobile_fraction" in stats and "mob_immob_ratio" not in stats
    assert sdf["mobile_fraction"].between(0, 1).all()
    ax = next(a for a in fig.axes if a.get_ylabel())
    assert ax.get_ylabel() == "Mobile fraction" and ax.get_title() == "Mobile fraction"
    plt.close(fig)


def test_the_p_value_clears_the_dots_and_stays_inside_the_panel(tmp_path):
    """Forcing the axis to start at 0 after the bracket was placed squashed the
    data into the top of the panel: the bracket sat on the highest dots and the
    label ran into the frame.  Fractions of 0.6–0.75, as on the fly data."""
    from test_workspace_data import make_run_folder
    from firefly.analysis.fa_diffusion import MOBILE_D_THRESHOLD_DEFAULT as thr
    groups = [{"label": f"G{gi}", "color": c,
               "folders": [make_run_folder(str(tmp_path), f"m{gi}_{k}", seed=40 + 10 * gi + k,
                                           n_tracks=300, d_centre=thr * f) for k in range(4)]}
              for gi, (c, f) in enumerate([("#0072b2", 1.45), ("#d55e00", 1.2)])]
    fig, sdf, _st = compare_groups(groups, output_dir=None, panels={"mob_immob"},
                                   pdf_report=False, group_style="box_points")
    assert sdf["mobile_fraction"].between(0.5, 0.85).all(), sdf["mobile_fraction"]
    fig.canvas.draw()
    ax = next(a for a in fig.axes if a.get_ylabel() == "Mobile fraction")
    top_dot = ax.transData.transform((0, sdf["mobile_fraction"].max()))[1]
    frame = ax.get_window_extent()
    (txt,) = [t for t in ax.texts if t.get_text().startswith("p ")]
    bb = txt.get_window_extent()
    assert bb.y0 > top_dot and bb.y1 <= frame.y1 + 1
    plt.close(fig)


def test_the_live_card_quotes_the_same_fraction_as_the_report(tmp_path):
    from firefly.ui.controllers.workspace import workspace_data as wd
    extras = tmp_path / "extras"; extras.mkdir()
    d = _diff([0.5, 0.2, 0.001, np.nan], ["ok", "ok", "ok", "nonpositive_slope"])
    d.to_csv(extras / "rec_diffusion_summary.csv", index=False)
    run = wd.RunData(str(tmp_path), "rec", str(extras), {"mobile_fraction": 0.9})
    try:
        wd.RunData.mobile_d = 0.01
        assert wd._mobile_pct(run) == pytest.approx(100 * _mobile_fraction(d, 0.01))
    finally:
        wd.RunData.mobile_d = wd.MOBILE_D_THRESHOLD_DEFAULT
