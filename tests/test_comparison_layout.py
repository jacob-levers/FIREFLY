"""The full comparison figure: panels of the same shape share rows, and the
most important measures sit at the top.

Shapes: box / violin / bar comparisons are near square (a third of a row); the
distribution curves and per-group MSD facets are wide (half a row); with many
groups a panel widens so its labels fit.  A row holds one shape only, a short
row is centred, and rows run in PANEL_IMPORTANCE order of their first panel —
the lab's headline measures (MSD, AUC, D distribution, mobile fraction) first,
trajectory bookkeeping last.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from firefly.analysis.fa_compare import (COMPARISON_GRID_COLS, PANEL_IMPORTANCE,
                                         comparison_layout, compare_groups, panel_span)

G = COMPARISON_GRID_COLS
SQUARE, WIDE = G // 3, G // 2
ALL = ["msd", "auc", "fluor", "logd_dist", "mob_immob", "motion_classes",
       "states_occupancy", "states_d", "track_length",
       "rg", "netdisp", "path", "step", "speed", "linkstep", "linkspeed", "dir", "dur",
       "track_count", "nlocs", "jdd", "dwell_cdf", "turning_angles", "radial_dist",
       "van_hove", "vacf"]


def test_panel_shapes():
    assert panel_span("auc", 3) == SQUARE and panel_span("rg", 2) == SQUARE
    for key in ("logd_dist", "track_length", "dwell_cdf", "turning_angles", "msd"):
        assert panel_span(key, 3) == WIDE
    assert panel_span("msd", 5) == G                              # one facet a group
    assert panel_span("radial_dist", 3) == SQUARE                 # polar: square
    assert panel_span("auc", 6) == WIDE and panel_span("auc", 10) == G
    assert panel_span("logd_dist", 3, logd_style="violin") == SQUARE


def _rows(keys, n=3):
    spans = [panel_span(k, n) for k in keys]
    place, nrows = comparison_layout(keys, spans)
    rows = [[] for _ in range(nrows)]
    for k, (r, c, s, _h) in zip(keys, place):
        rows[r].append((c, s, k))
    return [sorted(r) for r in rows]


def test_a_row_holds_one_shape_and_never_overflows():
    rows = _rows(ALL)
    mixed = [row for row in rows if len({s for _c, s, _k in row}) > 1]
    assert len(mixed) <= 1                       # only the leftovers may share a row
    for row in rows:
        assert sum(s for _c, s, _k in row) <= G
        cols = [c for c, _s, _k in row]
        assert cols == [cols[0] + sum(s for _c, s, _k in row[:j]) for j in range(len(row))]


def test_short_rows_are_centred():
    for row in _rows(ALL):
        used = sum(s for _c, s, _k in row)
        assert row[0][0] == (G - used) // 2


def test_the_headline_measures_lead():
    rows = _rows(ALL)
    assert [k for _c, _s, k in rows[0]] == ["msd", "logd_dist"]
    assert [k for _c, _s, k in rows[1]] == ["auc", "mob_immob", "jdd"]
    assert [k for _c, _s, k in rows[2]] == ["states_occupancy", "states_d"]
    flat = [k for row in rows for _c, _s, k in row]
    for late in ("track_count", "nlocs", "fluor", "linkstep", "linkspeed"):
        assert flat.index(late) > flat.index("netdisp")
    rank = {k: i for i, k in enumerate(PANEL_IMPORTANCE)}
    firsts = [min(rank[k] for _c, _s, k in row) for row in rows]
    assert firsts == sorted(firsts)                               # rows run by importance


def test_a_warning_card_comes_first():
    keys = ["__contract_diffusion", "auc", "rg"]
    rows = _rows(keys)
    assert [k for _c, _s, k in rows[0]] == ["__contract_diffusion"]   # a full-width notice


def _groups(tmp_path):
    from test_per_recording_curves import _uneven
    return _uneven(tmp_path)


def test_the_report_draws_square_and_wide_panels(tmp_path):
    fig, _s, _st = compare_groups(_groups(tmp_path), output_dir=None, pdf_report=False,
                                  panels={"auc", "logd_dist", "rg", "dwell_cdf"},
                                  auc_plot_style="box_points")
    auc = next(a for a in fig.axes if a.get_ylabel().startswith("AUC")).get_position()
    dens = next(a for a in fig.axes if a.get_ylabel().startswith("Probability density")).get_position()
    w_in, h_in = fig.get_size_inches()
    aspect = lambda p: (p.width * w_in) / (p.height * h_in)
    assert 0.7 < aspect(auc) < 1.5                                # near square
    assert aspect(dens) > 1.4 * aspect(auc)                       # wide
    assert auc.y0 > dens.y0          # AUC outranks the D distribution: its row is higher
    plt.close(fig)


@pytest.mark.parametrize("panel,inches", [("auc", 4.2), ("logd_dist", 6.3)])
def test_a_single_panel_render_is_that_panel_s_shape(tmp_path, panel, inches):
    """The Analysis tab renders one panel at a time; the figure is that panel."""
    fig, _s, _st = compare_groups(_groups(tmp_path), output_dir=None, pdf_report=False,
                                  panels={panel}, minimal=True)
    assert fig.get_size_inches()[0] == pytest.approx(inches)
    plt.close(fig)


def test_leftover_panels_share_a_row_instead_of_standing_alone():
    """MB112C's default panels used to end in a lone trajectory-length row and a
    lone fluorescence row."""
    keys = [k for k in ALL if k not in ("track_count", "linkstep", "linkspeed",
                                        "states_occupancy", "states_d")]
    rows = _rows(keys)
    assert sum(1 for row in rows if len(row) == 1) <= 1
    last = {k for _c, _s, k in rows[-1]}
    assert {"track_length", "fluor"} <= last
