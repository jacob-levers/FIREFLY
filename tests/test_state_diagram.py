"""The diffusive-state diagram: one per group, numbers that never collide.

A first draft drew the stay-probability loops into their own circles (so they
vanished) and set some labels on top of loops and arrows.  Here every diagram is
rendered at the smallest and the largest size the report uses, for extreme and
ordinary values, and no two labels may overlap, leave the panel, or sit on a
circle other than their own.
"""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from firefly.analysis.fa_figure_common import (STATE_DIAGRAM_POS, draw_state_diagram,
                                               format_probability, state_radius)
from firefly.analysis.fa_states import STATE_COLORS, STATE_NAMES

CASES = {
    "MB112C-like": ([0.018, 0.066, 0.261], [0.11, 0.29, 0.60],
                    [[0.87, 0.10, 0.03], [0.05, 0.85, 0.10], [0.01, 0.07, 0.92]]),
    "one state dominates": ([0.001, 0.024, 0.229], [0.02, 0.03, 0.95],
                            [[0.999, 0.0004, 0.0006], [0.30, 0.40, 0.30], [0.0004, 0.002, 0.9976]]),
    "immobile dominates": ([0.009, 0.120, 0.480], [0.90, 0.05, 0.05],
                           [[0.98, 0.01, 0.01], [0.20, 0.60, 0.20], [0.30, 0.20, 0.50]]),
    "even": ([0.012, 0.150, 0.600], [0.33, 0.33, 0.34],
             [[0.55, 0.25, 0.20], [0.25, 0.50, 0.25], [0.20, 0.25, 0.55]]),
}


def _render(case, side_in, fontscale):
    D, occ, P = CASES[case]
    fig = plt.figure(figsize=(side_in, side_in))
    ax = fig.add_axes([0, 0, 1, 1])
    draw_state_diagram(ax, D, occ, P, names=STATE_NAMES, colors=STATE_COLORS,
                       text_color="#222222", background="#ffffff", title="Control  (n = 9)",
                       title_color="#5b6770", fontscale=fontscale)
    fig.canvas.draw()
    return fig, ax


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("side_in,fontscale", [(3.15, 0.94), (2.45, 0.72), (3.35, 1.0), (2.0, 0.6)])
def test_no_label_collides(case, side_in, fontscale):
    fig, ax = _render(case, side_in, fontscale)
    r = fig.canvas.get_renderer()
    texts = [t for t in ax.texts if t.get_text().strip()]
    boxes = [t.get_window_extent(r) for t in texts]
    for (ta, a), (tb, b) in itertools.combinations(zip(texts, boxes), 2):
        assert not a.overlaps(b), (case, ta.get_text(), tb.get_text())
    frame = fig.bbox
    for t, b in zip(texts, boxes):
        assert frame.x0 - 0.5 <= b.x0 and b.x1 <= frame.x1 + 0.5, t.get_text()
        assert frame.y0 - 0.5 <= b.y0 and b.y1 <= frame.y1 + 0.5, t.get_text()
    # nothing but a state's own occupancy sits on a circle
    _D, occ, _P = CASES[case]
    for i, (cx, cy) in enumerate(STATE_DIAGRAM_POS):
        c = ax.transData.transform((cx, cy))
        rad = ax.transData.transform((cx + state_radius(occ[i]), cy))[0] - c[0]
        for t, b in zip(texts, boxes):
            if t.get_text() == f"{100 * occ[i]:.0f}%":
                continue
            nearest = np.array([np.clip(c[0], b.x0, b.x1), np.clip(c[1], b.y0, b.y1)])
            assert np.linalg.norm(nearest - c) > rad, (case, i, t.get_text())
    plt.close(fig)


def test_every_number_is_shown_cleanly():
    _fig, ax = _render("one state dominates", 3.15, 0.94)
    shown = {t.get_text() for t in ax.texts}
    assert {"<0.01", "1.00", "0.40"} <= shown            # 0.999 and 0.9976 round to 1.00
    assert "0.00" not in shown
    assert format_probability(0.004) == "<0.01" and format_probability(0.27) == "0.27"
    assert {"2%", "3%", "95%"} <= shown
    plt.close(_fig)


# ── in the comparison report ────────────────────────────────────────────────
def test_the_report_draws_one_diagram_per_group(tmp_path):
    from test_diffusive_states import _state_groups
    from firefly.analysis.fa_compare import compare_groups
    groups = _state_groups(tmp_path)
    fig, sdf, _st = compare_groups(groups, output_dir=None, pdf_report=False,
                                   panels={"states_diagram"})
    titled = [a for a in fig.axes for t in a.texts if t.get_text().startswith(("A  (n", "B  (n"))]
    assert len(titled) == 2
    a_ax = next(a for a in titled if any(t.get_text().startswith("A  (n = 3)") for t in a.texts))
    occ = sdf.loc[sdf.group == "A", "state_occupancy_immobile"].mean()
    assert f"{100 * occ:.0f}%" in {t.get_text() for t in a_ax.texts}
    plt.close(fig)


def test_many_groups_wrap_onto_more_rows():
    from firefly.analysis.fa_compare import comparison_layout, panel_rows, panel_span
    assert panel_rows("states_diagram", 3) == 1 and panel_rows("states_diagram", 6) == 2
    keys = ["auc", "states_diagram", "rg"]
    place, nrows = comparison_layout(keys, [panel_span(k, 6) for k in keys],
                                     [panel_rows(k, 6) for k in keys])
    r, c, s, h = place[1]
    assert h == 2 and s == 12 and nrows == place[1][0] + 2 + (place[2][0] > r)
    assert all(not (r <= rr < r + h) for i, (rr, _c, _s, _h) in enumerate(place) if i != 1)
