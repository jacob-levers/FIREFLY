"""Each comparison panel can be exported on its own, at publication quality.

"Generate full report" saved only the combined figure, and the Analysis tab's
quick export was a screenshot (dark app theme, screen resolution, no title).
Now the report also writes every panel as a vector PDF and a 300-dpi PNG,
numbered in the figure's reading order, and the quick export saves the
selected panel the same way.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from firefly.analysis.fa_compare import (EXPORT_PANEL_DPI, compare_groups, compute_report,
                                         export_panel, panel_file_label)


def _groups(tmp_path):
    from test_per_recording_curves import _uneven
    return _uneven(tmp_path / "data")


def test_the_full_report_writes_every_panel(tmp_path):
    out = tmp_path / "out"; out.mkdir()
    panels = {"auc", "logd_dist", "mob_immob", "states_diagram"}
    fig, _s, _st = compare_groups(_groups(tmp_path), output_dir=str(out), output_stem="cmp",
                                  pdf_report=False, panels=panels)
    plt.close(fig)
    files = sorted(os.listdir(out / "cmp_panels"))
    stems = sorted({os.path.splitext(f)[0] for f in files})
    assert len(stems) == len(panels)
    assert {os.path.splitext(f)[1] for f in files} == {".pdf", ".png"}
    # numbered in the figure's reading order: the most important first
    assert stems[0] == "01_Area_under_the_MSD_curve"
    assert any(s.endswith("Diffusive_state_model") for s in stems)


def test_a_panel_file_is_the_panel_alone(tmp_path):
    from PIL import Image
    rd = compute_report(_groups(tmp_path))
    pdf, png = export_panel(rd, "auc", str(tmp_path / "auc"), theme="Light")
    assert os.path.getsize(pdf) > 0
    w, h = Image.open(png).size
    assert w < 4.5 * EXPORT_PANEL_DPI and h < 4.5 * EXPORT_PANEL_DPI    # one panel, not the grid
    assert w > 3 * EXPORT_PANEL_DPI                                      # at print resolution


def test_a_panel_only_render_has_no_figure_title_or_group_band(tmp_path):
    from firefly.analysis.fa_compare import render_report
    rd = compute_report(_groups(tmp_path))
    fig, _s, _st = render_report(rd, panels={"auc"}, panel_only=True, pdf_report=False)
    assert not (fig._suptitle and fig._suptitle.get_text())
    assert fig.texts == []
    assert any(a.get_title() == "Area under the MSD curve" for a in fig.axes)
    plt.close(fig)


def test_file_labels_are_safe_names():
    assert panel_file_label("states_occupancy") == "Diffusive_state_occupancy"
    assert panel_file_label("auc") == "Area_under_the_MSD_curve"
    assert panel_file_label("unknown_key") == "unknown_key"


def test_the_day_key_goes_only_on_panels_with_day_shaped_dots(tmp_path):
    """The state bars and the curves use plain dots; a key there misleads."""
    from firefly.analysis.fa_compare import render_report
    from test_recording_day_markers import DAYS, _groups as dated
    rd = compute_report(dated(tmp_path, DAYS))
    keyed = lambda fig: [lg for lg in fig.legends if getattr(lg, "_firefly_day_key", False)]
    for panel, want in (("rg", True), ("logd_dist", False), ("states_occupancy", False)):
        fig, _s, _st = render_report(rd, panels={panel}, panel_only=True, pdf_report=False)
        assert bool(keyed(fig)) == want, panel
        plt.close(fig)


def test_a_wrapped_day_key_reads_in_date_order(tmp_path):
    """Legends fill column by column; a narrow panel's key must still read
    16 Sep, 25 Sep, 1 Oct… left to right, row by row."""
    from firefly.analysis.fa_compare import render_report
    from test_recording_day_markers import _groups as dated
    days = [[f"2026-09-{d:02d}" for d in (16, 18, 20, 22)], [f"2026-09-{d:02d}" for d in (24, 26, 28, 30)]]
    rd = compute_report(dated(tmp_path, days))
    fig, _s, _st = render_report(rd, panels={"rg"}, panel_only=True, pdf_report=False)
    fig.canvas.draw()
    lg = next(lg for lg in fig.legends if getattr(lg, "_firefly_day_key", False))
    assert lg.get_title().get_text() == "Recording day"
    pos = [(round(-t.get_window_extent().y0), t.get_window_extent().x0, t.get_text())
           for t in lg.get_texts()]
    reading = [txt for _y, _x, txt in sorted(pos)]
    assert reading == [f"{d} Sep" for d in (16, 18, 20, 22, 24, 26, 28, 30)]
    plt.close(fig)
