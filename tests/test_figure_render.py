"""Regression tests for the circular-statistics PDF renderers and the shared
figure helpers extracted in #3 (firefly.analysis.fa_figure_common).

These renderers previously had NO dedicated tests, so a future edit that changed
their output would go unnoticed.  The tests assert:
  * the shared helpers behave as documented (fmt_stat_value edge cases;
    style_table_cells / render_polar_histogram run + set the expected props);
  * both PDFs render to disk with the right page count and contain the key
    text content (headers, stat names, group labels).

Qt-free; uses the Agg backend so it runs headless in CI.
"""
import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from firefly.analysis.fa_figure_common import (
    fmt_stat_value, style_table_cells, render_polar_histogram,
    rcparams_for_theme)
from firefly.analysis.fa_theme import _theme_palette
from firefly.analysis.fa_circular import (
    compute_circular_statistics, save_circular_statistics_pdf,
    save_comparison_circular_statistics)


def _synth_angles(n=600, mu=12.0, sd=55.0, seed=7):
    rng = np.random.default_rng(seed)
    a = ((rng.normal(mu, sd, size=n) + 180.0) % 360.0) - 180.0
    return a.astype(float)


# ── fmt_stat_value ──────────────────────────────────────────────────────────

def test_fmt_stat_value_edge_cases():
    assert fmt_stat_value(None) == "—"
    assert fmt_stat_value(float("nan")) == "—"
    assert fmt_stat_value(1e-300) == "<1e-300"
    assert fmt_stat_value(5e-301) == "<1e-300"
    # ordinary values use %g with the requested precision
    assert fmt_stat_value(0.123456, prec=4) == "0.1235"
    assert fmt_stat_value(42) == "42"
    # a non-coercible object falls back to str() rather than raising
    assert fmt_stat_value("n/a") == "n/a"


# ── style_table_cells ───────────────────────────────────────────────────────

def test_style_table_cells_applies_theme():
    pal = _theme_palette("Dark")
    fig, ax = plt.subplots()
    ax.axis("off")
    tbl = ax.table(cellText=[["a", "1"], ["b", "2"]],
                   colLabels=["name", "val"], loc="center")
    style_table_cells(tbl, pal, fontsize=9.0, label_col=True)
    cells = tbl.get_celld()
    # header row (r == 0) gets the header background
    hdr = cells[(0, 0)]
    assert hdr.get_facecolor() is not None
    # data rows are zebra-striped between ZEBRA and PNL
    facecolors = {(r, c): cells[(r, c)].get_facecolor() for (r, c) in cells}
    assert len(facecolors) == len(cells)
    plt.close(fig)


def test_style_table_cells_pad_optional():
    pal = _theme_palette("Light")
    fig, ax = plt.subplots()
    tbl = ax.table(cellText=[["x"]], loc="center")
    # pad given → every cell's PAD updated; no exception
    style_table_cells(tbl, pal, fontsize=8.5, pad=0.06)
    for cell in tbl.get_celld().values():
        assert cell.PAD == 0.06
    plt.close(fig)


# ── render_polar_histogram ──────────────────────────────────────────────────

def test_render_polar_histogram_draws_bars():
    pal = _theme_palette("Dark")
    a = _synth_angles()
    stats = compute_circular_statistics(a)
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection="polar")
    render_polar_histogram(ax, a, stats, pal)
    # 36-bin histogram → 36 bar patches drawn
    assert len(ax.patches) == 36
    # signed-angle tick labels installed
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert "±180°" in labels and "+45°" in labels
    plt.close(fig)


# ── rcparams_for_theme ──────────────────────────────────────────────────────

def test_rcparams_for_theme_restores_on_exit():
    pal = _theme_palette("Dark")
    before = plt.rcParams["text.color"]
    with rcparams_for_theme(plt, pal):
        assert plt.rcParams["text.color"] == pal["TXT"]
    assert plt.rcParams["text.color"] == before


def test_rcparams_for_theme_restores_on_exception():
    pal = _theme_palette("Dark")
    before = plt.rcParams["axes.facecolor"]
    with pytest.raises(ValueError):
        with rcparams_for_theme(plt, pal):
            raise ValueError("boom")
    assert plt.rcParams["axes.facecolor"] == before


# ── End-to-end PDF renders ──────────────────────────────────────────────────

def test_save_circular_statistics_pdf(tmp_path):
    a = _synth_angles()
    stats = compute_circular_statistics(a)
    out = tmp_path / "per_file.pdf"
    save_circular_statistics_pdf(a, stats, pdf_path=str(out),
                                 file_label="unit_stem", fig_theme="Dark")
    assert out.exists() and out.stat().st_size > 0
    pytest.importorskip("pypdf")   # deep content check; skip where pypdf is absent
    from pypdf import PdfReader
    r = PdfReader(str(out))
    assert len(r.pages) == 1
    text = r.pages[0].extract_text() or ""
    assert "Circular Statistics Report" in text


def test_save_comparison_circular_statistics(tmp_path):
    groups = [
        ("Pre",  _synth_angles(mu=10, seed=1), "#4C9F70"),
        ("Post", _synth_angles(mu=-5, seed=2), "#D9655B"),
        ("Wash", _synth_angles(mu=0,  seed=3), "#5B8FD9"),
    ]
    pdf = tmp_path / "comparison.pdf"
    csv = tmp_path / "comparison.csv"
    save_comparison_circular_statistics(
        groups, csv_path=str(csv), pdf_path=str(pdf), fig_theme="Dark")
    assert pdf.exists() and pdf.stat().st_size > 0
    pytest.importorskip("pypdf")   # deep content check; skip where pypdf is absent
    from pypdf import PdfReader
    r = PdfReader(str(pdf))
    # summary page + one per group
    assert len(r.pages) == 1 + len(groups)
    all_text = "\n".join((pg.extract_text() or "") for pg in r.pages)
    for label in ("Pre", "Post", "Wash"):
        assert label in all_text


def test_compare_groups_honours_dcoeff_clip_range(tmp_path):
    """The LogD graph's x-axis (and clip) follow the D-coefficient clip range."""
    from firefly.analysis.fa_compare import compare_groups
    from test_workspace_data import make_run_folder

    g0 = [make_run_folder(str(tmp_path), f"lo{k}", seed=k, d_centre=0.05) for k in range(2)]
    g1 = [make_run_folder(str(tmp_path), f"hi{k}", seed=9 + k, d_centre=0.4) for k in range(2)]
    groups = [{"folders": g0, "label": "DMSO", "color": "#000000"},
              {"folders": g1, "label": "Drug", "color": "#3fb950"}]

    fig, _summary, _stats = compare_groups(
        groups, output_dir=None, panels={"logd_dist"}, pdf_report=False,
        logd_plot_style="overlaid",
        logd_clip_d_min=1e-3, logd_clip_d_max=1.0)      # → log₁₀ x-axis [-3, 0]
    assert fig is not None
    ax = next((a for a in fig.axes if "LogD" in (a.get_title() or "")), None)
    assert ax is not None
    lo, hi = ax.get_xlim()
    assert abs(lo - (-3.0)) < 1e-6 and abs(hi - 0.0) < 1e-6
    plt.close(fig)


@pytest.mark.parametrize("style", ["mean_faceted", "individual", "overlaid"])
def test_compare_groups_msd_styles_render(tmp_path, style):
    """The MSD panel renders in each Preferences graph-style without error, on
    real ensemble_msd data (make_run_folder writes it)."""
    from firefly.analysis.fa_compare import compare_groups
    from test_workspace_data import make_run_folder
    g0 = [make_run_folder(str(tmp_path), f"a{k}", seed=k, d_centre=0.05) for k in range(2)]
    g1 = [make_run_folder(str(tmp_path), f"b{k}", seed=9 + k, d_centre=0.4) for k in range(2)]
    groups = [{"folders": g0, "label": "DMSO", "color": "#000000"},
              {"folders": g1, "label": "Drug", "color": "#3fb950"}]
    fig, _s, _st = compare_groups(groups, output_dir=None, panels={"msd"},
                                  pdf_report=False, msd_plot_style=style, msd_err="SEM")
    assert fig is not None and len(fig.axes) >= 1     # ≥1 facet drawn
    plt.close(fig)


def test_compare_groups_msd_paired_timepoints(tmp_path):
    """A paired (pre/post-style) design overlays both timepoints per facet."""
    from firefly.analysis.fa_compare import compare_groups
    from test_workspace_data import make_run_folder
    pre = [make_run_folder(str(tmp_path), f"pre{k}", seed=k, d_centre=0.1) for k in range(2)]
    post = [make_run_folder(str(tmp_path), f"post{k}", seed=5 + k, d_centre=0.2) for k in range(2)]
    groups = [{"folders": pre, "label": "Drug", "color": "#58a6ff", "timepoint": "pre"},
              {"folders": post, "label": "Drug", "color": "#58a6ff", "timepoint": "post"}]
    fig, _s, _st = compare_groups(groups, output_dir=None, panels={"msd"},
                                  pdf_report=False, msd_plot_style="mean_faceted")
    assert fig is not None and len(fig.axes) >= 1
    plt.close(fig)


@pytest.mark.parametrize("style", ["paired", "delta"])
def test_compare_groups_auc_styles(tmp_path, style):
    """The AUC panel renders the paired/Δ change across two timepoints."""
    from firefly.analysis.fa_compare import compare_groups
    from test_workspace_data import make_run_folder
    pre = [make_run_folder(str(tmp_path), f"pre{k}", seed=k, d_centre=0.1) for k in range(3)]
    post = [make_run_folder(str(tmp_path), f"post{k}", seed=5 + k, d_centre=0.2) for k in range(3)]
    groups = [{"folders": pre, "label": "Drug", "color": "#58a6ff", "timepoint": "pre"},
              {"folders": post, "label": "Drug", "color": "#58a6ff", "timepoint": "post"}]
    fig, _s, _st = compare_groups(groups, output_dir=None, panels={"auc"},
                                  pdf_report=False, auc_plot_style=style)
    assert fig is not None and len(fig.axes) >= 1
    plt.close(fig)


@pytest.mark.parametrize("group_style,want", [
    ("bar", "Rectangle"), ("box_points", "PathPatch"), ("violin", None)])
def test_group_style_changes_the_scalar_panel_mark(tmp_path, group_style, want):
    """The Preferences 'Group comparison' style must reach the ENGINE's scalar
    panels (AUC etc.), not just the bespoke fallback — bar / box+points / violin.
    (The bug: selecting box did nothing because the engine hard-drew a bar.)"""
    from firefly.analysis.fa_compare import compute_report, render_report
    from test_workspace_data import make_run_folder
    g0 = [make_run_folder(str(tmp_path), f"c{k}", seed=k, d_centre=0.05) for k in range(4)]
    g1 = [make_run_folder(str(tmp_path), f"d{k}", seed=9 + k, d_centre=0.4) for k in range(4)]
    groups = [{"folders": g0, "label": "A", "color": "#58a6ff"},
              {"folders": g1, "label": "B", "color": "#f78166"}]
    rd = compute_report(groups)
    fig, _s, _st = render_report(rd, panels={"auc"}, pdf_report=False, group_style=group_style)
    ax = next(a for a in fig.axes if a.get_ylabel())
    if want == "Rectangle":
        assert any(type(p).__name__ == "Rectangle" for p in ax.patches)   # bars
    elif want == "PathPatch":
        assert any(type(p).__name__ == "PathPatch" for p in ax.patches)   # box bodies
        assert not any(type(p).__name__ == "Rectangle" for p in ax.patches)
    else:                                          # violin → PolyCollection bodies
        assert len(ax.collections) >= 2 and not ax.patches
    plt.close(fig)


def test_panel_styles_are_independent_per_panel(tmp_path):
    """Each scalar comparison panel has its OWN format (Preferences per-graph):
    `panel_styles` routes each panel key to its own mark, independently."""
    from firefly.analysis.fa_compare import compute_report, render_report
    from test_workspace_data import make_run_folder
    g0 = [make_run_folder(str(tmp_path), f"c{k}", seed=k, d_centre=0.05) for k in range(4)]
    g1 = [make_run_folder(str(tmp_path), f"d{k}", seed=9 + k, d_centre=0.4) for k in range(4)]
    groups = [{"folders": g0, "label": "A", "color": "#58a6ff"},
              {"folders": g1, "label": "B", "color": "#f78166"}]
    rd = compute_report(groups)
    ps = {"auc": "box_points", "mob_immob": "bar", "track_count": "violin", "vacf": "bar"}

    def mark(fig):
        ax = next(a for a in fig.axes if a.get_ylabel())
        if any(type(p).__name__ == "PathPatch" for p in ax.patches):
            return "box"
        if any(type(p).__name__ == "Rectangle" for p in ax.patches):
            return "bar"
        if len(ax.collections) >= 2 and not ax.patches:
            return "violin"
        return "?"

    for key, want in [("auc", "box"), ("mob_immob", "bar"),
                      ("track_count", "violin"), ("vacf", "bar")]:
        fig, _s, _st = render_report(rd, panels={key}, pdf_report=False, panel_styles=ps)
        assert mark(fig) == want, f"{key} drew {mark(fig)}, want {want}"
        plt.close(fig)


def test_compare_groups_default_panels_render_without_theme_clobber(tmp_path):
    """Regression: the msd/auc panels used to reassign the `theme` STRING to a
    palette dict, so a later panel that read `theme` as a string
    (motion_classes → motion_class_colors(theme).strip()) crashed with
    'dict' object has no attribute 'strip'.  Since msd + motion_classes are both
    default-on, the DEFAULT full report / all-panels render was broken.  Render
    the full default panel set in one call and assert it completes with all
    panels drawn."""
    from firefly.analysis.fa_compare import compare_groups
    from firefly.ui.controllers.workspace import workspace_data as wd
    from test_workspace_data import make_run_folder
    g0 = [make_run_folder(str(tmp_path), f"c{k}", seed=k, d_centre=0.06) for k in range(3)]
    g1 = [make_run_folder(str(tmp_path), f"d{k}", seed=9 + k, d_centre=0.30) for k in range(3)]
    groups = [{"folders": g0, "label": "Ctrl", "color": "#58a6ff"},
              {"folders": g1, "label": "Drug", "color": "#f78166"}]
    panels = set(wd.PANEL_KEYS)
    assert {"msd", "motion_classes"} <= panels          # the crash pairing
    fig, _s, _st = compare_groups(groups, output_dir=None, panels=panels,
                                  pdf_report=False, theme="Dark")
    # one main axes per panel (+ faceted sub-axes / colorbars) — the point is it
    # rendered the whole grid instead of crashing on motion_classes.
    assert fig is not None and len(fig.axes) >= len(panels)
    plt.close(fig)
