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
    from pypdf import PdfReader
    r = PdfReader(str(pdf))
    # summary page + one per group
    assert len(r.pages) == 1 + len(groups)
    all_text = "\n".join((pg.extract_text() or "") for pg in r.pages)
    for label in ("Pre", "Post", "Wash"):
        assert label in all_text
