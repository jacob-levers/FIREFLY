"""Shared figure/PDF rendering helpers (#3 de-dup).

These were byte-identical (or parameter-only-different) blocks copy-pasted
between the per-file and comparison circular-statistics PDF renderers in
`fa_circular.py` and the comparison renderer in `fa_compare.py`.  Extracting
them means a fix (a colour, a number format, a footer line) is made ONCE.

Every function here is a pure *move* of existing drawing code — the rendered
output is byte-for-byte identical (verified by content-stream + extracted-text
fingerprint of the rendered PDFs).  Do NOT add behaviour here that the original
inline blocks didn't have.

Qt-free; matplotlib is imported lazily inside the drawing helpers so importing
this module stays cheap.
"""
from __future__ import annotations

import contextlib

import numpy as np


# rcParams keys the PDF renderers force onto the theme palette.  Shared so the
# snapshot list and the restore can't drift apart.
_THEME_RC_KEYS = (
    "text.color", "axes.labelcolor", "axes.edgecolor",
    "xtick.color", "ytick.color", "axes.facecolor",
    "axes.titlecolor", "figure.facecolor", "grid.color",
    "font.family",
)


@contextlib.contextmanager
def rcparams_for_theme(plt, pal):
    """Temporarily force matplotlib's global rcParams onto the FIREFLY theme
    palette ``pal``, restoring the previous values on exit (even on exception).

    ``plt.rcParams`` persists across figures in the same process — the master
    figure renderer might have left ``text.color`` etc. on the Dark palette.
    This snapshots the affected keys, forces everything to OUR palette so the
    PDF can't accidentally pick up someone else's colours, then restores on
    exit so we don't bleed our palette into whatever the caller draws next.

    Exact move of the snapshot / update / restore try-finally duplicated in the
    fa_circular.py PDF renderers (and fa_compare.py).  ``plt`` is passed in so
    this module needn't import pyplot at load time.
    """
    save = {k: plt.rcParams.get(k) for k in _THEME_RC_KEYS}
    plt.rcParams.update({
        "text.color":       pal["TXT"],
        "axes.labelcolor":  pal["TXT"],
        "axes.edgecolor":   pal["GRD"],
        "xtick.color":      pal["TXT"],
        "ytick.color":      pal["TXT"],
        "axes.facecolor":   pal["PNL"],
        "axes.titlecolor":  pal["TXT"],
        "figure.facecolor": pal["BG"],
        "grid.color":       pal["GRD"],
        "font.family":      pal["FONT"],
    })
    try:
        yield
    finally:
        plt.rcParams.update(save)


def fmt_stat_value(x, prec=4):
    """Format a statistic for a PDF table cell.

    NaN / None render as an em-dash; the 1e-300 underflow sentinel produced by
    the log-space p-value computations collapses to a human-readable "<1e-300"
    (otherwise the reader sees a literal "1e-300" repeated across rows and
    assumes a bug); everything else uses ``f"{x:.{prec}g}"``.

    Exact move of the `_fmt` closure duplicated 3× in fa_circular.py.
    """
    try:
        if x is None:
            return "—"
        xf = float(x)
        if np.isnan(xf):
            return "—"
        if xf > 0.0 and xf <= 1e-300:
            return "<1e-300"
        return f"{xf:.{prec}g}"
    except Exception:
        return str(x)


def style_table_cells(tbl, pal, *, fontsize, label_col=False, pad=None):
    """Apply the FIREFLY circular-PDF table style to a matplotlib table:
    0.5-pt grid in the theme grid colour, a dark header row (HDR_BG fill +
    bold HDR_TXT), and zebra-striped data rows (ZEBRA on even rows, PNL on
    odd).

    ``label_col=True`` renders the row-label pseudo-column (cell column
    ``-1``) in muted monospaced 8-pt — used by the per-statistic tables whose
    left column is a stat name.  ``pad`` (when given) sets each cell's ``PAD``
    for extra in-cell breathing room.

    Exact move of the cell-styling loop duplicated 4× in fa_circular.py; the
    differences between those copies were only ``fontsize`` (9.0 / 8.5),
    whether the row-label column got special treatment, and the ``PAD`` nudge.
    """
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor(pal["GRD"])
        if pad is not None:
            cell.PAD = pad
        if r == 0:                       # column header row
            cell.set_facecolor(pal["HDR_BG"])
            cell.set_text_props(color=pal["HDR_TXT"], fontweight="bold")
        else:
            # Zebra-stripe data rows: ZEBRA for the lighter stripes, PNL for
            # the darker ones.
            cell.set_facecolor(pal["ZEBRA"] if r % 2 == 0 else pal["PNL"])
            if label_col and c == -1:    # row-label column
                cell.set_text_props(family="monospace", fontsize=8.0,
                                    color=pal["MUT"])
            else:
                cell.set_text_props(color=pal["TXT"])


def render_polar_histogram(ax, a, stats, pal, *, color=None, tick_fontsize=8):
    """Draw the signed-turning-angle polar histogram into a polar ``ax``.

    Matches the master figure's Radial-Distribution convention: 0° at the top,
    positive angles sweeping CLOCKWISE (so +θ lands on the right hemisphere),
    signed-angle labels at the eight slot positions, and a μ-direction arrow at
    the mean resultant direction.  Signed angles on (−180°, +180°] are wrapped
    into [0, 2π) before histogramming because matplotlib's polar ``bar()``
    silently drops bars at negative θ once ``set_theta_direction(-1)`` is active.

    The CALLER owns the axes (position + ``set_facecolor``), the ``a.size >= 10``
    guard, and the empty-data fallback — this is the exact inner drawing block
    that was duplicated 3× in fa_circular.py.  Its copies differed only in the
    bar ``color`` (group tint vs the ACC default) and the tick ``fontsize``.

    Parameters
    ----------
    ax    : a polar matplotlib Axes (already positioned + facecolour set).
    a     : 1-D array of signed turning angles in degrees (already finite).
    stats : dict from compute_circular_statistics — ``mean_direction_deg`` is
            read for the μ arrow.
    pal   : theme palette dict.
    color : bar fill; falls back to ``pal["ACC"]`` when None (the per-file page).
    tick_fontsize : font size for the angle tick labels (8 on the single-group
            pages, smaller on the dense per-group grid).
    """
    nbins = 36
    angles_rad = np.mod(np.deg2rad(a), 2.0 * np.pi)
    bins  = np.linspace(0.0, 2.0 * np.pi, nbins + 1)
    counts, edges = np.histogram(angles_rad, bins=bins)
    widths  = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # CW positive — match master fig
    ax.bar(centers, counts, width=widths * 0.95,
           align="center", color=color or pal["ACC"],
           edgecolor=pal["PNL"], linewidth=0.4, alpha=0.92)
    mu = stats.get("mean_direction_deg")
    if mu is not None and not (isinstance(mu, float) and np.isnan(mu)):
        r_max = float(counts.max()) if counts.size else 1.0
        # Wrap signed μ into [0, 2π) so the arrow lands where the bars do.
        mu_rad = np.mod(np.deg2rad(mu), 2.0 * np.pi)
        ax.annotate("",
            xy=(mu_rad, r_max * 0.95), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=pal["ARROW"], lw=2.0))
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(
        ["0°", "+45°", "+90°", "+135°", "±180°", "−135°", "−90°", "−45°"],
        fontsize=tick_fontsize)
    ax.set_yticklabels([])
    ax.tick_params(colors=pal["TXT"], labelsize=tick_fontsize)
    ax.grid(True, ls=":", alpha=0.4)
