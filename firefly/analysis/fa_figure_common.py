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


def strip_legends(fig):
    """Remove every legend from ``fig`` (minimal figures carry none).

    ``ax.get_legend().remove()`` is not enough: a key drawn with ``ax.legend``
    and then kept with ``ax.add_artist`` (how a second key shares one axes) is
    only detached from the added copy, and the axes still draws it.
    """
    from matplotlib.legend import Legend
    for ax in fig.axes:
        found = {id(c): c for c in ax.get_children() if isinstance(c, Legend)}
        ax.legend_ = None
        for lg in found.values():
            try:
                lg.remove()
            except (NotImplementedError, ValueError):
                lg.set_visible(False)
    for lg in list(fig.legends):
        lg.remove()


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


# ── Pairwise p-value brackets ────────────────────────────────────────────────
# Every comparison panel (the report's and the live Analysis tab's) labels its
# groups the same way: a bracket between each compared pair with that pair's
# p-value over it, stacked above the data so nothing overlaps the points.
BRACKET_ALL_PAIRS_MAX_GROUPS = 4     # up to 6 brackets; beyond, significant pairs only


def select_bracket_pairs(pairs, n_groups, alpha=0.05):
    """Which pairs get a bracket.  `pairs` is ``[(i, j, p)]``.  Untestable pairs
    (p not finite — e.g. a one-replicate group) never do.  Up to
    BRACKET_ALL_PAIRS_MAX_GROUPS groups every testable pair is shown; with more,
    only the significant ones, since 10+ brackets would bury the data."""
    ok = [(int(i), int(j), float(p)) for i, j, p in pairs
          if p is not None and np.isfinite(p)]
    if n_groups > BRACKET_ALL_PAIRS_MAX_GROUPS:
        ok = [t for t in ok if t[2] < alpha]
    return ok


def draw_pvalue_brackets(ax, pairs, *, color, fontsize=8, data_top=None,
                         data_bottom=None):
    """Draw a bracket + label for each ``(x_i, x_j, label)`` above the data and
    extend the y-limit to fit.  Brackets that would overlap go on separate
    levels (shortest spans lowest).  Returns the label Text artists in the order
    given.  Works for positive and signed data; `data_top` / `data_bottom`
    default to the current y-limits."""
    lo0, hi0 = ax.get_ylim()
    top = float(hi0 if data_top is None else data_top)
    bottom = float(lo0 if data_bottom is None else data_bottom)
    rng = (top - bottom) or (abs(top) or 1.0)
    order = sorted(range(len(pairs)), key=lambda k: (abs(pairs[k][1] - pairs[k][0]), min(pairs[k][:2])))
    levels, placed = {}, []                      # placed: (level, a, b)
    for k in order:
        a, b = sorted(pairs[k][:2])
        lvl = 0
        while any(L == lvl and not (b < a2 or a > b2) for L, a2, b2 in placed):
            lvl += 1
        placed.append((lvl, a, b)); levels[k] = lvl
    texts = [None] * len(pairs)
    tick, gap, step = rng * 0.025, rng * 0.06, rng * 0.12
    for k, (xi, xj, label) in enumerate(pairs):
        y = top + gap + levels[k] * step
        a, b = sorted((xi, xj))
        ax.plot([a, a, b, b], [y, y + tick, y + tick, y], color=color, lw=0.8,
                clip_on=False, zorder=5)
        texts[k] = ax.text((a + b) / 2.0, y + tick * 1.15, label, ha="center",
                           va="bottom", fontsize=fontsize, color=color, zorder=5)
    n_levels = (max(levels.values()) + 1) if levels else 0
    if n_levels:
        ax.set_ylim(min(lo0, bottom), top + gap + (n_levels - 1) * step + rng * 0.10)
    return texts


# ── Group curves from per-recording data ────────────────────────────────────
# A distribution curve for a group (log D density, track-length CDF, dwell
# survival, turning-angle histogram) can be drawn two ways:
#   per recording — each recording's curve, averaged; every recording counts
#                   once and the band is the SEM across recordings.  This is
#                   how the van Swinderen lab's sptPALM papers show them
#                   (n = recordings, mean ± s.e.m.), and it matches every
#                   other comparison
#                   panel, where the recording is the unit.  The default.
#   pooled tracks — one curve of every track in the group, so a recording
#                   with more tracks counts for more.  No band.
CURVE_WEIGHTINGS = ("recording", "tracks")


def recording_cells(x):
    """A group's per-recording value arrays (finite, non-empty).  Accepts a
    list of arrays or one pooled array (then treated as a single recording)."""
    if x is None:
        return []
    seq = x if isinstance(x, (list, tuple)) else [x]
    out = []
    for a in seq:
        if a is None:
            continue
        a = np.asarray(a, float).ravel()
        a = a[np.isfinite(a)]
        if len(a):
            out.append(a)
    return out


def group_curve(cells, curve_fn, per_recording=True):
    """``(mean, sem)`` of ``curve_fn(values)`` for one group.

    ``per_recording``: ``curve_fn`` of each recording, averaged, and the SEM
    across recordings (None with fewer than two).  Otherwise ``curve_fn`` of
    every value pooled, with no SEM.  ``curve_fn`` may return None for data it
    cannot draw (e.g. a KDE of one point); such recordings are left out.
    ``(None, None)`` when nothing is drawable."""
    cells = recording_cells(cells)
    if not cells:
        return None, None
    if not per_recording or len(cells) == 1:
        return curve_fn(np.concatenate(cells)), None
    ys = [y for y in (curve_fn(c) for c in cells) if y is not None]
    if not ys:
        return None, None
    Y = np.vstack([np.asarray(y, float) for y in ys])
    sem = Y.std(axis=0, ddof=1) / np.sqrt(len(ys)) if len(ys) >= 2 else None
    return Y.mean(axis=0), sem


def draw_sem_band(ax, x, mean, sem, color, *, alpha=0.25, zorder=2, floor=None):
    """Shade mean ± SEM (nothing when ``sem`` is None); ``floor`` clips the
    lower edge for log axes."""
    if sem is None or mean is None:
        return
    lo, hi = mean - sem, mean + sem
    if floor is not None:
        lo = np.maximum(lo, floor)
        hi = np.maximum(hi, floor)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=zorder)


# ── Diffusive-state diagram ─────────────────────────────────────────────────
# Three circles (immobile top, slow mobile bottom-left, fast mobile bottom-
# right): area ∝ occupancy, each labelled with its apparent D; a loop for the
# probability of staying in a state from one frame to the next and an arrow
# each way for every switch.  The geometry is fixed so no label can collide
# whatever the occupancies: names / D sit radially outside their circle, loops
# on the free outer side, and each switching probability on its own side of
# its arrow pair (tests/test_state_diagram.py renders and checks this).
STATE_DIAGRAM_POS = ((0.0, 0.78), (-0.8, -0.52), (0.8, -0.52))
# loops point sideways (left, left, right) — never towards the name / D labels
# above (immobile) or below (slow, fast) their circle, whatever its size
_LOOP_ANGLE = (np.pi, np.pi, 0.0)
_NAME_ABOVE = (True, False, False)


def state_radius(occupancy):
    """Circle radius (data units) for an occupancy fraction: area ∝ occupancy,
    with a floor so a rare state stays visible."""
    return 0.13 + 0.30 * np.sqrt(max(0.0, float(occupancy)))


def format_probability(p):
    if p is None or not np.isfinite(p):
        return "–"
    return "<0.01" if p < 0.005 else f"{p:.2f}"


def draw_state_diagram(ax, D, occupancy, P, *, names, colors, text_color,
                       background, title=None, title_color=None, fontscale=1.0):
    """Draw one group's diffusive-state diagram into ``ax``.  ``D``,
    ``occupancy``: per state; ``P``: 3×3 per-frame transition probabilities."""
    from matplotlib.patches import Circle, FancyArrowPatch
    pos = np.asarray(STATE_DIAGRAM_POS, float)
    r = [state_radius(o) for o in occupancy]
    fs = lambda base: base * fontscale
    label_box = dict(boxstyle="round,pad=0.12", fc=background, ec="none", alpha=0.9)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.55, 2.0)
    r_max = state_radius(1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    for i in range(3):
        ax.add_patch(Circle(pos[i], r[i], color=colors[i], zorder=2))
        ax.text(*pos[i], f"{100 * occupancy[i]:.0f}%", ha="center", va="center",
                color="white", fontsize=fs(9.5), fontweight="bold", zorder=5)
        # a fixed label row (as if the circle were full size), so a small
        # circle cannot pull its label into the arrows' numbers
        dy = (r_max + 0.07) * (1 if _NAME_ABOVE[i] else -1)
        ax.text(pos[i][0], pos[i][1] + dy, f"{names[i]}\nD = {D[i]:.3f} µm²/s",
                ha="center", va="bottom" if _NAME_ABOVE[i] else "top",
                color=colors[i], fontsize=fs(8), fontweight="bold", linespacing=1.15,
                zorder=5)
        # stay probability: a loop on the circle's free outer side
        ang = _LOOP_ANGLE[i]
        a0 = pos[i] + r[i] * np.array([np.cos(ang - 0.42), np.sin(ang - 0.42)])
        a1 = pos[i] + r[i] * np.array([np.cos(ang + 0.42), np.sin(ang + 0.42)])
        ax.add_patch(FancyArrowPatch(a0, a1, connectionstyle="arc3,rad=1.9",
                                     arrowstyle="-|>", mutation_scale=9 * fontscale,
                                     color=colors[i], lw=1.3, shrinkA=0, shrinkB=0, zorder=3))
        lp = pos[i] + (r[i] + 0.38) * np.array([np.cos(ang), np.sin(ang)])
        ax.text(*lp, format_probability(P[i][i]), ha="center", va="center",
                color=text_color, fontsize=fs(8), zorder=6, bbox=label_box)
    for i in range(3):                           # one arrow each way per pair
        for j in range(3):
            if i == j:
                continue
            u = (pos[j] - pos[i]) / np.linalg.norm(pos[j] - pos[i])
            n = np.array([-u[1], u[0]])
            p0 = pos[i] + u * (r[i] + 0.04) + n * 0.065
            p1 = pos[j] - u * (r[j] + 0.04) + n * 0.065
            ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=9 * fontscale,
                                         color=colors[i], lw=1.2, shrinkA=0, shrinkB=0, zorder=3))
            # this arrow's number: on its own side (n), a third of the way from
            # its source, so a pair's two numbers part along the edge as well
            at = p0 + 0.36 * (p1 - p0)
            ax.text(*(at + n * 0.17), format_probability(P[i][j]),
                    ha="center", va="center", color=text_color, fontsize=fs(7.5),
                    zorder=6, bbox=label_box)
    if title:
        ax.text(0.5, 0.995, title, transform=ax.transAxes, ha="center", va="top",
                color=title_color or text_color, fontsize=fs(9.5), fontweight="bold",
                zorder=6)
    ax._firefly_state_diagram_scale = fontscale


STATE_DIAGRAM_SIDE_IN = 3.35      # the side the base text sizes are set for


def state_diagram_fontscale(side_in):
    """Text scale for a diagram ``side_in`` inches square (tested down to 0.6)."""
    return float(np.clip(side_in / STATE_DIAGRAM_SIDE_IN, 0.6, 1.0))


def rescale_state_diagrams(fig):
    """After layout: size every state diagram's text (and arrow heads) for the
    square it actually got — the size is only known once tight_layout ran."""
    W, H = fig.get_size_inches()
    for ax in fig.axes:
        old = getattr(ax, "_firefly_state_diagram_scale", None)
        if old is None:
            continue
        p = ax.get_position()
        new = state_diagram_fontscale(min(p.width * W, p.height * H))
        k = new / old
        if abs(k - 1) < 1e-3:
            continue
        for t in ax.texts:
            t.set_fontsize(t.get_fontsize() * k)
        for patch in ax.patches:
            if hasattr(patch, "set_mutation_scale"):
                patch.set_mutation_scale(patch.get_mutation_scale() * k)
        ax._firefly_state_diagram_scale = new
