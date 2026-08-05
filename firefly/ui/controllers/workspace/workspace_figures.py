"""Matplotlib → QImage renderers for the Analysis workspace (Qt-light).

These draw the *live* comparison picture for one metric across conditions.  They
use matplotlib's object-oriented Agg API (no pyplot, no global state), so a
worker thread can build its own ``Figure`` and hand back a detached ``QImage``
without touching the GUI thread — that's what keeps the figure lane async while
the numbers update instantly.

The statistical content (p-values, effect sizes) lives in the dedicated
significance card; the figure's job is the *distribution picture* — violin / box
/ ECDF / histogram of the pooled per-track values per condition, with SuperPlot
per-folder replicate dots and a mean ± error overlay.  Metrics that only expose
a per-folder scalar (no per-track distribution) fall back to a bar + replicate
dots, which is the honest visualisation for a replicate-level quantity.
"""
from __future__ import annotations

import numpy as np

from firefly.analysis.fa_constants import MOBILE_D_THRESHOLD_DEFAULT
from . import workspace_data as _wd
from .workspace_data import Metric, MOTION_CLASSES, MOTION_COLORS

# design mat / ink
_MAT = "#0a0d12"
_INK = "#e6edf3"
_MUTED = "#8b949e"
_FAINT = "#5b636e"
_GRID = "#1d232c"

# ── Combined-figure panel geometry ────────────────────────────────────────────
# The saved single-run figure (<stem>_sptpalm_figure.png) is fa_figure's full
# 6×3 grid: figsize (20, 38), GridSpec(6, 3, hspace=0.45, wspace=0.32,
# left=0.06, right=0.97, top=0.95, bottom=0.035).  We can therefore crop the
# exact cell of any panel out of it — so a spatial panel that didn't export its
# own per-panel PNG still shows JUST that graph (never the whole grid).  Cells
# below are (x0, y_bottom, x1, y_top) in figure fractions (y is bottom-up).
def _combined_cells():
    left, right, top, bottom = 0.06, 0.97, 0.95, 0.035
    nr, nc, hspace, wspace = 6, 3, 0.45, 0.32
    cw = (right - left) / (nc + (nc - 1) * wspace)
    ch = (top - bottom) / (nr + (nr - 1) * hspace)
    xs = [left + c * (cw + wspace * cw) for c in range(nc)]          # col left edges
    ytops = [top - r * (ch + hspace * ch) for r in range(nr)]        # row top edges
    # only the spatial panels are cropped (the rest are group-averaged)
    pos = {"A": (0, 0), "B": (0, 1), "C": (0, 2), "H": (2, 1), "L": (4, 0)}
    return {l: (xs[c], ytops[r] - ch, xs[c] + cw, ytops[r])
            for l, (r, c) in pos.items()}

_COMBINED_CELL = _combined_cells()
_COMBINED_ASPECT = 20.0 / 38.0                 # full-grid figure aspect (W/H)
# The figure is saved with bbox_inches="tight", which trims the outer margins
# and so slightly compresses the raw canvas fractions toward the centre.  A
# small linear correction about the grid's vertical centre (calibrated against a
# rendered reference: top-row titles were riding high, bottom-row panels sat low)
# realigns every row; the middle row is the pivot and is already exact.
_CROP_Y_PIVOT = 0.573
_CROP_Y_GAIN = 0.08
# Padding around each cell to keep its title/letter + axis labels.
_CROP_PAD = {"top": 0.024, "left": 0.048, "bottom": 0.024, "right": 0.020}
_COMBINED_QIMG_CACHE: dict = {}                # path → decoded QImage (small LRU)


def _crop_combined_panel(path, letter):
    """Crop panel `letter`'s cell out of a saved combined single-run figure.
    Returns a detached QImage, or None if the file isn't the full 6×3 grid (a
    user-selected panel subset has a different geometry → don't guess)."""
    from PySide6.QtGui import QImage
    from PySide6.QtCore import QRect
    box = _COMBINED_CELL.get(letter)
    if box is None:
        return None
    img = _COMBINED_QIMG_CACHE.get(path)
    if img is None:
        img = QImage(path)
        if img.isNull():
            return None
        if len(_COMBINED_QIMG_CACHE) > 3:      # tiny cache: avoid re-decoding the big PNG
            _COMBINED_QIMG_CACHE.clear()
        _COMBINED_QIMG_CACHE[path] = img
    W, H = img.width(), img.height()
    if W <= 0 or H <= 0 or abs(W / H - _COMBINED_ASPECT) > 0.04:
        return None                            # not the full grid → geometry unknown
    x0, yb, x1, yt = box
    # correct the bbox_inches="tight" compression about the grid's vertical centre
    yt += _CROP_Y_GAIN * (yt - _CROP_Y_PIVOT)
    yb += _CROP_Y_GAIN * (yb - _CROP_Y_PIVOT)
    p = _CROP_PAD
    left = int(max(0.0, x0 - p["left"]) * W)
    right = int(min(1.0, x1 + p["right"]) * W)
    # figure fraction y is bottom-up; image pixels are top-down
    ptop = int(max(0.0, 1.0 - (yt + p["top"])) * H)
    pbot = int(min(1.0, 1.0 - (yb - p["bottom"])) * H)
    if right <= left or pbot <= ptop:
        return None
    return img.copy(QRect(left, ptop, right - left, pbot - ptop))


def _robust_limits(vals, positive=False, logd=False, k=3.0):
    """Tukey-fence display limits so a heavy tail of near-zero / failed-fit
    values (e.g. stuck-particle D ≈ 1e-25, often several % of tracks) can't blow
    the axis out to 24 decades.  Fences are computed in log space for log
    metrics, so they reject the junk tail even at ~10–25% contamination (a fixed
    0.5-percentile trim does not).  Returns (lo, hi) or None."""
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if positive or logd:
        v = v[v > 0]
    if v.size < 5:
        return None
    w = np.log10(v) if logd else v
    q25, q75 = np.percentile(w, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        lo, hi = np.percentile(w, [1, 99])
    else:
        lo = max(float(w.min()), q25 - k * iqr)
        hi = min(float(w.max()), q75 + k * iqr)
    if logd:
        lo, hi = 10.0 ** lo, 10.0 ** hi
    return (lo, hi) if hi > lo else None


def _to_log(d):
    """log10 of the finite, positive entries.  Log metrics (Diffusion D) are drawn
    as log10(value) on a LINEAR axis — exactly how the engine's LogD distribution
    renders — rather than the raw value on a matplotlib log-scaled axis."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    return np.log10(d[d > 0])


def _safe_linear_bins(values, n):
    """Strictly increasing finite bin edges for constant/zero populations."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return int(n)
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        pad = max(abs(lo) * 0.05, 1e-12)
        lo = max(0.0, lo - pad) if lo >= 0 else lo - pad
        hi = hi + pad
        if hi <= lo:
            hi = lo + pad
    return np.linspace(lo, hi, max(int(n), 2))


def _safe_log_bins(values, n):
    """Strictly increasing logarithmic edges for finite positive values."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if not v.size:
        return int(n)
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        factor = 10.0 ** 0.05
        lo, hi = lo / factor, hi * factor
    return np.logspace(np.log10(lo), np.log10(hi), max(int(n), 2))


def _qimage_from_figure(fig):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PySide6.QtGui import QImage
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = canvas.buffer_rgba()
    img = QImage(bytes(buf), w, h, QImage.Format.Format_RGBA8888).copy()
    return img


def _new_axes(width_px, height_px, dpi):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(max(width_px, 80) / dpi, max(height_px, 80) / dpi),
                 dpi=dpi, facecolor=_MAT)
    ax = fig.add_subplot(111)
    ax.set_facecolor(_MAT)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=8, length=3)
    ax.title.set_color(_INK)
    ax.xaxis.label.set_color(_MUTED)
    ax.yaxis.label.set_color(_MUTED)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.18)
    return fig, ax


def _render_logd_engine(groups, style, mobile_d, width_px, height_px, dpi,
                        logd_clip=(0.00001, 10.0)):
    """Render the Diffusion-D metric with the REAL engine LogD renderer so the
    live preview is exactly the report's ``logd_dist`` panel (KDE distribution),
    not a bespoke ECDF.  Built on the live data; drawn into the live dark axes."""
    from firefly.analysis.fa_compare import (
        _render_logd_overlaid, _render_logd_ridgeline, _render_logd_violin)
    from firefly.analysis.fa_theme import _theme_palette
    if style not in ("overlaid", "ridgeline", "violin"):
        style = "overlaid"          # faceted needs facet data → overlaid live
    pal = dict(_theme_palette("Dark"))
    pal["GRD"] = _GRID              # threshold line tint matches the live grid
    # Match the engine EXACTLY: log10 then CLIP to the D-coefficient range, so
    # immobile particles pile at the floor instead of smearing the KDE.
    _lo = float(np.log10(logd_clip[0])) if logd_clip[0] and logd_clip[0] > 0 else -5.0
    _hi = float(np.log10(logd_clip[1])) if logd_clip[1] and logd_clip[1] > 0 else 1.0
    def _clip_log(arr):
        a = np.asarray(arr, float); a = a[np.isfinite(a)]; a = a[a > 0]
        return np.clip(np.log10(a), _lo, _hi) if a.size else None
    per_card = []
    for g in groups:
        pooled = _clip_log(g["dist"]) if g.get("dist") is not None else None
        medians = _clip_log(g.get("values"))
        per_card.append((g["label"], g["color"], pooled, medians))
    thr = float(np.log10(mobile_d if mobile_d and mobile_d > 0
                        else MOBILE_D_THRESHOLD_DEFAULT))
    fig, ax = _new_axes(width_px, height_px, dpi)
    fn = {"ridgeline": _render_logd_ridgeline, "violin": _render_logd_violin}.get(
        style, _render_logd_overlaid)
    fn(ax, per_card, thr, pal, mobile_d or MOBILE_D_THRESHOLD_DEFAULT, xlim=(_lo, _hi))
    # keep the live dark theme: re-tint labels/title/legend the engine left default
    ax.title.set_color(_INK)
    ax.xaxis.label.set_color(_MUTED); ax.yaxis.label.set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelsize=8)
    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_color(_MUTED)
    fig.tight_layout(pad=1.1)
    return _qimage_from_figure(fig)


def _gf_theme():
    return {"bg": _MAT, "fg": _INK, "grid": _GRID, "spine": _GRID, "muted": _MUTED}


def _render_group_comparison(groups, metric, style, err, width_px, height_px, dpi,
                             grouped_data=None):
    """Scalar-metric group comparison via the shared renderer (Preferences
    figures/group_style): box+points / grouped / violin / bar, with a KW label.

    For ``grouped`` we prefer ``grouped_data`` (per condition-NAME × timepoint,
    the "between-dish variability" split); it falls back to one series per group.
    """
    from matplotlib.figure import Figure
    from firefly.analysis import fa_group_figures as _gf
    order, values, gcolors = [], {}, {}
    tp_order = None
    if style == "grouped" and grouped_data and grouped_data.get("data"):
        gd = grouped_data
        tp_order = gd.get("phases") or None
        for nm in gd["names"]:
            tps = {ph: np.asarray(v, float)[np.isfinite(np.asarray(v, float))]
                   for ph, v in gd["data"].get(nm, {}).items()}
            tps = {ph: v for ph, v in tps.items() if len(v)}
            if tps:
                order.append(nm); values[nm] = tps
                gcolors[nm] = gd.get("colors", {}).get(nm)
    else:
        for g in groups:
            v = np.asarray(g.get("values", []), float); v = v[np.isfinite(v)]
            if not len(v):
                continue
            order.append(g["label"]); values[g["label"]] = {"": v}
            gcolors[g["label"]] = g.get("color")
    stat_label = ""
    arrs = [np.concatenate(list(values[l].values())) for l in order if values[l]]
    if len(arrs) >= 2 and all(len(a) >= 1 for a in arrs):
        try:
            from scipy.stats import kruskal
            stat_label = f"Kruskal–Wallis, p = {kruskal(*arrs).pvalue:.3g}"
        except Exception:
            pass
    fig = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=_MAT)
    _gf.draw_group_comparison(fig, fig.add_gridspec(1, 1)[0], order, values, style=style,
                              stat_label=stat_label, group_colors=gcolors, tp_order=tp_order,
                              theme=_gf_theme(), ylabel=metric.axis, err=err)
    fig.tight_layout(pad=1.1)
    return _qimage_from_figure(fig)


def _render_length_density(groups, metric, width_px, height_px, dpi):
    """Overlaid per-group track-length density (Preferences figures/length_style)."""
    from matplotlib.figure import Figure
    from firefly.analysis import fa_group_figures as _gf
    order, dists, gcolors = [], {}, {}
    for g in groups:
        d = g.get("dist")
        if d is None or not len(d):
            continue
        order.append(g["label"]); dists[g["label"]] = np.asarray(d, float)
        gcolors[g["label"]] = g.get("color")
    if not order:
        return None
    fig = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, facecolor=_MAT)
    _gf.draw_length_density(fig, fig.add_gridspec(1, 1)[0], order, dists,
                            group_colors=gcolors, theme=_gf_theme(), xlabel=metric.axis)
    fig.tight_layout(pad=1.1)
    return _qimage_from_figure(fig)


def render_metric(groups: list[dict], metric: Metric, *, plot: str = "Violin",
                  err: str = "95% CI", log_x: bool = False,
                  width_px: int = 720, height_px: int = 380, dpi: int = 100,
                  logd_style: str = "overlaid",
                  mobile_d: float = MOBILE_D_THRESHOLD_DEFAULT,
                  logd_clip: tuple = (0.00001, 10.0),
                  group_style: str = "box_points", length_style: str = "density",
                  grouped_data=None):
    """Render one metric across ``groups`` → detached ``QImage``.

    ``groups``: ``[{"label", "color", "values": ndarray(per-folder scalars),
    "dist": ndarray|None (pooled per-track)}]``.
    """
    if metric.id == "motion":
        return _render_motion(groups, width_px, height_px, dpi)
    # Diffusion D → the engine's own LogD-distribution panel, so the live preview
    # is exactly what the full report exports (KDE, not a bespoke ECDF).
    if metric.id == "D" and any(g.get("dist") is not None and len(g["dist"]) for g in groups):
        try:
            return _render_logd_engine(groups, logd_style, mobile_d,
                                       width_px, height_px, dpi, logd_clip=logd_clip)
        except Exception:
            pass   # fall through to the generic renderer if the engine path fails

    has_dist = any(g.get("dist") is not None and len(g["dist"]) for g in groups)

    # Track-length distribution → overlaid density (Preferences figures/length_style).
    if metric.id == "len" and length_style == "density" and has_dist:
        try:
            img = _render_length_density(groups, metric, width_px, height_px, dpi)
            if img is not None:
                return img
        except Exception:
            pass

    # Scalar-metric group comparison → the styled renderer (figures/group_style).
    if group_style in ("box_points", "grouped", "violin", "bar"):
        try:
            return _render_group_comparison(groups, metric, group_style, err,
                                            width_px, height_px, dpi,
                                            grouped_data=grouped_data)
        except Exception:
            pass   # fall through to the legacy renderer
    fig, ax = _new_axes(width_px, height_px, dpi)
    title = f"{metric.label}" + (f"  ({metric.unit})" if metric.unit else "")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    if has_dist and plot in ("Violin", "Box", "ECDF", "Histogram"):
        if plot == "ECDF":
            _draw_ecdf(ax, groups, metric, log_x)
        elif plot == "Histogram":
            _draw_hist(ax, groups, metric, log_x)
        else:
            _draw_violin_or_box(ax, groups, metric, err, log_x, box=(plot == "Box"))
    else:
        _draw_bars(ax, groups, metric, err)

    fig.tight_layout(pad=1.1)
    return _qimage_from_figure(fig)


def _err_value(vals: np.ndarray, err: str) -> float:
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return 0.0
    sd = float(np.std(vals, ddof=1))
    if err == "SD":
        return sd
    sem = sd / np.sqrt(vals.size)
    if err == "SEM":
        return sem
    return 1.96 * sem          # 95% CI


def _draw_violin_or_box(ax, groups, metric, err, log_x, *, box):
    positions = np.arange(len(groups)) + 1
    logd = log_x and metric.log_default

    def prep(d):
        d = np.asarray(d, float)
        return _to_log(d) if logd else d[np.isfinite(d)]

    raw = [prep(g.get("dist") if g.get("dist") is not None else g["values"])
           for g in groups]
    alld = (np.concatenate([d for d in raw if len(d)])
            if any(len(d) for d in raw) else np.array([]))
    rng = _robust_limits(alld)                       # alld already log10 when logd
    # clip the *picture* to a robust range (the stats use medians, unaffected);
    # fall back to the unclipped data if a group would be left with < 2 points.
    dists = []
    for d in raw:
        if rng is not None and len(d):
            dc = d[(d >= rng[0]) & (d <= rng[1])]
            dists.append(dc if len(dc) >= 2 else d)
        else:
            dists.append(d if len(d) >= 2 else np.array([0.0, 0.0]))
    colors = [g["color"] for g in groups]
    if box:
        bp = ax.boxplot(dists, positions=positions, widths=0.55, patch_artist=True,
                        showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.30); patch.set_edgecolor(c)
        for key in ("whiskers", "caps", "medians"):
            for ln in bp[key]:
                ln.set_color(_MUTED)
    else:
        parts = ax.violinplot(dists, positions=positions, widths=0.8,
                              showmeans=False, showextrema=False)
        for body, c in zip(parts["bodies"], colors):
            body.set_facecolor(c); body.set_alpha(0.28); body.set_edgecolor(c)
            body.set_linewidth(1.0)
    # SuperPlot: per-folder replicate dots + mean ± error
    for i, g in enumerate(groups):
        vals = prep(g["values"])                     # log10 too, so dots match the violins
        if not vals.size:
            continue
        jit = (np.random.default_rng(i + 1).random(vals.size) - 0.5) * 0.18
        ax.scatter(positions[i] + jit, vals, s=26, color=g["color"],
                   edgecolor=_MAT, linewidth=0.6, zorder=4)
        m = float(np.mean(vals)); e = _err_value(vals, err)
        ax.errorbar(positions[i], m, yerr=e, fmt="_", color=_INK, ecolor=_INK,
                    elinewidth=1.2, capsize=4, markersize=14, zorder=5)
    ax.set_xticks(positions)
    ax.set_xticklabels([g["label"] for g in groups], fontsize=8, color=_MUTED,
                       rotation=0)
    ax.set_ylabel(metric.axis if logd
                  else metric.label + (f" ({metric.unit})" if metric.unit else ""))
    if logd and rng:
        pad = 0.08 * (rng[1] - rng[0])
        ax.set_ylim(rng[0] - pad, rng[1] + pad)
    ax.grid(axis="y", color=_GRID, linewidth=0.6, alpha=0.7)


def _draw_ecdf(ax, groups, metric, log_x):
    logd = log_x and metric.log_default
    alld = []
    for g in groups:
        d = np.asarray(g.get("dist"), dtype=float)
        d = _to_log(d) if logd else d[np.isfinite(d)]
        d = np.sort(d)
        if not d.size:
            continue
        y = np.arange(1, d.size + 1) / d.size
        ax.plot(d, y, color=g["color"], linewidth=1.8, label=g["label"])
        alld.append(d)
    ax.set_xlabel(metric.axis)
    ax.set_ylabel("cumulative fraction")
    # data is already log10 when logd → plain (linear-space) robust fences
    rng = _robust_limits(np.concatenate(alld)) if alld else None
    if rng:
        ax.set_xlim(*rng)
    ax.grid(color=_GRID, linewidth=0.6, alpha=0.6)
    ax.legend(loc="lower right", fontsize=7, frameon=False, labelcolor=_MUTED)


def _draw_hist(ax, groups, metric, log_x):
    logd = log_x and metric.log_default

    def prep(d):
        d = np.asarray(d, float)
        return _to_log(d) if logd else d[np.isfinite(d)]

    alld = (np.concatenate([prep(g["dist"]) for g in groups if g.get("dist") is not None])
            if any(g.get("dist") is not None for g in groups) else np.array([]))
    rng = _robust_limits(alld)                       # alld already log10 when logd
    if alld.size:
        lo, hi = rng if rng else (float(alld.min()), float(alld.max()))
        bins = _safe_linear_bins([lo, hi], 40)
    else:
        bins = 30
    for g in groups:
        d = prep(g.get("dist", np.array([])))
        if not d.size:
            continue
        ax.hist(d, bins=bins, histtype="step", color=g["color"], linewidth=1.6,
                label=g["label"], density=True)
    ax.set_xlabel(metric.axis)
    ax.set_ylabel("density")
    if rng:
        ax.set_xlim(*rng)
    ax.grid(color=_GRID, linewidth=0.6, alpha=0.6)
    ax.legend(loc="upper right", fontsize=7, frameon=False, labelcolor=_MUTED)


def _draw_bars(ax, groups, metric, err):
    positions = np.arange(len(groups)) + 1
    for i, g in enumerate(groups):
        vals = np.asarray(g["values"], dtype=float); vals = vals[np.isfinite(vals)]
        if not vals.size:
            continue
        m = float(np.mean(vals)); e = _err_value(vals, err)
        ax.bar(positions[i], m, width=0.6, color=g["color"], alpha=0.30,
               edgecolor=g["color"], linewidth=1.2, zorder=2)
        ax.errorbar(positions[i], m, yerr=e, fmt="none", ecolor=_INK,
                    elinewidth=1.2, capsize=4, zorder=4)
        jit = (np.random.default_rng(i + 1).random(vals.size) - 0.5) * 0.22
        ax.scatter(positions[i] + jit, vals, s=26, color=g["color"],
                   edgecolor=_MAT, linewidth=0.6, zorder=5)
    ax.set_xticks(positions)
    ax.set_xticklabels([g["label"] for g in groups], fontsize=8, color=_MUTED)
    ax.set_ylabel(metric.label + (f" ({metric.unit})" if metric.unit else ""))
    ax.grid(axis="y", color=_GRID, linewidth=0.6, alpha=0.7)


def render_panel(panel: dict, runs, color="#58a6ff", *,
                 width_px: int = 720, height_px: int = 380, dpi: int = 100):
    """Render one per-condition publication panel from the condition's pooled
    run folders → detached ``QImage``.  See ``workspace_data.PANELS``."""
    kind = panel.get("kind")
    if kind == "raster":
        return _render_raster(panel, runs, width_px, height_px, dpi)
    if kind == "msd":
        return _render_msd(runs, width_px, height_px, dpi)
    if kind == "motion":
        return _render_motion_single(runs, width_px, height_px, dpi)

    # distribution panels (metric / column)
    fig, ax = _new_axes(width_px, height_px, dpi)
    ax.set_title(panel["name"], fontsize=11, fontweight="bold", pad=10)
    scalar_only = False
    if kind == "metric":
        m = _wd.METRIC_BY_ID.get(panel["ref"])
        vals = None
        if m is not None:
            chunks = [m.dist(r) for r in runs]
            chunks = [c for c in chunks if c is not None and len(c)]
            if chunks:
                vals = np.concatenate(chunks)
            else:                          # scalar-only metric → per-folder values
                sc = [m.scalar(r) for r in runs]
                vals = np.array([v for v in sc if v is not None], dtype=float)
                scalar_only = True
            axis = m.axis; logd = m.log_default
        else:
            axis = panel["name"]; logd = False
    else:                                   # 'col'
        vals = _wd.pooled_column(runs, panel["col"])
        axis = panel.get("label", panel["name"]) + (f" ({panel['unit']})" if panel.get("unit") else "")
        logd = False
    if scalar_only:
        _draw_dots_mean_sem(ax, vals, color, axis)
    else:
        _draw_single_dist(ax, vals, color, axis, logd)
    fig.tight_layout(pad=1.1)
    return _qimage_from_figure(fig)


def _draw_dots_mean_sem(ax, vals, color, axis):
    """Scalar-per-folder metrics (mobile fraction, α₂, confinement): one dot per
    replicate folder with the group mean ± SEM — the honest 'group average'."""
    vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
    if not vals.size:
        ax.text(0.5, 0.5, "no data for this panel", ha="center", va="center",
                transform=ax.transAxes, color=_FAINT, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        return
    n = vals.size
    rng = np.random.default_rng(0)
    xj = 1.0 + (rng.random(n) - 0.5) * 0.22
    ax.scatter(xj, vals, s=70, color=color, alpha=0.85, edgecolor=_MAT,
               linewidth=0.8, zorder=3)
    mean = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    ax.errorbar([1.0], [mean], yerr=[sem], fmt="_", color=color, markersize=46,
                capsize=9, elinewidth=2.2, markeredgewidth=2.6, zorder=4)
    ax.set_xlim(0.5, 1.5)
    ax.set_xticks([1.0]); ax.set_xticklabels([f"n = {n}"])
    ax.set_ylabel(axis, fontsize=9)
    ax.text(0.98, 0.97, f"mean {mean:.4g} ± {sem:.2g}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=_MUTED)
    ax.grid(True, axis="y", ls="--", alpha=0.22, lw=0.5)


def _draw_single_dist(ax, vals, color, axis, logd):
    if vals is None or not len(vals):
        ax.text(0.5, 0.5, "no data for this panel", ha="center", va="center",
                transform=ax.transAxes, color=_FAINT, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        return
    vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
    med = float(np.median(vals)) if vals.size else 0.0
    pos = vals[vals > 0]
    rng = _robust_limits(pos if (logd and pos.size > 5) else vals, logd=logd)
    if logd:
        if not pos.size:
            ax.text(0.5, 0.5, "no positive fitted values", ha="center",
                    va="center", transform=ax.transAxes, color=_FAINT,
                    fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            return
        ax.set_xscale("log"); vals = pos
        lo, hi = rng if rng else (float(pos.min()), float(pos.max()))
        bins = _safe_log_bins([lo, hi], 44)
        med = float(np.median(pos))
    else:
        lo, hi = rng if rng else (
            (float(vals.min()), float(vals.max())) if vals.size else (0.0, 1.0)
        )
        bins = _safe_linear_bins([lo, hi], 44)
    ax.hist(vals, bins=bins, color=color, alpha=0.55, edgecolor=color, linewidth=0.8)
    ax.axvline(med, color=_INK, linewidth=1.2, linestyle="--", alpha=0.8)
    if rng:
        ax.set_xlim(*rng)
    ax.set_xlabel(axis); ax.set_ylabel("count")
    ax.grid(axis="y", color=_GRID, linewidth=0.6, alpha=0.6)


def _render_msd(runs, width_px, height_px, dpi):
    fig, ax = _new_axes(width_px, height_px, dpi)
    ax.set_title("Ensemble MSD curves", fontsize=11, fontweight="bold", pad=10)
    drew = False
    for r in runs:
        emsd = r._read_csv("_ensemble_msd.csv")
        if emsd is None or {"lag_frame", "msd_um2"} - set(emsd.columns) or not r.fi_s:
            continue
        ax.plot(emsd["lag_frame"].to_numpy(float) * r.fi_s,
                emsd["msd_um2"].to_numpy(float), color="#58a6ff", alpha=0.6, linewidth=1.4)
        drew = True
    if not drew:
        ax.text(0.5, 0.5, "no MSD curves exported", ha="center", va="center",
                transform=ax.transAxes, color=_FAINT, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ax.set_xlabel("lag time (s)"); ax.set_ylabel("MSD (µm²)")
        ax.grid(color=_GRID, linewidth=0.6, alpha=0.6)
    fig.tight_layout(pad=1.1)
    return _qimage_from_figure(fig)


def _render_motion_single(runs, width_px, height_px, dpi):
    fig, ax = _new_axes(width_px, height_px, dpi)
    ax.set_title("Motion-class fractions", fontsize=11, fontweight="bold", pad=10)
    counts = {}
    for r in runs:
        for k, v in (r.summary.get("motion_counts") or {}).items():
            counts[k] = counts.get(k, 0) + int(v)
    total = sum(counts.values())
    if not total:
        ax.text(0.5, 0.5, "no classification data", ha="center", va="center",
                transform=ax.transAxes, color=_FAINT, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    else:
        left = 0.0
        for cls in ("Immobile", "Confined", "Brownian", "Directed"):
            frac = 100.0 * counts.get(cls, 0) / total
            ax.barh(0, frac, left=left, color=MOTION_COLORS[cls], label=f"{cls} {frac:.0f}%",
                    edgecolor=_MAT, height=0.5)
            left += frac
        ax.set_xlim(0, 100); ax.set_ylim(-1, 1); ax.set_yticks([])
        ax.set_xlabel("fraction (%)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4,
                  fontsize=7, frameon=False, labelcolor=_MUTED)
    fig.tight_layout(pad=1.1)
    return _qimage_from_figure(fig)


def _render_raster(panel, runs, width_px, height_px, dpi):
    """Per-replicate spatial panel from the selected folder.  Source order:

      1. the exact per-panel PNG (figures/panels/<stem>_panel_<L>.png), exported
         only when 'Per-panel PNGs' is on — the cleanest source;
      2. else crop JUST this panel's cell out of the saved combined figure
         (<stem>_sptpalm_figure.png) — the real graph, never the whole grid;
      3. else another saved artifact for this panel (e.g. the superres density);
      4. else a 'not exported' placeholder.

    Showing the whole combined figure (the old fallback) is gone — it's always
    cropped to the appropriate graph instead."""
    from PySide6.QtGui import QImage
    pl = panel.get("panel_letter")
    # 1) exact per-panel PNG
    if pl:
        for r in runs:
            path = _wd.find_artifact(r, f"_panel_{pl}")
            if path:
                img = QImage(path)
                if not img.isNull():
                    return img
    # 2) crop this panel out of the combined single-run figure
    if pl:
        for r in runs:
            cpath = _wd.find_artifact(r, "_sptpalm_figure")
            if cpath:
                crop = _crop_combined_panel(cpath, pl)
                if crop is not None and not crop.isNull():
                    return crop
    # 3) a panel-specific artifact (superres density map for H/L, …)
    if panel.get("art"):
        for r in runs:
            path = _wd.find_artifact(r, panel["art"])
            if path:
                img = QImage(path)
                if not img.isNull():
                    return img
    # 4) placeholder
    fig, ax = _new_axes(width_px, height_px, dpi)
    ax.set_title(panel["name"], fontsize=11, fontweight="bold", pad=10)
    ax.text(0.5, 0.5, "Per-cell raster — not exported for this run.\n"
            "(Re-run with per-panel figures to populate.)", ha="center",
            va="center", transform=ax.transAxes, color=_FAINT, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=1.1)
    return _qimage_from_figure(fig)


def _render_motion(groups, width_px, height_px, dpi):
    """Stacked motion-class fractions per condition."""
    fig, ax = _new_axes(width_px, height_px, dpi)
    ax.set_title("Motion-class fractions  (%)", fontsize=11, fontweight="bold", pad=10)
    positions = np.arange(len(groups)) + 1
    bottoms = np.zeros(len(groups))
    for cls in ("Immobile", "Confined", "Brownian", "Directed"):
        fr = []
        for g in groups:
            counts = g.get("motion_counts") or {}
            total = sum(int(v) for v in counts.values()) if counts else 0
            fr.append(100.0 * int(counts.get(cls, 0)) / total if total else 0.0)
        fr = np.asarray(fr)
        ax.bar(positions, fr, bottom=bottoms, width=0.6, color=MOTION_COLORS[cls],
               alpha=0.85, label=cls, edgecolor=_MAT, linewidth=0.6)
        bottoms += fr
    ax.set_xticks(positions)
    ax.set_xticklabels([g["label"] for g in groups], fontsize=8, color=_MUTED)
    ax.set_ylabel("fraction (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4,
              fontsize=7, frameon=False, labelcolor=_MUTED)
    fig.tight_layout(pad=1.1)
    return _qimage_from_figure(fig)
