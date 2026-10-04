"""Group-level comparison figures for the experimental-design report and the
live Analysis tab.

These are *pure* renderers: they take already-aggregated, per-dish data and
return a matplotlib ``Figure``, so the report engine (:mod:`fa_compare`) and the
live tab (``workspace_figures``) share one implementation.  Statistics (which
test, which error type) are chosen by the caller — on the Analysis tab — while
the *style* comes from ``figures/*_style`` (Preferences → Figures → Graph
styles).  The second categorical axis is a generic "timepoint" whose labels are
whatever the user named them (pre/post, baseline/treated, …); a design with a
single timepoint just draws one series.
"""
from __future__ import annotations

import math

import numpy as np


# ── small helpers ────────────────────────────────────────────────────────────
def dispersion(arr: np.ndarray, kind: str) -> np.ndarray:
    """Per-column dispersion of a ``(n_dishes, n_lags)`` array along dishes.

    ``kind`` is the Analysis-tab error control: ``"SD"`` / ``"SEM"`` /
    ``"95% CI"``.  The dish is the experimental unit, so this is spread across
    dishes, not tracks.
    """
    arr = np.asarray(arr, dtype=float)
    n = arr.shape[0]
    sd = arr.std(0, ddof=1) if n > 1 else np.zeros(arr.shape[1])
    if kind == "SD":
        return sd
    sem = sd / math.sqrt(max(n, 1))
    return 1.96 * sem if kind == "95% CI" else sem


def facet_grid(n: int) -> tuple[int, int]:
    """(rows, cols) for ``n`` facets — a single row up to 3 groups, then
    near-square (cols ≥ rows) so 4 → 2×2, 5–6 → 2×3, etc."""
    if n <= 3:
        return 1, max(1, n)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


_DEFAULT_THEME = {"bg": "#ffffff", "fg": "#1b2027", "grid": "#e6e6e6",
                  "spine": "#cccccc", "muted": "#5b636e"}


_FALLBACK = ["#58a6ff", "#f78166", "#56d364", "#27c0e8", "#f6a623", "#a371f7"]


def draw_msd(fig, subplotspec, groups, data, lags_s, *, style="mean_faceted",
             err="SEM", tp_order=None, tp_colors=None, group_colors=None,
             theme=None, ylabel="MSD (µm²)", xlabel="Time (s)"):
    """Draw the MSD comparison into ``subplotspec`` (a matplotlib SubplotSpec)
    of ``fig``.

    Single-axes for ``overlaid``; a facet grid (via ``subplotspec.subgridspec``)
    for ``mean_faceted`` / ``individual`` — mirroring the LogD faceted panel so
    it slots into the combined report grid AND per-panel export.  See
    :func:`render_msd` for the argument meanings.
    """
    from matplotlib.lines import Line2D

    th = {**_DEFAULT_THEME, **(theme or {})}
    T = np.asarray(lags_s, dtype=float)
    if tp_order is None:
        tp_order = sorted({tp for g in groups for tp in data.get(g, {})})
    tp_colors = tp_colors or {}
    group_colors = group_colors or {}
    tpc = {tp: tp_colors.get(tp, _FALLBACK[i % len(_FALLBACK)])
           for i, tp in enumerate(tp_order)}

    def _axstyle(ax):
        ax.set_facecolor(th["bg"])
        ax.grid(True, color=th["grid"], lw=0.7)
        for s in ax.spines.values():
            s.set_color(th["spine"])
        ax.tick_params(labelsize=8, colors=th["fg"])
        ax.xaxis.label.set_color(th["fg"]); ax.yaxis.label.set_color(th["fg"])
        ax.title.set_color(th["fg"])

    # ── overlaid: all group means on one axes (group = colour, tp = linestyle),
    # with error bars ("overlaid") or without them ("overlaid_plain")
    if style in ("overlaid", "overlaid_plain"):
        bars = style == "overlaid"
        ax = fig.add_subplot(subplotspec)
        dashes = ["-", "--", ":", "-."]
        # Each curve sits a little to one side of its lag so the bars of
        # different conditions do not hide one another (≤ 0.15 lag in total).
        series = [(g, tp) for g in groups for tp in tp_order
                  if data.get(g, {}).get(tp) is not None and len(data[g][tp])]
        lag = float(np.median(np.diff(T))) if len(T) > 1 else 0.0
        step = (0.15 * lag / max(len(series) - 1, 1)) if bars else 0.0
        for gi, g in enumerate(groups):
            gc = group_colors.get(g, _FALLBACK[gi % len(_FALLBACK)])
            for ti, tp in enumerate(tp_order):
                arr = data.get(g, {}).get(tp)
                if arr is None or not len(arr):
                    continue
                # mean ± the Analysis tab's error between recordings, as faceted
                arr = np.asarray(arr, float)
                dx = (series.index((g, tp)) - (len(series) - 1) / 2) * step
                ax.errorbar(T + dx, arr.mean(0), yerr=(dispersion(arr, err) if bars else None),
                            color=gc, lw=2.0,
                            ls=dashes[ti % len(dashes)], marker="o", ms=3.2,
                            capsize=2.2, elinewidth=1.0)
        _axstyle(ax); ax.set_xlabel(xlabel, fontsize=10); ax.set_ylabel(ylabel, fontsize=10)
        gl = [Line2D([0], [0], color=group_colors.get(g, _FALLBACK[i % len(_FALLBACK)]), lw=2.4)
              for i, g in enumerate(groups)]
        l1 = ax.legend(gl, groups, title="Group", fontsize=8, title_fontsize=8,
                       frameon=False, loc="upper left", labelcolor=th["fg"])
        ax.add_artist(l1)
        if len(tp_order) > 1:
            tl = [Line2D([0], [0], color=th["muted"], lw=2, ls=dashes[i % len(dashes)])
                  for i in range(len(tp_order))]
            ax.legend(tl, tp_order, title="Timepoint", fontsize=8, title_fontsize=8,
                      frameon=False, loc="lower right", labelcolor=th["fg"])
        return

    # ── faceted: one facet per group, via a subgridspec of the panel slot ────
    rows, cols = facet_grid(len(groups))
    sub = subplotspec.subgridspec(rows, cols, hspace=0.32, wspace=0.12)
    axes = []
    for i in range(len(groups)):
        r, c = divmod(i, cols)
        shared = axes[0] if axes else None
        ax = fig.add_subplot(sub[r, c], sharex=shared, sharey=shared)
        axes.append(ax)
    # Within a facet the curve is coloured by TIME POINT (so pre/post read as two
    # traces).  But when there's no within-facet time split — a one-way comparison
    # where each facet IS a distinct condition (e.g. two cards that differ only by
    # time point, collapsed to one-way) — colour each facet by its GROUP colour so
    # the facets match the condition legend instead of all sharing the fallback.
    multi_tp = len(tp_order) > 1
    for i, (ax, g) in enumerate(zip(axes, groups)):
        gc = group_colors.get(g, _FALLBACK[i % len(_FALLBACK)])
        for tp in tp_order:
            arr = data.get(g, {}).get(tp)
            if arr is None or not len(arr):
                continue
            arr = np.asarray(arr, float); m = arr.mean(0)
            col = tpc[tp] if multi_tp else gc
            if style == "individual":
                for row in arr:
                    ax.plot(T, row, color=col, lw=0.8, alpha=0.30)
                ax.plot(T, m, color=col, lw=2.4, marker="o", ms=3.5, label=tp)
            else:  # mean_faceted
                ax.errorbar(T, m, yerr=dispersion(arr, err), color=col, marker="o",
                            ms=3.5, lw=1.6, capsize=2.2, elinewidth=1.0, label=tp)
        _axstyle(ax)
        ax.set_title(g, fontsize=9, fontweight="bold")
        if i < len(groups) - cols:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(xlabel, fontsize=8)
        if i % cols == 0:
            ax.set_ylabel(ylabel, fontsize=8)
        else:
            ax.tick_params(labelleft=False)
    if len(tp_order) > 1 and axes:
        axes[min(1, len(axes) - 1)].legend(title="Timepoint", fontsize=7.5,
                                           title_fontsize=7.5, frameon=False,
                                           loc="upper left", labelcolor=th["fg"])


def render_msd(groups, data, lags_s, *, style="mean_faceted", err="SEM",
               tp_order=None, tp_colors=None, group_colors=None, theme=None,
               ylabel="MSD (µm²)", xlabel="Time lag (s)", width_in=9.0, dpi=110):
    """Standalone ensemble-MSD comparison figure (for the live tab / a per-panel
    export).  Wraps :func:`draw_msd` in its own figure + gridspec.

    groups   : ordered group (condition) names.
    data     : ``{group: {timepoint_label: (n_dishes, n_lags) array}}``.
    lags_s   : ``(n_lags,)`` time-lag axis (seconds).
    style    : ``'mean_faceted'`` | ``'individual'`` | ``'overlaid'`` |
               ``'overlaid_plain'`` (overlaid, no error bars).
    err      : ``'SD'`` | ``'SEM'`` | ``'95% CI'`` (mean_faceted and overlaid).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = {**_DEFAULT_THEME, **(theme or {})}
    if style in ("overlaid", "overlaid_plain"):
        h = width_in * 0.6
    else:
        rows, cols = facet_grid(len(groups))
        h = max(3.0, (width_in / cols) * rows * 0.74)
    fig = plt.figure(figsize=(width_in, h), dpi=dpi, facecolor=th["bg"])
    draw_msd(fig, fig.add_gridspec(1, 1)[0], groups, data, lags_s, style=style,
             err=err, tp_order=tp_order, tp_colors=tp_colors,
             group_colors=group_colors, theme=theme, ylabel=ylabel, xlabel=xlabel)
    try:
        fig.tight_layout()
    except Exception:
        pass
    return fig


# ── group comparison (scalar metrics): box+points / grouped / violin / bar ───
def draw_group_comparison(fig, subplotspec, groups, values, *, style="box_points",
                          stat_label="", pairs=None, tp_order=None, tp_colors=None,
                          group_colors=None, theme=None, ylabel="",
                          err="SEM", markers=None, day_key=None):
    """Draw a scalar-metric group comparison into ``subplotspec``.

    values : ``{group: {timepoint: 1-D array of per-dish values}}`` (the dish is
             the unit).  ``box_points`` / ``violin`` / ``bar`` pool timepoints;
             ``grouped`` splits each group into its timepoints.
    style  : ``'box_points'`` | ``'grouped'`` | ``'violin'`` | ``'bar'``.
    stat_label : the p-value label (e.g. "p = 0.013"), drawn above the axes when
                 there are no `pairs` to bracket.
    pairs      : ``[(i, j, "p = …")]`` — a bracket with that label is drawn between
                 groups i and j, above the data.
    markers    : optional dot shape per value, shaped like ``values`` (the
                 recording day); ``day_key`` ``[(marker, label)]`` keys them.
    """
    from matplotlib.lines import Line2D

    th = {**_DEFAULT_THEME, **(theme or {})}
    group_colors = group_colors or {}
    if tp_order is None:
        tp_order = sorted({tp for g in groups for tp in values.get(g, {})})
    tp_colors = tp_colors or {}
    tpc = {tp: tp_colors.get(tp, _FALLBACK[i % len(_FALLBACK)])
           for i, tp in enumerate(tp_order)}
    gc = {g: group_colors.get(g, _FALLBACK[i % len(_FALLBACK)]) for i, g in enumerate(groups)}
    ax = fig.add_subplot(subplotspec)
    rng = np.random.default_rng(0)

    def _pool(g):
        chunks = [np.asarray(v, float) for v in values.get(g, {}).values() if v is not None and len(v)]
        return np.concatenate(chunks) if chunks else np.array([])

    def _shapes(g):
        """Dot shapes aligned with _pool(g) (all circles without `markers`)."""
        out = []
        for tp, v in values.get(g, {}).items():
            if v is None or not len(v):
                continue
            m = (markers or {}).get(g, {}).get(tp)
            out += list(m) if m is not None and len(m) == len(v) else ["o"] * len(v)
        return np.asarray(out, dtype=object)

    def _dots(x, d, shapes, **kw):
        for m in dict.fromkeys(shapes):
            sel = shapes == m
            ax.scatter(x[sel], d[sel], marker=m, **kw)

    def _style(a):
        a.set_facecolor(th["bg"]); a.grid(True, axis="y", color=th["grid"], lw=0.7)
        a.set_axisbelow(True)
        for s in a.spines.values():
            s.set_color(th["spine"])
        a.tick_params(labelsize=8, colors=th["fg"])
        a.set_xticks(range(len(groups)))
        a.set_xticklabels(groups, rotation=20, ha="right", fontsize=8, color=th["fg"])
        a.set_ylabel(ylabel, fontsize=10, color=th["fg"])

    def _box(a, data, positions, widths, facecols):
        from matplotlib.colors import to_rgb
        bp = a.boxplot(data, positions=positions, widths=widths, patch_artist=True,
                       showfliers=False)
        # Pale tint + group-coloured outline (as the report): a solid group-colour
        # face hid the group-coloured replicate dots drawn inside it.
        for patch, c in zip(bp["boxes"], facecols):
            patch.set_facecolor((*to_rgb(c), 0.22))
            patch.set_edgecolor(c); patch.set_linewidth(1.4)
        for el in ("whiskers", "caps", "medians"):
            for ln in bp[el]:
                ln.set_color(th["fg"])

    if style == "grouped":
        w = 0.8 / max(1, len(tp_order))
        for i, g in enumerate(groups):
            for j, tp in enumerate(tp_order):
                v = values.get(g, {}).get(tp)
                if v is None or not len(v):
                    continue
                v = np.asarray(v, float); pos = i + (j - (len(tp_order) - 1) / 2) * w
                _box(ax, [v], [pos], w * 0.9, [tpc[tp]])
                ax.scatter(np.full(len(v), pos) + rng.uniform(-0.04, 0.04, len(v)),
                           v, color=tpc[tp], s=9, zorder=3, edgecolors=th["bg"],
                           linewidths=0.5)
        _style(ax)
        if len(tp_order) > 1:
            ax.legend([Line2D([0], [0], color=tpc[tp], lw=8) for tp in tp_order],
                      tp_order, title="Timepoint", frameon=False, fontsize=8,
                      title_fontsize=8, loc="best", labelcolor=th["fg"])
    elif style == "violin":
        data = [_pool(g) for g in groups]
        idx = [i for i, d in enumerate(data) if len(d)]
        if idx:
            vp = ax.violinplot([data[i] for i in idx], positions=idx, widths=0.7,
                               showextrema=False)
            for b, i in zip(vp["bodies"], idx):
                b.set_facecolor(gc[groups[i]]); b.set_alpha(0.5); b.set_edgecolor(th["fg"])
            for i in idx:
                d = data[i]
                _dots(np.full(len(d), i) + rng.uniform(-0.09, 0.09, len(d)), d,
                      _shapes(groups[i]), color=gc[groups[i]], s=22, zorder=3,
                      edgecolors=th["bg"], linewidths=0.4)
        _style(ax)
    elif style == "bar":
        means = [float(np.mean(_pool(g))) if len(_pool(g)) else 0.0 for g in groups]
        errs = [float(dispersion(_pool(g)[None, :].T if len(_pool(g)) else np.zeros((1, 1)), err)[0])
                if len(_pool(g)) > 1 else 0.0 for g in groups]
        ax.bar(range(len(groups)), means, yerr=errs, color=[gc[g] for g in groups],
               edgecolor=th["fg"], width=0.62, zorder=3, capsize=3)
        _style(ax)
    else:  # box_points (default)
        data = [_pool(g) for g in groups]
        idx = [i for i, d in enumerate(data) if len(d)]
        boxed = [i for i in idx if len(data[i]) >= 2]     # one value: just its dot
        if boxed:
            _box(ax, [data[i] for i in boxed], boxed, 0.55, [gc[groups[i]] for i in boxed])
        for i in idx:
            d = data[i]
            _dots(np.full(len(d), i) + rng.uniform(-0.12, 0.12, len(d)), d,
                  _shapes(groups[i]), color=gc[groups[i]], s=24, zorder=4,
                  edgecolors=th["bg"], linewidths=0.4)
        _style(ax)

    if day_key and style in ("box_points", "violin"):     # key for the dot shapes
        _lg = ax.legend(handles=[Line2D([], [], ls="none", marker=m, ms=5, mfc=th["muted"],
                                        mec=th["bg"], mew=0.5, label=lab) for m, lab in day_key],
                        title="Recording day", loc="upper left", bbox_to_anchor=(1.01, 1.0),
                        frameon=False, fontsize=7, title_fontsize=7, labelcolor=th["fg"],
                        handletextpad=0.3, borderaxespad=0)
        _lg.get_title().set_color(th["fg"])
    if pairs:                             # a bracket + p-value per compared pair
        from firefly.analysis.fa_figure_common import draw_pvalue_brackets
        top = max((np.nanmax(_pool(g)) for g in groups if len(_pool(g))), default=None)
        draw_pvalue_brackets(ax, list(pairs), color=th["fg"], fontsize=9, data_top=top)
    elif stat_label:                      # above the plot, clear of the data
        ax.text(0.5, 1.02, stat_label, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=10, color=th["fg"])
    return ax


# ── track-length distribution: overlaid density with the filter threshold ────
def draw_length_cdf(ax, cells, labels, colors, *, per_recording=True):
    """Cumulative fraction of trajectories up to each length, one curve per
    group.  ``cells[label]`` is a list of per-recording length arrays; with
    ``per_recording`` each recording's curve is averaged (± SEM, shaded),
    otherwise all a group's trajectories are pooled.  Returns the 99th
    percentile of every length, for the x-range."""
    from firefly.analysis.fa_figure_common import group_curve
    pooled = {g: np.concatenate(c) for g, c in cells.items() if c}
    combined = np.concatenate(list(pooled.values())) if pooled else np.array([])
    x_clip = float(np.percentile(combined, 99)) if len(combined) else None
    if per_recording and len(combined):
        # each recording's CDF at every observed length (whole frames, so a
        # few hundred values), averaged (± SEM)
        _xg = np.unique(combined)
        _cdf = lambda v: np.searchsorted(np.sort(v), _xg, side="right") / len(v)
    for grp_label, color in zip(labels, colors):
        p = pooled.get(grp_label)
        if p is None or len(p) == 0:
            continue
        if per_recording:
            y, sem = group_curve(cells[grp_label], _cdf, True)
            if sem is not None:
                ax.fill_between(_xg, np.clip(y - sem, 0, 1), np.clip(y + sem, 0, 1),
                                color=color, alpha=0.25, linewidth=0, step="post", zorder=2)
            ax.plot(_xg, y, color=color, lw=1.5, label=grp_label, drawstyle="steps-post")
            continue
        x_sorted = np.sort(p)
        y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
        ax.plot(x_sorted, y, color=color, lw=1.5, label=grp_label)
    return x_clip


def draw_length_density(fig, subplotspec, groups, dists, *, threshold=None,
                        group_colors=None, theme=None, xlabel="Trajectory length (frames)",
                        per_recording=True, ylabel="Probability density (per frame)"):
    """Overlaid per-group KDEs of track length; dashed line = filter threshold.

    ``dists[g]`` is one array, or a list of per-recording arrays: then, with
    ``per_recording``, the curve is the mean of the recordings' KDEs, shaded
    ± SEM (see ``fa_figure_common.group_curve``)."""
    from scipy.stats import gaussian_kde
    from firefly.analysis.fa_figure_common import draw_sem_band, group_curve, recording_cells
    th = {**_DEFAULT_THEME, **(theme or {})}
    group_colors = group_colors or {}
    ax = fig.add_subplot(subplotspec)
    pooled = {g: np.concatenate(c) for g, c in
              ((g, recording_cells(dists.get(g))) for g in groups) if c}
    lo = min((float(np.min(d)) for d in pooled.values()), default=0.0)
    hi = max((float(np.percentile(d, 99)) for d in pooled.values()), default=1.0)
    x = np.linspace(lo, max(hi, lo + 1), 200)

    def _kde(v):
        if len(v) < 2 or np.ptp(v) <= 0:
            return None
        try:
            return gaussian_kde(v)(x)
        except Exception:
            return None
    for i, g in enumerate(groups):
        if g not in pooled or len(pooled[g]) < 2:
            continue
        y, sem = group_curve(dists.get(g), _kde, per_recording)
        if y is None:
            continue
        c = group_colors.get(g, _FALLBACK[i % len(_FALLBACK)])
        if per_recording:
            draw_sem_band(ax, x, y, sem, c, alpha=0.3, floor=0.0)
        else:
            ax.fill_between(x, y, color=c, alpha=0.28)
        ax.plot(x, y, color=c, lw=1.8, label=g)
    if threshold is not None:
        ax.axvline(float(threshold), ls="--", color=th["fg"], lw=1.2)
    ax.set_facecolor(th["bg"])
    for s in ax.spines.values():
        s.set_color(th["spine"])
    ax.tick_params(labelsize=8, colors=th["fg"])
    ax.set_xlabel(xlabel, fontsize=10, color=th["fg"])
    ax.set_ylabel(ylabel, fontsize=10, color=th["fg"])
    ax.legend(title="Group", frameon=False, fontsize=8, title_fontsize=8, labelcolor=th["fg"])
    return ax


# ── MSD-AUC change across timepoints: paired lines / Δ box ───────────────────
def draw_auc_change(fig, subplotspec, groups, paired, *, style="paired",
                    stat_labels=None, group_colors=None, tp_order=None,
                    theme=None, ylabel="AUC (µm²s)"):
    """Pre/post-style AUC change.

    paired : ``{group: {timepoint: 1-D array}}`` with two timepoints, matched by
             dish order.  ``paired`` → per-dish lines + group-median + p per
             facet; ``delta`` → box of (t2 − t1) per group with an omnibus test.
    stat_labels : ``{group: "p = ..."}`` (paired) or a single str (delta).
    """
    th = {**_DEFAULT_THEME, **(theme or {})}
    group_colors = group_colors or {}
    stat_labels = stat_labels or {}
    if tp_order is None:
        tp_order = sorted({tp for g in groups for tp in paired.get(g, {})})[:2]
    rng = np.random.default_rng(1)

    if style == "delta" or len(tp_order) < 2:
        ax = fig.add_subplot(subplotspec)
        deltas = []
        for g in groups:
            a = paired.get(g, {}).get(tp_order[0]) if len(tp_order) else None
            b = paired.get(g, {}).get(tp_order[1]) if len(tp_order) > 1 else None
            deltas.append(np.asarray(b, float) - np.asarray(a, float)
                          if a is not None and b is not None and len(a) == len(b) and len(a)
                          else np.array([]))
        idx = [i for i, d in enumerate(deltas) if len(d)]
        if idx:
            bp = ax.boxplot([deltas[i] for i in idx], positions=idx, widths=0.55,
                            patch_artist=True, showfliers=False)
            for patch, i in zip(bp["boxes"], idx):
                patch.set_facecolor(group_colors.get(groups[i], _FALLBACK[i % len(_FALLBACK)]))
                patch.set_alpha(0.65); patch.set_edgecolor(th["fg"])
            for el in ("whiskers", "caps", "medians"):
                for ln in bp[el]:
                    ln.set_color(th["fg"])
            for i in idx:
                d = deltas[i]
                ax.scatter(np.full(len(d), i) + rng.uniform(-0.1, 0.1, len(d)), d,
                           color=th["fg"], s=15, zorder=3)
        ax.axhline(0, ls="--", color=th["muted"])
        ax.set_facecolor(th["bg"]); ax.grid(True, axis="y", color=th["grid"], lw=0.7); ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(th["spine"])
        ax.tick_params(labelsize=8, colors=th["fg"])
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=8, color=th["fg"])
        ax.set_ylabel(f"Δ {ylabel}", fontsize=10, color=th["fg"])
        if isinstance(stat_labels, str) and stat_labels:
            ax.text(0.02, 0.97, stat_labels, transform=ax.transAxes, va="top",
                    fontsize=10, color=th["fg"])
        return ax

    # paired: facet per group
    rows, cols = facet_grid(len(groups))
    sub = subplotspec.subgridspec(rows, cols, hspace=0.4, wspace=0.28)
    for i, g in enumerate(groups):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(sub[r, c])
        a = paired.get(g, {}).get(tp_order[0]); b = paired.get(g, {}).get(tp_order[1])
        if a is not None and b is not None and len(a) == len(b) and len(a):
            a = np.asarray(a, float); b = np.asarray(b, float)
            for x0, x1 in zip(a, b):
                ax.plot([0, 1], [x0, x1], color=th["muted"], lw=1.2, marker="o", ms=4, alpha=0.75)
            ax.plot([0, 1], [np.median(a), np.median(b)], color=th["fg"], lw=2.4, marker="o", ms=6)
        ax.set_facecolor(th["bg"])
        for s in ax.spines.values():
            s.set_color(th["spine"])
        ax.set_xticks([0, 1]); ax.set_xticklabels(tp_order, fontsize=8, color=th["fg"])
        ax.set_xlim(-0.3, 1.3); ax.tick_params(labelsize=8, colors=th["fg"])
        ax.set_title(g, fontsize=9, fontweight="bold", color=th["fg"])
        if c == 0:
            ax.set_ylabel(ylabel, fontsize=8, color=th["fg"])
        lbl = stat_labels.get(g) if isinstance(stat_labels, dict) else None
        if lbl:
            ax.text(0.04, 0.96, lbl, transform=ax.transAxes, va="top", fontsize=8, color=th["fg"])
    return None
