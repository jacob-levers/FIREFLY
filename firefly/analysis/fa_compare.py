"""Multi-group comparison figure, statistics and PDF report.

Extracted from sptpalm_analysis.py (#7); re-exported there for compatibility.
"""
from __future__ import annotations

import os
import json
import math
import copy
import hashlib
import threading
from dataclasses import dataclass, field
from firefly.analysis.fa_constants import (MOTION_CLASS_COLORS, MOTION_CLASS_ORDER,
                                           motion_class_colors, label_text_color,
                                           DEFAULT_FRAME_INTERVAL_S)
from firefly.analysis.fa_theme import _theme_palette, style_axes
from firefly.analysis.fa_palmtracer import load_summary_from_folder, _win_long_path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as _stats
from firefly.analysis.fa_diffusion import (_msd_auc, _mob_immob_ratio, MOBILE_D_THRESHOLD_DEFAULT,
                          _motion_fractions, _track_lengths,
                          compute_van_hove, compute_vacf)
from firefly.analysis.fa_circular import (save_comparison_circular_statistics,
                         _stat_test, _stat_test_n, _hedges_g_ci,
                         _paired_test, _paired_hedges_g,
                         _p_stars, compute_per_track_mean_angle,
                         compute_circular_comparison_tests)


from firefly.analysis import fa_twoway


_FI_DEFAULT_WARNED: set = set()   # stems already warned about a missing Δt


# ── ReportData: the style/theme-independent compute output (cacheable) ─────────
@dataclass
class ReportData:
    """Everything a comparison needs that does NOT depend on theme, graph style or
    panel selection: loaded per-folder summaries, per-replicate ``summary_df``,
    factor levels + colours, the two-way ANOVA, and a memo cache for the expensive
    effect-size statistics.  Produced by :func:`compute_report`, consumed by
    :func:`render_report`.  The controller caches this and reuses it across
    re-renders, so a style/theme change is a pure redraw with no recompute."""
    cfg: dict
    groups: list
    n_groups: int
    group_factor: list
    timepoints_per_card: list
    timepoint_tokens: list
    distinct_tp: bool
    two_factor: bool
    labels: list
    colors: list
    folder_lists: list
    all_summaries: list
    skipped: list
    summary_df: object
    group_order: list
    tp_order: list
    card_colors: dict
    group_colors: dict
    many_groups: bool
    bar_xticks: list
    twoway_df: object
    twoway_msg: object
    pair_warn: object
    paired_df: object
    mobile_d_threshold: float
    stat_cache: dict = field(default_factory=dict)


# ── Memoised statistics ───────────────────────────────────────────────────────
# The effect-size CIs / power in `_stat_test_n` are the dominant cost of a render
# (~1.2 s across the panels).  They depend only on (data arrays, labels, cfg) —
# NOT on theme/style — so re-rendering the same data in a new style shouldn't
# recompute them.  `render_report` points this thread-local at the ReportData's
# `stat_cache`; the draw's calls go through `_STN`/`_PT`, which memoise per unique
# input.  A deep copy is returned so callers can mutate the result freely (the bar
# panels annotate the pairwise dicts in place) without poisoning the cache.
_TL = threading.local()


def _arr_key(arrs):
    out = []
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
        out.append((a.shape, hashlib.blake2b(a.tobytes(), digest_size=16).digest()))
    return tuple(out)


def _cfg_key(cfg):
    try:
        return json.dumps(cfg, sort_keys=True, default=str)
    except Exception:
        return repr(cfg)


def _STN(arrs, labels, cfg):
    """Memoised :func:`_stat_test_n` — same (data, labels, cfg) reuses the result."""
    cache = getattr(_TL, "stat_cache", None)
    if cache is None:
        return _stat_test_n(arrs, labels, cfg)
    key = ("n", _arr_key(arrs), tuple(str(x) for x in labels), _cfg_key(cfg))
    hit = cache.get(key)
    if hit is None:
        hit = _stat_test_n(arrs, labels, cfg)
        cache[key] = hit
    return copy.deepcopy(hit)


def _PT(a, b, cfg):
    """Memoised :func:`_paired_test` (returns an immutable ``(p, stars)`` tuple)."""
    cache = getattr(_TL, "stat_cache", None)
    if cache is None:
        return _paired_test(a, b, cfg)
    key = ("p", _arr_key([a, b]), _cfg_key(cfg))
    hit = cache.get(key)
    if hit is None:
        hit = _paired_test(a, b, cfg)
        cache[key] = hit
    return hit


def _fi_or_default(params, stem="", _warned=_FI_DEFAULT_WARNED):
    """Frame interval (s) from a folder's params dict, falling back to the
    app-wide default and WARNING (once per stem) when the key is absent.

    A missing ``frame_interval_s`` used to silently default to 0.05 s here while
    the rest of the app uses 0.02 s, rescaling this folder's AUC-MSD and MSD
    time axis 2.5x against its siblings with no warning (#11).  Now it uses the
    SAME default as everywhere else and says so, so the discrepancy is visible.
    """
    v = params.get("frame_interval_s")
    if v is not None:
        return float(v)
    if stem not in _warned:
        _warned.add(stem)
        print(f"  Compare WARNING: '{stem or '<folder>'}' has no frame_interval_s "
              f"in its params — assuming Δt = {DEFAULT_FRAME_INTERVAL_S} s.  MSD and "
              f"AUC scale with Δt; set it so groups are compared on the same axis.")
    return float(DEFAULT_FRAME_INTERVAL_S)


class CompareInputError(Exception):
    """A user-input problem with a comparison (no valid folders, <2 groups,
    inaccessible paths).  The worker turns this into a friendly popup instead of
    a crash report — it is an expected condition, not a bug."""




def _replicate_colors(k):
    """k visually distinct colours for per-replicate SuperPlot dots.  Uses
    tab10 for ≤10 replicates, tab20 beyond that (cycling if even larger)."""
    if k <= 0:
        return []
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab10" if k <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(k)]


def _bar_with_dots_n(ax, data_per_group, labels, colors, palette,
                     ylabel="", record_stats=None, metric_name="",
                     xtick_labels=None, stats_config=None, annot_sink=None,
                     style="box_points"):
    """Per-group scalar comparison with individual replicate dots, generalised
    to N groups.  ``style`` (Preferences → Figures → Group comparison) picks the
    backdrop mark: ``box_points`` (box + median/IQR, the default) / ``violin`` /
    ``bar`` (mean ± SEM).  ``grouped`` falls back to the box here (the grouped-by-
    timepoint layout is a separate two-factor renderer).  The per-replicate dots
    and the full stats annotation are identical across styles.

    For 2 groups: shows pairwise stars on a bracket (matches lab style).
    For 3+ groups: shows omnibus ANOVA / Kruskal p-value as a panel
    annotation; full pairwise comparisons go to record_stats[metric_name].

    `xtick_labels` overrides the x-axis tick text (display only — `labels`
    still drives the statistics); used to put short tokens on the axis when
    there are many groups, with the full names carried by the shared legend."""
    sig_col = palette["SIG"]
    # Bar body = a wash of each group's OWN colour blended toward the figure
    # background, with the saturated group colour kept as the edge ("tint +
    # outline").  This is theme-adaptive: a pale pastel on white themes
    # (Light / Publication) and a subtle dark tint on dark themes (Dark /
    # AMOLED) — never the near-black bar a fixed dark BAR_FILL produced on a
    # white background.  Per-group, so each bar reads as its own condition.
    import matplotlib.colors as _mc
    _bgc = np.array(_mc.to_rgb(palette.get("BG", "#ffffff")))
    def _bar_fill_for(i, frac=0.78):
        rgb = np.array(_mc.to_rgb(colors[i]))
        return tuple((1.0 - frac) * rgb + frac * _bgc)

    arrs = [np.asarray(d, dtype=float) for d in data_per_group]
    arrs = [a[np.isfinite(a)] for a in arrs]
    n = len(arrs)
    means = [float(a.mean()) if len(a) else 0.0 for a in arrs]
    sems  = [float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
             for a in arrs]
    x = np.arange(n)
    # Bar = mean across REPLICATES; error = SEM across replicates.  The dots
    # are the per-replicate values the stats are actually computed on — a
    # SuperPlot (Lord et al. 2020): each replicate gets its own colour so the
    # reader sees the true unit of replication, not pooled localisations.
    # SuperPlot styling: the bar is just a faint backdrop for the mean — the
    # per-replicate DOTS and the mean±SEM error bar carry the information.  So
    # the bar face is a very pale wash (blended 88% toward the background) at low
    # opacity, while the EDGE keeps the saturated group colour and the error bars
    # stay solid — the dots read as the data, not the bar.
    style = (style or "box_points").lower()
    if style == "bar":
        ax.bar(x, means, yerr=sems, capsize=4,
               color=[(*_bar_fill_for(i, frac=0.88), 0.45) for i in range(n)],
               edgecolor=colors, linewidth=1.6,
               ecolor=sig_col)
    elif style in ("violin", "violin_points"):
        vi = [i for i in range(n) if len(arrs[i]) >= 2]
        if vi:
            parts = ax.violinplot([arrs[i] for i in vi], positions=vi,
                                  widths=0.7, showextrema=False)
            for body, i in zip(parts["bodies"], vi):
                body.set_facecolor(_bar_fill_for(i, 0.85))
                body.set_edgecolor(colors[i]); body.set_linewidth(1.3)
                body.set_alpha(0.45)
        for i in range(n):                       # mean ± SEM marker on top
            if len(arrs[i]):
                ax.errorbar(i, means[i], yerr=sems[i], fmt="_", ms=14,
                            color=sig_col, capsize=4, lw=1.5, zorder=4)
    else:                                        # box_points / grouped → box + IQR
        bp = ax.boxplot([arrs[i] if len(arrs[i]) else [np.nan] for i in range(n)],
                        positions=list(x), widths=0.5, showfliers=False,
                        patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor((*_bar_fill_for(i, 0.85), 0.35))
            patch.set_edgecolor(colors[i]); patch.set_linewidth(1.6)
        for ln in bp["whiskers"] + bp["caps"]:
            ln.set_color(palette.get("MUT", sig_col)); ln.set_linewidth(1.0)
        for md in bp["medians"]:
            md.set_color(sig_col); md.set_linewidth(1.8)
    rng = np.random.default_rng(0)
    max_rep = max((len(a) for a in arrs), default=0)
    rep_colors = _replicate_colors(max_rep)
    for i, a in enumerate(arrs):
        if len(a):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(a)), a,
                       c=[rep_colors[k] for k in range(len(a))],
                       s=34, zorder=3, edgecolors=colors[i], linewidths=0.6)
    ax.set_xticks(x)
    disp = list(xtick_labels) if xtick_labels is not None else list(labels)
    _short = all(len(str(t)) <= 6 for t in disp)
    ax.set_xticklabels(disp, rotation=0 if _short else (30 if n > 3 else 0),
                       ha="center" if _short else ("right" if n > 3 else "center"),
                       fontsize=8 if n > 6 else 9)
    ax.set_ylabel(ylabel)

    # Stats — config-driven; the displayed star uses the chosen multiple-
    # comparison correction so the figure agrees with the CSV (no "* on the
    # figure / ns in the table" mismatch).  The panel also NAMES the test and
    # correction, so it is self-describing.
    from firefly.analysis.fa_stats_config import (
        normalize_stats_config, correct_pvalues, stars_for, describe_test_label)
    cfg = normalize_stats_config(stats_config)
    omnibus, pairwise = _STN(arrs, labels, cfg)
    # Within-metric correction onto the pairwise list (also done centrally for
    # the CSV; computed here so the initial draw is already correct).
    # Self-correcting post-hocs (Games-Howell / Tukey / Dunnett) already control
    # their own family — don't double-correct them here either.
    _corr_idx = [k for k, pw in enumerate(pairwise)
                 if not pw.get("self_corrected")]
    _wp = correct_pvalues([pairwise[k]["p"] for k in _corr_idx], cfg["correction"])
    _wmap = dict(zip(_corr_idx, _wp))
    for k, pw in enumerate(pairwise):
        if pw.get("self_corrected"):
            pw["p_within"] = pw.get("p")
            pw["stars_within"] = stars_for(pw.get("p"), cfg["alpha"])
        else:
            pw["p_within"] = _wmap[k]
            pw["stars_within"] = stars_for(_wmap[k], cfg["alpha"])
    if record_stats is not None and metric_name:
        record_stats[metric_name] = {"omnibus": omnibus, "pairwise": pairwise}

    def _g_ci_str(rec):
        """'g = -1.20 [95% CI -2.10, -0.30]' or '' if g unavailable."""
        g = rec.get("hedges_g")
        if g is None or not np.isfinite(g):
            return ""
        lo, hi = rec.get("hedges_g_ci_low"), rec.get("hedges_g_ci_high")
        if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
            return f"g = {g:.2f} [95% CI {lo:.2f}, {hi:.2f}]"
        return f"g = {g:.2f}"

    def _delta_str(rec):
        """Cliff's delta line: 'δ = 0.62 [0.20, 0.90]' or '' if unavailable."""
        d = rec.get("cliffs_delta")
        if d is None or not np.isfinite(d):
            return ""
        lo, hi = rec.get("cliffs_delta_ci_low"), rec.get("cliffs_delta_ci_high")
        if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
            return f"δ = {d:.2f} [{lo:.2f}, {hi:.2f}]"
        return f"δ = {d:.2f}"

    def _tost_str(rec):
        """Equivalence verdict line (only when TOST is on)."""
        if not cfg["equivalence_tost"]:
            return ""
        teq = rec.get("tost_equivalent")
        if teq is None:
            return ""
        tp = rec.get("tost_p")
        tail = (f" (TOST p={tp:.3f})"
                if (tp is not None and np.isfinite(tp)) else "")
        verdict = "equivalent" if teq else "not equivalent"
        return f"±{cfg['tost_margin']:g} SD: {verdict}{tail}"

    use_corr = cfg["figure_stars_use_corrected"]
    corr_caption = describe_test_label("", cfg["correction"],
                                       cfg["across_metric_correction"])

    def _annot_text(which="within"):
        """Build the panel's stats annotation. `which` ∈ {'within','across'}
        selects which corrected star to show (the post-pass switches to
        'across' when across-metric correction is on)."""
        if n == 2 and pairwise:
            pair = pairwise[0]
            if not np.isfinite(pair.get("p", np.nan)):
                return None
            if use_corr:
                pv = pair.get(f"p_{which}", pair.get("p_within", pair["p"]))
                st = pair.get(f"stars_{which}", pair.get("stars_within", ""))
            else:
                pv, st = pair["p"], pair["stars"]
            p_str = f"p = {pv:.2e}" if pv < 0.001 else f"p = {pv:.3f}"
            lines = [pair.get("test", ""), f"{p_str}  {st}".rstrip()]
            g_str = _g_ci_str(pair)
            if g_str:
                lines.append(g_str)
            d_str = _delta_str(pair)
            if d_str:
                lines.append(d_str)
            t_str = _tost_str(pair)
            if t_str:
                lines.append(t_str)
            lines.append(corr_caption)
            if pair.get("note"):
                lines.append(pair["note"])
            return "\n".join([ln for ln in lines if ln])
        if n > 2 and omnibus:
            pv = omnibus["p"]
            p_str = f"p = {pv:.2e}" if pv < 0.001 else f"p = {pv:.3f}"
            lines = [omnibus["test"], f"{p_str}   {omnibus['stars']}"]
            es = omnibus.get("effect_size")
            if es is not None and np.isfinite(es):
                sym = "η²" if omnibus.get("effect_size_kind") == "eta_sq" else "ε²"
                lines.append(f"{sym} = {es:.3f}")
            lines.append(f"pairwise: {corr_caption}")
            if omnibus.get("note"):
                lines.append(omnibus["note"])
            return "\n".join(lines)
        return None

    # Data extent — INCLUDING negatives.  Most metrics (D, mobile fraction,
    # α, …) are ≥ 0, but signed metrics such as the VACF lag-1 persistence can
    # be < 0 (anti-persistent / caged motion, or the localisation-noise
    # anti-correlation between consecutive steps).  For those we must NOT pin
    # the y-axis to a 0 floor or the negative bars get clipped below the
    # baseline and read as "impossibly low / broken".
    _have = any(len(a) for a in arrs)
    _allv = (np.concatenate([a for a in arrs if len(a)])
             if _have else np.array([0.0]))
    dmin, dmax = float(_allv.min()), float(_allv.max())
    has_neg = dmin < 0.0

    # A 0 reference line so a downward bar reads as a sign, not an error.
    if has_neg:
        ax.axhline(0.0, color=palette.get("GRD", "#cccccc"), lw=0.8, zorder=1)

    txt = None
    text = _annot_text("within")
    if not has_neg:
        # ── all-positive metric: original layout (unchanged) ──────────────
        top_data = max([a.max() if len(a) else 0 for a in arrs]
                       + [max(means) * 1.2 if max(means) > 0 else 1])
        if text is not None and n == 2:
            top = top_data * 1.05
            ax.plot([0, 0, 1, 1], [top, top * 1.03, top * 1.03, top],
                    color=sig_col, lw=0.8)
            txt = ax.text(0.5, top * 1.05, text, ha="center", va="bottom",
                          fontsize=7.5, color=sig_col)
            ax.set_ylim(0, top * 1.62)
        elif text is not None and n > 2:
            txt = ax.text(0.02, 0.98, text, transform=ax.transAxes,
                          ha="left", va="top", fontsize=7.5, color=sig_col,
                          bbox=dict(facecolor=palette["PNL"], edgecolor="none",
                                    alpha=0.7, pad=3))
    else:
        # ── signed metric: span-aware limits, brackets above the max ──────
        lo, hi = min(0.0, dmin), max(0.0, dmax)
        span = (hi - lo) or 1.0
        if text is not None and n == 2:
            bracket = hi + span * 0.06
            ax.plot([0, 0, 1, 1],
                    [bracket, bracket + span * 0.03,
                     bracket + span * 0.03, bracket],
                    color=sig_col, lw=0.8)
            txt = ax.text(0.5, bracket + span * 0.05, text,
                          ha="center", va="bottom",
                          fontsize=7.5, color=sig_col)
            ax.set_ylim(lo - span * 0.08, bracket + span * 0.55)
        else:
            ax.set_ylim(lo - span * 0.08, hi + span * 0.30)
            if text is not None and n > 2:
                txt = ax.text(0.02, 0.98, text, transform=ax.transAxes,
                              ha="left", va="top", fontsize=7.5, color=sig_col,
                              bbox=dict(facecolor=palette["PNL"],
                                        edgecolor="none", alpha=0.7, pad=3))
    if txt is not None and annot_sink is not None and metric_name:
        annot_sink[metric_name] = (txt, _annot_text)


def _maybe_end_labels(ax, ends, colors, labels, *,
                      pad_frac=0.08, min_gap_frac=0.045):
    """Label a set of curves DIRECTLY at their right-hand ends, each in its own
    colour (the r-graph-gallery preference — the eye follows a line to its label
    instead of bouncing to a legend key).

    Bails out (returns False, draws nothing) when any two end-points are closer
    than ``min_gap_frac`` of the y-range — the labels would collide and overlap,
    so the caller keeps the shared bottom legend instead.  On success it pads the
    right x-limit by ``pad_frac`` so the labels are not clipped, and returns True.
    """
    pts = [(xe, ye, c, l) for (xe, ye), c, l in zip(ends, colors, labels)
           if xe is not None and ye is not None
           and np.isfinite(xe) and np.isfinite(ye)]
    if len(pts) < 2:
        return False
    y0, y1 = ax.get_ylim()
    yr = (y1 - y0) or 1.0
    ys = sorted(p[1] for p in pts)
    if any((b - a) < min_gap_frac * yr for a, b in zip(ys, ys[1:])):
        return False                       # collision → keep the shared legend
    x0, x1 = ax.get_xlim()
    xr = (x1 - x0) or 1.0
    ax.set_xlim(x0, x1 + xr * pad_frac)
    for xe, ye, c, l in pts:
        ax.text(xe + xr * 0.012, ye, str(l), color=c, fontsize=7.5,
                va="center", ha="left", clip_on=False, zorder=6)
    return True


# Qualitative colours for the time-point series in two-factor interaction plots.
_TP_SERIES_COLORS = ["#3b6ed8", "#f78166", "#56d364", "#d2a8ff",
                     "#ffa657", "#79c0ff", "#e3b341", "#ff7b72"]


def _stars_of(p):
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _twoway_headline(twoway_df, metric):
    """Pull the two-way mixed-ANOVA **Interaction** (group × time) and **Group**
    main-effect p-values for `metric`, so an interaction plot can show the "do
    the groups differ when both group AND time are taken into account" result.
    Returns a dict, or None when unavailable (pingouin missing, <2 groups/time
    points, or the metric wasn't fitted)."""
    if twoway_df is None or not len(twoway_df):
        return None
    try:
        a = twoway_df[(twoway_df["section"] == "anova")
                      & (twoway_df["metric"] == metric)]
    except Exception:
        return None
    if not len(a):
        return None

    def _p(effect, *cols):
        r = a[a["effect"] == effect]
        if not len(r):
            return None
        row = r.iloc[0]
        for c in cols:
            v = row.get(c)
            try:
                v = float(v)
                if np.isfinite(v):
                    return v
            except (TypeError, ValueError):
                pass
        return None

    ip = _p("Interaction", "p_GG", "p_unc")          # GG-corrected interaction
    gp = _p("group", "p_unc")                         # between-subjects main effect
    if ip is None and gp is None:
        return None
    return {"interaction_p": ip, "interaction_stars": _stars_of(ip),
            "group_p": gp, "group_stars": _stars_of(gp)}


def _gradient_line(ax, x0, y0, x1, y1, c0, c1, lw=1.8, zorder=3, n=48):
    """Draw a straight line (x0,y0)->(x1,y1) whose colour fades from c0 to c1,
    so a segment joining a group's consecutive time points starts in the first
    cell's colour and ends in the next cell's colour."""
    import matplotlib.colors as _mc
    from matplotlib.collections import LineCollection
    xs = np.linspace(x0, x1, n + 1)
    ys = np.linspace(y0, y1, n + 1)
    pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    a = np.array(_mc.to_rgba(c0))
    b = np.array(_mc.to_rgba(c1))
    t = ((np.arange(n) + 0.5) / n)[:, None]
    cols = a[None, :] * (1 - t) + b[None, :] * t
    ax.add_collection(LineCollection(segs, colors=cols, linewidth=lw,
                                     zorder=zorder, capstyle="round"))


def _interaction_plot(ax, summary_df, metric, group_order, tp_order,
                      group_colors, palette, ylabel="", headline=None,
                      card_colors=None, stats_config=None):
    """Group × time-point interaction plot: x = TIME POINTS (in the order the
    user assigned them), one mean±SEM line per group drawn in that group's
    assigned colour — a time-course view.  Cell-level points are jittered
    behind each mean.

    Significance is shown two ways, neither of which sits on the lines:
      * the BETWEEN-group difference at each time point — a label above each
        time-point cluster (independent Welch-t / Mann-Whitney + Hedges' g);
      * each group's CHANGE across time (first→last time point, paired by
        cell) — folded into that group's legend entry, so it never overlaps
        the data and stays one line per group no matter how many time points.
    """
    x = np.arange(len(tp_order))
    rng = np.random.default_rng(2)

    def _cells(grp, tp):
        return summary_df.loc[(summary_df["group"] == grp)
                              & (summary_df["timepoint"] == tp),
                              metric]

    # Per-group change across time (first→last time point, paired by cell).
    # Only used as the FALLBACK header when the two-way ANOVA headline isn't
    # available — otherwise we show the interaction / group p instead.
    change = []   # (text, colour, line-centre height)
    if headline is None and len(tp_order) >= 2:
        tp_a, tp_b = tp_order[0], tp_order[-1]
        for gi, grp in enumerate(group_order):
            col = (group_colors or {}).get(grp) \
                or _TP_SERIES_COLORS[gi % len(_TP_SERIES_COLORS)]
            s0 = _cells(grp, tp_a).set_axis(
                summary_df.loc[(summary_df["group"] == grp)
                               & (summary_df["timepoint"] == tp_a), "cell"])
            s1 = _cells(grp, tp_b).set_axis(
                summary_df.loc[(summary_df["group"] == grp)
                               & (summary_df["timepoint"] == tp_b), "cell"])
            common = s0.index.intersection(s1.index)
            if len(common) < 2:
                continue
            a = s0.loc[common].to_numpy(dtype=float)
            b = s1.loc[common].to_numpy(dtype=float)
            p, stars = _PT(a, b, stats_config)
            if not np.isfinite(p):
                continue
            g = _paired_hedges_g(a, b)
            p_str = (f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}")
            t = f"{p_str}  {stars}"
            if g is not None and np.isfinite(g):
                t += f"    g = {g:+.2f}"
            change.append((t, col, 0.5 * (float(np.nanmean(a)) + float(np.nanmean(b)))))

    for gi, grp in enumerate(group_order):
        g_col = (group_colors or {}).get(grp) \
            or _TP_SERIES_COLORS[gi % len(_TP_SERIES_COLORS)]

        def _cell_col(tp):
            # Per-(group × time point) cell colour so the dots match the
            # bottom-legend entries; fall back to the group colour.
            if card_colors:
                c = card_colors.get((grp, tp))
                if c:
                    return c
            return g_col

        means, sems, pt_cols = [], [], []
        for ti, tp in enumerate(tp_order):
            c_cell = _cell_col(tp)
            pt_cols.append(c_cell)
            vals = _cells(grp, tp).to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                means.append(float(np.mean(vals)))
                sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                            if len(vals) > 1 else 0.0)
                ax.scatter(np.full(len(vals), x[ti]) + rng.uniform(-0.06, 0.06, len(vals)),
                           vals, color=c_cell, s=12, alpha=0.6, zorder=2)
            else:
                means.append(np.nan); sems.append(0.0)
        means = np.asarray(means, dtype=float)
        sems = np.asarray(sems, dtype=float)
        # Line joining a group's consecutive time points: a colour gradient from
        # one cell's colour to the next (e.g. blue PRE → orange POST) when we
        # have per-cell colours, otherwise a plain group-coloured line.
        for ti in range(len(tp_order) - 1):
            if np.isfinite(means[ti]) and np.isfinite(means[ti + 1]):
                if card_colors:
                    _gradient_line(ax, x[ti], means[ti], x[ti + 1], means[ti + 1],
                                   pt_cols[ti], pt_cols[ti + 1], lw=1.8, zorder=3)
                else:
                    ax.plot([x[ti], x[ti + 1]], [means[ti], means[ti + 1]],
                            color=g_col, lw=1.8, zorder=3)
        # Mean marker + SEM bar at each time point, in that cell's colour.
        for ti, tp in enumerate(tp_order):
            if np.isfinite(means[ti]):
                ax.errorbar(x[ti], means[ti], yerr=sems[ti], color=pt_cols[ti],
                            marker="o", ms=5, capsize=3, lw=0, zorder=4,
                            label=(str(grp) if (ti == 0 and not card_colors)
                                   else None))
    ax.set_xticks(x)
    ax.set_xticklabels(tp_order)
    ax.set_xlim(-0.35, len(tp_order) - 0.65)   # pad so end markers aren't clipped
    ax.set_xlabel("Time point")
    ax.set_ylabel(ylabel)
    # Plain group key (names + colour swatch only — stats are the centred labels
    # above).  Bottom-CENTRE: the dots cluster at the PRE/POST edges, so the
    # horizontal centre is the clearest spot for both the legend and the labels.
    ax.legend(frameon=False, loc="lower center", fontsize=8, title="Group",
              title_fontsize=8, ncol=len(group_order))

    # Stats header in a clear band ABOVE all the data.  Prefer the two-way
    # mixed-ANOVA result — the INTERACTION (group × time) answers "do the groups
    # differ when both group and time are taken into account" (e.g. both arms
    # drop but the drug drops more), plus the Group main effect.  Fall back to
    # the per-group PRE→POST change labels when the ANOVA isn't available.
    x_mid = (len(tp_order) - 1) / 2.0
    ymin, ymax = ax.get_ylim()
    span = (ymax - ymin) or 1.0

    def _pstr(p):
        return (f"p = {p:.1e}" if p < 0.001 else f"p = {p:.3f}")

    if headline is not None:
        lines = []
        if headline.get("interaction_p") is not None:
            lines.append("Group × Time interaction:  "
                         f"{_pstr(headline['interaction_p'])}  "
                         f"{headline['interaction_stars']}")
        if headline.get("group_p") is not None:
            lines.append("Group (overall):  "
                         f"{_pstr(headline['group_p'])}  {headline['group_stars']}")
        if lines:
            col = palette.get("SIG", palette.get("TXT", "#e0e0e0"))
            ax.set_ylim(ymin, ymax + (0.05 + 0.07 * len(lines)) * span)
            y = ymax + 0.03 * span
            for t in lines:
                ax.text(x_mid, y, t, ha="center", va="bottom", fontsize=8.5,
                        color=col, fontweight="bold")
                y += 0.07 * span
    elif change:
        ax.set_ylim(ymin, ymax + (0.05 + 0.075 * len(change)) * span)
        y = ymax + 0.03 * span
        for t, col, _h in change:           # stack upward, above all data
            ax.text(x_mid, y, t, ha="center", va="bottom", fontsize=8,
                    color=col, fontweight="bold")
            y += 0.075 * span


def _render_logd_facets(fig, subplotspec, facets, thr, pal, title,
                        threshold_label="", xlim=(-5.0, 1.0)):
    """Render the LogD distribution as small-multiple facets.

    Each facet overlays the pooled-per-track KDE(s) for its series (e.g. PRE
    vs POST for one drug) AND a strip of **per-replicate median dots** — one
    dot per cell — so the figure shows the distribution SHAPE while staying
    honest about the replicate-level data (a pooled per-track density alone
    can be dominated by a few high-track cells; the dots reveal the true n and
    spread).  The mobile/immobile threshold is drawn as a vertical guide.

    `facets`: list of (facet_title, facet_title_color, series) where each
    `series` item is
        (color, pooled_per_track_logD, [per_cell_medians], label, dashed).
    Facets share the x-axis (log₁₀ D).
    """
    from scipy import stats as _stats
    xlo, xhi = xlim
    xk = np.linspace(xlo, xhi, 300)
    facets = [f for f in facets if f[2]]            # drop empty facets
    if not facets:
        ax = fig.add_subplot(subplotspec)
        ax.axis("off")
        ax.text(0.5, 0.5, "No diffusion data", ha="center", va="center",
                color=pal.get("MUT", "#9aa4b2"), transform=ax.transAxes)
        return
    # A DEDICATED strip above the facets holds the title + a neutral key, so the
    # legend can never overlap a density curve — its placement no longer depends
    # on the distribution's shape (which broke an in-axes legend on real data).
    from matplotlib.lines import Line2D
    nF = len(facets)
    sub = subplotspec.subgridspec(nF + 1, 1,
                                  height_ratios=[0.7] + [1.0] * nF, hspace=0.16)

    lax = fig.add_subplot(sub[0]); lax.axis("off")
    lax.set_title(title)
    key = pal.get("MUT", "#9aa4b2")
    handles = []
    if any(d for (_ft, _fc, ser) in facets for (*_s, d) in ser):  # PRE/POST present
        handles += [
            Line2D([0], [0], color=key, lw=1.5, ls="-", label="PRE"),
            Line2D([0], [0], color=key, lw=1.5, ls="--", label="POST"),
        ]
    handles.append(Line2D([0], [0], color=key, ls="none", marker="o",
                          mfc=key, markersize=6, label="per-cell median"))
    if threshold_label:
        handles.append(Line2D([0], [0], color=pal["GRD"], lw=0.9, ls="--",
                              label=threshold_label))
    leg = lax.legend(handles=handles, loc="center",
                     ncol=(2 if len(handles) >= 4 else len(handles)),
                     frameon=False, fontsize=6.8, handlelength=1.6,
                     columnspacing=1.4, labelspacing=0.35, borderaxespad=0.0)
    for _t in leg.get_texts():
        _t.set_color(pal["TXT"])

    for fi, (ftitle, fcolor, series) in enumerate(facets):
        ax = fig.add_subplot(sub[fi + 1])
        maxd = 1e-9
        for (scolor, pooled, _medians, slabel, dashed) in series:
            v = np.asarray(pooled, float); v = v[np.isfinite(v)]
            if len(v) >= 2 and np.ptp(v) > 1e-9:
                try:
                    y = _stats.gaussian_kde(v)(xk)
                except Exception:
                    y = None
                if y is not None:
                    maxd = max(maxd, float(np.nanmax(y)))
                    ax.plot(xk, y, color=scolor, lw=1.4,
                            ls=("--" if dashed else "-"), zorder=4, label=slabel)
                    ax.fill_between(xk, 0.0, y, color=scolor, alpha=0.13,
                                    linewidth=0, zorder=2)
        # ── per-replicate median dots, in a strip below the baseline ──────────
        band = maxd * 0.42
        ns = len(series)
        for si, (scolor, _pooled, medians, _slabel, dashed) in enumerate(series):
            meds = [m for m in (medians or []) if np.isfinite(m)]
            if not meds:
                continue
            y0 = (-0.5 * band if ns == 1
                  else -(0.30 + 0.55 * si / (ns - 1)) * band)
            ax.scatter(meds, np.full(len(meds), y0), s=16,
                       facecolor=("none" if dashed else scolor),
                       edgecolor=scolor, linewidth=1.0, alpha=0.9,
                       marker="o", zorder=6, clip_on=False)
        ax.axvline(thr, color=pal["GRD"], ls="--", lw=0.8, zorder=15)
        ax.axhline(0.0, color=pal["GRD"], lw=0.5, alpha=0.5, zorder=1)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(-band * 1.05, maxd * 1.18)
        ax.set_yticks([])
        if ftitle:
            ax.text(0.015, 0.93, ftitle, transform=ax.transAxes, ha="left",
                    va="top", fontsize=8, fontweight="bold", color=fcolor,
                    zorder=20)
        if fi < nF - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("log₁₀ D  (µm²/s)")


def _logd_kde_or_none(pooled, xk):
    """gaussian_kde(pooled) sampled on xk, or None when undefined."""
    from scipy import stats as _stats
    if pooled is None:
        return None
    v = np.asarray(pooled, float); v = v[np.isfinite(v)]
    if len(v) < 2 or np.ptp(v) < 1e-9:
        return None
    try:
        return _stats.gaussian_kde(v)(xk)
    except Exception:
        return None


def _render_logd_ridgeline(ax, per_card, thr, pal, mobile_d_threshold, xlim=(-5.0, 1.0)):
    """Classic ridgeline: one filled KDE per group, stacked with a vertical
    offset and directly labelled, plus a tick per replicate (per-cell median)
    on each ridge baseline so the honest n is still visible."""
    xlo, xhi = xlim
    xk = np.linspace(xlo, xhi, 300)
    dens = [(lbl, col, _logd_kde_or_none(pooled, xk), medians)
            for (lbl, col, pooled, medians) in per_card]
    maxd = max((float(np.nanmax(y)) for _, _, y, _ in dens if y is not None),
               default=1.0) or 1.0
    step = maxd * 0.55
    for i, (lbl, col, y, medians) in enumerate(reversed(dens)):
        base = i * step
        ax.axhline(base, color=pal["GRD"], lw=0.5, alpha=0.4, zorder=1)
        if y is not None:
            ax.fill_between(xk, base, base + y, color=col, alpha=0.75,
                            linewidth=0, zorder=2 + i)
            ax.plot(xk, base + y, color=pal["BG"], lw=0.8, zorder=2 + i)
        meds = [m for m in (medians or []) if np.isfinite(m)]
        if meds:
            ax.scatter(meds, np.full(len(meds), base), s=11, color=pal["BG"],
                       edgecolor=col, linewidth=0.8, zorder=12 + i, clip_on=False)
        ax.text(xlo, base, f" {lbl}", va="bottom", ha="left", fontsize=7.5,
                color=col, fontweight="bold", zorder=20)
    ax.axvline(thr, color=pal["GRD"], ls="--", lw=0.8, zorder=15)
    ax.set_yticks([])
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-step * 0.3, (len(dens) - 1) * step + maxd * 1.15)
    ax.set_xlabel("log₁₀ D  (µm²/s)")
    ax.set_title("LogD distribution  (ridgeline)")


def _render_logd_overlaid(ax, per_card, thr, pal, mobile_d_threshold, xlim=(-5.0, 1.0)):
    """All groups' KDEs overlaid on one axes (best for ≤3 groups)."""
    xlo, xhi = xlim
    xk = np.linspace(xlo, xhi, 300)
    bins = np.linspace(xlo, xhi, 31)
    for (lbl, col, pooled, _medians) in per_card:
        if pooled is None:
            continue
        y = _logd_kde_or_none(pooled, xk)
        if y is not None:
            ax.fill_between(xk, 0.0, y, color=col, alpha=0.18, linewidth=0,
                            zorder=2)
            ax.plot(xk, y, color=col, lw=1.6, label=lbl, zorder=3)
        else:
            v = np.asarray(pooled, float); v = v[np.isfinite(v)]
            counts, edges = np.histogram(v, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            frac = counts / counts.sum() if counts.sum() else counts
            ax.plot(centers, frac, "-o", color=col, label=lbl, ms=4, lw=1.2)
    ax.axvline(thr, color=pal["GRD"], ls="--", lw=0.8,
               label=f"D = {mobile_d_threshold:g} µm²/s")
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("log₁₀ D  (µm²/s)")
    ax.set_ylabel("Density")
    ax.set_title("LogD distribution  (overlaid)")
    ax.legend(frameon=False, loc="best", fontsize=7)


def _render_logd_violin(ax, per_card, thr, pal, mobile_d_threshold, xlim=(-5.0, 1.0)):
    """Per-group violins (log₁₀ D on y) with a per-cell median dot strip — a
    SuperPlot-style view that shows shape AND the replicate-level data."""
    valid = []
    for (lbl, col, pooled, medians) in per_card:
        if pooled is None:
            continue
        v = np.asarray(pooled, float); v = v[np.isfinite(v)]
        if len(v) >= 2 and np.ptp(v) > 1e-9:
            valid.append((lbl, col, v, medians))
    if not valid:
        ax.axis("off")
        ax.text(0.5, 0.5, "No diffusion data", ha="center", va="center",
                color=pal.get("MUT", "#9aa4b2"), transform=ax.transAxes)
        return
    positions = list(range(1, len(valid) + 1))
    parts = ax.violinplot([v for _, _, v, _ in valid], positions=positions,
                          showmeans=False, showextrema=False, widths=0.82)
    for pc, (_lbl, col, _v, _m) in zip(parts["bodies"], valid):
        pc.set_facecolor(col); pc.set_edgecolor(col); pc.set_alpha(0.38)
    rng = np.random.default_rng(0)
    for pos, (_lbl, col, _v, medians) in zip(positions, valid):
        meds = [m for m in (medians or []) if np.isfinite(m)]
        if meds:
            jit = (rng.random(len(meds)) - 0.5) * 0.24
            ax.scatter(np.full(len(meds), pos) + jit, meds, s=18, color=col,
                       edgecolor=pal["BG"], linewidth=0.6, zorder=6)
    ax.axhline(thr, color=pal["GRD"], ls="--", lw=0.8, zorder=2,
               label=f"D = {mobile_d_threshold:g} µm²/s")
    ax.set_xticks(positions)
    ax.set_xticklabels([l for l, _, _, _ in valid], rotation=30, ha="right",
                       fontsize=7)
    ax.set_ylabel("log₁₀ D  (µm²/s)")
    ax.set_ylim(xlim)
    ax.set_title("LogD distribution  (violins + per-cell medians)")
    ax.legend(frameon=False, loc="lower right", fontsize=7)


# ── Style picker: descriptions + a live preview for the Preferences menu ──────
LOGD_STYLE_DESCRIPTIONS = {
    "overlaid": (
        "Overlaid KDEs — every group's curve on one axes. Best for directly "
        "comparing overall shape and peak position across a few groups at a "
        "glance. Gets crowded (“spaghetti”) with many groups."),
    "faceted": (
        "Faceted (per-replicate) — one panel per group, PRE vs POST overlaid, "
        "with a per-cell median dot strip. Best for paired designs and for being "
        "honest about replicate count; the most information-dense."),
    "ridgeline": (
        "Ridgeline — filled KDEs stacked with an offset, directly labelled. Best "
        "for comparing many groups' shapes compactly and spotting multi-modality. "
        "The vertical offset makes exact peak heights harder to compare."),
    "violin": (
        "Violins + points — a violin per group with a per-cell median dot strip "
        "(SuperPlot style). Best for showing each group's spread plus the "
        "replicate-level data; pairs naturally with the stats panels."),
}


def _example_logd_data(rng=None):
    """Small illustrative synthetic LogD data for the Preferences preview:
    two 'drugs' A/B × PRE/POST, where drug B immobilises (shifts left) at POST.
    Returns (per_card, facets) matching the render helpers' shapes."""
    rng = rng or np.random.default_rng(7)
    cols = {("A", "PRE"): "#4c8edb", ("A", "POST"): "#e0922f",
            ("B", "PRE"): "#3fa45b", ("B", "POST"): "#d8534f"}

    def _pool(centers, weights, n=900, sd=0.32):
        c = np.array(centers)
        comp = rng.choice(len(centers), size=n, p=weights)
        return np.clip(rng.normal(c[comp], sd), -5.0, 1.0)

    def _meds(center, n=6, sd=0.13):
        return list(np.clip(rng.normal(center, sd, n), -5.0, 1.0))

    specs = {
        ("A", "PRE"):  ([-1.0], [1.0], -1.0),
        ("A", "POST"): ([-1.05], [1.0], -1.05),
        ("B", "PRE"):  ([-1.0], [1.0], -1.0),
        ("B", "POST"): ([-2.2, -1.0], [0.65, 0.35], -1.9),   # immobilised shift
    }
    per_card, facets_map = [], {"A": [], "B": []}
    for (drug, tp), (centers, weights, mc) in specs.items():
        pooled = _pool(centers, weights)
        meds = _meds(mc)
        col = cols[(drug, tp)]
        per_card.append((f"{drug} / {tp}", col, pooled, meds))
        facets_map[drug].append((col, pooled, meds, tp, tp == "POST"))
    grp_col = {"A": "#4c8edb", "B": "#3fa45b"}
    facets = [(d, grp_col[d], facets_map[d]) for d in ("A", "B")]
    return per_card, facets


def render_logd_preview(fig, style, theme="Dark"):
    """Render a small illustrative example of a LogD-distribution `style` into
    `fig` (used by the Preferences preview).  UI-free; safe to call repeatedly."""
    import matplotlib.pyplot as _plt
    if style not in ("faceted", "ridgeline", "overlaid", "violin"):
        style = "overlaid"
    pal = _theme_palette(theme)
    fig.clear()
    fig.set_facecolor(pal["BG"])
    per_card, facets = _example_logd_data()
    thr = float(np.log10(0.05))
    thr_lbl = "D = 0.05 µm²/s"
    mut = pal.get("MUT", "#9aa4b2")
    with _plt.rc_context({
            "font.size": 6.5, "axes.titlesize": 8, "axes.labelsize": 7,
            "xtick.labelsize": 6, "ytick.labelsize": 6,
            "axes.facecolor": pal["PNL"], "figure.facecolor": pal["BG"],
            "text.color": pal["TXT"], "axes.labelcolor": pal["TXT"],
            "axes.edgecolor": pal["GRD"], "xtick.color": mut,
            "ytick.color": mut, "axes.titlecolor": pal["TXT"]}):
        if style == "ridgeline":
            _render_logd_ridgeline(fig.add_subplot(111), per_card, thr, pal, 0.05)
        elif style == "violin":
            _render_logd_violin(fig.add_subplot(111), per_card, thr, pal, 0.05)
        elif style == "faceted":
            _render_logd_facets(fig, fig.add_gridspec(1, 1)[0], facets, thr, pal,
                                "LogD distribution", threshold_label=thr_lbl)
        else:
            _render_logd_overlaid(fig.add_subplot(111), per_card, thr, pal, 0.05)
        # Tidy small previews: drop axes legends for the non-faceted styles
        # (the faceted strip legend is part of its layout).
        if style != "faceted":
            for _ax in fig.axes:
                _lg = _ax.get_legend()
                if _lg is not None:
                    _lg.remove()
    try:
        fig.tight_layout(pad=0.4)
    except Exception:
        pass


def comparison_grid(n):
    """(rows, cols) the comparison figure packs `n` panels into — the single
    source of truth shared with the UI panel-picker's live grid count."""
    c = 3 if n > 4 else 2
    return ((n + c - 1) // c, c)


def compute_report(groups, *, mobile_d_threshold=MOBILE_D_THRESHOLD_DEFAULT,
                   stats_config=None, use_native=False, progress_cb=None):
    """Load + compute everything a comparison needs that is INDEPENDENT of
    theme, graph style and panel selection: per-replicate scalars
    (`summary_df`), factor levels, colours and the two-way ANOVA, plus an
    (initially empty) memo cache for the expensive effect-size statistics.
    Returns a `ReportData` the controller caches and hands to `render_report`,
    so a style/theme change re-draws with no recompute.  Raises
    `CompareInputError` on bad input (fewer than 2 groups, or a group with no
    loadable folders)."""
    from firefly.analysis.fa_stats_config import normalize_stats_config
    # Reset the "already warned about a missing Δt" dedup per comparison run.
    # It's a module global (so _fi_or_default can warn once per stem within a
    # run), but if it persisted across runs a GUI re-run would suppress the
    # warning entirely, and stale stems would linger for the whole session (R4-3).
    _FI_DEFAULT_WARNED.clear()
    cfg = normalize_stats_config(stats_config)
    # Bar-panel annotation handles, so the optional across-metric correction
    # post-pass can update the on-figure stars to agree with the CSV.
    panel_annots = {}

    if len(groups) < 2:
        raise CompareInputError(
            f"A comparison needs at least 2 groups; only {len(groups)} was given. "
            "Add another group on the Compare tab.")


    n_groups = len(groups)
    # `group_factor` is the raw group label of each card (the between-subjects
    # factor); `timepoints_per_card` is the optional within-subjects factor.
    # When ANY card carries a time point we enter two-factor (group × time
    # point) mode: several cards can share a group label but differ in time
    # point.  `labels` is the per-card DISPLAY label (group / time point) used
    # in legends and the suptitle so duplicated group names stay distinct.
    group_factor = [g.get("label", f"Group {i+1}") for i, g in enumerate(groups)]
    timepoints_per_card = [str(g.get("timepoint", "")).strip() for g in groups]
    timepoint_tokens = sorted({t for t in timepoints_per_card if t})
    # Two-factor (group × time) only makes sense with ≥2 DISTINCT time points.
    # A single shared time point (e.g. every card tagged "Pre") is just a
    # one-factor group comparison — treat it exactly as if no time points were
    # set, otherwise the interaction plots degenerate to a single x-position and
    # render weirdly.
    # Two-factor (group × time) needs ≥2 DISTINCT time points AND ≥2 distinct
    # groups — a group × time interaction is undefined with a single group.
    # When one group name spans ≥2 time points (e.g. "Munc 18" tagged Pre-drug
    # and Post-drug on separate cards) the interaction degenerates to a single
    # group line and the subject-pairing drops every unpaired cell, blanking the
    # figure.  Detect that and fold it down to a plain one-way comparison of the
    # (group · time) cells — exactly what the scalar stats already report.
    distinct_tp = len(timepoint_tokens) >= 2
    two_factor = distinct_tp and len(set(group_factor)) >= 2
    labels = [(f"{group_factor[i]} / {timepoints_per_card[i]}"
               if (distinct_tp and timepoints_per_card[i]) else group_factor[i])
              for i in range(n_groups)]
    if distinct_tp and not two_factor:
        # single group across ≥2 time points → treat each card as its own group
        # (drop the now single-level time factor) so the one-way panels render.
        group_factor = list(labels)
        timepoints_per_card = ["" for _ in group_factor]
        timepoint_tokens = []
    colors   = [g.get("color", "#3b6ed8")     for g in groups]
    folder_lists = [list(g["folders"]) for g in groups]

    # ── Load summaries for all groups ─────────────────────────────────────────
    all_summaries = [[] for _ in groups]
    skipped = [[] for _ in groups]   # per group: (folder, reason) for failures
    total = sum(len(f) for f in folder_lists)
    done = 0
    for gi, folders in enumerate(folder_lists):
        for f in folders:
            if progress_cb:
                progress_cb(done, total, f"Loading: {os.path.basename(f)}")
            try:
                all_summaries[gi].append(
                    load_summary_from_folder(f, use_native=use_native))
            except Exception as e:
                # Classify so the user gets an actionable reason, not a stack
                # trace.  The #1 cause is an unmounted external drive.
                if not os.path.exists(f):
                    reason = "folder not found — is the drive/network share connected?"
                elif not os.path.isdir(f):
                    reason = "not a folder"
                else:
                    reason = str(e)
                skipped[gi].append((f, reason))
                print(f"  Skipping {f}: {reason}")
            done += 1

    empty_groups = [i for i, ss in enumerate(all_summaries) if len(ss) == 0]
    if empty_groups:
        lines = ["The comparison can't run — these group(s) have no valid "
                 "analysis folders:"]
        for i in empty_groups:
            lines.append(f"\n• {labels[i]}:")
            if not skipped[i]:
                lines.append("    (no folders were added)")
            for f, reason in skipped[i]:
                lines.append(f"    – {os.path.basename(f.rstrip(os.sep)) or f}: {reason}")
        lines.append(
            "\nTip: add each analysis OUTPUT folder (the one containing a "
            "'firefly_extras' subfolder), not the raw data or a parent folder — "
            "and reconnect the drive if a path is missing.")
        raise CompareInputError("\n".join(lines))

    if progress_cb:
        progress_cb(total, total, "Computing scalars and rendering...")

    # ── Compute per-folder scalars (one row per replicate) ────────────────────
    # `group` holds the raw group factor; `timepoint` is the (optional) within
    # factor; `cell` is the subject key (the stem with the time-point token
    # stripped) used to pair the same cell across time points.
    summary_rows = []
    def _row(group_label, timepoint, summary):
        p = summary["params"]
        fi = _fi_or_default(p, summary.get("stem", ""))
        d = summary["diffusion"]
        stem = summary["stem"]
        cell, _matched = (fa_twoway.derive_subject_key(stem, timepoint_tokens)
                          if two_factor else (stem, True))
        # van Hove non-Gaussian alpha2 and VACF persistence are both
        # dimensionless ratios (scale- and time-invariant), so we can compute
        # them per replicate straight from the (pixel-unit) tracks with
        # px=1/dt=1 and get the identical value — no dependence on the saved
        # JSON extras or on the per-folder calibration.
        trk = summary["tracks"]
        try:
            _vh = compute_van_hove(trk, 1.0) if trk is not None else None
            nongauss_alpha2 = float(_vh["non_gaussian_alpha2"]) if _vh else np.nan
        except Exception:
            nongauss_alpha2 = np.nan
        try:
            _vc = compute_vacf(trk, 1.0, 1.0) if trk is not None else None
            vacf_persistence = float(_vc["persistence"]) if _vc else np.nan
        except Exception:
            vacf_persistence = np.nan
        return {
            "group":            group_label,
            "timepoint":        timepoint,
            "cell":             cell,
            "folder":           summary["folder"],
            "stem":             stem,
            "n_tracks":         len(d) if d is not None else 0,
            "auc_msd":          _msd_auc(summary["ensemble_msd"], fi),
            "mob_immob_ratio":  _mob_immob_ratio(d, mobile_d_threshold),
            "median_D":         float(d["D"].median()) if d is not None and "D" in d.columns else np.nan,
            "median_alpha":     float(d["alpha"].median()) if d is not None and "alpha" in d.columns else np.nan,
            "mean_track_length_s": float(_track_lengths(summary["tracks"], fi).mean())
                                   if summary["tracks"] is not None else np.nan,
            "nongauss_alpha2":  nongauss_alpha2,
            "vacf_persistence": vacf_persistence,
        }
    for gi, summaries in enumerate(all_summaries):
        for s in summaries:
            summary_rows.append(_row(group_factor[gi], timepoints_per_card[gi], s))
    summary_df = pd.DataFrame(summary_rows)

    # Ordered factor levels.  Time points keep ASSIGNMENT order (the order the
    # user entered them across cards) so the x-axis reads e.g. Pre → Post, not
    # alphabetical.  Each group keeps its user-assigned colour for its line.
    group_order = list(dict.fromkeys(group_factor))
    tp_order = list(dict.fromkeys(t for t in timepoints_per_card if t))

    # Per-(group × time point) cell colours for the two-factor interaction plots.
    # Each cell gets its OWN colour so the dots line up with the bottom-legend
    # entries (e.g. "DMSO / PRE" blue, "DMSO / POST" orange), and the line that
    # joins a group's time points fades between them.  The GUI commonly reuses a
    # single colour per group across its time points, which collapses the legend
    # to one colour per group — so when the incoming per-card colours aren't all
    # distinct, fan them out across a qualitative palette and write them back
    # into `colors` so the shared bottom legend shows the SAME colours.
    card_colors = {}
    if two_factor:
        _QUAL = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2",
                 "#9d755d", "#ff9da6", "#79706e", "#bab0ac", "#d67195",
                 "#86bcb6", "#fabfd2", "#b4d2b1", "#c7b0c1", "#f2cf5b"]
        if len(set(colors)) < n_groups:        # not already all-distinct
            colors = [_QUAL[i % len(_QUAL)] for i in range(n_groups)]
        for i in range(n_groups):
            card_colors[(group_factor[i], timepoints_per_card[i])] = colors[i]

    group_colors = {}
    for i, gf in enumerate(group_factor):
        group_colors.setdefault(gf, colors[i])

    # With many groups the full names won't fit on a bar x-axis (they rotate
    # into an unreadable smear), so use short numeric tokens on the axis and
    # carry the full names in the shared legend.  Threshold: >4 groups.
    many_groups = n_groups > 4
    bar_xticks = [str(i + 1) for i in range(n_groups)] if many_groups else labels

    # Per-metric statistics dict — populated as panels render
    stats_records = {}

    # Two-way mixed ANOVA computed UP FRONT (paired: between=group,
    # within=timepoint, subject=cell) so the interaction panels can show the
    # interaction / group p as they render.  Reused by the report block later.
    twoway_df, twoway_msg, pair_warn, paired_df = None, None, None, None
    if two_factor:
        paired_df, pair_warn, _dropped = fa_twoway.validate_pairing(summary_df)
        if pair_warn:
            print(f"  Two-way pairing: {pair_warn}")
        # The two-way ANOVA is a SECONDARY headline stat — it must never be able
        # to blank the primary figure.  If it fails (singular/underpowered design,
        # pingouin quirk, …) skip it and let every panel render regardless.
        try:
            twoway_df, twoway_msg = fa_twoway.compute_twoway_anova(paired_df, stats_config=cfg)
        except Exception as e:
            twoway_df, twoway_msg = None, f"skipped ({type(e).__name__}: {e})"
        print(f"  Two-way ANOVA: {twoway_msg}")

    return ReportData(
        cfg=cfg, groups=groups, n_groups=n_groups, group_factor=group_factor,
        timepoints_per_card=timepoints_per_card, timepoint_tokens=timepoint_tokens,
        distinct_tp=distinct_tp, two_factor=two_factor, labels=labels, colors=colors,
        folder_lists=folder_lists, all_summaries=all_summaries, skipped=skipped,
        summary_df=summary_df, group_order=group_order, tp_order=tp_order,
        card_colors=card_colors, group_colors=group_colors, many_groups=many_groups,
        bar_xticks=bar_xticks, twoway_df=twoway_df, twoway_msg=twoway_msg,
        pair_warn=pair_warn, paired_df=paired_df,
        mobile_d_threshold=mobile_d_threshold)


def _draw_report(rd, *, output_dir=None, output_stem="comparison",
                 panels=None, theme="Dark", pdf_report=True,
                 logd_plot_style="overlaid", msd_plot_style="mean_faceted",
                 msd_err="SEM", auc_plot_style="paired", group_style="box_points",
                 panel_styles=None,
                 logd_clip_d_min=1e-5, logd_clip_d_max=10.0, progress_cb=None):
    """Compare N≥2 groups of analysis output folders and render a multi-panel
    figure, summary CSV, statistics CSV and combined PDF report.

    Parameters
    ----------
    groups : list[dict]
        [{"folders": [path, ...], "label": "Pre", "color": "#000000"}, ...]
    output_dir : str or None
        Where to save the figure / CSVs / PDF report.  If None, nothing is
        saved to disk and only the figure is returned.
    panels : set[str] or None
        Subset of panels to render.  Default: all of {"msd", "auc",
        "logd_dist", "mob_immob", "motion_classes", "track_length",
        "jdd", "dwell_cdf", "turning_angles"}.
    theme : str
        Figure theme — "Dark" (default), "Light" or "Publication".
    pdf_report : bool
        If True (default) and output_dir is given, also write a multi-page
        PDF report bundling the figure, parameters, folder lists and stats.
    progress_cb : callable or None
        Optional callback(done:int, total:int, msg:str) for UI progress.

    Returns
    -------
    fig         : matplotlib.figure.Figure
    summary_df  : pandas.DataFrame  — per-replicate scalar metrics
    stats       : dict[str, dict]   — per-metric omnibus + pairwise tests
    """
    import matplotlib.pyplot as plt
    # Per-panel comparison mark (box+points / violin / bar), keyed by panel key;
    # any panel not overridden falls back to the global `group_style`.
    panel_styles = panel_styles or {}
    def _pstyle(key):
        return panel_styles.get(key, group_style)
    cfg = rd.cfg
    groups = rd.groups
    n_groups = rd.n_groups
    group_factor = rd.group_factor
    timepoints_per_card = rd.timepoints_per_card
    timepoint_tokens = rd.timepoint_tokens
    distinct_tp = rd.distinct_tp
    two_factor = rd.two_factor
    labels = rd.labels
    colors = rd.colors
    folder_lists = rd.folder_lists
    all_summaries = rd.all_summaries
    skipped = rd.skipped
    summary_df = rd.summary_df
    group_order = rd.group_order
    tp_order = rd.tp_order
    card_colors = rd.card_colors
    group_colors = rd.group_colors
    many_groups = rd.many_groups
    bar_xticks = rd.bar_xticks
    twoway_df = rd.twoway_df
    twoway_msg = rd.twoway_msg
    pair_warn = rd.pair_warn
    paired_df = rd.paired_df
    mobile_d_threshold = rd.mobile_d_threshold
    # Per-panel annotation handles + the returned per-metric stats dict are built
    # as the panels draw (the returned `stats` only ever covers RENDERED panels).
    panel_annots = {}
    stats_records = {}
    if panels is None:
        panels = {"msd", "auc", "logd_dist", "mob_immob", "motion_classes",
                  "track_length", "jdd", "dwell_cdf", "turning_angles",
                  "radial_dist", "van_hove", "vacf"}

    # ── Render the figure ────────────────────────────────────────────────────
    panel_order = ["msd", "auc", "logd_dist", "mob_immob",
                   "motion_classes", "track_length", "track_count",
                   "jdd", "dwell_cdf", "turning_angles", "radial_dist",
                   "van_hove", "vacf"]
    enabled = [p for p in panel_order if p in panels]
    n_plots = len(enabled)
    if n_plots == 0:
        raise RuntimeError("No panels enabled")
    print(f"  Compare: rendering {n_plots} panel(s): {enabled}")
    if "radial_dist" not in panels:
        print(f"  Compare: 'radial_dist' NOT in requested panels — "
              f"check the 'Radial distribution (polar)' tickbox in the "
              f"Compare tab to include it.")
    nrows, ncols = comparison_grid(n_plots)

    # Quick-glance summary-band geometry (the band itself is drawn after the
    # suptitle, below).  Size it in ABSOLUTE inches and grow the figure height to
    # fit, so the panels keep their size whatever the group / panel count.
    # One column for a few groups (full detail); two columns beyond that (the
    # entries are long, so 3 columns collide horizontally on a narrow figure).
    band_ncol = 1 if n_groups <= 3 else 2
    band_nrow = (n_groups + band_ncol - 1) // band_ncol
    band_compact = band_ncol > 1
    band_fs = 9 if n_groups <= 8 else 8
    band_row_in = 0.26                       # inches per band row
    band_h_in = band_nrow * band_row_in + 0.46   # band + gap under the suptitle
    base_h = nrows * 3.6

    pal = _theme_palette(theme)
    plt.rcParams.update({
        "text.color":      pal["TXT"], "axes.labelcolor": pal["TXT"],
        "xtick.color":     pal["TXT"], "ytick.color":     pal["TXT"],
        "axes.titlecolor": pal["TXT"],
        "axes.edgecolor":  pal["GRD"], "axes.facecolor":  pal["PNL"],
        "figure.facecolor": pal["BG"], "figure.edgecolor": pal["BG"],
        "savefig.facecolor": pal["BG"], "savefig.edgecolor": pal["BG"],
        "grid.color":      pal["GRD"], "grid.alpha": 0.22,
        "font.family":     pal["FONT"],
        "legend.facecolor": pal["PNL"], "legend.edgecolor": pal["GRD"],
    })

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, base_h + band_h_in),
                             facecolor=pal["BG"])
    axes = np.array(axes).reshape(-1)
    for ax in axes[n_plots:]:
        ax.axis("off")

    panel_idx = 0
    def _next_ax():
        nonlocal panel_idx
        ax = axes[panel_idx]; panel_idx += 1
        return ax

    def _zip_groups():
        """Iterator: (label, summaries, color) for each group."""
        for i in range(n_groups):
            yield labels[i], all_summaries[i], colors[i]

    # ── 1. MSD comparison (styled: mean±err faceted / individual / overlaid) ──
    #   Style comes from Preferences (figures/msd_style); the error type comes
    #   from the Analysis tab (msd_err).  Curves are grouped by condition ×
    #   timepoint so a paired (pre/post-style) design overlays both series per
    #   facet, while a single-timepoint design draws one.
    if "msd" in panels:
        from firefly.analysis import fa_group_figures as _gfig
        ax = _next_ax()
        ss = ax.get_subplotspec(); ax.remove()
        msd_by_gt = {}                      # {group: {timepoint: [curve arrays]}}
        tref = None
        for gi in range(n_groups):
            grp = group_factor[gi]
            tp = timepoints_per_card[gi] or ""
            for s in all_summaries[gi]:
                e = s.get("ensemble_msd") if hasattr(s, "get") else s["ensemble_msd"]
                if e is None:
                    continue
                fi = _fi_or_default(s["params"], s.get("stem", ""))
                t = e["lag_frame"].values * fi
                y = e["msd_um2"].values
                order = np.argsort(t)
                t, y = t[order], y[order]
                if tref is None:
                    tref = t
                if len(t) != len(tref) or not np.allclose(t, tref):
                    y = np.interp(tref, t, y)
                msd_by_gt.setdefault(grp, {}).setdefault(tp, []).append(y)
        if tref is not None and msd_by_gt:
            data = {g: {tp: np.vstack(v) for tp, v in tps.items()}
                    for g, tps in msd_by_gt.items()}
            groups_order = ([g for g in group_order if g in data]
                            if two_factor else list(data))
            tp_seen = [tp for tps in msd_by_gt.values() for tp in tps]
            tp_ord = ([t for t in tp_order if t in tp_seen] if two_factor
                      else sorted(set(tp_seen)))
            gtheme = {"bg": pal["PNL"], "fg": pal["TXT"], "grid": pal["GRD"],
                      "spine": pal["GRD"], "muted": pal["MUT"]}
            _gfig.draw_msd(fig, ss, groups_order, data, tref,
                           style=(msd_plot_style if msd_plot_style in
                                  ("mean_faceted", "individual", "overlaid")
                                  else "mean_faceted"),
                           err=msd_err,
                           tp_order=tp_ord,
                           group_colors={g: group_colors.get(g) for g in groups_order},
                           theme=gtheme, xlabel="Time delta (s)")
        else:
            _ax = fig.add_subplot(ss)
            _ax.text(0.5, 0.5, "no MSD curves exported", ha="center", va="center",
                     transform=_ax.transAxes, color=pal["MUT"], fontsize=10)
            _ax.set_xticks([]); _ax.set_yticks([])

    # ── 2. MSD-AUC change (styled: paired lines / Δ box) ──────────────────────
    #   For a 2-timepoint design, the AUC panel shows the per-dish change across
    #   timepoints in the Preferences style (paired lines + per-group p, or a Δ
    #   box with an omnibus test).  A single-timepoint design keeps the group bar.
    if "auc" in panels:
        ax = _next_ax()
        if two_factor and len(tp_order) >= 2:
            from firefly.analysis import fa_group_figures as _gfig
            ss = ax.get_subplotspec(); ax.remove()
            paired = {}
            for grp in group_order:
                a = summary_df.loc[(summary_df["group"] == grp)
                                   & (summary_df["timepoint"] == tp_order[0])
                                   ].groupby("cell")["auc_msd"].mean()
                b = summary_df.loc[(summary_df["group"] == grp)
                                   & (summary_df["timepoint"] == tp_order[1])
                                   ].groupby("cell")["auc_msd"].mean()
                common = a.index.intersection(b.index)
                if len(common):
                    paired[grp] = {tp_order[0]: a.loc[common].to_numpy(float),
                                   tp_order[1]: b.loc[common].to_numpy(float)}
            style = auc_plot_style if auc_plot_style in ("paired", "delta") else "paired"
            if paired:
                groups_o = [g for g in group_order if g in paired]
                gtheme = {"bg": pal["PNL"], "fg": pal["TXT"], "grid": pal["GRD"],
                          "spine": pal["GRD"], "muted": pal["MUT"]}
                if style == "delta":
                    dd = [paired[g][tp_order[1]] - paired[g][tp_order[0]] for g in groups_o]
                    dd = [d for d in dd if len(d)]
                    stat_labels = ""
                    if len(dd) >= 2:
                        try:
                            from scipy.stats import kruskal
                            stat_labels = f"Kruskal–Wallis, p = {kruskal(*dd).pvalue:.2g}"
                        except Exception:
                            pass
                else:
                    stat_labels = {}
                    for g in groups_o:
                        try:
                            p, _stars = _PT(paired[g][tp_order[0]],
                                            paired[g][tp_order[1]], cfg)
                            stat_labels[g] = f"p = {p:.2g}" if p == p else ""
                        except Exception:
                            stat_labels[g] = ""
                _gfig.draw_auc_change(fig, ss, groups_o, paired, style=style,
                                      tp_order=list(tp_order), stat_labels=stat_labels,
                                      group_colors={g: group_colors.get(g) for g in groups_o},
                                      theme=gtheme, ylabel="MSD AUC")
            else:
                _ax = fig.add_subplot(ss)
                _ax.text(0.5, 0.5, "no paired AUC (unmatched timepoints)",
                         ha="center", va="center", transform=_ax.transAxes,
                         color=pal["MUT"], fontsize=9)
                _ax.set_xticks([]); _ax.set_yticks([])
        else:
            data = [summary_df.loc[summary_df["group"] == lbl, "auc_msd"].values
                    for lbl in labels]
            _bar_with_dots_n(ax, data, labels, colors, pal,
                             ylabel="AUC (µm²·s)",
                             record_stats=stats_records, metric_name="auc_msd", xtick_labels=bar_xticks, stats_config=cfg, annot_sink=panel_annots, style=_pstyle("auc"))
            ax.set_title("Area Under the Curve")

    # ── 3. LogD distribution (filled KDEs; ridgeline when many groups) ────────
    if "logd_dist" in panels:
        # LogD distribution — the rendering STYLE is chosen in Preferences
        # (Faceted / Ridgeline / Overlaid / Violin).  Every style is fed the
        # per-replicate MEDIANS so the pooled-per-track KDE can't masquerade as
        # the replicate-level truth (a few high-track cells would otherwise
        # dominate).  The immobile tail is clipped INTO range (not dropped) so
        # an immobilising drug effect stays visible.
        _styles = ("faceted", "ridgeline", "overlaid", "violin")
        style = logd_plot_style if logd_plot_style in _styles else "overlaid"
        ax = _next_ax()
        thr = np.log10(mobile_d_threshold)
        thr_lbl = f"D = {mobile_d_threshold:g} µm²/s"
        # D-coefficient clip range → log₁₀ bounds (defaults −5…1).  Applied to the
        # pooled distribution AND the per-cell medians, and to the axes below.
        _clip_lo = np.log10(logd_clip_d_min) if logd_clip_d_min and logd_clip_d_min > 0 else -5.0
        _clip_hi = np.log10(logd_clip_d_max) if logd_clip_d_max and logd_clip_d_max > 0 else 1.0
        _xlim = (_clip_lo, _clip_hi)

        def _cell_logd(s):
            d = s.get("diffusion") if hasattr(s, "get") else None
            if d is None or "D" not in getattr(d, "columns", []):
                return None
            vals = d["D"].values
            vals = vals[vals > 0]
            if not len(vals):
                return None
            return np.clip(np.log10(vals), _clip_lo, _clip_hi)

        def _gather(card_indices):
            pooled, medians = [], []
            for gi in card_indices:
                for s in all_summaries[gi]:
                    lg = _cell_logd(s)
                    if lg is None:
                        continue
                    pooled.append(lg)
                    medians.append(float(np.median(lg)))
            return (np.concatenate(pooled) if pooled else None), medians

        # One entry per card/group — used by ridgeline / overlaid / violin.
        per_card = []
        for gi in range(n_groups):
            pooled, medians = _gather([gi])
            per_card.append((labels[gi], colors[gi], pooled, medians))

        if style == "ridgeline":
            _render_logd_ridgeline(ax, per_card, thr, pal, mobile_d_threshold, xlim=_xlim)
        elif style == "overlaid":
            _render_logd_overlaid(ax, per_card, thr, pal, mobile_d_threshold, xlim=_xlim)
        elif style == "violin":
            _render_logd_violin(ax, per_card, thr, pal, mobile_d_threshold, xlim=_xlim)
        else:
            # ── Faceted (default): facet by drug, PRE vs POST overlaid, with a
            # per-replicate median dot strip beneath each density. ──
            ss = ax.get_subplotspec()
            ax.remove()
            facets = []
            if two_factor:
                for drug in group_order:
                    series = []
                    for ti, tp in enumerate(tp_order):
                        idx = [gi for gi in range(n_groups)
                               if group_factor[gi] == drug
                               and timepoints_per_card[gi] == tp]
                        pooled, medians = _gather(idx)
                        if pooled is not None:
                            scol = (card_colors.get((drug, tp))
                                    or group_colors.get(drug) or "#3b6ed8")
                            series.append((scol, pooled, medians, str(tp), ti > 0))
                    facets.append((str(drug),
                                   group_colors.get(drug, pal["TXT"]), series))
            elif n_groups <= 3:
                series = [(colors[gi], pld, med, labels[gi], False)
                          for gi in range(n_groups)
                          for pld, med in [_gather([gi])] if pld is not None]
                facets = [("", pal["TXT"], series)]
            else:
                for gi in range(n_groups):
                    pld, med = _gather([gi])
                    if pld is not None:
                        facets.append((labels[gi], colors[gi],
                                       [(colors[gi], pld, med, labels[gi], False)]))
            _render_logd_facets(fig, ss, facets, thr, pal,
                                "LogD distribution  (● per-cell median)",
                                threshold_label=thr_lbl, xlim=_xlim)

    # ── 4. Mobile/Immobile ratio bar ──────────────────────────────────────────
    if "mob_immob" in panels:
        ax = _next_ax()
        if two_factor:
            _interaction_plot(ax, summary_df, "mob_immob_ratio", group_order,
                              tp_order, group_colors, pal,
                              ylabel="Mobile/Immobile ratio",
                              headline=_twoway_headline(twoway_df, "mob_immob_ratio"),
                              card_colors=card_colors, stats_config=cfg)
        else:
            data = [summary_df.loc[summary_df["group"] == lbl, "mob_immob_ratio"].values
                    for lbl in labels]
            _bar_with_dots_n(ax, data, labels, colors, pal,
                             ylabel="Mobile/Immobile ratio",
                             record_stats=stats_records, metric_name="mob_immob_ratio", xtick_labels=bar_xticks, stats_config=cfg, annot_sink=panel_annots, style=_pstyle("mob_immob"))
        ax.set_title("Mobile/Immobile Ratio")

    # ── 5. Motion class fractions (stacked bars: x = population, colour = class) ─
    if "motion_classes" in panels:
        ax = _next_ax()
        # Canonical motion-class colours/order — shared with the single-run
        # figure AND the napari viewer (fa_constants) so a class is the same
        # colour everywhere: Immobile=red, Confined=orange, Brownian=blue,
        # Directed=green.
        classes = list(MOTION_CLASS_ORDER)
        class_colors = [motion_class_colors(theme)[c] for c in classes]
        # Each replicate's 4 named-class fractions are renormalised to sum to 1
        # (matching the single-run figure's panel-F stacked bar, which
        # renormalises the same 4 classes), so the stacked bars reach the top.
        # The dropped mass is the
        # "Unknown" share — tracks too short to fit a D/α (common on dense
        # palmTRACER data) — which is surfaced honestly in the x-axis labels
        # below, never silently hidden.
        def _fracs(summaries):
            rows = []
            for s in summaries:
                f = _motion_fractions(s["diffusion"])
                named = np.array([f.get(c, 0.0) for c in classes], dtype=float)
                tot = named.sum()
                if tot > 0:                 # skip replicates with no classifiable track
                    rows.append(named / tot)
            return np.array(rows) if rows else np.zeros((0, len(classes)))
        per_group, unclassified = [], []
        for ss in all_summaries:
            per_group.append(_fracs(ss))
            uncl = []
            for s in ss:
                f = _motion_fractions(s["diffusion"])
                uncl.append(1.0 - sum(f.get(c, 0.0) for c in classes))
            unclassified.append(float(np.mean(uncl)) if uncl else 0.0)
        # Mean composition per group (each replicate's fractions sum to 1, so the
        # per-group means also sum to ~1 → each stacked bar reaches the top).
        means = np.array([fr.mean(axis=0) if len(fr) else np.zeros(len(classes))
                          for fr in per_group])
        x = np.arange(n_groups)
        bottom = np.zeros(n_groups)
        for ci, (cname, ccol) in enumerate(zip(classes, class_colors)):
            seg = means[:, ci]
            ax.bar(x, seg, 0.7, bottom=bottom,
                   color=ccol, edgecolor=pal["BG"], linewidth=0.6, label=cname)
            # Label each segment with its mean % when it's big enough to read.
            for gi in range(n_groups):
                if seg[gi] >= 0.06:
                    ax.text(x[gi], bottom[gi] + seg[gi] / 2, f"{seg[gi]*100:.0f}%",
                            ha="center", va="center", fontsize=7,
                            color=label_text_color(ccol), zorder=4)
            bottom += seg
        # Per-class one-way stats — only meaningful when each card is an
        # independent group.  In two-factor mode the cards are paired across
        # time points, so a one-way test across them is invalid; the two-way
        # ANOVA report covers it instead.
        if not two_factor:
            for ci, cname in enumerate(classes):
                arrs = [fr[:, ci] if len(fr) else np.array([]) for fr in per_group]
                omn, pw = _STN(arrs, labels, cfg)
                stats_records[f"motion_frac_{cname}"] = {"omnibus": omn, "pairwise": pw}
        ax.set_xticks(x)
        # Show the GROUP NAME on the axis — the numeric stand-ins (bar_xticks)
        # hid the names once there were >4 groups, and a shallow 15° rotation let
        # long names overlap into an unreadable smear.  Angle scales with the
        # group count so names stay legible.  The replicate n is NOT repeated
        # here — it lives in the shared bottom legend.
        _mc_rot = 0 if n_groups <= 2 else (30 if n_groups <= 6 else 45)
        # Append the % of tracks that were too short to classify (the renormalised
        # "Unknown" share), so the composition bars stay honest about what was
        # excluded.  Only shown when it's non-trivial.
        mc_labels = [(f"{lbl}\n({u*100:.0f}% uncl.)" if u >= 0.005 else str(lbl))
                     for lbl, u in zip(labels, unclassified)]
        ax.set_xticklabels(mc_labels, rotation=_mc_rot,
                           ha="center" if _mc_rot == 0 else "right",
                           rotation_mode="anchor",
                           fontsize=8 if n_groups <= 6 else 7)
        # Headroom above the full (=1.0) stacks for an in-axes legend, so
        # tight_layout reserves space for it (a below-axis legend would not be
        # accounted for and could overlap the panel beneath).
        ax.set_ylim(0, 1.42)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("Fraction of tracks")
        ax.set_title("Motion Class Fractions")
        ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=7.5,
                  columnspacing=1.0, handlelength=1.1, handletextpad=0.4)
        # This panel's legend maps colour→motion class (group identity is already
        # on the x-axis), so exempt it from the shared-group-legend stripping pass.
        ax._firefly_keep_legend = True

    # ── 6. Track length distribution (CDF, x clipped at 99th %ile) ────────────
    if "track_length" in panels:
        ax = _next_ax()
        pooled_per_group = {}
        for grp_label, summaries, _ in _zip_groups():
            arrs = []
            for s in summaries:
                fi = _fi_or_default(s["params"], s.get("stem", ""))
                tl = _track_lengths(s["tracks"], fi)
                if len(tl):
                    arrs.append(tl)
            if arrs:
                pooled_per_group[grp_label] = np.concatenate(arrs)
        combined = (np.concatenate(list(pooled_per_group.values()))
                    if pooled_per_group else np.array([]))
        x_clip = float(np.percentile(combined, 99)) if len(combined) else None
        for grp_label, color in zip(labels, colors):
            p = pooled_per_group.get(grp_label)
            if p is None or len(p) == 0: continue
            x_sorted = np.sort(p)
            y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
            ax.plot(x_sorted, y, color=color, lw=1.5, label=grp_label)
        if pooled_per_group:
            if x_clip and x_clip > 0:
                ax.set_xlim(0, x_clip)
                ax.set_title("Track Length Distribution  (x clipped at 99th %ile)")
            else:
                ax.set_title("Track Length Distribution")
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Track length (s)")
            ax.set_ylabel("Cumulative fraction")
            ax.legend(frameon=False, loc="best")
        else:
            ax.text(0.5, 0.5, "No track-length data",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Track Length Distribution")
        # Stats: mean track length (per-replicate) — one-way only in flat mode;
        # the two-way ANOVA report covers it in two-factor mode.
        if not two_factor:
            arrs = [summary_df.loc[summary_df["group"] == lbl, "mean_track_length_s"].values
                    for lbl in labels]
            omn, pw = _STN(arrs, labels, cfg)
            stats_records["mean_track_length_s"] = {"omnibus": omn, "pairwise": pw}

    # ── 6b. Track count (trajectories detected per group) ─────────────────────
    if "track_count" in panels:
        ax = _next_ax()
        if two_factor:
            _interaction_plot(ax, summary_df, "n_tracks", group_order, tp_order,
                              group_colors, pal, ylabel="Tracks (n)",
                              headline=_twoway_headline(twoway_df, "n_tracks"),
                              card_colors=card_colors, stats_config=cfg)
        else:
            data = [summary_df.loc[summary_df["group"] == lbl, "n_tracks"].values
                    for lbl in labels]
            _bar_with_dots_n(ax, data, labels, colors, pal,
                             ylabel="Tracks (n)",
                             record_stats=stats_records, metric_name="n_tracks",
                             xtick_labels=bar_xticks, stats_config=cfg, annot_sink=panel_annots, style=_pstyle("track_count"))
        ax.set_title("Tracks detected")

    # ── 7. JDD: per-population D + fraction (N groups) ────────────────────────
    if "jdd" in panels:
        ax = _next_ax()
        any_data = False
        max_pop_overall = 0
        all_D = []                       # every plotted D, for y-axis tick choice
        # Spread groups across ±0.18 around each population index
        if n_groups > 1:
            offsets = np.linspace(-0.18, 0.18, n_groups)
        else:
            offsets = np.array([0.0])
        for gi, (grp_label, summaries, color) in enumerate(_zip_groups()):
            label_done = False
            for s in summaries:
                jd = s.get("jdd")
                if not jd or "D_values" not in jd: continue
                D = np.asarray(jd["D_values"], dtype=float)
                f = np.asarray(jd.get("fractions", np.ones_like(D)), dtype=float)
                if D.size == 0: continue
                any_data = True
                all_D.append(D[np.isfinite(D) & (D > 0)])
                max_pop_overall = max(max_pop_overall, len(D))
                sizes = 25 + 175 * np.clip(f, 0, 1)
                xs = np.arange(len(D)) + offsets[gi]
                ax.scatter(xs, D, s=sizes, color=color,
                           alpha=0.55, edgecolor=color,
                           label=(grp_label if not label_done else None))
                label_done = True
        if any_data:
            # Label the JDD populations Slow/Medium/Fast — matching the
            # single-run figure (panel K) so the two plots use ONE vocabulary.
            # Deliberately NOT "Immobile/Mobile": those words name the MSD
            # motion classes (Immobile/Confined/Brownian/Directed), a separate
            # analysis, and reusing them here invites cross-reading. These are
            # diffusion-coefficient populations from the jump-distance fit.
            tick_labels = ["Slow", "Medium", "Fast"][:max_pop_overall]
            if max_pop_overall == 1: tick_labels = ["All"]
            ax.set_xticks(np.arange(max_pop_overall))
            ax.set_xticklabels(tick_labels)
            ax.set_xlim(-0.5, max_pop_overall - 0.5)
            ax.set_ylabel("D (µm²/s, log scale)")
            ax.set_yscale("log")
            # ── Readable log y-axis ──────────────────────────────────────────
            # Plain decimal tick labels (0.001, 0.01, 0.1, 1) instead of 10^x
            # power notation, horizontal gridlines to trace a dot's value across
            # to the axis, and — when the data spans < 2 decades (so the decade
            # majors alone are too sparse) — labelled 2×/5× minor ticks.
            from matplotlib.ticker import (LogLocator, FuncFormatter,
                                           NullFormatter)
            _plain = FuncFormatter(lambda y, _p: f"{y:g}" if y > 0 else "")
            ax.yaxis.set_major_locator(LogLocator(base=10.0))
            ax.yaxis.set_major_formatter(_plain)
            _dd = np.concatenate(all_D) if all_D else np.array([])
            _dd = _dd[np.isfinite(_dd) & (_dd > 0)]
            _narrow = (_dd.size > 0 and
                       (np.log10(_dd.max()) - np.log10(_dd.min())) < 2.0)
            if _narrow:
                ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0)))
                ax.yaxis.set_minor_formatter(_plain)
                ax.tick_params(axis="y", which="minor", labelsize=7)
            else:
                ax.yaxis.set_minor_locator(
                    LogLocator(base=10.0, subs=tuple(np.arange(2, 10))))
                ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, axis="y", which="major", color=pal["GRD"],
                    lw=0.6, alpha=0.55)
            ax.grid(True, axis="y", which="minor", color=pal["GRD"],
                    lw=0.4, alpha=0.30)
            ax.set_axisbelow(True)       # gridlines behind the markers
            ax.set_title("JDD: per-population D  (marker size ∝ population fraction)")
            ax.legend(frameon=False, loc="best")
        else:
            ax.text(0.5, 0.5, "No JDD data\n(re-run analysis to generate)",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Jump Distance Distribution")

    # ── 8. Dwell time CDF (N groups) ──────────────────────────────────────────
    if "dwell_cdf" in panels:
        ax = _next_ax()
        any_data = False
        for grp_label, summaries, color in _zip_groups():
            pooled = []
            for s in summaries:
                d = s.get("dwell_times")
                if d is None or len(d) == 0: continue
                col = next((c for c in ("dwell_time_s", "dwell_s",
                                        "dwell_time", "dwell", "tau_s")
                            if c in d.columns), None)
                if col is None: continue
                pooled.extend(d[col].values)
            if not pooled: continue
            any_data = True
            arr = np.sort(np.asarray(pooled, dtype=float))
            arr = arr[arr > 0]
            if len(arr) == 0: continue
            y = 1 - np.arange(1, len(arr) + 1) / len(arr)
            ax.plot(arr, y, color=color, lw=1.5, label=grp_label)
        if any_data:
            ax.set_xlabel("Dwell time (s)")
            ax.set_ylabel("Survival fraction")
            ax.set_title("Dwell Time Survival")
            ax.set_yscale("log")
            ax.legend(frameon=False, loc="best")
        else:
            ax.text(0.5, 0.5, "No dwell-time data\n(re-run analysis to generate)",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Dwell Time Survival")

    # ── 9. Turning angle distribution (N groups, unsigned |angle|) ────────────
    # Single line per group, plotting the count of each |θ| bin on
    # the same 0°–180° x-axis.  Sign / rotational direction is handled
    # separately by the Radial Distribution panel.
    if "turning_angles" in panels:
        ax = _next_ax()
        any_data = False
        bins = np.linspace(0, 180, 37)                 # 5° bins
        centers = 0.5 * (bins[:-1] + bins[1:])
        pooled_per_group = []
        for grp_label, summaries, color in _zip_groups():
            pooled = []
            for s in summaries:
                ta = s.get("turning_angles")
                if ta is None or len(ta) == 0: continue
                pooled.extend(np.abs(np.asarray(ta).ravel()))
            pooled_per_group.append((grp_label, color, pooled))
        ta_ends, ta_cols, ta_labs = [], [], []
        for grp_label, color, pooled in pooled_per_group:
            if not pooled: continue
            any_data = True
            counts, _ = np.histogram(pooled, bins=bins)
            frac = counts / counts.sum() if counts.sum() else counts
            ax.plot(centers, frac, "-o", color=color, lw=1.5, ms=3, label=grp_label)
            ta_ends.append((float(centers[-1]), float(frac[-1])))
            ta_cols.append(color); ta_labs.append(grp_label)
        if any_data:
            ax.set_xlabel("|Turning angle|  (°)")
            ax.set_ylabel("Relative frequency")
            ax.set_xlim(0, 180)
            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_title("Turning Angle Distribution")
            # Direct end labels when the curves separate at 180°; else shared legend.
            if not _maybe_end_labels(ax, ta_ends, ta_cols, ta_labs):
                ax.legend(frameon=False, loc="best")
        else:
            ax.text(0.5, 0.5, "No turning-angle data\n(re-run analysis to generate)",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Turning Angle Distribution")

    # ── 10. Radial distribution (polar, signed turning angles) ────────────────
    # Polar histogram showing the angular distribution of step-to-step
    # turning angles.  Each group is plotted as a separate set of bars
    # offset around each bin centre.
    #
    # Implementation note: we replace the auto-created cartesian axis with
    # a polar one at the SAME SubplotSpec (not via fig.add_axes with raw
    # bounds), so that the polar axis remains a managed gridspec member.
    # If we used add_axes(bounds), tight_layout would later reposition the
    # other (gridspec-managed) subplots but leave the polar in its original
    # location, causing visible overlap.
    if "radial_dist" in panels:
        old_ax = axes[panel_idx]
        ss = old_ax.get_subplotspec()
        old_ax.remove()
        ax = fig.add_subplot(ss, projection="polar")
        axes[panel_idx] = ax
        panel_idx += 1

        any_data = False
        n_bins = 36
        # matplotlib polar bar() only renders correctly when theta ∈ [0, 2π);
        # shift the data accordingly.  The xticks are placed at positive-only
        # angles but labelled with their signed equivalents.
        bin_edges   = np.linspace(0, 2 * np.pi, n_bins + 1)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bar_width   = (bin_edges[1] - bin_edges[0]) * 0.95

        # First pass: get raw counts per group per bin.
        counts_per_group = []     # list of (group_idx, counts_array)
        for gi in range(n_groups):
            pooled = []
            for s in all_summaries[gi]:
                ta = s.get("turning_angles")
                if ta is None or len(ta) == 0: continue
                pooled.extend(np.asarray(ta).ravel())
            if not pooled:
                counts_per_group.append((gi, np.zeros(n_bins)))
                continue
            arr = np.asarray(pooled, dtype=float)
            if not np.any(arr < -1e-3):
                arr = np.concatenate([arr, -arr])
            angles_rad = np.mod(np.deg2rad(arr), 2 * np.pi)
            counts, _ = np.histogram(angles_rad, bins=bin_edges)
            counts_per_group.append((gi, counts.astype(float)))
            if counts.sum() > 0:
                any_data = True

        if any_data:
            # ── Normalise each group to ITS OWN total ─────────────────────
            # Otherwise a group with more total angles automatically draws
            # bigger bars everywhere — a sample-size artefact, not a real
            # shape difference.  After dividing by the per-group total, each
            # group's values sum to 1.0 across the full circle, so the bars
            # compare distribution SHAPE.
            # Bars from different groups are offset around each bin centre
            # for easy side-by-side comparison.
            per_bar_width = bar_width / max(1, n_groups) * 0.95
            for gi, counts in counts_per_group:
                total = counts.sum()
                if total <= 0:
                    continue
                normalised = counts / total
                offset = (gi - (n_groups - 1) / 2) * per_bar_width
                ax.bar(bin_centres + offset, normalised,
                       width=per_bar_width, bottom=0.0,
                       color=colors[gi], alpha=0.85,
                       edgecolor=pal["GRD"], linewidth=0.3,
                       label=labels[gi])

        if any_data:
            # Conventional orientation: 0° at top (straight ahead),
            # right hemisphere = positive turns, left hemisphere = negative.
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            # Positive-only xticks; labelled with signed equivalents.
            ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
            ax.set_xticklabels(["0°", "+45°", "+90°", "+135°", "±180°",
                                "−135°", "−90°", "−45°"], fontsize=7)
            # Hide the radial-axis numeric labels — bar length is
            # interpreted comparatively, not in absolute density units.
            ax.set_yticklabels([])
            ax.tick_params(axis="y", which="both", left=False)
            ax.set_title("Radial Distribution  (each group normalised to "
                         "its own total)", pad=14, fontsize=9)
            ax.legend(loc="upper right", bbox_to_anchor=(1.20, 1.10),
                      frameon=False, fontsize=8)
            ax.grid(True, ls="--", alpha=0.22, lw=0.5)
        else:
            ax.text(0.5, 0.5, "No turning-angle data",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Radial Distribution")

    # ── van Hove non-Gaussian α₂ (population heterogeneity) ───────────────────
    if "van_hove" in panels and "nongauss_alpha2" in summary_df.columns:
        ax = _next_ax()
        if two_factor:
            _interaction_plot(ax, summary_df, "nongauss_alpha2", group_order,
                              tp_order, group_colors, pal,
                              ylabel="Non-Gaussian α₂",
                              headline=_twoway_headline(twoway_df, "nongauss_alpha2"),
                              card_colors=card_colors, stats_config=cfg)
        else:
            data = [summary_df.loc[summary_df["group"] == lbl, "nongauss_alpha2"]
                    .dropna().to_numpy() for lbl in labels]
            _bar_with_dots_n(ax, data, labels, colors, pal,
                             ylabel="Non-Gaussian α₂",
                             record_stats=stats_records,
                             metric_name="nongauss_alpha2", xtick_labels=bar_xticks, stats_config=cfg, annot_sink=panel_annots, style=_pstyle("van_hove"))
        ax.set_title("Population heterogeneity (α₂)")

    # ── VACF persistence (directional memory) ─────────────────────────────────
    if "vacf" in panels and "vacf_persistence" in summary_df.columns:
        ax = _next_ax()
        if two_factor:
            _interaction_plot(ax, summary_df, "vacf_persistence", group_order,
                              tp_order, group_colors, pal,
                              ylabel="VACF persistence (lag 1)",
                              headline=_twoway_headline(twoway_df, "vacf_persistence"),
                              card_colors=card_colors, stats_config=cfg)
        else:
            data = [summary_df.loc[summary_df["group"] == lbl, "vacf_persistence"]
                    .dropna().to_numpy() for lbl in labels]
            _bar_with_dots_n(ax, data, labels, colors, pal,
                             ylabel="VACF persistence (lag 1)",
                             record_stats=stats_records,
                             metric_name="vacf_persistence", xtick_labels=bar_xticks, stats_config=cfg, annot_sink=panel_annots, style=_pstyle("vacf"))
        ax.set_title("Directional persistence (VACF lag 1)")

    # ── Drop per-panel legends (the top band is the shared key) ───────────────
    # Per-panel `loc="best"` legends overlap the data badly once there are many
    # groups / time points, and they repeat the same key in every panel.  Remove
    # them all.  The colour ↔ group ↔ n key now lives in the quick-glance summary
    # band at the TOP of the figure (drawn below), which also carries the track
    # count / median D / median alpha — so a separate bottom legend would only
    # duplicate it.  When the bars use short numeric x-tick tokens (>4 groups),
    # the band entries are numbered to match (see the band loop below).
    for ax in axes[:n_plots]:
        if getattr(ax, "_firefly_keep_legend", False):
            continue  # panel carries its own colour key (e.g. motion classes)
        _lg = ax.get_legend()
        if _lg is not None:
            _lg.remove()
    legend_rows = 0   # no reserved bottom strip — the band replaces the legend

    # ── Suptitle ──────────────────────────────────────────────────────────────
    # "A vs B [vs C]" only stays readable for a few groups; beyond that it
    # overflows the figure, so collapse to a count (the names + n live in the
    # shared legend).
    if n_groups <= 3:
        parts = [f"{labels[i]}  (n={len(all_summaries[i])})" for i in range(n_groups)]
        suptitle = "   vs   ".join(parts)
    elif two_factor:
        suptitle = (f"{len(group_order)} groups × {len(tp_order)} time points  "
                    f"({n_groups} cells)")
    else:
        suptitle = f"Comparison of {n_groups} groups"
    fig_h = base_h + band_h_in
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", color=pal["TXT"],
                 y=1.0 - 0.16 / fig_h)
    for ax in axes[:n_plots]:
        ax.set_facecolor(pal["PNL"])
        # Modern look: drop the top/right spines, thin the rest (polar +
        # image panels keep their frame — handled inside style_axes).
        style_axes(ax, pal,
                   kind=("polar" if getattr(ax, "name", "") == "polar"
                         else "cartesian"))

    # ── Quick-glance per-group summary band (top) ─────────────────────────────
    # Mirror the individual-analysis stats panel: each group's trajectory count,
    # median D and median alpha at a glance, so the headline differences are
    # obvious without opening the stats CSV.  One colour-matched entry per card
    # (= the shared-legend order).  D / alpha are the MEDIAN of the per-cell
    # medians — the same per-replicate scalars the across-group tests use below,
    # so the header agrees with the statistics; track count is the group total.
    def _card_summary(i):
        m = (summary_df["group"] == group_factor[i])
        if two_factor:
            m = m & (summary_df["timepoint"] == timepoints_per_card[i])
        sub = summary_df[m]
        n_cells = int(len(sub))
        n_trk = int(sub["n_tracks"].sum()) if "n_tracks" in sub else 0
        med_D = (float(np.nanmedian(sub["median_D"]))
                 if "median_D" in sub and sub["median_D"].notna().any() else np.nan)
        med_a = (float(np.nanmedian(sub["median_alpha"]))
                 if "median_alpha" in sub and sub["median_alpha"].notna().any() else np.nan)
        return n_cells, n_trk, med_D, med_a

    def _band_entry(label, n_cells, n_trk, d, a, compact):
        lab = label if len(label) <= 18 else label[:17] + "…"
        if compact:
            d_s = f"{d:.3f}" if np.isfinite(d) else "—"
            a_s = f"{a:.2f}" if np.isfinite(a) else "—"
            return f"●  {lab} — {n_trk:,} trk · D {d_s} · α {a_s}  (n={n_cells})"
        d_s = f"{d:.4f} µm²/s" if np.isfinite(d) else "—"
        a_s = f"{a:.3f}" if np.isfinite(a) else "—"
        return (f"●  {lab} — {n_trk:,} tracks · med D {d_s} · "
                f"med α {a_s}   (n={n_cells})")

    band_top = 1.0 - 0.52 / fig_h            # first band row, below the suptitle
    row_step = band_row_in / fig_h
    col_xs = [(c + 0.5) / band_ncol for c in range(band_ncol)]
    for i in range(n_groups):
        r, c = divmod(i, band_ncol)
        n_cells, n_trk, med_D, med_a = _card_summary(i)
        # When the bar panels use numeric x-tick tokens (>4 groups), number the
        # band entries to match — the band is then the key for those axes.
        lbl = f"{i + 1}. {labels[i]}" if many_groups else labels[i]
        fig.text(col_xs[c], band_top - r * row_step,
                 _band_entry(lbl, n_cells, n_trk, med_D, med_a, band_compact),
                 color=colors[i], fontsize=band_fs, ha="center", va="top")

    # No bottom strip (the band replaced the shared legend → legend_rows == 0);
    # reserve only the inch-sized top strip for the summary band.
    bottom = min(0.18, 0.03 + 0.026 * legend_rows) if legend_rows else 0.0
    top = 1.0 - band_h_in / fig_h
    fig.tight_layout(rect=[0, bottom, 1, top])

    # ── Build statistics dataframe (per metric × pairwise) ────────────────────
    # Bonferroni correction across pairwise comparisons WITHIN each metric:
    # multiplies the raw p-value by the number of pairs (capped at 1.0).
    # The omnibus row gets the raw p-value only — it's a single test.
    # Format the power-based sample-size estimate: int, ">500" when the
    # effect is too small to be practical, or "" when not computable.
    def _fmt_n_needed(v):
        if v is None:
            return ""
        try:
            return ">500" if int(v) > 500 else int(v)
        except (TypeError, ValueError):
            return ""

    # The headline diffusion endpoints (median_D, median_alpha), population
    # heterogeneity (non-Gaussian alpha2) and directionality (VACF persistence)
    # are per-replicate scalars in summary_df with no dedicated figure panel;
    # test them across groups (flat mode) so they appear in the stats table and
    # CSV.  median_D / median_alpha were declared in the across-metric family
    # below but never actually tested — the most important diffusion metrics had
    # no comparison stats at all.  (#35)  The two-way report covers the
    # two-factor case separately.
    if not two_factor:
        for _m in ("median_D", "median_alpha",
                   "nongauss_alpha2", "vacf_persistence"):
            if _m in summary_df.columns and _m not in stats_records:
                arrs = [summary_df.loc[summary_df["group"] == lbl, _m]
                        .dropna().to_numpy() for lbl in labels]
                if sum(len(a) for a in arrs) >= 2:
                    omn, pw = _STN(arrs, labels, cfg)
                    stats_records[_m] = {"omnibus": omn, "pairwise": pw}

    # ── Multiple-comparison correction (within metric + optional across) ──────
    # Correction is applied HERE, centrally, so (a) the across-metric family can
    # see every p-value and (b) the on-figure stars (drawn from the same
    # corrected values) agree with the CSV.
    from firefly.analysis.fa_stats_config import (
        correct_pvalues, stars_for, correction_display)
    _alpha, _method = cfg["alpha"], cfg["correction"]
    # The across-metric family is the pairwise comparisons of the canonical
    # SCALAR metrics (omnibus rows and per-class motion fractions excluded), so
    # the family size is reproducible and matches the "scalar metrics" framing.
    _ACROSS_FAMILY = {"auc_msd", "mob_immob_ratio", "median_D", "median_alpha",
                      "mean_track_length_s", "n_tracks", "nongauss_alpha2",
                      "vacf_persistence"}
    across_pw = []
    for metric, rec in stats_records.items():
        pairs = rec.get("pairwise", [])
        # Self-correcting post-hocs (Games-Howell / Tukey / Dunnett) already
        # control their own family — DON'T double-correct them.  Only ordinary
        # pairwise + Dunn rows pass through correct_pvalues.
        corr_idx = [k for k, pw in enumerate(pairs)
                    if not pw.get("self_corrected")]
        wcorr = correct_pvalues([pairs[k].get("p") for k in corr_idx], _method)
        wmap = dict(zip(corr_idx, wcorr))
        for k, pw in enumerate(pairs):
            if pw.get("self_corrected"):
                pw["p_within"] = pw.get("p")
                # Preserve the underpowered / blank-star convention: if the raw
                # comparison was blanked (n<3, not interpretable), the corrected
                # star stays blank too — otherwise correction would manufacture a
                # star for a comparison the engine declared uninterpretable.
                pw["stars_within"] = (stars_for(pw.get("p"), _alpha)
                                      if pw.get("stars") else "")
            else:
                pw["p_within"] = wmap[k]
                pw["stars_within"] = (stars_for(wmap[k], _alpha)
                                      if pw.get("stars") else "")
            pw.setdefault("p_across", np.nan)
            pw.setdefault("stars_across", "")
            # Across-metric family = ordinary pairwise only; exclude
            # self-corrected post-hocs + the Dunnett family (separate regimes).
            if (metric in _ACROSS_FAMILY and not pw.get("self_corrected")
                    and pw.get("family", "pairwise") == "pairwise"
                    and np.isfinite(pw.get("p", np.nan))):
                across_pw.append(pw)
    family_size = len(across_pw)
    if cfg["across_metric_correction"] and across_pw:
        acorr = correct_pvalues([pw["p"] for pw in across_pw], _method)
        for pw, ap in zip(across_pw, acorr):
            pw["p_across"] = ap
            pw["stars_across"] = stars_for(ap, _alpha)
    elif family_size > 1:
        # Across-metric correction is OFF by default; make the multiplicity the
        # reader is exposed to explicit rather than silent.  With N scalar-metric
        # comparisons in the family, the chance of ≥1 false positive at α is
        # ~1−(1−α)^N — surface N so the user can judge / enable correction.
        print(f"  Stats: across-metric correction OFF — {family_size} scalar-"
              f"metric comparisons in the family (family-wise error not "
              f"controlled across metrics; enable across-metric correction to "
              f"control it).")

    stats_rows = []
    for metric, rec in stats_records.items():
        omn = rec.get("omnibus")
        if omn:
            stats_rows.append({
                "metric": metric, "comparison": "omnibus",
                "test": omn["test"],
                "p_value": omn["p"], "stars": omn["stars"],
                "correction_method": "none (omnibus needs no correction)",
                "p_value_corrected": omn["p"], "stars_corrected": omn["stars"],
                "p_value_across_metric": "", "stars_across_metric": "",
                "n_a": "", "n_b": "", "mean_a": "", "mean_b": "",
                "sem_a": "", "sem_b": "", "label_a": "all groups", "label_b": "",
                "cohens_d": "", "hedges_g": "",
                "hedges_g_ci_low": "", "hedges_g_ci_high": "",
                "cliffs_delta": "", "cliffs_delta_ci_low": "",
                "cliffs_delta_ci_high": "", "rank_biserial": "",
                "tost_p": "", "tost_equivalent": "",
                "omnibus_effect_size": (omn.get("effect_size")
                    if omn.get("effect_size") is not None else ""),
                "omnibus_effect_size_kind": omn.get("effect_size_kind") or "",
                "n_per_group_for_80pct_power": "",
                "n_per_group_for_90pct_power": "",
                "note": omn.get("note", ""),
            })
        for pw in rec.get("pairwise", []):
            _is_dunnett = pw.get("family") == "dunnett"
            stats_rows.append({
                "metric": metric,
                "comparison": (f"{pw['label_i']} vs {pw['label_j']}"
                               + (" (Dunnett)" if _is_dunnett else "")),
                "test": pw["test"],
                "p_value": pw["p"], "stars": pw["stars"],
                "correction_method": (f"{pw['test']} (family-wise)"
                    if pw.get("self_corrected") else correction_display(_method)),
                "p_value_corrected": pw.get("p_within"),
                "stars_corrected": pw.get("stars_within", ""),
                "p_value_across_metric": (pw.get("p_across")
                    if cfg["across_metric_correction"] else ""),
                "stars_across_metric": (pw.get("stars_across", "")
                    if cfg["across_metric_correction"] else ""),
                "n_a": pw["n_i"], "n_b": pw["n_j"],
                "mean_a": pw["mean_i"], "mean_b": pw["mean_j"],
                "sem_a": pw["sem_i"], "sem_b": pw["sem_j"],
                "label_a": pw["label_i"], "label_b": pw["label_j"],
                "cohens_d": pw.get("cohens_d"),
                "hedges_g": pw.get("hedges_g"),
                "hedges_g_ci_low": pw.get("hedges_g_ci_low"),
                "hedges_g_ci_high": pw.get("hedges_g_ci_high"),
                "cliffs_delta": pw.get("cliffs_delta"),
                "cliffs_delta_ci_low": pw.get("cliffs_delta_ci_low"),
                "cliffs_delta_ci_high": pw.get("cliffs_delta_ci_high"),
                "rank_biserial": pw.get("rank_biserial"),
                "tost_p": (pw.get("tost_p") if cfg["equivalence_tost"] else ""),
                "tost_equivalent": (pw.get("tost_equivalent")
                    if cfg["equivalence_tost"] else ""),
                "omnibus_effect_size": "", "omnibus_effect_size_kind": "",
                "n_per_group_for_80pct_power": _fmt_n_needed(pw.get("n_needed_80")),
                "n_per_group_for_90pct_power": _fmt_n_needed(pw.get("n_needed_90")),
                "note": pw.get("note", ""),
            })
    stats_df = pd.DataFrame(stats_rows)

    # Optional post-pass: when across-metric correction is on, switch the
    # on-figure stars to the across-metric-corrected value so the figure agrees
    # with the CSV.  (Within-metric correction was already drawn at panel time.)
    if cfg["figure_stars_use_corrected"] and cfg["across_metric_correction"]:
        for _metric, (txt, build) in panel_annots.items():
            try:
                new = build("across")
                if new:
                    txt.set_text(new)
            except Exception:
                pass

    # ── Two-factor (group × time point) mixed ANOVA ───────────────────────────
    # Paired design: between=group, within=timepoint, subject=cell.  Scalars run
    # directly; the curve graphs (MSD, LogD) get a per-time-point group×(lag|bin)
    # drill-down because pingouin can't fit 1-between + 2-within in one model.
    # twoway_df / twoway_msg / pair_warn / paired_df were already computed up
    # front (so the interaction panels could use them) — reuse here.
    drilldown = {}
    if two_factor:
        # Build per-cell curve records, restricted to the paired cell set.
        paired_cells = (set(zip(paired_df["group"], paired_df["cell"]))
                        if paired_df is not None and len(paired_df) else set())
        records = []
        for gi, summaries in enumerate(all_summaries):
            for s in summaries:
                cell, _m = fa_twoway.derive_subject_key(s["stem"], timepoint_tokens)
                if paired_cells and (group_factor[gi], cell) not in paired_cells:
                    continue
                d = s.get("diffusion")
                e = s.get("ensemble_msd")
                msd = None
                if e is not None and "lag_frame" in getattr(e, "columns", []):
                    msd = e.sort_values("lag_frame").set_index("lag_frame")["msd_um2"]
                records.append({
                    "group": group_factor[gi], "timepoint": timepoints_per_card[gi],
                    "cell": cell, "msd": msd,
                    "diffusion_D": (d["D"].to_numpy() if d is not None
                                    and "D" in d.columns else None),
                })
        for kind, pnl in (("msd", "msd"), ("logd", "logd_dist")):
            if pnl in panels:
                ddf, dmsg = fa_twoway.curve_drilldown_per_timepoint(records, kind)
                if ddf is not None and len(ddf):
                    drilldown[kind] = (ddf, dmsg)

    # ── Save outputs ──────────────────────────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        png_path  = os.path.join(output_dir, f"{output_stem}.png")
        pdf_path  = os.path.join(output_dir, f"{output_stem}.pdf")
        csv_path  = os.path.join(output_dir, f"{output_stem}_summary.csv")
        stats_csv = os.path.join(output_dir, f"{output_stem}_stats.csv")
        # Write the IRREPLACEABLE data tables first and guard every save
        # independently, so a figure-save failure (disk full, or a deep Windows
        # output path) can't discard the summary/stats CSVs the scientist needs
        # — the same "one save kills everything" trap the single-file pipeline
        # was hardened against.  Paths run through _win_long_path so a >260-char
        # Windows path (OneDrive/RDM) doesn't fail the writes.  (#23)
        try:
            from firefly.analysis.fa_io import atomic_to_csv
            atomic_to_csv(summary_df, _win_long_path(csv_path), index=False)
            print(f"  Saved: {csv_path}")
        except Exception as _e:
            print(f"  WARN: comparison summary CSV save failed: {_e}")
        if len(stats_df):
            try:
                _write_prism_ttests(_win_long_path(stats_csv), stats_df,
                                    stats_config=cfg)
                print(f"  Saved: {stats_csv}")
            except Exception as _e:
                print(f"  WARN: comparison stats CSV save failed: {_e}")
        try:
            fig.savefig(_win_long_path(png_path), dpi=200, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  Saved: {png_path}")
        except Exception as _e:
            print(f"  WARN: comparison figure PNG save failed: {_e}")
        try:
            fig.savefig(_win_long_path(pdf_path), bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"  Saved: {pdf_path}")
        except Exception as _e:
            print(f"  WARN: comparison figure PDF save failed: {_e}")

        # ── Two-factor ANOVA CSV ─────────────────────────────────────────────
        if two_factor and twoway_df is not None and len(twoway_df):
            tw_csv = os.path.join(output_dir, f"{output_stem}_twoway_anova.csv")
            try:
                # Writes split single-table files and prints each saved path.
                _write_twoway_csv(tw_csv, twoway_df, twoway_msg, drilldown,
                                  summary_df, group_order, tp_order, pair_warn,
                                  stats_config=cfg)
            except Exception as exc:
                print(f"  Two-way CSV skipped ({type(exc).__name__}: {exc})")

        # ── Combined PDF report (figure + parameters + folders + stats) ──────
        if pdf_report:
            report_path = os.path.join(output_dir, f"{output_stem}_report.pdf")
            try:
                _write_pdf_report(report_path, fig, groups, all_summaries,
                                  labels, colors, summary_df, stats_df,
                                  panels=panels, theme=theme, palette=pal,
                                  twoway_df=twoway_df, twoway_msg=twoway_msg,
                                  drilldown=drilldown, pair_warn=pair_warn,
                                  stats_config=cfg)
                print(f"  Saved: {report_path}")
            except Exception as exc:
                print(f"  PDF report skipped ({type(exc).__name__}: {exc})")

        # ── Per-comparison circular-statistics CSV + PDF ────────────────────
        # Pool angles per group (across all replicates), compute the full
        # CircStat suite for each group, and emit:
        #   * {stem}_circular_statistics.csv  — one row per group
        #   * {stem}_circular_statistics.pdf  — themed multi-page PDF
        #     (page 1 = summary grid + comparison table; pages 2..N+1 =
        #     per-group detail mirroring the per-file report).
        if not cfg["include_circular_outputs"]:
            print("  Comparison circular-stats: disabled in stats config")
        else:
            try:
                groups_angles_pooled = []
                # Per-replicate angle arrays — one list of arrays per group.
                # Used to compute per-replicate κ, R̄, μ for the Welch t-test
                # and per-replicate Watson-Williams F-test (treats each
                # replicate as one data point, the statistically defensible
                # framing for n=5 vs n=3 designs).
                #
                # NOTE: the per-track (angle, D) pairs for a circular-linear
                # correlation used to be assembled here — running
                # compute_per_track_mean_angle on every track of every replicate
                # — and then thrown away, because the pooled correlation is
                # disabled in compute_circular_comparison_tests for the SAME
                # pseudoreplication reason as the other pooled tests (n =
                # thousands of tracks → p ≈ 0 regardless of the real effect).
                # That dead computation is removed.  (#31)
                per_replicate_angles = {}
                for label, ss, color in zip(labels, all_summaries, colors):
                    pooled = []
                    rep_angle_arrays = []
                    for s in ss:
                        ta = s.get("turning_angles")
                        if ta is not None:
                            arr = np.asarray(ta, dtype=float).ravel()
                            if arr.size:
                                pooled.append(arr)
                                rep_angle_arrays.append(arr)
                    pooled_arr = (np.concatenate(pooled)
                                  if pooled else np.array([], dtype=float))
                    groups_angles_pooled.append((label, pooled_arr, color))
                    per_replicate_angles[label] = rep_angle_arrays
                # Stem for the split circular CSVs (the function derives
                # _circular_per_group / _per_replicate / _tests from this and
                # prints each saved path itself).
                cs_csv = os.path.join(
                    output_dir, f"{output_stem}_circular_statistics.csv")
                cs_pdf = os.path.join(
                    output_dir, f"{output_stem}_circular_statistics.pdf")
                save_comparison_circular_statistics(
                    groups_angles_pooled,
                    csv_path=cs_csv, pdf_path=cs_pdf,
                    fig_theme=theme,
                    per_replicate_angles=per_replicate_angles,
                    stats_config=cfg)
                print(f"  Saved: {cs_pdf}")
            except Exception as exc:
                print(f"  Comparison circular-stats skipped "
                      f"({type(exc).__name__}: {exc})")

        # ── Machine-readable results snapshot (drives the Results tab) ──────
        try:
            from firefly.analysis.fa_stats_config import config_summary_rows
            results_json = os.path.join(output_dir,
                                        f"{output_stem}_results.json")
            meta = {
                "stem": output_stem,
                "output_dir": output_dir,
                "n_groups": int(n_groups),
                "two_factor": bool(two_factor),
                "mobile_d_threshold": float(mobile_d_threshold),
                "group_labels": list(labels),
                "group_colors": list(colors),
                "group_order": list(group_order),
                "timepoints": list(tp_order),
                "pair_warn": pair_warn or "",
                "files": {
                    "png":          f"{output_stem}.png",
                    "figure_pdf":   f"{output_stem}.pdf",
                    "report_pdf":   f"{output_stem}_report.pdf",
                    "summary_csv":  f"{output_stem}_summary.csv",
                    "stats_csv":    f"{output_stem}_stats.csv",
                    "twoway_csv":   f"{output_stem}_twoway_anova.csv",
                    "circular_per_group":
                        f"{output_stem}_circular_per_group.csv",
                    "circular_per_replicate":
                        f"{output_stem}_circular_per_replicate.csv",
                    "circular_tests":
                        f"{output_stem}_circular_tests.csv",
                },
            }
            _write_results_json(results_json, meta=meta,
                                config_summary=config_summary_rows(cfg),
                                stats_config=cfg, summary_df=summary_df,
                                stats_records=stats_records,
                                twoway_df=twoway_df, twoway_msg=twoway_msg)
            print(f"  Saved: {results_json}")
        except Exception as exc:
            print(f"  Results JSON skipped ({type(exc).__name__}: {exc})")

    return fig, summary_df, stats_records


def render_report(report_data, *, output_dir=None, output_stem="comparison",
                  panels=None, theme="Dark", pdf_report=True,
                  logd_plot_style="overlaid", msd_plot_style="mean_faceted",
                  msd_err="SEM", auc_plot_style="paired", group_style="box_points",
                  panel_styles=None,
                  logd_clip_d_min=1e-5, logd_clip_d_max=10.0, progress_cb=None):
    """Draw (+ optionally save) a comparison from a precomputed `ReportData`.
    Only theme / graph style / panel selection vary here, so this is the cheap
    part to re-run for a live style change on cached data.  Points the memo cache
    at `report_data.stat_cache` for the duration of the draw.  Returns
    ``(fig, summary_df, stats)`` exactly like `compare_groups`."""
    _TL.stat_cache = report_data.stat_cache
    try:
        return _draw_report(
            report_data, output_dir=output_dir, output_stem=output_stem,
            panels=panels, theme=theme, pdf_report=pdf_report,
            logd_plot_style=logd_plot_style, msd_plot_style=msd_plot_style,
            msd_err=msd_err, auc_plot_style=auc_plot_style, group_style=group_style,
            panel_styles=panel_styles,
            logd_clip_d_min=logd_clip_d_min, logd_clip_d_max=logd_clip_d_max,
            progress_cb=progress_cb)
    finally:
        _TL.stat_cache = None


def compare_groups(groups=None, output_dir=None, output_stem="comparison",
                   panels=None, theme="Dark", pdf_report=True,
                   mobile_d_threshold=MOBILE_D_THRESHOLD_DEFAULT,
                   logd_plot_style="overlaid", msd_plot_style="mean_faceted",
                   msd_err="SEM", auc_plot_style="paired", group_style="box_points",
                   panel_styles=None,
                   logd_clip_d_min=1e-5, logd_clip_d_max=10.0,
                   progress_cb=None, stats_config=None, use_native=False,
                   report_data=None):
    """Compare N>=2 groups of analysis-output folders -> multi-panel figure,
    summary CSV, statistics CSV and combined PDF report.  Thin wrapper: computes a
    `ReportData` (unless one is supplied via ``report_data``) then renders it, so
    the public API + return contract are unchanged.  Returns
    ``(fig, summary_df, stats)``.  The live Analysis tab caches the `ReportData`
    (see :func:`compute_report`) and re-runs only :func:`render_report` on a style
    or theme change."""
    rd = report_data if report_data is not None else compute_report(
        groups, mobile_d_threshold=mobile_d_threshold, stats_config=stats_config,
        use_native=use_native, progress_cb=progress_cb)
    return render_report(
        rd, output_dir=output_dir, output_stem=output_stem, panels=panels,
        theme=theme, pdf_report=pdf_report, logd_plot_style=logd_plot_style,
        msd_plot_style=msd_plot_style, msd_err=msd_err, auc_plot_style=auc_plot_style,
        group_style=group_style, panel_styles=panel_styles,
        logd_clip_d_min=logd_clip_d_min, logd_clip_d_max=logd_clip_d_max,
        progress_cb=progress_cb)


def _json_safe(obj):
    """Recursively coerce numpy/pandas scalars + non-finite floats into
    JSON-native values, so the results file is strict, valid JSON (no NaN/Inf
    tokens, no numpy types).  NaN / Inf / pandas-NA → None."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (str, bytes)):
        return obj.decode() if isinstance(obj, bytes) else obj
    # pandas NA / NaT or any other scalar
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def _write_results_json(path, *, meta, config_summary, stats_config,
                        summary_df, stats_records, twoway_df, twoway_msg):
    """Write a compact, machine-readable snapshot of a comparison's results —
    consumed by the in-app Results tab (auto-show after a run, or 'Open a
    previous comparison').  Everything is sanitised through `_json_safe` so the
    file is strict JSON.  Circular stats are NOT duplicated here; the Results
    tab reads them from the already-clean `_circular_*.csv` files."""
    payload = {
        "schema_version": 1,
        "meta": _json_safe(meta),
        "config_summary": [[str(lbl), str(val)] for lbl, val in config_summary],
        "stats_config": _json_safe(stats_config),
        "summary": _json_safe(summary_df.to_dict(orient="records")),
        "stats": _json_safe(stats_records),
        "twoway": {
            "rows": (_json_safe(twoway_df.to_dict(orient="records"))
                     if (twoway_df is not None and len(twoway_df)) else []),
            "message": str(twoway_msg or ""),
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _write_pdf_report(path, fig, groups, all_summaries, labels, colors,
                      summary_df, stats_df, panels, theme, palette,
                      twoway_df=None, twoway_msg=None, drilldown=None,
                      pair_warn=None, stats_config=None):
    """Multi-page PDF: cover + figure, parameters & folders, statistics, and
    (in two-factor mode) the group × time-point ANOVA tables."""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    pal = palette
    with PdfPages(path) as pdf:
        # ── Page 1: the comparison figure itself ──────────────────────────────
        pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches="tight")

        # ── Page 2: cover / parameters ────────────────────────────────────────
        page2 = plt.figure(figsize=(8.5, 11), facecolor=pal["BG"])
        page2.text(0.5, 0.96, "sptPALM Comparison Report",
                   ha="center", fontsize=18, fontweight="bold", color=pal["TXT"])

        meta_lines = [
            f"Theme:              {theme}",
            f"Panels rendered:    {', '.join(sorted(panels))}",
            f"Number of groups:   {len(groups)}",
            "",
            "Groups:",
        ]
        for i, g in enumerate(groups):
            meta_lines.append(
                f"  • {labels[i]}   "
                f"(n={len(all_summaries[i])} folder(s), "
                f"colour {colors[i]})")
        meta_lines.append("")
        meta_lines.append("Folders:")
        for i in range(len(groups)):
            meta_lines.append(f"  [{labels[i]}]")
            for f in groups[i]["folders"]:
                meta_lines.append(f"    {f}")
            meta_lines.append("")

        page2.text(0.06, 0.92, "\n".join(meta_lines),
                   ha="left", va="top", fontsize=9, family="monospace",
                   color=pal["TXT"])
        pdf.savefig(page2, facecolor=pal["BG"], bbox_inches="tight")
        plt.close(page2)

        # ── Page 3: per-replicate scalar summary table ────────────────────────
        if len(summary_df):
            page3 = plt.figure(figsize=(11, 8.5), facecolor=pal["BG"])
            page3.text(0.5, 0.96, "Per-replicate scalar metrics",
                       ha="center", fontsize=14, fontweight="bold",
                       color=pal["TXT"])
            ax = page3.add_axes([0.04, 0.04, 0.92, 0.86])
            ax.axis("off")
            disp = summary_df.copy()
            for c in disp.select_dtypes(include="float").columns:
                disp[c] = disp[c].apply(
                    lambda x: f"{x:.4g}" if np.isfinite(x) else "")
            disp["folder"] = disp["folder"].apply(
                lambda p: "..." + p[-40:] if isinstance(p, str) and len(p) > 43 else p)
            tbl = ax.table(cellText=disp.values.tolist(),
                           colLabels=list(disp.columns), loc="center",
                           cellLoc="left")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.scale(1, 1.2)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor(pal["GRD"])
                cell.set_text_props(color=pal["TXT"])
                cell.set_facecolor(pal["PNL"] if r > 0 else pal["BG"])
                if r == 0:
                    cell.set_text_props(weight="bold", color=pal["TXT"])
            pdf.savefig(page3, facecolor=pal["BG"], bbox_inches="tight")
            plt.close(page3)

        # ── Page 4: statistical tests ─────────────────────────────────────────
        if len(stats_df):
            page4 = plt.figure(figsize=(11, 8.5), facecolor=pal["BG"])
            page4.text(0.5, 0.96, "Statistical tests",
                       ha="center", fontsize=14, fontweight="bold",
                       color=pal["TXT"])
            ax = page4.add_axes([0.03, 0.04, 0.94, 0.86])
            ax.axis("off")
            # Curated column subset — the full stats_df has ~30 columns now
            # (effect sizes, CIs, TOST, power), which overflow a landscape
            # page; the complete set lives in the CSV.
            _pdf_cols = ["metric", "comparison", "test", "p_value",
                         "p_value_corrected", "stars_corrected",
                         "hedges_g", "cliffs_delta", "note"]
            disp = stats_df[[c for c in _pdf_cols
                             if c in stats_df.columns]].copy()
            for c in ("p_value", "p_value_corrected", "hedges_g", "cliffs_delta"):
                if c in disp.columns:
                    disp[c] = disp[c].apply(
                        lambda x: f"{x:.4g}" if isinstance(x, (int, float)) and np.isfinite(x) else x)
            tbl = ax.table(cellText=disp.values.tolist(),
                           colLabels=list(disp.columns), loc="center",
                           cellLoc="left")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.scale(1, 1.2)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor(pal["GRD"])
                cell.set_text_props(color=pal["TXT"])
                cell.set_facecolor(pal["PNL"] if r > 0 else pal["BG"])
                if r == 0:
                    cell.set_text_props(weight="bold", color=pal["TXT"])
            pdf.savefig(page4, facecolor=pal["BG"], bbox_inches="tight")
            plt.close(page4)

        # ── Page 5+: two-factor (group × time point) ANOVA ────────────────────
        if twoway_df is not None and len(twoway_df):
            _twoway_pdf_pages(pdf, twoway_df, twoway_msg, drilldown or {},
                              pair_warn, pal)


# ── Prism-style ("tabular results") formatting helpers ───────────────────────
def _prism_p(p):
    """Format a p-value the way Prism prints it: '<0.0001', else 4 decimals."""
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def _prism_summary(p):
    """Prism's 'P value summary' stars: ns / * / ** / *** / **** ."""
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 0.0001: return "****"
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "ns"


def _prism_sig(p):
    return "Yes" if (p is not None and np.isfinite(p) and p < 0.05) else "No"


def _prism_summary_row(stars, note=None):
    """Prism 'P value summary' that RESPECTS the stats engine's own
    `stars`/`stars_corrected` column instead of re-deriving from the raw
    p-value.  The engine value is already gated on the configured α and
    blanked for underpowered (n<3) comparisons; recomputing from the raw p
    ignored both and could print '****'/'significant' for a comparison the
    engine flagged uninterpretable.  `stars` is stars_for() output: ''
    (underpowered/uninterpretable), 'ns', '*', '**', '***'."""
    s = ("" if stars is None else str(stars)).strip()
    if not s:
        return "underpowered (n<3)" if note else "n/a"
    return s


def _prism_sig_row(stars, note=None):
    """'Yes'/'No'/'n/a' significance gated on the CONFIGURED α via the engine
    `stars` (any star ⇒ p<α; 'ns' ⇒ not significant; '' ⇒ underpowered)."""
    s = ("" if stars is None else str(stars)).strip()
    if not s:
        return "n/a"
    return "No" if s == "ns" else "Yes"


def _fnum(v, fmt="{:.5g}"):
    return fmt.format(v) if (v is not None and np.isfinite(v)) else ""


def _f(v):
    """Coerce to a finite float or None (tolerates blanks/strings from the CSV)."""
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _write_prism_ttests(path, stats_df, stats_config=None):
    """Write the per-metric group comparisons in Prism unpaired-t-test
    'tabular results' style (one sectioned block per metric).  A header block
    records the exact statistical configuration that was applied, so the CSV is
    self-describing."""
    import csv as _csv
    from firefly.analysis.fa_stats_config import (
        normalize_stats_config, config_summary_rows, correction_display)
    cfg = normalize_stats_config(stats_config)
    corr_disp = correction_display(cfg["correction"])
    _alpha = float(cfg["alpha"])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["Group comparisons  (per metric)"])
        # Exact configuration that produced these results.
        w.writerow(["Statistics configuration"])
        for label, value in config_summary_rows(cfg):
            w.writerow([f"    {label}", value])
        w.writerow(["    Unit of analysis", "one value per cell / replicate (not per track)"])
        w.writerow([])
        for metric in dict.fromkeys(stats_df["metric"].tolist()):
            block = stats_df[stats_df["metric"] == metric]
            pairs = block[block["comparison"] != "omnibus"]
            if not len(pairs):
                continue
            w.writerow([f"=====  {metric}  ====="])
            w.writerow(["Table Analyzed", metric])
            omn = block[block["comparison"] == "omnibus"]
            if len(omn):
                r = omn.iloc[0]
                w.writerow(["Omnibus test", r.get("test", "")])
                w.writerow(["    P value", _prism_p(_f(r.get("p_value")))])
                w.writerow(["    P value summary",
                            _prism_summary_row(r.get("stars"), r.get("note"))])
                if r.get("note"):
                    w.writerow(["    Note", str(r.get("note"))])
            w.writerow([])
            for _, r in pairs.iterrows():
                la, lb = r.get("label_a", "A"), r.get("label_b", "B")
                p = _f(r.get("p_value")); padj = _f(r.get("p_value_corrected"))
                pax = _f(r.get("p_value_across_metric"))
                ma, mb = _f(r.get("mean_a")), _f(r.get("mean_b"))
                w.writerow([f"{la}  vs  {lb}"])
                w.writerow(["    " + str(r.get("test", "t test")), ""])
                w.writerow(["    P value (raw)", _prism_p(p)])
                w.writerow(["    P value summary",
                            _prism_summary_row(r.get("stars"), r.get("note"))])
                w.writerow([f"    Significantly different (P<{_alpha:g})?",
                            _prism_sig_row(r.get("stars"), r.get("note"))])
                if r.get("note"):
                    w.writerow(["    Note", str(r.get("note"))])
                # Self-correcting post-hocs (Games-Howell / Tukey / Dunnett)
                # carry their own family-wise label; everyone else gets the
                # chosen within-metric correction.
                cm = str(r.get("correction_method") or "")
                _adj_lbl = (f"    Adjusted P value ({cm})" if "family-wise" in cm
                            else f"    Adjusted P value ({corr_disp}, within metric)")
                w.writerow([_adj_lbl, _prism_p(padj)])
                w.writerow(["    Adjusted summary",
                            _prism_summary_row(r.get("stars_corrected"), r.get("note"))])
                if cfg["across_metric_correction"] and pax is not None:
                    w.writerow([f"    Adjusted P value ({corr_disp}, across metrics)",
                                _prism_p(pax)])
                w.writerow(["How big is the difference?", ""])
                w.writerow([f"    Mean of {la}", _fnum(ma)])
                w.writerow([f"    Mean of {lb}", _fnum(mb)])
                if ma is not None and mb is not None:
                    w.writerow([f"    Difference between means ({la} - {lb})",
                                _fnum(ma - mb)])
                w.writerow(["    SEM of " + str(la), _fnum(_f(r.get("sem_a")))])
                w.writerow(["    SEM of " + str(lb), _fnum(_f(r.get("sem_b")))])
                w.writerow(["    Effect size (Cohen's d)", _fnum(_f(r.get("cohens_d")), "{:.3f}")])
                w.writerow(["    Effect size (Hedges' g)", _fnum(_f(r.get("hedges_g")), "{:.3f}")])
                cd = _f(r.get("cliffs_delta"))
                if cd is not None:
                    clo, chi = _f(r.get("cliffs_delta_ci_low")), _f(r.get("cliffs_delta_ci_high"))
                    ci_txt = (f"  [{clo:.2f}, {chi:.2f}]"
                              if (clo is not None and chi is not None) else "")
                    w.writerow(["    Effect size (Cliff's delta)",
                                f"{cd:.3f}" + ci_txt])
                rb = _f(r.get("rank_biserial"))
                if rb is not None:
                    w.writerow(["    Rank-biserial r", _fnum(rb, "{:.3f}")])
                if cfg["equivalence_tost"]:
                    teq = r.get("tost_equivalent")
                    tp = _f(r.get("tost_p"))
                    verdict = ("Yes" if teq is True
                               else "No" if teq is False else "—")
                    w.writerow([f"    Equivalent within ±{cfg['tost_margin']:g} SD (TOST)?",
                                verdict + (f"  (p={tp:.4g})" if tp is not None else "")])
                w.writerow(["Data analyzed", ""])
                w.writerow([f"    Sample size, {la}", r.get("n_a", "")])
                w.writerow([f"    Sample size, {lb}", r.get("n_b", "")])
                w.writerow(["    n/group for 80% power",
                            r.get("n_per_group_for_80pct_power", "")])
                w.writerow([])
            w.writerow([])


def _write_twoway_csv(path, twoway_df, twoway_msg, drilldown, summary_df,
                      group_order, tp_order, pair_warn, stats_config=None):
    """Write the group × time-point ANOVA report in GraphPad-Prism "tabular
    results" style: per metric, a sectioned vertical sheet with the Source-of-
    Variation summary, the full ANOVA table (SS / DF / MS / F / P), the
    Geisser-Greenhouse epsilon, then the Holm-Šídák multiple comparisons —
    matching how the lab's Prism files lay out an analysis."""
    import csv as _csv

    def _g(df, **eq):
        m = pd.Series(True, index=df.index)
        for k, v in eq.items():
            m &= (df[k] == v)
        return df[m]

    n_cells = summary_df["cell"].nunique() if "cell" in summary_df else 0
    count_bits = []
    for grp in group_order:
        cells = [f"{tp}={_g(summary_df, group=grp, timepoint=tp)['cell'].nunique()}"
                 for tp in tp_order]
        count_bits.append(f"{grp} ({', '.join(cells)})")

    # Effect → (Prism row name, which p-value to use).  Between-subjects (Group)
    # is unaffected by sphericity, so it uses the uncorrected p; the within and
    # interaction effects use the Greenhouse-Geisser-corrected p.
    eff_map = {
        "Interaction": ("Time point x Group (Interaction)", "p_GG"),
        "timepoint":   ("Time point (within-subjects)",      "p_GG"),
        "group":       ("Group (between-subjects)",          "p_unc"),
    }
    eff_order = ["Interaction", "timepoint", "group"]

    def _pick_p(r, pk):
        # GG-corrected p; fall back to uncorrected when pingouin leaves p_GG
        # blank (it does so for a 2-level within factor, where GG=uncorrected).
        p = r.get(pk)
        if (p is None or not np.isfinite(p)) and pk == "p_GG":
            p = r.get("p_unc")
        return p

    metrics = list(dict.fromkeys(twoway_df["metric"].tolist()))

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["Two-way mixed-effects ANOVA  (repeated measures on Time point)"])
        w.writerow(["Model", twoway_msg or ""])
        from firefly.analysis.fa_stats_config import normalize_stats_config, correction_display
        _cfg = normalize_stats_config(stats_config)
        w.writerow(["Sphericity correction", "Greenhouse-Geisser (within-subjects + interaction)"])
        w.writerow(["Post-hoc correction", correction_display(_cfg["correction"])])
        w.writerow(["Significance level (alpha)", f"{_cfg['alpha']:g}"])
        w.writerow(["Groups (between-subjects)", ", ".join(map(str, group_order))])
        w.writerow(["Time points (within-subjects)", ", ".join(map(str, tp_order))])
        w.writerow(["Subjects", "biological replicate (cell), paired across time points"])
        w.writerow(["Cells per group (per time point)", " ; ".join(count_bits)])
        w.writerow(["Pairing", pair_warn or "all cells present at every time point"])
        w.writerow(["Sphericity correction", "Geisser-Greenhouse (applied to within-subjects "
                    "and interaction p-values; for 2 time points it equals the uncorrected p)"])
        w.writerow([])

        for m in metrics:
            a = _g(twoway_df, metric=m, section="anova")
            if not len(a):
                continue
            arows = {r["effect"]: r for _, r in a.iterrows()}
            w.writerow([f"=====  {m}  ====="])
            w.writerow(["Table Analyzed", m])
            w.writerow(["Alpha", "0.05"])
            w.writerow([])
            # Source-of-Variation summary
            w.writerow(["Source of Variation", "P value", "P value summary",
                        "Significant? (P<0.05)"])
            for e in eff_order:
                if e not in arows:
                    continue
                name, pk = eff_map[e]
                p = _pick_p(arows[e], pk)
                w.writerow([name, _prism_p(p), _prism_summary(p), _prism_sig(p)])
            w.writerow([])
            # Full ANOVA table
            w.writerow(["ANOVA table", "SS", "DF", "MS", "F (DFn, DFd)", "P value"])
            for e in eff_order:
                if e not in arows:
                    continue
                r = arows[e]; name, pk = eff_map[e]
                f_str = (f"F ({_fnum(r.get('df1'),'{:.0f}')}, "
                         f"{_fnum(r.get('df2'),'{:.0f}')}) = {_fnum(r.get('F'),'{:.4g}')}")
                w.writerow([name, _fnum(r.get("SS")), _fnum(r.get("df1"), "{:.0f}"),
                            _fnum(r.get("MS")), f_str, _prism_p(_pick_p(r, pk))])
            w.writerow([])
            # epsilon + effect size
            eps = arows.get("timepoint", {}).get("eps") if "timepoint" in arows else None
            w.writerow(["Geisser-Greenhouse's epsilon", _fnum(eps, "{:.4f}")])
            inter_np2 = arows.get("Interaction", {}).get("np2") if "Interaction" in arows else None
            w.writerow(["Partial eta squared (Interaction)", _fnum(inter_np2, "{:.4f}")])
            w.writerow(["Number of subjects (cells)", n_cells])
            w.writerow([])

            # Multiple comparisons — Prism shows the simple effects (groups
            # compared at each time point).  Keep only those interaction
            # contrasts (an 'at' time point set); the bare main-effect rows
            # would just duplicate the ANOVA above.
            post = _g(twoway_df, metric=m, section="posthoc")
            def _has_at(v):
                # pingouin marks main-effect rows with "-" in the time column;
                # keep only real time-point levels (the interaction simple effects).
                return v not in (None, "", "-") and str(v) not in ("nan", "-")
            simple = post[post["at"].map(_has_at)] if len(post) else post
            if len(simple):
                w.writerow(["Multiple comparisons (groups at each time point, Holm-Šídák)",
                            "P value (uncorrected)", "Adjusted P value",
                            "Summary", "Significant?"])
                for _, r in simple.iterrows():
                    name = f"{r.get('level_A')} vs {r.get('level_B')}  @ {r.get('at')}"
                    padj = r.get("p_holm"); pu = r.get("p")
                    keyp = padj if padj is not None else pu
                    w.writerow([name, _prism_p(pu), _prism_p(padj),
                                _prism_summary(keyp), _prism_sig(keyp)])
                w.writerow([])
            w.writerow([])

        # Per-time-point curve drill-downs, Prism-style per time point.
        for kind in ("msd", "logd"):
            if kind not in drilldown:
                continue
            ddf, dmsg = drilldown[kind]
            w.writerow([f"=====  Curve drill-down: {kind.upper()}  ====="])
            w.writerow(["Model", dmsg])
            w.writerow([])
            for tp in dict.fromkeys(ddf["timepoint"].tolist()):
                sub = ddf[ddf["timepoint"] == tp]
                w.writerow([f"Time point: {tp}", "SS", "DF", "MS",
                            "F (DFn, DFd)", "P value"])
                for _, r in sub.iterrows():
                    pk = "p_GG" if r.get("effect") in ("Interaction", "lag", "bin") else "p_unc"
                    f_str = (f"F ({_fnum(r.get('df1'),'{:.0f}')}, "
                             f"{_fnum(r.get('df2'),'{:.0f}')}) = {_fnum(r.get('F'),'{:.4g}')}")
                    w.writerow([r.get("effect"), "", _fnum(r.get("df1"), "{:.0f}"),
                                "", f_str, _prism_p(r.get(pk))])
                w.writerow([])

    print(f"  Saved: {path}")


def _twoway_table_page(pdf, title, subtitle, df, cols, pal):
    """Render one themed table page from a DataFrame subset."""
    import matplotlib.pyplot as plt
    if df is None or not len(df):
        return
    page = plt.figure(figsize=(11, 8.5), facecolor=pal["BG"])
    page.text(0.5, 0.965, title, ha="center", fontsize=14, fontweight="bold",
              color=pal["TXT"])
    if subtitle:
        page.text(0.5, 0.93, subtitle, ha="center", fontsize=8,
                  color=pal["TXT"], wrap=True)
    ax = page.add_axes([0.03, 0.04, 0.94, 0.84])
    ax.axis("off")
    disp = df[cols].copy()
    for c in cols:
        disp[c] = disp[c].apply(
            lambda x: (f"{x:.4g}" if isinstance(x, float) and np.isfinite(x)
                       else ("" if x is None else x)))
    tbl = ax.table(cellText=disp.values.tolist(), colLabels=cols,
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(pal["GRD"])
        cell.set_text_props(color=pal["TXT"])
        cell.set_facecolor(pal["PNL"] if r > 0 else pal["BG"])
        if r == 0:
            cell.set_text_props(weight="bold", color=pal["TXT"])
    pdf.savefig(page, facecolor=pal["BG"], bbox_inches="tight")
    plt.close(page)


def _twoway_pdf_pages(pdf, twoway_df, twoway_msg, drilldown, pair_warn, pal):
    """Render the two-factor ANOVA pages into an open PdfPages."""
    note = (twoway_msg or "")
    if pair_warn:
        note += "\n" + pair_warn
    anova = twoway_df[twoway_df["section"] == "anova"]
    _twoway_table_page(pdf, "Group × Time-point ANOVA — scalar metrics", note,
                       anova, ["metric", "effect", "F", "df1", "df2",
                               "p_unc", "p_GG", "np2", "eps"], pal)
    post = twoway_df[twoway_df["section"] == "posthoc"]
    _twoway_table_page(pdf, "Simple effects (Holm-corrected)", "",
                       post, ["metric", "contrast", "at", "level_A", "level_B",
                              "paired", "p", "p_holm", "stars"], pal)
    for kind in ("msd", "logd"):
        if kind in drilldown:
            ddf, dmsg = drilldown[kind]
            _twoway_table_page(
                pdf, f"Curve drill-down — {kind.upper()}", dmsg, ddf,
                ["graph", "timepoint", "effect", "F", "df1", "df2",
                 "p_unc", "p_GG", "np2", "eps"], pal)
