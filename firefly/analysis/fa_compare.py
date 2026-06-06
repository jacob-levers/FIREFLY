"""Multi-group comparison figure, statistics and PDF report.

Extracted from sptpalm_analysis.py (#7); re-exported there for compatibility.
"""
from __future__ import annotations

import os
from firefly.analysis.fa_constants import (MOTION_CLASS_COLORS, MOTION_CLASS_ORDER,
                                           motion_class_colors)
from firefly.analysis.fa_theme import _theme_palette, style_axes
from firefly.analysis.fa_palmtracer import load_summary_from_folder

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
                     xtick_labels=None, stats_config=None, annot_sink=None):
    """Bar chart with mean ± SEM and individual replicate dots, generalised
    to N groups.

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
    ax.bar(x, means, yerr=sems, capsize=4,
           color=[_bar_fill_for(i) for i in range(n)],
           edgecolor=colors, linewidth=1.6,
           ecolor=sig_col)
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
    omnibus, pairwise = _stat_test_n(arrs, labels, cfg)
    # Within-metric correction onto the pairwise list (also done centrally for
    # the CSV; computed here so the initial draw is already correct).
    _wp = correct_pvalues([pw["p"] for pw in pairwise], cfg["correction"])
    for k, pw in enumerate(pairwise):
        pw["p_within"] = _wp[k]
        pw["stars_within"] = stars_for(_wp[k], cfg["alpha"])
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
            lines.append(corr_caption)
            if pair.get("note"):
                lines.append(pair["note"])
            return "\n".join([ln for ln in lines if ln])
        if n > 2 and omnibus:
            pv = omnibus["p"]
            p_str = f"p = {pv:.2e}" if pv < 0.001 else f"p = {pv:.3f}"
            lines = [omnibus["test"], f"{p_str}   {omnibus['stars']}",
                     f"pairwise: {corr_caption}"]
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
            p, stars = _paired_test(a, b, stats_config)
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


def compare_groups(groups,
                   output_dir=None, output_stem="comparison",
                   panels=None, theme="Dark",
                   pdf_report=True,
                   mobile_d_threshold=MOBILE_D_THRESHOLD_DEFAULT,
                   progress_cb=None, stats_config=None):
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
    from firefly.analysis.fa_stats_config import normalize_stats_config
    cfg = normalize_stats_config(stats_config)
    # Bar-panel annotation handles, so the optional across-metric correction
    # post-pass can update the on-figure stars to agree with the CSV.
    panel_annots = {}

    if len(groups) < 2:
        raise CompareInputError(
            f"A comparison needs at least 2 groups; only {len(groups)} was given. "
            "Add another group on the Compare tab.")

    if panels is None:
        panels = {"msd", "auc", "logd_dist", "mob_immob", "motion_classes",
                  "track_length", "jdd", "dwell_cdf", "turning_angles",
                  "radial_dist", "van_hove", "vacf"}

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
    two_factor = len(timepoint_tokens) >= 2
    labels = [(f"{group_factor[i]} / {timepoints_per_card[i]}"
               if (two_factor and timepoints_per_card[i]) else group_factor[i])
              for i in range(n_groups)]
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
                all_summaries[gi].append(load_summary_from_folder(f))
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
        fi = float(p.get("frame_interval_s", 0.05))
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
        twoway_df, twoway_msg = fa_twoway.compute_twoway_anova(paired_df, stats_config=cfg)
        print(f"  Two-way ANOVA: {twoway_msg}")

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
    ncols = 3 if n_plots > 4 else 2
    nrows = (n_plots + ncols - 1) // ncols

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

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.6),
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

    # ── 1. MSD overlay ────────────────────────────────────────────────────────
    if "msd" in panels:
        ax = _next_ax()
        for grp_label, summaries, color in _zip_groups():
            curves = []
            tref = None
            for s in summaries:
                e = s["ensemble_msd"]
                if e is None: continue
                fi = float(s["params"].get("frame_interval_s", 0.05))
                t = e["lag_frame"].values * fi
                y = e["msd_um2"].values
                order = np.argsort(t)
                t, y = t[order], y[order]
                if tref is None:
                    tref = t
                if len(t) != len(tref) or not np.allclose(t, tref):
                    y = np.interp(tref, t, y)
                curves.append(y)
            if not curves:
                continue
            arr = np.vstack(curves)
            mean = arr.mean(axis=0)
            sem = arr.std(axis=0, ddof=1) / np.sqrt(len(curves)) if len(curves) > 1 else None
            ax.plot(tref, mean, "-o", color=color, label=grp_label, ms=4, lw=1.5)
            if sem is not None:
                ax.fill_between(tref, mean - sem, mean + sem, color=color, alpha=0.15)
        ax.set_xlabel("Time delta (s)")
        ax.set_ylabel("MSD (µm²)")
        ax.set_title("Mean Square Displacement")
        ax.legend(frameon=False, loc="best")

    # ── 2. AUC bar chart ──────────────────────────────────────────────────────
    if "auc" in panels:
        ax = _next_ax()
        if two_factor:
            _interaction_plot(ax, summary_df, "auc_msd", group_order, tp_order,
                              group_colors, pal, ylabel="AUC (µm²·s)",
                              headline=_twoway_headline(twoway_df, "auc_msd"),
                              card_colors=card_colors, stats_config=cfg)
        else:
            data = [summary_df.loc[summary_df["group"] == lbl, "auc_msd"].values
                    for lbl in labels]
            _bar_with_dots_n(ax, data, labels, colors, pal,
                             ylabel="AUC (µm²·s)",
                             record_stats=stats_records, metric_name="auc_msd", xtick_labels=bar_xticks, stats_config=cfg, annot_sink=panel_annots)
        ax.set_title("Area Under the Curve")

    # ── 3. LogD frequency distribution ────────────────────────────────────────
    if "logd_dist" in panels:
        ax = _next_ax()
        bins = np.linspace(-5, 1, 31)
        for grp_label, summaries, color in _zip_groups():
            all_logD = []
            for s in summaries:
                d = s["diffusion"]
                if d is None or "D" not in d.columns: continue
                vals = d["D"].values
                vals = vals[vals > 0]
                if len(vals): all_logD.append(np.log10(vals))
            if not all_logD: continue
            pooled = np.concatenate(all_logD)
            # Clip the sub-resolution (immobile) tail to the bin range so it
            # piles into the first bin and stays IN the normalisation, instead
            # of being silently dropped by np.histogram (which would inflate the
            # mobile frequencies and hide an immobilising drug effect).  Both
            # FIREFLY and PALM-Tracer produce this ~10-12% immobile floor.
            pooled = np.clip(pooled, bins[0], bins[-1])
            counts, edges = np.histogram(pooled, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            frac = counts / counts.sum() if counts.sum() else counts
            ax.plot(centers, frac, "-o", color=color, label=grp_label, ms=4, lw=1.2)
        ax.axvline(np.log10(mobile_d_threshold), color=pal["GRD"], ls="--", lw=0.8,
                   label=f"D = {mobile_d_threshold} µm²/s")
        ax.set_xlabel("log₁₀ D  (µm²/s)")
        ax.set_ylabel("Relative frequency")
        ax.set_title("LogD Frequency Distribution")
        ax.legend(frameon=False, loc="best")

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
                             record_stats=stats_records, metric_name="mob_immob_ratio", xtick_labels=bar_xticks, stats_config=cfg, annot_sink=panel_annots)
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
        def _txt_on(hexcol):
            """Black or white label text, whichever has the higher WCAG contrast
            against the fill (robust across themes incl. the Publication
            colour-blind palette, where a naive luminance cut mislabels amber)."""
            h = hexcol.lstrip("#")
            r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
            def _lin(c):
                return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
            L = 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)
            return "#101010" if (L + 0.05) / 0.05 >= 1.05 / (L + 0.05) else "#ffffff"
        # Each replicate's 4 named-class fractions are renormalised to sum to 1
        # (matching the single-run pie, which auto-normalises the same 4
        # classes), so the stacked bars reach the top.  The dropped mass is the
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
                            color=_txt_on(ccol), zorder=4)
            bottom += seg
        # Per-class one-way stats — only meaningful when each card is an
        # independent group.  In two-factor mode the cards are paired across
        # time points, so a one-way test across them is invalid; the two-way
        # ANOVA report covers it instead.
        if not two_factor:
            for ci, cname in enumerate(classes):
                arrs = [fr[:, ci] if len(fr) else np.array([]) for fr in per_group]
                omn, pw = _stat_test_n(arrs, labels, cfg)
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
                fi = float(s["params"].get("frame_interval_s", 0.05))
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
            omn, pw = _stat_test_n(arrs, labels, cfg)
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
                             xtick_labels=bar_xticks, stats_config=cfg, annot_sink=panel_annots)
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
        for grp_label, color, pooled in pooled_per_group:
            if not pooled: continue
            any_data = True
            counts, _ = np.histogram(pooled, bins=bins)
            frac = counts / counts.sum() if counts.sum() else counts
            ax.plot(centers, frac, "-o", color=color, lw=1.5, ms=3, label=grp_label)
        if any_data:
            ax.set_xlabel("|Turning angle|  (°)")
            ax.set_ylabel("Relative frequency")
            ax.set_xlim(0, 180)
            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_title("Turning Angle Distribution")
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
                             metric_name="nongauss_alpha2", xtick_labels=bar_xticks, stats_config=cfg, annot_sink=panel_annots)
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
                             metric_name="vacf_persistence", xtick_labels=bar_xticks, stats_config=cfg, annot_sink=panel_annots)
        ax.set_title("Directional persistence (VACF lag 1)")

    # ── Single shared legend ─────────────────────────────────────────────────
    # Per-panel `loc="best"` legends overlap the data badly once there are many
    # groups / time points, and they repeat the same key in every panel.  Drop
    # every per-panel legend and place ONE deterministic legend (group order)
    # in a reserved bottom strip so nothing covers the plots.  When the bars use
    # short numeric x-tick tokens, prefix the legend entries with the same
    # number so the axis ↔ legend mapping is explicit.
    from matplotlib.lines import Line2D
    for ax in axes[:n_plots]:
        if getattr(ax, "_firefly_keep_legend", False):
            continue  # panel carries its own colour key (e.g. motion classes)
        _lg = ax.get_legend()
        if _lg is not None:
            _lg.remove()

    number_legend = many_groups and not two_factor
    leg_handles, leg_labels = [], []
    for i in range(n_groups):
        leg_handles.append(Line2D([0], [0], color=colors[i], marker="o",
                                   lw=2.0, ms=5))
        tag = f"{i + 1}.  " if number_legend else ""
        leg_labels.append(f"{tag}{labels[i]}  (n={len(all_summaries[i])})")

    legend_rows = 0
    if leg_handles:
        ncol = min(len(leg_labels), 4)
        legend_rows = (len(leg_labels) + ncol - 1) // ncol
        fig.legend(leg_handles, leg_labels, loc="lower center", ncol=ncol,
                   frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0),
                   labelcolor=pal["TXT"])

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
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", color=pal["TXT"])
    for ax in axes[:n_plots]:
        ax.set_facecolor(pal["PNL"])
        # Modern look: drop the top/right spines, thin the rest (polar +
        # image panels keep their frame — handled inside style_axes).
        style_axes(ax, pal,
                   kind=("polar" if getattr(ax, "name", "") == "polar"
                         else "cartesian"))
    # Reserve a bottom strip for the shared legend (grows with its row count).
    bottom = min(0.18, 0.03 + 0.026 * legend_rows) if legend_rows else 0.0
    fig.tight_layout(rect=[0, bottom, 1, 0.96])

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

    # Population-heterogeneity (non-Gaussian alpha2) and directionality (VACF
    # persistence) are per-replicate scalars in summary_df; test them across
    # groups (flat mode) so they appear in the stats table and CSV even though
    # they have no dedicated figure panel.  The two-way report covers the
    # two-factor case separately.
    if not two_factor:
        for _m in ("nongauss_alpha2", "vacf_persistence"):
            if _m in summary_df.columns and _m not in stats_records:
                arrs = [summary_df.loc[summary_df["group"] == lbl, _m]
                        .dropna().to_numpy() for lbl in labels]
                if sum(len(a) for a in arrs) >= 2:
                    omn, pw = _stat_test_n(arrs, labels, cfg)
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
        wcorr = correct_pvalues([pw.get("p") for pw in pairs], _method)
        for k, pw in enumerate(pairs):
            pw["p_within"] = wcorr[k]
            pw["stars_within"] = stars_for(wcorr[k], _alpha)
            pw.setdefault("p_across", np.nan)
            pw.setdefault("stars_across", "")
            if metric in _ACROSS_FAMILY and np.isfinite(pw.get("p", np.nan)):
                across_pw.append(pw)
    family_size = len(across_pw)
    if cfg["across_metric_correction"] and across_pw:
        acorr = correct_pvalues([pw["p"] for pw in across_pw], _method)
        for pw, ap in zip(across_pw, acorr):
            pw["p_across"] = ap
            pw["stars_across"] = stars_for(ap, _alpha)

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
                "n_per_group_for_80pct_power": "",
                "n_per_group_for_90pct_power": "",
                "note": omn.get("note", ""),
            })
        for pw in rec.get("pairwise", []):
            stats_rows.append({
                "metric": metric, "comparison": f"{pw['label_i']} vs {pw['label_j']}",
                "test": pw["test"],
                "p_value": pw["p"], "stars": pw["stars"],
                "correction_method": correction_display(_method),
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
        fig.savefig(png_path, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        summary_df.to_csv(csv_path, index=False)
        if len(stats_df):
            _write_prism_ttests(stats_csv, stats_df, stats_config=cfg)
        print(f"  Saved: {png_path}")
        print(f"  Saved: {pdf_path}")
        print(f"  Saved: {csv_path}")
        if len(stats_df):
            print(f"  Saved: {stats_csv}")

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
        try:
            groups_angles_pooled = []
            # Per-track (mean_angle_deg, D) pairs per group, used for
            # the circular-linear correlation between a track's
            # average turning bias and its diffusion coefficient.
            # One list of pairs per group; each list pools across
            # the group's replicates.
            track_angle_d_pairs = []
            # Per-replicate angle arrays — one list of arrays per group.
            # Used to compute per-replicate κ, R̄, μ for the Welch t-test
            # and per-replicate Watson-Williams F-test (treats each
            # replicate as one data point, the statistically defensible
            # framing for n=5 vs n=3 designs).
            per_replicate_angles = {}
            for label, ss, color in zip(labels, all_summaries, colors):
                pooled = []
                t_angles_g = []
                t_D_g      = []
                rep_angle_arrays = []
                for s in ss:
                    ta = s.get("turning_angles")
                    if ta is not None:
                        arr = np.asarray(ta, dtype=float).ravel()
                        if arr.size:
                            pooled.append(arr)
                            rep_angle_arrays.append(arr)
                    tracks = s.get("tracks")
                    diff_df = s.get("diffusion")
                    if tracks is None or diff_df is None:
                        continue
                    if "D" not in diff_df.columns:
                        continue
                    try:
                        pairs = compute_per_track_mean_angle(tracks)
                        if not pairs:
                            continue
                        d_map = dict(zip(diff_df["particle"].astype(int),
                                         diff_df["D"].astype(float)))
                        for pid, mu_deg in pairs:
                            d_val = d_map.get(int(pid))
                            if d_val is None or not np.isfinite(d_val):
                                continue
                            t_angles_g.append(float(mu_deg))
                            t_D_g.append(float(d_val))
                    except Exception:
                        continue
                pooled_arr = (np.concatenate(pooled)
                              if pooled else np.array([], dtype=float))
                groups_angles_pooled.append((label, pooled_arr, color))
                track_angle_d_pairs.append(
                    (np.asarray(t_angles_g, dtype=float),
                     np.asarray(t_D_g,      dtype=float)))
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
                track_angle_d_pairs=track_angle_d_pairs,
                per_replicate_angles=per_replicate_angles)
            print(f"  Saved: {cs_pdf}")
        except Exception as exc:
            print(f"  Comparison circular-stats skipped "
                  f"({type(exc).__name__}: {exc})")

    return fig, summary_df, stats_records


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
            disp = stats_df.copy()
            for c in ("p_value", "mean_a", "mean_b", "sem_a", "sem_b"):
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
    with open(path, "w", newline="") as fh:
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
                w.writerow(["    P value summary", _prism_summary(_f(r.get("p_value")))])
            w.writerow([])
            for _, r in pairs.iterrows():
                la, lb = r.get("label_a", "A"), r.get("label_b", "B")
                p = _f(r.get("p_value")); padj = _f(r.get("p_value_corrected"))
                pax = _f(r.get("p_value_across_metric"))
                ma, mb = _f(r.get("mean_a")), _f(r.get("mean_b"))
                w.writerow([f"{la}  vs  {lb}"])
                w.writerow(["    " + str(r.get("test", "t test")), ""])
                w.writerow(["    P value (raw)", _prism_p(p)])
                w.writerow(["    P value summary", _prism_summary(p)])
                w.writerow(["    Significantly different (P<0.05)?", _prism_sig(p)])
                w.writerow([f"    Adjusted P value ({corr_disp}, within metric)",
                            _prism_p(padj)])
                w.writerow(["    Adjusted summary", _prism_summary(padj)])
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

    with open(path, "w", newline="") as fh:
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
