"""Multi-group comparison figure, statistics and PDF report.

Extracted from sptpalm_analysis.py (#7); re-exported there for compatibility.
"""
from __future__ import annotations

import os
from firefly.analysis.fa_theme import _theme_palette
from firefly.analysis.fa_palmtracer import load_summary_from_folder

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as _stats
from firefly.analysis.fa_diffusion import (_msd_auc, _mob_immob_ratio, MOBILE_D_THRESHOLD_DEFAULT,
                          _motion_fractions, _track_lengths)
from firefly.analysis.fa_circular import (save_comparison_circular_statistics,
                         _stat_test, _stat_test_n,
                         _p_stars, compute_per_track_mean_angle,
                         compute_circular_comparison_tests)
from firefly.analysis import fa_twoway






def _bar_with_dots_n(ax, data_per_group, labels, colors, palette,
                     ylabel="", record_stats=None, metric_name=""):
    """Bar chart with mean ± SEM and individual replicate dots, generalised
    to N groups.

    For 2 groups: shows pairwise stars on a bracket (matches lab style).
    For 3+ groups: shows omnibus ANOVA / Kruskal p-value as a panel
    annotation; full pairwise comparisons go to record_stats[metric_name]."""
    fill = palette["BAR_FILL"]
    sig_col = palette["SIG"]

    arrs = [np.asarray(d, dtype=float) for d in data_per_group]
    arrs = [a[np.isfinite(a)] for a in arrs]
    n = len(arrs)
    means = [float(a.mean()) if len(a) else 0.0 for a in arrs]
    sems  = [float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
             for a in arrs]
    x = np.arange(n)
    ax.bar(x, means, yerr=sems, capsize=4,
           color=[fill] * n,
           edgecolor=colors, linewidth=1.5,
           ecolor=sig_col)
    rng = np.random.default_rng(0)
    for i, a in enumerate(arrs):
        if len(a):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(a)), a,
                       color=colors[i], s=18, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15 if n > 3 else 0)
    ax.set_ylabel(ylabel)

    # Stats
    omnibus, pairwise = _stat_test_n(arrs, labels)
    if record_stats is not None and metric_name:
        record_stats[metric_name] = {"omnibus": omnibus, "pairwise": pairwise}

    # Annotation
    top_data = max([a.max() if len(a) else 0 for a in arrs] + [max(means) * 1.2 if max(means) > 0 else 1])
    if n == 2 and pairwise:
        pair = pairwise[0]
        if pair["stars"] and np.isfinite(pair["p"]):
            top = top_data * 1.05
            ax.plot([0, 0, 1, 1], [top, top * 1.03, top * 1.03, top],
                    color=sig_col, lw=0.8)
            # Numeric p plus stars, e.g. "p = 0.003  **"
            p_str = (f"p = {pair['p']:.2e}" if pair['p'] < 0.001
                     else f"p = {pair['p']:.3f}")
            label = f"{p_str}  {pair['stars']}"
            ax.text(0.5, top * 1.05, label, ha="center", va="bottom",
                    fontsize=9, color=sig_col)
            # Make room above the bracket for the longer label
            ax.set_ylim(0, top * 1.30)
    elif n > 2 and omnibus:
        # Show test name + omnibus p + stars in the upper-left corner.
        # Numeric format adapts to magnitude: scientific < 0.001, fixed otherwise.
        p_val = omnibus['p']
        p_str = (f"p = {p_val:.2e}" if p_val < 0.001
                 else f"p = {p_val:.3f}")
        text = f"{omnibus['test']}\n{p_str}   {omnibus['stars']}"
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color=sig_col,
                bbox=dict(facecolor=palette["PNL"], edgecolor="none",
                          alpha=0.7, pad=3))


# Qualitative colours for the time-point series in two-factor interaction plots.
_TP_SERIES_COLORS = ["#3b6ed8", "#f78166", "#56d364", "#d2a8ff",
                     "#ffa657", "#79c0ff", "#e3b341", "#ff7b72"]


def _interaction_plot(ax, summary_df, metric, group_order, tp_order,
                      group_colors, palette, ylabel=""):
    """Group × time-point interaction plot: x = TIME POINTS (in the order the
    user assigned them), one mean±SEM line per group drawn in that group's
    assigned colour — a time-course view.  Cell-level points are jittered
    behind each mean."""
    x = np.arange(len(tp_order))
    rng = np.random.default_rng(2)
    for gi, grp in enumerate(group_order):
        col = (group_colors or {}).get(grp) \
            or _TP_SERIES_COLORS[gi % len(_TP_SERIES_COLORS)]
        means, sems = [], []
        for ti, tp in enumerate(tp_order):
            vals = summary_df.loc[(summary_df["group"] == grp)
                                  & (summary_df["timepoint"] == tp),
                                  metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                means.append(float(np.mean(vals)))
                sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                            if len(vals) > 1 else 0.0)
                ax.scatter(np.full(len(vals), x[ti]) + rng.uniform(-0.06, 0.06, len(vals)),
                           vals, color=col, s=12, alpha=0.6, zorder=2)
            else:
                means.append(np.nan); sems.append(0.0)
        ax.errorbar(x, means, yerr=sems, color=col, marker="o", ms=5,
                    lw=1.8, capsize=3, label=str(grp), zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(tp_order)
    ax.set_xlim(-0.35, len(tp_order) - 0.65)   # pad so end markers aren't clipped
    ax.set_xlabel("Time point")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, loc="best", fontsize=8, title="Group")


def compare_groups(groups,
                   output_dir=None, output_stem="comparison",
                   panels=None, theme="Dark",
                   pdf_report=True,
                   mobile_d_threshold=MOBILE_D_THRESHOLD_DEFAULT,
                   progress_cb=None):
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

    if len(groups) < 2:
        raise ValueError(f"Need at least 2 groups; got {len(groups)}")

    if panels is None:
        panels = {"msd", "auc", "logd_dist", "mob_immob", "motion_classes",
                  "track_length", "jdd", "dwell_cdf", "turning_angles",
                  "radial_dist"}

    n_groups = len(groups)
    # `group_factor` is the raw group label of each card (the between-subjects
    # factor); `timepoints_per_card` is the optional within-subjects factor.
    # When ANY card carries a time point we enter two-factor (group × time
    # point) mode: several cards can share a group label but differ in time
    # point.  `labels` is the per-card DISPLAY label (group / time point) used
    # in legends and the suptitle so duplicated group names stay distinct.
    group_factor = [g.get("label", f"Group {i+1}") for i, g in enumerate(groups)]
    timepoints_per_card = [str(g.get("timepoint", "")).strip() for g in groups]
    two_factor = any(timepoints_per_card)
    timepoint_tokens = sorted({t for t in timepoints_per_card if t})
    labels = [(f"{group_factor[i]} / {timepoints_per_card[i]}"
               if (two_factor and timepoints_per_card[i]) else group_factor[i])
              for i in range(n_groups)]
    colors   = [g.get("color", "#3b6ed8")     for g in groups]
    folder_lists = [list(g["folders"]) for g in groups]

    # ── Load summaries for all groups ─────────────────────────────────────────
    all_summaries = [[] for _ in groups]
    total = sum(len(f) for f in folder_lists)
    done = 0
    for gi, folders in enumerate(folder_lists):
        for f in folders:
            if progress_cb:
                progress_cb(done, total, f"Loading: {os.path.basename(f)}")
            try:
                all_summaries[gi].append(load_summary_from_folder(f))
            except Exception as e:
                print(f"  Skipping {f}: {e}")
            done += 1

    empty_groups = [labels[i] for i, ss in enumerate(all_summaries) if len(ss) == 0]
    if empty_groups:
        raise RuntimeError(
            "Need at least one valid folder per group; these are empty: "
            + ", ".join(empty_groups))

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
    group_colors = {}
    for i, gf in enumerate(group_factor):
        group_colors.setdefault(gf, colors[i])

    # Per-metric statistics dict — populated as panels render
    stats_records = {}

    # ── Render the figure ────────────────────────────────────────────────────
    panel_order = ["msd", "auc", "logd_dist", "mob_immob",
                   "motion_classes", "track_length",
                   "jdd", "dwell_cdf", "turning_angles", "radial_dist"]
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
        "grid.color":      pal["GRD"], "grid.alpha": 0.4,
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
                              group_colors, pal, ylabel="AUC (µm²·s)")
        else:
            data = [summary_df.loc[summary_df["group"] == lbl, "auc_msd"].values
                    for lbl in labels]
            _bar_with_dots_n(ax, data, labels, colors, pal,
                             ylabel="AUC (µm²·s)",
                             record_stats=stats_records, metric_name="auc_msd")
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
                              ylabel="Mobile/Immobile ratio")
        else:
            data = [summary_df.loc[summary_df["group"] == lbl, "mob_immob_ratio"].values
                    for lbl in labels]
            _bar_with_dots_n(ax, data, labels, colors, pal,
                             ylabel="Mobile/Immobile ratio",
                             record_stats=stats_records, metric_name="mob_immob_ratio")
        ax.set_title("Mobile/Immobile Ratio")

    # ── 5. Motion class fractions (grouped bars, N groups) ────────────────────
    if "motion_classes" in panels:
        ax = _next_ax()
        classes = ["Immobile", "Confined", "Brownian", "Directed"]
        def _fracs(summaries):
            rows = []
            for s in summaries:
                f = _motion_fractions(s["diffusion"])
                rows.append([f.get(c, 0.0) for c in classes])
            return np.array(rows) if rows else np.zeros((0, len(classes)))
        per_group = [_fracs(ss) for ss in all_summaries]
        x = np.arange(len(classes))
        # Group-bar width: total slot ~0.8, divided across N groups
        slot = 0.8
        w = slot / n_groups
        rng = np.random.default_rng(1)
        for gi, (grp_label, color, fracs) in enumerate(zip(labels, colors, per_group)):
            if not len(fracs): continue
            x_off = (gi - (n_groups - 1) / 2) * w
            ax.bar(x + x_off, fracs.mean(axis=0), w * 0.9,
                   yerr=fracs.std(axis=0, ddof=1)/np.sqrt(len(fracs)) if len(fracs) > 1 else None,
                   color=pal["BAR_FILL"], edgecolor=color, linewidth=1.5,
                   ecolor=pal["SIG"], capsize=3, label=grp_label)
            for ci in range(len(classes)):
                ax.scatter(np.full(len(fracs), x[ci] + x_off)
                           + rng.uniform(-w*0.25, w*0.25, len(fracs)),
                           fracs[:, ci], color=color, s=12, zorder=3)
        # Per-class one-way stats — only meaningful when each card is an
        # independent group.  In two-factor mode the cards are paired across
        # time points, so a one-way test across them is invalid; the two-way
        # ANOVA report covers it instead.
        if not two_factor:
            for ci, cname in enumerate(classes):
                arrs = [fracs[:, ci] if len(fracs) else np.array([]) for fracs in per_group]
                omn, pw = _stat_test_n(arrs, labels)
                stats_records[f"motion_frac_{cname}"] = {"omnibus": omn, "pairwise": pw}
        ax.set_xticks(x); ax.set_xticklabels(classes, rotation=15)
        ax.set_ylabel("Fraction of tracks")
        ax.set_title("Motion Class Fractions")
        ax.legend(frameon=False, loc="best", fontsize=8)

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
            omn, pw = _stat_test_n(arrs, labels)
            stats_records["mean_track_length_s"] = {"omnibus": omn, "pairwise": pw}

    # ── 7. JDD: per-population D + fraction (N groups) ────────────────────────
    if "jdd" in panels:
        ax = _next_ax()
        any_data = False
        max_pop_overall = 0
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
                max_pop_overall = max(max_pop_overall, len(D))
                sizes = 25 + 175 * np.clip(f, 0, 1)
                xs = np.arange(len(D)) + offsets[gi]
                ax.scatter(xs, D, s=sizes, color=color,
                           alpha=0.55, edgecolor=color,
                           label=(grp_label if not label_done else None))
                label_done = True
        if any_data:
            tick_labels = ["Immobile", "Mobile", "Fast"][:max_pop_overall]
            if max_pop_overall == 1: tick_labels = ["All"]
            ax.set_xticks(np.arange(max_pop_overall))
            ax.set_xticklabels(tick_labels)
            ax.set_xlim(-0.5, max_pop_overall - 0.5)
            ax.set_ylabel("D (µm²/s, log)")
            ax.set_yscale("log")
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
            ax.grid(True, ls=":", alpha=0.4)
        else:
            ax.text(0.5, 0.5, "No turning-angle data",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Radial Distribution")

    # ── Suptitle: Group A (n=…) vs Group B (n=…) [vs Group C …] ───────────────
    parts = [f"{labels[i]}  (n={len(all_summaries[i])})" for i in range(n_groups)]
    fig.suptitle("   vs   ".join(parts),
                 fontsize=12, fontweight="bold", color=pal["TXT"])
    for ax in axes[:n_plots]:
        ax.set_facecolor(pal["PNL"])
        for spine in ax.spines.values():
            spine.set_edgecolor(pal["GRD"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])

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

    stats_rows = []
    for metric, rec in stats_records.items():
        omn = rec.get("omnibus")
        if omn:
            stars = omn["stars"]
            stars_bonf = stars  # omnibus needs no correction
            stats_rows.append({
                "metric": metric, "comparison": "omnibus",
                "test": omn["test"],
                "p_value": omn["p"], "stars": stars,
                "p_value_bonferroni": omn["p"], "stars_bonferroni": stars_bonf,
                "n_a": "", "n_b": "", "mean_a": "", "mean_b": "",
                "sem_a": "", "sem_b": "", "label_a": "all groups", "label_b": "",
                "cohens_d": "",
                "n_per_group_for_80pct_power": "",
                "n_per_group_for_90pct_power": "",
            })
        pairs = rec.get("pairwise", [])
        n_pairs = max(1, len(pairs))
        for pw in pairs:
            p = pw["p"]
            if np.isfinite(p):
                p_bonf = min(1.0, p * n_pairs)
                if   p_bonf < 0.001: stars_bonf = "***"
                elif p_bonf < 0.01:  stars_bonf = "**"
                elif p_bonf < 0.05:  stars_bonf = "*"
                else:                stars_bonf = "ns"
            else:
                p_bonf = np.nan
                stars_bonf = ""
            stats_rows.append({
                "metric": metric, "comparison": f"{pw['label_i']} vs {pw['label_j']}",
                "test": pw["test"],
                "p_value": pw["p"], "stars": pw["stars"],
                "p_value_bonferroni": p_bonf, "stars_bonferroni": stars_bonf,
                "n_a": pw["n_i"], "n_b": pw["n_j"],
                "mean_a": pw["mean_i"], "mean_b": pw["mean_j"],
                "sem_a": pw["sem_i"], "sem_b": pw["sem_j"],
                "label_a": pw["label_i"], "label_b": pw["label_j"],
                "cohens_d": pw.get("cohens_d"),
                "n_per_group_for_80pct_power": _fmt_n_needed(pw.get("n_needed_80")),
                "n_per_group_for_90pct_power": _fmt_n_needed(pw.get("n_needed_90")),
            })
    stats_df = pd.DataFrame(stats_rows)

    # ── Two-factor (group × time point) mixed ANOVA ───────────────────────────
    # Paired design: between=group, within=timepoint, subject=cell.  Scalars run
    # directly; the curve graphs (MSD, LogD) get a per-time-point group×(lag|bin)
    # drill-down because pingouin can't fit 1-between + 2-within in one model.
    twoway_df, twoway_msg, pair_warn = None, None, None
    drilldown = {}
    if two_factor:
        paired_df, pair_warn, _dropped = fa_twoway.validate_pairing(summary_df)
        if pair_warn:
            print(f"  Two-way pairing: {pair_warn}")
        twoway_df, twoway_msg = fa_twoway.compute_twoway_anova(paired_df)
        print(f"  Two-way ANOVA: {twoway_msg}")
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
            stats_df.to_csv(stats_csv, index=False)
        print(f"  Saved: {png_path}")
        print(f"  Saved: {pdf_path}")
        print(f"  Saved: {csv_path}")
        if len(stats_df):
            print(f"  Saved: {stats_csv}")

        # ── Two-factor ANOVA CSV ─────────────────────────────────────────────
        if two_factor and twoway_df is not None and len(twoway_df):
            tw_csv = os.path.join(output_dir, f"{output_stem}_twoway_anova.csv")
            try:
                _write_twoway_csv(tw_csv, twoway_df, twoway_msg, drilldown,
                                  summary_df, group_order, tp_order, pair_warn)
                print(f"  Saved: {tw_csv}")
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
                                  drilldown=drilldown, pair_warn=pair_warn)
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
            print(f"  Saved: {cs_csv}")
            print(f"  Saved: {cs_pdf}")
        except Exception as exc:
            print(f"  Comparison circular-stats skipped "
                  f"({type(exc).__name__}: {exc})")

    return fig, summary_df, stats_records


def _write_pdf_report(path, fig, groups, all_summaries, labels, colors,
                      summary_df, stats_df, panels, theme, palette,
                      twoway_df=None, twoway_msg=None, drilldown=None,
                      pair_warn=None):
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


def _write_twoway_csv(path, twoway_df, twoway_msg, drilldown, summary_df,
                      group_order, tp_order, pair_warn):
    """Write the group × time-point ANOVA report: a settings header, the scalar
    mixed-ANOVA table, Holm-corrected simple effects, and the per-time-point
    curve drill-down tables."""
    import csv

    def _g(df, **eq):
        m = pd.Series(True, index=df.index)
        for k, v in eq.items():
            m &= (df[k] == v)
        return df[m]

    # per group×timepoint cell counts
    count_bits = []
    for grp in group_order:
        cells = [f"{tp}:{_g(summary_df, group=grp, timepoint=tp)['cell'].nunique()}"
                 for tp in tp_order]
        count_bits.append(f"{grp} [{', '.join(cells)}]")

    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# FIREFLY two-factor comparison — group × time point mixed ANOVA"])
        w.writerow(["# model", twoway_msg or ""])
        w.writerow(["setting", "value"])
        w.writerow(["groups", ", ".join(map(str, group_order))])
        w.writerow(["time_points", ", ".join(map(str, tp_order))])
        w.writerow(["cells_per_group_timepoint", " ; ".join(count_bits)])
        w.writerow(["pairing", pair_warn or "all cells paired across every time point"])
        w.writerow(["alpha", 0.05])
        w.writerow(["effect_size", "np2 = partial eta-squared"])
        w.writerow(["sphericity", "Greenhouse-Geisser (p_GG); for 2 time points GG=uncorrected"])
        w.writerow(["curve_note",
                    "MSD/LogD tested via scalar summary (auc_msd / median_D / "
                    "mob_immob_ratio) PLUS a per-time-point group×(lag|bin) drill-down; "
                    "pingouin cannot fit 1-between + 2-within in one model"])
        w.writerow([])

        anova = twoway_df[twoway_df["section"] == "anova"]
        if len(anova):
            w.writerow(["=== SCALAR METRICS: two-way mixed ANOVA "
                        "(between=group, within=timepoint, subject=cell) ==="])
            cols = ["metric", "effect", "F", "df1", "df2", "p_unc", "p_GG", "np2", "eps"]
            w.writerow(cols)
            for _, r in anova.iterrows():
                w.writerow([r.get(c, "") if r.get(c) is not None else "" for c in cols])
            w.writerow([])

        post = twoway_df[twoway_df["section"] == "posthoc"]
        if len(post):
            w.writerow(["=== SIMPLE EFFECTS (Holm-corrected) ==="])
            cols = ["metric", "contrast", "at", "level_A", "level_B", "paired",
                    "p", "p_holm", "stars"]
            w.writerow(cols)
            for _, r in post.iterrows():
                w.writerow([r.get(c, "") if r.get(c) is not None else "" for c in cols])
            w.writerow([])

        for kind in ("msd", "logd"):
            if kind in drilldown:
                ddf, dmsg = drilldown[kind]
                w.writerow([f"=== CURVE DRILL-DOWN ({kind.upper()}): {dmsg} ==="])
                cols = ["graph", "timepoint", "effect", "F", "df1", "df2",
                        "p_unc", "p_GG", "np2", "eps"]
                w.writerow(cols)
                for _, r in ddf.iterrows():
                    w.writerow([r.get(c, "") if r.get(c) is not None else "" for c in cols])
                w.writerow([])


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
