"""Single-sample combined figure rendering.

Extracted from sptpalm_analysis.py (#7); re-exported there for compatibility.
"""
from __future__ import annotations

import functools
import os
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from firefly.analysis.fa_theme import (_theme_palette, _THEME_REQUIRED_KEYS,
                                       style_axes)
from firefly.analysis.fa_diffusion import classify_motion, msd_linear
from firefly.analysis.fa_constants import (MOTION_CLASS_COLORS, MOTION_CLASS_ORDER,
                                           motion_class_colors)


# Canonical motion-class colours / order — shared with the comparison figure and
# the napari overlay (see fa_constants) so a class is the same colour everywhere.
# `MC` is the Dark default at module scope; `make_figure` rebinds it to the
# theme-specific palette so panels render in colours that suit the figure theme.
MC   = dict(MOTION_CLASS_COLORS)
MORD = list(MOTION_CLASS_ORDER)

# Resolution floor for the log10(D) distribution panels.  Below this, the
# fitted D of a flat-MSD (immobile) track is indistinguishable from zero — both
# FIREFLY and PALM-Tracer drive it down to ~1e-14.  The D-distribution panels
# clip to this floor and label the immobile fraction rather than rendering a
# misleading spike at an arbitrary clip value.  1e-5 µm²/s is well below what
# localisation precision can resolve at typical sptPALM frame rates.
_D_RES_FLOOR = 1e-5


def _draw_track(grp, color, ax, lw=0.8, alpha=0.6):
    """Draw one track with a tail-to-head alpha fade.

    Old implementation called ax.plot() once per segment (N-1 calls
    per track).  For 2000 tracks × 50 frames = ~100 000 plot calls,
    figure rendering became the bottleneck of the whole save phase.

    LineCollection batches all segments of a single track into one
    artist with a per-segment alpha array — same visual result,
    ~30× faster on dense track sets.
    """
    xy = grp[["x", "y"]].values
    n = len(xy)
    if n < 2:
        return
    # Build segment endpoints: shape (N-1, 2, 2) — i.e. for each
    # segment, [start_xy, end_xy].
    import numpy as _np
    from matplotlib.collections import LineCollection as _LC
    segs = _np.stack([xy[:-1], xy[1:]], axis=1)
    # Per-segment alpha ramp from 0.2 → `alpha`.
    alphas = _np.linspace(0.2, alpha, max(n - 1, 1))
    # Pre-multiply RGBA so each segment carries its own alpha through
    # the LineCollection.  Accept either a hex string or RGB tuple.
    try:
        import matplotlib.colors as _mc
        r, g, b = _mc.to_rgb(color)
    except Exception:
        r, g, b = 0.5, 0.5, 0.5
    colors = _np.column_stack(
        [_np.full(len(alphas), r),
         _np.full(len(alphas), g),
         _np.full(len(alphas), b),
         alphas])
    lc = _LC(segs, colors=colors, linewidths=lw,
             capstyle="round", antialiased=True)
    ax.add_collection(lc)


def _filled_kde(ax, x_grid, binwidth, values, color, label, *,
                bins=None, fill_alpha=0.25, lw=1.4):
    """Draw one motion class's distribution as a translucent **filled KDE**
    (Wilke / r-graph-gallery: a smooth curve reads far better than overlaid
    semi-transparent histogram bars, which turn to mud once several classes pile
    up).

    The KDE is **count-scaled** (``density × n × binwidth``) so the y-axis stays
    a count — matching the histogram idiom this replaces, keeping every existing
    annotation/limit valid.  Robust by design: a class with <2 finite values,
    zero variance, or a singular covariance (a value pile-up) can't be KDE'd, so
    it falls back to that class's histogram; empty input is a no-op.

    Returns the peak height drawn (so the caller can size the y-axis headroom),
    or 0.0 if nothing was drawn.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 0.0
    if len(v) >= 2 and np.ptp(v) > 1e-9:
        try:
            kde = gaussian_kde(v)
            # Evaluate on a grid that runs ≥3.5 KDE bandwidths past THIS class's
            # own data on both sides, so the curve always tapers fully to ~0
            # instead of being cut off mid-air at the shared grid's edge — a wide
            # class (e.g. Directed) otherwise ends abruptly on the right.  The
            # passed x_grid only sets a lower bound on the span.
            bw = float(getattr(kde, "factor", 0.2)) * (float(np.std(v)) or 1.0)
            g_lo = min(float(x_grid[0]),  float(v.min()) - 3.5 * bw)
            g_hi = max(float(x_grid[-1]), float(v.max()) + 3.5 * bw)
            grid = np.linspace(g_lo, g_hi, 320)
            y = kde(grid) * len(v) * binwidth
            ax.fill_between(grid, 0.0, y, color=color, alpha=fill_alpha,
                            linewidth=0, zorder=2)
            ax.plot(grid, y, color=color, lw=lw, label=label, zorder=3)
            return float(np.nanmax(y)) if len(y) else 0.0
        except Exception:
            pass
    # Fallback — sparse / singular: this class's histogram (still labelled).
    if bins is not None:
        n_, _, _ = ax.hist(v, bins=bins, color=color, alpha=0.55, label=label,
                           edgecolor="none", zorder=2)
        return float(np.nanmax(n_)) if len(n_) else 0.0
    return 0.0


def _close_figs_on_error(fn):
    """Ensure a renderer that raises part-way through doesn't LEAK its
    matplotlib figure.  make_figure() builds a single (20x38) pyplot figure and
    only closes it on the success path; an exception anywhere in the ~600 lines
    of panel rendering used to skip that close, and pyplot retains every leaked
    figure for the life of the process.  Across a batch (one worker process for
    N files) that is unbounded memory growth that can trip the very OOM guard
    the run was hardened against.  This closes only the figures the wrapped call
    itself created (the fignum diff), then re-raises.  (#4)
    """
    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        _before = set(plt.get_fignums())
        try:
            return fn(*args, **kwargs)
        except BaseException:
            for _n in set(plt.get_fignums()) - _before:
                try:
                    plt.close(_n)
                except Exception:
                    pass
            raise
    return _wrapper


@_close_figs_on_error
def make_figure(stack, tracks, imsd_df, emsd_df, diff_df,
                pixel_size, frame_interval, output_path=None, roi_mask=None,
                fig_theme="Dark", proj_cmap="Inferno", jdd=None,
                turning_angles=None, mobile_frac_df=None,
                cluster_labels=None, cluster_locs=None,
                cluster_subsampled_n=None,
                dwell_df=None, dwell_tau=None, return_pdf_bytes=False,
                van_hove=None, vacf=None,
                want_panels=None, traj_background=True):
    # want_panels controls the per-panel PNG export, which is expensive:
    # each panel is produced by a full-figure savefig() cropped to that
    # panel's bbox, so rendering all 15 panels means ~15 full rasterisations
    # of the whole figure.  Callers that don't need per-panel PNGs should
    # pass an empty collection to skip the loop entirely.
    #   * None            → render every panel (back-compat default)
    #   * set()/[]        → render no panels (just the combined figure)
    #   * {"A","C", ...}  → render only those panels
    print("  Rendering figure ...")

    # ── Theme palettes ─────────────────────────────────────────────────────────
    if fig_theme == "Light":
        BG, PNL   = "#ffffff", "#f6f8fa"
        TXT, GRD  = "#24292f", "#d0d7de"
        ACC       = "#0969da"
        _kde_col  = "#000000"
        _traj_bg  = "Greys"
        _font     = "sans-serif"
    elif fig_theme == "Publication":
        BG, PNL   = "#ffffff", "#ffffff"
        TXT, GRD  = "#000000", "#cccccc"
        ACC       = "#333333"
        _kde_col  = "#000000"
        _traj_bg  = "Greys"
        _font     = "DejaVu Sans"
    elif fig_theme == "AMOLED":
        # Pure-black BG variant of Dark.
        BG, PNL   = "#000000", "#0a0a0a"
        TXT, GRD  = "#e6edf3", "#30363d"
        ACC       = "#58a6ff"
        _kde_col  = "white"
        _traj_bg  = "Greys_r"
        _font     = "monospace"
    else:                                    # Dark (default)
        BG, PNL   = "#0d1117", "#161b22"
        TXT, GRD  = "#e6edf3", "#30363d"
        ACC       = "#58a6ff"
        _kde_col  = "white"
        _traj_bg  = "Greys_r"
        _font     = "monospace"

    # Theme-specific motion-class colours (shadow the module-global `MC` for the
    # whole figure body so every motion-coloured panel suits this theme — and
    # Publication uses the colour-blind-safe palette).
    MC = motion_class_colors(fig_theme)

    # ── Projection colourmap ───────────────────────────────────────────────────
    _cmap_map = {
        "Inferno": "inferno",
        "Hot":     "hot",
        "Viridis": "viridis",
        "Plasma":  "plasma",
        "Greys":   "Greys" if fig_theme in ("Light", "Publication") else "Greys_r",   # Dark + AMOLED → Greys_r
    }
    _pcmap = _cmap_map.get(proj_cmap, "inferno")

    plt.rcParams.update({
        "text.color":       TXT, "axes.labelcolor": TXT,
        "xtick.color":      TXT, "ytick.color":     TXT,
        "axes.edgecolor":   GRD, "axes.facecolor":  PNL,
        "grid.color":       GRD, "grid.alpha":      0.22,
        "font.family":      _font})

    _has_jdd = jdd is not None
    # Grid expanded from 5 to 6 rows in v1.0.64 to fit the new Radial
    # Distribution polar panel.
    fig = plt.figure(figsize=(20, 38), facecolor=BG)
    gs  = GridSpec(6, 3, figure=fig, hspace=0.45, wspace=0.32,
                   left=0.06, right=0.97, top=0.95, bottom=0.035)

    _panels          = []   # (letter, axes) collected for per-panel export
    _letter_artists  = []   # text objects for letter labels (hidden for panel renders)

    def sax(ax, ltr, ttl, kind="cartesian"):
        ax.set_facecolor(PNL)
        # Modern look: drop top/right spines + thin the rest.  Image/spatial
        # panels (kind="image") and the polar panel keep their full frame.
        style_axes(ax, {"GRD": GRD, "TXT": TXT}, kind=kind)
        ax.set_title(f"  {ttl}", loc="left", fontsize=11,
                     color=TXT, pad=8, fontweight="bold")
        txt = ax.text(-0.04, 1.06, ltr, transform=ax.transAxes, fontsize=14,
                      color=ACC, fontweight="bold", va="top", ha="right")
        _panels.append((ltr, ax))
        _letter_artists.append(txt)

    # Use up to 200 evenly-spaced frames for the max projection to save memory
    idx  = np.linspace(0, len(stack)-1, min(200, len(stack)), dtype=int)
    proj = stack[idx].max(axis=0)
    from skimage import exposure as _exp
    proj_eq = _exp.equalize_adapthist(
        (proj / proj.max()).astype(np.float32), clip_limit=0.03)
    mcol = diff_df.set_index("particle")["motion"].to_dict()

    # A — max projection
    ax = fig.add_subplot(gs[0,0])
    ax.imshow(proj_eq, cmap=_pcmap, origin="lower", aspect="equal")
    bp = 5/pixel_size; y0,x0 = proj.shape[0]*.05, proj.shape[1]*.05
    ax.plot([x0,x0+bp],[y0,y0],"-",color="white",lw=3)
    ax.text(x0+bp/2,y0+proj.shape[0]*.025,"5 um",
            ha="center",va="bottom",color="white",fontsize=8)
    ax.set_xlabel(f"X  ({pixel_size} um/px)",fontsize=9)
    ax.set_ylabel("Y (px)",fontsize=9)
    if roi_mask is not None:
        ax.contour(roi_mask.astype(float), levels=[0.5],
                   colors=["#58a6ff"], linewidths=[1.2], alpha=0.8)
        ax.text(0.02, 0.02, f"ROI", transform=ax.transAxes,
                color="#58a6ff", fontsize=8, va="bottom")
    sax(ax,"A","Max Projection", kind="image")

    # B — trajectory map coloured by motion type (subsample if very many tracks)
    ax = fig.add_subplot(gs[0,1])
    if traj_background:
        ax.imshow(proj_eq,cmap=_traj_bg,origin="lower",aspect="equal",alpha=0.35)
    all_pids  = list(tracks["particle"].unique())
    draw_pids = set(np.random.default_rng(42).choice(
        all_pids, min(2000, len(all_pids)), replace=False))
    n_drawn = 0
    for pid, grp in (tracks[tracks["particle"].isin(draw_pids)]
                     .reset_index(drop=True).sort_values("frame")
                     .groupby("particle")):
        _draw_track(grp, MC.get(mcol.get(pid,"Unknown"), MC["Unknown"]), ax)
        n_drawn += 1
    els = [Line2D([0],[0],color=MC[m],lw=2,label=m)
           for m in MORD if m in mcol.values()]
    ax.legend(handles=els,fontsize=8,loc="upper right",
              framealpha=0.7,facecolor=PNL,edgecolor=GRD,labelcolor=TXT)
    ax.set_xlim(0,proj.shape[1]); ax.set_ylim(0,proj.shape[0])
    ax.set_xlabel("X (px)",fontsize=9); ax.set_ylabel("Y (px)",fontsize=9)
    shown = f"{n_drawn:,}" + (f" of {len(all_pids):,}" if n_drawn < len(all_pids) else "")
    sax(ax,"B",f"Trajectories  (n={shown})", kind="image")

    # C — trajectories coloured by D value
    ax = fig.add_subplot(gs[0,2])
    if traj_background:
        ax.imshow(proj_eq, cmap=_traj_bg, origin="lower", aspect="equal", alpha=0.35)
    d_map = diff_df.set_index("particle")["D"].to_dict()
    d_vals_valid = [v for v in d_map.values() if v is not None and np.isfinite(v) and v > 0]
    if d_vals_valid:
        log_d_vals = np.log10(d_vals_valid)
        _p5  = np.percentile(log_d_vals, 5)
        _p95 = np.percentile(log_d_vals, 95)
        _cmap_d = plt.cm.plasma
        _norm_d = plt.Normalize(vmin=_p5, vmax=_p95)
        _sm_d   = plt.cm.ScalarMappable(cmap=_cmap_d, norm=_norm_d)
        _sm_d.set_array([])
        draw_pids_c = set(np.random.default_rng(43).choice(
            all_pids, min(2000, len(all_pids)), replace=False))
        for pid, grp in (tracks[tracks["particle"].isin(draw_pids_c)]
                         .reset_index(drop=True).sort_values("frame")
                         .groupby("particle")):
            D_val = d_map.get(pid)
            if D_val is not None and np.isfinite(D_val) and D_val > 0:
                col = _cmap_d(_norm_d(np.log10(D_val)))
            else:
                col = "#555555"
            _draw_track(grp, col, ax)
        cb = plt.colorbar(_sm_d, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("log10(D)  [µm²/s]", fontsize=8, color=TXT)
        cb.ax.yaxis.set_tick_params(color=TXT)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TXT, fontsize=7)
    ax.set_xlim(0, proj.shape[1]); ax.set_ylim(0, proj.shape[0])
    ax.set_xlabel("X (px)", fontsize=9); ax.set_ylabel("Y (px)", fontsize=9)
    sax(ax, "C", "Trajectories by D value", kind="image")

    # D — MSD curves
    ax = fig.add_subplot(gs[1,0])
    lt  = emsd_df.index.values * frame_interval
    rng = np.random.default_rng(42)
    for pid in rng.choice(list(imsd_df.columns), min(200,len(imsd_df.columns)), replace=False):
        v  = imsd_df[pid].values
        t  = imsd_df.index.values * frame_interval
        ok = np.isfinite(v) & (v > 0)
        if ok.sum() >= 2:
            ax.plot(t[ok],v[ok],"-",color="#8b949e",lw=0.4,alpha=0.3)
    ax.plot(lt,emsd_df.values,"-o",color=ACC,lw=2.5,ms=4,zorder=5,
            label="Ensemble MSD")
    try:
        t6,m6 = lt[:6], emsd_df.values[:6].ravel()
        ok6   = np.isfinite(m6) & (m6>0)
        po,_  = curve_fit(msd_linear,t6[ok6],m6[ok6],p0=[0.01,0],maxfev=2000)
        te    = np.linspace(t6[0],lt[-1],200)
        ax.plot(te,msd_linear(te,*po),"--",color="#f78166",lw=2,
                label=f"Fit D={po[0]:.4f} µm²/s")
    except Exception: pass
    ax.set_xlabel("Lag time (s)",fontsize=9)
    ax.set_ylabel("MSD (µm²)",fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True,which="both",ls="--",alpha=0.22,lw=0.5)
    ax.legend(fontsize=8,loc="upper left",framealpha=0.85,facecolor=PNL,edgecolor=GRD,labelcolor=TXT)
    sax(ax,"D","MSD Curves")

    # E — D distribution
    ax = fig.add_subplot(gs[1,1])
    dv = diff_df["D"].dropna()
    dv = dv[(dv>0) & (dv<dv.quantile(0.995))]
    if len(dv) > 5:
        # Resolution floor.  Genuinely immobile tracks have a flat MSD, so the
        # fit (here and in PALM-Tracer alike) drives D toward zero — values run
        # down to ~1e-14, far below anything localisation precision can resolve.
        # Rather than pile them into a misleading spike at an arbitrary clip
        # value with the bars/KDE disagreeing, we clip EVERYTHING consistently
        # to a single floor and label the immobile fraction honestly.  This is a
        # real immobile population (PALM-Tracer's own D export shows the same
        # ~10-12% below resolution), not an artifact.
        n_total = int(len(dv))
        n_imm   = int((dv <= _D_RES_FLOOR).sum())
        pct_imm = 100.0 * n_imm / n_total
        ld   = np.log10(dv.clip(lower=_D_RES_FLOOR))
        bins = np.linspace(ld.min(), ld.max(), 40)
        bw   = bins[1] - bins[0]
        floor_x = float(np.log10(_D_RES_FLOOR))
        peaks = []
        # Per-class filled KDE of the RESOLVED (mobile, D > floor) values only —
        # the smooth curves aren't distorted by the hard pile-up at the floor.
        mob_all = np.log10(dv[dv > _D_RES_FLOOR])
        if len(mob_all):
            _xlo, _xhi = float(mob_all.min()), float(ld.max())
        else:
            _xlo, _xhi = float(ld.min()), float(ld.max())
        # Pad the grid so each class's filled KDE tapers smoothly to ~0 at its
        # tails instead of being cut off vertically at the data min/max.
        _xpad = 0.10 * (_xhi - _xlo)
        xk = np.linspace(_xlo - _xpad, _xhi + _xpad, 300)
        for m in MORD:
            sub = diff_df[(diff_df["motion"] == m) & (diff_df["D"] > _D_RES_FLOOR)]
            if len(sub):
                peaks.append(_filled_kde(
                    ax, xk, bw, np.log10(sub["D"].clip(lower=_D_RES_FLOOR)),
                    MC[m], m, bins=bins))
        # Below-resolution (immobile) tracks are a true delta at the floor — they
        # can't be resolved, so they're drawn as a hatched bar AT the floor (not
        # smeared into a fake KDE bump), on the same count axis as the curves so
        # the immobile peak is visible alongside the mobile distribution.
        if n_imm > 0:
            ax.bar(floor_x, n_imm, width=bw * 1.8,
                   color=MC.get("Immobile", "#e05252"), edgecolor=PNL,
                   linewidth=1.0, hatch="///", alpha=0.9, zorder=3,
                   label="Immobile / below res.")
            peaks.append(float(n_imm))
        ax.axvline(np.log10(dv.median()),color=ACC,ls="--",lw=1.5,
                   label=f"Median={dv.median():.4f}")
        # Headroom above the tallest peak so nothing (a sharp KDE peak or the
        # floor bar) is clipped by the top axis.
        ymax = max(peaks) if peaks else 1.0
        ax.set_ylim(0, ymax * 1.42)
        # Label the floor bar directly with its share.
        if pct_imm >= 0.5:
            ax.text(floor_x, n_imm + ymax * 0.03, f"{pct_imm:.0f}%",
                    ha="center", va="bottom", fontsize=7, color=TXT, alpha=0.9)
        ax.set_xlabel("log10(D)  [µm²/s]",fontsize=9)
        ax.set_ylabel("Count",fontsize=9)
        ax.legend(fontsize=7.5,loc="upper right",framealpha=0.9,facecolor=PNL,edgecolor=GRD,labelcolor=TXT)
    ax.grid(True,ls="--",alpha=0.22,lw=0.5)
    sax(ax,"E","Diffusion Coefficient Distribution")

    # F — motion-class composition as a simple vertical bar chart: one bar per
    # class (height = % of classified tracks), the class names on the x-axis and
    # a fixed 0–100% y-axis so the proportions are read on an absolute scale.
    # Each bar keeps its class colour so it matches the trajectory / distribution
    # panels (Immobile=red, Confined=orange, Brownian=blue, Directed=green).
    ax = fig.add_subplot(gs[1,2])
    mc_ = diff_df["motion"].value_counts()
    classes = [m for m in MORD if m in mc_]
    counts  = np.array([float(mc_[m]) for m in classes])
    total   = float(counts.sum())
    if total <= 0:
        ax.text(0.5, 0.5, "No classified tracks", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=10, alpha=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    else:
        pct = counts / total * 100.0
        x = np.arange(len(classes))
        ax.bar(x, pct, width=0.72, color=[MC[c] for c in classes],
               edgecolor=PNL, linewidth=1.0, zorder=2)
        for xi, p in zip(x, pct):                # value label above each bar
            ax.text(xi, p + 2.0, f"{p:.0f}%", ha="center", va="bottom",
                    fontsize=8, color=TXT, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("% of classified tracks", fontsize=9)
        ax.grid(True, axis="y", ls="--", alpha=0.22, lw=0.5)
    sax(ax,"F","Motion Classification")

    # G — alpha distribution
    ax = fig.add_subplot(gs[2,0])
    av = diff_df["alpha"].dropna()
    av = av[(av>-1) & (av<4)]
    if len(av) > 5:
        ba = np.linspace(av.min(), av.max(), 40)
        bw = ba[1] - ba[0]
        _xpad = 0.10 * (float(av.max()) - float(av.min()))
        xk = np.linspace(float(av.min()) - _xpad, float(av.max()) + _xpad, 300)
        for m in MORD:
            sub = diff_df[(diff_df["motion"]==m) & diff_df["alpha"].notna()]
            if len(sub):
                _filled_kde(ax, xk, bw, sub["alpha"].clip(-1, 4),
                            MC[m], m, bins=ba)
        for xv,lb,ls in [(0.5,"a=0.5",":"),(1.0,"a=1 Brownian","--"),(2.0,"a=2 directed",":")]:
            ax.axvline(xv,color=GRD,ls=ls,lw=1.2,label=lb)
        # Honest note: immobile / jitter-dominated tracks have a flat MSD, so no
        # anomalous exponent can be fitted (alpha = NaN) — they are NOT in this
        # histogram (they're counted as Immobile in the motion-class bar F instead).
        n_tot = int(len(diff_df)); n_nan = int(diff_df["alpha"].isna().sum())
        if n_tot and (100.0 * n_nan / n_tot) >= 0.5:
            ax.text(0.02, 0.97,
                    f"α unmeasurable: {100.0*n_nan/n_tot:.0f}%\n(immobile, excluded)",
                    transform=ax.transAxes, fontsize=7, color=TXT,
                    va="top", ha="left", alpha=0.9)
        ax.set_xlabel("Anomalous exponent alpha",fontsize=9)
        ax.set_ylabel("Count",fontsize=9)
        ax.legend(fontsize=7,loc="upper right",framealpha=0.85,facecolor=PNL,edgecolor=GRD,labelcolor=TXT)
    ax.grid(True,ls="--",alpha=0.22,lw=0.5)
    sax(ax,"G","Anomalous Exponent Alpha Distribution")

    # H — Position Density Heatmap
    ax = fig.add_subplot(gs[2, 1])
    try:
        x_um = tracks["x"].values * pixel_size
        y_um = tracks["y"].values * pixel_size
        h, xe, ye = np.histogram2d(x_um, y_um, bins=120)
        from scipy.ndimage import gaussian_filter as _gf
        h_sm = _gf(h, sigma=1.5)
        ax.imshow(h_sm.T, origin="lower", cmap="hot",
                  extent=[xe[0], xe[-1], ye[0], ye[-1]],
                  aspect="equal", interpolation="bilinear")
        ax.set_xlabel("X  (µm)", fontsize=9)
        ax.set_ylabel("Y  (µm)", fontsize=9)
        if roi_mask is not None:
            H_px, W_px = roi_mask.shape
            ax.contour(
                np.linspace(0, W_px * pixel_size, W_px),
                np.linspace(0, H_px * pixel_size, H_px),
                roi_mask.astype(float), levels=[0.5],
                colors=["#58a6ff"], linewidths=[1.0], alpha=0.7)
    except Exception:
        pass
    sax(ax, "H", "Position Density Map", kind="image")

    # I — Turning Angle Distribution
    # Plotted as a single LINE following the count of each |angle| bin,
    # using UNSIGNED magnitudes (|θ|) so the x-axis runs 0°–180°.
    # 0° = continued straight; 180° = full reversal; 90° = right-angle
    # deflection; the radial-distribution panel (O) shows the rotational
    # direction (sign) separately.
    ax = fig.add_subplot(gs[2, 2])
    if turning_angles is None or len(turning_angles) < 10:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=12)
    else:
        ta_unsigned = np.abs(np.asarray(turning_angles, dtype=float))
        _ta_bins = np.linspace(0, 180, 37)            # 5° bins
        _ta_centres = 0.5 * (_ta_bins[:-1] + _ta_bins[1:])
        _ta_counts, _ = np.histogram(ta_unsigned, bins=_ta_bins)
        # Normalise to relative frequency so the shape is comparable across
        # runs (and consistent with the Compare-mode panel).  Total track
        # count is already reported in the suptitle / Summary tab.
        _ta_freq = (_ta_counts / _ta_counts.sum()
                    if _ta_counts.sum() else _ta_counts)
        ax.plot(_ta_centres, _ta_freq, "-o",
                color=ACC, lw=2, ms=3, alpha=0.95)
        # Uniform-distribution reference line (1/N_bins)
        ax.axhline(1.0 / len(_ta_centres),
                   color=GRD, lw=0.6, ls=":", label="uniform")
        # Reference verticals: 90° (right-angle), 180° (full reversal)
        ax.axvline(90,  color=GRD, lw=0.8, ls="--")
        ax.axvline(180, color=GRD, lw=0.6, ls=":")
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xlabel("|Turning angle|  (°)", fontsize=9)
        ax.set_ylabel("Relative frequency", fontsize=9)
        ax.grid(True, ls="--", alpha=0.22, lw=0.5)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.85,
                  facecolor=PNL, edgecolor=GRD, labelcolor=TXT)
    sax(ax, "I", "Turning Angle Distribution")

    # J — Mobile Fraction Over Time
    ax = fig.add_subplot(gs[3, 0])
    if mobile_frac_df is None or len(mobile_frac_df) < 2:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=12)
    else:
        ts  = mobile_frac_df["time_s"].values
        mf  = mobile_frac_df["mobile_fraction"].values * 100
        ax.plot(ts, mf, "o-", color=ACC, lw=2, ms=5)
        ax.fill_between(ts, 0, mf, alpha=0.2, color=ACC)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Mobile fraction (%)", fontsize=9)
        ax.grid(True, ls="--", alpha=0.22, lw=0.5)
    sax(ax, "J", "Mobile Fraction Over Time")

    # K — Jump Distance Distribution (spans cols 1–2)
    ax = fig.add_subplot(gs[3, 1:])
    if _has_jdd:
        _jdd_colors = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff"]

        r_max_plot = np.percentile(jdd["jumps"], 99.5)
        bins = np.linspace(0, r_max_plot, 60)
        ax.hist(jdd["jumps"], bins=bins, density=True,
                color="#8b949e", alpha=0.45, edgecolor="none",
                label=f"Observed  (n={jdd['n_jumps']:,})")

        _comp_labels = ["Slow", "Medium", "Fast"]
        for k, (pdf_k, D_k, f_k) in enumerate(
                zip(jdd["pdfs"], jdd["D_values"], jdd["fractions"])):
            lbl = (f"{_comp_labels[k]}  D={D_k:.4f} µm²/s  "
                   f"({f_k*100:.1f}%)")
            ax.plot(jdd["r_range"], pdf_k,
                    color=_jdd_colors[k], lw=2, label=lbl)

        ax.plot(jdd["r_range"], jdd["pdf_total"],
                color=TXT, lw=2.5, ls="--", label="Total fit")
        ax.set_xlabel("Jump distance  (µm)", fontsize=9)
        ax.set_ylabel("Probability density", fontsize=9)
        ax.set_xlim(0, r_max_plot)
        ax.set_ylim(bottom=0)
        ax.grid(True, ls="--", alpha=0.22, lw=0.5)
        ax.legend(fontsize=8, framealpha=0.6,
                  facecolor=PNL, edgecolor=GRD, labelcolor=TXT,
                  loc="upper right")
        _r2 = jdd.get("r_squared")
        _r2str = f"  |  R²={_r2:.3f}" if _r2 is not None and np.isfinite(_r2) else ""
        sax(ax, "K",
            f"Jump Distance Distribution  "
            f"({jdd['n_components']}-population fit  |  "
            f"{jdd['n_jumps']:,} jumps{_r2str})")
    else:
        ax.text(0.5, 0.5, "JDD not computed", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=12)
        sax(ax, "K", "Jump Distance Distribution")

    # L — Cluster Map
    ax = fig.add_subplot(gs[4, 0])
    if cluster_labels is not None and cluster_locs is not None and len(cluster_locs) > 0:
        xy_um = cluster_locs  # already in µm, subsampled to match labels
        noise = cluster_labels == -1
        if noise.any():
            ax.scatter(xy_um[noise, 0], xy_um[noise, 1],
                       s=0.5, c="#444", alpha=0.3, linewidths=0, rasterized=True)
        clustered = ~noise
        if clustered.any():
            n_c = max(cluster_labels.max() + 1, 1)
            # matplotlib.cm.get_cmap was removed in mpl 3.9 — use the colormap
            # registry (mpl >= 3.6) so this works on the bundled build too.
            cmap_c = matplotlib.colormaps["tab20"].resampled(n_c)
            ax.scatter(xy_um[clustered, 0], xy_um[clustered, 1],
                       s=1.5, c=cluster_labels[clustered], cmap=cmap_c,
                       alpha=0.7, linewidths=0, rasterized=True,
                       vmin=0, vmax=n_c - 1)
        ax.set_xlabel("X  (µm)", fontsize=9)
        ax.set_ylabel("Y  (µm)", fontsize=9)
        n_shown = int(cluster_labels.max()) + 1 if cluster_labels.max() >= 0 else 0
        ax.text(0.02, 0.98, f"n={n_shown} clusters",
                transform=ax.transAxes, fontsize=8, color=TXT, va="top")
    else:
        ax.text(0.5, 0.5, "Cluster analysis\nnot computed",
                transform=ax.transAxes, ha="center", va="center", color=TXT, fontsize=10)
    _clu_title = "Cluster Map  (DBSCAN)"
    if cluster_subsampled_n:
        _clu_title += f"  (sub-sampled to {int(cluster_subsampled_n):,})"
    sax(ax, "L", _clu_title, kind="image")

    # M — Dwell Time Distribution
    ax = fig.add_subplot(gs[4, 1])
    if dwell_df is not None and len(dwell_df) >= 5:
        dt_vals = dwell_df["dwell_time_s"].values
        ax.hist(dt_vals, bins=30, color=ACC, alpha=0.75, edgecolor="none", density=True)
        if np.isfinite(dwell_tau):
            t_fit = np.linspace(0, dt_vals.max(), 200)
            ax.plot(t_fit, (1/dwell_tau) * np.exp(-t_fit / dwell_tau),
                    "--", color="#f78166", lw=2,
                    label=f"τ = {dwell_tau:.2f} s")
            ax.legend(fontsize=8, loc="upper right", framealpha=0.85,
                      facecolor=PNL, edgecolor=GRD, labelcolor=TXT)
        ax.set_xlabel("Dwell time  (s)", fontsize=9)
        ax.set_ylabel("Probability density", fontsize=9)
        ax.grid(True, ls="--", alpha=0.22, lw=0.5)
    else:
        ax.text(0.5, 0.5, "Insufficient data\n(need confined/immobile tracks)",
                transform=ax.transAxes, ha="center", va="center", color=TXT, fontsize=10)
    sax(ax, "M", "Dwell Time Distribution")

    # N — MSS Slope Distribution
    ax = fig.add_subplot(gs[4, 2])
    if "mss_slope" in diff_df.columns and diff_df["mss_slope"].notna().sum() >= 5:
        ms = diff_df["mss_slope"].dropna()
        ms = ms[ms.between(-0.5, 1.5)]
        bins = np.linspace(ms.min(), ms.max(), 40)
        bw = bins[1] - bins[0]
        _xpad = 0.10 * (float(ms.max()) - float(ms.min()))
        xk = np.linspace(float(ms.min()) - _xpad, float(ms.max()) + _xpad, 300)
        for m in MORD:
            sub = diff_df[(diff_df["motion"] == m) & diff_df["mss_slope"].notna()]
            sub = sub[sub["mss_slope"].between(-0.5, 1.5)]
            if len(sub):
                _filled_kde(ax, xk, bw, sub["mss_slope"], MC[m], m, bins=bins)
        for xv, lb, ls_ in [(0.25, "Confined", ":"), (0.5, "Brownian", "--"), (0.75, "Directed", ":")]:
            ax.axvline(xv, color=GRD, ls=ls_, lw=1.2, label=lb)
        ax.set_xlabel("MSS slope  (ν)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.85, facecolor=PNL, edgecolor=GRD, labelcolor=TXT)
        ax.grid(True, ls="--", alpha=0.22, lw=0.5)
    else:
        ax.text(0.5, 0.5, "MSS not computed\n(tracks too short)",
                transform=ax.transAxes, ha="center", va="center", color=TXT, fontsize=10)
    sax(ax, "N", "Moment Scaling Spectrum  (MSS slope)")

    # O — Radial Distribution of turning angles (polar)
    # A polar histogram of signed turning angles, oriented so 0° (straight
    # ahead) is at the top and positive angles sweep CLOCKWISE around to the
    # right (i.e. right hemisphere = positive turns, left hemisphere =
    # negative turns).  The bars radiate outward; their angular position is
    # the turning direction, their height the relative frequency.  Uniform
    # circle = Brownian motion; lobe at 0° = directional persistence; lobe
    # at ±180° = back-tracking / confinement.
    # Placed at the centre column of row 5 so it sits visually balanced
    # rather than pinned to a corner.
    ax = fig.add_subplot(gs[5, 1], projection="polar")
    if turning_angles is None or len(turning_angles) < 10:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ta_arr = np.asarray(turning_angles, dtype=float)
        is_signed = bool(np.any(ta_arr < -1e-3))
        print(f"  Radial-dist input: n={len(ta_arr):,}  "
              f"signed={is_signed}  "
              f"pos={int((ta_arr>0).sum()):,}  neg={int((ta_arr<0).sum()):,}  "
              f"min={ta_arr.min():.1f}°  max={ta_arr.max():.1f}°")
        if not is_signed:
            ta_arr = np.concatenate([ta_arr, -ta_arr])
        # CRITICAL: matplotlib polar's ax.bar() does NOT render correctly
        # when theta values are in (-π, +π].  Half the bars (the side with
        # negative theta after applying set_theta_direction) silently fail
        # to draw, producing only a half-circle of bars.
        # Empirical fix: shift the angles to [0, 2π) before histogramming.
        # The xticks are then placed at positive-only angles too, but
        # *labelled* with the signed values the user expects.
        angles_rad = np.mod(np.deg2rad(ta_arr), 2 * np.pi)
        n_bins = 36
        bins   = np.linspace(0, 2 * np.pi, n_bins + 1)
        counts, edges = np.histogram(angles_rad, bins=bins, density=True)
        theta = 0.5 * (edges[:-1] + edges[1:])
        width = bins[1] - bins[0]
        ax.bar(theta, counts, width=width * 0.95, bottom=0.0,
               color=ACC, alpha=0.75, edgecolor=GRD, linewidth=0.5)
        ax.set_theta_zero_location("N")     # 0° at the top
        ax.set_theta_direction(-1)          # clockwise positive (right = +)
        # xticks at 0°, 45°, ..., 315° (positive only); labels show signed
        # equivalents so the reader still sees "-45°" on the left, etc.
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(["0°", "+45°", "+90°", "+135°", "±180°",
                            "−135°", "−90°", "−45°"], fontsize=8)
        # Hide the radial-axis numeric labels.
        ax.set_yticklabels([])
        ax.tick_params(axis="y", which="both", left=False)
        ax.grid(True, ls="--", alpha=0.22, lw=0.5)
    sax(ax, "O", "Radial Distribution  (signed turning angles)")

    # P — van Hove displacement distribution (left slot of row 5)
    # The pooled single-frame step distribution with a same-σ Gaussian
    # reference overlaid.  Heavy (non-Gaussian) tails => a heterogeneous
    # population; the non-Gaussian parameter α₂ quantifies the deviation
    # (α₂ ≈ 0 for Brownian, > 0 for mixed mobile/trapped).
    ax = fig.add_subplot(gs[5, 0])
    if van_hove is None or van_hove.get("pdf") is None:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    else:
        c   = np.asarray(van_hove["bin_centers_um"], float)
        pdf = np.asarray(van_hove["pdf"], float)
        g   = np.asarray(van_hove["gaussian_pdf"], float)
        a2  = van_hove.get("non_gaussian_alpha2", float("nan"))
        ax.fill_between(c, pdf, step="mid", color=ACC, alpha=0.35)
        ax.plot(c, pdf, drawstyle="steps-mid", color=ACC, lw=1.2,
                label="van Hove")
        ax.plot(c, g, color=_kde_col, lw=1.3, ls="--", label="Gaussian")
        ax.set_yscale("log")
        _pos = pdf[pdf > 0]
        if _pos.size:
            ax.set_ylim(_pos.min() * 0.5, pdf.max() * 2.0)
        ax.set_xlabel("Δx, Δy  (µm)"); ax.set_ylabel("P(Δ)  (log)")
        ax.legend(fontsize=8, framealpha=0.3, loc="upper right")
        ax.text(0.03, 0.95, f"α₂ = {a2:.3f}", transform=ax.transAxes,
                ha="left", va="top", color=TXT, fontsize=9,
                bbox=dict(boxstyle="round", fc=PNL, ec=GRD, alpha=0.7))
        ax.grid(True, ls="--", alpha=0.22, lw=0.5)
    sax(ax, "P", "van Hove  (single-frame displacements)")

    # Q — velocity autocorrelation function (right slot of row 5)
    # Normalised ensemble VACF vs lag.  Flat-at-zero => Brownian (no
    # directional memory); positive decay => persistent/directed motion;
    # a negative lag-1 dip => caged / anti-persistent motion.
    ax = fig.add_subplot(gs[5, 2])
    if vacf is None or vacf.get("vacf") is None:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    else:
        lags = np.asarray(vacf["lags_frames"], float)
        cv   = np.asarray(vacf["vacf"], float)
        pers = vacf.get("persistence", float("nan"))
        ax.axhline(0.0, color=GRD, lw=1.0, ls=":")
        ax.plot(lags, cv, marker="o", ms=4, color=ACC, lw=1.3)
        ax.set_xlabel("lag  (frames)"); ax.set_ylabel("VACF  (normalised)")
        ax.set_xlim(left=0)
        ax.text(0.97, 0.95, f"persistence = {pers:.3f}",
                transform=ax.transAxes, ha="right", va="top", color=TXT,
                fontsize=9,
                bbox=dict(boxstyle="round", fc=PNL, ec=GRD, alpha=0.7))
        ax.grid(True, ls="--", alpha=0.22, lw=0.5)
    sax(ax, "Q", "Velocity Autocorrelation")

    md = diff_df["D"].dropna().median()
    ma = diff_df["alpha"].dropna().median()
    fig.suptitle(
        f"FIREFLY Analysis  |  {diff_df.shape[0]:,} trajectories  |  "
        f"Median D = {md:.4f} µm²/s  |  Median alpha = {ma:.2f}",
        fontsize=13,color=TXT,y=0.97,fontweight="bold")

    import io as _io
    from matplotlib.transforms import Bbox as _Bbox

    from PIL import Image as _PILImage

    # Render individual panels WITHOUT letter labels.  Each panel costs a
    # full-figure savefig(), so only do this for the panels actually
    # requested — and skip the whole block (and its two extra draws) when
    # none are wanted.
    panel_images = {}
    _render_panels = (want_panels is None) or bool(want_panels)
    if _render_panels:
        for _txt in _letter_artists:
            _txt.set_visible(False)
        fig.canvas.draw()
        _renderer = fig.canvas.get_renderer()
        _pad_px   = fig.dpi * 0.12
        for _ltr, _pax in _panels:
            if want_panels is not None and _ltr not in want_panels:
                continue
            _bbox = _pax.get_tightbbox(_renderer)
            if _bbox is None:
                continue
            _bbox_pad = _Bbox([[_bbox.x0 - _pad_px, _bbox.y0 - _pad_px],
                                [_bbox.x1 + _pad_px, _bbox.y1 + _pad_px]])
            _bbox_in  = _bbox_pad.transformed(fig.dpi_scale_trans.inverted())
            _pbuf = _io.BytesIO()
            fig.savefig(_pbuf, format="png", dpi=150, bbox_inches=_bbox_in,
                        facecolor=fig.get_facecolor())
            _pbuf.seek(0)
            panel_images[_ltr] = _PILImage.open(_pbuf).copy()
            _pbuf.close()

        # Restore letter labels for the combined figure
        for _txt in _letter_artists:
            _txt.set_visible(True)
        fig.canvas.draw()
    _buf = _io.BytesIO()
    fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    _buf.seek(0)
    combined_pil = _PILImage.open(_buf).copy()
    _buf.close()

    # Save to disk only if output_path explicitly provided (CLI / legacy callers)
    if output_path:
        combined_pil.save(output_path, dpi=(150, 150))
        print(f"  Figure -> {output_path}")
        _pdf = os.path.splitext(output_path)[0] + ".pdf"
        fig.savefig(_pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Figure (PDF) -> {_pdf}")

    pdf_bytes = None
    if return_pdf_bytes:
        try:
            _pdfbuf = _io.BytesIO()
            fig.savefig(_pdfbuf, format="pdf", bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            pdf_bytes = _pdfbuf.getvalue()
            _pdfbuf.close()
        except Exception as _exc:
            print(f"  WARN: PDF render failed: {_exc}")

    plt.close(fig)
    print("  Figure rendered.")
    return {
        "combined":     combined_pil,
        "panels":       panel_images,
        "panel_titles": {ltr: ax.get_title().strip() for ltr, ax in _panels},
        "pdf_bytes":    pdf_bytes,
    }
