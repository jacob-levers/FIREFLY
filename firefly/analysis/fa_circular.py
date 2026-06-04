"""Circular (angular) statistics and their PDF reports.

Extracted from sptpalm_analysis.py (#7); re-exported there for compatibility.
"""
from __future__ import annotations

import pandas as pd

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as _stats
from firefly.analysis.fa_theme import _theme_palette, _THEME_REQUIRED_KEYS


def compute_circular_statistics(angles_deg):
    """Full circular-statistics summary for an array of angles, in
    degrees, on the interval (-180°, +180°] (signed turning-angle
    convention used by `compute_turning_angles`).

    Returns a dict whose keys are the same statistic names MATLAB's
    CircStat toolbox (Berens 2009) uses, so a supervisor familiar with
    that toolbox can map results 1:1.  All angles in the output are in
    DEGREES; rates / dispersions in their natural units.

    Computed statistics
    -------------------
    n                            : sample size
    mean_direction_deg           : μ = atan2(S, C), in (-180°, +180°]
    mean_resultant_length        : R̄ in [0, 1]    (1 = perfect alignment)
    circular_variance            : 1 - R̄          ("S" in Fisher 1993)
    circular_std_deg             : √(-2·ln R̄)·(180/π)
    angular_deviation_deg        : √(2·(1 - R̄))·(180/π)  ("s₀" in Fisher)
    median_deg                   : circular median
    concentration_kappa          : von Mises κ via standard piecewise
                                   approximation (Best & Fisher 1981)
    rayleigh_z                   : n·R̄²   (test statistic for uniformity)
    rayleigh_p                   : Wilkie-Mardia approximation (good
                                   to ~5e-4 for n ≥ 10)
    v_test_z, v_test_p           : V-test against μ₀ = 0° (tests for a
                                   preferred mean direction at "straight
                                   ahead")
    circular_skewness            : b̄ / (1 - R̄)^1.5
    circular_kurtosis            : (ā - R̄⁴) / (1 - R̄)²
    ci95_lower_deg, ci95_upper_deg
                                 : approximate 95% CI for μ (Fisher 1993
                                   §4.4.4, large-sample normal approx)

    References
    ----------
    Mardia & Jupp 2000, "Directional Statistics".
    Fisher 1993, "Statistical Analysis of Circular Data".
    Berens 2009, "CircStat: A MATLAB Toolbox for Circular Statistics",
    J. Stat. Soft. 31(10).
    """
    a = np.asarray(angles_deg, dtype=float).ravel()
    a = a[np.isfinite(a)]
    n = int(a.size)
    out = {"n": n}
    if n < 2:
        # Nothing meaningful with <2 points.  Fill the schema with NaN
        # so downstream CSV consumers see the same columns regardless.
        for k in ("mean_direction_deg", "mean_resultant_length",
                 "circular_variance", "circular_std_deg",
                 "angular_deviation_deg", "median_deg",
                 "concentration_kappa", "rayleigh_z", "rayleigh_p",
                 "v_test_z", "v_test_p", "circular_skewness",
                 "circular_kurtosis", "ci95_lower_deg", "ci95_upper_deg"):
            out[k] = float("nan")
        return out

    rad = np.radians(a)
    C = float(np.mean(np.cos(rad)))
    S = float(np.mean(np.sin(rad)))
    R_bar = float(np.hypot(C, S))           # mean resultant length
    mu_rad = float(np.arctan2(S, C))         # mean direction (radians)
    mu_deg = float(np.degrees(mu_rad))
    # Standard CircStat convention: report direction on (-180°, +180°]
    if mu_deg <= -180.0: mu_deg += 360.0
    if mu_deg >   180.0: mu_deg -= 360.0

    # Dispersion measures
    circ_var = 1.0 - R_bar
    # √(-2·ln R̄) is undefined at R̄=0 (uniform), gigantic for tiny R̄.
    # Clamp to avoid log(0) screaming; report NaN for R̄ ≤ 0 instead.
    if R_bar > 0:
        circ_std_deg = float(np.degrees(np.sqrt(-2.0 * np.log(R_bar))))
    else:
        circ_std_deg = float("nan")
    ang_dev_deg = float(np.degrees(np.sqrt(2.0 * max(circ_var, 0.0))))

    # Circular median: angle θ̃ minimising Σ (π − |π − |θᵢ − θ̃||).
    # Evaluating the objective at every datum is O(n²) in time AND
    # memory if we do it with broadcasting (the 50k × 50k float64
    # array alone is 20 GB).  We instead:
    #   * cap CANDIDATES at 3000 (random subsample of the data)
    #   * cap SUMMAND points at 8000 (random subsample of the data)
    # which gives 24 million ops + ~190 MB temporary — fast enough,
    # and the median estimate from a 8000-point subsample is accurate
    # to a couple of degrees, well below other sources of noise here.
    _rng = np.random.default_rng(0)
    if n > 3000:
        cand = rad[_rng.choice(n, size=3000, replace=False)]
    else:
        cand = rad
    if n > 8000:
        ref = rad[_rng.choice(n, size=8000, replace=False)]
    else:
        ref = rad
    diff = np.abs(cand[:, None] - ref[None, :])
    diff = np.minimum(diff, 2.0 * np.pi - diff)        # circular distance
    obj = diff.sum(axis=1)
    median_rad = float(cand[int(np.argmin(obj))])
    median_deg = float(np.degrees(median_rad))
    if median_deg <= -180.0: median_deg += 360.0
    if median_deg >   180.0: median_deg -= 360.0

    # Concentration κ — Best & Fisher 1981 piecewise approximation,
    # with a small-n bias correction (Fisher 1993 eq. 4.41).
    if R_bar < 0.53:
        kappa = 2.0 * R_bar + R_bar ** 3 + 5.0 * R_bar ** 5 / 6.0
    elif R_bar < 0.85:
        kappa = -0.4 + 1.39 * R_bar + 0.43 / max(1.0 - R_bar, 1e-12)
    else:
        denom = max(R_bar ** 3 - 4.0 * R_bar ** 2 + 3.0 * R_bar, 1e-12)
        kappa = 1.0 / denom
    if n < 15:
        if kappa < 2.0:
            kappa = max(kappa - 2.0 / (n * kappa), 0.0)
        else:
            kappa = ((n - 1.0) ** 3) * kappa / (n ** 3 + n)

    # Rayleigh test for uniformity (Wilkie 1983 / Mardia & Jupp eq. 6.3.5).
    # We compute in LOG space so the result doesn't underflow to 0
    # when n is large (e.g. n=240k with R̄=0.08 → z≈1500 → exp(-z)
    # rounds to 0 in float64, which the user sees as a spurious
    # "p = 0").  The leading term is exp(-z); we still apply the
    # Mardia higher-order correction multiplicatively in log-space.
    R_total = n * R_bar
    z_ray = R_total ** 2 / n
    correction = (1.0 + (2.0 * z_ray - z_ray ** 2) / (4.0 * n)
                  - (24.0 * z_ray - 132.0 * z_ray ** 2
                     + 76.0 * z_ray ** 3 - 9.0 * z_ray ** 4)
                    / (288.0 * n ** 2))
    if correction <= 0:
        correction = 1.0   # higher-order correction overshot; ignore.
    log_p_ray = -z_ray + np.log(correction)
    # If log p < ~-700, exp underflows.  Convert to a tiny positive
    # number that survives float64 (1e-300) so downstream callers see
    # "very small" rather than zero, and formatters can render it as
    # "<1e-300".
    if log_p_ray < -700.0:
        p_ray = 1e-300
    else:
        p_ray = float(np.exp(log_p_ray))
    p_ray = float(np.clip(p_ray, 0.0, 1.0))

    # V-test against μ₀ = 0° ("are tracks preferentially going
    # straight ahead?").  V = R̄·cos(μ − μ₀); z = V·√(2n); one-tailed.
    mu0 = 0.0
    V = R_bar * np.cos(mu_rad - mu0)
    z_v = V * np.sqrt(2.0 * n)
    # One-tailed p via the standard normal survival function.  Use
    # scipy's norm.sf where available (numerically stable to ~p≈1e-300);
    # fall back to a math.erf-based computation otherwise, and floor at
    # 1e-300 so a huge z doesn't round to exactly 0.
    try:
        from scipy.stats import norm as _norm
        p_v = float(_norm.sf(z_v))
    except Exception:
        from math import erf
        p_v = float(0.5 * (1.0 - erf(z_v / np.sqrt(2.0))))
    if p_v == 0.0:
        p_v = 1e-300    # underflow sentinel
    p_v = float(np.clip(p_v, 0.0, 1.0))

    # Circular skewness and kurtosis (Mardia & Jupp §2.3).
    # b̄ = (1/n) Σ sin(2(θᵢ − μ))   ;   ā = (1/n) Σ cos(2(θᵢ − μ))
    b_bar = float(np.mean(np.sin(2.0 * (rad - mu_rad))))
    a_bar = float(np.mean(np.cos(2.0 * (rad - mu_rad))))
    sigma = max(1.0 - R_bar, 1e-12)
    skew = b_bar / (sigma ** 1.5)
    kurt = (a_bar - R_bar ** 4) / (sigma ** 2)

    # 95% CI for μ — large-sample normal approximation (Fisher 1993
    # eq. 4.46).  Only meaningful when R̄ is appreciable AND n ≥ ~15;
    # report NaN when the approximation breaks down.
    if R_bar >= 0.4 and n >= 15:
        sd_mu = np.sqrt((1.0 - a_bar) / (2.0 * n * R_bar ** 2))
        half = float(np.degrees(1.959964 * sd_mu))   # 1.96 σ
        lo = mu_deg - half
        hi = mu_deg + half
        # Keep both endpoints on (-180°, +180°] without wrapping the
        # interval ordering — supervisor will read this from the CSV.
        ci_lo, ci_hi = lo, hi
    else:
        ci_lo = float("nan")
        ci_hi = float("nan")

    out.update({
        "mean_direction_deg":     mu_deg,
        "mean_resultant_length":  R_bar,
        "circular_variance":      circ_var,
        "circular_std_deg":       circ_std_deg,
        "angular_deviation_deg":  ang_dev_deg,
        "median_deg":             median_deg,
        "concentration_kappa":    float(kappa),
        "rayleigh_z":             float(z_ray),
        "rayleigh_p":             p_ray,
        "v_test_z":               float(z_v),
        "v_test_p":               p_v,
        "circular_skewness":      float(skew),
        "circular_kurtosis":      float(kurt),
        "ci95_lower_deg":         float(ci_lo),
        "ci95_upper_deg":         float(ci_hi),
    })
    return out


def save_circular_statistics_pdf(angles_deg, stats, *, pdf_path,
                                  file_label="", fig_theme="Dark",
                                  circ_lin_result=None):
    """Render a single-page A4-portrait PDF report summarising the
    circular statistics in `stats` (as produced by
    `compute_circular_statistics`) alongside a small polar histogram of
    the underlying angle distribution.

    Designed to be supervisor-facing: stat names match MATLAB CircStat,
    each value is annotated with a one-line plain-English meaning, and
    the polar plot orients 0° at the top with positive angles sweeping
    counter-clockwise (the convention `compute_turning_angles` uses).

    Parameters
    ----------
    angles_deg : 1-D array of turning angles in degrees (signed, on
                 (-180°, +180°]).  Used only for the polar histogram.
    stats      : dict returned by `compute_circular_statistics`.
    pdf_path   : where to write the PDF.
    file_label : appears in the page header (typically the analysis stem).
    fig_theme  : "Dark" | "Light" | "Publication" — palette to match the
                 master figure renderer.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pal = _theme_palette(fig_theme)

    a = np.asarray(angles_deg, dtype=float).ravel()
    a = a[np.isfinite(a)]

    # Helper: render NaN as an em-dash so the PDF doesn't look broken
    # when a stat couldn't be computed (small-n or R̄ ≈ 0 cases).
    # Also collapse the 1e-300 underflow sentinel produced by the
    # log-space p-value computations into a human-readable "<1e-300"
    # — otherwise the supervisor sees "1e-300" and wonders why so
    # many tests give exactly that value.
    def _fmt(x, prec=4):
        try:
            if x is None: return "—"
            xf = float(x)
            if np.isnan(xf): return "—"
            if xf > 0.0 and xf <= 1e-300:
                return "<1e-300"
            return f"{xf:.{prec}g}"
        except Exception:
            return str(x)

    # One-line plain-English gloss per statistic.  Order matches the
    # CSV column order so the table reads top-to-bottom like the CSV.
    rows = [
        ("n",                          "Sample size", "count",
         f"{int(stats.get('n', 0)):,}"),
        ("mean_direction_deg",         "Mean direction μ", "deg",
         _fmt(stats.get("mean_direction_deg"), 4)),
        ("mean_resultant_length",      "Mean resultant length R̄  (0 = uniform, 1 = aligned)",
         "—",
         _fmt(stats.get("mean_resultant_length"), 4)),
        ("circular_variance",          "Circular variance  1 − R̄", "—",
         _fmt(stats.get("circular_variance"), 4)),
        ("circular_std_deg",           "Circular standard deviation  √(−2·ln R̄)",
         "deg",
         _fmt(stats.get("circular_std_deg"), 4)),
        ("angular_deviation_deg",      "Angular deviation  s₀ = √(2·(1−R̄))",
         "deg",
         _fmt(stats.get("angular_deviation_deg"), 4)),
        ("median_deg",                 "Circular median", "deg",
         _fmt(stats.get("median_deg"), 4)),
        ("concentration_kappa",        "Von Mises concentration κ  (Best & Fisher 1981)",
         "—",
         _fmt(stats.get("concentration_kappa"), 4)),
        ("rayleigh_z",                 "Rayleigh test statistic  z = n·R̄²", "—",
         _fmt(stats.get("rayleigh_z"), 4)),
        ("rayleigh_p",                 "Rayleigh test p-value  (uniformity)", "—",
         _fmt(stats.get("rayleigh_p"), 3)),
        ("v_test_z",                   "V-test statistic against μ₀ = 0°", "—",
         _fmt(stats.get("v_test_z"), 4)),
        ("v_test_p",                   "V-test p-value  (preferred direction)", "—",
         _fmt(stats.get("v_test_p"), 3)),
        ("circular_skewness",          "Circular skewness  (Mardia & Jupp §2.3)", "—",
         _fmt(stats.get("circular_skewness"), 4)),
        ("circular_kurtosis",          "Circular kurtosis  (Mardia & Jupp §2.3)", "—",
         _fmt(stats.get("circular_kurtosis"), 4)),
        ("ci95_lower_deg",             "95% CI lower bound for μ  (large-sample)", "deg",
         _fmt(stats.get("ci95_lower_deg"), 4)),
        ("ci95_upper_deg",             "95% CI upper bound for μ  (large-sample)", "deg",
         _fmt(stats.get("ci95_upper_deg"), 4)),
    ]
    # Circ-lin correlation rows — optional; only present when the
    # caller passed a `circ_lin_result` (computed from per-track
    # (mean_angle, D) pairs).  Three rows: r, χ²(2), p, n.  Treated
    # as a single stats block so it can be excluded silently when
    # the caller has no D data (e.g. external-CSV input path).
    if circ_lin_result:
        rows.extend([
            ("circ_lin_angle_vs_D_r",
             "Circ-lin correlation r — turning bias vs D", "—",
             _fmt(circ_lin_result.get("r"), 4)),
            ("circ_lin_angle_vs_D_chi2",
             "Circ-lin χ²(2) test statistic  (n·r²)", "—",
             _fmt(circ_lin_result.get("test_stat"), 4)),
            ("circ_lin_angle_vs_D_p",
             "Circ-lin correlation p-value", "—",
             _fmt(circ_lin_result.get("p"), 3)),
            ("circ_lin_angle_vs_D_n",
             "Circ-lin sample size  (tracks with ≥ 3 frames + D)",
             "count",
             f"{int(circ_lin_result.get('n', 0)):,}"
             if circ_lin_result.get("n") is not None else "—"),
        ])

    # ── rcParams snapshot ──────────────────────────────────────────────
    # plt.rcParams persists across figures in the same process — the
    # master figure renderer might have left things on the Dark palette
    # (text.color = #e6edf3 etc.).  Snapshot then force everything to
    # OUR palette so we can't accidentally pick up someone else's
    # colours.  Restored at the end.
    _rc_keys = ("text.color", "axes.labelcolor", "axes.edgecolor",
                "xtick.color", "ytick.color", "axes.facecolor",
                "axes.titlecolor", "figure.facecolor", "grid.color",
                "font.family")
    _rc_save = {k: plt.rcParams.get(k) for k in _rc_keys}
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
        # ── Layout (A4 portrait, all coords in figure-fraction) ─────────
        #
        # Vertical bands, top → bottom:
        #   y 0.94 – 0.98  : header bar (title + n)
        #   y 0.89 – 0.93  : file label
        #   y 0.61 – 0.86  : polar  |  interpretation banner
        #   y 0.54 – 0.58  : "Statistics" section title
        #   y 0.12 – 0.52  : statistics table
        #   y 0.06 – 0.10  : sign-convention footer (3 short lines)
        #   y 0.02 – 0.04  : references footer
        #
        # The earlier layout placed the Statistics title with
        # `transform=ax_tbl.transAxes` at y=1.04 which sits at about
        # figure-y 0.53 — directly underneath the polar's "±180°" tick.
        # Moving it to its own fig.text at a fixed y resolves the overlap.
        # The footer used to be at y=0.04 which collided with the
        # table's bottom row at y=0.05; both footers now live below
        # y=0.10 with the table topping at y=0.52.
        fig = plt.figure(figsize=(8.27, 11.69), facecolor=pal["BG"])

        # Header (full width)
        ax_hdr = fig.add_axes([0.07, 0.94, 0.86, 0.04])
        ax_hdr.axis("off")
        title = "Circular Statistics Report"
        ax_hdr.text(0.0, 0.5, title, fontsize=16, fontweight="bold",
                    va="center", ha="left", color=pal["TXT"])
        n_val = int(stats.get("n", 0))
        ax_hdr.text(1.0, 0.5,
                    f"n = {n_val:,} turning angles",
                    fontsize=11, color=pal["MUT"], va="center", ha="right")
        if file_label:
            # File label on its own dedicated row so it can't fight the
            # polar plot's "0°" tick label below.
            fig.text(0.07, 0.91, file_label, fontsize=10,
                     color=pal["MUT"], va="top", ha="left",
                     family=pal["FONT"])

        # Polar histogram (left side of middle band).
        # Convention matched to the master figure's Radial-Distribution
        # panel (see sax "O" in make_figure): 0° at the top, positive
        # angles sweep CLOCKWISE so they appear on the right hemisphere.
        # Signed angles on (-180°, +180°] are first wrapped to [0, 2π)
        # before histogramming — matplotlib's polar bar() silently drops
        # bars at negative theta when set_theta_direction(-1) is active.
        ax_polar = fig.add_axes([0.08, 0.61, 0.36, 0.25], projection="polar")
        ax_polar.set_facecolor(pal["PNL"])
        if a.size >= 10:
            nbins = 36
            angles_rad = np.mod(np.deg2rad(a), 2.0 * np.pi)
            bins  = np.linspace(0.0, 2.0 * np.pi, nbins + 1)
            counts, edges = np.histogram(angles_rad, bins=bins)
            widths  = np.diff(edges)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax_polar.set_theta_zero_location("N")
            ax_polar.set_theta_direction(-1)  # CW positive — match master fig
            ax_polar.bar(centers, counts, width=widths * 0.95,
                         align="center", color=pal["ACC"],
                         edgecolor=pal["PNL"], linewidth=0.4, alpha=0.92)
            mu = stats.get("mean_direction_deg")
            if mu is not None and not (isinstance(mu, float) and np.isnan(mu)):
                r_max = float(counts.max()) if counts.size else 1.0
                # Wrap signed μ into [0, 2π) so the arrow lands at the
                # same place the bar histogram does.
                mu_rad = np.mod(np.deg2rad(mu), 2.0 * np.pi)
                ax_polar.annotate("",
                    xy=(mu_rad, r_max * 0.95),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->",
                                    color=pal["ARROW"], lw=2.0))
            # Show signed-angle labels at positive-angle slot positions
            # so the visual reads "+45° upper-right, -45° upper-left",
            # exactly like the master figure.
            ax_polar.set_xticks(np.deg2rad(
                [0, 45, 90, 135, 180, 225, 270, 315]))
            ax_polar.set_xticklabels(
                ["0°", "+45°", "+90°", "+135°", "±180°",
                 "−135°", "−90°", "−45°"], fontsize=8)
            ax_polar.set_yticklabels([])
            ax_polar.tick_params(colors=pal["TXT"], labelsize=8)
            ax_polar.grid(True, ls=":", alpha=0.4)
            # NB: deliberately no `set_title` here — matplotlib places
            # the polar title above the axes box (offset by `pad`), and
            # at this layout that overlaps the file-label rendered in
            # the header area.  The page header already identifies the
            # report, and the footer covers the sign convention, so a
            # title on the polar would be redundant anyway.
        else:
            ax_polar.axis("off")
            ax_polar.text(0.5, 0.5, "Too few angles for histogram",
                          transform=ax_polar.transAxes,
                          ha="center", va="center", color=pal["MUT"],
                          fontsize=10)

        # Interpretation banner (right side of middle band)
        ax_intr = fig.add_axes([0.48, 0.61, 0.46, 0.25])
        ax_intr.axis("off")
        R = stats.get("mean_resultant_length")
        p = stats.get("rayleigh_p")
        if R is None or (isinstance(R, float) and np.isnan(R)):
            interp = "Distribution: insufficient data."
        elif R < 0.10:
            interp = ("Distribution is consistent with uniform circular "
                      "scatter — no preferred turning direction is "
                      "evident.  Typical of free 2-D diffusion.")
        elif R < 0.30:
            interp = ("Weak directional bias.  Most steps are close "
                      "to uniform, but a slight tendency toward "
                      f"{stats.get('mean_direction_deg', 0):.0f}° is "
                      "present.")
        elif R < 0.60:
            interp = ("Moderate directional bias toward "
                      f"{stats.get('mean_direction_deg', 0):.0f}°.  "
                      "Consider whether this reflects biology (e.g. "
                      "transport along a cytoskeletal track) or an "
                      "artefact (uncorrected drift, anisotropic ROI).")
        else:
            interp = ("Strong directional bias toward "
                      f"{stats.get('mean_direction_deg', 0):.0f}°.  "
                      "Verify the drift correction and ROI geometry "
                      "before biological interpretation.")
        if p is not None and not (isinstance(p, float) and np.isnan(p)):
            if p < 0.001:
                verdict = ("Rayleigh test strongly rejects uniformity "
                           f"(p = {p:.3g}).")
            elif p < 0.05:
                verdict = ("Rayleigh test rejects uniformity at α = "
                           f"0.05 (p = {p:.3g}).")
            else:
                verdict = ("Rayleigh test does NOT reject uniformity "
                           f"(p = {p:.3g}).")
            interp = interp + "\n\n" + verdict
        ax_intr.text(0.0, 1.0, "Interpretation",
                     fontsize=12, fontweight="bold", va="top",
                     color=pal["TXT"])
        ax_intr.text(0.0, 0.9, interp, fontsize=10, va="top",
                     wrap=True, color=pal["TXT"])

        # Section title — placed in FIGURE coords so its vertical
        # position is decoupled from the table's bbox and can't
        # collide with the polar's bottom ticks above.
        fig.text(0.07, 0.555, "Statistics  (MATLAB CircStat conventions)",
                 fontsize=12, fontweight="bold", va="bottom",
                 ha="left", color=pal["TXT"])
        # Statistics table — pinned with a clear gap above (title) and
        # below (footer block).  Bottom edge y=0.12 leaves room for two
        # footer lines without collision.
        ax_tbl = fig.add_axes([0.07, 0.12, 0.88, 0.40])
        ax_tbl.axis("off")

        cell_text, row_labels = [], []
        for key, gloss, unit, val in rows:
            unit_s = "" if unit in ("", "—") else f"  ({unit})"
            cell_text.append([f"{gloss}", f"{val}{unit_s}"])
            row_labels.append(key)
        tbl = ax_tbl.table(cellText=cell_text,
                           rowLabels=row_labels,
                           colLabels=["Description", "Value"],
                           cellLoc="left", rowLoc="left",
                           colLoc="left",
                           colWidths=[0.62, 0.28],
                           bbox=[0.20, 0.0, 0.80, 1.0])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.0)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_linewidth(0.5)
            cell.set_edgecolor(pal["GRD"])
            if r == 0:                       # column header row
                cell.set_facecolor(pal["HDR_BG"])
                cell.set_text_props(color=pal["HDR_TXT"], fontweight="bold")
            else:
                # Zebra-stripe data rows.  Use theme PNL for the
                # "darker" stripes and ZEBRA for the lighter ones.
                cell.set_facecolor(pal["ZEBRA"] if r % 2 == 0 else pal["PNL"])
                if c == -1:                  # row-label column
                    cell.set_text_props(family="monospace", fontsize=8.0,
                                        color=pal["MUT"])
                else:
                    cell.set_text_props(color=pal["TXT"])

        # Footer — explicit short lines instead of `wrap=True`, because
        # matplotlib's fig.text wrap only kicks in when the text would
        # exceed a containing artist's width, NOT the figure width, so
        # long strings just run off the right edge of the PDF (which is
        # what was happening to the References line).  Breaking into
        # pre-wrapped lines side-steps that entirely.
        _foot_kw = dict(fontsize=7, color=pal["MUT"], ha="left",
                        va="bottom", family=pal["FONT"])
        sign_lines = [
            "Sign convention: turning angles are SIGNED on (−180°, +180°].",
            "0° = straight ahead.  +θ = left turn (CCW).  −θ = right "
            "turn (CW).  ±180° = full reversal.",
            "Unsigned 0–360° equivalent: u = θ if θ ≥ 0, else θ + 360 "
            "(so −90° ≡ 270°, +90° ≡ 90°).",
        ]
        ref_lines = [
            "References:",
            "  Mardia & Jupp 2000 — Directional Statistics.",
            "  Fisher 1993 — Statistical Analysis of Circular Data.",
            "  Berens 2009 — CircStat: A MATLAB Toolbox for Circular "
            "Statistics, J. Stat. Soft. 31(10).",
        ]
        y = 0.095
        for line in sign_lines:
            fig.text(0.07, y, line, **_foot_kw)
            y -= 0.014
        y -= 0.006
        for line in ref_lines:
            fig.text(0.07, y, line, **_foot_kw)
            y -= 0.014

        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, facecolor=pal["BG"])
        plt.close(fig)
    finally:
        # Restore rcParams so we don't bleed our palette into whatever
        # plot the caller draws next.
        plt.rcParams.update(_rc_save)


def _circ_watson_williams(samples_deg):
    """k-sample Watson-Williams F-test for equality of mean directions
    across k≥2 circular samples (Mardia & Jupp 2000 §6.4.2).  This is
    the circular analogue of one-way ANOVA: H₀ = all groups share a
    common mean direction.

    Parameters
    ----------
    samples_deg : list of 1-D angle arrays (degrees, any range)

    Returns
    -------
    None if fewer than 2 valid samples, else dict with:
      F, df1, df2, p           — test statistic + degrees of freedom + p
      kappa_pooled, R_bar_pooled
      valid                     — True iff κ̂ ≥ 2 and R̄ ≥ 0.45 (the test
                                  assumes concentrated von Mises samples;
                                  flag the result if not).
      n_per_group, n_total, k
    """
    rad = [np.radians(np.asarray(s, dtype=float).ravel())
           for s in samples_deg]
    rad = [r[np.isfinite(r)] for r in rad]
    rad = [r for r in rad if r.size >= 2]
    k = len(rad)
    if k < 2:
        return None
    n_per = [int(r.size) for r in rad]
    N = int(sum(n_per))
    Ci = np.array([float(np.cos(r).sum()) for r in rad])
    Si = np.array([float(np.sin(r).sum()) for r in rad])
    Ri = np.hypot(Ci, Si)
    Cp = float(Ci.sum()); Sp = float(Si.sum())
    Rp = float(np.hypot(Cp, Sp))
    R_bar = Rp / N
    # Pooled concentration (Best & Fisher 1981).
    if R_bar < 0.53:
        kappa = 2.0 * R_bar + R_bar ** 3 + 5.0 * R_bar ** 5 / 6.0
    elif R_bar < 0.85:
        kappa = -0.4 + 1.39 * R_bar + 0.43 / max(1.0 - R_bar, 1e-12)
    else:
        denom = max(R_bar ** 3 - 4.0 * R_bar ** 2 + 3.0 * R_bar, 1e-12)
        kappa = 1.0 / denom
    # Stephens 1972 K correction (≈1 when κ is large; sharper at low κ).
    K = 1.0 + 3.0 / (8.0 * kappa) if kappa > 0 else 1.0
    sumR = float(Ri.sum())
    denom_f = (k - 1) * (N - sumR)
    if denom_f <= 0:
        return None
    F = K * (N - k) * (sumR - Rp) / denom_f
    df1, df2 = int(k - 1), int(N - k)
    try:
        from scipy.stats import f as _f_dist
        # Use logsf → exp so we get a meaningful tiny p instead of a
        # rounded-to-zero float when F is huge (which is normal with
        # 100k+ angles per group).  logsf returns log(1 - cdf) with
        # log-space stability.
        log_p = float(_f_dist.logsf(F, df1, df2))
        p = 1e-300 if log_p < -700.0 else float(np.exp(log_p))
    except Exception:
        p = float("nan")
    return {
        "F": float(F), "df1": df1, "df2": df2, "p": p,
        "kappa_pooled": float(kappa),
        "R_bar_pooled": float(R_bar),
        "valid": bool(kappa >= 2.0 and R_bar >= 0.45),
        "n_per_group": n_per, "n_total": N, "k": int(k),
    }


def _circ_mardia_watson_wheeler(samples_deg):
    """Mardia-Watson-Wheeler (uniform-scores) non-parametric k-sample
    test for equal CIRCULAR DISTRIBUTIONS across k≥2 groups (Mardia &
    Jupp 2000 §7.6.1).  Unlike Watson-Williams it makes no assumption
    about concentration, so it's the safe fallback when κ < 2 or when
    you suspect groups differ in spread rather than only in mean
    direction.

    Returns None if fewer than 2 valid samples, else dict with:
      W, df, p, n_per_group, n_total, k
    """
    rad = [np.radians(np.asarray(s, dtype=float).ravel())
           for s in samples_deg]
    rad = [r[np.isfinite(r)] for r in rad]
    rad = [r for r in rad if r.size >= 1]
    k = len(rad)
    if k < 2:
        return None
    pooled = np.concatenate(rad)
    N = int(pooled.size)
    try:
        from scipy.stats import rankdata, chi2
    except Exception:
        return None
    ranks = rankdata(pooled, method="average")
    # Convert ranks → uniform circular scores in [0, 2π).
    beta = 2.0 * np.pi * ranks / N
    # Sample-wise C/S sums, then W = 2 · Σ (C² + S²) / n_j.
    W_stat = 0.0
    cursor = 0
    for r in rad:
        n_j = int(r.size)
        end = cursor + n_j
        b = beta[cursor:end]
        Cj = float(np.cos(b).sum())
        Sj = float(np.sin(b).sum())
        W_stat += (Cj * Cj + Sj * Sj) / n_j
        cursor = end
    W = 2.0 * W_stat
    df = int(2 * (k - 1))
    try:
        # logsf for numerical stability — chi2.sf(3.4e3, 2) underflows
        # to 0.0 in float64 but chi2.logsf returns the actual log p.
        log_p = float(chi2.logsf(W, df))
        p = 1e-300 if log_p < -700.0 else float(np.exp(log_p))
    except Exception:
        p = float("nan")
    return {
        "W": float(W), "df": df, "p": p,
        "n_per_group": [int(r.size) for r in rad],
        "n_total": N, "k": int(k),
    }


def _circ_wallraff_ktest(samples_deg):
    """Wallraff k-sample test for equality of circular concentrations.

    H₀ = all samples share the same concentration κ.  Implementation
    follows Mardia & Jupp (2000) §7.5.5: convert each angle to its
    deviation from its own sample's mean direction (mapped to [0, π]),
    then run a rank-sum test on those deviations across groups.

    For k = 2 we use the Mann-Whitney U test; for k > 2 we use the
    Kruskal-Wallis H test.  Returns None if fewer than 2 valid samples.

    Returned dict:
      H or U   : test statistic (key name depends on k)
      df       : degrees of freedom (Kruskal-Wallis only)
      p        : p-value
      n_per_group, n_total, k
    """
    rad = [np.radians(np.asarray(s, dtype=float).ravel())
           for s in samples_deg]
    rad = [r[np.isfinite(r)] for r in rad]
    rad = [r for r in rad if r.size >= 2]
    k = len(rad)
    if k < 2:
        return None
    # Per-sample angular deviation from its OWN mean direction,
    # mapped to [0, π] (the circular distance).
    deviations = []
    for r in rad:
        mu = np.arctan2(np.sin(r).mean(), np.cos(r).mean())
        d  = np.abs(r - mu)
        d  = np.minimum(d, 2.0 * np.pi - d)
        deviations.append(d)
    n_per = [int(d.size) for d in deviations]
    try:
        if k == 2:
            from scipy.stats import mannwhitneyu
            stat, p = mannwhitneyu(deviations[0], deviations[1],
                                   alternative="two-sided")
            return {
                "U": float(stat), "p": float(p), "k": 2,
                "n_per_group": n_per, "n_total": int(sum(n_per)),
            }
        else:
            from scipy.stats import kruskal
            stat, p = kruskal(*deviations)
            return {
                "H": float(stat), "df": int(k - 1),
                "p": float(p), "k": int(k),
                "n_per_group": n_per, "n_total": int(sum(n_per)),
            }
    except Exception:
        return None


def _circ_kuiper_two_sample(a_deg, b_deg):
    """Kuiper two-sample test for equality of circular distributions.

    Non-parametric, distribution-free analogue of the Kolmogorov-Smirnov
    test, adapted for circular data.  Sensitive to differences anywhere
    in the distribution (not just shifts in mean), and unlike the KS
    statistic the Kuiper statistic V = D⁺ + D⁻ is invariant to the
    choice of origin on the circle — a property that matters because
    "where you put 0°" is arbitrary for circular data.

    Returns None if either sample is < 2 elements, else dict:
      V       : Kuiper statistic
      p       : asymptotic p-value (Stephens 1965 series approximation)
      n1, n2  : sample sizes
    """
    a = np.sort(np.mod(np.radians(np.asarray(a_deg, dtype=float).ravel()),
                       2.0 * np.pi))
    b = np.sort(np.mod(np.radians(np.asarray(b_deg, dtype=float).ravel()),
                       2.0 * np.pi))
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    n1, n2 = int(a.size), int(b.size)
    if n1 < 2 or n2 < 2:
        return None

    # Empirical CDFs evaluated at every observation in the combined
    # sample.  V = max(F1 - F2) + max(F2 - F1).
    combined = np.sort(np.concatenate([a, b]))
    F1 = np.searchsorted(a, combined, side="right") / n1
    F2 = np.searchsorted(b, combined, side="right") / n2
    D_plus  = float((F1 - F2).max())
    D_minus = float((F2 - F1).max())
    V = D_plus + D_minus

    # Stephens (1965) asymptotic p-value: λ = (√n_eff + 0.155 + 0.24/√n_eff)·V.
    n_eff = n1 * n2 / (n1 + n2)
    lam = (np.sqrt(n_eff) + 0.155 + 0.24 / np.sqrt(n_eff)) * V
    if lam <= 0:
        p = 1.0
    else:
        # Convergent series in j; cap at j=100 (terms decay
        # exponentially in j²).
        s_terms = 0.0
        l2 = lam * lam
        for j in range(1, 101):
            j2 = j * j
            term = 2.0 * (4.0 * j2 * l2 - 1.0) * np.exp(-2.0 * j2 * l2)
            s_terms += term
            if abs(term) < 1e-18:
                break
        p = float(np.clip(s_terms, 0.0, 1.0))
    if p > 0.0 and p <= 1e-300:
        p = 1e-300
    return {
        "V": float(V), "p": float(p),
        "n1": n1, "n2": n2,
    }


def _circ_lin_correlation(theta_deg, x):
    """Circular-linear correlation (Mardia 1976; Mardia & Jupp 2000
    §6.5.1).

    Tests whether a circular variable θ is associated with a linear
    variable x.  Compute the three Pearson correlations
        r_xc = corr(x, cos θ),  r_xs = corr(x, sin θ),  r_cs = corr(cos θ, sin θ)
    and combine them into the circular-linear coefficient

        R² = (r_xc² + r_xs² − 2·r_xc·r_xs·r_cs) / (1 − r_cs²)

    R ∈ [0, 1] (analogous to a Pearson |r|).  Under H₀ of independence
    and large n, n·R² ~ χ²(2), giving a usable p-value.

    Returns None if n < 3 or the data are degenerate; else dict:
      r, r2          : coefficient and its square
      test_stat      : n · r²
      df, p          : χ²(2) p-value
      n              : effective sample size after finite-mask
    """
    theta = np.asarray(theta_deg, dtype=float).ravel()
    x     = np.asarray(x,         dtype=float).ravel()
    if theta.size != x.size:
        return None
    mask = np.isfinite(theta) & np.isfinite(x)
    theta = theta[mask]; x = x[mask]
    n = int(theta.size)
    if n < 3:
        return None
    rad = np.radians(theta)
    c = np.cos(rad); s = np.sin(rad)
    # Need non-zero variance in x AND in c/s for the correlations to
    # exist.  If all angles are identical (or all x identical), bail.
    if np.std(x) == 0 or np.std(c) == 0 or np.std(s) == 0:
        return None
    rxc = float(np.corrcoef(x, c)[0, 1])
    rxs = float(np.corrcoef(x, s)[0, 1])
    rcs = float(np.corrcoef(c, s)[0, 1])
    denom = 1.0 - rcs ** 2
    if abs(denom) < 1e-12:
        return None
    r2 = (rxc ** 2 + rxs ** 2 - 2.0 * rxc * rxs * rcs) / denom
    r2 = float(np.clip(r2, 0.0, 1.0))
    test_stat = n * r2
    try:
        from scipy.stats import chi2
        log_p = float(chi2.logsf(test_stat, 2))
        p = 1e-300 if log_p < -700.0 else float(np.exp(log_p))
    except Exception:
        p = float("nan")
    return {
        "r": float(np.sqrt(r2)), "r2": r2,
        "test_stat": float(test_stat), "df": 2,
        "p": float(p), "n": n,
    }


def compute_per_track_mean_angle(tracks):
    """For each track in `tracks` with ≥ 3 localisations, compute the
    circular mean of its signed turning angles (degrees on
    (-180°, +180°]).  Returns a list of (particle_id, mean_angle_deg).

    Used to build (angle, D) pairs for the circular-linear correlation
    between a track's turning bias and its diffusion coefficient.
    """
    if len(tracks) < 3:
        return []
    srt = (tracks.reset_index(drop=True)
                 .sort_values(["particle", "frame"], kind="stable"))
    pid_arr = srt["particle"].to_numpy()
    xy_arr  = srt[["x", "y"]].to_numpy()
    steps = np.diff(xy_arr, axis=0)
    same_step = (pid_arr[1:] == pid_arr[:-1])
    if len(steps) < 2:
        return []
    v1 = steps[:-1]; v2 = steps[1:]
    both_in_track = same_step[:-1] & same_step[1:]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot   = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    valid = both_in_track & (norm1 > 0) & (norm2 > 0)
    if not valid.any():
        return []
    angles = np.arctan2(cross[valid], dot[valid])    # radians
    # The middle row of each (i, i+1, i+2) triple is pid_arr[i+1].
    pid_at_turn = pid_arr[1:-1][valid]
    # Bucket angles by particle and compute the circular mean.
    out = []
    for pid in np.unique(pid_at_turn):
        sel = (pid_at_turn == pid)
        rad = angles[sel]
        if rad.size == 0:
            continue
        mu = np.degrees(np.arctan2(np.sin(rad).mean(),
                                   np.cos(rad).mean()))
        out.append((int(pid), float(mu)))
    return out


def _watson_williams_mu_per_replicate(mu_lists_per_group):
    """Watson-Williams F-test on per-replicate mean directions.

    Treats each replicate's mean direction μ_ij as a single circular
    observation (not the underlying angles).  This is the supervisor-
    facing way to compare directionality between groups: the n is
    the number of REPLICATES, not the number of pooled localisations,
    so the test isn't inflated by huge per-file angle counts.

    Parameters
    ----------
    mu_lists_per_group : list aligned with the groups, each entry is
        a 1-D array/list of per-replicate mean directions in DEGREES
        (signed, (-180°, +180°]).

    Returns dict matching the shape `_circ_watson_williams` already
    uses (F, df1, df2, p, valid, ...), or None if fewer than 2 groups
    have ≥ 2 replicates each.
    """
    # _circ_watson_williams already does k-sample WW on a list of
    # angle arrays — pass the per-replicate μ values in as samples.
    samples = [np.asarray(arr, dtype=float).ravel()
               for arr in mu_lists_per_group]
    samples = [a[np.isfinite(a)] for a in samples]
    if sum(1 for a in samples if a.size >= 2) < 2:
        return None
    return _circ_watson_williams(samples)


def compute_circular_comparison_tests(groups, *, track_angle_d_pairs=None,
                                       per_replicate_angles=None):
    """Run all the standard 'do these circular samples differ?' tests on
    a list of labelled groups.

    Parameters
    ----------
    groups : list of (label, angles_deg_array)
        One entry per comparison group; the array is the pooled
        turning angles across all replicates in that group.
    track_angle_d_pairs : optional list aligned with `groups`
        Each element is a 2-tuple of arrays (per_track_mean_angle_deg,
        per_track_D_um2_s).  Used to compute the per-group circular-
        linear correlation between a track's average turning bias and
        its diffusion coefficient.  Pass None to skip the correlation.

    Returns
    -------
    dict with keys:
      omnibus_ww   : Watson-Williams F-test (equal mean directions)
      omnibus_mww  : Mardia-Watson-Wheeler W-test (equal distributions)
      omnibus_wallraff
                   : Wallraff k-sample test (equal concentrations);
                     directly addresses "is one group more tightly
                     clustered than the other?".
      pairwise     : list, one entry per (i, j) with i<j, each with
                     keys label_a, label_b, ww, mww, wallraff, kuiper
                     (Kuiper two-sample test for equal distributions).
      circ_lin_per_group
                   : list aligned with `groups`, dict per group with
                     keys label and result (the _circ_lin_correlation
                     dict, or None if not enough data).  Only populated
                     when track_angle_d_pairs is provided.
    """
    labels = [g[0] for g in groups]
    samples = [g[1] for g in groups]
    # The pooled-angle inferential tests (Watson-Williams,
    # Mardia-Watson-Wheeler, Wallraff κ-test, Kuiper two-sample, and the
    # circular-linear angle-vs-D correlation) are intentionally NOT
    # computed.  Each pools every localisation across all replicates, so
    # n = hundreds of thousands of angles and the test returns p ≈ 0
    # regardless of the true biological effect — classic pseudoreplication.
    # Only the per-replicate tests below (each replicate = one data point)
    # are reported.  The keys are kept (None / []) so the CSV and PDF
    # writers' existing truthiness guards simply skip them.
    # `track_angle_d_pairs` is accepted for call-site compatibility but no
    # longer used.
    out = {
        "omnibus_ww":       None,
        "omnibus_mww":      None,
        "omnibus_wallraff": None,
        "pairwise":         [],
        "circ_lin_per_group": [],
        "per_replicate_kappa_test": None,
        "per_replicate_rbar_test":  None,
        "per_replicate_mu_ww":      None,
        "per_replicate_scalars":    None,
    }

    # ── Per-replicate tests ──────────────────────────────────────────
    # Treats each replicate as ONE data point (its own κ, R̄, μ),
    # producing a defensible Welch's t-test on κ + R̄ (linear scalars)
    # and a Watson-Williams F-test on μ (a circular quantity).  This
    # is the right framing when the user has e.g. 5 vs 3 movies and
    # wants stats that respect the biological replicate count, not the
    # inflated angle-count produced by pooling.
    if per_replicate_angles is not None:
        per_kappa  = []      # list-per-group of replicate κ values
        per_rbar   = []      #          "          "        R̄
        per_mu     = []      #          "          "        μ (deg)
        per_n_reps = []
        scalars_per_group = []
        for label in labels:
            arrs = per_replicate_angles.get(label, [])
            kappas, rbars, mus = [], [], []
            for arr in arrs:
                a = np.asarray(arr, dtype=float).ravel()
                a = a[np.isfinite(a)]
                if a.size < 2:
                    continue
                cs = compute_circular_statistics(a)
                if cs is None:
                    continue
                k_val  = cs.get("concentration_kappa")
                r_val  = cs.get("mean_resultant_length")
                mu_val = cs.get("mean_direction_deg")
                if k_val is not None and np.isfinite(k_val):
                    kappas.append(float(k_val))
                if r_val is not None and np.isfinite(r_val):
                    rbars.append(float(r_val))
                if mu_val is not None and np.isfinite(mu_val):
                    mus.append(float(mu_val))
            per_kappa.append(np.asarray(kappas, dtype=float))
            per_rbar.append(np.asarray(rbars, dtype=float))
            per_mu.append(np.asarray(mus, dtype=float))
            per_n_reps.append(len(kappas))
            scalars_per_group.append({
                "label": label, "n_replicates": len(kappas),
                "kappa": list(kappas), "rbar": list(rbars),
                "mu_deg": list(mus),
            })
        out["per_replicate_scalars"] = scalars_per_group

        # _stat_test_n returns (omnibus_dict, pairwise_list).
        # Welch's t for 2 groups, ANOVA for N>2 (auto-selected).
        if sum(1 for arr in per_kappa if arr.size >= 1) >= 2:
            try:
                om_k, pw_k = _stat_test_n(per_kappa, labels)
                out["per_replicate_kappa_test"] = {
                    "omnibus": om_k, "pairwise": pw_k}
            except Exception:
                pass
            try:
                om_r, pw_r = _stat_test_n(per_rbar, labels)
                out["per_replicate_rbar_test"] = {
                    "omnibus": om_r, "pairwise": pw_r}
            except Exception:
                pass
        out["per_replicate_mu_ww"] = _watson_williams_mu_per_replicate(per_mu)

    return out


def _p_stars(p):
    """Three-tier significance markers used in the comparison PDF."""
    try:
        if p is None: return ""
        pf = float(p)
        if np.isnan(pf): return ""
        if pf < 0.001: return "***"
        if pf < 0.01:  return "**"
        if pf < 0.05:  return "*"
        return "ns"
    except Exception:
        return ""


def save_comparison_circular_statistics(groups_angles, *,
                                         csv_path=None, pdf_path=None,
                                         fig_theme="Dark",
                                         track_angle_d_pairs=None,
                                         per_replicate_angles=None):
    """Pool turning angles per group, compute circular statistics for
    each group, write a combined CSV (one row per group) and a multi-
    page themed PDF (one page per group + a comparative summary page).

    Parameters
    ----------
    groups_angles : list of (label, angles_deg_array, color)
        One entry per comparison group.  `angles_deg_array` is the
        concatenation of every replicate's turning angles within the
        group; `color` is the group's display colour (used to tint the
        polar histograms so PDF and master figure agree visually).
    csv_path : str or None
        If given, write a long-form CSV with columns `group`, `n`,
        `mean_direction_deg`, … (all keys from compute_circular_statistics).
    pdf_path : str or None
        If given, write the multi-page PDF.
    fig_theme : str
        "Dark" | "Light" | "Publication".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pal = _theme_palette(fig_theme)

    # ── Per-group stats ────────────────────────────────────────────────
    rows = []
    per_group_stats = []
    for label, angles, color in groups_angles:
        a = np.asarray(angles, dtype=float).ravel()
        a = a[np.isfinite(a)]
        stats = compute_circular_statistics(a)
        per_group_stats.append((label, a, color, stats))
        row = {"group": label}
        # Drop the pooled-angle uniformity p-values (Rayleigh, V-test):
        # at ~10^5 localisations they are always ~0 and uninformative.
        # Keep the descriptive concentration measures (R̄, κ, mean
        # direction, circular variance/SD) which ARE meaningful.
        row.update({k: v for k, v in stats.items()
                    if k not in ("rayleigh_z", "rayleigh_p",
                                 "v_test_z", "v_test_p", "n")})
        # Name the angle count explicitly (n_angles) so it can't be
        # confused with the per-replicate n (n_replicates) reported in the
        # per-replicate test rows further down.
        if "n" in stats:
            row["n_angles"] = stats["n"]
        rows.append(row)

    # ── Between-group tests ────────────────────────────────────────────
    # Watson-Williams (parametric, tests equal mean directions; assumes
    # κ ≥ 2) plus Mardia-Watson-Wheeler (non-parametric, tests equal
    # distributions; valid at any κ).  Both are reported so the
    # supervisor can pick the appropriate one for their data — and so
    # disagreement between them (one significant, the other not) is
    # visible rather than hidden.
    test_groups = [(g[0], np.asarray(g[1], dtype=float).ravel())
                   for g in groups_angles]
    test_groups = [(lbl, a[np.isfinite(a)]) for lbl, a in test_groups]
    comp_tests = compute_circular_comparison_tests(
        test_groups,
        track_angle_d_pairs=track_angle_d_pairs,
        per_replicate_angles=per_replicate_angles)

    # ── CSV (split into clean single-table files) ──────────────────────
    # Previously every kind of row (per-group descriptive + per-replicate
    # scalars + between-group tests) was unioned into ONE wide, mostly-empty
    # sheet — unreadable in Excel.  Split into three single-purpose tables,
    # each with a uniform schema.  `csv_path` is the legacy
    # "{stem}_circular_statistics.csv"; sibling names are derived from its stem.
    if csv_path is not None:
        try:
            base = csv_path[:-4] if csv_path.lower().endswith(".csv") else csv_path
            if base.endswith("_circular_statistics"):
                base = base[:-len("_circular_statistics")]
            pg_path    = base + "_circular_per_group.csv"
            rep_path   = base + "_circular_per_replicate.csv"
            tests_path = base + "_circular_tests.csv"

            scalars = comp_tests.get("per_replicate_scalars") or []
            n_rep_by_label = {
                g.get("label"): max(len(g.get("kappa") or []),
                                    len(g.get("rbar") or []),
                                    len(g.get("mu_deg") or []))
                for g in scalars}

            # 1) Per-group descriptive statistics — one row per group.
            pg_rows = []
            for r in rows:
                rr = dict(r)
                rr["n_replicates"] = n_rep_by_label.get(rr.get("group"))
                pg_rows.append(rr)
            pg_df = pd.DataFrame(pg_rows)
            lead = [c for c in ("group", "n_angles", "n_replicates")
                    if c in pg_df.columns]
            pg_df = pg_df[lead + [c for c in pg_df.columns if c not in lead]]
            pg_df.to_csv(pg_path, index=False)

            # 2) Per-replicate scalars — one row per (group, replicate).
            rep_rows = []
            for g in scalars:
                lbl = g.get("label", "?")
                ks, rs, ms = (g.get("kappa") or [], g.get("rbar") or [],
                              g.get("mu_deg") or [])
                for i in range(max(len(ks), len(rs), len(ms))):
                    rep_rows.append({
                        "group": lbl, "replicate": i + 1,
                        "kappa":  ks[i] if i < len(ks) else None,
                        "rbar":   rs[i] if i < len(rs) else None,
                        "mu_deg": ms[i] if i < len(ms) else None,
                    })
            pd.DataFrame(rep_rows, columns=["group", "replicate",
                                            "kappa", "rbar", "mu_deg"]
                         ).to_csv(rep_path, index=False)

            # 3) Between-group per-replicate tests — uniform schema.
            test_rows = []

            def _add_scalar_test(slot, metric):
                t = comp_tests.get(slot)
                if not t:
                    return
                om = t.get("omnibus") or {}
                if om:
                    test_rows.append({
                        "metric": metric, "scope": "omnibus",
                        "label_a": "all groups", "label_b": "",
                        "test": om.get("test"), "p_value": om.get("p"),
                        "stars": om.get("stars", ""), "note": om.get("note", ""),
                    })
                for pw in (t.get("pairwise") or []):
                    test_rows.append({
                        "metric": metric, "scope": "pairwise",
                        "label_a": pw.get("label_i"), "label_b": pw.get("label_j"),
                        "test": pw.get("test"), "p_value": pw.get("p"),
                        "stars": pw.get("stars", ""),
                        "n_rep_a": pw.get("n_i"), "n_rep_b": pw.get("n_j"),
                        "mean_a": pw.get("mean_i"), "mean_b": pw.get("mean_j"),
                        "sem_a": pw.get("sem_i"), "sem_b": pw.get("sem_j"),
                        "cohens_d": pw.get("cohens_d"),
                        "hedges_g": pw.get("hedges_g"),
                        "hedges_g_ci_low": pw.get("hedges_g_ci_low"),
                        "hedges_g_ci_high": pw.get("hedges_g_ci_high"),
                        "note": pw.get("note", ""),
                    })

            _add_scalar_test("per_replicate_kappa_test", "kappa (concentration)")
            _add_scalar_test("per_replicate_rbar_test",  "Rbar (resultant length)")

            mu_ww = comp_tests.get("per_replicate_mu_ww")
            if mu_ww is not None:
                test_rows.append({
                    "metric": "mu (mean direction)", "scope": "omnibus",
                    "label_a": "all groups", "label_b": "",
                    "test": "Watson-Williams F (per-replicate)",
                    "statistic_F": mu_ww.get("F"),
                    "df1": mu_ww.get("df1"), "df2": mu_ww.get("df2"),
                    "p_value": mu_ww.get("p"), "note": mu_ww.get("note", ""),
                })

            tests_cols = ["metric", "scope", "label_a", "label_b", "test",
                          "statistic_F", "df1", "df2", "p_value", "stars",
                          "n_rep_a", "n_rep_b", "mean_a", "mean_b",
                          "sem_a", "sem_b", "cohens_d", "hedges_g",
                          "hedges_g_ci_low", "hedges_g_ci_high", "note"]
            tdf = pd.DataFrame(test_rows)
            tdf = tdf.reindex(columns=[c for c in tests_cols if c in tdf.columns])
            tdf.to_csv(tests_path, index=False)

            for p in (pg_path, rep_path, tests_path):
                print(f"  Saved: {p}")
        except Exception as exc:
            print(f"  comparison-circstats CSV failed: {exc}")

    # ── PDF ────────────────────────────────────────────────────────────
    if pdf_path is None:
        return per_group_stats

    _rc_keys = ("text.color", "axes.labelcolor", "axes.edgecolor",
                "xtick.color", "ytick.color", "axes.facecolor",
                "axes.titlecolor", "figure.facecolor", "grid.color",
                "font.family")
    _rc_save = {k: plt.rcParams.get(k) for k in _rc_keys}
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

    def _fmt(x, prec=4):
        try:
            if x is None: return "—"
            xf = float(x)
            if np.isnan(xf): return "—"
            if xf > 0.0 and xf <= 1e-300:
                return "<1e-300"
            return f"{xf:.{prec}g}"
        except Exception:
            return str(x)

    try:
        with PdfPages(pdf_path) as pdf:
            # ── Page 1: comparison summary ─────────────────────────────
            # Landscape A4.  Layout, top → bottom:
            #   y 0.93 – 0.97  header bar
            #   y 0.58 – 0.88  row of polar histograms (one per group)
            #   y 0.51 – 0.55  "Summary" title
            #   y 0.36 – 0.50  per-group summary table
            #   y 0.30 – 0.34  "Between-group tests" title
            #   y 0.13 – 0.29  comparison-tests table
            #   y 0.02 – 0.10  footer block (sign convention + refs)
            fig = plt.figure(figsize=(11.69, 8.27), facecolor=pal["BG"])
            ax_hdr = fig.add_axes([0.05, 0.93, 0.90, 0.04])
            ax_hdr.axis("off")
            ax_hdr.text(0.0, 0.5, "Comparison: Circular Statistics",
                        fontsize=18, fontweight="bold", va="center",
                        ha="left", color=pal["TXT"])
            ax_hdr.text(1.0, 0.5,
                        f"{len(per_group_stats)} groups",
                        fontsize=11, color=pal["MUT"],
                        va="center", ha="right")

            # Grid of polar histograms — auto-wraps to multiple rows
            # when n_groups > 5 so plots don't get sliver-thin.  Each
            # cell is divided VERTICALLY into a label strip (top) and
            # the polar plot itself (below); doing it this way means
            # the group name + n count can never collide with the
            # polar's 0° tick label, regardless of how thick that tick
            # label is at any given font size.
            #
            #   1 ≤ n ≤ 5  →  1 row of n cols  (cell height 0.30, y 0.55–0.88)
            #   6 ≤ n ≤ 10 →  2 rows of ≤ 5 cols  (cell height ~0.16)
            #   n ≥ 11     →  3 rows; outer caller may also paginate.
            #
            # Within each cell:
            #   top 22%  → label band  (group name + "n = N")
            #   bottom 78% → polar plot
            #
            # When in multi-row mode the per-polar font sizes shrink so
            # the tick labels stay readable in a smaller plot.
            n_g  = len(per_group_stats)
            # Polar band height tuned so the polar plot + its top label
            # band (group name + n) sit comfortably above the
            # per-group summary title at y=0.555.  polar_bot=0.58
            # leaves a 0.025 gap to that title.
            polar_top, polar_bot = 0.88, 0.58
            if n_g <= 5:
                n_cols, n_rows = n_g, 1
            elif n_g <= 10:
                n_cols, n_rows = 5, 2
            else:
                # Cap at 12 polars/page; the table-pagination below
                # handles "lots of groups" by giving each batch its
                # own summary page.  For now assume ≤ 12 on page 1.
                n_cols = 6
                n_rows = (min(n_g, 12) + n_cols - 1) // n_cols
            row_h    = (polar_top - polar_bot) / n_rows
            cell_w   = 0.86 / n_cols
            left     = 0.07
            # Tick / label fontsizes shrink when polars get small.
            tick_fs  = 7 if n_cols <= 4 else 6
            lbl_fs   = 10 if n_cols <= 4 else 8
            n_fs     = 8  if n_cols <= 4 else 7
            # Bottom-margin reserves space for the polar's ±180° tick
            # label, which matplotlib renders OUTSIDE the axes box just
            # below the polar circle.  Needs to be large enough that
            # the tick label can't reach down into the Per-group title
            # at y=0.535 below (single-row case) or into the next-row's
            # group label (multi-row case).
            bottom_margin = 0.040 if n_rows == 1 else 0.045
            label_band_frac = 0.22 if n_rows == 1 else 0.28
            for i, (label, a, color, stats) in enumerate(per_group_stats):
                if i >= n_rows * n_cols:
                    break    # truncate at the page's polar capacity
                row = i // n_cols
                col = i % n_cols
                cell_y = polar_top - (row + 1) * row_h
                label_band_h = label_band_frac * row_h
                polar_band_h = row_h - label_band_h - bottom_margin
                ax = fig.add_axes(
                    [left + col * cell_w + 0.015,
                     cell_y + bottom_margin,
                     cell_w - 0.03, polar_band_h],
                    projection="polar")
                ax.set_facecolor(pal["PNL"])
                if a.size >= 10:
                    # Match the master figure's polar convention:
                    # 0° at top, CW positive, signed labels on slot
                    # positions, [0, 2π) wrap for bar rendering.
                    nbins = 36
                    angles_rad = np.mod(np.deg2rad(a), 2.0 * np.pi)
                    bins  = np.linspace(0.0, 2.0 * np.pi, nbins + 1)
                    counts, edges = np.histogram(angles_rad, bins=bins)
                    widths  = np.diff(edges)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    ax.set_theta_zero_location("N")
                    ax.set_theta_direction(-1)
                    bar_col = color or pal["ACC"]
                    ax.bar(centers, counts, width=widths * 0.95,
                           align="center", color=bar_col,
                           edgecolor=pal["PNL"], linewidth=0.4,
                           alpha=0.92)
                    mu = stats.get("mean_direction_deg")
                    if mu is not None and not (
                            isinstance(mu, float) and np.isnan(mu)):
                        r_max = float(counts.max()) if counts.size else 1.0
                        mu_rad = np.mod(np.deg2rad(mu), 2.0 * np.pi)
                        ax.annotate("",
                            xy=(mu_rad, r_max * 0.95),
                            xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->",
                                            color=pal["ARROW"], lw=2.0))
                    ax.set_xticks(np.deg2rad(
                        [0, 45, 90, 135, 180, 225, 270, 315]))
                    ax.set_xticklabels(
                        ["0°", "+45°", "+90°", "+135°", "±180°",
                         "−135°", "−90°", "−45°"], fontsize=tick_fs)
                    ax.set_yticklabels([])
                    ax.tick_params(colors=pal["TXT"], labelsize=tick_fs)
                    ax.grid(True, ls=":", alpha=0.4)
                    # Labels live ABOVE the cell.  Raising the label
                    # block above `cell_y + row_h` (rather than just
                    # inside the top of the cell) creates a clean gap
                    # between the "n = …" line and the polar's 0° tick
                    # label, which renders just outside the polar
                    # circle at the top of the axes box.
                    label_x = (left + col * cell_w + 0.015
                               + (cell_w - 0.03) / 2.0)
                    label_top  = cell_y + row_h + 0.020
                    line2_top  = label_top - 0.018
                    fig.text(label_x, label_top, label,
                             fontsize=lbl_fs, fontweight="bold",
                             ha="center", va="top", color=pal["TXT"])
                    fig.text(label_x, line2_top,
                             f"n = {int(stats.get('n', 0)):,}",
                             fontsize=n_fs, ha="center", va="top",
                             color=pal["MUT"])
                else:
                    ax.axis("off")
                    label_x = (left + col * cell_w + 0.015
                               + (cell_w - 0.03) / 2.0)
                    label_top = cell_y + row_h + 0.020
                    fig.text(label_x, label_top,
                             f"{label}\ntoo few angles",
                             fontsize=n_fs, ha="center", va="top",
                             color=pal["MUT"])

            # Section title placed in FIGURE coords.  Sits below the
            # polar band's bottom (y=0.58) with a generous gap so the
            # polar's ±180° tick label can't reach down into it.
            fig.text(0.05, 0.535,
                     "Per-group summary  (MATLAB CircStat conventions)",
                     fontsize=11, fontweight="bold", va="bottom",
                     ha="left", color=pal["TXT"])
            # Combined summary table — one row per group, columns =
            # the most informative stats for an at-a-glance comparison.
            ax_tbl = fig.add_axes([0.05, 0.43, 0.90, 0.10])
            ax_tbl.axis("off")
            # Descriptive per-group concentration measures only.  The
            # Rayleigh / V-test uniformity p-values were removed: at
            # ~10^5 pooled localisations they are always ~0 and convey
            # nothing — R̄ and κ already quantify how concentrated each
            # group's turning-angle distribution is.
            cols = ["group", "n", "mean_direction_deg",
                    "mean_resultant_length", "circular_std_deg",
                    "concentration_kappa"]
            col_labels = ["Group", "n (angles)", "μ (°)", "R̄",
                          "σ_circ (°)", "κ"]
            cell = []
            for r in rows:
                cell.append([
                    str(r["group"]),
                    f"{int(r['n_angles']):,}",
                    _fmt(r["mean_direction_deg"], 4),
                    _fmt(r["mean_resultant_length"], 4),
                    _fmt(r["circular_std_deg"], 4),
                    _fmt(r["concentration_kappa"], 4),
                ])
            tbl = ax_tbl.table(cellText=cell, colLabels=col_labels,
                               cellLoc="left", colLoc="left",
                               bbox=[0.0, 0.0, 1.0, 1.0])
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9.0)
            for (rr, cc), c_obj in tbl.get_celld().items():
                c_obj.set_linewidth(0.5)
                c_obj.set_edgecolor(pal["GRD"])
                if rr == 0:
                    c_obj.set_facecolor(pal["HDR_BG"])
                    c_obj.set_text_props(color=pal["HDR_TXT"],
                                         fontweight="bold")
                else:
                    c_obj.set_facecolor(
                        pal["ZEBRA"] if rr % 2 == 0 else pal["PNL"])
                    c_obj.set_text_props(color=pal["TXT"])

            # ── Between-group tests section ────────────────────────
            #
            # Layout (y-coords):
            #   0.34 : section title
            #   0.27 – 0.33 : plain-English explanation of each test
            #   0.13 – 0.26 : results table (Test  Statistic  p  sig)
            #   0.02 – 0.10 : footer
            #
            # The previous version had a 5th "Note" column for H₀
            # descriptions which overflowed the page; pulling that
            # description out into a separate explanatory paragraph
            # both fixes the overflow AND makes the tests intelligible
            # to a reader who isn't already a circular-statistics
            # expert (supervisor's request).
            fig.text(0.05, 0.395,
                     "Between-group tests — does the turning-angle "
                     "distribution differ between groups?",
                     fontsize=11, fontweight="bold", va="bottom",
                     ha="left", color=pal["TXT"])

            # Plain-English explanation block — 3 compact lines so the
            # tests table below still has room.  Each line covers one
            # test (or the significance convention).  Italicised
            # caveats appear at the end of each line, not on their own
            # row.
            txt_kw = dict(fontsize=8.0, color=pal["TXT"], ha="left",
                          va="top", family=pal["FONT"])
            explain_block = [
                "Per-replicate tests treat each replicate (movie / cell) "
                "as ONE data point, so n = the number of replicates — "
                "the correct biological unit.",
                "Per-replicate κ and R̄ (turning-angle concentration) are "
                "compared with a normality-checked test — one-way ANOVA / "
                "Welch's t when normal, Kruskal-Wallis / Mann-Whitney otherwise "
                "(the test used is named on each row); Watson-Williams F on "
                "per-replicate μ tests the mean direction.",
                "Pooled-angle tests (Watson-Williams / Mardia-Watson-"
                "Wheeler / Wallraff / Kuiper on all localisations) are "
                "deliberately NOT shown: with ~10^5 pooled angles they "
                "return p ≈ 0 regardless of effect (pseudoreplication).",
                "Significant p (< 0.05, stars) rejects H₀ — i.e. the "
                "groups DO differ at the replicate level.",
            ]
            yE = 0.395
            for line in explain_block:
                fig.text(0.05, yE, line, **txt_kw)
                yE -= 0.012
            # Build comparison-test rows in priority order:
            #   1. Omnibus Watson-Williams
            #   2. Omnibus Mardia-Watson-Wheeler
            #   3. Pairwise WW / MWW (one row per test per pair)
            # _fmt_p collapses underflow-sentinel p (1e-300) to
            # "<1e-300" so the supervisor doesn't see a literal
            # "1e-300" repeated across rows and assume there's a bug.
            def _fmt_p(p):
                if p is None: return "—"
                pf = float(p)
                if np.isnan(pf): return "—"
                if pf > 0.0 and pf <= 1e-300:
                    return "<1e-300"
                return f"{pf:.3g}"

            omnibus_rows = []
            ow  = comp_tests.get("omnibus_ww")
            om  = comp_tests.get("omnibus_mww")
            owk = comp_tests.get("omnibus_wallraff")
            if ow is not None:
                tag = "" if ow.get("valid", False) else "  (κ<2, caution)"
                omnibus_rows.append([
                    f"Watson-Williams · all groups{tag}",
                    f"F({ow['df1']}, {ow['df2']}) = {ow['F']:.3g}",
                    _fmt_p(ow["p"]),
                    _p_stars(ow["p"]),
                ])
            if om is not None:
                omnibus_rows.append([
                    "Mardia-Watson-Wheeler · all groups",
                    f"W({om['df']}) = {om['W']:.3g}",
                    _fmt_p(om["p"]),
                    _p_stars(om["p"]),
                ])
            if owk is not None:
                # k=2 → Mann-Whitney U; k>2 → Kruskal-Wallis H.
                if "H" in owk:
                    stat_str = f"H({owk['df']}) = {owk['H']:.3g}"
                else:
                    stat_str = f"U = {owk.get('U', 0):.3g}"
                omnibus_rows.append([
                    "Wallraff κ-test · all groups",
                    stat_str,
                    _fmt_p(owk["p"]),
                    _p_stars(owk["p"]),
                ])

            pairwise_rows = []
            for pw in comp_tests.get("pairwise", []):
                ww  = pw.get("ww")
                mww = pw.get("mww")
                wal = pw.get("wallraff")
                kup = pw.get("kuiper")
                pair = f"{pw['label_a']}  vs  {pw['label_b']}"
                if ww is not None:
                    tag = "" if ww.get("valid", False) else "  (κ<2)"
                    pairwise_rows.append([
                        f"Watson-Williams · {pair}{tag}",
                        f"F({ww['df1']}, {ww['df2']}) = {ww['F']:.3g}",
                        _fmt_p(ww["p"]),
                        _p_stars(ww["p"]),
                    ])
                if mww is not None:
                    pairwise_rows.append([
                        f"Mardia-Watson-Wheeler · {pair}",
                        f"W({mww['df']}) = {mww['W']:.3g}",
                        _fmt_p(mww["p"]),
                        _p_stars(mww["p"]),
                    ])
                if wal is not None:
                    pairwise_rows.append([
                        f"Wallraff κ-test · {pair}",
                        f"U = {wal.get('U', 0):.3g}",
                        _fmt_p(wal["p"]),
                        _p_stars(wal["p"]),
                    ])
                if kup is not None:
                    pairwise_rows.append([
                        f"Kuiper 2-sample · {pair}",
                        f"V = {kup['V']:.4g}",
                        _fmt_p(kup["p"]),
                        _p_stars(kup["p"]),
                    ])

            # Per-group circular-linear correlation rows.  These are
            # descriptive stats (one per group), not between-group
            # tests, but they live in the same table because they share
            # the same "name · stat · p · sig" template.
            corr_rows = []
            for cl in comp_tests.get("circ_lin_per_group", []):
                res = cl.get("result")
                grp = cl.get("label", "?")
                if not res:
                    corr_rows.append([
                        f"Circ-lin angle vs D · {grp}",
                        "n < 3", "—", "",
                    ])
                    continue
                corr_rows.append([
                    f"Circ-lin angle vs D · {grp}",
                    (f"r = {res['r']:.3g}  "
                     f"(χ²({res['df']}) = {res['test_stat']:.3g})"),
                    _fmt_p(res["p"]),
                    _p_stars(res["p"]),
                ])

            # ── Per-replicate test rows ─────────────────────────────
            # n = number of biological replicates, not pooled angles.
            # Each row reads "Welch κ · all groups" or "Welch κ · A vs
            # B" plus the t/F statistic and the p-value with stars.
            per_rep_rows = []

            def _push_per_rep(slot, label):
                t = comp_tests.get(slot)
                if not t:
                    return
                om = t.get("omnibus") or {}
                if om and om.get("p") is not None:
                    test_name = om.get("test", "")
                    per_rep_rows.append([
                        f"{label} · all groups  ({test_name})",
                        "(see CSV for full stats)",
                        _fmt_p(om["p"]),
                        _p_stars(om["p"]),
                    ])
                for pw in (t.get("pairwise") or []):
                    if pw.get("p") is None:
                        continue
                    pair = (f"{pw.get('label_i', '?')}  vs  "
                            f"{pw.get('label_j', '?')}")
                    n_i = pw.get("n_i", 0)
                    n_j = pw.get("n_j", 0)
                    # "reps" makes explicit these are replicate counts, not
                    # turning-angle counts.  When the test is NOT significant,
                    # append the power-based sample size so the reader sees
                    # how many replicates/group would be needed at 80% power.
                    stat_cell = f"{n_i} vs {n_j} reps"
                    p_val = pw.get("p")
                    n80 = pw.get("n_needed_80")
                    if (p_val is not None and np.isfinite(p_val)
                            and p_val >= 0.05 and n80 is not None):
                        n80_str = ">500" if n80 > 500 else str(int(n80))
                        stat_cell += f" · need ~{n80_str}/grp (80%)"
                    per_rep_rows.append([
                        f"{label} · {pair}  ({pw.get('test', '')})",
                        stat_cell,
                        _fmt_p(pw["p"]),
                        _p_stars(pw["p"]),
                    ])

            # Prefix names the METRIC only — the actual test (auto-selected by a
            # normality check: ANOVA/Welch when normal, Kruskal-Wallis/Mann-
            # Whitney otherwise) is shown in parentheses per row.  (It used to
            # say "Welch κ", which contradicted rows whose test was Kruskal-
            # Wallis / Mann-Whitney.)
            _push_per_rep("per_replicate_kappa_test", "κ (per-replicate)")
            _push_per_rep("per_replicate_rbar_test",  "R̄ (per-replicate)")

            mu_ww = comp_tests.get("per_replicate_mu_ww")
            if mu_ww is not None and mu_ww.get("p") is not None:
                tag = "" if mu_ww.get("valid", False) else "  (κ<2)"
                per_rep_rows.append([
                    f"Watson-Williams μ · all groups (per-replicate){tag}",
                    f"F({mu_ww['df1']}, {mu_ww['df2']}) = {mu_ww['F']:.3g}",
                    _fmt_p(mu_ww["p"]),
                    _p_stars(mu_ww["p"]),
                ])

            # Paginate ALL test rows so page 1 is never overstuffed.  The
            # omnibus + per-replicate "fixed" rows used to bypass the cap, so
            # with several groups 20+ rows got squashed into a ~1.4" strip.  Now
            # everything paginates, ordered by importance (omnibus → per-
            # replicate → circ-lin → pooled pairwise) so the most relevant tests
            # land on page 1 and the rest flow onto roomy continuation pages.
            PAGE1_TESTS_CAP = 11        # page 1 shares space with summary + text
            CONT_PAGE_CAP   = 20        # full-height continuation pages
            all_test_rows = (omnibus_rows + per_rep_rows + corr_rows
                             + pairwise_rows)
            if not all_test_rows:
                all_test_rows = [["Insufficient data", "—", "—", "—"]]
            page1_tests    = all_test_rows[:PAGE1_TESTS_CAP]
            overflow_pairs = all_test_rows[PAGE1_TESTS_CAP:]

            def _render_tests_table(host_fig, rect, cells, pal):
                """Render a 4-column tests table into the given fig+rect."""
                ax = host_fig.add_axes(rect); ax.axis("off")
                tbl = ax.table(
                    cellText=cells,
                    colLabels=["Test  ·  Comparison", "Statistic",
                               "p-value", "sig"],
                    cellLoc="left", colLoc="left",
                    colWidths=[0.60, 0.25, 0.10, 0.05],
                    bbox=[0.0, 0.0, 1.0, 1.0])
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(8.5)
                for (rr, cc), c_obj in tbl.get_celld().items():
                    c_obj.set_linewidth(0.5)
                    c_obj.set_edgecolor(pal["GRD"])
                    c_obj.PAD = 0.06          # a little breathing room in-cell
                    if rr == 0:
                        c_obj.set_facecolor(pal["HDR_BG"])
                        c_obj.set_text_props(color=pal["HDR_TXT"],
                                             fontweight="bold")
                    else:
                        c_obj.set_facecolor(
                            pal["ZEBRA"] if rr % 2 == 0 else pal["PNL"])
                        c_obj.set_text_props(color=pal["TXT"])

            # Taller area + a hard 11-row cap → each row gets ~2x the height it
            # had before, so the table reads cleanly instead of cramped.
            _render_tests_table(fig, [0.05, 0.12, 0.90, 0.205],
                                page1_tests, pal)

            # ── Footer block ──────────────────────────────────────
            # Pre-wrapped lines instead of one long sign-convention
            # string: matplotlib's fig.text doesn't wrap against the
            # figure margins, so the long "Sign convention…" line was
            # being cut off at the right edge.  Same wrapping pattern
            # the per-file PDF footer uses (search `sign_lines = [`).
            def _render_footer(host_fig, pal, *, top_y=0.105):
                _foot_kw2 = dict(fontsize=7, color=pal["MUT"], ha="left",
                                 va="bottom", family=pal["FONT"])
                foot_lines = [
                    "Sign convention: turning angles are SIGNED on "
                    "(−180°, +180°].",
                    "0° = straight ahead.  +θ = left turn (CCW).  "
                    "−θ = right turn (CW).  ±180° = full reversal.",
                    "Plots use clockwise-positive direction so +θ "
                    "labels appear on the right hemisphere.",
                    "Significance markers: *** p<0.001,  ** p<0.01,  "
                    "* p<0.05,  ns = not significant.",
                    "References: Mardia & Jupp 2000 §6.4.2, §7.6.1; "
                    "Fisher 1993; Berens 2009 (CircStat).",
                ]
                y2 = top_y
                for line in foot_lines:
                    host_fig.text(0.05, y2, line, **_foot_kw2)
                    y2 -= 0.014

            _render_footer(fig, pal)
            pdf.savefig(fig, facecolor=pal["BG"])
            plt.close(fig)

            # ── Continuation pages for overflow pairwise tests ──────
            # When the pairwise count is large (e.g. 6+ groups → 15+
            # pairs × 2 tests = 30+ rows), we paginate the remainder
            # onto fresh landscape pages so nothing gets squashed off
            # the bottom of page 1.
            if overflow_pairs:
                page_num = 2
                total_cont_pages = (len(overflow_pairs)
                                    + CONT_PAGE_CAP - 1) // CONT_PAGE_CAP
                for chunk_start in range(0, len(overflow_pairs),
                                         CONT_PAGE_CAP):
                    chunk = overflow_pairs[chunk_start:
                                           chunk_start + CONT_PAGE_CAP]
                    fig_c = plt.figure(figsize=(11.69, 8.27),
                                       facecolor=pal["BG"])
                    ax_h = fig_c.add_axes([0.05, 0.93, 0.90, 0.04])
                    ax_h.axis("off")
                    ax_h.text(0.0, 0.5,
                              "Comparison: Circular Statistics  —  "
                              f"between-group tests, continued "
                              f"(page {page_num - 1} of {total_cont_pages})",
                              fontsize=14, fontweight="bold",
                              va="center", ha="left", color=pal["TXT"])
                    # Big tests-table area on a continuation page.
                    _render_tests_table(fig_c,
                                        [0.05, 0.13, 0.90, 0.75],
                                        chunk, pal)
                    _render_footer(fig_c, pal)
                    pdf.savefig(fig_c, facecolor=pal["BG"])
                    plt.close(fig_c)
                    page_num += 1

            # ── Pages 2..N+1: per-group full report ───────────────────
            for label, a, color, stats in per_group_stats:
                # Reuse the single-file renderer by writing to a
                # temp page object isn't supported directly — instead,
                # we mirror its layout here in a fresh figure so the
                # per-group pages all live in ONE multi-page PDF.
                _write_single_group_page(pdf, a, stats, label, pal, color)
    finally:
        plt.rcParams.update(_rc_save)

    return per_group_stats


def _write_single_group_page(pdf, angles_deg, stats, label, pal,
                              group_color=None):
    """Render one A4-portrait page mirroring save_circular_statistics_pdf
    into an open PdfPages stream.  Used by save_comparison_circular_
    statistics so the per-group full reports all live inside the same
    multi-page comparison PDF.
    """
    import matplotlib.pyplot as plt

    a = np.asarray(angles_deg, dtype=float).ravel()
    a = a[np.isfinite(a)]

    def _fmt(x, prec=4):
        try:
            if x is None: return "—"
            xf = float(x)
            if np.isnan(xf): return "—"
            if xf > 0.0 and xf <= 1e-300:
                return "<1e-300"
            return f"{xf:.{prec}g}"
        except Exception:
            return str(x)

    rows = [
        ("n", "Sample size", "count", f"{int(stats.get('n', 0)):,}"),
        ("mean_direction_deg", "Mean direction μ", "deg",
         _fmt(stats.get("mean_direction_deg"), 4)),
        ("mean_resultant_length",
         "Mean resultant length R̄  (0 = uniform, 1 = aligned)", "—",
         _fmt(stats.get("mean_resultant_length"), 4)),
        ("circular_variance", "Circular variance  1 − R̄", "—",
         _fmt(stats.get("circular_variance"), 4)),
        ("circular_std_deg",
         "Circular standard deviation  √(−2·ln R̄)", "deg",
         _fmt(stats.get("circular_std_deg"), 4)),
        ("angular_deviation_deg",
         "Angular deviation  s₀ = √(2·(1−R̄))", "deg",
         _fmt(stats.get("angular_deviation_deg"), 4)),
        ("median_deg", "Circular median", "deg",
         _fmt(stats.get("median_deg"), 4)),
        ("concentration_kappa",
         "Von Mises concentration κ  (Best & Fisher 1981)", "—",
         _fmt(stats.get("concentration_kappa"), 4)),
        ("rayleigh_z", "Rayleigh test statistic  z = n·R̄²", "—",
         _fmt(stats.get("rayleigh_z"), 4)),
        ("rayleigh_p", "Rayleigh test p-value  (uniformity)", "—",
         _fmt(stats.get("rayleigh_p"), 3)),
        ("v_test_z", "V-test statistic against μ₀ = 0°", "—",
         _fmt(stats.get("v_test_z"), 4)),
        ("v_test_p", "V-test p-value  (preferred direction)", "—",
         _fmt(stats.get("v_test_p"), 3)),
        ("circular_skewness",
         "Circular skewness  (Mardia & Jupp §2.3)", "—",
         _fmt(stats.get("circular_skewness"), 4)),
        ("circular_kurtosis",
         "Circular kurtosis  (Mardia & Jupp §2.3)", "—",
         _fmt(stats.get("circular_kurtosis"), 4)),
        ("ci95_lower_deg",
         "95% CI lower bound for μ  (large-sample)", "deg",
         _fmt(stats.get("ci95_lower_deg"), 4)),
        ("ci95_upper_deg",
         "95% CI upper bound for μ  (large-sample)", "deg",
         _fmt(stats.get("ci95_upper_deg"), 4)),
    ]

    # Layout mirrors save_circular_statistics_pdf — same coord bands so
    # the per-group page in a comparison PDF reads like a per-file PDF.
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=pal["BG"])
    ax_hdr = fig.add_axes([0.07, 0.94, 0.86, 0.04])
    ax_hdr.axis("off")
    ax_hdr.text(0.0, 0.5, f"Circular Statistics — {label}",
                fontsize=15, fontweight="bold", va="center",
                ha="left", color=pal["TXT"])
    ax_hdr.text(1.0, 0.5,
                f"n = {int(stats.get('n', 0)):,} turning angles",
                fontsize=11, color=pal["MUT"], va="center", ha="right")

    ax_polar = fig.add_axes([0.08, 0.61, 0.36, 0.25], projection="polar")
    ax_polar.set_facecolor(pal["PNL"])
    if a.size >= 10:
        # Same convention as the master figure: 0° top, CW positive,
        # signed labels on slot positions, [0, 2π) wrap for bars.
        nbins = 36
        angles_rad = np.mod(np.deg2rad(a), 2.0 * np.pi)
        bins  = np.linspace(0.0, 2.0 * np.pi, nbins + 1)
        counts, edges = np.histogram(angles_rad, bins=bins)
        widths = np.diff(edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax_polar.set_theta_zero_location("N")
        ax_polar.set_theta_direction(-1)
        ax_polar.bar(centers, counts, width=widths * 0.95, align="center",
                     color=group_color or pal["ACC"],
                     edgecolor=pal["PNL"], linewidth=0.4, alpha=0.92)
        mu = stats.get("mean_direction_deg")
        if mu is not None and not (isinstance(mu, float) and np.isnan(mu)):
            r_max = float(counts.max()) if counts.size else 1.0
            mu_rad = np.mod(np.deg2rad(mu), 2.0 * np.pi)
            ax_polar.annotate("",
                xy=(mu_rad, r_max * 0.95), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=pal["ARROW"],
                                lw=2.0))
        ax_polar.set_xticks(np.deg2rad(
            [0, 45, 90, 135, 180, 225, 270, 315]))
        ax_polar.set_xticklabels(
            ["0°", "+45°", "+90°", "+135°", "±180°",
             "−135°", "−90°", "−45°"], fontsize=8)
        ax_polar.set_yticklabels([])
        ax_polar.tick_params(colors=pal["TXT"], labelsize=8)
        ax_polar.grid(True, ls=":", alpha=0.4)
        # Title intentionally omitted — see save_circular_statistics_pdf
        # for the rationale (header + footer already cover it).
    else:
        ax_polar.axis("off")
        ax_polar.text(0.5, 0.5, "Too few angles for histogram",
                      transform=ax_polar.transAxes,
                      ha="center", va="center", color=pal["MUT"],
                      fontsize=10)

    # Compact "Top stats" box for the right side.
    ax_top = fig.add_axes([0.48, 0.61, 0.46, 0.25]); ax_top.axis("off")
    R = stats.get("mean_resultant_length")
    p = stats.get("rayleigh_p")
    lines = [
        f"Mean direction μ:        {_fmt(stats.get('mean_direction_deg'), 4)}°",
        f"Resultant length R̄:      {_fmt(stats.get('mean_resultant_length'), 4)}",
        f"Concentration κ:         {_fmt(stats.get('concentration_kappa'), 4)}",
        f"Rayleigh p (uniformity): {_fmt(stats.get('rayleigh_p'), 3)}",
        f"V-test p (μ₀ = 0°):      {_fmt(stats.get('v_test_p'), 3)}",
    ]
    ax_top.text(0.0, 1.0, "Headline stats", fontsize=12,
                fontweight="bold", va="top", color=pal["TXT"])
    ax_top.text(0.0, 0.88, "\n".join(lines), fontsize=10, va="top",
                family="monospace", color=pal["TXT"])

    # Section title in figure coords so it can't collide with the polar
    # plot's bottom tick labels above.
    fig.text(0.07, 0.555, "Statistics  (MATLAB CircStat conventions)",
             fontsize=12, fontweight="bold", va="bottom",
             ha="left", color=pal["TXT"])
    ax_tbl = fig.add_axes([0.07, 0.12, 0.88, 0.40]); ax_tbl.axis("off")

    cell_text, row_labels = [], []
    for key, gloss, unit, val in rows:
        unit_s = "" if unit in ("", "—") else f"  ({unit})"
        cell_text.append([f"{gloss}", f"{val}{unit_s}"])
        row_labels.append(key)
    tbl = ax_tbl.table(cellText=cell_text, rowLabels=row_labels,
                       colLabels=["Description", "Value"],
                       cellLoc="left", rowLoc="left", colLoc="left",
                       colWidths=[0.62, 0.28],
                       bbox=[0.20, 0.0, 0.80, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.0)
    for (rr, cc), c_obj in tbl.get_celld().items():
        c_obj.set_linewidth(0.5)
        c_obj.set_edgecolor(pal["GRD"])
        if rr == 0:
            c_obj.set_facecolor(pal["HDR_BG"])
            c_obj.set_text_props(color=pal["HDR_TXT"], fontweight="bold")
        else:
            c_obj.set_facecolor(
                pal["ZEBRA"] if rr % 2 == 0 else pal["PNL"])
            if cc == -1:
                c_obj.set_text_props(family="monospace", fontsize=8.0,
                                     color=pal["MUT"])
            else:
                c_obj.set_text_props(color=pal["TXT"])

    _foot_kw = dict(fontsize=7, color=pal["MUT"], ha="left",
                    va="bottom", family=pal["FONT"])
    sign_lines = [
        "Sign convention: turning angles SIGNED on (−180°, +180°].",
        "0° = straight.  +θ = left turn (CCW).  −θ = right turn (CW).  "
        "±180° = reversal.",
        "Unsigned 0–360° equivalent: u = θ if θ ≥ 0, else θ + 360 "
        "(so −90° ≡ 270°, +90° ≡ 90°).",
    ]
    ref_lines = [
        "References: Mardia & Jupp 2000; Fisher 1993; "
        "Berens 2009 (CircStat).",
    ]
    y = 0.095
    for line in sign_lines:
        fig.text(0.07, y, line, **_foot_kw); y -= 0.014
    y -= 0.006
    for line in ref_lines:
        fig.text(0.07, y, line, **_foot_kw); y -= 0.014
    pdf.savefig(fig, facecolor=pal["BG"])
    plt.close(fig)


def _stat_test(a, b):
    """Two-sample test on per-experiment scalars.  Welch's t by default,
    Mann-Whitney as fallback for non-normal data.  Returns (p, label)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return (np.nan, "")
    try:
        from scipy.stats import ttest_ind, mannwhitneyu, shapiro
        normal = True
        for arr in (a, b):
            if 3 <= len(arr) <= 5000:
                try:
                    if shapiro(arr).pvalue < 0.05:
                        normal = False
                        break
                except Exception:
                    pass
        if normal:
            p = ttest_ind(a, b, equal_var=False).pvalue
        else:
            p = mannwhitneyu(a, b, alternative="two-sided").pvalue
        if not np.isfinite(p):
            return (np.nan, "")
        if p < 0.001: stars = "***"
        elif p < 0.01: stars = "**"
        elif p < 0.05: stars = "*"
        else: stars = "ns"
        return (float(p), stars)
    except Exception:
        return (np.nan, "")

def _cohens_d_pooled(a, b):
    """Pooled-SD Cohen's d for two 1-D arrays.  None if either group has
    < 2 points or the pooled SD is zero."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if not np.isfinite(sp) or sp == 0:
        return None
    return float((np.mean(a) - np.mean(b)) / sp)


def _hedges_g_ci(a, b, n_boot=2000, seed=0):
    """Hedges' g (small-sample-corrected Cohen's d) with a percentile
    bootstrap 95% CI.

    Cohen's d is biased upward at small n; Hedges' g multiplies it by the
    correction factor J = 1 − 3/(4·(na+nb) − 9).  Returns
    ``(g, ci_low, ci_high)`` or ``(None, None, None)`` when undefined.
    The CI is more honest than a star at the small replicate counts typical
    of SPT (n = a few cells/experiments per group)."""
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return (None, None, None)
    d = _cohens_d_pooled(a, b)
    if d is None:
        return (None, None, None)
    J = 1.0 - 3.0 / (4.0 * (na + nb) - 9.0)
    g = float(J * d)
    try:
        rng = np.random.default_rng(seed)
        boot = np.empty(n_boot, dtype=float)
        k = 0
        for _ in range(n_boot):
            da = _cohens_d_pooled(rng.choice(a, na, replace=True),
                                  rng.choice(b, nb, replace=True))
            if da is not None and np.isfinite(da):
                boot[k] = J * da
                k += 1
        if k >= max(20, n_boot // 10):
            lo, hi = np.percentile(boot[:k], [2.5, 97.5])
            return (g, float(lo), float(hi))
    except Exception:
        pass
    return (g, None, None)


def _paired_test(pre, post):
    """Two-sided PAIRED test on matched per-cell values (the same cell measured
    at two time points).  Paired t-test when the differences look normal,
    Wilcoxon signed-rank otherwise.  `pre` and `post` must be aligned 1-D
    arrays (same cell at the same index).  Returns (p, stars)."""
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    m = np.isfinite(pre) & np.isfinite(post)
    pre, post = pre[m], post[m]
    if len(pre) < 2:
        return (np.nan, "")
    diff = post - pre
    if np.allclose(diff, 0.0):
        return (np.nan, "")
    try:
        from scipy.stats import ttest_rel, wilcoxon, shapiro
        normal = True
        if 3 <= len(diff) <= 5000:
            try:
                if shapiro(diff).pvalue < 0.05:
                    normal = False
            except Exception:
                pass
        if normal:
            p = ttest_rel(pre, post).pvalue
        else:
            p = wilcoxon(pre, post).pvalue
        if not np.isfinite(p):
            return (np.nan, "")
        if p < 0.001: stars = "***"
        elif p < 0.01: stars = "**"
        elif p < 0.05: stars = "*"
        else: stars = "ns"
        return (float(p), stars)
    except Exception:
        return (np.nan, "")


def _paired_hedges_g(pre, post):
    """Within-subject (paired) effect size: d_z = mean(diff) / SD(diff), with
    the small-sample Hedges correction J = 1 − 3/(4·(n−1) − 1).  This is the
    paired analogue of Hedges' g — it uses the SD of the per-cell differences,
    so it accounts for the PRE/POST correlation.  Returns g or None."""
    pre = np.asarray(pre, dtype=float)
    post = np.asarray(post, dtype=float)
    m = np.isfinite(pre) & np.isfinite(post)
    diff = post[m] - pre[m]
    n = len(diff)
    if n < 2:
        return None
    sd = float(np.std(diff, ddof=1))
    if not np.isfinite(sd) or sd == 0:
        return None
    dz = float(np.mean(diff)) / sd
    J = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0)
    return float(J * dz)


def _n_per_group_for_power(d, power=0.80, alpha=0.05, n_max=500):
    """Smallest n PER GROUP for a two-sample, two-sided t-test to reach
    `power` at significance `alpha`, given Cohen's d.

    Uses exact noncentral-t power (no statsmodels dependency).  Returns
    an int n, or None if d is unusable, or `n_max + 1` as a ">n_max"
    sentinel when the effect is too small to be practically reachable.

    Caveat for callers: this is a PARAMETRIC (t-test) estimate.  When the
    auto-selected test was Mann-Whitney (non-normal data) it is an
    approximation, and it assumes the OBSERVED effect size is the true
    one — noisy at small n.  Treat it as a planning guide, not a promise.
    """
    if d is None:
        return None
    d = abs(float(d))
    if d < 1e-6:
        return n_max + 1
    try:
        from scipy import stats as _st
        for n in range(2, n_max + 1):
            df = 2 * n - 2
            nc = d * np.sqrt(n / 2.0)
            tcrit = _st.t.ppf(1.0 - alpha / 2.0, df)
            pw = (1.0 - _st.nct.cdf(tcrit, df, nc)
                  + _st.nct.cdf(-tcrit, df, nc))
            if pw >= power:
                return int(n)
    except Exception:
        return None
    return n_max + 1


def _stat_test_n(arrays, labels):
    """Statistical test across N≥2 groups.

    Returns
    -------
    omnibus : dict with keys {"test", "p", "stars"} or None if n<2 each
    pairwise : list of dicts with keys
        {"i", "j", "label_i", "label_j", "test", "p", "stars",
         "n_i", "n_j", "mean_i", "mean_j", "sem_i", "sem_j"}
    """
    arrs = [np.asarray(a, dtype=float)[np.isfinite(np.asarray(a, dtype=float))]
            for a in arrays]
    valid_idx = [i for i, a in enumerate(arrs) if len(a) >= 2]

    omnibus = None
    pairwise = []

    def _star(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    if len(valid_idx) < 2:
        # Still record per-pair "ns" rows for stats CSV completeness
        for i in range(len(arrs)):
            for j in range(i + 1, len(arrs)):
                pairwise.append({
                    "i": i, "j": j,
                    "label_i": labels[i], "label_j": labels[j],
                    "test": "n<2", "p": np.nan, "stars": "",
                    "note": "n<2 replicates - no test possible",
                    "n_i": int(len(arrs[i])), "n_j": int(len(arrs[j])),
                    "mean_i": float(arrs[i].mean()) if len(arrs[i]) else np.nan,
                    "mean_j": float(arrs[j].mean()) if len(arrs[j]) else np.nan,
                    "sem_i": (float(arrs[i].std(ddof=1) / np.sqrt(len(arrs[i])))
                              if len(arrs[i]) > 1 else np.nan),
                    "sem_j": (float(arrs[j].std(ddof=1) / np.sqrt(len(arrs[j])))
                              if len(arrs[j]) > 1 else np.nan),
                    "cohens_d": None,
                    "hedges_g": None, "hedges_g_ci_low": None,
                    "hedges_g_ci_high": None,
                    "n_needed_80": None, "n_needed_90": None,
                })
        return omnibus, pairwise

    # Omnibus test
    try:
        from scipy.stats import f_oneway, kruskal, shapiro
        valid_arrs = [arrs[i] for i in valid_idx]

        normal = True
        for a in valid_arrs:
            if 3 <= len(a) <= 5000:
                try:
                    if shapiro(a).pvalue < 0.05:
                        normal = False
                        break
                except Exception:
                    pass

        if len(valid_arrs) == 2:
            from scipy.stats import ttest_ind, mannwhitneyu
            if normal:
                p = ttest_ind(*valid_arrs, equal_var=False).pvalue
                test_name = "Welch's t-test"
            else:
                p = mannwhitneyu(*valid_arrs, alternative="two-sided").pvalue
                test_name = "Mann-Whitney U"
        else:
            if normal:
                p = f_oneway(*valid_arrs).pvalue
                test_name = "One-way ANOVA"
            else:
                p = kruskal(*valid_arrs).pvalue
                test_name = "Kruskal-Wallis"
        if np.isfinite(p):
            # Same n<3 validity guard as the pairwise rows: if any group has
            # fewer than 3 replicates, report the p but blank the stars and
            # flag it underpowered.
            underpowered = any(len(a) < 3 for a in valid_arrs)
            omnibus = {
                "test": test_name, "p": float(p),
                "stars": "" if underpowered else _star(p),
                "note": ("n<3 replicates - underpowered, not interpretable"
                         if underpowered else ""),
            }
    except Exception:
        pass

    # Pairwise comparisons
    try:
        from scipy.stats import ttest_ind, mannwhitneyu, shapiro
        for i in range(len(arrs)):
            for j in range(i + 1, len(arrs)):
                a, b = arrs[i], arrs[j]
                if len(a) < 2 or len(b) < 2:
                    p = np.nan
                    test_name = "n<2"
                else:
                    is_normal = True
                    for arr in (a, b):
                        if 3 <= len(arr) <= 5000:
                            try:
                                if shapiro(arr).pvalue < 0.05:
                                    is_normal = False
                                    break
                            except Exception:
                                pass
                    if is_normal:
                        p = ttest_ind(a, b, equal_var=False).pvalue
                        test_name = "Welch's t-test"
                    else:
                        p = mannwhitneyu(a, b, alternative="two-sided").pvalue
                        test_name = "Mann-Whitney U"
                # Effect size + power-based sample-size estimate: how many
                # replicates PER GROUP would be needed to detect this
                # observed effect at 80% / 90% power (α=0.05, two-sided).
                d_eff = _cohens_d_pooled(a, b)
                g_eff, g_lo, g_hi = _hedges_g_ci(a, b)
                # Validity guard: with < 3 replicates per group the test is
                # uninterpretable (n=2 has 1 d.o.f.).  Keep the p-value for
                # reference but DON'T advertise significance — blank the stars
                # and flag the comparison as underpowered.
                underpowered = (len(a) < 3 or len(b) < 3)
                note = ("n<3 replicates - underpowered, not interpretable"
                        if underpowered else "")
                pairwise.append({
                    "i": i, "j": j,
                    "label_i": labels[i], "label_j": labels[j],
                    "test": test_name,
                    "p": float(p) if np.isfinite(p) else np.nan,
                    "stars": ("" if underpowered else
                              (_star(p) if np.isfinite(p) else "")),
                    "note": note,
                    "n_i": int(len(a)), "n_j": int(len(b)),
                    "mean_i": float(a.mean()) if len(a) else np.nan,
                    "mean_j": float(b.mean()) if len(b) else np.nan,
                    "sem_i": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.nan,
                    "sem_j": float(b.std(ddof=1) / np.sqrt(len(b))) if len(b) > 1 else np.nan,
                    "cohens_d":    d_eff,
                    "hedges_g":    g_eff,
                    "hedges_g_ci_low":  g_lo,
                    "hedges_g_ci_high": g_hi,
                    "n_needed_80": _n_per_group_for_power(d_eff, 0.80),
                    "n_needed_90": _n_per_group_for_power(d_eff, 0.90),
                })
    except Exception:
        pass

    return omnibus, pairwise
