"""Single source of truth for the Compare-tab statistics configuration.

This module is deliberately UI-free and dependency-light (numpy + an optional
lazy scipy import for FDR) so it can be imported on both the GUI side (to build
the Statistics tab + persist settings) and the worker/analysis side (to drive
test selection, multiple-comparison correction, and the figure/CSV/PDF labels).

The whole point of routing every consumer through `normalize_stats_config` is
transparency + backward compatibility: a `None` config reproduces sensible
defaults, missing keys are back-filled, and bad values are clamped — so an old
saved settings file or an old caller can never put the pipeline into an
undefined statistical state.
"""
from __future__ import annotations

import numpy as np


# ── Canonical config shape + defaults ────────────────────────────────────────
DEFAULT_STATS_CONFIG = {
    "alpha":                      0.05,     # significance level
    "correction":                 "holm",   # none | bonferroni | holm | fdr_bh
    "across_metric_correction":   False,    # also correct across the scalar metrics
    "parametric_strategy":        "auto",   # auto | force_parametric | force_nonparametric
    "anova3plus":                 "welch",  # welch | oneway | auto   (3+ groups, parametric)
    "ci_level":                   0.95,     # effect-size confidence-interval coverage
    "figure_stars_use_corrected": True,     # on-figure stars use corrected p (not raw)
}

_CORRECTIONS = ("none", "bonferroni", "holm", "fdr_bh")
_STRATEGIES  = ("auto", "force_parametric", "force_nonparametric")
_ANOVA3      = ("welch", "oneway", "auto")

# Human-readable names for labels on figures / CSV / PDF.
_CORRECTION_DISPLAY = {
    "none":       "uncorrected",
    "bonferroni": "Bonferroni",
    "holm":       "Holm",
    "fdr_bh":     "Benjamini-Hochberg FDR",
}


def normalize_stats_config(cfg):
    """Return a complete, validated copy of `cfg` overlaid on the defaults.

    Every consumer should call this first.  Unknown keys are ignored; missing
    keys are filled from `DEFAULT_STATS_CONFIG`; out-of-range / unrecognised
    values fall back to the default for that field.  Always returns a dict of
    plain JSON/pickle-safe primitives.
    """
    out = dict(DEFAULT_STATS_CONFIG)
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if k in out:
                out[k] = v

    # alpha ∈ (0, 0.5]
    try:
        a = float(out["alpha"])
        out["alpha"] = a if (0.0 < a <= 0.5) else DEFAULT_STATS_CONFIG["alpha"]
    except (TypeError, ValueError):
        out["alpha"] = DEFAULT_STATS_CONFIG["alpha"]

    # ci_level ∈ (0.5, 0.999]
    try:
        c = float(out["ci_level"])
        out["ci_level"] = c if (0.5 < c <= 0.999) else DEFAULT_STATS_CONFIG["ci_level"]
    except (TypeError, ValueError):
        out["ci_level"] = DEFAULT_STATS_CONFIG["ci_level"]

    # enum strings
    out["correction"] = str(out["correction"]).lower()
    if out["correction"] not in _CORRECTIONS:
        out["correction"] = DEFAULT_STATS_CONFIG["correction"]
    out["parametric_strategy"] = str(out["parametric_strategy"]).lower()
    if out["parametric_strategy"] not in _STRATEGIES:
        out["parametric_strategy"] = DEFAULT_STATS_CONFIG["parametric_strategy"]
    out["anova3plus"] = str(out["anova3plus"]).lower()
    if out["anova3plus"] not in _ANOVA3:
        out["anova3plus"] = DEFAULT_STATS_CONFIG["anova3plus"]

    # bools
    out["across_metric_correction"]   = bool(out["across_metric_correction"])
    out["figure_stars_use_corrected"] = bool(out["figure_stars_use_corrected"])
    return out


def correct_pvalues(pvals, method="holm"):
    """Apply a multiple-comparison correction to a list of p-values.

    Returns a list aligned with the input.  Non-finite entries (None / NaN) are
    passed through unchanged and are EXCLUDED from the comparison count `m`, so
    blank/underpowered rows never inflate the correction.  With m ≤ 1 the
    correction is the identity (a single test needs no correction).
    """
    method = str(method or "none").lower()
    n = len(pvals)
    out = [float("nan")] * n
    idx = [i for i, v in enumerate(pvals)
           if v is not None and np.isfinite(v)]
    if not idx:
        return out
    finite = np.array([float(pvals[i]) for i in idx], dtype=float)
    m = finite.size

    if method in ("none", "raw") or m == 1:
        corr = np.clip(finite, 0.0, 1.0)
    elif method == "bonferroni":
        corr = np.clip(finite * m, 0.0, 1.0)
    elif method == "holm":
        # Step-down: sort ascending, corrected_(k) = max_{l≤k} (m-l)·p_(l),
        # enforced monotone non-decreasing, clipped to 1.
        order = np.argsort(finite, kind="stable")
        corr_sorted = np.empty(m, dtype=float)
        running = 0.0
        for rank, oi in enumerate(order):
            running = max(running, (m - rank) * finite[oi])
            corr_sorted[rank] = min(running, 1.0)
        corr = np.empty(m, dtype=float)
        corr[order] = corr_sorted
    elif method in ("fdr_bh", "bh", "fdr"):
        try:
            from scipy.stats import false_discovery_control
            corr = np.clip(false_discovery_control(finite, method="bh"), 0.0, 1.0)
        except Exception:
            # Hand-rolled Benjamini-Hochberg step-up.
            order = np.argsort(finite, kind="stable")
            corr_sorted = np.empty(m, dtype=float)
            prev = 1.0
            for rank in range(m - 1, -1, -1):
                oi = order[rank]
                prev = min(prev, finite[oi] * m / (rank + 1))
                corr_sorted[rank] = min(prev, 1.0)
            corr = np.empty(m, dtype=float)
            corr[order] = corr_sorted
    else:
        corr = np.clip(finite, 0.0, 1.0)

    for k, i in enumerate(idx):
        out[i] = float(corr[k])
    return out


def stars_for(p, alpha=0.05):
    """Significance stars.  'ns' is gated on `alpha` (so a stricter α makes a
    borderline p non-significant), while the ***/**/* sub-tiers keep the
    conventional 0.001 / 0.01 / 0.05 breakpoints."""
    try:
        if p is None:
            return ""
        pf = float(p)
        if not np.isfinite(pf):
            return ""
    except (TypeError, ValueError):
        return ""
    if pf >= float(alpha):
        return "ns"
    if pf < 0.001:
        return "***"
    if pf < 0.01:
        return "**"
    return "*"


def correction_display(method):
    """Human-readable name for a correction method key."""
    return _CORRECTION_DISPLAY.get(str(method).lower(), str(method))


def describe_test_label(test_name, correction, across_metric=False):
    """Build the 'what test + what correction' string shown on figures, in CSV
    headers, and in the PDF report, so the output is self-describing."""
    test_name = test_name or ""
    corr = str(correction or "none").lower()
    disp = correction_display(corr)
    if corr == "none":
        tail = "uncorrected"
        if across_metric:
            tail += " (family-wise applied across metrics)"
    else:
        tail = f"{disp} within metric"
        if across_metric:
            tail += f" + {disp} across metrics"
    return f"{test_name} · {tail}" if test_name else tail


def config_summary_rows(cfg):
    """Return a list of (label, value) rows describing the config — used by the
    CSV/PDF header blocks so the exact statistical settings are recorded with
    the results."""
    cfg = normalize_stats_config(cfg)
    return [
        ("Significance level (alpha)",   f"{cfg['alpha']:g}"),
        ("Parametric strategy",          cfg["parametric_strategy"]),
        ("Test for 3+ groups",           cfg["anova3plus"]),
        ("Within-metric correction",     correction_display(cfg["correction"])),
        ("Across-metric correction",
         (f"yes ({correction_display(cfg['correction'])})"
          if cfg["across_metric_correction"] else "no")),
        ("Effect-size CI level",         f"{cfg['ci_level']:g}"),
        ("Figure stars use corrected p", "yes" if cfg["figure_stars_use_corrected"] else "no"),
    ]
