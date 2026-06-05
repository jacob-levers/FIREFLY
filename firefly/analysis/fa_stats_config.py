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


# ── Plain-English glossary for the Compare/Statistics UI ─────────────────────
# One-sentence, jargon-light definitions surfaced next to technical terms via
# small "ⓘ" info icons (see ui_widgets._info_icon).  Kept here — UI-free and
# import-light — so the same wording can later feed CSV/PDF captions or tests.
# Keys are the exact term strings the UI passes in.
STATS_GLOSSARY = {
    "Significance α":
        "The p-value threshold below which a result is called statistically "
        "significant (conventionally 0.05).",
    "Correction":
        "A multiple-comparison adjustment that stops you from collecting false "
        "positives when many tests are run at once.",
    "Family-wise":
        "Correct p-values across all the metrics together, not just within each "
        "metric on its own — a stricter, more honest bar.",
    "Parametric":
        "A test that assumes the data follow a normal (bell-curve) distribution.",
    "Parametric strategy":
        "How FIREFLY decides between normal-theory (parametric) and rank-based "
        "(non-parametric) tests: automatically per metric, or forced one way.",
    "Mann–Whitney":
        "A rank-based two-group test that makes no assumption of a normal "
        "distribution — the non-parametric counterpart of the t-test.",
    "Welch's t-test":
        "A two-group test of means that does not assume the groups have equal "
        "variance (the safer default t-test).",
    "Welch's ANOVA":
        "A test for 3+ groups that does not assume equal variances across "
        "groups — the unequal-variance version of one-way ANOVA.",
    "One-way ANOVA":
        "The classic 3+-group test of means, which assumes every group has the "
        "same variance.",
    "Kruskal–Wallis":
        "A rank-based test for 3+ groups — the non-parametric counterpart of "
        "one-way ANOVA.",
    "Holm":
        "A step-down correction that controls the chance of any false positive "
        "across your tests, with more power than plain Bonferroni.",
    "Benjamini–Hochberg FDR":
        "Controls the expected proportion of false positives among the results "
        "you call significant, rather than the chance of any single one.",
    "Effect-size CI":
        "The confidence interval around the effect size — the range of "
        "plausible true effect magnitudes, not just whether it's significant.",
    "Hedges' g":
        "A standardized effect size (like Cohen's d) saying how large a "
        "difference is in standard-deviation units, corrected for small samples.",
    "Replicate":
        "An independent experimental unit — here one cell/field of view — which "
        "is the level the statistics are computed on (never per-track).",
    "Two-way mixed ANOVA":
        "Tests group, time, and their interaction together when the same cells "
        "are measured at more than one time point.",
    "Sphericity":
        "The assumption that the spread of the differences between every pair of "
        "time points is about equal; when violated a correction is applied.",
    "Greenhouse–Geisser":
        "A correction that adjusts a repeated-measures ANOVA's degrees of "
        "freedom when sphericity is violated, guarding against false positives.",
    "Interaction effect":
        "Whether the change over time differs between your groups — usually the "
        "key result in a group × time experiment.",
    "Figure stars":
        "Whether the asterisks drawn on the figure use the corrected p-values "
        "(matching the CSV) rather than the raw ones.",
}


def glossary_def(term):
    """Return the plain-English definition for a glossary term, or '' if the
    term isn't in `STATS_GLOSSARY`."""
    return STATS_GLOSSARY.get(term, "")


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
