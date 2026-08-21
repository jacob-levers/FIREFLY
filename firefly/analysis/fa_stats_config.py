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
    # ── Alternative tests / post-hocs / equivalence ──────────────────────────
    "nonparametric_test":         "mann_whitney",  # mann_whitney | brunner_munzel | permutation
    "posthoc":                    "auto",   # auto | games_howell | dunn | tukey (3+ groups)
    "control_group":              "",       # label of the control group, or "" (none)
    "dunnett":                    False,    # all-vs-control Dunnett test when a control is set
    "equivalence_tost":           False,    # report TOST equivalence per pair
    "tost_margin":                0.5,      # equivalence margin in pooled-SD units
    # ── Circular (turning-angle) statistics ──────────────────────────────────
    # The circular per-replicate comparison tests REUSE the keys above (alpha,
    # correction, parametric_strategy, anova3plus, figure_stars_use_corrected)
    # so they agree with the scalar output.  These extra keys only gate which
    # circular outputs/tests are produced.
    "include_circular_outputs":   True,     # write the circular CSV + PDF outputs
    "circ_test_kappa":            True,     # report between-group concentration-κ test
    "circ_test_rbar":             True,     # report between-group resultant-length R̄ test
    "circ_test_mu":               True,     # report Watson-Williams mean-direction μ test
    "circ_test_circlin":          True,     # report circular-linear (angle vs D) correlation
}

_CORRECTIONS   = ("none", "bonferroni", "holm", "fdr_bh", "sidak", "hochberg")
_STRATEGIES    = ("auto", "force_parametric", "force_nonparametric")
_ANOVA3        = ("welch", "oneway", "auto")
_NONPARAMETRIC = ("mann_whitney", "brunner_munzel", "permutation")
_POSTHOC       = ("auto", "games_howell", "dunn", "tukey")

# Human-readable names for labels on figures / CSV / PDF.
_CORRECTION_DISPLAY = {
    "none":       "uncorrected",
    "bonferroni": "Bonferroni",
    "holm":       "Holm",
    "fdr_bh":     "Benjamini-Hochberg FDR",
    "sidak":      "Šidák",
    "hochberg":   "Hochberg",
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

    out["nonparametric_test"] = str(out["nonparametric_test"]).lower()
    if out["nonparametric_test"] not in _NONPARAMETRIC:
        out["nonparametric_test"] = DEFAULT_STATS_CONFIG["nonparametric_test"]

    # post-hoc: accept the legacy alias "pairwise" as "auto"
    out["posthoc"] = str(out["posthoc"]).lower()
    if out["posthoc"] == "pairwise":
        out["posthoc"] = "auto"
    if out["posthoc"] not in _POSTHOC:
        out["posthoc"] = DEFAULT_STATS_CONFIG["posthoc"]

    # control_group is a free-text group label (validity vs the actual labels is
    # checked at run time in the test engine, since this module is data-free).
    out["control_group"] = str(out["control_group"] or "")

    # tost_margin ∈ (0, 5]  (pooled-SD units)
    try:
        m = float(out["tost_margin"])
        out["tost_margin"] = m if (0.0 < m <= 5.0) else DEFAULT_STATS_CONFIG["tost_margin"]
    except (TypeError, ValueError):
        out["tost_margin"] = DEFAULT_STATS_CONFIG["tost_margin"]

    # bools
    out["across_metric_correction"]   = bool(out["across_metric_correction"])
    out["figure_stars_use_corrected"] = bool(out["figure_stars_use_corrected"])
    out["dunnett"]          = bool(out["dunnett"])
    out["equivalence_tost"] = bool(out["equivalence_tost"])
    for _k in ("include_circular_outputs", "circ_test_kappa", "circ_test_rbar",
               "circ_test_mu", "circ_test_circlin"):
        out[_k] = bool(out[_k])
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
    elif method in ("sidak", "šidák", "sidák"):
        # Single-step Šidák: slightly sharper than Bonferroni under
        # independence.  adj = 1 - (1 - p)^m.
        corr = np.clip(1.0 - np.power(1.0 - finite, m), 0.0, 1.0)
    elif method == "hochberg":
        # Step-up (Hochberg, 1988): like the BH sweep below but with the
        # Holm-style multiplier (m - rank) — more powerful than Holm.
        order = np.argsort(finite, kind="stable")
        corr_sorted = np.empty(m, dtype=float)
        prev = 1.0
        for rank in range(m - 1, -1, -1):
            oi = order[rank]
            prev = min(prev, (m - rank) * finite[oi])
            corr_sorted[rank] = min(prev, 1.0)
        corr = np.empty(m, dtype=float)
        corr[order] = corr_sorted
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
    # ── Alternative tests / post-hocs / equivalence / robust effect sizes ─────
    "Non-parametric test":
        "Which rank/permutation-based test to use when the non-parametric "
        "branch is taken: Mann-Whitney, Brunner-Munzel, or a permutation test.",
    "Šidák":
        "A multiple-comparison correction slightly sharper than Bonferroni "
        "when the tests are independent (adjusted p = 1 − (1 − p)ᵐ).",
    "Hochberg":
        "A step-up correction that controls the chance of any false positive — "
        "uniformly more powerful than Holm under independence.",
    "Brunner–Munzel":
        "A robust two-group rank test that — unlike Mann-Whitney — does NOT "
        "assume the two groups have the same shape/spread.",
    "Permutation test":
        "Builds the null by reshuffling the group labels thousands of times — "
        "makes no distributional assumption, ideal for very small samples.",
    "Post-hoc test":
        "Which pairwise follow-up to run after a 3+-group omnibus test: "
        "per-pair, Games-Howell, Dunn, or Tukey HSD.",
    "Games–Howell":
        "A pairwise post-hoc for 3+ groups that does not assume equal "
        "variances or equal group sizes; it controls the family-wise error itself.",
    "Dunn's test":
        "The standard rank-based pairwise follow-up after a Kruskal-Wallis "
        "test (non-parametric).",
    "Tukey HSD":
        "The classic pairwise post-hoc after one-way ANOVA; assumes equal "
        "variances and controls the family-wise error itself.",
    "Dunnett's test":
        "Compares every group to one designated control group (not all pairs), "
        "with built-in family-wise control — fewer, more powerful comparisons.",
    "Control group":
        "The reference group (e.g. wild-type / untreated) that Dunnett's test "
        "compares every other group against.",
    "Cliff's delta":
        "A distribution-free effect size from −1 to +1: the probability a value "
        "from one group exceeds one from the other, minus the reverse.",
    "Rank-biserial":
        "A rank-based effect size paired with Mann-Whitney — the standardized "
        "difference expressed on a −1 to +1 scale.",
    "Omnibus effect size":
        "How much of the total variation the grouping explains overall — η² for "
        "ANOVA-type tests, ε² for Kruskal-Wallis.",
    "Equivalence (TOST)":
        "Two one-sided tests asking whether two groups are practically the same "
        "within a chosen margin — the opposite question to a difference test.",
    # ── Circular (turning-angle) statistics ──────────────────────────────────
    "Turning angle":
        "The change in direction between two consecutive steps of a track — 0° "
        "means it kept going straight, ±180° means it reversed.",
    "Sign convention":
        "Turning angles are signed on (−180°, +180°]: 0° = straight ahead, "
        "+θ = a left turn (counter-clockwise), −θ = a right turn, ±180° = a full "
        "reversal.",
    "Radial distribution":
        "A polar (compass-style) view of the turning angles, highlighting any "
        "left/right or forward/back asymmetry in the motion.",
    "Rayleigh test":
        "Tests whether turning angles are spread uniformly around the circle (no "
        "preferred direction) versus clustered around one direction.",
    "V-test":
        "Like the Rayleigh test but asks specifically whether angles cluster "
        "around a chosen direction (here 0°, i.e. directed / straight-ahead).",
    "Directional persistence (VACF)":
        "How much a particle tends to keep moving in the same direction step to "
        "step — positive = directed/persistent, near zero = Brownian, negative = "
        "bouncing back (caged).",
    "Concentration κ":
        "How tightly turning angles cluster around their mean direction — larger "
        "κ means more sharply directed motion (the von Mises 1/variance analogue).",
    "Mean resultant length R̄":
        "A 0-to-1 measure of how concentrated the turning angles are: R̄≈0 = "
        "uniformly scattered, R̄≈1 = all pointing the same way.",
    "Watson-Williams":
        "The circular analogue of ANOVA/t-test — tests whether groups share the "
        "same mean turning direction; assumes reasonably concentrated angles (κ≥2).",
    "Circular-linear correlation":
        "Correlates a circular quantity (a track's average turning angle) with a "
        "linear one (its diffusion coefficient) — do more-directed tracks diffuse "
        "differently?  (Not currently emitted in a comparison: pooling per-track "
        "pairs across replicates inflates n to thousands, so the p-value is "
        "meaningless — the same pseudoreplication reason the other pooled "
        "circular tests are reported per-replicate instead.)",
    "Circular outputs":
        "The extra circular-statistics CSV files and PDF report produced alongside "
        "the comparison figure (separate from the turning-angle figure panels).",
}


# Plain-English definitions for the jargon-heavy ANALYSIS-tab parameters,
# surfaced by the same ⓘ info icons (ui_widgets._info_icon).  Kept here next to
# STATS_GLOSSARY so a single glossary_def() lookup resolves both.  Wording is
# distilled from the controls' own tooltips.
ANALYSIS_GLOSSARY = {
    "pixel size":
        "The physical size of one camera pixel, in micrometres — it sets the "
        "length scale for every distance and for the diffusion coefficient.",
    "frame interval":
        "The time between consecutive frames, in seconds — it sets the time "
        "axis of the MSD curve and the units of D.",
    "channel":
        "Which channel of a multi-channel CZI image to load and analyse.",
    "background method":
        "How the slowly-varying background is removed before spot detection "
        "(e.g. a uniform/rolling-ball filter) so real PSFs stand out.",
    "background radius":
        "The size (px) of the background-smoothing window — roughly a few times "
        "the spot diameter; too small eats real signal, too large leaves haze.",
    "diameter":
        "The expected spot size in pixels (an odd number) — it should match "
        "your point-spread function so the detector finds true emitters.",
    "minmass":
        "The minimum integrated brightness a detection must have to be kept — "
        "the main signal-vs-noise cut. Auto picks it from the data per file.",
    "sensitivity":
        "Nudges the automatic threshold stricter (fewer, cleaner detections) or "
        "more lenient (more detections, more noise).",
    "false-track rate":
        "Caps the measured fraction of spurious short fragments the automatic "
        "threshold is allowed to leave behind.",
    "search range":
        "The maximum distance (px) a particle may move between frames for the "
        "linker to connect it into a track (palmTRACER's \"maximum distance\") "
        "— too large invites mis-links. Guide: ~5 px for cytosolic proteins "
        "(e.g. Munc18), ~3 px for transmembrane proteins (e.g. Syntaxin).",
    "memory":
        "How many frames a particle may disappear (blink) and still be re-linked "
        "to the same track.",
    "min track length":
        "Discard tracks shorter than this many frames — too short to fit a "
        "reliable diffusion model.",
    "max track length":
        "Optionally cap track length (0 = off) to drop stuck or aggregated "
        "particles that linger in one spot.",
    "max lag time":
        "A lag time is the gap between two positions on a track that you "
        "compare; the MSD curve averages the squared displacement over all "
        "pairs separated by each lag. This sets the largest lag (in frames) "
        "included in that curve.",
    "n fit lags":
        "How many of the first MSD points are used to fit the diffusion "
        "coefficient — fewer points emphasise short-time (local) diffusion.",
    "alpha threshold":
        "The anomalous exponent α from MSD ∝ τ^α; these cut-offs label each "
        "track Immobile / Confined / Brownian (α≈1) / Directed (α>1).",
    "mobile-D threshold":
        "The diffusion coefficient below which a track is treated as immobile.",
    "JDD components":
        "How many populations are fit to the jump-distance distribution — often "
        "2 (a slow and a fast pool) to resolve mixed mobility.",
    "filter by D":
        "Keep only tracks whose diffusion coefficient falls within a chosen "
        "range — useful for isolating one mobility population.",
    "ROI mode":
        "How the region of interest is chosen: none, automatic (thresholded "
        "cell footprint), a manual threshold, or drawn polygons.",
    "ROI auto method":
        "The thresholding rule (Otsu, Li, …) used to auto-detect the cell "
        "footprint from the density image.",
    "ROI threshold":
        "The manual intensity cut-off separating cell from background when ROI "
        "mode is set to manual.",
    "ROI projection":
        "Which density image the ROI is computed from — the max projection or a "
        "blink-density map.",
    "background sigma":
        "The smoothing scale (σ) used to flatten the background before the "
        "automatic ROI threshold is applied.",
    "RCC drift":
        "Redundant Cross-Correlation drift correction — aligns sub-movies "
        "against each other to cancel stage/sample drift over the acquisition.",
    "drift segment":
        "How many frames per block when estimating drift — finer blocks capture "
        "faster drift but are noisier.",
    "DBSCAN EPS":
        "How close two localisations must be (in nm) to count as neighbours when "
        "clustering — larger merges nearby clusters, smaller splits them.",
    "min samples":
        "How many neighbours within EPS are needed to seed a cluster — higher "
        "gives fewer, denser clusters and more points labelled as noise.",
    "detection backend":
        "Which engine runs spot detection — the Trackpy CPU path or the "
        "GPU-accelerated Torch path (when available).",
    "chunk size":
        "How many frames are processed per batch — trades memory use against "
        "per-batch overhead.",
    # ── Time-like quantities: three DIFFERENT things, easily confused ─────────
    # These read almost identically but differ once memory-linking bridges a
    # gap, and two of them differ by exactly one frame by design.  Spell out
    # each formula so a value can never be silently mistaken for another.
    "track duration":
        "How long a trajectory lasted: (last frame − first frame) × frame "
        "interval — the elapsed time SPANNED between its first and last "
        "localisation. Gaps are included, because a molecule missed in one "
        "frame did not stop existing. For a gapless track this equals the "
        "number of intervals between localisations × frame interval.",
    "observed sampling time":
        "Localisation count × frame interval — how much time was actually "
        "SAMPLED, not how long the track lasted. It is shorter than the track "
        "duration whenever there are gaps, and (being a rescaled localisation "
        "count) it carries the same information as track length.",
    "dwell time":
        "Residence time of a confined/immobile track: (last frame − first "
        "frame + 1) × frame interval — frames OCCUPIED, so it is one frame "
        "longer than the track duration of the same track. The +1 is "
        "deliberate: a molecule seen in a single frame occupied that frame, "
        "whereas it spanned no interval.",
    "censored (dwell)":
        "A dwell time is right-censored when the molecule is still present in "
        "the movie's last frame, so its true residence time is only a lower "
        "bound. The reported τ uses the censored maximum-likelihood estimate "
        "rather than averaging truncated dwells.",
}


def glossary_def(term):
    """Return the plain-English definition for a glossary term (stats OR
    analysis), or '' if the term isn't known."""
    return STATS_GLOSSARY.get(term) or ANALYSIS_GLOSSARY.get(term, "")


def config_summary_rows(cfg):
    """Return a list of (label, value) rows describing the config — used by the
    CSV/PDF header blocks so the exact statistical settings are recorded with
    the results."""
    cfg = normalize_stats_config(cfg)
    return [
        ("Significance level (alpha)",   f"{cfg['alpha']:g}"),
        ("Parametric strategy",          cfg["parametric_strategy"]),
        ("Non-parametric test",          cfg["nonparametric_test"]),
        ("Test for 3+ groups",           cfg["anova3plus"]),
        ("Post-hoc (3+ groups)",         cfg["posthoc"]),
        ("Within-metric correction",     correction_display(cfg["correction"])),
        ("Across-metric correction",
         (f"yes ({correction_display(cfg['correction'])})"
          if cfg["across_metric_correction"] else "no")),
        ("Control group",                cfg["control_group"] or "none"),
        ("Dunnett (all-vs-control)",     "yes" if cfg["dunnett"] else "no"),
        ("Equivalence (TOST)",
         (f"yes (±{cfg['tost_margin']:g} SD)" if cfg["equivalence_tost"] else "no")),
        ("Effect-size CI level",         f"{cfg['ci_level']:g}"),
        ("Figure stars use corrected p", "yes" if cfg["figure_stars_use_corrected"] else "no"),
    ]
