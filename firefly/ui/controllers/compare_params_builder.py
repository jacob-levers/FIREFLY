"""Pure builder for the worker's ``comparison_params`` dict (Phase 5b).

Reproduces ``_start_compare_run``'s params block byte-identically from the QML
Compare groups + persisted QSettings, so ``firefly_worker.run_comparison`` needs
no changes.  The QML Compare tab doesn't expose the statistics-config editors
(those land in Phase-6 Preferences), so ``stats_config`` is sourced from the same
``stats/*`` QSettings keys the Widgets ``_collect_stats_config`` writes, applying
the same combo label→value maps.  No Qt — unit-testable with a fake settings.
"""
from __future__ import annotations

# combo label → value maps (captured from the Widgets stat combos)
CORR_MAP = {"None": "none", "Bonferroni": "bonferroni", "Holm": "holm",
            "Benjamini-Hochberg (FDR)": "fdr_bh", "Šidák": "sidak", "Hochberg": "hochberg"}
STRAT_MAP = {"Auto (normality test)": "auto", "Force parametric": "force_parametric",
             "Force non-parametric": "force_nonparametric"}
ANOVA_MAP = {"Welch's ANOVA": "welch", "One-way ANOVA": "oneway", "Auto": "auto"}
NONPARAM_MAP = {"Mann-Whitney U": "mann_whitney", "Brunner-Munzel": "brunner_munzel",
                "Permutation": "permutation"}
POSTHOC_MAP = {"Auto (pairwise)": "auto", "Games-Howell": "games_howell",
               "Dunn": "dunn", "Tukey HSD": "tukey"}

# pristine stats_config (captured from a freshly-built Widgets app)
STATS_DEFAULTS = {
    "alpha": 0.05, "correction": "holm", "across_metric_correction": False,
    "parametric_strategy": "auto", "anova3plus": "welch",
    "nonparametric_test": "mann_whitney", "posthoc": "auto", "control_group": "",
    "dunnett": False, "equivalence_tost": False, "tost_margin": 0.5,
    "ci_level": 0.95, "figure_stars_use_corrected": True,
    "include_circular_outputs": True, "circ_test_kappa": True,
    "circ_test_rbar": True, "circ_test_mu": True, "circ_test_circlin": True,
}

# all comparison-figure panel keys (default = all selected)
COMPARE_PANEL_KEYS = ["msd", "auc", "logd_dist", "mob_immob", "motion_classes",
                      "track_length", "track_count", "jdd", "dwell_cdf",
                      "turning_angles", "radial_dist", "van_hove", "vacf"]


def build_stats_config(settings) -> dict:
    g = settings
    return {
        "alpha": g.get_float("stats/alpha", STATS_DEFAULTS["alpha"]),
        "correction": CORR_MAP.get(g.get_str("stats/correction", "Holm"), "holm"),
        "across_metric_correction": g.get_bool("stats/across_metric", False),
        "parametric_strategy": STRAT_MAP.get(
            g.get_str("stats/strategy", "Auto (normality test)"), "auto"),
        "anova3plus": ANOVA_MAP.get(g.get_str("stats/anova3plus", "Welch's ANOVA"), "welch"),
        "nonparametric_test": NONPARAM_MAP.get(
            g.get_str("stats/nonparam", "Mann-Whitney U"), "mann_whitney"),
        "posthoc": POSTHOC_MAP.get(g.get_str("stats/posthoc", "Auto (pairwise)"), "auto"),
        "control_group": ("" if g.get_str("stats/control_group", "(none)") == "(none)"
                          else g.get_str("stats/control_group", "")),
        "dunnett": g.get_bool("stats/dunnett", False),
        "equivalence_tost": g.get_bool("stats/tost", False),
        "tost_margin": g.get_float("stats/tost_margin", STATS_DEFAULTS["tost_margin"]),
        "ci_level": g.get_float("stats/ci_level", STATS_DEFAULTS["ci_level"]),
        "figure_stars_use_corrected": g.get_bool("stats/figure_stars_corrected", True),
        "include_circular_outputs": g.get_bool("stats/include_circular", True),
        "circ_test_kappa": g.get_bool("stats/circ_kappa", True),
        "circ_test_rbar": g.get_bool("stats/circ_rbar", True),
        "circ_test_mu": g.get_bool("stats/circ_mu", True),
        "circ_test_circlin": g.get_bool("stats/circ_circlin", True),
    }


def selected_panels(settings) -> list:
    """Panel keys whose ``compare/panel_<key>`` is set (default all selected)."""
    return [k for k in COMPARE_PANEL_KEYS
            if settings.get_bool(f"compare/panel_{k}", True)]


def build_comparison_params(settings, groups, output_dir, output_stem="comparison") -> dict:
    """The worker params dict.  ``groups`` is the non-empty groups list (>=2,
    each ``{label, color, timepoint, folders}``)."""
    g = settings
    return {
        "groups": list(groups),
        "output_dir": output_dir,
        "output_stem": (output_stem or "comparison").strip() or "comparison",
        "theme": g.get_str("compare/theme", "Dark") or "Dark",
        "pdf_report": g.get_bool("compare/pdf_report", True),
        "panels": selected_panels(g),
        "mobile_d_threshold": g.get_float("analysis/mobile_d", 0.05),
        "logd_plot_style": g.get_str("figures/logd_style", "overlaid") or "overlaid",
        "stats_config": build_stats_config(g),
        "use_native": False,
    }
