"""Pure formatters for a FIREFLY comparison ``{stem}_results.json`` snapshot.

Pure metric ordering, p-value / number formatting, effect-size phrasing, and
plain-language verdicts for the Analysis-tab workspace controller (the merged
Compare + Results view).  No Qt; unit-testable in isolation.
"""
from __future__ import annotations

import math

# ── Metric display order + friendly names (the 8 scalar metrics) ─────────────
METRIC_DISPLAY = [
    ("auc_msd",              "MSD area-under-curve"),
    ("mob_immob_ratio",      "Mobile / immobile ratio"),
    ("median_D",             "Median D"),
    ("median_alpha",         "Median α (mobile tracks)"),
    ("mean_track_duration_s", "Mean elapsed track duration"),
    ("mean_observed_time_s", "Mean observed sampling time"),
    ("mean_track_length_s",  "Mean observed time (legacy field)"),
    ("n_tracks",             "Track count"),
    ("nongauss_alpha2",      "Population heterogeneity (α₂)"),
    ("vacf_persistence",     "Directional persistence (VACF)"),
]
SUMMARY_COLS = [
    ("group", "Group"), ("timepoint", "Time"), ("cell", "Cell"),
    ("n_tracks", "# tracks"), ("auc_msd", "AUC MSD"),
    ("mob_immob_ratio", "Mob/Immob"), ("median_D", "Median D"),
    ("median_alpha", "Median α"),
    ("mean_track_duration_s", "Duration (s)"),
    ("mean_observed_time_s", "Observed (s)"),
    ("n_below_resolution", "Below res."),
    ("n_diffusion_eligible", "D eligible"),
    ("metric_contract", "Metric contract"),
    ("nongauss_alpha2", "α₂"), ("vacf_persistence", "VACF"),
]
SUMMARY_NUMERIC = {"n_tracks", "auc_msd", "mob_immob_ratio", "median_D",
                   "median_alpha", "mean_track_duration_s",
                   "mean_observed_time_s", "mean_track_length_s",
                   "n_below_resolution", "n_diffusion_eligible",
                   "nongauss_alpha2", "vacf_persistence"}


def _isfinite(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _fmt_p(p):
    if not _isfinite(p):
        return "—"
    p = float(p)
    return f"{p:.2e}" if p < 1e-3 else f"{p:.3f}"


def _fmt_num(v, fmt="{:.4g}"):
    if v is None:
        return "—"
    if _isfinite(v):
        return fmt.format(float(v))
    return str(v)


def _pretty_metric(key):
    if key.startswith("motion_frac_"):
        return f"{key[len('motion_frac_'):]} fraction"
    return key.replace("_", " ")


def _mag_bucket(x, thresholds):
    a = abs(x)
    names = ["negligible", "small", "medium", "large"]
    for t, n in zip(thresholds, names[:-1]):
        if a < t:
            return n
    return names[-1]


def _effect_phrase(pw):
    """(magnitude_word, text) e.g. ('large', 'g = 1.20 [0.40, 2.00]')."""
    g = pw.get("hedges_g")
    if _isfinite(g):
        lo, hi = pw.get("hedges_g_ci_low"), pw.get("hedges_g_ci_high")
        ci = (f" [{float(lo):.2f}, {float(hi):.2f}]"
              if (_isfinite(lo) and _isfinite(hi)) else "")
        return _mag_bucket(float(g), [0.2, 0.5, 0.8]), f"g = {float(g):.2f}{ci}"
    d = pw.get("cliffs_delta")
    if _isfinite(d):
        lo, hi = pw.get("cliffs_delta_ci_low"), pw.get("cliffs_delta_ci_high")
        ci = (f" [{float(lo):.2f}, {float(hi):.2f}]"
              if (_isfinite(lo) and _isfinite(hi)) else "")
        return _mag_bucket(float(d), [0.147, 0.33, 0.474]), f"δ = {float(d):.2f}{ci}"
    return "", ""


def _verdict_for_metric(disp, rec, n_groups):
    """Plain-language one-liner → (severity, html, underpowered)."""
    omn = rec.get("omnibus") or {}
    pairs = rec.get("pairwise") or []
    notes = [str(omn.get("note") or "")] + [str(p.get("note") or "") for p in pairs]
    # A pair the engine could not test at all.  This must be caught BEFORE the
    # two-group branch below, which would otherwise render a missing p-value as
    # "not statistically significant" — asserting a null result that was never
    # measured.
    if any("no test possible" in nt for nt in notes):
        return ("warn",
                "<b>Not tested.</b> A condition has only one replicate, so there "
                "is no between-animal variation for a test to work against. The "
                "difference below is descriptive only.", True)
    if any("n<3" in nt for nt in notes):
        return ("warn",
                "<b>Not interpretable.</b> At least one group has fewer than 3 "
                "replicates — p-values are shown for reference only.", True)
    if n_groups <= 2 and pairs:
        pw = pairs[0]
        a, b = pw.get("label_i", "A"), pw.get("label_j", "B")
        mi, mj = pw.get("mean_i"), pw.get("mean_j")
        if _isfinite(mi) and _isfinite(mj):
            direction = ("higher than" if float(mi) > float(mj)
                         else "lower than" if float(mi) < float(mj) else "≈")
        else:
            direction = "differs from"
        p = pw.get("p_within", pw.get("p"))
        stars = pw.get("stars_within") or pw.get("stars") or ""
        sig = bool(stars) and stars != "ns"
        mag, eff = _effect_phrase(pw)
        eff_txt = f" — a <b>{mag}</b> difference ({eff})" if eff else ""
        sig_txt = (f"<b>statistically significant</b> (p = {_fmt_p(p)})" if sig
                   else f"not statistically significant (p = {_fmt_p(p)})")
        return ("success" if sig else "info",
                f"<b>{a}</b> is {direction} <b>{b}</b>{eff_txt}; {sig_txt}.",
                False)
    # 3+ groups → omnibus verdict
    test = omn.get("test", "omnibus test")
    p = omn.get("p")
    stars = omn.get("stars") or ""
    sig = bool(stars) and stars != "ns"
    es, kind = omn.get("effect_size"), omn.get("effect_size_kind")
    es_txt = ""
    if _isfinite(es):
        es_txt = f" ({'η²' if kind == 'eta_sq' else 'ε²'} = {float(es):.3f})"
    head = ("a significant overall difference" if sig
            else "no significant overall difference")
    return ("success" if sig else "info",
            f"There is {head} across {n_groups} groups "
            f"({test}, p = {_fmt_p(p)}){es_txt}. See details for the pairwise "
            f"comparisons.", False)


def ordered_metrics(stats):
    """[(key, display)] — the 8 known metrics present (fixed order), then any
    extra ``stats`` keys with pretty names appended."""
    stats = stats or {}
    out = [(k, d) for k, d in METRIC_DISPLAY if k in stats]
    seen = {k for k, _ in METRIC_DISPLAY}
    for k in stats:
        if k not in seen:
            out.append((k, _pretty_metric(k)))
    return out
