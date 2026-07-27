"""Two-factor (group × time point) repeated-measures comparison.

The experimental design FIREFLY supports here is *paired*: the same cell is
imaged at every time point (e.g. before / after a drug).  The correct model is
a two-way MIXED ANOVA — between-subjects factor = group, within-subjects factor
= time point, subject = cell — with a Greenhouse-Geisser sphericity correction.

This module isolates the optional ``pingouin`` dependency.  If pingouin cannot
be imported, :data:`HAVE_PINGOUIN` is ``False`` and every entry point returns
``(None, message)`` so the caller (``fa_compare.compare_groups``) can degrade
gracefully — the comparison figure still renders; only the ANOVA is skipped.

Curve graphs (MSD, LogD distribution) cannot be tested as a single
group × time × lag model because pingouin's ``mixed_anova`` supports only one
within factor + one between factor.  They are therefore handled two ways:

  * at the two-factor level via their per-(cell, time point) SCALAR summary
    (MSD → ``auc_msd``; LogD → ``median_D`` / ``mob_immob_ratio``), which slot
    straight into :func:`compute_twoway_anova`; and
  * an optional per-time-point group × (lag | bin) mixed-ANOVA drill-down via
    :func:`curve_drilldown_per_timepoint`, which recovers curve-shape
    sensitivity within each time point.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from scipy import stats as _stats

try:
    import pingouin as _pg
    HAVE_PINGOUIN = True
    _PG_VERSION = getattr(_pg, "__version__", "?")
except Exception:                                   # pragma: no cover
    _pg = None
    HAVE_PINGOUIN = False
    _PG_VERSION = None

# LogD histogram bins — MUST match the logd_dist panel in fa_compare.py.
_LOGD_BINS = np.linspace(-5, 1, 31)
_LOGD_CENTERS = 0.5 * (_LOGD_BINS[:-1] + _LOGD_BINS[1:])

# Per-(cell, time point) scalar metrics the two-way ANOVA runs on.  These are
# the columns produced by fa_compare's `_row` helper.
SCALAR_METRICS = ["auc_msd", "spot_intensity", "mob_immob_ratio",
                  "median_D", "median_alpha", "radius_of_gyration",
                  "net_displacement", "path_length", "step_distance",
                  "step_speed", "directionality",
                  "track_duration", "n_localisations",
                  "mean_track_length_s", "n_tracks",
                  "nongauss_alpha2", "vacf_persistence"]


# ── small helpers ───────────────────────────────────────────────────────────
def _f(x):
    """Coerce to a finite float or None (keeps NaN/inf out of the CSV/PDF)."""
    try:
        xf = float(x)
        return xf if np.isfinite(xf) else None
    except (TypeError, ValueError):
        return None


def _stars(p):
    pf = _f(p)
    if pf is None:
        return ""
    if pf < 0.001:
        return "***"
    if pf < 0.01:
        return "**"
    if pf < 0.05:
        return "*"
    return "ns"


def _col(row, *names):
    """Fetch the first present, non-null column from a Series — pingouin uses
    hyphenated names in some tables (`p-unc`) and underscores in others
    (`p_unc`) across versions, so we try every spelling."""
    for n in names:
        if n in row.index and pd.notna(row[n]):
            return row[n]
    return None


def _within_eps(aov, within_name):
    """Greenhouse-Geisser epsilon from the within-factor row of a mixed_anova
    table (pingouin only reports the GG-corrected p for the within MAIN effect,
    not for the interaction — we reuse this epsilon to correct the interaction)."""
    try:
        wr = aov[aov["Source"] == within_name]
        if len(wr):
            return float(wr["eps"].iloc[0])
    except Exception:
        pass
    return None


# ── cell-identity derivation & pairing ──────────────────────────────────────
def derive_subject_key(stem, timepoint_tokens):
    """Strip the time-point token from a folder stem to recover cell identity.

    e.g. ``derive_subject_key('…_DMSO_D1_Post', ['Pre', 'Post'])``
         → ``('…_DMSO_D1', True)``

    Matching is case-insensitive and also consumes an adjacent separator
    (``_``, ``-`` or space).  Returns ``(cell_key, matched)`` — ``matched`` is
    False when no token was found (caller should warn / fall back to the stem).
    """
    s = (stem or "").strip()
    for tok in timepoint_tokens:
        if not tok:
            continue
        pat = re.compile(r"[_\-\s]*" + re.escape(str(tok)) + r"[_\-\s]*",
                         re.IGNORECASE)
        if pat.search(s):
            cleaned = pat.sub("_", s)
            cleaned = re.sub(r"[_\-\s]{2,}", "_", cleaned).strip("_- ")
            return (cleaned or s), True
    return s, False


def validate_pairing(df):
    """Keep only cells present exactly once at every time point within a group.

    `df` must have columns ``group``, ``timepoint``, ``cell``.  Cells missing a
    time point (or duplicated within one) are listwise-dropped.

    Returns ``(clean_df, warning_or_None, dropped)`` where ``dropped`` is a list
    of ``(group, cell, missing_timepoints)`` tuples.
    """
    timepoints = sorted(df["timepoint"].unique())
    keep_idx, dropped = [], []
    for grp, gdf in df.groupby("group"):
        for cell, cdf in gdf.groupby("cell"):
            present = set(cdf["timepoint"].unique())
            complete = present == set(timepoints) and len(cdf) == len(timepoints)
            if complete:
                keep_idx.extend(cdf.index.tolist())
            else:
                missing = sorted(set(timepoints) - present)
                dropped.append((grp, cell, missing))
    clean = df.loc[keep_idx].copy()
    warn = None
    if dropped:
        bits = [f"{g}/{c} (missing: {', '.join(m) if m else 'duplicate time point'})"
                for g, c, m in dropped]
        warn = (f"Dropped {len(dropped)} unpaired cell(s) not present once at "
                f"every time point: " + "; ".join(bits))
    return clean, warn, dropped


# ── scalar two-way mixed ANOVA ───────────────────────────────────────────────
# Map the global correction key (fa_stats_config) → pingouin's padjust token.
_PADJUST_MAP = {"none": "none", "bonferroni": "bonf", "holm": "holm",
                "fdr_bh": "fdr_bh"}


def compute_twoway_anova(df, metrics=None, stats_config=None):
    """Two-way mixed ANOVA per scalar metric.

    between = ``group`` · within = ``timepoint`` · subject = ``cell``.

    The mixed-ANOVA model (Greenhouse-Geisser sphericity correction) is fixed;
    `stats_config` only controls the post-hoc multiple-comparison method
    (mapped to pingouin's ``padjust``), so the two-way post-hoc matches the
    global correction choice.

    Returns ``(results_df, message)``.  ``results_df`` has ``section='anova'``
    rows (effect = Group / timepoint / Interaction, with F, df, p_unc, p_GG,
    np2, eps) and ``section='posthoc'`` simple-effects rows.  ``(None, msg)`` if
    pingouin is unavailable or the design is degenerate.
    """
    if not HAVE_PINGOUIN:
        return None, "pingouin not installed — two-way ANOVA skipped."
    if metrics is None:
        metrics = SCALAR_METRICS
    from firefly.analysis.fa_stats_config import normalize_stats_config
    _cfg = normalize_stats_config(stats_config)
    _padjust = _PADJUST_MAP.get(_cfg["correction"], "holm")

    groups = sorted(df["group"].unique())
    tps = sorted(df["timepoint"].unique())
    if len(groups) < 2 or len(tps) < 2:
        return None, (f"Two-way ANOVA needs ≥2 groups and ≥2 time points "
                      f"(have {len(groups)} group(s), {len(tps)} time point(s)).")

    rows = []
    for m in metrics:
        if m not in df.columns:
            continue
        sub = df[["group", "timepoint", "cell", m]].dropna().copy()
        if sub[m].nunique() < 2 or len(sub) < 4:
            continue
        # pingouin's mixed_anova requires subject IDs that are unique ACROSS the
        # between-groups factor ("Subject IDs cannot overlap between groups").
        # Two cells sharing a base name in different groups (a common folder
        # naming pattern) collide and raise ValueError — which was caught below
        # and SILENTLY dropped the whole metric from the ANOVA.  Namespace the
        # subject by group so identical cell names in different groups stay
        # distinct; the between factor is still `group`, so the model is
        # unchanged (within-group cell identities are preserved).
        sub["_subject"] = (sub["group"].astype(str) + "::"
                           + sub["cell"].astype(str))
        try:
            aov = _pg.mixed_anova(data=sub, dv=m, within="timepoint",
                                  subject="_subject", between="group",
                                  correction=True)
        except Exception as e:
            rows.append({"metric": m, "section": "anova", "effect": "ERROR",
                         "detail": f"{type(e).__name__}: {e}"})
            continue

        eps = _within_eps(aov, "timepoint")
        for _, r in aov.iterrows():
            src = r.get("Source")
            pgg = _col(r, "p_GG_corr", "p-GG-corr")
            # GG-correct the interaction p — but only when F and both dfs are
            # actually present.  A degenerate/underpowered table (e.g. 1 subject
            # per cell) omits the F column, and the bare r["F"] raised KeyError
            # here, propagating out and blanking the whole comparison figure.
            _F, _d1, _d2 = _f(r.get("F")), _f(r.get("DF1")), _f(r.get("DF2"))
            if (src == "Interaction" and eps and np.isfinite(eps)
                    and _F is not None and _d1 is not None and _d2 is not None):
                pgg = float(_stats.f.sf(_F, eps * _d1, eps * _d2))
            rows.append({
                "metric": m, "section": "anova", "effect": src,
                "SS": _f(r.get("SS")), "MS": _f(r.get("MS")),
                "F": _f(r.get("F")), "df1": _f(r.get("DF1")), "df2": _f(r.get("DF2")),
                "p_unc": _f(_col(r, "p_unc", "p-unc")), "p_GG": _f(pgg),
                "np2": _f(r.get("np2")), "eps": _f(r.get("eps")),
            })

        # Simple effects (group within time point, time within group), Holm-corrected.
        try:
            pw = _pg.pairwise_tests(data=sub, dv=m, within="timepoint",
                                    subject="_subject", between="group",
                                    padjust=_padjust)
            for _, r in pw.iterrows():
                # For the interaction ('timepoint * group') rows, pingouin puts
                # the time-point level being contrasted in the 'timepoint' col.
                at = r["timepoint"] if "timepoint" in r.index else None
                rows.append({
                    "metric": m, "section": "posthoc",
                    "contrast": r.get("Contrast"),
                    "at": at if (at is not None and pd.notna(at)) else "",
                    "level_A": r.get("A"), "level_B": r.get("B"),
                    "paired": bool(r.get("Paired")) if "Paired" in r.index else "",
                    "p": _f(_col(r, "p-unc", "p_unc")),
                    "p_holm": _f(_col(r, "p-corr", "p_corr")),
                    "stars": _stars(_col(r, "p-corr", "p_corr", "p-unc", "p_unc")),
                })
        except Exception as e:
            rows.append({"metric": m, "section": "posthoc", "contrast": "ERROR",
                         "detail": f"{type(e).__name__}: {e}"})

    if not rows:
        return None, "No metric had enough paired data for a two-way ANOVA."
    msg = (f"pingouin {_PG_VERSION} mixed_anova "
           f"(between=group, within=timepoint, subject=cell; GG-corrected; "
           f"post-hoc padjust={_padjust})")
    return pd.DataFrame(rows), msg


# ── curve-shape drill-down (per time point) ──────────────────────────────────
def _logd_hist(d_values):
    v = np.asarray(d_values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return None
    counts, _ = np.histogram(np.log10(v), bins=_LOGD_BINS)
    s = counts.sum()
    return counts / s if s else counts.astype(float)


def _build_curve_long(recs, kind):
    """Build a long (cell, group, level, val) DataFrame for one time point.

    `recs` is a list of dicts with keys group, cell, and either ``msd`` (a
    pandas Series indexed by lag_frame) or ``diffusion_D`` (1-D array).
    """
    rows = []
    if kind == "msd":
        series, common = {}, None
        for r in recs:
            s = r.get("msd")
            if s is None or len(s) == 0:
                continue
            series[r["cell"]] = (r["group"], s)
            idx = set(s.index)
            common = idx if common is None else (common & idx)
        if not series or not common:
            return None, None
        levels = sorted(common)
        for cell, (grp, s) in series.items():
            for lv in levels:
                rows.append({"cell": cell, "group": grp,
                             "level": int(lv), "val": float(s.loc[lv])})
    else:  # logd
        hists = {}
        for r in recs:
            h = _logd_hist(r.get("diffusion_D"))
            if h is not None:
                hists[r["cell"]] = (r["group"], h)
        if not hists:
            return None, None
        mat = np.vstack([h for _, (_, h) in hists.items()])
        keep = np.where(mat.sum(axis=0) > 0)[0]
        levels = [round(float(_LOGD_CENTERS[b]), 3) for b in keep]
        for cell, (grp, h) in hists.items():
            for b in keep:
                rows.append({"cell": cell, "group": grp,
                             "level": round(float(_LOGD_CENTERS[b]), 3),
                             "val": float(h[b])})
    if not rows:
        return None, None
    return pd.DataFrame(rows), levels


def curve_drilldown_per_timepoint(records, kind):
    """Per-time-point group × (lag | bin) mixed ANOVA for a curve graph.

    `records`: list of dicts {group, timepoint, cell, msd(Series) | diffusion_D}.
    `kind`: 'msd' or 'logd'.  within = lag/bin, between = group, subject = cell.

    Returns ``(results_df, message)``.
    """
    if not HAVE_PINGOUIN:
        return None, "pingouin not installed — curve drill-down skipped."
    within_label = "lag" if kind == "msd" else "bin"
    rows = []
    for tp in sorted({r["timepoint"] for r in records}):
        recs = [r for r in records if r["timepoint"] == tp]
        long, _levels = _build_curve_long(recs, kind)
        if long is None or long["cell"].nunique() < 3 or long["group"].nunique() < 2:
            continue
        try:
            aov = _pg.mixed_anova(data=long, dv="val", within="level",
                                  subject="cell", between="group", correction=True)
        except Exception as e:
            rows.append({"graph": kind, "timepoint": tp, "effect": "ERROR",
                         "detail": f"{type(e).__name__}: {e}"})
            continue
        eps = _within_eps(aov, "level")
        for _, r in aov.iterrows():
            src = r.get("Source")
            label = within_label if src == "level" else src
            pgg = _col(r, "p_GG_corr", "p-GG-corr")
            if src == "Interaction" and eps and np.isfinite(eps):
                d1, d2 = eps * float(r["DF1"]), eps * float(r["DF2"])
                pgg = float(_stats.f.sf(float(r["F"]), d1, d2))
            rows.append({
                "graph": kind, "timepoint": tp, "effect": label,
                "F": _f(r.get("F")), "df1": _f(r.get("DF1")), "df2": _f(r.get("DF2")),
                "p_unc": _f(_col(r, "p_unc", "p-unc")), "p_GG": _f(pgg),
                "np2": _f(r.get("np2")), "eps": _f(r.get("eps")),
            })
    if not rows:
        return None, f"No time point had enough paired {kind} data for a drill-down."
    return (pd.DataFrame(rows),
            f"per-time-point group×{within_label} mixed ANOVA (within={within_label}, "
            f"between=group, subject=cell; GG-corrected)")
