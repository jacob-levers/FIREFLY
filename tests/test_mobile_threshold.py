"""The mobile/immobile split.

D below the threshold is called immobile, above it mobile — and the ratio of
the two is the headline population statistic.  The value is therefore a
scientific claim, not a tuning knob: it is the published split for
single-molecule receptor tracking in neurons (Constals et al. 2015, Neuron
85:787-803, Fig. 1C, which cuts the log10(D) distribution at log10 D = -2).

Two failure modes are guarded here.  First, the number had been duplicated as a
bare ``0.05`` literal across the worker, the sidebar schema, the params builder,
the figure layer and the QML preferences field, so any one of them could drift
out of step with the constant.  Second, ``mobile_fraction`` used to be read from
summary_metrics.json — frozen at process time — while the panel drawn directly
above the stats card was recomputed live, so the two could disagree about the
same run.
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from firefly.analysis.fa_constants import MOBILE_D_THRESHOLD_DEFAULT
from firefly.analysis.fa_diffusion import (MOBILE_D_THRESHOLD_DEFAULT as _VIA_DIFFUSION,
                                           _mob_immob_ratio)

ROOT = Path(__file__).resolve().parents[1]


# ── one canonical value, reachable by both import paths ──────────────────────
def test_the_threshold_is_the_published_value():
    assert MOBILE_D_THRESHOLD_DEFAULT == 0.01
    assert abs(np.log10(MOBILE_D_THRESHOLD_DEFAULT) + 2.0) < 1e-12, (
        "the paper's split is log10 D = -2")


def test_fa_diffusion_reexports_the_same_object():
    """Callers import it from either module; they must not diverge."""
    assert _VIA_DIFFUSION is MOBILE_D_THRESHOLD_DEFAULT


def test_every_default_site_agrees_with_the_constant():
    from firefly.ui.controllers.params.params_builder import _DEFAULTS
    from firefly.ui.controllers.params import sidebar_schema as S
    assert _DEFAULTS["mobile_d"] == MOBILE_D_THRESHOLD_DEFAULT
    assert S.BY_KEY["analysis/mobile_d"]["default"] == MOBILE_D_THRESHOLD_DEFAULT


def test_no_bare_literal_survives_in_the_source():
    """The old duplicated ``0.05`` fallbacks must all be gone — one of them
    silently reintroduces the old split for any caller that omits the key."""
    offenders = []
    pat = re.compile(r'mobile_d[^\n]*?0\.05|0\.05[^\n]*?mobile_d')
    for f in ROOT.joinpath("firefly").rglob("*.py"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if pat.search(line):
                offenders.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not offenders, "stale 0.05 mobile-D literal:\n" + "\n".join(offenders)


def test_the_qml_preferences_field_defaults_to_the_constant():
    q = ROOT.joinpath("firefly/ui/qml/PreferencesDialog.qml").read_text(encoding="utf-8")
    m = re.search(r'getStr\("analysis/mobile_d",\s*"([^"]+)"\)', q)
    assert m, "PreferencesDialog no longer reads analysis/mobile_d"
    assert float(m.group(1)) == MOBILE_D_THRESHOLD_DEFAULT


def test_the_fly_preset_uses_the_published_threshold():
    from firefly.ui.controllers.params import sidebar_schema as S
    p = json.loads(ROOT.joinpath("firefly/ui/presets/Drosophila Neurons.json")
                   .read_text(encoding="utf-8"))
    p.pop("__firefly_builtin__", None)
    assert p["analysis/mobile_d"] == MOBILE_D_THRESHOLD_DEFAULT
    assert [k for k in p if k not in S.BY_KEY] == []


# ── the ratio itself ─────────────────────────────────────────────────────────
def _diff(dvals):
    import pandas as pd
    return pd.DataFrame({"D": np.asarray(dvals, float)})


def test_ratio_counts_at_the_threshold_as_mobile():
    """>= threshold is mobile, matching the figure's `Mobile ->` side."""
    r = _mob_immob_ratio(_diff([0.01, 0.01, 0.001]), d_threshold=0.01)
    assert r == pytest.approx(2.0)


def test_non_positive_and_non_finite_D_leave_both_counts():
    """An unfittable track is not evidence of either state."""
    a = _mob_immob_ratio(_diff([0.1, 0.1, 0.001]), d_threshold=0.01)
    b = _mob_immob_ratio(_diff([0.1, 0.1, 0.001, 0.0, -1.0, np.nan, np.inf]),
                         d_threshold=0.01)
    assert a == pytest.approx(b)


def test_no_immobile_tracks_gives_nan_not_a_divide_by_zero():
    assert np.isnan(_mob_immob_ratio(_diff([0.5, 0.9]), d_threshold=0.01))


# ── mobile fraction tracks the CURRENT threshold, not the stored one ──────────
def _run_with(tmp_path, d_values, stored_fraction):
    """A run whose per-track table and whose frozen summary disagree."""
    import pandas as pd
    from firefly.ui.controllers.workspace.workspace_data import RunData
    extras = tmp_path / "extras"; extras.mkdir(exist_ok=True)
    pd.DataFrame({"D": np.asarray(d_values, float),
                  "alpha": np.ones(len(d_values))}).to_csv(
        extras / "rec_diffusion_summary.csv", index=False)
    return RunData(str(tmp_path), "rec", str(extras),
                   {"mobile_fraction": stored_fraction})


def test_mobile_fraction_is_recomputed_from_per_track_D(tmp_path):
    from firefly.ui.controllers.workspace import workspace_data as wd
    # 3 of 4 tracks sit above 0.01; the stored value claims something else.
    run = _run_with(tmp_path, [0.001, 0.02, 0.05, 0.5], stored_fraction=0.25)
    wd.RunData.mobile_d = 0.01
    assert wd._mobile_pct(run) == pytest.approx(75.0), (
        "the frozen summary value was used instead of the live per-track D")


def test_changing_the_threshold_moves_the_fraction_without_reprocessing(tmp_path):
    from firefly.ui.controllers.workspace import workspace_data as wd
    run = _run_with(tmp_path, [0.001, 0.02, 0.05, 0.5], stored_fraction=0.25)
    try:
        wd.RunData.mobile_d = 0.01
        low = wd._mobile_pct(run)
        run._cache.clear()
        wd.RunData.mobile_d = 0.1
        high = wd._mobile_pct(run)
    finally:
        wd.RunData.mobile_d = MOBILE_D_THRESHOLD_DEFAULT
    assert low == pytest.approx(75.0) and high == pytest.approx(25.0)


def test_a_run_with_no_per_track_table_falls_back_to_the_stored_value(tmp_path):
    """palmTRACER caches and partial outputs must still report something."""
    from firefly.ui.controllers.workspace import workspace_data as wd
    from firefly.ui.controllers.workspace.workspace_data import RunData
    empty = tmp_path / "empty"; empty.mkdir()
    run = RunData(str(tmp_path), "rec", str(empty), {"mobile_fraction": 0.42})
    assert wd._mobile_pct(run) == pytest.approx(42.0)


# ── the figure key that explains the split ───────────────────────────────────
def _axes():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    return Figure().add_subplot(111)


def test_the_panel_labels_which_side_is_which():
    """Fig. 1C's `<- Immobile | Mobile ->` key: without it the dashed guide is
    an unexplained line."""
    from firefly.analysis.fa_compare import _annotate_logd_mobility
    ax = _axes()
    _annotate_logd_mobility(ax, -2.0, {"MUT": "#999"}, (-5.0, 1.0))
    said = [t.get_text() for t in ax.texts]
    assert any("Immobile" in s for s in said) and any("Mobile" in s for s in said)
    # the two labels straddle the threshold
    imm = next(t for t in ax.texts if "Immobile" in t.get_text())
    mob = next(t for t in ax.texts if "Immobile" not in t.get_text())
    assert imm.get_position()[0] < -2.0 < mob.get_position()[0]


def test_an_off_scale_threshold_is_not_labelled():
    """Clip the axis past the guide and the key would point at nothing."""
    from firefly.analysis.fa_compare import _annotate_logd_mobility
    ax = _axes()
    _annotate_logd_mobility(ax, -6.0, {"MUT": "#999"}, (-5.0, 1.0))
    assert not ax.texts


@pytest.mark.parametrize("style", ["overlaid", "ridgeline", "violin"])
def test_logd_styles_accept_ndarray_per_cell_medians(style):
    """The live preview passes ndarrays for the per-cell medians; ``medians or
    []`` raised on those, so ridgeline/violin silently fell back to the generic
    renderer instead of showing the panel that was asked for."""
    from firefly.analysis import fa_compare as fc
    from firefly.analysis.fa_theme import _theme_palette
    fn = {"overlaid": fc._render_logd_overlaid,
          "ridgeline": fc._render_logd_ridgeline,
          "violin": fc._render_logd_violin}[style]
    rng = np.random.default_rng(0)
    cards = [("Ctrl", "#58a6ff", rng.normal(-2, 1, 200), rng.normal(-2, 0.4, 6)),
             ("Drug", "#f78166", rng.normal(-1, 1, 200), rng.normal(-1, 0.4, 6))]
    fn(_axes(), cards, -2.0, dict(_theme_palette("Dark")),
       MOBILE_D_THRESHOLD_DEFAULT)      # must not raise
