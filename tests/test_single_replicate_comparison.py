"""A condition with ONE replicate can take part in a comparison.

It used to be filtered out of the Analysis tab entirely (``_shown`` required two
active folders), so a pilot n=1 arm was invisible: no figures, no numbers, and
no hint as to why.  Everything descriptive about one replicate is perfectly
real, so it is now let in — the analysis core was already prepared for it
(``fa_circular._stat_test_n`` emits "n<2 replicates - no test possible" rows).

What must NOT happen is the tab implying a test ran.  A single replicate has no
between-animal variance, so every claim below is about refusing to state a null
result that was never measured:

  * an absent Cliff's δ must not render as "δ 0.00 · negligible"
  * the verdict must say "not tested", never "not statistically significant"
  * the methods paragraph — written to be pasted into a write-up — must not
    name a test that did not run
  * "Significant pairs 0 / 1" must not count an untested pair as a null result

The trap being guarded against is pseudoreplication: the per-track tables still
hold thousands of tracks, so a track-level test would return a tiny p-value that
measures tracks within one animal rather than a difference between conditions.
"""
import numpy as np
import pytest

from PySide6.QtWidgets import QApplication

from firefly.ui.controllers.workspace import workspace_data as wd
from firefly.ui.controllers.workspace.workspace_controller import \
    AnalysisWorkspaceController
from test_workspace_data import make_run_folder
from test_workspace_controller import _await_load


@pytest.fixture(scope="module", autouse=True)
def _app():
    yield QApplication.instance() or QApplication([])


def _ctrl(tmp_path, n_a=4, n_b=1):
    """Two conditions: `n_a` replicates against `n_b`."""
    c = AnalysisWorkspaceController(settings=None)
    ids = [x["id"] for x in c.conditions]
    for k in range(n_a):
        c.addFolders(ids[0], [make_run_folder(str(tmp_path), f"a{k}",
                                              seed=10 + k, d_centre=0.04)])
    for k in range(n_b):
        c.addFolders(ids[1], [make_run_folder(str(tmp_path), f"b{k}",
                                              seed=50 + k, d_centre=0.35)])
    _await_load(c)
    return c, ids


# ── it takes part at all ─────────────────────────────────────────────────────
def test_one_replicate_is_enough_to_appear(tmp_path):
    c, _ = _ctrl(tmp_path)
    conds = {x["name"]: x for x in c.conditions}
    single = conds["Condition 2"]
    assert single["activeFolders"] == 1
    assert single["ready"] is True, "a one-replicate condition must not be hidden"
    assert single["singleReplicate"] is True
    assert c.enough is True and c.readyCount == 2
    assert len(c.statsRows) == 2, "both conditions must reach the stats cards"


def test_its_descriptive_numbers_are_real(tmp_path):
    """The point of letting it in: the figures and pooled values still work."""
    c, _ = _ctrl(tmp_path)
    labels = {r["label"] for r in c.statsRows}
    assert {"Condition 1", "Condition 2"} <= labels
    vals = {r["label"]: r["value"] for r in c.statsRows}
    assert float(vals["Condition 2"]) > float(vals["Condition 1"])


def test_two_conditions_of_one_replicate_each_still_compare(tmp_path):
    """The n=1 vs n=1 case — the smallest thing a user can put in the tab."""
    c, _ = _ctrl(tmp_path, n_a=1, n_b=1)
    assert c.enough is True and c.readyCount == 2
    assert len(c.significanceRows) == 1
    assert c.significanceRows[0]["testable"] is False


# ── but nothing may imply a test ran ─────────────────────────────────────────
def test_an_unknown_effect_size_is_not_reported_as_negligible(tmp_path):
    """The specific bug this guards: the engine returns no Cliff's δ for an
    untestable pair, and coercing that to 0.0 made the table read
    "δ 0.00 · negligible" — a null result nobody measured."""
    c, _ = _ctrl(tmp_path)
    row = c.significanceRows[0]
    assert row["delta"] == "δ —"
    assert row["mag"] != "negligible"
    assert row["mag"] == "untested"
    assert row["sig"] is False
    assert row["stars"] == ""


def test_effect_magnitude_distinguishes_unknown_from_negligible():
    assert wd.effect_magnitude(None) == ""
    assert wd.effect_magnitude(float("nan")) == ""
    assert wd.effect_magnitude(0.0) == "negligible"


def test_the_row_says_why_there_is_no_p_value(tmp_path):
    """An em dash on its own reads as "not significant"."""
    c, _ = _ctrl(tmp_path)
    row = c.significanceRows[0]
    assert row["p"] == "p = —"
    assert "2+ replicates" in row["note"]


def test_the_verdict_says_not_tested_not_not_significant(tmp_path):
    c, _ = _ctrl(tmp_path)
    html = c.metricVerdict.get("html", "")
    assert "Not tested" in html
    assert "not statistically significant" not in html
    assert c.metricVerdict.get("severity") == "warning" or \
        c.metricVerdict.get("severity") == "warn"


def test_the_methods_paragraph_names_the_untested_condition(tmp_path):
    """This paragraph is written to be pasted into a methods section, so it must
    not cite a test that never ran on that condition."""
    c, _ = _ctrl(tmp_path)
    m = c.methods
    assert "Condition 2" in m
    assert "single replicate" in m and "not tested" in m


def test_with_every_condition_at_one_replicate_no_test_is_claimed(tmp_path):
    c, _ = _ctrl(tmp_path, n_a=1, n_b=1)
    m = c.methods
    assert "No significance test was run" in m
    assert "Mann" not in m and "t-test" not in m


def test_untestable_pairs_are_excluded_from_the_significant_pairs_count(tmp_path):
    """0 / 1 implies one pair was tested and came back null."""
    c, _ = _ctrl(tmp_path)
    card = next(h for h in c.headline if h["label"] == "Significant pairs")
    assert card["value"] == "0 / 0"
    assert "untestable" in card["unit"]


def test_a_normal_comparison_is_untouched(tmp_path):
    """Everything above is conditional on a single replicate being present — the
    ordinary path must still report its test, effect size and denominator."""
    c, _ = _ctrl(tmp_path, n_a=4, n_b=4)
    row = c.significanceRows[0]
    assert row["testable"] is True
    assert row["note"] == ""
    assert row["delta"].startswith("δ ") and row["delta"] != "δ —"
    assert row["mag"] in ("negligible", "small", "medium", "large")
    card = next(h for h in c.headline if h["label"] == "Significant pairs")
    assert card["value"].endswith("/ 1") and card["unit"] == ""
    assert "not tested" not in c.methods
    assert c.singleReplicateWarning["show"] is False


# ── the warning the user actually sees ───────────────────────────────────────
def test_the_warning_names_the_conditions_and_the_pseudoreplication_trap(tmp_path):
    c, _ = _ctrl(tmp_path)
    w = c.singleReplicateWarning
    assert w["show"] is True and w["count"] == 1
    assert w["names"] == ["Condition 2"]
    text = w["text"].lower()
    # what still works
    assert "figures" in text and "descriptive" in text
    # what does not, and the tempting wrong way round it
    assert "one animal" in text or "one replicate" in text
    assert "tracks" in text, "the pseudoreplication trap must be spelled out"
    assert "two or three replicates" in text


def test_the_warning_key_is_stable_so_it_shows_once_per_set(tmp_path):
    """Keyed on the offending set, like legacyDataWarning — the modal fires when
    the set changes, not on every recompute."""
    c, ids = _ctrl(tmp_path)
    first = c.singleReplicateWarning["key"]
    c._recompute()
    assert c.singleReplicateWarning["key"] == first
    # a second single-replicate condition changes the set → the modal reappears
    c.addFolders(ids[0], [make_run_folder(str(tmp_path), "extra",
                                          seed=99, d_centre=0.04)])
    _await_load(c)
    assert c.singleReplicateWarning["key"] == first, \
        "adding a replicate to the OTHER condition must not re-nag"


def test_the_recommendation_does_not_advise_a_test_that_cannot_run(tmp_path):
    """The "smallest group" recommendation used to say "a non-parametric test is
    safer" at any n < 3 — advice about a choice that has no effect when no test
    runs at all."""
    c, _ = _ctrl(tmp_path)
    top = c.recommendations[0]
    assert top["tone"] == "warn"
    assert "no significance test can run" in top["text"]
    assert "non-parametric test is safer" not in top["text"]


def test_two_replicates_still_gets_the_exploratory_advice(tmp_path):
    c, _ = _ctrl(tmp_path, n_a=4, n_b=2)
    assert "non-parametric test is safer" in c.recommendations[0]["text"]


def test_no_warning_when_every_condition_has_replicates(tmp_path):
    c, _ = _ctrl(tmp_path, n_a=3, n_b=2)
    w = c.singleReplicateWarning
    assert w["show"] is False and w["key"] == "" and w["names"] == []
