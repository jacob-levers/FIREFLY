"""Dropping palmTRACER data onto an Analysis-tab condition.

Two bugs made real palmTRACER data unusable in the Analysis tab (it appeared as
an invalid / red chip):

1. ``is_run_folder`` only looked for FIREFLY's own sidecars, so a raw
   palmTRACER folder was rejected by the staging gate even though
   ``load_summary_from_folder`` reads it natively.  It therefore only worked
   *after* something else had cached it.
2. The parent-folder scan went one level deep, but a real acquisition layout is
   ``<experiment>/02_Analysis/<cell>.PT`` — two levels — so dropping the
   experiment folder found nothing.

Fixing (2) exposed a hazard worth its own test: an experiment folder can hold
the SAME cells analysed twice (palmTRACER in ``02_Analysis``, FIREFLY's own run
in ``01_Raw/batch_results``), so the scan must return ONE level, not every
level, or each cell is silently counted twice from two different pipelines.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication              # noqa: E402

from firefly.ui.controllers.workspace import workspace_data as wd   # noqa: E402
from firefly.ui.controllers.workspace.workspace_controller import (  # noqa: E402
    AnalysisWorkspaceController)
from test_workspace_data import make_run_folder          # noqa: E402

_app = QApplication.instance() or QApplication([])


def _fake_palmtracer(path):
    """Minimal RAW palmTRACER folder: the two files that identify one, and
    deliberately NO firefly_extras cache."""
    os.makedirs(path, exist_ok=True)
    for name in ("locPALMTracer.txt", "trcPALMTracer.txt"):
        with open(os.path.join(path, name), "w") as fh:
            fh.write("placeholder\n")
    assert not os.path.isdir(os.path.join(path, "firefly_extras"))
    return path


def _stage(path):
    c = AnalysisWorkspaceController(settings=None)
    cond = c._cond(c.conditions[0]["id"])
    staged, to_analyse, flagged = c._stage_paths(cond, [path])
    return c, cond, staged, flagged


# ── 1. the gate must accept raw palmTRACER, uncached ─────────────────────────
def test_uncached_palmtracer_folder_is_accepted(tmp_path):
    pt = _fake_palmtracer(str(tmp_path / "cell_D1_Pre.PT"))
    assert wd.is_run_folder(pt) is True, "raw palmTRACER rejected by the gate"


def test_dropping_a_raw_palmtracer_folder_is_not_flagged(tmp_path):
    pt = _fake_palmtracer(str(tmp_path / "cell_D1_Pre.PT"))
    _c, cond, staged, flagged = _stage(pt)
    assert flagged is False and len(staged) == 1
    assert cond.folders[0].path == pt


def test_a_plain_folder_is_still_flagged(tmp_path):
    """The gate must not become a blanket yes — an unrelated folder still fails."""
    plain = tmp_path / "holiday_photos"
    (plain / "sub").mkdir(parents=True)
    (plain / "sub" / "note.txt").write_text("x")
    _c, _cond, staged, flagged = _stage(str(plain))
    assert staged == [] and flagged is True


# ── 2. an experiment folder two levels up must resolve ───────────────────────
def test_experiment_folder_finds_runs_two_levels_down(tmp_path):
    exp = tmp_path / "20260402_experiment"
    (exp / "01_Raw").mkdir(parents=True)
    (exp / "01_Raw" / "movie.tif").write_text("x")
    analysis = exp / "02_Analysis"
    analysis.mkdir()
    for cell in ("Cip_D1_Pre", "Cip_D1_Post", "DMSO_D1_Pre"):
        _fake_palmtracer(str(analysis / f"{cell}.PT"))
    _c, cond, staged, flagged = _stage(str(exp))
    assert flagged is False, "experiment folder still flagged"
    assert len(staged) == 3
    assert all(f.path.endswith(".PT") for f in cond.folders)


# ── 3. the same cells analysed twice must not both be added ──────────────────
def test_shallowest_level_wins_so_cells_are_not_counted_twice(tmp_path):
    """palmTRACER output at depth 2 and FIREFLY's own run at depth 3 describe the
    SAME cell.  Only the shallower set may be staged."""
    exp = tmp_path / "experiment"
    analysis = exp / "02_Analysis"
    analysis.mkdir(parents=True)
    _fake_palmtracer(str(analysis / "cellA.PT"))
    batch = exp / "01_Raw" / "batch_results"
    batch.mkdir(parents=True)
    make_run_folder(str(batch), "cellA", seed=1)          # FIREFLY run, depth 3

    _c, cond, staged, _flagged = _stage(str(exp))
    assert len(staged) == 1, f"double-counted: {[f.path for f in cond.folders]}"
    assert cond.folders[0].path.endswith("cellA.PT")      # shallowest level

    # …and the deeper set is still reachable by dropping it directly.
    _c2, cond2, staged2, flagged2 = _stage(str(batch))
    assert flagged2 is False and len(staged2) == 1
    assert not cond2.folders[0].path.endswith(".PT")


def test_scan_does_not_descend_into_a_runs_own_sidecars(tmp_path):
    """firefly_extras/ + data/ inside a run must never be staged as runs."""
    root = tmp_path / "runs"
    root.mkdir()
    run = make_run_folder(str(root), "cellB", seed=2)
    _c, cond, staged, _f = _stage(str(root))
    assert len(staged) == 1 and cond.folders[0].path == run
