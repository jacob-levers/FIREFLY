"""Parse the acquisition naming convention and sort folders into conditions.

The convention in use is::

    <parent>/N=2 MB543B-Sx1A-mEos3.2_CrimsonVenus 31July[ <Drug>]/
        Fly-1-16k Frames-LSide.czi

so the DRUG is whatever trails the date in an ANCESTOR folder (a bare date means
control), while animal and side come from the recording's own name.  Searching
ancestors is the whole point: a FIREFLY run folder is named from the recording
stem ("Fly-1-16k Frames-LSide"), which never carries the condition.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication                  # noqa: E402

from firefly.ui.controllers.workspace import workspace_data as wd   # noqa: E402
from firefly.ui.controllers.workspace.workspace_controller import (  # noqa: E402
    AnalysisWorkspaceController)
from test_workspace_data import make_run_folder              # noqa: E402

_app = QApplication.instance() or QApplication([])

_ROOT = "/data/N=2 MB543B-Sx1A-mEos3.2_CrimsonVenus"


# ── parsing ──────────────────────────────────────────────────────────────────
def test_bare_date_is_control():
    r = wd.parse_experiment_path(f"{_ROOT} 31July/Fly-1-16k Frames-LSide.czi")
    assert r["matched"] and r["condition"] == wd.CONTROL_LABEL
    assert r["genotype"] == "MB543B"
    assert r["animal"] == "Fly-1" and r["side"] == "LSide"


def test_token_after_the_date_is_the_drug():
    r = wd.parse_experiment_path(f"{_ROOT} 31July Propofol/Fly-3-16k Frames-RSide.czi")
    assert r["condition"] == "Propofol"
    assert r["animal"] == "Fly-3" and r["side"] == "RSide"


def test_hyphenated_drug_and_other_genotype():
    r = wd.parse_experiment_path(
        "/d/N=5 MB112C-Sx1A-mEos3.2_CrimsonVenus 2Aug 1-AMA/Fly-12-16k Frames-LSide.czi")
    assert r["condition"] == "1-AMA" and r["genotype"] == "MB112C"
    assert r["animal"] == "Fly-12"


def test_condition_is_found_from_an_ANCESTOR_of_the_run_folder():
    """The run folder itself only has animal+side — the drug is further up."""
    r = wd.parse_experiment_path(
        f"{_ROOT} 31July Propofol/01_Raw/batch_results/Fly-7-16k Frames-LSide")
    assert r["condition"] == "Propofol" and r["animal"] == "Fly-7"


def test_paths_outside_the_convention_are_not_guessed():
    r = wd.parse_experiment_path("/somewhere/random folder/thing.czi")
    assert r["matched"] is False and r["condition"] == ""


# ── sorting ──────────────────────────────────────────────────────────────────
def _runs(tmp_path, day, stems):
    base = os.path.join(str(tmp_path),
                        f"N=2 MB543B-Sx1A-mEos3.2_CrimsonVenus {day}",
                        "batch_results")
    return [make_run_folder(base, s, seed=i) for i, s in enumerate(stems)]


def _loaded(folders):
    import time
    c = AnalysisWorkspaceController(settings=None)
    c.addFolders(c.conditions[0]["id"], folders)
    deadline = time.monotonic() + 30
    while c.loadingFolders and time.monotonic() < deadline:
        _app.processEvents(); time.sleep(0.01)
    _app.processEvents()
    return c


def test_folders_are_grouped_by_drug_with_control_first(tmp_path):
    folders = (_runs(tmp_path, "31July", ["Fly-1-16k Frames-LSide",
                                          "Fly-2-16k Frames-LSide"])
               + _runs(tmp_path, "31July Propofol", ["Fly-3-16k Frames-RSide"])
               + _runs(tmp_path, "2Aug 1-AMA", ["Fly-4-16k Frames-LSide"]))
    c = _loaded(folders)
    assert c.autoSortPreview["canSort"] is True
    c.autoSortByName(); _app.processEvents()

    names = [x["name"] for x in c.conditions]
    assert names[0] == wd.CONTROL_LABEL, "control must lead the legend"
    assert set(names) == {wd.CONTROL_LABEL, "Propofol", "1-AMA"}
    by = {x["name"]: x["totalFolders"] for x in c.conditions}
    assert by == {wd.CONTROL_LABEL: 2, "Propofol": 1, "1-AMA": 1}


def test_sorting_does_not_reload_the_runs(tmp_path):
    """Reorganising ~50 folders must not re-read them — that would be a long
    stall for what is purely a regrouping."""
    folders = (_runs(tmp_path, "31July", ["Fly-1-16k Frames-LSide"])
               + _runs(tmp_path, "31July Propofol", ["Fly-2-16k Frames-RSide"]))
    c = _loaded(folders)
    before = [f["id"] for x in c.conditions for f in x["folders"]]
    c.autoSortByName(); _app.processEvents()
    after = [f["id"] for x in c.conditions for f in x["folders"]]
    assert sorted(before) == sorted(after)
    assert all(not f["loading"] for x in c.conditions for f in x["folders"])
    assert sum(x["activeFolders"] for x in c.conditions) == 2


def test_unconvention_folders_are_kept_not_dropped(tmp_path):
    odd = make_run_folder(str(tmp_path / "random place"), "mystery", seed=9)
    folders = (_runs(tmp_path, "31July", ["Fly-1-16k Frames-LSide"])
               + _runs(tmp_path, "31July Propofol", ["Fly-2-16k Frames-RSide"])
               + [odd])
    c = _loaded(folders)
    assert c.autoSortPreview["unmatched"] == 1
    c.autoSortByName(); _app.processEvents()
    paths = [f["path"] for x in c.conditions for f in x["folders"]]
    assert odd in paths, "an unparsed folder must not silently disappear"


def test_no_sort_offered_when_only_one_condition_is_present(tmp_path):
    c = _loaded(_runs(tmp_path, "31July", ["Fly-1-16k Frames-LSide",
                                           "Fly-2-16k Frames-LSide"]))
    assert c.autoSortPreview["canSort"] is False
    before = [x["name"] for x in c.conditions]
    c.autoSortByName()                       # must be a no-op
    assert [x["name"] for x in c.conditions] == before
