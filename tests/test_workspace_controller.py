"""Headless tests for AnalysisWorkspaceController (the merged Analysis tab).

Drives the controller the way QML would — add folders to conditions, switch
metric, toggle folders, set settings, go paired — and checks the live numbers it
exposes.  The matplotlib figure lane is covered separately (it's async); here we
assert the synchronous readout.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
# A QApplication (not bare QGuiApplication) so this file coexists with the
# widget-based QML tests in one pytest process — only one app instance is
# allowed, and QMainWindow/QQuickWidget need the QtWidgets app.
from PySide6.QtWidgets import QApplication

from firefly.ui.controllers.workspace.workspace_controller import AnalysisWorkspaceController
from test_workspace_data import make_run_folder

_app = QApplication.instance() or QApplication([])


def _await_load(c, timeout=10.0):
    """Folder loading is async (off the GUI thread); pump the event loop until
    every dropped folder has resolved so the synchronous readout is settled."""
    import time
    deadline = time.monotonic() + timeout
    while c.loadingFolders and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.01)
    _app.processEvents()


def _ctrl_with_two_conditions(tmp_path, d_lo=0.04, d_hi=0.35, nf=4):
    c = AnalysisWorkspaceController(settings=None)
    ids = [cond["id"] for cond in c.conditions]
    assert len(ids) == 2
    for k in range(nf):
        c.addFolders(ids[0], [make_run_folder(str(tmp_path), f"lo{k}", seed=10 + k, d_centre=d_lo)])
        c.addFolders(ids[1], [make_run_folder(str(tmp_path), f"hi{k}", seed=50 + k, d_centre=d_hi)])
    _await_load(c)
    return c, ids


def test_add_folders_loads_off_thread_without_blocking(tmp_path):
    # Adding a run folder returns immediately with a *loading* placeholder chip;
    # the run's sidecars are read on a background thread, so the GUI never
    # blocks.  Pumping the event loop resolves the chip to real run data.
    c = AnalysisWorkspaceController(settings=None)
    cid = c.conditions[0]["id"]
    run = make_run_folder(str(tmp_path), "async0", seed=3, d_centre=0.1)
    c.addFolders(cid, [run])
    # synchronous readout right after the call: one chip, still loading
    fol = c.conditions[0]["folders"]
    assert len(fol) == 1
    assert fol[0]["loading"] is True
    assert fol[0]["qc"] == "loading"
    assert c.loadingFolders is True
    # nothing is "active" until it resolves (active() filters run is None)
    assert c.conditions[0]["activeFolders"] == 0
    _await_load(c)
    fol = c.conditions[0]["folders"]
    assert fol[0]["loading"] is False
    assert fol[0]["qc"] in ("ok", "warn")
    assert c.loadingFolders is False
    assert c.conditions[0]["activeFolders"] == 1


def test_live_numbers_appear_with_two_ready_conditions(tmp_path):
    c, ids = _ctrl_with_two_conditions(tmp_path)
    assert c.enough is True
    assert c.readyCount == 2
    assert len(c.headline) == 6
    assert c.headline[0]["label"] == "Total tracks"
    assert len(c.statsRows) == 2
    assert len(c.significanceRows) == 1          # one pair
    assert c.methods.startswith("Diffusion D was compared across 2 conditions")
    # clearly separated groups → significant with a large effect
    assert c.significanceRows[0]["mag"] == "large"


def test_metric_switch_rescopes_everything(tmp_path):
    c, ids = _ctrl_with_two_conditions(tmp_path)
    c.setMetric("mob_immob")                      # scroller selects export PANEL keys
    assert c.metric == "mob_immob"
    assert "Mobile fraction" in c.methods
    assert c.statsRows[0]["err"] == ""           # ± only shown for D


def test_folder_exclude_drops_below_ready(tmp_path):
    c, ids = _ctrl_with_two_conditions(tmp_path, nf=2)
    assert c.enough is True
    # exclude one of condition-0's two folders → only 1 active → not ready
    fid = c.conditions[0]["folders"][0]["id"]
    c.toggleFolder(ids[0], fid)
    assert c.readyCount == 1
    assert c.enough is False
    assert c.headline == []


def test_failed_qc_folder_excluded_by_default(tmp_path):
    c = AnalysisWorkspaceController(settings=None)
    cid = c.conditions[0]["id"]
    # a run with zero tracks → qc 'error' → excluded on add
    bad = make_run_folder(str(tmp_path), "empty", seed=1, n_tracks=0)
    c.addFolders(cid, [bad])
    _await_load(c)
    fol = c.conditions[0]["folders"][0]
    assert fol["qc"] == "error"
    assert fol["excluded"] is True


def test_paired_by_timepoint(tmp_path):
    c = AnalysisWorkspaceController(settings=None)
    ids = [cond["id"] for cond in c.conditions]
    # two conditions sharing a name, different timepoints = one subject, two times
    c.setConditionName(ids[0], "Propofol")
    c.setConditionName(ids[1], "Propofol")
    c.setConditionPhase(ids[0], "Pre-drug")
    c.setConditionPhase(ids[1], "Post-drug")
    for k in range(3):
        c.addFolders(ids[0], [make_run_folder(str(tmp_path), f"pre{k}", seed=k, d_centre=0.2)])
        c.addFolders(ids[1], [make_run_folder(str(tmp_path), f"post{k}", seed=20 + k, d_centre=0.1)])
    _await_load(c)
    c.setCfg("groupBy", "Timepoint (pre/post)")
    assert c.paired is True
    assert c.pairedAxis == ["Pre-drug", "Post-drug"]
    series = c.pairedSeries
    assert len(series) == 1 and series[0]["subject"] == "Propofol"
    assert len(series[0]["points"]) == 2


def test_panels_view_state(tmp_path):
    c, ids = _ctrl_with_two_conditions(tmp_path)
    c.setView("panels")
    assert c.view == "panels"
    assert len(c.panelConditions) == 2
    cats = c.panelCategories
    assert [g["cat"] for g in cats] == ["Imaging", "Tracking", "Diffusion", "Population"]
    assert sum(g["count"] for g in cats) == c.panelCount == 17
    c.setPanelSel(0)
    assert c.panelSel == 0 and c.panelHeroCat == "Imaging"
    assert c.panelIndexLabel == "01 / 17"


def test_export_stats_writes_csv(tmp_path):
    c, ids = _ctrl_with_two_conditions(tmp_path)
    # export lands next to the first folder's parent
    c.exportStats()
    import glob
    hits = glob.glob(str(tmp_path / "**" / "firefly_comparison_stats.csv"), recursive=True) \
        + glob.glob(str(tmp_path / "firefly_comparison_stats.csv"))
    assert hits, "stats CSV should be written"


def test_recommend_and_presets_roundtrip(tmp_path):
    c, ids = _ctrl_with_two_conditions(tmp_path)
    c.applyRecommended()
    assert c.cfg["test"] in ("Mann–Whitney U", "Kruskal–Wallis")
    c.setMetric("radial_dist")                   # panel keys, not scalar metric ids
    c.savePreset()
    assert len(c.presets) == 1
    c.setMetric("logd_dist")
    c.loadPreset(c.presets[0]["name"])
    assert c.metric == "radial_dist"


def test_figpref_change_invalidates_figure_caches(tmp_path):
    """A Figures preference change (e.g. the log-D graph style) must invalidate
    the cached renders + bump the revs, so the Analysis tab redraws instead of
    serving a stale panel — the bug where the style never reached the tab."""
    c, ids = _ctrl_with_two_conditions(tmp_path)     # settings=None
    rev0, prev0 = c._engfig_rev, c._panel_rev
    c._engfig_cache[("logd_dist", rev0)] = object()  # seed a stale cached render
    seen = []
    c.panelRevChanged.connect(lambda: seen.append(1))

    c._on_figpref_changed("figures/logd_style")
    assert c._engfig_rev == rev0 + 1                  # data-rev bumped → cache key stale
    assert c._engfig_cache == {} and c._slices == {}  # every cached render dropped
    assert c._panel_rev == prev0 + 1 and seen         # thumbnails told to re-request

    # unrelated keys and the self-managed panel-selection key are ignored (no loop)
    r = c._engfig_rev
    c._on_figpref_changed("updates/channel")
    c._on_figpref_changed("figures/compare_panels")
    assert c._engfig_rev == r


def test_settings_change_signal_is_wired_to_figure_invalidation(tmp_path):
    """Writing figures/logd_style through the settings object must reach the
    workspace controller (the missing connection was the root cause)."""
    from PySide6.QtCore import QObject, Signal

    class FakeSettings(QObject):
        changed = Signal(str)
        def __init__(self): super().__init__(); self._d = {}
        def get(self, k, d=None): return self._d.get(k, d)
        def getStr(self, k, d=""): return str(self._d.get(k, d))
        def setValue(self, k, v): self._d[str(k)] = v; self.changed.emit(str(k))
        def sync(self): pass

    c = AnalysisWorkspaceController(settings=FakeSettings())
    rev0 = c._engfig_rev
    c._settings.setValue("figures/logd_style", "ridgeline")
    assert c._engfig_rev == rev0 + 1                  # signal → handler → invalidation


def test_report_progress_drain():
    """compare_groups' progress_cb writes (done,total,msg) off-thread; the GUI
    drain turns it into a determinate bar during loading and an indeterminate one
    while the engine renders the rest."""
    c = AnalysisWorkspaceController(settings=None)
    # mid-load → determinate fraction + the per-folder message
    c._report_prog_raw = (3, 12, "Loading: cellA")
    c._drain_report_progress()
    assert abs(c.reportProgress - 0.25) < 1e-6
    assert "Loading: cellA" in c.reportStatus and "(3/12)" in c.reportStatus
    # everything loaded (done == total) → indeterminate while it renders + writes
    c._report_prog_raw = (12, 12, "Computing scalars and rendering...")
    c._drain_report_progress()
    assert c.reportProgress == -1.0
    assert "Computing" in c.reportStatus
