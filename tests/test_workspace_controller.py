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


@pytest.fixture(autouse=True)
def _dispose_controllers(monkeypatch):
    """Tear every controller down before the interpreter exits.

    Each test builds an AnalysisWorkspaceController, which owns two QTimers and
    a background loader, and nothing here ever disposed of them.  On Linux the
    process then SEGFAULTED at interpreter shutdown — after all 19 tests had
    PASSED — because Qt was destroyed with live timers still attached.  That
    crash took down the whole release-validation job, and while the suite ran in
    one process it surfaced in whichever unrelated test happened to be running.

    Stop the timers, let the loader settle, then flush the deferred deletes.
    Mirrors the qml_window fixture in test_qml_smoke.
    """
    from PySide6.QtCore import QTimer, QEvent
    made = []
    original = AnalysisWorkspaceController.__init__

    def _record(self, *args, **kwargs):
        original(self, *args, **kwargs)
        made.append(self)

    monkeypatch.setattr(AnalysisWorkspaceController, "__init__", _record)
    yield
    import time
    for ctrl in made:
        deadline = time.monotonic() + 5.0
        try:
            while ctrl.loadingFolders and time.monotonic() < deadline:
                _app.processEvents(); time.sleep(0.01)
        except Exception:
            pass
        for value in list(vars(ctrl).values()):
            if isinstance(value, QTimer):
                try: value.stop()
                except Exception: pass
        try: ctrl.deleteLater()
        except Exception: pass
    for _ in range(3):
        _app.processEvents()
        _app.sendPostedEvents(None, QEvent.Type.DeferredDelete)


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

    dr0 = c._data_rev
    c._on_figpref_changed("figures/logd_style")
    assert c._engfig_rev == rev0 + 1                  # render-rev bumped → cache key stale
    assert c._engfig_cache == {}                      # every cached panel render dropped
    assert c._data_rev == dr0                         # …but a pure style change keeps ReportData
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


def _qimg(argb):
    from PySide6.QtGui import QImage
    im = QImage(4, 4, QImage.Format.Format_RGBA8888)
    im.fill(argb)
    return im


def test_stale_bespoke_render_never_overwrites_the_live_figure(tmp_path):
    """A late-finishing bespoke render from a superseded generation must be
    dropped — otherwise it repaints the tab with the OLD graph while the state
    reads LIVE (the 'have to flick between graph types to update it' bug)."""
    c, ids = _ctrl_with_two_conditions(tmp_path)
    fresh, stale = _qimg(0xFF00FF00), _qimg(0xFFFF0000)

    gen = c._fig_gen()
    c._deliver_figure(gen, fresh)        # worker stash (generation-tagged) …
    c._on_figure_rendered()              # … GUI drain
    tok = c.figureToken
    assert c._fig_image is fresh

    # a render tagged with a prior generation arrives late → ignored entirely
    c._deliver_figure((gen[0] - 1, gen[1]), stale)
    c._on_figure_rendered()
    assert c._fig_image is fresh         # not overwritten
    assert c.figureToken == tok          # token not bumped → QML not told to reload


def test_stale_single_panel_render_is_dropped_and_leaves_cache_clean(tmp_path):
    """A stale single-panel engine render must neither repaint nor poison the
    rev-keyed cache under a since-reused key."""
    c, ids = _ctrl_with_two_conditions(tmp_path)
    fresh, stale = _qimg(0xFF00FF00), _qimg(0xFFFF0000)
    cur = c._fig_gen()
    key = (c._metric, c._engfig_rev)

    with c._fig_lock:
        c._pending_engfig.append((cur, key, fresh))
    c._on_engfig_rendered()
    assert c._engfig_cache.get(key) is fresh and c._fig_image is fresh
    tok = c.figureToken

    with c._fig_lock:                    # older generation, same panel key
        c._pending_engfig.append(((cur[0] - 1, cur[1]), key, stale))
    c._on_engfig_rendered()
    assert c._engfig_cache.get(key) is fresh    # cache not poisoned by the stale img
    assert c._fig_image is fresh and c.figureToken == tok


def test_freshest_delivery_wins_when_stale_and_fresh_arrive_together(tmp_path):
    """When a stale and a fresh render land before the drain runs, the fresh one
    (matching the current generation) is the one applied — the accumulation list
    means the stale arrival can't clobber or hide the fresh result."""
    c, ids = _ctrl_with_two_conditions(tmp_path)
    fresh, stale = _qimg(0xFF00FF00), _qimg(0xFFFF0000)
    cur = c._fig_gen()
    # stale queued first, fresh second — both pending when the handler drains
    c._deliver_figure((cur[0] - 1, cur[1]), stale)
    c._deliver_figure(cur, fresh)
    c._on_figure_rendered()
    assert c._fig_image is fresh


def test_stale_gallery_hero_render_is_dropped(tmp_path):
    """The gallery (All-panels) hero has the same guard: a panel render that
    finishes after the user switched panel/condition/replicate is ignored."""
    c, ids = _ctrl_with_two_conditions(tmp_path)
    fresh, stale = _qimg(0xFF00FF00), _qimg(0xFFFF0000)
    cur = c._panel_gen()
    c._deliver_panel(cur, fresh)
    c._on_panel_rendered()
    assert c._panel_image is fresh
    tok = c.panelToken
    # a render from a superseded selection lands late → ignored
    c._deliver_panel((cur[0], cur[1], cur[2] + 1, cur[3]), stale)
    c._on_panel_rendered()
    assert c._panel_image is fresh and c.panelToken == tok


def test_data_rev_split_style_vs_data(tmp_path):
    """The ReportData cache hinges on the rev split: a style/theme/clip/live-view
    change bumps only `_engfig_rev` (re-render), while a data / stats-config /
    mobile-D change also bumps `_data_rev` (recompute).  Guards the caching wiring."""
    c, ids = _ctrl_with_two_conditions(tmp_path)
    d0, e0 = c._data_rev, c._engfig_rev

    # pure render changes → engfig only, ReportData preserved
    for k in ("figures/theme", "figures/msd_style", "figures/logd_style",
              "analysis/dcoeff_clip_min"):
        d, e = c._data_rev, c._engfig_rev
        c._on_figpref_changed(k)
        assert c._data_rev == d, f"{k} must NOT bump _data_rev"
        assert c._engfig_rev == e + 1, f"{k} must bump _engfig_rev"

    for k in ("err", "plot", "logX", "outputStem"):
        d = c._data_rev
        c.setCfg(k, ("SD" if k == "err" else "Box" if k == "plot"
                     else False if k == "logX" else "run2"))
        assert c._data_rev == d, f"cfg '{k}' must NOT bump _data_rev"

    # data-affecting changes → both revs
    d, e = c._data_rev, c._engfig_rev
    c._on_figpref_changed("analysis/mobile_d")     # feeds mob/immob scalar
    assert c._data_rev == d + 1 and c._engfig_rev == e + 1

    d = c._data_rev
    c.setCfg("correction", "Holm")                 # stats-config → two-way + stats
    assert c._data_rev == d + 1

    d = c._data_rev
    fid = c.conditions[0]["folders"][0]["id"]
    c.toggleFolder(ids[0], fid)                     # data change
    assert c._data_rev == d + 1


def test_report_data_cache_reuses_until_data_rev_moves(tmp_path):
    """`_cached_report_data` returns the SAME ReportData while the data-rev holds,
    and recomputes once it moves — the mechanism that makes a style change a redraw."""
    c, ids = _ctrl_with_two_conditions(tmp_path)
    ck = c._compute_report_kwargs()
    rd1 = c._cached_report_data(ck, c._data_rev)
    rd2 = c._cached_report_data(ck, c._data_rev)
    assert rd1 is rd2                               # cache hit → same object
    # a data change moves the rev → next fetch recomputes a fresh ReportData
    c._changed(conditions=True)
    rd3 = c._cached_report_data(c._compute_report_kwargs(), c._data_rev)
    assert rd3 is not rd1
    # ReportData carries the pieces render_report needs
    import pandas as pd
    assert isinstance(rd3.summary_df, pd.DataFrame) and not rd3.summary_df.empty
    assert isinstance(rd3.stat_cache, dict)


def test_engine_render_lane_produces_figure_and_warms_cache(tmp_path):
    """End-to-end through the real engine lane: the async all-panels/single-panel
    render (now routed through the ReportData cache + render_report) must produce a
    live figure and leave the cache warm for the current data-rev."""
    import time
    c, ids = _ctrl_with_two_conditions(tmp_path)
    c._metric = "track_count"                 # the panel that showed the wrong slice
    c._cg = c._build_groups()                 # what _recompute sets before rendering
    c._launch_figure()                        # real entry → per-panel engine render
    deadline = time.monotonic() + 40
    while not c.hasFigure and time.monotonic() < deadline:
        _app.processEvents(); time.sleep(0.02)
    _app.processEvents()
    assert c.hasFigure, "engine render lane produced no figure"
    assert c._rd_cache is not None and c._rd_cache_rev == c._data_rev
    # the render is cached per (panel, rev) so a re-request is instant
    assert ("track_count", c._engfig_rev) in c._engfig_cache


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
