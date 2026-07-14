"""A serial (non-HYPER-FLY) batch mirrors the Process cockpit.

Covers the wiring added so a regular batch moves to the Process screen with its
live preview + in-depth console, plus the compact queue (Batch.runQueue):
  * AnalysisController's external-run API (begin/feed*/end + Stop routing), and
  * BatchController driving that cockpit through its message forwarders.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

import numpy as np                                       # noqa: E402
import pytest                                            # noqa: E402

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                            # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class FakeSettings:
    def get_str(self, k, d=""): return d
    def get_float(self, k, d=0.0): return d
    def get_bool(self, k, d=False): return d
    def set(self, k, v): pass
    def sync(self): pass


class FakeImport:
    hasFile = False
    filePath = ""; outDir = ""; isCsv = False


def _frame_payload(h=2, w=2):
    return {"shape": [h, w],
            "frame": np.zeros((h, w), dtype=np.float32).tobytes(),
            "xs": [0], "ys": [0]}


# ── AnalysisController external-run API ──────────────────────────────────────
@pytest.fixture
def cockpit(monkeypatch):
    """A real AnalysisController with resource sampling stubbed out — the live
    ioreg/nvidia-smi probe spawns daemon threads, and leaking several of those
    across the shared QApplication destabilises the later QML-shell tests."""
    from firefly.ui.controllers.analysis_controller import AnalysisController
    monkeypatch.setattr(AnalysisController, "_sample_resources", lambda self: None)
    c = AnalysisController(FakeSettings(), FakeImport())
    yield c
    for t in (c._res_timer, c._elapsed_timer, c._poll_timer):
        t.stop()
    c.deleteLater()
    _app.processEvents()


def test_external_run_lifecycle_drives_cockpit_state(cockpit):
    c = cockpit
    assert not c.running and not c._external

    assert c.beginExternalRun() is True
    assert c.running and c._external

    # a second begin while already running is a no-op (guards a live single run)
    assert c.beginExternalRun() is False

    logs = []
    c.logLine.connect(logs.append)
    c.feedLog("hello from the batch")
    assert logs == ["hello from the batch"]

    # a preview frame lights up the live-detection view
    tok0 = c.frameToken
    c.feedPreview(_frame_payload())
    assert c.hasLiveFrame and c.frameToken == tok0 + 1

    # progress advances the bar; a recognised stage message advances the stepper
    c.feedProgress(50, "Linking trajectories")
    assert c.progress == 50 and c.stage >= 0

    # a new file rewinds the per-file stepper
    c.resetStepper()
    assert c.stage == -1 and not c.complete

    c.endExternalRun("ok", "Batch complete — 3 ok", "/tmp/out",
                     {"n_tracks": 42, "n_locs": 1000})
    assert not c.running and not c._external
    assert c.resultSeverity == "ok"
    assert c.resultHeadline == "Batch complete — 3 ok"
    assert c.stats.get("n_tracks") == 42
    assert c.complete and c.progress == 100


def test_external_stop_is_routed_not_self_handled(cockpit):
    c = cockpit
    c.beginExternalRun()
    fired = []
    c.externalStopRequested.connect(lambda: fired.append(True))
    c.stop()                     # must NOT touch our (absent) subprocess
    assert fired == [True]
    # still marked running — the owner (BatchController) ends it
    assert c.running


def test_feeds_are_ignored_when_not_in_external_mode(cockpit):
    c = cockpit
    logs = []
    c.logLine.connect(logs.append)
    c.feedLog("ignored"); c.feedPreview(_frame_payload()); c.feedProgress(10)
    assert logs == [] and not c.hasLiveFrame and c.progress == 0


# ── BatchController driving the cockpit ──────────────────────────────────────
class StubCockpit:
    def __init__(self): self.calls = []; self._running = False
    def beginExternalRun(self):
        self._running = True; self.calls.append(("begin",)); return True
    def feedLog(self, line): self.calls.append(("log", line))
    def feedMass(self, vals): self.calls.append(("mass", list(vals)))
    def feedProgress(self, pct, msg=""): self.calls.append(("prog", pct, msg))
    def feedPreview(self, payload): self.calls.append(("preview",))
    def resetStepper(self): self.calls.append(("reset",))
    def endExternalRun(self, sev="ok", headline="", out_dir="", stats=None):
        self._running = False
        self.calls.append(("end", sev, headline, dict(stats or {})))


def _mk_batch(stub):
    from firefly.ui.controllers.batch_controller import BatchController
    return BatchController(FakeSettings(), FakeImport(), cockpit=stub)


def _kinds(stub): return [c[0] for c in stub.calls]


def test_batch_forwarders_feed_cockpit_when_driving():
    stub = StubCockpit()
    c = _mk_batch(stub)
    # simulate an in-flight serial run
    c._driving_cockpit = True
    c._series = [{"key": "s1", "name": "Cell 1"}, {"key": "s2", "name": "Cell 2"}]
    c._run_order = ["s1", "s2"]
    c._series_status = {"s1": "running", "s2": "queued"}
    c._cur_index = 0
    c._files_total = 2
    c._files_done = 0

    c._on_log("── [1/2] Cell 1 ──")
    c._on_mass([1.0, 2.0, 3.0])
    c._on_progress((40, "Detecting spots"))          # real stage → forwarded
    c._on_progress((80, "[2/2] Cell 2"))             # batch marker → skipped
    c._on_file_starting({"index": 2, "total": 2, "file": "Cell 2"})
    c._on_hf_preview(_frame_payload())

    assert ("log", "── [1/2] Cell 1 ──") in stub.calls
    assert ("mass", [1.0, 2.0, 3.0]) in stub.calls
    assert "preview" in _kinds(stub) and "reset" in _kinds(stub)
    progs = [c for c in stub.calls if c[0] == "prog"]
    assert len(progs) == 1                            # marker was skipped
    # overall = completed files + current fraction → smooth, not the raw per-file %
    assert progs[0][1] == int(100 * (0 + 40 / 100.0) / 2)


def test_run_queue_reflects_active_set():
    stub = StubCockpit()
    c = _mk_batch(stub)
    c._series = [{"key": "s1", "name": "Cell 1"}, {"key": "s2", "name": "Cell 2"}]
    c._run_order = ["s1", "s2"]
    c._series_status = {"s1": "done", "s2": "running"}
    c._cur_index = 1
    c._cur_progress = 37
    q = c.runQueue
    assert [r["name"] for r in q] == ["Cell 1", "Cell 2"]
    assert q[0]["status"] == "done" and not q[0]["current"]
    assert q[1]["status"] == "running" and q[1]["current"]
    # the 'Now' caption reads the running series + its per-file progress
    assert c.currentName == "Cell 2" and c.currentProgress == 37


def test_clear_running_ends_external_run_with_final_result():
    stub = StubCockpit()
    c = _mk_batch(stub)
    c._driving_cockpit = True
    c._final_result = ("warn", "Batch complete — 2 ok, 1 failed", "/out", {"n_tracks": 9})
    c._clear_running()
    end = [x for x in stub.calls if x[0] == "end"]
    assert end and end[0][1] == "warn" and end[0][3] == {"n_tracks": 9}
    assert not c._driving_cockpit and c._final_result is None


def test_late_hyperfly_status_detaches_cockpit_mirror():
    stub = StubCockpit()
    c = _mk_batch(stub)
    c._driving_cockpit = True
    tabs = []
    c.requestTab.connect(tabs.append)
    c._on_hf_status({"active": True})
    assert not c._driving_cockpit
    assert ("end", "", "", {}) in stub.calls
    assert tabs == [4]                                # moved to HYPER-FLY dashboard


def test_forwarders_are_inert_when_not_driving():
    stub = StubCockpit()
    c = _mk_batch(stub)
    c._driving_cockpit = False
    c._series = [{"key": "s1", "name": "Cell 1"}]
    c._run_order = ["s1"]; c._series_status = {"s1": "running"}; c._cur_index = 0
    c._files_total = 1
    c._on_log("x"); c._on_mass([1.0]); c._on_progress((10, "Detecting"))
    c._on_file_starting({"index": 1, "total": 1, "file": "Cell 1"})
    c._on_hf_preview(_frame_payload())
    assert stub.calls == []                           # a non-serial run never feeds it


def test_serial_prediction_on_this_machine():
    # On any machine that doesn't clear the HYPER-FLY bar (the common case, incl.
    # CI), a batch is predicted serial → it mirrors the Process cockpit.
    from firefly.analysis.fa_hyperfly import plan_concurrency
    params = [{"file": "a.tif"}, {"file": "b.tif"}]
    active = plan_concurrency(params).get("active", False)
    from firefly.analysis.fa_hyperfly import hyperfly_machine_eligible
    # prediction must match eligibility: serial unless the machine can fan out
    assert active == (active and hyperfly_machine_eligible())
