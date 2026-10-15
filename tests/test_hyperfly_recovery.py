"""HYPER-FLY must never lose a file to some other file's trouble.

Two ways a whole batch used to go wrong at once:

* One file's worker process crashing outright (a native crash: GPU/driver
  fault, access violation) broke the shared process pool, and Python then
  failed EVERY in-flight and queued file with ``BrokenProcessPool`` — one bad
  file, "0/12 succeeded".
* A memory-watchdog abort kept the finished files and silently dropped the
  rest: never run, no error message, just counted as "failed".

These drive the real engine (spawned pool, Manager queue, forwarder, collector)
with a stub per-file job in place of the analysis pipeline.  The stub runs in
the spawned workers, so it reads its knobs from the environment.
"""
import os
import queue
import sys
import threading
import time

import pytest

import firefly.analysis.fa_hyperfly as fa_hyperfly
import firefly.firefly_worker as fw
from firefly.analysis.fa_enums import MsgKind

pytestmark = pytest.mark.slow

CRASHER = 2


def _stub_job(index, n_total, params):
    """Stands in for ``_file_worker_run`` inside the spawned workers."""
    stem = params["stem_override"]
    fw._HF_FANQ.put((MsgKind.HF_TILE, {"file": index, "stem": stem,
                                        "state": "running"}))
    if index == CRASHER and os.environ.get("HF_TEST_CRASH"):
        time.sleep(0.5)
        os._exit(3)                       # dies like a native crash: no traceback
    t_end = time.monotonic() + float(os.environ.get("HF_TEST_SECONDS", "1.0"))
    while time.monotonic() < t_end:
        if fw._HF_CANCEL.is_set():
            return {"ok": False, "cancelled": True, "error": "cancelled",
                    "stem": stem}
        time.sleep(0.05)
    return {"ok": True, "stem": stem, "out_dir": "", "n_tracks": 1, "n_locs": 1}


class _TripQueue(queue.Queue):
    """GUI-queue stand-in that fires a callback on the first FILE_DONE — a
    deterministic "mid-wave" moment to Stop or trip the memory watchdog."""
    def __init__(self, on_first_done=None):
        super().__init__()
        self._on_first_done = on_first_done

    def put(self, item, block=True, timeout=None):
        super().put(item, block, timeout)
        if (self._on_first_done and isinstance(item, tuple)
                and item[0] == MsgKind.FILE_DONE):
            cb, self._on_first_done = self._on_first_done, None
            cb()


@pytest.fixture
def engine(monkeypatch, tmp_path):
    """Run ``run_batch_analysis`` over ``n`` stub files at a forced width of 4;
    returns ``(messages, watchdog_events)``."""
    monkeypatch.setattr(fw, "_file_worker_run", _stub_job)
    monkeypatch.setattr(fa_hyperfly, "plan_concurrency", lambda plist: {
        "active": True, "n_concurrent": 4, "per_file_workers": 1,
        "reason": "test wave"})
    wd = {}

    def _fake_watchdog(cancel_event, msg_queue, mem_abort_event=None, **kw):
        wd["cancel"], wd["mem_abort"] = cancel_event, mem_abort_event
        return threading.Event()
    monkeypatch.setattr(fw, "_start_memory_watchdog", _fake_watchdog)
    # run_batch_analysis rewires stdout/stderr and sets these env vars; let
    # monkeypatch put them all back.
    monkeypatch.setattr(sys, "stdout", sys.stdout)
    monkeypatch.setattr(sys, "stderr", sys.stderr)
    monkeypatch.setenv("FIREFLY_BULK_FIGURES", "1")
    monkeypatch.setenv("FIREFLY_HYPERFLY_GPU_SLOTS", "1")
    monkeypatch.setenv("HF_TEST_SECONDS", "1.5")

    def run(n, on_first_done=None):
        q = _TripQueue(on_first_done and (lambda: on_first_done(wd)))
        plist = [{"file": str(tmp_path / f"file{i:02d}.tif"),
                  "stem_override": f"file{i:02d}", "out_dir": str(tmp_path)}
                 for i in range(1, n + 1)]
        fw.run_batch_analysis(plist, q, threading.Event())
        msgs = []
        while not q.empty():
            msgs.append(q.get())
        return msgs
    return run


def _reported(msgs):
    done = [p["index"] for k, p in msgs if k == MsgKind.FILE_DONE]
    errs = {p["index"]: p["tb"] for k, p in msgs if k == MsgKind.FILE_ERROR}
    summary = next((p for k, p in msgs if k == MsgKind.BATCH_DONE), None)
    return done, errs, summary


def test_a_crashing_file_fails_alone(engine, monkeypatch):
    monkeypatch.setenv("HF_TEST_CRASH", "1")
    msgs = engine(8)
    done, errs, summary = _reported(msgs)

    assert set(errs) == {CRASHER}, errs
    assert "crashed" in errs[CRASHER]
    assert sorted(done) == [i for i in range(1, 9) if i != CRASHER]
    assert summary["n_ok"] == 7 and summary["n_fail"] == 1
    assert [r["index"] for r in summary["results"]] == list(range(1, 9))


def test_a_memory_abort_reruns_the_rest_instead_of_dropping_them(engine):
    def trip(wd):                         # the watchdog fires mid-wave
        wd["mem_abort"].set()
        wd["cancel"].set()
    msgs = engine(12, on_first_done=trip)
    done, errs, summary = _reported(msgs)

    assert errs == {}
    assert sorted(done) == list(range(1, 13)), "every file runs exactly once"
    assert summary["n_ok"] == 12 and summary["n_fail"] == 0
    logs = " ".join(str(p) for k, p in msgs if k == MsgKind.LOG)
    assert "re-running" in logs


def test_stop_reports_no_failures_for_stopped_files(engine):
    msgs = engine(12, on_first_done=lambda wd: wd["cancel"].set())
    done, errs, summary = _reported(msgs)

    assert errs == {}, "a stopped file is not a failed file"
    assert any(k == MsgKind.STOPPED for k, _ in msgs)
    assert summary is None
    assert len(done) < 12
