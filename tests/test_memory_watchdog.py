"""The memory watchdog must keep watching after it has aborted once.

After a trip the batch skips the file (serial) or shrinks the wave
(HYPER-FLY), clears the events and carries on — but the watchdog thread used to
exit on its first trip, leaving every later file with no low-RAM protection.
"""
import queue
import threading
import time
import types

import psutil

import firefly.firefly_worker as fw


def _wait(cond, timeout=5.0):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def test_the_watchdog_rearms_after_the_batch_carries_on(monkeypatch):
    # Far below any abort threshold, for the whole test.
    monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(
        total=64e9, available=0.1e9))
    cancel, mem_abort = threading.Event(), threading.Event()
    stop = fw._start_memory_watchdog(cancel, queue.Queue(), poll_s=0.01,
                                     mem_abort_event=mem_abort, sustained_polls=2)
    try:
        assert _wait(cancel.is_set) and mem_abort.is_set(), "first trip"
        # The batch handles the abort and moves on to the next file …
        mem_abort.clear()
        cancel.clear()
        # … and RAM is still critically low: the watchdog must fire again.
        assert _wait(cancel.is_set), "the watchdog stopped watching after one trip"
        assert mem_abort.is_set()
    finally:
        stop.set()
