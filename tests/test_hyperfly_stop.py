"""Stopping a HYPER-FLY batch must not report the files it never started as failed.

Stop cancels every future that has not started.  `as_completed` still yields
them, and `fut.result()` on a cancelled future raises `CancelledError` — which
is an `Exception`, so the loop's generic handler caught it and reported a
FAILURE whose message was `str(CancelledError())`, i.e. ''.  Stop a 12-file run
after 3 and the status read "12 / 12 done · 9 failed", nine queue rows went red,
and nine dashboard tiles said "Failed" with no reason given.

The same blank appears for any exception whose str() is empty — notably
MemoryError, which a worker decoding a large CZI can genuinely hit.
"""
from concurrent.futures import Future

from firefly.firefly_worker import _hf_future_outcome


def _done(value=None, exc=None):
    f = Future()
    f.set_running_or_notify_cancel()
    f.set_exception(exc) if exc is not None else f.set_result(value)
    return f


def test_a_file_stopped_before_it_started_is_not_a_failure():
    f = Future()
    assert f.cancel()
    kind, payload = _hf_future_outcome(f)
    assert kind == "cancelled"
    assert payload is None


def test_a_finished_file_is_ok():
    kind, payload = _hf_future_outcome(_done({"ok": True, "stem": "a"}))
    assert kind == "ok" and payload["stem"] == "a"


def test_a_file_the_worker_reported_as_failed_stays_failed():
    kind, payload = _hf_future_outcome(_done({"ok": False, "error": "no frames"}))
    assert kind == "failed" and payload["error"] == "no frames"


def test_a_crash_with_an_empty_message_still_says_what_happened():
    """str(MemoryError()) is '' — the user would see "failed:" and nothing."""
    kind, payload = _hf_future_outcome(_done(exc=MemoryError()))
    assert kind == "failed"
    assert payload["error"], "a failure with no reason is not actionable"
    assert "MemoryError" in payload["error"]


def test_a_file_stopped_mid_run_is_not_a_failure():
    """Stop / a watchdog abort reaching a RUNNING file: the worker says so, and
    the batch sets it aside (or re-runs it) instead of painting it red."""
    kind, payload = _hf_future_outcome(
        _done({"ok": False, "cancelled": True, "error": "cancelled", "stem": "a"}))
    assert kind == "cancelled"
    assert payload["stem"] == "a"


def test_a_broken_pool_is_no_verdict_on_the_file():
    """One worker process dying outright fails EVERY in-flight and queued
    future with BrokenProcessPool — that says nothing about this file."""
    from concurrent.futures.process import BrokenProcessPool
    kind, payload = _hf_future_outcome(_done(exc=BrokenProcessPool("boom")))
    assert kind == "crashed"
    assert payload is None
