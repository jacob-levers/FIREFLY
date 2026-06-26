"""RunSession — a reusable worker-process run lifecycle for QML controllers.

Generalises the spawn + 33 ms ``msg_queue`` drain + ``MsgKind`` re-emission +
three-stage Stop (cooperative → SIGTERM 5s → SIGKILL 3s) + dead-process recovery
+ cleanup that AnalysisController ported from the Widgets ``_drain_msg_queue``.
BatchController composes one of these instead of duplicating ~150 lines; the
caller registers per-kind callbacks so the same machinery drives any worker
entry point (run_analysis, run_comparison, …).

Behaviour is byte-identical to the Widgets path: bounded queue (maxsize=2000),
1000-msg/tick budget, ``daemon=False`` so a successfully-enqueued terminal
message still flushes, and the exit-0-but-no-terminal case is treated as a clean
finish (NOT a crash).
"""
from __future__ import annotations

import multiprocessing
import queue
import time

from PySide6.QtCore import QObject, QTimer

from firefly.analysis.fa_enums import MsgKind


class RunSession(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._proc = None
        self._msg_queue = None
        self._cancel_event = None
        self._running = False
        self._stop_requested_at = None
        self._stop_stage = 0
        self._draining = False

        # per-run config (set in start)
        self._on_log = None
        self._on_progress = None
        self._terminal = {}        # MsgKind -> callable(payload)
        self._forward = {}         # MsgKind -> callable(payload) for streaming kinds
        self._done_kind = MsgKind.DONE
        self._on_done_no_payload = None
        self._on_dead_error = None
        self._on_finished = None   # called after cleanup on ANY terminal path

        self._poll = QTimer(self)
        self._poll.setInterval(33)
        self._poll.timeout.connect(self._tick)

    @property
    def running(self):
        return self._running

    def start(self, target, params, *, name="FIREFLY-Worker",
              on_log=None, on_progress=None, terminal=None, forward=None,
              done_kind=MsgKind.DONE, on_done_no_payload=None,
              on_dead_error=None, on_finished=None):
        if self._running:
            return False
        self._on_log = on_log
        self._on_progress = on_progress
        self._terminal = dict(terminal or {})
        self._forward = dict(forward or {})
        self._done_kind = done_kind
        self._on_done_no_payload = on_done_no_payload
        self._on_dead_error = on_dead_error
        self._on_finished = on_finished
        self._stop_requested_at = None
        self._stop_stage = 0

        self._msg_queue = multiprocessing.Queue(maxsize=2000)
        self._cancel_event = multiprocessing.Event()
        self._proc = multiprocessing.Process(
            target=target, args=(params, self._msg_queue, self._cancel_event),
            name=name, daemon=False)
        self._proc.start()
        self._running = True
        self._poll.start()
        return True

    def stop(self):
        if self._proc is not None and self._proc.is_alive():
            if self._cancel_event is not None:
                self._cancel_event.set()
            self._stop_requested_at = time.time()
            self._stop_stage = 0
            if self._on_log:
                self._on_log("\n── Stop requested.  Waiting for the current stage to "
                             "reach a checkpoint (up to 5 s); will force-terminate if "
                             "it doesn't.")

    # ── drain ────────────────────────────────────────────────────────────
    def _tick(self):
        if self._draining or self._msg_queue is None:
            return
        self._draining = True
        try:
            self._drain()
        finally:
            self._draining = False

    def _drain(self):
        budget = 1000
        worker_done = False
        log_buf = []
        last_progress = None
        while budget > 0:
            try:
                kind, payload = self._msg_queue.get_nowait()
            except (queue.Empty, Exception):
                break
            budget -= 1
            if kind == MsgKind.LOG:
                log_buf.append(payload)
            elif kind == MsgKind.PROGRESS:
                last_progress = payload
            elif kind in self._forward:
                try:    self._forward[kind](payload)
                except Exception: pass
            elif kind in self._terminal:
                try:    self._terminal[kind](payload)
                except Exception: pass
                worker_done = True

        if log_buf and self._on_log:
            self._on_log("\n".join(log_buf))
        if last_progress is not None and self._on_progress:
            try:    self._on_progress(last_progress)
            except Exception: pass

        self._escalate_stop()

        if not worker_done and self._proc is not None and not self._proc.is_alive():
            worker_done = self._handle_dead()

        if worker_done:
            self._cleanup()
            if self._on_finished:
                try:    self._on_finished()
                except Exception: pass

    def _escalate_stop(self):
        stop_at = self._stop_requested_at
        if not (stop_at is not None and self._proc is not None and self._proc.is_alive()):
            return
        elapsed = time.time() - stop_at
        if self._stop_stage == 0 and elapsed > 5.0:
            if self._on_log:
                self._on_log("  Cooperative cancel didn't take effect within 5 s — "
                             "sending SIGTERM to the worker subprocess.")
            try:    self._proc.terminate()
            except Exception: pass
            self._stop_stage = 1
            self._stop_requested_at = time.time()
        elif self._stop_stage == 1 and elapsed > 3.0:
            if self._on_log:
                self._on_log("  SIGTERM didn't take effect within 3 s — sending SIGKILL.")
            try:    self._proc.kill()
            except Exception: pass
            self._stop_stage = 2

    def _handle_dead(self) -> bool:
        terminal = None
        for _ in range(256):
            try:
                kind, payload = self._msg_queue.get_nowait()
            except Exception:
                break
            if kind == MsgKind.LOG and self._on_log:
                self._on_log(payload)
            elif kind == self._done_kind:
                terminal = payload
            elif kind in self._terminal:
                try:    self._terminal[kind](payload)
                except Exception: pass
        if terminal is not None and self._done_kind in self._terminal:
            self._terminal[self._done_kind](terminal)
        elif self._stop_requested_at is not None and MsgKind.STOPPED in self._terminal:
            self._terminal[MsgKind.STOPPED](None)
        elif self._proc.exitcode == 0:
            if self._on_done_no_payload:
                self._on_done_no_payload()
        else:
            if self._on_dead_error:
                self._on_dead_error(
                    f"Worker subprocess exited abnormally "
                    f"(exit code {self._proc.exitcode}).  See log for details.")
        return True

    def _cleanup(self):
        self._poll.stop()
        if self._proc is not None:
            try:
                if self._proc.is_alive():
                    self._proc.join(timeout=2.0)
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=1.0)
            except Exception:
                pass
        self._proc = None
        q = self._msg_queue
        if q is not None:
            try:    q.cancel_join_thread()
            except Exception: pass
            try:    q.close()
            except Exception: pass
        self._msg_queue = None
        self._cancel_event = None
        self._stop_requested_at = None
        self._running = False
