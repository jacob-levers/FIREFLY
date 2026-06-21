"""AnalysisController — the QML run lifecycle + live cockpit bridge (Phase 3).

A faithful port of the Widgets run path (``_start_single_run`` +
``_drain_msg_queue`` + the three-stage Stop in ``ui_mixin_handlers``), wrapped as
a QObject so QML binds to it.  It:

  * builds the worker params dict from persisted settings + the Import tab
    (:mod:`firefly.ui.controllers.params.params_builder` — byte-identical shape),
  * spawns ``firefly_worker.run_analysis`` in a ``multiprocessing.Process``
    (spawn context, clean interpreter for MPS/CUDA — same rationale as the
    Widgets app),
  * drains the message queue on a 33 ms QTimer and re-emits ``MsgKind`` traffic
    as Qt signals / properties (progress, stage, log lines, mass-histogram
    chunks, live detection frames, terminal done/stopped/failed),
  * escalates Stop cooperatively → SIGTERM (5 s) → SIGKILL (3 s),
  * tracks elapsed time + samples CPU/RAM for the resource meters.

The analysis core is untouched; this only relays its existing queue protocol.
"""
from __future__ import annotations

import multiprocessing
import os
import queue
import sys
import time

from PySide6.QtCore import (Property, QObject, QTimer, Signal, Slot)

from firefly.analysis.fa_enums import MsgKind
from firefly.ui.controllers.params import params_builder
from firefly.ui.controllers.providers.live_frame_provider import render_frame

# Connected-stepper stages — mirror ui_widgets._PipelineDiagram._STAGES so the
# QML stepper lights up identically.
STAGES = ["Preprocess", "Detect", "Link", "Drift", "Diffuse", "Classify"]


def _index_for_msg(msg):
    """('complete', None) / ('idx', i) / (None, None) for a progress message —
    ported verbatim from _PipelineDiagram._index_for_msg (most-specific first)."""
    m = (msg or "").lower()
    if any(k in m for k in ("complete", "all done", "batch complete")):
        return ("complete", None)
    if any(k in m for k in ("saving", "rendering", "secondary", "cluster", "classif")):
        return ("idx", 5)
    if any(k in m for k in ("msd", "fits", "diffus")):
        return ("idx", 4)
    if "drift" in m:
        return ("idx", 3)
    if any(k in m for k in ("linking", "pre-linked", "link")):
        return ("idx", 2)
    if any(k in m for k in ("localising", "detect")):
        return ("idx", 1)
    if any(k in m for k in ("loading stack", "reading localisations",
                            "loaded from csv", "loading")):
        return ("idx", 0)
    return (None, None)


def _format_elapsed(secs: float) -> str:
    secs = max(0, int(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class AnalysisController(QObject):
    # property-notify signals
    runningChanged = Signal()
    stageChanged = Signal()
    progressChanged = Signal()
    elapsedChanged = Signal()
    frameTokenChanged = Signal()
    resultChanged = Signal()
    statsChanged = Signal()
    resourcesChanged = Signal()
    minmassChanged = Signal()
    # event signals
    logLine = Signal(str)
    massChunk = Signal("QVariantList")
    runFinished = Signal()
    runFailed = Signal(str)
    runStopped = Signal()

    def __init__(self, settings, importc, roi_store=None, override_store=None,
                 parent=None):
        super().__init__(parent)
        self._s = settings
        self._import = importc
        self._roi_store = roi_store
        self._override_store = override_store

        self._proc = None
        self._msg_queue = None
        self._cancel_event = None

        self._running = False
        self._stage = -1
        self._complete = False
        self._progress = 0
        self._progress_text = ""
        self._stage_label = ""
        self._elapsed = "00:00"
        self._run_start = None

        self._frame_token = 0
        self._live_image = None        # read by LiveFrameProvider

        self._headline = ""
        self._out_dir = ""
        self._severity = ""
        self._stats: dict = {}

        self._cpu = 0.0
        self._mem = 0.0
        self._gpu = -1.0               # 0..100 %, or -1 when there's no percentage
        self._gpu_text = ""            # shown when _gpu < 0 (e.g. "MPS", "CPU only")
        self._vram_text = "—"          # "Unified" on Apple Silicon, "X / Y GB" on CUDA
        self._mps_util_cache = None    # latest ioreg GPU % (written off-thread)
        self._mps_polling = False
        self._torch = None
        self._minmass = -1.0           # -1 → auto (no histogram threshold line)

        self._stop_requested_at = None
        self._stop_stage = 0

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(33)
        self._poll_timer.timeout.connect(self._drain)
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)
        self._res_timer = QTimer(self)
        self._res_timer.setInterval(1000)
        self._res_timer.timeout.connect(self._sample_resources)
        # Always-on (1 Hz) so all four meters are live at idle and during runs,
        # matching the legacy Widgets _ResourceMonitor.
        self._res_timer.start()
        # First sample is deferred to the event loop (not the __init__ critical
        # path) so the ioreg / nvidia-smi probe can never stall app startup.
        QTimer.singleShot(0, self._sample_resources)

    # ── properties ───────────────────────────────────────────────────────
    @Property(bool, notify=runningChanged)
    def running(self):
        return self._running

    @Property("QStringList", constant=True)
    def stages(self):
        return STAGES

    @Property(int, notify=stageChanged)
    def stage(self):
        return self._stage

    @Property(bool, notify=stageChanged)
    def complete(self):
        return self._complete

    @Property(int, notify=progressChanged)
    def progress(self):
        return self._progress

    @Property(str, notify=progressChanged)
    def progressText(self):
        return self._progress_text

    @Property(str, notify=progressChanged)
    def stageLabel(self):
        return self._stage_label

    @Property(str, notify=elapsedChanged)
    def elapsed(self):
        return self._elapsed

    @Property(int, notify=frameTokenChanged)
    def frameToken(self):
        return self._frame_token

    @Property(bool, notify=frameTokenChanged)
    def hasLiveFrame(self):
        return self._live_image is not None and not self._live_image.isNull()

    @Property(str, notify=resultChanged)
    def resultHeadline(self):
        return self._headline

    @Property(str, notify=resultChanged)
    def resultOutDir(self):
        return self._out_dir

    @Property(str, notify=resultChanged)
    def resultSeverity(self):
        return self._severity

    @Property("QVariantMap", notify=statsChanged)
    def stats(self):
        return self._stats

    @Property(float, notify=resourcesChanged)
    def cpuPercent(self):
        return self._cpu

    @Property(float, notify=resourcesChanged)
    def memPercent(self):
        return self._mem

    @Property(float, notify=resourcesChanged)
    def gpuPercent(self):
        return self._gpu               # 0..100, or -1 → show gpuText instead of a bar

    @Property(str, notify=resourcesChanged)
    def gpuText(self):
        return self._gpu_text          # "MPS" / "CPU only" when there's no percentage

    @Property(str, notify=resourcesChanged)
    def vramText(self):
        return self._vram_text         # "Unified" on Apple Silicon, "X / Y GB" on CUDA

    @Property(float, notify=minmassChanged)
    def minmassThreshold(self):
        return self._minmass

    # ── lifecycle ────────────────────────────────────────────────────────
    @Slot()
    def start(self):
        if self._running:
            return
        if not self._import.hasFile:
            self.runFailed.emit("Pick an input file on the Import tab first.")
            return

        params = params_builder.build_params(self._s, self._import,
                                             roi_store=self._roi_store,
                                             override_store=self._override_store)

        # Histogram threshold: only meaningful with a manual minmass.
        self._minmass = (-1.0 if params.get("auto_minmass")
                         else float(params.get("minmass") or 0.0))
        self.minmassChanged.emit()

        # Reset cockpit state for the new run.
        self._stage = -1
        self._complete = False
        self._progress = 0
        self._progress_text = "Starting…"
        self._stage_label = "Starting…"
        self._headline = ""
        self._out_dir = ""
        self._severity = ""
        self._stats = {}
        self._live_image = None
        self._stop_requested_at = None
        self._stop_stage = 0
        self.stageChanged.emit()
        self.progressChanged.emit()
        self.resultChanged.emit()
        self.statsChanged.emit()
        self.frameTokenChanged.emit()

        # Persist settings before the long task (parity with _start_single_run).
        try:
            self._s.sync()
        except Exception:
            pass

        from firefly import firefly_worker
        self._msg_queue = multiprocessing.Queue(maxsize=2000)
        self._cancel_event = multiprocessing.Event()
        self._proc = multiprocessing.Process(
            target=firefly_worker.run_analysis,
            args=(params, self._msg_queue, self._cancel_event),
            name="FIREFLY-AnalysisWorker", daemon=False)
        self._proc.start()

        self._running = True
        self._run_start = time.monotonic()
        self._elapsed = "00:00"
        self.runningChanged.emit()
        self.elapsedChanged.emit()
        self._poll_timer.start()
        self._elapsed_timer.start()
        self._sample_resources()       # immediate refresh on run start

    @Slot()
    def stop(self):
        """Cooperative cancel; the poller escalates to SIGTERM→SIGKILL."""
        if self._proc is not None and self._proc.is_alive():
            if self._cancel_event is not None:
                self._cancel_event.set()
            self._stop_requested_at = time.time()
            self._stop_stage = 0
            self.logLine.emit(
                "\n── Stop requested.  Waiting for the current stage to reach a "
                "checkpoint (up to 5 s); will force-terminate if it doesn't.")

    # ── queue drain (33 Hz) ──────────────────────────────────────────────
    def _drain(self):
        if self._msg_queue is None:
            return
        budget = 1000
        worker_done = False
        log_buf: list[str] = []
        last_progress = None

        while budget > 0:
            try:
                kind, payload = self._msg_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            budget -= 1
            if kind == MsgKind.LOG:
                log_buf.append(payload)
            elif kind == MsgKind.PROGRESS:
                last_progress = payload
            elif kind == MsgKind.MASS_CHUNK:
                try:
                    self.massChunk.emit([float(v) for v in payload])
                except Exception:
                    pass
            elif kind == MsgKind.PREVIEW_FRAME:
                self._on_preview(payload)
            elif kind == MsgKind.DONE:
                self._handle_done(payload)
                worker_done = True
            elif kind == MsgKind.STOPPED:
                self._handle_stopped()
                worker_done = True
            elif kind == MsgKind.ERROR:
                self._handle_failed(payload)
                worker_done = True
            # batch / compare / hyperfly kinds are not produced by the
            # single-file run_analysis path (Phase 5/6).

        if log_buf:
            self.logLine.emit("\n".join(log_buf))
        if last_progress is not None:
            self._apply_progress(last_progress)

        self._escalate_stop_if_needed(log_buf)

        # Subprocess died without a terminal message (crash / kill).
        if (not worker_done and self._proc is not None
                and not self._proc.is_alive()):
            worker_done = self._handle_dead_process()

        if worker_done:
            self._cleanup_after_run()

    def _apply_progress(self, payload):
        try:
            pct, msg = payload
        except Exception:
            return
        self._progress = int(pct)
        m = (msg or "").strip()
        if len(m) > 48:
            m = m[:47] + "…"
        self._progress_text = f"{m}  —  {pct}%" if m else f"{pct}%"
        self._stage_label = msg or ""
        self.progressChanged.emit()
        # Advance the connected stepper (furthest-reached wins).
        kind, idx = _index_for_msg(msg)
        if kind == "complete":
            self._set_complete()
        elif kind == "idx" and idx is not None and idx > self._stage:
            self._stage = idx
            self._complete = False
            self.stageChanged.emit()

    def _on_preview(self, payload):
        try:
            import numpy as np
            shape = payload.get("shape") or [0, 0]
            blob = payload.get("frame")
            if not (blob and shape[0] and shape[1]):
                return
            arr = np.frombuffer(blob, dtype=np.float32).reshape(shape[0], shape[1])
            xs = payload.get("xs", [])
            ys = payload.get("ys", [])
            img = render_frame(arr, xs, ys)
            if not img.isNull():
                self._live_image = img
                self._frame_token += 1
                self.frameTokenChanged.emit()
        except Exception:
            pass

    # ── terminal handlers ────────────────────────────────────────────────
    def _set_complete(self):
        self._stage = len(STAGES) - 1
        self._complete = True
        self.stageChanged.emit()

    def _handle_done(self, payload):
        payload = payload or {}
        out_dir = payload.get("out_dir", "") or ""
        stem = payload.get("stem", "") or ""
        summary = payload.get("summary") or {}
        n_tracks = summary.get("n_tracks", payload.get("n_tracks", 0)) or 0
        if not n_tracks:
            self._headline = (f"{stem} — no trajectories produced" if stem
                              else "Analysis finished — no trajectories produced")
            self._severity = "warn"
        else:
            self._headline = (f"{stem} — {n_tracks:,} trajectories" if stem
                              else f"Analysis complete — {n_tracks:,} trajectories")
            self._severity = "ok"
        self._out_dir = out_dir
        self._stats = {str(k): v for k, v in summary.items()}
        self._set_complete()
        self._progress = 100
        self._progress_text = "Complete"
        self._stage_label = "Done"
        self.progressChanged.emit()
        self.resultChanged.emit()
        self.statsChanged.emit()
        self.runFinished.emit()

    def _handle_stopped(self):
        self._headline = "Stopped by user"
        self._severity = "warn"
        self._progress_text = "Stopped"
        self.resultChanged.emit()
        self.progressChanged.emit()
        self.runStopped.emit()

    def _handle_failed(self, tb: str):
        tb = tb or "Analysis failed."
        self._headline = "Analysis error — see log"
        self._severity = "error"
        self._progress_text = "Error"
        self.resultChanged.emit()
        self.progressChanged.emit()
        self.logLine.emit("\n" + tb)
        self.runFailed.emit(tb)

    def _escalate_stop_if_needed(self, log_buf):
        stop_at = self._stop_requested_at
        if not (stop_at is not None and self._proc is not None
                and self._proc.is_alive()):
            return
        elapsed = time.time() - stop_at
        if self._stop_stage == 0 and elapsed > 5.0:
            self.logLine.emit(
                "  Cooperative cancel didn't take effect within 5 s — "
                "sending SIGTERM to the analysis subprocess.")
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._stop_stage = 1
            self._stop_requested_at = time.time()
        elif self._stop_stage == 1 and elapsed > 3.0:
            self.logLine.emit("  SIGTERM didn't take effect within 3 s — sending SIGKILL.")
            try:
                self._proc.kill()
            except Exception:
                pass
            self._stop_stage = 2

    def _handle_dead_process(self) -> bool:
        """Drain any final messages a dead worker left, then report a terminal
        state.  Mirrors the dead-process branch of _drain_msg_queue."""
        terminal = None
        for _ in range(256):
            try:
                kind, payload = self._msg_queue.get_nowait()
            except Exception:
                break
            if kind == MsgKind.LOG:
                self.logLine.emit(payload)
            elif kind == MsgKind.DONE:
                terminal = payload
        if terminal is not None:
            self._handle_done(terminal)
        elif self._stop_requested_at is not None:
            self._handle_stopped()
        elif self._proc.exitcode == 0:
            self._headline = "Analysis finished — result didn't reach the UI"
            self._severity = "warn"
            self._out_dir = ""
            self.resultChanged.emit()
            self.runFinished.emit()
        else:
            self._handle_failed(
                f"Analysis subprocess exited abnormally "
                f"(exit code {self._proc.exitcode}).  See log for details.")
        return True

    # ── timers ───────────────────────────────────────────────────────────
    def _tick_elapsed(self):
        if self._run_start is None:
            return
        self._elapsed = _format_elapsed(time.monotonic() - self._run_start)
        self.elapsedChanged.emit()

    def _sample_resources(self):
        # CPU + RAM via psutil (cheap, always available).
        try:
            import psutil
            self._cpu = float(psutil.cpu_percent(interval=None))
            self._mem = float(psutil.virtual_memory().percent)
        except Exception:
            pass
        # GPU % + VRAM — probed torch-free so we never import torch into the
        # GUI process (subprocess-isolation invariant; see firefly_worker).
        self._gpu, self._gpu_text, self._vram_text = self._sample_gpu()
        self.resourcesChanged.emit()

    def _sample_gpu(self):
        """Return ``(gpu_percent, gpu_text, vram_text)``.  ``gpu_percent`` is -1
        when there's no numeric utilisation to show (then ``gpu_text`` labels
        it).  Never imports torch — uses ioreg (macOS) / pynvml / nvidia-smi."""
        # ── macOS: Metal/MPS utilisation via ioreg (torch-free) ───────────
        if sys.platform == "darwin":
            self._poll_mps_async()
            if self._mps_util_cache is not None:
                gpu, gpu_text = float(self._mps_util_cache), ""
            else:
                gpu, gpu_text = -1.0, "MPS"
            # Apple-Silicon memory is unified with system RAM — no separate
            # VRAM pool.  Report torch's MPS allocator footprint only if torch
            # is already loaded (rare in the GUI process), else say "Unified".
            vram = "Unified"
            torch_mod = self._maybe_get_torch()
            if torch_mod is not None:
                try:
                    if (hasattr(torch_mod.backends, "mps")
                            and torch_mod.backends.mps.is_available()):
                        vram = f"{torch_mod.mps.current_allocated_memory()/1e9:.1f} GB"
                except Exception:
                    pass
            return gpu, gpu_text, vram

        # ── NVIDIA (any other OS): util + VRAM via pynvml (torch-free) ─────
        try:
            import pynvml as _nv
            _nv.nvmlInit()
            try:
                h = _nv.nvmlDeviceGetHandleByIndex(0)
                util = _nv.nvmlDeviceGetUtilizationRates(h).gpu
                mem = _nv.nvmlDeviceGetMemoryInfo(h)
                return (float(util), "",
                        f"{mem.used/1e9:.1f} / {mem.total/1e9:.0f} GB")
            finally:
                try:    _nv.nvmlShutdown()
                except Exception: pass
        except Exception:
            pass
        # nvidia-smi fallback (no pynvml installed).
        try:
            import subprocess as _sub
            flags = (getattr(_sub, "CREATE_NO_WINDOW", 0)
                     if sys.platform == "win32" else 0)
            out = _sub.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
                creationflags=flags).stdout.strip().splitlines()
            if out:
                u, used, total = [s.strip() for s in out[0].split(",")]
                return (float(u), "",
                        f"{float(used)/1024:.1f} / {float(total)/1024:.0f} GB")
        except Exception:
            pass

        # ── nothing detected → CPU-only run ───────────────────────────────
        return -1.0, "CPU only", "—"

    @staticmethod
    def _mps_gpu_utilization():
        """Best-effort macOS GPU utilisation via ``ioreg`` (a subprocess — call
        only off the GUI thread, via :meth:`_poll_mps_async`).  Returns an int
        percent or None.  Chip/OS combos expose the field under a few keys."""
        import re
        import subprocess
        try:
            out = subprocess.check_output(
                ["ioreg", "-r", "-c", "IOAccelerator", "-d", "1"],
                stderr=subprocess.DEVNULL, timeout=1.0).decode(
                    "utf-8", errors="ignore")
        except Exception:
            return None
        for pattern in (r'"Device Utilization\s*%"\s*=\s*(\d+)',
                        r'"GPU Busy %"\s*=\s*(\d+)',
                        r'"GPU Core Utilization\s*%"\s*=\s*(\d+)'):
            m = re.search(pattern, out)
            if m:
                try:    return int(m.group(1))
                except Exception: pass
        return None

    def _poll_mps_async(self):
        """Run the ioreg parse on a daemon thread → ``self._mps_util_cache``.
        Re-entrancy-guarded so 1 Hz ticks don't pile up threads."""
        if self._mps_polling:
            return
        self._mps_polling = True
        import threading

        def _run():
            try:    self._mps_util_cache = self._mps_gpu_utilization()
            finally: self._mps_polling = False
        threading.Thread(target=_run, daemon=True).start()

    def _maybe_get_torch(self):
        """Return torch ONLY if it's already in ``sys.modules`` — never triggers
        a fresh import (keeps torch/Metal out of the GUI process)."""
        if self._torch is None:
            self._torch = sys.modules.get("torch")
        return self._torch

    def _cleanup_after_run(self):
        self._poll_timer.stop()
        self._elapsed_timer.stop()
        # _res_timer stays running — the meters keep updating at idle.
        if self._run_start is not None:
            self._elapsed = _format_elapsed(time.monotonic() - self._run_start)
            self.elapsedChanged.emit()
        self._run_start = None
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
            try:
                q.cancel_join_thread()
            except Exception:
                pass
            try:
                q.close()
            except Exception:
                pass
        self._msg_queue = None
        self._cancel_event = None
        self._stop_requested_at = None
        self._running = False
        self.runningChanged.emit()
