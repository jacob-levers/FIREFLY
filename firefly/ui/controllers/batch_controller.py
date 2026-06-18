"""BatchController — folder-batch run lifecycle for the QML app (Phase 6b).

Scans a folder into analysable series (batch_scan), lets the user pick which to
run, builds the per-series params list byte-identically to the Widgets
``_collect_batch_params`` (build_params + stem_override + series_files), exports
the HYPER-FLY env vars, and runs ``firefly_worker.run_batch_analysis`` through
the shared RunSession — re-emitting the batch ``MsgKind`` traffic (FILE_STARTING
/ FILE_DONE / FILE_ERROR / BATCH_DONE) as progress.
"""
from __future__ import annotations

import os

from PySide6 import QtWidgets
from PySide6.QtCore import Property, QObject, Signal, Slot

from firefly.analysis.fa_enums import MsgKind
from firefly.ui.controllers import batch_scan
from firefly.ui.controllers import params_builder
from firefly.ui.controllers.run_session import RunSession


class BatchController(QObject):
    seriesChanged = Signal()
    folderChanged = Signal()
    runningChanged = Signal()
    progressChanged = Signal()
    statusChanged = Signal()
    # events
    logLine = Signal(str)
    runFinished = Signal()
    runFailed = Signal(str)
    runStopped = Signal()

    def __init__(self, settings, importc, roi_store=None, parent=None):
        super().__init__(parent)
        self._s = settings
        self._import = importc
        self._roi_store = roi_store
        self._folder = ""
        self._recursive = False
        self._out_dir = ""
        self._series = []                 # list of scan dicts
        self._checked = set()             # series keys
        self._running = False
        self._progress = 0
        self._status = ""
        self._error = ""
        self._files_total = 0
        self._files_done = 0
        self._files_failed = 0
        self._session = RunSession(self)

    # ── folder + scan ────────────────────────────────────────────────────
    @Property(str, notify=folderChanged)
    def folder(self):
        return self._folder

    @Property(bool, notify=folderChanged)
    def recursive(self):
        return self._recursive

    @recursive.setter
    def recursive(self, v):
        v = bool(v)
        if v != self._recursive:
            self._recursive = v
            self.folderChanged.emit()
            if self._folder:
                self.rescan()

    @Slot()
    def browseFolder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Pick a folder to batch-process",
            self._folder or os.path.expanduser("~"))
        if path:
            self._folder = path
            self.folderChanged.emit()
            self.rescan()

    @Slot(str)
    def scan(self, folder):
        self._folder = folder
        self.folderChanged.emit()
        self.rescan()

    @Slot()
    def rescan(self):
        try:
            self._series = batch_scan.scan_series(self._folder, self._recursive)
        except Exception as exc:
            self._series = []
            self.logLine.emit(f"Scan failed: {exc}")
        self._checked = {s["key"] for s in self._series}   # all selected by default
        self.seriesChanged.emit()

    @Property("QVariantList", notify=seriesChanged)
    def series(self):
        return [{"key": s["key"], "name": s["name"], "fileCount": s["fileCount"],
                 "checked": s["key"] in self._checked} for s in self._series]

    @Property(str, notify=seriesChanged)
    def summary(self):
        n = len(self._series)
        sel = len(self._checked)
        if not n:
            return "No analysable files found." if self._folder else "Pick a folder."
        return f"{n} series · {sel} selected"

    @Slot(str, bool)
    def setChecked(self, key, on):
        if on:
            self._checked.add(key)
        else:
            self._checked.discard(key)
        self.seriesChanged.emit()

    @Slot(bool)
    def selectAll(self, on):
        self._checked = {s["key"] for s in self._series} if on else set()
        self.seriesChanged.emit()

    # ── output ───────────────────────────────────────────────────────────
    @Property(str, notify=folderChanged)
    def outputDir(self):
        return self._out_dir

    @Slot()
    def browseOutputDir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Batch output folder (optional)",
            self._out_dir or self._folder or os.path.expanduser("~"))
        if path:
            self._out_dir = path
            self.folderChanged.emit()

    def _out_root(self):
        if self._out_dir:
            return self._out_dir
        return os.path.join(self._folder, "batch_results") if self._folder else ""

    # ── run state ────────────────────────────────────────────────────────
    @Property(bool, notify=runningChanged)
    def running(self):
        return self._running

    @Property(int, notify=progressChanged)
    def progress(self):
        return self._progress

    @Property(str, notify=statusChanged)
    def status(self):
        return self._status

    @Property(str, notify=statusChanged)
    def generateError(self):
        return self._error

    @Property(bool, notify=seriesChanged)
    def canRun(self):
        return bool(self._checked) and not self._running

    @Property(int, notify=progressChanged)
    def filesDone(self):
        return self._files_done

    @Property(int, notify=progressChanged)
    def filesTotal(self):
        return self._files_total

    @Property(int, notify=progressChanged)
    def filesFailed(self):
        return self._files_failed

    # ── params + spawn ───────────────────────────────────────────────────
    def _build_params_list(self):
        out_root = self._out_root()
        plist = []
        for s in self._series:
            if s["key"] not in self._checked:
                continue
            primary = s["primary"]
            p = params_builder.build_params(self._s, self._import,
                                            fpath=primary, out_dir=out_root,
                                            roi_store=self._roi_store)
            p["stem_override"] = s["key"]
            if not str(primary).lower().endswith((".csv", ".txt", ".tsv")):
                p["series_files"] = list(s["files"])
            plist.append(p)
        return plist

    def _export_hyperfly_env(self):
        try:
            g = self._s
            hf = g.get_str("performance/hyperfly", "Auto (recommended)").lower()
            os.environ["FIREFLY_HYPERFLY"] = ("off" if "off" in hf
                                              else "on" if "always" in hf else "auto")
            os.environ["FIREFLY_HYPERFLY_MAX_FILES"] = str(int(g.get_float("performance/hyperfly_max_files", 0)))
            os.environ["FIREFLY_HYPERFLY_MAX_CORES"] = str(int(g.get_float("performance/hyperfly_max_cores", 0)))
            os.environ["FIREFLY_HYPERFLY_MAX_RAM_GB"] = str(int(g.get_float("performance/hyperfly_max_ram", 0)))
            os.environ["FIREFLY_HYPERFLY_LOAD_SLOTS"] = str(int(g.get_float("performance/hyperfly_load_slots", 0)))
            os.environ["FIREFLY_HYPERFLY_GPU_SLOTS"] = str(int(g.get_float("performance/hyperfly_gpu_slots", 0)))
            os.environ["FIREFLY_CZI_PARALLEL_DECODE"] = (
                "1" if g.get_bool("performance/czi_parallel_decode", True) else "0")
        except Exception:
            pass

    @Slot()
    def generate(self):
        if self._running:
            return
        params_list = self._build_params_list()
        if not params_list:
            self._set_error("Check at least one series to run.")
            return
        out_root = self._out_root()
        try:
            os.makedirs(out_root, exist_ok=True)
        except OSError as exc:
            self._set_error(f"Can't create output folder: {exc}")
            return
        self._export_hyperfly_env()
        if self._s:
            try:    self._s.sync()
            except Exception: pass

        self._error = ""
        self._files_total = len(params_list)
        self._files_done = 0
        self._files_failed = 0
        self._progress = 0
        self._status = f"Batch: 0 / {self._files_total} series…"
        self.progressChanged.emit()
        self.statusChanged.emit()

        from firefly import firefly_worker
        ok = self._session.start(
            firefly_worker.run_batch_analysis, params_list, name="FIREFLY-BatchWorker",
            on_log=self.logLine.emit, on_progress=self._on_progress,
            done_kind=MsgKind.BATCH_DONE,
            terminal={MsgKind.BATCH_DONE: self._on_batch_done,
                      MsgKind.STOPPED: self._on_stopped,
                      MsgKind.ERROR: self._on_error},
            forward={MsgKind.FILE_STARTING: self._on_file_starting,
                     MsgKind.FILE_DONE: self._on_file_done,
                     MsgKind.FILE_ERROR: self._on_file_error},
            on_done_no_payload=lambda: self._finish("Finished — result didn't reach the UI"),
            on_dead_error=self._on_error,
            on_finished=self._clear_running)
        if ok:
            self._running = True
            self.runningChanged.emit()
            self.seriesChanged.emit()

    @Slot()
    def stop(self):
        self._session.stop()

    # ── drain callbacks ──────────────────────────────────────────────────
    def _on_progress(self, payload):
        try:
            pct, _ = payload
            self._progress = int(pct)
            self.progressChanged.emit()
        except Exception:
            pass

    def _on_file_starting(self, payload):
        i = (payload or {}).get("index", 0)
        total = (payload or {}).get("total", self._files_total)
        self._files_total = total or self._files_total
        self._status = f"Batch: {max(0, i - 1)} / {self._files_total} done"
        self.statusChanged.emit()

    def _on_file_done(self, payload):
        self._files_done += 1
        self._bump_file_progress()

    def _on_file_error(self, payload):
        self._files_failed += 1
        self._files_done += 1
        self._bump_file_progress()

    def _bump_file_progress(self):
        if self._files_total:
            self._progress = int(100 * self._files_done / self._files_total)
        self._status = (f"Batch: {self._files_done} / {self._files_total} done"
                        + (f" · {self._files_failed} failed" if self._files_failed else ""))
        self.progressChanged.emit()
        self.statusChanged.emit()

    def _on_batch_done(self, payload):
        payload = payload or {}
        n_ok = payload.get("n_ok", self._files_done - self._files_failed)
        n_fail = payload.get("n_fail", self._files_failed)
        self._progress = 100
        self._status = f"Batch complete — {n_ok} ok" + (f", {n_fail} failed" if n_fail else "")
        self.progressChanged.emit()
        self.statusChanged.emit()
        self.runFinished.emit()

    def _on_stopped(self, _payload=None):
        self._finish("Stopped")
        self.runStopped.emit()

    def _on_error(self, message):
        self._finish("Error — see log")
        self._error = str(message or "Batch failed.")
        self.logLine.emit("\n" + self._error)
        self.statusChanged.emit()
        self.runFailed.emit(self._error)

    def _finish(self, status):
        self._status = status
        self.statusChanged.emit()

    def _set_error(self, msg):
        self._error = msg
        self._status = ""
        self.statusChanged.emit()

    def _clear_running(self):
        self._running = False
        self.runningChanged.emit()
        self.seriesChanged.emit()
