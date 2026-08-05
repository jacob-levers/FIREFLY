"""BatchController — folder-batch run lifecycle for the QML app (Phase 6b).

Scans a folder into analysable series (batch_scan), lets the user pick which
series — and which constituent split files — to run, builds the per-series
params list byte-identically to the Widgets ``_collect_batch_params``
(build_params + stem_override + series_files), exports the HYPER-FLY env vars,
and runs ``firefly_worker.run_batch_analysis`` through the shared RunSession —
re-emitting the batch ``MsgKind`` traffic (FILE_STARTING / FILE_DONE /
FILE_ERROR / BATCH_DONE) as progress.

The queue is surfaced to QML two ways: a flat ``series`` QVariantList (used by
the scalar bindings + the headless tests) and a ``seriesModel``
``QAbstractListModel`` whose stable delegates let the expand-drawer / chevron /
ROI-badge animations survive a selection change without a full Repeater rebuild.
"""
from __future__ import annotations

import os
import queue
import threading

from PySide6 import QtWidgets
from PySide6.QtCore import (Property, QAbstractListModel, QByteArray, QModelIndex,
                            QObject, Qt, QTimer, QUrl, Signal, Slot)

from firefly.analysis.fa_enums import MsgKind
from firefly.ui.controllers.params import batch_scan
from firefly.ui.controllers.params import params_builder
from firefly.ui.controllers.params.run_session import RunSession


class BatchSeriesModel(QAbstractListModel):
    """A thin list model over the owning BatchController's series state.  Each
    row is one series; ``data`` reads a freshly built row dict so the model
    always reflects the controller's live selection/status.  The controller
    drives structural changes through ``reset()`` and per-row updates through
    ``rowChanged()`` / ``allChanged()`` so delegates animate in place."""

    _ROLES = ["key", "name", "fileCount", "selState", "checked", "open",
              "status", "progress", "hasRoi", "roiLabel", "framesTotal",
              "hasUnreadable", "sizeStr", "primaryPath", "parts", "removing",
              "sourceType", "canPreview"]

    def __init__(self, owner: "BatchController"):
        super().__init__(owner)
        self._owner = owner
        self._roles = {Qt.UserRole + 1 + i: QByteArray(n.encode())
                       for i, n in enumerate(self._ROLES)}
        self._names = {Qt.UserRole + 1 + i: n for i, n in enumerate(self._ROLES)}

    def roleNames(self):
        return dict(self._roles)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._owner._series)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        if not (0 <= row < len(self._owner._series)):
            return None
        name = self._names.get(role)
        if name is None:
            return None
        return self._owner._row_dict(self._owner._series[row]).get(name)

    # ── controller-driven refresh helpers ────────────────────────────────
    def reset(self):
        self.beginResetModel()
        self.endResetModel()

    def rowChanged(self, row: int):
        if 0 <= row < len(self._owner._series):
            idx = self.index(row, 0)
            self.dataChanged.emit(idx, idx, list(self._roles.keys()))

    def allChanged(self):
        n = len(self._owner._series)
        if n:
            self.dataChanged.emit(self.index(0, 0), self.index(n - 1, 0),
                                  list(self._roles.keys()))


class BatchController(QObject):
    seriesChanged = Signal()
    folderChanged = Signal()
    scanningChanged = Signal()
    probingChanged = Signal()
    runningChanged = Signal()
    progressChanged = Signal()
    statusChanged = Signal()
    # events
    logLine = Signal(str)
    runFinished = Signal()
    runFailed = Signal(str)
    runStopped = Signal()
    # Ask the shell to switch tabs when a run starts (Process cockpit for a
    # serial batch, HYPER-FLY dashboard for a parallel one).
    requestTab = Signal(int)

    def __init__(self, settings, importc, roi_store=None, override_store=None,
                 hyperfly=None, cockpit=None, parent=None):
        super().__init__(parent)
        self._s = settings
        self._import = importc
        self._roi_store = roi_store
        self._override_store = override_store
        self._hf = hyperfly       # HyperflyController (live parallel dashboard)
        self._cockpit = cockpit   # AnalysisController — a serial batch mirrors it
        self._driving_cockpit = False   # True while feeding the Process cockpit
        self._final_result = None       # (severity, headline, out_dir, stats) for endExternalRun
        self._folder = ""
        self._recursive = False
        self._out_dir = ""
        self._series = []                 # list of scan dicts (with "parts")
        self._checked_files = {}          # series key → set(int file indices)
        self._open = set()                # series keys with the drawer expanded
        self._removing = set()            # series keys mid collapse-out animation
        self._frames = {}                 # file abspath → frame count (lazy)
        self._unreadable = set()          # file abspaths that couldn't be read (probed lazily)
        self._running = False
        self._progress = 0
        self._status = ""
        self._error = ""
        self._files_total = 0
        self._files_done = 0
        self._files_failed = 0
        self._series_status = {}          # series key → queued/running/done/error
        self._run_order = []              # checked series keys, in run order
        self._cur_index = -1              # index into _run_order of the live series
        self._cur_progress = 0            # running series' progress %
        self._session = RunSession(self)
        self._model = BatchSeriesModel(self)
        self._scanning = False            # folder scan runs off-thread (it probes
        self._scan_result = None          # files, so it's I/O-bound)
        self._scan_poll = QTimer(self)    # drains the scan result on the GUI thread
        self._scan_poll.setInterval(30)
        self._scan_poll.timeout.connect(self._drain_scan)

        # Background "probe all" — frame counts + readability for every queued
        # file, so the collapsed rows flag unreadable files WITHOUT the user
        # expanding each series.  Cheap metadata reads, off the GUI thread; rows
        # refresh progressively as results land.  A generation counter supersedes
        # an in-flight probe when the queue changes (rescan / add / clear).
        self._probing = False
        self._probe_done = False
        self._probe_gen = 0
        self._probe_q: "queue.Queue" = queue.Queue()
        self._probe_poll = QTimer(self)
        self._probe_poll.setInterval(150)
        self._probe_poll.timeout.connect(self._drain_probe)

    # ── folder + scan ────────────────────────────────────────────────────
    @Property(QObject, constant=True)
    def seriesModel(self):
        return self._model

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

    def _sister_suffix(self) -> str:
        """The user's companion-image suffix, so the scan folds companions into
        their recording instead of queueing each one as its own analysis."""
        try:
            return (self._s.get_str("analysis/roi_sister_suffix", "_green") or "").strip()
        except Exception:
            return "_green"

    @Slot()
    def rescan(self):
        if self._scanning or not self._folder:
            return
        folder, recursive = self._folder, self._recursive
        sfx = self._sister_suffix()
        self._scanning = True
        self._scan_result = None
        self.scanningChanged.emit()
        self._scan_poll.start()

        def _work():
            try:
                self._scan_result = ("ok", batch_scan.scan_series(
                    folder, recursive, sister_suffix=sfx))
            except Exception as exc:
                self._scan_result = ("err", str(exc))
        threading.Thread(target=_work, daemon=True).start()

    def _drain_scan(self):
        """GUI-thread: apply the off-thread folder-scan result to the model."""
        r = self._scan_result
        if r is None:
            return
        self._scan_result = None
        self._scan_poll.stop()
        self._scanning = False
        if r[0] == "err":
            self._series = []
            self.logLine.emit(f"Scan failed: {r[1]}")
        else:
            self._series = r[1]
        self._reset_selection()
        self._series_status = {}
        self._model.reset()
        self.scanningChanged.emit()
        self.seriesChanged.emit()
        self._start_probe_all()           # eagerly flag unreadable files in the queue

    @Property(bool, notify=scanningChanged)
    def scanning(self):
        return self._scanning

    @Property(bool, notify=probingChanged)
    def probing(self):
        return self._probing

    # ── background probe-all (frame counts + readability) ─────────────────
    def _start_probe_all(self):
        """Probe every queued file's frame count + readability off the GUI
        thread, so the queue flags unreadable files (and shows frame totals)
        without the user expanding each series.  Supersedes any in-flight probe
        via a generation counter; results land progressively."""
        self._probe_gen += 1
        gen = self._probe_gen
        self._probe_poll.stop()
        try:                                   # drop any stale queued results
            while True:
                self._probe_q.get_nowait()
        except queue.Empty:
            pass
        paths = [p["path"] for s in self._series for p in s.get("parts", [])
                 if p["path"] not in self._frames]
        if not paths:
            if self._probing:
                self._probing = False
                self.probingChanged.emit()
            return
        self._probing = True
        self.probingChanged.emit()
        self._probe_poll.start()

        def _work():
            from firefly.ui.controllers.params.preview_loader import quick_frame_count
            from firefly.analysis.fa_loaders import probe_external_locs_text
            for path in paths:
                if self._probe_gen != gen:     # superseded → bail
                    return
                is_csv = path.lower().endswith((".csv", ".txt", ".tsv"))
                try:
                    if is_csv:
                        # Text inputs have no cheap frame count, but do run the
                        # same bounded corruption guard as the worker so an
                        # all-NUL/truncated export is visible before Start.
                        probe_external_locs_text(path)
                        n, bad = 0, False
                    else:
                        n = quick_frame_count(path)
                        bad = n <= 0
                except Exception:
                    n, bad = 0, True
                self._probe_q.put((gen, path, int(n), bool(bad)))
            self._probe_q.put((gen, None, 0, False))     # sentinel: done
        threading.Thread(target=_work, daemon=True).start()

    def _drain_probe(self):
        """GUI thread: fold probe results into _frames / _unreadable + refresh."""
        got = done = False
        try:
            while True:
                g, path, n, bad = self._probe_q.get_nowait()
                if g != self._probe_gen:       # stale (superseded probe) → skip
                    continue
                if path is None:
                    done = True
                    continue
                self._frames[path] = n
                if bad:
                    self._unreadable.add(path)
                else:
                    self._unreadable.discard(path)
                got = True
        except queue.Empty:
            pass
        if got:
            self._model.allChanged()
            self.seriesChanged.emit()
        if done:
            self._probe_poll.stop()
            self._probing = False
            self.probingChanged.emit()

    def _reset_selection(self):
        """Default-select every constituent file of every series."""
        self._checked_files = {s["key"]: set(range(len(s["parts"])))
                               for s in self._series}
        self._open = set()
        self._removing = set()
        self._frames = {}
        self._unreadable = set()

    # ── series → row dicts ────────────────────────────────────────────────
    def _row_dict(self, s):
        k = s["key"]
        parts = s.get("parts", [])
        checked_idx = self._checked_files.get(k, set())
        n = len(parts)
        sel = ("on" if (n and len(checked_idx) == n)
               else "off" if not checked_idx else "ind")
        is_sel = bool(checked_idx)
        status = self._series_status.get(k, "queued" if is_sel else "skipped")
        ftot = sum(self._frames.get(parts[i]["path"], 0)
                   for i in checked_idx if 0 <= i < n)
        source_type = s.get("sourceType") or batch_scan.input_kind(s.get("primary", "")) or "image"
        rows = [{"name": p["name"], "sizeStr": p["sizeStr"],
                 "frames": self._frames.get(p["path"], 0),
                 "unreadable": p["path"] in self._unreadable,
                 "sourceType": p.get("sourceType", source_type),
                 "checked": i in checked_idx}
                for i, p in enumerate(parts)]
        return {"key": k, "name": s["name"], "fileCount": s["fileCount"],
                "selState": sel, "checked": is_sel, "open": k in self._open,
                "status": status,
                "progress": self._cur_progress if status == "running" else 0,
                "hasRoi": self._has_roi(s), "roiLabel": self._roi_label(s),
                "framesTotal": ftot,
                "hasUnreadable": any(p["path"] in self._unreadable for p in parts),
                "sizeStr": s.get("sizeStr", ""), "primaryPath": s["primary"],
                "parts": rows, "removing": k in self._removing,
                "sourceType": source_type, "canPreview": source_type == "image"}

    # short label for the per-file ROI override (empty when the file uses the
    # sidebar default), shown as a badge on the series row
    _ROI_SHORT = {"None": "None", "Auto threshold": "Auto",
                  "Manual threshold": "Manual", "Manual polygon": "Polygon",
                  "Sister TIFF": "Sister", "ImageJ ROI": "ImageJ"}

    def _has_roi(self, s):
        try:
            return bool((self._override_store and self._override_store.has(s["primary"]))
                        or (self._roi_store and self._roi_store.has(s["primary"])))
        except Exception:
            return False

    def _roi_label(self, s):
        try:
            ovr = self._override_store.get(s["primary"]) if self._override_store else None
            if ovr:
                return self._ROI_SHORT.get(ovr.get("roi_mode", ""), "ROI")
            if self._roi_store and self._roi_store.has(s["primary"]):
                return "Polygon"
        except Exception:
            pass
        return ""

    def _index_of(self, key):
        for i, s in enumerate(self._series):
            if s["key"] == key:
                return i
        return -1

    def _selected_keys(self):
        return [s["key"] for s in self._series if self._checked_files.get(s["key"])]

    @Property("QVariantList", notify=seriesChanged)
    def series(self):
        return [self._row_dict(s) for s in self._series]

    @Property(int, notify=seriesChanged)
    def seriesCount(self):
        return len(self._series)

    @Property(int, notify=seriesChanged)
    def fileCountTotal(self):
        return sum(len(s.get("parts", [])) for s in self._series)

    @Property(int, notify=seriesChanged)
    def selectedFileCount(self):
        return sum(len(self._checked_files.get(s["key"], set())) for s in self._series)

    @Property(bool, notify=seriesChanged)
    def allFilesSelected(self):
        tot = sum(len(s.get("parts", [])) for s in self._series)
        return tot > 0 and self.selectedFileCount == tot

    @Property(int, notify=seriesChanged)
    def unreadableCount(self):
        """How many queued files couldn't be read — a top-level signal so a bad
        file is visible without expanding every series."""
        return sum(1 for s in self._series for p in s.get("parts", [])
                   if p["path"] in self._unreadable)

    @Property(bool, notify=seriesChanged)
    def allExpanded(self):
        return bool(self._series) and all(s["key"] in self._open for s in self._series)

    @Property(int, notify=seriesChanged)
    def queueDone(self):
        return sum(1 for v in self._series_status.values() if v in ("done", "error"))

    @Property(int, notify=seriesChanged)
    def queuePending(self):
        return sum(1 for k in self._selected_keys()
                   if self._series_status.get(k, "queued") in ("queued", "running"))

    @Property(str, notify=seriesChanged)
    def summary(self):
        n = len(self._series)
        sel = len(self._selected_keys())
        if not n:
            return "No analysable files found." if self._folder else "Pick a folder."
        return f"{n} series · {sel} selected"

    # ── selection ─────────────────────────────────────────────────────────
    @Slot(str, bool)
    def setChecked(self, key, on):
        """Check/uncheck a whole series (all its constituent files)."""
        i = self._index_of(key)
        if i < 0:
            return
        n = len(self._series[i]["parts"])
        self._checked_files[key] = set(range(n)) if on else set()
        self._model.rowChanged(i)
        self.seriesChanged.emit()

    @Slot(str, int, bool)
    def setFileChecked(self, key, idx, on):
        i = self._index_of(key)
        if i < 0 or not (0 <= idx < len(self._series[i]["parts"])):
            return
        cur = self._checked_files.setdefault(key, set())
        cur.add(idx) if on else cur.discard(idx)
        self._model.rowChanged(i)
        self.seriesChanged.emit()

    @Slot(bool)
    def selectAll(self, on):
        for s in self._series:
            n = len(s["parts"])
            self._checked_files[s["key"]] = set(range(n)) if on else set()
        self._model.allChanged()
        self.seriesChanged.emit()

    # ── expand / collapse ─────────────────────────────────────────────────
    # Expanding a drawer is now a pure visibility toggle — frame counts and the
    # unreadable flag are filled in by the background probe (_start_probe_all,
    # kicked off on scan / add), so neither setOpen nor expandAll blocks the GUI
    # thread reading file metadata.  "Expand all" on a big queue used to freeze
    # while it probed every file inline.
    @Slot(str, bool)
    def setOpen(self, key, on):
        i = self._index_of(key)
        if i < 0:
            return
        if on:
            self._open.add(key)
        else:
            self._open.discard(key)
        self._model.rowChanged(i)
        self.seriesChanged.emit()

    @Slot(bool)
    def expandAll(self, on):
        self._open = {s["key"] for s in self._series} if on else set()
        self._model.allChanged()
        self.seriesChanged.emit()

    # ── add / remove / clear ──────────────────────────────────────────────
    def _append(self, new_series):
        """Append scanned series, preserving distinct same-stem inputs.

        Scanner-local key allocation cannot see a queue populated by an earlier
        Add-files action.  Treat an identical source as a duplicate, but qualify
        a distinct image/table collision instead of silently dropping it.
        """
        existing = {s["key"]: s for s in self._series}
        to_add = []
        for s in new_series:
            key = s["key"]
            old = existing.get(key)
            if old is not None:
                old_files = set(old.get("files") or [])
                new_files = set(s.get("files") or [])
                if (old_files == new_files and
                        old.get("sourceType") == s.get("sourceType")):
                    continue
                tag = "locs" if s.get("sourceType") == "external_loc" else "image"
                candidate, suffix = f"{key}__{tag}", 2
                while candidate in existing:
                    candidate = f"{key}__{tag}_{suffix}"
                    suffix += 1
                s = dict(s)
                s["key"] = candidate
                key = candidate
            existing[key] = s
            to_add.append(s)
        if not to_add:
            return 0
        start = len(self._series)
        self._model.beginInsertRows(QModelIndex(), start, start + len(to_add) - 1)
        for s in to_add:
            self._series.append(s)
            self._checked_files[s["key"]] = set(range(len(s["parts"])))
        self._model.endInsertRows()
        self.seriesChanged.emit()
        self._start_probe_all()           # probe the newly-added files too
        return len(to_add)

    @Slot()
    def addFolder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Add a folder to the queue",
            self._folder or os.path.expanduser("~"))
        if not path:
            return
        if not self._folder:
            self._folder = path
            self.folderChanged.emit()
        try:
            self._append(batch_scan.scan_series(
                path, self._recursive, sister_suffix=self._sister_suffix()))
        except Exception as exc:
            self.logLine.emit(f"Scan failed: {exc}")

    @Slot()
    def addFiles(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            None, "Add files to the queue",
            self._folder or os.path.expanduser("~"),
            "Recordings (*.tif *.tiff *.czi *.csv *.txt *.tsv);;All files (*)")
        if not paths:
            return
        if not self._folder:
            self._folder = os.path.dirname(paths[0])
            self.folderChanged.emit()
        self._append(batch_scan.scan_paths(
            list(paths), sister_suffix=self._sister_suffix()))

    @Slot("QVariantList", result=int)
    def addPaths(self, urls):
        """Drag-and-drop: accept a mix of folders + files (file:// URLs or plain
        paths), scan/group each, and append."""
        files, dirs = [], []
        for u in urls or []:
            p = self._local_path(u)
            if not p:
                continue
            if os.path.isdir(p):
                dirs.append(p)
            elif os.path.isfile(p):
                files.append(p)
        added = 0
        for d in dirs:
            if not self._folder:
                self._folder = d
                self.folderChanged.emit()
            try:
                added += self._append(batch_scan.scan_series(
                    d, self._recursive, sister_suffix=self._sister_suffix()))
            except Exception:
                pass
        if files:
            if not self._folder:
                self._folder = os.path.dirname(files[0])
                self.folderChanged.emit()
            added += self._append(batch_scan.scan_paths(
                files, sister_suffix=self._sister_suffix()))
        return added

    @staticmethod
    def _local_path(u):
        s = str(u)
        if s.startswith("file:"):
            return QUrl(s).toLocalFile()
        return s

    @Slot(str)
    def removeSeries(self, key):
        """Flag the row for removal — the QML delegate animates a collapse + fade
        and then calls ``finalizeRemove`` to actually drop it from the model (so
        the row doesn't just vanish).  Reduce-motion collapses instantly."""
        i = self._index_of(key)
        if i < 0 or key in self._removing:
            return
        self._removing.add(key)
        self._model.rowChanged(i)
        self.seriesChanged.emit()

    @Slot(str)
    def finalizeRemove(self, key):
        i = self._index_of(key)
        if i < 0:
            self._removing.discard(key)
            return
        self._model.beginRemoveRows(QModelIndex(), i, i)
        self._series.pop(i)
        self._model.endRemoveRows()
        self._checked_files.pop(key, None)
        self._open.discard(key)
        self._removing.discard(key)
        self._series_status.pop(key, None)
        self.seriesChanged.emit()

    @Slot()
    def clear(self):
        self._probe_gen += 1              # cancel any in-flight probe
        self._probe_poll.stop()
        if self._probing:
            self._probing = False
            self.probingChanged.emit()
        self._series = []
        self._checked_files = {}
        self._open = set()
        self._removing = set()
        self._frames = {}
        self._unreadable = set()
        self._series_status = {}
        self._model.reset()
        self.seriesChanged.emit()

    @Slot()
    def notifyRoiChanged(self):
        """Called by QML when the ROI editor closes so the per-series ROI badge
        refreshes (the polygon is persisted per file in the RoiStore)."""
        self._model.allChanged()
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
        return bool(self._selected_keys()) and not self._running

    @Property(int, notify=progressChanged)
    def filesDone(self):
        return self._files_done

    @Property(int, notify=progressChanged)
    def filesTotal(self):
        return self._files_total

    @Property(int, notify=progressChanged)
    def filesFailed(self):
        return self._files_failed

    @Property("QVariantList", notify=seriesChanged)
    def runQueue(self):
        """Compact view of the active run set — the series actually queued, with
        their status — for the at-a-glance pip grid on the Process tab."""
        names = {s["key"]: s.get("name", s["key"]) for s in self._series}
        out = []
        for i, key in enumerate(self._run_order):
            st = self._series_status.get(key, "queued")
            out.append({
                "name": names.get(key, key),
                "status": st,
                "current": (i == self._cur_index),
            })
        return out

    @Property(str, notify=seriesChanged)
    def currentName(self):
        """Name of the series currently running (for the queue's 'Now' caption)."""
        if 0 <= self._cur_index < len(self._run_order):
            key = self._run_order[self._cur_index]
            for s in self._series:
                if s["key"] == key:
                    return s.get("name", key)
            return key
        return ""

    @Property(int, notify=seriesChanged)
    def currentProgress(self):
        """Progress % of the series currently running (0 when none is live)."""
        return self._cur_progress if 0 <= self._cur_index < len(self._run_order) else 0

    # ── params + spawn ───────────────────────────────────────────────────
    def _build_params_list(self):
        out_root = self._out_root()
        plist = []
        for s in self._series:
            idx = sorted(self._checked_files.get(s["key"], set()))
            if not idx:
                continue
            checked_paths = [s["parts"][i]["path"] for i in idx
                             if 0 <= i < len(s["parts"])]
            if not checked_paths:
                continue
            primary = (s["primary"] if s["primary"] in checked_paths
                       else checked_paths[0])
            source_type = (s.get("sourceType") or
                           batch_scan.input_kind(primary) or "image")
            selected_types = {batch_scan.input_kind(p) for p in checked_paths}
            # The scanner produces homogeneous rows.  Keep this validation at
            # the launch boundary as a safety net: a mixed list would otherwise
            # reach ``load_file`` as a bogus TIFF/CZI series.
            if selected_types != {source_type}:
                self.logLine.emit(
                    f"Skipped mixed input series '{s['key']}' — queue images "
                    "and localisation tables as separate runs.")
                continue
            p = params_builder.build_params(self._s, self._import,
                                            fpath=primary, out_dir=out_root,
                                            roi_store=self._roi_store,
                                            override_store=self._override_store)
            p["stem_override"] = s["key"]
            if source_type == "image":
                p["series_files"] = list(checked_paths)
            # Multiple ROIs flagged as individual replicates → one run per ROI,
            # each writing to <key>_cell{n} (or the user's label).  No-op otherwise.
            plist.extend(params_builder.expand_roi_replicates(p))
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
        self._driving_cockpit = False
        self._final_result = None
        self._files_total = len(params_list)
        self._files_done = 0
        self._files_failed = 0
        self._progress = 0
        # Per-series queue state: the checked series, in scan order, all queued.
        self._run_order = [s["key"] for s in self._series
                           if self._checked_files.get(s["key"])]
        self._series_status = {k: "queued" for k in self._run_order}
        self._cur_index = -1
        self._cur_progress = 0
        self._status = f"Batch: 0 / {self._files_total} series…"
        self.progressChanged.emit()
        self.statusChanged.emit()
        self._model.allChanged()

        # Decide serial vs HYPER-FLY the same way the worker will (reads the env
        # we just exported), so we can move to the right screen: a serial batch
        # mirrors the Process cockpit; a parallel one has its own dashboard.
        serial = True
        try:
            from firefly.analysis.fa_hyperfly import plan_concurrency
            serial = not plan_concurrency(params_list).get("active", False)
        except Exception:
            serial = True

        if self._hf:
            # stem → input path, so the dashboard can show each running tile's
            # max-projection (HF_TILE only carries the stem, not the path).
            stem_paths = {p.get("stem_override"): p.get("file")
                          for p in params_list
                          if p.get("stem_override") and p.get("file")}
            self._hf.start(self._files_total, stem_paths)   # arm the HYPER-FLY dashboard

        from firefly import firefly_worker
        ok = self._session.start(
            firefly_worker.run_batch_analysis, params_list, name="FIREFLY-BatchWorker",
            on_log=self._on_log, on_progress=self._on_progress,
            done_kind=MsgKind.BATCH_DONE,
            terminal={MsgKind.BATCH_DONE: self._on_batch_done,
                      MsgKind.STOPPED: self._on_stopped,
                      MsgKind.ERROR: self._on_error},
            forward={MsgKind.FILE_STARTING: self._on_file_starting,
                     MsgKind.FILE_DONE: self._on_file_done,
                     MsgKind.FILE_ERROR: self._on_file_error,
                     MsgKind.MASS_CHUNK: self._on_mass,
                     MsgKind.HF_TILE: self._on_hf_tile,
                     MsgKind.PREVIEW_FRAME: self._on_hf_preview,
                     MsgKind.HYPERFLY_STATUS: self._on_hf_status},
            on_done_no_payload=lambda: self._finish("Finished — result didn't reach the UI"),
            on_dead_error=self._on_error,
            on_finished=self._clear_running)
        if ok:
            self._running = True
            self.runningChanged.emit()
            self._model.allChanged()
            self.seriesChanged.emit()
            # A serial batch takes over the Process cockpit (live preview, log,
            # stepper, meters); a parallel one lights up the HYPER-FLY dashboard.
            if serial and self._cockpit is not None and self._cockpit.beginExternalRun():
                self._driving_cockpit = True
                self.requestTab.emit(1)          # Process
            elif not serial:
                self.requestTab.emit(4)          # HYPER-FLY

    @Slot()
    def stop(self):
        self._session.stop()

    # ── drain callbacks ──────────────────────────────────────────────────
    def _on_log(self, line):
        self.logLine.emit(line)
        if self._driving_cockpit and self._cockpit is not None:
            self._cockpit.feedLog(line)

    def _on_mass(self, payload):
        if self._driving_cockpit and self._cockpit is not None:
            self._cockpit.feedMass(payload)

    def _on_progress(self, payload):
        try:
            pct, msg = payload
        except Exception:
            try:
                pct, msg = int(payload[0]), ""
            except Exception:
                return
        self._progress = int(pct)
        self._cur_progress = int(pct)
        self.progressChanged.emit()
        self._refresh_running_row()          # advance the running series' bar
        if self._driving_cockpit and self._cockpit is not None:
            # The batch stamps a "[i/n] file" marker (overall %) at each file
            # boundary; skip it so the cockpit's bar reads as a smooth overall
            # (completed files + the current file's fraction) and the stepper
            # only advances on real per-file stage messages.
            m = str(msg or "")
            if not m.startswith("["):
                p = max(0, min(100, int(pct)))
                overall = int(100 * (self._files_done + p / 100.0)
                              / max(1, self._files_total))
                self._cockpit.feedProgress(overall, m)

    def _mark(self, status):
        if 0 <= self._cur_index < len(self._run_order):
            self._series_status[self._run_order[self._cur_index]] = status

    def _refresh_running_row(self):
        if 0 <= self._cur_index < len(self._run_order):
            self._model.rowChanged(self._index_of(self._run_order[self._cur_index]))
        self.seriesChanged.emit()

    def _on_file_starting(self, payload):
        i = (payload or {}).get("index", 0)
        total = (payload or {}).get("total", self._files_total)
        self._files_total = total or self._files_total
        self._cur_index = i - 1
        self._cur_progress = 0
        self._mark("running")
        self._status = f"Batch: {max(0, i - 1)} / {self._files_total} done"
        self.statusChanged.emit()
        self._model.allChanged()
        self.seriesChanged.emit()
        if self._driving_cockpit and self._cockpit is not None:
            self._cockpit.resetStepper()      # rewind the per-file stepper

    def _on_file_done(self, payload):
        self._mark("done")
        self._files_done += 1
        self._cur_progress = 0
        self._bump_file_progress()
        self._refresh_running_row()

    def _on_file_error(self, payload):
        self._mark("error")
        self._files_failed += 1
        self._files_done += 1
        self._cur_progress = 0
        self._bump_file_progress()
        self._refresh_running_row()

    def _bump_file_progress(self):
        if self._files_total:
            self._progress = int(100 * self._files_done / self._files_total)
        self._status = (f"Batch: {self._files_done} / {self._files_total} done"
                        + (f" · {self._files_failed} failed" if self._files_failed else ""))
        if self._hf:
            self._hf.onProgress(self._files_done, self._files_total, self._files_failed)
        self.progressChanged.emit()
        self.statusChanged.emit()

    # ── HYPER-FLY dashboard forwarders ────────────────────────────────────
    def _on_hf_tile(self, payload):
        if self._hf:
            self._hf.onTile(payload)

    def _on_hf_preview(self, payload):
        if self._hf:
            self._hf.onPreview(payload)
        if self._driving_cockpit and self._cockpit is not None:
            self._cockpit.feedPreview(payload)

    def _on_hf_status(self, payload):
        if self._hf:
            self._hf.onStatus(payload)
        # We predicted serial but HYPER-FLY engaged after all — detach the
        # cockpit mirror and move to the dashboard (belt-and-braces; the
        # prediction reads the same env the worker does, so this is rare).
        if self._driving_cockpit and (payload or {}).get("active"):
            self._driving_cockpit = False
            if self._cockpit is not None:
                self._cockpit.endExternalRun("", "", "", {})
            self.requestTab.emit(4)

    def _on_batch_done(self, payload):
        payload = payload or {}
        n_ok = payload.get("n_ok", self._files_done - self._files_failed)
        n_fail = payload.get("n_fail", self._files_failed)
        self._progress = 100
        self._status = f"Batch complete — {n_ok} ok" + (f", {n_fail} failed" if n_fail else "")
        self.progressChanged.emit()
        self.statusChanged.emit()
        # Cockpit result card: headline + aggregate trajectory/localisation totals.
        results = payload.get("results") or []
        stats = {}
        try:
            stats = {"n_tracks": int(sum(r.get("n_tracks", 0) or 0 for r in results)),
                     "n_locs":   int(sum(r.get("n_locs", 0) or 0 for r in results))}
        except Exception:
            stats = {}
        self._final_result = (
            "warn" if n_fail else "ok",
            f"Batch complete — {n_ok} ok" + (f", {n_fail} failed" if n_fail else ""),
            self._out_root(), stats)
        self.runFinished.emit()

    def _on_stopped(self, _payload=None):
        self._finish("Stopped")
        self._final_result = ("warn", "Batch stopped", "", {})
        self.runStopped.emit()

    def _on_error(self, message):
        self._finish("Error — see log")
        self._error = str(message or "Batch failed.")
        self.logLine.emit("\n" + self._error)
        self.statusChanged.emit()
        self._final_result = ("error", "Batch error — see log", "", {})
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
        if self._hf:
            self._hf.finish()
        if self._driving_cockpit and self._cockpit is not None:
            sev, headline, out_dir, stats = (
                self._final_result or ("ok", "Batch complete", "", {}))
            self._cockpit.endExternalRun(sev, headline, out_dir, stats)
        self._driving_cockpit = False
        self._final_result = None
        self.runningChanged.emit()
        self._model.allChanged()
        self.seriesChanged.emit()
