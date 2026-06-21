"""CompareController — QML bridge for the Compare tab (Phase 5b).

Owns the comparison groups model (condition chips), builds the worker's
``comparison_params`` byte-identically (compare_params_builder), and runs
``firefly_worker.run_comparison`` through a shared RunSession.  On success it
re-emits ``resultsReady`` so the shell loads the snapshot into the Results tab.
The statistics-config editors live in Phase-6 Preferences; here stats_config is
read from QSettings.
"""
from __future__ import annotations

import os
import statistics

from PySide6 import QtWidgets
from PySide6.QtCore import Property, QObject, Signal, Slot

from firefly.analysis.fa_enums import MsgKind
from firefly.ui.controllers.params import compare_params_builder as cpb
from firefly.ui.controllers.params.run_session import RunSession
from firefly.ui import results_format as rf

_DEFAULT_COLORS = ["#3b6ed8", "#f78166", "#56d364", "#d2a8ff",
                   "#ffa657", "#79c0ff", "#e3b341", "#ff7b72",
                   "#39c5cf", "#bc8cff", "#7ee787", "#ffa198"]
_MAX_GROUPS = 6


class CompareController(QObject):
    conditionsChanged = Signal()
    runReadyChanged = Signal()
    runningChanged = Signal()
    progressChanged = Signal()
    statusChanged = Signal()
    figureChanged = Signal()
    paramsChanged = Signal()
    statsRowsChanged = Signal()
    # events
    logLine = Signal(str)
    runFinished = Signal()
    runFailed = Signal(str)
    runStopped = Signal()
    resultsReady = Signal(str)        # results_json path

    def __init__(self, settings, results=None, parent=None):
        super().__init__(parent)
        self._s = settings
        self._results = results
        self._groups: list = []
        self._next_id = 0
        self._out_dir = settings.get_str("compare/output_dir", "") if settings else ""
        self._out_stem = settings.get_str("compare/output_stem", "comparison") if settings else "comparison"

        self._running = False
        self._progress = 0
        self._progress_text = ""
        self._status = ""
        self._error = ""
        self._figure_path = ""
        self._figure_token = 0
        self._has_result = False
        self._p_label = ""
        self._significant = False
        self._test_label = ""
        self._stats_rows: list = []

        self._session = RunSession(self)

        # seed two empty conditions so the tab opens ready to fill
        self.addCondition()
        self.addCondition()

    # provider hook
    def figure_path(self):
        return self._figure_path

    # ── groups model ─────────────────────────────────────────────────────
    def _find(self, gid):
        return next((g for g in self._groups if g["id"] == gid), None)

    @Property("QVariantList", notify=conditionsChanged)
    def conditions(self):
        return [{"id": g["id"], "name": g["label"], "colorHex": g["color"],
                 "timepoint": g["timepoint"], "folderCount": len(g["folders"]),
                 "nTracksLabel": ""} for g in self._groups]

    @Slot()
    def addCondition(self):
        if len(self._groups) >= _MAX_GROUPS:
            return
        idx = len(self._groups)
        self._groups.append({
            "id": self._next_id, "label": f"Group {idx + 1}",
            "color": _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)],
            "timepoint": "", "folders": []})
        self._next_id += 1
        self._emit_conditions()

    @Slot(int)
    def removeCondition(self, gid):
        self._groups = [g for g in self._groups if g["id"] != gid]
        self._emit_conditions()

    @Slot(int, str)
    def setLabel(self, gid, label):
        g = self._find(gid)
        if g is not None:
            g["label"] = label
            self._emit_conditions()

    @Slot(int, str)
    def setColor(self, gid, hex_):
        g = self._find(gid)
        if g is not None:
            g["color"] = hex_
            self._emit_conditions()

    @Slot(int, str)
    def setTimepoint(self, gid, tp):
        g = self._find(gid)
        if g is not None:
            g["timepoint"] = tp
            self._emit_conditions()

    @Slot(int, "QVariantList")
    def addFolders(self, gid, urls):
        g = self._find(gid)
        if g is None:
            return
        for u in urls:
            p = u
            if isinstance(u, str) and u.startswith("file://"):
                from PySide6.QtCore import QUrl
                p = QUrl(u).toLocalFile()
            if p and os.path.isdir(p) and p not in g["folders"]:
                g["folders"].append(p)
        self._emit_conditions()

    @Slot(int)
    def browseAddFolder(self, gid):
        g = self._find(gid)
        if g is None:
            return
        path = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Add an analysis-output folder",
            self._out_dir or os.path.expanduser("~"))
        if path and path not in g["folders"]:
            g["folders"].append(path)
            self._emit_conditions()

    @Slot(int, str)
    def removeFolder(self, gid, path):
        g = self._find(gid)
        if g is not None and path in g["folders"]:
            g["folders"].remove(path)
            self._emit_conditions()

    @Slot(int, result="QVariantList")
    def folders(self, gid):
        g = self._find(gid)
        return list(g["folders"]) if g else []

    def _emit_conditions(self):
        self.conditionsChanged.emit()
        self.runReadyChanged.emit()

    # ── output ───────────────────────────────────────────────────────────
    @Property(str, notify=paramsChanged)
    def outputDir(self):
        return self._out_dir

    @outputDir.setter
    def outputDir(self, v):
        self._out_dir = v
        if self._s:
            self._s.set("compare/output_dir", v)
        self.paramsChanged.emit()

    @Slot()
    def browseOutputDir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Choose the comparison output folder",
            self._out_dir or os.path.expanduser("~"))
        if path:
            self.outputDir = path

    @Property(str, notify=paramsChanged)
    def outputStem(self):
        return self._out_stem

    @outputStem.setter
    def outputStem(self, v):
        self._out_stem = v or "comparison"
        if self._s:
            self._s.set("compare/output_stem", self._out_stem)
        self.paramsChanged.emit()

    # ── run state ────────────────────────────────────────────────────────
    @Property(bool, notify=runningChanged)
    def running(self):
        return self._running

    @Property(int, notify=progressChanged)
    def progress(self):
        return self._progress

    @Property(str, notify=progressChanged)
    def progressText(self):
        return self._progress_text

    @Property(str, notify=statusChanged)
    def status(self):
        return self._status

    @Property(str, notify=statusChanged)
    def generateError(self):
        return self._error

    @Property(bool, notify=runReadyChanged)
    def canGenerate(self):
        non_empty = [g for g in self._groups if g["folders"]]
        return len(non_empty) >= 2 and not self._running

    @Property(int, notify=figureChanged)
    def overlayFigureToken(self):
        return self._figure_token

    @Property(bool, notify=figureChanged)
    def hasResult(self):
        return self._has_result

    @Property(str, notify=figureChanged)
    def pValueLabel(self):
        return self._p_label

    @Property(bool, notify=figureChanged)
    def significant(self):
        return self._significant

    @Property(str, notify=figureChanged)
    def testLabel(self):
        return self._test_label

    @Property("QVariantList", notify=statsRowsChanged)
    def statsRows(self):
        return self._stats_rows

    @Property("QVariantList", constant=True)
    def motionClasses(self):
        from firefly.analysis.fa_constants import motion_class_colors
        pal = motion_class_colors("Dark")
        return [{"label": c, "colorHex": pal.get(c, "#aaaaaa")}
                for c in ("Immobile", "Confined", "Brownian", "Directed")]

    # ── generate ─────────────────────────────────────────────────────────
    @Slot()
    def generate(self):
        if self._running:
            return
        non_empty = [g for g in self._groups if g["folders"]]
        if len(non_empty) < 2:
            self._set_error("Need at least 2 groups, each with at least 1 analysis folder.")
            return
        if not self._out_dir:
            self._set_error("Pick a folder to save the comparison outputs.")
            return
        try:
            os.makedirs(self._out_dir, exist_ok=True)
        except OSError as exc:
            self._set_error(f"Cannot create output folder: {exc}")
            return
        params = cpb.build_comparison_params(
            self._s, [{"label": g["label"], "color": g["color"],
                       "timepoint": g["timepoint"], "folders": list(g["folders"])}
                      for g in non_empty],
            self._out_dir, self._out_stem)
        if not params["panels"]:
            self._set_error("No comparison-figure panels are enabled.")
            return
        if self._s:
            try:    self._s.sync()
            except Exception: pass

        self._error = ""
        self._progress = 0
        self._progress_text = "Starting…"
        self._status = f"Comparing {len(non_empty)} group(s)…"
        self.progressChanged.emit()
        self.statusChanged.emit()

        from firefly import firefly_worker
        ok = self._session.start(
            firefly_worker.run_comparison, params, name="FIREFLY-CompareWorker",
            on_log=self.logLine.emit, on_progress=self._on_progress,
            done_kind=MsgKind.COMPARE_DONE,
            terminal={MsgKind.COMPARE_DONE: self._on_done,
                      MsgKind.COMPARE_ERROR: self._on_compare_error,
                      MsgKind.STOPPED: self._on_stopped,
                      MsgKind.ERROR: self._on_error},
            on_done_no_payload=lambda: self._finish_status("Finished — result didn't reach the UI", ""),
            on_dead_error=self._on_error,
            on_finished=self._clear_running)
        if ok:
            self._running = True
            self.runningChanged.emit()
            self.runReadyChanged.emit()

    @Slot()
    def stop(self):
        self._session.stop()

    def _on_progress(self, payload):
        try:
            pct, msg = payload
        except Exception:
            return
        self._progress = int(pct)
        m = (msg or "").strip()
        self._progress_text = f"{m}  —  {pct}%" if m else f"{pct}%"
        self.progressChanged.emit()

    def _on_done(self, payload):
        payload = payload or {}
        out_dir = payload.get("output_dir", "") or ""
        n_groups = payload.get("n_groups", 0)
        self._progress = 100
        self._progress_text = "Complete"
        self._status = f"Comparison complete — {n_groups} group(s)"
        fig = payload.get("figure_path", "")
        if fig and os.path.isfile(fig):
            self._figure_path = fig
            self._figure_token += 1
        self._has_result = True
        rj = payload.get("results_json", "")
        if rj and os.path.isfile(rj):
            self._derive_summary(rj)
            self.resultsReady.emit(rj)
        self.progressChanged.emit()
        self.statusChanged.emit()
        self.figureChanged.emit()
        self.runFinished.emit()

    def _derive_summary(self, results_json_path):
        """Pull the primary-metric verdict + per-group stats from the snapshot
        for the Compare tab's right-rail (no re-computation)."""
        import json
        try:
            with open(results_json_path) as fh:
                data = json.load(fh)
        except Exception:
            return
        stats = data.get("stats") or {}
        ordered = rf.ordered_metrics(stats)
        if ordered:
            key, _ = ordered[0]
            rec = stats.get(key) or {}
            omn = rec.get("omnibus") or {}
            pairs = rec.get("pairwise") or []
            if omn:
                p = omn.get("p")
                stars = omn.get("stars") or ""
                self._test_label = str(omn.get("test", ""))
            elif pairs:
                p = pairs[0].get("p_within", pairs[0].get("p"))
                stars = pairs[0].get("stars_within") or pairs[0].get("stars") or ""
                self._test_label = str(pairs[0].get("test", ""))
            else:
                p, stars = None, ""
            self._p_label = f"p = {rf._fmt_p(p)}" if p is not None else ""
            self._significant = bool(stars) and stars != "ns"
        # per-group medians
        summary = data.get("summary") or []
        by_group = {}
        for row in summary:
            by_group.setdefault(str(row.get("group", "")), []).append(row)
        rows = []
        for label, grp in by_group.items():
            def med(key):
                xs = [float(r[key]) for r in grp if rf._isfinite(r.get(key))]
                return statistics.median(xs) if xs else None
            rows.append({"groupLabel": label,
                         "d": rf._fmt_num(med("median_D"), "{:.3f}"),
                         "a": rf._fmt_num(med("median_alpha"), "{:.2f}"),
                         "mobPct": rf._fmt_num(med("mob_immob_ratio"), "{:.2f}"),
                         "highlight": False})
        self._stats_rows = rows
        self.statsRowsChanged.emit()

    def _on_compare_error(self, message):
        self._finish_status("Couldn't run", "")
        self._error = str(message or "The comparison couldn't run.")
        self.statusChanged.emit()
        self.runFailed.emit(self._error)

    def _on_stopped(self, _payload=None):
        self._finish_status("Stopped", "")
        self.runStopped.emit()

    def _on_error(self, message):
        self._finish_status("Error — see log", "")
        self._error = str(message or "The comparison failed.")
        self.statusChanged.emit()
        self.logLine.emit("\n" + self._error)
        self.runFailed.emit(self._error)

    def _finish_status(self, status, progress_text):
        self._status = status
        self._progress_text = progress_text
        self.progressChanged.emit()
        self.statusChanged.emit()

    def _set_error(self, msg):
        self._error = msg
        self._status = ""
        self.statusChanged.emit()

    # called by RunSession.on_finished implicitly via terminal handlers; ensure
    # the running flag clears on every terminal path.
    @Slot()
    def _clear_running(self):
        self._running = False
        self.runningChanged.emit()
        self.runReadyChanged.emit()
