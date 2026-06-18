"""ResultsController — QML bridge for a comparison ``{stem}_results.json``.

Loads a comparison snapshot (the same JSON the Widgets ``_ResultsView`` reads)
and exposes it as QML-bindable models: the comparison figure (via a cache-busting
image provider), headline metrics, group/condition chips, per-metric verdict
cards, the statistics-config grid, and the run's output files.  All formatting
goes through the shared :mod:`firefly.ui.results_format` so the QML view and the
Widgets view never fork.  Read-only — no run lifecycle (that's CompareController).
"""
from __future__ import annotations

import json
import os
import statistics

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Property, QObject, QUrl, Signal, Slot

from firefly.analysis.fa_constants import motion_class_colors
from firefly.ui import results_format as rf

_MOTION_ORDER = ["Immobile", "Confined", "Brownian", "Directed"]


def _median(vals):
    xs = [float(v) for v in vals if rf._isfinite(v)]
    return statistics.median(xs) if xs else None


class ResultsController(QObject):
    resultsChanged = Signal()
    figureChanged = Signal()
    renderFailed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: dict = {}
        self._base_dir = ""
        self._figure_path = ""
        self._figure_token = 0
        self._headline = {"tracks": "—", "medianD": "—", "medianAlpha": "—", "alpha2": "—"}
        self._header_title = ""
        self._two_factor = False
        self._group_chips: list = []
        self._pair_warn = ""
        self._config_rows: list = []
        self._output_files: list = []
        self._metric_cards: list = []
        self._urls = {"figure": "", "pdf": "", "stats": ""}

    # provider hook
    def figure_path(self):
        return self._figure_path

    # ── load ─────────────────────────────────────────────────────────────
    @Slot("QVariant", str)
    def load(self, results, base_dir):
        try:
            self._data = dict(results or {})
            self._base_dir = base_dir or ""
            self._rebuild()
            self._figure_token += 1
            self.figureChanged.emit()
            self.resultsChanged.emit()
        except Exception as exc:
            self.renderFailed.emit(f"Couldn't fully render these results: {exc}")

    @Slot(str)
    def loadFromFile(self, path):
        try:
            with open(path) as fh:
                data = json.load(fh)
            self.load(data, os.path.dirname(path))
        except Exception as exc:
            self.renderFailed.emit(f"Couldn't open {os.path.basename(path)}: {exc}")

    @Slot()
    def openPrevious(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Open a previous comparison", os.path.expanduser("~"),
            "Comparison results (*_results.json);;JSON (*.json)")
        if path:
            self.loadFromFile(path)

    def _file(self, name):
        """Absolute path for a relative filename in the JSON, or '' if absent."""
        if not name:
            return ""
        p = os.path.join(self._base_dir, name)
        return p if os.path.isfile(p) else ""

    def _rebuild(self):
        meta = self._data.get("meta") or {}
        summary = self._data.get("summary") or []
        files = meta.get("files") or {}

        # figure
        self._figure_path = self._file(files.get("png"))
        self._urls = {
            "figure": QUrl.fromLocalFile(self._figure_path).toString() if self._figure_path else "",
            "pdf": QUrl.fromLocalFile(self._file(files.get("report_pdf"))).toString()
                   if self._file(files.get("report_pdf")) else "",
            "stats": QUrl.fromLocalFile(self._file(files.get("stats_csv"))).toString()
                     if self._file(files.get("stats_csv")) else "",
        }

        # header
        two = bool(meta.get("two_factor"))
        self._two_factor = two
        if two:
            ng = len(meta.get("group_order") or [])
            nt = len(meta.get("timepoints") or [])
            self._header_title = f"{ng} groups × {nt} time points"
        else:
            ng = int(meta.get("n_groups") or 0)
            self._header_title = f"{ng} group{'s' if ng != 1 else ''}"
        self._pair_warn = str(meta.get("pair_warn") or "")

        # group chips (by first appearance in summary)
        color_map = dict(zip(meta.get("group_labels") or [],
                             meta.get("group_colors") or []))
        from collections import Counter, OrderedDict
        order = OrderedDict()
        for row in summary:
            g = str(row.get("group", ""))
            label = f"{g} / {row.get('timepoint', '')}" if two else g
            order.setdefault(label, g)
        counts = Counter(
            (f"{r.get('group','')} / {r.get('timepoint','')}" if two else str(r.get("group", "")))
            for r in summary)
        self._group_chips = [
            {"label": label, "color": color_map.get(g, "#8b949e"), "count": counts[label]}
            for label, g in order.items()]

        # config grid
        self._config_rows = [
            {"label": str(p[0]), "value": str(p[1])}
            for p in (self._data.get("config_summary") or []) if len(p) >= 2]

        # headline metrics (aggregate across replicates)
        n_tracks = sum(int(r.get("n_tracks") or 0) for r in summary)
        self._headline = {
            "tracks": f"{n_tracks:,}" if n_tracks else "—",
            "medianD": rf._fmt_num(_median(r.get("median_D") for r in summary), "{:.3f}"),
            "medianAlpha": rf._fmt_num(_median(r.get("median_alpha") for r in summary), "{:.2f}"),
            "alpha2": rf._fmt_num(_median(r.get("nongauss_alpha2") for r in summary), "{:.2f}"),
        }

        # output files
        of = []
        for key, kind in (("png", "figure"), ("report_pdf", "pdf"),
                          ("stats_csv", "csv"), ("summary_csv", "csv"),
                          ("circular_tests", "csv"), ("circular_per_replicate", "csv")):
            p = self._file(files.get(key))
            if p:
                of.append({"path": p, "relPath": os.path.basename(p), "kind": kind})
        self._output_files = of

        # per-metric verdict cards
        n_groups = int(meta.get("n_groups") or 0)
        stats = self._data.get("stats") or {}
        cards = []
        for key, disp in rf.ordered_metrics(stats):
            rec = stats.get(key) or {}
            sev, html, _ = rf._verdict_for_metric(disp, rec, n_groups)
            cards.append({"key": key, "title": disp, "severity": sev,
                          "verdictHtml": html,
                          "hasDetails": bool(rec.get("pairwise"))})
        self._metric_cards = cards

    # ── properties ───────────────────────────────────────────────────────
    @Property(int, notify=figureChanged)
    def figureToken(self):
        return self._figure_token

    @Property(bool, notify=figureChanged)
    def hasFigure(self):
        return bool(self._figure_path)

    @Property(str, notify=resultsChanged)
    def headerTitle(self):
        return self._header_title

    @Property(bool, notify=resultsChanged)
    def hasResults(self):
        return bool(self._data)

    @Property(str, notify=resultsChanged)
    def tracksLabel(self):
        return self._headline["tracks"]

    @Property(str, notify=resultsChanged)
    def medianD(self):
        return self._headline["medianD"]

    @Property(str, notify=resultsChanged)
    def medianAlpha(self):
        return self._headline["medianAlpha"]

    @Property(str, notify=resultsChanged)
    def alpha2(self):
        return self._headline["alpha2"]

    @Property("QVariantList", notify=resultsChanged)
    def groupChips(self):
        return self._group_chips

    @Property(str, notify=resultsChanged)
    def pairWarn(self):
        return self._pair_warn

    @Property("QVariantList", notify=resultsChanged)
    def configRows(self):
        return self._config_rows

    @Property("QVariantList", notify=resultsChanged)
    def outputFiles(self):
        return self._output_files

    @Property("QVariantList", notify=resultsChanged)
    def metricCards(self):
        return self._metric_cards

    @Property(bool, notify=resultsChanged)
    def hasOutputFolder(self):
        return bool(self._base_dir) and os.path.isdir(self._base_dir)

    @Property(str, notify=resultsChanged)
    def figureUrl(self):
        return self._urls["figure"]

    @Property(str, notify=resultsChanged)
    def reportPdfUrl(self):
        return self._urls["pdf"]

    @Property(str, notify=resultsChanged)
    def statsCsvUrl(self):
        return self._urls["stats"]

    @Property("QVariantList", constant=True)
    def motionClasses(self):
        pal = motion_class_colors("Dark")
        return [{"label": c, "colorHex": pal.get(c, "#aaaaaa")} for c in _MOTION_ORDER]

    # ── actions ──────────────────────────────────────────────────────────
    @Slot()
    def openFolder(self):
        if self._base_dir:
            QtGui.QDesktopServices.openUrl(QUrl.fromLocalFile(self._base_dir))

    @Slot(str)
    def openFile(self, abs_path):
        if abs_path and os.path.exists(abs_path):
            QtGui.QDesktopServices.openUrl(QUrl.fromLocalFile(abs_path))

    @Slot(str)
    def openUrl(self, url):
        if url:
            QtGui.QDesktopServices.openUrl(QUrl(url))

    @Slot(str)
    def copyPath(self, abs_path):
        cb = QtWidgets.QApplication.clipboard()
        if cb is not None:
            cb.setText(abs_path or "")
