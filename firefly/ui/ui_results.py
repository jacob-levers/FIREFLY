"""The interactive Results tab.

Renders a comparison's results from a ``{stem}_results.json`` snapshot
(written by ``firefly.analysis.fa_compare``) as friendly, guided cards with
plain-language verdicts plus expandable detail tables — so users read the
outcome without opening the raw CSVs.  Populated automatically after a run, or
via "Open a previous comparison…".  Pure UI; no analysis/matplotlib here (the
comparison figure is shown from its saved PNG).
"""
from __future__ import annotations

import os
import csv
import math
from collections import Counter

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QPixmap, QDesktopServices

from firefly.ui.ui_theme import _THEME
from firefly.ui.ui_widgets import (_CollapsibleSection, _AlertBanner,
                                   _color_chip)
from firefly.ui.ui_helpers import _open_folder
from firefly.analysis.fa_stats_config import glossary_def
# Pure formatters shared with the QML ResultsController (single source of truth).
from firefly.ui.results_format import (
    METRIC_DISPLAY as _METRIC_DISPLAY,
    SUMMARY_COLS as _SUMMARY_COLS,
    SUMMARY_NUMERIC as _SUMMARY_NUMERIC,
    _isfinite, _fmt_p, _fmt_num, _pretty_metric, _mag_bucket,
    _effect_phrase, _verdict_for_metric, ordered_metrics as _ordered_metrics_fn)


class _NumItem(QtWidgets.QTableWidgetItem):
    """Table item that sorts numerically by an attached value."""

    def __init__(self, text, value):
        super().__init__(text)
        self.setData(Qt.ItemDataRole.UserRole,
                     float(value) if _isfinite(value) else float("-inf"))
        self.setTextAlignment(Qt.AlignmentFlag.AlignRight
                              | Qt.AlignmentFlag.AlignVCenter)

    def __lt__(self, other):
        try:
            return (self.data(Qt.ItemDataRole.UserRole)
                    < other.data(Qt.ItemDataRole.UserRole))
        except Exception:
            return super().__lt__(other)


def _txt(s):
    it = QtWidgets.QTableWidgetItem("" if s is None else str(s))
    return it


class _ClickableLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal()

    def mousePressEvent(self, ev):
        self.clicked.emit()
        super().mousePressEvent(ev)


class _ResultsView(QtWidgets.QWidget):
    """Scrollable results surface; rebuilt idempotently by `load`."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None
        self._base_dir = ""
        self._open_prev_cb = None
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        outer.addWidget(self._scroll)
        self._body = QtWidgets.QWidget()
        self._body_v = QtWidgets.QVBoxLayout(self._body)
        self._body_v.setContentsMargins(16, 14, 16, 14)
        self._body_v.setSpacing(12)
        self._scroll.setWidget(self._body)
        self._show_placeholder()

    def set_open_previous_callback(self, cb):
        self._open_prev_cb = cb

    # ── body management ──────────────────────────────────────────────────
    def _clear_body(self):
        while self._body_v.count():
            it = self._body_v.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()

    def _show_placeholder(self):
        self._clear_body()
        lbl = QtWidgets.QLabel(
            "Run a comparison — or open a previous one — to see results here.")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(f"color:{_THEME['TXT_MUTED']}; padding:40px;")
        self._body_v.addWidget(lbl)
        if self._open_prev_cb:
            row = QtWidgets.QHBoxLayout()
            row.addStretch(1)
            btn = QtWidgets.QPushButton("Open a previous comparison…")
            btn.clicked.connect(lambda: self._open_prev_cb())
            row.addWidget(btn)
            row.addStretch(1)
            self._body_v.addLayout(row)
        self._body_v.addStretch(1)

    def load(self, results, base_dir):
        self._data = results or {}
        self._base_dir = base_dir or ""
        self._clear_body()
        try:
            self._build_header()
            self._build_figure()
            self._build_metric_cards()
            self._build_replicate_table()
            self._build_twoway()
            self._build_circular()
        except Exception:
            self._body_v.addWidget(_AlertBanner(
                "danger", "Couldn't fully render these results."))
        self._body_v.addStretch(1)
        self._stagger_fade_in()

    def _stagger_fade_in(self):
        """Gently fade the top cards in as the results 'arrive'.  Capped to the
        first few (light) cards — the heavy per-replicate table / circular cards
        near the bottom just appear, so nothing stutters.  No-op under
        reduce-motion."""
        from firefly.ui import ui_anim
        if ui_anim.reduce_motion():
            return
        shown = 0
        for i in range(self._body_v.count()):
            w = self._body_v.itemAt(i).widget()
            if w is None:
                continue
            if shown >= 9:
                break
            ui_anim.fade_in(w, duration=ui_anim.NORMAL, delay=shown * 35)
            shown += 1

    # ── helpers ──────────────────────────────────────────────────────────
    def _card(self, title):
        card = QtWidgets.QFrame()
        card.setObjectName("wizard_card")
        cl = QtWidgets.QVBoxLayout(card)
        cl.setContentsMargins(12, 10, 12, 12)
        cl.setSpacing(8)
        t = QtWidgets.QLabel(title)
        f = t.font(); f.setBold(True); f.setPointSize(12); t.setFont(f)
        t.setStyleSheet(f"color:{_THEME['TXT']};")
        cl.addWidget(t)
        self._body_v.addWidget(card)
        return cl

    def _file(self, key):
        files = (self._data.get("meta") or {}).get("files") or {}
        name = files.get(key)
        return os.path.join(self._base_dir, name) if name else ""

    def _table(self, headers, item_rows, tips=None):
        tbl = QtWidgets.QTableWidget(len(item_rows), len(headers))
        tbl.setHorizontalHeaderLabels(headers)
        if tips:
            for c, tip in enumerate(tips):
                if tip and tbl.horizontalHeaderItem(c):
                    tbl.horizontalHeaderItem(c).setToolTip(tip)
        for r, row in enumerate(item_rows):
            for c, item in enumerate(row):
                tbl.setItem(r, c, item)
        tbl.setSortingEnabled(True)
        tbl.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        tbl.verticalHeader().setVisible(False)
        tbl.horizontalHeader().setStretchLastSection(True)
        tbl.resizeColumnsToContents()
        tbl.setMinimumHeight(min(40 + 24 * max(1, len(item_rows)), 380))
        return tbl

    # ── sections ─────────────────────────────────────────────────────────
    def _build_header(self):
        meta = self._data.get("meta") or {}
        two = bool(meta.get("two_factor"))
        tps = meta.get("timepoints") or []
        n_groups = int(meta.get("n_groups", 0) or 0)
        head = (f"{len(meta.get('group_order') or [])} groups × {len(tps)} "
                f"time points" if (two and tps) else f"{n_groups} groups")
        cl = self._card(f"Comparison results — {head}")

        # design chips (one per card group, with replicate counts)
        summary = self._data.get("summary") or []
        color_map = dict(zip(meta.get("group_labels") or [],
                             meta.get("group_colors") or []))

        def _chip_label(r):
            g = str(r.get("group", "") or "")
            tp = str(r.get("timepoint", "") or "")
            return f"{g} / {tp}" if (two and tp) else g

        order, counts = [], Counter()
        for r in summary:
            lab = _chip_label(r)
            if lab not in counts:
                order.append(lab)
            counts[lab] += 1
        if order:
            chips = QtWidgets.QHBoxLayout()
            chips.setSpacing(6)
            for lab in order:
                chips.addWidget(_color_chip(
                    lab, color_map.get(lab, _THEME["TXT_MUTED"]), counts[lab]))
            chips.addStretch(1)
            cl.addLayout(chips)

        if meta.get("pair_warn"):
            cl.addWidget(_AlertBanner("warn", str(meta["pair_warn"])))

        # config summary (collapsed)
        cfg_rows = self._data.get("config_summary") or []
        if cfg_rows:
            sec = _CollapsibleSection("Statistics configuration")
            grid = QtWidgets.QGridLayout()
            grid.setHorizontalSpacing(14)
            grid.setVerticalSpacing(3)
            for i, pair in enumerate(cfg_rows):
                lbl, val = (pair + ["", ""])[:2]
                k = QtWidgets.QLabel(str(lbl))
                k.setStyleSheet(f"color:{_THEME['TXT_MUTED']};")
                v = QtWidgets.QLabel(str(val))
                v.setStyleSheet(f"color:{_THEME['TXT']};")
                grid.addWidget(k, i, 0)
                grid.addWidget(v, i, 1)
            sec.content_layout.addLayout(grid)
            sec.set_expanded(False)
            cl.addWidget(sec)

        # action buttons
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        if self._base_dir and os.path.isdir(self._base_dir):
            b = QtWidgets.QPushButton("Open output folder")
            b.clicked.connect(lambda: _open_folder(self._base_dir))
            row.addWidget(b)
        for label, key in (("Open PDF report", "report_pdf"),
                           ("Open figure", "png"),
                           ("Open stats CSV", "stats_csv")):
            path = self._file(key)
            if path and os.path.isfile(path):
                b = QtWidgets.QPushButton(label)
                b.clicked.connect(lambda _=False, p=path:
                                  QDesktopServices.openUrl(QUrl.fromLocalFile(p)))
                row.addWidget(b)
        row.addStretch(1)
        if self._open_prev_cb:
            b = QtWidgets.QPushButton("Open a previous comparison…")
            b.clicked.connect(lambda: self._open_prev_cb())
            row.addWidget(b)
        cl.addLayout(row)

    def _build_figure(self):
        png = self._file("png")
        if not png or not os.path.isfile(png):
            return
        pm = QPixmap(png)
        if pm.isNull():
            return
        cl = self._card("Comparison figure  (click to open full size)")
        lbl = _ClickableLabel()
        lbl.setCursor(Qt.CursorShape.PointingHandCursor)
        lbl.setPixmap(pm.scaledToWidth(min(pm.width(), 880),
                                       Qt.TransformationMode.SmoothTransformation))
        lbl.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(png)))
        cl.addWidget(lbl)

    def _ordered_metrics(self):
        return _ordered_metrics_fn(self._data.get("stats") or {})

    def _pairwise_rows(self, rec, n_groups):
        headers = ["Comparison", "Test", "p (raw)", "p (corr.)", "Sig.",
                   "Effect size", "n (A/B)"]
        tips = ["", glossary_def("Welch's t-test"), "Uncorrected p-value",
                "p after multiple-comparison correction",
                "*** p<0.001, ** p<0.01, * p<0.05, ns = not significant",
                glossary_def("Hedges' g") or "Standardized effect size + CI",
                "Replicates per group"]
        rows = []
        omn = rec.get("omnibus") or {}
        if n_groups > 2 and omn:
            es, kind = omn.get("effect_size"), omn.get("effect_size_kind")
            eff = (f"{'η²' if kind == 'eta_sq' else 'ε²'} = {float(es):.3f}"
                   if _isfinite(es) else "—")
            sig = _txt(omn.get("stars") or "ns")
            rows.append([_txt("Overall"), _txt(omn.get("test", "")),
                         _NumItem(_fmt_p(omn.get("p")), omn.get("p")),
                         _txt("—"), sig, _txt(eff), _txt("—")])
        for pw in rec.get("pairwise") or []:
            comp = f"{pw.get('label_i', 'A')} vs {pw.get('label_j', 'B')}"
            if pw.get("family") == "dunnett":
                comp += " (Dunnett)"
            pcorr = pw.get("p_within", pw.get("p"))
            stars = pw.get("stars_within") or pw.get("stars") or "ns"
            sig_item = _txt(stars)
            sig = bool(stars) and stars != "ns"
            sig_item.setForeground(QtGui.QBrush(QtGui.QColor(
                _THEME["SUCCESS"] if sig else _THEME["TXT_MUTED"])))
            _, eff = _effect_phrase(pw)
            rows.append([
                _txt(comp), _txt(pw.get("test", "")),
                _NumItem(_fmt_p(pw.get("p")), pw.get("p")),
                _NumItem(_fmt_p(pcorr), pcorr),
                sig_item, _txt(eff or "—"),
                _txt(f"{pw.get('n_i', '?')}/{pw.get('n_j', '?')}")])
        return headers, rows, tips

    def _build_metric_cards(self):
        stats = self._data.get("stats") or {}
        if not stats:
            return
        n_groups = int((self._data.get("meta") or {}).get("n_groups", 0) or 0)
        hdr = self._card("Per-metric results")
        hdr.addWidget(QtWidgets.QLabel(
            "<span style='color:%s'>Each metric is tested across your groups. "
            "Expand a metric for the full pairwise detail.</span>"
            % _THEME["TXT_MUTED"]))
        for key, disp in self._ordered_metrics():
            rec = stats.get(key) or {}
            cl = self._card(disp)
            sev, html, _under = _verdict_for_metric(disp, rec, n_groups)
            cl.addWidget(_AlertBanner(sev, html))
            if rec.get("pairwise"):
                sec = _CollapsibleSection("Details")
                headers, rows, tips = self._pairwise_rows(rec, n_groups)
                sec.content_layout.addWidget(self._table(headers, rows, tips))
                sec.set_expanded(False)
                cl.addWidget(sec)

    def _build_replicate_table(self):
        summary = self._data.get("summary") or []
        if not summary:
            return
        present = [(k, lbl) for k, lbl in _SUMMARY_COLS if k in summary[0]]
        meta = self._data.get("meta") or {}
        color_map = dict(zip(meta.get("group_labels") or [],
                             meta.get("group_colors") or []))
        two = bool(meta.get("two_factor"))
        rows = []
        for r in summary:
            items = []
            for k, _lbl in present:
                v = r.get(k)
                if k in _SUMMARY_NUMERIC:
                    items.append(_NumItem(_fmt_num(v), v))
                else:
                    it = _txt(v)
                    if k == "group":
                        lab = (f"{r.get('group','')} / {r.get('timepoint','')}"
                               if (two and r.get("timepoint")) else
                               str(r.get("group", "")))
                        col = color_map.get(lab)
                        if col:
                            c = QtGui.QColor(col); c.setAlpha(48)
                            it.setBackground(QtGui.QBrush(c))
                    items.append(it)
            rows.append(items)
        cl = self._card("Per-replicate values  (one row per cell)")
        cl.addWidget(self._table([lbl for _k, lbl in present], rows))

    def _build_twoway(self):
        tw = self._data.get("twoway") or {}
        rows = tw.get("rows") or []
        msg = tw.get("message") or ""
        if not rows and not msg:
            return
        cl = self._card("Two-way ANOVA  (group × time)")
        if msg:
            cl.addWidget(_AlertBanner("info", str(msg)))
        anova = [r for r in rows if r.get("section") == "anova"]
        posthoc = [r for r in rows if r.get("section") == "posthoc"]
        if anova:
            headers = ["Metric", "Effect", "F", "df1", "df2", "p", "p (GG)",
                       "np²", "ε"]
            tips = ["", "", "F-statistic", "", "",
                    "Uncorrected p-value",
                    glossary_def("Greenhouse–Geisser") or "Sphericity-corrected p",
                    "Partial η² (effect size)", "Sphericity epsilon"]
            irows = [[_txt(r.get("metric", "")), _txt(r.get("effect", "")),
                      _NumItem(_fmt_num(r.get("F")), r.get("F")),
                      _NumItem(_fmt_num(r.get("df1")), r.get("df1")),
                      _NumItem(_fmt_num(r.get("df2")), r.get("df2")),
                      _NumItem(_fmt_p(r.get("p_unc")), r.get("p_unc")),
                      _NumItem(_fmt_p(r.get("p_GG")), r.get("p_GG")),
                      _NumItem(_fmt_num(r.get("np2")), r.get("np2")),
                      _NumItem(_fmt_num(r.get("eps")), r.get("eps"))]
                     for r in anova]
            cl.addWidget(self._table(headers, irows, tips))
        if posthoc:
            sec = _CollapsibleSection("Post-hoc simple effects")
            headers = ["Metric", "Contrast", "At", "A", "B", "p", "p (Holm)",
                       "Sig."]
            irows = []
            for r in posthoc:
                stars = r.get("stars") or "ns"
                si = _txt(stars)
                si.setForeground(QtGui.QBrush(QtGui.QColor(
                    _THEME["SUCCESS"] if (stars and stars != "ns")
                    else _THEME["TXT_MUTED"])))
                irows.append([
                    _txt(r.get("metric", "")), _txt(r.get("contrast", "")),
                    _txt(r.get("at", "")), _txt(r.get("level_A", "")),
                    _txt(r.get("level_B", "")),
                    _NumItem(_fmt_p(r.get("p")), r.get("p")),
                    _NumItem(_fmt_p(r.get("p_holm")), r.get("p_holm")), si])
            sec.content_layout.addWidget(self._table(headers, irows))
            sec.set_expanded(False)
            cl.addWidget(sec)

    def _csv_table(self, path):
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                rd = csv.reader(fh)
                rows = list(rd)
        except Exception:
            return None
        if not rows:
            return None
        headers, data = rows[0], rows[1:]
        irows = [[_txt(c) for c in row] for row in data]
        return self._table(headers, irows)

    def _build_circular(self):
        tests = self._file("circular_tests")
        per_rep = self._file("circular_per_replicate")
        have = [p for p in (tests, per_rep) if p and os.path.isfile(p)]
        if not have:
            return
        cl = self._card("Circular statistics  (turning-angle direction)")
        if tests and os.path.isfile(tests):
            sec = _CollapsibleSection("Between-group tests (κ, R̄, μ)")
            t = self._csv_table(tests)
            if t is not None:
                sec.content_layout.addWidget(t)
            sec.set_expanded(False)
            cl.addWidget(sec)
        if per_rep and os.path.isfile(per_rep):
            sec = _CollapsibleSection("Per-replicate values")
            t = self._csv_table(per_rep)
            if t is not None:
                sec.content_layout.addWidget(t)
            sec.set_expanded(False)
            cl.addWidget(sec)
