"""Self-contained custom widgets/dialogs used by the FIREFLY GUI.

Extracted from app_qt.py (#7); re-exported there for compatibility.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from firefly.ui.app_qt import MainWindow  # noqa: F401  (forward ref only)

import os
import sys
import time
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar
from firefly import crash_reporter
from firefly.ui.ui_theme import _THEME, _ACTIVE_THEME_NAME
from firefly.ui.ui_helpers import (_make_cogwheel_icon, _make_close_x_icon,
                        _make_napari_container_layout_opaque,
                        _hide_napari_chrome, _register_motion_colormap,
                        _MOTION_PALETTE, _MOTION_ORDER,
                        _MOTION_CMAP_NAME, _open_folder)
from firefly.analysis.fa_stats_config import glossary_def


class _InfoIcon(QtWidgets.QLabel):
    """A small "ⓘ" affordance that reveals a one-sentence, plain-English
    definition of a technical term on hover (and on click, for trackpad /
    touch users who don't hover).

    The definition text comes from `fa_stats_config.STATS_GLOSSARY` via the
    `term` key.  An unknown term renders nothing (the icon hides itself), so
    callers can pass any label safely.  Used next to statistics terms in the
    Compare tab's "Analysis Configuration" wizard.
    """

    def __init__(self, term: str, parent=None):
        super().__init__("ⓘ", parent)   # ⓘ
        self._term = term
        definition = glossary_def(term)
        if not definition:
            self.hide()
            return
        # Rich-text tooltip names the term it defines, then the sentence.
        self.setToolTip(f"<b>{term}</b><br>{definition}")
        # macOS can suppress tooltips on non-focused widgets — force them
        # (mirrors the fix used for the form-row labels).
        try:
            self.setAttribute(Qt.WidgetAttribute.WA_AlwaysShowToolTips, True)
        except Exception:
            pass
        self.setCursor(Qt.CursorShape.WhatsThisCursor)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(15, 15)
        # Muted by default, accent on hover — a recognisable info affordance
        # without competing with the label text it sits beside.
        self.setStyleSheet(
            "QLabel { color: %s; font-size: 12px; font-weight: bold;"
            " background: transparent; }"
            "QLabel:hover { color: %s; }"
            % (_THEME["TXT_MUTED"], _THEME["ACC"]))

    def mousePressEvent(self, ev):
        # Click anywhere on the icon → pop the definition immediately, anchored
        # just beneath it, so it works without a hover.
        try:
            QtWidgets.QToolTip.showText(
                self.mapToGlobal(QtCore.QPoint(0, self.height())),
                self.toolTip(), self)
        except Exception:
            pass
        super().mousePressEvent(ev)


def _info_icon(term: str, parent=None) -> _InfoIcon:
    """Factory for an `_InfoIcon` explaining `term` (see STATS_GLOSSARY)."""
    return _InfoIcon(term, parent)


def _label_with_info(text: str, term: str, parent=None) -> QtWidgets.QLabel:
    """A form-row label whose plain-English definition appears when you HOVER
    the label text — no separate ⓘ icon (per user preference, the icon is gone
    and the text itself is the affordance).

    The definition comes from `glossary_def(term)`.  When the term has no entry
    the label is just plain text with no tooltip.  A 'help' cursor on hover is
    the only (cursor-only, no visual clutter) hint that an explanation exists."""
    lbl = QtWidgets.QLabel(text, parent)
    definition = glossary_def(term)
    if definition:
        lbl.setToolTip(f"<b>{text.strip()}</b><br>{definition}")
        try:
            lbl.setAttribute(Qt.WidgetAttribute.WA_AlwaysShowToolTips, True)
        except Exception:
            pass
        lbl.setCursor(Qt.CursorShape.WhatsThisCursor)
    return lbl


# ─────────────────────────────────────────────────────────────────────────────
#  Compare-tab "Analysis Configuration" wizard widgets
# ─────────────────────────────────────────────────────────────────────────────

class _AlertBanner(QtWidgets.QFrame):
    """A severity-coloured callout for the data-aware recommendation: a subtle
    panel with a thick coloured left bar + an icon + rich-text message.  Styled
    by the `#alert_banner[severity=...]` QSS rules; the icon colour is read
    live from `_THEME`."""

    _ICON = {"danger": "⚠", "warn": "⚠", "success": "✓", "info": "ⓘ"}
    _ICON_KEY = {"danger": "DANGER", "warn": "WARN",
                 "success": "SUCCESS", "info": "ACC"}

    def __init__(self, severity: str, html: str, parent=None):
        super().__init__(parent)
        self.setObjectName("alert_banner")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 11, 8)
        lay.setSpacing(9)
        self._icon = QtWidgets.QLabel()
        self._icon.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self._icon.setFixedWidth(16)
        lay.addWidget(self._icon)
        self._msg = QtWidgets.QLabel()
        self._msg.setWordWrap(True)
        self._msg.setTextFormat(Qt.TextFormat.RichText)
        self._msg.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self._msg, 1)
        self.set_message(severity, html)

    def set_message(self, severity: str, html: str):
        """Re-skin the banner to a new severity + message (re-polishes the QSS
        so the coloured left bar updates).  Lets a single banner be reused
        rather than rebuilt (e.g. the Preferences GPU-status banner)."""
        sev = severity if severity in self._ICON else "info"
        if self.property("severity") != sev:
            self.setProperty("severity", sev)
            self.style().unpolish(self)
            self.style().polish(self)
        self._icon.setText(self._ICON[sev])
        self._icon.setStyleSheet(
            "color: %s; font-size: 14px; font-weight: bold; "
            "background: transparent;"
            % _THEME.get(self._ICON_KEY[sev], _THEME["ACC"]))
        self._msg.setText(html)


class _StatusBadge(QtWidgets.QLabel):
    """Small run-readiness pill in the wizard header.  `set_state(kind, text)`
    where kind ∈ {ready, blocked, muted} recolours it.

    Styled with an INLINE stylesheet (not the app QSS): the global
    `QLabel { background: transparent }` rule otherwise wins over an app-level
    `background-color`, so a widget-specific stylesheet is the robust way to
    paint the pill."""

    _STYLES = {                       # kind → (bg key, fg key, bordered)
        "ready":   ("SUCCESS",   "ACC_FG",    False),
        "blocked": ("DANGER",    "ACC_FG",    False),
        "warn":    ("WARN",      "ACC_FG",    False),
        "muted":   ("PANEL_ALT", "TXT_MUTED", True),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("status_badge")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.set_state("muted", "—")

    def set_state(self, kind: str, text: str):
        self.setText(text)
        self.setProperty("kind", kind)   # introspectable; styling is inline
        bg, fg, bordered = self._STYLES.get(kind, self._STYLES["muted"])
        border = (f"border: 1px solid {_THEME['BORDER']};"
                  if bordered else "border: none;")
        self.setStyleSheet(
            "QLabel#status_badge { background-color: %s; color: %s; "
            "border-radius: 9px; padding: 2px 11px; font-weight: 600; "
            "font-size: 11px; %s }" % (_THEME[bg], _THEME[fg], border))


class _HyperflyPill(QtWidgets.QLabel):
    """Green status pill shown in the header while a HYPER-FLY (parallel
    multi-file) batch is engaged.

    Matches the **'Update available' pill's shape and size** (radius 10,
    4×10 padding, 11px bold) and the Compare tab's **'Ready to run' pill's
    green** (SUCCESS fill, ACC_FG text).  Static — no animation.  Hidden until
    `engage()`."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("hyperfly_pill")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Fixed vertical size policy so the pill sits at its NATURAL height
        # instead of stretching to the full header-bar height (which made it
        # read as a chunky banner rather than a slim pill like the button
        # beside it).
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                           QtWidgets.QSizePolicy.Policy.Fixed)
        # Retained as no-op attributes for API/back-compat (there is no longer
        # any pulse animation or opacity effect to track).
        self._anim = None
        self._eff = None
        self.setStyleSheet(
            "QLabel#hyperfly_pill { background-color: %s; color: %s; "
            "border: none; border-radius: 10px; padding: 4px 10px; "
            "font-weight: 700; font-size: 11px; }"
            % (_THEME.get("SUCCESS", "#3fb950"), _THEME.get("ACC_FG", "#ffffff")))
        self.hide()

    def engage(self, text: str = "HYPER-FLY engaged"):
        self.setText(text)
        self.show()

    def disengage(self):
        self.hide()


def _step_badge(n, parent=None) -> QtWidgets.QLabel:
    """A small accent-filled number chip ('1'..'5') prefixing a wizard step.
    Inline-styled for the same reason as `_StatusBadge`."""
    lbl = QtWidgets.QLabel(str(n), parent)
    lbl.setObjectName("step_badge")
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setFixedSize(20, 20)
    lbl.setStyleSheet(
        "QLabel#step_badge { background-color: %s; color: %s; border-radius: 10px; "
        "font-weight: 700; font-size: 11px; }" % (_THEME["ACC"], _THEME["ACC_FG"]))
    return lbl


def _color_chip(label: str, color: str, n_reps: int,
                parent=None, show_count: bool = True) -> QtWidgets.QFrame:
    """A rounded chip: a colour swatch + a label, optionally a muted count.
    Used both for the Compare group summary (with replicate counts) and as a
    plain colour legend (`show_count=False`, e.g. the motion-class legend)."""
    chip = QtWidgets.QFrame(parent)
    chip.setObjectName("group_chip")
    lay = QtWidgets.QHBoxLayout(chip)
    lay.setContentsMargins(8, 3, 10, 3)
    lay.setSpacing(6)
    sw = QtWidgets.QLabel()
    sw.setFixedSize(11, 11)
    # Inline stylesheet wins over the global `QLabel { background: transparent }`
    # for this specific widget, so the swatch shows the supplied colour.
    sw.setStyleSheet(
        "background: %s; border-radius: 5px;" % (color or _THEME["TXT_MUTED"]))
    lay.addWidget(sw)
    name = QtWidgets.QLabel(str(label) or "Group")
    lay.addWidget(name)
    if show_count:
        cnt = QtWidgets.QLabel("n=%d" % int(n_reps))
        cnt.setStyleSheet(
            "color: %s; background: transparent;" % _THEME["TXT_MUTED"])
        lay.addWidget(cnt)
    return chip


class _DecisionDiagram(QtWidgets.QWidget):
    """Native (QPainter) decision flow for how the statistical test is chosen —
    replaces the old matplotlib→QPixmap diagram.  Painted in logical
    coordinates so it stays retina-crisp at any size, reads `_THEME` live, and
    is on-palette (the result box uses the accent, not an off-theme green).

    `set_flow(n_groups, paired, strat, result_text)` updates it.  Nodes are
    rebuilt every paint into `self._nodes` so hovering shows a one-line tooltip
    and clicking emits `node_clicked` (room to grow into richer interactivity)."""

    node_clicked = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._n_groups = 0
        self._paired = False
        self._strat = "auto"
        self._result = "—"
        self._nodes: list = []           # (QRectF, label, tip)
        self.setMinimumHeight(165)
        self.setMouseTracking(True)

    def sizeHint(self):
        return QtCore.QSize(660, 175)

    def set_flow(self, n_groups, paired, strat, result_text):
        self._n_groups = int(n_groups or 0)
        self._paired = bool(paired)
        self._strat = str(strat or "auto")
        self._result = str(result_text or "—")
        self.update()

    # ── tooltips for each node (falls back to the shared glossary) ──
    def _tip(self, label):
        tips = {
            "2 groups": "Exactly two groups → a two-sample test.",
            "3+ groups":
                "Three or more groups → an omnibus test + pairwise follow-ups.",
            "Unpaired":
                "Independent groups; no cell appears in more than one group.",
            "Paired": "The same cells measured at two or more time points.",
            "Parametric": glossary_def("Parametric"),
            "Non-parametric":
                "Rank-based; makes no normal-distribution assumption.",
        }
        return tips.get(label) or glossary_def(label) or ""

    def paintEvent(self, _ev):
        T = _THEME
        W, H = self.width(), self.height()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QtGui.QColor(T["PANEL"]))
        self._nodes = []

        strat = self._strat
        n = self._n_groups
        paired = self._paired

        col_x = [0.135 * W, 0.350 * W, 0.560 * W]
        res_cx = 0.850 * W
        pill_w = max(104.0, 0.180 * W)
        pill_h = max(30.0, min(40.0, 0.225 * H))
        y_title = 0.115 * H
        y_top = 0.430 * H
        y_bot = 0.730 * H
        y_mid = 0.5 * (y_top + y_bot)

        base = self.font()

        def _title(x, text):
            f = QtGui.QFont(base); f.setPointSizeF(8.5); f.setBold(False)
            p.setFont(f); p.setPen(QtGui.QColor(T["TXT_MUTED"]))
            r = QtCore.QRectF(x - pill_w / 2, y_title - 10, pill_w, 20)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)

        def _pill(x, yc, text, active):
            r = QtCore.QRectF(x - pill_w / 2, yc - pill_h / 2, pill_w, pill_h)
            fill = QtGui.QColor(T["ACC"] if active else T["PANEL_ALT"])
            edge = QtGui.QColor(T["ACC"] if active else T["BORDER"])
            tcol = QtGui.QColor(T["ACC_FG"] if active else T["TXT_MUTED"])
            p.setPen(QtGui.QPen(edge, 1.6 if active else 1.0))
            p.setBrush(QtGui.QBrush(fill))
            p.drawRoundedRect(r, pill_h / 2, pill_h / 2)
            f = QtGui.QFont(base); f.setPointSizeF(9.5); f.setBold(active)
            p.setFont(f); p.setPen(QtGui.QPen(tcol))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter
                       | Qt.TextFlag.TextWordWrap, text)
            self._nodes.append((r, text, self._tip(text)))

        def _arrow(x0, x1, y):
            p.setPen(QtGui.QPen(QtGui.QColor(T["BORDER_HI"]), 1.6))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawLine(QtCore.QPointF(x0, y), QtCore.QPointF(x1, y))
            ah = 5.0
            p.drawLine(QtCore.QPointF(x1, y), QtCore.QPointF(x1 - ah, y - ah))
            p.drawLine(QtCore.QPointF(x1, y), QtCore.QPointF(x1 - ah, y + ah))

        # Column titles
        _title(col_x[0], "Groups")
        _title(col_x[1], "Design")
        _title(col_x[2], "Distribution")

        # Connecting arrows (drawn first, behind the pills).  The last arrow
        # stops a clear gap short of the result box so its head never touches /
        # overlaps the box border.
        half = pill_w / 2
        res_w = max(120.0, 0.21 * W)
        _GAP = 12.0
        _arrow(col_x[0] + half, col_x[1] - half, y_mid)
        _arrow(col_x[1] + half, col_x[2] - half, y_mid)
        _arrow(col_x[2] + half, res_cx - res_w / 2 - _GAP, y_mid)

        # Column 1 — number of groups
        _pill(col_x[0], y_top, "2 groups", n == 2)
        _pill(col_x[0], y_bot, "3+ groups", n >= 3)
        # Column 2 — design
        _pill(col_x[1], y_top, "Unpaired", (not paired) and n >= 2)
        _pill(col_x[1], y_bot, "Paired", paired)
        # Column 3 — distribution
        _pill(col_x[2], y_top, "Parametric",
              strat in ("auto", "force_parametric"))
        _pill(col_x[2], y_bot, "Non-parametric",
              strat in ("auto", "force_nonparametric"))
        if strat == "auto":
            f = QtGui.QFont(base); f.setPointSizeF(7.5); p.setFont(f)
            p.setPen(QtGui.QColor(T["TXT_MUTED"]))
            r = QtCore.QRectF(col_x[2] - pill_w / 2, y_bot + pill_h / 2 + 1,
                              pill_w, 14)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter,
                       "auto: per metric")

        # Result box (accent border — on-theme, replacing the old green)
        res_h = max(pill_h * 1.7, (y_bot - y_top) + pill_h)
        rr = QtCore.QRectF(res_cx - res_w / 2, y_mid - res_h / 2, res_w, res_h)
        p.setPen(QtGui.QPen(QtGui.QColor(T["ACC"]), 2.0))
        p.setBrush(QtGui.QBrush(QtGui.QColor(T["PANEL_ALT"])))
        p.drawRoundedRect(rr, 9, 9)
        f = QtGui.QFont(base); f.setPointSizeF(10.5); f.setBold(True)
        p.setFont(f); p.setPen(QtGui.QPen(QtGui.QColor(T["TXT"])))
        p.drawText(rr, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                   self._result)
        self._nodes.append((rr, self._result, glossary_def(self._result) or ""))
        p.end()

    def _node_at(self, pos):
        for rect, label, tip in self._nodes:
            if rect.contains(pos):
                return label, tip
        return None, None

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        _label, tip = self._node_at(pos)
        if tip:
            QtWidgets.QToolTip.showText(ev.globalPosition().toPoint(), tip, self)
        else:
            QtWidgets.QToolTip.hideText()
        super().mouseMoveEvent(ev)

    def mousePressEvent(self, ev):
        label, tip = self._node_at(ev.position())
        if label:
            self.node_clicked.emit(label)
            if tip:
                QtWidgets.QToolTip.showText(
                    ev.globalPosition().toPoint(), tip, self)
        super().mousePressEvent(ev)


class _PipelineDiagram(QtWidgets.QWidget):
    """A compact left-to-right map of the analysis stages shown in the run
    cockpit: done stages in success-green, the running stage in accent, pending
    stages muted.  Native QPainter (DPR-crisp), reads `_THEME` live, hover for a
    one-line description.  Driven by the worker's progress messages via
    `set_stage_from_msg()` — the index only ever advances (the worker's progress
    %s are non-monotonic across stages, so we never regress)."""

    _STAGES = ["Preprocess", "Detect", "Link", "Drift", "Diffuse", "Classify"]
    _TIPS = {
        "Preprocess": "Load the image stack (or localisations) and remove "
                      "background.",
        "Detect":     "Find candidate emitters frame by frame.",
        "Link":       "Connect detections across frames into trajectories.",
        "Drift":      "Correct sample/stage drift (when enabled).",
        "Diffuse":    "Build MSD curves and fit diffusion (D, α).",
        "Classify":   "Motion classes, clustering, secondary metrics, and "
                      "saving the outputs.",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._idx = -1          # furthest-reached active stage (-1 = idle)
        self._complete = False
        self._nodes: list = []  # (QRectF, label, tip)
        self.setMinimumHeight(56)
        self.setMouseTracking(True)

    def sizeHint(self):
        return QtCore.QSize(560, 60)

    def reset(self):
        self._idx = -1
        self._complete = False
        self.update()

    def set_complete(self):
        self._idx = len(self._STAGES) - 1
        self._complete = True
        self.update()

    @classmethod
    def _index_for_msg(cls, msg):
        """Return ('complete', None), ('idx', i) or (None, None) for a worker
        progress message.  Checked most-specific first so e.g. 'Linking…' never
        falls through to the generic 'loading' bucket."""
        m = (msg or "").lower()
        if any(k in m for k in ("complete", "all done", "batch complete")):
            return ("complete", None)
        if any(k in m for k in ("saving", "rendering", "secondary",
                                "cluster", "classif")):
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

    def set_stage_from_msg(self, msg):
        kind, idx = self._index_for_msg(msg)
        if kind == "complete":
            self.set_complete()
        elif kind == "idx" and idx is not None and idx > self._idx:
            self._idx = idx
            self._complete = False
            self.update()

    def paintEvent(self, _ev):
        T = _THEME
        W, H = self.width(), self.height()
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QtGui.QColor(T["PANEL"]))
        self._nodes = []
        n = len(self._STAGES)
        gap = 16.0
        node_w = max(54.0, (W - (n - 1) * gap - 8) / n)
        node_h = max(26.0, min(34.0, H - 18))
        yc = H / 2.0
        base = self.font()
        x = 4.0
        for i, name in enumerate(self._STAGES):
            r = QtCore.QRectF(x, yc - node_h / 2, node_w, node_h)
            if self._complete or i < self._idx:
                fill, edge, tcol = T["SUCCESS"], T["SUCCESS"], T["ACC_FG"]
            elif i == self._idx:
                fill, edge, tcol = T["ACC"], T["ACC"], T["ACC_FG"]
            else:
                fill, edge, tcol = T["PANEL_ALT"], T["BORDER"], T["TXT_MUTED"]
            p.setPen(QtGui.QPen(QtGui.QColor(edge), 1.4))
            p.setBrush(QtGui.QBrush(QtGui.QColor(fill)))
            p.drawRoundedRect(r, 6, 6)
            f = QtGui.QFont(base); f.setPointSizeF(8.5)
            f.setBold(self._complete or i <= self._idx)
            p.setFont(f); p.setPen(QtGui.QPen(QtGui.QColor(tcol)))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter
                       | Qt.TextFlag.TextWordWrap, name)
            self._nodes.append((r, name, self._TIPS.get(name, "")))
            x += node_w + gap
        # Connecting arrows in the gaps.
        p.setPen(QtGui.QPen(QtGui.QColor(T["BORDER_HI"]), 1.4))
        p.setBrush(Qt.BrushStyle.NoBrush)
        for i in range(len(self._nodes) - 1):
            x0 = self._nodes[i][0].right()
            x1 = self._nodes[i + 1][0].left()
            p.drawLine(QtCore.QPointF(x0 + 1, yc), QtCore.QPointF(x1 - 1, yc))
            ah = 3.5
            p.drawLine(QtCore.QPointF(x1 - 1, yc),
                       QtCore.QPointF(x1 - 1 - ah, yc - ah))
            p.drawLine(QtCore.QPointF(x1 - 1, yc),
                       QtCore.QPointF(x1 - 1 - ah, yc + ah))
        p.end()

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        shown = False
        for rect, _label, tip in self._nodes:
            if rect.contains(pos) and tip:
                QtWidgets.QToolTip.showText(
                    ev.globalPosition().toPoint(), tip, self)
                shown = True
                break
        if not shown:
            QtWidgets.QToolTip.hideText()
        super().mouseMoveEvent(ev)


class _StackedBar(QtWidgets.QWidget):
    """A thin horizontal bar split into proportional coloured segments — e.g.
    the motion-class composition (Immobile / Confined / Brownian / Directed).
    `segments` is a list of (label, value, colour_hex).  Native QPainter
    (DPR-crisp), rounded ends, hover a segment for its label + count + %."""

    def __init__(self, segments=None, parent=None):
        super().__init__(parent)
        self._segs = list(segments or [])
        self.setFixedHeight(16)
        self.setMouseTracking(True)

    def set_segments(self, segments):
        self._segs = list(segments or [])
        self.update()

    def _total(self):
        return sum(max(0.0, float(v)) for _l, v, _c in self._segs) or 1.0

    def paintEvent(self, _ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        W, H = float(self.width()), float(self.height())
        path = QtGui.QPainterPath()
        path.addRoundedRect(0.0, 0.0, W, H, H / 2, H / 2)
        p.setClipPath(path)
        p.fillRect(self.rect(), QtGui.QColor(_THEME["PANEL_ALT"]))
        total = self._total()
        x = 0.0
        for _label, v, col in self._segs:
            w = W * (max(0.0, float(v)) / total)
            p.fillRect(QtCore.QRectF(x, 0.0, w + 0.5, H),
                       QtGui.QColor(col or _THEME["TXT_MUTED"]))
            x += w
        p.end()

    def mouseMoveEvent(self, ev):
        W = float(self.width()) or 1.0
        total = self._total()
        x = ev.position().x()
        acc = 0.0
        for label, v, _c in self._segs:
            w = W * (max(0.0, float(v)) / total)
            if acc <= x < acc + w:
                pct = 100.0 * float(v) / total
                QtWidgets.QToolTip.showText(
                    ev.globalPosition().toPoint(),
                    f"{label}: {int(v):,} ({pct:.1f}%)", self)
                break
            acc += w
        super().mouseMoveEvent(ev)


class _UpdateCheckThread(QtCore.QThread):
    """Background thread that asks GitHub for the latest release.

    Two modes:
      • auto  (force=False) — emits ``update_available(tag, release)``
        ONLY when the latest tag is newer than the running version;
        silent on every other outcome (offline, up-to-date, error).
        Drives the startup check that lights the header pill.
      • force (force=True)  — emits ``check_finished(release_or_None)``
        ALWAYS (the parsed release dict, or None if GitHub was
        unreachable) so a user-triggered "Check for updates…" can show a
        result even when already up to date.

    ``release`` is the dict from ``updater.parse_release`` (tag / html_url
    / body / asset).  All network + parsing is delegated to
    ``firefly.updater`` so the version-compare logic lives in one place.
    """

    update_available = QtCore.Signal(str, object)   # (tag, release dict)
    check_finished   = QtCore.Signal(object)        # release dict or None

    def __init__(self, api_url: str, current_version: str,
                 parent=None, force: bool = False):
        super().__init__(parent)
        self._api_url = api_url
        self._current = current_version
        self._force = force

    @staticmethod
    def _parse_version(s: str) -> "tuple[int, ...]":
        """Delegates to the single canonical comparator in firefly.updater."""
        from firefly import updater
        return updater.parse_version(s)

    def run(self):
        try:
            from firefly import updater
        except Exception:
            if self._force:
                self.check_finished.emit(None)
            return
        rel = updater.fetch_latest_release(self._api_url)
        info = updater.parse_release(rel) if rel else None
        if self._force:
            self.check_finished.emit(info)      # may be None on failure
            return
        if not info:
            return
        tag = info.get("tag") or ""
        if tag and updater.is_newer(tag, self._current):
            self.update_available.emit(tag, info)


class _UpdateWorker(QtCore.QObject):
    """Downloads the release asset on a background QThread.  Emits
    ``progress(pct, label)`` / ``finished(path)`` / ``failed(msg)`` back
    to the GUI thread.  Network + file I/O only — never touches napari —
    so a QThread is safe here."""

    progress = QtCore.Signal(int, str)
    finished = QtCore.Signal(str)
    failed   = QtCore.Signal(str)

    def __init__(self, asset: dict, cancel_check):
        super().__init__()
        self._asset = asset
        self._cancel_check = cancel_check

    @QtCore.Slot()
    def run(self):
        from firefly import updater
        self.progress.emit(0, "Connecting to GitHub…")
        try:
            def _cb(done, total):
                mb = done / 1e6
                if total > 0:
                    pct = int(done * 100 / total)
                    self.progress.emit(
                        pct, f"Downloading… {mb:.0f} / {total/1e6:.0f} MB")
                else:
                    self.progress.emit(0, f"Downloading… {mb:.0f} MB")

            def _status(msg):
                # Retry-backoff messages ("Server busy — retrying in Ns…")
                # so the bar/label keep moving during a transient 504.
                self.progress.emit(0, msg)

            path = updater.download_asset(
                self._asset, progress_cb=_cb, cancel_cb=self._cancel_check,
                status_cb=_status)
            self.finished.emit(path)
        except Exception as exc:
            self.failed.emit(str(exc))


class _UpdateDialog(QtWidgets.QDialog):
    """Software-update dialog.

    Shows current vs latest version + release notes.  On a frozen build
    with a matching release asset it offers a one-click "Update now" that
    downloads the new build (progress + cancel) and hands off to the
    MainWindow to install + relaunch.  Otherwise it just points the user
    at the release page.

    Construct with the parsed release dict from ``updater.parse_release``
    (or None if the check itself failed)."""

    def __init__(self, main_window: "MainWindow", current_version: str,
                 release):
        super().__init__(main_window)
        self._main = main_window
        self._current = current_version
        self._release = release or {}
        self._tag = self._release.get("tag") or ""
        self._asset = self._release.get("asset")
        self._html_url = (self._release.get("html_url")
                          or getattr(main_window, "_UPDATE_RELEASES_URL", ""))
        self._downloaded_path = None
        self._thread = None
        self._worker = None
        self._relay = None
        self._cancel_event = None
        self._can_auto = False

        self.setWindowTitle("FIREFLY — Software Update")
        self.setModal(True)
        self.setMinimumWidth(560)
        self._build()

    # ── construction ──────────────────────────────────────────────────
    def _build(self):
        from firefly import updater
        newer = bool(self._tag) and updater.is_newer(self._tag, self._current)
        self._can_auto = bool(newer and updater.is_frozen() and self._asset)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(22, 22, 22, 18)
        v.setSpacing(12)

        if not self._release:
            head, sub = ("Couldn't check for updates",
                         "FIREFLY couldn't reach GitHub. Check your internet "
                         "connection and try again.")
        elif newer:
            head, sub = (f"FIREFLY {self._tag} is available",
                         f"You're currently on {self._current}.")
        else:
            head, sub = ("You're up to date",
                         f"FIREFLY {self._current} is the latest version.")

        title = QtWidgets.QLabel(head)
        tf = title.font(); tf.setBold(True); tf.setPointSize(16)
        title.setFont(tf)
        title.setStyleSheet(f"color: {_THEME['TXT']};")
        v.addWidget(title)
        sub_lbl = QtWidgets.QLabel(sub)
        sub_lbl.setWordWrap(True)
        sub_lbl.setStyleSheet(f"color: {_THEME['TXT_MUTED']};")
        v.addWidget(sub_lbl)

        body = self._release.get("body") if self._release else ""
        if body:
            notes_lbl = QtWidgets.QLabel("Release notes:")
            notes_lbl.setStyleSheet(f"color: {_THEME['TXT_MUTED']}; "
                                    f"font-weight: 600; padding-top: 4px;")
            v.addWidget(notes_lbl)
            notes = QtWidgets.QTextBrowser()
            notes.setOpenExternalLinks(True)
            notes.setPlainText(body)
            notes.setMinimumHeight(170)
            v.addWidget(notes, 1)

        # Download progress (hidden until "Update now").
        self._status = QtWidgets.QLabel("")
        self._status.setWordWrap(True)
        self._status.setVisible(False)
        self._status.setStyleSheet(f"color: {_THEME['TXT_MUTED']};")
        v.addWidget(self._status)
        self._progress = QtWidgets.QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        v.addWidget(self._progress)

        if self._can_auto:
            note = QtWidgets.QLabel(self._platform_note())
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {_THEME['TXT_MUTED']}; "
                               f"font-style: italic; font-size: 11px;")
            v.addWidget(note)
        elif newer and updater.is_frozen() and not self._asset:
            # A newer release exists, but the installer for THIS platform
            # isn't on it yet — the macOS and Windows builds upload a few
            # minutes apart, so there's a brief window where one OS sees
            # "update available" with nothing to download.  Say so plainly
            # instead of silently offering only the release page.
            hint = QtWidgets.QLabel(
                "The installer for your platform is still being published "
                "(the macOS and Windows builds finish a few minutes apart). "
                "Please try again shortly — Preferences → Updates → "
                "“Check for updates now”.")
            hint.setWordWrap(True)
            hint.setStyleSheet(f"color: {_THEME['TXT_MUTED']}; "
                               f"font-style: italic;")
            v.addWidget(hint)
        elif newer and not updater.is_frozen():
            hint = QtWidgets.QLabel(
                "This is a from-source install — update with "
                "<code>git pull</code>, or use the release page.")
            hint.setTextFormat(Qt.TextFormat.RichText)
            hint.setWordWrap(True)
            hint.setStyleSheet(f"color: {_THEME['TXT_MUTED']}; "
                               f"font-style: italic;")
            v.addWidget(hint)

        # Button row.
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self._btn_page = QtWidgets.QPushButton("View release page")
        self._btn_page.clicked.connect(self._open_page)
        row.addWidget(self._btn_page)
        row.addStretch(1)

        self._btn_cancel = QtWidgets.QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self._on_cancel)
        self._btn_cancel.setVisible(False)
        row.addWidget(self._btn_cancel)

        self._btn_skip = self._btn_later = self._btn_update = None
        self._btn_close = None
        if self._can_auto:
            self._btn_skip = QtWidgets.QPushButton("Skip this version")
            self._btn_skip.clicked.connect(self._on_skip)
            row.addWidget(self._btn_skip)
            self._btn_later = QtWidgets.QPushButton("Later")
            self._btn_later.clicked.connect(self.reject)
            row.addWidget(self._btn_later)
            self._btn_update = QtWidgets.QPushButton("Update now")
            self._btn_update.setDefault(True)
            self._btn_update.clicked.connect(self._on_update_now)
            row.addWidget(self._btn_update)
        else:
            self._btn_close = QtWidgets.QPushButton("Close")
            self._btn_close.setDefault(True)
            self._btn_close.clicked.connect(self.reject)
            row.addWidget(self._btn_close)
        v.addLayout(row)

    def _platform_note(self) -> str:
        if sys.platform == "win32":
            return ("FIREFLY will download the new version and restart. "
                    "Windows may show a SmartScreen prompt — choose "
                    "“More info → Run anyway”.")
        return ("FIREFLY will download the new version, replace itself, and "
                "restart automatically.")

    # ── actions ───────────────────────────────────────────────────────
    def _open_page(self):
        try:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(self._html_url))
        except Exception:
            pass

    def _on_skip(self):
        try:
            QtCore.QSettings("jacoblevers", "FIREFLY").setValue(
                "updates/skip_version", self._tag)
        except Exception:
            pass
        self.reject()

    def _on_update_now(self):
        import threading
        if not self._asset:
            QtWidgets.QMessageBox.warning(
                self, "No installer",
                "No installer is available for your platform.")
            return
        self._cancel_event = threading.Event()
        for b in (self._btn_skip, self._btn_later, self._btn_update,
                  self._btn_page):
            if b is not None:
                b.setVisible(False)
        self._btn_cancel.setVisible(True)
        self._btn_cancel.setEnabled(True)
        self._status.setVisible(True)
        self._status.setText("Connecting to GitHub…")
        self._progress.setVisible(True)
        self._progress.setValue(0)

        thread = QtCore.QThread(self)
        worker = _UpdateWorker(self._asset, self._cancel_event.is_set)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        # Relay so the worker's signals are delivered on the GUI thread
        # (real @Slot methods → AutoConnection picks QueuedConnection).
        dlg = self

        class _Relay(QtCore.QObject):
            @QtCore.Slot(int, str)
            def on_progress(self, pct, label):
                dlg._on_progress(pct, label)

            @QtCore.Slot(str)
            def on_finished(self, path):
                dlg._on_finished(path)

            @QtCore.Slot(str)
            def on_failed(self, msg):
                dlg._on_failed(msg)

        relay = _Relay()
        worker.progress.connect(relay.on_progress)
        worker.finished.connect(relay.on_finished)
        worker.failed.connect(relay.on_failed)

        self._thread = thread
        self._worker = worker
        self._relay = relay
        try:
            self._main._update_dl_thread = thread
        except Exception:
            pass
        thread.start()

    def _on_cancel(self):
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._btn_cancel.setEnabled(False)
        self._status.setText("Cancelling…")

    def _on_progress(self, pct, label):
        try:
            self._status.setText(label)
            self._progress.setValue(max(0, min(100, int(pct))))
        except Exception:
            pass

    def _restore_action_buttons(self):
        self._btn_cancel.setVisible(False)
        self._progress.setVisible(False)
        for b in (self._btn_skip, self._btn_later, self._btn_update,
                  self._btn_page):
            if b is not None:
                b.setVisible(True)

    def _cleanup_thread(self):
        try:
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait(2000)
        except Exception:
            pass
        self._thread = None
        self._worker = None
        self._relay = None
        try:
            self._main._update_dl_thread = None
        except Exception:
            pass

    def _on_finished(self, path):
        self._cleanup_thread()
        self._downloaded_path = path
        self._progress.setValue(100)
        self._status.setText("Download complete — restarting FIREFLY…")
        ok = False
        try:
            ok = bool(self._main._apply_downloaded_update(path))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Update failed", str(exc))
        if ok:
            self.accept()          # the app is quitting to swap + relaunch
        else:
            self._restore_action_buttons()
            self._status.setText(
                "Couldn't install the update automatically — the downloaded "
                "installer was revealed so you can finish manually.")

    def _on_failed(self, msg):
        self._cleanup_thread()
        self._restore_action_buttons()
        cancelled = "cancel" in (msg or "").lower()
        self._status.setText("Download cancelled." if cancelled
                             else f"Download failed: {msg}")
        if not cancelled:
            box = QtWidgets.QMessageBox(
                QtWidgets.QMessageBox.Icon.Warning,
                "Update download failed", msg or "Unknown error",
                QtWidgets.QMessageBox.StandardButton.Ok, self)
            box.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            box.exec()

    def reject(self):
        # If a download is in flight, cancel + tear it down before closing.
        if self._thread is not None:
            if self._cancel_event is not None:
                self._cancel_event.set()
            self._cleanup_thread()
        super().reject()


class _ModeTile(QtWidgets.QFrame):
    """A big card-shaped clickable tile.  Acts like a checkable button:
    clicking toggles its state and emits `toggled(bool)`.  Used by the
    Import-tab Single-file / Batch mode toggle.

    QPushButton can't render rich-text (HTML in setText shows literally),
    so a custom QFrame with child QLabels is the cleanest way to get a
    button with multi-line bold-title + muted-subtitle styling.
    """
    toggled = QtCore.Signal(bool)

    def __init__(self, title: str, subtitle: str,
                 icon_char: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("mode_tile")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setProperty("checked", False)
        self._checked = False

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(2)

        title_text = f"{icon_char}  {title}" if icon_char else title
        self._title_lbl = QtWidgets.QLabel(title_text)
        self._title_lbl.setObjectName("mode_tile_title")
        self._title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(self._title_lbl)

        self._sub_lbl = QtWidgets.QLabel(subtitle)
        self._sub_lbl.setObjectName("mode_tile_subtitle")
        self._sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sub_lbl.setWordWrap(True)
        v.addWidget(self._sub_lbl)

        self.setMinimumHeight(82)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool):
        if self._checked == bool(checked):
            return
        self._checked = bool(checked)
        self.setProperty("checked", self._checked)
        # Re-evaluate QSS so the :checked-state border kicks in
        self.style().unpolish(self)
        self.style().polish(self)
        self.toggled.emit(self._checked)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and not self._checked:
            self.setChecked(True)
        super().mousePressEvent(e)


class _ActionTile(QtWidgets.QFrame):
    """Large clickable card for the landing page.  Title + multi-line
    description + optional icon glyph.  Emits `clicked` when the user
    clicks anywhere on the tile."""
    clicked = QtCore.Signal()

    def __init__(self, title: str, description: str, icon_char: str = "",
                 parent=None):
        super().__init__(parent)
        self.setObjectName("action_tile")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(150)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(8)

        if icon_char:
            ico = QtWidgets.QLabel(icon_char)
            ico.setObjectName("action_tile_icon")
            ico.setAlignment(Qt.AlignmentFlag.AlignLeft)
            v.addWidget(ico)

        ttl = QtWidgets.QLabel(title)
        ttl.setObjectName("action_tile_title")
        ttl.setAlignment(Qt.AlignmentFlag.AlignLeft)
        v.addWidget(ttl)

        desc = QtWidgets.QLabel(description)
        desc.setObjectName("action_tile_desc")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignLeft)
        v.addWidget(desc)
        v.addStretch(1)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(e)


class _QuietSpinBox(QtWidgets.QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    def wheelEvent(self, e):
        # Pass to parent so the sidebar can scroll; don't change the value
        e.ignore()


class _QuietDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    def wheelEvent(self, e):
        e.ignore()


class _QuietComboBox(QtWidgets.QComboBox):
    """QComboBox that doesn't change its selection on mouse wheel.

    Same rationale as the spinbox subclasses: when the sidebar's scroll
    area gets a wheel event over a combo, the combo would silently
    change its value before the parent ever sees the wheel.  Override
    wheelEvent to ignore so the wheel bubbles up to the QScrollArea
    instead.  To deliberately change the value, click the dropdown.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Without StrongFocus the combo can also change via arrow keys
        # only after a mouse click anyway — fine for our usage.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Don't let a long item dictate the combo's width.  The default
        # AdjustToContentsOnFirstShow sizes the box to its WIDEST item, so a
        # long entry (e.g. "Torch — GPU (auto device)") pushes the whole
        # sidebar form past the viewport and the user can scroll it sideways.
        # Size to a small minimum instead and let the layout stretch the box to
        # fill its column; the dropdown popup still shows every item in full.
        self.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy
            .AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(6)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Fixed)
    def wheelEvent(self, e):
        e.ignore()


class _CollapsibleSection(QtWidgets.QWidget):
    """A section with a clickable header (with ▶/▼ arrow) that toggles
    visibility of its content panel below.

    Usage:
        sec = _CollapsibleSection("My Section")
        form = QtWidgets.QFormLayout()
        sec.content_layout.addLayout(form)
        form.addRow("Field", widget)
        parent_layout.addWidget(sec)
    """
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        # Escape any literal ampersands once at construction.  QToolButton
        # (like every other Qt button-like widget) treats `&` in its text
        # as a keyboard-shortcut marker — "Diffusion & motion classification"
        # would render as "Diffusion _motion classification" (m underlined,
        # Alt+M activates the button).  Doubling the ampersand is the
        # documented way to display a literal `&`.
        self._title = title.replace("&", "&&")

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._header = QtWidgets.QToolButton()
        self._header.setObjectName("section_header")
        self._header.setText(f"▾  {self._title}")
        self._header.setCheckable(True)
        self._header.setChecked(True)
        self._header.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed)
        self._header.toggled.connect(self._on_toggled)
        # Optional status chip on the right of the header (see set_badge), so a
        # section's active state reads even when collapsed.  Hidden by default →
        # zero width → the header bar stays full-width for sections that never
        # set a badge.  Placed BESIDE the toolbutton (not over it) so header
        # clicks still toggle the section.
        self._badge = QtWidgets.QLabel()
        self._badge.setObjectName("section_badge")
        self._badge.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed,
                                  QtWidgets.QSizePolicy.Policy.Fixed)
        self._badge.hide()
        _hrow = QtWidgets.QHBoxLayout()
        _hrow.setContentsMargins(0, 0, 0, 0)
        _hrow.setSpacing(4)
        _hrow.addWidget(self._header, 1)
        _hrow.addWidget(self._badge, 0, Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter)
        outer.addLayout(_hrow)

        self._content = QtWidgets.QFrame()
        self._content.setObjectName("section_content")
        self._content_layout = QtWidgets.QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(10, 8, 10, 10)
        self._content_layout.setSpacing(6)
        outer.addWidget(self._content)
        # User-driven header clicks animate; programmatic set_expanded stays
        # instant (so e.g. the Results tab doesn't animate every section on load).
        self._animate_next = True

    def _on_toggled(self, checked: bool):
        self._header.setText(f"{'▾' if checked else '▸'}  {self._title}")
        if self._animate_next:
            self._animate_content(checked)
        else:
            self._animate_next = True
            self._content.setVisible(checked)
            self._content.setMaximumHeight(16777215)   # QWIDGETSIZE_MAX

    def _content_target_height(self) -> int:
        """The content frame's TRUE expanded height at its current width.

        Measured via the layout's ``heightForWidth`` — NOT ``c.height()`` and
        NOT ``sizeHint()``:

          * ``sizeHint().height()`` is width-agnostic, so for word-wrapped
            labels it badly under-counts (it assumes a default, much narrower
            width) → the section would "teleport" to its real height when the
            cap is lifted at the end.
          * ``c.height()`` is wrong at expand time too: the content is freshly
            shown but ``self`` (the section) still has its *collapsed*
            geometry, because the parent (wizard) layout hasn't re-flowed yet —
            so the outer layout clamps ``c.height()`` to the small collapsed
            size and the animation slides to a too-short target, then springs.

        ``heightForWidth`` asks the layout directly and is independent of the
        not-yet-reflowed parent, so the target is correct on the first frame.
        """
        c = self._content
        inner = c.layout()
        w = c.width()
        h = -1
        if inner is not None and w > 0 and inner.hasHeightForWidth():
            h = inner.heightForWidth(w)
        if h <= 0:                                   # no word-wrap content
            h = max(c.height(), c.sizeHint().height())
        return max(0, h)

    def _animate_content(self, expand: bool):
        from firefly.ui import ui_anim
        c = self._content
        if expand:
            # Cap the content to ZERO *before* showing it, so the enclosing
            # layout — frequently a QScrollArea with widgetResizable=True (the
            # Compare/Results panels) — never sees a full-height body, not even
            # for one synchronous frame.
            #
            # The previous implementation briefly uncapped to QWIDGETSIZE_MAX to
            # "give the content its real width" before measuring, then re-capped
            # to 0.  That single full-height layout pass made the scroll area
            # lurch: the viewport jumped (content above leapt off-screen and
            # back) and a blank box flashed before the text painted — exactly
            # the jank reported on the "What these terms mean" section.
            #
            # Width is set by the *horizontal* layout and is independent of
            # maximumHeight, so a capped-height layout pass yields the correct
            # width — and therefore the correct height-for-width target.  This
            # was verified to produce an identical target with or without the
            # flash, on a tall word-wrapped panel inside a scroll area.
            c.setMaximumHeight(0)
            c.setVisible(True)
            lay = self.layout()
            if lay is not None:
                lay.activate()
            # Activate the content's OWN layout too, so every (word-wrapped)
            # child already has its final geometry on the very first animated
            # frame.  The content then reveals top-down fully painted instead of
            # as an empty box that fills in late.
            inner = c.layout()
            if inner is not None:
                inner.activate()
            target = self._content_target_height()
            ui_anim.animate_height(
                c, 0, target,
                on_finish=lambda: c.setMaximumHeight(16777215))
        else:
            # Collapse from the content's current live height (its true expanded
            # size); fall back to a fresh measure if it somehow reports zero.
            start = c.height() if c.height() > 0 else self._content_target_height()

            def _fin():
                c.setVisible(False)
                c.setMaximumHeight(16777215)
            ui_anim.animate_height(c, start, 0, on_finish=_fin)

    @property
    def content_layout(self) -> QtWidgets.QVBoxLayout:
        return self._content_layout

    def set_expanded(self, expanded: bool):
        if self._header.isChecked() != expanded:
            self._animate_next = False          # programmatic → instant
            self._header.setChecked(expanded)

    def set_badge(self, text: str, kind: str | None = None):
        """Show a small status chip on the header right (e.g. a section's active
        config: 'auto', 'none', 'on').  `kind="active"` paints it accent-filled,
        otherwise a muted outline.  Empty text hides it.  Inline-styled (a global
        transparent-QLabel rule would otherwise beat an app-level background)."""
        if not text:
            self._badge.hide()
            return
        if kind in ("active", "on"):
            bg, fg, border = _THEME["ACC"], _THEME["ACC_FG"], "none"
        else:
            bg, fg, border = (_THEME["PANEL_ALT"], _THEME["TXT_MUTED"],
                              "1px solid %s" % _THEME["BORDER"])
        self._badge.setStyleSheet(
            "QLabel#section_badge { background-color: %s; color: %s; "
            "border: %s; border-radius: 8px; padding: 1px 8px; "
            "font-size: 10px; font-weight: 600; }" % (bg, fg, border))
        self._badge.setText(text)
        self._badge.show()


class _ResourceMonitor(QtWidgets.QFrame):
    """1 Hz strip of system-resource gauges shown at the top of the
    Analysis tab.  Four cells: CPU%, RAM used / total, GPU%, GPU VRAM.

    Catches "why is my run slow" instantly — if the GPU sits at 0% the
    backend silently fell back to CPU; if RAM is pinned the OS is
    paging; etc.  Gracefully degrades when psutil or torch is missing.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("resource_monitor")
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(14)
        self._cells: "dict[str, QtWidgets.QLabel]" = {}
        for key, label in (("cpu",  "CPU"),
                           ("ram",  "RAM"),
                           ("gpu",  "GPU"),
                           ("vram", "VRAM")):
            cell = QtWidgets.QHBoxLayout()
            cell.setSpacing(4)
            cap = QtWidgets.QLabel(label)
            cap.setStyleSheet(
                f"color: {_THEME['TXT_MUTED']}; font-size: 11px;"
                "font-weight: 600;")
            val = QtWidgets.QLabel("—")
            val.setStyleSheet(
                f"color: {_THEME['TXT']}; font-size: 12px;")
            val.setMinimumWidth(70)
            self._cells[key] = val
            cell.addWidget(cap)
            cell.addWidget(val)
            h.addLayout(cell)
        h.addStretch(1)

        # Probe deps once at construction so we don't pay the import cost
        # per tick.  All three are optional — graceful no-op if absent.
        self._psutil = None
        # CRITICAL: do NOT eagerly import torch into the GUI process.
        # The whole subprocess-isolation architecture (firefly_worker.py
        # comment, top of file) exists to keep torch + Metal/MPS out of
        # the Qt process on Apple Silicon.  Importing it here defeats
        # that and contributes to MPS allocator "Insufficient Memory"
        # crashes mid-localisation.
        #
        # Instead, we only USE torch for VRAM reporting if some other
        # part of the app has already imported it (i.e. never in the
        # GUI process — only when the user is on Windows + CUDA where
        # subprocess isolation isn't needed and torch may be on
        # sys.path via the sidecar installer).
        self._torch = None
        try:
            import psutil as _ps
            self._psutil = _ps
            # Warm the per-process cpu_percent baseline so the first
            # reading isn't a misleading 0.0.
            try:    _ps.cpu_percent(interval=None)
            except Exception: pass
        except Exception:
            pass

        # Cached MPS utilisation (Apple Silicon has no Python API — we
        # shell out to `ioreg`, which is fast but worth backgrounding).
        self._mps_util_cache: "int | None" = None
        self._mps_polling: bool = False

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()
        self._refresh()

    def _maybe_get_torch(self):
        """Return torch ONLY if it's already in sys.modules (i.e. some
        other code path imported it).  Never triggers a fresh import.
        Re-checks each tick so it picks up torch after the CUDA sidecar
        installer adds it to sys.path mid-session."""
        if self._torch is not None:
            return self._torch
        import sys as _sys
        mod = _sys.modules.get("torch")
        if mod is not None:
            self._torch = mod
        return self._torch

    @staticmethod
    def _mps_gpu_utilization() -> "int | None":
        """Best-effort macOS GPU utilisation via `ioreg`.  Returns an
        integer percent or None if we can't tell.  Runs subprocess
        — never call from the GUI thread; use `_poll_mps_async`.

        Different macOS / chip combos expose the field under slightly
        different keys, so we try a few.  Cheap regex parse — no
        plistlib needed."""
        import subprocess, re
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
        """Run the ioreg parse on a background thread, write the result
        to the cache.  Re-entrancy-guarded so we don't pile up threads."""
        if self._mps_polling:
            return
        self._mps_polling = True
        import threading
        def _run():
            try:    self._mps_util_cache = self._mps_gpu_utilization()
            finally: self._mps_polling = False
        threading.Thread(target=_run, daemon=True).start()

    def _set(self, key: str, txt: str, *, warn: bool = False):
        lbl = self._cells.get(key)
        if lbl is None: return
        col = _THEME['WARN'] if warn else _THEME['TXT']
        lbl.setStyleSheet(
            f"color: {col}; font-size: 12px;")
        lbl.setText(txt)

    def _refresh(self):
        # CPU + RAM via psutil
        if self._psutil is not None:
            try:
                pct = self._psutil.cpu_percent(interval=None)
                self._set("cpu", f"{pct:5.1f} %",
                          warn=(pct > 90))
            except Exception:
                self._set("cpu", "—")
            try:
                vm = self._psutil.virtual_memory()
                used_gb = (vm.total - vm.available) / 1e9
                total_gb = vm.total / 1e9
                self._set("ram",
                          f"{used_gb:4.1f} / {total_gb:.0f} GB",
                          warn=(vm.percent > 90))
            except Exception:
                self._set("ram", "—")
        else:
            self._set("cpu", "(psutil)")
            self._set("ram", "(psutil)")

        # GPU usage + VRAM.
        #
        # Subprocess-isolation invariant (B2): we MUST NOT import torch
        # into the GUI process, especially on Apple Silicon, because Qt
        # + Metal contend with the analysis subprocess's MPS allocator.
        # But that doesn't mean we can't report GPU stats — both the
        # MPS path (ioreg) and the NVIDIA path (nvidia-smi / pynvml)
        # are torch-free.  Use those directly so users actually SEE
        # what their GPU is doing during a run.
        torch_mod = self._maybe_get_torch()
        gpu_reported = False
        vram_reported = False

        # ── macOS: MPS utilisation via ioreg (torch-free) ─────────────
        if sys.platform == "darwin":
            self._poll_mps_async()
            if self._mps_util_cache is not None:
                self._set("gpu", f"{self._mps_util_cache:5.1f} %",
                          warn=(self._mps_util_cache < 1))
                gpu_reported = True
            else:
                self._set("gpu", "MPS")
                gpu_reported = True
            # VRAM on Apple Silicon is unified with system RAM — no
            # separate readout.  If torch IS already in this process
            # (uncommon but possible), report its allocator footprint;
            # otherwise display a dash.
            if torch_mod is not None:
                try:
                    if (hasattr(torch_mod.backends, "mps")
                            and torch_mod.backends.mps.is_available()):
                        alloc = torch_mod.mps.current_allocated_memory() / 1e9
                        self._set("vram", f"{alloc:4.1f} GB")
                        vram_reported = True
                except Exception:
                    pass
            if not vram_reported:
                self._set("vram", "unified")
                vram_reported = True

        # ── NVIDIA (any OS): util + VRAM via pynvml (torch-free) ──────
        if not gpu_reported:
            try:
                import pynvml as _nv
                _nv.nvmlInit()
                try:
                    h = _nv.nvmlDeviceGetHandleByIndex(0)
                    util = _nv.nvmlDeviceGetUtilizationRates(h).gpu
                    mem  = _nv.nvmlDeviceGetMemoryInfo(h)
                    self._set("gpu", f"{util:5.1f} %",
                              warn=(util < 1))
                    self._set("vram",
                              f"{mem.used/1e9:4.1f} / {mem.total/1e9:.0f} GB")
                    gpu_reported = True
                    vram_reported = True
                finally:
                    try: _nv.nvmlShutdown()
                    except Exception: pass
            except Exception:
                # pynvml not installed OR no NVIDIA driver.  Try
                # nvidia-smi as a last resort (Windows .exe bundles
                # pynvml but on a fresh from-source install it may
                # be missing).
                try:
                    import subprocess as _sub
                    flags = getattr(_sub, "CREATE_NO_WINDOW", 0) \
                            if sys.platform == "win32" else 0
                    out = _sub.run(
                        ["nvidia-smi",
                         "--query-gpu=utilization.gpu,memory.used,memory.total",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=2,
                        creationflags=flags).stdout.strip().splitlines()
                    if out:
                        u, used, total = [s.strip() for s in out[0].split(",")]
                        self._set("gpu", f"{float(u):5.1f} %",
                                  warn=(float(u) < 1))
                        self._set("vram",
                                  f"{float(used)/1024:4.1f} / "
                                  f"{float(total)/1024:.0f} GB")
                        gpu_reported = True
                        vram_reported = True
                except Exception:
                    pass

        # ── Last resort: CPU only / nothing detected ──────────────────
        if not gpu_reported:
            self._set("gpu", "CPU only")
        if not vram_reported:
            self._set("vram", "—")


class _MassHistogram(QtWidgets.QWidget):
    """Lightweight live histogram of localisation mass values.

    Renders with QPainter — no matplotlib dependency in the GUI process.
    Bars accumulate across chunks; clear via `reset()`.  Designed to
    show during a run so the user can sanity-check minmass on the fly.
    """
    BIN_COUNT = 40

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Preferred)
        self._counts = None          # np.ndarray | None
        self._edges  = None          # np.ndarray | None
        self._total  = 0
        self._minmass = None         # float | None (vertical guide line)
        # Throttle repaints — accumulate updates and only repaint at most
        # ~6 Hz to keep the GUI responsive when chunks land in rapid fire.
        self._dirty = False
        self._repaint_timer = QTimer(self)
        self._repaint_timer.setInterval(16)   # ~60 Hz
        self._repaint_timer.setSingleShot(False)
        self._repaint_timer.timeout.connect(self._maybe_repaint)
        self._repaint_timer.start()

    def reset(self):
        self._counts = None
        self._edges  = None
        self._total  = 0
        self._dirty  = True

    def set_minmass(self, value):
        try:
            self._minmass = float(value) if value is not None else None
        except (TypeError, ValueError):
            self._minmass = None
        self._dirty = True

    def add_chunk(self, mass_values) -> None:
        """Accept an iterable of mass values and merge into the histogram."""
        try:
            import numpy as _np
            arr = _np.asarray(list(mass_values), dtype=_np.float32)
            arr = arr[_np.isfinite(arr)]
            if arr.size == 0:
                return
            if self._counts is None:
                # First chunk seeds the bin edges.  Range from 0 to 99th-pct
                # of the data, expanded slightly for headroom.
                hi = float(_np.percentile(arr, 99.0)) * 1.3 + 1e-6
                self._edges = _np.linspace(0.0, hi, self.BIN_COUNT + 1)
                self._counts = _np.zeros(self.BIN_COUNT, dtype=_np.int64)
            new_counts, _ = _np.histogram(arr, bins=self._edges)
            self._counts += new_counts
            self._total += int(arr.size)
            self._dirty = True
        except Exception:
            pass

    def _maybe_repaint(self):
        if self._dirty:
            self._dirty = False
            self.update()

    def paintEvent(self, _evt):
        import math
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, False)
        r = self.rect()
        # Background
        p.fillRect(r, QtGui.QColor(_THEME['PANEL']))
        # Border
        pen = QtGui.QPen(QtGui.QColor(_THEME['BORDER']), 1)
        p.setPen(pen)
        p.drawRect(r.adjusted(0, 0, -1, -1))

        # Padding
        pad_l, pad_t, pad_r, pad_b = 8, 22, 8, 16
        plot = r.adjusted(pad_l, pad_t, -pad_r, -pad_b)

        # Title
        p.setPen(QtGui.QColor(_THEME['TXT_MUTED']))
        f = p.font(); f.setPointSize(10); p.setFont(f)
        if self._counts is None or self._counts.sum() == 0:
            p.drawText(r, Qt.AlignmentFlag.AlignCenter,
                       "Localisation mass distribution will appear here\n"
                       "as chunks finish.")
            return
        title = f"Localisation mass  ·  n = {self._total:,}"
        p.drawText(QtCore.QRect(r.left() + pad_l, r.top() + 4,
                                r.width() - pad_l - pad_r, 18),
                   Qt.AlignmentFlag.AlignLeft, title)

        # Bars
        n_bins = len(self._counts)
        max_h  = float(self._counts.max()) or 1.0
        bar_w  = plot.width() / n_bins
        bar_pen = QtGui.QPen(QtGui.QColor(_THEME['ACC']))
        bar_pen.setWidth(0)
        p.setPen(bar_pen)
        p.setBrush(QtGui.QBrush(QtGui.QColor(_THEME['ACC'])))
        for i, c in enumerate(self._counts):
            h = (c / max_h) * plot.height()
            x = plot.left() + i * bar_w
            y = plot.bottom() - h
            p.drawRect(QtCore.QRectF(x + 0.5, y, max(1.0, bar_w - 1.0), h))

        # X-axis ticks (just min + max edges)
        p.setPen(QtGui.QColor(_THEME['TXT_MUTED']))
        f = p.font(); f.setPointSize(8); p.setFont(f)
        p.drawText(plot.left(),       r.bottom() - 2,
                   f"{self._edges[0]:.2f}")
        p.drawText(plot.right() - 60, r.bottom() - 2,
                   f"{self._edges[-1]:.2f}  mass")

        # Min-mass guide line
        if self._minmass is not None and self._edges is not None:
            lo, hi = float(self._edges[0]), float(self._edges[-1])
            if hi > lo and lo <= self._minmass <= hi * 1.2:
                frac = (self._minmass - lo) / (hi - lo)
                x = plot.left() + min(1.0, max(0.0, frac)) * plot.width()
                pen = QtGui.QPen(QtGui.QColor(_THEME['WARN']))
                pen.setStyle(Qt.PenStyle.DashLine)
                pen.setWidth(1)
                p.setPen(pen)
                p.drawLine(QtCore.QPointF(x, plot.top()),
                           QtCore.QPointF(x, plot.bottom()))
                p.setPen(QtGui.QColor(_THEME['WARN']))
                p.drawText(QtCore.QPointF(x + 3, plot.top() + 10),
                           f"min={self._minmass:g}")
        p.end()


class _LiveFrameView(QtWidgets.QWidget):
    """Renders the most-recent preprocessed frame from the localisation
    stream plus the detections found on it.  Pairs with `_MassHistogram`
    to form a 'detection cockpit' on the Analysis tab so the user can
    watch what's actually being detected during a run."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(220, 160)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)
        self._frame = None           # 2D float32 array
        self._xs = None
        self._ys = None
        self._idx = None
        self._n_frames = None
        self._n_spots = 0
        # Throttle repaints to ~6 Hz so a hot stream of chunks doesn't
        # pin the GUI thread.
        self._dirty = False
        self._timer = QTimer(self)
        self._timer.setInterval(16)   # ~60 Hz
        self._timer.timeout.connect(self._maybe_repaint)
        self._timer.start()

    def reset(self):
        self._frame = None
        self._xs = None
        self._ys = None
        self._idx = None
        self._n_frames = None
        self._n_spots = 0
        self._dirty = True

    def set_frame(self, frame, xs, ys, idx, n_frames):
        try:
            import numpy as _np
            self._frame = _np.asarray(frame, dtype=_np.float32)
            self._xs = _np.asarray(xs, dtype=_np.float32)
            self._ys = _np.asarray(ys, dtype=_np.float32)
            self._idx = int(idx)
            self._n_frames = int(n_frames)
            self._n_spots = int(self._xs.size)
            self._dirty = True
        except Exception:
            pass

    def _maybe_repaint(self):
        if self._dirty:
            self._dirty = False
            self.update()

    def paintEvent(self, _evt):
        p = QtGui.QPainter(self)
        r = self.rect()
        p.fillRect(r, QtGui.QColor(_THEME['PANEL']))
        p.setPen(QtGui.QColor(_THEME['BORDER']))
        p.drawRect(r.adjusted(0, 0, -1, -1))

        if self._frame is None:
            p.setPen(QtGui.QColor(_THEME['TXT_MUTED']))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter,
                       "Live detection view will appear here\n"
                       "during a run.")
            return

        try:
            import numpy as _np
            f = self._frame
            lo, hi = _np.percentile(f, [1.0, 99.5])
            if hi <= lo:
                hi = lo + 1.0
            u8 = _np.clip((f - lo) * (255.0 / (hi - lo)),
                          0, 255).astype(_np.uint8, copy=False)
            # Pad to a contiguous buffer that QImage can wrap safely
            u8 = _np.ascontiguousarray(u8)
            h, w = u8.shape
            img = QtGui.QImage(u8.tobytes(), w, h, w,
                                QtGui.QImage.Format.Format_Grayscale8)
        except Exception:
            return

        # Compute the rect inside the widget where we draw the frame
        pad_t = 22; pad = 8
        avail = r.adjusted(pad, pad_t, -pad, -pad)
        if avail.width() <= 0 or avail.height() <= 0:
            return
        scale = min(avail.width() / w, avail.height() / h)
        disp_w = max(1, int(w * scale))
        disp_h = max(1, int(h * scale))
        disp_x = avail.left() + (avail.width()  - disp_w) // 2
        disp_y = avail.top()  + (avail.height() - disp_h) // 2
        p.drawImage(QtCore.QRect(disp_x, disp_y, disp_w, disp_h), img)

        # Title strip
        p.setPen(QtGui.QColor(_THEME['TXT_MUTED']))
        font = p.font(); font.setPointSize(10); p.setFont(font)
        idx = self._idx if self._idx is not None else 0
        total = self._n_frames if self._n_frames else 0
        title = (f"Live detection  ·  frame {idx + 1}/{total}"
                 f"  ·  {self._n_spots} spots")
        p.drawText(QtCore.QRect(r.left() + pad, r.top() + 4,
                                r.width() - 2 * pad, 18),
                   Qt.AlignmentFlag.AlignLeft, title)

        # Detection circles — flat accent for visibility on greyscale
        if self._xs is not None and self._xs.size > 0:
            pen = QtGui.QPen(QtGui.QColor(_THEME['ACC']))
            pen.setWidthF(1.2)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            radius = max(3.0, 4.0 * scale)
            for x, y in zip(self._xs, self._ys):
                cx = disp_x + float(x) * scale
                cy = disp_y + float(y) * scale
                p.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)
        p.end()


class _HyperflyTile(QtWidgets.QFrame):
    """One lane of the HYPERFLY dashboard: a compact live detection-view for the
    file currently running in this lane — frame + spot overlay, a header with
    the file stem, and a footer with stage / progress / counts.  Border colour
    encodes state (running = accent, done = green, failed = red).

    Painting is driven by the parent dashboard's single shared timer (set
    ``_dirty`` and it repaints on the next flush) so K tiles don't each run
    their own timer."""

    _IDLE, _RUNNING, _DONE, _FAILED = "idle", "running", "done", "failed"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(150, 132)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)
        self._state = self._IDLE
        self._stem = ""
        self._stage = ""
        self._pct = 0
        self._frame = None
        self._xs = None
        self._ys = None
        self._n_spots = 0
        self._n_locs = None
        self._n_tracks = None
        self._dirty = True

    def clear(self):
        self._state = self._IDLE
        self._stem = ""
        self._stage = ""
        self._pct = 0
        self._frame = self._xs = self._ys = None
        self._n_spots = 0
        self._n_locs = self._n_tracks = None
        self._dirty = True

    # ── state setters (mark dirty; the dashboard timer repaints) ──
    def set_running(self, stem):
        self._state = self._RUNNING
        self._stem = str(stem)
        self._dirty = True

    def set_preview(self, frame, xs, ys, n_spots=None):
        try:
            import numpy as _np
            self._frame = _np.asarray(frame, dtype=_np.float32)
            self._xs = _np.asarray(xs, dtype=_np.float32)
            self._ys = _np.asarray(ys, dtype=_np.float32)
            self._n_spots = int(self._xs.size if n_spots is None else n_spots)
            if self._state == self._IDLE:
                self._state = self._RUNNING
            self._dirty = True
        except Exception:
            pass

    def set_progress(self, pct, stage):
        try:    self._pct = max(0, min(100, int(pct)))
        except Exception: pass
        self._stage = str(stage)
        self._dirty = True

    def set_done(self, n_locs=None, n_tracks=None):
        self._state = self._DONE
        self._n_locs = n_locs
        self._n_tracks = n_tracks
        self._pct = 100
        self._dirty = True

    def set_failed(self, err=""):
        self._state = self._FAILED
        self._stage = str(err)[:60]
        self._dirty = True

    def is_free(self) -> bool:
        """A tile is reusable once its file finished (or it never started)."""
        return self._state in (self._IDLE, self._DONE, self._FAILED)

    def _border(self):
        return {
            self._RUNNING: _THEME.get("ACC", "#2f81f7"),
            self._DONE:    _THEME.get("SUCCESS", "#3fb950"),
            self._FAILED:  _THEME.get("DANGER", "#f85149"),
        }.get(self._state, _THEME.get("BORDER", "#30363d"))

    def paintEvent(self, _evt):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        r = self.rect()
        p.fillRect(r, QtGui.QColor(_THEME['PANEL']))
        pen = QtGui.QPen(QtGui.QColor(self._border())); pen.setWidth(2)
        p.setPen(pen)
        p.drawRoundedRect(r.adjusted(1, 1, -1, -1), 6, 6)

        pad = 6
        head_h = 16
        foot_h = 14
        # Header — stem + a tiny state glyph.
        p.setPen(QtGui.QColor(_THEME['TXT']))
        f = p.font(); f.setPointSize(8); f.setBold(True); p.setFont(f)
        glyph = {self._DONE: "✓ ", self._FAILED: "✗ ",
                 self._RUNNING: "", self._IDLE: ""}.get(self._state, "")
        head_txt = (glyph + self._stem) if self._stem else "idle"
        p.drawText(QtCore.QRect(r.left() + pad, r.top() + 3,
                                r.width() - 2 * pad, head_h),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   p.fontMetrics().elidedText(
                       head_txt, Qt.TextElideMode.ElideMiddle,
                       r.width() - 2 * pad))

        img_top = r.top() + 3 + head_h
        img_rect = QtCore.QRect(r.left() + pad, img_top,
                                r.width() - 2 * pad,
                                r.bottom() - img_top - foot_h - pad)
        # Live frame (greyscale) + detections.
        if self._frame is not None and img_rect.width() > 4 and img_rect.height() > 4:
            try:
                import numpy as _np
                fr = self._frame
                lo, hi = _np.percentile(fr, [1.0, 99.5])
                if hi <= lo:
                    hi = lo + 1.0
                u8 = _np.clip((fr - lo) * (255.0 / (hi - lo)), 0, 255
                              ).astype(_np.uint8, copy=False)
                u8 = _np.ascontiguousarray(u8)
                h, w = u8.shape
                img = QtGui.QImage(u8.tobytes(), w, h, w,
                                   QtGui.QImage.Format.Format_Grayscale8)
                scale = min(img_rect.width() / w, img_rect.height() / h)
                dw = max(1, int(w * scale)); dh = max(1, int(h * scale))
                dx = img_rect.left() + (img_rect.width() - dw) // 2
                dy = img_rect.top() + (img_rect.height() - dh) // 2
                p.drawImage(QtCore.QRect(dx, dy, dw, dh), img)
                if self._xs is not None and self._xs.size > 0:
                    cpen = QtGui.QPen(QtGui.QColor(_THEME['ACC']))
                    cpen.setWidthF(1.0); p.setPen(cpen)
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    rad = max(2.0, 3.0 * scale)
                    for x, y in zip(self._xs, self._ys):
                        p.drawEllipse(QtCore.QPointF(dx + float(x) * scale,
                                                     dy + float(y) * scale),
                                      rad, rad)
            except Exception:
                pass
        else:
            p.setPen(QtGui.QColor(_THEME['TXT_MUTED']))
            fs = p.font(); fs.setBold(False); fs.setPointSize(8); p.setFont(fs)
            p.drawText(img_rect, Qt.AlignmentFlag.AlignCenter,
                       "…" if self._state == self._RUNNING else "")

        # Footer — stage / pct / counts.
        p.setPen(QtGui.QColor(_THEME['TXT_MUTED']))
        ff = p.font(); ff.setBold(False); ff.setPointSize(8); p.setFont(ff)
        if self._state == self._DONE:
            foot = f"✓ {int(self._n_locs or 0):,} locs · {int(self._n_tracks or 0):,} tracks"
        elif self._state == self._FAILED:
            foot = f"✗ {self._stage}"
        elif self._state == self._RUNNING:
            bits = []
            if self._stage: bits.append(self._stage)
            if self._n_spots: bits.append(f"{self._n_spots} spots")
            foot = f"{self._pct}%  ·  " + " · ".join(bits) if bits else f"{self._pct}%"
        else:
            foot = ""
        p.drawText(QtCore.QRect(r.left() + pad, r.bottom() - foot_h - 2,
                                r.width() - 2 * pad, foot_h),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   p.fontMetrics().elidedText(foot, Qt.TextElideMode.ElideRight,
                                              r.width() - 2 * pad))
        p.end()


class _HyperflyDashboard(QtWidgets.QWidget):
    """Grid of `_HyperflyTile`s — one lane per concurrent file — for the
    Analysis cockpit while a HYPERFLY batch runs.  Files are routed to a free
    tile as they start and release it when they finish, so K tiles cycle
    through all N files.  A single ~12 Hz timer flushes dirty tiles."""

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        self._caption = QtWidgets.QLabel("")
        self._caption.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']}; font-size: 11px;")
        outer.addWidget(self._caption)
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._grid_host = QtWidgets.QWidget()
        self._grid = QtWidgets.QGridLayout(self._grid_host)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(6)
        self._scroll.setWidget(self._grid_host)
        outer.addWidget(self._scroll, 1)

        self._tiles: list = []
        self._map: dict = {}          # file index → tile
        self._timer = QTimer(self)
        self._timer.setInterval(80)   # ~12 Hz flush
        self._timer.timeout.connect(self._flush)

    def build(self, n_lanes: int):
        """(Re)build the grid for `n_lanes` lanes."""
        self._timer.stop()
        for t in self._tiles:
            t.setParent(None)
            t.deleteLater()
        self._tiles = []
        self._map = {}
        n = max(1, int(n_lanes))
        import math
        cols = max(1, min(n, int(math.ceil(math.sqrt(n)))))
        for i in range(n):
            t = _HyperflyTile()
            self._grid.addWidget(t, i // cols, i % cols)
            self._tiles.append(t)
        self._timer.start()

    def set_caption(self, text: str):
        self._caption.setText(text)

    def _slot_for(self, fidx, create=False):
        t = self._map.get(fidx)
        if t is not None:
            return t
        if not create:
            return None
        for t in self._tiles:
            if t not in self._map.values() and t.is_free():
                self._map[fidx] = t
                return t
        return None

    def _free(self, fidx):
        self._map.pop(fidx, None)

    # ── routing from the message pump ──
    def on_running(self, fidx, stem):
        t = self._slot_for(fidx, create=True)
        if t is not None:
            t.set_running(stem)

    def on_preview(self, fidx, frame, xs, ys, n_spots=None):
        t = self._slot_for(fidx, create=True)
        if t is not None:
            t.set_preview(frame, xs, ys, n_spots)

    def on_progress(self, fidx, pct, stage):
        t = self._slot_for(fidx, create=True)
        if t is not None:
            t.set_progress(pct, stage)

    def on_done(self, fidx, n_locs=None, n_tracks=None):
        t = self._slot_for(fidx)
        if t is not None:
            t.set_done(n_locs, n_tracks)
        self._free(fidx)

    def on_failed(self, fidx, err=""):
        t = self._slot_for(fidx)
        if t is not None:
            t.set_failed(err)
        self._free(fidx)

    def _flush(self):
        for t in self._tiles:
            if getattr(t, "_dirty", False):
                t._dirty = False
                t.update()

    def stop(self):
        self._timer.stop()


class _TrackInspector(QtWidgets.QFrame):
    """Right-side panel for the Visualise tab.  Displays per-particle stats
    for whichever track the user clicked on in the embedded napari viewer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("track_inspector")
        self.setMinimumWidth(280)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        title = QtWidgets.QLabel("Track inspector")
        title.setStyleSheet(
            f"color: {_THEME['TXT']}; font-size: 14px; font-weight: 700;")
        v.addWidget(title)

        self._hint = QtWidgets.QLabel(
            "Click a track in the viewer to inspect it.")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']}; font-size: 12px;")
        v.addWidget(self._hint)

        # Stats grid
        self._grid = QtWidgets.QGridLayout()
        self._grid.setColumnStretch(0, 0)
        self._grid.setColumnStretch(1, 1)
        self._grid.setHorizontalSpacing(12)
        self._grid.setVerticalSpacing(4)
        grid_w = QtWidgets.QWidget()
        grid_w.setLayout(self._grid)
        self._grid_w = grid_w
        v.addWidget(grid_w)
        grid_w.hide()

        v.addStretch(1)

    def clear(self):
        self._hint.show()
        self._grid_w.hide()
        # Wipe the grid
        while self._grid.count():
            it = self._grid.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()

    def show_track(self, *, particle_id: int,
                   length: int | None = None,
                   d: float | None = None,
                   alpha: float | None = None,
                   motion: str | None = None,
                   mean_mass: float | None = None,
                   start_frame: int | None = None,
                   end_frame: int | None = None,
                   net_displacement_um: float | None = None,
                   total_path_um: float | None = None,
                   straightness: float | None = None):
        # Clear and re-populate
        while self._grid.count():
            it = self._grid.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()

        def _row(r, label, value, *, color=None):
            lbl = QtWidgets.QLabel(label)
            lbl.setStyleSheet(
                f"color: {_THEME['TXT_MUTED']}; font-size: 12px;")
            val = QtWidgets.QLabel(value)
            val.setStyleSheet(
                f"color: {color or _THEME['TXT']}; font-size: 13px; "
                "font-weight: 600;")
            val.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            self._grid.addWidget(lbl, r, 0, Qt.AlignmentFlag.AlignLeft)
            self._grid.addWidget(val, r, 1, Qt.AlignmentFlag.AlignLeft)

        # Pull straight from _MOTION_PALETTE so the inspector's swatch
        # matches the napari layer colour AND the figure colour for that
        # motion class — single source of truth.
        motion_colour = dict(_MOTION_PALETTE)
        r = 0
        _row(r, "Particle ID", f"#{particle_id}"); r += 1
        if length is not None:
            _row(r, "Track length", f"{length} frames"); r += 1
        if start_frame is not None and end_frame is not None:
            _row(r, "Frame span", f"{start_frame} → {end_frame}"); r += 1
        if motion:
            _row(r, "Motion class", motion,
                 color=motion_colour.get(motion)); r += 1
        if d is not None:
            _row(r, "Diffusion D",  f"{d:.4f}  µm²/s"); r += 1
        if alpha is not None:
            _row(r, "α (anomalous)", f"{alpha:.3f}"); r += 1
        if net_displacement_um is not None:
            _row(r, "Net displacement",
                 f"{net_displacement_um*1000:.0f} nm"); r += 1
        if total_path_um is not None:
            _row(r, "Total path",
                 f"{total_path_um*1000:.0f} nm"); r += 1
        if straightness is not None:
            _row(r, "Straightness", f"{straightness:.3f}"); r += 1
        if mean_mass is not None:
            _row(r, "Mean mass", f"{mean_mass:.1f}"); r += 1

        self._hint.hide()
        self._grid_w.show()

    # ── Cluster-mode inspector ───────────────────────────────────────
    # Visualise tab's interactive DBSCAN cluster map needs to surface
    # cluster (not track) stats in this same panel.  Kept as a separate
    # method so each kwarg list stays explicit and self-documenting —
    # the alternative of taking **kwargs and dispatching by key would
    # silently swallow typos.
    def show_cluster(self, *, cluster_id: int,
                      n_locs: int | None = None,
                      area_um2: float | None = None,
                      density_locs_per_um2: float | None = None,
                      rg_um: float | None = None,
                      centroid_x_um: float | None = None,
                      centroid_y_um: float | None = None,
                      note: str | None = None):
        while self._grid.count():
            it = self._grid.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()

        def _row(r, label, value, *, color=None):
            lbl = QtWidgets.QLabel(label)
            lbl.setStyleSheet(
                f"color: {_THEME['TXT_MUTED']}; font-size: 12px;")
            val = QtWidgets.QLabel(value)
            val.setStyleSheet(
                f"color: {color or _THEME['TXT']}; font-size: 13px; "
                "font-weight: 600;")
            val.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)
            self._grid.addWidget(lbl, r, 0, Qt.AlignmentFlag.AlignLeft)
            self._grid.addWidget(val, r, 1, Qt.AlignmentFlag.AlignLeft)

        r = 0
        if cluster_id == -1:
            _row(r, "Cluster ID", "Noise",
                 color=_THEME['TXT_MUTED']); r += 1
            if note:
                _row(r, "Note", note); r += 1
        else:
            _row(r, "Cluster ID", f"#{cluster_id}",
                 color=_THEME['ACC']); r += 1
            if n_locs is not None:
                _row(r, "Localisations", f"{int(n_locs):,}"); r += 1
            if area_um2 is not None:
                _row(r, "Area", f"{area_um2:.4f} µm²"); r += 1
            if density_locs_per_um2 is not None:
                _row(r, "Density",
                     f"{density_locs_per_um2:.1f} locs/µm²"); r += 1
            if rg_um is not None:
                _row(r, "Radius of gyration", f"{rg_um:.4f} µm"); r += 1
            if centroid_x_um is not None and centroid_y_um is not None:
                _row(r, "Centroid",
                     f"({centroid_x_um:.3f}, {centroid_y_um:.3f}) µm")
                r += 1
            if note:
                _row(r, "Note", note); r += 1

        self._hint.hide()
        self._grid_w.show()


class _ResultsPanel(QtWidgets.QFrame):
    """Compact "results" panel shown below the progress bar on each tab.

    Replaces the in-app matplotlib canvas — figures are now saved to disk
    only, per user preference.  After a run completes, this panel lists
    the saved files and offers a button to open the output folder in the
    system file manager.
    """
    def __init__(self, idle_text: str, parent=None):
        super().__init__(parent)
        self.setObjectName("results_panel")
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(12)

        # Header row: run title (left) + the run-readiness pill (right).  The
        # pill is set from the QC flag levels after a run ("Analysis
        # successful" / "Completed with warnings") and hidden until then.
        self._headline = QtWidgets.QLabel(idle_text)
        self._headline.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']}; font-size: 14px;")
        self._headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._headline.setWordWrap(True)
        self._qc_badge = _StatusBadge()
        self._qc_badge.hide()
        _hdr = QtWidgets.QHBoxLayout()
        _hdr.setSpacing(8)
        _hdr.addWidget(self._headline, 1)
        _hdr.addWidget(self._qc_badge, 0, Qt.AlignmentFlag.AlignVCenter)
        v.addLayout(_hdr)

        # Stats grid — populated post-run with key numbers (median D / α,
        # motion-class breakdown, cluster count, etc.).
        self._stats_grid = QtWidgets.QGridLayout()
        self._stats_grid.setColumnStretch(0, 0)
        self._stats_grid.setColumnStretch(1, 1)
        self._stats_grid.setHorizontalSpacing(18)
        self._stats_grid.setVerticalSpacing(7)
        stats_container = QtWidgets.QWidget()
        stats_container.setLayout(self._stats_grid)
        self._stats_container = stats_container
        self._stats_container.hide()
        v.addWidget(self._stats_container)

        # QC flag messages — these are multi-line, word-wrapped warnings.
        # They live in their OWN vertical layout, not the stats grid:
        # QGridLayout doesn't honour heightForWidth for a wrapped label that
        # spans columns, so wrapped flags rendered there get clipped to one
        # row and overlap the cell above.  A QVBoxLayout sizes them correctly.
        self._flags_container = QtWidgets.QWidget()
        self._flags_layout = QtWidgets.QVBoxLayout(self._flags_container)
        self._flags_layout.setContentsMargins(0, 2, 0, 0)
        self._flags_layout.setSpacing(4)
        self._flags_container.hide()
        v.addWidget(self._flags_container)

        # Output-folder row (visible only after a run)
        self._folder_row = QtWidgets.QWidget()
        fr = QtWidgets.QHBoxLayout(self._folder_row)
        fr.setContentsMargins(0, 0, 0, 0)
        self._folder_label = QtWidgets.QLabel("")
        self._folder_label.setStyleSheet(f"color: {_THEME['TXT']};")
        self._folder_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self._folder_label.setWordWrap(True)
        fr.addWidget(self._folder_label, 1)
        self._open_btn = QtWidgets.QPushButton("Open folder")
        self._open_btn.clicked.connect(self._on_open_folder)
        fr.addWidget(self._open_btn)
        self._folder_row.hide()
        v.addWidget(self._folder_row)

        # File list (the saved CSVs / PDFs / PNGs)
        self._files = QtWidgets.QListWidget()
        self._files.setObjectName("results_files")
        self._files.setAlternatingRowColors(True)
        self._files.itemDoubleClicked.connect(self._on_file_dbl)
        self._files.hide()
        v.addWidget(self._files, stretch=1)

        # Trailing stretch when idle so the headline centres vertically
        self._stretch_when_idle = True
        v.addStretch(1)

        self._out_dir = ""

    def reset(self, idle_text: str = ""):
        if idle_text:
            self._headline.setText(idle_text)
        self._headline.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']}; font-size: 14px;")
        self._headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._folder_row.hide()
        self._files.clear()
        self._files.hide()
        self._clear_stats()
        self._stats_container.hide()
        self._out_dir = ""

    def _clear_stats(self):
        while self._stats_grid.count():
            item = self._stats_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        # QC flag messages live in their own layout/container — clear those too.
        while self._flags_layout.count():
            item = self._flags_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._flags_container.hide()
        self._qc_badge.hide()

    def _add_stat_row(self, row: int, label: str, value: str,
                      value_colour: str | None = None,
                      tooltip: str | None = None):
        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']}; font-size: 12.5px;")
        val = QtWidgets.QLabel(value)
        col = value_colour or _THEME['TXT']
        val.setStyleSheet(
            f"color: {col}; font-size: 14px; font-weight: 600;")
        val.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        if tooltip:
            lbl.setToolTip(tooltip)
            val.setToolTip(tooltip)
        self._stats_grid.addWidget(lbl, row, 0,
                                   Qt.AlignmentFlag.AlignLeft)
        self._stats_grid.addWidget(val, row, 1,
                                   Qt.AlignmentFlag.AlignLeft)

    def _add_section_header(self, row: int, text: str):
        """A small bold section header spanning both grid columns, with a
        hairline divider above it to group the stats into sections."""
        hdr = QtWidgets.QLabel(text)
        hdr.setStyleSheet(
            "color: %s; font-size: 12px; font-weight: 700; "
            "text-transform: uppercase; letter-spacing: 0.5px; "
            "border-top: 1px solid %s; padding-top: 10px; margin-top: 12px;"
            % (_THEME['TXT'], _THEME['BORDER']))
        self._stats_grid.addWidget(hdr, row, 0, 1, 2,
                                   Qt.AlignmentFlag.AlignLeft)

    def _swatch_label(self, text: str, color: str) -> QtWidgets.QWidget:
        """A '[colour swatch] label' cell for the motion-class legend."""
        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        sw = QtWidgets.QLabel()
        sw.setFixedSize(10, 10)
        sw.setStyleSheet(
            "background: %s; border-radius: 5px;"
            % (color or _THEME['TXT_MUTED']))
        h.addWidget(sw)
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet(
            "color: %s; font-size: 12px; background: transparent;"
            % _THEME['TXT_MUTED'])
        h.addWidget(lbl)
        h.addStretch(1)
        return w

    def show_stats(self, summary: dict):
        """Populate the stats grid from a worker `summary` dict.

        Expected keys (all optional; missing keys are skipped):
            n_tracks, n_locs, median_d, median_alpha, mobile_fraction,
            median_loc_sigma_nm, nongauss_alpha2, vacf_persistence,
            motion_counts (dict), n_clusters, dwell_tau_s, frames, px_um, fi_s
        """
        self._clear_stats()
        if not summary:
            return
        r = 0

        def _fmt_int(n):    return f"{n:,}" if n is not None else "—"
        def _fmt_pct(f):    return f"{100 * f:.1f} %" if f is not None else "—"
        def _fmt_um2(d):    return f"{d:.4f} µm²/s" if d is not None else "—"
        def _fmt_alpha(a):  return f"{a:.3f}" if a is not None else "—"
        def _fmt_secs(s):   return f"{s:.2f} s" if s is not None else "—"

        # Counts ─────────────────────────────────────────────────────
        self._add_stat_row(r, "Trajectories",
                           _fmt_int(summary.get("n_tracks", 0))); r += 1
        self._add_stat_row(r, "Localisations",
                           _fmt_int(summary.get("n_locs", 0))); r += 1

        # Diffusion ──────────────────────────────────────────────────
        self._add_section_header(r, "Diffusion & dynamics"); r += 1
        self._add_stat_row(r, "Median D",
                           _fmt_um2(summary.get("median_d"))); r += 1
        self._add_stat_row(r, "Median α",
                           _fmt_alpha(summary.get("median_alpha"))); r += 1
        mf = summary.get("mobile_fraction")
        if mf is not None:
            self._add_stat_row(
                r, "Mobile fraction", _fmt_pct(mf),
                tooltip="Fraction of tracks with D above the mobile-D "
                        "threshold."); r += 1
        ls = summary.get("median_loc_sigma_nm")
        if ls is not None:
            self._add_stat_row(r, "Localisation precision  σ",
                               f"{ls:.1f} nm"); r += 1
        a2 = summary.get("nongauss_alpha2")
        if a2 is not None:
            self._add_stat_row(r, "Non-Gaussian  α₂",
                               f"{a2:.3f}"); r += 1
        vp = summary.get("vacf_persistence")
        if vp is not None:
            self._add_stat_row(r, "VACF persistence",
                               f"{vp:.3f}"); r += 1

        # Motion-class composition — a stacked proportion bar + a swatch
        # legend (replaces the old flat list of coloured rows).
        motion_counts = summary.get("motion_counts") or {}
        if motion_counts:
            total = sum(motion_counts.values()) or 1
            order = ["Immobile", "Confined", "Brownian", "Directed", "Unknown"]
            colour_map = dict(_MOTION_PALETTE)   # same colours as figures/napari
            present = [(cls, int(motion_counts[cls]), colour_map.get(cls))
                       for cls in order
                       if cls in motion_counts and motion_counts[cls] > 0]
            if present:
                self._add_section_header(r, "Motion classes"); r += 1
                self._stats_grid.addWidget(_StackedBar(present), r, 0, 1, 2)
                r += 1
                for cls, n, col in present:
                    self._stats_grid.addWidget(
                        self._swatch_label(cls, col), r, 0,
                        Qt.AlignmentFlag.AlignLeft)
                    val = QtWidgets.QLabel(f"{n:,}  ({100 * n / total:.1f} %)")
                    val.setStyleSheet(
                        "color: %s; font-size: 14px; font-weight: 600;" % col)
                    val.setTextInteractionFlags(
                        Qt.TextInteractionFlag.TextSelectableByMouse)
                    self._stats_grid.addWidget(
                        val, r, 1, Qt.AlignmentFlag.AlignLeft)
                    r += 1

        # Clustering & acquisition ───────────────────────────────────
        nc = summary.get("n_clusters", 0)
        dwell = summary.get("dwell_tau_s")
        frames = summary.get("frames")
        if nc or dwell is not None or frames:
            self._add_section_header(r, "Clustering & acquisition"); r += 1
        if nc:
            _sub = summary.get("cluster_subsampled_n")
            _cval = f"{_fmt_int(nc)}  (subsampled)" if _sub else _fmt_int(nc)
            _ctip = (f"DBSCAN ran on a {int(_sub):,}-localisation subsample "
                     f"(the dataset exceeded the 250k cap), so the cluster "
                     f"count and per-cluster stats reflect that subset."
                     if _sub else None)
            self._add_stat_row(r, "DBSCAN clusters", _cval,
                               tooltip=_ctip); r += 1
        if dwell is not None:
            self._add_stat_row(r, "Dwell time  τ",
                               _fmt_secs(dwell)); r += 1
        if frames:
            self._add_stat_row(
                r, "Source movie",
                f"{frames:,} frames  |  "
                f"px = {summary.get('px_um', 0):.3f} µm  |  "
                f"fi = {summary.get('fi_s', 0):.3f} s",
                value_colour=_THEME['TXT_MUTED']); r += 1

        # ── Quality control ──────────────────────────────────────────
        qc = summary.get("qc") or {}
        if qc:
            self._add_section_header(r, "Quality control"); r += 1

            lr = qc.get("link_ratio")
            if lr is not None:
                col = (_THEME['DANGER'] if lr < 0.10
                       else _THEME['WARN'] if lr < 0.25
                       else _THEME['SUCCESS'])
                self._add_stat_row(r, "Localisations linked",
                                   _fmt_pct(lr),
                                   value_colour=col); r += 1
            avg_pf = qc.get("avg_locs_per_frame")
            if avg_pf is not None:
                col = (_THEME['WARN'] if avg_pf > 800
                       else _THEME['TXT'])
                self._add_stat_row(r, "Locs / frame (avg)",
                                   f"{avg_pf:,.1f}",
                                   value_colour=col); r += 1
            ml = qc.get("median_track_length")
            if ml is not None:
                col = (_THEME['WARN'] if ml < 6 else _THEME['TXT'])
                self._add_stat_row(r, "Median track length",
                                   f"{ml:.1f}  frames",
                                   value_colour=col); r += 1
            gf = qc.get("gap_fraction")
            if gf is not None:
                self._add_stat_row(r, "Tracks with gaps",
                                   _fmt_pct(gf)); r += 1
            sf = qc.get("stuck_fraction")
            if sf is not None:
                col = (_THEME['WARN'] if sf > 0.30 else _THEME['TXT'])
                self._add_stat_row(
                    r, "Stuck tracks", _fmt_pct(sf), value_colour=col,
                    tooltip="Fraction of tracks with D < 1e-3 µm²/s — likely "
                            "stuck or aggregated particles."); r += 1
            dn = qc.get("drift_total_nm")
            if dn is not None:
                col = (_THEME['WARN'] if dn > 500 else _THEME['TXT'])
                self._add_stat_row(r, "Drift corrected (total)",
                                   f"{dn:.0f} nm",
                                   value_colour=col); r += 1

            # QC flags → severity banners (rendered in the dedicated
            # _flags_layout QVBoxLayout so wrapped text keeps its full height).
            # The worker's `msg` already carries the remedy, so we keep it
            # verbatim and just prepend a short bold lead derived from it.
            flags = qc.get("flags") or []
            levels = {str(f.get("level", "info")).lower() for f in flags}
            for f in flags:
                level = str(f.get("level", "info")).lower()
                sev = level if level in ("danger", "warn", "info") else "info"
                msg = str(f.get("msg", ""))
                lead = self._qc_flag_lead(msg)
                html = f"<b>{lead}</b><br>{msg}" if lead else msg
                self._flags_layout.addWidget(_AlertBanner(sev, html))
            if flags:
                self._flags_container.show()

            # Run-readiness pill from the flag levels.
            if levels & {"warn", "danger"}:
                self._qc_badge.set_state("blocked", "Completed with warnings")
            else:
                self._qc_badge.set_state("ready", "Analysis successful")
            self._qc_badge.show()

        self._stats_container.show()

    @staticmethod
    def _qc_flag_lead(msg: str) -> str:
        """A short bold headline for a QC flag, derived from its message text
        (the message itself already contains the detail + remedy)."""
        m = (msg or "").lower()
        if "linked" in m or "link ratio" in m:          return "Low link ratio"
        if "density" in m or "locs/frame" in m:         return "High density"
        if "track length" in m:                         return "Short tracks"
        if "stuck" in m or "aggregated" in m:           return "Stuck tracks"
        if "gap" in m:                                  return "Track gaps"
        if "drift" in m:                                return "Drift corrected"
        return ""

    def show_results(self, headline: str, out_dir: str,
                     files: list[str] | None = None):
        """Populate the panel with a completed run's outputs."""
        self._headline.setText(headline)
        self._headline.setStyleSheet(
            f"color: {_THEME['SUCCESS']}; font-size: 16px; font-weight: 700;")
        self._headline.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self._out_dir = out_dir
        if out_dir:
            self._folder_label.setText(out_dir)
            self._folder_row.show()

        self._files.clear()
        files = files or []
        # Auto-discover saved outputs if the caller didn't pass them
        if out_dir and os.path.isdir(out_dir) and not files:
            for sub in ("data", "firefly_extras", "figures", ""):
                d = os.path.join(out_dir, sub) if sub else out_dir
                if not os.path.isdir(d):
                    continue
                for name in sorted(os.listdir(d)):
                    full = os.path.join(d, name)
                    if os.path.isfile(full):
                        files.append(full)
        for f in files:
            item = QtWidgets.QListWidgetItem(
                f"  {os.path.relpath(f, out_dir) if out_dir else f}")
            item.setData(Qt.ItemDataRole.UserRole, f)
            item.setToolTip(f)
            self._files.addItem(item)
        if files:
            self._files.show()

    def _on_open_folder(self):
        if self._out_dir and os.path.isdir(self._out_dir):
            _open_folder(self._out_dir)

    def _on_file_dbl(self, item: QtWidgets.QListWidgetItem):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path and os.path.isfile(path):
            _open_folder(os.path.dirname(path))


class _RoiDialog(QtWidgets.QDialog):
    """Modal ROI editor.  Loads the mean projection of an input file into
    an embedded napari viewer with a Shapes layer in polygon mode.  User
    draws one or more polygons; clicking Save returns the vertices.

    Vertices are stored as (y, x) coordinate pairs in pixels of the
    original (Y, X) frame — directly consumable by
    skimage.draw.polygon2mask.
    """

    def __init__(self, file_path: str,
                 current_polygons: list | None = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"ROI — {os.path.basename(file_path)}")
        self.resize(1000, 760)
        self._file_path = file_path
        self._result_polygons: list[list[tuple[float, float]]] = []
        self._viewer = None
        self._shapes_layer = None

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(8)

        # ── Instructions ────────────────────────────────────────────────
        hint = QtWidgets.QLabel(
            "Use the <b>polygon</b> tool in the layer controls (top-left) "
            "to draw a region of interest.  Click points to add vertices, "
            "then press <b>Esc</b> or right-click to finish the polygon.  "
            "You can draw multiple polygons — they'll be combined into one "
            "ROI mask.  Save when done; Cancel discards changes."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_THEME['TXT_MUTED']}; padding: 4px 0;")
        v.addWidget(hint)

        # ── Status line ─────────────────────────────────────────────────
        self._status = QtWidgets.QLabel("Loading preview…")
        self._status.setStyleSheet(f"color: {_THEME['TXT_MUTED']};")
        v.addWidget(self._status)

        # ── Embedded napari viewer (placeholder until lazy-init) ────────
        self._viewer_container = QtWidgets.QWidget()
        self._viewer_layout = QtWidgets.QVBoxLayout(self._viewer_container)
        self._viewer_layout.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self._viewer_container, stretch=1)

        # ── Buttons ─────────────────────────────────────────────────────
        btn_row = QtWidgets.QHBoxLayout()
        self._b_clear = QtWidgets.QPushButton("Clear ROI")
        self._b_clear.setToolTip("Remove all polygons (file will fall back to "
                                  "the global ROI mode in settings).")
        self._b_clear.clicked.connect(self._on_clear)
        btn_row.addWidget(self._b_clear)
        btn_row.addStretch(1)
        b_cancel = QtWidgets.QPushButton("Cancel")
        b_cancel.clicked.connect(self.reject)
        btn_row.addWidget(b_cancel)
        b_save = QtWidgets.QPushButton("Save ROI")
        b_save.setObjectName("primary")
        b_save.clicked.connect(self._on_save)
        btn_row.addWidget(b_save)
        v.addLayout(btn_row)

        # Defer the heavy lifting (napari init + file load) so the dialog
        # appears immediately with the "Loading preview…" status.
        QtCore.QTimer.singleShot(50, lambda: self._init_viewer(current_polygons))

    @staticmethod
    def _quick_preview(file_path: str, max_frames: int = 30):
        """Read just enough of `file_path` to render a representative
        background image for ROI drawing.  Returns a 2D float32 array
        of shape (Y, X), or raises.

        This DELIBERATELY does not use `load_file` — that loads the full
        stack (and for multi-file TIF series, concatenates them), which
        on a tight-RAM machine can take minutes and trigger swap.  For
        the ROI editor we only need a clear preview, not the full data,
        so we read just the first `max_frames` pages of the first file
        directly via tifffile / aicspylibczi.
        """
        import os as _os
        import numpy as _np
        ext = _os.path.splitext(file_path)[1].lower()

        if ext in (".tif", ".tiff"):
            import tifffile
            with tifffile.TiffFile(file_path) as tif:
                n_pages = len(tif.pages)
                n = min(max_frames, n_pages)
                # Sample evenly across the (single) file so blinks /
                # bleaches don't dominate the preview
                if n_pages > n:
                    idx = _np.linspace(0, n_pages - 1, n, dtype=int)
                else:
                    idx = _np.arange(n_pages)
                frames = []
                for i in idx:
                    frames.append(tif.pages[int(i)].asarray()
                                  .astype(_np.float32))
                return _np.mean(_np.stack(frames), axis=0)

        if ext == ".czi":
            from aicspylibczi import CziFile
            czi = CziFile(file_path)
            # Read first frame.  CZI reads can return shape (1, 1, 1, Y, X)
            # or similar depending on dim order — squeeze to (Y, X).
            try:
                img, _ = czi.read_image(T=0)
            except Exception:
                # Some CZIs have different dim names; fall back to
                # reading the whole thing if T isn't a valid dim
                img = czi.read_mosaic(C=0, scale_factor=1)
            arr = _np.squeeze(_np.asarray(img))
            # If we accidentally got >2D (multichannel etc.) take a mean
            while arr.ndim > 2:
                arr = arr.mean(axis=0)
            return arr.astype(_np.float32)

        raise ValueError(f"Unsupported file extension: {ext}")

    @staticmethod
    def _quick_preview_stack(file_path: str, max_frames: int = 30):
        """Like `_quick_preview` but returns a 3D (T, Y, X) stack instead of
        the mean.  Used by the embedded ROI viewer so the user can scrub
        through real frames and live-preview detections."""
        import os as _os
        import numpy as _np
        ext = _os.path.splitext(file_path)[1].lower()

        if ext in (".tif", ".tiff"):
            import tifffile
            with tifffile.TiffFile(file_path) as tif:
                n_pages = len(tif.pages)
                n = min(max_frames, n_pages)
                if n_pages > n:
                    idx = _np.linspace(0, n_pages - 1, n, dtype=int)
                else:
                    idx = _np.arange(n_pages)
                # Batched asarray(key=...) with internal multithreading
                # is ~5× faster than the old per-page asarray() loop —
                # tifffile re-initialises its codec on every per-page
                # call, so the loop ran the codec setup N times.
                try:
                    import os as _os2
                    workers = max(1, (_os2.cpu_count() or 1) // 2)
                    arr = tif.asarray(key=[int(i) for i in idx],
                                       maxworkers=workers)
                    # asarray may return 2-D for a single page; normalise
                    # to (T, Y, X) regardless.
                    if arr.ndim == 2:
                        arr = arr[None, ...]
                    return arr.astype(_np.float32, copy=False), [int(i) for i in idx]
                except Exception:
                    # Fallback to the old per-page path if batched read
                    # fails (rare — old tifffile versions).
                    frames = [tif.pages[int(i)].asarray().astype(_np.float32)
                              for i in idx]
                    return _np.stack(frames), [int(i) for i in idx]

        if ext == ".czi":
            from aicspylibczi import CziFile
            czi = CziFile(file_path)
            frames = []
            indices = []
            for t in range(max_frames):
                try:
                    img, _ = czi.read_image(T=t)
                except Exception:
                    break
                arr = _np.squeeze(_np.asarray(img))
                while arr.ndim > 2:
                    arr = arr.mean(axis=0)
                frames.append(arr.astype(_np.float32))
                indices.append(t)
            if not frames:
                raise ValueError("No frames could be read from CZI")
            return _np.stack(frames), indices

        raise ValueError(f"Unsupported file extension: {ext}")

    def _init_viewer(self, current_polygons):
        try:
            import napari
        except Exception as exc:
            self._status.setText(
                f"napari isn't installed: {exc}.\n"
                f"Run `pip install \"napari[pyside6]>=0.4.19\"` and restart.")
            return

        try:
            self._viewer = napari.Viewer(show=False)
            qt_window = self._viewer.window._qt_window
            # Seal the OUTER container so napari's internal size hints
            # can't propagate up and grow the parent FIREFLY window.
            # Napari itself is left completely untouched — its dim
            # slider / layer panel / canvas all work as designed.
            _make_napari_container_layout_opaque(self._viewer_container)
            self._viewer_layout.addWidget(qt_window)
            _hide_napari_chrome(self._viewer)
        except Exception as exc:
            self._status.setText(f"Couldn't embed napari viewer: {exc}")
            return

        # Load just enough to render an ROI background.  No full-stack load,
        # no concat — see _quick_preview's docstring.  Synchronous on the
        # dialog's event loop but the read is tiny (~30 frames).
        try:
            import numpy as _np
            mean_img = self._quick_preview(self._file_path, max_frames=30)
            self._viewer.add_image(mean_img, name="ROI background",
                                    colormap="gray")
            # Shapes layer for the polygon
            initial_shapes = [_np.asarray(poly)
                              for poly in (current_polygons or [])]
            self._shapes_layer = self._viewer.add_shapes(
                data=initial_shapes if initial_shapes else None,
                shape_type="polygon",
                edge_color="#58a6ff",
                face_color="rgba(88,166,255,0.18)",
                edge_width=2,
                name="ROI",
            )
            # Switch to polygon-add mode so the user can start drawing
            try:
                self._shapes_layer.mode = "add_polygon"
            except Exception:
                pass
            self._status.setText(
                f"{mean_img.shape[1]} × {mean_img.shape[0]} px preview "
                f"(quick load).  Draw polygon(s) on the ROI layer; "
                "right-click or Esc to close each polygon.")
        except Exception as exc:
            import traceback as _tb
            self._status.setText(
                f"Couldn't load file preview: {exc}\n\n"
                f"{_tb.format_exc()}")

    def _polygons_from_layer(self) -> list[list[tuple[float, float]]]:
        """Pull current polygon vertices out of the Shapes layer.
        Each entry is a list of (y, x) tuples."""
        polys: list[list[tuple[float, float]]] = []
        if self._shapes_layer is None:
            return polys
        try:
            for shape_data, shape_type in zip(self._shapes_layer.data,
                                              self._shapes_layer.shape_type):
                if shape_type not in ("polygon", "rectangle", "ellipse"):
                    continue
                if shape_type == "polygon":
                    polys.append([(float(y), float(x))
                                  for y, x in shape_data])
                else:
                    # Rectangles / ellipses are stored as 4-vertex bounding
                    # boxes; treat as polygons (rectangle is exact, ellipse
                    # is approximated by its bounding box for now).
                    polys.append([(float(y), float(x))
                                  for y, x in shape_data])
        except Exception:
            pass
        return polys

    def _on_save(self):
        self._result_polygons = self._polygons_from_layer()
        self.accept()

    def _on_clear(self):
        self._result_polygons = []
        self.accept()

    def result_polygons(self) -> list[list[tuple[float, float]]]:
        return self._result_polygons


# ROI-file parsers now live in firefly.analysis.fa_roi (Qt-free, so the
# analysis worker can reuse them for sibling-ROI auto-detect).  Re-exported
# here so existing GUI imports (`from ...ui_widgets import _load_any_roi_file`,
# etc.) keep working unchanged.
from firefly.analysis.fa_roi import (        # noqa: E402,F401
    _load_imagej_roi_polygons,
    _load_tif_mask_polygons,
    _load_any_roi_file,
)


class _RoiViewer(QtWidgets.QWidget):
    """Inline ROI editor for the Import tab.

    Same drawing model as the modal `_RoiDialog` (embedded napari + Shapes
    layer with polygon tool), but lives inside the tab and supports
    switching between files via `set_file`.  Polygons are auto-emitted as
    they change so the host MainWindow can save them per file without
    requiring an explicit Save button.
    """

    polygons_changed = QtCore.Signal(str, list)  # (file_path, polygons)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_file: str = ""
        self._viewer = None
        self._shapes_layer = None
        self._image_layer  = None
        self._points_layer = None
        self._stack = None             # cached 3-D preview stack (raw)
        self._stack_filtered = None    # cached bandpass-filtered version (lazy)
        self._stack_preprocessed = None  # cached pipeline-preprocessed stack
        self._pp_signature = None      # (bg_method, bg_radius) of cached stack
        self._last_mass = None         # mass array from the most-recent locate
        self._roi_mask_layer = None    # auto/manual-threshold overlay layer
        self._max_proj_layer = None    # static max-projection anatomy layer
        self._roi_mask_params = {"mode": "None", "auto_method": "li",
                                 "threshold": 0.08, "mask_mode": "mean",
                                 "bg_sigma": 25.0}
        # When true, _on_layer_removed is a no-op.  Used by set_file
        # while it tears down the previous file's layers — we DON'T want
        # the "user deleted our ROI, recreate it" recovery path to fire
        # during a programmatic teardown, because re-entering
        # add_shapes() mid-clear corrupts napari's layer iterator and
        # freezes the GUI.
        self._suppress_layer_events = False
        self._lazy_init_pending = True
        self._detect_enabled = False
        self._detect_params  = {"diameter": 7, "minmass": 1.0,
                                "bg_method": "uniform_filter",
                                "bg_radius": 50}
        self._dims_connected = False
        self._detect_debounce = QTimer(self)
        self._detect_debounce.setSingleShot(True)
        self._detect_debounce.setInterval(250)
        self._detect_debounce.timeout.connect(self._run_detection)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        header = QtWidgets.QHBoxLayout()
        self._title = QtWidgets.QLabel("Preview viewer")
        self._title.setStyleSheet(
            f"color: {_THEME['TXT']}; font-weight: 600; font-size: 13px;")
        header.addWidget(self._title)
        self._status = QtWidgets.QLabel("Pick a file to start")
        self._status.setStyleSheet(f"color: {_THEME['TXT_MUTED']};")
        header.addWidget(self._status, 1)
        # Turbo-colormap legend bar (low mass → high mass).
        legend_w = QtWidgets.QWidget()
        legend_w.setToolTip(
            "Detections are coloured by integrated mass on a log scale using "
            "the 'turbo' colormap, auto-stretched per frame.  Dim spots "
            "(likely noise) sit at the blue / purple end; bright spots "
            "(likely real PSFs) sit at the red end.  Raise minmass and the "
            "blue end disappears first.")
        lh = QtWidgets.QHBoxLayout(legend_w)
        lh.setContentsMargins(8, 0, 0, 0)
        lh.setSpacing(4)
        lbl_lo = QtWidgets.QLabel("dim")
        lbl_lo.setStyleSheet(f"color: {_THEME['TXT_MUTED']}; font-size: 10px;")
        lh.addWidget(lbl_lo)
        bar = QtWidgets.QFrame()
        bar.setFixedSize(120, 10)
        # Turbo colormap stops (approximate, perceptually-uniform)
        bar.setStyleSheet(
            "QFrame { border: 1px solid #2d3138; border-radius: 2px; "
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #30123b, stop:0.15 #4661e0, stop:0.30 #1ce5d5, "
            "stop:0.50 #6cfd62, stop:0.70 #fdbb2d, stop:0.85 #f06b1d, "
            "stop:1 #7a0402); }")
        lh.addWidget(bar)
        lbl_hi = QtWidgets.QLabel("bright")
        lbl_hi.setStyleSheet(f"color: {_THEME['TXT_MUTED']}; font-size: 10px;")
        lh.addWidget(lbl_hi)
        header.addWidget(legend_w)
        # Bandpass-filtered view toggle — shows what trackpy actually sees
        # after its preprocessing step, which makes real PSFs pop and flat
        # background noise drop away.  Useful for picking a good minmass.
        self._cb_filtered = QtWidgets.QCheckBox("Filtered view")
        self._cb_filtered.setToolTip(
            "Show the bandpass-filtered image (what trackpy sees) instead of\n"
            "the raw frame.  Real PSFs come up bright on a flat dark background;\n"
            "noise stays small.  Detection runs against this same filtering\n"
            "internally, so what you see is closer to what the detector decides.")
        self._cb_filtered.toggled.connect(self._on_filtered_toggled)
        header.addWidget(self._cb_filtered)
        self._b_clear = QtWidgets.QPushButton("Clear polygons")
        self._b_clear.setToolTip(
            "Remove every polygon drawn on the current file's ROI.")
        self._b_clear.clicked.connect(self._on_clear)
        header.addWidget(self._b_clear)
        # "Load ROI…" — reads an ImageJ / palmTRACER .roi or .zip file
        # and adds the polygons to the current file's ROI set.  Works
        # in single-file, batch, and external-CSV modes because the
        # ROI viewer is shared across all three import flows.
        self._b_load_roi = QtWidgets.QPushButton("Load ROI…")
        self._b_load_roi.setToolTip(
            "Load polygon ROI(s) from an ImageJ / palmTRACER .roi or .zip\n"
            "file.  Coordinates are interpreted as image pixels (matching\n"
            "the active file's preview).  Adds to any polygons already drawn.")
        self._b_load_roi.clicked.connect(self._on_load_roi_file)
        header.addWidget(self._b_load_roi)
        v.addLayout(header)

        self._viewer_container = QtWidgets.QFrame()
        self._viewer_container.setObjectName("results_panel")
        # Modest minimum so the Re-process and Import tabs stay
        # vertical-compressible on small (1366×768 / 1440×900) laptops.
        # The container expands freely upwards via its parent's stretch
        # factor when there's room — this just sets a floor.
        self._viewer_container.setMinimumHeight(200)
        self._viewer_layout = QtWidgets.QVBoxLayout(self._viewer_container)
        self._viewer_layout.setContentsMargins(0, 0, 0, 0)
        # Placeholder until napari is loaded (lazy)
        self._placeholder = QtWidgets.QLabel(
            "Pick a file above and the ROI viewer will load here."
        )
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']}; padding: 40px;")
        self._viewer_layout.addWidget(self._placeholder)
        v.addWidget(self._viewer_container, stretch=1)

    # ── Lazy napari init ─────────────────────────────────────────────────
    def _ensure_viewer(self) -> bool:
        if self._viewer is not None:
            return True
        try:
            import napari
        except Exception as exc:
            self._status.setText(
                f"napari not available: {exc} — install it to use the preview viewer.")
            return False
        try:
            self._viewer = napari.Viewer(show=False)
            qt_window = self._viewer.window._qt_window
            # Seal the OUTER container so napari's internal size hints
            # can't propagate up and grow the parent FIREFLY window.
            # Napari itself is left completely untouched.
            _make_napari_container_layout_opaque(self._viewer_container)
            self._viewer_layout.removeWidget(self._placeholder)
            self._placeholder.hide()
            self._viewer_layout.addWidget(qt_window)
            # Hide napari's "Drag image(s) here" welcome overlay + the
            # bottom-of-canvas viewer-buttons row (ndisplay / grid / home /
            # console / etc.) + the new-layer/delete buttons under the
            # layer list — all visual noise for a viewer driven entirely
            # programmatically by FIREFLY.
            _hide_napari_chrome(self._viewer)
            # Re-run detection when the user scrubs frames (idempotent)
            if not self._dims_connected:
                try:
                    self._viewer.dims.events.current_step.connect(
                        self._on_dims_changed)
                    self._dims_connected = True
                except Exception:
                    pass
            # Auto-recover if the user deletes a layer we own (e.g. the
            # ROI shapes layer via napari's trash button).
            try:
                self._viewer.layers.events.removed.connect(
                    self._on_layer_removed)
            except Exception:
                pass
            return True
        except Exception as exc:
            self._status.setText(f"Couldn't embed napari viewer: {exc}")
            return False

    def _on_layer_removed(self, event):
        """Recover from the user deleting one of our managed layers.

        Shapes layer → recreate empty (so they can keep drawing polygons,
        and let the host know the previous polygons are gone).
        Image / points / mask layers → just null the stored reference;
        the next set_file / detection run / mask-refresh will re-create
        them lazily.
        """
        # Programmatic layer teardown (e.g. set_file clearing the old
        # file's layers before loading a new one) sets this flag so we
        # don't fight napari's iterator by reinserting layers mid-clear.
        if self._suppress_layer_events:
            return
        try:
            removed = getattr(event, "value", None)
        except Exception:
            removed = None
        if removed is None or self._viewer is None:
            return
        if removed is self._shapes_layer:
            try:
                import numpy as _np
                self._shapes_layer = self._viewer.add_shapes(
                    data=None,
                    shape_type="polygon",
                    edge_color="#58a6ff",
                    face_color="rgba(88,166,255,0.18)",
                    edge_width=2,
                    name="ROI",
                )
                try:    self._shapes_layer.mode = "add_polygon"
                except Exception: pass
                try:    self._shapes_layer.events.data.connect(
                            self._on_shapes_changed)
                except Exception: pass
                # Notify host that polygons for the current file have
                # been wiped (matches the napari state on disk).
                if self._current_file:
                    self.polygons_changed.emit(self._current_file, [])
                self._status.setText(
                    "ROI layer was deleted — recreated empty.  "
                    "Draw a new polygon to set the ROI.")
            except Exception:
                self._shapes_layer = None
            return
        if removed is self._image_layer:
            self._image_layer = None
            return
        if removed is self._points_layer:
            self._points_layer = None
            return
        if removed is self._roi_mask_layer:
            self._roi_mask_layer = None
            return

    # ── Public API ───────────────────────────────────────────────────────
    def set_file(self, file_path: str,
                 current_polygons: list | None = None):
        """Switch the viewer to `file_path` and load its preview.

        Auto-emits the current polygons as `polygons_changed` whenever
        the user adds / edits / removes a shape, so the host MainWindow
        can persist the change without an explicit Save button.
        """
        # If we have an active file with edits, flush them before switching.
        self._flush_current_polygons_if_changed()

        self._current_file = file_path or ""

        if not file_path:
            self._title.setText("Preview viewer")
            self._status.setText("Pick a file to start")
            return
        if not os.path.isfile(file_path):
            self._status.setText(f"File not found: {file_path}")
            return

        if not self._ensure_viewer():
            return

        # Clear out any old layers from a previous file.  Wrap the clear
        # in `_suppress_layer_events = True` so _on_layer_removed doesn't
        # try to recreate layers MID-clear — that recursion is what
        # froze the GUI when switching files.
        self._suppress_layer_events = True
        try:
            try:    self._viewer.layers.clear()
            except Exception: pass
        finally:
            self._suppress_layer_events = False
        self._image_layer = None
        self._shapes_layer = None
        self._points_layer = None
        self._stack = None
        self._stack_filtered = None
        self._stack_preprocessed = None
        self._pp_signature = None
        self._last_mass = None
        self._roi_mask_layer = None
        self._max_proj_layer = None
        # Reset the toggle silently so it doesn't fight the new image
        try:
            self._cb_filtered.blockSignals(True)
            self._cb_filtered.setChecked(False)
            self._cb_filtered.blockSignals(False)
        except AttributeError:
            pass

        self._title.setText(f"Preview — {os.path.basename(file_path)}")
        self._status.setText("Loading preview…")

        try:
            import numpy as _np
            stack, _idx = _RoiDialog._quick_preview_stack(file_path, max_frames=30)
            self._stack = stack
            # Percentile-based contrast so a few hot pixels don't blow out the
            # display.  Sample a single mid-stack frame for speed.
            sample = stack[stack.shape[0] // 2]
            lo, hi = _np.percentile(sample, [1.0, 99.5])
            if hi <= lo:
                hi = lo + 1.0
            self._image_layer = self._viewer.add_image(
                stack, name="ROI background", colormap="gray",
                contrast_limits=(float(lo), float(hi)))
            # Anatomy layer — max projection across the sampled frames.
            # Individual frames in sptPALM data are mostly sparse blinks
            # on noise: hard to see where the cells actually are.  The
            # max projection reveals the underlying structure (each pixel
            # = its brightest moment over the sample) so the user knows
            # where to draw a manual polygon AND can sanity-check the
            # auto-threshold mask against true anatomy.  Off by default
            # (eye icon toggles it) so it doesn't surprise users used to
            # the old single-layer view.
            try:
                max_proj = stack.max(axis=0).astype(_np.float32)
                lo_m, hi_m = _np.percentile(max_proj, [1.0, 99.5])
                if hi_m <= lo_m:
                    hi_m = lo_m + 1.0
                self._max_proj_layer = self._viewer.add_image(
                    max_proj, name="Max projection",
                    colormap="inferno",
                    contrast_limits=(float(lo_m), float(hi_m)),
                    opacity=0.85, visible=False, blending="additive")
            except Exception:
                self._max_proj_layer = None
            initial_shapes = [_np.asarray(poly)
                              for poly in (current_polygons or [])]
            self._shapes_layer = self._viewer.add_shapes(
                data=initial_shapes if initial_shapes else None,
                shape_type="polygon",
                edge_color="#58a6ff",
                face_color="rgba(88,166,255,0.18)",
                edge_width=2,
                name="ROI",
            )
            try:
                self._shapes_layer.mode = "add_polygon"
            except Exception:
                pass
            # Auto-save: emit whenever the shapes layer changes
            try:
                self._shapes_layer.events.data.connect(self._on_shapes_changed)
            except Exception:
                pass
            self._status.setText(
                f"{stack.shape[0]}-frame preview, "
                f"{stack.shape[2]} × {stack.shape[1]} px — "
                "draw polygon(s); right-click or Esc to close each one.")
            # Re-arm detection preview if the host has it enabled.
            if self._detect_enabled:
                self._detect_debounce.start()
            # Re-draw the auto / manual-threshold mask overlay if the
            # host previously set its parameters.
            self._refresh_roi_mask_overlay()
        except Exception as exc:
            import traceback as _tb
            self._status.setText(
                f"Couldn't load preview: {exc}\n{_tb.format_exc()}")

    def set_image_array(self, image: "np.ndarray", label: str = "background",
                        current_polygons: list | None = None) -> None:
        """Show a pre-computed 2-D image as the ROI background.

        Used by the Post-process tab when the original input file isn't
        available on disk — we feed it a 2-D histogram of the run's
        localisations instead, so the user still has an anatomy proxy
        to draw the ROI on.
        """
        import numpy as _np
        self._flush_current_polygons_if_changed()
        # Synthetic source path so polygon save/restore keyed off
        # `_current_file` still works (each loaded run gets its own slot).
        self._current_file = f"<postproc:{label}>"

        if not self._ensure_viewer():
            return

        self._suppress_layer_events = True
        try:
            try:    self._viewer.layers.clear()
            except Exception: pass
        finally:
            self._suppress_layer_events = False
        self._image_layer = None
        self._shapes_layer = None
        self._points_layer = None
        self._stack = None
        self._stack_filtered = None
        self._stack_preprocessed = None
        self._pp_signature = None
        self._last_mass = None
        self._roi_mask_layer = None
        self._max_proj_layer = None
        try:
            self._cb_filtered.blockSignals(True)
            self._cb_filtered.setChecked(False)
            self._cb_filtered.blockSignals(False)
        except AttributeError:
            pass

        self._title.setText(f"Preview — {label}")
        self._status.setText("Rendering background…")
        try:
            arr = _np.asarray(image, dtype=_np.float32)
            if arr.ndim != 2:
                raise ValueError(f"expected 2-D image, got shape {arr.shape}")
            lo, hi = _np.percentile(arr, [1.0, 99.5])
            if hi <= lo:
                hi = lo + 1.0
            self._image_layer = self._viewer.add_image(
                arr, name="ROI background", colormap="inferno",
                contrast_limits=(float(lo), float(hi)))
            initial_shapes = [_np.asarray(poly)
                              for poly in (current_polygons or [])]
            self._shapes_layer = self._viewer.add_shapes(
                data=initial_shapes if initial_shapes else None,
                shape_type="polygon",
                edge_color="#58a6ff",
                face_color="rgba(88,166,255,0.18)",
                edge_width=2,
                name="ROI",
            )
            try:
                self._shapes_layer.mode = "add_polygon"
            except Exception:
                pass
            try:
                self._shapes_layer.events.data.connect(self._on_shapes_changed)
            except Exception:
                pass
            self._status.setText(
                f"{arr.shape[1]} × {arr.shape[0]} px — "
                "draw polygon(s); right-click or Esc to close each one.")
        except Exception as exc:
            import traceback as _tb
            self._status.setText(
                f"Couldn't render background: {exc}\n{_tb.format_exc()}")

    def current_file(self) -> str:
        return self._current_file

    def current_polygons(self) -> list:
        polys = []
        if self._shapes_layer is None:
            return polys
        try:
            for shape_data, shape_type in zip(self._shapes_layer.data,
                                              self._shapes_layer.shape_type):
                if shape_type in ("polygon", "rectangle", "ellipse"):
                    polys.append([(float(y), float(x))
                                  for y, x in shape_data])
        except Exception:
            pass
        return polys

    # ── Internal ─────────────────────────────────────────────────────────
    def _flush_current_polygons_if_changed(self):
        """Emit a final polygons_changed for the outgoing file, in case
        the user drew something but never triggered the data-changed event."""
        if self._current_file and self._shapes_layer is not None:
            try:
                self.polygons_changed.emit(
                    self._current_file, self.current_polygons())
            except Exception:
                pass

    def _on_shapes_changed(self, _event=None):
        if self._current_file:
            try:
                self.polygons_changed.emit(
                    self._current_file, self.current_polygons())
            except Exception:
                pass

    def _on_clear(self):
        if self._shapes_layer is None:
            return
        try:
            self._shapes_layer.data = []
        except Exception:
            pass
        if self._current_file:
            self.polygons_changed.emit(self._current_file, [])

    # ── palmTRACER / ImageJ ROI loader ───────────────────────────────────
    def _on_load_roi_file(self):
        """Pick a `.roi` / `.zip` / `.tif` file and append its polygons
        to the current file's ROI.  No-op when no file is currently loaded.

        palmTRACER writes ROIs in two formats:
          • `.roi` / `.zip`  — ImageJ binary ROI (vector polygons)
          • `.tif` / `.tiff` — raster mask of the drawn region
        We accept both; the dispatch is by file extension.
        """
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load ImageJ / palmTRACER ROI",
            "",
            "All ROI formats (*.roi *.zip *.tif *.tiff);;"
            "ImageJ ROI (*.roi *.zip);;"
            "palmTRACER ROI mask (*.tif *.tiff);;"
            "All files (*)")
        if not path:
            return
        try:
            new_polys = _load_any_roi_file(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Couldn't load ROI",
                f"Failed to read {os.path.basename(path)}:\n\n{exc}")
            return
        if not new_polys:
            ext = os.path.splitext(path.lower())[1]
            if ext in (".tif", ".tiff"):
                msg = (f"{os.path.basename(path)} decoded but contained "
                       f"no non-zero pixels — make sure you picked the "
                       f"ROI mask itself, not the raw movie.")
            else:
                msg = (f"{os.path.basename(path)} contained no closed "
                       f"polygons (only line / point / text ROIs are "
                       f"unsupported).")
            QtWidgets.QMessageBox.information(self, "No polygons", msg)
            return
        # Append to the existing shapes layer, falling back to creating
        # one if napari isn't initialised yet.
        try:
            if self._shapes_layer is None:
                if not self._ensure_viewer():
                    return
                import napari   # noqa: F401  (ensure import)
                # If there's no image yet, add the shapes layer empty.
                self._shapes_layer = self._viewer.add_shapes(
                    shape_type="polygon",
                    edge_color="cyan", face_color="transparent",
                    edge_width=2, name="ROIs")
            existing = list(self._shapes_layer.data) \
                if self._shapes_layer.data is not None else []
            self._shapes_layer.data = existing + new_polys
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "ROI applied with warning",
                f"Polygons loaded but the napari layer raised: {exc}")
        if self._current_file:
            try:
                polys_out = list(self._shapes_layer.data)
            except Exception:
                polys_out = new_polys
            self.polygons_changed.emit(self._current_file, polys_out)

    # ── Detection preview ────────────────────────────────────────────────
    def enable_detection_preview(self, enabled: bool):
        """Toggle a `tp.locate` overlay on the current frame."""
        self._detect_enabled = bool(enabled)
        if not self._detect_enabled:
            self._remove_points_layer()
            return
        if self._stack is None:
            return
        self._detect_debounce.start()

    def set_detection_params(self, *, diameter: int, minmass: float,
                             bg_method: str = "uniform_filter",
                             bg_radius: int = 50):
        """Update the diameter / minmass / preprocessing used by the
        live overlay.  Matching the pipeline's preprocessing is what makes
        the mass scale here agree with what the run will actually see."""
        new_sig = (str(bg_method), int(bg_radius))
        if self._pp_signature is not None and new_sig != self._pp_signature:
            # Background settings changed → invalidate cached preprocessed stack
            self._stack_preprocessed = None
        self._detect_params = {"diameter": int(diameter),
                               "minmass":  float(minmass),
                               "bg_method": str(bg_method),
                               "bg_radius": int(bg_radius)}
        if self._detect_enabled and self._stack is not None:
            self._detect_debounce.start()

    def _on_dims_changed(self, _evt=None):
        if self._detect_enabled and self._stack is not None:
            self._detect_debounce.start()

    def _current_frame_idx(self) -> int:
        if self._viewer is None or self._stack is None:
            return 0
        try:
            return int(self._viewer.dims.current_step[0])
        except Exception:
            return 0

    def _run_detection(self):
        if not self._detect_enabled or self._stack is None or self._viewer is None:
            return
        idx = max(0, min(self._current_frame_idx(), self._stack.shape[0] - 1))
        # Locate on the PREPROCESSED frame so the mass scale here matches
        # what the pipeline produces during the real run.  The bandpass view
        # toggle is a display aid only and doesn't affect numbers.
        pp = self._ensure_preprocessed_stack()
        if pp is None:
            self._status.setText("Preprocessing failed — falling back to raw frame.")
            frame = self._stack[idx]
        else:
            frame = pp[idx]
        diameter = self._detect_params["diameter"]
        if diameter % 2 == 0:
            diameter += 1
        minmass = self._detect_params["minmass"]
        try:
            import trackpy as tp
            df = tp.locate(frame, diameter=diameter, minmass=minmass)
        except Exception as exc:
            self._status.setText(f"Detection preview failed: {exc}")
            return
        import numpy as _np
        if len(df):
            pts  = df[["y", "x"]].to_numpy()
            mass = df["mass"].to_numpy() if "mass" in df.columns else None
        else:
            pts  = _np.zeros((0, 2), dtype=float)
            mass = None
        self._last_mass = mass
        self._update_points_layer(pts, diameter, mass)
        # Mass-distribution summary helps the user pick a useful minmass.
        if mass is not None and len(mass):
            m_lo, m_med, m_hi = (float(_np.min(mass)),
                                 float(_np.median(mass)),
                                 float(_np.max(mass)))
            mass_summary = f"mass {m_lo:.0f} / med {m_med:.0f} / {m_hi:.0f}"
        else:
            mass_summary = "no spots"
        self._status.setText(
            f"Frame {idx + 1}/{self._stack.shape[0]} — "
            f"{len(df)} spots (d={diameter}, minmass={minmass:g}) — {mass_summary}")

    def _mass_to_rgba(self, mass):
        """Return (N, 4) RGBA in 0..1 using turbo on log(mass) — high
        contrast on a grey background at both ends of the scale."""
        import numpy as _np
        try:
            import matplotlib.cm as _cm
            import matplotlib.colors as _mc
            m = _np.asarray(mass, dtype=float)
            # log scale keeps dim spots distinguishable from bright ones
            logm = _np.log10(_np.clip(m, 1e-3, None))
            if logm.size == 0:
                return _np.zeros((0, 4))
            vmin = float(_np.min(logm))
            vmax = float(_np.max(logm))
            if vmax <= vmin + 1e-9:
                vmax = vmin + 1.0
            norm = _mc.Normalize(vmin=vmin, vmax=vmax)
            try:
                cmap = _cm.get_cmap("turbo")
            except Exception:
                cmap = _cm.viridis
            rgba = cmap(norm(logm))
            return _np.asarray(rgba, dtype=float)
        except Exception:
            n = len(mass) if mass is not None else 0
            return _np.tile([0.0, 1.0, 1.0, 1.0], (n, 1))

    def _update_points_layer(self, pts, diameter: int, mass=None):
        import numpy as _np
        size = max(4, int(diameter) + 6)
        # Build per-point colour array (turbo on log mass) for visibility.
        if mass is not None and len(mass) > 0:
            colours = self._mass_to_rgba(mass)
        else:
            colours = _np.tile([0.0, 1.0, 1.0, 1.0],
                               (len(pts), 1)) if len(pts) else None
        if self._points_layer is None:
            kwargs = dict(
                size=size,
                face_color="transparent",
                symbol="o",
                name="Detections",
                opacity=1.0,
            )
            try:
                self._points_layer = self._viewer.add_points(
                    pts,
                    border_color=(colours if colours is not None else "#00ffff"),
                    border_width=0.30,
                    **kwargs)
            except TypeError:
                # napari < 0.5 — edge_* names
                self._points_layer = self._viewer.add_points(
                    pts,
                    edge_color=(colours if colours is not None else "#00ffff"),
                    edge_width=0.30,
                    **kwargs)
            except Exception as exc:
                self._status.setText(f"Points layer failed: {exc}")
                self._points_layer = None
                return
            try:
                if self._shapes_layer is not None:
                    self._viewer.layers.selection.active = self._shapes_layer
            except Exception:
                pass
        else:
            try:
                self._points_layer.data = pts
                self._points_layer.size = size
                if colours is not None and len(colours):
                    try:
                        self._points_layer.border_color = colours
                    except Exception:
                        try: self._points_layer.edge_color = colours
                        except Exception: pass
            except Exception:
                pass

    # ── Bandpass-filtered view ───────────────────────────────────────────
    def _on_filtered_toggled(self, checked: bool):
        if self._viewer is None or self._image_layer is None or self._stack is None:
            return
        if checked:
            if self._stack_filtered is None:
                self._stack_filtered = self._compute_filtered_stack(self._stack)
            target = self._stack_filtered
        else:
            target = self._stack
        try:
            self._image_layer.data = target
            import numpy as _np
            sample = target[target.shape[0] // 2]
            lo, hi = _np.percentile(sample, [1.0, 99.5])
            if hi <= lo:
                hi = lo + 1.0
            self._image_layer.contrast_limits = (float(lo), float(hi))
        except Exception as exc:
            self._status.setText(f"Couldn't swap image: {exc}")

    def _ensure_preprocessed_stack(self):
        """Lazily build a pipeline-equivalent preprocessed stack so that
        `tp.locate` here sees the same intensities as the real run.  Mirrors
        sptpalm_analysis._preprocess_fast / _preprocess_rolling: background
        subtract → clip ≥0 → gaussian sigma=1 → per-frame normalise to [0,1].
        """
        if self._stack is None:
            return None
        bg_method = self._detect_params.get("bg_method", "uniform_filter")
        bg_radius = int(self._detect_params.get("bg_radius", 50)) or 50
        sig = (str(bg_method), bg_radius)
        if self._stack_preprocessed is not None and self._pp_signature == sig:
            return self._stack_preprocessed
        try:
            import numpy as _np
            from scipy.ndimage import uniform_filter, gaussian_filter
        except Exception:
            return None
        rolling_fn = None
        if bg_method == "rolling_ball":
            try:
                from skimage.restoration import rolling_ball as rolling_fn
            except Exception:
                rolling_fn = None  # fall back to uniform filter silently

        size = int(bg_radius * 2 + 1)
        out = _np.empty(self._stack.shape, dtype=_np.float32)
        for i in range(self._stack.shape[0]):
            f = self._stack[i].astype(_np.float32, copy=False)
            if rolling_fn is not None:
                try:
                    bg = rolling_fn(f, radius=bg_radius)
                except Exception:
                    bg = uniform_filter(f, size=size)
            else:
                bg = uniform_filter(f, size=size)
            corrected = _np.clip(f - bg, 0, None)
            smoothed  = gaussian_filter(corrected, sigma=1.0)
            mn = float(smoothed.min()); mx = float(smoothed.max())
            if mx > mn:
                smoothed = (smoothed - mn) / (mx - mn)
            out[i] = smoothed
        self._stack_preprocessed = out
        self._pp_signature = sig
        return self._stack_preprocessed

    # ── Auto / manual-threshold ROI overlay ──────────────────────────────
    def set_roi_mask_params(self, *, mode: str, auto_method: str,
                            threshold: float, mask_mode: str,
                            bg_sigma: float = 25.0):
        """Update + redraw the auto/manual-threshold ROI overlay layer.
        `mode` is "None", "Auto threshold", "Manual threshold" or
        "Manual polygon"; the overlay is drawn for the first two.
        `bg_sigma` (px) controls the DoG background-suppression scale —
        see _refresh_roi_mask_overlay for semantics."""
        self._roi_mask_params = {"mode": str(mode),
                                 "auto_method": str(auto_method).lower(),
                                 "threshold": float(threshold),
                                 "mask_mode": str(mask_mode).lower(),
                                 "bg_sigma": float(bg_sigma)}
        self._refresh_roi_mask_overlay()

    def _refresh_roi_mask_overlay(self):
        if self._viewer is None or self._stack is None:
            return
        mode = self._roi_mask_params.get("mode", "None")
        if mode not in ("Auto threshold", "Manual threshold"):
            self._remove_roi_mask_layer()
            return
        try:
            import numpy as _np
            # The actual mask pipeline lives in sptpalm_analysis.
            # build_roi_mask_advanced — same function the firefly_worker
            # calls during analysis — so what the user sees in this
            # preview is what gets applied to the localisations.
            from firefly.sptpalm_analysis import build_roi_mask_advanced
        except Exception:
            return
        # Build the projection that will be thresholded.  All four modes
        # output a float32 image; we normalise to [0,1] at the end so the
        # threshold slider's [0,1] semantics work the same way for all
        # of them.
        #
        #   mean  — mean intensity per pixel.  Cheap, but autofluorescence
        #           accumulates with frames the same way real signal does,
        #           so cell vs background contrast is poor on sptPALM.
        #   sum   — same shape as mean, kept for backward compatibility.
        #   max   — brightest value each pixel ever reached.  Strongly
        #           amplifies sparse-blink signal; this is what makes
        #           cells unmistakable in a sptPALM max projection.
        #   blink — per-pixel count of frames where the pixel was
        #           significantly above its temporal median (median + 3·MAD).
        #           Most discriminative mode for sptPALM: cells fire
        #           blinks repeatedly, autofluorescent background is
        #           steady so its blink-count is ~zero.
        # ── 1. Build the projection from the cached preview stack ──────
        mode_proj = self._roi_mask_params.get("mask_mode", "mean")
        stk = self._stack.astype(_np.float32, copy=False)
        if mode_proj == "max":
            proj = stk.max(axis=0)
        elif mode_proj == "blink":
            # Robust per-pixel baseline: median + 3*MAD across time.
            # MAD is cheap and noise-resistant vs std.  We use the
            # temporal axis only, so output shape stays (Y, X).  This
            # mode is preview-only — the streaming firefly_worker falls
            # back to Max because true per-pixel MAD needs a 2-pass.
            med = _np.median(stk, axis=0)
            mad = _np.median(_np.abs(stk - med[None]), axis=0) + 1e-6
            thresh_per_px = med + 3.0 * 1.4826 * mad   # MAD→σ for normal
            proj = (stk > thresh_per_px[None]).sum(axis=0).astype(_np.float32)
        elif mode_proj == "sum":
            proj = stk.sum(axis=0)
        else:  # "mean" (default)
            proj = stk.mean(axis=0)

        # ── 2. Threshold (manual or auto) ──────────────────────────────
        manual_thresh = None
        method = self._roi_mask_params.get("auto_method", "li")
        if mode == "Manual threshold":
            manual_thresh = float(self._roi_mask_params["threshold"])

        # ── 3. Delegate to the shared GUI/worker mask pipeline ─────────
        bg_sigma = float(self._roi_mask_params.get("bg_sigma", 25.0))
        mask, info = build_roi_mask_advanced(
            proj,
            threshold=manual_thresh,
            threshold_method=method,
            bg_sigma=bg_sigma,
            mode_hint=mode_proj if mode_proj in ("max", "blink", "mean", "sum")
                                else "max")
        self._draw_roi_mask_layer(mask, info["threshold"])

    @staticmethod
    def _auto_threshold(image_norm, method: str):
        try:
            from skimage.filters import (threshold_otsu, threshold_li,
                                         threshold_triangle)
        except Exception:
            return None
        method = (method or "li").lower()
        try:
            if method == "otsu":     return float(threshold_otsu(image_norm))
            if method == "li":       return float(threshold_li(image_norm))
            if method == "triangle": return float(threshold_triangle(image_norm))
            if method == "mean":     return float(image_norm.mean())
        except Exception:
            pass
        return None

    def _draw_roi_mask_layer(self, mask, threshold: float):
        import numpy as _np
        # Convert bool mask to (Y, X) uint8 so we can colour it via a
        # custom colormap with transparency at 0.
        layer_data = mask.astype(_np.uint8)
        if self._roi_mask_layer is None:
            try:
                # Render through a 2-stop colormap: 0 = transparent,
                # 1 = bright lime so the mask is unmistakable on grey.
                from napari.utils.colormaps import Colormap as _NCmap
                cmap = _NCmap([[0, 0, 0, 0], [0.20, 1.00, 0.30, 1.0]],
                              name="firefly_roi_mask")
                self._roi_mask_layer = self._viewer.add_image(
                    layer_data, name="ROI mask", colormap=cmap,
                    contrast_limits=(0, 1), opacity=0.35,
                    blending="translucent")
            except Exception:
                try:
                    self._roi_mask_layer = self._viewer.add_image(
                        layer_data, name="ROI mask", colormap="green",
                        contrast_limits=(0, 1), opacity=0.35,
                        blending="translucent")
                except Exception as exc:
                    self._status.setText(f"ROI mask layer failed: {exc}")
                    return
            # Re-select shapes layer so polygon drawing keeps working
            try:
                if self._shapes_layer is not None:
                    self._viewer.layers.selection.active = self._shapes_layer
            except Exception:
                pass
        else:
            try:
                self._roi_mask_layer.data = layer_data
            except Exception:
                pass
        # Refresh status with the threshold the user can see and tune
        try:
            n_in = int(_np.sum(mask))
            total = int(mask.size)
            pct = 100.0 * n_in / total if total else 0.0
            self._roi_mask_layer.metadata = {"threshold": threshold,
                                             "fraction": pct}
        except Exception:
            pass

    def _remove_roi_mask_layer(self):
        if self._roi_mask_layer is not None and self._viewer is not None:
            try:
                self._viewer.layers.remove(self._roi_mask_layer)
            except Exception:
                pass
        self._roi_mask_layer = None

    def _compute_filtered_stack(self, stack):
        """Bandpass-filter every frame using trackpy.bandpass (matches what
        tp.locate does internally), so the viewer shows what the detector
        actually sees."""
        import numpy as _np
        diameter = int(self._detect_params.get("diameter", 7)) or 7
        if diameter % 2 == 0:
            diameter += 1
        try:
            import trackpy as tp
            out = _np.empty_like(stack)
            for i in range(stack.shape[0]):
                out[i] = tp.bandpass(stack[i], lshort=1, llong=diameter)
            return out
        except Exception:
            # Fall back to a difference-of-gaussians if trackpy.bandpass is
            # missing or fails — same idea, slightly different kernel.
            try:
                from scipy.ndimage import gaussian_filter
                out = _np.empty_like(stack, dtype=_np.float32)
                short = 1.0
                long  = max(1.5, diameter / 2.0)
                for i in range(stack.shape[0]):
                    f = stack[i].astype(_np.float32)
                    out[i] = gaussian_filter(f, short) - gaussian_filter(f, long)
                return out
            except Exception:
                return stack

    def _remove_points_layer(self):
        if self._points_layer is not None and self._viewer is not None:
            try:
                self._viewer.layers.remove(self._points_layer)
            except Exception:
                pass
        self._points_layer = None


class _NoHScrollArea(QtWidgets.QScrollArea):
    """A vertical-only scroll area: the inner widget is clamped to the viewport
    width so the content can never scroll or drag sideways.

    Setting `setHorizontalScrollBarPolicy(AlwaysOff)` only HIDES the bar — a
    trackpad two-finger swipe still scrolls horizontally whenever the content's
    minimum width exceeds the viewport (e.g. the Compare sidebar's group cards).
    Clamping the inner widget's maximum width to the viewport removes the
    overflow entirely; children compress to fit instead."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def _clamp(self):
        w = self.widget()
        if w is not None:
            w.setMaximumWidth(self.viewport().width())

    def setWidget(self, w):
        super().setWidget(w)
        self._clamp()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._clamp()


class _FolderDropList(QtWidgets.QListWidget):
    """QListWidget that accepts dropped folders (Qt-native, no tkinterdnd2).

    Files dropped onto the widget are added by their full path; non-folders
    are ignored.  Duplicates are silently de-duped.  Behaviour matches the
    Tk app's drag-and-drop on the Compare-tab cards.
    """
    folders_dropped = QtCore.Signal(list)   # list[str] of newly-added folders

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAlternatingRowColors(True)
        # Keep the list inside the narrow sidebar: never grow a horizontal
        # scrollbar (which made the whole card draggable left/right), and elide
        # long folder basenames with "…" instead.  The full path is still in
        # each item's UserRole + tooltip.
        self.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setWordWrap(False)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                           QtWidgets.QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(0)
        # Subtle styling cue that this is a drop target
        self.setStyleSheet(
            "QListWidget { border: 1px dashed #777; border-radius: 4px; "
            "padding: 4px; }")

    # ── path storage: show the readable basename, keep the full path ──
    # Each row displays `basename` but stores the FULL path in UserRole (and
    # as its tooltip), so the narrow sidebar stays readable while every
    # consumer still gets the absolute path via `folder_paths()`.
    def folder_paths(self) -> "list[str]":
        out = []
        for i in range(self.count()):
            it = self.item(i)
            p = it.data(Qt.ItemDataRole.UserRole)
            out.append(str(p) if p is not None else it.text())
        return out

    def add_folder(self, path: str) -> bool:
        """Add one folder (basename shown, full path stored). Returns False if
        it was already present (de-duped on the full path)."""
        path = str(path)
        if path in self.folder_paths():
            return False
        name = os.path.basename(path.rstrip("/\\")) or path
        item = QtWidgets.QListWidgetItem(name)
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)
        self.addItem(item)
        return True

    # Qt drag-and-drop event handlers
    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e: QtGui.QDragMoveEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent):
        if not e.mimeData().hasUrls():
            return
        added: list[str] = []
        for url in e.mimeData().urls():
            path = url.toLocalFile()
            if not path or not os.path.isdir(path):
                continue
            if self.add_folder(path):
                added.append(path)
        if added:
            self.folders_dropped.emit(added)
        e.acceptProposedAction()


class _CompareGroupCard(QtWidgets.QGroupBox):
    """One row in the Compare tab — label, colour swatch, folder list,
    +/− buttons.  Emits `changed` whenever its contents change so the
    parent window can persist the state."""
    changed = QtCore.Signal()
    delete_requested = QtCore.Signal(object)   # self

    # Default colour palette — cycled through for new cards.  12 distinct
    # hues so every card up to COMPARE_MAX_GROUPS gets its own swatch before
    # the palette repeats.
    _DEFAULT_COLORS = ["#3b6ed8", "#f78166", "#56d364", "#d2a8ff",
                       "#ffa657", "#79c0ff", "#e3b341", "#ff7b72",
                       "#39c5cf", "#bc8cff", "#7ee787", "#ffa198"]

    def __init__(self, index: int, label: str = "", color: str = "",
                 timepoint: str = "", parent=None):
        super().__init__(parent)
        if not label:
            # Default labels are just "Group N" — the previous
            # "Pre"/"Post" defaults assumed a paired-condition
            # workflow that's only one of several use cases.
            # Users rename to whatever's meaningful for their study.
            label = f"Group {index + 1}"
        if not color:
            color = self._DEFAULT_COLORS[index % len(self._DEFAULT_COLORS)]
        self._color = color
        self.setTitle(f"Group {index + 1}")

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        # Top row: label edit + colour swatch + delete card.
        #
        # Earlier attempts used a styled QPushButton for the colour
        # swatch and pinned every widget to the same _ROW_H.  On macOS
        # that still left the swatch looking taller than the line edit
        # because Qt's macOS QPushButton renders a couple of pixels of
        # native chrome OUTSIDE its setFixedSize bounds (the rounded
        # button face + focus ring).  The reliable fix is a chromeless
        # widget — QToolButton with no border + a stylesheet-driven
        # background — sized to the line edit's actual rendered height
        # (~22 px on macOS dark mode).
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        row.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(QtWidgets.QLabel("Label"))

        self.e_label = QtWidgets.QLineEdit(label)
        self.e_label.textChanged.connect(lambda _: self.changed.emit())
        row.addWidget(self.e_label, 1)
        # Read the actual line-edit render height AFTER it has been
        # constructed but before it's laid out — gives us pixel-
        # perfect match across themes / DPI.
        _ROW_H = max(20, self.e_label.sizeHint().height() - 4)

        # ── Colour swatch — flat QToolButton, no native chrome ─────
        self.btn_color = QtWidgets.QToolButton()
        self.btn_color.setAutoRaise(True)
        self.btn_color.setFixedSize(_ROW_H + 8, _ROW_H)
        self._refresh_color_button()
        self.btn_color.clicked.connect(self._on_pick_color)
        self.btn_color.setToolTip("Pick a colour for this group's plots.")
        self.btn_color.setCursor(Qt.CursorShape.PointingHandCursor)
        row.addWidget(self.btn_color, 0, Qt.AlignmentFlag.AlignVCenter)

        # ── Delete button — big, obvious × ────────────────────────
        self.btn_delete = QtWidgets.QToolButton()
        self.btn_delete.setText("×")
        self.btn_delete.setAutoRaise(True)
        self.btn_delete.setFixedSize(_ROW_H + 4, _ROW_H)
        _del_font = self.btn_delete.font()
        # 24 pt minimum so the × is unmistakably a click target.
        _del_font.setPointSize(max(_del_font.pointSize() + 12, 24))
        _del_font.setBold(True)
        self.btn_delete.setFont(_del_font)
        self.btn_delete.setToolTip("Remove this group.")
        self.btn_delete.setCursor(Qt.CursorShape.PointingHandCursor)
        # Subtle danger-tinted hover so the button telegraphs that it
        # is destructive without screaming about it at rest.
        self.btn_delete.setStyleSheet(
            "QToolButton {"
            "  padding: 0px; margin: 0px; border: none;"
            f"  color: {_THEME['TXT_MUTED']};"
            "}"
            "QToolButton:hover {"
            f"  color: {_THEME['DANGER']};"
            "  background: rgba(255, 80, 80, 0.15);"
            "  border-radius: 4px;"
            "}"
        )
        self.btn_delete.clicked.connect(lambda: self.delete_requested.emit(self))
        row.addWidget(self.btn_delete, 0, Qt.AlignmentFlag.AlignVCenter)
        v.addLayout(row)

        # Optional time-point field.  When ≥2 cards carry a time point, the
        # Compare tab switches to a paired group × time-point two-way mixed
        # ANOVA (cells matched across time points by folder name).
        tp_row = QtWidgets.QHBoxLayout()
        tp_row.setSpacing(6)
        tp_row.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        tp_row.addWidget(QtWidgets.QLabel("Time point"))
        self.e_timepoint = QtWidgets.QLineEdit(timepoint)
        self.e_timepoint.setPlaceholderText("optional — e.g. Pre / Post / T0")
        self.e_timepoint.setToolTip(
            "Optional.  Set a time point (e.g. Pre / Post) on two or more cards "
            "to run a paired group × time-point two-way mixed ANOVA.\n\n"
            "Cells are paired across time points by folder name — the time-point "
            "word is stripped from each folder's stem to identify the same cell "
            "(so '…_DMSO_D1_Pre' and '…_DMSO_D1_Post' are matched).\n\n"
            "Leave blank on every card for a normal group comparison.")
        self.e_timepoint.textChanged.connect(lambda _: self.changed.emit())
        tp_row.addWidget(self.e_timepoint, 1)
        v.addLayout(tp_row)

        # Folder list (drop target)
        self.lst_folders = _FolderDropList()
        self.lst_folders.setMinimumHeight(80)
        self.lst_folders.folders_dropped.connect(lambda _: self.changed.emit())
        v.addWidget(self.lst_folders, 1)

        # Add / Remove buttons — short labels so the row fits the narrow
        # sidebar without forcing the card wider than the viewport.
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(4)
        self.btn_add = QtWidgets.QPushButton("+ Add")
        self.btn_add.setToolTip("Add a folder to this group.")
        self.btn_add.clicked.connect(self._on_add_folder)
        self.btn_remove = QtWidgets.QPushButton("Remove")
        self.btn_remove.setToolTip("Remove the selected folder(s).")
        self.btn_remove.clicked.connect(self._on_remove_selected)
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_clear.setToolTip("Remove all folders from this group.")
        self.btn_clear.clicked.connect(self._on_clear)
        for _b in (self.btn_add, self.btn_remove, self.btn_clear):
            _b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum,
                             QtWidgets.QSizePolicy.Policy.Fixed)
            btn_row.addWidget(_b)
        btn_row.addStretch(1)
        self.lbl_count = QtWidgets.QLabel("0 folders")
        btn_row.addWidget(self.lbl_count)
        v.addLayout(btn_row)

        self.lst_folders.itemSelectionChanged.connect(lambda: self.changed.emit())
        self._refresh_count()

    # ── Helpers ────────────────────────────────────────────────────────────
    @property
    def color(self) -> str:
        return self._color

    def _refresh_color_button(self):
        # Flat-filled QToolButton.  The `:hover` state nudges the
        # border slightly lighter so it's still obvious the chip is
        # clickable, but no auto-chrome / 3-D outline from the native
        # style — that was making the swatch render taller than the
        # adjacent line edit on macOS.
        self.btn_color.setStyleSheet(
            "QToolButton {"
            f"  background-color: {self._color};"
            "  border: 1px solid #555;"
            "  border-radius: 3px;"
            "  padding: 0px;"
            "  margin: 0px;"
            "}"
            "QToolButton:hover {"
            "  border: 1px solid #aaa;"
            "}"
        )

    def _refresh_count(self):
        n = self.lst_folders.count()
        self.lbl_count.setText(f"{n} folder{'s' if n != 1 else ''}")

    def _on_pick_color(self):
        col = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(self._color), self, "Choose group colour")
        if col.isValid():
            self._color = col.name()
            self._refresh_color_button()
            self.changed.emit()

    def _on_add_folder(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Add folder to this group", os.path.expanduser("~"))
        if not path:
            return
        if self.lst_folders.add_folder(path):
            self._refresh_count()
            self.changed.emit()

    def _on_remove_selected(self):
        for it in reversed(self.lst_folders.selectedItems()):
            self.lst_folders.takeItem(self.lst_folders.row(it))
        self._refresh_count()
        self.changed.emit()

    def _on_clear(self):
        self.lst_folders.clear()
        self._refresh_count()
        self.changed.emit()

    def get_state(self) -> dict:
        """Return the group as the dict shape `compare_groups` expects.
        Folders are the FULL absolute paths (the list shows basenames but
        stores the full path in each item's UserRole)."""
        return {"label":  self.e_label.text().strip() or "Group",
                "color":  self._color,
                "timepoint": self.e_timepoint.text().strip(),
                "folders": self.lst_folders.folder_paths()}

    def set_state(self, label: str, color: str, folders: list,
                  timepoint: str = ""):
        self.e_label.setText(label)
        if color:
            self._color = color
            self._refresh_color_button()
        self.e_timepoint.setText(timepoint or "")
        self.lst_folders.clear()
        for f in folders:
            self.lst_folders.add_folder(str(f))
        self._refresh_count()


class _PreferencesDialog(QtWidgets.QDialog):
    """FIREFLY application-wide settings.

    Sections (left rail):
      • Appearance       — app theme
      • Figure defaults  — re-parents the existing figures-widget so all
                           per-run figure-export knobs live behind one
                           preferences surface instead of as a top-level tab.

    Opened from the cogwheel in the header bar (and ⌘, on macOS).
    Settings persist via QSettings — closing the dialog auto-saves.
    """

    # Display-name ↔ stored-value for the Compare-tab LogD distribution style.
    _LOGD_STYLE_MAP = {
        "Faceted (per-replicate)": "faceted",
        "Ridgeline":               "ridgeline",
        "Overlaid KDEs":           "overlaid",
        "Violins + points":        "violin",
    }
    _LOGD_STYLE_DISP = {v: k for k, v in _LOGD_STYLE_MAP.items()}

    def __init__(self, parent: "MainWindow"):
        super().__init__(parent)
        self.setWindowTitle("FIREFLY Preferences")
        self.setModal(True)
        self.resize(960, 640)
        self._main = parent

        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        # Left rail — section list
        self._rail = QtWidgets.QListWidget()
        self._rail.setObjectName("pref_rail")
        self._rail.setFixedWidth(180)
        self._rail.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._rail.setSpacing(2)
        f = self._rail.font(); f.setPointSize(13); self._rail.setFont(f)
        h.addWidget(self._rail)

        # Right side — stacked pages
        self._pages = QtWidgets.QStackedWidget()
        h.addWidget(self._pages, stretch=1)

        # ── Page: Appearance ────────────────────────────────────────────
        appearance_page = QtWidgets.QWidget()
        ap = QtWidgets.QVBoxLayout(appearance_page)
        ap.setContentsMargins(24, 24, 24, 24); ap.setSpacing(14)

        ap.addWidget(self._heading("Appearance"))

        # Theme combo
        theme_row = QtWidgets.QFormLayout()
        theme_row.setHorizontalSpacing(12); theme_row.setVerticalSpacing(8)
        self.c_app_theme = _QuietComboBox()
        self.c_app_theme.addItems(["Dark", "AMOLED", "Light"])
        self.c_app_theme.setMaximumWidth(180)
        try:
            _saved = QtCore.QSettings("FIREFLY", "sptPALM") \
                .value("ui/app_theme", "Dark") or "Dark"
            if str(_saved) in ("Dark", "AMOLED", "Light"):
                self.c_app_theme.setCurrentText(str(_saved))
        except Exception:
            pass
        self.c_app_theme.setToolTip(
            "Colour scheme for the FIREFLY GUI itself.\n"
            "• Dark   — default GitHub-dark\n"
            "• AMOLED — pure-black backgrounds for OLED displays\n"
            "• Light  — GitHub-light for daytime use\n\n"
            "Switching takes effect after restarting the app.")
        self.c_app_theme.currentTextChanged.connect(
            self._on_theme_changed)
        theme_row.addRow("App theme:", self.c_app_theme)

        ap.addLayout(theme_row)
        ap.addWidget(self._restart_hint(
            "App-theme changes take effect after restarting FIREFLY."))

        # Reduce motion — disable UI animations (instant transitions).
        self.c_reduce_motion = QtWidgets.QCheckBox(
            "Reduce motion (disable UI animations)")
        try:
            rm = QtCore.QSettings("jacoblevers", "FIREFLY").value(
                "ui/reduce_motion", False)
            if isinstance(rm, str):
                rm = rm.strip().lower() in ("1", "true", "yes", "on")
            self.c_reduce_motion.setChecked(bool(rm))
        except Exception:
            pass
        self.c_reduce_motion.setToolTip(
            "Turn off the small fade / expand animations for instant, static "
            "transitions. Takes effect immediately.")
        self.c_reduce_motion.toggled.connect(self._on_reduce_motion_toggled)
        ap.addWidget(self.c_reduce_motion)

        ap.addStretch(1)

        self._pages.addWidget(appearance_page)
        self._add_rail_entry("Appearance")

        # ── Page: Figure defaults (re-parent the figures widget) ────────
        # Scrollable so the LogD style picker + preview + the (large) figures
        # widget never overflow the dialog.
        fig_page = QtWidgets.QScrollArea()
        fig_page.setWidgetResizable(True)
        fig_page.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        _fig_inner = QtWidgets.QWidget()
        fp = QtWidgets.QVBoxLayout(_fig_inner)
        fp.setContentsMargins(0, 0, 0, 0); fp.setSpacing(0)

        # Compare-tab LogD distribution style — lives at the top of the
        # Figure-defaults page (it's a figure-look choice).
        _logd_box = QtWidgets.QWidget()
        _logd_form = QtWidgets.QFormLayout(_logd_box)
        _logd_form.setContentsMargins(24, 18, 24, 4)
        _logd_form.setHorizontalSpacing(12); _logd_form.setVerticalSpacing(6)
        self.c_logd_style = _QuietComboBox()
        self.c_logd_style.addItems(list(self._LOGD_STYLE_MAP.keys()))
        self.c_logd_style.setMaximumWidth(220)
        try:
            _cur = str(QtCore.QSettings("jacoblevers", "FIREFLY").value(
                "figures/logd_style", "overlaid") or "overlaid")
            self.c_logd_style.setCurrentText(
                self._LOGD_STYLE_DISP.get(_cur, "Overlaid KDEs"))
        except Exception:
            pass
        self.c_logd_style.setToolTip(
            "How the Compare tab's LogD-distribution panel is drawn:\n"
            "• Faceted — one panel per group; PRE/POST overlaid + per-cell "
            "median dots\n"
            "• Ridgeline — classic stacked filled KDEs\n"
            "• Overlaid KDEs — every group on one axes\n"
            "• Violins + points — per-group violins with per-cell medians\n"
            "Applies to the next comparison you run.")
        self.c_logd_style.currentTextChanged.connect(self._on_logd_style_changed)
        _logd_form.addRow("LogD graph style:", self.c_logd_style)
        # Live preview of the chosen style, plus a "best for" description.
        self._logd_preview_fig = Figure(figsize=(3.8, 2.5))
        self._logd_preview_canvas = FigureCanvas(self._logd_preview_fig)
        self._logd_preview_canvas.setMinimumHeight(200)
        self._logd_preview_canvas.setMaximumHeight(260)
        _logd_form.addRow(self._logd_preview_canvas)
        self._logd_style_desc = QtWidgets.QLabel("")
        self._logd_style_desc.setWordWrap(True)
        self._logd_style_desc.setMaximumWidth(560)
        self._logd_style_desc.setStyleSheet(f"color: {_THEME['TXT_MUTED']};")
        _logd_form.addRow(self._logd_style_desc)
        _logd_form.addRow(self._restart_hint(
            "Applies to the next comparison you run."))
        fp.addWidget(_logd_box)

        self._fig_widget = parent._figures_widget
        # Re-parent into this dialog — but we restore the parent back to
        # MainWindow in `done()` so the widget survives multiple open
        # cycles instead of being destroyed with the dialog.
        self._fig_widget.setParent(_fig_inner)
        fp.addWidget(self._fig_widget)
        # IMPORTANT: setParent() implicitly hides the widget, and we
        # also hide it explicitly in done().  Show it on each open so
        # the second-and-later openings of Preferences aren't blank.
        self._fig_widget.show()
        fig_page.setWidget(_fig_inner)
        self._pages.addWidget(fig_page)
        self._add_rail_entry("Figure defaults")
        self._refresh_logd_preview()

        # ── Page: GPU acceleration (Windows only) ───────────────────────
        # CUDA install/uninstall/relocate lives here now (it used to be a
        # single toggle button buried in the Performance section).
        import sys as _sys
        if _sys.platform == "win32":
            self._build_gpu_page()

        # ── Page: Updates ───────────────────────────────────────────────
        self._build_updates_page()

        # Connect rail → stack
        self._rail.currentRowChanged.connect(self._pages.setCurrentIndex)
        self._rail.setCurrentRow(0)

    # ── GPU acceleration page ────────────────────────────────────────────
    def _build_gpu_page(self) -> None:
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(24, 24, 24, 24); v.setSpacing(14)
        v.addWidget(self._heading("GPU acceleration"))

        # Cache the (slow) nvidia-smi probe once per dialog.
        self._gpu_name = None
        try:
            from firefly import cuda_installer as _cu
            self._gpu_name = _cu.detect_nvidia_gpu()
        except Exception:
            pass

        self._gpu_status = QtWidgets.QLabel()
        self._gpu_status.setWordWrap(True)
        self._gpu_status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self._gpu_status)

        btn_row = QtWidgets.QHBoxLayout(); btn_row.setSpacing(8)
        self._gpu_install_btn = QtWidgets.QPushButton("Install")
        self._gpu_install_btn.clicked.connect(self._on_gpu_install)
        self._gpu_uninstall_btn = QtWidgets.QPushButton("Uninstall")
        self._gpu_uninstall_btn.clicked.connect(self._on_gpu_uninstall)
        self._gpu_move_btn = QtWidgets.QPushButton("Change location…")
        self._gpu_move_btn.clicked.connect(self._on_gpu_change_location)
        for b in (self._gpu_install_btn, self._gpu_uninstall_btn,
                  self._gpu_move_btn):
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        v.addLayout(btn_row)

        v.addWidget(self._restart_hint(
            "Installing or removing CUDA takes effect after restarting FIREFLY. "
            "The CUDA build of PyTorch is ~2.5 GB."))
        v.addStretch(1)

        self._pages.addWidget(page)
        self._add_rail_entry("GPU acceleration")
        self._refresh_gpu_status()

    def _refresh_gpu_status(self) -> None:
        try:
            from firefly import cuda_installer as _cu
        except Exception:
            self._gpu_status.setText("CUDA installer module unavailable.")
            for b in (self._gpu_install_btn, self._gpu_uninstall_btn,
                      self._gpu_move_btn):
                b.setEnabled(False)
            return
        installed = _cu.is_installed()
        gpu = self._gpu_name or "no NVIDIA GPU detected"
        lines = [f"GPU: {gpu}"]
        if installed:
            ver = _cu.installed_torch_version() or "unknown version"
            lines.append(f"Status: installed — PyTorch {ver}")
            lines.append(f"Location: {_cu.sidecar_dir()}")
        else:
            lines.append("Status: not installed (FIREFLY is using the bundled "
                         "CPU build)")
            lines.append(f"Location: {_cu.sidecar_base()}")
        self._gpu_status.setText("\n".join(lines))
        self._gpu_install_btn.setText("Reinstall" if installed else "Install")
        self._gpu_uninstall_btn.setEnabled(installed)
        # Relocation is allowed even before install (records where future
        # installs land); only blocked if the installer module is missing.
        self._gpu_move_btn.setEnabled(True)

    def _on_gpu_install(self) -> None:
        # Close Preferences first so the installer's own progress dialog isn't
        # blocked behind this application-modal window, then kick it off.
        self.accept()
        QtCore.QTimer.singleShot(0, self._main._run_cuda_install)

    def _on_gpu_uninstall(self) -> None:
        from firefly import cuda_installer as _cu
        reply = QtWidgets.QMessageBox.question(
            self, "Remove CUDA acceleration?",
            f"Remove the CUDA build of PyTorch at\n{_cu.sidecar_dir()}\n\n"
            "FIREFLY will fall back to the bundled CPU build. You can "
            "reinstall any time.",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No)
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        try:
            _cu.uninstall()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Removal failed", str(exc))
            return
        QtWidgets.QMessageBox.information(
            self, "CUDA removed",
            "CUDA acceleration removed. Restart FIREFLY to drop back to the "
            "bundled CPU build.")
        self._refresh_gpu_status()
        if hasattr(self._main, "_refresh_cuda_perf_ui"):
            self._main._refresh_cuda_perf_ui()

    def _on_gpu_change_location(self) -> None:
        from firefly import cuda_installer as _cu
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose a folder for the CUDA install",
            _cu.sidecar_base())
        if not folder:
            return
        moving = _cu.is_installed()
        verb = "Move the existing install" if moving else "Use"
        reply = QtWidgets.QMessageBox.question(
            self, "Change CUDA location?",
            f"{verb} CUDA folder under:\n{folder}\\torch-cuda\n\n"
            + ("This moves ~2.5 GB and may take a moment on a different drive."
               if moving else "Future installs will go here."),
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes)
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(Qt.CursorShape.WaitCursor))
        try:
            new_base = _cu.move_install(folder)
        except Exception as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, "Couldn't change location", str(exc))
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        QtWidgets.QMessageBox.information(
            self, "Location updated", f"CUDA location is now:\n{new_base}")
        self._refresh_gpu_status()

    # ── Updates page ──────────────────────────────────────────────────────
    def _build_updates_page(self) -> None:
        from firefly import updater
        page = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(page)
        v.setContentsMargins(24, 24, 24, 24); v.setSpacing(14)
        v.addWidget(self._heading("Updates"))

        cur = "unknown"
        try:
            from firefly import sptpalm_analysis as _sa
            cur = str(getattr(_sa, "__version__", "unknown"))
        except Exception:
            pass
        ver_lbl = QtWidgets.QLabel(f"Current version: FIREFLY {cur}")
        ver_lbl.setStyleSheet(f"color: {_THEME['TXT']};")
        v.addWidget(ver_lbl)

        s = QtCore.QSettings("jacoblevers", "FIREFLY")
        auto = s.value("updates/auto_check", True)
        if isinstance(auto, str):            # some backends stringify bools
            auto = auto.lower() not in ("false", "0", "no", "")
        self.chk_auto_update = QtWidgets.QCheckBox(
            "Automatically check for updates on startup")
        self.chk_auto_update.setChecked(bool(auto))
        self.chk_auto_update.toggled.connect(self._on_auto_update_toggled)
        v.addWidget(self.chk_auto_update)

        btn = QtWidgets.QPushButton("Check for updates now")
        btn.clicked.connect(self._on_check_updates_now)
        row = QtWidgets.QHBoxLayout(); row.addWidget(btn); row.addStretch(1)
        v.addLayout(row)

        if not updater.is_frozen():
            v.addWidget(self._restart_hint(
                "Automatic updates apply to the packaged FIREFLY app only. "
                "This is a from-source install — update with 'git pull'."))

        v.addStretch(1)
        self._pages.addWidget(page)
        self._add_rail_entry("Updates")

    def _on_auto_update_toggled(self, on: bool) -> None:
        try:
            QtCore.QSettings("jacoblevers", "FIREFLY").setValue(
                "updates/auto_check", bool(on))
        except Exception:
            pass

    def _on_check_updates_now(self) -> None:
        # Close Preferences first so the (modal) update dialog is visible.
        main = self._main
        self.accept()
        try:
            main._force_check_for_updates()
        except Exception:
            pass

    def _heading(self, txt: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(txt)
        f = lbl.font(); f.setBold(True); f.setPointSize(16); lbl.setFont(f)
        lbl.setStyleSheet(f"color: {_THEME['TXT']}; padding-bottom: 6px;")
        return lbl

    def _restart_hint(self, msg: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(msg)
        lbl.setStyleSheet(f"color: {_THEME['TXT_MUTED']}; font-style: italic;")
        lbl.setWordWrap(True)
        return lbl

    def _on_reduce_motion_toggled(self, on: bool) -> None:
        try:
            QtCore.QSettings("jacoblevers", "FIREFLY").setValue(
                "ui/reduce_motion", bool(on))
        except Exception:
            pass

    def _add_rail_entry(self, name: str) -> None:
        item = QtWidgets.QListWidgetItem(name)
        item.setSizeHint(QtCore.QSize(0, 36))
        self._rail.addItem(item)

    def _on_theme_changed(self, name: str) -> None:
        try:
            QtCore.QSettings("FIREFLY", "sptPALM").setValue(
                "ui/app_theme", str(name))
            if str(name) != _ACTIVE_THEME_NAME:
                QtWidgets.QMessageBox.information(
                    self, "Restart to apply theme",
                    f"App theme set to {name}.  Restart FIREFLY to see "
                    f"the change take effect.")
        except Exception:
            pass

    def _on_logd_style_changed(self, name: str) -> None:
        """Persist the chosen LogD-distribution style to the main settings
        store (read by _start_compare_run when launching a comparison)."""
        try:
            val = self._LOGD_STYLE_MAP.get(name, "overlaid")
            store = getattr(self._main, "_settings", None)
            if store is not None:
                store.setValue("figures/logd_style", val)
            else:
                QtCore.QSettings("jacoblevers", "FIREFLY").setValue(
                    "figures/logd_style", val)
        except Exception:
            pass
        self._refresh_logd_preview()

    def _refresh_logd_preview(self) -> None:
        """Re-render the small example figure for the currently-selected LogD
        style and update the 'best for' description."""
        try:
            from firefly.analysis.fa_compare import (render_logd_preview,
                                                     LOGD_STYLE_DESCRIPTIONS)
            val = self._LOGD_STYLE_MAP.get(
                self.c_logd_style.currentText(), "overlaid")
            render_logd_preview(self._logd_preview_fig, val, _ACTIVE_THEME_NAME)
            self._logd_preview_canvas.draw_idle()
            self._logd_style_desc.setText(LOGD_STYLE_DESCRIPTIONS.get(val, ""))
        except Exception:
            pass

    def done(self, code: int) -> None:
        """Detach the borrowed figures-widget before destruction so it
        survives to be hosted by the next Preferences dialog opening.

        Without this, the figures widget (and every persisted setting
        widget inside it: c_fig_theme, c_fig_proj_cmap, panel checkboxes,
        the preview labels…) would be deleted along with this dialog
        and the next open would crash on access.
        """
        try:
            if getattr(self, "_fig_widget", None) is not None:
                self._fig_widget.setParent(self._main)
                self._fig_widget.hide()
        except Exception:
            pass
        super().done(code)
