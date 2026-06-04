"""Icon factories, napari-chrome helpers, the motion colormap and
small GUI utilities (open-folder, Qt message handler).

Extracted from app_qt.py (#7); re-exported there for compatibility.
"""
from __future__ import annotations

import os
import subprocess
import sys
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from firefly import crash_reporter
from firefly.analysis.fa_constants import MOTION_CLASS_COLORS, MOTION_CLASS_ORDER
from firefly.ui.ui_theme import _THEME


_NAPARI_WELCOME_PHRASES = (
    "Drag image",
    "open image",
    "key bindings",
    "menu shortcuts",
    "Use the menu",
)


# Canonical motion-class colours/order live in fa_constants so the napari
# overlay, the single-run figure and the comparison figure can never drift
# apart.  (This palette is the reference scheme the others were standardised on.)
_MOTION_PALETTE = dict(MOTION_CLASS_COLORS)


_MOTION_ORDER = list(MOTION_CLASS_ORDER) + ["Unknown"]


_MOTION_CMAP_NAME = "firefly_motion"


def _register_motion_colormap() -> str:
    """Register a discrete 5-stop step colormap matching `_MOTION_PALETTE`
    in napari's global colormap registry and return its name.

    Idempotent — re-calling is cheap (dict overwrite).  We return the
    NAME (not the Colormap instance) because napari's Tracks layer
    internally hashes its `colormap` argument; passing a Colormap
    instance raises `TypeError: unhashable type: 'Colormap'` and
    aborts the layer build (the "filter error: unhashable type:
    'Colormap'" the user hit).  Passing a registered name dodges
    that code path entirely.
    """
    try:
        from napari.utils.colormaps import Colormap, AVAILABLE_COLORMAPS
    except Exception:
        return "turbo"   # graceful fallback for very old napari
    rgba = [QtGui.QColor(_MOTION_PALETTE[k]).getRgbF() for k in _MOTION_ORDER]
    # napari step colormaps need N+1 control points for N colours
    # (each pair defines a half-open interval).  Five 0.2-wide cells
    # over [0, 1] ensure motion_int values 0..4 (normalised to
    # 0.0, 0.25, 0.5, 0.75, 1.0) each land in their own bin.
    cmap = Colormap(colors=rgba,
                    controls=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                    interpolation="zero",
                    name=_MOTION_CMAP_NAME)
    AVAILABLE_COLORMAPS[_MOTION_CMAP_NAME] = cmap
    return _MOTION_CMAP_NAME


def _make_cogwheel_icon(*, color: "QtGui.QColor | None" = None,
                        px: int = 24) -> "QtGui.QIcon":
    """Procedurally draw a settings cogwheel as a QIcon.

    Builds the gear as a single QPainterPath:
      • Outer silhouette = 8 teeth, alternating root/tip vertices stepped
        evenly around the circle so the teeth join the body cleanly with
        no gap.
      • A central round hole punched out via the path's even-odd fill rule.

    Drawn to a 2×-DPR pixmap so the gear stays crisp on retina screens.
    """
    import math
    if color is None:
        color = QtGui.QColor(_THEME["TXT_MUTED"])
    pm = QtGui.QPixmap(px * 2, px * 2)
    pm.setDevicePixelRatio(2.0)
    pm.fill(Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    p.setBrush(color)
    p.setPen(Qt.PenStyle.NoPen)

    cx = cy = px / 2.0
    outer_r = px * 0.46          # tooth tip
    root_r  = px * 0.33          # tooth root / gear-body outer edge
    hole_r  = px * 0.13          # central hole

    n_teeth = 8
    # Each tooth occupies 360/n_teeth degrees.  Within that, the tip is
    # the centre 50% of the angular span; the two flanks transition out
    # to the root on either side; the gap between teeth sits at the
    # remaining root span.
    span = (2 * math.pi) / n_teeth          # angle per tooth+gap
    tip_half = span * 0.22                  # half-width of tooth tip
    root_half = span * 0.50                 # half-span at the root

    path = QtGui.QPainterPath()
    first = True
    for i in range(n_teeth):
        a = -math.pi / 2 + i * span         # centre angle (-pi/2 = top)
        # 4 control angles per tooth — root-left, tip-left, tip-right, root-right
        pts = [
            (a - root_half, root_r),
            (a - tip_half,  outer_r),
            (a + tip_half,  outer_r),
            (a + root_half, root_r),
        ]
        for ang, r in pts:
            pt = QtCore.QPointF(cx + r * math.cos(ang),
                                cy + r * math.sin(ang))
            if first:
                path.moveTo(pt); first = False
            else:
                path.lineTo(pt)
    path.closeSubpath()

    # Central hole — added as a second subpath; with the default
    # OddEvenFill rule that punches it out of the gear.
    path.setFillRule(Qt.FillRule.OddEvenFill)
    path.addEllipse(QtCore.QPointF(cx, cy), hole_r, hole_r)

    p.drawPath(path)
    p.end()
    return QtGui.QIcon(pm)


def _make_close_x_icon(*, color: "QtGui.QColor | None" = None,
                       px: int = 22) -> "QtGui.QIcon":
    """Procedurally draw a thin close-X as a QIcon.

    Used by the header Quit button.  Qt's `SP_DialogCloseButton` renders
    as a platform-specific icon that doesn't blend with FIREFLY's
    chrome on macOS (it's a thick circle-X).  This version matches the
    cogwheel — same line weight, same colour, centred.
    """
    if color is None:
        color = QtGui.QColor(_THEME["TXT_MUTED"])
    pm = QtGui.QPixmap(px * 2, px * 2)
    pm.setDevicePixelRatio(2.0)
    pm.fill(Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    pen = QtGui.QPen(color)
    pen.setWidthF(px * 0.13)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    # Diagonals fit within a px×px logical area — we drew to a 2× pixmap
    # for retina sharpness, but devicePixelRatio=2 means coordinates are
    # still in logical (`px`-sized) space.
    pad = px * 0.28
    p.drawLine(QtCore.QPointF(pad, pad),
               QtCore.QPointF(px - pad, px - pad))
    p.drawLine(QtCore.QPointF(px - pad, pad),
               QtCore.QPointF(pad, px - pad))
    p.end()
    return QtGui.QIcon(pm)


def _make_napari_container_layout_opaque(container: QtWidgets.QWidget) -> None:
    """Seal a FIREFLY-owned QWidget that hosts an embedded napari Qt
    window so napari's internal size hints can never propagate up
    into FIREFLY's MainWindow layout (which used to grow the whole
    window every time a file was loaded).

    Mechanism: `QSizePolicy.Ignored` on both axes + `setMinimumSize(0, 0)`
    on the OUTER container only.  Qt's layout solver then doesn't
    consult the container's `sizeHint()` when deciding how much room
    to allocate — it just gives the container whatever the parent's
    stretch factor says.  Napari fills the container.  Napari's own
    internal layout (dim slider, layer panel, canvas) is **completely
    untouched** — we never modify any napari descendant, so the
    "dim scrubber vanishes" regression from earlier recursive
    sizeHint hacks can't happen.

    Idempotent — safe to call repeatedly.
    """
    if container is None:
        return
    try:
        container.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Ignored)
        container.setMinimumSize(0, 0)
    except Exception:
        pass


def _hide_napari_chrome(viewer) -> None:
    """Hide napari's bottom-of-canvas viewer-button row (ndisplay / grid /
    home / console / etc.), the new-layer/delete button row under the
    layer list, and the empty-canvas 'Drag image(s) here…' welcome text.

    The welcome text is stripped by emptying the QLabel widgets that
    contain it (rather than hiding the welcome widget itself, which on
    some napari versions also hides the canvas).

    Defensive against napari version changes — attribute names shift
    between minor releases; every access is guarded."""
    try:
        qtv = (getattr(viewer.window, "qt_viewer", None)
               or getattr(viewer.window, "_qt_viewer", None))
    except Exception:
        qtv = None
    if qtv is None:
        return

    # Bottom-of-canvas button strip (ndisplay / grid / transpose / home / console)
    for attr in ("viewerButtons", "_viewer_buttons", "viewer_buttons"):
        w = getattr(qtv, attr, None)
        if w is not None and hasattr(w, "hide"):
            try:    w.hide()
            except Exception: pass

    # New-layer / delete buttons under the layer list
    for attr in ("layerButtons", "_layer_buttons", "layer_buttons"):
        w = getattr(qtv, attr, None)
        if w is not None and hasattr(w, "hide"):
            try:    w.hide()
            except Exception: pass

    # Welcome-overlay text — walk every QLabel under the qt_viewer and
    # blank out any that mention the welcome phrases.  Hiding the
    # parent welcome widget itself takes the canvas with it on some
    # napari versions, so we hide JUST the text content.
    try:
        for lbl in qtv.findChildren(QtWidgets.QLabel):
            try:
                txt = lbl.text() or ""
            except Exception:
                continue
            if any(p in txt for p in _NAPARI_WELCOME_PHRASES):
                try:    lbl.setText("")
                except Exception: pass
    except Exception:
        pass

    # Strip napari's menubar — on macOS its menus (File / View / Plugins /
    # Window / Help) get merged into the system menu bar and ⌘, opens
    # napari's Preferences dialog from "Python → Preferences".  None of
    # those belong to FIREFLY; clear them so the only menus the user sees
    # are the ones the host app actually owns.  Also disable any QActions
    # napari attached to the window — clearing the menubar removes them
    # from the menu, but their global shortcuts (⌘,, ⌘W, ⌘?, etc.) keep
    # firing until the actions themselves are disabled.
    try:
        qt_window = getattr(viewer.window, "_qt_window", None)
        mb = qt_window.menuBar() if qt_window is not None else None
        if mb is not None:
            # NOTE: do NOT call mb.clear() — on macOS the native menu bar
            # is shared with the application, and clearing it nukes the
            # standard QActions including the one ⌘Q routes through.
            # Detaching from the native menu + hiding the widget is
            # enough to keep napari's menus out of sight without taking
            # the host's Quit action down with them.
            try:    mb.setNativeMenuBar(False)
            except Exception: pass
            try:    mb.hide()
            except Exception: pass
        # Disable every QAction owned by the napari window so its global
        # shortcuts stop responding.  We don't delete them — that can
        # crash Qt mid-event — just set them disabled + remove their
        # shortcut binding.
        if qt_window is not None:
            for act in qt_window.findChildren(QtGui.QAction):
                try:    act.setEnabled(False)
                except Exception: pass
                try:    act.setShortcut(QtGui.QKeySequence())
                except Exception: pass
                try:    act.setShortcuts([])
                except Exception: pass
            # QShortcut objects are SEPARATE from QAction — napari uses
            # them for ⌘,, ⌘?, ⌘Y, etc.  Disable + strip them too.
            for sc in qt_window.findChildren(QtGui.QShortcut):
                try:    sc.setEnabled(False)
                except Exception: pass
                try:    sc.setKey(QtGui.QKeySequence())
                except Exception: pass
    except Exception:
        pass

    # Trim the shapes-layer toolbar — for ROI use we only need the
    # polygon tool + vertex edit + select + delete; rectangle / ellipse /
    # line / path / Z-order shuffling are noise.  Walk the WHOLE napari
    # window for buttons (the shape-mode buttons can sit deep inside
    # the layer-controls stack, several parents under `qt_viewer`), and
    # match against tooltip *and* object name + property hints.
    try:
        # Roots we'll walk in order of specificity
        roots = []
        for attr in ("controls", "layer_controls", "_layer_controls",
                     "dockLayerControls"):
            r = getattr(qtv, attr, None)
            if r is not None:
                # docks may carry the widget on .widget()
                inner = r.widget() if hasattr(r, "widget") and callable(r.widget) else r
                if inner is not None: roots.append(inner)
                roots.append(r)
        roots.append(qtv)
        if qt_window is not None:
            roots.append(qt_window)
        # Tooltip / objectName substrings that identify buttons we hide
        _kill = (
            "rectangle", "ellipse", "line ",
            "add lines", "add paths", "path mode", " path ",
            "polygon lasso", "lasso",
            "move to front", "move to back",
            "raise", "lower",   # napari uses these for Z-order too
        )
        seen: set = set()
        for root in roots:
            try:    btns = root.findChildren(QtWidgets.QAbstractButton)
            except Exception: continue
            for btn in btns:
                if id(btn) in seen:
                    continue
                seen.add(id(btn))
                try:
                    tip  = (btn.toolTip() or "").lower()
                    name = (btn.objectName() or "").lower()
                except Exception:
                    continue
                blob = tip + " | " + name
                if any(k in blob for k in _kill):
                    try:    btn.hide()
                    except Exception: pass
    except Exception:
        pass


def _open_folder(path: str) -> None:
    """Open path in the system file manager."""
    import subprocess
    if sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    elif sys.platform == "win32":
        subprocess.run(["explorer", os.path.normpath(path)], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)


def _qt_message_handler(mode, context, message):
    """Route Qt's own log messages into the FIREFLY log file instead of
    spamming the console.  Warnings/info are recorded to the log only;
    critical/fatal also surface on stderr.  Falls back to stderr if logging
    isn't available."""
    try:
        lg = crash_reporter.get_logger("firefly.qt")
        name = getattr(mode, "name", str(mode))
        if name in ("QtCriticalMsg", "QtFatalMsg"):
            lg.error(message)
        elif name == "QtDebugMsg":
            lg.debug(message)
        else:  # QtWarningMsg / QtInfoMsg — keep them off the console
            lg.info(message)
    except Exception:
        try:
            sys.stderr.write(f"[Qt {getattr(mode, 'name', mode)}] {message}\n")
        except Exception:
            pass
