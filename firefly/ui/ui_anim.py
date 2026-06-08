"""Lightweight UI animation helpers (Qt's widget animation framework — no new
dependencies).

Designed to stay smooth on high-refresh-rate displays.  The widget animation
timer ticks at ~60 fps (the QML scene-graph path is the vsync-native one;
classic widgets aren't), so the way to avoid the stutter people notice on
120/144 Hz is to keep every frame *cheap* and never leave heavy machinery
running:

  * short, eased durations (so the motion reads smooth even at 60 fps);
  * a ``QGraphicsOpacityEffect`` is REMOVED the instant a fade finishes — a
    lingering one re-renders the widget to an offscreen buffer on every single
    repaint, which is the usual cause of "choppy" widget UIs;
  * animations only run on the GUI thread for tiny, bounded property changes
    (heavy compute lives in the worker subprocess, so it never competes).

Every helper is a no-op when the user turns on **Preferences → Reduce motion**.
"""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QPropertyAnimation, QEasingCurve

FAST = 160      # ms — expand/collapse
NORMAL = 200    # ms — fades
_EASE = QEasingCurve.Type.OutCubic
_WIDGET_MAX = 16777215          # Qt's QWIDGETSIZE_MAX — "grow freely"
_running = set()                # keep refs so animations aren't GC'd mid-flight


def reduce_motion() -> bool:
    """True when the user has asked for reduced/!no motion."""
    try:
        v = QtCore.QSettings("jacoblevers", "FIREFLY").value(
            "ui/reduce_motion", False)
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)
    except Exception:
        return False


def _keep(anim):
    _running.add(anim)
    anim.finished.connect(lambda: _running.discard(anim))


def fade_in(widget, duration=NORMAL, delay=0):
    """Fade `widget` 0→1, then drop the opacity effect so it stops buffering.
    No-op under reduce-motion."""
    if widget is None or reduce_motion():
        return None
    eff = QtWidgets.QGraphicsOpacityEffect(widget)
    eff.setOpacity(0.0)
    widget.setGraphicsEffect(eff)
    anim = QPropertyAnimation(eff, b"opacity", widget)
    anim.setDuration(int(duration))
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(_EASE)

    def _done():
        try:
            widget.setGraphicsEffect(None)
        except Exception:
            pass
    anim.finished.connect(_done)
    _keep(anim)
    if delay > 0:
        QtCore.QTimer.singleShot(int(delay), anim.start)
    else:
        anim.start()
    return anim


def animate_height(widget, start, end, duration=FAST, on_finish=None):
    """Animate `widget.maximumHeight` start→end (px).  The caller is
    responsible for visibility and for resetting maximumHeight afterwards (do
    that in `on_finish`).  Under reduce-motion this runs `on_finish`
    immediately and returns None."""
    if widget is None or reduce_motion():
        if on_finish:
            on_finish()
        return None
    anim = QPropertyAnimation(widget, b"maximumHeight", widget)
    anim.setDuration(int(duration))
    anim.setStartValue(int(start))
    anim.setEndValue(int(end))
    anim.setEasingCurve(_EASE)
    if on_finish:
        anim.finished.connect(on_finish)
    _keep(anim)
    anim.start()
    return anim
