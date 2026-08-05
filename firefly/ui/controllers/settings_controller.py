"""SettingsController — the single owner of QSettings for the QML UI.

Other controllers read/write persisted prefs through this one object (rather
than scattering ``QSettings(...)`` calls), under the SAME org/app name as the
Widgets app so the two share settings. Keys mirror the Widgets app exactly
(e.g. ``analysis/pixel_size``).
"""
from __future__ import annotations

from PySide6.QtCore import QCoreApplication, QObject, QSettings, Signal, Slot


class SettingsController(QObject):
    changed = Signal(str)       # a key was written via any setter → observers refresh

    def __init__(self, parent=None):
        super().__init__(parent)
        self._s = QSettings("jacoblevers", "FIREFLY")
        # The updater asks QApplication to quit immediately after staging the
        # replacement.  Flush buffered preferences at that explicit lifecycle
        # boundary, before a detached update helper could ever time out and
        # terminate a slow shutdown.
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.sync)

    # ── QML-facing generic accessors ─────────────────────────────────────
    @Slot(str, "QVariant", result="QVariant")
    def get(self, key, default=None):
        return self._s.value(key, default)

    @Slot(str, "QVariant")
    def setValue(self, key, value):
        self._s.setValue(key, value)
        self.changed.emit(str(key))

    @Slot()
    def sync(self):
        self._s.sync()

    # ── QML-callable typed accessors ─────────────────────────────────────
    @Slot(str, bool, result=bool)
    def getBool(self, key, default=False):
        return self.get_bool(key, default)

    @Slot(str, str, result=str)
    def getStr(self, key, default=""):
        return self.get_str(key, default)

    # ── typed helpers for Python callers ─────────────────────────────────
    def get_str(self, key, default=""):
        v = self._s.value(key, default)
        return "" if v is None else str(v)

    def get_float(self, key, default=0.0):
        try:
            return float(self._s.value(key, default))
        except (TypeError, ValueError):
            return float(default)

    def get_bool(self, key, default=False):
        v = self._s.value(key, default)
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)

    def set(self, key, value):
        self._s.setValue(key, value)
        self.changed.emit(str(key))
