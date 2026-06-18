"""PresetController — save / load named analysis-parameter presets (Phase 6d).

A preset is a JSON snapshot of the sidebar's schema keys, one file per preset at
``~/.firefly/presets/<name>.json`` (the same location + format as the Widgets
app, so presets are shared).  The snapshot/apply goes through the
SidebarController (which writes the QSettings keys params_builder reads), so a
loaded preset reconfigures a run exactly.  A ``modified`` flag tracks divergence
from the active preset for the "• modified" pill.
"""
from __future__ import annotations

import json
import os
import re

from PySide6 import QtWidgets
from PySide6.QtCore import Property, QObject, Signal, Slot

_SENTINEL = "— Current settings —"
_BUILTIN_TAG = "__firefly_builtin__"


class PresetController(QObject):
    namesChanged = Signal()
    activeChanged = Signal()
    modifiedChanged = Signal()
    statusMessage = Signal(str)

    def __init__(self, sidebar, parent=None):
        super().__init__(parent)
        self._sidebar = sidebar
        self._active = ""
        self._baseline: dict = {}
        # recompute the modified flag whenever any sidebar field changes
        try:
            self._sidebar.fieldChanged.connect(lambda *_: self.modifiedChanged.emit())
            self._sidebar.revisionChanged.connect(self.modifiedChanged.emit)
        except Exception:
            pass

    # ── files ────────────────────────────────────────────────────────────
    @staticmethod
    def _dir():
        d = os.path.join(os.path.expanduser("~"), ".firefly", "presets")
        os.makedirs(d, exist_ok=True)
        return d

    def _path(self, name):
        return os.path.join(self._dir(), f"{name}.json")

    def _list(self):
        try:
            return sorted(f[:-5] for f in os.listdir(self._dir())
                          if f.endswith(".json"))
        except Exception:
            return []

    # ── model ────────────────────────────────────────────────────────────
    @Property("QStringList", notify=namesChanged)
    def names(self):
        """Preset names with the leading '— Current settings —' sentinel."""
        return [_SENTINEL] + self._list()

    @Property(str, notify=activeChanged)
    def active(self):
        return self._active or _SENTINEL

    @Property(bool, notify=modifiedChanged)
    def modified(self):
        if not self._baseline:
            return False
        snap = self._sidebar.snapshot()
        return any(snap.get(k) != v for k, v in self._baseline.items())

    @Slot()
    def refresh(self):
        self.namesChanged.emit()

    # ── save / load / delete ─────────────────────────────────────────────
    @Slot(str, result=bool)
    def save(self, name):
        name = re.sub(r"[^A-Za-z0-9 _-]+", "", str(name or "")).strip()
        if not name or name.startswith("—"):
            self.statusMessage.emit("Enter a preset name (letters, numbers, - _).")
            return False
        state = self._sidebar.snapshot()
        try:
            with open(self._path(name), "w", encoding="utf-8") as fh:
                json.dump(state, fh, indent=2)
        except Exception as exc:
            self.statusMessage.emit(f"Save failed: {exc}")
            return False
        self._active = name
        self._baseline = dict(state)
        self.namesChanged.emit()
        self.activeChanged.emit()
        self.modifiedChanged.emit()
        self.statusMessage.emit(f"Saved preset “{name}”.")
        return True

    @Slot(str)
    def load(self, name):
        if not name or name.startswith("—"):     # sentinel → no-op
            self._active = ""
            self._baseline = {}
            self.activeChanged.emit()
            self.modifiedChanged.emit()
            return
        try:
            with open(self._path(name), encoding="utf-8") as fh:
                state = json.load(fh)
        except Exception as exc:
            self.statusMessage.emit(f"Couldn't load “{name}”: {exc}")
            return
        state.pop(_BUILTIN_TAG, None)
        self._sidebar.applyState(state)
        self._active = name
        # baseline = the applied values as the sidebar coerced them
        snap = self._sidebar.snapshot()
        self._baseline = {k: snap.get(k) for k in state if k in snap}
        self.activeChanged.emit()
        self.modifiedChanged.emit()
        self.statusMessage.emit(f"Loaded preset “{name}”.")

    @Slot()
    def saveAs(self):
        """Prompt for a name (native dialog) and save the current settings."""
        name, ok = QtWidgets.QInputDialog.getText(
            None, "Save preset", "Preset name:",
            text=("" if self._active.startswith("—") else self._active))
        if ok and name.strip():
            self.save(name)

    @Slot(str)
    def confirmRemove(self, name):
        if not name or name.startswith("—"):
            return
        ret = QtWidgets.QMessageBox.question(
            None, "Delete preset", f"Delete preset “{name}”?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No)
        if ret == QtWidgets.QMessageBox.StandardButton.Yes:
            self.remove(name)

    @Slot(str, result=bool)
    def remove(self, name):
        if not name or name.startswith("—"):
            return False
        try:
            os.remove(self._path(name))
        except FileNotFoundError:
            pass
        except Exception as exc:
            self.statusMessage.emit(f"Delete failed: {exc}")
            return False
        if self._active == name:
            self._active = ""
            self._baseline = {}
            self.activeChanged.emit()
        self.namesChanged.emit()
        self.modifiedChanged.emit()
        self.statusMessage.emit(f"Deleted preset “{name}”.")
        return True
