"""SidebarController — the QML analysis-parameter sidebar bridge (Phase 6).

A thin, schema-driven generic accessor over :mod:`sidebar_schema`.  It owns NO
values — every edit is written straight through SettingsController to the SAME
``analysis/*`` / ``figures/*`` / ``performance/*`` QSettings keys that
``params_builder`` reads, so the worker param dict stays byte-identical (the
sidebar simply replaces the QSettings-default fallback with live user edits).

QML reads values reactively via ``get(key)`` / ``isEnabled(key)`` keyed off the
``revision`` counter (bumped on any change), so a single notify refreshes the
whole sidebar without per-field properties.
"""
from __future__ import annotations

from PySide6.QtCore import Property, QObject, Signal, Slot

from firefly.ui.controllers import sidebar_schema as S


class SidebarController(QObject):
    revisionChanged = Signal()
    fieldChanged = Signal(str)

    def __init__(self, settings, importc=None, parent=None):
        super().__init__(parent)
        self._s = settings
        self._import = importc
        self._revision = 0

    # ── sections / fields ────────────────────────────────────────────────
    @Property("QVariantList", constant=True)
    def sections(self):
        return list(S.SECTIONS)

    @Property(bool, constant=True)
    def hyperflyEligible(self):
        return S.hyperfly_machine_eligible()

    @Slot(str, result="QVariantList")
    def fields(self, section_key):
        """Static field specs for a section (value/enabled fetched separately).
        HYPER-FLY rows are dropped on non-eligible machines (keys stay valid)."""
        elig = S.hyperfly_machine_eligible()
        out = []
        for f in S.FIELDS:
            if f["section"] != section_key:
                continue
            if f["hyperfly"] and not elig:
                continue
            out.append({
                "key": f["key"], "kind": f["kind"], "label": f["label"],
                "items": f["items"], "min": f["min"], "max": f["max"],
                "step": f["step"], "decimals": f["decimals"], "suffix": f["suffix"],
                "special": f["special"], "tooltip": f["tooltip"]})
        return out

    # ── value get / set (coerced per kind) ───────────────────────────────
    @Slot(str, result="QVariant")
    def get(self, key):
        f = S.BY_KEY.get(key)
        if f is None:
            return None
        kind, default = f["kind"], f["default"]
        if kind == "bool":
            return self._s.get_bool(key, default)
        if kind == "int":
            return int(round(self._s.get_float(key, default)))
        if kind == "double":
            return self._s.get_float(key, default)
        return self._s.get_str(key, default)        # combo

    @Slot(str, "QVariant")
    def setValue(self, key, v):
        f = S.BY_KEY.get(key)
        if f is None:
            return
        kind = f["kind"]
        if kind == "bool":
            cv = bool(v)
        elif kind == "int":
            try:    cv = int(round(float(v)))
            except (TypeError, ValueError): return
        elif kind == "double":
            try:    cv = float(v)
            except (TypeError, ValueError): return
        else:
            cv = str(v)
        # clamp numerics to the field range
        if kind in ("int", "double"):
            if f["min"] is not None:
                cv = max(f["min"], cv)
            if f["max"] is not None:
                cv = min(f["max"], cv)
        self._s.set(key, cv)
        self._bump(key)

    @Slot(str, result=bool)
    def isEnabled(self, key):
        f = S.BY_KEY.get(key)
        if f is None or not f.get("enable"):
            return True
        en = f["enable"]
        other = self.get(en["key"])
        if "eq" in en:
            return other == en["eq"]
        if "truthy" in en:
            return bool(other) == bool(en["truthy"])
        return True

    # ── reset ────────────────────────────────────────────────────────────
    @Slot(str)
    def resetSection(self, section_key):
        for f in S.FIELDS:
            if f["section"] == section_key:
                self._s.set(f["key"], f["default"])
        self._bump("")

    @Slot()
    def resetAll(self):
        for f in S.FIELDS:
            self._s.set(f["key"], f["default"])
        self._bump("")

    # ── revision (reactivity) ────────────────────────────────────────────
    @Property(int, notify=revisionChanged)
    def revision(self):
        return self._revision

    def _bump(self, key):
        self._revision += 1
        self.revisionChanged.emit()
        if key:
            self.fieldChanged.emit(key)
