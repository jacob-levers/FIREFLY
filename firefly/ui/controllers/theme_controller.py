"""ThemeController — the single source of design tokens for the QML UI.

Colours come straight from `ui_theme._THEMES` (Dark / Light / AMOLED), so the
QML app and the legacy Widgets app share one palette; the scale tokens (spacing,
radii, shadow, type) mirror `FIREFLY Design System-2/tokens/*.css`. Exposed to
QML as `Theme`; `palette` re-emits on `setTheme(name)` so every binding
re-evaluates — the **live theme switching** the QSS app couldn't do.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Property, QSettings, Signal, Slot

from firefly.ui.ui_theme import _THEMES, _pick_startup_theme


def _font_mult(label):
    """UI font-size label ('Medium — 12px' / 'Small' / 'Large') → text multiplier
    (relative to the 12px base)."""
    import re
    m = re.search(r"(\d+)\s*px", str(label or ""))
    if m:
        return max(0.7, min(1.6, int(m.group(1)) / 12.0))
    lo = str(label or "").lower()
    return 11 / 12 if "small" in lo else 14 / 12 if "large" in lo else 1.0


def _density_mult(label):
    """Interface-density label → spacing multiplier."""
    lo = str(label or "").lower()
    if "comfortable" in lo:
        return 1.18
    if "spacious" in lo:
        return 1.32
    return 1.0       # Compact (default)


class ThemeController(QObject):
    changed = Signal()                       # active palette changed
    scaleChanged = Signal()                  # font-size / density scale changed

    # Static scale tokens (2px grid; radii / shadow / type), from the design
    # system's tokens/*.css. Constant — only the colour palette switches.
    _SCALE = {
        # spacing (2px grid)
        "sp1": 2, "sp2": 4, "sp3": 6, "sp4": 8, "sp5": 10, "sp6": 12,
        "sp8": 16, "sp10": 20, "sp12": 24, "sp16": 32, "sp20": 40, "sp24": 48,
        # radii
        "radiusXs": 3, "radiusSm": 4, "radiusMd": 5, "radiusLg": 6,
        "radiusXl": 8, "radius2xl": 10, "radiusPill": 999,
        # borders
        "borderWidth": 1, "borderAccent": 3, "borderFocus": 2,
        # type scale (product UI; dense desktop)
        "textXs": 11, "textSm": 12, "textMd": 13, "textLg": 14,
        "textXl": 18, "text2xl": 28,
        # brand display scale
        "displaySm": 22, "displayMd": 34, "displayLg": 54, "displayXl": 88,
        # weights
        "weightRegular": 400, "weightMedium": 500, "weightSemibold": 600,
        "weightBold": 700, "weightHeavy": 800,
        # misc
        "eyebrowTracking": 0.18, "sidebarWidth": 380,
    }

    # Accent presets — a colour axis independent of the light/dark theme. Each
    # overrides ACC / ACC_HOVER / ACC_PRESSED in the active palette.
    _ACCENTS = [
        {"name": "Luminous blue",  "v": "#58a6ff", "h": "#79c0ff", "p": "#388bfd"},
        {"name": "Firefly amber",  "v": "#f6a623", "h": "#ffc65c", "p": "#d98e10"},
        {"name": "Motion green",   "v": "#4fe0a0", "h": "#7ef0bd", "p": "#33c084"},
        {"name": "Detection cyan", "v": "#27c0e8", "h": "#5fd4f0", "p": "#15a3c8"},
    ]
    accentChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._name = _pick_startup_theme()
        # reduce-motion lives in the primary store (jacoblevers/FIREFLY).
        try:
            self._reduced_motion = QSettings("jacoblevers", "FIREFLY").value(
                "ui/reduce_motion", False, type=bool)
        except Exception:
            self._reduced_motion = False
        # accent lives alongside the theme (FIREFLY/sptPALM, key ui/accent).
        self._accent = "Luminous blue"
        try:
            a = QSettings("FIREFLY", "sptPALM").value("ui/accent", "Luminous blue")
            if any(x["name"] == a for x in self._ACCENTS):
                self._accent = a
        except Exception:
            pass
        self._sync_marker_accent()      # live-detection dots track the accent
        # font-size + interface-density multipliers (Preferences ▸ Appearance).
        # Live in the primary store (jacoblevers/FIREFLY), written by Settings.
        self._fmult = 1.0
        self._dmult = 1.0
        try:
            ps = QSettings("jacoblevers", "FIREFLY")
            self._fmult = _font_mult(ps.value("ui/font_size", "Medium — 12px"))
            self._dmult = _density_mult(ps.value("ui/density", "Compact"))
        except Exception:
            pass

    def _accent_def(self):
        for a in self._ACCENTS:
            if a["name"] == self._accent:
                return a
        return self._ACCENTS[0]

    def _sync_marker_accent(self):
        """Push the chosen accent colour to the live-detection marker so the
        detection dots match the accent picked in Preferences."""
        try:
            from firefly.ui.controllers.providers import live_frame_provider
            h = str(self._accent_def()["v"]).lstrip("#")
            live_frame_provider.set_marker_rgb(
                (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)))
        except Exception:
            pass

    # ── active palette (the 14 semantic colour tokens) ───────────────────
    @Property("QVariantMap", notify=changed)
    def palette(self):
        pal = dict(_THEMES.get(self._name, _THEMES["Dark"]))
        a = self._accent_def()                  # apply the chosen accent over the theme
        pal["ACC"], pal["ACC_HOVER"], pal["ACC_PRESSED"] = a["v"], a["h"], a["p"]
        return pal

    # ── accent (independent colour axis) ─────────────────────────────────
    @Property("QVariantList", constant=True)
    def accents(self):
        return [dict(a) for a in self._ACCENTS]

    @Property(str, notify=accentChanged)
    def accentName(self):
        return self._accent

    @Slot(str)
    def setAccent(self, name: str):
        if any(a["name"] == name for a in self._ACCENTS) and name != self._accent:
            self._accent = name
            try:
                s = QSettings("FIREFLY", "sptPALM")
                s.setValue("ui/accent", name); s.sync()
            except Exception:
                pass
            self._sync_marker_accent()          # recolour detection dots to match
            self.accentChanged.emit()
            self.changed.emit()                 # repaint every palette binding live

    @Property(str, notify=changed)
    def name(self):
        return self._name

    @Property("QVariantMap", notify=scaleChanged)
    def scale(self):
        # Font size scales the type tokens; density scales the spacing tokens.
        # Radii/weights/sidebarWidth stay fixed for visual consistency.
        sc = dict(self._SCALE)
        if self._fmult != 1.0 or self._dmult != 1.0:
            for k, v in list(sc.items()):
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                if k.startswith("text") or k.startswith("display"):
                    sc[k] = max(1, round(v * self._fmult))
                elif k.startswith("sp"):
                    sc[k] = max(1, round(v * self._dmult))
        return sc

    @Slot(str)
    def setFontSize(self, label):
        m = _font_mult(label)
        if abs(m - self._fmult) > 1e-6:
            self._fmult = m
            self.scaleChanged.emit()

    @Slot(str)
    def setDensity(self, label):
        m = _density_mult(label)
        if abs(m - self._dmult) > 1e-6:
            self._dmult = m
            self.scaleChanged.emit()

    @Property("QStringList", constant=True)
    def themes(self):
        return list(_THEMES.keys())

    reducedMotionChanged = Signal()

    @Property(bool, notify=reducedMotionChanged)
    def reducedMotion(self):
        return self._reduced_motion

    @reducedMotion.setter
    def reducedMotion(self, v: bool):
        if bool(v) != self._reduced_motion:
            self._reduced_motion = bool(v)
            try:
                s = QSettings("jacoblevers", "FIREFLY")
                s.setValue("ui/reduce_motion", self._reduced_motion)
                s.sync()                          # flush now — survive a crash
            except Exception:
                pass
            self.reducedMotionChanged.emit()

    @Slot(str)
    def setTheme(self, name: str):
        if name in _THEMES and name != self._name:
            self._name = name
            # persist to the SAME key the Widgets Figures-tab dropdown writes.
            try:
                s = QSettings("FIREFLY", "sptPALM")
                s.setValue("ui/app_theme", name); s.sync()
            except Exception:
                pass
            self.changed.emit()
