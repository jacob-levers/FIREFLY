"""Theme palettes, QSS and theme application for the FIREFLY GUI.

Extracted from app_qt.py (#7); re-exported there for compatibility.
"""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


_THEMES = {
    "Dark": {
        "BG":          "#0d1117",
        "PANEL":       "#161b22",
        "PANEL_ALT":   "#1c2128",
        "BORDER":      "#30363d",
        "BORDER_HI":   "#484f58",
        "TXT":         "#e6edf3",
        "TXT_MUTED":   "#8b949e",
        "ACC":         "#58a6ff",
        "ACC_HOVER":   "#79c0ff",
        "ACC_PRESSED": "#388bfd",
        "ACC_FG":      "#0d1117",
        "DANGER":      "#f85149",
        "SUCCESS":     "#56d364",
        "WARN":        "#f78166",
    },
    # AMOLED — identical to Dark but with pure-black main BG (#000000)
    # so OLED displays power down individual pixels.  PANEL stays a
    # near-black so cards still read as cards against the BG.
    "AMOLED": {
        "BG":          "#000000",
        "PANEL":       "#0a0a0a",
        "PANEL_ALT":   "#141414",
        "BORDER":      "#30363d",
        "BORDER_HI":   "#484f58",
        "TXT":         "#e6edf3",
        "TXT_MUTED":   "#8b949e",
        "ACC":         "#58a6ff",
        "ACC_HOVER":   "#79c0ff",
        "ACC_PRESSED": "#388bfd",
        "ACC_FG":      "#000000",
        "DANGER":      "#f85149",
        "SUCCESS":     "#56d364",
        "WARN":        "#f78166",
    },
    # Light — high-contrast for daytime use.  Mirrors the matplotlib
    # Light figure theme (`_theme_palette("Light")` in sptpalm_analysis.py).
    "Light": {
        "BG":          "#ffffff",
        "PANEL":       "#f6f8fa",
        "PANEL_ALT":   "#eaeef2",
        "BORDER":      "#d0d7de",
        "BORDER_HI":   "#afb8c1",
        "TXT":         "#24292f",
        "TXT_MUTED":   "#57606a",
        "ACC":         "#0969da",
        "ACC_HOVER":   "#218bff",
        "ACC_PRESSED": "#0550ae",
        "ACC_FG":      "#ffffff",
        "DANGER":      "#cf222e",
        "SUCCESS":     "#1f883d",
        "WARN":        "#9a6700",
    },
}


def _pick_startup_theme() -> str:
    """Read the user's chosen app theme from QSettings (the same key
    the Figures-tab dropdown writes to), defaulting to "Dark".  Used
    once at module-load time to select which palette `_THEME` points
    at for the lifetime of this process — see the AMOLED-theme block
    docstring above for why a live-switch isn't currently supported.
    """
    try:
        s = QtCore.QSettings("FIREFLY", "sptPALM")
        name = str(s.value("ui/app_theme", "Dark") or "Dark")
        if name in _THEMES:
            return name
    except Exception:
        pass
    return "Dark"


_ACTIVE_THEME_NAME = _pick_startup_theme()


_THEME = dict(_THEMES[_ACTIVE_THEME_NAME])   # mutable copy for hot-patching


_FIREFLY_QSS = """
/* ── Base ────────────────────────────────────────────────────────────────── */
/* Note: we deliberately DO NOT set a background on the bare QWidget rule.
   Doing so paints every transparent wrapper widget (e.g. the QWidgets used
   as containers for QHBoxLayout rows inside a QGroupBox) in the darkest
   shade, which then shows through as a dark rectangle against the lighter
   panel background.  Widgets that need an explicit background get one
   from their own rule (QMainWindow, QGroupBox, sidebar frame, etc.). */
QWidget {{
    color:            {TXT};
    font-family:      -apple-system, "SF Pro Text", "Segoe UI", "Inter",
                      "Helvetica Neue", Arial, sans-serif;
    font-size:        12px;
}}

QMainWindow, QDialog {{
    background-color: {BG};
}}

/* Sidebar frame gets a slightly different shade so it visually separates
   from the central tab area. */
QMainWindow > QWidget > QFrame[frameShape="6"] {{   /* StyledPanel */
    background-color: {PANEL};
    border-right:     1px solid {BORDER};
}}

/* ── Labels ──────────────────────────────────────────────────────────────── */
QLabel {{
    background:       transparent;
    color:            {TXT};
}}

/* ── Group boxes ─────────────────────────────────────────────────────────── */
QGroupBox {{
    background-color: {PANEL};
    border:           1px solid {BORDER};
    border-radius:    6px;
    /* Big top margin so the title sits cleanly above the rounded
       top border instead of colliding with its top-left corner curve. */
    margin-top:       18px;
    padding:          10px 8px 6px 8px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    /* Negative top lifts the title baseline above the border line so
       descenders ("g" in "Single file" etc.) don't intersect the
       rounded corner.  Horizontal padding hides the border behind
       the title's background swatch on either side of the text. */
    top:               -2px;
    padding:           0 8px;
    margin-left:       8px;
    color:             {TXT_MUTED};
    font-weight:       600;
    font-size:         11px;
    background-color:  {BG};
}}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {PANEL_ALT};
    color:            {TXT};
    border:           1px solid {BORDER};
    border-radius:    5px;
    padding:          5px 12px;
    min-height:       20px;
}}

QPushButton:hover {{
    background-color: {PANEL};
    border-color:     {BORDER_HI};
}}

QPushButton:pressed {{
    background-color: {BG};
    border-color:     {ACC};
}}

QPushButton:disabled {{
    color:            {TXT_MUTED};
    background-color: {PANEL};
    border-color:     {BORDER};
}}

QPushButton:default,
QPushButton#primary {{
    background-color: {ACC};
    color:            {ACC_FG};
    border:           1px solid {ACC};
    font-weight:      600;
}}

QPushButton:default:hover,
QPushButton#primary:hover {{
    background-color: {ACC_HOVER};
    border-color:     {ACC_HOVER};
}}

QPushButton:default:pressed,
QPushButton#primary:pressed {{
    background-color: {ACC_PRESSED};
    border-color:     {ACC_PRESSED};
}}

QToolButton {{
    background:       transparent;
    color:            {TXT_MUTED};
    border:           1px solid transparent;
    border-radius:    4px;
    padding:          2px 6px;
}}

QToolButton:hover {{
    color:            {DANGER};
    background:       {PANEL_ALT};
    border-color:     {BORDER};
}}

/* ── Line edits / spinboxes / combos ─────────────────────────────────────── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background-color: {PANEL_ALT};
    color:            {TXT};
    border:           1px solid {BORDER};
    border-radius:    4px;
    padding:          3px 6px;
    selection-background-color: {ACC};
    selection-color:  {ACC_FG};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
    border-color:     {ACC};
}}

QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background-color: {PANEL};
    color:            {TXT_MUTED};
}}

/* Spinbox stepper buttons are removed entirely via NoButtons in the
   _QuietSpinBox subclasses — these QSS rules collapse the cells so any
   stray third-party spinbox (e.g. from napari's plugin UI) also reads
   as a clean borderless number field. */
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: transparent;
    border:           none;
    width:            0;
    height:           0;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow,
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image:  none;
    width:  0;
    height: 0;
}}

QComboBox::drop-down {{
    subcontrol-origin:    padding;
    subcontrol-position:  top right;
    background:           transparent;
    background-color:     transparent;
    border:               none;
    width:                22px;
}}
/* Drop-down indicator rendered as a small light circle instead of the
   default arrow / rectangle.  Width = height + border-radius half-of-side
   keeps it perfectly round. */
QComboBox::down-arrow {{
    image:                none;
    width:                8px;
    height:               8px;
    border-radius:        4px;
    background-color:     {TXT_MUTED};
    margin:               0 8px 0 0;
}}
QComboBox::down-arrow:on,
QComboBox::down-arrow:hover {{
    background-color:     {ACC};
}}
QComboBox QAbstractItemView {{
    background-color: {PANEL};
    color:            {TXT};
    border:           1px solid {BORDER};
    selection-background-color: {ACC};
    selection-color:  {ACC_FG};
    outline:          0;
}}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
QTabWidget::pane {{
    background-color: {BG};
    /* Pane carries side + bottom border only — the top border is provided
       by the tab bar's own bottom-edge hairline, so the selected tab's
       bottom edge can sit flush against the pane without a misaligned
       corner-radius seam where the curve meets the tab's straight bottom. */
    border:                 1px solid {BORDER};
    border-top:             none;
    border-bottom-left-radius:  6px;
    border-bottom-right-radius: 6px;
    border-top-left-radius:     0px;
    border-top-right-radius:    0px;
    top:                   -1px;
}}

/* Hairline under the tab bar — replaces what was the pane's top border.
   Drawn as a 1px solid bottom of the tab-bar element so it can be
   covered cleanly by the selected tab via a negative bottom margin. */
QTabBar {{
    border-bottom: 1px solid {BORDER};
}}

QTabBar::tab {{
    background-color: transparent;
    color:            {TXT_MUTED};
    border:           1px solid transparent;
    padding:          6px 14px;
    margin-right:     2px;
    border-top-left-radius:  6px;
    border-top-right-radius: 6px;
}}

QTabBar::tab:hover {{
    color:            {TXT};
    background-color: {PANEL};
}}

QTabBar::tab:selected {{
    color:            {TXT};
    background-color: {BG};
    border:           1px solid {BORDER};
    border-bottom:    1px solid {BG};
    /* Pull 1 px down so the tab's bottom edge overlaps (covers) the
       hairline on QTabBar, killing the gap that used to show where
       the selected tab met the pane's rounded corner. */
    margin-bottom:    -1px;
    font-weight:      600;
}}

/* ── Progress bar ────────────────────────────────────────────────────────── */
QProgressBar {{
    background-color: {PANEL_ALT};
    border:           1px solid {BORDER};
    border-radius:    5px;
    text-align:       center;
    color:            {TXT};
    min-height:       18px;
}}
QProgressBar::chunk {{
    background-color: {ACC};
    border-radius:    4px;
}}

/* ── List widgets (folder lists, batch file list) ───────────────────────── */
QListWidget {{
    background-color: {PANEL_ALT};
    color:            {TXT};
    border:           1px solid {BORDER};
    border-radius:    5px;
    outline:          0;
    alternate-background-color: {PANEL};
}}

QListWidget::item {{
    padding:          4px 6px;
    border-radius:    3px;
}}

QListWidget::item:hover {{
    background-color: {PANEL};
}}

QListWidget::item:selected {{
    background-color: {ACC};
    color:            {ACC_FG};
}}

/* ── Checkboxes ─────────────────────────────────────────────────────────── */
QCheckBox, QRadioButton {{
    spacing:          6px;
    background:       transparent;
}}

QCheckBox::indicator, QRadioButton::indicator {{
    width:            14px;
    height:           14px;
    border:           1px solid {BORDER_HI};
    background-color: {PANEL_ALT};
    border-radius:    3px;
}}

QRadioButton::indicator {{
    border-radius:    8px;
}}

QCheckBox::indicator:hover, QRadioButton::indicator:hover {{
    border-color:     {ACC};
}}

QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {ACC};
    border-color:     {ACC};
    image:            none;
}}

/* ── Scrollbars ─────────────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background:       transparent;
    width:            10px;
    margin:           0;
}}
QScrollBar::handle:vertical {{
    background:       {BORDER};
    min-height:       24px;
    border-radius:    5px;
}}
QScrollBar::handle:vertical:hover {{
    background:       {BORDER_HI};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    background:       transparent; border: none; height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background:       transparent;
}}

QScrollBar:horizontal {{
    background:       transparent;
    height:           10px;
    margin:           0;
}}
QScrollBar::handle:horizontal {{
    background:       {BORDER};
    min-width:        24px;
    border-radius:    5px;
}}
QScrollBar::handle:horizontal:hover {{
    background:       {BORDER_HI};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    background:       transparent; border: none; width: 0;
}}

/* ── Splitter handle ────────────────────────────────────────────────────── */
QSplitter::handle {{
    background:       {BORDER};
}}
QSplitter::handle:hover {{
    background:       {BORDER_HI};
}}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical   {{ height: 1px; }}

/* ── Status bar ─────────────────────────────────────────────────────────── */
QStatusBar {{
    background-color: {PANEL};
    color:            {TXT_MUTED};
    border-top:       1px solid {BORDER};
}}

/* ── ScrollArea (sidebar parameter list) ────────────────────────────────── */
QScrollArea {{
    background-color: {PANEL};
    border:           none;
}}

/* ── Tooltips ───────────────────────────────────────────────────────────── */
QToolTip {{
    background-color: {PANEL};
    color:            {TXT};
    border:           1px solid {BORDER};
    padding:          4px 6px;
    border-radius:    4px;
}}

/* ── Collapsible section (accordion-style) ───────────────────────────────── */
QToolButton#section_header {{
    background-color:    {PANEL_ALT};
    color:               {TXT};
    border:              1px solid {BORDER};
    border-top-left-radius:  5px;
    border-top-right-radius: 5px;
    padding:             7px 10px;
    margin-top:          4px;
    text-align:          left;
    font-weight:         600;
    font-size:           12px;
}}
QToolButton#section_header:hover {{
    border-color:        {BORDER_HI};
    background-color:    {PANEL};
}}
QToolButton#section_header:checked {{
    border-bottom-left-radius:  0;
    border-bottom-right-radius: 0;
}}
QFrame#section_content {{
    background-color:    {PANEL};
    border:              1px solid {BORDER};
    border-top:          none;
    border-bottom-left-radius:  5px;
    border-bottom-right-radius: 5px;
}}

/* ── Mode-toggle tiles (Import tab) ─────────────────────────────────────── */
/* Big segmented-control cards.  Custom QFrame subclass (_ModeTile) with
   a 'checked' Qt property — QSS uses property selectors to switch the
   border between border-color: BORDER (unchecked) and ACC (checked). */
QFrame#mode_tile {{
    background-color:    {PANEL};
    border:              1px solid {BORDER};
    border-radius:       8px;
}}
QFrame#mode_tile:hover {{
    background-color:    {PANEL_ALT};
    border-color:        {BORDER_HI};
}}
QFrame#mode_tile[checked="true"] {{
    background-color:    {PANEL_ALT};
    border:              2px solid {ACC};
}}
QFrame#mode_tile[checked="true"]:hover {{
    border-color:        {ACC_HOVER};
}}

QLabel#mode_tile_title {{
    font-size:           14px;
    font-weight:         700;
    color:               {TXT};
    background:          transparent;
    border:              none;
}}
QLabel#mode_tile_subtitle {{
    font-size:           11px;
    color:               {TXT_MUTED};
    background:          transparent;
    border:              none;
}}

/* ── Action tiles (Home / landing tab) ──────────────────────────────────── */
QFrame#action_tile {{
    background-color:    {PANEL};
    border:              1px solid {BORDER};
    border-radius:       10px;
}}
QFrame#action_tile:hover {{
    background-color:    {PANEL_ALT};
    border:              1px solid {ACC};
}}
QLabel#action_tile_icon {{
    font-size:           28px;
    color:               {ACC};
    background:          transparent;
    border:              none;
}}
QLabel#action_tile_title {{
    font-size:           18px;
    font-weight:         700;
    color:               {TXT};
    background:          transparent;
    border:              none;
}}
QLabel#action_tile_desc {{
    font-size:           12px;
    color:               {TXT_MUTED};
    background:          transparent;
    border:              none;
}}

/* ── Results panel ──────────────────────────────────────────────────────── */
QFrame#resource_monitor {{
    background-color:    {PANEL};
    border:              1px solid {BORDER};
    border-radius:       4px;
}}

QFrame#results_panel {{
    background-color:    {PANEL};
    border:              1px solid {BORDER};
    border-radius:       6px;
}}
QFrame#results_panel QLabel {{ background: transparent; border: none; }}
QListWidget#results_files {{
    background-color:    {PANEL_ALT};
    border:              1px solid {BORDER};
    border-radius:       4px;
}}

/* ── Menus ──────────────────────────────────────────────────────────────── */
QMenu {{
    background-color: {PANEL};
    color:            {TXT};
    border:           1px solid {BORDER};
    padding:          4px;
}}
QMenu::item {{
    padding:          5px 18px;
    border-radius:    3px;
}}
QMenu::item:selected {{
    background-color: {ACC};
    color:            {ACC_FG};
}}
""".format(**_THEME)


def _apply_firefly_theme(app: QtWidgets.QApplication):
    """Apply the FIREFLY dark theme: QPalette + comprehensive QSS.

    Also nudge the platform style toward "Fusion" — macOS's native style
    ignores most QSS properties (background colours, borders), so without
    Fusion the stylesheet would only partially apply.  Fusion respects
    everything in our QSS and renders identically on macOS / Windows /
    Linux, which is what we want for a cohesive look.
    """
    # Fusion style — required on macOS for our QSS to actually take effect.
    # Without this, the system style overrides background-color etc.
    app.setStyle("Fusion")

    # QPalette — mostly redundant alongside QSS but covers the few widgets
    # that don't read QSS (some native dialogs, scroll bars on some
    # platforms).  Keeps us looking consistent everywhere.
    pal = QtGui.QPalette()
    bg     = QtGui.QColor(_THEME["BG"])
    panel  = QtGui.QColor(_THEME["PANEL"])
    txt    = QtGui.QColor(_THEME["TXT"])
    muted  = QtGui.QColor(_THEME["TXT_MUTED"])
    acc    = QtGui.QColor(_THEME["ACC"])
    border = QtGui.QColor(_THEME["BORDER"])
    pal.setColor(QtGui.QPalette.ColorRole.Window,          bg)
    pal.setColor(QtGui.QPalette.ColorRole.WindowText,      txt)
    pal.setColor(QtGui.QPalette.ColorRole.Base,            panel)
    pal.setColor(QtGui.QPalette.ColorRole.AlternateBase,   bg)
    pal.setColor(QtGui.QPalette.ColorRole.ToolTipBase,     panel)
    pal.setColor(QtGui.QPalette.ColorRole.ToolTipText,     txt)
    pal.setColor(QtGui.QPalette.ColorRole.Text,            txt)
    pal.setColor(QtGui.QPalette.ColorRole.PlaceholderText, muted)
    pal.setColor(QtGui.QPalette.ColorRole.Button,          panel)
    pal.setColor(QtGui.QPalette.ColorRole.ButtonText,      txt)
    pal.setColor(QtGui.QPalette.ColorRole.Highlight,       acc)
    pal.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(_THEME["ACC_FG"]))
    pal.setColor(QtGui.QPalette.ColorRole.Link,            acc)
    pal.setColor(QtGui.QPalette.ColorRole.Mid,             border)
    pal.setColor(QtGui.QPalette.ColorRole.Dark,            bg)
    pal.setColor(QtGui.QPalette.ColorRole.Shadow,          bg)
    app.setPalette(pal)

    app.setStyleSheet(_FIREFLY_QSS)
