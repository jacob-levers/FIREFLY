"""MainWindow BuildMixin methods, split out of app_qt.py (#7)."""
from __future__ import annotations
from firefly.analysis.fa_constants import N_CPUS

import os
import sys
import time
import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar

from firefly import sptpalm_analysis
from firefly import crash_reporter
from firefly import cuda_installer
from firefly.ui.ui_theme import _THEME
from firefly.ui.ui_constants import (TAB_IMPORT, TAB_ANALYSIS, TAB_COMPARE,
                          TAB_VISUALISE, TAB_REPROCESS)
from firefly.ui.ui_helpers import (_make_cogwheel_icon, _make_close_x_icon,
                        _make_napari_container_layout_opaque, _hide_napari_chrome,
                        _register_motion_colormap, _open_folder,
                        _MOTION_PALETTE, _MOTION_ORDER, _MOTION_CMAP_NAME)
from firefly.ui.ui_widgets import (_UpdateCheckThread, _ModeTile, _ActionTile, _QuietSpinBox,
                        _QuietDoubleSpinBox, _QuietComboBox, _CollapsibleSection,
                        _ResourceMonitor, _MassHistogram, _LiveFrameView,
                        _TrackInspector, _ResultsPanel, _RoiDialog, _RoiViewer,
                        _FolderDropList, _CompareGroupCard, _PreferencesDialog,
                        _load_imagej_roi_polygons, _load_tif_mask_polygons,
                        _load_any_roi_file)


class BuildMixin:
    def _build_ui(self):
        central = QtWidgets.QWidget()
        # Seal the centralWidget so its natural sizeHint (sum of header
        # banner + main_stack page contents) can't override the window
        # size requested via `self.resize(_w, _h)` in `__init__`.  The
        # window opens at the size we asked for; user can drag freely.
        central.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Ignored)
        central.setMinimumSize(0, 0)
        self.setCentralWidget(central)

        # Top-level vertical: [header bar] / [stack: landing OR main UI]
        top = QtWidgets.QVBoxLayout(central)
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(0)

        # ── Header banner ────────────────────────────────────────────────
        top.addWidget(self._build_header_banner())

        # ── Stacked body: landing page (idx 0) vs main UI (idx 1) ────────
        # Landing is a one-way gateway — once the user picks an action it
        # disappears for the rest of the session.  Main UI rebuilds the
        # sidebar + tab interface from before.
        self._main_stack = QtWidgets.QStackedWidget()
        # Seal the central stack so its current page's natural sizeHint
        # (landing-page hero text, sidebar widths, etc.) can't grow the
        # MainWindow past the size requested in __init__.  Combined with
        # the same treatment on the QTabWidget and napari containers,
        # the FIREFLY window's natural size is determined solely by
        # `self.resize(_w, _h)`, freely overridable by user drag.
        self._main_stack.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Ignored)
        self._main_stack.setMinimumSize(0, 0)
        top.addWidget(self._main_stack, stretch=1)
        self._main_stack.addWidget(self._build_landing_page())

        body = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._main_stack.addWidget(body)

        # ── Sidebar ───────────────────────────────────────────────────────
        # Fixed-width left panel; the scrollable parameter list lives
        # inside it so the Start/Stop button can stay pinned at the bottom
        # regardless of how far the user has scrolled.  380 px is wide
        # enough to fit a [QLineEdit + Browse] row at typical font sizes
        # without the button clipping over the line edit's right edge.
        # Sidebar is a fixed-width frame containing TWO QStackedWidgets:
        # the upper one swaps its content per tab (parameters for
        # Import, filters for Visualise, etc.); the lower one swaps
        # the pinned bottom button (Start for Import, Re-run for
        # Re-process, etc.).  Tab → page-index mapping is set in
        # `_on_tab_changed_swap_sidebar`.
        sidebar = QtWidgets.QFrame()
        sidebar.setFixedWidth(380)
        sidebar.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        sb_outer = QtWidgets.QVBoxLayout(sidebar)
        sb_outer.setContentsMargins(0, 0, 0, 0)
        sb_outer.setSpacing(0)

        # Top header — title swaps per tab so the sidebar self-labels.
        self._sidebar_title = QtWidgets.QLabel("Analysis Parameters")
        self._sidebar_title.setStyleSheet(
            f"color: {_THEME['TXT']}; font-weight: 700; font-size: 13px; "
            f"padding: 12px 12px 4px 12px;")
        sb_outer.addWidget(self._sidebar_title)

        self._sidebar_stack = QtWidgets.QStackedWidget()
        sb_outer.addWidget(self._sidebar_stack, stretch=1)

        # Build the Import page now — the rest are built post-tabs
        # because they re-parent widgets created by the tab builders
        # (which run later in __init__).
        import_page = QtWidgets.QWidget()
        ip_v = QtWidgets.QVBoxLayout(import_page)
        ip_v.setContentsMargins(0, 0, 0, 0); ip_v.setSpacing(0)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll_inner = QtWidgets.QWidget()
        sb_layout = QtWidgets.QVBoxLayout(scroll_inner)
        sb_layout.setContentsMargins(12, 0, 12, 12)
        sb_layout.setSpacing(8)
        scroll.setWidget(scroll_inner)
        ip_v.addWidget(scroll)
        self._build_sidebar(sb_layout)
        self._propagate_form_tooltips(scroll_inner)
        self._sidebar_stack.addWidget(import_page)   # index 0

        # The other 4 sidebar pages are added later by
        # `_build_remaining_sidebar_pages()` after the tabs are built.

        # Pinned bottom-button area — also a QStackedWidget so each tab
        # can have its own primary action (or none).
        self._sidebar_action = QtWidgets.QStackedWidget()
        sb_outer.addWidget(self._sidebar_action)

        # Page 0 — Import: the Start/Stop button.
        action_import = QtWidgets.QWidget()
        ai_v = QtWidgets.QVBoxLayout(action_import)
        ai_v.setContentsMargins(12, 6, 12, 12)
        self.btn_run = QtWidgets.QPushButton("Start")
        self.btn_run.setObjectName("primary")  # picks up accent-fill QSS rule
        self.btn_run.setMinimumHeight(36)
        f = self.btn_run.font(); f.setBold(True); f.setPointSize(13)
        self.btn_run.setFont(f)
        self.btn_run.clicked.connect(self._on_run_clicked)
        ai_v.addWidget(self.btn_run)
        self._sidebar_action.addWidget(action_import)   # index 0

        layout.addWidget(sidebar)

        # ── Tabs ──────────────────────────────────────────────────────────
        # Tab order:  Run → Compare → Visualise → Re-process
        # The Run tab merges the old Import + Analysis surfaces: input
        # mode tiles + path pickers up top, status / progress / results
        # panel below.  Figures tab has moved into the Preferences
        # dialog (cogwheel button in the header) — its widgets are
        # built up-front and re-parented into the dialog at open time.
        # Tab order: Import → Analysis → Compare → Visualise → Re-process.
        # Figures has moved into Preferences (cogwheel in the header).
        self.tabs = QtWidgets.QTabWidget()
        self._build_import_tab()
        self._build_analysis_tab()
        # Figures widget is built once, parked unattached, and re-parented
        # into the Preferences dialog when it opens.
        self._figures_widget = self._build_figures_widget()
        self._build_compare_tab()
        self._build_visualise_tab()
        # Seal the QTabWidget too — same trick as `_make_napari_container_layout_opaque`.
        # Without this, the natural sizeHint of the widest tab body
        # (Import's mode tiles + path-picker rows) pushes the window
        # past the requested 1240px on first show.  Ignored policy +
        # zero min size means the tabs consume whatever the parent
        # layout's stretch=1 gives them and never ask for more.
        self.tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Ignored)
        self.tabs.setMinimumSize(0, 0)
        layout.addWidget(self.tabs, stretch=1)

        # Now that all tabs are built and their inline widgets exist,
        # populate the remaining sidebar pages (which re-parent those
        # widgets) and wire up the tab-change → sidebar-swap signal.
        self._build_remaining_sidebar_pages()
        self.tabs.currentChanged.connect(self._on_tab_changed_swap_sidebar)

        # Start on the landing page; main UI activates only after the user
        # picks an action card.
        self._main_stack.setCurrentIndex(0)

        # ── Console dock (hidden by default) ──────────────────────────────
        # One shared console for all tabs.  Stays hidden until the user
        # clicks the Console button in the status bar.  Log lines accumulate
        # in the widget even when the dock is hidden, so opening it shows
        # the complete history.
        self._build_console_dock()

        # Status bar with a permanent "Console" toggle button on the right
        self.btn_show_console = QtWidgets.QToolButton()
        self.btn_show_console.setText("Console")
        self.btn_show_console.setCheckable(True)
        self.btn_show_console.setToolTip(
            "Show/hide the debug console.  Captures every log line from "
            "all stages — useful for diagnosing problems but normally not "
            "needed; the progress bar tells you what's happening.")
        self.btn_show_console.clicked.connect(self._toggle_console)
        self.statusBar().addPermanentWidget(self.btn_show_console)
        self.statusBar().showMessage("Ready")

    def _build_header_banner(self) -> QtWidgets.QWidget:
        """Thin header strip:  FIREFLY                Fluorescence Inference & Reconstruction Engine | By Jacob Levers"""
        bar = QtWidgets.QFrame()
        bar.setObjectName("header_bar")
        bar.setStyleSheet(
            f"QFrame#header_bar {{ background-color: {_THEME['PANEL']}; "
            f"border-bottom: 1px solid {_THEME['BORDER']}; }}"
            # The right-side label is a transparent QLabel inside the
            # styled frame; explicitly null out its border so it doesn't
            # inherit the frame's border-bottom rule from the global QSS.
            f"QFrame#header_bar QLabel {{ background: transparent; border: none; }}"
        )
        bar.setFixedHeight(46)
        h = QtWidgets.QHBoxLayout(bar)
        h.setContentsMargins(18, 0, 18, 0)
        h.setSpacing(8)

        # Left: FIREFLY logo
        logo = QtWidgets.QLabel("FIREFLY")
        logo.setStyleSheet(
            f"color: {_THEME['ACC']}; font-weight: 800; "
            f"font-size: 22px; letter-spacing: 2px;"
        )
        logo.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        h.addWidget(logo)

        h.addStretch(1)

        # "Update available" pill — hidden on startup, lit up by the
        # background update-check thread if GitHub Releases reports a
        # newer tag than __version__.  Clicking opens the Releases page.
        self.btn_update_pill = QtWidgets.QPushButton("")
        self.btn_update_pill.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_update_pill.setVisible(False)
        self.btn_update_pill.setStyleSheet(
            f"QPushButton {{ background-color: {_THEME['ACC']}; "
            f"color: {_THEME['ACC_FG']}; border: none; "
            "border-radius: 10px; padding: 4px 10px; "
            "font-size: 11px; font-weight: 700; } "
            f"QPushButton:hover {{ background-color: {_THEME['ACC_HOVER']}; }}")
        self.btn_update_pill.clicked.connect(self._on_update_pill_clicked)
        h.addWidget(self.btn_update_pill)

        # Right: tagline + author on ONE line, joined with a pipe.  Using
        # rich-text formatting on a single QLabel sidesteps the nested-
        # container-border issue and looks tidier than two stacked labels.
        right = QtWidgets.QLabel(
            f"<span style='color:{_THEME['TXT']};font-weight:600;'>"
            f"Fluorescence Inference &amp; Reconstruction Engine"
            f"</span>"
            f"<span style='color:{_THEME['TXT_MUTED']};'>"
            f"&nbsp;&nbsp;|&nbsp;&nbsp;By Jacob Levers"
            f"</span>"
        )
        right.setTextFormat(Qt.TextFormat.RichText)
        right.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        right.setStyleSheet("font-size: 12px;")
        h.addWidget(right)

        # Header cogwheel for Preferences.  Hidden initially (the landing
        # page has its own Settings tile); revealed once the user enters
        # the main UI via `_enter_main_ui`, where it stays as a permanent
        # cross-tab shortcut for app-wide settings.  No quit button —
        # macOS menubar / window-close button / landing-page Quit tile
        # already cover every exit path.
        h.addSpacing(10)
        _ICON_PX = 18           # icon size inside the button
        _BTN_SIZE = 34          # button bounding box (square)

        self.btn_header_prefs = QtWidgets.QToolButton()
        self.btn_header_prefs.setObjectName("header_btn")
        self.btn_header_prefs.setIcon(_make_cogwheel_icon(
            color=QtGui.QColor(_THEME["TXT_MUTED"]), px=_ICON_PX))
        self.btn_header_prefs.setIconSize(QtCore.QSize(_ICON_PX, _ICON_PX))
        self.btn_header_prefs.setFixedSize(_BTN_SIZE, _BTN_SIZE)
        self.btn_header_prefs.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.btn_header_prefs.setToolTip("Preferences  (⌘,)")
        self.btn_header_prefs.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_header_prefs.setAutoRaise(True)
        self.btn_header_prefs.setStyleSheet(
            f"QToolButton#header_btn {{"
            f"  background: transparent;"
            f"  border: none;"
            f"  padding: 0px;"
            f"  border-radius: 6px;"
            f"}}"
            f"QToolButton#header_btn:hover {{"
            f"  background-color: rgba(255, 255, 255, 0.08);"
            f"}}"
            f"QToolButton#header_btn:pressed {{"
            f"  background-color: rgba(255, 255, 255, 0.14);"
            f"}}")
        self.btn_header_prefs.clicked.connect(self._open_preferences)
        self.btn_header_prefs.hide()      # shown by `_enter_main_ui`
        h.addWidget(self.btn_header_prefs)

        # ⌘, / Ctrl+,  shortcut for the Preferences dialog
        _prefs_sc = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+,"), self)
        _prefs_sc.activated.connect(self._open_preferences)

        # Fire off the update check 2 s after startup so it doesn't
        # block the initial paint.
        QtCore.QTimer.singleShot(2000, self._kick_off_update_check)

        return bar

    def _build_console_dock(self):
        """Create the dockable console.  Hidden by default — toggled via
        the status-bar Console button.  Shared by all tabs, so log lines
        from any analysis stage land in one place.
        """
        self._console_dock = QtWidgets.QDockWidget("Console", self)
        self._console_dock.setObjectName("console_dock")
        self._console_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea)

        self.console_log = QtWidgets.QPlainTextEdit()
        self.console_log.setReadOnly(True)
        self.console_log.setMaximumBlockCount(20000)
        mono = QtGui.QFont("Menlo, Consolas, monospace")
        mono.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        mono.setPointSize(10)
        self.console_log.setFont(mono)
        self._console_dock.setWidget(self.console_log)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea,
                           self._console_dock)
        # Docked at the bottom by default — Qt will shrink the central
        # widget to accommodate, rather than growing the window.  Users
        # can still drag the dock's title bar out to float it, or close
        # it via the × button.  Hidden until the Console toolbar button
        # is clicked.
        self._console_dock.setFloating(False)
        self._console_dock.hide()
        # Minimum width matters when the dock lands on the right side —
        # without it, Qt opens a ~40-px-wide strip where no log text
        # can actually be read.  Tall enough to show 12 lines bottom-docked.
        # Set on BOTH the inner widget AND the dock itself; Qt's splitter
        # honours the dock's own minimums, not just the child widget's.
        self.console_log.setMinimumWidth(360)
        self.console_log.setMinimumHeight(120)
        self._console_dock.setMinimumWidth(360)
        self._console_dock.setMinimumHeight(120)
        # Keep the toggle button in sync when the dock is closed via its
        # own ✕ button
        self._console_dock.visibilityChanged.connect(self._on_console_visibility)
        # When the user drags the dock to a different edge, give it a
        # sensible width / height.  Without this the right-docked
        # console opens as a 1-column strip — the regression the user hit.
        self._console_dock.dockLocationChanged.connect(
            self._on_console_dock_moved)

    def _build_sidebar(self, layout: QtWidgets.QVBoxLayout):
        """Build the full B1.1 parameter panel — every knob from the Tk
        sidebar grouped into collapsible QGroupBox sections.

        The "Analysis Parameters" header lives on the OUTER sidebar
        frame (`self._sidebar_title`, set per-tab in _build_sidebar
        scaffolding) — don't re-add it here or it duplicates.
        """
        # ── Presets ───────────────────────────────────────────────────────
        # Quick switcher for labelled parameter bundles.  Selecting a
        # preset applies its widget snapshot to the rest of the sidebar.
        preset_row = QtWidgets.QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 6)
        preset_row.setSpacing(6)
        preset_row.addWidget(QtWidgets.QLabel("Preset"))
        self.c_preset = _QuietComboBox()
        self.c_preset.setToolTip(
            "Switch between labelled parameter bundles stored in\n"
            "~/.firefly/presets/.  Two defaults ship out of the box;\n"
            "use the disk icon to save the current sidebar as a new one.")
        preset_row.addWidget(self.c_preset, 1)
        self.btn_preset_save = QtWidgets.QToolButton()
        self.btn_preset_save.setText("Save")
        self.btn_preset_save.setToolTip(
            "Save the current sidebar values as a new preset.")
        self.btn_preset_save.clicked.connect(self._on_preset_save)
        preset_row.addWidget(self.btn_preset_save)
        self.btn_preset_delete = QtWidgets.QToolButton()
        self.btn_preset_delete.setText("Delete")
        self.btn_preset_delete.setToolTip(
            "Delete the currently-selected preset from\n"
            "~/.firefly/presets/.  Built-in presets are re-seeded on the\n"
            "next launch unless you save your own version with the same\n"
            "name first.")
        self.btn_preset_delete.clicked.connect(self._on_preset_delete)
        preset_row.addWidget(self.btn_preset_delete)
        layout.addLayout(preset_row)
        # Deferred wiring — combobox change must apply only after construction
        # of every widget the preset references.  See `_finalise_presets`.
        QtCore.QTimer.singleShot(0, self._finalise_presets)

        # NOTE: Input/output pickers used to live here but moved to the
        # Import tab in v2.1 — see `_build_import_tab`.  The QLineEdit
        # widgets `self.e_file` and `self.e_outdir` are still owned by
        # this MainWindow (created in the Import tab), so the rest of
        # the worker code that reads them keeps working unchanged.

        # ── Imaging metadata ──────────────────────────────────────────────
        # File-embedded values are used by default; checkbox enables manual
        # override.  Matches the Tk app's behaviour.
        sec, gl = self._make_form_section("Imaging metadata")

        row = QtWidgets.QHBoxLayout()
        self.c_override_px = QtWidgets.QCheckBox("Override")
        self.c_override_px.setToolTip(
            "If unchecked, the pixel size from the file's metadata is used.\n"
            "Check this only if the metadata is missing or wrong.")
        self.s_pixel_size  = self._spin_dbl(0.106, 0.01, 1.0, 0.001, decimals=3,
            tip="Physical pixel size in µm. Used to convert px → µm for D, MSD, etc.")
        row.addWidget(self.c_override_px); row.addWidget(self.s_pixel_size, 1)
        wpx = QtWidgets.QWidget(); wpx.setLayout(row)
        gl.addRow("Pixel size (µm)", wpx)

        row = QtWidgets.QHBoxLayout()
        self.c_override_fi = QtWidgets.QCheckBox("Override")
        self.c_override_fi.setToolTip(
            "If unchecked, the frame interval from the file's metadata is used.")
        self.s_frame_interval = self._spin_dbl(0.02, 0.001, 10.0, 0.001, decimals=3,
            tip="Time between frames in seconds. Used for diffusion coefficient units.")
        row.addWidget(self.c_override_fi); row.addWidget(self.s_frame_interval, 1)
        wfi = QtWidgets.QWidget(); wfi.setLayout(row)
        gl.addRow("Frame interval (s)", wfi)

        self.s_channel = self._spin_int(0, 0, 8,
            tip="Channel index to load (CZI files only). Most single-channel data uses 0.")
        gl.addRow("Channel (CZI)", self.s_channel)
        layout.addWidget(sec)

        # ── Preprocessing ─────────────────────────────────────────────────
        sec, gl = self._make_form_section("Preprocessing")
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.c_bg_method = _QuietComboBox()
        self.c_bg_method.addItems(["Uniform Filter", "Rolling Ball"])
        self.c_bg_method.setToolTip(
            "Method for subtracting local background before detection.\n"
            "• Uniform Filter — fast box-mean subtraction. Good default.\n"
            "• Rolling Ball — slower but better on uneven illumination.")
        gl.addRow("Background method", self.c_bg_method)
        self.s_bg_radius = self._spin_int(10, 3, 200,
            tip="Radius (px) of the local-mean window for background subtraction.\n"
                "Use ~3× spot diameter for diffraction-limited spots.")
        gl.addRow("Background radius (px)", self.s_bg_radius)
        layout.addWidget(sec)

        # ── Detection ─────────────────────────────────────────────────────
        sec, gl = self._make_form_section("Detection")
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.s_diameter = self._spin_int(7, 3, 21, step=2,
            tip="Expected spot diameter in pixels. Must be ODD (the GUI enforces this).\n"
                "Use ~2× the diffraction-limited PSF FWHM. Too small misses spots; "
                "too big merges adjacent ones.")
        gl.addRow("Diameter (px, odd)", self.s_diameter)

        # Auto minmass: when checked, the threshold is computed PER FILE from
        # the candidate spot-mass distribution (a 2-component Gaussian mixture
        # that splits the dim-noise and bright-real-spot populations, with a
        # count-knee cross-check) — the automated version of "set minmass at
        # the valley of the mass histogram".  Default ON: because every frame
        # is min-max normalised, a single manual threshold doesn't transfer
        # between files, so per-file auto is the right default.
        self.c_auto_minmass = QtWidgets.QCheckBox("Auto (per-file)")
        self.c_auto_minmass.setToolTip(
            "Recommended.  Computes the detection threshold separately for each\n"
            "file from its own data: it harvests every candidate spot, splits\n"
            "the dim-noise and bright-signal populations by integrated mass\n"
            "(Gaussian-mixture valley + count-knee), and sets the cutoff in the\n"
            "gap.  Robust across .czi/.tif with different brightness — no need\n"
            "to eyeball a value per dataset.\n\n"
            "An audit histogram ({stem}_minmass_hist.png) is saved per file so\n"
            "you can check the chosen cutoff.  Use the Strict/Balanced/Lenient\n"
            "selector to bias it globally; untick to enter a value by hand.")
        self.c_auto_minmass.setChecked(True)
        self.s_minmass = self._spin_dbl(1.0, 0.0, 100.0, 0.05, decimals=2,
            tip="Detection threshold — minimum integrated intensity\n"
                "(trackpy 'mass') for a spot to be kept.\n\n"
                "Too low → many false-positive spots, slow linking, garbage tracks.\n"
                "Too high → real spots filtered out.\n\n"
                "After preprocessing (background subtract + per-frame\n"
                "normalise to [0,1]) values typically land in the 0.5–50\n"
                "range.  Start near 1.0 and sweep the slider — dim points\n"
                "vanish first as you raise it.\n\n"
                "Equivalent role to PALM-Tracer's 'Threshold' field, but the\n"
                "unit is integrated raw intensity here, not k-σ on the\n"
                "wavelet plane — values don't transfer directly between tools.")
        # Slider companion — QSlider is integer-only, and a plain linear
        # mapping over 0..5000 wastes 99% of slider travel on values above
        # the useful range.  Use a square law: minmass = (slider/1000)² × MAX
        # so slider≈100 ↔ minmass 50, slider≈316 ↔ minmass 500.
        _MM_SLD_MAX = 1000
        _MM_VAL_MAX = float(self.s_minmass.maximum())
        def _slider_to_mass(s: int) -> float:
            t = max(0, min(_MM_SLD_MAX, int(s))) / _MM_SLD_MAX
            return float(t * t * _MM_VAL_MAX)
        def _mass_to_slider(m: float) -> int:
            import math as _math
            t = max(0.0, min(_MM_VAL_MAX, float(m))) / _MM_VAL_MAX
            return int(round(_math.sqrt(t) * _MM_SLD_MAX))
        self.sld_minmass = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.sld_minmass.setMinimum(0)
        self.sld_minmass.setMaximum(_MM_SLD_MAX)
        self.sld_minmass.setSingleStep(1)
        self.sld_minmass.setPageStep(20)
        self.sld_minmass.setValue(_mass_to_slider(self.s_minmass.value()))
        self.sld_minmass.setToolTip(
            "Drag to sweep the detection threshold (square-law: fine at the\n"
            "low end, coarse at the high end).  The preview viewer updates\n"
            "spot overlays as you move.  Type into the spinbox for exact values.")
        self._minmass_sync_guard = False
        def _on_slider(v: int):
            if self._minmass_sync_guard: return
            self._minmass_sync_guard = True
            try: self.s_minmass.setValue(_slider_to_mass(v))
            finally: self._minmass_sync_guard = False
        def _on_spin(v: float):
            if self._minmass_sync_guard: return
            self._minmass_sync_guard = True
            try: self.sld_minmass.setValue(_mass_to_slider(v))
            finally: self._minmass_sync_guard = False
        self.sld_minmass.valueChanged.connect(_on_slider)
        self.s_minmass.valueChanged.connect(_on_spin)

        # Sensitivity for the per-file auto threshold: nudges the computed
        # noise/signal boundary by ±1σ.  Balanced = the boundary itself.
        self.c_minmass_sensitivity = QtWidgets.QComboBox()
        self.c_minmass_sensitivity.addItems(["Strict", "Balanced", "Lenient"])
        self.c_minmass_sensitivity.setCurrentText("Balanced")
        self.c_minmass_sensitivity.setToolTip(
            "Bias the auto threshold without per-file eyeballing:\n"
            "• Strict   — higher cutoff, fewer but cleaner spots\n"
            "• Balanced — the computed noise/signal boundary (recommended)\n"
            "• Lenient  — lower cutoff, keeps more (dimmer) spots")

        def _on_auto_toggled(checked):
            self.s_minmass.setEnabled(not checked)
            self.sld_minmass.setEnabled(not checked)
            self.c_minmass_sensitivity.setEnabled(checked)
        self.c_auto_minmass.toggled.connect(_on_auto_toggled)

        wmm = QtWidgets.QWidget()
        vmm = QtWidgets.QVBoxLayout(wmm)
        vmm.setContentsMargins(0, 0, 0, 0)
        vmm.setSpacing(4)
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.c_auto_minmass)
        row.addWidget(self.s_minmass, 1)
        vmm.addLayout(row)
        vmm.addWidget(self.sld_minmass)
        srow = QtWidgets.QHBoxLayout()
        srow.setContentsMargins(0, 0, 0, 0)
        srow.addWidget(QtWidgets.QLabel("Sensitivity"))
        srow.addWidget(self.c_minmass_sensitivity, 1)
        vmm.addLayout(srow)
        gl.addRow("Threshold", wmm)

        # Apply the initial enabled/disabled state (auto is on by default).
        _on_auto_toggled(self.c_auto_minmass.isChecked())

        # Push spinbox / combo edits into the live preview.  Background
        # widgets are wired here too because the preview re-preprocesses
        # frames using these settings to match the pipeline's mass scale.
        self.s_diameter.valueChanged.connect(
            lambda _=None: self._push_detection_preview_params())
        self.s_minmass.valueChanged.connect(
            lambda _=None: self._push_detection_preview_params())
        self.c_bg_method.currentTextChanged.connect(
            lambda _=None: self._push_detection_preview_params())
        self.s_bg_radius.valueChanged.connect(
            lambda _=None: self._push_detection_preview_params())

        layout.addWidget(sec)

        # ── Linking ───────────────────────────────────────────────────────
        # FIREFLY uses trackpy's recursive subnet linker exclusively.
        sec, gl = self._make_form_section("Linking")
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.s_search_range = self._spin_int(5, 1, 30,
            tip="Maximum pixel distance a particle can move between consecutive\n"
                "frames. Calibrate from your data: bigger search_range tolerates\n"
                "fast motion but increases linker subnetwork-explosion risk.")
        gl.addRow("Search range (px)", self.s_search_range)
        self.s_memory = self._spin_int(3, 0, 10,
            tip="Number of frames a track can disappear and still be re-linked.\n"
                "0 = strict (no gaps). 3 is typical for blinking PALM probes.")
        gl.addRow("Memory (frames)", self.s_memory)
        self.s_min_track_len = self._spin_int(8, 3, 50,
            tip="Tracks shorter than this are discarded. 8 is the de-facto minimum\n"
                "for reliable MSD fits.")
        gl.addRow("Min track length", self.s_min_track_len)
        self.s_max_track_len = self._spin_int(0, 0, 100000,
            tip="0 = disabled. If set, drops tracks longer than this. Useful for\n"
                "removing stuck/aggregated particles that masquerade as long tracks.")
        gl.addRow("Max track length (0 = off)", self.s_max_track_len)
        layout.addWidget(sec)

        # ── Diffusion fit + motion classification ─────────────────────────
        sec, gl = self._make_form_section("Diffusion & motion classification")
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.s_max_lagtime = self._spin_int(20, 5, 100,
            tip="Maximum lag-time (in frames) used in the MSD curve.")
        _row = QtWidgets.QHBoxLayout(); _row.setContentsMargins(0, 0, 0, 0)
        self.lbl_max_lag_sec = QtWidgets.QLabel()
        self.lbl_max_lag_sec.setStyleSheet("color: gray;")
        _row.addWidget(self.s_max_lagtime, 1); _row.addWidget(self.lbl_max_lag_sec)
        _w = QtWidgets.QWidget(); _w.setLayout(_row)
        gl.addRow("Max lag time (frames)", _w)

        self.s_n_fit = self._spin_int(5, 2, 20,
            tip="Number of initial lag times used to fit D and α.\n"
                "Fewer = more local (short-time D); more = more global.\n"
                "Tip: dial this until the seconds readout matches your lab's MSD\n"
                "fit window (e.g. 0.2 s).")
        _row2 = QtWidgets.QHBoxLayout(); _row2.setContentsMargins(0, 0, 0, 0)
        self.lbl_n_fit_sec = QtWidgets.QLabel()
        self.lbl_n_fit_sec.setStyleSheet("color: gray;")
        _row2.addWidget(self.s_n_fit, 1); _row2.addWidget(self.lbl_n_fit_sec)
        _w2 = QtWidgets.QWidget(); _w2.setLayout(_row2)
        gl.addRow("N fit lags (frames)", _w2)

        # Live seconds readout: lag_frames × frame_interval.  The analysis works
        # in frames, but labs usually express the MSD fit window in seconds
        # (e.g. "cap at 0.2 s").  Showing the equivalent live — and updating it
        # when either the lag count OR the Frame-interval field changes — lets
        # the user match their convention without doing the conversion by hand.
        # (Uses the Frame-interval field's value; when that's read from file
        # metadata the readout reflects the field shown above it.)
        def _update_lag_seconds(*_):
            fi = float(self.s_frame_interval.value())
            self.lbl_max_lag_sec.setText(
                f"= {self.s_max_lagtime.value() * fi:.3f} s")
            self.lbl_n_fit_sec.setText(
                f"= {self.s_n_fit.value() * fi:.3f} s")
        self._update_lag_seconds = _update_lag_seconds
        self.s_max_lagtime.valueChanged.connect(_update_lag_seconds)
        self.s_n_fit.valueChanged.connect(_update_lag_seconds)
        self.s_frame_interval.valueChanged.connect(_update_lag_seconds)
        _update_lag_seconds()
        self.s_alpha_immobile = self._spin_dbl(0.5, 0.0, 2.0, 0.01, decimals=2,
            tip="α below this → 'Immobile'. Default 0.5 from the SPT literature.")
        gl.addRow("α  immobile threshold", self.s_alpha_immobile)
        self.s_alpha_confined = self._spin_dbl(0.9, 0.0, 2.0, 0.01, decimals=2,
            tip="α between immobile and this → 'Confined'.")
        gl.addRow("α  confined threshold", self.s_alpha_confined)
        self.s_alpha_directed = self._spin_dbl(1.1, 0.0, 2.0, 0.01, decimals=2,
            tip="α above this → 'Directed'. Between confined and directed → 'Brownian'.")
        gl.addRow("α  directed threshold", self.s_alpha_directed)
        self.s_mobile_d_threshold = self._spin_dbl(0.05, 0.0, 10.0, 0.01, decimals=3,
            tip="Diffusion coefficient threshold separating 'mobile' from\n"
                "'immobile' tracks for the mobile-fraction-over-time panel.")
        gl.addRow("Mobile D threshold (µm²/s)", self.s_mobile_d_threshold)
        self.s_jdd_components = self._spin_int(2, 1, 4,
            tip="Number of exponential components in the Jump Distance Distribution\n"
                "fit. 2 is typical (mobile + immobile populations).")
        gl.addRow("JDD components", self.s_jdd_components)

        # Filter-by-D toggle + range
        self.c_filter_d_enabled = QtWidgets.QCheckBox("Filter tracks by D")
        self.c_filter_d_enabled.setToolTip(
            "When checked, drop tracks with D outside the [min, max] range.\n"
            "Useful for isolating a specific population for downstream analysis.")
        gl.addRow(self.c_filter_d_enabled)
        self.s_filter_d_min = self._spin_dbl(0.0, 0.0, 10.0, 0.000001, decimals=7,
            tip="Minimum D (µm²/s). Tracks slower than this are excluded.\n"
                "7-decimal precision (down to 0.0000001) — sptPALM data is often\n"
                "near-immobile (D ~1e-6 to 1e-3), so set fine thresholds here.")
        self.s_filter_d_max = self._spin_dbl(1.0, 0.0, 10.0, 0.000001, decimals=7,
            tip="Maximum D (µm²/s). Tracks faster than this are excluded.")
        self.s_filter_d_min.setEnabled(False)
        self.s_filter_d_max.setEnabled(False)
        self.c_filter_d_enabled.toggled.connect(
            lambda checked: (self.s_filter_d_min.setEnabled(checked),
                              self.s_filter_d_max.setEnabled(checked)))
        gl.addRow("  D min (µm²/s)", self.s_filter_d_min)
        gl.addRow("  D max (µm²/s)", self.s_filter_d_max)
        layout.addWidget(sec)

        # ── ROI ───────────────────────────────────────────────────────────
        sec, gl = self._make_form_section("ROI")
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.c_roi_mode = _QuietComboBox()
        self.c_roi_mode.addItems(
            ["None", "Auto threshold", "Manual threshold",
             "Manual polygon", "From sister TIFF"])
        self.c_roi_mode.setCurrentText("Auto threshold")
        self.c_roi_mode.setToolTip(
            "Restrict analysis to a region of interest in the field of view.\n"
            "• None — analyse the whole image.\n"
            "• Auto threshold — pick a threshold from the chosen projection.\n"
            "• Manual threshold — use the value below.\n"
            "• Manual polygon — draw a polygon per file on the Import tab\n"
            "  (Set ROI… buttons).  Files without a saved polygon fall back\n"
            "  to the global Auto-threshold behaviour.\n"
            "• From sister TIFF — use a microscope-exported ROI image\n"
            "  saved next to the data as `<base><suffix>.tif`.  Suffix\n"
            "  defaults to `_green` (palmTRACER / Zeiss convention).\n"
            "  Auto-thresholded with Li if it's a fluorescence channel,\n"
            "  or non-zero pixels if it's a binary segmentation mask.\n"
            "  Multi-frame ROIs are max-projected.")
        gl.addRow("Mode", self.c_roi_mode)
        self.c_roi_auto_method = _QuietComboBox()
        self.c_roi_auto_method.addItems(["Li", "Otsu", "Triangle", "Mean"])
        self.c_roi_auto_method.setToolTip(
            "Auto-thresholding method (from scikit-image).  Li is robust for\n"
            "low-contrast SMLM data; Otsu for bimodal histograms.")
        gl.addRow("Auto method", self.c_roi_auto_method)
        self.s_roi_threshold = self._spin_dbl(0.08, 0.0, 1.0, 0.005, decimals=3,
            tip="Manual threshold on the normalised projection [0, 1].\n"
                "Drag the slider below to sweep — the green mask overlay in\n"
                "the ROI viewer updates as you move.")
        # Slider companion — linear ×1000 mapping (range 0..1.000, step 0.001).
        self.sld_roi_threshold = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.sld_roi_threshold.setMinimum(0)
        self.sld_roi_threshold.setMaximum(1000)
        self.sld_roi_threshold.setSingleStep(5)
        self.sld_roi_threshold.setPageStep(50)
        self.sld_roi_threshold.setValue(int(round(self.s_roi_threshold.value() * 1000)))
        self.sld_roi_threshold.setToolTip(
            "Drag to sweep manual threshold (0.000 – 1.000).  The green mask\n"
            "in the ROI viewer redraws live, so you can see exactly which\n"
            "pixels end up inside / outside the ROI.")
        self._roi_thresh_sync_guard = False
        def _on_roi_sld(v: int):
            if self._roi_thresh_sync_guard: return
            self._roi_thresh_sync_guard = True
            try: self.s_roi_threshold.setValue(v / 1000.0)
            finally: self._roi_thresh_sync_guard = False
        def _on_roi_spin(v: float):
            if self._roi_thresh_sync_guard: return
            self._roi_thresh_sync_guard = True
            try: self.sld_roi_threshold.setValue(int(round(v * 1000)))
            finally: self._roi_thresh_sync_guard = False
        self.sld_roi_threshold.valueChanged.connect(_on_roi_sld)
        self.s_roi_threshold.valueChanged.connect(_on_roi_spin)

        wrt = QtWidgets.QWidget()
        vrt = QtWidgets.QVBoxLayout(wrt)
        vrt.setContentsMargins(0, 0, 0, 0)
        vrt.setSpacing(4)
        vrt.addWidget(self.s_roi_threshold)
        vrt.addWidget(self.sld_roi_threshold)
        gl.addRow("Manual threshold", wrt)
        self.c_roi_mask_mode = _QuietComboBox()
        self.c_roi_mask_mode.addItems(["Max", "Blink density", "Mean", "Sum"])
        self.c_roi_mask_mode.setCurrentText("Max")
        self.c_roi_mask_mode.setToolTip(
            "Projection used to compute the ROI threshold mask.\n"
            "• Max — brightest value each pixel ever reached.  Best default\n"
            "  for sptPALM: cells light up clearly, background stays dim.\n"
            "• Blink density — count of frames where each pixel exceeds\n"
            "  its temporal median + 3·MAD.  Most discriminative: cells\n"
            "  blink repeatedly, autofluorescent background does not.\n"
            "• Mean — average intensity per pixel.  Poor on sptPALM data\n"
            "  because steady autofluorescence outweighs sparse blinks.\n"
            "• Sum — same shape as Mean, kept for backward compatibility.\n"
            "NB: the analysis backend currently uses Mean regardless —\n"
            "this control is for choosing where to draw the ROI mask\n"
            "in the preview viewer.")
        gl.addRow("Projection for ROI", self.c_roi_mask_mode)

        # Background-suppression scale (DoG σ_bg).  Subtracts a heavily-
        # blurred copy of the projection from a lightly-blurred copy so
        # slow autofluorescent gradients fall away before thresholding.
        # The right value scales with cell size: anything larger than
        # the cell radius works; anything smaller eats into the cell.
        self.s_roi_bg_sigma = self._spin_dbl(25.0, 0.0, 100.0, 1.0, decimals=1,
            tip="Background-suppression scale σ_bg (pixels).\n"
                "Before thresholding, a heavily-blurred copy of the projection\n"
                "(this σ) is subtracted from a lightly-blurred copy — slow\n"
                "autofluorescent illumination falls away, leaving only cell-\n"
                "scale blink texture.  Pick a value LARGER than the cells\n"
                "you want to keep:\n"
                "  • 0    — disable (no background suppression)\n"
                "  • 15   — small cells / very tight masks\n"
                "  • 25   — typical mammalian cells (default, ~2.6 µm @ 0.106 µm/px)\n"
                "  • 40+  — large cells or wide processes\n"
                "If your cell ROI looks eroded in the middle, raise this.\n"
                "If background is still bleeding in, lower it.")
        # Companion slider — 0..100 px, integer steps (drag-to-sweep).
        self.sld_roi_bg_sigma = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.sld_roi_bg_sigma.setMinimum(0)
        self.sld_roi_bg_sigma.setMaximum(100)
        self.sld_roi_bg_sigma.setSingleStep(1)
        self.sld_roi_bg_sigma.setPageStep(5)
        self.sld_roi_bg_sigma.setValue(int(round(self.s_roi_bg_sigma.value())))
        self.sld_roi_bg_sigma.setToolTip(
            "Drag to sweep background-suppression σ_bg (0–100 px).  Watch\n"
            "the green mask reshape: too small eats into cells, too large\n"
            "lets diffuse background back in.")
        self._roi_bg_sigma_sync_guard = False
        def _on_bg_sld(v: int):
            if self._roi_bg_sigma_sync_guard: return
            self._roi_bg_sigma_sync_guard = True
            try: self.s_roi_bg_sigma.setValue(float(v))
            finally: self._roi_bg_sigma_sync_guard = False
        def _on_bg_spin(v: float):
            if self._roi_bg_sigma_sync_guard: return
            self._roi_bg_sigma_sync_guard = True
            try: self.sld_roi_bg_sigma.setValue(int(round(v)))
            finally: self._roi_bg_sigma_sync_guard = False
        self.sld_roi_bg_sigma.valueChanged.connect(_on_bg_sld)
        self.s_roi_bg_sigma.valueChanged.connect(_on_bg_spin)
        wbg = QtWidgets.QWidget()
        vbg = QtWidgets.QVBoxLayout(wbg)
        vbg.setContentsMargins(0, 0, 0, 0)
        vbg.setSpacing(4)
        vbg.addWidget(self.s_roi_bg_sigma)
        vbg.addWidget(self.sld_roi_bg_sigma)
        gl.addRow("Background scale σ", wbg)

        # Auto-pair a sibling ImageJ ROI (RoiSet.zip / a RoiSet/ folder / .roi)
        # found next to each input file, applying it as a polygon ROI — so a
        # batch reuses ROIs drawn in ImageJ/Fiji without loading each by hand.
        self.c_roi_imagej_auto = QtWidgets.QCheckBox(
            "Auto-detect ImageJ ROI")
        self.c_roi_imagej_auto.setChecked(True)
        self.c_roi_imagej_auto.setToolTip(
            "When a movie has an ImageJ ROI next to it (RoiSet.zip, a RoiSet/\n"
            "folder of .roi files, or <name>.roi / <name>.zip), use it as a\n"
            "polygon ROI automatically.  Lets a batch reuse ROIs drawn in\n"
            "ImageJ/Fiji without loading each one by hand.  A polygon you set\n"
            "explicitly in the ROI editor still takes precedence.")
        # Span the whole row (no label column) so the checkbox doesn't force
        # the sidebar's scroll content wider than the viewport.
        gl.addRow(self.c_roi_imagej_auto)

        # Grey out threshold-related controls when the mode doesn't use
        # them, AND show/hide the embedded ROI viewer on the Import tab
        # when "Manual polygon" is selected.
        self.c_roi_mode.currentTextChanged.connect(self._on_roi_mode_changed)
        # Push ROI mask updates to the embedded viewer whenever the user
        # changes any of these knobs.
        self.c_roi_auto_method.currentTextChanged.connect(
            lambda _=None: self._push_roi_mask_params())
        self.s_roi_threshold.valueChanged.connect(
            lambda _=None: self._push_roi_mask_params())
        self.c_roi_mask_mode.currentTextChanged.connect(
            lambda _=None: self._push_roi_mask_params())
        self.s_roi_bg_sigma.valueChanged.connect(
            lambda _=None: self._push_roi_mask_params())
        # When ImageJ ROI auto-detect is on, a sibling RoiSet/.roi overrides the
        # Mode below — grey it (and its sub-controls) out to make that clear.
        self.c_roi_imagej_auto.toggled.connect(self._on_roi_imagej_auto_toggled)
        self._on_roi_imagej_auto_toggled(self.c_roi_imagej_auto.isChecked())
        layout.addWidget(sec)

        # ── Drift correction ──────────────────────────────────────────────
        sec, gl = self._make_form_section("Drift correction")
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.c_drift_correct = QtWidgets.QCheckBox("Apply RCC drift correction")
        self.c_drift_correct.setToolTip(
            "Redundant Cross-Correlation (RCC) drift correction: estimates the\n"
            "sample drift over time by all-pairs cross-correlation between\n"
            "segments, then subtracts it from every localisation.\n"
            "Strongly recommended for sptPALM movies > 1 minute long.")
        self.c_drift_correct.setChecked(True)
        gl.addRow(self.c_drift_correct)
        self.s_drift_segment = self._spin_int(500, 50, 5000, step=50,
            tip="Frames per RCC segment. Smaller = finer drift tracking but\n"
                "noisier. 500 is a reasonable default for 4000+ frame movies.")
        gl.addRow("Segment size (frames)", self.s_drift_segment)
        layout.addWidget(sec)

        # ── Clustering ────────────────────────────────────────────────────
        sec, gl = self._make_form_section("Clustering (DBSCAN)")
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.s_cluster_eps_nm = self._spin_dbl(50.0, 5.0, 1000.0, 5.0, decimals=1,
            tip="DBSCAN neighbourhood radius (nm). Two localisations are in the\n"
                "same cluster if they're within this distance.")
        gl.addRow("eps (nm)", self.s_cluster_eps_nm)
        self.s_cluster_min_samples = self._spin_int(10, 2, 100,
            tip="Minimum localisations to form a DBSCAN cluster. Lower = more\n"
                "clusters detected but noisier; higher = stricter.")
        gl.addRow("min samples", self.s_cluster_min_samples)
        layout.addWidget(sec)

        # ── Performance ───────────────────────────────────────────────────
        sec, gl = self._make_form_section(f"Performance  —  {N_CPUS} cores")
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.c_backend = _QuietComboBox()
        self.c_backend.addItems(self._available_backends())
        # Default to the PyTorch engine with the device auto-selected.  The
        # "torch" value routes through TorchBackend.select_device(), which
        # sanity-checks the GPU and transparently falls back to CPU on cards
        # the bundled CUDA build can't run (e.g. a Pascal GTX 1060).  A saved
        # user preference, if present, overrides this on settings restore.
        self.c_backend.setCurrentText("Torch (auto)")
        self.c_backend.setToolTip(
            "Which implementation to use for spot localisation.\n"
            "• Auto                — pick the fastest healthy backend on this machine.\n"
            "• Trackpy (CPU)       — reference CPU implementation (battle-tested).\n"
            "• Torch (auto)        — PyTorch, device auto-selected.\n"
            "• Torch — Apple MPS   — force Apple GPU.  Fast when stable; on some\n"
            "                        macOS/M-chip combinations may hit memory-\n"
            "                        allocator issues at very low minmass.\n"
            "• Torch — NVIDIA CUDA — force NVIDIA GPU.\n"
            "• Torch — CPU         — force PyTorch on CPU (for benchmarking).")
        gl.addRow("Detection backend", self.c_backend)
        self.s_workers = self._spin_int(N_CPUS, 1, N_CPUS,
            tip="Parallel CPU workers for the trackpy backend's multiprocessing\n"
                "pool and the MSD fitting thread pool.  Default = all cores.")
        gl.addRow(f"CPU workers (max {N_CPUS})", self.s_workers)
        self.s_chunk_size = self._spin_int(500, 50, 5000, step=100,
            tip="Frames per processing chunk. Bigger = less per-chunk overhead\n"
                "(esp. on GPU) but more RAM. 500 is balanced; tune up if your\n"
                "stack and free RAM are large.")
        gl.addRow("Chunk size (frames)", self.s_chunk_size)

        # GPU-acceleration entry point — Windows only.  When CUDA is NOT
        # installed we show the Set-up button right here, where the user picks
        # the CUDA backend.  Once installed it's hidden (install / uninstall /
        # relocate live under Settings › GPU acceleration).
        self._cuda_btn = QtWidgets.QPushButton("Set up GPU acceleration…")
        self._cuda_btn.setToolTip(
            "Download the CUDA build of PyTorch (~2.5 GB) so FIREFLY can use\n"
            "your NVIDIA GPU for ~5–10× faster localisation.\n"
            "Manage it later under Settings › GPU acceleration.")
        self._cuda_btn.clicked.connect(self._on_cuda_button_clicked)
        gl.addRow("", self._cuda_btn)
        self._refresh_cuda_perf_ui()

        layout.addWidget(sec)

        layout.addStretch(1)

    def _refresh_cuda_perf_ui(self):
        """Performance-section GPU control state.  Show the Set-up button only
        when CUDA isn't installed; once it is, hide it (install / uninstall /
        relocate live under Settings › GPU acceleration).  Windows only — the
        button stays hidden elsewhere.  Safe to call repeatedly (after install
        / uninstall / relocate)."""
        if not hasattr(self, "_cuda_btn"):
            return
        if sys.platform != "win32":
            self._cuda_btn.setVisible(False)
            return
        try:
            from firefly import cuda_installer as _cu
            installed = _cu.is_installed()
        except Exception:
            installed = False
        self._cuda_btn.setVisible(not installed)

    def _build_landing_page(self) -> QtWidgets.QWidget:
        """Full-window welcome screen, Minecraft-menu style.

        Four full-width primary actions stacked vertically (the "Play"
        block in Minecraft's main menu), then a single row beneath with
        Settings + Quit (the "Options / Quit Game" pair).  Once the
        user picks an action, the QStackedWidget swaps to the main
        sidebar+tabs UI and there's no way back this session.
        """
        page = QtWidgets.QWidget()
        page.setObjectName("landing_page")

        # Horizontal centring so the menu column has a fixed maximum
        # width regardless of how wide the window is.
        wrap = QtWidgets.QHBoxLayout(page)
        wrap.setContentsMargins(40, 24, 40, 24)
        wrap.addStretch(1)

        column = QtWidgets.QWidget()
        column.setMaximumWidth(640)
        column.setMinimumWidth(480)
        outer = QtWidgets.QVBoxLayout(column)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)
        outer.addStretch(1)

        # Hero: "Welcome to" line above the big FIREFLY logo, with the
        # tagline underneath.
        welcome = QtWidgets.QLabel("Welcome to")
        welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']}; font-size: 18px; "
            f"font-weight: 500; letter-spacing: 2px;")
        outer.addWidget(welcome)
        title = QtWidgets.QLabel(
            f"<span style='color:{_THEME['ACC']};letter-spacing:6px;'>"
            f"FIREFLY</span>")
        title.setTextFormat(Qt.TextFormat.RichText)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"color: {_THEME['TXT']}; font-size: 48px; font-weight: 800;")
        outer.addWidget(title)
        sub = QtWidgets.QLabel(
            "Fluorescence Inference & Reconstruction Engine")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color: {_THEME['TXT_MUTED']}; font-size: 13px;")
        outer.addWidget(sub)
        outer.addSpacing(20)

        def _go(target_tab: str, *, batch: bool | None = None):
            def _fn():
                if batch is True:
                    try: self.r_mode_batch.setChecked(True)
                    except AttributeError: pass
                elif batch is False:
                    try: self.r_mode_single.setChecked(True)
                    except AttributeError: pass
                self._enter_main_ui(target_tab)
            return _fn

        def _menu_tile(label: str, description: str, slot, *,
                       big: bool = True) -> QtWidgets.QFrame:
            """Build one landing-menu tile styled to match the program's
            existing `_ModeTile` look — rounded PANEL fill, accent
            border on hover, centred bold title with a muted subtitle.

            `big=True`  for primary actions (taller, larger font).
            `big=False` for the Settings / Quit row.
            """
            tile = QtWidgets.QFrame()
            tile.setObjectName("mode_tile")  # picks up QSS rules
            tile.setCursor(Qt.CursorShape.PointingHandCursor)
            tile.setProperty("checked", False)
            tile.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                               QtWidgets.QSizePolicy.Policy.Fixed)
            tile.setMinimumHeight(82 if big else 60)

            v = QtWidgets.QVBoxLayout(tile)
            v.setContentsMargins(20, 12, 20, 12)
            v.setSpacing(3)
            lbl = QtWidgets.QLabel(label)
            lbl.setObjectName("mode_tile_title")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            f = lbl.font(); f.setBold(True)
            f.setPointSize(18 if big else 14)
            lbl.setFont(f)
            v.addWidget(lbl)
            if description:
                desc_lbl = QtWidgets.QLabel(description)
                desc_lbl.setObjectName("mode_tile_subtitle")
                desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                desc_lbl.setWordWrap(True)
                v.addWidget(desc_lbl)

            # mousePressEvent-style click — tile is a QFrame so we wire
            # the click ourselves rather than relying on QAbstractButton.
            def _press(_event, _slot=slot):
                if _event.button() == Qt.MouseButton.LeftButton:
                    _slot()
            tile.mousePressEvent = _press   # type: ignore[assignment]
            return tile

        # ── Primary actions (stacked vertically) ─────────────────────────
        primary_actions = [
            ("Analysis",
             "Pick one .czi or .tif and run the full sptPALM pipeline.",
             _go(TAB_IMPORT, batch=False)),
            ("Batch",
             "Process every file in a folder, one after another, with shared settings.",
             _go(TAB_IMPORT, batch=True)),
            ("Compare",
             "Load previous analysis outputs and produce a side-by-side comparison figure.",
             _go(TAB_COMPARE)),
            ("Inspect",
             "Open a previous run in an embedded napari viewer to scrub frames and explore tracks.",
             _go(TAB_VISUALISE)),
        ]
        for label, desc, slot in primary_actions:
            outer.addWidget(_menu_tile(label, desc, slot, big=True))

        outer.addSpacing(18)

        # ── Settings + Quit row (side by side) ───────────────────────────
        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.setSpacing(12)
        bottom_row.addWidget(_menu_tile(
            "Settings", "",
            lambda: self._open_preferences(),
            big=False))
        bottom_row.addWidget(_menu_tile(
            "Quit", "",
            lambda: self.close(),
            big=False))
        outer.addLayout(bottom_row)

        outer.addStretch(2)

        wrap.addWidget(column, stretch=0)
        wrap.addStretch(1)
        return page

    def _build_remaining_sidebar_pages(self):
        """Populate sidebar pages 1–4 by re-parenting widgets that the
        tab-body builders created.  Page 0 (Import) was built in
        `__init__` already.

        Mapping (must match the tab order set by __init__):
          1 → Analysis tab     → muted info label, no controls
          2 → Compare tab      → muted info label, no controls
          3 → Visualise tab    → Load / Track filters / Cluster sections
          4 → Re-process tab   → Source-run + ROI helper
        """
        muted = f"color: {_THEME['TXT_MUTED']}; padding: 16px;"

        # Page 1 — Analysis (no settings; live progress only).
        analysis_page = QtWidgets.QWidget()
        ap_v = QtWidgets.QVBoxLayout(analysis_page)
        ap_v.setContentsMargins(0, 0, 0, 0)
        _lbl = QtWidgets.QLabel(
            "No settings for this tab.\n\nLive analysis progress, "
            "resource usage, and results appear on the right.")
        _lbl.setWordWrap(True); _lbl.setStyleSheet(muted)
        _lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        ap_v.addWidget(_lbl); ap_v.addStretch(1)
        self._sidebar_stack.addWidget(analysis_page)         # index 1

        # Page 2 — Compare: comparison settings (output folder/name) in
        # the sidebar; the group cards stay inline in the tab body
        # because they're full-width drop targets.
        compare_page = QtWidgets.QWidget()
        cp_outer = QtWidgets.QVBoxLayout(compare_page)
        cp_outer.setContentsMargins(0, 0, 0, 0); cp_outer.setSpacing(0)
        cp_scroll = QtWidgets.QScrollArea()
        cp_scroll.setWidgetResizable(True)
        cp_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        cp_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        cp_inner = QtWidgets.QWidget()
        cp_v = QtWidgets.QVBoxLayout(cp_inner)
        cp_v.setContentsMargins(12, 0, 12, 12); cp_v.setSpacing(8)
        sec_cmp = _CollapsibleSection("Comparison settings")
        sec_cmp.content_layout.addWidget(self._cmp_settings_widget)
        cp_v.addWidget(sec_cmp)
        _hint = QtWidgets.QLabel(
            "Add groups on the right, then click Generate comparison.")
        _hint.setWordWrap(True)
        _hint.setStyleSheet(f"color: {_THEME['TXT_MUTED']}; padding: 4px;")
        cp_v.addWidget(_hint)
        cp_v.addStretch(1)
        cp_scroll.setWidget(cp_inner)
        cp_outer.addWidget(cp_scroll)
        self._sidebar_stack.addWidget(compare_page)          # index 2

        # Page 3 — Visualise (re-parents load/filter/DBSCAN widgets).
        vis_page = QtWidgets.QWidget()
        vp_outer = QtWidgets.QVBoxLayout(vis_page)
        vp_outer.setContentsMargins(0, 0, 0, 0); vp_outer.setSpacing(0)
        vp_scroll = QtWidgets.QScrollArea()
        vp_scroll.setWidgetResizable(True)
        vp_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        vp_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        vp_inner = QtWidgets.QWidget()
        vp_v = QtWidgets.QVBoxLayout(vp_inner)
        vp_v.setContentsMargins(12, 0, 12, 12); vp_v.setSpacing(8)
        sec_load = _CollapsibleSection("Load")
        sec_load.content_layout.addWidget(self._vis_load_widget)
        vp_v.addWidget(sec_load)
        sec_filt = _CollapsibleSection("Track filters")
        sec_filt.content_layout.addWidget(self._vis_filter_widget)
        vp_v.addWidget(sec_filt)
        sec_clu  = _CollapsibleSection("Cluster (DBSCAN)")
        sec_clu.content_layout.addWidget(self._vis_dbscan_widget)
        vp_v.addWidget(sec_clu)
        vp_v.addStretch(1)
        vp_scroll.setWidget(vp_inner)
        vp_outer.addWidget(vp_scroll)
        self._sidebar_stack.addWidget(vis_page)              # index 3

        # Page 4 — Re-process (re-parents source picker).
        pp_page = QtWidgets.QWidget()
        pp_outer = QtWidgets.QVBoxLayout(pp_page)
        pp_outer.setContentsMargins(0, 0, 0, 0); pp_outer.setSpacing(0)
        pp_scroll = QtWidgets.QScrollArea()
        pp_scroll.setWidgetResizable(True)
        pp_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        pp_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        pp_inner = QtWidgets.QWidget()
        pp_v = QtWidgets.QVBoxLayout(pp_inner)
        pp_v.setContentsMargins(12, 0, 12, 12); pp_v.setSpacing(8)
        sec_src = _CollapsibleSection("Source run")
        sec_src.content_layout.addWidget(self._pp_source_widget)
        pp_v.addWidget(sec_src)
        # Re-parent the ROI viewer's toolbar controls (Filtered view /
        # Clear polygons / Load ROI…) out of the viewer's inline
        # toolbar into the sidebar so the tab body is just the viewer
        # canvas with no header chrome.
        sec_roi = _CollapsibleSection("ROI controls")
        roi_w = QtWidgets.QWidget()
        roi_v = QtWidgets.QVBoxLayout(roi_w)
        roi_v.setContentsMargins(0, 0, 0, 0); roi_v.setSpacing(6)
        _pv = self._postproc_roi_viewer
        for _btn_attr in ("_cb_filtered", "_b_clear", "_b_load_roi"):
            _w = getattr(_pv, _btn_attr, None)
            if _w is not None:
                roi_v.addWidget(_w)
        sec_roi.content_layout.addWidget(roi_w)
        pp_v.addWidget(sec_roi)
        pp_v.addStretch(1)
        pp_scroll.setWidget(pp_inner)
        pp_outer.addWidget(pp_scroll)
        self._sidebar_stack.addWidget(pp_page)               # index 4

        # ── Bottom-button pages (action stack) ────────────────────────
        # Page 1 — Analysis: no action (empty placeholder).
        self._sidebar_action.addWidget(QtWidgets.QWidget())   # index 1
        # Page 2 — Compare: re-parent the existing Generate button.
        cmp_act = QtWidgets.QWidget()
        cmp_av = QtWidgets.QVBoxLayout(cmp_act)
        cmp_av.setContentsMargins(12, 6, 12, 12)
        _btn = getattr(self, "btn_cmp_run", None)
        if _btn is not None:
            _btn.setMinimumHeight(36)
            cmp_av.addWidget(_btn)
        self._sidebar_action.addWidget(cmp_act)               # index 2
        # Page 3 — Visualise: no action.
        self._sidebar_action.addWidget(QtWidgets.QWidget())   # index 3
        # Page 4 — Re-process: the action widget built by the tab.
        self._sidebar_action.addWidget(self._pp_action_widget)  # index 4

    def _build_import_tab(self):
        """Import tab — single-source-of-truth for input/output config.

        Mode toggle switches the visible sub-panel between:
          • Single file — pick a .czi/.tif + an output folder
          • Batch       — pick a folder of files, choose which to process,
                           output goes to <folder>/batch_results/<stem>/

        The Start button (sidebar) reads from this tab and dispatches to
        the appropriate worker; after starting, the app auto-switches to
        the Analysis tab so the user sees progress immediately.
        """
        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(12)

        # ── Mode toggle ───────────────────────────────────────────────────
        # Segmented control: two big tile buttons, exclusive.  Looks like
        # a pair of cards — fills the available width and makes the choice
        # feel deliberate rather than incidental.
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setSpacing(12)

        self.r_mode_single = self._make_mode_tile(
            "Single file",
            "Analyse one .czi / .tif image — or an external\n"
            "localisations table (.csv / .txt / .tsv), auto-detected")
        self.r_mode_batch = self._make_mode_tile(
            "Batch (folder)",
            "Process every file in a folder, one after another")
        self.r_mode_single.setChecked(True)

        # Manual exclusivity (these custom tiles aren't QAbstractButtons,
        # so QButtonGroup can't manage them).  Clicking one unchecks the
        # other and fires the mode-change handler.
        def _set_mode(name: str):
            self.r_mode_single.setChecked(name == "single")
            self.r_mode_batch.setChecked(name == "batch")
            self._on_import_mode_changed(name)

        self.r_mode_single.toggled.connect(
            lambda checked: _set_mode("single") if checked else None)
        self.r_mode_batch.toggled.connect(
            lambda checked: _set_mode("batch")  if checked else None)

        mode_row.addWidget(self.r_mode_single, 1)
        mode_row.addWidget(self.r_mode_batch,  1)
        v.addLayout(mode_row)

        # ── Single-file sub-panel ─────────────────────────────────────────
        self._single_panel = QtWidgets.QGroupBox("Single file")
        sg = QtWidgets.QFormLayout(self._single_panel)
        sg.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        row = QtWidgets.QHBoxLayout()
        self.e_file = QtWidgets.QLineEdit()
        self.e_file.setPlaceholderText(
            "Browse for a .czi / .tif image or a .csv / .txt / .tsv "
            "localisations table…")
        b1 = QtWidgets.QPushButton("Browse")
        b1.clicked.connect(self._on_browse_file)
        row.addWidget(self.e_file); row.addWidget(b1)
        w_file = QtWidgets.QWidget(); w_file.setLayout(row)
        sg.addRow("Input file", w_file)

        row = QtWidgets.QHBoxLayout()
        self.e_outdir = QtWidgets.QLineEdit()
        self.e_outdir.setPlaceholderText(
            "Defaults to input file's folder.  Each run is wrapped in "
            "a subfolder named after the input stem.")
        b2 = QtWidgets.QPushButton("Browse")
        b2.clicked.connect(self._on_browse_outdir)
        row.addWidget(self.e_outdir); row.addWidget(b2)
        w_out = QtWidgets.QWidget(); w_out.setLayout(row)
        sg.addRow("Output folder", w_out)

        # Replay-from-manifest row — load a previous run's parameters
        # from its <stem>_run_manifest.json so you can reproduce it.
        row = QtWidgets.QHBoxLayout()
        self.btn_load_manifest = QtWidgets.QPushButton(
            "Load run manifest…")
        self.btn_load_manifest.setToolTip(
            "Open a previous run's <stem>_run_manifest.json and apply its\n"
            "parameters to the sidebar.  Useful for reproducing a run\n"
            "exactly or starting a new analysis from a known-good config.")
        self.btn_load_manifest.clicked.connect(self._on_load_manifest)
        row.addStretch(1)
        row.addWidget(self.btn_load_manifest)
        w_manifest = QtWidgets.QWidget(); w_manifest.setLayout(row)
        sg.addRow("", w_manifest)

        # ROI status + explicit "Load into ROI viewer" button.  The viewer
        # is the embedded _RoiViewer below — always visible, auto-loads
        # whenever the input path settles.
        self.lbl_single_roi_status = QtWidgets.QLabel(
            "ROI: using global setting")
        self.lbl_single_roi_status.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']};")
        sg.addRow("Region of interest", self.lbl_single_roi_status)

        # Localisation-table controls — revealed only when the chosen input
        # is an external .csv / .txt / .tsv table (auto-detected from the
        # path).  Hidden for image inputs.  Widget names are kept as
        # c_csv_preset / e_csv_bg because Batch mode and the run-launch code
        # read them directly.
        self._single_loc_panel = QtWidgets.QWidget()
        lg = QtWidgets.QFormLayout(self._single_loc_panel)
        lg.setContentsMargins(0, 0, 0, 0)
        lg.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self.c_csv_preset = _QuietComboBox()
        self.c_csv_preset.addItems(
            ["Auto-detect", "PALM-Tracer", "ThunderSTORM",
             "Picasso", "TrackMate", "Custom"])
        self.c_csv_preset.setToolTip(
            "Source-tool preset.  Tells FIREFLY how to interpret the\n"
            "CSV's columns (frame indexing, x/y units, mass column).\n"
            "Auto-detect sniffs the header; pick a specific preset if\n"
            "auto-detect picks the wrong one.\n\n"
            "TrackMate note: when the CSV's TRACK_ID column is present,\n"
            "FIREFLY uses TrackMate's tracks directly and SKIPS its own\n"
            "linker — the Linking sidebar params have no effect in that\n"
            "case.")
        lg.addRow("Source preset", self.c_csv_preset)

        row = QtWidgets.QHBoxLayout()
        self.e_csv_bg = QtWidgets.QLineEdit()
        self.e_csv_bg.setPlaceholderText(
            "Optional — used only for the figure's max-projection panel")
        btn_csv_bg = QtWidgets.QPushButton("Browse")
        btn_csv_bg.clicked.connect(self._on_browse_csv_bg)
        row.addWidget(self.e_csv_bg, 1); row.addWidget(btn_csv_bg)
        w_csv_bg = QtWidgets.QWidget(); w_csv_bg.setLayout(row)
        lg.addRow("Background image", w_csv_bg)

        _loc_hint = QtWidgets.QLabel(
            "Localisations table detected — detection / preprocessing are "
            "skipped; linking + downstream analyses run on the imported "
            "spots.  Set Pixel size / Frame interval in the sidebar.")
        _loc_hint.setWordWrap(True)
        _loc_hint.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']}; font-size: 11px;")
        lg.addRow("", _loc_hint)

        self._single_loc_panel.hide()
        sg.addRow(self._single_loc_panel)

        v.addWidget(self._single_panel)

        # ── Batch sub-panel ───────────────────────────────────────────────
        self._batch_panel = QtWidgets.QGroupBox("Batch")
        bg = QtWidgets.QVBoxLayout(self._batch_panel)
        bg.setSpacing(6)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Input folder"))
        self.e_batch_folder = QtWidgets.QLineEdit()
        self.e_batch_folder.setPlaceholderText(
            "Pick a folder containing .czi / .tif images or "
            ".csv / .txt localisation tables…")
        btn_pick = QtWidgets.QPushButton("Browse")
        btn_pick.clicked.connect(self._on_batch_pick_folder)
        btn_refresh = QtWidgets.QPushButton("↻ Rescan")
        btn_refresh.setToolTip("Re-scan the folder for input files.")
        btn_refresh.clicked.connect(self._on_batch_rescan)
        row.addWidget(self.e_batch_folder, 1)
        row.addWidget(btn_pick)
        row.addWidget(btn_refresh)
        bg.addLayout(row)

        bg.addWidget(QtWidgets.QLabel(
            "Series to process  (expand a series to deselect individual files):"))
        self.tree_batch_files = QtWidgets.QTreeWidget()
        self.tree_batch_files.setHeaderHidden(True)
        self.tree_batch_files.setColumnCount(1)
        self.tree_batch_files.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.tree_batch_files.setMinimumHeight(200)
        self.tree_batch_files.setRootIsDecorated(True)
        self.tree_batch_files.setUniformRowHeights(True)
        self.tree_batch_files.setIndentation(18)
        bg.addWidget(self.tree_batch_files, stretch=1)

        sel_row = QtWidgets.QHBoxLayout()
        for label, fn in (("Select all",     self._on_batch_select_all),
                          ("Select none",    self._on_batch_select_none),
                          ("Invert selection", self._on_batch_select_inverse)):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(fn)
            sel_row.addWidget(b)
        sel_row.addStretch(1)
        # Explicit "open in preview viewer" — file loading is heavy
        # (reads ~30 frames + embeds in napari) and was previously
        # triggered on every checkbox toggle via itemClicked, which froze
        # the UI when the user was rapidly de-selecting files.  Now the
        # load only fires when the user explicitly asks for it.
        self.btn_batch_open_in_viewer = QtWidgets.QPushButton(
            "Open in viewer")
        self.btn_batch_open_in_viewer.setToolTip(
            "Load the highlighted series (or file) into the preview\n"
            "viewer below.  Double-clicking a row in the tree does the\n"
            "same thing.")
        self.btn_batch_open_in_viewer.clicked.connect(
            self._on_batch_open_in_viewer)
        sel_row.addWidget(self.btn_batch_open_in_viewer)
        self.lbl_batch_summary = QtWidgets.QLabel("0 series / 0 selected")
        sel_row.addWidget(self.lbl_batch_summary)
        bg.addLayout(sel_row)

        # Power-user shortcut: double-click a row to load it without
        # using the toolbar button.  Single-click only highlights —
        # no heavy work happens on checkbox toggles.
        self.tree_batch_files.itemDoubleClicked.connect(
            self._on_batch_tree_item_double_clicked)
        # Parent ↔ child check-state propagation.  Wired in _batch_rescan
        # after population to avoid spurious fires during seeding.

        # Where the batch outputs land
        self.lbl_batch_output_path = QtWidgets.QLabel(
            "Output → (pick an input folder first)")
        self.lbl_batch_output_path.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']};")
        bg.addWidget(self.lbl_batch_output_path)

        # ── Job queue ─────────────────────────────────────────────────────
        # Stack several batch jobs, each capturing the folder + settings that
        # were active when it was added, then run them all back-to-back.
        self.btn_batch_add_queue = QtWidgets.QPushButton(
            "Add current selection to queue")
        self.btn_batch_add_queue.setToolTip(
            "Snapshot the checked files + the current settings/preset as a\n"
            "queued job.  Change the folder and/or preset and add another to\n"
            "stack a different job, then 'Run queue' to process them all in\n"
            "sequence — each with its own captured settings and output folder.")
        self.btn_batch_add_queue.clicked.connect(self._on_batch_add_to_queue)
        bg.addWidget(self.btn_batch_add_queue)

        self.lst_batch_queue = QtWidgets.QListWidget()
        self.lst_batch_queue.setMaximumHeight(90)
        self.lst_batch_queue.setToolTip("Queued jobs (run top-to-bottom).")
        bg.addWidget(self.lst_batch_queue)

        _qrow = QtWidgets.QHBoxLayout()
        self.lbl_batch_queue = QtWidgets.QLabel("Queue: 0 job(s), 0 run(s)")
        self.lbl_batch_queue.setStyleSheet(f"color: {_THEME['TXT_MUTED']};")
        _qrow.addWidget(self.lbl_batch_queue)
        _qrow.addStretch(1)
        self.btn_batch_remove_queue = QtWidgets.QPushButton("Remove")
        self.btn_batch_remove_queue.clicked.connect(self._on_batch_remove_queued)
        self.btn_batch_clear_queue = QtWidgets.QPushButton("Clear")
        self.btn_batch_clear_queue.clicked.connect(self._on_batch_clear_queue)
        self.btn_batch_run_queue = QtWidgets.QPushButton("Run queue")
        self.btn_batch_run_queue.clicked.connect(self._on_batch_run_queue)
        for _b in (self.btn_batch_remove_queue, self.btn_batch_clear_queue,
                   self.btn_batch_run_queue):
            _qrow.addWidget(_b)
        bg.addLayout(_qrow)
        self._batch_queue = []
        self._refresh_batch_queue()

        v.addWidget(self._batch_panel, stretch=1)

        # Start visible state: single mode shown, batch hidden
        self._batch_panel.hide()
        self._import_mode = "single"

        # ── Embedded ROI viewer (always visible) ──────────────────────────
        self._roi_viewer_container = QtWidgets.QFrame()
        # Reserve a min height so the panel doesn't grow from nothing the
        # first time a file is loaded — that resize is what macOS animates
        # as a "slide".  Kept modest so the Import tab stays comfortably
        # vertical-compressible on small (1366×768 / 1440×900) laptops.
        self._roi_viewer_container.setMinimumHeight(180)
        rvl = QtWidgets.QVBoxLayout(self._roi_viewer_container)
        rvl.setContentsMargins(0, 8, 0, 0)
        self._roi_viewer = _RoiViewer()
        self._roi_viewer.polygons_changed.connect(self._on_roi_polygons_changed)
        rvl.addWidget(self._roi_viewer)
        v.addWidget(self._roi_viewer_container, stretch=2)
        # Pre-init the napari viewer right after construction so the very
        # first file load doesn't have to embed napari + load data + grow
        # the layout in one step (that triple causes the macOS slide).
        QtCore.QTimer.singleShot(0, lambda: self._roi_viewer._ensure_viewer())

        self.tabs.addTab(tab, TAB_IMPORT)

    def _build_analysis_tab(self):
        """Analysis tab — pure status display.  Stage label, progress bar,
        and a results panel that fills in after a run completes."""
        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(10)

        # Stage label on the left, elapsed-time counter on the right.
        # Both updated by the polling timer (stage) and a 1 Hz elapsed
        # timer (clock).
        stage_row = QtWidgets.QHBoxLayout()
        self.run_stage_label = QtWidgets.QLabel("Idle")
        self.run_stage_label.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']}; font-weight: 600; padding: 2px 0;")
        stage_row.addWidget(self.run_stage_label, 1)
        self.lbl_elapsed = QtWidgets.QLabel("")
        self.lbl_elapsed.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']};")
        self.lbl_elapsed.setAlignment(Qt.AlignmentFlag.AlignRight)
        stage_row.addWidget(self.lbl_elapsed)
        v.addLayout(stage_row)

        # Resource monitor — CPU / RAM / GPU / VRAM at 1 Hz
        self.resource_monitor = _ResourceMonitor()
        v.addWidget(self.resource_monitor)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Ready")
        v.addWidget(self.progress_bar)

        # Mirror widgets for batch runs.  Single set of widgets — one
        # set of state per tab is excessive when Analysis is universal.
        # Use the same stage label + progress bar for batch by aliasing.
        self.batch_stage_label = self.run_stage_label
        self.batch_progress    = self.progress_bar

        # Per-file mini-progress for batch.  Shown only during a batch
        # run (sits between the overall progress bar and the results
        # panel).  Lets the user see "currently processing X" even when
        # the overall % only ticks once per file.
        self.batch_subprogress = QtWidgets.QProgressBar()
        self.batch_subprogress.setRange(0, 100)
        self.batch_subprogress.setValue(0)
        self.batch_subprogress.setTextVisible(True)
        self.batch_subprogress.setFormat("")
        self.batch_subprogress.hide()
        v.addWidget(self.batch_subprogress)

        # Detection cockpit (during a run) vs results panel (post-run)
        # share the same bottom slot via a QStackedWidget.
        self._analysis_stack = QtWidgets.QStackedWidget()

        # Page 0 — cockpit: narrow mass histogram across the top (its
        # original position / aspect), live frame view below it filling
        # all remaining vertical space.
        cockpit_w = QtWidgets.QWidget()
        cockpit   = QtWidgets.QVBoxLayout(cockpit_w)
        cockpit.setContentsMargins(0, 0, 0, 0)
        cockpit.setSpacing(6)
        self.mass_hist = _MassHistogram()
        self.live_view = _LiveFrameView()
        cockpit.addWidget(self.mass_hist)             # narrow, no stretch
        cockpit.addWidget(self.live_view, 1)          # fills the rest
        self._analysis_stack.addWidget(cockpit_w)

        # Page 1 — results: same _ResultsPanel as before.
        self.run_results = _ResultsPanel(
            "Results will appear here after analysis.")
        self._analysis_stack.addWidget(self.run_results)

        v.addWidget(self._analysis_stack, stretch=1)
        # Start on the results page (cockpit only shows during runs)
        self._analysis_stack.setCurrentIndex(1)

        self.tabs.addTab(tab, TAB_ANALYSIS)

    def _build_figures_widget(self) -> QtWidgets.QWidget:
        """Customisation for figure outputs — single-sample (Run tab) and
        comparison (Compare tab) — plus a live preview that updates as
        the user changes theme / colormap settings.

        Returns the widget so the caller can decide where to host it.
        Currently hosted inside the Preferences dialog (see
        `_PreferencesDialog`) instead of as a top-level tab — the
        figure-render knobs are app-wide defaults, not per-run state.
        """
        tab = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(tab)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        # ── Settings column ──────────────────────────────────────────────
        settings_col = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(settings_col)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)

        intro = QtWidgets.QLabel(
            "Style and output format for the figures produced by the "
            "Analysis and Compare tabs.  Preview on the right updates as "
            "you change the theme / colormap.")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_THEME['TXT_MUTED']};")
        v.addWidget(intro)

        # App theme (Qt UI) has moved to Preferences → Appearance
        # (cogwheel button in the header).  This widget is concerned
        # purely with figure-output styling now.

        # ── Single-sample figure ──────────────────────────────────────────
        sec, gl = self._make_form_section("Single-sample figure (Analysis tab)")
        self.c_fig_theme = _QuietComboBox()
        self.c_fig_theme.addItems(["Dark", "AMOLED", "Light", "Publication"])
        self.c_fig_theme.setToolTip(
            "Overall colour scheme for figure backgrounds, axes, and text.\n"
            "• Dark         — GitHub-dark (matches the GUI).\n"
            "• AMOLED       — pure-black backgrounds, matches AMOLED app theme.\n"
            "• Light        — GitHub-light, sans-serif.\n"
            "• Publication  — White background, black axes, serif font.")
        gl.addRow("Theme", self.c_fig_theme)
        self.c_fig_proj_cmap = _QuietComboBox()
        self.c_fig_proj_cmap.addItems(
            ["Inferno", "Hot", "Viridis", "Plasma", "Greys"])
        self.c_fig_proj_cmap.setToolTip(
            "Colormap for the max-projection panel.  Inferno is the\n"
            "default — perceptually uniform with deep blacks for dark\n"
            "backgrounds.  Greys flips automatically for light themes.")
        gl.addRow("Projection colormap", self.c_fig_proj_cmap)
        self.s_fig_dpi = self._spin_int(150, 72, 600, step=10,
            tip="Pixel density for the combined PNG.  150 DPI matches the\n"
                "default print size; bump to 300 for posters / publications.")
        gl.addRow("PNG DPI", self.s_fig_dpi)
        self.c_fig_save_pdf = QtWidgets.QCheckBox(
            "Also save vector PDF alongside the PNG")
        self.c_fig_save_pdf.setToolTip(
            "Write a vector PDF copy of the figure.  Same content as the\n"
            "PNG but infinitely zoomable — recommended for talks and papers.")
        gl.addRow("", self.c_fig_save_pdf)
        self.c_fig_per_panel = QtWidgets.QCheckBox(
            "Also save each panel as a separate PNG")
        self.c_fig_per_panel.setToolTip(
            "Export each labelled panel (A, B, C, …) of the combined figure\n"
            "to figures/panels/.  Useful when you want a single chart for a\n"
            "talk without cropping the full grid.")
        gl.addRow("", self.c_fig_per_panel)
        v.addWidget(sec)

        # Single-sample panel selector — only affects per-panel PNG exports
        # (combined figure always contains every panel that has data).
        single_panels_grp = QtWidgets.QGroupBox(
            "Single-sample panels to export individually")
        spg = QtWidgets.QGridLayout(single_panels_grp)
        self._single_panel_checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        for i, (key, label) in enumerate(self.SINGLE_PANELS):
            cb = QtWidgets.QCheckBox(f"{key}.  {label}")
            cb.setChecked(True)
            cb.setToolTip(
                f"Include panel {key} ({label}) when 'Also save each panel\n"
                "as a separate PNG' is on.  The combined figure always shows\n"
                "every panel that has data.")
            self._single_panel_checkboxes[key] = cb
            spg.addWidget(cb, i // 2, i % 2)
        v.addWidget(single_panels_grp)

        # ── Comparison figure (moved from Compare tab) ────────────────────
        sec, gl = self._make_form_section("Comparison figure (Compare tab)")
        self.c_cmp_theme = _QuietComboBox()
        self.c_cmp_theme.addItems(["Dark", "AMOLED", "Light", "Publication"])
        self.c_cmp_theme.setToolTip(
            "Theme for the multi-group comparison figure.  Independent\n"
            "from the single-sample theme so you can mix and match.")
        gl.addRow("Theme", self.c_cmp_theme)
        self.c_cmp_pdf = QtWidgets.QCheckBox(
            "Generate multi-page PDF report (figure + parameters + stats)")
        self.c_cmp_pdf.setChecked(True)
        gl.addRow("", self.c_cmp_pdf)
        v.addWidget(sec)

        # Comparison panels (which sub-panels to include in the figure)
        panels_grp = QtWidgets.QGroupBox("Comparison panels to include")
        pg = QtWidgets.QGridLayout(panels_grp)
        self._cmp_panel_checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        for i, (key, label) in enumerate(self.COMPARE_PANELS):
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(True)
            self._cmp_panel_checkboxes[key] = cb
            pg.addWidget(cb, i // 2, i % 2)
        v.addWidget(panels_grp)
        v.addStretch(1)

        # ── Preview column (two stacked previews) ────────────────────────
        preview_col = QtWidgets.QWidget()
        pv = QtWidgets.QVBoxLayout(preview_col)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(8)

        def _make_preview_label(caption: str) -> QtWidgets.QLabel:
            lbl = QtWidgets.QLabel("Rendering preview…")
            # Shrunk from 560×320 → 400×240 so the Figures tab can fit
            # on a narrower screen.  The previous floor was forcing the
            # whole MainWindow to claim a minimum width that exceeded
            # 13-inch laptop screens, which made fullscreen-not-zoomed
            # impossible and stopped the console dock from snapping
            # to the right side.  The preview labels are rendered to
            # an Ignored-Ignored size policy below so they still scale
            # to whatever width the user gives the tab.
            lbl.setMinimumSize(400, 240)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            # Ignored policy in both directions → layout sizes the label
            # from the stretch / minimum hints only, NOT from the pixmap's
            # natural size.  Without this, every theme change produces a
            # slightly different matplotlib output → the label's sizeHint
            # bumps up → layout reallocates → bigger label → bigger render…
            lbl.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                              QtWidgets.QSizePolicy.Policy.Ignored)
            lbl.setStyleSheet(
                f"QLabel {{ border: 1px solid {_THEME['BORDER']}; "
                f"background: {_THEME['PANEL']}; color: {_THEME['TXT_MUTED']}; "
                "border-radius: 4px; }")
            return lbl

        cap_single = QtWidgets.QLabel("Single-sample figure")
        cap_single.setStyleSheet(
            f"color: {_THEME['TXT']}; font-weight: 600;")
        pv.addWidget(cap_single)
        self.lbl_fig_preview_single = _make_preview_label("single")
        pv.addWidget(self.lbl_fig_preview_single, stretch=1)

        cap_compare = QtWidgets.QLabel("Comparison figure")
        cap_compare.setStyleSheet(
            f"color: {_THEME['TXT']}; font-weight: 600;")
        pv.addWidget(cap_compare)
        self.lbl_fig_preview_compare = _make_preview_label("comparison")
        pv.addWidget(self.lbl_fig_preview_compare, stretch=1)

        # Cache of unscaled preview pixmaps so we can re-fit them when the
        # labels resize (e.g. on window resize) without re-rendering.
        self._fig_preview_pixmaps: dict[QtWidgets.QLabel, QtGui.QPixmap] = {}
        # Install a resize filter on both labels — re-scales the cached
        # raw pixmap to fit the new label dimensions.
        for _lbl in (self.lbl_fig_preview_single, self.lbl_fig_preview_compare):
            _lbl.installEventFilter(self)

        hint = QtWidgets.QLabel(
            "Rendered on a synthetic dataset — actual figures will use "
            "your data but keep these style choices.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {_THEME['TXT_MUTED']}; font-size: 11px;")
        pv.addWidget(hint)

        outer.addWidget(settings_col, 1)
        outer.addWidget(preview_col, 2)

        # ── Debounced preview refresh ────────────────────────────────────
        self._figpreview_timer = QTimer(self)
        self._figpreview_timer.setSingleShot(True)
        self._figpreview_timer.setInterval(120)
        self._figpreview_timer.timeout.connect(self._refresh_figures_preview)
        for w in (self.c_fig_theme, self.c_fig_proj_cmap, self.c_cmp_theme):
            w.currentTextChanged.connect(
                lambda _=None: self._figpreview_timer.start())
        # First render after construction settles
        QtCore.QTimer.singleShot(80, self._refresh_figures_preview)

        return tab

    def _build_compare_tab(self):
        """Compare tab: N≥2 groups of analysis-output folders → comparison
        figure + summary CSV + stats CSV + multi-page PDF report."""
        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        # ── Comparison settings (re-parented into the Compare sidebar) ──
        # Output folder + name used to live as a GroupBox above the
        # group cards; they're now hosted in the left sidebar so the
        # tab body gives all its room to the group cards.
        self._cmp_settings_widget = QtWidgets.QWidget()
        sg = QtWidgets.QVBoxLayout(self._cmp_settings_widget)
        sg.setContentsMargins(0, 0, 0, 0); sg.setSpacing(6)

        sg.addWidget(QtWidgets.QLabel("Output folder"))
        out_row = QtWidgets.QHBoxLayout()
        out_row.setContentsMargins(0, 0, 0, 0); out_row.setSpacing(6)
        self.e_cmp_outdir = QtWidgets.QLineEdit()
        self.e_cmp_outdir.setPlaceholderText(
            "Where to save the comparison figure + CSVs + PDF report")
        btn_cmp_out = QtWidgets.QPushButton("Browse")
        btn_cmp_out.clicked.connect(self._on_cmp_browse_outdir)
        out_row.addWidget(self.e_cmp_outdir, 1)
        out_row.addWidget(btn_cmp_out)
        sg.addLayout(out_row)

        sg.addSpacing(4)
        sg.addWidget(QtWidgets.QLabel("Output name"))
        self.e_cmp_stem = QtWidgets.QLineEdit("Comparison")
        self.e_cmp_stem.setToolTip(
            "Prefix for the saved files (figure.png, summary.csv, "
            "stats.csv, report.pdf).")
        sg.addWidget(self.e_cmp_stem)

        # Pointer to where style settings now live (Preferences → Figure
        # defaults; the old standalone Figures tab no longer exists).
        style_hint = QtWidgets.QLabel(
            "<i>Figure theme, panel selection and PDF report toggle now "
            "live in <b>Preferences</b> (cogwheel in the header).</i>")
        style_hint.setTextFormat(Qt.TextFormat.RichText)
        style_hint.setWordWrap(True)
        style_hint.setStyleSheet(f"color: {_THEME['TXT_MUTED']};")
        sg.addWidget(style_hint)

        # ── Group cards (scrollable) ──────────────────────────────────────
        groups_area_label = QtWidgets.QLabel(
            "Groups  —  drop folders directly onto a card to add them, "
            "or use the buttons:")
        v.addWidget(groups_area_label)

        groups_scroll = QtWidgets.QScrollArea()
        groups_scroll.setWidgetResizable(True)
        groups_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        groups_inner = QtWidgets.QWidget()
        self._cmp_groups_layout = QtWidgets.QVBoxLayout(groups_inner)
        self._cmp_groups_layout.setContentsMargins(0, 0, 0, 0)
        self._cmp_groups_layout.setSpacing(6)
        self._cmp_groups_layout.addStretch(1)   # pushes cards to top
        groups_scroll.setWidget(groups_inner)
        v.addWidget(groups_scroll, stretch=1)

        self._cmp_group_cards: list[_CompareGroupCard] = []
        # Seed with the two default groups (Pre / Post) — matches the Tk app
        self._cmp_add_group()
        self._cmp_add_group()

        # ── Action row ────────────────────────────────────────────────────
        actions = QtWidgets.QHBoxLayout()
        self.btn_cmp_add_group = QtWidgets.QPushButton("+ Add group")
        self.btn_cmp_add_group.clicked.connect(self._cmp_add_group)
        actions.addWidget(self.btn_cmp_add_group)
        actions.addStretch(1)
        self.btn_cmp_run = QtWidgets.QPushButton("Generate comparison")
        self.btn_cmp_run.setMinimumHeight(32)
        self.btn_cmp_run.clicked.connect(self._on_run_clicked)
        actions.addWidget(self.btn_cmp_run)
        v.addLayout(actions)

        # The status widgets (stage label, progress bar, results panel)
        # used to live below the action row, but they were visually noisy
        # for a tab that mostly just configures + kicks off the comparison.
        # They're still constructed and parented to the tab so the rest of
        # the run-machinery can call .setText / .setValue / .reset on them
        # — but they're hidden so they don't show in the UI.  Progress is
        # surfaced via the status bar instead.
        self.cmp_stage_label = QtWidgets.QLabel("Idle", tab)
        self.cmp_progress    = QtWidgets.QProgressBar(tab)
        self.cmp_progress.setRange(0, 100)
        self.cmp_results     = _ResultsPanel("", parent=tab)
        for w in (self.cmp_stage_label, self.cmp_progress, self.cmp_results):
            w.hide()

        self.tabs.addTab(tab, TAB_COMPARE)

    def _build_visualise_tab(self):
        """Build the Visualise tab — toolbar + lazy-loaded napari viewer.

        Napari is imported lazily on first activation so a missing dep
        doesn't block the rest of FIREFLY from launching.  If the import
        succeeds, the viewer is embedded into this tab.  If it fails, the
        tab shows a clear placeholder with install instructions, and the
        rest of the app keeps working.
        """
        self._napari_viewer = None         # populated lazily
        self._workspace_initialised = False

        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        # ── Load buttons (re-parented into the Visualise sidebar) ──────
        # All four load buttons + the auto-load checkbox were originally
        # an inline toolbar across the top of the tab.  They've moved
        # into the left-sidebar Visualise page so the tab body can give
        # all its room to the napari viewer + track inspector.
        self._vis_load_widget = QtWidgets.QWidget()
        load_v = QtWidgets.QVBoxLayout(self._vis_load_widget)
        load_v.setContentsMargins(0, 0, 0, 0)
        load_v.setSpacing(6)

        self.btn_ws_load_stack = QtWidgets.QPushButton("Load image stack…")
        self.btn_ws_load_stack.setToolTip(
            "Open a .czi or .tif file as an Image layer in napari.")
        self.btn_ws_load_stack.clicked.connect(self._ws_on_load_stack)
        load_v.addWidget(self.btn_ws_load_stack)

        self.btn_ws_load_tracks = QtWidgets.QPushButton("Load tracks…")
        self.btn_ws_load_tracks.setToolTip(
            "Open a FIREFLY trajectories CSV as a Tracks layer overlay.")
        self.btn_ws_load_tracks.clicked.connect(self._ws_on_load_tracks)
        load_v.addWidget(self.btn_ws_load_tracks)

        self.btn_ws_load_run = QtWidgets.QPushButton("Load analysis run…")
        self.btn_ws_load_run.setToolTip(
            "Pick a FIREFLY run output folder.  Auto-loads the original\n"
            "stack and overlays the trajectories.csv as a Tracks layer.")
        self.btn_ws_load_run.clicked.connect(self._ws_on_load_run)
        load_v.addWidget(self.btn_ws_load_run)

        # ── Cluster map ─────────────────────────────────────────────
        # Loads {stem}_cluster_labels.csv from a FIREFLY run and adds
        # a colour-coded napari Points layer.  The DBSCAN sliders below
        # let the user re-cluster on the fly without re-running the
        # full pipeline.
        self.btn_ws_load_clusters = QtWidgets.QPushButton("Load cluster map…")
        self.btn_ws_load_clusters.setToolTip(
            "Load a FIREFLY run's per-localisation cluster labels and\n"
            "render them as a coloured Points layer (one colour per\n"
            "DBSCAN cluster, noise = grey).  Click any point to inspect\n"
            "its cluster's stats in the Track Inspector.\n"
            "\n"
            "The eps and min-samples sliders alongside this button let\n"
            "you re-cluster the loaded localisations on the fly.")
        self.btn_ws_load_clusters.clicked.connect(self._ws_on_load_clusters)
        load_v.addWidget(self.btn_ws_load_clusters)

        self.c_ws_auto = QtWidgets.QCheckBox("Auto-load after analysis")
        self.c_ws_auto.setChecked(False)
        self.c_ws_auto.setToolTip(
            "When checked, the Workspace tab loads the stack + tracks\n"
            "automatically after a Run-Analysis completes.\n"
            "Off by default — large stacks can use a lot of GPU memory in\n"
            "napari and slow the rest of FIREFLY down.")
        load_v.addWidget(self.c_ws_auto)

        # Reset-view: re-centre + re-fit napari camera on all visible
        # layers.  One-click recovery after the user zooms / pans off
        # the sample and gets lost.
        self.btn_ws_reset_view = QtWidgets.QPushButton("Reset view")
        self.btn_ws_reset_view.setToolTip(
            "Re-centre the napari viewer on all visible layers — recover "
            "from accidental pan/zoom that lost the sample off-screen.")
        self.btn_ws_reset_view.clicked.connect(self._ws_reset_view)
        load_v.addWidget(self.btn_ws_reset_view)

        # ── Track filter sidebar (now layer-based) ────────────────────────
        # Each motion class becomes its OWN napari Tracks layer (built by
        # `_ws_apply_motion_filter`), so visibility is controlled directly
        # from napari's built-in layer-list checkboxes — no parallel UI
        # needed.  Sidebar keeps only the min-length filter + an
        # informational label pointing users at the layer list.
        self._vis_filter_widget = QtWidgets.QWidget()
        filter_v = QtWidgets.QVBoxLayout(self._vis_filter_widget)
        filter_v.setContentsMargins(0, 0, 0, 0)
        filter_v.setSpacing(4)
        _hint = QtWidgets.QLabel(
            "Each motion class is its own layer in the viewer — "
            "toggle visibility from the napari layer list.")
        _hint.setWordWrap(True)
        _hint.setStyleSheet("color: #888;")
        filter_v.addWidget(_hint)
        # `_ws_motion_checks` is kept (empty) so any leftover references
        # in `_ws_apply_motion_filter` degrade gracefully if hit.
        self._ws_motion_checks: dict[str, QtWidgets.QCheckBox] = {}

        # Min-length filter — drop one/two-point detections that
        # clutter the field visually and aren't diffusion-classifiable.
        _ml_row = QtWidgets.QHBoxLayout()
        _ml_row.setContentsMargins(0, 6, 0, 0)
        _ml_row.addWidget(QtWidgets.QLabel("Min length:"))
        self._ws_min_len = QtWidgets.QSpinBox()
        self._ws_min_len.setRange(1, 1000)
        self._ws_min_len.setValue(1)
        self._ws_min_len.setSuffix(" frames")
        self._ws_min_len.setMaximumWidth(120)
        self._ws_min_len.valueChanged.connect(self._ws_apply_motion_filter)
        _ml_row.addWidget(self._ws_min_len, 1)
        filter_v.addLayout(_ml_row)

        self._ws_filter_status = QtWidgets.QLabel("")
        self._ws_filter_status.setStyleSheet("color: #888;")
        self._ws_filter_status.setWordWrap(True)
        filter_v.addWidget(self._ws_filter_status)

        # ── DBSCAN live-tune controls (re-parented into the sidebar) ──
        # Sliders adjust eps (nm) and min_samples for the cluster
        # overlay loaded via "Load cluster map…".  Changes are
        # debounced (300 ms) so dragging doesn't spam re-clusterings.
        # Laid out vertically as form rows so it fits the sidebar.
        self._vis_dbscan_widget = QtWidgets.QWidget()
        dbscan_form = QtWidgets.QFormLayout(self._vis_dbscan_widget)
        dbscan_form.setContentsMargins(0, 0, 0, 0)
        dbscan_form.setHorizontalSpacing(8)
        dbscan_form.setVerticalSpacing(6)

        # eps slider + value label, side by side under one form row.
        eps_w = QtWidgets.QWidget()
        eps_row = QtWidgets.QHBoxLayout(eps_w)
        eps_row.setContentsMargins(0, 0, 0, 0); eps_row.setSpacing(6)
        self._ws_eps_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self._ws_eps_slider.setRange(5, 500)
        self._ws_eps_slider.setValue(50)
        self._ws_eps_slider.setMinimumWidth(80)
        self._ws_eps_slider.setToolTip(
            "Maximum distance between two points for them to be considered\n"
            "neighbours (nm).  Larger values merge nearby clusters; smaller\n"
            "splits them.  Default 50 nm matches the standard sptPALM preset.")
        self._ws_eps_value = QtWidgets.QLabel("50 nm")
        self._ws_eps_value.setMinimumWidth(50)
        eps_row.addWidget(self._ws_eps_slider, 1)
        eps_row.addWidget(self._ws_eps_value)
        dbscan_form.addRow("eps (nm)", eps_w)

        self._ws_minsamp_spin = QtWidgets.QSpinBox()
        self._ws_minsamp_spin.setRange(2, 200)
        self._ws_minsamp_spin.setValue(8)
        self._ws_minsamp_spin.setToolTip(
            "Minimum number of points within eps to form a dense core.\n"
            "Higher = fewer, larger, denser clusters.  Default 8.")
        dbscan_form.addRow("min samples", self._ws_minsamp_spin)

        self._ws_cluster_color_mode = _QuietComboBox()
        self._ws_cluster_color_mode.addItems(["ID", "Motion"])
        self._ws_cluster_color_mode.setToolTip(
            "How to colour the cluster overlay:\n"
            "• ID     — one colour per DBSCAN cluster (turbo).\n"
            "• Motion — each loc gets the colour of its track's motion class.\n"
            "Requires the run to have saved per-loc motion data — re-run "
            "the analysis if this option appears to have no effect.")
        self._ws_cluster_color_mode.currentTextChanged.connect(
            lambda _=None: self._ws_render_cluster_layer())
        dbscan_form.addRow("Colour by", self._ws_cluster_color_mode)

        self._ws_cluster_status = QtWidgets.QLabel("")
        self._ws_cluster_status.setStyleSheet("color: #888;")
        self._ws_cluster_status.setWordWrap(True)
        dbscan_form.addRow(self._ws_cluster_status)

        # Debounce re-clustering so the slider doesn't fire on every
        # pixel of drag — coalesce to one re-cluster call per 300 ms.
        self._ws_dbscan_debounce = QTimer(self)
        self._ws_dbscan_debounce.setSingleShot(True)
        self._ws_dbscan_debounce.setInterval(300)
        self._ws_dbscan_debounce.timeout.connect(self._ws_recluster_now)
        def _on_eps_changed(v):
            self._ws_eps_value.setText(f"{int(v)} nm")
            self._ws_dbscan_debounce.start()
        self._ws_eps_slider.valueChanged.connect(_on_eps_changed)
        self._ws_minsamp_spin.valueChanged.connect(
            lambda _: self._ws_dbscan_debounce.start())

        # ── Viewer container ─────────────────────────────────────────────
        # Filled lazily on first tab activation.
        self._ws_container = QtWidgets.QWidget()
        self._ws_container_layout = QtWidgets.QVBoxLayout(self._ws_container)
        self._ws_container_layout.setContentsMargins(0, 0, 0, 0)

        # Placeholder until napari is loaded
        self._ws_placeholder = QtWidgets.QLabel(
            "napari viewer will appear here when this tab is first opened.\n\n"
            "If napari isn't installed, run:\n"
            "    pip install \"napari[pyside6]>=0.4.19\"\n"
            "and restart FIREFLY.")
        self._ws_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ws_placeholder.setStyleSheet(
            "color: #888; padding: 40px; font-size: 13px;")
        self._ws_container_layout.addWidget(self._ws_placeholder)

        # ── Inspector panel (right side, populated on track click) ───────
        self._ws_inspector = _TrackInspector()
        # Per-run state for click→stats lookup
        self._ws_tracks_df: "pd.DataFrame | None" = None
        # Set of particle IDs currently visible after the motion/length
        # filter.  `None` means "no filter has run yet — treat full df as
        # visible".  Used by the click-resolver so clicks can never land
        # on a hidden track (which would otherwise let the inspector
        # report e.g. a Confined track even when only Immobile is
        # checked, because the nearest localisation across the FULL df
        # might belong to an invisible particle that happens to lie
        # near where the user clicked).
        self._ws_visible_pids: "set[int] | None" = None
        self._ws_diff_df:   "pd.DataFrame | None" = None
        self._ws_tracks_layer = None
        # Per-motion-class napari layer names so a rebuild can clean up
        # cleanly without stale layers lingering.  Keyed by class name
        # (Immobile / Confined / Brownian / Directed / Unknown) →
        # napari layer name string.
        self._ws_motion_layer_names: dict[str, str] = {}
        # Per-class particle-ID sets so the click resolver can map a
        # click back to "which class is this from" by checking layer
        # visibility against these sets — no need for a separate cached
        # `_ws_visible_pids`.
        self._ws_motion_pids: dict[str, set] = {}
        # Cluster-map state.  Populated by _ws_on_load_clusters; the
        # DBSCAN sliders refresh _ws_cluster_layer via _ws_recluster_now.
        self._ws_cluster_layer = None
        self._ws_cluster_xy_um = None       # (N, 2) µm coords used by DBSCAN
        self._ws_cluster_xy_px = None       # (N, 2) px coords for napari overlay
        self._ws_cluster_labels = None      # (N,) int cluster IDs (-1 = noise)
        self._ws_cluster_stats_df = None    # per-cluster summary (n_locs, area…)
        self._ws_cluster_motion = None      # (N,) per-loc motion class string
        self._ws_cluster_pixel_size_um = 1.0

        split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._ws_container)
        split.addWidget(self._ws_inspector)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([1000, 320])
        v.addWidget(split, stretch=1)

        self.tabs.addTab(tab, TAB_VISUALISE)

        # Lazy-init when the tab is first switched to.
        self.tabs.currentChanged.connect(self._ws_maybe_init)

        # ── Post-Processing tab ──────────────────────────────────────
        self._init_postproc_tab()

    def _build_params_for_file(self, fpath: str, out_dir: str | None) -> dict:
        """Build the full analysis-params dict for one input file.

        Reads every spinbox / combo / checkbox in the sidebar and produces
        the kwargs dict the worker expects.  Used by both the Run-Analysis
        and Batch tabs — the only thing that differs between modes is the
        input file path and (for batch) the per-file output folder.
        """
        bg_method_map = {
            "Uniform Filter": "uniform_filter",
            "Rolling Ball":   "rolling_ball",
        }
        roi_mode_map = {
            "None":              "none",
            "Auto threshold":    "auto",
            "Manual threshold":  "manual",
            "Manual polygon":    "polygon",
            "From sister TIFF":  "sister",
        }
        max_tl = int(self.s_max_track_len.value())
        return {
            "file":              fpath,
            "out_dir":           out_dir,
            "pixel_size":        (self.s_pixel_size.value()
                                  if self.c_override_px.isChecked() else None),
            "frame_interval":    (self.s_frame_interval.value()
                                  if self.c_override_fi.isChecked() else None),
            "channel":           int(self.s_channel.value()),
            "bg_method":         bg_method_map.get(
                                    self.c_bg_method.currentText(),
                                    "uniform_filter"),
            "bg_radius":         int(self.s_bg_radius.value()),
            "diameter":          int(self.s_diameter.value()),
            "auto_minmass":      bool(self.c_auto_minmass.isChecked()),
            "minmass":           float(self.s_minmass.value()),
            "minmass_sensitivity": self.c_minmass_sensitivity.currentText().lower(),
            "search_range":      int(self.s_search_range.value()),
            "memory":            int(self.s_memory.value()),
            "min_track_len":     int(self.s_min_track_len.value()),
            "max_track_len":     max_tl if max_tl > 0 else None,
            "max_lagtime":       int(self.s_max_lagtime.value()),
            "n_fit":             int(self.s_n_fit.value()),
            "alpha_thresholds":  (float(self.s_alpha_immobile.value()),
                                  float(self.s_alpha_confined.value()),
                                  float(self.s_alpha_directed.value())),
            "mobile_d_threshold": float(self.s_mobile_d_threshold.value()),
            "jdd_components":    int(self.s_jdd_components.value()),
            "filter_d_enabled":  bool(self.c_filter_d_enabled.isChecked()),
            "filter_d_min":      float(self.s_filter_d_min.value()),
            "filter_d_max":      float(self.s_filter_d_max.value()),
            "roi_mode":          roi_mode_map.get(
                                    self.c_roi_mode.currentText(), "none"),
            "roi_auto_method":   self.c_roi_auto_method.currentText(),
            "roi_threshold":     float(self.s_roi_threshold.value()),
            "roi_mask_mode":     self.c_roi_mask_mode.currentText(),
            "roi_bg_sigma":      float(self.s_roi_bg_sigma.value()),
            # Sister-TIFF ROI — passed through whether or not the
            # explicit "From sister TIFF" mode is picked, so the worker
            # can auto-detect a `<base>_green.tif` next to the data and
            # prefer it over the intensity-based ROI when present.
            "roi_sister_suffix":      "_green",
            "roi_sister_autodetect":  True,
            # ImageJ ROI auto-pairing — find a sibling RoiSet.zip / RoiSet
            # folder / .roi next to each file and apply it as a polygon ROI.
            "roi_imagej_autodetect":  bool(self.c_roi_imagej_auto.isChecked()),
            # Per-file polygon ROI lookup.  If this file has a saved
            # polygon, it's sent regardless of the ROI-mode setting and
            # the worker treats it as if mode were "polygon".  Files
            # without a saved polygon fall back to the global ROI mode.
            "roi_polygon":       self._roi_polygons.get(
                                    os.path.abspath(fpath)) or None,
            "drift_correct":     bool(self.c_drift_correct.isChecked()),
            "drift_segment":     int(self.s_drift_segment.value()),
            "cluster_eps_nm":      float(self.s_cluster_eps_nm.value()),
            "cluster_min_samples": int(self.s_cluster_min_samples.value()),
            "backend":           self._backend_value_from_label(
                                    self.c_backend.currentText()),
            "workers":           int(self.s_workers.value()),
            "chunk_size":        int(self.s_chunk_size.value()),
            # FIREFLY links exclusively with trackpy's recursive subnet
            # linker.  Kept as an explicit key for the worker / manifest.
            "linker":            "trackpy",
            # ── Figures-tab knobs (single-sample figure output) ───────────
            "fig_theme":         self.c_fig_theme.currentText(),
            "fig_proj_cmap":     self.c_fig_proj_cmap.currentText(),
            "fig_dpi":           int(self.s_fig_dpi.value()),
            "fig_save_pdf":      bool(self.c_fig_save_pdf.isChecked()),
            "fig_per_panel":     bool(self.c_fig_per_panel.isChecked()),
            "fig_single_panels": [k for k, cb in
                                  self._single_panel_checkboxes.items()
                                  if cb.isChecked()],
            # Full widget-state snapshot — written into the run manifest
            # so the run can be exactly replayed later via "Load manifest…"
            "widget_state":      self._widget_state_dict(),
        }
