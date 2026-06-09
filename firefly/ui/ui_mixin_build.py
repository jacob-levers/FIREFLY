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
                          TAB_RESULTS, TAB_VISUALISE, TAB_REPROCESS)
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
                        _load_any_roi_file, _info_icon, _InfoIcon,
                        _label_with_info, _AlertBanner, _StatusBadge,
                        _HyperflyPill,
                        _step_badge, _color_chip, _DecisionDiagram,
                        _PipelineDiagram, _NoHScrollArea)


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
            f"color: {_THEME['TXT']}; font-weight: 800; font-size: 15px; "
            f"padding: 14px 12px 10px 12px; "
            f"border-bottom: 1px solid {_THEME['BORDER']};")
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
        self._build_results_tab()
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
        # Keep the Compare tab's "Analysis Configuration" test-plan preview in
        # step with the group cards whenever the user switches to it.
        self.tabs.currentChanged.connect(lambda *_: self._refresh_stats_preview())

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
        # ── Parameter search ──────────────────────────────────────────────
        # Filters the (long) parameter list down to the rows whose label
        # matches — typing 'minmass' or 'roi' surfaces just those knobs and
        # expands their sections.  Registry built at the end of this method.
        self._sidebar_sections = []
        self._param_search = QtWidgets.QLineEdit()
        self._param_search.setObjectName("param_search")
        self._param_search.setPlaceholderText("Search parameters…")
        self._param_search.setClearButtonEnabled(True)
        self._param_search.setToolTip(
            "Filter the parameters below by name (e.g. minmass, search, ROI).")
        self._param_search.textChanged.connect(self._filter_sidebar_params)
        layout.addWidget(self._param_search)

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
        # "• modified" pill — shown when the current parameters differ from the
        # selected preset (baseline tracked in _on_preset_picked).
        self._active_preset_state = None
        self._suspend_modified_watch = False
        self._modified_badge = _StatusBadge()
        self._modified_badge.setToolTip(
            "The parameters have been changed since this preset was applied. "
            "Save a new preset to keep them.")
        self._modified_badge.hide()
        preset_row.addWidget(self._modified_badge)
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
        gl.addRow(_label_with_info("Pixel size (µm)", "pixel size"), wpx)

        row = QtWidgets.QHBoxLayout()
        self.c_override_fi = QtWidgets.QCheckBox("Override")
        self.c_override_fi.setToolTip(
            "If unchecked, the frame interval from the file's metadata is used.")
        self.s_frame_interval = self._spin_dbl(0.02, 0.001, 10.0, 0.001, decimals=3,
            tip="Time between frames in seconds. Used for diffusion coefficient units.")
        row.addWidget(self.c_override_fi); row.addWidget(self.s_frame_interval, 1)
        wfi = QtWidgets.QWidget(); wfi.setLayout(row)
        gl.addRow(_label_with_info("Frame interval (s)", "frame interval"), wfi)

        self.s_channel = self._spin_int(0, 0, 8,
            tip="Channel index to load (CZI files only). Most single-channel data uses 0.")
        gl.addRow(_label_with_info("Channel (CZI)", "channel"), self.s_channel)
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
        gl.addRow(_label_with_info("Background method", "background method"), self.c_bg_method)
        self.s_bg_radius = self._spin_int(10, 3, 200,
            tip="Radius (px) of the local-mean window for background subtraction.\n"
                "Use ~3× spot diameter for diffraction-limited spots.")
        gl.addRow(_label_with_info("Background radius (px)", "background radius"), self.s_bg_radius)
        layout.addWidget(sec)

        # ── Detection ─────────────────────────────────────────────────────
        sec, gl = self._make_form_section("Detection")
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.s_diameter = self._spin_int(7, 3, 21, step=2,
            tip="Expected spot diameter in pixels. Must be ODD (the GUI enforces this).\n"
                "Use ~2× the diffraction-limited PSF FWHM. Too small misses spots; "
                "too big merges adjacent ones.")
        gl.addRow(_label_with_info("Diameter (px, odd)", "diameter"), self.s_diameter)

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

        # Advanced override: directly cap the MEASURED spurious-fragment rate
        # of the linkability sweep.  0 % = off (use the Strict/Balanced/Lenient
        # selector instead).  When set, the estimator picks the most permissive
        # threshold whose measured 1–2-frame-fragment rate stays at/below this.
        self.s_minmass_false_rate = QtWidgets.QDoubleSpinBox()
        self.s_minmass_false_rate.setRange(0.0, 50.0)
        self.s_minmass_false_rate.setDecimals(1)
        self.s_minmass_false_rate.setSingleStep(1.0)
        self.s_minmass_false_rate.setValue(0.0)
        self.s_minmass_false_rate.setSuffix(" %")
        # Keep the special text SHORT — a long string here inflates the
        # spinbox's minimum size hint, which (with the sidebar's fixed width and
        # horizontal scrollbar disabled) would clip the whole parameters panel.
        self.s_minmass_false_rate.setSpecialValueText("off")
        self.s_minmass_false_rate.setMaximumWidth(160)
        self.s_minmass_false_rate.setToolTip(
            "Advanced — max false-track rate (linkability sweep only):\n"
            "Directly caps the measured fraction of 1–2-frame spurious\n"
            "fragments among surviving detections.  The estimator then picks\n"
            "the most permissive threshold meeting that ceiling, overriding\n"
            "the Sensitivity selector.  0 % = off.")

        def _on_auto_toggled(checked):
            self.s_minmass.setEnabled(not checked)
            self.sld_minmass.setEnabled(not checked)
            self.c_minmass_sensitivity.setEnabled(checked)
            self.s_minmass_false_rate.setEnabled(checked)
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
        # These are plain inner labels (not QFormLayout row labels), so the
        # auto tooltip-propagation can't reach them — give each its own tooltip
        # (same text as the control) so hovering the NAME explains it too.
        _sens_lbl = QtWidgets.QLabel("Sensitivity")
        _sens_lbl.setToolTip(self.c_minmass_sensitivity.toolTip())
        srow.addWidget(_sens_lbl)
        srow.addWidget(self.c_minmass_sensitivity, 1)
        vmm.addLayout(srow)
        frow = QtWidgets.QHBoxLayout()
        frow.setContentsMargins(0, 0, 0, 0)
        _ftr_lbl = QtWidgets.QLabel("Max false-track rate")
        _ftr_lbl.setToolTip(self.s_minmass_false_rate.toolTip())
        frow.addWidget(_ftr_lbl)
        frow.addWidget(self.s_minmass_false_rate, 1)
        vmm.addLayout(frow)
        gl.addRow(_label_with_info("Threshold", "minmass"), wmm)

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
        gl.addRow(_label_with_info("Search range (px)", "search range"), self.s_search_range)
        self.s_memory = self._spin_int(3, 0, 10,
            tip="Number of frames a track can disappear and still be re-linked.\n"
                "0 = strict (no gaps). 3 is typical for blinking PALM probes.")
        gl.addRow(_label_with_info("Memory (frames)", "memory"), self.s_memory)
        self.s_min_track_len = self._spin_int(8, 3, 50,
            tip="Tracks shorter than this are discarded. 8 is the de-facto minimum\n"
                "for reliable MSD fits.")
        gl.addRow(_label_with_info("Min track length", "min track length"), self.s_min_track_len)
        self.s_max_track_len = self._spin_int(0, 0, 100000,
            tip="0 = disabled. If set, drops tracks longer than this. Useful for\n"
                "removing stuck/aggregated particles that masquerade as long tracks.")
        gl.addRow(_label_with_info("Max track length (0 = off)", "max track length"), self.s_max_track_len)
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
        gl.addRow(_label_with_info("Max lag time (frames)", "max lag time"), _w)

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
        gl.addRow(_label_with_info("N fit lags (frames)", "n fit lags"), _w2)

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
        gl.addRow(_label_with_info("α  immobile threshold", "alpha threshold"), self.s_alpha_immobile)
        self.s_alpha_confined = self._spin_dbl(0.9, 0.0, 2.0, 0.01, decimals=2,
            tip="α between immobile and this → 'Confined'.")
        gl.addRow(_label_with_info("α  confined threshold", "alpha threshold"), self.s_alpha_confined)
        self.s_alpha_directed = self._spin_dbl(1.1, 0.0, 2.0, 0.01, decimals=2,
            tip="α above this → 'Directed'. Between confined and directed → 'Brownian'.")
        gl.addRow(_label_with_info("α  directed threshold", "alpha threshold"), self.s_alpha_directed)
        self.s_mobile_d_threshold = self._spin_dbl(0.05, 0.0, 10.0, 0.01, decimals=3,
            tip="Diffusion coefficient threshold separating 'mobile' from\n"
                "'immobile' tracks for the mobile-fraction-over-time panel.")
        gl.addRow(_label_with_info("Mobile D threshold (µm²/s)", "mobile-D threshold"), self.s_mobile_d_threshold)
        self.s_jdd_components = self._spin_int(2, 1, 4,
            tip="Number of exponential components in the Jump Distance Distribution\n"
                "fit. 2 is typical (mobile + immobile populations).")
        gl.addRow(_label_with_info("JDD components", "JDD components"), self.s_jdd_components)

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
        gl.addRow(_label_with_info("  D min (µm²/s)", "filter by D"), self.s_filter_d_min)
        gl.addRow(_label_with_info("  D max (µm²/s)", "filter by D"), self.s_filter_d_max)
        layout.addWidget(sec)

        # ── ROI ───────────────────────────────────────────────────────────
        sec, gl = self._make_form_section("ROI")
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.c_roi_mode = _QuietComboBox()
        self.c_roi_mode.addItems(
            ["None", "Auto threshold", "Manual threshold",
             "Manual polygon", "Sister TIFF", "ImageJ ROI"])
        self.c_roi_mode.setCurrentText("Auto threshold")
        self.c_roi_mode.setToolTip(
            "Restrict analysis to a region of interest in the field of view.\n"
            "• None — analyse the whole image.\n"
            "• Auto threshold — pick a threshold from the chosen projection.\n"
            "• Manual threshold — use the value below.\n"
            "• Manual polygon — draw a polygon per file on the Import tab\n"
            "  (Set ROI… buttons).  Files without a saved polygon fall back\n"
            "  to the global Auto-threshold behaviour.\n"
            "• Sister TIFF — use a microscope-exported ROI image saved next\n"
            "  to the data as `<base><suffix>.tif` (suffix defaults to\n"
            "  `_green`).  Auto-thresholded with Li if a fluorescence channel,\n"
            "  or non-zero pixels if a binary mask; multi-frame ROIs are\n"
            "  max-projected.\n"
            "• ImageJ ROI — use a sibling ImageJ/Fiji ROI next to each movie\n"
            "  (RoiSet.zip, a RoiSet/ folder, or <name>.roi / <name>.zip) as\n"
            "  a polygon ROI, so a batch reuses ROIs drawn in ImageJ without\n"
            "  loading each by hand.  Files with no sibling ROI fall back to\n"
            "  the whole image (logged).")
        gl.addRow(_label_with_info("Mode", "ROI mode"), self.c_roi_mode)
        self.c_roi_auto_method = _QuietComboBox()
        self.c_roi_auto_method.addItems(["Li", "Otsu", "Triangle", "Mean"])
        self.c_roi_auto_method.setToolTip(
            "Auto-thresholding method (from scikit-image).  Li is robust for\n"
            "low-contrast SMLM data; Otsu for bimodal histograms.")
        gl.addRow(_label_with_info("Auto method", "ROI auto method"), self.c_roi_auto_method)
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
        gl.addRow(_label_with_info("Manual threshold", "ROI threshold"), wrt)
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
        gl.addRow(_label_with_info("Projection for ROI", "ROI projection"), self.c_roi_mask_mode)

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
        gl.addRow(_label_with_info("Background scale σ", "background sigma"), wbg)

        # (ImageJ ROI auto-pairing is now the "ImageJ ROI" entry in the Mode
        # dropdown above — it is a ROI source, not a global toggle.)

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
        # Apply the initial per-mode greying of sub-controls.
        self._on_roi_mode_changed(self.c_roi_mode.currentText())
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
        gl.addRow(_label_with_info("Segment size (frames)", "drift segment"), self.s_drift_segment)
        layout.addWidget(sec)

        # ── Clustering ────────────────────────────────────────────────────
        sec, gl = self._make_form_section("Clustering (DBSCAN)")
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.s_cluster_eps_nm = self._spin_dbl(50.0, 5.0, 2000.0, 5.0, decimals=1,
            tip="DBSCAN neighbourhood radius (nm). Two localisations are in the\n"
                "same cluster if they're within this distance.")
        gl.addRow(_label_with_info("eps (nm)", "DBSCAN eps"), self.s_cluster_eps_nm)
        self.s_cluster_min_samples = self._spin_int(10, 2, 100,
            tip="Minimum localisations to form a DBSCAN cluster. Lower = more\n"
                "clusters detected but noisier; higher = stricter.")
        gl.addRow(_label_with_info("min samples", "min samples"), self.s_cluster_min_samples)
        layout.addWidget(sec)

        # ── Performance ───────────────────────────────────────────────────
        sec, gl = self._make_form_section(f"Performance  —  {N_CPUS} cores")
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.c_backend = _QuietComboBox()
        self.c_backend.addItems(self._available_backends())
        # Default to "Auto": it picks a healthy GPU (CUDA / Apple MPS) when one
        # is present, and the multi-core trackpy path when there isn't.  The old
        # default ("Torch (auto)") FORCED the Torch backend — which on a no-GPU
        # machine silently runs single-process on CPU and badly under-uses a
        # many-core box.  A saved user preference overrides this on restore.
        self.c_backend.setCurrentText("Auto")
        self.c_backend.setToolTip(
            "Which implementation to use for spot localisation.\n"
            "• Auto                      — recommended: GPU when healthy, else the\n"
            "                              multi-core trackpy CPU path.\n"
            "• Trackpy (CPU)             — reference CPU implementation (battle-tested,\n"
            "                              uses all cores).\n"
            "• Torch — GPU (auto device) — force PyTorch; auto-selects CUDA/MPS, and\n"
            "                              falls back to CPU if no GPU (single-process\n"
            "                              — slow on CPU-only machines).\n"
            "• Torch — Apple MPS         — force Apple GPU.  Fast when stable; on some\n"
            "                              macOS/M-chip combinations may hit memory-\n"
            "                              allocator issues at very low minmass.\n"
            "• Torch — NVIDIA CUDA       — force NVIDIA GPU.\n"
            "• Torch — CPU               — force PyTorch on CPU (for benchmarking).")
        gl.addRow(_label_with_info("Detection backend", "detection backend"), self.c_backend)
        self.s_workers = self._spin_int(N_CPUS, 1, N_CPUS,
            tip="Parallel CPU workers for the trackpy backend's multiprocessing\n"
                "pool and the MSD fitting thread pool.  Default = all cores.")
        gl.addRow(f"CPU workers (max {N_CPUS})", self.s_workers)
        self.s_chunk_size = self._spin_int(500, 50, 5000, step=100,
            tip="Frames per processing chunk. Bigger = less per-chunk overhead\n"
                "(esp. on GPU) but more RAM. 500 is balanced; tune up if your\n"
                "stack and free RAM are large.")
        gl.addRow(_label_with_info("Chunk size (frames)", "chunk size"), self.s_chunk_size)

        # ── HYPERFLY — big-machine parallel batch ──
        self.c_hyperfly = _QuietComboBox()
        self.c_hyperfly.addItems(["Auto (recommended)", "Always on", "Off"])
        self.c_hyperfly.setCurrentText("Auto (recommended)")
        self.c_hyperfly.setToolTip(
            "HYPERFLY: on a big machine (≥32 cores AND ≥192 GB RAM) process\n"
            "several files at once, RAM-resident, so a batch uses the whole box\n"
            "instead of one file at a time. Per-file results are identical.\n"
            "• Auto       — engage automatically on capable machines.\n"
            "• Always on  — force it (best-effort on smaller boxes).\n"
            "• Off        — always process one file at a time.")
        gl.addRow("HYPERFLY batch", self.c_hyperfly)
        self.s_hyperfly_max_files = self._spin_int(0, 0, 999,
            tip="Cap how many files HYPERFLY runs at once (0 = automatic).\n"
                "Lower it if IT wants FIREFLY to use fewer resources.")
        gl.addRow("Max concurrent files (0 = auto)", self.s_hyperfly_max_files)
        self.s_hyperfly_max_cores = self._spin_int(0, 0, N_CPUS,
            tip="Cap the total CPU cores HYPERFLY uses across all files\n"
                "(0 = all cores).")
        gl.addRow(f"Max cores (0 = auto, ≤{N_CPUS})", self.s_hyperfly_max_cores)

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

        # Index every section + row for the search box, and collapse the
        # advanced sections by default so the sidebar opens scannable.
        self._build_param_registry(layout)
        # Status chips on the section headers (active config at a glance).
        self._wire_sidebar_chips()
        layout.addStretch(1)

    def _build_param_registry(self, layout):
        """Walk the just-built sidebar and record each collapsible section + its
        form rows, so `_filter_sidebar_params` can show/hide rows by name.  Also
        collapses the advanced sections (ROI / Drift / Clustering / Performance)
        by default — the common knobs (Detection / Linking / Diffusion …) stay
        open."""
        advanced = {"ROI", "Drift correction", "Clustering (DBSCAN)"}
        LabelRole = QtWidgets.QFormLayout.ItemRole.LabelRole
        FieldRole = QtWidgets.QFormLayout.ItemRole.FieldRole
        self._sidebar_sections = []
        for i in range(layout.count()):
            w = layout.itemAt(i).widget()
            if not isinstance(w, _CollapsibleSection):
                continue
            title = (getattr(w, "_title", "") or "").replace("&&", "&")
            form = w.findChild(QtWidgets.QFormLayout)
            rows = []
            if form is not None:
                for r in range(form.rowCount()):
                    li = form.itemAt(r, LabelRole)
                    fi = form.itemAt(r, FieldRole)
                    label = li.widget() if li is not None else None
                    field = fi.widget() if fi is not None else None
                    if field is None:
                        continue
                    if label is not None and hasattr(label, "text"):
                        text = label.text()
                    elif isinstance(field, QtWidgets.QCheckBox):
                        text = field.text()          # checkbox-only rows
                    else:
                        text = ""
                    rows.append({"label": label, "field": field,
                                 "text": text.strip().lower()})
            is_advanced = (title in advanced) or title.startswith("Performance")
            if is_advanced:
                w.set_expanded(False)
            self._sidebar_sections.append(
                {"sec": w, "form": form, "title": title,
                 "default_expanded": not is_advanced, "rows": rows})

    def _filter_sidebar_params(self, query):
        """Show only the parameter rows whose label matches `query`; hide
        sections with no match and force-expand the ones that do.  An empty
        query restores every row and each section's default expand state."""
        q = (query or "").strip().lower()
        for entry in getattr(self, "_sidebar_sections", []):
            sec = entry["sec"]
            title_match = bool(q) and q in entry["title"].lower()
            any_vis = False
            for row in entry["rows"]:
                vis = (not q) or (q in row["text"]) or title_match
                if row["label"] is not None:
                    row["label"].setVisible(vis)
                row["field"].setVisible(vis)
                any_vis = any_vis or vis
            if not q:
                sec.setVisible(True)
                sec.set_expanded(entry["default_expanded"])
            else:
                show = any_vis or title_match
                sec.setVisible(show)
                if show:
                    sec.set_expanded(True)

    # ── Section status chips ────────────────────────────────────────────────
    def _sec_by_title(self, title):
        return next((e["sec"] for e in getattr(self, "_sidebar_sections", [])
                     if e["title"] == title), None)

    def _wire_sidebar_chips(self):
        """Show a small status chip on the Detection / ROI / Diffusion / Drift
        section headers so the active config reads at a glance — even when the
        section is collapsed.  Updates live from the relevant controls (and
        automatically when a preset is applied, since that fires the signals)."""
        self._sec_detection = self._sec_by_title("Detection")
        self._sec_roi       = self._sec_by_title("ROI")
        self._sec_diffusion = self._sec_by_title(
            "Diffusion & motion classification")
        self._sec_drift     = self._sec_by_title("Drift correction")
        try:
            self.c_auto_minmass.toggled.connect(
                lambda *_: self._refresh_minmass_chip())
            self.s_minmass.valueChanged.connect(
                lambda *_: self._refresh_minmass_chip())
            self.c_roi_mode.currentTextChanged.connect(
                lambda *_: self._refresh_roi_chip())
            self.c_filter_d_enabled.toggled.connect(
                lambda *_: self._refresh_diffusion_chip())
            self.c_drift_correct.toggled.connect(
                lambda *_: self._refresh_drift_chip())
        except Exception:
            pass
        # Initial state (before settings-restore fires the signals).
        self._refresh_minmass_chip()
        self._refresh_roi_chip()
        self._refresh_diffusion_chip()
        self._refresh_drift_chip()

    def _refresh_minmass_chip(self):
        sec = getattr(self, "_sec_detection", None)
        if sec is None:
            return
        if self.c_auto_minmass.isChecked():
            sec.set_badge("auto", "active")
        else:
            sec.set_badge("manual", "muted")

    def _refresh_roi_chip(self):
        sec = getattr(self, "_sec_roi", None)
        if sec is None:
            return
        mode = self.c_roi_mode.currentText()
        if mode == "None":
            sec.set_badge("none", "muted")
        else:
            sec.set_badge(mode.lower(), "active")

    def _refresh_diffusion_chip(self):
        sec = getattr(self, "_sec_diffusion", None)
        if sec is None:
            return
        # Only flag the optional D-filter when it's ON (off is the default).
        if self.c_filter_d_enabled.isChecked():
            sec.set_badge("D-filter on", "active")
        else:
            sec.set_badge("")

    def _refresh_drift_chip(self):
        sec = getattr(self, "_sec_drift", None)
        if sec is None:
            return
        # Drift correction off is the default → show the chip only when ON.
        if self.c_drift_correct.isChecked():
            sec.set_badge("RCC on", "active")
        else:
            sec.set_badge("")

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
          2 → Compare tab      → Comparison settings + group cards
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

        # Page 2 — Compare: output folder/name AND the group/file/colour
        # cards both live in the sidebar; the tab body is the wizard.
        compare_page = QtWidgets.QWidget()
        cp_outer = QtWidgets.QVBoxLayout(compare_page)
        cp_outer.setContentsMargins(0, 0, 0, 0); cp_outer.setSpacing(0)
        # Vertical-only scroll area — clamps its content to the viewport width
        # so the group cards can't be dragged sideways (the bug AlwaysOff alone
        # didn't stop, since trackpad swipes still scroll hidden overflow).
        cp_scroll = _NoHScrollArea()
        cp_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        cp_inner = QtWidgets.QWidget()
        cp_v = QtWidgets.QVBoxLayout(cp_inner)
        cp_v.setContentsMargins(12, 0, 12, 12); cp_v.setSpacing(8)
        sec_cmp = _CollapsibleSection("Comparison settings")
        sec_cmp.content_layout.addWidget(self._cmp_settings_widget)
        cp_v.addWidget(sec_cmp)
        sec_groups = _CollapsibleSection("Groups")
        sec_groups.content_layout.addWidget(self._cmp_groups_container)
        cp_v.addWidget(sec_groups)
        cp_v.addStretch(1)
        cp_scroll.setWidget(cp_inner)
        cp_outer.addWidget(cp_scroll)
        self._sidebar_stack.addWidget(compare_page)          # index 2

        # Page 3 — Results (minimal: just an "open a previous comparison" entry;
        # the tab body itself carries the same affordance).
        results_page = QtWidgets.QWidget()
        rp_v = QtWidgets.QVBoxLayout(results_page)
        rp_v.setContentsMargins(12, 8, 12, 12); rp_v.setSpacing(8)
        _rp_hint = QtWidgets.QLabel(
            "Results from the last comparison appear in this tab. You can also "
            "open a saved comparison from any past output folder.")
        _rp_hint.setWordWrap(True)
        _rp_hint.setStyleSheet(f"color:{_THEME['TXT_MUTED']};")
        rp_v.addWidget(_rp_hint)
        _rp_btn = QtWidgets.QPushButton("Open a previous comparison…")
        _rp_btn.clicked.connect(self._open_previous_comparison)
        rp_v.addWidget(_rp_btn)
        rp_v.addStretch(1)
        self._sidebar_stack.addWidget(results_page)          # index 3

        # Page 4 — Visualise (re-parents load/filter/DBSCAN widgets).
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
        self._sidebar_stack.addWidget(vis_page)              # index 4

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
        self._sidebar_stack.addWidget(pp_page)               # index 5

        # ── Bottom-button pages (action stack) ────────────────────────
        # Page 1 — Analysis: no action (empty placeholder).
        self._sidebar_action.addWidget(QtWidgets.QWidget())   # index 1
        # Page 2 — Compare: the primary "Generate comparison" button.  (The
        # old "Configure statistics" button is gone — the statistics wizard
        # now lives in this same tab's centre.)
        cmp_act = QtWidgets.QWidget()
        cmp_av = QtWidgets.QVBoxLayout(cmp_act)
        cmp_av.setContentsMargins(12, 6, 12, 12)
        cmp_av.setSpacing(6)
        _btn = getattr(self, "btn_cmp_run", None)
        if _btn is not None:
            _btn.setMinimumHeight(36)
            cmp_av.addWidget(_btn)
        self._sidebar_action.addWidget(cmp_act)               # index 2
        # Page 3 — Results: no bottom action.
        self._sidebar_action.addWidget(QtWidgets.QWidget())   # index 3
        # Page 4 — Visualise: no action.
        self._sidebar_action.addWidget(QtWidgets.QWidget())   # index 4
        # Page 5 — Re-process: the action widget built by the tab.
        self._sidebar_action.addWidget(self._pp_action_widget)  # index 5

    def _refresh_import_readiness(self, *_args):
        """Update the Import tab's readiness pill: green 'Ready to analyse' once
        an input is chosen (a file in single mode, or a folder in batch mode),
        muted 'Incomplete' otherwise.  Wired to the input fields + mode tiles."""
        badge = getattr(self, "_import_status_badge", None)
        if badge is None:
            return
        is_batch = (getattr(self, "r_mode_batch", None) is not None
                    and self.r_mode_batch.isChecked())
        if is_batch:
            e = getattr(self, "e_batch_folder", None)
        else:
            e = getattr(self, "e_file", None)
        has_input = bool(e is not None and e.text().strip())
        if has_input:
            badge.set_state("ready", "Ready to analyse")
        else:
            badge.set_state("muted", "Incomplete")

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

        # ── Header: title + a live "ready to analyse" pill ──
        _hdr = QtWidgets.QHBoxLayout()
        _ht = QtWidgets.QLabel("Import")
        _htf = _ht.font(); _htf.setBold(True); _htf.setPointSize(15)
        _ht.setFont(_htf)
        _hdr.addWidget(_ht)
        _hdr.addStretch(1)
        # Blue animated "HYPERFLY engaged" pill — hidden until a parallel
        # multi-file batch starts (see _handle_hyperfly_status).  Sits just
        # left of the run-readiness pill.
        self._hyperfly_pill = _HyperflyPill()
        _hdr.addWidget(self._hyperfly_pill)
        self._import_status_badge = _StatusBadge()
        self._import_status_badge.set_state("muted", "Incomplete")
        _hdr.addWidget(self._import_status_badge)
        v.addLayout(_hdr)

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

        # Pipeline stage map — shows which analysis stage is running / done /
        # pending.  Driven by the worker's progress messages (see the poll
        # handler); idle = all pending, set_complete() on finish.
        self.pipeline_diagram = _PipelineDiagram()
        v.addWidget(self.pipeline_diagram)

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
        self.c_fig_theme.addItems(["Dark", "Light", "Publication"])
        self.c_fig_theme.setToolTip(
            "Overall colour scheme for figure backgrounds, axes, and text.\n"
            "• Dark         — GitHub-dark (matches the GUI).\n"
            "• Light        — GitHub-light, sans-serif.\n"
            "• Publication  — White background, black axes, sans-serif.")
        gl.addRow("Theme", self.c_fig_theme)
        self.c_fig_proj_cmap = _QuietComboBox()
        self.c_fig_proj_cmap.addItems(
            ["Inferno", "Hot", "Viridis", "Plasma", "Greys"])
        self.c_fig_proj_cmap.setToolTip(
            "Colormap for the max-projection panel.  Inferno is the\n"
            "default — perceptually uniform with deep blacks for dark\n"
            "backgrounds.  Greys flips automatically for light themes.")
        gl.addRow("Projection colormap", self.c_fig_proj_cmap)
        self.c_fig_traj_bg = QtWidgets.QCheckBox(
            "Show cell image behind trajectories")
        self.c_fig_traj_bg.setChecked(True)
        self.c_fig_traj_bg.setToolTip(
            "Draw the faint max-projection (the 'cell') behind the trajectory\n"
            "panels (B — Trajectories, C — Trajectories by D value).\n"
            "Turn off to plot the tracks on a plain background — only the\n"
            "trajectories, no cell image.")
        gl.addRow("", self.c_fig_traj_bg)
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
        self.c_cmp_theme.addItems(["Dark", "Light", "Publication"])
        self.c_cmp_theme.setToolTip(
            "Theme for the multi-group comparison figure.  Independent\n"
            "from the single-sample theme so you can mix and match.")
        gl.addRow("Theme", self.c_cmp_theme)
        self.c_cmp_pdf = QtWidgets.QCheckBox(
            "Generate multi-page PDF report (figure + parameters + stats)")
        self.c_cmp_pdf.setToolTip(
            "Save a multi-page PDF alongside the comparison PNG: page 1 the\n"
            "figure, then the analysis parameters and the full statistics\n"
            "tables (per-panel tests and, when time points are set, the\n"
            "two-way mixed-ANOVA results) in GraphPad-style tabular form.")
        self.c_cmp_pdf.setChecked(True)
        gl.addRow("", self.c_cmp_pdf)
        v.addWidget(sec)

        # Comparison panels (which sub-panels to include in the figure).
        # Per-panel hover help so it's clear what each comparison shows —
        # especially the metric-based ones (AUC, mobile/immobile, α₂, VACF).
        # When time points are set the bar panels become group × time-point
        # interaction plots carrying the two-way-ANOVA p-values.
        _cmp_panel_tips = {
            "msd": "Ensemble-averaged mean-squared-displacement curve per group.\n"
                   "Overall mobility — a steeper / higher curve = faster diffusion.",
            "auc": "Area under each group's ensemble MSD curve — a single mobility\n"
                   "summary per group (higher = more mobile). With time points this\n"
                   "becomes a group × time interaction plot with the mixed-ANOVA p.",
            "logd_dist": "Distribution of per-track log10 diffusion coefficients per\n"
                   "group. Shifts or extra peaks reveal mobile / immobile sub-\n"
                   "populations the means alone hide.",
            "mob_immob": "Ratio of mobile to immobile tracks per group (split at the\n"
                   "Mobile-D threshold). Higher = larger mobile fraction. With time\n"
                   "points: group × time interaction plot with the mixed-ANOVA p.",
            "motion_classes": "Fraction of tracks in each motion class\n"
                   "(Immobile / Confined / Brownian / Directed, classified by the\n"
                   "anomalous exponent α) per group.",
            "track_length": "Distribution of track durations per group. Longer tracks\n"
                   "mean better-sampled motion (less photobleaching / blinking drop-out).",
            "track_count": "Number of tracks detected per group. Flags a group with\n"
                   "anomalously many or few tracks — a detection-threshold or\n"
                   "sample-quality difference rather than a biological one.",
            "jdd": "Single-frame jump-distance distribution per group, fitted to\n"
                   "mobile + immobile populations (an alternative to MSD for\n"
                   "estimating diffusion coefficients and population fractions).",
            "dwell_cdf": "Cumulative distribution of immobile dwell times per group —\n"
                   "differences in binding / residence time.",
            "turning_angles": "Distribution of step-to-step turning angles per group.\n"
                   "A peak near 180° = caged / back-tracking motion; near 0° = directed.",
            "radial_dist": "Polar (radial) view of turning-angle magnitudes |θ| per\n"
                   "group — highlights directional asymmetry in the motion.",
            "van_hove": "Non-Gaussian parameter α₂ of the van Hove displacement\n"
                   "distribution per group. α₂ ≈ 0 → uniform / Brownian population;\n"
                   "α₂ > 0 → a mixed / heterogeneous population (fast + slow movers).",
            "vacf": "Velocity autocorrelation at lag 1 per group (directional\n"
                   "persistence). ≈ 0 → Brownian, > 0 → directed / persistent,\n"
                   "< 0 → caged / anti-persistent (bounces back).",
        }
        panels_grp = QtWidgets.QGroupBox("Comparison panels to include")
        pg = QtWidgets.QGridLayout(panels_grp)
        self._cmp_panel_checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        for i, (key, label) in enumerate(self.COMPARE_PANELS):
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(True)
            cb.setToolTip(_cmp_panel_tips.get(
                key, f"Include the {label} panel in the comparison figure."))
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
        figure + summary CSV + stats CSV + multi-page PDF report.

        Layout: the LEFT SIDEBAR holds the group/file/colour cards plus the
        output folder/name (built here, re-parented in
        `_build_remaining_sidebar_pages`); the CENTRE is the wizard-style
        'Analysis Configuration' panel built by `_build_stats_centre`."""
        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        # ── Comparison settings (output folder/name) — re-parented to the
        # Compare sidebar's "Comparison settings" section. ──
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

        # ── Group cards container — re-parented to the Compare sidebar's
        # "Groups" area (the sidebar page provides the scroll). ──
        self._cmp_groups_container = QtWidgets.QWidget()
        gc = QtWidgets.QVBoxLayout(self._cmp_groups_container)
        gc.setContentsMargins(0, 0, 0, 0); gc.setSpacing(6)
        groups_area_label = QtWidgets.QLabel(
            "Drop analysis-output folders onto a card to add them, or use the "
            "buttons on each card.")
        groups_area_label.setWordWrap(True)
        groups_area_label.setStyleSheet(f"color: {_THEME['TXT_MUTED']};")
        gc.addWidget(groups_area_label)

        groups_inner = QtWidgets.QWidget()
        self._cmp_groups_layout = QtWidgets.QVBoxLayout(groups_inner)
        self._cmp_groups_layout.setContentsMargins(0, 0, 0, 0)
        self._cmp_groups_layout.setSpacing(6)
        gc.addWidget(groups_inner)

        self.btn_cmp_add_group = QtWidgets.QPushButton("+ Add group")
        self.btn_cmp_add_group.clicked.connect(self._cmp_add_group)
        gc.addWidget(self.btn_cmp_add_group)
        gc.addStretch(1)

        self._cmp_group_cards: list[_CompareGroupCard] = []

        # Primary action — kept in the bottom-left action stack for app-wide
        # consistency (re-parented in `_build_remaining_sidebar_pages`).
        self.btn_cmp_run = QtWidgets.QPushButton("Generate comparison")
        self.btn_cmp_run.setObjectName("primary")  # accent-fill QSS rule (blue)
        self.btn_cmp_run.setMinimumHeight(36)
        _cf = self.btn_cmp_run.font(); _cf.setBold(True); _cf.setPointSize(13)
        self.btn_cmp_run.setFont(_cf)
        self.btn_cmp_run.clicked.connect(self._on_run_clicked)

        # ── Centre: the 'Analysis Configuration' wizard. ──
        centre = self._build_stats_centre()
        v.addWidget(centre, stretch=1)

        # Seed the two default groups AFTER the centre exists so the live
        # `changed → _refresh_stats_preview` wiring has somewhere to render.
        self._cmp_add_group()
        self._cmp_add_group()

        # The status widgets (stage label, progress bar, results panel) are
        # still constructed + parented to the tab so the run-machinery can
        # call .setText / .setValue / .reset / .show_results on them — but
        # hidden, since progress is surfaced via the status bar instead.
        self.cmp_stage_label = QtWidgets.QLabel("Idle", tab)
        self.cmp_progress    = QtWidgets.QProgressBar(tab)
        self.cmp_progress.setRange(0, 100)
        self.cmp_results     = _ResultsPanel("", parent=tab)
        for w in (self.cmp_stage_label, self.cmp_progress, self.cmp_results):
            w.hide()

        self.tabs.addTab(tab, TAB_COMPARE)
        # Now that both the sidebar group cards and the centre wizard exist,
        # render the initial test plan.
        self._refresh_stats_preview()

    def _build_results_tab(self):
        """The interactive Results tab — populated after a comparison (or via
        'Open a previous comparison…').  Just hosts the _ResultsView; all the
        rendering lives in firefly.ui.ui_results."""
        from firefly.ui.ui_results import _ResultsView
        self._results_view = _ResultsView()
        self._results_view.set_open_previous_callback(
            self._open_previous_comparison)
        self.tabs.addTab(self._results_view, TAB_RESULTS)

    # ── Statistics config (the Compare-tab "Analysis Configuration" wizard) ──
    _STAT_CORR_MAP   = {"None": "none", "Bonferroni": "bonferroni",
                        "Holm": "holm", "Benjamini-Hochberg (FDR)": "fdr_bh",
                        "Šidák": "sidak", "Hochberg": "hochberg"}
    _STAT_STRAT_MAP  = {"Auto (normality test)": "auto",
                        "Force parametric": "force_parametric",
                        "Force non-parametric": "force_nonparametric"}
    _STAT_ANOVA_MAP  = {"Welch's ANOVA": "welch", "One-way ANOVA": "oneway",
                        "Auto": "auto"}
    _STAT_NONPARAM_MAP = {"Mann-Whitney U": "mann_whitney",
                          "Brunner-Munzel": "brunner_munzel",
                          "Permutation": "permutation"}
    _STAT_POSTHOC_MAP  = {"Auto (pairwise)": "auto", "Games-Howell": "games_howell",
                          "Dunn": "dunn", "Tukey HSD": "tukey"}
    # Scalar metrics shown in the preview (label · whether it's in the
    # across-metric family).
    _STAT_PREVIEW_METRICS = [
        ("MSD area-under-curve",          "auc_msd"),
        ("Mobile / immobile ratio",       "mob_immob_ratio"),
        ("Median D",                      "median_D"),
        ("Median α  (mobile tracks only)", "median_alpha"),
        ("Mean track length",             "mean_track_length_s"),
        ("Track count",                   "n_tracks"),
        ("Population heterogeneity (α₂)", "nongauss_alpha2"),
        ("Directional persistence (VACF)", "vacf_persistence"),
    ]

    def _build_stats_centre(self):
        """Build the Compare tab's CENTRE — a wizard-style 'Analysis
        Configuration' panel (returns a QScrollArea to use as the tab body).

        It reads the groups defined in the LEFT SIDEBAR and shows, live: the
        detected experimental design, a data-aware recommendation, the
        test-choosing options (each label carrying a ⓘ plain-English
        definition), the resulting plain-language test plan, and a decision
        diagram.  Every choice is mirrored on the figure captions, the stats
        CSV and the PDF report."""
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(body)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(12)

        def _card(n, title):
            """A numbered wizard-step card: an accent step badge + bold title
            above a content area.  Returns (card, content_layout)."""
            card = QtWidgets.QFrame()
            card.setObjectName("wizard_card")
            cl = QtWidgets.QVBoxLayout(card)
            cl.setContentsMargins(12, 10, 12, 12)
            cl.setSpacing(8)
            hdr = QtWidgets.QHBoxLayout()
            hdr.setContentsMargins(0, 0, 0, 0)
            hdr.setSpacing(8)
            hdr.addWidget(_step_badge(n))
            t = QtWidgets.QLabel(title)
            _tf = t.font(); _tf.setBold(True); _tf.setPointSize(12); t.setFont(_tf)
            hdr.addWidget(t)
            hdr.addStretch(1)
            cl.addLayout(hdr)
            return card, cl

        # ── Header row: title + live run-readiness badge ──
        head = QtWidgets.QHBoxLayout()
        head.setSpacing(10)
        _title = QtWidgets.QLabel("Analysis Configuration")
        _htf = _title.font(); _htf.setBold(True); _htf.setPointSize(16)
        _title.setFont(_htf)
        head.addWidget(_title)
        head.addStretch(1)
        self._stats_status_badge = _StatusBadge()
        head.addWidget(self._stats_status_badge)
        v.addLayout(head)
        intro = QtWidgets.QLabel(
            "Define your groups in the left sidebar — this panel shows how "
            "they'll be compared and lets you choose the tests. Tests use "
            "<b>one value per cell / replicate</b> (never pooled per-track).")
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {_THEME['TXT_MUTED']};")
        v.addWidget(intro)

        # ── 1 · Experimental design — colour chips + a paired/unpaired note ──
        card1, c1 = _card(1, "Your experimental design")
        self._stats_design_chips = QtWidgets.QWidget()
        self._stats_design_grid = QtWidgets.QGridLayout(self._stats_design_chips)
        self._stats_design_grid.setContentsMargins(0, 0, 0, 0)
        self._stats_design_grid.setHorizontalSpacing(6)
        self._stats_design_grid.setVerticalSpacing(6)
        self._stats_design_grid.setColumnStretch(3, 1)   # keep chips left-packed
        c1.addWidget(self._stats_design_chips)
        self._stats_design_note = QtWidgets.QLabel("—")
        self._stats_design_note.setWordWrap(True)
        self._stats_design_note.setTextFormat(Qt.TextFormat.RichText)
        c1.addWidget(self._stats_design_note)
        v.addWidget(card1)

        # ── 2 · Recommendation — severity banners + one-click apply ──
        card2, c2 = _card(2, "Recommended for your data")
        self._stats_banner_host = QtWidgets.QWidget()
        self._stats_banner_layout = QtWidgets.QVBoxLayout(self._stats_banner_host)
        self._stats_banner_layout.setContentsMargins(0, 0, 0, 0)
        self._stats_banner_layout.setSpacing(6)
        c2.addWidget(self._stats_banner_host)
        _rrow = QtWidgets.QHBoxLayout()
        _rrow.addStretch(1)
        self.btn_stats_apply_rec = QtWidgets.QPushButton(
            "Apply recommended settings")
        self.btn_stats_apply_rec.setToolTip(
            "Set the options below to the recommended values for the current "
            "groups. You can still adjust them afterwards.")
        self.btn_stats_apply_rec.clicked.connect(self._apply_stats_recommendation)
        _rrow.addWidget(self.btn_stats_apply_rec)
        c2.addLayout(_rrow)
        v.addWidget(card2)
        self._stats_recommended_cfg = None

        # ── 3 · Options — the test-choosing controls.  Each row label carries
        # a ⓘ icon defining the term in one plain-English sentence. ──
        card3, c3 = _card(3, "Analysis options")
        self._stats_settings_widget = QtWidgets.QWidget()
        sw = QtWidgets.QVBoxLayout(self._stats_settings_widget)
        sw.setContentsMargins(0, 0, 0, 0)
        sw.setSpacing(6)
        gl = QtWidgets.QFormLayout()
        gl.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.setHorizontalSpacing(10)
        gl.setVerticalSpacing(7)
        # Keep value controls at their natural width (not stretched edge-to-edge).
        gl.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        _CTRL_W = 240

        self.s_stat_alpha = self._spin_dbl(
            0.05, 0.0001, 0.5, step=0.005, decimals=4,
            tip="Significance level α. A comparison is 'significant' when its\n"
                "(corrected) p-value < α. The ***/**/* tiers stay at 0.001/0.01/0.05.")
        self.s_stat_alpha.setMaximumWidth(_CTRL_W)
        gl.addRow(_label_with_info("Significance α", "Significance α"),
                  self.s_stat_alpha)

        self.c_stat_correction = _QuietComboBox()
        self.c_stat_correction.addItems(
            ["None", "Bonferroni", "Holm", "Benjamini-Hochberg (FDR)",
             "Šidák", "Hochberg"])
        self.c_stat_correction.setCurrentText("Holm")
        self.c_stat_correction.setMaximumWidth(_CTRL_W)
        self.c_stat_correction.setToolTip(
            "Multiple-comparison correction applied to the pairwise tests.\n"
            "Holm is a uniformly more powerful step-down version of Bonferroni;\n"
            "Benjamini-Hochberg controls the false-discovery rate instead.")
        gl.addRow(_label_with_info("Correction", "Correction"),
                  self.c_stat_correction)

        self.c_stat_across_metric = QtWidgets.QCheckBox(
            "Across the 8 scalar metrics too")
        self.c_stat_across_metric.setToolTip(
            "Scanning all metric panels for significance inflates the\n"
            "family-wise false-positive rate. When on, the correction is also\n"
            "applied across every scalar metric's pairwise tests, reported in\n"
            "the CSV and reflected in the on-figure stars.")
        gl.addRow(_label_with_info("Family-wise", "Family-wise"),
                  self.c_stat_across_metric)

        self.c_stat_strategy = _QuietComboBox()
        self.c_stat_strategy.addItems(
            ["Auto (normality test)", "Force parametric", "Force non-parametric"])
        self.c_stat_strategy.setMaximumWidth(_CTRL_W)
        self.c_stat_strategy.setToolTip(
            "Auto: a Shapiro-Wilk normality test per metric picks parametric\n"
            "(t-test / ANOVA) vs non-parametric (Mann-Whitney / Kruskal-Wallis).\n"
            "Force the choice to keep one family across all metrics.")
        gl.addRow(_label_with_info("Parametric strategy", "Parametric strategy"),
                  self.c_stat_strategy)

        self.c_stat_nonparam = _QuietComboBox()
        self.c_stat_nonparam.addItems(
            ["Mann-Whitney U", "Brunner-Munzel", "Permutation"])
        self.c_stat_nonparam.setMaximumWidth(_CTRL_W)
        self.c_stat_nonparam.setToolTip(
            "Which non-parametric two-group test to use when the non-parametric\n"
            "branch is taken (by Auto or by Force). Brunner-Munzel tolerates\n"
            "unequal spread; Permutation makes no distributional assumption\n"
            "(best for very small n).")
        gl.addRow(_label_with_info("Non-parametric test", "Non-parametric test"),
                  self.c_stat_nonparam)

        self.c_stat_anova3 = _QuietComboBox()
        self.c_stat_anova3.addItems(["Welch's ANOVA", "One-way ANOVA", "Auto"])
        self.c_stat_anova3.setMaximumWidth(_CTRL_W)
        self.c_stat_anova3.setToolTip(
            "Parametric test for 3+ groups. Welch's ANOVA does NOT assume equal\n"
            "variances (consistent with the Welch's t-test used for 2 groups);\n"
            "one-way ANOVA does. Auto = Welch's.")
        gl.addRow(_label_with_info("Test for 3+ groups", "Welch's ANOVA"),
                  self.c_stat_anova3)

        self.c_stat_posthoc = _QuietComboBox()
        self.c_stat_posthoc.addItems(
            ["Auto (pairwise)", "Games-Howell", "Dunn", "Tukey HSD"])
        self.c_stat_posthoc.setMaximumWidth(_CTRL_W)
        self.c_stat_posthoc.setToolTip(
            "Pairwise follow-up after a 3+-group omnibus test. Auto = per-pair\n"
            "t / non-parametric. Games-Howell (unequal variance) and Tukey HSD\n"
            "(equal variance) self-correct; Dunn is the rank-based follow-up\n"
            "after Kruskal-Wallis.")
        gl.addRow(_label_with_info("Post-hoc (3+ groups)", "Post-hoc test"),
                  self.c_stat_posthoc)

        self.c_stat_control = _QuietComboBox()
        self.c_stat_control.addItem("(none)")
        self.c_stat_control.setMaximumWidth(_CTRL_W)
        self.c_stat_control.setToolTip(
            "Designate a control / reference group (e.g. wild-type, untreated)\n"
            "to enable Dunnett's all-vs-control test. The list stays in sync\n"
            "with your group labels.")
        gl.addRow(_label_with_info("Control group", "Control group"),
                  self.c_stat_control)

        self.c_stat_dunnett = QtWidgets.QCheckBox(
            "Dunnett's test (every group vs the control)")
        self.c_stat_dunnett.setToolTip(
            "Compares every group to the chosen control with built-in\n"
            "family-wise control — fewer, more powerful comparisons than\n"
            "all-pairs. Needs a control group set above.")
        gl.addRow(_label_with_info("Dunnett's test", "Dunnett's test"),
                  self.c_stat_dunnett)

        self.c_stat_tost = QtWidgets.QCheckBox("Report equivalence (TOST)")
        self.c_stat_tost.setToolTip(
            "Two one-sided tests: asks whether two groups are practically the\n"
            "SAME within a margin — the opposite question to a difference test.")
        gl.addRow(_label_with_info("Equivalence (TOST)", "Equivalence (TOST)"),
                  self.c_stat_tost)

        self.s_stat_tost_margin = self._spin_dbl(
            0.5, 0.05, 5.0, step=0.05, decimals=2,
            tip="Equivalence margin in pooled-SD units (works across all "
                "metrics).\nGroups count as 'equivalent' when the difference "
                "stays within\n±this many SD. 0.5 SD is a conventional 'small' "
                "margin.")
        self.s_stat_tost_margin.setMaximumWidth(_CTRL_W)
        gl.addRow(_label_with_info("Equivalence margin (SD)", "Equivalence (TOST)"),
                  self.s_stat_tost_margin)

        self.s_stat_ci = self._spin_dbl(
            0.95, 0.50, 0.999, step=0.01, decimals=3,
            tip="Confidence-interval coverage for the Hedges' g effect sizes.")
        self.s_stat_ci.setMaximumWidth(_CTRL_W)
        gl.addRow(_label_with_info("Effect-size CI", "Effect-size CI"),
                  self.s_stat_ci)

        self.c_stat_fig_corrected = QtWidgets.QCheckBox(
            "Corrected p on figure stars")
        self.c_stat_fig_corrected.setChecked(True)
        self.c_stat_fig_corrected.setToolTip(
            "On (recommended): the stars drawn on the figure use the chosen\n"
            "correction, so the figure agrees with the CSV. Off: figure shows\n"
            "raw-p stars (the corrected values are still in the CSV).")
        gl.addRow(_label_with_info("Figure stars", "Figure stars"),
                  self.c_stat_fig_corrected)
        sw.addLayout(gl)
        c3.addWidget(self._stats_settings_widget)
        self._propagate_form_tooltips(self._stats_settings_widget)
        v.addWidget(card3)

        # ── 4 · Circular-statistics options (turning-angle direction) ──
        card_circ, cC = _card(4, "Circular statistics (turning-angle direction)")
        cC.addWidget(_label_with_info("Circular outputs", "Circular outputs"))
        self.c_circ_include = QtWidgets.QCheckBox(
            "Write circular-statistics CSV + PDF outputs")
        self.c_circ_include.setChecked(True)
        self.c_circ_include.setToolTip(
            "When on, the comparison writes the circular-statistics files\n"
            "(_circular_per_group.csv, _circular_per_replicate.csv,\n"
            "_circular_tests.csv) and a circular-statistics PDF.  This is\n"
            "separate from the Turning-angle / Radial-distribution FIGURE panels\n"
            "in the sidebar — turn this off to skip the extra files while keeping\n"
            "the figure panels.")
        cC.addWidget(self.c_circ_include)

        _circ_sign = QtWidgets.QLabel(
            "<b>Sign convention:</b> turning angles are signed on (−180°, +180°]. "
            "0° = straight · +θ = left turn (CCW) · −θ = right turn (CW) · "
            "±180° = reversal.")
        _circ_sign.setWordWrap(True)
        _circ_sign.setTextFormat(Qt.TextFormat.RichText)
        _circ_sign.setStyleSheet(f"color: {_THEME['TXT_MUTED']};")
        cC.addWidget(_circ_sign)

        cC.addWidget(QtWidgets.QLabel("Between-group tests to report:"))
        self.c_circ_kappa = QtWidgets.QCheckBox(
            "Concentration κ  (is motion more tightly directed?)")
        self.c_circ_kappa.setChecked(True)
        self.c_circ_kappa.setToolTip(
            "Per-replicate test of turning-angle concentration κ between groups.\n"
            "Uses the chosen α, correction and parametric strategy above.")
        self.c_circ_rbar = QtWidgets.QCheckBox(
            "Resultant length R̄  (how concentrated the directions are)")
        self.c_circ_rbar.setChecked(True)
        self.c_circ_rbar.setToolTip(
            "Per-replicate test of the mean resultant length R̄ between groups.\n"
            "Uses the chosen α, correction and parametric strategy above.")
        self.c_circ_mu = QtWidgets.QCheckBox(
            "Mean direction μ  (Watson-Williams)")
        self.c_circ_mu.setChecked(True)
        self.c_circ_mu.setToolTip(
            "Watson-Williams F-test on per-replicate mean directions — do groups\n"
            "point, on average, in different directions?  Reuses your chosen α.")
        self.c_circ_circlin = QtWidgets.QCheckBox(
            "Circular-linear correlation  (turning angle vs D)")
        self.c_circ_circlin.setChecked(True)
        self.c_circ_circlin.setToolTip(
            "Per-group correlation between a track's average turning bias and its\n"
            "diffusion coefficient.")
        for _cb in (self.c_circ_kappa, self.c_circ_rbar, self.c_circ_mu,
                    self.c_circ_circlin):
            cC.addWidget(_cb)

        _circ_cap = QtWidgets.QLabel(
            "<i>Circular tests run per replicate (each movie / cell = one data "
            "point) and reuse the α + correction set above, so they agree with "
            "the scalar results. Pooled-angle tests on every localisation are "
            "deliberately omitted — at ~10⁵ angles they return p ≈ 0 regardless "
            "of effect (pseudoreplication).</i>")
        _circ_cap.setWordWrap(True)
        _circ_cap.setStyleSheet(f"color: {_THEME['TXT_MUTED']};")
        cC.addWidget(_circ_cap)
        v.addWidget(card_circ)

        # ── 5 · The resulting plain-language test plan. ──
        card4, c4 = _card(5, "Tests that will run")
        self.lbl_stats_plan = QtWidgets.QLabel("—")
        self.lbl_stats_plan.setWordWrap(True)
        self.lbl_stats_plan.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_stats_plan.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        c4.addWidget(self.lbl_stats_plan)
        v.addWidget(card4)

        # ── 6 · Native decision diagram (live; crisp; on-theme). ──
        card5, c5 = _card(6, "How the test is chosen")
        self._stats_diagram = _DecisionDiagram()
        c5.addWidget(self._stats_diagram)
        v.addWidget(card5)

        # ── Glossary — inline term — definition rows (collapsed by default). ──
        sec_terms = _CollapsibleSection("What these terms mean")
        terms_w = QtWidgets.QWidget()
        tgl = QtWidgets.QVBoxLayout(terms_w)
        tgl.setContentsMargins(0, 0, 0, 0)
        tgl.setSpacing(4)
        from firefly.analysis.fa_stats_config import glossary_def as _gdef
        _key_terms = [
            "Parametric", "Welch's t-test", "Mann–Whitney", "Welch's ANOVA",
            "Holm", "Benjamini–Hochberg FDR", "Hedges' g", "Replicate",
            "Two-way mixed ANOVA", "Sphericity", "Greenhouse–Geisser",
            "Interaction effect",
            # Alternative tests / post-hocs / robust effect sizes / equivalence
            "Non-parametric test", "Brunner–Munzel", "Permutation test",
            "Post-hoc test", "Games–Howell", "Dunn's test", "Tukey HSD",
            "Dunnett's test", "Control group", "Šidák", "Hochberg",
            "Cliff's delta", "Rank-biserial", "Omnibus effect size",
            "Equivalence (TOST)",
            # Circular (turning-angle) statistics
            "Turning angle", "Sign convention", "Concentration κ",
            "Mean resultant length R̄", "Watson-Williams",
            "Circular-linear correlation", "Rayleigh test",
            "Directional persistence (VACF)",
        ]
        for _term in _key_terms:
            _row = QtWidgets.QLabel(f"<b>{_term}</b> — {_gdef(_term)}")
            _row.setWordWrap(True)
            _row.setTextFormat(Qt.TextFormat.RichText)
            tgl.addWidget(_row)
        sec_terms.content_layout.addWidget(terms_w)
        sec_terms.set_expanded(False)
        v.addWidget(sec_terms)

        cap = QtWidgets.QLabel(
            "<i>The same test applies to every scalar metric — it depends on the "
            "group structure, not the metric. With “Auto”, the parametric vs "
            "non-parametric choice is made per metric from a Shapiro-Wilk "
            "normality test at run time. Comparisons with fewer than 3 "
            "replicates per group are reported but not starred.</i>")
        cap.setWordWrap(True)
        cap.setStyleSheet(f"color: {_THEME['TXT_MUTED']};")
        v.addWidget(cap)
        v.addStretch(1)

        for _w in (self.s_stat_alpha, self.s_stat_ci, self.s_stat_tost_margin):
            _w.valueChanged.connect(lambda *_: self._refresh_stats_preview())
        for _w in (self.c_stat_correction, self.c_stat_strategy, self.c_stat_anova3,
                   self.c_stat_nonparam, self.c_stat_posthoc, self.c_stat_control):
            _w.currentIndexChanged.connect(lambda *_: self._refresh_stats_preview())
        for _w in (self.c_stat_across_metric, self.c_stat_fig_corrected,
                   self.c_stat_dunnett, self.c_stat_tost,
                   self.c_circ_include, self.c_circ_kappa, self.c_circ_rbar,
                   self.c_circ_mu, self.c_circ_circlin):
            _w.toggled.connect(lambda *_: self._refresh_stats_preview())

        scroll.setWidget(body)
        return scroll

    def _collect_stats_config(self) -> dict:
        """Read the Statistics-tab widgets into the canonical stats_config dict
        (see fa_stats_config)."""
        return {
            "alpha":      float(self.s_stat_alpha.value()),
            "correction": self._STAT_CORR_MAP.get(
                self.c_stat_correction.currentText(), "holm"),
            "across_metric_correction": bool(self.c_stat_across_metric.isChecked()),
            "parametric_strategy": self._STAT_STRAT_MAP.get(
                self.c_stat_strategy.currentText(), "auto"),
            "anova3plus": self._STAT_ANOVA_MAP.get(
                self.c_stat_anova3.currentText(), "welch"),
            "nonparametric_test": self._STAT_NONPARAM_MAP.get(
                self.c_stat_nonparam.currentText(), "mann_whitney"),
            "posthoc": self._STAT_POSTHOC_MAP.get(
                self.c_stat_posthoc.currentText(), "auto"),
            "control_group": ("" if self.c_stat_control.currentText() == "(none)"
                              else self.c_stat_control.currentText()),
            "dunnett": bool(self.c_stat_dunnett.isChecked()),
            "equivalence_tost": bool(self.c_stat_tost.isChecked()),
            "tost_margin": float(self.s_stat_tost_margin.value()),
            "ci_level": float(self.s_stat_ci.value()),
            "figure_stars_use_corrected": bool(self.c_stat_fig_corrected.isChecked()),
            # Circular (turning-angle) statistics — reuse the alpha/correction
            # above; these only gate which circular outputs/tests are produced.
            "include_circular_outputs": bool(self.c_circ_include.isChecked()),
            "circ_test_kappa":   bool(self.c_circ_kappa.isChecked()),
            "circ_test_rbar":    bool(self.c_circ_rbar.isChecked()),
            "circ_test_mu":      bool(self.c_circ_mu.isChecked()),
            "circ_test_circlin": bool(self.c_circ_circlin.isChecked()),
        }

    def _refresh_stats_preview(self):
        """Update the wizard (design chips, recommendation banners, status badge,
        test plan, decision diagram) from the current config + the group /
        time-point structure in the sidebar.  Branch selection only — it can't
        know normality (that needs the data), so 'Auto' shows both."""
        if not hasattr(self, "_stats_diagram"):
            return
        try:
            cfg = self._collect_stats_config()
        except Exception:
            return
        from firefly.analysis.fa_stats_config import describe_test_label
        try:
            from firefly.analysis.fa_twoway import HAVE_PINGOUIN as _have_pg
        except Exception:
            _have_pg = False

        # Group / time-point structure from the Compare cards.
        cards = []
        for c in getattr(self, "_cmp_group_cards", []) or []:
            try:
                st = c.get_state()
            except Exception:
                continue
            if st.get("folders"):
                cards.append(st)
        labels = list(dict.fromkeys(str(c.get("label", "")) for c in cards))
        n_groups = len(labels)
        tps = list(dict.fromkeys(str(c.get("timepoint", "")).strip()
                                 for c in cards
                                 if str(c.get("timepoint", "")).strip()))
        # Paired (two-factor) needs ≥2 DISTINCT time points; a single shared
        # time point is treated as a plain one-factor comparison (matches
        # compare_groups).
        paired = len(tps) >= 2
        strat = cfg["parametric_strategy"]
        _NP_NAME = {"mann_whitney": "Mann-Whitney U",
                    "brunner_munzel": "Brunner-Munzel",
                    "permutation": "permutation test"}
        _np = _NP_NAME.get(cfg["nonparametric_test"], "Mann-Whitney U")
        _PH_NAME = {"games_howell": "Games-Howell", "dunn": "Dunn",
                    "tukey": "Tukey HSD"}

        def _two_group():
            if strat == "force_parametric":     return "Welch's t-test"
            if strat == "force_nonparametric":  return _np
            return f"Welch's t-test <i>or</i> {_np} <i>(auto)</i>"

        def _paired_change():
            if strat == "force_parametric":     return "paired t-test"
            if strat == "force_nonparametric":  return "Wilcoxon signed-rank"
            return "paired t-test <i>or</i> Wilcoxon <i>(auto)</i>"

        def _n_group():
            para = {"welch": "Welch's ANOVA", "oneway": "one-way ANOVA",
                    "auto": "Welch's ANOVA"}[cfg["anova3plus"]]
            if n_groups == 2:
                return _two_group()
            if strat == "force_parametric":     return para
            if strat == "force_nonparametric":  return "Kruskal-Wallis"
            return f"{para} <i>or</i> Kruskal-Wallis <i>(auto)</i>"

        # Replicate = one analysis-output FOLDER (compare_groups computes one
        # scalar row per folder).  So a group's replicate count is the number of
        # folders it holds, summed across any cards sharing its label — NOT the
        # number of cards.  This is what drives the recommendation + readiness.
        def _reps(lbl):
            return sum(len(c.get("folders") or [])
                       for c in cards if str(c.get("label", "")) == lbl)
        group_counts = [_reps(lbl) for lbl in labels]
        min_reps = min(group_counts) if group_counts else 0

        # ── Keep the control-group combo in sync with the live group labels ───
        # (single authority — a persisted/stale label gracefully falls back to
        # "(none)", so Dunnett degrades safely rather than naming a dead group).
        try:
            combo = self.c_stat_control
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(none)")
            for lbl in labels:
                combo.addItem(str(lbl))
            combo.setCurrentText(prev if prev in
                                 ([str(x) for x in labels] + ["(none)"]) else "(none)")
            combo.blockSignals(False)
            control_label = ("" if combo.currentText() == "(none)"
                             else combo.currentText())
            # Dunnett only makes sense with a control set and ≥2 groups.
            self.c_stat_dunnett.setEnabled(bool(control_label) and n_groups >= 2)
            if not (bool(control_label) and n_groups >= 2):
                self.c_stat_dunnett.setChecked(False)
        except Exception:
            control_label = ""

        # ── 1 · Design summary: colour chips + a paired/unpaired note ─────────
        while self._stats_design_grid.count():
            _it = self._stats_design_grid.takeAt(0)
            _w = _it.widget()
            if _w is not None:
                _w.deleteLater()
        if n_groups < 2:
            self._stats_design_note.setText(
                "<i>No comparison defined yet — add at least 2 groups (each with "
                "≥1 folder) in the sidebar.</i>")
        else:
            color_by_label = {}
            for c in cards:
                color_by_label.setdefault(
                    str(c.get("label", "")), c.get("color") or _THEME["TXT_MUTED"])
            for i, lbl in enumerate(labels):
                self._stats_design_grid.addWidget(
                    _color_chip(lbl, color_by_label.get(lbl), _reps(lbl)),
                    i // 3, i % 3, Qt.AlignmentFlag.AlignLeft)
            if paired:
                note = (f"<b>Paired</b> design — {n_groups} group(s) × {len(tps)} "
                        f"time points ({', '.join(map(str, tps))}). Cells are "
                        "matched across time points by folder name.")
            elif tps:
                note = (f"<b>Unpaired</b> — every group shares one time point "
                        f"(“{tps[0]}”), so there's nothing to pair across; the "
                        "groups are compared as independent samples.")
            else:
                note = (f"<b>Unpaired</b> — {n_groups} independent group(s) "
                        "(no time points set).")
            self._stats_design_note.setText(note)

        # ── Test plan ────────────────────────────────────────────────────────
        corr = describe_test_label("", cfg["correction"],
                                   cfg["across_metric_correction"])
        ci_pct = f"{cfg['ci_level'] * 100:g}%"
        stars_src = ("corrected" if cfg["figure_stars_use_corrected"] else "raw")
        if n_groups < 2:
            self.lbl_stats_plan.setText("<i>—</i>")
        else:
            items = []
            if paired:
                pg_note = ("" if _have_pg
                           else " <span style='color:#e0673a;'>"
                                "(install <b>pingouin</b> to enable)</span>")
                items.append("<b>Overall, each metric:</b> two-way mixed ANOVA "
                             "(group × time), Greenhouse-Geisser corrected" + pg_note)
                items.append(f"<b>Between groups at each time point:</b> {_two_group()}")
                items.append(f"<b>Change across time (per group, paired):</b> {_paired_change()}")
            else:
                items.append(f"<b>Each metric ({n_groups} groups):</b> {_n_group()}")
                if n_groups > 2:
                    _ph = _PH_NAME.get(cfg["posthoc"])
                    items.append("<b>Pairwise follow-up:</b> "
                                 + (f"{_ph}" if _ph else _two_group()))
            if cfg.get("dunnett") and cfg.get("control_group"):
                items.append("<b>Vs control:</b> Dunnett's test (every group vs "
                             f"“{cfg['control_group']}”)")
            items.append(f"<b>Multiple-comparison correction:</b> {corr}")
            items.append("<b>Effect size:</b> Hedges' g, Cliff's δ "
                         f"(both with {ci_pct} CI) + rank-biserial")
            if cfg.get("equivalence_tost"):
                items.append("<b>Equivalence:</b> TOST at "
                             f"±{cfg['tost_margin']:g} SD")
            items.append(f"<b>Significance:</b> α = {cfg['alpha']:g}; "
                         f"on-figure stars use <b>{stars_src}</b> p-values")
            if cfg.get("include_circular_outputs", True):
                _circ_on = [nm for nm, key in (
                    ("κ", "circ_test_kappa"), ("R̄", "circ_test_rbar"),
                    ("μ (Watson-Williams)", "circ_test_mu"),
                    ("circ-lin", "circ_test_circlin")) if cfg.get(key, True)]
                _circ_txt = (" · ".join(_circ_on) if _circ_on
                             else "none selected")
                items.append("<b>Circular (turning-angle) tests, per replicate:"
                             f"</b> {_circ_txt} (reuse the α + correction above)")
            else:
                items.append("<b>Circular outputs:</b> off")
            html = "<ul style='margin-left:-22px;'>" + "".join(
                f"<li style='margin-bottom:4px;'>{it}</li>" for it in items) + "</ul>"
            metrics = " · ".join(disp for disp, _ in self._STAT_PREVIEW_METRICS)
            html += (f"<div style='color:{_THEME['TXT_MUTED']};'>"
                     f"<b>Scalar metrics covered:</b> {metrics}</div>")
            self.lbl_stats_plan.setText(html)

        # ── 5 · Decision diagram: compute the chosen-test text + push it ──────
        if n_groups < 2:
            res = "Define ≥2 groups"
        elif paired:
            res = "Two-way mixed ANOVA + post-hoc"
        elif n_groups == 2:
            res = {"force_parametric": "Welch's t-test",
                   "force_nonparametric": "Mann-Whitney U"}.get(
                strat, "Welch's t-test / Mann-Whitney")
        else:
            para = {"welch": "Welch's ANOVA", "oneway": "One-way ANOVA",
                    "auto": "Welch's ANOVA"}[cfg["anova3plus"]]
            res = {"force_parametric": para,
                   "force_nonparametric": "Kruskal-Wallis"}.get(
                strat, para + " / Kruskal-Wallis")
        self._stats_diagram.set_flow(n_groups, paired, strat, res)

        # ── Run-readiness badge ──────────────────────────────────────────────
        if n_groups < 2:
            self._stats_status_badge.set_state("muted", "Add 2 groups")
        elif min_reps < 3:
            self._stats_status_badge.set_state("blocked", "Need ≥3 replicates")
        else:
            self._stats_status_badge.set_state("ready", "Ready to run")

        # ── 2 · Recommendation banners (rebuild from structured advice) ───────
        while self._stats_banner_layout.count():
            _it = self._stats_banner_layout.takeAt(0)
            _w = _it.widget()
            if _w is not None:
                _w.deleteLater()
        rec_items, rec = self._stats_recommendation(
            n_groups, paired, group_counts, control_label, _have_pg)
        for _item in rec_items:
            self._stats_banner_layout.addWidget(
                _AlertBanner(_item["severity"], _item["html"]))
        self._stats_recommended_cfg = rec
        self.btn_stats_apply_rec.setEnabled(rec is not None)

    def _stats_recommendation(self, n_groups, paired, group_counts,
                              control_label="", have_pg=True):
        """Data-aware advice as a list of {severity, html} banner items, plus a
        recommended config dict (None when there's nothing to recommend yet).

        Severity drives the banner colour: danger / warn / success / info.  The
        recommendation is design/count-based (it sees the per-group replicate
        counts + control label, not the metric values themselves)."""
        if n_groups < 2:
            return ([{"severity": "muted",
                      "html": "Add at least 2 groups (each with ≥1 folder) in "
                              "the sidebar and I'll recommend settings for your "
                              "data."}], None)
        group_counts = list(group_counts or [])
        min_reps = min(group_counts) if group_counts else 0
        max_reps = max(group_counts) if group_counts else 0
        rec = {"alpha": 0.05, "correction": "holm",
               "across_metric_correction": True, "parametric_strategy": "auto",
               "anova3plus": "welch", "ci_level": 0.95,
               "figure_stars_use_corrected": True,
               "nonparametric_test": "mann_whitney", "posthoc": "auto",
               "control_group": "", "dunnett": False,
               "equivalence_tost": False, "tost_margin": 0.5}
        items = []
        # Replicate-count guidance — the dominant issue in SPT statistics.
        if min_reps < 3:
            items.append({"severity": "danger",
                "html": f"Only <b>{min_reps}</b> replicate(s) in the smallest "
                        "group — comparisons can't be interpreted (they'll be "
                        "reported but not starred). Add more cells / experiments "
                        "per group."})
            rec["parametric_strategy"] = "force_nonparametric"
        elif min_reps < 6:
            items.append({"severity": "warn",
                "html": f"Few replicates (<b>{min_reps}</b> in the smallest "
                        "group): the normality test has little power, so "
                        "parametric vs non-parametric barely differ. <b>Auto</b> "
                        "is fine; pick <b>Force non-parametric</b> for an "
                        "assumption-free test, and lean on the <b>effect size + "
                        "CI</b> rather than the p-value alone."})
        else:
            items.append({"severity": "success",
                "html": f"<b>{min_reps}+</b> replicates per group — enough for "
                        "the normality test to choose sensibly. <b>Auto</b> is a "
                        "good default."})
        # Very small n → a permutation test makes no distributional assumption.
        if 0 < min_reps <= 4:
            items.append({"severity": "info",
                "html": "Tiny groups — a <b>permutation test</b> builds the null "
                        "by reshuffling labels and assumes nothing about the "
                        "distribution; a robust choice at this n."})
            rec["nonparametric_test"] = "permutation"
        # Unbalanced design → unequal-variance-robust tests.
        if min_reps >= 1 and max_reps >= int(1.5 * max(min_reps, 1)):
            items.append({"severity": "warn",
                "html": f"Unbalanced design (n = {', '.join(map(str, group_counts))}) "
                        "— prefer <b>Welch's ANOVA</b> with <b>Games-Howell</b> "
                        "follow-ups (neither assumes equal variance or equal n), "
                        "and report effect sizes."})
            rec["anova3plus"] = "welch"
            if n_groups > 2:
                rec["posthoc"] = "games_howell"
        if paired:
            if have_pg:
                items.append({"severity": "info",
                    "html": "Paired design detected (time points set) → a "
                            "<b>two-way mixed ANOVA</b> (Greenhouse-Geisser) tests "
                            "the group × time interaction."})
            else:
                items.append({"severity": "warn",
                    "html": "Paired design detected → a <b>two-way mixed ANOVA</b> "
                            "(Greenhouse-Geisser) tests the group × time "
                            "interaction. Install <b>pingouin</b> to enable it."})
        if n_groups > 2:
            items.append({"severity": "info",
                "html": "3+ groups → <b>Welch's ANOVA</b> (robust to unequal "
                        "variances) with Holm-corrected pairwise follow-ups."})
        # Many groups → all-pairs explodes; a control-vs-all design is leaner.
        if n_groups >= 4:
            n_pairs = n_groups * (n_groups - 1) // 2
            items.append({"severity": "info",
                "html": f"{n_groups} groups means <b>{n_pairs}</b> pairwise "
                        "comparisons — keep a strong correction (Holm/Hochberg), "
                        "and if one group is a control, a <b>Dunnett</b> "
                        "all-vs-control test is fewer, more powerful comparisons."})
        # Control group set → recommend Dunnett.
        if control_label:
            items.append({"severity": "info",
                "html": f"Control group “<b>{control_label}</b>” set → "
                        "<b>Dunnett's test</b> compares every group to it with "
                        "built-in family-wise control."})
            rec["control_group"] = control_label
            rec["dunnett"] = True
        items.append({"severity": "info",
            "html": "You're comparing 8 metrics, so <b>family-wise correction "
                    "across metrics</b> is recommended to keep false positives "
                    "in check."})
        return items, rec

    def _apply_stats_recommendation(self):
        """Set the sidebar controls to the stored recommended config."""
        rec = getattr(self, "_stats_recommended_cfg", None)
        if not rec:
            return
        inv = lambda d, val: next((k for k, v in d.items() if v == val), None)
        try:
            self.s_stat_alpha.setValue(float(rec["alpha"]))
            self.s_stat_ci.setValue(float(rec["ci_level"]))
            self.c_stat_across_metric.setChecked(bool(rec["across_metric_correction"]))
            self.c_stat_fig_corrected.setChecked(bool(rec["figure_stars_use_corrected"]))
            for combo, dmap, key in (
                    (self.c_stat_correction, self._STAT_CORR_MAP, "correction"),
                    (self.c_stat_strategy, self._STAT_STRAT_MAP, "parametric_strategy"),
                    (self.c_stat_anova3, self._STAT_ANOVA_MAP, "anova3plus"),
                    (self.c_stat_nonparam, self._STAT_NONPARAM_MAP, "nonparametric_test"),
                    (self.c_stat_posthoc, self._STAT_POSTHOC_MAP, "posthoc")):
                disp = inv(dmap, rec.get(key))
                if disp is not None:
                    combo.setCurrentText(disp)
            if "tost_margin" in rec:
                self.s_stat_tost_margin.setValue(float(rec["tost_margin"]))
            if rec.get("control_group"):
                self.c_stat_control.setCurrentText(str(rec["control_group"]))
            if "dunnett" in rec:
                self.c_stat_dunnett.setChecked(bool(rec["dunnett"]))
            if "equivalence_tost" in rec:
                self.c_stat_tost.setChecked(bool(rec["equivalence_tost"]))
        except Exception:
            pass
        self._refresh_stats_preview()

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
        _ml_row.addWidget(_label_with_info("Min length:", "min track length"))
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

        # Motion-class colour scheme for the viewer.  Both options are tuned
        # for the viewer's DARK canvas: "Default" is the standard bright
        # dark-mode palette (unchanged behaviour); "Colour-blind safe" is the
        # Okabe-Ito palette — the same one the Publication figure theme uses,
        # so a colour-blind-safe export and the 3-D view agree.  (The light
        # figure palette is deliberately NOT offered here: its deep hues are
        # near-invisible on the dark canvas.)  Recolours the loaded track
        # layers — and the motion-coloured cluster overlay — live.
        _mc_row = QtWidgets.QHBoxLayout()
        _mc_row.setContentsMargins(0, 8, 0, 0)
        _mc_row.addWidget(QtWidgets.QLabel("Motion colours:"))
        self._ws_motion_colour_mode = QtWidgets.QComboBox()
        self._ws_motion_colour_mode.addItems(
            ["Default", "Colour-blind safe"])
        self._ws_motion_colour_mode.setToolTip(
            "Colour scheme for the per-class track layers and the motion-"
            "coloured cluster overlay.\n"
            "• Default — standard bright dark-mode palette\n"
            "• Colour-blind safe — Okabe-Ito, matches the "
            "Publication figure theme")
        self._ws_motion_colour_mode.currentIndexChanged.connect(
            self._ws_recolour_motion_layers)
        _mc_row.addWidget(self._ws_motion_colour_mode, 1)
        filter_v.addLayout(_mc_row)

        # Motion-class colour legend — a swatch per class in the active
        # palette, so the user can read which colour is which class without
        # opening the napari layer list.  Rebuilt when the palette changes.
        self._ws_motion_legend = QtWidgets.QWidget()
        _leg = QtWidgets.QGridLayout(self._ws_motion_legend)
        _leg.setContentsMargins(0, 4, 0, 0)
        _leg.setHorizontalSpacing(4)
        _leg.setVerticalSpacing(4)
        _leg.setColumnStretch(2, 1)
        filter_v.addWidget(self._ws_motion_legend)
        try:    self._ws_rebuild_motion_legend()
        except Exception: pass

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
        self._ws_eps_slider.setRange(5, 2000)
        self._ws_eps_slider.setValue(50)
        self._ws_eps_slider.setMinimumWidth(80)
        self._ws_eps_slider.setToolTip(
            "Maximum distance between two points for them to be considered\n"
            "neighbours (nm).  Larger values merge nearby clusters; smaller\n"
            "splits them.  Default 50 nm matches the standard sptPALM preset;\n"
            "spread-out data may need several hundred nm (use 'Suggest eps').\n"
            "Tip: use the arrow keys for fine 1 nm steps.")
        self._ws_eps_value = QtWidgets.QLabel("50 nm")
        self._ws_eps_value.setMinimumWidth(50)
        eps_row.addWidget(self._ws_eps_slider, 1)
        eps_row.addWidget(self._ws_eps_value)
        dbscan_form.addRow(_label_with_info("eps (nm)", "DBSCAN eps"), eps_w)

        self._ws_minsamp_spin = QtWidgets.QSpinBox()
        self._ws_minsamp_spin.setRange(2, 200)
        self._ws_minsamp_spin.setValue(8)
        self._ws_minsamp_spin.setToolTip(
            "Minimum number of points within eps to form a dense core.\n"
            "Higher = fewer, larger, denser clusters.  Default 8.")
        dbscan_form.addRow(_label_with_info("min samples", "min samples"),
                           self._ws_minsamp_spin)

        self.btn_ws_suggest_eps = QtWidgets.QPushButton("Suggest eps")
        self.btn_ws_suggest_eps.setToolTip(
            "Estimate a good eps from the k-distance knee (k = min samples) on\n"
            "the loaded localisations — the standard DBSCAN heuristic — then\n"
            "set the eps slider to it and re-cluster.")
        self.btn_ws_suggest_eps.clicked.connect(self._ws_suggest_eps)
        dbscan_form.addRow("", self.btn_ws_suggest_eps)

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

        self._ws_cluster_point_size = QtWidgets.QSpinBox()
        self._ws_cluster_point_size.setRange(1, 20)
        self._ws_cluster_point_size.setValue(3)
        self._ws_cluster_point_size.setToolTip(
            "Marker size (px) for the cluster-overlay points.  Increase if the\n"
            "points are hard to see when zoomed out, decrease if they merge.")
        self._ws_cluster_point_size.valueChanged.connect(
            self._ws_on_point_size_changed)
        dbscan_form.addRow("Point size", self._ws_cluster_point_size)

        self._ws_cluster_status = QtWidgets.QLabel("")
        self._ws_cluster_status.setStyleSheet("color: #888;")
        self._ws_cluster_status.setWordWrap(True)
        dbscan_form.addRow(self._ws_cluster_status)

        # Shown only when a clustering run finds nothing — a clear nudge to
        # widen eps / lower min-samples (toggled in _ws_render_cluster_layer).
        self._ws_cluster_banner = _AlertBanner(
            "warn", "No clusters found — try a larger <b>eps</b> or a lower "
                    "<b>min&nbsp;samples</b>.")
        self._ws_cluster_banner.hide()
        dbscan_form.addRow(self._ws_cluster_banner)

        # Save the live-tuned clustering back to the run as *_tuned.csv files
        # and sync the tuned params into the Analysis sidebar.  Disabled until
        # a run's clusters are loaded.
        self.btn_ws_export_clusters = QtWidgets.QPushButton(
            "Export tuned clusters…")
        self.btn_ws_export_clusters.setEnabled(False)
        self.btn_ws_export_clusters.setToolTip(
            "Save the current (live-tuned) clustering as new\n"
            "*_cluster_labels_tuned.csv / *_cluster_stats_tuned.csv next to the\n"
            "loaded run (the originals are kept), and copy the tuned eps /\n"
            "min-samples into the Analysis sidebar so a re-run uses them.")
        self.btn_ws_export_clusters.clicked.connect(
            self._ws_export_tuned_clusters)
        dbscan_form.addRow(self.btn_ws_export_clusters)

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
            "Sister TIFF":       "sister",
            "ImageJ ROI":        "imagej",
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
            # Advanced linkability override: percent → fraction; 0 = off.
            "minmass_max_false_track_rate": (
                (self.s_minmass_false_rate.value() / 100.0)
                if self.s_minmass_false_rate.value() > 0 else None),
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
            # explicit "Sister TIFF" mode is picked, so the worker
            # can auto-detect a `<base>_green.tif` next to the data and
            # prefer it over the intensity-based ROI when present.
            "roi_sister_suffix":      "_green",
            "roi_sister_autodetect":  True,
            # ImageJ ROI is now a Mode: only pair a sibling RoiSet.zip /
            # RoiSet folder / .roi when the user picked "ImageJ ROI".
            "roi_imagej_autodetect":  (roi_mode_map.get(
                                        self.c_roi_mode.currentText()) == "imagej"),
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
            "fig_traj_bg":       bool(self.c_fig_traj_bg.isChecked()),
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
