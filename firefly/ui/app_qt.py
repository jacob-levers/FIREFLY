"""
FIREFLY — Fluorescence Inference & Reconstruction Engine.

PySide6 / Qt frontend (v2.0+).  Replaced the original Tkinter UI that
shipped through v1.1.x.

Tabs:
    Run Analysis  — single-file analysis with full parameter coverage.
    Batch         — folder of files processed sequentially in one
                    subprocess (one spawn cost, N analyses).
    Compare       — N-group comparison with drag-and-drop folder loading,
                    multi-panel comparison figure + summary CSVs + PDF
                    report (output of sptpalm_analysis.compare_groups).
    Workspace     — embedded napari viewer for frame scrubbing + track
                    overlay; loads lazily on first activation.

Architecture notes:
    • All long-running analyses execute in a separate subprocess via
      multiprocessing.spawn.  The worker function lives in firefly_worker.py
      (a Qt-free module) so spawn doesn't re-import PySide6 into the child
      process — critical on Apple Silicon to keep Qt's Metal-backed window
      compositor from competing with PyTorch MPS for the unified-memory
      pool.
    • Stop button is three-stage: cooperative cancel → SIGTERM (5 s) →
      SIGKILL (8 s).  Guarantees the analysis halts within ~8 s.
    • Settings persisted via QSettings (per-user, OS-native location).
    • Crash reporter (sys.excepthook + threading.excepthook + Qt's
      qInstallMessageHandler) writes detailed text reports to the OS log
      directory and surfaces them via dialog.

NOTE on MPS environment variables (set below before any imports)
----------------------------------------------------------------
PyTorch's MPS allocator on macOS 26 / Apple M-series can leak memory across
operations even with explicit synchronize() + empty_cache() between stages.
The official mitigation is to disable the high-watermark allocator check
(PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0) and enable graceful CPU fallback for
unimplemented ops (PYTORCH_ENABLE_MPS_FALLBACK=1).  These MUST be set before
torch is imported anywhere in the process — putting them at the very top of
the entry-point module is the only reliable way.
"""
from __future__ import annotations

import os
import sys

# ── HTTPS / CA certificates ───────────────────────────────────────────────────
# Frozen PyInstaller builds don't reliably ship a usable CA store, so HTTPS
# certificate verification fails for EVERY request in the .exe — which surfaced
# as a misleading "No CUDA wheel exists" error (the wheel was present; the HEAD
# probe just failed verification) and silently broke the auto-update check.
# Point Python's default HTTPS context at certifi's bundled cacert.pem so all
# urllib HTTPS in this process verifies correctly.  Harmless from source.
try:
    import ssl as _ssl
    import certifi as _certifi
    _ca = _certifi.where()
    if _ca and os.path.isfile(_ca):
        os.environ.setdefault("SSL_CERT_FILE", _ca)
        _ssl._create_default_https_context = (
            lambda *a, **k: _ssl.create_default_context(cafile=_ca))
except Exception:
    pass

# ── CUDA sidecar injection ───────────────────────────────────────────────────
# Must run BEFORE any torch import so the CUDA-built torch in
# %LOCALAPPDATA%\FIREFLY\torch-cuda can shadow the bundled CPU build.
# A failure here (no GPU, no sidecar, permissions, etc.) must NEVER
# crash startup — fall through silently to the CPU build.
try:
    from firefly.cuda_installer import inject_sidecar_into_sys_path
    inject_sidecar_into_sys_path()
except Exception:
    pass

# ── MPS allocator tuning (must be set BEFORE torch import anywhere) ───────────
# Disable the high-watermark check so MPS aggressively reuses memory instead
# of holding committed blocks across ops.  Enable CPU fallback so missing
# MPS kernels don't kill the process.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import multiprocessing
import queue
import time
import traceback
from typing import Any

# macOS + multiprocessing: spawn is the only safe context for PyInstaller
# frozen apps, and it also gives the analysis subprocess a clean Python
# interpreter (no Qt, no Metal claim) — the whole point of running the
# heavy GPU work in a child process to avoid contention with Qt's window
# compositor on M-series Macs.
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer

# Matplotlib Qt embedding
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavToolbar

from firefly import crash_reporter
from firefly.ui.ui_mixin_handlers import HandlersMixin
from firefly.ui.ui_mixin_build import BuildMixin
from firefly.ui.ui_mixin_batch import BatchMixin
from firefly.ui.ui_mixin_compare import CompareMixin
from firefly.ui.ui_mixin_visualise import VisualiseMixin
# ui_constants extracted; re-exported here.
from firefly.ui.ui_constants import (
    TAB_IMPORT, TAB_ANALYSIS, TAB_COMPARE, TAB_VISUALISE, TAB_REPROCESS,
)
# ui_widgets extracted; re-exported here.
from firefly.ui.ui_widgets import (
    _UpdateCheckThread, _ModeTile, _ActionTile, _QuietSpinBox,
    _QuietDoubleSpinBox, _QuietComboBox, _CollapsibleSection,
    _ResourceMonitor, _MassHistogram, _LiveFrameView, _TrackInspector,
    _ResultsPanel, _RoiDialog, _load_imagej_roi_polygons,
    _load_tif_mask_polygons, _load_any_roi_file, _RoiViewer, _FolderDropList,
    _CompareGroupCard, _PreferencesDialog, _AlertBanner, _StatusBadge,
)
# ui_helpers extracted; re-exported here.
from firefly.ui.ui_helpers import (
    _MOTION_PALETTE, _MOTION_ORDER, _MOTION_CMAP_NAME,
    _register_motion_colormap, _make_cogwheel_icon, _make_close_x_icon,
    _make_napari_container_layout_opaque, _hide_napari_chrome, _open_folder,
    _qt_message_handler,
)
# ui_theme extracted; re-exported here.
from firefly.ui.ui_theme import (
    _THEMES, _pick_startup_theme, _ACTIVE_THEME_NAME, _THEME, _FIREFLY_QSS,
    _apply_firefly_theme,
)


N_CPUS = multiprocessing.cpu_count()

# Worker target lives in a *Qt-free* module so the spawned analysis
# subprocess doesn't accidentally re-import PySide6 (which would defeat
# the whole point of subprocess isolation on macOS Metal — see the
# firefly_worker.py module docstring for the full rationale).
from firefly import firefly_worker
_run_analysis_in_subprocess = firefly_worker.run_analysis
_run_batch_in_subprocess    = firefly_worker.run_batch_analysis
_run_compare_in_subprocess  = firefly_worker.run_comparison
_run_postproc_in_subprocess = firefly_worker.run_postproc

# ── Tab display names ────────────────────────────────────────────────────────
# Single source of truth so a rename touches one place, not every
# tabText(...) == "X" check.  Used by both addTab() calls and the
# string-comparison sites that drive tab-specific behaviour.


# ── Motion-class palette (single source of truth) ────────────────────────────
# Used by:
#   • Visualise tab — track-filter checkbox swatches
#   • Visualise tab — napari Tracks layer colormap (via _make_motion_colormap)
#   • Visualise tab — DBSCAN cluster overlay's motion-mode palette
# Order matches the `motion_to_int` mapping in _ws_apply_motion_filter:
#   0=Immobile, 1=Confined, 2=Brownian, 3=Directed, 4=Unknown.
# Must stay in sync with sptpalm_analysis.MC — the figure-generation
# palette.  Inspector swatches, napari layer colours, and the PDF
# figures all read from these strings, so editing them here updates
# every surface at once.






# ══════════════════════════════════════════════════════════════════════════════
#  SUBPROCESS WORKER  (defined in firefly_worker.py — DO NOT REDEFINE HERE)
# ══════════════════════════════════════════════════════════════════════════════
# The reference is bound at the top of this module (above) to
# `firefly_worker.run_analysis`.  The worker function MUST live in a
# Qt-free module — multiprocessing.spawn re-imports the module containing
# the target function in the child process, so defining it in app_qt.py
# would pull PySide6 into the subprocess, defeating the whole point of
# subprocess isolation on Apple Silicon (see the firefly_worker.py module
# docstring for the chain of causation).
#
# Old code that used to live below this comment has been removed.  If you
# need to inspect or modify the worker, see firefly_worker.py.

# ══════════════════════════════════════════════════════════════════════════════
#  MODE TILE — big segmented-control button with icon + title + subtitle
# ══════════════════════════════════════════════════════════════════════════════














# ══════════════════════════════════════════════════════════════════════════════
#  ACTION TILE — landing-page clickable card (not checkable, emits clicked)
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  SPINBOX SUBCLASSES — no up/down buttons, no scroll-wheel value changes
# ══════════════════════════════════════════════════════════════════════════════
# Two issues with Qt's default spinboxes that surfaced during user testing:
#   1. The little up/down stepper buttons on the right edge clutter the
#      look at the small sizes used in our compact parameter form.
#   2. Scrolling the mouse wheel over a spinbox silently changes the value
#      — easy to do by accident when scrolling the sidebar past a control,
#      with no visual cue that the value just changed.
#
# These subclasses fix both by setting NoButtons + AlignCenter at construction
# and ignoring wheel events.  Wheel events bubble up to the parent (the
# QScrollArea) so the user can scroll the sidebar past them as expected.






# ══════════════════════════════════════════════════════════════════════════════
#  COLLAPSIBLE SECTION — reusable accordion-style header + content panel
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS PANEL — shown after a run completes (replaces the figure canvas)
# ══════════════════════════════════════════════════════════════════════════════










# ══════════════════════════════════════════════════════════════════════════════
#  ROI DIALOG — embedded napari viewer for drawing per-file polygon ROIs
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  ImageJ / palmTRACER ROI file loader
# ══════════════════════════════════════════════════════════════════════════════






# ══════════════════════════════════════════════════════════════════════════════
#  EMBEDDED ROI VIEWER — same idea as _RoiDialog but lives in the Import tab
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARE TAB — folder-drop list + group card
# ══════════════════════════════════════════════════════════════════════════════




# ══════════════════════════════════════════════════════════════════════════════
#  PREFERENCES DIALOG
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════
class MainWindow(QtWidgets.QMainWindow, VisualiseMixin, CompareMixin, BatchMixin, BuildMixin, HandlersMixin):
    """Top-level FIREFLY window: persistent left sidebar + central QTabWidget.

    The sidebar holds analysis parameters and the Start/Stop button so they
    remain visible regardless of which tab is active (per the architecture
    spec).  The central tab widget hosts the "Run Analysis" view in B1.0 and
    will gain Batch / Compare / Workspace tabs in later phases.
    """

    # Bumped manually when a stored-setting layout changes incompatibly.
    # v2: a series of napari-grow bugs caused saved window geometries to
    #     be much wider than intended; the v2 launch invalidates those.
    SETTINGS_VERSION = 2

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FIREFLY — Fluorescence Inference & "
                            "Reconstruction Engine")
        # Pick an initial size that fits the user's actual screen.  The
        # previous unconditional resize(1280, 820) overflowed laptops
        # whose available width was < 1280, leaving the right edge of
        # the window past the desktop.  We clamp to availableGeometry()
        # with a small margin so the title bar + bottom edge stay visible.
        _scr = QtGui.QGuiApplication.primaryScreen()
        _avail = _scr.availableGeometry() if _scr is not None else None
        # Default 1240×760 — small enough to comfortably fit a 1440×900
        # laptop screen.  Clamped further to the actual available area
        # so external-monitor or smaller-screen launches don't overflow.
        _w = max(min(1240, (_avail.width()  - 40) if _avail is not None else 1240), 900)
        _h = max(min(760,  (_avail.height() - 80) if _avail is not None else 760), 600)
        self.resize(_w, _h)
        # No max-size cap on MainWindow — the napari containers are
        # sealed via `_make_napari_container_layout_opaque` (called in
        # each `_*_init_viewer`), so napari can't push us wider; the
        # window itself is freely resizable / maximisable / fullscreen.

        # QSettings stores per-user preferences in the OS-native location
        # (~/Library/Preferences on macOS, registry on Windows).  Keyed by
        # the org/app names set in main(); no extra setup needed.
        self._settings = QtCore.QSettings("jacoblevers", "FIREFLY")

        # Subprocess + queue + cancellation event; populated when Start
        # is clicked.  See _on_run_clicked.
        self._proc:         multiprocessing.Process | None = None
        self._msg_queue:    multiprocessing.Queue   | None = None
        self._cancel_event: Any                     | None = None
        # QTimer that polls the message queue at 30 Hz when a run is
        # active.  Lives the lifetime of the window.
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(33)   # ms
        self._poll_timer.timeout.connect(self._on_poll_queue)

        # Elapsed-time tracker for the Analysis tab.  1 Hz tick that
        # updates the "Elapsed: 00:32" label while a run is active.
        self._run_start_time: float | None = None
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._on_elapsed_tick)

        # Per-file polygon ROIs.  Keyed by absolute file path; each value
        # is a list of polygons, each polygon a list of (y, x) vertex
        # tuples in pixel coords.  Persisted to QSettings as JSON.
        self._roi_polygons: dict[str, list] = {}

        self._build_ui()
        self._install_menubar()
        self._install_crash_hooks()
        # First-launch CUDA-acceleration prompt (Windows + NVIDIA only;
        # no-op everywhere else).  Deferred so the main window paints
        # first, then the modal QMessageBox appears on top of it.
        QtCore.QTimer.singleShot(500, self._maybe_offer_cuda_install)
        self._load_icon()
        self._load_roi_polygons()
        self._restore_settings()
        # Initialise ROI status labels now that both settings and polygons
        # are loaded.
        try:
            self._refresh_single_roi_status()
            self._refresh_batch_roi_markers()
        except Exception:
            pass
        # Live "ready to analyse" pill on the Import tab — track the input
        # fields + mode tiles, and set the initial state from restored values.
        try:
            self.e_file.textChanged.connect(self._refresh_import_readiness)
            self.e_batch_folder.textChanged.connect(self._refresh_import_readiness)
            self.r_mode_single.toggled.connect(
                lambda *_: self._refresh_import_readiness())
            self.r_mode_batch.toggled.connect(
                lambda *_: self._refresh_import_readiness())
            self._refresh_import_readiness()
        except Exception:
            pass
        # ROI viewer loading is now EXPLICIT — driven by the
        # "Load into ROI viewer" button or a double-click on a batch
        # list item.  The earlier auto-load-on-selection design caused
        # heavy work (napari + file load) on every single mouse click,
        # which raced with the user's checkbox toggles and made some
        # files un-toggleable.
        # All we still do reactively is keep the single-file ROI status
        # label in sync.
        try:
            self.e_file.textChanged.connect(
                lambda _: self._refresh_single_roi_status())
        except Exception:
            pass

        # Reveal/hide the localisation-table controls as the input path
        # changes (a .csv/.txt/.tsv path shows them; an image hides them).
        try:
            self.e_file.textChanged.connect(
                lambda _: self._update_single_loc_controls())
        except Exception:
            pass

        # Auto-load the active file into the embedded ROI viewer whenever
        # the path settles (debounced so we don't fire load-after-every-
        # keystroke while the user is typing or pasting a path).
        self._roi_autoload_timer = QTimer(self)
        self._roi_autoload_timer.setSingleShot(True)
        self._roi_autoload_timer.setInterval(400)
        self._roi_autoload_timer.timeout.connect(
            self._roi_embedded_load_current_file)
        try:
            self.e_file.textChanged.connect(
                lambda _: self._roi_autoload_timer.start())
        except Exception:
            pass

    # ── UI construction ───────────────────────────────────────────────────


    # ── Auto-update check ─────────────────────────────────────────────────
    _UPDATE_REPO = "jacob-levers/FIREFLY"
    _UPDATE_RELEASES_URL = (
        f"https://github.com/jacob-levers/FIREFLY/releases")
    _UPDATE_API_URL = (
        f"https://api.github.com/repos/jacob-levers/FIREFLY/releases/latest")

    def _kick_off_update_check(self):
        """Hit GitHub Releases asynchronously and show the update pill
        in the header if a newer tag is available than __version__."""
        try:
            from firefly import sptpalm_analysis as _sa
            current = str(getattr(_sa, "__version__", "0.0.0"))
        except Exception:
            current = "0.0.0"

        self._update_thread = _UpdateCheckThread(
            self._UPDATE_API_URL, current, parent=self)
        self._update_thread.update_available.connect(self._on_update_available)
        self._update_thread.start()




    def _toggle_console(self):
        """Show / hide the console dock from the status-bar button."""
        if self._console_dock.isVisible():
            self._console_dock.hide()
        else:
            self._console_dock.show()
            # Force a usable size every time the dock is shown.  Qt's
            # remembered geometry from the previous show() can be a
            # 1-column strip if the user resized the splitter narrow,
            # or if the central widget grew between shows.
            try:
                area = self.dockWidgetArea(self._console_dock)
                if area in (Qt.DockWidgetArea.RightDockWidgetArea,
                            Qt.DockWidgetArea.LeftDockWidgetArea):
                    self.resizeDocks([self._console_dock], [420],
                                     Qt.Orientation.Horizontal)
                else:
                    self.resizeDocks([self._console_dock], [200],
                                     Qt.Orientation.Vertical)
            except Exception:
                pass



    # ── Tiny helpers for compact widget construction ──────────────────────
    @staticmethod
    def _make_form_section(title: str):
        """Return (CollapsibleSection, QFormLayout) for use in the sidebar.
        The form layout is already wired into the section's content; the
        caller just adds rows."""
        sec = _CollapsibleSection(title)
        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(6)
        sec.content_layout.addLayout(form)
        return sec, form

    @staticmethod
    def _propagate_form_tooltips(root: "QtWidgets.QWidget") -> None:
        """Walk every QFormLayout in `root`'s subtree and copy each
        row-widget's tooltip onto its label so hovering the SETTING
        NAME (not just the spinbox / combo) reveals the explanation.

        For rows whose field is a wrapper QWidget containing other
        widgets (e.g. the Pixel-size row's `[Override checkbox]
        [spinbox]` pair), we look for the most informative child
        tooltip instead of the wrapper's own (which is empty).
        """
        def _best_tip(widget: "QtWidgets.QWidget") -> str:
            tip = widget.toolTip() or ""
            if tip.strip():
                return tip
            # Wrapper widget — fall through to the most specific child
            # that carries a tooltip.  Prefer spinboxes / combos /
            # sliders, then any other tooltip-bearing widget.
            preferred = (QtWidgets.QAbstractSpinBox,
                         QtWidgets.QComboBox,
                         QtWidgets.QSlider,
                         QtWidgets.QCheckBox,
                         QtWidgets.QLineEdit)
            best = ""
            for cls in preferred:
                for child in widget.findChildren(cls):
                    t = (child.toolTip() or "").strip()
                    if t:
                        return t
            for child in widget.findChildren(QtWidgets.QWidget):
                t = (child.toolTip() or "").strip()
                if t and len(t) > len(best):
                    best = t
            return best

        # findChildren on a QWidget returns every QObject descendant
        # matching the type; QFormLayout is a QObject under the parent
        # widget hierarchy.
        for form in root.findChildren(QtWidgets.QFormLayout):
            try:
                rows = form.rowCount()
            except Exception:
                continue
            label_role = QtWidgets.QFormLayout.ItemRole.LabelRole
            field_role = QtWidgets.QFormLayout.ItemRole.FieldRole
            for r in range(rows):
                lbl_item = form.itemAt(r, label_role)
                fld_item = form.itemAt(r, field_role)
                if lbl_item is None or fld_item is None:
                    continue
                lbl = lbl_item.widget()
                fld = fld_item.widget()
                if lbl is None or fld is None:
                    continue
                if (lbl.toolTip() or "").strip():
                    continue   # caller already set one explicitly
                tip = _best_tip(fld)
                if tip:
                    lbl.setToolTip(tip)
                    # macOS sometimes needs a wider hover region than
                    # the label's tightly-fitted geometry; this attribute
                    # lets the label receive enter/leave events properly.
                    lbl.setAttribute(
                        Qt.WidgetAttribute.WA_AlwaysShowToolTips, True)

    @staticmethod
    def _make_vbox_section(title: str):
        """Return (CollapsibleSection, QVBoxLayout)."""
        sec = _CollapsibleSection(title)
        vb = QtWidgets.QVBoxLayout()
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(6)
        sec.content_layout.addLayout(vb)
        return sec, vb

    @staticmethod
    def _make_mode_tile(title: str, subtitle: str,
                        icon_char: str = "") -> "_ModeTile":
        """Big segmented-control tile (custom widget — see `_ModeTile`)."""
        return _ModeTile(title, subtitle, icon_char)

    @staticmethod
    def _spin_int(value: int, lo: int, hi: int, step: int = 1,
                  tip: str = "") -> "QtWidgets.QSpinBox":
        s = _QuietSpinBox()
        s.setRange(lo, hi); s.setSingleStep(step); s.setValue(value)
        if tip: s.setToolTip(tip)
        return s

    @staticmethod
    def _spin_dbl(value: float, lo: float, hi: float, step: float = 0.01,
                  decimals: int = 3, tip: str = "") -> "QtWidgets.QDoubleSpinBox":
        s = _QuietDoubleSpinBox()
        s.setRange(lo, hi); s.setSingleStep(step); s.setDecimals(decimals)
        s.setValue(value)
        if tip: s.setToolTip(tip)
        return s


    # ── Landing page (one-way gateway, not a tab) ─────────────────────────

    def _enter_main_ui(self, target_tab: str):
        """Swap the QStackedWidget from landing → main UI and activate the
        named tab.  Called once per session, on action-card click."""
        self._main_stack.setCurrentIndex(1)
        # Reveal the header cogwheel — it stays hidden on the landing
        # page (where the Settings tile is the entry point) but becomes
        # the persistent cross-tab Preferences shortcut from here on.
        if hasattr(self, "btn_header_prefs"):
            self.btn_header_prefs.show()
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == target_tab:
                self.tabs.setCurrentIndex(i)
                return

    # ── Tab-aware sidebar ────────────────────────────────────────────────


    def _open_preferences(self, focus_section: str | None = None):
        """Open the FIREFLY preferences dialog.  If `focus_section` is
        provided, the dialog opens with that left-rail row selected."""
        try:
            dlg = _PreferencesDialog(self)
            if focus_section:
                for i in range(dlg._rail.count()):
                    if dlg._rail.item(i).text() == focus_section:
                        dlg._rail.setCurrentRow(i)
                        break
            dlg.exec()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Preferences",
                f"Couldn't open preferences:\n\n{exc}")



    def _update_single_loc_controls(self):
        """Show the loc-table controls (Source preset + Background image)
        only when the single-file input is an external localisations table;
        hide them for image inputs."""
        try:
            is_loc = self._is_csv_input(self.e_file.text().strip())
            self._single_loc_panel.setVisible(bool(is_loc))
        except Exception:
            pass


    # ── Figures tab ───────────────────────────────────────────────────────

    @staticmethod
    def _figure_theme_palette(theme: str):
        """Return (BG, PNL, TXT, GRD, ACC, font) for a theme name —
        mirrors what make_figure() and compare_groups() do internally."""
        if theme == "Light":
            return ("#ffffff", "#f6f8fa", "#24292f",
                    "#d0d7de", "#0969da", "sans-serif")
        if theme == "Publication":
            return ("#ffffff", "#ffffff", "#000000",
                    "#cccccc", "#333333", "serif")
        return ("#0d1117", "#161b22", "#e6edf3",
                "#30363d", "#58a6ff", "monospace")

    def _refresh_figures_preview(self):
        """Render both single-sample and comparison previews in the user's
        chosen styles and push them into their respective QLabels.  Runs
        in-process (Agg backend) so it can't interfere with the analysis
        subprocess."""
        if not hasattr(self, "lbl_fig_preview_single"):
            return
        try:
            import io
            import numpy as np
            import matplotlib
            matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
        except Exception as exc:
            self.lbl_fig_preview_single.setText(f"Preview unavailable: {exc}")
            self.lbl_fig_preview_compare.setText(f"Preview unavailable: {exc}")
            return

        proj_cmap_name = self.c_fig_proj_cmap.currentText()

        def _rc(theme):
            BG, PNL, TXT, GRD, _ACC, font = self._figure_theme_palette(theme)
            return {
                "text.color": TXT, "axes.labelcolor": TXT,
                "xtick.color": TXT, "ytick.color": TXT,
                "axes.edgecolor": GRD, "axes.facecolor": PNL,
                "grid.color": GRD, "grid.alpha": 0.4,
                "font.family": font,
            }

        def _render(kind: str, theme: str) -> "QtGui.QPixmap | None":
            BG, PNL, TXT, GRD, ACC, _font = self._figure_theme_palette(theme)
            cmap_map = {"Inferno": "inferno", "Hot": "hot",
                        "Viridis": "viridis", "Plasma": "plasma",
                        "Greys": "Greys" if theme in ("Light", "Publication")
                                         else "Greys_r"}
            proj = cmap_map.get(proj_cmap_name, "inferno")
            rng = np.random.default_rng(0 if kind == "single" else 1)
            buf = io.BytesIO()
            try:
                with plt.rc_context(_rc(theme)):
                    if kind == "comparison":
                        fig = self._render_comparison_preview(
                            plt, np, rng, BG, PNL, TXT, GRD)
                    else:
                        fig = self._render_single_sample_preview(
                            plt, np, rng, BG, PNL, TXT, GRD, ACC, proj)
                    fig.savefig(buf, format="png", facecolor=BG, dpi=440,
                                bbox_inches="tight")
                    plt.close(fig)
            except Exception as exc:
                return None, str(exc)
            buf.seek(0)
            pix = QtGui.QPixmap()
            if not pix.loadFromData(buf.read()):
                return None, "decode failed"
            return pix, None

        for label_widget, kind, theme in (
                (self.lbl_fig_preview_single,
                 "single", self.c_fig_theme.currentText()),
                (self.lbl_fig_preview_compare,
                 "comparison", self.c_cmp_theme.currentText())):
            pix, err = _render(kind, theme)
            if pix is None:
                label_widget.setText(f"Preview render failed: {err}")
                continue
            # Cache raw, then scale once for the current label size.
            self._fig_preview_pixmaps[label_widget] = pix
            self._fit_preview_pixmap(label_widget)

    def _fit_preview_pixmap(self, label: QtWidgets.QLabel) -> None:
        """Scale the cached raw pixmap for `label` to its current size."""
        pix = self._fig_preview_pixmaps.get(label)
        if pix is None or pix.isNull():
            return
        size = label.size()
        if size.width() <= 1 or size.height() <= 1:
            return
        scaled = pix.scaled(size,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(scaled)

    def eventFilter(self, obj, event):
        # Re-scale cached preview pixmaps when their labels resize.
        if (event.type() == QtCore.QEvent.Type.Resize
                and isinstance(obj, QtWidgets.QLabel)
                and hasattr(self, "_fig_preview_pixmaps")
                and obj in self._fig_preview_pixmaps):
            self._fit_preview_pixmap(obj)
        return super().eventFilter(obj, event)

    def _render_single_sample_preview(self, plt, np, rng,
                                       BG, PNL, TXT, GRD, ACC, proj_cmap):
        """Two-panel mock-up: projection + MSD curves."""
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.2),
                                  facecolor=BG, dpi=110)
        # Panel A — fake max projection (Gaussian blob + noise)
        H = W = 48
        Y, X = np.mgrid[0:H, 0:W]
        img = (np.exp(-((X - 26)**2 + (Y - 22)**2) / 70) * 0.9
               + np.exp(-((X - 12)**2 + (Y - 30)**2) / 30) * 0.5
               + rng.random((H, W)) * 0.08)
        ax = axes[0]
        ax.set_facecolor(PNL)
        ax.imshow(img, cmap=proj_cmap)
        ax.set_title("  A   Max projection", color=TXT, loc="left",
                     fontsize=10, fontweight="bold", pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(GRD)

        # Panel B — MSD-like curves for three motion classes
        ax = axes[1]
        ax.set_facecolor(PNL)
        t = np.linspace(0.02, 1.0, 25)
        for alpha, label, col in ((1.0, "Brownian", ACC),
                                  (0.55, "Confined", "#f78166"),
                                  (1.45, "Directed", "#56d364")):
            msd = 0.05 * t**alpha + rng.normal(0, 0.004, t.size)
            ax.plot(t, msd, marker="o", markersize=3, linewidth=1.4,
                    color=col, label=label)
        ax.set_xlabel("τ (s)", fontsize=9)
        ax.set_ylabel("MSD (μm²)", fontsize=9)
        ax.set_title("  B   Ensemble MSD", color=TXT, loc="left",
                     fontsize=10, fontweight="bold", pad=6)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        leg = ax.legend(fontsize=8, frameon=False)
        for txt in leg.get_texts(): txt.set_color(TXT)
        for sp in ax.spines.values(): sp.set_edgecolor(GRD)
        fig.tight_layout()
        return fig

    def _render_comparison_preview(self, plt, np, rng, BG, PNL, TXT, GRD):
        """Two-panel mock-up resembling the Compare-tab figure: grouped
        MSD lines + bar chart for two synthetic groups."""
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.2),
                                  facecolor=BG, dpi=110)
        groups = [("Pre",  "#3b6ed8", 1.00),
                  ("Post", "#f78166", 0.70)]
        # Panel 1 — MSD per group
        ax = axes[0]
        ax.set_facecolor(PNL)
        t = np.linspace(0.02, 1.0, 25)
        for label, col, scale in groups:
            msd = 0.06 * scale * t**0.95 + rng.normal(0, 0.003, t.size)
            ax.plot(t, msd, marker="o", markersize=3, linewidth=1.6,
                    color=col, label=label)
        ax.set_xlabel("τ (s)", fontsize=9)
        ax.set_ylabel("MSD (μm²)", fontsize=9)
        ax.set_title("  Ensemble MSD", color=TXT, loc="left",
                     fontsize=10, fontweight="bold", pad=6)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        leg = ax.legend(fontsize=8, frameon=False)
        for txt in leg.get_texts(): txt.set_color(TXT)
        for sp in ax.spines.values(): sp.set_edgecolor(GRD)
        # Panel 2 — mobile fraction bar chart
        ax = axes[1]
        ax.set_facecolor(PNL)
        labels = [g[0] for g in groups]
        cols   = [g[1] for g in groups]
        vals   = [0.62, 0.41]
        errs   = [0.04, 0.05]
        ax.bar(labels, vals, yerr=errs, color=cols, edgecolor=GRD,
               capsize=5, linewidth=1.0)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Mobile fraction", fontsize=9)
        ax.set_title("  Mobile fraction", color=TXT, loc="left",
                     fontsize=10, fontweight="bold", pad=6)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        for sp in ax.spines.values(): sp.set_edgecolor(GRD)
        fig.tight_layout()
        return fig

    # ── Batch helpers (Import-tab batch sub-panel) ───────────────────────
    @staticmethod
    def _looks_like_input_file(name: str) -> bool:
        # macOS writes AppleDouble metadata sidecars next to every file
        # when copied to a non-HFS volume; they start with `._` and have
        # the same extension as the original.  These aren't real TIFFs
        # and crash the loader if we try to open them.  Drop them here
        # before the batch tree even sees them.
        if name.startswith("._"):
            return False
        n = name.lower()
        # Image stacks — the standard PALM input.
        if n.endswith(".czi") or n.endswith(".tif") or n.endswith(".tiff"):
            # palmTRACER also writes TIFF *outputs* (track-map
            # visualisations) into its `.PT` analysis folders.  Those
            # have characteristic filename markers like
            # `-Tracks-Z6-DCoef-Filtered-...-InROI-1-100.tif`.  Skip
            # them: they are not acquisitions, and treating them as
            # input would produce garbage analyses.
            if "-tracks-z" in n:
                return False
            return True
        # External localisation tables — let the user batch-process
        # CSVs / TXTs exported by FIREFLY itself or by palmTRACER /
        # Picasso / ThunderSTORM / TrackMate.  We accept .csv and .txt
        # but filter out FIREFLY's own auxiliary outputs and palmTRACER's
        # derived files (tracks / D / MSD tables) so a folder of mixed
        # outputs only shows the localisation file per dataset.
        if not (n.endswith(".csv") or n.endswith(".txt")):
            return False
        # palmTRACER's track-derived files (tracks, D, MSD) ALL contain
        # the token `trcPALMTracer` in their name, across every ROI /
        # mode variant palmTRACER emits:
        #     trcPALMTracer.txt
        #     trcPALMTracer-1-D.txt        trcPALMTracer-1-MSD.txt
        #     trcPALMTracer-2-D.txt        …
        #     trcPALMTracer-Full-D.txt     trcPALMTracer-Full-MSD.txt
        #     trcPALMTracer-AllROI-D.txt   trcPALMTracer-AllROI-MSD.txt
        # The localisation file we DO want is `locPALMTracer`, which does
        # NOT contain `trcpalmtracer`.  A single substring test cleanly
        # excludes every track-derived variant (present or future ROI
        # tags) while keeping the loc file — and works whether or not
        # FIREFLY's `<stem>_` export prefix is present.
        if "trcpalmtracer" in n:
            return False
        # FIREFLY's own auxiliary outputs (never the right input).
        FIREFLY_AUX_SUFFIXES = (
            "_run_manifest.json",
            "_diffusion_summary.csv",
            "_ensemble_msd.csv",
            "_trajectories.csv",
            "_localisations.csv",
            "_drift.csv",
            "_dwell_times.csv",
            "_turning_angles.csv",
            "_mobile_fraction.csv",
            "_cluster_labels.csv",
            "_cluster_stats.csv",
            "_postproc_input.csv",
            "_circular_statistics.csv",
        )
        if any(n.endswith(suf) for suf in FIREFLY_AUX_SUFFIXES):
            return False
        return True

    @staticmethod
    def _series_key(filename: str) -> str:
        """Return the series key — the common stem across all sibling
        files of a single acquisition.  Strips:

          * Trailing `(N)`              — ImageJ split-TIFFs
          * Trailing `-fileNNN`         — palmTRACER split-TIFFs
          * Trailing `_locPALMTracer`   — FIREFLY's own loc CSV export

        so e.g. `expt.tif`, `expt(1).tif`, `expt-file002.tif`, and
        `expt_locPALMTracer.csv` all map to the same key `expt`,
        letting the batch UI collapse them into one series row.

        Underscore-suffix sister files like `expt_green.tif` (palmTRACER's
        ROI image) are NOT stripped here — they keep their own key and
        are filtered out at a later pass in `_batch_rescan` so we don't
        accidentally hide files that just happen to end in `_green` /
        `_red` etc. with no real series companion.
        """
        import re as _re
        stem = os.path.splitext(filename)[0]
        # palmTRACER first — its `-fileNNN` always sits at the end, no
        # extra whitespace between it and the dot.
        stem = _re.sub(r"-file\d+$", "", stem, flags=_re.IGNORECASE)
        # ImageJ split-TIFF `(N)` — may have trailing whitespace before
        # the dot, hence the `\s*` allowance.
        stem = _re.sub(r"\(\d+\)\s*$", "", stem).rstrip()
        # FIREFLY's localisation-CSV export uses `<stem>_locPALMTracer.csv`
        # — strip the suffix so the CSV groups with its source stack if
        # both happen to live in the same folder, and so two re-exports
        # of the same dataset don't appear as separate batch rows.
        stem = _re.sub(r"_locpalmtracer$", "", stem, flags=_re.IGNORECASE)
        return stem

    @staticmethod
    def _is_csv_input(path: str) -> bool:
        """True if `path` is an external-localisations table the worker
        should load via load_external_locs (rather than load_file)."""
        n = path.lower()
        return n.endswith(".csv") or n.endswith(".txt") or n.endswith(".tsv")

    @staticmethod
    def _file_looks_corrupt(path: str) -> bool:
        """Cheap O(1) integrity probe run at folder-scan time.

        Catches the 'allocated but never written' failure mode we've
        seen from aborted acquisitions / interrupted copies: the file
        is full-size on disk but every byte is 0x00.  Such files pass a
        size check yet crash the loader mid-batch.

        Strategy — read at most a few KB total regardless of file size:
          • size 0                       → corrupt
          • TIFF with a bad magic number → corrupt (must be II*\\0 / MM\\0*)
          • every byte null across ~5 evenly-spaced 4 KB samples → corrupt

        Returns False (assume healthy) on any read error EXCEPT a stat
        failure — we don't want a transient FS hiccup to hide good data,
        but an unstattable file genuinely can't be opened.
        """
        try:
            size = os.path.getsize(path)
        except OSError:
            return True
        if size == 0:
            return True
        n = path.lower()
        is_tiff = n.endswith((".tif", ".tiff"))
        try:
            with open(path, "rb") as fh:
                if is_tiff:
                    head = fh.read(4)
                    # Valid TIFF byte-order marks; anything else (e.g.
                    # the all-zero header b'\\x00\\x00\\x00\\x00') is junk.
                    if head not in (b"II*\x00", b"MM\x00*"):
                        return True
                CHUNK = 4096
                N_SAMPLES = 5
                for i in range(N_SAMPLES):
                    off = int(size * i / N_SAMPLES)
                    fh.seek(off)
                    buf = fh.read(CHUNK)
                    # `.strip(b'\\x00')` is a C-level op — empty result
                    # means the whole chunk was null.  As soon as ANY
                    # sample has real content the file isn't all-null.
                    if buf.strip(b"\x00"):
                        return False
                # Every sample was null → empty allocation.
                return True
        except OSError:
            return False



    # Custom data roles for tree items
    _ROLE_PATH        = Qt.ItemDataRole.UserRole       # full file path
    _ROLE_KIND        = Qt.ItemDataRole.UserRole + 1   # "series" or "file"
    _ROLE_SERIES_KEY  = Qt.ItemDataRole.UserRole + 2   # series identifier
    _ROLE_FILE_COUNT  = Qt.ItemDataRole.UserRole + 3   # series count (series items only)


    # ── Tree iteration helpers ───────────────────────────────────────────





    def _roi_load_specific_path(self, path: str):
        """Load `path` into the embedded preview viewer.  Defers the
        actual work one event-loop tick so the UI repaints first
        (highlight / button-press feedback), then runs the heavy
        napari load.  Status is surfaced via the viewer's own status
        line so the user sees that something is happening."""
        if not (path and os.path.isfile(path)):
            return
        try:
            self.statusBar().showMessage(
                f"Loading {os.path.basename(path)} into viewer…", 2000)
        except Exception:
            pass

        def _go():
            try:
                existing = self._roi_polygons.get(os.path.abspath(path))
                self._roi_viewer.set_file(path, current_polygons=existing)
                self._push_detection_preview_params()
                self._push_roi_mask_params()
                self._roi_viewer.enable_detection_preview(True)
            except Exception as exc:
                try:
                    self.statusBar().showMessage(
                        f"Couldn't load preview: {exc}", 8000)
                except Exception:
                    pass
        QtCore.QTimer.singleShot(0, _go)







    # ── Embedded ROI viewer (Import tab) ─────────────────────────────────
    def _roi_embedded_load_current_file(self):
        """Load whichever file is currently 'active' into the embedded
        ROI viewer.  In single mode that's `e_file`; in batch mode it's
        the currently-highlighted item in the file list."""
        if not hasattr(self, "_roi_viewer"):
            return
        path = ""
        if self.r_mode_batch.isChecked():
            it = self.tree_batch_files.currentItem() \
                if hasattr(self, "tree_batch_files") else None
            if it is not None:
                path = it.data(0, self._ROLE_PATH) or ""
        else:
            path = self.e_file.text().strip()
            # An external localisations table isn't an image — don't try to
            # load it into the preview viewer (it would error / show junk).
            if self._is_csv_input(path):
                self._roi_viewer.set_file("", None)
                return
        if path and os.path.isfile(path):
            existing = self._roi_polygons.get(os.path.abspath(path))
            self._roi_viewer.set_file(path, current_polygons=existing)
            # Always push current parameters and turn the live overlay on —
            # the viewer is the only detection-preview surface now, so it
            # may as well be on whenever a file is loaded.
            self._push_detection_preview_params()
            self._push_roi_mask_params()
            self._roi_viewer.enable_detection_preview(True)
        else:
            self._roi_viewer.set_file("", None)

    def _push_roi_mask_params(self):
        """Forward the current ROI-mode settings to the embedded viewer
        so its auto/manual-threshold overlay reflects the sidebar in real
        time."""
        if not hasattr(self, "_roi_viewer"):
            return
        try:
            mode = self.c_roi_mode.currentText()
            method = self.c_roi_auto_method.currentText().lower()  # otsu/li/triangle/mean
            threshold = float(self.s_roi_threshold.value())
            # Normalise the combo label to the short tokens the viewer
            # expects: "max", "blink", "mean", or "sum".  "Blink density"
            # in particular needs to collapse to plain "blink".
            _mm_raw = self.c_roi_mask_mode.currentText().lower()
            if _mm_raw.startswith("blink"):
                mask_mode = "blink"
            elif _mm_raw.startswith("max"):
                mask_mode = "max"
            elif _mm_raw.startswith("sum"):
                mask_mode = "sum"
            else:
                mask_mode = "mean"
            bg_sigma = float(self.s_roi_bg_sigma.value())
            self._roi_viewer.set_roi_mask_params(
                mode=mode, auto_method=method,
                threshold=threshold, mask_mode=mask_mode,
                bg_sigma=bg_sigma)
        except Exception:
            pass

    def _push_detection_preview_params(self):
        """Forward the current diameter / minmass / bg settings to the
        embedded viewer.  Bg settings matter because the pipeline runs
        detection on background-subtracted, renormalised frames — the
        preview must do the same or the mass scale won't match."""
        if not hasattr(self, "_roi_viewer"):
            return
        bg_method_map = {"Uniform Filter": "uniform_filter",
                         "Rolling Ball":   "rolling_ball"}
        try:
            self._roi_viewer.set_detection_params(
                diameter=int(self.s_diameter.value()),
                minmass=float(self.s_minmass.value()),
                bg_method=bg_method_map.get(
                    self.c_bg_method.currentText(), "uniform_filter"),
                bg_radius=int(self.s_bg_radius.value()),
            )
        except Exception:
            pass


    # ── ROI mode enabled-state + embedded viewer visibility ───────────────

    # ── Per-file ROI editor ────────────────────────────────────────────────
    _ROI_MARKER = "  ◉"

    def _decorate_filename_with_roi(self, name: str, has_roi: bool) -> str:
        """Add/remove the ◉ marker on a batch file-list item name."""
        base = name[:-len(self._ROI_MARKER)] \
               if name.endswith(self._ROI_MARKER) else name
        return f"{base}{self._ROI_MARKER}" if has_roi else base

    def _refresh_batch_roi_markers(self):
        """Walk the batch tree and refresh ◉ markers.

        File children get the marker when their own path has a saved
        ROI; the series parent gets it when *any* of its files do.
        Wrapped in blockSignals so the resulting itemChanged cascade
        doesn't re-trigger the parent/child propagation logic."""
        if not hasattr(self, "tree_batch_files"):
            return
        self.tree_batch_files.blockSignals(True)
        try:
            for ser in self._batch_iter_series():
                any_roi = False
                for j in range(ser.childCount()):
                    child = ser.child(j)
                    path = child.data(0, self._ROLE_PATH)
                    has_roi = bool(self._roi_polygons.get(
                        os.path.abspath(path))) if path else False
                    any_roi = any_roi or has_roi
                    new = self._decorate_filename_with_roi(
                        child.text(0), has_roi)
                    if new != child.text(0):
                        child.setText(0, new)
                new_parent = self._decorate_filename_with_roi(
                    ser.text(0), any_roi)
                if new_parent != ser.text(0):
                    ser.setText(0, new_parent)
        finally:
            self.tree_batch_files.blockSignals(False)

    def _refresh_single_roi_status(self):
        """Update the single-file ROI status label."""
        path = self.e_file.text().strip()
        if not path:
            self.lbl_single_roi_status.setText("Pick an input file first")
        elif self._roi_polygons.get(os.path.abspath(path)):
            n = len(self._roi_polygons[os.path.abspath(path)])
            self.lbl_single_roi_status.setText(
                f"✓ Custom polygon ROI saved ({n} shape{'s' if n != 1 else ''})")
            self.lbl_single_roi_status.setStyleSheet(
                f"color: {_THEME['SUCCESS']};")
        else:
            self.lbl_single_roi_status.setText("No custom ROI for this file")
            self.lbl_single_roi_status.setStyleSheet(
                f"color: {_THEME['TXT_MUTED']};")

    def _open_roi_dialog(self, file_path: str) -> bool:
        """Open the ROI editor for `file_path`.  Returns True if the user
        saved (including saving an empty / cleared polygon), False on
        Cancel."""
        if not os.path.isfile(file_path):
            QtWidgets.QMessageBox.warning(
                self, "ROI editor", f"File not found:\n{file_path}")
            return False
        existing = self._roi_polygons.get(os.path.abspath(file_path))
        dlg = _RoiDialog(file_path, current_polygons=existing, parent=self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return False
        polys = dlg.result_polygons()
        key = os.path.abspath(file_path)
        if polys:
            self._roi_polygons[key] = polys
        else:
            self._roi_polygons.pop(key, None)
        self._save_roi_polygons()
        return True




    def _save_roi_polygons(self):
        """Persist all per-file ROIs to QSettings as a JSON blob."""
        try:
            import json
            payload = {k: v for k, v in self._roi_polygons.items()}
            self._settings.setValue("roi/polygons", json.dumps(payload))
        except Exception:
            pass

    def _load_roi_polygons(self):
        """Load all saved per-file ROIs from QSettings."""
        try:
            import json
            raw = self._settings.value("roi/polygons", type=str)
            if raw:
                data = json.loads(raw)
                # Vertices come back as lists-of-lists; that's fine
                self._roi_polygons = {k: v for k, v in data.items()
                                       if isinstance(v, list) and v}
        except Exception:
            self._roi_polygons = {}

    # ══════════════════════════════════════════════════════════════════════
    #  COMPARE TAB
    # ══════════════════════════════════════════════════════════════════════
    SINGLE_PANELS = [
        ("A", "Max projection"),
        ("B", "Trajectories"),
        ("C", "Trajectories by D"),
        ("D", "MSD curves"),
        ("E", "Diffusion coefficient distribution"),
        ("F", "Motion classification"),
        ("G", "α (anomalous exponent) distribution"),
        ("H", "Position density map"),
        ("I", "Turning-angle distribution"),
        ("J", "Mobile fraction over time"),
        ("K", "Jump-distance distribution"),
        ("L", "Cluster map (DBSCAN)"),
        ("M", "Dwell-time distribution"),
        ("N", "Moment-scaling spectrum"),
        ("O", "Radial distribution"),
    ]
    COMPARE_PANELS = [
        ("msd",            "Ensemble MSD"),
        ("auc",            "MSD AUC bars"),
        ("logd_dist",      "log10(D) distributions"),
        ("mob_immob",      "Mobile / immobile ratio"),
        ("motion_classes", "Motion-class fractions"),
        ("track_length",   "Track-length distribution"),
        ("track_count",    "Track count"),
        ("jdd",            "Jump-distance distribution"),
        ("dwell_cdf",      "Dwell-time CDF"),
        ("turning_angles", "Turning-angle distribution"),
        ("radial_dist",    "Radial distribution of |θ|"),
        ("van_hove",       "Population heterogeneity (α₂)"),
        ("vacf",           "Directional persistence (VACF)"),
    ]
    # Raised from 6 to 12: the group × time-point design needs one card per
    # (group, time point) cell, so counts climb fast (e.g. 3 groups × 3 time
    # points = 9).  Cards live in a scrollable container, so the higher cap is
    # purely a guard rail against accidental runaway, not a layout limit.
    COMPARE_MAX_GROUPS = 12






    # ══════════════════════════════════════════════════════════════════════
    #  WORKSPACE TAB  (Napari)
    # ══════════════════════════════════════════════════════════════════════

    # ────────────────────────────────────────────────────────────────────
    #  POST-PROCESSING TAB — change/add ROI on a finished analysis
    # ────────────────────────────────────────────────────────────────────
    def _init_postproc_tab(self):
        """Build the Post-Processing tab body.  Pickers + run button
        live in the left sidebar's Re-process page (see
        `_build_sidebar_page_postproc`); the tab body shows only the
        help label, the embedded ROI viewer, and the run-status label.
        """
        tab = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(tab)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(12)

        # ── Source-run picker (re-parented into the sidebar) ────────
        # The picker widgets exist here so the existing slots
        # (_postproc_pick_source) keep working untouched, but the
        # container is held as `self._pp_source_widget` for the
        # sidebar builder to host.
        self._pp_source_widget = QtWidgets.QWidget()
        src_v = QtWidgets.QVBoxLayout(self._pp_source_widget)
        src_v.setContentsMargins(0, 0, 0, 0); src_v.setSpacing(4)
        _src_hdr = QtWidgets.QHBoxLayout()
        _src_hdr.setContentsMargins(0, 0, 0, 0)
        _src_hdr.addWidget(QtWidgets.QLabel("Source run:"))
        _src_hdr.addStretch(1)
        self._pp_status_badge = _StatusBadge()
        self._pp_status_badge.set_state("muted", "No run selected")
        _src_hdr.addWidget(self._pp_status_badge)
        src_v.addLayout(_src_hdr)
        src_row = QtWidgets.QHBoxLayout()
        src_row.setContentsMargins(0, 0, 0, 0); src_row.setSpacing(6)
        self.e_postproc_src = QtWidgets.QLineEdit()
        self.e_postproc_src.setPlaceholderText(
            "Pick a FIREFLY output folder…")
        self.e_postproc_src.setReadOnly(True)
        src_row.addWidget(self.e_postproc_src, stretch=1)
        btn_pick = QtWidgets.QPushButton("Browse…")
        btn_pick.clicked.connect(self._postproc_pick_source)
        src_row.addWidget(btn_pick)
        src_v.addLayout(src_row)

        # ── Help text — a short info banner; the detail is in the tooltip ──
        help_lbl = _AlertBanner(
            "info",
            "Pick a source run, draw or load a new ROI below, then click "
            "<b>Re-run with new ROI</b>. Results are written to a new "
            "post-processing folder — your original run is untouched.")
        help_lbl.setToolTip(
            "Post-processing reloads the original localisations from "
            "{stem}_localisations.csv, applies the ROI you draw below "
            "(or load from a palmTRACER .roi / .tif file), and re-runs "
            "every downstream stage — linking, MSD, JDD, turning angles, "
            "dwell times, clustering, and figure rendering.")
        v.addWidget(help_lbl)

        # ── Embedded ROI viewer (reuses the import-tab class) ──────
        self._postproc_roi_viewer = _RoiViewer()
        v.addWidget(self._postproc_roi_viewer, stretch=1)

        # ── Run + status (re-parented into the sidebar) ────────────
        self._pp_action_widget = QtWidgets.QWidget()
        act_v = QtWidgets.QVBoxLayout(self._pp_action_widget)
        act_v.setContentsMargins(0, 0, 0, 0); act_v.setSpacing(6)
        self.btn_postproc_run = QtWidgets.QPushButton(
            "Re-run with new ROI")
        self.btn_postproc_run.setObjectName("primary")
        self.btn_postproc_run.setMinimumHeight(36)
        _f = self.btn_postproc_run.font(); _f.setBold(True); _f.setPointSize(13)
        self.btn_postproc_run.setFont(_f)
        self.btn_postproc_run.setToolTip(
            "Apply the polygon(s) drawn above to the original "
            "localisations and re-run the analysis pipeline.  Output "
            "goes to <source>_postproc{N}/ — the original run is not "
            "modified.")
        self.btn_postproc_run.clicked.connect(self._postproc_start)
        act_v.addWidget(self.btn_postproc_run)
        self._postproc_status = QtWidgets.QLabel("")
        self._postproc_status.setStyleSheet(
            f"color: {_THEME['TXT_MUTED']};")
        self._postproc_status.setWordWrap(True)
        act_v.addWidget(self._postproc_status)
        # Status label ALSO needs to appear in the tab body so the
        # user sees feedback after clicking.  Add a reference here
        # so it's reachable; the actual widget lives in the action
        # container above.

        self.tabs.addTab(tab, TAB_REPROCESS)

    def _postproc_pick_source(self):
        """Browse for an analysis run folder + auto-load a background
        image (mean projection) into the ROI viewer."""
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Pick a FIREFLY run folder",
            self.e_outdir.text() or os.path.expanduser("~"))
        if not d:
            return
        if not os.path.isdir(os.path.join(d, "firefly_extras")):
            self._pp_status_badge.set_state("blocked", "Not a FIREFLY run")
            QtWidgets.QMessageBox.warning(
                self, "Not a FIREFLY run",
                f"{os.path.basename(d)!r} doesn't contain a "
                f"firefly_extras/ subfolder.  Pick a folder that "
                f"was produced by a FIREFLY analysis.")
            return
        self.e_postproc_src.setText(d)
        self._pp_status_badge.set_state("ready", "Run loaded")
        # Try to point the embedded ROI viewer at the original input
        # file so the user has a real image to draw on.
        try:
            import json as _json
            extras = os.path.join(d, "firefly_extras")
            params_files = [f for f in os.listdir(extras)
                            if f.endswith("_params.json")]
            op = {}
            stem = None
            if params_files:
                with open(os.path.join(extras, params_files[0])) as fh:
                    op = _json.load(fh) or {}
                stem = op.get("stem") or params_files[0][:-len("_params.json")]
                input_file = op.get("input_file") or op.get("file")
                if input_file and os.path.isfile(input_file):
                    self._postproc_roi_viewer.set_file(input_file)
                    self._postproc_status.setText(
                        f"Loaded preview: {os.path.basename(input_file)}")
                    return
            # Fallback: render a 2-D histogram of the run's localisations
            # as background.  Runs produced before `input_file` was
            # persisted in params.json land here; so do runs whose source
            # file has since been moved or unmounted.
            if stem is not None:
                locs_csv = os.path.join(extras, f"{stem}_localisations.csv")
                if os.path.isfile(locs_csv):
                    try:
                        import pandas as _pd, numpy as _np
                        df = _pd.read_csv(locs_csv, usecols=["x", "y"])
                        # Histogram in pixel space — width/height come
                        # from params.json (the run's recorded image
                        # dimensions) so the ROI you draw maps 1:1 onto
                        # what the worker will re-apply.
                        W = int(op.get("width") or
                                _np.ceil(df["x"].max()) + 1)
                        H = int(op.get("height") or
                                _np.ceil(df["y"].max()) + 1)
                        hist, _, _ = _np.histogram2d(
                            df["y"].values, df["x"].values,
                            bins=(H, W), range=[[0, H], [0, W]])
                        self._postproc_roi_viewer.set_image_array(
                            hist.astype(_np.float32),
                            label=f"{stem} (localisation density)")
                        self._postproc_status.setText(
                            f"Original image unavailable — showing "
                            f"localisation-density heatmap ({len(df):,} locs).")
                        return
                    except Exception as exc:
                        self._postproc_status.setText(
                            f"Source loaded; localisation fallback failed "
                            f"({exc}).")
                        return
            self._postproc_status.setText(
                "Source loaded; couldn't find the original image or "
                "localisations CSV — pick a different run folder.")
        except Exception as exc:
            self._postproc_status.setText(
                f"Source loaded; preview unavailable ({exc}).")

    def _postproc_start(self):
        """Dispatch the post-process subprocess."""
        src = self.e_postproc_src.text().strip()
        if not src:
            QtWidgets.QMessageBox.warning(
                self, "No source run",
                "Pick a FIREFLY run folder first.")
            return
        polys = []
        try:
            shapes = self._postproc_roi_viewer._shapes_layer
            if shapes is not None and shapes.data is not None:
                polys = [list(p.tolist()) for p in shapes.data]
        except Exception:
            polys = []
        if not polys:
            QtWidgets.QMessageBox.warning(
                self, "No ROI drawn",
                "Draw at least one polygon (or load one from a .roi "
                "file) before re-running.")
            return
        params = {
            "source_folder": src,
            "new_polygons":  polys,
            "output_folder": None,    # auto-pick <src>_postproc{N}
        }
        # Refuse to start while another worker is running so the UI
        # status / poll timer don't get confused between subprocesses.
        if (getattr(self, "_proc", None) is not None
                and self._proc.is_alive()):
            QtWidgets.QMessageBox.warning(
                self, "Worker busy",
                "Another analysis is already running — wait for it to "
                "finish (or stop it) before starting a post-process "
                "run.")
            return
        self._is_batch_run = False
        self._is_compare_run = False
        self._is_postproc_run = True
        self._msg_queue    = multiprocessing.Queue(maxsize=2000)
        self._cancel_event = multiprocessing.Event()
        self._proc = multiprocessing.Process(
            target=_run_postproc_in_subprocess,
            args=(params, self._msg_queue, self._cancel_event),
            name="FIREFLY-PostProcWorker",
            daemon=False)
        self._proc.start()
        self._poll_timer.start()
        self._postproc_status.setText("Post-processing run started…")
        self.statusBar().showMessage(
            "Running post-processing…", 5000)









    # ── Per-motion-class layer builder ────────────────────────────────────


    def _attach_track_click_handler(self, layer):
        """Hook a mouse-drag callback onto the Tracks layer so clicking a
        track populates the inspector panel.  Idempotent — replaces any
        previous handler on the same layer."""
        if layer is None:
            return
        try:
            # Wipe previous callbacks we attached
            keep = [cb for cb in layer.mouse_drag_callbacks
                    if getattr(cb, "_firefly_inspector", False) is False]
            layer.mouse_drag_callbacks.clear()
            for cb in keep:
                layer.mouse_drag_callbacks.append(cb)
        except Exception:
            pass

        def _on_click(_layer, event):
            if event.type != "mouse_press":
                return
            try:
                pid = self._track_id_at(event.position)
            except Exception:
                pid = None
            if pid is None:
                return
            self._show_track_in_inspector(int(pid))
        _on_click._firefly_inspector = True
        try:
            layer.mouse_drag_callbacks.append(_on_click)
        except Exception:
            pass

    def _track_id_at(self, world_pos) -> "int | None":
        """Return the particle id of the track whose nearest localisation
        is closest to `world_pos`.

        We search ALL localisations globally rather than only the
        current frame — napari draws each track's trail across many
        past frames, so a click on a trail line will rarely land on a
        frame where that particular track has a point.  A small
        temporal penalty (~weighted) prefers hits at or near the
        current time when there's a tie, but doesn't exclude trails.
        """
        if self._ws_tracks_df is None:
            return None
        import numpy as _np
        # world_pos comes from napari's Tracks layer.  Shape depends on
        # what the viewer's current dim layout is:
        #   * (t, y, x)  — typical case, viewer has a time slider
        #   * (y, x)     — when only a Tracks layer is loaded with no
        #                  Image layer to give the viewer 3 dimensions.
        # We accept both: if there's no time component, search purely
        # spatially and don't apply the temporal tie-breaker.
        if len(world_pos) < 2:
            return None
        if len(world_pos) >= 3:
            t = float(world_pos[0])
            have_time = True
        else:
            t = 0.0
            have_time = False
        y = float(world_pos[-2])
        x = float(world_pos[-1])
        df = self._ws_tracks_df
        # Build the candidate set frame-by-frame against what napari is
        # ACTUALLY DRAWING right now, not just "any localisation of any
        # visible-layer track".  napari renders each Tracks layer's
        # trail across `[t - tail_length, t + head_length]` (and only if
        # `layer.tail` is on); a vertex outside that window is invisible
        # to the user.  Previously we ignored the time window and used
        # only a tiny temporal tie-breaker — that let clicks land on
        # tracks whose localisations are nowhere near the current frame
        # (e.g. an Immobile track at frame 800 stealing a click meant
        # for a Confined trail at frame 175 just because the Immobile
        # point sat slightly closer in xy).  Filtering by each layer's
        # current visible time-window fixes that.
        v = getattr(self, "_napari_viewer", None)
        names = getattr(self, "_ws_motion_layer_names", {}) or {}
        pids_by_cls = getattr(self, "_ws_motion_pids", {}) or {}

        # If we don't have per-class layers yet (e.g. tracks loaded but
        # rebuild hasn't fired), fall back to old global-search behaviour.
        if not names or v is None:
            xs = df["x"].values
            ys = df["y"].values
            fs = df["frame"].values
            if len(xs) == 0:
                return None
            d2 = (xs - x) ** 2 + (ys - y) ** 2
            if have_time:
                d2 = d2 + 0.01 * (fs - t) ** 2
            idx = int(_np.argmin(d2))
            sp_d2 = (xs[idx] - x) ** 2 + (ys[idx] - y) ** 2
            if sp_d2 > 256.0:
                return None
            return int(df["particle"].values[idx])

        # Resolve the viewer's current time even when world_pos doesn't
        # carry one (e.g. tracks-only viewer with no image stack).  The
        # napari viewer always has a `dims.current_step` we can read.
        if not have_time:
            try:
                t = float(v.dims.current_step[0])
                have_time = True
            except Exception:
                t = 0.0

        # Collect candidate rows from each VISIBLE per-class layer,
        # honouring that layer's own tail/head window.
        candidates_idx: list = []
        for cls, layer_name in names.items():
            try:
                lyr = v.layers[layer_name]
            except Exception:
                continue
            if not bool(getattr(lyr, "visible", True)):
                continue

            pids = pids_by_cls.get(cls, set())
            if not pids:
                continue

            # Trails-off → only the current frame's vertices are
            # rendered.  Trails-on → a window of frames around t.
            tail_on = bool(getattr(lyr, "tail", True))
            tail_len  = float(getattr(lyr, "tail_length", 30.0))
            head_len  = float(getattr(lyr, "head_length", 0.0))
            if tail_on:
                # +0.5 fudge so the boundary frames are inclusive even
                # under float rounding; clicks on the very edge of the
                # tail otherwise miss.
                t_lo = t - tail_len - 0.5
                t_hi = t + head_len + 0.5
            else:
                # No tail rendered — only the current frame counts.
                t_lo = t - 0.5
                t_hi = t + 0.5

            mask = (
                df["particle"].isin(pids).values
                & (df["frame"].values >= t_lo)
                & (df["frame"].values <= t_hi)
            )
            if mask.any():
                candidates_idx.append(_np.where(mask)[0])

        if not candidates_idx:
            return None
        all_idx = _np.concatenate(candidates_idx)

        xs = df["x"].values[all_idx]
        ys = df["y"].values[all_idx]
        # Pure spatial argmin — every candidate is already in the
        # currently-rendered time window, so no temporal penalty needed
        # (would just bias against trail tails the user can clearly see).
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        sub_idx = int(_np.argmin(d2))
        if d2[sub_idx] > 256.0:          # > 16 px in xy
            return None
        return int(df["particle"].values[all_idx[sub_idx]])

    def _show_track_in_inspector(self, particle_id: int):
        """Look up per-particle stats and push into the inspector panel."""
        if self._ws_tracks_df is None:
            return
        df = self._ws_tracks_df
        rows = df[df["particle"] == particle_id]
        if rows.empty:
            return
        kwargs: dict = {"particle_id": particle_id}
        kwargs["length"] = int(len(rows))
        kwargs["start_frame"] = int(rows["frame"].min())
        kwargs["end_frame"]   = int(rows["frame"].max())
        # Net displacement + path length in PIXELS — convert to µm if we
        # know the pixel size (overridden in the sidebar or 1.0 fallback).
        try:
            px = (float(self.s_pixel_size.value())
                  if self.c_override_px.isChecked() else 1.0)
        except AttributeError:
            px = 1.0
        try:
            import numpy as _np
            xs = rows["x"].values; ys = rows["y"].values
            if len(xs) >= 2:
                net = float(_np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])) * px
                seg = _np.hypot(_np.diff(xs), _np.diff(ys)).sum() * px
                kwargs["net_displacement_um"] = net
                kwargs["total_path_um"]       = float(seg)
                if seg > 0:
                    kwargs["straightness"] = net / float(seg)
        except Exception:
            pass
        if "mass" in rows.columns:
            try:    kwargs["mean_mass"] = float(rows["mass"].mean())
            except Exception: pass
        # Diff-summary lookups
        diff = self._ws_diff_df
        if diff is not None and "particle" in diff.columns:
            d_row = diff[diff["particle"] == particle_id]
            if not d_row.empty:
                r = d_row.iloc[0]
                if "D" in d_row.columns:
                    try:    kwargs["d"] = float(r["D"])
                    except Exception: pass
                if "alpha" in d_row.columns:
                    try:    kwargs["alpha"] = float(r["alpha"])
                    except Exception: pass
                if "motion" in d_row.columns:
                    kwargs["motion"] = str(r["motion"])
        self._ws_inspector.show_track(**kwargs)


    # ── Cluster-map loading + live re-cluster ────────────────────────────







    # ── Settings persistence ──────────────────────────────────────────────
    # ── Settings layout ───────────────────────────────────────────────────
    def _setting_specs(self):
        """Single source of truth for every persisted widget.

        Each entry is (qsettings_key, widget, caster).  Used by both
        `_save_settings` and `_restore_settings` so they can't drift out
        of sync.  Widgets whose value is a string (combo / line edit) use
        a `str` caster; spinboxes use `int` or `float`; checkboxes use
        a small lambda that handles QSettings' "true"/"false" round-trip.
        """
        def _bool_cast(v):
            if isinstance(v, bool): return v
            if isinstance(v, str):  return v.lower() in ("1", "true", "yes")
            return bool(v)

        return [
            # ── Paths ─────────────────────────────────────────────────────
            ("analysis/file",            self.e_file,            "text"),
            ("analysis/outdir",          self.e_outdir,          "text"),
            ("analysis/batch_folder",    self.e_batch_folder,    "text"),
            ("analysis/mode_batch",      self.r_mode_batch,      "check", _bool_cast),

            # ── Imaging metadata ──────────────────────────────────────────
            ("analysis/override_px",     self.c_override_px,     "check", _bool_cast),
            ("analysis/pixel_size",      self.s_pixel_size,      "spin",  float),
            ("analysis/override_fi",     self.c_override_fi,     "check", _bool_cast),
            ("analysis/frame_interval",  self.s_frame_interval,  "spin",  float),
            ("analysis/channel",         self.s_channel,         "spin",  int),

            # ── Preprocessing ─────────────────────────────────────────────
            ("analysis/bg_method",       self.c_bg_method,       "combo"),
            ("analysis/bg_radius",       self.s_bg_radius,       "spin",  int),

            # ── Detection ─────────────────────────────────────────────────
            ("analysis/diameter",        self.s_diameter,        "spin",  int),
            ("analysis/auto_minmass",    self.c_auto_minmass,    "check", _bool_cast),
            ("analysis/minmass",         self.s_minmass,         "spin",  float),
            ("analysis/minmass_sensitivity", self.c_minmass_sensitivity, "combo"),
            ("analysis/minmass_max_false_track_rate", self.s_minmass_false_rate, "spin", float),

            # ── Linking ───────────────────────────────────────────────────
            ("analysis/search_range",    self.s_search_range,    "spin",  int),
            ("analysis/memory",          self.s_memory,          "spin",  int),
            ("analysis/min_track_len",   self.s_min_track_len,   "spin",  int),
            ("analysis/max_track_len",   self.s_max_track_len,   "spin",  int),

            # ── Diffusion & motion ────────────────────────────────────────
            ("analysis/max_lagtime",     self.s_max_lagtime,     "spin",  int),
            ("analysis/n_fit",           self.s_n_fit,           "spin",  int),
            ("analysis/alpha_immobile",  self.s_alpha_immobile,  "spin",  float),
            ("analysis/alpha_confined",  self.s_alpha_confined,  "spin",  float),
            ("analysis/alpha_directed",  self.s_alpha_directed,  "spin",  float),
            ("analysis/mobile_d",        self.s_mobile_d_threshold, "spin", float),
            ("analysis/jdd_components",  self.s_jdd_components,  "spin",  int),
            ("analysis/filter_d_enable", self.c_filter_d_enabled,"check", _bool_cast),
            ("analysis/filter_d_min",    self.s_filter_d_min,    "spin",  float),
            ("analysis/filter_d_max",    self.s_filter_d_max,    "spin",  float),

            # ── ROI ───────────────────────────────────────────────────────
            ("analysis/roi_mode",        self.c_roi_mode,        "combo"),
            ("analysis/roi_auto_method", self.c_roi_auto_method, "combo"),
            ("analysis/roi_threshold",   self.s_roi_threshold,   "spin",  float),
            ("analysis/roi_mask_mode",   self.c_roi_mask_mode,   "combo"),
            ("analysis/roi_bg_sigma",    self.s_roi_bg_sigma,    "spin",  float),

            # ── Drift correction ──────────────────────────────────────────
            ("analysis/drift_correct",   self.c_drift_correct,   "check", _bool_cast),
            ("analysis/drift_segment",   self.s_drift_segment,   "spin",  int),

            # ── Clustering ────────────────────────────────────────────────
            ("analysis/cluster_eps_nm",  self.s_cluster_eps_nm,  "spin",  float),
            ("analysis/cluster_min_samples", self.s_cluster_min_samples, "spin", int),

            # ── Performance ───────────────────────────────────────────────
            ("analysis/backend",         self.c_backend,         "combo"),
            ("analysis/workers",         self.s_workers,         "spin",  int),
            ("analysis/chunk_size",      self.s_chunk_size,      "spin",  int),

            # ── Figures tab ───────────────────────────────────────────────
            ("figures/theme",            self.c_fig_theme,       "combo"),
            ("figures/proj_cmap",        self.c_fig_proj_cmap,   "combo"),
            ("figures/dpi",              self.s_fig_dpi,         "spin",  int),
            ("figures/save_pdf",         self.c_fig_save_pdf,    "check", _bool_cast),
            ("figures/per_panel",        self.c_fig_per_panel,   "check", _bool_cast),

            # ── Compare tab ───────────────────────────────────────────────
            ("compare/outdir",           self.e_cmp_outdir,      "text"),
            ("compare/stem",             self.e_cmp_stem,        "text"),
            ("compare/theme",            self.c_cmp_theme,       "combo"),
            ("compare/pdf_report",       self.c_cmp_pdf,         "check", _bool_cast),

            # ── Statistics (the Compare-tab "Analysis Configuration" wizard) ──
            ("stats/alpha",              self.s_stat_alpha,         "spin",  float),
            ("stats/correction",         self.c_stat_correction,    "combo"),
            ("stats/across_metric",      self.c_stat_across_metric, "check", _bool_cast),
            ("stats/strategy",           self.c_stat_strategy,      "combo"),
            ("stats/anova3plus",         self.c_stat_anova3,        "combo"),
            ("stats/ci_level",           self.s_stat_ci,            "spin",  float),
            ("stats/figure_stars_corrected", self.c_stat_fig_corrected, "check", _bool_cast),

            # ── Visualise tab ─────────────────────────────────────────────
            ("visualise/motion_colours", self._ws_motion_colour_mode, "combo"),
        ]

    def _restore_settings(self):
        """Restore the user's saved selections.  Best-effort — silently
        ignores any malformed values rather than failing the launch."""
        s = self._settings
        # SETTINGS_VERSION bump invalidates any saved window geometry —
        # earlier napari-grow regressions persisted oversized windows
        # to QSettings, and restoring them on launch defeats the new
        # sealed-container fix.  Clear the stale key so the explicit
        # `self.resize(_w, _h)` from __init__ wins on first v2 launch.
        try:
            stored_ver = int(s.value("settings/version", 0) or 0)
        except Exception:
            stored_ver = 0
        if stored_ver < self.SETTINGS_VERSION:
            try:
                s.remove("window/geometry")
            except Exception:
                pass
        try:
            geom = s.value("window/geometry")
            if geom is not None:
                self.restoreGeometry(geom)
        except Exception:
            pass
        # Clamp the window to the available screen rect so a saved
        # geometry from a previous session with a bigger / external
        # monitor doesn't push the window off-screen.  Leaves a small
        # margin so the title bar + bottom edge stay visible.
        try:
            screen = self.screen() or QtGui.QGuiApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                w = min(self.width(),  max(900, avail.width()  - 20))
                h = min(self.height(), max(640, avail.height() - 40))
                self.resize(w, h)
                # If the top-left is off-screen (negative or beyond the
                # edges), re-centre instead of restoring.
                x = self.x(); y = self.y()
                if (x < avail.left() or y < avail.top()
                        or x + w > avail.right()
                        or y + h > avail.bottom()):
                    self.move(avail.left() + (avail.width()  - w) // 2,
                              avail.top()  + (avail.height() - h) // 2)
        except Exception:
            pass

        # Path entries are deliberately NOT restored — every launch starts
        # with empty file / folder fields so the user always picks fresh
        # inputs.  Clear any previously-saved values from the QSettings
        # store too, so they don't linger on disk.
        _skip_paths = {"analysis/file", "analysis/outdir",
                       "analysis/batch_folder"}
        for _k in _skip_paths:
            try: s.remove(_k)
            except Exception: pass

        for spec in self._setting_specs():
            key, widget, kind = spec[0], spec[1], spec[2]
            if key in _skip_paths:
                continue
            try:
                v = s.value(key)
                if v is None or v == "":
                    continue
                if kind == "text":
                    widget.setText(str(v))
                elif kind == "combo":
                    v_str = str(v)
                    items = [widget.itemText(i)
                             for i in range(widget.count())]
                    if v_str in items:
                        widget.setCurrentText(v_str)
                    # Migration: old saved backend values were stored as
                    # internal strings ("torch-mps") but the combo now
                    # shows labels ("Torch — Apple MPS").  Translate if
                    # this widget is the backend combo and the saved
                    # value is a recognised internal value.
                    elif widget is getattr(self, "c_backend", None):
                        lbl = self._BACKEND_VALUE_TO_LABEL.get(v_str)
                        if lbl and lbl in items:
                            widget.setCurrentText(lbl)
                elif kind == "spin":
                    caster = spec[3]
                    widget.setValue(caster(v))
                elif kind == "check":
                    caster = spec[3]
                    widget.setChecked(caster(v))
            except Exception:
                pass

        # Sync derived enabled-state (auto-minmass disables minmass spin,
        # filter-D toggles the D-min/max spins) AFTER restoring values.
        try:
            _auto_mm = self.c_auto_minmass.isChecked()
            self.s_minmass.setEnabled(not _auto_mm)
            self.sld_minmass.setEnabled(not _auto_mm)
            self.c_minmass_sensitivity.setEnabled(_auto_mm)
            self.s_minmass_false_rate.setEnabled(_auto_mm)
            on = self.c_filter_d_enabled.isChecked()
            self.s_filter_d_min.setEnabled(on)
            self.s_filter_d_max.setEnabled(on)
        except Exception:
            pass

        # Import-tab mode visibility — show the right sub-panel based on
        # the restored mode_batch flag.  Also re-scan the batch folder
        # if one was previously selected so the file list re-populates.
        try:
            if self.r_mode_batch.isChecked():
                self._single_panel.hide()
                self._batch_panel.show()
            else:
                self._single_panel.show()
                self._batch_panel.hide()
            folder = self.e_batch_folder.text().strip()
            if folder and os.path.isdir(folder):
                self._batch_rescan(folder)
        except Exception:
            pass

        # Compare-tab group cards (label/color/folders) — JSON blob
        try:
            import json
            blob = s.value("compare/groups", type=str)
            if blob:
                data = json.loads(blob)
                if isinstance(data, list) and len(data) >= 2:
                    # Replace existing cards with restored ones
                    while len(self._cmp_group_cards) > 0:
                        card = self._cmp_group_cards.pop()
                        self._cmp_groups_layout.removeWidget(card)
                        card.deleteLater()
                    for i, g in enumerate(data[:self.COMPARE_MAX_GROUPS]):
                        self._cmp_add_group()
                        self._cmp_group_cards[-1].set_state(
                            g.get("label", f"Group {i+1}"),
                            g.get("color", ""),
                            g.get("folders", []),
                            g.get("timepoint", ""))
        except Exception:
            pass

        # Compare-tab panel checkbox states
        try:
            for key, cb in self._cmp_panel_checkboxes.items():
                v = s.value(f"compare/panel_{key}")
                if v is not None:
                    cb.setChecked(_bool_cast(v))
        except Exception:
            pass

        # Single-sample panel checkbox states (Figures tab)
        try:
            for key, cb in self._single_panel_checkboxes.items():
                v = s.value(f"figures/single_panel_{key}")
                if v is not None:
                    cb.setChecked(_bool_cast(v))
        except Exception:
            pass

    def _save_settings(self):
        """Write current selections to QSettings.  Called when starting a
        run and on window close."""
        s = self._settings
        s.setValue("settings/version", self.SETTINGS_VERSION)
        try:
            s.setValue("window/geometry", self.saveGeometry())
        except Exception:
            pass
        # Path entries are intentionally not persisted — see _restore_settings.
        _skip_paths = {"analysis/file", "analysis/outdir",
                       "analysis/batch_folder"}
        for spec in self._setting_specs():
            key, widget, kind = spec[0], spec[1], spec[2]
            if key in _skip_paths:
                continue
            try:
                if kind == "text":
                    s.setValue(key, widget.text())
                elif kind == "combo":
                    s.setValue(key, widget.currentText())
                elif kind == "spin":
                    s.setValue(key, widget.value())
                elif kind == "check":
                    s.setValue(key, bool(widget.isChecked()))
            except Exception:
                pass

        # Compare-tab group cards — serialised as JSON
        try:
            import json
            blob = json.dumps([c.get_state() for c in self._cmp_group_cards])
            s.setValue("compare/groups", blob)
        except Exception:
            pass

        # Compare-tab panel checkbox states
        try:
            for key, cb in self._cmp_panel_checkboxes.items():
                s.setValue(f"compare/panel_{key}", bool(cb.isChecked()))
        except Exception:
            pass

        # Single-sample panel checkbox states (Figures tab)
        try:
            for key, cb in self._single_panel_checkboxes.items():
                s.setValue(f"figures/single_panel_{key}", bool(cb.isChecked()))
        except Exception:
            pass

    def closeEvent(self, event):
        """Tear EVERYTHING down on close, and guarantee process exit.

        Symptoms before this rewrite: closing FIREFLY left the process
        hanging in the OS task manager (Windows) or the Dock (macOS)
        until force-quit.  Root causes were varied — orphan analysis
        subprocess, multiprocessing.Queue feeder thread keeping the
        parent alive on Windows, napari viewer's Vispy/Metal context
        not releasing, CUDA-install QThread still running, modal
        QMessageBox left visible, etc.  Rather than guess which one
        is responsible on any given user's machine, we shut every
        known live object down in turn AND schedule a hard-exit
        fallback so the process always terminates within ~3 s.
        """
        # 0. FIRST THING: arm an independent hard-exit watchdog on a
        # daemon OS thread.  CRITICAL — this must be scheduled BEFORE
        # any cleanup, and via threading (NOT QTimer.singleShot),
        # because:
        #   * If we used QTimer, the timer wouldn't fire after
        #     QApplication.quit() because the Qt event loop is gone.
        #   * If we put it at the end of closeEvent, anything earlier
        #     that hangs (cleanup, napari teardown, subprocess.join())
        #     never reaches the timer-arming code.
        # threading.Timer runs in its own OS thread and the daemon
        # flag means it doesn't keep the process alive on its own
        # — it just fires os._exit(0) after the wall-clock delay if
        # the natural shutdown path hasn't already finished by then.
        try:
            import threading as _th
            def _hard_exit():
                import os as _os
                _os._exit(0)
            _wd = _th.Timer(3.0, _hard_exit)
            _wd.daemon = True
            _wd.start()
        except Exception:
            pass

        # 0b. POSIX belt-and-braces: arm a kernel-level SIGALRM with the
        # default disposition (terminate).  The threading.Timer above
        # needs the GIL to run its Python callback, so if a C extension
        # (Vispy/Metal layer release on macOS is a known offender) holds
        # the GIL through its teardown, the Timer thread can't fire
        # os._exit().  SIGALRM with SIG_DFL is handled entirely by the
        # kernel — no Python code, no GIL — so the process dies even
        # when CPython is wedged inside a non-GIL-releasing C call.
        # Windows has no SIGALRM; the try/except just no-ops there and
        # the threading.Timer fallback covers it.
        try:
            import signal as _sig
            _sig.signal(_sig.SIGALRM, _sig.SIG_DFL)
            _sig.alarm(3)
        except Exception:
            pass

        # 1. Persist settings BEFORE anything that could fail.
        try:    self._save_settings()
        except Exception: crash_reporter.log_exception("failed to save settings")

        # 2. Stop all QTimers so none of them can fire during teardown.
        # Relying on Qt parent-cleanup is racy — an in-flight singleShot
        # callback after the C++ object is dealloc'd raises "wrapped C/C++
        # object has been deleted" warnings (or in some Qt builds, crashes).
        for _tname in (
            "_poll_timer", "_elapsed_timer", "_repaint_timer",
            "_figpreview_timer", "_detect_debounce",
            "_roi_autoload_timer", "_ws_dbscan_debounce",
        ):
            _t = getattr(self, _tname, None)
            if _t is None:
                continue
            try: _t.stop()
            except Exception: pass

        # 3. Cancel + terminate the analysis subprocess.
        try:
            if self._proc is not None and self._proc.is_alive():
                if self._cancel_event is not None:
                    try: self._cancel_event.set()
                    except Exception: pass
                self._proc.join(timeout=1.5)
                if self._proc.is_alive():
                    try: self._proc.terminate()
                    except Exception: pass
                    self._proc.join(timeout=0.5)
                if self._proc.is_alive():
                    # multiprocessing.Process.kill is the nuclear option
                    # (SIGKILL on POSIX, TerminateProcess on Windows).
                    try: self._proc.kill()
                    except Exception: pass
        except Exception: pass

        # 4. Drop the multiprocessing.Queue WITHOUT joining the feeder
        # thread.  On Windows the feeder thread can block process exit
        # waiting to flush pending writes to a pipe whose other end is
        # already dead — cancel_join_thread() abandons it cleanly.
        try:
            q = self._msg_queue
            if q is not None:
                try: q.cancel_join_thread()
                except Exception: pass
                try: q.close()
                except Exception: pass
        except Exception: pass
        self._msg_queue = None

        # 5. Stop any in-flight CUDA installer thread/timer.  The worker
        # may be wedged inside urlopen; we don't try to join it — daemon
        # status means it dies with the process below.
        try:
            if getattr(self, "_cuda_heartbeat", None) is not None:
                self._cuda_heartbeat.stop()
        except Exception: pass
        try:
            if getattr(self, "_cuda_thread", None) is not None:
                self._cuda_thread.quit()
                self._cuda_thread.wait(500)
        except Exception: pass
        try:
            from firefly import cuda_installer as _cu
            _cu.set_log_callback(None)
        except Exception: pass

        # 6. Close every embedded napari viewer.  On macOS the Vispy
        # backend holds a Metal CAMetalLayer that prevents Cocoa from
        # tearing down the QApplication until the layer is released —
        # calling viewer.close() (which calls window.close() →
        # Vispy.app.canvas.close()) takes care of it.
        for attr_chain in (
                ("_roi_viewer", "_viewer"),       # Import-tab preview
                ("_napari_viewer",),              # Visualise-tab viewer
        ):
            try:
                obj = self
                for a in attr_chain:
                    obj = getattr(obj, a, None)
                    if obj is None:
                        break
                if obj is not None and hasattr(obj, "close"):
                    obj.close()
            except Exception: pass

        # 7. Close any leftover modal dialogs that might still be alive
        # (CUDA install progress dialog, etc.).
        try:
            for w in QtWidgets.QApplication.topLevelWidgets():
                if w is self:
                    continue
                try: w.close()
                except Exception: pass
        except Exception: pass

        # 8. Close every matplotlib figure we created.  Live figure
        # objects hold a Qt FigureCanvas which counts as a top-level
        # window on macOS and blocks quitOnLastWindowClosed.
        try:
            import matplotlib.pyplot as _plt
            _plt.close("all")
        except Exception: pass

        super().closeEvent(event)

        # 9. Ask the QApplication to quit (covers stray top-level windows).
        # The hard-exit watchdog armed at step 0 will fire in 3 s if
        # the natural shutdown hasn't completed by then — no need for
        # any further QTimer-based fallback (which would itself
        # depend on the Qt event loop still being alive).
        try:    QtWidgets.QApplication.instance().quit()
        except Exception: pass

    # ── Backend availability helper ───────────────────────────────────────
    # Two-way mapping between GUI labels and internal backend strings.
    # The pipeline's `_resolve_backend` understands the hyphenated forms
    # (auto / trackpy / torch / torch-mps / torch-cuda / torch-cpu); the
    # GUI shows them as proper grammar so users don't see lowercase
    # snake-case-y identifiers in their face.
    _BACKEND_LABEL_TO_VALUE = {
        "Auto":               "auto",
        "Trackpy (CPU)":      "trackpy",
        "Torch (auto)":       "torch",
        "Torch — Apple MPS":  "torch-mps",
        "Torch — NVIDIA CUDA": "torch-cuda",
        "Torch — CPU":        "torch-cpu",
    }
    _BACKEND_VALUE_TO_LABEL = {v: k for k, v in _BACKEND_LABEL_TO_VALUE.items()}

    def _available_backends(self) -> list[str]:
        """Return the static list of selectable backend LABELS (display
        strings) for the dropdown.  Internal values are resolved via
        `_backend_value_from_label` before being sent to the worker.

        IMPORTANT: we deliberately do NOT probe torch here.  On some macOS
        / PyTorch / Apple-Silicon combinations, just importing torch and
        calling `torch.backends.mps.is_available()` is enough to trigger
        noisy MPS command-buffer errors on stderr and, in the worst case,
        kill the process before the GUI is fully up.  Probing happens
        lazily inside the analysis worker only when actually selected.
        """
        return list(self._BACKEND_LABEL_TO_VALUE.keys())

    def _backend_value_from_label(self, label: str) -> str:
        """Translate a dropdown label to the internal pipeline string.
        Falls through to the label itself so old saved-settings values
        (`torch-mps` etc.) still work after upgrading."""
        if label in self._BACKEND_LABEL_TO_VALUE:
            return self._BACKEND_LABEL_TO_VALUE[label]
        if label in self._BACKEND_VALUE_TO_LABEL:
            return label   # already an internal value
        return label or "auto"

    def _validate_selected_backend(self) -> bool:
        """Pre-flight check before a run.  Catches the common "user picked
        CUDA on a CPU-only torch" footgun BEFORE the analysis subprocess
        wastes minutes on frame loading + preprocessing only to crash on
        the first CUDA call with "Torch not compiled with CUDA enabled".

        Returns True if the run should proceed, False if the user
        cancelled after we surfaced the problem.  Offers a one-click
        switch to a safe fallback so the user doesn't have to navigate
        back to the dropdown.
        """
        value = self._backend_value_from_label(self.c_backend.currentText())
        if not value or not value.startswith("torch-"):
            return True
        forced = value[len("torch-"):].split(":", 1)[0]
        try:
            import torch as _t
            if forced == "cuda" and not _t.cuda.is_available():
                msg = (
                    "You picked the NVIDIA CUDA backend, but the bundled "
                    "PyTorch is CPU-only.\n\n"
                    "Options:\n"
                    "  • Cancel, then click 'Set up GPU acceleration…' in "
                    "the Performance section to install the CUDA wheel.\n"
                    "  • Or switch to a CPU/Auto backend now and continue.")
                box = QtWidgets.QMessageBox(
                    QtWidgets.QMessageBox.Icon.Warning,
                    "CUDA backend unavailable", msg,
                    QtWidgets.QMessageBox.StandardButton.NoButton, self)
                btn_switch = box.addButton(
                    "Switch to Auto and continue",
                    QtWidgets.QMessageBox.ButtonRole.AcceptRole)
                btn_cancel = box.addButton(
                    "Cancel",
                    QtWidgets.QMessageBox.ButtonRole.RejectRole)
                box.setDefaultButton(btn_switch)
                box.exec()
                if box.clickedButton() is btn_switch:
                    self.c_backend.setCurrentText("Auto")
                    return True
                return False
            if forced == "mps":
                has_mps = (hasattr(_t.backends, "mps")
                           and _t.backends.mps.is_available())
                if not has_mps:
                    QtWidgets.QMessageBox.warning(
                        self, "MPS backend unavailable",
                        "You picked the Apple MPS backend, but this system "
                        "doesn't have MPS available (needs Apple Silicon + "
                        "macOS 12+).  Switch to Auto or Torch — CPU.")
                    return False
        except Exception:
            # If torch import itself fails, let the worker emit its own
            # error — don't block the user here.
            pass
        return True

    # ── Event handlers ────────────────────────────────────────────────────




    # ── Params builder (shared by single-file and batch) ──────────────────

    def _widget_state_dict(self) -> dict:
        """Return the current sidebar widget values keyed by their QSettings
        path (e.g. 'analysis/diameter').  Mirrors `_save_settings` but
        in-memory so we can embed it in run manifests."""
        out: dict = {}
        for spec in self._setting_specs():
            key, widget, kind = spec[0], spec[1], spec[2]
            try:
                if   kind == "text":  out[key] = widget.text()
                elif kind == "combo": out[key] = widget.currentText()
                elif kind == "spin":  out[key] = widget.value()
                elif kind == "check": out[key] = bool(widget.isChecked())
            except Exception:
                pass
        # Panel-checkbox selections (single + compare) too
        try:
            out["figures/single_panels"] = [
                k for k, cb in self._single_panel_checkboxes.items()
                if cb.isChecked()]
        except AttributeError: pass
        try:
            out["compare/panels"] = [
                k for k, cb in self._cmp_panel_checkboxes.items()
                if cb.isChecked()]
        except AttributeError: pass
        return out

    def _apply_widget_state(self, state: dict) -> None:
        """Push a widget-state dict (produced by `_widget_state_dict`) back
        into the sidebar widgets.  Used by the manifest 'Replay' button."""
        if not isinstance(state, dict):
            return
        # Cast helpers (mirror _restore_settings)
        def _bool_cast(v):
            if isinstance(v, bool): return v
            if isinstance(v, str):  return v.lower() in ("1", "true", "yes")
            return bool(v)
        for spec in self._setting_specs():
            key, widget, kind = spec[0], spec[1], spec[2]
            if key not in state:
                continue
            v = state[key]
            try:
                if kind == "text":
                    widget.setText(str(v))
                elif kind == "combo":
                    items = [widget.itemText(i)
                             for i in range(widget.count())]
                    if str(v) in items:
                        widget.setCurrentText(str(v))
                elif kind == "spin":
                    caster = spec[3]
                    widget.setValue(caster(v))
                elif kind == "check":
                    caster = spec[3]
                    widget.setChecked(caster(v))
            except Exception:
                pass
        # Panel checkboxes
        try:
            wanted = set(state.get("figures/single_panels", []))
            if wanted and hasattr(self, "_single_panel_checkboxes"):
                for k, cb in self._single_panel_checkboxes.items():
                    cb.setChecked(k in wanted)
        except Exception: pass
        try:
            wanted = set(state.get("compare/panels", []))
            if wanted and hasattr(self, "_cmp_panel_checkboxes"):
                for k, cb in self._cmp_panel_checkboxes.items():
                    cb.setChecked(k in wanted)
        except Exception: pass

    def _refresh_preset_modified(self):
        """Show the '• modified' pill when the current analysis parameters
        differ from the preset that was last applied (`_active_preset_state`).
        No baseline (sentinel / startup) → never modified."""
        badge = getattr(self, "_modified_badge", None)
        if badge is None or getattr(self, "_suspend_modified_watch", False):
            return
        base = getattr(self, "_active_preset_state", None)
        if not base:
            badge.hide()
            return
        cur = self._widget_state_dict()

        def _norm(x):
            return round(x, 7) if isinstance(x, float) else x

        modified = any(
            _norm(cur.get(k)) != _norm(v)
            for k, v in base.items() if k.startswith("analysis/"))
        if modified:
            badge.set_state("warn", "• modified")
            badge.show()
        else:
            badge.hide()

    # ── Parameter presets ────────────────────────────────────────────────
    _BUILTIN_PRESETS_TAG = "__firefly_builtin__"

    @staticmethod
    def _presets_dir() -> str:
        d = os.path.expanduser("~/.firefly/presets")
        os.makedirs(d, exist_ok=True)
        return d

    @classmethod
    def _builtin_presets(cls) -> "dict[str, dict]":
        """Default presets seeded on first launch.  Two reasonable
        starting points for common rigs in the lab — users can override
        or extend them via the 'Save…' button."""
        return {
            "PC12 Cells": {
                cls._BUILTIN_PRESETS_TAG: True,
                # Imaging metadata — 100x oil, fast PALM acquisition
                "analysis/override_px":     True,
                "analysis/pixel_size":      0.106,
                "analysis/override_fi":     True,
                "analysis/frame_interval":  0.020,
                # Preprocessing — flat well-spread cytoplasm; small radius
                # tracks local background tightly without smearing the spots.
                "analysis/bg_method":       "Uniform Filter",
                "analysis/bg_radius":       15,
                # Detection — typical PALM PSF after preprocessing
                "analysis/diameter":        7,
                "analysis/auto_minmass":    True,
                "analysis/minmass":         1.35,
                # Linking — small per-step displacements at 50 fps.
                # Memory=0 is correct for sptPALM: a fluorophore that
                # blinks off for >1 frame is a different molecule and
                # should NOT be re-linked across the gap.  Re-enabling
                # memory inflates track-length statistics and corrupts
                # diffusion fits.
                "analysis/search_range":    5,
                "analysis/memory":          0,
                "analysis/min_track_len":   8,
                "analysis/max_track_len":   0,
                # Diffusion + motion classification — standard sptPALM
                "analysis/max_lagtime":     10,
                "analysis/n_fit":           5,
                "analysis/alpha_immobile":  0.5,
                "analysis/alpha_confined":  0.9,
                "analysis/alpha_directed":  1.1,
                "analysis/mobile_d":        0.03,
                "analysis/jdd_components":  2,
                # ROI — Sister TIFF mask with Li auto-threshold against
                # the max-projection; the wide background sigma (σ=100)
                # deliberately picks up the whole cell footprint rather
                # than individual nanodomains.
                "analysis/roi_mode":        "Sister TIFF",
                "analysis/roi_auto_method": "Li",
                "analysis/roi_threshold":   0.030,
                "analysis/roi_mask_mode":   "Max",
                "analysis/roi_bg_sigma":    100.0,
                # Drift correction — segment length tuned for ~4 k frames
                "analysis/drift_correct":   True,
                "analysis/drift_segment":   400,
                # Clustering — receptor-nanodomain defaults.
                "analysis/cluster_eps_nm":  40.0,
                "analysis/cluster_min_samples": 8,
            },
            "Drosophila Neurons": {
                cls._BUILTIN_PRESETS_TAG: True,
                # Imaging metadata — leave Override OFF so the values read
                # from the CZI (e.g. 0.1 µm / 0.1 s on the lab's ZEN exports)
                # are used; the numbers below are only fallbacks for files
                # that carry no metadata.
                "analysis/override_px":     False,
                "analysis/pixel_size":      0.1,
                "analysis/override_fi":     False,
                "analysis/frame_interval":  0.1,
                # Preprocessing — tight bg radius removes the structured
                # neuron background (larger radii leave diffuse haze).
                "analysis/bg_method":       "Uniform Filter",
                "analysis/bg_radius":       15,
                # Detection — auto-tuned on lab fly-neuron sptPALM data:
                # diameter 9 is the smallest with flat pixel-bias while spots
                # stay round; minmass 2.0 keeps discrete blinks without
                # detecting the dendritic structure.
                "analysis/diameter":        9,
                "analysis/auto_minmass":    False,
                "analysis/minmass":         2.0,
                # Linking — slower diffusion in axons / dendrites.  memory=5
                # bridges the multi-frame blink gaps typical of this (largely
                # immobile, punctate) data so a molecule isn't fragmented into
                # several short tracks that each α-classify to a different,
                # noisy motion class.  Safe here because the molecules barely
                # move, so re-linking across a short gap can't grab a different
                # one.
                "analysis/search_range":    4,
                "analysis/memory":          5,
                "analysis/min_track_len":   10,
                "analysis/max_track_len":   0,
                # Diffusion + motion classification
                "analysis/max_lagtime":     20,
                "analysis/n_fit":           5,
                "analysis/alpha_immobile":  0.5,
                "analysis/alpha_confined":  0.9,
                "analysis/alpha_directed":  1.1,
                # Mobile/immobile split at 0.021 µm²/s (log10 D = -1.6) to match
                # the Hines thesis / Constals et al. 2015 Sx1a sptPALM threshold.
                "analysis/mobile_d":        0.021,
                "analysis/jdd_components":  2,
                # ROI — small / branched cells; manual polygon is more robust
                "analysis/roi_mode":        "Manual polygon",
                "analysis/roi_auto_method": "Li",
                "analysis/roi_threshold":   0.10,
                "analysis/roi_mask_mode":   "Max",
                "analysis/roi_bg_sigma":    25.0,
                # Drift correction
                "analysis/drift_correct":   True,
                "analysis/drift_segment":   400,
                # Clustering — synaptic-density defaults
                "analysis/cluster_eps_nm":  40.0,
                "analysis/cluster_min_samples": 8,
            },
        }

    def _finalise_presets(self) -> None:
        """Called once on startup (deferred so every sidebar widget is
        constructed first).  Seeds the two built-in presets to disk if
        the user hasn't already saved overrides, then populates the
        combobox."""
        try:
            self._seed_builtin_presets()
            self._refresh_preset_combo()
            try:
                self.c_preset.currentTextChanged.connect(
                    self._on_preset_picked)
            except Exception:
                pass
            # Watch the analysis params so the "• modified" pill lights up when
            # the user diverges from the applied preset.
            for spec in self._setting_specs():
                key, widget, kind = spec[0], spec[1], spec[2]
                if not key.startswith("analysis/"):
                    continue
                try:
                    if   kind == "spin":  sig = widget.valueChanged
                    elif kind == "combo": sig = widget.currentIndexChanged
                    elif kind == "check": sig = widget.toggled
                    elif kind == "text":  sig = widget.textChanged
                    else: continue
                    sig.connect(lambda *_a: self._refresh_preset_modified())
                except Exception:
                    pass
        except Exception:
            pass

    def _seed_builtin_presets(self) -> None:
        """Write the built-in presets to ~/.firefly/presets/ on first
        launch.  Skips any name the user has already saved their own
        version of, so user-customised presets aren't overwritten."""
        import json
        d = self._presets_dir()
        for name, payload in self._builtin_presets().items():
            path = os.path.join(d, f"{name}.json")
            if os.path.isfile(path):
                # Only overwrite if our previous write also tagged it as
                # built-in (i.e. user hasn't customised it).
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        prev = json.load(fh)
                    if not prev.get(self._BUILTIN_PRESETS_TAG, False):
                        continue
                except Exception:
                    continue
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2)
            except Exception:
                pass

    def _list_presets(self) -> "list[str]":
        d = self._presets_dir()
        try:
            return sorted(
                os.path.splitext(f)[0] for f in os.listdir(d)
                if f.endswith(".json"))
        except Exception:
            return []

    def _refresh_preset_combo(self) -> None:
        if not hasattr(self, "c_preset"):
            return
        names = self._list_presets()
        self.c_preset.blockSignals(True)
        try:
            self.c_preset.clear()
            self.c_preset.addItem("— Current settings —")
            for n in names:
                self.c_preset.addItem(n)
        finally:
            self.c_preset.blockSignals(False)





    def _switch_to_tab(self, label: str):
        """Switch the central tab widget to the tab whose visible text
        starts with `label` (e.g. "Analysis" → finds "Analysis")."""
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i).startswith(label):
                self.tabs.setCurrentIndex(i)
                return

    def _start_single_run(self):
        fpath = self.e_file.text().strip()
        if not fpath or not os.path.isfile(fpath):
            QtWidgets.QMessageBox.warning(
                self, "No file",
                "Pick an input file on the Import tab first.")
            self._switch_to_tab(TAB_IMPORT)
            return

        # External localisations table (.csv / .txt / .tsv) → skip detection
        # and run the downstream pipeline on the imported spots.  Anything
        # else is an image stack and runs the full localiser.
        is_loc = self._is_csv_input(fpath)

        if not is_loc:
            # Pre-flight backend validation — catches "user picked CUDA on
            # a CPU-only torch" before we waste minutes on frame loading.
            # Irrelevant for a localisations table (no detection step).
            if not self._validate_selected_backend():
                return
        # Auto-switch to the Analysis tab so the user sees progress
        self._switch_to_tab(TAB_ANALYSIS)
        self._start_elapsed_timer()

        out_dir = self.e_outdir.text().strip() or (
            os.path.dirname(fpath) if is_loc else None)
        params = self._build_params_for_file(fpath, out_dir)

        if is_loc:
            # Pixel size / frame interval come from the sidebar even when the
            # Override checkboxes are unticked — a localisations table has no
            # embedded image metadata to fall back on.
            if not params.get("pixel_size"):
                params["pixel_size"] = float(self.s_pixel_size.value())
            if not params.get("frame_interval"):
                params["frame_interval"] = float(self.s_frame_interval.value())
            preset = self.c_csv_preset.currentText()
            params["source"] = "external_csv"
            params["csv_preset"] = (
                "auto" if preset == "Auto-detect" else preset)
            bg = self.e_csv_bg.text().strip()
            if bg and os.path.isfile(bg):
                params["bg_image_path"] = bg

        # Persist before the long-running task in case of crash/abort.
        try:
            self._save_settings()
        except Exception:
            crash_reporter.log_exception("failed to save settings before run")

        # Clear UI for new run
        self.console_log.clear()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Starting…")
        self.run_stage_label.setText("Starting…")
        self.run_results.reset("Run in progress…")
        try:
            self.mass_hist.reset()
            self.mass_hist.set_minmass(float(self.s_minmass.value())
                                      if not self.c_auto_minmass.isChecked()
                                      else None)
            self.live_view.reset()
            self.pipeline_diagram.reset()
            self._analysis_stack.setCurrentIndex(0)   # show cockpit
        except AttributeError:
            pass
        self._is_batch_run   = False
        self._is_compare_run = False

        # Spawn analysis SUBPROCESS (not thread).  Rationale: Qt holds a
        # Metal-backed surface for window compositing on macOS, and that
        # contends with PyTorch's MPS allocator in the same process.  A
        # subprocess gives PyTorch a clean Python interpreter with no Qt
        # loaded — MPS gets the full unified-memory pool to itself.
        # Bounded queue — a 60 FPS live preview at ~590 KB/frame can
        # push ~35 MB/s through this pipe.  If the GUI ever stalls for
        # a few seconds the queue grows unbounded and pushes the
        # system into swap (we had a user hard-freeze caused by that).
        # 2000 messages is enough headroom for normal jitter; the
        # worker drops preview/mass messages when full and keeps only
        # the analysis-critical ones (log/progress/done/etc.).
        self._msg_queue    = multiprocessing.Queue(maxsize=2000)
        self._cancel_event = multiprocessing.Event()
        self._proc = multiprocessing.Process(
            target=_run_analysis_in_subprocess,
            args=(params, self._msg_queue, self._cancel_event),
            name="FIREFLY-AnalysisWorker",
            daemon=False)
        self._proc.start()
        self._poll_timer.start()

        self.btn_run.setText("Stop")
        self.statusBar().showMessage(
            "Running (external localisations)…" if is_loc else "Running…")

    def _collect_batch_params(self) -> "list[dict]":
        """Snapshot the currently-checked batch series + the current sidebar
        settings into a per-run params list.  Returns [] if nothing is checked.
        Used both by the immediate 'Start' (Batch mode) and by 'Add to queue'
        — because the params capture the live widget values at call time, each
        queued job keeps the settings/preset that were active when it was
        added."""
        groups = self._batch_checked_series()
        if not groups:
            return []
        # Batch outputs go to <input_folder>/batch_results/<stem>/.  Pass the
        # parent `batch_results/` only; the worker wraps each run in its own
        # per-stem subfolder (avoids the batch_results/<stem>/<stem>/ double-
        # enclosure bug).
        out_root = os.path.join(self.e_batch_folder.text().strip(),
                                "batch_results")
        _csv_preset_choice = self.c_csv_preset.currentText()
        _csv_preset = ("auto" if _csv_preset_choice == "Auto-detect"
                       else _csv_preset_choice)
        params_list = []
        for g in groups:
            fpath = g["primary"]
            p = self._build_params_for_file(fpath, out_root)
            if g.get("key"):
                p["stem_override"] = str(g["key"])
            if self._is_csv_input(fpath):
                if not p.get("pixel_size"):
                    p["pixel_size"] = float(self.s_pixel_size.value())
                if not p.get("frame_interval"):
                    p["frame_interval"] = float(self.s_frame_interval.value())
                p["source"]     = "external_csv"
                p["csv_preset"] = _csv_preset
            else:
                p["series_files"] = list(g.get("files") or [])
            params_list.append(p)
        return params_list

    def _launch_batch(self, params_list: "list[dict]") -> bool:
        """Spawn the batch worker over an already-built params list (one or
        many jobs concatenated).  Shared by 'Start' and 'Run queue'.
        Returns True if the worker actually started, False if it bailed
        (empty list / backend validation failed) so callers can keep the
        queue intact."""
        if not params_list:
            return False
        if not self._validate_selected_backend():
            return False
        self._switch_to_tab(TAB_ANALYSIS)
        self._start_elapsed_timer()
        try:
            self._save_settings()
        except Exception:
            crash_reporter.log_exception("failed to save settings before run")

        self.console_log.clear()
        self.batch_progress.setValue(0)
        self.batch_progress.setFormat("Starting…")
        self.batch_stage_label.setText("Starting…")
        self.batch_subprogress.setValue(0)
        self.batch_subprogress.setFormat("Preparing…")
        self.batch_subprogress.show()
        self.run_results.reset("Batch in progress…")
        try:
            self.mass_hist.reset()
            self.mass_hist.set_minmass(float(self.s_minmass.value())
                                      if not self.c_auto_minmass.isChecked()
                                      else None)
            self.live_view.reset()
            self.pipeline_diagram.reset()
            self._analysis_stack.setCurrentIndex(0)   # show cockpit
        except AttributeError:
            pass
        self._is_batch_run   = True
        self._is_compare_run = False

        # Bounded queue (see note in _start_single_run) — caps the live-preview
        # pipe so a GUI stall can't push the system into swap.
        self._msg_queue    = multiprocessing.Queue(maxsize=2000)
        self._cancel_event = multiprocessing.Event()
        self._proc = multiprocessing.Process(
            target=_run_batch_in_subprocess,
            args=(params_list, self._msg_queue, self._cancel_event),
            name="FIREFLY-BatchWorker",
            daemon=False)
        self._proc.start()
        self._poll_timer.start()

        self.btn_run.setText("Stop")
        self.statusBar().showMessage(f"Batch: 0 / {len(params_list)} runs")
        return True

    def _start_batch_run(self):
        """Start button (Batch mode).  If jobs are queued, run the WHOLE queue
        — so the primary button 'just works' and the user can't run only the
        first batch by mistake.  Otherwise run the current checked selection."""
        if getattr(self, "_batch_queue", []):
            self._on_batch_run_queue()
            return
        params_list = self._collect_batch_params()
        if not params_list:
            QtWidgets.QMessageBox.warning(
                self, "No files",
                "On the Import tab, switch to Batch mode and pick a "
                "folder + at least one file.")
            self._switch_to_tab(TAB_IMPORT)
            return
        self._launch_batch(params_list)

    def _start_compare_run(self):
        """Kick off a comparison over the configured groups."""
        groups = self._cmp_collect_groups()
        # Validation: ≥2 non-empty groups
        non_empty = [g for g in groups if g.get("folders")]
        if len(non_empty) < 2:
            QtWidgets.QMessageBox.warning(
                self, "Not enough groups",
                "Need at least 2 groups, each with at least 1 analysis "
                "folder.")
            return

        outdir = self.e_cmp_outdir.text().strip()
        if not outdir:
            QtWidgets.QMessageBox.warning(
                self, "No output folder",
                "Pick a folder to save the comparison outputs.")
            return
        try:
            os.makedirs(outdir, exist_ok=True)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, "Cannot create output folder", str(exc))
            return

        # Selected panels
        selected_panels = {key for key, cb in self._cmp_panel_checkboxes.items()
                           if cb.isChecked()}
        if not selected_panels:
            QtWidgets.QMessageBox.warning(
                self, "No panels selected",
                "Pick at least one panel to include in the comparison "
                "figure.")
            return

        # Two-factor pre-flight: if any card carries a time point, the
        # group × time-point ANOVA needs pingouin.  Warn (don't block) if it's
        # missing — the comparison figure still renders, just without the ANOVA.
        if any(str(g.get("timepoint", "")).strip() for g in non_empty):
            try:
                from firefly.analysis import fa_twoway
                have_pg = fa_twoway.HAVE_PINGOUIN
            except Exception:
                have_pg = False
            if not have_pg:
                box = QtWidgets.QMessageBox(
                    QtWidgets.QMessageBox.Icon.Warning,
                    "Two-factor stats unavailable",
                    "You set time points, but the statistics package "
                    "'pingouin' isn't installed, so the group × time-point "
                    "ANOVA will be skipped.\n\nThe comparison figure and CSVs "
                    "will still be produced. Continue anyway?",
                    QtWidgets.QMessageBox.StandardButton.NoButton, self)
                box.addButton("Continue",
                              QtWidgets.QMessageBox.ButtonRole.AcceptRole)
                btn_cancel = box.addButton(
                    "Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
                box.exec()
                if box.clickedButton() is btn_cancel:
                    return

        comparison_params = {
            "groups":      non_empty,
            "output_dir":  outdir,
            "output_stem": self.e_cmp_stem.text().strip() or "comparison",
            "theme":       self.c_cmp_theme.currentText(),
            "pdf_report":  bool(self.c_cmp_pdf.isChecked()),
            "panels":      list(selected_panels),
            "mobile_d_threshold": float(self.s_mobile_d_threshold.value()),
            "stats_config": self._collect_stats_config(),
        }

        try:
            self._save_settings()
        except Exception:
            crash_reporter.log_exception("failed to save settings before run")

        # Clear compare UI for new run
        self.console_log.clear()
        self.cmp_progress.setValue(0)
        self.cmp_stage_label.setText("Starting…")
        self.cmp_results.reset("Comparison in progress…")
        self.cmp_progress.setFormat("Starting…")
        self._is_batch_run    = False
        self._is_compare_run  = True

        # Bounded queue — a 60 FPS live preview at ~590 KB/frame can
        # push ~35 MB/s through this pipe.  If the GUI ever stalls for
        # a few seconds the queue grows unbounded and pushes the
        # system into swap (we had a user hard-freeze caused by that).
        # 2000 messages is enough headroom for normal jitter; the
        # worker drops preview/mass messages when full and keeps only
        # the analysis-critical ones (log/progress/done/etc.).
        self._msg_queue    = multiprocessing.Queue(maxsize=2000)
        self._cancel_event = multiprocessing.Event()
        self._proc = multiprocessing.Process(
            target=_run_compare_in_subprocess,
            args=(comparison_params, self._msg_queue, self._cancel_event),
            name="FIREFLY-CompareWorker",
            daemon=False)
        self._proc.start()
        self._poll_timer.start()

        self.btn_run.setText("Stop")
        self.btn_cmp_run.setEnabled(False)
        self.statusBar().showMessage(
            f"Comparing {len(non_empty)} group(s)…")

    # ── Queue polling (replaces QThread signals) ──────────────────────────

    # ── Subprocess result handlers ────────────────────────────────────────
    def _handle_done(self, payload: dict):
        out_dir = payload.get("out_dir", "")
        stem    = payload.get("stem", "")
        summary = payload.get("summary") or {}
        n_tracks = summary.get("n_tracks", payload.get("n_tracks", 0))
        headline = f"{stem}  —  {n_tracks:,} trajectories" if stem else \
                   f"Analysis complete — {n_tracks:,} trajectories"
        self.run_results.show_results(headline, out_dir)
        self.run_results.show_stats(summary)
        try:    self.pipeline_diagram.set_complete()
        except AttributeError: pass
        self.run_stage_label.setText("Done")
        self.progress_bar.setFormat("Complete")
        self.statusBar().showMessage(f"Analysis complete — output at {out_dir}")
        # Results are written to disk; to inspect them in the 3-D viewer, open
        # the run from the Visualise tab ("Load analysis run…").  We deliberately
        # do NOT auto-push into napari here — a large result can exhaust the GPU
        # backend and hard-crash the app, and loading on demand is clearer.

    def _handle_file_starting(self, payload: dict):
        """A new series is starting in a batch — update the overall-batch bar to
        show how many files remain.  `index` is 1-based (the file about to run),
        so `index - 1` files are already done.  The top bar then tracks this
        file's per-step progress."""
        i, total = payload.get("index", 0), payload.get("total", 0)
        if total:
            done = max(0, i - 1)
            left = max(0, total - done)
            self.batch_subprogress.setValue(int(100 * done / total))
            self.batch_subprogress.setFormat(
                f"{left} of {total} file(s) left   ·   {done} done")

    def _handle_file_done(self, payload: dict):
        """One series in a batch finished successfully — not the terminal msg.
        ('file' here = 'series' in the GUI sense — the batch list now has
        one entry per series, and the worker calls it once per series.)
        """
        i, total = payload.get("index", 0), payload.get("total", 0)
        n_tracks = payload.get("n_tracks", 0)
        stem     = payload.get("stem", "")
        self.statusBar().showMessage(
            f"Batch: {i} / {total} series complete  ({n_tracks:,} tracks)")
        if total:
            # Bottom bar = overall BATCH position: how many files are LEFT to
            # analyse (the top bar already tracks the current file's step + %).
            left = max(0, total - i)
            self.batch_subprogress.setValue(int(100 * i / total))
            self.batch_subprogress.setFormat(
                f"{left} of {total} file(s) left   ·   {i} done"
                if left else f"All {total} file(s) done — finishing…")

    def _handle_file_error(self, payload: dict):
        """One series in a batch failed — log it, batch continues."""
        i, total = payload.get("index", 0), payload.get("total", 0)
        f = payload.get("file", "?")
        self.console_log.appendPlainText(
            f"\n  ⚠ [{i}/{total}] failed: {os.path.basename(f)}")
        self.batch_stage_label.setText(
            f"[{i}/{total}] failed: {os.path.basename(f)} — batch continues")
        self.statusBar().showMessage(f"Batch: series {i} failed (continuing)")

    def _handle_batch_done(self, payload: dict):
        """Batch terminal message — all series attempted."""
        n_total = payload.get("n_total", 0)
        n_ok    = payload.get("n_ok",    0)
        n_fail  = payload.get("n_fail",  0)
        self.batch_progress.setValue(100)
        self.batch_progress.setFormat(
            f"Batch complete  —  {n_ok}/{n_total} series succeeded, "
            f"{n_fail} failed")
        self.batch_subprogress.hide()
        try:    self.pipeline_diagram.set_complete()
        except AttributeError: pass
        self.statusBar().showMessage(
            f"Batch complete — {n_ok}/{n_total} series succeeded, "
            f"{n_fail} failed")

        # Populate the results panel with the batch summary
        headline = (f"Batch complete — {n_ok}/{n_total} series succeeded"
                    + (f", {n_fail} failed" if n_fail else ""))
        # Aggregate stats across successful files
        results = payload.get("results") or []
        total_tracks = sum(r.get("n_tracks", 0) for r in results
                           if r.get("ok"))
        total_locs   = sum(r.get("n_locs", 0)   for r in results
                           if r.get("ok"))
        agg_summary = {
            "n_tracks": total_tracks,
            "n_locs":   total_locs,
            "motion_counts": {},
        }
        # Find the common output root (parent of every file's out_dir).
        # On Windows, os.path.commonpath raises ValueError when paths
        # straddle different drive letters (e.g. C:\ + D:\) — handle
        # that explicitly so a batch writing across drives doesn't
        # surface a misleading "root = first file's parent" link.
        common_root = ""
        ok_dirs = [r.get("out_dir") for r in results if r.get("ok") and r.get("out_dir")]
        if ok_dirs:
            try:
                common_root = os.path.commonpath(ok_dirs)
            except ValueError:
                # Mixed drives on Windows.  Don't fabricate a wrong root.
                common_root = ""
            except Exception:
                common_root = os.path.dirname(ok_dirs[0])
        self.run_results.show_results(headline, common_root,
                                      files=None)
        self.run_results.show_stats(agg_summary)

    def _handle_compare_done(self, payload: dict):
        """Compare terminal message — figure + CSVs + PDF have been saved."""
        self.cmp_progress.setValue(100)
        self.cmp_progress.setFormat("Complete")
        out_dir   = payload.get("output_dir", "")
        n_groups  = payload.get("n_groups", 0)
        headline = f"Comparison complete — {n_groups} group(s)"
        self.cmp_results.show_results(headline, out_dir)
        self.cmp_stage_label.setText("Done")
        self.statusBar().showMessage(
            f"Comparison complete — output at {out_dir}")

    def _handle_compare_error(self, message: str):
        """Expected comparison-input problem (no valid folders, drive unmounted,
        <2 groups) — show a clear, actionable popup instead of a crash report."""
        try:
            self.cmp_progress.setValue(0)
            self.cmp_progress.setFormat("Couldn't run")
            self.cmp_stage_label.setText("Comparison not run")
        except Exception:
            pass
        self.statusBar().showMessage("Comparison couldn't run — see message")
        self.console_log.appendPlainText(f"\n{message}")
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Comparison couldn't run")
        box.setText("The comparison was not run.")
        box.setInformativeText(str(message))
        box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        box.exec()

    def _handle_stopped(self):
        self.statusBar().showMessage("Stopped by user")
        is_batch = getattr(self, "_is_batch_run", False)
        bar = self.batch_progress if is_batch else self.progress_bar
        bar.setFormat("Stopped")

    def _handle_failed(self, tb: str):
        try:
            path = crash_reporter.write_crash_report(
                RuntimeError, RuntimeError("Analysis subprocess raised"),
                None, source="analysis subprocess", context=tb)
            self.console_log.appendPlainText(f"\nCrash report: {path}")
            self._show_crash_dialog(path)
        except Exception:
            QtWidgets.QMessageBox.critical(
                self, "Analysis error", tb[-1500:])
        self.statusBar().showMessage("Error — see log")

    # ── Elapsed-time tracker for the Analysis tab ─────────────────────────
    @staticmethod
    def _format_elapsed(secs: float) -> str:
        secs = max(0, int(secs))
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _start_elapsed_timer(self):
        import time as _time
        self._run_start_time = _time.monotonic()
        try:
            self.lbl_elapsed.setText("Elapsed: 00:00")
        except AttributeError:
            return
        self._elapsed_timer.start()


    def _stop_elapsed_timer(self):
        self._elapsed_timer.stop()
        if self._run_start_time is not None:
            import time as _time
            final = self._format_elapsed(_time.monotonic() - self._run_start_time)
            try:
                self.lbl_elapsed.setText(f"Elapsed: {final}")
            except AttributeError:
                pass
        self._run_start_time = None

    def _cleanup_after_run(self):
        """Tear down the subprocess + queue after a run ends."""
        self._poll_timer.stop()
        self._stop_elapsed_timer()
        if self._proc is not None:
            try:
                if self._proc.is_alive():
                    self._proc.join(timeout=2.0)
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=1.0)
            except Exception:
                pass
        self._proc                  = None
        self._msg_queue             = None
        self._cancel_event          = None
        self._stop_requested_at     = None
        self._stop_escalation_stage = 0
        self._is_batch_run          = False
        self._is_compare_run        = False
        self.btn_run.setText("Start")
        self.btn_run.setEnabled(True)
        try:
            self.btn_cmp_run.setEnabled(True)
        except AttributeError:
            pass
        try:
            self.batch_subprogress.hide()
        except AttributeError:
            pass
        # Run is over — swap from cockpit to results panel.
        try:
            self._analysis_stack.setCurrentIndex(1)
        except AttributeError:
            pass

    # ── CUDA-torch sidecar installer (Windows + NVIDIA only) ─────────────
    def _maybe_offer_cuda_install(self):
        """First-launch (and every-launch-until-handled) prompt that
        offers to download the CUDA build of PyTorch into a sidecar
        directory.  Silent no-op on macOS/Linux or when the user
        previously declined or already installed."""
        try:
            from firefly import cuda_installer as _cu
        except Exception:
            return
        try:
            if not _cu.is_windows():
                return
            if _cu.is_installed():
                return
            if _cu.user_declined():
                return
            gpu = _cu.detect_nvidia_gpu()
            if not gpu:
                return
            msg = (
                f"NVIDIA {gpu} detected.\n\n"
                "Install CUDA acceleration for ~5–10× faster "
                "localisation?\n\n"
                "One-time ~2.5 GB download to "
                "%LOCALAPPDATA%\\FIREFLY\\torch-cuda."
            )
            reply = QtWidgets.QMessageBox.question(
                self, "Install CUDA acceleration?", msg,
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.Yes,
            )
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
                self._run_cuda_install()
            else:
                try:
                    _cu.mark_declined()
                except Exception:
                    pass
        except Exception:
            # Never let installer code crash the GUI on startup.
            pass


    def _run_cuda_install(self):
        """Drive the download + extract in a background QThread and show
        a cancellable QProgressDialog with two phases."""
        try:
            from firefly import cuda_installer as _cu
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "CUDA installer unavailable", str(exc))
            return

        # ── Resolve torch version + wheel URL on the MAIN thread ─────────
        # `import torch` from a Windows onefile bundle takes 10–30 s on
        # first call (it unpacks ~500 MB of DLLs to %TEMP%) and holds
        # the GIL the whole time.  If this ran inside the QThread worker,
        # the dialog would sit at 0 % / "Preparing download…" for half a
        # minute with no feedback and the user would think the app froze.
        # Doing it here, with a busy cursor + status-bar message, gives
        # immediate feedback and means the worker can start downloading
        # the moment its thread spins up.
        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(Qt.CursorShape.WaitCursor))
        try:
            try:
                self.statusBar().showMessage(
                    "Resolving PyTorch version (first time may take 30 s)…")
            except Exception: pass
            try:
                ver = _cu.bundled_torch_version()
            except Exception:
                ver = None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            try: self.statusBar().clearMessage()
            except Exception: pass

        if not ver:
            QtWidgets.QMessageBox.warning(
                self, "CUDA install failed",
                "Could not determine the bundled PyTorch version — cannot "
                "pick a matching CUDA wheel.  Try installing from source "
                "instead (see README).")
            return

        try:
            url = _cu.cuda_wheel_url(ver, cuda_tag="cu124")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "CUDA install failed",
                f"Could not build the CUDA wheel URL: {exc}")
            return

        dlg = QtWidgets.QProgressDialog(
            f"Connecting to download.pytorch.org…\n"
            f"(torch {ver} + cu124)",
            "Cancel", 0, 100, self)
        dlg.setWindowTitle("Installing CUDA acceleration")
        # NON-modal so the debug log window below can stay visible
        # alongside.  Cancel button still works.
        dlg.setWindowModality(Qt.WindowModality.NonModal)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)

        # Thread-safe cancellation.  The worker thread must NOT call
        # dlg.wasCanceled() — QProgressDialog is a QWidget, and touching a
        # widget from a non-GUI thread is undefined behaviour that
        # intermittently deadlocks the main thread on Windows (the whole app
        # white-outs / goes "Not Responding" mid-download).  Instead the
        # dialog's `canceled` signal — emitted on the GUI thread when the
        # user clicks Cancel — flips a threading.Event the worker polls.
        import threading as _threading
        cancel_event = _threading.Event()
        dlg.canceled.connect(cancel_event.set)

        # cuda_installer still prints its step-by-step diagnostics to stdout
        # (visible in the launching terminal), so no in-app log window is
        # needed for normal use.

        # Background worker — QObject moved to a QThread (NOT a
        # QThread subclass).  Signals dispatch back to the GUI thread.
        class _CudaWorker(QtCore.QObject):
            progress = QtCore.Signal(int, str)   # pct, label
            finished = QtCore.Signal()
            failed   = QtCore.Signal(str)

            def __init__(self, cancel_check, wheel_url, torch_version):
                super().__init__()
                self._cancel_check  = cancel_check
                self._wheel_url     = wheel_url
                self._torch_version = torch_version

            @QtCore.Slot()
            def run(self):
                # Heartbeat immediately so the user sees activity instead
                # of an apparently-frozen 0 % dialog while urlopen is
                # waiting for the server to start sending bytes.
                self.progress.emit(
                    0, "Connecting to download.pytorch.org…")
                try:
                    def _dl_cb(done, total):
                        if total > 0:
                            pct = int(done * 100 / total)
                        else:
                            # Unknown total — show MB downloaded
                            pct = 0
                        mb = done / (1024 * 1024)
                        # BITS sits in Connecting/Transferring with
                        # done=0,total=0 emitting heartbeats every 500
                        # ms.  Emitting "Downloading… 0 MB" with that
                        # is misleading — it freezes the heartbeat
                        # (which the GUI switches off as soon as it
                        # sees "MB" in a label) and the user assumes
                        # the app is hung.  Until BITS has actually
                        # moved any bytes, keep the label as a status
                        # message that does NOT contain "MB" so the
                        # elapsed-time heartbeat keeps ticking.
                        if done <= 0:
                            label = ("Waiting for transfer to start "
                                     "(handshake / proxy)…")
                            self.progress.emit(0, label)
                            return
                        if total > 0:
                            tot_mb = total / (1024 * 1024)
                            label = (f"Downloading torch-CUDA wheel… "
                                     f"{mb:.0f} / {tot_mb:.0f} MB")
                        else:
                            label = (f"Downloading torch-CUDA wheel… "
                                     f"{mb:.0f} MB")
                        self.progress.emit(pct, label)

                    def _ex_cb(done, total):
                        if total > 0:
                            pct = int(done * 100 / total)
                        else:
                            pct = 0
                        self.progress.emit(
                            pct, f"Extracting… {done} / {total} files")

                    # Auto-fallback across cu124 → cu121 → cu118 so a
                    # missing cu124 wheel for a brand-new torch version
                    # doesn't dead-end with HTTP 404.
                    def _status(msg):
                        self.progress.emit(0, msg)
                    _cu.install_cuda_torch_auto(
                        torch_version=self._torch_version,
                        download_progress_cb=_dl_cb,
                        extract_progress_cb=_ex_cb,
                        cancel_cb=self._cancel_check,
                        status_cb=_status,
                    )
                    self.finished.emit()
                except Exception as exc:
                    self.failed.emit(str(exc))

        thread = QtCore.QThread(self)
        worker = _CudaWorker(
            cancel_check=cancel_event.is_set,
            wheel_url=url,
            torch_version=ver)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        # Keep refs alive on self until the thread exits
        self._cuda_thread = thread
        self._cuda_worker = worker

        # Heartbeat — while we're still in the "connecting" phase (no
        # bytes downloaded yet) the dialog would otherwise show a frozen
        # static label.  Tick once a second so the user can see the app
        # is alive.  Stops automatically when real download progress
        # starts arriving.
        import time as _t
        self._cuda_started_at = _t.monotonic()
        self._cuda_last_label = (
            f"Connecting to download.pytorch.org…\n"
            f"(torch {ver} + cu124)")
        self._cuda_in_real_progress = False

        def _heartbeat():
            if self._cuda_in_real_progress:
                return
            try:
                elapsed = int(_t.monotonic() - self._cuda_started_at)
                dlg.setLabelText(
                    f"{self._cuda_last_label}  ({elapsed} s elapsed)")
            except Exception:
                pass

        self._cuda_heartbeat = QtCore.QTimer(self)
        self._cuda_heartbeat.setInterval(1000)
        self._cuda_heartbeat.timeout.connect(_heartbeat)
        self._cuda_heartbeat.start()

        def _on_progress(pct, label):
            try:
                # Switch off the elapsed-time heartbeat only when REAL
                # bytes / files are showing up.  Previous logic keyed
                # off the literal substring "MB" — but BITS likes to
                # emit "Downloading… 0 MB" with no actual transfer
                # happening, which killed the heartbeat and gave the
                # appearance of a freeze.  Match a positive number
                # before "MB" / "files" instead.
                import re as _re
                _has_real = bool(
                    _re.search(r"\b([1-9]\d*)(?:\.\d+)?\s*(?:MB|files)\b",
                               label)
                    or _re.search(r"(?:\bfiles\b).*?\b([1-9]\d*)\b", label)
                    or pct > 0
                )
                if _has_real:
                    self._cuda_in_real_progress = True
                else:
                    # Still in connecting / fallback-tag-trying phase
                    # — remember the label so the heartbeat can append
                    # elapsed time without flicker.
                    self._cuda_last_label = label
                dlg.setLabelText(label)
                dlg.setValue(max(0, min(100, pct)))
            except Exception:
                pass

        def _cleanup():
            try:
                self._cuda_heartbeat.stop()
            except Exception:
                pass
            try:
                thread.quit()
                thread.wait(2000)
            except Exception:
                pass
            try:
                dlg.close()
            except Exception:
                pass
            self._cuda_thread = None
            self._cuda_worker = None
            self._cuda_heartbeat = None
            self._cuda_relay = None

        def _on_finished():
            _cleanup()
            # Update the Performance-section control (hide the Set-up button,
            # show the "installed — manage in Settings" status).
            try:
                self._refresh_cuda_perf_ui()
            except Exception:
                pass
            QtWidgets.QMessageBox.information(
                self, "CUDA installed",
                "CUDA acceleration installed successfully.\n\n"
                "Restart FIREFLY to use GPU acceleration.")

        def _on_failed(msg):
            _cleanup()
            # Make sure the user can retry on next launch.
            try:
                _cu.clear_declined()
            except Exception:
                pass
            # Use a detailed box (not just .warning) so the multi-line
            # error message — which lists every URL we tried — renders
            # fully instead of being clipped.  Raise + activateWindow so
            # the box pops to the front on Windows even if focus drifted
            # while the worker ran.
            box = QtWidgets.QMessageBox(
                QtWidgets.QMessageBox.Icon.Warning,
                "CUDA install failed", msg,
                QtWidgets.QMessageBox.StandardButton.Ok, self)
            box.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            box.raise_()
            box.activateWindow()
            box.exec()

        # Route worker signals through a QObject that lives in the GUI
        # thread.  Connecting a signal directly to a bare closure gives a
        # *Direct* connection (a functor has no thread affinity), so the
        # slot would run in whatever thread emits the signal — including
        # cuda_installer's non-Qt daemon threads.  Touching QWidgets (the
        # progress dialog) off the GUI thread is an access violation that
        # hard-crashes the process.  A relay whose slots are real @Slot
        # methods on a main-thread QObject makes AutoConnection pick
        # QueuedConnection, so the slots always run on the GUI thread.
        class _CudaSignalRelay(QtCore.QObject):
            @QtCore.Slot(int, str)
            def on_progress(self, pct, label): _on_progress(pct, label)
            @QtCore.Slot()
            def on_finished(self): _on_finished()
            @QtCore.Slot(str)
            def on_failed(self, msg): _on_failed(msg)

        relay = _CudaSignalRelay()
        self._cuda_relay = relay   # keep alive until cleanup

        worker.progress.connect(relay.on_progress)
        worker.finished.connect(relay.on_finished)
        worker.failed.connect(relay.on_failed)

        thread.start()
        dlg.show()

    # ── Crash reporter integration ────────────────────────────────────────
    def _install_menubar(self):
        """Give FIREFLY its own QMenuBar with at minimum a File → Quit
        action.  On macOS the menubar is global; if we don't own one,
        any embedded napari Viewer will claim it, and clearing napari's
        menubar later (we no longer do that, but defensively) used to
        take ⌘Q down with it.  Adding our own keeps Quit reliable.

        On Windows/Linux the menubar is per-window and would show a
        useless "File → Quit FIREFLY" strip across the top of the
        main window — the window's own close button (X) already does
        the same thing.  Skip it entirely off macOS.
        """
        if sys.platform != "darwin":
            return
        mb = self.menuBar()
        # Use native (system) menu bar on macOS so the entries show in
        # the system bar instead of inside the window.
        try:    mb.setNativeMenuBar(True)
        except Exception: pass

        file_menu = mb.addMenu("File")

        # Quit — ⌘Q on macOS, Ctrl+Q elsewhere.  Qt's StandardKey.Quit
        # maps to the right shortcut per platform.  We bind the shortcut
        # with ApplicationShortcut context so it fires no matter which
        # QMainWindow currently has focus (the embedded napari viewer
        # is also a QMainWindow, and without this the shortcut only
        # fires when *our* window is key — hence the flaky ⌘Q).
        act_quit = QtGui.QAction("Quit FIREFLY", self)
        act_quit.setMenuRole(QtGui.QAction.MenuRole.QuitRole)
        act_quit.setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        act_quit.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # Preferences — macOS automatically promotes this into the
        # application menu (FIREFLY → Preferences…) via PreferencesRole.
        act_prefs = QtGui.QAction("Preferences…", self)
        act_prefs.setMenuRole(QtGui.QAction.MenuRole.PreferencesRole)
        act_prefs.setShortcut(QtGui.QKeySequence("Ctrl+,"))
        act_prefs.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
        act_prefs.triggered.connect(self._open_preferences)
        file_menu.addAction(act_prefs)

        # Belt-and-braces backup — a standalone QShortcut at the same
        # application-wide context.  If something downstream resets
        # the action's shortcut (some napari versions reach into the
        # global QAction list), this still catches ⌘Q.
        self._sc_quit = QtGui.QShortcut(
            QtGui.QKeySequence.StandardKey.Quit, self)
        self._sc_quit.setContext(Qt.ShortcutContext.ApplicationShortcut)
        self._sc_quit.activated.connect(self.close)

    def _install_crash_hooks(self):
        """Wire FIREFLY into the global crash reporter.  Same idea as the Tk
        version: capture every uncaught exception, write a self-contained
        text report, surface the path to the user via a dialog."""

        def _log_provider(n: int = 120) -> str:
            try:
                txt = self.console_log.toPlainText()
                return "\n".join(txt.splitlines()[-n:])
            except Exception:
                return ""

        def _state_provider() -> dict:
            # Guard: if the main window's C++ side has already been
            # destroyed (e.g. crash during shutdown after a background
            # thread queued its exception), accessing self.attribute on
            # a deleted QObject raises RuntimeError or worse.  shiboken's
            # isValid() check returns False once the C++ peer is gone.
            try:
                from shiboken6 import isValid as _is_valid
                if not _is_valid(self):
                    return {"<state>": "main window already destroyed"}
            except Exception:
                pass
            try:
                return {
                    "UI":                 "PySide6 (v2.0-dev)",
                    "Current file":       self.e_file.text(),
                    "Output folder":      self.e_outdir.text() or "(default)",
                    "Pixel size":         self.s_pixel_size.value(),
                    "Frame interval":     self.s_frame_interval.value(),
                    "Detection diameter": self.s_diameter.value(),
                    "Threshold":          self.s_minmass.value(),
                    "Detection backend":  self.c_backend.currentText(),
                    "Running":            (self._proc is not None
                                            and self._proc.is_alive()),
                }
            except Exception as e:
                return {"<state error>": repr(e)}

        crash_reporter.set_log_provider(_log_provider)
        crash_reporter.set_app_state_provider(_state_provider)

        def _on_crash(path: str):
            # Marshal to Qt main thread before touching widgets
            QtCore.QMetaObject.invokeMethod(
                self, "_show_crash_dialog",
                Qt.ConnectionType.QueuedConnection,
                QtCore.Q_ARG(str, path))

        crash_reporter.install_global_handlers(on_crash=_on_crash)

    @QtCore.Slot(str)
    def _show_crash_dialog(self, path: str):
        """Modal dialog with the crash-report path; offers to open the folder."""
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        msg.setWindowTitle("FIREFLY — Unexpected error")
        msg.setText("FIREFLY hit an unexpected error.")
        msg.setInformativeText(
            f"A detailed crash report has been saved:\n\n"
            f"    {os.path.basename(path)}\n\n"
            f"Location:\n    {os.path.dirname(path)}")
        msg.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Open
            | QtWidgets.QMessageBox.StandardButton.Close)
        msg.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Open)
        if msg.exec() == QtWidgets.QMessageBox.StandardButton.Open:
            _open_folder(os.path.dirname(path))

    def _load_icon(self):
        """Best-effort: load assets/icon.png as the window/dock icon."""
        # Source layout: this file lives at <repo>/firefly/ui/app_qt.py, so
        # the assets/ folder is three dirnames up.  Frozen: _MEIPASS/assets.
        _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        for cand in (
                os.path.join(_repo_root, "assets", "icon.png"),
                os.path.join(getattr(sys, "_MEIPASS", ""), "assets", "icon.png"),
        ):
            if os.path.isfile(cand):
                self.setWindowIcon(QtGui.QIcon(cand))
                QtWidgets.QApplication.setWindowIcon(QtGui.QIcon(cand))
                return


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════




# ══════════════════════════════════════════════════════════════════════════════
#  THEME — GitHub-dark-style palette matching the legacy Tk app
# ══════════════════════════════════════════════════════════════════════════════
# Colour constants are duplicated as Python globals here AND injected into
# the QSS template via .format() so they're a single source of truth for
# both stylesheet and any programmatic widget colouring (matplotlib
# canvas backgrounds, error messages, etc.).
#
# The supported themes are stored in `_THEMES` (dict-of-dicts).  The
# active theme `_THEME` is selected at app start from QSettings — see
# `_pick_startup_theme()` below.  Switching themes requires restarting
# the app (most widgets read `_THEME[...]` at construction time, so a
# live switch would only repaint a fraction of the UI).










# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # Start the rotating log file first so even early failures are recorded.
    crash_reporter.setup_logging()
    # Install crash handlers BEFORE creating QApplication so an early failure
    # (e.g. Qt plugin load, OpenGL init) still produces a useful report.
    crash_reporter.install_global_handlers()

    QtCore.qInstallMessageHandler(_qt_message_handler)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("FIREFLY")
    app.setOrganizationName("jacoblevers")

    # Apply the FIREFLY dark theme (Fusion style + QSS + QPalette).
    _apply_firefly_theme(app)

    window = MainWindow()
    window.show()

    # CI smoke-test marker (mirrors the Tk app behaviour)
    marker_path = os.environ.get("SPTPALM_READY_MARKER")
    if marker_path:
        try:
            window.repaint()
        except Exception:
            pass
        try:
            with open(marker_path, "w") as f:
                f.write("ready\n")
        except Exception:
            pass

    sys.exit(app.exec())


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
