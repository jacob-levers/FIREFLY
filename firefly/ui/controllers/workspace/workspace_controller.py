"""AnalysisWorkspaceController — the merged live Compare + Results workspace.

Backs the QML "Analysis" tab.  The right rail is the input surface (conditions,
each a set of analysis-output run folders; user timepoints; comparison
settings); the left is a live readout (figure → headline metrics → group
statistics → pairwise significance → methods sentence).  There is **no Generate
button** — every change recomputes:

* the *numbers* (headline / stats / significance / methods) recompute
  synchronously in :meth:`_recompute` — scipy on per-folder replicate arrays is
  microseconds;
* the *figure* re-renders on a short debounce on a worker thread
  (:class:`_FigureJob`), flashing the amber "Recomputing" state, then settles.

All heavy data work lives in :mod:`workspace_data` (loading, metrics, stats) and
:mod:`workspace_figures` (matplotlib → QImage); this class is the Qt glue +
state machine.
"""
from __future__ import annotations

import os
import queue
import threading
import weakref

import numpy as np
from PySide6.QtCore import (QObject, Property, Signal, Slot, QTimer, QUrl,
                            QStandardPaths)
from PySide6.QtGui import QImage, QGuiApplication, QClipboard, QDesktopServices

from . import workspace_data as wd
from . import workspace_figures as wf

_PHASE_COLORS = ["#58a6ff", "#f78166", "#56d364", "#27c0e8", "#f6a623",
                 "#a371f7", "#e05252", "#7ed321"]
_DEBOUNCE_MS = 280


def _glabel(name: str, phase: str) -> str:
    return f"{name} · {phase}" if phase and phase not in ("—", name) else name


# ── internal model ─────────────────────────────────────────────────────────
class _Folder:
    __slots__ = ("path", "run", "excluded", "loading", "analysing")

    def __init__(self, path, run, loading=False, analysing=False):
        self.path = path
        self.run = run                       # wd.RunData | None
        # failed-QC folders start excluded (matches the prototype)
        self.excluded = bool(run is not None and run.qc_level == "error")
        # a placeholder chip while the run loads off the GUI thread
        self.loading = bool(loading)
        # an external localisation file being analysed into a run (longer job
        # than a folder load — same chip, different label)
        self.analysing = bool(analysing)


class _Condition:
    def __init__(self, cid, name, color, phase="—"):
        self.id = cid
        self.name = name
        self.color = color
        self.phase = phase
        self.folders: list[_Folder] = []

    def active(self) -> list[_Folder]:
        return [f for f in self.folders if not f.excluded and f.run is not None]


# ── async figure render ─────────────────────────────────────────────────────
class _FigureJob(threading.Thread):
    """Render the live metric figure off the GUI thread."""

    def __init__(self, groups, metric, cfg, size, done_cb):
        super().__init__(daemon=True)
        self._groups = groups
        self._metric = metric
        self._cfg = cfg
        self._size = size
        self._done = done_cb

    def run(self):
        img = None
        try:
            w, h = self._size
            img = wf.render_metric(
                self._groups, self._metric, plot=self._cfg.get("plot", "Violin"),
                err=self._cfg.get("err", "95% CI"),
                log_x=bool(self._cfg.get("logX", False)),
                logd_style=self._cfg.get("_logd_style", "overlaid"),
                mobile_d=float(self._cfg.get("_mobile_d", 0.05)),
                logd_clip=(float(self._cfg.get("_logd_clip_min", 0.00001)),
                           float(self._cfg.get("_logd_clip_max", 10.0))),
                group_style=self._cfg.get("_group_style", "box_points"),
                length_style=self._cfg.get("_length_style", "density"),
                grouped_data=self._cfg.get("_grouped_data"),
                width_px=w, height_px=h, dpi=110)
        except Exception:
            img = None
        self._done(img)


class _PanelJob(threading.Thread):
    """Render one per-condition publication panel off the GUI thread."""

    def __init__(self, panel, runs, color, size, done_cb):
        super().__init__(daemon=True)
        self._panel = panel
        self._runs = runs
        self._color = color
        self._size = size
        self._done = done_cb

    def run(self):
        img = None
        try:
            w, h = self._size
            img = wf.render_panel(self._panel, self._runs, self._color,
                                  width_px=w, height_px=h, dpi=110)
        except Exception:
            img = None
        self._done(img)


class _GroupAllPanelsJob(threading.Thread):
    """Render ALL averageable fa_figure analysis panels for a group in ONE
    make_figure call (the heavy part is shared), returning {letter: QImage}.
    The controller caches the dict per group so switching panels is instant."""

    def __init__(self, folders, theme, color, done_cb):
        super().__init__(daemon=True)
        self._folders = list(folders)
        self._theme = theme
        self._color = color
        self._done = done_cb

    def run(self):
        out = {}
        try:
            from firefly.ui.controllers.workspace import workspace_group_figures as gpf
            panels = gpf.render_group_panels(self._folders, gpf.AVERAGEABLE_LETTERS,
                                             theme=self._theme, group_color=self._color)
            for letter, pil in panels.items():
                try:
                    out[letter] = gpf.pil_to_qimage(pil)
                except Exception:
                    pass
        except Exception:
            out = {}
        self._done(out)


class _ReportJob(threading.Thread):
    """Run the real fa_compare.compare_groups engine off the GUI thread — it
    produces the multi-panel figure + every CSV / PDF / JSON artefact in one
    call.  Heavy (reads every replicate folder, renders ~13 panels), so it's an
    explicit action, not part of the live recompute."""

    def __init__(self, render_fn, out_dir, done_cb):
        super().__init__(daemon=True)
        self._render = render_fn
        self._out_dir = out_dir
        self._done = done_cb

    def run(self):
        result = {"ok": False, "error": "", "dir": self._out_dir}
        try:
            # Deferred import: fa_compare pulls in matplotlib/scipy/pingouin, which
            # we don't want to pay for unless the user actually generates a report.
            from firefly.analysis.fa_compare import CompareInputError
            try:
                with _COMPARE_LOCK:
                    self._render()          # compute_report (cached) → render_report
                result["ok"] = True
            except CompareInputError as e:
                result["error"] = str(e)               # expected user-error path
        except Exception as e:                          # pragma: no cover
            result["error"] = str(e)
        self._done(result)


# compare_groups uses matplotlib's global state — serialise every engine render
# (live single-panel figure + the full report) so they never run concurrently.
_COMPARE_LOCK = threading.Lock()


def _crop_engine_header(fig, canvas, img):
    """Crop a single-panel render down to JUST the panel content — the union of
    its real axes' tight bboxes (axis title, tick/axis labels, in-axes legends,
    colorbars, the radial polar frame).  Trims the shared top header (the "A vs B"
    title + per-group summary band, figure-level artists above every Axes) AND
    the surrounding figure margins, so the panel is tight and horizontally
    balanced — it centres cleanly, matching the sliced multi-panel path.

    Works on the ALREADY-rendered Agg buffer (engine facecolor + dpi preserved
    exactly, no re-render).  Anything uncertain → return the image untouched.
    fa_compare.py is not modified.
    """
    try:
        if img is None or img.isNull():
            return img
        from matplotlib.transforms import Bbox
        from PySide6.QtCore import QRect
        r = canvas.get_renderer()
        Hpx = float(fig.bbox.height)             # display px == img.height()
        H, W = img.height(), img.width()

        boxes = []
        for ax in fig.axes:
            try:
                if not getattr(ax, "axison", True) and not ax.has_data():
                    continue
                bb = ax.get_tightbbox(r)
                if bb is not None and bb.height > 0 and bb.width > 0:
                    boxes.append(bb)
            except Exception:
                continue
        if not boxes:
            return img
        u = Bbox.union(boxes)                     # full panel content bbox
        pad = 4
        x0 = max(0, int(round(u.x0)) - pad)
        x1 = min(W, int(round(u.x1)) + pad)
        # display (bottom-origin) → QImage rows (top-origin): row = Hpx - y
        row_top = max(0, int(round(Hpx - u.y1)) - pad)
        row_bot = min(H, int(round(Hpx - u.y0)) + pad)
        if x1 - x0 < 8 or row_bot - row_top < 8:
            return img
        cropped = img.copy(QRect(x0, row_top, x1 - x0, row_bot - row_top))
        return cropped if (cropped is not None and not cropped.isNull()) else img
    except Exception:
        return img


class _EngineFigJob(threading.Thread):
    """Render ONE export panel via the real engine (compare_groups with a single
    panel) so the live figure is byte-for-byte the report's panel.  Heavier than
    the bespoke renderer (full compute), so the controller serialises + caches it."""

    def __init__(self, render_fn, done_cb):
        super().__init__(daemon=True)
        self._render = render_fn
        self._done = done_cb

    def run(self):
        img = None
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            import matplotlib.pyplot as plt
            from PySide6.QtGui import QImage
            with _COMPARE_LOCK:
                fig, _summary, _stats = self._render()
                # Strip the per-panel axes title for the IN-APP preview — it's
                # redundant with the scroller chip / LIVE label (and the long
                # ones overflowed).  Operates on the returned fig only; the
                # exported report (a separate render) keeps its titles.
                try:
                    for ax in fig.axes:
                        if ax.get_title():
                            ax.set_title("")
                except Exception:
                    pass
                try:
                    canvas = FigureCanvasAgg(fig)
                    canvas.draw()
                    w, h = canvas.get_width_height()
                    img = QImage(bytes(canvas.buffer_rgba()), w, h,
                                 QImage.Format.Format_RGBA8888).copy()
                    img = _crop_engine_header(fig, canvas, img)
                finally:
                    try:    plt.close(fig)
                    except Exception: pass
        except Exception:
            img = None
        self._done(img)


class AnalysisWorkspaceController(QObject):
    # right-rail / inputs
    conditionsChanged = Signal()
    loadingChanged = Signal()          # count of folders still loading changed
    timepointsChanged = Signal()
    # left-rail / outputs
    metricChanged = Signal()
    viewChanged = Signal()
    cfgChanged = Signal()
    busyChanged = Signal()
    resultsChanged = Signal()          # numbers recomputed
    figureChanged = Signal()           # figure image ready (token bumped)
    presetsChanged = Signal()
    toast = Signal(str)
    panelChanged = Signal()
    panelImageChanged = Signal()
    panelRevChanged = Signal()
    panelGroupRevChanged = Signal()    # group-averaged panels (re)rendered
    reportChanged = Signal()           # full-engine report state changed
    reportReady = Signal(str)          # output dir of a finished full report

    # cross-thread delivery of the rendered figure (queued to GUI thread)
    _figureRendered = Signal()
    _panelRendered = Signal()
    _groupRendered = Signal()          # group-averaged fa_figure panels ready
    _reportRendered = Signal()         # full-engine report finished (worker→GUI)
    _engfigRendered = Signal()         # live single-panel engine figure (worker→GUI)
    _foldersLoaded = Signal()          # condition folders loaded (worker→GUI)

    def __init__(self, settings=None, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._cid_seq = 0
        self._conditions: list[_Condition] = []
        self._timepoints = [
            {"name": "Pre-drug", "color": "#58a6ff"},
            {"name": "Post-drug", "color": "#f78166"},
            {"name": "Baseline", "color": "#56d364"},
        ]
        self._metric = "logd_dist"   # the scroller now selects an export PANEL key
        self._view = "comparison"
        self._cfg = {
            "groupBy": "Condition", "test": "Mann–Whitney U",
            "correction": "Benjamini–Hochberg (FDR)", "alpha": "0.05",
            "err": "95% CI", "plot": "Violin", "logX": True,
            # stats-config (engine enum values where applicable) — feed _stats_config()
            "posthoc": "auto", "anova3plus": "welch", "control_group": "",
            "dunnett": False, "across_metric_correction": False,
            "ci_level": "0.95", "equivalence_tost": False, "tost_margin": "0.5",
            # circular-statistics suite (turning angles) — fed to compare_groups
            "include_circular_outputs": False, "circ_test_kappa": True,
            "circ_test_rbar": True, "circ_test_mu": True, "circ_test_circlin": False,
            # report output
            "outputStem": "comparison",
        }
        self._output_dir = ""   # "" → default to the first folder's parent
        # report-figure defaults live in Preferences (QSettings) — theme/logd_style/
        # mobile_d are read there at generate time; only the panel selection is held
        # here (persisted) and surfaced via the Preferences panel picker.
        self._panels = set(wd.DEFAULT_COMPARE_PANELS)
        self._busy = False
        self._panel_cond = 0
        self._panel_replicate = 0    # which folder for the per-replicate spatial panels
        self._panel_sel = 3          # MSD curves (a group-averageable panel)
        self._panel_image: QImage | None = None
        self._panel_token = 0
        self._panel_job = None
        # generation-tagged like the comparison figure, so a stale gallery-hero
        # render can't repaint after a panel / condition / replicate switch
        self._pending_panel: list = []     # [(gen, img)]
        # on-demand thumbnail rendering (the gallery grid): cache keyed by
        # (cond, panel, w, h, rev); rev bumps when condition data changes.
        self._panel_thumb_cache: dict = {}
        self._panel_rev = 0
        self._panel_render_lock = threading.Lock()
        # group-averaged fa_figure panels (All-panels view): one make_figure
        # render per group → {letter: QImage}, cached by (cond_idx, panel_rev).
        self._group_cache: dict = {}
        self._group_job = None
        self._group_job_key = None
        self._pending_group = None
        self._panel_group_rev = 0          # bumps when a group's panels finish
        self._presets = []

        # computed outputs (rebuilt by _recompute)
        self._cg: list[dict] = []          # the groups currently being compared
        self._paired = False
        self._headline: list[dict] = []
        self._stats_rows: list[dict] = []
        self._sig_rows: list[dict] = []
        self._omnibus: dict | None = None
        self._verdict: dict = {}
        self._twoway: list[dict] = []
        self._twoway_note = ""
        self._methods = ""
        self._legend: list[dict] = []

        # figure
        self._fig_image: QImage | None = None
        self._fig_token = 0
        self._fig_job = None
        # Async figure deliveries are generation-tagged and accumulated in lists
        # (drained under _fig_lock) so a late-finishing STALE render can never
        # overwrite a fresh one — the cause of the "says LIVE but shows the old
        # graph until you flick panels" staleness.  Generation = (engfig_rev,
        # metric): rev bumps on data/cfg/figure-pref change, metric on panel
        # switch, so a result only applies when both still match.
        self._fig_lock = threading.Lock()
        self._pending_image: list = []       # [(gen, img)] bespoke renders
        # full-engine report (fa_compare.compare_groups) lane
        self._report_job = None
        self._pending_report: dict | None = None
        self._report_busy = False
        self._last_report_dir = ""
        # real report progress (compare_groups calls progress_cb(done,total,msg)
        # off-thread; we stash the latest tuple and drain it on the GUI thread)
        self._report_progress = -1.0       # 0..1, or -1 → indeterminate
        self._report_status = ""
        self._report_prog_raw = None       # written off-thread by progress_cb
        self._report_prog_poll = QTimer(self)
        self._report_prog_poll.setInterval(80)
        self._report_prog_poll.timeout.connect(self._drain_report_progress)
        # live engine figure: ONE all-panels compare_groups render is sliced into
        # every panel (titles stripped so the narrow-cell slices stay clean), so
        # the ~9s compute is paid once and switching panels is then instant.
        self._engfig_job = None
        self._pending_engfig: list = []    # [(gen, key, img)] single-panel renders
        self._engfig_cache: dict = {}      # (panel_key, rev) → QImage  (fallback path)
        self._engfig_rev = 0               # bumps on ANY render-affecting change
        # ReportData cache: compute_report() is style/theme/panel-INDEPENDENT, so a
        # style or theme change reuses it and only re-runs render_report (the cheap
        # part).  `_data_rev` bumps ONLY on a data / stats-config / mobile-d change
        # (a subset of what bumps `_engfig_rev`); a matching `_data_rev` means the
        # cached ReportData is still valid.  Guarded by `_rd_lock` (worker threads).
        self._data_rev = 0
        self._rd_lock = threading.Lock()
        self._rd_cache = None              # the cached ReportData
        self._rd_cache_rev = -1            # the _data_rev it was computed for
        self._figureRendered.connect(self._on_figure_rendered)
        self._panelRendered.connect(self._on_panel_rendered)
        self._groupRendered.connect(self._on_group_rendered)
        self._reportRendered.connect(self._on_report_rendered)
        self._engfigRendered.connect(self._on_engfig_rendered)
        # async condition-folder loading: worker threads read run sidecars off
        # the GUI thread and hand back results through this queue + signal.
        self._load_q: queue.Queue = queue.Queue()
        self._loading_n = 0
        self._foldersLoaded.connect(self._on_folders_loaded)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._launch_figure)

        # Live-update the Analysis figures when a Figures preference changes
        # (graph styles, theme, mobile-D threshold): the cached renders are keyed
        # only by data-rev, so without this a style change never reaches the tab.
        if self._settings is not None:
            try:
                self._settings.changed.connect(self._on_figpref_changed)
            except (AttributeError, TypeError):
                pass

        self._load_persisted()
        if not self._conditions:
            self._seed_empty()
        self._recompute()

    # ── seeding / persistence ───────────────────────────────────────────
    def _new_cid(self):
        self._cid_seq += 1
        return self._cid_seq

    def _seed_empty(self):
        for i in range(2):
            self._conditions.append(
                _Condition(self._new_cid(), f"Condition {i + 1}",
                           wd.GROUP_COLORS[i % len(wd.GROUP_COLORS)]))

    def _load_persisted(self):
        s = self._settings
        if s is None:
            return
        try:
            m = s.get("analysis/metric", self._metric) or self._metric
            self._metric = m if m in wd.PANEL_KEYS else self._metric
            import json
            raw = s.get("analysis/presets", "")
            if raw:
                self._presets = json.loads(raw)
            pn = s.get("figures/compare_panels", "")
            if pn:
                keys = {k for k, _ in wd.COMPARE_PANELS}
                self._panels = {p for p in str(pn).split(",") if p in keys}
        except Exception:
            pass

    def _persist(self):
        s = self._settings
        if s is None:
            return
        try:
            import json
            s.setValue("analysis/metric", self._metric)
            s.setValue("analysis/presets", json.dumps(self._presets))
            s.setValue("figures/compare_panels", ",".join(sorted(self._panels)))
            s.sync()
        except Exception:
            pass

    # ── recompute pipeline ──────────────────────────────────────────────
    def _shown(self) -> list[_Condition]:
        return [c for c in self._conditions if len(c.active()) >= 2]

    def _metric_obj(self):
        # self._metric is a panel key; map it to the scalar metric that backs the
        # stats cards (falls back to the D metric for visualisation-only panels —
        # those panels hide the stats cards via `hasStats`, so it's never shown).
        return wd.METRIC_BY_ID.get(wd.PANEL_METRIC.get(self._metric, ""), wd.METRICS[0])

    def _has_metric(self) -> bool:
        """True when the selected panel has a scalar metric backing the stats
        cards (False for visualisation-only panels → figure only)."""
        return bool(wd.PANEL_METRIC.get(self._metric))

    def _values(self, cond: _Condition, metric) -> np.ndarray:
        vals = [metric.scalar(f.run) for f in cond.active()]
        return np.array([v for v in vals if v is not None], dtype=float)

    def _pooled_dist(self, cond: _Condition, metric):
        chunks = [metric.dist(f.run) for f in cond.active()]
        chunks = [c for c in chunks if c is not None and len(c)]
        return np.concatenate(chunks) if chunks else None

    def _build_groups(self):
        """Build the list of comparison groups for the current Group-by mode."""
        metric = self._metric_obj()
        shown = self._shown()
        paired = self._cfg["groupBy"].startswith("Timepoint")
        if paired:
            order = [t["name"] for t in self._timepoints]
            present = sorted({c.phase for c in shown if c.phase not in ("—", "")},
                             key=lambda n: order.index(n) if n in order else 99)
            groups = []
            for name in present:
                members = [c for c in shown if c.phase == name]
                tp = next((t for t in self._timepoints if t["name"] == name), None)
                vals, dists = [], []
                mc = {}
                for c in members:
                    vals.append(self._values(c, metric))
                    d = self._pooled_dist(c, metric)
                    if d is not None:
                        dists.append(d)
                    for f in c.active():
                        for k, v in (f.run.summary.get("motion_counts") or {}).items():
                            mc[k] = mc.get(k, 0) + int(v)
                ntracks = sum(f.run.n_tracks for c in members for f in c.active())
                groups.append({
                    "label": name, "color": tp["color"] if tp else wd.MOTION_COLORS["Unknown"],
                    "phase": name,
                    "values": np.concatenate(vals) if vals else np.array([]),
                    "dist": np.concatenate(dists) if dists else None,
                    "motion_counts": mc, "subjects": [c.name for c in members],
                    "ntracks": int(ntracks),
                })
            self._paired = len(groups) >= 2
            return (groups if self._paired else
                    self._plain_groups(shown, metric))
        self._paired = False
        return self._plain_groups(shown, metric)

    def _plain_groups(self, shown, metric):
        groups = []
        for c in shown:
            mc = {}
            for f in c.active():
                for k, v in (f.run.summary.get("motion_counts") or {}).items():
                    mc[k] = mc.get(k, 0) + int(v)
            groups.append({
                "label": _glabel(c.name, c.phase), "name": c.name, "color": c.color,
                "phase": c.phase, "values": self._values(c, metric),
                "dist": self._pooled_dist(c, metric), "motion_counts": mc,
                "ntracks": int(sum(f.run.n_tracks for f in c.active())),
            })
        return groups

    def _grouped_tp_data(self, metric) -> dict:
        """For the 'Grouped by timepoint' style: per-dish values keyed by
        condition NAME → timepoint (phase), so each group can show its timepoints
        side by side (the "between-dish variability" figure) — independent of the
        Group-by mode, which otherwise pools or separates them."""
        names, phases = [], []
        data: dict = {}
        colors: dict = {}
        for c in self._shown():
            v = self._values(c, metric)
            if v is None or not len(v):
                continue
            nm = c.name
            ph = c.phase if c.phase not in ("—", "") else ""
            if nm not in data:
                data[nm] = {}; names.append(nm); colors[nm] = c.color
            data[nm].setdefault(ph, []).append(np.asarray(v, dtype=float))
            if ph and ph not in phases:
                phases.append(ph)
        data = {nm: {ph: np.concatenate(arrs) for ph, arrs in tps.items()}
                for nm, tps in data.items()}
        return {"names": names, "phases": phases, "data": data, "colors": colors}

    def _engine_groups(self) -> list:
        """Folder-path groups for the real engine: one dict per shown condition
        carrying its analysis-output folders + its timepoint token (which drives
        the engine's one-way vs two-way (group×time) auto-selection)."""
        groups = []
        for c in self._shown():
            folders = [f.path for f in c.active()]
            if not folders:
                continue
            tp = "" if c.phase in ("—", "") else c.phase
            groups.append({"folders": folders, "label": c.name,
                           "color": c.color, "timepoint": tp})
        return groups

    def _stats_config(self) -> dict:
        """Translate the UI cfg into the engine's validated stats_config.  Display
        strings (test / correction) map to enum tokens; the richer keys (post-hoc,
        Dunnett, CI, TOST, circular …) pass straight through once their controls
        land, then everything is normalised by the engine."""
        from firefly.analysis import fa_stats_config as fsc
        cfg = self._cfg
        corr_map = {
            "None": "none", "Uncorrected": "none", "Bonferroni": "bonferroni",
            "Holm": "holm", "Holm–Bonferroni": "holm",
            "Benjamini–Hochberg (FDR)": "fdr_bh", "FDR (BH)": "fdr_bh",
            "Šidák": "sidak", "Sidak": "sidak", "Hochberg": "hochberg",
        }
        test_map = {   # display test → (parametric_strategy, nonparametric_test)
            "Auto": ("auto", "mann_whitney"),
            "Welch's t-test": ("force_parametric", "mann_whitney"),
            "Student's t-test": ("force_parametric", "mann_whitney"),
            "t-test": ("force_parametric", "mann_whitney"),
            "Mann–Whitney U": ("force_nonparametric", "mann_whitney"),
            "Brunner–Munzel": ("force_nonparametric", "brunner_munzel"),
            "Permutation": ("force_nonparametric", "permutation"),
            "Kruskal–Wallis": ("force_nonparametric", "mann_whitney"),
        }
        strat, nonpar = test_map.get(cfg.get("test", "Auto"), ("auto", "mann_whitney"))
        raw = {
            "alpha": float(cfg.get("alpha") or 0.05),
            "correction": corr_map.get(cfg.get("correction", "Holm"), "holm"),
            "parametric_strategy": cfg.get("parametric_strategy", strat),
            "nonparametric_test": cfg.get("nonparametric_test", nonpar),
            "anova3plus": cfg.get("anova3plus", "welch"),
            "posthoc": cfg.get("posthoc", "auto"),
            "across_metric_correction": bool(cfg.get("across_metric_correction", False)),
            "ci_level": float(cfg.get("ci_level", 0.95)),
            "figure_stars_use_corrected": bool(cfg.get("figure_stars_use_corrected", True)),
            "control_group": cfg.get("control_group", ""),
            "dunnett": bool(cfg.get("dunnett", False)),
            "equivalence_tost": bool(cfg.get("equivalence_tost", False)),
            "tost_margin": float(cfg.get("tost_margin", 0.5)),
            "include_circular_outputs": bool(cfg.get("include_circular_outputs", False)),
            "circ_test_kappa": bool(cfg.get("circ_test_kappa", True)),
            "circ_test_rbar": bool(cfg.get("circ_test_rbar", True)),
            "circ_test_mu": bool(cfg.get("circ_test_mu", True)),
            "circ_test_circlin": bool(cfg.get("circ_test_circlin", False)),
        }
        return fsc.normalize_stats_config(raw)

    def _recompute(self):
        metric = self._metric_obj()
        groups = self._build_groups()
        self._cg = groups
        n_cond = len(groups)
        alpha = float(self._cfg.get("alpha") or 0.05)
        has_metric = self._has_metric()

        if has_metric and n_cond >= 2:
            all_vals = np.concatenate([g["values"] for g in groups if len(g["values"])]) \
                if any(len(g["values"]) for g in groups) else np.array([])
            means = [float(np.mean(g["values"])) if len(g["values"]) else float("nan")
                     for g in groups]
            ntot = int(sum(g_run_n(g) for g in groups))
            pooled = (float(np.average([m for m in means if np.isfinite(m)]))
                      if any(np.isfinite(m) for m in means) else float("nan"))
            finite_means = [m for m in means if np.isfinite(m)]
            lo = min(finite_means) if finite_means else float("nan")
            hi = max(finite_means) if finite_means else float("nan")
            # Unpaired → the real engine (_stat_test_n) so live numbers match the
            # report; paired (timepoint) keeps the quick Wilcoxon path until the
            # two-way ANOVA surfacing lands.
            if self._paired:
                sig = wd.pairwise_stats(
                    groups, metric, test=self._cfg["test"],
                    correction=self._cfg["correction"], alpha=alpha, paired=True)
                self._omnibus = None
                self._verdict = {}
            else:
                sig, self._omnibus, pw_raw = wd.engine_pairwise_stats(
                    groups, self._stats_config())
                self._verdict = self._build_verdict(metric, self._omnibus, pw_raw, n_cond)
            n_sig = sum(1 for r in sig if r["sig"])

            self._headline = [
                {"label": "Total tracks", "value": f"{ntot:,}", "unit": "",
                 "color": "#58a6ff", "jump": False},
                {"label": "Conditions", "value": str(n_cond), "unit": "",
                 "color": "", "jump": False},
                {"label": f"Pooled {metric.label}", "value": metric.fmt(pooled),
                 "unit": metric.unit, "color": "", "jump": False},
                {"label": "Range across conditions",
                 "value": f"{metric.fmt(lo)}–{metric.fmt(hi)}", "unit": metric.unit,
                 "color": "", "jump": False},
                {"label": "Max difference", "value": metric.fmt(hi - lo),
                 "unit": metric.unit, "color": "#f6a623", "jump": False},
                {"label": "Significant pairs", "value": f"{n_sig} / {len(sig)}",
                 "unit": "", "color": "#56d364" if n_sig else "#5b636e", "jump": True},
            ]
            hi_bar = max([m for m in means if np.isfinite(m)] or [1.0]) or 1.0
            self._stats_rows = []
            for g, m in zip(groups, means):
                self._stats_rows.append({
                    "label": g["label"], "color": g["color"],
                    "n": f"{g_run_n(g):,}",
                    "value": metric.fmt(m if np.isfinite(m) else None),
                    "err": (f"±{np.std(g['values'], ddof=1):.3f}"
                            if metric.id == "D" and len(g["values"]) > 1 else ""),
                    "barFrac": max(0.04, (m / hi_bar) if (np.isfinite(m) and hi_bar) else 0.04),
                })
            self._sig_rows = [self._sig_row(r) for r in sig]
            # Surface the overall (omnibus) test as the lead row for 3+ groups.
            if self._omnibus is not None and n_cond >= 3:
                self._sig_rows.insert(0, self._omnibus_row(self._omnibus, alpha))
            self._twoway, self._twoway_note = self._twoway_rows()
            self._methods = self._build_methods(groups, metric, alpha)
            self._legend = self._build_legend(groups, metric)
        else:
            self._headline = self._stats_rows = self._sig_rows = []
            self._twoway = []
            self._twoway_note = ""
            self._verdict = {}
            self._methods = ""
            self._legend = []

        self.resultsChanged.emit()
        # kick the (debounced) figure render
        self._set_busy(True)
        self._debounce.start()
        if self._view == "panels":
            self._render_panel_async()

    def _sig_row(self, r):
        mag = r["magnitude"]
        mag_color = {"large": "#56d364", "medium": "#e6edf3",
                     "small": "#8b949e", "negligible": "#5b636e"}[mag]
        a, b = r["a"], r["b"]
        if self._paired:
            label = f"{a.get('label')} vs {b.get('label')}"
        else:
            label = f"{a['label']} vs {b['label']}"
        p = r["p"]
        p_txt = ("—" if p is None or not np.isfinite(p)
                 else f"{p:.1e}" if p < 0.001 else f"{p:.3f}")
        return {
            "aColor": a["color"], "bColor": b["color"], "label": label,
            "delta": f"δ {abs(r['delta']):.2f}", "mag": mag, "magColor": mag_color,
            "p": f"p = {p_txt}", "stars": r["stars"], "sig": r["sig"],
        }

    def _omnibus_row(self, o, alpha):
        """Display row for the overall (omnibus) test — Welch / one-way ANOVA or
        Kruskal across all conditions — shown above the pairwise rows for 3+ groups."""
        p = o.get("p")
        p_txt = ("—" if p is None or not np.isfinite(p)
                 else f"{p:.1e}" if p < 0.001 else f"{p:.3f}")
        es, kind = o.get("effect_size"), o.get("effect_size_kind")
        sym = "η²" if kind == "eta_sq" else "ε²" if kind == "epsilon_sq" else ""
        es_txt = (f"{sym} {es:.2f}" if (es is not None and np.isfinite(es) and sym)
                  else "overall")
        mag = wd.es_magnitude(es)
        mag_color = {"large": "#56d364", "medium": "#e6edf3", "small": "#8b949e",
                     "negligible": "#5b636e", "": "#8b949e"}[mag]
        return {
            "aColor": "#8b949e", "bColor": "#8b949e",
            "label": "All conditions · " + (o.get("test") or "overall"),
            "delta": es_txt, "mag": mag or "overall", "magColor": mag_color,
            "p": f"p = {p_txt}", "stars": o.get("stars", ""),
            "sig": bool(p is not None and np.isfinite(p) and p < alpha),
        }

    def _build_verdict(self, metric, omnibus, pw_raw, n_groups):
        """Plain-language one-liner for the selected metric, via the (non-core)
        results_format helper.  Returns {severity, html, underpowered} or {}."""
        if n_groups < 2 or (omnibus is None and not pw_raw):
            return {}
        try:
            from firefly.ui import results_format as rf
            sev, html, under = rf._verdict_for_metric(
                metric.label, {"omnibus": omnibus or {}, "pairwise": pw_raw or []},
                n_groups)
            return {"severity": sev, "html": html, "underpowered": bool(under)}
        except Exception:
            return {}

    def _twoway_rows(self):
        """Live two-way mixed ANOVA (group × time) for the selected metric, via the
        engine's fa_twoway — the same model the report runs.  Returns (rows, note):
        rows = the two main effects + interaction; note = a status/warning string
        ('' when clean).  Empty unless the design is genuinely factorial (≥2
        condition names AND ≥2 timepoints)."""
        shown = self._shown()
        phases = {c.phase for c in shown if c.phase not in ("—", "")}
        names = {c.name for c in shown if c.phase not in ("—", "")}
        if len(phases) < 2 or len(names) < 2:
            return [], ""
        try:
            import pandas as pd
            from firefly.analysis import fa_twoway as tw
            from firefly.analysis.fa_stats_config import stars_for
            if not tw.HAVE_PINGOUIN:
                return [], "Two-way ANOVA needs the ‘pingouin’ package."
            metric = self._metric_obj()
            tokens = [t["name"] for t in self._timepoints]
            recs = []
            for c in shown:
                if c.phase in ("—", ""):
                    continue
                for f in c.active():
                    v = metric.scalar(f.run)
                    if v is None:
                        continue
                    stem = getattr(f.run, "stem", "") or os.path.basename(f.path)
                    cell, _ = tw.derive_subject_key(stem, tokens)
                    recs.append({"group": c.name, "timepoint": c.phase,
                                 "cell": cell, "value": float(v)})
            if len(recs) < 4:
                return [], "Not enough paired replicates for a two-way ANOVA."
            df = pd.DataFrame(recs)
            clean, warning, _ = tw.validate_pairing(df)
            res, msg = tw.compute_twoway_anova(clean, metrics=["value"],
                                               stats_config=self._stats_config())
            if res is None:
                return [], (warning or msg or "Two-way ANOVA unavailable.")
            alpha = float(self._cfg.get("alpha") or 0.05)
            disp = {"group": "Condition (between)", "timepoint": "Timepoint (within)",
                    "Interaction": "Group × Time"}
            fmt = lambda x, d=1: (f"{x:.{d}f}" if (x is not None and np.isfinite(x)) else "—")
            dfi = lambda x: (str(int(x)) if (x is not None and np.isfinite(x)) else "—")
            rows = []
            for _, r in res[res["section"] == "anova"].iterrows():
                eff = r["effect"]
                if eff == "ERROR":
                    return [], "Two-way ANOVA failed: " + str(r.get("detail", ""))[:70]
                pgg = r.get("p_GG")
                p = pgg if (pgg is not None and np.isfinite(pgg)) else r.get("p_unc")
                np2 = r.get("np2")
                p_txt = ("—" if p is None or not np.isfinite(p)
                         else f"{p:.1e}" if p < 0.001 else f"{p:.3f}")
                rows.append({
                    "effect": disp.get(eff, eff),
                    "stat": f"F({dfi(r.get('df1'))},{dfi(r.get('df2'))}) = {fmt(r.get('F'))}",
                    "p": "p = " + p_txt,
                    "eta": (f"η²ₚ {np2:.2f}" if (np2 is not None and np.isfinite(np2)) else ""),
                    "stars": stars_for(p, alpha),
                    "sig": bool(p is not None and np.isfinite(p) and p < alpha),
                })
            return rows, (warning or "")
        except Exception as e:                              # pragma: no cover
            return [], "Two-way ANOVA failed: " + str(e)[:80]

    def _build_methods(self, groups, metric, alpha):
        ntot = int(sum(g_run_n(g) for g in groups))
        corr = self._cfg["correction"]
        corr_txt = ("" if corr == "None"
                    else f" with {corr.replace(' (FDR)', '')} correction")
        unit = "timepoints" if self._paired else "conditions"
        return (f"{metric.label} was compared across {len(groups)} {unit} "
                f"(n = {ntot:,} localisations) using {self._cfg['test']}{corr_txt} "
                f"(α = {self._cfg['alpha']})"
                + (", paired by timepoint" if self._paired else "") + ".")

    def _build_legend(self, groups, metric):
        if metric.id == "motion":
            return [{"name": k, "color": wd.MOTION_COLORS[k]}
                    for k in ("Immobile", "Confined", "Brownian", "Directed")]
        return [{"name": g["label"], "color": g["color"]} for g in groups]

    # ── figure lane ─────────────────────────────────────────────────────
    def _compute_report_kwargs(self) -> dict:
        """The style/theme-INDEPENDENT ``compute_report`` kwargs — these (with
        ``_data_rev``) identify the cached ReportData.  Only a data / stats-config /
        mobile-d change alters them, and those all bump ``_data_rev``."""
        s = self._settings
        try:
            mobile_d = float(s.get("analysis/mobile_d", 0.05)) if s else 0.05
        except (TypeError, ValueError):
            mobile_d = 0.05
        return dict(groups=self._engine_groups(), mobile_d_threshold=mobile_d,
                    stats_config=self._stats_config())

    def _render_report_kwargs(self, panels) -> dict:
        """``render_report`` kwargs — theme / graph style / panel selection / clip:
        the parts that can change WITHOUT invalidating the cached ReportData."""
        s = self._settings
        theme = (s.getStr("figures/theme", "Dark") if s else "Dark")
        if theme not in ("Dark", "Light", "Publication"):
            theme = "Dark"
        dlo, dhi = self._dcoeff_clip()
        return dict(
            panels=panels, theme=theme,
            logd_plot_style=(s.getStr("figures/logd_style", "overlaid") if s else "overlaid"),
            msd_plot_style=(s.getStr("figures/msd_style", "mean_faceted") if s else "mean_faceted"),
            msd_err=self._cfg.get("err", "SEM"),
            auc_plot_style=(s.getStr("figures/auc_style", "box_points") if s else "box_points"),
            group_style=(s.getStr("figures/group_style", "box_points") if s else "box_points"),
            panel_styles=self._panel_styles(),
            logd_clip_d_min=dlo, logd_clip_d_max=dhi)

    # Comparison panels drawn as a scalar bar/box/violin — each gets its OWN
    # format in Preferences (figures/style_<key>).  AUC is NOT here: its single
    # "MSD-AUC" control (figures/auc_style) also offers the paired/Δ timepoint
    # views, so it's handled through auc_plot_style instead.
    SCALAR_STYLE_PANELS = ("fluor", "mob_immob", "track_count", "van_hove", "vacf")

    def _panel_styles(self) -> dict:
        """Per-panel comparison mark (box_points / violin / bar) from Preferences —
        one setting per scalar comparison graph."""
        s = self._settings
        if s is None:
            return {}
        return {k: s.getStr(f"figures/style_{k}", "box_points")
                for k in self.SCALAR_STYLE_PANELS}

    def _cached_report_data(self, compute_kwargs, data_rev):
        """Return the ReportData for ``data_rev`` — reuse the cache when its rev
        still matches, else compute (outside the lock) and store it.  Called from
        the render worker threads; the loads run off the GUI thread as before."""
        with self._rd_lock:
            if self._rd_cache_rev == data_rev and self._rd_cache is not None:
                return self._rd_cache
        from firefly.analysis.fa_compare import compute_report
        rd = compute_report(**compute_kwargs)
        with self._rd_lock:
            self._rd_cache = rd
            self._rd_cache_rev = data_rev
        return rd

    def _make_engine_render(self, render_kwargs, compute_progress_cb=None):
        """Build the zero-arg callable a render job runs: fetch the cached
        ReportData (compute once per ``_data_rev``) then ``render_report`` on it, so
        a style/theme change re-renders without recomputing scalars or statistics.
        ``compute_progress_cb`` (report lane only) drives the loading bar while the
        ReportData is (re)computed — skipped on a cache hit, which is already
        instant.  Returns ``(fig, summary_df, stats)`` — the compare_groups contract."""
        compute_kwargs = self._compute_report_kwargs()
        if compute_progress_cb is not None:
            compute_kwargs["progress_cb"] = compute_progress_cb
        data_rev = self._data_rev
        ref = weakref.ref(self)

        def _render():
            from firefly.analysis.fa_compare import render_report, compare_groups
            c = ref()
            if c is None:                          # controller gone → one-shot
                return compare_groups(**compute_kwargs, **render_kwargs)
            rd = c._cached_report_data(compute_kwargs, data_rev)
            return render_report(rd, **render_kwargs)
        return _render

    def _dcoeff_clip(self):
        """(min, max) D-coefficient clip range (µm²/s) for the LogD graph, read
        from the Diffusion-&-motion setting.  Defaults 1e-5…10 → log₁₀ −5…1."""
        s = self._settings
        try:
            lo = float(s.get("analysis/dcoeff_clip_min", 0.00001)) if s else 0.00001
            hi = float(s.get("analysis/dcoeff_clip_max", 10.0)) if s else 10.0
        except (TypeError, ValueError):
            lo, hi = 0.00001, 10.0
        return lo, hi

    def _fig_gen(self):
        """The identity of the figure the user is currently looking at: bumps on
        any data / cfg / figure-preference change (via ``_engfig_rev``) AND on a
        panel switch (``_metric``).  Panel-specific async renders capture this at
        launch and are dropped on delivery if it no longer matches."""
        return (self._engfig_rev, self._metric)

    def _drain_pending(self, lst):
        """Atomically take + clear a generation-tagged delivery list (workers
        append under the same lock), so no in-flight append is lost mid-drain."""
        with self._fig_lock:
            items = lst[:]
            del lst[:]
        return items

    def _launch_figure(self):
        if len(self._cg) < 2 or self._paired:
            # paired view is drawn in QML (SVG); no matplotlib figure
            self._set_busy(False)
            return
        panel = self._metric                       # the scroller selects a panel key
        if not panel or len(self._engine_groups()) < 2:
            if self._has_metric():
                self._launch_bespoke_figure(self._metric_obj())
            else:
                self._set_busy(False)
            return
        # Render THIS panel as its own single-panel engine figure (cached per
        # (panel, rev)).  Correct by construction — the panel IS what it draws, no
        # slice-mapping — and cheap now that compute_report is cached (the heavy
        # load + stats are shared across panels; a panel is just its ~10-30 ms draw).
        # The old "render the whole 6×3 grid once and slice it" path was replaced:
        # its geometry-based key→cell mapping mis-assigned panels (faceted/polar
        # panels remove or reorder their axes), which had been masked while the
        # all-panels render was crashing on the theme bug.
        self._launch_single_panel(panel)

    def _launch_single_panel(self, panel):
        """Render ONE export panel via the real engine (compare_groups/render_report
        with a single panel) and cache it per (panel, rev).  This is THE live-figure
        path: correct by construction and cheap because compute_report is cached.
        Reuses _EngineFigJob + header crop."""
        if not panel or len(self._engine_groups()) < 2:
            if self._has_metric():
                self._launch_bespoke_figure(self._metric_obj())
            else:
                self._set_busy(False)
            return
        key = (panel, self._engfig_rev)
        cached = self._engfig_cache.get(key)
        if cached is not None and not cached.isNull():
            self._fig_image = cached
            self._fig_token += 1
            self._set_busy(False)
            self.figureChanged.emit()
            return
        render_fn = self._make_engine_render(self._render_report_kwargs({panel}))
        self._set_busy(True)
        gen = self._fig_gen()
        ref = weakref.ref(self)

        def deliver(img):
            c = ref()
            if c is not None:
                with c._fig_lock:
                    c._pending_engfig.append((gen, key, img))
                c._engfigRendered.emit()

        self._engfig_job = _EngineFigJob(render_fn, deliver)
        self._engfig_job.start()

    def _launch_bespoke_figure(self, metric):
        """Fast non-engine preview (used for metrics without an export panel, and
        as a fallback if the engine render fails)."""
        gen = self._fig_gen()
        ref = weakref.ref(self)

        def deliver(img):
            c = ref()
            if c is not None:
                c._deliver_figure(gen, img)

        cfg = dict(self._cfg)
        s = self._settings
        if s is not None:
            cfg["_logd_style"] = s.getStr("figures/logd_style", "overlaid")
            cfg["_group_style"] = s.getStr("figures/group_style", "box_points")
            cfg["_length_style"] = s.getStr("figures/length_style", "density")
            try:
                cfg["_mobile_d"] = float(s.get("analysis/mobile_d", 0.05))
            except (TypeError, ValueError):
                cfg["_mobile_d"] = 0.05
            cfg["_logd_clip_min"], cfg["_logd_clip_max"] = self._dcoeff_clip()
            # Grouped-by-timepoint style needs the per-name × phase breakdown
            # (the live groups otherwise pool it away) — compute it for this metric.
            if cfg.get("_group_style") == "grouped":
                try:
                    cfg["_grouped_data"] = self._grouped_tp_data(metric)
                except Exception:
                    cfg["_grouped_data"] = None
        self._fig_job = _FigureJob(self._cg, metric, cfg, (760, 400), deliver)
        self._fig_job.start()

    def _on_engfig_rendered(self):
        # Keep only deliveries whose generation still matches (same rev + panel);
        # a stale single-panel render is dropped so it can't overwrite the live
        # figure or poison the rev-keyed cache under a since-reused key.
        cur = self._fig_gen()
        fresh = [(k, im) for (g, k, im) in self._drain_pending(self._pending_engfig)
                 if g == cur]
        if not fresh:
            return                                # every result is stale → ignore
        self._engfig_job = None
        key, img = fresh[-1]
        if img is not None and not img.isNull():
            self._engfig_cache[key] = img
            self._fig_image = img
            self._fig_token += 1
            self._set_busy(False)
            self.figureChanged.emit()
        elif self._has_metric():                 # engine failed → bespoke fallback
            self._launch_bespoke_figure(self._metric_obj())
        else:                                    # viz-only panel → no preview
            self._set_busy(False)

    def _deliver_figure(self, gen, img):
        # called on the worker thread — stash (generation-tagged) + signal
        with self._fig_lock:
            self._pending_image.append((gen, img))
        self._figureRendered.emit()

    def _on_figure_rendered(self):
        cur = self._fig_gen()
        fresh = [im for (g, im) in self._drain_pending(self._pending_image)
                 if g == cur]
        if not fresh:
            return                                # stale bespoke render → ignore
        if fresh[-1] is not None and not fresh[-1].isNull():
            self._fig_image = fresh[-1]
            self._fig_token += 1
        self._set_busy(False)
        self.figureChanged.emit()

    def figure_image(self) -> QImage:
        if self._fig_image is not None:
            return self._fig_image
        return QImage(2, 2, QImage.Format.Format_RGB888)

    # ── full report lane (real fa_compare engine) ───────────────────────
    @Slot()
    def generateComparison(self):
        """Run the full engine and write the complete artefact bundle (multi-panel
        figure + Prism stats CSV + two-way CSV + PDF report + results JSON) to the
        output folder.  Live preview stays instant; this is the explicit action."""
        if self._report_busy:
            return
        groups = self._engine_groups()
        if len(groups) < 2:
            self.toast.emit("Add at least two conditions with run folders")
            return
        out_dir = (self._output_dir or self._export_dir())
        ref = weakref.ref(self)

        def deliver(result):
            c = ref()
            if c is not None:
                c._pending_report = result
                c._reportRendered.emit()

        def progress(done, total, msg):
            # off-thread: stash only (weakref so the worker never keeps the
            # controller alive / destroys it from this thread)
            c = ref()
            if c is not None:
                c._report_prog_raw = (int(done), int(total), str(msg))

        # Same split as the live tab: the report reuses the cached ReportData (so
        # data loaded for the live figure isn't re-read) and renders the full
        # bundle to disk.  Figure defaults come from Preferences (Figures section).
        render_kwargs = self._render_report_kwargs(
            set(self._panels) if self._panels else None)
        render_kwargs.update(
            output_dir=out_dir,
            output_stem=(self._cfg.get("outputStem") or "comparison"),
            pdf_report=True)
        render_fn = self._make_engine_render(render_kwargs, compute_progress_cb=progress)

        self._report_busy = True
        self._report_progress = 0.0
        self._report_status = "Starting…"
        self._report_prog_raw = None
        self._report_prog_poll.start()
        self.reportChanged.emit()

        self._report_job = _ReportJob(render_fn, out_dir, deliver)
        self._report_job.start()

    def _drain_report_progress(self):
        raw = self._report_prog_raw
        if raw is None:
            return
        self._report_prog_raw = None
        done, total, msg = raw
        # determinate through the load phase (show the folder count — more concrete
        # than a %); once everything's loaded the engine renders for a while with no
        # further callbacks → indeterminate
        if total and done < total:
            self._report_progress = done / total
            self._report_status = f"{msg}  ({done}/{total})"
        else:
            self._report_progress = -1.0
            self._report_status = msg
        self.reportChanged.emit()

    def _on_report_rendered(self):
        r = self._pending_report or {}
        self._pending_report = None
        self._report_busy = False
        self._report_prog_poll.stop()
        self._report_progress = -1.0
        self._report_status = ""
        self.reportChanged.emit()
        if r.get("ok"):
            self._last_report_dir = r.get("dir", "")
            self.toast.emit("Full report generated")
            self.reportReady.emit(self._last_report_dir)
            if self._last_report_dir:
                QDesktopServices.openUrl(QUrl.fromLocalFile(self._last_report_dir))
        else:
            self.toast.emit("Report failed: " + (r.get("error") or "unknown error")[:90])

    @Property(bool, notify=reportChanged)
    def reportBusy(self):
        return self._report_busy

    @Property(float, notify=reportChanged)
    def reportProgress(self):
        return self._report_progress      # 0..1 during load, -1 → indeterminate

    @Property(str, notify=reportChanged)
    def reportStatus(self):
        return self._report_status

    @Property(str, notify=reportChanged)
    def lastReportDir(self):
        return self._last_report_dir

    @Property(str, notify=cfgChanged)
    def outputDir(self):
        """Where the full report lands — the chosen folder, else the default."""
        return self._output_dir or self._export_dir()

    @Slot()
    def chooseOutputDir(self):
        from PySide6.QtWidgets import QFileDialog
        d = QFileDialog.getExistingDirectory(None, "Choose report output folder",
                                             self.outputDir)
        if d:
            self._output_dir = d
            self.cfgChanged.emit()

    @Slot()
    def openPreviousComparison(self):
        """Open a previously generated comparison report (its PDF) for review.  A
        full snapshot reload into the *live* workspace isn't meaningful — the view
        always recomputes from the loaded folders — so we open the saved report."""
        from PySide6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            None, "Open a previous comparison report", self.outputDir,
            "Comparison report (*_report.pdf *.pdf *_results.json);;All files (*)")
        if not path:
            return
        if path.endswith("_results.json"):
            pdf = path[:-len("_results.json")] + "_report.pdf"
            if os.path.exists(pdf):
                path = pdf
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _set_busy(self, b):
        if b != self._busy:
            self._busy = b
            self.busyChanged.emit()

    # ── panels lane ─────────────────────────────────────────────────────
    def _panel_gen(self):
        """Identity of the gallery hero the user is looking at — bumps on a data
        change (``_panel_rev``) or a condition / panel / replicate switch."""
        return (self._panel_rev, self._panel_cond, self._panel_sel,
                self._panel_replicate)

    def _render_panel_async(self):
        shown = self._shown()
        if not shown:
            return
        ci = min(max(0, self._panel_cond), len(shown) - 1)
        cond = shown[ci]
        panel = wd.PANELS[self._panel_sel]
        gen = self._panel_gen()
        ref = weakref.ref(self)

        def deliver(img):
            c = ref()
            if c is not None:
                c._deliver_panel(gen, img)

        # Averageable panels → the EXACT fa_figure panel, pooled over the group's
        # folders.  ONE make_figure render per group fills a cache of all panels,
        # so panel-switching within a group is instant.
        letter = wd.MFIG_LETTER.get(self._panel_sel)
        if letter is not None:
            if not self._show_group_panel():       # cache miss → render the group
                self._warm_group_cache()
            return

        runs = [f.run for f in cond.active()]
        # spatial/raster panels can't be pooled → show the chosen replicate folder
        if panel.get("kind") == "raster" and runs:
            ri = min(max(0, self._panel_replicate), len(runs) - 1)
            runs = [runs[ri]]
        self._panel_job = _PanelJob(panel, runs, cond.color, (760, 420), deliver)
        self._panel_job.start()

    def _group_key(self):
        shown = self._shown()
        if not shown:
            return None
        ci = min(max(0, self._panel_cond), len(shown) - 1)
        return (ci, self._panel_rev)

    def _show_group_panel(self) -> bool:
        """Show the selected panel from the group cache, if present."""
        letter = wd.MFIG_LETTER.get(self._panel_sel)
        key = self._group_key()
        if letter is None or key is None:
            return False
        cached = self._group_cache.get(key)
        img = cached.get(letter) if cached else None
        if img is not None and not img.isNull():
            self._panel_image = img
            self._panel_token += 1
            self.panelImageChanged.emit()
            return True
        return False

    def _warm_group_cache(self):
        """Render the current group's averageable panels (one make_figure call)
        into the cache, off-thread — so the hero AND the gallery thumbnails are
        instant.  No-op if already cached or already rendering this group."""
        key = self._group_key()
        if key is None or key in self._group_cache:
            return
        if self._group_job is not None and self._group_job_key == key:
            return
        shown = self._shown()
        cond = shown[key[0]]
        folders = [f.path for f in cond.active()]
        if not folders:
            return
        s = self._settings
        theme = (s.getStr("figures/theme", "Dark") if s else "Dark")
        self._group_job_key = key
        gref = weakref.ref(self)

        def gdeliver(d):
            c = gref()
            if c is not None:
                c._pending_group = (key, d)
                c._groupRendered.emit()

        self._group_job = _GroupAllPanelsJob(folders, theme, cond.color, gdeliver)
        self._group_job.start()

    def _on_group_rendered(self):
        item = self._pending_group
        self._pending_group = None
        self._group_job = None
        self._group_job_key = None
        if item is None:
            return
        key, d = item
        if d:
            self._group_cache[key] = d
            self._panel_group_rev += 1             # → gallery thumbnails re-request
            self.panelGroupRevChanged.emit()
        self._show_group_panel()                   # show current selection if ready

    def _deliver_panel(self, gen, img):
        with self._fig_lock:
            self._pending_panel.append((gen, img))
        self._panelRendered.emit()

    def _on_panel_rendered(self):
        cur = self._panel_gen()
        fresh = [im for (g, im) in self._drain_pending(self._pending_panel)
                 if g == cur]
        if not fresh:
            return                                # stale hero render → ignore
        self._panel_job = None
        if fresh[-1] is not None and not fresh[-1].isNull():
            self._panel_image = fresh[-1]
            self._panel_token += 1
        self.panelImageChanged.emit()

    def panel_image(self) -> QImage:
        if self._panel_image is not None:
            return self._panel_image
        return QImage(2, 2, QImage.Format.Format_RGB888)

    def render_panel_for(self, cond_idx: int, panel_idx: int, w: int, h: int) -> QImage:
        """Render one gallery panel for a condition at the requested size — used
        by the thumbnail image provider.  Cached per (cond, panel, size, rev);
        the lock serialises the matplotlib renders so concurrent thumbnail
        requests can't trample each other."""
        blank = QImage(2, 2, QImage.Format.Format_RGB888)
        # Group-averaged fa_figure panel (lettered) → the cached render, scaled.
        # Falls through to the existing renderer as a placeholder until the
        # group's panels finish (then panelGroupRev bumps → thumbnail re-requests).
        if 0 <= panel_idx < len(wd.PANELS) and wd.PANELS[panel_idx].get("letter"):
            letter = wd.PANELS[panel_idx]["letter"]
            cached = self._group_cache.get((cond_idx, self._panel_rev))
            gimg = cached.get(letter) if cached else None
            if gimg is not None and not gimg.isNull():
                from PySide6.QtCore import Qt
                return gimg.scaled(max(40, w), max(30, h),
                                   Qt.AspectRatioMode.KeepAspectRatio,
                                   Qt.TransformationMode.SmoothTransformation)
            # not rendered yet → stay blank (don't flash the old-style render);
            # the panelGroupRev bump re-requests once the group finishes.
            empty = QImage(2, 2, QImage.Format.Format_ARGB32)
            empty.fill(0)
            return empty
        with self._panel_render_lock:
            key = (cond_idx, panel_idx, w, h, self._panel_rev)
            cached = self._panel_thumb_cache.get(key)
            if cached is not None:
                return cached
            shown = self._shown()
            if not (0 <= cond_idx < len(shown)) or not (0 <= panel_idx < len(wd.PANELS)):
                return blank
            cond = shown[cond_idx]
            runs = [f.run for f in cond.active()]
            try:
                img = wf.render_panel(wd.PANELS[panel_idx], runs, cond.color,
                                      width_px=max(40, w), height_px=max(30, h), dpi=96)
            except Exception:
                img = blank
            if img is None or img.isNull():
                img = blank
            self._panel_thumb_cache[key] = img
            return img

    @Property("QVariantList", notify=resultsChanged)
    def panelConditions(self):
        return [{"label": _glabel(c.name, c.phase), "color": c.color,
                 "n": len(c.active())}
                for c in self._shown()]

    @Property("QVariantList", constant=True)
    def panelCategories(self):
        out = []
        for cat in wd.PANEL_CATS:
            items = [{"idx": i, "name": p["name"]}
                     for i, p in enumerate(wd.PANELS) if p["cat"] == cat]
            out.append({"cat": cat, "count": len(items), "items": items})
        return out

    @Property(int, notify=panelChanged)
    def panelSel(self):
        return self._panel_sel

    @Property(int, notify=panelChanged)
    def panelCondIdx(self):
        return min(self._panel_cond, max(0, len(self._shown()) - 1))

    @Property(str, notify=panelChanged)
    def panelHeroName(self):
        return wd.PANELS[self._panel_sel]["name"]

    @Property(str, notify=panelChanged)
    def panelHeroCat(self):
        return wd.PANELS[self._panel_sel]["cat"]

    @Property(str, notify=panelChanged)
    def panelIndexLabel(self):
        return f"{self._panel_sel + 1:02d} / {len(wd.PANELS)}"

    @Property(int, notify=panelRevChanged)
    def panelDataRev(self):
        return self._panel_rev

    @Property(int, notify=panelGroupRevChanged)
    def panelGroupRev(self):
        return self._panel_group_rev

    @Property(bool, notify=panelChanged)
    def panelIsSpatial(self):
        # spatial/heat-map panels (raster) can't be averaged → per-replicate
        p = wd.PANELS[self._panel_sel] if 0 <= self._panel_sel < len(wd.PANELS) else {}
        return p.get("kind") == "raster"

    @Property("QVariantList", notify=panelChanged)
    def panelReplicates(self):
        shown = self._shown()
        if not shown:
            return []
        ci = min(max(0, self._panel_cond), len(shown) - 1)
        out = []
        for f in shown[ci].active():
            stem = getattr(f.run, "stem", "") or os.path.basename(str(f.path).rstrip("/\\"))
            out.append({"label": stem or "folder"})
        return out

    @Property(int, notify=panelChanged)
    def panelReplicateIdx(self):
        return self._panel_replicate

    @Slot(int)
    def setPanelReplicate(self, idx):
        if idx != self._panel_replicate:
            self._panel_replicate = idx
            self.panelChanged.emit()
            self._render_panel_async()

    @Property(int, constant=True)
    def panelCount(self):
        return len(wd.PANELS)

    @Property(int, notify=panelImageChanged)
    def panelToken(self):
        return self._panel_token

    @Property(bool, notify=panelImageChanged)
    def hasPanel(self):
        return self._panel_image is not None

    @Slot(int)
    def setPanelCond(self, idx):
        self._panel_cond = idx
        self._panel_replicate = 0          # new group → reset to its first folder
        self.panelChanged.emit()
        self._render_panel_async()
        self._warm_group_cache()           # warm this group for the thumbnails

    @Slot(int)
    def setPanelSel(self, idx):
        if 0 <= idx < len(wd.PANELS):
            self._panel_sel = idx
            self.panelChanged.emit()
            self._render_panel_async()

    # ── change entry-point ──────────────────────────────────────────────
    def _changed(self, *, conditions=False, timepoints=False):
        if conditions:
            self.conditionsChanged.emit()
        if timepoints:
            self.timepointsChanged.emit()
        # data/grouping changed → invalidate the thumbnail cache and bump the
        # rev so the gallery thumbnails re-request.
        self._panel_rev += 1
        self._panel_thumb_cache.clear()
        self._group_cache.clear()
        self.panelRevChanged.emit()
        self._engfig_rev += 1              # data changed → invalidate engine-fig cache
        self._data_rev += 1                # …and the cached ReportData (scalars/stats)
        self._engfig_cache.clear()
        self._recompute()

    def _on_figpref_changed(self, key):
        """A Figures preference changed (graph styles, theme, mobile-D threshold)
        → drop every cached render and redraw the current view live.  Reacts to
        the whole ``figures/`` namespace so new graph-style keys work for free;
        ``figures/compare_panels`` is skipped (its own toggle path re-renders,
        and the controller writes it itself — reacting would loop)."""
        if not isinstance(key, str):
            return
        if not (key.startswith("figures/")
                or key in ("analysis/mobile_d",
                           "analysis/dcoeff_clip_min", "analysis/dcoeff_clip_max")):
            return
        if key == "figures/compare_panels":
            return
        # Invalidate every cached RENDER (engine figure + sliced panels + gallery
        # thumbnails + group-averaged panels), then redraw.  The cached ReportData
        # (scalars/stats) survives a pure style/theme/clip change — only the
        # mobile-D threshold feeds a scalar (mob/immob), so ONLY it bumps _data_rev.
        self._engfig_rev += 1
        if key == "analysis/mobile_d":
            self._data_rev += 1
        self._engfig_cache.clear()
        self._panel_rev += 1
        self._panel_thumb_cache.clear()
        self._group_cache.clear()
        self.panelRevChanged.emit()
        # Debounced so a "Restore defaults" burst (many keys at once) coalesces
        # into a single re-render.
        self._set_busy(True)
        self._debounce.start()
        if self._view == "panels":
            self._render_panel_async()

    # ════════════════ QML PROPERTIES ════════════════════════════════════
    @Property("QVariantList", notify=conditionsChanged)
    def conditions(self):
        out = []
        for c in self._conditions:
            tp = next((t for t in self._timepoints if t["name"] == c.phase), None)
            active = c.active()
            out.append({
                "id": c.id, "name": c.name, "colorHex": c.color, "phase": c.phase,
                "phaseColor": tp["color"] if tp else "#5b636e",
                "activeFolders": len(active), "totalFolders": len(c.folders),
                "ready": len(active) >= 2,
                "folders": [{
                    "id": f.run.id if f.run else os.path.basename(f.path),
                    "n": ("analysing…" if f.analysing else
                          "…" if f.loading else
                          (f.run.n_label if f.run else "—")),
                    "qc": "loading" if (f.loading or f.analysing) else (
                        f.run.qc_level if f.run else "error"),
                    "loading": f.loading or f.analysing,
                    "analysing": f.analysing,
                    "excluded": f.excluded, "path": f.path,
                } for f in c.folders],
            })
        return out

    @Property("QVariantList", notify=timepointsChanged)
    def timepoints(self):
        return [{"name": t["name"], "colorHex": t["color"]} for t in self._timepoints]

    @Property(int, notify=conditionsChanged)
    def conditionCount(self):
        return len(self._conditions)

    @Property(bool, notify=loadingChanged)
    def loadingFolders(self):
        """True while one or more dropped folders are still loading."""
        return self._loading_n > 0

    @Property(int, constant=True)
    def maxConditions(self):
        return wd.MAX_CONDITIONS

    # ── experimental-design summary + stats recommendations (the "design &
    #    recommended settings" panel ported from the Widgets UI) ──────────────
    def _design_info(self):
        """(groups-by-name, sorted timepoints, factorial?) for the design panel.
        Groups are the GROUP factor (distinct condition names) with replicates
        pooled across time points; factorial = ≥2 names AND ≥2 time points."""
        shown = self._shown()
        phases = sorted({c.phase for c in shown if c.phase not in ("—", "")})
        by_name: dict = {}
        for c in shown:
            e = by_name.setdefault(c.name, {"name": c.name, "n": 0, "color": c.color})
            e["n"] += len(c.active())
        groups = list(by_name.values())
        factorial = len(by_name) >= 2 and len(phases) >= 2
        return groups, phases, factorial

    @Property("QVariantMap", notify=resultsChanged)
    def designSummary(self):
        groups, phases, factorial = self._design_info()
        n = len(groups)
        if factorial:
            desc = (f"Paired design — {n} group(s) × {len(phases)} time points "
                    f"({', '.join(phases)}). Cells are matched across time points "
                    f"by folder name.")
        elif n >= 2:
            desc = f"Unpaired — {n} conditions compared independently."
        else:
            desc = "Add at least two conditions (with run folders) to compare."
        return {"groups": groups, "timepoints": phases, "factorial": factorial,
                "ready": n >= 2, "description": desc}

    @Property("QVariantList", notify=resultsChanged)
    def recommendations(self):
        groups, phases, factorial = self._design_info()
        n = len(groups)
        if n < 2:
            return []
        min_n = min(g["n"] for g in groups)
        n_metrics = len(wd.PANEL_METRIC)
        recs = []
        # 1. replicate adequacy → normality test / Auto default
        if min_n >= 12:
            recs.append({"tone": "ok", "text":
                f"{min_n}+ replicates per group — enough for the normality test to "
                f"choose sensibly. Auto is a good default."})
        elif min_n >= 3:
            recs.append({"tone": "info", "text":
                f"{min_n} replicates in the smallest group — Auto picks a sensible "
                f"test; more would let the normality check decide."})
        else:
            recs.append({"tone": "warn", "text":
                f"Only {min_n} replicate(s) in the smallest group — results are "
                f"exploratory; a non-parametric test is safer."})
        # 2. design → test choice
        if factorial:
            recs.append({"tone": "info", "text":
                "Paired design detected (time points set) → a two-way mixed ANOVA "
                "(Greenhouse–Geisser) tests the group × time interaction."})
        elif n > 2:
            recs.append({"tone": "info", "text":
                f"Comparing {n} groups → Kruskal–Wallis with post-hoc pairwise tests."})
        else:
            recs.append({"tone": "info", "text":
                "Two groups → Mann–Whitney U (a t-test if both look normal under Auto)."})
        # 3. multiple metrics → family-wise correction
        if n_metrics > 1:
            recs.append({"tone": "info", "text":
                f"You're comparing {n_metrics} metrics, so family-wise correction "
                f"across metrics is recommended to keep false positives in check."})
        return recs

    @Property("QStringList", notify=conditionsChanged)
    def groupLabels(self):
        """Current condition names — populates the Dunnett control-group picker."""
        return [c.name for c in self._conditions]

    # ── report-figure panel picker ──────────────────────────────────────
    @Property("QVariantList", notify=cfgChanged)
    def comparePanels(self):
        return [{"key": k, "label": lbl, "on": k in self._panels}
                for k, lbl in wd.COMPARE_PANELS]

    @Property("QVariantList", notify=cfgChanged)
    def comparePanelPresets(self):
        """Preset buttons + whether the current selection exactly matches each —
        named curated combos first, then the All/None utility actions."""
        out = [{"name": n, "active": self._panels == set(keys), "util": False}
               for n, keys in wd.COMPARE_PANEL_PRESETS.items()]
        out.append({"name": "All", "util": True,
                    "active": self._panels == {k for k, _ in wd.COMPARE_PANELS}})
        out.append({"name": "None", "util": True, "active": not self._panels})
        return out

    @Slot(str, bool)
    def togglePanel(self, key, on):
        (self._panels.add if on else self._panels.discard)(key)
        self._persist()
        self.cfgChanged.emit()

    @Slot(str)
    def setPanelPreset(self, name):
        if name == "All":
            self._panels = {k for k, _ in wd.COMPARE_PANELS}
        elif name == "None":
            self._panels = set()
        elif name in wd.COMPARE_PANEL_PRESETS:
            self._panels = set(wd.COMPARE_PANEL_PRESETS[name])
        self._persist()
        self.cfgChanged.emit()

    @Property(int, notify=resultsChanged)
    def readyCount(self):
        return len(self._shown())

    @Property(bool, notify=resultsChanged)
    def enough(self):
        return len(self._cg) >= 2

    @Property(bool, notify=resultsChanged)
    def paired(self):
        return self._paired

    @Property(bool, notify=conditionsChanged)
    def hasTimepointsSet(self):
        return any(c.phase not in ("—", "") for c in self._conditions)

    @Property(str, notify=metricChanged)
    def metric(self):
        return self._metric

    @Property(bool, notify=metricChanged)
    def hasStats(self):
        # The selected panel has a scalar metric → show the stats/verdict cards.
        return self._has_metric()

    @Property("QVariantList", constant=True)
    def metrics(self):
        # The scroller is the comparison-figure panels (not scalar metrics).
        return [{"id": k, "label": lbl, "approx": False}
                for k, lbl, _m in wd.COMPARE_PANEL_TABS]

    @Property(str, notify=viewChanged)
    def view(self):
        return self._view

    @Property("QVariantMap", notify=cfgChanged)
    def cfg(self):
        return dict(self._cfg)

    @Property(bool, notify=busyChanged)
    def busy(self):
        return self._busy

    @Property("QVariantList", notify=resultsChanged)
    def headline(self):
        return self._headline

    @Property("QVariantList", notify=resultsChanged)
    def statsRows(self):
        return self._stats_rows

    @Property("QVariantList", notify=resultsChanged)
    def significanceRows(self):
        return self._sig_rows

    @Property("QVariantMap", notify=resultsChanged)
    def metricVerdict(self):
        """Plain-language verdict for the selected metric: {severity, html}."""
        return self._verdict

    @Property("QVariantList", constant=True)
    def statsGlossary(self):
        """Statistics term → definition pairs (the Preferences Glossary tab)."""
        try:
            from firefly.analysis.fa_stats_config import STATS_GLOSSARY
            return [{"term": k, "definition": v} for k, v in STATS_GLOSSARY.items()]
        except Exception:
            return []

    @Property("QVariantList", constant=True)
    def analysisGlossary(self):
        """Analysis/processing term → definition pairs (pixel size, search range …)."""
        try:
            from firefly.analysis.fa_stats_config import ANALYSIS_GLOSSARY
            return [{"term": k, "definition": v} for k, v in ANALYSIS_GLOSSARY.items()]
        except Exception:
            return []

    @Property("QVariantList", notify=resultsChanged)
    def twowayRows(self):
        return self._twoway

    @Property(str, notify=resultsChanged)
    def twowayNote(self):
        return self._twoway_note

    @Property(bool, notify=resultsChanged)
    def hasTwoway(self):
        return bool(self._twoway) or bool(self._twoway_note)

    @Property(str, notify=resultsChanged)
    def methods(self):
        return self._methods

    @Property("QVariantList", notify=resultsChanged)
    def legend(self):
        return self._legend

    @Property(str, notify=resultsChanged)
    def figureTitle(self):
        m = self._metric_obj()
        return ("Paired comparison — across timepoints" if self._paired
                else "Live comparison figure")

    @Property(str, notify=resultsChanged)
    def metricLabel(self):
        return wd.PANEL_LABEL.get(self._metric, self._metric_obj().label)

    @Property(str, notify=resultsChanged)
    def caption(self):
        label = wd.PANEL_LABEL.get(self._metric, self._metric)
        if not self._has_metric():
            return f"{label} · exact export panel"
        m = self._metric_obj()
        if self._paired:
            return f"paired · {label}"
        return f"{label} · exact export panel · {self._cfg['err']}"

    @Property(int, notify=figureChanged)
    def figureToken(self):
        return self._fig_token

    @Property(bool, notify=figureChanged)
    def hasFigure(self):
        return self._fig_image is not None

    @Property(str, notify=figureChanged)
    def figureBg(self):
        """The engine's figure background colour for the current figures/theme,
        so the QML mat matches the rendered panel seamlessly — the tight panel's
        facecolor edges blend into the mat and the rounded frame reads cleanly."""
        s = self._settings
        theme = (s.getStr("figures/theme", "Dark") if s else "Dark")
        return {"Dark": "#0d1117", "Light": "#ffffff",
                "Publication": "#ffffff", "OLED": "#000000"}.get(theme, "#0d1117")

    # paired SVG data: [{subject, color, points:[{x_label, y}]}] for QML to draw
    @Property("QVariantList", notify=resultsChanged)
    def pairedSeries(self):
        if not self._paired:
            return []
        metric = self._metric_obj()
        order = [g["label"] for g in self._cg]
        shown = self._shown()
        subjects = []
        seen = set()
        for c in shown:
            if c.name not in seen:
                seen.add(c.name)
                subjects.append((c.name, c.color))
        series = []
        for name, color in subjects:
            pts = []
            for ph in order:
                c = next((x for x in shown if x.name == name and x.phase == ph), None)
                if c is not None:
                    v = self._values(c, metric)
                    if len(v):
                        pts.append({"x": ph, "y": float(np.mean(v))})
            if pts:
                series.append({"subject": name, "color": color, "points": pts})
        return series

    @Property("QVariantList", notify=resultsChanged)
    def pairedAxis(self):
        return [g["label"] for g in self._cg] if self._paired else []

    @Property("QVariantList", notify=presetsChanged)
    def presets(self):
        return [{"name": p["name"]} for p in self._presets]

    # ════════════════ QML SLOTS ═════════════════════════════════════════
    @Slot()
    def addCondition(self):
        if len(self._conditions) >= wd.MAX_CONDITIONS:
            return
        i = len(self._conditions)
        self._conditions.append(
            _Condition(self._new_cid(), f"Condition {i + 1}",
                       wd.GROUP_COLORS[i % len(wd.GROUP_COLORS)]))
        self._changed(conditions=True)

    @Slot(int)
    def removeCondition(self, cid):
        if len(self._conditions) <= 2:
            return
        self._conditions = [c for c in self._conditions if c.id != cid]
        self._changed(conditions=True)

    def _cond(self, cid):
        return next((c for c in self._conditions if c.id == cid), None)

    @Slot(int, str)
    def setConditionName(self, cid, name):
        c = self._cond(cid)
        if c is not None:
            c.name = name
            self._changed(conditions=True)

    @Slot(int, str)
    def setConditionPhase(self, cid, phase):
        c = self._cond(cid)
        if c is not None:
            c.phase = phase
            self._changed(conditions=True)

    @staticmethod
    def _to_path(u):
        """Resolve a dropped/dialog value (QUrl, file:// string, or plain path)
        to a local filesystem path.  drop.urls hands us QUrl objects, which the
        old `isinstance(u, str)` check silently dropped."""
        if isinstance(u, QUrl):
            return u.toLocalFile()
        s = str(u)
        return QUrl(s).toLocalFile() if s.startswith("file:") else s

    @Slot(int, list)
    def addFolders(self, cid, urls):
        c = self._cond(cid)
        if c is None:
            return
        staged, to_analyse, flagged = self._stage_paths(
            c, [self._to_path(u) for u in urls])
        if staged or to_analyse:
            # Show the loading / analysing chips immediately, then do the work
            # off the GUI thread so the tab never freezes and more can be
            # dropped while these run.
            self.conditionsChanged.emit()
            self._loading_n += len(staged) + len(to_analyse)
            self.loadingChanged.emit()
        if staged:
            # Run FOLDERS: just read the (potentially large) sidecars.
            def _work(folders=staged):
                for folder in folders:
                    try:
                        run = wd.load_run(folder.path)
                    except Exception:
                        run = None
                    if run is not None:
                        try:
                            run.diff()      # warm the per-track cache off-thread
                        except Exception:
                            pass
                    self._load_q.put((folder, run))
                self._foldersLoaded.emit()
            threading.Thread(target=_work, daemon=True,
                             name="FIREFLY-CondLoad").start()
        if to_analyse:
            # External localisation FILES: analyse each into a run folder (using
            # the current sidebar settings so D/α match the FIREFLY replicates)
            # then load it as an ordinary replicate.  One thread keeps the
            # analyses serial so N dropped files don't oversubscribe the CPU.
            cache_root = self._external_cache_root()
            def _analyse(folders=to_analyse, cache_root=cache_root):
                for folder in folders:
                    run = None
                    try:
                        run_dir = wd.analyse_external_file(
                            folder.path, self._settings, cache_root=cache_root)
                        if run_dir:
                            run = wd.load_run(run_dir)
                            if run is not None:
                                folder.path = run_dir      # chip now IS the run
                                try:
                                    run.diff()
                                except Exception:
                                    pass
                    except Exception:
                        run = None
                    self._load_q.put((folder, run))
                self._foldersLoaded.emit()
            threading.Thread(target=_analyse, daemon=True,
                             name="FIREFLY-ExtAnalyse").start()
        if flagged and not (staged or to_analyse):
            # only unrecognised (flagged) paths were added — no work needed
            self._changed(conditions=True)

    def _external_cache_root(self) -> str:
        """Per-user cache dir for analyses of dropped localisation files, so we
        never write into the user's source folder and can reuse a prior run."""
        base = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.CacheLocation) or os.path.join(
            os.path.expanduser("~"), ".firefly_cache")
        d = os.path.join(base, "external_runs")
        os.makedirs(d, exist_ok=True)
        return d

    def _stage_paths(self, c, paths):
        """Resolve *paths* to placeholder chips using only cheap probes — the
        heavy work happens off-thread.

        * a run folder (or every run child of a parent folder) → a *loading*
          chip (``staged``);
        * an external localisation FILE (palmTRACER / ThunderSTORM / … export)
          → an *analysing* chip (``to_analyse``) — it's analysed into a run;
        * anything else → a flagged (invalid) chip (``flagged``).

        Returns ``(staged, to_analyse, flagged)``.
        """
        staged, to_analyse, flagged = [], [], False
        for path in paths:
            if not path or any(f.path == path for f in c.folders):
                continue
            if wd.is_external_loc_file(path):           # a raw loc table → analyse
                f = _Folder(path, None, analysing=True)
                c.folders.append(f); to_analyse.append(f)
                continue
            if not os.path.isdir(path):
                continue
            if wd.is_run_folder(path):                  # a run folder itself
                f = _Folder(path, None, loading=True)
                c.folders.append(f); staged.append(f)
                continue
            # not a run folder — accept a *parent* folder of runs by scanning one
            # level down for child run folders.
            found = False
            try:
                children = [os.path.join(path, x) for x in sorted(os.listdir(path))]
            except OSError:
                children = []
            for ch in children:
                if (os.path.isdir(ch) and wd.is_run_folder(ch)
                        and not any(f.path == ch for f in c.folders)):
                    f = _Folder(ch, None, loading=True)
                    c.folders.append(f); staged.append(f); found = True
            if not found:                               # show it anyway (flagged)
                c.folders.append(_Folder(path, None)); flagged = True
        return staged, to_analyse, flagged

    @Slot()
    def _on_folders_loaded(self):
        """GUI-thread: drain loaded runs into their placeholder chips."""
        got = False
        while True:
            try:
                folder, run = self._load_q.get_nowait()
            except queue.Empty:
                break
            folder.run = run
            folder.loading = False
            folder.analysing = False
            folder.excluded = bool(run is not None and run.qc_level == "error")
            self._loading_n = max(0, self._loading_n - 1)
            got = True
        if got:
            self.loadingChanged.emit()
            self._changed(conditions=True)     # recompute — caches are warm now

    @Slot(int)
    def browseAddFolder(self, cid):
        from PySide6.QtWidgets import QFileDialog
        d = QFileDialog.getExistingDirectory(None, "Add run folder")
        if d:
            self.addFolders(cid, [d])

    @Slot(int)
    def browseAddFiles(self, cid):
        """Pick one or more external localisation files (palmTRACER / ThunderSTORM
        / Picasso / TrackMate).  Each is analysed into a replicate on add."""
        from PySide6.QtWidgets import QFileDialog
        files, _ = QFileDialog.getOpenFileNames(
            None, "Add localisation files", "",
            "Localisations (*.csv *.txt *.tsv);;All files (*)")
        if files:
            self.addFolders(cid, files)

    @Slot(int, str)
    def toggleFolder(self, cid, fid):
        c = self._cond(cid)
        if c is None:
            return
        for f in c.folders:
            if (f.run.id if f.run else os.path.basename(f.path)) == fid:
                f.excluded = not f.excluded
                self._changed(conditions=True)
                return

    @Slot(int, str)
    def removeFolder(self, cid, fid):
        c = self._cond(cid)
        if c is None:
            return
        c.folders = [f for f in c.folders
                     if (f.run.id if f.run else os.path.basename(f.path)) != fid]
        self._changed(conditions=True)

    @Slot(str)
    def addTimepoint(self, name):
        name = name.strip()
        if not name or any(t["name"].lower() == name.lower() for t in self._timepoints):
            return
        self._timepoints.append(
            {"name": name, "color": _PHASE_COLORS[len(self._timepoints) % len(_PHASE_COLORS)]})
        self._changed(timepoints=True)

    @Slot(str)
    def removeTimepoint(self, name):
        self._timepoints = [t for t in self._timepoints if t["name"] != name]
        for c in self._conditions:
            if c.phase == name:
                c.phase = "—"
        self._changed(timepoints=True)
        self.conditionsChanged.emit()

    @Slot(str)
    def setMetric(self, mid):
        # `mid` is a comparison-panel key (the scroller IS the export panels).
        if mid in wd.PANEL_KEYS and mid != self._metric:
            self._metric = mid
            self.metricChanged.emit()
            self._persist()
            self._recompute()

    @Slot(str)
    def setView(self, v):
        if v in ("comparison", "panels") and v != self._view:
            self._view = v
            self.viewChanged.emit()
            if v == "panels":
                self._render_panel_async()
                self._warm_group_cache()       # pre-render the group's panels

    @Slot(str, "QVariant")
    def setCfg(self, key, value):
        if key not in self._cfg:
            return
        # coerce JS bool/number sensibly
        cur = self._cfg[key]
        if isinstance(cur, bool):
            value = bool(value)
        self._cfg[key] = value
        self.cfgChanged.emit()
        self._engfig_rev += 1             # stats settings change the panel → invalidate
        # A stats-config change (test / correction / alpha / post-hoc …) alters the
        # computed statistics AND the two-way ANOVA baked into the ReportData, so it
        # must invalidate it.  The render-only / live-view keys don't touch the
        # engine compute — skip them so a bar-type toggle, plot-kind switch, log-axis
        # flip or filename keystroke stays a cheap redraw.
        if key not in ("err", "groupBy", "outputStem", "plot", "logX"):
            self._data_rev += 1
        self._engfig_cache.clear()
        self._recompute()

    @Slot()
    def applyRecommended(self):
        shown = self._shown()
        multi = any(len(c.active()) > 2 for c in shown)
        rec = wd.recommend_config(len(self._cg), self._paired, multi)
        self._cfg.update(rec["cfg"])
        self.cfgChanged.emit()
        self._recompute()
        self.toast.emit("Recommended settings applied")

    @Property(str, notify=resultsChanged)
    def recommendWhy(self):
        shown = self._shown()
        multi = any(len(c.active()) > 2 for c in shown)
        return wd.recommend_config(max(2, len(self._cg)), self._paired, multi)["why"]

    @Slot()
    def savePreset(self):
        self._presets.append({
            "name": f"Preset {len(self._presets) + 1}",
            "cfg": dict(self._cfg), "metric": self._metric})
        self.presetsChanged.emit()
        self._persist()

    @Slot(str)
    def loadPreset(self, name):
        p = next((x for x in self._presets if x["name"] == name), None)
        if p:
            self._cfg.update(p["cfg"])
            m = p.get("metric", self._metric)
            self._metric = m if m in wd.PANEL_KEYS else self._metric
            self.cfgChanged.emit()
            self.metricChanged.emit()
            self._recompute()

    @Slot(str)
    def deletePreset(self, name):
        n = len(self._presets)
        self._presets = [x for x in self._presets if x["name"] != name]
        if len(self._presets) != n:
            self.presetsChanged.emit()
            self._persist()
            self.toast.emit("Preset deleted")

    @Slot()
    def copyMethods(self):
        cb = QGuiApplication.clipboard()
        if cb is not None:
            cb.setText(self._methods, QClipboard.Mode.Clipboard)
            self.toast.emit("Methods sentence copied")

    @Slot()
    def openOutputFolder(self):
        for c in self._shown():
            for f in c.active():
                QDesktopServices.openUrl(QUrl.fromLocalFile(f.path))
                return

    def _export_dir(self):
        """Where exports land: the first active folder's parent, else Desktop."""
        for c in self._shown():
            for f in c.active():
                return os.path.dirname(f.path) or os.path.dirname(f.path)
        return QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DesktopLocation) or os.getcwd()

    @Slot()
    def exportFigure(self):
        view = "panel" if self._view == "panels" else "figure"
        img = self._panel_image if self._view == "panels" else self._fig_image
        if img is None or img.isNull():
            self.toast.emit("Nothing to export yet")
            return
        metric = (self._metric_obj().label if self._view == "comparison"
                  else wd.PANELS[self._panel_sel]["name"])
        safe = "".join(ch if ch.isalnum() else "_" for ch in metric)
        path = os.path.join(self._export_dir(), f"firefly_{view}_{safe}.png")
        if img.save(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            self.toast.emit(f"Saved {os.path.basename(path)}")
        else:
            self.toast.emit("Export failed")

    @Slot()
    def exportStats(self):
        if not self._stats_rows:
            self.toast.emit("No statistics to export yet")
            return
        import csv
        from firefly.analysis.fa_io import atomic_write   # stdlib-only; cheap
        metric = self._metric_obj()
        path = os.path.join(self._export_dir(), "firefly_comparison_stats.csv")
        try:
            with atomic_write(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["# metric", metric.label, metric.unit])
                w.writerow(["# test", self._cfg["test"], "correction",
                            self._cfg["correction"], "alpha", self._cfg["alpha"]])
                w.writerow(["condition", "n_tracks", f"{metric.label} ({metric.unit})", "± err"])
                for r in self._stats_rows:
                    w.writerow([r["label"], r["n"], r["value"], r["err"].replace("±", "")])
                w.writerow([])
                w.writerow(["pair", "Cliff's delta", "magnitude", "p", "significant"])
                for r in self._sig_rows:
                    w.writerow([r["label"], r["delta"].replace("δ ", ""), r["mag"],
                                r["p"].replace("p = ", ""), "yes" if r["sig"] else "no"])
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            self.toast.emit(f"Saved {os.path.basename(path)}")
        except Exception:
            self.toast.emit("Stats export failed")


def g_run_n(g) -> int:
    """Total tracks contributing to a comparison group (for the n labels)."""
    return int(g.get("ntracks", 0))
