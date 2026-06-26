"""VisualiseController — QML bridge for the interactive viewer (Phase 4).

Owns a lazily-built bespoke FireflyViewer (the native island embedded under the
QML chrome) and is the single source of truth the QML layer rail / transport /
HUD / inspector bind to.  It drives the viewer by calling its EXISTING public API
(set_stack / set_tracks_from_df / set_class_visible / background_mode / current_
frame / pick), so behaviour matches the Widgets Visualise tab exactly.

This first slice covers the core viewer: load-run / tracks / stack, the per-
motion-class track layer model + visibility, background-layer selection, playback
transport (frame / play / fps / tail / head / width), motion-colour palette, and
the click→inspector bridge.  Clusters, super-resolution rendering, and the track
explorer hang off the same controller and land in a follow-up slice.

The analysis core is untouched; the viewer's headless transport façade (the
tail/head/track_width/fps/playing/background_mode properties) was added additively
to viewer.py with no behaviour change.
"""
from __future__ import annotations

import os
import threading

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Property, QObject, QTimer, Signal, Slot

from firefly.analysis.fa_constants import motion_class_colors
from firefly.ui.ui_helpers import _MOTION_ORDER

# Sidebar "Motion colours" label → figure-theme palette name (mirrors
# VisualiseMixin._WS_MOTION_COLOUR_THEMES).
_MOTION_COLOUR_THEMES = {"Default": "Dark", "Colour-blind safe": "Publication"}

# Multi-run track comparison: each loaded run's particle ids are offset by
# _RUN_OFFSET * run_index so picking can decode which run a track belongs to
# (assumes < 1e6 particles per run). Per-run "colour by file" palette:
_RUN_OFFSET = 1_000_000
_SEP = "\x1f"                       # compound class separator: f"{run_idx}{_SEP}{motion}"
_RUN_COLOURS = ["#58a6ff", "#f6a623", "#4fe0a0", "#27c0e8", "#e05252",
                "#a371f7", "#7ed321", "#f78166", "#d2a8ff", "#56d364"]


class VisualiseController(QObject):
    # property-notify signals
    dataChanged = Signal()
    layersChanged = Signal()
    currentFrameChanged = Signal()
    playingChanged = Signal()
    fpsChanged = Signal()
    tailChanged = Signal()
    headChanged = Signal()
    trackWidthChanged = Signal()
    minLenChanged = Signal()
    motionColourModeChanged = Signal()
    colourByChanged = Signal()
    backgroundChanged = Signal()
    inspectorChanged = Signal()
    nFramesChanged = Signal()
    clusterChanged = Signal()
    srChanged = Signal()
    explorerChanged = Signal()
    explorerFiltersChanged = Signal()
    # event signals
    statusMessage = Signal(str)
    warn = Signal(str, str)
    trackPicked = Signal(int)
    clusterPicked = Signal(int)

    def __init__(self, settings=None, importc=None, parent=None):
        super().__init__(parent)
        self._s = settings
        self._import = importc
        self._viewer = None

        self._tracks_df = None             # PRIMARY run (drives clusters/super-res/explorer)
        self._diff_df = None
        self._runs: list = []              # [{name, df, diff, color, offset}] — track overlays
        self._colour_by = (settings.get_str("visualise/track_colour_by", "auto")
                           if settings else "auto")
        if self._colour_by not in ("auto", "motion", "file"):
            self._colour_by = "auto"
        self._motion_pids: dict = {}       # compound class → set(pids)
        self._class_visible: dict = {}     # compound class → bool (persists across loads)
        self._min_len = 1
        self._motion_mode = (settings.get_str("visualise/motion_colours", "Default")
                             if settings else "Default")
        # Pick up the motion-class palette when changed from Preferences (so the
        # colour-blind choice applies to an already-open viewer without restart).
        try:
            settings.changed.connect(self._on_settings_changed)
        except Exception:
            pass
        self._layers: list = []
        self._inspector: dict = {"mode": "none"}
        self._inspector_visible = False
        self._has_run = False
        self._hud_tracks = 0

        # ── clusters ─────────────────────────────────────────────────────
        self._cl_xy_um = None             # DISPLAYED loc coords (may be subsampled)
        self._cl_xy_um_full = None        # full loc set — the input to every recluster
        self._cl_labels = None
        self._cl_motion = None
        self._cl_motion_tried_derive = False   # re-join cluster locs↔track motion once
        self._motion_src = []             # [(traj_df, diff_df)] — data-only motion for
                                          # a standalone cluster map (no track overlay)
        self._cl_xy_px = None
        self._cl_px_um = 1.0
        self._cl_stats_df = None
        self._cl_extras_dir = None
        self._cl_stem = None
        self._cl_present = False
        self._cl_visible = True          # cluster overlay shown (toggle in LAYERS)
        self._cl_eps_nm = 50
        self._cl_min_samples = 8
        self._cl_point_size = (int(settings.get_float("visualise/cluster_point_size", 3))
                               if settings else 3)
        self._cl_color_mode = "Motion"
        self._cl_count = 0
        self._cl_status = ""
        # ── super-resolution ─────────────────────────────────────────────
        self._sr_img = None
        self._field_px = None            # (H, W) of the loaded camera frame, so the
                                         # super-res canvas matches the full field
                                         # (not just the localisations' bounding box)
        self._sr_nm = 20
        self._sr_blur = 20
        self._sr_status = ""
        self._sr_rendering = False       # super-res render runs off-thread
        self._sr_result = None           # (status, img, x, y, sr_nm) written off-thread
        self._sr_poll = QTimer(self)     # drains the result on the GUI thread
        self._sr_poll.setInterval(30)
        self._sr_poll.timeout.connect(self._drain_superres)
        # ── track explorer ───────────────────────────────────────────────
        self._exp_df = None
        self._exp_filtered = None
        self._exp_rows: list = []
        self._exp_count = ""
        self._exp_d_min = 0.0
        self._exp_d_max = 10.0
        self._exp_a_min = 0.0
        self._exp_a_max = 2.0
        self._exp_min_len = 1
        self._exp_motion = {m: True for m in ("Immobile", "Confined",
                                              "Brownian", "Directed")}
        # debounce timers (slider drags); direct slot calls run synchronously
        self._recluster_timer = QTimer(self)
        self._recluster_timer.setSingleShot(True)
        self._recluster_timer.setInterval(300)
        self._recluster_timer.timeout.connect(self.recluster)
        self._explorer_timer = QTimer(self)
        self._explorer_timer.setSingleShot(True)
        self._explorer_timer.setInterval(250)
        self._explorer_timer.timeout.connect(self.refreshExplorer)

    # ── viewer lifecycle ─────────────────────────────────────────────────
    def ensureViewer(self):
        """Build the FireflyViewer island on first use and wire its signals."""
        if self._viewer is not None:
            return self._viewer
        from firefly.ui.viewer import FireflyViewer
        v = FireflyViewer()
        v.trackClicked.connect(self._on_track_clicked)
        v.clusterClicked.connect(self._on_cluster_clicked)
        v.frameChanged.connect(self._on_frame_changed)
        try:
            v._play_btn.toggled.connect(lambda *_: self.playingChanged.emit())
        except Exception:
            pass
        self._viewer = v
        return v

    def viewerWidget(self):
        """The native FireflyViewer QWidget (handed to EmbedController)."""
        return self.ensureViewer()

    # ── motion palette ───────────────────────────────────────────────────
    def _palette(self) -> dict:
        return motion_class_colors(_MOTION_COLOUR_THEMES.get(self._motion_mode, "Dark"))

    # ── data loading ─────────────────────────────────────────────────────
    @Slot()
    def loadRun(self):
        start = ""
        if self._import is not None:
            start = self._import.outDir or ""
        path = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Open a FIREFLY analysis run", start or os.path.expanduser("~"))
        if path:
            self.loadRunFolder(path)

    @Slot(str, result=bool)
    def loadRunFolder(self, run_dir: str) -> bool:
        """Load a complete run: stack (if recorded) + trajectories + diffusion
        summary from ``firefly_extras/``.  Ports VisualiseMixin._ws_load_run_
        folder's headless resolution (descends a single-run parent folder)."""
        import json
        self._field_px = None            # reset; set again only if this run has a stack
        try:
            extras = os.path.join(run_dir, "firefly_extras")
            if not os.path.isdir(extras):
                children = [os.path.join(run_dir, n)
                            for n in sorted(os.listdir(run_dir))
                            if os.path.isdir(os.path.join(run_dir, n, "firefly_extras"))]
                if len(children) == 1:
                    return self.loadRunFolder(children[0])
                if len(children) > 1:
                    self.warn.emit("Multiple runs",
                                   f"{os.path.basename(run_dir)!r} contains "
                                   f"{len(children)} runs — open one directly.")
                    return False
                raise FileNotFoundError(
                    f"No firefly_extras/ in {os.path.basename(run_dir)}")
            stem = None
            stack_path = None
            params_files = [f for f in os.listdir(extras) if f.endswith("_params.json")]
            if params_files:
                with open(os.path.join(extras, params_files[0])) as fh:
                    params = json.load(fh)
                stack_path = params.get("input_file") or params.get("stem")
                stem = params_files[0][:-len("_params.json")]
                try:                               # real pixel size → sane super-res canvas
                    if params.get("pixel_size"):
                        self._cl_px_um = float(params["pixel_size"])
                except (TypeError, ValueError):
                    pass
            if not stem:
                tr = [f for f in os.listdir(extras) if f.endswith("_trajectories.csv")]
                if tr:
                    stem = tr[0][:-len("_trajectories.csv")]
            if not stem:
                raise FileNotFoundError("No params.json or trajectories.csv found")

            tracks_path = os.path.join(extras, f"{stem}_trajectories.csv")
            if not os.path.isfile(tracks_path):
                raise FileNotFoundError(f"Missing {os.path.basename(tracks_path)}")

            if stack_path and os.path.isfile(stack_path):
                self.loadStackPath(stack_path)
            diff = os.path.join(extras, f"{stem}_diffusion_summary.csv")
            self.loadTracksPath(tracks_path,
                                diff if os.path.isfile(diff) else None)
            return True
        except Exception as exc:
            self.warn.emit("Load failed",
                           f"Couldn't load run {os.path.basename(run_dir)}:\n{exc}")
            return False

    @Slot()
    def loadStack(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Load image stack", os.path.expanduser("~"),
            "Image stacks (*.czi *.tif *.tiff);;All files (*)")
        if path:
            self.loadStackPath(path)

    @Slot(str)
    def loadStackPath(self, path: str):
        try:
            from firefly.sptpalm_analysis import load_file
            self.statusMessage.emit(f"Loading {os.path.basename(path)}…")
            stack, _, _ = load_file(path, channel=0)
            self.ensureViewer().set_stack(stack)
            # remember the camera field size so a super-res render fills the same
            # extent as the Raw movie / Max projection backgrounds
            try:
                import numpy as _np
                s = _np.asarray(stack)
                self._field_px = (int(s.shape[-2]), int(s.shape[-1]))
            except Exception:
                self._field_px = None
            self._after_background_change()
            self.statusMessage.emit(f"Loaded {len(stack):,} frames")
        except Exception as exc:
            self.warn.emit("Load failed", f"Couldn't load {os.path.basename(path)}:\n{exc}")

    @Slot()
    def loadTracks(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None, "Load tracks CSV", os.path.expanduser("~"),
            "Tracks CSV (*trajectories.csv);;All CSVs (*.csv)")
        if path:
            self.loadTracksPath(path, None)

    @Slot(str, "QVariant")
    def loadTracksPath(self, csv_path: str, diff_csv_path=None):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            need = {"particle", "frame", "x", "y"}
            missing = need - set(df.columns)
            if missing:
                raise ValueError(f"CSV missing columns: {sorted(missing)}")
            diff_df = None
            if diff_csv_path and os.path.isfile(diff_csv_path):
                try:    diff_df = pd.read_csv(diff_csv_path)
                except Exception: diff_df = None
            elif not diff_csv_path:
                guess = csv_path.replace("_trajectories.csv", "_diffusion_summary.csv")
                if guess != csv_path and os.path.isfile(guess):
                    try:    diff_df = pd.read_csv(guess)
                    except Exception: diff_df = None
            # Register as an overlay run (multi-run track comparison). Each run's
            # particle ids are offset so picking can decode which run a track is
            # from; the first run is the PRIMARY (drives clusters/super-res/explorer).
            name = os.path.splitext(os.path.basename(csv_path))[0].replace("_trajectories", "")
            ri = len(self._runs)
            off = ri * _RUN_OFFSET
            df = df.copy(); df["particle"] = df["particle"].astype("int64") + off
            if diff_df is not None and "particle" in diff_df.columns:
                diff_df = diff_df.copy()
                diff_df["particle"] = diff_df["particle"].astype("int64") + off
            self._runs.append({"name": name, "df": df, "diff": diff_df,
                               "color": _RUN_COLOURS[ri % len(_RUN_COLOURS)], "offset": off})
            self._cl_motion_tried_derive = False   # new tracks → re-derive cluster motion
            self._tracks_df = self._runs[0]["df"]      # primary
            self._diff_df = self._runs[0]["diff"]
            self._render_tracks()
            self._build_explorer_data()
            if self._cl_present:          # clusters loaded before tracks → recolour
                self._render_cluster_layer()   # with the now-available motion data
            self._has_run = True
            self.dataChanged.emit()
            extra = f" · {len(self._runs)} runs" if len(self._runs) > 1 else ""
            self.statusMessage.emit(
                f"Loaded {df['particle'].nunique():,} tracks "
                f"({len(df):,} points){extra} — click a track to inspect.")
        except Exception as exc:
            self.warn.emit("Load failed",
                           f"Couldn't load tracks from {os.path.basename(csv_path)}:\n{exc}")

    def _effective_colour_by(self) -> str:
        """'motion' or 'file' — resolves 'auto' (file when ≥2 runs)."""
        if self._colour_by in ("file", "motion"):
            return self._colour_by
        return "file" if len(self._runs) > 1 else "motion"

    def _class_key(self, ri, cls):
        """Viewer class key: plain motion class for a single run (back-compat),
        compound ``f"{run}{SEP}{motion}"`` once ≥2 runs are loaded."""
        return cls if len(self._runs) <= 1 else f"{ri}{_SEP}{cls}"

    def _render_tracks(self):
        """(Re)render all loaded runs' tracks into the viewer. Each track's
        'class' is the compound ``f"{run_idx}{SEP}{motion}"`` so the viewer's
        generic per-class colour + visibility doubles as per-(file, motion)."""
        if not self._runs:
            return
        import pandas as pd
        v = self.ensureViewer()
        mode = self._effective_colour_by()
        motion_pal = self._palette()
        class_map, colors, frames = {}, {}, []
        for ri, run in enumerate(self._runs):
            df, diff = run["df"], run["diff"]
            mmap = (dict(zip(diff["particle"], diff["motion"]))
                    if diff is not None and "motion" in diff.columns else {})
            for pid in df["particle"].unique():
                cls = mmap.get(pid, "Unknown")
                comp = self._class_key(ri, cls)
                class_map[int(pid)] = comp
                colors[comp] = run["color"] if mode == "file" else motion_pal.get(cls, "#aaaaaa")
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        pids_by_cls = v.set_tracks_from_df(combined, class_map, colors, min_len=int(self._min_len))
        self._motion_pids = {c: set(s) for c, s in pids_by_cls.items()}
        for cls in pids_by_cls:
            self._class_visible.setdefault(cls, True)
            try:    v.set_class_visible(cls, self._class_visible[cls])
            except Exception: pass
        self._hud_tracks = sum(len(s) for s in pids_by_cls.values())
        self._rebuild_layers()
        self.nFramesChanged.emit()
        self.currentFrameChanged.emit()

    # ── layer model (QML rail) ───────────────────────────────────────────
    def _rebuild_layers(self):
        pal = self._palette()
        mode = self._effective_colour_by()
        multi = len(self._runs) > 1
        layers = []
        for ri, run in enumerate(self._runs):
            for cls in _MOTION_ORDER:
                comp = self._class_key(ri, cls)
                if comp in self._motion_pids:
                    layers.append({
                        "id": f"tracks:{comp}", "kind": "tracks", "name": cls,
                        "file": run["name"] if multi else "", "runColor": run["color"],
                        "present": True, "visible": bool(self._class_visible.get(comp, True)),
                        "opacity": 1.0,
                        "colorHex": run["color"] if mode == "file" else pal.get(cls, "#aaaaaa"),
                        "motionClass": cls, "count": len(self._motion_pids.get(comp, ())),
                    })
        # DBSCAN cluster overlay gets its own layer row (toggle to close it).
        if self._cl_present:
            layers.append({
                "id": "clusters:main", "kind": "clusters", "name": "Clusters",
                "file": "", "runColor": "",
                "present": True, "visible": bool(self._cl_visible), "opacity": 1.0,
                "colorHex": "#a371f7", "motionClass": "", "count": self._cl_count,
            })
        # Background images are NOT track layers — they're chosen via the
        # Background dropdown above the LAYERS list, so they're excluded here.
        self._layers = layers
        self.layersChanged.emit()
        self.backgroundChanged.emit()      # keep the LAYERS-panel bg dropdown in sync

    @Property("QVariantList", notify=layersChanged)
    def layers(self):
        return self._layers

    @Property("QVariantList", notify=layersChanged)
    def layerGroups(self):
        """Layers grouped by owning file/run (for multi-run track comparison).
        Single-run layers carry no ``file`` → one unnamed group; the QML hides the
        group header when ``file`` is empty."""
        order, by_file = [], {}
        for lyr in self._layers:
            f = lyr.get("file", "")
            if f not in by_file:
                by_file[f] = {"file": f, "colorHex": lyr.get("runColor", ""), "layers": []}
                order.append(f)
            by_file[f]["layers"].append(lyr)
        return [by_file[f] for f in order]

    @Slot(str, bool)
    def setLayerVisible(self, layer_id: str, on: bool):
        if layer_id.startswith("tracks:"):
            cls = layer_id.split(":", 1)[1]
            self._class_visible[cls] = bool(on)
            if self._viewer is not None:
                try:    self._viewer.set_class_visible(cls, bool(on))
                except Exception: pass
            self._rebuild_layers()
        elif layer_id.startswith("bg:"):
            name = layer_id.split(":", 1)[1]
            self.selectBackground(name if on else "Off")
        elif layer_id.startswith("clusters:"):
            self._cl_visible = bool(on)
            self._render_cluster_layer()      # renders if visible, clears if not
            self._rebuild_layers()

    @Slot(str, float)
    def setLayerOpacity(self, layer_id: str, value: float):
        # Per-layer opacity is a display nicety; tracks/background don't expose
        # per-item alpha yet, so record it in the model for the rail's bar.
        for lyr in self._layers:
            if lyr.get("id") == layer_id:
                lyr["opacity"] = max(0.0, min(1.0, float(value)))
                self.layersChanged.emit()
                break

    # ── background-layer selection ───────────────────────────────────────
    @Property("QStringList", notify=backgroundChanged)
    def backgroundOptions(self):
        return self._viewer.background_options() if self._viewer is not None else []

    @Property(str, notify=backgroundChanged)
    def backgroundMode(self):
        return self._viewer.background_mode if self._viewer is not None else "Off"

    @backgroundMode.setter
    def backgroundMode(self, mode: str):
        self.selectBackground(mode)

    @Slot(str)
    def selectBackground(self, mode: str):
        if self._viewer is not None:
            self._viewer.background_mode = mode
            self._after_background_change()

    def _after_background_change(self):
        self.backgroundChanged.emit()
        self._rebuild_layers()

    # ── transport ────────────────────────────────────────────────────────
    @Property(int, notify=nFramesChanged)
    def nFrames(self):
        return self._viewer.n_frames if self._viewer is not None else 0

    @Property(int, notify=currentFrameChanged)
    def currentFrame(self):
        return self._viewer.current_frame if self._viewer is not None else 0

    @currentFrame.setter
    def currentFrame(self, i):
        if self._viewer is not None:
            self._viewer.current_frame = int(i)

    @Property(str, notify=currentFrameChanged)
    def frameLabel(self):
        if self._viewer is None or self._viewer.n_frames <= 0:
            return "—"
        return f"frame {self._viewer.current_frame + 1} / {self._viewer.n_frames}"

    @Slot(int)
    def seek(self, i):
        self.currentFrame = i

    @Slot()
    def stepBack(self):
        self.currentFrame = 0

    @Slot()
    def stepForward(self):
        if self._viewer is not None:
            self.currentFrame = max(0, self._viewer.n_frames - 1)

    @Slot()
    def playPause(self):
        if self._viewer is not None:
            self._viewer.playing = not self._viewer.playing

    @Property(bool, notify=playingChanged)
    def playing(self):
        return bool(self._viewer.playing) if self._viewer is not None else False

    @Property(int, notify=fpsChanged)
    def fps(self):
        return self._viewer.fps if self._viewer is not None else 60

    @fps.setter
    def fps(self, v):
        if self._viewer is not None:
            self._viewer.fps = int(v)
            self.fpsChanged.emit()

    def _on_frame_changed(self, _i):
        self.currentFrameChanged.emit()

    # ── track-display knobs ──────────────────────────────────────────────
    @Property(int, notify=tailChanged)
    def tail(self):
        return self._viewer.tail if self._viewer is not None else 30

    @tail.setter
    def tail(self, v):
        if self._viewer is not None:
            self._viewer.tail = int(v); self.tailChanged.emit()

    @Property(int, notify=headChanged)
    def head(self):
        return self._viewer.head if self._viewer is not None else 0

    @head.setter
    def head(self, v):
        if self._viewer is not None:
            self._viewer.head = int(v); self.headChanged.emit()

    @Property(float, notify=trackWidthChanged)
    def trackWidth(self):
        return self._viewer.track_width if self._viewer is not None else 1.5

    @trackWidth.setter
    def trackWidth(self, v):
        if self._viewer is not None:
            self._viewer.track_width = float(v); self.trackWidthChanged.emit()

    @Property(int, notify=minLenChanged)
    def minLen(self):
        return int(self._min_len)

    @minLen.setter
    def minLen(self, v):
        v = max(1, int(v))
        if v != self._min_len:
            self._min_len = v
            self.minLenChanged.emit()
            if self._runs:
                self._render_tracks()

    @Property("QStringList", constant=True)
    def motionColourModes(self):
        return list(_MOTION_COLOUR_THEMES.keys())

    @Property(str, notify=motionColourModeChanged)
    def motionColourMode(self):
        return self._motion_mode

    def _on_settings_changed(self, key):
        if str(key) == "visualise/motion_colours" and self._s is not None:
            v = self._s.get_str("visualise/motion_colours", "Default")
            if v in _MOTION_COLOUR_THEMES and v != self._motion_mode:
                self.motionColourMode = v       # re-renders motion-coloured layers

    @motionColourMode.setter
    def motionColourMode(self, mode):
        if mode in _MOTION_COLOUR_THEMES and mode != self._motion_mode:
            self._motion_mode = mode
            if self._s:
                try:    self._s.set("visualise/motion_colours", mode)
                except Exception: pass
            self.motionColourModeChanged.emit()
            # Compound classes carry the palette, so re-render rather than recolour.
            if self._runs:
                self._render_tracks()
            else:
                self._rebuild_layers()

    # ── track colour mode (motion class vs by file) ──────────────────────
    @Property("QStringList", constant=True)
    def colourByModes(self):
        return ["Auto", "Motion", "File"]

    @Property(str, notify=colourByChanged)
    def colourBy(self):
        return self._colour_by.capitalize()

    @Property(str, notify=colourByChanged)
    def effectiveColourBy(self):
        return self._effective_colour_by().capitalize()

    @Slot(str)
    def setColourBy(self, mode):
        m = str(mode).lower()
        if m in ("auto", "motion", "file") and m != self._colour_by:
            self._colour_by = m
            if self._s:
                try:    self._s.set("visualise/track_colour_by", m)
                except Exception: pass
            self.colourByChanged.emit()
            if self._runs:
                self._render_tracks()

    @Slot(str, bool)
    def setRunVisible(self, file_name, on):
        """Toggle every motion-class layer of one run (the file-group header)."""
        for ri, run in enumerate(self._runs):
            if run["name"] == file_name:
                for cls in _MOTION_ORDER:
                    comp = self._class_key(ri, cls)
                    if comp in self._motion_pids:
                        self._class_visible[comp] = bool(on)
                        if self._viewer is not None:
                            try:    self._viewer.set_class_visible(comp, bool(on))
                            except Exception: pass
                self._rebuild_layers()
                break

    @Slot()
    def resetView(self):
        if self._viewer is not None:
            try:    self._viewer.reset_view()
            except Exception: pass

    @Property(bool, notify=dataChanged)
    def hasRun(self):
        return self._has_run

    @Property(bool, notify=dataChanged)
    def hasContent(self):
        """The viewer has something to show — tracks OR a standalone cluster map.
        Gates the floating viewer island + the 'open a run' placeholder, so a
        cluster map can be explored without first loading trajectories."""
        return self._has_run or self._cl_present

    @Property(int, notify=dataChanged)
    def hudTrackCount(self):
        return self._hud_tracks

    # ── pick → inspector ─────────────────────────────────────────────────
    def _on_track_clicked(self, pid):
        self.trackPicked.emit(int(pid))
        info = self._track_info(int(pid))
        if info is not None:
            self._inspector = info
            self._inspector_visible = True
            self.inspectorChanged.emit()

    def _on_cluster_clicked(self, cid):
        self.clusterPicked.emit(int(cid))
        self._inspector = self._cluster_info(int(cid))
        self._inspector_visible = True
        self.inspectorChanged.emit()

    def _cluster_info(self, cid: int) -> dict:
        if cid == -1:
            return {"mode": "cluster", "cluster_id": -1,
                    "note": "Noise point — not assigned to any cluster."}
        info = {"mode": "cluster", "cluster_id": cid}
        df = self._cl_stats_df
        if df is not None and "cluster_id" in getattr(df, "columns", []):
            try:
                row = df[df["cluster_id"] == cid]
                if len(row):
                    r = row.iloc[0]
                    for k in ("n_locs", "area_um2", "density_locs_per_um2",
                              "rg_um", "centroid_x_um", "centroid_y_um"):
                        if k in r.index:
                            info[k] = float(r[k])
            except Exception:
                pass
        try:
            if self._cl_motion is not None and self._cl_labels is not None:
                import numpy as np
                from collections import Counter
                motions = self._cl_motion[self._cl_labels == cid]
                if motions.size:
                    counts = Counter(motions.tolist())
                    total = sum(counts.values())
                    top, top_n = counts.most_common(1)[0]
                    info["note"] = (f"Dominant motion: {top} "
                                    f"({100.0 * top_n / max(1, total):.0f}%)")
        except Exception:
            pass
        return info

    def _track_info(self, pid: int):
        pid = int(pid)
        ri = pid // _RUN_OFFSET                     # decode which run this track is from
        run = self._runs[ri] if 0 <= ri < len(self._runs) else None
        df = run["df"] if run else self._tracks_df
        diff = run["diff"] if run else self._diff_df
        if df is None:
            return None
        rows = df[df["particle"] == pid]
        if rows.empty:
            return None
        info = {"mode": "track", "particle_id": pid % _RUN_OFFSET, "length": int(len(rows)),
                "start_frame": int(rows["frame"].min()),
                "end_frame": int(rows["frame"].max())}
        if run is not None and len(self._runs) > 1:
            info["file"] = run["name"]
        px = 1.0
        if self._import is not None:
            try:
                px = float(self._import.pixelSize) if self._import.overridePx else 1.0
            except Exception:
                px = 1.0
        try:
            import numpy as np
            xs, ys = rows["x"].values, rows["y"].values
            if len(xs) >= 2:
                net = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0])) * px
                seg = float(np.hypot(np.diff(xs), np.diff(ys)).sum()) * px
                info["net_displacement_um"] = net
                info["total_path_um"] = seg
                if seg > 0:
                    info["straightness"] = net / seg
        except Exception:
            pass
        if "mass" in rows.columns:
            try:    info["mean_mass"] = float(rows["mass"].mean())
            except Exception: pass
        if diff is not None and "particle" in diff.columns:
            d_row = diff[diff["particle"] == pid]
            if not d_row.empty:
                r = d_row.iloc[0]
                for k_src, k_dst in (("D", "d"), ("alpha", "alpha")):
                    if k_src in d_row.columns:
                        try:    info[k_dst] = float(r[k_src])
                        except Exception: pass
                if "motion" in d_row.columns:
                    info["motion"] = str(r["motion"])
                    info["motionColor"] = self._palette().get(str(r["motion"]), "#aaaaaa")
        return info

    @Slot(int)
    def selectTrack(self, pid: int):
        self._on_track_clicked(int(pid))
        # Centre the camera on the track's mid-vertex.
        df = self._tracks_df
        if df is not None and self._viewer is not None:
            rows = df[df["particle"] == int(pid)]
            if not rows.empty:
                mid = len(rows) // 2
                try:
                    self._viewer.center_on(float(rows["y"].values[mid]),
                                           float(rows["x"].values[mid]), span=40)
                except Exception:
                    pass

    @Property("QVariantMap", notify=inspectorChanged)
    def inspector(self):
        return self._inspector

    @Property(bool, notify=inspectorChanged)
    def inspectorVisible(self):
        return self._inspector_visible

    @Slot()
    def clearInspector(self):
        self._inspector = {"mode": "none"}
        self._inspector_visible = False
        self.inspectorChanged.emit()

    # ═════════════════════════════════════════════════════════════════════
    #  Clusters — ported from VisualiseMixin (load / recluster / suggest-eps)
    # ═════════════════════════════════════════════════════════════════════
    @Property(bool, notify=clusterChanged)
    def hasClusters(self):
        return self._cl_present

    @Property(int, notify=clusterChanged)
    def clusterCount(self):
        return self._cl_count

    @Property(str, notify=clusterChanged)
    def clusterStatus(self):
        return self._cl_status

    @Property(bool, notify=clusterChanged)
    def noClustersBanner(self):
        return self._cl_present and self._cl_count == 0

    @Property(bool, notify=clusterChanged)
    def clusterMotionAvailable(self):
        return self._cl_motion is not None

    @Property(int, notify=clusterChanged)
    def clusterEpsNm(self):
        return self._cl_eps_nm

    @clusterEpsNm.setter
    def clusterEpsNm(self, v):
        v = int(v)
        if v != self._cl_eps_nm:
            self._cl_eps_nm = v
            self.clusterChanged.emit()
            if self._cl_xy_um is not None:
                self._recluster_timer.start()

    @Property(int, notify=clusterChanged)
    def clusterMinSamples(self):
        return self._cl_min_samples

    @clusterMinSamples.setter
    def clusterMinSamples(self, v):
        v = max(2, int(v))
        if v != self._cl_min_samples:
            self._cl_min_samples = v
            self.clusterChanged.emit()
            if self._cl_xy_um is not None:
                self._recluster_timer.start()

    @Property(int, notify=clusterChanged)
    def clusterPointSize(self):
        return self._cl_point_size

    @clusterPointSize.setter
    def clusterPointSize(self, v):
        v = max(1, int(v))
        if v != self._cl_point_size:
            self._cl_point_size = v
            if self._s:
                try:    self._s.set("visualise/cluster_point_size", v)
                except Exception: pass
            self.clusterChanged.emit()
            if self._cl_present:
                self._render_cluster_layer()

    # "Motion" = colour each localisation by its own motion class.
    # "Cluster motion" = colour each whole cluster by its DOMINANT motion class.
    # "ID" = a distinct colour per cluster.
    _CLUSTER_COLOR_MODES = ("Motion", "Cluster motion", "ID")

    @Property("QStringList", constant=True)
    def clusterColorModes(self):
        return list(self._CLUSTER_COLOR_MODES)

    @Property(str, notify=clusterChanged)
    def clusterColorMode(self):
        return self._cl_color_mode

    @clusterColorMode.setter
    def clusterColorMode(self, mode):
        if mode in self._CLUSTER_COLOR_MODES and mode != self._cl_color_mode:
            self._cl_color_mode = mode
            self.clusterChanged.emit()
            if self._cl_present:
                self._render_cluster_layer()

    @Slot()
    def loadClusters(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Open a run with cluster labels",
            (self._import.outDir if self._import else "") or os.path.expanduser("~"))
        if path:
            self.loadClustersFolder(path)

    @Slot(str, result=bool)
    def loadClustersFolder(self, run_dir: str) -> bool:
        import json
        try:
            import numpy as np
            import pandas as pd
            extras = os.path.join(run_dir, "firefly_extras")
            if not os.path.isdir(extras):
                raise FileNotFoundError("No firefly_extras/ subfolder")
            lbl = [f for f in os.listdir(extras) if f.endswith("_cluster_labels.csv")]
            if not lbl:
                raise FileNotFoundError("No *_cluster_labels.csv (re-run analysis "
                                        "to generate per-loc labels)")
            stem = lbl[0][:-len("_cluster_labels.csv")]
            labels_df = pd.read_csv(os.path.join(extras, lbl[0]))
            stats_path = os.path.join(extras, f"{stem}_cluster_stats.csv")
            stats_df = pd.read_csv(stats_path) if os.path.isfile(stats_path) else None
            px_um = 1.0
            params_path = os.path.join(extras, f"{stem}_params.json")
            if os.path.isfile(params_path):
                try:
                    with open(params_path) as fh:
                        pj = json.load(fh)
                    px_um = float(pj.get("pixel_size_um", 1.0)) or 1.0
                    if pj.get("cluster_eps_nm") is not None:
                        self._cl_eps_nm = int(round(float(pj["cluster_eps_nm"])))
                    if pj.get("cluster_min_samples") is not None:
                        self._cl_min_samples = int(pj["cluster_min_samples"])
                except Exception:
                    pass
            self._cl_xy_um = np.column_stack([
                labels_df["x_um"].to_numpy(dtype=np.float32),
                labels_df["y_um"].to_numpy(dtype=np.float32)])
            self._cl_xy_um_full = self._cl_xy_um     # recluster always re-reads this

            self._cl_labels = labels_df["cluster_id"].to_numpy(dtype=np.int32)
            self._cl_motion = (labels_df["motion"].astype(str).to_numpy()
                               if "motion" in labels_df.columns else None)
            self._cl_motion_tried_derive = False   # allow re-derive for new clusters
            self._cl_px_um = px_um
            self._cl_xy_px = np.column_stack([
                self._cl_xy_um[:, 1] / px_um, self._cl_xy_um[:, 0] / px_um])
            self._cl_stats_df = stats_df
            self._cl_extras_dir = extras
            self._cl_stem = stem
            # Colour-by-motion needs per-track motion classes, which live in the
            # run's trajectories/diffusion CSVs, NOT the cluster-labels file. If
            # a cluster map is opened on its own, pull those siblings in AS DATA
            # ONLY (no track overlay) so the scatter can be coloured by motion
            # without cluttering the view with track tails.
            self._load_motion_source(extras, stem)
            # Only fall back to ID when there's genuinely no motion data anywhere
            # (no motion column AND nothing to derive it from).
            if (self._cl_motion is None and not self._runs and not self._motion_src
                    and self._cl_color_mode in ("Motion", "Cluster motion")):
                self._cl_color_mode = "ID"
            self._cl_present = True
            self._cl_visible = True          # new clusters load shown
            self._render_cluster_layer()
            self._update_cluster_counts()
            self._rebuild_layers()           # surface the cluster layer row + toggle
            self.clusterChanged.emit()
            # Surface the viewer island even with no tracks loaded, then fit the
            # view to the scatter once the island has been shown + sized.
            self.dataChanged.emit()
            QTimer.singleShot(0, self.resetView)
            return True
        except Exception as exc:
            self.warn.emit("Couldn't load clusters",
                           f"{os.path.basename(run_dir)}:\n{exc}")
            return False

    def _load_motion_source(self, extras_dir: str, stem: str):
        """Load the sibling trajectories + diffusion CSVs as DATA ONLY (no track
        overlay, no layer rows) so the cluster scatter can be coloured by motion
        class.  The user opened a cluster map, not a run — a hidden track overlay
        still surfaced its tails on zoom-in, so the motion data is kept entirely
        separate from the viewer and used only by _derive_cluster_motion()."""
        if self._runs or self._motion_src:
            return                                # already have motion data
        traj = os.path.join(extras_dir, f"{stem}_trajectories.csv")
        diff = os.path.join(extras_dir, f"{stem}_diffusion_summary.csv")
        if not (os.path.isfile(traj) and os.path.isfile(diff)):
            return                                # standalone export, no tracks
        try:
            import pandas as pd
            df = pd.read_csv(traj)
            dd = pd.read_csv(diff)
            if ({"x", "y", "particle"}.issubset(df.columns)
                    and "motion" in getattr(dd, "columns", [])):
                self._motion_src = [(df, dd)]
        except Exception:
            self._motion_src = []

    _REAL_MOTION = frozenset(_MOTION_ORDER)   # MOTION_CLASS_ORDER + "Unknown"

    def _has_real_motion(self) -> bool:
        """True when the per-loc cluster motion holds at least one genuine class.
        'Unmatched'/'Unclassified' (locs that never made it into a classified
        track) are NOT real — they collapse to the Unknown grey, so a column of
        only those reads as 'no motion data'."""
        import numpy as np
        m = self._cl_motion
        if m is None:
            return False
        try:
            return any(str(x) in self._REAL_MOTION for x in np.unique(m))
        except Exception:
            return False

    def _ensure_cluster_motion(self):
        """Re-derive per-loc cluster motion from the loaded run tracks when the
        analysis column is missing or wholesale 'Unmatched' (the worker's
        loc↔track join can fail entirely, leaving every cluster loc grey).

        Coordinate-joins each cluster loc (µm → px via the run's pixel size) to a
        track row's motion class.  Cheap, runs at most once per cluster load, and
        only replaces the column when it actually recovers matches — so it can
        never make the colouring worse."""
        if self._has_real_motion():
            return                            # analysis column already usable
        if (self._cl_motion_tried_derive or self._cl_xy_um is None
                or (not self._runs and not self._motion_src)):
            return
        self._cl_motion_tried_derive = True
        derived = self._derive_cluster_motion()
        if derived is not None:
            self._cl_motion = derived

    def _derive_cluster_motion(self):
        """Vectorised (rounded px x, px y) → motion join: build the lookup from
        every loaded run's tracks and left-merge the cluster locs onto it.
        Returns a str ndarray, or None if no track motion data is available or
        nothing matched (so the caller keeps the analysis column unchanged)."""
        import numpy as np, pandas as pd
        px = float(self._cl_px_um or 1.0) or 1.0
        frames = []
        # Loaded run overlays + the data-only motion source (a standalone cluster
        # map) both contribute (frame, x, y) → motion rows.
        sources = [(r.get("df"), r.get("diff")) for r in self._runs] + self._motion_src
        for df, diff in sources:
            if (df is None or diff is None
                    or "motion" not in getattr(diff, "columns", [])
                    or not {"x", "y", "particle"}.issubset(getattr(df, "columns", []))):
                continue
            mmap = dict(zip(diff["particle"].astype("int64"),
                            diff["motion"].astype(str)))
            frames.append(pd.DataFrame({
                "kx": np.round(df["x"].to_numpy(dtype=float), 3),
                "ky": np.round(df["y"].to_numpy(dtype=float), 3),
                "motion": df["particle"].astype("int64").map(mmap)
                            .fillna("Unknown").to_numpy(),
            }))
        if not frames:
            return None
        keyed = (pd.concat(frames, ignore_index=True)
                   .drop_duplicates(["kx", "ky"], keep="last"))
        cl = pd.DataFrame({
            "kx": np.round(self._cl_xy_um[:, 0].astype(float) / px, 3),
            "ky": np.round(self._cl_xy_um[:, 1].astype(float) / px, 3),
        })
        merged = cl.merge(keyed, on=["kx", "ky"], how="left")
        motion = merged["motion"].to_numpy()
        if int(pd.notna(merged["motion"]).sum()) == 0:
            return None
        return np.where(pd.isna(motion), "Unmatched", motion).astype(object)

    def _render_cluster_layer(self):
        if self._cl_xy_px is None:
            return
        v = self.ensureViewer()
        # Closed from the LAYERS panel → drop the scatter item from the scene.
        if not self._cl_visible:
            try:    v.clear_points()
            except Exception: pass
            return
        import numpy as np
        self._ensure_cluster_motion()        # recover motion from tracks if the
                                             # worker's loc↔track join came up empty
        ids = self._cl_labels.astype(np.int32)
        mode = self._cl_color_mode
        # Fall back to per-cluster ID colours when there is no real motion data
        # — otherwise "colour by motion" would be a uniform grey blob (every loc
        # tagged "Unmatched"/"Unclassified" collapses to the Unknown grey).
        if mode in ("Motion", "Cluster motion") and not self._has_real_motion():
            mode = "ID"
        ys, xs = self._cl_xy_px[:, 0], self._cl_xy_px[:, 1]
        noise = ids == -1
        if mode in ("Motion", "Cluster motion"):
            pal = self._palette()

            def rgba(hex_str, a):
                c = QtGui.QColor(hex_str).getRgbF()
                return (c[0], c[1], c[2], a)
            mcol = {cls: rgba(pal.get(cls, pal["Unknown"]), 0.85)
                    for cls in _MOTION_ORDER}
            if mode == "Cluster motion":
                # colour every loc in a cluster by that cluster's DOMINANT motion
                from collections import Counter
                dom = {}
                for cid in np.unique(ids[ids >= 0]):
                    ms = self._cl_motion[ids == cid]
                    dom[int(cid)] = (Counter(str(m) for m in ms).most_common(1)[0][0]
                                     if len(ms) else "Unknown")
                brushes = [mcol["Unknown"] if c < 0
                           else mcol.get(dom.get(int(c), "Unknown"), mcol["Unknown"])
                           for c in ids]
            else:
                brushes = [mcol.get(str(m), mcol["Unknown"]) for m in self._cl_motion]
            for i in np.nonzero(noise)[0]:
                brushes[int(i)] = (0.30, 0.30, 0.30, 0.45)
        else:
            import matplotlib
            cmap = matplotlib.colormaps["turbo"]
            valid = ids[ids >= 0]
            lo = float(valid.min()) if valid.size else 0.0
            span = (float(valid.max()) - lo) if valid.size else 1.0
            span = span or 1.0
            brushes = [(0.30, 0.30, 0.30, 0.55) if cid < 0
                       else tuple(cmap((float(cid) - lo) / span)) for cid in ids]
        try:
            v.set_points(ys, xs, ids=ids, brushes=brushes, size=int(self._cl_point_size))
            self._cl_present = True
        except Exception as exc:
            self.warn.emit("Cluster overlay failed", str(exc))

    def _update_cluster_counts(self):
        import numpy as np
        lab = self._cl_labels
        if lab is None or lab.size == 0:
            self._cl_count = 0
            self._cl_status = ""
            return
        n_clu = int(lab.max() + 1) if (lab >= 0).any() else 0
        n_noise = int((lab == -1).sum())
        self._cl_count = n_clu
        self._cl_status = f"{n_clu:,} clusters · {n_noise:,} noise locs"

    @Slot()
    def recluster(self):
        src = self._cl_xy_um_full if self._cl_xy_um_full is not None else self._cl_xy_um
        if src is None:
            return
        import numpy as np
        import pandas as pd
        eps_nm = float(self._cl_eps_nm)
        try:
            from firefly.sptpalm_analysis import compute_clusters
            locs = pd.DataFrame({"x": src[:, 0], "y": src[:, 1],
                                 "frame": np.zeros(len(src), dtype=np.int32)})
            # compute_clusters SUB-SAMPLES for a large eps to bound the DBSCAN
            # neighbour graph, so `labels` may be shorter than `src`.  cluster_xy
            # is the coordinate array that actually aligns with `labels` — use it
            # for every per-loc array or the overlay/pick/Counter paths read past
            # the end and crash (this was the eps=500 crash).
            labels, stats_df, _, cluster_xy = compute_clusters(
                locs, pixel_size_um=1.0, eps_um=eps_nm / 1000.0,
                min_samples=int(self._cl_min_samples))
        except Exception as exc:
            self._cl_status = f"re-cluster failed: {exc}"
            self.clusterChanged.emit()
            return
        if (getattr(stats_df, "attrs", {}) or {}).get("eps_too_large"):
            self._cl_status = (f"eps = {eps_nm:.0f} nm too large — lower it "
                               f"(clustering skipped)")
            self._cl_count = 0
            self.clusterChanged.emit()
            return
        cxy = np.asarray(cluster_xy, dtype=np.float32)
        px = float(self._cl_px_um or 1.0) or 1.0
        self._cl_labels = np.asarray(labels, dtype=np.int32)
        self._cl_xy_um = cxy                                  # displayed set (aligned)
        self._cl_xy_px = np.column_stack([cxy[:, 1] / px, cxy[:, 0] / px])
        self._cl_motion = None                               # old per-loc motion no
        self._cl_motion_tried_derive = False                 # longer aligns → re-derive
        self._cl_stats_df = stats_df
        self._render_cluster_layer()
        self._update_cluster_counts()
        note = ""
        attrs = getattr(stats_df, "attrs", {}) or {}
        if attrs.get("subsampled") and attrs.get("n_used_locs"):
            note = f", subsampled to {int(attrs['n_used_locs']):,}"
        self._cl_status += f"  (eps={eps_nm:.0f} nm, min={self._cl_min_samples}{note})"
        self.clusterChanged.emit()

    @Slot()
    def suggestEps(self):
        if self._cl_xy_um is None:
            return
        import numpy as np
        xy = np.asarray(self._cl_xy_um, dtype=float)
        try:
            from sklearn.neighbors import NearestNeighbors
            n = len(xy)
            k = max(2, min(int(self._cl_min_samples), n - 1))
            nn = NearestNeighbors(n_neighbors=k).fit(xy)
            d, _ = nn.kneighbors(xy)
            kd = np.sort(d[:, -1])
            m = len(kd)
            lo, hi = max(0, int(0.02 * m)), min(m - 1, int(0.98 * m))
            seg = kd[lo:hi + 1]
            mm = len(seg)
            if mm < 3:
                eps_nm = float(np.median(kd)) * 1000.0
            else:
                x = np.arange(mm, dtype=float)
                y0, y1 = seg[0], seg[-1]
                num = np.abs((y1 - y0) * x - (mm - 1) * seg + (mm - 1) * y0)
                den = np.hypot(y1 - y0, mm - 1) or 1.0
                eps_nm = float(seg[int(np.argmax(num / den))]) * 1000.0
        except Exception as exc:
            self._cl_status = f"eps estimate failed: {exc}"
            self.clusterChanged.emit()
            return
        v = int(round(max(5, min(2000, eps_nm))))
        self._cl_status = f"suggested eps ≈ {v} nm (k-distance knee)"
        self.clusterEpsNm = v          # triggers debounced re-cluster

    @Slot(result=bool)
    def exportTunedClusters(self) -> bool:
        if (self._cl_labels is None or self._cl_xy_um is None
                or self._cl_extras_dir is None):
            self.warn.emit("No clusters loaded",
                           "Load a run's cluster map, tune, then export.")
            return False
        import numpy as np
        import pandas as pd
        stem = self._cl_stem or "clusters"
        cols = {"loc_index": np.arange(len(self._cl_labels), dtype=np.int64),
                "x_um": np.asarray(self._cl_xy_um[:, 0], dtype=float),
                "y_um": np.asarray(self._cl_xy_um[:, 1], dtype=float),
                "cluster_id": np.asarray(self._cl_labels, dtype=np.int64)}
        if self._cl_motion is not None and len(self._cl_motion) == len(self._cl_labels):
            cols["motion"] = self._cl_motion
        lpath = os.path.join(self._cl_extras_dir, f"{stem}_cluster_labels_tuned.csv")
        spath = os.path.join(self._cl_extras_dir, f"{stem}_cluster_stats_tuned.csv")
        try:
            pd.DataFrame(cols).to_csv(lpath, index=False)
            if self._cl_stats_df is not None and len(self._cl_stats_df):
                self._cl_stats_df.to_csv(spath, index=False)
        except Exception as exc:
            self.warn.emit("Export failed", str(exc))
            return False
        self._cl_status = f"Exported → {os.path.basename(lpath)} (+ stats)"
        self.clusterChanged.emit()
        return True

    # ═════════════════════════════════════════════════════════════════════
    #  Super-resolution reconstruction
    # ═════════════════════════════════════════════════════════════════════
    @Property(int, notify=srChanged)
    def srPixelNm(self):
        return self._sr_nm

    @srPixelNm.setter
    def srPixelNm(self, v):
        v = max(2, int(v))
        if v != self._sr_nm:
            self._sr_nm = v
            self.srChanged.emit()

    @Property(int, notify=srChanged)
    def srBlurNm(self):
        return self._sr_blur

    @srBlurNm.setter
    def srBlurNm(self, v):
        v = max(0, int(v))
        if v != self._sr_blur:
            self._sr_blur = v
            self.srChanged.emit()

    @Property(str, notify=srChanged)
    def srStatus(self):
        return self._sr_status

    @Property(bool, notify=srChanged)
    def hasSuperresRender(self):
        return self._sr_img is not None

    @Property(bool, notify=srChanged)
    def srRendering(self):
        return self._sr_rendering

    @Slot()
    def renderSuperres(self):
        df = self._tracks_df
        if df is None or not len(df) or not {"x", "y"} <= set(df.columns):
            self._sr_status = "Load a run or trajectories first."
            self.srChanged.emit()
            return
        if self._sr_rendering:
            return
        # render_superres is a Gaussian density map over up to ~10⁵ locs — it
        # blocks for a noticeable beat, so compute it off the GUI thread and
        # apply the result to the viewer on the GUI thread via the drain timer.
        px = float(self._cl_px_um or 1.0)
        sr_nm = float(self._sr_nm)
        blur = float(self._sr_blur)
        field = self._field_px               # full camera frame, when a stack is loaded
        x, y = df["x"].to_numpy(), df["y"].to_numpy()
        self._sr_rendering = True
        self._sr_status = "Rendering…"
        self._sr_result = None
        self.srChanged.emit()
        self._sr_poll.start()

        def _work():
            try:
                from firefly.analysis.fa_render import render_superres
                img = render_superres(x, y, px, sr_nm=sr_nm, blur_nm=blur,
                                      field_px=field)
                self._sr_result = ("ok", img, x, y, sr_nm, field)
            except Exception as exc:
                self._sr_result = ("err", str(exc))
        threading.Thread(target=_work, daemon=True).start()

    def _drain_superres(self):
        """GUI-thread: apply the off-thread render result to the viewer."""
        import numpy as np
        r = self._sr_result
        if r is None:
            return
        self._sr_result = None
        self._sr_poll.stop()
        self._sr_rendering = False
        if r[0] == "err":
            self._sr_status = f"Render failed: {r[1]}"
            self.srChanged.emit()
            return
        _, img, x, y, sr_nm, field = r
        self._sr_img = img
        # Map the rendered image back to CAMERA-px coordinates so it overlays the
        # other backgrounds 1:1.  When rendered to the full field the canvas starts
        # at the origin and spans the whole frame; otherwise it covers just the
        # localisations' bounding box.  Scale is derived from the ACTUAL image size
        # so it's correct even when render_superres caps a huge canvas.
        h, w = img.shape[0], img.shape[1]
        if field is not None:
            H, W = int(field[0]), int(field[1])
            scale = max(W / max(1, w), H / max(1, h), 1e-6)
            translate = (0.0, 0.0)
        else:
            xext = float(np.nanmax(x) - np.nanmin(x))
            yext = float(np.nanmax(y) - np.nanmin(y))
            scale = max(xext / max(1, w), yext / max(1, h), 1e-6)
            translate = (float(np.nanmin(y)), float(np.nanmin(x)))
        try:
            self.ensureViewer().set_superres(img, scale=scale, translate=translate)
        except Exception as exc:
            self._sr_status = f"Layer add failed: {exc}"
            self.srChanged.emit()
            return
        self._sr_status = (f"{img.shape[1]}×{img.shape[0]} px @ {int(sr_nm)} nm/px "
                           f"({len(x):,} locs)")
        self._after_background_change()
        self.srChanged.emit()

    @Slot(result=bool)
    def saveSuperres(self) -> bool:
        import numpy as np
        if self._sr_img is None:
            return False
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None, "Save super-resolution PNG", "superres.png", "PNG image (*.png)")
        if not path:
            return False
        try:
            import matplotlib.image as mpimg
            img = self._sr_img
            vmax = float(np.percentile(img[img > 0], 99.5)) if (img > 0).any() else 1.0
            mpimg.imsave(path, img, cmap="inferno", vmin=0.0,
                         vmax=max(vmax, 1e-9), origin="lower")
            self._sr_status = f"Saved {os.path.basename(path)}"
            self.srChanged.emit()
            return True
        except Exception as exc:
            self._sr_status = f"Save failed: {exc}"
            self.srChanged.emit()
            return False

    # ═════════════════════════════════════════════════════════════════════
    #  Track explorer
    # ═════════════════════════════════════════════════════════════════════
    _EXP_KNOWN_MOTION = ("Immobile", "Confined", "Brownian", "Directed")

    def _build_explorer_data(self):
        import numpy as np
        tracks, diff = self._tracks_df, self._diff_df
        if tracks is None or not len(tracks) or "particle" not in tracks.columns:
            self._exp_df = None
            self._exp_rows = []
            self._exp_count = "Load a run to explore tracks."
            self.explorerChanged.emit()
            return
        df = tracks.groupby("particle").size().rename("length").reset_index()
        if diff is not None and "particle" in getattr(diff, "columns", []):
            cols = [c for c in ("particle", "D", "alpha", "motion") if c in diff.columns]
            df = df.merge(diff[cols], on="particle", how="left")
        for c in ("D", "alpha"):
            if c not in df.columns:
                df[c] = np.nan
        if "motion" not in df.columns:
            df["motion"] = "Unclassified"
        df["motion"] = df["motion"].astype(str)
        self._exp_df = df
        self.refreshExplorer()

    @Slot()
    def refreshExplorer(self):
        import pandas as pd
        d = self._exp_df
        if d is None or not len(d):
            self._exp_rows = []
            self._exp_count = "Load a run to explore tracks."
            self.explorerChanged.emit()
            return
        keep = {m for m, on in self._exp_motion.items() if on}
        known = d["motion"].isin(self._EXP_KNOWN_MOTION)
        mask = ((d["length"] >= self._exp_min_len)
                & (d["D"].isna() | d["D"].between(self._exp_d_min, self._exp_d_max))
                & (d["alpha"].isna() | d["alpha"].between(self._exp_a_min, self._exp_a_max))
                & ((~known) | d["motion"].isin(keep)))
        filt = d[mask]
        self._exp_filtered = filt
        CAP = 1500
        rows = []
        for _, row in filt.head(CAP).iterrows():
            dv, av = row["D"], row["alpha"]
            rows.append({
                "particle": int(row["particle"]),
                "length": int(row["length"]),
                "d": float(dv) if pd.notna(dv) else None,
                "alpha": float(av) if pd.notna(av) else None,
                "motion": str(row["motion"])})
        self._exp_rows = rows
        n = len(filt)
        self._exp_count = (f"{n:,} of {len(d):,} tracks"
                           + (f" (showing first {CAP:,})" if n > CAP else ""))
        self.explorerChanged.emit()

    @Property("QVariantList", notify=explorerChanged)
    def explorerRows(self):
        return self._exp_rows

    @Property(str, notify=explorerChanged)
    def explorerCount(self):
        return self._exp_count

    def _exp_setter(self, attr, v, cast):
        v = cast(v)
        if getattr(self, attr) != v:
            setattr(self, attr, v)
            self.explorerFiltersChanged.emit()
            self._explorer_timer.start()

    @Property(float, notify=explorerFiltersChanged)
    def expDMin(self):
        return self._exp_d_min

    @expDMin.setter
    def expDMin(self, v):
        self._exp_setter("_exp_d_min", v, float)

    @Property(float, notify=explorerFiltersChanged)
    def expDMax(self):
        return self._exp_d_max

    @expDMax.setter
    def expDMax(self, v):
        self._exp_setter("_exp_d_max", v, float)

    @Property(float, notify=explorerFiltersChanged)
    def expAMin(self):
        return self._exp_a_min

    @expAMin.setter
    def expAMin(self, v):
        self._exp_setter("_exp_a_min", v, float)

    @Property(float, notify=explorerFiltersChanged)
    def expAMax(self):
        return self._exp_a_max

    @expAMax.setter
    def expAMax(self, v):
        self._exp_setter("_exp_a_max", v, float)

    @Property(int, notify=explorerFiltersChanged)
    def expMinLen(self):
        return self._exp_min_len

    @expMinLen.setter
    def expMinLen(self, v):
        self._exp_setter("_exp_min_len", v, lambda x: max(1, int(x)))

    @Slot(str, bool)
    def setExpMotion(self, motion: str, on: bool):
        if motion in self._exp_motion and self._exp_motion[motion] != bool(on):
            self._exp_motion[motion] = bool(on)
            self.explorerFiltersChanged.emit()
            self._explorer_timer.start()

    @Property("QVariantMap", notify=explorerFiltersChanged)
    def expMotionMask(self):
        return dict(self._exp_motion)

    @Slot(result=bool)
    def exportFilteredTracks(self) -> bool:
        f = self._exp_filtered
        if f is None or not len(f):
            return False
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None, "Export filtered tracks", "filtered_tracks.csv", "CSV (*.csv)")
        if not path:
            return False
        try:
            f.to_csv(path, index=False)
            self._exp_count = f"Exported {len(f):,} tracks → {os.path.basename(path)}"
            self.explorerChanged.emit()
            return True
        except Exception as exc:
            self.warn.emit("Export failed", str(exc))
            return False
