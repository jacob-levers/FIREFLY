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

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Property, QObject, QTimer, Signal, Slot

from firefly.analysis.fa_constants import motion_class_colors
from firefly.ui.ui_helpers import _MOTION_ORDER

# Sidebar "Motion colours" label → figure-theme palette name (mirrors
# VisualiseMixin._WS_MOTION_COLOUR_THEMES).
_MOTION_COLOUR_THEMES = {"Default": "Dark", "Colour-blind safe": "Publication"}


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

        self._tracks_df = None
        self._diff_df = None
        self._motion_pids: dict = {}
        self._class_visible: dict = {}     # class → bool (persists across loads)
        self._min_len = 1
        self._motion_mode = "Default"
        self._layers: list = []
        self._inspector: dict = {"mode": "none"}
        self._inspector_visible = False
        self._has_run = False
        self._hud_tracks = 0

        # ── clusters ─────────────────────────────────────────────────────
        self._cl_xy_um = None
        self._cl_labels = None
        self._cl_motion = None
        self._cl_xy_px = None
        self._cl_px_um = 1.0
        self._cl_stats_df = None
        self._cl_extras_dir = None
        self._cl_stem = None
        self._cl_present = False
        self._cl_eps_nm = 50
        self._cl_min_samples = 8
        self._cl_point_size = 3
        self._cl_color_mode = "Motion"
        self._cl_count = 0
        self._cl_status = ""
        # ── super-resolution ─────────────────────────────────────────────
        self._sr_img = None
        self._sr_nm = 20
        self._sr_blur = 20
        self._sr_status = ""
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
            self._tracks_df = df
            self._diff_df = diff_df
            self._apply_motion_filter()
            self._build_explorer_data()
            self._has_run = True
            self.dataChanged.emit()
            self.statusMessage.emit(
                f"Loaded {df['particle'].nunique():,} tracks "
                f"({len(df):,} points) — click a track to inspect.")
        except Exception as exc:
            self.warn.emit("Load failed",
                           f"Couldn't load tracks from {os.path.basename(csv_path)}:\n{exc}")

    def _apply_motion_filter(self):
        df = self._tracks_df
        if df is None:
            return
        v = self.ensureViewer()
        diff_df = self._diff_df
        motion_map = {}
        if diff_df is not None and "motion" in diff_df.columns:
            motion_map = dict(zip(diff_df["particle"], diff_df["motion"]))
        pal = self._palette()
        pids_by_cls = v.set_tracks_from_df(df, motion_map, pal, min_len=int(self._min_len))
        self._motion_pids = {c: set(s) for c, s in pids_by_cls.items()}
        # Default any newly-seen class to visible; preserve prior toggles.
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
        layers = []
        for cls in _MOTION_ORDER:
            if cls in self._motion_pids:
                layers.append({
                    "id": f"tracks:{cls}", "kind": "tracks", "name": cls,
                    "present": True, "visible": bool(self._class_visible.get(cls, True)),
                    "opacity": 1.0, "colorHex": pal.get(cls, "#aaaaaa"),
                    "motionClass": cls,
                    "count": len(self._motion_pids.get(cls, ())),
                })
        # Background image layers (selectable via the same backgroundMode).
        if self._viewer is not None:
            opts = self._viewer.background_options()
            mode = self._viewer.background_mode
            for name, kind, col in (("Max projection", "maxproj", "#8b949e"),
                                    ("Super-resolution", "superres", "#a371f7")):
                if name in opts:
                    layers.append({
                        "id": f"bg:{name}", "kind": kind, "name": name,
                        "present": True, "visible": (mode == name),
                        "opacity": 1.0, "colorHex": col, "motionClass": "",
                        "count": 0,
                    })
        self._layers = layers
        self.layersChanged.emit()

    @Property("QVariantList", notify=layersChanged)
    def layers(self):
        return self._layers

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
        return self._viewer.fps if self._viewer is not None else 7

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
            if self._tracks_df is not None:
                self._apply_motion_filter()

    @Property("QStringList", constant=True)
    def motionColourModes(self):
        return list(_MOTION_COLOUR_THEMES.keys())

    @Property(str, notify=motionColourModeChanged)
    def motionColourMode(self):
        return self._motion_mode

    @motionColourMode.setter
    def motionColourMode(self, mode):
        if mode in _MOTION_COLOUR_THEMES and mode != self._motion_mode:
            self._motion_mode = mode
            self.motionColourModeChanged.emit()
            if self._viewer is not None:
                try:    self._viewer.recolor_tracks(self._palette())
                except Exception: pass
                for cls, on in self._class_visible.items():
                    try:    self._viewer.set_class_visible(cls, on)
                    except Exception: pass
            self._rebuild_layers()

    @Slot()
    def resetView(self):
        if self._viewer is not None:
            try:    self._viewer.reset_view()
            except Exception: pass

    @Property(bool, notify=dataChanged)
    def hasRun(self):
        return self._has_run

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
        df = self._tracks_df
        if df is None:
            return None
        rows = df[df["particle"] == pid]
        if rows.empty:
            return None
        info = {"mode": "track", "particle_id": pid, "length": int(len(rows)),
                "start_frame": int(rows["frame"].min()),
                "end_frame": int(rows["frame"].max())}
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
        diff = self._diff_df
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
            self.clusterChanged.emit()
            if self._cl_present:
                self._render_cluster_layer()

    @Property("QStringList", constant=True)
    def clusterColorModes(self):
        return ["Motion", "ID"]

    @Property(str, notify=clusterChanged)
    def clusterColorMode(self):
        return self._cl_color_mode

    @clusterColorMode.setter
    def clusterColorMode(self, mode):
        if mode in ("Motion", "ID") and mode != self._cl_color_mode:
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
            self._cl_labels = labels_df["cluster_id"].to_numpy(dtype=np.int32)
            self._cl_motion = (labels_df["motion"].astype(str).to_numpy()
                               if "motion" in labels_df.columns else None)
            self._cl_px_um = px_um
            self._cl_xy_px = np.column_stack([
                self._cl_xy_um[:, 1] / px_um, self._cl_xy_um[:, 0] / px_um])
            self._cl_stats_df = stats_df
            self._cl_extras_dir = extras
            self._cl_stem = stem
            if self._cl_motion is None and self._cl_color_mode == "Motion":
                self._cl_color_mode = "ID"
            self._render_cluster_layer()
            self._update_cluster_counts()
            self.clusterChanged.emit()
            return True
        except Exception as exc:
            self.warn.emit("Couldn't load clusters",
                           f"{os.path.basename(run_dir)}:\n{exc}")
            return False

    def _render_cluster_layer(self):
        if self._cl_xy_px is None:
            return
        v = self.ensureViewer()
        import numpy as np
        ids = self._cl_labels.astype(np.int32)
        mode = self._cl_color_mode
        if mode == "Motion" and self._cl_motion is None:
            mode = "ID"
        ys, xs = self._cl_xy_px[:, 0], self._cl_xy_px[:, 1]
        noise = ids == -1
        if mode == "Motion":
            pal = self._palette()

            def rgba(hex_str, a):
                c = QtGui.QColor(hex_str).getRgbF()
                return (c[0], c[1], c[2], a)
            mcol = {cls: rgba(pal.get(cls, pal["Unknown"]), 0.85)
                    for cls in _MOTION_ORDER}
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
        if self._cl_xy_um is None:
            return
        import numpy as np
        import pandas as pd
        eps_nm = float(self._cl_eps_nm)
        try:
            from firefly.sptpalm_analysis import compute_clusters
            locs = pd.DataFrame({"x": self._cl_xy_um[:, 0], "y": self._cl_xy_um[:, 1],
                                 "frame": np.zeros(len(self._cl_xy_um), dtype=np.int32)})
            labels, stats_df, _, _ = compute_clusters(
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
        self._cl_labels = np.asarray(labels, dtype=np.int32)
        self._cl_stats_df = stats_df
        self._render_cluster_layer()
        self._update_cluster_counts()
        self._cl_status += f"  (eps={eps_nm:.0f} nm, min={self._cl_min_samples})"
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

    @Slot()
    def renderSuperres(self):
        import numpy as np
        df = self._tracks_df
        if df is None or not len(df) or not {"x", "y"} <= set(df.columns):
            self._sr_status = "Load a run or trajectories first."
            self.srChanged.emit()
            return
        from firefly.analysis.fa_render import render_superres
        px = float(self._cl_px_um or 1.0)
        sr_nm = float(self._sr_nm)
        x, y = df["x"].to_numpy(), df["y"].to_numpy()
        try:
            img = render_superres(x, y, px, sr_nm=sr_nm, blur_nm=float(self._sr_blur))
        except Exception as exc:
            self._sr_status = f"Render failed: {exc}"
            self.srChanged.emit()
            return
        self._sr_img = img
        scale = (sr_nm / 1000.0) / px
        try:
            self.ensureViewer().set_superres(
                img, scale=scale, translate=(float(np.nanmin(y)), float(np.nanmin(x))))
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
