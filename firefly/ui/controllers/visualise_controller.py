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

from PySide6 import QtWidgets
from PySide6.QtCore import Property, QObject, Signal, Slot

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
        cid = int(cid)
        if cid == -1:
            self._inspector = {"mode": "cluster", "cluster_id": -1, "note": "Noise point"}
        else:
            self._inspector = {"mode": "cluster", "cluster_id": cid}
        self._inspector_visible = True
        self.inspectorChanged.emit()

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
