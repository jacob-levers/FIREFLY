"""MainWindow VisualiseMixin methods, split out of app_qt.py (#7)."""
from __future__ import annotations
import sys
from firefly.ui.ui_helpers import (_make_napari_container_layout_opaque,
                        _hide_napari_chrome, _MOTION_ORDER)
from firefly.analysis.fa_constants import motion_class_colors
from firefly.ui.ui_widgets import _color_chip

import os
import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from firefly import sptpalm_analysis
from firefly import crash_reporter
from firefly.ui.ui_theme import _THEME
from firefly.ui.ui_constants import (TAB_IMPORT, TAB_ANALYSIS, TAB_COMPARE,
                          TAB_VISUALISE, TAB_REPROCESS)


class _NumItem(QtWidgets.QTableWidgetItem):
    """Table cell that displays formatted text but sorts by its numeric value
    (so the Track-explorer table sorts D / α / length as numbers, not strings)."""
    def __init__(self, value, text):
        super().__init__(text)
        self._v = value

    def __lt__(self, other):
        try:
            return self._v < other._v
        except Exception:
            return super().__lt__(other)


class VisualiseMixin:
    def _ws_maybe_init(self, idx: int):
        """If the user just switched to the Workspace tab, try to embed
        a napari viewer.  Idempotent — only the first switch actually
        does work."""
        if self._workspace_initialised:
            return
        if self.tabs.tabText(idx) != TAB_VISUALISE:
            return
        self._workspace_initialised = True   # mark even on failure — don't retry
        self._ws_init_viewer()

    def _ws_init_viewer(self):
        """Try to import napari and embed its viewer into the tab.
        Shows a clear error message in the placeholder on failure."""
        try:
            import napari
        except Exception as exc:
            self._ws_placeholder.setText(
                f"napari failed to import:\n\n  {type(exc).__name__}: {exc}\n\n"
                f"To enable the Workspace tab, run:\n"
                f"    pip install \"napari[pyside6]>=0.4.19\"\n"
                f"and restart FIREFLY.")
            self._ws_placeholder.setStyleSheet(
                "color: #f78166; padding: 40px; font-size: 13px;")
            return

        try:
            # Embedding pattern: create a Viewer with show=False, then
            # take its underlying QtMainWindow as our embedded widget.
            # `viewer.window._qt_window` is the documented internal handle
            # that's been stable across napari 0.4.x.
            viewer = napari.Viewer(show=False)
            qt_window = viewer.window._qt_window
            # Seal the OUTER container so napari's internal size hints
            # can't propagate up and grow the parent FIREFLY window.
            _make_napari_container_layout_opaque(self._ws_container)
            # Replace the placeholder with the viewer widget
            self._ws_container_layout.removeWidget(self._ws_placeholder)
            self._ws_placeholder.deleteLater()
            self._ws_container_layout.addWidget(qt_window)
            _hide_napari_chrome(viewer)
            self._napari_viewer = viewer
        except Exception as exc:
            # Replace placeholder text with the real error
            self._ws_placeholder.setText(
                f"napari is installed but the embedded viewer couldn't start:\n\n"
                f"  {type(exc).__name__}: {exc}\n\n"
                f"This is sometimes caused by a napari version mismatch with\n"
                f"PySide6.  Try:\n"
                f"    pip install --upgrade \"napari[pyside6]>=0.4.19,<0.5\"")
            self._ws_placeholder.setStyleSheet(
                "color: #f78166; padding: 40px; font-size: 13px;")
            import traceback as _tb
            print(f"[FIREFLY] napari embed failed:\n{_tb.format_exc()}",
                  file=sys.stderr)

    def _ws_viewer_or_warn(self):
        """Return the embedded napari viewer or None, with a UI warning if
        unavailable."""
        if self._napari_viewer is None:
            QtWidgets.QMessageBox.warning(
                self, "Workspace not ready",
                "The napari viewer hasn't initialised on this machine.\n"
                "See the Workspace tab for details.")
            return None
        return self._napari_viewer

    # ── Motion-class colours for the viewer ──────────────────────────────
    # The 3-D Visualise viewer colours each motion class (Immobile /
    # Confined / Brownian / Directed / Unknown) both as its own Tracks
    # layer and in the DBSCAN cluster overlay.  Historically these were a
    # single fixed dark palette; the sidebar "Motion colours" selector now
    # lets the user pick the colour-blind-safe Okabe-Ito scheme (the same
    # one the Publication figure theme uses).  Only palettes that read well
    # on the viewer's DARK canvas are offered — the light figure palette is
    # excluded because its deep hues are near-invisible on dark.
    _WS_MOTION_COLOUR_THEMES = {
        "Default": "Dark",
        "Colour-blind safe": "Publication",
    }

    def _ws_motion_theme(self) -> str:
        """Figure-theme name backing the viewer's motion colours, from the
        sidebar selector.  Falls back to the dark default when the control
        isn't built yet or carries an unexpected label."""
        try:
            txt = str(self._ws_motion_colour_mode.currentText()).strip()
        except Exception:
            return "Dark"
        return self._WS_MOTION_COLOUR_THEMES.get(txt, "Dark")

    def _ws_motion_palette(self) -> dict:
        """Resolve the motion-class → hex-colour map for the viewer,
        honouring the sidebar selector.  Single source of truth shared by
        the per-class Tracks layers and the cluster overlay."""
        return motion_class_colors(self._ws_motion_theme())

    def _ws_rebuild_motion_legend(self):
        """(Re)build the sidebar motion-class colour legend from the active
        palette — a swatch + name per class, so the user can read which colour
        means which motion class without opening the napari layer list."""
        host = getattr(self, "_ws_motion_legend", None)
        if host is None:
            return
        lay = host.layout()
        while lay.count():
            it = lay.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        pal = self._ws_motion_palette()
        i = 0
        for cls in _MOTION_ORDER:
            col = pal.get(cls)
            if not col:
                continue
            lay.addWidget(_color_chip(cls, col, 0, show_count=False),
                          i // 2, i % 2, Qt.AlignmentFlag.AlignLeft)
            i += 1

    def _ws_set_cluster_banner(self, n_clu):
        """Show the 'no clusters found' warning banner only when a clustering
        run produced zero clusters."""
        banner = getattr(self, "_ws_cluster_banner", None)
        if banner is not None:
            banner.setVisible(int(n_clu) == 0)

    def _ws_recolour_motion_layers(self, *_args):
        """Live-recolour the per-class Tracks layers (and the motion-coloured
        cluster overlay) when the 'Motion colours' selector changes — in
        place, without a full layer rebuild, so there's no flicker."""
        # The legend reflects the selected palette regardless of whether a
        # viewer/tracks are loaded, so refresh it first.
        try:    self._ws_rebuild_motion_legend()
        except Exception: pass
        v = getattr(self, "_napari_viewer", None)
        if v is None:
            return
        pal = self._ws_motion_palette()
        names = getattr(self, "_ws_motion_layer_names", {}) or {}
        for cls, layer_name in names.items():
            try:
                if layer_name not in v.layers:
                    continue
                layer = v.layers[layer_name]
                rgba = QtGui.QColor(
                    pal.get(cls, pal["Unknown"])).getRgbF()
                n_vertices = len(layer.data)
                layer._track_colors = np.tile(
                    np.asarray(rgba, dtype=float), (n_vertices, 1))
                # Same repaint trick as the build loop: stop napari from
                # re-deriving turbo colours, then nudge tail_length so the
                # vispy node re-reads `_track_colors`.
                try:    layer._recolor_tracks = lambda *a, **kw: None
                except Exception: pass
                try:    layer.tail_length = layer.tail_length
                except Exception: pass
                try:    layer.refresh()
                except Exception: pass
            except Exception:
                # Don't fail the whole recolour over one bad layer, but DO
                # leave a breadcrumb (the build loop logs the same way) so a
                # silent half-recolour is diagnosable rather than invisible.
                import traceback as _tb, sys as _sys
                print(f"[FIREFLY] motion-colour recolour failed for "
                      f"{cls!r}:\n{_tb.format_exc()}", file=_sys.stderr)
        # Recolour the DBSCAN overlay only if it's already on screen — don't
        # conjure one into existence as a side effect of a colour change.
        try:
            if "DBSCAN clusters" in v.layers:
                self._ws_render_cluster_layer()
        except Exception:
            pass

    def _ws_reset_view(self):
        """Re-centre + re-fit the napari camera on all visible layers.
        Recovery affordance for users who've zoomed / panned off the
        sample and need to get back to the data quickly.
        """
        v = self._ws_viewer_or_warn()
        if v is None:
            return
        try:
            v.reset_view()
        except Exception:
            pass

    def _ws_on_load_stack(self):
        v = self._ws_viewer_or_warn()
        if v is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load image stack",
            self.e_file.text() or os.path.expanduser("~"),
            "Image stacks (*.czi *.tif *.tiff);;All files (*)")
        if not path:
            return
        self._ws_load_stack_path(path)

    def _ws_load_stack_path(self, path: str):
        """Load `path` as an Image layer using FIREFLY's loader.  Heavy
        ops happen on the GUI thread — fine for small/medium files; large
        stacks block the UI briefly (acceptable for an interactive
        inspect-this-file workflow)."""
        v = self._ws_viewer_or_warn()
        if v is None:
            return
        try:
            from firefly.sptpalm_analysis import load_file
            self.statusBar().showMessage(f"Loading {os.path.basename(path)} into napari…")
            stack, _, _ = load_file(path, channel=0)
            v.add_image(stack, name=os.path.basename(path),
                        colormap="gray", blending="translucent_no_depth")
            self.statusBar().showMessage(
                f"Loaded {len(stack):,} frames into napari", 5000)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Load failed",
                f"Couldn't load {os.path.basename(path)}:\n\n{exc}")

    def _ws_on_load_tracks(self):
        v = self._ws_viewer_or_warn()
        if v is None:
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load tracks CSV",
            self.e_outdir.text() or os.path.expanduser("~"),
            "Tracks CSV (*trajectories.csv);;All CSVs (*.csv)")
        if not path:
            return
        self._ws_load_tracks_path(path)

    def _ws_load_tracks_path(self, csv_path: str,
                              diff_csv_path: "str | None" = None):
        """Read a trajectories CSV and add as a napari Tracks layer.

        FIREFLY's trajectories.csv has columns particle, frame, x, y.
        napari Tracks expects (track_id, t, [z,] y, x) per row.  If a
        diffusion-summary CSV is also supplied, the tracks are coloured
        by motion class and stored on `self` for click→stats lookup.
        """
        v = self._ws_viewer_or_warn()
        if v is None:
            return
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            need = {"particle", "frame", "x", "y"}
            missing = need - set(df.columns)
            if missing:
                raise ValueError(
                    f"CSV is missing required columns: {sorted(missing)}")

            # Optional sidecar — diffusion summary keyed by particle id
            diff_df = None
            if diff_csv_path and os.path.isfile(diff_csv_path):
                try:    diff_df = pd.read_csv(diff_csv_path)
                except Exception: diff_df = None
            else:
                # Auto-detect: same folder, <prefix>_diffusion_summary.csv
                guess = csv_path.replace("_trajectories.csv",
                                          "_diffusion_summary.csv")
                if guess != csv_path and os.path.isfile(guess):
                    try:    diff_df = pd.read_csv(guess)
                    except Exception: diff_df = None

            # Cache full data BEFORE building the layer so the motion-
            # class filter can re-derive subsets without re-reading the
            # CSV.  _ws_tracks_layer_name is the napari layer name we
            # need to find/remove when the filter rebuilds.
            self._ws_tracks_df         = df
            self._ws_diff_df           = diff_df
            self._ws_tracks_csv_path   = csv_path
            self._ws_tracks_layer_name = os.path.basename(csv_path)

            # Build the initial layer honouring whatever the filter
            # checkboxes are currently set to (defaults to ALL classes
            # checked → no filtering on first load).
            self._ws_apply_motion_filter(initial=True)

            # Populate the Track-explorer table from the freshly-loaded data.
            try:
                self._ws_build_explorer_data()
            except Exception:
                pass

            self.statusBar().showMessage(
                f"Loaded {df['particle'].nunique():,} tracks "
                f"({len(df):,} points) into napari — "
                "click a track to inspect.", 6000)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Load failed",
                f"Couldn't load tracks from {os.path.basename(csv_path)}:\n\n{exc}")

    def _ws_apply_motion_filter(self, *_args, initial: bool = False):
        """Re-build the napari viewer's per-motion-class Tracks layers.

        One layer is created per motion class present in the dataset
        (Immobile / Confined / Brownian / Directed / Unknown), each
        rendered in its colour from the theme-aware motion palette (chosen
        via the sidebar "Motion colours" selector) — the same palettes the
        analysis figures use, so the napari view matches the report PDFs.
        Visibility of individual classes is controlled directly from
        napari's layer list — there's no parallel checkbox UI.

        The min-length spinbox still drops short tracks across all
        layers.  `initial=True` is passed by `_ws_load_tracks_path` on
        first load so we don't bail out when the viewer hasn't been
        warned about yet.
        """
        df = getattr(self, "_ws_tracks_df", None)
        if df is None:
            try: self._ws_filter_status.setText("")
            except Exception: pass
            return

        v = self._ws_viewer_or_warn() if not initial else getattr(
            self, "_napari_viewer", None)
        if v is None:
            return

        try:
            import numpy as _np
            import pandas as pd  # noqa: F401 — used implicitly via df

            diff_df = getattr(self, "_ws_diff_df", None)
            motion_map: dict = {}
            if diff_df is not None and "motion" in diff_df.columns:
                motion_map = dict(zip(diff_df["particle"],
                                      diff_df["motion"]))

            min_len = 1
            try:    min_len = int(self._ws_min_len.value())
            except Exception: pass

            # Per-track length gate.  Apply BEFORE the per-class split
            # so each layer only ever sees ≥ min_len tracks.
            try:
                track_lens = df.groupby("particle").size()
            except Exception:
                track_lens = None

            # ── Tear down any prior per-class layers ──────────────────
            # Recorded layer names from the previous rebuild so we can
            # remove them surgically without touching unrelated layers
            # (the image stack, cluster overlay, etc.).  Also remove the
            # legacy single-layer name if it lingers from an older session.
            old_names = list(
                getattr(self, "_ws_motion_layer_names", {}).values())
            legacy = getattr(self, "_ws_tracks_layer_name", None)
            if legacy:
                old_names.append(legacy)
            for nm in old_names:
                try:
                    if nm in v.layers:
                        v.layers.remove(nm)
                except Exception:
                    pass
            self._ws_motion_layer_names = {}
            self._ws_motion_pids = {}

            # Layer names are just "<Class> Tracks" — no file prefix.
            # napari's layer list is narrow, and the file basename was
            # truncating to a useless ellipsis anyway.  Users have the
            # current file shown in the status bar / title elsewhere.

            # ── Per-class build loop ──────────────────────────────────
            # If we don't have a diffusion summary, treat every track
            # as "Unknown" so the user still sees them — just in a
            # single grey layer.
            pids_all = df["particle"].values
            if motion_map:
                row_motion = _np.array(
                    [motion_map.get(int(p), "Unknown") for p in pids_all])
            else:
                row_motion = _np.array(["Unknown"] * len(pids_all))

            n_visible_total = 0
            n_total = int(df["particle"].nunique())
            built_any = False
            first_layer_for_click = None

            # Theme-aware motion palette from the sidebar selector (defaults
            # to the dark scheme).  Resolved once so every per-class layer in
            # this rebuild uses one consistent palette.
            motion_pal = self._ws_motion_palette()

            for cls in _MOTION_ORDER:
                cls_mask = (row_motion == cls)
                if not cls_mask.any():
                    continue
                sub = df[cls_mask]

                # Length gate
                if track_lens is not None and min_len > 1:
                    ok_pids = set(track_lens[track_lens >= min_len].index)
                    sub = sub[sub["particle"].isin(ok_pids)]

                # napari's Tracks layer needs ≥ 2 vertices per track,
                # else it crashes deep in draw code with a useless
                # IndexError.  Drop single-point particles up front.
                if len(sub) == 0:
                    continue
                try:
                    _counts = sub.groupby("particle").size()
                    _good_pids = _counts[_counts >= 2].index
                    sub = sub[sub["particle"].isin(_good_pids)]
                except Exception:
                    pass
                if len(sub) == 0:
                    continue

                # napari sorts internally by (track_id, frame); doing it
                # ourselves first keeps our per-vertex `_track_colors`
                # array aligned with napari's vertex order.
                sub = sub.sort_values(["particle", "frame"],
                                       kind="mergesort")

                data = _np.column_stack([
                    sub["particle"].values.astype(_np.int64),
                    sub["frame"].values.astype(_np.float64),
                    sub["y"].values.astype(_np.float64),
                    sub["x"].values.astype(_np.float64),
                ])

                layer_name = f"{cls} Tracks"
                try:
                    # `additive` is napari's default Tracks blending — it
                    # disables depth testing, so a SINGLE visible class layer
                    # renders at full brightness instead of the dimming we saw
                    # with `opaque` (whose depth test darkened thin antialiased
                    # lines on the black canvas until a second layer was shown).
                    layer = v.add_tracks(
                        data,
                        name=layer_name,
                        blending="additive",
                    )
                except Exception:
                    import traceback as _tb, sys as _sys
                    _tb.print_exc(file=_sys.stderr)
                    continue

                # Solid per-class colour.  napari auto-derives turbo
                # colours from the (otherwise unused) head/tail/track
                # properties; overwrite `_track_colors` directly with a
                # uniform RGBA so every vertex of every track in this
                # layer wears `motion_pal[cls]`.  Same trick as before —
                # disable `_recolor_tracks` on the instance so napari
                # doesn't stomp on us when the data setter is later called
                # by a refresh.
                try:
                    rgba = QtGui.QColor(
                        motion_pal.get(cls, motion_pal["Unknown"])
                    ).getRgbF()
                    n_vertices = len(data)
                    colors_arr = _np.tile(_np.asarray(rgba, dtype=float),
                                           (n_vertices, 1))
                    layer._track_colors = colors_arr
                    try:    layer._recolor_tracks = lambda *a, **kw: None
                    except Exception: pass
                    # napari's vispy node only reads `_track_colors` when
                    # one of the layer events it listens to fires.  A
                    # plain `refresh()` repaints with the OLD buffer
                    # (turbo via track_id) — hence the user sees purple
                    # until they touch the tail-length slider, which
                    # *does* fire the right event.  Nudge tail_length
                    # by setting it to its current value: triggers
                    # `events.tail_length`, the vispy listener re-reads
                    # `_track_colors`, and the layer is correct from
                    # frame zero.
                    try:
                        layer.tail_length = layer.tail_length
                    except Exception:
                        pass
                    try:    layer.refresh()
                    except Exception: pass
                except Exception:
                    import traceback as _tb, sys as _sys
                    _tb.print_exc(file=_sys.stderr)

                # Record so the next rebuild can clean up & the click
                # resolver can identify which class a particle belongs to.
                self._ws_motion_layer_names[cls] = layer_name
                try:
                    self._ws_motion_pids[cls] = set(
                        int(p) for p in sub["particle"].unique())
                except Exception:
                    self._ws_motion_pids[cls] = set()

                self._attach_track_click_handler(layer)
                if first_layer_for_click is None:
                    first_layer_for_click = layer

                n_visible_total += int(sub["particle"].nunique())
                built_any = True

            self._ws_tracks_layer = first_layer_for_click
            self._ws_inspector.clear()

            try:
                if built_any:
                    self._ws_filter_status.setText(
                        f"{n_visible_total:,} / {n_total:,} tracks visible "
                        f"across {len(self._ws_motion_layer_names)} layer(s)")
                else:
                    self._ws_filter_status.setText(
                        f"0 / {n_total:,} tracks visible")
            except Exception: pass
        except Exception as exc:
            # Filter failures shouldn't tear the GUI down — log + show
            # a transient status message.  Also dump the full traceback
            # to stderr; the status label only has room for the str(exc)
            # and historically "list index out of range" by itself was
            # essentially undiagnosable.
            try:
                import traceback as _tb, sys as _sys
                _tb.print_exc(file=_sys.stderr)
            except Exception:
                pass
            try:
                self._ws_filter_status.setText(f"filter error: {exc}")
            except Exception: pass

    def _ws_currently_visible_pids(self) -> "set[int] | None":
        """Return the set of particle IDs belonging to per-class layers
        that are currently `visible` in the napari viewer.  Returns
        `None` if no per-class layers have been built yet (the caller
        should then treat the full df as visible).
        """
        names = getattr(self, "_ws_motion_layer_names", {}) or {}
        pids_by_cls = getattr(self, "_ws_motion_pids", {}) or {}
        if not names:
            return None
        v = getattr(self, "_napari_viewer", None)
        if v is None:
            return None
        out: set = set()
        for cls, layer_name in names.items():
            try:
                lyr = v.layers[layer_name]
            except Exception:
                continue
            try:
                if not bool(getattr(lyr, "visible", True)):
                    continue
            except Exception:
                pass
            out |= pids_by_cls.get(cls, set())
        return out

    def _ws_on_load_run(self):
        v = self._ws_viewer_or_warn()
        if v is None:
            return
        run_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Pick a FIREFLY run output folder",
            self.e_outdir.text() or os.path.expanduser("~"))
        if not run_dir:
            return
        self._ws_load_run_folder(run_dir)

    def _ws_on_load_clusters(self):
        """Pick a FIREFLY run folder, locate `firefly_extras/{stem}_cluster_
        labels.csv` (+ optional `_cluster_stats.csv`), and load both into
        the Visualise tab as a coloured Points layer.  The DBSCAN sliders
        then re-cluster on this loaded localisation set without touching
        the original on-disk CSV.
        """
        v = self._ws_viewer_or_warn()
        if v is None:
            return
        run_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Pick a FIREFLY run folder containing cluster labels",
            self.e_outdir.text() or os.path.expanduser("~"))
        if not run_dir:
            return
        extras_dir = os.path.join(run_dir, "firefly_extras")
        if not os.path.isdir(extras_dir):
            QtWidgets.QMessageBox.warning(
                self, "No firefly_extras/",
                f"{os.path.basename(run_dir)!r} doesn't look like a "
                f"FIREFLY run folder (no firefly_extras subfolder).")
            return
        labels_files = [f for f in os.listdir(extras_dir)
                        if f.endswith("_cluster_labels.csv")]
        if not labels_files:
            QtWidgets.QMessageBox.warning(
                self, "No cluster labels",
                "This run doesn't have a *_cluster_labels.csv.  Re-run "
                "FIREFLY analysis on the source file to generate one — "
                "older runs only saved per-cluster stats, not per-loc "
                "labels.")
            return
        stem = labels_files[0][:-len("_cluster_labels.csv")]
        try:
            import pandas as _pd
            labels_df = _pd.read_csv(os.path.join(extras_dir,
                                                  labels_files[0]))
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Couldn't read cluster labels",
                f"Reading {labels_files[0]}:\n\n{exc}")
            return
        stats_path = os.path.join(extras_dir, f"{stem}_cluster_stats.csv")
        try:
            stats_df = (_pd.read_csv(stats_path)
                        if os.path.isfile(stats_path) else None)
        except Exception:
            stats_df = None
        # Pixel size for converting µm ↔ px for the napari layer; pull
        # from params.json if present so the points align with any
        # image stack already loaded in the viewer.
        px_um = 1.0
        run_eps_nm = None
        run_min_samples = None
        params_path = os.path.join(extras_dir, f"{stem}_params.json")
        if os.path.isfile(params_path):
            try:
                import json as _json
                with open(params_path) as fh:
                    _pj = _json.load(fh)
                px_um = float(_pj.get("pixel_size_um", 1.0))
                if _pj.get("cluster_eps_nm") is not None:
                    run_eps_nm = float(_pj.get("cluster_eps_nm"))
                if _pj.get("cluster_min_samples") is not None:
                    run_min_samples = int(_pj.get("cluster_min_samples"))
            except Exception:
                pass
        # Cache for live re-clustering and click-inspection.
        try:
            import numpy as _np
            self._ws_cluster_xy_um = _np.column_stack([
                labels_df["x_um"].to_numpy(dtype=_np.float32),
                labels_df["y_um"].to_numpy(dtype=_np.float32),
            ])
            self._ws_cluster_labels = labels_df["cluster_id"].to_numpy(
                dtype=_np.int32)
            # Optional motion column (added after v1.x of FIREFLY).
            # Falls back to None on older runs so the "Color by motion
            # class" toggle just no-ops.
            if "motion" in labels_df.columns:
                self._ws_cluster_motion = (labels_df["motion"]
                                            .astype(str).to_numpy())
            else:
                self._ws_cluster_motion = None
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Invalid cluster_labels CSV",
                f"{labels_files[0]} is missing the expected x_um / "
                f"y_um / cluster_id columns:\n\n{exc}")
            return
        self._ws_cluster_pixel_size_um = px_um if px_um > 0 else 1.0
        # px coords for the overlay — napari Points expects (y, x).
        self._ws_cluster_xy_px = _np.column_stack([
            self._ws_cluster_xy_um[:, 1] / self._ws_cluster_pixel_size_um,
            self._ws_cluster_xy_um[:, 0] / self._ws_cluster_pixel_size_um,
        ])
        self._ws_cluster_stats_df = stats_df
        # Remember the source run so "Export tuned clusters" can write back.
        self._ws_cluster_extras_dir = extras_dir
        self._ws_cluster_stem = stem
        # Sync the live-tune sliders to the run's ACTUAL clustering params, so
        # the displayed eps / min-samples match the loaded labels (and nudging
        # refines from there).  Signals blocked so this doesn't fire a
        # re-cluster on load.
        try:
            if run_eps_nm is not None:
                self._ws_eps_slider.blockSignals(True)
                self._ws_eps_slider.setValue(int(round(run_eps_nm)))
                self._ws_eps_slider.blockSignals(False)
                if getattr(self, "_ws_eps_value", None) is not None:
                    self._ws_eps_value.setText(
                        f"{int(self._ws_eps_slider.value())} nm")
            if run_min_samples is not None:
                self._ws_minsamp_spin.blockSignals(True)
                self._ws_minsamp_spin.setValue(int(run_min_samples))
                self._ws_minsamp_spin.blockSignals(False)
        except Exception:
            pass
        # "Colour by: Motion" only works when the run saved per-loc motion.
        self._ws_set_motion_colour_enabled(self._ws_cluster_motion is not None)
        # Default to Motion colouring (so the dots match the sidebar motion
        # legend out of the box) when the run has per-loc motion; fall back to
        # ID otherwise.  Signals blocked so it doesn't fire an extra render
        # before the one below.
        try:
            self._ws_cluster_color_mode.blockSignals(True)
            self._ws_cluster_color_mode.setCurrentText(
                "Motion" if self._ws_cluster_motion is not None else "ID")
            self._ws_cluster_color_mode.blockSignals(False)
        except Exception:
            pass
        # Export is now possible (a run is loaded).
        if getattr(self, "btn_ws_export_clusters", None) is not None:
            self.btn_ws_export_clusters.setEnabled(True)
        self._ws_render_cluster_layer()
        n_clu = int((self._ws_cluster_labels >= 0).any() and
                    self._ws_cluster_labels.max() + 1) or 0
        n_noise = int((self._ws_cluster_labels == -1).sum())
        self._ws_cluster_status.setText(
            f"{n_clu:,} clusters  |  {n_noise:,} noise locs  "
            f"({stem})")
        self._ws_set_cluster_banner(n_clu)

    @staticmethod
    def _ws_add_points_compat(viewer, data, **kw):
        """add_points robust to napari's Points `edge_color` → `border_color`
        rename (0.5/0.6; the old name was removed in 0.6.x).  Callers pass
        `border_color`; on older napari we retry with `edge_color`."""
        try:
            return viewer.add_points(data, **kw)
        except TypeError as exc:
            if "border_color" in kw and "border_color" in str(exc):
                kw["edge_color"] = kw.pop("border_color")
                return viewer.add_points(data, **kw)
            raise

    def _ws_render_cluster_layer(self):
        """(Re-)create the napari Points layer from the current
        `_ws_cluster_xy_px` + `_ws_cluster_labels` buffers.  Called on
        first load AND after every debounced DBSCAN re-cluster."""
        if self._ws_cluster_xy_px is None:
            return
        v = self._ws_viewer_or_warn()
        if v is None:
            return
        import numpy as _np
        # Preserve the user's zoom/pan across a re-render: removing + re-adding
        # the Points layer otherwise makes napari auto-fit the camera to the new
        # layer, throwing away the current view.  Only capture on a RE-render (a
        # layer already exists); on the first render we let napari fit the data.
        _cam_state = None
        if self._ws_cluster_layer is not None:
            try:
                _cam_state = (tuple(v.camera.center),
                              float(v.camera.zoom),
                              tuple(v.camera.angles))
            except Exception:
                _cam_state = None
        # Drop the old layer if present so colours refresh cleanly.
        if self._ws_cluster_layer is not None:
            try:
                v.layers.remove(self._ws_cluster_layer)
            except Exception:
                pass
            self._ws_cluster_layer = None
        ids = self._ws_cluster_labels.astype(_np.int32)
        pt_size = 3
        try:
            pt_size = int(self._ws_cluster_point_size.value())
        except Exception:
            pass
        # Decide colouring mode.  When the user picks "Motion" but no
        # per-loc motion column is available (older runs), fall back to
        # ID silently — the dropdown stays on the user's choice but the
        # recolour does the safe thing.  Accepts the legacy long names
        # "Cluster ID" / "Motion class" too in case any persisted state
        # restores them.
        mode_text = "ID"
        try:
            mode_text = str(self._ws_cluster_color_mode.currentText())
        except Exception:
            pass
        mode = "Motion" if mode_text.startswith("Motion") else "ID"
        if mode == "Motion" and self._ws_cluster_motion is None:
            mode = "ID"

        # Per-class colours from the theme-aware viewer palette — single
        # source of truth shared with the per-class Tracks layers, so the
        # overlay matches them under whichever "Motion colours" scheme the
        # user picked.  Alpha 0.85 makes the points slightly translucent so
        # the underlying image stays visible.
        def _swatch_rgba(hex_str: str, a: float) -> tuple:
            c = QtGui.QColor(hex_str).getRgbF()
            return (c[0], c[1], c[2], a)
        _vis_pal = self._ws_motion_palette()
        _MOTION_COLORS = {
            cls: _swatch_rgba(_vis_pal.get(cls, _vis_pal["Unknown"]), 0.85)
            for cls in _MOTION_ORDER
        }
        # "Unmatched" is unique to the cluster overlay — used when a
        # cluster point has no recoverable motion class from the linked
        # tracks (e.g. noise points DBSCAN couldn't assign).
        _MOTION_COLORS["Unmatched"] = (0.30, 0.30, 0.30, 0.55)

        try:
            if mode == "Motion":
                # Build the colour array directly — napari's categorical
                # face_color only works for numeric properties.  Clustered
                # points get their motion-class colour (matching the legend);
                # NOISE points (cluster_id == -1) are greyed, so re-tuning eps
                # is visibly reflected (more noise → more grey) — otherwise the
                # overlay looks identical at every eps because motion class is
                # independent of the clustering.
                colors = _np.array([
                    _MOTION_COLORS.get(str(m), _MOTION_COLORS["Unknown"])
                    for m in self._ws_cluster_motion
                ], dtype=float)
                noise = ids == -1
                if colors.shape[0] == ids.shape[0] and noise.any():
                    colors[noise] = [0.30, 0.30, 0.30, 0.45]
                layer = self._ws_add_points_compat(
                    v, self._ws_cluster_xy_px,
                    properties={"cluster_id": ids,
                                "motion":     self._ws_cluster_motion},
                    face_color=colors,
                    border_color="transparent",
                    size=pt_size, opacity=0.85,
                    name="DBSCAN clusters")
            else:
                layer = self._ws_add_points_compat(
                    v, self._ws_cluster_xy_px,
                    properties={"cluster_id": ids},
                    face_color="cluster_id",
                    face_colormap="turbo",
                    border_color="transparent",
                    size=pt_size, opacity=0.85,
                    name="DBSCAN clusters")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Cluster overlay failed",
                f"napari refused to render the Points layer:\n\n{exc}")
            return
        # Dim noise (cluster_id == -1) to grey.  Motion mode already did this in
        # its branch above; ID mode does it here because its colours come from
        # napari's turbo colormap, which we can only override after the layer
        # exists.
        if mode == "ID":
            try:
                colors = _np.asarray(layer.face_color, dtype=float).copy()
                mask = ids == -1
                if colors.shape[0] == ids.shape[0]:
                    colors[mask] = [0.30, 0.30, 0.30, 0.55]
                    layer.face_color = colors
            except Exception:
                pass
        self._ws_cluster_layer = layer
        # Click handler — populates the side panel with cluster stats.
        try:
            @layer.mouse_drag_callbacks.append
            def _on_cluster_click(_layer, event):
                if event.type != "mouse_press":
                    return
                try:
                    cid = self._ws_nearest_cluster_id(event.position)
                except Exception:
                    cid = None
                if cid is None:
                    return
                self._ws_show_cluster_in_inspector(int(cid))
        except Exception:
            pass
        # Restore the pre-render camera on a re-render so tuning eps / point
        # size / colour doesn't snap the view back to the default fit.
        if _cam_state is not None:
            try:
                v.camera.center = _cam_state[0]
                v.camera.zoom = _cam_state[1]
                v.camera.angles = _cam_state[2]
            except Exception:
                pass

    def _ws_nearest_cluster_id(self, world_pos):
        """Find the cluster_id of the loaded point nearest to a napari
        click position.  Mirrors the existing track-click resolver:
        accepts (y, x) or (t, y, x) world positions.
        """
        if self._ws_cluster_xy_px is None or self._ws_cluster_labels is None:
            return None
        if len(world_pos) < 2:
            return None
        import numpy as _np
        # napari Points layer is 2D (Y, X) — only the last two coords matter.
        y = float(world_pos[-2]); x = float(world_pos[-1])
        ys = self._ws_cluster_xy_px[:, 0]
        xs = self._ws_cluster_xy_px[:, 1]
        if ys.size == 0:
            return None
        d2 = (xs - x) ** 2 + (ys - y) ** 2
        idx = int(_np.argmin(d2))
        # Tolerance: only register clicks within ~6 px of a point.
        if d2[idx] > 36.0:
            return None
        return int(self._ws_cluster_labels[idx])

    def _ws_show_cluster_in_inspector(self, cluster_id: int):
        """Render the picked cluster's stats in the side Inspector
        panel via its dedicated `show_cluster` method.  Falls back to
        a "Noise" mode for cluster_id == -1."""
        if cluster_id == -1:
            try:
                self._ws_inspector.show_cluster(
                    cluster_id=-1,
                    note="Noise point — not assigned to any cluster.")
            except Exception:
                pass
            return
        kw = {"cluster_id": cluster_id}
        df = self._ws_cluster_stats_df
        if df is not None and "cluster_id" in df.columns:
            try:
                row = df[df["cluster_id"] == cluster_id]
                if len(row):
                    r = row.iloc[0]
                    for k in ("n_locs", "area_um2",
                              "density_locs_per_um2", "rg_um",
                              "centroid_x_um", "centroid_y_um"):
                        if k in r.index:
                            kw[k] = float(r[k])
            except Exception:
                pass
        # Dominant motion class within the cluster, if motion data
        # is available.  Surfaced as a free-form note so the inspector
        # template doesn't need a new field.
        try:
            if self._ws_cluster_motion is not None:
                import numpy as _np
                from collections import Counter
                mask = self._ws_cluster_labels == cluster_id
                motions = self._ws_cluster_motion[mask]
                if motions.size:
                    counts = Counter(motions.tolist())
                    total = sum(counts.values())
                    top, top_n = counts.most_common(1)[0]
                    frac = 100.0 * top_n / max(1, total)
                    # Build a sorted "all classes" breakdown for context.
                    breakdown = ", ".join(
                        f"{cls} {100.0 * n / total:.0f}%"
                        for cls, n in counts.most_common())
                    kw["note"] = (
                        f"Dominant motion: {top} ({frac:.0f}%)  ·  "
                        f"breakdown: {breakdown}")
        except Exception:
            pass
        try:
            self._ws_inspector.show_cluster(**kw)
        except Exception:
            pass

    def _ws_recluster_now(self):
        """Re-run DBSCAN on the loaded localisations with the slider's eps +
        min_samples and refresh the overlay.

        Runs SYNCHRONOUSLY on the GUI thread: a background QThread here caused
        napari/vispy cross-thread crashes on macOS ("Cannot set parent … in a
        different thread" → hard segfault).  The eps guard in compute_clusters
        bounds the work, so this is a brief wait-cursor pause, not a freeze."""
        if self._ws_cluster_xy_um is None:
            return
        import numpy as _np, pandas as _pd
        eps_nm = float(self._ws_eps_slider.value())
        min_samples = int(self._ws_minsamp_spin.value())
        self._ws_cluster_status.setText(
            f"clustering…  (eps={eps_nm:.0f} nm, min={min_samples})")
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            from firefly.sptpalm_analysis import compute_clusters
            locs = _pd.DataFrame({
                "x": self._ws_cluster_xy_um[:, 0],
                "y": self._ws_cluster_xy_um[:, 1],
                "frame": _np.zeros(len(self._ws_cluster_xy_um), dtype=_np.int32)})
            labels, stats_df, _, _ = compute_clusters(
                locs, pixel_size_um=1.0,
                eps_um=eps_nm / 1000.0, min_samples=min_samples)
        except Exception as exc:
            self._ws_cluster_status.setText(f"re-cluster failed: {exc}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        # eps too large to cluster safely — keep the previous overlay and tell
        # the user to lower it, rather than showing an all-noise field.
        if (getattr(stats_df, "attrs", {}) or {}).get("eps_too_large"):
            self._ws_cluster_status.setText(
                f"eps = {eps_nm:.0f} nm is too large for this data — lower it "
                f"(clustering skipped to avoid a memory blow-up).")
            self._ws_set_cluster_banner(0)
            return
        self._ws_cluster_labels = _np.asarray(labels, dtype=_np.int32)
        self._ws_cluster_stats_df = stats_df
        self._ws_render_cluster_layer()
        n_clu = int(self._ws_cluster_labels.max() + 1) if (
            self._ws_cluster_labels.size
            and (self._ws_cluster_labels >= 0).any()) else 0
        n_noise = int((self._ws_cluster_labels == -1).sum())
        a = getattr(stats_df, "attrs", {}) or {}
        sub = (f"  · sub-sampled to {a['n_used_locs']:,}"
               if a.get("n_used_locs") and a.get("n_input_locs")
               and a["n_used_locs"] < a["n_input_locs"] else "")
        self._ws_cluster_status.setText(
            f"{n_clu:,} clusters  |  {n_noise:,} noise locs  "
            f"(eps={eps_nm:.0f} nm, min={min_samples}){sub}")
        self._ws_set_cluster_banner(n_clu)

    def _ws_suggest_eps(self):
        """Estimate a good eps from the k-distance knee (k = min_samples) — the
        standard DBSCAN heuristic — and set the eps slider to it (which triggers
        a re-cluster).  Synchronous; the k-distance is cheap.  Outlier tails are
        clipped so a few far points can't skew the knee to an absurd value."""
        if self._ws_cluster_xy_um is None:
            return
        import numpy as _np
        xy = _np.asarray(self._ws_cluster_xy_um, dtype=float)
        k = int(self._ws_minsamp_spin.value())
        self._ws_cluster_status.setText("estimating eps (k-distance knee)…")
        QtWidgets.QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            from sklearn.neighbors import NearestNeighbors
            n = len(xy)
            k = max(2, min(k, n - 1))
            nn = NearestNeighbors(n_neighbors=k).fit(xy)
            d, _ = nn.kneighbors(xy)
            kd = _np.sort(d[:, -1])              # k-th NN dist (µm), ascending
            m = len(kd)
            lo = max(0, int(0.02 * m)); hi = min(m - 1, int(0.98 * m))
            seg = kd[lo:hi + 1]; mm = len(seg)
            if mm < 3:
                eps_nm = float(_np.median(kd)) * 1000.0
            else:
                x = _np.arange(mm, dtype=float)
                x0, y0, x1, y1 = 0.0, seg[0], float(mm - 1), seg[-1]
                num = _np.abs((y1 - y0) * x - (x1 - x0) * seg
                              + x1 * y0 - y1 * x0)
                den = _np.hypot(y1 - y0, x1 - x0) or 1.0
                eps_nm = float(seg[int(_np.argmax(num / den))]) * 1000.0
        except Exception as exc:
            self._ws_cluster_status.setText(f"eps estimate failed: {exc}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        # Clamp to the slider's actual range (not a hardcoded 500) so a genuinely
        # large knee on spread-out data isn't silently pinned to an old ceiling.
        _lo, _hi = self._ws_eps_slider.minimum(), self._ws_eps_slider.maximum()
        v = int(round(max(_lo, min(_hi, eps_nm))))
        capped = "  (slider max — raise the range if you need more)" \
            if v >= _hi and eps_nm > _hi else ""
        self._ws_cluster_status.setText(
            f"suggested eps ≈ {v} nm (k-distance knee){capped}")
        if getattr(self, "_ws_eps_value", None) is not None:
            self._ws_eps_value.setText(f"{v} nm")
        self._ws_eps_slider.setValue(v)        # triggers the debounced re-cluster

    def _ws_set_motion_colour_enabled(self, enabled: bool):
        """Enable/disable the 'Motion' colour option depending on whether the
        loaded run carries per-localisation motion data.  If it's disabled
        while selected, fall back to 'ID' so the overlay never silently shows
        ID colours under a 'Motion' label."""
        combo = getattr(self, "_ws_cluster_color_mode", None)
        if combo is None:
            return
        try:
            idx = combo.findText("Motion")
            if idx < 0:
                return
            item = combo.model().item(idx)
            if item is not None:
                item.setEnabled(enabled)
                item.setToolTip("" if enabled else
                                "This run has no per-localisation motion data.")
            if not enabled and combo.currentText().startswith("Motion"):
                combo.blockSignals(True)
                combo.setCurrentText("ID")
                combo.blockSignals(False)
        except Exception:
            pass

    def _ws_on_point_size_changed(self, val):
        """Resize the existing cluster-overlay points live (no full re-render);
        rebuild only if the live resize isn't possible."""
        layer = getattr(self, "_ws_cluster_layer", None)
        if layer is None:
            return
        try:
            layer.size = int(val)
        except Exception:
            try:
                self._ws_render_cluster_layer()
            except Exception:
                pass

    def _ws_export_tuned_clusters(self):
        """Write the current (live-tuned) cluster labels + stats as new
        *_tuned.csv files next to the loaded run (originals untouched), and
        copy the tuned eps / min-samples into the Analysis sidebar so a re-run
        reproduces them."""
        if (getattr(self, "_ws_cluster_labels", None) is None
                or getattr(self, "_ws_cluster_xy_um", None) is None
                or getattr(self, "_ws_cluster_extras_dir", None) is None):
            QtWidgets.QMessageBox.information(
                self, "No clusters loaded",
                "Load a run's cluster map first, tune eps / min-samples, "
                "then export.")
            return
        import numpy as _np, pandas as _pd
        extras = self._ws_cluster_extras_dir
        stem = self._ws_cluster_stem or "clusters"
        xy = self._ws_cluster_xy_um
        labels = self._ws_cluster_labels
        # Same schema the worker writes for {stem}_cluster_labels.csv.
        cols = {
            "loc_index": _np.arange(len(labels), dtype=_np.int64),
            "x_um": _np.asarray(xy[:, 0], dtype=float),
            "y_um": _np.asarray(xy[:, 1], dtype=float),
            "cluster_id": _np.asarray(labels, dtype=_np.int64),
        }
        if (self._ws_cluster_motion is not None
                and len(self._ws_cluster_motion) == len(labels)):
            cols["motion"] = self._ws_cluster_motion
        labels_path = os.path.join(extras, f"{stem}_cluster_labels_tuned.csv")
        stats_path = os.path.join(extras, f"{stem}_cluster_stats_tuned.csv")
        try:
            _pd.DataFrame(cols).to_csv(labels_path, index=False)
            if (self._ws_cluster_stats_df is not None
                    and len(self._ws_cluster_stats_df)):
                self._ws_cluster_stats_df.to_csv(stats_path, index=False)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Export failed",
                f"Couldn't write the tuned cluster CSVs:\n\n{exc}")
            return
        # Sync the tuned params into the Analysis sidebar.
        try:
            self.s_cluster_eps_nm.setValue(float(self._ws_eps_slider.value()))
            self.s_cluster_min_samples.setValue(
                int(self._ws_minsamp_spin.value()))
        except Exception:
            pass
        self._ws_cluster_status.setText(
            f"Exported → {os.path.basename(labels_path)} (+ stats); "
            f"Analysis eps/min-samples updated.")

    def _ws_render_superres(self):
        """Render the loaded localisations into a super-resolution image layer,
        scaled + positioned to overlay the raw image in the napari viewer."""
        df = getattr(self, "_ws_tracks_df", None)
        v = getattr(self, "_napari_viewer", None)
        if v is None:
            return
        if df is None or not len(df) or not {"x", "y"} <= set(df.columns):
            self._ws_sr_status.setText("Load a run or trajectories first.")
            return
        from firefly.analysis.fa_render import render_superres
        px = float(getattr(self, "_ws_cluster_pixel_size_um", 1.0) or 1.0)
        sr_nm = float(self._ws_sr_nm.value())
        blur_nm = float(self._ws_sr_blur.value())
        x = df["x"].to_numpy(); y = df["y"].to_numpy()
        try:
            img = render_superres(x, y, px, sr_nm=sr_nm, blur_nm=blur_nm)
        except Exception as exc:
            self._ws_sr_status.setText(f"Render failed: {exc}")
            return
        self._ws_sr_img = img
        _scale = (sr_nm / 1000.0) / px          # SR-pixel size in camera-px units
        _ty, _tx = float(np.nanmin(y)), float(np.nanmin(x))
        name = "Super-resolution"
        try:
            if name in v.layers:
                del v.layers[name]
        except Exception:
            pass
        try:
            v.add_image(img, name=name, colormap="inferno", blending="additive",
                        scale=(_scale, _scale), translate=(_ty, _tx))
        except Exception as exc:
            self._ws_sr_status.setText(f"Layer add failed: {exc}")
            return
        self.btn_ws_save_sr.setEnabled(True)
        self._ws_sr_status.setText(
            f"{img.shape[1]}×{img.shape[0]} px @ {int(sr_nm)} nm/px "
            f"({len(x):,} locs)")

    def _ws_save_superres(self):
        """Save the last-rendered super-resolution image as a PNG."""
        img = getattr(self, "_ws_sr_img", None)
        if img is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save super-resolution PNG", "superres.png",
            "PNG image (*.png)")
        if not path:
            return
        try:
            import matplotlib.image as _mpimg
            vmax = (float(np.percentile(img[img > 0], 99.5))
                    if (img > 0).any() else 1.0)
            _mpimg.imsave(path, img, cmap="inferno", vmin=0.0,
                          vmax=max(vmax, 1e-9), origin="lower")
            self._ws_sr_status.setText(f"Saved {os.path.basename(path)}")
        except Exception as exc:
            self._ws_sr_status.setText(f"Save failed: {exc}")

    _WS_EXP_KNOWN_MOTION = ("Immobile", "Confined", "Brownian", "Directed")

    def _ws_build_explorer_data(self):
        """Build the per-track table (particle, D, alpha, motion, length) from
        the loaded trajectories + diffusion summary.  Called when a run loads."""
        tracks = getattr(self, "_ws_tracks_df", None)
        diff = getattr(self, "_ws_diff_df", None)
        if (tracks is None or not len(tracks)
                or "particle" not in tracks.columns):
            self._ws_explorer_df = None
            self.btn_ws_export_tracks.setEnabled(False)
            self._ws_exp_count.setText("Load a run to explore tracks.")
            return
        df = tracks.groupby("particle").size().rename("length").reset_index()
        if diff is not None and "particle" in getattr(diff, "columns", []):
            cols = [c for c in ("particle", "D", "alpha", "motion")
                    if c in diff.columns]
            df = df.merge(diff[cols], on="particle", how="left")
        for c in ("D", "alpha"):
            if c not in df.columns:
                df[c] = np.nan
        if "motion" not in df.columns:
            df["motion"] = "Unclassified"
        df["motion"] = df["motion"].astype(str)
        self._ws_explorer_df = df
        self.btn_ws_export_tracks.setEnabled(True)
        self._ws_refresh_explorer()

    def _ws_refresh_explorer(self):
        """Apply the explorer filters and repopulate the table + count."""
        d = getattr(self, "_ws_explorer_df", None)
        if d is None or not len(d):
            self._ws_exp_count.setText("Load a run to explore tracks.")
            return
        dmin, dmax = self._ws_exp_d_min.value(), self._ws_exp_d_max.value()
        amin, amax = self._ws_exp_a_min.value(), self._ws_exp_a_max.value()
        mlen = int(self._ws_exp_min_len.value())
        keep = {m for m, cb in self._ws_exp_motion.items() if cb.isChecked()}
        known = d["motion"].isin(self._WS_EXP_KNOWN_MOTION)
        mask = ((d["length"] >= mlen)
                & (d["D"].isna() | d["D"].between(dmin, dmax))
                & (d["alpha"].isna() | d["alpha"].between(amin, amax))
                & ((~known) | d["motion"].isin(keep)))
        filt = d[mask]
        self._ws_exp_filtered = filt
        CAP = 1500
        shown = filt.head(CAP)
        tbl = self._ws_exp_table
        tbl.setSortingEnabled(False)
        tbl.setRowCount(len(shown))
        for i, (_, row) in enumerate(shown.iterrows()):
            pid = int(row["particle"]); ln = int(row["length"])
            dv, av = row["D"], row["alpha"]
            tbl.setItem(i, 0, _NumItem(pid, str(pid)))
            tbl.setItem(i, 1, _NumItem(dv if pd.notna(dv) else -1.0,
                                       f"{dv:.4g}" if pd.notna(dv) else "—"))
            tbl.setItem(i, 2, _NumItem(av if pd.notna(av) else -1.0,
                                       f"{av:.3g}" if pd.notna(av) else "—"))
            tbl.setItem(i, 3, QtWidgets.QTableWidgetItem(str(row["motion"])))
            tbl.setItem(i, 4, _NumItem(ln, str(ln)))
        tbl.setSortingEnabled(True)
        n = len(filt)
        extra = f"  (showing first {CAP:,})" if n > CAP else ""
        self._ws_exp_count.setText(f"{n:,} of {len(d):,} tracks{extra}")

    def _ws_explorer_row_clicked(self):
        """Centre the viewer on the selected track + fill the inspector."""
        sel = self._ws_exp_table.selectionModel().selectedRows()
        if not sel:
            return
        it = self._ws_exp_table.item(sel[0].row(), 0)
        if it is None:
            return
        try:
            pid = int(it.text())
        except Exception:
            return
        # Reuse the canonical inspector populator (start/end frame, net
        # displacement, path length, straightness, mass, D, α, motion) so a
        # row-click matches a direct viewer click exactly.
        if hasattr(self, "_show_track_in_inspector"):
            try:
                self._show_track_in_inspector(pid)
            except Exception:
                pass
        v = getattr(self, "_napari_viewer", None)
        tracks = getattr(self, "_ws_tracks_df", None)
        if v is not None and tracks is not None:
            t = tracks[tracks["particle"] == pid]
            if len(t):
                cy, cx = float(t["y"].mean()), float(t["x"].mean())
                try:
                    cur = list(v.camera.center)
                    cur[-2], cur[-1] = cy, cx
                    v.camera.center = tuple(cur)
                except Exception:
                    pass

    def _ws_export_filtered_tracks(self):
        """Write the currently-filtered tracks to a CSV."""
        f = getattr(self, "_ws_exp_filtered", None)
        if f is None or not len(f):
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export filtered tracks", "filtered_tracks.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            f.to_csv(path, index=False)
            self._ws_exp_count.setText(
                f"Exported {len(f):,} tracks → {os.path.basename(path)}")
        except Exception as exc:
            self._ws_exp_count.setText(f"Export failed: {exc}")

    def _ws_load_run_folder(self, run_dir: str):
        """Load a complete FIREFLY analysis run:  finds the stack via the
        params.json (if present) and the matching trajectories.csv from
        firefly_extras/.

        If the picked folder isn't itself a run folder but instead
        CONTAINS run folders (e.g. a `batch_results/` parent or any
        folder housing several analyses), descend into it and either
        auto-load the single run found, or pop a chooser if there are
        many.  This is the common confusion the user hit when picking
        "Region's of Interest" — which holds the individual analyses
        but isn't itself one.
        """
        v = self._ws_viewer_or_warn()
        if v is None:
            return
        try:
            import json
            extras_dir = os.path.join(run_dir, "firefly_extras")
            if not os.path.isdir(extras_dir):
                # Try treating run_dir as a PARENT of run folders.
                children = []
                try:
                    for name in sorted(os.listdir(run_dir)):
                        child_path = os.path.join(run_dir, name)
                        if not os.path.isdir(child_path):
                            continue
                        if os.path.isdir(os.path.join(
                                child_path, "firefly_extras")):
                            children.append(child_path)
                except Exception:
                    children = []
                if len(children) == 1:
                    # Unambiguous — just descend.
                    self._ws_load_run_folder(children[0])
                    return
                if len(children) > 1:
                    # Multiple runs → ask which one.
                    names = [os.path.basename(p) for p in children]
                    pick, ok = QtWidgets.QInputDialog.getItem(
                        self,
                        "Pick a run",
                        f"{os.path.basename(run_dir)!r} contains "
                        f"{len(children)} analysis runs.  "
                        f"Which one would you like to load?",
                        names, 0, False)
                    if not ok:
                        return
                    try:
                        chosen = children[names.index(pick)]
                    except ValueError:
                        return
                    self._ws_load_run_folder(chosen)
                    return
                # Otherwise fall through with a clearer error.
                raise FileNotFoundError(
                    f"No firefly_extras/ subfolder in {run_dir}, and "
                    f"no run subfolders were found inside it either.  "
                    f"Pick an individual analysis folder (one that "
                    f"contains a firefly_extras/ subfolder).")
            # Find the params.json (any *_params.json)
            params_files = [f for f in os.listdir(extras_dir)
                            if f.endswith("_params.json")]
            stack_path = None
            stem = None
            if params_files:
                with open(os.path.join(extras_dir, params_files[0])) as fh:
                    params = json.load(fh)
                stack_path = params.get("input_file") or params.get("stem")
                stem = params_files[0][:-len("_params.json")]
            # Fallback: derive from trajectories filename
            if not stem:
                tr_files = [f for f in os.listdir(extras_dir)
                            if f.endswith("_trajectories.csv")]
                if tr_files:
                    stem = tr_files[0][:-len("_trajectories.csv")]
            if not stem:
                raise FileNotFoundError(
                    "Couldn't determine the run's stem (no params.json or "
                    "trajectories.csv found).")

            tracks_path = os.path.join(extras_dir, f"{stem}_trajectories.csv")
            if not os.path.isfile(tracks_path):
                raise FileNotFoundError(
                    f"Missing {os.path.basename(tracks_path)}")

            # Size guard — a very large run can SEGFAULT the GPU viewer (a
            # native crash that takes the whole app down).  Unlike auto-load
            # (which silently skips), the user explicitly asked to open this
            # one, so warn and let them opt in rather than deciding for them.
            n_locs, n_tracks, frames = self._ws_run_counts(run_dir)
            risk = self._ws_load_risk(n_locs, n_tracks, frames)
            if risk:
                resp = QtWidgets.QMessageBox.warning(
                    self, "Large run — may crash the viewer",
                    f"This run is large ({risk}).\n\n"
                    f"Loading it into the 3-D viewer can be very slow and may "
                    f"crash the application on this machine (the GPU layer "
                    f"can run out of memory).\n\nLoad it anyway?",
                    QtWidgets.QMessageBox.StandardButton.Yes
                    | QtWidgets.QMessageBox.StandardButton.No,
                    QtWidgets.QMessageBox.StandardButton.No)
                if resp != QtWidgets.QMessageBox.StandardButton.Yes:
                    self.statusBar().showMessage(
                        "Load cancelled — run not opened in the viewer.", 5000)
                    return

            # If we have a recorded input-file path that still exists, load
            # it as an image layer.  Otherwise just load the tracks (still
            # useful — user can drop the stack later).
            if stack_path and os.path.isfile(stack_path):
                self._ws_load_stack_path(stack_path)
            else:
                self.statusBar().showMessage(
                    "Tracks loaded; original input stack not found.", 5000)
            diff_path = os.path.join(extras_dir,
                                       f"{stem}_diffusion_summary.csv")
            self._ws_load_tracks_path(
                tracks_path,
                diff_csv_path=diff_path if os.path.isfile(diff_path) else None)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Load failed",
                f"Couldn't load run {os.path.basename(run_dir)}:\n\n{exc}")

    # ── Viewer load-size safety guard ────────────────────────────────────────
    # napari's Tracks/Points layers are drawn by Vispy on a native GPU backend
    # (Metal on macOS, OpenGL elsewhere).  Pushing a very large result — hundreds
    # of thousands of track vertices over many thousands of frames — can SEGFAULT
    # that backend, which is a native crash Python's try/except cannot catch (it
    # takes the whole app down).  We therefore estimate the load BEFORE touching
    # napari and refuse / warn when it's dangerously large.
    _WS_MAX_SAFE_LOCS   = 200_000     # total localisations (≈ track vertices)
    _WS_MAX_SAFE_TRACKS = 20_000      # number of trajectories
    _WS_MAX_SAFE_FRAMES = 8_000       # length of the time axis

    def _ws_load_risk(self, n_locs=0, n_tracks=0, frames=0):
        """Return a human-readable reason string if loading a run of this size
        risks crashing the GPU viewer, else None."""
        bits = []
        if n_locs and n_locs > self._WS_MAX_SAFE_LOCS:
            bits.append(f"{int(n_locs):,} localisations")
        if n_tracks and n_tracks > self._WS_MAX_SAFE_TRACKS:
            bits.append(f"{int(n_tracks):,} tracks")
        if frames and frames > self._WS_MAX_SAFE_FRAMES:
            bits.append(f"{int(frames):,} frames")
        return " / ".join(bits) if bits else None

    def _ws_run_counts(self, run_dir: str):
        """Best-effort (n_locs, n_tracks, frames) for a run folder, read from
        its summary_metrics.json (cheap) without loading any heavy data."""
        try:
            import json
            extras = os.path.join(run_dir, "firefly_extras")
            sm = [f for f in os.listdir(extras)
                  if f.endswith("_summary_metrics.json")]
            if sm:
                d = json.load(open(os.path.join(extras, sm[0])))
                return (int(d.get("n_locs", 0)), int(d.get("n_tracks", 0)),
                        int(d.get("frames", 0)))
        except Exception:
            pass
        return (0, 0, 0)

