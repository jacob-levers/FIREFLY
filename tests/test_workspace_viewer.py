"""Headless integration tests for the Workspace tab after the napari→pyqtgraph
port: the viewer is a FireflyViewer, motion classes build per-class polylines +
visibility checkboxes, the cluster overlay renders + picks, and the super-res
overlay attaches.  Skipped in the Qt-less CI image.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402
import pytest                                          # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6 import QtWidgets                           # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _window_with_tracks(n=20, seed=2):
    from firefly.ui.app_qt import MainWindow
    w = MainWindow()
    w._ws_init_viewer()
    rng = np.random.default_rng(seed)
    rows, motion = [], {}
    classes = ["Immobile", "Confined", "Brownian", "Directed"]
    for pid in range(n):
        k = int(rng.integers(3, 9))
        rows.append(pd.DataFrame({"particle": pid, "frame": np.arange(k),
                                  "x": rng.uniform(0, 60, k),
                                  "y": rng.uniform(0, 40, k)}))
        motion[pid] = classes[pid % 4]
    w._ws_tracks_df = pd.concat(rows, ignore_index=True)
    w._ws_diff_df = pd.DataFrame({"particle": list(motion),
                                  "motion": list(motion.values())})
    w._ws_apply_motion_filter(initial=True)
    return w


def test_viewer_is_fireflyviewer():
    from firefly.ui.viewer_pg import FireflyViewer
    from firefly.ui.app_qt import MainWindow
    w = MainWindow()
    w._ws_init_viewer()
    assert isinstance(w._napari_viewer, FireflyViewer)


def test_motion_filter_builds_classes_and_checkboxes():
    w = _window_with_tracks()
    v = w._napari_viewer
    assert set(v.class_names()) == {"Immobile", "Confined", "Brownian",
                                    "Directed"}
    # a colour-coded checkbox exists per palette class (incl. Unknown)
    assert {"Immobile", "Confined", "Brownian", "Directed"} <= set(
        w._ws_motion_checks)


def test_visibility_checkbox_hides_class():
    w = _window_with_tracks()
    v = w._napari_viewer
    w._ws_motion_checks["Immobile"].setChecked(False)
    assert v._track_items["Immobile"].isVisible() is False
    vis = w._ws_currently_visible_pids()
    assert vis is not None
    # the hidden class's particles (pid % 4 == 0) are excluded
    assert not any(pid % 4 == 0 for pid in vis)


def test_cluster_overlay_renders_and_picks():
    w = _window_with_tracks()
    v = w._napari_viewer
    rng = np.random.default_rng(5)
    xy = np.column_stack([rng.uniform(0, 40, 150), rng.uniform(0, 60, 150)])
    labels = rng.integers(-1, 6, 150)
    w._ws_cluster_xy_px = xy
    w._ws_cluster_labels = labels
    w._ws_cluster_motion = np.array(["Brownian"] * 150)
    w._ws_render_cluster_layer()
    assert w._ws_cluster_layer is True
    assert v.pick_at(xy[10, 0], xy[10, 1], tol=1.5) == ("cluster",
                                                        int(labels[10]))


def test_superres_overlay_attaches():
    w = _window_with_tracks()
    v = w._napari_viewer
    w._ws_cluster_pixel_size_um = 0.108
    w._ws_sr_nm.setValue(20)
    w._ws_sr_blur.setValue(20)
    w._ws_render_superres()
    assert v.has_superres is True


def test_recolour_preserves_hidden_state():
    w = _window_with_tracks()
    v = w._napari_viewer
    w._ws_motion_checks["Directed"].setChecked(False)
    w._ws_motion_colour_mode.setCurrentText("Colour-blind safe")
    w._ws_recolour_motion_layers()
    assert v._track_items["Directed"].isVisible() is False
