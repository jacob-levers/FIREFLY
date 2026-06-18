"""Headless tests for FireflyViewer (the pyqtgraph napari replacement).

Builds the widget offscreen (skipped in the Qt-less CI image) and exercises the
data paths: stack scrubbing, per-class tracks + visibility, points overlay,
super-res overlay, camera centring, and spatial click resolution.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np                                     # noqa: E402
import pandas as pd                                    # noqa: E402
import pytest                                          # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
from PySide6 import QtWidgets                           # noqa: E402
from firefly.ui.viewer_pg import FireflyViewer         # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _viewer():
    return FireflyViewer()


def test_set_stack_frames_and_scrubbing():
    v = _viewer()
    stack = np.random.default_rng(0).random((10, 32, 48)).astype("float32")
    v.set_stack(stack)
    assert v.n_frames == 10
    assert v.current_frame == 0
    seen = []
    v.frameChanged.connect(lambda i: seen.append(i))
    v.current_frame = 5
    assert v.current_frame == 5
    assert seen == [5]
    # 2-D input is promoted to a single frame
    v.set_stack(np.zeros((16, 16)))
    assert v.n_frames == 1


def test_tracks_from_df_build_per_class_and_visibility():
    v = _viewer()
    rng = np.random.default_rng(1)
    rows = []
    motion = {}
    classes = ["Immobile", "Brownian", "Directed"]
    for pid in range(12):
        n = int(rng.integers(3, 8))
        rows.append(pd.DataFrame({"particle": pid, "frame": np.arange(n),
                                  "x": rng.uniform(0, 50, n),
                                  "y": rng.uniform(0, 50, n)}))
        motion[pid] = classes[pid % 3]
    df = pd.concat(rows, ignore_index=True)
    colors = {"Immobile": "#888", "Brownian": "#0a0", "Directed": "#a00",
              "Unknown": "#777"}
    pids_by_cls = v.set_tracks_from_df(df, motion, colors)
    assert set(pids_by_cls) <= {"Immobile", "Brownian", "Directed"}
    assert sum(len(s) for s in pids_by_cls.values()) == 12
    assert set(v.class_names()) == set(pids_by_cls)
    # visibility toggle hides the item and removes it from picking
    cls0 = v.class_names()[0]
    v.set_class_visible(cls0, False)
    assert v._class_visible[cls0] is False
    assert v._track_items[cls0].isVisible() is False


def test_points_overlay_and_cluster_pick():
    v = _viewer()
    ys = np.array([10.0, 20.0, 30.0])
    xs = np.array([10.0, 20.0, 30.0])
    v.set_points(ys, xs, ids=np.array([100, 101, 102]),
                 brushes=["#f00", "#0f0", "#00f"])
    # a click right on the middle point resolves to its id
    hit = v.pick_at(20.0, 20.0, tol=1.0)
    assert hit == ("cluster", 101)
    # far away → no hit
    assert v.pick_at(200.0, 200.0, tol=1.0) is None
    v.clear_points()
    assert v._point_xy is None


def test_track_pick_resolves_particle_id():
    v = _viewer()
    df = pd.DataFrame({
        "particle": [7, 7, 7, 9, 9, 9],
        "frame": [0, 1, 2, 0, 1, 2],
        "x": [5.0, 6.0, 7.0, 40.0, 41.0, 42.0],
        "y": [5.0, 5.0, 5.0, 40.0, 40.0, 40.0]})
    motion = {7: "Brownian", 9: "Directed"}
    v.set_tracks_from_df(df, motion, {"Brownian": "#0a0", "Directed": "#a00",
                                      "Unknown": "#777"})
    assert v.pick_at(5.0, 6.0, tol=2.0) == ("track", 7)
    assert v.pick_at(40.0, 41.0, tol=2.0) == ("track", 9)
    # hidden class is excluded from picking
    v.set_class_visible("Directed", False)
    assert v.pick_at(40.0, 41.0, tol=2.0) is None


def test_points_take_priority_over_tracks():
    v = _viewer()
    df = pd.DataFrame({"particle": [1, 1], "frame": [0, 1],
                       "x": [10.0, 11.0], "y": [10.0, 10.0]})
    v.set_tracks_from_df(df, {1: "Brownian"},
                         {"Brownian": "#0a0", "Unknown": "#777"})
    v.set_points([10.0], [10.0], ids=[55])
    assert v.pick_at(10.0, 10.0, tol=2.0) == ("cluster", 55)


def test_superres_overlay_add_clear():
    v = _viewer()
    assert v.has_superres is False
    img = np.random.default_rng(2).random((64, 64)).astype("float32")
    v.set_superres(img, scale=0.5, translate=(3.0, 4.0))
    assert v.has_superres is True
    v.clear_superres()
    assert v.has_superres is False


def test_center_on_keeps_span_and_recentres():
    v = _viewer()
    v.set_stack(np.zeros((1, 100, 100)))
    v.center_on(25.0, 75.0, span=20.0)
    (xr, yr) = v._vb.viewRange()
    assert abs((xr[0] + xr[1]) / 2 - 75.0) < 1e-6
    assert abs((yr[0] + yr[1]) / 2 - 25.0) < 1e-6
    v.reset_view()   # must not raise


def test_empty_df_is_graceful():
    v = _viewer()
    out = v.set_tracks_from_df(pd.DataFrame(), {}, {"Unknown": "#777"})
    assert out == {}
    assert v.class_names() == []
