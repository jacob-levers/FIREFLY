"""Headless tests for FireflyViewer (the bespoke Qt-only napari replacement).

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
from PySide6 import QtWidgets                           # noqa: E402
from firefly.ui.viewer import FireflyViewer             # noqa: E402

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
    v.set_stack(np.zeros((16, 16)))   # 2-D promoted to a single frame
    assert v.n_frames == 1


def test_tracks_from_df_build_per_class_and_visibility():
    v = _viewer()
    rng = np.random.default_rng(1)
    rows, motion = [], {}
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
    cls0 = v.class_names()[0]
    v.set_class_visible(cls0, False)
    assert v._class_visible[cls0] is False
    assert v._track_items[cls0].isVisible() is False


def test_points_overlay_and_cluster_pick():
    v = _viewer()
    v.set_points(np.array([10.0, 20.0, 30.0]), np.array([10.0, 20.0, 30.0]),
                 ids=np.array([100, 101, 102]),
                 brushes=["#f00", "#0f0", "#00f"])
    assert v.pick_at(20.0, 20.0, tol=1.0) == ("cluster", 101)
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
    v.set_tracks_from_df(df, {7: "Brownian", 9: "Directed"},
                         {"Brownian": "#0a0", "Directed": "#a00",
                          "Unknown": "#777"})
    assert v.pick_at(5.0, 6.0, tol=2.0) == ("track", 7)
    assert v.pick_at(40.0, 41.0, tol=2.0) == ("track", 9)
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


def test_background_selector():
    v = _viewer()
    v.set_stack(np.random.default_rng(0).random((6, 16, 16)).astype("float32"))
    opts = [v._bg_combo.itemText(i) for i in range(v._bg_combo.count())]
    assert opts == ["Raw movie", "Max projection", "Off"]
    assert v._bg_mode == "Raw movie" and v._img_item.isVisible()
    v._bg_combo.setCurrentText("Max projection")
    assert v._bg_mode == "Max projection" and v._maxproj is not None
    assert v._img_item.isVisible()
    v._bg_combo.setCurrentText("Off")
    assert not v._img_item.isVisible()
    # rendering super-res adds it as a selectable, auto-shown background
    v.set_superres(np.random.default_rng(1).random((32, 32)).astype("float32"),
                   scale=0.5, translate=(2.0, 3.0))
    opts = [v._bg_combo.itemText(i) for i in range(v._bg_combo.count())]
    assert "Super-resolution" in opts and v._bg_mode == "Super-resolution"
    assert v.has_superres


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
    v.resize(400, 400)
    v.show()
    _app.processEvents()
    v.set_stack(np.zeros((1, 200, 200)))
    v.center_on(25.0, 75.0, span=20.0)
    _app.processEvents()
    r = v.visible_rect()
    assert abs(r.center().x() - 75.0) < 2.0
    assert abs(r.center().y() - 25.0) < 2.0
    v.reset_view()   # must not raise
    v.hide()


def test_empty_df_is_graceful():
    v = _viewer()
    out = v.set_tracks_from_df(pd.DataFrame(), {}, {"Unknown": "#777"})
    assert out == {}
    assert v.class_names() == []


def _tracks_df(n_particles=6, length=10):
    rows = []
    for pid in range(n_particles):
        f = np.arange(pid, pid + length)         # staggered start frames
        rows.append(pd.DataFrame({"particle": pid, "frame": f,
                                  "x": np.linspace(0, 40, length) + pid,
                                  "y": np.linspace(0, 30, length)}))
    return pd.concat(rows, ignore_index=True)


def _head_count(v):
    if v._head_item is None or not v._head_item.isVisible():
        return 0
    return v._head_item.count()


def test_timeline_comes_from_tracks_when_no_stack():
    v = _viewer()
    df = _tracks_df()
    v.set_tracks_from_df(df, {p: "Brownian" for p in range(6)},
                         {"Brownian": "#0a0", "Unknown": "#777"})
    assert v.n_frames == int(df["frame"].max()) + 1     # 15, no raw stack needed


def test_head_markers_track_the_current_frame():
    v = _viewer()
    df = _tracks_df()
    v.set_tracks_from_df(df, {p: "Brownian" for p in range(6)},
                         {"Brownian": "#0a0", "Unknown": "#777"})
    v.current_frame = 7
    assert _head_count(v) == int((df["frame"] == 7).sum())
    v.current_frame = 0
    assert _head_count(v) == int((df["frame"] == 0).sum())


def test_play_pause_and_fps_control_the_timer():
    v = _viewer()
    v.set_stack(np.zeros((30, 20, 20), "float32"))
    v._fps_spin.setValue(10)
    v._play_btn.setChecked(True)
    assert v._play_timer.isActive()
    assert v._play_timer.interval() == 100          # 1000 / 10 fps
    v._fps_spin.setValue(20)
    assert v._play_timer.interval() == 50
    v._play_btn.setChecked(False)
    assert not v._play_timer.isActive()


def test_advance_wraps_around():
    v = _viewer()
    v.set_stack(np.zeros((12, 16, 16), "float32"))
    v.current_frame = v.n_frames - 1
    v._advance()
    assert v.current_frame == 0


def test_tail_windows_segments_by_frame():
    v = _viewer()
    df = _tracks_df(n_particles=6, length=10)            # frames 0..14
    v.set_tracks_from_df(df, {p: "Brownian" for p in range(6)},
                         {"Brownian": "#0a0", "Unknown": "#777"})
    item = v._track_items["Brownian"]
    v._tail_spin.setValue(3)
    v.current_frame = 8
    fr = item._frames
    # window is the segments with frame in (8-3, 8] = [6, 8]
    expected = int(((fr > 8 - 3) & (fr <= 8)).sum())
    assert (item._hi - item._lo) == expected
    # a very long tail shows every segment
    v._tail_spin.setValue(100000)
    v.current_frame = 14
    assert (item._hi - item._lo) == len(item._frames)


def test_head_window_includes_future_segments():
    v = _viewer()
    df = _tracks_df(n_particles=6, length=10)            # frames 0..14
    v.set_tracks_from_df(df, {p: "Brownian" for p in range(6)},
                         {"Brownian": "#0a0", "Unknown": "#777"})
    item = v._track_items["Brownian"]
    v._tail_spin.setValue(2)
    v._head_spin.setValue(0)
    v.current_frame = 7
    base = item._hi - item._lo
    v._head_spin.setValue(5)                              # 5 frames ahead
    fr = item._frames
    # window is now (7-2, 7+5] = [6, 12]
    assert (item._hi - item._lo) == int(((fr > 5) & (fr <= 12)).sum())
    assert (item._hi - item._lo) > base


def test_width_control_updates_pen():
    v = _viewer()
    df = _tracks_df(n_particles=4, length=8)
    v.set_tracks_from_df(df, {p: "Brownian" for p in range(4)},
                         {"Brownian": "#0a0", "Unknown": "#777"})
    item = v._track_items["Brownian"]
    v._width_spin.setValue(4.0)
    assert item._pen.widthF() == 4.0


def test_axis_is_max_of_stack_and_tracks():
    v = _viewer()
    df = _tracks_df()                                # frames up to 14
    v.set_tracks_from_df(df, {p: "Brownian" for p in range(6)},
                         {"Brownian": "#0a0", "Unknown": "#777"})
    v.set_stack(np.zeros((40, 30, 30), "float32"))
    assert v.n_frames == max(40, int(df["frame"].max()) + 1)
