"""Phase-8a-pre: unit tests for the extracted pure viewer-core logic.

These cover the rendering/picking primitives that the native FireflyViewer and
the in-scene FireflyPaintedItem (Phase 8) share, so a regression in either shows
up here once, not twice.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np                                       # noqa: E402
import pytest                                            # noqa: E402

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                             # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from firefly.ui import _viewer_core as C                 # noqa: E402


def test_robust_levels_and_gray_qimage():
    arr = np.array([[0, 50], [100, np.nan]], float)
    lo, hi = C._robust_levels(arr)
    assert lo <= hi
    img = C._gray_qimage(np.array([[0, 255]], float), (0, 255))
    assert not img.isNull() and img.width() == 2 and img.height() == 1


def test_order_classes():
    assert C._order_classes(["Brownian", "Zed", "Immobile"]) == \
        ["Immobile", "Brownian", "Zed"]


def test_build_class_model_and_tail_window():
    # two tracks, 3 vertices each → 2 segments each, 4 total
    trajs = [(7, np.array([[0., 0.], [1., 1.], [2., 2.]]), np.array([0, 1, 2])),
             (9, np.array([[5., 5.], [6., 6.], [7., 7.]]), np.array([3, 4, 5]))]
    m = C.build_class_model(trajs)
    assert m is not None
    assert len(m["lines"]) == 4
    # frames sorted ascending (the searchsorted invariant)
    assert list(m["frames"]) == sorted(m["frames"])
    assert m["pick_pid"].tolist().count(7) == 3 and m["pick_pid"].tolist().count(9) == 3
    # tail window: at t=4, tail=2 → segments with frame in (2,4] → frames 3,4
    lo, hi = C.tail_window(m["frames"], t=4, tail=2)
    assert set(m["frames"][lo:hi]) <= {3, 4}
    # tail<=0 → from the start
    assert C.tail_window(m["frames"], t=5, tail=0)[0] == 0


def test_build_class_model_empty():
    assert C.build_class_model([(1, np.array([[0., 0.]]), None)]) is None  # <2 verts


def test_df_to_class_trajs():
    import pandas as pd
    df = pd.DataFrame({"particle": [0, 0, 0, 1, 1, 1], "frame": [0, 1, 2, 0, 1, 2],
                       "x": [1., 2., 3., 4., 5., 6.], "y": [1., 1., 1., 2., 2., 2.]})
    mm = {0: "Brownian", 1: "Immobile"}
    by = C.df_to_class_trajs(df, mm, min_len=2)
    assert set(by) == {"Brownian", "Immobile"}
    assert by["Brownian"][0][0] == 0
    assert C.df_to_class_trajs(None, {}) == {}


def test_group_points_by_color():
    groups, rect = C.group_points_by_color(
        [0, 1, 2], [0, 1, 2], ["#ff0000", "#ff0000", "#00ff00"], 4)
    assert len(groups) == 2                       # 2 distinct colours
    assert rect.width() > 0


def test_head_xy_for_frame():
    xy = np.array([[0., 0.], [1., 1.], [2., 2.]])
    track_pick = {"Brownian": (xy, np.array([0, 0, 0]))}
    track_frames = {"Brownian": np.array([0, 1, 2])}
    xs, ys = C.head_xy_for_frame(track_pick, track_frames, {"Brownian": True}, 1)
    assert list(xs) == [1.0] and list(ys) == [1.0]
    # hidden class → empty
    xs2, _ = C.head_xy_for_frame(track_pick, track_frames, {"Brownian": False}, 1)
    assert len(xs2) == 0


def test_pick_at_points_priority_and_tracks():
    pts = np.array([[10., 10.]])
    xy = np.array([[0., 0.], [1., 1.]])
    tp = {"Brownian": (xy, np.array([3, 3]))}
    cv = {"Brownian": True}
    # near a point → cluster (priority)
    assert C.pick_at(pts, np.array([42]), tp, cv, 10.1, 10.1, tol=1.0) == ("cluster", 42)
    # near a track vertex, away from points → track
    assert C.pick_at(pts, np.array([42]), tp, cv, 0.0, 0.0, tol=1.0) == ("track", 3)
    # nothing within tol → None
    assert C.pick_at(pts, np.array([42]), tp, cv, 500, 500, tol=1.0) is None
    # hidden class is not pickable
    assert C.pick_at(None, None, tp, {"Brownian": False}, 0.0, 0.0, tol=1.0) is None
