"""Phase-8a: headless tests for the in-scene FireflyView (QQuickPaintedItem).

Exercises the public surface VisualiseController drives, the camera math, the
pick path, and an off-screen paint(), without a window.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

import numpy as np                                       # noqa: E402
import pytest                                            # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("PySide6.QtQuick")
from PySide6 import QtGui, QtWidgets                     # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

from firefly.ui.firefly_view import FireflyView          # noqa: E402


def _df():
    import pandas as pd
    rows = []
    for pid in range(4):
        for f in range(10):
            rows.append((pid, f, 10 + pid * 5 + f, 10 + pid * 3))
    return pd.DataFrame(rows, columns=["particle", "frame", "x", "y"])


def _view(w=200, h=200):
    v = FireflyView()
    v.setWidth(w); v.setHeight(h)
    return v


def test_set_stack_background_options():
    v = _view()
    v.set_stack(np.zeros((6, 40, 50), np.float32))
    assert v.n_frames == 6
    assert "Raw movie" in v.background_options() and "Max projection" in v.background_options()
    v.background_mode = "Max projection"
    assert v.background_mode == "Max projection"


def test_tracks_model_and_pick():
    v = _view()
    pids = v.set_tracks_from_df(_df(), {0: "Brownian", 1: "Immobile",
                                        2: "Brownian", 3: "Directed"},
                               {"Brownian": "#4a90d9", "Immobile": "#e05252",
                                "Directed": "#7ed321", "Unknown": "#888"})
    assert set(pids) == {"Brownian", "Immobile", "Directed"}
    assert v.n_frames == 10
    assert set(v.class_names()) == {"Brownian", "Immobile", "Directed"}
    # pick a known vertex (particle 0 at frame 0 → x=10,y=10) in data space
    assert v.pick_at(10.0, 10.0, tol=1.0) == ("track", 0)
    # hidden class isn't pickable
    v.set_class_visible("Brownian", False)
    assert v.pick_at(10.0, 10.0, tol=1.0) != ("track", 0)


def test_points_priority_pick():
    v = _view()
    v.set_tracks_from_df(_df(), {0: "Brownian"}, {"Brownian": "#4a90d9"})
    v.set_points([10.0], [10.0], ids=[99], brushes=["#ff0000"], size=5)
    assert v.pick_at(10.0, 10.0, tol=1.0) == ("cluster", 99)
    v.clear_points()
    assert v.pick_at(10.0, 10.0, tol=1.0) == ("track", 0)


def test_camera_reset_and_roundtrip():
    v = _view(200, 200)
    v.set_stack(np.zeros((4, 100, 100), np.float32))      # triggers reset_view
    # the 100x100 image fits into 200x200 → scale ~ 1.96, centred
    assert 1.5 < v._scale < 2.1
    # widget centre maps back to image centre (50,50)
    y, x = v._to_data(100, 100)
    assert abs(x - 50) < 1.0 and abs(y - 50) < 1.0
    r = v.visible_rect()
    assert r.width() > 0 and r.contains(50, 50)


def test_transport_and_frame_signal():
    v = _view()
    v.set_stack(np.random.default_rng(0).random((8, 30, 30)).astype(np.float32))
    seen = []
    v.frameChanged.connect(lambda i: seen.append(i))
    v.current_frame = 3
    assert v.current_frame == 3 and seen == [3]
    v.tail = 50; v.head = 5; v.track_width = 3.0; v.fps = 20
    assert v.tail == 50 and v.head == 5 and v.track_width == 3.0 and v.fps == 20
    pchg = []
    v.playingChanged.connect(lambda: pchg.append(v.playing))
    v.playing = True
    assert v.playing and pchg == [True]
    v.playing = False


def test_paint_renders_without_crash():
    v = _view(220, 180)
    v.set_stack(np.full((3, 60, 80), 128, np.float32))
    v.set_tracks_from_df(_df(), {0: "Brownian", 1: "Brownian"},
                        {"Brownian": "#4a90d9"})
    img = QtGui.QImage(220, 180, QtGui.QImage.Format.Format_ARGB32)
    img.fill(0)
    p = QtGui.QPainter(img)
    v.paint(p)            # the single paint() — must not raise
    p.end()
    # something was drawn (background blit + tracks → non-zero pixels)
    arr = [img.pixelColor(x, y).alpha() for x in range(0, 220, 11)
           for y in range(0, 180, 11)]
    assert any(a > 0 for a in arr)


def test_superres_layer():
    v = _view()
    v.set_superres(np.random.default_rng(1).random((20, 20)).astype(np.float32),
                   scale=2.0, translate=(5.0, 5.0))
    assert v.has_superres and v.background_mode == "Super-resolution"
    assert "Super-resolution" in v.background_options()
    v.clear_superres()
    assert not v.has_superres
