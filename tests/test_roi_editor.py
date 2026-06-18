"""Headless tests for the bespoke RoiEditor (Qt-only ROI drawing/preview).

Skipped in the Qt-less CI image.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np                                     # noqa: E402
import pytest                                          # noqa: E402

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                           # noqa: E402
from PySide6.QtCore import QPointF                      # noqa: E402
from firefly.ui.roi_editor import RoiEditor             # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _editor():
    e = RoiEditor()
    e.resize(400, 400)
    return e


def test_stack_and_frame_slider():
    e = _editor()
    e.set_stack(np.random.default_rng(0).random((8, 30, 40)).astype("float32"))
    assert e._slider.maximum() == 7
    assert e.current_frame == 0
    e._slider.setValue(4)
    assert e.current_frame == 4


def test_polygon_set_get_roundtrip_yx():
    e = _editor()
    e.set_stack(np.zeros((1, 50, 50)))
    polys = [[(5, 5), (5, 20), (20, 20), (20, 5)],
             [(30, 30), (30, 40), (38, 38)]]
    e.set_polygons(polys)
    got = e.polygons()
    assert [len(p) for p in got] == [4, 3]
    # vertices come back as (y, x) in the same order
    assert got[0][0] == (5.0, 5.0)
    assert got[1][2] == (38.0, 38.0)


def test_add_polygons_appends_and_emits():
    e = _editor()
    e.set_stack(np.zeros((1, 50, 50)))
    seen = []
    e.polygonsChanged.connect(lambda: seen.append(1))
    e.set_polygons([[(1, 1), (1, 9), (9, 9)]])     # emit=False inside set
    e.add_polygons([[(20, 20), (20, 30), (30, 30)]])
    assert len(e.polygons()) == 2
    assert len(seen) == 1                           # add emitted once


def test_draft_drawing_creates_polygon():
    e = _editor()
    e.set_stack(np.zeros((1, 50, 50)))
    for p in [(2, 2), (2, 12), (12, 12)]:
        e._add_draft_vertex(QPointF(p[1], p[0]))    # scene is (x, y)
    e._finish_draft()
    assert len(e.polygons()) == 1
    assert len(e.polygons()[0]) == 3


def test_draft_under_three_vertices_is_discarded():
    e = _editor()
    e.set_stack(np.zeros((1, 50, 50)))
    e._add_draft_vertex(QPointF(1, 1))
    e._add_draft_vertex(QPointF(5, 5))
    e._finish_draft()
    assert e.polygons() == []


def test_overlays_add_and_clear():
    e = _editor()
    e.set_stack(np.zeros((3, 40, 40)))
    e.set_detections([10, 20], [10, 20])
    assert e._points_item is not None
    e.clear_detections()
    assert e._points_item is None
    e.set_mask(np.eye(40, dtype=bool))
    assert e._mask_item is not None
    e.clear_mask()
    assert e._mask_item is None
    e.set_max_projection(np.ones((40, 40), "float32"))
    e.set_max_projection_visible(True)
    assert e._maxproj_item is not None and e._maxproj_item.isVisible()


def test_clear_polygons():
    e = _editor()
    e.set_stack(np.zeros((1, 50, 50)))
    e.set_polygons([[(5, 5), (5, 20), (20, 5)]])
    e.clear_polygons()
    assert e.polygons() == []
