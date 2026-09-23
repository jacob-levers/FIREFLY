"""Brush ROI editing — painting produces ORDINARY polygons.

The brush is an input method, not a new ROI kind: strokes accumulate in a
boolean mask and each stroke end retraces that mask into the same `(y, x)`
polygon list the worker already rasterises.  That is what keeps `roi_mode`,
`params.json`'s `roi_polygon`, multi-ROI replicate fan-out and the post-hoc
shrink guard working untouched — so these tests pin the conversion's fidelity
and the round trip through the controller.
"""
import numpy as np
import pytest

from firefly.analysis.fa_roi import (count_mask_holes, mask_to_polygons,
                                     polygons_to_mask)


def _iou(a, b):
    union = (a | b).sum()
    return 1.0 if not union else float((a & b).sum() / union)


# ── the pure conversion ──────────────────────────────────────────────────────
def test_tracing_a_mask_and_rasterising_it_back_is_exact():
    """Lossless at tolerance 0 — otherwise a painted ROI would not be the region
    actually analysed."""
    m = np.zeros((64, 80), bool)
    m[10:30, 12:40] = True                       # a blob
    m[45:60, 5:20] = True                        # a second, disconnected blob
    polys = mask_to_polygons(m, simplify_tol=0.0)

    assert len(polys) == 2, "each connected region becomes one polygon"
    assert _iou(polygons_to_mask(polys, m.shape), m) == 1.0


def test_a_region_touching_the_frame_edge_still_closes():
    """Without padding before tracing, an edge-touching region yields an OPEN
    contour that rasterises back to a different shape."""
    m = np.zeros((32, 32), bool)
    m[0:12, 0:12] = True                         # hard against two edges
    back = polygons_to_mask(mask_to_polygons(m, simplify_tol=0.0), m.shape)
    assert _iou(back, m) == 1.0


def test_simplification_keeps_the_area_it_claims():
    """The UI traces at 0.5 px to keep the vertex count sane; that must not move
    the boundary meaningfully."""
    rng = np.random.default_rng(3)
    m = np.zeros((96, 96), bool)
    yy, xx = np.mgrid[:96, :96]
    m |= ((yy - 40) ** 2 + (xx - 45) ** 2) < 22 ** 2      # a disc
    m |= ((yy - 60) ** 2 + (xx - 70) ** 2) < 12 ** 2      # overlapping lobe

    exact = mask_to_polygons(m, simplify_tol=0.0)
    coarse = mask_to_polygons(m, simplify_tol=0.5)
    assert sum(len(p) for p in coarse) < sum(len(p) for p in exact)
    assert _iou(polygons_to_mask(coarse, m.shape), m) > 0.98


def test_an_enclosed_hole_is_reported_and_filled():
    """A polygon list is UNIONed downstream, so a hole cannot be represented.
    Filling it silently would analyse a region the user did not draw."""
    m = np.zeros((64, 64), bool)
    m[10:50, 10:50] = True
    m[25:35, 25:35] = False                      # a hole in the middle

    assert count_mask_holes(m) == 1
    back = polygons_to_mask(mask_to_polygons(m), m.shape)
    assert back[30, 30], "the hole must be filled, not silently dropped"
    assert count_mask_holes(back) == 0


def test_empty_mask_yields_no_polygons():
    assert mask_to_polygons(np.zeros((16, 16), bool)) == []


# ── the controller round trip ────────────────────────────────────────────────
@pytest.fixture
def ctrl():
    pytest.importorskip("PySide6")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from firefly.ui.controllers.roi_controller import RoiController
    app = QApplication.instance() or QApplication([])
    c = RoiController()
    c._img_h, c._img_w = 64, 64          # pretend an image is loaded
    yield c
    c.deleteLater()
    app.processEvents()


def test_painting_produces_polygons_the_worker_can_rasterise(ctrl):
    ctrl.setTool("brush")
    ctrl.brushRadius = 5.0
    ctrl.beginStroke()
    ctrl.paintAt(32.0, 20.0)
    ctrl.paintAt(32.0, 44.0, 32.0, 20.0)        # drag across
    ctrl.endStroke()

    polys = ctrl.getPolygons()
    assert len(polys) == 1, "one continuous stroke is one region"
    mask = polygons_to_mask(polys, (64, 64))
    assert mask[32, 20] and mask[32, 32] and mask[32, 44], (
        "the dragged segment must be filled, not just its endpoints")
    assert not mask[5, 5]


def test_a_fast_drag_paints_a_continuous_stroke(ctrl):
    """Mouse events arrive sparsely; without segment interpolation a quick drag
    leaves a dotted line of disconnected blobs."""
    ctrl.setTool("brush")
    ctrl.brushRadius = 2.0
    ctrl.beginStroke()
    ctrl.paintAt(10.0, 5.0)
    ctrl.paintAt(10.0, 55.0, 10.0, 5.0)         # 50 px in one event
    ctrl.endStroke()

    assert len(ctrl.getPolygons()) == 1, "the stroke broke into separate blobs"


def test_the_brush_extends_an_existing_polygon_roi(ctrl):
    """Switching to the brush must seed from what is already drawn, so the two
    tools compose instead of the brush starting from blank."""
    ctrl.setPolygons([[(5.0, 5.0), (5.0, 25.0), (25.0, 25.0), (25.0, 5.0)]])
    ctrl.setTool("brush")
    ctrl.brushRadius = 4.0
    ctrl.beginStroke()
    ctrl.paintAt(50.0, 50.0)                     # a second, separate region
    ctrl.endStroke()

    mask = polygons_to_mask(ctrl.getPolygons(), (64, 64))
    assert mask[15, 15], "the original polygon was discarded"
    assert mask[50, 50], "the painted region is missing"


def test_the_eraser_removes_painted_area(ctrl):
    ctrl.setTool("brush")
    ctrl.brushRadius = 10.0
    ctrl.beginStroke(); ctrl.paintAt(32.0, 32.0); ctrl.endStroke()
    assert polygons_to_mask(ctrl.getPolygons(), (64, 64))[32, 32]

    ctrl.setTool("eraser")
    ctrl.brushRadius = 12.0
    ctrl.beginStroke(); ctrl.paintAt(32.0, 32.0); ctrl.endStroke()
    assert ctrl.getPolygons() == [], "erasing the whole region must clear it"


def test_undo_restores_the_previous_stroke(ctrl):
    ctrl.setTool("brush")
    ctrl.brushRadius = 4.0
    ctrl.beginStroke(); ctrl.paintAt(20.0, 20.0); ctrl.endStroke()
    after_first = polygons_to_mask(ctrl.getPolygons(), (64, 64)).sum()

    ctrl.beginStroke(); ctrl.paintAt(45.0, 45.0); ctrl.endStroke()
    assert len(ctrl.getPolygons()) == 2

    assert ctrl.canUndoStroke
    ctrl.undoStroke()
    assert len(ctrl.getPolygons()) == 1
    assert polygons_to_mask(ctrl.getPolygons(), (64, 64)).sum() == after_first


def test_switching_back_to_polygon_keeps_the_painted_regions(ctrl):
    ctrl.setTool("brush")
    ctrl.brushRadius = 6.0
    ctrl.beginStroke(); ctrl.paintAt(30.0, 30.0); ctrl.endStroke()
    painted = ctrl.getPolygons()

    ctrl.setTool("polygon")
    assert ctrl.getPolygons() == painted, "leaving the brush dropped the ROI"
    assert not ctrl.brushActive


def test_the_roi_mode_never_changes(ctrl):
    """The whole design rests on this: downstream still sees a polygon ROI, so
    params, replicate fan-out and the post-hoc shrink guard are untouched."""
    before = ctrl.roiMode
    ctrl.setTool("brush")
    ctrl.beginStroke(); ctrl.paintAt(30.0, 30.0); ctrl.endStroke()
    assert ctrl.roiMode == before
