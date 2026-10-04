"""An ROI belongs to the file it was drawn on.

Drawing an ROI on one recording and opening the next showed the first file's
ROI on the second.  Opening a file reloaded its polygons but left the brush
session alone: the painted mask, its green preview, the undo history and the
active tool all carried over, and the next stroke extended the previous file's
mask — so the second file's ROI came out as the first file's region plus the
new strokes.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                            # noqa: E402

from firefly.analysis.fa_roi import polygons_to_mask    # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
SHAPE = (64, 64)


class _Settings(dict):
    """In-memory settings (never the real preferences)."""
    def get_str(self, k, d=""): return dict.get(self, k, d)
    def get_float(self, k, d=0.0): return float(dict.get(self, k, d))
    def get_bool(self, k, d=False): return bool(dict.get(self, k, d))
    def get(self, k, d=None): return dict.get(self, k, d)
    def set(self, k, v): self[k] = v
    def sync(self): pass


@pytest.fixture
def two_files(tmp_path):
    import tifffile
    paths = []
    for name, seed in (("a.tif", 1), ("b.tif", 2)):
        stack = np.random.default_rng(seed).normal(100, 5, (3,) + SHAPE).astype(np.uint16)
        p = tmp_path / name
        tifffile.imwrite(p, stack, photometric="minisblack")
        paths.append(str(p))
    return paths


@pytest.fixture
def viewer(two_files):
    from firefly.ui.controllers.roi_controller import RoiController
    from firefly.ui.controllers.roi_store import RoiOverrideStore, RoiStore
    s = _Settings({"analysis/roi_mode": "Manual polygon", "analysis/diameter": 7,
                   "analysis/bg_radius": 10})
    c = RoiController(store=RoiStore(), settings=s, override_store=RoiOverrideStore())
    yield c, two_files
    c.dispose()
    c.deleteLater(); _app.processEvents()


def _mask(c):
    return polygons_to_mask(c.getPolygons(), SHAPE)


def _paint(c, y, x, r=6.0):
    c.setTool("brush")
    c.brushRadius = r
    c.beginStroke()
    c.paintAt(float(y), float(x))
    c.endStroke()


def test_a_painted_roi_does_not_follow_you_to_the_next_file(viewer):
    c, (a, b) = viewer
    c.editFile(a)
    _paint(c, 20, 20)
    c.commit()

    c.editFile(b)
    assert c.getPolygons() == [], "file B opened showing file A's ROI"
    assert not c.hasBrushPreview, "file A's painted region is still drawn over file B"
    assert not c.canUndoStroke

    _paint(c, 45, 45)
    m = _mask(c)
    assert m[45, 45] and not m[20, 20], "B's ROI absorbed A's painted region"
    c.commit()
    assert polygons_to_mask(c._store.get(a), SHAPE)[20, 20]
    assert not polygons_to_mask(c._store.get(b), SHAPE)[20, 20]


def test_opening_another_file_without_saving_starts_clean(viewer):
    """The Import tab can open the next file while the panel is still up."""
    c, (a, b) = viewer
    c.editFile(a)
    _paint(c, 20, 20)
    c.editFile(b)
    assert c.getPolygons() == [] and not c.hasBrushPreview
    _paint(c, 45, 45)
    assert not _mask(c)[20, 20]


def test_coming_back_shows_that_files_own_roi(viewer):
    c, (a, b) = viewer
    c.editFile(a); _paint(c, 20, 20); c.commit()
    c.editFile(b); _paint(c, 45, 45); c.commit()
    c.editFile(a)
    m = _mask(c)
    assert m[20, 20] and not m[45, 45]
    _paint(c, 20, 40)                      # the brush extends A's own region only
    m = _mask(c)
    assert m[20, 20] and m[20, 40] and not m[45, 45]


def test_a_drawn_polygon_does_not_follow_you_either(viewer):
    c, (a, b) = viewer
    c.editFile(a)
    for y, x in ((10, 10), (10, 30), (30, 30)):
        c.addVertex(float(y), float(x))
    assert c.closeDraft()
    c.commit()
    c.editFile(b)
    assert c.getPolygons() == []
