"""Phase-6c tests: RoiStore + RoiController editing + params_builder wiring."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

import pytest                                            # noqa: E402

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                            # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

_POLY = [[(1.0, 2.0), (3.0, 4.0), (5.0, 1.0)]]


# ── RoiStore ─────────────────────────────────────────────────────────────────
def test_roi_store_roundtrip(tmp_path):
    from firefly.ui.controllers.roi_store import RoiStore
    s = RoiStore()
    f = str(tmp_path / "cell.tif")
    assert s.get(f) is None and not s.has(f)
    s.set(f, _POLY)
    assert s.has(f)
    got = s.get(f)
    assert got == [[(1.0, 2.0), (3.0, 4.0), (5.0, 1.0)]]
    # keyed by abspath (relative path resolves to the same entry)
    s.clear(f)
    assert not s.has(f)
    s.set(f, [])                                  # empty clears
    assert not s.has(f)


# ── params_builder reads the per-file polygon ───────────────────────────────
def test_params_builder_roi_polygon_from_store():
    from firefly.ui.controllers import params_builder as pb
    from firefly.ui.controllers.roi_store import RoiStore

    class FakeSettings:
        def get_str(self, k, d=""): return d
        def get_float(self, k, d=0.0): return d
        def get_bool(self, k, d=False): return d
        def set(self, k, v): pass
        def sync(self): pass

    class Imp:
        filePath = "/tmp/cell.tif"; outDir = ""; isCsv = False
        overridePx = False; pixelSize = 0.106; overrideFi = False; frameInterval = 0.02

    store = RoiStore()
    # no polygon → None
    p = pb.build_params(FakeSettings(), Imp(), "/tmp/cell.tif", None, roi_store=store)
    assert p["roi_polygon"] is None
    # with a polygon → sent through regardless of roi mode
    store.set("/tmp/cell.tif", _POLY)
    p2 = pb.build_params(FakeSettings(), Imp(), "/tmp/cell.tif", None, roi_store=store)
    assert p2["roi_polygon"] == [[(1.0, 2.0), (3.0, 4.0), (5.0, 1.0)]]


# ── RoiController editing lifecycle ──────────────────────────────────────────
def test_roi_controller_edit_commit(tmp_path):
    from firefly.ui.controllers.roi_store import RoiStore
    from firefly.ui.controllers.roi_controller import RoiController
    store = RoiStore()
    c = RoiController(store)
    f = str(tmp_path / "cell.tif")        # no real image — background just stays empty
    c.editFile(f)
    assert c.editing and c.fileName == "cell.tif"
    c.addVertex(0, 0); c.addVertex(0, 10); c.addVertex(10, 5)
    assert c.draftLength == 3 and c.canClose
    assert c.draftPoints == [[0.0, 0.0], [0.0, 10.0], [10.0, 5.0]]
    assert c.closeDraft() is True
    c.commit()
    assert not c.editing
    assert store.has(f)
    assert store.get(f) == [[(0.0, 0.0), (0.0, 10.0), (10.0, 5.0)]]


def test_roi_controller_edit_reloads_existing(tmp_path):
    from firefly.ui.controllers.roi_store import RoiStore
    from firefly.ui.controllers.roi_controller import RoiController
    store = RoiStore()
    f = str(tmp_path / "cell.tif")
    store.set(f, _POLY)
    c = RoiController(store)
    c.editFile(f)                          # should load the stored polygon
    assert c.polygonCount == 1
    assert c.fileHasRoi(f) is True


def test_roi_controller_cancel_reverts(tmp_path):
    from firefly.ui.controllers.roi_store import RoiStore
    from firefly.ui.controllers.roi_controller import RoiController
    store = RoiStore()
    f = str(tmp_path / "cell.tif")
    store.set(f, _POLY)                     # 1 stored polygon
    c = RoiController(store)
    c.editFile(f)
    c.setPolygons([[[0, 0], [0, 9], [9, 9], [9, 0]]])  # edit in-session
    assert c.polygonCount == 1
    c.cancel()                              # discard → revert to stored
    assert not c.editing
    assert c.getPolygons() == [[[1.0, 2.0], [3.0, 4.0], [5.0, 1.0]]]
