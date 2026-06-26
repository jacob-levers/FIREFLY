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
    from firefly.ui.controllers.params import params_builder as pb
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


class _RecSettings:
    """Recording in-memory settings (never touches real QSettings)."""
    def __init__(self): self.d = {}
    def get_str(self, k, d=""): return self.d.get(k, d)
    def get_float(self, k, d=0.0): return float(self.d.get(k, d))
    def get_bool(self, k, d=False): return self.d.get(k, d)
    def set(self, k, v): self.d[k] = v
    def sync(self): pass


def test_roi_viewer_single_mode_mirrors_sidebar_default():
    """Single analysis: ROI-viewer edits write through to the global sidebar
    default so the left-hand Threshold field reflects them."""
    from firefly.ui.controllers.roi_store import RoiStore
    from firefly.ui.controllers.roi_controller import RoiController
    s = _RecSettings()
    c = RoiController(RoiStore(), s)        # default = single mode
    assert c._batch_mode is False
    c.threshold = 0.222
    c.maskMode = "Mean"
    c.autoMethod = "Otsu"
    c.bgSigma = 3.0
    c.roiMode = "Manual threshold"
    assert abs(s.d["analysis/roi_threshold"] - 0.222) < 1e-9
    assert s.d["analysis/roi_mask_mode"] == "Mean"
    assert s.d["analysis/roi_auto_method"] == "Otsu"
    assert abs(s.d["analysis/roi_bg_sigma"] - 3.0) < 1e-9
    assert s.d["analysis/roi_mode"] == "Manual threshold"


def test_roi_viewer_batch_mode_keeps_per_file_override():
    """Batch: ROI-viewer edits stay a per-file override and must NOT move the
    shared sidebar default."""
    from firefly.ui.controllers.roi_store import RoiStore
    from firefly.ui.controllers.roi_controller import RoiController
    s = _RecSettings()
    s.set("analysis/roi_threshold", 0.030)   # the sidebar default
    c = RoiController(RoiStore(), s)
    c.setBatchMode(True)
    c.threshold = 0.444                       # tweak the per-file value in the viewer
    assert s.d["analysis/roi_threshold"] == 0.030, "batch must not move the default"
    assert abs(c.threshold - 0.444) < 1e-9    # but the viewer value still updates


def test_import_batch_mode_drives_roi_viewer():
    """ImportController.batchMode (set by the landing card / mode toggle) flips
    the ROI viewer's single/batch behaviour through the app wiring."""
    from firefly.ui.controllers.import_controller import ImportController
    from firefly.ui.controllers.roi_store import RoiStore
    from firefly.ui.controllers.roi_controller import RoiController
    s = _RecSettings()
    importc = ImportController(s)
    roi = RoiController(RoiStore(), s)
    importc.batchModeChanged.connect(lambda: roi.setBatchMode(importc.batchMode))
    assert importc.batchMode is False
    importc.setBatchMode(True)
    assert roi._batch_mode is True
    importc.setBatchMode(False)
    assert roi._batch_mode is False


# ── single analysis surfaces the auto-combined multi-file series ──────────────
def test_import_shows_combined_series(tmp_path):
    """Single analysis auto-combines a split recording; the Import tab now lists
    exactly the files the run will load (the same _find_tif_series detection),
    with the combined frame total — sister-channel files excluded."""
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    from firefly.ui.controllers.import_controller import ImportController
    tifffile.imwrite(str(tmp_path / "cellA.tif"),         np.zeros((3, 8, 8), np.uint16))
    tifffile.imwrite(str(tmp_path / "cellA-file002.tif"), np.zeros((2, 8, 8), np.uint16))
    tifffile.imwrite(str(tmp_path / "cellA_green.tif"),   np.zeros((9, 8, 8), np.uint16))  # sister → excluded

    c = ImportController(_RecSettings())
    c.filePath = str(tmp_path / "cellA-file002.tif")        # pick a non-primary part
    names = [r["name"] for r in c.seriesFiles]
    assert names == ["cellA.tif", "cellA-file002.tif"]      # ordered, sister excluded
    assert c.seriesCount == 2
    assert c.seriesFrameTotal == 5                          # 3 + 2 (combined total)


def test_import_lone_file_has_no_series(tmp_path):
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    from firefly.ui.controllers.import_controller import ImportController
    tifffile.imwrite(str(tmp_path / "solo.tif"), np.zeros((6, 8, 8), np.uint16))
    c = ImportController(_RecSettings())
    c.filePath = str(tmp_path / "solo.tif")
    assert c.seriesCount == 0                               # no series → nothing to show


def test_import_flags_corrupt_series_part(tmp_path):
    """A corrupt PART of a single-analysis series is flagged unreadable (and blocks
    Start), even when the file you picked is fine — the combined run would fail on
    the bad part."""
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    from firefly.ui.controllers.import_controller import ImportController
    tifffile.imwrite(str(tmp_path / "rec.tif"),         np.zeros((3, 8, 8), np.uint16))
    open(str(tmp_path / "rec-file002.tif"), "w").close()           # 0-byte → corrupt part
    tifffile.imwrite(str(tmp_path / "rec-file003.tif"), np.zeros((4, 8, 8), np.uint16))

    c = ImportController(_RecSettings())
    c.filePath = str(tmp_path / "rec.tif")                         # the picked file is fine
    rows = {r["name"]: r for r in c.seriesFiles}
    assert c.seriesCount == 3
    assert rows["rec.tif"]["unreadable"] is False
    assert rows["rec-file002.tif"]["unreadable"] is True           # the corrupt part
    assert c.seriesUnreadableCount == 1
    assert c.seriesFrameTotal == 7                                 # 3 + 0 + 4 (readable only)
    assert c.hasReadError is True                                  # blocks Start analysis
