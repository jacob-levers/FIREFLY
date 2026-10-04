"""Per-file ROIs survive closing FIREFLY.

The polygon store and the per-file override store (ROI settings, "analyse each
ROI separately", ROI names, a per-file detection threshold) lived only in
memory, so every ROI had to be redrawn after a restart or an update.  Both now
save to small JSON files in FIREFLY's per-user data folder, beside the theme
file.  Each entry remembers the recording's size and modification time: an ROI
is pixel coordinates on ONE recording, so a different file later found at the
same path does not inherit it.
"""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from firefly.ui.controllers.roi_store import RoiOverrideStore, RoiStore

POLY = [[(10.0, 10.0), (10.0, 30.0), (30.0, 30.0)]]
SPEC = {"roi_mode": "Manual polygon", "roi_auto_method": "Li", "roi_threshold": 0.1,
        "roi_mask_mode": "Max", "roi_bg_sigma": 25.0, "roi_split_replicates": True,
        "roi_labels": ["soma", "axon"], "wavelet_threshold": 150.0}


@pytest.fixture
def movie(tmp_path):
    p = tmp_path / "fly.czi"
    p.write_bytes(b"x" * 1000)
    return str(p)


def test_drawn_rois_come_back_after_a_restart(tmp_path, movie):
    path = str(tmp_path / "state" / "roi_polygons.json")
    RoiStore(path).set(movie, POLY)
    again = RoiStore(path)
    assert again.has(movie) and again.get(movie) == POLY


def test_per_file_settings_and_thresholds_come_back(tmp_path, movie):
    path = str(tmp_path / "roi_overrides.json")
    RoiOverrideStore(path).set(movie, SPEC)
    assert RoiOverrideStore(path).get(movie) == SPEC


def test_removing_an_roi_is_remembered_too(tmp_path, movie):
    path = str(tmp_path / "roi_polygons.json")
    s = RoiStore(path); s.set(movie, POLY); s.set(movie, [])
    assert not RoiStore(path).has(movie)
    o = RoiOverrideStore(str(tmp_path / "o.json")); o.set(movie, SPEC); o.clear(movie)
    assert RoiOverrideStore(str(tmp_path / "o.json")).get(movie) is None


def test_a_different_recording_at_the_same_path_does_not_inherit_it(tmp_path, movie):
    path = str(tmp_path / "roi_polygons.json")
    opath = str(tmp_path / "roi_overrides.json")
    RoiStore(path).set(movie, POLY)
    RoiOverrideStore(opath).set(movie, SPEC)
    with open(movie, "wb") as fh:                   # replaced by another export
        fh.write(b"y" * 2000)
    assert not RoiStore(path).has(movie) and RoiStore(path).get(movie) is None
    assert RoiOverrideStore(opath).get(movie) is None


def test_an_unplugged_drive_keeps_its_rois(tmp_path, movie):
    """A missing file cannot be checked — and nothing can run on it — so its
    ROI is kept for when the drive is back."""
    path = str(tmp_path / "roi_polygons.json")
    RoiStore(path).set(movie, POLY)
    os.rename(movie, movie + ".away")
    assert RoiStore(path).get(movie) == POLY
    os.rename(movie + ".away", movie)
    assert RoiStore(path).get(movie) == POLY


def test_an_unreadable_file_is_set_aside_not_overwritten(tmp_path, movie):
    path = tmp_path / "roi_polygons.json"
    path.write_text("{ not json", encoding="utf-8")
    s = RoiStore(str(path))
    assert not s.has(movie)
    s.set(movie, POLY)
    kept = [f for f in os.listdir(tmp_path) if f.startswith("roi_polygons.json.unreadable")]
    assert len(kept) == 1
    assert (tmp_path / kept[0]).read_text(encoding="utf-8") == "{ not json"
    assert RoiStore(str(path)).get(movie) == POLY


def test_saves_are_atomic_and_leave_no_temp_files(tmp_path, movie):
    path = tmp_path / "roi_polygons.json"
    s = RoiStore(str(path))
    for k in range(5):
        s.set(movie, [[(1.0 * k, 2.0), (3.0, 4.0), (5.0, 6.0)]])
    assert sorted(os.listdir(tmp_path)) == ["fly.czi", "roi_polygons.json"]
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1


def test_without_a_path_nothing_is_written(tmp_path, movie, monkeypatch):
    monkeypatch.chdir(tmp_path)
    RoiStore().set(movie, POLY)
    RoiOverrideStore().set(movie, SPEC)
    assert sorted(os.listdir(tmp_path)) == ["fly.czi"]


def test_tests_never_touch_the_real_data_folder(tmp_path_factory):
    from firefly.ui.ui_theme import app_data_dir
    base = os.path.realpath(str(tmp_path_factory.getbasetemp()))
    assert os.path.realpath(app_data_dir()).startswith(base)


# ── through the app ─────────────────────────────────────────────────────────
pytest.importorskip("PySide6")


def test_the_app_keeps_its_rois_in_the_data_folder():
    from firefly.ui.app_qml import roi_state_paths
    from firefly.ui.ui_theme import app_data_dir
    polys, overrides = roi_state_paths()
    assert os.path.dirname(polys) == os.path.dirname(overrides) == app_data_dir()


def test_an_roi_saved_in_one_session_is_used_in_the_next(tmp_path):
    import tifffile
    from PySide6 import QtWidgets
    from firefly.ui.controllers.batch_controller import BatchController
    from firefly.ui.controllers.params import params_builder
    from firefly.ui.controllers.roi_controller import RoiController
    from test_roi_per_file import _Settings
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    mov = str(tmp_path / "rec.tif")
    tifffile.imwrite(mov, np.random.default_rng(0).normal(100, 5, (3, 64, 64)).astype(np.uint16))
    pp, op = str(tmp_path / "p.json"), str(tmp_path / "o.json")
    s = _Settings({"analysis/roi_mode": "None", "analysis/diameter": 7, "analysis/bg_radius": 10,
                   "analysis/backend": "Wavelet — palmTRACER-style (CPU)"})

    c = RoiController(store=RoiStore(pp), settings=s, override_store=RoiOverrideStore(op))
    c.setBatchMode(True)
    c.editFile(mov); c.roiMode = "Manual polygon"
    for y, x in ((10, 10), (10, 40), (40, 40)):
        c.addVertex(float(y), float(x))
    c.closeDraft(); c.commit()
    c.editDetection(mov); c.setThresholdPerFile(True); c.commitThreshold(150.0); c.commit()
    drawn = c.getPolygons()
    c.dispose(); c.deleteLater(); app.processEvents()

    # — FIREFLY restarts —
    store, ovr = RoiStore(pp), RoiOverrideStore(op)
    c2 = RoiController(store=store, settings=s, override_store=ovr)
    c2.setBatchMode(True)
    c2.editFile(mov)
    assert c2.getPolygons() == drawn
    c2.cancel()
    q = BatchController.__new__(BatchController)
    q._override_store, q._roi_store = ovr, store
    assert q._roi_label({"primary": mov}) == "Polygon"

    class _Import:
        filePath, outDir, isCsv = mov, None, False
        overridePx = overrideFi = False
        pixelSize, frameInterval = 0.106, 0.02
    p = params_builder.build_params(s, _Import(), fpath=mov, roi_store=store, override_store=ovr)
    assert p["roi_polygon"] and p["wavelet_threshold"] == 150.0
    c2.dispose(); c2.deleteLater(); app.processEvents()
