"""A wavelet threshold set for ONE file in a batch — the same path as per-file
minmass, for the palmTRACER-style detector.

palmTRACER itself was run with a threshold per cell (150–380 on the PC12 data),
so reproducing such an analysis needs this; FIREFLY records it per run and the
comparison report warns when conditions end up on different thresholds.
"""
import os

import numpy as np
import pytest

from firefly.ui.controllers.params.params_builder import build_params
from firefly.ui.controllers.roi_store import RoiOverrideStore, RoiStore

PT_LABEL = "Wavelet — palmTRACER-style (CPU)"


class _Settings(dict):
    def get_str(self, k, d=""): return dict.get(self, k, d)
    def get_float(self, k, d=0.0): return float(dict.get(self, k, d))
    def get_bool(self, k, d=False): return bool(dict.get(self, k, d))
    def get(self, k, d=None): return dict.get(self, k, d)
    def set(self, k, v): self[k] = v
    def sync(self): pass


class _Import:
    outDir = "/tmp/out"; isCsv = False; pixelSize = 0.1; frameInterval = 0.02
    overridePx = True; overrideFi = True; filePath = "/tmp/a.tif"


def _spec(**extra):
    base = {"roi_mode": "None", "roi_auto_method": "Li", "roi_threshold": 0.1,
            "roi_mask_mode": "Max", "roi_bg_sigma": 25.0}
    base.update(extra)
    return base


def _g(**extra):
    d = {"analysis/backend": PT_LABEL, "analysis/wavelet_threshold": 250.0, "analysis/minmass": 0.45}
    d.update(extra)
    return _Settings(d)


# ── params ──────────────────────────────────────────────────────────────────
def test_a_file_without_an_override_uses_the_sidebar_threshold():
    p = build_params(_g(), _Import(), fpath="/tmp/a.tif", roi_store=RoiStore(), override_store=RoiOverrideStore())
    assert p["wavelet_threshold"] == 250.0
    assert p["wavelet_threshold_per_file"] is False


def test_batch_gives_each_file_its_own_wavelet_threshold():
    ovr = RoiOverrideStore()
    ovr.set("/data/cell1.tif", _spec(wavelet_threshold=150.0))
    ovr.set("/data/cell3.tif", _spec(wavelet_threshold=380.0))
    got = {f: build_params(_g(), _Import(), fpath=f, out_dir="/tmp/out", roi_store=RoiStore(),
                           override_store=ovr)
           for f in ("/data/cell1.tif", "/data/cell3.tif", "/data/plain.tif")}
    assert {f: p["wavelet_threshold"] for f, p in got.items()} == {
        "/data/cell1.tif": 150.0, "/data/cell3.tif": 380.0, "/data/plain.tif": 250.0}
    assert got["/data/cell1.tif"]["wavelet_threshold_per_file"] is True


def test_a_wavelet_override_does_not_touch_minmass():
    ovr = RoiOverrideStore()
    ovr.set("/data/x.tif", _spec(wavelet_threshold=300.0))
    p = build_params(_g(), _Import(), fpath="/data/x.tif", roi_store=RoiStore(), override_store=ovr)
    assert p["minmass"] == pytest.approx(0.45) and p["minmass_per_file"] is False


# ── the ROI viewer ──────────────────────────────────────────────────────────
@pytest.fixture
def movie(tmp_path):
    import tifffile
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:48, 0:48]
    frames = []
    for _ in range(3):
        img = np.full((48, 48), 1000.0) + rng.normal(0, 15, (48, 48))
        for cy, cx, a in ((14, 15, 2000), (30, 33, 1500)):
            img += a * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.2 ** 2))
        frames.append(img)
    path = tmp_path / "m.tif"
    tifffile.imwrite(path, np.stack(frames).astype(np.uint16), photometric="minisblack")
    return str(path)


@pytest.fixture
def viewer(movie):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from firefly.ui.controllers.roi_controller import RoiController
    app = QApplication.instance() or QApplication([])
    s, ovr = _g(), RoiOverrideStore()
    c = RoiController(store=RoiStore(), settings=s, override_store=ovr)
    c.editDetection(movie)
    yield c, s, ovr, movie
    c.deleteLater(); app.processEvents()


def test_the_panel_drives_the_wavelet_threshold(viewer):
    c, s, ovr, movie = viewer
    assert c.waveletThreshold == pytest.approx(250.0)
    c.waveletThreshold = 320.0
    assert s["analysis/wavelet_threshold"] == pytest.approx(320.0), "off per-file: the shared value moves"


def test_per_file_leaves_the_shared_threshold_alone_and_is_saved(viewer):
    c, s, ovr, movie = viewer
    c.setThresholdPerFile(True)
    assert c.thresholdPerFile is True
    c.waveletThreshold = 150.0
    assert s["analysis/wavelet_threshold"] == pytest.approx(250.0)
    c.commit()
    assert ovr.get(movie)["wavelet_threshold"] == pytest.approx(150.0)


def test_reopening_restores_the_files_own_wavelet_threshold(viewer):
    c, s, ovr, movie = viewer
    c.setThresholdPerFile(True); c.waveletThreshold = 150.0; c.commit()
    s["analysis/wavelet_threshold"] = 400.0
    c.editDetection(movie)                               # the REAL open path
    assert c.thresholdPerFile is True
    assert c.waveletThreshold == pytest.approx(150.0)
    c.commit()
    assert ovr.get(movie)["wavelet_threshold"] == pytest.approx(150.0), "saving again must not overwrite it"


def test_turning_per_file_off_drops_the_files_threshold(viewer):
    c, s, ovr, movie = viewer
    c.setThresholdPerFile(True); c.waveletThreshold = 150.0; c.commit()
    c.editDetection(movie)
    c.setThresholdPerFile(False)
    assert "wavelet_threshold" not in (ovr.get(movie) or {})


def test_the_preview_uses_the_files_threshold(viewer):
    from firefly.analysis.fa_localize_backends import PalmTracerWaveletBackend
    import tifffile
    c, s, ovr, movie = viewer
    frame = tifffile.imread(movie)[0].astype(np.float32)
    count = lambda T: len(PalmTracerWaveletBackend().localise(frame[None], wavelet_threshold=T, quiet=True))
    T1 = next(T for T in np.arange(250.0, 1000.0, 5.0) if count(T) == 1)
    c.setThresholdPerFile(True)
    c.waveletThreshold = float(T1)
    c.refreshSpots()
    assert c.spotCount == 1, c.spotSummary


# ── the same reopen path for per-file MINMASS (a real bug, found here) ──────
def test_reopening_restores_the_files_own_minmass(movie):
    """The open path re-read the SIDEBAR minmass after applying the file's saved
    override, so a reopened file previewed with the wrong threshold — and the
    next Save quietly replaced its own value with the sidebar's."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from firefly.ui.controllers.roi_controller import RoiController
    app = QApplication.instance() or QApplication([])
    s = _Settings({"analysis/backend": "Auto", "analysis/minmass": 0.45})
    ovr = RoiOverrideStore()
    c = RoiController(store=RoiStore(), settings=s, override_store=ovr)
    c.editDetection(movie)
    c.setMinmassPerFile(True); c.detectMinmass = 0.77; c.commit()
    c.editDetection(movie)
    assert c.detectMinmass == pytest.approx(0.77)
    c.commit()
    assert ovr.get(movie)["minmass"] == pytest.approx(0.77)
    c.deleteLater(); app.processEvents()


def test_a_sidebar_minmass_change_leaves_a_per_file_minmass_alone(movie):
    """The settings hook copied every sidebar minmass change into the open
    file's threshold, per-file or not — the wavelet branch beside it already
    checked.  The per-file value then went to the next Save."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from firefly.ui.controllers.roi_controller import RoiController
    app = QApplication.instance() or QApplication([])
    s = _Settings({"analysis/backend": "Auto", "analysis/minmass": 0.45})
    ovr = RoiOverrideStore()
    c = RoiController(store=RoiStore(), settings=s, override_store=ovr)
    c.editDetection(movie)
    c.setMinmassPerFile(True); c.detectMinmass = 0.77
    s["analysis/minmass"] = 0.30
    c._spot_settings_changed("analysis/minmass")      # what settings.changed delivers
    assert c.detectMinmass == pytest.approx(0.77)
    c.commit()
    assert ovr.get(movie)["minmass"] == pytest.approx(0.77)
    c.deleteLater(); app.processEvents()


def test_a_sidebar_minmass_change_still_reaches_a_shared_threshold(movie):
    """The other half: off per-file, the panel shows the sidebar's value."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from firefly.ui.controllers.roi_controller import RoiController
    app = QApplication.instance() or QApplication([])
    s = _Settings({"analysis/backend": "Auto", "analysis/minmass": 0.45})
    c = RoiController(store=RoiStore(), settings=s, override_store=RoiOverrideStore())
    c.editDetection(movie)
    s["analysis/minmass"] = 0.30
    c._spot_settings_changed("analysis/minmass")
    assert c.detectMinmass == pytest.approx(0.30)
    c.cancel(); c.deleteLater(); app.processEvents()
