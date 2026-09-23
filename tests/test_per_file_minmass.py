"""A detection threshold set for ONE file in a batch.

Batch runs previously read a single global `analysis/minmass` for every file.
Per-file thresholds are genuinely useful — mass is file-relative, because every
frame is min-max normalised, so one number is not the same brightness in two
recordings — and genuinely dangerous, because tuning each file until the counts
agree is how a detection difference gets manufactured. So: opt-in per file,
recorded in the run's params, and flagged by Compare when a group mixes values.
"""
import numpy as np
import pytest

from firefly.ui.controllers.params.params_builder import build_params
from firefly.ui.controllers.roi_store import RoiOverrideStore, RoiStore


class _Settings(dict):
    def get_str(self, k, d=""): return dict.get(self, k, d)
    def get_float(self, k, d=0.0): return float(dict.get(self, k, d))
    def get_bool(self, k, d=False): return bool(dict.get(self, k, d))
    def get(self, k, d=None): return dict.get(self, k, d)
    def set(self, k, v): self[k] = v


class _Import:
    outDir = "/tmp/out"; isCsv = False
    pixelSize = 0.1; frameInterval = 0.02
    overridePx = True; overrideFi = True
    filePath = "/tmp/a.czi"


def _spec(**extra):
    base = {"roi_mode": "None", "roi_auto_method": "Li", "roi_threshold": 0.1,
            "roi_mask_mode": "Max", "roi_bg_sigma": 25.0}
    base.update(extra)
    return base


def _settings():
    return _Settings({"analysis/minmass": 0.45, "analysis/auto_minmass": False})


# ── the params contract ──────────────────────────────────────────────────────
def test_a_file_without_an_override_uses_the_shared_threshold():
    p = build_params(_settings(), _Import(), fpath="/tmp/a.czi",
                     override_store=RoiOverrideStore())
    assert p["minmass"] == 0.45
    assert p["minmass_per_file"] is False


def test_a_per_file_threshold_replaces_the_shared_one_for_that_file_only():
    ovr = RoiOverrideStore()
    ovr.set("/tmp/a.czi", _spec(minmass=0.92, auto_minmass=False))
    g = _settings()

    a = build_params(g, _Import(), fpath="/tmp/a.czi", override_store=ovr)
    b = build_params(g, _Import(), fpath="/tmp/b.czi", override_store=ovr)

    assert a["minmass"] == 0.92 and a["minmass_per_file"] is True
    assert b["minmass"] == 0.45 and b["minmass_per_file"] is False, (
        "a per-file threshold leaked to another file in the batch")


def test_a_per_file_threshold_turns_off_auto_for_that_file():
    """A manual value and auto-picking are mutually exclusive: leaving auto on
    would silently discard the number the user set."""
    ovr = RoiOverrideStore()
    ovr.set("/tmp/a.czi", _spec(minmass=0.7, auto_minmass=False))
    g = _Settings({"analysis/minmass": 0.45, "analysis/auto_minmass": True})

    a = build_params(g, _Import(), fpath="/tmp/a.czi", override_store=ovr)
    b = build_params(g, _Import(), fpath="/tmp/b.czi", override_store=ovr)
    assert a["auto_minmass"] is False and a["minmass"] == 0.7
    assert b["auto_minmass"] is True, "other files keep auto"


def test_an_roi_only_override_does_not_touch_the_threshold():
    """Most files carry an ROI override and nothing else; they must keep
    inheriting the shared threshold."""
    ovr = RoiOverrideStore()
    ovr.set("/tmp/a.czi", _spec(roi_mode="Manual polygon"))
    p = build_params(_settings(), _Import(), fpath="/tmp/a.czi", override_store=ovr)
    assert p["minmass"] == 0.45 and p["minmass_per_file"] is False


# ── the batch path ───────────────────────────────────────────────────────────
def test_batch_gives_each_file_its_own_threshold():
    """What the feature is for: one queue, different thresholds per recording."""
    ovr = RoiOverrideStore()
    ovr.set("/data/dense.czi", _spec(minmass=1.2, auto_minmass=False))
    ovr.set("/data/dim.czi", _spec(minmass=0.2, auto_minmass=False))
    g = _settings()

    got = {f: build_params(g, _Import(), fpath=f, out_dir="/tmp/out",
                           roi_store=RoiStore(), override_store=ovr)["minmass"]
           for f in ("/data/dense.czi", "/data/dim.czi", "/data/plain.czi")}
    assert got == {"/data/dense.czi": 1.2, "/data/dim.czi": 0.2,
                   "/data/plain.czi": 0.45}


# ── the controller switch ────────────────────────────────────────────────────
@pytest.fixture
def ctrl():
    pytest.importorskip("PySide6")
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from firefly.ui.controllers.roi_controller import RoiController
    app = QApplication.instance() or QApplication([])
    ovr = RoiOverrideStore()
    c = RoiController(store=RoiStore(), settings=_settings(), override_store=ovr)
    c._file = "/tmp/a.czi"; c._img_h = c._img_w = 64
    yield c, ovr
    c.deleteLater(); app.processEvents()


def test_off_by_default_the_slider_still_sets_the_shared_threshold(ctrl):
    c, ovr = ctrl
    assert c.minmassPerFile is False
    c.detectMinmass = 0.8
    assert c._s["analysis/minmass"] == 0.8, "shared threshold not updated"


def test_on_the_slider_leaves_every_other_file_alone(ctrl):
    """The whole point: setting one recording's threshold must not move the
    value the rest of the queue runs at."""
    c, ovr = ctrl
    c.setMinmassPerFile(True)
    c.detectMinmass = 0.92
    assert c._s["analysis/minmass"] == 0.45, (
        "per-file mode wrote the shared sidebar threshold")

    c.commit()
    assert ovr.get("/tmp/a.czi")["minmass"] == 0.92
    p = build_params(c._s, _Import(), fpath="/tmp/other.czi", override_store=ovr)
    assert p["minmass"] == 0.45


def test_turning_it_off_drops_the_files_threshold(ctrl):
    c, ovr = ctrl
    c.setMinmassPerFile(True)
    c.detectMinmass = 0.92
    c.commit()
    assert "minmass" in ovr.get("/tmp/a.czi")

    c.setMinmassPerFile(False)
    c.commit()
    saved = ovr.get("/tmp/a.czi") or {}
    assert "minmass" not in saved, "the file still carries its own threshold"


def test_reopening_a_file_shows_the_threshold_it_carries(ctrl):
    c, ovr = ctrl
    c.setMinmassPerFile(True)
    c.detectMinmass = 0.77
    c.commit()

    c._minmass_per_file = False          # simulate a fresh open
    c._apply_spec(ovr.get("/tmp/a.czi"))
    assert c.minmassPerFile is True
    assert c.detectMinmass == pytest.approx(0.77)
