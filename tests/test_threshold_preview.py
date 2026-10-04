"""Manual detection thresholding: exact, live and responsive.

The Preview & ROI threshold slider lagged.  Every step wrote the shared
setting, and every write made the parameter sidebar re-evaluate all of its
fields; the overlay was blanked on each step and only re-detected after a
200 ms pause, on the GUI thread.  Now a drag only moves the preview's own
value — the setting is written once, on release — detection runs off the GUI
thread (newest request wins), the previous overlay stays up until the new one
arrives, and each (frame, threshold) detection is cached.

The preview must also find exactly what a run finds at that threshold: it
calls the production localiser, and this is checked spot for spot.
"""
import os
import time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                            # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

PT_LABEL = "Wavelet — palmTRACER-style (CPU)"


class _Settings(dict):
    """In-memory settings that count writes (never the real preferences)."""
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.writes = []
    def get_str(self, k, d=""): return dict.get(self, k, d)
    def get_float(self, k, d=0.0): return float(dict.get(self, k, d))
    def get_bool(self, k, d=False): return bool(dict.get(self, k, d))
    def get(self, k, d=None): return dict.get(self, k, d)
    def set(self, k, v): self.writes.append(k); self[k] = v
    def sync(self): pass


def _frames(n=6, seed=1, size=64):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    out = []
    for f in range(n):
        img = np.full((size, size), 1000.0) + rng.normal(0, 15, (size, size))
        for _ in range(6):
            cy, cx = rng.uniform(6, size - 6, 2)
            img += rng.uniform(300, 2500) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.3 ** 2))
        out.append(img)
    return np.stack(out).astype(np.float32)


@pytest.mark.parametrize("backend,kw", [("palmtracer", {"wavelet_threshold": 120.0}),
                                        ("trackpy", {})])
def test_the_preview_finds_exactly_what_the_run_finds(backend, kw):
    from firefly.analysis.fa_detection_preview import preview_detections
    from firefly.analysis.fa_localize import preprocess_and_localise_adaptive
    stack = _frames()
    minmass = 0.0 if backend == "palmtracer" else 0.45
    locs, *_ = preprocess_and_localise_adaptive(stack, diameter=7, minmass=minmass, bg_radius=10,
                                                bg_method="uniform_filter", workers=1,
                                                chunk_size=500, backend=backend, **kw)
    assert len(locs)
    for k, frame in enumerate(stack):
        rows, _ = preview_detections(frame, diameter=7, minmass=minmass, bg_radius=10,
                                     bg_method="uniform_filter", backend=backend,
                                     roi_mask=None, roi_known=True, **kw)
        run = np.round(locs.loc[locs.frame == k, ["x", "y"]].to_numpy(), 4)
        prev = np.round(rows[["x", "y"]].to_numpy(), 4)
        assert sorted(map(tuple, run)) == sorted(map(tuple, prev)), k


@pytest.fixture
def movie(tmp_path):
    import tifffile
    path = tmp_path / "m.tif"
    tifffile.imwrite(path, _frames(n=3).astype(np.uint16), photometric="minisblack")
    return str(path)


@pytest.fixture
def viewer(movie):
    from firefly.ui.controllers.roi_controller import RoiController
    from firefly.ui.controllers.roi_store import RoiOverrideStore, RoiStore
    s = _Settings({"analysis/backend": PT_LABEL, "analysis/wavelet_threshold": 100.0,
                   "analysis/diameter": 7, "analysis/bg_radius": 10, "analysis/roi_mode": "None"})
    c = RoiController(store=RoiStore(), settings=s, override_store=RoiOverrideStore())
    c.editDetection(movie)
    s.writes.clear()
    yield c, s, movie
    c.dispose()
    c.deleteLater(); _app.processEvents()


def _expected(movie, T):
    import tifffile
    from firefly.analysis.fa_detection_preview import preview_detections
    frame = tifffile.imread(movie)[0].astype(np.float32)
    rows, summary = preview_detections(frame, diameter=7, minmass=0.0, bg_radius=10,
                                       backend="palmtracer", roi_mask=None, roi_known=True,
                                       wavelet_threshold=T)
    return summary["passed"]


def _settle(c, timeout=10.0):
    deadline = time.monotonic() + timeout
    while c.spotsUpdating and time.monotonic() < deadline:
        _app.processEvents(); time.sleep(0.005)
    _app.processEvents()


def test_dragging_writes_no_setting_until_release(viewer):
    c, s, _m = viewer
    for v in (110, 130, 150, 170, 190):
        c.previewThreshold(float(v))
    _settle(c)
    assert s.writes == [] and s["analysis/wavelet_threshold"] == 100.0
    assert c.waveletThreshold == 190.0                  # the panel shows where it is
    c.commitThreshold(190.0)
    assert s.writes == ["analysis/wavelet_threshold"] and s["analysis/wavelet_threshold"] == 190.0


def test_the_overlay_follows_the_slider_without_blanking(viewer):
    c, _s, movie = viewer
    assert c.hasSpots and not c.spotsStale
    T = 400.0
    c.previewThreshold(T)
    assert c.hasSpots and not c.spotsStale               # the last overlay stays up meanwhile
    _settle(c)
    assert c.spotCount == _expected(movie, T), c.spotSummary


def test_revisiting_a_threshold_reuses_its_detection(viewer, monkeypatch):
    c, _s, movie = viewer
    from firefly.analysis import fa_detection_preview as fdp
    calls = []
    real = fdp.preview_detections
    monkeypatch.setattr(fdp, "preview_detections", lambda *a, **k: (calls.append(k.get("wavelet_threshold")), real(*a, **k))[1])
    for T in (300.0, 500.0):
        c.previewThreshold(T); _settle(c)
    n = len(calls)
    c.previewThreshold(300.0)
    shown, updating = c.spotCount, c.spotsUpdating
    assert len(calls) == n                               # no new detection …
    assert not updating and shown == _expected(movie, 300.0)   # … and shown at once


def test_release_shows_the_final_threshold_at_once(viewer):
    c, s, movie = viewer
    c.previewThreshold(250.0)
    c.commitThreshold(450.0)
    assert not c.spotsStale and c.hasSpots
    assert c.spotCount == _expected(movie, 450.0)
    _settle(c)
    assert c.spotCount == _expected(movie, 450.0)        # a late drag result cannot overwrite it


def test_the_panel_uses_the_new_slots():
    qml = open(os.path.join(os.path.dirname(__file__), "..", "firefly", "ui", "qml",
                            "RoiOverlay.qml"), encoding="utf-8").read()
    assert "Roi.previewThreshold(" in qml and "Roi.commitThreshold(" in qml


# ── what the threshold excludes, in red ─────────────────────────────────────
def _detect(movie, T):
    import tifffile
    from firefly.analysis.fa_detection_preview import preview_detections
    frame = tifffile.imread(movie)[0].astype(np.float32)
    rows, _ = preview_detections(frame, diameter=7, minmass=0.0, bg_radius=10,
                                 backend="palmtracer", roi_mask=None, roi_known=True,
                                 wavelet_threshold=T)
    return rows


def test_the_overlay_marks_what_the_threshold_excludes(viewer):
    """Red = found at ¾ of the threshold, not at the threshold; green stays
    exactly the run's detection."""
    from firefly.analysis.fa_detection_preview import below_threshold
    from firefly.ui.controllers.roi_controller import _EXCLUDED_FRACTION
    c, _s, movie = viewer
    T = 400.0
    c.previewThreshold(T); _settle(c)
    shown = c._spot_rows
    red = shown[shown.decision == "below_threshold"]
    green = shown[shown.decision != "below_threshold"]
    want = below_threshold(_detect(movie, T), _detect(movie, T * _EXCLUDED_FRACTION), radius=3.5)
    assert len(red) == len(want) > 0
    assert sorted(map(tuple, np.round(red[["x", "y"]].to_numpy(), 4))) == \
        sorted(map(tuple, np.round(want[["x", "y"]].to_numpy(), 4)))
    assert len(green) == c.spotCount == _expected(movie, T)
    assert "excluded by the threshold" in c.spotSummary
    img = c.roi_spots_image()
    r = red.iloc[0]
    ring = [img.pixelColor(int(round(r.x + dx)), int(round(r.y + dy)))
            for dx, dy in ((4, 0), (-4, 0), (0, 4), (0, -4))]
    assert any(p.red() > 150 and p.green() < 100 for p in ring)


def test_clicking_a_red_spot_says_why_it_is_excluded(viewer):
    c, _s, _m = viewer
    c.previewThreshold(400.0); _settle(c)
    red = c._spot_rows[c._spot_rows.decision == "below_threshold"].iloc[0]
    c.inspectSpot(red.y, red.x)
    assert "excluded by the threshold" in c.spotInspection


def test_an_roi_edit_relabels_without_losing_the_red_spots(viewer):
    c, _s, _m = viewer
    c.previewThreshold(400.0); _settle(c)
    n_red = int((c._spot_rows.decision == "below_threshold").sum())
    c._reclassify_spots()
    assert int((c._spot_rows.decision == "below_threshold").sum()) == n_red


# ── "Set to recommended" ────────────────────────────────────────────────────
def _until_recommended(c, timeout=60.0):
    deadline = time.monotonic() + timeout
    while c.recommending and time.monotonic() < deadline:
        _app.processEvents(); time.sleep(0.01)
    _app.processEvents()
    _settle(c)


def test_recommended_wavelet_threshold_is_4_4_noise_and_committed(viewer):
    from firefly.analysis.fa_localize import recommend_wavelet_threshold
    from firefly.ui.controllers.params.preview_loader import DetectionFrames
    c, s, movie = viewer
    want, _info = recommend_wavelet_threshold(DetectionFrames(movie, 3))
    c.recommendThreshold()
    assert c.recommending
    _until_recommended(c)
    assert c.waveletThreshold == want
    assert s["analysis/wavelet_threshold"] == want           # committed, like a typed value
    assert "4.4 ×" in c.recommendNote and not c.spotsStale


def test_recommended_minmass_is_the_runs_auto_threshold(movie, monkeypatch):
    from firefly.analysis import fa_localize
    from firefly.ui.controllers.roi_controller import RoiController
    from firefly.ui.controllers.roi_store import RoiOverrideStore, RoiStore
    seen = {}
    def fake(stack, p, log_cb=None):
        seen.update(n=len(stack), backend=p["backend"], diameter=p["diameter"],
                    sensitivity=p["minmass_sensitivity"])
        return 0.61, {"method": "linkability"}
    monkeypatch.setattr(fa_localize, "estimate_minmass_for_run", fake)
    s = _Settings({"analysis/backend": "Crocker–Grier — Trackpy (CPU)", "analysis/minmass": 0.45,
                   "analysis/diameter": 7, "analysis/bg_radius": 10, "analysis/roi_mode": "None",
                   "analysis/minmass_sensitivity": "Strict"})
    c = RoiController(store=RoiStore(), settings=s, override_store=RoiOverrideStore())
    try:
        c.editDetection(movie)
        c.recommendThreshold(); _until_recommended(c)
        assert seen == dict(n=3, backend="trackpy", diameter=7, sensitivity="strict")
        assert c.detectMinmass == 0.61 and s["analysis/minmass"] == 0.61
        assert "Auto-threshold run" in c.recommendNote
    finally:
        c.dispose(); c.deleteLater(); _app.processEvents()


def test_a_recommendation_for_a_file_no_longer_open_is_dropped(viewer):
    c, _s, movie = viewer
    before = c.waveletThreshold
    c._on_recommended(dict(file=movie + ".other", wavelet=True), (999.0, "x"))
    assert c.waveletThreshold == before and c.recommendNote == ""


def test_the_panel_offers_the_recommendation():
    qml = open(os.path.join(os.path.dirname(__file__), "..", "firefly", "ui", "qml",
                            "RoiOverlay.qml"), encoding="utf-8").read()
    assert "Roi.recommendThreshold()" in qml and "Set to recommended" in qml
    assert "Roi.recommendNote" in qml
