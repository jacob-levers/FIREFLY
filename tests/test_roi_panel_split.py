"""Drawing an ROI and choosing a detection threshold are two separate jobs.

Both used to live in one modal reached by one button, so picking a threshold
meant opening "Preview & ROI" and scrolling past the polygon tools, and drawing
a region meant scrolling past the mass histogram.  They are now two panels of
the same viewer, each with its own entry point: `editFile` opens the ROI panel,
`editDetection` opens the threshold panel with the spot overlay already on
(seeing the dots move is the entire point of that screen).

The viewer is one component, so the split is enforced by `Roi.panel` alone —
which is why these tests pin both the property and what each panel actually
renders.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                            # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Settings(dict):
    def get_str(self, k, d=""): return dict.get(self, k, d)
    def get_float(self, k, d=0.0): return float(dict.get(self, k, d))
    def get_bool(self, k, d=False): return bool(dict.get(self, k, d))
    def get(self, k, d=None): return dict.get(self, k, d)
    def set(self, k, v): self[k] = v
    def sync(self): pass


@pytest.fixture
def movie(tmp_path):
    """A 3-frame recording with a few bright spots, so detection has something
    to find and the projection has something to show."""
    import tifffile
    rng = np.random.default_rng(5)
    stack = rng.normal(100, 2, (3, 64, 64)).astype("float32")
    for y, x in ((16, 20), (40, 44), (30, 12)):
        stack[:, y - 1:y + 2, x - 1:x + 2] += 900.0
    path = tmp_path / "cell.tif"
    tifffile.imwrite(path, stack, photometric="minisblack")
    return str(path)


@pytest.fixture
def roi(tmp_path):
    from firefly.ui.controllers.roi_controller import RoiController
    from firefly.ui.controllers.roi_store import RoiOverrideStore, RoiStore
    c = RoiController(store=RoiStore(), settings=_Settings({"analysis/minmass": 0.45}),
                      override_store=RoiOverrideStore())
    yield c
    c.deleteLater()


# ── the property that drives the split ───────────────────────────────────────
def test_the_viewer_opens_on_the_roi_panel_by_default(roi, movie):
    roi.editFile(movie)
    assert roi.editing
    assert roi.panel == "roi"


def test_the_threshold_entry_point_opens_the_detection_panel(roi, movie):
    roi.editDetection(movie)
    assert roi.editing
    assert roi.panel == "detect"
    assert roi.fileName == "cell.tif", "the same file is loaded either way"


def test_the_detection_panel_turns_the_spot_overlay_on(roi, movie):
    """Its whole job is watching the dots respond to the slider; arriving with
    the overlay off puts a click between the user and the evidence."""
    roi.editFile(movie)
    assert not roi.detectEnabled
    roi.cancel()
    roi.editDetection(movie)
    assert roi.detectEnabled


def test_switching_panels_notifies_qml(roi, movie):
    seen = []
    roi.panelChanged.connect(lambda: seen.append(roi.panel))
    roi.editFile(movie)
    roi.cancel()
    roi.editDetection(movie)
    assert "detect" in seen, "the binding would never update without the signal"


def test_reopening_for_an_roi_leaves_the_detection_panel_behind(roi, movie):
    """Panel state must not leak between opens — the previous session's panel
    surviving is how a button silently stops doing what it says."""
    roi.editDetection(movie)
    roi.cancel()
    roi.editFile(movie)
    assert roi.panel == "roi"


def test_a_run_scoped_edit_is_always_the_roi_panel(roi, tmp_path, movie):
    """`editRun` shrinks a FINISHED run's region; its detections are already
    written, so a threshold panel there would offer a control that does nothing.
    """
    import json
    run = tmp_path / "run_001"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps(
        {"input_file": movie, "params": {"width": 64, "height": 64}}))
    roi.editDetection(movie)
    roi.cancel()
    assert roi.editRun(str(run)) in (True, False)      # background may fall back
    if roi.editing:
        assert roi.panel == "roi"
        assert not roi.detectEnabled


# ── the detection panel must be able to SHOW detections ─────────────────────
def test_the_detection_panel_opens_on_raw_frames(roi, movie):
    """Detection runs on acquired frames, so `_recompute_spots` refuses to draw
    anything over a max projection — it posts "Select Raw frames" instead.  The
    ROI panel opens on the projection (the right canvas for tracing a region);
    this one must not, or it opens with the overlay on, no dots, and a nag.
    """
    roi.editDetection(movie)
    assert roi.viewMode == "raw"
    assert roi.detectEnabled
    assert "Select Raw frames" not in roi.spotSummary


def test_the_roi_panel_still_opens_on_the_projection(roi, movie):
    """Tracing a neuron wants the projection: every frame's signal at once."""
    roi.editFile(movie)
    assert roi.viewMode == "proj"


def test_a_single_frame_file_stays_on_the_projection(roi, tmp_path):
    """There is no frame to scrub to, so asking for "raw" would offer a view the
    toggle does not even list."""
    import tifffile
    frame = np.random.default_rng(9).normal(100, 2, (48, 48)).astype("float32")
    path = tmp_path / "one.tif"
    tifffile.imwrite(path, frame, photometric="minisblack")
    roi.editDetection(str(path))
    assert roi.nFrames <= 1
    assert roi.viewMode == "proj"


# ── the panel must not block the UI ─────────────────────────────────────────
def test_the_noise_floor_is_computed_once_per_recording(roi, movie, monkeypatch):
    """`massProfile` is called on every slider move and every toggle, and it
    re-ran `estimate_noise_floor` each time — three GaussianMixture fits, ~15 ms
    on the GUI thread, measured as ~100% of the call's cost.  The floor depends
    only on the harvested candidates, not on the threshold, so it belongs in the
    same cache as the candidates.
    """
    from firefly.analysis import fa_localize

    calls = []
    real = fa_localize.estimate_noise_floor

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(fa_localize, "estimate_noise_floor", counted)

    roi.editDetection(movie)
    for _ in range(6):                      # as dragging the slider does
        roi.massProfile()
    assert len(calls) <= 1, (
        f"the noise floor was refitted {len(calls)} times for one recording")


def test_reprofiling_recomputes_the_floor(roi, movie, monkeypatch):
    """Caching must not outlive the harvest it describes: asking for a fresh
    profile has to refit, or the marker would describe discarded candidates."""
    from firefly.analysis import fa_localize

    calls = []
    real = fa_localize.estimate_noise_floor
    monkeypatch.setattr(fa_localize, "estimate_noise_floor",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    roi.editDetection(movie)
    roi.massProfile()
    roi.invalidateMassProfile()
    roi.massProfile()
    assert len(calls) == 2
