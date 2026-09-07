"""Adding / editing an ROI on a run that has already been analysed.

An ROI used to have to be decided before the analysis, over the raw movie, in
the Import tab — the worst possible moment, before a single track exists.  The
engine has always been able to do it afterwards (``firefly_worker.run_postproc``)
but nothing called it, and the editor had no way to alter a shape once drawn.

The dangerous part is not the wiring, it is that a post-hoc ROI can only ever
SHRINK the previous one.  A run's ``_localisations.csv`` is written AFTER its ROI
was applied, so it holds only what that ROI kept.  Draw a larger region and the
result covers the overlap while the app reports the shape you drew.  Most of what
follows is about refusing to do that.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from firefly.ui.controllers.postproc_controller import PostprocController
from firefly.ui.controllers.roi_controller import RoiController

_app = QApplication.instance() or QApplication([])

SQUARE_50_150 = [[50.0, 50.0], [50.0, 150.0], [150.0, 150.0], [150.0, 50.0]]
INNER = [[70.0, 70.0], [70.0, 130.0], [130.0, 130.0], [130.0, 70.0]]
OUTER = [[0.0, 0.0], [0.0, 250.0], [250.0, 250.0], [250.0, 0.0]]


def make_run(tmp_path, name="run", *, had_roi=False, movie=None,
             width=256, height=256, seed=0):
    """A finished-run folder: saved localisations + params, optionally recording
    the ROI that produced it and the movie it came from."""
    run = tmp_path / name
    extras = run / "firefly_extras"
    extras.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    # localisations confined to 50..150 — as if an ROI had kept only these
    pd.DataFrame({"x": rng.uniform(50, 150, 2000),
                  "y": rng.uniform(50, 150, 2000),
                  "frame": rng.integers(0, 100, 2000),
                  "mass": rng.uniform(1, 5, 2000)}).to_csv(
        extras / "rec_localisations.csv", index=False)
    params = {"width": width, "height": height,
              "pixel_size_um": 0.1, "frame_interval_s": 0.02}
    if had_roi:
        params["roi_polygon"] = [SQUARE_50_150]
    (extras / "rec_params.json").write_text(json.dumps(params), encoding="utf-8")
    if movie is not None:
        (run / "rec_run_manifest.json").write_text(
            json.dumps({"input": {"path": str(movie)}}), encoding="utf-8")
    return str(run)


# ── the guard: a post-hoc ROI may only shrink ────────────────────────────────
def test_a_region_inside_the_original_is_allowed(tmp_path):
    c = PostprocController()
    run = make_run(tmp_path, had_roi=True)
    assert c.canApply(run, [INNER])["ok"] is True


def test_a_region_reaching_outside_the_original_is_refused(tmp_path):
    """The whole point.  Those localisations are not in the run's output, so
    re-processing cannot recover them — it would quietly return the overlap."""
    c = PostprocController()
    run = make_run(tmp_path, had_roi=True)
    verdict = c.canApply(run, [OUTER])
    assert verdict["ok"] is False
    assert "already had an ROI" in verdict["reason"]
    assert "original movie" in verdict["reason"], (
        "the refusal must say how to actually get a bigger region")


def test_a_run_with_no_original_roi_accepts_anything(tmp_path):
    """The intended workflow: analyse without worrying about regions, then draw
    one.  The full field was saved, so nothing is missing."""
    c = PostprocController()
    run = make_run(tmp_path, had_roi=False)
    assert c.canApply(run, [OUTER])["ok"] is True


def test_an_older_run_is_judged_by_its_saved_mask(tmp_path):
    """Runs analysed before the polygon was persisted have only the rasterised
    mask.  Its presence still proves an ROI was applied."""
    c = PostprocController()
    run = make_run(tmp_path, had_roi=False)
    np.save(os.path.join(run, "firefly_extras", "rec_roi_mask.npy"),
            np.ones((256, 256), dtype=bool))
    assert c.canApply(run, [OUTER])["ok"] is False


def test_no_polygon_and_no_run_are_both_refused(tmp_path):
    c = PostprocController()
    run = make_run(tmp_path)
    assert c.canApply(run, [])["ok"] is False
    assert c.canApply(str(tmp_path / "nope"), [INNER])["ok"] is False


def test_a_run_without_saved_localisations_cannot_be_reprocessed(tmp_path):
    c = PostprocController()
    run = tmp_path / "bare"
    (run / "firefly_extras").mkdir(parents=True)
    verdict = c.canApply(str(run), [INNER])
    assert verdict["ok"] is False and "no saved localisations" in verdict["reason"]


def test_start_refuses_and_reports_rather_than_dispatching(tmp_path):
    c = PostprocController()
    run = make_run(tmp_path, had_roi=True)
    seen = []
    c.failed.connect(seen.append)
    assert c.start(run, [OUTER]) is False
    assert c.running is False
    assert seen and "already had an ROI" in seen[0]


# ── editRun: opening a finished run in the editor ────────────────────────────
def test_editrun_works_with_the_movie_missing(tmp_path):
    """The common case — the movies live on a removable drive.  Without the
    rebuild the feature would be unavailable whenever it is unplugged."""
    c = RoiController(settings=None)
    run = make_run(tmp_path, movie=tmp_path / "not_here.tif", width=256, height=256)
    assert c.editRun(run) is True
    assert c.editing is True
    assert c.hasImage is True
    assert (c.imageWidth, c.imageHeight) == (256, 256), (
        "the rebuilt canvas must sit on the run's recorded pixel grid, or "
        "polygons drawn on it are rejected by the worker's extent check")


def test_editrun_uses_the_movie_when_it_exists(tmp_path):
    import tifffile
    movie = tmp_path / "mov.tif"
    tifffile.imwrite(str(movie),
                     np.random.default_rng(0).random((4, 64, 48)).astype("float32"))
    c = RoiController(settings=None)
    run = make_run(tmp_path, movie=movie, width=999, height=999)
    assert c.editRun(run) is True
    # 48x64 from the movie, not the deliberately-wrong 999 in params
    assert (c.imageWidth, c.imageHeight) == (48, 64)


def test_editrun_forces_polygon_mode_and_seeds_the_saved_roi(tmp_path):
    """Without forcing the mode the drawing canvas is hidden entirely (the
    default is Auto threshold), and without seeding you could add a region but
    never EDIT the one already there."""
    c = RoiController(settings=None)
    run = make_run(tmp_path, had_roi=True)
    c.editRun(run)
    assert c.roiMode == "Manual polygon"
    assert c.polygonCount == 1
    assert c.runPolygons()[0][0] == pytest.approx(SQUARE_50_150[0])


def test_editrun_on_a_run_predating_the_polygon_opens_empty(tmp_path):
    c = RoiController(settings=None)
    c.editRun(make_run(tmp_path, had_roi=False))
    assert c.polygonCount == 0 and c.editing is True


def test_editrun_hides_the_frame_scrubber_and_is_run_scoped(tmp_path):
    c = RoiController(settings=None)
    run = make_run(tmp_path)
    c.editRun(run)
    assert c.nFrames == 0
    assert c.runScoped is True
    assert os.path.abspath(c.runDir) == os.path.abspath(run)


def test_a_missing_run_folder_is_declined(tmp_path):
    c = RoiController(settings=None)
    assert c.editRun(str(tmp_path / "gone")) is False


# ── a finished run's ROI must not rewrite the sidebar ────────────────────────
class _RecSettings:
    """Records writes; SettingsController hardcodes the real QSettings domain,
    so a headless test must never use it."""

    def __init__(self):
        self.written = {}

    def get(self, k, d=""):
        return d

    def get_str(self, k, d=""):
        return d

    def get_bool(self, k, d=False):
        return d

    def get_float(self, k, d=0.0):
        return d

    def get_int(self, k, d=0):
        return d

    def set(self, k, v):
        self.written[k] = v

    def setValue(self, k, v):
        self.written[k] = v

    def sync(self):
        pass


def test_editing_a_finished_runs_roi_does_not_touch_the_sidebar(tmp_path):
    """That run is history.  Rewriting analysis/roi_mode or analysis/minmass
    from it would change the parameters of the NEXT run on the strength of a
    post-hoc edit."""
    s = _RecSettings()
    c = RoiController(settings=s)
    c.editRun(make_run(tmp_path))
    s.written.clear()
    c.roiMode = "Auto threshold"
    c.detectMinmass = 12.5
    assert s.written == {}, f"run-scoped edit leaked into settings: {s.written}"


def test_the_ordinary_import_path_still_writes_through(tmp_path):
    """The suppression must be conditional — a normal pre-run edit still mirrors
    into the sidebar default, which is how the run picks it up."""
    s = _RecSettings()
    c = RoiController(settings=s)
    c.editFile(str(tmp_path / "cell.tif"))     # non-existent is fine, no background
    s.written.clear()
    c.roiMode = "Manual polygon"
    assert s.written.get("analysis/roi_mode") == "Manual polygon"


def test_run_scope_is_cleared_on_cancel(tmp_path):
    c = RoiController(settings=None)
    c.editRun(make_run(tmp_path))
    c.cancel()
    assert c.runScoped is False and c.editing is False


def test_close_run_leaves_without_writing(tmp_path):
    c = RoiController(settings=None)
    c.editRun(make_run(tmp_path))
    c.closeRun()
    assert c.editing is False and c.runScoped is False and c.runDir == ""


# ── vertex editing (the controller half; dragging is manual-only) ────────────
def test_a_seeded_polygon_can_be_edited_not_just_replaced(tmp_path):
    c = RoiController(settings=None)
    c.editRun(make_run(tmp_path, had_roi=True))
    c.moveVertex(0, 0, 60.0, 65.0)
    assert c.runPolygons()[0][0] == pytest.approx([60.0, 65.0])
    c.deleteVertex(0, 0)
    assert len(c.runPolygons()[0]) == 3
    c.deleteVertex(0, 0)                       # would leave 2 → drops the polygon
    assert c.polygonCount == 0


def test_deleting_a_whole_region(tmp_path):
    c = RoiController(settings=None)
    c.editRun(make_run(tmp_path, had_roi=True))
    c.deletePolygon(0)
    assert c.polygonCount == 0
