"""Multiple ROIs → separate replicate outputs.

The split lives in the WORKER (`_roi_replicate_jobs`), not the launcher: the movie
is decoded + localised ONCE and the per-ROI analysis is then looped, so two cells
on one dish each get their own output without pooling into each other's D-values
— and without re-localising the movie per cell.  `expand_roi_replicates` is now a
pass-through seam in the launcher.
"""
from firefly.ui.controllers.params.params_builder import expand_roi_replicates
from firefly.firefly_worker import _roi_replicate_jobs

# two simple triangle polygons in (y, x)
POLY_A = [(0.0, 0.0), (0.0, 10.0), (10.0, 0.0)]
POLY_B = [(20.0, 20.0), (20.0, 30.0), (30.0, 20.0)]


def _base(**kw):
    p = {"file": "/data/dish1.czi", "stem_override": "dish1",
         "roi_polygon": [POLY_A, POLY_B], "roi_split_replicates": True,
         "roi_labels": None, "diameter": 7}
    p.update(kw)
    return p


# ── the launcher hands the worker ONE run (it localises once) ────────────────
def test_launcher_sends_a_single_run_carrying_every_polygon():
    p = _base()
    out = expand_roi_replicates(p)
    assert out == [p], "the launcher must not fan out — the worker splits"
    # …and that single params still carries both polygons + the split request,
    # which is what lets the worker localise once and then loop.
    assert len(out[0]["roi_polygon"]) == 2
    assert out[0]["roi_split_replicates"] is True


# ── the worker's split ───────────────────────────────────────────────────────
def test_split_makes_one_job_per_roi():
    jobs = _roi_replicate_jobs(_base())
    assert len(jobs) == 2
    (polys_a, label_a), (polys_b, label_b) = jobs
    assert polys_a == [POLY_A] and polys_b == [POLY_B]   # one ROI per job
    assert label_a == "cell1" and label_b == "cell2"     # auto-numbered outputs


def test_labels_override_the_auto_number():
    jobs = _roi_replicate_jobs(_base(roi_labels=["nucleus", ""]))
    assert [lbl for _p, lbl in jobs] == ["nucleus", "cell2"]   # blank → auto


def test_off_is_a_single_unioned_job():
    jobs = _roi_replicate_jobs(_base(roi_split_replicates=False))
    assert jobs == [(None, None)]      # None → leave the params' polygons alone


def test_single_roi_is_one_job_even_when_split_on():
    assert _roi_replicate_jobs(_base(roi_polygon=[POLY_A])) == [(None, None)]


def test_bare_single_polygon_shape_is_handled():
    # roi_polygon given as ONE polygon (not a list of polygons) → still one job
    assert _roi_replicate_jobs(_base(roi_polygon=POLY_A)) == [(None, None)]


def test_duplicate_labels_do_not_collide():
    labels = [lbl for _p, lbl in _roi_replicate_jobs(_base(roi_labels=["c", "c"]))]
    assert len(set(labels)) == 2       # made unique so outputs can't overwrite


def test_no_polygons_is_one_job():
    assert _roi_replicate_jobs(_base(roi_polygon=None)) == [(None, None)]
