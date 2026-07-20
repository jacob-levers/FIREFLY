"""Fan-out of multiple ROIs into separate replicate runs (params_builder).

`expand_roi_replicates` turns a params dict for a movie with N drawn ROIs into N
single-polygon params (one output each) when the user asked to treat the ROIs as
individual replicates — so two cells on one dish never pool into each other's
D-values.  Off / single-ROI is a no-op (the polygons stay unioned into one run).
"""
from firefly.ui.controllers.params.params_builder import expand_roi_replicates

# two simple triangle polygons in (y, x)
POLY_A = [(0.0, 0.0), (0.0, 10.0), (10.0, 0.0)]
POLY_B = [(20.0, 20.0), (20.0, 30.0), (30.0, 20.0)]


def _base(**kw):
    p = {"file": "/data/dish1.czi", "stem_override": "dish1",
         "roi_polygon": [POLY_A, POLY_B], "roi_split_replicates": True,
         "roi_labels": None, "diameter": 7}
    p.update(kw)
    return p


def test_split_fans_out_one_run_per_roi():
    out = expand_roi_replicates(_base())
    assert len(out) == 2
    # each run carries exactly ONE polygon, and its own output stem
    assert out[0]["roi_polygon"] == [POLY_A]
    assert out[1]["roi_polygon"] == [POLY_B]
    assert out[0]["stem_override"] == "dish1_cell1"
    assert out[1]["stem_override"] == "dish1_cell2"
    # a fanned-out run is itself a single-ROI run (no further splitting)
    assert all(p["roi_split_replicates"] is False for p in out)
    # unrelated params are preserved
    assert all(p["diameter"] == 7 and p["file"] == "/data/dish1.czi" for p in out)


def test_labels_override_the_auto_number():
    out = expand_roi_replicates(_base(roi_labels=["nucleus", ""]))
    assert out[0]["stem_override"] == "dish1_nucleus"   # user label
    assert out[1]["stem_override"] == "dish1_cell2"     # blank → auto-number


def test_off_is_a_noop_single_run():
    p = _base(roi_split_replicates=False)
    out = expand_roi_replicates(p)
    assert out == [p]                                   # unchanged, one run (union)


def test_single_roi_is_a_noop_even_when_split_on():
    p = _base(roi_polygon=[POLY_A])                     # one ROI
    out = expand_roi_replicates(p)
    assert len(out) == 1 and out[0] is p


def test_bare_single_polygon_shape_is_handled():
    # roi_polygon given as ONE polygon (not a list of polygons) → still one run
    p = _base(roi_polygon=POLY_A)
    out = expand_roi_replicates(p)
    assert len(out) == 1


def test_duplicate_labels_do_not_collide():
    out = expand_roi_replicates(_base(roi_labels=["c", "c"]))
    stems = [p["stem_override"] for p in out]
    assert len(set(stems)) == 2                         # made unique


def test_derives_stem_from_filename_when_no_override():
    p = _base(stem_override=None)
    out = expand_roi_replicates(p)
    assert out[0]["stem_override"] == "dish1_cell1"     # from /data/dish1.czi
