"""Re-applying an edited ROI to a finished run — the WORKER half.

Deliberately free of any Qt import.  These are the tests that matter most (they
exercise the real pipeline end to end), and the `pytest core` CI job runs the
whole suite WITHOUT PySide6 installed — a module-level Qt import would skip this
file there, while the Qt job deselects the slow marker.  Kept apart, the
end-to-end test actually runs somewhere.

The UI half — the shrink guard, editRun, the settings-leak guards — lives in
tests/test_postproc_roi.py.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

SQUARE_50_150 = [[50.0, 50.0], [50.0, 150.0], [150.0, 150.0], [150.0, 50.0]]
INNER = [[70.0, 70.0], [70.0, 130.0], [130.0, 130.0], [130.0, 70.0]]
OUTER = [[0.0, 0.0], [0.0, 250.0], [250.0, 250.0], [250.0, 0.0]]


# ── the worker's own helpers ─────────────────────────────────────────────────
def test_the_polygon_is_recorded_in_a_form_that_reloads():
    """params.json must round-trip straight back into setPolygons, whatever
    shape the polygon arrived in — a bare polygon, a list, or numpy."""
    from firefly.firefly_worker import _roi_polygon_for_record
    single = _roi_polygon_for_record({"roi_polygon": SQUARE_50_150})
    assert single == [SQUARE_50_150], "a bare polygon must be wrapped to a list"
    assert len(_roi_polygon_for_record({"roi_polygon": [SQUARE_50_150] * 2})) == 2
    assert _roi_polygon_for_record(
        {"roi_polygon": [np.array(SQUARE_50_150)]}) == [SQUARE_50_150]
    assert _roi_polygon_for_record({}) is None
    assert _roi_polygon_for_record({"roi_polygon": []}) is None
    # JSON-safe: no numpy scalars left behind
    json.dumps(_roi_polygon_for_record({"roi_polygon": [np.array(SQUARE_50_150)]}))


def test_the_worker_warns_when_a_script_grows_an_roi():
    """The UI blocks this, but the worker is also reachable from scripts."""
    from firefly.firefly_worker import _warn_if_roi_grows
    locs = pd.DataFrame({"x": [50.0, 150.0], "y": [50.0, 150.0]})
    msgs = []
    assert _warn_if_roi_grows(locs, [INNER], True, msgs.append) is False
    assert _warn_if_roi_grows(locs, [OUTER], True, msgs.append) is True
    assert "OUTSIDE" in msgs[0]
    msgs.clear()
    # source had no ROI → the saved field is complete → nothing to warn about
    assert _warn_if_roi_grows(locs, [OUTER], False, msgs.append) is False
    assert not msgs


# ── end to end ───────────────────────────────────────────────────────────────
@pytest.mark.slow
def test_run_postproc_produces_a_complete_sibling_run(tmp_path):
    """The whole feature: a finished run + a smaller region → a new run folder
    with the full artefact set, named after the ORIGINAL stem."""
    import multiprocessing as mp
    import queue as _queue
    import sys

    from firefly.analysis.fa_enums import MsgKind
    from firefly.firefly_worker import run_postproc

    run = tmp_path / "rec"
    extras = run / "firefly_extras"
    extras.mkdir(parents=True)
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(300):                       # 300 short, linkable tracks
        x0, y0 = rng.uniform(20, 180, 2)
        for f in range(8):
            rows.append({"x": x0 + rng.normal(0, 0.3),
                         "y": y0 + rng.normal(0, 0.3), "frame": f, "mass": 3.0})
    pd.DataFrame(rows).to_csv(extras / "rec_localisations.csv", index=False)
    (extras / "rec_params.json").write_text(json.dumps({
        "width": 200, "height": 200, "pixel_size_um": 0.1,
        "frame_interval_s": 0.02, "search_range": 3, "memory": 0,
        "min_track_len": 5, "max_lagtime": 10, "n_fit": 5, "diameter": 7,
        "linker": "trackpy"}), encoding="utf-8")

    q, ev = mp.Queue(), mp.Event()
    real_stdout, real_stderr = sys.stdout, sys.stderr
    try:
        run_postproc({"source_folder": str(run),
                      "new_polygons": [[[40.0, 40.0], [40.0, 160.0],
                                        [160.0, 160.0], [160.0, 40.0]]]}, q, ev)
    finally:
        sys.stdout, sys.stderr = real_stdout, real_stderr   # the worker redirects both

    done = None
    while True:                                # mp.Queue.empty() is racy
        try:
            kind, payload = q.get(timeout=5.0)
        except _queue.Empty:
            break
        if kind == MsgKind.DONE:
            done = payload
        elif kind == MsgKind.ERROR:
            pytest.fail(f"run_postproc errored: {payload}")

    assert done, "no DONE payload reached the queue"
    out = str(run) + "_postproc1"
    assert os.path.abspath(done["postproc_output"]) == os.path.abspath(out)
    assert os.path.abspath(done["source_folder"]) == os.path.abspath(str(run))
    assert done["n_tracks"] > 0
    assert os.path.isdir(out)

    # named after the ORIGINAL stem, not the temp CSV we feed the engine
    new_extras = os.path.join(out, "firefly_extras")
    names = os.listdir(new_extras)
    assert any(n == "rec_diffusion_summary.csv" for n in names), sorted(names)[:6]
    assert not any("postproc_input" in n for n in names), (
        "outputs are named after the temp input CSV")
    for suffix in ("_localisations.csv", "_trajectories.csv",
                   "_ensemble_msd.csv", "_params.json"):
        assert os.path.isfile(os.path.join(new_extras, "rec" + suffix)), suffix

    # the ROI really was applied — fewer localisations than the source
    before = len(pd.read_csv(extras / "rec_localisations.csv"))
    after = len(pd.read_csv(os.path.join(new_extras, "rec_localisations.csv")))
    assert 0 < after < before
