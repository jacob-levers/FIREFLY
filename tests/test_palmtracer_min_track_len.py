"""palmTRACER imports honour a minimum track length.

palmTRACER's raw ``trc`` export is UNFILTERED — on real data roughly half of its
entries are single localisations, which are not trajectories.  palmTRACER's own
filtered outputs are restricted (e.g. ``TrackLength [8,1000]``) and FIREFLY's own
pipeline defaults to ``min_track_len=8``, so reading the raw export unfiltered
made the two pipelines silently incomparable and dragged every per-track median
toward zero.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from firefly.analysis.fa_palmtracer import (PALMTRACER_MIN_TRACK_LEN,
                                            load_summary_from_palmtracer)


def _write_pt_folder(root, track_lengths, *, frames_per=None):
    """A minimal but REAL palmTRACER folder: 3 metadata lines then the table,
    tab-separated, 1-based ``Plane``, matching the loader's column order."""
    os.makedirs(root, exist_ok=True)
    trc_rows, loc_rows = [], []
    rng = np.random.default_rng(0)
    for track_id, n in enumerate(track_lengths, start=1):
        for k in range(n):
            x, y = 10.0 + 0.3 * k, 20.0 + 0.2 * k
            plane = k + 1                                   # 1-based
            trc_rows.append((track_id, plane, x, y, 0.0, 900.0, len(trc_rows) + 1, 0.0))
            loc_rows.append((len(loc_rows) + 1, plane, 0, 0, 900.0, x, y,
                             1.2, 1.2, 0.0, 0.01, 0.0, 0.0, 0.0))

    def _dump(path, rows):
        with open(path, "w") as fh:
            fh.write("Width\tHeight\tnb_Planes\tPixel_Size(um)\tFrame_Duration(s)\n")
            fh.write("512\t512\t2000\t0.106\t0.02\n")
            fh.write("\t".join(f"c{i}" for i in range(len(rows[0]))) + "\n")
            for r in rows:
                fh.write("\t".join(str(v) for v in r) + "\n")

    _dump(os.path.join(root, "trcPALMTracer.txt"), trc_rows)
    _dump(os.path.join(root, "locPALMTracer.txt"), loc_rows)
    return root


def _n_tracks(summary):
    d = summary.get("diffusion")
    return 0 if d is None else int(len(d))


# ── the default matches FIREFLY's own pipeline ────────────────────────────────
def test_default_minimum_matches_fireflys_own_default():
    """The whole point is comparability: if these drift apart, an imported
    palmTRACER folder is filtered differently from a native FIREFLY run."""
    from firefly.ui.controllers.params.params_builder import _DEFAULTS
    assert PALMTRACER_MIN_TRACK_LEN == _DEFAULTS["min_track_len"]


# ── the filter actually drops short entries ──────────────────────────────────
def test_short_tracks_including_singletons_are_dropped(tmp_path):
    # 3 usable tracks (>=8) plus a pile of junk: singletons and 2-point entries
    lengths = [10, 12, 9] + [1] * 20 + [2] * 5
    folder = _write_pt_folder(str(tmp_path / "cell.PT"), lengths)
    s = load_summary_from_palmtracer(folder, cache=False)
    assert _n_tracks(s) == 3
    n = s["tracks"].groupby("particle").size()
    assert int(n.min()) >= PALMTRACER_MIN_TRACK_LEN


def test_the_minimum_is_overridable(tmp_path):
    lengths = [10, 3, 3, 1]
    folder = _write_pt_folder(str(tmp_path / "cell.PT"), lengths)
    assert _n_tracks(load_summary_from_palmtracer(
        folder, cache=False, min_track_len=3)) == 3          # 10, 3, 3
    assert _n_tracks(load_summary_from_palmtracer(
        folder, cache=False, min_track_len=1)) == 4          # unfiltered
    assert _n_tracks(load_summary_from_palmtracer(
        folder, cache=False, min_track_len=None)) == 4


def test_an_export_with_nothing_long_enough_fails_clearly(tmp_path):
    """Silently returning zero tracks would surface much later as an empty graph."""
    folder = _write_pt_folder(str(tmp_path / "tiny.PT"), [1, 2, 3])
    with pytest.raises(ValueError, match="minimum track length|reaches the"):
        load_summary_from_palmtracer(folder, cache=False)


# ── the cache records the minimum so a change invalidates it ──────────────────
def test_cache_records_the_minimum_and_a_change_makes_it_stale(tmp_path):
    from firefly.ui.controllers.workspace import workspace_data as wd
    folder = _write_pt_folder(str(tmp_path / "cell.PT"), [10, 12, 1, 1])
    load_summary_from_palmtracer(folder, cache=True)

    extras = os.path.join(folder, "firefly_extras")
    pj = [f for f in os.listdir(extras) if f.endswith("_params.json")][0]
    with open(os.path.join(extras, pj)) as fh:
        recorded = json.load(fh)["min_track_len"]
    assert recorded == PALMTRACER_MIN_TRACK_LEN

    resolved = wd._resolve_extras(folder)
    assert resolved is not None
    assert wd._palmtracer_cache_is_stale(folder, *resolved) is False

    # Simulate a cache written under a different policy → must be stale.
    with open(os.path.join(extras, pj)) as fh:
        data = json.load(fh)
    data["min_track_len"] = 1
    with open(os.path.join(extras, pj), "w") as fh:
        json.dump(data, fh)
    assert wd._palmtracer_cache_is_stale(folder, *resolved) is True
