"""FIREFLY's PALMTracer-format D files must write non-moving tracks the way
PALMTracer does.

PALMTracer pins every D ≤ 1e-5 µm²/s at 1e-5 (LogD −5) and keeps the row.
FIREFLY wrote those tracks — the ones whose MSD slope came out non-positive —
as BLANK D and blank LogD, although the exporter's own comment promised that
"immobile tracks pile at the floor": clipping leaves NaN as NaN.  So anything
reading the file the PALMTracer way (a median, a LogD histogram) silently lost
~15% of tracks, all of them immobile.

The Mobile/Immobile ratio in the same file deliberately follows PALMTracer's
rule, which leaves those pinned tracks out — pinning the D column must not
change it.
"""
import csv

import numpy as np
import pandas as pd
import pytest

from firefly.analysis.fa_palmtracer import save_palmtracer_csvs


def _rows(path):
    with open(path, newline="") as fh:
        return [r for r in csv.reader(fh) if r and not r[0].startswith("#")]


@pytest.fixture
def export(tmp_path):
    # four tracks: two moved, one did not (non-positive slope), one unmeasurable
    tracks = pd.DataFrame({
        "particle": np.repeat([1, 2, 3, 4], 3), "frame": np.tile([0, 1, 2], 4),
        "x": np.arange(12, dtype=float), "y": np.arange(12, dtype=float),
        "mass": np.ones(12)})
    locs = tracks[["x", "y", "frame", "mass"]].copy()
    diff = pd.DataFrame({
        "particle": [1, 2, 3, 4],
        "D": [0.20, 0.01, np.nan, np.nan],
        "fit_status": ["fit", "fit", "nonpositive_slope", "insufficient_lags"],
        "MSD0": [0.001, 0.002, 0.003, np.nan], "MSE": [1e-6, 1e-6, 1e-6, np.nan]})
    imsd = pd.DataFrame({p: [0.01, 0.02] for p in (1, 2, 3, 4)}, index=[1, 2])
    save_palmtracer_csvs(str(tmp_path), "s", locs, tracks, diff, imsd, 0.106, 0.02,
                         width=20, height=20, n_frames=3, mobile_D_threshold=0.03,
                         logd_clip_min=1e-5, logd_clip_max=10.0)
    return tmp_path


def _by_trace(rows, n_cols):
    header, body = rows[0], rows[1:]
    return header, {int(r[1]): r for r in body}


def test_a_non_moving_track_is_written_at_the_floor_not_blank(export):
    header, rows = _by_trace(_rows(export / "s_trcPALMTracer-1-D.csv"), 8)
    d, logd = header.index("D(um2/s)"), header.index("LogD")
    still = [r for r in rows.values() if r[d] and float(r[d]) == pytest.approx(1e-5)]
    assert len(still) == 1, "the non-positive-slope track should be pinned at 1e-5"
    assert float(still[0][logd]) == pytest.approx(-5.0)


def test_the_allroi_file_pins_it_too(export):
    header, rows = _by_trace(_rows(export / "s_trcPALMTracer-AllROI-D.csv"), 5)
    d = header.index("D(um2/s)")
    assert sum(1 for r in rows.values() if r[d] and float(r[d]) == pytest.approx(1e-5)) == 1


def test_an_unmeasurable_track_stays_blank(export):
    """Too few lags is not evidence of immobility — it stays out, as everywhere."""
    header, rows = _by_trace(_rows(export / "s_trcPALMTracer-1-D.csv"), 8)
    d = header.index("D(um2/s)")
    assert sum(1 for r in rows.values() if r[d] == "") == 1


def test_measured_values_are_untouched(export):
    header, rows = _by_trace(_rows(export / "s_trcPALMTracer-1-D.csv"), 8)
    d = header.index("D(um2/s)")
    vals = sorted(float(r[d]) for r in rows.values() if r[d] and float(r[d]) > 1e-5)
    assert vals == pytest.approx([0.01, 0.20])


def test_the_mobile_ratio_still_follows_palmtracers_rule(export):
    """PALMTracer leaves pinned tracks out of Mobile/Immobile: 1 mobile (0.20)
    over 1 immobile (0.01) = 1.0.  Counting the pinned track would give 0.5."""
    header, rows = _by_trace(_rows(export / "s_trcPALMTracer-1-D.csv"), 8)
    col = header.index("Mobile/Immobile")
    ratio = [float(r[col]) for r in rows.values() if r[col]]
    assert ratio == [pytest.approx(1.0)]
