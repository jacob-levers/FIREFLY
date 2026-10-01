"""The comparison must flag detection thresholds that differ — within a condition
AND between conditions.

It only checked within a group, and only minmass.  The 2015 palmTRACER analysis
of these PC12 cells set the wavelet threshold per cell — 150 to 380 — with a
much wider spread in propofol than in DMSO.  Different thresholds BETWEEN
conditions are the most direct way to manufacture a difference between them,
and a group-by-group check cannot see it when each group happens to be uniform.
"""
from firefly.analysis.fa_compare import _detection_threshold, _threshold_warnings


def _s(**params):
    return {"params": params}


def test_minmass_runs_report_their_minmass():
    assert _detection_threshold({"resolved_minmass": 0.45, "minmass": 1.0}) == ("minmass", 0.45)
    assert _detection_threshold({"minmass": 1.0}) == ("minmass", 1.0)


def test_palmtracer_runs_report_their_wavelet_threshold():
    """minmass is unused (0) for that detector — reading it would hide every difference."""
    assert _detection_threshold({"backend": "palmtracer", "minmass": 0.0,
                                 "wavelet_threshold": 350.0}) == ("wavelet", 350.0)


def test_mixed_within_a_group_is_flagged():
    w = _threshold_warnings(["DMSO"], [[_s(minmass=0.4), _s(minmass=0.5)]])
    assert len(w) == 1 and "DMSO" in w[0]


def test_uniform_groups_on_different_thresholds_are_flagged():
    w = _threshold_warnings(["DMSO", "Propofol"],
                            [[_s(backend="palmtracer", minmass=0, wavelet_threshold=250)] * 3,
                             [_s(backend="palmtracer", minmass=0, wavelet_threshold=350)] * 3])
    assert any("between" in x.lower() for x in w), w


def test_one_threshold_everywhere_is_silent():
    w = _threshold_warnings(["A", "B"], [[_s(minmass=0.45)] * 2, [_s(minmass=0.45)] * 3])
    assert w == []


def test_runs_without_a_recorded_threshold_are_ignored():
    assert _threshold_warnings(["A", "B"], [[_s()], [_s(minmass=0.45)]]) == []
