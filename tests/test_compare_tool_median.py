"""tools/compare_to_palmtracer.py printed "median D — FIREFLY: nan".

FIREFLY exports before rc.15 wrote non-moving tracks with a blank D, and a plain
np.median over an array holding NaN is NaN.  Both files are now put on
PALMTracer's convention on load — blank or ≤ 1e-5 µm²/s counts as 1e-5 — so the
two medians describe the same population.
"""
import importlib.util
import os

import numpy as np
import pytest

_TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "tools", "compare_to_palmtracer.py")


@pytest.fixture(scope="module")
def tool():
    if not os.path.exists(_TOOL):          # local developer tooling, gitignored
        pytest.skip("tools/compare_to_palmtracer.py is not in this checkout")
    spec = importlib.util.spec_from_file_location("compare_to_palmtracer", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_blanks_count_as_non_moving_and_the_median_is_finite(tool):
    d, n_blank = tool._palmtracer_convention_d(np.array([0.1, np.nan, 0.3, np.nan, 0.2]))
    assert n_blank == 2
    assert np.all(np.isfinite(d))
    assert np.median(d) == pytest.approx(0.1)      # [1e-5, 1e-5, 0.1, 0.2, 0.3]


def test_values_at_or_below_the_floor_are_pinned_like_palmtracer(tool):
    d, n_blank = tool._palmtracer_convention_d(np.array([1e-7, 1e-5, 0.4]))
    assert n_blank == 0
    assert d.tolist() == pytest.approx([1e-5, 1e-5, 0.4])


def test_palmtracers_own_file_is_unchanged(tool):
    pt = np.array([1e-5, 0.02, 0.3])
    d, n_blank = tool._palmtracer_convention_d(pt)
    assert n_blank == 0 and d.tolist() == pytest.approx(pt.tolist())
