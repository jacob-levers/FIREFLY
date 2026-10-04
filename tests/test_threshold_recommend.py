"""The recommended detection threshold, and the spots a threshold excludes.

"Set to recommended" moves the Preview & ROI threshold to the value FIREFLY
would choose for the open recording:

* palmTRACER-style detector — 4.4 × the recording's noise on the detection
  image (the second wavelet plane), the rule the MB112C analysis used;
* every other detector — exactly the minmass an Auto-threshold run picks
  (the worker's own call, shared rather than re-implemented).

The overlay also shows, in red, spots the detector finds at ¾ of the threshold
but not at the threshold itself, so what a threshold throws away is visible.
"""
import numpy as np
import pandas as pd
import pytest

from firefly.analysis.fa_detection_preview import below_threshold
from firefly.analysis.fa_localize import (WAVELET_NOISE_FACTOR, recommend_wavelet_threshold,
                                          wavelet_noise)
from firefly.analysis.fa_localize_backends import palmtracer_w2


class _Counting:
    """A stack that records which frames were read."""
    def __init__(self, arr):
        self.arr, self.read = arr, []
        self.shape = arr.shape
    def __len__(self): return len(self.arr)
    def __getitem__(self, i):
        self.read.append(int(i))
        return self.arr[int(i)]


def _noisy(n=100, size=64, sd=20.0, seed=0):
    return np.random.default_rng(seed).normal(1000, sd, (n, size, size)).astype(np.float32)


def test_wavelet_noise_is_the_median_robust_sigma_of_the_detection_image():
    stack = _noisy()
    idx = np.linspace(0, len(stack) - 1, 40).astype(int)
    want = []
    for i in idx:
        w = palmtracer_w2(stack[i]).ravel()
        want.append(1.4826 * np.median(np.abs(w - np.median(w))))
    sigma, used = wavelet_noise(stack, n_frames=40)
    assert sigma == pytest.approx(float(np.median(want)))
    assert used == 40


def test_the_recommendation_is_4_4_sigma_to_the_nearest_5():
    stack = _noisy(sd=35.0)
    sigma, _ = wavelet_noise(stack)
    t, info = recommend_wavelet_threshold(stack)
    assert WAVELET_NOISE_FACTOR == 4.4
    assert t == 5 * round(4.4 * sigma / 5) and t % 5 == 0
    assert info["sigma"] == pytest.approx(sigma) and info["factor"] == 4.4


def test_only_the_sampled_frames_are_read():
    """A 16 000-frame recording must not be loaded to sample 40 frames."""
    s = _Counting(_noisy(n=500, size=32))
    wavelet_noise(s, n_frames=40)
    assert sorted(set(s.read)) == list(np.linspace(0, 499, 40).astype(int))


def test_the_minmass_recommendation_is_the_runs_own_auto_threshold(monkeypatch):
    """Same function, same arguments as the worker — not a look-alike."""
    from firefly.analysis import fa_localize
    seen = {}
    monkeypatch.setattr(fa_localize, "estimate_minmass",
                        lambda stack, **kw: (seen.update(kw), (0.37, {"method": "x"}))[1])
    p = {"diameter": 7, "backend": "trackpy", "minmass_sensitivity": "strict",
         "minmass_mode": "linkability", "minmass_target_density": 25.0, "bg_radius": 9,
         "bg_method": "gaussian", "workers": 2, "search_range": 4, "memory": 2,
         "min_track_len": 8, "minmass_max_false_track_rate": 0.05}
    mm, _diag = fa_localize.estimate_minmass_for_run(np.zeros((3, 8, 8)), p)
    assert mm == 0.37
    assert seen == dict(diameter=7, percentile=64, backend="trackpy", sensitivity="strict",
                        mode="linkability", target_density=25.0, bg_radius=9,
                        bg_method="gaussian", workers=2, log_cb=None, search_range=4,
                        memory=2, link_min_len=8, max_false_track_rate=0.05)


def test_the_worker_uses_the_shared_auto_threshold_call():
    import inspect
    from firefly import firefly_worker
    src = inspect.getsource(firefly_worker)
    assert "estimate_minmass_for_run(" in src and "estimate_minmass(\n" not in src


def test_below_threshold_keeps_only_spots_the_threshold_drops():
    kept = pd.DataFrame({"x": [10.0, 30.0], "y": [10.0, 30.0]})
    low = pd.DataFrame({"x": [10.6, 30.0, 50.0, 70.0], "y": [9.5, 31.2, 50.0, 12.0]})
    out = below_threshold(kept, low, radius=3.0)
    assert sorted(out.x.tolist()) == [50.0, 70.0]
    assert below_threshold(kept.iloc[:0], low, radius=3.0).shape[0] == 4
    assert below_threshold(kept, low.iloc[:0], radius=3.0).shape[0] == 0
