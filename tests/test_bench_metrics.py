"""Tests for the benchmark metrics (Qt-free; numpy + scipy + pandas)."""
import numpy as np
import pandas as pd

from firefly.bench.metrics import (detection_metrics, tracking_isbi,
                                    diffusion_recovery, crlb_sigma_nm)


def test_detection_counts_and_rmse():
    gt = pd.DataFrame({"frame": [0, 0, 0], "x": [10., 20., 30.], "y": [10., 20., 30.]})
    est = pd.DataFrame({"frame": [0, 0, 0], "x": [10.05, 20.0, 50.],
                        "y": [10., 20.05, 50.]})
    m = detection_metrics(est, gt, tol_px=2.0, pixel_size_um=0.1)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 1)
    assert abs(m["precision"] - 2 / 3) < 1e-9
    assert abs(m["recall"] - 2 / 3) < 1e-9
    assert abs(m["jaccard"] - 0.5) < 1e-9
    # matched offset is 0.05 px → 5 nm at 0.1 µm/px
    assert abs(m["rmse_nm"] - 5.0) < 1e-6


def test_detection_hungarian_beats_greedy():
    # two GT close together; a greedy NN could mis-assign, Hungarian is optimal
    gt = pd.DataFrame({"frame": [0, 0], "x": [10.0, 11.0], "y": [10.0, 10.0]})
    est = pd.DataFrame({"frame": [0, 0], "x": [10.1, 11.1], "y": [10.0, 10.0]})
    m = detection_metrics(est, gt, tol_px=1.5, pixel_size_um=0.1)
    assert m["tp"] == 2 and m["fp"] == 0 and m["fn"] == 0


def _tracks(ids, n=10):
    rows = []
    for p in ids:
        for f in range(n):
            rows.append((f, p * 5.0, p * 5.0 + f * 0.1, p))
    return pd.DataFrame(rows, columns=["frame", "x", "y", "particle"])


def test_tracking_perfect():
    gt = _tracks([1, 2, 3])
    t = tracking_isbi(_tracks([1, 2, 3]), gt, gate_px=2.0, pixel_size_um=0.1)
    assert abs(t["alpha"] - 1) < 1e-9
    assert abs(t["beta"] - 1) < 1e-9
    assert abs(t["jsc"] - 1) < 1e-9
    assert abs(t["jsc_theta"] - 1) < 1e-9
    assert t["rmse_nm"] < 1e-6


def test_tracking_missed_track_lowers_jsc_theta():
    gt = _tracks([1, 2, 3])
    t = tracking_isbi(_tracks([1, 2]), gt, gate_px=2.0, pixel_size_um=0.1)
    assert abs(t["jsc_theta"] - 2 / 3) < 1e-6
    assert t["alpha"] < 1.0


def test_tracking_spurious_track_makes_beta_below_alpha():
    gt = _tracks([1, 2, 3])
    t = tracking_isbi(_tracks([1, 2, 3, 4]), gt, gate_px=2.0, pixel_size_um=0.1)
    assert t["beta"] < t["alpha"]
    assert abs(t["jsc_theta"] - 0.75) < 1e-6


def test_diffusion_recovery_unbiased_on_truth():
    gt = _tracks([1, 2])
    diff = pd.DataFrame({"particle": [1, 2], "D": [0.05, 0.10],
                         "alpha": [1.0, 1.0], "motion": ["Brownian", "Brownian"]})
    gt2 = gt.copy()
    gt2["population"] = np.where(gt2["particle"] == 1, "brownian", "directed")
    meta = {"populations": {"brownian": {"D_um2_s": 0.05},
                            "directed": {"D_um2_s": 0.10}}}
    pairs = [(1, 1), (2, 2)]
    rec = diffusion_recovery(diff, gt2, pairs, meta)
    assert abs(rec["per_population"]["brownian"]["D_bias_pct"]) < 1e-6


def test_crlb_decreases_with_more_photons():
    a = crlb_sigma_nm(200, 5, 0.9, 0.1)
    b = crlb_sigma_nm(2000, 5, 0.9, 0.1)
    assert b < a and np.isfinite(a) and np.isfinite(b)
