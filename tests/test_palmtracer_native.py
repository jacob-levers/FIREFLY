"""palmTRACER use_native: overlay native D but KEEP FIREFLY's motion class.

Regression for the bug where use_native took palmTRACER's native -D wholesale,
blanking alpha/motion (Compare's Motion-Class panel read ~100% unclassified).
"""
import numpy as np
import pandas as pd

from firefly.analysis.fa_palmtracer import _native_d_overrides


def test_native_d_overrides_keeps_alpha_and_motion():
    ff = pd.DataFrame({
        "particle": [1, 2, 3],
        "D": [0.10, 0.20, 0.30], "MSD0": [1.0, 2.0, 3.0],
        "MSE": [0.01, 0.02, 0.03], "logD": [-1.0, -0.70, -0.52],
        "alpha": [0.9, 1.4, np.nan],
        "motion": ["Brownian", "Directed", "Immobile"],
        "loc_sigma_nm": [20.0, 21.0, 22.0],
        "radius_of_gyration_um": [0.10, 0.11, 0.12],
    })
    native = pd.DataFrame({
        "particle": [1, 2, 3],
        "D": [0.15, 0.25, 0.35], "MSD0": [1.5, 2.5, 3.5],
        "MSE": [0.015, 0.025, 0.035], "logD": [-0.82, -0.60, -0.46],
        "alpha": [np.nan] * 3, "motion": ["Unclassified"] * 3,
    })
    out = _native_d_overrides(ff, native).sort_values("particle").reset_index(drop=True)
    # D / MSD family taken from palmTRACER's native values
    assert list(out["D"]) == [0.15, 0.25, 0.35]
    assert list(out["MSD0"]) == [1.5, 2.5, 3.5]
    # alpha / motion / FIREFLY-only metrics KEPT (not blanked)
    assert list(out["motion"]) == ["Brownian", "Directed", "Immobile"]
    assert out.loc[0, "alpha"] == 0.9
    assert out.loc[0, "loc_sigma_nm"] == 20.0
    assert out.loc[0, "radius_of_gyration_um"] == 0.10
    # the whole point: NOT a single track is left "Unclassified"
    assert (out["motion"] == "Unclassified").sum() == 0


def test_native_d_overrides_falls_back_when_track_absent():
    ff = pd.DataFrame({"particle": [1, 2, 3], "D": [0.1, 0.2, 0.3],
                       "alpha": [1.0, 1.0, 1.0], "motion": ["Brownian"] * 3})
    native = pd.DataFrame({"particle": [1, 2], "D": [0.15, 0.25],
                           "alpha": [np.nan] * 2, "motion": ["Unclassified"] * 2})
    out = _native_d_overrides(ff, native).set_index("particle")
    assert out.loc[1, "D"] == 0.15 and out.loc[2, "D"] == 0.25   # overridden
    assert out.loc[3, "D"] == 0.3                                 # FIREFLY's kept
    assert (out["motion"] == "Brownian").all()                   # motion intact
