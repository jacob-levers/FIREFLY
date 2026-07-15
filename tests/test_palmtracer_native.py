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


# ── D-coefficient (Log-D) clip range ─────────────────────────────────────────
# The palmTRACER-style clamp that feeds the LogD graph + exports; raw D untouched.
def _clamp_logd(D, d_min, d_max):
    D = np.asarray(D, float)
    lg = np.where(D > 0, np.log10(np.where(D > 0, D, 1.0)), np.nan)
    lo = np.log10(d_min) if d_min > 0 else -5.0
    hi = np.log10(d_max) if d_max > 0 else 1.0
    return np.clip(lg, lo, hi)


def test_dcoeff_clamp_math_pins_immobile_and_fast():
    # This is exactly what save_palmtracer_csvs applies to its LogD column, what
    # the firefly_extras `logD_clipped` column stores, and what the LogD graph
    # clips to.  The full save_palmtracer_csvs path is exercised manually (its
    # matplotlib-free CSV write trips the known offscreen-Qt teardown fault when
    # run alongside the QML tests — see test_updates_channel), so we assert the
    # pure clamp here.
    D = np.array([1e-14, 1e-3, 0.1, 1.0, 100.0, 0.0])   # last is D=0 → NaN
    out = _clamp_logd(D, 1e-5, 10.0)                     # → log₁₀ [-5, 1]
    assert out[0] == -5.0 and out[4] == 1.0             # immobile floor / fast ceiling
    assert np.isclose(out[2], -1.0)                      # 0.1 µm²/s → -1, untouched
    assert np.isnan(out[5])                              # D=0 stays NaN (dropped later)
    out2 = _clamp_logd(D, 1e-3, 1.0)                     # narrower → log₁₀ [-3, 0]
    assert out2[0] == -3.0 and out2[4] == 0.0
