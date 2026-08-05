"""Legacy density-matched auto-threshold and auto-minmass policy plumbing.

Density matching remains loadable for existing presets/runs, but it is count
normalisation rather than a guarantee of unbiased detection: a genuine abundance
difference can also be normalised away.  New Drosophila presets use the separate
quality-first policy, whose core estimator is tested in its own module.
"""
import numpy as np
import pytest

from firefly.analysis.fa_localize import estimate_minmass


def _spots(rng, n_frames, per_frame, size=128, bright=True):
    """Frames of Gaussian spots at a controlled density."""
    st = np.zeros((n_frames, size, size), np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    for f in range(n_frames):
        img = rng.normal(100.0, 3.0, (size, size))
        for _ in range(per_frame):
            cy, cx = rng.uniform(6, size - 6, 2)
            amp = rng.uniform(300, 900) if bright else rng.uniform(60, 120)
            img += amp * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 1.3 ** 2)))
        st[f] = img
    return st


def _estimate(st, **kw):
    return estimate_minmass(st, diameter=7, bg_radius=10,
                            bg_method="uniform_filter", workers=1,
                            search_range=3, memory=3, link_min_len=4,
                            log_cb=lambda m: None, **kw)


# ── the target is actually hit ───────────────────────────────────────────────
@pytest.mark.slow
def test_density_mode_lands_near_the_requested_density():
    rng = np.random.default_rng(0)
    st = _spots(rng, 60, per_frame=40)
    mm, diag = _estimate(st, mode="density", target_density=12.0)
    assert diag["method"] == "density_matched"
    assert diag["density_target"] == 12.0
    # the achieved density is what the chosen threshold actually yields
    assert abs(diag["density_achieved"] - 12.0) < 4.0, diag["density_achieved"]
    assert mm > 0
    assert not diag.get("qc"), "a reachable target must not raise a QC flag"


@pytest.mark.slow
def test_two_recordings_of_different_brightness_get_matched_densities():
    """Legacy count normalisation does what it says, including the important
    caveat that it can suppress a genuine difference in emitter abundance."""
    a = _spots(np.random.default_rng(1), 60, per_frame=25)
    b = _spots(np.random.default_rng(2), 60, per_frame=90)   # much denser
    _mma, da = _estimate(a, mode="density", target_density=10.0)
    _mmb, db = _estimate(b, mode="density", target_density=10.0)
    assert abs(da["density_achieved"] - db["density_achieved"]) < 4.0, (
        da["density_achieved"], db["density_achieved"])


# ── an unreachable target is flagged, not silently accepted ──────────────────
@pytest.mark.slow
def test_a_sparse_recording_is_flagged_rather_than_silently_included():
    rng = np.random.default_rng(3)
    st = _spots(rng, 40, per_frame=3)
    mm, diag = _estimate(st, mode="density", target_density=200.0)
    assert diag["method"] == "density_matched"
    assert diag.get("qc"), "an unreachable density target must raise a QC flag"
    assert "below_target_density" in diag["qc"]
    assert diag["density_achieved"] < 200.0
    assert mm > 0, "a flagged file still gets a usable threshold"


# ── the default path is untouched ────────────────────────────────────────────
@pytest.mark.slow
def test_linkability_remains_the_default_and_ignores_target_density():
    rng = np.random.default_rng(4)
    st = _spots(rng, 60, per_frame=30)
    _mm, diag = _estimate(st)                       # no mode → linkability
    assert diag["method"] != "density_matched"
    assert "density_target" not in diag


def test_settings_plumbing_reaches_worker_params():
    """The sidebar keys must survive into the params the worker consumes."""
    from firefly.ui.controllers.params.params_builder import build_params, _DEFAULTS
    from firefly.analysis.fa_enums import MinmassMode

    class _S:
        def __init__(self, mode=None, dens=None, false_rate=None):
            self.mode, self.dens, self.false_rate = mode, dens, false_rate
        def get_str(self, k, d=""):
            return (self.mode if (k == "analysis/minmass_mode"
                                  and self.mode is not None) else d)
        def get_bool(self, k, d=False): return d
        def get_float(self, k, d=0.0):
            if k == "analysis/minmass_target_density" and self.dens is not None:
                return self.dens
            if (k == "analysis/minmass_max_false_track_rate"
                    and self.false_rate is not None):
                return self.false_rate
            return d

    class _Imp:
        filePath = "/d/rec.czi"; outDir = None; isCsv = False
        overridePx = False; pixelSize = 0.1; overrideFi = False; frameInterval = 0.02

    assert _DEFAULTS["minmass_mode"] == "Linkability"
    p = build_params(_S(), _Imp())
    assert p["minmass_mode"] == "linkability"
    p = build_params(_S("Density-matched", 25.0), _Imp())
    assert p["minmass_mode"] == "density" and p["minmass_target_density"] == 25.0
    # The new policy has an explicit wire and survives into replay provenance.
    p = build_params(_S("Quality-first (track ambiguity)", 25.0, 10.0), _Imp())
    assert p["minmass_mode"] == "quality_first"
    assert p["minmass_max_false_track_rate"] == 0.10
    assert p["widget_state"]["analysis/minmass_mode"] == \
        "Quality-first (track ambiguity)"
    assert p["widget_state"]["analysis/minmass_target_density"] == 25.0
    # Canonical legacy wire values remain loadable (saved programmatic presets).
    assert build_params(_S("density", 25.0), _Imp())["minmass_mode"] == "density"
    assert build_params(_S("quality_first", 25.0), _Imp())["minmass_mode"] == \
        "quality_first"
    assert MinmassMode.parse("Density-matched") is MinmassMode.DENSITY
    assert MinmassMode.parse("Quality-first (track ambiguity)") is \
        MinmassMode.QUALITY_FIRST
    # A future/garbled policy cannot silently fall through to a different method.
    with pytest.warns(RuntimeWarning, match="unknown auto-minmass mode"):
        p = build_params(_S("future-policy", 25.0), _Imp())
    assert p["minmass_mode"] == "linkability"


def test_the_fly_preset_requests_quality_first_policy():
    """The Drosophila preset locks its empirical floor, provisional ambiguity
    ceiling and backend, and every key must be understood by the sidebar."""
    import json
    from pathlib import Path
    from firefly.ui.controllers.params import sidebar_schema as S
    p = json.loads((Path(__file__).resolve().parents[1]
                    / "firefly/ui/presets/Drosophila Neurons.json").read_text())
    p.pop("__firefly_builtin__", None)
    assert p["analysis/minmass_mode"] == "Quality-first (track ambiguity)"
    assert p["analysis/auto_minmass"] is True
    assert p["analysis/minmass"] == 0.16
    assert p["analysis/minmass_max_false_track_rate"] == 10.0
    assert p["analysis/backend"] == "Crocker–Grier — PyTorch (GPU)"
    assert [k for k in p if k not in S.BY_KEY] == []
