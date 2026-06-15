"""Engine-comparison driver gate: run the SAME simulated ground truth through
multiple detection backends (and linkers) and confirm the comparison rows/table
are sane.

Mirrors test_bench_firefly_selfbench.py's easy (sparse, high-SNR) config so every
detector should score well — the point is the DRIVER (compare_engines +
build_report_table threading the engine label through), not stress-testing
detection (that lives in the selfbench + calibration tests).
"""
import pytest

from firefly.bench.config import SimConfig, RunConfig, DiffusionPopulation
from firefly.bench.simulator import simulate
from firefly.bench.runners import compare_engines
from firefly.bench.report import build_report_table, _TABLE_COLS


def _easy_cfg():
    return SimConfig(seed=7, n_frames=20, height=96, width=96, n_emitters=12,
                     photons_per_emitter=2200, bg_photons=8, read_noise_e=1.0,
                     photon_cv=0.0, k_on=0.3, k_off=0.06, bleach_prob=0.0,
                     populations=(DiffusionPopulation("brownian", 1.0,
                                                      D_um2_s=0.05),))


def test_compare_engines_trackpy_table():
    """compare_engines + build_report_table give one row per engine, scored."""
    sim = simulate(_easy_cfg())
    rows = compare_engines(sim, backends=("trackpy",),
                           base_run_cfg=RunConfig(minmass=2.0, workers=1))
    assert len(rows) == 1
    assert rows[0]["tool"] == "trackpy"
    for k in ("f1", "jsc", "recall", "precision"):
        assert 0.0 <= rows[0][k] <= 1.0
    assert rows[0]["recall"] >= 0.8, f"trackpy recall too low: {rows[0]['recall']}"
    table = build_report_table(rows)
    assert list(table.columns) == _TABLE_COLS
    assert len(table) == 1


def test_compare_engines_multi_backend():
    """trackpy + à trous run through the same driver; both find the bright spots
    and are distinguishable by name in the table."""
    pytest.importorskip("torch")
    sim = simulate(_easy_cfg())
    rows = compare_engines(sim, backends=("trackpy", "atrous"),
                           base_run_cfg=RunConfig(minmass=2.0, workers=1))
    assert [r["tool"] for r in rows] == ["trackpy", "atrous"]
    for r in rows:
        assert 0.0 <= r["f1"] <= 1.0
        assert r["recall"] >= 0.7, f"{r['tool']} recall too low: {r['recall']}"
    assert set(build_report_table(rows)["tool"]) == {"trackpy", "atrous"}


def test_compare_engines_threads_linker():
    """The linker is threaded through: same detector + two linkers → two rows
    with IDENTICAL detection (only tracking can differ)."""
    sim = simulate(_easy_cfg())
    linkers = ("trackpy", "kalman", "nn", "simple_lap")
    rows = compare_engines(sim, backends=("trackpy",),
                           linkers=linkers,
                           base_run_cfg=RunConfig(minmass=2.0, workers=1))
    assert [r["tool"] for r in rows] == [f"trackpy/{lk}" for lk in linkers]
    # Same detector → identical detection metrics across every linker; only the
    # tracking can differ.
    for r in rows[1:]:
        assert abs(rows[0]["f1"] - r["f1"]) < 1e-9
        assert abs(rows[0]["recall"] - r["recall"]) < 1e-9
