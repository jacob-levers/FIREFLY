"""Phase-1 end-to-end gate: simulate ground truth, run FIREFLY fully in-process
on the in-memory stack (no TIFF / tifffile), score it, and assert the numbers
are sane on an easy (sparse-ish, high-SNR) sim.

A low F1 or a large RMSE here means something is mis-wired (a px↔nm unit slip, an
off-by-one frame index, a broken matcher) — this is the canary.  Runs in a couple
of seconds with `workers=1` and the deterministic trackpy backend.
"""
import numpy as np

from firefly.bench.config import SimConfig, RunConfig, DiffusionPopulation
from firefly.bench.simulator import simulate
from firefly.bench.runners import run_firefly_in_process
from firefly.bench.report import evaluate


def test_firefly_selfbench_sane():
    cfg = SimConfig(seed=5, n_frames=40, height=128, width=128, n_emitters=16,
                    photons_per_emitter=2200, bg_photons=8, read_noise_e=1.0,
                    photon_cv=0.0, k_on=0.3, k_off=0.06, bleach_prob=0.0,
                    populations=(DiffusionPopulation("immobile", 0.5),
                                 DiffusionPopulation("brownian", 0.5, D_um2_s=0.05)))
    sim = simulate(cfg)
    res = run_firefly_in_process(
        sim.stack, pixel_size_um=cfg.pixel_size_um,
        frame_interval_s=cfg.frame_interval_s,
        run_cfg=RunConfig(minmass=2.0, backend="trackpy", workers=1))
    row = evaluate(res, sim)

    assert row["f1"] >= 0.85, f"detection F1 too low: {row['f1']}"
    assert row["recall"] >= 0.85, f"recall too low: {row['recall']}"
    assert row["rmse_nm"] <= 35.0, f"localisation RMSE too high: {row['rmse_nm']}"
    assert row["jsc_theta"] >= 0.7, f"tracking JSCθ too low: {row['jsc_theta']}"

    pops = row["_detail"]["diffusion"]["per_population"]
    # immobile recovered as ~0; Brownian within a few-fold of the true 0.05 µm²/s
    assert pops["immobile"]["D_est_median"] < 0.02
    assert 0.01 < pops["brownian"]["D_est_median"] < 0.15
