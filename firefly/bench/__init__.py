"""FIREFLY benchmark harness — simulate ground-truth sptPALM data, run FIREFLY
(and ingest palmTRACER/TrackMate exports), and score everyone with standard
detection / localisation / tracking / diffusion metrics.

Qt-free and import-light: heavy deps (torch via the localiser, tifffile) are only
pulled in by the functions that need them, not at package import time. CLI:
``python -m firefly.bench.cli selfbench``.
"""
from firefly.bench.config import (SimConfig, RunConfig, DiffusionPopulation,
                                   load_sim_config, dump_sim_config)

__all__ = [
    "SimConfig", "RunConfig", "DiffusionPopulation",
    "load_sim_config", "dump_sim_config",
    "simulate", "run_firefly_in_process", "ingest_external_locs",
    "evaluate", "build_report_table", "render_report_figure",
]


def __getattr__(name):
    # Lazy re-exports keep `import firefly.bench` cheap (no torch/scipy pulled
    # until you actually simulate or run a tool).
    if name == "simulate":
        from firefly.bench.simulator import simulate
        return simulate
    if name in ("run_firefly_in_process", "ingest_external_locs"):
        import firefly.bench.runners as r
        return getattr(r, name)
    if name in ("evaluate", "build_report_table", "render_report_figure"):
        import firefly.bench.report as rp
        return getattr(rp, name)
    raise AttributeError(name)
