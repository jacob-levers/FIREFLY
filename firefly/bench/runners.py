"""Run a tool end-to-end and return a uniform result for scoring.

FIREFLY runs IN-PROCESS on the in-memory numpy stack (no TIFF/tifffile needed →
CI-safe).  External tools (palmTRACER / TrackMate) are bring-your-own-output:
the user runs them on the shared simulated `.tif` and exports a CSV, which we
ingest via FIREFLY's existing `load_external_locs`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd

from firefly.bench.config import RunConfig


@dataclass
class ToolRunResult:
    name: str
    locs: pd.DataFrame
    tracks: pd.DataFrame
    diff: pd.DataFrame
    extra: dict = field(default_factory=dict)


def _link_and_diffuse(locs, *, pixel_size_um, frame_interval_s, rc, already_linked=False):
    from firefly.analysis.fa_linking import link_trajectories
    from firefly.analysis.fa_diffusion import compute_msd_and_fit
    tracks = locs if already_linked else link_trajectories(
        locs, search_range=rc.search_range, memory=rc.memory, min_len=rc.min_len,
        linker=rc.linker)
    if tracks is None or len(tracks) == 0 or "particle" not in tracks.columns:
        return tracks, pd.DataFrame(columns=["particle", "D", "alpha", "motion"])
    _imsd, _emsd, diff = compute_msd_and_fit(
        tracks, pixel_size_um, frame_interval_s,
        max_lagtime=rc.max_lagtime, n_fit=rc.n_fit, workers=rc.workers)
    return tracks, diff


def run_firefly_in_process(stack, *, pixel_size_um, frame_interval_s,
                           run_cfg: RunConfig | None = None) -> ToolRunResult:
    """Full FIREFLY pipeline on an in-memory (T,H,W) stack: localise → link → MSD.
    `minmass=None` (default) uses FIREFLY's built-in auto-threshold."""
    from firefly.analysis.fa_localize import preprocess_and_localise_adaptive
    rc = run_cfg or RunConfig()
    mm = rc.minmass
    if mm is None and rc.auto_threshold == "linkability":
        # FIREFLY's flagship auto-threshold: a linkability sweep that picks the
        # F1-optimal mass cutoff (far more robust than the lighter peak
        # heuristic, especially on sparse / uniform-brightness fields).
        from firefly.analysis.fa_localize import estimate_minmass
        mm, _diag = estimate_minmass(stack, diameter=rc.diameter,
                                     percentile=rc.percentile, backend=rc.backend)
    # Returns (locs, mean_proj, max_proj, blink_proj, minmass) — take the locs
    # (first) and minmass (last) robustly regardless of the projection count.
    out = preprocess_and_localise_adaptive(
        stack, diameter=rc.diameter, minmass=mm, percentile=rc.percentile,
        workers=rc.workers, backend=rc.backend)
    locs, minmass_used = out[0], out[-1]
    tracks, diff = _link_and_diffuse(
        locs, pixel_size_um=pixel_size_um, frame_interval_s=frame_interval_s, rc=rc)
    return ToolRunResult("FIREFLY", locs, tracks, diff,
                         {"minmass": float(minmass_used)})


def compare_engines(sim, *, backends, linkers=("trackpy",), base_run_cfg=None):
    """Run the SAME simulated ground truth through each (backend, linker) engine
    and score it — returns one `evaluate()` row per engine, ready for
    `build_report_table` / `render_report_figure` (both already N-tool).

    Reuses `run_firefly_in_process` + `evaluate` unchanged; only the engine label
    (`ToolRunResult.name`, which the report keys on) and the per-engine `minmass`
    actually used are stamped on.  `base_run_cfg` supplies the shared knobs
    (diameter / percentile / minmass / search_range …); `backend` and `linker`
    are overridden per engine.
    """
    from dataclasses import replace
    from firefly.bench.report import evaluate
    base = base_run_cfg or RunConfig()
    one_linker = len(tuple(linkers)) == 1
    rows = []
    for backend in backends:
        for linker in linkers:
            rc = replace(base, backend=backend, linker=linker)
            res = run_firefly_in_process(
                sim.stack, pixel_size_um=sim.meta["pixel_size_um"],
                frame_interval_s=sim.meta["frame_interval_s"], run_cfg=rc)
            res.name = backend if one_linker else f"{backend}/{linker}"
            row = evaluate(res, sim)
            row["minmass"] = res.extra.get("minmass")
            rows.append(row)
    return rows


def ingest_external_locs(csv_path: str, preset: str, *, pixel_size_um,
                         frame_interval_s, name: str | None = None,
                         run_cfg: RunConfig | None = None,
                         force_link: bool = False) -> ToolRunResult:
    """Score an external tool's exported localisations/tracks against the same GT.

    Reuses `load_external_locs` (auto-maps TrackMate/PALM-Tracer/ThunderSTORM/
    Picasso). If the export already carries a `particle` column (e.g. TrackMate
    tracks) it's used as-is unless `force_link=True` (which isolates the tool's
    detector from FIREFLY's linker)."""
    from firefly.analysis.fa_loaders import load_external_locs
    rc = run_cfg or RunConfig()
    locs = load_external_locs(csv_path, preset=preset, pixel_size_um=pixel_size_um)
    already = ("particle" in locs.columns) and not force_link
    tracks, diff = _link_and_diffuse(
        locs, pixel_size_um=pixel_size_um, frame_interval_s=frame_interval_s,
        rc=rc, already_linked=already)
    return ToolRunResult(name or preset, locs, tracks, diff, {"preset": preset})


def run_trackmate_headless(tif_path: str, out_csv: str, *, fiji_path: str | None = None,
                           radius_px: float = 3.0, threshold: float = 0.0):
    """Phase-3 optional: drive TrackMate headless via Fiji + a Groovy script.

    Gated on a local Fiji install (`fiji_path` or $FIREFLY_FIJI_PATH); never runs
    in CI (pip can't install Fiji). Returns the exported CSV path; ingest it with
    `ingest_external_locs(..., preset="TrackMate")`.
    """
    fiji = fiji_path or os.environ.get("FIREFLY_FIJI_PATH")
    if not fiji or not os.path.exists(fiji):
        raise RuntimeError(
            "TrackMate automation needs a local Fiji launcher. Set "
            "$FIREFLY_FIJI_PATH (or pass fiji_path) to the ImageJ executable. "
            "Otherwise run TrackMate manually and use ingest_external_locs().")
    import subprocess
    script = os.path.join(os.path.dirname(__file__), "scripts", "trackmate_headless.groovy")
    arg = f"tif='{tif_path}',out='{out_csv}',radius={radius_px},threshold={threshold}"
    subprocess.run([fiji, "--headless", "--console", "-macro", script, arg], check=True)
    return out_csv
