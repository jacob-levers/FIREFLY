"""Command-line front-end for the benchmark harness.

    python -m firefly.bench.cli selfbench [--config cfg.json] [--out DIR] [--tiff]
    python -m firefly.bench.cli ingest --gt-dir DIR --csv f.csv --preset TrackMate

`selfbench` simulates ground truth, runs FIREFLY in-process, scores it, and
writes a summary CSV + report PNG (and optionally the stack TIFF + GT CSVs).
"""
from __future__ import annotations

import argparse
import os
import sys


def _selfbench(args):
    from firefly.bench.config import SimConfig, RunConfig, load_sim_config
    from firefly.bench.simulator import simulate
    from firefly.bench.runners import run_firefly_in_process
    from firefly.bench.report import evaluate, build_report_table, render_report_figure

    cfg = load_sim_config(args.config) if args.config else SimConfig()
    print(f"[bench] simulating seed={cfg.seed} {cfg.n_frames}f "
          f"{cfg.height}x{cfg.width} n_emitters={cfg.n_emitters} ...")
    sim = simulate(cfg)
    print(f"[bench] stack {sim.stack.shape}; {len(sim.gt_locs):,} GT loc-frames")

    res = run_firefly_in_process(
        sim.stack, pixel_size_um=cfg.pixel_size_um,
        frame_interval_s=cfg.frame_interval_s, run_cfg=RunConfig())
    row = evaluate(res, sim)
    table = build_report_table([row])

    os.makedirs(args.out, exist_ok=True)
    csv_p = os.path.join(args.out, "benchmark_summary.csv")
    png_p = os.path.join(args.out, "benchmark_report.png")
    table.to_csv(csv_p, index=False)
    render_report_figure([row], sim, out_path=png_p)
    print(table.to_string(index=False))
    print(f"[bench] wrote {csv_p}\n[bench] wrote {png_p}")

    if args.tiff or args.gt:
        from firefly.bench import io as bio
        if args.gt:
            paths = bio.write_gt_csvs(sim, args.out)
            print(f"[bench] wrote GT: {paths['locs']}, {paths['tracks']}")
        if args.tiff:
            tif = bio.write_tiff(sim.stack, os.path.join(args.out, "sim_stack.tif"))
            print(f"[bench] wrote {tif}")
    return 0


def _ingest(args):
    import json
    import pandas as pd
    from firefly.bench.runners import ingest_external_locs
    from firefly.bench.report import evaluate, build_report_table
    from firefly.bench.simulator import SimResult

    meta = json.load(open(os.path.join(args.gt_dir, "ground_truth_meta.json")))
    gt_locs = pd.read_csv(os.path.join(args.gt_dir, "ground_truth_locs.csv"))
    gt_tracks = pd.read_csv(os.path.join(args.gt_dir, "ground_truth_tracks.csv"))
    sim = SimResult(stack=None, gt_locs=gt_locs, gt_tracks=gt_tracks, meta=meta)

    res = ingest_external_locs(
        args.csv, preset=args.preset, pixel_size_um=meta["pixel_size_um"],
        frame_interval_s=meta["frame_interval_s"], name=args.name or args.preset,
        force_link=args.force_link)
    row = evaluate(res, sim)
    print(build_report_table([row]).to_string(index=False))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="firefly.bench.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    sb = sub.add_parser("selfbench", help="simulate + run FIREFLY + score")
    sb.add_argument("--config", help="SimConfig JSON (default: built-in)")
    sb.add_argument("--out", default="bench_run", help="output dir")
    sb.add_argument("--tiff", action="store_true", help="also write sim_stack.tif")
    sb.add_argument("--gt", action="store_true", help="also write GT CSVs")
    sb.set_defaults(func=_selfbench)

    ig = sub.add_parser("ingest", help="score an external tool export vs a saved GT")
    ig.add_argument("--gt-dir", required=True, help="dir with ground_truth_* files")
    ig.add_argument("--csv", required=True, help="tool's exported localisations/tracks")
    ig.add_argument("--preset", required=True,
                    help="TrackMate | PALM-Tracer | ThunderSTORM | Picasso")
    ig.add_argument("--name", help="display name (default: preset)")
    ig.add_argument("--force-link", action="store_true",
                    help="re-link with FIREFLY even if the export has tracks")
    ig.set_defaults(func=_ingest)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
