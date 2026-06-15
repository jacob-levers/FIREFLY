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


def _compare(args):
    from firefly.bench.config import SimConfig, RunConfig, load_sim_config
    from firefly.bench.simulator import simulate
    from firefly.bench.runners import compare_engines
    from firefly.bench.report import build_report_table, render_report_figure

    cfg = load_sim_config(args.config) if args.config else SimConfig()
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    linkers = [l.strip() for l in args.linkers.split(",") if l.strip()] or ["trackpy"]
    # Torch / à trous need PyTorch — drop them with a clear note on a torch-less box.
    if any(b != "trackpy" for b in backends):
        try:
            import torch  # noqa: F401
        except Exception:
            dropped = [b for b in backends if b != "trackpy"]
            backends = [b for b in backends if b == "trackpy"]
            print(f"[bench] PyTorch unavailable — skipping {dropped}; "
                  f"running {backends or ['(none)']}")
    if not backends:
        print("[bench] no runnable backends; nothing to compare.")
        return 1

    base_rc = (RunConfig(minmass=args.minmass) if args.minmass is not None
               else RunConfig())
    print(f"[bench] simulating seed={cfg.seed} {cfg.n_frames}f "
          f"{cfg.height}x{cfg.width} n_emitters={cfg.n_emitters} ...")
    sim = simulate(cfg)
    print(f"[bench] comparing backends={backends} linkers={linkers} on "
          f"{len(sim.gt_locs):,} GT loc-frames"
          + (f" (fixed minmass={args.minmass})" if args.minmass is not None
             else " (each engine auto-thresholds)"))
    rows = compare_engines(sim, backends=backends, linkers=linkers,
                           base_run_cfg=base_rc)
    table = build_report_table(rows)

    os.makedirs(args.out, exist_ok=True)
    csv_p = os.path.join(args.out, "engine_comparison.csv")
    png_p = os.path.join(args.out, "engine_comparison.png")
    table.to_csv(csv_p, index=False)
    render_report_figure(rows, sim, out_path=png_p)
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
    # Headless on Windows, stdout defaults to cp1252 but the analysis pipeline
    # prints Unicode (→, µ, …) — force UTF-8 so a bench run doesn't die with a
    # UnicodeEncodeError mid-pipeline.  (The GUI redirects stdout, so this only
    # ever bites the CLI.)
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    p = argparse.ArgumentParser(prog="firefly.bench.cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    sb = sub.add_parser("selfbench", help="simulate + run FIREFLY + score")
    sb.add_argument("--config", help="SimConfig JSON (default: built-in)")
    sb.add_argument("--out", default="bench_run", help="output dir")
    sb.add_argument("--tiff", action="store_true", help="also write sim_stack.tif")
    sb.add_argument("--gt", action="store_true", help="also write GT CSVs")
    sb.set_defaults(func=_selfbench)

    cm = sub.add_parser("compare",
                        help="simulate + run MULTIPLE engines + score side-by-side")
    cm.add_argument("--config", help="SimConfig JSON (default: built-in)")
    cm.add_argument("--out", default="bench_compare", help="output dir")
    cm.add_argument("--backends", default="trackpy,torch,atrous",
                    help="comma list of detection backends "
                         "(trackpy,torch,atrous; torch/atrous need PyTorch)")
    cm.add_argument("--linkers", default="trackpy",
                    help="comma list of linkers (trackpy|lap|kalman)")
    cm.add_argument("--minmass", type=float, default=None,
                    help="fixed minmass for ALL engines (apples-to-apples "
                         "detection); default = each engine's auto-threshold")
    cm.add_argument("--tiff", action="store_true", help="also write sim_stack.tif")
    cm.add_argument("--gt", action="store_true", help="also write GT CSVs")
    cm.set_defaults(func=_compare)

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
