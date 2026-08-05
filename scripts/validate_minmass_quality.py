#!/usr/bin/env python3
"""Bounded, read-only validation harness for FIREFLY quality-first minmass.

The harness intentionally performs only the sampled calibration pass used by
``estimate_minmass(mode="quality_first")``.  It never runs a full analysis and
never writes beside a movie or an existing FIREFLY result.  All generated
files must live in a caller-provided directory below ``/private/tmp``.

The default ``honours-core`` panel covers the deliberately contrasting runs
used to validate the Drosophila policy:

* 03Aug Fly-1 L — low density-matched threshold;
* 03Aug Fly-3 R — high included threshold;
* 04Aug excluded Fly-1 L — known low-link-ratio stress case;
* 05Aug Fly-1 R — analysed 1-AMA recording.

Use ``--plan-only`` first to inspect the exact files and four 120-frame source
windows without decoding image pixels.  Omitting ``--plan-only`` reads only
those 480 frames per movie, reconstructs the saved polygon ROI, and calls the
production estimator API.  Existing results and raw movies are read-only.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Iterable

import numpy as np


sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_HONOURS_ROOT = Path(
    "/Volumes/Jacob's EXD/Data/My Data/ELYRA/Honours")
PRIVATE_TMP_ROOT = Path("/private/tmp")
N_WINDOWS = 4
FRAMES_PER_WINDOW = 120


@dataclass(frozen=True)
class RelativeCase:
    label: str
    movie: str
    manifest: str


@dataclass(frozen=True)
class Case:
    label: str
    movie: Path
    manifest: Path


CORE_PANEL: tuple[RelativeCase, ...] = (
    RelativeCase(
        "control_03aug_fly1_l_low",
        "N=3 MB543B-Sx1A-mEos3.2_CrimsonVenus 03Aug/"
        "Fly-1-16k Frames-LSide.czi",
        "N=3 MB543B-Sx1A-mEos3.2_CrimsonVenus 03Aug/batch_results/"
        "Fly-1-16k Frames-LSide/Fly-1-16k Frames-LSide_run_manifest.json",
    ),
    RelativeCase(
        "control_03aug_fly3_r_high",
        "N=3 MB543B-Sx1A-mEos3.2_CrimsonVenus 03Aug/"
        "Fly-3-16k Frames-RSide.czi",
        "N=3 MB543B-Sx1A-mEos3.2_CrimsonVenus 03Aug/batch_results/"
        "Fly-3-16k Frames-RSide/Fly-3-16k Frames-RSide_run_manifest.json",
    ),
    RelativeCase(
        "control_04aug_fly1_l_excluded_stress",
        "N=1 MB543B-Sx1A-mEos3.2_CrimsonVenus 04Aug/Excluded/"
        "Fly-1-16k Frames-LSide.czi",
        "N=1 MB543B-Sx1A-mEos3.2_CrimsonVenus 04Aug/Excluded/"
        "Fly-1-16k Frames-LSide/Fly-1-16k Frames-LSide_run_manifest.json",
    ),
    RelativeCase(
        "ama_05aug_fly1_r",
        "N=1 MB543B-Sx1A-mEos3.2_CrimsonVenus 05Aug 1-AMA/"
        "Fly-1-16k Frames-RSide.czi",
        "N=1 MB543B-Sx1A-mEos3.2_CrimsonVenus 05Aug 1-AMA/"
        "Fly-1-16k Frames-RSide/Fly-1-16k Frames-RSide_run_manifest.json",
    ),
)


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_output_dir(output_dir: Path, *, source_roots: Iterable[Path]) -> Path:
    """Resolve and validate the sole writable directory used by this script.

    Requiring ``/private/tmp`` also protects against an output symlink whose
    spelling looks temporary but whose target is on ``/Volumes``.
    """
    out = output_dir.expanduser().resolve(strict=False)
    tmp = PRIVATE_TMP_ROOT.resolve(strict=True)
    if out == tmp or not _within(out, tmp):
        raise ValueError(
            f"--output-dir must be a dedicated directory below {tmp}; got {out}")
    volumes = Path("/Volumes").resolve(strict=False)
    if out == volumes or _within(out, volumes):
        raise ValueError("refusing to write validation output under /Volumes")
    for raw_root in source_roots:
        root = raw_root.expanduser().resolve(strict=False)
        if out == root or _within(out, root):
            raise ValueError(
                f"refusing to place validation output inside source root {root}")
    return out


def _parse_case(value: str) -> Case:
    parts = value.split("::", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "--case must be LABEL::MOVIE.czi::RUN_MANIFEST.json")
    return Case(parts[0].strip(), Path(parts[1]), Path(parts[2]))


def core_panel(root: Path) -> list[Case]:
    root = root.expanduser()
    return [Case(c.label, root / c.movie, root / c.manifest)
            for c in CORE_PANEL]


def _source_windows(n_frames: int) -> list[tuple[int, int]]:
    """Mirror FIREFLY's production 4x120 contiguous-window sampler."""
    n = int(n_frames)
    if n <= 0:
        raise ValueError("movie reports no frames")
    if n < 250:
        return [(0, n)]
    win_len = int(min(FRAMES_PER_WINDOW, max(16, n // N_WINDOWS)))
    starts = np.linspace(0, n - win_len, N_WINDOWS).astype(int)
    windows: list[tuple[int, int]] = []
    for start in starts:
        start = int(start)
        if windows and start < windows[-1][1]:
            continue
        windows.append((start, start + win_len))
    return windows


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(
            payload.get("parameters"), dict):
        raise ValueError(f"manifest has no parameters object: {path}")
    return payload


def _inspect_czi(path: Path) -> dict[str, Any]:
    from aicspylibczi import CziFile

    czi = CziFile(str(path))
    dims = str(czi.dims)
    size = tuple(int(v) for v in czi.size)

    def dim(name: str, default: int = 1) -> int:
        return size[dims.index(name)] if name in dims else default

    return {
        "dims": dims,
        "size": list(size),
        "n_frames": int(dim("T")),
        "frame_shape": [int(dim("Y")), int(dim("X"))],
        "channels": int(dim("C")),
        "pixel_type": str(getattr(czi, "pixel_type", "unknown")),
        "file_bytes": int(path.stat().st_size),
    }


def _read_sampled_stack(path: Path, *, windows: list[tuple[int, int]],
                        channel: int, frame_shape: tuple[int, int]) -> np.ndarray:
    """Decode only the declared source windows into a bounded float32 stack."""
    from aicspylibczi import CziFile

    indices = [frame for start, stop in windows for frame in range(start, stop)]
    h, w = map(int, frame_shape)
    stack = np.empty((len(indices), h, w), dtype=np.float32)
    czi = CziFile(str(path))
    for out_i, frame_i in enumerate(indices):
        image, _ = czi.read_image(T=int(frame_i), C=int(channel))
        frame = np.asarray(image).squeeze()
        while frame.ndim > 2:
            frame = frame[0]
        if frame.shape != (h, w):
            raise ValueError(
                f"frame {frame_i} has shape {frame.shape}, expected {(h, w)}")
        stack[out_i] = frame.astype(np.float32, copy=False)
    return stack


def _calibration(manifest: dict[str, Any]) -> tuple[float, float]:
    params = manifest["parameters"]
    for source in (
            manifest.get("effective_calibration"),
            params.get("effective_calibration")):
        if isinstance(source, dict):
            px = source.get("pixel_size_um")
            fi = source.get("frame_interval_s")
            if px is not None and fi is not None:
                return float(px), float(fi)
    px = params.get("pixel_size")
    fi = params.get("frame_interval")
    if px is None or fi is None:
        raise ValueError("manifest does not contain effective calibration")
    return float(px), float(fi)


def _roi_mask(manifest: dict[str, Any], frame_shape: tuple[int, int], *,
              allow_full_frame: bool) -> tuple[np.ndarray | None, str]:
    params = manifest["parameters"]
    vertices = params.get("roi_polygon")
    if vertices:
        from firefly.analysis.fa_roi import build_polygon_roi_mask

        mask, n_polygons = build_polygon_roi_mask(vertices, frame_shape)
        return mask, f"polygon_union:{n_polygons}"
    if str(params.get("roi_mode", "")).strip().lower() == "none":
        # This is not a fallback: the saved run explicitly analysed the whole
        # frame, so full-frame calibration preserves its declared population.
        return None, "full_frame:saved_roi_none"
    if allow_full_frame:
        return None, "full_frame:explicit_override"
    raise ValueError(
        "saved manifest has no polygon ROI; pass --allow-full-frame only when "
        "whole-frame calibration is scientifically intended")


def _estimate_quality(stack: np.ndarray, **kwargs):
    """Late-bound production call, kept separate for plan-only operation/tests."""
    from firefly.analysis.fa_localize import estimate_minmass

    return estimate_minmass(stack, **kwargs)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")
    os.replace(tmp, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        tmp = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _jsonable(value) for key, value in row.items()})
    os.replace(tmp, path)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return slug or "case"


def _case_plan(case: Case) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_manifest(case.manifest)
    movie = _inspect_czi(case.movie)
    windows = _source_windows(movie["n_frames"])
    params = manifest["parameters"]
    px, fi = _calibration(manifest)
    manifest_input_obj = manifest.get("input") or {}
    manifest_input = str(manifest_input_obj.get("path") or
                         params.get("file") or "")
    if params.get("roi_polygon"):
        planned_roi_scope = "polygon_union"
    elif str(params.get("roi_mode", "")).strip().lower() == "none":
        planned_roi_scope = "full_frame:saved_roi_none"
    else:
        planned_roi_scope = "missing_polygon_requires_explicit_override"
    plan = {
        "label": case.label,
        "movie": str(case.movie.resolve(strict=True)),
        "manifest": str(case.manifest.resolve(strict=True)),
        "manifest_input_path": manifest_input,
        "manifest_input_sha256": manifest_input_obj.get("sha256"),
        "manifest_input_matches_current_path": (
            Path(manifest_input).resolve(strict=False) ==
            case.movie.resolve(strict=True) if manifest_input else None),
        "movie_metadata": movie,
        "source_windows": [list(w) for w in windows],
        "sample_frames": int(sum(stop - start for start, stop in windows)),
        "pixel_size_um": px,
        "frame_interval_s": fi,
        "saved_minmass": manifest.get("resolved_minmass"),
        "saved_mode": params.get("minmass_mode"),
        "saved_backend": params.get("backend", "auto"),
        "saved_roi_mode": params.get("roi_mode"),
        "saved_roi_has_polygon": bool(params.get("roi_polygon")),
        "planned_validation_roi_scope": planned_roi_scope,
    }
    return plan, manifest


def _run_case(case: Case, plan: dict[str, Any], manifest: dict[str, Any], *,
              quality_floor: float, max_null_track_fraction: float,
              null_replicates: int, null_seed: int, workers: int,
              backend_override: str | None,
              allow_full_frame: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    params = manifest["parameters"]
    shape = tuple(int(v) for v in plan["movie_metadata"]["frame_shape"])
    windows = [tuple(map(int, w)) for w in plan["source_windows"]]
    channel = int(params.get("channel", 0))
    if channel < 0 or channel >= int(plan["movie_metadata"]["channels"]):
        raise ValueError(f"channel {channel} is outside the movie channel range")
    roi_mask, roi_scope = _roi_mask(
        manifest, shape, allow_full_frame=allow_full_frame)

    started = time.monotonic()
    stack = _read_sampled_stack(
        case.movie, windows=windows, channel=channel, frame_shape=shape)
    # Concatenating the source windows is safe here: production harvesting
    # resets frame numbers and linking independently for each 120-frame block.
    # With 4x120 frames, estimate_minmass reconstructs [0:120, ..., 360:480].
    backend = (str(params.get("backend") or "auto")
               if backend_override in (None, "saved")
               else str(backend_override))
    threshold, diagnostics = _estimate_quality(
        stack,
        diameter=int(params.get("diameter", 7)),
        percentile=64,
        backend=backend,
        sensitivity=str(params.get("minmass_sensitivity", "balanced")),
        bg_radius=int(params.get("bg_radius", 10)),
        bg_method=str(params.get("bg_method", "uniform_filter")),
        workers=int(workers),
        search_range=int(params.get("search_range", 3)),
        memory=int(params.get("memory", 5)),
        link_min_len=max(4, int(params.get("min_track_len", 8) or 8)),
        max_false_track_rate=float(max_null_track_fraction),
        mode="quality_first",
        quality_floor=float(quality_floor),
        roi_mask=roi_mask,
        pixel_size_um=float(plan["pixel_size_um"]),
        quality_null_replicates=int(null_replicates),
        quality_null_seed=int(null_seed),
    )
    elapsed = float(time.monotonic() - started)
    diagnostics = dict(diagnostics)
    diagnostics.update({
        "validation_label": case.label,
        "validation_source_movie": str(case.movie.resolve(strict=True)),
        "validation_source_manifest": str(case.manifest.resolve(strict=True)),
        "validation_source_windows": [list(w) for w in windows],
        "validation_sample_only": True,
        "validation_roi_scope": roi_scope,
        "validation_runtime_seconds": elapsed,
    })
    info = diagnostics.get("quality_info") or {}
    summary = {
        "label": case.label,
        "status": diagnostics.get("quality_status", "unresolved"),
        "reason": diagnostics.get("quality_reason"),
        "method": diagnostics.get("method"),
        "selected_minmass": float(threshold),
        "saved_minmass": plan.get("saved_minmass"),
        "quality_floor_assay": diagnostics.get("quality_floor_assay"),
        "quality_floor_effective": diagnostics.get("quality_floor_effective"),
        "null_track_fraction_ceiling": diagnostics.get(
            "quality_max_null_track_fraction"),
        "selected_null_track_fraction_upper": info.get(
            "null_good_fraction_upper"),
        "observed_good_tracks": info.get("observed_good_tracks"),
        "n_candidates_raw_full_frame": diagnostics.get(
            "n_candidates_raw_full_frame"),
        "n_candidates_raw_roi": diagnostics.get("n_candidates_raw_roi"),
        "n_candidates_positive_roi": diagnostics.get("n_candidates"),
        "sample_n_selected": diagnostics.get("sample_n_selected"),
        "sample_mean_per_frame": diagnostics.get("sample_mean_per_frame"),
        "sample_density_per_um2_frame": diagnostics.get(
            "sample_density_per_um2_frame"),
        "sample_window_cv": diagnostics.get("sample_window_cv"),
        "roi_fraction": diagnostics.get("quality_roi_fraction"),
        "backend_requested": backend,
        "backend_resolved": diagnostics.get("harvest_backend"),
        "runtime_seconds": elapsed,
    }
    return summary, diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded 4x120-frame quality-first calibration audit. "
            "No full analysis is performed and output is restricted to "
            "/private/tmp."))
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_HONOURS_ROOT,
        help="Honours source root used by the default four-case panel")
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="new/dedicated output directory below /private/tmp (required)")
    parser.add_argument(
        "--case", action="append", type=_parse_case,
        help=("custom case, repeatable: LABEL::MOVIE.czi::RUN_MANIFEST.json; "
              "supplying any custom case replaces the default panel"))
    parser.add_argument(
        "--plan-only", action="store_true",
        help="write run_plan.json after metadata checks; decode no image pixels")
    parser.add_argument("--quality-floor", type=float, default=0.16)
    parser.add_argument("--max-null-track-fraction", type=float, default=0.10)
    parser.add_argument("--null-replicates", type=int, default=3)
    parser.add_argument("--null-seed", type=int, default=20260805)
    parser.add_argument(
        "--workers", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument(
        "--backend", default="torch",
        help=("validation backend (default: torch, matching the new Drosophila "
              "preset); pass 'saved' to reuse each legacy manifest backend"))
    parser.add_argument(
        "--max-files", type=int, default=4,
        help="hard case-count bound (default 4; raise explicitly for custom panels)")
    parser.add_argument(
        "--allow-full-frame", action="store_true",
        help="allow a custom case without a saved polygon ROI")
    return parser


def _validate_args(parser: argparse.ArgumentParser,
                   args: argparse.Namespace) -> None:
    if not np.isfinite(args.quality_floor) or args.quality_floor <= 0:
        parser.error("--quality-floor must be positive and finite")
    if not (0 < args.max_null_track_fraction < 1):
        parser.error("--max-null-track-fraction must be between 0 and 1")
    if not (1 <= args.null_replicates <= 20):
        parser.error("--null-replicates must be between 1 and 20")
    if not (1 <= args.workers <= 64):
        parser.error("--workers must be between 1 and 64")
    if not (1 <= args.max_files <= 20):
        parser.error("--max-files must be between 1 and 20")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    cases = list(args.case) if args.case else core_panel(args.root)
    if len(cases) > int(args.max_files):
        parser.error(
            f"panel has {len(cases)} cases, exceeding --max-files={args.max_files}")

    source_roots = ([args.root] if not args.case else
                    [case.movie.parent for case in cases] +
                    [case.manifest.parent for case in cases])
    try:
        out = validate_output_dir(args.output_dir, source_roots=source_roots)
    except ValueError as exc:
        parser.error(str(exc))
    if out.exists() and not out.is_dir():
        parser.error(f"--output-dir exists and is not a directory: {out}")
    if out.exists() and any(out.iterdir()):
        parser.error(
            f"--output-dir must be new or empty: {out}")

    for case in cases:
        if not case.movie.is_file():
            parser.error(f"movie does not exist: {case.movie}")
        if case.movie.suffix.lower() != ".czi":
            parser.error(f"sampled harness currently accepts CZI movies: {case.movie}")
        if not case.manifest.is_file():
            parser.error(f"manifest does not exist: {case.manifest}")

    plans: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    try:
        for case in cases:
            plan, manifest = _case_plan(case)
            plans.append(plan)
            manifests.append(manifest)
    except Exception as exc:
        parser.error(f"source preflight failed: {exc}")

    out.mkdir(parents=True, exist_ok=True)
    run_plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_only": True,
        "full_analysis_permitted": False,
        "source_root": (str(args.root.resolve(strict=True))
                        if not args.case else None),
        "output_dir": str(out),
        "sampling_contract": {
            "windows": N_WINDOWS,
            "frames_per_window": FRAMES_PER_WINDOW,
            "maximum_frames_per_16k_case": N_WINDOWS * FRAMES_PER_WINDOW,
        },
        "quality_policy": {
            "mode": "quality_first",
            "assay_floor": args.quality_floor,
            "max_null_track_fraction": args.max_null_track_fraction,
            "null_replicates": args.null_replicates,
            "null_seed": args.null_seed,
            "workers": args.workers,
            "backend_override": args.backend,
        },
        "cases": plans,
    }
    _atomic_json(out / "run_plan.json", run_plan)
    if args.plan_only:
        print(f"Validated {len(cases)} cases; plan written to {out / 'run_plan.json'}")
        print("No image pixels were decoded and no full analysis was run.")
        return 0

    diag_dir = out / "diagnostics"
    diag_dir.mkdir()
    summaries: list[dict[str, Any]] = []
    for case, plan, manifest in zip(cases, plans, manifests):
        print(f"[{case.label}] sampled quality-first validation", flush=True)
        try:
            summary, diagnostics = _run_case(
                case, plan, manifest,
                quality_floor=args.quality_floor,
                max_null_track_fraction=args.max_null_track_fraction,
                null_replicates=args.null_replicates,
                null_seed=args.null_seed,
                workers=args.workers,
                backend_override=args.backend,
                allow_full_frame=args.allow_full_frame)
        except Exception as exc:
            summary = {
                "label": case.label, "status": "error",
                "reason": f"{type(exc).__name__}: {exc}"}
            diagnostics = {
                "validation_label": case.label,
                "validation_sample_only": True,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        summaries.append(summary)
        _atomic_json(diag_dir / f"{_slug(case.label)}.json", diagnostics)

    _atomic_csv(out / "summary.csv", summaries)
    _atomic_json(out / "summary.json", summaries)
    valid = sum(row.get("status") == "valid" for row in summaries)
    print(f"Sampled validation complete: {valid}/{len(summaries)} valid")
    print(f"Results: {out}")
    return 0 if valid == len(summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
