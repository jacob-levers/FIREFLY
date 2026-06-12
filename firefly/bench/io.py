"""Disk I/O for the benchmark harness. The only optional dependency (tifffile)
is isolated here behind HAS_TIFFFILE so the rest of the package imports safely on
a tifffile-less CI."""
from __future__ import annotations

import os
import json

import numpy as np

try:
    import tifffile as _tifffile
    HAS_TIFFFILE = True
except Exception:                       # pragma: no cover
    _tifffile = None
    HAS_TIFFFILE = False


def write_tiff(stack: np.ndarray, path: str) -> str:
    """Write the simulated stack as an ImageJ-readable uint16 TIFF so FIREFLY,
    TrackMate and palmTRACER can all open it."""
    if not HAS_TIFFFILE:
        raise RuntimeError("tifffile is not installed — cannot write the stack TIFF.")
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    _tifffile.imwrite(path, np.clip(stack, 0, 65535).astype(np.uint16), imagej=True)
    return path


def write_gt_csvs(sim_result, out_dir: str) -> dict:
    """Write ground-truth localisation + track tables (+ meta JSON)."""
    os.makedirs(out_dir, exist_ok=True)
    loc_p = os.path.join(out_dir, "ground_truth_locs.csv")
    trk_p = os.path.join(out_dir, "ground_truth_tracks.csv")
    meta_p = os.path.join(out_dir, "ground_truth_meta.json")
    sim_result.gt_locs.to_csv(loc_p, index=False)
    sim_result.gt_tracks.to_csv(trk_p, index=False)
    with open(meta_p, "w", encoding="utf-8") as fh:
        json.dump(sim_result.meta, fh, indent=2, default=float)
    return {"locs": loc_p, "tracks": trk_p, "meta": meta_p}
