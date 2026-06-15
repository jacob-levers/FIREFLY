"""Load an EXTERNAL ground-truth dataset (an image stack + a per-localisation GT
table) into the bench `SimResult` shape, so the engine-comparison driver can
score the detectors against ground truth FIREFLY did NOT generate.

Handles the common SMLM / SPT ground-truth CSV conventions — the EPFL SMLM
challenge, ThunderSTORM, the rkurre SPT-Simulator, etc.: a ``frame`` column plus
x / y in nm or pixels, with optional sigma / intensity / background.  Coordinates
are converted to PIXELS (the bench schema); the pixel size is taken from
``pixel_size_um`` or, when omitted, inferred from the field extent.

These datasets are per-localisation (no track IDs), so they score DETECTION
accuracy (F1 / Jaccard / RMSE vs CRLB) — exactly the detector-engine question.
``gt_tracks`` is left empty (tracking metrics are N/A for a localisation set).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Lower-cased, unit-suffix-stripped column name -> canonical key.
_ALIASES = {
    "frame": "frame", "t": "frame", "slice": "frame", "plane": "frame",
    "x": "x", "xnano": "x", "x_nm": "x", "x_pix": "x", "position x": "x",
    "y": "y", "ynano": "y", "y_nm": "y", "y_pix": "y", "position y": "y",
    "sigma": "sigma", "intensity": "intensity", "photons": "intensity",
    "background": "background", "bg": "background", "offset": "background",
}


def _norm(col: str) -> str:
    c = str(col).strip().strip('"').lower()
    c = c.split("[")[0].split("(")[0].strip()   # drop "[nm]" / "(px)" suffixes
    return c


def _is_nm(col: str) -> bool:
    c = str(col).lower()
    return "nm" in c or "nano" in c


def load_gt_dataset(tif_path, gt_csv_path, *, pixel_size_um=None,
                    frame_interval_s=0.02, frame_offset=None):
    """Return a `SimResult(stack, gt_locs, gt_tracks=empty, meta)` for an external
    image stack + per-localisation ground-truth CSV.  Reuse it with
    `compare_engines` / `evaluate` to score the detectors against this GT."""
    import tifffile
    from firefly.bench.simulator import SimResult

    stack = tifffile.imread(tif_path)
    if stack.ndim == 2:
        stack = stack[None]
    T, H, W = stack.shape

    gt = pd.read_csv(gt_csv_path)
    canon = {}
    for col in gt.columns:
        key = _ALIASES.get(_norm(col))
        if key and key not in canon:
            canon[key] = col
    for need in ("frame", "x", "y"):
        if need not in canon:
            raise ValueError(
                f"GT CSV missing a '{need}' column; got {list(gt.columns)}")

    frame = gt[canon["frame"]].to_numpy(float).astype(int)
    if frame_offset is None:                       # 1-indexed → 0; 0-indexed → 0
        frame_offset = -int(frame.min())
    frame = frame + frame_offset

    x = gt[canon["x"]].to_numpy(float)
    y = gt[canon["y"]].to_numpy(float)
    in_nm = _is_nm(canon["x"])
    inferred_px = False
    if pixel_size_um is None:
        if in_nm:
            # Infer nm/px from the field extent — UNRELIABLE: simulators often
            # place emitters slightly OUTSIDE the camera FOV, so the extent
            # OVERESTIMATES the pixel size and silently misaligns the GT (the
            # exact failure on the SPT-Simulator set: 106.7 inferred vs the true
            # 100).  Always prefer passing the dataset's real pixel_size_um.
            pixel_size_um = max(np.nanmax(x) / W, np.nanmax(y) / H) / 1000.0
            inferred_px = True
        else:
            pixel_size_um = 0.106                   # already px; size only scales RMSE-nm
    px_nm = pixel_size_um * 1000.0
    if in_nm:
        x = x / px_nm
        y = y / px_nm
    if inferred_px:
        print(f"  WARNING: pixel size INFERRED from the GT field extent as "
              f"{px_nm:.1f} nm/px — this is unreliable (ground truth often extends "
              f"beyond the camera FOV).  Pass pixel_size_um / --pixel-size with the "
              f"dataset's true value to avoid a coordinate MISALIGNMENT.")

    photons = (gt[canon["intensity"]].to_numpy(float) if "intensity" in canon
               else np.ones(len(gt)))
    gt_locs = pd.DataFrame({"frame": frame, "x": x, "y": y, "photons": photons})
    n0 = len(gt_locs)
    # Keep only GT inside the frame range AND the image FOV — emitters the
    # simulator placed outside the camera bounds are not detectable, so counting
    # them as missed detections would unfairly depress recall.
    gt_locs = gt_locs[(gt_locs["frame"] >= 0) & (gt_locs["frame"] < T)
                      & (gt_locs["x"] >= 0) & (gt_locs["x"] < W)
                      & (gt_locs["y"] >= 0) & (gt_locs["y"] < H)].reset_index(drop=True)
    if n0 - len(gt_locs):
        print(f"  Dropped {n0 - len(gt_locs):,} ground-truth localisation(s) "
              f"outside the {W}x{H} FOV / frame range (not detectable).")

    sigma_nm = (float(gt[canon["sigma"]].median()) if "sigma" in canon
                else 0.21 * 660.0 / 1.4)            # ~Gaussian-PSF default
    meta = {
        "pixel_size_um": float(pixel_size_um),
        "frame_interval_s": float(frame_interval_s),
        "psf_sigma_px": max(0.5, sigma_nm / px_nm),
        "photons_per_emitter": float(np.median(photons)) if len(photons) else 500.0,
        "bg_photons": (float(gt[canon["background"]].median())
                       if "background" in canon else 1.0),
    }
    gt_tracks = pd.DataFrame(columns=["frame", "x", "y", "particle", "population"])
    return SimResult(stack=stack.astype(np.float32), gt_locs=gt_locs,
                     gt_tracks=gt_tracks, meta=meta)
