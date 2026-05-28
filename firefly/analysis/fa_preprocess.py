"""Per-frame preprocessing (background subtraction / bandpass) and
auto-thresholding for the FIREFLY pipeline.

Extracted from sptpalm_analysis.py (#7); re-exported there for compatibility.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from firefly.analysis.fa_constants import N_CPUS, _tqdm

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter, gaussian_filter1d
from skimage import filters, exposure


def _preprocess_fast(frame, bg_radius=50, sigma=1.0):
    """
    Fast background subtraction using uniform_filter.
    ~1700x faster than rolling_ball with comparable results for PALM data.
    """
    bg        = uniform_filter(frame, size=int(bg_radius * 2 + 1))
    corrected = np.clip(frame - bg, 0, None)
    smoothed  = filters.gaussian(corrected, sigma=sigma, preserve_range=True)
    mn, mx    = smoothed.min(), smoothed.max()
    if mx > mn:
        smoothed = (smoothed - mn) / (mx - mn)
    return smoothed.astype(np.float32)


def _preprocess_rolling(frame, bg_radius=50, sigma=1.0):
    """Legacy rolling-ball background subtraction (slow but thorough)."""
    from skimage.restoration import rolling_ball
    bg        = rolling_ball(frame, radius=bg_radius)
    corrected = np.clip(frame - bg, 0, None)
    smoothed  = filters.gaussian(corrected, sigma=sigma, preserve_range=True)
    mn, mx    = smoothed.min(), smoothed.max()
    if mx > mn:
        smoothed = (smoothed - mn) / (mx - mn)
    return smoothed.astype(np.float32)


def preprocess_stack(stack, bg_radius=50, bg_method="uniform_filter",
                     workers=N_CPUS):
    n = len(stack)
    fn = _preprocess_fast if bg_method == "uniform_filter" else _preprocess_rolling

    print(f"  Background method : {bg_method}")
    print(f"  Workers           : {workers} / {N_CPUS} CPU cores")
    t0 = time.perf_counter()

    if workers == 1:
        processed = [fn(f, bg_radius) for f in
                     _tqdm(stack, desc="  Preprocessing", unit="fr", ncols=70)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as _exe:
            _futs = [_exe.submit(fn, f, bg_radius) for f in stack]
            processed = [_f.result() for _f in
                         _tqdm(_futs, desc="  Preprocessing", unit="fr", ncols=70)]

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s  ({elapsed/n*1000:.1f} ms/frame)")
    return np.stack(processed)


def auto_threshold(image_norm, method="auto"):
    """
    Determine an intensity threshold automatically from the normalised
    mean projection using scikit-image thresholding algorithms.

    Parameters
    ----------
    image_norm : float array (Y, X), values in [0, 1]
    method     : "auto"     — tries otsu, li, triangle; picks best for sptPALM
                 "otsu"     — maximises inter-class variance
                 "li"       — minimises cross-entropy (good for sparse cells)
                 "triangle" — geometric method, good for large dark backgrounds

    Returns
    -------
    threshold : float in [0, 1]
    chosen    : str — which method was used
    all_vals  : dict — thresholds from all three methods for reference
    """
    from skimage.filters import threshold_otsu, threshold_li, threshold_triangle

    results = {}
    try:    results["otsu"]     = float(threshold_otsu(image_norm))
    except Exception: results["otsu"]     = None
    try:    results["li"]       = float(threshold_li(image_norm))
    except Exception: results["li"]       = None
    try:    results["triangle"] = float(threshold_triangle(image_norm))
    except Exception: results["triangle"] = None

    print(f"  Auto-threshold candidates:")
    for name, val in results.items():
        print(f"    {name:<10} : {val:.4f}" if val is not None
              else f"    {name:<10} : failed")

    if method != "auto":
        chosen = method
        val    = results.get(method)
        if val is None:
            print(f"  WARNING: {method} failed, falling back to otsu")
            chosen = "otsu"
            val    = results["otsu"]
    else:
        # For sptPALM the cell typically occupies < 30% of the frame.
        # Triangle handles large dark backgrounds best in this scenario.
        # Fall back to Li then Otsu if triangle is unavailable.
        for preferred in ("triangle", "li", "otsu"):
            if results[preferred] is not None:
                chosen = preferred
                val    = results[preferred]
                break

    print(f"  Selected method : {chosen}  ->  threshold = {val:.4f}")
    return val, chosen, results
