"""Redundant cross-correlation (RCC) drift correction.

Extracted from sptpalm_analysis.py (#7); re-exported there for compatibility.
"""
from __future__ import annotations

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from firefly.analysis.fa_constants import N_CPUS

import numpy as np
from scipy.signal import correlate as _correlate2d
from scipy.interpolate import interp1d


def correct_drift(locs, n_seg_frames=200, upsampling=4, smooth_sigma=1.5):
    """
    Reference-free drift correction via cross-correlation of localization
    density maps (simplified RCC approach; Wang et al. 2014, Nat Methods).

    The acquisition is divided into time segments.  A 2-D localization density
    histogram is built for each segment at ``upsampling``× the raw pixel
    resolution.  Consecutive histograms are cross-correlated (FFT) to measure
    the inter-segment drift.  The cumulative, Gaussian-smoothed drift trajectory
    is interpolated to per-frame resolution and subtracted from every
    localization.

    Applied *before* linking so that drift-corrected positions produce better
    trajectories.

    Parameters
    ----------
    locs          : DataFrame with 'x', 'y', 'frame' columns (in pixels)
    n_seg_frames  : target number of frames per time segment (default 200).
                    Smaller → finer time resolution but fewer localisations
                    per segment (noisier cross-correlation).
    upsampling    : density-map super-resolution factor.  upsampling=4 gives
                    ~25 nm accuracy at 0.1 µm/px (default 4).
    smooth_sigma  : Gaussian smoothing sigma in units of *segments* applied to
                    the raw drift trajectory before interpolation (default 1.5).

    Returns
    -------
    locs_corrected : DataFrame with corrected 'x' and 'y'
    drift_df       : DataFrame with columns ['frame', 'dx', 'dy'] (pixels)
    """
    if len(locs) == 0:
        return locs.copy(), pd.DataFrame({"frame": [0], "dx": [0.0], "dy": [0.0]})

    x = locs["x"].values.astype(np.float64)
    y = locs["y"].values.astype(np.float64)
    f = locs["frame"].values.astype(int)

    n_frames   = int(f.max()) + 1
    n_segments = max(4, int(np.ceil(n_frames / n_seg_frames)))
    n_segments = min(n_segments, max(2, len(locs) // 10))  # need ≥10 locs/seg

    print(f"  Drift correction : {n_segments} segments "
          f"(~{n_frames // n_segments} frames each, upsampling={upsampling})")

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    W = max(int((x_max - x_min) * upsampling) + 1, 16)
    H = max(int((y_max - y_min) * upsampling) + 1, 16)

    seg_bounds  = np.linspace(0, n_frames, n_segments + 1).astype(int)
    seg_centers = (seg_bounds[:-1] + seg_bounds[1:]) / 2.0

    # ── Build upsampled density maps ──────────────────────────────────────────
    density_maps = []
    seg_counts   = []
    for i in range(n_segments):
        sel = (f >= seg_bounds[i]) & (f < seg_bounds[i + 1])
        seg_counts.append(int(sel.sum()))
        dm  = np.zeros((H, W), dtype=np.float32)
        if sel.sum() > 0:
            xi = np.clip(((x[sel] - x_min) * upsampling).astype(int), 0, W - 1)
            yi = np.clip(((y[sel] - y_min) * upsampling).astype(int), 0, H - 1)
            np.add.at(dm, (yi, xi), 1.0)
            dm = gaussian_filter(dm, sigma=upsampling * 0.7)   # spread spots
        density_maps.append(dm)

    print(f"  Localisations/segment: min {min(seg_counts):,}, "
          f"max {max(seg_counts):,}")

    # ── Cross-correlate ALL pairs (i, j) → solve cumulative drift ─────────────
    # This is the redundant cross-correlation (RCC) algorithm of Wang et al.
    # 2014 (Nat. Methods).  Instead of relying only on consecutive pairs, we
    # measure the inter-segment shift Δ_{ij} for every pair (i, j) with i<j
    # and then solve the over-determined linear system
    #
    #     drift[j] − drift[i] = Δ_{ij}      for all valid pairs
    #
    # by least-squares.  Drift[0] is fixed at zero (gauge fixing).  The
    # redundancy averages out cross-correlation noise far better than the
    # consecutive-only chain, and is robust to any single bad pair (e.g. a
    # segment with too few localisations).
    #
    # Performance note:  scipy.signal.correlate(method="fft") re-FFTs both
    # density maps on every pair call, so an N-segment run does ~N(N-1)
    # FFTs.  We precompute rfft2 of each (zero-padded) map ONCE and just
    # run an IFFT per pair — quadratic-cost FFT work collapses to linear,
    # plus the IFFT loop parallelises trivially via threads.
    from scipy.fft import rfft2 as _rfft2, irfft2 as _irfft2, \
                          next_fast_len as _next_fast_len
    pad_H = _next_fast_len(2 * H - 1)
    pad_W = _next_fast_len(2 * W - 1)
    fft_maps = [_rfft2(dm, s=(pad_H, pad_W)) for dm in density_maps]

    pair_indices = [(i, j) for i in range(n_segments)
                    for j in range(i + 1, n_segments)
                    if seg_counts[i] >= 5 and seg_counts[j] >= 5]

    def _pair_shift(i, j):
        # Cross-correlation r[τ] = Σ a[k+τ] b[k]  via  IFFT(F_a · conj(F_b))
        cross = _irfft2(fft_maps[i] * np.conj(fft_maps[j]),
                        s=(pad_H, pad_W))
        # Zero-lag at index 0; positive shifts up to (H-1, W-1) sit at low
        # indices, negative shifts wrap to the end.  Re-centre by treating
        # any index beyond half-extent as negative.
        peak = int(np.argmax(cross))
        py, px = divmod(peak, pad_W)
        if py >= pad_H // 2: py -= pad_H
        if px >= pad_W // 2: px -= pad_W
        return i, j, float(px), float(py)

    A_rows_x, A_rows_y = [], []
    b_x, b_y = [], []
    if pair_indices:
        with ThreadPoolExecutor(max_workers=N_CPUS) as _exe:
            for i, j, dx_pair, dy_pair in _exe.map(
                    lambda ij: _pair_shift(*ij), pair_indices):
                row = np.zeros(n_segments)
                row[i], row[j] = -1.0, 1.0
                A_rows_x.append(row); b_x.append(dx_pair)
                A_rows_y.append(row); b_y.append(dy_pair)

    if not A_rows_x:
        # Fallback: zero drift
        dx_cum = np.zeros(n_segments)
        dy_cum = np.zeros(n_segments)
    else:
        # Add gauge-fixing row: drift[0] = 0 (heavy weight)
        gauge = np.zeros(n_segments); gauge[0] = 1.0
        A = np.vstack(A_rows_x + [gauge * 1e3])
        bx = np.append(np.array(b_x), 0.0)
        by = np.append(np.array(b_y), 0.0)
        dx_cum, *_ = np.linalg.lstsq(A, bx, rcond=None)
        dy_cum, *_ = np.linalg.lstsq(A, by, rcond=None)

    # Smooth then convert to localization pixels
    dx_sm = gaussian_filter1d(dx_cum, sigma=smooth_sigma) / upsampling
    dy_sm = gaussian_filter1d(dy_cum, sigma=smooth_sigma) / upsampling

    # Zero-centre so overall position is preserved
    dx_sm -= dx_sm.mean()
    dy_sm -= dy_sm.mean()

    rng_x, rng_y = float(np.ptp(dx_sm)), float(np.ptp(dy_sm))
    print(f"  Drift range  x={rng_x:.3f} px  y={rng_y:.3f} px")

    # ── Interpolate to every frame ────────────────────────────────────────────
    frame_arr = np.arange(n_frames, dtype=float)
    ix = interp1d(seg_centers, dx_sm, kind="linear",
                  bounds_error=False, fill_value=(dx_sm[0], dx_sm[-1]))
    iy = interp1d(seg_centers, dy_sm, kind="linear",
                  bounds_error=False, fill_value=(dy_sm[0], dy_sm[-1]))
    drift_x = ix(frame_arr)
    drift_y = iy(frame_arr)

    # ── Subtract from localisations ────────────────────────────────────────────
    locs_out = locs.copy()
    fi       = np.clip(f, 0, n_frames - 1)
    locs_out["x"] = x - drift_x[fi]
    locs_out["y"] = y - drift_y[fi]

    drift_df = pd.DataFrame({"frame": frame_arr.astype(int),
                             "dx": drift_x, "dy": drift_y})
    return locs_out, drift_df
