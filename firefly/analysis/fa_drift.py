"""Redundant cross-correlation (RCC) drift correction.

Extracted from sptpalm_analysis.py (#7); re-exported there for compatibility.
"""
from __future__ import annotations

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from firefly.analysis.fa_constants import N_CPUS

import numpy as np
from scipy.interpolate import interp1d


def correct_drift(locs, n_seg_frames=200, upsampling=4, smooth_sigma=1.5,
                  max_shift_frac=0.30, outlier_k=6.0, outlier_tol_px=6.0):
    """
    Reference-free drift correction via cross-correlation of localization
    density maps (simplified RCC approach; Wang et al. 2014,
    Opt. Express 22(13):15982, DOI 10.1364/OE.22.015982).

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
    max_shift_frac: cross-correlation peak search is restricted to inter-segment
                    shifts within ``max_shift_frac`` of the density-map extent
                    along each axis (default 0.30).  This rejects gross spurious
                    / wrap-around correlation peaks on sparse or poorly-overlapping
                    segments (which otherwise produce non-physical drifts like
                    150 px on a 512 px frame) while leaving real drift — always
                    far inside this window — untouched.  Scales with the data so
                    larger structures permit larger absolute drift.
    outlier_k     : robustness factor for inconsistent-pair rejection.  After the
                    redundant cross-correlation least-squares solve, segment pairs
                    whose measured shift disagrees with the global solution by more
                    than ``max(outlier_tol_px, outlier_k · 1.4826 · MAD)``
                    (in upsampled px) are dropped and the system is re-solved (one
                    IRLS pass).  On clean data all residuals are tiny → nothing is
                    dropped → the result is identical to the un-guarded solve.
    outlier_tol_px: absolute residual floor (upsampled px) for the rejection rule,
                    so clean data with near-zero MAD never rejects good pairs.

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
    # 2014 (Opt. Express 22(13):15982).  Instead of relying only on consecutive pairs, we
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

    # ── Plausible-shift search mask ───────────────────────────────────────────
    # The cross-correlation lives on a (pad_H × pad_W) wrapped grid: index k on
    # an axis of length L means lag k (k < L/2) or k−L (k ≥ L/2).  Restricting
    # argmax to lags within ±(max_shift_frac · extent) per axis prevents a
    # spurious / wrap-around peak on a sparse or poorly-overlapping segment from
    # being selected as the drift (the root cause of 150 px artefacts).  Real
    # drift sits far inside this window, so on well-behaved data the masked
    # argmax returns the SAME index as the un-masked one — byte-identical output.
    _lag0 = np.where(np.arange(pad_H) < pad_H // 2,
                     np.arange(pad_H), np.arange(pad_H) - pad_H)
    _lag1 = np.where(np.arange(pad_W) < pad_W // 2,
                     np.arange(pad_W), np.arange(pad_W) - pad_W)
    _R_y = max(1, int(round(max_shift_frac * H)))
    _R_x = max(1, int(round(max_shift_frac * W)))
    search_mask = ((np.abs(_lag0) <= _R_y)[:, None]
                   & (np.abs(_lag1) <= _R_x)[None, :])

    def _pair_shift(i, j):
        # Cross-correlation r[τ] = Σ map_j[k+τ]·map_i[k]  via  IFFT(F_j · conj(F_i)).
        # The peak τ is the shift of the LATER segment j's density relative to the
        # EARLIER segment i — i.e. (drift_j − drift_i).  Combined with the row
        # encoding `drift[j] − drift[i] = τ` and the gauge `drift[0]=0`, the solved
        # `drift_cum` is then the TRUE sample drift, so `locs_out = x − drift`
        # REMOVES the motion.
        #
        # SIGN BUG (fixed): the previous `IFFT(F_i · conj(F_j))` peaks at
        # (drift_i − drift_j) = −τ, so the solver returned −(true drift) and the
        # subtraction DOUBLED the drift instead of removing it.  The old test only
        # checked the recovered range (max−min), which a sign flip also satisfies,
        # so it slipped through — now locked by test_correct_drift_recovers_sign.
        cross = _irfft2(fft_maps[j] * np.conj(fft_maps[i]),
                        s=(pad_H, pad_W))
        # Search only the plausible-shift window; everything else is masked to
        # −∞ so it can never win the argmax.
        peak = int(np.argmax(np.where(search_mask, cross, -np.inf)))
        py, px = divmod(peak, pad_W)
        if py >= pad_H // 2: py -= pad_H
        if px >= pad_W // 2: px -= pad_W
        return i, j, float(px), float(py)

    pairs = []          # (i, j, dx, dy) in upsampled px
    if pair_indices:
        with ThreadPoolExecutor(max_workers=N_CPUS) as _exe:
            for i, j, dx_pair, dy_pair in _exe.map(
                    lambda ij: _pair_shift(*ij), pair_indices):
                pairs.append((int(i), int(j), float(dx_pair), float(dy_pair)))

    def _solve(pair_list):
        """Gauge-fixed least-squares (drift[0]=0) over the given pair shifts."""
        if not pair_list:
            return np.zeros(n_segments), np.zeros(n_segments)
        rows, bx, by = [], [], []
        for (i, j, dx, dy) in pair_list:
            row = np.zeros(n_segments)
            row[i], row[j] = -1.0, 1.0
            rows.append(row); bx.append(dx); by.append(dy)
        gauge = np.zeros(n_segments); gauge[0] = 1.0
        A  = np.vstack(rows + [gauge * 1e3])
        bxv = np.append(np.array(bx), 0.0)
        byv = np.append(np.array(by), 0.0)
        dxc, *_ = np.linalg.lstsq(A, bxv, rcond=None)
        dyc, *_ = np.linalg.lstsq(A, byv, rcond=None)
        return dxc, dyc

    dx_cum, dy_cum = _solve(pairs)

    # ── Robust pair rejection (IRLS) ──────────────────────────────────────────
    # Use the RCC redundancy: a good pair's measured shift agrees with the global
    # solution (drift[j]−drift[i]).  Drop pairs whose residual exceeds
    # max(outlier_tol_px, k·1.4826·MAD) and re-solve.  On clean data residuals
    # are ~0 → threshold is the floor → nothing is dropped → identical curve.
    n_rejected = 0
    if len(pairs) > n_segments:
        for _ in range(2):                 # at most two refinement passes
            resid = np.array([
                np.hypot((dx_cum[j] - dx_cum[i]) - dx,
                         (dy_cum[j] - dy_cum[i]) - dy)
                for (i, j, dx, dy) in pairs])
            med = float(np.median(resid))
            mad = float(np.median(np.abs(resid - med)))
            thresh = max(float(outlier_tol_px), float(outlier_k) * 1.4826 * mad)
            keep = resid <= thresh
            if keep.all() or int(keep.sum()) < n_segments:
                break
            n_rejected += int((~keep).sum())
            pairs = [p for p, k in zip(pairs, keep) if k]
            dx_cum, dy_cum = _solve(pairs)
    if n_rejected:
        print(f"  Drift: rejected {n_rejected} inconsistent segment pair(s) "
              f"(robust RCC)")

    # ── Unconstrained-segment guard ───────────────────────────────────────────
    # A segment that appears in NO surviving pair (too sparse to enter
    # pair_indices, or all its pairs were rejected above) has an all-zero column
    # in the design matrix, so lstsq leaves its drift at the min-norm 0.  The
    # smoothing + interpolation below would then spread that as a spurious
    # "drift snaps back to zero" spike around that segment's time.  Interpolate
    # such segments from their nearest constrained neighbours instead (segment 0
    # is the fixed gauge anchor, so it counts as constrained at 0).
    constrained = np.zeros(n_segments, dtype=bool)
    for (i, j, _dx, _dy) in pairs:
        constrained[i] = True
        constrained[j] = True
    constrained[0] = True
    if constrained.any() and not constrained.all():
        seg_idx = np.arange(n_segments)
        ci = seg_idx[constrained]
        dx_cum = np.interp(seg_idx, ci, dx_cum[constrained])
        dy_cum = np.interp(seg_idx, ci, dy_cum[constrained])
        print(f"  Drift: interpolated {int((~constrained).sum())} "
              f"unconstrained segment(s) from neighbours")

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
