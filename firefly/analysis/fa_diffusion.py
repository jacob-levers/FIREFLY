"""MSD / diffusion fits, motion classification, JDD, dwell times,
MSS, turning angles, mobile fraction, and summary helpers.

Extracted from sptpalm_analysis.py (#7); re-exported there for compatibility.
"""
from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from firefly.analysis.fa_constants import N_CPUS, _tqdm, safe_process_workers


def msd_linear(t, D, offset):
    return 4 * D * t + offset


def msd_anomalous(t, D, alpha, offset):
    """2D MSD with an anomalous exponent and a localisation-error floor:

        MSD(t) = 4·D·t^alpha + offset

    `offset` is the static localisation-error term (≈ 4·sigma²); it lifts the
    whole curve and must be modelled jointly with `alpha`, otherwise a plain
    log-log slope of the raw MSD reads alpha < 1 for genuinely Brownian (and
    especially slow) particles — the offset dominates the first lags and
    flattens the log-log curve.
    """
    return 4 * D * t ** alpha + offset


ALPHA_THRESHOLDS_DEFAULT = (0.5, 0.9, 1.1)


MOBILE_D_THRESHOLD_DEFAULT = 0.05


def classify_motion(alpha, thresholds=ALPHA_THRESHOLDS_DEFAULT):
    """Classify a track by its anomalous exponent α.

    thresholds = (t_immobile, t_confined, t_directed):
        α  <  t_immobile   → "Immobile"
        t_immobile  ≤ α  <  t_confined → "Confined"
        t_confined  ≤ α  <  t_directed → "Brownian"
        α  ≥  t_directed   → "Directed"
    """
    t_imm, t_conf, t_dir = thresholds
    if   alpha < t_imm:  return "Immobile"
    elif alpha < t_conf: return "Confined"
    elif alpha < t_dir:  return "Brownian"
    else:                return "Directed"


def _msd_and_fit_one(xy_um, frames, pid, lag_times, max_lagtime, n_fit,
                     alpha_thresholds=ALPHA_THRESHOLDS_DEFAULT):
    """
    Compute per-track MSD array AND fit D + alpha in a single pass.

    Uses actual frame numbers (not row indices) so that gaps in a trajectory
    caused by memory-linking do not inflate the MSD.  Only pairs of positions
    whose frame difference exactly equals the requested lag are included.
    """
    msd_vals = np.full(max_lagtime, np.nan)
    n_pts = len(xy_um)
    # Fast path: a gapless track (consecutive frame numbers — the common case,
    # and the ONLY case when memory=0) needs no per-lag frame-difference mask,
    # so we skip that work and slice directly.  Numerically identical to the
    # masked path below (which still handles memory-bridged gaps).
    gapless = (n_pts >= 2
               and int(frames[-1] - frames[0]) == n_pts - 1)
    if gapless:
        x = xy_um[:, 0]; y = xy_um[:, 1]
        for lag_idx, lag in enumerate(range(1, max_lagtime + 1)):
            if lag >= n_pts:
                break
            dx = x[lag:] - x[:-lag]
            dy = y[lag:] - y[:-lag]
            msd_vals[lag_idx] = np.mean(dx * dx + dy * dy)
    else:
        for lag_idx, lag in enumerate(range(1, max_lagtime + 1)):
            if lag >= n_pts:
                break
            # Only use pairs where the actual frame separation equals lag
            frame_diff = frames[lag:] - frames[:-lag]
            valid      = frame_diff == lag
            if valid.sum() > 0:
                d = xy_um[lag:][valid] - xy_um[:-lag][valid]
                msd_vals[lag_idx] = np.mean(d[:, 0] ** 2 + d[:, 1] ** 2)

    # Fit using the first n_fit lag times.  ONE consistent model —
    #     MSD(t) = 4·D·t^alpha + offset
    # — so the anomalous exponent (alpha) and the diffusion coefficient (D)
    # come from the SAME curve.  Crucially this fits the localisation-error
    # floor (offset == 4·sigma²) jointly: a naive log-log slope of the raw MSD
    # ignores that floor and biases alpha DOWN at short lags, mislabelling slow
    # Brownian tracks as confined/immobile.  offset is constrained ≥ 0 (it is a
    # squared quantity).
    t   = lag_times[:n_fit]
    m   = msd_vals[:n_fit]
    ok  = np.isfinite(m) & (m > 0)
    D = alpha = np.nan
    msd0 = np.nan        # localisation-error offset (4·sigma²) == PALM-Tracer "MSD(0)"
    mse  = np.nan        # mean squared residual of the fit
    immobile = False     # set when alpha can't be measured (jitter-dominated track)
    n_ok = int(ok.sum())
    t_ok, m_ok = t[ok], m[ok]
    if n_ok >= 4:
        # Seed D and offset from a quick linear (alpha=1) fit; seed alpha=1.
        try:
            slope, intercept = np.polyfit(t_ok, m_ok, 1)
            d_seed   = max(slope / 4.0, 1e-6)
            off_seed = max(intercept, 0.0)
        except Exception:
            d_seed, off_seed = 0.01, max(0.0, float(m_ok[0]))
        try:
            # alpha upper bound 2.0 — physically alpha ∈ [0, 2] (2 = ballistic);
            # anything above is non-physical and only ever appears as a fitting
            # artefact.
            popt, _ = curve_fit(msd_anomalous, t_ok, m_ok,
                                p0=[d_seed, 1.0, off_seed],
                                bounds=([0, 0, 0], [np.inf, 2.0, np.inf]),
                                maxfev=5000)
            D, alpha, msd0 = float(popt[0]), float(popt[1]), float(popt[2])
            _resid = m_ok - msd_anomalous(t_ok, *popt)
            mse = float(np.mean(_resid ** 2))
            # ── Identifiability guard ──────────────────────────────────────
            # For a near-immobile / jitter-dominated track the measured MSD is
            # essentially the flat localisation floor (offset): the dynamic
            # 4·D·t^alpha term ≈ 0, so alpha is UNCONSTRAINED and curve_fit
            # parks it at a bound — the unphysical spike pinned at the maximum
            # that an alpha histogram shows as a "wall".  When alpha is pinned
            # at a bound, or the dynamic rise is a negligible fraction of the
            # MSD across the fit window, alpha simply cannot be measured: drop
            # it (NaN) and treat the track as Immobile (classified by its
            # displacement, not by a meaningless exponent).  Genuinely mobile
            # tracks (dynamics well above the floor) are unaffected.
            t_hi     = float(t_ok[-1])
            dyn      = 4.0 * D * (t_hi ** alpha)          # moving part at the longest fit lag
            total    = dyn + max(msd0, 0.0)               # ≈ MSD(t_hi)
            dyn_frac = (dyn / total) if total > 0 else 0.0
            if alpha <= 1e-3 or alpha >= 2.0 - 1e-3 or dyn_frac < 0.10:
                alpha = np.nan
                immobile = True
        except Exception:
            pass
    if not np.isfinite(D) and not immobile and n_ok >= 3:
        # Fallback for very short tracks (or a non-converging joint fit): the
        # legacy two-step estimate (linear D + log-log alpha).  Less accurate
        # near the localisation floor but always returns something.
        try:    alpha = float(np.polyfit(np.log(t_ok), np.log(m_ok), 1)[0])
        except Exception: pass
        try:
            popt, _ = curve_fit(msd_linear, t_ok, m_ok, p0=[0.01, 0],
                                bounds=([0, -np.inf], [np.inf, np.inf]),
                                maxfev=2000)
            D = float(popt[0])
            msd0 = float(popt[1])
            _resid = m_ok - msd_linear(t_ok, *popt)
            mse = float(np.mean(_resid ** 2))
        except Exception: pass

    # Motion class: a measurable alpha → threshold classification; an
    # unmeasurable alpha on a jitter-dominated track → "Immobile"; a complete
    # fit failure → "Unknown".
    if np.isfinite(alpha):
        motion = classify_motion(alpha, alpha_thresholds)
    elif immobile:
        motion = "Immobile"
    else:
        motion = "Unknown"

    # Two distinct radial-spread metrics, both useful and named explicitly:
    #   mean_radial_displacement_um  = ⟨|r − r̄|⟩       (1st moment)
    #   radius_of_gyration_um        = √⟨|r − r̄|²⟩    (RMS, the standard Rg)
    centroid    = xy_um.mean(axis=0)
    sq_dists    = np.sum((xy_um - centroid) ** 2, axis=1)
    mean_radial = float(np.mean(np.sqrt(sq_dists)))
    rg          = float(np.sqrt(np.mean(sq_dists)))

    # Localisation precision from the fitted MSD offset.  Static localisation
    # error adds a constant 4·sigma² to the 2D MSD (sigma = 1D per-axis
    # precision), which is exactly the `offset` term of the joint fit.  So
    # sigma = sqrt(MSD0 / 4); report it in nm.  This is an always-available,
    # per-track estimate of the effective localisation precision — and because
    # the offset is modelled jointly, the fitted D is already free of this
    # static-error inflation.  NaN when the offset isn't a usable positive.
    if np.isfinite(msd0) and msd0 > 0:
        loc_sigma_nm = float(np.sqrt(msd0 / 4.0) * 1000.0)
    else:
        loc_sigma_nm = np.nan

    return pid, msd_vals, dict(particle=pid, D=D, alpha=alpha, motion=motion,
                               MSD0=msd0, MSE=mse, loc_sigma_nm=loc_sigma_nm,
                               mean_radial_displacement_um=mean_radial,
                               radius_of_gyration_um=rg)


def _require_positive_finite(name, val):
    """Raise a clear ValueError if a calibration value isn't a positive, finite
    number.  Used by the analysis entry points so a bad pixel size / frame
    interval fails loudly instead of silently producing meaningless results."""
    if not (np.isfinite(val) and val > 0):
        raise ValueError(
            f"{name} must be a positive, finite number (got {val!r})")


def compute_msd_and_fit(tracks, pixel_size, frame_interval,
                        max_lagtime=20, n_fit=5, workers=N_CPUS,
                        alpha_thresholds=ALPHA_THRESHOLDS_DEFAULT):
    """
    Single parallel pass that computes both MSD and diffusion fits.
    Replaces tp.imsd + tp.emsd + separate fit loop — all in one go.
    """
    # Fail loudly on nonsensical calibration: a zero/negative/NaN pixel size or
    # frame interval would silently collapse every displacement (or invert the
    # time axis) and yield meaningless D/alpha rather than an obvious error.
    _require_positive_finite("pixel_size", pixel_size)
    _require_positive_finite("frame_interval", frame_interval)
    if max_lagtime < 1:
        raise ValueError(f"max_lagtime must be >= 1 (got {max_lagtime!r})")

    lag_times  = np.arange(1, max_lagtime + 1) * frame_interval
    grouped    = tracks.groupby("particle")
    pid_list   = list(grouped.groups.keys())
    n_tracks   = len(pid_list)

    print(f"  Tracks to process : {n_tracks:,}")
    print(f"  Workers           : {workers} / {N_CPUS} CPU cores")
    t0 = time.perf_counter()

    # Defensive: if linking produced zero trajectories (e.g. localiser
    # returned no spots, or every spot is an isolated singleton), return
    # empty results instead of crashing pandas with "Empty data passed
    # with indices specified".  The caller still sees the empty result
    # and can produce a sensible "no tracks found" log message.
    if n_tracks == 0:
        print("  No trajectories — skipping MSD/fit (returning empty result).")
        imsd_empty = pd.DataFrame(
            np.full((max_lagtime, 0), np.nan, dtype=float),
            index=np.arange(1, max_lagtime + 1))
        emsd_empty = pd.Series(
            np.full(max_lagtime, np.nan, dtype=float),
            index=np.arange(1, max_lagtime + 1))
        diff_empty = pd.DataFrame(columns=[
            "particle", "D", "alpha", "motion", "MSD0", "MSE", "loc_sigma_nm",
            "mean_radial_displacement_um", "radius_of_gyration_um"])
        return imsd_empty, emsd_empty, diff_empty

    # Threading vs processing trade-off:
    # The per-track work is curve_fit + numpy slicing.  curve_fit
    # releases the GIL inside its LAPACK calls but the Python wrapping
    # does not, so ThreadPool stalls on the GIL once n_tracks gets
    # large and the per-track work is dominated by Python overhead.
    # ProcessPool gives true parallelism (each worker has its own GIL)
    # but pays a one-time ~1-5 s spawn cost on Windows.
    #
    # Heuristic: use processes when there are enough tracks for the
    # parallelism win to outweigh spawn cost.  Below threshold, stick
    # with threads.
    PROCESS_POOL_THRESHOLD = 5000
    use_processes = n_tracks >= PROCESS_POOL_THRESHOLD

    # Pre-extract per-track arrays ONCE.  Iterating the groupby in a single
    # pass is markedly faster than calling get_group(pid) per particle in a
    # loop (which re-does the group lookup each time).  Results carry their
    # own particle id, so the iteration order is irrelevant downstream.
    per_track_inputs = [
        (g[["x", "y"]].values * pixel_size, g["frame"].values, pid)
        for pid, g in grouped]

    if use_processes:
        from concurrent.futures import ProcessPoolExecutor
        # ProcessPoolExecutor is capped at 61 workers on Windows (the
        # WaitForMultipleObjects 64-handle limit) — without this clamp a
        # 128-core Windows box raises "max_workers must be <= 61" before
        # processing a single track.  Threads have no such cap.
        pool_workers = safe_process_workers(workers)
        _clamp_note = (f" (clamped from {workers} — Windows 61 cap)"
                       if pool_workers != workers else "")
        print(f"  Pool              : ProcessPool × {pool_workers}{_clamp_note} "
              f"(>{PROCESS_POOL_THRESHOLD} tracks → true multi-core)")
        ExecutorCls = ProcessPoolExecutor
    else:
        pool_workers = workers
        print(f"  Pool              : ThreadPool × {pool_workers} "
              f"(<{PROCESS_POOL_THRESHOLD} tracks → low-overhead path)")
        ExecutorCls = ThreadPoolExecutor

    with ExecutorCls(max_workers=pool_workers) as _exe:
        _futs = [_exe.submit(
                    _msd_and_fit_one,
                    xy, fr, pid,
                    lag_times, max_lagtime, n_fit, alpha_thresholds)
                 for xy, fr, pid in per_track_inputs]
        results = [_f.result() for _f in
                   _tqdm(_futs, desc="  MSD + fitting", unit="track", ncols=70)]

    elapsed = time.perf_counter() - t0
    rate    = n_tracks / elapsed
    print(f"  Done in {elapsed:.1f}s  ({rate:.0f} tracks/s)")

    # Assemble imsd DataFrame  (rows = lag index, cols = particle id).
    # column_stack avoids the np.array(...).T double-allocation.
    msd_matrix = np.column_stack([r[1] for r in results])   # (max_lagtime, n_tracks)
    imsd_df    = pd.DataFrame(msd_matrix,
                              index=np.arange(1, max_lagtime + 1),
                              columns=[r[0] for r in results])

    # Ensemble MSD = nanmean across tracks at each lag
    emsd_series = pd.Series(np.nanmean(msd_matrix, axis=1),
                            index=np.arange(1, max_lagtime + 1))

    diff_df = pd.DataFrame([r[2] for r in results])

    # Merge per-track mean localisation precision (pixels → nm)
    if "ep" in tracks.columns:
        ep_nm = (tracks.groupby("particle")["ep"].mean() * pixel_size * 1000
                 ).rename("loc_precision_nm").reset_index()
        diff_df = diff_df.merge(ep_nm, on="particle", how="left")

    return imsd_df, emsd_series, diff_df


def compute_jdd(tracks, pixel_size_um, frame_interval_s, n_components=2):
    """
    Jump Distance Distribution (JDD) analysis.

    Extracts single-frame displacements from all tracks, then fits the
    empirical CDF to a mixture of 2D Brownian populations:

        CDF(r) = 1 - Σᵢ fᵢ · exp(–r² / 4Dᵢ Δt)

    Fitting the CDF (rather than histogram) avoids binning artefacts and
    gives robust estimates even with short tracks — ideal for sptPALM where
    many tracks have only 2–5 frames.

    Parameters
    ----------
    n_components : 1, 2, or 3

    Returns
    -------
    dict or None (if too few jumps to fit)
    """
    _require_positive_finite("pixel_size_um", pixel_size_um)
    _require_positive_finite("frame_interval_s", frame_interval_s)
    print(f"  JDD analysis      : {n_components} component(s)  "
          f"|  {tracks['particle'].nunique():,} tracks")
    dt = frame_interval_s

    # Vectorised across all tracks at once.  Old per-track Python loop
    # was O(n_tracks × Python-step) → seconds on 100k tracks.  We:
    #   1. Sort by (particle, frame)
    #   2. np.diff over the full arrays
    #   3. Mask out any "step" that crossed a particle boundary OR
    #      isn't between consecutive frames (frame gap > 1)
    # …then compute the displacement magnitudes in one numpy call.
    if len(tracks) < 2:
        jumps = np.array([], dtype=np.float64)
    else:
        # Drop the index level first — trackpy.link sets `frame` as
        # both an index level AND a column, which makes sort_values
        # raise "ambiguous" on those keys.  reset_index(drop=True)
        # discards the index but keeps the column intact.
        srt = (tracks
               .reset_index(drop=True)
               .sort_values(["particle", "frame"], kind="stable"))
        pid_arr   = srt["particle"].to_numpy()
        frame_arr = srt["frame"].to_numpy()
        x_arr     = srt["x"].to_numpy() * pixel_size_um
        y_arr     = srt["y"].to_numpy() * pixel_size_um
        dx = np.diff(x_arr)
        dy = np.diff(y_arr)
        same_track = pid_arr[1:] == pid_arr[:-1]
        consec     = np.diff(frame_arr) == 1
        mask = same_track & consec
        jumps = np.sqrt(dx[mask] ** 2 + dy[mask] ** 2)
    jumps = np.asarray(jumps, dtype=np.float64)
    if len(jumps) < 30:
        return None

    r_sorted = np.sort(jumps)
    cdf_emp  = np.arange(1, len(r_sorted) + 1) / len(r_sorted)

    # ── CDF model definitions ─────────────────────────────────────────────────
    def _cdf1(r, D1):
        return 1.0 - np.exp(-r ** 2 / (4 * D1 * dt))

    def _cdf2(r, D1, D2, f1):
        f2 = 1.0 - f1
        return 1.0 - f1 * np.exp(-r**2 / (4*D1*dt)) \
                   - f2 * np.exp(-r**2 / (4*D2*dt))

    def _cdf3(r, D1, D2, D3, f1, f2):
        f3 = 1.0 - f1 - f2
        return (1.0 - f1 * np.exp(-r**2 / (4*D1*dt))
                    - f2 * np.exp(-r**2 / (4*D2*dt))
                    - f3 * np.exp(-r**2 / (4*D3*dt)))

    configs = {
        1: (_cdf1, [0.05],                   ([1e-6],        [100.0])),
        2: (_cdf2, [0.005, 0.3, 0.4],        ([1e-6, 1e-5, 0.01], [10.0, 100.0, 0.99])),
        3: (_cdf3, [0.003, 0.05, 0.5, 0.3, 0.35],
                                              ([1e-6, 1e-5, 1e-4, 0.01, 0.01],
                                               [1.0, 10.0, 100.0, 0.97, 0.97])),
    }

    model, p0, (lb, ub) = configs[n_components]
    try:
        popt, _ = curve_fit(model, r_sorted, cdf_emp,
                            p0=p0, bounds=(lb, ub), maxfev=20000)
    except Exception:
        return None

    # ── Extract sorted (D, fraction) pairs ───────────────────────────────────
    if n_components == 1:
        pairs = [(popt[0], 1.0)]
    elif n_components == 2:
        pairs = sorted([(popt[0], popt[2]), (popt[1], 1.0 - popt[2])])
    else:
        f3    = 1.0 - popt[3] - popt[4]
        pairs = sorted([(popt[0], popt[3]), (popt[1], popt[4]), (popt[2], f3)])

    D_values  = [p[0] for p in pairs]
    fractions = [p[1] for p in pairs]

    # ── PDF for plotting ──────────────────────────────────────────────────────
    # Rayleigh-like: f_i(r) = r/(2DᵢΔt) · exp(–r²/4DᵢΔt)
    r_range = np.linspace(0, np.percentile(jumps, 99.5), 500)

    def _pdf_component(r, D):
        return (r / (2 * D * dt)) * np.exp(-r**2 / (4 * D * dt))

    pdfs = [frac * _pdf_component(r_range, D)
            for D, frac in zip(D_values, fractions)]
    pdf_total = np.sum(pdfs, axis=0)

    # ── Goodness of fit on the CDF (for objective model selection) ────────
    # Residuals of the fitted vs empirical CDF; R²/RMSE describe fit quality,
    # AIC/BIC penalise the extra free parameters of richer mixtures so the
    # user can justify 1- vs 2- vs 3-population models (lower AIC/BIC = better,
    # but only worth the extra component if it drops by ≳10).
    cdf_fit = model(r_sorted, *popt)
    resid   = cdf_emp - cdf_fit
    n_obs   = len(r_sorted)
    k       = len(popt)
    ss_res  = float(np.sum(resid ** 2))
    ss_tot  = float(np.sum((cdf_emp - cdf_emp.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse      = float(np.sqrt(ss_res / n_obs))
    # Gaussian-error AIC/BIC from the residual sum of squares.
    _mse = max(ss_res / n_obs, 1e-300)
    aic  = float(n_obs * np.log(_mse) + 2 * k)
    bic  = float(n_obs * np.log(_mse) + k * np.log(n_obs))

    return {
        "jumps":         jumps,
        "D_values":      D_values,
        "fractions":     fractions,
        "n_components":  n_components,
        "n_jumps":       len(jumps),
        "r_range":       r_range,
        "pdfs":          pdfs,           # per-component PDF arrays
        "pdf_total":     pdf_total,
        "cdf_r":         r_sorted,
        "cdf_empirical": cdf_emp,
        "cdf_fit":       cdf_fit,
        "r_squared":     float(r_squared) if np.isfinite(r_squared) else np.nan,
        "rmse":          rmse,
        "aic":           aic,
        "bic":           bic,
        "n_params":      k,
    }


def compute_van_hove(tracks, pixel_size_um, lag_frames=1, n_bins=80):
    """Self-part van Hove displacement distribution + non-Gaussian parameter.

    Pools the per-axis displacements (x and y) at the given frame lag across
    every track — using only consecutive, same-particle pairs whose frame gap
    is exactly `lag_frames`, so memory-bridged gaps don't leak in.

    For simple Brownian motion the van Hove distribution is Gaussian; a single
    heavy-tailed deviation is the classic signature of a *heterogeneous*
    population (e.g. mobile + trapped molecules) or anomalous transport.  The
    2D non-Gaussian parameter

        alpha2 = <r^4> / (2 <r^2>^2) - 1

    is 0 for a Gaussian/Brownian ensemble and grows positive with heterogeneity
    — a single scalar that complements the per-track D/alpha by capturing
    *population* structure the averages hide.

    Returns a dict (displacement samples in µm, a symmetric density histogram,
    the best-fit Gaussian sigma, alpha2, and counts) or None if too few pairs.
    """
    if tracks is None or len(tracks) < 2:
        return None
    srt = (tracks.reset_index(drop=True)
                 .sort_values(["particle", "frame"], kind="stable"))
    pid = srt["particle"].to_numpy()
    fr  = srt["frame"].to_numpy()
    x   = srt["x"].to_numpy() * pixel_size_um
    y   = srt["y"].to_numpy() * pixel_size_um
    L = int(max(1, lag_frames))
    if len(srt) <= L:
        return None
    same = pid[L:] == pid[:-L]
    consec = (fr[L:] - fr[:-L]) == L
    mask = same & consec
    dx = (x[L:] - x[:-L])[mask]
    dy = (y[L:] - y[:-L])[mask]
    if dx.size < 50:
        return None

    disp = np.concatenate([dx, dy])          # per-axis, symmetric about 0
    r2 = dx ** 2 + dy ** 2                    # 2D squared displacement
    m2 = float(np.mean(r2))
    m4 = float(np.mean(r2 ** 2))
    alpha2 = float(m4 / (2.0 * m2 * m2) - 1.0) if m2 > 0 else np.nan
    sigma = float(np.std(disp))

    lim = float(np.percentile(np.abs(disp), 99.5))
    if not (lim > 0):
        lim = max(sigma * 4.0, 1e-3)
    edges = np.linspace(-lim, lim, n_bins + 1)
    pdf, _ = np.histogram(disp, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Reference Gaussian with the same sigma (for overlay / comparison).
    gauss = (np.exp(-centers ** 2 / (2.0 * sigma ** 2))
             / (sigma * np.sqrt(2.0 * np.pi))) if sigma > 0 else centers * 0.0
    return {
        "displacements_um":    disp,
        "dx_um":               dx,
        "dy_um":               dy,
        "bin_centers_um":      centers,
        "pdf":                 pdf,
        "gaussian_pdf":        gauss,
        "gaussian_sigma_um":   sigma,
        "non_gaussian_alpha2": alpha2,
        "lag_frames":          L,
        "n_displacements":     int(dx.size),
    }


def compute_vacf(tracks, frame_interval_s, pixel_size_um, max_lag=10):
    """Ensemble velocity autocorrelation function (VACF) + directional persistence.

    For each track the per-frame velocity is the consecutive step vector
    v(t) = [r(t+1) - r(t)] / dt (only gapless, same-particle steps).  The VACF
    at lag tau is the ensemble average of v(t)·v(t+tau) normalised by the
    zero-lag value, so VACF(0) = 1 by construction.

        VACF(tau) = <v(t)·v(t+tau)> / <v(t)·v(t)>

    For ideal Brownian motion successive steps are uncorrelated, so VACF(tau>=1)
    ~ 0.  Persistent / directed motion gives a positive VACF that decays over a
    characteristic persistence time; anti-persistent (caged) motion gives a
    negative VACF(1).  The lag-1 value is reported as `persistence` — a compact
    directionality index that complements the turning-angle distribution.

    Returns a dict (lags in frames & seconds, normalised VACF, persistence,
    step count) or None if too few velocity pairs.
    """
    if tracks is None or len(tracks) < 3:
        return None
    dt = float(frame_interval_s) if frame_interval_s and frame_interval_s > 0 else 1.0
    srt = (tracks.reset_index(drop=True)
                 .sort_values(["particle", "frame"], kind="stable"))
    max_lag = int(max(1, max_lag))
    # numerator[tau] = sum over tracks & t of v(t)·v(t+tau); counts[tau] = #pairs
    num = np.zeros(max_lag + 1)
    cnt = np.zeros(max_lag + 1, dtype=np.int64)
    for _pid, g in srt.groupby("particle", sort=False):
        fr = g["frame"].to_numpy()
        x  = g["x"].to_numpy() * pixel_size_um
        y  = g["y"].to_numpy() * pixel_size_um
        if len(fr) < 3:
            continue
        # per-frame velocities from consecutive (gap == 1) steps only
        gap1 = (fr[1:] - fr[:-1]) == 1
        vx = (x[1:] - x[:-1]) / dt
        vy = (y[1:] - y[:-1]) / dt
        nv = len(vx)
        for tau in range(0, max_lag + 1):
            if tau >= nv:
                break
            # both the step at i and the step at i+tau must be real (gap==1)
            ok = gap1[:nv - tau] & gap1[tau:]
            if not ok.any():
                continue
            dot = vx[:nv - tau][ok] * vx[tau:][ok] + vy[:nv - tau][ok] * vy[tau:][ok]
            num[tau] += float(dot.sum())
            cnt[tau] += int(ok.sum())
    if cnt[0] < 20 or num[0] <= 0:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_dot = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)
        vacf = mean_dot / mean_dot[0]
    lags = np.arange(0, max_lag + 1)
    return {
        "lags_frames":   lags,
        "lags_s":        lags * dt,
        "vacf":          vacf,
        "persistence":   float(vacf[1]) if max_lag >= 1 and cnt[1] > 0 else np.nan,
        "n_velocities":  int(cnt[0]),
    }


def compute_turning_angles(tracks):
    """For each track with ≥3 points, compute step-to-step **signed** turning
    angles in degrees, in the range (-180°, +180°].

    Sign convention (standard 2D right-handed):
        +90°  =  90° left turn (counter-clockwise rotation from v1 to v2)
        -90°  =  90° right turn (clockwise rotation)
          0°  =  continued straight
        ±180° =  full reversal

    Computation: for consecutive step vectors v1 = r(t_{i+1}) - r(t_i)
    and v2 = r(t_{i+2}) - r(t_{i+1}),

        θ = atan2( v1.x · v2.y - v1.y · v2.x,    v1 · v2 )

    where the first argument is the z-component of the 3-D cross product
    v1 × v2 (positive for counter-clockwise rotation). Returns a flat
    array of all angles across all tracks, in degrees.
    """
    print(f"  Turning angles    : {tracks['particle'].nunique():,} tracks")
    # Vectorised across all tracks at once.  Sort by (particle, frame),
    # take np.diff over the full arrays, then mask out segments that
    # cross a track boundary (where particle id changed between
    # consecutive rows).  ~50× faster than the per-track loop on 100k
    # tracks because we never re-enter the Python interpreter.
    if len(tracks) < 3:
        result = np.array([], dtype=float)
    else:
        # Drop the index level first — trackpy.link sets `frame` as
        # both an index level AND a column, which makes sort_values
        # raise "ambiguous" on those keys.
        srt = (tracks
               .reset_index(drop=True)
               .sort_values(["particle", "frame"], kind="stable"))
        pid_arr = srt["particle"].to_numpy()
        xy_arr  = srt[["x", "y"]].to_numpy()
        # Step vectors v[i] = xy[i+1] - xy[i].  same_track_step[i] is True
        # iff rows i and i+1 belong to the same particle.
        steps = np.diff(xy_arr, axis=0)                       # (n-1, 2)
        same_step  = (pid_arr[1:] == pid_arr[:-1])            # (n-1,)
        if len(steps) < 2:
            result = np.array([], dtype=float)
        else:
            v1 = steps[:-1]
            v2 = steps[1:]
            # A turn at position i requires three consecutive same-track
            # rows: (i, i+1, i+2).  Equivalently both steps must be
            # within-track AND the middle row must be the same in both.
            both_in_track = same_step[:-1] & same_step[1:]
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            dot   = np.sum(v1 * v2, axis=1)
            norm1 = np.linalg.norm(v1, axis=1)
            norm2 = np.linalg.norm(v2, axis=1)
            valid = both_in_track & (norm1 > 0) & (norm2 > 0)
            if valid.any():
                result = np.degrees(np.arctan2(cross[valid], dot[valid]))
            else:
                result = np.array([], dtype=float)
    if result.size:
        all_angles = [result]   # keep the downstream concatenate code happy
    else:
        all_angles = []
    if all_angles:
        result = np.concatenate(all_angles)
        # Distribution sanity check — Brownian motion should produce a
        # roughly symmetric distribution around 0°.  Strong asymmetry can
        # indicate uncorrected drift, an asymmetric cellular geometry, or
        # a real biological turn bias.  Printed for diagnostic verification.
        if len(result) > 0:
            pos = int((result > 0).sum())
            neg = int((result < 0).sum())
            zer = int((result == 0).sum())
            print(f"    signed turning angles: "
                  f"{pos:,} positive  /  {neg:,} negative  /  {zer:,} zero  "
                  f"|  min={result.min():.1f}°  max={result.max():.1f}°  "
                  f"mean={result.mean():.2f}°  median={np.median(result):.2f}°")
        return result
    return np.array([])


def compute_mobile_fraction_over_time(tracks, diff_df, frame_interval,
                                       window_frames=100,
                                       d_threshold=MOBILE_D_THRESHOLD_DEFAULT):
    """Compute mobile fraction in sliding windows of `window_frames` frames.

    Mobile = tracks with D ≥ d_threshold (consistent with _mob_immob_ratio
    and the LogD-distribution panel's threshold line).  Tracks with
    non-finite D are excluded from the window denominator.

    Returns DataFrame with columns: time_s, mobile_fraction, n_tracks.
    Only windows with ≥5 valid tracks are included.
    """
    if len(tracks) == 0 or len(diff_df) == 0:
        return pd.DataFrame(columns=["time_s", "mobile_fraction", "n_tracks"])

    track_times = tracks.groupby("particle")["frame"].mean().reset_index()
    track_times.columns = ["particle", "mean_frame"]
    merged = track_times.merge(diff_df[["particle", "D"]], on="particle", how="inner")
    # Drop tracks where D could not be fit
    merged = merged[np.isfinite(merged["D"]) & (merged["D"] > 0)]

    max_frame = int(tracks["frame"].max())
    windows   = range(0, max_frame, window_frames)
    rows = []
    for w in windows:
        sel = merged[(merged["mean_frame"] >= w) &
                     (merged["mean_frame"] < w + window_frames)]
        total = len(sel)
        if total < 5:
            continue
        mobile = int((sel["D"] >= d_threshold).sum())
        rows.append({
            "time_s":          (w + window_frames / 2) * frame_interval,
            "mobile_fraction": mobile / total,
            "n_tracks":        total,
        })
    return pd.DataFrame(rows)


def compute_dwell_times(tracks, diff_df, frame_interval):
    """Per-track dwell times for confined / immobile tracks.

    Returns a DataFrame with three durations per track:

      dwell_time_total_s     (last_frame − first_frame + 1) × Δt   ← canonical
      dwell_time_observed_s  n_observations × Δt                   ← fewer if gaps
      dwell_time_s           alias for dwell_time_total_s          ← back-compat

    The exponential τ is fit to dwell_time_total_s (residence-time semantics).
    """
    confined_pids = diff_df[diff_df["motion"].isin(["Confined", "Immobile"])]["particle"]
    print(f"  Dwell times       : {len(confined_pids):,} confined/immobile tracks")
    rows = []
    # Group once by particle for speed
    grouped = tracks.groupby("particle")["frame"]
    for pid in confined_pids:
        if pid not in grouped.groups:
            continue
        frames = grouped.get_group(pid).values
        n_obs = len(frames)
        if n_obs == 0:
            continue
        f_min = int(frames.min())
        f_max = int(frames.max())
        dur_total = (f_max - f_min + 1) * frame_interval
        dur_obs   = n_obs * frame_interval
        rows.append({
            "particle":              int(pid),
            "dwell_time_s":          dur_total,   # back-compat alias
            "dwell_time_total_s":    dur_total,   # full duration including gaps
            "dwell_time_observed_s": dur_obs,     # observed frames × Δt
            "n_observations":        int(n_obs),
        })
    dwell_df = pd.DataFrame(rows)
    tau = np.nan
    if len(dwell_df) >= 10:
        try:
            dt = np.sort(dwell_df["dwell_time_total_s"].values)
            cdf = np.arange(1, len(dt) + 1) / len(dt)
            popt, _ = curve_fit(lambda t, tau: 1 - np.exp(-t / tau),
                                dt, cdf, p0=[dt.mean()], bounds=(1e-6, np.inf),
                                maxfev=2000)
            tau = float(popt[0])
        except Exception:
            pass
    return dwell_df, tau


def _ols_slope(x_centered, denom, y):
    """OLS slope of y on x, given pre-centred x and its Σ(x-x̄)² denominator.
    Identical to np.polyfit(x, y, 1)[0] but avoids the lstsq overhead when the
    same x-axis is reused many times."""
    yc = y - y.mean()
    return float(np.dot(x_centered, yc) / denom)


def compute_mss(tracks, pixel_size_um, frame_interval, max_lagtime=10):
    _require_positive_finite("pixel_size_um", pixel_size_um)
    _require_positive_finite("frame_interval", frame_interval)
    n_tracks = tracks["particle"].nunique()
    print(f"  MSS analysis      : {n_tracks:,} tracks")
    q_values = np.array([1.0, 2.0, 3.0, 4.0])
    # q-axis is constant across every track → pre-centre it once for the
    # final slope-of-gammas fit.
    q_ctr   = q_values - q_values.mean()
    q_denom = float(np.dot(q_ctr, q_ctr))
    results = []
    # Group with contiguous rows (sort by particle THEN frame) so groupby
    # doesn't gather scattered rows, and frames within a track are ordered.
    # reset_index(drop=True) FIRST: trackpy.link leaves `frame` as both an
    # index level AND a column, which makes sort_values(["particle","frame"])
    # raise "ambiguous" — the same guard compute_jdd/van_hove/vacf use.
    for pid, grp in (tracks.reset_index(drop=True)
                           .sort_values(["particle", "frame"], kind="stable")
                           .groupby("particle", sort=False)):
        xy = grp[["x", "y"]].values * pixel_size_um
        n = len(xy)
        if n < 6:
            continue
        # The per-track lag range is capped at n//2 regardless of max_lagtime,
        # so gate on the number of *usable* lags (>=4 for a stable log-log
        # moment fit), NOT on max_lagtime+2.  The old max_lagtime+2 gate was
        # over-restrictive and inconsistent: it demanded >=12 frames (for the
        # default max_lagtime=10) yet only ever used n//2 lags, so a 10-frame
        # track — which yields 4 valid lags — was needlessly rejected.  Short
        # single-molecule tracks (sptPALM) now contribute an MSS slope.
        lag_arr = list(range(1, min(max_lagtime + 1, n // 2)))
        if len(lag_arr) < 4:
            continue
        # log-time axis is the same for all four moments → centre it once.
        log_t   = np.log(np.array(lag_arr, dtype=float) * frame_interval)
        t_ctr   = log_t - log_t.mean()
        t_denom = float(np.dot(t_ctr, t_ctr))
        # r depends only on the lag, NOT on q — compute it once per lag and
        # raise to each power (was recomputed 4× per lag before).
        moments = np.empty((4, len(lag_arr)))
        for li, lag in enumerate(lag_arr):
            d = xy[lag:] - xy[:-lag]
            r = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
            for qi in range(4):
                moments[qi, li] = np.mean(r ** q_values[qi])
        gammas = np.empty(4)
        for qi in range(4):
            gammas[qi] = _ols_slope(t_ctr, t_denom,
                                    np.log(moments[qi] + 1e-15))
        mss_slope = _ols_slope(q_ctr, q_denom, gammas)
        results.append({"particle": int(pid), "mss_slope": float(mss_slope)})
    print(f"  MSS: {len(results):,}/{n_tracks:,} tracks long enough "
          f"(>= 10 frames) for a moment-scaling slope")
    return pd.DataFrame(results)


def _msd_auc(emsd_df, frame_interval):
    """Trapezoidal AUC of the MSD curve in µm²·s units."""
    if emsd_df is None or len(emsd_df) == 0:
        return np.nan
    t = emsd_df["lag_frame"].values * frame_interval
    y = emsd_df["msd_um2"].values
    order = np.argsort(t)
    # NumPy 2.x renamed trapz → trapezoid
    _trap = getattr(np, "trapezoid", None) or np.trapz
    return float(_trap(y[order], t[order]))


def _mob_immob_ratio(diff_df, d_threshold=MOBILE_D_THRESHOLD_DEFAULT):
    """Mobile / Immobile ratio defined by a diffusion-coefficient threshold.

    Tracks with D ≥ d_threshold count as Mobile; D < d_threshold count as
    Immobile.  Tracks with non-finite D (alpha fit failed) are excluded
    from BOTH numerator and denominator — they contribute neither mobility
    state, which avoids inflating either count.
    """
    if diff_df is None or "D" not in diff_df.columns:
        return np.nan
    d = diff_df["D"].values
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() == 0:
        return np.nan
    d = d[valid]
    n_mob = int((d >= d_threshold).sum())
    n_imm = int((d <  d_threshold).sum())
    return float(n_mob / n_imm) if n_imm > 0 else np.nan


def _motion_fractions(diff_df):
    """Return dict of fractions per motion class."""
    if diff_df is None or "motion" not in diff_df.columns:
        return {}
    counts = diff_df["motion"].value_counts()
    total = counts.sum()
    if total == 0:
        return {}
    return {k: float(v / total) for k, v in counts.items()}


def _track_lengths(tracks_df, frame_interval):
    """Return per-track lengths in seconds."""
    if tracks_df is None or "particle" not in tracks_df.columns:
        return np.array([])
    counts = tracks_df.groupby("particle").size().values
    return counts * frame_interval
