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
from firefly.analysis.fa_enums import GapPolicy


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


# Bump whenever an existing output column changes scientific meaning.  The
# worker persists this alongside the gap policy so Compare can distinguish a
# legacy run from one made with the timestamp-aware estimators below.
DIFFUSION_METRICS_SCHEMA_VERSION = 2


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


def _canonicalize_tracks(tracks, *, require_xy=True):
    """Return trajectories in deterministic particle/frame order.

    Time-dependent analyses have no meaningful answer when rows within a track
    are not chronological.  Upstream linkers normally guarantee this, but an
    imported pre-linked table does not have to.  We also reject duplicate
    ``(particle, frame)`` observations: there is no unambiguous trajectory or
    timestamp-lag pair definition for them, so silently picking a row would
    fabricate a result.

    ``reset_index(drop=True)`` is deliberate: trackpy can leave ``frame`` both
    as an index level and a column, making ``sort_values`` otherwise ambiguous.
    """
    if not isinstance(tracks, pd.DataFrame):
        raise ValueError("tracks must be a DataFrame with particle and frame columns")
    required = {"particle", "frame"}
    if require_xy:
        required |= {"x", "y"}
    missing = required - set(getattr(tracks, "columns", []))
    if missing:
        # Keep the public empty-result contract: a caller with no trajectories
        # may naturally hand us ``pd.DataFrame()`` rather than an empty frame
        # that already carries every trajectory column.  There is nothing to
        # validate or sort in that case, but downstream groupby still needs the
        # canonical empty schema to reach its deliberate empty-result branch.
        if len(tracks) == 0:
            ordered = tracks.copy()
            for col in missing:
                ordered[col] = pd.Series(dtype=float)
            return ordered.reset_index(drop=True)
        raise ValueError(f"tracks is missing required columns: {sorted(missing)}")
    ordered = (tracks.reset_index(drop=True)
                     .sort_values(["particle", "frame"], kind="stable")
                     .reset_index(drop=True))
    duplicate = ordered.duplicated(["particle", "frame"], keep=False)
    if duplicate.any():
        sample = ordered.loc[duplicate, ["particle", "frame"]].head(3)
        pairs = ", ".join(
            f"({row.particle!r}, {row.frame!r})" for row in sample.itertuples(index=False))
        raise ValueError(
            "trajectory has multiple localisations for the same particle/frame "
            "(often a branched TrackMate export): "
            f"{pairs}. Export linear tracks or resolve branches before analysis.")
    return ordered


def _timestamp_lag_indices(frames, lag):
    """Return index pairs separated by exactly ``lag`` frame numbers.

    ``frames`` must be sorted and unique.  The vectorised search keeps the
    all-pairs estimator O(n log n) per requested lag instead of accidentally
    treating row offsets as time offsets.
    """
    frames = np.asarray(frames)
    n = len(frames)
    if n == 0 or lag < 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    if lag == 0:
        idx = np.arange(n, dtype=int)
        return idx, idx
    targets = frames + lag
    right = np.searchsorted(frames, targets, side="left")
    possible = right < n
    if not possible.any():
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    left = np.flatnonzero(possible)
    right = right[possible]
    matched = frames[right] == targets[left]
    return left[matched], right[matched]


def _lag_displacements(xy_um, frames, lag, gap_policy=GapPolicy.ALL_PAIRS):
    """Displacements at one requested frame lag under a declared gap policy."""
    policy = GapPolicy.parse(gap_policy)
    frames = np.asarray(frames)
    xy_um = np.asarray(xy_um)
    n = len(frames)
    if lag < 1 or n < 2:
        return np.empty((0, 2), dtype=float)

    if policy is GapPolicy.CONTIGUOUS:
        # With sorted unique integer frame numbers, a row offset of ``lag``
        # whose endpoint difference is also ``lag`` implies every intervening
        # frame was observed.  This is the historical contiguous-run policy.
        if lag >= n:
            return np.empty((0, 2), dtype=float)
        left = np.arange(n - lag, dtype=int)
        right = left + lag
        valid = (frames[right] - frames[left]) == lag
        return xy_um[right[valid]] - xy_um[left[valid]]

    left, right = _timestamp_lag_indices(frames, lag)
    return xy_um[right] - xy_um[left]


def _longest_contiguous_run_span(frames):
    """Largest usable frame lag inside an uninterrupted observed run.

    ``frames`` is expected to be sorted and unique by :func:`_canonicalize_tracks`.
    A run containing N consecutive observations supports lags 1 through N - 1;
    the value returned here is therefore a frame *span*, not an observation
    count.  MSS's contiguous compatibility mode must be bounded by this actual
    temporal availability, never by the total rows scattered across gaps.
    """
    frames = np.asarray(frames)
    if len(frames) < 2:
        return 0
    run_starts = np.r_[0, np.flatnonzero(np.diff(frames) != 1) + 1]
    run_lengths = np.diff(np.r_[run_starts, len(frames)])
    return int(run_lengths.max() - 1)


def _msd_and_fit_one(xy_um, frames, pid, lag_times, max_lagtime, n_fit,
                     alpha_thresholds=ALPHA_THRESHOLDS_DEFAULT,
                     gap_policy=GapPolicy.ALL_PAIRS):
    """
    Compute per-track MSD array AND fit D + alpha in a single pass.

    ``GapPolicy.ALL_PAIRS`` (the default) uses every pair of positions whose
    *frame numbers* differ by the requested lag.  ``CONTIGUOUS`` retains the
    old policy of using only uninterrupted observed runs.
    """
    msd_vals = np.full(max_lagtime, np.nan)
    n_pts = len(xy_um)
    policy = GapPolicy.parse(gap_policy)
    frame_interval = float(lag_times[0]) if len(lag_times) else np.nan
    # Fast path: a gapless track (consecutive frame numbers — the common case,
    # and the ONLY case when memory=0) has equivalent pair sets under both
    # policies, so direct slicing avoids repeated timestamp lookup.
    gapless = n_pts >= 2 and np.all(np.diff(frames) == 1)
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
            # A two-observation track at frames 0 and 10 legitimately has a
            # lag-10 MSD.  Stop only once the requested *time* exceeds the
            # temporal span for the timestamp estimator.
            if (policy is GapPolicy.ALL_PAIRS
                    and lag > frames[-1] - frames[0]):
                break
            d = _lag_displacements(xy_um, frames, lag, policy)
            if len(d):
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
    D = alpha = np.nan
    msd0 = np.nan        # localisation-error offset (4·sigma²) == PALM-Tracer "MSD(0)"
    mse  = np.nan        # mean squared residual of the fit
    immobile = False     # set when alpha can't be measured (jitter-dominated track)
    fit_status = "insufficient_lags"
    # The zero-MSD diagnosis is a property of the whole measured curve, not
    # merely the user-selected fit window.  A track can sit still in the first
    # five lags yet have a legitimate longer-lag displacement across a blink;
    # calling that ``below_resolution`` would discard real geometry/data.
    finite_msd = msd_vals[np.isfinite(msd_vals)]

    # An exactly zero MSD is below the measurement resolution, not evidence for
    # a numerically exact physical D=0.  Preserve the finite geometric zeros,
    # but leave D/alpha unmeasured so mobility and log-D populations do not
    # accidentally turn a degenerate fit into a hard biological class.
    if finite_msd.size and np.all(finite_msd == 0.0):
        D = np.nan
        alpha = np.nan
        msd0 = np.nan
        mse = 0.0
        immobile = False
        fit_status = "below_resolution"

    ok  = np.isfinite(m) & (m > 0)
    n_ok = int(ok.sum())
    t_ok, m_ok = t[ok], m[ok]
    if not immobile and n_ok >= 4:
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
            fit_status = "fit"
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
                # A non-zero curve that is dominated by the fitted static
                # offset is scientifically different from an exactly-zero MSD:
                # D remains a finite (albeit floor-limited) fit, while alpha
                # is not identifiable.  Keep that distinction in the output
                # rather than conflating both cases as below-resolution.
                fit_status = ("offset_dominated" if dyn_frac < 0.10
                              else "alpha_unmeasurable")
        except Exception:
            fit_status = "fit_failed"
            pass
    if not np.isfinite(D) and not immobile and n_ok >= 3:
        # Fallback for very short tracks (or a non-converging joint fit): the
        # legacy two-step estimate (linear D + log-log alpha).  Less accurate
        # near the localisation floor but always returns something.
        # The bare log-log slope on as few as 3 noisy MSD points is UNBOUNDED,
        # so a 4-frame jitter track could yield alpha = 4.5 / -3 and be
        # confidently mislabelled "Directed"/"Immobile".  Keep it only when it
        # lands in the physical [0, 2] the joint fit enforces; an out-of-range
        # slope is a noise artefact, not super-/sub-diffusion → treat alpha as
        # UNMEASURABLE rather than trusting it.  (#10)  We classify it as
        # "Unknown", NOT "Immobile": a directed-but-noisy 3-point track is not
        # necessarily immobile, so claiming Immobile would just swap one
        # misclassification for another — Unknown is the honest label.  (R2-14)
        try:
            _slope = float(np.polyfit(np.log(t_ok), np.log(m_ok), 1)[0])
            if 0.0 <= _slope <= 2.0:
                alpha = _slope
            else:
                alpha = np.nan   # → "Unknown" (immobile stays False)
        except Exception:
            pass
        try:
            popt, _ = curve_fit(msd_linear, t_ok, m_ok, p0=[0.01, 0],
                                bounds=([0, -np.inf], [np.inf, np.inf]),
                                maxfev=2000)
            D = float(popt[0])
            msd0 = float(popt[1])
            _resid = m_ok - msd_linear(t_ok, *popt)
            mse = float(np.mean(_resid ** 2))
            fit_status = ("fit" if np.isfinite(alpha)
                          else "alpha_unmeasurable")
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

    # Path geometry (µm):
    #   path_length_um       = Σ straight-line step distances (the polyline length)
    #   net_displacement_um  = straight-line distance first → last position
    #   directionality_ratio = net / path ∈ [0, 1]  (1 = perfectly straight;
    #                          → 0 = returns near its start).  A zero-length
    #                          path is defined as 0: static/singleton tracks
    #                          have no directed persistence, and a finite
    #                          ratio keeps scalar summaries well-defined.
    if n_pts >= 2:
        _steps      = np.sqrt(np.sum(np.diff(xy_um, axis=0) ** 2, axis=1))
        path_length = float(_steps.sum())
        frame_gaps = np.diff(frames)
        mean_link_displacement = float(_steps.mean())  # every observed link
        mean_link_speed = float(np.mean(_steps / (frame_gaps * frame_interval)))
        unit_step = frame_gaps == 1
        n_single_frame_steps = int(unit_step.sum())
        mean_step = (float(_steps[unit_step].mean())
                     if n_single_frame_steps else np.nan)
        net_disp    = float(np.sqrt(np.sum((xy_um[-1] - xy_um[0]) ** 2)))
        directionality = float(net_disp / path_length) if path_length > 0 else 0.0
    else:
        path_length = 0.0
        mean_link_displacement = np.nan
        mean_link_speed = np.nan
        mean_step   = np.nan                          # no unit-frame step to measure
        n_single_frame_steps = 0
        net_disp    = 0.0
        directionality = 0.0

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

    track_duration_s = (float(frames[-1] - frames[0]) * frame_interval
                        if n_pts else np.nan)

    return pid, msd_vals, dict(particle=pid, D=D, alpha=alpha, motion=motion,
                               fit_status=fit_status,
                               MSD0=msd0, MSE=mse, loc_sigma_nm=loc_sigma_nm,
                               mean_radial_displacement_um=mean_radial,
                               radius_of_gyration_um=rg,
                               path_length_um=path_length,
                               mean_link_displacement_um=mean_link_displacement,
                               mean_link_speed_um_s=mean_link_speed,
                               mean_step_um=mean_step,
                               n_single_frame_steps=n_single_frame_steps,
                               track_duration_s=track_duration_s,
                               n_observations=int(n_pts),
                               net_displacement_um=net_disp,
                               directionality_ratio=directionality)


def _require_positive_finite(name, val):
    """Raise a clear ValueError if a calibration value isn't a positive, finite
    number.  Used by the analysis entry points so a bad pixel size / frame
    interval fails loudly instead of silently producing meaningless results."""
    if not (np.isfinite(val) and val > 0):
        raise ValueError(
            f"{name} must be a positive, finite number (got {val!r})")


def compute_msd_and_fit(tracks, pixel_size, frame_interval,
                        max_lagtime=20, n_fit=5, workers=N_CPUS,
                        alpha_thresholds=ALPHA_THRESHOLDS_DEFAULT,
                        gap_policy=GapPolicy.ALL_PAIRS.value):
    """
    Single parallel pass that computes both MSD and diffusion fits.
    Replaces tp.imsd + tp.emsd + separate fit loop — all in one go.

    ``gap_policy='all_pairs'`` (default) means each lag is formed from all
    timestamp-separated observation pairs.  ``'contiguous'`` preserves the
    historical uninterrupted-run estimator for reproducibility.

    Caveat: the per-track fit is an UNWEIGHTED least-squares over the first
    `n_fit` lags.  MSD points are heteroscedastic (longer lags average over
    fewer pairs → higher variance) and strongly correlated, so this gives equal
    weight to noisier long lags.  This is the standard simple estimator and is
    fine as a default, but an optimally-weighted fit (cf. Michalet & Berglund
    2012) would be more efficient; the ensemble MSD is likewise an equal-weight
    mean across tracks regardless of track length.
    """
    # Fail loudly on nonsensical calibration: a zero/negative/NaN pixel size or
    # frame interval would silently collapse every displacement (or invert the
    # time axis) and yield meaningless D/alpha rather than an obvious error.
    _require_positive_finite("pixel_size", pixel_size)
    _require_positive_finite("frame_interval", frame_interval)
    policy = GapPolicy.parse(gap_policy)
    if max_lagtime < 1:
        raise ValueError(f"max_lagtime must be >= 1 (got {max_lagtime!r})")
    # The MSD has only `max_lagtime` lags, so a fit window n_fit > max_lagtime
    # was silently truncated to max_lagtime — the fit quietly used fewer lags
    # than the user requested, changing D/alpha with no warning.  Clamp loudly
    # so the actual window is what the user sees.  (#34)
    if n_fit > max_lagtime:
        print(f"  WARN: n_fit ({n_fit}) exceeds max_lagtime ({max_lagtime}); "
              f"the MSD only has {max_lagtime} lags, so the fit uses "
              f"{max_lagtime} (not {n_fit}).  Lower n_fit or raise max_lagtime.")
        n_fit = int(max_lagtime)

    lag_times  = np.arange(1, max_lagtime + 1) * frame_interval
    ordered_tracks = _canonicalize_tracks(tracks)
    grouped    = ordered_tracks.groupby("particle", sort=False)
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
            "particle", "D", "alpha", "motion", "fit_status", "MSD0", "MSE", "loc_sigma_nm",
            "mean_radial_displacement_um", "radius_of_gyration_um",
            "path_length_um", "mean_link_displacement_um", "mean_link_speed_um_s",
            "mean_step_um", "n_single_frame_steps", "track_duration_s",
            "n_observations", "net_displacement_um",
            "directionality_ratio"])
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
                    lag_times, max_lagtime, n_fit, alpha_thresholds, policy)
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

    # Ensemble MSD = nanmean across tracks at each lag.  Avoid NumPy's
    # all-NaN RuntimeWarning: sparse timestamp lags legitimately have no pairs.
    valid_counts = np.isfinite(msd_matrix).sum(axis=1)
    emsd_vals = np.full(max_lagtime, np.nan, dtype=float)
    np.divide(np.nansum(msd_matrix, axis=1), valid_counts,
              out=emsd_vals, where=valid_counts > 0)
    emsd_series = pd.Series(emsd_vals, index=np.arange(1, max_lagtime + 1))

    diff_df = pd.DataFrame([r[2] for r in results])

    # Merge per-track mean localisation precision (pixels → nm)
    if "ep" in ordered_tracks.columns:
        ep_nm = (ordered_tracks.groupby("particle")["ep"].mean() * pixel_size * 1000
                 ).rename("loc_precision_nm").reset_index()
        diff_df = diff_df.merge(ep_nm, on="particle", how="left")

    # Merge per-track mean MEASURED per-localisation precision: the per-spot
    # loc_sigma_x_nm/_y_nm from detection (Gaussian-MLE Fisher info / trackpy ep /
    # camera-CRLB), reduced to a 1D-equivalent (hypot/√2) so it is directly
    # comparable to the MSD-offset `loc_sigma_nm`.  Three independent precision
    # estimates that should agree — a useful cross-check (kept distinct columns).
    if {"loc_sigma_x_nm", "loc_sigma_y_nm"} <= set(ordered_tracks.columns):
        _sm = (np.hypot(ordered_tracks["loc_sigma_x_nm"],
                        ordered_tracks["loc_sigma_y_nm"]) / np.sqrt(2.0))
        meas = (ordered_tracks.assign(_loc_sigma_meas=_sm)
                .groupby("particle")["_loc_sigma_meas"].mean()
                .rename("loc_sigma_meas_nm").reset_index())
        diff_df = diff_df.merge(meas, on="particle", how="left")

    return imsd_df, emsd_series, diff_df


def compute_jdd(tracks, pixel_size_um, frame_interval_s, n_components=2,
                loc_offset_um2=0.0):
    """
    Jump Distance Distribution (JDD) analysis.

    Extracts single-frame displacements from all tracks, then fits the
    empirical CDF to a mixture of 2D Brownian populations WITH a shared
    static localisation-error offset:

        CDF(r) = 1 - Σᵢ fᵢ · exp(–r² / (4·Dᵢ·Δt + loc_offset_um2))

    where ``loc_offset_um2 = 4σ²`` is the SAME static localisation-error offset
    the MSD fit subtracts jointly (``msd_anomalous``'s ``offset`` / ``MSD0``).
    It is supplied by the caller (NOT fit): from single-lag jumps alone D and σ
    are mathematically degenerate (the Rayleigh scale is only ``4DΔt + 4σ²``), so
    σ cannot be separated here — it must come from the multi-lag MSD.  Passing the
    MSD's offset removes the σ²/Δt inflation that otherwise made the JDD D's
    systematically LARGER than the offset-corrected MSD D, so the two estimators
    now agree.  ``loc_offset_um2 = 0`` (the default) reproduces the legacy
    uncorrected fit.

    Fitting the CDF (rather than histogram) avoids binning artefacts and
    gives robust estimates even with short tracks — ideal for sptPALM where
    many tracks have only 2–5 frames.

    Parameters
    ----------
    n_components   : 1, 2, or 3
    loc_offset_um2 : static localisation-error offset 4σ² (µm²) to subtract,
                     typically the MSD fit's median ``MSD0``.  0 → no correction.

    Returns
    -------
    dict or None (if too few jumps to fit)
    """
    _require_positive_finite("pixel_size_um", pixel_size_um)
    _require_positive_finite("frame_interval_s", frame_interval_s)
    srt = _canonicalize_tracks(tracks)
    print(f"  JDD analysis      : {n_components} component(s)  "
          f"|  {srt['particle'].nunique():,} tracks"
          + (f"  |  loc offset {loc_offset_um2:.2g} µm²"
             if loc_offset_um2 else ""))
    dt = frame_interval_s
    loc_offset_um2 = float(max(0.0, loc_offset_um2))

    # Vectorised across all tracks at once.  Old per-track Python loop
    # was O(n_tracks × Python-step) → seconds on 100k tracks.  We:
    #   1. Sort by (particle, frame)
    #   2. np.diff over the full arrays
    #   3. Mask out any "step" that crossed a particle boundary OR
    #      isn't between consecutive frames (frame gap > 1)
    # …then compute the displacement magnitudes in one numpy call.
    if len(srt) < 2:
        jumps = np.array([], dtype=np.float64)
    else:
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
    # `loc_offset_um2` (= 4σ², µm²) is a FIXED static localisation-error offset
    # (closure constant, NOT fit — see the docstring on why σ is unidentifiable
    # from single-lag jumps).  With offset=0 these are byte-identical to the
    # legacy models, so the parameter layout / indices are unchanged.
    _ofs = loc_offset_um2

    def _cdf1(r, D1):
        return 1.0 - np.exp(-r ** 2 / (4 * D1 * dt + _ofs))

    def _cdf2(r, D1, D2, f1):
        f2 = 1.0 - f1
        return 1.0 - f1 * np.exp(-r**2 / (4*D1*dt + _ofs)) \
                   - f2 * np.exp(-r**2 / (4*D2*dt + _ofs))

    def _cdf3(r, D1, D2, D3, f1, f2):
        f3 = 1.0 - f1 - f2
        return (1.0 - f1 * np.exp(-r**2 / (4*D1*dt + _ofs))
                    - f2 * np.exp(-r**2 / (4*D2*dt + _ofs))
                    - f3 * np.exp(-r**2 / (4*D3*dt + _ofs)))

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
        # f1 (popt[3]) and f2 (popt[4]) are each bounded to 0.97 with NO
        # constraint that f1 + f2 <= 1, so f3 = 1 - f1 - f2 can be NEGATIVE — an
        # impossible population weight that also draws a negative PDF lobe.  (#8)
        f1, f2 = float(popt[3]), float(popt[4])
        f3 = 1.0 - f1 - f2
        if f3 < -0.02:
            # The 3-component model is unjustified for this data → fall back to
            # the physically-valid 2-component fit rather than report a negative
            # fraction.
            print(f"  WARN: 3-component JDD gave a negative population fraction "
                  f"(f3 = {f3:.3f}); falling back to a 2-component fit.")
            return compute_jdd(srt, pixel_size_um, frame_interval_s,
                               n_components=2, loc_offset_um2=loc_offset_um2)
        # Tiny negative from optimiser noise → clamp + renormalise to a valid
        # simplex (fractions in [0,1] summing to 1).
        f3 = max(f3, 0.0)
        _tot = f1 + f2 + f3
        if _tot > 0:
            f1, f2, f3 = f1 / _tot, f2 / _tot, f3 / _tot
        pairs = sorted([(popt[0], f1), (popt[1], f2), (popt[2], f3)])

    D_values  = [p[0] for p in pairs]
    fractions = [p[1] for p in pairs]

    # The static localisation-error offset is the (known, fixed) `loc_offset_um2`
    # = 4σ², so σ = √(offset/4) is the 1-D precision the JDD D's are corrected
    # for (NOT a fitted quantity — see the docstring).
    s2_fit = float(loc_offset_um2)
    sigma_loc_um = float(np.sqrt(max(s2_fit, 0.0) / 4.0))

    # ── PDF for plotting ──────────────────────────────────────────────────────
    # Rayleigh-like with the static-error offset: the derivative of the component
    # CDF 1 − exp(−r²/(4DᵢΔt + offset)) is  (2r/denom)·exp(−r²/denom).
    r_range = np.linspace(0, np.percentile(jumps, 99.5), 500)

    def _pdf_component(r, D):
        denom = 4.0 * D * dt + s2_fit
        return (2.0 * r / denom) * np.exp(-r**2 / denom)

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
        "sigma_loc_um":  sigma_loc_um,    # 1-D localisation precision √(s2/4)
        "s2_um2":        s2_fit,          # fitted static offset 4σ² (µm²)
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
    if tracks is None:
        return None
    srt = _canonicalize_tracks(tracks)
    if len(srt) < 2:
        return None
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
    if tracks is None:
        return None
    dt = float(frame_interval_s) if frame_interval_s and frame_interval_s > 0 else 1.0
    srt = _canonicalize_tracks(tracks)
    if len(srt) < 2:
        return None
    max_lag = int(max(1, max_lag))
    # numerator[tau] = sum over tracks & t of v(t)·v(t+tau); counts[tau] = #pairs
    num = np.zeros(max_lag + 1)
    cnt = np.zeros(max_lag + 1, dtype=np.int64)
    for _pid, g in srt.groupby("particle", sort=False):
        fr = g["frame"].to_numpy()
        x  = g["x"].to_numpy() * pixel_size_um
        y  = g["y"].to_numpy() * pixel_size_um
        # A two-localisation trajectory contains one perfectly valid velocity.
        # It contributes to the ensemble zero-lag normalisation even though it
        # cannot by itself provide a positive-lag autocorrelation pair.
        if len(fr) < 2:
            continue
        # Keep the *start frame* of every real unit-frame velocity.  Array
        # position is not a time coordinate once a track contains a gap.
        unit = (fr[1:] - fr[:-1]) == 1
        starts = fr[:-1][unit]
        vx = ((x[1:] - x[:-1]) / dt)[unit]
        vy = ((y[1:] - y[:-1]) / dt)[unit]
        nv = len(starts)
        if not nv:
            continue
        for tau in range(0, max_lag + 1):
            if tau > starts[-1] - starts[0]:
                break
            left, right = _timestamp_lag_indices(starts, tau)
            if not len(left):
                continue
            dot = vx[left] * vx[right] + vy[left] * vy[right]
            num[tau] += float(dot.sum())
            cnt[tau] += int(len(left))
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
        "n_pairs":       cnt.copy(),
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
    srt = _canonicalize_tracks(tracks)
    print(f"  Turning angles    : {srt['particle'].nunique():,} tracks")
    # Vectorised across all tracks at once.  Sort by (particle, frame),
    # take np.diff over the full arrays, then mask out segments that
    # cross a track boundary (where particle id changed between
    # consecutive rows).  ~50× faster than the per-track loop on 100k
    # tracks because we never re-enter the Python interpreter.
    if len(srt) < 3:
        result = np.array([], dtype=float)
    else:
        pid_arr = srt["particle"].to_numpy()
        frame_arr = srt["frame"].to_numpy()
        xy_arr  = srt[["x", "y"]].to_numpy()
        # Step vectors v[i] = xy[i+1] - xy[i].  A step is a real single-frame
        # displacement only when rows i and i+1 are the SAME particle AND
        # exactly one frame apart — a memory-bridged gap (e.g. frame 5 → 8) is
        # NOT a single step and must not enter a turning angle, which compares
        # two consecutive single-frame steps over equal time intervals.  This
        # mirrors the frame-contiguity mask in compute_jdd / compute_van_hove /
        # compute_vacf; turning angles previously masked only on the particle
        # boundary, so gap-spanning steps leaked in and biased the distribution.
        steps = np.diff(xy_arr, axis=0)                       # (n-1, 2)
        unit_step = ((pid_arr[1:] == pid_arr[:-1])
                     & (np.diff(frame_arr) == 1))             # (n-1,)
        if len(steps) < 2:
            result = np.array([], dtype=float)
        else:
            v1 = steps[:-1]
            v2 = steps[1:]
            # A turn at position i needs two consecutive single-frame steps:
            # rows (i, i+1, i+2) all same-particle at frames f, f+1, f+2.
            both_in_track = unit_step[:-1] & unit_step[1:]
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

    ordered_tracks = _canonicalize_tracks(tracks, require_xy=False)
    track_times = ordered_tracks.groupby("particle")["frame"].mean().reset_index()
    track_times.columns = ["particle", "mean_frame"]
    merged = track_times.merge(diff_df[["particle", "D"]], on="particle", how="inner")
    # A non-positive D is not log-/mobility-identifiable.  In particular,
    # below-resolution zero-MSD tracks intentionally retain D=NaN upstream.
    merged = merged[np.isfinite(merged["D"]) & (merged["D"] > 0)]

    max_frame = int(ordered_tracks["frame"].max())
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


_DWELL_COLS = ["particle", "dwell_time_s", "dwell_time_total_s",
               "dwell_time_observed_s", "n_observations", "censored"]


def compute_dwell_times(tracks, diff_df, frame_interval, n_frames=None):
    """Per-track dwell times for confined / immobile tracks.

    Returns a DataFrame with three durations per track:

      dwell_time_total_s     (last_frame − first_frame + 1) × Δt   ← canonical
      dwell_time_observed_s  n_observations × Δt                   ← fewer if gaps
      dwell_time_s           alias for dwell_time_total_s          ← back-compat
      censored               True if the track is still present at the LAST movie
                             frame (right-censored — see below)

    Residence-time τ is the maximum-likelihood estimate of an exponential dwell
    distribution UNDER RIGHT-CENSORING: a track that is still present at the
    final frame (`f_max >= n_frames-1`) has not been observed to end, so its
    true dwell is only a lower bound.  Fitting a plain (uncensored) exponential
    to such truncated durations systematically UNDER-estimates τ.  The censored
    MLE is closed-form:  τ̂ = Σ(all durations) / (#uncensored events).  When
    `n_frames` is None the last observed frame across all tracks is used as the
    movie end (best effort).  (Photobleaching also truncates dwells; correcting
    that needs a bleaching model and is out of scope — only movie-end censoring
    is handled here.)
    """
    confined_pids = diff_df[diff_df["motion"].isin(
        ["Confined", "Immobile"])]["particle"].astype(int)
    print(f"  Dwell times       : {len(confined_pids):,} confined/immobile tracks")
    if len(confined_pids) == 0 or len(tracks) == 0:
        return pd.DataFrame(columns=_DWELL_COLS), np.nan

    ordered_tracks = _canonicalize_tracks(tracks, require_xy=False)
    # Vectorised: one groupby pass for first/last frame + observation count
    # (replaces the per-pid get_group loop).
    agg = ordered_tracks.groupby("particle")["frame"].agg(["min", "max", "count"])
    agg = agg[agg.index.isin(set(confined_pids))]
    if len(agg) == 0:
        return pd.DataFrame(columns=_DWELL_COLS), np.nan

    f_min = agg["min"].to_numpy(dtype=float)
    f_max = agg["max"].to_numpy(dtype=float)
    n_obs = agg["count"].to_numpy(dtype=float)
    last_frame = (int(n_frames) - 1 if n_frames
                  else int(ordered_tracks["frame"].max()))
    dur_total = (f_max - f_min + 1.0) * frame_interval
    dur_obs   = n_obs * frame_interval
    censored  = f_max >= last_frame            # still present at the movie end

    dwell_df = pd.DataFrame({
        "particle":              agg.index.to_numpy().astype(int),
        "dwell_time_s":          dur_total,    # back-compat alias
        "dwell_time_total_s":    dur_total,    # full duration including gaps
        "dwell_time_observed_s": dur_obs,      # observed frames × Δt
        "n_observations":        n_obs.astype(int),
        "censored":              censored,
    })

    # Right-censored exponential MLE: τ̂ = total observed time / #events.
    tau = np.nan
    if len(dwell_df) >= 10:
        durs = dwell_df["dwell_time_total_s"].to_numpy(dtype=float)
        n_events = int((~dwell_df["censored"].to_numpy()).sum())
        if n_events > 0:
            tau = float(durs.sum() / n_events)
        n_cens = int(dwell_df["censored"].sum())
        if n_cens:
            print(f"  Dwell τ (censored MLE): {tau:.3g}s  "
                  f"({n_cens}/{len(dwell_df)} tracks right-censored at movie end)")
    return dwell_df, tau


def _ols_slope(x_centered, denom, y):
    """OLS slope of y on x, given pre-centred x and its Σ(x-x̄)² denominator.
    Identical to np.polyfit(x, y, 1)[0] but avoids the lstsq overhead when the
    same x-axis is reused many times."""
    yc = y - y.mean()
    return float(np.dot(x_centered, yc) / denom)


def compute_mss(tracks, pixel_size_um, frame_interval, max_lagtime=10,
                gap_policy=GapPolicy.ALL_PAIRS.value):
    """Compute per-track moment-scaling-spectrum slopes.

    The default ``all_pairs`` policy shares MSD's timestamp-lag definition.
    ``contiguous`` retains the legacy observed-run estimator for reproducible
    re-analysis of older work.
    """
    _require_positive_finite("pixel_size_um", pixel_size_um)
    _require_positive_finite("frame_interval", frame_interval)
    policy = GapPolicy.parse(gap_policy)
    ordered_tracks = _canonicalize_tracks(tracks)
    n_tracks = ordered_tracks["particle"].nunique()
    print(f"  MSS analysis      : {n_tracks:,} tracks")
    q_values = np.array([1.0, 2.0, 3.0, 4.0])
    # q-axis is constant across every track → pre-centre it once for the
    # final slope-of-gammas fit.
    q_ctr   = q_values - q_values.mean()
    q_denom = float(np.dot(q_ctr, q_ctr))
    results = []
    for pid, grp in ordered_tracks.groupby("particle", sort=False):
        xy = grp[["x", "y"]].values * pixel_size_um
        fr = grp["frame"].to_numpy()
        n = len(xy)
        if n < 6:
            continue
        # A timestamp-aware lag horizon is based on elapsed acquisition time,
        # not the number of present observations.  In contiguous compatibility
        # mode the only valid lags live inside uninterrupted observed runs, so
        # cap by the longest such frame span rather than total row count.
        available_span = (
            int(fr[-1] - fr[0])
            if policy is GapPolicy.ALL_PAIRS
            else _longest_contiguous_run_span(fr)
        )
        lag_cap = min(int(max_lagtime), available_span)
        lag_arr = list(range(1, lag_cap + 1))
        if len(lag_arr) < 4:
            continue
        # Share the exact same timestamp/contiguous pair definition as MSD.
        # r depends only on lag, not q, so compute it once then raise to each
        # requested moment.  A stable moment fit still needs four usable lags.
        good_lags, moment_cols = [], []
        for lag in lag_arr:
            d = _lag_displacements(xy, fr, lag, policy)
            if not len(d):
                continue
            r = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2)
            good_lags.append(lag)
            moment_cols.append([np.mean(r ** q) for q in q_values])
        if len(good_lags) < 4:
            continue
        # log-time axis (same for all four moments) — centre once over the
        # lags that actually contributed.
        log_t   = np.log(np.array(good_lags, dtype=float) * frame_interval)
        t_ctr   = log_t - log_t.mean()
        t_denom = float(np.dot(t_ctr, t_ctr))
        moments = np.array(moment_cols, dtype=float).T        # (4, n_good_lags)
        gammas = np.empty(4)
        for qi in range(4):
            gammas[qi] = _ols_slope(t_ctr, t_denom,
                                    np.log(moments[qi] + 1e-15))
        mss_slope = _ols_slope(q_ctr, q_denom, gammas)
        results.append({"particle": int(pid), "mss_slope": float(mss_slope)})
    print(f"  MSS: {len(results):,}/{n_tracks:,} tracks with >=4 usable "
          f"timestamp lag bins for a moment-scaling slope")
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
    # Keep the established population contract: only finite positive D values
    # are mobile/immobile-classifiable; below-resolution tracks are excluded.
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() == 0:
        return np.nan
    d = d[valid]
    n_mob = int((d >= d_threshold).sum())
    n_imm = int((d <  d_threshold).sum())
    return float(n_mob / n_imm) if n_imm > 0 else np.nan


def _motion_fractions(diff_df):
    """Fractions among alpha-classified tracks only.

    Exact-zero trajectories carry ``motion='Unknown'`` and are reported through
    the separate below-resolution fields; including them in this denominator
    would make the four displayed classes sum to less than one.
    """
    if diff_df is None or "motion" not in diff_df.columns:
        return {}
    classified = ("Immobile", "Confined", "Brownian", "Directed")
    counts = diff_df.loc[diff_df["motion"].isin(classified), "motion"].value_counts()
    total = counts.sum()
    if total == 0:
        return {}
    return {k: float(v / total) for k, v in counts.items()}


def track_elapsed_durations(tracks_df, frame_interval):
    """Return elapsed per-track duration in seconds (last frame − first frame).

    Observation count and elapsed time diverge as soon as a trajectory contains
    a blink/memory-linked gap.  This helper is intentionally small and shared by
    the reporting/UI layer so all duration cards use the same definition.
    """
    if (tracks_df is None
            or not {"particle", "frame"} <= set(getattr(tracks_df, "columns", []))):
        return np.array([])
    _require_positive_finite("frame_interval", frame_interval)
    ordered_tracks = _canonicalize_tracks(tracks_df, require_xy=False)
    span = ordered_tracks.groupby("particle")["frame"].agg(["min", "max"])
    if not len(span):
        return np.array([])
    return ((span["max"].to_numpy(dtype=float) - span["min"].to_numpy(dtype=float))
            * float(frame_interval))


def _track_lengths(tracks_df, frame_interval):
    """Legacy helper name for elapsed per-track durations in seconds."""
    return track_elapsed_durations(tracks_df, frame_interval)
