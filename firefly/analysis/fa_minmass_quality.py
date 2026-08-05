"""Quality-first minmass selection and detection-QC helpers.

This module deliberately separates three ideas that were previously conflated:

* ``minmass`` is a lower, assay-specific detection floor;
* a spatial null estimates *random long-track participation* at the configured
  linker settings (it is not a candidate false-discovery rate);
* density, temporal stability, and local assignment ambiguity are reported as
  QC outcomes and never used to fill a per-file detection quota.

The helpers are Qt-free and deterministic so the analysis worker, tests, and
standalone validation tooling can all use the same contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


QUALITY_POLICY_VERSION = "quality_first_track_ambiguity_v1"
QUALITY_NULL_DEFINITION = "roi_smoothed_spatial_redraw_v1"
QUALITY_NULL_SEED = 20260805
QUALITY_NULL_REPLICATES = 3
QUALITY_NULL_UNIFORM_MIX = 0.02
QUALITY_NULL_UPPER_QUANTILE = 0.90
QUALITY_STABLE_GRID_POINTS = 2
QUALITY_MIN_GOOD_TRACKS = 5
QUALITY_MIN_YIELD_FRACTION = 0.25
QUALITY_TEMPORAL_RATIO_LIMIT = 2.0
QUALITY_LOCAL_AMBIGUITY_WARN_FRACTION = 0.10


@dataclass(frozen=True)
class QualitySelection:
    """Result of selecting a threshold from observed/null sweep rows."""

    threshold: float
    status: str
    reason: str | None
    info: dict


def filter_candidates_to_roi(candidates: pd.DataFrame,
                             roi_mask: np.ndarray | None) -> pd.DataFrame:
    """Return candidates inside a static ROI using FIREFLY's pixel convention.

    A missing mask means the whole frame.  An empty or non-2-D mask is an error:
    silently falling back to the full frame would change the scientific
    population used to choose the threshold.
    """
    if roi_mask is None:
        return candidates.copy().reset_index(drop=True)
    mask = np.asarray(roi_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("quality-first thresholding requires one static 2-D ROI")
    if not mask.any():
        raise ValueError("quality-first thresholding received an empty ROI")
    if candidates is None or len(candidates) == 0:
        return candidates.copy().reset_index(drop=True)
    if not {"x", "y"}.issubset(candidates.columns):
        raise ValueError("candidate table must contain x and y coordinates")
    x = candidates["x"].to_numpy(dtype=float)
    y = candidates["y"].to_numpy(dtype=float)
    xi = np.clip(np.floor(x + 0.5).astype(int), 0, mask.shape[1] - 1)
    yi = np.clip(np.floor(y + 0.5).astype(int), 0, mask.shape[0] - 1)
    return candidates[mask[yi, xi]].copy().reset_index(drop=True)


def _analysis_mask(frame_shape: tuple[int, int],
                   roi_mask: np.ndarray | None) -> np.ndarray:
    h, w = int(frame_shape[0]), int(frame_shape[1])
    if h <= 0 or w <= 0:
        raise ValueError("frame shape must be positive")
    if roi_mask is None:
        return np.ones((h, w), dtype=bool)
    mask = np.asarray(roi_mask, dtype=bool)
    if mask.shape != (h, w):
        raise ValueError(
            f"ROI shape {mask.shape} does not match movie frame {(h, w)}")
    if not mask.any():
        raise ValueError("analysis ROI contains zero pixels")
    return mask


def make_spatial_null(candidates: pd.DataFrame, *,
                      frame_shape: tuple[int, int],
                      roi_mask: np.ndarray | None,
                      search_range: float,
                      seed: int,
                      uniform_mix: float = QUALITY_NULL_UNIFORM_MIX,
                      ) -> pd.DataFrame:
    """Redraw positions while preserving every row's frame and mass.

    The redraw distribution is a pooled candidate-density image smoothed by at
    least the link search radius and restricted to the exact analysis ROI.  A
    small uniform component prevents zero-probability islands.  Thus the null
    preserves ROI geometry, coarse spatial heterogeneity, per-frame counts,
    mass values, and threshold nesting, while destroying actual trajectories
    and exact fixed-pixel identity.
    """
    out = candidates.copy().reset_index(drop=True)
    if len(out) == 0:
        return out
    mask = _analysis_mask(frame_shape, roi_mask)
    h, w = mask.shape
    x = out["x"].to_numpy(dtype=float)
    y = out["y"].to_numpy(dtype=float)
    xi = np.clip(np.floor(x + 0.5).astype(int), 0, w - 1)
    yi = np.clip(np.floor(y + 0.5).astype(int), 0, h - 1)

    density = np.zeros((h, w), dtype=float)
    np.add.at(density, (yi, xi), 1.0)
    sigma = max(float(search_range), 1.0)
    density = gaussian_filter(density, sigma=sigma, mode="constant")
    density *= mask

    mix = float(np.clip(uniform_mix, 0.0, 1.0))
    support = mask.astype(float)
    if density.sum() <= 0:
        density = support
        mix = 0.0
    p_spatial = density / density.sum()
    p_uniform = support / support.sum()
    prob = (1.0 - mix) * p_spatial + mix * p_uniform
    prob = prob.ravel()
    prob /= prob.sum()

    rng = np.random.default_rng(int(seed))
    flat = rng.choice(prob.size, size=len(out), replace=True, p=prob)
    yy, xx = np.divmod(flat, w)
    # Continuous jitter avoids exact-coordinate duplicates while remaining in
    # the sampled pixel.  Clip at the image edge; the pixel itself is in-mask.
    out["x"] = np.clip(xx + rng.uniform(-0.49, 0.49, len(out)), 0, w - 1)
    out["y"] = np.clip(yy + rng.uniform(-0.49, 0.49, len(out)), 0, h - 1)
    return out


def merge_observed_and_null_sweeps(observed: list[dict],
                                   null_sweeps: list[list[dict]],
                                   *, upper_quantile: float =
                                   QUALITY_NULL_UPPER_QUANTILE) -> list[dict]:
    """Attach deterministic null statistics to the observed threshold sweep."""
    merged: list[dict] = []
    for i, row in enumerate(observed):
        r = dict(row)
        gf = np.asarray(
            [s[i].get("good_fraction", np.nan) for s in null_sweeps
             if i < len(s)], dtype=float)
        ng = np.asarray(
            [s[i].get("N_good", np.nan) for s in null_sweeps
             if i < len(s)], dtype=float)
        gf = gf[np.isfinite(gf)]
        ng = ng[np.isfinite(ng)]
        if gf.size:
            r["null_good_fraction_median"] = float(np.median(gf))
            r["null_good_fraction_upper"] = float(
                np.quantile(gf, float(upper_quantile), method="higher"))
        else:
            r["null_good_fraction_median"] = float("nan")
            r["null_good_fraction_upper"] = float("nan")
        r["null_N_good_median"] = (float(np.median(ng)) if ng.size
                                    else float("nan"))
        r["good_detection_yield"] = float(
            r.get("good_fraction", 0.0) * r.get("n_surv", 0))
        merged.append(r)
    return merged


def select_quality_threshold(sweep: list[dict], *, quality_floor: float,
                             max_null_fraction: float,
                             stable_points: int = QUALITY_STABLE_GRID_POINTS,
                             min_good_tracks: int = QUALITY_MIN_GOOD_TRACKS,
                             min_yield_fraction: float =
                             QUALITY_MIN_YIELD_FRACTION) -> QualitySelection:
    """Pick the lowest stable threshold satisfying the null-track ceiling.

    The returned threshold is *always* at least ``quality_floor``.  A valid
    selection requires the criterion to hold for adjacent grid points and to
    retain a minimum fraction of the best observed linked-detection yield.  If
    no stable point exists, a conservative diagnostic threshold is returned but
    the status is ``unresolved`` so downstream comparison can exclude the run.
    """
    floor = float(quality_floor)
    ceiling = float(max_null_fraction)
    if not sweep:
        return QualitySelection(
            floor, "unresolved", "empty_sweep",
            {"quality_floor": floor, "max_null_fraction": ceiling})

    rows = sorted((dict(r) for r in sweep), key=lambda r: float(r["t"]))
    good_yield = np.asarray(
        [float(r.get("good_detection_yield",
                     r.get("good_fraction", 0.0) * r.get("n_surv", 0)))
         for r in rows], dtype=float)
    max_yield = float(np.nanmax(good_yield)) if good_yield.size else 0.0
    yield_floor = float(min_yield_fraction) * max_yield

    eligible: list[bool] = []
    for r, gy in zip(rows, good_yield):
        null_upper = float(r.get("null_good_fraction_upper", np.nan))
        eligible.append(bool(
            float(r["t"]) >= floor
            and np.isfinite(null_upper)
            and null_upper <= ceiling
            and int(r.get("N_good", 0)) >= int(min_good_tracks)
            and gy >= yield_floor))

    need = max(1, int(stable_points))
    chosen_i = None
    for i in range(0, len(rows) - need + 1):
        if all(eligible[i:i + need]):
            chosen_i = i
            break
    if chosen_i is not None:
        row = rows[chosen_i]
        threshold = max(floor, float(row["t"]))
        return QualitySelection(
            threshold, "valid", None,
            {"quality_floor": floor,
             "max_null_fraction": ceiling,
             "stable_grid_points": need,
             "selected_index": int(chosen_i),
             "null_good_fraction_upper": float(
                 row["null_good_fraction_upper"]),
             "observed_good_tracks": int(row.get("N_good", 0)),
             "observed_good_detection_yield": float(good_yield[chosen_i]),
             "max_observed_good_detection_yield": max_yield})

    # Diagnostic fallback: among thresholds that produced some real linkage,
    # minimize the null-linked fraction, then prefer the lower threshold.  It is
    # deliberately marked unresolved and must not be pooled as a valid run.
    candidates = [
        (i, r) for i, r in enumerate(rows)
        if float(r["t"]) >= floor
        and np.isfinite(float(r.get("null_good_fraction_upper", np.nan)))
        and int(r.get("N_good", 0)) >= int(min_good_tracks)
    ]
    if candidates:
        best_i, best = min(
            candidates,
            key=lambda item: (float(item[1]["null_good_fraction_upper"]),
                              float(item[1]["t"])))
        threshold = max(floor, float(best["t"]))
        best_null = float(best["null_good_fraction_upper"])
        reason = ("null_ceiling_not_met" if best_null > ceiling
                  else "no_stable_quality_plateau")
        info = {"quality_floor": floor,
                "max_null_fraction": ceiling,
                "stable_grid_points": need,
                "selected_index": int(best_i),
                "best_null_good_fraction_upper": best_null,
                "max_observed_good_detection_yield": max_yield}
    else:
        threshold = floor
        reason = "no_linkage_above_floor"
        info = {"quality_floor": floor,
                "max_null_fraction": ceiling,
                "stable_grid_points": need,
                "max_observed_good_detection_yield": max_yield}
    return QualitySelection(threshold, "unresolved", reason, info)


def sampled_detection_diagnostics(candidates: pd.DataFrame, *, threshold: float,
                                  windows: list[tuple[int, int]],
                                  roi_area_pixels: int,
                                  pixel_size_um: float) -> dict:
    """Summarize selected candidates over the exact calibration windows."""
    area_px = int(roi_area_pixels)
    px = float(pixel_size_um)
    if area_px <= 0 or px <= 0:
        raise ValueError("analysis area and pixel size must be positive")
    frame_sample = int(sum(int(e) - int(s) for s, e in windows))
    if frame_sample <= 0:
        raise ValueError("calibration windows contain no frames")
    keep = candidates[np.asarray(candidates["mass"], dtype=float)
                      >= float(threshold)]
    per_window: list[float] = []
    for wid, (start, stop) in enumerate(windows):
        n_frames = int(stop) - int(start)
        if n_frames <= 0:
            continue
        if "window_id" in keep.columns:
            n = int((keep["window_id"] == wid).sum())
        else:
            n = len(keep) if wid == 0 else 0
        per_window.append(float(n / n_frames))
    area_um2 = float(area_px * px * px)
    mean_pf = float(len(keep) / frame_sample)
    arr = np.asarray(per_window, dtype=float)
    return {
        "sample_n_selected": int(len(keep)),
        "sample_frames": frame_sample,
        "sample_mean_per_frame": mean_pf,
        "sample_density_per_um2_frame": float(mean_pf / area_um2),
        "sample_window_mean_per_frame": [float(v) for v in arr],
        "sample_window_cv": (float(np.std(arr) / np.mean(arr))
                             if arr.size and np.mean(arr) > 0 else None),
        "roi_area_pixels": area_px,
        "roi_area_um2": area_um2,
    }


def full_detection_diagnostics(locs: pd.DataFrame, *, n_frames: int,
                               frame_shape: tuple[int, int],
                               pixel_size_um: float,
                               roi_mask: np.ndarray | None,
                               search_range_px: float,
                               temporal_bins: int = 4,
                               temporal_ratio_limit: float =
                               QUALITY_TEMPORAL_RATIO_LIMIT,
                               ambiguity_warn_fraction: float =
                               QUALITY_LOCAL_AMBIGUITY_WARN_FRACTION) -> dict:
    """Compute post-ROI/full-run density, temporal, and ambiguity diagnostics."""
    nf = int(n_frames)
    px = float(pixel_size_um)
    if nf <= 0 or px <= 0:
        raise ValueError("n_frames and pixel size must be positive")
    mask = _analysis_mask(frame_shape, roi_mask)
    area_px = int(mask.sum())
    area_um2 = float(area_px * px * px)

    frames = (pd.to_numeric(locs["frame"], errors="coerce").to_numpy(dtype=float)
              if locs is not None and len(locs) else np.array([], dtype=float))
    valid = np.isfinite(frames) & (frames >= 0) & (frames < nf)
    fi = frames[valid].astype(int, copy=False)
    counts = np.bincount(fi, minlength=nf)[:nf]
    mean_pf = float(counts.mean())
    bins = [b for b in np.array_split(counts, max(1, int(temporal_bins)))
            if len(b)]
    bin_means = np.asarray([float(b.mean()) for b in bins], dtype=float)
    first_last = None
    if bin_means.size >= 2 and bin_means[0] > 0:
        first_last = float(bin_means[-1] / bin_means[0])

    # Fraction of each frame's localisations with zero, one, or multiple
    # feasible next-frame successors inside the configured search radius.
    any_successor = ambiguous = denominator = 0
    if locs is not None and len(locs) and {"x", "y", "frame"}.issubset(locs.columns):
        work = locs.loc[valid, ["x", "y", "frame"]].copy()
        work["frame"] = work["frame"].astype(int)
        grouped = {
            int(f): g[["x", "y"]].to_numpy(dtype=float)
            for f, g in work.groupby("frame", sort=False)
        }
        radius = float(search_range_px)
        for f, current in grouped.items():
            if f >= nf - 1 or len(current) == 0:
                continue
            denominator += int(len(current))
            nxt = grouped.get(f + 1)
            if nxt is None or len(nxt) == 0:
                continue
            tree = cKDTree(nxt)
            try:
                n_near = np.asarray(
                    tree.query_ball_point(current, r=radius, return_length=True))
            except TypeError:  # older SciPy compatibility
                n_near = np.asarray(
                    [len(v) for v in tree.query_ball_point(current, r=radius)])
            any_successor += int((n_near >= 1).sum())
            ambiguous += int((n_near > 1).sum())
    any_fraction = (float(any_successor / denominator) if denominator else None)
    ambiguity_fraction = (float(ambiguous / denominator) if denominator else None)

    qc_codes: list[str] = []
    limit = float(temporal_ratio_limit)
    if first_last is not None and (first_last > limit or first_last < 1.0 / limit):
        qc_codes.append("temporal_density_shift")
    if (ambiguity_fraction is not None
            and ambiguity_fraction > float(ambiguity_warn_fraction)):
        qc_codes.append("high_local_assignment_ambiguity")

    return {
        "detection_contract": QUALITY_POLICY_VERSION,
        "full_run_n_localisations": int(valid.sum()),
        "full_run_n_frames": nf,
        "full_run_mean_per_frame": mean_pf,
        "full_run_median_per_frame": float(np.median(counts)),
        "full_run_p10_per_frame": float(np.percentile(counts, 10)),
        "full_run_p90_per_frame": float(np.percentile(counts, 90)),
        "full_run_zero_frame_fraction": float(np.mean(counts == 0)),
        "full_run_frame_count_cv": (float(np.std(counts) / mean_pf)
                                    if mean_pf > 0 else None),
        "full_run_density_per_um2_frame": float(mean_pf / area_um2),
        "temporal_bin_mean_per_frame": [float(v) for v in bin_means],
        "temporal_bin_density_per_um2_frame": [float(v / area_um2)
                                               for v in bin_means],
        "temporal_last_first_ratio": first_last,
        "roi_area_pixels": area_px,
        "roi_area_um2": area_um2,
        "roi_fraction": float(area_px / mask.size),
        "next_frame_any_successor_fraction": any_fraction,
        "next_frame_ambiguous_successor_fraction": ambiguity_fraction,
        "next_frame_ambiguity_denominator": int(denominator),
        "temporal_ratio_limit": limit,
        "next_frame_ambiguity_warn_fraction": float(
            ambiguity_warn_fraction),
        "qc_codes": qc_codes,
    }
