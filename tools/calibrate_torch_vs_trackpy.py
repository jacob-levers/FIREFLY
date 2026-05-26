#!/usr/bin/env python3
"""
calibrate_torch_vs_trackpy.py
─────────────────────────────

Developer-only tool that tunes FIREFLY's TorchBackend so its localisations
match TrackpyBackend's as closely as possible on a representative sample
stack.  Treats trackpy (the peer-reviewed standard) as ground truth, runs
a small parameter sweep over the residual knobs in TorchBackend, and
prints the optimised θ + the achieved agreement metrics.

The script does NOT modify any source files.  After you find a good θ,
copy the printed constants into `TorchBackend._TP_*` in
`sptpalm_analysis.py` and commit them.

Typical usage
─────────────

    python tools/calibrate_torch_vs_trackpy.py \\
        path/to/sample.tif \\
        --diameter 7 --minmass 1.0 --percentile 64

    # Quick check, no sweep — just measure current agreement:
    python tools/calibrate_torch_vs_trackpy.py path/to/sample.tif --report-only

    # Verify the test threshold passes at a tight tolerance:
    python tools/calibrate_torch_vs_trackpy.py path/to/sample.tif --verify 0.05

Outputs
───────

    * Per-θ row showing spot counts, matched-pair count, median /
      mean / 90-percentile centroid disagreement in pixels.
    * Final summary with the best θ found and the held-out validation
      score (10 % of frames are held out from the optimiser).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Make sure the FIREFLY package is importable when running from a checkout.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Helpers ──────────────────────────────────────────────────────────────────


@dataclass
class Theta:
    """Parameter vector for TorchBackend that the sweep optimises.
    Defaults mirror the constants currently baked into TorchBackend so
    `--report-only` runs reflect the as-shipped behaviour."""
    noise_size: float = 1.0
    smoothing_offset: int = 1
    refine_max_iters: int = 10
    refine_shift_thresh: float = 0.6
    # Threshold offset is added on top of trackpy's percentile-based
    # threshold so we can absorb small bandpass-magnitude differences
    # without changing the percentile knob the user sees in the GUI.
    threshold_offset: float = 0.0

    def as_dict(self) -> dict:
        return {
            "noise_size":          self.noise_size,
            "smoothing_offset":    self.smoothing_offset,
            "refine_max_iters":    self.refine_max_iters,
            "refine_shift_thresh": self.refine_shift_thresh,
            "threshold_offset":    self.threshold_offset,
        }


@dataclass
class AgreementMetrics:
    """Comparison summary between a Torch detection set and trackpy's."""
    n_trackpy: int
    n_torch: int
    n_matched: int
    median_xy_px: float
    mean_xy_px: float
    p90_xy_px: float
    spot_count_ratio: float   # n_torch / n_trackpy

    def composite_loss(
        self, alpha: float = 1.0, beta: float = 5.0, gamma: float = 1.0
    ) -> float:
        """Objective the optimiser minimises.

        * `(1 - count_ratio)`  — penalty for missing or extra spots.
        * `median_xy_px`       — primary scientific metric (median is robust
                                 to a handful of outlier mis-matches).
        * `mean_xy_px`         — secondary metric; catches a long tail of
                                 small biases the median ignores.

        Heavy weight on `median_xy_px` (β=5) because that's what the
        agreement test asserts on and what reviewers care about.
        """
        count_mismatch = abs(1.0 - self.spot_count_ratio)
        return (alpha * count_mismatch
                + beta * self.median_xy_px
                + gamma * self.mean_xy_px)


def _match_spots(
    tp_df, torch_df, *, frame_key: str = "frame", cutoff_px: float = 2.0
) -> AgreementMetrics:
    """Frame-by-frame nearest-neighbour match between two localisation
    DataFrames.  Returns an `AgreementMetrics` aggregated over all frames.
    """
    from scipy.spatial import cKDTree
    n_tp_total    = len(tp_df)
    n_torch_total = len(torch_df)
    if n_tp_total == 0 or n_torch_total == 0:
        return AgreementMetrics(
            n_trackpy=n_tp_total,
            n_torch=n_torch_total,
            n_matched=0,
            median_xy_px=float("inf"),
            mean_xy_px=float("inf"),
            p90_xy_px=float("inf"),
            spot_count_ratio=(n_torch_total / max(1, n_tp_total))
        )
    diffs: list[float] = []
    n_matched = 0
    tp_grouped    = tp_df.groupby(frame_key)
    torch_grouped = torch_df.groupby(frame_key)
    for f in sorted(set(tp_df[frame_key]) & set(torch_df[frame_key])):
        try:
            sub_tp = tp_grouped.get_group(f)
            sub_th = torch_grouped.get_group(f)
        except KeyError:
            continue
        tp_xy = sub_tp[["x", "y"]].to_numpy()
        th_xy = sub_th[["x", "y"]].to_numpy()
        if len(tp_xy) == 0 or len(th_xy) == 0:
            continue
        tree = cKDTree(th_xy)
        dist, _idx = tree.query(tp_xy, distance_upper_bound=cutoff_px)
        valid = np.isfinite(dist) & (dist < cutoff_px)
        if not valid.any():
            continue
        diffs.extend(dist[valid].tolist())
        n_matched += int(valid.sum())
    if not diffs:
        return AgreementMetrics(
            n_trackpy=n_tp_total,
            n_torch=n_torch_total,
            n_matched=0,
            median_xy_px=float("inf"),
            mean_xy_px=float("inf"),
            p90_xy_px=float("inf"),
            spot_count_ratio=(n_torch_total / max(1, n_tp_total))
        )
    arr = np.asarray(diffs, dtype=float)
    return AgreementMetrics(
        n_trackpy=n_tp_total,
        n_torch=n_torch_total,
        n_matched=n_matched,
        median_xy_px=float(np.median(arr)),
        mean_xy_px=float(np.mean(arr)),
        p90_xy_px=float(np.percentile(arr, 90)),
        spot_count_ratio=(n_torch_total / max(1, n_tp_total))
    )


def _run_trackpy(stack, *, diameter, minmass, percentile):
    """Reference trackpy run."""
    from sptpalm_analysis import TrackpyBackend
    backend = TrackpyBackend()
    return backend.localise(
        stack,
        diameter=diameter, minmass=minmass, percentile=percentile,
        workers=1, chunk_size=len(stack),
    )


def _run_torch_with_theta(stack, theta: Theta, *, diameter, minmass,
                            percentile, device="cpu"):
    """Run TorchBackend after temporarily patching its module-level
    trackpy-compat constants to the values in `theta`.  Restores the
    originals afterwards so callers can sweep without contaminating
    each other."""
    from sptpalm_analysis import TorchBackend
    backup = {
        "_TP_NOISE_SIZE":            TorchBackend._TP_NOISE_SIZE,
        "_TP_SMOOTHING_SIZE_OFFSET": TorchBackend._TP_SMOOTHING_SIZE_OFFSET,
        "_TP_REFINE_MAX_ITERS":      TorchBackend._TP_REFINE_MAX_ITERS,
        "_TP_REFINE_SHIFT_THRESH":   TorchBackend._TP_REFINE_SHIFT_THRESH,
    }
    TorchBackend._TP_NOISE_SIZE            = float(theta.noise_size)
    TorchBackend._TP_SMOOTHING_SIZE_OFFSET = int(theta.smoothing_offset)
    TorchBackend._TP_REFINE_MAX_ITERS      = int(theta.refine_max_iters)
    TorchBackend._TP_REFINE_SHIFT_THRESH   = float(theta.refine_shift_thresh)
    try:
        backend = TorchBackend()
        backend._forced_device = device
        return backend.localise(
            stack,
            diameter=diameter,
            minmass=max(0.0, minmass + theta.threshold_offset),
            percentile=percentile,
            workers=1, chunk_size=len(stack),
        )
    finally:
        for k, v in backup.items():
            setattr(TorchBackend, k, v)


def _split_stack(stack, holdout_frac: float, seed: int = 7
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically pick a hold-out subset of frames.  Both subsets
    are contiguous-slice views — works with memmaps + plain arrays."""
    n = len(stack)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = max(1, int(round(n * holdout_frac)))
    val_idx   = np.sort(perm[:k])
    train_idx = np.sort(perm[k:])
    return np.asarray(stack)[train_idx], np.asarray(stack)[val_idx]


def _format_metrics(label: str, m: AgreementMetrics) -> str:
    return (f"  {label:<10s}  "
            f"trackpy={m.n_trackpy:>6d}  torch={m.n_torch:>6d}  "
            f"matched={m.n_matched:>6d}  "
            f"med={m.median_xy_px:.4f}  mean={m.mean_xy_px:.4f}  "
            f"p90={m.p90_xy_px:.4f}  ratio={m.spot_count_ratio:.3f}")


# ── Match-spot-count mode ────────────────────────────────────────────────────


def _match_spot_count(args, full_stack, full_pp, theta: Theta,
                        out_dir: Path):
    """Find the Torch minmass that produces ~N_trackpy localisations.

    Strategy
    --------
    1. Run Trackpy once at the user's minmass → reference count N_tp.
    2. Run Torch once at a very low minmass (effectively no mass
       filter on the candidate set) → all candidates with their
       masses.
    3. The matched minmass is simply the (N_tp)-th largest mass in
       Torch's output — by construction this filter yields exactly
       N_tp localisations.
    4. Re-run the centroid-agreement metrics at the matched threshold.
    5. Save a cumulative-count-vs-mass plot with both backends and a
       vertical line at the matched threshold.

    Single Torch run, no binary search — the mass array is all we
    need.  Whether the same N spots survive into the same tracks is
    a separate question (linking is order-dependent) but the
    localisation-count match should bring D and α into agreement
    with Trackpy, which is what the user is debating.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Trackpy reference at user's minmass ─────────────────────────
    print(f"\n── Match-spot-count ───────────────")
    print(f"  Reference: trackpy at minmass={args.minmass}")
    tp_locs = _run_trackpy(full_pp, diameter=args.diameter,
                             minmass=args.minmass,
                             percentile=args.percentile)
    n_tp = int(len(tp_locs))
    print(f"    → {n_tp:,} trackpy locs")

    # ── 2. Torch at very low minmass to capture every candidate ─────────
    # 0.01 is well below any reasonable per-frame integration so it
    # acts as "no filter at all" — anything Torch found gets kept.
    low_minmass = 0.01
    print(f"  Probing: torch at minmass={low_minmass} (capture all "
          f"candidates)")
    torch_all = _run_torch_with_theta(
        full_pp, theta, diameter=args.diameter,
        minmass=low_minmass, percentile=args.percentile,
        device=args.device)
    n_torch_all = int(len(torch_all))
    print(f"    → {n_torch_all:,} torch candidates")

    if n_torch_all < n_tp:
        print(f"\n  ⚠  Torch produced fewer candidates ({n_torch_all:,}) "
              f"than Trackpy reported ({n_tp:,}) even with no mass "
              f"filter.  This shouldn't happen — check that the "
              f"`percentile` and `diameter` arguments match.")
        return

    # ── 3. The (N_tp)-th largest mass = matched threshold ──────────────
    masses_sorted = np.sort(torch_all["mass"].values)[::-1]   # descending
    matched_minmass = float(masses_sorted[n_tp - 1])
    print(f"  Matched minmass = {matched_minmass:.4f} "
          f"(yields exactly {n_tp:,} torch locs by construction)")

    # ── 4. Filter Torch to the matched threshold + measure agreement ──
    torch_matched = torch_all[torch_all["mass"] >= matched_minmass]\
                    .reset_index(drop=True)
    print(f"    → {len(torch_matched):,} torch locs after filter")
    metrics = _match_spots(tp_locs, torch_matched)
    print(_format_metrics("matched", metrics))

    # Spot-count + centroid summary
    print(f"\n  Headline numbers at the matched minmass:")
    print(f"    Spot count:    trackpy {metrics.n_trackpy:,}  "
          f"torch {metrics.n_torch:,}  (ratio {metrics.spot_count_ratio:.3f})")
    print(f"    Matched pairs: {metrics.n_matched:,} "
          f"({100 * metrics.n_matched / max(1, metrics.n_trackpy):.1f}% "
          f"of trackpy)")
    print(f"    Centroid:      median {metrics.median_xy_px:.4f} px  "
          f"mean {metrics.mean_xy_px:.4f} px  "
          f"p90 {metrics.p90_xy_px:.4f} px")

    # ── 5. Cumulative-count plot ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Torch curve: descending-sorted masses give us "N above threshold"
    # as a function of threshold by reversing the cumulative count.
    th_masses_desc = np.sort(torch_all["mass"].values)[::-1]
    th_cum = np.arange(1, len(th_masses_desc) + 1)
    ax.semilogx(th_masses_desc, th_cum, "-", color="#d62728",
                  lw=1.5, label=f"Torch  (N_total = {n_torch_all:,})")

    # Trackpy curve: same construction but on Trackpy's own mass column.
    if len(tp_locs) > 0 and "mass" in tp_locs.columns:
        tp_masses_desc = np.sort(tp_locs["mass"].values)[::-1]
        tp_cum = np.arange(1, len(tp_masses_desc) + 1)
        ax.semilogx(tp_masses_desc, tp_cum, "-", color="#1f77b4",
                      lw=1.5, label=f"Trackpy  (N_total = {n_tp:,})")

    # Markers: user's minmass on Trackpy, matched minmass on Torch.
    ax.axvline(args.minmass, color="#1f77b4", lw=1.0, ls="--",
                 alpha=0.6, label=f"trackpy minmass = {args.minmass}")
    ax.axvline(matched_minmass, color="#d62728", lw=1.0, ls="--",
                 alpha=0.8,
                 label=f"matched torch minmass = {matched_minmass:.3f}")
    # Horizontal at the reference count so the visual intersection is
    # obvious.
    ax.axhline(n_tp, color="gray", lw=0.8, ls=":", alpha=0.6)

    ax.set_xlabel("Mass threshold")
    ax.set_ylabel("N localisations above threshold")
    ax.set_title(
        f"Cumulative localisation count vs mass threshold\n"
        f"Read across at N={n_tp:,} (the Trackpy reference) to find\n"
        f"the Torch minmass that yields equal spot counts.")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_png = out_dir / "match_spot_count.png"
    fig.savefig(out_png, dpi=140); plt.close(fig)
    print(f"\n  Saved {out_png}")

    print(f"\n  → Set the GUI's `minmass` to {matched_minmass:.3f} for "
          f"trackpy-equivalent results on this dataset.")


# ── Diagnose-extras mode ─────────────────────────────────────────────────────


def _classify_torch_tracks_against_trackpy(
    tp_locs, torch_locs, torch_tracks, cutoff_px: float = 2.0,
    match_frac: float = 0.5,
):
    """For each Torch track, decide whether it's "shared" with Trackpy
    (≥ `match_frac` of its localisations have a Trackpy neighbour
    within `cutoff_px`) or "torch-only" (the rest).

    Returns
    -------
    shared_pids   : set[int]  — Torch particle IDs classed as shared.
    torchonly_pids: set[int]  — Torch particle IDs with no Trackpy
                                 counterpart.
    """
    from scipy.spatial import cKDTree
    # Build per-frame KDTrees over Trackpy localisations.
    tp_by_frame = {}
    for f, sub in tp_locs.groupby("frame"):
        tp_by_frame[int(f)] = sub[["x", "y"]].to_numpy()
    # For each Torch row, was there a Trackpy spot within cutoff?
    torch_locs = torch_locs.copy()
    matched_flag = np.zeros(len(torch_locs), dtype=bool)
    for f, sub_idx in torch_locs.groupby("frame").indices.items():
        ref = tp_by_frame.get(int(f))
        if ref is None or len(ref) == 0:
            continue
        torch_xy = torch_locs.iloc[sub_idx][["x", "y"]].to_numpy()
        tree = cKDTree(ref)
        dist, _ = tree.query(torch_xy, distance_upper_bound=cutoff_px)
        matched_flag[sub_idx] = np.isfinite(dist) & (dist < cutoff_px)
    torch_locs["matched"] = matched_flag
    # Aggregate per Torch particle.  A track is "shared" when ≥match_frac
    # of its rows have a Trackpy neighbour.  The threshold of 0.5 is
    # forgiving — even a partly-overlapping track counts as shared,
    # which mirrors how a human reviewer would think about "did Trackpy
    # find this molecule".
    shared_pids: set[int]    = set()
    torchonly_pids: set[int] = set()
    for pid, sub in torch_tracks.groupby("particle"):
        # Track DataFrame doesn't carry our match flag — re-join by
        # (frame, x, y) tuple, which is unique within a single run.
        key_track = list(zip(sub["frame"].astype(int).tolist(),
                              sub["x"].round(6).tolist(),
                              sub["y"].round(6).tolist()))
        key_loc = list(zip(torch_locs["frame"].astype(int).tolist(),
                            torch_locs["x"].round(6).tolist(),
                            torch_locs["y"].round(6).tolist()))
        # Build a dict for O(1) lookup.
        match_lookup = dict(zip(key_loc, matched_flag.tolist()))
        match_counts = [bool(match_lookup.get(k, False)) for k in key_track]
        frac_matched = sum(match_counts) / max(1, len(match_counts))
        if frac_matched >= match_frac:
            shared_pids.add(int(pid))
        else:
            torchonly_pids.add(int(pid))
    return shared_pids, torchonly_pids


def _diagnose_extras(args, tp_locs, torch_locs, out_dir: Path,
                      diameter: int, search_range: int, memory: int,
                      min_track_len: int, pixel_size_um: float,
                      frame_interval_s: float):
    """Full link + MSD-fit pipeline on both backends, then plots showing
    whether the Torch-only tracks look statistically like the shared
    tracks (real biology) or like outliers (false positives)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sptpalm_analysis import (link_trajectories,
                                    compute_msd_and_fit,
                                    classify_motion,
                                    ALPHA_THRESHOLDS_DEFAULT)

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n── Diagnose-extras: linking + MSD-fitting both backends ──")

    print("  Linking trackpy locs ...")
    tp_tracks = link_trajectories(
        tp_locs.copy(), search_range=search_range, memory=memory,
        min_len=min_track_len, linker="trackpy")
    print(f"    → {tp_tracks['particle'].nunique():,} trackpy tracks")

    print("  Linking torch locs ...")
    th_tracks = link_trajectories(
        torch_locs.copy(), search_range=search_range, memory=memory,
        min_len=min_track_len, linker="trackpy")
    n_th_tracks = th_tracks["particle"].nunique()
    print(f"    → {n_th_tracks:,} torch tracks")

    # MSD + α fit on Torch tracks (the side that has the "extras").
    # `compute_msd_and_fit` returns (imsd_df, emsd_series, diff_df) —
    # we only care about the per-track diffusion summary.
    print("  Fitting per-track D + α on Torch tracks ...")
    _imsd, _emsd, th_diff = compute_msd_and_fit(
        th_tracks, pixel_size=pixel_size_um,
        frame_interval=frame_interval_s,
        max_lagtime=20, n_fit=5)
    print(f"    → {len(th_diff):,} per-track fits")

    # Classify Torch tracks as shared vs torch-only.
    shared_pids, torchonly_pids = _classify_torch_tracks_against_trackpy(
        tp_locs, torch_locs, th_tracks)
    print(f"  Shared with trackpy: {len(shared_pids):,}  |  "
          f"torch-only: {len(torchonly_pids):,}")

    # ── Plot 1: Spatial map ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    shared_rows = th_tracks[th_tracks["particle"].isin(shared_pids)]
    only_rows   = th_tracks[th_tracks["particle"].isin(torchonly_pids)]
    if len(shared_rows):
        for pid, sub in shared_rows.groupby("particle"):
            ax.plot(sub["x"], sub["y"], "-", color="#1f77b4",
                    alpha=0.35, lw=0.5)
    if len(only_rows):
        for pid, sub in only_rows.groupby("particle"):
            ax.plot(sub["x"], sub["y"], "-", color="#d62728",
                    alpha=0.70, lw=0.7)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title(
        f"Torch tracks — shared with trackpy (blue, n={len(shared_pids)}) "
        f"vs torch-only (red, n={len(torchonly_pids)})\n"
        f"If reds cluster with blues → real molecules below trackpy's "
        f"intrinsic threshold.\nIf reds scatter randomly → likely artefacts.")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    fig.tight_layout()
    out_spatial = out_dir / "diagnose_extras_spatial.png"
    fig.savefig(out_spatial, dpi=140); plt.close(fig)
    print(f"  Saved {out_spatial}")

    # ── Plot 2: Track-length histogram ─────────────────────────────────
    lens_shared = th_tracks[th_tracks["particle"].isin(shared_pids)]\
                    .groupby("particle").size().to_numpy()
    lens_only   = th_tracks[th_tracks["particle"].isin(torchonly_pids)]\
                    .groupby("particle").size().to_numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    max_len = int(max(lens_shared.max() if len(lens_shared) else 0,
                       lens_only.max()   if len(lens_only)   else 0,
                       min_track_len + 1))
    bins = np.arange(min_track_len, max_len + 2)
    if len(lens_shared):
        ax.hist(lens_shared, bins=bins, alpha=0.6, color="#1f77b4",
                label=f"shared (n={len(lens_shared)})", density=True)
    if len(lens_only):
        ax.hist(lens_only, bins=bins, alpha=0.6, color="#d62728",
                label=f"torch-only (n={len(lens_only)})", density=True)
    ax.set_xlabel("Track length (frames)")
    ax.set_ylabel("Density")
    ax.set_title("Track-length distribution\n"
                 "Overlapping distributions → torch-only tracks are real "
                 "molecules.  Mass at min_track_len → marginal detections.")
    ax.legend()
    fig.tight_layout()
    out_lens = out_dir / "diagnose_extras_track_lengths.png"
    fig.savefig(out_lens, dpi=140); plt.close(fig)
    print(f"  Saved {out_lens}")

    # ── Plot 3: α distribution (motion classes) ────────────────────────
    if len(th_diff) and "alpha" in th_diff.columns:
        diff_shared = th_diff[th_diff["particle"].isin(shared_pids)]["alpha"]\
                        .dropna().to_numpy()
        diff_only   = th_diff[th_diff["particle"].isin(torchonly_pids)]["alpha"]\
                        .dropna().to_numpy()
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(0, 2.0, 41)
        if len(diff_shared):
            ax.hist(diff_shared, bins=bins, alpha=0.6, color="#1f77b4",
                    label=f"shared (n={len(diff_shared)})", density=True)
        if len(diff_only):
            ax.hist(diff_only, bins=bins, alpha=0.6, color="#d62728",
                    label=f"torch-only (n={len(diff_only)})", density=True)
        # Mark the motion-class thresholds.
        t_imm, t_conf, t_dir = ALPHA_THRESHOLDS_DEFAULT
        for t in (t_imm, t_conf, t_dir):
            ax.axvline(t, color="k", lw=0.6, ls="--", alpha=0.4)
        ax.set_xlabel("Anomalous exponent α")
        ax.set_ylabel("Density")
        ax.set_title(
            "α (motion-class) distribution\n"
            "Closely-overlapping curves → torch-only tracks share the\n"
            "same motion physics as shared ones (i.e. real biology).")
        ax.legend()
        fig.tight_layout()
        out_alpha = out_dir / "diagnose_extras_alpha.png"
        fig.savefig(out_alpha, dpi=140); plt.close(fig)
        print(f"  Saved {out_alpha}")

        # Print motion-class summary.
        if len(diff_only):
            order = ["Immobile", "Confined", "Brownian", "Directed"]
            print(f"\n  Motion-class breakdown (torch-only vs shared):")
            for cls in order:
                n_shared = sum(1 for a in diff_shared
                                if classify_motion(a) == cls)
                n_only   = sum(1 for a in diff_only
                                if classify_motion(a) == cls)
                frac_shared = n_shared / max(1, len(diff_shared))
                frac_only   = n_only / max(1, len(diff_only))
                print(f"    {cls:<10s}  shared {n_shared:>4d} "
                      f"({100*frac_shared:>5.1f}%)   torch-only "
                      f"{n_only:>4d} ({100*frac_only:>5.1f}%)")

    print(f"\nDiagnostic plots written to {out_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Calibrate FIREFLY's TorchBackend to match TrackpyBackend.")
    ap.add_argument("stack", type=str,
                    help="Path to a sample TIF / CZI file to calibrate against.")
    ap.add_argument("--diameter", type=int, default=7,
                    help="Spot diameter in pixels (odd).  Default: 7.")
    ap.add_argument("--minmass", type=float, default=1.0,
                    help="Mass threshold passed to both backends.")
    ap.add_argument("--percentile", type=float, default=64,
                    help="Percentile threshold passed to both backends.")
    ap.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "mps", "cuda"],
                    help="Torch device for the calibration runs.")
    ap.add_argument("--max-frames", type=int, default=500,
                    help="Cap the number of frames evaluated to keep "
                         "iterations fast.  Default: 500.")
    ap.add_argument("--holdout", type=float, default=0.1,
                    help="Fraction of frames held out from the optimiser "
                         "and scored at the end.")
    ap.add_argument("--report-only", action="store_true",
                    help="Skip the sweep — just print agreement for the "
                         "currently-shipped TorchBackend defaults.")
    ap.add_argument("--verify", type=float, default=None,
                    help="Verify mode: run the as-shipped TorchBackend "
                         "against trackpy and exit 0 iff the median "
                         "centroid disagreement is below this many px.")
    ap.add_argument("--max-evals", type=int, default=80,
                    help="Upper bound on objective evaluations during "
                         "the Nelder-Mead sweep.  Default: 80.")
    # Override-θ CLI knobs — useful for `--report-only` runs on a
    # second stack to validate a calibration found elsewhere.  When
    # any of these are supplied, the corresponding Theta default is
    # replaced before the report runs.
    ap.add_argument("--noise-size", type=float, default=None,
                    help="Override Theta.noise_size for the report.")
    ap.add_argument("--smoothing-offset", type=int, default=None,
                    help="Override Theta.smoothing_offset for the report.")
    ap.add_argument("--refine-iters", type=int, default=None,
                    help="Override Theta.refine_max_iters for the report.")
    ap.add_argument("--shift-thresh", type=float, default=None,
                    help="Override Theta.refine_shift_thresh for the report.")
    # Diagnostics mode — runs the full localise + link + MSD-fit
    # pipeline through both backends and writes side-by-side plots
    # showing whether Torch's "extra" tracks (those with no Trackpy
    # counterpart) are statistically indistinguishable from the
    # matched set.  Three plots are produced:
    #   * spatial map of all tracks (shared vs torch-only)
    #   * histogram of track lengths (overlaid)
    #   * histogram of per-track α (overlaid)
    # Use this to defend either backend's track count in a methods
    # section.  Output PNGs land next to the input file or in --out.
    ap.add_argument("--diagnose-extras", action="store_true",
                    help="Run full localise+link pipeline on both "
                         "backends and emit diagnostic plots that "
                         "characterise Torch's extra tracks.")
    ap.add_argument("--match-spot-count", action="store_true",
                    help="Find the Torch minmass value that makes "
                         "Torch's localisation count match Trackpy's. "
                         "Prints the matched minmass, re-runs the "
                         "agreement metrics at that value, and saves "
                         "a cumulative-mass plot showing where the "
                         "threshold lands on both backends.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output folder for --diagnose-extras plots. "
                         "Defaults to the input file's folder.")
    ap.add_argument("--search-range", type=int, default=5,
                    help="Linker search range (px). Default: 5.")
    ap.add_argument("--memory", type=int, default=0,
                    help="Linker memory (frames). Default: 0.")
    ap.add_argument("--min-track-len", type=int, default=8,
                    help="Minimum track length filter. Default: 8.")
    args = ap.parse_args()

    # ── Load + slice the stack ──────────────────────────────────────────────
    from sptpalm_analysis import load_file, preprocess_stack
    t0 = time.perf_counter()
    print(f"Loading {args.stack} ...")
    stack, meta_px, meta_fi = load_file(args.stack)
    print(f"  Loaded in {time.perf_counter() - t0:.1f}s — "
          f"shape={stack.shape}, dtype={stack.dtype}")
    if len(stack) > args.max_frames:
        # Evenly-spaced subsample — keeps temporal coverage representative.
        step = max(1, len(stack) // args.max_frames)
        stack = np.asarray(stack[::step][:args.max_frames])
        print(f"  Subsampled to {len(stack)} frames "
              f"(every {step}th frame).")
    train, val = _split_stack(stack, holdout_frac=args.holdout)
    print(f"  Train: {len(train)} frames | Hold-out: {len(val)} frames")

    # ── Preprocess once with FIREFLY's standard path so the inputs to
    # both backends are identical.  Trackpy uses its own internal
    # bandpass too — but on a preprocessed image both backends see the
    # same starting point, which is what we want.
    print("Preprocessing (uniform_filter background, σ=1 smooth) ...")
    train_pp = preprocess_stack(train, bg_radius=int(args.diameter * 2 + 1))
    val_pp   = preprocess_stack(val,   bg_radius=int(args.diameter * 2 + 1))
    print(f"  Preprocessed in {time.perf_counter() - t0:.1f}s")

    # ── Reference: trackpy on the training set ──────────────────────────────
    print("Running trackpy (reference) ...")
    t1 = time.perf_counter()
    tp_train = _run_trackpy(train_pp, diameter=args.diameter,
                              minmass=args.minmass,
                              percentile=args.percentile)
    print(f"  trackpy train: {len(tp_train):,} locs in "
          f"{time.perf_counter() - t1:.1f}s")
    t1 = time.perf_counter()
    tp_val = _run_trackpy(val_pp, diameter=args.diameter,
                            minmass=args.minmass,
                            percentile=args.percentile)
    print(f"  trackpy hold-out: {len(tp_val):,} locs in "
          f"{time.perf_counter() - t1:.1f}s")

    # ── Match-spot-count mode ───────────────────────────────────────────────
    # Find the Torch minmass that produces ≈ Trackpy's count.  Saves a
    # cumulative-mass plot showing where the threshold lands on both
    # backends.  Faster than --diagnose-extras (no linking + MSD fit),
    # so reach for this first when you want a single number to plug
    # into the GUI's minmass field.
    if args.match_spot_count:
        theta = Theta()
        if args.noise_size       is not None: theta.noise_size          = float(args.noise_size)
        if args.smoothing_offset is not None: theta.smoothing_offset    = int(args.smoothing_offset)
        if args.refine_iters     is not None: theta.refine_max_iters    = int(args.refine_iters)
        if args.shift_thresh     is not None: theta.refine_shift_thresh = float(args.shift_thresh)
        print(f"  θ = {theta.as_dict()}")
        from sptpalm_analysis import load_file as _load_file
        _full, _meta_px, _meta_fi = _load_file(args.stack)
        _full_pp = preprocess_stack(
            _full, bg_radius=int(args.diameter * 2 + 1))
        out_root = Path(args.out) if args.out else Path(args.stack).parent
        _match_spot_count(args, _full, _full_pp, theta, out_root)
        return

    # ── Diagnose-extras mode ────────────────────────────────────────────────
    # Runs the full pipeline (detect → link → MSD-fit) through both
    # backends and writes side-by-side plots that characterise the
    # extra tracks Torch finds.  This is the mode you reach for when
    # you see "Trackpy found 73, Torch found 103" and want to defend
    # either number in a methods section.
    if args.diagnose_extras:
        theta = Theta()
        if args.noise_size       is not None: theta.noise_size          = float(args.noise_size)
        if args.smoothing_offset is not None: theta.smoothing_offset    = int(args.smoothing_offset)
        if args.refine_iters     is not None: theta.refine_max_iters    = int(args.refine_iters)
        if args.shift_thresh     is not None: theta.refine_shift_thresh = float(args.shift_thresh)
        print(f"  θ = {theta.as_dict()}")
        # Use the FULL (non-subsampled, non-held-out) stack for the
        # diagnostic — track counts are the headline number we're
        # debating, and they need every frame to be representative.
        from sptpalm_analysis import load_file as _load_file
        _full, _meta_px, _meta_fi = _load_file(args.stack)
        _full_pp = preprocess_stack(
            _full, bg_radius=int(args.diameter * 2 + 1))
        print(f"\nRunning trackpy on full stack ({len(_full_pp)} frames) ...")
        tp_full = _run_trackpy(_full_pp, diameter=args.diameter,
                                minmass=args.minmass,
                                percentile=args.percentile)
        print(f"  → {len(tp_full):,} trackpy locs")
        print(f"Running torch on full stack ...")
        th_full = _run_torch_with_theta(
            _full_pp, theta,
            diameter=args.diameter, minmass=args.minmass,
            percentile=args.percentile, device=args.device)
        print(f"  → {len(th_full):,} torch locs")

        out_root = Path(args.out) if args.out else Path(args.stack).parent
        pixel_size_um    = float(_meta_px) if _meta_px else 0.106
        frame_interval_s = float(_meta_fi) if _meta_fi else 0.020
        _diagnose_extras(
            args, tp_full, th_full, out_root,
            diameter=args.diameter,
            search_range=args.search_range,
            memory=args.memory,
            min_track_len=args.min_track_len,
            pixel_size_um=pixel_size_um,
            frame_interval_s=frame_interval_s)
        return

    # ── Report-only / verify modes ──────────────────────────────────────────
    if args.report_only or args.verify is not None:
        theta = Theta()
        if args.noise_size       is not None: theta.noise_size          = float(args.noise_size)
        if args.smoothing_offset is not None: theta.smoothing_offset    = int(args.smoothing_offset)
        if args.refine_iters     is not None: theta.refine_max_iters    = int(args.refine_iters)
        if args.shift_thresh     is not None: theta.refine_shift_thresh = float(args.shift_thresh)
        print(f"  θ = {theta.as_dict()}")
        torch_train = _run_torch_with_theta(
            train_pp, theta,
            diameter=args.diameter, minmass=args.minmass,
            percentile=args.percentile, device=args.device)
        m_train = _match_spots(tp_train, torch_train)
        print(_format_metrics("train", m_train))
        torch_val = _run_torch_with_theta(
            val_pp, theta,
            diameter=args.diameter, minmass=args.minmass,
            percentile=args.percentile, device=args.device)
        m_val = _match_spots(tp_val, torch_val)
        print(_format_metrics("hold-out", m_val))
        if args.verify is not None:
            tol = float(args.verify)
            if m_val.median_xy_px <= tol:
                print(f"\nPASS: hold-out median {m_val.median_xy_px:.4f} "
                      f"<= tolerance {tol:.4f}")
                sys.exit(0)
            else:
                print(f"\nFAIL: hold-out median {m_val.median_xy_px:.4f} "
                      f"> tolerance {tol:.4f}")
                sys.exit(1)
        return

    # ── Sweep ───────────────────────────────────────────────────────────────
    from scipy.optimize import minimize

    eval_count = {"n": 0}
    history: list[tuple[Theta, AgreementMetrics, float]] = []

    def objective(vec: np.ndarray) -> float:
        eval_count["n"] += 1
        theta = Theta(
            noise_size          = float(np.clip(vec[0], 0.3, 3.0)),
            smoothing_offset    = int(round(np.clip(vec[1], -2, 4))),
            refine_max_iters    = int(round(np.clip(vec[2], 1, 20))),
            refine_shift_thresh = float(np.clip(vec[3], 0.2, 1.0)),
            threshold_offset    = float(np.clip(vec[4], -0.5, 0.5)),
        )
        torch_locs = _run_torch_with_theta(
            train_pp, theta,
            diameter=args.diameter, minmass=args.minmass,
            percentile=args.percentile, device=args.device)
        m = _match_spots(tp_train, torch_locs)
        loss = m.composite_loss()
        history.append((theta, m, loss))
        print(f"  [{eval_count['n']:3d}/{args.max_evals:3d}] "
              f"θ={theta.as_dict()}  loss={loss:.4f}  "
              f"med={m.median_xy_px:.4f}  count_ratio={m.spot_count_ratio:.3f}")
        return loss

    print(f"\nSweeping {args.max_evals} evaluations of θ via "
          f"scipy.optimize.minimize (Nelder-Mead) ...")
    x0 = np.array([1.0, 1.0, 10.0, 0.6, 0.0], dtype=float)
    res = minimize(
        objective, x0,
        method="Nelder-Mead",
        options={"maxiter": args.max_evals,
                  "maxfev": args.max_evals,
                  "xatol": 1e-3, "fatol": 1e-3,
                  "adaptive": True},
    )

    # Pick the best θ seen (Nelder-Mead's final point isn't always the
    # min; tracking history-min is safer).
    best_idx = int(np.argmin([h[2] for h in history]))
    best_theta, best_metrics, best_loss = history[best_idx]

    # ── Hold-out validation ─────────────────────────────────────────────────
    print(f"\nBest θ found:  loss={best_loss:.4f}")
    print(f"  {best_theta.as_dict()}")
    print(f"  train metrics: med={best_metrics.median_xy_px:.4f}  "
          f"mean={best_metrics.mean_xy_px:.4f}  "
          f"count_ratio={best_metrics.spot_count_ratio:.3f}")

    torch_val = _run_torch_with_theta(
        val_pp, best_theta,
        diameter=args.diameter, minmass=args.minmass,
        percentile=args.percentile, device=args.device)
    m_val = _match_spots(tp_val, torch_val)
    print(_format_metrics("hold-out", m_val))

    # ── Print the constants the user should paste into TorchBackend ────────
    print("\n" + "─" * 60)
    print("Calibrated constants — paste into TorchBackend (sptpalm_analysis.py):")
    print("─" * 60)
    print(f"    _TP_NOISE_SIZE             = {best_theta.noise_size!r}")
    print(f"    _TP_SMOOTHING_SIZE_OFFSET  = {best_theta.smoothing_offset!r}")
    print(f"    _TP_REFINE_MAX_ITERS       = {best_theta.refine_max_iters!r}")
    print(f"    _TP_REFINE_SHIFT_THRESH    = {best_theta.refine_shift_thresh!r}")
    if abs(best_theta.threshold_offset) > 0.01:
        print(f"    # NOTE: best threshold offset was "
              f"{best_theta.threshold_offset:+.3f}; the default minmass")
        print(f"    # passed to the analysis should account for this.")
    print(f"\nHold-out median centroid error: "
          f"{m_val.median_xy_px:.4f} px  "
          f"({m_val.n_matched:,}/{m_val.n_trackpy:,} matched)")


if __name__ == "__main__":
    main()
