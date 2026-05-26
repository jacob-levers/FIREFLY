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
