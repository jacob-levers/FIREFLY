"""Offline calibration of ``AtrousWaveletBackend._ATROUS_K_SIGMA``.

Sweeps the à trous detection sensitivity against SYNTHETIC stacks with KNOWN
spot positions (Poisson shot + Gaussian read noise across several SNRs and
densities) and a trackpy reference, and reports recall / precision / F1 and
count parity per ``k_sigma`` — so a single calibrated value can be baked into
the backend with provenance.

NOT imported by the app (kept out of the hot path).  Run manually:

    QT_QPA_PLATFORM=offscreen python scripts/calibrate_atrous.py

The recommended value is the k_sigma with the best mean F1 against ground truth,
tie-broken toward count parity with trackpy.  All RNG is seeded, so the result
is reproducible.
"""
from __future__ import annotations

import numpy as np

from firefly.analysis import fa_localize as L
from firefly.analysis.fa_localize_backends import AtrousWaveletBackend

DIAMETER = 7
MINMASS = 8.0
PERCENTILE = 64
MATCH_TOL_PX = 2.0
BG = 100.0
SIGMA = 1.3


def make_stack(n_spots, snr, *, n_frames=3, H=128, W=128, seed=0):
    """A noisy stack with known sub-pixel spot positions per frame.

    ``snr`` is the spot amplitude expressed in units of the background shot-noise
    std (sqrt(BG)); snr=2 is a faint spot, snr=8 a bright one."""
    rng = np.random.default_rng(seed)
    amp = snr * np.sqrt(BG)
    yy, xx = np.mgrid[0:H, 0:W]
    frames, truths = [], []
    for _ in range(n_frames):
        xs = rng.uniform(10, W - 10, n_spots)
        ys = rng.uniform(10, H - 10, n_spots)
        img = np.full((H, W), BG, dtype=np.float64)
        for cx, cy in zip(xs, ys):
            img += amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2)
                                  / (2 * SIGMA * SIGMA)))
        noisy = (rng.poisson(img).astype(np.float32)
                 + rng.normal(0.0, 2.0, (H, W)).astype(np.float32))
        frames.append(noisy)
        truths.append(np.column_stack([xs, ys]))
    return np.stack(frames), truths


def score(df, truths, tol=MATCH_TOL_PX):
    """Greedy nearest-neighbour match of detections to ground truth, pooled over
    frames.  Returns (recall, precision, f1, n_detected, n_truth)."""
    n_match = n_truth = n_det = 0
    for f, truth in enumerate(truths):
        det = df[df["frame"] == f][["x", "y"]].to_numpy()
        n_det += len(det)
        n_truth += len(truth)
        used = np.zeros(len(det), dtype=bool)
        for cx, cy in truth:
            if not len(det):
                break
            d = np.hypot(det[:, 0] - cx, det[:, 1] - cy)
            d[used] = np.inf
            j = int(np.argmin(d))
            if d[j] <= tol:
                used[j] = True
                n_match += 1
    recall = n_match / max(1, n_truth)
    precision = n_match / max(1, n_det)
    f1 = (2 * recall * precision / max(1e-9, recall + precision))
    return recall, precision, f1, n_det, n_truth


def main():
    configs = [(n, snr, seed)
               for n in (20, 60)
               for snr in (2.0, 4.0, 8.0)
               for seed in (0, 1)]
    stacks = [(make_stack(n, snr, seed=seed), n, snr) for (n, snr, seed) in configs]

    # trackpy reference: per-stack detected count + its own F1 vs truth.
    tp_counts, tp_f1s = [], []
    for (stack, truths), n, snr in stacks:
        df = L.localise_particles(stack, diameter=DIAMETER, minmass=MINMASS,
                                  percentile=PERCENTILE, backend="trackpy")
        r, p, f1, ndet, _ = score(df, truths)
        tp_counts.append(ndet)
        tp_f1s.append(f1)
    print(f"trackpy reference: mean F1={np.mean(tp_f1s):.3f}, "
          f"mean count={np.mean(tp_counts):.0f}\n")

    print(f"{'k_sigma':>8}  {'recall':>7}  {'prec':>7}  {'F1':>7}  "
          f"{'count/tp':>9}")
    rows = []
    for k in np.round(np.linspace(1.0, 6.0, 11), 2):
        AtrousWaveletBackend._ATROUS_K_SIGMA = float(k)
        recs, precs, f1s, ratios = [], [], [], []
        for ((stack, truths), n, snr), tpc in zip(stacks, tp_counts):
            df = L.localise_particles(stack, diameter=DIAMETER, minmass=MINMASS,
                                      percentile=PERCENTILE, backend="atrous")
            r, p, f1, ndet, _ = score(df, truths)
            recs.append(r); precs.append(p); f1s.append(f1)
            ratios.append(ndet / max(1, tpc))
        mf1 = float(np.mean(f1s))
        rows.append((float(k), mf1, float(np.mean(ratios))))
        print(f"{k:8.2f}  {np.mean(recs):7.3f}  {np.mean(precs):7.3f}  "
              f"{mf1:7.3f}  {np.mean(ratios):9.2f}")

    # Recommend: best mean F1, tie-broken toward count parity with trackpy.
    best = max(rows, key=lambda r: (round(r[1], 3), -abs(r[2] - 1.0)))
    print(f"\nRECOMMENDED _ATROUS_K_SIGMA = {best[0]:.2f}  "
          f"(mean F1={best[1]:.3f}, count/trackpy={best[2]:.2f})")


if __name__ == "__main__":
    main()
