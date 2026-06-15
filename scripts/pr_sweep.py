"""Precision–recall sweep: Crocker–Grier (PyTorch) vs à trous wavelet on an
external ground-truth dataset, to see whether the detectors separate across
detection thresholds (one operating point can hide a difference the full PR
curve reveals).

  python scripts/pr_sweep.py --tif S.tif --gt GT.csv --pixel-size 0.1 [--frames N]

Sweeps Crocker–Grier's `minmass` and à trous's `_ATROUS_K_SIGMA`, scoring each
point's DETECTION (precision / recall / F1) against the GT; prints a table and
writes a PR-curve PNG (--out).
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pixel-size", type=float, default=None, dest="px")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--tol-nm", type=float, default=250.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="pr_sweep.png")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from firefly.bench.public import load_gt_dataset
    from firefly.bench.metrics import detection_metrics
    from firefly.analysis.fa_localize import preprocess_and_localise_adaptive
    from firefly.analysis.fa_localize_backends import AtrousWaveletBackend

    full = load_gt_dataset(a.tif, a.gt, pixel_size_um=a.px)
    n = min(a.frames, full.stack.shape[0])
    stack = full.stack[:n]
    gt = full.gt_locs[full.gt_locs["frame"] < n].reset_index(drop=True)
    px_um = full.meta["pixel_size_um"]
    tol_px = a.tol_nm / (px_um * 1000.0)
    print(f"sweep on {n} frames, {len(gt):,} GT locs, "
          f"tol={a.tol_nm:.0f} nm ({tol_px:.2f} px)\n")

    def score(locs):
        d = detection_metrics(locs, gt, tol_px, px_um)
        return d["precision"], d["recall"], d["f1"], len(locs), d["n_matched"]

    rows = []
    hdr = f"{'engine':16s} {'thresh':>9s} {'prec':>6s} {'recall':>6s} {'F1':>6s} {'n_det':>7s}"
    print(hdr)
    for mm in (0.02, 0.05, 0.08, 0.12, 0.2, 0.35, 0.6, 1.0):
        out = preprocess_and_localise_adaptive(
            stack, diameter=7, minmass=mm, percentile=64,
            workers=a.workers, backend="torch")
        p, r, f, nd, _ = score(out[0])
        rows.append(("Crocker–Grier", r, p, f))
        print(f"{'Crocker–Grier':16s} {('mm='+str(mm)):>9s} {p:6.3f} {r:6.3f} {f:6.3f} {nd:7d}")
    for k in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        AtrousWaveletBackend._ATROUS_K_SIGMA = float(k)
        out = preprocess_and_localise_adaptive(
            stack, diameter=7, minmass=0.02, percentile=64,
            workers=a.workers, backend="atrous")
        p, r, f, nd, _ = score(out[0])
        rows.append(("à trous", r, p, f))
        print(f"{'à trous':16s} {('k='+str(k)):>9s} {p:6.3f} {r:6.3f} {f:6.3f} {nd:7d}")

    for name in ("Crocker–Grier", "à trous"):
        best = max((row for row in rows if row[0] == name), key=lambda z: z[3])
        print(f"  best F1 — {name:14s}: {best[3]:.3f}  (recall {best[1]:.3f}, prec {best[2]:.3f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, mk in (("Crocker–Grier", "o-"), ("à trous", "s-")):
        pts = sorted((row[1], row[2]) for row in rows if row[0] == name)  # (recall, prec)
        ax.plot([q[0] for q in pts], [q[1] for q in pts], mk, label=name, lw=2, ms=7)
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend()
    ax.set_title(f"Detection P–R: Crocker–Grier vs à trous  (tol {a.tol_nm:.0f} nm)")
    fig.tight_layout(); fig.savefig(a.out, dpi=140)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
