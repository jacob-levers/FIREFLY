"""Test-only vendored ISBI-2012 tracking metric (Chenouard et al. 2014).

The benchmark engine now lives in the separate FIREFLY-VERIFICATION program, so
FIREFLY no longer ships `firefly.bench`.  This tiny copy keeps FIREFLY's own
linker-accuracy regression (test_linking_sa_palmtracer) self-contained — it needs
only the track-to-track α score, not the whole harness.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

_BIG = 1e9


def _track_dict(tracks):
    out = {}
    if tracks is None or len(tracks) == 0 or "particle" not in tracks.columns:
        return out
    for pid, g in tracks.groupby("particle"):
        out[int(pid)] = {int(f): (float(x), float(y))
                         for f, x, y in zip(g["frame"], g["x"], g["y"])}
    return out


def _pair_cost(a, b, eps):
    cost = 0.0
    n_match = 0
    ssd = 0.0
    for t in set(a) | set(b):
        pa, pb = a.get(t), b.get(t)
        if pa is not None and pb is not None:
            d = np.hypot(pa[0] - pb[0], pa[1] - pb[1])
            if d <= eps:
                cost += d; n_match += 1; ssd += d * d
            else:
                cost += eps
        else:
            cost += eps
    return cost, n_match, ssd


def tracking_isbi(est_tracks, gt_tracks, gate_px, pixel_size_um):
    """Subset of the ISBI metric: returns at least {'alpha': ...} (GT-side score)."""
    G = _track_dict(gt_tracks)
    E = _track_dict(est_tracks)
    gids, eids = list(G), list(E)
    ng, ne = len(gids), len(eids)
    eps = float(gate_px)
    gt_pts = sum(len(G[g]) for g in gids)
    dummy_cost = eps * gt_pts
    if ng == 0:
        return {"alpha": float("nan")}
    if ne == 0:
        return {"alpha": 0.0}
    gt_dummy = np.array([eps * len(G[g]) for g in gids])
    est_dummy = np.array([eps * len(E[e]) for e in eids])
    pc = np.empty((ng, ne))
    for i, g in enumerate(gids):
        for j, e in enumerate(eids):
            pc[i, j] = _pair_cost(G[g], E[e], eps)[0]
    n = ng + ne
    C = np.full((n, n), _BIG)
    C[:ng, :ne] = pc
    for i in range(ng):
        C[i, ne + i] = gt_dummy[i]
    for j in range(ne):
        C[ng + j, j] = est_dummy[j]
    C[ng:, ne:] = 0.0
    ri, ci = linear_sum_assignment(C)
    d_gt = 0.0
    for r, c in zip(ri, ci):
        if r < ng and c < ne:
            d_gt += _pair_cost(G[gids[r]], E[eids[c]], eps)[0]
        elif r < ng and c >= ne:
            d_gt += gt_dummy[r]
    alpha = 1.0 - d_gt / dummy_cost if dummy_cost > 0 else float("nan")
    return {"alpha": float(alpha)}
