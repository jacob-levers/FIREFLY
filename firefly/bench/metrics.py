"""Standard benchmark metrics for SPT/SMLM (Qt-free; numpy + scipy + pandas).

Detection  — precision/recall/F1/Jaccard via optimal bipartite (Hungarian)
             assignment per frame, gated by a spatial tolerance (Sage et al.,
             Nat Methods 2015 — SMLM software challenge scoring).
Localisation precision — lateral RMSE (nm) of accepted true-positive pairs.
Tracking   — ISBI 2012 Particle Tracking Challenge metrics α, β, JSC, JSCθ,
             RMSE (Chenouard et al., Nat Methods 2014): a global track-to-track
             optimal assignment over a dummy-padded cost matrix.
Diffusion recovery — recovered vs known D/α per ground-truth population, plus a
             motion-class confusion matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

_BIG = 1e9


# ── detection + localisation precision ────────────────────────────────────────
def _match_points(est_xy: np.ndarray, gt_xy: np.ndarray, tol: float):
    """Optimal gated bipartite match. Returns (dx, dy) of accepted pairs
    (est − gt) plus counts of matched pairs."""
    if len(est_xy) == 0 or len(gt_xy) == 0:
        return np.empty(0), np.empty(0)
    d = np.hypot(est_xy[:, None, 0] - gt_xy[None, :, 0],
                 est_xy[:, None, 1] - gt_xy[None, :, 1])
    cost = np.where(d <= tol, d, _BIG)
    r, c = linear_sum_assignment(cost)
    keep = d[r, c] <= tol
    r, c = r[keep], c[keep]
    return est_xy[r, 0] - gt_xy[c, 0], est_xy[r, 1] - gt_xy[c, 1]


def detection_metrics(est_locs: pd.DataFrame, gt_locs: pd.DataFrame,
                      tol_px: float, pixel_size_um: float | None = None) -> dict:
    """Per-frame Hungarian matching → precision/recall/F1/Jaccard and the lateral
    RMSE of the matched (true-positive) pairs."""
    gg = {int(f): g[["x", "y"]].to_numpy(float)
          for f, g in gt_locs.groupby("frame")} if len(gt_locs) else {}
    eg = {int(f): g[["x", "y"]].to_numpy(float)
          for f, g in est_locs.groupby("frame")} if len(est_locs) else {}
    tp = fp = fn = 0
    dxs, dys = [], []
    for f in set(gg) | set(eg):
        e = eg.get(f, np.empty((0, 2)))
        g = gg.get(f, np.empty((0, 2)))
        dx, dy = _match_points(e, g, tol_px)
        ntp = len(dx)
        tp += ntp
        fp += len(e) - ntp
        fn += len(g) - ntp
        dxs.append(dx); dys.append(dy)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    jaccard = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    dx = np.concatenate(dxs) if dxs else np.empty(0)
    dy = np.concatenate(dys) if dys else np.empty(0)
    out = dict(precision=precision, recall=recall, f1=f1, jaccard=jaccard,
               tp=int(tp), fp=int(fp), fn=int(fn), n_matched=int(len(dx)))
    if len(dx):
        rmse_px = float(np.sqrt(np.mean(dx ** 2 + dy ** 2)))
        scale = (pixel_size_um or 0.0) * 1000.0
        out.update(rmse_px=rmse_px,
                   rmse_nm=rmse_px * scale if pixel_size_um else float("nan"),
                   rmse_x_nm=float(np.sqrt(np.mean(dx ** 2))) * scale if pixel_size_um else float("nan"),
                   rmse_y_nm=float(np.sqrt(np.mean(dy ** 2))) * scale if pixel_size_um else float("nan"),
                   median_err_nm=float(np.median(np.hypot(dx, dy))) * scale if pixel_size_um else float("nan"))
    else:
        out.update(rmse_px=float("nan"), rmse_nm=float("nan"),
                   rmse_x_nm=float("nan"), rmse_y_nm=float("nan"),
                   median_err_nm=float("nan"))
    return out


def localisation_rmse(est_locs, gt_locs, tol_px, pixel_size_um) -> dict:
    """Convenience subset of detection_metrics: just the precision numbers."""
    m = detection_metrics(est_locs, gt_locs, tol_px, pixel_size_um)
    return {k: m[k] for k in ("rmse_nm", "rmse_x_nm", "rmse_y_nm",
                              "median_err_nm", "n_matched")}


def crlb_sigma_nm(photons: float, bg_photons: float, psf_sigma_px: float,
                  pixel_size_um: float) -> float:
    """Thompson/Mortensen-style Gaussian-PSF CRLB lateral precision (nm), as a
    reference floor on a tool's achievable RMSE at the given photon budget."""
    a = 1.0                                  # pixel size in pixel units
    s = psf_sigma_px
    sa2 = s * s + a * a / 12.0               # finite-pixel-corrected variance (px²)
    if photons <= 0:
        return float("nan")
    # Mortensen 2010 Eq. 54 (2D, background-limited correction)
    tau = 2.0 * np.pi * (sa2) * max(bg_photons, 1e-9) / (photons * a * a)
    var_px = (sa2 / photons) * (1.0 + 4.0 * tau + np.sqrt(max(2.0 * tau / (1.0 + 4.0 * tau), 0.0)))
    return float(np.sqrt(var_px) * pixel_size_um * 1000.0)


# ── ISBI 2012 tracking metrics ────────────────────────────────────────────────
def _track_dict(tracks: pd.DataFrame) -> dict:
    """particle id → {frame: (x, y)}."""
    out = {}
    if tracks is None or len(tracks) == 0 or "particle" not in tracks.columns:
        return out
    for pid, g in tracks.groupby("particle"):
        out[int(pid)] = {int(f): (float(x), float(y))
                         for f, x, y in zip(g["frame"], g["x"], g["y"])}
    return out


def _pair_cost(a: dict, b: dict, eps: float) -> tuple[float, int, float]:
    """Gated track-distance between two tracks. Returns
    (cost, n_matched_points, sum_sq_dist_of_matched)."""
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
                cost += eps                # too far → counts as a miss
        else:
            cost += eps                    # present in one track only
    return cost, n_match, ssd


def tracking_isbi(est_tracks: pd.DataFrame, gt_tracks: pd.DataFrame,
                  gate_px: float, pixel_size_um: float) -> dict:
    """ISBI 2012 Particle Tracking Challenge scoring (Chenouard et al. 2014).

    Global optimal assignment of GT tracks ↔ estimated tracks over a dummy-
    padded cost matrix; from the pairing we derive α (normalised distance, GT
    side), β (α penalised by spurious estimated tracks), JSC (point-level
    Jaccard), JSCθ (track-level Jaccard) and matched-point RMSE.
    """
    G = _track_dict(gt_tracks)
    E = _track_dict(est_tracks)
    gids, eids = list(G), list(E)
    ng, ne = len(gids), len(eids)
    eps = float(gate_px)
    gt_pts = sum(len(G[g]) for g in gids)
    dummy_cost = eps * gt_pts                       # d(θ, ∅) — all GT unmatched

    if ng == 0:
        return dict(alpha=float("nan"), beta=float("nan"), jsc=float("nan"),
                    jsc_theta=float("nan"), rmse_nm=float("nan"),
                    n_gt_tracks=0, n_est_tracks=ne, n_paired=0, pairs=[])
    if ne == 0:
        return dict(alpha=0.0, beta=0.0, jsc=0.0, jsc_theta=0.0,
                    rmse_nm=float("nan"), n_gt_tracks=ng, n_est_tracks=0,
                    n_paired=0, pairs=[])

    gt_dummy = np.array([eps * len(G[g]) for g in gids])
    est_dummy = np.array([eps * len(E[e]) for e in eids])
    pc = np.empty((ng, ne))
    for i, g in enumerate(gids):
        for j, e in enumerate(eids):
            pc[i, j] = _pair_cost(G[g], E[e], eps)[0]

    # padded square matrix: real GT/EST + one dummy per real track
    n = ng + ne
    C = np.full((n, n), _BIG)
    C[:ng, :ne] = pc
    for i in range(ng):
        C[i, ne + i] = gt_dummy[i]                  # GT_i → its dummy
    for j in range(ne):
        C[ng + j, j] = est_dummy[j]                 # EST_j → its dummy
    C[ng:, ne:] = 0.0                               # dummy ↔ dummy
    ri, ci = linear_sum_assignment(C)

    pairs = []
    d_gt = 0.0          # GT-side optimal distance d(θ, θ̂)
    spurious = 0.0
    tp_pts = fn_pts = fp_pts = 0
    ssd = 0.0
    paired_est = set()
    for r, c in zip(ri, ci):
        if r < ng and c < ne:                       # real GT ↔ real EST
            g, e = gids[r], eids[c]
            cost, nm, s = _pair_cost(G[g], E[e], eps)
            d_gt += cost
            tp_pts += nm
            fn_pts += len(G[g]) - nm
            fp_pts += len(E[e]) - nm
            ssd += s
            # Track-level TP only when the two tracks actually share ≥1 gated
            # point.  A zero-overlap real↔real assignment can be optimal (its cost
            # ties the dummy cost), but it is NOT a true track association — count
            # it as a missed GT track + a spurious EST track for JSCθ.  α/β/JSC
            # already charge every unmatched point above, so they are unaffected.
            if nm > 0:
                pairs.append((g, e))
                paired_est.add(c)
        elif r < ng and c >= ne:                    # GT → dummy (missed track)
            d_gt += gt_dummy[r]
            fn_pts += len(G[gids[r]])
        elif r >= ng and c < ne:                    # EST → dummy (spurious track)
            spurious += est_dummy[c]
            fp_pts += len(E[eids[c]])

    alpha = 1.0 - d_gt / dummy_cost if dummy_cost > 0 else float("nan")
    beta = ((dummy_cost - d_gt) / (dummy_cost + spurious)
            if (dummy_cost + spurious) > 0 else float("nan"))
    jsc = tp_pts / (tp_pts + fn_pts + fp_pts) if (tp_pts + fn_pts + fp_pts) else float("nan")
    tp_tracks = len(pairs)
    fn_tracks = ng - tp_tracks
    fp_tracks = ne - len(paired_est)
    jsc_theta = (tp_tracks / (tp_tracks + fn_tracks + fp_tracks)
                 if (tp_tracks + fn_tracks + fp_tracks) else float("nan"))
    rmse_nm = (float(np.sqrt(ssd / tp_pts)) * pixel_size_um * 1000.0
               if tp_pts else float("nan"))
    return dict(alpha=float(alpha), beta=float(beta), jsc=float(jsc),
                jsc_theta=float(jsc_theta), rmse_nm=rmse_nm,
                n_gt_tracks=ng, n_est_tracks=ne, n_paired=tp_tracks, pairs=pairs)


# ── diffusion recovery ────────────────────────────────────────────────────────
_POP_TO_MOTION = {"immobile": "Immobile", "confined": "Confined",
                  "brownian": "Brownian", "directed": "Directed"}


def diffusion_recovery(est_diff: pd.DataFrame, gt_tracks: pd.DataFrame,
                       track_pairs: list, meta: dict) -> dict:
    """Recovered vs known D/α per ground-truth population + motion-class confusion.

    `track_pairs` is the (gt_particle, est_particle) list from `tracking_isbi`.
    `est_diff` is `compute_msd_and_fit`'s diff_df (per-particle D, alpha, motion).
    """
    if est_diff is None or len(est_diff) == 0 or not track_pairs:
        return dict(per_population={}, confusion={})
    diff = est_diff.set_index("particle") if "particle" in est_diff.columns else est_diff
    pop_of = {int(p): str(pop) for p, pop in
              zip(gt_tracks["particle"], gt_tracks["population"])}
    true_D = {k.lower(): v.get("D_um2_s") for k, v in meta.get("populations", {}).items()}

    by_pop: dict[str, dict] = {}
    confusion: dict = {}
    for gp, ep in track_pairs:
        if ep not in diff.index:
            continue
        pop = pop_of.get(int(gp), "?").lower()
        row = diff.loc[ep]
        D_est = float(np.atleast_1d(row.get("D"))[0]) if "D" in diff.columns else np.nan
        a_est = float(np.atleast_1d(row.get("alpha"))[0]) if "alpha" in diff.columns else np.nan
        m_est = str(np.atleast_1d(row.get("motion"))[0]) if "motion" in diff.columns else "?"
        b = by_pop.setdefault(pop, {"D_est": [], "D_true": [], "alpha_est": []})
        b["D_est"].append(D_est)
        b["D_true"].append(true_D.get(pop, np.nan))
        b["alpha_est"].append(a_est)
        m_true = _POP_TO_MOTION.get(pop, "?")
        confusion.setdefault(m_true, {}).setdefault(m_est, 0)
        confusion[m_true][m_est] += 1

    per_pop = {}
    for pop, b in by_pop.items():
        D_est = np.array(b["D_est"], float)
        D_true = np.array(b["D_true"], float)
        a_est = np.array(b["alpha_est"], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = D_est / D_true - 1.0
        per_pop[pop] = dict(
            n=int(len(D_est)),
            D_true=float(np.nanmedian(D_true)),
            D_est_median=float(np.nanmedian(D_est)),
            D_bias_pct=float(np.nanmedian(rel) * 100.0) if np.isfinite(rel).any() else float("nan"),
            D_iqr=float(np.nanpercentile(D_est, 75) - np.nanpercentile(D_est, 25)) if len(D_est) > 1 else 0.0,
            alpha_est_median=float(np.nanmedian(a_est)),
        )
    return dict(per_population=per_pop, confusion=confusion)
