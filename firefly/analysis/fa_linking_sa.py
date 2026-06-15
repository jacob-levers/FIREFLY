"""Simulated-annealing multi-target tracker (palmTRACER-style).

INDEPENDENT, clean-room reimplementation of the *published method* — Racine &
Sibarita et al., "Multiple-Target Tracking of 3D Fluorescent Objects Based on
Simulated Annealing", IEEE ISBI 2006 — which is the documented basis of
palmTRACER's tracking (palmTRACER itself is a closed-source, proprietary
MetaMorph plugin and is NOT used or referenced here).  The paper specifies a
simulated-annealing optimiser over the frame-to-frame correspondence
configuration with birth/death (and optionally merge/split) events minimising a
global energy; the exact energy terms and cooling schedule are not published, so
the formulation below is a principled, fully-parameterised reconstruction whose
constants are meant to be calibrated against real data.

Energy of a configuration (each localisation has ≤1 predecessor and ≤1
successor):

    E = Σ_links [ w_disp·d²/σ² + C_gap0 + κ·(Δframe−1) + w_feat·featpenalty ]
        + n_births·C_birth + n_deaths·C_death

A link i→j is admissible only when 1 ≤ frame(j)−frame(i) ≤ max_gap and
‖xy_i−xy_j‖ ≤ search_range.  With the default costs (C_birth+C_death = σ-scaled
search_range²) a link is energetically favourable exactly when it is closer than
the search radius — so the global optimum is the low-displacement,
few-tracks configuration, found by annealing rather than a single LAP.

Qt-free (numpy/scipy/pandas).  Deterministic for a fixed ``seed`` (seeded RNG,
no set/dict iteration in the hot loop) and returns the best configuration seen
(so the result's energy never exceeds the greedy seed's).  Same contract as
``link_trajectories``: the input frame with an added integer ``particle`` column.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from firefly.analysis.fa_constants import _Cancelled
from firefly.analysis.fa_linking_lap import _stack_features


def _scalar_feat_penalty(fi, fj, weight):
    """Per-pair feature penalty Σ_f |fi−fj|/(fi+fj)·weight (≥0)."""
    if fi is None or fj is None or weight == 0:
        return 0.0
    s = 0.0
    for a, b in zip(fi, fj):
        denom = a + b
        if denom:
            s += abs(a - b) / denom
    return float(weight) * s


def _build_candidates(frame, x, y, feats, search_range, max_gap,
                      w_disp, sigma2, C_gap0, kappa, w_feat):
    """For each localisation i, the admissible successors j (1 ≤ Δframe ≤
    max_gap, distance ≤ search_range) with their precomputed link cost.
    Returns a list ``cand`` where ``cand[i]`` is a dict ``{j: cost}``."""
    n = len(frame)
    cand: list[dict] = [dict() for _ in range(n)]
    sr = float(search_range)
    uframes = np.unique(frame)
    idx_by_frame = {int(f): np.where(frame == f)[0] for f in uframes}
    trees = {int(f): cKDTree(np.column_stack([x[idx], y[idx]]))
             for f, idx in idx_by_frame.items()}
    fset = set(int(f) for f in uframes)
    for fi in uframes:
        fi = int(fi)
        src = idx_by_frame[fi]
        src_xy = np.column_stack([x[src], y[src]])
        for dfr in range(1, int(max_gap) + 1):
            fj = fi + dfr
            if fj not in fset:
                continue
            dst = idx_by_frame[fj]
            nbrs = trees[fj].query_ball_point(src_xy, r=sr)
            for a, js in enumerate(nbrs):
                i = int(src[a])
                for b in js:
                    j = int(dst[b])
                    d2 = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
                    cost = w_disp * d2 / sigma2 + C_gap0 + kappa * (dfr - 1)
                    if w_feat and feats is not None:
                        cost += _scalar_feat_penalty(feats[i], feats[j], w_feat)
                    cand[i][j] = float(cost)
    return cand


def _config_energy(succ, pred, cand, C_birth, C_death):
    """Total energy of a configuration (links + births/deaths)."""
    e = 0.0
    n = len(succ)
    for i in range(n):
        j = succ[i]
        if j >= 0:
            e += cand[i][j]
    n_birth = int(np.count_nonzero(pred < 0))
    n_death = int(np.count_nonzero(succ < 0))
    return e + n_birth * C_birth + n_death * C_death


def _nn_seed(frame, x, y, search_range, max_gap, n):
    """Frame-ordered greedy nearest-neighbour initial configuration (succ/pred).

    A global cost-sorted seed fragments tracks — it prefers a spatially-closer
    GAP link over the temporally-adjacent one — so SA starts in a poor basin its
    local moves can't escape.  Frame-adjacent-first NN linking (the same logic as
    ``link_trajectories_nn``) gives a much better starting point; SA refines from
    there and the best-config tracking guarantees it never does worse."""
    succ = np.full(n, -1, dtype=np.int64)
    pred = np.full(n, -1, dtype=np.int64)
    sr = float(search_range)
    uframes = np.unique(frame)
    idx_by_frame = {int(f): np.where(frame == f)[0] for f in uframes}
    active: list[int] = []                 # global idx of each active track's last point
    for f in uframes:
        f = int(f)
        det = idx_by_frame[f]
        active = [t for t in active if (f - int(frame[t])) <= max_gap]
        used = np.zeros(len(det), dtype=bool)
        if active and len(det):
            dx = x[det]; dy = y[det]
            tree = cKDTree(np.column_stack([dx, dy]))
            tl = np.array([[x[t], y[t]] for t in active], dtype=float)
            nbrs = tree.query_ball_point(tl, r=sr)
            pairs = []
            for k, js in enumerate(nbrs):
                for j in js:
                    d2 = (x[active[k]] - dx[j]) ** 2 + (y[active[k]] - dy[j]) ** 2
                    pairs.append((float(d2), int(j), k))
            pairs.sort()
            matched = {}
            done_trk = np.zeros(len(active), dtype=bool)
            for _d2, j, k in pairs:
                if done_trk[k] or used[j]:
                    continue
                jg = int(det[j])
                succ[active[k]] = jg; pred[jg] = active[k]
                done_trk[k] = True; used[j] = True
                matched[k] = jg
            active = [matched.get(k, t) for k, t in enumerate(active)]
        for j in range(len(det)):
            if not used[j]:
                active.append(int(det[j]))
    return succ, pred


def link_trajectories_sa(locs: pd.DataFrame, search_range: float = 2.0,
                         max_gap: int = 5, min_len: int = 5, *,
                         seed: int = 0, T0=None, cooling: float = 0.95,
                         moves_per_temp=None, T_min: float = 1e-3,
                         w_disp: float = 1.0, w_feat: float = 0.0,
                         sigma_px: float = 1.0, C_birth=None, C_death=None,
                         C_gap0: float = 0.0, kappa: float = 0.0,
                         feature_cols=("mass",), progress_cb=None,
                         stop_event=None) -> pd.DataFrame:
    """Simulated-annealing tracker.  Defaults follow palmTRACER's documented
    settings (search_range≈2 px, min trajectory 5).  ``max_gap`` is the max
    frame difference a link may span.  The energy constants (``C_birth`` /
    ``C_death`` default to a σ-scaled ``search_range²``; ``C_gap0`` / ``kappa``
    gap penalties; ``w_feat`` feature weight) and the schedule (``T0`` auto-tuned
    to ≈0.8 initial accept, geometric ``cooling``, ``moves_per_temp``) are
    calibration knobs.  Returns ``locs`` + integer ``particle``; tracks shorter
    than ``min_len`` are dropped."""
    if locs is None or len(locs) == 0:
        cols = list(locs.columns) if locs is not None else ["x", "y", "frame"]
        if "particle" not in cols:
            cols = cols + ["particle"]
        return pd.DataFrame(columns=cols)
    for c in ("x", "y", "frame"):
        if c not in locs.columns:
            raise ValueError(f"link_trajectories_sa: missing column '{c}'")

    df = locs.reset_index(drop=True).copy()
    frame = df["frame"].to_numpy(np.int64)
    x = df["x"].to_numpy(float); y = df["y"].to_numpy(float)
    n = len(df)
    feats = _stack_features(df, feature_cols) if w_feat else None
    sigma2 = float(sigma_px) ** 2 or 1.0
    sr2 = float(search_range) ** 2
    # Default birth/death so that a within-radius link beats two singletons:
    # link favourable ⇔ w·d²/σ² < C_birth + C_death ⇔ d² < sr² (with defaults).
    if C_birth is None:
        C_birth = 0.5 * sr2 * w_disp / sigma2
    if C_death is None:
        C_death = 0.5 * sr2 * w_disp / sigma2
    C_birth = float(C_birth); C_death = float(C_death)
    link_thresh = C_birth + C_death

    cand = _build_candidates(frame, x, y, feats, search_range, max_gap,
                             w_disp, sigma2, C_gap0, kappa, w_feat)
    has_cand = any(cand[i] for i in range(n))

    # Nearest-neighbour seed (deterministic) → SA refines from a good basin and
    # never does worse than this (we keep the best configuration seen).
    succ, pred = _nn_seed(frame, x, y, search_range, max_gap, n)
    cur_E = _config_energy(succ, pred, cand, C_birth, C_death)
    best_E = cur_E
    best_succ = succ.copy()

    rng = np.random.default_rng(int(seed))

    if has_cand and n > 1:
        # Auto-tune T0 to ≈0.8 acceptance of a worsening move: sample ΔE
        # magnitudes of random ADD proposals (cost − link_thresh).
        if T0 is None:
            sample = []
            srcs = [i for i in range(n) if cand[i]]
            for i in srcs[:512]:
                for j, c in cand[i].items():
                    sample.append(abs(c - link_thresh))
            m = float(np.mean(sample)) if sample else 1.0
            T0 = (m / (-math.log(0.8))) if m > 0 else 1.0
        T0 = float(T0)
        if moves_per_temp is None:
            moves_per_temp = max(200, 2 * n)
        moves_per_temp = int(moves_per_temp)

        def _cost(i, j):
            d = cand[i]
            return d[j] if j in d else None

        T = T0
        while T > T_min:
            for _ in range(moves_per_temp):
                mv = rng.integers(0, 3)
                if mv == 0:                                  # ADD
                    i = int(rng.integers(0, n))
                    if succ[i] >= 0 or not cand[i]:
                        continue
                    js = list(cand[i].keys())
                    j = int(js[int(rng.integers(0, len(js)))])
                    if pred[j] >= 0:
                        continue
                    dE = cand[i][j] - C_death - C_birth
                    if dE <= 0 or rng.random() < math.exp(-dE / T):
                        succ[i] = j; pred[j] = i; cur_E += dE
                elif mv == 1:                                # REMOVE
                    i = int(rng.integers(0, n))
                    j = succ[i]
                    if j < 0:
                        continue
                    dE = -cand[i][j] + C_death + C_birth
                    if dE <= 0 or rng.random() < math.exp(-dE / T):
                        succ[i] = -1; pred[j] = -1; cur_E += dE
                else:                                        # SWAP two links
                    a = int(rng.integers(0, n)); b = succ[a]
                    c = int(rng.integers(0, n)); d = succ[c]
                    if b < 0 or d < 0 or a == c or b == d:
                        continue
                    cad = _cost(a, d); ccb = _cost(c, b)
                    if cad is None or ccb is None:
                        continue
                    dE = cad + ccb - cand[a][b] - cand[c][d]
                    if dE <= 0 or rng.random() < math.exp(-dE / T):
                        succ[a] = d; pred[d] = a
                        succ[c] = b; pred[b] = c
                        cur_E += dE
                if cur_E < best_E:
                    best_E = cur_E; best_succ = succ.copy()
            if stop_event is not None and stop_event.is_set():
                raise _Cancelled()
            if progress_cb is not None:
                try:
                    progress_cb(min(1.0, math.log(T0 / max(T, T_min) + 1e-9) /
                                    max(1e-9, math.log(T0 / T_min))))
                except Exception:
                    pass
            T *= cooling

    # Trace tracks from the best configuration (follow succ from each birth).
    succ = best_succ
    pred = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if succ[i] >= 0:
            pred[succ[i]] = i
    labels = np.full(n, -1, dtype=np.int64)
    pid = 0
    for i in range(n):
        if pred[i] < 0:                       # a track start
            chain = []
            k = i
            while k >= 0:
                chain.append(k); k = succ[k]
            if len(chain) >= int(min_len):
                for k in chain:
                    labels[k] = pid
                pid += 1
    out = df[labels >= 0].copy()
    out["particle"] = labels[labels >= 0]
    return out.reset_index(drop=True)
