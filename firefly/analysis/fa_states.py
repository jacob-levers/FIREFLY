"""Diffusive states by hidden Markov modelling — the vbSPT analysis.

vbSPT (Persson et al. 2013, Nat Methods 10:265) treats every trajectory as a
walk that switches between hidden states, each with its own diffusion
coefficient, and fits the states and the per-frame switching probabilities to
all of a recording's steps at once.  Because it uses every step of every
trajectory, it works on the short trajectories of in-vivo sptPALM, where a
per-trajectory MSD exponent cannot (see tests/test_diffusive_states.py).

In sptPALM work it is run per recording and the three-state model analysed:
immobile, slow mobile and fast mobile states, each with an apparent diffusion
coefficient, an occupancy and transition probabilities per frame, compared
across recordings.  This module does the same:

* steps are taken between consecutive frames only; a trajectory is split where
  the linker bridged a gap (vbSPT has no notion of a missing frame);
* a step's 2-D displacement d in state k is Gaussian with variance 2·D_k·Δt
  per axis, D_k being the APPARENT diffusion coefficient (localisation error
  not removed — as in vbSPT and the lab's papers);
* the model is fitted by maximum likelihood (Baum–Welch EM).  With a
  recording's tens of thousands of steps this is where vbSPT's variational
  posterior sits; vbSPT's Bayesian model selection is replaced by BIC, used
  only to report how many states the data support — the analysis itself is
  the three-state model;
* states are ordered by D: immobile < slow mobile < fast mobile; occupancy is
  the fraction of steps spent in each state (posterior-weighted).
"""
from __future__ import annotations

import numpy as np

STATE_NAMES = ("Immobile", "Slow mobile", "Fast mobile")
STATE_KEYS = ("immobile", "slow", "fast")
# Okabe–Ito blue, bluish green, vermillion: distinct for colour-blind readers.
STATE_COLORS = ("#0072b2", "#009e73", "#d55e00")


def step_segments(tracks, pixel_size_um, frame_interval_s=None):
    """Squared consecutive-frame step lengths (µm²), one array per gap-free
    stretch of each trajectory.  A trajectory is split where frames are not
    consecutive; stretches with no step are dropped."""
    if tracks is None or not len(tracks):
        return []
    cols = getattr(tracks, "columns", ())
    if not {"particle", "frame", "x", "y"} <= set(cols):
        return []
    t = tracks[["particle", "frame", "x", "y"]].dropna().sort_values(["particle", "frame"])
    p = t["particle"].to_numpy()
    f = t["frame"].to_numpy(dtype=float)
    xy = t[["x", "y"]].to_numpy(dtype=float) * float(pixel_size_um)
    same = (p[1:] == p[:-1]) & (np.diff(f) == 1)
    r2 = np.sum(np.diff(xy, axis=0) ** 2, axis=1)
    out, start = [], 0
    for brk in np.flatnonzero(~same):            # segment boundaries between rows brk, brk+1
        if brk > start:
            out.append(r2[start:brk])
        start = brk + 1
    if len(r2) > start:
        out.append(r2[start:])
    return [s for s in out if len(s)]


# Trajectories are cut into pieces of at most this many steps before fitting.
# The forward–backward pass runs over trajectories padded to a common length,
# so one 5,000-frame immobile trajectory would otherwise pad every other one to
# 5,000 (gigabytes on a real recording).  A cut loses one transition per piece —
# negligible — and each piece is treated as its own trajectory, as vbSPT treats
# trajectories.
MAX_PIECE_STEPS = 64


def _pieces(segments, max_len=MAX_PIECE_STEPS):
    out = []
    for s in segments:
        out += [s[i:i + max_len] for i in range(0, len(s), max_len)]
    return out


def _padded(segments):
    lengths = np.array([len(s) for s in segments])
    order = np.argsort(-lengths, kind="stable")
    lengths = lengths[order]
    R = np.zeros((len(segments), int(lengths.max())))
    for row, i in enumerate(order):
        R[row, :lengths[row]] = segments[i]
    active = np.array([(lengths > t).sum() for t in range(R.shape[1])])
    return R, lengths, active


def _forward_backward(R, lengths, active, logB_fn, pi, A):
    """Scaled forward–backward over trajectories padded to a common length and
    sorted longest first (so the trajectories still running at step t are the
    first ``active[t]`` rows).  Returns (gamma, xi_sum, loglik)."""
    n, T = R.shape
    K = len(pi)
    logB = logB_fn(R)                                    # (n, T, K)
    m = logB.max(axis=2)
    B = np.exp(logB - m[..., None])
    alpha = np.zeros((n, T, K))
    c = np.ones((n, T))
    a = pi[None, :] * B[:, 0]
    c[:, 0] = a.sum(1)
    alpha[:, 0] = a / c[:, 0, None]
    for t in range(1, T):
        k = active[t]
        a = (alpha[:k, t - 1] @ A) * B[:k, t]
        c[:k, t] = a.sum(1)
        alpha[:k, t] = a / c[:k, t, None]
    beta = np.ones((n, T, K))
    xi_sum = np.zeros((K, K))
    for t in range(T - 2, -1, -1):
        k = active[t + 1]
        bb = B[:k, t + 1] * beta[:k, t + 1]
        beta[:k, t] = (bb @ A.T) / c[:k, t + 1, None]
        xi_sum += A * np.einsum("ij,ik->jk", alpha[:k, t], bb / c[:k, t + 1, None])
    valid = np.arange(T)[None, :] < lengths[:, None]
    gamma = alpha * beta * valid[..., None]
    loglik = float(np.sum((np.log(c) + m) * valid))
    return gamma, xi_sum, loglik


def _buckets(pieces):
    """Pieces grouped by length (1–2, 3–4, 5–8 … 33–64 steps), each group padded
    only to its own longest: most gap-free stretches are a few steps long, and
    padding them all to 64 wasted ~90% of the work on a real recording."""
    edges = [0, 2, 4, 8, 16, 32, MAX_PIECE_STEPS]
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        group = [p for p in pieces if lo < len(p) <= hi]
        if group:
            out.append(_padded(group))
    return out


def fit_hmm(segments, frame_interval_s, n_states=3, max_iter=300, tol=1e-6):
    """Maximum-likelihood K-state diffusive HMM of ``segments`` (squared step
    lengths, µm²).  Returns a dict, states ordered by D ascending: ``D``
    (µm²/s), ``transition`` (per frame), ``occupancy`` (fraction of steps),
    ``initial``, ``loglik``, ``n_steps``, ``n_segments``, ``bic``.  EM stops
    when the log-likelihood changes by less than ``tol`` of itself."""
    dt = float(frame_interval_s)
    segments = [np.asarray(s, float) for s in segments if len(s)]
    if not segments:
        return None
    n_traj = len(segments)
    pieces = _pieces(segments)
    buckets = _buckets(pieces)
    allr = np.concatenate(segments)
    n_steps = len(allr)
    K = int(n_states)
    d_obs = np.maximum(allr / (4 * dt), 1e-12)
    D = np.quantile(d_obs, (np.arange(K) + 0.5) / K) if K > 1 else np.array([d_obs.mean()])
    D = np.maximum(np.sort(D), 1e-9)
    if K > 1 and np.any(np.diff(D) <= 0):        # ties in heavily quantised data
        D = D * np.geomspace(1.0, 4.0, K)
    A = np.full((K, K), 0.1 / max(K - 1, 1)) + np.eye(K) * (0.9 - 0.1 / max(K - 1, 1))
    if K == 1:
        A = np.ones((1, 1))
    pi = np.full(K, 1.0 / K)
    prev = -np.inf
    for _ in range(max_iter):
        logB_fn = lambda X, D=D: (-np.log(4 * np.pi * D * dt)[None, None, :]
                                  - X[..., None] / (4 * D * dt)[None, None, :])
        w = np.zeros(K); wr = np.zeros(K); g0 = np.zeros(K); xi = np.zeros((K, K)); ll = 0.0
        for R, lengths, active in buckets:
            gamma, xi_b, ll_b = _forward_backward(R, lengths, active, logB_fn, pi, A)
            w += gamma.sum(axis=(0, 1))
            wr += (gamma * R[..., None]).sum(axis=(0, 1))       # padded γ are 0
            g0 += gamma[:, 0].sum(0)
            xi += xi_b
            ll += ll_b
        pi = g0 / len(pieces)
        if K > 1:
            A = xi / np.maximum(xi.sum(1, keepdims=True), 1e-300)
        D = np.maximum(wr / np.maximum(4 * dt * w, 1e-300), 1e-9)
        if abs(ll - prev) <= tol * abs(ll):
            break
        prev = ll
    order = np.argsort(D)
    occ = w[order] / w.sum()
    n_par = K + K * (K - 1) + (K - 1)
    return {
        "n_states": K,
        "D": D[order].tolist(),
        "transition": A[np.ix_(order, order)].tolist(),
        "initial": pi[order].tolist(),
        "occupancy": occ.tolist(),
        "loglik": ll,
        "bic": -2 * ll + n_par * np.log(n_steps),
        "n_steps": int(n_steps),
        "n_segments": int(n_traj),
    }


def diffusive_states(tracks, pixel_size_um, frame_interval_s, *, n_states=3,
                     k_max=4, min_steps=200):
    """The vbSPT-style analysis of one recording: the ``n_states``-state model
    (immobile / slow mobile / fast mobile for 3), plus the number of states BIC
    prefers among 1..``k_max`` for reference (``k_max=None`` skips that — a
    third of the time).  None when the recording has fewer than ``min_steps``
    consecutive-frame steps."""
    segs = step_segments(tracks, pixel_size_um)
    if sum(len(s) for s in segs) < min_steps:
        return None
    main = fit_hmm(segs, frame_interval_s, n_states)
    bic = {}
    for k in range(1, int(k_max or 0) + 1):
        fit = main if k == n_states else fit_hmm(segs, frame_interval_s, k)
        if fit is not None:
            bic[k] = fit["bic"]
    main["best_n_states"] = int(min(bic, key=bic.get)) if bic else None
    main["bic_by_n_states"] = {str(k): float(v) for k, v in bic.items()}
    if n_states == len(STATE_NAMES):
        main["state_names"] = list(STATE_NAMES)
    return main


def load_saved_states(data_dir, stem):
    """A run's saved ``{stem}_diffusive_states.json``, or None."""
    import json
    import os
    path = os.path.join(data_dir or "", f"{stem}_diffusive_states.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def state_summary(result):
    """Flat ``{occupancy_immobile, …, D_fast}`` for a three-state result
    (NaNs when there is none) — the per-recording scalars the comparison
    tests across groups."""
    out = {}
    ok = result is not None and result.get("n_states") == len(STATE_KEYS)
    for i, key in enumerate(STATE_KEYS):
        out[f"state_occupancy_{key}"] = float(result["occupancy"][i]) if ok else np.nan
        out[f"state_D_{key}"] = float(result["D"][i]) if ok else np.nan
    for i, a in enumerate(STATE_KEYS):           # per-frame transition probabilities
        for j, b in enumerate(STATE_KEYS):
            out[f"state_P_{a}_{b}"] = float(result["transition"][i][j]) if ok else np.nan
    out["state_best_n"] = (float(result["best_n_states"])
                           if ok and result.get("best_n_states") else np.nan)
    return out
