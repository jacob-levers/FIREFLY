"""LAP (linear-assignment) trajectory linker — a Jaqaman-style two-step tracker.

A higher-fidelity alternative to the trackpy nearest-neighbour linker for dense /
blinking sptPALM data.  Two global optimisation steps (Jaqaman et al., Nat
Methods 2008, "u-track"):

  Step 1 — frame-to-frame linking: optimal assignment of points between
           consecutive frames (with birth/death) → gap-free track *segments*.
  Step 2 — segment gap-closing: a second global assignment links segment ENDS to
           later segment STARTS across blink gaps (within a time window + a
           diffusion-scaled spatial gate) → reconnected trajectories.

Where trackpy resolves links greedily across its `memory` window (and mis-links
under density), both steps here are solved globally with the Hungarian/Jonker-
Volgenant algorithm, so ambiguous links are resolved to the joint optimum.

Qt-free: numpy + scipy + pandas only (CI-safe).  Returns the same contract as
`link_trajectories`: the input frame with an added integer `particle` column.

Merge/split events are intentionally NOT modelled: single fluorophores do not
physically coalesce, so for sptPALM the high-value step is gap-closing.  The
gap-closing matrix can be extended with merge/split blocks if a use case needs it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

_BIG = 1e12


def _solve_birth_death(link_cost: np.ndarray, alt: float) -> np.ndarray:
    """Solve one assignment with birth/death escapes.

    `link_cost` is an (n, m) matrix of source→target costs (ungated entries must
    already be `>= _BIG`).  Builds the (n+m)x(n+m) augmented matrix
        [[ link , death ],
         [ birth, 0     ]]
    where death/birth are alt-cost diagonals and the lower-right is free, then
    returns, for each source i, the matched target j or -1 (died / no link)."""
    n, m = link_cost.shape
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if m == 0:
        return np.full(n, -1, dtype=np.int64)
    C = np.full((n + m, n + m), _BIG, dtype=float)
    C[:n, :m] = link_cost
    di = np.arange(n)
    C[di, m + di] = alt                 # source i → death dummy
    bj = np.arange(m)
    C[n + bj, bj] = alt                 # birth dummy → target j
    C[n:, m:] = 0.0                     # dummy ↔ dummy (free, makes it solvable)
    r, c = linear_sum_assignment(C)
    out = np.full(n, -1, dtype=np.int64)
    for ri, ci in zip(r, c):
        if ri < n and ci < m and link_cost[ri, ci] < _BIG:
            out[ri] = ci
    return out


def _frame_to_frame(frame: np.ndarray, x: np.ndarray, y: np.ndarray,
                    search_range: float) -> np.ndarray:
    """Link points between CONSECUTIVE frames into gap-free segments.
    Returns a segment id per point."""
    n = len(frame)
    seg = np.full(n, -1, dtype=np.int64)
    sr2 = float(search_range) ** 2
    uframes = np.unique(frame)
    idx_by_frame = {int(f): np.where(frame == f)[0] for f in uframes}
    next_seg = 0
    prev_f = None
    prev_idx = np.empty(0, dtype=np.int64)
    for f in uframes:
        cur = idx_by_frame[int(f)]
        if prev_f is not None and int(f) == prev_f + 1 and len(prev_idx):
            xa, ya = x[prev_idx], y[prev_idx]
            xb, yb = x[cur], y[cur]
            d2 = ((xa[:, None] - xb[None, :]) ** 2 +
                  (ya[:, None] - yb[None, :]) ** 2)
            link = np.where(d2 <= sr2, d2, _BIG)
            assign = _solve_birth_death(link, alt=sr2)
            used = np.zeros(len(cur), dtype=bool)
            for k, tj in enumerate(assign):
                if tj >= 0:
                    seg[cur[tj]] = seg[prev_idx[k]]    # extend the segment
                    used[tj] = True
            for j in np.where(~used)[0]:
                seg[cur[j]] = next_seg; next_seg += 1   # birth
        else:
            for p in cur:
                seg[p] = next_seg; next_seg += 1
        prev_f = int(f)
        prev_idx = cur
    return seg


def _gap_close(seg: np.ndarray, frame: np.ndarray, x: np.ndarray, y: np.ndarray,
               search_range: float, max_gap: int) -> np.ndarray:
    """Globally link segment ends → later segment starts across gaps.
    Returns a new label per point (segments merged transitively)."""
    seg_ids = np.unique(seg)
    S = len(seg_ids)
    if S <= 1 or max_gap <= 0:
        return seg
    remap = {int(s): i for i, s in enumerate(seg_ids)}
    end_f = np.empty(S); end_x = np.empty(S); end_y = np.empty(S)
    sta_f = np.empty(S); sta_x = np.empty(S); sta_y = np.empty(S)
    for s in seg_ids:
        si = remap[int(s)]
        m = seg == s
        ff = frame[m]; xx = x[m]; yy = y[m]
        o = np.argsort(ff, kind="stable")
        sta_f[si], sta_x[si], sta_y[si] = ff[o[0]], xx[o[0]], yy[o[0]]
        end_f[si], end_x[si], end_y[si] = ff[o[-1]], xx[o[-1]], yy[o[-1]]

    # cost(end a -> start b): gated by gap window and a diffusion-scaled radius
    # (Brownian spread grows ~sqrt(gap)); cost = squared distance.
    sr2 = float(search_range) ** 2
    gap = sta_f[None, :] - end_f[:, None]            # (S_end a, S_start b)
    d2 = ((end_x[:, None] - sta_x[None, :]) ** 2 +
          (end_y[:, None] - sta_y[None, :]) ** 2)
    # FIXED spatial gate, like trackpy's `memory`: the re-link radius must NOT
    # blow up with the gap.  The per-frame diffusion step is << search_range, so
    # a search_range disc already covers many-frame reconnections; a radius that
    # grows with the gap just over-connects (splices wrong segments) under
    # density — the exact failure the benchmark caught.
    gate2 = sr2
    ok = (gap >= 1) & (gap <= max_gap) & (d2 <= gate2)
    link = np.where(ok, d2, _BIG)
    alt = sr2                                        # only close within search_range
    assign = _solve_birth_death(link, alt=alt)

    # union-find over segment indices for the chosen end→start links
    parent = list(range(S))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in enumerate(assign):
        if b >= 0:
            parent[find(a)] = find(int(b))

    root_of_segidx = np.array([find(i) for i in range(S)])
    out = np.empty_like(seg)
    inv = {int(s): i for i, s in enumerate(seg_ids)}
    for p in range(len(seg)):
        out[p] = root_of_segidx[inv[int(seg[p])]]
    return out


def link_trajectories_lap(locs: pd.DataFrame, search_range: float = 5.0,
                          max_gap: int = 12, min_len: int = 2) -> pd.DataFrame:
    """Two-step LAP linker.  `max_gap` is the gap-closing horizon (the LAP
    analogue of trackpy's `memory`).  Returns `locs` with an integer `particle`
    column; tracks shorter than `min_len` points are dropped."""
    if locs is None or len(locs) == 0:
        cols = list(locs.columns) if locs is not None else ["x", "y", "frame"]
        if "particle" not in cols:
            cols = cols + ["particle"]
        return pd.DataFrame(columns=cols)
    for c in ("x", "y", "frame"):
        if c not in locs.columns:
            raise ValueError(f"link_trajectories_lap: missing column '{c}'")
    df = locs.reset_index(drop=True).copy()
    frame = df["frame"].to_numpy(np.int64)
    x = df["x"].to_numpy(float); y = df["y"].to_numpy(float)

    seg = _frame_to_frame(frame, x, y, search_range)
    lab = _gap_close(seg, frame, x, y, search_range, max_gap)

    # relabel to compact 0..K-1 and drop short tracks
    uniq, inv = np.unique(lab, return_inverse=True)
    counts = np.bincount(inv)
    keep_label = counts >= int(min_len)
    df["particle"] = inv
    df = df[keep_label[inv]].reset_index(drop=True)
    # compact particle ids after filtering
    if len(df):
        u2, inv2 = np.unique(df["particle"].to_numpy(), return_inverse=True)
        df["particle"] = inv2
    return df


# ── Kalman constant-velocity LAP tracker ─────────────────────────────────────
# Online tracker matching TrackMate's "Linear Motion" / Kalman LAP tracker: each
# track carries a 4-state constant-velocity Kalman filter [x, y, vx, vy].  Every
# frame we PREDICT each active track's next position from its velocity, assign
# detections to predictions by global LAP (birth/death), then CORRECT the linked
# filters.  Un-linked tracks COAST on prediction alone (gap-closing through
# blinks) for up to `max_gap` frames.  Unlike trackpy (no motion model) this
# follows a trajectory's momentum, so it keeps identities through crossings and
# fast/directed motion where nearest-position linking swaps or loses tracks.

_F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)


def link_trajectories_kalman(locs: pd.DataFrame, search_range: float = 5.0,
                             max_gap: int = 12, min_len: int = 2,
                             q_vel: float = 0.05, r_meas: float = 0.25,
                             p0_vel: float = 25.0) -> pd.DataFrame:
    """Constant-velocity Kalman LAP tracker.

    search_range : gate radius (px) around the PREDICTED position.
    max_gap      : frames a track may coast (predict-only) through a gap.
    q_vel        : process-noise variance on velocity (how fast velocity may
                   change; larger = more Brownian/agile, smaller = more ballistic).
    r_meas       : measurement-noise variance (localisation, px²).
    p0_vel       : initial velocity variance for a new track (velocity unknown).
    """
    if locs is None or len(locs) == 0:
        cols = list(locs.columns) if locs is not None else ["x", "y", "frame"]
        if "particle" not in cols:
            cols = cols + ["particle"]
        return pd.DataFrame(columns=cols)
    for c in ("x", "y", "frame"):
        if c not in locs.columns:
            raise ValueError(f"link_trajectories_kalman: missing column '{c}'")
    df = locs.reset_index(drop=True).copy()
    frame = df["frame"].to_numpy(np.int64)
    x = df["x"].to_numpy(float); y = df["y"].to_numpy(float)
    uframes = np.unique(frame)
    idx_by_frame = {int(f): np.where(frame == f)[0] for f in uframes}

    Q = np.diag([1e-4, 1e-4, float(q_vel), float(q_vel)])
    R = np.diag([float(r_meas), float(r_meas)])
    I4 = np.eye(4)
    sr2 = float(search_range) ** 2

    # each track: dict(state(4,), P(4,4), missed:int, pts:list[int])
    tracks: list[dict] = []
    active: list[dict] = []

    def _birth(j_global):
        st = np.array([x[j_global], y[j_global], 0.0, 0.0])
        P = np.diag([float(r_meas), float(r_meas), p0_vel, p0_vel])
        t = {"state": st, "P": P, "missed": 0, "pts": [int(j_global)]}
        tracks.append(t); active.append(t)

    for f in uframes:
        det = idx_by_frame[int(f)]
        if active:
            preds = np.empty((len(active), 2))
            for k, t in enumerate(active):
                t["state"] = _F @ t["state"]
                t["P"] = _F @ t["P"] @ _F.T + Q
                preds[k] = t["state"][:2]
            if len(det):
                dz = np.stack([x[det], y[det]], axis=1)
                d2 = ((preds[:, None, 0] - dz[None, :, 0]) ** 2 +
                      (preds[:, None, 1] - dz[None, :, 1]) ** 2)
                link = np.where(d2 <= sr2, d2, _BIG)
                assign = _solve_birth_death(link, alt=sr2)
            else:
                assign = np.full(len(active), -1, dtype=np.int64)
            used = np.zeros(len(det), dtype=bool)
            still: list[dict] = []
            for k, t in enumerate(active):
                j = int(assign[k]) if k < len(assign) else -1
                if j >= 0:
                    z = dz[j]
                    S = _H @ t["P"] @ _H.T + R
                    K = t["P"] @ _H.T @ np.linalg.inv(S)
                    t["state"] = t["state"] + K @ (z - _H @ t["state"])
                    t["P"] = (I4 - K @ _H) @ t["P"]
                    t["missed"] = 0; t["pts"].append(int(det[j])); used[j] = True
                    still.append(t)
                else:
                    t["missed"] += 1
                    if t["missed"] <= max_gap:
                        still.append(t)        # coast (prediction kept)
            active[:] = still
            for j in np.where(~used)[0]:
                _birth(int(det[j]))
        else:
            for j in det:
                _birth(int(j))

    labels = np.full(len(df), -1, dtype=np.int64)
    pid = 0
    for t in tracks:
        if len(t["pts"]) >= int(min_len):
            for p in t["pts"]:
                labels[p] = pid
            pid += 1
    out = df[labels >= 0].copy()
    out["particle"] = labels[labels >= 0]
    return out.reset_index(drop=True)
