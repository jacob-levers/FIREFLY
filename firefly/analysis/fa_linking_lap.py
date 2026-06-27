"""LAP (linear-assignment) trajectory linker — a Jaqaman-style two-step tracker.

A higher-fidelity alternative to the trackpy nearest-neighbour linker for dense /
blinking sptPALM data.  Two global optimisation steps (Jaqaman et al., Nat
Methods 2008, "u-track"):

  Step 1 — frame-to-frame linking: optimal assignment of points between
           consecutive frames (with birth/death) → gap-free track *segments*.
  Step 2 — segment gap-closing: a second global assignment links segment ENDS to
           later segment STARTS across blink gaps (within a time window + a
           fixed search-radius spatial gate) → reconnected trajectories.

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
from scipy.spatial import cKDTree

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


def _feature_penalty(fi, fj, weight):
    """Feature-penalty multiplier ``P >= 1`` so the link cost becomes ``(D·P)²``
    (the TrackMate/Jaqaman cost FORM).  ``fi`` (n,F), ``fj`` (m,F) → (n,m).  Per
    feature, ``P += weight·|fi−fj|/(fi+fj)`` (ratio-based ⇒ scale-free); a spot
    pair with very different intensity/quality is penalised toward the no-link
    cost.

    NOTE: ``penalty_weight`` is a FIREFLY-specific knob, NOT directly comparable
    to TrackMate's same-named weight: TrackMate's per-feature penalty carries an
    extra factor of 3 (``p = 3·W·|f1−f2|/(f1+f2)``), so for the same numeric
    weight FIREFLY's penalty deviation from 1 is ~⅓ of TrackMate's.  (Off by
    default; the feature penalty is opt-in.)"""
    if fi is None or fj is None or weight == 0:
        return None
    n, F = fi.shape
    P = np.ones((n, fj.shape[0]), dtype=float)
    for c in range(F):
        a = fi[:, c][:, None]; b = fj[:, c][None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.abs(a - b) / (a + b)
        term[~np.isfinite(term)] = 0.0
        P = P + float(weight) * term
    return P


def _alt_cost(link, fallback):
    """Birth/death alternative cost: just above the worst finite link so any
    gated link beats a death/birth.  With pure squared-distance costs (no feature
    penalty) the caller passes ``fallback=sr2`` and never reaches here, keeping
    the legacy numeric path identical."""
    finite = link[link < _BIG]
    return float(1.05 * finite.max()) if finite.size else float(fallback)


def _stack_features(df, feature_cols):
    """(N,F) float array of the requested feature columns present in ``df`` (e.g.
    ``mass``), or None when none are available — disables the feature penalty."""
    cols = [c for c in feature_cols if c in df.columns]
    return df[cols].to_numpy(float) if cols else None


def _frame_to_frame(frame: np.ndarray, x: np.ndarray, y: np.ndarray,
                    search_range: float, feats=None,
                    penalty_weight: float = 1.0) -> np.ndarray:
    """Link points between CONSECUTIVE frames into gap-free segments.
    Returns a segment id per point.  With ``feats`` the link cost gains the
    ``(D·P)²`` feature penalty; without it the cost is plain squared distance
    (identical to the legacy path)."""
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
            if feats is not None:
                P = _feature_penalty(feats[prev_idx], feats[cur], penalty_weight)
                cost = d2 * (P * P) if P is not None else d2
                link = np.where(d2 <= sr2, cost, _BIG)
                assign = _solve_birth_death(link, alt=_alt_cost(link, sr2))
            else:
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


def _segment_ends(seg, frame, x, y, feats=None):
    """Per-segment start/end frame, position (and feature vector when ``feats``
    is given), plus the global point index of each start/end.  Shared by
    `_gap_close` and `_gap_close_full`."""
    seg_ids = np.unique(seg)
    S = len(seg_ids)
    remap = {int(s): i for i, s in enumerate(seg_ids)}
    end_f = np.empty(S); end_x = np.empty(S); end_y = np.empty(S)
    sta_f = np.empty(S); sta_x = np.empty(S); sta_y = np.empty(S)
    end_i = np.empty(S, dtype=np.int64); sta_i = np.empty(S, dtype=np.int64)
    for s in seg_ids:
        si = remap[int(s)]
        gi = np.where(seg == s)[0]
        o = np.argsort(frame[gi], kind="stable")
        a, b = gi[o[0]], gi[o[-1]]
        sta_i[si], end_i[si] = a, b
        sta_f[si], sta_x[si], sta_y[si] = frame[a], x[a], y[a]
        end_f[si], end_x[si], end_y[si] = frame[b], x[b], y[b]
    end_feat = feats[end_i] if feats is not None else None
    sta_feat = feats[sta_i] if feats is not None else None
    return (seg_ids, remap, sta_f, sta_x, sta_y, sta_i, sta_feat,
            end_f, end_x, end_y, end_i, end_feat)


def _gap_close(seg: np.ndarray, frame: np.ndarray, x: np.ndarray, y: np.ndarray,
               search_range: float, max_gap: int, feats=None,
               penalty_weight: float = 1.0) -> np.ndarray:
    """Globally link segment ends → later segment starts across gaps.
    Returns a new label per point (segments merged transitively).  With
    ``feats`` the gap-close cost gains the ``(D·P)²`` feature penalty."""
    seg_ids = np.unique(seg)
    S = len(seg_ids)
    if S <= 1 or max_gap <= 0:
        return seg
    (seg_ids, remap, sta_f, sta_x, sta_y, _sta_i, sta_feat,
     end_f, end_x, end_y, _end_i, end_feat) = _segment_ends(
        seg, frame, x, y, feats)

    sr2 = float(search_range) ** 2

    # ── Memory guard for large segment counts ─────────────────────────────────
    # The dense cost matrix is S×S and `_solve_birth_death` augments it to
    # (2S)×(2S); for tens of thousands of segments (e.g. dense over-detection)
    # that is tens of GB and OOMs.  A segment with NO candidate end→start link
    # inside the (search_range, max_gap) gate can only ever be a birth/death in
    # the full solve, so restricting the assignment to the segments that DO have
    # a candidate yields an IDENTICAL union-find while bounding the matrix to
    # |active|².  Below `_DENSE_S` we keep the original full-S dense path
    # verbatim (active = all segments), so the common case is byte-identical.
    _DENSE_S, _ACTIVE_CAP = 3000, 5000
    if S <= _DENSE_S:
        active = np.arange(S)
    else:
        # cKDTree candidate ends→starts within the spatial gate (d ≤ sr); the
        # temporal gate (gap ∈ [1, max_gap]) is applied per neighbour.
        tree = cKDTree(np.column_stack([sta_x, sta_y]))
        neigh = tree.query_ball_point(np.column_stack([end_x, end_y]),
                                      r=float(search_range))
        touched = np.zeros(S, dtype=bool)
        for a, starts in enumerate(neigh):
            for b in starts:
                g = sta_f[b] - end_f[a]
                if 1 <= g <= max_gap:
                    touched[a] = True
                    touched[b] = True
        active = np.where(touched)[0]
        if active.size == 0:
            return seg                                # no closable gaps
        if active.size > _ACTIVE_CAP:
            print(f"  WARN: gap-closing limited — {active.size} of {S} segments "
                  f"have candidate links, exceeding the dense-matrix safety cap "
                  f"({_ACTIVE_CAP}); some gaps left unclosed to bound memory.")
            return seg

    # cost(end a -> start b): gated by the gap window and a FIXED search radius
    # (TrackMate-style; intentionally does NOT grow with the gap — see below);
    # cost = squared distance.  Arrays are restricted to `active` segments.
    ax, ay, af = end_x[active], end_y[active], end_f[active]
    bx, by, bf = sta_x[active], sta_y[active], sta_f[active]
    afeat = end_feat[active] if end_feat is not None else None
    bfeat = sta_feat[active] if sta_feat is not None else None
    gap = bf[None, :] - af[:, None]                  # (active end a, active start b)
    d2 = ((ax[:, None] - bx[None, :]) ** 2 +
          (ay[:, None] - by[None, :]) ** 2)
    # FIXED spatial gate, like trackpy's `memory`: the re-link radius must NOT
    # blow up with the gap.  The per-frame diffusion step is << search_range, so
    # a search_range disc already covers many-frame reconnections; a radius that
    # grows with the gap just over-connects (splices wrong segments) under
    # density — the exact failure the benchmark caught.
    gate2 = sr2
    ok = (gap >= 1) & (gap <= max_gap) & (d2 <= gate2)
    if feats is not None:
        P = _feature_penalty(afeat, bfeat, penalty_weight)
        cost = d2 * (P * P) if P is not None else d2
        link = np.where(ok, cost, _BIG)
        alt = _alt_cost(link, sr2)
    else:
        link = np.where(ok, d2, _BIG)
        alt = sr2                                    # only close within search_range
    assign = _solve_birth_death(link, alt=alt)

    # union-find over segment indices for the chosen end→start links.  `assign`
    # is indexed in `active`-local positions, so map back to segment indices.
    parent = list(range(S))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a_local, b_local in enumerate(assign):
        if b_local >= 0:
            parent[find(int(active[a_local]))] = find(int(active[b_local]))

    root_of_segidx = np.array([find(i) for i in range(S)])
    out = np.empty_like(seg)
    inv = {int(s): i for i, s in enumerate(seg_ids)}
    for p in range(len(seg)):
        out[p] = root_of_segidx[inv[int(seg[p])]]
    return out


def _gap_close_full(seg, frame, x, y, search_range, max_gap, feats=None,
                    penalty_weight=1.0, allow_merging=False,
                    allow_splitting=False, cost_factor=1.0, df=None):
    """TrackMate full-LAP step 2: gap-closing PLUS optional MERGE (segment
    END → an ongoing segment) and SPLIT (an ongoing segment → segment START),
    solved as ONE global assignment.

    Composite cost matrix over ``dim = 2S`` (S = number of segments):
        rows  0:S  = segment ENDS        (gap-close / merge sources)
        rows  S:2S = segment SPLIT slots (ongoing-at-(start-1) split parents)
        cols  0:S  = segment STARTS      (gap-close / split targets)
        cols  S:2S = segment MERGE slots (ongoing-at-(end+1) merge parents)
    Blocks: ends→starts = gap-close (gap∈[1,max_gap], d≤sr); ends→merge-slots =
    merge (Δframe=1, d≤sr); split-slots→starts = split (Δframe=1, d≤sr); all
    others blocked.  Costs are ``(D·P)²`` (P = feature penalty when ``feats``);
    merge/split blocks are scaled by ``cost_factor``.  Birth/death come from
    `_solve_birth_death`'s augmentation.

    Merge/split make a branching FOREST, which a flat ``particle`` column can't
    represent, so segments joined by ANY event are unioned into one particle id;
    the detected events are recorded in ``df.attrs['lap_events']``."""
    (seg_ids, remap, sta_f, sta_x, sta_y, sta_i, sta_feat,
     end_f, end_x, end_y, end_i, end_feat) = _segment_ends(seg, frame, x, y, feats)
    S = len(seg_ids)
    if S <= 1:
        return seg
    # The dense step-2 matrix is ~(2S)² (and `_solve_birth_death` doubles it
    # again).  Fall back to gap-closing only on very large problems to bound
    # memory — merge/split is an opt-in refinement, not the high-value step.
    if S > 1500:
        print(f"  WARN: full-LAP merge/split disabled — {S} segments exceeds the "
              f"dense-matrix safety cap; using gap-closing only.")
        return _gap_close(seg, frame, x, y, search_range, max_gap, feats,
                          penalty_weight)

    sr2 = float(search_range) ** 2
    inv = {int(s): i for i, s in enumerate(seg_ids)}
    seg_of = np.array([inv[int(s)] for s in seg], dtype=np.int64)
    # per-segment frame -> global point index (for ongoing-point lookups)
    pts_by_seg = [dict() for _ in range(S)]
    for p in range(len(seg)):
        pts_by_seg[seg_of[p]][int(frame[p])] = p

    dim = 2 * S
    C = np.full((dim, dim), _BIG, dtype=float)

    # GAP-CLOSE block: ends (rows 0:S) -> starts (cols 0:S)
    gap = sta_f[None, :] - end_f[:, None]
    d2 = ((end_x[:, None] - sta_x[None, :]) ** 2 +
          (end_y[:, None] - sta_y[None, :]) ** 2)
    ok = (gap >= 1) & (gap <= max_gap) & (d2 <= sr2)
    if feats is not None:
        P = _feature_penalty(end_feat, sta_feat, penalty_weight)
        gc = d2 * (P * P) if P is not None else d2
    else:
        gc = d2
    block = np.where(ok, gc, _BIG)
    np.fill_diagonal(block, _BIG)              # no self gap-close
    C[:S, :S] = block

    def _pair_cost(gi_a, p_b):
        dd = (x[gi_a] - x[p_b]) ** 2 + (y[gi_a] - y[p_b]) ** 2
        if dd > sr2:
            return None
        if feats is not None:
            Pp = _feature_penalty(feats[gi_a][None, :], feats[p_b][None, :],
                                  penalty_weight)
            if Pp is not None:
                dd = dd * float(Pp[0, 0]) ** 2
        return dd * cost_factor

    # MERGE block: end a -> segment b ongoing at frame(end_a)+1 (col S+b)
    if allow_merging:
        for ai in range(S):
            ft = int(end_f[ai]) + 1
            for bi in range(S):
                if bi == ai:
                    continue
                p = pts_by_seg[bi].get(ft)
                if p is None or ft <= int(sta_f[bi]):   # must be ONGOING (not start)
                    continue
                c = _pair_cost(int(end_i[ai]), p)
                if c is not None:
                    C[ai, S + bi] = min(C[ai, S + bi], c)

    # SPLIT block: segment a ongoing at frame(start_b)-1 (row S+a) -> start b
    if allow_splitting:
        for bi in range(S):
            fp = int(sta_f[bi]) - 1
            for ai in range(S):
                if ai == bi:
                    continue
                p = pts_by_seg[ai].get(fp)
                if p is None or fp >= int(end_f[ai]):    # must be ONGOING (not end)
                    continue
                c = _pair_cost(int(sta_i[bi]), p)
                if c is not None:
                    C[S + ai, bi] = min(C[S + ai, bi], c)

    finite = C[C < _BIG]
    alt = float(1.05 * finite.max()) if finite.size else sr2
    assign = _solve_birth_death(C, alt=alt)

    parent = list(range(S))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    events = []
    for r, cc in enumerate(assign):
        if cc < 0:
            continue
        if r < S and cc < S:
            a, b, kind = r, cc, "gap_close"
        elif r < S and cc >= S:
            a, b, kind = r, cc - S, "merge"
        elif r >= S and cc < S:
            a, b, kind = r - S, cc, "split"
        else:
            continue
        if a == b:
            continue
        parent[find(a)] = find(b)
        events.append((kind, int(seg_ids[a]), int(seg_ids[b])))

    root = np.array([find(i) for i in range(S)])
    out = np.array([root[seg_of[p]] for p in range(len(seg))], dtype=np.int64)
    if df is not None:
        df.attrs["lap_events"] = events
    return out


def link_trajectories_lap(locs: pd.DataFrame, search_range: float = 5.0,
                          max_gap: int = 12, min_len: int = 2,
                          allow_merging: bool = False,
                          allow_splitting: bool = False,
                          feature_penalty: bool = False,
                          feature_cols=("mass",), penalty_weight: float = 1.0,
                          merge_split_cost_factor: float = 1.0) -> pd.DataFrame:
    """Two-step LAP linker.  `max_gap` is the gap-closing horizon (the LAP
    analogue of trackpy's `memory`).  Returns `locs` with an integer `particle`
    column; tracks shorter than `min_len` points are dropped.

    With every flag at its default this is TrackMate's "Simple LAP" and is
    numerically identical to the original implementation.  Optional TrackMate
    full-LAP extras:
      • ``feature_penalty`` — link cost becomes ``(D·P)²`` with an intensity/
        quality penalty over ``feature_cols`` (weight ``penalty_weight``);
      • ``allow_merging`` / ``allow_splitting`` — the step-2 assignment also
        considers segment END→MIDDLE (merge) and MIDDLE→START (split) events
        (``merge_split_cost_factor`` scales those blocks).  Merge/split make a
        branching forest, which a flat ``particle`` column can't fully express,
        so the involved segments are unioned into one particle id and the event
        list is attached as ``df.attrs['lap_events']``.  OFF by default — single
        fluorophores don't physically coalesce, so for sptPALM the high-value
        step is gap-closing.
    """
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
    feats = _stack_features(df, feature_cols) if feature_penalty else None

    seg = _frame_to_frame(frame, x, y, search_range, feats=feats,
                          penalty_weight=penalty_weight)
    if allow_merging or allow_splitting:
        lab = _gap_close_full(
            seg, frame, x, y, search_range, max_gap, feats=feats,
            penalty_weight=penalty_weight, allow_merging=allow_merging,
            allow_splitting=allow_splitting, cost_factor=merge_split_cost_factor,
            df=df)
    else:
        lab = _gap_close(seg, frame, x, y, search_range, max_gap, feats=feats,
                         penalty_weight=penalty_weight)

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


def _F_dt(dt):
    """Constant-velocity state transition propagated over `dt` frames.

    Reduces to `_F` at dt=1, so contiguous (gapless) tracking is unchanged; a
    multi-frame coast across empty frames advances the predicted position by the
    true number of frames rather than a single step.
    """
    return np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                     [0, 0, 1, 0], [0, 0, 0, 1]], float)


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

    # each track: dict(state(4,), P(4,4), cf:int, lf:int, pts:list[int])
    tracks: list[dict] = []
    active: list[dict] = []

    def _birth(j_global, f):
        st = np.array([x[j_global], y[j_global], 0.0, 0.0])
        P = np.diag([float(r_meas), float(r_meas), p0_vel, p0_vel])
        # cf = frame the state currently represents (drives the predict step);
        # lf = last MATCHED frame (drives the coast gate).  Both are tracked by
        # true frame NUMBER, not the loop-iteration count, so empty frames
        # (absent from `uframes`) cannot silently bridge an arbitrary gap.
        t = {"state": st, "P": P, "cf": int(f), "lf": int(f),
             "pts": [int(j_global)]}
        tracks.append(t); active.append(t)

    for f in uframes:
        f = int(f)
        # Expire tracks whose last MATCH is now more than max_gap frames back,
        # BEFORE they predict-and-match here — a constant-velocity prediction can
        # otherwise leap an arbitrarily large empty-frame gap and match anyway.
        if active:
            active[:] = [t for t in active if (f - t["lf"]) <= max_gap]
        det = idx_by_frame[f]
        if active:
            preds = np.empty((len(active), 2))
            for k, t in enumerate(active):
                dt = f - t["cf"]               # true frames since last propagation
                if dt > 0:
                    F = _F_dt(dt)
                    t["state"] = F @ t["state"]
                    t["P"] = F @ t["P"] @ F.T + dt * Q   # noise accrues over the gap
                    t["cf"] = f
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
                    t["lf"] = f; t["pts"].append(int(det[j])); used[j] = True
                    still.append(t)
                else:
                    still.append(t)            # coast (pruned next frame if stale)
            active[:] = still
            for j in np.where(~used)[0]:
                _birth(int(det[j]), f)
        else:
            for j in det:
                _birth(int(j), f)

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


# Simple LAP == the two-step LAP above (TrackMate "Simple LAP": frame-to-frame +
# gap-closing, no merge/split, no feature penalties).  Exposed under its own name
# so the registry/UI can label it; the legacy token "lap" maps here too.
link_trajectories_simple_lap = link_trajectories_lap


# ── Nearest-neighbour linker (TrackMate "Nearest-neighbour") ─────────────────
def link_trajectories_nn(locs: pd.DataFrame, search_range: float = 5.0,
                         max_gap: int = 3, min_len: int = 2) -> pd.DataFrame:
    """Greedy nearest-neighbour linker.

    Per frame: for every ACTIVE track find candidate detections within
    `search_range` of its last known position (cKDTree), then assign GREEDILY in
    ascending distance — first-come, skipping already-used tracks/detections.
    Unmatched detections start new tracks; unmatched tracks COAST (keep their
    last position) for up to `max_gap` frames, then terminate.  Unlike the LAP
    this is local/greedy (no global optimum) — it is TrackMate's simplest
    tracker and a useful fast baseline.  Deterministic: candidate pairs are
    sorted by (distance, detection index, track index), so the result never
    depends on dict/iteration order.  `max_gap` is the trackpy-`memory`
    analogue; the registry's `nn` adapter pins it to 1 (strictly frame-to-frame,
    canonical TrackMate NN — no gap bridging), but the function accepts any
    `max_gap` for callers that want coasting.  Returns `locs` with an integer
    `particle` column; tracks shorter than `min_len` points are dropped."""
    if locs is None or len(locs) == 0:
        cols = list(locs.columns) if locs is not None else ["x", "y", "frame"]
        if "particle" not in cols:
            cols = cols + ["particle"]
        return pd.DataFrame(columns=cols)
    for c in ("x", "y", "frame"):
        if c not in locs.columns:
            raise ValueError(f"link_trajectories_nn: missing column '{c}'")
    df = locs.reset_index(drop=True).copy()
    frame = df["frame"].to_numpy(np.int64)
    x = df["x"].to_numpy(float); y = df["y"].to_numpy(float)
    uframes = np.unique(frame)
    idx_by_frame = {int(f): np.where(frame == f)[0] for f in uframes}
    sr = float(search_range)

    tracks: list[dict] = []      # every track ever started (for final labelling)
    active: list[dict] = []      # tracks still eligible to extend

    def _birth(j_global, f):
        t = {"lx": float(x[j_global]), "ly": float(y[j_global]),
             "lf": int(f), "pts": [int(j_global)]}      # lf = last-matched frame
        tracks.append(t); active.append(t)

    for f in uframes:
        f = int(f)
        det = idx_by_frame[f]
        n_det = len(det)
        # Expire tracks whose gap to NOW exceeds max_gap.  Gaps are measured by
        # actual frame NUMBER (not iteration count), so a multi-frame gap with no
        # detections anywhere is still counted — matching the LAP gap semantics.
        # Frames only increase, so an expired track can never match again.
        active = [t for t in active if (f - t["lf"]) <= max_gap]
        used_det = np.zeros(n_det, dtype=bool)
        if len(active) and n_det:
            dx = x[det]; dy = y[det]
            tree = cKDTree(np.column_stack([dx, dy]))
            tlast = np.array([[t["lx"], t["ly"]] for t in active], dtype=float)
            nbrs = tree.query_ball_point(tlast, r=sr)       # per-track det indices
            pairs = []                                      # (dist², det idx, trk idx)
            for k, js in enumerate(nbrs):
                for j in js:
                    d2 = (tlast[k, 0] - dx[j]) ** 2 + (tlast[k, 1] - dy[j]) ** 2
                    pairs.append((float(d2), int(j), k))
            pairs.sort()                                     # deterministic greedy order
            matched_trk = np.zeros(len(active), dtype=bool)
            for _d2, j, k in pairs:
                if matched_trk[k] or used_det[j]:
                    continue
                t = active[k]
                t["lx"] = float(dx[j]); t["ly"] = float(dy[j])
                t["lf"] = f; t["pts"].append(int(det[j]))
                matched_trk[k] = True; used_det[j] = True
        # Unmatched-but-young tracks stay in `active` (last position kept) and may
        # match a later frame; they're dropped by the expiry check above once too
        # old.  New detections start tracks.
        for j in range(n_det):
            if not used_det[j]:
                _birth(int(det[j]), f)

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
