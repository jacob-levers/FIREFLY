"""Auto-estimate the linking ``search_range`` — the linking analogue of
``estimate_minmass`` for detection.

The single biggest lever on linker accuracy (no algorithm change) is matching the
search range to the motion: too small breaks real tracks, too large invites
spurious cross-links (and can blow up trackpy's subnet solver).  Tracking quality
PLATEAUS once the range is just large enough to complete the real links — so this
sweeps candidate ranges on a frame window, links each with the fast
nearest-neighbour linker, and picks the PLATEAU ONSET: the smallest range at which
the mean reconstructed track length stops growing (tracks stop fragmenting),
capped well below the inter-molecule spacing so cross-links never enter the
picture.  Validated against an oracle GT sweep across diffusion / directed /
dense regimes.

No ground truth is needed, and the estimate is LINKER-AGNOSTIC: the optimal
search range is a property of the MOTION + density, so it is estimated once with
the cheap NN linker and then applied to whichever linker the run actually uses
(including the slow SA tracker).  ``memory`` (gap horizon) is left to the caller
for now — it is governed by blinking statistics, a separate estimation.

Qt-free: numpy + scipy + pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def _median_nn_spacing(frame, x, y):
    """Median nearest-neighbour distance WITHIN frames — the inter-molecule
    spacing.  A link longer than ~half this is almost surely a cross-link, so it
    caps the search-range grid (and keeps trackpy's subnet solver out of its
    combinatorial-explosion regime)."""
    ds = []
    for f in np.unique(frame):
        pts = np.column_stack([x[frame == f], y[frame == f]])
        if len(pts) < 2:
            continue
        d, _ = cKDTree(pts).query(pts, k=2)
        ds.append(d[:, 1])
    return float(np.median(np.concatenate(ds))) if ds else float("inf")


def estimate_link_params(locs, *, memory=3, min_len=4, frame_sample=300,
                         search_grid=None, sensitivity="balanced",
                         log_cb=None):
    """Estimate a good ``search_range`` (px) for linking ``locs``.

    Returns ``(search_range, diag)``.  ``sensitivity`` shifts the operating point
    along the captured-motion curve: ``strict`` hugs the saturation knee (fewest
    cross-links), ``lenient`` allows more headroom for fast/heterogeneous motion.
    """
    def _log(m):
        if log_cb:
            log_cb(m)

    if locs is None or len(locs) == 0:
        return 5.0, {"method": "empty_default"}
    for c in ("x", "y", "frame"):
        if c not in locs.columns:
            raise ValueError(f"estimate_link_params: missing column '{c}'")

    df = locs.reset_index(drop=True)
    frame = df["frame"].to_numpy(np.int64)
    # A contiguous window keeps the sweep cheap while preserving the per-frame
    # motion the estimate depends on.
    f0 = int(frame.min())
    win = frame < f0 + int(frame_sample)
    sub = df[win][["x", "y", "frame"]].reset_index(drop=True)
    fr = sub["frame"].to_numpy(np.int64)
    xx = sub["x"].to_numpy(float); yy = sub["y"].to_numpy(float)
    if len(sub) < 4 * max(1, min_len):
        return 5.0, {"method": "too_few_locs", "n": int(len(sub))}

    # Cap the grid well below the inter-molecule spacing: a link longer than this
    # is almost surely a cross-link, and it keeps trackpy's subnet solver out of
    # its combinatorial-explosion regime.
    spacing = _median_nn_spacing(fr, xx, yy)
    cap = float(np.clip(0.7 * spacing if np.isfinite(spacing) else 20.0, 3.0, 30.0))
    if search_grid is None:
        search_grid = np.unique(np.round(np.geomspace(1.0, cap, 12), 2))
    search_grid = np.asarray(sorted(float(s) for s in search_grid), dtype=float)

    from firefly.analysis.fa_linking_lap import link_trajectories_nn
    _log(f"  Auto search-range: sweeping {len(search_grid)} ranges "
         f"(≤{cap:.1f}px, inter-spot spacing {spacing:.1f}px) on "
         f"{len(sub):,} locs…")
    mean_len = np.zeros(len(search_grid))
    n_good = np.zeros(len(search_grid), dtype=int)
    for i, sr in enumerate(search_grid):
        t = link_trajectories_nn(sub, search_range=sr, max_gap=int(memory),
                                 min_len=int(min_len))
        if len(t) == 0:
            continue
        ng = int(t["particle"].nunique())
        n_good[i] = ng
        mean_len[i] = (len(t) / ng) if ng else 0.0

    # Plateau onset: the smallest range at which the mean track length has
    # essentially reached its MAXIMUM — i.e. real tracks have stopped fragmenting
    # and all real links are made.  Anchoring on the max (only reached AFTER the
    # rise) is what makes this robust: a simple "growth stalled" test would fire
    # in the flat-LOW region BEFORE linking starts (sr ≪ motion → almost nothing
    # links), picking far too small a range on fast motion.  Cross-links would
    # eventually inflate mean length further, but `cap` (< inter-spot spacing)
    # keeps the sweep out of that regime.
    ml_max = float(np.max(mean_len)) if np.any(mean_len > 0) else 0.0
    if ml_max <= 1e-6:
        return 5.0, {"method": "no_links", "spacing_px": float(spacing)}
    target = 0.95 * ml_max
    onset_i = int(np.argmax(mean_len >= target))      # first range reaching 95 %
    margin = {"strict": 1.0, "balanced": 1.1, "lenient": 1.3}.get(
        str(sensitivity).lower(), 1.1)
    onset_sr = float(search_grid[onset_i])
    chosen = float(np.clip(onset_sr * margin, search_grid[0], cap))

    diag = {
        "method": "meanlen_plateau",
        "search_range": chosen,
        "onset_search_range": onset_sr,
        "spacing_px": float(spacing),
        "cap_px": cap,
        "grid": [float(s) for s in search_grid],
        "mean_len": [float(v) for v in mean_len],
        "n_good": [int(v) for v in n_good],
        "sensitivity": str(sensitivity).lower(),
    }
    _log(f"  Auto search-range: tracks stop fragmenting at ~{onset_sr:.1f}px → "
         f"search_range = {chosen:.1f}px ({sensitivity}).")
    return chosen, diag
