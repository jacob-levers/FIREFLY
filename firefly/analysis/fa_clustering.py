"""DBSCAN clustering of localisations.

Extracted from sptpalm_analysis.py (#7); re-exported there for compatibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def compute_clusters(locs, pixel_size_um, eps_um=0.05, min_samples=5,
                     max_locs=250_000):
    from sklearn.cluster import DBSCAN
    from scipy.spatial import ConvexHull
    xy = locs[["x", "y"]].values * pixel_size_um
    n_input = len(xy)
    subsampled = n_input > max_locs
    if subsampled:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_input, max_locs, replace=False)
        xy = xy[idx]
        print(f"  Cluster analysis  : sub-sampled to {max_locs:,} of "
              f"{n_input:,} localisations")
    else:
        print(f"  Cluster analysis  : {len(xy):,} localisations  "
              f"(eps={eps_um*1000:.0f} nm, min_samples={min_samples})")

    # ── Safety guard: refuse an eps so large the neighbourhood graph explodes ──
    # DBSCAN's region queries return ~all points when eps approaches the data's
    # spatial extent, so memory grows as O(n × avg_neighbours) and can crash the
    # process (an over-large "Suggest eps" used to do exactly this).  Estimate
    # the average neighbourhood size cheaply (count-only on a small sample, so
    # the guard itself can't blow up) and skip DBSCAN if it would be enormous —
    # returning all-noise + an `eps_too_large` flag instead of crashing.
    eps_too_large = False
    avg_nbr = None
    try:
        from sklearn.neighbors import KDTree
        _tree = KDTree(xy)
        _rng = np.random.default_rng(1)
        _samp = (xy[_rng.choice(len(xy), 1000, replace=False)]
                 if len(xy) > 1000 else xy)
        avg_nbr = float(np.mean(
            _tree.query_radius(_samp, r=eps_um, count_only=True)))
        # ~25M total neighbour entries ≈ a few hundred MB — comfortably above a
        # real clustering (tens of neighbours/point) but well below a blow-up.
        eps_too_large = (avg_nbr * len(xy)) > 25_000_000
    except Exception:
        eps_too_large = False

    if eps_too_large:
        print(f"  Cluster analysis  : eps={eps_um*1000:.0f} nm is too large for "
              f"this data (~{avg_nbr:.0f} neighbours/point) — skipped to avoid a "
              f"memory blow-up; lower eps.")
        labels = np.full(len(xy), -1, dtype=int)
        df = pd.DataFrame(columns=["cluster_id", "n_locs", "area_um2",
                                   "density_locs_per_um2", "rg_um",
                                   "centroid_x_um", "centroid_y_um"])
        df.attrs["n_input_locs"] = int(n_input)
        df.attrs["n_used_locs"] = int(len(xy))
        df.attrs["subsampled"] = bool(subsampled)
        df.attrs["eps_too_large"] = True
        return labels, df, 0, xy

    labels = DBSCAN(eps=eps_um, min_samples=min_samples).fit_predict(xy)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    rows = []
    for c in sorted(set(labels)):
        if c == -1:
            continue
        pts = xy[labels == c]
        n = len(pts)
        try:
            area = ConvexHull(pts).volume if n >= 3 else np.nan
        except Exception:
            area = np.nan
        density = n / area if (area and area > 0) else np.nan
        centroid = pts.mean(axis=0)
        # Radius of gyration — RMS distance of a cluster's points from its
        # centroid.  Always defined (unlike the ConvexHull area, which is NaN
        # for <3 or collinear points), so it's a robust cluster-size measure.
        rg = (float(np.sqrt(np.mean(np.sum((pts - centroid) ** 2, axis=1))))
              if n else np.nan)
        rows.append({"cluster_id": int(c), "n_locs": int(n),
                     "area_um2": area, "density_locs_per_um2": density,
                     "rg_um": rg,
                     "centroid_x_um": centroid[0],
                     "centroid_y_um": centroid[1]})
    df = pd.DataFrame(rows)
    # In-memory subsample provenance (not written to CSV) so callers can
    # surface "sub-sampled to N" honestly in the log / figure / Results panel.
    df.attrs["n_input_locs"] = int(n_input)
    df.attrs["n_used_locs"] = int(len(xy))
    df.attrs["subsampled"] = bool(subsampled)
    df.attrs["eps_too_large"] = False
    return labels, df, int(n_clusters), xy
