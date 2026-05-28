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
    subsampled = len(xy) > max_locs
    if subsampled:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(xy), max_locs, replace=False)
        xy = xy[idx]
        print(f"  Cluster analysis  : sub-sampled to {max_locs:,} localisations")
    else:
        print(f"  Cluster analysis  : {len(xy):,} localisations  "
              f"(eps={eps_um*1000:.0f} nm, min_samples={min_samples})")
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
        rows.append({"cluster_id": int(c), "n_locs": int(n),
                     "area_um2": area, "density_locs_per_um2": density,
                     "centroid_x_um": pts[:,0].mean(),
                     "centroid_y_um": pts[:,1].mean()})
    return labels, pd.DataFrame(rows), int(n_clusters), xy
