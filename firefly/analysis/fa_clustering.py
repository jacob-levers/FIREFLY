"""DBSCAN clustering of localisations.

Extracted from sptpalm_analysis.py (#7); re-exported there for compatibility.

Caveats on the per-cluster statistics (the worker logs these per run):
  * SUBSAMPLING: DBSCAN is density-based, so above `max_locs` (250k) the
    localisations are randomly sub-sampled.  Lowering the point density lowers
    realised neighbour counts, so `n_locs`, `area_um2` and especially
    `density_locs_per_um2` are NOT comparable between a sub-sampled run and a
    full one — `df.attrs["subsampled"]` flags when this happened.
  * HULL-DENSITY BIAS: `area_um2` is the convex-hull area, which grows with the
    number of points even at constant true density, so `density = n / hull_area`
    is a hull-fill density, not an unbiased spatial density, and is not directly
    comparable across clusters of very different size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def suggest_eps_nm(xy_um, min_samples=8, max_locs=50_000):
    """k-distance knee estimate of DBSCAN's eps, in nm, or None.

    Sort every point's distance to its k-th nearest neighbour (k = min_samples)
    and take the point of maximum deviation from the chord joining the ends of
    that curve — the classic DBSCAN elbow.  Trim the outer 2% first so a handful
    of isolated localisations cannot drag the chord.

    A fixed eps cannot suit every recording: a dense field and a sparse one need
    different neighbourhood radii, and using one number for both silently turns
    the sparse recording into noise.  Shared by the Visualise tab's "Suggest
    eps" button and the worker's automatic mode so the two cannot disagree.
    """
    xy = np.asarray(xy_um, dtype=float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 3:
        return None
    if len(xy) > max_locs:                 # the knee is a shape, not a census
        rng = np.random.default_rng(0)
        xy = xy[rng.choice(len(xy), max_locs, replace=False)]
    try:
        from sklearn.neighbors import NearestNeighbors
        k = max(2, min(int(min_samples), len(xy) - 1))
        dist, _ = NearestNeighbors(n_neighbors=k).fit(xy).kneighbors(xy)
        kd = np.sort(dist[:, -1])
        m = len(kd)
        seg = kd[max(0, int(0.02 * m)): min(m - 1, int(0.98 * m)) + 1]
        mm = len(seg)
        if mm < 3:
            eps_nm = float(np.median(kd)) * 1000.0
        else:
            x = np.arange(mm, dtype=float)
            y0, y1 = seg[0], seg[-1]
            num = np.abs((y1 - y0) * x - (mm - 1) * seg + (mm - 1) * y0)
            den = np.hypot(y1 - y0, mm - 1) or 1.0
            eps_nm = float(seg[int(np.argmax(num / den))]) * 1000.0
    except Exception:
        return None
    if not np.isfinite(eps_nm) or eps_nm <= 0:
        return None
    return float(np.clip(eps_nm, 5.0, 2000.0))


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
              f"(EPS={eps_um*1000:.0f} nm, min_samples={min_samples})")

    # ── Safety guard: bound the neighbourhood graph for a large eps ───────────
    # DBSCAN's region queries return ~all points when eps approaches the data's
    # spatial extent, so memory grows as O(n × avg_neighbours) and can crash the
    # process.  Estimate the average neighbourhood size cheaply (count-only on a
    # small sample, so the guard itself can't blow up); if it would be enormous,
    # SUB-SAMPLE down to a size where DBSCAN at this eps stays bounded — so
    # large-eps exploration still returns a (changing) result instead of being
    # refused.  Sub-sampling lowers the realised neighbour count too, so
    # n_safe = LIMIT / avg_nbr is a conservative cap.
    _LIMIT = 25_000_000          # ~ neighbour entries ≈ a few hundred MB
    try:
        from sklearn.neighbors import KDTree
        _tree = KDTree(xy)
        _rng = np.random.default_rng(1)
        _samp = (xy[_rng.choice(len(xy), 1000, replace=False)]
                 if len(xy) > 1000 else xy)
        avg_nbr = float(np.mean(
            _tree.query_radius(_samp, r=eps_um, count_only=True)))
    except Exception:
        avg_nbr = 0.0
    if avg_nbr * len(xy) > _LIMIT:
        n_safe = max(2000, int(_LIMIT / max(avg_nbr, 1.0)))
        if n_safe < len(xy):
            _r2 = np.random.default_rng(7)
            xy = xy[_r2.choice(len(xy), n_safe, replace=False)]
            subsampled = True
            print(f"  Cluster analysis  : EPS={eps_um*1000:.0f} nm large for this "
                  f"data — sub-sampled to {n_safe:,} localisations to stay within "
                  f"memory")

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
