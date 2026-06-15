"""Trajectory linking (trackpy recursive subnet linker).

Extracted from sptpalm_analysis.py (#7); re-exported there for compatibility.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import trackpy as tp
from firefly.analysis.fa_constants import _Cancelled, _tqdm


def link_trajectories(locs, search_range=5, memory=3, min_len=5, max_len=None,
                       linker="trackpy", link_params=None,
                       progress_cb=None, stop_event=None):
    """Link localisations into trajectories.

    linker      : one of (see `firefly.analysis.fa_enums.Linker`):
        "trackpy" (default) — Crocker-Grier recursive-subnet nearest-neighbour,
        best for pure Brownian diffusion; "kalman" — constant-velocity Kalman
        LAP (TrackMate "Linear Motion"), better on directed / crossing motion;
        "simple_lap" (alias "lap") — Jaqaman two-step LAP (frame-to-frame +
        gap-closing); "full_lap" — Simple LAP + optional merge/split + feature
        penalties (TrackMate's full LAP); "nn" — greedy nearest-neighbour;
        "sa" — simulated-annealing multi-target tracker (palmTRACER-style).
        Dispatch is via the linker registry (`fa_linking_registry`); every
        linker produces a `particle` column on the returned DataFrame.
    link_params : dict | None
        Per-linker knobs (e.g. full_lap's allow_merging/allow_splitting/
        feature_penalty; sa's seed/cooling/…).  Ignored by linkers that don't
        use them.
    progress_cb : callable(fraction) → None
        Optional.  Called periodically with a [0, 1] float so the host
        can update a progress bar.  Updates are throttled to roughly
        once every 32 frames + once on completion.
    stop_event  : threading.Event-like
        Optional.  Polled between frames; if `.is_set()` the linker
        raises `_Cancelled` and aborts cleanly.
    """
    # Empty input → trackpy's coords_from_df indexes unique_times[0] on a
    # size-0 array and raises a cryptic IndexError.  Bail out cleanly with a
    # clear message instead (e.g. when an ROI mask excluded the whole frame
    # or minmass filtered out every spot).
    if locs is None or len(locs) == 0:
        print("  No localisations to link (0 spots) — returning 0 "
              "trajectories.  If ROI masking is enabled it may have excluded "
              "the whole frame; otherwise try a lower minmass.")
        cols = list(locs.columns) if locs is not None else \
            ["x", "y", "frame", "mass"]
        if "particle" not in cols:
            cols = cols + ["particle"]
        return pd.DataFrame(columns=cols)

    # Validate the required schema up front so a malformed input yields a clear
    # message instead of a cryptic trackpy KeyError/IndexError deep in linking.
    _missing = [c for c in ("x", "y", "frame") if c not in locs.columns]
    if _missing:
        raise ValueError(
            f"link_trajectories: localisations are missing required "
            f"column(s) {_missing}; got {list(locs.columns)}.")
    # Negative frames break trackpy's frame indexing — drop them with a clear
    # warning (mirrors the external-loader behaviour) rather than crashing.
    if (locs["frame"] < 0).any():
        _n_bad = int((locs["frame"] < 0).sum())
        print(f"  WARN: dropping {_n_bad:,} localisation(s) with frame < 0 "
              f"before linking.")
        locs = locs[locs["frame"] >= 0].reset_index(drop=True)
        if len(locs) == 0:
            _cols = list(locs.columns)
            if "particle" not in _cols:
                _cols = _cols + ["particle"]
            return pd.DataFrame(columns=_cols)

    # Dispatch to the selected linker via the registry (mirrors the localiser
    # backend registry).  Each adapter wraps the existing linker function and
    # applies the min/max-length filters; trackpy stays the default.
    from firefly.analysis.fa_linking_registry import _resolve_linker
    backend = _resolve_linker(linker)
    print(f"  Linking {len(locs):,} localisations  "
          f"(linker={backend.name}, search_range={search_range}px, "
          f"memory/max_gap={memory}) ...")
    t0 = time.perf_counter()
    filtered = backend.link(
        locs, search_range=search_range, memory=memory, min_len=min_len,
        max_len=max_len, params=link_params or {},
        progress_cb=progress_cb, stop_event=stop_event)
    n = filtered["particle"].nunique() if len(filtered) else 0
    max_str = str(max_len) if max_len else "inf"
    print(f"  {n:,} trajectories (len {min_len}-{max_str}) in "
          f"{time.perf_counter() - t0:.1f}s")
    return filtered


def _link_via_trackpy(locs, *, search_range, memory,
                       progress_cb=None, stop_event=None):
    """Trackpy linker — extracted verbatim from the original
    `link_trajectories` body.  Uses `tp.link_iter` when available
    so the user can see progress and cancel mid-link; falls back
    to atomic `tp.link` on older trackpy versions or if the
    iterator path errors out.
    """
    iter_ok = hasattr(tp, "link_iter") and len(locs) > 0
    linked = None
    if iter_ok:
        # Per-frame coordinate iterator + index map so we can re-attach
        # particle IDs to the original locs DataFrame.
        try:
            frame_nums = sorted(int(f) for f in locs["frame"].unique())
            grouped = locs.groupby("frame")
            coords_per_frame: list = []
            indices_per_frame: list = []
            for f in frame_nums:
                sub = grouped.get_group(f)
                coords_per_frame.append(sub[["y", "x"]].to_numpy())
                indices_per_frame.append(sub.index.to_numpy())
            n_frames = len(frame_nums)

            particle_ids = np.full(len(locs), -1, dtype=np.int64)
            iterator = tp.link_iter(
                iter(coords_per_frame),
                search_range=search_range, memory=memory)
            for f_idx, raw in enumerate(iterator):
                row_idx = indices_per_frame[f_idx]
                # Across trackpy versions `link_iter` yields one of:
                #   • an int ndarray of particle IDs (the common case)
                #   • a 1-D pandas Series of IDs
                #   • a 2-tuple `(coords_array, ids_array)` (older versions)
                #   • a DataFrame with a 'particle' column (when fed DFs)
                # Normalise all of these to a plain int ndarray before
                # we try to slot it into `particle_ids` — without this,
                # numpy 2.x raises an "inhomogeneous shape" error on the
                # tuple form (the user's stack trace).
                p_ids = raw
                if isinstance(p_ids, tuple):
                    # (coords, ids) or (ids, coords) — pick the 1-D one.
                    a, b = p_ids
                    p_ids = a if np.ndim(a) == 1 else b
                elif hasattr(p_ids, "columns") and "particle" in getattr(
                        p_ids, "columns", []):
                    p_ids = p_ids["particle"].to_numpy()
                elif hasattr(p_ids, "to_numpy"):
                    p_ids = p_ids.to_numpy()
                arr = np.asarray(p_ids, dtype=np.int64).ravel()
                if arr.shape[0] != row_idx.shape[0]:
                    # Mismatch — trackpy's iter may have emitted in a
                    # different shape than we expected.  Bail to atomic.
                    raise RuntimeError(
                        f"link_iter shape mismatch (got {arr.shape[0]} "
                        f"ids for {row_idx.shape[0]} rows)")
                particle_ids[row_idx] = arr
                # Progress + cancel — only every 32 frames to keep cost
                # well under linking cost itself
                if (f_idx & 31) == 0:
                    if progress_cb is not None:
                        try:    progress_cb((f_idx + 1) / max(1, n_frames))
                        except Exception: pass
                    if stop_event is not None and stop_event.is_set():
                        raise _Cancelled()
            if progress_cb is not None:
                try:    progress_cb(1.0)
                except Exception: pass

            linked = locs.copy()
            linked["particle"] = particle_ids
            linked = linked[linked["particle"] >= 0].reset_index(drop=True)
            print(f"  tp.link_iter done — filtering stubs in caller ...")
        except _Cancelled:
            raise
        except Exception as exc:
            # Iter path didn't work — fall through to atomic tp.link
            print(f"  link_iter failed ({type(exc).__name__}: {exc}); "
                  f"falling back to atomic tp.link")
            linked = None

    if linked is None:
        # Atomic path — uninterruptible but works on any trackpy version
        if stop_event is not None and stop_event.is_set():
            raise _Cancelled()
        try:
            linked = tp.link(locs, search_range=search_range, memory=memory)
            print(f"  tp.link done — filtering stubs in caller ...")
        except Exception as exc:
            if ("SubnetOversizeException" in type(exc).__name__
                    or "Subnetwork" in str(exc)):
                print(f"  WARNING: SubnetOversizeException — switching to "
                      f"nonrecursive linker (consider reducing Search range)")
                linked = tp.link(locs, search_range=search_range,
                                  memory=memory,
                                  link_strategy="nonrecursive")
            else:
                raise
    return linked
