"""Linker registry — one plug-in entry per trajectory-linking algorithm.

Mirrors the localiser-backend pattern (`_BACKEND_REGISTRY` / `_resolve_backend`
in ``fa_localize_backends.py``): a small ``LinkerBackend`` adapter per algorithm,
looked up by name via :class:`firefly.analysis.fa_enums.Linker`.  Each adapter is
a THIN wrapper that calls the existing linker function unchanged — the linking
maths lives in ``fa_linking`` (trackpy), ``fa_linking_lap`` (LAP / Kalman / NN)
and ``fa_linking_sa`` (simulated annealing); only the dispatch lives here.

Adapters import those modules LAZILY inside ``.link()`` so this module stays
import-cheap and avoids an import cycle (``fa_localize_backends`` imports
``fa_linking._link_via_trackpy``).  Every adapter returns the input frame with an
integer ``particle`` column, rows with ``particle < 0`` dropped — the same
contract as ``link_trajectories``.
"""
from __future__ import annotations

import pandas as pd

from firefly.analysis.fa_enums import Linker


def _apply_max_len(df: pd.DataFrame, max_len) -> pd.DataFrame:
    """Drop trajectories LONGER than ``max_len`` points (0/None disables).
    ``min_len`` is applied by each linker itself (trackpy via ``filter_stubs``;
    the LAP/Kalman/NN/SA functions via their ``min_len`` arg)."""
    if (max_len and max_len > 0 and df is not None and len(df)
            and "particle" in df.columns):
        lengths = df.groupby("particle")["frame"].count()
        keep = lengths[lengths <= max_len].index
        df = df[df["particle"].isin(keep)].reset_index(drop=True)
    return df


class LinkerBackend:
    """Base adapter.  Subclasses set ``name`` / ``label`` and implement
    ``link``."""
    name: str = "abstract"
    label: str = ""

    @classmethod
    def is_available(cls) -> bool:
        return True

    def link(self, locs, *, search_range, memory, min_len, max_len,
             params: dict, progress_cb=None, stop_event=None) -> pd.DataFrame:
        raise NotImplementedError


class TrackpyLinker(LinkerBackend):
    name = "trackpy"
    label = "Crocker–Grier — Trackpy"

    def link(self, locs, *, search_range, memory, min_len, max_len,
             params, progress_cb=None, stop_event=None):
        import trackpy as tp
        from firefly.analysis.fa_linking import _link_via_trackpy
        linked = _link_via_trackpy(locs, search_range=search_range,
                                   memory=memory, progress_cb=progress_cb,
                                   stop_event=stop_event)
        filtered = tp.filter_stubs(linked, min_len)
        return _apply_max_len(filtered, max_len)


class KalmanLinker(LinkerBackend):
    name = "kalman"
    label = "Kalman filter — TrackMate (Linear Motion)"

    def link(self, locs, *, search_range, memory, min_len, max_len,
             params, progress_cb=None, stop_event=None):
        from firefly.analysis import fa_linking_lap as _lap
        out = _lap.link_trajectories_kalman(
            locs, search_range=search_range, max_gap=memory, min_len=min_len)
        return _apply_max_len(out, max_len)


class SimpleLapLinker(LinkerBackend):
    name = "simple_lap"
    label = "Jaqaman LAP — TrackMate (simple)"

    def link(self, locs, *, search_range, memory, min_len, max_len,
             params, progress_cb=None, stop_event=None):
        from firefly.analysis import fa_linking_lap as _lap
        out = _lap.link_trajectories_lap(
            locs, search_range=search_range, max_gap=memory, min_len=min_len)
        return _apply_max_len(out, max_len)


class FullLapLinker(LinkerBackend):
    name = "full_lap"
    label = "Jaqaman LAP — TrackMate (merge/split)"

    def link(self, locs, *, search_range, memory, min_len, max_len,
             params, progress_cb=None, stop_event=None):
        from firefly.analysis import fa_linking_lap as _lap
        out = _lap.link_trajectories_lap(
            locs, search_range=search_range, max_gap=memory, min_len=min_len,
            allow_merging=bool(params.get("allow_merging", False)),
            allow_splitting=bool(params.get("allow_splitting", False)),
            feature_penalty=bool(params.get("feature_penalty", False)),
            feature_cols=tuple(params.get("feature_cols", ("mass",))),
            penalty_weight=float(params.get("penalty_weight", 1.0)),
            merge_split_cost_factor=float(
                params.get("merge_split_cost_factor", 1.0)))
        return _apply_max_len(out, max_len)


class NearestNeighbourLinker(LinkerBackend):
    name = "nn"
    label = "Nearest-neighbour — greedy"

    def link(self, locs, *, search_range, memory, min_len, max_len,
             params, progress_cb=None, stop_event=None):
        from firefly.analysis import fa_linking_lap as _lap
        out = _lap.link_trajectories_nn(
            locs, search_range=search_range, max_gap=memory, min_len=min_len)
        return _apply_max_len(out, max_len)


class SaLinker(LinkerBackend):
    name = "sa"
    label = "Simulated annealing — palmTRACER (inspired)"

    def link(self, locs, *, search_range, memory, min_len, max_len,
             params, progress_cb=None, stop_event=None):
        from firefly.analysis import fa_linking_sa as _sa
        kw = dict(search_range=search_range, max_gap=memory, min_len=min_len)
        # Pass through only the SA knobs the caller actually set.
        for k in ("seed", "T0", "cooling", "moves_per_temp", "T_min",
                  "w_disp", "w_feat", "sigma_px", "C_birth", "C_death",
                  "C_gap0", "kappa", "allow_merging", "allow_splitting",
                  "C_merge", "C_split"):
            if k in params and params[k] is not None:
                kw[k] = params[k]
        out = _sa.link_trajectories_sa(
            locs, progress_cb=progress_cb, stop_event=stop_event, **kw)
        return _apply_max_len(out, max_len)


# Canonical token → adapter.  ``"lap"`` is a legacy alias for the Simple LAP.
_LINKER_REGISTRY = {
    "trackpy": TrackpyLinker,
    "kalman": KalmanLinker,
    "simple_lap": SimpleLapLinker,
    "lap": SimpleLapLinker,            # legacy token
    "full_lap": FullLapLinker,
    "nn": NearestNeighbourLinker,
    "sa": SaLinker,
}


def list_linkers() -> list[str]:
    """Canonical linker tokens, excluding aliases (UI / bench enumeration)."""
    return [m.value for m in Linker]


def _resolve_linker(name) -> LinkerBackend:
    """Return the adapter for ``name`` (via :meth:`Linker.parse`), defaulting to
    trackpy on an unknown token."""
    key = Linker.parse(name).value
    cls = _LINKER_REGISTRY.get(key) or _LINKER_REGISTRY["trackpy"]
    if not cls.is_available():
        raise RuntimeError(f"linker '{key}' is not available in this build")
    return cls()
