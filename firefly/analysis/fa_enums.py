"""Typed dispatch tokens for FIREFLY.

ONE source of truth for the stringly-typed modes that used to be compared with
bare ``==`` / ``.startswith()`` chains (and could silently fall through to a wrong
default on an unknown value — the bug class behind the "Mean projection vs Sister
TIFF" / "Welch κ … (Kruskal-Wallis)" mislabels).

Each enum's ``.value`` is the EXACT wire string persisted to
``<stem>_run_manifest.json`` / the GUI settings, so ``parse()`` round-trips
byte-for-byte.  ``parse()`` is case-insensitive, accepts the legacy / GUI-display
spellings, and on an UNKNOWN value calls ``log`` (if given) and returns the
documented fallback — never a *silent* wrong default.

Qt-free so the spawned analysis worker can import it.
"""
from __future__ import annotations

from enum import Enum

try:                       # Python 3.11+
    from enum import StrEnum
except ImportError:        # pragma: no cover - fallback for <3.11
    class StrEnum(str, Enum):  # type: ignore
        pass


def _warn(log, msg: str) -> None:
    if callable(log):
        try:
            log(msg)
        except Exception:
            pass


class ROIMode(Enum):
    """How the analysis builds its region-of-interest mask."""
    NONE = "none"        # no ROI — whole frame
    AUTO = "auto"        # intensity-projection auto-threshold
    MANUAL = "manual"    # intensity-projection at a manual threshold
    POLYGON = "polygon"  # user-drawn / ImageJ polygon rasterised to a mask
    SISTER = "sister"    # microscope-exported sister TIFF (e.g. _green.tif)
    IMAGEJ = "imagej"    # auto-paired sibling ImageJ RoiSet.zip / .roi

    @classmethod
    def parse(cls, value, *, log=None) -> "ROIMode":
        s = str(value if value is not None else "none").strip().lower()
        for m in cls:
            if s == m.value:
                return m
        _warn(log, f"  WARNING: unknown ROI mode {value!r} — treating as 'none' "
                   f"(no ROI).")
        return cls.NONE


class MaskMode(Enum):
    """Which per-pixel projection an intensity-threshold ROI is built from.
    ``.value`` doubles as the ``mode_hint`` the mask builder expects."""
    MAX = "max"      # max projection
    MEAN = "mean"    # mean projection (also the Sum target — sum ∝ mean)
    SUM = "sum"      # sum projection (routed to the mean projection)
    BLINK = "blink"  # blink-density projection

    @classmethod
    def parse(cls, value, *, log=None) -> "MaskMode":
        """Maps the GUI display strings ("Max" / "Mean" / "Sum" / "Blink
        density") case-insensitively, preserving the historical prefix dispatch.
        An empty/missing value → Mean (the legacy default for the `else` branch);
        an *unknown* non-empty value → Mean with a logged warning (was silent)."""
        s = str(value if value is not None else "").strip().lower()
        if s.startswith("blink"):
            return cls.BLINK
        if s.startswith("max"):
            return cls.MAX
        if s.startswith("sum"):
            return cls.SUM
        if s.startswith("mean") or s == "":
            return cls.MEAN
        _warn(log, f"  WARNING: unknown ROI mask mode {value!r} — using Mean "
                   f"projection.")
        return cls.MEAN


class FigureTheme(Enum):
    """Figure colour theme.  Values are the exact strings stored in the run
    manifest / settings."""
    DARK = "Dark"
    LIGHT = "Light"
    PUBLICATION = "Publication"
    AMOLED = "AMOLED"

    @classmethod
    def parse(cls, value, *, log=None) -> "FigureTheme":
        s = str(value if value is not None else "Dark").strip()
        for m in cls:
            if s.lower() == m.value.lower():
                return m
        _warn(log, f"  WARNING: unknown figure theme {value!r} — using Dark.")
        return cls.DARK


class Backend(Enum):
    """Detection backend (base name; a ``:<device-index>`` suffix like
    ``torch-cuda:0`` is handled by the caller, not the enum)."""
    AUTO = "auto"
    TRACKPY = "trackpy"
    TORCH = "torch"            # GPU-auto device
    TORCH_CPU = "torch-cpu"
    TORCH_CUDA = "torch-cuda"
    TORCH_MPS = "torch-mps"
    ATROUS = "atrous"          # à trous wavelet detector (auto-device, like torch)
    GAUSSIAN_MLE = "gaussian-mle"      # Crocker–Grier + Gaussian-MLE refiner (auto-device)
    RADIAL_SYMMETRY = "radial-symmetry"  # Crocker–Grier + radial-symmetry refiner (auto-device)

    @classmethod
    def parse(cls, value, *, log=None) -> "Backend":
        s = str(value if value is not None else "auto").strip().lower()
        s = s.split(":", 1)[0]                 # drop a ":device-index" suffix
        for m in cls:
            if s == m.value:
                return m
        _warn(log, f"  WARNING: unknown backend {value!r} — using auto.")
        return cls.AUTO

    @property
    def is_torch(self) -> bool:
        return self in (Backend.TORCH, Backend.TORCH_CPU,
                        Backend.TORCH_CUDA, Backend.TORCH_MPS)

    @property
    def is_explicit_gpu(self) -> bool:
        """True for the device-pinned GPU backends.  Plain ``torch`` / ``auto``
        resolve their device at runtime, so GPU-ness is decided there, not here."""
        return self in (Backend.TORCH_CUDA, Backend.TORCH_MPS)


class Linker(Enum):
    """Trajectory-linking algorithm.

    ``trackpy``    — Crocker–Grier recursive-subnet nearest-neighbour (best for
                     pure Brownian diffusion).  **Forward default** for a fresh
                     run (`DEFAULT_LINKER`), and the parse / pre-linker-manifest
                     REPLAY fallback — see `parse` and `DEFAULT_LINKER` below.
    ``kalman``     — constant-velocity Kalman LAP (TrackMate "Linear Motion";
                     directed / crossing motion).
    ``simple_lap`` — Jaqaman two-step LAP: frame-to-frame + gap-closing, no
                     merge/split (TrackMate "Simple LAP"; legacy token ``lap``).
    ``full_lap``   — Simple LAP + optional merge/split + feature penalties
                     (TrackMate's full LAP tracker).
    ``nn``         — greedy nearest-neighbour (TrackMate "Nearest-neighbour").
    ``sa``         — simulated-annealing multi-target tracker (palmTRACER-style;
                     independent reimplementation of Racine & Sibarita 2006).
    """
    TRACKPY = "trackpy"
    KALMAN = "kalman"
    SIMPLE_LAP = "simple_lap"
    FULL_LAP = "full_lap"
    NN = "nn"
    SA = "sa"

    @classmethod
    def parse(cls, value, *, log=None) -> "Linker":
        s = str(value if value is not None else "trackpy").strip().lower()
        for m in cls:
            if s == m.value:
                return m
        # Accepted spellings / legacy tokens → canonical member.
        aliases = {
            "lap": cls.SIMPLE_LAP, "jaqaman": cls.SIMPLE_LAP,
            "simple-lap": cls.SIMPLE_LAP,
            "nearest": cls.NN, "nearest_neighbour": cls.NN,
            "nearest-neighbour": cls.NN, "nearest_neighbor": cls.NN,
            "nearest-neighbor": cls.NN,
            "annealing": cls.SA, "simulated_annealing": cls.SA,
            "palmtracer": cls.SA, "palm-tracer": cls.SA,
        }
        if s in aliases:
            return aliases[s]
        _warn(log, f"  WARNING: unknown linker {value!r} — using trackpy.")
        return cls.TRACKPY


class GapPolicy(Enum):
    """How trajectory gaps contribute to time-lag analyses.

    ``all_pairs`` is the conventional timestamp-lag estimator: for a requested
    lag ``L`` it uses every pair of observations whose frame numbers differ by
    exactly ``L``, even when one or more intermediate observations are absent.
    ``contiguous`` is retained for reproducibility with FIREFLY's historical
    row-offset behaviour: it only uses pairs inside an uninterrupted observed
    run.  The wire values are persisted in run metadata.
    """
    ALL_PAIRS = "all_pairs"
    CONTIGUOUS = "contiguous"

    @classmethod
    def parse(cls, value, *, log=None) -> "GapPolicy":
        # Internal callers deliberately pass the parsed enum through to lower
        # level helpers.  ``str(GapPolicy.CONTIGUOUS)`` is not its wire value,
        # so preserve an already-canonical member before normalising strings.
        if isinstance(value, cls):
            return value
        s = str(value if value is not None else cls.ALL_PAIRS.value).strip().lower()
        aliases = {
            "all_pairs": cls.ALL_PAIRS,
            "all-pairs": cls.ALL_PAIRS,
            "actual_frame": cls.ALL_PAIRS,
            "actual-frame": cls.ALL_PAIRS,
            "timestamp": cls.ALL_PAIRS,
            "frame_lag": cls.ALL_PAIRS,
            "contiguous": cls.CONTIGUOUS,
            "contiguous_runs": cls.CONTIGUOUS,
            "contiguous-runs": cls.CONTIGUOUS,
        }
        if s in aliases:
            return aliases[s]
        _warn(log, f"  WARNING: unknown gap policy {value!r} — using all_pairs.")
        return cls.ALL_PAIRS


# Single source of truth for the FORWARD default linker (a fresh run that does
# not specify one) — must match the GUI's first-listed combo entry and the
# README.  The re-ROI / pre-linker-manifest REPLAY default
# (`firefly_worker._POSTPROC_LINK_DEFAULTS["linker"]`) is also "trackpy", so an
# old run with no recorded linker replays faithfully.
DEFAULT_LINKER = Linker.TRACKPY.value   # "trackpy"


class MsgKind(StrEnum):
    """Worker→GUI message-queue kinds — one source of truth so a typo can't
    silently drop a message.  StrEnum members compare equal to their plain
    string, so emit/read sites can be migrated incrementally.

    Payload contracts (the worker sets these keys; the GUI reads them
    defensively with ``.get()``):
      LOG            : str
      PROGRESS       : (int pct, str stage)
      MASS_CHUNK     : list[float]
      PREVIEW_FRAME  : {shape:[H,W], frame:bytes, xs, ys, idx, n_frames, [file]}
      DONE           : {stem, out_dir, figure_path, summary:{...}, n_tracks, n_locs}
      FILE_STARTING  : {index, total, file}
      FILE_DONE      : {index, total, stem, out_dir, n_tracks, n_locs}
      FILE_ERROR     : {index, total, file, tb}
      BATCH_DONE     : {n_total, n_ok, n_fail, results:[...]}
      COMPARE_DONE   : {output_dir, figure_path, summary_csv, stats_csv,
                        pdf_report, results_json, n_groups}
      COMPARE_ERROR  : str
      HF_TILE        : {file, stem, state, [pct, stage, n_locs, n_tracks, error]}
      HYPERFLY_STATUS: {active, n_concurrent, per_file_workers, reason}
      STOPPED        : None
      ERROR          : str
    """
    LOG = "log"
    PROGRESS = "progress"
    MASS_CHUNK = "mass_chunk"
    PREVIEW_FRAME = "preview_frame"
    DONE = "done"
    FILE_STARTING = "file_starting"
    FILE_DONE = "file_done"
    FILE_ERROR = "file_error"
    BATCH_DONE = "batch_done"
    COMPARE_DONE = "compare_done"
    COMPARE_ERROR = "compare_error"
    HF_TILE = "hf_tile"
    HYPERFLY_STATUS = "hyperfly_status"
    STOPPED = "stopped"
    ERROR = "error"
