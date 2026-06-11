"""HYPERFLY mode — auto-detected high-throughput batch processing.

On a big machine (many cores + lots of RAM, e.g. a 128-core / 752 GB box) the
batch pipeline is bottlenecked by per-file *serial* stages — TIF loading (I/O),
trackpy linking (single-threaded), figure rendering — which leave most cores
idle.  HYPERFLY processes several files **concurrently** and **RAM-resident**,
overlapping one file's idle stages with another's compute so the whole machine
stays busy.  The actual engine lives in ``firefly_worker.run_batch_analysis``;
this module only decides *whether* to engage and *how wide* to go.

Qt-free on purpose: this module is imported by the analysis worker subprocess,
which must never pull in Qt.  Configuration therefore comes from **environment
variables** (the GUI reads its QSettings and exports these before spawning the
worker), NOT from QSettings directly:

    FIREFLY_HYPERFLY            auto | on | off   (default: auto)
    FIREFLY_HYPERFLY_MAX_FILES  int, 0 = automatic (cap concurrent files)
    FIREFLY_HYPERFLY_MAX_CORES  int, 0 = automatic (cap total cores used)
    FIREFLY_HYPERFLY_MAX_RAM_GB int, 0 = automatic (cap peak RAM, GB)
"""
from __future__ import annotations

import os

from firefly.analysis.fa_constants import N_CPUS

# Auto-trigger thresholds — a machine must clear BOTH to auto-engage HYPERFLY.
HYPERFLY_MIN_CORES = 32
HYPERFLY_MIN_RAM_GB = 192
# Never split so thin that a file gets fewer than this many cores (detection /
# MSD stop scaling below a handful of cores, and the per-process overhead grows).
MIN_CORES_PER_FILE = 4


def _total_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except Exception:
        return 0.0


def _free_ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        return 0.0


def _env_int(name: str) -> int:
    try:
        return max(0, int(os.environ.get(name, "0")))
    except (ValueError, TypeError):
        return 0


def hyperfly_mode() -> str:
    """Configured mode: 'auto' (default), 'on', or 'off'."""
    v = str(os.environ.get("FIREFLY_HYPERFLY", "auto")).strip().lower()
    return v if v in ("auto", "on", "off") else "auto"


def hyperfly_active() -> bool:
    """True when HYPERFLY should engage: forced on, or 'auto' on a machine with
    enough cores AND RAM."""
    mode = hyperfly_mode()
    if mode == "off":
        return False
    if mode == "on":
        return True
    return (N_CPUS >= HYPERFLY_MIN_CORES
            and _total_ram_gb() >= HYPERFLY_MIN_RAM_GB)


def _per_file_peak_gb(params: dict) -> float:
    """Conservative peak-RAM estimate (GB) for one file, from its input size on
    disk.  Peak ≈ raw + preprocessed + slop ≈ 3× the raw bytes (mirrors the
    multiplier in fa_localize._ram_strategy).  Floored so an unknown/tiny file
    still reserves something sane."""
    files = params.get("series_files") or [params.get("file")]
    total = 0
    for f in files:
        try:
            if f and os.path.isfile(f):
                total += os.path.getsize(f)
        except Exception:
            pass
    return max(0.5, (total / 1e9) * 3.0)


def plan_concurrency(params_list: list) -> dict:
    """Decide how many files to run at once and the per-file core budget.

    Returns a dict: ``{active, n_concurrent, per_file_workers, free_gb,
    per_file_gb, reason}``.  ``active`` is False (n_concurrent=1) when HYPERFLY
    is off, there's only one file, or the machine can only fit one file at a
    time — in which case the caller runs the normal serial path.

    Invariants:
      * ``n_concurrent`` files fit in usable RAM at once (so each takes the
        FAST in-RAM path), and
      * ``n_concurrent × per_file_workers`` stays within the core budget (no
        oversubscription), honouring the optional IT caps.
    """
    from firefly.analysis.fa_memory import _user_ram_reserve_gb

    n_files = len(params_list)
    if not hyperfly_active() or n_files < 2:
        return {"active": False, "n_concurrent": 1, "per_file_workers": N_CPUS,
                "free_gb": _free_ram_gb(), "per_file_gb": 0.0,
                "reason": "HYPER-FLY inactive or single file"}

    free_gb = _free_ram_gb()
    usable_gb = max(0.0, free_gb - _user_ram_reserve_gb())
    # Optional hard RAM cap (FIREFLY_HYPERFLY_MAX_RAM_GB) — bound HYPERFLY's peak
    # footprint so it stays a good neighbour on a shared machine even when lots
    # of RAM is free.  0 = auto (just free-RAM bounded).
    max_ram = _env_int("FIREFLY_HYPERFLY_MAX_RAM_GB")    # 0 = auto
    ram_budget_gb = min(usable_gb, max_ram) if max_ram > 0 else usable_gb
    # Size the wave to the LARGEST file so even the biggest fits in RAM.
    per_file_gb = max((_per_file_peak_gb(p) for p in params_list), default=0.5)

    k_ram = int(ram_budget_gb // per_file_gb) if per_file_gb > 0 else n_files
    k_cores = max(1, N_CPUS // MIN_CORES_PER_FILE)
    max_files = _env_int("FIREFLY_HYPERFLY_MAX_FILES")    # 0 = auto
    max_cores = _env_int("FIREFLY_HYPERFLY_MAX_CORES")    # 0 = auto
    budget_cores = max_cores if max_cores > 0 else N_CPUS

    k = min(n_files, max(1, k_ram), k_cores)
    if max_files > 0:
        k = min(k, max_files)
    k = max(1, k)

    per_file_workers = max(1, budget_cores // k)
    active = k >= 2
    _ram_note = (f"≤{max_ram} GB cap" if max_ram > 0 else f"free {free_gb:.0f} GB")
    return {
        "active": active,
        "n_concurrent": k if active else 1,
        "per_file_workers": per_file_workers if active else N_CPUS,
        "free_gb": round(free_gb, 1),
        "per_file_gb": round(per_file_gb, 1),
        "ram_budget_gb": round(ram_budget_gb, 1),
        "reason": (f"{k} files × {per_file_workers} cores "
                   f"({_ram_note}, ~{per_file_gb:.0f} GB/file)"
                   if active else
                   f"only one file fits in RAM ({_ram_note}, "
                   f"~{per_file_gb:.0f} GB/file)"),
    }
