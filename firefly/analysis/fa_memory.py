"""Memory / disk-backed-stack management for the FIREFLY pipeline.

Shared by both the loaders (deciding RAM vs disk-memmap when reading a stack)
and the streaming localiser (chunk sizing).  Extracted from
sptpalm_analysis.py as part of modularising it (#7); re-exported there so
existing call sites — including firefly_worker's `cleanup_temp_stack_paths`
import — keep working unchanged.
"""
from __future__ import annotations

import os

import numpy as np

# ── Disk-backed memmap bookkeeping ────────────────────────────────────────────
# When a loader falls back to a disk-backed memmap (because the stack won't fit
# in RAM), we leave the file on disk for the duration of the run.  An atexit
# hook removes these temp files so they don't accumulate.
_firefly_temp_stack_paths: list = []
# Optional override for where the disk-backed memmap is created.  Set via env
# var FIREFLY_TEMP_DIR or set_temp_stack_dir().  Defaults to the OS temp dir
# (often the smallest drive); pointing it at the data drive avoids ENOSPC on
# long batches.
_firefly_temp_stack_dir: "str | None" = None


def set_temp_stack_dir(d: "str | None") -> None:
    """Override the directory used for disk-backed memmap stacks."""
    global _firefly_temp_stack_dir
    _firefly_temp_stack_dir = d or None


def _resolve_temp_stack_dir() -> "str | None":
    d = _firefly_temp_stack_dir or os.environ.get("FIREFLY_TEMP_DIR") or None
    if d and not os.path.isdir(d):
        try: os.makedirs(d, exist_ok=True)
        except Exception: return None
    return d


def _register_temp_stack_path(p: str) -> None:
    import atexit
    if not _firefly_temp_stack_paths:
        atexit.register(_cleanup_temp_stack_paths)
    _firefly_temp_stack_paths.append(p)


# Public alias used by the batch runner between files.
def cleanup_temp_stack_paths() -> None:
    _cleanup_temp_stack_paths()


def _cleanup_temp_stack_paths() -> None:
    """Remove every temp memmap file registered so far and clear the list.

    Safe to call mid-run between batch files — by the time a per-file
    analysis returns, the `combined` memmap reference has gone out of
    scope, so `os.remove` will succeed on POSIX (the OS unlinks the
    inode; any lingering mapping survives until the last fd closes).

    On Windows the file is still locked until the underlying `mmap`
    object's handle is explicitly closed and a gc cycle has run —
    we force both before each unlink, retry once if the first
    PermissionError says "file in use", and silently skip the path
    if it's still locked (atexit will get it on process exit).
    """
    import gc as _gc
    _gc.collect()
    still_locked = []
    for p in list(_firefly_temp_stack_paths):
        try:
            os.remove(p)
        except PermissionError:
            # Windows: handle still alive somewhere.  One more gc +
            # retry usually does it.
            _gc.collect()
            try:    os.remove(p)
            except Exception:
                still_locked.append(p)
        except Exception:
            pass
    _firefly_temp_stack_paths.clear()
    # Re-register any we couldn't delete so the next call (or atexit)
    # gets another chance.
    _firefly_temp_stack_paths.extend(still_locked)


#  How much physical RAM to leave for the OS + the user's other apps.
#  Without this reserve, FIREFLY's memory checks would happily consume every
#  free byte; the moment the user opens a Safari tab the system starts
#  swapping or OOM-killing.  Formula: a fixed 4 GB floor, but at most
#  0.15 × total RAM capped at 8 GB.  Override via FIREFLY_USER_RAM_RESERVE_GB.
def _user_ram_reserve_gb() -> float:
    """RAM (in GB) we deliberately keep available for non-FIREFLY uses."""
    try:
        env = os.environ.get("FIREFLY_USER_RAM_RESERVE_GB")
        if env:
            return max(0.5, float(env))
    except Exception:
        pass
    try:
        import psutil as _ps
        total_gb = _ps.virtual_memory().total / 1e9
    except Exception:
        total_gb = 8.0   # conservative fallback if psutil is missing
    return max(4.0, min(8.0, 0.15 * total_gb))


def _alloc_or_memmap_stack(shape, dtype=np.float32, reserve_gb=None):
    """Allocate a single-file (T, H, W) stack.  Returns a plain in-RAM array
    when it fits in available memory (minus the OS reserve), otherwise a
    disk-backed np.memmap so a stack larger than RAM doesn't OOM or silently
    swap.  Mirrors the RAM-vs-memmap policy the multi-file / TIF loaders
    already use, extending it to the single-file CZI/TIF paths.

    ``reserve_gb`` overrides the default OS/user RAM reserve — a display-only
    caller (the Visualiser) can pass a smaller value so a movie it just shows
    stays in RAM instead of spilling to a slow disk memmap."""
    import numpy as _np
    nbytes = int(_np.prod(shape)) * _np.dtype(dtype).itemsize
    fits_ram = True
    try:
        import psutil as _ps
        avail   = _ps.virtual_memory().available
        reserve = (reserve_gb if reserve_gb is not None
                   else _user_ram_reserve_gb()) * (1024 ** 3)
        fits_ram = nbytes < (avail - reserve)
    except Exception:
        fits_ram = True
    if fits_ram:
        return _np.empty(shape, dtype=dtype)
    # Too big for RAM → disk-backed memmap.
    import tempfile, shutil
    tmp_dir   = _resolve_temp_stack_dir()
    probe_dir = tmp_dir or tempfile.gettempdir()
    try:
        if shutil.disk_usage(probe_dir).free < nbytes * 1.05:
            print("  WARN: stack exceeds available RAM and the temp disk is "
                  "low; attempting an in-RAM load anyway.", flush=True)
            return _np.empty(shape, dtype=dtype)
    except Exception:
        pass
    fd, tmp_path = tempfile.mkstemp(suffix=".dat", prefix="firefly_stack_",
                                    dir=tmp_dir)
    os.close(fd)
    _register_temp_stack_path(tmp_path)
    print(f"  Stack too large for RAM ({nbytes/1e9:.1f} GB) -> disk memmap at "
          f"{tmp_path}", flush=True)
    return _np.memmap(tmp_path, dtype=dtype, mode="w+", shape=tuple(shape))
