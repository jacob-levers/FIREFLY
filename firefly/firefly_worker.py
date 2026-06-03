"""
FIREFLY analysis subprocess worker.

This module deliberately imports NOTHING related to Qt / PySide6 / GUI
toolkits.  Why: when `multiprocessing.spawn` (the macOS-default start
method) creates a child process, it re-imports the module that defines
the target function in order to unpickle and call it.  If the target
lived in `app_qt.py`, the spawned subprocess would re-import
`app_qt.py` → `PySide6` → Qt 6's Metal-backed window compositor —
which on Apple Silicon claims memory from the same unified memory pool
PyTorch's MPS allocator needs.  Two Metal-using processes on a 16 GB
M-series Mac is enough to push PyTorch over the edge with "Insufficient
Memory" command-buffer errors.

By keeping the worker in this Qt-free module, the analysis subprocess
imports only Python stdlib + sptpalm_analysis (numpy / scipy / trackpy /
optionally torch) — no Metal-using framework, full unified-memory pool
available for MPS.

Public entry points (both used by app_qt.py):
    run_analysis(params, msg_queue, cancel_event)
        Single-file analysis.

    run_batch_analysis(params_list, msg_queue, cancel_event)
        Batch mode — same pipeline run sequentially over multiple files
        in a single subprocess (one spawn cost, N analyses).
"""
from __future__ import annotations

import os
import sys
import time
import traceback


# ── CUDA sidecar injection ───────────────────────────────────────────────────
# Must run BEFORE any torch import so the CUDA-built torch in
# %LOCALAPPDATA%\FIREFLY\torch-cuda can shadow the bundled CPU build.
# A failure here (no GPU, no sidecar, permissions, etc.) must NEVER
# crash the worker — fall through silently to the CPU build.
try:
    from firefly.cuda_installer import inject_sidecar_into_sys_path
    inject_sidecar_into_sys_path()
except Exception:
    pass


# ── MPS allocator tuning — must be set BEFORE torch import anywhere ──────────
# See app_qt.py for the rationale.  Setting here too is cheap
# insurance in case the parent's setting somehow didn't reach the child.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# ══════════════════════════════════════════════════════════════════════════════
#  CROSS-PROCESS LOG STREAM
# ══════════════════════════════════════════════════════════════════════════════
class QueueLogStream:
    """File-like stream that posts each newline-/carriage-return-terminated
    line to a multiprocessing.Queue as a ('log', line) tuple.

    Used inside the analysis subprocess to forward `print()` calls and
    tqdm progress bars to the parent's Qt log box.  tqdm rewrites a single
    line with '\\r'; we treat both '\\r' and '\\n' as terminators so each
    tqdm update becomes one log entry instead of one giant line.
    """
    def __init__(self, q):
        self._q   = q
        self._buf = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while True:
            idx_n = self._buf.find("\n")
            idx_r = self._buf.find("\r")
            cuts = [i for i in (idx_n, idx_r) if i >= 0]
            if not cuts:
                break
            cut = min(cuts)
            line = self._buf[:cut]
            self._buf = self._buf[cut + 1:]
            if line.strip():
                self._q.put(("log", line.rstrip()))
        return len(s)

    def flush(self):
        if self._buf.strip():
            self._q.put(("log", self._buf.rstrip()))
            self._buf = ""

    def isatty(self) -> bool: return False
    def fileno(self):         raise OSError("not a real fd")


# ══════════════════════════════════════════════════════════════════════════════
#  CORE PIPELINE (shared by single-file and batch entry points)
# ══════════════════════════════════════════════════════════════════════════════
def _write_run_manifest(*, out_dir: str, stem: str, fpath: str,
                        params: dict) -> str:
    """Write a `<stem>_run_manifest.json` file alongside the run outputs.
    The manifest captures everything needed to reproduce the run:
      • full parameters (worker-format kwargs + widget-state for the GUI)
      • input file path + SHA-256 checksum
      • FIREFLY version, git SHA (if available), host info
      • timestamp + output directory
    """
    import datetime as _dt
    import hashlib
    import json
    import platform
    import socket
    import subprocess

    def _file_sha256(path: str, _chunk: int = 1 << 20) -> str:
        h = hashlib.sha256()
        try:
            with open(path, "rb") as fh:
                while True:
                    blk = fh.read(_chunk)
                    if not blk:
                        break
                    h.update(blk)
            return h.hexdigest()
        except Exception:
            return ""

    def _firefly_version() -> str:
        try:
            from firefly import sptpalm_analysis as _sa
            v = getattr(_sa, "__version__", None)
            return str(v) if v else "unknown"
        except Exception:
            return "unknown"

    def _git_sha() -> str:
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            out = subprocess.check_output(
                ["git", "-C", here, "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL, timeout=2)
            return out.decode().strip()
        except Exception:
            return ""

    # Strip non-JSON-serialisable bits out of the params dict (roi_polygon
    # is a list-of-tuples, widget_state is a flat str/num/bool dict — both
    # are fine).  `json.dumps` raises on numpy arrays / etc., so we coerce.
    def _jsonify(obj):
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, dict):
            return {str(k): _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify(x) for x in obj]
        # numpy scalars / pandas / etc.
        try:    return float(obj)
        except Exception: pass
        try:    return int(obj)
        except Exception: pass
        return str(obj)

    widget_state = params.get("widget_state") or {}
    # Worker-format kwargs minus the widget snapshot (it lives in its own field)
    worker_params = {k: _jsonify(v) for k, v in params.items()
                     if k != "widget_state"}

    manifest = {
        "schema_version":   1,
        "firefly_version":  _firefly_version(),
        "git_sha":          _git_sha(),
        "created_at":       _dt.datetime.now().isoformat(timespec="seconds"),
        "host": {
            "name":     socket.gethostname(),
            "platform": platform.platform(),
            "python":   platform.python_version(),
        },
        "input": {
            "path":   fpath,
            "sha256": _file_sha256(fpath),
        },
        "output_dir":    out_dir,
        "stem":          stem,
        "parameters":    worker_params,
        "widget_state":  _jsonify(widget_state),
    }

    path = os.path.join(out_dir, f"{stem}_run_manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def _start_memory_watchdog(cancel_event, msg_queue,
                            critical_gb: float | None = None,
                            warn_gb: float | None = None,
                            poll_s: float = 0.5,
                            mem_abort_event=None,
                            sustained_polls: int = 6):
    """Background daemon that aborts the run cleanly if free RAM gets
    dangerously low.

    Scaling-aware thresholds (key change from earlier versions):
    `critical_gb` and `warn_gb` default to fractions of total system
    RAM, NOT hardcoded constants.  A 32 GB Windows box was hitting
    the old 0.8 GB abort threshold during normal preprocessing-to-
    localisation transitions (6 parallel workers each holding a
    frame buffer for a beat before Python releases) — even though
    the OS had GBs to spare.  Fractions of total mean smaller
    machines still get protection, bigger machines aren't aborted
    on transient spikes.

    `sustained_polls` debounces aborts: we need to see N consecutive
    sub-critical readings before pulling the plug.  Default 6 polls
    × 0.5 s = 3 s sustained low memory, which is long enough to
    distinguish "Python hasn't GC'd the last chunk yet" from "we're
    truly OOM".

    Returns a `stop` Event the caller `.set()`s in a `finally` block
    to shut the watchdog down at the end of the run.
    """
    import threading
    try:
        import psutil as _psutil
    except Exception:
        # Can't monitor without psutil — degrade silently.
        return None

    # Resolve thresholds from total RAM if caller didn't pin them.
    try:
        total_gb = _psutil.virtual_memory().total / 1e9
    except Exception:
        total_gb = 8.0
    # Env-var override — FIREFLY_MEM_ABORT_GB takes precedence over
    # everything else.  Useful for users on shared machines who need a
    # tighter or looser threshold than the default.
    _env_abort = os.environ.get("FIREFLY_MEM_ABORT_GB")
    if _env_abort:
        try:    critical_gb = float(_env_abort)
        except ValueError: pass
    if critical_gb is None:
        # 3% of total RAM, never less than 0.5 GB, never more than 2 GB.
        critical_gb = max(0.5, min(2.0, total_gb * 0.03))
    if warn_gb is None:
        warn_gb = max(critical_gb * 2.0, 1.0)

    stop = threading.Event()

    def _watch():
        warned = False
        low_streak = 0
        while not stop.wait(poll_s):
            try:
                free_gb = _psutil.virtual_memory().available / 1e9
            except Exception:
                continue
            if free_gb < critical_gb:
                low_streak += 1
                if low_streak < sustained_polls:
                    # Transient spike — don't pull the plug yet.
                    continue
                if cancel_event is not None and not cancel_event.is_set():
                    # Set the abort signals BEFORE attempting to enqueue
                    # the log message.  Under memory pressure the queue
                    # may itself be full (GUI is starved of CPU and not
                    # draining); a blocking put_nowait → queue.Full
                    # exception is far better than hanging here forever
                    # and never tripping cancel_event at all.
                    if mem_abort_event is not None:
                        try: mem_abort_event.set()
                        except Exception: pass
                    cancel_event.set()
                    try:
                        msg_queue.put_nowait(("log",
                            f"\n  ⚠ CRITICAL: only {free_gb:.2f} GB RAM "
                            f"free for {sustained_polls * poll_s:.1f}s "
                            f"(threshold {critical_gb:.2f} GB on a "
                            f"{total_gb:.0f} GB machine) — aborting to "
                            f"keep the system responsive.  Close other "
                            f"apps or set FIREFLY_MEM_ABORT_GB to "
                            f"override."))
                    except Exception: pass
                return
            else:
                # RAM recovered — reset the streak counter and the
                # one-shot warn flag so future dips are flagged again.
                if low_streak > 0:
                    low_streak = 0
                    warned = False
            if not warned and free_gb < warn_gb:
                # Same put_nowait pattern — never block the watchdog
                # on a queue that the memory-starved GUI may have
                # stopped draining.
                try:
                    msg_queue.put_nowait(("log",
                        f"  ⚠ Memory pressure: {free_gb:.2f} GB RAM "
                        f"free (warn < {warn_gb:.1f} GB / abort < "
                        f"{critical_gb:.1f} GB).  Run will abort if "
                        f"this stays low for "
                        f"{sustained_polls * poll_s:.1f}s."))
                except Exception: pass
                warned = True

    t = threading.Thread(target=_watch, daemon=True,
                          name="FIREFLY-MemWatchdog")
    t.start()
    return stop


def _start_disk_watchdog(out_dir, cancel_event, msg_queue,
                          critical_mb: float = 200.0,
                          warn_mb: float = 1024.0,
                          poll_s: float = 2.0):
    """Background daemon that aborts the run if free disk space on the
    OUTPUT volume drops dangerously low.

    Thresholds (in MB of free space on the volume holding `out_dir`):
      * `warn_mb`     — log a warning the first time we cross it
      * `critical_mb` — set `cancel_event` so the pipeline raises
                        `_Cancelled` at its next stop-check

    This catches the common ENOSPC-mid-analysis case where a multi-GB
    movie load + intermediate memmap + 200k-row CSV outputs slowly
    consume the drive.  Without it, a single failed `to_csv` torpedoes
    a 5-minute run at the very last save step.

    Returns the `stop` Event to .set() in a finally block, or None if
    psutil/shutil checks aren't available on this platform.
    """
    import threading, shutil
    stop = threading.Event()
    target = os.path.dirname(os.path.abspath(out_dir)) or out_dir

    def _free_mb():
        try:
            return shutil.disk_usage(target).free / 1e6
        except Exception:
            return None

    # If we can't measure, don't try to enforce.
    if _free_mb() is None:
        return None

    def _watch():
        warned = False
        while not stop.wait(poll_s):
            free_mb = _free_mb()
            if free_mb is None:
                continue
            if free_mb < critical_mb:
                if cancel_event is not None and not cancel_event.is_set():
                    cancel_event.set()
                    try:
                        msg_queue.put_nowait(("log",
                            f"\n  ⚠ CRITICAL: only {free_mb:.0f} MB "
                            f"free on output disk — aborting before "
                            f"a save fails and corrupts the output "
                            f"folder.  Free up space or change the "
                            f"output folder and re-run."))
                    except Exception: pass
                return
            if not warned and free_mb < warn_mb:
                try:
                    msg_queue.put_nowait(("log",
                        f"  ⚠ Disk space low on output volume: "
                        f"{free_mb:.0f} MB free (warn < "
                        f"{warn_mb:.0f} MB / abort < {critical_mb:.0f} "
                        f"MB).  Outputs may fail to save."))
                except Exception: pass
                warned = True

    t = threading.Thread(target=_watch, daemon=True,
                          name="FIREFLY-DiskWatchdog")
    t.start()
    return stop


class _NoTracks(Exception):
    """Raised inside _run_one_analysis when linking produces 0 trajectories.
    The wrapper catches this and emits a sensible 'done' (single-file) or
    'file_done' (batch) payload — no crash report."""


def _make_loc_histogram_proj(locs_df, p: dict):
    """Build a 2-D localisation-density image for use as the figure's
    Max-Projection background when no real image stack is available.

    Used by the external_csv path (which post-processing dispatches to).
    Without this, `proj_sample` defaulted to zeros((1, 256, 256)) — a black
    square that ALSO miscalibrated the trajectory panels' axis limits
    when the real data extent was larger than 256 px.

    Returns shape (1, H, W) so the downstream `make_figure` code which
    expects a 3-D stack still works unchanged.
    """
    import numpy as _np
    try:
        xs = _np.asarray(locs_df["x"], dtype=float)
        ys = _np.asarray(locs_df["y"], dtype=float)
        if xs.size == 0 or ys.size == 0:
            return _np.zeros((1, 256, 256), dtype=_np.float32)
        # Prefer the run's recorded image size when present (so axes match
        # the original image exactly), otherwise size to the data extent.
        W = int(p.get("width")  or _np.ceil(xs.max()) + 1)
        H = int(p.get("height") or _np.ceil(ys.max()) + 1)
        W = max(W, 32); H = max(H, 32)
        hist, _, _ = _np.histogram2d(
            ys, xs, bins=(H, W), range=[[0, H], [0, W]])
        return hist.astype(_np.float32)[None, :, :]
    except Exception:
        return _np.zeros((1, 256, 256), dtype=_np.float32)


def _run_one_analysis(params: dict, msg_queue, cancel_event,
                      _log, _prog) -> dict:
    """Run the FIREFLY pipeline on one input file.

    Returns the "done"-payload dict (stem, out_dir, figure_path, n_tracks,
    n_locs).  Raises `sptpalm_analysis._Cancelled` if the user stopped via
    `cancel_event`.  Raises `_NoTracks` if linking yielded nothing.  Other
    exceptions propagate so the caller decides whether to abort or continue
    (batch continues to next file; single-file emits a crash report).
    """
    p = params

    from firefly.sptpalm_analysis import (
        load_file, preprocess_and_localise_adaptive, link_trajectories,
        compute_msd_and_fit, compute_jdd, compute_turning_angles,
        compute_van_hove,
        compute_vacf,
        compute_circular_statistics, save_circular_statistics_pdf,
        compute_per_track_mean_angle, _circ_lin_correlation,
        compute_mobile_fraction_over_time, compute_clusters,
        compute_dwell_times, compute_mss, correct_drift,
        make_figure, save_palmtracer_csvs, apply_roi_mask, _Cancelled,
        load_external_locs,
    )

    # Helper: check stop event at major pipeline boundaries.  Most of the
    # pipeline's interruptibility comes from passing `cancel_event` deep
    # into load_file / preprocess_and_localise_adaptive, but those functions
    # poll only periodically.  Adding explicit checks BETWEEN stages means
    # a Stop click during e.g. the linker's long uninterruptible region
    # will at least halt before the next stage starts.
    def _check_stop():
        if cancel_event.is_set():
            raise _Cancelled()

    fpath = p["file"]
    # By default the output-folder stem is just the file's basename, but
    # the batch caller can override this via `stem_override` — used to
    # disambiguate same-named files coming from different subfolders
    # (e.g. Cell1/Loc.txt vs Cell2/Loc.txt → "Cell1__Loc" vs "Cell2__Loc"
    # so each cell gets its own per-stem subfolder under batch_results/).
    stem  = (str(p.get("stem_override")
                 or os.path.splitext(os.path.basename(fpath))[0]))
    # Wrap every run's artifacts inside a per-stem subfolder so the user's
    # chosen output directory stays tidy when batch-processing multiple
    # files (and so the per-run files are obviously grouped together
    # rather than scattered across figures/, data/, firefly_extras/, plus
    # a loose run_manifest.json sitting at the top level).
    # The caller can opt out by passing `wrap_in_stem_folder=False` — used
    # by the Post-process path which appends `_postproc{N}` to the raw
    # source folder and doesn't want to nest again.
    raw_out_dir = p.get("out_dir") or os.path.dirname(os.path.abspath(fpath))
    if bool(p.get("wrap_in_stem_folder", True)):
        out_dir = os.path.join(raw_out_dir, stem)
    else:
        out_dir = raw_out_dir
    fig_dir    = os.path.join(out_dir, "figures")
    data_dir   = os.path.join(out_dir, "data")
    extras_dir = os.path.join(out_dir, "firefly_extras")
    for d in (fig_dir, data_dir, extras_dir):
        os.makedirs(d, exist_ok=True)

    # ── Pre-flight disk-space check ────────────────────────────────────────
    # A 200k-localisation run produces ~50 MB of CSVs, plus PDFs / PNGs /
    # the optional ROI mask npy.  500 MB free is plenty; below that we
    # WARN; below 100 MB we hard-fail before doing any work so the user
    # doesn't lose 5 minutes of analysis to a save that was always
    # going to fail.  The disk watchdog above continues monitoring
    # during the run for accumulating writes.
    try:
        import shutil
        free_mb = shutil.disk_usage(out_dir).free / 1e6
        if free_mb < 100.0:
            raise RuntimeError(
                f"Only {free_mb:.0f} MB free on the output disk — "
                f"refusing to start (analysis outputs need at least "
                f"~100 MB).  Free up space, or pick a different "
                f"output folder, and re-run.")
        if free_mb < 500.0:
            _log(f"  ⚠ Disk space low: {free_mb:.0f} MB free on "
                 f"output volume.  The run will continue but later "
                 f"saves may fail if more files accumulate.")
    except RuntimeError:
        raise
    except Exception:
        # If shutil.disk_usage isn't supported (rare), proceed and
        # let the watchdog catch problems.
        pass

    # Route disk-backed memmap stacks next to the output dir by default
    # (typically on the user's data drive, which has more headroom than
    # the system /var/folders temp).  An explicit FIREFLY_TEMP_DIR env
    # var still wins.  Skipped for external-CSV mode (no memmap needed).
    try:
        from firefly.sptpalm_analysis import set_temp_stack_dir as _set_tmp
        if not os.environ.get("FIREFLY_TEMP_DIR"):
            _set_tmp(out_dir)
    except Exception: pass

    # ── Source-of-localisations branch ────────────────────────────────────
    # Two modes:
    #   • "image"        — load stack + preprocess + localise (the default
    #                       PALM pipeline; uses GPU when available).
    #   • "external_csv" — skip detection; load a CSV exported from
    #                       PALM-Tracer / ThunderSTORM / Picasso and feed
    #                       its localisations into linking + downstream
    #                       analyses unchanged.
    source = p.get("source", "image")
    external_csv = source == "external_csv"

    import numpy as _np
    if not external_csv:
        # ── Load ──────────────────────────────────────────────────────────
        _log(f"\n── Load ──────────────────────────")
        _prog(5, "Loading stack…")
        stack, meta_px, meta_fi = load_file(
            fpath, channel=int(p.get("channel", 0)),
            stop_event=cancel_event,
            files=p.get("series_files"))
        # Override file-embedded metadata only when the user explicitly
        # ticked the "Override" checkbox.
        px = p.get("pixel_size") or meta_px or 0.106
        fi = p.get("frame_interval") or meta_fi or 0.02
        n_frames = len(stack)
        _log(f"  Shape: {stack.shape}  (T x Y x X)")
        _log(f"  Frames: {n_frames:,}  |  px={px} µm  fi={fi} s")
        # Surface missing acquisition metadata once per file — explains why
        # px/fi fell back to a sidebar override or the built-in default.
        if not meta_px or not meta_fi:
            _missing = " and ".join(
                lbl for lbl, val in (("pixel size", meta_px),
                                     ("frame interval", meta_fi)) if not val)
            _log(f"  NOTE: {_missing} not found in file metadata; "
                 f"using px={px} µm, fi={fi} s — set an override in the "
                 f"sidebar if these are wrong")
        # Sample frames for the figure-background panel.
        n_proj = min(200, n_frames)
        proj_idx = _np.linspace(0, n_frames - 1, n_proj, dtype=int)
        proj_sample = stack[proj_idx].copy()
    else:
        # ── External CSV path ────────────────────────────────────────────
        _log(f"\n── Load (external CSV) ───────────")
        _prog(5, "Reading localisations from CSV…")
        # Pixel size and frame interval come from the GUI; we don't try
        # to infer them from the CSV (most tools don't embed them).
        px = float(p.get("pixel_size") or 0.106)
        fi = float(p.get("frame_interval") or 0.02)
        locs_extern = load_external_locs(
            fpath,
            preset=p.get("csv_preset", "auto"),
            pixel_size_um=px,
            column_map=p.get("csv_column_map"),
            frame_offset=p.get("csv_frame_offset"))
        n_frames = int(locs_extern["frame"].max()) + 1
        _log(f"  Frames: {n_frames:,}  |  px={px} µm  fi={fi} s")
        # If the user provided a background image, sample frames from it
        # for the figure's max-projection panel.  Otherwise hand the
        # downstream code a blank canvas — make_figure handles that case.
        bg_path = p.get("bg_image_path") or ""
        if bg_path and os.path.isfile(bg_path):
            try:
                _log(f"  Loading background image: "
                     f"{os.path.basename(bg_path)}")
                bg_stack, _, _ = load_file(
                    bg_path, channel=int(p.get("channel", 0)),
                    stop_event=cancel_event)
                n_bg = len(bg_stack)
                n_proj = min(200, n_bg)
                proj_idx = _np.linspace(0, n_bg - 1, n_proj, dtype=int)
                proj_sample = bg_stack[proj_idx].copy()
                del bg_stack
            except Exception as exc:
                _log(f"  WARN: background image failed to load — "
                     f"figure projection panel will be blank.  ({exc})")
                proj_sample = _make_loc_histogram_proj(locs_extern, p)
        else:
            proj_sample = _make_loc_histogram_proj(locs_extern, p)

    # ── Localisation ──────────────────────────────────────────────────────
    _log(f"\n── Localisation ──────────────────")
    if external_csv:
        _prog(45, "Localisations loaded from CSV")
    else:
        _prog(20, "Localising…")

    # Detection threshold (minmass).  When "auto" is on, compute a robust
    # per-file value up front from the candidate spot-mass distribution (GMM
    # noise/signal valley + knee cross-check) so each .czi/.tif gets its own
    # threshold; otherwise use the manual value.  CSV inputs have no image to
    # threshold.
    mm_diag = None
    if external_csv:
        minmass_arg = None
    elif p.get("auto_minmass", False):
        from firefly.analysis.fa_localize import estimate_minmass
        _mftr = p.get("minmass_max_false_track_rate")
        try:
            _mftr = float(_mftr) if _mftr not in (None, "", 0, 0.0) else None
        except (TypeError, ValueError):
            _mftr = None
        minmass_arg, mm_diag = estimate_minmass(
            stack,
            diameter=int(p["diameter"]),
            percentile=64,
            backend=p["backend"],
            sensitivity=p.get("minmass_sensitivity", "balanced"),
            bg_radius=int(p.get("bg_radius", 10)),
            bg_method=p.get("bg_method", "uniform_filter"),
            workers=int(p["workers"]),
            log_cb=_log,
            search_range=int(p.get("search_range", 5)),
            memory=int(p.get("memory", 3)),
            link_min_len=max(4, int(p.get("min_track_len", 4) or 4)),
            max_false_track_rate=_mftr)
    else:
        minmass_arg = float(p["minmass"])

    # Real-time mass histogram: each chunk's mass values get pushed into
    # the GUI via the msg queue so the user can spot a bad minmass early.
    # Uses put_nowait so a stalled GUI can never block the worker on the
    # IPC queue (which would lock the analysis up).
    def _mass_cb(masses):
        try:
            import queue as _q
            arr = masses if len(masses) <= 20000 else masses[:20000]
            try:    msg_queue.put_nowait(("mass_chunk", arr.tolist()))
            except _q.Full: pass     # drop — non-essential
        except Exception:
            pass

    # Live detection view: emit ~60 frames/s of (preprocessed frame +
    # detected spots) to the GUI.  The main worker fires `_preview_cb`
    # at whatever rate the pipeline produces frames (potentially many
    # hundreds per second in a burst after each chunk's locate); a
    # dedicated background thread pulls from a small internal queue
    # and forwards to `msg_queue` paced at 60 Hz.  This decouples the
    # main loop's speed from the GUI's frame budget — analysis stays
    # fast, but the GUI only sees one frame every 16 ms (≈ 60 FPS).
    import queue as _queue
    import threading as _threading
    _preview_internal_q: "_queue.Queue" = _queue.Queue(maxsize=240)
    _preview_stop = _threading.Event()

    # Memory-pressure brake.  Hot loops with a per-frame preview can
    # push the system into swap on tight-RAM laptops; if free memory
    # drops below this floor we silently stop emitting preview frames
    # (analysis itself keeps running — only the cosmetic stream pauses).
    try:    import psutil as _ps
    except Exception: _ps = None

    # Scale the brake floor with total RAM so it doesn't fire
    # spuriously on bigger machines.  Previously a flat 2.5 GB,
    # which was 16% of a 16 GB laptop (fine) but only 8% of a 32 GB
    # box (way too conservative).  On a 32 GB Windows machine
    # running torch-cpu, `available` routinely dips to ~1.87 GB
    # while PyTorch holds its caching allocator full of intermediates;
    # the old brake then froze the live preview at whatever frame
    # came in just before, even though the box had >25 GB usable.
    #
    # Formula: 5 % of total RAM, clamped to [0.75, 2.5] GB.  The
    # upper cap keeps the brake tight on tiny VMs; the lower cap
    # keeps it from disabling entirely on hosts that misreport.
    # Override with FIREFLY_PREVIEW_MEM_FLOOR_GB if needed.
    try:
        _total_gb_pv = _ps.virtual_memory().total / 1e9 if _ps else 16.0
    except Exception:
        _total_gb_pv = 16.0
    _override = os.environ.get("FIREFLY_PREVIEW_MEM_FLOOR_GB")
    if _override:
        try:    _MEM_FLOOR_GB = float(_override)
        except ValueError: _MEM_FLOOR_GB = max(0.75, min(2.5, _total_gb_pv * 0.05))
    else:
        _MEM_FLOOR_GB = max(0.75, min(2.5, _total_gb_pv * 0.05))

    # One-shot log so a "live view froze" report never goes silent
    # again — the user (and we) can grep the log to see whether the
    # brake fired AND at what free-RAM value.  Toggles back to "armed"
    # if memory recovers, so a second event also produces a log.
    _preview_brake_state = {"engaged": False}

    def _system_under_pressure() -> bool:
        if _ps is None:
            return False
        try:
            avail = _ps.virtual_memory().available / 1e9
        except Exception:
            return False
        engaged = avail < _MEM_FLOOR_GB
        if engaged and not _preview_brake_state["engaged"]:
            try:
                msg_queue.put_nowait((
                    "log",
                    f"\n  ⚠ Preview brake ON: only {avail:.2f} GB RAM "
                    f"available (floor {_MEM_FLOOR_GB:.2f} GB).  Live "
                    f"detection view will pause until memory frees up — "
                    f"analysis continues normally."))
            except Exception:
                pass
            _preview_brake_state["engaged"] = True
        elif (not engaged) and _preview_brake_state["engaged"]:
            try:
                msg_queue.put_nowait((
                    "log",
                    f"  ✓ Preview brake OFF: {avail:.2f} GB available "
                    f"(floor {_MEM_FLOOR_GB:.2f} GB).  Live detection "
                    f"view resuming."))
            except Exception:
                pass
            _preview_brake_state["engaged"] = False
        return engaged

    def _preview_pump():
        period = 1.0 / 60.0
        while not _preview_stop.is_set():
            try:
                payload = _preview_internal_q.get(timeout=0.1)
            except _queue.Empty:
                continue
            # Skip the emit if RAM is critically low.  Better to drop
            # the visual than to push the host into swap and freeze.
            if _system_under_pressure():
                time.sleep(period)
                continue
            try:    msg_queue.put_nowait(("preview_frame", payload))
            except _queue.Full: pass    # IPC queue saturated; drop
            except Exception:   pass
            time.sleep(period)

    _preview_thread = _threading.Thread(target=_preview_pump, daemon=True)
    _preview_thread.start()

    def _preview_cb(frame_idx, frame, xs, ys, n_frames):
        # Same brake as the pump — if memory is tight, don't even allocate
        # the downsampled frame.
        if _system_under_pressure():
            return
        try:
            import numpy as _np
            f = _np.asarray(frame, dtype=_np.float32)
            # Downsample anything larger than 384 px on the long edge
            scale_y = scale_x = 1.0
            max_side = 384
            if f.shape[0] > max_side or f.shape[1] > max_side:
                step_y = max(1, f.shape[0] // max_side)
                step_x = max(1, f.shape[1] // max_side)
                f = f[::step_y, ::step_x]
                scale_y, scale_x = 1.0 / step_y, 1.0 / step_x
            xs_a = _np.asarray(xs, dtype=_np.float32) * scale_x
            ys_a = _np.asarray(ys, dtype=_np.float32) * scale_y
            payload = {
                "idx":      int(frame_idx),
                "n_frames": int(n_frames),
                "shape":    [int(f.shape[0]), int(f.shape[1])],
                "frame":    f.tobytes(),
                "xs":       xs_a.tolist(),
                "ys":       ys_a.tolist(),
            }
            # Non-blocking insert with drop-oldest when full so the worker
            # never has to wait on the GUI.
            if _preview_internal_q.full():
                try:    _preview_internal_q.get_nowait()
                except _queue.Empty: pass
            try:    _preview_internal_q.put_nowait(payload)
            except _queue.Full: pass
        except Exception:
            pass

    # Default projections to None so the ROI block can detect the
    # external_csv path (where we have no stack to project) and bail
    # out cleanly with a warning instead of NameError.
    mean_proj  = None
    max_proj   = None
    blink_proj = None
    if not external_csv:
        try:
            locs, mean_proj, max_proj, blink_proj, _mm = preprocess_and_localise_adaptive(
                stack,
                diameter=int(p["diameter"]),
                minmass=minmass_arg,
                bg_radius=int(p.get("bg_radius", 10)),
                bg_method=p.get("bg_method", "uniform_filter"),
                workers=int(p["workers"]),
                chunk_size=int(p["chunk_size"]),
                stop_event=cancel_event,
                mass_cb=_mass_cb,
                preview_cb=_preview_cb,
                backend=p["backend"])
        finally:
            # Stop the preview pump and let it drain whatever's left
            _preview_stop.set()
            try:    _preview_thread.join(timeout=1.0)
            except Exception: pass
        # Fast-path users get a single bulk emit (no per-chunk hook there).
        try:
            if locs is not None and len(locs) > 0 and "mass" in locs.columns:
                _mass_cb(locs["mass"].values.astype("float32"))
        except Exception:
            pass
        stack_h = stack.shape[1] if stack.ndim >= 3 else 0
        stack_w = stack.shape[2] if stack.ndim >= 3 else 0
    else:
        # External CSV path: locs already loaded, no preview pump.
        locs = locs_extern
        stack_h = stack_w = 0
        if proj_sample is not None and proj_sample.ndim >= 3:
            stack_h = int(proj_sample.shape[1])
            stack_w = int(proj_sample.shape[2])
        # Stop the (idle) preview pump cleanly so it doesn't linger.
        _preview_stop.set()
        try:    _preview_thread.join(timeout=1.0)
        except Exception: pass
        # Emit one mass-histogram update so the live histogram isn't empty
        try:
            if "mass" in locs.columns and len(locs) > 0:
                _mass_cb(locs["mass"].values.astype("float32"))
        except Exception:
            pass
    if not external_csv:
        del stack
    _log(f"  → {len(locs):,} localisations")
    _check_stop()

    # ── ROI mask (optional) ───────────────────────────────────────────────
    # Per-file polygon overrides the global mode: if a polygon was set
    # for this file in the Import-tab ROI editor, treat it as polygon-mode
    # regardless of what the sidebar says.
    roi_mode_user = p.get("roi_mode", "none")   # what the user picked
    roi_mode = roi_mode_user
    if p.get("roi_polygon"):
        roi_mode = "polygon"

    # "ImageJ ROI" mode: pair a sibling ImageJ ROI (RoiSet.zip / a RoiSet/
    # folder / *.roi) found next to the movie and use it as a polygon ROI — so
    # a batch reuses ROIs drawn in ImageJ/Fiji without loading each by hand.
    # If none is found, fall back to the whole image (logged).  Skipped for
    # external-CSV inputs and when a polygon was already set for this file.
    # A local copy of `p` is used so the polygon never leaks to the next file.
    if (roi_mode == "imagej" and not external_csv and not p.get("roi_polygon")):
        _found = False
        try:
            from firefly.analysis import fa_roi as _far
            _roi_path = _far.find_sibling_imagej_roi(
                os.path.dirname(fpath),
                os.path.splitext(os.path.basename(fpath))[0])
            if _roi_path:
                _polys = _far.load_roi_polygons_any(_roi_path)
                if _polys:
                    p = dict(p)
                    p["roi_polygon"] = [poly.tolist() for poly in _polys]
                    roi_mode = "polygon"
                    _found = True
                    _log(f"  NOTE: ImageJ ROI '{os.path.basename(_roi_path)}' "
                         f"({len(_polys)} region(s)) — using as polygon ROI.")
        except Exception as _exc:
            _log(f"  WARN: ImageJ ROI load failed: {_exc}")
        if not _found:
            _log("  NOTE: ROI mode 'ImageJ ROI' but no sibling RoiSet/.roi "
                 "found — analysing the whole image.")
            roi_mode = "none"

    # Auto-detect a microscope-exported sister ROI image (e.g.
    # `<base>_green.tif`).  When `roi_mode == "auto_sister"` we ONLY use
    # the sister TIFF, falling back to no-ROI if missing.  When
    # `roi_mode` is something else but `roi_sister_autodetect` is on AND
    # the dropdown is in "none"/"auto"/"manual" mode, we'll still check
    # for a sister file and prefer it over the intensity-based modes.
    roi_sister_suffix = str(p.get("roi_sister_suffix", "_green")).strip()
    roi_sister_path: "str | None" = None
    if not external_csv and roi_sister_suffix:
        try:
            _base = os.path.splitext(os.path.basename(fpath))[0]
            # palmTRACER series: strip `-fileNNN` so the suffix sits
            # against the bare root name (`<root>_green.tif`).
            import re as _re
            _root = _re.sub(r"-file\d+$", "", _base, flags=_re.IGNORECASE)
            for _ext in (".tif", ".tiff"):
                _cand = os.path.join(os.path.dirname(fpath),
                                      f"{_root}{roi_sister_suffix}{_ext}")
                if os.path.isfile(_cand):
                    roi_sister_path = _cand
                    break
        except Exception:
            roi_sister_path = None
    # Promote to active mode if user explicitly picked "sister" OR if
    # auto-detect is on and we found a file.
    if roi_mode == "sister":
        if roi_sister_path is None:
            _log(f"  NOTE: ROI mode set to 'Sister TIFF' but no "
                 f"`<base>{roi_sister_suffix}.tif` found — falling back "
                 f"to no ROI.")
            roi_mode = "none"
    elif (roi_mode_user in ("none", "auto", "manual")
            and roi_mode in ("none", "auto", "manual")
            and bool(p.get("roi_sister_autodetect", True))
            and roi_sister_path is not None):
        _log(f"  NOTE: found sister ROI image "
             f"{os.path.basename(roi_sister_path)} — using it instead "
             f"of intensity-based ROI.  Set roi_sister_autodetect=False "
             f"to disable.")
        roi_mode = "sister"

    if roi_mode != "none" and len(locs) > 0:
        _log(f"\n── ROI mask ───────────────────────")
        try:
            from firefly.sptpalm_analysis import build_roi_mask_advanced
            roi_mask = None
            # These are populated ONLY by the threshold-projection branch
            # below; the sister-TIFF and polygon paths leave them unset.
            # Initialise here so the shared roi_mask.png save block can't hit
            # an UnboundLocalError (it previously assumed only polygon or
            # threshold mode existed).
            mode_hint = None
            bg_sigma  = None
            info      = None

            # ── Sister TIFF ROI (microscope export, e.g. _green.tif) ─
            if roi_mode == "sister" and roi_sister_path is not None:
                try:
                    import tifffile as _tf
                    with _tf.TiffFile(roi_sister_path) as _t:
                        _arr = _t.asarray()
                    # Multi-frame → max projection so static ROI
                    # outlines come through regardless of which frame
                    # the microscope saved them on.
                    if _arr.ndim == 3:
                        _arr = _arr.max(axis=0)
                    elif _arr.ndim > 3:
                        # Squeeze leading singleton dims, then max-project.
                        _arr = _np.squeeze(_arr)
                        if _arr.ndim == 3:
                            _arr = _arr.max(axis=0)
                    # Resize / check shape matches the analysis stack.
                    if mean_proj is not None and _arr.shape != mean_proj.shape:
                        _log(f"  WARN: sister ROI image shape "
                             f"{_arr.shape} ≠ stack shape "
                             f"{mean_proj.shape} — skipping ROI.")
                    else:
                        # Robust mask logic: if the image is already
                        # mostly zeros (i.e. a binary/labelled
                        # segmentation), treat non-zero as inside.
                        # Otherwise it's a grayscale fluorescence
                        # channel — auto-threshold with Li.
                        _nonzero_frac = float(
                            (_arr > 0).sum()) / float(_arr.size or 1)
                        if _nonzero_frac < 0.4:
                            roi_mask = _arr > 0
                            _log(f"  Sister ROI "
                                 f"({os.path.basename(roi_sister_path)}): "
                                 f"non-zero pixel mask, "
                                 f"{100.0 * roi_mask.mean():.1f}% of frame")
                        else:
                            # Grayscale — normalise + Li threshold via
                            # the existing pipeline so we benefit from
                            # its bg-subtraction and morphology.
                            try:
                                _arrf = _arr.astype(_np.float32)
                                _mn, _mx = float(_arrf.min()), float(_arrf.max())
                                if _mx > _mn:
                                    _arrf = (_arrf - _mn) / (_mx - _mn)
                                roi_mask, _info = build_roi_mask_advanced(
                                    _arrf,
                                    threshold=None,
                                    threshold_method="li",
                                    bg_sigma=float(p.get("roi_bg_sigma", 25.0)),
                                    mode_hint="mean")
                                _log(f"  Sister ROI "
                                     f"({os.path.basename(roi_sister_path)}): "
                                     f"Li-threshold mask, "
                                     f"{100.0 * roi_mask.mean():.1f}% of frame")
                            except Exception as _exc:
                                _log(f"  WARN: sister ROI Li threshold "
                                     f"failed — {_exc}.  Falling back to "
                                     f"non-zero mask.")
                                roi_mask = _arr > 0
                except Exception as _exc:
                    _log(f"  WARN: could not load sister ROI image "
                         f"{roi_sister_path}: {_exc}.  Continuing "
                         f"without ROI.")
                    roi_mask = None

            if roi_mode == "polygon":
                # User-drawn polygon ROI.  `roi_polygon` is a list of
                # (y, x) vertex pairs in pixel coordinates of the
                # original frame (Y by X).  skimage's polygon2mask
                # rasterises it into a boolean array of the same shape.
                vertices = p.get("roi_polygon") or []
                if not vertices:
                    _log("  WARN: roi_mode is 'polygon' but no vertices "
                         "were provided.  Skipping ROI.")
                elif mean_proj is None:
                    _log("  WARN: roi_mode is 'polygon' but no stack was "
                         "loaded (external CSV?).  Skipping ROI.")
                else:
                    try:
                        from skimage.draw import polygon2mask
                        polys = vertices if isinstance(vertices[0][0],
                                                       (list, tuple)) \
                                          else [vertices]
                        h, w = mean_proj.shape
                        # NB: keep this `bool`, not `uint8` — see
                        # apply_roi_mask for why uint8 breaks pandas
                        # row indexing.
                        roi_mask = _np.zeros((h, w), dtype=bool)
                        for poly in polys:
                            m = polygon2mask((h, w), _np.asarray(poly))
                            roi_mask |= m.astype(bool)
                        n_polys = len(polys)
                        _log(f"  User polygon ROI: {n_polys} shape(s), "
                             f"{100.0 * roi_mask.mean():.1f}% of frame")
                    except Exception as poly_exc:
                        _log(f"  WARN: polygon ROI failed — {poly_exc}.")
                        roi_mask = None

            if roi_mask is None and mean_proj is not None:
                # Shared GUI/worker ROI pipeline — DoG background
                # subtraction + morphology + top-N components.
                # Whatever the user tunes in the ROI preview viewer is
                # what gets applied here, byte-for-byte identical.
                mask_mode = str(p.get("roi_mask_mode", "Max")).strip().lower()
                if mask_mode.startswith("blink"):
                    # Streaming Welford-based per-pixel mean+std baseline
                    # gives us a real blink-density map in a single
                    # pass over the stack (see preprocess_and_localise_
                    # adaptive in sptpalm_analysis.py).  Approximation
                    # only in that the threshold per pixel evolves as
                    # the estimate stabilises — for 4000+ frame movies
                    # the early-chunk transient is negligible.
                    if blink_proj is not None:
                        proj = blink_proj
                        mode_hint = "blink"
                    else:
                        _log("  NOTE: Blink-density not available "
                             "(no stack loaded?) — falling back to Max.")
                        proj = max_proj
                        mode_hint = "max"
                elif mask_mode.startswith("max"):
                    proj = max_proj
                    mode_hint = "max"
                elif mask_mode.startswith("sum"):
                    # Sum is just mean × frame_count up to a scale that
                    # normalisation removes, so route it to mean_proj.
                    proj = mean_proj
                    mode_hint = "sum"
                else:  # "mean" (default for legacy presets)
                    proj = mean_proj
                    mode_hint = "mean"

                if proj is None:
                    proj = mean_proj  # final safety net

                if roi_mode == "auto":
                    method = (p.get("roi_auto_method") or "Li").lower()
                    manual_thresh = None
                else:  # manual threshold
                    method = "li"
                    manual_thresh = float(p.get("roi_threshold", 0.08))
                bg_sigma = float(p.get("roi_bg_sigma", 25.0))
                roi_mask, info = build_roi_mask_advanced(
                    proj,
                    threshold=manual_thresh,
                    threshold_method=method,
                    bg_sigma=bg_sigma,
                    mode_hint=mode_hint)
                _log(f"  Projection={mode_hint}, σ_bg={bg_sigma:.1f}, "
                     f"threshold={info['threshold']:.4f}  |  "
                     f"{100.0 * info['fraction']:.1f}% of frame")

            if roi_mask is None:
                _log("  WARN: could not build a ROI mask.  "
                     "Continuing without ROI.")
            else:
                n_before = len(locs)
                locs = apply_roi_mask(locs, roi_mask)
                _log(f"  Locs after ROI : {len(locs):,}  "
                     f"(dropped {n_before - len(locs):,})")

                # ── Persist the exact mask that was applied ──────────
                # Two artefacts:
                #   * {stem}_roi_mask.npy   — raw bool array, for any
                #     downstream re-analysis that wants pixel-perfect
                #     reproducibility.
                #   * {stem}_roi_mask.png   — overlay of projection +
                #     mask outline + translucent fill, so a human
                #     looking at the run folder can see at a glance
                #     which pixels were kept and which excluded.
                # Both live in firefly_extras/ (raw) and figures/ (PNG)
                # to match the existing output-folder convention.
                try:
                    os.makedirs(extras_dir, exist_ok=True)
                    _np.save(
                        os.path.join(extras_dir, f"{stem}_roi_mask.npy"),
                        roi_mask.astype(bool, copy=False))
                except Exception as save_exc:
                    _log(f"  NOTE: could not save roi_mask.npy "
                         f"({save_exc}).")

                # Pick a projection background for the PNG — prefer the
                # same one used to BUILD the mask, falling back through
                # max → mean.  If we get here, at least one is non-None.
                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as _plt
                    if mode_hint == "blink" and blink_proj is not None:
                        bg = blink_proj
                        bg_label = "Blink density"
                    elif mode_hint == "max" and max_proj is not None:
                        bg = max_proj
                        bg_label = "Max projection"
                    elif mean_proj is not None:
                        bg = mean_proj
                        bg_label = "Mean projection"
                    elif max_proj is not None:
                        bg = max_proj
                        bg_label = "Max projection"
                    else:
                        bg = None
                        bg_label = ""

                    if bg is not None:
                        os.makedirs(fig_dir, exist_ok=True)
                        fig, ax = _plt.subplots(figsize=(6, 6),
                                                facecolor="#0d1117")
                        ax.set_facecolor("#0d1117")
                        lo, hi = _np.percentile(bg, [1.0, 99.5])
                        if hi <= lo:
                            hi = lo + 1.0
                        ax.imshow(bg, cmap="inferno",
                                  vmin=float(lo), vmax=float(hi),
                                  interpolation="nearest")
                        # Translucent green fill + sharper green outline
                        ax.imshow(
                            _np.ma.masked_where(~roi_mask, roi_mask),
                            cmap="Greens", alpha=0.30,
                            interpolation="nearest")
                        ax.contour(roi_mask.astype(float), levels=[0.5],
                                   colors=["#39ff14"], linewidths=1.4)
                        if roi_mode == "polygon":
                            title = (
                                f"ROI applied — {bg_label}  |  "
                                f"polygon, "
                                f"{100.0 * float(roi_mask.mean()):.1f}% "
                                f"of frame"
                            )
                        elif mode_hint is not None and info is not None:
                            # Threshold-projection branch — full provenance.
                            title = (
                                f"ROI applied — {bg_label}  |  "
                                f"projection={mode_hint}, "
                                f"σ_bg={bg_sigma:.0f}, "
                                f"t={info['threshold']:.3f}, "
                                f"{100.0 * info['fraction']:.1f}% of frame"
                            )
                        else:
                            # Sister-TIFF (or any other) ROI: we only know the
                            # kept fraction, not a threshold / σ_bg.
                            title = (
                                f"ROI applied — {bg_label}  |  "
                                f"{100.0 * float(roi_mask.mean()):.1f}% of frame"
                            )
                        ax.set_title(title, color="#e6edf3", fontsize=9)
                        ax.set_xticks([]); ax.set_yticks([])
                        for sp in ax.spines.values():
                            sp.set_edgecolor("#30363d")
                        fig.tight_layout()
                        fig.savefig(
                            os.path.join(fig_dir, f"{stem}_roi_mask.png"),
                            dpi=150, facecolor=fig.get_facecolor())
                        _plt.close(fig)
                except Exception as save_exc:
                    _log(f"  NOTE: could not save roi_mask.png "
                         f"({save_exc}).")
        except Exception as roi_exc:
            import traceback as _tb, sys as _sys
            _tb.print_exc(file=_sys.stderr)
            _log(f"  WARN: ROI mask failed — {roi_exc}.  Continuing without ROI.")

    # ── Drift correction (optional) ───────────────────────────────────────
    drift_df = None
    if p.get("drift_correct", False) and len(locs) > 0:
        _log(f"\n── Drift correction ───────────────")
        _prog(40, "Correcting drift…")
        try:
            locs, drift_df = correct_drift(
                locs, n_seg_frames=int(p.get("drift_segment", 500)))
            drift_df.to_csv(
                os.path.join(extras_dir, f"{stem}_drift.csv"), index=False)
            _log(f"  Drift correction applied  |  saved {stem}_drift.csv")
        except Exception as exc:
            _log(f"  WARN: drift correction failed — {exc}")

    _check_stop()

    def _drain_gpu():
        """Force a GPU cache drain.  Cheap, mostly defensive — calling
        between heavy stages prevents PyTorch's caching allocator from
        sitting on multi-GB allocations long after they're needed,
        which can push tight-RAM laptops over the edge."""
        try:
            import torch as _torch, gc as _gc
            _gc.collect()
            if hasattr(_torch.backends, "mps") and \
                    _torch.backends.mps.is_available():
                if hasattr(_torch.mps, "synchronize"): _torch.mps.synchronize()
                if hasattr(_torch.mps, "empty_cache"):  _torch.mps.empty_cache()
            if _torch.cuda.is_available():
                _torch.cuda.synchronize(); _torch.cuda.empty_cache()
        except Exception:
            pass

    # Belt-and-braces GPU drain before the long CPU-only linking stage.
    _drain_gpu()

    # ── Linking ───────────────────────────────────────────────────────────
    # Fast path: the input already carries a `particle` column (e.g.
    # TrackMate CSV imported via load_external_locs with the TRACK_ID
    # mapping turned on).  In that case there's nothing to link —
    # we just rename the existing IDs to dense 0..N-1 integers and
    # honour the min/max track-length filter.  Everything downstream
    # then runs against TrackMate's tracks instead of FIREFLY's
    # re-linked tracks, which is the supported way to get
    # "TrackMate detection + linking, FIREFLY analytics".
    if "particle" in locs.columns:
        _log(f"\n── Linking (skipped — pre-linked input) ─────────")
        _log(f"  Input CSV already has TRACK_ID — using upstream "
             f"linker's tracks directly.")
        _prog(50, "Using pre-linked tracks from CSV…")
        import pandas as _pd
        # Densify particle IDs to 0..N-1 so downstream code that
        # assumes integer-indexable particles (some clustering /
        # MSD paths) doesn't choke on TrackMate's wide ID space.
        _id_map = {old: new for new, old in enumerate(
            sorted(locs["particle"].unique()))}
        tracks = locs.copy()
        tracks["particle"] = tracks["particle"].map(_id_map).astype("int64")
        # Apply min/max track-length filter the same way the real
        # linker would — keeps stats comparable across paths.
        _min_len = int(p.get("min_track_len", 0) or 0)
        _max_len = p.get("max_track_len") or 0
        try:    _max_len = int(_max_len)
        except Exception: _max_len = 0
        if _min_len > 0 or _max_len > 0:
            _counts = tracks.groupby("particle").size()
            _keep = _counts.index[(_counts >= max(1, _min_len)) &
                                   ((_counts <= _max_len)
                                    if _max_len > 0 else True)]
            n_before = int(tracks["particle"].nunique())
            tracks = tracks[tracks["particle"].isin(_keep)].reset_index(
                drop=True)
            n_after = int(tracks["particle"].nunique())
            _log(f"  Track-length filter ({_min_len}–"
                 f"{'∞' if _max_len == 0 else _max_len} frames): "
                 f"{n_after:,} / {n_before:,} tracks kept")
    else:
        _log(f"\n── Linking ───────────────────────")
        _log(f"  Linking {len(locs):,} localisations — single-threaded, "
             f"may take several minutes at high density")
        if len(locs) > 100_000:
            _log(f"  NOTE: very high spot density ({len(locs):,} locs). "
                 f"Consider raising minmass to reduce false positives.")
        _prog(50, f"Linking {len(locs):,} localisations…")

        # Map linker [0, 1] progress onto the overall progress bar's 50–65%
        # range so the user sees genuine per-frame motion instead of a
        # multi-minute black box.
        def _link_progress(frac: float):
            try:    pct = 50 + int(frac * 15)
            except Exception: pct = 50
            _prog(pct, f"Linking… {frac*100:.0f} %")

        tracks = link_trajectories(
            locs,
            search_range=int(p["search_range"]),
            memory=int(p["memory"]),
            min_len=int(p["min_track_len"]),
            max_len=p.get("max_track_len"),
            linker=str(p.get("linker", "trackpy")),
            progress_cb=_link_progress,
            stop_event=cancel_event)
    n_tracks_found = tracks['particle'].nunique() if len(tracks) else 0
    _log(f"  → {n_tracks_found:,} trajectories")
    _check_stop()
    # Drain again before the MSD / figure stages — linking can leave
    # large temporaries behind that the next stage doesn't need.
    _drain_gpu()

    if n_tracks_found == 0:
        _log("")
        _log("  ⚠  No trajectories were formed.  Likely causes:")
        _log("     • minmass is too LOW → too many noise spots, "
             "linker can't form sensible tracks")
        _log("     • minmass is too HIGH → real spots filtered out, "
             "nothing left to link")
        _log("     • search_range too small for actual particle motion")
        _log("     • If using a GPU backend and only chunk 1 produced "
             "spots, MPS may be in a degraded state on this hw/os "
             "combo — retry with backend='trackpy' to confirm.")
        _log("")
        _log("── Stopping analysis (nothing more to do) ──")
        # Raise a sentinel — caller will turn this into a sensible payload.
        raise _NoTracks({
            "stem": stem, "out_dir": out_dir,
            "figure_path": "", "n_tracks": 0, "n_locs": int(len(locs)),
        })

    # ── MSD + diffusion ───────────────────────────────────────────────────
    _log(f"\n── MSD & diffusion ───────────────")
    _prog(65, "Computing MSD + fits…")
    imsd_df, emsd_df, diff_df = compute_msd_and_fit(
        tracks, px, fi,
        max_lagtime=int(p["max_lagtime"]),
        n_fit=int(p["n_fit"]),
        workers=int(p["workers"]),
        alpha_thresholds=tuple(p.get("alpha_thresholds", (0.5, 0.9, 1.1))))

    # Optional: filter tracks by diffusion coefficient
    if p.get("filter_d_enabled", False) and len(diff_df):
        d_min = float(p.get("filter_d_min", 0.0))
        d_max = float(p.get("filter_d_max", 1.0))
        n_before = len(diff_df)
        mask = diff_df["D"].between(d_min, d_max)
        keep_pids = set(diff_df.loc[mask, "particle"])
        diff_df = diff_df[mask].reset_index(drop=True)
        tracks  = tracks[tracks["particle"].isin(keep_pids)]
        _log(f"  Filter by D [{d_min}, {d_max}]: "
             f"{n_before} → {len(diff_df)} tracks")

    # ── Secondary analyses ────────────────────────────────────────────────
    _log(f"\n── Secondary analyses ────────────")
    _prog(80, "Secondary analyses…")
    jdd = compute_jdd(tracks, px, fi,
                      n_components=int(p.get("jdd_components", 2)))
    ta  = compute_turning_angles(tracks)
    try:
        van_hove = compute_van_hove(tracks, px, lag_frames=1)
    except Exception:
        van_hove = None
    try:
        vacf = compute_vacf(tracks, fi, px, max_lag=10)
    except Exception:
        vacf = None
    mf  = compute_mobile_fraction_over_time(
        tracks, diff_df, fi,
        d_threshold=float(p.get("mobile_d_threshold", 0.05)))
    cluster_labels, cluster_stats_df, _, cluster_xy = compute_clusters(
        locs, px,
        eps_um=float(p.get("cluster_eps_nm", 50.0)) / 1000.0,
        min_samples=int(p.get("cluster_min_samples", 10)))
    dwell_df, dwell_tau = compute_dwell_times(tracks, diff_df, fi)
    # MSS slope per track — merged into diff_df so the figure's MSS
    # panel and downstream CSVs see it.  Skipped silently when there
    # are no tracks long enough (compute_mss returns an empty frame).
    try:
        mss_df = compute_mss(tracks, px, fi)
        if mss_df is not None and len(mss_df) > 0:
            diff_df = diff_df.merge(mss_df, on="particle", how="left")
            _log(f"  MSS slopes computed for {len(mss_df):,} tracks")
        else:
            _log(f"  MSS: no tracks long enough — panel N will be empty")
    except Exception as exc:
        _log(f"  WARN: MSS computation failed: {exc}")
    _check_stop()

    # ── Render figure ─────────────────────────────────────────────────────
    _log(f"\n── Saving ────────────────────────")
    _prog(90, "Rendering figure…")
    fig_theme    = p.get("fig_theme", "Dark")
    fig_proj_cmap = p.get("fig_proj_cmap", "Inferno")
    want_pdf     = bool(p.get("fig_save_pdf", False))
    # Per-panel PNG rendering is the dominant figure-save cost (one full
    # figure rasterisation per panel).  Only render the panels that will
    # actually be written below: none unless fig_per_panel is on, and just
    # the selected subset when fig_single_panels narrows it.
    if bool(p.get("fig_per_panel", False)):
        _allowed_panels = p.get("fig_single_panels")
        want_panels = None if _allowed_panels is None else set(_allowed_panels)
    else:
        want_panels = set()
    fig_data = make_figure(
        proj_sample, tracks, imsd_df, emsd_df, diff_df, px, fi,
        fig_theme=fig_theme, proj_cmap=fig_proj_cmap,
        jdd=jdd, turning_angles=ta, mobile_frac_df=mf,
        cluster_labels=cluster_labels, cluster_locs=cluster_xy,
        dwell_df=dwell_df, dwell_tau=dwell_tau,
        van_hove=van_hove, vacf=vacf,
        return_pdf_bytes=want_pdf, want_panels=want_panels)
    del proj_sample

    # ── Save outputs ──────────────────────────────────────────────────────
    _prog(95, "Saving outputs…")
    try:
        save_palmtracer_csvs(data_dir, stem, locs, tracks, diff_df, imsd_df,
                             pixel_size_um=float(px),
                             frame_interval_s=float(fi),
                             width=stack_w, height=stack_h,
                             n_frames=int(n_frames))
        _log("  Saved (data/): PALM-Tracer CSVs")
    except Exception as exc:
        _log(f"  WARN: PALM-Tracer export failed: {exc}\n{traceback.format_exc()}")

    # Core CSV outputs.  Each one is wrapped in its own try/except so a
    # single disk-write failure (commonly ENOSPC mid-batch on a near-
    # full drive) doesn't tear the whole run down at the very last
    # moment.  Without this, a 5-minute analysis that produced 224k
    # localisations got lost because the disk filled up between
    # localising and writing.
    extras_saved = []
    def _safe_to_csv(df_obj, path, label):
        """Atomically write df_obj to `path`; log + clean up on
        ENOSPC, OSError, PermissionError etc.  Returns True on success.

        Writes to `<path>.tmp` first, then `os.replace()` — so a
        partially-written file from a mid-write failure (e.g. disk full)
        never appears at the final path, and downstream loaders can't
        be tricked into accepting a truncated CSV.
        """
        tmp = path + ".tmp"
        try:
            df_obj.to_csv(tmp, index=False)
            os.replace(tmp, path)
            extras_saved.append(label)
            return True
        except OSError as exc:
            _log(f"  WARN: {label} save failed: {exc}")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            return False
        except Exception as exc:
            _log(f"  WARN: {label} save failed: {exc}")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            return False

    _safe_to_csv(locs,
                 os.path.join(extras_dir, f"{stem}_localisations.csv"),
                 "locs")
    _safe_to_csv(tracks,
                 os.path.join(extras_dir, f"{stem}_trajectories.csv"),
                 "trajectories")
    _safe_to_csv(diff_df,
                 os.path.join(extras_dir, f"{stem}_diffusion_summary.csv"),
                 "diffusion summary")

    # ── Additional per-experiment artifacts ────────────────────────────────
    # The Compare tab reads these by name to plot JDD / dwell-time CDF /
    # turning-angle / radial-distribution / mobile-fraction-over-time
    # panels.  Previously they were computed for the single-sample figure
    # but never persisted, so a FIREFLY run dropped silently out of the
    # corresponding Compare-tab panels.
    try:
        # compute_msd_and_fit returns the ensemble curve as a Series
        # indexed by lag frame, NOT a DataFrame.  Compare expects a
        # DataFrame with columns `lag_frame` + `msd_um2` (matching the
        # PALM-Tracer summary loader), so we convert here.
        if emsd_df is not None and len(emsd_df):
            import pandas as _pd
            if isinstance(emsd_df, _pd.Series):
                emsd_out = (emsd_df.to_frame("msd_um2")
                                   .reset_index(names="lag_frame"))
            else:
                emsd_out = emsd_df
            emsd_out.to_csv(
                os.path.join(extras_dir, f"{stem}_ensemble_msd.csv"),
                index=False)
            extras_saved.append("ensemble MSD")
    except Exception as exc:
        _log(f"  WARN: ensemble-MSD save failed: {exc}")
    try:
        if jdd:
            import json as _json
            def _jsonable(x):
                # numpy scalars / arrays / pandas → JSON-friendly
                if hasattr(x, "tolist"):    return x.tolist()
                if isinstance(x, (set,)):   return list(x)
                return x
            payload = {k: _jsonable(v) for k, v in jdd.items()}
            with open(os.path.join(extras_dir,
                                    f"{stem}_jdd.json"), "w") as _fp:
                _json.dump(payload, _fp, indent=2, default=str)
            extras_saved.append("JDD")
    except Exception as exc:
        _log(f"  WARN: JDD save failed: {exc}")
    try:
        if van_hove is not None:
            import json as _json
            # Compact payload: the histogram + scalars for plotting, but NOT
            # the (potentially huge) raw displacement arrays.
            vh = {k: v for k, v in van_hove.items()
                  if k not in ("displacements_um", "dx_um", "dy_um")}
            vh = {k: (v.tolist() if hasattr(v, "tolist") else v)
                  for k, v in vh.items()}
            with open(os.path.join(extras_dir,
                                    f"{stem}_van_hove.json"), "w") as _fp:
                _json.dump(vh, _fp, indent=2, default=str)
            extras_saved.append(
                f"van Hove (alpha2={van_hove['non_gaussian_alpha2']:.3f})")
    except Exception as exc:
        _log(f"  WARN: van Hove save failed: {exc}")

    try:
        if vacf is not None:
            import json as _json
            vc = {k: (v.tolist() if hasattr(v, "tolist") else v)
                  for k, v in vacf.items()}
            with open(os.path.join(extras_dir,
                                    f"{stem}_vacf.json"), "w") as _fp:
                _json.dump(vc, _fp, indent=2, default=str)
            extras_saved.append(
                f"VACF (persistence={vacf['persistence']:.3f})")
    except Exception as exc:
        _log(f"  WARN: VACF save failed: {exc}")
    try:
        if dwell_df is not None and len(dwell_df):
            dwell_df.to_csv(
                os.path.join(extras_dir, f"{stem}_dwell_times.csv"),
                index=False)
            extras_saved.append("dwell times")
    except Exception as exc:
        _log(f"  WARN: dwell-times save failed: {exc}")
    try:
        if ta is not None and len(ta) > 0:
            import pandas as _pd
            _pd.DataFrame({"turning_angle_deg": ta}).to_csv(
                os.path.join(extras_dir, f"{stem}_turning_angles.csv"),
                index=False)
            extras_saved.append("turning angles")
            # Circular-statistics report — same keys as the MATLAB
            # CircStat toolbox a supervisor will already know how to
            # read.  Two columns: `statistic, value`.  All angle
            # statistics in degrees; rates/dispersions in their natural
            # units (R̄ is dimensionless on [0,1], κ is dimensionless,
            # etc.).  Saved alongside the raw turning-angles CSV so
            # downstream code can join them on filename stem.
            try:
                cs = compute_circular_statistics(ta)

                # Circular-linear correlation: per-track mean turning
                # angle vs that track's diffusion coefficient D.
                # Computed only when we have both tracks (with ≥ 3
                # frames each) AND a diff_df with a D column.  Tells
                # the user "do tracks with stronger turning bias also
                # have different diffusion behaviour?".
                circ_lin = None
                try:
                    if (tracks is not None and len(tracks) >= 3
                            and diff_df is not None
                            and "D" in diff_df.columns):
                        pairs = compute_per_track_mean_angle(tracks)
                        if pairs:
                            d_map = dict(zip(
                                diff_df["particle"].astype(int),
                                diff_df["D"].astype(float)))
                            ang_list, d_list = [], []
                            for pid, mu_deg in pairs:
                                d_val = d_map.get(int(pid))
                                if d_val is None or not _np.isfinite(d_val):
                                    continue
                                ang_list.append(float(mu_deg))
                                d_list.append(float(d_val))
                            if len(ang_list) >= 3:
                                circ_lin = _circ_lin_correlation(
                                    ang_list, d_list)
                except Exception as cl_exc:
                    _log(f"  NOTE: circ-lin correlation skipped "
                         f"({cl_exc}).")
                    circ_lin = None

                # Build the CSV: base stats followed by circ-lin rows
                # (if available).  The CSV stays a 2-column file —
                # supervisors can search for "circ_lin_" rows.
                cs_items = list(cs.items())
                if circ_lin is not None:
                    cs_items.extend([
                        ("circ_lin_angle_vs_D_r",
                         circ_lin.get("r")),
                        ("circ_lin_angle_vs_D_r_squared",
                         circ_lin.get("r2")),
                        ("circ_lin_angle_vs_D_chi2",
                         circ_lin.get("test_stat")),
                        ("circ_lin_angle_vs_D_df",
                         circ_lin.get("df")),
                        ("circ_lin_angle_vs_D_p",
                         circ_lin.get("p")),
                        ("circ_lin_angle_vs_D_n",
                         circ_lin.get("n")),
                    ])
                cs_df = _pd.DataFrame(
                    cs_items, columns=["statistic", "value"])
                cs_df.to_csv(
                    os.path.join(extras_dir,
                                 f"{stem}_circular_statistics.csv"),
                    index=False)
                extras_saved.append("circular statistics")
                # Echo the key numbers into the run log so they show up
                # in the live console without the user having to crack
                # open the CSV.
                _log(
                    "  Circular stats : "
                    f"n={cs['n']:,}  μ={cs['mean_direction_deg']:.2f}°  "
                    f"R̄={cs['mean_resultant_length']:.3f}  "
                    f"κ={cs['concentration_kappa']:.2f}  "
                    f"Rayleigh p={cs['rayleigh_p']:.3g}")
                if circ_lin is not None:
                    _log(
                        "  Circ-lin r (angle vs D): "
                        f"r={circ_lin['r']:.4f}  "
                        f"χ²({circ_lin['df']})={circ_lin['test_stat']:.3g}  "
                        f"p={circ_lin['p']:.3g}  "
                        f"n={circ_lin['n']:,} tracks")
                # Supervisor-facing PDF: single A4 page with the polar
                # histogram, plain-English interpretation, and the
                # CircStat-named statistics table (now including the
                # circ-lin correlation block if we computed one).
                try:
                    os.makedirs(fig_dir, exist_ok=True)
                    save_circular_statistics_pdf(
                        ta, cs,
                        pdf_path=os.path.join(
                            fig_dir, f"{stem}_circular_statistics.pdf"),
                        file_label=stem,
                        fig_theme=p.get("fig_theme", "Dark"),
                        circ_lin_result=circ_lin)
                    extras_saved.append("circular stats PDF")
                except Exception as pdf_exc:
                    _log(f"  WARN: circular-stats PDF failed: {pdf_exc}")
            except Exception as cs_exc:
                _log(f"  WARN: circular-stats save failed: {cs_exc}")
    except Exception as exc:
        _log(f"  WARN: turning-angles save failed: {exc}")
    try:
        if mf is not None and len(mf):
            mf.to_csv(
                os.path.join(extras_dir, f"{stem}_mobile_fraction.csv"),
                index=False)
            extras_saved.append("mobile fraction")
    except Exception as exc:
        _log(f"  WARN: mobile-fraction save failed: {exc}")
    try:
        if cluster_stats_df is not None and len(cluster_stats_df):
            cluster_stats_df.to_csv(
                os.path.join(extras_dir, f"{stem}_cluster_stats.csv"),
                index=False)
            extras_saved.append("cluster stats")
    except Exception as exc:
        _log(f"  WARN: cluster-stats save failed: {exc}")
    # Per-loc cluster labels: needed by the Visualise-tab interactive
    # cluster map (colours each localisation by its cluster_id, with -1
    # = noise).  cluster_xy is the µm-coordinate array compute_clusters
    # used internally, aligned with cluster_labels.
    #
    # We ALSO attach per-loc motion class (Immobile/Confined/Brownian/
    # Directed/Unknown) so the Visualise tab can colour clusters by
    # their dominant motion class — the user's natural follow-up
    # question after "what is each cluster?".  Locs that didn't end
    # up in any track (filtered out by min_track_len) are tagged
    # "Unmatched" and rendered as noise downstream.
    try:
        if (cluster_labels is not None and cluster_xy is not None
                and len(cluster_labels) == len(cluster_xy)):
            import pandas as _pd
            # Build the per-loc motion class array.  `tracks` rows are a
            # subset of the post-link locs (some dropped by min_len),
            # and within tracks each row has a `particle` ID we can
            # join against `diff_df` for the motion column.
            motion_per_loc = ["Unmatched"] * len(cluster_labels)
            try:
                if (tracks is not None and len(tracks) > 0
                        and diff_df is not None
                        and "motion" in diff_df.columns):
                    # particle → motion mapping
                    motion_map = dict(zip(
                        diff_df["particle"].astype(int),
                        diff_df["motion"].astype(str)))
                    # Build a quick (frame, x_rounded, y_rounded) → motion
                    # lookup so we can match track rows to original locs.
                    # Rounding to 4 dp handles float32 round-trip noise
                    # without falsely merging spatially-distinct locs.
                    key_to_motion = {}
                    for f_, x_, y_, p_ in zip(
                            tracks["frame"].to_numpy(),
                            tracks["x"].to_numpy(),
                            tracks["y"].to_numpy(),
                            tracks["particle"].to_numpy()):
                        key = (int(f_), round(float(x_), 4),
                               round(float(y_), 4))
                        m_ = motion_map.get(int(p_), "Unknown")
                        key_to_motion[key] = m_
                    # locs columns at this point: x, y, frame, mass.
                    loc_xy_um = cluster_xy  # already in µm
                    # locs.x / locs.y are in PIXELS — convert to µm to
                    # match the keys we built (tracks is also in pixels
                    # though; let's match in pixel space directly).
                    if (len(locs) == len(cluster_labels)
                            and "frame" in locs.columns
                            and "x" in locs.columns
                            and "y" in locs.columns):
                        for i, (f_, x_, y_) in enumerate(zip(
                                locs["frame"].to_numpy(),
                                locs["x"].to_numpy(),
                                locs["y"].to_numpy())):
                            key = (int(f_), round(float(x_), 4),
                                   round(float(y_), 4))
                            m_ = key_to_motion.get(key)
                            if m_:
                                motion_per_loc[i] = m_
            except Exception as exc_join:
                _log(f"  NOTE: cluster ↔ motion join skipped "
                     f"({exc_join}) — saving cluster_labels without "
                     f"motion column.")
            _pd.DataFrame({
                "loc_index": _np.arange(len(cluster_labels),
                                         dtype=_np.int64),
                "x_um":      _np.asarray(cluster_xy[:, 0],
                                          dtype=_np.float32),
                "y_um":      _np.asarray(cluster_xy[:, 1],
                                          dtype=_np.float32),
                "cluster_id": _np.asarray(cluster_labels,
                                           dtype=_np.int32),
                "motion":    motion_per_loc,
            }).to_csv(
                os.path.join(extras_dir, f"{stem}_cluster_labels.csv"),
                index=False)
            extras_saved.append("cluster labels")
    except Exception as exc:
        _log(f"  WARN: cluster-labels save failed: {exc}")
    # Per-run params — Compare relies on this for per-folder pixel size /
    # frame interval, etc.  Matches the schema written by the
    # PALM-Tracer summary loader.
    try:
        import json as _json
        with open(os.path.join(extras_dir,
                                f"{stem}_params.json"), "w") as _fp:
            _json.dump({
                "stem":             stem,
                "pixel_size_um":    float(px),
                "frame_interval_s": float(fi),
                "n_localisations":  int(len(locs)),
                "n_tracks":         int(diff_df.shape[0]) if diff_df is not None else 0,
                "n_frames":         int(n_frames),
                "width":            int(stack_w),
                "height":           int(stack_h),
                "source":           "firefly",
                # Detection threshold actually used (per-file when auto).
                "auto_minmass":     bool(p.get("auto_minmass", False)),
                "minmass_used":     (float(minmass_arg)
                                     if minmass_arg is not None else None),
                "minmass_method":   (mm_diag.get("method") if mm_diag else "manual"),
                "minmass_sensitivity": (mm_diag.get("sensitivity") if mm_diag else None),
                "minmass_n_candidates": (mm_diag.get("n_candidates") if mm_diag else None),
                # Linkability-sweep diagnostics (None on the static fallback).
                "minmass_n_good":   (mm_diag.get("n_good") if mm_diag else None),
                "minmass_spurious_rate": (mm_diag.get("spurious_rate") if mm_diag else None),
                "minmass_score":    (mm_diag.get("score") if mm_diag else None),
                "minmass_noise_floor": (mm_diag.get("noise_floor") if mm_diag else None),
                "backend":          p.get("backend"),
                # Path to the original input file/folder — Post-process
                # tab uses this to reload a background image.  Stored as
                # absolute path so the source location survives folder moves.
                "input_file":       os.path.abspath(p.get("file", "")) if p.get("file") else None,
            }, _fp, indent=2)
        extras_saved.append("params")
    except Exception as exc:
        _log(f"  WARN: params save failed: {exc}")

    # Auto-threshold audit histogram (always written when auto picked a value).
    if mm_diag is not None and mm_diag.get("_log_masses") is not None:
        try:
            from firefly.analysis.fa_localize import render_minmass_audit
            render_minmass_audit(
                mm_diag,
                os.path.join(extras_dir, f"{stem}_minmass_hist.png"),
                theme=p.get("theme", "Dark"), stem=stem)
            extras_saved.append("minmass_hist")
        except Exception as exc:
            _log(f"  WARN: auto-threshold audit save failed: {exc}")

    _log(f"  Saved (firefly_extras/): {', '.join(extras_saved)}")

    figure_path = ""
    fig_dpi = int(p.get("fig_dpi", 150)) or 150
    try:
        figure_path = os.path.join(fig_dir, f"{stem}_sptpalm_figure.png")
        fig_data["combined"].save(figure_path, dpi=(fig_dpi, fig_dpi))
    except Exception as e:
        _log(f"  WARN: figure save failed: {e}")
        figure_path = ""

    # Optional: vector PDF copy of the combined figure
    if want_pdf and fig_data.get("pdf_bytes"):
        try:
            pdf_path = os.path.join(fig_dir, f"{stem}_sptpalm_figure.pdf")
            with open(pdf_path, "wb") as _fh:
                _fh.write(fig_data["pdf_bytes"])
            _log(f"  Saved (figures/): vector PDF")
        except Exception as e:
            _log(f"  WARN: PDF save failed: {e}")

    # Optional: per-panel PNGs (one image per labelled panel of the grid).
    # The user can filter which panels get written via the Figures tab's
    # "Single-sample panels to export individually" checkbox grid.
    if bool(p.get("fig_per_panel", False)) and fig_data.get("panels"):
        try:
            allowed = p.get("fig_single_panels")
            if allowed is None:
                wanted_keys = list(fig_data["panels"].keys())
            else:
                allowed_set = set(allowed)
                wanted_keys = [k for k in fig_data["panels"].keys()
                               if k in allowed_set]
            panel_dir = os.path.join(fig_dir, "panels")
            os.makedirs(panel_dir, exist_ok=True)
            n_saved = 0
            for ltr in wanted_keys:
                fig_data["panels"][ltr].save(
                    os.path.join(panel_dir, f"{stem}_panel_{ltr}.png"),
                    dpi=(fig_dpi, fig_dpi))
                n_saved += 1
            _log(f"  Saved (figures/panels/): {n_saved} panel PNGs")
        except Exception as e:
            _log(f"  WARN: per-panel save failed: {e}")

    # ── Reproducibility manifest ──────────────────────────────────────────
    # Write a self-contained JSON next to the outputs that records the
    # exact parameters used + input-file checksum + FIREFLY version + git
    # SHA + host info, so the run can be exactly replayed later via the
    # "Load manifest…" button on the Import tab.
    manifest_path = ""
    try:
        manifest_path = _write_run_manifest(
            out_dir=out_dir, stem=stem, fpath=fpath, params=p)
        _log(f"  Saved (root): {os.path.basename(manifest_path)}")
    except Exception as e:
        _log(f"  WARN: manifest write failed: {e}")

    _log(f"\n  Output folder: {out_dir}")
    _prog(100, "Complete!")

    # ── Summary stats for the GUI results panel ──────────────────────────
    # Computed defensively so a partial pipeline still returns a valid
    # payload (e.g. when filter-by-D produced an empty diff_df).
    summary = {
        "n_tracks":     int(diff_df.shape[0]) if diff_df is not None else 0,
        "n_locs":       int(len(locs))         if locs    is not None else 0,
        "median_d":     None,
        "median_alpha": None,
        "median_loc_sigma_nm": None,
        "nongauss_alpha2": None,
        "vacf_persistence": None,
        "motion_counts": {},
        "mobile_fraction": None,
        "n_clusters":   0,
        "dwell_tau_s":  None,
        "frames":       int(n_frames),
        "px_um":        float(px),
        "fi_s":         float(fi),
    }
    try:
        if diff_df is not None and len(diff_df):
            if "D" in diff_df.columns:
                summary["median_d"] = float(diff_df["D"].median())
            if "alpha" in diff_df.columns:
                summary["median_alpha"] = float(diff_df["alpha"].median())
            if "motion" in diff_df.columns:
                summary["motion_counts"] = {
                    str(k): int(v) for k, v
                    in diff_df["motion"].value_counts().to_dict().items()
                }
            if "D" in diff_df.columns:
                d_thresh = float(p.get("mobile_d_threshold", 0.05))
                summary["mobile_fraction"] = float(
                    (diff_df["D"] > d_thresh).mean())
            if "loc_sigma_nm" in diff_df.columns:
                _ls = diff_df["loc_sigma_nm"].dropna()
                if len(_ls):
                    summary["median_loc_sigma_nm"] = float(_ls.median())
        if van_hove is not None:
            try:
                summary["nongauss_alpha2"] = float(
                    van_hove["non_gaussian_alpha2"])
            except Exception:
                pass
        if vacf is not None:
            try:
                summary["vacf_persistence"] = float(vacf["persistence"])
            except Exception:
                pass
        if cluster_stats_df is not None and len(cluster_stats_df):
            summary["n_clusters"] = int(len(cluster_stats_df))
        if dwell_tau is not None:
            try:
                summary["dwell_tau_s"] = float(dwell_tau)
            except Exception:
                pass
    except Exception:
        # Best-effort: don't let a stats-computation hiccup break the run
        pass

    # ── Quality-control metrics ──────────────────────────────────────────
    # Cheap to compute from data we already have; surfaced as a QC panel
    # in the GUI so the user can catch dud runs at a glance.
    qc: dict = {"flags": []}
    try:
        n_locs    = int(len(locs)) if locs is not None else 0
        n_tracked = 0
        gap_frac  = None
        median_len = None
        stuck_frac = None
        avg_locs_pf = None
        if tracks is not None and len(tracks) > 0:
            n_tracked = int(len(tracks))
            # Track-length distribution (frames per particle)
            lens = tracks.groupby("particle").size()
            median_len = float(lens.median())
            # Gap rate: a track has a gap when its frame range > its length
            try:
                frames_per_p = tracks.groupby("particle")["frame"]
                spans = frames_per_p.max() - frames_per_p.min() + 1
                gap_mask = spans > lens
                gap_frac = float(gap_mask.mean())
            except Exception:
                pass
        if diff_df is not None and len(diff_df) > 0 and "D" in diff_df.columns:
            stuck_frac = float((diff_df["D"] < 1e-3).mean())
        if n_locs and n_frames:
            avg_locs_pf = float(n_locs) / float(n_frames)
        link_ratio = (float(n_tracked) / n_locs) if n_locs else None

        # Total spatial extent of the corrected drift over the movie (nm) — a
        # QC read on how much stage/sample drift was present.  Only available
        # when drift correction ran.
        drift_total_nm = None
        if drift_df is not None and len(drift_df) > 1:
            try:
                ddx = drift_df["dx"].to_numpy(dtype=float)
                ddy = drift_df["dy"].to_numpy(dtype=float)
                span = float(np.hypot(ddx.max() - ddx.min(),
                                      ddy.max() - ddy.min()))
                drift_total_nm = span * float(px) * 1000.0
            except Exception:
                drift_total_nm = None

        qc.update({
            "n_locs":              n_locs,
            "n_tracked_locs":      n_tracked,
            "link_ratio":          link_ratio,
            "avg_locs_per_frame":  avg_locs_pf,
            "median_track_length": median_len,
            "gap_fraction":        gap_frac,
            "stuck_fraction":      stuck_frac,
            "drift_total_nm":      drift_total_nm,
        })

        # Threshold-based flags — surface as warnings in the GUI
        flags: list[dict] = []
        if link_ratio is not None and link_ratio < 0.10:
            flags.append({"level": "warn",
                "msg": f"Only {link_ratio*100:.1f}% of localisations were "
                       "linked into tracks — consider raising minmass or "
                       "lowering search_range."})
        if avg_locs_pf is not None and avg_locs_pf > 800:
            flags.append({"level": "warn",
                "msg": f"Very high localisation density "
                       f"({avg_locs_pf:.0f} locs/frame).  Linking accuracy "
                       "degrades above ~1000/frame; consider raising minmass."})
        if median_len is not None and median_len < 6:
            flags.append({"level": "warn",
                "msg": f"Median track length is only {median_len:.1f} "
                       "frames — MSD fits will be noisy.  Lower memory or "
                       "search_range, or raise minmass."})
        if stuck_frac is not None and stuck_frac > 0.30:
            flags.append({"level": "warn",
                "msg": f"{stuck_frac*100:.1f}% of tracks have "
                       "D < 1e-3 µm²/s (likely stuck / aggregated).  "
                       "Consider enabling Filter-by-D in the sidebar."})
        if gap_frac is not None and gap_frac > 0.50:
            flags.append({"level": "info",
                "msg": f"{gap_frac*100:.1f}% of tracks contain gaps.  "
                       "OK for blinking PALM probes; suspicious for "
                       "constitutive markers."})
        if drift_total_nm is not None and drift_total_nm > 500:
            flags.append({"level": "info",
                "msg": f"{drift_total_nm:.0f} nm of sample drift was corrected "
                       "over the acquisition — large drift can still leave "
                       "residual blur; inspect the drift trace if D looks high."})
        qc["flags"] = flags
    except Exception:
        pass
    summary["qc"] = qc

    # ── Persist the headline metrics as a single machine-readable file ───
    # Everything the GUI results panel shows (counts, median D/alpha, loc
    # precision, alpha2, persistence, mobile fraction, QC flags) in one JSON,
    # so a batch of N runs can be aggregated by globbing
    # firefly_extras/*_summary_metrics.json — no need to re-open each CSV.
    try:
        import json as _json
        _sm = dict(summary)
        _sm["stem"] = stem
        with open(os.path.join(extras_dir,
                               f"{stem}_summary_metrics.json"), "w") as _fp:
            _json.dump(_sm, _fp, indent=2, default=str)
    except Exception as exc:
        _log(f"  WARN: summary-metrics save failed: {exc}")

    return {
        "stem":        stem,
        "out_dir":     out_dir,
        "figure_path": figure_path,
        "summary":     summary,
        # Legacy top-level keys preserved for compatibility with callers
        # that haven't been updated yet.
        "n_tracks":    summary["n_tracks"],
        "n_locs":      summary["n_locs"],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — SINGLE FILE
# ══════════════════════════════════════════════════════════════════════════════
def run_analysis(params: dict, msg_queue, cancel_event):
    """Single-file subprocess entry.  Wraps `_run_one_analysis` with the
    stdout/stderr redirect and translates exceptions into terminal queue
    messages ("done" / "stopped" / "error")."""
    sys.stdout = QueueLogStream(msg_queue)
    sys.stderr = QueueLogStream(msg_queue)

    # Persist this subprocess's diagnostics to its own rotating log file
    # (separate from the GUI's, to avoid cross-process rotation races).
    # console=False: stderr is already the queue stream above, so the GUI
    # console still shows the run; this just adds a durable on-disk record.
    try:
        from firefly import crash_reporter as _cr
        _cr.setup_logging(filename="firefly_worker.log", console=False)
    except Exception:
        pass

    def _log(msg: str):       msg_queue.put(("log", msg))
    def _prog(pct, msg):      msg_queue.put(("progress", (int(pct), str(msg))))

    # Memory watchdog — aborts the run cleanly if free RAM falls
    # below the critical threshold, preventing the OS-level OOM /
    # kernel freeze that we saw previously.
    _wd_stop = _start_memory_watchdog(cancel_event, msg_queue)
    # Disk watchdog — aborts cleanly if free space on the output
    # volume drops below 200 MB.  Catches the ENOSPC-mid-save case
    # where 5 minutes of analysis is lost when a single CSV write
    # fails at the very end.
    _disk_stop = None
    try:
        _out_dir = (params.get("out_dir")
                    or os.path.dirname(os.path.abspath(params["file"])))
        _disk_stop = _start_disk_watchdog(_out_dir, cancel_event, msg_queue)
    except Exception as exc:
        _log(f"  WARN: disk-full watchdog could not start ({exc}); the run "
             f"will proceed without low-disk protection")

    try:
        _log("── Worker subprocess started ──")
        _prog(0, "Importing pipeline…")
        payload = _run_one_analysis(params, msg_queue, cancel_event, _log, _prog)
        msg_queue.put(("done", payload))
    except _NoTracks as nt:
        # Linker produced 0 trajectories — not a crash.  Treat as "done"
        # with the partial payload so the UI resets cleanly.
        msg_queue.put(("done", nt.args[0]))
    except BaseException as exc:
        if type(exc).__name__ in ("_Cancelled", "_Stopped"):
            msg_queue.put(("log", "\n── Stopped by user ──"))
            msg_queue.put(("stopped", None))
        else:
            msg_queue.put(("error", traceback.format_exc()))
    finally:
        if _wd_stop is not None:
            _wd_stop.set()
        if _disk_stop is not None:
            _disk_stop.set()
        try: sys.stdout.flush()
        except Exception: pass
        try: sys.stderr.flush()
        except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — POST-PROCESS  (re-apply ROI to an existing run)
# ══════════════════════════════════════════════════════════════════════════════
def run_postproc(params: dict, msg_queue, cancel_event):
    """Re-run a previously-completed FIREFLY analysis with a NEW ROI.

    Reloads the original `_localisations.csv` (= pre-ROI locs) from
    `source_folder/firefly_extras/`, applies the new polygon(s),
    writes the filtered locs to a temp CSV, then dispatches the
    existing `_run_one_analysis` in `external_csv` mode so the rest
    of the pipeline (link → MSD → JDD → turning angles → dwell times
    → clusters → figure → all CSVs) runs against the new loc set.

    `params` is a dict with:
        source_folder : str  — the analysis run to re-process
        new_polygons  : list of (N, 2) [y, x] arrays in pixel coords
        output_folder : str | None  — where to write outputs.  Defaults
                        to `<source_folder>_postproc1` (auto-increments
                        the suffix if that path already exists).
    """
    sys.stdout = QueueLogStream(msg_queue)
    sys.stderr = QueueLogStream(msg_queue)
    def _log(msg: str):  msg_queue.put(("log", msg))
    def _prog(pct, msg): msg_queue.put(("progress", (int(pct), str(msg))))

    _wd_stop = _start_memory_watchdog(cancel_event, msg_queue)
    _disk_stop = None
    try:
        _src = params.get("source_folder")
        if _src:
            _disk_stop = _start_disk_watchdog(_src, cancel_event, msg_queue)
    except Exception as exc:
        _log(f"  WARN: disk-full watchdog could not start ({exc}); the run "
             f"will proceed without low-disk protection")

    try:
        _log("── Post-process worker started ──")
        _prog(0, "Reading previous run…")
        src = params.get("source_folder")
        if not src or not os.path.isdir(src):
            raise FileNotFoundError(
                f"source_folder not a directory: {src!r}")
        extras_dir = os.path.join(src, "firefly_extras")
        if not os.path.isdir(extras_dir):
            raise FileNotFoundError(
                f"No firefly_extras/ inside {src!r}")
        # Find the run's stem from any *_params.json or *_locs CSV.
        params_files = [f for f in os.listdir(extras_dir)
                        if f.endswith("_params.json")]
        loc_files = [f for f in os.listdir(extras_dir)
                     if f.endswith("_localisations.csv")]
        if not loc_files:
            raise FileNotFoundError(
                f"Couldn't find *_localisations.csv in {extras_dir!r} — "
                f"older runs may not have saved it.  Re-run the original "
                f"analysis to regenerate it before post-processing.")
        stem = loc_files[0][:-len("_localisations.csv")]
        _log(f"  Source : {src}")
        _log(f"  Stem   : {stem}")

        # Read original params so we can copy detection/link/MSD etc.
        # settings into the new run.
        import json as _json
        orig_params = {}
        if params_files:
            try:
                with open(os.path.join(
                        extras_dir, params_files[0])) as fh:
                    orig_params = _json.load(fh) or {}
            except Exception as exc:
                _log(f"  WARN: couldn't read original params.json ({exc})"
                     f" — falling back to FIREFLY defaults for missing keys.")

        # Decide output folder: <source>_postproc{N}, auto-incrementing.
        out_dir = params.get("output_folder")
        if not out_dir:
            base = src.rstrip(os.sep)
            n = 1
            while os.path.isdir(f"{base}_postproc{n}"):
                n += 1
            out_dir = f"{base}_postproc{n}"
        os.makedirs(out_dir, exist_ok=True)
        _log(f"  Output : {out_dir}")

        # Load the pre-ROI localisations.
        import pandas as _pd, numpy as _np
        locs_csv = os.path.join(extras_dir, f"{stem}_localisations.csv")
        locs = _pd.read_csv(locs_csv)
        n_before = len(locs)
        _log(f"  Loaded {n_before:,} localisations from "
             f"{stem}_localisations.csv")

        # Build the bool ROI mask from the supplied polygons.
        polys = params.get("new_polygons") or []
        if not polys:
            raise ValueError(
                "No ROI polygons supplied — post-processing needs at "
                "least one polygon to filter the localisations.")

        # Apply ROI directly in xy space using a Path containment test
        # (no need to build a raster mask, which would require the
        # original image dimensions and pixel scale).
        from matplotlib.path import Path as _MplPath
        xs = locs["x"].to_numpy(dtype=float)
        ys = locs["y"].to_numpy(dtype=float)
        pts = _np.column_stack([xs, ys])
        keep_mask = _np.zeros(len(locs), dtype=bool)
        for poly in polys:
            arr = _np.asarray(poly, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] < 3:
                continue
            # Polygons come in [y, x] from the napari Shapes layer;
            # swap to [x, y] for the matplotlib Path test.
            path = _MplPath(_np.column_stack([arr[:, 1], arr[:, 0]]))
            keep_mask |= path.contains_points(pts)
        locs_new = locs.loc[keep_mask].reset_index(drop=True)
        n_after = len(locs_new)
        _log(f"  ROI applied: kept {n_after:,} / {n_before:,} "
             f"({100.0 * n_after / max(1, n_before):.1f}%)")
        if n_after == 0:
            raise RuntimeError(
                "New ROI excluded ALL localisations — nothing to "
                "analyse.  Draw a larger / better-placed polygon.")

        # Persist the filtered locs as a temp CSV and dispatch the
        # existing external-CSV pipeline at the new output folder.
        tmp_csv = os.path.join(out_dir, f"{stem}_postproc_input.csv")
        locs_new.to_csv(tmp_csv, index=False)

        # Build the derived params for _run_one_analysis.  Copy any
        # original analysis knobs (search_range, memory, MSD lags etc.)
        # so the post-processed result is comparable to the original
        # apart from the ROI change.
        new_p = dict(orig_params)
        # If the original input image is still on disk, feed it through as
        # the background so the figure's Max Projection + trajectory panels
        # plot against a real image instead of a 256×256 black square.
        _orig_input = orig_params.get("input_file") or ""
        _bg_path = _orig_input if (_orig_input and os.path.isfile(_orig_input)) else ""
        new_p.update({
            "file":           tmp_csv,
            "source":         "external_csv",
            "csv_preset":     "auto",   # FIREFLY's own loc columns autodetect
            "out_dir":        out_dir,
            # We've already chosen the wrapper name (`<source>_postprocN`);
            # don't let _run_one_analysis nest another `<stem>/` inside it.
            "wrap_in_stem_folder": False,
            "channel":        0,
            "bg_image_path":  _bg_path,
            # ROI is already applied — tell downstream to skip it.
            "roi_mode":       "none",
            "roi_polygon":    None,
        })
        # Sensible defaults if orig_params was missing any key.
        new_p.setdefault("pixel_size",   0.106)
        new_p.setdefault("frame_interval", 0.03)
        new_p.setdefault("diameter",     7)
        new_p.setdefault("minmass",      1.0)
        new_p.setdefault("auto_minmass", False)
        new_p.setdefault("search_range", 5)
        new_p.setdefault("memory",       3)
        new_p.setdefault("min_track_len", 5)
        new_p.setdefault("max_track_len", None)
        new_p.setdefault("max_lagtime",  20)
        new_p.setdefault("n_fit",        5)
        new_p.setdefault("workers",      max(1, os.cpu_count() or 1))
        new_p.setdefault("chunk_size",   500)

        _log("── Re-running downstream stages with new ROI ──")
        payload = _run_one_analysis(new_p, msg_queue, cancel_event,
                                    _log, _prog)
        # Surface the source + new output paths in the done payload so
        # the GUI can offer to "Open the post-processed run".
        payload["source_folder"] = src
        payload["postproc_output"] = out_dir
        msg_queue.put(("done", payload))
    except _NoTracks as nt:
        msg_queue.put(("done", nt.args[0]))
    except BaseException as exc:
        if type(exc).__name__ in ("_Cancelled", "_Stopped"):
            msg_queue.put(("log", "\n── Stopped by user ──"))
            msg_queue.put(("stopped", None))
        else:
            msg_queue.put(("error", traceback.format_exc()))
    finally:
        if _wd_stop is not None:
            _wd_stop.set()
        if _disk_stop is not None:
            _disk_stop.set()
        try: sys.stdout.flush()
        except Exception: pass
        try: sys.stderr.flush()
        except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — COMPARE  (N-group comparison run in a subprocess)
# ══════════════════════════════════════════════════════════════════════════════
def run_comparison(comparison_params: dict, msg_queue, cancel_event):
    """Run sptpalm_analysis.compare_groups in a subprocess.

    Same rationale for the subprocess as run_analysis: keep matplotlib +
    pandas + scipy import cost out of the Qt main process, and (on
    Apple Silicon) keep Metal contention with Qt to a minimum.

    Messages emitted
    ----------------
      log/progress  — same conventions as single-file mode
      compare_done(payload)
                    — terminal success message.  payload keys:
                        figure_path : str — saved .png
                        output_dir  : str — folder containing all outputs
                        summary_csv : str — per-replicate scalars
                        stats_csv   : str — pairwise tests
                        pdf_report  : str — combined PDF (if requested)
      stopped       — cooperative cancel fired
      error(tb)     — unrecoverable exception
    """
    sys.stdout = QueueLogStream(msg_queue)
    sys.stderr = QueueLogStream(msg_queue)

    def _log(msg: str):  msg_queue.put(("log", msg))
    def _prog(pct, msg): msg_queue.put(("progress", (int(pct), str(msg))))

    # Memory watchdog — Compare can load multiple full tracks
    # DataFrames simultaneously + re-derive MSD/JDD/dwell/turning
    # angles for each, easily blowing the budget on 16 GB machines.
    _wd_stop = _start_memory_watchdog(cancel_event, msg_queue)
    _disk_stop = None
    try:
        _cmp_out = comparison_params.get("output_dir")
        if _cmp_out:
            _disk_stop = _start_disk_watchdog(
                _cmp_out, cancel_event, msg_queue)
    except Exception as exc:
        _log(f"  WARN: disk-full watchdog could not start ({exc}); the run "
             f"will proceed without low-disk protection")

    try:
        _log("── Compare worker subprocess started ──")
        _prog(0, "Importing comparison pipeline…")

        from firefly.sptpalm_analysis import compare_groups, _Cancelled

        p = comparison_params

        # compare_groups expects panels as a set; the Qt side ships a list
        # (JSON-friendly).  Normalise here.
        panels = set(p.get("panels") or []) or None

        # Wire progress callback → queue.  compare_groups invokes this
        # periodically during folder loading; we map to percent.
        def _progress_cb(done: int, total: int, msg: str):
            if cancel_event.is_set():
                raise _Cancelled()
            pct = int(100 * done / total) if total else 0
            _prog(pct, msg)

        out_dir   = p.get("output_dir")
        out_stem  = p.get("output_stem", "comparison")
        theme     = p.get("theme", "Dark")
        pdf_report = bool(p.get("pdf_report", True))
        mob_d     = float(p.get("mobile_d_threshold", 0.05))

        _log(f"  Output dir : {out_dir}")
        _log(f"  Output stem: {out_stem}")
        _log(f"  Theme      : {theme}")
        _log(f"  Groups     : {len(p.get('groups', []))}")
        for g in p.get("groups", []):
            _log(f"    {g.get('label', '?'):<20s}"
                 f"({len(g.get('folders', []))} folders)")

        fig, summary_df, stats = compare_groups(
            groups=p["groups"],
            output_dir=out_dir,
            output_stem=out_stem,
            panels=panels,
            theme=theme,
            pdf_report=pdf_report,
            mobile_d_threshold=mob_d,
            progress_cb=_progress_cb)

        # Compose result paths.  compare_groups saves these by convention:
        figure_path = os.path.join(out_dir, f"{out_stem}.png")
        summary_csv = os.path.join(out_dir, f"{out_stem}_summary.csv")
        stats_csv   = os.path.join(out_dir, f"{out_stem}_stats.csv")
        pdf_path    = os.path.join(out_dir, f"{out_stem}_report.pdf")

        _prog(100, "Comparison complete")
        msg_queue.put(("compare_done", {
            "output_dir":  out_dir,
            "figure_path": figure_path if os.path.isfile(figure_path) else "",
            "summary_csv": summary_csv if os.path.isfile(summary_csv) else "",
            "stats_csv":   stats_csv   if os.path.isfile(stats_csv)   else "",
            "pdf_report":  pdf_path    if os.path.isfile(pdf_path)    else "",
            "n_groups":    len(p.get("groups", [])),
        }))

    except BaseException as exc:
        if type(exc).__name__ in ("_Cancelled", "_Stopped"):
            msg_queue.put(("log", "\n── Stopped by user ──"))
            msg_queue.put(("stopped", None))
        else:
            msg_queue.put(("error", traceback.format_exc()))
    finally:
        if _wd_stop is not None:
            _wd_stop.set()
        if _disk_stop is not None:
            _disk_stop.set()
        try: sys.stdout.flush()
        except Exception: pass
        try: sys.stderr.flush()
        except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT — BATCH (multiple files in one subprocess)
# ══════════════════════════════════════════════════════════════════════════════
def run_batch_analysis(params_list: list, msg_queue, cancel_event):
    """Run `_run_one_analysis` over each entry in `params_list`.

    One subprocess, N files — amortizes the spawn import cost across the
    whole batch.  Per-file failures don't abort the run: the failing file
    gets a `("file_error", ...)` message and the batch continues.  The
    final summary is `("batch_done", {n_total, n_ok, n_fail, results})`.

    Messages
    --------
    log/progress       — same as single-file mode (forwarded from
                         _run_one_analysis)
    file_done(payload) — emitted after each successful file
    file_error(info)   — emitted after each failed file
    stopped            — user cancelled; whole batch aborts
    batch_done(summary)— terminal message; batch completed normally
    error(tb)          — terminal message; unrecoverable worker error
    """
    sys.stdout = QueueLogStream(msg_queue)
    sys.stderr = QueueLogStream(msg_queue)

    def _log(msg: str):  msg_queue.put(("log", msg))
    def _prog(pct, msg): msg_queue.put(("progress", (int(pct), str(msg))))

    # Memory watchdog — one shared across the whole batch so a series
    # halfway through doesn't push the system into swap.  We pass a
    # separate `mem_abort` Event so the per-file exception handler can
    # tell a watchdog-triggered _Cancelled apart from a real user-stop:
    # the former should skip-and-continue, the latter aborts the batch.
    import threading as _th
    mem_abort = _th.Event()
    _wd_stop = _start_memory_watchdog(
        cancel_event, msg_queue, mem_abort_event=mem_abort)
    # Disk watchdog — same idea but for free space on the output disk.
    # Critical for batch mode where cumulative outputs (CSVs, PDFs,
    # PNGs per file) can fill a near-full data drive midway through.
    _disk_stop = None
    try:
        _first_out = (params_list[0].get("out_dir")
                      or os.path.dirname(
                          os.path.abspath(params_list[0]["file"])))
        _disk_stop = _start_disk_watchdog(
            _first_out, cancel_event, msg_queue)
    except Exception as exc:
        _log(f"  WARN: disk-full watchdog could not start ({exc}); the run "
             f"will proceed without low-disk protection")

    try:
        n = len(params_list)
        _log(f"── Batch worker subprocess started — {n} file(s) ──")
        results = []

        for i, params in enumerate(params_list, 1):
            if cancel_event.is_set():
                # Same distinction as the inner handler: a watchdog
                # abort raised at the file boundary should skip this
                # file, not terminate the batch.
                if mem_abort.is_set():
                    _log(f"\n  ⚠ Memory watchdog abort persisted into "
                         f"file {i}/{n} — skipping this file.")
                    mem_abort.clear()
                    cancel_event.clear()
                    try:
                        import gc as _gc
                        _gc.collect()
                    except Exception: pass
                    results.append({"index": i, "ok": False,
                                    "file": params["file"],
                                    "error": "memory watchdog abort"})
                    msg_queue.put(("file_error", {
                        "index": i, "total": n,
                        "file": params["file"],
                        "tb": "Aborted by memory watchdog "
                              "(insufficient free RAM).",
                    }))
                    continue
                _log("\n── Batch stopped by user ──")
                msg_queue.put(("stopped", None))
                return

            fname = os.path.basename(params["file"])
            _log("")
            _log("══════════════════════════════════════════════════════════════════")
            _log(f"  [{i}/{n}]  {fname}")
            _log("══════════════════════════════════════════════════════════════════")
            # Overall-batch progress: percent of files completed
            overall_pct = int(100 * (i - 1) / max(1, n))
            _prog(overall_pct, f"[{i}/{n}] {fname}")
            # GUI hook — reset per-file UI elements (mass histogram).  Live
            # view is fine; new preview_frame messages will overwrite the
            # previous file's frame so no explicit reset is needed there.
            msg_queue.put(("file_starting", {
                "index": i, "total": n, "file": fname,
            }))

            try:
                payload = _run_one_analysis(
                    params, msg_queue, cancel_event, _log, _prog)
                results.append({"index": i, "ok": True, "file": params["file"],
                                **payload})
                msg_queue.put(("file_done", {
                    "index": i, "total": n,
                    "stem":     payload.get("stem"),
                    "out_dir":  payload.get("out_dir"),
                    "n_tracks": payload.get("n_tracks", 0),
                    "n_locs":   payload.get("n_locs", 0),
                }))
            except _NoTracks as nt:
                results.append({"index": i, "ok": True, "file": params["file"],
                                **nt.args[0]})
                msg_queue.put(("file_done", {
                    "index": i, "total": n,
                    "stem":     nt.args[0].get("stem"),
                    "out_dir":  nt.args[0].get("out_dir"),
                    "n_tracks": 0,
                    "n_locs":   nt.args[0].get("n_locs", 0),
                }))
            except BaseException as exc:
                if type(exc).__name__ in ("_Cancelled", "_Stopped"):
                    # Distinguish memory-watchdog abort (skip this file,
                    # try the next — freeing the memmap usually recovers
                    # enough RAM) from a real user-cancel (terminate the
                    # whole batch).
                    if mem_abort.is_set():
                        _log(f"\n  ⚠ File {i}/{n} ({fname}) aborted by "
                             f"memory watchdog — skipping and continuing "
                             f"with the next file.")
                        results.append({"index": i, "ok": False,
                                        "file": params["file"],
                                        "error": "memory watchdog abort"})
                        msg_queue.put(("file_error", {
                            "index": i, "total": n,
                            "file": params["file"],
                            "tb": "Aborted by memory watchdog "
                                  "(insufficient free RAM).",
                        }))
                        # Reset both events + try to release the previous
                        # file's allocations before moving on.
                        mem_abort.clear()
                        cancel_event.clear()
                        try:
                            import gc as _gc
                            _gc.collect()
                        except Exception: pass
                        continue
                    _log("\n── Batch stopped by user ──")
                    msg_queue.put(("stopped", None))
                    return
                tb = traceback.format_exc()
                _log(f"\n  ⚠ File {i}/{n} ({fname}) FAILED: {exc}")
                _log(tb)
                results.append({"index": i, "ok": False, "file": params["file"],
                                "error": str(exc)})
                msg_queue.put(("file_error", {
                    "index": i, "total": n,
                    "file": params["file"], "tb": tb,
                }))
                # Continue with next file rather than aborting the whole batch
            finally:
                # Release the previous file's disk-backed memmap stack
                # before starting the next file.  Without this, a 13-file
                # batch can leave ~13 × 16 GB of temp files behind on the
                # boot volume (atexit cleanup only fires at process exit)
                # and hits ENOSPC mid-run.  Also gc.collect() so the
                # memmap object is dropped before the OS unlinks the file
                # (matters on Windows; harmless on POSIX).
                try:
                    import gc as _gc
                    _gc.collect()
                    from firefly.sptpalm_analysis import cleanup_temp_stack_paths
                    cleanup_temp_stack_paths()
                except Exception: pass
                # Free torch's MPS / CUDA caches between files.  PyTorch
                # holds onto allocated GPU memory in its cache after the
                # last tensor is freed; on Apple Silicon that cache can
                # easily reach several GB and is counted against the
                # unified RAM budget.  Clearing here gives the next file
                # a clean slate and prevents the slow growth that causes
                # OOM ~3-5 files into a batch.
                try:
                    import torch as _torch
                    if (hasattr(_torch, "mps")
                            and _torch.backends.mps.is_available()
                            and hasattr(_torch.mps, "empty_cache")):
                        _torch.mps.empty_cache()
                    if _torch.cuda.is_available():
                        _torch.cuda.synchronize()
                        _torch.cuda.empty_cache()
                except Exception: pass
                # Log post-file free RAM so the user can see whether
                # things are creeping toward the watchdog floor.
                try:
                    import psutil as _ps
                    free_gb_post = _ps.virtual_memory().available / 1e9
                    _log(f"  Post-file free RAM: {free_gb_post:.2f} GB")
                except Exception: pass

        # All done
        n_ok   = sum(1 for r in results if r.get("ok"))
        n_fail = n - n_ok
        _log("")
        _log("══════════════════════════════════════════════════════════════════")
        _log(f"  Batch complete: {n_ok}/{n} succeeded, {n_fail} failed")
        _log("══════════════════════════════════════════════════════════════════")
        _prog(100, "Batch complete!")
        msg_queue.put(("batch_done", {
            "n_total": n, "n_ok": n_ok, "n_fail": n_fail,
            "results": results,
        }))

    except BaseException:
        msg_queue.put(("error", traceback.format_exc()))
    finally:
        if _wd_stop is not None:
            _wd_stop.set()
        if _disk_stop is not None:
            _disk_stop.set()
        try: sys.stdout.flush()
        except Exception: pass
        try: sys.stderr.flush()
        except Exception: pass
