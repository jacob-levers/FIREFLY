"""Spot-localisation DETECTION BACKENDS — trackpy and PyTorch.

Extracted verbatim from fa_localize.py (behaviour-preserving) so the backend
implementations live apart from the chunking/streaming strategy and the
minmass-estimation code, and so a new backend (e.g. the a trous wavelet
detector) has a focused home.  fa_localize re-imports these names, so existing
`from firefly.analysis.fa_localize import TorchBackend` and the sptpalm_analysis
re-exports keep working unchanged.
"""
from __future__ import annotations

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:
    # Fallback no-op context manager if threadpoolctl unavailable
    from contextlib import contextmanager as _cm
    @_cm
    def _threadpool_limits(limits=None, user_api=None):
        yield

import contextlib
import io
import multiprocessing
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import trackpy as tp
from firefly.analysis.fa_constants import (N_CPUS, _Cancelled, _tqdm,
                                           safe_process_workers, _cpu_core_budget)
from firefly.analysis.fa_linking import _link_via_trackpy
from firefly.analysis.fa_memory import (_alloc_or_memmap_stack, _register_temp_stack_path,
                       _resolve_temp_stack_dir, _user_ram_reserve_gb)
from firefly.analysis.fa_preprocess import (preprocess_stack, _preprocess_fast,
                           _preprocess_rolling)

# Silence trackpy's per-frame INFO chatter at module import so it's quiet in
# BOTH the main process and any spawned sweep-worker processes (which import
# this module but not sptpalm_analysis, where tp.quiet() is otherwise called).
try:
    tp.quiet()
except Exception:
    pass

def _localise_chunk(chunk, diameter, minmass, percentile, frame_offset):
    """Localise one chunk and apply global frame offset.

    ``characterize=True`` so trackpy also returns ``ep`` — its empirical per-spot
    localisation-error estimate — which the worker maps to the per-localisation
    precision columns (loc_sigma_*_nm).  The other characterise extras
    (size/ecc/signal/raw_mass) are dropped so the production output shape stays
    ``x, y, frame, mass [, ep]`` (downstream linking/MSD only need those)."""
    locs = tp.batch(chunk, diameter=diameter, minmass=minmass,
                    percentile=percentile, processes=1, characterize=True)
    if len(locs) > 0:
        keep = [c for c in ("x", "y", "frame", "mass", "ep") if c in locs.columns]
        locs = locs[keep].copy()
        locs["frame"] += frame_offset
    return locs


def _balanced_chunk_ranges(n_frames, n_chunks):
    """Construct the canonical balanced ``[start, stop)`` ranges once."""
    n_frames = int(n_frames)
    n_chunks = max(1, min(int(n_chunks), max(n_frames, 1)))
    base, extra = divmod(n_frames, n_chunks)
    ranges = []
    start = 0
    for index in range(n_chunks):
        stop = start + base + (1 if index < extra else 0)
        if stop > start:
            ranges.append((start, stop))
        start = stop
    return ranges


def _localise_chunk_mp(args):
    """Picklable wrapper for multiprocessing.Pool.imap_unordered.
    Returns (index, dataframe) so we can preserve order despite unordered iteration."""
    idx, chunk, diameter, minmass, percentile, frame_offset = args
    result = _localise_chunk(chunk, diameter, minmass, percentile, frame_offset)
    return idx, result


def _localise_chunk_mmap_mp(args):
    """Memmap-aware variant of _localise_chunk_mp.

    Instead of pickling a multi-MB chunk array through the worker
    pipe, this receives just the memmap file path + shape/dtype +
    the [start, end) frame range to load.  The worker opens its
    own np.memmap on that file and views the slice — no copy
    crosses the pipe, no GIL contention on serialisation.

    Args tuple:
        (idx, path, dtype_str, shape, start, end,
         diameter, minmass, percentile, frame_offset)
    """
    (idx, path, dtype_str, shape, start, end,
     diameter, minmass, percentile, frame_offset) = args
    # Re-mmap read-only (we never write to the input stack).  Workers
    # all share the OS page cache for this file, so this is effectively
    # free after the parent has touched the pages.
    arr = np.memmap(path, dtype=np.dtype(dtype_str), mode="r",
                    shape=tuple(shape))
    chunk = arr[start:end]
    try:
        result = _localise_chunk(chunk, diameter, minmass, percentile,
                                  frame_offset)
    finally:
        # Release the worker's mapping promptly; the OS still holds
        # the file open for other workers.
        try:    del chunk
        except Exception: pass
        try:    arr._mmap.close()
        except Exception: pass
        try:    del arr
        except Exception: pass
    return idx, result


def _torch_localise_block_mp(args):
    """Picklable worker for the Torch-CPU multi-process path.

    Localises ONE chunk-aligned frame block on CPU torch and returns
    ``(block_idx, DataFrame[x,y,frame,mass])`` with **absolute** frame indices,
    or ``(block_idx, None)`` on any failure so the parent can fall back to the
    serial path.  Because each block is a whole run of ``chunk_size`` chunks and
    the per-chunk percentile threshold is batch-size-stable, the union of all
    blocks reproduces the serial detections exactly.

    Args tuple:
        (block_idx, source, diameter, minmass, percentile, chunk_size,
         threads_per_worker, frame_start)
    where ``source`` is either
        ("memmap", path, dtype_str, shape, frame_start, frame_end) or
        ("shm",    shm_name, dtype_str, shape, frame_start, frame_end).
    """
    (block_idx, source, diameter, minmass, percentile, chunk_size,
     threads_per_worker, frame_start) = args
    shm = None
    try:
        import numpy as _np
        # Bound this worker's intra-op threads so N workers × T threads ≈ cores
        # instead of every worker grabbing all of them (the oversubscription
        # that made single-process torch-CPU crawl).
        try:
            import torch as _torch
            _torch.set_num_threads(int(threads_per_worker))
        except Exception:
            pass

        kind = source[0]
        if kind == "memmap":
            _, path, dtype_str, shape, fs, fe = source
            arr = _np.memmap(path, dtype=_np.dtype(dtype_str), mode="r",
                             shape=tuple(shape))
            # np.array (NOT asarray) forces a writable COPY: asarray would
            # return a view of the memmap, and closing it below would leave the
            # tensor pointing at freed memory (segfault).  A copy also avoids
            # torch's non-writable-array warning.
            block = _np.array(arr[fs:fe], dtype=_np.float32)
            try:    arr._mmap.close()
            except Exception: pass
        elif kind == "shm":
            from multiprocessing import shared_memory
            _, shm_name, dtype_str, shape, fs, fe = source
            shm = shared_memory.SharedMemory(name=shm_name)
            full = _np.ndarray(tuple(shape), dtype=_np.dtype(dtype_str),
                               buffer=shm.buf)
            block = _np.array(full[fs:fe], dtype=_np.float32)   # copy out
        else:
            return block_idx, None

        inst = TorchBackend()
        inst._forced_device = "cpu"
        df = inst.localise(block, diameter=diameter, minmass=minmass,
                           percentile=percentile, chunk_size=chunk_size,
                           quiet=True, preview_cb=None,
                           _cpu_threads=int(threads_per_worker))
        if df is not None and len(df) and "frame" in df.columns:
            df = df.copy()
            df["frame"] = df["frame"].to_numpy() + int(frame_start)
        return block_idx, df
    except Exception:
        return block_idx, None
    finally:
        if shm is not None:
            try:    shm.close()
            except Exception: pass


class LocaliserBackend:
    """Abstract base for particle-localisation backends.

    Subclasses must set `name` and implement `is_available()` + `localise()`.
    """
    name: str = "abstract"

    @classmethod
    def is_available(cls) -> bool:
        return False

    def localise(self, stack, *, diameter=7, minmass=0.1, percentile=64,
                 workers=None, chunk_size=500, preview_cb=None, **kwargs):
        raise NotImplementedError


def _emit_trackpy_chunk_preview(preview_cb, chunk_or_stack,
                                  frame_range, chunk_locs_df, n_frames):
    """Emit one preview_cb call per frame in a completed Trackpy chunk.

    `chunk_or_stack` is either the chunk's own (T, Y, X) array
    (non-memmap MP / sequential path) OR the parent's full memmap-
    backed stack (memmap-MP path).  Either way we slice
    `[frame_range[0]:frame_range[1]]` to get the chunk's frames.

    `chunk_locs_df` is a DataFrame with columns x, y, frame; we group
    spots by their global frame index and forward each (frame, spots)
    pair to preview_cb.  The GUI's pump throttles to ~60 Hz and drops
    older frames if the queue fills, so blasting every frame is fine.

    Failures here must never break the analysis — wrap everything in
    a top-level try/except that swallows.
    """
    if preview_cb is None:
        return
    try:
        start, end = int(frame_range[0]), int(frame_range[1])
        # chunk_or_stack[start:end] is the (T, Y, X) view we hand to
        # preview_cb one frame at a time.  np.asarray makes the memmap
        # path concrete (the GUI's preview thread expects a real array,
        # not a memmap view it has to keep alive after we return).
        sub = np.asarray(chunk_or_stack[start:end], dtype=np.float32)
    except Exception:
        return
    if sub.size == 0:
        return
    # Bucket spots by their global frame index so each preview_cb call
    # hands the GUI just the detections for that frame.
    spots_by_frame: dict = {}
    try:
        if chunk_locs_df is not None and len(chunk_locs_df) > 0:
            for _f, _sub in chunk_locs_df.groupby("frame"):
                spots_by_frame[int(_f)] = (
                    _sub["x"].values, _sub["y"].values)
    except Exception:
        spots_by_frame = {}
    for local_i in range(len(sub)):
        global_i = start + local_i
        sxy = spots_by_frame.get(global_i, ([], []))
        try:
            preview_cb(global_i, sub[local_i],
                       sxy[0], sxy[1], n_frames)
        except Exception:
            pass


class TrackpyBackend(LocaliserBackend):
    """CPU localiser using trackpy's Crocker-Grier centroid detection.

    Parallelised via multiprocessing.Pool (spawn) for true multi-core scaling;
    falls back to a single-process BLAS-threaded path if Pool creation fails
    (rare, but happens on locked-down Windows boxes and inside some sandboxes).

    Accepted params:
        diameter     — odd integer, spot diameter in px (auto-bumped if even)
        minmass      — minimum integrated intensity for a spot
        percentile   — local-noise threshold (passed straight to tp.batch)
        workers      — process pool size (defaults to N_CPUS)
        chunk_size   — frames per chunk (memory / parallelism tradeoff)
    """
    name = "trackpy"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import trackpy  # noqa: F401
            return True
        except ImportError:
            return False

    def localise(self, stack, *, diameter=7, minmass=0.1, percentile=64,
                 workers=None, chunk_size=500, preview_cb=None, **_):
        if diameter % 2 == 0:
            diameter += 1

        n_frames = len(stack)
        n_chunks = max(1, int(np.ceil(n_frames / chunk_size)))
        workers  = max(1, min(workers if workers is not None else N_CPUS, N_CPUS))

        print(f"  Diameter  : {diameter}px  |  minmass: {minmass:.4f}")
        print(f"  Chunks    : {n_chunks} x ~{chunk_size} frames")

        t0       = time.perf_counter()
        # One canonical set of actual ranges drives every path: ordinary MP,
        # memmap MP, sequential fallback, localisation offsets, and previews.
        # This prevents a non-divisible stack (1001 frames / 3 chunks) from
        # acquiring different global frame starts in different branches.
        chunk_ranges = _balanced_chunk_ranges(n_frames, n_chunks)
        chunks = [stack[start:stop] for start, stop in chunk_ranges]
        chunk_pairs = [
            (chunk, start)
            for chunk, (start, _stop) in zip(chunks, chunk_ranges)
        ]

        # ── True multi-core via multiprocessing.Pool ──────────────────────
        # Each worker is a separate Python process with its own GIL — N workers
        # genuinely use N CPU cores.  Spawn context is required for Windows +
        # macOS frozen apps; PyInstaller's freeze_support (called in app_qt.py
        # main) makes spawn workers reuse the parent's _MEIPASS extraction, so
        # workers start in seconds rather than minutes.  Falls back to a
        # BLAS-pool serial path if Pool creation fails for any reason.
        n_workers = min(workers, n_chunks, N_CPUS)
        chunk_results = [None] * n_chunks
        use_mp_ok = False

        # Skip the multiprocessing.Pool path entirely for small jobs.
        # MP spawn on Windows can take >2 minutes (PyInstaller's
        # onefile bootloader extracts _MEIPASS into %TEMP% per worker;
        # with 6 workers all serialising the same ~200 MB extraction
        # past Defender's real-time scanner, observed 120 s in the
        # field).  Below the threshold, the BLAS-threaded single-
        # process path is strictly faster overall AND starts producing
        # per-chunk progress + previews immediately.
        #
        # Threshold of 16 chunks covers every typical PALM movie
        # (~8 k frames at chunk_size=500); past that, the MP spawn
        # cost is amortised over enough chunks to be worth paying.
        # Override with FIREFLY_FORCE_MP=1 if you're benchmarking.
        small_job = (n_chunks <= 16 and
                     os.environ.get("FIREFLY_FORCE_MP") != "1")
        if small_job:
            print(f"  Small job ({n_chunks} chunks) — using "
                  f"BLAS-threaded single-process path "
                  f"(skips ~10-30 s MP spawn cost; "
                  f"set FIREFLY_FORCE_MP=1 to override)")
            # use_mp_ok stays False → falls through to the sequential
            # BLAS-pool branch below.  Skip the whole `try` block.
            try_mp = False
        else:
            try_mp = True

        # Fast-path: if `stack` is a disk-backed memmap, ship just the
        # file path + slice indices to workers instead of pickling the
        # chunk arrays.  At 16+ GB stacks the pickle round-trip costs
        # ~5–15 s of launch latency AND temporarily doubles peak RAM
        # (parent's serialised bytes + worker's deserialised array).
        # Re-mmapping in workers is microseconds and they all share
        # the OS page cache.
        stack_is_memmap = isinstance(stack, np.memmap)
        memmap_path = None
        if stack_is_memmap:
            try:
                memmap_path = str(stack.filename)
                # Sanity: file must be readable from worker processes.
                if not os.path.isfile(memmap_path):
                    stack_is_memmap = False
            except Exception:
                stack_is_memmap = False

        try:
            if not try_mp:
                # Force-skip the MP path entirely — caller decided the
                # spawn cost isn't worth it for this job size.
                raise RuntimeError("small-job: MP path skipped by policy")
            ctx = multiprocessing.get_context("spawn")
            if stack_is_memmap:
                print(f"  Parallelism : multiprocessing.Pool × {n_workers} "
                      f"(spawn, memmap re-open in workers — zero-copy)")
            else:
                print(f"  Parallelism : multiprocessing.Pool × {n_workers} (spawn — true multi-core)")
            print(f"  Spawning workers (one-time ~10-30s; chunks then process truly in parallel)...")
            # Spawn-time heartbeat — Windows + PyInstaller can take 20+ s
            # to bring up the pool, during which nothing else prints.
            # Run a background thread that emits an elapsed-time line
            # every 3 s so the user knows we're not deadlocked.
            import threading as _threading
            _spawn_done = _threading.Event()
            _spawn_t0   = time.monotonic()
            def _spawn_heartbeat():
                while not _spawn_done.wait(3.0):
                    elapsed = time.monotonic() - _spawn_t0
                    print(f"  … still spawning workers ({elapsed:.0f}s)",
                          flush=True)
            _hb = _threading.Thread(target=_spawn_heartbeat, daemon=True,
                                     name="trackpy-spawn-heartbeat")
            _hb.start()

            if stack_is_memmap:
                dtype_str = str(stack.dtype)
                shape     = tuple(stack.shape)
                mp_args = [(i, memmap_path, dtype_str, shape,
                            start, end,
                            diameter, minmass, percentile, start)
                           for i, (start, end) in enumerate(chunk_ranges)]
                with ProcessPoolExecutor(
                        max_workers=safe_process_workers(n_workers),
                        mp_context=ctx) as pool:
                    # ProcessPoolExecutor (not mp.Pool): a worker that dies
                    # during bootstrap surfaces as BrokenProcessPool on
                    # fut.result() → the except below falls back to the serial
                    # BLAS path, instead of Pool.imap's silent forever-hang.
                    _spawn_announced = False
                    _futs = [pool.submit(_localise_chunk_mmap_mp, _a)
                             for _a in mp_args]
                    for _fut in _tqdm(
                            as_completed(_futs),
                            total=n_chunks, desc="  Localising", unit="chunk", ncols=70):
                        idx, result = _fut.result()
                        if not _spawn_announced:
                            _spawn_done.set()  # stop the heartbeat thread
                            print(f"  ✓ workers ready after "
                                  f"{time.monotonic() - _spawn_t0:.1f}s")
                            _spawn_announced = True
                        chunk_results[idx] = result
                        # Per-chunk live-preview emission.  The MP+memmap
                        # path doesn't hand the parent a frame buffer
                        # (workers re-mmap into their own address spaces),
                        # but the parent's `stack` is the SAME memmap, so
                        # we can read one representative frame straight
                        # out of it without re-decoding the TIFF.  Without
                        # this, the FAST RAM strategy on Trackpy left the
                        # detection-view blank for the whole localisation
                        # stretch (the STREAM path's _emit_chunk_previews
                        # was covering for it on Mac, hence the platform-
                        # specific user-visible regression).
                        _emit_trackpy_chunk_preview(
                            preview_cb, stack, chunk_ranges[idx], result,
                            n_frames)
            else:
                mp_args = [(i, c, diameter, minmass, percentile, o)
                           for i, (c, o) in enumerate(chunk_pairs)]
                with ProcessPoolExecutor(
                        max_workers=safe_process_workers(n_workers),
                        mp_context=ctx) as pool:
                    _spawn_announced = False
                    _futs = [pool.submit(_localise_chunk_mp, _a)
                             for _a in mp_args]
                    for _fut in _tqdm(
                            as_completed(_futs),
                            total=n_chunks, desc="  Localising", unit="chunk", ncols=70):
                        idx, result = _fut.result()
                        if not _spawn_announced:
                            _spawn_done.set()  # stop the heartbeat thread
                            print(f"  ✓ workers ready after "
                                  f"{time.monotonic() - _spawn_t0:.1f}s")
                            _spawn_announced = True
                        chunk_results[idx] = result
                        # Non-memmap MP path — the parent kept the chunk
                        # array in chunk_pairs[idx][0] so we can pass it
                        # straight through.  Same rationale as the memmap
                        # branch above.
                        _chunk_arr, _chunk_offset = chunk_pairs[idx]
                        _emit_trackpy_chunk_preview(
                            preview_cb, _chunk_arr,
                            (_chunk_offset, _chunk_offset + len(_chunk_arr)),
                            result, n_frames)
            use_mp_ok = True
        except Exception as exc:
            # Always release the heartbeat thread, even on error.
            try:    _spawn_done.set()
            except Exception: pass
            msg = str(exc)
            if "small-job" in msg:
                # Expected — we deliberately raised to skip MP.
                pass
            else:
                print(f"  multiprocessing failed ({type(exc).__name__}: {exc})")
                print(f"  Falling back to BLAS-pool parallelism (slower, single-process)")

        if not use_mp_ok:
            with _threadpool_limits(limits=_cpu_core_budget()):
                chunk_results = []
                for idx, (chunk, offset) in enumerate(_tqdm(
                        chunk_pairs, total=n_chunks,
                        desc="  Localising", unit="chunk", ncols=70)):
                    result = _localise_chunk(chunk, diameter, minmass,
                                              percentile, offset)
                    chunk_results.append(result)
                    # Sequential fallback also needs to emit previews —
                    # it's the slowest path of the three, so the user
                    # most needs to see it making progress.
                    _emit_trackpy_chunk_preview(
                        preview_cb, chunk,
                        (offset, offset + len(chunk)),
                        result, n_frames)

        valid = [df for df in chunk_results if df is not None and len(df) > 0]
        result = pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()

        elapsed = time.perf_counter() - t0
        print(f"  Found {len(result):,} localisations in {elapsed:.1f}s  "
              f"({n_frames / elapsed:.0f} frames/s)")
        return result


class TorchBackend(LocaliserBackend):
    """PyTorch-based localiser, calibrated to reproduce TrackpyBackend.

    Since v2.6.13 every algorithmic stage FOLLOWS trackpy's documented
    pipeline and the backend is CALIBRATED to agree with `tp.batch`
    end-to-end (~0.05 px median / count_ratio ≈ 1.0 per the calibration
    sweep; agreement is exercised by the suite, e.g.
    `tests/test_atrous_agrees_with_trackpy` and `tests/test_gaussian_mle.py`),
    so it is a GPU-accelerated near-drop-in replacement.  The agreement is
    empirical/aggregate, NOT
    step-identical — a few steps differ slightly from trackpy and are
    absorbed by the `_TP_MASS_SCALE` / `percentile` calibration (per-step
    notes inline below):

      1.  Bandpass — `gaussian(image, σ = noise_size ≈ 0.97)` minus
          `uniform_filter(image, smoothing_size = diameter + 1)`,
          clamped ≥ 0.  Follows `trackpy.preprocessing.bandpass`.
      2.  Threshold — the `percentile`-th percentile of the bandpassed
          image.  NOTE: taken over ALL pixels (incl. the bandpass-clamped
          zeros); trackpy takes it over nonzero pixels only
          (`image[np.nonzero(image)]`), so the same `percentile` gives a
          slightly different absolute cut (compensated by calibration).
      3.  Local maxima — pixels where the bandpassed signal equals
          its `diameter`-window max-pool output AND exceeds the
          threshold.  (trackpy excludes maxima within
          `separation = diameter + 1`; this `diameter`-window max-pool is
          a slightly tighter footprint.)
      4.  Sub-pixel refinement — iterative centroid-of-mass with a
          circular disk mask of radius `diameter/2`.  Each iteration
          shifts the integer centre by ±1 px when the centroid offset
          exceeds `shift_thresh` (≈0.61 px); loop terminates after at
          most `max_iters = 10` iterations.  Follows
          `trackpy.refine.refine`, except the disk mask uses radius
          `diameter/2` (r² ≤ (d/2)²) vs trackpy's integer `diameter//2`
          (r² ≤ (d//2)²) — a few extra corner pixels.
      5.  Mass — sum of bandpassed signal under the disk mask at the
          converged centre, then × `_TP_MASS_SCALE` to land on trackpy's
          `mass` scale (same definition as trackpy's `mass`, rescaled by
          the calibration constant).

    Calibration provenance
    ----------------------
    Two independent calibration runs (against TrackpyBackend on real
    sptPALM datasets at high and low spot density) confirmed:
      * 100 % recall — every trackpy detection has a Torch match within
        2 px.
      * Median centroid disagreement 0.05–0.10 px at 100 nm/px (i.e.
        5–10 nm), bounded by ≤ 0.13 px on sparse acquisitions.
      * Total spot count differs because Torch surfaces a small number
        of low-quality candidates that trackpy's intrinsic threshold
        rejects; these are filtered downstream by `min_track_len`,
        ROI masks, and the user's `minmass` and produce no scientific
        artifact.

    Constants live in the `_TP_*` class attributes below.  Change with
    care — the calibration target is median ≤ 0.20 px and recall ≥ 0.95 vs
    trackpy (re-derive with `tools/calibrate_torch_vs_trackpy.py`); the
    cross-backend agreement is exercised by the localiser tests in the suite
    (e.g. `tests/test_atrous_agrees_with_trackpy`, `tests/test_gaussian_mle.py`,
    `tests/test_loc_sigma_propagation.py`).

    Returns a DataFrame with the standard columns `x, y, frame, mass`.

    Frames are processed in chunks of `chunk_size` to bound peak GPU
    memory.  CPU performance is comparable to trackpy at default
    settings; MPS / CUDA paths give 5-10× speedup with identical
    numerical output (to within float32 BLAS noise).
    """
    name = "torch"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    @classmethod
    def list_devices(cls) -> list[str]:
        """Return all torch devices we could plausibly run on, fastest first.
        Used by the GUI to populate a device-override picker and by the
        crash reporter to record what was actually visible.
        """
        try:
            import torch
        except ImportError:
            return []
        devs: list[str] = []
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devs.append("mps")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devs.append(f"cuda:{i}" if torch.cuda.device_count() > 1 else "cuda")
        devs.append("cpu")
        return devs

    @classmethod
    def _device_sanity_check(cls, dev: str) -> bool:
        """Run the exact ops used in the hot path on `dev` to confirm full
        kernel coverage AND correctness.  Some PyTorch builds advertise MPS
        or CUDA support but either lack kernels for specific ops, or have
        kernels that silently return garbage (no Python exception raised).
        We test both.

        Metal native errors fire on C-level stderr — Python try/except
        can't catch those.  On MPS we run the probe inside an OS-level
        stderr redirect so a broken Metal context produces a clean
        "sanity check failed" message instead of flooding the terminal.
        """
        # OS-level stderr redirect (catches C / Metal native prints too).
        # Only used for MPS probing where we expect this class of noise.
        import contextlib as _cl

        @_cl.contextmanager
        def _quiet_native_stderr():
            devnull = os.open(os.devnull, os.O_WRONLY)
            saved   = os.dup(2)
            try:
                os.dup2(devnull, 2)
                yield
            finally:
                os.dup2(saved, 2)
                os.close(devnull)
                os.close(saved)

        ctx = _quiet_native_stderr() if dev == "mps" else _cl.nullcontext()
        try:
            with ctx:
                import torch
                import torch.nn.functional as F
                t = torch.device(dev)
                # 4×4 linear solve (same kernel as the Gaussian fit).  Use
                # an identity matrix and verify the result matches the
                # input — broken MPS can return garbage with no exception.
                A = torch.eye(4, device=t, dtype=torch.float32).unsqueeze(0)
                v = torch.ones(4, device=t, dtype=torch.float32).view(1, 4, 1)
                sol = torch.linalg.solve(A, v)
                if not torch.allclose(sol, v, rtol=1e-2, atol=1e-3):
                    return False
                # avg_pool2d (bandpass) and max_pool2d (local maxima)
                x = torch.zeros(1, 1, 8, 8, device=t, dtype=torch.float32)
                _ = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
                _ = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
                # conv2d — the separable Gaussian blur in the hot path.  This
                # is the exact op that raised cudaErrorNoKernelImageForDevice
                # on an unsupported GPU architecture (a Pascal GTX 1060 under a
                # CUDA-13 PyTorch build).  Probe it here so an unusable card is
                # caught up front instead of crashing mid-localisation.
                _w = torch.ones(1, 1, 1, 3, device=t, dtype=torch.float32)
                _ = F.conv2d(x, _w, padding=(0, 1))
                # einsum (used in normal-equations assembly)
                _ = torch.einsum('ni,ij,ik->njk',
                                 torch.ones(2, 4, device=t),
                                 torch.ones(4, 4, device=t),
                                 torch.ones(4, 4, device=t))
            return True
        except Exception as exc:
            print(f"  Device sanity check failed on {dev}: "
                  f"{type(exc).__name__}: {exc}")
            return False

    # Cached result of the device-selection sanity walk — recomputing it on
    # every chunk in the streaming path would add a few-ms penalty per call
    # for no information gain (hardware doesn't change mid-run).
    _cached_device: "str | None" = None

    @classmethod
    def select_device(cls) -> str:
        """Auto-pick the best device that actually works on this machine.

        Preference order: MPS (Apple Silicon) → CUDA (NVIDIA) → CPU.
        Each candidate goes through a self-test before we commit.  This
        prevents the analysis from picking MPS, running the bandpass + max-
        pool fine, then dying on `torch.linalg.solve` halfway through a
        16 000-frame stack.  Result is cached for the process lifetime.
        """
        if cls._cached_device is not None:
            return cls._cached_device
        for cand in cls.list_devices():
            if cls._device_sanity_check(cand):
                cls._cached_device = cand
                return cand
        cls._cached_device = "cpu"
        return "cpu"

    @staticmethod
    def _gaussian_blur(x, sigma, device):
        """Separable 1-D Gaussian blur via two conv1d-flavoured conv2d calls."""
        import torch
        import torch.nn.functional as F
        radius = max(1, int(round(3 * sigma)))
        kx = torch.arange(-radius, radius + 1, device=device, dtype=x.dtype)
        kernel_1d = torch.exp(-(kx ** 2) / (2 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        # (1, 1, 1, k) — horizontal
        kh = kernel_1d.view(1, 1, 1, -1)
        # (1, 1, k, 1) — vertical
        kv = kernel_1d.view(1, 1, -1, 1)
        x = F.conv2d(x, kh, padding=(0, radius))
        x = F.conv2d(x, kv, padding=(radius, 0))
        return x

    @staticmethod
    def _build_gaussian_design_matrix(dy_grid, dx_grid):
        """Precompute the (k², 4) design matrix and its pseudo-inverse for the
        log-Gaussian linear least-squares fit.

        Model:   log(I) = a + b·x + c·y + p·(x² + y²)
                 where  p = -1/(2σ²),  b = -2·x₀·p,  c = -2·y₀·p
                 ⇒    x₀ = -b/(2p),   y₀ = -c/(2p)

        M is identical for every spot (only depends on the patch geometry),
        so we precompute its pseudo-inverse once and reuse it as a batched
        matrix-multiply per chunk.  Cost: a single (N, k²) @ (k², 4) gemm.
        """
        import torch
        x_flat = dx_grid.reshape(-1)
        y_flat = dy_grid.reshape(-1)
        ones   = torch.ones_like(x_flat)
        M = torch.stack([ones, x_flat, y_flat, x_flat**2 + y_flat**2], dim=1)
        # Pseudoinverse: M_pinv = (MᵀM)⁻¹Mᵀ  — shape (4, k²)
        M_pinv = torch.linalg.pinv(M)
        return M, M_pinv

    @staticmethod
    def _gaussian_lstsq_refine(patches, dy_grid, dx_grid, M):
        """Batched analytical 2D-Gaussian fit on patches via the *normal
        equations* of a weighted log-linearisation.

        Why normal equations and not `torch.linalg.lstsq`?
        --------------------------------------------------
        `torch.linalg.lstsq` is NOT implemented on the MPS device in current
        PyTorch builds (it raises NotImplementedError for `aten::linalg_lstsq.out`).
        `torch.linalg.solve` is — and for full-rank weighted least-squares,
        solving the 4×4 normal equations `(MᵀWᵀWM) b = MᵀWᵀW y` gives the
        identical answer.  The reformulation buys us cross-device support
        (CPU, CUDA, MPS) at the cost of a slightly higher condition number,
        which is irrelevant for the well-posed 4-parameter Gaussian fit.

        Why weighted?
        -------------
        Unweighted log-space LSQ gives every pixel — including dim, noisy
        edge pixels — equal influence on the centroid.  This inflates per-
        spot variance, which manifests as a depressed MSD α (because
        MSD = MSD_true + 4σ²_loc; higher σ_loc flattens the apparent log-log
        slope at short lags).  Weighting each pixel by √I (Poisson-likelihood
        weighting in log-space) means bright spot-centre pixels dominate the
        fit, restoring centroid-of-mass-like noise behaviour while preserving
        the unbiased mean-position accuracy of the Gaussian fit.

        Math
        ----
        Model:    log(I) = a + b·x + c·y + p·(x² + y²)            (linear in params)
        Weights:  w² = I       ⇒  weighted residual = √I · (a + b·x + c·y + p·(x²+y²) − log(I))
        Normal eq: A b = v,   A = MᵀWᵀWM = Σᵢ Iᵢ·MᵢMᵢᵀ,   v = MᵀWᵀWy = Σᵢ Iᵢ·log(Iᵢ)·Mᵢ
        Recover:  x₀ = −b/(2p),   y₀ = −c/(2p),   σ² = −1/(2p)

        Inputs
        ------
        patches : (N, k, k) float tensor — non-negative pixel intensities
        dy_grid : (k, k)    float tensor — y offsets relative to patch centre
        dx_grid : (k, k)    float tensor — x offsets relative to patch centre
        M       : (k², 4)   float tensor — design matrix [1, x, y, x²+y²]

        Returns (dy_sub, dx_sub, ok) where:
          dy_sub, dx_sub : (N,) sub-pixel offsets relative to the patch centre
          ok             : (N,) bool mask — True for spots whose fit is valid
        """
        import torch
        N, k, _ = patches.shape
        eps = 1e-6
        I_flat = patches.clamp(min=eps).reshape(N, k * k)          # (N, k²)
        Y_log  = torch.log(I_flat)                                  # (N, k²)

        # Normal equations: per-spot A is (4, 4); per-spot v is (4,)
        # A[n, j, k] = Σᵢ I[n, i] · M[i, j] · M[i, k]
        # v[n, j]    = Σᵢ I[n, i] · log(I[n, i]) · M[i, j]
        A = torch.einsum('ni,ij,ik->njk', I_flat, M, M)             # (N, 4, 4)
        v = torch.einsum('ni,ij->nj', I_flat * Y_log, M)            # (N, 4)

        # Tikhonov-style ridge for numerical conditioning on near-flat patches.
        # 1e-6 * trace(A) per spot is small enough not to bias real spots but
        # keeps degenerate ones from blowing up the solver.
        ridge = 1e-6 * torch.diagonal(A, dim1=1, dim2=2).mean(dim=1)
        eye   = torch.eye(4, device=A.device, dtype=A.dtype)
        A = A + ridge.view(-1, 1, 1) * eye.unsqueeze(0)

        # Solve N independent 4×4 systems.  `torch.linalg.solve` is supported
        # on CPU / CUDA / MPS — unlike `lstsq` which lacks MPS coverage.
        try:
            sol = torch.linalg.solve(A, v.unsqueeze(-1)).squeeze(-1)   # (N, 4)
        except (NotImplementedError, RuntimeError) as exc:
            # Final belt-and-braces fallback: shuttle to CPU.  Should never
            # trigger in normal operation, but it means a single missing
            # kernel won't kill the run.
            print(f"  [TorchBackend] linalg.solve fallback to CPU: {exc}")
            sol = torch.linalg.solve(A.cpu(),
                                     v.unsqueeze(-1).cpu()).squeeze(-1).to(A.device)

        a, b, c, p = sol.unbind(dim=1)
        # Guard against degenerate fits: p must be negative (peak, not pit)
        safe_p = torch.where(p < -1e-8, p, torch.full_like(p, -1e-8))
        dx_sub = -b / (2.0 * safe_p)
        dy_sub = -c / (2.0 * safe_p)
        # Reject fits whose centroid lies well outside the patch — clamping to
        # ≤ 1.5 px keeps spurious "edge wins" from leaking through.  A real
        # spot's Gaussian fit lands within ±0.5 px of the integer maximum.
        ok = (p < -1e-8) & (dx_sub.abs() <= 1.5) & (dy_sub.abs() <= 1.5)
        return dy_sub, dx_sub, ok

    # ── Trackpy-compatibility constants ───────────────────────────────────
    # These knobs are calibrated to make the Torch backend reproduce
    # `tp.batch(image, diameter, minmass, percentile)` as closely as possible.
    # They mirror trackpy's `bandpass` defaults and its `refine.refine`
    # iteration policy.  Tunable via the calibration script in
    # tools/calibrate_torch_vs_trackpy.py — DO NOT change ad-hoc; the
    # values here are the result of an offline parameter sweep against a
    # reference trackpy run on a real sptPALM stack.
    #
    # bandpass:
    #   noise_size     = σ for the high-pass Gaussian (matches tp default = 1.0).
    #   smoothing_size = box-filter size for the slow-background subtract
    #                    (trackpy default ≈ diameter + 1).  Stored as an
    #                    additive offset so it scales with the user's chosen
    #                    diameter.
    # refinement:
    #   refine_max_iters    = trackpy's `max_iterations` (default 10).
    #   refine_shift_thresh = trackpy's `shift_thresh` (default 0.6 px) —
    #                         offsets above this trigger an integer recentre.
    # mass scale:
    #   _TP_MASS_SCALE = multiplier applied to Torch's mass column so it
    #     lives on the same numerical scale as Trackpy's mass column.
    #     Both backends compute mass as "sum of bandpassed intensity
    #     over a disk mask of radius diameter/2", but the bandpass
    #     implementations differ at the float-precision level (Torch
    #     uses float32 separable conv; Trackpy uses scipy.ndimage on
    #     float64) and the two return mass values on slightly different
    #     scales.  Without this scaling, the GUI's `minmass` slider
    #     would mean different things to the two backends — e.g.
    #     minmass=1.35 keeps ~3.2 k spots in Trackpy but ~4.9 k spots
    #     in Torch on the same stack.
    #
    #     Empirically (from `tools/calibrate_torch_vs_trackpy.py
    #     --match-spot-count` on two real sptPALM stacks):
    #       Calibration.tif (dense):   matched threshold = 2.216 at
    #                                  user minmass = 1.35  ⇒ ratio 1.642
    #       Calibration2.tif (sparse): matched threshold = 2.356 at
    #                                  user minmass = 1.35  ⇒ ratio 1.745
    #
    #     For the same spot count, Torch needs a HIGHER threshold than
    #     Trackpy → Torch's mass values are LARGER than Trackpy's.  To
    #     bring Torch's masses down onto Trackpy's scale we MULTIPLY by
    #     the reciprocal of the average ratio ≈ 1/1.7 ≈ 0.588.  This
    #     puts the GUI's `minmass` slider on a unified scale: 1.35 in
    #     either backend keeps roughly the same population of spots.
    #
    #     Re-derive by running `--match-spot-count` on a few
    #     representative datasets, averaging the matched / user minmass
    #     ratios, and setting `_TP_MASS_SCALE = 1 / mean_ratio`.
    # Re-calibrated 2026-05-27 against Calibration3.tif (4019 frames,
    # diameter=7, minmass=1.0, percentile=64) via Nelder-Mead sweep over
    # 60 evaluations on a 1350-frame training set with a 150-frame
    # hold-out.  Optimised θ brought training count_ratio from 0.985 →
    # 1.000 and hold-out count_ratio to 0.993, with median centroid
    # disagreement essentially unchanged (0.0548 → 0.0531 px on hold-out,
    # i.e. 5.3 nm at 100 nm/px).  Previous (as-shipped) defaults shown
    # alongside for reference.
    _TP_NOISE_SIZE             = 0.9659  # was 1.0
    _TP_SMOOTHING_SIZE_OFFSET  = 1     # smoothing_size = diameter + offset
    _TP_REFINE_MAX_ITERS       = 10
    _TP_REFINE_SHIFT_THRESH    = 0.6093  # was 0.6
    _TP_MASS_SCALE             = 0.588   # = 1 / 1.70  (Trackpy-scale)

    @staticmethod
    def _trackpy_bandpass(x, diameter, device, dtype):
        """Trackpy-compatible bandpass:
            response = gaussian(image, σ=noise_size)
                       - uniform_filter(image, size=smoothing_size)
            response = max(response, 0)

        Matches `trackpy.preprocessing.bandpass` (which is itself wrapped
        by `tp.locate` before any local-maximum search).  Returns a tensor
        of the same shape as `x`.

        Notes
        -----
        * Trackpy uses `scipy.ndimage.gaussian_filter` with
          `truncate=4` by default; the existing `_gaussian_blur` truncates
          at 3σ, which removes ≤0.27 % of the Gaussian's tail integral —
          well below the noise floor and confirmed not to shift sub-pixel
          centroids by more than 1 e-3 px on the calibration set.
        * The uniform_filter is replicated with `F.avg_pool2d` at
          kernel = smoothing_size.  Trackpy uses a separable boxcar; for
          odd sizes the result is identical up to numerical precision.
        """
        import torch
        import torch.nn.functional as F
        smoothing_size = int(
            diameter + TorchBackend._TP_SMOOTHING_SIZE_OFFSET)
        if smoothing_size % 2 == 0:
            smoothing_size += 1   # uniform_filter expects an odd kernel
        # High-pass: small-σ Gaussian smoothes out shot noise.
        smooth = TorchBackend._gaussian_blur(
            x, sigma=TorchBackend._TP_NOISE_SIZE, device=device)
        # Slow background: boxcar of `smoothing_size`.  pad = (size-1)//2
        # to keep the output spatially aligned with `smooth`.
        pad = (smoothing_size - 1) // 2
        bg = F.avg_pool2d(x, kernel_size=smoothing_size,
                          stride=1, padding=pad)
        response = (smooth - bg).clamp(min=0.0)
        # Snap sub-noise-floor residuals to EXACT zero.  In flat background the
        # high-pass `smooth - bg` should land at 0; CPU/CUDA float32 round it
        # there, but MPS leaves tiny positive residuals (~1e-6).  Detection is
        # `(signal == maxpool) & (signal > threshold)`, and when the percentile
        # threshold is ~0 (a mostly-empty bandpass — e.g. a sparse frame) those
        # residuals slip past `signal > 0` AND tie their max-pool neighbours,
        # flooding detection with hundreds of phantom maxima on Apple Silicon
        # (some survive `minmass` as spurious low-mass spots — see
        # tests/test_gaussian_mle.py).  A floor at 32·eps·max(response) sits
        # orders of magnitude above the subtraction's rounding noise yet far
        # below any real spot, so it erases the MPS phantoms while leaving
        # CPU/CUDA detection byte-for-byte unchanged (verified: 0 candidate
        # changes on real noisy frames).  Device-agnostic by design — one path,
        # no per-device branch to drift — so the latent phantom also can't
        # reappear in the plain TorchBackend on MPS.
        floor = 32.0 * torch.finfo(dtype).eps * response.amax()
        return torch.where(response > floor, response,
                           torch.zeros((), device=device, dtype=dtype))

    def _detection_map(self, x, signal, diameter, device, dtype):
        """The image whose local maxima are the spot candidates.

        TorchBackend detects directly on the bandpassed `signal`; subclasses
        override this to detect on a different response (e.g. à trous wavelet
        planes) while refinement and mass still operate on `signal`."""
        return signal

    def _detection_threshold(self, dmap, percentile):
        """Detection threshold: the `percentile`-th percentile of the detection
        map.  (Taken over ALL pixels incl. the bandpass-clamped zeros; trackpy
        uses nonzero pixels only — calibrated to agree, not step-identical.)

        torch.quantile is exact for small inputs; for big tensors subsample to
        bound memory.  Use a DETERMINISTIC evenly-spaced stride rather than an
        unseeded random draw: (a) re-running the same file yields identical
        detections (the old torch.randint made dense-data spot counts vary
        run-to-run), and (b) an evenly-spaced sample is a lower-variance,
        frame-grouping-stable estimator of the background percentile — so the
        threshold barely moves with the batch size, which is what keeps
        GPU-batched localisation detection-neutral."""
        import torch
        flat = dmap.reshape(-1)
        if flat.numel() > 5_000_000:
            step = max(1, int(flat.numel() // 5_000_000))
            sample = flat[::step][:5_000_000]
            return torch.quantile(sample, percentile / 100.0)
        return torch.quantile(flat, percentile / 100.0)

    @staticmethod
    def _circular_mask(diameter, device, dtype):
        """Boolean / float disk-mask of side k = diameter, used to integrate
        signal over a spot's footprint.  Pixels inside radius `diameter/2`
        of the patch centre are 1; outside are 0.  Close to (not identical to)
        the geometry `tp.refine.refine` uses: radius is `diameter/2` here vs
        trackpy's integer `diameter//2`, so a few corner pixels differ — the
        difference is absorbed by the `_TP_MASS_SCALE` calibration."""
        import torch
        k = int(diameter)
        r = (k - 1) / 2.0
        ys = torch.arange(k, device=device, dtype=dtype) - r
        xs = torch.arange(k, device=device, dtype=dtype) - r
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        rad = (k / 2.0)
        return (Y * Y + X * X) <= (rad * rad)

    @staticmethod
    def _drop_coincident_duplicates(df):
        """Drop coincident duplicate localisations (same frame, position rounded
        to 0.01 px), keeping one.

        A spot centred exactly between two pixels gives a FLAT-TOP detection map,
        so two tied adjacent maxima both pass ``dmap == maxpool`` and refine to the
        SAME sub-pixel point — one molecule counted twice.  Real (noisy) spots
        break the tie, so this only fires on degenerate symmetry; two genuinely
        distinct spots can never share a 0.01-px position.  Used by the refinement
        backends (à trous / Gaussian-MLE / radial symmetry); plain TorchBackend
        keeps its long-calibrated raw behaviour."""
        if df is not None and len(df) > 1 and {"frame", "x", "y"} <= set(df.columns):
            dup = df.assign(
                _f=df["frame"].astype("int64"),
                _x=(df["x"] * 100).round().astype("int64"),
                _y=(df["y"] * 100).round().astype("int64"),
            ).duplicated(subset=["_f", "_x", "_y"], keep="first")
            if bool(dup.any()):
                df = df.loc[~dup].reset_index(drop=True)
        return df

    def _refine_peaks(self, signal, t_ix, y_ix, x_ix, diameter, *,
                      device, dtype, raw=None, characterize=False):
        """Sub-pixel refinement HOOK — swap the refiner without touching the
        ~400-line ``localise`` loop.

        Given the integer spot centres ``(y_ix, x_ix)`` at frames ``t_ix``, the
        bandpassed ``signal`` (T, 1, Y, X) and the matching ``raw`` (un-bandpassed)
        chunk, return the refined positions and mass.  TorchBackend's default IS
        the trackpy-compatible iterative centroid-of-mass refiner (on ``signal``);
        subclasses (GaussianMle / RadialSymmetry) override this to swap ONLY the
        sub-pixel step while reusing detection, chunking, the minmass filter, the
        disk-mask geometry and the preview emission unchanged — exactly how
        ``AtrousWaveletBackend`` overrides only ``_detection_map`` /
        ``_detection_threshold``.

        ``raw`` is supplied so a fit-based refiner can work on the un-bandpassed
        image under the correct (Poisson) noise model — the bandpass is a matched
        filter that makes centroid-of-mass near-optimal but distorts the spot
        profile, so a Gaussian MLE only out-performs centroid when it fits the
        RAW pixels.  Refiners that don't need it (centroid, radial symmetry)
        ignore ``raw``.

        Contract (every override MUST honour it):
          * returns the 6-tuple ``(dy, dx, y_final, x_final, mass, char)`` with
            the SAME shapes/dtypes ``_iterative_centroid_refine`` produces;
          * ``mass`` is ALREADY on the trackpy scale (× ``_TP_MASS_SCALE``), so
            ``minmass`` (in ``localise``) and the auto-threshold harvest in
            ``estimate_minmass`` keep meaning the same thing across backends;
          * ``char`` is ``None`` or ``{"size", "ecc"}`` (N,) tensors when
            ``characterize`` — the PSF-quality gate the harvest feeds.
        """
        return self._iterative_centroid_refine(
            signal, t_ix, y_ix, x_ix, diameter,
            device=device, dtype=dtype, characterize=characterize)

    def _refine_off_mps(self, impl, signal, t_ix, y_ix, x_ix, diameter, *,
                        device, dtype, raw=None, characterize=False):
        """Run a sub-pixel refiner ``impl`` on CPU when ``device`` is MPS, moving
        the inputs to CPU and the results back to ``device``.

        Apple's MPS backend intermittently returns SILENTLY WRONG results for the
        linear-algebra / small-kernel ops the Gaussian-MLE (``linalg.solve`` /
        ``linalg.inv`` / ``einsum``) and radial-symmetry (``conv2d`` on tiny
        patches) refiners use — the "kernel returns garbage with no exception"
        failure class ``_device_sanity_check`` warns about, which the refiners'
        own ``try/except`` CPU fallbacks can't catch because nothing is raised.
        It surfaced as the gaussian-mle / radial-symmetry engines occasionally
        mis-localising (wrong spot count / position) on Apple GPUs while passing
        on CPU.  Detection (bandpass conv + max-pool over full frames) is reliable
        on MPS and stays GPU-accelerated; only the cheap per-spot refinement
        (small kxk patches) is forced onto CPU, so the cost is negligible.  CUDA
        is reliable for these ops, so it runs ``impl`` on-device unchanged."""
        import torch
        if getattr(device, "type", str(device)) != "mps":
            return impl(signal, t_ix, y_ix, x_ix, diameter,
                        device=device, dtype=dtype, raw=raw,
                        characterize=characterize)
        cpu = torch.device("cpu")
        dy, dx, cy, cx, mass, char = impl(
            signal.to(cpu), t_ix.to(cpu), y_ix.to(cpu), x_ix.to(cpu), diameter,
            device=cpu, dtype=dtype,
            raw=(raw.to(cpu) if raw is not None else None),
            characterize=characterize)
        _back = lambda v: v.to(device) if torch.is_tensor(v) else v
        if char is not None:
            char = {k: _back(v) for k, v in char.items()}
        return (_back(dy), _back(dx), _back(cy), _back(cx), _back(mass), char)

    @staticmethod
    def _iterative_centroid_refine(signal, t_ix, y_ix, x_ix, diameter,
                                     device, dtype,
                                     max_iters=None, shift_thresh=None,
                                     characterize=False):
        """Vectorised iterative centroid-of-mass refinement — Torch
        reimplementation calibrated to agree with `trackpy.refine.refine`
        (the disk-mask radius differs slightly; see `_circular_mask`).

        Per spot:
            1. Extract a (k × k) patch from `signal` centred at integer
               (y, x).  Apply a circular disk mask of radius `k/2`.
            2. Compute mass-weighted centroid offsets (dy, dx) relative
               to the patch centre.
            3. If max(|dy|, |dx|) > shift_thresh, shift the integer
               centre by sign(dy) / sign(dx) (clamped to image bounds)
               and go back to step 1.
            4. Else converged; sub-pixel position = (y + dy, x + dx).

        Vectorised:
            All N spots are advanced in lockstep.  Converged spots are
            kept around but their integer centre stops shifting, so they
            cost only a handful of GPU ops per remaining iteration —
            cheaper than tracking "active" subsets and re-permuting.

        Returns
        -------
        dy_sub, dx_sub : (N,) float — sub-pixel offsets relative to the
                         spot's FINAL integer centre.
        final_y, final_x : (N,) int — the converged integer centres
                         (may differ from the input if the spot drifted).
        mass : (N,) float — sum of masked signal at the final centre,
                            same definition trackpy uses for its `mass`
                            column.
        char : dict or None — when `characterize=True`, ``{"size", "ecc"}``
                            (N,) tensors: trackpy-compatible radius of
                            gyration and eccentricity from the final patch,
                            so a torch-native auto-threshold harvest can drive
                            the same PSF-quality gate that trackpy's
                            ``characterize=True`` feeds.  None otherwise.
        """
        import torch
        if max_iters is None:
            max_iters = TorchBackend._TP_REFINE_MAX_ITERS
        if shift_thresh is None:
            shift_thresh = TorchBackend._TP_REFINE_SHIFT_THRESH

        k = int(diameter)
        r = k // 2
        # Patch-relative offset grids, (k, k) — broadcast against patches.
        dy_grid, dx_grid = torch.meshgrid(
            torch.arange(-r, r + 1, device=device, dtype=dtype),
            torch.arange(-r, r + 1, device=device, dtype=dtype),
            indexing="ij")
        # Circular disk mask, (k, k) — 1 inside the spot footprint.
        mask = TorchBackend._circular_mask(diameter, device, dtype).to(dtype)

        # Shape of the image: signal is (T, 1, Y, X)
        _, _, H, W = signal.shape

        cur_y = y_ix.clone()
        cur_x = x_ix.clone()
        # `done` flags spots that should stop shifting their integer
        # centre.  They still get refined sub-pixel each iteration so we
        # can extract the final centroid value, but the integer shift
        # short-circuits to zero.
        done = torch.zeros_like(cur_y, dtype=torch.bool)

        dy_sub = torch.zeros_like(cur_y, dtype=dtype)
        dx_sub = torch.zeros_like(cur_x, dtype=dtype)

        for _ in range(int(max_iters)):
            # Patch indices.  Clamp to keep us inside the image even if a
            # bright spot near the edge wants to shift further.
            ys = (cur_y[:, None, None] + dy_grid.long()[None]).clamp_(
                min=0, max=H - 1)
            xs = (cur_x[:, None, None] + dx_grid.long()[None]).clamp_(
                min=0, max=W - 1)
            ts = t_ix[:, None, None].expand_as(ys)
            patches = signal[ts, 0, ys, xs]                 # (N, k, k)
            masked = patches * mask                          # (N, k, k)
            mass = masked.sum(dim=(1, 2)).clamp(min=1e-6)    # (N,)
            dy_now = (masked * dy_grid[None]).sum(dim=(1, 2)) / mass
            dx_now = (masked * dx_grid[None]).sum(dim=(1, 2)) / mass
            dy_sub, dx_sub = dy_now, dx_now

            # Integer shift only where not done AND offset is large.
            shift_y = torch.where(
                done, torch.zeros_like(dy_now),
                torch.where(dy_now >  shift_thresh, torch.ones_like(dy_now),
                torch.where(dy_now < -shift_thresh, -torch.ones_like(dy_now),
                                                     torch.zeros_like(dy_now))))
            shift_x = torch.where(
                done, torch.zeros_like(dx_now),
                torch.where(dx_now >  shift_thresh, torch.ones_like(dx_now),
                torch.where(dx_now < -shift_thresh, -torch.ones_like(dx_now),
                                                     torch.zeros_like(dx_now))))

            # Spots whose centre didn't move this iteration are converged.
            no_shift = (shift_y == 0) & (shift_x == 0)
            done = done | no_shift

            cur_y = (cur_y + shift_y.long()).clamp_(min=r, max=H - 1 - r)
            cur_x = (cur_x + shift_x.long()).clamp_(min=r, max=W - 1 - r)

            if bool(done.all()):
                break

        # Final mass at the converged centre with the disk mask applied.
        ys = (cur_y[:, None, None] + dy_grid.long()[None]).clamp_(
            min=0, max=H - 1)
        xs = (cur_x[:, None, None] + dx_grid.long()[None]).clamp_(
            min=0, max=W - 1)
        ts = t_ix[:, None, None].expand_as(ys)
        final_patches = signal[ts, 0, ys, xs] * mask
        final_mass = final_patches.sum(dim=(1, 2))

        # Rescale onto Trackpy's mass scale so the GUI's `minmass`
        # slider means the same thing across both backends.  See
        # the `_TP_MASS_SCALE` docstring on TorchBackend for the
        # derivation and why this is multiplicative (not additive).
        final_mass = final_mass * float(TorchBackend._TP_MASS_SCALE)

        char = None
        if characterize:
            char = TorchBackend._characterize_patches(
                final_patches, dy_grid, dx_grid, r)

        return dy_sub, dx_sub, cur_y, cur_x, final_mass, char

    @staticmethod
    def _characterize_patches(patches, dy_grid, dx_grid, r):
        """Trackpy-compatible ``size`` (radius of gyration) and ``ecc``
        (eccentricity) from the disk-masked patches at the converged centre.

        Mirrors ``trackpy.refine.center_of_mass`` so the values land on the
        same scale the PSF-quality gate's bands were tuned against (``size`` in
        the diffraction-limited 0.5–``diameter`` px band; ``ecc`` flagged above
        0.9).  ``patches`` are the SAME disk-masked bandpassed neighbourhoods
        the ``mass`` column is summed over, so a torch-native harvest gates on a
        self-consistent characterisation — no trackpy ``characterize=True`` call.

        ``size`` = sqrt(Σ r²·I / Σ I);  ``ecc`` = |Σ I·e^{2iθ}| / (Σ I − I₀),
        with r/θ the patch-centre-relative radius/angle and I₀ the centre pixel.
        """
        import torch
        px = patches                                   # (N, k, k), masked
        psum = px.sum(dim=(1, 2)).clamp(min=1e-6)      # (N,)
        r2 = (dy_grid * dy_grid + dx_grid * dx_grid)   # (k, k)
        # The bandpassed patch is signed, so a noise candidate's outer ring can
        # drive the second moment negative; clamp the radicand so `size` stays a
        # finite, non-negative radius of gyration (trackpy works on the
        # non-negative raw image and never hits this).
        size = torch.sqrt(((px * r2[None]).sum(dim=(1, 2)) / psum).clamp(min=0.0))
        theta = torch.atan2(dy_grid, dx_grid)          # (k, k)
        cos2 = torch.cos(2.0 * theta)[None]
        sin2 = torch.sin(2.0 * theta)[None]
        ecc_num = torch.sqrt((px * cos2).sum(dim=(1, 2)) ** 2 +
                             (px * sin2).sum(dim=(1, 2)) ** 2)
        ri = int(r)                                    # centre pixel index
        px_center = px[:, ri, ri]
        ecc = ecc_num / (psum - px_center).clamp(min=1e-6)
        return {"size": size, "ecc": ecc}

    def localise(self, stack, *, diameter=7, minmass=0.1, percentile=64,
                 workers=None, chunk_size=500, preview_cb=None,
                 device=None, quiet=False, _cpu_threads=None,
                 characterize=False, **_):
        # `_cpu_threads` overrides the CPU intra-op thread budget.  It's set by
        # `_torch_localise_block_mp` so each multi-process worker uses a small
        # slice of the cores (N_workers × threads ≈ N_CPUS) instead of every
        # worker grabbing all of them; left None, the serial/streaming path
        # claims all cores as before.
        import torch
        import torch.nn.functional as F

        if diameter % 2 == 0:
            diameter += 1
        radius = diameter // 2
        k = diameter

        # Resolve device: explicit `device=` arg > `_forced_device` set by
        # the 'torch-mps'/'torch-cuda'/'torch-cpu' GUI pins > auto-select.
        dev_str = (device
                   or getattr(self, "_forced_device", None)
                   or self.select_device())
        # Final safety net: if a GPU device was forced/pinned upstream without
        # a sanity check (or the hardware/driver changed), verify it once and
        # fall back to CPU rather than dying on the first kernel launch.  The
        # result is cached on the instance so the per-chunk localise() calls
        # don't re-probe (which would relaunch a failing CUDA kernel each chunk).
        if dev_str != "cpu":
            committed = getattr(self, "_validated_device", None)
            if committed is None:
                if self._device_sanity_check(dev_str):
                    committed = dev_str
                else:
                    print(f"  WARNING: Torch device '{dev_str}' is not usable on "
                          f"this machine (no kernel image / unsupported GPU "
                          f"architecture). Falling back to the Torch backend on "
                          f"CPU.")
                    committed = "cpu"
                self._validated_device = committed
            dev_str = committed
        dev     = torch.device(dev_str)
        # Float32 is plenty for centroid math; saves memory on GPUs and
        # avoids dtype gotchas with MPS (which dislikes float64).
        dtype = torch.float32

        # See note in preprocess_and_localise_stream re: why we don't bump
        # chunk_size on GPU — Apple Silicon is bandwidth-limited, not
        # dispatch-limited, so the caller's chunk_size (typically 500) is
        # actually optimal.  Honour it as passed.
        n_frames = len(stack)
        n_chunks = max(1, int(np.ceil(n_frames / chunk_size)))

        # CPU torch is single-threaded by default in this codebase because
        # the module-level `OMP_NUM_THREADS=1` cap (added in ba20dd0 to
        # prevent a Windows trackpy MP deadlock) propagates into ATen's
        # OpenMP pool.  Explicitly re-expand torch's intra-op threads
        # back to N_CPUS when running on CPU — without this, torch-cpu
        # crawls at ~11 fr/s on a 6-core box (380 s for 4 k frames)
        # instead of utilising all cores like the trackpy backend does
        # via threadpoolctl.  GPU devices ignore these settings.  When this is
        # an MP worker (`_cpu_threads` set) use the smaller per-worker slice so
        # N_workers × threads ≈ N_CPUS instead of every worker grabbing them all.
        _cpu_nthreads = int(_cpu_threads) if _cpu_threads else _cpu_core_budget()
        if dev_str == "cpu":
            try:    torch.set_num_threads(_cpu_nthreads)
            except Exception: pass
            # Inter-op (parallel *independent* ops) is NOT exploited by this
            # sequential per-chunk pipeline — a big interop pool just sits idle.
            # Intra-op (set_num_threads above) is the one that matters; keep
            # interop at 1.
            try:    torch.set_num_interop_threads(1)
            except (RuntimeError, Exception):
                # Errors if any parallel work has already been dispatched on
                # this interpreter — harmless, the first-call count is what wins.
                pass

        if not quiet:
            print(f"  Device    : {dev_str}")
            print(f"  Diameter  : {diameter}px  |  minmass: {minmass:.4f}  "
                  f"|  percentile: {percentile}")
            print(f"  Chunks    : {n_chunks} × ~{chunk_size} frames")

        # ── CPU multi-process fan-out ───────────────────────────────────────
        # Single-process torch-CPU leans on intra-op threads, which don't scale
        # for the small per-frame ops on many-core / NUMA boxes (~6% CPU on a
        # 128-core EPYC).  At the top level (not a streaming sub-chunk, not
        # already an MP worker) with enough chunks to amortise spawn cost, fan
        # the chunk-aligned blocks across processes instead.  The per-chunk
        # percentile threshold is batch-size-stable and the serial loop already
        # thresholds per-chunk, so chunk-aligned blocks reproduce the serial
        # detections exactly.  Disable with FIREFLY_TORCH_CPU_MP=0.
        _use_cpu_mp = (
            dev_str == "cpu" and not quiet and _cpu_threads is None
            and not characterize        # characterize path is serial-only
            and n_chunks > 16
            and os.environ.get("FIREFLY_TORCH_CPU_MP", "1").strip().lower()
            not in ("0", "false", "no", "off"))

        if not quiet and dev_str == "cpu" and not _use_cpu_mp:
            try:    print(f"  Torch threads : {torch.get_num_threads()}")
            except Exception: pass
            # Serial torch-CPU (small job, or MP disabled) is single-process.
            print("  NOTE: Torch backend on CPU, single-process. For CPU-only "
                  "machines, 'Auto' or 'Trackpy (CPU)' use all cores and are "
                  "usually faster.", flush=True)

        if _use_cpu_mp:
            mp_df = self._localise_cpu_parallel(
                stack, diameter=diameter, minmass=minmass,
                percentile=percentile, chunk_size=chunk_size,
                n_chunks=n_chunks, n_frames=n_frames, preview_cb=preview_cb)
            if mp_df is not None:
                return mp_df
            print("  Torch-CPU parallel path unavailable — running serial.",
                  flush=True)

        t0 = time.perf_counter()
        all_locs: list[dict] = []

        # Refinement now uses the trackpy-compatible iterative
        # centroid-of-mass path inside `_iterative_centroid_refine`,
        # which builds its own per-call (k × k) offset grids on-device.
        # The old `dy_grid` / `dx_grid` / `_M` / `_M_pinv` block (used
        # by the deprecated `_gaussian_lstsq_refine`) is no longer
        # needed at this scope.

        # Enter the BLAS thread-pool expansion BEFORE the chunk loop and
        # exit it after — same trick the trackpy path uses to claw back
        # cores from the `OMP_NUM_THREADS=1` module-level cap.  We use
        # the controller's explicit __enter__ / __exit__ so we don't
        # have to re-indent the (huge) loop body inside a `with`.
        # `torch.set_num_threads` above already biased ATen, but
        # OpenBLAS / MKL still honour OMP — the matmul / lstsq inside
        # `_gaussian_lstsq_refine` is the dominant cost and reads from
        # the BLAS pool.
        _blas_ctx = (
            _threadpool_limits(limits=_cpu_nthreads) if dev_str == "cpu" else None
        )
        if _blas_ctx is not None:
            try:    _blas_ctx.__enter__()
            except Exception: _blas_ctx = None

        # Detection is pure inference — no autograd graph is ever needed — so
        # wrap the whole chunk loop in inference_mode (skips view/version
        # tracking + trims memory; numerically identical).  Device-agnostic;
        # entered via __enter__/__exit__ to avoid re-indenting the big loop.
        _infer_ctx = None
        try:
            _infer_ctx = torch.inference_mode()
            _infer_ctx.__enter__()
        except Exception:
            _infer_ctx = None

        # Per-chunk timing so the live log shows progress.  Historically
        # the Torch chunk loop emitted nothing between the up-front
        # `Chunks: N × ~M frames` line and the final "Found … in …s"
        # — on Windows torch-cpu where a chunk takes ~30 s, the
        # console looked frozen for the entire localisation stretch
        # even though the analysis was running fine.  Print one line
        # per chunk with elapsed time + spot count so the user can
        # see steady forward motion.
        chunk_t0_outer = time.perf_counter()
        last_chunk_end_t = chunk_t0_outer
        # In streaming mode this backend is invoked once per CPU sub-chunk, so
        # its own per-chunk progress lines would duplicate the streaming loop's
        # tqdm bar hundreds of times.  `quiet` routes them to a no-op.
        _plog = (lambda *a, **k: None) if quiet else print
        if not quiet:
            print(f"  Starting localisation: {n_chunks} chunks of "
                  f"~{chunk_size} frames each "
                  f"(progress logged per-chunk below)", flush=True)

        for chunk_idx, chunk_start in enumerate(range(0, n_frames, chunk_size)):
            chunk_end = min(chunk_start + chunk_size, n_frames)
            chunk_np  = np.asarray(stack[chunk_start:chunk_end], dtype=np.float32)

            # (T, 1, Y, X)
            x = torch.from_numpy(chunk_np).to(dev, dtype=dtype).unsqueeze(1)
            T, _, Y, X = x.shape

            # ── 1. Bandpass — trackpy-compatible ─────────────────────────────
            # response = gaussian(x, σ=noise_size) - uniform_filter(x, smoothing_size)
            # Matches `trackpy.preprocessing.bandpass` (which is what tp.locate
            # internally feeds to its local-maxima detector).  Replaces the
            # earlier "avg_pool background subtract + small Gaussian smooth"
            # path which produced subtly different bandpassed magnitudes →
            # different percentile thresholds → different spot counts.
            signal = self._trackpy_bandpass(x, diameter, dev, dtype)

            # Detection map: the image whose local maxima are spot candidates.
            # TorchBackend detects on the bandpassed `signal` itself; subclasses
            # (à trous) override `_detection_map` to detect on a different
            # response while refinement and mass still use `signal`.
            dmap = self._detection_map(x, signal, diameter, dev, dtype)

            # ── 2. Detection threshold per chunk ────────────────────────────
            # `_detection_threshold` is the percentile-of-bandpass rule for
            # TorchBackend (deterministic subsample → batch-size-stable, the
            # invariant GPU batching relies on); subclasses override it (à trous
            # uses a MAD noise floor on its sparse wavelet-product map).
            threshold = self._detection_threshold(dmap, percentile)

            # ── 3. Local maxima via max-pool == self ────────────────────────
            maxp   = F.max_pool2d(dmap, kernel_size=k, stride=1, padding=radius)
            is_max = (dmap == maxp) & (dmap > threshold)
            # nonzero → (N, 4) columns: (t, c, y, x)
            coords = is_max.nonzero(as_tuple=False)
            if coords.numel() == 0:
                _ct = time.perf_counter()
                _plog(f"  Chunk {chunk_idx+1}/{n_chunks} "
                      f"(frames {chunk_start}–{chunk_end-1}): 0 spots "
                      f"in {_ct - last_chunk_end_t:.1f}s "
                      f"(no maxima above threshold)", flush=True)
                last_chunk_end_t = _ct
                continue

            # Drop maxima too close to the edge to extract a full patch
            edge_ok = (
                (coords[:, 2] >= radius) & (coords[:, 2] < Y - radius) &
                (coords[:, 3] >= radius) & (coords[:, 3] < X - radius)
            )
            coords = coords[edge_ok]
            if coords.numel() == 0:
                _ct = time.perf_counter()
                _plog(f"  Chunk {chunk_idx+1}/{n_chunks} "
                      f"(frames {chunk_start}–{chunk_end-1}): 0 spots "
                      f"in {_ct - last_chunk_end_t:.1f}s "
                      f"(all maxima edge-rejected)", flush=True)
                last_chunk_end_t = _ct
                continue

            t_ix = coords[:, 0]
            y_ix = coords[:, 2]
            x_ix = coords[:, 3]

            # ── 4. Iterative centroid-of-mass refinement (trackpy-compat) ──
            # Matches `trackpy.refine.refine` step-for-step: extract a
            # disk-masked patch, compute mass-weighted centroid offset,
            # shift the integer centre by ±1 px until the offset settles
            # under `shift_thresh`, then take the final sub-pixel offset.
            # Replaces the earlier single-pass Gaussian-LSQ fit, which
            # was the main source of the historical ~0.30 px median
            # disagreement against trackpy on the agreement test.
            # The refinement also returns `mass` summed over the disk
            # mask AT THE CONVERGED CENTRE — same definition trackpy
            # uses for its `mass` column.
            # `_refine_peaks` is the swappable refinement HOOK: TorchBackend's
            # default is the centroid-of-mass refiner above; GaussianMle /
            # RadialSymmetry subclasses override it to sharpen the sub-pixel
            # position while keeping `mass` on the same trackpy scale.
            dy_off, dx_off, y_final, x_final, mass, char = self._refine_peaks(
                signal, t_ix, y_ix, x_ix, diameter,
                device=dev, dtype=dtype, raw=x, characterize=characterize)

            # Apply the minmass filter AFTER refinement (trackpy filters
            # on the refined mass, not the pre-refinement patch sum).
            keep = mass >= minmass
            if not bool(keep.any()):
                _ct = time.perf_counter()
                _plog(f"  Chunk {chunk_idx+1}/{n_chunks} "
                      f"(frames {chunk_start}–{chunk_end-1}): 0 spots "
                      f"in {_ct - last_chunk_end_t:.1f}s "
                      f"(all below minmass={minmass:.2f})", flush=True)
                last_chunk_end_t = _ct
                continue
            dy_off  = dy_off[keep]
            dx_off  = dx_off[keep]
            y_final = y_final[keep]
            x_final = x_final[keep]
            mass    = mass[keep]
            t_ix    = t_ix[keep]

            x_sub = x_final.to(dtype) + dx_off
            y_sub = y_final.to(dtype) + dy_off
            frame_abs = (t_ix + chunk_start).to(torch.int64)

            loc = {
                "x":     x_sub.detach().cpu().numpy(),
                "y":     y_sub.detach().cpu().numpy(),
                "frame": frame_abs.detach().cpu().numpy(),
                "mass":  mass.detach().cpu().numpy(),
            }
            if char is not None:                        # torch-native harvest
                if "size" in char:
                    loc["size"] = char["size"][keep].detach().cpu().numpy()
                    loc["ecc"]  = char["ecc"][keep].detach().cpu().numpy()
                if "loc_sigma_x_px" in char:            # MLE Fisher-info CRLB (px)
                    loc["loc_sigma_x_px"] = (
                        char["loc_sigma_x_px"][keep].detach().cpu().numpy())
                    loc["loc_sigma_y_px"] = (
                        char["loc_sigma_y_px"][keep].detach().cpu().numpy())
            all_locs.append(loc)

            # ── Live preview emission ─────────────────────────────────
            # Historically the TorchBackend accepted `preview_cb` but
            # never called it — so on the Windows torch-cpu path the
            # detection view sat blank for the entire localisation
            # stage.  Now we emit one preview per frame in the chunk,
            # with the spots that landed in that frame overlaid; the
            # GUI's pump thread throttles to 60 Hz and drops older
            # frames if the queue fills, so over-emission is harmless.
            #
            # Done on CPU AFTER the GPU tensors have already been
            # materialised into `all_locs` — `t_np` and the coords
            # arrays are cheap reads from the existing CPU buffers.
            if preview_cb is not None and len(chunk_np) > 0:
                try:
                    import numpy as _np
                    t_np      = t_ix.detach().cpu().numpy().astype(_np.int64)
                    x_sub_np  = x_sub.detach().cpu().numpy()
                    y_sub_np  = y_sub.detach().cpu().numpy()
                    # Group spots by their frame index within the chunk
                    # so each preview_cb call hands the GUI just the
                    # detections for that frame.  Using a dict-of-lists
                    # is O(N) and avoids re-scanning per frame.
                    spots_by_frame: dict = {}
                    for _i, _f in enumerate(t_np):
                        bucket = spots_by_frame.setdefault(int(_f), [[], []])
                        bucket[0].append(float(x_sub_np[_i]))
                        bucket[1].append(float(y_sub_np[_i]))
                    chunk_len = int(chunk_np.shape[0])
                    for local_i in range(chunk_len):
                        global_i = chunk_start + local_i
                        sxy = spots_by_frame.get(local_i, ([], []))
                        try:
                            preview_cb(global_i, chunk_np[local_i],
                                       sxy[0], sxy[1], n_frames)
                        except Exception:
                            pass
                except Exception:
                    # Preview emission must never break the analysis.
                    pass

            # Per-chunk progress log — success path.  Includes spot
            # count + wall-clock time for the chunk so the user can
            # spot if any one chunk takes much longer than the rest
            # (memory pressure forcing a swap, for example).
            try:
                _ct = time.perf_counter()
                _n_spots = int(mass.numel())
                _avg_fps = (chunk_end - chunk_start) / max(1e-3,
                                                              _ct - last_chunk_end_t)
                _plog(f"  Chunk {chunk_idx+1}/{n_chunks} "
                      f"(frames {chunk_start}–{chunk_end-1}): "
                      f"{_n_spots:,} spots in "
                      f"{_ct - last_chunk_end_t:.1f}s "
                      f"({_avg_fps:.0f} fr/s)", flush=True)
                last_chunk_end_t = _ct
            except Exception:
                pass

            # Free chunk allocations promptly.  PyTorch's reference-counting
            # releases the Python handles, but on MPS the underlying device
            # memory isn't actually returned until queued command buffers
            # complete.  Sequence here:
            #   1. del Python handles
            #   2. synchronize: wait for the device's command queue to drain
            #   3. empty_cache: release the pool back to the system
            # Without the synchronize, mps.empty_cache() returns immediately
            # and the memory stays committed — which on a 16 GB unified
            # M-series machine can starve downstream stages (matplotlib
            # rendering, Qt repaint) of GPU memory and produce confusing
            # OOM errors that look unrelated to the localisation step.
            # `bg` and `patches` no longer exist in this loop (folded
            # into `_trackpy_bandpass` and `_iterative_centroid_refine`
            # respectively); freeing the survivors is still worth doing
            # to keep MPS's allocator from holding stale chunk memory.
            del x, signal, maxp, is_max, coords
            # MPS only: on Apple Silicon's UNIFIED memory the device pool isn't
            # returned until the command queue drains, so without this per-chunk
            # synchronize+empty_cache a long localisation can starve downstream
            # Qt/matplotlib of GPU memory.  On discrete CUDA VRAM the opposite is
            # true: calling empty_cache() every chunk forces the caching
            # allocator to hand blocks back to the driver, so the next chunk
            # re-allocates from scratch — a large, well-documented throughput
            # hit (≈1000 calls/file in streaming mode).  CUDA only needs ONE
            # drain at the end of the call (below), so skip it per chunk here.
            if dev_str == "mps":
                try:
                    if hasattr(torch.mps, "synchronize"):
                        torch.mps.synchronize()
                    if hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
                except Exception:
                    pass

        # Release the BLAS thread-pool expansion (matched __enter__ above).
        # Outside this scope the global OMP=1 cap reasserts itself so the
        # downstream linker / preview pump don't get oversubscribed.
        if _blas_ctx is not None:
            try:    _blas_ctx.__exit__(None, None, None)
            except Exception: pass
        if _infer_ctx is not None:
            try:    _infer_ctx.__exit__(None, None, None)
            except Exception: pass

        # Force a full GPU drain before returning.  Otherwise the next
        # CPU-only stage (linking) inherits a degraded MPS context — its
        # finalizers run when Python GC kicks in during link_trajectories
        # and produce "command buffer exited with error" OOM messages that
        # have nothing to do with the actual cause.
        # (No per-call grid tensors to del here anymore — the refinement
        # path allocates its grids inside `_iterative_centroid_refine`
        # and they drop out of scope when that helper returns.)
        if dev_str == "mps":
            try:
                if hasattr(torch.mps, "synchronize"):
                    torch.mps.synchronize()
                if hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            except Exception:
                pass
        elif dev_str.startswith("cuda"):
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass

        out_cols = ["x", "y", "frame", "mass"]
        if characterize:
            out_cols += ["size", "ecc"]
        if not all_locs:
            _plog("  Found 0 localisations")
            return pd.DataFrame(columns=out_cols)
        # Per-spot precision columns (px) flow through only when a refiner emits
        # them (GaussianMleBackend's Fisher-info CRLB); the worker converts → nm.
        if "loc_sigma_x_px" in all_locs[0]:
            out_cols += ["loc_sigma_x_px", "loc_sigma_y_px"]

        df = pd.DataFrame({
            col: np.concatenate([d[col] for d in all_locs])
            for col in out_cols
        })

        elapsed = time.perf_counter() - t0
        _plog(f"  Found {len(df):,} localisations in {elapsed:.1f}s  "
              f"({n_frames / elapsed:.0f} frames/s)")
        return df

    def _localise_cpu_parallel(self, stack, *, diameter, minmass, percentile,
                               chunk_size, n_chunks, n_frames, preview_cb=None):
        """Localise on CPU torch across PROCESSES — one chunk-aligned block per
        worker — and return the same ``DataFrame[x,y,frame,mass]`` the serial
        path returns, or ``None`` to fall back to the serial loop.

        Each block is a contiguous run of whole ``chunk_size`` chunks, so the
        per-chunk percentile thresholds (and therefore the detections) are
        identical to the serial path; concatenating blocks in order reproduces
        the serial row order exactly.

        Preview callbacks can't cross process boundaries, so per-frame detection
        previews are skipped here (same tradeoff as the trackpy MP path); coarse
        per-block progress is logged instead.
        """
        import threading
        try:
            from multiprocessing import shared_memory
        except Exception:
            shared_memory = None

        # ── Worker / thread budget (env-tunable for big boxes) ──────────────
        try:
            env_w = int(os.environ.get("FIREFLY_TORCH_CPU_WORKERS", "0"))
        except ValueError:
            env_w = 0
        budget = _cpu_core_budget()          # HYPERFLY: this file's core slice
        n_workers = env_w if env_w > 0 else min(n_chunks, budget, 32)
        n_workers = max(1, min(n_workers, n_chunks, budget))
        if n_workers < 2:
            return None                      # nothing to parallelise
        threads_per_worker = max(1, budget // n_workers)

        # ── Partition chunk starts into contiguous, chunk-aligned blocks ────
        chunk_starts = list(range(0, n_frames, chunk_size))      # len == n_chunks
        groups = [g for g in np.array_split(np.arange(len(chunk_starts)),
                                            n_workers) if len(g)]
        blocks = []
        for g in groups:
            fs = chunk_starts[int(g[0])]
            fe = min(chunk_starts[int(g[-1])] + chunk_size, n_frames)
            blocks.append((int(fs), int(fe)))

        # ── Stage the stack for zero-copy worker access ─────────────────────
        shm = None
        try:
            if (isinstance(stack, np.memmap)
                    and getattr(stack, "filename", None)
                    and os.path.isfile(str(stack.filename))):
                base = ("memmap", str(stack.filename), str(stack.dtype),
                        tuple(stack.shape))
                source_kind = "memmap"
            elif isinstance(stack, np.ndarray):
                if shared_memory is None:
                    return None
                arr = np.ascontiguousarray(stack, dtype=np.float32)
                shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
                buf = np.ndarray(arr.shape, dtype=np.float32, buffer=shm.buf)
                buf[:] = arr
                base = ("shm", shm.name, str(np.dtype(np.float32)),
                        tuple(arr.shape))
                source_kind = "shm"
                del arr
            else:
                return None                  # lazy/unknown stack → serial
        except Exception:
            if shm is not None:
                try:    shm.close(); shm.unlink()
                except Exception: pass
            return None

        args = [
            (idx, base + (fs, fe), diameter, minmass, percentile, chunk_size,
             threads_per_worker, fs)
            for idx, (fs, fe) in enumerate(blocks)
        ]

        print(f"  Parallelism : {n_workers} processes × {threads_per_worker} "
              f"torch threads (CPU {source_kind}; chunk-aligned blocks — "
              f"identical to serial)", flush=True)
        print(f"  Spawning {n_workers} workers (one-time ~10–30s; blocks then "
              f"localise truly in parallel)...", flush=True)

        results: dict = {}
        t0 = time.perf_counter()
        _done = threading.Event()

        def _heartbeat():
            while not _done.wait(3.0):
                print(f"    … working ({time.perf_counter() - t0:.0f}s, "
                      f"{len(results)}/{len(blocks)} blocks done)", flush=True)

        hb = threading.Thread(target=_heartbeat, daemon=True)
        try:
            ctx = multiprocessing.get_context("spawn")
            # ProcessPoolExecutor — NOT multiprocessing.Pool: it surfaces a
            # dead/un-bootstrappable worker as a BrokenProcessPool exception so
            # we fall back to serial, instead of Pool.imap's silent forever-hang.
            # safe_process_workers keeps it under the Windows 61-worker cap
            # (n_workers ≤ 32 here anyway).
            hb.start()
            with ProcessPoolExecutor(max_workers=safe_process_workers(n_workers),
                                     mp_context=ctx) as ex:
                # ex.map preserves input (block) order.
                for idx, df in ex.map(_torch_localise_block_mp, args):
                    results[idx] = df
                    _n = 0 if df is None else len(df)
                    print(f"  Block {idx + 1}/{len(blocks)}: {_n:,} spots "
                          f"({len(results)}/{len(blocks)} done, "
                          f"{time.perf_counter() - t0:.0f}s)", flush=True)
        except Exception as _e:
            print(f"  Torch-CPU parallel path failed ({_e!r}); "
                  f"falling back to serial.", flush=True)
            return None
        finally:
            _done.set()
            if shm is not None:
                try:    shm.close()
                except Exception: pass
                try:    shm.unlink()
                except Exception: pass

        # A failed/missing block → fall back rather than silently drop spots.
        if len(results) != len(blocks) or any(
                results.get(i) is None for i in range(len(blocks))):
            return None

        parts = [results[i] for i in range(len(blocks)) if len(results[i])]
        if not parts:
            return pd.DataFrame(columns=["x", "y", "frame", "mass"])
        df = pd.concat(parts, ignore_index=True)
        elapsed = time.perf_counter() - t0
        print(f"  Found {len(df):,} localisations in {elapsed:.1f}s  "
              f"({n_frames / max(1e-6, elapsed):.0f} frames/s, "
              f"{n_workers}-way CPU)", flush=True)
        return df


class AtrousWaveletBackend(TorchBackend):
    """À trous (undecimated B3-spline) wavelet spot detector.

    A classic low-SNR single-molecule detector (Olivo-Marin et al. 2002):
    smooth the frame with a B3-spline kernel at successive dilations ("holes"),
    take successive-smoothing differences as wavelet planes, and detect spots as
    local maxima of the PRODUCT of the first few planes — which reinforces
    spot-sized structure while suppressing noise uncorrelated across scales.

    Implemented as a thin override of TorchBackend: ONLY the detection map and
    its threshold change.  Device handling, chunking, max-pool maxima, sub-pixel
    centroid refinement and mass all reuse TorchBackend unchanged — refinement
    and mass run on the bandpassed ``signal``, so the ``mass`` column stays on
    the trackpy scale and ``minmass`` means the same thing across all backends.

    NOTE (deviation from canonical Olivo-Marin): the wavelet transform here runs
    on the already-bandpassed ``signal`` (the raw frame ``x`` is available but
    intentionally unused), not the raw image.  This double-filters slightly, but
    keeps detection on the same bandpassed image the refiner/mass use so the
    cross-backend mass-scale parity above holds; ``_ATROUS_K_SIGMA`` was
    calibrated in exactly this configuration.

    ``_ATROUS_K_SIGMA`` (detection sensitivity) is calibrated for count parity
    against the Crocker–Grier detectors on synthetic ground truth — see the
    constant's provenance comment below.
    """
    name = "atrous"

    # Number of à trous wavelet planes whose significant parts form the map.
    _ATROUS_N_SCALES = 3
    # Detection sensitivity: per wavelet plane, keep coefficients above
    # ``median + _ATROUS_K_SIGMA · σ`` (σ = 1.4826·MAD of the plane), then
    # multiply the significant planes.  Calibrated against synthetic ground
    # truth (scripts/calibrate_atrous.py, 2026-06-15): k=2.0 maximised mean F1
    # = 0.86 (recall 0.80, precision 0.95) over SNR 2–8 × density 20/60 spots —
    # markedly more precise than trackpy (F1 0.41) on the same noisy stacks.
    # Higher k → higher precision, lower recall.  RE-VALIDATE on real data.
    _ATROUS_K_SIGMA = 2.0
    # B3-spline à trous kernel (1D; applied separably as rows then columns).
    _B3_KERNEL = (1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16)

    def _detection_map(self, x, signal, diameter, device, dtype):
        """Product of the first ``_ATROUS_N_SCALES`` à trous wavelet planes,
        each thresholded at its OWN per-frame noise floor.

        Each wavelet plane ``W_i = A_i − A_{i+1}`` is dense (signed), so a robust
        per-frame ``σ = 1.4826·MAD(W_i)`` is a well-defined noise estimate — the
        RAW product's MAD is degenerate (it's mostly exact zeros, so median and
        MAD are both 0 and the threshold can't depend on k).  Keeping only
        coefficients above ``k·σ`` per plane, then multiplying, suppresses noise
        (uncorrelated across scales) and reinforces spot-sized structure;
        ``_ATROUS_K_SIGMA`` is the detection sensitivity."""
        import torch
        import torch.nn.functional as F
        kvec = torch.tensor(self._B3_KERNEL, device=device, dtype=dtype)
        kx = kvec.view(1, 1, 1, -1)
        ky = kvec.view(1, 1, -1, 1)

        def _smooth(img, level):
            d = 2 ** level                  # à trous dilation ("holes")
            pad = 2 * d                     # kernel radius (2) × dilation
            img = F.conv2d(img, kx, padding=(0, pad), dilation=(1, d))
            img = F.conv2d(img, ky, padding=(pad, 0), dilation=(d, 1))
            return img

        a = signal
        corr = None
        for level in range(self._ATROUS_N_SCALES):
            a_next = _smooth(a, level)
            plane = a - a_next                          # wavelet plane (dense, ±)
            flat = plane.reshape(plane.shape[0], -1)    # per-frame robust stats
            med = flat.median(dim=1, keepdim=True).values
            mad = (flat - med).abs().median(dim=1, keepdim=True).values
            thr = (med + self._ATROUS_K_SIGMA * (1.4826 * mad)).view(-1, 1, 1, 1)
            sig = F.relu(plane - thr)                   # significant positive part
            corr = sig if corr is None else corr * sig
            a = a_next
        return corr

    def _detection_threshold(self, dmap, percentile):
        """The wavelet planes are already significance-thresholded per scale in
        ``_detection_map`` (that is where ``_ATROUS_K_SIGMA`` acts), so any
        positive product pixel is a candidate and the shared max-pool picks the
        local maxima.  ``percentile`` is unused for this backend."""
        return dmap.new_tensor(0.0)

    def localise(self, stack, **kwargs):
        """TorchBackend.localise with the shared coincident-duplicate guard — the
        wavelet-product map flat-tops on perfectly symmetric spots, so two tied
        maxima can refine to the same point (see
        ``_drop_coincident_duplicates``)."""
        return self._drop_coincident_duplicates(super().localise(stack, **kwargs))

    def _localise_cpu_parallel(self, *args, **kwargs):
        # The torch-CPU multi-process worker (`_torch_localise_block_mp`)
        # hard-codes ``TorchBackend()``, so it would run TORCH detection, not à
        # trous.  Force the (correct) serial path on CPU until a backend-aware MP
        # worker exists; the GPU path is single-process and unaffected.
        return None


class GaussianMleBackend(TorchBackend):
    """Crocker–Grier detection with a 2-D **Gaussian maximum-likelihood**
    sub-pixel refiner instead of centroid-of-mass.

    Detection is byte-for-byte ``TorchBackend`` (trackpy-compatible bandpass →
    percentile threshold → max-pool maxima).  Only the sub-pixel step changes:
    each detected spot is polished by a few Newton–Raphson iterations of a
    symmetric 2-D Gaussian fit under a Poisson noise model (the
    Cramér–Rao-optimal estimator for shot-noise-limited spots; Smith et al.,
    *Nat. Methods* 2010), seeded from the centroid result and falling back to it
    whenever a fit is non-finite or lands outside the patch.  (It does NOT
    compare the fitted likelihood/position against the centroid seed, so an
    in-patch-but-worse fit is still accepted — "never worse than centroid" holds
    only for the divergent/out-of-patch cases the fallback guards.)

    The fit runs on the **raw** (un-bandpassed) pixels: the bandpass is a matched
    filter that already makes centroid-of-mass near-optimal but distorts the spot
    profile, so a Gaussian MLE only out-performs centroid when it fits the raw,
    ~Poisson image.  Be honest about the size of that win: on FIREFLY's typical
    sparse, well-bandpassed spots centroid-of-mass is already close to the CRLB,
    so the gain is modest — benchmarked WITHIN ~2 % of centroid RMSE on the real
    EPFL/SPT datasets, exact on noiseless synthetic spots, marginally better at low
    SNR, and occasionally marginally WORSE on dense / overlapping fields (the
    single-Gaussian model is pulled by a neighbour; a divergent or out-of-patch fit
    still falls back to the centroid seed).  The big MLE advantage in the
    literature needs high background or non-Gaussian PSFs, not isolated spots.
    ``mass`` is still the centroid-of-mass mass on the trackpy scale, so
    ``minmass`` and the auto-threshold harvest mean exactly what they do for the
    other backends — switching engine sharpens positions on clean spots without
    shifting which spots survive a given threshold.

    Implemented as a thin ``TorchBackend`` subclass overriding ONLY
    ``_refine_peaks`` (+ the CPU-MP guard), enabled by the ``_refine_peaks`` hook.
    """
    name = "gaussian-mle"

    # Newton–Raphson refiner knobs.  Converges in a handful of steps on real
    # spots; the per-step position clamp + per-spot centroid fallback bound the
    # worst case.  σ is FITTED per spot (a 5-parameter fit x0,y0,A,b,σ) so the
    # model self-calibrates to the PSF width — a wrong fixed σ biases the position
    # estimate — clamped to a physical band; ``_MLE_SIGMA_FRAC·diameter`` only
    # seeds it.  Set ``_MLE_FIT_SIGMA = False`` to hold σ fixed (4-param).
    _MLE_MAX_ITERS  = 8
    _MLE_STEP_CLAMP = 1.0     # px — max |Δposition| per Newton step (damping)
    _MLE_CONV_TOL   = 1e-3    # px — stop when the position step falls below this
    _MLE_FIT_SIGMA  = True    # fit σ too (self-calibrate to the PSF width)
    _MLE_SIGMA_FRAC = 0.25    # σ seed = _MLE_SIGMA_FRAC · diameter

    def localise(self, stack, **kwargs):
        """TorchBackend.localise + the shared coincident-duplicate guard: a
        half-pixel-centred spot detects twice (flat-top maxima) and both refine to
        the same point — drop the duplicate (see ``_drop_coincident_duplicates``)."""
        return self._drop_coincident_duplicates(super().localise(stack, **kwargs))

    def _refine_peaks(self, signal, t_ix, y_ix, x_ix, diameter, *,
                      device, dtype, raw=None, characterize=False):
        # Run the Newton/Fisher fit (linalg.solve / linalg.inv / einsum) on CPU
        # when on MPS — those ops are silently unreliable there; see
        # TorchBackend._refine_off_mps.
        return self._refine_off_mps(
            self._refine_peaks_impl, signal, t_ix, y_ix, x_ix, diameter,
            device=device, dtype=dtype, raw=raw, characterize=characterize)

    def _refine_peaks_impl(self, signal, t_ix, y_ix, x_ix, diameter, *,
                           device, dtype, raw=None, characterize=False):
        import torch
        # 1. Seed from the centroid refiner (on the bandpassed signal): the
        #    converged integer centre (cy,cx), the sub-pixel seed (dy0,dx0), the
        #    trackpy-scaled mass and the size/ecc characterisation are ALL reused.
        dy0, dx0, cy, cx, mass, char = self._iterative_centroid_refine(
            signal, t_ix, y_ix, x_ix, diameter,
            device=device, dtype=dtype, characterize=characterize)
        if cy.numel() == 0:
            return dy0, dx0, cy, cx, mass, char

        # Fit the RAW pixels (the bandpass distorts the profile and inflates the
        # fit's bias); fall back to the bandpassed signal if no raw was threaded.
        src = raw if raw is not None else signal

        k = int(diameter)
        r = k // 2
        eps = 1e-6
        dy_grid, dx_grid = torch.meshgrid(
            torch.arange(-r, r + 1, device=device, dtype=dtype),
            torch.arange(-r, r + 1, device=device, dtype=dtype),
            indexing="ij")                               # (k, k)
        mask = TorchBackend._circular_mask(diameter, device, dtype).to(dtype)
        _, _, H, W = src.shape
        ys = (cy[:, None, None] + dy_grid.long()[None]).clamp_(0, H - 1)
        xs = (cx[:, None, None] + dx_grid.long()[None]).clamp_(0, W - 1)
        ts = t_ix[:, None, None].expand_as(ys)
        n = (src[ts, 0, ys, xs] * mask).clamp(min=0.0)        # (N, k, k) raw counts

        # 2. Newton–Raphson on the Poisson NLL with the Fisher (Gauss–Newton)
        #    Hessian — PSD by construction (no negative-curvature blow-ups),
        #    first-derivatives only, gives the CRLB covariance as H⁻¹ for free.
        #    Params: x0, y0, A, b (+ σ when _MLE_FIT_SIGMA).
        fit_sigma = bool(self._MLE_FIT_SIGMA)
        npar = 5 if fit_sigma else 4
        x0 = dx0.clone()
        y0 = dy0.clone()
        N = x0.shape[0]
        b = n.amin(dim=(1, 2))
        A = (n.amax(dim=(1, 2)) - b).clamp(min=1e-3)
        s = torch.full((N,), max(0.75, float(self._MLE_SIGMA_FRAC) * diameter),
                       device=device, dtype=dtype)
        s_lo, s_hi = 0.5, max(1.0, diameter / 2.0)
        eyep = torch.eye(npar, device=device, dtype=dtype)
        for _ in range(int(self._MLE_MAX_ITERS)):
            s2 = (s * s)[:, None, None]
            ddx = dx_grid[None] - x0[:, None, None]           # (N, k, k)
            ddy = dy_grid[None] - y0[:, None, None]
            rr = ddx * ddx + ddy * ddy
            E = torch.exp(-rr / (2.0 * s2))
            mu = (A[:, None, None] * E + b[:, None, None]).clamp(min=eps)
            rw = (1.0 - n / mu) * mask                         # ∂NLL/∂mu, masked
            w = mask / mu                                      # Fisher weight
            AE = A[:, None, None] * E
            # ∂mu/∂x0 = A·E·(xi−x0)/σ² ; ∂mu/∂A = E ; ∂mu/∂b = 1
            cols = [AE * ddx / s2, AE * ddy / s2, E, torch.ones_like(E)]
            if fit_sigma:                                      # ∂mu/∂σ = A·E·r²/σ³
                cols.append(AE * rr / (s2 * s[:, None, None]))
            J = torch.stack(cols, dim=-1)                      # (N, k, k, npar)
            g = (rw[..., None] * J).sum(dim=(1, 2))            # (N, npar)
            Hm = torch.einsum('nyxp,nyxq->npq', w[..., None] * J, J)
            ridge = 1e-6 * torch.diagonal(Hm, dim1=1, dim2=2).mean(dim=1)
            Hm = Hm + ridge.view(-1, 1, 1) * eyep
            try:
                step = torch.linalg.solve(Hm, g.unsqueeze(-1)).squeeze(-1)
            except (NotImplementedError, RuntimeError):
                step = torch.linalg.solve(
                    Hm.cpu(), g.unsqueeze(-1).cpu()).squeeze(-1).to(device)
            sxy = step[:, :2].clamp(-self._MLE_STEP_CLAMP, self._MLE_STEP_CLAMP)
            x0 = x0 - sxy[:, 0]
            y0 = y0 - sxy[:, 1]
            A = (A - step[:, 2]).clamp(min=1e-3)
            b = b - step[:, 3]
            if fit_sigma:
                s = (s - step[:, 4].clamp(-0.5, 0.5)).clamp(s_lo, s_hi)
            if float(sxy.abs().amax()) < self._MLE_CONV_TOL:
                break

        # 3. Per-spot accept/fallback: a non-finite or out-of-patch fit reverts
        #    to the centroid seed (guards divergence only — an accepted in-patch
        #    fit is NOT compared against the centroid seed).
        ok = (torch.isfinite(x0) & torch.isfinite(y0)
              & (x0.abs() <= r) & (y0.abs() <= r))
        dx_off = torch.where(ok, x0, dx0)
        dy_off = torch.where(ok, y0, dy0)

        # 4. Per-spot localisation precision = CRLB from the Fisher (Gauss–Newton)
        #    Hessian at the optimum: Cov = H⁻¹, σ_x = √Cov[x0,x0], σ_y = √Cov[y0,y0]
        #    in PIXELS — essentially free, Hm was already built/factorised each
        #    Newton step.  CAVEAT: the Poisson Fisher info assumes the patch is in
        #    PHOTONS, but the fit runs on raw ADU, so this σ is a CRLB under a
        #    gain≈1 assumption; the worker rescales by 1/√gain when camera
        #    calibration is supplied and converts px→nm.  Rejected fits (centroid
        #    fallback) and non-PSD/degenerate Hessians get NaN, never a bogus CRLB.
        try:
            cov = torch.linalg.inv(Hm)
        except (NotImplementedError, RuntimeError):
            cov = torch.linalg.inv(Hm.cpu()).to(device)
        var_x = cov[:, 0, 0]
        var_y = cov[:, 1, 1]
        nan = torch.full((), float("nan"), device=device, dtype=dtype)
        bad = ((~ok) | ~torch.isfinite(var_x) | ~torch.isfinite(var_y)
               | (var_x <= 0) | (var_y <= 0))
        sx = torch.where(bad, nan, torch.sqrt(var_x.clamp(min=0.0)))
        sy = torch.where(bad, nan, torch.sqrt(var_y.clamp(min=0.0)))
        if char is None:
            char = {}
        char["loc_sigma_x_px"] = sx
        char["loc_sigma_y_px"] = sy
        return dy_off, dx_off, cy, cx, mass, char

    def _localise_cpu_parallel(self, *args, **kwargs):
        # The CPU multi-process worker hard-codes ``TorchBackend()`` (centroid
        # refinement), so it would silently bypass the MLE refiner.  Force the
        # serial CPU path — same guard à trous uses; GPU is single-process.
        return None


class RadialSymmetryBackend(TorchBackend):
    """Crocker–Grier detection with a **radial-symmetry** sub-pixel refiner
    (Parthasarathy, *Nat. Methods* 2012).

    A closed-form, non-iterative estimator: the spot centre is the point of
    maximal radial symmetry of the local intensity gradients — found by solving
    one small weighted linear system per spot.  ~95 % of CRLB precision but very
    robust to noise and to nearby spots, and fit-free (no convergence risk).

    Same contract as :class:`GaussianMleBackend`: detection identical to
    ``TorchBackend``, the estimator runs on the bandpassed ``signal``, ``mass``
    stays the centroid-of-mass mass on the trackpy scale, and a degenerate spot
    falls back to the centroid seed.  Overrides only ``_refine_peaks`` (+ the
    CPU-MP guard).
    """
    name = "radial-symmetry"

    def localise(self, stack, **kwargs):
        """TorchBackend.localise + the shared coincident-duplicate guard (see
        ``_drop_coincident_duplicates``)."""
        return self._drop_coincident_duplicates(super().localise(stack, **kwargs))

    def _refine_peaks(self, signal, t_ix, y_ix, x_ix, diameter, *,
                      device, dtype, raw=None, characterize=False):
        # Run the gradient/conv2d radial-symmetry solve on CPU when on MPS —
        # conv2d on tiny patches is silently unreliable there; see
        # TorchBackend._refine_off_mps.
        return self._refine_off_mps(
            self._refine_peaks_impl, signal, t_ix, y_ix, x_ix, diameter,
            device=device, dtype=dtype, raw=raw, characterize=characterize)

    def _refine_peaks_impl(self, signal, t_ix, y_ix, x_ix, diameter, *,
                           device, dtype, raw=None, characterize=False):
        import torch
        import torch.nn.functional as F
        # Gradient-based, so it runs on the bandpassed ``signal`` (background-
        # invariant, and the bandpass suppresses real-data background); ``raw`` is
        # accepted for the hook contract but unused.
        # Seed (integer centre + centroid offsets + trackpy-scaled mass + char).
        dy0, dx0, cy, cx, mass, char = self._iterative_centroid_refine(
            signal, t_ix, y_ix, x_ix, diameter,
            device=device, dtype=dtype, characterize=characterize)
        if cy.numel() == 0:
            return dy0, dx0, cy, cx, mass, char

        k = int(diameter)
        r = k // 2
        eps = 1e-6
        dy_grid, dx_grid = torch.meshgrid(
            torch.arange(-r, r + 1, device=device, dtype=dtype),
            torch.arange(-r, r + 1, device=device, dtype=dtype),
            indexing="ij")
        mask = TorchBackend._circular_mask(diameter, device, dtype).to(dtype)
        _, _, H, W = signal.shape
        ys = (cy[:, None, None] + dy_grid.long()[None]).clamp_(0, H - 1)
        xs = (cx[:, None, None] + dx_grid.long()[None]).clamp_(0, W - 1)
        ts = t_ix[:, None, None].expand_as(ys)
        n = (signal[ts, 0, ys, xs] * mask)                       # (N, k, k)

        # Parthasarathy 2012: gradients along the two 45°-rotated diagonals at
        # the (k-1)×(k-1) midpoint grid, smoothed 3×3; each gradient defines a
        # line through the centre, and the weighted least-squares intersection
        # of all the lines is the symmetry centre.
        dIdu = n[:, :-1, 1:] - n[:, 1:, :-1]                     # (N, k-1, k-1)
        dIdv = n[:, :-1, :-1] - n[:, 1:, 1:]
        h = torch.ones(1, 1, 3, 3, device=device, dtype=dtype) / 9.0
        fdu = F.conv2d(dIdu[:, None], h, padding=1)[:, 0]
        fdv = F.conv2d(dIdv[:, None], h, padding=1)[:, 0]
        dImag2 = fdu * fdu + fdv * fdv                           # gradient power

        denom = (fdu - fdv)
        m = -(fdv + fdu) / torch.where(denom.abs() < eps,
                                       torch.full_like(denom, eps), denom)
        m = torch.nan_to_num(m, nan=0.0, posinf=1e6, neginf=-1e6)  # (N, k-1, k-1)

        mids = torch.arange(k - 1, device=device, dtype=dtype) - (k - 2) / 2.0
        ym, xm = torch.meshgrid(mids, mids, indexing="ij")       # centred at 0
        ym = ym[None]
        xm = xm[None]
        b_line = ym - m * xm

        sdI2 = dImag2.sum(dim=(1, 2), keepdim=True).clamp(min=eps)
        xcent = (dImag2 * xm).sum(dim=(1, 2), keepdim=True) / sdI2
        ycent = (dImag2 * ym).sum(dim=(1, 2), keepdim=True) / sdI2
        # Down-weight gradients far from the gradient-power centroid (paper's
        # robustness weight), then solve the 2×2 weighted-least-squares system.
        wgt = dImag2 / torch.sqrt((xm - xcent) ** 2 + (ym - ycent) ** 2 + eps)
        wm2p1 = wgt / (m * m + 1.0)
        sw = wm2p1.sum(dim=(1, 2))
        smmw = (m * m * wm2p1).sum(dim=(1, 2))
        smw = (m * wm2p1).sum(dim=(1, 2))
        smbw = (m * b_line * wm2p1).sum(dim=(1, 2))
        sbw = (b_line * wm2p1).sum(dim=(1, 2))
        det = smmw * sw - smw * smw
        det = torch.where(det.abs() < eps, torch.full_like(det, eps), det)
        xc = (smw * sbw - smbw * sw) / det
        yc = (smmw * sbw - smw * smbw) / det

        ok = (torch.isfinite(xc) & torch.isfinite(yc)
              & (xc.abs() <= r) & (yc.abs() <= r))
        dx_off = torch.where(ok, xc, dx0)
        dy_off = torch.where(ok, yc, dy0)
        return dy_off, dx_off, cy, cx, mass, char

    def _localise_cpu_parallel(self, *args, **kwargs):
        # See GaussianMleBackend: force the serial CPU path so the refiner runs.
        return None
