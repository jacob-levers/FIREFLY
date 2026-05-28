"""Spot localisation: detection backends (trackpy, PyTorch),
adaptive/streaming chunking, and the localise_particles API.

Extracted from sptpalm_analysis.py (#7); re-exported there for compatibility.
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

import multiprocessing
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import trackpy as tp
from fa_constants import N_CPUS, _Cancelled, _tqdm
from fa_memory import (_alloc_or_memmap_stack, _register_temp_stack_path,
                       _resolve_temp_stack_dir, _user_ram_reserve_gb)
from fa_preprocess import (preprocess_stack, _preprocess_fast,
                           _preprocess_rolling)


def _ram_strategy(stack, headroom: float = 0.75) -> tuple[bool, float, float]:
    """
    Decide whether the full preprocessed stack fits in free RAM.

    Returns (use_fast, free_gb, needed_gb).
    Falls back to streaming if psutil is not installed.

    Holds back `_user_ram_reserve_gb()` for the OS + the user's other
    apps so a parallel Safari tab doesn't push the machine into swap.
    """
    # Peak fast-path RAM is roughly:
    #   * raw stack             1 ×
    #   * preprocessed copy     1 ×   (preprocess_stack output)
    #   * per-frame transient   ~ workers × 1 frame (small)
    #   * locate buffers        ~ 1 × chunk-size frame block
    #   * (mean + max + blink) projections — small, (Y,X) each
    # The 2.0× multiplier covers raw + preprocessed + a healthy slop
    # for locate / projections / fragmentation.
    needed_gb = stack.nbytes * 2.0 / 1e9
    try:
        import psutil
        free_gb    = psutil.virtual_memory().available / 1e9
        reserve_gb = _user_ram_reserve_gb()
        usable_gb  = max(0.0, free_gb - reserve_gb)
        return needed_gb < usable_gb * headroom, free_gb, needed_gb
    except ImportError:
        return False, 0.0, needed_gb


def _adaptive_chunk_and_workers(stack, requested_chunk: int,
                                 requested_workers: int) -> tuple[int, int]:
    """Adapt streaming-path `chunk_size` and `workers` to currently
    free RAM so the per-chunk peak doesn't blow the budget.

    Per-chunk peak (streaming) is approximately
        chunk_size · frame_bytes · (1 raw + 1 preprocessed + 1 transient)
    multiplied by `workers` for parallel preprocessing.  We require
    that to stay under half the usable RAM (the other half covers
    accumulators, locate buffers, drift correction, linking).

    Returns (chunk_size, workers) — both clamped to ≥ 1 and never
    above the user's requested values.
    """
    try:
        import psutil
        free_gb    = psutil.virtual_memory().available / 1e9
        reserve_gb = _user_ram_reserve_gb()
        usable_gb  = max(0.5, free_gb - reserve_gb)
    except Exception:
        # No psutil — trust the user's settings and hope for the best.
        return max(1, int(requested_chunk)), max(1, int(requested_workers))

    # Per-frame footprint (input dtype) ×3 (raw + preprocessed + transient).
    frame_bytes = stack.shape[1] * stack.shape[2] * stack.dtype.itemsize
    per_frame_peak = frame_bytes * 3.0

    # Budget half of usable RAM for the parallel preprocessing buffers.
    budget_bytes = usable_gb * 0.5e9
    if budget_bytes <= 0 or per_frame_peak <= 0:
        return max(1, int(requested_chunk)), max(1, int(requested_workers))

    # Total parallel frames we can afford in flight at once.
    max_frames_in_flight = int(budget_bytes / per_frame_peak)
    if max_frames_in_flight < 1:
        max_frames_in_flight = 1

    # Strategy: keep the user's chunk size if it fits with at least 1
    # worker.  Otherwise shrink chunk first (better data locality),
    # then reduce worker count.
    chunk = max(1, int(requested_chunk))
    workers = max(1, int(requested_workers))
    while workers >= 1 and chunk * workers > max_frames_in_flight:
        if chunk > 32:
            chunk = max(32, chunk // 2)
        elif workers > 1:
            workers -= 1
        else:
            break
    return chunk, workers


def _fast_preprocess_and_localise(stack, diameter=7, minmass=None, percentile=64,
                                   bg_radius=50, bg_method="uniform_filter",
                                   workers=N_CPUS, chunk_size=500,
                                   preview_cb=None, backend="auto",
                                   **backend_kwargs):
    """
    Fast path (ample RAM): preprocess the full stack in parallel, then localise
    in parallel chunks.  Faster than streaming because all preprocessing jobs
    run simultaneously rather than serially.

    Returns (locs, mean_proj_norm, max_proj, blink_proj, minmass_used)
    — same 5-tuple contract as the streaming path.  The fast path has
    the full stack in RAM so all three projections (mean, max, blink-
    count) are computed in a single vectorised pass.
    """
    import gc
    if diameter % 2 == 0:
        diameter += 1

    stack_pp = preprocess_stack(stack, bg_radius=bg_radius,
                                bg_method=bg_method, workers=workers)

    if minmass is None:
        # Auto-detect minmass.  The "mass" trackpy returns is *integrated*
        # intensity over the spot (≈π(d/2)² ≈ d²/π px ≈ d²/4 effective px,
        # depending on PSF shape).  The old formula used `peak × 0.4` which
        # is the *per-pixel* threshold — that under-shoots the integrated
        # threshold by ~10× and produces 100k+ false-positive "spots" on
        # PALM-density data.  Corrected to account for the spot's pixel
        # support: `peak × diameter² / 8` (0.5 × effective area).
        # This is still a heuristic and may need manual tuning; users with
        # known data should set minmass explicitly via the GUI spinbox.
        _peak = float(np.percentile(stack_pp[min(5, len(stack_pp) - 1)], 99))
        minmass = float(_peak * (diameter ** 2) / 8.0)
        print(f"  Auto minmass: {minmass:.4f}  "
              f"(from 99th-pct peak {_peak:.4f} × d²/8)")

    # ── Projections for ROI / circular-stats downstream ──────────────────
    # Mean projection (normalised) — same contract as before.
    mean_proj = stack_pp.mean(axis=0).astype(np.float32)
    mn, mx    = mean_proj.min(), mean_proj.max()
    if mx > mn:
        mean_proj = (mean_proj - mn) / (mx - mn)

    # Max projection — un-normalised; build_roi_mask_advanced normalises.
    max_proj = stack_pp.max(axis=0).astype(np.float32)

    # Blink-count projection — per-pixel count of frames significantly
    # above the pixel's own mean + 3·std baseline.
    #
    # Memory note: doing `(stack_pp > thresh[None]).sum(axis=0)` in one
    # shot materialises a (T, Y, X) bool tensor (~T × frame_bytes /4),
    # which for a 4000-frame 512² movie is 1 GB.  On a 16 GB machine
    # under memory pressure (FIREFLY's typical OOM scenario) that's
    # enough to push the run over.  We instead accumulate the count
    # in chunks of ≤ 256 frames so peak extra memory stays ~64 MB.
    px_mean = stack_pp.mean(axis=0)
    px_std  = stack_pp.std(axis=0)
    thresh  = px_mean + 3.0 * px_std
    blink_proj = np.zeros(stack_pp.shape[1:], dtype=np.float32)
    _BLINK_CHUNK = 256
    for _s in range(0, stack_pp.shape[0], _BLINK_CHUNK):
        _e = min(_s + _BLINK_CHUNK, stack_pp.shape[0])
        blink_proj += (stack_pp[_s:_e] > thresh[None]).sum(
            axis=0, dtype=np.float32)
    del px_mean, px_std, thresh
    gc.collect()

    locs = localise_particles(stack_pp, diameter=diameter, minmass=minmass,
                              percentile=percentile, workers=workers,
                              chunk_size=chunk_size, preview_cb=preview_cb,
                              backend=backend, **backend_kwargs)
    del stack_pp
    gc.collect()
    return locs, mean_proj, max_proj, blink_proj, minmass


def preprocess_and_localise_adaptive(stack, diameter=7, minmass=None, percentile=64,
                                     bg_radius=50, bg_method="uniform_filter",
                                     workers=N_CPUS, chunk_size=500,
                                     ram_headroom: float = 0.75,
                                     preview_cb=None, stop_event=None,
                                     mass_cb=None, backend="auto",
                                     **backend_kwargs):
    """
    Adaptive dispatcher — automatically selects the fastest strategy that fits
    in available RAM.

    Fast path   (plenty of RAM): full parallel preprocessing → parallel localisation.
                                 Scales with both CPU count and RAM size.
    Stream path (tight RAM):     one chunk preprocessed + localised + discarded at
                                 a time.  Peak extra RAM = one chunk only.

    The decision is made at runtime using psutil to query free memory.
    ``ram_headroom`` (default 0.75) means the preprocessed copy must fit in
    75 % of currently free RAM so the OS and other processes retain a buffer.

    Returns (locs, mean_proj_norm, minmass_used)
    """
    # Resolve and announce the backend once, up front — visible in the log
    # regardless of which RAM strategy we end up taking (the FAST path goes
    # through localise_particles which re-prints; the STREAM path bypasses it
    # entirely, so we need this line here too).
    try:
        _impl = _resolve_backend(backend)
        print(f"  Backend   : {_impl.name}  (requested: {backend})")
    except Exception as _e:
        print(f"  Backend   : (resolution failed: {_e})")

    use_fast, free_gb, needed_gb = _ram_strategy(stack, headroom=ram_headroom)
    reserve_gb = _user_ram_reserve_gb()

    if use_fast:
        print(f"  RAM strategy : FAST (parallel)   — "
              f"{free_gb:.1f} GB free, {needed_gb:.1f} GB needed, "
              f"{reserve_gb:.1f} GB reserved for OS/apps")
        return _fast_preprocess_and_localise(
            stack, diameter, minmass, percentile,
            bg_radius, bg_method, workers, chunk_size,
            preview_cb=preview_cb, backend=backend,
            **backend_kwargs)
    else:
        print(f"  RAM strategy : STREAM (low-mem)  — "
              f"{free_gb:.1f} GB free, {needed_gb:.1f} GB needed, "
              f"{reserve_gb:.1f} GB reserved for OS/apps")
        return preprocess_and_localise_stream(
            stack, diameter, minmass, percentile,
            bg_radius, bg_method, workers, chunk_size,
            preview_cb=preview_cb, stop_event=stop_event,
            mass_cb=mass_cb, backend=backend,
            **backend_kwargs)


def preprocess_and_localise_stream(stack, diameter=7, minmass=None, percentile=64,
                                   bg_radius=50, bg_method="uniform_filter",
                                   workers=N_CPUS, chunk_size=500,
                                   preview_cb=None, stop_event=None,
                                   mass_cb=None, backend="auto",
                                   **backend_kwargs):
    """
    Memory-efficient single streaming pass: preprocess + localise without ever
    materialising the full preprocessed stack in RAM.

    Each chunk is preprocessed, localised, and immediately discarded, so peak
    extra memory above the raw stack is one chunk (~chunk_size frames).
    For a 10 000-frame 512×512 stack this cuts peak RAM from ~2× to ~1× stack size.

    Parameters
    ----------
    stack    : raw float32 stack (T x Y x X)
    minmass  : if None, auto-detected from the first preprocessed chunk

    Returns
    -------
    locs             : DataFrame of all localised particles
    mean_proj_norm   : float32 (Y, X) normalised [0,1] mean of preprocessed frames
                       — suitable for ROI thresholding
    minmass          : the minmass value actually used
    """
    import gc
    if diameter % 2 == 0:
        diameter += 1

    fn       = _preprocess_fast if bg_method == "uniform_filter" else _preprocess_rolling
    n_frames = len(stack)
    # Adapt chunk_size + worker count to currently free RAM so the
    # parallel-preprocessing inner pool doesn't push the box into OOM
    # on the user's first dense file.  This is the most common cause
    # of the symptom "FIREFLY crashed mid-run with OOM" on 16 GB Macs.
    chunk_size_adj, workers_adj = _adaptive_chunk_and_workers(
        stack, chunk_size, max(1, min(workers, N_CPUS)))
    if (chunk_size_adj != chunk_size) or (workers_adj != workers):
        print(f"  RAM auto-tune: chunk_size {chunk_size} → "
              f"{chunk_size_adj},  workers {workers} → {workers_adj}  "
              f"(reduced to stay within free RAM)")
    chunk_size = chunk_size_adj
    n_chunks = max(1, int(np.ceil(n_frames / chunk_size)))
    workers_ = workers_adj

    # Resolve the backend up front so each chunk goes through the same
    # implementation.  Trackpy is special-cased below to skip the per-chunk
    # process-pool spawn cost; everything else delegates to .localise().
    #
    # NOTE: an earlier version of this code bumped chunk_size to 1500 on
    # MPS/CUDA hoping to amortize dispatch overhead.  Empirically that made
    # things *slower* on Apple Silicon — the GPU is bandwidth-limited at
    # these convolution sizes, and 500-frame chunks fit better in cache
    # than 1500-frame chunks.  Per-frame throughput dropped ~3× when we
    # tried the bigger chunks.  Sticking with the caller's chunk_size now.
    _impl = _resolve_backend(backend)
    print(f"  Mode      : streaming preprocess + localise  (low memory)")
    print(f"  Backend   : {_impl.name}")
    print(f"  Diameter  : {diameter}px  |  bg_method: {bg_method}")
    print(f"  Chunks    : {n_chunks} × ~{chunk_size} frames  |  workers: {workers_}")
    t0 = time.perf_counter()

    def _localise_chunk_via_backend(chunk_pp):
        """Run the active backend on a single preprocessed chunk and return
        a DataFrame with at least columns x, y, frame, mass.

        Trackpy: call `tp.batch` directly with processes=1 to skip the
                 multiprocessing-pool spawn overhead (per-chunk, the pool
                 startup cost would dominate the actual work).
        Other:   delegate to the backend's `.localise()` (single iteration
                 because the chunk is already smaller than chunk_size).
        """
        if _impl.name == "trackpy":
            with _threadpool_limits(limits=N_CPUS):
                return tp.batch(chunk_pp, diameter=diameter, minmass=minmass,
                                percentile=percentile, processes=1)
        return _impl.localise(chunk_pp, diameter=diameter, minmass=minmass,
                              percentile=percentile, workers=workers_,
                              chunk_size=len(chunk_pp),
                              **backend_kwargs)

    # ── First chunk: preprocess now so we can auto-detect minmass ─────────────
    first_end  = min(chunk_size, n_frames)
    with ThreadPoolExecutor(max_workers=workers_) as _exe:
        first_pp = np.stack([_f.result() for _f in
                             [_exe.submit(fn, f, bg_radius) for f in stack[:first_end]]])

    if minmass is None:
        # Auto-detect minmass.  trackpy's "mass" is *integrated* intensity
        # over a (diameter × diameter) spot patch, not a single-pixel value.
        # The old formula `peak × 0.4` was a per-pixel threshold and under-
        # shoots integrated mass by ~10×, producing 100k+ false-positive
        # spots on dense PALM data.  Corrected to `peak × d²/8` — accounts
        # for the spot's pixel support area at the standard 50% acceptance.
        # Still a heuristic; users with known data should set minmass
        # explicitly via the GUI spinbox.
        _peak = float(np.percentile(first_pp[min(5, first_end - 1)], 99))
        minmass = float(_peak * (diameter ** 2) / 8.0)
        print(f"  Auto minmass: {minmass:.4f}  "
              f"(from 99th-pct peak {_peak:.4f} × d²/8)")
    else:
        print(f"  Minmass   : {minmass:.4f}")

    # ── Stream all chunks ──────────────────────────────────────────────────────
    all_locs  = []
    mean_acc  = first_pp.sum(axis=0).astype(np.float64)
    # Max-projection accumulator.  Cheap to stream (one np.maximum per
    # chunk) and unlocks the same Max-projection ROI mode the GUI
    # preview uses, so what-you-see-is-what-you-get for ROI.
    max_acc   = first_pp.max(axis=0).astype(np.float32)
    frame_count = len(first_pp)

    # ── Per-pixel Welford (streaming variance) + blink-count ────────────
    # Welford's online algorithm gives running per-pixel mean and M2
    # (sum of squared deltas from the running mean) without ever
    # needing to keep the stack in RAM.  Combined with the standard
    # chunk-merge formula it's vectorisable: one merge per chunk, not
    # per frame.
    #
    # After each chunk merges in, we use `mean + 3*std` as a per-pixel
    # "this pixel is unusually bright right now" threshold and count
    # how many frames in *this chunk* exceeded it.  Across a 4000+
    # frame movie the estimate stabilises within the first chunk, so
    # the running-baseline approximation is close to the 2-pass
    # ground truth that the GUI preview uses on its 30-frame stack.
    welford_mean = first_pp.mean(axis=0).astype(np.float64)
    welford_M2   = (first_pp.var(axis=0, dtype=np.float64)
                    * first_pp.shape[0]).astype(np.float64)
    welford_n    = first_pp.shape[0]
    blink_count  = np.zeros(first_pp.shape[1:], dtype=np.uint32)
    # MAD→σ factor: skimage Welford std is normal-distribution std,
    # whereas the GUI uses median+3·MAD≈median+3·1.4826·σ.  We use 3·σ
    # here to match — Welford only sees one realisation per frame, so
    # MAD-vs-σ correction isn't applicable.
    _BLINK_K = 3.0
    # Count blinks in the first chunk against its own stats — slightly
    # circular, but the mean/std of 500 frames is a reasonable baseline
    # and using it avoids "no blinks counted for the first chunk".
    if welford_n > 0:
        _std_est = np.sqrt(welford_M2 / max(welford_n, 1))
        _thresh  = welford_mean + _BLINK_K * _std_est
        blink_count += (first_pp > _thresh[None]).sum(axis=0).astype(np.uint32)

    # Localise first chunk (already preprocessed) — through the active backend
    locs0 = _localise_chunk_via_backend(first_pp)
    if len(locs0) > 0:
        all_locs.append(locs0)
    if mass_cb is not None and len(locs0) > 0 and "mass" in locs0.columns:
        try:    mass_cb(np.asarray(locs0["mass"].values, dtype=np.float32))
        except Exception: pass

    # ── Live preview: emit EVERY frame of each chunk after localisation
    # so the GUI's live view scrolls through the actual movie at 60 Hz
    # rather than ticking once per chunk.  The GUI's repaint timer
    # naturally drops in-between frames it can't paint in time, so we
    # just fire-and-forget every frame — the message queue + per-frame
    # cost is tiny next to localisation itself.
    def _emit_chunk_previews(chunk_pp, locs_chunk, frame_offset):
        if preview_cb is None or len(chunk_pp) == 0:
            return
        # Pre-index spots by frame for cheap per-frame lookups
        spots_by_frame = {}
        if len(locs_chunk) > 0 and "frame" in locs_chunk.columns:
            for f, sub in locs_chunk.groupby("frame"):
                spots_by_frame[int(f)] = (sub["x"].values, sub["y"].values)
        for local_i in range(len(chunk_pp)):
            global_i = frame_offset + local_i
            sxy = spots_by_frame.get(global_i, ([], []))
            try:
                preview_cb(global_i, chunk_pp[local_i],
                           sxy[0], sxy[1], n_frames)
            except Exception:
                pass

    _emit_chunk_previews(first_pp, locs0, frame_offset=0)

    del first_pp
    gc.collect()

    # Remaining chunks
    for i in _tqdm(range(1, n_chunks), desc="  Streaming", unit="chunk", ncols=70):
        # Honour a stop request between chunks
        if stop_event is not None and stop_event.is_set():
            print("  Streaming stopped by user.")
            break

        start     = i * chunk_size
        end       = min(start + chunk_size, n_frames)
        with ThreadPoolExecutor(max_workers=workers_) as _exe:
            chunk_pp = np.stack([_f.result() for _f in
                                 [_exe.submit(fn, f, bg_radius) for f in stack[start:end]]])

        mean_acc   += chunk_pp.sum(axis=0)
        np.maximum(max_acc, chunk_pp.max(axis=0), out=max_acc)
        frame_count += len(chunk_pp)

        # ── Chunk-merge Welford for per-pixel mean/variance ────────
        # Parallel-Welford combine of two means:
        #   delta = mean_b - mean_a
        #   n     = n_a + n_b
        #   M2    = M2_a + M2_b + delta**2 * n_a * n_b / n
        #   mean  = mean_a + delta * n_b / n
        n_b = chunk_pp.shape[0]
        if n_b > 0:
            chunk_mean = chunk_pp.mean(axis=0, dtype=np.float64)
            chunk_M2   = (chunk_pp.var(axis=0, dtype=np.float64)
                          * n_b).astype(np.float64)
            n_total    = welford_n + n_b
            delta      = chunk_mean - welford_mean
            welford_M2 = (welford_M2 + chunk_M2
                          + (delta * delta) * welford_n * n_b / n_total)
            welford_mean = welford_mean + delta * (n_b / n_total)
            welford_n  = n_total
            # Per-pixel threshold from the latest running estimate, then
            # count blinks in *this* chunk.  Population std (divide by n
            # not n-1) — at n>>1 the difference is irrelevant and avoids
            # a degenerate case at n=1.
            _std_est = np.sqrt(welford_M2 / max(welford_n, 1))
            _thresh  = welford_mean + _BLINK_K * _std_est
            blink_count += (chunk_pp > _thresh[None]).sum(
                axis=0).astype(np.uint32)

        locs_i = _localise_chunk_via_backend(chunk_pp)

        if len(locs_i) > 0:
            locs_i = locs_i.copy()
            locs_i["frame"] += start
            all_locs.append(locs_i)
        if mass_cb is not None and len(locs_i) > 0 and "mass" in locs_i.columns:
            try:    mass_cb(np.asarray(locs_i["mass"].values, dtype=np.float32))
            except Exception: pass

        # Live previews — multiple evenly-spaced frames within this chunk
        _emit_chunk_previews(chunk_pp, locs_i, frame_offset=start)

        del chunk_pp
        gc.collect()

    # ── Mean projection (normalised) ──────────────────────────────────────────
    mean_proj = (mean_acc / frame_count).astype(np.float32)
    mn, mx    = mean_proj.min(), mean_proj.max()
    if mx > mn:
        mean_proj = (mean_proj - mn) / (mx - mn)

    # ── Max projection (un-normalised; build_roi_mask_advanced normalises) ────
    max_proj = max_acc.astype(np.float32)

    # ── Blink-density projection ─────────────────────────────────────────────
    # Per-pixel count of frames where the pixel exceeded its own running
    # mean + 3·std (cumulative-up-to-that-frame).  Most discriminative ROI
    # projection for sptPALM: cells blink repeatedly, autofluorescent
    # background is steady so its blink-count is ~zero.  Cast to float32
    # so build_roi_mask_advanced can DoG / smooth it like any image.
    blink_proj = blink_count.astype(np.float32)

    result  = pd.concat(all_locs, ignore_index=True) if all_locs else pd.DataFrame()
    elapsed = time.perf_counter() - t0
    print(f"  Found {len(result):,} localisations in {elapsed:.1f}s  "
          f"({n_frames / elapsed:.0f} frames/s)")
    # Returns (locs, mean_proj, max_proj, blink_proj, minmass).
    # max_proj + blink_proj are the streaming-accumulator projections
    # consumed by firefly_worker.py → build_roi_mask_advanced so the
    # worker's ROI mask matches whatever the user picked in the GUI
    # preview.  Old callers that unpack the 3-tuple need updating —
    # firefly_worker is the only one.
    return result, mean_proj, max_proj, blink_proj, minmass


def _localise_chunk(chunk, diameter, minmass, percentile, frame_offset):
    """Localise one chunk and apply global frame offset."""
    locs = tp.batch(chunk, diameter=diameter, minmass=minmass,
                    percentile=percentile, processes=1)
    if len(locs) > 0:
        locs = locs.copy()
        locs["frame"] += frame_offset
    return locs


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
        chunks   = np.array_split(stack, n_chunks)
        offsets  = [i * chunk_size for i in range(len(chunks))]
        chunk_pairs = list(zip(chunks, offsets))

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
                # Build a list of (start, end) slice indices that
                # mirror what np.array_split would have produced —
                # but never materialise the chunks in the parent.
                splits = np.array_split(np.arange(n_frames), n_chunks)
                slice_ranges = [(int(s[0]), int(s[-1]) + 1) for s in splits if len(s)]
                dtype_str = str(stack.dtype)
                shape     = tuple(stack.shape)
                mp_args = [(i, memmap_path, dtype_str, shape,
                            start, end,
                            diameter, minmass, percentile, start)
                           for i, (start, end) in enumerate(slice_ranges)]
                with ctx.Pool(processes=n_workers) as pool:
                    _spawn_announced = False
                    for idx, result in _tqdm(
                            pool.imap_unordered(_localise_chunk_mmap_mp, mp_args),
                            total=n_chunks, desc="  Localising", unit="chunk", ncols=70):
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
                            preview_cb, stack, slice_ranges[idx], result,
                            n_frames)
            else:
                mp_args = [(i, c, diameter, minmass, percentile, o)
                           for i, (c, o) in enumerate(chunk_pairs)]
                with ctx.Pool(processes=n_workers) as pool:
                    _spawn_announced = False
                    for idx, result in _tqdm(
                            pool.imap_unordered(_localise_chunk_mp, mp_args),
                            total=n_chunks, desc="  Localising", unit="chunk", ncols=70):
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
            with _threadpool_limits(limits=N_CPUS):
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

    Since v2.6.13 every algorithmic stage mirrors trackpy's documented
    pipeline so the Torch backend can be used as a GPU-accelerated
    drop-in replacement for `tp.batch` without changing the scientific
    interpretation of the output:

      1.  Bandpass — `gaussian(image, σ = noise_size = 1)` minus
          `uniform_filter(image, smoothing_size = diameter + 1)`,
          clamped ≥ 0.  Matches `trackpy.preprocessing.bandpass`.
      2.  Threshold — the `percentile`-th percentile of the bandpassed
          image (identical semantics to trackpy's `percentile` arg).
      3.  Local maxima — pixels where the bandpassed signal equals
          its `diameter`-window max-pool output AND exceeds the
          threshold.
      4.  Sub-pixel refinement — iterative centroid-of-mass with a
          circular disk mask of radius `diameter/2`.  Each iteration
          shifts the integer centre by ±1 px when the centroid offset
          exceeds `shift_thresh = 0.6 px`; loop terminates after at
          most `max_iters = 10` iterations.  Identical to
          `trackpy.refine.refine`.
      5.  Mass — sum of bandpassed signal under the disk mask at the
          converged centre.  Same definition trackpy uses for its
          `mass` column.

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
    care — the calibration is validated by `tests/test_localiser_agreement.py`,
    which asserts median ≤ 0.20 px and recall ≥ 0.95.

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
        response = smooth - bg
        return torch.clamp(response, min=0.0)

    @staticmethod
    def _circular_mask(diameter, device, dtype):
        """Boolean / float disk-mask of side k = diameter, used to integrate
        signal over a spot's footprint.  Pixels inside radius `diameter/2`
        of the patch centre are 1; outside are 0.  Matches the geometry
        used by `tp.refine.refine` for both centroid-of-mass and mass
        calculation."""
        import torch
        k = int(diameter)
        r = (k - 1) / 2.0
        ys = torch.arange(k, device=device, dtype=dtype) - r
        xs = torch.arange(k, device=device, dtype=dtype) - r
        Y, X = torch.meshgrid(ys, xs, indexing="ij")
        rad = (k / 2.0)
        return (Y * Y + X * X) <= (rad * rad)

    @staticmethod
    def _iterative_centroid_refine(signal, t_ix, y_ix, x_ix, diameter,
                                     device, dtype,
                                     max_iters=None, shift_thresh=None):
        """Vectorised iterative centroid-of-mass refinement — Torch
        port of `trackpy.refine.refine`.

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

        return dy_sub, dx_sub, cur_y, cur_x, final_mass

    def localise(self, stack, *, diameter=7, minmass=0.1, percentile=64,
                 workers=None, chunk_size=500, preview_cb=None,
                 device=None, **_):
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
        # via threadpoolctl.  GPU devices ignore these settings.
        if dev_str == "cpu":
            try:    torch.set_num_threads(int(N_CPUS))
            except Exception: pass
            try:    torch.set_num_interop_threads(int(N_CPUS))
            except (RuntimeError, Exception):
                # set_num_interop_threads errors if any parallel work has
                # already been dispatched on this interpreter — harmless,
                # the first-call thread count is what counts.
                pass

        print(f"  Device    : {dev_str}")
        print(f"  Diameter  : {diameter}px  |  minmass: {minmass:.4f}  "
              f"|  percentile: {percentile}")
        print(f"  Chunks    : {n_chunks} × ~{chunk_size} frames")
        if dev_str == "cpu":
            try:    print(f"  Torch threads : {torch.get_num_threads()}")
            except Exception: pass

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
            _threadpool_limits(limits=int(N_CPUS)) if dev_str == "cpu" else None
        )
        if _blas_ctx is not None:
            try:    _blas_ctx.__enter__()
            except Exception: _blas_ctx = None

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

            # ── 2. Percentile threshold per chunk ───────────────────────────
            # torch.quantile is exact for small inputs; for big tensors use
            # sample-based estimate to bound memory.
            flat = signal.reshape(-1)
            if flat.numel() > 5_000_000:
                idx = torch.randint(0, flat.numel(),
                                    (5_000_000,), device=dev)
                sample = flat[idx]
                threshold = torch.quantile(sample, percentile / 100.0)
            else:
                threshold = torch.quantile(flat, percentile / 100.0)

            # ── 3. Local maxima via max-pool == self ────────────────────────
            maxp   = F.max_pool2d(signal, kernel_size=k, stride=1, padding=radius)
            is_max = (signal == maxp) & (signal > threshold)
            # nonzero → (N, 4) columns: (t, c, y, x)
            coords = is_max.nonzero(as_tuple=False)
            if coords.numel() == 0:
                _ct = time.perf_counter()
                print(f"  Chunk {chunk_idx+1}/{n_chunks} "
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
                print(f"  Chunk {chunk_idx+1}/{n_chunks} "
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
            dy_off, dx_off, y_final, x_final, mass = (
                self._iterative_centroid_refine(
                    signal, t_ix, y_ix, x_ix, diameter,
                    device=dev, dtype=dtype)
            )

            # Apply the minmass filter AFTER refinement (trackpy filters
            # on the refined mass, not the pre-refinement patch sum).
            keep = mass >= minmass
            if not bool(keep.any()):
                _ct = time.perf_counter()
                print(f"  Chunk {chunk_idx+1}/{n_chunks} "
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

            all_locs.append({
                "x":     x_sub.detach().cpu().numpy(),
                "y":     y_sub.detach().cpu().numpy(),
                "frame": frame_abs.detach().cpu().numpy(),
                "mass":  mass.detach().cpu().numpy(),
            })

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
                print(f"  Chunk {chunk_idx+1}/{n_chunks} "
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

        # Release the BLAS thread-pool expansion (matched __enter__ above).
        # Outside this scope the global OMP=1 cap reasserts itself so the
        # downstream linker / preview pump don't get oversubscribed.
        if _blas_ctx is not None:
            try:    _blas_ctx.__exit__(None, None, None)
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

        if not all_locs:
            print("  Found 0 localisations")
            return pd.DataFrame(columns=["x", "y", "frame", "mass"])

        df = pd.DataFrame({
            col: np.concatenate([d[col] for d in all_locs])
            for col in ("x", "y", "frame", "mass")
        })

        elapsed = time.perf_counter() - t0
        print(f"  Found {len(df):,} localisations in {elapsed:.1f}s  "
              f"({n_frames / elapsed:.0f} frames/s)")
        return df


_BACKEND_REGISTRY: list[type[LocaliserBackend]] = [
    TrackpyBackend, TorchBackend,
]


def list_available_backends() -> list[str]:
    """Return the names of all backends usable on this machine.

    For TorchBackend this expands to one entry per visible device
    (`torch` = auto-select fastest; `torch-mps` / `torch-cuda` / `torch-cpu`
    = explicit device pin, useful for benchmarking or reproducibility).
    """
    out: list[str] = []
    for b in _BACKEND_REGISTRY:
        if not b.is_available():
            continue
        out.append(b.name)
        if b is TorchBackend:
            for dev in TorchBackend.list_devices():
                out.append(f"torch-{dev.replace(':', '')}")
    return out


def _resolve_backend(name: str | None):
    """Look up a backend by name; resolve 'auto' to the FASTEST available
    backend that's actually healthy on this machine.

    Auto-selection logic:
      1. Prefer TorchBackend if a GPU device (MPS / CUDA) passes the sanity
         check — that's the only configuration where torch beats trackpy.
      2. Otherwise pick TrackpyBackend.  Torch-on-CPU is comparable to
         trackpy in speed but less battle-tested, so trackpy wins ties.

    This keeps users on M-series Macs out of the MPS-OOM trap when their
    Metal context is degraded (e.g. after an aborted prior process): the
    sanity check fails, select_device() returns "cpu", and auto picks
    trackpy.  After a reboot when MPS works again, auto picks torch
    automatically — the user never has to touch the dropdown.

    Accepts torch-device pins (`torch-mps`, `torch-cuda`, `torch-cpu`) that
    pre-set the device on the returned instance — used for benchmarking
    and to let users force a specific device path.
    """
    if name in (None, "", "auto"):
        # Smart-auto: GPU-first.  Order is CUDA → MPS → trackpy → torch-CPU.
        #
        # Earlier versions skipped MPS in auto-resolution because of
        # reliability issues observed on macOS 26 + M4 + PyTorch 2.12 (the
        # MPS allocator producing Metal command-buffer OOMs at extreme
        # spot density).  Most of those have been mitigated since:
        #   • PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 set at process start
        #   • per-chunk + end-of-localise mps.synchronize + empty_cache
        #   • Gaussian fit sub-batched at 5k spots/call to avoid the
        #     batched linalg.solve issue
        #   • subprocess isolation so Qt's Metal claim doesn't compete
        #     with PyTorch's MPS for unified memory on Apple Silicon
        # With those in place, MPS is the right default on Apple Silicon
        # (~6× faster than CPU on typical SPT stacks).  If a specific
        # machine still has trouble, users can manually pick Trackpy or
        # Torch — CPU from the dropdown.
        if TorchBackend.is_available():
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    inst = TorchBackend()
                    inst._forced_device = "cuda"
                    return inst
                if (hasattr(_torch.backends, "mps")
                        and _torch.backends.mps.is_available()):
                    inst = TorchBackend()
                    inst._forced_device = "mps"
                    return inst
            except Exception:
                pass
        # No GPU available → reference CPU implementation (trackpy).
        for cls in _BACKEND_REGISTRY:
            if cls is TorchBackend:
                continue
            if cls.is_available():
                return cls()
        # Last resort: torch on CPU, if even trackpy is missing.
        if TorchBackend.is_available():
            inst = TorchBackend()
            inst._forced_device = "cpu"
            return inst
        raise RuntimeError(
            "No localiser backend available — install trackpy or torch.")

    # Torch-device pins (e.g. 'torch-mps', 'torch-cuda:0', 'torch-cpu')
    if name.startswith("torch-"):
        if not TorchBackend.is_available():
            raise RuntimeError(
                "Torch device pin requested but PyTorch isn't installed.")
        forced = name[len("torch-"):]
        # Validate the requested device is actually available BEFORE
        # we start running the pipeline — otherwise the failure happens
        # mid-localisation with a cryptic "Torch not compiled with CUDA
        # enabled" assertion, after the user has already waited for
        # frame loading + preprocessing.
        try:
            import torch as _torch
            short = forced.split(":", 1)[0]
            if short == "cuda" and not _torch.cuda.is_available():
                raise RuntimeError(
                    "You selected the NVIDIA CUDA backend but the "
                    "bundled PyTorch is CPU-only.\n\n"
                    "Fix: on Windows, click the 'Set up GPU acceleration…' "
                    "button in the Analysis sidebar to install the CUDA "
                    "wheel — or change the Detection backend dropdown to "
                    "'Auto' or 'Torch — CPU' to continue without GPU.")
            if short == "mps":
                has_mps = (hasattr(_torch.backends, "mps")
                           and _torch.backends.mps.is_available())
                if not has_mps:
                    raise RuntimeError(
                        "You selected the Apple MPS backend but this "
                        "system doesn't have MPS available "
                        "(MPS requires Apple Silicon + macOS 12+).\n\n"
                        "Change the Detection backend dropdown to 'Auto' "
                        "or 'Torch — CPU' to continue.")
        except RuntimeError:
            raise
        except Exception:
            # If we can't introspect torch for any reason, fall through
            # and let the original code path produce its native error.
            pass
        inst = TorchBackend()
        inst._forced_device = forced
        return inst

    for cls in _BACKEND_REGISTRY:
        if cls.name == name:
            if not cls.is_available():
                raise RuntimeError(
                    f"Localiser backend '{name}' is registered but its "
                    f"dependencies aren't installed on this machine.")
            return cls()
    raise ValueError(
        f"Unknown localiser backend '{name}'. "
        f"Registered: {[c.name for c in _BACKEND_REGISTRY]}; "
        f"available here: {list_available_backends()}.")


def localise_particles(stack, diameter=7, minmass=0.1, percentile=64,
                       workers=N_CPUS, chunk_size=500, preview_cb=None,
                       backend="auto", **backend_kwargs):
    """Localise spots in every frame of a preprocessed stack.

    `backend` selects the implementation:
        "auto"     — first available entry in _BACKEND_REGISTRY
        "trackpy"  — Crocker-Grier centroid (CPU, multi-process)
        "torch[-mps|-cuda|-cpu]" — GPU/CPU PyTorch localiser

    Extra `backend_kwargs` are forwarded verbatim to the active
    backend's `.localise()`.

    Returns a DataFrame with columns: x, y, frame, mass.
    """
    impl = _resolve_backend(backend)
    print(f"  Backend   : {impl.name}")
    return impl.localise(stack, diameter=diameter, minmass=minmass,
                         percentile=percentile, workers=workers,
                         chunk_size=chunk_size, preview_cb=preview_cb,
                         **backend_kwargs)
