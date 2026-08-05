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


# Detection backends were extracted to fa_localize_backends (behaviour-preserving).
# Re-imported here so `from firefly.analysis.fa_localize import TorchBackend` and the
# sptpalm_analysis re-exports keep resolving unchanged.
from firefly.analysis.fa_localize_backends import (  # noqa: F401
    _localise_chunk, _localise_chunk_mp, _localise_chunk_mmap_mp,
    _torch_localise_block_mp, LocaliserBackend, _emit_trackpy_chunk_preview,
    TrackpyBackend, TorchBackend, AtrousWaveletBackend,
    GaussianMleBackend, RadialSymmetryBackend,
)


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

    # A lazy on-demand stack (e.g. LazyTiffStack) must never take the FAST path:
    # FAST preprocesses the whole stack at once, which would pull every frame
    # off disk into RAM and defeat the point.  Force STREAM, which indexes it in
    # bounded slices.
    if getattr(stack, "_is_lazy_stack", False) and use_fast:
        print("  RAM strategy : forcing STREAM (lazy on-demand stack)")
        use_fast = False

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


def _resolve_gpu_batch(impl, stack, sub_chunk, n_frames, interactive=False):
    """Frames per backend localise-call in streaming mode (≥ the preprocessing
    `sub_chunk`).  Device-aware:

      • CUDA   → up to 256, bounded by free VRAM (`mem_get_info`) so we don't OOM.
      • MPS    → keep = sub_chunk.  Bigger batches empirically SLOW the Apple
                 Silicon GPU (bandwidth-bound), so Apple users are left untouched.
      • CPU / trackpy → up to 256 (the win there is amortising per-call setup).

    `FIREFLY_GPU_BATCH` overrides everything.  Interactive (live-preview) runs cap
    at 64 so the preview still scrolls smoothly.  Detection is unaffected by the
    batch size (the percentile threshold is a frame-grouping-stable background
    estimate), so this only changes throughput."""
    sub_chunk = max(1, int(sub_chunk))
    env = os.environ.get("FIREFLY_GPU_BATCH")
    if env:
        try:
            return int(max(sub_chunk, min(int(env), n_frames)))
        except Exception:
            pass
    dev = getattr(impl, "_forced_device", None)
    if dev is None and hasattr(impl, "select_device"):
        try:    dev = impl.select_device()
        except Exception: dev = None
    dev = str(dev or "").lower()
    if "mps" in dev:
        return sub_chunk
    target = 64 if interactive else 256
    if "cuda" in dev:
        try:
            import torch
            free, _total = torch.cuda.mem_get_info()
            frame_px = int(stack.shape[1]) * int(stack.shape[2])
            # f32 input + bandpass / max-pool / refinement intermediates.
            per_frame = frame_px * 4 * 12
            vram_cap = int((free * 0.4) // max(per_frame, 1))
            target = min(target, max(sub_chunk, vram_cap))
        except Exception:
            pass
    return int(min(max(sub_chunk, target), n_frames))


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
    # NOTE: the backend name + requested device are already printed by
    # preprocess_and_localise_adaptive (the only production caller) — don't
    # repeat them here.
    print(f"  Mode      : streaming ({_impl.name}, low memory)  |  "
          f"diameter {diameter}px, bg {bg_method}")
    print(f"  Chunks    : {n_chunks} sub-chunks × ~{chunk_size} frames  "
          f"|  workers: {workers_}")
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
            with _threadpool_limits(limits=_cpu_core_budget()):
                return tp.batch(chunk_pp, diameter=diameter, minmass=minmass,
                                percentile=percentile, processes=1)
        # quiet=True: this backend is invoked once per sub-chunk in streaming
        # mode, so its own per-call banner / per-chunk progress lines would
        # duplicate the streaming loop's tqdm bar hundreds of times.
        return _impl.localise(chunk_pp, diameter=diameter, minmass=minmass,
                              percentile=percentile, workers=workers_,
                              chunk_size=len(chunk_pp), quiet=True,
                              **backend_kwargs)

    # One preprocessing thread-pool for the WHOLE stream.  Spawning a fresh
    # ThreadPoolExecutor per chunk (as before) paid thread-creation cost ~n_chunks
    # times (hundreds per file); a single persistent pool removes that overhead
    # without changing what gets computed.
    _exe = ThreadPoolExecutor(max_workers=workers_)

    def _preprocess_block(a, b):
        return np.stack([_f.result() for _f in
                         [_exe.submit(fn, f, bg_radius) for f in stack[a:b]]])

    # ── GPU batch sizing ──────────────────────────────────────────────────────
    # Preprocessing must stay in small RAM-bounded sub-chunks (that's why the RAM
    # auto-tune shrank chunk_size), but the GPU/backend is far more efficient on
    # bigger batches — at chunk_size 32 the backend is re-invoked hundreds of
    # times per file, paying fixed per-call setup each time and starving the GPU.
    # So we DECOUPLE the two: preprocess in `chunk_size` sub-chunks, accumulate
    # them into a buffer, and localise once the buffer reaches `gpu_batch` frames.
    # Detection is unchanged because the percentile threshold is a deterministic,
    # frame-grouping-stable background estimate (see the Torch backend).
    gpu_batch = _resolve_gpu_batch(_impl, stack, chunk_size, n_frames,
                                   interactive=(preview_cb is not None))
    if gpu_batch > chunk_size:
        print(f"  GPU batch : {gpu_batch} frames/localise  "
              f"(preprocess in {chunk_size}-frame sub-chunks)")

    # ── First chunk: preprocess now so we can auto-detect minmass ─────────────
    first_end  = min(chunk_size, n_frames)
    first_pp = _preprocess_block(0, first_end)

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

    # ── Live preview: emit EVERY frame of each (buffered) batch after
    # localisation so the GUI's live view scrolls through the actual movie
    # rather than ticking once per chunk.  The GUI's repaint timer naturally
    # drops in-between frames it can't paint in time, so we just fire-and-forget
    # every frame — the message queue + per-frame cost is tiny next to
    # localisation itself.  (`locs_chunk["frame"]` is GLOBAL here.)
    def _emit_chunk_previews(chunk_pp, locs_chunk, frame_offset):
        if preview_cb is None or len(chunk_pp) == 0:
            return
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

    # ── GPU-batch accumulator: localise preprocessed sub-chunks in groups of
    # ~gpu_batch frames instead of one backend call per sub-chunk. ─────────────
    _buf = []                 # pending preprocessed sub-chunk arrays
    _buf_start = [None]       # global frame index of _buf[0]  (mutable cell)
    _gpu_batch_cur = [int(gpu_batch)]

    def _localise_buffer(batch):
        """Localise one accumulated buffer (frames LOCAL 0..len-1).  On a CUDA
        out-of-memory error, permanently halve the batch target, split, and
        retry — so a too-optimistic VRAM estimate degrades gracefully instead of
        crashing the run."""
        try:
            return _localise_chunk_via_backend(batch)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and len(batch) > chunk_size:
                _gpu_batch_cur[0] = max(int(chunk_size), len(batch) // 2)
                try:
                    import torch as _t
                    if _t.cuda.is_available():
                        _t.cuda.empty_cache()
                except Exception:
                    pass
                mid = len(batch) // 2
                a = _localise_buffer(batch[:mid])
                b = _localise_buffer(batch[mid:])
                if len(b):
                    b = b.copy(); b["frame"] += mid
                parts = [p for p in (a, b) if len(p)]
                return (pd.concat(parts, ignore_index=True)
                        if parts else a)
            raise

    def _flush():
        if not _buf:
            return
        base = _buf_start[0]
        batch = _buf[0] if len(_buf) == 1 else np.concatenate(_buf, axis=0)
        locs = _localise_buffer(batch)
        if len(locs) > 0:
            locs = locs.copy()
            locs["frame"] += base
            all_locs.append(locs)
            if mass_cb is not None and "mass" in locs.columns:
                try:    mass_cb(np.asarray(locs["mass"].values, dtype=np.float32))
                except Exception: pass
        _emit_chunk_previews(batch, locs, frame_offset=base)
        _buf.clear(); _buf_start[0] = None

    def _add(pp, start):
        if _buf_start[0] is None:
            _buf_start[0] = start
        _buf.append(pp)
        if sum(len(b) for b in _buf) >= _gpu_batch_cur[0]:
            _flush()

    # First chunk into the buffer (accumulators already updated above).
    _add(first_pp, 0)
    del first_pp

    # Remaining chunks
    for i in _tqdm(range(1, n_chunks), desc="  Streaming", unit="chunk", ncols=70):
        # Honour a stop request between chunks
        if stop_event is not None and stop_event.is_set():
            print("  Streaming stopped by user.")
            break

        start     = i * chunk_size
        end       = min(start + chunk_size, n_frames)
        chunk_pp = _preprocess_block(start, end)

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

        # Hand the preprocessed sub-chunk to the GPU-batch accumulator (it owns
        # the reference now and frees it on flush — so no `del` here).
        _add(chunk_pp, start)

        # numpy chunk buffers are refcount-freed on flush; a full cyclic GC
        # sweep every chunk (hundreds per file) is wasted work.  Sweep
        # occasionally instead.
        if (i & 15) == 0:
            gc.collect()

    _flush()                       # localise any frames still buffered

    _exe.shutdown(wait=True)

    # ── Mean projection (normalised) ──────────────────────────────────────────
    # Guard frame_count (defensive: a 0-frame stack can't reach here, but a
    # divide-by-zero would poison every downstream ROI/threshold with NaNs).
    if frame_count > 0:
        mean_proj = (mean_acc / frame_count).astype(np.float32)
    else:
        mean_proj = np.zeros(mean_acc.shape, dtype=np.float32)
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




_BACKEND_REGISTRY: list[type[LocaliserBackend]] = [
    TrackpyBackend, TorchBackend, AtrousWaveletBackend,
    GaussianMleBackend, RadialSymmetryBackend,
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

    Auto-selection logic (Torch-first; trackpy is NEVER auto-selected):
      1. Prefer TorchBackend on a GPU device (CUDA → MPS) when it passes the
         hot-path sanity check — the fastest configuration by far.
      2. Otherwise pick TorchBackend on CPU.  Torch-CPU now runs a parallel,
         multi-process localiser, so on a CPU-only box — including big
         many-core servers — it matches or beats trackpy while keeping the
         SAME Torch code path on every machine.
      3. Only if PyTorch isn't installed at all fall back to TrackpyBackend.

    Trackpy stays available as a deliberate MANUAL selection in the dropdown;
    auto never chooses it when Torch is present.  This keeps users on M-series
    Macs out of the MPS-OOM trap when their Metal context is degraded (e.g.
    after an aborted prior process): the sanity check fails, select_device()
    returns "cpu", and auto picks Torch-CPU.  After a reboot when MPS works
    again, auto picks the GPU automatically — the user never has to touch the
    dropdown.

    Accepts torch-device pins (`torch-mps`, `torch-cuda`, `torch-cpu`) that
    pre-set the device on the returned instance — used for benchmarking
    and to let users force a specific device path.
    """
    if name in (None, "", "auto"):
        # Smart-auto: Torch-first.  Order is CUDA → MPS → torch-CPU, with
        # trackpy reached only when PyTorch is absent entirely.
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
                # Only commit to a GPU if it passes the hot-path sanity check.
                # torch.cuda.is_available() returns True even for cards the
                # bundled CUDA build has no kernels for (e.g. a Pascal GTX 1060
                # under a CUDA-13 wheel); committing on that basis crashes
                # mid-localisation with cudaErrorNoKernelImageForDevice.  The
                # probe launches the real kernels and fails cleanly, so auto
                # then falls through to the Torch-CPU path on such machines.
                if (_torch.cuda.is_available()
                        and TorchBackend._device_sanity_check("cuda")):
                    inst = TorchBackend()
                    inst._forced_device = "cuda"
                    return inst
                if (hasattr(_torch.backends, "mps")
                        and _torch.backends.mps.is_available()
                        and TorchBackend._device_sanity_check("mps")):
                    inst = TorchBackend()
                    inst._forced_device = "mps"
                    return inst
            except Exception:
                pass
        # No healthy GPU → Torch on CPU.  Torch-CPU runs a parallel,
        # multi-process localiser (one worker per core-budget slice), so on a
        # CPU-only box — including big many-core servers like Falcon — it
        # matches or beats the trackpy path while keeping the SAME Torch code
        # on every machine.  Trackpy is never auto-selected: it stays a
        # deliberate manual choice in the Detection-backend dropdown.
        if TorchBackend.is_available():
            inst = TorchBackend()
            inst._forced_device = "cpu"
            return inst
        # Last resort: PyTorch isn't installed at all → reference CPU trackpy.
        for cls in _BACKEND_REGISTRY:
            if cls is TorchBackend:
                continue
            if cls.is_available():
                return cls()
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
                    "'Auto' or 'Crocker–Grier — PyTorch (CPU)' to continue "
                    "without GPU.")
            if short == "mps":
                has_mps = (hasattr(_torch.backends, "mps")
                           and _torch.backends.mps.is_available())
                if not has_mps:
                    raise RuntimeError(
                        "You selected the Apple MPS backend but this "
                        "system doesn't have MPS available "
                        "(MPS requires Apple Silicon + macOS 12+).\n\n"
                        "Change the Detection backend dropdown to 'Auto' "
                        "or 'Crocker–Grier — PyTorch (CPU)' to continue.")
        except RuntimeError:
            raise
        except Exception:
            # If we can't introspect torch for any reason, fall through
            # and let the original code path produce its native error.
            pass
        # The pinned GPU is present, but make sure the bundled torch actually
        # has kernels for it.  An unsupported architecture (e.g. Pascal under
        # CUDA 13) passes is_available() yet has no kernel image — rather than
        # crash mid-localisation, downgrade to CPU with a clear warning.
        if forced.split(":", 1)[0] in ("cuda", "mps"):
            try:
                if not TorchBackend._device_sanity_check(forced):
                    print(f"  WARNING: the selected Torch device '{forced}' is "
                          f"present but this PyTorch build has no usable kernels "
                          f"for it (likely an unsupported GPU architecture — e.g. "
                          f"a Pascal card under a CUDA-13 build). Falling back to "
                          f"PyTorch CPU.")
                    forced = "cpu"
            except Exception:
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


def backend_uses_gpu(name) -> bool:
    """True if resolving ``name`` yields a Torch backend that will run
    detection on a **GPU device** (CUDA / MPS).

    HYPER-FLY uses this to gate concurrent GPU detection: a single GPU shared
    by K file-workers would otherwise have them all pile onto its VRAM at once
    (likely OOM) and serialise anyway.  Returns False for Trackpy, Torch-CPU,
    and anything that can't be resolved — those run fully concurrently.

    Mirrors the device-resolution idiom used elsewhere (``_forced_device``
    pin → ``select_device()`` fallback)."""
    try:
        impl = _resolve_backend(name)
    except Exception:
        return False
    if not isinstance(impl, TorchBackend):
        return False
    dev = getattr(impl, "_forced_device", None)
    if dev is None and hasattr(impl, "select_device"):
        try:    dev = impl.select_device()
        except Exception: dev = None
    dev = str(dev or "").lower()
    return ("cuda" in dev) or ("mps" in dev)


def _gmm_crossover(means, sds, weights):
    """Return the log-mass value where the two 1-D Gaussian components have
    equal posterior weight, restricted to the interval between their means
    (the noise/signal boundary).  Solves w0·N(x|m0,s0) = w1·N(x|m1,s1)."""
    (m0, s0, w0), (m1, s1, w1) = sorted(zip(means, sds, weights))
    s0 = max(float(s0), 1e-6); s1 = max(float(s1), 1e-6)
    lo, hi = float(m0), float(m1)
    if hi <= lo:
        return lo
    # Quadratic a x² + b x + c = 0 from log-likelihood-ratio = 0.
    a = 1.0 / (2 * s0 * s0) - 1.0 / (2 * s1 * s1)
    b = m1 / (s1 * s1) - m0 / (s0 * s0)
    c = (m0 * m0) / (2 * s0 * s0) - (m1 * m1) / (2 * s1 * s1) \
        + np.log((max(w0, 1e-9) * s1) / (max(w1, 1e-9) * s0))
    roots = []
    if abs(a) < 1e-12:
        if abs(b) > 1e-12:
            roots = [-c / b]
    else:
        disc = b * b - 4 * a * c
        if disc >= 0:
            sq = np.sqrt(disc)
            roots = [(-b + sq) / (2 * a), (-b - sq) / (2 * a)]
    between = [r for r in roots if lo <= r <= hi]
    if between:
        return float(between[0])
    # Degenerate (root outside the means): fall back to the midpoint.
    return float(0.5 * (lo + hi))


def _knee_minmass(masses, n=60):
    """Knee of the count-vs-threshold survival curve, in log10(mass).  As the
    cutoff rises the surviving-spot count drops steeply through the noise then
    flattens over the real spots; the knee (point of maximum downward
    curvature) is what manual eyeballing approximates.  Returns a log10 value
    or None."""
    lm = np.log10(masses[masses > 0])
    if lm.size < 50:
        return None
    grid = np.linspace(lm.min(), lm.max(), n)
    counts = np.array([(lm >= g).sum() for g in grid], dtype=float)
    if counts.max() <= 0:
        return None
    x = (grid - grid.min()) / (float(np.ptp(grid)) or 1.0)
    y = counts / counts.max()
    # Distance below the chord from first to last point (kneedle, convex-down).
    chord = y[0] + (y[-1] - y[0]) * x
    diff = chord - y
    k = int(np.argmax(diff))
    if diff[k] <= 1e-3:
        return None
    return float(grid[k])


def _noise_floor_valley(masses, sep_min=0.5):
    """Candidate noise/signal valley for the linkability picker's floor, in
    LINEAR mass.

    Returns ``(valley, sep, w_min)``.  ``valley`` is the Bayes-optimal crossover
    between a low-mass mode and a higher-mass mode of a 2-component GMM fit to
    log10(mass), returned whenever the two modes are at least minimally usable
    (``sep = m_hi − m_lo ≥ sep_min`` dex apart, minor weight ≥2%); else None.

    This function only DETECTS a second mode — it does NOT decide whether to
    apply it as a floor.  That decision lives in `estimate_minmass`, because
    whether a low-mass second mode is NOISE (floor it) or real OVERLAPPING signal
    (keep it) depends on emitter DENSITY, which the caller knows: a strongly
    separated mode (≥0.85 dex) is unambiguous noise at any density, but a MODEST
    separation (0.5–0.85) is only trusted as noise on SPARSE data, where isolated
    real emitters form a tight single mass mode and can't manufacture a low-mass
    overlap mode.  At high density that low mode is merged/overlapping real spots
    and must not be cut.

    Why a floor is needed at all: the old count-vs-threshold kneedle returned a
    value UNCONDITIONALLY, so on a unimodal distribution (a detector that doesn't
    over-detect noise — trackpy — or dense all-real data) it found a spurious
    bend inside the single real mode and over-cut, collapsing recall (trackpy/SPT
    picked ~2.3 vs a true ~0.1; every detector on the dense EPFL set picked ~3–4
    and detected almost nothing).  Gating on a real second mode fixes that while
    still suppressing the genuine low-mass flood the PyTorch/à-trous detectors
    produce.  Validated on 33 cases (EPFL + palmTRACER + simulated density×SNR
    grid): the apply-gate in estimate_minmass lands the production threshold near
    the best-achievable F1 in every regime."""
    lm = np.log10(np.asarray(masses, dtype=float))
    lm = lm[np.isfinite(lm)]
    if lm.size < 200:
        return None, None, None
    try:
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(n_components=2, n_init=3,
                             random_state=0).fit(lm.reshape(-1, 1))
        means = gm.means_.ravel()
        sds = np.sqrt(gm.covariances_.ravel())
        wts = gm.weights_.ravel()
        order = np.argsort(means)
        sep = float(means[order][1] - means[order][0])
        w_min = float(wts.min())
        if sep >= sep_min and w_min >= 0.02:
            return float(10.0 ** _gmm_crossover(means, sds, wts)), sep, w_min
        return None, sep, w_min
    except Exception:
        return None, None, None


# ── Linkability-optimised auto-threshold ────────────────────────────────────
# A human picks the detection threshold by eyeballing single-frame spot
# brightness.  The signal a human CANNOT see is temporal linkability: a real
# emitter persists and links into a coherent ≥L-frame trajectory, whereas noise
# (shot noise, hot pixels, fixed-pattern flicker) makes 1-frame blips and
# 2–3-frame spurious fragments that the linker cannot stitch into real tracks
# (Jaqaman 2008).  We harvest every candidate at minmass=0 over a few
# CONTIGUOUS frame windows (linking needs frame adjacency), then sweep the mass
# threshold and pick the operating point that maximises real-track yield while
# suppressing spurious fragments — a criterion no single-frame human inspection
# can reproduce.

def _contiguous_windows(n_frames, n_windows=4, win_len=120):
    """Pick up to `n_windows` blocks of `win_len` CONSECUTIVE frames spread
    across the movie.  Linking needs adjacent frames, so (unlike the static
    estimator's evenly-spaced single frames) we sample contiguous runs and
    spread them to average over photobleaching.  Returns a list of (start,
    stop) half-open index pairs."""
    n_frames = int(n_frames)
    if n_frames <= 0:
        return []
    if n_frames < 250:
        return [(0, n_frames)]
    win_len = int(min(win_len, max(16, n_frames // n_windows)))
    if n_windows <= 1 or win_len >= n_frames:
        return [(0, min(win_len, n_frames))]
    starts = np.linspace(0, n_frames - win_len, n_windows).astype(int)
    wins = []
    for s in starts:
        s = int(s)
        if wins and s < wins[-1][1]:          # merge accidental overlaps
            continue
        wins.append((s, s + win_len))
    return wins


def _harvest_locate_one(args):
    """Process-pool worker: run the minmass=0 + characterize trackpy locate on
    ONE preprocessed window.  This is the EXACT same single-process `tp.batch`
    call (incl. the numba→default engine fallback) that the serial harvest runs
    inline, so the per-window result is byte-identical — only where it executes
    changes.  Module-level + tuple-arg so it pickles cleanly to a worker."""
    pp, diameter, percentile = args
    for kw in ({"engine": "numba"}, {}):
        try:
            return tp.batch(pp, diameter, minmass=0.0, percentile=percentile,
                            characterize=True, processes=1, **kw)
        except Exception:
            continue
    return None


def _harvest_windows(stack, windows, diameter, percentile,
                     bg_radius, bg_method, workers, backend_impl=None):
    """Detect every candidate at minmass=0 inside each contiguous window, with
    the PSF features (size / ecc) the quality gate needs plus mass / position
    for linking.  The harvest runs through the SAME backend the production run
    will use (`backend_impl`), so the chosen threshold is in that backend's
    native mass units — no cross-backend `_TP_MASS_SCALE` transfer:

      • trackpy (Crocker–Grier CPU, the default when `backend_impl` is None or
        names trackpy) → `tp.batch(..., characterize=True)`, fanned across a
        process pool (windows are independent; the analysis worker is non-daemon
        so nested pools are allowed; threads wouldn't help — numba locate holds
        the GIL).  Each worker runs the identical single-process `tp.batch`, so
        the harvested candidates are byte-identical to the serial path (proven
        by a regression test).  Falls back to serial for a single window, when
        FIREFLY_HARVEST_PARALLEL is off, or if the pool can't start.

      • torch family (Crocker–Grier PyTorch, à trous wavelet) → the backend's
        own `localise(..., minmass=0, characterize=True)` per window, which
        supplies torch-native size / ecc (trackpy-only `ep` / `signal` are
        simply absent — the gate skips a missing column).  The backend manages
        its own device / parallelism, so the windows are harvested serially.

    `frame` is kept LOCAL to each window (0..win_len) and tagged with
    `window_id`, so per-window linking never bridges the gap between windows.
    Returns `(H, pp0)` where H is a DataFrame with columns x, y, frame, mass,
    window_id (+ size, ecc, and trackpy's ep / signal when available) and `pp0`
    is window 0's preprocessed frames (retained for backward-compatibility with
    callers; preprocessing is per-frame independent, so `pp0[:k]` is identical
    to preprocessing the first k frames)."""
    cols = ["x", "y", "frame", "mass", "size", "ecc", "ep", "signal",
            "window_id"]
    # Preprocess every window up front in the main process (thread-parallel and
    # cheap next to the locate) so we can keep window 0's frames for the audit
    # AND ship ready-to-locate arrays to the pool.
    prepped = []                           # (wid, pp) in window order
    pp0 = None
    for wid, (s, e) in enumerate(windows):
        block = np.asarray(stack[s:e])
        if block.size == 0:
            continue
        pp = preprocess_stack(block, bg_radius=bg_radius,
                              bg_method=bg_method, workers=workers, quiet=True)
        if wid == 0:
            pp0 = pp                       # retained for the mass-scale audit
        prepped.append((wid, pp))
    if not prepped:
        return pd.DataFrame(columns=cols), pp0

    # ── Torch-family self-harvest (Crocker–Grier PyTorch / à trous) ──────────
    # Harvest with the run's OWN backend so the chosen minmass is in that
    # backend's native mass units — no trackpy harvest, no `_TP_MASS_SCALE`
    # cross-scale transfer.  The backend supplies torch-native size / ecc for
    # the quality gate (ep / signal stay trackpy-only; the gate skips them).
    if backend_impl is not None and getattr(backend_impl, "name", "") != "trackpy":
        parts = []
        for wid, pp in prepped:
            try:
                f = backend_impl.localise(
                    pp, diameter=diameter, minmass=0.0, percentile=percentile,
                    workers=workers, chunk_size=len(pp), quiet=True,
                    characterize=True)
            except Exception:
                f = None
            if f is None or len(f) == 0:
                continue
            f = f.copy()
            f["window_id"] = wid
            parts.append(f)
        if not parts:
            return pd.DataFrame(columns=cols), pp0
        H = pd.concat(parts, ignore_index=True)
        return H[[c for c in cols if c in H.columns]], pp0

    # ── Trackpy harvest (Crocker–Grier CPU): parallel tp.batch ───────────────
    args = [(pp, diameter, percentile) for _wid, pp in prepped]
    frames = None
    _parallel = os.environ.get("FIREFLY_HARVEST_PARALLEL", "1").strip().lower() \
        not in ("0", "false", "no", "off")
    if _parallel and len(prepped) >= 2 and N_CPUS > 1:
        try:
            with ProcessPoolExecutor(
                    max_workers=safe_process_workers(
                        min(len(prepped), N_CPUS))) as ex:
                # ex.map preserves input order → window order is unchanged.
                frames = list(ex.map(_harvest_locate_one, args))
        except Exception:
            frames = None                  # any pool failure → serial fallback
    if frames is None:
        frames = [_harvest_locate_one(a) for a in args]

    parts = []
    for (wid, _pp), f in zip(prepped, frames):
        if f is None or len(f) == 0:
            continue
        f = f.copy()
        f["window_id"] = wid
        parts.append(f)
    if not parts:
        return pd.DataFrame(columns=cols), pp0
    H = pd.concat(parts, ignore_index=True)
    return H[[c for c in cols if c in H.columns]], pp0


def _quality_pregate(H, diameter):
    """Drop obvious non-spots before the linkability sweep: trackpy `size`
    (radius of gyration) outside a diffraction-limited band, implausible
    eccentricity, or a grossly large localisation error `ep`.  This sharpens
    the sweep without doing the thresholding itself.  Refuses to fire if it
    would nuke (almost) everything (mis-tuned bands on unusual data).  Returns
    (filtered_df, n_dropped, info)."""
    if H is None or len(H) == 0:
        return H, 0, {}
    n0 = len(H)
    mask = np.ones(n0, dtype=bool)
    info = {}
    if "size" in H.columns:
        sz = H["size"].to_numpy(dtype=float)
        lo, hi = 0.5, float(diameter)          # generous band around the PSF
        mask &= np.isfinite(sz) & (sz >= lo) & (sz <= hi)
        info["size_band"] = [lo, hi]
    if "ecc" in H.columns:
        ec = H["ecc"].to_numpy(dtype=float)
        mask &= ~(np.isfinite(ec) & (ec > 0.9))
    if "ep" in H.columns:
        ep = H["ep"].to_numpy(dtype=float)
        mask &= ~(np.isfinite(ep) & (ep > float(diameter)))
    filt = H[mask].reset_index(drop=True)
    if len(filt) < max(50, 0.05 * n0):
        return H, 0, {"pregate": "skipped_too_aggressive"}
    return filt, int(n0 - len(filt)), info


def _link_track_lengths(sub, search_range, memory):
    """Link one window's surviving detections WITHOUT the stub filter and
    return the array of per-track frame counts (so we can count both good
    tracks and short spurious fragments).  Quiet — swallows the linker's
    chatter (stdout redirect is process-/thread-local to this call)."""
    if sub is None or len(sub) < 2:
        return np.array([], dtype=int)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            linked = _link_via_trackpy(
                sub[["x", "y", "frame"]].copy(),
                search_range=search_range, memory=memory)
    except Exception:
        return np.array([], dtype=int)
    if linked is None or "particle" not in getattr(linked, "columns", []) \
            or len(linked) == 0:
        return np.array([], dtype=int)
    return linked.groupby("particle")["frame"].count().to_numpy()


def _split_windows(H):
    """Split the harvested candidate table into per-window column dicts
    (numpy arrays only) so the sweep can be evaluated without re-slicing the
    DataFrame at every threshold, and so the data ships cheaply to worker
    processes."""
    if "window_id" in H.columns:
        wids = sorted(int(w) for w in H["window_id"].unique())
    else:
        wids = [0]
    has_ep = "ep" in H.columns
    out = []
    for wid in wids:
        sub = H[H["window_id"] == wid] if "window_id" in H.columns else H
        d = {"x": sub["x"].to_numpy(dtype=float),
             "y": sub["y"].to_numpy(dtype=float),
             "frame": sub["frame"].to_numpy(),
             "mass": sub["mass"].to_numpy(dtype=float)}
        if has_ep:
            d["ep"] = sub["ep"].to_numpy(dtype=float)
        out.append(d)
    return out


def _sweep_threshold_row(t, window_arrays, search_range, memory, link_min_len):
    """Evaluate one candidate threshold `t`: per window, keep mass≥t, link, and
    tally good/spurious tracks.  Identical maths to the original inline loop —
    factored out so the serial and parallel sweep paths share one code path and
    produce byte-identical numbers."""
    keeps = [w["mass"] >= t for w in window_arrays]
    n_surv = int(sum(int(k.sum()) for k in keeps))
    if n_surv < 4:
        return dict(t=float(t), n_surv=n_surv, N_good=0,
                    good_fraction=0.0, spurious_rate=0.0,
                    median_ep=float("nan"))
    n_good = good_det = spur_det = 0
    ep_parts = []
    for w, keep in zip(window_arrays, keeps):
        if "ep" in w:
            ep_parts.append(w["ep"][keep])
        if int(keep.sum()) < 2:
            continue
        sub = pd.DataFrame({"x": w["x"][keep], "y": w["y"][keep],
                            "frame": w["frame"][keep]})
        lens = _link_track_lengths(sub, search_range, memory)
        if lens.size == 0:
            continue
        good = lens[lens >= link_min_len]
        spur = lens[lens <= 2]
        n_good += int(good.size)
        good_det += int(good.sum())
        spur_det += int(spur.sum())
    if ep_parts:
        _ep = np.concatenate(ep_parts)
        med_ep = float(np.nanmedian(_ep)) if _ep.size else float("nan")
    else:
        med_ep = float("nan")
    return dict(t=float(t), n_surv=n_surv, N_good=int(n_good),
                good_fraction=float(good_det / n_surv),
                spurious_rate=float(spur_det / n_surv),
                median_ep=med_ep)


# Module-level worker glue for the parallel sweep.  The (potentially large)
# per-window candidate arrays are shipped to each worker ONCE via the pool
# initializer; each task then carries only the scalar threshold.
_SWEEP_POOL_STATE: dict = {}


def _sweep_pool_init(window_arrays, search_range, memory, link_min_len):
    _SWEEP_POOL_STATE["args"] = (window_arrays, search_range, memory,
                                 link_min_len)


def _sweep_pool_one(t):
    window_arrays, search_range, memory, link_min_len = _SWEEP_POOL_STATE["args"]
    return _sweep_threshold_row(t, window_arrays, search_range, memory,
                                link_min_len)


# Run the threshold sweep on a process pool only when there's enough work to
# amortise pool startup (Windows spawn ≈ 0.5–1 s); below this the serial path
# is faster.  Same grid, same linker → identical results either way.
_SWEEP_PARALLEL_MIN_CANDIDATES = 8000

# Auto-threshold harvest cap: max candidates kept per frame (brightest by mass)
# for the torch family ONLY, whose minmass=0 detection over-detects noise.  The
# cap bounds the linkability sweep's combinatorics, but it MUST stay ABOVE the
# real per-frame spot density: if it falls below, the brightest-N kept are ALL
# real spots with no noise tail, so the sweep can't locate the signal/noise
# boundary and over-thresholds (recall collapses — e.g. on a ~74-spot/frame
# stack, cap=40 picked minmass≈2.1 and dropped recall 0.78→0.18).  120 keeps the
# real population PLUS the noise tail on dense stacks while still capping a
# torch flood (~900/frame); recall recovers to 0.78 at precision 1.0, with margin
# before precision degrades (~200).  Verified across 64²/160²/256² stacks.
_HARVEST_MAX_PER_FRAME = 120


def _sweep_thresholds(H, grid, search_range, memory, link_min_len):
    """For each candidate threshold t: keep mass≥t, link each window, and
    measure track quality.  Returns a list of per-t dicts (t, n_surv, N_good,
    good_fraction, spurious_rate, median_ep), one per grid point, in grid order.

    Parallelised across grid points on a process pool when the candidate set is
    large; otherwise evaluated serially.  Both paths call `_sweep_threshold_row`
    so the numbers are identical."""
    window_arrays = _split_windows(H)
    grid = [float(t) for t in grid]
    n_cand = int(sum(w["mass"].size for w in window_arrays))

    if (N_CPUS > 1 and len(grid) >= 4
            and n_cand >= _SWEEP_PARALLEL_MIN_CANDIDATES):
        try:
            with ProcessPoolExecutor(
                    max_workers=safe_process_workers(min(N_CPUS, len(grid))),
                    initializer=_sweep_pool_init,
                    initargs=(window_arrays, search_range, memory,
                              link_min_len)) as ex:
                # ex.map preserves input order → rows stay in grid order.
                return list(ex.map(_sweep_pool_one, grid))
        except Exception:
            pass  # any pool failure → deterministic serial fallback

    return [_sweep_threshold_row(t, window_arrays, search_range, memory,
                                 link_min_len) for t in grid]


def _pick_linkability_threshold(sweep, sensitivity, max_false_track_rate,
                                noise_floor):
    """Choose the operating threshold from the sweep table, or return None when
    linkability is inconclusive (flat N_good curve → caller falls back to the
    static estimator).  Returns (minmass_or_None, info)."""
    if not sweep:
        return None, {"reason": "empty_sweep"}
    arr = lambda k: np.array([r[k] for r in sweep], dtype=float)
    t, Ng, gf, sr = arr("t"), arr("N_good"), arr("good_fraction"), arr("spurious_rate")
    ns = arr("n_surv")
    order = np.argsort(t)
    t, Ng, gf, sr, ns = t[order], Ng[order], gf[order], sr[order], ns[order]

    # Guards — the sweep only adds value over the static estimator when real
    # emitters link AND there is a suppressible spurious population to act on.
    #   • Nothing links (sparse / non-persistent / wrong search range) →
    #     N_good ≈ 0 at every threshold.  Defer to the static method.
    #   • No spurious population: good_fraction is already high at the LOWEST
    #     (most permissive) threshold, so even admitting everything yields
    #     mostly-good tracks — linkability cannot beat a static cut, and the
    #     N_good "knee" would be meaningless (the immobile-dominated case).
    if np.nanmax(Ng) < 5:
        return None, {"reason": "no_linkage", "N_good_max": float(np.nanmax(Ng))}
    if np.nanmin(gf) > 0.8:
        return None, {"reason": "no_spurious_population",
                      "min_good_fraction": float(np.nanmin(gf))}

    nf = float(noise_floor)

    # Advanced override: lowest (most permissive) t whose MEASURED spurious
    # fragment rate is within the user's ceiling, but never below the noise
    # floor.  This directly controls the false-track rate.
    if max_false_track_rate is not None:
        r = float(max_false_track_rate)
        ok = np.where((sr <= r) & (t >= nf))[0]
        if ok.size:
            return float(t[ok].min()), {
                "rule": "max_false_track_rate", "r": r,
                "spurious_at_pick": float(sr[ok][np.argmin(t[ok])])}

    # Operating point = F1 balance of purity vs real-detection recall.
    #   precision ≈ good_fraction (survivors that land in real ≥L tracks)
    #   recall    ≈ good-detection count relative to its best over the grid
    # F1 = 2·p·r/(p+r) peaks where the spurious population is suppressed WITHOUT
    # yet cutting the real spots.  N_good is deliberately NOT maximised: at high
    # thresholds real tracks fragment (gaps the linker can't bridge), inflating
    # the track COUNT while recall actually falls — F1 on detection counts is
    # immune to that gaming.
    good_det = gf * ns
    recall = good_det / (np.nanmax(good_det) or 1.0)
    prec = gf
    score = 2.0 * prec * recall / np.clip(prec + recall, 1e-9, None)
    op = int(np.nanargmax(score))

    # Knee of good_fraction vs log10(t), recorded only for the audit marker.
    lt = np.log10(np.clip(t, 1e-9, None))
    x = (lt - lt.min()) / (float(np.ptp(lt)) or 1.0)
    y = gf / (np.nanmax(gf) or 1.0)
    chord = y[0] + (y[-1] - y[0]) * x
    knee = int(np.argmax(np.abs(chord - y)))

    # Sensitivity = ±1 grid step along the precision/recall curve
    # (Strict → higher cut/purer, Lenient → lower cut/more recall).
    step = {"strict": +1, "balanced": 0, "lenient": -1}.get(
        str(sensitivity).lower(), 0)
    idx = int(np.clip(op + step, 0, len(t) - 1))

    # Floor the operating point at the shot-noise level.  `noise_floor` is the
    # bimodality-gated GMM noise/signal valley when a separable noise population
    # exists, else a no-op low floor (the 2nd mass percentile) — see
    # estimate_minmass / `_noise_floor_valley`.  So this clamp lifts the pick
    # ONLY when there is genuinely a low-mass noise flood to suppress (the
    # PyTorch/à-trous over-detection regime), and never above the real-spot mode
    # on detectors that don't over-detect (trackpy) or on dense all-real data,
    # where over-flooring collapsed recall in the old kneedle-always design.
    t_op = float(t[idx])
    chosen = max(t_op, nf)
    return chosen, {"rule": "f1_purity_recall", "op_index": int(op),
                    "chosen_index": int(idx),
                    "knee_index": int(knee), "sensitivity": str(sensitivity).lower(),
                    "N_good_at_op": float(Ng[op]),
                    "good_fraction_at_op": float(gf[op]),
                    "good_fraction_at_chosen": float(gf[idx]),
                    "noise_floor_applied": bool(nf > t_op),
                    "spurious_at_op": float(sr[op]),
                    "score_at_op": float(score[op])}


def _static_minmass(masses, sensitivity, diag, log_fn):
    """The original single-frame mass-distribution estimator (GMM noise/signal
    valley → calibrated mass-quantile → count-vs-threshold knee floor).  Used
    as the automatic, flagged FALLBACK when the linkability sweep is
    inconclusive (immobile-dominated / sparse data).  Operates on a 1-D
    `masses` array, mutates `diag` (writes `static_method`, knee, gmm_*), and
    returns the chosen minmass."""
    masses = np.asarray(masses, dtype=float)
    masses = masses[np.isfinite(masses) & (masses > 0)]
    lm = np.log10(masses)
    knee = _knee_minmass(masses)
    diag["knee"] = None if knee is None else float(knee)

    cross = None
    try:
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(n_components=2, n_init=3, random_state=0)
        gm.fit(lm.reshape(-1, 1))
        means = gm.means_.ravel()
        sds = np.sqrt(gm.covariances_.ravel())
        weights = gm.weights_.ravel()
        order = np.argsort(means)
        m_lo, m_hi = means[order]
        s_lo = sds[order][0]
        w_min = float(weights.min())
        diag["gmm_means"] = [float(means[order][0]), float(means[order][1])]
        diag["gmm_sds"] = [float(sds[order][0]), float(sds[order][1])]
        diag["gmm_weights"] = [float(weights[order][0]), float(weights[order][1])]
        if (m_hi - m_lo) >= 0.3 and w_min >= 0.02:
            cross = _gmm_crossover(means, sds, weights)
            diag["static_method"] = "gmm"
            diag["sigma_noise"] = float(s_lo)
        else:
            diag["static_method"] = "gmm_unimodal"
    except Exception as e:
        diag["static_method"] = f"gmm_failed:{type(e).__name__}"

    sens = str(sensitivity).lower()
    if cross is not None:
        diag["gmm_crossover"] = float(cross)
        sig = float(diag.get("sigma_noise", 0.1)) or 0.1
        shift = {"strict": +1.0, "balanced": 0.0, "lenient": -1.0}.get(sens, 0.0)
        mm = float(10.0 ** (cross + shift * sig))
        diag["static_method"] = "gmm_valley"
        if knee is not None and abs(knee - cross) > 0.4:
            log_fn(f"  Auto-threshold: GMM valley (10^{cross:.2f}) and knee "
                   f"(10^{knee:.2f}) disagree by >0.4 dex; both logged.")
    else:
        q = {"strict": 0.40, "balanced": 0.30, "lenient": 0.20}.get(sens, 0.30)
        mm = float(np.quantile(masses, q))
        diag["static_method"] = f"mass_quantile_p{int(round(q * 100))}"
        diag["quantile"] = q

    if knee is not None:
        mm_knee = float(10.0 ** knee)
        if mm_knee > mm * 1.02:
            log_fn(f"  Auto-threshold: chosen cut {mm:.4g} is below the "
                   f"noise/signal knee {mm_knee:.4g} (would admit noise) — "
                   f"raising to the knee.")
            mm = mm_knee
            diag["knee_floor_applied"] = True
            diag["static_method"] = (diag.get("static_method") or "") + "+knee_floor"
    return mm


def _audit_mass_scale(stack, windows, H, diameter, percentile,
                      bg_radius, bg_method, workers, backend, log_cb=None,
                      pp0=None):
    """Self-audit the trackpy↔Torch mass calibration.

    LEGACY / no longer wired into `estimate_minmass`: the auto-threshold harvest
    now runs through the run's OWN backend (see `_harvest_windows`), so the
    chosen minmass is already in that backend's native mass units and there is
    no trackpy→Torch `_TP_MASS_SCALE` transfer left to audit.  Retained (and
    still unit-tested) for its defensive no-op behaviour and in case a
    cross-backend transfer is ever reintroduced.

    Re-localises a few frames of the first window with the Torch backend and
    reports the empirical Torch/Trackpy bright-tail mass ratio (target ≈ 1.0).
    Best-effort: returns the ratio (float) or None, and never raises except on
    cancellation.  No-op on the trackpy backend.
    """
    _log = log_cb or print
    try:
        impl = _resolve_backend(backend)
        if getattr(impl, "name", "") != "torch" or not len(H) or not windows:
            return None
        s0, e0 = windows[0]
        e0 = min(e0, s0 + 32)                 # a handful of frames → robust median
        cap = e0 - s0
        # Reuse window 0's preprocessed frames from the harvest when available —
        # preprocessing is per-frame independent, so pp0[:cap] is bit-identical
        # to preprocessing stack[s0:e0] afresh (just without the redundant work).
        if pp0 is not None and len(pp0) >= cap:
            pp = np.asarray(pp0[:cap])
        else:
            blk = np.asarray(stack[s0:e0])
            if blk.size == 0:
                return None
            pp = preprocess_stack(blk, bg_radius=bg_radius,
                                  bg_method=bg_method, workers=workers,
                                  quiet=True)
        tdf = impl.localise(pp, diameter=diameter, minmass=0.0,
                            percentile=percentile, workers=workers,
                            chunk_size=len(pp), quiet=True)
        tm = np.asarray(tdf["mass"].values, dtype=float)
        tm = tm[np.isfinite(tm) & (tm > 0)]
        # Trackpy masses from the SAME physical frames (window 0, frame < cap).
        h0 = H[(H["window_id"] == 0) & (H["frame"] < (e0 - s0))]
        hm = np.asarray(h0["mass"].values, dtype=float)
        hm = hm[np.isfinite(hm) & (hm > 0)]
        if tm.size < 30 or hm.size < 30:
            return None
        # Compare the BRIGHT tail (p90), not the median: the detectors find
        # different numbers of noise blips at minmass=0, which skews the median,
        # whereas the upper tail is dominated by real spots in both — the
        # population the _TP_MASS_SCALE calibration actually targets.  This is a
        # coarse population-level sanity check, not a per-spot calibration.
        ratio = float(np.percentile(tm, 90) / np.percentile(hm, 90))
        # Only flag EGREGIOUS drift (>2x either way); modest deviation is normal
        # population variation and shouldn't cry wolf.
        egregious = not (0.5 <= ratio <= 2.0)
        _log(f"  Mass-scale check (Torch vs Trackpy, coarse): bright-tail ratio "
             f"= {ratio:.2f}  (≈1.0 = calibration holds)" +
             ("  — WARNING: large mismatch; auto minmass may transfer poorly to "
              "the Torch run — sanity-check detection or set minmass manually"
              if egregious else ""))
        return ratio
    except _Cancelled:
        raise
    except Exception:
        return None


def estimate_minmass(stack, diameter=7, percentile=64, backend="auto",
                     sensitivity="balanced", frame_sample=80,
                     bg_radius=50, bg_method="uniform_filter",
                     workers=N_CPUS, log_cb=None,
                     search_range=5, memory=3, link_min_len=4,
                     max_false_track_rate=None,
                     mode="linkability", target_density=None):
    """Estimate a robust per-file detection `minmass`, better than manual
    eyeballing.

    PRIMARY engine — linkability sweep: harvest every candidate at minmass=0
    over a few contiguous frame windows (with trackpy PSF features), apply a
    quality pre-gate, then sweep the mass threshold and link at each candidate.
    Real emitters persist into ≥L-frame tracks; noise makes 1-frame blips and
    2–3-frame fragments.  The operating point maximises good-track yield ×
    purity, floored at the noise level — a temporal criterion a single-frame
    human inspection cannot reproduce.  `Strict/Balanced/Lenient` shift it ±1
    grid step; an optional `max_false_track_rate` caps the measured spurious
    fragment rate directly.

    FALLBACK — static estimator (`_static_minmass`): on immobile-dominated or
    sparse data the N_good curve is flat and the sweep is inconclusive, so we
    fall back to the GMM-valley / mass-quantile / knee method and flag it in
    `diag["method"]` as `static_fallback:<reason>`.

    Returns (minmass: float, diagnostics: dict).
    """
    def _log(msg):
        # Encoding-safe: a default Windows (cp1252) console can't encode some of
        # the chars in these log lines (e.g. the '→' arrow, 'σ').  A raw print
        # would raise UnicodeEncodeError, which the outer handler below would
        # catch and SILENTLY downgrade the auto-threshold to the legacy heuristic
        # (a worse minmass) — so never let a log line abort estimation.
        try:
            (log_cb or print)(msg)
        except UnicodeEncodeError:
            import sys
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            try:
                (log_cb or print)(msg.encode(enc, "replace").decode(enc))
            except Exception:
                pass

    if diameter % 2 == 0:
        diameter += 1
    n = len(stack)
    diag = {"method": None, "n_candidates": 0, "sensitivity": sensitivity,
            "backend": backend, "gmm_crossover": None, "knee": None,
            "frame_sample": 0}

    # Hard clamp range for the returned value.
    MM_MIN, MM_MAX = 0.05, 1e6

    try:
        # ── Harvest contiguous windows (mass + PSF features, link-ready) ──────
        windows = _contiguous_windows(n)
        diag["windows"] = [[int(s), int(e)] for s, e in windows]
        diag["frame_sample"] = int(sum(e - s for s, e in windows))
        # Harvest through the SAME backend the run will use, so the chosen
        # minmass is in that backend's native mass units (the torch family no
        # longer borrows trackpy's threshold via `_TP_MASS_SCALE`).
        _impl = _resolve_backend(backend)
        _harvest_name = getattr(_impl, "name", "?")
        diag["harvest_backend"] = _harvest_name
        _t_harvest = time.perf_counter()
        _log(f"  Auto-threshold: harvesting candidate spots over "
             f"{len(windows)} window(s) ({diag['frame_sample']} frames) "
             f"via {_harvest_name}…")
        H, _pp0 = _harvest_windows(stack, windows, diameter, percentile,
                                   bg_radius, bg_method, workers,
                                   backend_impl=_impl)
        _log(f"  Auto-threshold: harvested {len(H):,} candidates in "
             f"{time.perf_counter() - _t_harvest:.1f}s")

        # ── Harvest density cap (torch family only) ──────────────────────────
        # The auto-threshold's signal/noise separation — the linkability sweep
        # AND the static GMM/knee — both break down once the per-frame candidate
        # count is high: dense points link by chance into long "tracks" (so the
        # spurious metric collapses and the picker is fooled to the floor) and
        # the noise so dominates the mass histogram that the GMM can't resolve
        # the signal mode.  Crocker–Grier on the Torch backend detects EVERY
        # positive bandpass maximum at minmass=0 (its clipped, mostly-zero
        # bandpass makes the percentile threshold ≈ 0), so on noisy data it
        # harvests ~10–70× more candidates than trackpy and trips exactly this
        # failure — whereas trackpy's stricter detection and à trous's wavelet
        # significance gate stay well under the cap on their own.  So cap ONLY
        # the torch-family harvest, keeping the brightest `_HARVEST_MAX_PER_FRAME`
        # by mass per frame: real (bright) emitters always survive, only the
        # faint noise excess is trimmed.  The cap is an absolute per-frame count
        # (not an FOV-scaled density): validated to recover the right threshold
        # across 64²/160²/256² stacks, where the separable regime held at
        # ~40/frame regardless of field size (it is the linker's combinatorics +
        # noise fraction that bite, not areal density).  Genuinely high-density
        # data (>~40 real emitters/frame) is the known limitation — there the
        # auto-estimate may sit slightly high and the user should set minmass
        # manually (auto-thresholding is unreliable at that density anyway).
        if (_harvest_name != "trackpy" and len(H) and "window_id" in H.columns
                and diag["frame_sample"] > 0):
            cap = _HARVEST_MAX_PER_FRAME
            per_frame = len(H) / float(diag["frame_sample"])
            if per_frame > cap:
                n_before = len(H)
                H = (H.sort_values("mass", ascending=False, kind="mergesort")
                     .groupby(["window_id", "frame"], sort=False).head(cap)
                     .reset_index(drop=True))
                diag["harvest_density_cap"] = int(cap)
                _log(f"  Auto-threshold: {_harvest_name} harvested "
                     f"{per_frame:.0f} candidates/frame (over-detection at "
                     f"minmass=0) — kept the brightest {cap}/frame "
                     f"({n_before:,} → {len(H):,}) so the signal/noise "
                     f"separation stays reliable.")

        masses = (np.asarray(H["mass"].values, dtype=float)
                  if len(H) else np.array([], dtype=float))
        masses = masses[np.isfinite(masses) & (masses > 0)]
        diag["n_candidates"] = int(masses.size)

        if masses.size < 200:
            med = float(np.median(masses)) if masses.size else MM_MIN
            mad = float(np.median(np.abs(masses - med))) if masses.size else 0.0
            mm = float(np.clip(med + 3.0 * 1.4826 * mad, MM_MIN, MM_MAX))
            diag["method"] = "noise_floor_lowN"
            diag["minmass"] = mm
            _log(f"  Auto-threshold: only {masses.size} candidates → noise "
                 f"floor minmass = {mm:.4g}")
            return mm, diag

        # ── Density-matched mode ─────────────────────────────────────────────
        # The linkability sweep optimises each file INDEPENDENTLY, so two
        # recordings can settle at different points on the threshold curve — and
        # the anomalous exponent moves along that curve.  A difference between
        # conditions can then be methodological rather than biological.  Matching
        # a common detections-per-frame across every recording removes the
        # threshold as a confound.  It is a pure quantile of the harvested
        # masses: no linking, so it is also much cheaper than the sweep.
        if str(mode).lower().startswith("density") and target_density:
            want = float(target_density) * float(diag["frame_sample"])
            achieved = float(masses.size) / float(diag["frame_sample"])
            if want >= masses.size:
                # Even keeping EVERY candidate is below target — a sparse or dim
                # recording.  Take the noise floor and flag it: silently running
                # a recording that cannot reach the common density would put an
                # unmatched file into the comparison.
                mm = float(np.clip(np.percentile(masses, 1.0), MM_MIN, MM_MAX))
                diag["method"] = "density_matched"
                diag["density_target"] = float(target_density)
                diag["density_achieved"] = achieved
                diag["qc"] = ("below_target_density: only "
                              f"{achieved:.1f} candidates/frame available vs a "
                              f"{float(target_density):.1f}/frame target")
                _log(f"  Auto-threshold: density target "
                     f"{float(target_density):.1f}/frame NOT reachable — only "
                     f"{achieved:.1f}/frame candidates exist; using the noise "
                     f"floor ({mm:.4g}).  This recording is sparser than the "
                     f"rest of the set.")
            else:
                pct = 100.0 * (1.0 - want / float(masses.size))
                mm = float(np.clip(np.percentile(masses, pct), MM_MIN, MM_MAX))
                diag["method"] = "density_matched"
                diag["density_target"] = float(target_density)
                diag["density_achieved"] = float((masses >= mm).sum()) / float(
                    diag["frame_sample"])
                _log(f"  Auto-threshold: density-matched to "
                     f"{float(target_density):.1f} spots/frame → "
                     f"minmass = {mm:.4g} "
                     f"(kept {diag['density_achieved']:.1f}/frame of "
                     f"{achieved:.1f}/frame candidates)")
            diag["minmass"] = mm
            return mm, diag

        # Reservoir-cap the masses used for the audit / static stats (NOT the
        # link harvest H, whose per-frame density must stay intact for linking).
        mstat = masses
        if mstat.size > 200_000:
            mstat = np.random.default_rng(0).choice(mstat, 200_000, replace=False)
        lm = np.log10(mstat)
        diag["_log_masses"] = (np.random.default_rng(1).choice(lm, 20000, replace=False)
                               if lm.size > 20000 else lm.copy())

        # Shot-noise floor for the linkability operating point.  We floor at the
        # GMM noise/signal valley ONLY when a low-mass second mode is genuinely
        # NOISE; otherwise there is nothing to suppress (a raw kneedle would find
        # a spurious bend inside the single real mode and over-cut, collapsing
        # recall — trackpy/SPT, the dense EPFL set) and we floor only at the 2nd
        # mass percentile (just below the population, never lifting the pick above
        # real spots).  Whether a detected low-mass mode is NOISE or real
        # OVERLAPPING signal depends on BOTH its separation AND areal CROWDING
        # (overlap is an areal phenomenon — emitters per PSF footprint):
        #   • strong separation (≥0.85 dex) → unambiguous noise at ANY crowding;
        #   • modest separation (0.5–0.85)  → trusted as noise ONLY on an
        #     UNCROWDED field, where isolated real emitters form one tight mass
        #     mode so a second mode must be noise.  On a CROWDED field the low
        #     mode is merged/overlapping real signal and must NOT be cut — e.g.
        #     dense PyTorch CG sits at sep≈0.75 with crowding ≈0.28 (real
        #     overlap), whereas à-trous on sparse low-SNR data sits at sep≈0.79
        #     with crowding ≈0.11 (genuine noise).
        # "Uncrowded" = candidate areal density `crowding` (candidates per PSF
        # footprint, π·(d/2)²/frame-area) below π/16 ≈ 0.2 — the value at which
        # the mean nearest-neighbour spacing equals one PSF diameter (overlap
        # onset), DIAMETER-INDEPENDENT since both crowding and spacing scale with
        # the PSF area.  Crowding (not raw count/frame) is the right scale: dense
        # data makes the detector UNDER-count merged peaks, so count/frame alone
        # mis-classifies a small crowded field as sparse.  Validated on 33 cases
        # (real + simulated density×SNR grid).
        # The kneedle is still computed for the audit figure's marker.
        _knee_log = _knee_minmass(mstat)
        diag["knee"] = None if _knee_log is None else float(_knee_log)
        _valley, _sep, _wmin = _noise_floor_valley(mstat)
        _cand_per_frame = masses.size / float(max(diag["frame_sample"], 1))
        _psf_area = np.pi * (diameter / 2.0) ** 2
        _crowding = _cand_per_frame * _psf_area / float(
            max(int(stack.shape[-2]) * int(stack.shape[-1]), 1))
        _apply_floor = _valley is not None and (
            _sep >= 0.85 or _crowding < np.pi / 16.0)
        if _apply_floor:
            noise_floor = float(_valley)
            diag["noise_floor_kind"] = "gmm_valley"
        else:
            noise_floor = float(np.percentile(mstat, 2))
            diag["noise_floor_kind"] = "p2_unimodal"
        diag["noise_floor"] = noise_floor
        diag["mass_bimodality"] = {"sep_dex": _sep, "w_min": _wmin,
                                   "cand_per_frame": float(_cand_per_frame),
                                   "crowding": float(_crowding)}

        # ── Linkability sweep (primary) ──────────────────────────────────────
        chosen = None
        try:
            Hq, n_drop, _qinfo = _quality_pregate(H, diameter)
            diag["quality_dropped"] = int(n_drop)
            mq = np.asarray(Hq["mass"].values, dtype=float)
            mq = mq[np.isfinite(mq) & (mq > 0)]
            if mq.size >= 200:
                p2, p98 = np.percentile(mq, [2, 98])
                # Span the grid DOWN to the shot-noise floor so real spots
                # survive at every candidate threshold and can link; the lowest
                # grid points (≈ floor) admit the spurious population so the
                # sweep can measure and then suppress it.
                lo = float(min(p2, noise_floor))
                hi = float(p98)
                if hi > lo:
                    grid = np.unique(np.geomspace(lo, hi, 18))
                    _t_sweep = time.perf_counter()
                    _log(f"  Auto-threshold: linkability sweep "
                         f"({len(grid)} thresholds × {len(windows)} window(s))…")
                    sweep = _sweep_thresholds(Hq, grid, int(search_range),
                                              int(memory), int(link_min_len))
                    _log(f"  Auto-threshold: sweep done in "
                         f"{time.perf_counter() - _t_sweep:.1f}s")
                    pick, pinfo = _pick_linkability_threshold(
                        sweep, sensitivity, max_false_track_rate, noise_floor)
                    diag["link_info"] = pinfo
                    if pick is not None:
                        chosen = float(pick)
                        diag["method"] = "linkability"
                        diag["sweep"] = sweep
                        _op = int(pinfo.get("op_index", 0))
                        _op = max(0, min(_op, len(sweep) - 1))
                        diag["n_good"] = int(sweep[_op]["N_good"])
                        diag["good_fraction"] = float(sweep[_op]["good_fraction"])
                        diag["spurious_rate"] = float(sweep[_op]["spurious_rate"])
                        diag["score"] = float(pinfo.get("score_at_op", 0.0))
                    else:
                        diag["link_fallback_reason"] = pinfo.get("reason", "inconclusive")
                else:
                    diag["link_fallback_reason"] = "degenerate_grid"
            else:
                diag["link_fallback_reason"] = "sparse_after_pregate"
        except _Cancelled:
            raise
        except Exception as e:
            diag["link_fallback_reason"] = f"error:{type(e).__name__}"

        if chosen is None:
            # ── Static fallback (flagged) ────────────────────────────────────
            reason = diag.get("link_fallback_reason", "inconclusive")
            mm = _static_minmass(mstat, sensitivity, diag, _log)
            diag["method"] = f"static_fallback:{reason}"
            diag["static_minmass"] = float(mm)
        else:
            mm = chosen

        mm = float(np.clip(mm, MM_MIN, MM_MAX))
        diag["minmass"] = mm
        _log(f"  Auto-threshold [{diag['method']}, {str(sensitivity).lower()}]: "
             f"minmass = {mm:.4g}  (from {diag['n_candidates']} candidates over "
             f"{diag['frame_sample']} frames in {len(windows)} window(s), "
             f"harvested via {_harvest_name})")
        # No trackpy↔Torch mass-scale audit needed: the harvest above already
        # ran through the run's own backend, so the threshold is natively in its
        # mass units (the `_TP_MASS_SCALE` cross-scale transfer is gone).
        return mm, diag

    except _Cancelled:
        raise
    except Exception as e:
        # Last-ditch fallback: the legacy peak×d²/8 heuristic on a sample frame.
        try:
            f = stack[min(5, n - 1)]
            pp = _preprocess_fast(np.asarray(f), bg_radius=bg_radius)
            peak = float(np.percentile(pp, 99))
            mm = float(np.clip(peak * (diameter ** 2) / 8.0, MM_MIN, MM_MAX))
        except Exception:
            mm = 1.0
        diag["method"] = f"fallback_heuristic:{type(e).__name__}"
        diag["minmass"] = mm
        _log(f"  Auto-threshold failed ({type(e).__name__}: {e}); "
             f"fallback minmass = {mm:.4g}")
        return mm, diag


def render_minmass_audit(diagnostics, path, theme="Dark", stem=""):
    """Write an audit figure for the auto-threshold.  Left panel: the
    log10(mass) histogram of candidate spots, any fitted noise/signal Gaussian
    components, the knee, and the chosen cutoff.  Right panel (only when the
    linkability sweep ran): real-track yield (N_good), spurious-fragment rate
    and good-fraction versus log-threshold, with the chosen operating point and
    knee marked — so the per-file threshold is fully inspectable.  Returns the
    path on success, else None."""
    lm = diagnostics.get("_log_masses")
    if lm is None or len(lm) < 10:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from firefly.analysis.fa_theme import _theme_palette
        pal = _theme_palette(theme)
        lm = np.asarray(lm, dtype=float)

        sweep = diagnostics.get("sweep")
        has_sweep = bool(sweep) and len(sweep) >= 3
        if has_sweep:
            fig, (ax, ax2) = plt.subplots(
                1, 2, figsize=(11.2, 4.4), facecolor=pal["BG"])
        else:
            fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=pal["BG"])
            ax2 = None

        # ── Left: mass histogram ─────────────────────────────────────────────
        ax.set_facecolor(pal["PNL"])
        counts, edges, _ = ax.hist(lm, bins=80, color=pal["BAR_FILL"],
                                   edgecolor=pal["GRD"], linewidth=0.3)
        binw = float(edges[1] - edges[0])
        n = lm.size

        means = diagnostics.get("gmm_means")
        sds = diagnostics.get("gmm_sds")
        wts = diagnostics.get("gmm_weights")
        # Only overlay the fitted noise/signal components when the cut was
        # actually placed at their valley (static GMM fallback).
        if (means and sds and wts
                and str(diagnostics.get("static_method", "")).startswith("gmm_valley")):
            xs = np.linspace(lm.min(), lm.max(), 400)
            cols = [pal["MUT"], pal["SIG"]]
            labels = ["noise component", "signal component"]
            for (m, s, w, c, lab) in zip(means, sds, wts, cols, labels):
                s = max(s, 1e-6)
                pdf = (w / (s * np.sqrt(2 * np.pi))
                       * np.exp(-0.5 * ((xs - m) / s) ** 2))
                ax.plot(xs, pdf * n * binw, color=c, lw=2.0, label=lab)

        chosen = diagnostics.get("minmass")
        if chosen and chosen > 0:
            ax.axvline(np.log10(chosen), color="#FFD33D", lw=2.4,
                       label=f"chosen minmass = {chosen:.3g}")
        nf = diagnostics.get("noise_floor")
        if nf and nf > 0:
            ax.axvline(np.log10(nf), color=pal["MUT"], ls="--", lw=1.0,
                       alpha=0.7, label="noise floor")
        knee = diagnostics.get("knee")
        if knee is not None:
            ax.axvline(knee, color=pal["TXT"], ls=":", lw=1.2, alpha=0.7,
                       label="count knee")

        ax.set_xlabel("log₁₀( candidate spot mass )", color=pal["TXT"])
        ax.set_ylabel("count", color=pal["TXT"])
        for sp in ax.spines.values():
            sp.set_edgecolor(pal["GRD"])
        ax.tick_params(colors=pal["TXT"])
        # Legend BELOW the axes (horizontal) so it can never sit on top of the
        # histogram / curves.  savefig(bbox_inches="tight") expands the canvas to
        # include it, so nothing is clipped.
        ax.legend(frameon=False, fontsize=8, labelcolor=pal["TXT"],
                  loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)

        # ── Right: linkability sweep ─────────────────────────────────────────
        if has_sweep:
            ax2.set_facecolor(pal["PNL"])
            t = np.array([r["t"] for r in sweep], dtype=float)
            lt = np.log10(np.clip(t, 1e-12, None))
            Ng = np.array([r["N_good"] for r in sweep], dtype=float)
            sr = np.array([r["spurious_rate"] for r in sweep], dtype=float)
            gf = np.array([r["good_fraction"] for r in sweep], dtype=float)
            o = np.argsort(lt)
            lt, Ng, sr, gf = lt[o], Ng[o], sr[o], gf[o]

            l1, = ax2.plot(lt, Ng, color=pal["SIG"], lw=2.0, marker="o",
                           ms=3, label="real tracks (N_good)")
            ax2.set_xlabel("log₁₀( mass threshold )", color=pal["TXT"])
            ax2.set_ylabel("real tracks  (N_good)", color=pal["SIG"])
            ax2.tick_params(colors=pal["TXT"])
            for sp in ax2.spines.values():
                sp.set_edgecolor(pal["GRD"])

            axr = ax2.twinx()
            axr.set_facecolor("none")
            l2, = axr.plot(lt, sr, color="#FF6B6B", lw=1.6, ls="--",
                           marker="s", ms=2.5, label="spurious-fragment rate")
            l3, = axr.plot(lt, gf, color="#FFD33D", lw=1.4, ls=":",
                           marker="^", ms=2.5, label="good fraction")
            axr.set_ylabel("fragment rate / good fraction", color=pal["TXT"])
            axr.tick_params(colors=pal["TXT"])
            axr.set_ylim(0, 1.02)
            for sp in axr.spines.values():
                sp.set_edgecolor(pal["GRD"])

            if chosen and chosen > 0:
                ax2.axvline(np.log10(chosen), color="#FFD33D", lw=2.2,
                            label="chosen")
            info = diagnostics.get("link_info") or {}
            ki = info.get("knee_index")
            if isinstance(ki, int) and 0 <= ki < len(t):
                ax2.axvline(np.log10(max(t[ki], 1e-12)), color=pal["TXT"],
                            ls=":", lw=1.0, alpha=0.6, label="knee")
            ax2.legend(handles=[l1, l2, l3], frameon=False, fontsize=7.5,
                       labelcolor=pal["TXT"], loc="upper center",
                       bbox_to_anchor=(0.5, -0.16), ncol=3)

        title = "Auto-threshold audit"
        if stem:
            title += f" — {stem}"
        sub = (f"{diagnostics.get('method','?')} · {diagnostics.get('sensitivity','?')} · "
               f"{diagnostics.get('n_candidates','?')} candidates · "
               f"{diagnostics.get('backend','?')} backend")
        fig.suptitle(title, color=pal["TXT"], fontsize=12, fontweight="bold", y=0.99)
        fig.text(0.5, 0.93, sub, color=pal["MUT"], fontsize=8.5, ha="center")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(path, dpi=150, facecolor=pal["BG"], bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as exc:
        print(f"  Auto-threshold audit plot skipped ({type(exc).__name__}: {exc})")
        return None


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
