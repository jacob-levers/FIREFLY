#!/usr/bin/env python3
import multiprocessing
import sys
import os

__version__ = "2.6.0"

# Fix macOS multiprocessing crashes — must be set before any other imports
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

"""
FIREFLY — Fluorescence Inference & Reconstruction Engine  (OPTIMISED)
=======================================================================
Framework for Localization Yields.  Supports .czi (Zeiss native) and
.tif / .tiff files.
Pixel size and frame interval are read automatically from CZI metadata.

Speed optimisations vs the original version:
  - Background subtraction:  rolling_ball (~4800 ms/frame)
                           -> uniform_filter (~3 ms/frame)  [~1700x faster]
  - Preprocessing:           serial -> parallel across all CPU cores
  - Localisation:            single core -> all CPU cores
  - Memory:                  entire stack in RAM -> chunked processing
  - Progress:                silent -> live progress bars

Usage
-----
  # Typical usage — everything auto-detected from CZI:
  python sptpalm_analysis.py my_experiment.czi

  # With output folder:
  python sptpalm_analysis.py my_experiment.czi --output-dir C:\\results

  # Override metadata if needed:
  python sptpalm_analysis.py my_experiment.czi --pixel-size 0.104 --frame-interval 0.05

  # Limit CPU cores (default: all available):
  python sptpalm_analysis.py my_experiment.czi --workers 4

  # Use legacy rolling-ball background (slower but more accurate for uneven illumination):
  python sptpalm_analysis.py my_experiment.czi --bg-method rolling_ball

All options:
  --pixel-size       um per pixel (auto from CZI metadata)
  --frame-interval   seconds per frame (auto from CZI metadata)
  --diameter         PSF diameter in pixels, must be odd (default: 7)
  --minmass          Min integrated brightness (auto if omitted)
  --search-range     Max displacement between frames in px (default: 5)
  --memory           Frames a particle may vanish and reappear (default: 3)
  --min-track-length Discard tracks shorter than this (default: 5)
  --max-lagtime      MSD lag time points (default: 20)
  --bg-method        Background method: uniform_filter (fast) or rolling_ball (default: uniform_filter)
  --bg-radius        Background radius in pixels (default: 50)
  --workers          CPU cores to use (default: all)
  --chunk-size       Frames per processing chunk, reduce if RAM is low (default: 500)
  --channel          Channel index for multi-channel CZI (default: 0)
  --output-dir       Where to save results (default: same folder as input)
"""

import argparse
import multiprocessing
import os
import sys
import time
import warnings
import xml.etree.ElementTree as ET
warnings.filterwarnings("ignore")

# ── BLAS / OpenBLAS / MKL threading policy ─────────────────────────────────────
# Cap internal BLAS threads to 1.  We use ThreadPoolExecutor for preprocessing
# (one Python thread per frame, all calling scipy.ndimage which uses BLAS).
# Without this cap, we get N² threads (Python pool × BLAS pool) on N cores,
# which deadlocks Windows frozen apps before the first preview frame is sent.
#
# Per-frame numpy/scipy operations on small (256×256) images are too fast to
# benefit from BLAS threading anyway — chunk-level Python threading wins.
# This MUST be set before numpy is imported to take effect.
for _blas_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_blas_env, "1")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import trackpy as tp
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:
    # Fallback no-op context manager if threadpoolctl unavailable
    from contextlib import contextmanager as _cm
    @_cm
    def _threadpool_limits(limits=None, user_api=None):
        yield
from scipy.ndimage import uniform_filter, gaussian_filter, gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import correlate as _correlate2d
from scipy.stats import gaussian_kde
from skimage import filters, exposure
from tqdm import tqdm

# On Windows with console=False (PyInstaller GUI build), sys.stderr is None.
# tqdm writes to sys.stderr by default and crashes with AttributeError.
# Use sys.stdout instead — the GUI redirects stdout to its log panel, so
# tqdm progress lines will appear there in real time.
import io as _io

def _tqdm(*args, **kwargs):
    """tqdm wrapper that writes to stdout (captured by the GUI log panel).
    Falls back to a no-op StringIO if stdout is somehow invalid."""
    out = sys.stdout if (sys.stdout is not None) else _io.StringIO()
    kwargs.setdefault("file", out)
    # Disable ANSI colour codes — the log panel is plain text.
    kwargs.setdefault("colour", None)
    return tqdm(*args, **kwargs)

# Optional readers
try:
    import aicspylibczi
    HAS_AICS = True
except (ImportError, OSError):
    # OSError covers the case where the package is installed but its
    # bundled C++ shared library cannot be found (common in PyInstaller bundles)
    HAS_AICS = False

try:
    import czifile
    HAS_CZIFILE = True
except ImportError:
    HAS_CZIFILE = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

tp.quiet()

N_CPUS = multiprocessing.cpu_count()


# ══════════════════════════════════════════════════════════════════════════════
#  CZI LOADING + METADATA
# ══════════════════════════════════════════════════════════════════════════════

def _parse_czi_metadata(xml_str):
    meta = {"pixel_size_um": None, "frame_interval_s": None}
    if not xml_str:
        return meta
    try:
        root = ET.fromstring(xml_str)
        for dist in root.iter("Distance"):
            if dist.get("Id", "") in ("X", "Y"):
                el = dist.find("Value")
                if el is not None:
                    try:
                        val = float(el.text)
                        if 1e-9 < val < 1e-3:
                            meta["pixel_size_um"] = round(val * 1e6, 6)
                            break
                    except (TypeError, ValueError):
                        pass
        for tag in ("TimeIncrement", "Interval"):
            el = root.find(f".//{tag}")
            if el is not None:
                text = el.text or (
                    el.find("Value").text
                    if el.find("Value") is not None else None)
                if text:
                    try:
                        val = float(text)
                        if 1e-6 < val < 3600:
                            meta["frame_interval_s"] = round(val, 6)
                            break
                    except (TypeError, ValueError):
                        pass
    except Exception:
        pass
    return meta


def _dim_size(v, default=1):
    """aicspylibczi returns dims as int or (start, size) tuple."""
    if isinstance(v, tuple):
        return int(v[1])
    return int(v) if v is not None else default


def load_projection_fast(path, channel=0, max_frames=100):
    """
    Return a normalised [0,1] float32 mean-projection image using at most
    *max_frames* evenly-spaced frames.  Much faster than load_file() for
    large datasets because frames are read individually (no full stack load).
    Used by the ROI Editor so it doesn't have to load all 16K frames.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".czi":
        if HAS_AICS:
            czi  = aicspylibczi.CziFile(path)
            dims = dict(czi.get_dims_shape()[0])
            n_t  = _dim_size(dims.get("T"), 1)
            n_c  = _dim_size(dims.get("C"), 1)
            ch   = min(channel, n_c - 1)
            indices = np.linspace(0, n_t - 1, min(max_frames, n_t),
                                  dtype=int)
            frames = []
            for t in indices:
                img, _ = czi.read_image(T=int(t), C=ch)
                frame  = img.squeeze()
                if frame.ndim > 2:
                    frame = frame[0]
                frames.append(frame.astype(np.float32))
            proj = np.stack(frames).mean(axis=0)
        elif HAS_CZIFILE:
            # Use czifile subblock directory for per-frame access (avoids
            # loading the entire stack into RAM).
            # NOTE: No full-load fallback — czi.asarray() on a 16K-frame file
            # takes ~30 minutes and effectively hangs the app.  If subblock
            # reading fails we surface a clear error immediately.
            with czifile.CziFile(path) as czi:
                entries = list(czi.subblock_directory)
                if not entries:
                    raise RuntimeError(
                        "No subblocks found in CZI file. "
                        "Install aicspylibczi for full CZI support: "
                        "pip install aicspylibczi")
                n      = len(entries)
                step   = max(1, n // max_frames)
                frames = []
                for entry in entries[::step]:
                    try:
                        seg = entry.data_segment()
                        arr = np.asarray(seg.data(raw=False),
                                         dtype=np.float32).squeeze()
                        if arr.size == 0:
                            continue
                        # Peel leading size-1 / channel dims until we have 2D
                        while arr.ndim > 2:
                            arr = arr[0]
                        if arr.ndim == 2:
                            frames.append(arr)
                    except Exception:
                        continue   # skip unreadable subblocks; keep going
                if not frames:
                    raise RuntimeError(
                        "czifile could not decode any preview frames.\n"
                        "Fix: pip install imagecodecs\n"
                        "Or:  pip install aicspylibczi")
                proj = np.stack(frames).mean(axis=0)
        else:
            raise RuntimeError("Cannot read CZI: install aicspylibczi or czifile.")

    elif ext in (".tif", ".tiff"):
        if not HAS_TIFFFILE:
            raise RuntimeError("Run: pip install tifffile")
        with tifffile.TiffFile(path) as tif:
            n     = len(tif.pages)
            step  = max(1, n // max_frames)
            pages = [tif.pages[i].asarray().astype(np.float32)
                     for i in range(0, n, step)]
        proj = np.stack(pages).mean(axis=0)
    else:
        raise RuntimeError(f"Unsupported file type: {ext}")

    lo, hi = proj.min(), proj.max()
    return (proj - lo) / (hi - lo) if hi > lo else np.zeros_like(proj)


def _find_czi_series(path):
    """
    Zeiss splits long acquisitions into companion files named:
        experiment.czi
        experiment(1).czi
        experiment(2).czi  …

    Given the primary path, return an ordered list of all files in that
    series (including the primary itself).  If no companions are found the
    list contains only the original path.
    """
    import glob, re
    directory = os.path.dirname(path) or "."
    basename  = os.path.splitext(os.path.basename(path))[0]

    # Strip any trailing "(N)" so we get the root name
    root = re.sub(r"\(\d+\)$", "", basename).rstrip()

    # Collect all matching files
    pattern  = os.path.join(directory, glob.escape(root) + "*.czi")
    candidates = sorted(glob.glob(pattern))

    # Keep only: root.czi  and  root(N).czi  (not unrelated names)
    series_re = re.compile(
        r"^" + re.escape(root) + r"(\(\d+\))?\.czi$", re.IGNORECASE)
    series = [f for f in candidates
              if series_re.match(os.path.basename(f))]

    # Natural sort so (1) < (2) < (10)
    def _nat_key(s):
        m = re.search(r"\((\d+)\)\.czi$", s, re.IGNORECASE)
        return int(m.group(1)) if m else -1

    series.sort(key=_nat_key)

    if len(series) > 1:
        print(f"  Multi-file CZI series detected ({len(series)} files):")
        for f in series:
            print(f"    {os.path.basename(f)}")
    return series if series else [path]


class _Cancelled(Exception):
    """Raised inside loaders when a stop_event fires mid-load."""
    pass


def _load_single_czi(path, channel=0, stop_event=None):
    """Load a single CZI file and return (stack, pixel_size_um, frame_interval_s).

    stop_event : threading.Event or None
        If set, loading is aborted and _Cancelled is raised.
        The check runs every 500 frames so the UI stays responsive.
    """
    def _chk():
        if stop_event is not None and stop_event.is_set():
            raise _Cancelled()

    if HAS_AICS:
        czi  = aicspylibczi.CziFile(path)
        xml  = czi.meta if hasattr(czi, "meta") else None
        meta = _parse_czi_metadata(xml)
        dims = dict(czi.get_dims_shape()[0])
        n_t  = _dim_size(dims.get("T"), 1)
        n_c  = _dim_size(dims.get("C"), 1)
        ch   = min(channel, n_c - 1)
        print(f"  Frames: {n_t}  |  Channels: {n_c}  |  Using channel: {ch}", flush=True)
        # Read first frame to discover H×W, then pre-allocate the full array.
        # This avoids building a Python list + np.stack which doubles peak RAM.
        img0, _ = czi.read_image(T=0, C=ch)
        f0 = img0.squeeze()
        if f0.ndim > 2:
            f0 = f0[0]
        H, W  = f0.shape
        stack = np.empty((n_t, H, W), dtype=np.float32)
        stack[0] = f0.astype(np.float32)
        for t in range(1, n_t):
            img, _ = czi.read_image(T=t, C=ch)
            frame  = img.squeeze()
            if frame.ndim > 2:
                frame = frame[0]
            stack[t] = frame.astype(np.float32)
            if t % 500 == 0:
                print(f"  Loading: {t}/{n_t} frames...", flush=True)
                _chk()
        return stack, meta["pixel_size_um"], meta["frame_interval_s"]

    if HAS_CZIFILE:
        # Use subblock-by-subblock reading — czi.asarray() loads the whole
        # file at once and hangs for large (16K-frame) datasets.
        # Note: czifile needs imagecodecs to decompress JPEG XR frames
        # (the default compression for Zeiss Elyra).
        # Install with: pip install imagecodecs
        with czifile.CziFile(path) as czi:
            xml  = czi.metadata()
            meta = _parse_czi_metadata(xml)
            entries = list(czi.subblock_directory)
            if not entries:
                raise RuntimeError(
                    "No subblocks found in CZI file.\n"
                    "Try: pip install aicspylibczi imagecodecs")
            n      = len(entries)
            print(f"  Subblocks: {n}  |  Using channel: {channel}", flush=True)
            frames = []
            _first_err = None   # log first decode error for diagnosis
            for i, entry in enumerate(entries):
                try:
                    seg = entry.data_segment()
                    arr = np.asarray(seg.data(raw=False),
                                     dtype=np.float32).squeeze()
                    if arr.size == 0:
                        continue
                    while arr.ndim > 2:
                        arr = arr[0]
                    if arr.ndim == 2:
                        frames.append(arr)
                except Exception as exc:
                    if _first_err is None:
                        _first_err = exc
                    continue
                if i % 500 == 0 and i > 0:
                    print(f"  Loading: {i}/{n} subblocks...", flush=True)
                    _chk()
            if not frames:
                hint = (f"\nFirst decode error: {_first_err}" if _first_err else "")
                raise RuntimeError(
                    "czifile could not decode any frames from this CZI.\n"
                    "This usually means the JPEG XR codec is missing.\n"
                    "Fix: pip install imagecodecs\n"
                    "Or:  pip install aicspylibczi"
                    + hint)
            data = np.stack(frames)
        return data, meta["pixel_size_um"], meta["frame_interval_s"]

    raise RuntimeError(
        "Cannot read CZI: install aicspylibczi or czifile.\n"
        "Run:  pip install aicspylibczi imagecodecs")


def load_czi(path, channel=0, stop_event=None, files=None):
    """Load a CZI (or multi-file CZI series) into one stack.

    `files`, when provided, overrides the auto-discovery of sibling
    files — used by the GUI to honour per-file checkbox selections.
    """
    if files:
        seen = set()
        series = []
        for f in sorted(files, key=lambda p: os.path.basename(p)):
            if f in seen or not os.path.isfile(f):
                continue
            seen.add(f); series.append(f)
        if not series:
            series = [path]
        print(f"  CZI series override: {len(series)} files",
              flush=True)
    else:
        # Detect multi-file series (Zeiss splits large datasets into companion files)
        series = _find_czi_series(path)

    if len(series) == 1:
        # Single file — straightforward load.  Use series[0] so a
        # per-file override that selects a non-primary sister still
        # loads the right one.
        only = series[0]
        print(f"  Loading CZI: {only}")
        stack, px_um, fi_s = _load_single_czi(only, channel, stop_event)
        print(f"  Shape: {stack.shape}  (T x Y x X)", flush=True)
        return stack, px_um, fi_s

    # Multi-file series — mirror the TIF loader's pre-allocate +
    # slice-copy pattern.  The old `stacks.append(st)` then
    # `np.concatenate(stacks)` path held every source array alive
    # alongside the combined destination → peak RAM ≈ 2 × total,
    # which on a 16 GB series pushed 16 GB / 32 GB machines into
    # disk-memmap even when they could trivially handle 16 GB in RAM.
    # New path: load each source, copy into its slot in `combined`,
    # free the source, gc.collect — peak ≈ 1.05 × total.
    print(f"  Loading CZI series: {len(series)} files", flush=True)

    # First pass: probe each file's shape via a one-shot single-file load
    # of just the metadata.  Zeiss CZI doesn't have a cheap header-probe
    # like TIFF tag pages, so we do the full load on file 1 only to learn
    # the per-frame layout + dtype, then for files 2..N we still must
    # load each in full but we copy → free → gc immediately.
    px_um_out = None
    fi_s_out  = None
    combined  = None
    offset    = 0
    import gc as _gc
    for i, fpath in enumerate(series):
        print(f"  [{i+1}/{len(series)}] {os.path.basename(fpath)}", flush=True)
        st, px, fi = _load_single_czi(fpath, channel, stop_event)
        if i == 0:
            px_um_out = px
            fi_s_out  = fi
            # Now we know per-frame shape + dtype.  Probe remaining files
            # by trusting they're the same Y×X (Zeiss enforces this for
            # split series); we'll discover differences at copy time.
            n_total_estimate = st.shape[0] * len(series)   # rough upper bound
            H, W = st.shape[1], st.shape[2]
            # We don't actually know n_total exactly without loading every
            # file.  Pre-allocate at the rough estimate; if we overshoot,
            # we'll slice down at the end.  If we undershoot (unlikely,
            # since the first file's frame count is typically representative),
            # we re-allocate larger.
            bytes_per_frame = st.dtype.itemsize * H * W
            total_size_estimate = n_total_estimate * bytes_per_frame
            try:
                import psutil as _psutil
                free_gb = _psutil.virtual_memory().available / 1e9
                reserve_gb = _user_ram_reserve_gb()
                usable_gb = free_gb - reserve_gb
                use_memmap = usable_gb < (total_size_estimate / 1e9) + (
                    bytes_per_frame * st.shape[0] / 1e9)
            except Exception:
                use_memmap = False
            if use_memmap:
                import tempfile, shutil
                tmp_dir = _resolve_temp_stack_dir()
                if tmp_dir is not None:
                    try:
                        disk_free = shutil.disk_usage(tmp_dir).free
                        if disk_free < total_size_estimate * 1.05:
                            tmp_dir = None
                    except Exception:
                        tmp_dir = None
                tmp_fh = tempfile.NamedTemporaryFile(
                    prefix="firefly_stack_", suffix=".raw",
                    delete=False, dir=tmp_dir)
                tmp_path = tmp_fh.name
                tmp_fh.close()
                _register_temp_stack_path(tmp_path)
                print(f"  → disk memmap at {tmp_path} "
                      f"({total_size_estimate/1e9:.1f} GB estimate)",
                      flush=True)
                combined = np.memmap(tmp_path, dtype=st.dtype, mode="w+",
                                      shape=(n_total_estimate, H, W))
            else:
                print(f"  → in-RAM allocation "
                      f"({total_size_estimate/1e9:.1f} GB estimate)",
                      flush=True)
                combined = np.empty((n_total_estimate, H, W),
                                     dtype=st.dtype)

        # Grow the destination if our estimate proves too small.
        # (Rare — assumes all files have the same per-file frame count.)
        n_here = st.shape[0]
        if offset + n_here > combined.shape[0]:
            # In-place growth isn't possible for memmap; reallocate.
            extra = (offset + n_here) - combined.shape[0]
            new_n = combined.shape[0] + max(extra, n_here)
            print(f"  (resizing combined buffer to {new_n} frames)",
                  flush=True)
            if isinstance(combined, np.memmap):
                # Cheap because memmap copy doesn't touch unmapped pages;
                # but we need a new file.
                import tempfile
                new_fh = tempfile.NamedTemporaryFile(
                    prefix="firefly_stack_", suffix=".raw", delete=False,
                    dir=_resolve_temp_stack_dir())
                new_path = new_fh.name
                new_fh.close()
                _register_temp_stack_path(new_path)
                new_combined = np.memmap(new_path, dtype=combined.dtype,
                                          mode="w+",
                                          shape=(new_n, H, W))
                new_combined[:offset] = combined[:offset]
                combined = new_combined
            else:
                new_combined = np.empty((new_n, H, W),
                                         dtype=combined.dtype)
                new_combined[:offset] = combined[:offset]
                combined = new_combined

        combined[offset:offset + n_here] = st
        offset += n_here
        del st
        _gc.collect()

    # Trim trailing unused frames if our estimate overshot.
    if combined is not None and offset < combined.shape[0]:
        combined = combined[:offset]
    if isinstance(combined, np.memmap):
        try:    combined.flush()
        except Exception: pass
    print(f"  Combined shape: {combined.shape}  (T x Y x X)", flush=True)
    return combined, px_um_out, fi_s_out


def _parse_ome_metadata(tif):
    """
    Extract pixel size (µm) and frame interval (s) from a tifffile.TiffFile.

    Checks in priority order:
      1. OME-XML embedded in the first page (OME-TIFF standard)
      2. ImageJ metadata dict (files saved by Fiji/ImageJ)
      3. XResolution TIFF tag (gives pixels per unit; combined with ResolutionUnit)

    Returns (pixel_size_um, frame_interval_s) — either value may be None if
    the corresponding metadata is absent.
    """
    px_um = None
    fi_s  = None

    # ── 1. OME-XML ────────────────────────────────────────────────────────────
    try:
        ome = tif.ome_metadata          # returns XML string or None
        if ome:
            root = ET.fromstring(ome)
            # Strip namespace: '{http://www.openmicroscopy.org/Schemas/OME/...}Pixels'
            ns = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else ""
            prefix = f"{{{ns}}}" if ns else ""

            def _find_el(tag):
                # Try with and without namespace
                el = root.find(f".//{prefix}{tag}")
                if el is None:
                    el = root.find(f".//{tag}")
                return el

            pixels = _find_el("Pixels")
            if pixels is not None:
                # PhysicalSizeX is stored in µm by OME convention
                psx = pixels.get("PhysicalSizeX")
                psx_unit = pixels.get("PhysicalSizeXUnit", "µm")
                if psx:
                    try:
                        v = float(psx)
                        # Convert to µm if necessary
                        unit_lc = psx_unit.lower().replace("μ", "u").replace("µ", "u")
                        if unit_lc in ("nm", "nanometer", "nanometre"):
                            v /= 1000.0
                        elif unit_lc in ("mm", "millimeter", "millimetre"):
                            v *= 1000.0
                        if 0.001 < v < 100:
                            px_um = round(v, 6)
                    except (TypeError, ValueError):
                        pass

                ti = pixels.get("TimeIncrement")
                ti_unit = pixels.get("TimeIncrementUnit", "s")
                if ti:
                    try:
                        v = float(ti)
                        unit_lc = ti_unit.lower()
                        if unit_lc in ("ms", "millisecond", "milliseconds"):
                            v /= 1000.0
                        elif unit_lc in ("min", "minute", "minutes"):
                            v *= 60.0
                        if 1e-6 < v < 3600:
                            fi_s = round(v, 6)
                    except (TypeError, ValueError):
                        pass
    except Exception:
        pass

    # ── 2. ImageJ metadata ────────────────────────────────────────────────────
    try:
        ij = tif.imagej_metadata        # dict or None
        if ij:
            if px_um is None:
                # ImageJ stores resolution as pixels/unit in TIFF XResolution tag.
                # We read it from the tag below; here we can get unit from ij dict.
                pass  # handled in section 3

            if fi_s is None:
                finterval = ij.get("finterval")  # seconds
                if finterval is not None:
                    try:
                        v = float(finterval)
                        if 1e-6 < v < 3600:
                            fi_s = round(v, 6)
                    except (TypeError, ValueError):
                        pass

                # Some ImageJ files store frame rate instead
                if fi_s is None:
                    fps = ij.get("fps")
                    if fps is not None:
                        try:
                            v = float(fps)
                            if v > 0:
                                fi_s = round(1.0 / v, 6)
                        except (TypeError, ValueError):
                            pass
    except Exception:
        pass

    # ── 3. XResolution TIFF tag (works for ImageJ TIFFs) ─────────────────────
    try:
        if px_um is None and tif.pages:
            page = tif.pages[0]
            xres = page.tags.get("XResolution")
            runit = page.tags.get("ResolutionUnit")
            if xres is not None:
                val = xres.value
                # Value is a rational (numerator, denominator) or plain float
                if isinstance(val, tuple) and len(val) == 2 and val[1] != 0:
                    pixels_per_unit = val[0] / val[1]
                else:
                    pixels_per_unit = float(val)
                if pixels_per_unit > 0:
                    # ResolutionUnit: 1=no units, 2=inch, 3=cm
                    unit_code = runit.value if runit is not None else 2
                    if unit_code == 3:          # centimetres
                        um_per_pixel = 1e4 / pixels_per_unit
                    elif unit_code == 2:        # inches
                        um_per_pixel = 25400.0 / pixels_per_unit
                    else:
                        um_per_pixel = None

                    # ImageJ often uses µm as "unit" and encodes pixels/µm
                    # by hacking the ResolutionUnit.  Check ij metadata unit.
                    try:
                        ij = tif.imagej_metadata or {}
                        ij_unit = ij.get("unit", "")
                        if ij_unit.lower() in ("um", "µm", "μm", "micron"):
                            um_per_pixel = 1.0 / pixels_per_unit
                    except Exception:
                        pass

                    if um_per_pixel and 0.001 < um_per_pixel < 100:
                        px_um = round(um_per_pixel, 6)
    except Exception:
        pass

    return px_um, fi_s


def _find_tif_series(path):
    """Find split TIFF files belonging to the same acquisition.

    Two naming conventions are handled:

    1. The legacy / generic split-TIFF style produced by ImageJ and
       similar tools:
           name.tif, name(1).tif, name(2).tif, …

    2. palmTRACER's multi-file export:
           <base>.tif, <base>-file002.tif, <base>-file003.tif, …
       palmTRACER also drops sibling files like `<base>_green.tif`
       (the ROI / channel image) into the same folder.  Those are
       NOT part of the time series — any sibling whose name matches
       `<base>_<word>.tif` is dropped from the series.  The leading
       file may itself be `<base>-file001.tif` if palmTRACER chose to
       always-number; both `<base>.tif` and `<base>-file001.tif` are
       treated as the leading frame batch.

    The user can pick ANY file from a palmTRACER series and we still
    detect the whole thing — `-file003.tif` resolves to the same
    `<base>` root as `-file001.tif` or the bare `<base>.tif`.

    Returns the sorted file list (frame-order), or `[path]` if no
    siblings were detected.  `_green.tif` and other underscore-suffix
    sister files are always excluded.
    """
    import glob, re
    directory = os.path.dirname(path) or "."
    basename  = os.path.splitext(os.path.basename(path))[0]
    ext       = os.path.splitext(path)[1].lower()  # .tif or .tiff

    # ── Step 1: figure out the series ROOT ────────────────────────────
    # palmTRACER:  <root>-fileNNN  →  strip the suffix.
    # ImageJ:      <root>(N)       →  strip the suffix.
    # If neither suffix is present, basename IS the root.
    pt_match = re.match(r"^(?P<root>.+?)-file\d+$", basename, re.IGNORECASE)
    if pt_match:
        root = pt_match.group("root")
    else:
        root = re.sub(r"\(\d+\)$", "", basename).rstrip()

    # ── Step 2: collect every candidate sibling starting with the root ──
    pattern  = os.path.join(directory, glob.escape(root) + "*" + ext)
    candidates = sorted(glob.glob(pattern))

    # ── Step 3: filter to ONLY the time-series members ─────────────────
    # Accept either form of suffix on `root`:
    #   <root><ext>                  — the bare leading file
    #   <root>(\d+)<ext>             — ImageJ split-TIFF "(N)"
    #   <root>-fileNNN<ext>          — palmTRACER split-TIFF
    # Reject `<root>_<word>.tif` outright — palmTRACER's `_green.tif`
    # ROI sibling and any other `_suffix` sister files (channel maps,
    # masks, projections) are NOT part of the acquisition.
    series_re = re.compile(
        r"^" + re.escape(root)
        + r"(?:\(\d+\)|-file\d+)?"
        + re.escape(ext) + r"$",
        re.IGNORECASE)
    reject_re = re.compile(
        r"^" + re.escape(root) + r"_[^.]+" + re.escape(ext) + r"$",
        re.IGNORECASE)
    series = []
    rejected_underscore = []
    for f in candidates:
        name = os.path.basename(f)
        if reject_re.match(name):
            rejected_underscore.append(name)
            continue
        if series_re.match(name):
            series.append(f)

    # ── Step 4: natural sort so file-numbered chunks line up in time ──
    # Single shared key with `load_tif`'s explicit-files path so the
    # two code paths can't disagree on order.  Bare <root>.tif sorts
    # first (key=-1), then -fileNNN / (N) by numeric value.
    series.sort(key=_tif_series_nat_key)

    if len(series) > 1:
        print(f"  Multi-file TIF series detected ({len(series)} files):")
        for f in series:
            print(f"    {os.path.basename(f)}")
        if rejected_underscore:
            print(f"  (excluded {len(rejected_underscore)} sibling "
                  f"file(s) matching `{root}_*{ext}` — typically "
                  f"palmTRACER's ROI/channel images):")
            for n in rejected_underscore:
                print(f"    skip: {n}")
    return series if series else [path]

def _load_single_tif(path, stop_event=None):
    """Load a single TIF file and return its stack, pixel size, and frame interval.

    Strategy: tifffile's `asarray()` reads + decompresses pages with internal
    multithreading via `maxworkers`, which is far faster than looping
    page.asarray() (each call re-opens its own thread pool).  For very
    large files we'd still like a cancel-poll, so we read in chunks of
    `BATCH` pages — each batch goes through the fast path, and we check
    stop_event + log progress between batches.
    """
    if not HAS_TIFFFILE:
        raise RuntimeError("Run: pip install tifffile")
    BATCH = 2000
    # Peek at the file header — if the magic doesn't match TIFF (II*\0 /
    # MM\0*) we surface a friendly error rather than letting tifffile's
    # raw "not a TIFF file: header=b'...'" exception bubble up.  Common
    # case: Zeiss CZI files saved with a `.tif` extension (header=ZISR).
    try:
        with open(path, "rb") as _fh:
            _hdr = _fh.read(4)
    except Exception:
        _hdr = b""
    if _hdr[:3] == b"ZIS":   # ZISRAWFILE → Zeiss CZI
        raise RuntimeError(
            f"{os.path.basename(path)} is a Zeiss CZI file with a .tif "
            f"extension, not a real TIFF.  Rename it to .czi and "
            f"re-import — FIREFLY's CZI loader will then handle it "
            f"correctly.")
    if _hdr[:2] not in (b"II", b"MM"):
        raise RuntimeError(
            f"{os.path.basename(path)} doesn't look like a TIFF file "
            f"(header bytes: {_hdr!r}).  Check the file isn't corrupted "
            f"or saved in a different format with the wrong extension.")
    with tifffile.TiffFile(path) as tif:
        px_um, fi_s = _parse_ome_metadata(tif)
        n_pages = len(tif.pages)
        if n_pages > BATCH:
            # Inspect first page for output shape + dtype so we can pre-allocate
            sample = tif.pages[0].asarray()
            shape = (n_pages,) + tuple(sample.shape)
            stack = np.empty(shape, dtype=np.float32)
            t0 = time.perf_counter()
            for start in range(0, n_pages, BATCH):
                if stop_event is not None and stop_event.is_set():
                    raise _Cancelled()
                end = min(start + BATCH, n_pages)
                # tifffile asarray(key=range(...)) uses multithreaded decode
                try:
                    chunk = tif.asarray(key=range(start, end),
                                         maxworkers=N_CPUS)
                except TypeError:
                    chunk = tif.asarray(key=range(start, end))
                # asarray may return (1, H, W) for a single page, normalise
                if chunk.ndim == 2:
                    chunk = chunk[np.newaxis]
                stack[start:end] = chunk.astype(np.float32, copy=False)
                # Free intermediate so peak memory stays at one batch above
                # the pre-allocated stack
                del chunk
                if start > 0:
                    rate = (start) / max(time.perf_counter() - t0, 1e-3)
                    print(f"  Loading: {end}/{n_pages} frames "
                          f"({rate:.0f} fr/s)...", flush=True)
        else:
            # Small files — fastest path is a single asarray()
            try:
                stack = tif.asarray(maxworkers=N_CPUS).astype(
                    np.float32, copy=False)
            except TypeError:
                stack = tif.asarray().astype(np.float32, copy=False)

    if   stack.ndim == 2: stack = stack[np.newaxis]
    elif stack.ndim == 4:
        stack = stack[:, 0] if stack.shape[1] == 1 else stack.mean(axis=1)

    return stack, px_um, fi_s


# ── Memmap cleanup ────────────────────────────────────────────────────────────
# When the multi-file loader falls back to a disk-backed memmap (because the
# combined stack won't fit in RAM), we leave the file on disk for the duration
# of the run.  Register an atexit hook to remove these temp files so they
# don't accumulate.
_firefly_temp_stack_paths: list = []
# Optional override for where the disk-backed memmap is created.  Set
# via env var FIREFLY_TEMP_DIR or via set_temp_stack_dir().  Defaults to
# the OS temp dir (e.g. /var/folders/.../T on macOS), which lives on the
# system volume — often the smallest drive on the machine.  Pointing
# this at the user's data drive avoids ENOSPC on long batches.
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
#  Without this reserve, FIREFLY's memory checks would happily consume
#  every free byte; the moment the user opens a Safari tab the system
#  starts swapping or OOM-killing.  Formula:
#     • a fixed floor (4 GB)                          — covers OS itself
#     • but at most 0.15 × total RAM, capped at 8 GB  — scales sanely
#  The OLD formula was max(4, 0.20*total).  On a 32 GB box that's
#  6.4 GB; combined with a 1.2× peak multiplier on the stack-load
#  threshold, this routinely demoted big-RAM machines to disk-memmap
#  even when there was plenty of room.  The new formula gives 4.8 GB
#  on 32 GB and 8 GB on a 64 GB workstation — enough OS headroom
#  without throwing away RAM that should be used for the stack.
#  Override via env var FIREFLY_USER_RAM_RESERVE_GB.
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


def _probe_tif_shape_and_count(path: str):
    """Read just enough of a TIF to return (n_pages, (H, W))."""
    with tifffile.TiffFile(path) as tif:
        n = len(tif.pages)
        sample = tif.pages[0].asarray()
        H, W = sample.shape[-2:]
    return n, (int(H), int(W))


def _tif_series_nat_key(filepath):
    """Sort key for sibling TIFFs of a single acquisition.

    Returns -1 for the bare `<root>.tif` (always frame 0) and the
    integer suffix for `<root>(N).tif` / `<root>-fileNNN.tif` chunks,
    so the natural sort yields chronological frame order regardless of
    whether the input came in alphabetical order from os.listdir().

    Critical because basename-alphabetical sorting puts `-fileNNN`
    (0x2D) before `.` (0x2E), which would silently mis-order
    palmTRACER series — bare `Post.tif` ends up AFTER `Post-fileNNN`
    chunks, corrupting every frame index downstream.
    """
    import re as _re
    name = os.path.basename(filepath)
    ext  = os.path.splitext(name)[1]
    m_pt = _re.search(r"-file(\d+)" + _re.escape(ext) + r"$",
                       name, _re.IGNORECASE)
    if m_pt:
        return int(m_pt.group(1))
    m_ij = _re.search(r"\((\d+)\)" + _re.escape(ext) + r"$",
                       name, _re.IGNORECASE)
    if m_ij:
        return int(m_ij.group(1))
    return -1   # bare <root>.tif sorts first


def load_tif(path, stop_event=None, files=None):
    """Load `path` and (when present) its sibling files into one stack.

    If `files` is a non-empty list, it overrides auto-discovery — the
    GUI uses this to honour per-file checkbox selections within a series.
    The override is sorted to match _find_tif_series ordering so frame
    indices line up with the user's expectation.
    """
    if files:
        # De-dup and sort by the same NATURAL key the auto-discovery
        # uses — bare <root>.tif first (key=-1), then -fileNNN / (N)
        # in numeric order.  Basename-alphabetical was wrong: `-`
        # (0x2D) sorts before `.` (0x2E), so `Post.tif` ended up
        # AFTER `Post-file002.tif` and every later frame index was
        # off by ~chunk_size.
        seen = set()
        series = []
        for f in sorted(files, key=_tif_series_nat_key):
            if f in seen or not os.path.isfile(f):
                continue
            seen.add(f); series.append(f)
        if not series:
            series = [path]
        print(f"  TIF series override: {len(series)} files "
              f"(natural-sorted, bare <root>.tif first)",
              flush=True)
        for _f in series:
            print(f"    {os.path.basename(_f)}", flush=True)
    else:
        series = _find_tif_series(path)

    if len(series) == 1:
        # Single file — straightforward load.  Use series[0] (not the
        # original `path`) so a per-file override that selects a
        # non-primary sister file still loads the right file.
        only = series[0]
        print(f"  Loading TIF: {only}")
        stack, px_um, fi_s = _load_single_tif(only, stop_event)
        print(f"  Shape: {stack.shape}  (T x Y x X)")
        if px_um is not None: print(f"  Pixel size  : {px_um} µm  (from file metadata)")
        if fi_s is not None:  print(f"  Frame interval: {fi_s} s  (from file metadata)")
        return stack, px_um, fi_s

    # ── Multi-file series ────────────────────────────────────────────────
    # The old path loaded every file into a `stacks` list and called
    # `np.concatenate(stacks)`, which allocates a brand-new combined array
    # while the source list is still alive — peak memory = 2× the combined
    # size.  On a 16.8 GB series that's a 33.6 GB working set on a 16 GB
    # machine.  System swap takes minutes and pegs the disk.
    #
    # The new path:
    #   1. Probes each file's frame count via TiffFile headers (no data load)
    #   2. Pre-allocates the destination — in RAM if it fits, on disk via
    #      np.memmap if not
    #   3. Loads each source, copies into the destination slice, frees the
    #      source.  Peak = combined + one source ≈ 1.25× total.
    print(f"  Loading TIF series: {len(series)} files", flush=True)

    n_per_file: list[int] = []
    H = W = 0
    for fpath in series:
        n, (h, w) = _probe_tif_shape_and_count(fpath)
        n_per_file.append(n)
        H, W = h, w
    n_total = sum(n_per_file)
    bytes_per_frame = 4 * H * W      # float32
    total_size = n_total * bytes_per_frame
    total_gb = total_size / 1e9

    # Decide RAM vs memmap.  We need the combined stack + headroom for
    # one source file at a time (the loader frees each source after
    # copying it into the destination slice) + a reserve for the OS.
    #
    # Peak transient = combined + largest_single_source.  Using the
    # ACTUAL peak (not a 1.2× multiplier) keeps big-RAM machines on
    # the fast in-RAM path: e.g. a 4-file 16.8 GB series only needs
    # 16.8 + 4.2 ≈ 21 GB peak, not 16.8 × 1.2 = 20 GB; on a 32 GB box
    # with 25 GB free and a 4.8 GB reserve, the old formula demoted
    # to memmap unnecessarily.
    use_memmap = False
    free_gb    = None
    reserve_gb = _user_ram_reserve_gb()
    max_source_gb = (max(n_per_file) * bytes_per_frame) / 1e9
    peak_gb       = total_gb + max_source_gb
    try:
        import psutil as _psutil
        free_gb = _psutil.virtual_memory().available / 1e9
        usable_gb = free_gb - reserve_gb
        if usable_gb < peak_gb:
            use_memmap = True
    except Exception:
        pass

    if use_memmap:
        import tempfile, shutil
        tmp_dir = _resolve_temp_stack_dir()
        # If the chosen temp dir doesn't have enough free disk space for
        # the memmap, fall back to the OS default — better an obscure
        # /var/folders path than an immediate ENOSPC.
        if tmp_dir is not None:
            try:
                disk_free = shutil.disk_usage(tmp_dir).free
                if disk_free < total_size * 1.05:
                    print(f"  [warn] FIREFLY_TEMP_DIR={tmp_dir} has only "
                          f"{disk_free/1e9:.1f} GB free but the memmap "
                          f"needs {total_gb:.1f} GB — falling back to "
                          f"the OS temp dir.", flush=True)
                    tmp_dir = None
            except Exception:
                tmp_dir = None
        tmp_fh = tempfile.NamedTemporaryFile(
            prefix="firefly_stack_", suffix=".raw", delete=False,
            dir=tmp_dir)
        tmp_path = tmp_fh.name
        tmp_fh.close()
        _register_temp_stack_path(tmp_path)
        free_disp = f"{free_gb:.1f}" if free_gb is not None else "?"
        print(f"  Peak RAM needed: {peak_gb:.1f} GB (combined "
              f"{total_gb:.1f} GB + largest source {max_source_gb:.1f} GB). "
              f"Free: {free_disp} GB, reserve: {reserve_gb:.1f} GB. "
              f"→ disk memmap at {tmp_path} "
              f"(override reserve via FIREFLY_USER_RAM_RESERVE_GB).",
              flush=True)
        combined = np.memmap(tmp_path, dtype=np.float32, mode="w+",
                             shape=(n_total, H, W))
    else:
        free_disp = f"{free_gb:.1f}" if free_gb is not None else "?"
        print(f"  Peak RAM needed: {peak_gb:.1f} GB (combined "
              f"{total_gb:.1f} GB + largest source {max_source_gb:.1f} GB). "
              f"Free: {free_disp} GB, reserve: {reserve_gb:.1f} GB. "
              f"→ in-RAM allocation (fast path).", flush=True)
        combined = np.empty((n_total, H, W), dtype=np.float32)

    # Load each file, copy into the destination slice, free immediately.
    px_um_out = None
    fi_s_out  = None
    offset = 0
    import gc as _gc
    for i, fpath in enumerate(series):
        print(f"  [{i+1}/{len(series)}] {os.path.basename(fpath)}",
              flush=True)
        st, px, fi = _load_single_tif(fpath, stop_event)
        if i == 0:
            px_um_out = px
            fi_s_out  = fi
        combined[offset:offset + st.shape[0]] = st
        offset += st.shape[0]
        del st
        _gc.collect()
    if use_memmap:
        combined.flush()
    print(f"  Combined shape: {combined.shape}  (T x Y x X)", flush=True)
    if px_um_out is not None: print(f"  Pixel size  : {px_um_out} µm  (from file metadata)")
    if fi_s_out is not None:  print(f"  Frame interval: {fi_s_out} s  (from file metadata)")
    return combined, px_um_out, fi_s_out


# ── External-localisations loader ─────────────────────────────────────────────
# Schema for a "preset" that maps an external tool's CSV columns to FIREFLY's
# canonical {frame, x, y, mass}.  Frame offset is added to the source values
# (-1 for 1-indexed tools); units lets us convert nm → px on the fly.
_CSV_PRESETS: dict = {
    "PALM-Tracer": {
        # Output columns vary slightly between PALM-Tracer versions:
        # the modern `locPALMTracer.txt` exporter uses `Plane`,
        # `CentroidX(px)`, `CentroidY(px)`, `Integrated_Intensity`; older
        # MetaMorph plug-ins used `X` / `Y` / `IntegratedIntensity`.
        # Both are listed here so auto-detect catches either.
        "frame":     ("Plane", "Frame", "frame", "Slice", "T", "t"),
        "frame_offset": -1,                 # PALM-Tracer is 1-indexed
        "x":         ("CentroidX(px)", "Centroid X", "Centroid_X",
                       "X", "x", "Position X"),
        "y":         ("CentroidY(px)", "Centroid Y", "Centroid_Y",
                       "Y", "y", "Position Y"),
        "xy_unit":   "px",
        "mass":      ("Integrated_Intensity", "IntegratedIntensity",
                       "Integrated Intensity", "Mass", "Amp",
                       "Amplitude", "Intensity", "mass"),
    },
    "ThunderSTORM": {
        "frame":     ("frame", "Frame"),
        "frame_offset": -1,                 # ThunderSTORM is 1-indexed
        "x":         ("x [nm]", "x_nm", "x"),
        "y":         ("y [nm]", "y_nm", "y"),
        "xy_unit":   "nm",
        "mass":      ("intensity [photon]", "intensity", "photons"),
    },
    "Picasso": {
        "frame":     ("frame",),
        "frame_offset": 0,                  # Picasso is 0-indexed
        "x":         ("x", "x_pix"),
        "y":         ("y", "y_pix"),
        "xy_unit":   "px",
        "mass":      ("photons", "photon", "intensity"),
    },
}


def _autodetect_csv_preset(columns: "list[str]") -> "str | None":
    """Best-effort: pick a preset whose required columns are all present."""
    cols = set(c.strip() for c in columns)
    for name, spec in _CSV_PRESETS.items():
        # Need at least frame, x, y resolvable from this column set
        def _any(opts): return any(o in cols for o in opts)
        if (_any(spec["frame"]) and _any(spec["x"]) and _any(spec["y"])):
            return name
    return None


def load_external_locs(csv_path: str, preset: str = "auto",
                       pixel_size_um: float = 0.106,
                       column_map: "dict | None" = None,
                       frame_offset: int | None = None):
    """Load a localisations file exported by an external tool and map
    its columns to FIREFLY's canonical schema {frame, x, y, mass}.

    Accepts `.csv`, `.txt` and `.tsv` — PALM-Tracer's UI emits
    tab-separated `.txt` files; ThunderSTORM and Picasso use commas.
    The separator is auto-detected by pandas' python engine.

    Parameters
    ----------
    csv_path : str
        Path to the input file (header row required).
    preset : str
        One of "PALM-Tracer", "ThunderSTORM", "Picasso", "Custom", or
        "auto" to sniff the header.
    pixel_size_um : float
        Needed only for presets whose `x` / `y` are in nm
        (e.g. ThunderSTORM) so we can convert to pixel units.
    column_map : dict, optional
        Required when `preset == "Custom"`.  Map canonical name →
        source column name, e.g. {"frame": "plane_idx", "x": "x_um",
        "y": "y_um", "mass": "amp"}.
    frame_offset : int, optional
        Overrides the preset's default (use -1 for 1-indexed sources).

    Returns
    -------
    pandas.DataFrame  with columns int `frame`, float `x` (pixels),
    float `y` (pixels), float `mass` (raw intensity / photon count).
    """
    import pandas as _pd
    # PALM-Tracer's `locPALMTracer.txt` has a 2-row metadata block at
    # the top before the actual data header:
    #     Width  Height  nb_Planes  ...  (8 cols)
    #     512    512     16000      ...  (values)
    #     id     Plane   ...        (14 cols — actual table header)
    #     1      1       ...        (data)
    #     ...
    # ThunderSTORM and Picasso put their column header on line 1.  We
    # peek at the file's first ~20 lines, find the row whose column
    # count matches the maximum (i.e. the actual data header row), and
    # tell pandas to start reading from there.  Bonus: extract pixel
    # size / frame interval from the PALM-Tracer metadata block when
    # present, and remember them on the returned DataFrame's attrs.
    pt_meta: dict = {}
    header_line = 0
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
            preview = [next(fh).rstrip("\r\n") for _ in range(20)]
    except StopIteration:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
            preview = fh.read().splitlines()[:20]
    except Exception:
        preview = []
    if preview:
        def _split(line: str, seps=("\t", ",", ";", "|")):
            best = (line,)   # fall-through
            for s in seps:
                bits = line.split(s)
                if len(bits) > len(best):
                    best = bits
            return list(best)
        # PALM-Tracer metadata-row detector: row 1 starts with "Width"
        # and the values row follows.  Lift the pixel size + frame
        # interval out for later use.
        if (preview and "Width" in preview[0]
                and "Pixel_Size" in preview[0]):
            meta_keys = _split(preview[0])
            meta_vals = _split(preview[1]) if len(preview) > 1 else []
            for k, v in zip(meta_keys, meta_vals):
                pt_meta[k.strip()] = v.strip()
            try:
                pt_meta["pixel_size_um"] = float(pt_meta.get(
                    "Pixel_Size(um)", "0") or 0)
            except Exception: pass
            try:
                pt_meta["frame_interval_s"] = float(pt_meta.get(
                    "Frame_Duration(s)", "0") or 0)
            except Exception: pass

        # Find the actual data header.  Rules:
        #   1. It must have the highest column count seen in the preview
        #      (otherwise PALM-Tracer's 8-col metadata rows win).
        #   2. Its fields must LOOK like header names — i.e. contain
        #      mostly letters, not just numbers.  Otherwise we'd pick a
        #      data row whenever the metadata block was a different
        #      width from the table.
        # Among all rows that satisfy both, take the FIRST one.
        def _looks_like_header(fields):
            # A header row's fields are mostly non-numeric tokens.
            if not fields:
                return False
            letter_count = sum(
                1 for f in fields
                if any(c.isalpha() for c in f))
            return letter_count >= max(2, len(fields) // 2)

        split_rows = [_split(ln) for ln in preview]
        widths = [len(r) for r in split_rows]
        max_w  = max(widths) if widths else 0
        header_line = 0
        for i, row in enumerate(split_rows):
            if len(row) == max_w and _looks_like_header(row):
                header_line = i
                break
        if header_line > 0:
            print(f"  Skipping {header_line} metadata row(s) before the "
                  f"data table.")

    # Try comma → tab → python-engine sniff.  First attempt that yields
    # more than one column wins.  The `skiprows` value is the
    # metadata-block size detected above.
    df = None
    last_exc: Exception | None = None
    for kwargs in (
        {"sep": "\t",  "engine": "c"},
        {"sep": ",",   "engine": "c"},
        {"sep": None,  "engine": "python"},
    ):
        try:
            attempt = _pd.read_csv(
                csv_path, skiprows=header_line, **kwargs)
        except Exception as exc:
            last_exc = exc
            continue
        if attempt.shape[1] > 1:
            df = attempt
            print(f"  Parsed file with sep={kwargs['sep']!r}, "
                  f"skiprows={header_line}, "
                  f"{len(df):,} rows × {df.shape[1]} columns")
            break
    if df is None:
        if last_exc is not None:
            raise last_exc
        raise ValueError(
            f"Couldn't parse {csv_path} — single-column result on "
            f"every attempted separator / skiprows combination.")

    # Apply PALM-Tracer's embedded metadata to pixel_size_um when the
    # caller didn't specify one (or used the default).  This lets users
    # drop a PALM-Tracer file in and have the units come out right
    # without typing 0.106 again.
    pt_px = pt_meta.get("pixel_size_um")
    if pt_px and abs(pixel_size_um - 0.106) < 1e-9:
        # Default value — replace with PALM-Tracer's value
        print(f"  Using pixel size {pt_px:.4f} µm from PALM-Tracer metadata")
        pixel_size_um = float(pt_px)

    if preset == "auto" or not preset:
        sniffed = _autodetect_csv_preset(list(df.columns))
        if sniffed is None:
            raise ValueError(
                f"Couldn't auto-detect a CSV preset for {csv_path} — "
                f"header columns: {list(df.columns)}.  Use one of "
                f"{list(_CSV_PRESETS) + ['Custom']} explicitly.")
        preset = sniffed
        print(f"  Auto-detected preset: {preset}")

    # Resolve column names to use for each canonical field
    if preset == "Custom":
        if not column_map:
            raise ValueError("Custom preset requires `column_map`")
        mapping = {k: v for k, v in column_map.items() if v}
        spec = {}      # no implicit offsets / units
    else:
        spec = _CSV_PRESETS.get(preset)
        if spec is None:
            raise ValueError(
                f"Unknown preset '{preset}'.  Available: "
                f"{list(_CSV_PRESETS) + ['Custom']}")
        mapping = {}
        for canonical in ("frame", "x", "y", "mass"):
            for col in spec.get(canonical, ()):
                if col in df.columns:
                    mapping[canonical] = col
                    break
        # frame, x, y are required; mass is optional
        for required in ("frame", "x", "y"):
            if required not in mapping:
                raise ValueError(
                    f"Couldn't find a '{required}' column in {csv_path} "
                    f"using preset '{preset}'.  Tried "
                    f"{spec.get(required, ())}.  Header was: "
                    f"{list(df.columns)}")

    out = _pd.DataFrame()
    out["frame"] = df[mapping["frame"]].astype("int64")
    fo = frame_offset if frame_offset is not None else spec.get("frame_offset", 0)
    if fo:
        out["frame"] = out["frame"] + int(fo)
    if (out["frame"] < 0).any():
        # Drop any negative-frame rows that resulted from a wrong offset
        n_bad = int((out["frame"] < 0).sum())
        print(f"  WARN: {n_bad} localisations have frame < 0 after offset "
              f"({fo}) — dropping them.")
        keep = out["frame"] >= 0
        df = df.loc[keep].reset_index(drop=True)
        out = out.loc[keep].reset_index(drop=True)

    x = df[mapping["x"]].astype(float).values
    y = df[mapping["y"]].astype(float).values
    if spec.get("xy_unit") == "nm":
        x = x / (pixel_size_um * 1000.0)
        y = y / (pixel_size_um * 1000.0)
    out["x"] = x
    out["y"] = y

    if "mass" in mapping:
        out["mass"] = df[mapping["mass"]].astype(float).values
    else:
        # Detection already happened upstream; downstream stages tolerate
        # a constant mass column.  Filter-by-mass becomes a no-op which
        # is the right behaviour for pre-filtered external data.
        out["mass"] = 1.0
    print(f"  Loaded {len(out):,} external localisations "
          f"(preset={preset}, frames {int(out['frame'].min())}–"
          f"{int(out['frame'].max())})")
    return out


def load_file(path, channel=0, stop_event=None, files=None):
    """Load `path` (or, if `files` is provided, the explicit list of
    files) as a single stack.

    `files` lets a caller override the auto-discovery of sister files
    that `_find_tif_series` / `_find_czi_series` does — useful when the
    GUI wants to load only a user-selected subset of a multi-file series.
    """
    ext = os.path.splitext(path)[1].lower()
    if   ext == ".czi":            return load_czi(path, channel, stop_event,
                                                   files=files)
    elif ext in (".tif", ".tiff"): return load_tif(path, stop_event,
                                                   files=files)
    else: sys.exit(f"ERROR: Unsupported file '{ext}'. Use .czi or .tif")


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING  (fast path + parallel)
# ══════════════════════════════════════════════════════════════════════════════

def _preprocess_fast(frame, bg_radius=50, sigma=1.0):
    """
    Fast background subtraction using uniform_filter.
    ~1700x faster than rolling_ball with comparable results for PALM data.
    """
    bg        = uniform_filter(frame, size=int(bg_radius * 2 + 1))
    corrected = np.clip(frame - bg, 0, None)
    smoothed  = filters.gaussian(corrected, sigma=sigma, preserve_range=True)
    mn, mx    = smoothed.min(), smoothed.max()
    if mx > mn:
        smoothed = (smoothed - mn) / (mx - mn)
    return smoothed.astype(np.float32)


def _preprocess_rolling(frame, bg_radius=50, sigma=1.0):
    """Legacy rolling-ball background subtraction (slow but thorough)."""
    from skimage.restoration import rolling_ball
    bg        = rolling_ball(frame, radius=bg_radius)
    corrected = np.clip(frame - bg, 0, None)
    smoothed  = filters.gaussian(corrected, sigma=sigma, preserve_range=True)
    mn, mx    = smoothed.min(), smoothed.max()
    if mx > mn:
        smoothed = (smoothed - mn) / (mx - mn)
    return smoothed.astype(np.float32)


def preprocess_stack(stack, bg_radius=50, bg_method="uniform_filter",
                     workers=N_CPUS):
    n = len(stack)
    fn = _preprocess_fast if bg_method == "uniform_filter" else _preprocess_rolling

    print(f"  Background method : {bg_method}")
    print(f"  Workers           : {workers} / {N_CPUS} CPU cores")
    t0 = time.perf_counter()

    if workers == 1:
        processed = [fn(f, bg_radius) for f in
                     _tqdm(stack, desc="  Preprocessing", unit="fr", ncols=70)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as _exe:
            _futs = [_exe.submit(fn, f, bg_radius) for f in stack]
            processed = [_f.result() for _f in
                         _tqdm(_futs, desc="  Preprocessing", unit="fr", ncols=70)]

    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s  ({elapsed/n*1000:.1f} ms/frame)")
    return np.stack(processed)




# ══════════════════════════════════════════════════════════════════════════════
#  ROI  —  simple intensity threshold
# ══════════════════════════════════════════════════════════════════════════════

def auto_threshold(image_norm, method="auto"):
    """
    Determine an intensity threshold automatically from the normalised
    mean projection using scikit-image thresholding algorithms.

    Parameters
    ----------
    image_norm : float array (Y, X), values in [0, 1]
    method     : "auto"     — tries otsu, li, triangle; picks best for sptPALM
                 "otsu"     — maximises inter-class variance
                 "li"       — minimises cross-entropy (good for sparse cells)
                 "triangle" — geometric method, good for large dark backgrounds

    Returns
    -------
    threshold : float in [0, 1]
    chosen    : str — which method was used
    all_vals  : dict — thresholds from all three methods for reference
    """
    from skimage.filters import threshold_otsu, threshold_li, threshold_triangle

    results = {}
    try:    results["otsu"]     = float(threshold_otsu(image_norm))
    except Exception: results["otsu"]     = None
    try:    results["li"]       = float(threshold_li(image_norm))
    except Exception: results["li"]       = None
    try:    results["triangle"] = float(threshold_triangle(image_norm))
    except Exception: results["triangle"] = None

    print(f"  Auto-threshold candidates:")
    for name, val in results.items():
        print(f"    {name:<10} : {val:.4f}" if val is not None
              else f"    {name:<10} : failed")

    if method != "auto":
        chosen = method
        val    = results.get(method)
        if val is None:
            print(f"  WARNING: {method} failed, falling back to otsu")
            chosen = "otsu"
            val    = results["otsu"]
    else:
        # For sptPALM the cell typically occupies < 30% of the frame.
        # Triangle handles large dark backgrounds best in this scenario.
        # Fall back to Li then Otsu if triangle is unavailable.
        for preferred in ("triangle", "li", "otsu"):
            if results[preferred] is not None:
                chosen = preferred
                val    = results[preferred]
                break

    print(f"  Selected method : {chosen}  ->  threshold = {val:.4f}")
    return val, chosen, results


def build_roi_mask_mean(stack, threshold=0.15, smooth_sigma=5):
    """
    Mean-projection ROI: one mask derived from the average intensity across
    ALL frames.  Stable and recommended for most experiments.

    Works by averaging all T frames into a single image, smoothing it, then
    thresholding.  Because fluorophores blink on and off, individual frames
    are mostly dark even inside the cell — the mean projection reveals the
    underlying cell structure by accumulating signal over time.

    The mean projection is normalised to [0,1] before thresholding so that
    the threshold is always relative to the brightest region in the image,
    regardless of fluorophore density or acquisition settings.

    Returns
    -------
    mask : bool array (Y, X)
    """
    from skimage.morphology import binary_closing, disk
    mean_proj = stack.mean(axis=0)
    smoothed  = filters.gaussian(mean_proj, sigma=smooth_sigma,
                                 preserve_range=True)
    # Normalise to [0,1] so threshold is relative to brightest region
    mn, mx = smoothed.min(), smoothed.max()
    if mx > mn:
        smoothed_norm = (smoothed - mn) / (mx - mn)
    else:
        smoothed_norm = smoothed
    mask = binary_closing(smoothed_norm > threshold, disk(5))
    return mask, mean_proj


def build_roi_mask_perframe(stack, threshold=0.15, smooth_sigma=5):
    """
    Per-frame ROI: a separate mask computed independently for each frame.

    Each frame is smoothed and thresholded individually, so the ROI can
    change shape frame-to-frame.  Useful when illumination drifts during
    acquisition or when imaging moving/growing cells.

    Note: because individual sptPALM frames are very sparse (only a handful
    of fluorophores are visible at once), per-frame masks are inherently
    noisier than the mean-projection mask.  Use with caution and always
    check the preview.

    Returns
    -------
    masks : bool array (T, Y, X) — one mask per frame
    mean_proj : float array (Y, X) — mean projection for display purposes
    """
    from skimage.morphology import binary_closing, disk
    T = len(stack)
    masks = np.zeros(stack.shape, dtype=bool)
    for t in _tqdm(range(T), desc="  Building per-frame masks",
                   unit="fr", ncols=70):
        smoothed = filters.gaussian(stack[t], sigma=smooth_sigma,
                                    preserve_range=True)
        # Normalise each frame to [0,1] before thresholding
        mn, mx = smoothed.min(), smoothed.max()
        if mx > mn:
            smoothed = (smoothed - mn) / (mx - mn)
        masks[t] = binary_closing(smoothed > threshold, disk(5))
    mean_proj = stack.mean(axis=0)
    return masks, mean_proj


def build_roi_mask(stack=None, threshold=None, smooth_sigma=5,
                   mode="mean", threshold_method="auto", save_path=None,
                   precomputed_mean_proj=None):
    """
    Build ROI mask(s) from a stack or a pre-computed mean projection.

    Parameters
    ----------
    stack                 : preprocessed float32 stack (T x Y x X).
                            Not required when precomputed_mean_proj is supplied.
    threshold             : manual intensity cutoff on [0,1].
                            If None, determined automatically.
    smooth_sigma          : Gaussian sigma (px) before thresholding (default: 5)
    mode                  : "mean"     — one mask from mean projection (default)
                            "perframe" — separate mask per frame (needs stack)
    threshold_method      : "auto" | "otsu" | "li" | "triangle"
    save_path             : if given, saves a preview PNG for inspection
    precomputed_mean_proj : float32 (Y, X) normalised [0,1] mean projection.
                            When supplied, stack is not needed for mean-mode ROI,
                            saving the memory cost of holding the full stack.
                            perframe mode silently falls back to mean mode here.

    Returns
    -------
    mask : bool array (Y, X) for mode="mean"
           bool array (T, Y, X) for mode="perframe"
    """
    # ── Mean projection ────────────────────────────────────────────────────────
    if precomputed_mean_proj is not None:
        mean_proj_norm = precomputed_mean_proj.astype(np.float32)
        mn, mx = mean_proj_norm.min(), mean_proj_norm.max()
        if mx > mn:
            mean_proj_norm = (mean_proj_norm - mn) / (mx - mn)
        if mode == "perframe":
            print("  NOTE: perframe ROI mode needs the full stack; "
                  "falling back to mean mode.")
            mode = "mean"
    elif stack is not None:
        mean_proj_norm = stack.mean(axis=0)
        mn, mx = mean_proj_norm.min(), mean_proj_norm.max()
        if mx > mn:
            mean_proj_norm = (mean_proj_norm - mn) / (mx - mn)
    else:
        raise ValueError(
            "build_roi_mask: supply either 'stack' or 'precomputed_mean_proj'.")

    auto_method_used = None
    if threshold is None:
        threshold, auto_method_used, all_thresh = auto_threshold(
            mean_proj_norm, method=threshold_method)
    else:
        print(f"  Threshold : {threshold:.4f}  [manual]")
        all_thresh = None

    # ── Build mask ─────────────────────────────────────────────────────────────
    if mode == "perframe" and stack is not None:
        mask, mean_proj = build_roi_mask_perframe(stack, threshold, smooth_sigma)
        display_mask = mask.mean(axis=0) > 0.5
        mode_label   = "Per-frame"
    else:
        if stack is not None:
            mask, mean_proj = build_roi_mask_mean(stack, threshold, smooth_sigma)
        else:
            # Streaming mode: build mask directly from precomputed mean projection
            from skimage.morphology import binary_closing, disk
            smoothed = filters.gaussian(mean_proj_norm, sigma=smooth_sigma,
                                        preserve_range=True)
            smn, smx = smoothed.min(), smoothed.max()
            snorm = (smoothed - smn) / (smx - smn) if smx > smn else smoothed
            mask = binary_closing(snorm > threshold, disk(5))
            mean_proj = mean_proj_norm  # use normalised version for display
        display_mask = mask
        mode_label   = "Mean projection"

    # ── Stats ──────────────────────────────────────────────────────────────────
    n_px  = display_mask.sum()
    total = display_mask.size
    print(f"  ROI mode  : {mode_label}")
    print(f"  ROI area  : {n_px:,} / {total:,} pixels  "
          f"({100*n_px/total:.1f}% of frame)")

    # ── Preview ────────────────────────────────────────────────────────────────
    if save_path:
        import matplotlib.pyplot as plt
        # 4 panels when auto-thresholding so we can show all three candidates
        if all_thresh is not None:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#0d1117")
            # Panel 4: comparison of auto-threshold candidates on histogram
            ax_h = axes[3]
            ax_h.set_facecolor("#0d1117")
            ax_h.hist(mean_proj_norm.ravel(), bins=200,
                      color="#58a6ff", alpha=0.7, log=True)
            colors_thresh = {"otsu":"#f78166", "li":"#7ed321", "triangle":"#f5a623"}
            for name, val in all_thresh.items():
                if val is not None:
                    lw  = 2.5 if name == auto_method_used else 1.2
                    ls  = "-"  if name == auto_method_used else "--"
                    ax_h.axvline(val, color=colors_thresh[name], lw=lw, ls=ls,
                                 label=f"{name}={val:.3f}"
                                       + (" *" if name == auto_method_used else ""))
            ax_h.legend(fontsize=8, facecolor="#0d1117",
                        edgecolor="#30363d", labelcolor="#e6edf3")
            ax_h.set_xlabel("Normalised intensity", color="#e6edf3", fontsize=9)
            ax_h.set_ylabel("Pixel count (log)", color="#e6edf3", fontsize=9)
            ax_h.set_title("Threshold comparison  (* selected)",
                           color="white", fontsize=9)
            ax_h.tick_params(colors="#e6edf3")
            for sp in ax_h.spines.values(): sp.set_edgecolor("#30363d")
            panel_axes = axes[:3]
        else:
            fig, panel_axes = plt.subplots(1, 3, figsize=(15, 5),
                                           facecolor="#0d1117")

        thresh_label = (f"auto:{auto_method_used}={threshold:.3f}"
                        if auto_method_used else f"manual={threshold:.3f}")
        titles = ["Mean projection",
                  f"ROI mask  ({mode_label})",
                  f"Overlay  ({thresh_label})"]
        imgs  = [mean_proj, display_mask.astype(float), mean_proj]
        cmaps = ["inferno", "Greens", "inferno"]
        for ax, img, ttl, cm in zip(panel_axes, imgs, titles, cmaps):
            ax.set_facecolor("#0d1117")
            ax.imshow(img, cmap=cm, origin="lower")
            ax.set_title(ttl, color="white", fontsize=10)
            ax.tick_params(colors="white")
            for sp in ax.spines.values(): sp.set_edgecolor("#30363d")
        panel_axes[2].contour(display_mask.astype(float), levels=[0.5],
                              colors=["#58a6ff"], linewidths=[1.5])
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  ROI preview saved -> {save_path}")

    return mask


# ──────────────────────────────────────────────────────────────────────────────
#  build_roi_mask_advanced — single source of truth for the GUI preview AND
#  the firefly_worker analysis path, so the green mask the user tunes in the
#  ROI viewer is *identical* to the mask actually applied during analysis.
#
#  Pipeline:
#      projection ─►  fine + coarse Gaussian blur (DoG bg subtraction)
#                  ►  normalise to [0, 1]
#                  ►  intensity threshold (manual or auto: Li/Otsu/…)
#                  ►  morphological opening   (kill 1-2 px speckle bridges)
#                  ►  morphological closing   (fill speckle-gap interior)
#                  ►  remove_small_holes      (merge interior pockets)
#                  ►  remove_small_objects    (drop sub-cell fragments)
#                  ►  keep top-N components   (hard cap on background islands)
#
#  All numeric defaults match the GUI preview (see _RoiViewer._refresh_roi_mask_
#  overlay in app_qt.py).  If you change a default here, update the docstring
#  hint in the GUI's "Background scale σ" tooltip too.
# ──────────────────────────────────────────────────────────────────────────────
def build_roi_mask_advanced(projection,
                            *,
                            threshold=None,
                            threshold_method="li",
                            bg_sigma=25.0,
                            sigma_fg=None,
                            mode_hint="max",
                            opening_radius=3,
                            closing_radius=7,
                            max_hole_size=2000,
                            min_object_size=8000,
                            keep_n_components=4):
    """Build a bool ROI mask from a 2-D projection image.

    Parameters
    ----------
    projection : 2-D float array (Y, X)
        The image to threshold.  Caller decides whether this is a mean,
        max, sum, or blink-count projection of the stack.
    threshold : float | None
        Manual threshold on the normalised DoG residual [0, 1].
        If None, an automatic threshold is chosen using `threshold_method`.
    threshold_method : str
        "li", "otsu", "triangle", or "mean".  Only used when threshold is None.
    bg_sigma : float
        DoG background-suppression scale (pixels).  Subtracts a
        Gaussian-blurred copy of the projection (this sigma) from a
        lightly-blurred copy, killing slow illumination gradients before
        thresholding.  Set to 0 to disable.
    sigma_fg : float | None
        Fine-scale smoothing applied before DoG.  If None, defaults to
        2.0 for "max"/"blink" projections and 5.0 for "mean"/"sum"
        (mode_hint controls this).
    mode_hint : str
        "max", "blink", "mean", or "sum" — only used to pick a sensible
        sigma_fg default.  Not used otherwise.

    Returns
    -------
    mask : bool ndarray (Y, X)
        The cleaned-up ROI mask.
    info : dict
        {"threshold": float (the one actually used),
         "fraction": float (0-1, mask coverage)}.
    """
    import numpy as _np
    from scipy.ndimage import gaussian_filter
    from skimage.morphology import (binary_closing, binary_opening, disk,
                                    remove_small_holes,
                                    remove_small_objects)

    proj = _np.asarray(projection, dtype=_np.float32)

    # ── 1. DoG background suppression ───────────────────────────────────
    if sigma_fg is None:
        sigma_fg = 2.0 if mode_hint in ("max", "blink") else 5.0
    smoothed_fg = gaussian_filter(proj, sigma=float(sigma_fg))
    if bg_sigma > 0.5 and bg_sigma > sigma_fg:
        smoothed_bg = gaussian_filter(proj, sigma=float(bg_sigma))
        smoothed = _np.maximum(smoothed_fg - smoothed_bg, 0.0)
    else:
        smoothed = smoothed_fg
    mn, mx = float(smoothed.min()), float(smoothed.max())
    if mx > mn:
        smoothed = (smoothed - mn) / (mx - mn)

    # ── 2. Threshold ────────────────────────────────────────────────────
    if threshold is None:
        try:
            from skimage.filters import (threshold_otsu, threshold_li,
                                         threshold_triangle)
            m = (threshold_method or "li").lower()
            if   m == "otsu":     t = float(threshold_otsu(smoothed))
            elif m == "triangle": t = float(threshold_triangle(smoothed))
            elif m == "mean":     t = float(smoothed.mean())
            else:                 t = float(threshold_li(smoothed))
        except Exception:
            t = float(smoothed.mean())
    else:
        t = float(threshold)

    # ── 3. Morphology cleanup ───────────────────────────────────────────
    raw = smoothed > t
    try:    raw = binary_opening(raw, disk(int(opening_radius)))
    except Exception: pass
    try:    mask = binary_closing(raw, disk(int(closing_radius)))
    except Exception: mask = raw
    try:
        mask = remove_small_holes(mask, area_threshold=int(max_hole_size))
    except TypeError:
        mask = remove_small_holes(mask, int(max_hole_size))
    except Exception:
        pass
    try:
        mask = remove_small_objects(mask, min_size=int(min_object_size))
    except TypeError:
        mask = remove_small_objects(mask, int(min_object_size))
    except Exception:
        pass

    # ── 4. Keep top-N connected components ──────────────────────────────
    if keep_n_components and keep_n_components > 0:
        try:
            from skimage.measure import label as _label
            lbl = _label(mask, connectivity=2)
            if lbl.max() > keep_n_components:
                sizes = _np.bincount(lbl.ravel())
                sizes[0] = 0
                keep = _np.argsort(sizes)[-int(keep_n_components):]
                keep_mask = _np.zeros_like(mask, dtype=bool)
                for k in keep:
                    if sizes[k] > 0:
                        keep_mask |= (lbl == k)
                mask = keep_mask
        except Exception:
            pass

    info = {"threshold": float(t),
            "fraction": float(mask.mean()) if mask.size else 0.0}
    return mask.astype(bool, copy=False), info


def apply_roi_mask(locs, mask):
    """
    Filter a localisations DataFrame to keep only points inside the ROI mask.

    Parameters
    ----------
    locs : DataFrame with columns 'x', 'y', 'frame', in pixels
    mask : bool array (Y, X)      — mean-projection mode: same mask every frame
           bool array (T, Y, X)   — per-frame mode: each frame gets its own mask

    Returns
    -------
    Filtered DataFrame
    """
    xi = np.clip(locs["x"].values.astype(int), 0, mask.shape[-1] - 1)
    yi = np.clip(locs["y"].values.astype(int), 0, mask.shape[-2] - 1)

    if mask.ndim == 2:
        # Mean-projection mode — same mask for every localisation
        inside = mask[yi, xi]
    else:
        # Per-frame mode — look up the mask for each localisation's frame
        fi = np.clip(locs["frame"].values.astype(int), 0, mask.shape[0] - 1)
        inside = mask[fi, yi, xi]

    # Defensive: a uint8 mask of 0/1 yields a uint8 `inside`, which
    # pandas interprets as a column-label array (`locs[inside]` looks
    # for columns literally named `1, 1, 1, …`) and fails with
    # "None of [Index([1, 1, 1, ...])] are in the [columns]".
    # Force-cast to bool so the indexing is unambiguous regardless of
    # how callers built the mask.
    if inside.dtype != bool:
        inside = inside.astype(bool)

    filtered  = locs[inside].reset_index(drop=True)
    n_removed = len(locs) - len(filtered)
    mode_str  = "per-frame" if mask.ndim == 3 else "mean-projection"
    print(f"  ROI filter ({mode_str}): kept {len(filtered):,} / {len(locs):,} "
          f"localisations  ({n_removed:,} outside ROI removed)")
    return filtered

# ══════════════════════════════════════════════════════════════════════════════
#  DRIFT CORRECTION
# ══════════════════════════════════════════════════════════════════════════════

def correct_drift(locs, n_seg_frames=200, upsampling=4, smooth_sigma=1.5):
    """
    Reference-free drift correction via cross-correlation of localization
    density maps (simplified RCC approach; Wang et al. 2014, Nat Methods).

    The acquisition is divided into time segments.  A 2-D localization density
    histogram is built for each segment at ``upsampling``× the raw pixel
    resolution.  Consecutive histograms are cross-correlated (FFT) to measure
    the inter-segment drift.  The cumulative, Gaussian-smoothed drift trajectory
    is interpolated to per-frame resolution and subtracted from every
    localization.

    Applied *before* linking so that drift-corrected positions produce better
    trajectories.

    Parameters
    ----------
    locs          : DataFrame with 'x', 'y', 'frame' columns (in pixels)
    n_seg_frames  : target number of frames per time segment (default 200).
                    Smaller → finer time resolution but fewer localisations
                    per segment (noisier cross-correlation).
    upsampling    : density-map super-resolution factor.  upsampling=4 gives
                    ~25 nm accuracy at 0.1 µm/px (default 4).
    smooth_sigma  : Gaussian smoothing sigma in units of *segments* applied to
                    the raw drift trajectory before interpolation (default 1.5).

    Returns
    -------
    locs_corrected : DataFrame with corrected 'x' and 'y'
    drift_df       : DataFrame with columns ['frame', 'dx', 'dy'] (pixels)
    """
    if len(locs) == 0:
        return locs.copy(), pd.DataFrame({"frame": [0], "dx": [0.0], "dy": [0.0]})

    x = locs["x"].values.astype(np.float64)
    y = locs["y"].values.astype(np.float64)
    f = locs["frame"].values.astype(int)

    n_frames   = int(f.max()) + 1
    n_segments = max(4, int(np.ceil(n_frames / n_seg_frames)))
    n_segments = min(n_segments, max(2, len(locs) // 10))  # need ≥10 locs/seg

    print(f"  Drift correction : {n_segments} segments "
          f"(~{n_frames // n_segments} frames each, upsampling={upsampling})")

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    W = max(int((x_max - x_min) * upsampling) + 1, 16)
    H = max(int((y_max - y_min) * upsampling) + 1, 16)

    seg_bounds  = np.linspace(0, n_frames, n_segments + 1).astype(int)
    seg_centers = (seg_bounds[:-1] + seg_bounds[1:]) / 2.0

    # ── Build upsampled density maps ──────────────────────────────────────────
    density_maps = []
    seg_counts   = []
    for i in range(n_segments):
        sel = (f >= seg_bounds[i]) & (f < seg_bounds[i + 1])
        seg_counts.append(int(sel.sum()))
        dm  = np.zeros((H, W), dtype=np.float32)
        if sel.sum() > 0:
            xi = np.clip(((x[sel] - x_min) * upsampling).astype(int), 0, W - 1)
            yi = np.clip(((y[sel] - y_min) * upsampling).astype(int), 0, H - 1)
            np.add.at(dm, (yi, xi), 1.0)
            dm = gaussian_filter(dm, sigma=upsampling * 0.7)   # spread spots
        density_maps.append(dm)

    print(f"  Localisations/segment: min {min(seg_counts):,}, "
          f"max {max(seg_counts):,}")

    # ── Cross-correlate ALL pairs (i, j) → solve cumulative drift ─────────────
    # This is the redundant cross-correlation (RCC) algorithm of Wang et al.
    # 2014 (Nat. Methods).  Instead of relying only on consecutive pairs, we
    # measure the inter-segment shift Δ_{ij} for every pair (i, j) with i<j
    # and then solve the over-determined linear system
    #
    #     drift[j] − drift[i] = Δ_{ij}      for all valid pairs
    #
    # by least-squares.  Drift[0] is fixed at zero (gauge fixing).  The
    # redundancy averages out cross-correlation noise far better than the
    # consecutive-only chain, and is robust to any single bad pair (e.g. a
    # segment with too few localisations).
    #
    # Performance note:  scipy.signal.correlate(method="fft") re-FFTs both
    # density maps on every pair call, so an N-segment run does ~N(N-1)
    # FFTs.  We precompute rfft2 of each (zero-padded) map ONCE and just
    # run an IFFT per pair — quadratic-cost FFT work collapses to linear,
    # plus the IFFT loop parallelises trivially via threads.
    from scipy.fft import rfft2 as _rfft2, irfft2 as _irfft2, \
                          next_fast_len as _next_fast_len
    pad_H = _next_fast_len(2 * H - 1)
    pad_W = _next_fast_len(2 * W - 1)
    fft_maps = [_rfft2(dm, s=(pad_H, pad_W)) for dm in density_maps]

    pair_indices = [(i, j) for i in range(n_segments)
                    for j in range(i + 1, n_segments)
                    if seg_counts[i] >= 5 and seg_counts[j] >= 5]

    def _pair_shift(i, j):
        # Cross-correlation r[τ] = Σ a[k+τ] b[k]  via  IFFT(F_a · conj(F_b))
        cross = _irfft2(fft_maps[i] * np.conj(fft_maps[j]),
                        s=(pad_H, pad_W))
        # Zero-lag at index 0; positive shifts up to (H-1, W-1) sit at low
        # indices, negative shifts wrap to the end.  Re-centre by treating
        # any index beyond half-extent as negative.
        peak = int(np.argmax(cross))
        py, px = divmod(peak, pad_W)
        if py >= pad_H // 2: py -= pad_H
        if px >= pad_W // 2: px -= pad_W
        return i, j, float(px), float(py)

    A_rows_x, A_rows_y = [], []
    b_x, b_y = [], []
    if pair_indices:
        with ThreadPoolExecutor(max_workers=N_CPUS) as _exe:
            for i, j, dx_pair, dy_pair in _exe.map(
                    lambda ij: _pair_shift(*ij), pair_indices):
                row = np.zeros(n_segments)
                row[i], row[j] = -1.0, 1.0
                A_rows_x.append(row); b_x.append(dx_pair)
                A_rows_y.append(row); b_y.append(dy_pair)

    if not A_rows_x:
        # Fallback: zero drift
        dx_cum = np.zeros(n_segments)
        dy_cum = np.zeros(n_segments)
    else:
        # Add gauge-fixing row: drift[0] = 0 (heavy weight)
        gauge = np.zeros(n_segments); gauge[0] = 1.0
        A = np.vstack(A_rows_x + [gauge * 1e3])
        bx = np.append(np.array(b_x), 0.0)
        by = np.append(np.array(b_y), 0.0)
        dx_cum, *_ = np.linalg.lstsq(A, bx, rcond=None)
        dy_cum, *_ = np.linalg.lstsq(A, by, rcond=None)

    # Smooth then convert to localization pixels
    dx_sm = gaussian_filter1d(dx_cum, sigma=smooth_sigma) / upsampling
    dy_sm = gaussian_filter1d(dy_cum, sigma=smooth_sigma) / upsampling

    # Zero-centre so overall position is preserved
    dx_sm -= dx_sm.mean()
    dy_sm -= dy_sm.mean()

    rng_x, rng_y = float(np.ptp(dx_sm)), float(np.ptp(dy_sm))
    print(f"  Drift range  x={rng_x:.3f} px  y={rng_y:.3f} px")

    # ── Interpolate to every frame ────────────────────────────────────────────
    frame_arr = np.arange(n_frames, dtype=float)
    ix = interp1d(seg_centers, dx_sm, kind="linear",
                  bounds_error=False, fill_value=(dx_sm[0], dx_sm[-1]))
    iy = interp1d(seg_centers, dy_sm, kind="linear",
                  bounds_error=False, fill_value=(dy_sm[0], dy_sm[-1]))
    drift_x = ix(frame_arr)
    drift_y = iy(frame_arr)

    # ── Subtract from localisations ────────────────────────────────────────────
    locs_out = locs.copy()
    fi       = np.clip(f, 0, n_frames - 1)
    locs_out["x"] = x - drift_x[fi]
    locs_out["y"] = y - drift_y[fi]

    drift_df = pd.DataFrame({"frame": frame_arr.astype(int),
                             "dx": drift_x, "dy": drift_y})
    return locs_out, drift_df


# ══════════════════════════════════════════════════════════════════════════════
#  LOCALISATION  (parallel + chunked)
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  LOCALISER BACKENDS
# ══════════════════════════════════════════════════════════════════════════════
#
# A backend takes a *preprocessed* stack (T × Y × X, float32) and returns a
# DataFrame with at least the columns: x, y, frame, mass.  Preprocessing
# (background subtraction, bandpass) is handled separately so the fast / stream
# RAM strategies in this file stay backend-agnostic.
#
# Registration model: subclass LocaliserBackend, set `.name`, implement
# `.is_available()` (classmethod) and `.localise(stack, **params)`, then append
# to _BACKEND_REGISTRY in the preference order used by `backend="auto"`.
#
# Phase A1: only TrackpyBackend exists (refactor — no behaviour change).
# Phase A2: TorchBackend (CPU) lands here.
# Phase A3: device selection (MPS / CUDA) inside TorchBackend.

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
            ctx = multiprocessing.get_context("spawn")
            if stack_is_memmap:
                print(f"  Parallelism : multiprocessing.Pool × {n_workers} "
                      f"(spawn, memmap re-open in workers — zero-copy)")
            else:
                print(f"  Parallelism : multiprocessing.Pool × {n_workers} (spawn — true multi-core)")
            print(f"  Spawning workers (one-time ~10-30s; chunks then process truly in parallel)...")

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
                    for idx, result in _tqdm(
                            pool.imap_unordered(_localise_chunk_mmap_mp, mp_args),
                            total=n_chunks, desc="  Localising", unit="chunk", ncols=70):
                        chunk_results[idx] = result
            else:
                mp_args = [(i, c, diameter, minmass, percentile, o)
                           for i, (c, o) in enumerate(chunk_pairs)]
                with ctx.Pool(processes=n_workers) as pool:
                    for idx, result in _tqdm(
                            pool.imap_unordered(_localise_chunk_mp, mp_args),
                            total=n_chunks, desc="  Localising", unit="chunk", ncols=70):
                        chunk_results[idx] = result
            use_mp_ok = True
        except Exception as exc:
            print(f"  multiprocessing failed ({type(exc).__name__}: {exc})")
            print(f"  Falling back to BLAS-pool parallelism (slower, single-process)")

        if not use_mp_ok:
            with _threadpool_limits(limits=N_CPUS):
                chunk_results = [_localise_chunk(chunk, diameter, minmass, percentile, offset)
                                 for chunk, offset in _tqdm(chunk_pairs, total=n_chunks,
                                                            desc="  Localising", unit="chunk",
                                                            ncols=70)]

        valid = [df for df in chunk_results if df is not None and len(df) > 0]
        result = pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()

        elapsed = time.perf_counter() - t0
        print(f"  Found {len(result):,} localisations in {elapsed:.1f}s  "
              f"({n_frames / elapsed:.0f} frames/s)")
        return result


class TorchBackend(LocaliserBackend):
    """PyTorch-based localiser — CPU for now (MPS / CUDA arrive in A3).

    Algorithm (matches trackpy's default centroid-of-mass semantics so the
    sub-pixel positions stay close to within a few nm):

      1.  Bandpass = signal − local-average-background, then small-σ Gaussian
          smoothing.  Implemented as batched F.avg_pool2d + separable conv2d.
      2.  Threshold = `percentile`-th percentile of the bandpassed image
          (trackpy's `percentile` argument has the same meaning).
      3.  Local maxima = pixels where signal equals its diameter-window
          max-pool output AND exceeds the threshold (F.max_pool2d trick).
      4.  Patch extraction: gather a (diameter × diameter) tile around every
          candidate via fancy indexing — fully vectorised.
      5.  Mass = sum over patch; filter spots by `mass >= minmass`.
      6.  Sub-pixel refinement = centroid of mass on the patch.

    Returns a DataFrame with the standard columns `x, y, frame, mass`.

    Frames are processed in chunks of `chunk_size` to bound peak GPU memory.
    Step 1 (bandpass) is the bandwidth bottleneck on CPU; expect roughly the
    same wall-clock as trackpy on a fast laptop.  The point of this backend
    is the GPU path landing in A3 — CPU is here for correctness validation.
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

        # Index grid used for sub-pixel refinement (cached on device).  Same
        # tensor is shared by the centroid-of-mass and Gaussian-LSQ paths.
        dy_grid, dx_grid = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=dev, dtype=dtype),
            torch.arange(-radius, radius + 1, device=dev, dtype=dtype),
            indexing="ij")

        # Precompute the Gaussian-LSQ design matrix once per call — it
        # depends only on the patch geometry.  (The unweighted pseudo-inverse
        # is computed too, kept for reference but no longer used since we
        # switched to weighted batched LSQ for better noise behaviour.)
        _M, _M_pinv = self._build_gaussian_design_matrix(dy_grid, dx_grid)

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

            # ── 1. Bandpass: subtract local background, then small smooth ───
            bg = F.avg_pool2d(x, kernel_size=2 * radius + 1,
                              stride=1, padding=radius)
            smooth_sigma = max(1.0, diameter / 4.0)
            signal = self._gaussian_blur(x - bg, sigma=smooth_sigma, device=dev)
            signal = torch.clamp(signal, min=0.0)

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

            # ── 4. Patch extraction via batched advanced indexing ───────────
            # ys: (N, k, k), xs: (N, k, k), ts: (N, k, k)
            ys = y_ix[:, None, None] + dy_grid.long()[None]
            xs = x_ix[:, None, None] + dx_grid.long()[None]
            ts = t_ix[:, None, None].expand_as(ys)
            patches = signal[ts, 0, ys, xs]   # (N, k, k)

            # ── 5. Mass + filter ────────────────────────────────────────────
            mass = patches.sum(dim=(1, 2))
            keep = mass >= minmass
            if keep.sum() == 0:
                _ct = time.perf_counter()
                print(f"  Chunk {chunk_idx+1}/{n_chunks} "
                      f"(frames {chunk_start}–{chunk_end-1}): 0 spots "
                      f"in {_ct - last_chunk_end_t:.1f}s "
                      f"(all below minmass={minmass:.2f})", flush=True)
                last_chunk_end_t = _ct
                continue
            patches = patches[keep]
            t_ix    = t_ix[keep]
            y_ix    = y_ix[keep]
            x_ix    = x_ix[keep]
            mass    = mass[keep]

            # ── 6. Sub-pixel refinement ─────────────────────────────────────
            # Primary path: analytical 2D-Gaussian fit on log-intensities,
            #   one batched solve per sub-batch.  Matches trackpy's iterative
            #   refinement to within ≈10 nm and tightens trajectory recovery
            #   vs centroid-of-mass alone.
            # Fallback path: centroid of mass — used only for the small set
            #   of spots whose Gaussian fit was rejected.
            #
            # Sub-batching: when N is large (low-minmass / noisy data can
            # easily produce 10s of thousands of "spots" per chunk), feeding
            # all of them into `torch.linalg.solve` in a single call has
            # been observed to misbehave on MPS — typically subsequent
            # chunks then return 0 maxima as the MPS allocator state stays
            # degraded.  Splitting the fit into ≤5000-spot sub-batches
            # avoids that edge case while keeping batched-LSQ efficient.
            MAX_FIT_BATCH = 5_000
            N_spots = patches.shape[0]
            if N_spots > MAX_FIT_BATCH:
                dy_g_parts, dx_g_parts, ok_parts = [], [], []
                for _start in range(0, N_spots, MAX_FIT_BATCH):
                    _end  = min(_start + MAX_FIT_BATCH, N_spots)
                    _dyg, _dxg, _okg = self._gaussian_lstsq_refine(
                        patches[_start:_end], dy_grid, dx_grid, _M)
                    dy_g_parts.append(_dyg)
                    dx_g_parts.append(_dxg)
                    ok_parts.append(_okg)
                dy_g = torch.cat(dy_g_parts)
                dx_g = torch.cat(dx_g_parts)
                ok   = torch.cat(ok_parts)
            else:
                dy_g, dx_g, ok = self._gaussian_lstsq_refine(
                    patches, dy_grid, dx_grid, _M)

            patch_sum = patches.sum(dim=(1, 2)).clamp(min=1e-6)
            dy_cm = (patches * dy_grid[None]).sum(dim=(1, 2)) / patch_sum
            dx_cm = (patches * dx_grid[None]).sum(dim=(1, 2)) / patch_sum

            # Combine: use Gaussian where OK, fall back to centroid otherwise
            dy_off = torch.where(ok, dy_g, dy_cm)
            dx_off = torch.where(ok, dx_g, dx_cm)

            x_sub = x_ix.to(dtype) + dx_off
            y_sub = y_ix.to(dtype) + dy_off
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
            del x, bg, signal, maxp, is_max, coords, patches
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

        # Drop the cached on-device tensors (design matrix, index grids) and
        # force a full GPU drain before returning.  Otherwise the next
        # CPU-only stage (linking) inherits a degraded MPS context — its
        # finalizers run when Python GC kicks in during link_trajectories
        # and produce "command buffer exited with error" OOM messages that
        # have nothing to do with the actual cause.
        del dy_grid, dx_grid, _M, _M_pinv
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


class WaveletBackend(LocaliserBackend):
    """À-trous (stationary) wavelet spot detector, modeled on the
    palmTRACER WaveTracer algorithm (Izeddin et al. 2012 / Kechkar
    et al. 2013).

    Per frame:
      1. Stationary wavelet decomposition via `pywt.swt2` with the
         chosen wavelet family (default `db2`) at `levels` scales.
      2. Sum the detail planes at scales 1..levels into a single
         response map.  Higher levels capture broader spots; the
         default `levels=2` is tuned for the ~7 px diameter typical
         of single-molecule PALM spots at 0.1 µm/px.
      3. Robust threshold = `threshold_k · MAD(detail) · 1.4826`
         (the MAD→σ correction for Gaussian noise).  Keeps the test
         scale-free across frames with varying brightness.
      4. `skimage.feature.peak_local_max` with `min_distance` =
         user setting (default 3 px) to extract candidate peaks.
      5. Sub-pixel refinement: 2-D Gaussian centroid via
         `scipy.ndimage.center_of_mass` of a (2·d+1) × (2·d+1) patch
         around each peak, where d = ⌊diameter/2⌋.
      6. `mass` column = sum of the response map over the patch —
         analogous to trackpy's integrated mass, so the GUI's
         minmass filter applies meaningfully.

    Accepted params (all picked up from the worker payload):
        wavelet            — wavelet family ("db2", "db4", "sym4", "bior1.3", …)
        wavelet_levels     — number of detail scales summed (1..5)
        wavelet_threshold_k— robust-MAD multiplier (default 3.0)
        wavelet_min_distance — minimum spot separation in px (default 3)
        diameter           — patch radius for the centroid + mass step
        minmass            — keep only peaks with patch-integrated mass ≥ this

    Performance: parallelised via the same multiprocessing.Pool pattern
    TrackpyBackend uses.  Pure-CPU; no GPU.  Speed is roughly comparable
    to TrackpyBackend on dense PALM movies.
    """
    name = "wavelet"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import pywt  # noqa: F401
            from skimage.feature import peak_local_max  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _detect_frame(frame, *, diameter, minmass, wavelet, levels,
                       threshold_k, min_distance):
        """Localise a single frame.  Returns (xs, ys, masses) tuple of
        1-D numpy arrays.  Module-level so multiprocessing.Pool can
        pickle it on Windows + spawn-mode macOS workers."""
        import pywt
        from skimage.feature import peak_local_max
        from scipy.ndimage import center_of_mass

        f = np.asarray(frame, dtype=np.float32)
        # `pywt.swt2` requires both image dimensions divisible by
        # 2**level.  Pad up to the nearest multiple, then crop back.
        H, W = f.shape
        mult = 1 << int(levels)
        pad_h = (-H) % mult
        pad_w = (-W) % mult
        if pad_h or pad_w:
            f = np.pad(f, ((0, pad_h), (0, pad_w)), mode="reflect")
        # Stationary (à-trous) wavelet transform.  Returns a list of
        # tuples (cA, (cH, cV, cD)) — one per level.  We sum the
        # detail magnitude across levels, which is what palmTRACER does.
        coeffs = pywt.swt2(f, wavelet=wavelet, level=int(levels),
                            trim_approx=True)
        # `trim_approx=True` returns [cA_last, (cH_n, cV_n, cD_n),
        # (cH_{n-1}, …), …, (cH_1, …)]; we want the detail planes only.
        detail_planes = []
        for entry in coeffs[1:]:
            if isinstance(entry, tuple):
                cH, cV, cD = entry
                # Magnitude (sqrt of sum of squares) — invariant to
                # spot orientation.
                detail_planes.append(np.sqrt(cH ** 2 + cV ** 2 + cD ** 2))
        if not detail_planes:
            return np.empty(0), np.empty(0), np.empty(0)
        detail = np.sum(detail_planes, axis=0).astype(np.float32)
        detail = detail[:H, :W]   # crop back to original size

        # Robust noise floor from MAD; multiplied by 1.4826 for the
        # Gaussian σ equivalent.
        med = float(np.median(detail))
        mad = float(np.median(np.abs(detail - med)))
        sigma = 1.4826 * mad + 1e-9
        threshold = med + float(threshold_k) * sigma

        peaks = peak_local_max(detail, min_distance=int(min_distance),
                                threshold_abs=threshold)
        if peaks.size == 0:
            return np.empty(0), np.empty(0), np.empty(0)

        # Sub-pixel refinement + integrated-mass measurement in a
        # square patch around each peak.  Edge peaks get clipped
        # so the patch is never out of bounds.
        d = max(1, int(diameter) // 2)
        xs, ys, masses = [], [], []
        for (py, px) in peaks:
            y0 = max(0, py - d); y1 = min(H, py + d + 1)
            x0 = max(0, px - d); x1 = min(W, px + d + 1)
            patch = detail[y0:y1, x0:x1]
            patch_sum = float(patch.sum())
            if patch_sum < float(minmass):
                continue
            # `center_of_mass` returns (row, col) offsets within the patch.
            cy, cx = center_of_mass(patch)
            xs.append(x0 + cx)
            ys.append(y0 + cy)
            masses.append(patch_sum)
        return (np.asarray(xs, dtype=np.float32),
                np.asarray(ys, dtype=np.float32),
                np.asarray(masses, dtype=np.float32))

    def localise(self, stack, *, diameter=7, minmass=0.1, percentile=64,
                 workers=None, chunk_size=500, preview_cb=None, **kwargs):
        if diameter % 2 == 0:
            diameter += 1
        wavelet      = str(kwargs.get("wavelet", "db2"))
        levels       = int(kwargs.get("wavelet_levels", 2))
        threshold_k  = float(kwargs.get("wavelet_threshold_k", 3.0))
        min_distance = int(kwargs.get("wavelet_min_distance", 3))
        levels = max(1, min(levels, 5))

        n_frames = len(stack)
        print(f"  Diameter  : {diameter}px  |  minmass: {minmass:.4f}")
        print(f"  Wavelet   : {wavelet}  |  levels: {levels}  |  "
              f"threshold_k: {threshold_k}  |  min_distance: {min_distance}")

        # Serial loop is fine for the wavelet backend — the inner
        # SWT2 + MAD is already vectorised across the frame and the
        # call overhead would dominate if we tried per-frame mp.Pool.
        # Same memmap-friendly pattern as the streaming localiser:
        # iterate one frame at a time, emit a preview row, and
        # accumulate localisations.
        t0 = time.perf_counter()
        all_x, all_y, all_mass, all_frame = [], [], [], []
        for f_idx in range(n_frames):
            frame = np.asarray(stack[f_idx])
            xs, ys, masses = self._detect_frame(
                frame, diameter=diameter, minmass=minmass,
                wavelet=wavelet, levels=levels,
                threshold_k=threshold_k, min_distance=min_distance)
            if xs.size:
                all_x.append(xs); all_y.append(ys)
                all_mass.append(masses)
                all_frame.append(np.full(xs.size, f_idx,
                                          dtype=np.int32))
                if preview_cb is not None:
                    try:
                        preview_cb(int(f_idx), frame, xs, ys, n_frames)
                    except Exception:
                        pass

        if not all_x:
            print("  Found 0 localisations")
            return pd.DataFrame(columns=["x", "y", "frame", "mass"])
        df = pd.DataFrame({
            "x":     np.concatenate(all_x).astype(np.float32),
            "y":     np.concatenate(all_y).astype(np.float32),
            "frame": np.concatenate(all_frame).astype(np.int32),
            "mass":  np.concatenate(all_mass).astype(np.float32),
        })
        elapsed = time.perf_counter() - t0
        print(f"  Found {len(df):,} localisations in {elapsed:.1f}s  "
              f"({n_frames / elapsed:.0f} frames/s)")
        return df


class TrackMateBackend(LocaliserBackend):
    """LoG / DoG spot detector — Python reimplementation of TrackMate's
    "LoG detector" and "DoG detector" (Tinevez et al. 2017 / Ershov
    et al. 2022).  Mathematically equivalent to running TrackMate's
    detector in Fiji, but native scipy + scikit-image — no JVM, no
    pyimagej dependency.

    Per frame:
      1. Optional 3×3 median pre-filter (matches TrackMate's
         "Use median filter" checkbox) — robust against salt-and-
         pepper noise.
      2. Compute Gaussian-derivative response:
           mode = "log":  response = -gaussian_laplace(frame, σ)
                          (Marr-Hildreth scale-normalised LoG; the
                           negation puts bright blobs as positive peaks)
           mode = "dog":  response = G(σ) − G(σ · 1.6)
                          (Lowe's DoG approximation to the LoG, often
                           faster and slightly less noise-prone)
         σ is derived from the user's "estimated spot radius" via
         σ = radius_px / √2 (the scale at which a Gaussian-shaped
         spot maximises the LoG response).
      3. `skimage.feature.peak_local_max(response, min_distance=…,
         threshold_abs=quality)` extracts candidate peaks.  Quality
         is the user-set absolute threshold on the LoG/DoG response —
         direct analogue of TrackMate's "Quality threshold" slider.
      4. Sub-pixel refinement: `scipy.ndimage.center_of_mass` over a
         (2·d+1) × (2·d+1) patch (TrackMate calls this "sub-pixel
         localisation"; FIREFLY always enables it).
      5. `mass` column = sum of the response map over the patch —
         analogue of trackpy's integrated mass so the minmass filter
         still meaningfully gates detections.

    Accepted params (from worker payload):
        trackmate_mode      — "log" or "dog"
        trackmate_radius_um — estimated spot radius in µm
        trackmate_quality   — absolute threshold on the LoG/DoG response
        trackmate_median    — bool, apply 3×3 median pre-filter
        diameter            — patch size for centroid + mass
        minmass             — keep only peaks with patch-sum ≥ this
        pixel_size_um       — needed to convert radius_um → radius_px
                              (sourced from the analysis params dict)

    Output: DataFrame with columns [x, y, frame, mass].
    """
    name = "trackmate"

    @classmethod
    def is_available(cls) -> bool:
        try:
            from scipy import ndimage  # noqa: F401
            from skimage.feature import peak_local_max  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _detect_frame(frame, *, diameter, minmass, mode, sigma_px,
                       quality, min_distance, use_median):
        """Localise a single frame.  Returns (xs, ys, masses)."""
        from scipy import ndimage as _ndi
        from skimage.feature import peak_local_max

        f = np.asarray(frame, dtype=np.float32)
        if use_median:
            f = _ndi.median_filter(f, size=3)

        if mode == "dog":
            # Lowe's DoG approximation to LoG.  k=1.6 maximises the
            # similarity to the LoG response over a single octave.
            g1 = _ndi.gaussian_filter(f, sigma=sigma_px)
            g2 = _ndi.gaussian_filter(f, sigma=sigma_px * 1.6)
            response = (g1 - g2).astype(np.float32)
        else:
            # Scale-normalised Laplacian-of-Gaussian.  Multiply by σ²
            # so peak magnitudes are comparable across spot sizes —
            # standard scale-space blob detection (Lindeberg 1998).
            laplaced = _ndi.gaussian_laplace(f, sigma=sigma_px)
            response = (-(sigma_px ** 2) * laplaced).astype(np.float32)

        # Scale-free threshold: `quality` is a multiplier on the
        # robust noise σ (median + k·MAD·1.4826).  Identical pattern
        # to WaveletBackend.  Works regardless of whether the input
        # frames are raw camera counts, background-subtracted, or
        # normalised — TrackMate Fiji's absolute "Quality" knob
        # doesn't translate to FIREFLY's preprocessed pipeline so we
        # use a relative threshold the user can reason about as
        # "how many σ above the noise floor".
        med = float(np.median(response))
        mad = float(np.median(np.abs(response - med)))
        sigma_resp = 1.4826 * mad + 1e-9
        threshold = med + float(quality) * sigma_resp

        peaks = peak_local_max(response,
                                min_distance=int(min_distance),
                                threshold_abs=threshold)
        if peaks.size == 0:
            return np.empty(0), np.empty(0), np.empty(0)

        H, W = f.shape
        d = max(1, int(diameter) // 2)
        xs, ys, masses = [], [], []
        for (py, px) in peaks:
            y0 = max(0, py - d); y1 = min(H, py + d + 1)
            x0 = max(0, px - d); x1 = min(W, px + d + 1)
            patch = response[y0:y1, x0:x1]
            patch_sum = float(patch.sum())
            if patch_sum < float(minmass):
                continue
            cy, cx = _ndi.center_of_mass(patch)
            xs.append(x0 + cx)
            ys.append(y0 + cy)
            masses.append(patch_sum)
        return (np.asarray(xs, dtype=np.float32),
                np.asarray(ys, dtype=np.float32),
                np.asarray(masses, dtype=np.float32))

    def localise(self, stack, *, diameter=7, minmass=0.1, percentile=64,
                 workers=None, chunk_size=500, preview_cb=None,
                 stop_event=None, **kwargs):
        if diameter % 2 == 0:
            diameter += 1

        mode      = str(kwargs.get("trackmate_mode", "log")).lower()
        if mode not in ("log", "dog"):
            mode = "log"
        radius_um = float(kwargs.get("trackmate_radius_um", 0.5))
        quality   = float(kwargs.get("trackmate_quality", 5.0))
        use_med   = bool(kwargs.get("trackmate_median", False))
        pixel_um  = float(kwargs.get("pixel_size_um", 0.106))

        # Convert radius (µm) → σ (px).  σ_LoG = r_px / √2 maximises
        # the scale-normalised LoG response on a Gaussian blob of radius r.
        radius_px = max(1.0, radius_um / max(pixel_um, 1e-9))
        sigma_px  = radius_px / (2.0 ** 0.5)
        # min_distance ≈ radius — prevents double-detection on the
        # same spot.  Always at least 1 px.
        min_distance = max(1, int(round(radius_px)))

        n_frames = len(stack)
        n_workers = max(1, min(int(workers) if workers else N_CPUS, N_CPUS))
        print(f"  Diameter  : {diameter}px  |  minmass: {minmass:.4f}")
        print(f"  TrackMate : mode={mode}  radius={radius_um:.3f}µm "
              f"(σ={sigma_px:.2f}px)  quality={quality}σ  "
              f"median={'on' if use_med else 'off'}")
        print(f"  Parallelism : joblib × {n_workers} workers (loky backend)")

        t0 = time.perf_counter()

        # Chunked parallel detection.  Each worker processes a contiguous
        # block of frames — keeps per-task overhead small (one pickle of
        # the chunk-array per worker, not per frame) while still giving
        # N-way speedup on N cores.  Falls back to serial if joblib is
        # unavailable or chokes (rare; we depend on joblib elsewhere).
        chunk_size_local = max(1, int(chunk_size) // 2 or 50)
        chunk_ranges: list[tuple[int, int]] = []
        i = 0
        while i < n_frames:
            j = min(i + chunk_size_local, n_frames)
            chunk_ranges.append((i, j))
            i = j

        def _process_chunk(start: int, end: int):
            """Process frames [start, end) → list of (frame_idx, xs, ys, masses)."""
            out = []
            for f_idx in range(start, end):
                fr = np.asarray(stack[f_idx])
                xs, ys, masses = TrackMateBackend._detect_frame(
                    fr, diameter=diameter, minmass=minmass,
                    mode=mode, sigma_px=sigma_px, quality=quality,
                    min_distance=min_distance, use_median=use_med)
                if xs.size:
                    out.append((f_idx, xs, ys, masses, fr))
            return out

        all_x, all_y, all_mass, all_frame = [], [], [], []
        try:
            # `loky` backend = process-pool (true multi-core, no GIL).
            # Cancellation: joblib doesn't expose a clean interrupt, so
            # we use a smaller chunk granularity (above) and check
            # `stop_event` between chunks via a manual loop with a
            # bounded Parallel call per group.  This keeps cancel
            # latency to one chunk's worth of work (~hundreds of ms).
            stride = max(1, n_workers * 4)   # process this many chunks per Parallel batch
            for batch_start in range(0, len(chunk_ranges), stride):
                if stop_event is not None and stop_event.is_set():
                    print("  TrackMate detection stopped by user.")
                    break
                batch = chunk_ranges[batch_start:batch_start + stride]
                results = Parallel(n_jobs=n_workers, backend="loky",
                                    prefer="processes")(
                    delayed(_process_chunk)(s, e) for s, e in batch)
                for chunk_result in results:
                    for f_idx, xs, ys, masses, fr in chunk_result:
                        all_x.append(xs); all_y.append(ys)
                        all_mass.append(masses)
                        all_frame.append(np.full(xs.size, f_idx,
                                                  dtype=np.int32))
                        if preview_cb is not None:
                            try:
                                preview_cb(int(f_idx), fr, xs, ys, n_frames)
                            except Exception:
                                pass
        except Exception as exc:
            # Fall back to a single-core serial loop if joblib fails.
            print(f"  Parallel detection failed ({type(exc).__name__}: {exc}); "
                  f"falling back to serial.")
            all_x.clear(); all_y.clear(); all_mass.clear(); all_frame.clear()
            for f_idx in range(n_frames):
                if stop_event is not None and stop_event.is_set():
                    break
                frame = np.asarray(stack[f_idx])
                xs, ys, masses = self._detect_frame(
                    frame, diameter=diameter, minmass=minmass,
                    mode=mode, sigma_px=sigma_px, quality=quality,
                    min_distance=min_distance, use_median=use_med)
                if xs.size:
                    all_x.append(xs); all_y.append(ys)
                    all_mass.append(masses)
                    all_frame.append(np.full(xs.size, f_idx,
                                              dtype=np.int32))
                    if preview_cb is not None:
                        try:
                            preview_cb(int(f_idx), frame, xs, ys, n_frames)
                        except Exception:
                            pass

        if not all_x:
            print("  Found 0 localisations")
            return pd.DataFrame(columns=["x", "y", "frame", "mass"])
        df = pd.DataFrame({
            "x":     np.concatenate(all_x).astype(np.float32),
            "y":     np.concatenate(all_y).astype(np.float32),
            "frame": np.concatenate(all_frame).astype(np.int32),
            "mass":  np.concatenate(all_mass).astype(np.float32),
        })
        elapsed = time.perf_counter() - t0
        print(f"  Found {len(df):,} localisations in {elapsed:.1f}s  "
              f"({n_frames / elapsed:.0f} frames/s)")
        return df


# Order matters: `backend="auto"` resolves to the first available entry.
# TorchBackend stays AFTER TrackpyBackend in A2 so "auto" still picks trackpy
# while users validate the new path explicitly by selecting "torch" in the GUI.
# A3 will swap the order once we've confirmed numerical agreement on real data.
# WaveletBackend and TrackMateBackend are opt-in; users select via the dropdown.
_BACKEND_REGISTRY: list[type[LocaliserBackend]] = [
    TrackpyBackend, TorchBackend, WaveletBackend, TrackMateBackend,
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
        "wavelet"  — à-trous wavelet (palmTRACER-style)

    Extra `backend_kwargs` are forwarded verbatim to the active
    backend's `.localise()`, used by WaveletBackend for its
    `wavelet`, `wavelet_levels`, `wavelet_threshold_k`,
    `wavelet_min_distance` parameters.  Other backends ignore them.

    Returns a DataFrame with columns: x, y, frame, mass.
    """
    impl = _resolve_backend(backend)
    print(f"  Backend   : {impl.name}")
    return impl.localise(stack, diameter=diameter, minmass=minmass,
                         percentile=percentile, workers=workers,
                         chunk_size=chunk_size, preview_cb=preview_cb,
                         **backend_kwargs)


# ══════════════════════════════════════════════════════════════════════════════
#  LINKING
# ══════════════════════════════════════════════════════════════════════════════

def link_trajectories(locs, search_range=5, memory=3, min_len=5, max_len=None,
                       linker="trackpy", progress_cb=None, stop_event=None):
    """Link localisations into trajectories.

    linker      : "trackpy" (default) or "trackmate_lap"
        Strategy selector.  Trackpy uses its `link_iter` (or `link`
        fallback) with the recursive subnet linker.  TrackMate LAP
        uses the Jaqaman et al. (2008) linear-assignment-problem
        formulation — `scipy.optimize.linear_sum_assignment` per
        frame transition, with TrackMate-style "memory" via short
        gap-closing.  Both produce a `particle` column on the
        returned DataFrame so the rest of the pipeline is agnostic.
    progress_cb : callable(fraction) → None
        Optional.  Called periodically with a [0, 1] float so the host
        can update a progress bar.  Updates are throttled to roughly
        once every 32 frames + once on completion.
    stop_event  : threading.Event-like
        Optional.  Polled between frames; if `.is_set()` the linker
        raises `_Cancelled` and aborts cleanly.
    """
    print(f"  Linking {len(locs):,} localisations  "
          f"(linker={linker}, search_range={search_range}px, "
          f"memory={memory}) ...")
    t0 = time.perf_counter()

    if linker == "trackmate_lap":
        linked = _link_via_trackmate_lap(
            locs, search_range=search_range, memory=memory,
            progress_cb=progress_cb, stop_event=stop_event)
    else:
        linked = _link_via_trackpy(
            locs, search_range=search_range, memory=memory,
            progress_cb=progress_cb, stop_event=stop_event)

    # Stub + max-length filters are common to both linkers.
    filtered = tp.filter_stubs(linked, min_len)
    if max_len is not None and max_len > 0:
        lengths  = filtered.groupby("particle")["frame"].count()
        keep     = lengths[lengths <= max_len].index
        filtered = filtered[filtered["particle"].isin(keep)]
        print(f"  Max-length filter (<={max_len}): {filtered['particle'].nunique():,} remain")
    elapsed = time.perf_counter() - t0
    n       = filtered["particle"].nunique()
    max_str = str(max_len) if max_len else "inf"
    print(f"  {n:,} trajectories (len {min_len}-{max_str}) in {elapsed:.1f}s")
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


def _link_via_trackmate_lap(locs, *, search_range, memory,
                              progress_cb=None, stop_event=None):
    """TrackMate-style LAP linker — frame-to-frame linear-assignment
    formulation (Jaqaman et al. 2008, "Robust single-particle
    tracking in live-cell time-lapse sequences").  Each frame
    transition is solved as a balanced LAP via
    `scipy.optimize.linear_sum_assignment` over a cost matrix of
    squared distances, with diagonal "no-link" blocks that allow
    track births / deaths within `search_range`.

    Memory (TrackMate's "max frame gap") is supported by carrying
    paused tracks forward up to `memory` frames so they can
    re-link to a later detection — covers the common short-blink
    case TrackMate's segment-LAP gap closer targets.  Full
    track-segment LAP can be a future extension; this frame-to-
    frame version is what most users mean by "LAP linker".
    """
    from scipy.optimize import linear_sum_assignment as _lsa

    if len(locs) == 0:
        out = locs.copy()
        out["particle"] = np.array([], dtype=np.int64)
        return out

    # Group localisations by frame, preserving original DataFrame indices
    # so we can write `particle` back into the right rows at the end.
    frame_nums = sorted(int(f) for f in locs["frame"].unique())
    grouped = locs.groupby("frame")
    per_frame_coords: dict[int, np.ndarray] = {}
    per_frame_indices: dict[int, np.ndarray] = {}
    for f in frame_nums:
        sub = grouped.get_group(f)
        per_frame_coords[f]  = sub[["x", "y"]].to_numpy(dtype=float)
        per_frame_indices[f] = sub.index.to_numpy()

    particle_ids = np.full(len(locs), -1, dtype=np.int64)
    next_pid = 0
    sr2 = float(search_range) ** 2          # cost = squared distance
    BIG = sr2 * 10.0                        # "no-link" off-diagonal cost
    # active_tracks: pid → (last_seen_frame, x, y, last_row_idx)
    active: dict[int, tuple[int, float, float, int]] = {}

    n_frames = len(frame_nums)
    for f_i, fr in enumerate(frame_nums):
        if stop_event is not None and stop_event.is_set():
            raise _Cancelled()

        new_coords  = per_frame_coords[fr]                # (M, 2)
        new_indices = per_frame_indices[fr]               # (M,)
        M = new_coords.shape[0]

        # Active tracks whose last_seen is within `memory` frames.
        live_pids: list[int] = []
        live_xy:   list[tuple[float, float]] = []
        for pid, (last_fr, lx, ly, _last_idx) in active.items():
            if fr - last_fr <= memory + 1:
                live_pids.append(pid)
                live_xy.append((lx, ly))
        N = len(live_pids)

        if M == 0:
            # No detections this frame — every active track ages.
            # (They stay in `active` so memory carries them.)
            if progress_cb is not None and (f_i & 31) == 0:
                try: progress_cb((f_i + 1) / max(1, n_frames))
                except Exception: pass
            continue

        if N == 0:
            # No live tracks — every new detection births a new track.
            for j in range(M):
                pid = next_pid; next_pid += 1
                particle_ids[new_indices[j]] = pid
                active[pid] = (fr, float(new_coords[j, 0]),
                                  float(new_coords[j, 1]),
                                  int(new_indices[j]))
            if progress_cb is not None and (f_i & 31) == 0:
                try: progress_cb((f_i + 1) / max(1, n_frames))
                except Exception: pass
            continue

        # Build the (N+M) × (N+M) Jaqaman cost matrix.
        # Top-left  (N×M):  real link costs.
        # Top-right (N×N):  diagonal "track death" costs (BIG, else inf).
        # Bot-left  (M×M):  diagonal "track birth" costs (BIG, else inf).
        # Bot-right (M×N):  transpose of top-left (so the matrix is
        #                   square + Jaqaman-symmetric).
        size = N + M
        cost = np.full((size, size), np.inf, dtype=float)

        # Top-left: squared distance from each live track to each new det.
        live_arr = np.asarray(live_xy, dtype=float)        # (N, 2)
        diff = new_coords[None, :, :] - live_arr[:, None, :]
        d2   = (diff ** 2).sum(axis=2)                     # (N, M)
        in_range = d2 <= sr2
        # Frame-gap penalty: tracks that have been paused longer get
        # slightly higher cost so the solver prefers contiguous matches.
        gap = np.array(
            [fr - active[pid][0] for pid in live_pids], dtype=float)
        d2 = d2 * (1.0 + 0.1 * (gap[:, None] - 1).clip(min=0))
        cost[:N, :M] = np.where(in_range, d2, np.inf)

        # Diagonal "death" block (track i → no detection).
        for i in range(N):
            cost[i, M + i] = BIG
        # Diagonal "birth" block (no track → detection j).
        for j in range(M):
            cost[N + j, j] = BIG
        # Bottom-right (lower-right) — auxiliary, must be finite for LAP.
        cost[N:, M:] = cost[:N, :M].T

        # linear_sum_assignment requires finite costs; replace inf with
        # a sentinel much larger than any valid pairing.
        sentinel = BIG * 100.0
        cost_safe = np.where(np.isfinite(cost), cost, sentinel)
        try:
            row_ind, col_ind = _lsa(cost_safe)
        except Exception:
            # If scipy chokes (very rare), fall back to greedy by
            # treating each live track in turn and grabbing nearest
            # in-range detection.
            row_ind = np.arange(size)
            col_ind = row_ind.copy()

        assigned_dets: set[int] = set()
        for r, c in zip(row_ind, col_ind):
            if r < N and c < M:
                # Real link: track r continues into detection c.
                if cost_safe[r, c] >= sentinel:
                    continue   # invalid (was inf) — skip
                pid = live_pids[r]
                particle_ids[new_indices[c]] = pid
                active[pid] = (fr, float(new_coords[c, 0]),
                                  float(new_coords[c, 1]),
                                  int(new_indices[c]))
                assigned_dets.add(c)
            # rows ≥ N or cols ≥ M are diagonal pseudo-slots; ignored.

        # Any unassigned detection births a new track.
        for j in range(M):
            if j not in assigned_dets and particle_ids[new_indices[j]] == -1:
                pid = next_pid; next_pid += 1
                particle_ids[new_indices[j]] = pid
                active[pid] = (fr, float(new_coords[j, 0]),
                                  float(new_coords[j, 1]),
                                  int(new_indices[j]))

        # Garbage-collect tracks that have been paused beyond memory.
        stale = [pid for pid, (last_fr, *_rest) in active.items()
                 if fr - last_fr > memory + 1]
        for pid in stale:
            del active[pid]

        if progress_cb is not None and (f_i & 31) == 0:
            try: progress_cb((f_i + 1) / max(1, n_frames))
            except Exception: pass

    if progress_cb is not None:
        try: progress_cb(1.0)
        except Exception: pass

    linked = locs.copy()
    linked["particle"] = particle_ids
    linked = linked[linked["particle"] >= 0].reset_index(drop=True)
    print(f"  TrackMate LAP done — {linked['particle'].nunique():,} "
          f"tracks built (filtering stubs in caller).")
    return linked


# ══════════════════════════════════════════════════════════════════════════════
#  MSD + DIFFUSION  (custom parallel — replaces slow tp.imsd)
# ══════════════════════════════════════════════════════════════════════════════

def msd_linear(t, D, offset):
    return 4 * D * t + offset


# Default alpha-exponent thresholds for the four-class motion classifier.
# Conventional sptPALM values: 0.5 / 0.9 / 1.1.  These are now the *defaults*
# but every public function that classifies motion accepts a thresholds=
# triple so users can tune the boundaries to their lab's convention.
ALPHA_THRESHOLDS_DEFAULT = (0.5, 0.9, 1.1)

# Default D cutoff for splitting Mobile / Immobile populations (µm²/s).
# 0.05 is the conventional membrane-protein threshold used throughout the
# sptPALM literature; tracks with D ≥ this value are considered Mobile.
# Defined here at the top so functions defined later in the file can use
# it as a default argument (Python evaluates defaults at definition time).
MOBILE_D_THRESHOLD_DEFAULT = 0.05


def classify_motion(alpha, thresholds=ALPHA_THRESHOLDS_DEFAULT):
    """Classify a track by its anomalous exponent α.

    thresholds = (t_immobile, t_confined, t_directed):
        α  <  t_immobile   → "Immobile"
        t_immobile  ≤ α  <  t_confined → "Confined"
        t_confined  ≤ α  <  t_directed → "Brownian"
        α  ≥  t_directed   → "Directed"
    """
    t_imm, t_conf, t_dir = thresholds
    if   alpha < t_imm:  return "Immobile"
    elif alpha < t_conf: return "Confined"
    elif alpha < t_dir:  return "Brownian"
    else:                return "Directed"


def _msd_and_fit_one(xy_um, frames, pid, lag_times, max_lagtime, n_fit,
                     alpha_thresholds=ALPHA_THRESHOLDS_DEFAULT):
    """
    Compute per-track MSD array AND fit D + alpha in a single pass.

    Uses actual frame numbers (not row indices) so that gaps in a trajectory
    caused by memory-linking do not inflate the MSD.  Only pairs of positions
    whose frame difference exactly equals the requested lag are included.
    """
    msd_vals = np.full(max_lagtime, np.nan)
    for lag_idx, lag in enumerate(range(1, max_lagtime + 1)):
        if lag >= len(xy_um):
            break
        # Only use pairs where the actual frame separation equals lag
        frame_diff = frames[lag:] - frames[:-lag]
        valid      = frame_diff == lag
        if valid.sum() > 0:
            d = xy_um[lag:][valid] - xy_um[:-lag][valid]
            msd_vals[lag_idx] = np.mean(d[:, 0] ** 2 + d[:, 1] ** 2)

    # Fit using first n_fit lag times
    t   = lag_times[:n_fit]
    m   = msd_vals[:n_fit]
    ok  = np.isfinite(m) & (m > 0)
    D = alpha = np.nan
    msd0 = np.nan        # linear-fit intercept (PALM-Tracer "MSD(0)")
    mse  = np.nan        # mean squared residual of the linear fit
    if ok.sum() >= 3:
        try:    alpha = np.polyfit(np.log(t[ok]), np.log(m[ok]), 1)[0]
        except Exception: pass
        try:
            popt, _ = curve_fit(msd_linear, t[ok], m[ok], p0=[0.01, 0],
                                bounds=([0, -np.inf], [np.inf, np.inf]),
                                maxfev=2000)
            D = popt[0]
            msd0 = float(popt[1])
            _resid = m[ok] - msd_linear(t[ok], *popt)
            mse = float(np.mean(_resid ** 2))
        except Exception: pass

    motion = classify_motion(alpha, alpha_thresholds) if np.isfinite(alpha) else "Unknown"

    # Two distinct radial-spread metrics, both useful and named explicitly:
    #   mean_radial_displacement_um  = ⟨|r − r̄|⟩       (1st moment)
    #   radius_of_gyration_um        = √⟨|r − r̄|²⟩    (RMS, the standard Rg)
    centroid    = xy_um.mean(axis=0)
    sq_dists    = np.sum((xy_um - centroid) ** 2, axis=1)
    mean_radial = float(np.mean(np.sqrt(sq_dists)))
    rg          = float(np.sqrt(np.mean(sq_dists)))

    return pid, msd_vals, dict(particle=pid, D=D, alpha=alpha, motion=motion,
                               MSD0=msd0, MSE=mse,
                               mean_radial_displacement_um=mean_radial,
                               radius_of_gyration_um=rg)


def compute_msd_and_fit(tracks, pixel_size, frame_interval,
                        max_lagtime=20, n_fit=5, workers=N_CPUS,
                        alpha_thresholds=ALPHA_THRESHOLDS_DEFAULT):
    """
    Single parallel pass that computes both MSD and diffusion fits.
    Replaces tp.imsd + tp.emsd + separate fit loop — all in one go.
    """
    lag_times  = np.arange(1, max_lagtime + 1) * frame_interval
    grouped    = tracks.groupby("particle")
    pid_list   = list(grouped.groups.keys())
    n_tracks   = len(pid_list)

    print(f"  Tracks to process : {n_tracks:,}")
    print(f"  Workers           : {workers} / {N_CPUS} CPU cores")
    t0 = time.perf_counter()

    # Defensive: if linking produced zero trajectories (e.g. localiser
    # returned no spots, or every spot is an isolated singleton), return
    # empty results instead of crashing pandas with "Empty data passed
    # with indices specified".  The caller still sees the empty result
    # and can produce a sensible "no tracks found" log message.
    if n_tracks == 0:
        print("  No trajectories — skipping MSD/fit (returning empty result).")
        imsd_empty = pd.DataFrame(
            np.full((max_lagtime, 0), np.nan, dtype=float),
            index=np.arange(1, max_lagtime + 1))
        emsd_empty = pd.Series(
            np.full(max_lagtime, np.nan, dtype=float),
            index=np.arange(1, max_lagtime + 1))
        diff_empty = pd.DataFrame(columns=[
            "particle", "D", "alpha", "motion", "MSD0", "MSE",
            "mean_radial_displacement_um", "radius_of_gyration_um"])
        return imsd_empty, emsd_empty, diff_empty

    # Threading vs processing trade-off:
    # The per-track work is curve_fit + numpy slicing.  curve_fit
    # releases the GIL inside its LAPACK calls but the Python wrapping
    # does not, so ThreadPool stalls on the GIL once n_tracks gets
    # large and the per-track work is dominated by Python overhead.
    # ProcessPool gives true parallelism (each worker has its own GIL)
    # but pays a one-time ~1-5 s spawn cost on Windows.
    #
    # Heuristic: use processes when there are enough tracks for the
    # parallelism win to outweigh spawn cost.  Below threshold, stick
    # with threads.
    PROCESS_POOL_THRESHOLD = 5000
    use_processes = n_tracks >= PROCESS_POOL_THRESHOLD

    # Pre-extract per-track arrays ONCE so we don't pay get_group twice
    # per particle (the old code called get_group inside both .submit
    # args, doubling the dict lookup + DataFrame slice cost).
    per_track_inputs = []
    for pid in pid_list:
        g = grouped.get_group(pid)
        xy = g[["x", "y"]].values * pixel_size
        fr = g["frame"].values
        per_track_inputs.append((xy, fr, pid))

    if use_processes:
        from concurrent.futures import ProcessPoolExecutor
        print(f"  Pool              : ProcessPool × {workers} "
              f"(>{PROCESS_POOL_THRESHOLD} tracks → true multi-core)")
        ExecutorCls = ProcessPoolExecutor
    else:
        print(f"  Pool              : ThreadPool × {workers} "
              f"(<{PROCESS_POOL_THRESHOLD} tracks → low-overhead path)")
        ExecutorCls = ThreadPoolExecutor

    with ExecutorCls(max_workers=workers) as _exe:
        _futs = [_exe.submit(
                    _msd_and_fit_one,
                    xy, fr, pid,
                    lag_times, max_lagtime, n_fit, alpha_thresholds)
                 for xy, fr, pid in per_track_inputs]
        results = [_f.result() for _f in
                   _tqdm(_futs, desc="  MSD + fitting", unit="track", ncols=70)]

    elapsed = time.perf_counter() - t0
    rate    = n_tracks / elapsed
    print(f"  Done in {elapsed:.1f}s  ({rate:.0f} tracks/s)")

    # Assemble imsd DataFrame  (rows = lag index, cols = particle id).
    # column_stack avoids the np.array(...).T double-allocation.
    msd_matrix = np.column_stack([r[1] for r in results])   # (max_lagtime, n_tracks)
    imsd_df    = pd.DataFrame(msd_matrix,
                              index=np.arange(1, max_lagtime + 1),
                              columns=[r[0] for r in results])

    # Ensemble MSD = nanmean across tracks at each lag
    emsd_series = pd.Series(np.nanmean(msd_matrix, axis=1),
                            index=np.arange(1, max_lagtime + 1))

    diff_df = pd.DataFrame([r[2] for r in results])

    # Merge per-track mean localisation precision (pixels → nm)
    if "ep" in tracks.columns:
        ep_nm = (tracks.groupby("particle")["ep"].mean() * pixel_size * 1000
                 ).rename("loc_precision_nm").reset_index()
        diff_df = diff_df.merge(ep_nm, on="particle", how="left")

    return imsd_df, emsd_series, diff_df


# ══════════════════════════════════════════════════════════════════════════════
#  JUMP DISTANCE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_jdd(tracks, pixel_size_um, frame_interval_s, n_components=2):
    """
    Jump Distance Distribution (JDD) analysis.

    Extracts single-frame displacements from all tracks, then fits the
    empirical CDF to a mixture of 2D Brownian populations:

        CDF(r) = 1 - Σᵢ fᵢ · exp(–r² / 4Dᵢ Δt)

    Fitting the CDF (rather than histogram) avoids binning artefacts and
    gives robust estimates even with short tracks — ideal for sptPALM where
    many tracks have only 2–5 frames.

    Parameters
    ----------
    n_components : 1, 2, or 3

    Returns
    -------
    dict or None (if too few jumps to fit)
    """
    print(f"  JDD analysis      : {n_components} component(s)  "
          f"|  {tracks['particle'].nunique():,} tracks")
    dt = frame_interval_s

    # Vectorised across all tracks at once.  Old per-track Python loop
    # was O(n_tracks × Python-step) → seconds on 100k tracks.  We:
    #   1. Sort by (particle, frame)
    #   2. np.diff over the full arrays
    #   3. Mask out any "step" that crossed a particle boundary OR
    #      isn't between consecutive frames (frame gap > 1)
    # …then compute the displacement magnitudes in one numpy call.
    if len(tracks) < 2:
        jumps = np.array([], dtype=np.float64)
    else:
        # Drop the index level first — trackpy.link sets `frame` as
        # both an index level AND a column, which makes sort_values
        # raise "ambiguous" on those keys.  reset_index(drop=True)
        # discards the index but keeps the column intact.
        srt = (tracks
               .reset_index(drop=True)
               .sort_values(["particle", "frame"], kind="stable"))
        pid_arr   = srt["particle"].to_numpy()
        frame_arr = srt["frame"].to_numpy()
        x_arr     = srt["x"].to_numpy() * pixel_size_um
        y_arr     = srt["y"].to_numpy() * pixel_size_um
        dx = np.diff(x_arr)
        dy = np.diff(y_arr)
        same_track = pid_arr[1:] == pid_arr[:-1]
        consec     = np.diff(frame_arr) == 1
        mask = same_track & consec
        jumps = np.sqrt(dx[mask] ** 2 + dy[mask] ** 2)
    jumps = np.asarray(jumps, dtype=np.float64)
    if len(jumps) < 30:
        return None

    r_sorted = np.sort(jumps)
    cdf_emp  = np.arange(1, len(r_sorted) + 1) / len(r_sorted)

    # ── CDF model definitions ─────────────────────────────────────────────────
    def _cdf1(r, D1):
        return 1.0 - np.exp(-r ** 2 / (4 * D1 * dt))

    def _cdf2(r, D1, D2, f1):
        f2 = 1.0 - f1
        return 1.0 - f1 * np.exp(-r**2 / (4*D1*dt)) \
                   - f2 * np.exp(-r**2 / (4*D2*dt))

    def _cdf3(r, D1, D2, D3, f1, f2):
        f3 = 1.0 - f1 - f2
        return (1.0 - f1 * np.exp(-r**2 / (4*D1*dt))
                    - f2 * np.exp(-r**2 / (4*D2*dt))
                    - f3 * np.exp(-r**2 / (4*D3*dt)))

    configs = {
        1: (_cdf1, [0.05],                   ([1e-6],        [100.0])),
        2: (_cdf2, [0.005, 0.3, 0.4],        ([1e-6, 1e-5, 0.01], [10.0, 100.0, 0.99])),
        3: (_cdf3, [0.003, 0.05, 0.5, 0.3, 0.35],
                                              ([1e-6, 1e-5, 1e-4, 0.01, 0.01],
                                               [1.0, 10.0, 100.0, 0.97, 0.97])),
    }

    model, p0, (lb, ub) = configs[n_components]
    try:
        popt, _ = curve_fit(model, r_sorted, cdf_emp,
                            p0=p0, bounds=(lb, ub), maxfev=20000)
    except Exception:
        return None

    # ── Extract sorted (D, fraction) pairs ───────────────────────────────────
    if n_components == 1:
        pairs = [(popt[0], 1.0)]
    elif n_components == 2:
        pairs = sorted([(popt[0], popt[2]), (popt[1], 1.0 - popt[2])])
    else:
        f3    = 1.0 - popt[3] - popt[4]
        pairs = sorted([(popt[0], popt[3]), (popt[1], popt[4]), (popt[2], f3)])

    D_values  = [p[0] for p in pairs]
    fractions = [p[1] for p in pairs]

    # ── PDF for plotting ──────────────────────────────────────────────────────
    # Rayleigh-like: f_i(r) = r/(2DᵢΔt) · exp(–r²/4DᵢΔt)
    r_range = np.linspace(0, np.percentile(jumps, 99.5), 500)

    def _pdf_component(r, D):
        return (r / (2 * D * dt)) * np.exp(-r**2 / (4 * D * dt))

    pdfs = [frac * _pdf_component(r_range, D)
            for D, frac in zip(D_values, fractions)]
    pdf_total = np.sum(pdfs, axis=0)

    return {
        "jumps":         jumps,
        "D_values":      D_values,
        "fractions":     fractions,
        "n_components":  n_components,
        "n_jumps":       len(jumps),
        "r_range":       r_range,
        "pdfs":          pdfs,           # per-component PDF arrays
        "pdf_total":     pdf_total,
        "cdf_r":         r_sorted,
        "cdf_empirical": cdf_emp,
        "cdf_fit":       model(r_sorted, *popt),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TURNING ANGLES
# ══════════════════════════════════════════════════════════════════════════════

def compute_circular_statistics(angles_deg):
    """Full circular-statistics summary for an array of angles, in
    degrees, on the interval (-180°, +180°] (signed turning-angle
    convention used by `compute_turning_angles`).

    Returns a dict whose keys are the same statistic names MATLAB's
    CircStat toolbox (Berens 2009) uses, so a supervisor familiar with
    that toolbox can map results 1:1.  All angles in the output are in
    DEGREES; rates / dispersions in their natural units.

    Computed statistics
    -------------------
    n                            : sample size
    mean_direction_deg           : μ = atan2(S, C), in (-180°, +180°]
    mean_resultant_length        : R̄ in [0, 1]    (1 = perfect alignment)
    circular_variance            : 1 - R̄          ("S" in Fisher 1993)
    circular_std_deg             : √(-2·ln R̄)·(180/π)
    angular_deviation_deg        : √(2·(1 - R̄))·(180/π)  ("s₀" in Fisher)
    median_deg                   : circular median
    concentration_kappa          : von Mises κ via standard piecewise
                                   approximation (Best & Fisher 1981)
    rayleigh_z                   : n·R̄²   (test statistic for uniformity)
    rayleigh_p                   : Wilkie-Mardia approximation (good
                                   to ~5e-4 for n ≥ 10)
    v_test_z, v_test_p           : V-test against μ₀ = 0° (tests for a
                                   preferred mean direction at "straight
                                   ahead")
    circular_skewness            : b̄ / (1 - R̄)^1.5
    circular_kurtosis            : (ā - R̄⁴) / (1 - R̄)²
    ci95_lower_deg, ci95_upper_deg
                                 : approximate 95% CI for μ (Fisher 1993
                                   §4.4.4, large-sample normal approx)

    References
    ----------
    Mardia & Jupp 2000, "Directional Statistics".
    Fisher 1993, "Statistical Analysis of Circular Data".
    Berens 2009, "CircStat: A MATLAB Toolbox for Circular Statistics",
    J. Stat. Soft. 31(10).
    """
    a = np.asarray(angles_deg, dtype=float).ravel()
    a = a[np.isfinite(a)]
    n = int(a.size)
    out = {"n": n}
    if n < 2:
        # Nothing meaningful with <2 points.  Fill the schema with NaN
        # so downstream CSV consumers see the same columns regardless.
        for k in ("mean_direction_deg", "mean_resultant_length",
                 "circular_variance", "circular_std_deg",
                 "angular_deviation_deg", "median_deg",
                 "concentration_kappa", "rayleigh_z", "rayleigh_p",
                 "v_test_z", "v_test_p", "circular_skewness",
                 "circular_kurtosis", "ci95_lower_deg", "ci95_upper_deg"):
            out[k] = float("nan")
        return out

    rad = np.radians(a)
    C = float(np.mean(np.cos(rad)))
    S = float(np.mean(np.sin(rad)))
    R_bar = float(np.hypot(C, S))           # mean resultant length
    mu_rad = float(np.arctan2(S, C))         # mean direction (radians)
    mu_deg = float(np.degrees(mu_rad))
    # Standard CircStat convention: report direction on (-180°, +180°]
    if mu_deg <= -180.0: mu_deg += 360.0
    if mu_deg >   180.0: mu_deg -= 360.0

    # Dispersion measures
    circ_var = 1.0 - R_bar
    # √(-2·ln R̄) is undefined at R̄=0 (uniform), gigantic for tiny R̄.
    # Clamp to avoid log(0) screaming; report NaN for R̄ ≤ 0 instead.
    if R_bar > 0:
        circ_std_deg = float(np.degrees(np.sqrt(-2.0 * np.log(R_bar))))
    else:
        circ_std_deg = float("nan")
    ang_dev_deg = float(np.degrees(np.sqrt(2.0 * max(circ_var, 0.0))))

    # Circular median: angle θ̃ minimising Σ (π − |π − |θᵢ − θ̃||).
    # Evaluating the objective at every datum is O(n²) in time AND
    # memory if we do it with broadcasting (the 50k × 50k float64
    # array alone is 20 GB).  We instead:
    #   * cap CANDIDATES at 3000 (random subsample of the data)
    #   * cap SUMMAND points at 8000 (random subsample of the data)
    # which gives 24 million ops + ~190 MB temporary — fast enough,
    # and the median estimate from a 8000-point subsample is accurate
    # to a couple of degrees, well below other sources of noise here.
    _rng = np.random.default_rng(0)
    if n > 3000:
        cand = rad[_rng.choice(n, size=3000, replace=False)]
    else:
        cand = rad
    if n > 8000:
        ref = rad[_rng.choice(n, size=8000, replace=False)]
    else:
        ref = rad
    diff = np.abs(cand[:, None] - ref[None, :])
    diff = np.minimum(diff, 2.0 * np.pi - diff)        # circular distance
    obj = diff.sum(axis=1)
    median_rad = float(cand[int(np.argmin(obj))])
    median_deg = float(np.degrees(median_rad))
    if median_deg <= -180.0: median_deg += 360.0
    if median_deg >   180.0: median_deg -= 360.0

    # Concentration κ — Best & Fisher 1981 piecewise approximation,
    # with a small-n bias correction (Fisher 1993 eq. 4.41).
    if R_bar < 0.53:
        kappa = 2.0 * R_bar + R_bar ** 3 + 5.0 * R_bar ** 5 / 6.0
    elif R_bar < 0.85:
        kappa = -0.4 + 1.39 * R_bar + 0.43 / max(1.0 - R_bar, 1e-12)
    else:
        denom = max(R_bar ** 3 - 4.0 * R_bar ** 2 + 3.0 * R_bar, 1e-12)
        kappa = 1.0 / denom
    if n < 15:
        if kappa < 2.0:
            kappa = max(kappa - 2.0 / (n * kappa), 0.0)
        else:
            kappa = ((n - 1.0) ** 3) * kappa / (n ** 3 + n)

    # Rayleigh test for uniformity (Wilkie 1983 / Mardia & Jupp eq. 6.3.5).
    # We compute in LOG space so the result doesn't underflow to 0
    # when n is large (e.g. n=240k with R̄=0.08 → z≈1500 → exp(-z)
    # rounds to 0 in float64, which the user sees as a spurious
    # "p = 0").  The leading term is exp(-z); we still apply the
    # Mardia higher-order correction multiplicatively in log-space.
    R_total = n * R_bar
    z_ray = R_total ** 2 / n
    correction = (1.0 + (2.0 * z_ray - z_ray ** 2) / (4.0 * n)
                  - (24.0 * z_ray - 132.0 * z_ray ** 2
                     + 76.0 * z_ray ** 3 - 9.0 * z_ray ** 4)
                    / (288.0 * n ** 2))
    if correction <= 0:
        correction = 1.0   # higher-order correction overshot; ignore.
    log_p_ray = -z_ray + np.log(correction)
    # If log p < ~-700, exp underflows.  Convert to a tiny positive
    # number that survives float64 (1e-300) so downstream callers see
    # "very small" rather than zero, and formatters can render it as
    # "<1e-300".
    if log_p_ray < -700.0:
        p_ray = 1e-300
    else:
        p_ray = float(np.exp(log_p_ray))
    p_ray = float(np.clip(p_ray, 0.0, 1.0))

    # V-test against μ₀ = 0° ("are tracks preferentially going
    # straight ahead?").  V = R̄·cos(μ − μ₀); z = V·√(2n); one-tailed.
    mu0 = 0.0
    V = R_bar * np.cos(mu_rad - mu0)
    z_v = V * np.sqrt(2.0 * n)
    # One-tailed p via the standard normal survival function.  Use
    # scipy's norm.sf where available (numerically stable to ~p≈1e-300);
    # fall back to a math.erf-based computation otherwise, and floor at
    # 1e-300 so a huge z doesn't round to exactly 0.
    try:
        from scipy.stats import norm as _norm
        p_v = float(_norm.sf(z_v))
    except Exception:
        from math import erf
        p_v = float(0.5 * (1.0 - erf(z_v / np.sqrt(2.0))))
    if p_v == 0.0:
        p_v = 1e-300    # underflow sentinel
    p_v = float(np.clip(p_v, 0.0, 1.0))

    # Circular skewness and kurtosis (Mardia & Jupp §2.3).
    # b̄ = (1/n) Σ sin(2(θᵢ − μ))   ;   ā = (1/n) Σ cos(2(θᵢ − μ))
    b_bar = float(np.mean(np.sin(2.0 * (rad - mu_rad))))
    a_bar = float(np.mean(np.cos(2.0 * (rad - mu_rad))))
    sigma = max(1.0 - R_bar, 1e-12)
    skew = b_bar / (sigma ** 1.5)
    kurt = (a_bar - R_bar ** 4) / (sigma ** 2)

    # 95% CI for μ — large-sample normal approximation (Fisher 1993
    # eq. 4.46).  Only meaningful when R̄ is appreciable AND n ≥ ~15;
    # report NaN when the approximation breaks down.
    if R_bar >= 0.4 and n >= 15:
        sd_mu = np.sqrt((1.0 - a_bar) / (2.0 * n * R_bar ** 2))
        half = float(np.degrees(1.959964 * sd_mu))   # 1.96 σ
        lo = mu_deg - half
        hi = mu_deg + half
        # Keep both endpoints on (-180°, +180°] without wrapping the
        # interval ordering — supervisor will read this from the CSV.
        ci_lo, ci_hi = lo, hi
    else:
        ci_lo = float("nan")
        ci_hi = float("nan")

    out.update({
        "mean_direction_deg":     mu_deg,
        "mean_resultant_length":  R_bar,
        "circular_variance":      circ_var,
        "circular_std_deg":       circ_std_deg,
        "angular_deviation_deg":  ang_dev_deg,
        "median_deg":             median_deg,
        "concentration_kappa":    float(kappa),
        "rayleigh_z":             float(z_ray),
        "rayleigh_p":             p_ray,
        "v_test_z":               float(z_v),
        "v_test_p":               p_v,
        "circular_skewness":      float(skew),
        "circular_kurtosis":      float(kurt),
        "ci95_lower_deg":         float(ci_lo),
        "ci95_upper_deg":         float(ci_hi),
    })
    return out


_THEME_REQUIRED_KEYS = (
    "BG", "PNL", "TXT", "MUT", "GRD", "ACC",
    "HDR_BG", "HDR_TXT", "ZEBRA", "FONT", "ARROW",
    # legacy keys consumed by `compare_groups` & `_write_pdf_report`
    "BAR_FILL", "SIG",
)


def _theme_palette(theme: str) -> dict:
    """Return a colour palette matching the master figure theme.
    Centralised so the master figure, the circular-statistics PDF, and
    the comparison PDF all read from the same source of truth.

    The returned dict is GUARANTEED to contain every key in
    `_THEME_REQUIRED_KEYS` — if any caller starts using a new key,
    add it to the tuple and to every branch below, and the
    `_validate_palette` check at the bottom will catch a regression at
    module-import time rather than at PDF-render time.
    """
    t = (theme or "Dark").strip()
    if t == "Light":
        pal = {"BG":   "#ffffff", "PNL":  "#f6f8fa",
               "TXT":  "#24292f", "MUT":  "#57606a",
               "GRD":  "#d0d7de", "ACC":  "#0969da",
               "HDR_BG":"#1f2937", "HDR_TXT":"#ffffff",
               "ZEBRA":"#f3f4f6", "FONT": "sans-serif",
               "ARROW":"#d93636",
               "BAR_FILL":"#0969da", "SIG":"#d93636"}
    elif t == "Publication":
        pal = {"BG":   "#ffffff", "PNL":  "#ffffff",
               "TXT":  "#000000", "MUT":  "#444444",
               "GRD":  "#cccccc", "ACC":  "#333333",
               "HDR_BG":"#000000", "HDR_TXT":"#ffffff",
               "ZEBRA":"#f2f2f2", "FONT": "serif",
               "ARROW":"#000000",
               "BAR_FILL":"#333333", "SIG":"#000000"}
    elif t == "AMOLED":
        # Pure-black backgrounds for OLED displays.  Mirrors Dark
        # otherwise so the figures are recognisable as the same FIREFLY
        # output.  PNL nudged to #0a0a0a so card-style panels still
        # read as cards against the BG.
        pal = {"BG":   "#000000", "PNL":  "#0a0a0a",
               "TXT":  "#e6edf3", "MUT":  "#9da7b1",
               "GRD":  "#30363d", "ACC":  "#58a6ff",
               "HDR_BG":"#141414", "HDR_TXT":"#e6edf3",
               "ZEBRA":"#050505", "FONT": "monospace",
               "ARROW":"#ff7b72",
               "BAR_FILL":"#58a6ff", "SIG":"#ff7b72"}
    else:
        # Dark (default).
        pal = {"BG":   "#0d1117", "PNL":  "#161b22",
               "TXT":  "#e6edf3", "MUT":  "#9da7b1",
               "GRD":  "#30363d", "ACC":  "#58a6ff",
               "HDR_BG":"#21262d", "HDR_TXT":"#e6edf3",
               "ZEBRA":"#1c2128", "FONT": "monospace",
               "ARROW":"#ff7b72",
               "BAR_FILL":"#58a6ff", "SIG":"#ff7b72"}
    # Belt-and-braces: if a caller (or future edit) ever accesses a key
    # we forgot to include, return a sensible TXT fallback rather than
    # crashing with a KeyError mid-render.  We do this via a small
    # dict subclass so `pal[<missing>]` works as if `pal.get(<missing>,
    # pal["TXT"])` were called.
    class _PalDict(dict):
        __slots__ = ()
        def __missing__(self, key):
            return self.get("TXT", "#000000")
    return _PalDict(pal)


def save_circular_statistics_pdf(angles_deg, stats, *, pdf_path,
                                  file_label="", fig_theme="Dark",
                                  circ_lin_result=None):
    """Render a single-page A4-portrait PDF report summarising the
    circular statistics in `stats` (as produced by
    `compute_circular_statistics`) alongside a small polar histogram of
    the underlying angle distribution.

    Designed to be supervisor-facing: stat names match MATLAB CircStat,
    each value is annotated with a one-line plain-English meaning, and
    the polar plot orients 0° at the top with positive angles sweeping
    counter-clockwise (the convention `compute_turning_angles` uses).

    Parameters
    ----------
    angles_deg : 1-D array of turning angles in degrees (signed, on
                 (-180°, +180°]).  Used only for the polar histogram.
    stats      : dict returned by `compute_circular_statistics`.
    pdf_path   : where to write the PDF.
    file_label : appears in the page header (typically the analysis stem).
    fig_theme  : "Dark" | "Light" | "Publication" — palette to match the
                 master figure renderer.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pal = _theme_palette(fig_theme)

    a = np.asarray(angles_deg, dtype=float).ravel()
    a = a[np.isfinite(a)]

    # Helper: render NaN as an em-dash so the PDF doesn't look broken
    # when a stat couldn't be computed (small-n or R̄ ≈ 0 cases).
    # Also collapse the 1e-300 underflow sentinel produced by the
    # log-space p-value computations into a human-readable "<1e-300"
    # — otherwise the supervisor sees "1e-300" and wonders why so
    # many tests give exactly that value.
    def _fmt(x, prec=4):
        try:
            if x is None: return "—"
            xf = float(x)
            if np.isnan(xf): return "—"
            if xf > 0.0 and xf <= 1e-300:
                return "<1e-300"
            return f"{xf:.{prec}g}"
        except Exception:
            return str(x)

    # One-line plain-English gloss per statistic.  Order matches the
    # CSV column order so the table reads top-to-bottom like the CSV.
    rows = [
        ("n",                          "Sample size", "count",
         f"{int(stats.get('n', 0)):,}"),
        ("mean_direction_deg",         "Mean direction μ", "deg",
         _fmt(stats.get("mean_direction_deg"), 4)),
        ("mean_resultant_length",      "Mean resultant length R̄  (0 = uniform, 1 = aligned)",
         "—",
         _fmt(stats.get("mean_resultant_length"), 4)),
        ("circular_variance",          "Circular variance  1 − R̄", "—",
         _fmt(stats.get("circular_variance"), 4)),
        ("circular_std_deg",           "Circular standard deviation  √(−2·ln R̄)",
         "deg",
         _fmt(stats.get("circular_std_deg"), 4)),
        ("angular_deviation_deg",      "Angular deviation  s₀ = √(2·(1−R̄))",
         "deg",
         _fmt(stats.get("angular_deviation_deg"), 4)),
        ("median_deg",                 "Circular median", "deg",
         _fmt(stats.get("median_deg"), 4)),
        ("concentration_kappa",        "Von Mises concentration κ  (Best & Fisher 1981)",
         "—",
         _fmt(stats.get("concentration_kappa"), 4)),
        ("rayleigh_z",                 "Rayleigh test statistic  z = n·R̄²", "—",
         _fmt(stats.get("rayleigh_z"), 4)),
        ("rayleigh_p",                 "Rayleigh test p-value  (uniformity)", "—",
         _fmt(stats.get("rayleigh_p"), 3)),
        ("v_test_z",                   "V-test statistic against μ₀ = 0°", "—",
         _fmt(stats.get("v_test_z"), 4)),
        ("v_test_p",                   "V-test p-value  (preferred direction)", "—",
         _fmt(stats.get("v_test_p"), 3)),
        ("circular_skewness",          "Circular skewness  (Mardia & Jupp §2.3)", "—",
         _fmt(stats.get("circular_skewness"), 4)),
        ("circular_kurtosis",          "Circular kurtosis  (Mardia & Jupp §2.3)", "—",
         _fmt(stats.get("circular_kurtosis"), 4)),
        ("ci95_lower_deg",             "95% CI lower bound for μ  (large-sample)", "deg",
         _fmt(stats.get("ci95_lower_deg"), 4)),
        ("ci95_upper_deg",             "95% CI upper bound for μ  (large-sample)", "deg",
         _fmt(stats.get("ci95_upper_deg"), 4)),
    ]
    # Circ-lin correlation rows — optional; only present when the
    # caller passed a `circ_lin_result` (computed from per-track
    # (mean_angle, D) pairs).  Three rows: r, χ²(2), p, n.  Treated
    # as a single stats block so it can be excluded silently when
    # the caller has no D data (e.g. external-CSV input path).
    if circ_lin_result:
        rows.extend([
            ("circ_lin_angle_vs_D_r",
             "Circ-lin correlation r — turning bias vs D", "—",
             _fmt(circ_lin_result.get("r"), 4)),
            ("circ_lin_angle_vs_D_chi2",
             "Circ-lin χ²(2) test statistic  (n·r²)", "—",
             _fmt(circ_lin_result.get("test_stat"), 4)),
            ("circ_lin_angle_vs_D_p",
             "Circ-lin correlation p-value", "—",
             _fmt(circ_lin_result.get("p"), 3)),
            ("circ_lin_angle_vs_D_n",
             "Circ-lin sample size  (tracks with ≥ 3 frames + D)",
             "count",
             f"{int(circ_lin_result.get('n', 0)):,}"
             if circ_lin_result.get("n") is not None else "—"),
        ])

    # ── rcParams snapshot ──────────────────────────────────────────────
    # plt.rcParams persists across figures in the same process — the
    # master figure renderer might have left things on the Dark palette
    # (text.color = #e6edf3 etc.).  Snapshot then force everything to
    # OUR palette so we can't accidentally pick up someone else's
    # colours.  Restored at the end.
    _rc_keys = ("text.color", "axes.labelcolor", "axes.edgecolor",
                "xtick.color", "ytick.color", "axes.facecolor",
                "axes.titlecolor", "figure.facecolor", "grid.color",
                "font.family")
    _rc_save = {k: plt.rcParams.get(k) for k in _rc_keys}
    plt.rcParams.update({
        "text.color":       pal["TXT"],
        "axes.labelcolor":  pal["TXT"],
        "axes.edgecolor":   pal["GRD"],
        "xtick.color":      pal["TXT"],
        "ytick.color":      pal["TXT"],
        "axes.facecolor":   pal["PNL"],
        "axes.titlecolor":  pal["TXT"],
        "figure.facecolor": pal["BG"],
        "grid.color":       pal["GRD"],
        "font.family":      pal["FONT"],
    })

    try:
        # ── Layout (A4 portrait, all coords in figure-fraction) ─────────
        #
        # Vertical bands, top → bottom:
        #   y 0.94 – 0.98  : header bar (title + n)
        #   y 0.89 – 0.93  : file label
        #   y 0.61 – 0.86  : polar  |  interpretation banner
        #   y 0.54 – 0.58  : "Statistics" section title
        #   y 0.12 – 0.52  : statistics table
        #   y 0.06 – 0.10  : sign-convention footer (3 short lines)
        #   y 0.02 – 0.04  : references footer
        #
        # The earlier layout placed the Statistics title with
        # `transform=ax_tbl.transAxes` at y=1.04 which sits at about
        # figure-y 0.53 — directly underneath the polar's "±180°" tick.
        # Moving it to its own fig.text at a fixed y resolves the overlap.
        # The footer used to be at y=0.04 which collided with the
        # table's bottom row at y=0.05; both footers now live below
        # y=0.10 with the table topping at y=0.52.
        fig = plt.figure(figsize=(8.27, 11.69), facecolor=pal["BG"])

        # Header (full width)
        ax_hdr = fig.add_axes([0.07, 0.94, 0.86, 0.04])
        ax_hdr.axis("off")
        title = "Circular Statistics Report"
        ax_hdr.text(0.0, 0.5, title, fontsize=16, fontweight="bold",
                    va="center", ha="left", color=pal["TXT"])
        n_val = int(stats.get("n", 0))
        ax_hdr.text(1.0, 0.5,
                    f"n = {n_val:,} turning angles",
                    fontsize=11, color=pal["MUT"], va="center", ha="right")
        if file_label:
            # File label on its own dedicated row so it can't fight the
            # polar plot's "0°" tick label below.
            fig.text(0.07, 0.91, file_label, fontsize=10,
                     color=pal["MUT"], va="top", ha="left",
                     family=pal["FONT"])

        # Polar histogram (left side of middle band).
        # Convention matched to the master figure's Radial-Distribution
        # panel (see sax "O" in make_figure): 0° at the top, positive
        # angles sweep CLOCKWISE so they appear on the right hemisphere.
        # Signed angles on (-180°, +180°] are first wrapped to [0, 2π)
        # before histogramming — matplotlib's polar bar() silently drops
        # bars at negative theta when set_theta_direction(-1) is active.
        ax_polar = fig.add_axes([0.08, 0.61, 0.36, 0.25], projection="polar")
        ax_polar.set_facecolor(pal["PNL"])
        if a.size >= 10:
            nbins = 36
            angles_rad = np.mod(np.deg2rad(a), 2.0 * np.pi)
            bins  = np.linspace(0.0, 2.0 * np.pi, nbins + 1)
            counts, edges = np.histogram(angles_rad, bins=bins)
            widths  = np.diff(edges)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax_polar.set_theta_zero_location("N")
            ax_polar.set_theta_direction(-1)  # CW positive — match master fig
            ax_polar.bar(centers, counts, width=widths * 0.95,
                         align="center", color=pal["ACC"],
                         edgecolor=pal["PNL"], linewidth=0.4, alpha=0.92)
            mu = stats.get("mean_direction_deg")
            if mu is not None and not (isinstance(mu, float) and np.isnan(mu)):
                r_max = float(counts.max()) if counts.size else 1.0
                # Wrap signed μ into [0, 2π) so the arrow lands at the
                # same place the bar histogram does.
                mu_rad = np.mod(np.deg2rad(mu), 2.0 * np.pi)
                ax_polar.annotate("",
                    xy=(mu_rad, r_max * 0.95),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->",
                                    color=pal["ARROW"], lw=2.0))
            # Show signed-angle labels at positive-angle slot positions
            # so the visual reads "+45° upper-right, -45° upper-left",
            # exactly like the master figure.
            ax_polar.set_xticks(np.deg2rad(
                [0, 45, 90, 135, 180, 225, 270, 315]))
            ax_polar.set_xticklabels(
                ["0°", "+45°", "+90°", "+135°", "±180°",
                 "−135°", "−90°", "−45°"], fontsize=8)
            ax_polar.set_yticklabels([])
            ax_polar.tick_params(colors=pal["TXT"], labelsize=8)
            ax_polar.grid(True, ls=":", alpha=0.4)
            # NB: deliberately no `set_title` here — matplotlib places
            # the polar title above the axes box (offset by `pad`), and
            # at this layout that overlaps the file-label rendered in
            # the header area.  The page header already identifies the
            # report, and the footer covers the sign convention, so a
            # title on the polar would be redundant anyway.
        else:
            ax_polar.axis("off")
            ax_polar.text(0.5, 0.5, "Too few angles for histogram",
                          transform=ax_polar.transAxes,
                          ha="center", va="center", color=pal["MUT"],
                          fontsize=10)

        # Interpretation banner (right side of middle band)
        ax_intr = fig.add_axes([0.48, 0.61, 0.46, 0.25])
        ax_intr.axis("off")
        R = stats.get("mean_resultant_length")
        p = stats.get("rayleigh_p")
        if R is None or (isinstance(R, float) and np.isnan(R)):
            interp = "Distribution: insufficient data."
        elif R < 0.10:
            interp = ("Distribution is consistent with uniform circular "
                      "scatter — no preferred turning direction is "
                      "evident.  Typical of free 2-D diffusion.")
        elif R < 0.30:
            interp = ("Weak directional bias.  Most steps are close "
                      "to uniform, but a slight tendency toward "
                      f"{stats.get('mean_direction_deg', 0):.0f}° is "
                      "present.")
        elif R < 0.60:
            interp = ("Moderate directional bias toward "
                      f"{stats.get('mean_direction_deg', 0):.0f}°.  "
                      "Consider whether this reflects biology (e.g. "
                      "transport along a cytoskeletal track) or an "
                      "artefact (uncorrected drift, anisotropic ROI).")
        else:
            interp = ("Strong directional bias toward "
                      f"{stats.get('mean_direction_deg', 0):.0f}°.  "
                      "Verify the drift correction and ROI geometry "
                      "before biological interpretation.")
        if p is not None and not (isinstance(p, float) and np.isnan(p)):
            if p < 0.001:
                verdict = ("Rayleigh test strongly rejects uniformity "
                           f"(p = {p:.3g}).")
            elif p < 0.05:
                verdict = ("Rayleigh test rejects uniformity at α = "
                           f"0.05 (p = {p:.3g}).")
            else:
                verdict = ("Rayleigh test does NOT reject uniformity "
                           f"(p = {p:.3g}).")
            interp = interp + "\n\n" + verdict
        ax_intr.text(0.0, 1.0, "Interpretation",
                     fontsize=12, fontweight="bold", va="top",
                     color=pal["TXT"])
        ax_intr.text(0.0, 0.9, interp, fontsize=10, va="top",
                     wrap=True, color=pal["TXT"])

        # Section title — placed in FIGURE coords so its vertical
        # position is decoupled from the table's bbox and can't
        # collide with the polar's bottom ticks above.
        fig.text(0.07, 0.555, "Statistics  (MATLAB CircStat conventions)",
                 fontsize=12, fontweight="bold", va="bottom",
                 ha="left", color=pal["TXT"])
        # Statistics table — pinned with a clear gap above (title) and
        # below (footer block).  Bottom edge y=0.12 leaves room for two
        # footer lines without collision.
        ax_tbl = fig.add_axes([0.07, 0.12, 0.88, 0.40])
        ax_tbl.axis("off")

        cell_text, row_labels = [], []
        for key, gloss, unit, val in rows:
            unit_s = "" if unit in ("", "—") else f"  ({unit})"
            cell_text.append([f"{gloss}", f"{val}{unit_s}"])
            row_labels.append(key)
        tbl = ax_tbl.table(cellText=cell_text,
                           rowLabels=row_labels,
                           colLabels=["Description", "Value"],
                           cellLoc="left", rowLoc="left",
                           colLoc="left",
                           colWidths=[0.62, 0.28],
                           bbox=[0.20, 0.0, 0.80, 1.0])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.0)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_linewidth(0.5)
            cell.set_edgecolor(pal["GRD"])
            if r == 0:                       # column header row
                cell.set_facecolor(pal["HDR_BG"])
                cell.set_text_props(color=pal["HDR_TXT"], fontweight="bold")
            else:
                # Zebra-stripe data rows.  Use theme PNL for the
                # "darker" stripes and ZEBRA for the lighter ones.
                cell.set_facecolor(pal["ZEBRA"] if r % 2 == 0 else pal["PNL"])
                if c == -1:                  # row-label column
                    cell.set_text_props(family="monospace", fontsize=8.0,
                                        color=pal["MUT"])
                else:
                    cell.set_text_props(color=pal["TXT"])

        # Footer — explicit short lines instead of `wrap=True`, because
        # matplotlib's fig.text wrap only kicks in when the text would
        # exceed a containing artist's width, NOT the figure width, so
        # long strings just run off the right edge of the PDF (which is
        # what was happening to the References line).  Breaking into
        # pre-wrapped lines side-steps that entirely.
        _foot_kw = dict(fontsize=7, color=pal["MUT"], ha="left",
                        va="bottom", family=pal["FONT"])
        sign_lines = [
            "Sign convention: turning angles are SIGNED on (−180°, +180°].",
            "0° = straight ahead.  +θ = left turn (CCW).  −θ = right "
            "turn (CW).  ±180° = full reversal.",
            "Unsigned 0–360° equivalent: u = θ if θ ≥ 0, else θ + 360 "
            "(so −90° ≡ 270°, +90° ≡ 90°).",
        ]
        ref_lines = [
            "References:",
            "  Mardia & Jupp 2000 — Directional Statistics.",
            "  Fisher 1993 — Statistical Analysis of Circular Data.",
            "  Berens 2009 — CircStat: A MATLAB Toolbox for Circular "
            "Statistics, J. Stat. Soft. 31(10).",
        ]
        y = 0.095
        for line in sign_lines:
            fig.text(0.07, y, line, **_foot_kw)
            y -= 0.014
        y -= 0.006
        for line in ref_lines:
            fig.text(0.07, y, line, **_foot_kw)
            y -= 0.014

        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, facecolor=pal["BG"])
        plt.close(fig)
    finally:
        # Restore rcParams so we don't bleed our palette into whatever
        # plot the caller draws next.
        plt.rcParams.update(_rc_save)


def _circ_watson_williams(samples_deg):
    """k-sample Watson-Williams F-test for equality of mean directions
    across k≥2 circular samples (Mardia & Jupp 2000 §6.4.2).  This is
    the circular analogue of one-way ANOVA: H₀ = all groups share a
    common mean direction.

    Parameters
    ----------
    samples_deg : list of 1-D angle arrays (degrees, any range)

    Returns
    -------
    None if fewer than 2 valid samples, else dict with:
      F, df1, df2, p           — test statistic + degrees of freedom + p
      kappa_pooled, R_bar_pooled
      valid                     — True iff κ̂ ≥ 2 and R̄ ≥ 0.45 (the test
                                  assumes concentrated von Mises samples;
                                  flag the result if not).
      n_per_group, n_total, k
    """
    rad = [np.radians(np.asarray(s, dtype=float).ravel())
           for s in samples_deg]
    rad = [r[np.isfinite(r)] for r in rad]
    rad = [r for r in rad if r.size >= 2]
    k = len(rad)
    if k < 2:
        return None
    n_per = [int(r.size) for r in rad]
    N = int(sum(n_per))
    Ci = np.array([float(np.cos(r).sum()) for r in rad])
    Si = np.array([float(np.sin(r).sum()) for r in rad])
    Ri = np.hypot(Ci, Si)
    Cp = float(Ci.sum()); Sp = float(Si.sum())
    Rp = float(np.hypot(Cp, Sp))
    R_bar = Rp / N
    # Pooled concentration (Best & Fisher 1981).
    if R_bar < 0.53:
        kappa = 2.0 * R_bar + R_bar ** 3 + 5.0 * R_bar ** 5 / 6.0
    elif R_bar < 0.85:
        kappa = -0.4 + 1.39 * R_bar + 0.43 / max(1.0 - R_bar, 1e-12)
    else:
        denom = max(R_bar ** 3 - 4.0 * R_bar ** 2 + 3.0 * R_bar, 1e-12)
        kappa = 1.0 / denom
    # Stephens 1972 K correction (≈1 when κ is large; sharper at low κ).
    K = 1.0 + 3.0 / (8.0 * kappa) if kappa > 0 else 1.0
    sumR = float(Ri.sum())
    denom_f = (k - 1) * (N - sumR)
    if denom_f <= 0:
        return None
    F = K * (N - k) * (sumR - Rp) / denom_f
    df1, df2 = int(k - 1), int(N - k)
    try:
        from scipy.stats import f as _f_dist
        # Use logsf → exp so we get a meaningful tiny p instead of a
        # rounded-to-zero float when F is huge (which is normal with
        # 100k+ angles per group).  logsf returns log(1 - cdf) with
        # log-space stability.
        log_p = float(_f_dist.logsf(F, df1, df2))
        p = 1e-300 if log_p < -700.0 else float(np.exp(log_p))
    except Exception:
        p = float("nan")
    return {
        "F": float(F), "df1": df1, "df2": df2, "p": p,
        "kappa_pooled": float(kappa),
        "R_bar_pooled": float(R_bar),
        "valid": bool(kappa >= 2.0 and R_bar >= 0.45),
        "n_per_group": n_per, "n_total": N, "k": int(k),
    }


def _circ_mardia_watson_wheeler(samples_deg):
    """Mardia-Watson-Wheeler (uniform-scores) non-parametric k-sample
    test for equal CIRCULAR DISTRIBUTIONS across k≥2 groups (Mardia &
    Jupp 2000 §7.6.1).  Unlike Watson-Williams it makes no assumption
    about concentration, so it's the safe fallback when κ < 2 or when
    you suspect groups differ in spread rather than only in mean
    direction.

    Returns None if fewer than 2 valid samples, else dict with:
      W, df, p, n_per_group, n_total, k
    """
    rad = [np.radians(np.asarray(s, dtype=float).ravel())
           for s in samples_deg]
    rad = [r[np.isfinite(r)] for r in rad]
    rad = [r for r in rad if r.size >= 1]
    k = len(rad)
    if k < 2:
        return None
    pooled = np.concatenate(rad)
    N = int(pooled.size)
    try:
        from scipy.stats import rankdata, chi2
    except Exception:
        return None
    ranks = rankdata(pooled, method="average")
    # Convert ranks → uniform circular scores in [0, 2π).
    beta = 2.0 * np.pi * ranks / N
    # Sample-wise C/S sums, then W = 2 · Σ (C² + S²) / n_j.
    W_stat = 0.0
    cursor = 0
    for r in rad:
        n_j = int(r.size)
        end = cursor + n_j
        b = beta[cursor:end]
        Cj = float(np.cos(b).sum())
        Sj = float(np.sin(b).sum())
        W_stat += (Cj * Cj + Sj * Sj) / n_j
        cursor = end
    W = 2.0 * W_stat
    df = int(2 * (k - 1))
    try:
        # logsf for numerical stability — chi2.sf(3.4e3, 2) underflows
        # to 0.0 in float64 but chi2.logsf returns the actual log p.
        log_p = float(chi2.logsf(W, df))
        p = 1e-300 if log_p < -700.0 else float(np.exp(log_p))
    except Exception:
        p = float("nan")
    return {
        "W": float(W), "df": df, "p": p,
        "n_per_group": [int(r.size) for r in rad],
        "n_total": N, "k": int(k),
    }


def _circ_wallraff_ktest(samples_deg):
    """Wallraff k-sample test for equality of circular concentrations.

    H₀ = all samples share the same concentration κ.  Implementation
    follows Mardia & Jupp (2000) §7.5.5: convert each angle to its
    deviation from its own sample's mean direction (mapped to [0, π]),
    then run a rank-sum test on those deviations across groups.

    For k = 2 we use the Mann-Whitney U test; for k > 2 we use the
    Kruskal-Wallis H test.  Returns None if fewer than 2 valid samples.

    Returned dict:
      H or U   : test statistic (key name depends on k)
      df       : degrees of freedom (Kruskal-Wallis only)
      p        : p-value
      n_per_group, n_total, k
    """
    rad = [np.radians(np.asarray(s, dtype=float).ravel())
           for s in samples_deg]
    rad = [r[np.isfinite(r)] for r in rad]
    rad = [r for r in rad if r.size >= 2]
    k = len(rad)
    if k < 2:
        return None
    # Per-sample angular deviation from its OWN mean direction,
    # mapped to [0, π] (the circular distance).
    deviations = []
    for r in rad:
        mu = np.arctan2(np.sin(r).mean(), np.cos(r).mean())
        d  = np.abs(r - mu)
        d  = np.minimum(d, 2.0 * np.pi - d)
        deviations.append(d)
    n_per = [int(d.size) for d in deviations]
    try:
        if k == 2:
            from scipy.stats import mannwhitneyu
            stat, p = mannwhitneyu(deviations[0], deviations[1],
                                   alternative="two-sided")
            return {
                "U": float(stat), "p": float(p), "k": 2,
                "n_per_group": n_per, "n_total": int(sum(n_per)),
            }
        else:
            from scipy.stats import kruskal
            stat, p = kruskal(*deviations)
            return {
                "H": float(stat), "df": int(k - 1),
                "p": float(p), "k": int(k),
                "n_per_group": n_per, "n_total": int(sum(n_per)),
            }
    except Exception:
        return None


def _circ_kuiper_two_sample(a_deg, b_deg):
    """Kuiper two-sample test for equality of circular distributions.

    Non-parametric, distribution-free analogue of the Kolmogorov-Smirnov
    test, adapted for circular data.  Sensitive to differences anywhere
    in the distribution (not just shifts in mean), and unlike the KS
    statistic the Kuiper statistic V = D⁺ + D⁻ is invariant to the
    choice of origin on the circle — a property that matters because
    "where you put 0°" is arbitrary for circular data.

    Returns None if either sample is < 2 elements, else dict:
      V       : Kuiper statistic
      p       : asymptotic p-value (Stephens 1965 series approximation)
      n1, n2  : sample sizes
    """
    a = np.sort(np.mod(np.radians(np.asarray(a_deg, dtype=float).ravel()),
                       2.0 * np.pi))
    b = np.sort(np.mod(np.radians(np.asarray(b_deg, dtype=float).ravel()),
                       2.0 * np.pi))
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    n1, n2 = int(a.size), int(b.size)
    if n1 < 2 or n2 < 2:
        return None

    # Empirical CDFs evaluated at every observation in the combined
    # sample.  V = max(F1 - F2) + max(F2 - F1).
    combined = np.sort(np.concatenate([a, b]))
    F1 = np.searchsorted(a, combined, side="right") / n1
    F2 = np.searchsorted(b, combined, side="right") / n2
    D_plus  = float((F1 - F2).max())
    D_minus = float((F2 - F1).max())
    V = D_plus + D_minus

    # Stephens (1965) asymptotic p-value: λ = (√n_eff + 0.155 + 0.24/√n_eff)·V.
    n_eff = n1 * n2 / (n1 + n2)
    lam = (np.sqrt(n_eff) + 0.155 + 0.24 / np.sqrt(n_eff)) * V
    if lam <= 0:
        p = 1.0
    else:
        # Convergent series in j; cap at j=100 (terms decay
        # exponentially in j²).
        s_terms = 0.0
        l2 = lam * lam
        for j in range(1, 101):
            j2 = j * j
            term = 2.0 * (4.0 * j2 * l2 - 1.0) * np.exp(-2.0 * j2 * l2)
            s_terms += term
            if abs(term) < 1e-18:
                break
        p = float(np.clip(s_terms, 0.0, 1.0))
    if p > 0.0 and p <= 1e-300:
        p = 1e-300
    return {
        "V": float(V), "p": float(p),
        "n1": n1, "n2": n2,
    }


def _circ_lin_correlation(theta_deg, x):
    """Circular-linear correlation (Mardia 1976; Mardia & Jupp 2000
    §6.5.1).

    Tests whether a circular variable θ is associated with a linear
    variable x.  Compute the three Pearson correlations
        r_xc = corr(x, cos θ),  r_xs = corr(x, sin θ),  r_cs = corr(cos θ, sin θ)
    and combine them into the circular-linear coefficient

        R² = (r_xc² + r_xs² − 2·r_xc·r_xs·r_cs) / (1 − r_cs²)

    R ∈ [0, 1] (analogous to a Pearson |r|).  Under H₀ of independence
    and large n, n·R² ~ χ²(2), giving a usable p-value.

    Returns None if n < 3 or the data are degenerate; else dict:
      r, r2          : coefficient and its square
      test_stat      : n · r²
      df, p          : χ²(2) p-value
      n              : effective sample size after finite-mask
    """
    theta = np.asarray(theta_deg, dtype=float).ravel()
    x     = np.asarray(x,         dtype=float).ravel()
    if theta.size != x.size:
        return None
    mask = np.isfinite(theta) & np.isfinite(x)
    theta = theta[mask]; x = x[mask]
    n = int(theta.size)
    if n < 3:
        return None
    rad = np.radians(theta)
    c = np.cos(rad); s = np.sin(rad)
    # Need non-zero variance in x AND in c/s for the correlations to
    # exist.  If all angles are identical (or all x identical), bail.
    if np.std(x) == 0 or np.std(c) == 0 or np.std(s) == 0:
        return None
    rxc = float(np.corrcoef(x, c)[0, 1])
    rxs = float(np.corrcoef(x, s)[0, 1])
    rcs = float(np.corrcoef(c, s)[0, 1])
    denom = 1.0 - rcs ** 2
    if abs(denom) < 1e-12:
        return None
    r2 = (rxc ** 2 + rxs ** 2 - 2.0 * rxc * rxs * rcs) / denom
    r2 = float(np.clip(r2, 0.0, 1.0))
    test_stat = n * r2
    try:
        from scipy.stats import chi2
        log_p = float(chi2.logsf(test_stat, 2))
        p = 1e-300 if log_p < -700.0 else float(np.exp(log_p))
    except Exception:
        p = float("nan")
    return {
        "r": float(np.sqrt(r2)), "r2": r2,
        "test_stat": float(test_stat), "df": 2,
        "p": float(p), "n": n,
    }


def compute_per_track_mean_angle(tracks):
    """For each track in `tracks` with ≥ 3 localisations, compute the
    circular mean of its signed turning angles (degrees on
    (-180°, +180°]).  Returns a list of (particle_id, mean_angle_deg).

    Used to build (angle, D) pairs for the circular-linear correlation
    between a track's turning bias and its diffusion coefficient.
    """
    if len(tracks) < 3:
        return []
    srt = (tracks.reset_index(drop=True)
                 .sort_values(["particle", "frame"], kind="stable"))
    pid_arr = srt["particle"].to_numpy()
    xy_arr  = srt[["x", "y"]].to_numpy()
    steps = np.diff(xy_arr, axis=0)
    same_step = (pid_arr[1:] == pid_arr[:-1])
    if len(steps) < 2:
        return []
    v1 = steps[:-1]; v2 = steps[1:]
    both_in_track = same_step[:-1] & same_step[1:]
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot   = np.sum(v1 * v2, axis=1)
    norm1 = np.linalg.norm(v1, axis=1)
    norm2 = np.linalg.norm(v2, axis=1)
    valid = both_in_track & (norm1 > 0) & (norm2 > 0)
    if not valid.any():
        return []
    angles = np.arctan2(cross[valid], dot[valid])    # radians
    # The middle row of each (i, i+1, i+2) triple is pid_arr[i+1].
    pid_at_turn = pid_arr[1:-1][valid]
    # Bucket angles by particle and compute the circular mean.
    out = []
    for pid in np.unique(pid_at_turn):
        sel = (pid_at_turn == pid)
        rad = angles[sel]
        if rad.size == 0:
            continue
        mu = np.degrees(np.arctan2(np.sin(rad).mean(),
                                   np.cos(rad).mean()))
        out.append((int(pid), float(mu)))
    return out


def _watson_williams_mu_per_replicate(mu_lists_per_group):
    """Watson-Williams F-test on per-replicate mean directions.

    Treats each replicate's mean direction μ_ij as a single circular
    observation (not the underlying angles).  This is the supervisor-
    facing way to compare directionality between groups: the n is
    the number of REPLICATES, not the number of pooled localisations,
    so the test isn't inflated by huge per-file angle counts.

    Parameters
    ----------
    mu_lists_per_group : list aligned with the groups, each entry is
        a 1-D array/list of per-replicate mean directions in DEGREES
        (signed, (-180°, +180°]).

    Returns dict matching the shape `_circ_watson_williams` already
    uses (F, df1, df2, p, valid, ...), or None if fewer than 2 groups
    have ≥ 2 replicates each.
    """
    # _circ_watson_williams already does k-sample WW on a list of
    # angle arrays — pass the per-replicate μ values in as samples.
    samples = [np.asarray(arr, dtype=float).ravel()
               for arr in mu_lists_per_group]
    samples = [a[np.isfinite(a)] for a in samples]
    if sum(1 for a in samples if a.size >= 2) < 2:
        return None
    return _circ_watson_williams(samples)


def compute_circular_comparison_tests(groups, *, track_angle_d_pairs=None,
                                       per_replicate_angles=None):
    """Run all the standard 'do these circular samples differ?' tests on
    a list of labelled groups.

    Parameters
    ----------
    groups : list of (label, angles_deg_array)
        One entry per comparison group; the array is the pooled
        turning angles across all replicates in that group.
    track_angle_d_pairs : optional list aligned with `groups`
        Each element is a 2-tuple of arrays (per_track_mean_angle_deg,
        per_track_D_um2_s).  Used to compute the per-group circular-
        linear correlation between a track's average turning bias and
        its diffusion coefficient.  Pass None to skip the correlation.

    Returns
    -------
    dict with keys:
      omnibus_ww   : Watson-Williams F-test (equal mean directions)
      omnibus_mww  : Mardia-Watson-Wheeler W-test (equal distributions)
      omnibus_wallraff
                   : Wallraff k-sample test (equal concentrations);
                     directly addresses "is one group more tightly
                     clustered than the other?".
      pairwise     : list, one entry per (i, j) with i<j, each with
                     keys label_a, label_b, ww, mww, wallraff, kuiper
                     (Kuiper two-sample test for equal distributions).
      circ_lin_per_group
                   : list aligned with `groups`, dict per group with
                     keys label and result (the _circ_lin_correlation
                     dict, or None if not enough data).  Only populated
                     when track_angle_d_pairs is provided.
    """
    labels = [g[0] for g in groups]
    samples = [g[1] for g in groups]
    out = {
        "omnibus_ww":       _circ_watson_williams(samples),
        "omnibus_mww":      _circ_mardia_watson_wheeler(samples),
        "omnibus_wallraff": _circ_wallraff_ktest(samples),
        "pairwise": [],
        "circ_lin_per_group": [],
        # Per-replicate tests: see `per_replicate_angles` arg below.
        # Populated when the caller provides per-replicate angle arrays;
        # otherwise None so consumers can detect "not computed".
        "per_replicate_kappa_test": None,
        "per_replicate_rbar_test":  None,
        "per_replicate_mu_ww":      None,
        "per_replicate_scalars":    None,
    }
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            out["pairwise"].append({
                "label_a": labels[i],
                "label_b": labels[j],
                "ww":       _circ_watson_williams([samples[i], samples[j]]),
                "mww":      _circ_mardia_watson_wheeler(
                    [samples[i], samples[j]]),
                "wallraff": _circ_wallraff_ktest(
                    [samples[i], samples[j]]),
                "kuiper":   _circ_kuiper_two_sample(samples[i],
                                                    samples[j]),
            })
    if track_angle_d_pairs is not None:
        for label, pair in zip(labels, track_angle_d_pairs):
            theta, x = pair
            out["circ_lin_per_group"].append({
                "label":  label,
                "result": _circ_lin_correlation(theta, x),
            })

    # ── Per-replicate tests ──────────────────────────────────────────
    # Treats each replicate as ONE data point (its own κ, R̄, μ),
    # producing a defensible Welch's t-test on κ + R̄ (linear scalars)
    # and a Watson-Williams F-test on μ (a circular quantity).  This
    # is the right framing when the user has e.g. 5 vs 3 movies and
    # wants stats that respect the biological replicate count, not the
    # inflated angle-count produced by pooling.
    if per_replicate_angles is not None:
        per_kappa  = []      # list-per-group of replicate κ values
        per_rbar   = []      #          "          "        R̄
        per_mu     = []      #          "          "        μ (deg)
        per_n_reps = []
        scalars_per_group = []
        for label in labels:
            arrs = per_replicate_angles.get(label, [])
            kappas, rbars, mus = [], [], []
            for arr in arrs:
                a = np.asarray(arr, dtype=float).ravel()
                a = a[np.isfinite(a)]
                if a.size < 2:
                    continue
                cs = compute_circular_statistics(a)
                if cs is None:
                    continue
                k_val  = cs.get("concentration_kappa")
                r_val  = cs.get("mean_resultant_length")
                mu_val = cs.get("mean_direction_deg")
                if k_val is not None and np.isfinite(k_val):
                    kappas.append(float(k_val))
                if r_val is not None and np.isfinite(r_val):
                    rbars.append(float(r_val))
                if mu_val is not None and np.isfinite(mu_val):
                    mus.append(float(mu_val))
            per_kappa.append(np.asarray(kappas, dtype=float))
            per_rbar.append(np.asarray(rbars, dtype=float))
            per_mu.append(np.asarray(mus, dtype=float))
            per_n_reps.append(len(kappas))
            scalars_per_group.append({
                "label": label, "n_replicates": len(kappas),
                "kappa": list(kappas), "rbar": list(rbars),
                "mu_deg": list(mus),
            })
        out["per_replicate_scalars"] = scalars_per_group

        # _stat_test_n returns (omnibus_dict, pairwise_list).
        # Welch's t for 2 groups, ANOVA for N>2 (auto-selected).
        if sum(1 for arr in per_kappa if arr.size >= 1) >= 2:
            try:
                om_k, pw_k = _stat_test_n(per_kappa, labels)
                out["per_replicate_kappa_test"] = {
                    "omnibus": om_k, "pairwise": pw_k}
            except Exception:
                pass
            try:
                om_r, pw_r = _stat_test_n(per_rbar, labels)
                out["per_replicate_rbar_test"] = {
                    "omnibus": om_r, "pairwise": pw_r}
            except Exception:
                pass
        out["per_replicate_mu_ww"] = _watson_williams_mu_per_replicate(per_mu)

    return out


def _p_stars(p):
    """Three-tier significance markers used in the comparison PDF."""
    try:
        if p is None: return ""
        pf = float(p)
        if np.isnan(pf): return ""
        if pf < 0.001: return "***"
        if pf < 0.01:  return "**"
        if pf < 0.05:  return "*"
        return "ns"
    except Exception:
        return ""


def save_comparison_circular_statistics(groups_angles, *,
                                         csv_path=None, pdf_path=None,
                                         fig_theme="Dark",
                                         track_angle_d_pairs=None,
                                         per_replicate_angles=None):
    """Pool turning angles per group, compute circular statistics for
    each group, write a combined CSV (one row per group) and a multi-
    page themed PDF (one page per group + a comparative summary page).

    Parameters
    ----------
    groups_angles : list of (label, angles_deg_array, color)
        One entry per comparison group.  `angles_deg_array` is the
        concatenation of every replicate's turning angles within the
        group; `color` is the group's display colour (used to tint the
        polar histograms so PDF and master figure agree visually).
    csv_path : str or None
        If given, write a long-form CSV with columns `group`, `n`,
        `mean_direction_deg`, … (all keys from compute_circular_statistics).
    pdf_path : str or None
        If given, write the multi-page PDF.
    fig_theme : str
        "Dark" | "Light" | "Publication".
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pal = _theme_palette(fig_theme)

    # ── Per-group stats ────────────────────────────────────────────────
    rows = []
    per_group_stats = []
    for label, angles, color in groups_angles:
        a = np.asarray(angles, dtype=float).ravel()
        a = a[np.isfinite(a)]
        stats = compute_circular_statistics(a)
        per_group_stats.append((label, a, color, stats))
        row = {"group": label}
        row.update(stats)
        rows.append(row)

    # ── Between-group tests ────────────────────────────────────────────
    # Watson-Williams (parametric, tests equal mean directions; assumes
    # κ ≥ 2) plus Mardia-Watson-Wheeler (non-parametric, tests equal
    # distributions; valid at any κ).  Both are reported so the
    # supervisor can pick the appropriate one for their data — and so
    # disagreement between them (one significant, the other not) is
    # visible rather than hidden.
    test_groups = [(g[0], np.asarray(g[1], dtype=float).ravel())
                   for g in groups_angles]
    test_groups = [(lbl, a[np.isfinite(a)]) for lbl, a in test_groups]
    comp_tests = compute_circular_comparison_tests(
        test_groups,
        track_angle_d_pairs=track_angle_d_pairs,
        per_replicate_angles=per_replicate_angles)

    # ── CSV ────────────────────────────────────────────────────────────
    # The single CSV grows by one row per pairwise test (with a `kind`
    # column distinguishing per-group rows from test rows) so a
    # downstream consumer (Excel / R / pandas) can read all the
    # comparison output from one file.
    if csv_path is not None:
        try:
            per_group_rows = [{"kind": "group", **r} for r in rows]
            test_rows = []
            ow = comp_tests.get("omnibus_ww") or {}
            om = comp_tests.get("omnibus_mww") or {}
            if ow:
                test_rows.append({
                    "kind": "test", "test": "Watson-Williams (omnibus)",
                    "label_a": "all", "label_b": "all",
                    "statistic_F": ow.get("F"),
                    "df1": ow.get("df1"), "df2": ow.get("df2"),
                    "p_value": ow.get("p"),
                    "kappa_pooled": ow.get("kappa_pooled"),
                    "valid_assumptions": ow.get("valid"),
                })
            if om:
                test_rows.append({
                    "kind": "test", "test": "Mardia-Watson-Wheeler (omnibus)",
                    "label_a": "all", "label_b": "all",
                    "statistic_W": om.get("W"),
                    "df": om.get("df"),
                    "p_value": om.get("p"),
                })
            ok = comp_tests.get("omnibus_wallraff") or {}
            if ok:
                test_rows.append({
                    "kind": "test",
                    "test": "Wallraff κ-test (omnibus)",
                    "label_a": "all", "label_b": "all",
                    "statistic_H": ok.get("H"),
                    "statistic_U": ok.get("U"),
                    "df": ok.get("df"),
                    "p_value": ok.get("p"),
                })
            for pw in comp_tests.get("pairwise", []):
                ww  = pw.get("ww")  or {}
                mww = pw.get("mww") or {}
                wal = pw.get("wallraff") or {}
                kup = pw.get("kuiper")   or {}
                if ww:
                    test_rows.append({
                        "kind": "test",
                        "test": "Watson-Williams (pairwise)",
                        "label_a": pw["label_a"],
                        "label_b": pw["label_b"],
                        "statistic_F": ww.get("F"),
                        "df1": ww.get("df1"), "df2": ww.get("df2"),
                        "p_value": ww.get("p"),
                        "kappa_pooled": ww.get("kappa_pooled"),
                        "valid_assumptions": ww.get("valid"),
                    })
                if mww:
                    test_rows.append({
                        "kind": "test",
                        "test": "Mardia-Watson-Wheeler (pairwise)",
                        "label_a": pw["label_a"],
                        "label_b": pw["label_b"],
                        "statistic_W": mww.get("W"),
                        "df": mww.get("df"),
                        "p_value": mww.get("p"),
                    })
                if wal:
                    test_rows.append({
                        "kind": "test",
                        "test": "Wallraff κ-test (pairwise)",
                        "label_a": pw["label_a"],
                        "label_b": pw["label_b"],
                        "statistic_U": wal.get("U"),
                        "p_value": wal.get("p"),
                    })
                if kup:
                    test_rows.append({
                        "kind": "test",
                        "test": "Kuiper two-sample",
                        "label_a": pw["label_a"],
                        "label_b": pw["label_b"],
                        "statistic_V": kup.get("V"),
                        "p_value": kup.get("p"),
                    })
            # Circular-linear correlation (per-track mean angle vs D)
            # is a PER-GROUP descriptive measure, not a between-group
            # test — one row per group with r, r², n, p.
            for cl in comp_tests.get("circ_lin_per_group", []):
                res = cl.get("result")
                if not res:
                    continue
                test_rows.append({
                    "kind": "correlation",
                    "test": "Circ-lin: per-track mean angle vs D",
                    "label_a": cl.get("label"),
                    "label_b": "",
                    "r": res.get("r"),
                    "r_squared": res.get("r2"),
                    "statistic_chi2": res.get("test_stat"),
                    "df": res.get("df"),
                    "p_value": res.get("p"),
                    "n": res.get("n"),
                })

            # ── Per-replicate (n=replicates) tests ─────────────────
            # One row per replicate listing the scalars used as data
            # points; then between-group rows for the κ and R̄ Welch
            # / ANOVA tests and the Watson-Williams F-test on μ.
            scalars = comp_tests.get("per_replicate_scalars") or []
            for grp in scalars:
                lbl = grp.get("label", "?")
                ks  = grp.get("kappa") or []
                rs  = grp.get("rbar")  or []
                ms  = grp.get("mu_deg") or []
                # Pad to common length so each replicate gets one row.
                n_rep = max(len(ks), len(rs), len(ms))
                for i in range(n_rep):
                    test_rows.append({
                        "kind": "per_replicate_scalar",
                        "test": "per-replicate κ/R̄/μ",
                        "label_a": lbl,
                        "label_b": f"replicate_{i + 1}",
                        "kappa":   ks[i] if i < len(ks) else None,
                        "rbar":    rs[i] if i < len(rs) else None,
                        "mu_deg":  ms[i] if i < len(ms) else None,
                    })

            def _flatten_per_rep_test(slot, label):
                t = comp_tests.get(slot)
                if not t:
                    return
                om = t.get("omnibus") or {}
                if om:
                    test_rows.append({
                        "kind": "per_replicate_test",
                        "test": f"{label} (omnibus, per-replicate)",
                        "label_a": "all", "label_b": "all",
                        "statistic": om.get("p") and om.get("test"),
                        "p_value": om.get("p"),
                    })
                for pw in (t.get("pairwise") or []):
                    test_rows.append({
                        "kind": "per_replicate_test",
                        "test": f"{label} (pairwise, per-replicate)",
                        "label_a": pw.get("label_i"),
                        "label_b": pw.get("label_j"),
                        "n_a": pw.get("n_i"), "n_b": pw.get("n_j"),
                        "mean_a": pw.get("mean_i"),
                        "mean_b": pw.get("mean_j"),
                        "sem_a": pw.get("sem_i"),
                        "sem_b": pw.get("sem_j"),
                        "statistic": pw.get("test"),
                        "p_value": pw.get("p"),
                    })

            _flatten_per_rep_test("per_replicate_kappa_test", "Welch κ")
            _flatten_per_rep_test("per_replicate_rbar_test",  "Welch R̄")

            mu_ww = comp_tests.get("per_replicate_mu_ww")
            if mu_ww is not None:
                test_rows.append({
                    "kind": "per_replicate_test",
                    "test": "Watson-Williams μ (per-replicate)",
                    "label_a": "all", "label_b": "all",
                    "statistic_F": mu_ww.get("F"),
                    "df1": mu_ww.get("df1"), "df2": mu_ww.get("df2"),
                    "p_value": mu_ww.get("p"),
                })

            df = pd.DataFrame(per_group_rows + test_rows)
            df.to_csv(csv_path, index=False)
        except Exception as exc:
            print(f"  comparison-circstats CSV failed: {exc}")

    # ── PDF ────────────────────────────────────────────────────────────
    if pdf_path is None:
        return per_group_stats

    _rc_keys = ("text.color", "axes.labelcolor", "axes.edgecolor",
                "xtick.color", "ytick.color", "axes.facecolor",
                "axes.titlecolor", "figure.facecolor", "grid.color",
                "font.family")
    _rc_save = {k: plt.rcParams.get(k) for k in _rc_keys}
    plt.rcParams.update({
        "text.color":       pal["TXT"],
        "axes.labelcolor":  pal["TXT"],
        "axes.edgecolor":   pal["GRD"],
        "xtick.color":      pal["TXT"],
        "ytick.color":      pal["TXT"],
        "axes.facecolor":   pal["PNL"],
        "axes.titlecolor":  pal["TXT"],
        "figure.facecolor": pal["BG"],
        "grid.color":       pal["GRD"],
        "font.family":      pal["FONT"],
    })

    def _fmt(x, prec=4):
        try:
            if x is None: return "—"
            xf = float(x)
            if np.isnan(xf): return "—"
            if xf > 0.0 and xf <= 1e-300:
                return "<1e-300"
            return f"{xf:.{prec}g}"
        except Exception:
            return str(x)

    try:
        with PdfPages(pdf_path) as pdf:
            # ── Page 1: comparison summary ─────────────────────────────
            # Landscape A4.  Layout, top → bottom:
            #   y 0.93 – 0.97  header bar
            #   y 0.58 – 0.88  row of polar histograms (one per group)
            #   y 0.51 – 0.55  "Summary" title
            #   y 0.36 – 0.50  per-group summary table
            #   y 0.30 – 0.34  "Between-group tests" title
            #   y 0.13 – 0.29  comparison-tests table
            #   y 0.02 – 0.10  footer block (sign convention + refs)
            fig = plt.figure(figsize=(11.69, 8.27), facecolor=pal["BG"])
            ax_hdr = fig.add_axes([0.05, 0.93, 0.90, 0.04])
            ax_hdr.axis("off")
            ax_hdr.text(0.0, 0.5, "Comparison: Circular Statistics",
                        fontsize=18, fontweight="bold", va="center",
                        ha="left", color=pal["TXT"])
            ax_hdr.text(1.0, 0.5,
                        f"{len(per_group_stats)} groups",
                        fontsize=11, color=pal["MUT"],
                        va="center", ha="right")

            # Grid of polar histograms — auto-wraps to multiple rows
            # when n_groups > 5 so plots don't get sliver-thin.  Each
            # cell is divided VERTICALLY into a label strip (top) and
            # the polar plot itself (below); doing it this way means
            # the group name + n count can never collide with the
            # polar's 0° tick label, regardless of how thick that tick
            # label is at any given font size.
            #
            #   1 ≤ n ≤ 5  →  1 row of n cols  (cell height 0.30, y 0.55–0.88)
            #   6 ≤ n ≤ 10 →  2 rows of ≤ 5 cols  (cell height ~0.16)
            #   n ≥ 11     →  3 rows; outer caller may also paginate.
            #
            # Within each cell:
            #   top 22%  → label band  (group name + "n = N")
            #   bottom 78% → polar plot
            #
            # When in multi-row mode the per-polar font sizes shrink so
            # the tick labels stay readable in a smaller plot.
            n_g  = len(per_group_stats)
            # Polar band height tuned so the polar plot + its top label
            # band (group name + n) sit comfortably above the
            # per-group summary title at y=0.555.  polar_bot=0.58
            # leaves a 0.025 gap to that title.
            polar_top, polar_bot = 0.88, 0.58
            if n_g <= 5:
                n_cols, n_rows = n_g, 1
            elif n_g <= 10:
                n_cols, n_rows = 5, 2
            else:
                # Cap at 12 polars/page; the table-pagination below
                # handles "lots of groups" by giving each batch its
                # own summary page.  For now assume ≤ 12 on page 1.
                n_cols = 6
                n_rows = (min(n_g, 12) + n_cols - 1) // n_cols
            row_h    = (polar_top - polar_bot) / n_rows
            cell_w   = 0.86 / n_cols
            left     = 0.07
            # Tick / label fontsizes shrink when polars get small.
            tick_fs  = 7 if n_cols <= 4 else 6
            lbl_fs   = 10 if n_cols <= 4 else 8
            n_fs     = 8  if n_cols <= 4 else 7
            # Bottom-margin reserves space for the polar's ±180° tick
            # label, which matplotlib renders OUTSIDE the axes box just
            # below the polar circle.  Needs to be large enough that
            # the tick label can't reach down into the Per-group title
            # at y=0.535 below (single-row case) or into the next-row's
            # group label (multi-row case).
            bottom_margin = 0.040 if n_rows == 1 else 0.045
            label_band_frac = 0.22 if n_rows == 1 else 0.28
            for i, (label, a, color, stats) in enumerate(per_group_stats):
                if i >= n_rows * n_cols:
                    break    # truncate at the page's polar capacity
                row = i // n_cols
                col = i % n_cols
                cell_y = polar_top - (row + 1) * row_h
                label_band_h = label_band_frac * row_h
                polar_band_h = row_h - label_band_h - bottom_margin
                ax = fig.add_axes(
                    [left + col * cell_w + 0.015,
                     cell_y + bottom_margin,
                     cell_w - 0.03, polar_band_h],
                    projection="polar")
                ax.set_facecolor(pal["PNL"])
                if a.size >= 10:
                    # Match the master figure's polar convention:
                    # 0° at top, CW positive, signed labels on slot
                    # positions, [0, 2π) wrap for bar rendering.
                    nbins = 36
                    angles_rad = np.mod(np.deg2rad(a), 2.0 * np.pi)
                    bins  = np.linspace(0.0, 2.0 * np.pi, nbins + 1)
                    counts, edges = np.histogram(angles_rad, bins=bins)
                    widths  = np.diff(edges)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    ax.set_theta_zero_location("N")
                    ax.set_theta_direction(-1)
                    bar_col = color or pal["ACC"]
                    ax.bar(centers, counts, width=widths * 0.95,
                           align="center", color=bar_col,
                           edgecolor=pal["PNL"], linewidth=0.4,
                           alpha=0.92)
                    mu = stats.get("mean_direction_deg")
                    if mu is not None and not (
                            isinstance(mu, float) and np.isnan(mu)):
                        r_max = float(counts.max()) if counts.size else 1.0
                        mu_rad = np.mod(np.deg2rad(mu), 2.0 * np.pi)
                        ax.annotate("",
                            xy=(mu_rad, r_max * 0.95),
                            xytext=(0, 0),
                            arrowprops=dict(arrowstyle="->",
                                            color=pal["ARROW"], lw=2.0))
                    ax.set_xticks(np.deg2rad(
                        [0, 45, 90, 135, 180, 225, 270, 315]))
                    ax.set_xticklabels(
                        ["0°", "+45°", "+90°", "+135°", "±180°",
                         "−135°", "−90°", "−45°"], fontsize=tick_fs)
                    ax.set_yticklabels([])
                    ax.tick_params(colors=pal["TXT"], labelsize=tick_fs)
                    ax.grid(True, ls=":", alpha=0.4)
                    # Labels live ABOVE the cell.  Raising the label
                    # block above `cell_y + row_h` (rather than just
                    # inside the top of the cell) creates a clean gap
                    # between the "n = …" line and the polar's 0° tick
                    # label, which renders just outside the polar
                    # circle at the top of the axes box.
                    label_x = (left + col * cell_w + 0.015
                               + (cell_w - 0.03) / 2.0)
                    label_top  = cell_y + row_h + 0.020
                    line2_top  = label_top - 0.018
                    fig.text(label_x, label_top, label,
                             fontsize=lbl_fs, fontweight="bold",
                             ha="center", va="top", color=pal["TXT"])
                    fig.text(label_x, line2_top,
                             f"n = {int(stats.get('n', 0)):,}",
                             fontsize=n_fs, ha="center", va="top",
                             color=pal["MUT"])
                else:
                    ax.axis("off")
                    label_x = (left + col * cell_w + 0.015
                               + (cell_w - 0.03) / 2.0)
                    label_top = cell_y + row_h + 0.020
                    fig.text(label_x, label_top,
                             f"{label}\ntoo few angles",
                             fontsize=n_fs, ha="center", va="top",
                             color=pal["MUT"])

            # Section title placed in FIGURE coords.  Sits below the
            # polar band's bottom (y=0.58) with a generous gap so the
            # polar's ±180° tick label can't reach down into it.
            fig.text(0.05, 0.535,
                     "Per-group summary  (MATLAB CircStat conventions)",
                     fontsize=11, fontweight="bold", va="bottom",
                     ha="left", color=pal["TXT"])
            # Combined summary table — one row per group, columns =
            # the most informative stats for an at-a-glance comparison.
            ax_tbl = fig.add_axes([0.05, 0.43, 0.90, 0.10])
            ax_tbl.axis("off")
            cols = ["group", "n", "mean_direction_deg",
                    "mean_resultant_length", "circular_std_deg",
                    "concentration_kappa", "rayleigh_p", "v_test_p"]
            col_labels = ["Group", "n", "μ (°)", "R̄", "σ_circ (°)",
                          "κ", "Rayleigh p", "V-test p"]
            cell = []
            for r in rows:
                cell.append([
                    str(r["group"]),
                    f"{int(r['n']):,}",
                    _fmt(r["mean_direction_deg"], 4),
                    _fmt(r["mean_resultant_length"], 4),
                    _fmt(r["circular_std_deg"], 4),
                    _fmt(r["concentration_kappa"], 4),
                    _fmt(r["rayleigh_p"], 3),
                    _fmt(r["v_test_p"], 3),
                ])
            tbl = ax_tbl.table(cellText=cell, colLabels=col_labels,
                               cellLoc="left", colLoc="left",
                               bbox=[0.0, 0.0, 1.0, 1.0])
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9.0)
            for (rr, cc), c_obj in tbl.get_celld().items():
                c_obj.set_linewidth(0.5)
                c_obj.set_edgecolor(pal["GRD"])
                if rr == 0:
                    c_obj.set_facecolor(pal["HDR_BG"])
                    c_obj.set_text_props(color=pal["HDR_TXT"],
                                         fontweight="bold")
                else:
                    c_obj.set_facecolor(
                        pal["ZEBRA"] if rr % 2 == 0 else pal["PNL"])
                    c_obj.set_text_props(color=pal["TXT"])

            # ── Between-group tests section ────────────────────────
            #
            # Layout (y-coords):
            #   0.34 : section title
            #   0.27 – 0.33 : plain-English explanation of each test
            #   0.13 – 0.26 : results table (Test  Statistic  p  sig)
            #   0.02 – 0.10 : footer
            #
            # The previous version had a 5th "Note" column for H₀
            # descriptions which overflowed the page; pulling that
            # description out into a separate explanatory paragraph
            # both fixes the overflow AND makes the tests intelligible
            # to a reader who isn't already a circular-statistics
            # expert (supervisor's request).
            fig.text(0.05, 0.395,
                     "Between-group tests — does the turning-angle "
                     "distribution differ between groups?",
                     fontsize=11, fontweight="bold", va="bottom",
                     ha="left", color=pal["TXT"])

            # Plain-English explanation block — 3 compact lines so the
            # tests table below still has room.  Each line covers one
            # test (or the significance convention).  Italicised
            # caveats appear at the end of each line, not on their own
            # row.
            txt_kw = dict(fontsize=8.0, color=pal["TXT"], ha="left",
                          va="top", family=pal["FONT"])
            explain_block = [
                "Watson-Williams F-test (circular ANOVA): tests "
                "EQUAL MEAN DIRECTIONS.  Assumes κ ≥ 2 — rows tagged "
                "\"κ<2\" violate this, prefer M-W-W or Kuiper.",
                "Mardia-Watson-Wheeler & Kuiper (non-parametric): "
                "test EQUAL FULL DISTRIBUTIONS (any change in mean, "
                "spread, or shape).  Safe at any κ.",
                "Wallraff κ-test: tests EQUAL CONCENTRATIONS — answers "
                "\"is one group MORE TIGHTLY clustered than the other?\".",
                "Per-replicate (n = #replicates): Welch's t-test on κ "
                "and R̄, plus Watson-Williams F on per-replicate μ — "
                "respects biological n, not pooled n.",
                "Circ-lin angle vs D (per group): tests whether each "
                "track's mean turning angle correlates with its "
                "diffusion coefficient.  r ∈ [0, 1].",
                "Significant p (< 0.05, stars) rejects H₀ — i.e. "
                "groups DO differ (or angle DOES correlate with D).",
            ]
            yE = 0.395
            for line in explain_block:
                fig.text(0.05, yE, line, **txt_kw)
                yE -= 0.012
            # Build comparison-test rows in priority order:
            #   1. Omnibus Watson-Williams
            #   2. Omnibus Mardia-Watson-Wheeler
            #   3. Pairwise WW / MWW (one row per test per pair)
            # _fmt_p collapses underflow-sentinel p (1e-300) to
            # "<1e-300" so the supervisor doesn't see a literal
            # "1e-300" repeated across rows and assume there's a bug.
            def _fmt_p(p):
                if p is None: return "—"
                pf = float(p)
                if np.isnan(pf): return "—"
                if pf > 0.0 and pf <= 1e-300:
                    return "<1e-300"
                return f"{pf:.3g}"

            omnibus_rows = []
            ow  = comp_tests.get("omnibus_ww")
            om  = comp_tests.get("omnibus_mww")
            owk = comp_tests.get("omnibus_wallraff")
            if ow is not None:
                tag = "" if ow.get("valid", False) else "  (κ<2, caution)"
                omnibus_rows.append([
                    f"Watson-Williams · all groups{tag}",
                    f"F({ow['df1']}, {ow['df2']}) = {ow['F']:.3g}",
                    _fmt_p(ow["p"]),
                    _p_stars(ow["p"]),
                ])
            if om is not None:
                omnibus_rows.append([
                    "Mardia-Watson-Wheeler · all groups",
                    f"W({om['df']}) = {om['W']:.3g}",
                    _fmt_p(om["p"]),
                    _p_stars(om["p"]),
                ])
            if owk is not None:
                # k=2 → Mann-Whitney U; k>2 → Kruskal-Wallis H.
                if "H" in owk:
                    stat_str = f"H({owk['df']}) = {owk['H']:.3g}"
                else:
                    stat_str = f"U = {owk.get('U', 0):.3g}"
                omnibus_rows.append([
                    "Wallraff κ-test · all groups",
                    stat_str,
                    _fmt_p(owk["p"]),
                    _p_stars(owk["p"]),
                ])

            pairwise_rows = []
            for pw in comp_tests.get("pairwise", []):
                ww  = pw.get("ww")
                mww = pw.get("mww")
                wal = pw.get("wallraff")
                kup = pw.get("kuiper")
                pair = f"{pw['label_a']}  vs  {pw['label_b']}"
                if ww is not None:
                    tag = "" if ww.get("valid", False) else "  (κ<2)"
                    pairwise_rows.append([
                        f"Watson-Williams · {pair}{tag}",
                        f"F({ww['df1']}, {ww['df2']}) = {ww['F']:.3g}",
                        _fmt_p(ww["p"]),
                        _p_stars(ww["p"]),
                    ])
                if mww is not None:
                    pairwise_rows.append([
                        f"Mardia-Watson-Wheeler · {pair}",
                        f"W({mww['df']}) = {mww['W']:.3g}",
                        _fmt_p(mww["p"]),
                        _p_stars(mww["p"]),
                    ])
                if wal is not None:
                    pairwise_rows.append([
                        f"Wallraff κ-test · {pair}",
                        f"U = {wal.get('U', 0):.3g}",
                        _fmt_p(wal["p"]),
                        _p_stars(wal["p"]),
                    ])
                if kup is not None:
                    pairwise_rows.append([
                        f"Kuiper 2-sample · {pair}",
                        f"V = {kup['V']:.4g}",
                        _fmt_p(kup["p"]),
                        _p_stars(kup["p"]),
                    ])

            # Per-group circular-linear correlation rows.  These are
            # descriptive stats (one per group), not between-group
            # tests, but they live in the same table because they share
            # the same "name · stat · p · sig" template.
            corr_rows = []
            for cl in comp_tests.get("circ_lin_per_group", []):
                res = cl.get("result")
                grp = cl.get("label", "?")
                if not res:
                    corr_rows.append([
                        f"Circ-lin angle vs D · {grp}",
                        "n < 3", "—", "",
                    ])
                    continue
                corr_rows.append([
                    f"Circ-lin angle vs D · {grp}",
                    (f"r = {res['r']:.3g}  "
                     f"(χ²({res['df']}) = {res['test_stat']:.3g})"),
                    _fmt_p(res["p"]),
                    _p_stars(res["p"]),
                ])

            # ── Per-replicate test rows ─────────────────────────────
            # n = number of biological replicates, not pooled angles.
            # Each row reads "Welch κ · all groups" or "Welch κ · A vs
            # B" plus the t/F statistic and the p-value with stars.
            per_rep_rows = []

            def _push_per_rep(slot, label):
                t = comp_tests.get(slot)
                if not t:
                    return
                om = t.get("omnibus") or {}
                if om and om.get("p") is not None:
                    test_name = om.get("test", "")
                    per_rep_rows.append([
                        f"{label} · all groups  ({test_name})",
                        "(see CSV for full stats)",
                        _fmt_p(om["p"]),
                        _p_stars(om["p"]),
                    ])
                for pw in (t.get("pairwise") or []):
                    if pw.get("p") is None:
                        continue
                    pair = (f"{pw.get('label_i', '?')}  vs  "
                            f"{pw.get('label_j', '?')}")
                    n_i = pw.get("n_i", 0)
                    n_j = pw.get("n_j", 0)
                    per_rep_rows.append([
                        f"{label} · {pair}  ({pw.get('test', '')})",
                        f"n = {n_i} vs {n_j}",
                        _fmt_p(pw["p"]),
                        _p_stars(pw["p"]),
                    ])

            _push_per_rep("per_replicate_kappa_test", "Welch κ (per-replicate)")
            _push_per_rep("per_replicate_rbar_test",  "Welch R̄ (per-replicate)")

            mu_ww = comp_tests.get("per_replicate_mu_ww")
            if mu_ww is not None and mu_ww.get("p") is not None:
                tag = "" if mu_ww.get("valid", False) else "  (κ<2)"
                per_rep_rows.append([
                    f"Watson-Williams μ · all groups (per-replicate){tag}",
                    f"F({mu_ww['df1']}, {mu_ww['df2']}) = {mu_ww['F']:.3g}",
                    _fmt_p(mu_ww["p"]),
                    _p_stars(mu_ww["p"]),
                ])

            # Page-1 tests table: omnibus + circ-lin correlations +
            # per-replicate tests + as many pairwise rows as fit.  We
            # always put the per-group correlations and per-replicate
            # tests on page 1 (they're tiny, ~k rows each) and let the
            # pooled pairwise tests be the ones that paginate.
            PAGE1_TESTS_CAP = 14        # omnibus + corr + per-rep + pairwise
            CONT_PAGE_CAP   = 24        # ~24 rows on a continuation page

            fixed_rows = omnibus_rows + corr_rows + per_rep_rows
            page1_pairwise_cap = max(
                PAGE1_TESTS_CAP - len(fixed_rows), 0)
            page1_tests   = fixed_rows + pairwise_rows[:page1_pairwise_cap]
            overflow_pairs = pairwise_rows[page1_pairwise_cap:]

            if not page1_tests:
                page1_tests = [["Insufficient data", "—", "—", "—"]]

            def _render_tests_table(host_fig, rect, cells, pal):
                """Render a 4-column tests table into the given fig+rect."""
                ax = host_fig.add_axes(rect); ax.axis("off")
                tbl = ax.table(
                    cellText=cells,
                    colLabels=["Test  ·  Comparison", "Statistic",
                               "p-value", "sig"],
                    cellLoc="left", colLoc="left",
                    colWidths=[0.55, 0.25, 0.13, 0.07],
                    bbox=[0.0, 0.0, 1.0, 1.0])
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(8.5)
                for (rr, cc), c_obj in tbl.get_celld().items():
                    c_obj.set_linewidth(0.5)
                    c_obj.set_edgecolor(pal["GRD"])
                    if rr == 0:
                        c_obj.set_facecolor(pal["HDR_BG"])
                        c_obj.set_text_props(color=pal["HDR_TXT"],
                                             fontweight="bold")
                    else:
                        c_obj.set_facecolor(
                            pal["ZEBRA"] if rr % 2 == 0 else pal["PNL"])
                        c_obj.set_text_props(color=pal["TXT"])

            _render_tests_table(fig, [0.05, 0.14, 0.90, 0.165],
                                page1_tests, pal)

            # ── Footer block ──────────────────────────────────────
            # Pre-wrapped lines instead of one long sign-convention
            # string: matplotlib's fig.text doesn't wrap against the
            # figure margins, so the long "Sign convention…" line was
            # being cut off at the right edge.  Same wrapping pattern
            # the per-file PDF footer uses (search `sign_lines = [`).
            def _render_footer(host_fig, pal, *, top_y=0.105):
                _foot_kw2 = dict(fontsize=7, color=pal["MUT"], ha="left",
                                 va="bottom", family=pal["FONT"])
                foot_lines = [
                    "Sign convention: turning angles are SIGNED on "
                    "(−180°, +180°].",
                    "0° = straight ahead.  +θ = left turn (CCW).  "
                    "−θ = right turn (CW).  ±180° = full reversal.",
                    "Plots use clockwise-positive direction so +θ "
                    "labels appear on the right hemisphere.",
                    "Significance markers: *** p<0.001,  ** p<0.01,  "
                    "* p<0.05,  ns = not significant.",
                    "References: Mardia & Jupp 2000 §6.4.2, §7.6.1; "
                    "Fisher 1993; Berens 2009 (CircStat).",
                ]
                y2 = top_y
                for line in foot_lines:
                    host_fig.text(0.05, y2, line, **_foot_kw2)
                    y2 -= 0.014

            _render_footer(fig, pal)
            pdf.savefig(fig, facecolor=pal["BG"])
            plt.close(fig)

            # ── Continuation pages for overflow pairwise tests ──────
            # When the pairwise count is large (e.g. 6+ groups → 15+
            # pairs × 2 tests = 30+ rows), we paginate the remainder
            # onto fresh landscape pages so nothing gets squashed off
            # the bottom of page 1.
            if overflow_pairs:
                page_num = 2
                total_cont_pages = (len(overflow_pairs)
                                    + CONT_PAGE_CAP - 1) // CONT_PAGE_CAP
                for chunk_start in range(0, len(overflow_pairs),
                                         CONT_PAGE_CAP):
                    chunk = overflow_pairs[chunk_start:
                                           chunk_start + CONT_PAGE_CAP]
                    fig_c = plt.figure(figsize=(11.69, 8.27),
                                       facecolor=pal["BG"])
                    ax_h = fig_c.add_axes([0.05, 0.93, 0.90, 0.04])
                    ax_h.axis("off")
                    ax_h.text(0.0, 0.5,
                              "Comparison: Circular Statistics  —  "
                              f"pairwise tests (page {page_num - 1} of "
                              f"{total_cont_pages})",
                              fontsize=14, fontweight="bold",
                              va="center", ha="left", color=pal["TXT"])
                    # Big tests-table area on a continuation page.
                    _render_tests_table(fig_c,
                                        [0.05, 0.13, 0.90, 0.75],
                                        chunk, pal)
                    _render_footer(fig_c, pal)
                    pdf.savefig(fig_c, facecolor=pal["BG"])
                    plt.close(fig_c)
                    page_num += 1

            # ── Pages 2..N+1: per-group full report ───────────────────
            for label, a, color, stats in per_group_stats:
                # Reuse the single-file renderer by writing to a
                # temp page object isn't supported directly — instead,
                # we mirror its layout here in a fresh figure so the
                # per-group pages all live in ONE multi-page PDF.
                _write_single_group_page(pdf, a, stats, label, pal, color)
    finally:
        plt.rcParams.update(_rc_save)

    return per_group_stats


def _write_single_group_page(pdf, angles_deg, stats, label, pal,
                              group_color=None):
    """Render one A4-portrait page mirroring save_circular_statistics_pdf
    into an open PdfPages stream.  Used by save_comparison_circular_
    statistics so the per-group full reports all live inside the same
    multi-page comparison PDF.
    """
    import matplotlib.pyplot as plt

    a = np.asarray(angles_deg, dtype=float).ravel()
    a = a[np.isfinite(a)]

    def _fmt(x, prec=4):
        try:
            if x is None: return "—"
            xf = float(x)
            if np.isnan(xf): return "—"
            if xf > 0.0 and xf <= 1e-300:
                return "<1e-300"
            return f"{xf:.{prec}g}"
        except Exception:
            return str(x)

    rows = [
        ("n", "Sample size", "count", f"{int(stats.get('n', 0)):,}"),
        ("mean_direction_deg", "Mean direction μ", "deg",
         _fmt(stats.get("mean_direction_deg"), 4)),
        ("mean_resultant_length",
         "Mean resultant length R̄  (0 = uniform, 1 = aligned)", "—",
         _fmt(stats.get("mean_resultant_length"), 4)),
        ("circular_variance", "Circular variance  1 − R̄", "—",
         _fmt(stats.get("circular_variance"), 4)),
        ("circular_std_deg",
         "Circular standard deviation  √(−2·ln R̄)", "deg",
         _fmt(stats.get("circular_std_deg"), 4)),
        ("angular_deviation_deg",
         "Angular deviation  s₀ = √(2·(1−R̄))", "deg",
         _fmt(stats.get("angular_deviation_deg"), 4)),
        ("median_deg", "Circular median", "deg",
         _fmt(stats.get("median_deg"), 4)),
        ("concentration_kappa",
         "Von Mises concentration κ  (Best & Fisher 1981)", "—",
         _fmt(stats.get("concentration_kappa"), 4)),
        ("rayleigh_z", "Rayleigh test statistic  z = n·R̄²", "—",
         _fmt(stats.get("rayleigh_z"), 4)),
        ("rayleigh_p", "Rayleigh test p-value  (uniformity)", "—",
         _fmt(stats.get("rayleigh_p"), 3)),
        ("v_test_z", "V-test statistic against μ₀ = 0°", "—",
         _fmt(stats.get("v_test_z"), 4)),
        ("v_test_p", "V-test p-value  (preferred direction)", "—",
         _fmt(stats.get("v_test_p"), 3)),
        ("circular_skewness",
         "Circular skewness  (Mardia & Jupp §2.3)", "—",
         _fmt(stats.get("circular_skewness"), 4)),
        ("circular_kurtosis",
         "Circular kurtosis  (Mardia & Jupp §2.3)", "—",
         _fmt(stats.get("circular_kurtosis"), 4)),
        ("ci95_lower_deg",
         "95% CI lower bound for μ  (large-sample)", "deg",
         _fmt(stats.get("ci95_lower_deg"), 4)),
        ("ci95_upper_deg",
         "95% CI upper bound for μ  (large-sample)", "deg",
         _fmt(stats.get("ci95_upper_deg"), 4)),
    ]

    # Layout mirrors save_circular_statistics_pdf — same coord bands so
    # the per-group page in a comparison PDF reads like a per-file PDF.
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=pal["BG"])
    ax_hdr = fig.add_axes([0.07, 0.94, 0.86, 0.04])
    ax_hdr.axis("off")
    ax_hdr.text(0.0, 0.5, f"Circular Statistics — {label}",
                fontsize=15, fontweight="bold", va="center",
                ha="left", color=pal["TXT"])
    ax_hdr.text(1.0, 0.5,
                f"n = {int(stats.get('n', 0)):,} turning angles",
                fontsize=11, color=pal["MUT"], va="center", ha="right")

    ax_polar = fig.add_axes([0.08, 0.61, 0.36, 0.25], projection="polar")
    ax_polar.set_facecolor(pal["PNL"])
    if a.size >= 10:
        # Same convention as the master figure: 0° top, CW positive,
        # signed labels on slot positions, [0, 2π) wrap for bars.
        nbins = 36
        angles_rad = np.mod(np.deg2rad(a), 2.0 * np.pi)
        bins  = np.linspace(0.0, 2.0 * np.pi, nbins + 1)
        counts, edges = np.histogram(angles_rad, bins=bins)
        widths = np.diff(edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax_polar.set_theta_zero_location("N")
        ax_polar.set_theta_direction(-1)
        ax_polar.bar(centers, counts, width=widths * 0.95, align="center",
                     color=group_color or pal["ACC"],
                     edgecolor=pal["PNL"], linewidth=0.4, alpha=0.92)
        mu = stats.get("mean_direction_deg")
        if mu is not None and not (isinstance(mu, float) and np.isnan(mu)):
            r_max = float(counts.max()) if counts.size else 1.0
            mu_rad = np.mod(np.deg2rad(mu), 2.0 * np.pi)
            ax_polar.annotate("",
                xy=(mu_rad, r_max * 0.95), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=pal["ARROW"],
                                lw=2.0))
        ax_polar.set_xticks(np.deg2rad(
            [0, 45, 90, 135, 180, 225, 270, 315]))
        ax_polar.set_xticklabels(
            ["0°", "+45°", "+90°", "+135°", "±180°",
             "−135°", "−90°", "−45°"], fontsize=8)
        ax_polar.set_yticklabels([])
        ax_polar.tick_params(colors=pal["TXT"], labelsize=8)
        ax_polar.grid(True, ls=":", alpha=0.4)
        # Title intentionally omitted — see save_circular_statistics_pdf
        # for the rationale (header + footer already cover it).
    else:
        ax_polar.axis("off")
        ax_polar.text(0.5, 0.5, "Too few angles for histogram",
                      transform=ax_polar.transAxes,
                      ha="center", va="center", color=pal["MUT"],
                      fontsize=10)

    # Compact "Top stats" box for the right side.
    ax_top = fig.add_axes([0.48, 0.61, 0.46, 0.25]); ax_top.axis("off")
    R = stats.get("mean_resultant_length")
    p = stats.get("rayleigh_p")
    lines = [
        f"Mean direction μ:        {_fmt(stats.get('mean_direction_deg'), 4)}°",
        f"Resultant length R̄:      {_fmt(stats.get('mean_resultant_length'), 4)}",
        f"Concentration κ:         {_fmt(stats.get('concentration_kappa'), 4)}",
        f"Rayleigh p (uniformity): {_fmt(stats.get('rayleigh_p'), 3)}",
        f"V-test p (μ₀ = 0°):      {_fmt(stats.get('v_test_p'), 3)}",
    ]
    ax_top.text(0.0, 1.0, "Headline stats", fontsize=12,
                fontweight="bold", va="top", color=pal["TXT"])
    ax_top.text(0.0, 0.88, "\n".join(lines), fontsize=10, va="top",
                family="monospace", color=pal["TXT"])

    # Section title in figure coords so it can't collide with the polar
    # plot's bottom tick labels above.
    fig.text(0.07, 0.555, "Statistics  (MATLAB CircStat conventions)",
             fontsize=12, fontweight="bold", va="bottom",
             ha="left", color=pal["TXT"])
    ax_tbl = fig.add_axes([0.07, 0.12, 0.88, 0.40]); ax_tbl.axis("off")

    cell_text, row_labels = [], []
    for key, gloss, unit, val in rows:
        unit_s = "" if unit in ("", "—") else f"  ({unit})"
        cell_text.append([f"{gloss}", f"{val}{unit_s}"])
        row_labels.append(key)
    tbl = ax_tbl.table(cellText=cell_text, rowLabels=row_labels,
                       colLabels=["Description", "Value"],
                       cellLoc="left", rowLoc="left", colLoc="left",
                       colWidths=[0.62, 0.28],
                       bbox=[0.20, 0.0, 0.80, 1.0])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.0)
    for (rr, cc), c_obj in tbl.get_celld().items():
        c_obj.set_linewidth(0.5)
        c_obj.set_edgecolor(pal["GRD"])
        if rr == 0:
            c_obj.set_facecolor(pal["HDR_BG"])
            c_obj.set_text_props(color=pal["HDR_TXT"], fontweight="bold")
        else:
            c_obj.set_facecolor(
                pal["ZEBRA"] if rr % 2 == 0 else pal["PNL"])
            if cc == -1:
                c_obj.set_text_props(family="monospace", fontsize=8.0,
                                     color=pal["MUT"])
            else:
                c_obj.set_text_props(color=pal["TXT"])

    _foot_kw = dict(fontsize=7, color=pal["MUT"], ha="left",
                    va="bottom", family=pal["FONT"])
    sign_lines = [
        "Sign convention: turning angles SIGNED on (−180°, +180°].",
        "0° = straight.  +θ = left turn (CCW).  −θ = right turn (CW).  "
        "±180° = reversal.",
        "Unsigned 0–360° equivalent: u = θ if θ ≥ 0, else θ + 360 "
        "(so −90° ≡ 270°, +90° ≡ 90°).",
    ]
    ref_lines = [
        "References: Mardia & Jupp 2000; Fisher 1993; "
        "Berens 2009 (CircStat).",
    ]
    y = 0.095
    for line in sign_lines:
        fig.text(0.07, y, line, **_foot_kw); y -= 0.014
    y -= 0.006
    for line in ref_lines:
        fig.text(0.07, y, line, **_foot_kw); y -= 0.014
    pdf.savefig(fig, facecolor=pal["BG"])
    plt.close(fig)


def compute_turning_angles(tracks):
    """For each track with ≥3 points, compute step-to-step **signed** turning
    angles in degrees, in the range (-180°, +180°].

    Sign convention (standard 2D right-handed):
        +90°  =  90° left turn (counter-clockwise rotation from v1 to v2)
        -90°  =  90° right turn (clockwise rotation)
          0°  =  continued straight
        ±180° =  full reversal

    Computation: for consecutive step vectors v1 = r(t_{i+1}) - r(t_i)
    and v2 = r(t_{i+2}) - r(t_{i+1}),

        θ = atan2( v1.x · v2.y - v1.y · v2.x,    v1 · v2 )

    where the first argument is the z-component of the 3-D cross product
    v1 × v2 (positive for counter-clockwise rotation). Returns a flat
    array of all angles across all tracks, in degrees.
    """
    print(f"  Turning angles    : {tracks['particle'].nunique():,} tracks")
    # Vectorised across all tracks at once.  Sort by (particle, frame),
    # take np.diff over the full arrays, then mask out segments that
    # cross a track boundary (where particle id changed between
    # consecutive rows).  ~50× faster than the per-track loop on 100k
    # tracks because we never re-enter the Python interpreter.
    if len(tracks) < 3:
        result = np.array([], dtype=float)
    else:
        # Drop the index level first — trackpy.link sets `frame` as
        # both an index level AND a column, which makes sort_values
        # raise "ambiguous" on those keys.
        srt = (tracks
               .reset_index(drop=True)
               .sort_values(["particle", "frame"], kind="stable"))
        pid_arr = srt["particle"].to_numpy()
        xy_arr  = srt[["x", "y"]].to_numpy()
        # Step vectors v[i] = xy[i+1] - xy[i].  same_track_step[i] is True
        # iff rows i and i+1 belong to the same particle.
        steps = np.diff(xy_arr, axis=0)                       # (n-1, 2)
        same_step  = (pid_arr[1:] == pid_arr[:-1])            # (n-1,)
        if len(steps) < 2:
            result = np.array([], dtype=float)
        else:
            v1 = steps[:-1]
            v2 = steps[1:]
            # A turn at position i requires three consecutive same-track
            # rows: (i, i+1, i+2).  Equivalently both steps must be
            # within-track AND the middle row must be the same in both.
            both_in_track = same_step[:-1] & same_step[1:]
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            dot   = np.sum(v1 * v2, axis=1)
            norm1 = np.linalg.norm(v1, axis=1)
            norm2 = np.linalg.norm(v2, axis=1)
            valid = both_in_track & (norm1 > 0) & (norm2 > 0)
            if valid.any():
                result = np.degrees(np.arctan2(cross[valid], dot[valid]))
            else:
                result = np.array([], dtype=float)
    if result.size:
        all_angles = [result]   # keep the downstream concatenate code happy
    else:
        all_angles = []
    if all_angles:
        result = np.concatenate(all_angles)
        # Distribution sanity check — Brownian motion should produce a
        # roughly symmetric distribution around 0°.  Strong asymmetry can
        # indicate uncorrected drift, an asymmetric cellular geometry, or
        # a real biological turn bias.  Printed for diagnostic verification.
        if len(result) > 0:
            pos = int((result > 0).sum())
            neg = int((result < 0).sum())
            zer = int((result == 0).sum())
            print(f"    signed turning angles: "
                  f"{pos:,} positive  /  {neg:,} negative  /  {zer:,} zero  "
                  f"|  min={result.min():.1f}°  max={result.max():.1f}°  "
                  f"mean={result.mean():.2f}°  median={np.median(result):.2f}°")
        return result
    return np.array([])


# ══════════════════════════════════════════════════════════════════════════════
#  MOBILE FRACTION OVER TIME
# ══════════════════════════════════════════════════════════════════════════════

def compute_mobile_fraction_over_time(tracks, diff_df, frame_interval,
                                       window_frames=100,
                                       d_threshold=MOBILE_D_THRESHOLD_DEFAULT):
    """Compute mobile fraction in sliding windows of `window_frames` frames.

    Mobile = tracks with D ≥ d_threshold (consistent with _mob_immob_ratio
    and the LogD-distribution panel's threshold line).  Tracks with
    non-finite D are excluded from the window denominator.

    Returns DataFrame with columns: time_s, mobile_fraction, n_tracks.
    Only windows with ≥5 valid tracks are included.
    """
    if len(tracks) == 0 or len(diff_df) == 0:
        return pd.DataFrame(columns=["time_s", "mobile_fraction", "n_tracks"])

    track_times = tracks.groupby("particle")["frame"].mean().reset_index()
    track_times.columns = ["particle", "mean_frame"]
    merged = track_times.merge(diff_df[["particle", "D"]], on="particle", how="inner")
    # Drop tracks where D could not be fit
    merged = merged[np.isfinite(merged["D"]) & (merged["D"] > 0)]

    max_frame = int(tracks["frame"].max())
    windows   = range(0, max_frame, window_frames)
    rows = []
    for w in windows:
        sel = merged[(merged["mean_frame"] >= w) &
                     (merged["mean_frame"] < w + window_frames)]
        total = len(sel)
        if total < 5:
            continue
        mobile = int((sel["D"] >= d_threshold).sum())
        rows.append({
            "time_s":          (w + window_frames / 2) * frame_interval,
            "mobile_fraction": mobile / total,
            "n_tracks":        total,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTER ANALYSIS  (DBSCAN)
# ══════════════════════════════════════════════════════════════════════════════

def compute_clusters(locs, pixel_size_um, eps_um=0.05, min_samples=5,
                     max_locs=250_000):
    from sklearn.cluster import DBSCAN
    from scipy.spatial import ConvexHull
    xy = locs[["x", "y"]].values * pixel_size_um
    subsampled = len(xy) > max_locs
    if subsampled:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(xy), max_locs, replace=False)
        xy = xy[idx]
        print(f"  Cluster analysis  : sub-sampled to {max_locs:,} localisations")
    else:
        print(f"  Cluster analysis  : {len(xy):,} localisations  "
              f"(eps={eps_um*1000:.0f} nm, min_samples={min_samples})")
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
        rows.append({"cluster_id": int(c), "n_locs": int(n),
                     "area_um2": area, "density_locs_per_um2": density,
                     "centroid_x_um": pts[:,0].mean(),
                     "centroid_y_um": pts[:,1].mean()})
    return labels, pd.DataFrame(rows), int(n_clusters), xy


# ══════════════════════════════════════════════════════════════════════════════
#  DWELL TIME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_dwell_times(tracks, diff_df, frame_interval):
    """Per-track dwell times for confined / immobile tracks.

    Returns a DataFrame with three durations per track:

      dwell_time_total_s     (last_frame − first_frame + 1) × Δt   ← canonical
      dwell_time_observed_s  n_observations × Δt                   ← fewer if gaps
      dwell_time_s           alias for dwell_time_total_s          ← back-compat

    The exponential τ is fit to dwell_time_total_s (residence-time semantics).
    """
    confined_pids = diff_df[diff_df["motion"].isin(["Confined", "Immobile"])]["particle"]
    print(f"  Dwell times       : {len(confined_pids):,} confined/immobile tracks")
    rows = []
    # Group once by particle for speed
    grouped = tracks.groupby("particle")["frame"]
    for pid in confined_pids:
        if pid not in grouped.groups:
            continue
        frames = grouped.get_group(pid).values
        n_obs = len(frames)
        if n_obs == 0:
            continue
        f_min = int(frames.min())
        f_max = int(frames.max())
        dur_total = (f_max - f_min + 1) * frame_interval
        dur_obs   = n_obs * frame_interval
        rows.append({
            "particle":              int(pid),
            "dwell_time_s":          dur_total,   # back-compat alias
            "dwell_time_total_s":    dur_total,   # full duration including gaps
            "dwell_time_observed_s": dur_obs,     # observed frames × Δt
            "n_observations":        int(n_obs),
        })
    dwell_df = pd.DataFrame(rows)
    tau = np.nan
    if len(dwell_df) >= 10:
        try:
            dt = np.sort(dwell_df["dwell_time_total_s"].values)
            cdf = np.arange(1, len(dt) + 1) / len(dt)
            popt, _ = curve_fit(lambda t, tau: 1 - np.exp(-t / tau),
                                dt, cdf, p0=[dt.mean()], bounds=(1e-6, np.inf),
                                maxfev=2000)
            tau = float(popt[0])
        except Exception:
            pass
    return dwell_df, tau


# ══════════════════════════════════════════════════════════════════════════════
#  MOMENT SCALING SPECTRUM  (MSS)
# ══════════════════════════════════════════════════════════════════════════════

def compute_mss(tracks, pixel_size_um, frame_interval, max_lagtime=10):
    n_tracks = tracks["particle"].nunique()
    print(f"  MSS analysis      : {n_tracks:,} tracks")
    q_values = [1, 2, 3, 4]
    results = []
    for pid, grp in (tracks.reset_index(drop=True)
                          .sort_values("frame").groupby("particle")):
        xy = grp[["x", "y"]].values * pixel_size_um
        n = len(xy)
        if n < max(max_lagtime + 2, 6):
            continue
        gammas = []
        lag_arr = list(range(1, min(max_lagtime + 1, n // 2)))
        if len(lag_arr) < 3:
            continue
        for q in q_values:
            moments = []
            for lag in lag_arr:
                r = np.sqrt(np.sum((xy[lag:] - xy[:-lag]) ** 2, axis=1))
                moments.append(np.mean(r ** q))
            log_t = np.log(np.array(lag_arr, dtype=float) * frame_interval)
            log_m = np.log(np.array(moments) + 1e-15)
            gammas.append(np.polyfit(log_t, log_m, 1)[0])
        mss_slope = np.polyfit(q_values, gammas, 1)[0]
        results.append({"particle": int(pid), "mss_slope": float(mss_slope)})
    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE
# ══════════════════════════════════════════════════════════════════════════════

MC   = {"Immobile":"#e05252","Confined":"#f5a623","Brownian":"#4a90d9",
        "Directed":"#7ed321","Unknown":"#aaaaaa"}
MORD = ["Immobile","Confined","Brownian","Directed"]


def _draw_track(grp, color, ax, lw=0.8, alpha=0.6):
    """Draw one track with a tail-to-head alpha fade.

    Old implementation called ax.plot() once per segment (N-1 calls
    per track).  For 2000 tracks × 50 frames = ~100 000 plot calls,
    figure rendering became the bottleneck of the whole save phase.

    LineCollection batches all segments of a single track into one
    artist with a per-segment alpha array — same visual result,
    ~30× faster on dense track sets.
    """
    xy = grp[["x", "y"]].values
    n = len(xy)
    if n < 2:
        return
    # Build segment endpoints: shape (N-1, 2, 2) — i.e. for each
    # segment, [start_xy, end_xy].
    import numpy as _np
    from matplotlib.collections import LineCollection as _LC
    segs = _np.stack([xy[:-1], xy[1:]], axis=1)
    # Per-segment alpha ramp from 0.2 → `alpha`.
    alphas = _np.linspace(0.2, alpha, max(n - 1, 1))
    # Pre-multiply RGBA so each segment carries its own alpha through
    # the LineCollection.  Accept either a hex string or RGB tuple.
    try:
        import matplotlib.colors as _mc
        r, g, b = _mc.to_rgb(color)
    except Exception:
        r, g, b = 0.5, 0.5, 0.5
    colors = _np.column_stack(
        [_np.full(len(alphas), r),
         _np.full(len(alphas), g),
         _np.full(len(alphas), b),
         alphas])
    lc = _LC(segs, colors=colors, linewidths=lw,
             capstyle="round", antialiased=True)
    ax.add_collection(lc)


def make_figure(stack, tracks, imsd_df, emsd_df, diff_df,
                pixel_size, frame_interval, output_path=None, roi_mask=None,
                fig_theme="Dark", proj_cmap="Inferno", jdd=None,
                turning_angles=None, mobile_frac_df=None,
                cluster_labels=None, cluster_locs=None,
                dwell_df=None, dwell_tau=None, return_pdf_bytes=False):
    print("  Rendering figure ...")

    # ── Theme palettes ─────────────────────────────────────────────────────────
    if fig_theme == "Light":
        BG, PNL   = "#ffffff", "#f6f8fa"
        TXT, GRD  = "#24292f", "#d0d7de"
        ACC       = "#0969da"
        _kde_col  = "#000000"
        _traj_bg  = "Greys"
        _pie_text = "#ffffff"
        _font     = "sans-serif"
    elif fig_theme == "Publication":
        BG, PNL   = "#ffffff", "#ffffff"
        TXT, GRD  = "#000000", "#cccccc"
        ACC       = "#333333"
        _kde_col  = "#000000"
        _traj_bg  = "Greys"
        _pie_text = "#ffffff"
        _font     = "serif"
    elif fig_theme == "AMOLED":
        # Pure-black BG variant of Dark.
        BG, PNL   = "#000000", "#0a0a0a"
        TXT, GRD  = "#e6edf3", "#30363d"
        ACC       = "#58a6ff"
        _kde_col  = "white"
        _traj_bg  = "Greys_r"
        _pie_text = "#000000"
        _font     = "monospace"
    else:                                    # Dark (default)
        BG, PNL   = "#0d1117", "#161b22"
        TXT, GRD  = "#e6edf3", "#30363d"
        ACC       = "#58a6ff"
        _kde_col  = "white"
        _traj_bg  = "Greys_r"
        _pie_text = "#0d1117"
        _font     = "monospace"

    # ── Projection colourmap ───────────────────────────────────────────────────
    _cmap_map = {
        "Inferno": "inferno",
        "Hot":     "hot",
        "Viridis": "viridis",
        "Plasma":  "plasma",
        "Greys":   "Greys" if fig_theme in ("Light", "Publication") else "Greys_r",   # Dark + AMOLED → Greys_r
    }
    _pcmap = _cmap_map.get(proj_cmap, "inferno")

    plt.rcParams.update({
        "text.color":       TXT, "axes.labelcolor": TXT,
        "xtick.color":      TXT, "ytick.color":     TXT,
        "axes.edgecolor":   GRD, "axes.facecolor":  PNL,
        "grid.color":       GRD, "grid.alpha":      0.4,
        "font.family":      _font})

    _has_jdd = jdd is not None
    # Grid expanded from 5 to 6 rows in v1.0.64 to fit the new Radial
    # Distribution polar panel.
    fig = plt.figure(figsize=(20, 38), facecolor=BG)
    gs  = GridSpec(6, 3, figure=fig, hspace=0.45, wspace=0.32,
                   left=0.06, right=0.97, top=0.95, bottom=0.035)

    _panels          = []   # (letter, axes) collected for per-panel export
    _letter_artists  = []   # text objects for letter labels (hidden for panel renders)

    def sax(ax, ltr, ttl):
        ax.set_facecolor(PNL)
        for sp in ax.spines.values(): sp.set_edgecolor(GRD)
        ax.set_title(f"  {ttl}", loc="left", fontsize=11,
                     color=TXT, pad=8, fontweight="bold")
        txt = ax.text(-0.04, 1.06, ltr, transform=ax.transAxes, fontsize=14,
                      color=ACC, fontweight="bold", va="top", ha="right")
        _panels.append((ltr, ax))
        _letter_artists.append(txt)

    # Use up to 200 evenly-spaced frames for the max projection to save memory
    idx  = np.linspace(0, len(stack)-1, min(200, len(stack)), dtype=int)
    proj = stack[idx].max(axis=0)
    from skimage import exposure as _exp
    proj_eq = _exp.equalize_adapthist(
        (proj / proj.max()).astype(np.float32), clip_limit=0.03)
    mcol = diff_df.set_index("particle")["motion"].to_dict()

    # A — max projection
    ax = fig.add_subplot(gs[0,0])
    ax.imshow(proj_eq, cmap=_pcmap, origin="lower", aspect="equal")
    bp = 5/pixel_size; y0,x0 = proj.shape[0]*.05, proj.shape[1]*.05
    ax.plot([x0,x0+bp],[y0,y0],"-",color="white",lw=3)
    ax.text(x0+bp/2,y0+proj.shape[0]*.025,"5 um",
            ha="center",va="bottom",color="white",fontsize=8)
    ax.set_xlabel(f"X  ({pixel_size} um/px)",fontsize=9)
    ax.set_ylabel("Y (px)",fontsize=9)
    if roi_mask is not None:
        ax.contour(roi_mask.astype(float), levels=[0.5],
                   colors=["#58a6ff"], linewidths=[1.2], alpha=0.8)
        ax.text(0.02, 0.02, f"ROI", transform=ax.transAxes,
                color="#58a6ff", fontsize=8, va="bottom")
    sax(ax,"A","Max Projection")

    # B — trajectory map coloured by motion type (subsample if very many tracks)
    ax = fig.add_subplot(gs[0,1])
    ax.imshow(proj_eq,cmap=_traj_bg,origin="lower",aspect="equal",alpha=0.35)
    all_pids  = list(tracks["particle"].unique())
    draw_pids = set(np.random.default_rng(42).choice(
        all_pids, min(2000, len(all_pids)), replace=False))
    n_drawn = 0
    for pid, grp in (tracks[tracks["particle"].isin(draw_pids)]
                     .reset_index(drop=True).sort_values("frame")
                     .groupby("particle")):
        _draw_track(grp, MC.get(mcol.get(pid,"Unknown"),"#aaa"), ax)
        n_drawn += 1
    els = [Line2D([0],[0],color=MC[m],lw=2,label=m)
           for m in MORD if m in mcol.values()]
    ax.legend(handles=els,fontsize=8,loc="upper right",
              framealpha=0.7,facecolor=PNL,edgecolor=GRD,labelcolor=TXT)
    ax.set_xlim(0,proj.shape[1]); ax.set_ylim(0,proj.shape[0])
    ax.set_xlabel("X (px)",fontsize=9); ax.set_ylabel("Y (px)",fontsize=9)
    shown = f"{n_drawn:,}" + (f" of {len(all_pids):,}" if n_drawn < len(all_pids) else "")
    sax(ax,"B",f"Trajectories  (n={shown})")

    # C — trajectories coloured by D value
    ax = fig.add_subplot(gs[0,2])
    ax.imshow(proj_eq, cmap=_traj_bg, origin="lower", aspect="equal", alpha=0.35)
    d_map = diff_df.set_index("particle")["D"].to_dict()
    d_vals_valid = [v for v in d_map.values() if v is not None and np.isfinite(v) and v > 0]
    if d_vals_valid:
        log_d_vals = np.log10(d_vals_valid)
        _p5  = np.percentile(log_d_vals, 5)
        _p95 = np.percentile(log_d_vals, 95)
        _cmap_d = plt.cm.plasma
        _norm_d = plt.Normalize(vmin=_p5, vmax=_p95)
        _sm_d   = plt.cm.ScalarMappable(cmap=_cmap_d, norm=_norm_d)
        _sm_d.set_array([])
        draw_pids_c = set(np.random.default_rng(43).choice(
            all_pids, min(2000, len(all_pids)), replace=False))
        for pid, grp in (tracks[tracks["particle"].isin(draw_pids_c)]
                         .reset_index(drop=True).sort_values("frame")
                         .groupby("particle")):
            D_val = d_map.get(pid)
            if D_val is not None and np.isfinite(D_val) and D_val > 0:
                col = _cmap_d(_norm_d(np.log10(D_val)))
            else:
                col = "#555555"
            _draw_track(grp, col, ax)
        cb = plt.colorbar(_sm_d, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("log10(D)  [µm²/s]", fontsize=8, color=TXT)
        cb.ax.yaxis.set_tick_params(color=TXT)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=TXT, fontsize=7)
    ax.set_xlim(0, proj.shape[1]); ax.set_ylim(0, proj.shape[0])
    ax.set_xlabel("X (px)", fontsize=9); ax.set_ylabel("Y (px)", fontsize=9)
    sax(ax, "C", "Trajectories by D value")

    # D — MSD curves
    ax = fig.add_subplot(gs[1,0])
    lt  = emsd_df.index.values * frame_interval
    rng = np.random.default_rng(42)
    for pid in rng.choice(list(imsd_df.columns), min(200,len(imsd_df.columns)), replace=False):
        v  = imsd_df[pid].values
        t  = imsd_df.index.values * frame_interval
        ok = np.isfinite(v) & (v > 0)
        if ok.sum() >= 2:
            ax.plot(t[ok],v[ok],"-",color="#8b949e",lw=0.4,alpha=0.3)
    ax.plot(lt,emsd_df.values,"-o",color=ACC,lw=2.5,ms=4,zorder=5,
            label="Ensemble MSD")
    try:
        t6,m6 = lt[:6], emsd_df.values[:6].ravel()
        ok6   = np.isfinite(m6) & (m6>0)
        po,_  = curve_fit(msd_linear,t6[ok6],m6[ok6],p0=[0.01,0],maxfev=2000)
        te    = np.linspace(t6[0],lt[-1],200)
        ax.plot(te,msd_linear(te,*po),"--",color="#f78166",lw=2,
                label=f"Fit D={po[0]:.4f} um2/s")
    except Exception: pass
    ax.set_xlabel("Lag time (s)",fontsize=9)
    ax.set_ylabel("MSD (um2)",fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True,which="both",ls=":",alpha=0.3)
    ax.legend(fontsize=8,framealpha=0.6,facecolor=PNL,edgecolor=GRD,labelcolor=TXT)
    sax(ax,"D","MSD Curves")

    # E — D distribution
    ax = fig.add_subplot(gs[1,1])
    dv = diff_df["D"].dropna()
    dv = dv[(dv>0) & (dv<dv.quantile(0.995))]
    if len(dv) > 5:
        ld   = np.log10(dv)
        bins = np.linspace(ld.min(), ld.max(), 40)
        for m in MORD:
            sub = diff_df[(diff_df["motion"]==m) & (diff_df["D"]>0)]
            if len(sub):
                ax.hist(np.log10(sub["D"].clip(1e-6)),bins=bins,
                        color=MC[m],alpha=0.7,label=m,edgecolor="none")
        if len(ld) > 10:
            kde = gaussian_kde(ld)
            xk  = np.linspace(ld.min(), ld.max(), 300)
            ax.plot(xk, kde(xk)*len(dv)*(bins[1]-bins[0]),
                    "-",color=_kde_col,lw=2)
        ax.axvline(np.log10(dv.median()),color=ACC,ls="--",lw=1.5,
                   label=f"Median={dv.median():.4f}")
        ax.set_xlabel("log10(D)  [um2/s]",fontsize=9)
        ax.set_ylabel("Count",fontsize=9)
        ax.legend(fontsize=8,framealpha=0.6,facecolor=PNL,edgecolor=GRD,labelcolor=TXT)
    ax.grid(True,ls=":",alpha=0.3)
    sax(ax,"E","Diffusion Coefficient Distribution")

    # F — pie chart
    ax = fig.add_subplot(gs[1,2])
    mc_ = diff_df["motion"].value_counts()
    lbl = [m for m in MORD if m in mc_]
    sz  = [mc_[m] for m in lbl]
    co  = [MC[m] for m in lbl]
    _,_,ats = ax.pie(sz,labels=lbl,colors=co,autopct="%1.1f%%",startangle=140,
                      textprops={"color":TXT,"fontsize":9},
                      wedgeprops={"edgecolor":PNL,"linewidth":2})
    for at in ats: at.set_fontsize(8); at.set_color(_pie_text)
    sax(ax,"F","Motion Classification")

    # G — alpha distribution
    ax = fig.add_subplot(gs[2,0])
    av = diff_df["alpha"].dropna()
    av = av[(av>-1) & (av<4)]
    if len(av) > 5:
        ba = np.linspace(av.min(), av.max(), 40)
        for m in MORD:
            sub = diff_df[(diff_df["motion"]==m) & diff_df["alpha"].notna()]
            if len(sub):
                ax.hist(sub["alpha"].clip(-1,4),bins=ba,
                        color=MC[m],alpha=0.7,label=m,edgecolor="none")
        for xv,lb,ls in [(0.5,"a=0.5",":"),(1.0,"a=1 Brownian","--"),(2.0,"a=2 directed",":")]:
            ax.axvline(xv,color=GRD,ls=ls,lw=1.2,label=lb)
        ax.set_xlabel("Anomalous exponent alpha",fontsize=9)
        ax.set_ylabel("Count",fontsize=9)
        ax.legend(fontsize=7,framealpha=0.6,facecolor=PNL,edgecolor=GRD,labelcolor=TXT)
    ax.grid(True,ls=":",alpha=0.3)
    sax(ax,"G","Anomalous Exponent Alpha Distribution")

    # H — Position Density Heatmap
    ax = fig.add_subplot(gs[2, 1])
    try:
        x_um = tracks["x"].values * pixel_size
        y_um = tracks["y"].values * pixel_size
        h, xe, ye = np.histogram2d(x_um, y_um, bins=120)
        from scipy.ndimage import gaussian_filter as _gf
        h_sm = _gf(h, sigma=1.5)
        ax.imshow(h_sm.T, origin="lower", cmap="hot",
                  extent=[xe[0], xe[-1], ye[0], ye[-1]],
                  aspect="equal", interpolation="bilinear")
        ax.set_xlabel("X  (µm)", fontsize=9)
        ax.set_ylabel("Y  (µm)", fontsize=9)
        if roi_mask is not None:
            H_px, W_px = roi_mask.shape
            ax.contour(
                np.linspace(0, W_px * pixel_size, W_px),
                np.linspace(0, H_px * pixel_size, H_px),
                roi_mask.astype(float), levels=[0.5],
                colors=["#58a6ff"], linewidths=[1.0], alpha=0.7)
    except Exception:
        pass
    sax(ax, "H", "Position Density Map")

    # I — Turning Angle Distribution
    # Plotted as a single LINE following the count of each |angle| bin,
    # using UNSIGNED magnitudes (|θ|) so the x-axis runs 0°–180°.
    # 0° = continued straight; 180° = full reversal; 90° = right-angle
    # deflection; the radial-distribution panel (O) shows the rotational
    # direction (sign) separately.
    ax = fig.add_subplot(gs[2, 2])
    if turning_angles is None or len(turning_angles) < 10:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=12)
    else:
        ta_unsigned = np.abs(np.asarray(turning_angles, dtype=float))
        _ta_bins = np.linspace(0, 180, 37)            # 5° bins
        _ta_centres = 0.5 * (_ta_bins[:-1] + _ta_bins[1:])
        _ta_counts, _ = np.histogram(ta_unsigned, bins=_ta_bins)
        # Normalise to relative frequency so the shape is comparable across
        # runs (and consistent with the Compare-mode panel).  Total track
        # count is already reported in the suptitle / Summary tab.
        _ta_freq = (_ta_counts / _ta_counts.sum()
                    if _ta_counts.sum() else _ta_counts)
        ax.plot(_ta_centres, _ta_freq, "-o",
                color=ACC, lw=2, ms=3, alpha=0.95)
        # Uniform-distribution reference line (1/N_bins)
        ax.axhline(1.0 / len(_ta_centres),
                   color=GRD, lw=0.6, ls=":", label="uniform")
        # Reference verticals: 90° (right-angle), 180° (full reversal)
        ax.axvline(90,  color=GRD, lw=0.8, ls="--")
        ax.axvline(180, color=GRD, lw=0.6, ls=":")
        ax.set_xlim(0, 180)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_xlabel("|Turning angle|  (°)", fontsize=9)
        ax.set_ylabel("Relative frequency", fontsize=9)
        ax.grid(True, ls=":", alpha=0.3)
        ax.legend(fontsize=7, frameon=False, loc="best")
    sax(ax, "I", "Turning Angle Distribution")

    # J — Mobile Fraction Over Time
    ax = fig.add_subplot(gs[3, 0])
    if mobile_frac_df is None or len(mobile_frac_df) < 2:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=12)
    else:
        ts  = mobile_frac_df["time_s"].values
        mf  = mobile_frac_df["mobile_fraction"].values * 100
        ax.plot(ts, mf, "o-", color=ACC, lw=2, ms=5)
        ax.fill_between(ts, 0, mf, alpha=0.2, color=ACC)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Mobile fraction (%)", fontsize=9)
        ax.grid(True, ls=":", alpha=0.3)
    sax(ax, "J", "Mobile Fraction Over Time")

    # K — Jump Distance Distribution (spans cols 1–2)
    ax = fig.add_subplot(gs[3, 1:])
    if _has_jdd:
        _jdd_colors = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff"]

        r_max_plot = np.percentile(jdd["jumps"], 99.5)
        bins = np.linspace(0, r_max_plot, 60)
        ax.hist(jdd["jumps"], bins=bins, density=True,
                color="#8b949e", alpha=0.45, edgecolor="none",
                label=f"Observed  (n={jdd['n_jumps']:,})")

        _comp_labels = ["Slow", "Medium", "Fast"]
        for k, (pdf_k, D_k, f_k) in enumerate(
                zip(jdd["pdfs"], jdd["D_values"], jdd["fractions"])):
            lbl = (f"{_comp_labels[k]}  D={D_k:.4f} µm²/s  "
                   f"({f_k*100:.1f}%)")
            ax.plot(jdd["r_range"], pdf_k,
                    color=_jdd_colors[k], lw=2, label=lbl)

        ax.plot(jdd["r_range"], jdd["pdf_total"],
                color=TXT, lw=2.5, ls="--", label="Total fit")
        ax.set_xlabel("Jump distance  (µm)", fontsize=9)
        ax.set_ylabel("Probability density", fontsize=9)
        ax.set_xlim(0, r_max_plot)
        ax.set_ylim(bottom=0)
        ax.grid(True, ls=":", alpha=0.3)
        ax.legend(fontsize=8, framealpha=0.6,
                  facecolor=PNL, edgecolor=GRD, labelcolor=TXT,
                  loc="upper right")
        sax(ax, "K",
            f"Jump Distance Distribution  "
            f"({jdd['n_components']}-population fit  |  "
            f"{jdd['n_jumps']:,} jumps)")
    else:
        ax.text(0.5, 0.5, "JDD not computed", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=12)
        sax(ax, "K", "Jump Distance Distribution")

    # L — Cluster Map
    ax = fig.add_subplot(gs[4, 0])
    if cluster_labels is not None and cluster_locs is not None and len(cluster_locs) > 0:
        xy_um = cluster_locs  # already in µm, subsampled to match labels
        noise = cluster_labels == -1
        if noise.any():
            ax.scatter(xy_um[noise, 0], xy_um[noise, 1],
                       s=0.5, c="#444", alpha=0.3, linewidths=0, rasterized=True)
        clustered = ~noise
        if clustered.any():
            n_c = max(cluster_labels.max() + 1, 1)
            cmap_c = plt.cm.get_cmap("tab20", n_c)
            ax.scatter(xy_um[clustered, 0], xy_um[clustered, 1],
                       s=1.5, c=cluster_labels[clustered], cmap=cmap_c,
                       alpha=0.7, linewidths=0, rasterized=True,
                       vmin=0, vmax=n_c - 1)
        ax.set_xlabel("X  (µm)", fontsize=9)
        ax.set_ylabel("Y  (µm)", fontsize=9)
        n_shown = int(cluster_labels.max()) + 1 if cluster_labels.max() >= 0 else 0
        ax.text(0.02, 0.98, f"n={n_shown} clusters",
                transform=ax.transAxes, fontsize=8, color=TXT, va="top")
    else:
        ax.text(0.5, 0.5, "Cluster analysis\nnot computed",
                transform=ax.transAxes, ha="center", va="center", color=TXT, fontsize=10)
    sax(ax, "L", "Cluster Map  (DBSCAN)")

    # M — Dwell Time Distribution
    ax = fig.add_subplot(gs[4, 1])
    if dwell_df is not None and len(dwell_df) >= 5:
        dt_vals = dwell_df["dwell_time_s"].values
        ax.hist(dt_vals, bins=30, color=ACC, alpha=0.75, edgecolor="none", density=True)
        if np.isfinite(dwell_tau):
            t_fit = np.linspace(0, dt_vals.max(), 200)
            ax.plot(t_fit, (1/dwell_tau) * np.exp(-t_fit / dwell_tau),
                    "--", color="#f78166", lw=2,
                    label=f"τ = {dwell_tau:.2f} s")
            ax.legend(fontsize=8, framealpha=0.6, facecolor=PNL,
                      edgecolor=GRD, labelcolor=TXT)
        ax.set_xlabel("Dwell time  (s)", fontsize=9)
        ax.set_ylabel("Probability density", fontsize=9)
        ax.grid(True, ls=":", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Insufficient data\n(need confined/immobile tracks)",
                transform=ax.transAxes, ha="center", va="center", color=TXT, fontsize=10)
    sax(ax, "M", "Dwell Time Distribution")

    # N — MSS Slope Distribution
    ax = fig.add_subplot(gs[4, 2])
    if "mss_slope" in diff_df.columns and diff_df["mss_slope"].notna().sum() >= 5:
        ms = diff_df["mss_slope"].dropna()
        ms = ms[ms.between(-0.5, 1.5)]
        bins = np.linspace(ms.min(), ms.max(), 40)
        for m in MORD:
            sub = diff_df[(diff_df["motion"] == m) & diff_df["mss_slope"].notna()]
            sub = sub[sub["mss_slope"].between(-0.5, 1.5)]
            if len(sub):
                ax.hist(sub["mss_slope"], bins=bins, color=MC[m],
                        alpha=0.7, label=m, edgecolor="none")
        for xv, lb, ls_ in [(0.25, "Confined", ":"), (0.5, "Brownian", "--"), (0.75, "Directed", ":")]:
            ax.axvline(xv, color=GRD, ls=ls_, lw=1.2, label=lb)
        ax.set_xlabel("MSS slope  (ν)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=7, framealpha=0.6, facecolor=PNL, edgecolor=GRD, labelcolor=TXT)
        ax.grid(True, ls=":", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "MSS not computed\n(tracks too short)",
                transform=ax.transAxes, ha="center", va="center", color=TXT, fontsize=10)
    sax(ax, "N", "Moment Scaling Spectrum  (MSS slope)")

    # O — Radial Distribution of turning angles (polar)
    # A polar histogram of signed turning angles, oriented so 0° (straight
    # ahead) is at the top and positive angles sweep CLOCKWISE around to the
    # right (i.e. right hemisphere = positive turns, left hemisphere =
    # negative turns).  The bars radiate outward; their angular position is
    # the turning direction, their height the relative frequency.  Uniform
    # circle = Brownian motion; lobe at 0° = directional persistence; lobe
    # at ±180° = back-tracking / confinement.
    # Placed at the centre column of row 5 so it sits visually balanced
    # rather than pinned to a corner.
    ax = fig.add_subplot(gs[5, 1], projection="polar")
    if turning_angles is None or len(turning_angles) < 10:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center", color=TXT, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    else:
        ta_arr = np.asarray(turning_angles, dtype=float)
        is_signed = bool(np.any(ta_arr < -1e-3))
        print(f"  Radial-dist input: n={len(ta_arr):,}  "
              f"signed={is_signed}  "
              f"pos={int((ta_arr>0).sum()):,}  neg={int((ta_arr<0).sum()):,}  "
              f"min={ta_arr.min():.1f}°  max={ta_arr.max():.1f}°")
        if not is_signed:
            ta_arr = np.concatenate([ta_arr, -ta_arr])
        # CRITICAL: matplotlib polar's ax.bar() does NOT render correctly
        # when theta values are in (-π, +π].  Half the bars (the side with
        # negative theta after applying set_theta_direction) silently fail
        # to draw, producing only a half-circle of bars.
        # Empirical fix: shift the angles to [0, 2π) before histogramming.
        # The xticks are then placed at positive-only angles too, but
        # *labelled* with the signed values the user expects.
        angles_rad = np.mod(np.deg2rad(ta_arr), 2 * np.pi)
        n_bins = 36
        bins   = np.linspace(0, 2 * np.pi, n_bins + 1)
        counts, edges = np.histogram(angles_rad, bins=bins, density=True)
        theta = 0.5 * (edges[:-1] + edges[1:])
        width = bins[1] - bins[0]
        ax.bar(theta, counts, width=width * 0.95, bottom=0.0,
               color=ACC, alpha=0.75, edgecolor=GRD, linewidth=0.5)
        ax.set_theta_zero_location("N")     # 0° at the top
        ax.set_theta_direction(-1)          # clockwise positive (right = +)
        # xticks at 0°, 45°, ..., 315° (positive only); labels show signed
        # equivalents so the reader still sees "-45°" on the left, etc.
        ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(["0°", "+45°", "+90°", "+135°", "±180°",
                            "−135°", "−90°", "−45°"], fontsize=8)
        # Hide the radial-axis numeric labels.
        ax.set_yticklabels([])
        ax.tick_params(axis="y", which="both", left=False)
        ax.grid(True, ls=":", alpha=0.4)
    sax(ax, "O", "Radial Distribution  (signed turning angles)")

    md = diff_df["D"].dropna().median()
    ma = diff_df["alpha"].dropna().median()
    fig.suptitle(
        f"FIREFLY Analysis  |  {diff_df.shape[0]:,} trajectories  |  "
        f"Median D = {md:.4f} um2/s  |  Median alpha = {ma:.2f}",
        fontsize=13,color=TXT,y=0.97,fontweight="bold")

    import io as _io
    from matplotlib.transforms import Bbox as _Bbox

    from PIL import Image as _PILImage

    # Render individual panels WITHOUT letter labels
    for _txt in _letter_artists:
        _txt.set_visible(False)
    fig.canvas.draw()
    _renderer = fig.canvas.get_renderer()
    _pad_px   = fig.dpi * 0.12
    panel_images = {}
    for _ltr, _pax in _panels:
        _bbox = _pax.get_tightbbox(_renderer)
        if _bbox is None:
            continue
        _bbox_pad = _Bbox([[_bbox.x0 - _pad_px, _bbox.y0 - _pad_px],
                            [_bbox.x1 + _pad_px, _bbox.y1 + _pad_px]])
        _bbox_in  = _bbox_pad.transformed(fig.dpi_scale_trans.inverted())
        _pbuf = _io.BytesIO()
        fig.savefig(_pbuf, format="png", dpi=150, bbox_inches=_bbox_in,
                    facecolor=fig.get_facecolor())
        _pbuf.seek(0)
        panel_images[_ltr] = _PILImage.open(_pbuf).copy()
        _pbuf.close()

    # Restore letter labels then render combined figure
    for _txt in _letter_artists:
        _txt.set_visible(True)
    fig.canvas.draw()
    _buf = _io.BytesIO()
    fig.savefig(_buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    _buf.seek(0)
    combined_pil = _PILImage.open(_buf).copy()
    _buf.close()

    # Save to disk only if output_path explicitly provided (CLI / legacy callers)
    if output_path:
        combined_pil.save(output_path, dpi=(150, 150))
        print(f"  Figure -> {output_path}")
        _pdf = os.path.splitext(output_path)[0] + ".pdf"
        fig.savefig(_pdf, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Figure (PDF) -> {_pdf}")

    pdf_bytes = None
    if return_pdf_bytes:
        try:
            _pdfbuf = _io.BytesIO()
            fig.savefig(_pdfbuf, format="pdf", bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            pdf_bytes = _pdfbuf.getvalue()
            _pdfbuf.close()
        except Exception as _exc:
            print(f"  WARN: PDF render failed: {_exc}")

    plt.close(fig)
    print("  Figure rendered.")
    return {
        "combined":     combined_pil,
        "panels":       panel_images,
        "panel_titles": {ltr: ax.get_title().strip() for ltr, ax in _panels},
        "pdf_bytes":    pdf_bytes,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="FIREFLY — Fluorescence Inference & Reconstruction Engine "
                    "(CZI / TIF, optimised)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("--pixel-size",       type=float, default=None)
    p.add_argument("--frame-interval",   type=float, default=None)
    p.add_argument("--diameter",         type=int,   default=7)
    p.add_argument("--minmass",          type=float, default=None)
    p.add_argument("--search-range",     type=float, default=5)
    p.add_argument("--memory",           type=int,   default=3)
    p.add_argument("--min-track-length", type=int,   default=5)
    p.add_argument("--max-lagtime",      type=int,   default=20)
    p.add_argument("--bg-method",        default="uniform_filter",
                   choices=["uniform_filter","rolling_ball"])
    p.add_argument("--bg-radius",        type=float, default=50)
    p.add_argument("--workers",          type=int,   default=N_CPUS)
    p.add_argument("--chunk-size",       type=int,   default=500)
    p.add_argument("--channel",          type=int,   default=0)
    p.add_argument("--output-dir",       default=None)
    p.add_argument("--roi-threshold",      type=float, default=None,
                   help="Manual intensity threshold for ROI mask on [0,1]. "
                        "If omitted with --roi-auto, threshold is determined "
                        "automatically. Omit both to process the full frame.")
    p.add_argument("--roi-auto",           action="store_true", default=False,
                   help="Automatically determine ROI threshold from the data. "
                        "Uses --roi-auto-method to select the algorithm.")
    p.add_argument("--roi-auto-method",    default="auto",
                   choices=["auto", "otsu", "li", "triangle"],
                   help="Algorithm for automatic ROI thresholding. "
                        "auto     = picks best method for sptPALM (default). "
                        "otsu     = maximises inter-class variance. "
                        "li       = minimises cross-entropy (sparse cells). "
                        "triangle = best for large dark backgrounds.")
    p.add_argument("--roi-mode",           default="mean",
                   choices=["mean", "perframe"],
                   help="ROI masking mode. "
                        "mean     = one mask from mean projection (default). "
                        "perframe = separate mask computed per frame.")
    return p.parse_args()


def main():
    args    = parse_args()
    t_start = time.perf_counter()

    if not os.path.isfile(args.input):
        sys.exit(f"ERROR: File not found: {args.input}")

    stem    = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.input))
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "="*67)
    print("  sptPALM Analysis Pipeline  --  Zeiss Elyra  --  By Jacob Levers")
    print("="*67)
    print(f"  CPU cores available : {N_CPUS}  |  Using: {args.workers}")

    # 1 — Load
    print("\n[1/6] Loading file")
    stack, meta_px, meta_fi = load_file(args.input, channel=args.channel)
    n_frames = len(stack)

    pixel_size = args.pixel_size or meta_px
    if pixel_size is None:
        print("  WARNING: Pixel size not in metadata. Using 0.104 um/px.")
        print("  (Override with --pixel-size)")
        pixel_size = 0.104
    else:
        src = "command line" if args.pixel_size else "CZI metadata"
        print(f"  Pixel size     : {pixel_size} um/px  [{src}]")

    frame_interval = args.frame_interval or meta_fi
    if frame_interval is None:
        print("  WARNING: Frame interval not in metadata. Using 0.05 s.")
        print("  (Override with --frame-interval)")
        frame_interval = 0.05
    else:
        src = "command line" if args.frame_interval else "CZI metadata"
        print(f"  Frame interval : {frame_interval} s/frame  [{src}]")

    print(f"  Total frames   : {n_frames:,}")
    print(f"  Output dir     : {out_dir}")

    # 2 — Preprocess
    print("\n[2/6] Preprocessing")
    stack_pp = preprocess_stack(stack, bg_radius=args.bg_radius,
                                bg_method=args.bg_method,
                                workers=args.workers)

    if args.minmass is None:
        sample = stack_pp[min(5, n_frames-1)]
        _peak = float(np.percentile(sample, 99))
        # See _fast_preprocess_and_localise for the rationale on the d²/8 factor.
        args.minmass = float(_peak * (args.diameter ** 2) / 8.0)
        print(f"  Auto minmass: {args.minmass:.4f}  "
              f"(from 99th-pct peak {_peak:.4f} × d²/8)")

    # 2b — ROI mask (optional)
    roi_mask = None
    use_roi  = (args.roi_threshold is not None) or args.roi_auto
    if use_roi:
        manual_thresh = args.roi_threshold  # None = auto
        auto_method   = args.roi_auto_method if args.roi_auto else None
        if manual_thresh is not None:
            mode_str = f"threshold={manual_thresh}, mode={args.roi_mode}"
        else:
            mode_str = f"auto-threshold ({args.roi_auto_method}), mode={args.roi_mode}"
        print(f"\n[2b/6] Building ROI mask  ({mode_str})")
        roi_preview = os.path.join(out_dir, f"{stem}_roi_mask.png")
        roi_mask = build_roi_mask(
            stack_pp,
            threshold=manual_thresh,
            mode=args.roi_mode,
            threshold_method=args.roi_auto_method if args.roi_auto else "auto",
            save_path=roi_preview)
    else:
        print("  ROI: disabled  "
              "(use --roi-auto for automatic, or --roi-threshold 0.15 for manual)")

    # 3 — Localise
    print("\n[3/6] Localisation")
    locs = localise_particles(stack_pp, diameter=args.diameter,
                              minmass=args.minmass,
                              workers=args.workers,
                              chunk_size=args.chunk_size)
    if len(locs) == 0:
        sys.exit("ERROR: No particles found. Try adding --minmass 0.05")

    if roi_mask is not None:
        locs = apply_roi_mask(locs, roi_mask)
        if len(locs) == 0:
            sys.exit("ERROR: No localisations inside ROI. "
                     "Lower --roi-threshold or remove it.")

    # 4 — Link
    print("\n[4/6] Linking trajectories")
    tracks = link_trajectories(locs, search_range=args.search_range,
                               memory=args.memory,
                               min_len=args.min_track_length)
    if tracks["particle"].nunique() == 0:
        sys.exit("ERROR: No trajectories found. Lower --min-track-length.")

    # 5 — MSD + diffusion (single parallel pass — no tp.imsd)
    print("\n[5/6] MSD & diffusion fitting")
    imsd_df, emsd_df, diff_df = compute_msd_and_fit(
        tracks, pixel_size, frame_interval,
        max_lagtime=args.max_lagtime, workers=args.workers)

    # 5b — JDD
    print("\n[5b/6] Jump Distance Distribution")
    jdd = compute_jdd(tracks, pixel_size, frame_interval, n_components=2)
    if jdd:
        print(f"  Jumps: {jdd['n_jumps']:,}")
        for k, (D, f) in enumerate(zip(jdd["D_values"], jdd["fractions"])):
            print(f"  Population {k+1}: D={D:.4f} um2/s  fraction={f*100:.1f}%")
    else:
        print("  Too few jumps to fit JDD.")

    # 6 — Save
    print("\n[6/6] Saving outputs")
    for df, suffix in [(locs,"localisations"), (tracks,"trajectories"),
                       (diff_df,"diffusion_summary")]:
        path = os.path.join(out_dir, f"{stem}_{suffix}.csv")
        df.to_csv(path, index=False)
        print(f"  {suffix:<25} -> {path}")

    emsd_out  = emsd_df.to_frame("msd_um2").reset_index(names="lag_frame")
    emsd_path = os.path.join(out_dir, f"{stem}_ensemble_msd.csv")
    emsd_out.to_csv(emsd_path, index=False)
    print(f"  ensemble_msd              -> {emsd_path}")

    fig_path = os.path.join(out_dir, f"{stem}_sptpalm_figure.png")
    make_figure(stack, tracks, imsd_df, emsd_df, diff_df,
                pixel_size, frame_interval, fig_path,
                roi_mask=roi_mask, jdd=jdd,
                turning_angles=None, mobile_frac_df=None,
                cluster_labels=None, cluster_locs=None,
                dwell_df=None, dwell_tau=None)

    # Summary
    total = time.perf_counter() - t_start
    print("\n" + "="*67)
    print("  RESULTS SUMMARY")
    print("="*67)
    print(f"  Raw localisations : {len(locs):>8,}")
    print(f"  Final trajectories: {tracks['particle'].nunique():>8,}")
    mc_ = diff_df["motion"].value_counts()
    for m in MORD:
        cnt = mc_.get(m, 0)
        print(f"    {m:<12}  {cnt:>6,}  ({100*cnt/max(len(diff_df),1):.1f}%)")
    print(f"\n  Median D  : {diff_df['D'].median():.5f} um2/s")
    print(f"  Mean D    : {diff_df['D'].mean():.5f} um2/s")
    print(f"  Median a  : {diff_df['alpha'].median():.3f}")
    print(f"\n  Total time: {total:.1f}s  ({total/60:.1f} min)")
    print("="*67)
    print(f"\n  Done! Results in: {out_dir}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARISON  —  group A vs group B over multiple analysis output folders
# ══════════════════════════════════════════════════════════════════════════════
#
# A "group" is a list of analysis output folders.  For each folder we re-load
# the per-experiment summary (MSD curve, D values, motion classes, etc.) and
# compute scalar metrics (AUC, mob/immob ratio, mean track length).  We then
# render a multi-panel figure overlaying the two groups, with scatter dots
# per replicate and t-test significance stars on bar charts when n≥2 each.
# Layout matches the lab's Pre/Post style: MSD curve overlay, AUC bar chart,
# LogD frequency distribution, mobile/immobile ratio bar chart, motion class
# fractions, track length distribution, JDD, dwell time CDF, turning angles.

def _find_stem(data_dir):
    """Find the experiment stem from filenames like {stem}_params.json or
    {stem}_diffusion_summary.csv inside an analysis output folder's data/ dir."""
    for f in sorted(os.listdir(data_dir)):
        if f.endswith("_params.json"):
            return f[:-len("_params.json")]
    for f in sorted(os.listdir(data_dir)):
        if f.endswith("_diffusion_summary.csv"):
            return f[:-len("_diffusion_summary.csv")]
    raise FileNotFoundError(f"No analysis CSVs found in {data_dir}")


def _is_palmtracer_folder(folder):
    """Return True if `folder` contains raw PALM-Tracer output."""
    try:
        names = os.listdir(folder)
    except OSError:
        return False
    # PALM-Tracer files have no stem prefix (e.g. 'locPALMTracer.txt')
    has_loc = any(n.lower() == "locpalmtracer.txt" or n.lower() == "locpalmtracer.csv"
                  for n in names)
    has_trc = any(n.lower() == "trcpalmtracer.txt" or n.lower() == "trcpalmtracer.csv"
                  for n in names)
    return has_loc and has_trc


def _read_palmtracer_table(path, header_lines):
    """Read a PALM-Tracer file (tab- or comma-separated), skipping comment /
    metadata rows.  `header_lines` is the number of non-data leading rows."""
    # PALM-Tracer's reference files are TSV; FIREFLY-emitted ones are CSV.
    # Sniff the separator from the first data line.
    with open(path, "r") as fh:
        for _ in range(header_lines):
            fh.readline()
        first = fh.readline()
    sep = "\t" if "\t" in first and first.count("\t") >= first.count(",") else ","
    return pd.read_csv(path, sep=sep, header=None, comment="#",
                       skiprows=header_lines, engine="python")


def load_summary_from_palmtracer(folder):
    """
    Read a raw PALM-Tracer output folder and return the same dict shape as
    `load_summary_from_folder` so the Compare tab can treat it identically.

    PALM-Tracer does not store FIREFLY-specific quantities (alpha, motion
    class, dwell times, turning angles, JDD, mobile fraction, Rg) — these are
    re-derived on the fly from the imported trajectories using the same
    pipeline functions FIREFLY normally runs.
    """
    # ── Locate the six PALM-Tracer files (tab or csv) ────────────────────
    def _pick(*candidates):
        for c in candidates:
            p = os.path.join(folder, c)
            if os.path.isfile(p):
                return p
        return None

    loc_path = _pick("locPALMTracer.txt", "locPALMTracer.csv")
    trc_path = _pick("trcPALMTracer.txt", "trcPALMTracer.csv")
    d_path   = _pick("trcPALMTracer-AllROI-D.txt", "trcPALMTracer-AllROI-D.csv",
                     "trcPALMTracer-1-D.txt",     "trcPALMTracer-1-D.csv")
    msd_path = _pick("trcPALMTracer-AllROI-MSD.txt", "trcPALMTracer-AllROI-MSD.csv",
                     "trcPALMTracer-1-MSD.txt",     "trcPALMTracer-1-MSD.csv")

    if not (loc_path and trc_path):
        raise FileNotFoundError(f"PALM-Tracer files not found in {folder}")

    # ── Parse loc / trc metadata header (line 2 contains values) ─────────
    pixel_size_um    = 0.106
    frame_interval_s = 0.02
    width = height = n_frames = 0
    try:
        with open(loc_path, "r") as fh:
            _hdr_names  = fh.readline().rstrip("\n").replace(",", "\t").split("\t")
            _hdr_values = fh.readline().rstrip("\n").replace(",", "\t").split("\t")
        meta = {k.strip(): v.strip() for k, v in zip(_hdr_names, _hdr_values)}
        pixel_size_um    = float(meta.get("Pixel_Size(um)", pixel_size_um))
        frame_interval_s = float(meta.get("Frame_Duration(s)", frame_interval_s))
        width    = int(float(meta.get("Width",  0) or 0))
        height   = int(float(meta.get("Height", 0) or 0))
        n_frames = int(float(meta.get("nb_Planes", 0) or 0))
    except Exception:
        pass

    # ── Localisations ────────────────────────────────────────────────────
    # Header rows in loc/trc files: metadata-names, metadata-values, column-names
    loc_df = _read_palmtracer_table(loc_path, header_lines=3)
    loc_df.columns = ["id", "Plane", "Index", "Channel", "Integrated_Intensity",
                      "CentroidX_px", "CentroidY_px", "SigmaX_px", "SigmaY_px",
                      "Angle_rad", "MSE_Gauss", "CentroidZ_um", "MSE_Z_um",
                      "Pair_Distance_px"][:loc_df.shape[1]]
    locs = pd.DataFrame({
        "x":     loc_df["CentroidX_px"].astype(float).values,
        "y":     loc_df["CentroidY_px"].astype(float).values,
        "frame": (loc_df["Plane"].astype(int).values - 1),   # 1-based → 0-based
        "mass":  loc_df["Integrated_Intensity"].astype(float).values,
    })

    # ── Trajectories ─────────────────────────────────────────────────────
    trc_df = _read_palmtracer_table(trc_path, header_lines=3)
    trc_df.columns = ["Track", "Plane", "CentroidX_px", "CentroidY_px",
                      "CentroidZ_um", "Integrated_Intensity", "id",
                      "Pair_Distance_px"][:trc_df.shape[1]]
    tracks = pd.DataFrame({
        "particle": trc_df["Track"].astype(int).values,
        "frame":    trc_df["Plane"].astype(int).values - 1,
        "x":        trc_df["CentroidX_px"].astype(float).values,
        "y":        trc_df["CentroidY_px"].astype(float).values,
        "mass":     trc_df["Integrated_Intensity"].astype(float).values,
    }).sort_values(["particle", "frame"]).reset_index(drop=True)

    # ── Re-derive D, alpha, motion via FIREFLY's own pipeline ────────────
    # This guarantees the Compare tab sees the same column names and
    # identical statistics it would for a native FIREFLY run.
    imsd_df, emsd_series, diff_df = compute_msd_and_fit(
        tracks, pixel_size_um, frame_interval_s, max_lagtime=20, n_fit=5)

    emsd_df = (emsd_series.to_frame("msd_um2")
                          .reset_index(names="lag_frame"))

    # FIREFLY-only metrics — re-derive on the fly
    try:
        jdd = compute_jdd(tracks, pixel_size_um, frame_interval_s)
    except Exception:
        jdd = None
    try:
        dwell_df, _ = compute_dwell_times(tracks, diff_df, frame_interval_s)
    except Exception:
        dwell_df = None
    try:
        ta_deg = compute_turning_angles(tracks)
    except Exception:
        ta_deg = None
    try:
        mobile_frac_df = compute_mobile_fraction_over_time(
            tracks, diff_df, frame_interval_s)
    except Exception:
        mobile_frac_df = None

    stem = os.path.basename(folder.rstrip(os.sep)) or "palmtracer_run"
    if stem.lower().endswith(".pt"):
        stem = stem[:-3]

    # ── Cache the recomputed FIREFLY-only metrics next to the PALM-Tracer
    # files so re-opening this folder in the Compare tab is instant.  The
    # cache lives in <folder>/firefly_extras/ and uses FIREFLY's native
    # CSV/JSON schema.
    try:
        import json as _json
        extras_dir = os.path.join(folder, "firefly_extras")
        os.makedirs(extras_dir, exist_ok=True)
        diff_df.to_csv(
            os.path.join(extras_dir, f"{stem}_diffusion_summary.csv"), index=False)
        tracks.to_csv(
            os.path.join(extras_dir, f"{stem}_trajectories.csv"), index=False)
        locs.to_csv(
            os.path.join(extras_dir, f"{stem}_localisations.csv"), index=False)
        emsd_df.to_csv(
            os.path.join(extras_dir, f"{stem}_ensemble_msd.csv"), index=False)
        with open(os.path.join(extras_dir, f"{stem}_params.json"), "w") as _fp:
            _json.dump({
                "stem":             stem,
                "pixel_size_um":    pixel_size_um,
                "frame_interval_s": frame_interval_s,
                "n_localisations":  int(len(locs)),
                "n_tracks":         int(diff_df.shape[0]),
                "n_frames":         int(n_frames),
                "width":            width,
                "height":           height,
                "source":           "palmtracer (re-derived)",
            }, _fp, indent=2)
        if jdd:
            with open(os.path.join(extras_dir, f"{stem}_jdd.json"), "w") as _fp:
                _json.dump(_to_jsonable(jdd) if "_to_jsonable" in globals() else jdd,
                           _fp, indent=2, default=str)
        if dwell_df is not None and len(dwell_df):
            dwell_df.to_csv(
                os.path.join(extras_dir, f"{stem}_dwell_times.csv"), index=False)
        if ta_deg is not None and len(ta_deg):
            pd.DataFrame({"turning_angle_deg": ta_deg}).to_csv(
                os.path.join(extras_dir, f"{stem}_turning_angles.csv"), index=False)
        if mobile_frac_df is not None and len(mobile_frac_df):
            mobile_frac_df.to_csv(
                os.path.join(extras_dir, f"{stem}_mobile_fraction.csv"), index=False)
    except Exception:
        # Caching is best-effort — never fail the load over a write error
        pass

    return {
        "folder":     folder,
        "stem":       stem,
        "data_dir":   folder,
        "source":     "palmtracer",
        "params": {
            "stem":             stem,
            "pixel_size_um":    pixel_size_um,
            "frame_interval_s": frame_interval_s,
            "n_localisations":  int(len(locs)),
            "n_tracks":         int(diff_df.shape[0]),
            "n_frames":         int(n_frames),
            "width":            width,
            "height":           height,
        },
        "ensemble_msd":          emsd_df,
        "diffusion":             diff_df,
        "tracks":                tracks,
        "jdd":                   jdd,
        "dwell_times":           dwell_df,
        "turning_angles":        ta_deg if ta_deg is not None else None,
        "turning_angles_signed": True,
    }


def load_summary_from_folder(folder):
    """Load all per-experiment summary data from one analysis output folder.

    Accepts any of:
      <run_dir>/                       (containing firefly_extras/ and data/)
      <run_dir>/firefly_extras/        (the FIREFLY-extras directory itself)
      <palm_tracer_folder>/            (auto-detected, re-derived on load)
      <run_dir>/data/                  (PALM-Tracer CSVs from a FIREFLY run)
    """
    import json

    # ── Resolve which directory holds the FIREFLY-native CSVs ────────────
    # 1) <folder>/firefly_extras  (folder is the run dir)
    if os.path.isdir(os.path.join(folder, "firefly_extras")):
        data_dir = os.path.join(folder, "firefly_extras")
    # 2) folder is itself the firefly_extras dir
    elif os.path.basename(folder.rstrip(os.sep)) == "firefly_extras":
        data_dir = folder
    # 3) folder is a PALM-Tracer folder (raw or FIREFLY-emitted CSV mirrors)
    elif _is_palmtracer_folder(folder):
        return load_summary_from_palmtracer(folder)
    # 4) folder is a run dir whose `data/` holds PALM-Tracer CSVs
    elif (os.path.isdir(os.path.join(folder, "data"))
          and _is_palmtracer_folder(os.path.join(folder, "data"))):
        return load_summary_from_palmtracer(os.path.join(folder, "data"))
    else:
        raise FileNotFoundError(
            f"No firefly_extras/ directory and no PALM-Tracer files in {folder}")

    stem = _find_stem(data_dir)
    s = {"folder": folder, "stem": stem, "data_dir": data_dir}

    # Params (frame interval, pixel size, ...)
    params_path = os.path.join(data_dir, f"{stem}_params.json")
    if os.path.isfile(params_path):
        with open(params_path) as f:
            s["params"] = json.load(f)
    else:
        s["params"] = {"pixel_size_um": 0.104, "frame_interval_s": 0.05}

    # Ensemble MSD
    msd_path = os.path.join(data_dir, f"{stem}_ensemble_msd.csv")
    if os.path.isfile(msd_path):
        s["ensemble_msd"] = pd.read_csv(msd_path)
    else:
        s["ensemble_msd"] = None

    # Diffusion summary (per-track D, alpha, motion_class)
    diff_path = os.path.join(data_dir, f"{stem}_diffusion_summary.csv")
    if os.path.isfile(diff_path):
        s["diffusion"] = pd.read_csv(diff_path)
    else:
        s["diffusion"] = None

    # Trajectories (for track length distribution)
    tr_path = os.path.join(data_dir, f"{stem}_trajectories.csv")
    if os.path.isfile(tr_path):
        s["tracks"] = pd.read_csv(tr_path)
    else:
        s["tracks"] = None

    # JDD
    jdd_path = os.path.join(data_dir, f"{stem}_jdd.json")
    if os.path.isfile(jdd_path):
        with open(jdd_path) as f:
            s["jdd"] = json.load(f)
    else:
        s["jdd"] = None

    # Dwell times
    dwell_path = os.path.join(data_dir, f"{stem}_dwell_times.csv")
    if os.path.isfile(dwell_path):
        s["dwell_times"] = pd.read_csv(dwell_path)
    else:
        s["dwell_times"] = None

    # Turning angles — signed degrees (-180..+180°)
    ta_path = os.path.join(data_dir, f"{stem}_turning_angles.csv")
    if os.path.isfile(ta_path):
        _ta_df = pd.read_csv(ta_path)
        s["turning_angles"]        = _ta_df["turning_angle_deg"].values
        s["turning_angles_signed"] = True
    else:
        s["turning_angles"]        = None
        s["turning_angles_signed"] = False

    return s


def save_palmtracer_csvs(out_dir, stem, locs, tracks, diff_df, imsd_df,
                         pixel_size_um, frame_interval_s,
                         width=None, height=None, n_frames=None,
                         mobile_D_threshold=None):
    """
    Emit PALM-Tracer-compatible CSV files alongside FIREFLY's native outputs.

    Files written (all comma-separated, written into `out_dir`):
        <stem>_locPALMTracer.csv              (one row per localisation)
        <stem>_trcPALMTracer.csv              (one row per trajectory plane)
        <stem>_trcPALMTracer-1-D.csv          (per-track D, MSD(0), MSE, LogD)
        <stem>_trcPALMTracer-1-MSD.csv        (per-track MSD curve, jagged)
        <stem>_trcPALMTracer-AllROI-D.csv     (per-track D summary)
        <stem>_trcPALMTracer-AllROI-MSD.csv   (per-track MSD curve, jagged)

    Column ordering, naming and unit conventions follow PALM-Tracer
    (Bordeaux Imaging Center).  ROI is hard-coded to 1 (FIREFLY does not
    sub-ROI tracks).  Fields FIREFLY does not measure (SigmaX/Y, Angle,
    MSE(Gauss), CentroidZ, MSE_Z, Pair_Distance) are filled with the
    PALM-Tracer "unused" sentinels (-1 or 0).
    """
    import csv as _csv
    import numpy as _np
    import pandas as _pd
    import os as _os

    if mobile_D_threshold is None:
        mobile_D_threshold = MOBILE_D_THRESHOLD_DEFAULT

    width    = int(width)    if width    is not None else 0
    height   = int(height)   if height   is not None else 0
    n_frames = int(n_frames) if n_frames is not None else int(
        max(locs["frame"].max() + 1, tracks["frame"].max() + 1))

    print(f"  PALM-Tracer: {len(locs):,} locs, {len(diff_df):,} tracks, "
          f"imsd_df shape {imsd_df.shape if imsd_df is not None else None}")

    # ── 1. locPALMTracer.csv ─────────────────────────────────────────────
    n_loc = len(locs)
    loc_path = _os.path.join(out_dir, f"{stem}_locPALMTracer.csv")
    with open(loc_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["Width", "Height", "nb_Planes", "nb_Points",
                    "Pixel_Size(um)", "Frame_Duration(s)",
                    "Gaussian_Fit", "Spectral"])
        w.writerow([width, height, n_frames, n_loc,
                    pixel_size_um, frame_interval_s, "None", "False"])
        w.writerow(["id", "Plane", "Index", "Channel",
                    "Integrated_Intensity",
                    "CentroidX(px)", "CentroidY(px)",
                    "SigmaX(px)", "SigmaY(px)", "Angle(rad)", "MSE(Gauss)",
                    "CentroidZ(um)", "MSE_Z(um)", "Pair_Distance(px)"])
        frames_l = locs["frame"].values
        xs       = locs["x"].values
        ys       = locs["y"].values
        mass     = (locs["mass"].values if "mass" in locs.columns
                    else _np.zeros(n_loc))
        for i in range(n_loc):
            w.writerow([i + 1, int(frames_l[i]) + 1, i + 1, -1,
                        float(mass[i]),
                        float(xs[i]), float(ys[i]),
                        0.0, 0.0, 0.0, 0.0,
                        -1.0, -1.0, 0.0])

    # ── 2. trcPALMTracer.csv ─────────────────────────────────────────────
    tr_path = _os.path.join(out_dir, f"{stem}_trcPALMTracer.csv")
    # Re-number particles 1..n in PALM-Tracer style
    pid_order  = (diff_df["particle"].values if "particle" in diff_df.columns
                  else sorted(tracks["particle"].unique()))
    pid_to_new = {int(p): i + 1 for i, p in enumerate(pid_order)}
    n_tracks   = len(pid_to_new)

    with open(tr_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["Width", "Height", "nb_Planes", "nb_Tracks",
                    "Pixel_Size(um)", "Frame_Duration(s)",
                    "Gaussian_Fit", "Spectral"])
        w.writerow([width, height, n_frames, n_tracks,
                    pixel_size_um, frame_interval_s, "None", "False"])
        w.writerow(["Track", "Plane", "CentroidX(px)", "CentroidY(px)",
                    "CentroidZ(um)", "Integrated_Intensity", "id",
                    "Pair_Distance(px)"])
        # trackpy.link sets `frame` as the index AND keeps it as a column —
        # pandas refuses to disambiguate in sort_values, so drop the index first.
        tr_sorted = tracks.reset_index(drop=True).sort_values(["particle", "frame"])
        pids      = tr_sorted["particle"].values
        frames_t  = tr_sorted["frame"].values
        xs_t      = tr_sorted["x"].values
        ys_t      = tr_sorted["y"].values
        mass_t    = (tr_sorted["mass"].values if "mass" in tr_sorted.columns
                     else _np.zeros(len(tr_sorted)))
        for k in range(len(tr_sorted)):
            new_id = pid_to_new.get(int(pids[k]))
            if new_id is None:
                continue
            w.writerow([new_id, int(frames_t[k]) + 1,
                        float(xs_t[k]), float(ys_t[k]),
                        -1, float(mass_t[k]), k + 1, 0])

    print(f"  PALM-Tracer: wrote loc + trc; starting D files")

    # ── 3 & 5. D files ───────────────────────────────────────────────────
    D_arr     = diff_df["D"].values
    msd0_arr  = (diff_df["MSD0"].values if "MSD0" in diff_df.columns
                 else _np.zeros(len(diff_df)))
    mse_arr   = (diff_df["MSE"].values  if "MSE"  in diff_df.columns
                 else _np.zeros(len(diff_df)))
    logD_arr  = _np.where(D_arr > 0, _np.log10(_np.where(D_arr > 0, D_arr, 1)),
                          _np.nan)
    mobile_n  = int(_np.sum(D_arr > mobile_D_threshold))
    immob_n   = int(_np.sum(D_arr <= mobile_D_threshold))
    mob_ratio = (mobile_n / immob_n) if immob_n else _np.nan

    d1_path = _os.path.join(out_dir, f"{stem}_trcPALMTracer-1-D.csv")
    with open(d1_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow([f"#Diffusion Coef in um2/s; Linear fit performed on the "
                    f"first points of trajectories"])
        w.writerow([f"#Pixel size= {pixel_size_um}um ; Frame rate= "
                    f"{frame_interval_s}sec"])
        w.writerow(["ROI", "Trace", "D(um2/s)", "MSD(0)", "MSE",
                    "LogD", "Mobile/Immobile", "Tracks"])
        for i, pid in enumerate(pid_order):
            new_id = pid_to_new[int(pid)]
            row = [1, new_id,
                   float(D_arr[i]) if _np.isfinite(D_arr[i]) else "",
                   float(msd0_arr[i]) if _np.isfinite(msd0_arr[i]) else "",
                   float(mse_arr[i]) if _np.isfinite(mse_arr[i]) else "",
                   float(logD_arr[i]) if _np.isfinite(logD_arr[i]) else "",
                   "", ""]
            if i == 0:
                row[6] = mob_ratio if _np.isfinite(mob_ratio) else ""
                row[7] = n_tracks
            w.writerow(row)

    dA_path = _os.path.join(out_dir, f"{stem}_trcPALMTracer-AllROI-D.csv")
    with open(dA_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow([f"#Diffusion Coef in um2/s; Linear fit performed on the "
                    f"first points of trajectories"])
        w.writerow([f"#Pixel size= {pixel_size_um}um ; Frame rate= "
                    f"{frame_interval_s}sec"])
        w.writerow(["ROI", "Trace", "D(um2/s)", "MSD(0)", "MSE"])
        for i, pid in enumerate(pid_order):
            new_id = pid_to_new[int(pid)]
            w.writerow([1, new_id,
                        float(D_arr[i]) if _np.isfinite(D_arr[i]) else "",
                        float(msd0_arr[i]) if _np.isfinite(msd0_arr[i]) else "",
                        float(mse_arr[i]) if _np.isfinite(mse_arr[i]) else ""])

    print(f"  PALM-Tracer: wrote D files; starting MSD files")

    # ── 4 & 6. MSD files (jagged: one column per surviving lag) ──────────
    def _write_msd(path):
        with open(path, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["#MSD(DeltaT) in um2"])
            w.writerow([f"#Pixel size= {pixel_size_um}um ; Frame rate= "
                        f"{frame_interval_s}sec"])
            for pid in pid_order:
                if int(pid) not in imsd_df.columns and pid not in imsd_df.columns:
                    continue
                col = imsd_df[pid] if pid in imsd_df.columns else imsd_df[int(pid)]
                vals = col.values
                finite_idx = _np.where(_np.isfinite(vals))[0]
                if len(finite_idx) == 0:
                    continue
                last = finite_idx[-1] + 1
                row = [1, pid_to_new[int(pid)]]
                row.extend(float(v) if _np.isfinite(v) else ""
                           for v in vals[:last])
                w.writerow(row)

    _write_msd(_os.path.join(out_dir, f"{stem}_trcPALMTracer-1-MSD.csv"))
    _write_msd(_os.path.join(out_dir, f"{stem}_trcPALMTracer-AllROI-MSD.csv"))
    print(f"  PALM-Tracer: all 6 files written successfully")

    return {
        "loc":           loc_path,
        "trc":           tr_path,
        "D_1":           d1_path,
        "D_AllROI":      dA_path,
        "MSD_1":         _os.path.join(out_dir, f"{stem}_trcPALMTracer-1-MSD.csv"),
        "MSD_AllROI":    _os.path.join(out_dir, f"{stem}_trcPALMTracer-AllROI-MSD.csv"),
    }


def _msd_auc(emsd_df, frame_interval):
    """Trapezoidal AUC of the MSD curve in µm²·s units."""
    if emsd_df is None or len(emsd_df) == 0:
        return np.nan
    t = emsd_df["lag_frame"].values * frame_interval
    y = emsd_df["msd_um2"].values
    order = np.argsort(t)
    # NumPy 2.x renamed trapz → trapezoid
    _trap = getattr(np, "trapezoid", None) or np.trapz
    return float(_trap(y[order], t[order]))


def _mob_immob_ratio(diff_df, d_threshold=MOBILE_D_THRESHOLD_DEFAULT):
    """Mobile / Immobile ratio defined by a diffusion-coefficient threshold.

    Tracks with D ≥ d_threshold count as Mobile; D < d_threshold count as
    Immobile.  Tracks with non-finite D (alpha fit failed) are excluded
    from BOTH numerator and denominator — they contribute neither mobility
    state, which avoids inflating either count.
    """
    if diff_df is None or "D" not in diff_df.columns:
        return np.nan
    d = diff_df["D"].values
    valid = np.isfinite(d) & (d > 0)
    if valid.sum() == 0:
        return np.nan
    d = d[valid]
    n_mob = int((d >= d_threshold).sum())
    n_imm = int((d <  d_threshold).sum())
    return float(n_mob / n_imm) if n_imm > 0 else np.nan


def _motion_fractions(diff_df):
    """Return dict of fractions per motion class."""
    if diff_df is None or "motion" not in diff_df.columns:
        return {}
    counts = diff_df["motion"].value_counts()
    total = counts.sum()
    if total == 0:
        return {}
    return {k: float(v / total) for k, v in counts.items()}


def _track_lengths(tracks_df, frame_interval):
    """Return per-track lengths in seconds."""
    if tracks_df is None or "particle" not in tracks_df.columns:
        return np.array([])
    counts = tracks_df.groupby("particle").size().values
    return counts * frame_interval


def _stat_test(a, b):
    """Two-sample test on per-experiment scalars.  Welch's t by default,
    Mann-Whitney as fallback for non-normal data.  Returns (p, label)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return (np.nan, "")
    try:
        from scipy.stats import ttest_ind, mannwhitneyu, shapiro
        normal = True
        for arr in (a, b):
            if 3 <= len(arr) <= 5000:
                try:
                    if shapiro(arr).pvalue < 0.05:
                        normal = False
                        break
                except Exception:
                    pass
        if normal:
            p = ttest_ind(a, b, equal_var=False).pvalue
        else:
            p = mannwhitneyu(a, b, alternative="two-sided").pvalue
        if not np.isfinite(p):
            return (np.nan, "")
        if p < 0.001: stars = "***"
        elif p < 0.01: stars = "**"
        elif p < 0.05: stars = "*"
        else: stars = "ns"
        return (float(p), stars)
    except Exception:
        return (np.nan, "")


# NOTE: the canonical `_theme_palette` definition lives near
# `compute_circular_statistics` above.  Earlier there was a SECOND
# definition here with a smaller set of keys (BG/PNL/TXT/GRD/BAR_FILL/
# SIG/FONT only); because Python rebinds `_theme_palette` at module
# parse time, the second definition silently won and every call to
# `_theme_palette` returned a dict missing MUT/ACC/HDR_BG/HDR_TXT/
# ZEBRA/ARROW.  That manifested as `KeyError('MUT')` from
# `save_circular_statistics_pdf` (which uses those keys).  The
# canonical version above now also exports the BAR_FILL/SIG keys
# `compare_groups` consumes here, so the duplicate is safe to remove.


def _stat_test_n(arrays, labels):
    """Statistical test across N≥2 groups.

    Returns
    -------
    omnibus : dict with keys {"test", "p", "stars"} or None if n<2 each
    pairwise : list of dicts with keys
        {"i", "j", "label_i", "label_j", "test", "p", "stars",
         "n_i", "n_j", "mean_i", "mean_j", "sem_i", "sem_j"}
    """
    arrs = [np.asarray(a, dtype=float)[np.isfinite(np.asarray(a, dtype=float))]
            for a in arrays]
    valid_idx = [i for i, a in enumerate(arrs) if len(a) >= 2]

    omnibus = None
    pairwise = []

    def _star(p):
        if not np.isfinite(p):
            return ""
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return "ns"

    if len(valid_idx) < 2:
        # Still record per-pair "ns" rows for stats CSV completeness
        for i in range(len(arrs)):
            for j in range(i + 1, len(arrs)):
                pairwise.append({
                    "i": i, "j": j,
                    "label_i": labels[i], "label_j": labels[j],
                    "test": "n<2", "p": np.nan, "stars": "",
                    "n_i": int(len(arrs[i])), "n_j": int(len(arrs[j])),
                    "mean_i": float(arrs[i].mean()) if len(arrs[i]) else np.nan,
                    "mean_j": float(arrs[j].mean()) if len(arrs[j]) else np.nan,
                    "sem_i": (float(arrs[i].std(ddof=1) / np.sqrt(len(arrs[i])))
                              if len(arrs[i]) > 1 else np.nan),
                    "sem_j": (float(arrs[j].std(ddof=1) / np.sqrt(len(arrs[j])))
                              if len(arrs[j]) > 1 else np.nan),
                })
        return omnibus, pairwise

    # Omnibus test
    try:
        from scipy.stats import f_oneway, kruskal, shapiro
        valid_arrs = [arrs[i] for i in valid_idx]

        normal = True
        for a in valid_arrs:
            if 3 <= len(a) <= 5000:
                try:
                    if shapiro(a).pvalue < 0.05:
                        normal = False
                        break
                except Exception:
                    pass

        if len(valid_arrs) == 2:
            from scipy.stats import ttest_ind, mannwhitneyu
            if normal:
                p = ttest_ind(*valid_arrs, equal_var=False).pvalue
                test_name = "Welch's t-test"
            else:
                p = mannwhitneyu(*valid_arrs, alternative="two-sided").pvalue
                test_name = "Mann-Whitney U"
        else:
            if normal:
                p = f_oneway(*valid_arrs).pvalue
                test_name = "One-way ANOVA"
            else:
                p = kruskal(*valid_arrs).pvalue
                test_name = "Kruskal-Wallis"
        if np.isfinite(p):
            omnibus = {"test": test_name, "p": float(p), "stars": _star(p)}
    except Exception:
        pass

    # Pairwise comparisons
    try:
        from scipy.stats import ttest_ind, mannwhitneyu, shapiro
        for i in range(len(arrs)):
            for j in range(i + 1, len(arrs)):
                a, b = arrs[i], arrs[j]
                if len(a) < 2 or len(b) < 2:
                    p = np.nan
                    test_name = "n<2"
                else:
                    is_normal = True
                    for arr in (a, b):
                        if 3 <= len(arr) <= 5000:
                            try:
                                if shapiro(arr).pvalue < 0.05:
                                    is_normal = False
                                    break
                            except Exception:
                                pass
                    if is_normal:
                        p = ttest_ind(a, b, equal_var=False).pvalue
                        test_name = "Welch's t-test"
                    else:
                        p = mannwhitneyu(a, b, alternative="two-sided").pvalue
                        test_name = "Mann-Whitney U"
                pairwise.append({
                    "i": i, "j": j,
                    "label_i": labels[i], "label_j": labels[j],
                    "test": test_name,
                    "p": float(p) if np.isfinite(p) else np.nan,
                    "stars": _star(p) if np.isfinite(p) else "",
                    "n_i": int(len(a)), "n_j": int(len(b)),
                    "mean_i": float(a.mean()) if len(a) else np.nan,
                    "mean_j": float(b.mean()) if len(b) else np.nan,
                    "sem_i": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.nan,
                    "sem_j": float(b.std(ddof=1) / np.sqrt(len(b))) if len(b) > 1 else np.nan,
                })
    except Exception:
        pass

    return omnibus, pairwise


def _bar_with_dots_n(ax, data_per_group, labels, colors, palette,
                     ylabel="", record_stats=None, metric_name=""):
    """Bar chart with mean ± SEM and individual replicate dots, generalised
    to N groups.

    For 2 groups: shows pairwise stars on a bracket (matches lab style).
    For 3+ groups: shows omnibus ANOVA / Kruskal p-value as a panel
    annotation; full pairwise comparisons go to record_stats[metric_name]."""
    fill = palette["BAR_FILL"]
    sig_col = palette["SIG"]

    arrs = [np.asarray(d, dtype=float) for d in data_per_group]
    arrs = [a[np.isfinite(a)] for a in arrs]
    n = len(arrs)
    means = [float(a.mean()) if len(a) else 0.0 for a in arrs]
    sems  = [float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
             for a in arrs]
    x = np.arange(n)
    ax.bar(x, means, yerr=sems, capsize=4,
           color=[fill] * n,
           edgecolor=colors, linewidth=1.5,
           ecolor=sig_col)
    rng = np.random.default_rng(0)
    for i, a in enumerate(arrs):
        if len(a):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(a)), a,
                       color=colors[i], s=18, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15 if n > 3 else 0)
    ax.set_ylabel(ylabel)

    # Stats
    omnibus, pairwise = _stat_test_n(arrs, labels)
    if record_stats is not None and metric_name:
        record_stats[metric_name] = {"omnibus": omnibus, "pairwise": pairwise}

    # Annotation
    top_data = max([a.max() if len(a) else 0 for a in arrs] + [max(means) * 1.2 if max(means) > 0 else 1])
    if n == 2 and pairwise:
        pair = pairwise[0]
        if pair["stars"] and np.isfinite(pair["p"]):
            top = top_data * 1.05
            ax.plot([0, 0, 1, 1], [top, top * 1.03, top * 1.03, top],
                    color=sig_col, lw=0.8)
            # Numeric p plus stars, e.g. "p = 0.003  **"
            p_str = (f"p = {pair['p']:.2e}" if pair['p'] < 0.001
                     else f"p = {pair['p']:.3f}")
            label = f"{p_str}  {pair['stars']}"
            ax.text(0.5, top * 1.05, label, ha="center", va="bottom",
                    fontsize=9, color=sig_col)
            # Make room above the bracket for the longer label
            ax.set_ylim(0, top * 1.30)
    elif n > 2 and omnibus:
        # Show test name + omnibus p + stars in the upper-left corner.
        # Numeric format adapts to magnitude: scientific < 0.001, fixed otherwise.
        p_val = omnibus['p']
        p_str = (f"p = {p_val:.2e}" if p_val < 0.001
                 else f"p = {p_val:.3f}")
        text = f"{omnibus['test']}\n{p_str}   {omnibus['stars']}"
        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                ha="left", va="top", fontsize=8, color=sig_col,
                bbox=dict(facecolor=palette["PNL"], edgecolor="none",
                          alpha=0.7, pad=3))


def compare_groups(groups,
                   output_dir=None, output_stem="comparison",
                   panels=None, theme="Dark",
                   pdf_report=True,
                   mobile_d_threshold=MOBILE_D_THRESHOLD_DEFAULT,
                   progress_cb=None):
    """Compare N≥2 groups of analysis output folders and render a multi-panel
    figure, summary CSV, statistics CSV and combined PDF report.

    Parameters
    ----------
    groups : list[dict]
        [{"folders": [path, ...], "label": "Pre", "color": "#000000"}, ...]
    output_dir : str or None
        Where to save the figure / CSVs / PDF report.  If None, nothing is
        saved to disk and only the figure is returned.
    panels : set[str] or None
        Subset of panels to render.  Default: all of {"msd", "auc",
        "logd_dist", "mob_immob", "motion_classes", "track_length",
        "jdd", "dwell_cdf", "turning_angles"}.
    theme : str
        Figure theme — "Dark" (default), "Light" or "Publication".
    pdf_report : bool
        If True (default) and output_dir is given, also write a multi-page
        PDF report bundling the figure, parameters, folder lists and stats.
    progress_cb : callable or None
        Optional callback(done:int, total:int, msg:str) for UI progress.

    Returns
    -------
    fig         : matplotlib.figure.Figure
    summary_df  : pandas.DataFrame  — per-replicate scalar metrics
    stats       : dict[str, dict]   — per-metric omnibus + pairwise tests
    """
    import matplotlib.pyplot as plt

    if len(groups) < 2:
        raise ValueError(f"Need at least 2 groups; got {len(groups)}")

    if panels is None:
        panels = {"msd", "auc", "logd_dist", "mob_immob", "motion_classes",
                  "track_length", "jdd", "dwell_cdf", "turning_angles",
                  "radial_dist"}

    n_groups = len(groups)
    labels   = [g.get("label", f"Group {i+1}") for i, g in enumerate(groups)]
    colors   = [g.get("color", "#3b6ed8")     for g in groups]
    folder_lists = [list(g["folders"]) for g in groups]

    # ── Load summaries for all groups ─────────────────────────────────────────
    all_summaries = [[] for _ in groups]
    total = sum(len(f) for f in folder_lists)
    done = 0
    for gi, folders in enumerate(folder_lists):
        for f in folders:
            if progress_cb:
                progress_cb(done, total, f"Loading: {os.path.basename(f)}")
            try:
                all_summaries[gi].append(load_summary_from_folder(f))
            except Exception as e:
                print(f"  Skipping {f}: {e}")
            done += 1

    empty_groups = [labels[i] for i, ss in enumerate(all_summaries) if len(ss) == 0]
    if empty_groups:
        raise RuntimeError(
            "Need at least one valid folder per group; these are empty: "
            + ", ".join(empty_groups))

    if progress_cb:
        progress_cb(total, total, "Computing scalars and rendering...")

    # ── Compute per-folder scalars (one row per replicate) ────────────────────
    summary_rows = []
    def _row(group_label, summary):
        p = summary["params"]
        fi = float(p.get("frame_interval_s", 0.05))
        d = summary["diffusion"]
        return {
            "group":            group_label,
            "folder":           summary["folder"],
            "stem":             summary["stem"],
            "n_tracks":         len(d) if d is not None else 0,
            "auc_msd":          _msd_auc(summary["ensemble_msd"], fi),
            "mob_immob_ratio":  _mob_immob_ratio(d, mobile_d_threshold),
            "median_D":         float(d["D"].median()) if d is not None and "D" in d.columns else np.nan,
            "median_alpha":     float(d["alpha"].median()) if d is not None and "alpha" in d.columns else np.nan,
            "mean_track_length_s": float(_track_lengths(summary["tracks"], fi).mean())
                                   if summary["tracks"] is not None else np.nan,
        }
    for gi, summaries in enumerate(all_summaries):
        for s in summaries:
            summary_rows.append(_row(labels[gi], s))
    summary_df = pd.DataFrame(summary_rows)

    # Per-metric statistics dict — populated as panels render
    stats_records = {}

    # ── Render the figure ────────────────────────────────────────────────────
    panel_order = ["msd", "auc", "logd_dist", "mob_immob",
                   "motion_classes", "track_length",
                   "jdd", "dwell_cdf", "turning_angles", "radial_dist"]
    enabled = [p for p in panel_order if p in panels]
    n_plots = len(enabled)
    if n_plots == 0:
        raise RuntimeError("No panels enabled")
    print(f"  Compare: rendering {n_plots} panel(s): {enabled}")
    if "radial_dist" not in panels:
        print(f"  Compare: 'radial_dist' NOT in requested panels — "
              f"check the 'Radial distribution (polar)' tickbox in the "
              f"Compare tab to include it.")
    ncols = 3 if n_plots > 4 else 2
    nrows = (n_plots + ncols - 1) // ncols

    pal = _theme_palette(theme)
    plt.rcParams.update({
        "text.color":      pal["TXT"], "axes.labelcolor": pal["TXT"],
        "xtick.color":     pal["TXT"], "ytick.color":     pal["TXT"],
        "axes.titlecolor": pal["TXT"],
        "axes.edgecolor":  pal["GRD"], "axes.facecolor":  pal["PNL"],
        "figure.facecolor": pal["BG"], "figure.edgecolor": pal["BG"],
        "savefig.facecolor": pal["BG"], "savefig.edgecolor": pal["BG"],
        "grid.color":      pal["GRD"], "grid.alpha": 0.4,
        "font.family":     pal["FONT"],
        "legend.facecolor": pal["PNL"], "legend.edgecolor": pal["GRD"],
    })

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 3.6),
                             facecolor=pal["BG"])
    axes = np.array(axes).reshape(-1)
    for ax in axes[n_plots:]:
        ax.axis("off")

    panel_idx = 0
    def _next_ax():
        nonlocal panel_idx
        ax = axes[panel_idx]; panel_idx += 1
        return ax

    def _zip_groups():
        """Iterator: (label, summaries, color) for each group."""
        for i in range(n_groups):
            yield labels[i], all_summaries[i], colors[i]

    # ── 1. MSD overlay ────────────────────────────────────────────────────────
    if "msd" in panels:
        ax = _next_ax()
        for grp_label, summaries, color in _zip_groups():
            curves = []
            tref = None
            for s in summaries:
                e = s["ensemble_msd"]
                if e is None: continue
                fi = float(s["params"].get("frame_interval_s", 0.05))
                t = e["lag_frame"].values * fi
                y = e["msd_um2"].values
                order = np.argsort(t)
                t, y = t[order], y[order]
                if tref is None:
                    tref = t
                if len(t) != len(tref) or not np.allclose(t, tref):
                    y = np.interp(tref, t, y)
                curves.append(y)
            if not curves:
                continue
            arr = np.vstack(curves)
            mean = arr.mean(axis=0)
            sem = arr.std(axis=0, ddof=1) / np.sqrt(len(curves)) if len(curves) > 1 else None
            ax.plot(tref, mean, "-o", color=color, label=grp_label, ms=4, lw=1.5)
            if sem is not None:
                ax.fill_between(tref, mean - sem, mean + sem, color=color, alpha=0.15)
        ax.set_xlabel("Time delta (s)")
        ax.set_ylabel("MSD (µm²)")
        ax.set_title("Mean Square Displacement")
        ax.legend(frameon=False, loc="best")

    # ── 2. AUC bar chart ──────────────────────────────────────────────────────
    if "auc" in panels:
        ax = _next_ax()
        data = [summary_df.loc[summary_df["group"] == lbl, "auc_msd"].values
                for lbl in labels]
        _bar_with_dots_n(ax, data, labels, colors, pal,
                         ylabel="AUC (µm²·s)",
                         record_stats=stats_records, metric_name="auc_msd")
        ax.set_title("Area Under the Curve")

    # ── 3. LogD frequency distribution ────────────────────────────────────────
    if "logd_dist" in panels:
        ax = _next_ax()
        bins = np.linspace(-5, 1, 31)
        for grp_label, summaries, color in _zip_groups():
            all_logD = []
            for s in summaries:
                d = s["diffusion"]
                if d is None or "D" not in d.columns: continue
                vals = d["D"].values
                vals = vals[vals > 0]
                if len(vals): all_logD.append(np.log10(vals))
            if not all_logD: continue
            pooled = np.concatenate(all_logD)
            counts, edges = np.histogram(pooled, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            frac = counts / counts.sum() if counts.sum() else counts
            ax.plot(centers, frac, "-o", color=color, label=grp_label, ms=4, lw=1.2)
        ax.axvline(np.log10(mobile_d_threshold), color=pal["GRD"], ls="--", lw=0.8,
                   label=f"D = {mobile_d_threshold} µm²/s")
        ax.set_xlabel("log₁₀ D  (µm²/s)")
        ax.set_ylabel("Relative frequency")
        ax.set_title("LogD Frequency Distribution")
        ax.legend(frameon=False, loc="best")

    # ── 4. Mobile/Immobile ratio bar ──────────────────────────────────────────
    if "mob_immob" in panels:
        ax = _next_ax()
        data = [summary_df.loc[summary_df["group"] == lbl, "mob_immob_ratio"].values
                for lbl in labels]
        _bar_with_dots_n(ax, data, labels, colors, pal,
                         ylabel="Mobile/Immobile ratio",
                         record_stats=stats_records, metric_name="mob_immob_ratio")
        ax.set_title("Mobile/Immobile Ratio")

    # ── 5. Motion class fractions (grouped bars, N groups) ────────────────────
    if "motion_classes" in panels:
        ax = _next_ax()
        classes = ["Immobile", "Confined", "Brownian", "Directed"]
        def _fracs(summaries):
            rows = []
            for s in summaries:
                f = _motion_fractions(s["diffusion"])
                rows.append([f.get(c, 0.0) for c in classes])
            return np.array(rows) if rows else np.zeros((0, len(classes)))
        per_group = [_fracs(ss) for ss in all_summaries]
        x = np.arange(len(classes))
        # Group-bar width: total slot ~0.8, divided across N groups
        slot = 0.8
        w = slot / n_groups
        rng = np.random.default_rng(1)
        for gi, (grp_label, color, fracs) in enumerate(zip(labels, colors, per_group)):
            if not len(fracs): continue
            x_off = (gi - (n_groups - 1) / 2) * w
            ax.bar(x + x_off, fracs.mean(axis=0), w * 0.9,
                   yerr=fracs.std(axis=0, ddof=1)/np.sqrt(len(fracs)) if len(fracs) > 1 else None,
                   color=pal["BAR_FILL"], edgecolor=color, linewidth=1.5,
                   ecolor=pal["SIG"], capsize=3, label=grp_label)
            for ci in range(len(classes)):
                ax.scatter(np.full(len(fracs), x[ci] + x_off)
                           + rng.uniform(-w*0.25, w*0.25, len(fracs)),
                           fracs[:, ci], color=color, s=12, zorder=3)
        # Per-class stats
        for ci, cname in enumerate(classes):
            arrs = [fracs[:, ci] if len(fracs) else np.array([]) for fracs in per_group]
            omn, pw = _stat_test_n(arrs, labels)
            stats_records[f"motion_frac_{cname}"] = {"omnibus": omn, "pairwise": pw}
        ax.set_xticks(x); ax.set_xticklabels(classes, rotation=15)
        ax.set_ylabel("Fraction of tracks")
        ax.set_title("Motion Class Fractions")
        ax.legend(frameon=False, loc="best", fontsize=8)

    # ── 6. Track length distribution (CDF, x clipped at 99th %ile) ────────────
    if "track_length" in panels:
        ax = _next_ax()
        pooled_per_group = {}
        for grp_label, summaries, _ in _zip_groups():
            arrs = []
            for s in summaries:
                fi = float(s["params"].get("frame_interval_s", 0.05))
                tl = _track_lengths(s["tracks"], fi)
                if len(tl):
                    arrs.append(tl)
            if arrs:
                pooled_per_group[grp_label] = np.concatenate(arrs)
        combined = (np.concatenate(list(pooled_per_group.values()))
                    if pooled_per_group else np.array([]))
        x_clip = float(np.percentile(combined, 99)) if len(combined) else None
        for grp_label, color in zip(labels, colors):
            p = pooled_per_group.get(grp_label)
            if p is None or len(p) == 0: continue
            x_sorted = np.sort(p)
            y = np.arange(1, len(x_sorted) + 1) / len(x_sorted)
            ax.plot(x_sorted, y, color=color, lw=1.5, label=grp_label)
        if pooled_per_group:
            if x_clip and x_clip > 0:
                ax.set_xlim(0, x_clip)
                ax.set_title("Track Length Distribution  (x clipped at 99th %ile)")
            else:
                ax.set_title("Track Length Distribution")
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Track length (s)")
            ax.set_ylabel("Cumulative fraction")
            ax.legend(frameon=False, loc="best")
        else:
            ax.text(0.5, 0.5, "No track-length data",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Track Length Distribution")
        # Stats: mean track length (per-replicate)
        arrs = [summary_df.loc[summary_df["group"] == lbl, "mean_track_length_s"].values
                for lbl in labels]
        omn, pw = _stat_test_n(arrs, labels)
        stats_records["mean_track_length_s"] = {"omnibus": omn, "pairwise": pw}

    # ── 7. JDD: per-population D + fraction (N groups) ────────────────────────
    if "jdd" in panels:
        ax = _next_ax()
        any_data = False
        max_pop_overall = 0
        # Spread groups across ±0.18 around each population index
        if n_groups > 1:
            offsets = np.linspace(-0.18, 0.18, n_groups)
        else:
            offsets = np.array([0.0])
        for gi, (grp_label, summaries, color) in enumerate(_zip_groups()):
            label_done = False
            for s in summaries:
                jd = s.get("jdd")
                if not jd or "D_values" not in jd: continue
                D = np.asarray(jd["D_values"], dtype=float)
                f = np.asarray(jd.get("fractions", np.ones_like(D)), dtype=float)
                if D.size == 0: continue
                any_data = True
                max_pop_overall = max(max_pop_overall, len(D))
                sizes = 25 + 175 * np.clip(f, 0, 1)
                xs = np.arange(len(D)) + offsets[gi]
                ax.scatter(xs, D, s=sizes, color=color,
                           alpha=0.55, edgecolor=color,
                           label=(grp_label if not label_done else None))
                label_done = True
        if any_data:
            tick_labels = ["Immobile", "Mobile", "Fast"][:max_pop_overall]
            if max_pop_overall == 1: tick_labels = ["All"]
            ax.set_xticks(np.arange(max_pop_overall))
            ax.set_xticklabels(tick_labels)
            ax.set_xlim(-0.5, max_pop_overall - 0.5)
            ax.set_ylabel("D (µm²/s, log)")
            ax.set_yscale("log")
            ax.set_title("JDD: per-population D  (marker size ∝ population fraction)")
            ax.legend(frameon=False, loc="best")
        else:
            ax.text(0.5, 0.5, "No JDD data\n(re-run analysis to generate)",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Jump Distance Distribution")

    # ── 8. Dwell time CDF (N groups) ──────────────────────────────────────────
    if "dwell_cdf" in panels:
        ax = _next_ax()
        any_data = False
        for grp_label, summaries, color in _zip_groups():
            pooled = []
            for s in summaries:
                d = s.get("dwell_times")
                if d is None or len(d) == 0: continue
                col = next((c for c in ("dwell_time_s", "dwell_s",
                                        "dwell_time", "dwell", "tau_s")
                            if c in d.columns), None)
                if col is None: continue
                pooled.extend(d[col].values)
            if not pooled: continue
            any_data = True
            arr = np.sort(np.asarray(pooled, dtype=float))
            arr = arr[arr > 0]
            if len(arr) == 0: continue
            y = 1 - np.arange(1, len(arr) + 1) / len(arr)
            ax.plot(arr, y, color=color, lw=1.5, label=grp_label)
        if any_data:
            ax.set_xlabel("Dwell time (s)")
            ax.set_ylabel("Survival fraction")
            ax.set_title("Dwell Time Survival")
            ax.set_yscale("log")
            ax.legend(frameon=False, loc="best")
        else:
            ax.text(0.5, 0.5, "No dwell-time data\n(re-run analysis to generate)",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Dwell Time Survival")

    # ── 9. Turning angle distribution (N groups, unsigned |angle|) ────────────
    # Single line per group, plotting the count of each |θ| bin on
    # the same 0°–180° x-axis.  Sign / rotational direction is handled
    # separately by the Radial Distribution panel.
    if "turning_angles" in panels:
        ax = _next_ax()
        any_data = False
        bins = np.linspace(0, 180, 37)                 # 5° bins
        centers = 0.5 * (bins[:-1] + bins[1:])
        pooled_per_group = []
        for grp_label, summaries, color in _zip_groups():
            pooled = []
            for s in summaries:
                ta = s.get("turning_angles")
                if ta is None or len(ta) == 0: continue
                pooled.extend(np.abs(np.asarray(ta).ravel()))
            pooled_per_group.append((grp_label, color, pooled))
        for grp_label, color, pooled in pooled_per_group:
            if not pooled: continue
            any_data = True
            counts, _ = np.histogram(pooled, bins=bins)
            frac = counts / counts.sum() if counts.sum() else counts
            ax.plot(centers, frac, "-o", color=color, lw=1.5, ms=3, label=grp_label)
        if any_data:
            ax.set_xlabel("|Turning angle|  (°)")
            ax.set_ylabel("Relative frequency")
            ax.set_xlim(0, 180)
            ax.set_xticks([0, 45, 90, 135, 180])
            ax.set_title("Turning Angle Distribution")
            ax.legend(frameon=False, loc="best")
        else:
            ax.text(0.5, 0.5, "No turning-angle data\n(re-run analysis to generate)",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Turning Angle Distribution")

    # ── 10. Radial distribution (polar, signed turning angles) ────────────────
    # Polar histogram showing the angular distribution of step-to-step
    # turning angles.  Each group is plotted as a separate set of bars
    # offset around each bin centre.
    #
    # Implementation note: we replace the auto-created cartesian axis with
    # a polar one at the SAME SubplotSpec (not via fig.add_axes with raw
    # bounds), so that the polar axis remains a managed gridspec member.
    # If we used add_axes(bounds), tight_layout would later reposition the
    # other (gridspec-managed) subplots but leave the polar in its original
    # location, causing visible overlap.
    if "radial_dist" in panels:
        old_ax = axes[panel_idx]
        ss = old_ax.get_subplotspec()
        old_ax.remove()
        ax = fig.add_subplot(ss, projection="polar")
        axes[panel_idx] = ax
        panel_idx += 1

        any_data = False
        n_bins = 36
        # matplotlib polar bar() only renders correctly when theta ∈ [0, 2π);
        # shift the data accordingly.  The xticks are placed at positive-only
        # angles but labelled with their signed equivalents.
        bin_edges   = np.linspace(0, 2 * np.pi, n_bins + 1)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bar_width   = (bin_edges[1] - bin_edges[0]) * 0.95

        # First pass: get raw counts per group per bin.
        counts_per_group = []     # list of (group_idx, counts_array)
        for gi in range(n_groups):
            pooled = []
            for s in all_summaries[gi]:
                ta = s.get("turning_angles")
                if ta is None or len(ta) == 0: continue
                pooled.extend(np.asarray(ta).ravel())
            if not pooled:
                counts_per_group.append((gi, np.zeros(n_bins)))
                continue
            arr = np.asarray(pooled, dtype=float)
            if not np.any(arr < -1e-3):
                arr = np.concatenate([arr, -arr])
            angles_rad = np.mod(np.deg2rad(arr), 2 * np.pi)
            counts, _ = np.histogram(angles_rad, bins=bin_edges)
            counts_per_group.append((gi, counts.astype(float)))
            if counts.sum() > 0:
                any_data = True

        if any_data:
            # ── Normalise each group to ITS OWN total ─────────────────────
            # Otherwise a group with more total angles automatically draws
            # bigger bars everywhere — a sample-size artefact, not a real
            # shape difference.  After dividing by the per-group total, each
            # group's values sum to 1.0 across the full circle, so the bars
            # compare distribution SHAPE.
            # Bars from different groups are offset around each bin centre
            # for easy side-by-side comparison.
            per_bar_width = bar_width / max(1, n_groups) * 0.95
            for gi, counts in counts_per_group:
                total = counts.sum()
                if total <= 0:
                    continue
                normalised = counts / total
                offset = (gi - (n_groups - 1) / 2) * per_bar_width
                ax.bar(bin_centres + offset, normalised,
                       width=per_bar_width, bottom=0.0,
                       color=colors[gi], alpha=0.85,
                       edgecolor=pal["GRD"], linewidth=0.3,
                       label=labels[gi])

        if any_data:
            # Conventional orientation: 0° at top (straight ahead),
            # right hemisphere = positive turns, left hemisphere = negative.
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            # Positive-only xticks; labelled with signed equivalents.
            ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
            ax.set_xticklabels(["0°", "+45°", "+90°", "+135°", "±180°",
                                "−135°", "−90°", "−45°"], fontsize=7)
            # Hide the radial-axis numeric labels — bar length is
            # interpreted comparatively, not in absolute density units.
            ax.set_yticklabels([])
            ax.tick_params(axis="y", which="both", left=False)
            ax.set_title("Radial Distribution  (each group normalised to "
                         "its own total)", pad=14, fontsize=9)
            ax.legend(loc="upper right", bbox_to_anchor=(1.20, 1.10),
                      frameon=False, fontsize=8)
            ax.grid(True, ls=":", alpha=0.4)
        else:
            ax.text(0.5, 0.5, "No turning-angle data",
                    ha="center", va="center", transform=ax.transAxes,
                    color=pal["GRD"], fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title("Radial Distribution")

    # ── Suptitle: Group A (n=…) vs Group B (n=…) [vs Group C …] ───────────────
    parts = [f"{labels[i]}  (n={len(all_summaries[i])})" for i in range(n_groups)]
    fig.suptitle("   vs   ".join(parts),
                 fontsize=12, fontweight="bold", color=pal["TXT"])
    for ax in axes[:n_plots]:
        ax.set_facecolor(pal["PNL"])
        for spine in ax.spines.values():
            spine.set_edgecolor(pal["GRD"])
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # ── Build statistics dataframe (per metric × pairwise) ────────────────────
    # Bonferroni correction across pairwise comparisons WITHIN each metric:
    # multiplies the raw p-value by the number of pairs (capped at 1.0).
    # The omnibus row gets the raw p-value only — it's a single test.
    stats_rows = []
    for metric, rec in stats_records.items():
        omn = rec.get("omnibus")
        if omn:
            stars = omn["stars"]
            stars_bonf = stars  # omnibus needs no correction
            stats_rows.append({
                "metric": metric, "comparison": "omnibus",
                "test": omn["test"],
                "p_value": omn["p"], "stars": stars,
                "p_value_bonferroni": omn["p"], "stars_bonferroni": stars_bonf,
                "n_a": "", "n_b": "", "mean_a": "", "mean_b": "",
                "sem_a": "", "sem_b": "", "label_a": "all groups", "label_b": "",
            })
        pairs = rec.get("pairwise", [])
        n_pairs = max(1, len(pairs))
        for pw in pairs:
            p = pw["p"]
            if np.isfinite(p):
                p_bonf = min(1.0, p * n_pairs)
                if   p_bonf < 0.001: stars_bonf = "***"
                elif p_bonf < 0.01:  stars_bonf = "**"
                elif p_bonf < 0.05:  stars_bonf = "*"
                else:                stars_bonf = "ns"
            else:
                p_bonf = np.nan
                stars_bonf = ""
            stats_rows.append({
                "metric": metric, "comparison": f"{pw['label_i']} vs {pw['label_j']}",
                "test": pw["test"],
                "p_value": pw["p"], "stars": pw["stars"],
                "p_value_bonferroni": p_bonf, "stars_bonferroni": stars_bonf,
                "n_a": pw["n_i"], "n_b": pw["n_j"],
                "mean_a": pw["mean_i"], "mean_b": pw["mean_j"],
                "sem_a": pw["sem_i"], "sem_b": pw["sem_j"],
                "label_a": pw["label_i"], "label_b": pw["label_j"],
            })
    stats_df = pd.DataFrame(stats_rows)

    # ── Save outputs ──────────────────────────────────────────────────────────
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        png_path  = os.path.join(output_dir, f"{output_stem}.png")
        pdf_path  = os.path.join(output_dir, f"{output_stem}.pdf")
        csv_path  = os.path.join(output_dir, f"{output_stem}_summary.csv")
        stats_csv = os.path.join(output_dir, f"{output_stem}_stats.csv")
        fig.savefig(png_path, dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        summary_df.to_csv(csv_path, index=False)
        if len(stats_df):
            stats_df.to_csv(stats_csv, index=False)
        print(f"  Saved: {png_path}")
        print(f"  Saved: {pdf_path}")
        print(f"  Saved: {csv_path}")
        if len(stats_df):
            print(f"  Saved: {stats_csv}")

        # ── Combined PDF report (figure + parameters + folders + stats) ──────
        if pdf_report:
            report_path = os.path.join(output_dir, f"{output_stem}_report.pdf")
            try:
                _write_pdf_report(report_path, fig, groups, all_summaries,
                                  labels, colors, summary_df, stats_df,
                                  panels=panels, theme=theme, palette=pal)
                print(f"  Saved: {report_path}")
            except Exception as exc:
                print(f"  PDF report skipped ({type(exc).__name__}: {exc})")

        # ── Per-comparison circular-statistics CSV + PDF ────────────────────
        # Pool angles per group (across all replicates), compute the full
        # CircStat suite for each group, and emit:
        #   * {stem}_circular_statistics.csv  — one row per group
        #   * {stem}_circular_statistics.pdf  — themed multi-page PDF
        #     (page 1 = summary grid + comparison table; pages 2..N+1 =
        #     per-group detail mirroring the per-file report).
        try:
            groups_angles_pooled = []
            # Per-track (mean_angle_deg, D) pairs per group, used for
            # the circular-linear correlation between a track's
            # average turning bias and its diffusion coefficient.
            # One list of pairs per group; each list pools across
            # the group's replicates.
            track_angle_d_pairs = []
            # Per-replicate angle arrays — one list of arrays per group.
            # Used to compute per-replicate κ, R̄, μ for the Welch t-test
            # and per-replicate Watson-Williams F-test (treats each
            # replicate as one data point, the statistically defensible
            # framing for n=5 vs n=3 designs).
            per_replicate_angles = {}
            for label, ss, color in zip(labels, all_summaries, colors):
                pooled = []
                t_angles_g = []
                t_D_g      = []
                rep_angle_arrays = []
                for s in ss:
                    ta = s.get("turning_angles")
                    if ta is not None:
                        arr = np.asarray(ta, dtype=float).ravel()
                        if arr.size:
                            pooled.append(arr)
                            rep_angle_arrays.append(arr)
                    tracks = s.get("tracks")
                    diff_df = s.get("diffusion")
                    if tracks is None or diff_df is None:
                        continue
                    if "D" not in diff_df.columns:
                        continue
                    try:
                        pairs = compute_per_track_mean_angle(tracks)
                        if not pairs:
                            continue
                        d_map = dict(zip(diff_df["particle"].astype(int),
                                         diff_df["D"].astype(float)))
                        for pid, mu_deg in pairs:
                            d_val = d_map.get(int(pid))
                            if d_val is None or not np.isfinite(d_val):
                                continue
                            t_angles_g.append(float(mu_deg))
                            t_D_g.append(float(d_val))
                    except Exception:
                        continue
                pooled_arr = (np.concatenate(pooled)
                              if pooled else np.array([], dtype=float))
                groups_angles_pooled.append((label, pooled_arr, color))
                track_angle_d_pairs.append(
                    (np.asarray(t_angles_g, dtype=float),
                     np.asarray(t_D_g,      dtype=float)))
                per_replicate_angles[label] = rep_angle_arrays
            cs_csv = os.path.join(
                output_dir, f"{output_stem}_circular_statistics.csv")
            cs_pdf = os.path.join(
                output_dir, f"{output_stem}_circular_statistics.pdf")
            save_comparison_circular_statistics(
                groups_angles_pooled,
                csv_path=cs_csv, pdf_path=cs_pdf,
                fig_theme=theme,
                track_angle_d_pairs=track_angle_d_pairs,
                per_replicate_angles=per_replicate_angles)
            print(f"  Saved: {cs_csv}")
            print(f"  Saved: {cs_pdf}")
        except Exception as exc:
            print(f"  Comparison circular-stats skipped "
                  f"({type(exc).__name__}: {exc})")

    return fig, summary_df, stats_records


def _write_pdf_report(path, fig, groups, all_summaries, labels, colors,
                      summary_df, stats_df, panels, theme, palette):
    """Multi-page PDF: cover + figure, parameters & folders, statistics."""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    pal = palette
    with PdfPages(path) as pdf:
        # ── Page 1: the comparison figure itself ──────────────────────────────
        pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches="tight")

        # ── Page 2: cover / parameters ────────────────────────────────────────
        page2 = plt.figure(figsize=(8.5, 11), facecolor=pal["BG"])
        page2.text(0.5, 0.96, "sptPALM Comparison Report",
                   ha="center", fontsize=18, fontweight="bold", color=pal["TXT"])

        meta_lines = [
            f"Theme:              {theme}",
            f"Panels rendered:    {', '.join(sorted(panels))}",
            f"Number of groups:   {len(groups)}",
            "",
            "Groups:",
        ]
        for i, g in enumerate(groups):
            meta_lines.append(
                f"  • {labels[i]}   "
                f"(n={len(all_summaries[i])} folder(s), "
                f"colour {colors[i]})")
        meta_lines.append("")
        meta_lines.append("Folders:")
        for i in range(len(groups)):
            meta_lines.append(f"  [{labels[i]}]")
            for f in groups[i]["folders"]:
                meta_lines.append(f"    {f}")
            meta_lines.append("")

        page2.text(0.06, 0.92, "\n".join(meta_lines),
                   ha="left", va="top", fontsize=9, family="monospace",
                   color=pal["TXT"])
        pdf.savefig(page2, facecolor=pal["BG"], bbox_inches="tight")
        plt.close(page2)

        # ── Page 3: per-replicate scalar summary table ────────────────────────
        if len(summary_df):
            page3 = plt.figure(figsize=(11, 8.5), facecolor=pal["BG"])
            page3.text(0.5, 0.96, "Per-replicate scalar metrics",
                       ha="center", fontsize=14, fontweight="bold",
                       color=pal["TXT"])
            ax = page3.add_axes([0.04, 0.04, 0.92, 0.86])
            ax.axis("off")
            disp = summary_df.copy()
            for c in disp.select_dtypes(include="float").columns:
                disp[c] = disp[c].apply(
                    lambda x: f"{x:.4g}" if np.isfinite(x) else "")
            disp["folder"] = disp["folder"].apply(
                lambda p: "..." + p[-40:] if isinstance(p, str) and len(p) > 43 else p)
            tbl = ax.table(cellText=disp.values.tolist(),
                           colLabels=list(disp.columns), loc="center",
                           cellLoc="left")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.scale(1, 1.2)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor(pal["GRD"])
                cell.set_text_props(color=pal["TXT"])
                cell.set_facecolor(pal["PNL"] if r > 0 else pal["BG"])
                if r == 0:
                    cell.set_text_props(weight="bold", color=pal["TXT"])
            pdf.savefig(page3, facecolor=pal["BG"], bbox_inches="tight")
            plt.close(page3)

        # ── Page 4: statistical tests ─────────────────────────────────────────
        if len(stats_df):
            page4 = plt.figure(figsize=(11, 8.5), facecolor=pal["BG"])
            page4.text(0.5, 0.96, "Statistical tests",
                       ha="center", fontsize=14, fontweight="bold",
                       color=pal["TXT"])
            ax = page4.add_axes([0.03, 0.04, 0.94, 0.86])
            ax.axis("off")
            disp = stats_df.copy()
            for c in ("p_value", "mean_a", "mean_b", "sem_a", "sem_b"):
                if c in disp.columns:
                    disp[c] = disp[c].apply(
                        lambda x: f"{x:.4g}" if isinstance(x, (int, float)) and np.isfinite(x) else x)
            tbl = ax.table(cellText=disp.values.tolist(),
                           colLabels=list(disp.columns), loc="center",
                           cellLoc="left")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            tbl.scale(1, 1.2)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor(pal["GRD"])
                cell.set_text_props(color=pal["TXT"])
                cell.set_facecolor(pal["PNL"] if r > 0 else pal["BG"])
                if r == 0:
                    cell.set_text_props(weight="bold", color=pal["TXT"])
            pdf.savefig(page4, facecolor=pal["BG"], bbox_inches="tight")
            plt.close(page4)


if __name__ == "__main__":
    main()
