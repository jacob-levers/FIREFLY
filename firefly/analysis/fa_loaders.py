"""CZI / TIF / external-localisation loaders for the FIREFLY pipeline.

Extracted from sptpalm_analysis.py (#7); re-exported there so existing
`sptpalm_analysis.load_file(...)` etc. call sites keep working.
"""
from __future__ import annotations

import glob
import os
import re
import sys
import time
import warnings
import io as _io
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from firefly.analysis.fa_constants import N_CPUS, _Cancelled, _dim_size, _tqdm
from firefly.analysis.fa_memory import (_alloc_or_memmap_stack, _register_temp_stack_path,
                       _resolve_temp_stack_dir, _user_ram_reserve_gb)


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
    "TrackMate": {
        # TrackMate (Fiji) export — column names are uppercase with
        # underscores in the "machine-readable" header row (`FRAME`,
        # `POSITION_X`, …); the file also contains 2 follow-up human-
        # readable header rows that the importer's multi-row header
        # detection skips automatically.
        "frame":         ("FRAME", "frame", "Frame"),
        "frame_offset":  0,                # TrackMate 7+ is 0-indexed
        "x":             ("POSITION_X", "Position X", "X", "x"),
        "y":             ("POSITION_Y", "Position Y", "Y", "y"),
        "xy_unit":       "um",             # spatial cols are in micrometres
        "mass":          ("MEAN_INTENSITY_CH1", "TOTAL_INTENSITY_CH1",
                          "Mean intensity ch1", "Sum intensity ch1",
                          "QUALITY", "Quality"),
        # NEW — when a TRACK_ID column is present, the importer ALSO
        # emits a `particle` column on the output DataFrame so the
        # worker can skip its own linker and analyse TrackMate's
        # tracks directly.  Unlinked spots (TRACK_ID empty / "None")
        # are dropped.
        "particle":      ("TRACK_ID", "Track ID"),
    },
}


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


def _parse_czi_metadata(xml):
    """Extract pixel size (µm) and frame interval (s) from CZI metadata.

    Accepts the metadata as an ElementTree Element (what aicspylibczi's
    `czi.meta` returns), an XML string (czifile's `metadata()`), or bytes.
    Earlier this only accepted a string, so the aicspylibczi path silently
    failed (ET.fromstring on an Element raises) and every CZI reported no
    pixel size / frame interval even though ZEN/Fiji read them fine.
    """
    meta = {"pixel_size_um": None, "frame_interval_s": None}
    if xml is None:
        return meta
    try:
        if isinstance(xml, (bytes, bytearray)):
            xml = xml.decode("utf-8", "replace")
        root = ET.fromstring(xml) if isinstance(xml, str) else xml
    except Exception:
        return meta

    # ── Pixel size: Scaling/Items/Distance Id="X"|"Y", Value in metres ──
    try:
        for dist in root.iter("Distance"):
            if dist.get("Id", "") in ("X", "Y"):
                el = dist.find("Value")
                if el is not None and el.text:
                    try:
                        val = float(el.text)
                        if 1e-9 < val < 1e-3:
                            meta["pixel_size_um"] = round(val * 1e6, 6)
                            break
                    except (TypeError, ValueError):
                        pass
    except Exception:
        pass

    # ── Frame interval ──────────────────────────────────────────────────
    # Prefer <FrameTime> (seconds) — what ZEN writes for time-lapse and what
    # Fiji surfaces.  Fall back to a unit-aware <TimeSpan>/<Interval> (ZEN
    # stores these in ms), then the legacy <TimeIncrement>/<Increment>.
    def _set_fi(v):
        if v is not None and 1e-6 < v < 3600:
            meta["frame_interval_s"] = round(v, 6)
            return True
        return False

    try:
        ft = root.find(".//FrameTime")
        if ft is not None and ft.text:
            _set_fi(float(ft.text))                      # already seconds
    except (TypeError, ValueError):
        pass

    if meta["frame_interval_s"] is None:
        _UNIT = {"ms": 1e-3, "µs": 1e-6, "us": 1e-6, "s": 1.0, "sec": 1.0}
        for node in list(root.iter("TimeSpan")) + list(root.iter("Interval")):
            try:
                vel = node.find("Value")
                if vel is None or not vel.text:
                    continue
                v = float(vel.text)
                if v <= 0:
                    continue
                unit = (node.findtext("DefaultUnitFormat") or "s").strip().lower()
                if _set_fi(v * _UNIT.get(unit, 1.0)):
                    break
            except (TypeError, ValueError):
                continue

    if meta["frame_interval_s"] is None:
        for tag in ("TimeIncrement", "Increment"):
            el = root.find(f".//{tag}")
            if el is not None:
                text = el.text or (el.find("Value").text
                                   if el.find("Value") is not None else None)
                if text:
                    try:
                        if _set_fi(float(text)):
                            break
                    except (TypeError, ValueError):
                        pass
    return meta


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
        # Read first frame to discover H×W.
        img0, _ = czi.read_image(T=0, C=ch)
        f0 = img0.squeeze()
        if f0.ndim > 2:
            f0 = f0[0]
        H, W  = f0.shape

        # Fast path: a single read_image() pulls the entire T stack in one
        # libCZI call.  Per-frame read_image(T=t) carries a large fixed
        # per-call cost (subblock-directory parsing), so looping it over
        # thousands of frames takes minutes — one bulk call is sub-second
        # (~300x faster on a 16 000-frame 256×256 stack).
        #
        # The bulk path briefly holds BOTH the uint16 array and its float32
        # copy (~1.5x the final stack), while the per-frame fallback peaks at
        # ~1x.  On tight-RAM machines (e.g. a 16 GB Mac) a large stack can fit
        # per-frame but not in bulk, so gate the fast path on free memory —
        # macOS swaps rather than raising MemoryError, so we must check up
        # front instead of relying on a failed allocation.
        bytes_f32 = n_t * H * W * 4
        peak_need = bytes_f32 + n_t * H * W * int(f0.dtype.itemsize)
        try:
            import psutil as _psutil
            avail   = _psutil.virtual_memory().available
            reserve = _user_ram_reserve_gb() * (1024 ** 3)
            bulk_ok = peak_need < (avail - reserve)
        except Exception:
            bulk_ok = bytes_f32 < 2 * (1024 ** 3)   # can't measure → only if small

        stack = None
        if bulk_ok:
            try:
                bulk, _ = czi.read_image(C=ch)
                arr = np.asarray(bulk).squeeze()
                while arr.ndim > 3:        # drop residual leading singleton dims
                    arr = arr[0]
                if arr.ndim == 2:          # single timepoint
                    arr = arr[None, ...]
                if arr.shape == (n_t, H, W):
                    stack = np.ascontiguousarray(arr, dtype=np.float32)
                    print(f"  Loaded {n_t} frames (bulk read).", flush=True)
                else:
                    print(f"  Bulk read shape {arr.shape} != expected "
                          f"{(n_t, H, W)}; using per-frame read.", flush=True)
                del bulk, arr
            except _Cancelled:
                raise
            except Exception as exc:
                print(f"  Bulk CZI read unavailable ({exc}); using per-frame "
                      f"read.", flush=True)
        else:
            print(f"  Limited free RAM (~{peak_need/1e9:.1f} GB needed for a "
                  f"bulk read); using memory-frugal per-frame read.",
                  flush=True)

        if stack is None:
            # Pre-allocate the full array (avoids a Python list + np.stack
            # that would double peak RAM) and fill it frame by frame.  Uses a
            # disk-backed memmap when the stack won't fit in RAM, so a single
            # CZI larger than memory loads instead of OOM-ing / swapping.
            stack = _alloc_or_memmap_stack((n_t, H, W))
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


def _probe_tif_shape_and_count(path: str):
    """Read just enough of a TIF to return (n_pages, (H, W))."""
    with tifffile.TiffFile(path) as tif:
        n = len(tif.pages)
        sample = tif.pages[0].asarray()
        H, W = sample.shape[-2:]
    return n, (int(H), int(W))


def _stream_tif_into(path, dest, offset, stop_event=None, chunk=1000):
    """Read a TIF's frames in `chunk`-page blocks straight into
    `dest[offset:offset+n]` (a RAM array OR a np.memmap), so the WHOLE source
    file is never materialised at once.  This is what lets a multi-file series
    whose *final* stack fits in RAM stay on the fast in-RAM path instead of
    demoting to a disk memmap over a brief per-file spike.

    Returns (n_frames_written, px_um, fi_s).  Raises on layouts it can't
    stream cleanly (multi-dim OME, shape drift) so the caller can fall back to
    a full `_load_single_tif` + copy.
    """
    with tifffile.TiffFile(path) as tif:
        px_um, fi_s = _parse_ome_metadata(tif)
        n_pages = len(tif.pages)
        local = 0
        for start in range(0, n_pages, chunk):
            if stop_event is not None and stop_event.is_set():
                raise _Cancelled()
            end = min(start + chunk, n_pages)
            try:
                block = tif.asarray(key=range(start, end), maxworkers=N_CPUS)
            except TypeError:
                block = tif.asarray(key=range(start, end))
            if block.ndim == 2:
                block = block[np.newaxis]
            elif block.ndim == 4:
                block = block[:, 0] if block.shape[1] == 1 else block.mean(axis=1)
            if block.ndim != 3:
                raise ValueError(f"unexpected chunk ndim {block.ndim}")
            n_here = block.shape[0]
            dest[offset + local: offset + local + n_here] = block.astype(
                dest.dtype, copy=False)
            local += n_here
            del block
    return local, px_um, fi_s


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
    # Each source is STREAMED into the combined array in _STREAM_CHUNK-frame
    # blocks (see the copy loop), so the transient on top of the combined
    # stack is just one chunk — not a whole source file.  Basing the RAM-vs-
    # memmap decision on this realistic peak keeps a series whose final stack
    # fits in RAM on the fast in-RAM path (the old `+ largest source` term
    # demoted big multi-file series to disk memmap over a brief spike).
    _STREAM_CHUNK = 1000
    chunk_gb = (min(_STREAM_CHUNK, max(n_per_file)) * bytes_per_frame) / 1e9
    peak_gb  = total_gb + chunk_gb
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
              f"{total_gb:.1f} GB + stream chunk {chunk_gb:.1f} GB). "
              f"Free: {free_disp} GB, reserve: {reserve_gb:.1f} GB. "
              f"→ disk memmap at {tmp_path} "
              f"(override reserve via FIREFLY_USER_RAM_RESERVE_GB).",
              flush=True)
        combined = np.memmap(tmp_path, dtype=np.float32, mode="w+",
                             shape=(n_total, H, W))
    else:
        free_disp = f"{free_gb:.1f}" if free_gb is not None else "?"
        print(f"  Peak RAM needed: {peak_gb:.1f} GB (combined "
              f"{total_gb:.1f} GB + stream chunk {chunk_gb:.1f} GB). "
              f"Free: {free_disp} GB, reserve: {reserve_gb:.1f} GB. "
              f"→ in-RAM allocation (fast path).", flush=True)
        combined = np.empty((n_total, H, W), dtype=np.float32)

    # Stream each file directly into the destination slice in chunks so the
    # whole source is never held in RAM (the per-file spike that used to demote
    # large series to disk memmap).  Fall back to a full load + copy for any
    # file the streamer can't handle.
    px_um_out = None
    fi_s_out  = None
    offset = 0
    import gc as _gc
    for i, fpath in enumerate(series):
        print(f"  [{i+1}/{len(series)}] {os.path.basename(fpath)}",
              flush=True)
        try:
            n_written, px, fi = _stream_tif_into(
                fpath, combined, offset, stop_event, chunk=_STREAM_CHUNK)
        except _Cancelled:
            raise
        except Exception as exc:
            print(f"  (chunked stream failed — {exc}; full-load fallback)",
                  flush=True)
            st, px, fi = _load_single_tif(fpath, stop_event)
            combined[offset:offset + st.shape[0]] = st.astype(
                combined.dtype, copy=False)
            n_written = st.shape[0]
            del st
        if i == 0:
            px_um_out = px
            fi_s_out  = fi
        offset += n_written
        _gc.collect()
    # Defensive: trim if the probed total over-counted (shouldn't, but a
    # mid-series shape change could leave trailing unwritten frames).
    if offset < combined.shape[0]:
        combined = combined[:offset]
    if use_memmap:
        try:    combined.flush()
        except Exception: pass
    print(f"  Combined shape: {combined.shape}  (T x Y x X)", flush=True)
    if px_um_out is not None: print(f"  Pixel size  : {px_um_out} µm  (from file metadata)")
    if fi_s_out is not None:  print(f"  Frame interval: {fi_s_out} s  (from file metadata)")
    return combined, px_um_out, fi_s_out


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
        #      mostly letters / unit-literals, not just numbers.
        #      Otherwise we'd pick a data row whenever the metadata
        #      block was a different width from the table.
        # Among all rows that satisfy both, take the FIRST one.
        def _looks_like_header(fields):
            # A header row's NON-EMPTY fields are mostly non-numeric
            # tokens (column names, parenthesised units like
            # "(micron)", or human-readable labels with spaces).
            # We ignore empty fields entirely so TrackMate's units row
            # (",,,,(micron),(micron),,(counts)") still qualifies even
            # though half of its cells are blank.
            if not fields:
                return False
            non_empty = [f.strip() for f in fields if f.strip()]
            if not non_empty:
                return False
            letter_count = sum(
                1 for f in non_empty
                if any(c.isalpha() for c in f))
            # >= half of NON-EMPTY fields must look like tokens, and
            # there must be at least 2 such fields overall.
            return (letter_count >= max(2, (len(non_empty) + 1) // 2))

        split_rows = [_split(ln) for ln in preview]
        widths = [len(r) for r in split_rows]
        max_w  = max(widths) if widths else 0
        header_line = 0
        for i, row in enumerate(split_rows):
            if len(row) == max_w and _looks_like_header(row):
                header_line = i
                break
        # Multi-row header skipping — TrackMate exports a 3-row header
        # block (machine names / human names / units like "(micron)"),
        # only the FIRST of which has the column names we recognise.
        # After picking the machine-name row, walk forward and skip
        # subsequent rows that ALSO look like headers (mostly non-
        # numeric) so pandas starts reading from genuine data.  Each
        # extra skipped row becomes a `skiprows` entry below.
        skip_extra: list[int] = []
        if header_line < len(split_rows):
            j = header_line + 1
            while j < len(split_rows):
                row = split_rows[j]
                if len(row) == max_w and _looks_like_header(row):
                    skip_extra.append(j)
                    j += 1
                else:
                    break
        if header_line > 0:
            print(f"  Skipping {header_line} metadata row(s) before the "
                  f"data table.")
        if skip_extra:
            print(f"  Skipping {len(skip_extra)} extra header row(s) "
                  f"after the column-name row (TrackMate-style).")

    # Try comma → tab → python-engine sniff.  First attempt that yields
    # more than one column wins.  `skiprows` is the metadata-block
    # size detected above; for TrackMate-style multi-row headers, we
    # also drop the extra header rows that follow the column-name row.
    # Note: pandas treats the first NON-skipped row as the header by
    # default — so we skip `header_line` rows BEFORE the column-name
    # row, then pass a `skiprows` callable that ALSO drops the extra
    # header rows AFTER it without disturbing pandas' header pick.
    df = None
    last_exc: Exception | None = None
    # Build the `skiprows` set in source-file row coordinates.
    # `header_line` rows before the header are unconditionally dropped
    # via the int form; extra header rows AFTER the column-name row
    # are dropped via a callable so pandas still treats the
    # column-name row as the header.
    if skip_extra:
        # row 0 of pandas == csv row `header_line`; the column-name
        # row is at csv index header_line, and pandas will consume
        # IT as the header.  The "extra" rows are at csv indices
        # `header_line + 1`, `header_line + 2`, …  After pandas has
        # skipped `header_line` rows and read the header from the
        # next row, the data rows it sees are numbered 0, 1, 2, …
        # which map back to csv rows header_line+1, header_line+2…
        # So we need to skip data-row indices 0, 1, …, len(skip_extra)-1.
        skip_extra_data_rows = set(range(len(skip_extra)))
        _skiprows = lambda i, _h=header_line, _s=skip_extra_data_rows: (
            i < _h or (i - _h - 1) in _s
            if i > _h else False
        )
    else:
        _skiprows = header_line
    for kwargs in (
        {"sep": "\t",  "engine": "c"},
        {"sep": ",",   "engine": "c"},
        {"sep": None,  "engine": "python"},
    ):
        # `skiprows=callable` requires the python engine.
        if callable(_skiprows) and kwargs.get("engine") == "c":
            continue
        try:
            attempt = _pd.read_csv(
                csv_path, skiprows=_skiprows, **kwargs)
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
        for canonical in ("frame", "x", "y", "mass", "particle"):
            for col in spec.get(canonical, ()):
                if col in df.columns:
                    mapping[canonical] = col
                    break
        # frame, x, y are required; mass + particle are optional
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
    _xy_unit = (spec.get("xy_unit") or "px").lower()
    if _xy_unit == "nm":
        x = x / (pixel_size_um * 1000.0)
        y = y / (pixel_size_um * 1000.0)
    elif _xy_unit in ("um", "µm", "micron", "microns"):
        # TrackMate-style: spatial columns in micrometres.  Convert to
        # pixels using the user's pixel size.
        x = x / float(pixel_size_um)
        y = y / float(pixel_size_um)
    out["x"] = x
    out["y"] = y

    if "mass" in mapping:
        out["mass"] = df[mapping["mass"]].astype(float).values
    else:
        # Detection already happened upstream; downstream stages tolerate
        # a constant mass column.  Filter-by-mass becomes a no-op which
        # is the right behaviour for pre-filtered external data.
        out["mass"] = 1.0

    # Optional: pass through pre-linked track IDs.  Allows the caller
    # to skip its own linker entirely and analyse the externally-
    # produced tracks directly (e.g. "TrackMate detection + linking,
    # FIREFLY analytics").  TRACK_ID may be empty / "None" / "" /
    # a negative integer for unlinked spots — those rows are dropped
    # before the canonical `particle` column is written.
    if "particle" in mapping:
        try:
            raw = df[mapping["particle"]]
            # Coerce to numeric; non-numeric strings ("None", "") → NaN.
            pid = _pd.to_numeric(raw, errors="coerce")
            # TrackMate uses -1 or absent for unlinked spots.
            valid = pid.notna() & (pid >= 0)
            n_dropped = int((~valid).sum())
            if n_dropped:
                print(f"  Dropping {n_dropped:,} unlinked spots "
                      f"(empty / 'None' / negative TRACK_ID)")
            out = out.loc[valid.values].reset_index(drop=True)
            pid = pid.loc[valid].astype("int64").values
            out["particle"] = pid
            print(f"  Pre-linked tracks detected: "
                  f"{int(_pd.Series(pid).nunique()):,} unique track IDs — "
                  f"the worker will skip its own linker and analyse "
                  f"these tracks directly.")
        except Exception as exc:
            print(f"  WARN: TRACK_ID column found but could not be "
                  f"parsed as integers: {exc}.  Falling back to "
                  f"re-linking via FIREFLY's linker.")

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
