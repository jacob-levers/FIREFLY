"""Headless batch folder scan (Phase 6b).

Ports the file-discovery + series-grouping core of the Widgets
``BatchMixin._batch_rescan`` (without the QTreeWidget building) so the QML
BatchController can list analysable series.  Groups split-TIFF / sister files
into one series per acquisition, prefers a ``.czi`` over derived siblings, and
drops palmTRACER ROI sister images that have a real companion.  No Qt.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict

from firefly.ui.ui_batch_filters import is_analysis_output_dir, is_raw_image_name

_FIREFLY_AUX = (
    "_run_manifest.json", "_diffusion_summary.csv", "_ensemble_msd.csv",
    "_trajectories.csv", "_localisations.csv", "_drift.csv", "_dwell_times.csv",
    "_turning_angles.csv", "_mobile_fraction.csv", "_cluster_labels.csv",
    "_cluster_stats.csv", "_postproc_input.csv", "_circular_statistics.csv",
)


def looks_like_input_file(name: str) -> bool:
    """Whether a filename is an analysable input (image stack or loc table) —
    ported verbatim from MainWindow._looks_like_input_file."""
    if name.startswith("._"):
        return False
    n = name.lower()
    if n.endswith((".czi", ".tif", ".tiff")):
        return "-tracks-z" not in n
    if not n.endswith((".csv", ".txt")):
        return False
    if "trcpalmtracer" in n:
        return False
    return not any(n.endswith(suf) for suf in _FIREFLY_AUX)


def series_key(filename: str) -> str:
    """The common stem across sibling files of one acquisition (strips ImageJ
    (N) / palmTRACER -fileNNN / FIREFLY _locPALMTracer)."""
    stem = os.path.splitext(filename)[0]
    stem = re.sub(r"-file\d+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\(\d+\)\s*$", "", stem).rstrip()
    stem = re.sub(r"_locpalmtracer$", "", stem, flags=re.IGNORECASE)
    return stem


def _nat_key(name):
    ext = os.path.splitext(name)[1]
    m = re.search(r"-file(\d+)" + re.escape(ext) + r"$", name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\((\d+)\)" + re.escape(ext) + r"$", name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return -1


def scan_series(folder: str, recursive: bool = False) -> list:
    """Return one entry per analysable series:
    ``[{"key", "primary", "files": [abspath…], "fileCount"}]``."""
    if not folder or not os.path.isdir(folder):
        return []
    candidates = []          # (display, full, sub)
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = sorted(
            d for d in dirnames
            if not d.startswith(".")
            and d.lower() not in ("batch_results", "compare_results")
            and not (recursive and is_analysis_output_dir(d)))
        rel = os.path.relpath(dirpath, folder)
        if not recursive and rel != os.curdir:
            dirnames[:] = []
        sub = None if rel == os.curdir else rel.replace(os.sep, "/")
        for cname in sorted(filenames):
            if cname.startswith("."):
                continue
            if recursive and not is_raw_image_name(cname):
                continue
            full = os.path.join(dirpath, cname)
            if os.path.isfile(full) and looks_like_input_file(cname):
                candidates.append((cname if sub is None else f"{sub}/{cname}", full, sub))

    # prefer the raw .czi over derived siblings in the same folder
    by_folder = defaultdict(list)
    for c in candidates:
        by_folder[c[2]].append(c)
    candidates = []
    for grp in by_folder.values():
        czis = [c for c in grp if c[1].lower().endswith(".czi")]
        candidates.extend(czis if czis else grp)

    # group by series key (subfolder-prefixed so same-named files stay separate)
    smap = defaultdict(list)
    generic = {"locpalmtracer", "trcpalmtracer"}
    for display, full, sub in candidates:
        if sub is None:
            key = series_key(display)
        else:
            base = series_key(os.path.basename(display))
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                          re.sub(r"\.PT$", "", sub, flags=re.IGNORECASE)).strip("_") or "sub"
            key = safe if base.lower() in generic else f"{safe}__{base}"
        smap[key].append((display, full))

    # drop ROI/channel sister keys (<base>_green) when a bare <base> exists
    existing = set(smap)
    for key in list(smap):
        m = re.match(r"^(.+)_[^_]+$", key)
        if m and m.group(1) in existing:
            smap.pop(key, None)

    out = []
    for key in sorted(smap):
        sisters = sorted(smap[key], key=lambda p: _nat_key(p[0]))
        primary_name, primary_full = sisters[0]
        for nm, pth in sisters:
            if os.path.splitext(os.path.basename(nm))[0] == key:
                primary_name, primary_full = nm, pth
                break
        out.append({"key": key, "primary": primary_full,
                    "files": [p for _, p in sisters], "fileCount": len(sisters),
                    "name": os.path.basename(primary_name)})
    return out
