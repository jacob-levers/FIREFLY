"""Headless batch folder scan (Phase 6b).

Ports the file-discovery + series-grouping core of the Widgets
``BatchMixin._batch_rescan`` (without the QTreeWidget building) so the QML
BatchController can list analysable series.  Image chunks are grouped into one
acquisition, while external-localisation tables stay independent jobs.  No Qt.
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
_IMAGE_EXTS = (".czi", ".tif", ".tiff")
_EXTERNAL_LOC_EXTS = (".csv", ".txt", ".tsv")
# This is the ROI companion the current analysis pipeline knows how to use.
# Do not discard arbitrary ``_<word>`` recordings: those can be independent
# channels / acquisitions and silently dropping them is worse than queueing
# one extra item for the user to deselect.
_ROI_SISTER_SUFFIXES = ("_green",)


def input_kind(name: str) -> str | None:
    """Return the supported batch input kind for *name*, or ``None``.

    Keep source classification here rather than inferring it later from a
    grouped row.  That prevents a localisation table with the same stem as an
    image from accidentally becoming an image-series member.
    """
    if not name or os.path.basename(name).startswith("._"):
        return None
    n = os.path.basename(name).lower()
    if n.endswith(_IMAGE_EXTS):
        return "image" if "-tracks-z" not in n else None
    if not n.endswith(_EXTERNAL_LOC_EXTS):
        return None
    if "trcpalmtracer" in n:
        return None
    return (None if any(n.endswith(suf) for suf in _FIREFLY_AUX)
            else "external_loc")


def looks_like_input_file(name: str) -> bool:
    """Whether a filename is an analysable image or localisation table."""
    return input_kind(name) is not None


def series_key(filename: str) -> str:
    """The common *image* stem across sibling acquisition chunks.

    Kept as the historical public helper for callers/tests.  Batch grouping no
    longer applies it to localisation tables, because stripping
    ``_locPALMTracer`` there can merge a table into an unrelated image job.
    """
    stem = os.path.splitext(filename)[0]
    stem = re.sub(r"-file\d+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\(\d+\)\s*$", "", stem).rstrip()
    stem = re.sub(r"_locpalmtracer$", "", stem, flags=re.IGNORECASE)
    return stem


def _nat_key(name):
    name = os.path.basename(name)
    ext = os.path.splitext(name)[1]
    m = re.search(r"-file(\d+)" + re.escape(ext) + r"$", name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\((\d+)\)" + re.escape(ext) + r"$", name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return -1


def _human_size(n) -> str:
    """A short human file size, e.g. ``2.1 GB`` / ``312 KB`` / ``0 B``."""
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _sub_key(sub: str | None, base: str) -> str:
    """Stable user-facing key for an item below the selected folder."""
    if sub is None:
        return base
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                  re.sub(r"\.PT$", "", sub, flags=re.IGNORECASE)).strip("_") or "sub"
    return f"{safe}__{base}"


def _sort_siblings(siblings: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return sorted(siblings,
                  key=lambda p: (_nat_key(p[0]), os.path.basename(p[0]).casefold()))


def _make_series(key: str, source_type: str,
                 siblings: list[tuple[str, str]]) -> dict:
    """Build the controller/QML record for one homogeneous input series."""
    sisters = _sort_siblings(siblings)
    primary_name, primary_full = sisters[0]
    parts, total = [], 0
    for nm, pth in sisters:
        try:
            sz = os.path.getsize(pth)
        except OSError:
            sz = 0
        total += sz
        parts.append({"path": pth, "name": os.path.basename(nm),
                      "size": sz, "sizeStr": _human_size(sz),
                      "sourceType": source_type})
    return {"key": key, "primary": primary_full,
            "files": [p for _, p in sisters], "parts": parts,
            "fileCount": len(sisters), "sizeBytes": total,
            "sizeStr": _human_size(total),
            "name": os.path.basename(primary_name),
            "sourceType": source_type}


def _unique_key(proposed: str, source_type: str, used: set[str]) -> str:
    """Allocate a deterministic output key without hiding a colliding input."""
    if proposed not in used:
        used.add(proposed)
        return proposed
    tag = "locs" if source_type == "external_loc" else "image"
    candidate = f"{proposed}__{tag}"
    suffix = 2
    while candidate in used:
        candidate = f"{proposed}__{tag}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _group_series(candidates: list) -> list:
    """Group candidates into homogeneous image series or standalone tables.

    ``.czi`` preference is deliberately scoped to one normalised image stem in
    one directory.  A CZI elsewhere in the folder must never erase unrelated
    TIFF/table inputs.  Tables are intentionally one-file jobs: their stems do
    not describe image chunks and are therefore never passed as ``series_files``.
    """
    image_groups: dict[tuple[str, str], list[tuple[str, str, str | None]]] = defaultdict(list)
    tables: list[tuple[str, str, str | None]] = []
    for display, full, sub in candidates:
        kind = input_kind(os.path.basename(full))
        if kind == "image":
            # Full directory rather than display/sub keeps explicit Add-files
            # selections from different folders with the same basename apart.
            bucket = (os.path.dirname(os.path.abspath(full)),
                      series_key(os.path.basename(display)))
            image_groups[bucket].append((display, full, sub))
        elif kind == "external_loc":
            tables.append((display, full, sub))

    # ROI companion images are known by an explicit suffix only.  Preserve all
    # other underscore-suffixed names as separately analysable acquisitions.
    existing = {(folder, base.casefold()) for folder, base in image_groups}
    for folder, base in list(image_groups):
        if any(base.lower().endswith(suffix)
               and (folder, base[:-len(suffix)].casefold()) in existing
               for suffix in _ROI_SISTER_SUFFIXES):
            image_groups.pop((folder, base), None)

    pending: list[tuple[str, str, list[tuple[str, str]]]] = []
    for (_folder, base), group in image_groups.items():
        # Prefer CZI only over matching image siblings, not every input in the
        # containing directory.  This preserves the established CZI-over-derived
        # TIFF behaviour without silent cohort omission.
        czis = [(display, full) for display, full, _sub in group
                if full.lower().endswith(".czi")]
        chosen = czis or [(display, full) for display, full, _sub in group]
        sub = group[0][2]
        pending.append((_sub_key(sub, base), "image", chosen))

    # A localisation table is already a complete analysis input.  Keep its
    # original stem (including ``_locPALMTracer``) for provenance and to avoid
    # merging it into an image acquisition of the same root name.
    for display, full, sub in tables:
        stem = os.path.splitext(os.path.basename(display))[0]
        pending.append((_sub_key(sub, stem), "external_loc", [(display, full)]))

    # Preserve historical short keys for images if there is a collision, then
    # qualify the later table/image deterministically rather than dropping it.
    pending.sort(key=lambda x: (x[0].casefold(),
                                0 if x[1] == "image" else 1,
                                x[2][0][1].casefold()))
    used: set[str] = set()
    out = []
    for proposed, source_type, siblings in pending:
        out.append(_make_series(_unique_key(proposed, source_type, used),
                                source_type, siblings))
    return out


def scan_series(folder: str, recursive: bool = False) -> list:
    """Return one entry per analysable series found by walking ``folder``:
    ``[{"key", "primary", "files": [abspath…], "parts": [...], "fileCount",
    "sizeBytes", "sizeStr", "name"}]``."""
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
    return _group_series(candidates)


def scan_paths(paths: list) -> list:
    """Group an explicit list of file paths into series (for ``Add files`` /
    drag-and-drop).  Same grouping + enrichment as ``scan_series``; non-input or
    missing paths are dropped."""
    candidates = []
    for full in paths or []:
        if not full:
            continue
        cname = os.path.basename(full)
        if (os.path.isfile(full) and not cname.startswith(".")
                and looks_like_input_file(cname)):
            candidates.append((cname, full, None))
    return _group_series(candidates)
