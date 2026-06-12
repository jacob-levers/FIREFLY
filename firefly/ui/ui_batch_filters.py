"""Pure, Qt-free predicates for the batch folder scanner (``_batch_rescan``).

These live in their own module — with no PySide6 import — so the recursive-scan
filtering rules can be unit-tested in CI, where only the analysis stack is
installed (no Qt / napari).  ``ui_mixin_batch`` imports them for the live scan.
"""
from __future__ import annotations

import re

# palmTRACER writes its analysis products into ``<acquisition>.PT`` folders and
# under numbered ``NN_Analysis…`` stage folders (e.g. ``02_Analysis_correct
# pixel setting``), sitting alongside the raw ``01_Raw`` acquisition folders.
# When a recursive batch walks a whole experiment tree we must skip those
# folders so derived outputs (loc/track tables, ``-Tracks-…`` map TIFFs,
# motion-corrected stacks) are never queued and analysed as if they were raw
# acquisitions.
_ANALYSIS_DIR_RE = re.compile(r"\d{2}_analysis", re.IGNORECASE)

_RAW_IMAGE_EXTS = (".tif", ".tiff", ".czi")


def is_analysis_output_dir(name: str) -> bool:
    """True if ``name`` is a palmTRACER analysis-output directory that a
    recursive raw-image scan should prune (not descend into).

    Matches ``<x>.PT`` analysis folders and numbered ``NN_Analysis…`` stage
    folders.  Deliberately does NOT match ``01_Raw`` (the raw acquisitions) or
    ordinary experiment / condition / cell folders, so only derived data is
    skipped.
    """
    nl = name.lower()
    return nl.endswith(".pt") or _ANALYSIS_DIR_RE.match(nl) is not None


def is_raw_image_name(name: str) -> bool:
    """True if ``name`` is a raw image acquisition (TIFF / CZI).

    Used to restrict the *recursive* batch auto-queue to raw images: external
    localisation tables (``.csv`` / ``.txt``) are abundant in analysis trees
    and are not raw data, so they are excluded when recursing a whole directory.
    One-level batch mode keeps the full CSV/TXT-capable filter.
    """
    return name.lower().endswith(_RAW_IMAGE_EXTS)


def is_palmtracer_loc_table(name: str) -> bool:
    """True if ``name`` is a palmTRACER localisation table — ``locPALMTracer.txt``
    / ``.csv`` (palmTRACER native) or FIREFLY's ``<stem>_locPALMTracer.csv``
    export.  Used by the batch 'palmTRACER data' input mode to surface one unit
    per analysed acquisition (the loc table inside each ``.PT`` folder), while
    ignoring the sibling track / D / MSD tables.
    """
    n = name.lower()
    return n.endswith("locpalmtracer.txt") or n.endswith("locpalmtracer.csv")
