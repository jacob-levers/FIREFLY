"""Cheap max-intensity projection for previews and the ROI background.

Reads up to ``cap`` evenly-spaced frames *directly* (never the whole stack), so a
multi-GB recording previews in well under a second and never blocks the GUI on a
full ``load_file``.  The per-frame ``(Y, X)`` layout matches what ``load_file``
returns (both yield ``T x Y x X``), so an ROI drawn on the projection maps onto
the analysis data correctly.  Returns a 2D ``float32`` array, or None.
"""
from __future__ import annotations

import os

DEFAULT_CAP = 120

# Preview/ROI colormaps. Labels mirror the Figures "Projection cmap" dropdown
# (+ a plain Grayscale). "Greys" maps to the reversed map so it reads
# white-on-dark like the rest of the dark UI. Shared by the Import preview
# thumbnail and the ROI editor background so they recolour identically.
PREVIEW_CMAPS = ["Grayscale", "Inferno", "Hot", "Viridis", "Plasma"]
_PREVIEW_CMAP_MPL = {"Inferno": "inferno", "Hot": "hot", "Viridis": "viridis",
                     "Plasma": "plasma", "Greys": "Greys_r"}


def render_projection(proj, label):
    """Render a 2D float32 projection to a QImage with the chosen colormap label.
    Grayscale (or any unknown label) goes through the shared render_frame path."""
    from firefly.ui.controllers.providers.live_frame_provider import render_frame
    if label not in _PREVIEW_CMAP_MPL:
        return render_frame(proj)
    import matplotlib
    import numpy as np
    from PySide6.QtGui import QImage
    a = np.asarray(proj, dtype=np.float32)
    finite = a[np.isfinite(a)]
    lo, hi = (np.percentile(finite, (1.0, 99.5)) if finite.size else (0.0, 1.0))
    if hi <= lo:
        hi = lo + 1.0
    a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    rgba = np.ascontiguousarray(
        (matplotlib.colormaps[_PREVIEW_CMAP_MPL[label]](a) * 255).astype(np.uint8))
    h, w = a.shape
    return QImage(rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888).copy()


def quick_frame_count(path) -> int:
    """Frame count from container metadata only — no pixel reads, so it stays
    fast even on a multi-GB recording over a network drive.  Returns 0 when the
    count can't be determined cheaply (CSV loc tables, unreadable files)."""
    if not path:
        return 0
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".tif", ".tiff"):
            import tifffile
            with tifffile.TiffFile(path) as t:
                return len(t.pages)
        if ext == ".czi":
            from aicspylibczi import CziFile
            czi = CziFile(path)
            dims = czi.dims
            return int(czi.size[dims.index("T")]) if "T" in dims else 1
    except Exception:
        return 0
    return 0


def _squeeze2d(fr):
    import numpy as np
    fr = np.asarray(fr)
    while fr.ndim > 2:                          # collapse stray leading dims
        fr = fr[0]
    return fr.astype("float32")


def sampled_projection(path, mode: str = "max", cap: int = DEFAULT_CAP):
    """Projection over up to ``cap`` evenly-spaced frames of a .tif/.czi
    recording, reduced per ``mode``: ``max`` (default), ``mean``, or ``sum``
    (``blink density`` falls back to ``max``).  Returns a 2D float32 ndarray, or
    None if unreadable.  Mirrors the projection the analysis thresholds on."""
    if not (path and os.path.isfile(path)):
        return None
    import numpy as np
    m = (mode or "max").lower()

    def _reduce(frames_iter):
        acc = None
        n = 0
        for fr in frames_iter:
            fr = _squeeze2d(fr)
            n += 1
            if acc is None:
                acc = fr.copy()
            elif m == "max":
                np.maximum(acc, fr, out=acc)
            else:                              # mean / sum accumulate
                acc += fr
        if acc is None:
            return None
        if m == "mean" and n:
            acc /= float(n)
        return acc

    def _indices(nf):
        return np.unique(np.linspace(0, max(0, nf - 1), min(nf, cap)).astype(int))

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".tif", ".tiff"):
            import tifffile
            with tifffile.TiffFile(path) as t:
                idxs = _indices(len(t.pages))
                return _reduce(t.pages[int(i)].asarray() for i in idxs)
        if ext == ".czi":
            from aicspylibczi import CziFile
            czi = CziFile(path)
            dims = czi.dims
            nf = int(czi.size[dims.index("T")]) if "T" in dims else 1
            idxs = _indices(nf)
            return _reduce(np.squeeze(czi.read_image(T=int(i), C=0)[0]) for i in idxs)
    except Exception:
        return None
    return None


def sampled_max_projection(path, cap: int = DEFAULT_CAP):
    """Max-intensity projection (back-compat shim for ``sampled_projection``)."""
    return sampled_projection(path, "max", cap)


def sampled_frame(path, idx: int):
    """Read a single raw frame ``idx`` (clamped) of a .tif/.czi recording as a
    2D float32 ndarray — one page/plane read, so scrubbing stays responsive.
    Returns None if unreadable."""
    if not (path and os.path.isfile(path)):
        return None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".tif", ".tiff"):
            import tifffile
            with tifffile.TiffFile(path) as t:
                n = len(t.pages)
                i = max(0, min(int(idx), n - 1))
                return _squeeze2d(t.pages[i].asarray())
        if ext == ".czi":
            import numpy as np
            from aicspylibczi import CziFile
            czi = CziFile(path)
            dims = czi.dims
            n = int(czi.size[dims.index("T")]) if "T" in dims else 1
            i = max(0, min(int(idx), n - 1))
            return _squeeze2d(np.squeeze(czi.read_image(T=i, C=0)[0]))
    except Exception:
        return None
    return None
