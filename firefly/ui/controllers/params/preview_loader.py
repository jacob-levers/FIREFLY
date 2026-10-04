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
                # len(t.pages) UNDERCOUNTS contiguous / ImageJ-hyperstack TIFFs,
                # which tifffile stores as a single IFD describing an N-frame block
                # (so len(pages)==1 for an N-frame stack).  Take the frame count
                # from the series shape (product of the non-YX dims) and fall back
                # to the page count.
                pages = len(t.pages)
                frames = pages
                try:
                    shp = t.series[0].shape
                    if len(shp) >= 3:
                        import numpy as _np
                        frames = int(_np.prod(shp[:-2]))
                except Exception:
                    pass
                return max(frames, pages)
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


def _czi_plane(czi, idx, channel):
    import numpy as np
    dims = dict(zip(czi.dims, czi.size))
    ch = min(int(channel), int(dims.get('C', 1)) - 1)  # same clamp as load_czi
    return np.squeeze(czi.read_image(T=int(idx), C=ch)[0])


def detection_frame(path, idx, channel=0, czi=None):
    """Read an exact raw plane for detection; never collapse unknown dimensions.

    Index is local to the selected file. Unsupported layouts fail explicitly
    instead of displaying a different plane as a faithful detection preview.
    ``czi`` reuses an already-open CziFile (opening one costs as much as
    reading two planes).
    """
    import numpy as np
    ext = os.path.splitext(path)[1].lower()
    if ext == '.czi':
        if czi is None:
            from aicspylibczi import CziFile
            czi = CziFile(path)
        frame = _czi_plane(czi, idx, channel)
    elif ext in ('.tif', '.tiff'):
        import tifffile
        with tifffile.TiffFile(path) as tif:
            series = tif.series[0]
            if len(series.shape) > 3 or series.axes[-2:] != 'YX':
                raise ValueError('Detection preview requires a planar YX/TYX TIFF; this layout is unsupported.')
            n = series.shape[0] if len(series.shape) == 3 else 1
            if not 0 <= int(idx) < n: raise ValueError('Frame outside TIFF series')
            if len(series.pages) == n:
                frame = series.pages[int(idx)].asarray()
            else:
                # Contiguous TIFF blocks may have only one physical IFD. Map
                # them without loading the complete recording into RAM.
                mapped = tifffile.memmap(path, series=0, mode='r')
                frame = np.array(mapped[int(idx)] if mapped.ndim == 3 else mapped)
    else:
        raise ValueError('Detection preview supports raw CZI and planar TIFF files.')
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 2:
        raise ValueError('Ambiguous image dimensions: cannot preview the production detector safely.')
    return frame


class DetectionFrames:
    """A recording's raw detection planes, read on demand.

    Enough of an array — ``len``, ``shape``, integer and slice indexing — for
    estimators that sample a few frames or windows (the auto threshold, the
    wavelet noise) without loading a multi-GB recording.  Every plane is
    exactly :func:`detection_frame`'s; a CZI is opened once, not per plane.
    """
    ndim = 3

    def __init__(self, path, n_frames, channel=0):
        self._path, self._channel = str(path), int(channel)
        self._czi = None
        if os.path.splitext(self._path)[1].lower() == '.czi':
            from aicspylibczi import CziFile
            self._czi = CziFile(self._path)
        first = self._plane(0)
        self.shape = (int(n_frames),) + tuple(first.shape)

    def _plane(self, i):
        return detection_frame(self._path, int(i), self._channel, czi=self._czi)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        import numpy as np
        if isinstance(key, slice):
            idx = range(*key.indices(len(self)))
            if not len(idx):
                return np.empty((0,) + self.shape[1:], dtype=np.float32)
            return np.stack([self._plane(i) for i in idx])
        i = int(key)
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(f"frame {key} outside 0-{len(self) - 1}")
        return self._plane(i)
