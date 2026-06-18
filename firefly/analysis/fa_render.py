"""Super-resolution reconstruction render (Qt-free, numpy + scipy only).

Turns a cloud of single-molecule localisations into a high-resolution image — a
2-D histogram on a fine (`sr_nm`) grid, optionally Gaussian-blurred (`blur_nm`)
to approximate per-localisation precision.  The canonical PALM/STORM deliverable.

Shared by the analysis worker (saves `figures/*_superres.png` per run) and the
Visualise tab (interactive layer), so the rendering is identical in both.
"""
from __future__ import annotations

import numpy as np

# Largest output edge (px) — guards against a tiny sr_nm blowing the image up to
# gigabytes; `sr_nm` is raised to stay within this if the field is huge.
_MAX_EDGE = 8000


def render_superres(x_px, y_px, pixel_size_um, *, sr_nm=20.0, blur_nm=20.0,
                    field_px=None):
    """Render localisations into a super-resolution image.

    Parameters
    ----------
    x_px, y_px : array-like
        Localisation centres in CAMERA pixels.
    pixel_size_um : float
        Camera pixel size (µm/px).
    sr_nm : float
        Output (super-res) pixel size in nm.  Smaller = finer (and bigger image).
    blur_nm : float
        Gaussian blur sigma in nm (≈ localisation precision).  0 → raw histogram.
    field_px : (H, W) or None
        Camera-frame size in px to fix the canvas to the full field.  If None,
        the localisations' bounding box is used.

    Returns
    -------
    img : 2-D float32 ndarray
        The reconstruction (counts, optionally blurred).  Origin lower-left
        consistent with image y, x order (row = y, col = x).
    """
    x = np.asarray(x_px, dtype=float).ravel()
    y = np.asarray(y_px, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return np.zeros((1, 1), dtype=np.float32)

    px_nm = float(pixel_size_um) * 1000.0
    sr = max(1e-3, float(sr_nm))

    if field_px is not None:
        H, W = int(field_px[0]), int(field_px[1])
        x0, x1 = 0.0, max(1.0, W) * px_nm
        y0, y1 = 0.0, max(1.0, H) * px_nm
    else:
        x0, x1 = x.min() * px_nm, x.max() * px_nm
        y0, y1 = y.min() * px_nm, y.max() * px_nm
    if x1 <= x0:
        x1 = x0 + px_nm
    if y1 <= y0:
        y1 = y0 + px_nm

    nx = max(1, int(np.ceil((x1 - x0) / sr)))
    ny = max(1, int(np.ceil((y1 - y0) / sr)))
    if max(nx, ny) > _MAX_EDGE:                      # keep memory sane
        sr *= max(nx, ny) / _MAX_EDGE
        nx = max(1, int(np.ceil((x1 - x0) / sr)))
        ny = max(1, int(np.ceil((y1 - y0) / sr)))

    ix = np.clip(((x * px_nm - x0) / sr).astype(np.int64), 0, nx - 1)
    iy = np.clip(((y * px_nm - y0) / sr).astype(np.int64), 0, ny - 1)
    img = np.zeros((ny, nx), dtype=np.float32)
    np.add.at(img, (iy, ix), 1.0)

    if blur_nm and blur_nm > 0:
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=float(blur_nm) / sr)
    return img
