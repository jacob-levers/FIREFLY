"""Region-of-interest mask building and application.

Extracted from sptpalm_analysis.py (#7); re-exported there for compatibility.
"""
from __future__ import annotations

from firefly.analysis.fa_constants import _tqdm

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from skimage import filters, exposure, morphology
from firefly.analysis.fa_preprocess import auto_threshold, preprocess_stack


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
    # The min_object_size / max_hole_size defaults (8000 / 2000 px) are tuned
    # for large frames; on a small frame (e.g. 256×256 = 65 k px) an 8000-px
    # object floor can delete a real but compact ROI entirely, leaving an
    # all-False mask that then drops every localisation.  Scale the absolute
    # floors down for small frames (never up — large frames keep the
    # defaults) so a genuine small structure survives.
    _area = int(proj.size)
    eff_min_obj  = min(int(min_object_size), max(8, int(_area * 0.02)))
    eff_max_hole = min(int(max_hole_size),  max(1, int(_area * 0.02)))

    raw = smoothed > t
    raw_signal = raw  # pre-morphology thresholded mask, for the fallback below
    try:    raw = binary_opening(raw, disk(int(opening_radius)))
    except Exception: pass
    try:    mask = binary_closing(raw, disk(int(closing_radius)))
    except Exception: mask = raw
    try:
        mask = remove_small_holes(mask, area_threshold=eff_max_hole)
    except TypeError:
        mask = remove_small_holes(mask, eff_max_hole)
    except Exception:
        pass
    try:
        mask = remove_small_objects(mask, min_size=eff_min_obj)
    except TypeError:
        mask = remove_small_objects(mask, eff_min_obj)
    except Exception:
        pass

    # Fallback: if cleanup removed everything but there WAS signal above the
    # threshold, keep the single largest connected component of the raw mask.
    # Guarantees we never return an empty ROI when a thresholdable region
    # exists (which would otherwise crash linking on zero localisations).
    if not mask.any() and raw_signal.any():
        try:
            from skimage.measure import label as _label
            lbl = _label(raw_signal, connectivity=2)
            sizes = _np.bincount(lbl.ravel())
            sizes[0] = 0
            if sizes.max() > 0:
                mask = (lbl == int(_np.argmax(sizes)))
                print(f"  ROI: morphology cleanup emptied the mask; kept the "
                      f"largest thresholded component ({int(sizes.max()):,} px) "
                      f"instead.")
        except Exception:
            mask = raw_signal

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
