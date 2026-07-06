"""Sister-TIFF ROI helpers (fa_roi.find_sister_roi_path / build_sister_roi_mask).

These back BOTH the analysis run (firefly_worker) and the ROI-viewer preview, so
the preview is exactly the region the run keeps. Cover path detection, the
value-based mask-vs-grayscale decision, the grayscale cell segmentation
(threshold → fill holes → keep largest) and resolution handling.
"""
import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from firefly.analysis.fa_roi import find_sister_roi_path, build_sister_roi_mask


def _write_tif(path, arr):
    tifffile.imwrite(str(path), arr)
    return str(path)


# ── path detection ───────────────────────────────────────────────────────────
def test_find_sister_basic(tmp_path):
    _write_tif(tmp_path / "cell.tif", np.zeros((8, 8), np.uint16))
    green = _write_tif(tmp_path / "cell_green.tif", np.zeros((8, 8), np.uint16))
    assert find_sister_roi_path(str(tmp_path / "cell.tif"), "_green") == green


def test_find_sister_strips_palmtracer_fileNNN(tmp_path):
    green = _write_tif(tmp_path / "cell_green.tif", np.zeros((8, 8), np.uint16))
    got = find_sister_roi_path(str(tmp_path / "cell-file001.tif"), "_green")
    assert got == green


def test_find_sister_missing_or_no_suffix(tmp_path):
    _write_tif(tmp_path / "cell.tif", np.zeros((8, 8), np.uint16))
    assert find_sister_roi_path(str(tmp_path / "cell.tif"), "_green") is None
    assert find_sister_roi_path(str(tmp_path / "cell.tif"), "") is None


# ── already-a-mask branch (detected by distinct levels, not coverage) ─────────
def test_binary_mask_used_directly(tmp_path):
    arr = np.zeros((200, 200), np.uint16)
    arr[40:80, 40:80] = 7                        # 1600 px, 2 distinct values
    path = _write_tif(tmp_path / "seg_green.tif", arr)
    mask, note = build_sister_roi_mask(path, target_shape=(200, 200))
    assert mask is not None and mask.dtype == bool
    assert int(mask.sum()) == 1600               # used verbatim, no thresholding
    assert "used directly" in note


def test_large_binary_mask_not_misrouted(tmp_path):
    # Regression: a real mask that fills MOST of the frame must still be used
    # directly — the old "<40% non-zero" heuristic mis-sent this to the
    # (hollowing) threshold path.
    arr = np.zeros((200, 200), np.uint16)
    arr[:, :140] = 1                             # 70% coverage, 2 distinct values
    path = _write_tif(tmp_path / "big_green.tif", arr)
    mask, note = build_sister_roi_mask(path, target_shape=(200, 200))
    assert int(mask.sum()) == 200 * 140
    assert "used directly" in note


def test_multiframe_mask_is_max_projected(tmp_path):
    stack = np.zeros((3, 100, 100), np.uint16)
    stack[1, 10:30, 10:30] = 9                   # only on frame 1
    path = _write_tif(tmp_path / "mf_green.tif", stack)
    mask, note = build_sister_roi_mask(path, target_shape=(100, 100))
    assert mask is not None and int(mask.sum()) == 400   # 20x20 survives max-proj


# ── grayscale overview branch (segment the cell footprint) ────────────────────
def _grayscale_cell(rng, hole=False):
    img = rng.integers(1000, 3000, (200, 200)).astype(np.uint16)   # noisy dark bg
    img[40:160, 40:160] = rng.integers(40000, 60000, (120, 120))   # bright cell
    if hole:
        img[90:110, 90:110] = rng.integers(1000, 3000, (20, 20))   # dark nucleus
    return img


def test_grayscale_segments_cell(tmp_path):
    img = _grayscale_cell(np.random.default_rng(0))
    path = _write_tif(tmp_path / "fluo_green.tif", img)
    mask, note = build_sister_roi_mask(path, target_shape=(200, 200))
    assert mask is not None
    assert mask[100, 100] and not mask[5, 5]     # cell in, background out
    assert 0.20 < mask.mean() < 0.55             # ~the 120x120 cell, not whole frame
    assert "threshold" in note


def test_grayscale_fills_interior_holes(tmp_path):
    # A dim nucleus inside the cell must NOT punch a hole in the ROI.
    img = _grayscale_cell(np.random.default_rng(1), hole=True)
    path = _write_tif(tmp_path / "fluo_green.tif", img)
    mask, note = build_sister_roi_mask(path, target_shape=(200, 200))
    assert mask[100, 100]                        # the nucleus hole is filled
    assert "holes filled" in note


def test_grayscale_keeps_largest_region(tmp_path):
    rng = np.random.default_rng(2)
    img = rng.integers(1000, 3000, (200, 200)).astype(np.uint16)
    img[30:150, 30:150] = 55000                  # big cell (120x120)
    img[170:185, 170:185] = 55000                # small bright speck (15x15)
    path = _write_tif(tmp_path / "two_green.tif", img)
    mask, _ = build_sister_roi_mask(path, target_shape=(200, 200))
    assert mask[90, 90] and not mask[177, 177]   # only the large region kept


# ── resolution handling ───────────────────────────────────────────────────────
def test_matched_aspect_is_resized(tmp_path):
    arr = np.zeros((100, 100), np.uint16); arr[20:80, 20:80] = 5   # mask, half res
    path = _write_tif(tmp_path / "half_green.tif", arr)
    mask, note = build_sister_roi_mask(path, target_shape=(200, 200))
    assert mask is not None and mask.shape == (200, 200)
    assert mask.sum() > 0 and "resized" in note


def test_different_aspect_is_skipped(tmp_path):
    arr = np.zeros((100, 50), np.uint16); arr[10:20, 10:20] = 5
    path = _write_tif(tmp_path / "wrongfield_green.tif", arr)
    mask, note = build_sister_roi_mask(path, target_shape=(200, 200))
    assert mask is None and "different field" in note


def test_missing_file_returns_reason(tmp_path):
    mask, note = build_sister_roi_mask(str(tmp_path / "nope_green.tif"),
                                       target_shape=(50, 50))
    assert mask is None and "could not load" in note
