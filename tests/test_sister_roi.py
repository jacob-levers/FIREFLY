"""Sister-TIFF ROI helpers (fa_roi.find_sister_roi_path / build_sister_roi_mask).

These back BOTH the analysis run (firefly_worker) and the ROI-viewer preview, so
the preview is exactly the region the run keeps. Cover the three mask branches
(non-zero segmentation, grayscale Li threshold, shape mismatch) + path detection.
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
    # palmTRACER series file -> the sister sits against the bare root name.
    green = _write_tif(tmp_path / "cell_green.tif", np.zeros((8, 8), np.uint16))
    got = find_sister_roi_path(str(tmp_path / "cell-file001.tif"), "_green")
    assert got == green


def test_find_sister_missing_or_no_suffix(tmp_path):
    _write_tif(tmp_path / "cell.tif", np.zeros((8, 8), np.uint16))
    assert find_sister_roi_path(str(tmp_path / "cell.tif"), "_green") is None
    assert find_sister_roi_path(str(tmp_path / "cell.tif"), "") is None


# ── mask branches ────────────────────────────────────────────────────────────
def test_binary_segmentation_nonzero_mask(tmp_path):
    # Mostly-zero image (<40% non-zero) → treated as a labelled segmentation.
    arr = np.zeros((200, 200), np.uint16)
    arr[40:80, 40:80] = 7                       # 1600 px = 4% of frame
    path = _write_tif(tmp_path / "seg_green.tif", arr)
    mask, note = build_sister_roi_mask(path, target_shape=(200, 200))
    assert mask is not None and mask.dtype == bool
    assert int(mask.sum()) == 1600              # exactly the non-zero region
    assert "non-zero" in note


def test_grayscale_uses_li_threshold(tmp_path):
    # A grayscale fluorescence channel (all pixels non-zero) → Li threshold.
    arr = np.full((200, 200), 6000, np.uint16)   # dim floor everywhere
    arr[40:160, 40:160] = 60000                  # 120x120 bright square (>8000 px)
    path = _write_tif(tmp_path / "fluo_green.tif", arr)
    mask, note = build_sister_roi_mask(path, target_shape=(200, 200))
    assert mask is not None
    assert mask.sum() > 0                         # the bright region survives
    # the ROI is roughly the bright square, not the whole frame
    assert 0.10 < mask.mean() < 0.60
    assert "Li threshold" in note


def test_multiframe_is_max_projected(tmp_path):
    stack = np.zeros((3, 100, 100), np.uint16)
    stack[1, 10:30, 10:30] = 9                    # only on frame 1
    path = _write_tif(tmp_path / "mf_green.tif", stack)
    mask, note = build_sister_roi_mask(path, target_shape=(100, 100))
    assert mask is not None and int(mask.sum()) == 400   # 20x20 survives the max-proj


def test_shape_mismatch_is_skipped(tmp_path):
    arr = np.zeros((100, 100), np.uint16); arr[10:20, 10:20] = 5
    path = _write_tif(tmp_path / "small_green.tif", arr)
    mask, note = build_sister_roi_mask(path, target_shape=(200, 200))
    assert mask is None
    assert "≠ stack" in note or "skipped" in note


def test_missing_file_returns_reason(tmp_path):
    mask, note = build_sister_roi_mask(str(tmp_path / "nope_green.tif"),
                                       target_shape=(50, 50))
    assert mask is None and "could not load" in note
