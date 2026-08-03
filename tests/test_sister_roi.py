"""Sister-TIFF ROI helpers (fa_roi.find_sister_roi_path / build_sister_roi_mask).

These back BOTH the analysis run (firefly_worker) and the ROI-viewer preview, so
the preview is exactly the region the run keeps. Cover path detection, the
value-based mask-vs-grayscale decision, the grayscale cell segmentation
(threshold → fill holes → keep largest) and resolution handling.
"""
import os
import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from firefly.analysis.fa_roi import (find_sister_roi_path, build_sister_roi_mask,
                                     load_sister_image)


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


# ── the shared loader (ROI viewer's "Green image" view + the mask builder) ────
# The viewer DISPLAYS what the mask builder thresholds, so both go through
# load_sister_image and normalise identically.
def test_loader_max_projects_a_stack(tmp_path):
    stack = np.zeros((3, 8, 8), np.uint16)
    stack[1, 4, 4] = 900                          # only in the middle frame
    path = _write_tif(tmp_path / "stk_green.tif", stack)
    arr, note = load_sister_image(path)
    assert arr.shape == (8, 8) and arr[4, 4] == 900 and note == ""


def test_loader_puts_the_image_on_the_stack_grid(tmp_path):
    # a half-resolution companion is resampled up, so overlays + drawn polygons
    # (which live in RECORDING pixel coordinates) land in the right place
    arr_in = np.zeros((100, 100), np.uint16); arr_in[20:80, 20:80] = 5
    path = _write_tif(tmp_path / "half_green.tif", arr_in)
    arr, note = load_sister_image(path, (200, 200))
    assert arr.shape == (200, 200) and "resized" in note


def test_loader_reports_why_it_cannot_show_an_image(tmp_path):
    arr_in = np.zeros((100, 50), np.uint16)
    path = _write_tif(tmp_path / "wrongfield_green.tif", arr_in)
    arr, note = load_sister_image(path, (200, 200))
    assert arr is None and "different field" in note
    arr, note = load_sister_image(str(tmp_path / "nope_green.tif"))
    assert arr is None and "could not load" in note


def test_loader_leaves_a_matching_image_untouched(tmp_path):
    arr_in = np.arange(64, dtype=np.uint16).reshape(8, 8)
    path = _write_tif(tmp_path / "same_green.tif", arr_in)
    arr, note = load_sister_image(path, (8, 8))
    assert note == "" and np.array_equal(arr, arr_in)


# ── Zeiss companions: '<stem>-Green Image.czi' beside a .czi recording ────────
# A microscope workflow exports the companion straight from the scope, so it is
# a .czi with a hand-typed suffix — not the '<stem>_green.tif' the loader
# originally assumed.  Neither was found before, so the ROI green-image view and
# Sister-TIFF mode silently did nothing on that data.
def _touch(p):
    open(p, "w").close()
    return str(p)


def test_czi_companion_is_found(tmp_path):
    rec = _touch(tmp_path / "Fly-1-16k Frames-LSide.czi")
    green = _touch(tmp_path / "Fly-1-16k Frames-LSide-Green Image.czi")
    assert find_sister_roi_path(rec, "-Green Image") == green


def test_companion_suffix_match_is_case_insensitive(tmp_path):
    rec = _touch(tmp_path / "Fly-2-16k Frames-RSide.czi")
    _touch(tmp_path / "Fly-2-16k Frames-RSide-Green Image.czi")
    got = find_sister_roi_path(rec, "-green image")
    assert got is not None and os.path.isfile(got)


def test_tif_companion_still_found(tmp_path):
    """The original .tif contract must keep working."""
    rec = _touch(tmp_path / "cell.tif")
    green = _touch(tmp_path / "cell_green.tif")
    assert find_sister_roi_path(rec, "_green") == green


def test_absent_companion_still_returns_none(tmp_path):
    rec = _touch(tmp_path / "Fly-3-16k Frames-LSide.czi")
    assert find_sister_roi_path(rec, "-Green Image") is None
