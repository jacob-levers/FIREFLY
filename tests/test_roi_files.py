"""Tests for ImageJ ROI parsing + sibling-ROI auto-detection (fa_roi).

These cover the relocation of the ROI parsers out of the Qt module and the
batch auto-pairing helpers — no Qt / GUI required.
"""
import os
import zipfile

import numpy as np
import pytest

from firefly.analysis import fa_roi


def _write_imagej_rect_roi(path, x=10, y=20, w=8, h=6):
    """Write a minimal ImageJ polygon .roi (magic 'Iout') from 4 corner points."""
    roifile = pytest.importorskip("roifile")
    roi = roifile.ImagejRoi.frompoints(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    roi.tofile(path)


def test_load_imagej_roi_zip(tmp_path):
    pytest.importorskip("roifile")
    r1 = tmp_path / "0011-0038.roi"
    r2 = tmp_path / "0028-0047.roi"
    _write_imagej_rect_roi(str(r1), x=5, y=5, w=4, h=4)
    _write_imagej_rect_roi(str(r2), x=40, y=40, w=6, h=6)
    zpath = tmp_path / "RoiSet.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(r1, arcname=r1.name)
        zf.write(r2, arcname=r2.name)
    polys = fa_roi._load_any_roi_file(str(zpath))
    assert len(polys) == 2
    # vertices are (y, x) pairs
    assert all(p.ndim == 2 and p.shape[1] == 2 for p in polys)


def test_find_sibling_prefers_roiset_zip(tmp_path):
    # RoiSet.zip wins over a same-stem .roi.
    (tmp_path / "RoiSet.zip").write_bytes(b"PK\x03\x04dummy")
    (tmp_path / "Movie.roi").write_bytes(b"Ioutdummy")
    got = fa_roi.find_sibling_imagej_roi(str(tmp_path), "Movie")
    assert os.path.basename(got) == "RoiSet.zip"


def test_find_sibling_roiset_folder(tmp_path):
    rs = tmp_path / "RoiSet"
    rs.mkdir()
    (rs / "0000-0181.roi").write_bytes(b"Ioutdummy")
    got = fa_roi.find_sibling_imagej_roi(str(tmp_path), "Movie")
    assert got == str(rs)


def test_find_sibling_ignores_appledouble(tmp_path):
    # A macOS AppleDouble stub must NOT be picked as the sole ROI.
    (tmp_path / "._RoiSet.zip").write_bytes(b"\x00\x05\x16\x07stub")
    (tmp_path / "RoiSet.zip").write_bytes(b"PK\x03\x04real")
    got = fa_roi.find_sibling_imagej_roi(str(tmp_path), "Movie")
    assert os.path.basename(got) == "RoiSet.zip"
    # And when ONLY the AppleDouble exists, nothing is returned.
    d2 = tmp_path / "only_dot"
    d2.mkdir()
    (d2 / "._RoiSet.zip").write_bytes(b"stub")
    assert fa_roi.find_sibling_imagej_roi(str(d2), "Movie") is None


def test_find_sibling_none_when_absent(tmp_path):
    (tmp_path / "Movie.czi").write_bytes(b"not really a czi")
    assert fa_roi.find_sibling_imagej_roi(str(tmp_path), "Movie") is None
