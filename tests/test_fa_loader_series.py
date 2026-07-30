"""Focused ordering/provenance tests for the image/localisation loaders."""
from __future__ import annotations

import os

import numpy as np


def test_load_czi_explicit_files_preserve_scanner_order(tmp_path, monkeypatch):
    """An explicit CZI selection is consumed in the caller's exact order.

    Auto-discovery/scanning owns natural ordering; the codec layer must not
    second-guess a reviewed or deliberately customised series.
    """
    from firefly.analysis import fa_loaders

    names = ["movie.czi", "movie(1).czi", "movie(2).czi", "movie(10).czi"]
    paths = [tmp_path / name for name in names]
    for path in paths:
        path.touch()
    seen = []
    marker = {name: i for i, name in enumerate(names)}

    def _fake_single(path, *_args):
        name = os.path.basename(path)
        seen.append(name)
        return np.full((1, 2, 2), marker[name], dtype=np.float32), 0.1, 0.02

    monkeypatch.setattr(fa_loaders, "_load_single_czi", _fake_single)
    explicit = [paths[3], paths[2], paths[1], paths[0]]
    stack, px, fi = fa_loaders.load_czi(
        str(paths[0]), files=[str(path) for path in explicit])

    assert seen == [path.name for path in explicit]
    assert stack[:, 0, 0].tolist() == [3.0, 2.0, 1.0, 0.0]
    assert (px, fi) == (0.1, 0.02)


def test_external_loader_keeps_palmtracer_metadata_advisory(tmp_path):
    """Embedded PALM-Tracer calibration is provenance, not a hidden override."""
    from firefly.analysis.fa_loaders import load_external_locs

    path = tmp_path / "locPALMTracer.txt"
    path.write_text(
        "Width\tHeight\tPixel_Size(um)\tFrame_Duration(s)\n"
        "512\t512\t0.25\t0.05\n"
        "id\tPlane\tCentroidX(px)\tCentroidY(px)\tIntegrated_Intensity\n"
        "1\t1\t10\t20\t100\n")

    locs = load_external_locs(str(path), preset="PALM-Tracer", pixel_size_um=0.106)
    assert locs.attrs["embedded_pixel_size_um"] == 0.25
    assert locs.attrs["embedded_frame_interval_s"] == 0.05
    # Pixel coordinates remain pixel coordinates; the passed sidebar value is
    # not silently changed inside this parser.
    assert locs.iloc[0]["x"] == 10.0


def test_external_text_probe_rejects_empty_file(tmp_path):
    from firefly.analysis.fa_loaders import probe_external_locs_text

    path = tmp_path / "empty.tsv"
    path.touch()
    try:
        probe_external_locs_text(str(path))
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("empty table was accepted")
