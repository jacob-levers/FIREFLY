from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import validate_minmass_quality as vmq


def _manifest(movie: Path) -> dict:
    return {
        "resolved_minmass": 0.42,
        "input": {"path": str(movie)},
        "effective_calibration": {
            "pixel_size_um": 0.1,
            "frame_interval_s": 0.02,
        },
        "parameters": {
            "file": str(movie),
            "channel": 0,
            "diameter": 7,
            "backend": "auto",
            "bg_radius": 10,
            "bg_method": "uniform_filter",
            "search_range": 3,
            "memory": 5,
            "min_track_len": 8,
            "minmass_mode": "density",
            "roi_polygon": [[
                [0.0, 0.0], [0.0, 7.0], [7.0, 7.0], [7.0, 0.0],
            ]],
        },
    }


def test_core_panel_is_the_bounded_four_case_honours_design():
    cases = vmq.core_panel(Path("/source"))
    assert [case.label for case in cases] == [
        "control_03aug_fly1_l_low",
        "control_03aug_fly3_r_high",
        "control_04aug_fly1_l_excluded_stress",
        "ama_05aug_fly1_r",
    ]
    assert all(str(case.movie).startswith("/source/") for case in cases)
    assert "04Aug/Excluded/Fly-1-16k Frames-LSide.czi" in str(cases[2].movie)
    assert all(case.manifest.name.endswith("_run_manifest.json")
               for case in cases)


def test_new_drosophila_validation_defaults_to_explicit_torch_backend():
    args = vmq.build_parser().parse_args([
        "--output-dir", "/private/tmp/quality-validation-test"])
    assert args.backend == "torch"


def test_source_windows_match_production_16k_sampling_contract():
    assert vmq._source_windows(16000) == [
        (0, 120),
        (5293, 5413),
        (10586, 10706),
        (15880, 16000),
    ]
    assert sum(stop - start for start, stop in vmq._source_windows(16000)) == 480


def test_full_frame_is_allowed_only_when_saved_or_explicitly_requested():
    saved_none = {"parameters": {"roi_mode": "none", "roi_polygon": None}}
    mask, scope = vmq._roi_mask(
        saved_none, (8, 8), allow_full_frame=False)
    assert mask is None
    assert scope == "full_frame:saved_roi_none"

    ambiguous = {"parameters": {"roi_polygon": None}}
    with pytest.raises(ValueError, match="no polygon ROI"):
        vmq._roi_mask(ambiguous, (8, 8), allow_full_frame=False)
    mask, scope = vmq._roi_mask(
        ambiguous, (8, 8), allow_full_frame=True)
    assert mask is None
    assert scope == "full_frame:explicit_override"


def test_output_validation_rejects_source_tree_and_symlink_escape(
        tmp_path, monkeypatch):
    monkeypatch.setattr(vmq, "PRIVATE_TMP_ROOT", tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    safe = vmq.validate_output_dir(
        tmp_path / "results", source_roots=[source])
    assert safe == (tmp_path / "results").resolve()

    with pytest.raises(ValueError, match="inside source root"):
        vmq.validate_output_dir(source / "results", source_roots=[source])

    target = source / "redirected"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="inside source root"):
        vmq.validate_output_dir(alias / "results", source_roots=[source])


def test_plan_only_decodes_no_pixels_and_writes_only_the_plan(
        tmp_path, monkeypatch):
    monkeypatch.setattr(vmq, "PRIVATE_TMP_ROOT", tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    movie = source / "sample.czi"
    movie.write_bytes(b"test fixture, not a real CZI")
    manifest_path = source / "sample_run_manifest.json"
    manifest_path.write_text(json.dumps(_manifest(movie)), encoding="utf-8")
    output = tmp_path / "output"

    monkeypatch.setattr(vmq, "_inspect_czi", lambda _path: {
        "dims": "TCYX", "size": [16000, 1, 8, 8], "n_frames": 16000,
        "frame_shape": [8, 8], "channels": 1, "pixel_type": "gray16",
        "file_bytes": movie.stat().st_size,
    })
    monkeypatch.setattr(
        vmq, "_read_sampled_stack",
        lambda *args, **kwargs: pytest.fail("plan-only decoded image pixels"))

    rc = vmq.main([
        "--output-dir", str(output),
        "--case", f"fixture::{movie}::{manifest_path}",
        "--plan-only",
    ])
    assert rc == 0
    assert sorted(path.name for path in output.iterdir()) == ["run_plan.json"]
    plan = json.loads((output / "run_plan.json").read_text(encoding="utf-8"))
    assert plan["sample_only"] is True
    assert plan["full_analysis_permitted"] is False
    assert plan["cases"][0]["sample_frames"] == 480
    assert plan["cases"][0]["source_windows"][-1] == [15880, 16000]
    assert plan["cases"][0]["planned_validation_roi_scope"] == "polygon_union"


def test_run_case_calls_production_quality_api_on_only_480_frames(
        tmp_path, monkeypatch):
    movie = tmp_path / "sample.czi"
    movie.write_bytes(b"fixture")
    manifest_path = tmp_path / "sample_run_manifest.json"
    manifest = _manifest(movie)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    case = vmq.Case("fixture", movie, manifest_path)
    plan = {
        "movie_metadata": {"frame_shape": [8, 8], "channels": 1},
        "source_windows": [[0, 120], [5293, 5413], [10586, 10706],
                           [15880, 16000]],
        "pixel_size_um": 0.1,
        "saved_minmass": 0.42,
    }
    captured = {}

    def fake_read(path, *, windows, channel, frame_shape):
        captured["read"] = (path, windows, channel, frame_shape)
        return np.zeros((480, 8, 8), dtype=np.float32)

    def fake_estimate(stack, **kwargs):
        captured["stack_shape"] = stack.shape
        captured["kwargs"] = kwargs
        return 0.31, {
            "quality_status": "valid",
            "quality_reason": None,
            "method": "quality_first",
            "quality_floor_assay": 0.16,
            "quality_floor_effective": 0.20,
            "quality_max_null_track_fraction": 0.10,
            "quality_info": {
                "null_good_fraction_upper": 0.04,
                "observed_good_tracks": 12,
            },
            "harvest_backend": "torch",
            "n_candidates_raw_full_frame": 1000,
            "n_candidates_raw_roi": 500,
            "n_candidates": 500,
            "sample_n_selected": 250,
            "sample_mean_per_frame": 250 / 480,
            "sample_density_per_um2_frame": 0.8,
            "sample_window_cv": 0.1,
            "quality_roi_fraction": 1.0,
        }

    monkeypatch.setattr(vmq, "_read_sampled_stack", fake_read)
    monkeypatch.setattr(vmq, "_estimate_quality", fake_estimate)
    monkeypatch.setattr(
        vmq, "_roi_mask",
        lambda *args, **kwargs: (np.ones((8, 8), dtype=bool), "polygon_union:1"))

    summary, diagnostics = vmq._run_case(
        case, plan, manifest,
        quality_floor=0.16,
        max_null_track_fraction=0.10,
        null_replicates=3,
        null_seed=20260805,
        workers=2,
        backend_override=None,
        allow_full_frame=False,
    )
    assert captured["stack_shape"] == (480, 8, 8)
    assert captured["read"][1] == [
        (0, 120), (5293, 5413), (10586, 10706), (15880, 16000)]
    assert captured["kwargs"]["mode"] == "quality_first"
    assert captured["kwargs"]["quality_floor"] == 0.16
    assert captured["kwargs"]["max_false_track_rate"] == 0.10
    assert captured["kwargs"]["quality_null_replicates"] == 3
    assert "target_density" not in captured["kwargs"]
    assert summary["status"] == "valid"
    assert summary["selected_minmass"] == 0.31
    assert diagnostics["validation_sample_only"] is True
    assert diagnostics["validation_source_windows"][-1] == [15880, 16000]
