"""Native PALM-Tracer output provenance and live metric-contract guards."""
import json
import queue
import threading
from pathlib import Path

import pandas as pd
import pytest


class _FakeCombinedFigure:
    def save(self, path, **_kwargs):
        Path(path).write_bytes(b"native-figure")


def test_native_palmtracer_render_writes_truthful_sidecars(tmp_path, monkeypatch):
    """Native D/MSD must not inherit the normal schema-2 worker contract."""
    from firefly import firefly_worker as fw
    from firefly.analysis import fa_figure, fa_palmtracer

    source = tmp_path / "source"
    source.mkdir()
    input_file = source / "locPALMTracer.txt"
    input_file.write_text("source")
    out_dir = tmp_path / "out"
    fig_dir = out_dir / "figures"
    data_dir = out_dir / "data"
    extras_dir = out_dir / "firefly_extras"
    for path in (fig_dir, data_dir, extras_dir):
        path.mkdir(parents=True, exist_ok=True)

    locs = pd.DataFrame({
        "x": [1.0, 2.0, 3.0], "y": [2.0, 3.0, 4.0],
        "frame": [0, 1, 2], "mass": [10.0, 11.0, 12.0],
    })
    tracks = pd.DataFrame({
        "particle": [1, 1, 2], "frame": [0, 1, 2],
        "x": [1.0, 2.0, 3.0], "y": [2.0, 3.0, 4.0],
    })
    diffusion = pd.DataFrame({
        "particle": [1, 2], "D": [0.10, 0.30],
        "alpha": [0.9, 1.1], "motion": ["Brownian", "Directed"],
        "loc_sigma_nm": [20.0, 30.0],
    })
    native_summary = {
        "params": {"pixel_size_um": 0.25, "frame_interval_s": 0.05},
        "locs": locs,
        "tracks": tracks,
        "diffusion": diffusion,
        "imsd": pd.DataFrame({1: [0.01, 0.02], 2: [0.03, 0.04]}, index=[1, 2]),
        "ensemble_msd": pd.DataFrame({"lag_frame": [1, 2], "msd_um2": [0.02, 0.03]}),
        "n_frames": 3,
        "width": 12,
        "height": 10,
    }
    monkeypatch.setattr(
        fa_palmtracer, "load_summary_from_palmtracer",
        lambda *_args, **_kwargs: native_summary)
    monkeypatch.setattr(
        fa_palmtracer, "save_palmtracer_csvs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        fa_figure, "make_figure",
        lambda *_args, **_kwargs: {"combined": _FakeCombinedFigure()})

    logs = []
    payload = fw._render_palmtracer_native(
        {
            "file": str(input_file),
            "palmtracer_folder": str(source),
            # Deliberately disagree with PALM-Tracer's native calibration.
            "pixel_size": 0.16,
            "frame_interval": 0.02,
        },
        str(out_dir), "native", str(fig_dir), str(data_dir), str(extras_dir),
        logs.append)

    params = json.loads((extras_dir / "native_params.json").read_text())
    summary = json.loads((extras_dir / "native_summary_metrics.json").read_text())
    manifest = json.loads((out_dir / "native_run_manifest.json").read_text())

    for sidecar in (params, summary, manifest):
        assert sidecar["metrics_schema_version"] == 1
        assert sidecar["gap_policy"] == "contiguous"
        assert sidecar["metric_contract"] == "palmtracer_native_legacy"
        assert "not recalculated under FIREFLY metrics schema 2" in sidecar[
            "metric_contract_note"]
        assert sidecar["effective_calibration"]["pixel_size_um"] == 0.25
        assert sidecar["effective_calibration"]["frame_interval_s"] == 0.05
        assert sidecar["embedded_calibration"]["pixel_size_um"] == 0.25
        assert sidecar["embedded_calibration"]["frame_interval_s"] == 0.05
        assert sidecar["fit_status_counts"] == {"native_unavailable": 2}
        assert sidecar["n_below_resolution"] is None

    assert params["pixel_size_um"] == 0.25
    assert params["frame_interval_s"] == 0.05
    assert summary["median_d"] == 0.20
    assert summary["mean_observed_time_s"] == pytest.approx(0.075)
    assert summary["mean_track_duration_s"] == pytest.approx(0.025)
    assert summary["mean_track_length_s"] == summary["mean_observed_time_s"]
    assert manifest["step_definition"] == fw.PALMTRACER_NATIVE_UNVERIFIED_DEFINITION
    assert manifest["parameters"]["metrics_schema_version"] == 1
    assert manifest["parameters"]["pixel_size"] == 0.25
    assert manifest["parameters"]["frame_interval"] == 0.05
    assert payload["summary"]["metric_contract"] == "palmtracer_native_legacy"
    assert any("Saved native contract sidecars" in line for line in logs)


def test_resolved_gap_policy_log_describes_the_effective_estimator():
    from firefly import firefly_worker as fw

    assert fw._resolved_gap_policy_message("all_pairs") == (
        "  Resolved gap policy: all_pairs (all timestamp-matched pairs).")
    assert fw._resolved_gap_policy_message("contiguous") == (
        "  Resolved gap policy: contiguous (contiguous observed runs "
        "(legacy compatibility)).")


def test_normal_worker_emits_the_resolved_gap_policy_before_loading(tmp_path):
    """The audit line is emitted on the actual non-native worker path."""
    from firefly import firefly_worker as fw

    logs = []
    with pytest.raises(Exception):
        fw._run_one_analysis(
            {
                "file": str(tmp_path / "missing.tif"),
                "out_dir": str(tmp_path / "out"),
                "source": "image",
                "gap_policy": "contiguous",
            },
            queue.Queue(), threading.Event(), logs.append,
            lambda *_args: None)
    assert "Resolved gap policy: contiguous" in "\n".join(logs)


def test_live_workspace_blocks_native_motion_and_vacf_mixing(tmp_path):
    """Motion and VACF are temporal inferences, not stable native sidecars."""
    from firefly.ui.controllers.workspace import workspace_data as wd
    from test_workspace_data import make_run_folder

    native_path = make_run_folder(str(tmp_path), "native", seed=1)
    current_path = make_run_folder(str(tmp_path), "current", seed=2)
    for path, schema, gap, contract in (
        (native_path, 1, "contiguous", "palmtracer_native_legacy"),
        (current_path, 2, "all_pairs", None),
    ):
        sidecar = Path(path) / "firefly_extras" / (
            f"{Path(path).name}_summary_metrics.json")
        data = json.loads(sidecar.read_text())
        data.update({"metrics_schema_version": schema, "gap_policy": gap})
        if contract:
            data["metric_contract"] = contract
        sidecar.write_text(json.dumps(data))

    native = wd.load_run(native_path)
    current = wd.load_run(current_path)
    assert wd.metric_contract_issue([native, current], "motion")
    assert wd.metric_contract_issue([native, current], "vacf")
    assert wd.metric_contract_issue([native, current], "rg") == ""
