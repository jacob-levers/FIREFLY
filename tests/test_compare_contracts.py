import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from firefly.analysis import fa_compare
from test_workspace_data import make_run_folder


def _params_path(folder):
    extras = os.path.join(folder, "firefly_extras")
    candidates = [
        os.path.join(extras, name)
        for name in os.listdir(extras) if name.endswith("_params.json")
    ]
    return (candidates[0] if candidates else
            os.path.join(extras, f"{os.path.basename(folder)}_params.json"))


def _update_params(folder, values):
    params_path = _params_path(folder)
    if os.path.exists(params_path):
        with open(params_path, encoding="utf-8") as handle:
            params = json.load(handle)
    else:
        params = {"pixel_size_um": 0.16, "frame_interval_s": 0.02}
    params.update(values)
    with open(params_path, "w", encoding="utf-8") as handle:
        json.dump(params, handle)


def _set_contract(folder, *, schema, gap_policy, metric_contract=None):
    values = {
        "metrics_schema_version": schema,
        "gap_policy": gap_policy,
        "step_definition": (
            "adjacent_observation" if schema < 2 else "single_frame"),
    }
    if metric_contract is not None:
        values["metric_contract"] = metric_contract
    _update_params(folder, values)


def _set_quality_contract(folder, **overrides):
    values = {
        "metrics_schema_version": 2,
        "gap_policy": "all_pairs",
        "metric_contract": "firefly_metrics_schema_2",
        "step_definition": "single_frame",
        "detection_policy": "quality_first",
        "detection_contract": "quality_first_track_ambiguity_v1",
        "detection_backend_resolved": "torch-cg",
        "quality_status": "valid",
        "quality_floor_assay": 0.16,
        # Deliberately file-specific outputs: compatibility must ignore these.
        "quality_floor_effective": 0.24,
        "minmass_used": 0.31,
        "quality_null_definition": "roi_smoothed_spatial_redraw_v1",
        "quality_null_ceiling": 0.10,
        "quality_null_replicates": 3,
        "diameter": 7,
        "bg_method": "uniform_filter",
        "bg_radius": 10,
        "search_range": 5,
        "memory": 3,
        "min_track_len": 8,
        "quality_roi_scope": "full_frame",
    }
    values.update(overrides)
    _update_params(folder, values)


def _groups(a, b):
    return [
        {"label": "A", "color": "#4c78a8", "folders": [a]},
        {"label": "B", "color": "#f58518", "folders": [b]},
    ]


def test_mixed_contracts_suppress_sensitive_but_keep_stable_inference(tmp_path):
    legacy = make_run_folder(
        str(tmp_path), "legacy", seed=1, d_centre=0.08)
    current = make_run_folder(
        str(tmp_path), "current", seed=2, d_centre=0.20)
    _set_contract(legacy, schema=1, gap_policy="contiguous")
    _set_contract(current, schema=2, gap_policy="all_pairs")
    groups = [
        {"label": "Legacy", "color": "#4c78a8", "folders": [legacy]},
        {"label": "Current", "color": "#f58518", "folders": [current]},
    ]

    report = fa_compare.compute_report(groups)
    assert set(report.compatibility_warnings) == {"diffusion", "step"}
    assert set(report.summary_df["metric_contract"]) == {
        "legacy schema 1 (contiguous)",
        "metrics schema 2 (all_pairs)",
    }

    fig, _summary, stats = fa_compare.render_report(
        report, panels={"msd", "step", "path"}, pdf_report=False)
    titles = {axis.get_title() for axis in fig.axes}
    assert any("Incompatible diffusion contract" in title for title in titles)
    assert any("Incompatible step contract" in title for title in titles)
    assert "path_length" in stats
    assert "step_distance" not in stats
    plt.close(fig)


def test_gap_policy_only_blocks_timestamp_lag_metrics(tmp_path):
    all_pairs = make_run_folder(
        str(tmp_path), "allpairs", seed=3, d_centre=0.08)
    contiguous = make_run_folder(
        str(tmp_path), "contiguous", seed=4, d_centre=0.20)
    _set_contract(all_pairs, schema=2, gap_policy="all_pairs")
    _set_contract(contiguous, schema=2, gap_policy="contiguous")
    groups = [
        {"label": "All pairs", "color": "#4c78a8", "folders": [all_pairs]},
        {"label": "Contiguous", "color": "#f58518", "folders": [contiguous]},
    ]

    report = fa_compare.compute_report(groups)
    assert "diffusion" in report.compatibility_warnings
    assert "step" not in report.compatibility_warnings


def test_native_and_firefly_legacy_contracts_do_not_pool(tmp_path):
    native = make_run_folder(str(tmp_path), "native", seed=5, d_centre=0.08)
    legacy = make_run_folder(str(tmp_path), "legacy2", seed=6, d_centre=0.20)
    _set_contract(
        native, schema=1, gap_policy="contiguous",
        metric_contract="palmtracer_native_legacy")
    _set_contract(legacy, schema=1, gap_policy="contiguous")
    report = fa_compare.compute_report([
        {"label": "Native", "color": "#4c78a8", "folders": [native]},
        {"label": "Legacy", "color": "#f58518", "folders": [legacy]},
    ])
    assert set(report.compatibility_warnings) == {"diffusion", "step"}


def test_summary_level_step_definition_is_part_of_the_contract():
    """Runs without a params sidecar still retain persisted schema-2 semantics."""
    contract = fa_compare._summary_metric_contract({
        "metrics_schema_version": 2,
        "gap_policy": "all_pairs",
        "metric_contract": "firefly_metrics_schema_2",
        "step_definition": "custom_frame_aware_definition",
        "params": {},
    })
    assert contract["step_definition"] == "custom_frame_aware_definition"
    assert contract["step"][1] == "custom_frame_aware_definition"


def test_matching_quality_contracts_pool_despite_different_resolved_thresholds(
        tmp_path):
    a = make_run_folder(str(tmp_path), "quality_a", seed=10, n_tracks=16)
    b = make_run_folder(str(tmp_path), "quality_b", seed=11, n_tracks=16)
    _set_quality_contract(a, quality_floor_effective=0.22, minmass_used=0.28)
    _set_quality_contract(b, quality_floor_effective=0.91, minmass_used=0.97)

    report = fa_compare.compute_report(_groups(a, b))

    assert "detection" not in report.compatibility_warnings


@pytest.mark.parametrize("field,changed", [
    ("detection_policy", "linkability"),
    ("detection_contract", "quality_first_track_ambiguity_v2"),
    ("detection_backend_resolved", "trackpy"),
    ("quality_floor_assay", 0.20),
    ("quality_null_definition", "different_null_v1"),
    ("quality_null_ceiling", 0.05),
    ("quality_null_replicates", 9),
    ("diameter", 9),
    ("bg_method", "rolling_ball"),
    ("bg_radius", 20),
    ("search_range", 7),
    ("memory", 1),
    ("min_track_len", 12),
    ("quality_roi_scope", "polygon_union"),
])
def test_quality_contract_mismatch_suppresses_all_detection_panels(
        tmp_path, field, changed):
    a = make_run_folder(str(tmp_path), f"a_{field}", seed=12, n_tracks=16)
    b = make_run_folder(str(tmp_path), f"b_{field}", seed=13, n_tracks=16)
    _set_quality_contract(a)
    _set_quality_contract(b, **{field: changed})

    report = fa_compare.compute_report(_groups(a, b))

    assert "detection" in report.compatibility_warnings
    warning = report.compatibility_warnings["detection"]
    assert "All detection-derived panels and pooled inference were suppressed" in warning
    assert "effective thresholds may differ" in warning

    # Path and fluorescence used to survive the narrower MSD/step guards.  They
    # are detection-derived too, so a mismatch must replace every requested
    # panel with one warning card and produce no inferential statistics.
    fig, _summary, stats = fa_compare.render_report(
        report, panels={"msd", "path", "fluor"}, pdf_report=False)
    titles = {axis.get_title() for axis in fig.axes}
    assert any("Incompatible detection contract" in title for title in titles)
    assert not stats
    plt.close(fig)


def test_quality_first_with_legacy_unknown_warns_but_legacy_only_still_loads(
        tmp_path):
    quality = make_run_folder(str(tmp_path), "quality", seed=14, n_tracks=16)
    legacy = make_run_folder(str(tmp_path), "legacy_unknown", seed=15, n_tracks=16)
    _set_quality_contract(quality)
    _set_contract(legacy, schema=2, gap_policy="all_pairs")

    mixed = fa_compare.compute_report(_groups(quality, legacy))
    assert "detection" in mixed.compatibility_warnings
    assert "legacy/unknown detection provenance" in mixed.compatibility_warnings["detection"]

    legacy_b = make_run_folder(str(tmp_path), "legacy_unknown_b", seed=16,
                               n_tracks=16)
    _set_contract(legacy_b, schema=2, gap_policy="all_pairs")
    legacy_only = fa_compare.compute_report(_groups(legacy, legacy_b))
    assert "detection" not in legacy_only.compatibility_warnings


def test_incomplete_quality_provenance_suppresses_pooling(tmp_path):
    a = make_run_folder(str(tmp_path), "complete", seed=17, n_tracks=16)
    b = make_run_folder(str(tmp_path), "incomplete", seed=18, n_tracks=16)
    _set_quality_contract(a)
    _set_quality_contract(b)
    params_path = _params_path(b)
    with open(params_path, encoding="utf-8") as handle:
        params = json.load(handle)
    params.pop("bg_radius")
    with open(params_path, "w", encoding="utf-8") as handle:
        json.dump(params, handle)

    report = fa_compare.compute_report(_groups(a, b))

    assert "detection" in report.compatibility_warnings
    assert "missing=bg_radius" in report.compatibility_warnings["detection"]


def test_invalid_quality_run_is_skipped_before_pooling(tmp_path):
    invalid = make_run_folder(str(tmp_path), "invalid", seed=19, n_tracks=16)
    valid_a = make_run_folder(str(tmp_path), "valid_a", seed=20, n_tracks=16)
    valid_b = make_run_folder(str(tmp_path), "valid_b", seed=21, n_tracks=16)
    _set_quality_contract(
        invalid, quality_status="invalid",
        quality_reason="full_run_assignment_ambiguity")
    _set_quality_contract(valid_a)
    _set_quality_contract(valid_b)
    groups = _groups(valid_a, valid_b)
    groups[0]["folders"].insert(0, invalid)

    report = fa_compare.compute_report(groups)

    assert [len(group) for group in report.all_summaries] == [1, 1]
    assert len(report.skipped[0]) == 1
    assert report.skipped[0][0][0] == invalid
    assert "quality_status is 'invalid'" in report.skipped[0][0][1]
    assert "full_run_assignment_ambiguity" in report.skipped[0][0][1]
    assert "detection" not in report.compatibility_warnings


def test_group_with_only_unvalidated_quality_runs_is_rejected(tmp_path):
    invalid = make_run_folder(str(tmp_path), "invalid_only", seed=22, n_tracks=16)
    valid = make_run_folder(str(tmp_path), "valid_other", seed=23, n_tracks=16)
    _set_quality_contract(invalid, quality_status="unresolved",
                          quality_reason="low_candidate_count")
    _set_quality_contract(valid)

    with pytest.raises(fa_compare.CompareInputError) as excinfo:
        fa_compare.compute_report(_groups(invalid, valid))

    message = str(excinfo.value)
    assert "no valid analysis folders" in message
    assert "quality_status is 'unresolved'" in message
