import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from firefly.analysis import fa_compare
from test_workspace_data import make_run_folder


def _set_contract(folder, *, schema, gap_policy, metric_contract=None):
    extras = os.path.join(folder, "firefly_extras")
    candidates = [
        os.path.join(extras, name)
        for name in os.listdir(extras) if name.endswith("_params.json")
    ]
    params_path = (candidates[0] if candidates else
                   os.path.join(extras, f"{os.path.basename(folder)}_params.json"))
    if os.path.exists(params_path):
        with open(params_path, encoding="utf-8") as handle:
            params = json.load(handle)
    else:
        params = {"pixel_size_um": 0.16, "frame_interval_s": 0.02}
    params.update({
        "metrics_schema_version": schema,
        "gap_policy": gap_policy,
        "step_definition": (
            "adjacent_observation" if schema < 2 else "single_frame"),
    })
    if metric_contract is not None:
        params["metric_contract"] = metric_contract
    with open(params_path, "w", encoding="utf-8") as handle:
        json.dump(params, handle)


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
