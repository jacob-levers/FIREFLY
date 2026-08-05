"""Tests for the merged Analysis-workspace data + stats backend.

Synthesises FIREFLY-shaped analysis-output run folders on disk (a
``firefly_extras/`` dir with ``*_summary_metrics.json`` + ``*_diffusion_summary.csv``
+ a couple of aux CSVs) and exercises loading, metric extraction and the
cross-condition statistics.  The folder builder is reused by the offscreen UI
verification, so keep it realistic.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from firefly.ui.controllers.workspace import workspace_data as wd


# ── synthetic run-folder builder (also used for UI demo data) ──────────────
def make_run_folder(root, stem, *, seed=0, n_tracks=600, d_centre=0.12,
                    alpha_centre=0.85, mob=0.6, qc="ok"):
    """Write a minimal-but-realistic analysis-output run folder under *root*.

    Returns the run-folder path (the parent of ``firefly_extras/``).
    """
    rng = np.random.default_rng(seed)
    run_dir = os.path.join(root, stem)
    extras = os.path.join(run_dir, "firefly_extras")
    os.makedirs(extras, exist_ok=True)

    # per-track distribution table
    D = np.abs(rng.lognormal(mean=np.log(max(d_centre, 1e-4)), sigma=0.6, size=n_tracks))
    alpha = np.clip(rng.normal(alpha_centre, 0.2, n_tracks), 0.0, 2.0)
    rg = np.abs(rng.normal(0.18, 0.05, n_tracks))
    motion = np.where(alpha < 0.5, "Immobile",
             np.where(alpha < 0.9, "Confined",
             np.where(alpha < 1.1, "Brownian", "Directed")))
    diff = pd.DataFrame({"particle": np.arange(n_tracks), "D": D, "alpha": alpha,
                         "motion": motion, "MSD0": rng.random(n_tracks) * 1e-3,
                         "loc_sigma_nm": rng.normal(25, 4, n_tracks),
                         "radius_of_gyration_um": rg,
                         "mean_step_um": np.abs(rng.normal(0.15, 0.05, n_tracks)),
                         "path_length_um": np.abs(rng.normal(2.0, 0.5, n_tracks)),
                         "net_displacement_um": np.abs(rng.normal(0.5, 0.2, n_tracks)),
                         "directionality_ratio": rng.uniform(0.1, 0.9, n_tracks)})
    diff.to_csv(os.path.join(extras, f"{stem}_diffusion_summary.csv"), index=False)

    # ensemble MSD curve (lag 1..20 frames)
    lags = np.arange(1, 21)
    pd.DataFrame({"lag_frame": lags,
                  "msd_um2": 4 * d_centre * (lags * 0.02) ** alpha_centre}
                 ).to_csv(os.path.join(extras, f"{stem}_ensemble_msd.csv"), index=False)

    # turning angles — raw per-step angles in degrees (matches the analysis
    # writer, fa_palmtracer: column ``turning_angle_deg``)
    pd.DataFrame({"turning_angle_deg": rng.uniform(0.0, 180.0, n_tracks * 4)}
                 ).to_csv(os.path.join(extras, f"{stem}_turning_angles.csv"), index=False)

    counts = {c: int((motion == c).sum()) for c in ("Immobile", "Confined", "Brownian", "Directed")}
    flags = [{"level": "warn", "msg": "low link ratio"}] if qc == "warn" else []
    summary = {
        "n_tracks": n_tracks, "n_locs": n_tracks * 8,
        "median_d": float(np.median(D)), "median_alpha": float(np.median(alpha)),
        "median_loc_sigma_nm": 25.0, "nongauss_alpha2": float(0.3 + rng.random() * 0.1),
        "vacf_persistence": 0.4, "motion_counts": counts, "mobile_fraction": mob,
        "n_clusters": 0, "dwell_tau_s": float(0.1 + rng.random() * 0.2),
        "frames": 2000, "px_um": 0.16, "fi_s": 0.02, "stem": stem,
        "qc": {"median_track_length": 8.0, "link_ratio": 0.4, "flags": flags},
    }
    with open(os.path.join(extras, f"{stem}_summary_metrics.json"), "w") as fh:
        json.dump(summary, fh)
    return run_dir


def _condition(root, name, n_folders, *, seed, d_centre, n_tracks=600):
    runs = []
    for k in range(n_folders):
        rd = make_run_folder(root, f"{name}_run{k:02d}", seed=seed * 100 + k,
                             d_centre=d_centre, n_tracks=n_tracks)
        runs.append(wd.load_run(rd))
    return runs


# ── tests ──────────────────────────────────────────────────────────────────
def test_load_run_reads_summary_and_qc(tmp_path):
    rd = make_run_folder(str(tmp_path), "cellA", seed=1, n_tracks=1200, qc="warn")
    run = wd.load_run(rd)
    assert run is not None
    assert run.n_tracks == 1200
    assert run.n_label == "1.2k"
    assert run.qc_level == "warn"
    assert wd.is_run_folder(rd)
    assert not wd.is_run_folder(str(tmp_path))   # parent isn't a run folder


def test_metric_scalars_and_distributions(tmp_path):
    run = wd.load_run(make_run_folder(str(tmp_path), "cellB", seed=2, n_tracks=800))
    by_id = wd.METRIC_BY_ID
    # scalars present for the well-supported metrics (incl. the panels that
    # previously showed blank/absent headline cards: angle, a2, radial, vacf)
    for mid in ("D", "a", "mob", "motion", "len", "msd", "step", "angle", "a2",
                "conf", "dwell", "radial", "vacf"):
        assert by_id[mid].scalar(run) is not None, mid
    # full per-track distributions where they should exist
    assert by_id["D"].dist(run).size == (run._diff_col("D", positive=True)).size
    assert by_id["conf"].dist(run) is not None
    assert by_id["mob"].dist(run) is None          # scalar-only metric
    # mobile fraction surfaced as a percentage
    assert 0 <= by_id["mob"].scalar(run) <= 100
    assert by_id["step"].approx is False           # measured → not badged


def test_cliffs_delta_bounds():
    a = np.array([1, 2, 3, 4, 5.0])
    b = np.array([10, 11, 12.0])
    assert wd.cliffs_delta(b, a) == pytest.approx(1.0)
    assert wd.cliffs_delta(a, b) == pytest.approx(-1.0)
    assert abs(wd.cliffs_delta(a, a)) < 1e-9


def test_pairwise_stats_separates_clear_groups(tmp_path):
    lo = _condition(str(tmp_path), "lo", 5, seed=1, d_centre=0.03)
    hi = _condition(str(tmp_path), "hi", 5, seed=2, d_centre=0.40)
    m = wd.METRIC_BY_ID["D"]
    groups = [
        {"label": "lo", "color": "#58a6ff", "phase": "—",
         "values": np.array([m.scalar(r) for r in lo])},
        {"label": "hi", "color": "#f78166", "phase": "—",
         "values": np.array([m.scalar(r) for r in hi])},
    ]
    rows = wd.pairwise_stats(groups, m, test="Mann–Whitney U",
                             correction="None", alpha=0.05)
    assert len(rows) == 1
    r = rows[0]
    assert np.isfinite(r["p"])
    assert abs(r["delta"]) == pytest.approx(1.0)        # fully separated
    assert r["magnitude"] == "large"


def test_correction_inflates_pvalues(tmp_path):
    # three conditions → three pairs; FDR/Bonferroni must not shrink p
    conds = [_condition(str(tmp_path), f"g{i}", 4, seed=i + 1, d_centre=0.05 + 0.05 * i)
             for i in range(3)]
    m = wd.METRIC_BY_ID["D"]
    groups = [{"label": f"g{i}", "color": "#fff", "phase": "—",
               "values": np.array([m.scalar(r) for r in c])}
              for i, c in enumerate(conds)]
    raw = wd.pairwise_stats(groups, m, test="Mann–Whitney U", correction="None", alpha=0.05)
    bon = wd.pairwise_stats(groups, m, test="Mann–Whitney U", correction="Bonferroni", alpha=0.05)
    assert len(raw) == 3 and len(bon) == 3
    for r0, r1 in zip(raw, bon):
        if np.isfinite(r0["p"]) and np.isfinite(r1["p"]):
            assert r1["p"] >= r0["p"] - 1e-9


def test_recommend_config_paired_vs_unpaired():
    assert wd.recommend_config(2, paired=False, multi_folder=False)["cfg"]["test"] == "Mann–Whitney U"
    assert wd.recommend_config(4, paired=False, multi_folder=True)["cfg"]["test"] == "Kruskal–Wallis"
    assert wd.recommend_config(2, paired=True, multi_folder=False)["cfg"]["test"] == "Wilcoxon signed-rank"


def test_new_metrics_read_their_data(tmp_path):
    # Rg / net-displacement / step / speed metrics read the per-track columns the
    # analysis writes (radius_of_gyration_um, mean_step_um, net_displacement_um…).
    import numpy as np
    from firefly.ui.controllers.workspace import workspace_data as wd
    run = wd.load_run(make_run_folder(str(tmp_path), "m", seed=1, d_centre=0.1))
    by_id = wd.METRIC_BY_ID
    # radius of gyration → per-track distribution present
    rg = by_id["rg"].dist(run)
    assert rg is not None and len(rg) > 0 and np.all(rg >= 0)
    # step distance → MEASURED (median of per-track mean step), not D-derived
    step = by_id["step"].scalar(run)
    assert step == pytest.approx(float(np.median(run._diff_col("mean_step_um"))))
    # step speed → measured step ÷ Δt
    sp = by_id["speed"].scalar(run)
    assert sp == pytest.approx(step / run.fi_s) and sp > 0
    # net displacement → the first→last column is present in the fixture now
    nd = by_id["netdisp"].dist(run)
    assert nd is not None and len(nd) > 0


def test_linear_metrics_keep_zero_and_duration_uses_frame_span(tmp_path):
    extras = tmp_path / "z" / "firefly_extras"
    extras.mkdir(parents=True)
    (extras / "z_summary_metrics.json").write_text(json.dumps({
        "n_tracks": 2, "fi_s": 0.5, "metrics_schema_version": 2,
        "gap_policy": "all_pairs",
    }))
    pd.DataFrame({
        "particle": [0, 1],
        "D": [0.1, 0.2],
        "mean_step_um": [0.0, 2.0],
        "radius_of_gyration_um": [0.0, 1.0],
        "net_displacement_um": [0.0, 4.0],
        "path_length_um": [0.0, 5.0],
        "directionality_ratio": [np.nan, 0.0],
    }).to_csv(extras / "z_diffusion_summary.csv", index=False)
    pd.DataFrame({
        "particle": [0, 0, 0, 1, 1],
        "frame": [0, 2, 5, 4, 4],
        "x": [0, 0, 0, 1, 1],
        "y": [0, 0, 0, 1, 1],
    }).to_csv(extras / "z_trajectories.csv", index=False)
    run = wd.load_run(str(tmp_path / "z"))

    assert np.array_equal(wd.METRIC_BY_ID["step"].dist(run), [0.0, 2.0])
    assert wd.METRIC_BY_ID["step"].scalar(run) == pytest.approx(1.0)
    assert np.array_equal(wd.METRIC_BY_ID["netdisp"].dist(run), [0.0, 4.0])
    # Track 0 spans 5 frames (=2.5 s); the one-observation track spans 0 s.
    assert np.array_equal(np.sort(wd.METRIC_BY_ID["dur"].dist(run)),
                          [0.0, 2.5])


def test_metric_contract_detects_mixed_but_allows_stable_metrics(tmp_path):
    legacy = wd.load_run(make_run_folder(str(tmp_path), "legacy", seed=1))
    modern_path = make_run_folder(str(tmp_path), "modern", seed=2)
    sm = tmp_path / "modern" / "firefly_extras" / "modern_summary_metrics.json"
    data = json.loads(sm.read_text())
    data.update({"metrics_schema_version": 2, "gap_policy": "all_pairs"})
    sm.write_text(json.dumps(data))
    modern = wd.load_run(modern_path)

    assert wd.metric_contract_issue([legacy, modern], "D")
    assert wd.metric_contract_issue([legacy, modern], "step")
    assert wd.metric_contract_issue([legacy, modern], "rg") == ""
    assert wd.metric_contract_label([legacy], "step") == "legacy definition"


@pytest.mark.parametrize("summary_state", ["missing", "corrupt"])
def test_metric_contract_recovers_from_params_when_summary_unavailable(
        tmp_path, summary_state):
    """A non-fatal summary export failure must not relabel schema-2 science as
    legacy and thereby make incompatible pooling appear safe."""
    modern_path = make_run_folder(
        str(tmp_path), f"modern_{summary_state}", seed=30, n_tracks=12)
    stem = os.path.basename(modern_path)
    extras = tmp_path / stem / "firefly_extras"
    summary_path = extras / f"{stem}_summary_metrics.json"
    params = {
        "metrics_schema_version": 2,
        "gap_policy": "all_pairs",
        "metric_contract": "firefly_metrics_schema_2",
        "step_definition": "single_frame",
        "link_definition": "adjacent_observed_localisations",
        "duration_definition": "elapsed_frame_span",
        "observed_time_definition": "localisation_count_times_dt",
        "metric_contract_note": "recovered from params",
        "effective_calibration": {
            "pixel_size_um": 0.16, "frame_interval_s": 0.02,
        },
        "embedded_calibration": {"advisory_only": True},
        "pixel_size_um": 0.16,
        "frame_interval_s": 0.02,
    }
    (extras / f"{stem}_params.json").write_text(json.dumps(params))
    if summary_state == "missing":
        summary_path.unlink()
    else:
        summary_path.write_text("{not valid json")

    modern = wd.load_run(modern_path)
    legacy = wd.load_run(make_run_folder(
        str(tmp_path), f"legacy_{summary_state}", seed=31, n_tracks=12))

    assert modern.metrics_schema_version == 2
    assert modern.gap_policy == "all_pairs"
    assert modern.metric_contract == "firefly_metrics_schema_2"
    assert modern.step_definition == "single_frame"
    assert modern.link_definition == "adjacent_observed_localisations"
    assert modern.duration_definition == "elapsed_frame_span"
    assert modern.observed_time_definition == "localisation_count_times_dt"
    assert modern.metric_contract_note == "recovered from params"
    assert modern.effective_calibration["pixel_size_um"] == pytest.approx(0.16)
    assert modern.embedded_calibration["advisory_only"] is True
    assert modern.fi_s == pytest.approx(0.02)
    assert modern.n_tracks == 12       # still falls back to the per-track table
    assert legacy.metrics_schema_version == 1
    assert wd.metric_contract_issue([legacy, modern], "D")
    assert wd.metric_contract_issue([legacy, modern], "step")
    assert wd.metric_contract_issue([legacy, modern], "rg") == ""


def test_metric_contract_detects_same_schema_different_step_definitions(tmp_path):
    runs = []
    definitions = (
        "mean displacement between adjacent observations exactly one frame apart",
        "mean displacement between every adjacent observed localisation",
    )
    for index, definition in enumerate(definitions):
        stem = f"modern_{index}"
        path = make_run_folder(str(tmp_path), stem, seed=index + 10)
        summary_path = (tmp_path / stem / "firefly_extras"
                        / f"{stem}_summary_metrics.json")
        data = json.loads(summary_path.read_text())
        data.update({
            "metrics_schema_version": 2,
            "gap_policy": "all_pairs",
            "metric_contract": "firefly_metrics_schema_2",
            "step_definition": definition,
        })
        summary_path.write_text(json.dumps(data))
        runs.append(wd.load_run(path))

    for metric_id in ("step", "speed", "linkstep", "linkspeed"):
        assert wd.metric_contract_issue(runs, metric_id)
    assert wd.metric_contract_issue(runs, "D") == ""
    assert wd.metric_contract_issue(runs, "rg") == ""
