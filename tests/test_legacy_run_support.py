"""Support for runs produced by an older FIREFLY.

Three behaviours, which together decide what a user sees when they load data
analysed months ago:

1. Track GEOMETRY is backfilled from the cached trajectories, because those
   quantities need no raw movie — otherwise every graph added after the run was
   made is silently empty.
2. D / alpha / MSD / motion are NOT recomputed: rewriting recorded scientific
   output would be worse than showing what was saved.
3. Because of (2) the user is warned, since the gap-aware MSD fix can move those
   numbers substantially on memory-linked runs.
"""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication                  # noqa: E402

from firefly.ui.controllers.workspace import workspace_data as wd   # noqa: E402
from test_workspace_data import make_run_folder              # noqa: E402

_app = QApplication.instance() or QApplication([])

_NEW_COLUMNS = ("path_length_um", "net_displacement_um", "directionality_ratio",
                "mean_step_um", "track_duration_s")


def _legacy_run(root, stem="oldrun", *, seed=1):
    """A run folder as an older FIREFLY left it: trajectories cached, but none
    of the newer per-track geometry columns."""
    folder = make_run_folder(root, stem, seed=seed)
    extras = os.path.join(folder, "firefly_extras")
    dpath = os.path.join(extras, f"{stem}_diffusion_summary.csv")
    diff = pd.read_csv(dpath)
    diff = diff[[c for c in diff.columns if c not in _NEW_COLUMNS]]
    diff.to_csv(dpath, index=False)
    # …and real trajectories to backfill from.
    rng = np.random.default_rng(seed)
    rows = []
    for pid in diff["particle"].to_numpy()[:40]:
        n = int(rng.integers(6, 20))
        x, y = 30.0, 40.0
        for k in range(n):
            x += rng.normal(0, 0.4); y += rng.normal(0, 0.4)
            rows.append((int(pid), k, x, y))
    pd.DataFrame(rows, columns=["particle", "frame", "x", "y"]).to_csv(
        os.path.join(extras, f"{stem}_trajectories.csv"), index=False)
    return folder, dpath


# ── 1. geometry is backfilled, and matches the canonical pipeline ────────────
def test_geometry_is_backfilled_for_a_legacy_run(tmp_path):
    folder, dpath = _legacy_run(str(tmp_path))
    assert not set(_NEW_COLUMNS) & set(pd.read_csv(dpath, nrows=0).columns)

    run = wd.load_run(folder)
    assert run is not None
    cols = set(pd.read_csv(dpath, nrows=0).columns)
    assert set(_NEW_COLUMNS).issubset(cols), "geometry not backfilled"
    for mid in ("path", "netdisp", "dir", "step", "dur"):
        metric = next(m for m in wd.METRICS if m.id == mid)
        assert metric.scalar(run) is not None, f"{mid} still blank"


def test_backfill_matches_the_canonical_pipeline(tmp_path):
    """The backfill re-implements the formulas for speed; pin it to the source
    of truth so the two cannot drift apart."""
    from firefly.analysis.fa_diffusion import compute_msd_and_fit
    folder, dpath = _legacy_run(str(tmp_path), stem="pin", seed=5)
    wd.load_run(folder)
    got = pd.read_csv(dpath)

    tr = pd.read_csv(os.path.join(folder, "firefly_extras", "pin_trajectories.csv"))
    px, fi = wd._run_calibration(os.path.join(folder, "firefly_extras"), "pin")
    _i, _e, ref = compute_msd_and_fit(tr, px, fi, max_lagtime=10, n_fit=5, workers=1)

    merged = got.merge(ref, on="particle", suffixes=("_bf", "_ref"))
    for col in _NEW_COLUMNS:
        a = pd.to_numeric(merged[col + "_bf"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(merged[col + "_ref"], errors="coerce").to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        assert (np.isfinite(a) == np.isfinite(b)).all(), f"{col}: NaN mismatch"
        assert np.allclose(a[both], b[both], rtol=1e-9, atol=1e-12), col


def test_gapped_step_backfill_obeys_each_persisted_metric_contract(tmp_path):
    """For frames [0, 2, 3], legacy adjacent-observation step is mean(2, 1)
    while schema-2 single-frame step uses only the 2→3 link.  Backfill must
    reconstruct each promised definition and keep their pooling guard active."""
    def _make(stem, schema):
        folder = tmp_path / stem
        extras = folder / "firefly_extras"
        extras.mkdir(parents=True)
        summary = {
            "n_tracks": 1,
            "n_locs": 3,
            "px_um": 1.0,
            "fi_s": 1.0,
            "metrics_schema_version": schema,
            "gap_policy": "contiguous" if schema < 2 else "all_pairs",
        }
        if schema >= 2:
            summary.update({
                "metric_contract": "firefly_metrics_schema_2",
                "step_definition": "single_frame",
            })
        (extras / f"{stem}_summary_metrics.json").write_text(
            json.dumps(summary))
        (extras / f"{stem}_params.json").write_text(json.dumps({
            "pixel_size_um": 1.0,
            "frame_interval_s": 1.0,
            "metrics_schema_version": schema,
            "gap_policy": summary["gap_policy"],
            "metric_contract": summary.get("metric_contract", ""),
            "step_definition": summary.get(
                "step_definition", "adjacent_observation"),
        }))
        pd.DataFrame({
            "particle": [1], "D": [0.1], "alpha": [1.0],
            "motion": ["Brownian"],
        }).to_csv(extras / f"{stem}_diffusion_summary.csv", index=False)
        pd.DataFrame({
            "particle": [1, 1, 1],
            "frame": [0, 2, 3],
            "x": [0.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0],
        }).to_csv(extras / f"{stem}_trajectories.csv", index=False)
        return wd.load_run(str(folder))

    legacy = _make("legacy_gap", 1)
    modern = _make("modern_gap", 2)

    assert legacy.step_definition == "adjacent_observation"
    assert wd.METRIC_BY_ID["step"].scalar(legacy) == pytest.approx(1.5)
    assert modern.step_definition == "single_frame"
    assert wd.METRIC_BY_ID["step"].scalar(modern) == pytest.approx(1.0)
    assert wd.metric_contract_issue([legacy, modern], "step")
    assert wd.metric_contract_issue([legacy, modern], "speed")


def test_backfill_never_touches_recorded_science(tmp_path):
    """D / alpha / motion must survive the backfill byte-for-byte."""
    folder, dpath = _legacy_run(str(tmp_path), stem="keep", seed=7)
    before = pd.read_csv(dpath)
    wd.load_run(folder)
    after = pd.read_csv(dpath)
    for col in ("D", "alpha", "motion"):
        pd.testing.assert_series_equal(before[col], after[col],
                                       check_names=False)


# ── 3. the user is warned about what was NOT recomputed ──────────────────────
def _await(controller, timeout=20.0):
    import time
    deadline = time.monotonic() + timeout
    while controller.loadingFolders and time.monotonic() < deadline:
        _app.processEvents(); time.sleep(0.01)
    _app.processEvents()


def _controller_with(folders):
    from firefly.ui.controllers.workspace.workspace_controller import (
        AnalysisWorkspaceController)
    c = AnalysisWorkspaceController(settings=None)
    c.addFolders(c.conditions[0]["id"], folders)
    _await(c)
    return c


def test_legacy_runs_raise_a_warning_naming_them(tmp_path):
    folder, _ = _legacy_run(str(tmp_path), stem="oldrun")
    w = _controller_with([folder]).legacyDataWarning
    assert w["show"] is True and w["count"] == 1
    assert "oldrun" in w["names"] and w["key"] == "oldrun"
    body = w["text"].lower()
    assert "older version of firefly" in body
    # it must say what IS trustworthy and what is not
    assert "net displacement" in body and "re-analys" in body


def test_current_schema_runs_raise_no_warning(tmp_path):
    folder = make_run_folder(str(tmp_path), "newrun", seed=3)
    p = os.path.join(folder, "firefly_extras", "newrun_summary_metrics.json")
    data = json.load(open(p))
    data["metrics_schema_version"] = 2
    json.dump(data, open(p, "w"))
    assert _controller_with([folder]).legacyDataWarning["show"] is False


def test_warning_key_identifies_the_set_so_it_can_be_shown_once(tmp_path):
    """The UI acknowledges by key; it must be stable for the same set and change
    when a different legacy run appears."""
    a, _ = _legacy_run(str(tmp_path / "a"), stem="runA")
    b, _ = _legacy_run(str(tmp_path / "b"), stem="runB")
    k1 = _controller_with([a]).legacyDataWarning["key"]
    k2 = _controller_with([a]).legacyDataWarning["key"]
    k3 = _controller_with([a, b]).legacyDataWarning["key"]
    assert k1 == k2 and k1 != k3
