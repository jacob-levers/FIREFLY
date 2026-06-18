"""Headless regression tests for the Visualise-tab Track explorer (W5).

The explorer filters the loaded trajectories by D / α / motion / length into a
sortable table, lets a row-click centre the viewer + populate the inspector, and
exports the filtered subset to CSV.  These tests build a real ``MainWindow`` off
screen (skipped in the Qt-less CI image) and exercise the data path end-to-end.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402
import pytest                                         # noqa: E402

pytest.importorskip("PySide6")
from unittest import mock                             # noqa: E402
from PySide6 import QtWidgets                          # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _make_window(n_tracks=40, seed=0):
    """Build a MainWindow with synthetic loaded-run data wired in."""
    from firefly.ui.app_qt import MainWindow
    w = MainWindow()
    rng = np.random.default_rng(seed)
    parts = []
    for pid in range(n_tracks):
        n = int(rng.integers(5, 30))
        parts.append(pd.DataFrame({
            "particle": pid, "frame": np.arange(n),
            "x": rng.uniform(0, 100, n), "y": rng.uniform(0, 100, n),
            "mass": rng.uniform(100, 500, n)}))
    w._ws_tracks_df = pd.concat(parts, ignore_index=True)
    w._ws_diff_df = pd.DataFrame({
        "particle": range(n_tracks),
        "D": rng.uniform(0.01, 5.0, n_tracks),
        "alpha": rng.uniform(0.2, 1.8, n_tracks),
        "motion": rng.choice(
            ["Immobile", "Confined", "Brownian", "Directed"], n_tracks)})
    w._ws_build_explorer_data()
    return w


def test_explorer_populates_one_row_per_track():
    w = _make_window(n_tracks=40)
    assert w._ws_explorer_df is not None
    assert len(w._ws_explorer_df) == 40
    assert w._ws_exp_table.rowCount() == 40
    assert w.btn_ws_export_tracks.isEnabled()
    # length column is the per-particle vertex count
    lens = w._ws_tracks_df.groupby("particle").size()
    assert (w._ws_explorer_df.set_index("particle")["length"] == lens).all()


def test_explorer_d_range_filter_matches_pandas():
    w = _make_window(n_tracks=40)
    w._ws_exp_d_min.setValue(2.0)
    w._ws_exp_d_max.setValue(4.0)
    w._ws_refresh_explorer()
    expected = int(w._ws_explorer_df["D"].between(2.0, 4.0).sum())
    assert w._ws_exp_table.rowCount() == expected
    assert len(w._ws_exp_filtered) == expected
    assert expected in (len(w._ws_exp_filtered),)  # sanity
    assert f"{expected:,} of 40" in w._ws_exp_count.text()


def test_explorer_motion_checkboxes_filter_classes():
    w = _make_window(n_tracks=40)
    for m, cb in w._ws_exp_motion.items():
        cb.setChecked(m == "Directed")
    w._ws_refresh_explorer()
    assert set(w._ws_exp_filtered["motion"].unique()) <= {"Directed"}
    expected = int((w._ws_explorer_df["motion"] == "Directed").sum())
    assert w._ws_exp_table.rowCount() == expected


def test_explorer_nan_diffusion_is_not_dropped_by_range():
    """Tracks with no D/α fit pass numeric-range filters (NaN-tolerant)."""
    w = _make_window(n_tracks=10)
    w._ws_diff_df.loc[0, "D"] = np.nan
    w._ws_diff_df.loc[0, "alpha"] = np.nan
    w._ws_build_explorer_data()
    w._ws_exp_d_min.setValue(1.0)
    w._ws_refresh_explorer()
    # particle 0 (NaN D) must still be present despite the D>=1 floor
    assert 0 in set(w._ws_exp_filtered["particle"])


def test_explorer_min_length_filter():
    w = _make_window(n_tracks=40)
    lens = w._ws_tracks_df.groupby("particle").size()
    thresh = int(lens.median())
    w._ws_exp_min_len.setValue(thresh)
    w._ws_refresh_explorer()
    assert (w._ws_exp_filtered["length"] >= thresh).all()
    assert w._ws_exp_table.rowCount() == int((lens >= thresh).sum())


def test_explorer_export_matches_filtered_set(tmp_path):
    w = _make_window(n_tracks=40)
    for m, cb in w._ws_exp_motion.items():
        cb.setChecked(m in ("Brownian", "Directed"))
    w._ws_refresh_explorer()
    out = tmp_path / "filtered.csv"
    with mock.patch.object(QtWidgets.QFileDialog, "getSaveFileName",
                           return_value=(str(out), "CSV (*.csv)")):
        w._ws_export_filtered_tracks()
    saved = pd.read_csv(out)
    assert len(saved) == len(w._ws_exp_filtered)
    assert set(saved["particle"]) == set(w._ws_exp_filtered["particle"])
    assert list(saved.columns) == ["particle", "length", "D", "alpha", "motion"]


def test_explorer_row_click_no_viewer_does_not_crash():
    w = _make_window(n_tracks=12)
    w._ws_exp_table.selectRow(0)
    w._ws_explorer_row_clicked()   # no napari viewer attached → must no-op safely


def test_explorer_empty_data_is_graceful():
    from firefly.ui.app_qt import MainWindow
    w = MainWindow()
    w._ws_tracks_df = None
    w._ws_build_explorer_data()
    assert w._ws_explorer_df is None
    assert not w.btn_ws_export_tracks.isEnabled()
    assert w._ws_exp_table.rowCount() == 0
