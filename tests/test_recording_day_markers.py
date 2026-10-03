"""Replicate dots shaped by recording day.

On the MB112C data day was confounded with condition (the later flies sat
lower in both groups); with every dot a plain circle that is invisible on the
figure.  Each recording's day comes from the CZI header — recorded into the
run's params at process time, or read from the recording for older runs.
"""
import json
import os
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from firefly.analysis.fa_compare import (DAY_MARKERS, UNDATED_MARKER, _recording_day,
                                         compare_groups, day_markers)
from firefly.analysis.fa_loaders import czi_acquired_at


def _fake_czi(path, xml):
    """The two CZI segments the reader touches: the file header (with the
    metadata segment's offset) and the metadata segment's XML."""
    meta_pos = 512
    head = bytearray(512)
    head[:10] = b"ZISRAWFILE"
    struct.pack_into("<q", head, 92, meta_pos)
    body = xml.encode()
    seg = bytearray(32 + 256)
    seg[:14] = b"ZISRAWMETADATA"
    struct.pack_into("<i", seg, 32, len(body))
    with open(path, "wb") as fh:
        fh.write(bytes(head) + bytes(seg) + body + b"\0" * 64)
    return str(path)


def test_the_czi_header_gives_the_recording_time(tmp_path):
    p = _fake_czi(tmp_path / "a.czi", "<Document><CreationDate>2026-09-25T12:09:00</CreationDate></Document>")
    assert czi_acquired_at(p) == "2026-09-25T12:09:00"
    p2 = _fake_czi(tmp_path / "b.czi", "<X><AcquisitionDateAndTime>2026-10-01T21:23:43.1Z</AcquisitionDateAndTime>"
                                       "<CreationDate>2026-10-02T09:00:00</CreationDate></X>")
    assert czi_acquired_at(p2).startswith("2026-10-01T21:23:43")
    (tmp_path / "c.tif").write_bytes(b"II*\0" + b"\0" * 200)
    assert czi_acquired_at(str(tmp_path / "c.tif")) is None
    assert czi_acquired_at(str(tmp_path / "missing.czi")) is None


def test_an_older_run_reads_its_day_from_the_recording(tmp_path):
    p = _fake_czi(tmp_path / "a.czi", "<D><CreationDate>2026-09-16T11:35:30</CreationDate></D>")
    assert _recording_day({"input_file": p}) == "2026-09-16"
    assert _recording_day({"acquired_at": "2026-09-20T14:30:57", "input_file": p}) == "2026-09-20"
    assert _recording_day({"input_file": str(tmp_path / "gone.czi")}) is None


def test_day_shapes():
    assert day_markers(["2026-09-25"] * 4) == {}                       # one day: nothing to tell
    m = day_markers(["2026-10-01", "2026-09-25", "2026-10-01"])
    assert m == {"2026-09-25": DAY_MARKERS[0], "2026-10-01": DAY_MARKERS[1]}  # date order
    assert day_markers(["2026-09-25", None])[None] == UNDATED_MARKER
    assert day_markers([f"2026-09-{d:02d}" for d in range(1, len(DAY_MARKERS) + 2)]) == {}


def _groups(tmp_path, days_per_group):
    from test_workspace_data import make_run_folder
    out = []
    for gi, days in enumerate(days_per_group):
        folders = []
        for k, day in enumerate(days):
            f = make_run_folder(str(tmp_path), f"g{gi}_{k}", seed=10 * gi + k, d_centre=0.05 + 0.1 * gi)
            with open(os.path.join(f, "firefly_extras", f"g{gi}_{k}_params.json"), "w") as fh:
                json.dump({"pixel_size_um": 0.16, "frame_interval_s": 0.02,
                           **({"acquired_at": f"{day}T10:00:00"} if day else {})}, fh)
            folders.append(f)
        out.append({"label": f"G{gi}", "color": ["#0072b2", "#d55e00"][gi], "folders": folders})
    return out


def _auc_axis(fig):
    return next(a for a in fig.axes if a.get_ylabel().startswith("AUC"))


def _dot_shapes(ax):
    """{marker path vertex count: number of dots} — a stand-in for the shape."""
    out = {}
    for c in ax.collections:
        if len(c.get_offsets()) and c.get_paths():
            k = len(c.get_paths()[0].vertices)
            out[k] = out.get(k, 0) + len(c.get_offsets())
    return out


DAYS = [["2026-09-16", "2026-09-16", "2026-09-25"], ["2026-09-25", "2026-10-01", "2026-10-01"]]


def test_dots_take_one_shape_per_day_and_the_figure_keys_them(tmp_path):
    fig, sdf, _st = compare_groups(_groups(tmp_path, DAYS), output_dir=None, panels={"auc", "rg"},
                                   pdf_report=False, auc_plot_style="box_points")
    assert sorted(sdf["recording_day"]) == sorted(d for g in DAYS for d in g)
    shapes = _dot_shapes(_auc_axis(fig))
    assert sum(shapes.values()) == 6 and len(shapes) == 3              # 3 days → 3 shapes
    key = next(lg for lg in fig.legends if getattr(lg, "_firefly_day_key", False))
    assert [t.get_text() for t in key.get_texts()] == ["Recording day:", "16 Sep", "25 Sep", "1 Oct"]
    plt.close(fig)


def test_minimal_mode_keeps_the_shapes_but_drops_the_key(tmp_path):
    fig, _s, _st = compare_groups(_groups(tmp_path, DAYS), output_dir=None, panels={"auc"},
                                  pdf_report=False, auc_plot_style="box_points", minimal=True)
    assert len(_dot_shapes(_auc_axis(fig))) == 3
    assert not fig.legends
    plt.close(fig)


def test_undated_runs_and_single_day_designs_draw_plain_circles(tmp_path):
    fig, sdf, _st = compare_groups(_groups(tmp_path, [[None] * 3, [None] * 3]), output_dir=None,
                                   panels={"auc"}, pdf_report=False, auc_plot_style="box_points")
    assert sdf["recording_day"].isna().all()
    assert len(_dot_shapes(_auc_axis(fig))) == 1 and not fig.legends
    plt.close(fig)


def test_shapes_follow_the_replicates_past_a_missing_value(tmp_path):
    """α₂ is NaN for some replicates; the shapes must stay on the right dots."""
    from firefly.analysis.fa_compare import _bar_with_dots_n, _theme_palette
    fig, ax = plt.subplots()
    _bar_with_dots_n(ax, [np.array([1.0, np.nan, 3.0])], ["A"], ["#0072b2"], _theme_palette("Light"),
                     markers=[["o", "s", "^"]])
    got = {len(c.get_paths()[0].vertices): tuple(c.get_offsets()[:, 1])
           for c in ax.collections if len(c.get_offsets())}
    from matplotlib.markers import MarkerStyle
    nv = lambda m: len(MarkerStyle(m).get_path().vertices)
    assert got == {nv("o"): (1.0,), nv("^"): (3.0,)}
    plt.close(fig)


# ── the live Analysis tab ───────────────────────────────────────────────────
def test_a_loaded_run_knows_its_recording_day(tmp_path):
    from firefly.ui.controllers.workspace import workspace_data as wd
    folder = _groups(tmp_path, [["2026-09-16"]])[0]["folders"][0]
    assert wd.load_run(folder).recording_day() == "2026-09-16"


def test_the_live_graph_shapes_dots_by_day_with_a_key(monkeypatch):
    pytest.importorskip("PySide6")
    from firefly.ui.controllers.workspace import workspace_data as wd, workspace_figures as wf
    seen = []
    monkeypatch.setattr(wf, "_qimage_from_figure", lambda fig: seen.append(fig))
    groups = [{"label": "A", "color": "#0072b2", "values": np.array([1.0, np.nan, 1.2, 1.1]),
               "days": ["2026-09-16", "2026-09-20", "2026-09-25", "2026-09-16"]},
              {"label": "B", "color": "#d55e00", "values": np.array([2.0, 2.1, 1.9]),
               "days": ["2026-09-25", "2026-09-25", None]}]
    wf.render_metric(groups, wd.METRIC_BY_ID["a"], group_style="box_points")
    ax = seen[0].axes[0]
    assert sum(_dot_shapes(ax).values()) == 6 and len(_dot_shapes(ax)) == 3
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["16 Sep", "25 Sep", "undated"]


def test_the_controller_keeps_days_aligned_with_values(tmp_path):
    pytest.importorskip("PySide6")
    from types import SimpleNamespace
    from firefly.ui.controllers.workspace import workspace_data as wd
    from firefly.ui.controllers.workspace.workspace_controller import AnalysisWorkspaceController as WC
    runs = [wd.load_run(f) for f in _groups(tmp_path, [["2026-09-16", "2026-09-25"]])[0]["folders"]]
    cond = SimpleNamespace(active=lambda: [SimpleNamespace(run=r) for r in runs])
    metric = SimpleNamespace(scalar=lambda run: None if run is runs[0] else 1.0)
    vals, days = WC._values_days(None, cond, metric)
    assert list(vals) == [1.0] and days == ["2026-09-25"]
