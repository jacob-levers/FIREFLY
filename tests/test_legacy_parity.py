"""Headless tests for three legacy-UI features restored in the QML app:

  1. run-manifest replay     (SidebarController.loadManifest)
  2. CSV import options       (ImportController.csvPreset / bgImagePath → params)
  3. crash reporter           (global excepthook writes a report)

All settings go through an in-memory fake — never the real ``jacoblevers/FIREFLY``
QSettings domain (SettingsController hardcodes it, so a real one would pollute
the user's app preferences).
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json
import sys
import threading

import pytest

pytest.importorskip("PySide6")
from PySide6 import QtWidgets
from PySide6.QtCore import QSettings  # noqa: F401  (ensure Qt is initialised)

from firefly.ui.controllers.import_controller import ImportController
from firefly.ui.controllers.params.sidebar_controller import SidebarController
from firefly.ui.controllers.params import params_builder as pb
from firefly import crash_reporter

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _MemSettings:
    """In-memory stand-in for SettingsController (same get/set surface)."""
    def __init__(self):
        self._d = {}

    def get(self, key, default=None):
        return self._d.get(key, default)

    def get_str(self, key, default=""):
        v = self._d.get(key, default)
        return str(v) if v is not None else default

    def get_float(self, key, default=0.0):
        try:
            return float(self._d.get(key, default))
        except (TypeError, ValueError):
            return default

    def get_bool(self, key, default=False):
        v = self._d.get(key, default)
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ("1", "true", "yes")
        return bool(v)

    def set(self, key, value):
        self._d[key] = value

    def sync(self):
        pass


# ── 1. run-manifest replay ───────────────────────────────────────────────────
def test_manifest_replay(tmp_path, monkeypatch):
    s = _MemSettings()
    importc = ImportController(s)
    sb = SidebarController(s, importc)

    fake_input = tmp_path / "movie.tif"
    fake_input.write_text("x")
    manifest = {
        "widget_state": {"analysis/diameter": 11, "analysis/memory": 7,
                         "analysis/min_track_len": 9},
        "input": {"path": str(fake_input)},
        "output_dir": str(tmp_path),
        "firefly_version": "9.9.9", "created_at": "2026-06-21T00:00:00",
    }
    mpath = tmp_path / "movie_run_manifest.json"
    mpath.write_text(json.dumps(manifest))

    msgs = []
    sb.manifestLoaded.connect(msgs.append)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(mpath), "")))
    sb.loadManifest()

    assert sb.get("analysis/diameter") == 11
    assert sb.get("analysis/memory") == 7
    assert sb.get("analysis/min_track_len") == 9
    assert importc.filePath == str(fake_input)
    assert importc.outDir == str(tmp_path)
    assert msgs and "9.9.9" in msgs[0]


def test_manifest_replay_bad_file_is_handled(tmp_path, monkeypatch):
    s = _MemSettings()
    sb = SidebarController(s, ImportController(s))
    bad = tmp_path / "broken_run_manifest.json"
    bad.write_text("{ not json")
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(bad), "")))
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical",
                        staticmethod(lambda *a, **k: None))
    sb.loadManifest()        # must not raise


# ── 2. CSV import options ────────────────────────────────────────────────────
def test_csv_preset_and_bg_flow_into_params(tmp_path):
    s = _MemSettings()
    importc = ImportController(s)
    csv = tmp_path / "locs.csv"
    csv.write_text("frame,x,y\n0,1,2\n")
    bg = tmp_path / "bg.tif"
    bg.write_text("x")

    assert importc.csvPreset == "auto"          # default
    assert importc.bgImagePath == ""

    importc.filePath = str(csv)
    importc.csvPreset = "ThunderSTORM"
    importc.bgImagePath = str(bg)

    params = pb.build_params(s, importc, fpath=str(csv))
    assert params["source"] == "external_csv"
    assert params["csv_preset"] == "ThunderSTORM"
    assert params["bg_image_path"] == str(bg)


def test_bg_clears_on_file_change_preset_persists(tmp_path):
    s = _MemSettings()
    importc = ImportController(s)
    c1 = tmp_path / "a.csv"; c1.write_text("frame,x,y\n0,1,2\n")
    c2 = tmp_path / "b.csv"; c2.write_text("frame,x,y\n0,1,2\n")
    bg = tmp_path / "bg.tif"; bg.write_text("x")

    importc.filePath = str(c1)
    importc.csvPreset = "Picasso"
    importc.bgImagePath = str(bg)

    importc.filePath = str(c2)                   # new recording
    assert importc.bgImagePath == ""             # bg is per-file → cleared
    assert importc.csvPreset == "Picasso"        # preset is a preference → kept


# ── 3. crash reporter ────────────────────────────────────────────────────────
def test_crash_reporter_writes_report(tmp_path, monkeypatch):
    monkeypatch.setattr(crash_reporter, "crash_report_dir", lambda: str(tmp_path))
    old_excepthook = sys.excepthook
    old_threadhook = getattr(threading, "excepthook", None)
    try:
        seen = []
        crash_reporter.set_log_provider(lambda n=120: "log tail line")
        crash_reporter.set_app_state_provider(lambda: {"UI": "PySide6 QML"})
        crash_reporter.install_global_handlers(on_crash=seen.append)
        try:
            raise RuntimeError("synthetic crash for test")
        except RuntimeError:
            crash_reporter._main_excepthook(*sys.exc_info())
        assert seen, "on_crash should fire with the report path"
        reports = list(tmp_path.glob("*.txt"))
        assert reports, "a crash report file should be written"
        body = reports[0].read_text()
        assert "PySide6 QML" in body
        assert "log tail line" in body
        assert "synthetic crash for test" in body
    finally:
        sys.excepthook = old_excepthook
        if old_threadhook is not None:
            threading.excepthook = old_threadhook
