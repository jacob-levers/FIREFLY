"""Phase-3 tests: the QML AnalysisController run lifecycle + params builder.

The headline guarantee is **worker parity**: the params dict the QML builder
produces from an empty settings store is byte-identical (same keys + values) to
the Widgets app's ``_build_params_for_file`` — so ``firefly_worker`` needs no
changes.  Plus pure-unit coverage of the builder's mapping, the stepper stage
classifier, and the controller's terminal-state transitions.

Qt-gated like the other UI tests (offscreen + ``importorskip``).
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

import tempfile                                          # noqa: E402
import pytest                                            # noqa: E402

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                            # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ── fakes ────────────────────────────────────────────────────────────────────
class FakeSettings:
    """Empty settings store → every get returns the supplied default."""
    def __init__(self, d=None): self.d = d or {}
    def get_str(self, k, default=""):
        v = self.d.get(k, default); return "" if v is None else str(v)
    def get_float(self, k, default=0.0):
        try: return float(self.d.get(k, default))
        except (TypeError, ValueError): return float(default)
    def get_bool(self, k, default=False):
        v = self.d.get(k, default)
        if isinstance(v, str): return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)
    def set(self, k, v): self.d[k] = v
    def sync(self): pass


class FakeImport:
    def __init__(self, path="/tmp/example.tif", out="", is_csv=False,
                 override_px=False, px=0.106, override_fi=False, fi=0.02):
        self.filePath = path
        self.outDir = out
        self.isCsv = is_csv
        self.overridePx = override_px
        self.pixelSize = px
        self.overrideFi = override_fi
        self.frameInterval = fi
        self.hasFile = bool(path)


# ── params builder: mapping + defaults ───────────────────────────────────────
def test_build_params_defaults_and_mapping():
    from firefly.ui.controllers import params_builder as pb
    p = pb.build_params(FakeSettings(), FakeImport())
    # worker-critical keys the analysis body subscripts WITHOUT a default
    for k in ("file", "diameter", "backend", "workers", "minmass", "chunk_size"):
        assert k in p
    assert p["diameter"] == 7
    assert p["backend"] == "auto"           # "Auto" label → "auto"
    assert p["linker"] == "kalman"
    assert p["roi_mode"] == "auto"          # "Auto threshold" → "auto"
    assert p["bg_method"] == "uniform_filter"
    assert p["minmass_sensitivity"] == "balanced"
    assert p["alpha_thresholds"] == (0.5, 0.9, 1.1)
    assert p["max_track_len"] is None       # 0 → None
    assert p["minmass_max_false_track_rate"] is None
    # no calibration override → None (worker reads file metadata)
    assert p["pixel_size"] is None and p["frame_interval"] is None
    assert isinstance(p["widget_state"], dict)


def test_build_params_calibration_override():
    from firefly.ui.controllers import params_builder as pb
    imp = FakeImport(override_px=True, px=0.108, override_fi=True, fi=0.03)
    p = pb.build_params(FakeSettings(), imp)
    assert p["pixel_size"] == 0.108 and p["frame_interval"] == 0.03


def test_build_params_csv_seeds_calibration():
    from firefly.ui.controllers import params_builder as pb
    imp = FakeImport(path="/tmp/locs.csv", is_csv=True, px=0.1, fi=0.05)
    p = pb.build_params(FakeSettings(), imp)
    # CSV has no embedded metadata → calibration seeded even without override
    assert p["source"] == "external_csv" and p["csv_preset"] == "auto"
    assert p["pixel_size"] == 0.1 and p["frame_interval"] == 0.05


def test_build_params_combo_overlay_from_settings():
    from firefly.ui.controllers import params_builder as pb
    s = FakeSettings({
        "analysis/backend": "Crocker–Grier — PyTorch (GPU)",
        "analysis/linker":  "Jaqaman LAP — TrackMate (merge/split)",
        "analysis/roi_mode": "Sister TIFF",
        "analysis/diameter": 11,
        "analysis/auto_minmass": "false",
        "analysis/minmass": 2.5,
    })
    p = pb.build_params(s, FakeImport())
    assert p["backend"] == "torch"
    assert p["linker"] == "full_lap"
    assert p["roi_mode"] == "sister"
    assert p["diameter"] == 11
    assert p["auto_minmass"] is False and p["minmass"] == 2.5


# ── WORKER PARITY: builder output == Widgets _build_params_for_file ───────────
def test_params_byte_identical_to_widgets_app():
    """Against an EMPTY settings store, the QML builder reproduces the Widgets
    app's params dict exactly (same keys + values), so firefly_worker is
    unchanged."""
    from PySide6 import QtCore
    # Redirect QSettings to a fresh empty ini so the Widgets app starts pristine.
    tmp = tempfile.mktemp(suffix=".ini")
    orig = QtCore.QSettings
    QtCore.QSettings = lambda *a, **k: orig(tmp, orig.Format.IniFormat)
    try:
        from firefly.ui.app_qt import MainWindow
        w = MainWindow()
        widgets = w._build_params_for_file("/tmp/example.tif", None)
    finally:
        QtCore.QSettings = orig
    widgets.pop("widget_state", None)

    from firefly.ui.controllers import params_builder as pb
    qml = pb.build_params(FakeSettings(), FakeImport(path="/tmp/example.tif"))
    qml.pop("widget_state", None)

    assert set(qml) == set(widgets), (
        f"key mismatch: only-qml={set(qml) - set(widgets)}, "
        f"only-widgets={set(widgets) - set(qml)}")
    assert qml == widgets, {
        k: (qml[k], widgets[k]) for k in qml if qml[k] != widgets[k]}


# ── stepper stage classifier (ported verbatim) ───────────────────────────────
def test_index_for_msg_classifier():
    from firefly.ui.controllers.analysis_controller import _index_for_msg
    assert _index_for_msg("Loading stack…") == ("idx", 0)
    assert _index_for_msg("Localising frame 12") == ("idx", 1)
    assert _index_for_msg("Linking trajectories") == ("idx", 2)
    assert _index_for_msg("Drift correction") == ("idx", 3)
    assert _index_for_msg("Computing MSD fits") == ("idx", 4)
    assert _index_for_msg("Saving outputs") == ("idx", 5)
    assert _index_for_msg("Analysis complete") == ("complete", None)
    assert _index_for_msg("nonsense") == (None, None)


def test_format_elapsed():
    from firefly.ui.controllers.analysis_controller import _format_elapsed
    assert _format_elapsed(0) == "00:00"
    assert _format_elapsed(75) == "01:15"
    assert _format_elapsed(3725) == "1:02:05"


# ── controller construction + terminal transitions ───────────────────────────
def test_analysis_controller_idle_state_and_done():
    from firefly.ui.controllers.analysis_controller import AnalysisController, STAGES
    c = AnalysisController(FakeSettings(), FakeImport())
    assert not c.running and c.stage == -1 and not c.complete
    assert list(c.stages) == STAGES

    finished = []
    c.runFinished.connect(lambda: finished.append(1))
    c._handle_done({"stem": "cell1", "out_dir": "/out",
                    "summary": {"n_tracks": 1234}})
    assert c.complete and c.stage == len(STAGES) - 1
    assert "1,234 trajectories" in c.resultHeadline
    assert c.resultSeverity == "ok" and c.resultOutDir == "/out"
    assert c.stats.get("n_tracks") == 1234 and finished == [1]


def test_analysis_controller_zero_tracks_is_warning():
    from firefly.ui.controllers.analysis_controller import AnalysisController
    c = AnalysisController(FakeSettings(), FakeImport())
    c._handle_done({"stem": "cell1", "out_dir": "/out", "summary": {"n_tracks": 0}})
    assert c.resultSeverity == "warn"
    assert "no trajectories" in c.resultHeadline.lower()


def test_analysis_controller_failure_transition():
    from firefly.ui.controllers.analysis_controller import AnalysisController
    c = AnalysisController(FakeSettings(), FakeImport())
    failed = []
    c.runFailed.connect(lambda tb: failed.append(tb))
    c._handle_failed("Traceback: boom")
    assert c.resultSeverity == "error" and failed and "boom" in failed[0]


# ── live-frame renderer ──────────────────────────────────────────────────────
def test_render_frame_grayscale_and_markers():
    import numpy as np
    from firefly.ui.controllers.live_frame_provider import render_frame
    arr = np.zeros((20, 24), dtype=np.float32)
    arr[10, 12] = 100.0
    img = render_frame(arr, xs=[5.0], ys=[8.0])
    assert not img.isNull() and img.width() == 24 and img.height() == 20
    # the marker pixel is tinted toward the accent (blue > red)
    c = img.pixelColor(5, 8)
    assert c.blue() >= c.red()
