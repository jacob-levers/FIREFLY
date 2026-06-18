"""Phase-6a tests: SidebarController + sidebar_schema.

The sidebar writes the SAME QSettings keys params_builder reads, so the worker
param dict stays byte-identical — the gate is (1) every schema key round-trips
through the controller, and (2) a sidebar-configured run reproduces the Widgets
params exactly.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

import pytest                                            # noqa: E402

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                            # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class FakeSettings:
    def __init__(self, d=None): self.d = dict(d or {})
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


# ── schema sanity ────────────────────────────────────────────────────────────
def test_schema_keys_cover_params_builder_settings():
    from firefly.ui.controllers import sidebar_schema as S
    keys = {f["key"] for f in S.FIELDS}
    # the keys params_builder reads from analysis/* + figures/* must be writable
    for k in ("analysis/diameter", "analysis/backend", "analysis/linker",
              "analysis/minmass", "analysis/roi_mode", "analysis/workers",
              "figures/theme", "figures/dpi", "analysis/cluster_eps_nm"):
        assert k in keys, k
    # every field belongs to a declared section
    sect = {s["key"] for s in S.SECTIONS}
    assert all(f["section"] in sect for f in S.FIELDS)


# ── round-trip every key ─────────────────────────────────────────────────────
def test_sidebar_roundtrips_every_key():
    from firefly.ui.controllers import sidebar_schema as S
    from firefly.ui.controllers.sidebar_controller import SidebarController
    c = SidebarController(FakeSettings())
    for f in S.FIELDS:
        k, kind = f["key"], f["kind"]
        if kind == "bool":
            c.setValue(k, True); assert c.get(k) is True
            c.setValue(k, False); assert c.get(k) is False
        elif kind == "combo":
            for label in f["items"]:
                c.setValue(k, label); assert c.get(k) == label
        elif kind == "int":
            mid = int((f["min"] + f["max"]) // 2) if f["max"] < 1e6 else 7
            c.setValue(k, mid); assert c.get(k) == mid
        elif kind == "double":
            c.setValue(k, f["min"]); assert abs(c.get(k) - f["min"]) < 1e-9


def test_sidebar_clamps_to_range():
    from firefly.ui.controllers.sidebar_controller import SidebarController
    c = SidebarController(FakeSettings())
    c.setValue("analysis/diameter", 999)
    assert c.get("analysis/diameter") == 21           # max
    c.setValue("analysis/diameter", -5)
    assert c.get("analysis/diameter") == 3            # min


def test_sidebar_enablement_matrix():
    from firefly.ui.controllers.sidebar_controller import SidebarController
    c = SidebarController(FakeSettings())
    # auto_minmass default True → minmass disabled, sensitivity enabled
    assert c.isEnabled("analysis/minmass") is False
    assert c.isEnabled("analysis/minmass_sensitivity") is True
    c.setValue("analysis/auto_minmass", False)
    assert c.isEnabled("analysis/minmass") is True
    assert c.isEnabled("analysis/minmass_sensitivity") is False
    # full-LAP gating
    assert c.isEnabled("analysis/allow_merging") is False
    c.setValue("analysis/linker", "Jaqaman LAP — TrackMate (merge/split)")
    assert c.isEnabled("analysis/allow_merging") is True
    # filter-D gating
    assert c.isEnabled("analysis/filter_d_min") is False
    c.setValue("analysis/filter_d_enable", True)
    assert c.isEnabled("analysis/filter_d_min") is True


def test_sidebar_revision_bumps():
    from firefly.ui.controllers.sidebar_controller import SidebarController
    c = SidebarController(FakeSettings())
    r0 = c.revision
    c.setValue("analysis/diameter", 9)
    assert c.revision == r0 + 1
    c.resetAll()
    assert c.revision == r0 + 2


def test_sidebar_reset_writes_defaults():
    from firefly.ui.controllers import sidebar_schema as S
    from firefly.ui.controllers.sidebar_controller import SidebarController
    fs = FakeSettings()
    c = SidebarController(fs)
    c.setValue("analysis/diameter", 11)
    c.setValue("analysis/backend", "Crocker–Grier — PyTorch (GPU)")
    c.resetAll()
    assert c.get("analysis/diameter") == S.BY_KEY["analysis/diameter"]["default"]
    assert c.get("analysis/backend") == "Auto"


def test_sidebar_fields_and_sections():
    from firefly.ui.controllers.sidebar_controller import SidebarController
    c = SidebarController(FakeSettings())
    assert len(c.sections) == 12
    det = c.fields("detection")
    keys = {f["key"] for f in det}
    assert "analysis/diameter" in keys and "analysis/minmass" in keys


# ── params parity: a sidebar-configured run == Widgets _build_params_for_file ─
def test_sidebar_edits_produce_widgets_identical_params(tmp_path):
    """Configure a few params via the SidebarController, then assert the QML
    builder output matches the Widgets app with the SAME QSettings."""
    from PySide6 import QtCore
    from firefly.ui.controllers.sidebar_controller import SidebarController
    from firefly.ui.controllers import params_builder as pb

    # a real (temp) QSettings the sidebar writes and the Widgets app reads
    import tempfile
    ini = tempfile.mktemp(suffix=".ini")

    class RealSettings:
        def __init__(self):
            self._s = QtCore.QSettings(ini, QtCore.QSettings.Format.IniFormat)
        def get_str(self, k, d=""):
            v = self._s.value(k, d); return "" if v is None else str(v)
        def get_float(self, k, d=0.0):
            try: return float(self._s.value(k, d))
            except (TypeError, ValueError): return float(d)
        def get_bool(self, k, d=False):
            v = self._s.value(k, d)
            return v.strip().lower() in ("1", "true", "yes", "on") if isinstance(v, str) else bool(v)
        def set(self, k, v): self._s.setValue(k, v)
        def sync(self): self._s.sync()

    rs = RealSettings()
    sb = SidebarController(rs)
    sb.setValue("analysis/diameter", 9)
    sb.setValue("analysis/backend", "Crocker–Grier — Trackpy (CPU)")
    sb.setValue("analysis/auto_minmass", False)
    sb.setValue("analysis/minmass", 2.5)
    sb.setValue("analysis/roi_mode", "Sister TIFF")
    sb.setValue("figures/dpi", 200)
    rs.sync()

    class Imp:
        filePath = "/tmp/x.tif"; outDir = ""; isCsv = False
        overridePx = False; pixelSize = 0.106; overrideFi = False; frameInterval = 0.02
    qml = pb.build_params(rs, Imp(), "/tmp/x.tif", None)
    qml.pop("widget_state", None)

    # Widgets app reading the SAME ini
    orig = QtCore.QSettings
    QtCore.QSettings = lambda *a, **k: orig(ini, orig.Format.IniFormat)
    try:
        from firefly.ui.app_qt import MainWindow
        w = MainWindow()
        widgets = w._build_params_for_file("/tmp/x.tif", None)
    finally:
        QtCore.QSettings = orig
    widgets.pop("widget_state", None)

    assert qml == widgets, {k: (qml.get(k), widgets.get(k))
                            for k in set(qml) | set(widgets) if qml.get(k) != widgets.get(k)}
