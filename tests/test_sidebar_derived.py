"""SidebarController.derivedHint — live real-units readouts (frames→s, px→µm/nm)
shown beside frame/pixel-based fields so the user sees what the MSD curve uses."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from firefly.ui.controllers.params.sidebar_controller import SidebarController

_app = QApplication.instance() or QApplication([])


class FakeSettings:
    def __init__(self): self.d = {}
    def get_str(self, k, d=""):
        v = self.d.get(k, d); return "" if v is None else str(v)
    def get_float(self, k, d=0.0):
        try: return float(self.d.get(k, d))
        except (TypeError, ValueError): return float(d)
    def get_bool(self, k, d=False):
        v = self.d.get(k, d)
        if isinstance(v, str): return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)
    def set(self, k, v): self.d[k] = v
    def sync(self): pass


def _sb(dt=0.02, px=0.106, **fields):
    sb = SidebarController(FakeSettings())
    sb.setValue("analysis/frame_interval", dt)
    sb.setValue("analysis/pixel_size", px)
    for k, v in fields.items():
        sb.setValue("analysis/" + k, v)
    return sb


def test_lagtime_and_fit_window_in_seconds():
    sb = _sb(max_lagtime=20, n_fit=5)
    assert sb.derivedHint("analysis/max_lagtime") == "0.4 s"   # 20 × 0.02 s
    assert sb.derivedHint("analysis/n_fit") == "0.1 s"         # 5 × 0.02 s


def test_fit_window_clamped_to_max_lagtime():
    # the analysis clamps n_fit to max_lagtime; the readout must reflect that
    sb = _sb(max_lagtime=5, n_fit=20)
    assert sb.derivedHint("analysis/n_fit") == "0.1 s"         # min(20,5) × 0.02


def test_memory_and_drift_segment_seconds():
    sb = _sb(memory=3, drift_segment=500)
    assert sb.derivedHint("analysis/memory") == "0.06 s"
    assert sb.derivedHint("analysis/drift_segment") == "10 s"


def test_pixel_based_um_and_nm():
    sb = _sb(search_range=5, diameter=7)
    assert sb.derivedHint("analysis/search_range") == "0.53 µm"  # 5 × 0.106 µm
    assert sb.derivedHint("analysis/diameter") == "742 nm"       # 7 × 0.106 µm → nm


def test_tracks_live_frame_interval():
    sb = _sb(max_lagtime=10)
    assert sb.derivedHint("analysis/max_lagtime") == "0.2 s"     # 10 × 0.02
    sb.setValue("analysis/frame_interval", 0.05)
    assert sb.derivedHint("analysis/max_lagtime") == "0.5 s"     # 10 × 0.05 (live)


def test_non_derived_field_is_blank():
    sb = _sb(max_lagtime=20)
    assert sb.derivedHint("analysis/minmass") == ""             # not a derived field
    assert sb.derivedHint("analysis/alpha_immobile") == ""


def test_missing_scale_is_blank():
    # a corrupt/zero scale in the settings store must yield no readout (defensive;
    # the field min normally prevents 0 via the UI) — write the raw value directly
    sb = _sb(max_lagtime=20, search_range=5)
    sb._s.d["analysis/frame_interval"] = 0.0
    assert sb.derivedHint("analysis/max_lagtime") == ""
    sb._s.d["analysis/pixel_size"] = 0.0
    assert sb.derivedHint("analysis/search_range") == ""
