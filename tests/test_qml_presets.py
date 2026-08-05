"""Phase-6d tests: PresetController save / load / delete / modified."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

import pytest                                            # noqa: E402

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                            # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


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


def _make(tmp_path, monkeypatch):
    from firefly.ui.controllers.params.sidebar_controller import SidebarController
    from firefly.ui.controllers.preset_controller import PresetController
    sb = SidebarController(FakeSettings())
    pc = PresetController(sb)
    monkeypatch.setattr(type(pc), "_dir", staticmethod(lambda: str(tmp_path)))
    return sb, pc


def test_preset_save_load_roundtrip(tmp_path, monkeypatch):
    sb, pc = _make(tmp_path, monkeypatch)
    sb.setValue("analysis/diameter", 11)
    sb.setValue("analysis/backend", "Crocker–Grier — Trackpy (CPU)")
    sb.setValue("analysis/gap_policy", "Contiguous observations (legacy)")
    assert pc.save("my preset") is True
    assert "my preset" in pc.names and pc.active == "my preset"
    assert os.path.isfile(os.path.join(tmp_path, "my preset.json"))

    # change values, then load the preset → restored
    sb.setValue("analysis/diameter", 5)
    sb.setValue("analysis/backend", "Auto")
    sb.setValue("analysis/gap_policy", "All timestamp pairs")
    pc.load("my preset")
    assert sb.get("analysis/diameter") == 11
    assert sb.get("analysis/backend") == "Crocker–Grier — Trackpy (CPU)"
    assert sb.get("analysis/gap_policy") == "Contiguous observations (legacy)"


def test_preset_modified_flag(tmp_path, monkeypatch):
    sb, pc = _make(tmp_path, monkeypatch)
    sb.setValue("analysis/diameter", 9)
    pc.save("p1")
    assert pc.modified is False
    sb.setValue("analysis/diameter", 13)     # diverge from the saved preset
    assert pc.modified is True
    pc.load("p1")                            # back to baseline
    assert pc.modified is False


def test_preset_names_sentinel_and_delete(tmp_path, monkeypatch):
    sb, pc = _make(tmp_path, monkeypatch)
    assert pc.names[0] == "— Current settings —"
    pc.save("a"); pc.save("b")
    assert pc.names == ["— Current settings —", "a", "b"]
    assert pc.remove("a") is True
    assert "a" not in pc.names
    # loading the sentinel clears the active preset (no-op apply)
    pc.load("— Current settings —")
    assert pc.active == "— Current settings —"


def test_preset_sanitises_name(tmp_path, monkeypatch):
    sb, pc = _make(tmp_path, monkeypatch)
    assert pc.save("bad/name*?!") is True
    assert "badname" in pc.names               # illegal chars stripped
    assert pc.save("   ") is False             # empty after strip


def test_preset_builtin_tag_stripped(tmp_path, monkeypatch):
    import json
    sb, pc = _make(tmp_path, monkeypatch)
    with open(os.path.join(tmp_path, "seed.json"), "w") as fh:
        json.dump({"__firefly_builtin__": True, "analysis/diameter": 15}, fh)
    pc.refresh()
    pc.load("seed")
    assert sb.get("analysis/diameter") == 15   # applied; tag ignored, no crash


def test_seed_builtins_installs_shipped_presets(tmp_path, monkeypatch):
    import json
    from firefly.ui.controllers.params.params_builder import LINKER_LABEL_TO_VALUE
    sb, pc = _make(tmp_path, monkeypatch)
    pc.seed_builtins()
    assert "PC12 Cells" in pc.names and "Drosophila Neurons" in pc.names
    pc12 = json.load(open(os.path.join(tmp_path, "PC12 Cells.json"), encoding="utf-8"))
    # the settings requested for the PC12 preset
    assert LINKER_LABEL_TO_VALUE[pc12["analysis/linker"]] == "trackpy"
    assert pc12["analysis/search_range"] == 3
    assert pc12["analysis/roi_mode"] == "Manual threshold"
    assert pc12["analysis/roi_bg_sigma"] == 100.0
    assert pc12["analysis/roi_threshold"] == 0.02
    assert pc12["analysis/gap_policy"] == "All timestamp pairs"
    # and it actually loads/applies cleanly through the sidebar
    pc.load("PC12 Cells")
    assert sb.get("analysis/search_range") == 3


def test_seed_builtins_never_clobbers_a_users_own_preset(tmp_path, monkeypatch):
    import json
    sb, pc = _make(tmp_path, monkeypatch)
    mine = {"analysis/diameter": 99}                 # no __firefly_builtin__ tag
    with open(os.path.join(tmp_path, "PC12 Cells.json"), "w", encoding="utf-8") as fh:
        json.dump(mine, fh)
    pc.seed_builtins()
    kept = json.load(open(os.path.join(tmp_path, "PC12 Cells.json"), encoding="utf-8"))
    assert kept == mine                              # user's own preset untouched
    assert "Drosophila Neurons" in pc.names          # the non-conflicting one still seeds


# ── manifest replay pins the resolved auto-minmass threshold ──────────────────
def test_manifest_replay_pins_auto_minmass():
    from firefly.ui.controllers.params.sidebar_controller import SidebarController
    sb = SidebarController(FakeSettings())
    manifest = {
        "firefly_version": "9.9.9", "created_at": "2026-01-01T00:00:00",
        "resolved_minmass": 0.6671899791370938,
        "widget_state": {"analysis/auto_minmass": True, "analysis/minmass": 1.0,
                         "analysis/diameter": 7},
    }
    sb._apply_manifest(manifest, "run_manifest.json")
    assert sb.get("analysis/auto_minmass") is False                 # auto pinned off
    assert abs(float(sb.get("analysis/minmass")) - 0.6671899791370938) < 1e-9
    assert int(sb.get("analysis/diameter")) == 7                    # rest of state still applied


def test_manifest_replay_manual_run_not_pinned():
    from firefly.ui.controllers.params.sidebar_controller import SidebarController
    sb = SidebarController(FakeSettings())
    manifest = {                                       # a manual run: nothing to pin
        "resolved_minmass": None,
        "widget_state": {"analysis/auto_minmass": False, "analysis/minmass": 0.5},
    }
    sb._apply_manifest(manifest, "m")
    assert sb.get("analysis/auto_minmass") is False
    assert abs(float(sb.get("analysis/minmass")) - 0.5) < 1e-9      # left untouched


def test_manifest_replay_restores_gap_policy_from_wire_contract():
    from firefly.ui.controllers.params.sidebar_controller import SidebarController
    sb = SidebarController(FakeSettings())
    sb._apply_manifest({
        "schema_version": 4,
        "gap_policy": "contiguous",
        "widget_state": {},
    }, "legacy-compatible")
    assert sb.get("analysis/gap_policy") == "Contiguous observations (legacy)"


def test_quality_first_exposes_floor_and_only_relevant_controls():
    """Quality-first uses the minmass field as a locked lower floor.

    Density matching and linkability sensitivity are different scientific
    policies, so their controls must not look active in this mode.
    """
    from firefly.ui.controllers.params.sidebar_controller import SidebarController

    sb = SidebarController(FakeSettings())
    sb.setValue("analysis/auto_minmass", True)
    sb.setValue("analysis/minmass_mode", "Quality-first (track ambiguity)")
    assert sb.isEnabled("analysis/minmass") is True
    assert sb.isEnabled("analysis/minmass_max_false_track_rate") is True
    assert sb.isEnabled("analysis/minmass_target_density") is False
    assert sb.isEnabled("analysis/minmass_sensitivity") is False

    sb.setValue("analysis/minmass_mode", "Density-matched")
    assert sb.isEnabled("analysis/minmass") is False
    assert sb.isEnabled("analysis/minmass_target_density") is True
    assert sb.isEnabled("analysis/minmass_max_false_track_rate") is False

    sb.setValue("analysis/auto_minmass", False)
    assert sb.isEnabled("analysis/minmass") is True
    assert sb.isEnabled("analysis/minmass_mode") is False
