"""Phase-5b tests: CompareController + compare_params_builder + RunSession.

The gate is params PARITY: the QML builder's comparison_params is byte-identical
to the Widgets _start_compare_run block (so run_comparison is unchanged).
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

import tempfile                                          # noqa: E402
import pytest                                            # noqa: E402

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                            # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class FakeSettings:
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


# ── params builder ───────────────────────────────────────────────────────────
def test_compare_params_shape_and_defaults():
    from firefly.ui.controllers import compare_params_builder as cpb
    groups = [{"label": "A", "color": "#fff", "timepoint": "", "folders": ["/a"]},
              {"label": "B", "color": "#000", "timepoint": "", "folders": ["/b"]}]
    p = cpb.build_comparison_params(FakeSettings(), groups, "/out", "cmp")
    assert set(p) == {"groups", "output_dir", "output_stem", "theme", "pdf_report",
                      "panels", "mobile_d_threshold", "logd_plot_style",
                      "stats_config", "use_native"}
    assert p["output_dir"] == "/out" and p["output_stem"] == "cmp"
    assert p["theme"] == "Dark" and p["use_native"] is False
    assert p["logd_plot_style"] == "overlaid"
    assert len(p["panels"]) == 13                      # all panels by default
    sc = p["stats_config"]
    assert sc["correction"] == "holm" and sc["anova3plus"] == "welch"
    assert sc["nonparametric_test"] == "mann_whitney" and sc["alpha"] == 0.05
    assert sc["include_circular_outputs"] is True


def test_compare_params_combo_overlay():
    from firefly.ui.controllers import compare_params_builder as cpb
    s = FakeSettings({"stats/correction": "Bonferroni",
                      "stats/nonparam": "Brunner-Munzel",
                      "compare/theme": "Publication",
                      "compare/panel_jdd": "false"})
    p = cpb.build_comparison_params(s, [], "/o")
    assert p["theme"] == "Publication"
    assert p["stats_config"]["correction"] == "bonferroni"
    assert p["stats_config"]["nonparametric_test"] == "brunner_munzel"
    assert "jdd" not in p["panels"]


def test_compare_params_parity_with_widgets():
    """The QML builder reproduces _start_compare_run's params block exactly."""
    from PySide6 import QtCore
    tmp = tempfile.mktemp(suffix=".ini")
    orig = QtCore.QSettings
    QtCore.QSettings = lambda *a, **k: orig(tmp, orig.Format.IniFormat)
    try:
        from firefly.ui.app_qt import MainWindow
        w = MainWindow()
        groups = [{"label": "A", "color": "#3b6ed8", "timepoint": "", "folders": ["/a"]},
                  {"label": "B", "color": "#f78166", "timepoint": "", "folders": ["/b"]}]
        # Replicate the widget params block (panels is a set→list; compare as sets).
        widget = {
            "groups": groups, "output_dir": "/out", "output_stem": "comparison",
            "theme": w.c_cmp_theme.currentText(),
            "pdf_report": bool(w.c_cmp_pdf.isChecked()),
            "panels": {k for k, cb in w._cmp_panel_checkboxes.items() if cb.isChecked()},
            "mobile_d_threshold": float(w.s_mobile_d_threshold.value()),
            "logd_plot_style": str(w._settings.value("figures/logd_style", "overlaid") or "overlaid"),
            "stats_config": w._collect_stats_config(),
            "use_native": False,
        }
    finally:
        QtCore.QSettings = orig
    from firefly.ui.controllers import compare_params_builder as cpb
    qml = cpb.build_comparison_params(FakeSettings(), groups, "/out", "comparison")
    assert qml["stats_config"] == widget["stats_config"]
    assert set(qml["panels"]) == set(widget["panels"])
    assert qml["theme"] == widget["theme"]
    assert qml["pdf_report"] == widget["pdf_report"]
    assert qml["mobile_d_threshold"] == widget["mobile_d_threshold"]
    assert qml["logd_plot_style"] == widget["logd_plot_style"]
    assert qml["use_native"] == widget["use_native"]


# ── controller groups model ──────────────────────────────────────────────────
def test_compare_controller_groups_model(tmp_path):
    from firefly.ui.controllers.compare_controller import CompareController
    c = CompareController(FakeSettings())
    assert len(c.conditions) == 2 and not c.canGenerate       # seeded, empty
    d1 = tmp_path / "g1"; d1.mkdir()
    d2 = tmp_path / "g2"; d2.mkdir()
    ids = [cond["id"] for cond in c.conditions]
    c.setLabel(ids[0], "control")
    c.addFolders(ids[0], [str(d1)])
    c.addFolders(ids[1], [str(d2)])
    assert c.conditions[0]["name"] == "control"
    assert c.conditions[0]["folderCount"] == 1
    assert c.canGenerate                                       # 2 non-empty groups
    c.removeFolder(ids[1], str(d2))
    assert not c.canGenerate
    # cap at 6
    while len(c.conditions) < 6:
        c.addCondition()
    n = len(c.conditions)
    c.addCondition()
    assert len(c.conditions) == n == 6


def test_compare_controller_validation():
    from firefly.ui.controllers.compare_controller import CompareController
    c = CompareController(FakeSettings())
    c.generate()                                              # no folders
    assert "at least 2 groups" in c.generateError.lower()


def test_compare_controller_done_derives_summary(tmp_path):
    import json
    from firefly.ui.controllers.compare_controller import CompareController
    c = CompareController(FakeSettings())
    rj = tmp_path / "c_results.json"
    rj.write_text(json.dumps({
        "stats": {"median_D": {"omnibus": {"test": "Welch ANOVA", "p": 0.004, "stars": "**"}}},
        "summary": [{"group": "A", "median_D": 0.1, "median_alpha": 0.8},
                    {"group": "B", "median_D": 0.2, "median_alpha": 0.9}]}))
    ready = []
    c.resultsReady.connect(lambda p: ready.append(p))
    c._on_done({"output_dir": str(tmp_path), "n_groups": 2, "results_json": str(rj)})
    assert c.hasResult and ready == [str(rj)]
    assert c.pValueLabel == "p = 0.004" and c.significant
    assert {r["groupLabel"] for r in c.statsRows} == {"A", "B"}


# ── RunSession ───────────────────────────────────────────────────────────────
def test_run_session_idle():
    from firefly.ui.controllers.run_session import RunSession
    s = RunSession()
    assert not s.running
