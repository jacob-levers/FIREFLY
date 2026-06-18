"""Phase-5a tests: ResultsController + the shared results_format helpers."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

import json                                              # noqa: E402
import pytest                                            # noqa: E402

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                            # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _results_json(figure_path=None):
    return {
        "meta": {
            "two_factor": False, "n_groups": 2,
            "group_labels": ["control", "+noco"],
            "group_colors": ["#58a6ff", "#f5a623"],
            "files": ({"png": os.path.basename(figure_path)} if figure_path else {}),
        },
        "summary": [
            {"group": "control", "cell": "c1", "n_tracks": 1200, "median_D": 0.10,
             "median_alpha": 0.85, "nongauss_alpha2": 0.30},
            {"group": "control", "cell": "c2", "n_tracks": 1300, "median_D": 0.12,
             "median_alpha": 0.88, "nongauss_alpha2": 0.32},
            {"group": "+noco", "cell": "c3", "n_tracks": 900, "median_D": 0.05,
             "median_alpha": 0.70, "nongauss_alpha2": 0.40},
        ],
        "config_summary": [["Test", "Welch's t-test"], ["α", "0.05"]],
        "stats": {
            "median_D": {
                "pairwise": [{"label_i": "control", "label_j": "+noco",
                              "mean_i": 0.11, "mean_j": 0.05, "p_within": 0.002,
                              "stars_within": "**", "hedges_g": 1.2,
                              "hedges_g_ci_low": 0.4, "hedges_g_ci_high": 2.0}]},
        },
    }


# ── shared formatters ────────────────────────────────────────────────────────
def test_results_format_helpers():
    from firefly.ui import results_format as rf
    assert rf._fmt_p(0.0001) == "1.00e-04"
    assert rf._fmt_p(0.5) == "0.500"
    assert rf._fmt_p(None) == "—"
    mag, txt = rf._effect_phrase({"hedges_g": 1.2, "hedges_g_ci_low": 0.4,
                                  "hedges_g_ci_high": 2.0})
    assert mag == "large" and txt == "g = 1.20 [0.40, 2.00]"
    sev, html, under = rf._verdict_for_metric(
        "Median D", _results_json()["stats"]["median_D"], 2)
    assert sev == "success" and "control" in html and not under
    assert rf.ordered_metrics({"median_D": {}, "zzz_custom": {}})[0][0] == "median_D"


# ── ResultsController ────────────────────────────────────────────────────────
def test_results_controller_idle():
    from firefly.ui.controllers.results_controller import ResultsController
    c = ResultsController()
    assert not c.hasResults and not c.hasFigure
    assert c.tracksLabel == "—"
    assert len(c.motionClasses) == 4


def test_results_controller_load(tmp_path):
    from firefly.ui.controllers.results_controller import ResultsController
    c = ResultsController()
    seen = []
    c.resultsChanged.connect(lambda: seen.append(1))
    c.load(_results_json(), str(tmp_path))
    assert c.hasResults and seen == [1]
    assert c.headerTitle == "2 groups"
    assert c.tracksLabel == "3,400"               # 1200+1300+900
    assert c.medianD == "0.100"                   # median(0.10,0.12,0.05)
    assert {ch["label"] for ch in c.groupChips} == {"control", "+noco"}
    ctrl = next(ch for ch in c.groupChips if ch["label"] == "control")
    assert ctrl["color"] == "#58a6ff" and ctrl["count"] == 2
    assert len(c.configRows) == 2
    cards = c.metricCards
    assert cards and cards[0]["key"] == "median_D"
    assert cards[0]["severity"] == "success" and cards[0]["hasDetails"]


def test_results_controller_figure_and_files(tmp_path):
    from firefly.ui.controllers.results_controller import ResultsController
    fig = tmp_path / "cmp_compare.png"
    # a tiny valid PNG
    from PySide6.QtGui import QImage
    QImage(4, 4, QImage.Format.Format_RGB888).save(str(fig))
    c = ResultsController()
    tok0 = c.figureToken
    c.load(_results_json(str(fig)), str(tmp_path))
    assert c.hasFigure and c.figure_path() == str(fig)
    assert c.figureToken == tok0 + 1
    assert any(f["kind"] == "figure" for f in c.outputFiles)


def test_results_controller_load_from_file(tmp_path):
    from firefly.ui.controllers.results_controller import ResultsController
    p = tmp_path / "cell_results.json"
    p.write_text(json.dumps(_results_json()))
    c = ResultsController()
    c.loadFromFile(str(p))
    assert c.hasResults and c.headerTitle == "2 groups"


def test_results_controller_bad_json_degrades(tmp_path):
    from firefly.ui.controllers.results_controller import ResultsController
    c = ResultsController()
    failed = []
    c.renderFailed.connect(lambda m: failed.append(m))
    c.loadFromFile(str(tmp_path / "nope.json"))
    assert failed
