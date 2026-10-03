"""Figure titles and axis labels use the van Swinderen lab's wording.

Taken from the lab's sptPALM figures — Hines & van Swinderen 2021 eNeuro
(Figs 1–4), Hines et al. 2024 J Neurosci
(bioRxiv 2023.02.27.530184, Figs 3, 4, 8; MB543B and MB112C lines):
"MSD (µm²)" against "Time (s)"; "AUC (µm²s)"; "Log₁₀ diffusion coefficient";
"Relative frequency (fractions)"; "Slow/immobile fraction" | "Fast/mobile
fraction"; trajectories, not tracks; sentence case.
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from firefly.analysis.fa_compare import compare_groups

ROOT = os.path.join(os.path.dirname(__file__), "..", "firefly")
SOURCES = ["analysis/fa_compare.py", "analysis/fa_figure.py", "analysis/fa_group_figures.py",
           "ui/controllers/workspace/workspace_figures.py", "ui/controllers/workspace/workspace_data.py"]


def _labels(fig):
    out = []
    for ax in fig.axes:
        out += [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
    return [t for t in out if t]


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    from test_per_recording_curves import _uneven
    from firefly.ui.controllers.workspace import workspace_data as wd
    fig, _s, _st = compare_groups(_uneven(tmp_path_factory.mktemp("lab")), output_dir=None,
                                  panels=set(wd.PANEL_KEYS), pdf_report=False, theme="Light")
    yield _labels(fig)
    plt.close(fig)


@pytest.mark.parametrize("label", ["MSD (µm²)", "Time (s)", "AUC (µm²s)",
                                   "Log₁₀ diffusion coefficient (µm²/s)",
                                   "Relative frequency (fractions)", "Trajectories (n)",
                                   "Probability density (per log₁₀ unit)",
                                   "Trajectory length (s)"])
def test_the_report_uses_the_lab_label(report, label):
    assert label in report


def test_the_report_has_no_old_wording(report):
    old = ("Time delta", "Time lag", "lag time", "µm²·s", "log₁₀ D", "Log D", "Tracks",
           "Density",
           "Track length", "R_g", "Net ÷ path")
    assert not [t for t in report for o in old if o in t]


def test_labels_are_sentence_case(report):
    """A capital after the first word only for symbols and acronyms (MSD, AUC,
    VACF, α₂ …), never Title Case."""
    allowed = {"MSD", "AUC", "VACF", "MSS", "DBSCAN", "D", "X", "Y", "P(Δ)",
               "Gaussian", "Hove"}                     # names stay capitalised
    for t in report:
        words = re.findall(r"[A-Za-z][\w()/₁₀₂]*", t)[1:]
        caps = [w for w in words if w[0].isupper() and w not in allowed]
        assert not caps, (t, caps)


def test_no_figure_source_draws_the_old_labels():
    """The per-run figure and the live tab draw outside compare_groups."""
    old = re.compile(r'set_(?:x|y)label\("(?:lag time|Lag time|Time delta|Time lag|log10\(D\)|'
                     r'count"|density"|cumulative fraction)|"Max Projection"|"MSD Curves"|'
                     r'"Track Length"|"Total Tracks"|"µm²·s"|label\(?=?"[^"\n]*µm²·s')
    hits = []
    for rel in SOURCES:
        src = open(os.path.join(ROOT, rel), encoding="utf-8").read()
        hits += [f"{rel}: {m.group(0)}" for m in old.finditer(src)]
    assert not hits, hits
