"""The motion-class panel on a light figure.

The Light palette gave Immobile (#d1242f) and Confined (#bc4c00) — neighbours in
every stacked bar — almost the same colour: 7.8 OKLab ΔE for normal vision, 1.8
under deuteranopia, and all four at one lightness.  The legend sat inside the
axes, padding the y-axis to 1.42.  The panel's width only changed at 5 and 9
groups.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from firefly.analysis.fa_compare import compare_groups, panel_span
from firefly.analysis.fa_constants import MOTION_CLASS_ORDER, motion_class_colors

# Machado et al. 2009 (severity 1) colour-vision-deficiency matrices
CVD = {"normal": np.eye(3),
       "deutan": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                           [-0.011820, 0.042940, 0.968881]]),
       "protan": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                           [-0.003882, -0.048116, 1.051998]])}


def _lin(h):
    c = np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _oklab(lin):
    A = np.array([[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566],
                  [0.0883024619, 0.2817188376, 0.6299787005]])
    B = np.array([[0.2104542553, 0.7936177850, -0.0040720468], [1.9779984951, -2.4285922050, 0.4505937099],
                  [0.0259040371, 0.7827717662, -0.8086757660]])
    return B @ np.cbrt(A @ np.clip(lin, 0, 1))


@pytest.mark.parametrize("theme", ["Light", "Publication"])
@pytest.mark.parametrize("vision", list(CVD))
def test_neighbouring_classes_are_distinguishable_on_a_light_figure(vision, theme):
    """Publication is the colour-blind set (also the colour-blind option)."""
    pal = motion_class_colors(theme)
    seq = [pal[c] for c in MOTION_CLASS_ORDER]                 # stack order
    for a, b in zip(seq, seq[1:]):
        de = 100 * np.linalg.norm(_oklab(CVD[vision] @ _lin(a)) - _oklab(CVD[vision] @ _lin(b)))
        assert de >= 15, (vision, a, b, de)


def test_the_panel_widens_with_its_groups():
    assert panel_span("motion_classes", 3) == 4
    assert panel_span("motion_classes", 5) == 6
    assert panel_span("motion_classes", 8) == 12


def test_the_legend_sits_beside_the_bars_in_stack_order(tmp_path):
    from test_per_recording_curves import _uneven
    fig, _s, _st = compare_groups(_uneven(tmp_path), output_dir=None, pdf_report=False,
                                  panels={"motion_classes"}, theme="Light")
    ax = next(a for a in fig.axes if a.get_ylabel() == "Fraction of trajectories")
    assert ax.get_ylim()[1] <= 1.05                            # no headroom for a legend
    leg = ax.get_legend()
    fig.canvas.draw()
    bars_right = max(p.get_window_extent().x1 for p in ax.patches if p.get_height() > 0)
    assert leg.get_window_extent().x0 >= bars_right                  # beside, not over, the bars
    assert leg.get_window_extent().x1 <= ax.get_window_extent().x1 + 1  # and inside the panel
    assert [t.get_text() for t in leg.get_texts()] == list(reversed(MOTION_CLASS_ORDER))
    plt.close(fig)


def test_the_whole_comparison_figure_still_lays_out(tmp_path):
    """A key drawn outside its axes made tight_layout give up on the full
    figure: panels overlapped and a gap opened at the top."""
    import warnings
    from test_per_recording_curves import _uneven
    from firefly.ui.controllers.workspace import workspace_data as wd
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.filterwarnings("ignore", message=".*(Precision loss|range zero|divide|invalid value|Degrees of freedom|Mean of empty|All-NaN).*")
        fig, _s, _st = compare_groups(_uneven(tmp_path), output_dir=None, pdf_report=False,
                                      panels=set(wd.PANEL_KEYS), theme="Light")
    plt.close(fig)
