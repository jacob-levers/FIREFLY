"""Group-level comparison renderers (experimental-design report / Analysis tab).

Pure matplotlib renderers, so no Qt — just assert they produce a figure of the
right shape for each style, across group counts, without error.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from firefly.analysis import fa_group_figures as gf


def _synth(groups, tps=("T1", "T2"), n_lags=10, seed=0):
    rng = np.random.default_rng(seed)
    lags = np.arange(1, n_lags + 1) * 0.02
    data = {g: {tp: np.abs(rng.normal(0.1, 0.03, (int(rng.integers(4, 8)), n_lags)))
                for tp in tps} for g in groups}
    return data, lags


def test_dispersion_sd_sem_ci_ordering():
    arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    sd = gf.dispersion(arr, "SD"); sem = gf.dispersion(arr, "SEM"); ci = gf.dispersion(arr, "95% CI")
    assert np.all(sem < sd) and np.all(ci > sem)          # SEM < SD, CI = 1.96·SEM > SEM
    # single dish → zero spread, never NaN
    assert np.all(gf.dispersion(arr[:1], "SEM") == 0)


def test_facet_grid_is_near_square():
    assert gf.facet_grid(1) == (1, 1)
    assert gf.facet_grid(2) == (1, 2)
    assert gf.facet_grid(3) == (1, 3)
    assert gf.facet_grid(4) == (2, 2)
    assert gf.facet_grid(5) == (2, 3)


@pytest.mark.parametrize("style", ["mean_faceted", "individual", "overlaid"])
def test_render_msd_all_styles(style):
    groups = ["BaCl", "Control", "KCl", "No stim"]
    data, lags = _synth(groups)
    fig = gf.render_msd(groups, data, lags, style=style, err="SEM")
    assert fig is not None and len(fig.axes) >= 1
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_render_msd_adapts_to_group_count():
    for n in (1, 2, 3, 5):
        groups = [f"G{i}" for i in range(n)]
        data, lags = _synth(groups, seed=n)
        fig = gf.render_msd(groups, data, lags, style="mean_faceted")
        # faceted styles use a rows×cols grid that covers every group
        assert len(fig.axes) >= n
        import matplotlib.pyplot as plt
        plt.close(fig)


def test_render_msd_single_timepoint_draws_one_series():
    groups = ["A", "B"]
    data, lags = _synth(groups, tps=("only",))
    fig = gf.render_msd(groups, data, lags, style="mean_faceted", tp_order=["only"])
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def _fig_ss():
    import matplotlib.pyplot as plt
    fig = plt.figure()
    return fig, fig.add_gridspec(1, 1)[0]


def _scalars(groups, tps=("T1", "T2"), seed=0):
    rng = np.random.default_rng(seed)
    return {g: {tp: np.abs(rng.normal(10 + i * 3, 4, int(rng.integers(4, 8))))
                for tp in tps} for i, g in enumerate(groups)}


@pytest.mark.parametrize("style", ["box_points", "grouped", "violin", "bar"])
def test_group_comparison_styles(style):
    import matplotlib.pyplot as plt
    groups = ["BaCl", "Control", "KCl"]
    fig, ss = _fig_ss()
    ax = gf.draw_group_comparison(fig, ss, groups, _scalars(groups), style=style,
                                  stat_label="Kruskal–Wallis, p = 0.013", ylabel="tracks")
    assert ax is not None
    plt.close(fig)


def test_length_density_renders():
    import matplotlib.pyplot as plt
    groups = ["A", "B"]
    rng = np.random.default_rng(3)
    dists = {g: rng.exponential(6, 500) + 8 for g in groups}
    fig, ss = _fig_ss()
    ax = gf.draw_length_density(fig, ss, groups, dists, threshold=8)
    assert ax is not None
    plt.close(fig)


@pytest.mark.parametrize("style", ["paired", "delta"])
def test_auc_change_styles(style):
    import matplotlib.pyplot as plt
    groups = ["BaCl", "Control", "KCl", "No stim"]
    rng = np.random.default_rng(4)
    paired = {}
    for g in groups:
        n = int(rng.integers(4, 7)); a = np.abs(rng.normal(120, 30, n))
        paired[g] = {"pre": a, "post": a * rng.uniform(0.9, 1.4)}
    fig, ss = _fig_ss()
    gf.draw_auc_change(fig, ss, groups, paired, style=style, tp_order=["pre", "post"],
                       stat_labels=({g: "p = 0.2" for g in groups} if style == "paired"
                                    else "Kruskal–Wallis, p = 0.4"))
    assert len(fig.axes) >= 1
    plt.close(fig)
