"""vbSPT-style diffusive states (fa_states) on trajectories with a known answer.

The per-trajectory α classes call a simulated Brownian trajectory "Brownian" 5%
of the time at fly lengths (~10 points).  A hidden Markov model fitted to all of
a recording's steps recovers states, occupancies and switching rates from the
same short trajectories — the reason sptPALM work uses vbSPT.
"""
import numpy as np
import pandas as pd
import pytest

from firefly.analysis.fa_states import (STATE_KEYS, diffusive_states, fit_hmm, state_summary,
                                        step_segments)

DT, PX = 0.02, 0.1
TRUE_D = np.array([0.01, 0.1, 0.5])                  # µm²/s: immobile / slow / fast
TRUE_A = np.array([[0.85, 0.10, 0.05],
                   [0.08, 0.84, 0.08],
                   [0.04, 0.11, 0.85]])


def _stationary(A):
    w, v = np.linalg.eig(A.T)
    p = np.real(v[:, np.argmin(abs(w - 1))])
    return p / p.sum()


def _simulate(n_tracks=3000, seed=0, sigma=0.0, mean_extra=7, D=TRUE_D, A=TRUE_A):
    """Tracks (px) whose state follows the Markov chain A; ≥ 3 points, median ≈ 10."""
    rng = np.random.default_rng(seed)
    pi = _stationary(A)
    rows, steps_by_state = [], np.zeros(len(D))
    for pid in range(n_tracks):
        n = 3 + rng.geometric(1 / mean_extra)
        s = rng.choice(len(D), p=pi)
        pos = np.zeros(2); xy = [pos.copy()]
        for _ in range(n - 1):
            pos = pos + rng.normal(0, np.sqrt(2 * D[s] * DT), 2)
            xy.append(pos.copy()); steps_by_state[s] += 1
            s = rng.choice(len(D), p=A[s])
        xy = np.array(xy) + rng.normal(0, sigma, (n, 2))
        rows += [(pid, f, x / PX, y / PX) for f, (x, y) in enumerate(xy)]
    return pd.DataFrame(rows, columns=["particle", "frame", "x", "y"]), steps_by_state / steps_by_state.sum()


@pytest.fixture(scope="module")
def short_tracks():
    return _simulate()


def test_steps_are_consecutive_frames_only():
    tr = pd.DataFrame({"particle": [1, 1, 1, 1, 2, 2], "frame": [0, 1, 3, 4, 0, 1],
                       "x": [0, 1, 1, 2, 5, 5], "y": [0, 0, 0, 0, 0, 3]})
    segs = step_segments(tr, 0.1)
    assert [list(np.round(s, 6)) for s in segs] == [[0.01], [0.01], [0.09]]   # the 1→3 gap is skipped


def test_three_states_are_recovered_from_short_trajectories(short_tracks):
    tracks, occ_true = short_tracks
    fit = fit_hmm(step_segments(tracks, PX), DT, 3)
    assert np.allclose(fit["D"], TRUE_D, rtol=0.12)
    assert np.allclose(fit["occupancy"], occ_true, atol=0.04)
    assert np.allclose(np.diag(fit["transition"]), np.diag(TRUE_A), atol=0.06)


def test_d_is_apparent_as_in_vbspt():
    """Localisation error is not removed: D_app ≈ D + σ²/Δt."""
    sigma = 0.02
    tracks, _ = _simulate(n_tracks=2500, seed=1, sigma=sigma)
    fit = fit_hmm(step_segments(tracks, PX), DT, 3)
    assert fit["D"][2] == pytest.approx(TRUE_D[2] + sigma ** 2 / DT, rel=0.15)


def test_bic_reports_the_number_of_states_the_data_hold():
    D2 = np.array([0.01, 0.3]); A2 = np.array([[0.9, 0.1], [0.1, 0.9]])
    tracks, _ = _simulate(n_tracks=2500, seed=2, D=D2, A=A2)
    res = diffusive_states(tracks, PX, DT, k_max=4)
    assert res["best_n_states"] == 2
    assert res["n_states"] == 3                          # the analysis is still three states


def test_the_summary_and_the_empty_case():
    tracks, _ = _simulate(n_tracks=400, seed=3)
    s = state_summary(diffusive_states(tracks, PX, DT, k_max=3))
    assert set(s) == ({f"state_{w}_{k}" for w in ("occupancy", "D") for k in STATE_KEYS} | {"state_best_n"}
                      | {f"state_P_{a}_{b}" for a in STATE_KEYS for b in STATE_KEYS})
    assert sum(s[f"state_occupancy_{k}"] for k in STATE_KEYS) == pytest.approx(1.0)
    assert s["state_D_immobile"] < s["state_D_slow"] < s["state_D_fast"]
    few = tracks[tracks.particle < 5]
    assert diffusive_states(few, PX, DT) is None
    assert all(np.isnan(v) for v in state_summary(None).values())


def test_a_very_long_trajectory_does_not_blow_up_memory():
    """Padding to the longest trajectory took 7 GB on a real recording with a
    5,000-frame immobile trajectory; pieces keep the padded array small."""
    from firefly.analysis import fa_states
    long_one = [np.full(5000, 1e-4)]
    short = [np.full(9, 4 * 0.1 * DT)] * 2000
    pieces = fa_states._pieces(long_one + short)
    R, _l, _a = fa_states._padded(pieces)
    assert R.shape[1] == fa_states.MAX_PIECE_STEPS
    assert R.size < 300_000
    fit = fit_hmm(long_one + short, DT, 2)
    assert fit["n_segments"] == 2001 and fit["n_steps"] == 5000 + 9 * 2000


# ── the comparison report ───────────────────────────────────────────────────
def _state_groups(tmp_path):
    """Two groups whose recordings differ in immobile-state occupancy."""
    import os
    from test_workspace_data import make_run_folder
    out = []
    for gi, stay in enumerate([0.95, 0.80]):          # group B leaves the immobile state sooner
        A = np.array([[stay, (1 - stay) / 2, (1 - stay) / 2], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        folders = []
        for k in range(3):
            stem = f"s{gi}_{k}"
            f = make_run_folder(str(tmp_path), stem, seed=50 + 10 * gi + k)
            tr, _occ = _simulate(n_tracks=600, seed=70 + 10 * gi + k, A=A)
            tr.to_csv(os.path.join(f, "firefly_extras", f"{stem}_trajectories.csv"), index=False)
            with open(os.path.join(f, "firefly_extras", f"{stem}_params.json"), "w") as fh:
                import json
                json.dump({"pixel_size_um": PX, "frame_interval_s": DT}, fh)
            folders.append(f)
        out.append({"label": "AB"[gi], "color": ["#0072b2", "#d55e00"][gi], "folders": folders})
    return out


@pytest.fixture(scope="module")
def state_report(tmp_path_factory):
    from firefly.analysis.fa_compare import compare_groups
    import matplotlib.pyplot as plt
    groups = _state_groups(tmp_path_factory.mktemp("states"))
    fig, sdf, stats = compare_groups(groups, output_dir=None, pdf_report=False,
                                     panels={"states_occupancy", "states_d"})
    yield groups, fig, sdf, stats
    plt.close(fig)


def test_each_recording_gets_its_three_states(state_report):
    groups, _fig, sdf, _st = state_report
    import os
    f = groups[0]["folders"][0]
    tr = pd.read_csv(os.path.join(f, "firefly_extras", f"{os.path.basename(f)}_trajectories.csv"))
    want = diffusive_states(tr, PX, DT, k_max=None)
    row = sdf[sdf["stem"] == os.path.basename(f)].iloc[0]
    assert row["state_occupancy_immobile"] == pytest.approx(want["occupancy"][0])
    assert row["state_D_fast"] == pytest.approx(want["D"][2])
    a = sdf.loc[sdf.group == "A", "state_occupancy_immobile"].mean()
    b = sdf.loc[sdf.group == "B", "state_occupancy_immobile"].mean()
    assert a > b + 0.1                                    # the difference that was simulated


def test_the_panels_show_occupancy_and_d_per_state(state_report):
    _g, fig, _sdf, stats = state_report
    by_ylabel = {a.get_ylabel(): a for a in fig.axes if a.get_ylabel()}
    occ = by_ylabel["State occupation (%)"]
    assert [t.get_text() for t in occ.get_xticklabels()] == ["Immobile", "Slow mobile", "Fast mobile"]
    assert "Apparent diffusion coefficient (µm²/s)" in by_ylabel
    for what in ("occupancy", "D"):
        for k in STATE_KEYS:
            assert f"state_{what}_{k}" in stats
    assert stats["state_occupancy_immobile"]["pairwise"][0]["p"] < 0.05
    assert [t.get_text() for t in occ.get_legend().get_texts()] == ["A", "B"]


def test_a_run_s_saved_states_are_used_instead_of_refitting(tmp_path):
    import json, os
    from firefly.analysis.fa_compare import compare_groups
    import matplotlib.pyplot as plt
    groups = _state_groups(tmp_path)
    f = groups[0]["folders"][0]; stem = os.path.basename(f)
    saved = {"n_states": 3, "D": [0.001, 0.002, 0.003], "occupancy": [0.5, 0.3, 0.2],
             "transition": np.eye(3).tolist(), "best_n_states": 3}
    with open(os.path.join(f, "firefly_extras", f"{stem}_diffusive_states.json"), "w") as fh:
        json.dump(saved, fh)
    fig, sdf, _st = compare_groups(groups, output_dir=None, pdf_report=False, panels={"states_d"})
    row = sdf[sdf["stem"] == stem].iloc[0]
    assert row["state_occupancy_immobile"] == 0.5 and row["state_D_fast"] == 0.003
    plt.close(fig)


def test_minimal_mode_marks_the_states_with_stars(state_report, tmp_path):
    from firefly.analysis.fa_compare import compare_groups
    import matplotlib.pyplot as plt
    groups, *_ = state_report
    fig, _s, _st = compare_groups(groups, output_dir=None, pdf_report=False, minimal=True,
                                  panels={"states_occupancy"})
    ax = next(a for a in fig.axes if a.get_ylabel() == "State occupation (%)")
    marks = [t.get_text() for t in ax.texts if t.get_text().strip()]
    assert len(marks) == 3 and set(marks) <= {"*", "**", "***", "n.s."}
    plt.close(fig)


def test_across_metric_correction_covers_the_states(state_report):
    """The six state measures are tested and starred like every other scalar,
    so across-metric correction must include them — and the state panel's marks
    must show the corrected value, as the other bar panels' do."""
    from firefly.analysis.fa_compare import compare_groups
    import matplotlib.pyplot as plt
    groups, *_ = state_report
    cfg = {"correction": "fdr_bh", "across_metric_correction": True}
    fig, _s, stats = compare_groups(groups, output_dir=None, pdf_report=False, minimal=True,
                                    panels={"states_occupancy", "states_d", "auc"},
                                    stats_config=cfg)
    for what in ("occupancy", "D"):
        for k in STATE_KEYS:
            pw = stats[f"state_{what}_{k}"]["pairwise"][0]
            assert np.isfinite(pw["p_across"]) and pw["p_across"] >= pw["p"]
    ax = next(a for a in fig.axes if a.get_ylabel() == "State occupation (%)")
    marks = [t.get_text() for t in ax.texts if t.get_text().strip()]
    from firefly.analysis.fa_compare import significance_label
    want = [significance_label(stats[f"state_occupancy_{k}"]["pairwise"][0]["p_across"], "stars", 0.05)
            for k in STATE_KEYS]
    assert marks == want
    plt.close(fig)


def test_each_run_saves_its_states_for_the_report(tmp_path):
    import json
    from firefly.firefly_worker import _save_diffusive_states
    from firefly.analysis.fa_states import load_saved_states
    tracks, _ = _simulate(n_tracks=500, seed=9)
    summary, said = {}, []
    res = _save_diffusive_states(tracks, PX, DT, str(tmp_path), "rec", summary, said.append)
    saved = load_saved_states(str(tmp_path), "rec")
    assert saved["occupancy"] == pytest.approx(res["occupancy"])
    assert json.loads(json.dumps(saved))["n_states"] == 3
    assert summary["state_occupancy_immobile"] == pytest.approx(res["occupancy"][0])
    assert said and "Diffusive states" in said[0]
    assert _save_diffusive_states(None, PX, DT, str(tmp_path), "x", {}, said.append) is None
