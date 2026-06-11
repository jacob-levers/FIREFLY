"""Tests for the Compare loader (`load_summary_from_folder`) and its
long-path helpers.

Regression guard for the v2.64.4 fix: Compare read each cell's
`firefly_extras/<stem>_*.csv` files with normal Windows paths, so on a
>260-char output folder `os.path.isfile` returned False and every metric was
treated as missing -> empty panels.  These run in CI (no Qt — the loader only
needs the analysis stack).
"""
import os
import json
import pandas as pd

from firefly.analysis.fa_palmtracer import (
    load_summary_from_folder, _win_long_path, _win_disp_path)


# ── long-path helpers (pure string logic) ───────────────────────────────
def test_win_disp_path_strips_prefix():
    assert _win_disp_path(r"\\?\C:\x\y") == r"C:\x\y"
    assert _win_disp_path(r"\\?\UNC\srv\share\f") == r"\\srv\share\f"
    assert _win_disp_path(r"C:\plain\path") == r"C:\plain\path"   # untouched
    assert _win_disp_path("") == ""


def test_win_long_path_noop_off_windows():
    # Must NEVER rewrite POSIX paths (the prefix is Windows-only).
    if os.name != "nt":
        assert _win_long_path("/data/run/firefly_extras") == "/data/run/firefly_extras"
        assert _win_long_path("") == ""


def test_win_long_path_idempotent_on_windows():
    if os.name == "nt":
        once = _win_long_path(r"C:\a\b")
        assert once.startswith("\\\\?\\")
        assert _win_long_path(once) == once          # already-prefixed -> unchanged


# ── loader populates metrics from a firefly_extras folder ────────────────
def _make_run(tmp_path, stem="expt"):
    extras = tmp_path / "run" / "firefly_extras"
    extras.mkdir(parents=True)
    (extras / f"{stem}_params.json").write_text(
        json.dumps({"pixel_size_um": 0.1, "frame_interval_s": 0.05}))
    pd.DataFrame({"lag_frame": [1, 2, 3], "msd_um2": [0.1, 0.2, 0.3]}).to_csv(
        extras / f"{stem}_ensemble_msd.csv", index=False)
    pd.DataFrame({"particle": [1, 2], "D": [0.01, 0.5], "alpha": [0.9, 1.1]}).to_csv(
        extras / f"{stem}_diffusion_summary.csv", index=False)
    pd.DataFrame({"particle": [1, 1, 2], "frame": [0, 1, 0],
                  "x": [1.0, 1.1, 2.0], "y": [1.0, 1.0, 2.0]}).to_csv(
        extras / f"{stem}_trajectories.csv", index=False)
    return str(tmp_path / "run")


def test_load_summary_populates_core_metrics(tmp_path):
    s = load_summary_from_folder(_make_run(tmp_path))
    assert s["stem"] == "expt"
    assert s["params"]["frame_interval_s"] == 0.05
    assert s["ensemble_msd"] is not None and len(s["ensemble_msd"]) == 3
    assert s["diffusion"] is not None and len(s["diffusion"]) == 2
    assert s["tracks"] is not None and len(s["tracks"]) == 3
    # optional metrics absent -> None, not a crash
    assert s["jdd"] is None
    assert s["dwell_times"] is None
    assert s["turning_angles"] is None


def test_load_summary_accepts_extras_dir_directly(tmp_path):
    # passing the firefly_extras dir itself must also work
    run = _make_run(tmp_path)
    s = load_summary_from_folder(os.path.join(run, "firefly_extras"))
    assert s["diffusion"] is not None and len(s["diffusion"]) == 2
