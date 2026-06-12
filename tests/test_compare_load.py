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
import pytest

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


# ── palmTRACER native MSD/D parsers (use_native mode) ────────────────────
def test_parse_pt_native_d(tmp_path):
    import numpy as np
    from firefly.analysis.fa_palmtracer import _parse_pt_native_d
    f = tmp_path / "trcPALMTracer-AllROI-D.txt"
    f.write_text(
        "#Diffusion Coef in um2/s; linear fit on first 4 pts\n"
        "#Pixel size= .106um ; Frame rate= .02sec\n"
        "ROI\tTrace\tD(um2/s)\tMSD(0)\tMSE\n"
        " 1\t 1\t .0157480045960291\t .00211001927606203\t .952396638411142\n"
        " 1\t 2\t 1E-05\t .00737197362608267\t .15236416156315\n")
    df = _parse_pt_native_d(str(f))
    assert df is not None and len(df) == 2
    r1 = df[df.particle == 1].iloc[0]
    assert abs(float(r1["D"]) - 0.0157480045960291) < 1e-12   # palmTRACER's own D
    assert abs(float(r1["logD"]) - float(np.log10(0.0157480045960291))) < 1e-6
    assert df["alpha"].isna().all()           # palmTRACER does not fit alpha
    assert (df["motion"] == "Unclassified").all()


def test_parse_pt_native_msd(tmp_path):
    import numpy as np
    from firefly.analysis.fa_palmtracer import _parse_pt_native_msd
    f = tmp_path / "trcPALMTracer-AllROI-MSD.txt"
    f.write_text(
        "#MSD(DeltaT) in um2\n"
        "#Pixel size= .106um ; Frame rate= .02sec\n"
        " 1\t 1\t .0037148\t .0041678\t .0057784\n"
        " 1\t 2\t .0081836\t .0036295\n")
    imsd, emsd = _parse_pt_native_msd(str(f), max_lagtime=20)
    assert imsd is not None and imsd.shape == (20, 2)
    assert list(imsd.index[:3]) == [1, 2, 3]
    assert abs(float(imsd[1].iloc[0]) - 0.0037148) < 1e-9    # palmTRACER's own MSD
    assert np.isnan(float(imsd[2].iloc[5]))                   # short track → NaN padded
    assert abs(float(emsd.iloc[0]) - (0.0037148 + 0.0081836) / 2) < 1e-9


def test_load_external_locs_rejects_corrupt_null_file(tmp_path):
    """A zero-filled loc file (interrupted copy / aborted acquisition) must fail
    FAST with a clear error — not hang pandas' sep=None python sniffer forever,
    which used to wedge the worker (and a HYPER-FLY slot) at "Reading
    localisations"."""
    from firefly.analysis.fa_loaders import load_external_locs
    p = tmp_path / "locPALMTracer.txt"
    p.write_bytes(b"\x00" * 200_000)          # full-size but all-NUL
    with pytest.raises(ValueError) as ei:
        load_external_locs(str(p), preset="auto")
    msg = str(ei.value).lower()
    assert "null" in msg or "corrupt" in msg


def test_load_external_locs_rejects_newlineless_file(tmp_path):
    """A large file with no line breaks at all (garbled / truncated) is rejected
    too, rather than spinning the sniffer."""
    from firefly.analysis.fa_loaders import load_external_locs
    p = tmp_path / "locPALMTracer.txt"
    p.write_bytes(b"x" * 200_000)             # no newline anywhere
    with pytest.raises(ValueError):
        load_external_locs(str(p), preset="auto")
