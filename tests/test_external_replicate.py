"""External localisation files → on-the-fly Analysis-tab replicates.

The Analysis tab compares FIREFLY run folders.  A raw localisation table from
another tool (palmTRACER / ThunderSTORM / …) is analysed on add — the SAME
pipeline the Process tab runs — into a cached run folder that then loads as an
ordinary replicate.  This covers the detector and the analyse-once engine
end to end on a small synthetic palmTRACER-style loc table.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("trackpy")
import pandas as pd                                     # noqa: E402

from firefly.ui.controllers.workspace import workspace_data as wd


class FakeSettings:
    """Returns each getter's default — so build_params yields its own defaults
    (the production analysis settings)."""
    def get_str(self, k, d=""): return d
    def get_float(self, k, d=0.0): return d
    def get_bool(self, k, d=False): return d
    def sync(self): pass


def _palmtracer_loc_csv(path, *, n_frames=40, n_spots=12, seed=1, step=0.5):
    """A palmTRACER-style loc table: diffusing spots, 1-indexed `Plane`,
    pixel-unit centroids, an intensity column — the PALM-Tracer preset's
    recognised names, so auto-detect maps it without an explicit preset."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(20, 80, size=(n_spots, 2))        # (x, y) in px
    rows = []
    for f in range(1, n_frames + 1):                    # palmTRACER is 1-indexed
        for s in range(n_spots):
            rows.append((f, pos[s, 0], pos[s, 1], 900.0 + rng.normal(0, 20)))
        pos += rng.normal(0.0, step, pos.shape)
        pos = np.clip(pos, 6, 94)
    df = pd.DataFrame(rows, columns=["Plane", "CentroidX(px)",
                                     "CentroidY(px)", "Integrated_Intensity"])
    df.to_csv(str(path), index=False)
    return str(path)


# ── detector ─────────────────────────────────────────────────────────────────
def test_is_external_loc_file(tmp_path):
    f = tmp_path / "cell_locPALMTracer.csv"
    f.write_text("Plane,CentroidX(px),CentroidY(px)\n1,1,1\n")
    assert wd.is_external_loc_file(str(f))
    assert wd.is_external_loc_file(str(tmp_path / "x.txt")) is False   # missing
    (tmp_path / "x.txt").write_text("a\tb\n1\t2\n")
    assert wd.is_external_loc_file(str(tmp_path / "x.txt"))
    # a run FOLDER (directory) is not an external-loc file
    d = tmp_path / "run"; d.mkdir()
    assert wd.is_external_loc_file(str(d)) is False
    # unsupported extension
    assert wd.is_external_loc_file(str(tmp_path / "movie.czi")) is False


# ── engine: analyse a loc table into a loadable replicate ────────────────────
@pytest.mark.slow
def test_analyse_external_file_produces_a_replicate(tmp_path):
    loc = _palmtracer_loc_csv(tmp_path / "cell_locPALMTracer.csv")
    cache = str(tmp_path / "cache")

    run_dir = wd.analyse_external_file(loc, FakeSettings(), cache_root=cache)
    assert run_dir and os.path.isdir(run_dir), "no run folder produced"
    assert wd.is_run_folder(run_dir), "output isn't a recognised run folder"

    # …and it loads as an ordinary replicate carrying diffusion metrics.
    run = wd.load_run(run_dir)
    assert run is not None
    diff = run.diff()
    assert diff is not None and len(diff) > 0, "no per-track diffusion table"
    assert {"D", "alpha"}.issubset(diff.columns)


@pytest.mark.slow
def test_analyse_external_file_reuses_the_cache(tmp_path):
    loc = _palmtracer_loc_csv(tmp_path / "cell_locPALMTracer.csv")
    cache = str(tmp_path / "cache")

    first = wd.analyse_external_file(loc, FakeSettings(), cache_root=cache)
    assert first is not None
    marker = os.path.join(first, "firefly_extras", "REUSE_MARKER")
    open(marker, "w").close()                           # prove no re-analysis

    logs = []
    second = wd.analyse_external_file(loc, FakeSettings(), cache_root=cache,
                                      log=logs.append)
    assert second == first
    assert os.path.isfile(marker), "cache folder was rebuilt (should reuse)"
    assert any("Reusing cached" in m for m in logs)


# ── controller: a dropped loc file becomes an analysing chip, then a replicate ─
def test_controller_stages_loc_file_then_loads_replicate(tmp_path, monkeypatch):
    """Dropping a localisation file onto a condition shows an *analysing* chip
    immediately, then resolves to a real replicate once analysed — without
    running the heavy pipeline (that's covered above).  Analysis is stubbed to
    produce a ready-made run folder."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from firefly.ui.controllers.workspace.workspace_controller import (
        AnalysisWorkspaceController)
    from test_workspace_data import make_run_folder

    app = QApplication.instance() or QApplication([])

    # Stub the analysis: return a prebuilt run folder instead of computing one.
    def _fake_analyse(path, settings, *, cache_root, log=None, cancel=None):
        return make_run_folder(str(tmp_path / "runs"),
                               os.path.splitext(os.path.basename(path))[0],
                               seed=7)
    monkeypatch.setattr(wd, "analyse_external_file", _fake_analyse)

    loc = _palmtracer_loc_csv(tmp_path / "cellA_locPALMTracer.csv", n_frames=5)
    c = AnalysisWorkspaceController(settings=FakeSettings())
    cid = c.conditions[0]["id"]
    c.addFolders(cid, [loc])

    # synchronous readout: one *analysing* chip, counted as loading
    fol = c.conditions[0]["folders"]
    assert len(fol) == 1
    assert fol[0]["analysing"] is True and fol[0]["loading"] is True
    assert fol[0]["n"] == "analysing…"
    assert c.loadingFolders is True
    assert c.conditions[0]["activeFolders"] == 0

    # pump until the background analyse+load resolves
    import time
    deadline = time.monotonic() + 10.0
    while c.loadingFolders and time.monotonic() < deadline:
        app.processEvents(); time.sleep(0.01)
    app.processEvents()

    fol = c.conditions[0]["folders"]
    assert fol[0]["analysing"] is False and fol[0]["loading"] is False
    assert fol[0]["qc"] in ("ok", "warn")
    assert c.conditions[0]["activeFolders"] == 1
