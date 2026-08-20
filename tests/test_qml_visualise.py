"""Phase-4 tests: VisualiseController / RoiController / EmbedController.

Controllers are exercised as plain QObjects (offscreen, no QML engine), plus a
real-widget embed geometry smoke.  Qt-gated like the other UI tests.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

import pytest                                            # noqa: E402

pytest.importorskip("PySide6")
pytest.importorskip("pandas")
from PySide6 import QtWidgets                            # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _make_run(tmp_path):
    """Build a minimal FIREFLY run folder: firefly_extras/{stem}_trajectories.csv
    + {stem}_diffusion_summary.csv with two motion classes."""
    import numpy as np, pandas as pd
    extras = tmp_path / "run1" / "firefly_extras"
    extras.mkdir(parents=True)
    rng = np.random.default_rng(0)
    rows = []
    motions = {}
    for pid in range(8):
        n = 12
        x = np.cumsum(rng.normal(0, 1.0, n)) + 50
        y = np.cumsum(rng.normal(0, 1.0, n)) + 50
        for f in range(n):
            rows.append((pid, f, float(x[f]), float(y[f])))
        motions[pid] = "Brownian" if pid % 2 == 0 else "Immobile"
    pd.DataFrame(rows, columns=["particle", "frame", "x", "y"]).to_csv(
        extras / "cell1_trajectories.csv", index=False)
    pd.DataFrame({"particle": list(motions),
                  "D": [0.05 * (p + 1) for p in motions],
                  "alpha": [0.9] * len(motions),
                  "motion": list(motions.values())}).to_csv(
        extras / "cell1_diffusion_summary.csv", index=False)
    return str(tmp_path / "run1")


# ── VisualiseController ──────────────────────────────────────────────────────
def test_visualise_idle_state():
    from firefly.ui.controllers.visualise_controller import VisualiseController
    c = VisualiseController()
    assert c.layers == [] and not c.hasRun
    assert c.currentFrame == 0 and not c.playing
    assert "Default" in c.motionColourModes


def test_visualise_load_run_builds_layer_model(tmp_path):
    from firefly.ui.controllers.visualise_controller import VisualiseController
    c = VisualiseController()
    assert c.loadRunFolder(_make_run(tmp_path)) is True
    assert c.hasRun
    names = {lyr["name"] for lyr in c.layers if lyr["kind"] == "tracks"}
    assert names == {"Brownian", "Immobile"}
    assert all(lyr["present"] and lyr["visible"] for lyr in c.layers
               if lyr["kind"] == "tracks")
    assert c.hudTrackCount == 8
    assert c.nFrames >= 12


def test_visualise_class_visibility_round_trip(tmp_path):
    from firefly.ui.controllers.visualise_controller import VisualiseController
    c = VisualiseController()
    c.loadRunFolder(_make_run(tmp_path))
    c.setLayerVisible("tracks:Brownian", False)
    row = next(l for l in c.layers if l["id"] == "tracks:Brownian")
    assert row["visible"] is False
    # the viewer was actually told to hide the class
    assert c._viewer._class_visible.get("Brownian") is False


def test_visualise_transport_and_knobs(tmp_path):
    from firefly.ui.controllers.visualise_controller import VisualiseController
    c = VisualiseController()
    c.loadRunFolder(_make_run(tmp_path))
    n = c.nFrames
    c.seek(3)
    assert c.currentFrame == 3 and c.frameLabel == f"frame 4 / {n}"
    c.stepForward()
    assert c.currentFrame == n - 1
    c.stepBack()
    assert c.currentFrame == 0
    c.tail = 50; c.head = 5; c.trackWidth = 3.0; c.fps = 12
    assert c.tail == 50 and c.head == 5 and c.trackWidth == 3.0 and c.fps == 12
    c.playPause(); assert c.playing is True
    c.playPause(); assert c.playing is False


def test_visualise_background_selection():
    import numpy as np
    from firefly.ui.controllers.visualise_controller import VisualiseController
    c = VisualiseController()
    c.ensureViewer().set_stack(np.zeros((6, 32, 32), dtype=np.float32))
    c._after_background_change()
    assert "Max projection" in c.backgroundOptions
    c.selectBackground("Max projection")
    assert c.backgroundMode == "Max projection"
    # background is the dropdown above the LAYERS list, NOT a track layer
    assert all(l["kind"] == "tracks" for l in c.layers)


def test_visualise_pick_to_inspector(tmp_path):
    from firefly.ui.controllers.visualise_controller import VisualiseController
    c = VisualiseController()
    c.loadRunFolder(_make_run(tmp_path))
    seen = []
    c.inspectorChanged.connect(lambda: seen.append(c.inspector["mode"]))
    c._viewer.trackClicked.emit(0)            # particle 0 = Brownian
    assert c.inspectorVisible
    insp = c.inspector
    assert insp["mode"] == "track" and insp["particle_id"] == 0
    assert insp["motion"] == "Brownian" and "d" in insp and "length" in insp
    assert seen == ["track"]
    c._viewer.clusterClicked.emit(-1)
    assert c.inspector["mode"] == "cluster" and "Noise point" in c.inspector["note"]


def _add_cluster_labels(run_dir):
    """Add a firefly_extras/cell1_cluster_labels.csv to an existing run."""
    import numpy as np, pandas as pd, os
    extras = os.path.join(run_dir, "firefly_extras")
    rng = np.random.default_rng(2)
    # two tight blobs + scatter (µm coords)
    a = rng.normal((1.0, 1.0), 0.05, (40, 2))
    b = rng.normal((3.0, 3.0), 0.05, (40, 2))
    noise = rng.uniform(0, 4, (20, 2))
    xy = np.vstack([a, b, noise])
    pd.DataFrame({"loc_index": np.arange(len(xy)),
                  "x_um": xy[:, 0], "y_um": xy[:, 1],
                  "cluster_id": [0] * 40 + [1] * 40 + [-1] * 20,
                  "motion": ["Brownian"] * 40 + ["Immobile"] * 40 + ["Unknown"] * 20}
                 ).to_csv(os.path.join(extras, "cell1_cluster_labels.csv"), index=False)


def test_visualise_clusters_load_recluster_suggest(tmp_path):
    pytest.importorskip("sklearn")
    from firefly.ui.controllers.visualise_controller import VisualiseController
    run = _make_run(tmp_path)
    _add_cluster_labels(run)
    c = VisualiseController()
    assert c.loadClustersFolder(run) is True
    assert c.hasClusters and c.clusterCount >= 2
    # cluster pick → inspector with dominant-motion note
    c._viewer.clusterClicked.emit(0)
    assert c.inspector["mode"] == "cluster" and c.inspector["cluster_id"] == 0
    # recluster with a tiny eps shatters the blobs into noise/fewer clusters
    c.clusterMinSamples = 5
    c.recluster()
    assert isinstance(c.clusterCount, int)
    c.suggestEps()
    assert 5 <= c.clusterEpsNm <= 2000
    # export tuned labels
    assert c.exportTunedClusters() is True
    import os
    assert os.path.isfile(os.path.join(run, "firefly_extras",
                                       "cell1_cluster_labels_tuned.csv"))


def test_visualise_cluster_map_standalone_no_siblings(tmp_path):
    """A cluster map with no sibling trajectories opens on its own: the viewer
    island surfaces (hasContent) and the scatter renders, with no run loaded
    (hasRun False) — colours fall back to per-cluster ID."""
    import numpy as np, pandas as pd
    from firefly.ui.controllers.visualise_controller import VisualiseController
    extras = tmp_path / "clusters_only" / "firefly_extras"
    extras.mkdir(parents=True)
    rng = np.random.default_rng(1)
    xy = np.vstack([rng.normal((1, 1), 0.05, (30, 2)),
                    rng.normal((3, 3), 0.05, (30, 2))])
    pd.DataFrame({"loc_index": np.arange(len(xy)),
                  "x_um": xy[:, 0], "y_um": xy[:, 1],
                  "cluster_id": [0] * 30 + [1] * 30}).to_csv(
        extras / "blob_cluster_labels.csv", index=False)
    c = VisualiseController()
    assert not c.hasContent
    assert c.loadClustersFolder(str(tmp_path / "clusters_only")) is True
    assert c.hasClusters and c.clusterCount >= 2
    assert c.hasContent                           # island appears…
    assert not c.hasRun                           # …no sibling tracks to load
    assert c._viewer._point_item is not None


def test_visualise_cluster_map_autoloads_motion(tmp_path):
    """Opening a cluster map on its own pulls in the sibling trajectories +
    diffusion as DATA ONLY (no track overlay) so the scatter colours by motion —
    even when the analysis 'motion' column is wholesale 'Unmatched' — without
    cluttering the view with track tails."""
    import os, numpy as np, pandas as pd
    from firefly.ui.controllers.visualise_controller import VisualiseController
    run = _make_run(tmp_path)
    extras = os.path.join(run, "firefly_extras")
    traj = pd.read_csv(os.path.join(extras, "cell1_trajectories.csv"))
    diff = pd.read_csv(os.path.join(extras, "cell1_diffusion_summary.csv"))
    pmotion = {int(p): m for p, m in zip(diff["particle"], diff["motion"])}
    # cluster locs sit on the track locs (px size 1.0 → µm == px), every motion
    # the worker sentinel — recovery must come from the data-only motion source.
    pd.DataFrame({
        "loc_index": np.arange(len(traj)),
        "x_um": traj["x"].to_numpy(), "y_um": traj["y"].to_numpy(),
        "cluster_id": traj["particle"].to_numpy().astype(int),
        "motion": ["Unmatched"] * len(traj),
    }).to_csv(os.path.join(extras, "cell1_cluster_labels.csv"), index=False)

    c = VisualiseController()
    assert c.loadClustersFolder(run) is True       # ONLY the cluster map
    assert c.hasContent and not c.hasRun           # no run / track overlay loaded
    assert not c._runs                             # motion is data-only…
    assert c._motion_src                           # …in the separate source
    assert not c._viewer._track_items              # NO track overlay items exist
    # motion recovered + each loc matches its particle's class
    assert c._has_real_motion()
    assert [str(m) for m in c._cl_motion] == [pmotion[int(p)] for p in traj["particle"]]
    # the scatter was painted with the motion palette, NOT the Unknown grey
    pal = c._palette()
    hexes = {col.name().lower() for col, *_rest in c._viewer._point_item._groups}
    assert pal["Brownian"].lower() in hexes
    assert pal["Immobile"].lower() in hexes
    assert pal["Unknown"].lower() not in hexes


def test_visualise_clusters_reuse_already_loaded_run(tmp_path):
    """If a run is already loaded, opening clusters reuses it — no duplicate
    auto-load — and recovers motion from it."""
    import os, numpy as np, pandas as pd
    from firefly.ui.controllers.visualise_controller import VisualiseController
    run = _make_run(tmp_path)
    extras = os.path.join(run, "firefly_extras")
    traj = pd.read_csv(os.path.join(extras, "cell1_trajectories.csv"))
    pd.DataFrame({
        "loc_index": np.arange(len(traj)),
        "x_um": traj["x"].to_numpy(), "y_um": traj["y"].to_numpy(),
        "cluster_id": traj["particle"].to_numpy().astype(int),
        "motion": ["Unmatched"] * len(traj),
    }).to_csv(os.path.join(extras, "cell1_cluster_labels.csv"), index=False)
    c = VisualiseController()
    assert c.loadRunFolder(run) is True
    assert len(c._runs) == 1
    assert c.loadClustersFolder(run) is True
    assert len(c._runs) == 1                        # did NOT auto-load a 2nd run
    assert c._has_real_motion()


def test_visualise_recluster_subsample_stays_aligned(tmp_path, monkeypatch):
    """recluster must realign every per-loc array to the coords compute_clusters
    returns — which are SUB-SAMPLED for a large eps.  Otherwise the overlay /
    pick / dominant-motion paths read past the end of the shorter labels array
    and crash (the eps=500 crash)."""
    import numpy as np, pandas as pd
    import firefly.sptpalm_analysis as sa
    from firefly.ui.controllers.visualise_controller import VisualiseController
    run = _make_run(tmp_path)
    _add_cluster_labels(run)                      # 100 cluster locs
    c = VisualiseController()
    assert c.loadClustersFolder(run) is True
    full_n = len(c._cl_xy_um_full)
    assert full_n == 100

    def fake_compute(locs, pixel_size_um=1.0, eps_um=0.05, min_samples=5, **kw):
        k = 40                                    # pretend a big eps subsampled
        xy = (locs[["x", "y"]].values[:k] * pixel_size_um).astype("float32")
        labels = np.array([0] * 20 + [1] * 20, dtype=int)
        stats = pd.DataFrame({"cluster_id": [0, 1]})
        stats.attrs["subsampled"] = True
        stats.attrs["n_used_locs"] = k
        stats.attrs["eps_too_large"] = False
        return labels, stats, None, xy
    monkeypatch.setattr(sa, "compute_clusters", fake_compute)

    c.clusterEpsNm = 500
    c.recluster()
    # all per-loc display arrays now share the subsampled length — no mismatch
    assert len(c._cl_labels) == len(c._cl_xy_um) == len(c._cl_xy_px) == 40
    assert len(c._cl_xy_um_full) == full_n        # full set preserved for next time
    # rendering in every colour mode must not raise (the crash repro)
    for mode in c.clusterColorModes:
        c.clusterColorMode = mode
        c._render_cluster_layer()
    assert "subsampled to 40" in c.clusterStatus


def test_visualise_broken_load_emits_warn(tmp_path):
    """A broken run folder / cluster map must surface a warn(title, message) —
    Main.qml wires it to an error toast, so the load no longer fails silently."""
    from firefly.ui.controllers.visualise_controller import VisualiseController
    c = VisualiseController()
    seen = []
    c.warn.connect(lambda t, m: seen.append((t, m)))
    assert c.loadRunFolder(str(tmp_path / "nope")) is False          # no firefly_extras
    assert c.loadClustersFolder(str(tmp_path / "nope")) is False
    titles = [t for t, _ in seen]
    assert "Load failed" in titles and "Couldn't load clusters" in titles
    assert all(m for _, m in seen)                                   # each carries a reason


def test_visualise_superres_render(tmp_path):
    import time
    from firefly.ui.controllers.visualise_controller import VisualiseController
    c = VisualiseController()
    c.loadRunFolder(_make_run(tmp_path))
    # With a stack/field loaded, the super-res canvas spans the full camera frame
    # (so it overlays the other backgrounds 1:1) instead of just the localisations'
    # bounding box — the fix for the render looking tiny next to the Max projection.
    c._field_px = (24, 48)                     # simulate a loaded 24×48 (H×W) frame
    c.srPixelNm = 100                          # coarse → tiny render (keep the test light)
    c.renderSuperres()
    assert c.srRendering                      # render kicked off on a worker thread
    # async: pump the event loop until the off-thread render drains in
    for _ in range(300):
        _app.processEvents()
        if c.hasSuperresRender:
            break
        time.sleep(0.01)
    assert c.hasSuperresRender and not c.srRendering
    assert "Super-resolution" in c.backgroundOptions
    # full-field canvas → image aspect matches the camera frame W/H (128/64 = 2),
    # placed at the origin (overlays the projection), not the data bbox
    img = c._sr_img
    assert abs(img.shape[1] / img.shape[0] - 128 / 64) < 0.1
    assert c.ensureViewer()._sr[2] == (0.0, 0.0)


def test_visualise_explorer_filters(tmp_path):
    from firefly.ui.controllers.visualise_controller import VisualiseController
    c = VisualiseController()
    c.loadRunFolder(_make_run(tmp_path))
    assert len(c.explorerRows) == 8
    cols = set(c.explorerRows[0].keys())
    assert {"particle", "length", "d", "alpha", "motion"} <= cols
    # filter to only Brownian → half the tracks
    c.setExpMotion("Immobile", False)
    c.refreshExplorer()
    assert all(r["motion"] == "Brownian" for r in c.explorerRows)
    assert len(c.explorerRows) == 4
    # min-length above the track length → empty
    c.expMinLen = 999
    c.refreshExplorer()
    assert c.explorerRows == []


def test_visualise_motion_colour_mode(tmp_path):
    from firefly.ui.controllers.visualise_controller import VisualiseController
    from firefly.analysis.fa_constants import motion_class_colors
    c = VisualiseController()
    c.loadRunFolder(_make_run(tmp_path))
    c.motionColourMode = "Colour-blind safe"
    assert c.motionColourMode == "Colour-blind safe"
    pub = motion_class_colors("Publication")
    row = next(l for l in c.layers if l["id"] == "tracks:Brownian")
    assert row["colorHex"] == pub["Brownian"]


# ── RoiController ────────────────────────────────────────────────────────────
def test_roi_draft_and_commit():
    from firefly.ui.controllers.roi_controller import RoiController
    r = RoiController()
    changed = []
    r.polygonsChanged.connect(lambda: changed.append(len(r.polygons)))
    assert not r.canClose
    r.addVertex(0, 0); r.addVertex(0, 10)
    assert r.draftLength == 2 and not r.canClose
    r.addVertex(10, 5)
    assert r.canClose
    assert r.closeDraft() is True
    assert r.polygonCount == 1 and r.draftLength == 0
    assert r.polygons[0] == [[0.0, 0.0], [0.0, 10.0], [10.0, 5.0]]
    assert changed[-1] == 1


def test_roi_delete_verbs_and_roundtrip():
    from firefly.ui.controllers.roi_controller import RoiController
    r = RoiController()
    r.setPolygons([[[0, 0], [0, 10], [10, 10], [10, 0]],
                   [[20, 20], [20, 30], [30, 25]]])
    assert r.polygonCount == 2
    r.deleteVertex(0, 0)                      # square → still ≥3 verts
    assert len(r.polygons[0]) == 3
    r.deleteVertex(1, 0)                      # triangle → 2 verts → drop polygon
    assert r.polygonCount == 1
    r.deletePolygon(0)
    assert r.polygonCount == 0
    r.setPolygons([[[1, 2], [3, 4], [5, 6]]])
    assert r.getPolygons() == [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]


# ── EmbedController ──────────────────────────────────────────────────────────
class _FakeIsland(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.geo_calls = 0
    def setGeometry(self, *a):
        self.geo_calls += 1
        super().setGeometry(*a)


def test_embed_geometry_sync_and_dedupe():
    from PySide6.QtCore import QRect
    from firefly.ui.controllers.embed_controller import EmbedController
    e = EmbedController()
    island = _FakeIsland()
    e.setIslands(viewer=island)
    e.showIsland("viewer")
    base = island.geo_calls
    rects = []
    e.anchorChanged.connect(lambda r: rects.append(r))
    e.setAnchorRect(40, 30, 500, 400)
    assert island.geometry() == QRect(40, 30, 500, 400)
    after = island.geo_calls
    assert after > base
    e.setAnchorRect(40, 30, 500, 400)         # identical → deduped, no setGeometry
    assert island.geo_calls == after
    assert len(rects) == 1                     # anchorChanged only on a real change


def test_embed_single_island_and_modal():
    from firefly.ui.controllers.embed_controller import EmbedController
    e = EmbedController()
    viewer, roi = _FakeIsland(), _FakeIsland()
    e.setIslands(viewer=viewer, roi=roi)
    e.setAnchorRect(0, 0, 100, 100)
    e.showIsland("viewer")
    assert e.activeIsland == "viewer"
    assert viewer.isVisible() is True and roi.isVisible() is False
    e.showIsland("roi")
    assert e.activeIsland == "roi"
    assert viewer.isVisible() is False and roi.isVisible() is True
    # modal hides the active island; closing restores it
    e.setModalOpen(True)
    assert roi.isVisible() is False
    e.setModalOpen(False)
    assert e.activeIsland == "roi"


def test_embed_location_changes_visibility():
    from firefly.ui.controllers.embed_controller import EmbedController, VISUALISE_TAB
    e = EmbedController()
    viewer = _FakeIsland()
    e.setIslands(viewer=viewer)
    e.setAnchorRect(0, 0, 80, 80)
    # On the Visualise tab but nothing loaded → island stays hidden (no blank card)
    e.onLocationChanged(VISUALISE_TAB, "main")
    assert viewer.isVisible() is False
    # Content arrives → island shows
    e.setViewerContent(True)
    assert e.activeIsland == "viewer"
    assert viewer.isVisible() is True
    # Leaving Visualise hides it even with content
    e.onLocationChanged(0, "main")
    assert viewer.isVisible() is False
    # Returning with content still loaded re-shows it
    e.onLocationChanged(VISUALISE_TAB, "main")
    assert viewer.isVisible() is True
    # Clearing content hides it again
    e.setViewerContent(False)
    assert viewer.isVisible() is False


def test_visualise_reads_are_encoding_tolerant(tmp_path):
    """A non-UTF-8 export (a stray ° / µ byte, 0xb0 / 0xb5 from palmTRACER or
    Excel) must not abort a load with 'utf-8 codec can't decode byte'."""
    import pandas as pd
    from firefly.ui.controllers.visualise_controller import _read_csv_enc, _load_json_enc

    csv = tmp_path / "t.csv"                       # header has a ° (0xb0), latin-1
    csv.write_bytes("particle,frame,x,y,temp_\xb0C\n0,0,1.0,2.0,37\n".encode("latin-1"))
    with pytest.raises(UnicodeDecodeError):        # strict utf-8 would blow up
        pd.read_csv(str(csv), encoding="utf-8")
    df = _read_csv_enc(str(csv))                   # the tolerant reader recovers
    assert list(df["particle"]) == [0] and df.shape == (1, 5)

    pj = tmp_path / "p_params.json"                # value has a µ (0xb5), latin-1
    pj.write_bytes('{"units": "\xb5m", "pixel_size": 0.106}'.encode("latin-1"))
    d = _load_json_enc(str(pj))
    assert d["pixel_size"] == 0.106


def test_load_json_enc_handles_bom_and_empty(tmp_path):
    # "Expecting value: line 1 column 1 (char 0)" came from a UTF-8 BOM at the
    # start of a params.json (or an empty file). The helper must strip the BOM
    # and raise a clean ValueError on empty content.
    import json
    from firefly.ui.controllers.visualise_controller import _load_json_enc
    bom = tmp_path / "bom_params.json"
    bom.write_bytes(b"\xef\xbb\xbf" + json.dumps({"pixel_size": 0.108}).encode())
    assert _load_json_enc(str(bom)) == {"pixel_size": 0.108}
    empty = tmp_path / "empty_params.json"
    empty.write_bytes(b"")
    with pytest.raises(ValueError):
        _load_json_enc(str(empty))


def test_visualise_load_run_survives_unreadable_params(tmp_path):
    # A run whose params.json is empty/corrupt must still load its tracks +
    # diffusion — the stem comes from the filename, so only the recorded stack
    # is lost (previously this aborted the whole load with a raw JSON error).
    from firefly.ui.controllers.visualise_controller import VisualiseController
    run = _make_run(tmp_path)
    # rename the stems' sidecars aren't needed; just drop an empty params.json in
    (tmp_path / "run1" / "firefly_extras" / "cell1_params.json").write_bytes(b"")
    c = VisualiseController()
    assert c.loadRunFolder(run) is True
    assert c.hasRun
    assert c.hudTrackCount == 8


def test_visualise_ignores_appledouble_sidecars(tmp_path):
    # On exFAT / SMB volumes macOS writes binary "._<name>" AppleDouble
    # siblings. They sort before the real file, so an unfiltered endswith scan
    # picks "._cell1_trajectories.csv" and fails with "CSV missing columns".
    # The loader must skip dotfiles and read the real sidecars.
    from firefly.ui.controllers.visualise_controller import (
        VisualiseController, _listdir_visible)
    run = _make_run(tmp_path)
    extras = tmp_path / "run1" / "firefly_extras"
    junk = b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X       \x00"
    for real in ("cell1_trajectories.csv", "cell1_diffusion_summary.csv"):
        (extras / f"._{real}").write_bytes(junk)
    (extras / "._cell1_params.json").write_bytes(junk)
    (extras / ".DS_Store").write_bytes(junk)
    # helper drops every dotfile
    assert not any(n.startswith(".") for n in _listdir_visible(str(extras)))
    c = VisualiseController()
    assert c.loadRunFolder(run) is True
    assert c.hudTrackCount == 8          # real trajectories, not the "._" junk


def test_read_csv_enc_strips_utf8_bom(tmp_path):
    # A trajectories.csv saved with a UTF-8 BOM would otherwise yield a first
    # column named "﻿particle", failing the required-column check.
    import pandas as pd
    from firefly.ui.controllers.visualise_controller import _read_csv_enc
    p = tmp_path / "bom_trajectories.csv"
    p.write_bytes(b"\xef\xbb\xbf" + b"particle,frame,x,y\n0,0,1.0,2.0\n")
    df = _read_csv_enc(str(p))
    assert list(df.columns) == ["particle", "frame", "x", "y"]
    assert len(df) == 1


def test_visualise_stack_loads_off_thread(monkeypatch):
    # A multi-GB .czi (e.g. 16000x256x256 over USB) must decode on a worker
    # thread so the window never freezes. loadStackPath returns immediately;
    # the viewer gets the movie via the GUI-thread drain timer.
    import time
    import numpy as np
    import firefly.sptpalm_analysis as core
    from firefly.ui.controllers.visualise_controller import VisualiseController
    stack = np.zeros((5, 16, 16), dtype=np.float32)
    monkeypatch.setattr(core, "load_file",
                        lambda path, channel=0, stop_event=None, dtype=None,
                        reserve_gb=None: (stack, 0.16, 0.32))
    c = VisualiseController()
    c.loadStackPath("/fake/Fly1-16k.czi")
    assert c._stack_loading is True          # returned without blocking
    assert c.movieLoading is True            # popup shows while decoding
    deadline = time.monotonic() + 5.0
    while c._stack_loading and time.monotonic() < deadline:
        _app.processEvents(); time.sleep(0.01)
    _app.processEvents()
    assert c._stack_loading is False         # worker + drain completed
    assert c._viewer is not None
    assert c.nFrames >= 5                    # movie reached the viewer


def test_visualise_clear_all_resets_to_idle(tmp_path):
    # The Clear button wipes every loaded run/movie/cluster back to empty.
    from firefly.ui.controllers.visualise_controller import VisualiseController
    c = VisualiseController()
    run = _make_run(tmp_path)
    c.loadRunFolder(run)
    assert c.hasRun and c.hudTrackCount == 8 and len(c.layers) > 0
    c.clearAll()
    assert c.hasRun is False
    assert c.hasContent is False
    assert c.hudTrackCount == 0
    assert c.layers == []
    assert c.nFrames == 0
    assert c.movieLoading is False
    # a fresh load after clearing still works
    c.loadRunFolder(run)
    assert c.hasRun and c.hudTrackCount == 8


def test_alloc_stack_honours_dtype_and_reserve_override():
    # The Visualiser loads movies as native uint16 with a small RAM reserve so a
    # 2 GB movie stays in RAM (fast bulk read) instead of a slow disk memmap.
    import numpy as np
    from firefly.analysis.fa_memory import _alloc_or_memmap_stack
    a = _alloc_or_memmap_stack((4, 8, 8), np.uint16, reserve_gb=0.5)
    assert a.dtype == np.uint16
    assert not isinstance(a, np.memmap)          # small + tiny reserve → in-RAM
    b = _alloc_or_memmap_stack((4, 8, 8))        # default float32
    assert b.dtype == np.float32


# ── cluster colour modes ────────────────────────────────────────────────────
def test_cluster_colour_modes_are_named_for_what_they_colour():
    """"Motion" was ambiguous next to "Cluster motion" — both are motion, they
    differ in whether the unit coloured is the localisation or the cluster."""
    from firefly.ui.controllers.visualise_controller import VisualiseController
    c = VisualiseController()
    assert c.clusterColorModes == ["Individual motion", "Cluster motion", "ID"]
    assert c.clusterColorMode == "Individual motion"      # the default


def test_cluster_info_note_describes_the_selected_mode():
    """The inspector reported "Dominant motion" whatever the mode, so in
    per-localisation mode it claimed one class for a cluster drawn as a mixture.
    """
    import numpy as np
    from firefly.ui.controllers.visualise_controller import VisualiseController
    c = VisualiseController()
    # cluster 0 holds a deliberate mixture: 3 Immobile, 2 Directed
    c._cl_labels = np.array([0, 0, 0, 0, 0])
    c._cl_motion = np.array(["Immobile", "Immobile", "Immobile",
                             "Directed", "Directed"])

    c._cl_color_mode = "Cluster motion"
    note = c._cluster_info(0)["note"]
    assert note.startswith("Dominant motion: Immobile"), note
    assert "60%" in note, note

    c._cl_color_mode = "Individual motion"
    note = c._cluster_info(0)["note"]
    assert note.startswith("Motion mix:"), note
    assert "Immobile 60%" in note and "Directed 40%" in note, note

    # ID colours by cluster, not motion — the mix stays informative, but it must
    # not assert a single dominant class either.
    c._cl_color_mode = "ID"
    assert c._cluster_info(0)["note"].startswith("Motion mix:")


# ── one-click cluster load for the run already open ─────────────────────────
def test_open_run_cluster_button_is_hidden_until_a_run_can_supply_clusters(tmp_path):
    from firefly.ui.controllers.visualise_controller import VisualiseController
    import os
    c = VisualiseController()
    assert c.openRunHasClusters is False          # nothing loaded at all
    assert c.openRunClusterName == ""

    run_dir = _make_run(tmp_path)
    c.loadTracksPath(os.path.join(run_dir, "firefly_extras",
                                  "cell1_trajectories.csv"), None)
    # tracks are open, but this run was analysed WITHOUT clustering
    assert c.openRunHasClusters is False

    _add_cluster_labels(run_dir)
    assert c.openRunHasClusters is True
    assert c.openRunClusterName == "run1"         # names the run on the button


def test_load_clusters_for_open_run_needs_no_directory_navigation(tmp_path):
    """The whole point: same run, no file dialog."""
    from firefly.ui.controllers.visualise_controller import VisualiseController
    import os
    run_dir = _make_run(tmp_path)
    _add_cluster_labels(run_dir)
    c = VisualiseController()
    c.loadTracksPath(os.path.join(run_dir, "firefly_extras",
                                  "cell1_trajectories.csv"), None)
    assert c.loadClustersForOpenRun() is True
    assert c.hasClusters
    assert c.clusterCount == 2                    # ids 0 and 1; -1 is noise


def test_browse_loading_still_works_after_the_one_click_load(tmp_path):
    """The new button must not replace the browse path — a second map still
    loads, so several cluster maps can be worked through in one session."""
    from firefly.ui.controllers.visualise_controller import VisualiseController
    import os
    run_a = _make_run(tmp_path)
    _add_cluster_labels(run_a)
    c = VisualiseController()
    c.loadTracksPath(os.path.join(run_a, "firefly_extras",
                                  "cell1_trajectories.csv"), None)
    assert c.loadClustersForOpenRun() is True

    other = tmp_path / "other"
    other.mkdir()
    run_b = _make_run(other)
    _add_cluster_labels(run_b)
    assert c.loadClustersFolder(run_b) is True    # the browse path, unchanged
    assert c.hasClusters


def test_a_run_outside_a_firefly_extras_folder_does_not_offer_the_button(tmp_path):
    """Tracks opened from a loose CSV have no run folder to take clusters from."""
    from firefly.ui.controllers.visualise_controller import VisualiseController
    import shutil, os
    run_dir = _make_run(tmp_path)
    loose = tmp_path / "loose_trajectories.csv"
    shutil.copy(os.path.join(run_dir, "firefly_extras", "cell1_trajectories.csv"),
                loose)
    c = VisualiseController()
    c.loadTracksPath(str(loose), None)
    assert c.openRunHasClusters is False
    assert c.loadClustersForOpenRun() is False    # reports, does not raise
