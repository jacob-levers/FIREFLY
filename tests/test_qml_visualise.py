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
    # exposed as a layer-rail image layer
    assert any(l["id"] == "bg:Max projection" and l["visible"] for l in c.layers)


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


def test_visualise_superres_render(tmp_path):
    from firefly.ui.controllers.visualise_controller import VisualiseController
    c = VisualiseController()
    c.loadRunFolder(_make_run(tmp_path))
    c.srPixelNm = 30
    c.renderSuperres()
    assert c.hasSuperresRender
    assert "Super-resolution" in c.backgroundOptions


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
    e.onLocationChanged(VISUALISE_TAB, "main")
    assert e.activeIsland == "viewer"
    e.onLocationChanged(0, "main")            # left Visualise → hidden
    assert viewer.isVisible() is False
