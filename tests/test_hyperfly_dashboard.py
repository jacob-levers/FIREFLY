"""HYPER-FLY dashboard + Batch queue bookkeeping during a parallel run.

* A throttled progress tile can arrive after its file finished — even after
  another file took over that tile.  It used to claim a free tile of its own: a
  ghost "running" tile for a finished file, held until the run ended.
* A tile's max-projection fallback must not re-read the file over the network
  while the worker is still queued for, or busy with, its own gated load.
* HYPER-FLY sends no FILE_STARTING, and the Batch queue only advanced on that,
  so every series row stayed "queued" for the whole run.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

import pytest                                            # noqa: E402

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                            # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def hf():
    from firefly.ui.controllers.hyperfly_controller import HyperflyController
    h = HyperflyController()
    yield h
    h._timer.stop()
    h._proj_timer.stop()
    h.deleteLater()
    _app.processEvents()


def _tiles(h):
    return [(d["file"], d["state"]) for d in h._slots]


def test_a_late_progress_tile_does_not_resurrect_a_finished_file(hf):
    hf.start(4, {})
    hf.onStatus({"active": True, "n_concurrent": 2})
    hf.onTile({"file": 1, "stem": "A", "state": "running"})
    hf.onTile({"file": 2, "stem": "B", "state": "running"})
    hf.onTile({"file": 1, "stem": "A", "state": "done"})
    hf.onTile({"file": 3, "stem": "C", "state": "running"})    # takes A's tile
    hf.onTile({"file": 2, "stem": "B", "state": "done"})
    hf.onTile({"file": 1, "stem": "A", "pct": 97, "stage": "Writing figures"})
    assert (1, "running") not in _tiles(hf)
    hf.onTile({"file": 4, "stem": "D", "state": "running"})
    assert _tiles(hf) == [(3, "running"), (4, "running")]
    assert hf.runningCount == 2


def test_a_stopped_file_frees_its_tile_and_a_rerun_reclaims_one(hf):
    hf.start(2, {})
    hf.onStatus({"active": True, "n_concurrent": 1})
    hf.onTile({"file": 1, "stem": "A", "state": "running"})
    hf.onTile({"file": 1, "stem": "A", "state": "cancelled"})
    assert _tiles(hf) == [(1, "idle")]
    hf.onTile({"file": 1, "stem": "A", "pct": 40, "stage": "Linking"})   # stale
    assert _tiles(hf) == [(1, "idle")]
    hf.onTile({"file": 1, "stem": "A", "state": "running"})              # re-run
    assert _tiles(hf) == [(1, "running")]


def test_no_projection_read_until_the_worker_has_loaded_the_file(hf, tmp_path):
    hf.start(1, {"A": str(tmp_path / "A.tif")})
    hf.onStatus({"active": True, "n_concurrent": 1})
    hf.onTile({"file": 1, "stem": "A", "state": "running"})
    assert not hf._proj_inflight, "read before the worker even started loading"
    for stage in ("Waiting for a load slot…", "Loading stack…"):
        hf.onTile({"file": 1, "stem": "A", "pct": 5, "stage": stage})
        assert not hf._proj_inflight, stage
    hf.onTile({"file": 1, "stem": "A", "pct": 30, "stage": "Localising"})
    assert "A" in hf._proj_inflight


# ── Batch queue rows ─────────────────────────────────────────────────────────
class FakeSettings:
    def get_str(self, k, d=""): return d
    def get_float(self, k, d=0.0): return d
    def get_bool(self, k, d=False): return d
    def set(self, k, v): pass
    def sync(self): pass


class FakeImport:
    hasFile = False
    filePath = ""; outDir = ""; isCsv = False


def test_batch_rows_follow_a_parallel_run():
    from firefly.ui.controllers.batch_controller import BatchController
    c = BatchController(FakeSettings(), FakeImport())
    c._series = [{"key": k, "name": k} for k in ("s1", "s2", "s3")]
    c._index_keys = ["s1", "s2", "s3"]
    c._run_order = ["s1", "s2", "s3"]
    c._series_status = {k: "queued" for k in c._run_order}
    c._files_total = 3

    c._on_hf_tile({"file": 2, "stem": "s2", "state": "running"})
    c._on_hf_tile({"file": 3, "stem": "s3", "state": "running"})
    assert c._series_status == {"s1": "queued", "s2": "running", "s3": "running"}
    c._on_file_done({"index": 3, "total": 3})
    c._on_file_error({"index": 2, "total": 3, "file": "s2.tif", "tb": "x"})
    c._on_hf_tile({"file": 1, "stem": "s1", "state": "running"})
    c._on_hf_tile({"file": 1, "stem": "s1", "state": "cancelled"})   # Stop
    assert c._series_status == {"s1": "queued", "s2": "error", "s3": "done"}
    c.deleteLater()
    _app.processEvents()
