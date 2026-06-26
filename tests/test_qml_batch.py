"""Phase-6b tests: batch_scan series grouping + BatchController params list."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

import pytest                                            # noqa: E402

pytest.importorskip("PySide6")
from PySide6 import QtWidgets                            # noqa: E402

_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _wait_scan(c, timeout=3.0):
    """Folder scan runs off-thread now — pump the loop until it drains in."""
    import time
    t0 = time.time()
    while c.scanning and time.time() - t0 < timeout:
        _app.processEvents()
        time.sleep(0.005)
    _app.processEvents()


def _touch(p):
    open(p, "w").close()


class FakeSettings:
    def get_str(self, k, d=""): return d
    def get_float(self, k, d=0.0): return d
    def get_bool(self, k, d=False): return d
    def set(self, k, v): pass
    def sync(self): pass


class FakeImport:
    filePath = ""; outDir = ""; isCsv = False
    overridePx = False; pixelSize = 0.106; overrideFi = False; frameInterval = 0.02


# ── batch_scan ───────────────────────────────────────────────────────────────
def test_scan_series_groups_and_filters(tmp_path):
    from firefly.ui.controllers.params import batch_scan
    d = str(tmp_path)
    # top folder (no czi): a.tif + its split-TIFF siblings → one series "a"
    _touch(os.path.join(d, "a.tif"))
    _touch(os.path.join(d, "a-file001.tif"))
    _touch(os.path.join(d, "a-file002.tif"))
    _touch(os.path.join(d, "a_diffusion_summary.csv"))  # FIREFLY aux → excluded
    _touch(os.path.join(d, "._a.tif"))                  # AppleDouble → excluded
    # a subfolder with a .czi + a derived .tif → czi preference keeps only the czi
    os.makedirs(os.path.join(d, "cellB"))
    _touch(os.path.join(d, "cellB", "b.czi"))
    _touch(os.path.join(d, "cellB", "b.tif"))

    series = {s["key"]: s for s in batch_scan.scan_series(d)}
    assert "a" in series and series["a"]["fileCount"] == 3
    bkey = next(k for k in series if k.endswith("b"))   # subfolder-prefixed key
    assert series[bkey]["fileCount"] == 1
    assert series[bkey]["primary"].endswith("b.czi")


def test_scan_series_empty(tmp_path):
    from firefly.ui.controllers.params import batch_scan
    assert batch_scan.scan_series(str(tmp_path)) == []
    assert batch_scan.scan_series("/nonexistent/xyz") == []


def test_series_key_strips_suffixes():
    from firefly.ui.controllers.params.batch_scan import series_key
    assert series_key("expt.tif") == "expt"
    assert series_key("expt-file002.tif") == "expt"
    assert series_key("expt(1).tif") == "expt"
    assert series_key("expt_locPALMTracer.csv") == "expt"


# ── BatchController ──────────────────────────────────────────────────────────
def test_batch_controller_scan_and_select(tmp_path):
    from firefly.ui.controllers.batch_controller import BatchController
    for n in ("c1.tif", "c2.tif", "c3.tif"):
        _touch(os.path.join(tmp_path, n))
    c = BatchController(FakeSettings(), FakeImport())
    c.scan(str(tmp_path)); _wait_scan(c)
    keys = {s["key"] for s in c.series}
    assert keys == {"c1", "c2", "c3"}
    assert all(s["checked"] for s in c.series)        # all selected by default
    assert c.canRun
    c.setChecked("c2", False)
    assert not next(s for s in c.series if s["key"] == "c2")["checked"]
    c.selectAll(False)
    assert not c.canRun
    c.selectAll(True)
    assert c.canRun and "3 series" in c.summary


def test_batch_params_list_overrides(tmp_path):
    from firefly.ui.controllers.batch_controller import BatchController
    _touch(os.path.join(tmp_path, "cellA.tif"))
    _touch(os.path.join(tmp_path, "cellA-file001.tif"))
    _touch(os.path.join(tmp_path, "cellB.tif"))
    c = BatchController(FakeSettings(), FakeImport())
    c.scan(str(tmp_path)); _wait_scan(c)
    plist = c._build_params_list()
    assert len(plist) == 2
    by_stem = {p["stem_override"]: p for p in plist}
    assert set(by_stem) == {"cellA", "cellB"}
    # series_files override carries the sister files for the multi-file series
    assert len(by_stem["cellA"]["series_files"]) == 2
    # output goes under <folder>/batch_results/<stem> (worker wraps the stem)
    assert by_stem["cellA"]["out_dir"].endswith("batch_results")
    # the params dict is otherwise the standard build_params shape
    assert "diameter" in by_stem["cellA"] and "backend" in by_stem["cellA"]


def test_batch_csv_series_gets_csv_source(tmp_path):
    from firefly.ui.controllers.batch_controller import BatchController
    _touch(os.path.join(tmp_path, "locs1.csv"))
    c = BatchController(FakeSettings(), FakeImport())
    c.scan(str(tmp_path)); _wait_scan(c)
    p = c._build_params_list()[0]
    assert p["source"] == "external_csv"          # derived from the .csv fpath
    assert "series_files" not in p                # CSVs don't get the image series list


def test_batch_flags_unreadable_file(tmp_path):
    """A corrupt/empty recording is flagged in the queue once its series is
    probed, so it's visible before the batch runs (not only when the run fails)."""
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    from firefly.ui.controllers.batch_controller import BatchController
    tifffile.imwrite(str(tmp_path / "good.tif"), np.zeros((3, 8, 8), np.uint16))
    _touch(os.path.join(tmp_path, "bad.tif"))         # 0-byte → can't be read
    c = BatchController(FakeSettings(), FakeImport())
    c.scan(str(tmp_path)); _wait_scan(c)
    for s in c.series:                                # expand → probes frames
        c.setOpen(s["key"], True)
    rows = {s["key"]: s for s in c.series}
    assert rows["good"]["hasUnreadable"] is False
    assert rows["good"]["parts"][0]["unreadable"] is False
    assert rows["bad"]["hasUnreadable"] is True
    assert rows["bad"]["parts"][0]["unreadable"] is True
