"""Unit tests for the recursive-batch folder-scan predicates.

These gate which folders a recursive batch descends into and which files it
auto-queues.  They live in firefly.ui.ui_batch_filters precisely so they import
WITHOUT PySide6 — the CI test runner installs only the analysis stack (no Qt),
so importing ui_mixin_batch here would fail; the pure predicates do not.
"""
from firefly.ui.ui_batch_filters import (is_analysis_output_dir,
                                          is_raw_image_name,
                                          is_palmtracer_loc_table)


def _touch(path):
    path.touch()


# ── analysis-output folders the recursive sweep must prune ──────────────
def test_is_analysis_output_dir_palmtracer_dirs():
    # palmTRACER `.PT` analysis folders (real names from the lab tree)
    assert is_analysis_output_dir("20260122_PC12-P09_Syntaxin1a_Propofol_D3_Post.PT")
    assert is_analysis_output_dir("foo.pt")            # case-insensitive
    assert is_analysis_output_dir("FOO.PT")
    # numbered analysis-stage folders
    assert is_analysis_output_dir("02_Analysis_correct pixel setting")
    assert is_analysis_output_dir("02_analysis")
    assert is_analysis_output_dir("03_Analysis")


def test_is_analysis_output_dir_keeps_raw_and_ordinary():
    # raw acquisition folder must NOT be pruned
    assert not is_analysis_output_dir("01_Raw")
    assert not is_analysis_output_dir("01_raw")
    # ordinary experiment / condition / cell folders survive
    assert not is_analysis_output_dir("Syntaxin1a")
    assert not is_analysis_output_dir("Ciprofol")
    assert not is_analysis_output_dir("Cell1")
    assert not is_analysis_output_dir("1-AMA")
    assert not is_analysis_output_dir("Propofol reversal")
    # a `.pt` substring that isn't the suffix stays put
    assert not is_analysis_output_dir("ptychography")


# ── recursive auto-queue is raw-images-only ─────────────────────────────
def test_is_raw_image_name_accepts_images():
    for n in ("a.tif", "a.tiff", "a.czi", "A.TIF", "stack.TIFF", "x.CZI"):
        assert is_raw_image_name(n), n


def test_is_raw_image_name_rejects_loc_tables():
    # the abundant palmTRACER/FIREFLY exports a recursive sweep must skip
    for n in ("locPALMTracer.txt", "locPALMTracer.csv",
              "trcPALMTracer-1-D.txt", "results.csv", "notes.txt",
              "data.tsv", "blob.bin"):
        assert not is_raw_image_name(n), n


# ── palmTRACER loc-table detection (batch 'palmTRACER data' input mode) ──
def test_is_palmtracer_loc_table_accepts_loc_tables():
    for n in ("locPALMTracer.txt", "locPALMTracer.csv",
              "LOCPALMTRACER.TXT",
              "20260122_Syntaxin1a_DMSO_D1_Post_locPALMTracer.csv"):
        assert is_palmtracer_loc_table(n), n


def test_is_palmtracer_loc_table_rejects_others():
    # raw images, track/D/MSD tables, and unrelated files are NOT loc tables
    for n in ("stack.tif", "movie.czi",
              "trcPALMTracer.txt", "trcPALMTracer-AllROI-MSD.csv",
              "trcPALMTracer-1-D.txt", "results.csv", "notes.txt"):
        assert not is_palmtracer_loc_table(n), n


# ── QML batch scanner: source-aware grouping ───────────────────────────────
def test_batch_scan_keeps_independent_inputs_beside_a_czi(tmp_path):
    """A CZI only wins over its own matching image siblings, never a folder."""
    from firefly.ui.controllers.params import batch_scan
    for name in ("rawA.czi", "independentB.tif", "independentC.csv"):
        _touch(tmp_path / name)
    rows = {s["key"]: s for s in batch_scan.scan_series(str(tmp_path))}
    assert set(rows) == {"rawA", "independentB", "independentC"}
    assert rows["rawA"]["sourceType"] == "image"
    assert rows["independentB"]["sourceType"] == "image"
    assert rows["independentC"]["sourceType"] == "external_loc"


def test_batch_scan_czi_preference_is_limited_to_matching_image_stem(tmp_path):
    from firefly.ui.controllers.params import batch_scan
    _touch(tmp_path / "cell.czi")
    _touch(tmp_path / "cell.tif")
    rows = batch_scan.scan_series(str(tmp_path))
    assert len(rows) == 1
    assert rows[0]["primary"].endswith("cell.czi")
    assert rows[0]["files"] == [str(tmp_path / "cell.czi")]


def test_batch_scan_never_mixes_images_and_localisation_tables(tmp_path):
    from firefly.ui.controllers.params import batch_scan
    _touch(tmp_path / "cell.tif")
    _touch(tmp_path / "cell_locPALMTracer.csv")
    rows = {s["key"]: s for s in batch_scan.scan_series(str(tmp_path))}
    assert set(rows) == {"cell", "cell_locPALMTracer"}
    assert rows["cell"]["sourceType"] == "image"
    assert rows["cell"]["files"] == [str(tmp_path / "cell.tif")]
    assert rows["cell_locPALMTracer"]["sourceType"] == "external_loc"
    assert rows["cell_locPALMTracer"]["fileCount"] == 1


def test_batch_scan_keeps_unknown_suffixes_but_skips_known_green_sister(tmp_path):
    from firefly.ui.controllers.params import batch_scan
    for name in ("cell.tif", "cell_green.tif", "cell_control.tif"):
        _touch(tmp_path / name)
    rows = {s["key"] for s in batch_scan.scan_series(str(tmp_path))}
    assert rows == {"cell", "cell_control"}


def test_batch_scan_tables_are_standalone_and_tsv_is_supported(tmp_path):
    from firefly.ui.controllers.params import batch_scan
    for name in ("same.csv", "same.txt", "table.tsv"):
        _touch(tmp_path / name)
    rows = batch_scan.scan_series(str(tmp_path))
    assert len(rows) == 3
    assert len({r["key"] for r in rows}) == 3
    assert all(r["sourceType"] == "external_loc" for r in rows)
    assert all(r["fileCount"] == 1 for r in rows)
    assert any(r["primary"].endswith("table.tsv") for r in rows)


def test_batch_scan_recursive_mode_remains_raw_images_only(tmp_path):
    from firefly.ui.controllers.params import batch_scan
    nested = tmp_path / "nested"; nested.mkdir()
    _touch(nested / "table.tsv")
    assert batch_scan.scan_series(str(tmp_path), recursive=True) == []


# ── HYPER-FLY hardware-eligibility gate (controls the UI exposure) ───────
def test_hyperfly_machine_eligible(monkeypatch):
    from firefly.analysis import fa_hyperfly as hf
    # clears both bars -> eligible
    monkeypatch.setattr(hf, "N_CPUS", 128)
    monkeypatch.setattr(hf, "_total_ram_gb", lambda: 752.0)
    assert hf.hyperfly_machine_eligible() is True
    # too few cores -> not eligible (even with plenty of RAM)
    monkeypatch.setattr(hf, "N_CPUS", 8)
    assert hf.hyperfly_machine_eligible() is False
    # enough cores but too little RAM -> not eligible
    monkeypatch.setattr(hf, "N_CPUS", 128)
    monkeypatch.setattr(hf, "_total_ram_gb", lambda: 64.0)
    assert hf.hyperfly_machine_eligible() is False
    # exactly on the thresholds -> eligible (>=)
    monkeypatch.setattr(hf, "N_CPUS", hf.HYPERFLY_MIN_CORES)
    monkeypatch.setattr(hf, "_total_ram_gb", lambda: float(hf.HYPERFLY_MIN_RAM_GB))
    assert hf.hyperfly_machine_eligible() is True
