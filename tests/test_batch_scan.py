"""Unit tests for the recursive-batch folder-scan predicates.

These gate which folders a recursive batch descends into and which files it
auto-queues.  They live in firefly.ui.ui_batch_filters precisely so they import
WITHOUT PySide6 — the CI test runner installs only the analysis stack (no Qt),
so importing ui_mixin_batch here would fail; the pure predicates do not.
"""
from firefly.ui.ui_batch_filters import (is_analysis_output_dir,
                                          is_raw_image_name,
                                          is_palmtracer_loc_table)


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
