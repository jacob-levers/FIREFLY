"""Tests for the typed dispatch enums (firefly.analysis.fa_enums).

Guards the two properties the refactor relies on:
  1. parse() maps every known/legacy/cased spelling to the right member, and an
     UNKNOWN value returns the documented fallback AND logs (was silent before).
  2. .value equals the exact wire string for every member (manifest round-trip).
"""
from firefly.analysis.fa_enums import (
    ROIMode, MaskMode, FigureTheme, Backend, MsgKind)


def test_roi_mode_parse_and_wire_values():
    for m in ROIMode:
        assert ROIMode.parse(m.value) is m
        assert ROIMode.parse(m.value.upper()) is m          # case-insensitive
    logs = []
    assert ROIMode.parse("spectral_unmix", log=logs.append) is ROIMode.NONE
    assert logs and "unknown ROI mode" in logs[0]           # not silent
    # exact wire strings (persisted in the manifest)
    assert {m.value for m in ROIMode} == {
        "none", "auto", "manual", "polygon", "sister", "imagej"}


def test_mask_mode_parse_display_strings_and_prefixes():
    assert MaskMode.parse("Max") is MaskMode.MAX
    assert MaskMode.parse("mean") is MaskMode.MEAN
    assert MaskMode.parse("Sum") is MaskMode.SUM
    assert MaskMode.parse("Blink density") is MaskMode.BLINK   # GUI display string
    assert MaskMode.parse("MaxProjection") is MaskMode.MAX     # prefix preserved
    assert MaskMode.parse("") is MaskMode.MEAN                 # legacy empty → mean
    logs = []
    assert MaskMode.parse("median", log=logs.append) is MaskMode.MEAN
    assert logs and "unknown ROI mask mode" in logs[0]        # used to be silent
    # value == the mode_hint the mask builder expects
    assert {m.value for m in MaskMode} == {"max", "mean", "sum", "blink"}


def test_figure_theme_parse():
    for m in FigureTheme:
        assert FigureTheme.parse(m.value) is m
        assert FigureTheme.parse(m.value.lower()) is m
    logs = []
    assert FigureTheme.parse("Neon", log=logs.append) is FigureTheme.DARK
    assert logs
    assert {m.value for m in FigureTheme} == {
        "Dark", "Light", "Publication", "AMOLED"}


def test_backend_parse_strips_device_suffix():
    assert Backend.parse("trackpy") is Backend.TRACKPY
    assert Backend.parse("torch-cuda") is Backend.TORCH_CUDA
    assert Backend.parse("torch-cuda:0") is Backend.TORCH_CUDA   # suffix dropped
    assert Backend.parse("Torch-MPS") is Backend.TORCH_MPS
    assert Backend.TORCH_CUDA.is_explicit_gpu is True
    assert Backend.TORCH_CPU.is_explicit_gpu is False
    assert Backend.TRACKPY.is_torch is False
    logs = []
    assert Backend.parse("jax", log=logs.append) is Backend.AUTO
    assert logs
    assert Backend.parse("atrous") is Backend.ATROUS
    assert Backend.ATROUS.is_explicit_gpu is False         # auto-device, not a pin
    assert {m.value for m in Backend} == {
        "auto", "trackpy", "torch", "torch-cpu", "torch-cuda", "torch-mps",
        "atrous"}


def test_msgkind_is_str_and_complete():
    # StrEnum members compare equal to their plain string → emit/read sites can
    # migrate incrementally.
    assert MsgKind.LOG == "log"
    assert MsgKind.DONE == "done"
    assert {k.value for k in MsgKind} == {
        "log", "progress", "mass_chunk", "preview_frame", "done",
        "file_starting", "file_done", "file_error", "batch_done",
        "compare_done", "compare_error", "hf_tile", "hyperfly_status",
        "stopped", "error"}
