"""Pure analysis-params builder for the QML UI (Phase 3).

The Widgets app builds the worker's params ``dict`` from ~80 sidebar widgets in
``BuildMixin._build_params_for_file``.  The QML sidebar lands in Phase 6, so
until then this module reproduces that dict from:

  * the user's persisted ``analysis/*`` / ``figures/*`` QSettings (so a sidebar
    configured in the Widgets app carries over), overlaid on
  * pristine widget-construction defaults (``_DEFAULTS`` — captured 1:1 from a
    freshly-built Widgets sidebar so a never-run user still gets valid values),
  * plus the live Import-tab file / output / calibration.

The output **shape is byte-identical** to ``_build_params_for_file`` (same keys,
same value types, same combo label→value mapping) so ``firefly_worker`` needs no
changes.  Combos are stored in QSettings as their display *labels*; the same
label→value maps the Widgets app uses are reproduced here.

No Qt imports — pure data, unit-testable with a fake settings object.
"""
from __future__ import annotations

import multiprocessing
import os

try:
    _N_CPUS = multiprocessing.cpu_count()
except Exception:
    _N_CPUS = 10

# ── combo label → worker-value maps (mirror _build_params_for_file) ──────────
BG_METHOD_MAP = {
    "Uniform Filter": "uniform_filter",
    "Rolling Ball":   "rolling_ball",
}
ROI_MODE_MAP = {
    "None":             "none",
    "Auto threshold":   "auto",
    "Manual threshold": "manual",
    "Manual polygon":   "polygon",
    "Sister TIFF":      "sister",
    "ImageJ ROI":       "imagej",
}
# Captured from c_backend / c_linker (label → currentData / internal value).
BACKEND_LABEL_TO_VALUE = {
    "Auto":                              "auto",
    "Crocker–Grier — Trackpy (CPU)":     "trackpy",
    "Crocker–Grier — PyTorch (GPU)":     "torch",
    "À trous wavelet — PyTorch (GPU)":   "atrous",
    "Gaussian MLE — PyTorch (GPU)":      "gaussian-mle",
    "Radial symmetry — PyTorch (GPU)":   "radial-symmetry",
}
LINKER_LABEL_TO_VALUE = {
    "Kalman filter — TrackMate (Linear Motion)":   "kalman",
    "Crocker–Grier — Trackpy":                     "trackpy",
    "Jaqaman LAP — TrackMate (simple)":            "simple_lap",
    "Jaqaman LAP — TrackMate (merge/split)":       "full_lap",
    "Nearest-neighbour — greedy":                  "nn",
    "Simulated annealing — palmTRACER (inspired)": "sa",
}
GAP_POLICY_LABEL_TO_VALUE = {
    "All timestamp pairs": "all_pairs",
    "Contiguous observations (legacy)": "contiguous",
}

# Pristine widget-construction defaults — captured from a freshly-built Widgets
# sidebar against an EMPTY QSettings store.  Used as the fallback for any key the
# user's QSettings doesn't carry, and as the canonical default set in tests.
FIG_PANELS_ALL = list("ABCDEFGHIJKLMNOPQ")

_DEFAULTS = {
    "channel":               0,
    "bg_method":             "Uniform Filter",
    "bg_radius":             10,
    "camera_gain":           0.0,
    "camera_qe":             0.0,
    "camera_bg_photons":     0.0,
    "diameter":              7,
    "auto_minmass":          True,
    "minmass":               1.0,
    "minmass_sensitivity":   "Balanced",
    "minmass_false_rate":    0.0,
    "search_range":          5,
    "auto_search_range":     False,
    "memory":                3,
    "min_track_len":         8,
    "max_track_len":         0,
    "max_lagtime":           20,
    "n_fit":                 5,
    "gap_policy":            "All timestamp pairs",
    "alpha_immobile":        0.5,
    "alpha_confined":        0.9,
    "alpha_directed":        1.1,
    "mobile_d":              0.05,
    "jdd_components":        2,
    "dcoeff_clip_min":       0.00001,   # LogD clip lower D bound (µm²/s) → log₁₀ −5
    "dcoeff_clip_max":       10.0,      # LogD clip upper D bound (µm²/s) → log₁₀ 1
    "filter_d_enable":       False,
    "filter_d_min":          0.0,
    "filter_d_max":          1.0,
    "roi_mode":              "Auto threshold",
    "roi_auto_method":       "Li",
    "roi_threshold":         0.08,
    "roi_mask_mode":         "Max",
    "roi_bg_sigma":          25.0,
    # Suffix identifying a companion ROI image beside a recording
    # (<stem><suffix>.tif/.tiff/.czi) — e.g. "_green", or "-Green Image" for a
    # Zeiss export.  User-editable in Preferences.
    "roi_sister_suffix":     "_green",
    "drift_correct":         True,
    "drift_segment":         500,
    "cluster_eps_nm":        50.0,
    "cluster_min_samples":   10,
    "backend":               "Auto",
    "workers":               _N_CPUS,
    "chunk_size":            500,
    "linker":                "Crocker–Grier — Trackpy",
    "allow_merging":         False,
    "allow_splitting":       False,
    "fig_theme":             "Dark",
    "fig_proj_cmap":         "Inferno",
    "fig_traj_bg":           True,
    "fig_dpi":               150,
    "fig_save_pdf":          False,
    "fig_per_panel":         False,
    "batch_fig_dpi":         110,
    "batch_fig_save_pdf":    False,
    "batch_fig_per_panel":   False,
}


def _i(settings, key, default):
    return int(round(settings.get_float(key, float(default))))


def expand_roi_replicates(base_params: dict) -> list:
    """The run list for one input file — now always a SINGLE run.

    "Analyse each ROI as its own replicate" used to fan out HERE into one full
    run per ROI, which re-decoded and re-localised the whole movie once per cell
    — the expensive half of the pipeline repeated for data that is identical.
    The worker now does the split internally (``_roi_replicate_jobs``): it
    localises ONCE, then loops the per-ROI analysis and writes one output folder
    per cell.  So the launcher hands it a single params carrying every polygon
    plus ``roi_split_replicates`` / ``roi_labels``, and this returns it unchanged.

    Kept as the launcher's seam so both call sites (batch + single run) stay
    honest that one input file may produce several outputs.
    """
    return [base_params]


def build_params(settings, importc, fpath: str | None = None,
                 out_dir: str | None = None, roi_store=None,
                 override_store=None) -> dict:
    """Build the worker params dict from persisted settings + the Import tab.

    ``settings`` is a SettingsController (or any object with
    ``get_str/get_float/get_bool``); ``importc`` is the ImportController
    (file / out_dir / calibration + overrides).  ``fpath`` / ``out_dir`` default
    to the Import tab's current file / output (the latter falling back to beside
    the input file, matching ``_start_single_run``).  ``roi_store`` (optional) is
    a per-file manual-polygon store; its polygon for ``fpath`` is sent as
    ``roi_polygon`` regardless of ROI mode (parity with ``_build_params_for_file``).
    ``override_store`` (optional) is a per-file ROI-settings override
    (RoiOverrideStore): if it has a spec for ``fpath`` it REPLACES the global
    ``analysis/roi_*`` defaults for this file only (the Preview & ROI viewer).
    """
    g = settings
    if fpath is None:
        fpath = importc.filePath
    # Derive is_csv from the actual file being built (correct for batch, where
    # fpath is a series file, not importc's single Import-tab file).
    is_csv = (str(fpath).lower().endswith((".csv", ".txt", ".tsv")) if fpath
              else bool(getattr(importc, "isCsv", False)))
    if out_dir is None:
        out_dir = importc.outDir or (
            os.path.dirname(fpath) if (is_csv and fpath) else None)

    # Calibration comes from the Import tab (override → value, else None so the
    # worker reads the file's embedded metadata).
    pixel_size = importc.pixelSize if importc.overridePx else None
    frame_interval = importc.frameInterval if importc.overrideFi else None

    false_rate = g.get_float("analysis/minmass_max_false_track_rate",
                             _DEFAULTS["minmass_false_rate"])
    max_tl = _i(g, "analysis/max_track_len", _DEFAULTS["max_track_len"])

    # ROI: global sidebar defaults, optionally replaced by a per-file override.
    roi_mode_label  = g.get_str("analysis/roi_mode", _DEFAULTS["roi_mode"])
    roi_auto_method = g.get_str("analysis/roi_auto_method", _DEFAULTS["roi_auto_method"])
    roi_threshold   = g.get_float("analysis/roi_threshold", _DEFAULTS["roi_threshold"])
    roi_mask_mode   = g.get_str("analysis/roi_mask_mode", _DEFAULTS["roi_mask_mode"])
    roi_bg_sigma    = g.get_float("analysis/roi_bg_sigma", _DEFAULTS["roi_bg_sigma"])
    ovr = override_store.get(fpath) if (override_store and fpath) else None
    # Multiple drawn ROIs → treat each as its own replicate (separate output).
    # Per-file, set in the ROI viewer (flag + optional per-ROI labels).
    roi_split_replicates = bool(ovr.get("roi_split_replicates", False)) if ovr else False
    roi_labels = (list(ovr.get("roi_labels") or []) if ovr else []) or None
    if ovr:
        roi_mode_label  = ovr.get("roi_mode", roi_mode_label)
        roi_auto_method = ovr.get("roi_auto_method", roi_auto_method)
        roi_threshold   = float(ovr.get("roi_threshold", roi_threshold))
        roi_mask_mode   = ovr.get("roi_mask_mode", roi_mask_mode)
        roi_bg_sigma    = float(ovr.get("roi_bg_sigma", roi_bg_sigma))
    roi_mode = ROI_MODE_MAP.get(roi_mode_label, "none")
    backend_label = g.get_str("analysis/backend", _DEFAULTS["backend"])
    linker_label = g.get_str("analysis/linker", _DEFAULTS["linker"])

    params = {
        "file":           fpath,
        "out_dir":        out_dir,
        "pixel_size":     pixel_size,
        "frame_interval": frame_interval,
        "channel":        _i(g, "analysis/channel", _DEFAULTS["channel"]),
        "bg_method":      BG_METHOD_MAP.get(
            g.get_str("analysis/bg_method", _DEFAULTS["bg_method"]),
            "uniform_filter"),
        "bg_radius":      _i(g, "analysis/bg_radius", _DEFAULTS["bg_radius"]),
        "camera_gain":       g.get_float("analysis/camera_gain", _DEFAULTS["camera_gain"]),
        "camera_qe":         g.get_float("analysis/camera_qe", _DEFAULTS["camera_qe"]),
        "camera_bg_photons": g.get_float("analysis/camera_bg_photons", _DEFAULTS["camera_bg_photons"]),
        "diameter":       _i(g, "analysis/diameter", _DEFAULTS["diameter"]),
        "auto_minmass":   g.get_bool("analysis/auto_minmass", _DEFAULTS["auto_minmass"]),
        "minmass":        g.get_float("analysis/minmass", _DEFAULTS["minmass"]),
        "minmass_sensitivity": g.get_str(
            "analysis/minmass_sensitivity", _DEFAULTS["minmass_sensitivity"]).lower(),
        "minmass_max_false_track_rate": (
            (false_rate / 100.0) if false_rate > 0 else None),
        "search_range":   _i(g, "analysis/search_range", _DEFAULTS["search_range"]),
        "auto_search_range": g.get_bool("analysis/auto_search_range", _DEFAULTS["auto_search_range"]),
        "memory":         _i(g, "analysis/memory", _DEFAULTS["memory"]),
        "min_track_len":  _i(g, "analysis/min_track_len", _DEFAULTS["min_track_len"]),
        "max_track_len":  max_tl if max_tl > 0 else None,
        "max_lagtime":    _i(g, "analysis/max_lagtime", _DEFAULTS["max_lagtime"]),
        "n_fit":          _i(g, "analysis/n_fit", _DEFAULTS["n_fit"]),
        "gap_policy":     GAP_POLICY_LABEL_TO_VALUE.get(
            g.get_str("analysis/gap_policy", _DEFAULTS["gap_policy"]),
            "all_pairs"),
        "alpha_thresholds": (
            g.get_float("analysis/alpha_immobile", _DEFAULTS["alpha_immobile"]),
            g.get_float("analysis/alpha_confined", _DEFAULTS["alpha_confined"]),
            g.get_float("analysis/alpha_directed", _DEFAULTS["alpha_directed"])),
        "mobile_d_threshold": g.get_float("analysis/mobile_d", _DEFAULTS["mobile_d"]),
        "jdd_components": _i(g, "analysis/jdd_components", _DEFAULTS["jdd_components"]),
        # The sidebar stores these ranges in log₁₀D; the worker/exports want D
        # (µm²/s), so convert here (10**logD).  Keeps the analysis params intact.
        "dcoeff_clip_min": 10.0 ** g.get_float("analysis/dcoeff_clip_logmin", -5.0),
        "dcoeff_clip_max": 10.0 ** g.get_float("analysis/dcoeff_clip_logmax", 1.0),
        "filter_d_enabled": g.get_bool("analysis/filter_d_enable", _DEFAULTS["filter_d_enable"]),
        "filter_d_min":   10.0 ** g.get_float("analysis/filter_d_logmin", -5.0),
        "filter_d_max":   10.0 ** g.get_float("analysis/filter_d_logmax", 0.0),
        "roi_mode":       roi_mode,
        "roi_auto_method": roi_auto_method,
        "roi_threshold":  roi_threshold,
        "roi_mask_mode":  roi_mask_mode,
        "roi_bg_sigma":   roi_bg_sigma,
        # Must come from the SAME setting the ROI viewer reads: hardcoding it
        # meant a custom suffix found the companion image in the preview but not
        # in the actual run, so the analysed region silently differed from the
        # one shown.
        # An UNSET key yields the default; an explicitly EMPTY one is the user
        # deliberately turning companion matching off, which the worker and
        # find_sister_roi_path both already treat as "don't look".  So no
        # `or default` here — that would make clearing the field impossible.
        "roi_sister_suffix":     g.get_str("analysis/roi_sister_suffix",
                                           _DEFAULTS["roi_sister_suffix"]),
        "roi_imagej_autodetect": (roi_mode == "imagej"),
        "roi_polygon":    ((roi_store.get(fpath) if (roi_store and fpath) else None) or None),
        "roi_split_replicates": roi_split_replicates,
        "roi_labels":     roi_labels,
        "drift_correct":  g.get_bool("analysis/drift_correct", _DEFAULTS["drift_correct"]),
        "drift_segment":  _i(g, "analysis/drift_segment", _DEFAULTS["drift_segment"]),
        "cluster_eps_nm": g.get_float("analysis/cluster_eps_nm", _DEFAULTS["cluster_eps_nm"]),
        "cluster_min_samples": _i(g, "analysis/cluster_min_samples", _DEFAULTS["cluster_min_samples"]),
        "backend":        BACKEND_LABEL_TO_VALUE.get(backend_label, "auto"),
        "workers":        _i(g, "analysis/workers", _DEFAULTS["workers"]),
        "chunk_size":     _i(g, "analysis/chunk_size", _DEFAULTS["chunk_size"]),
        "linker":         LINKER_LABEL_TO_VALUE.get(linker_label, "kalman"),
        "link_params": {
            "allow_merging":   g.get_bool("analysis/allow_merging", _DEFAULTS["allow_merging"]),
            "allow_splitting": g.get_bool("analysis/allow_splitting", _DEFAULTS["allow_splitting"]),
        },
        "fig_theme":      g.get_str("figures/theme", _DEFAULTS["fig_theme"]),
        "fig_proj_cmap":  g.get_str("figures/proj_cmap", _DEFAULTS["fig_proj_cmap"]),
        "fig_traj_bg":    g.get_bool("figures/traj_bg", _DEFAULTS["fig_traj_bg"]),
        "fig_dpi":        _i(g, "figures/dpi", _DEFAULTS["fig_dpi"]),
        "fig_save_pdf":   g.get_bool("figures/save_pdf", _DEFAULTS["fig_save_pdf"]),
        "fig_per_panel":  g.get_bool("figures/per_panel", _DEFAULTS["fig_per_panel"]),
        "fig_single_panels": list(FIG_PANELS_ALL),
        "batch_fig_dpi":  _i(g, "figures/batch_dpi", _DEFAULTS["batch_fig_dpi"]),
        "batch_fig_save_pdf":  g.get_bool("figures/batch_save_pdf", _DEFAULTS["batch_fig_save_pdf"]),
        "batch_fig_per_panel": g.get_bool("figures/batch_per_panel", _DEFAULTS["batch_fig_per_panel"]),
    }

    # External-localisations table: skip detection and seed calibration from the
    # sidebar even without an Override (a CSV has no embedded metadata).  Mirrors
    # the is_loc branch of _start_single_run.
    if is_csv:
        if not params["pixel_size"]:
            params["pixel_size"] = float(importc.pixelSize)
        if not params["frame_interval"]:
            params["frame_interval"] = float(importc.frameInterval)
        params["source"] = "external_csv"
        # Source-format override (default "auto" auto-detects) + optional
        # background image so the figure gets a real max-projection.  getattr so
        # a minimal import double (batch path / tests) still builds.
        params["csv_preset"] = getattr(importc, "csvPreset", "auto") or "auto"
        params["bg_image_path"] = getattr(importc, "bgImagePath", "") or ""

    # Widget-state snapshot for the run manifest (manifest "Replay" reads this).
    # Partial until the QML sidebar lands in Phase 6 — the analysis/* + figures/*
    # keys we actually source, stored under their QSettings paths.
    params["widget_state"] = _widget_state_snapshot(g, importc)
    return params


def _widget_state_snapshot(g, importc) -> dict:
    """Best-effort widget-state dict (QSettings-key → value), mirroring
    ``_widget_state_dict`` for the keys this builder sources."""
    keys_str = [
        "analysis/bg_method", "analysis/minmass_sensitivity", "analysis/roi_mode",
        "analysis/roi_auto_method", "analysis/roi_mask_mode", "analysis/backend",
        "analysis/linker", "analysis/gap_policy",
        "figures/theme", "figures/proj_cmap",
    ]
    keys_num = [
        "analysis/bg_radius", "analysis/camera_gain", "analysis/camera_qe",
        "analysis/camera_bg_photons", "analysis/diameter", "analysis/minmass",
        "analysis/minmass_max_false_track_rate", "analysis/search_range",
        "analysis/memory", "analysis/min_track_len", "analysis/max_track_len",
        "analysis/max_lagtime", "analysis/n_fit", "analysis/alpha_immobile",
        "analysis/alpha_confined", "analysis/alpha_directed", "analysis/mobile_d",
        "analysis/jdd_components",
        "analysis/dcoeff_clip_logmin", "analysis/dcoeff_clip_logmax",
        "analysis/filter_d_logmin", "analysis/filter_d_logmax",
        "analysis/roi_threshold", "analysis/roi_bg_sigma", "analysis/drift_segment",
        "analysis/cluster_eps_nm", "analysis/cluster_min_samples",
        "analysis/workers", "analysis/chunk_size", "analysis/channel", "figures/dpi",
    ]
    keys_bool = [
        "analysis/auto_minmass", "analysis/auto_search_range", "analysis/filter_d_enable",
        "analysis/drift_correct", "analysis/allow_merging", "analysis/allow_splitting",
        "figures/traj_bg", "figures/save_pdf", "figures/per_panel",
    ]
    out: dict = {}
    for k in keys_str:
        out[k] = g.get_str(k, _DEFAULTS.get(k.split("/")[-1], ""))
    for k in keys_num:
        out[k] = g.get_float(k, float(_DEFAULTS.get(k.split("/")[-1], 0.0) or 0.0))
    for k in keys_bool:
        out[k] = g.get_bool(k, bool(_DEFAULTS.get(k.split("/")[-1], False)))
    out["analysis/pixel_size"] = float(importc.pixelSize)
    out["analysis/frame_interval"] = float(importc.frameInterval)
    out["analysis/override_px"] = bool(importc.overridePx)
    out["analysis/override_fi"] = bool(importc.overrideFi)
    return out
