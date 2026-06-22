"""Schema for the QML analysis-parameter sidebar (Phase 6).

One field spec per persisted sidebar control, mirroring the Widgets
``_setting_specs`` + ``_build_sidebar`` construction (ranges / steps / decimals /
defaults / labels / combo items).  The SidebarController is a thin generic
accessor over this list — it writes the user's edits straight through to the
SAME ``analysis/*`` / ``figures/*`` / ``performance/*`` QSettings keys that
``params_builder`` reads, so the worker param dict stays byte-identical.

Combos persist by LABEL (``currentText``), byte-identical to the keys of the
label→value maps in ``params_builder`` — never by index.  No Qt here; pure data.
"""
from __future__ import annotations

import multiprocessing

try:
    _N_CPUS = multiprocessing.cpu_count()
except Exception:
    _N_CPUS = 8

# Per-concern section order + icons (icons from the bundled Lucide set).
SECTIONS = [
    {"key": "imaging",       "title": "Imaging metadata",      "icon": "sliders-horizontal"},
    {"key": "preprocessing", "title": "Preprocessing",          "icon": "layers"},
    {"key": "camera",        "title": "Camera calibration",     "icon": "image"},
    {"key": "detection",     "title": "Detection",              "icon": "scan-search"},
    {"key": "linking",       "title": "Linking",                "icon": "link"},
    {"key": "diffusion",     "title": "Diffusion & motion",     "icon": "zap"},
    {"key": "roi",           "title": "ROI",                    "icon": "move"},
    {"key": "drift",         "title": "Drift correction",       "icon": "waypoints"},
    {"key": "clustering",    "title": "Clustering (DBSCAN)",    "icon": "circle-dot"},
    {"key": "performance",   "title": "Performance",            "icon": "cpu"},
]


def _f(section, key, kind, label, default, *, min=None, max=None, step=None,
       decimals=None, items=None, suffix="", special="", enable=None,
       hyperfly=False, tooltip="", slider=False):
    return {"section": section, "key": key, "kind": kind, "label": label,
            "default": default, "min": min, "max": max, "step": step,
            "decimals": decimals, "items": items or [], "suffix": suffix,
            "special": special, "enable": enable, "hyperfly": hyperfly,
            "tooltip": tooltip or _TOOLTIPS.get(key, ""), "slider": slider}


# Hover help for every parameter (shown by FieldRow on hover).
_TOOLTIPS = {
    # Imaging
    "analysis/override_px": "Override the pixel size read from the file's metadata.",
    "analysis/pixel_size": "Camera pixel size in the sample plane (µm). Sets the spatial scale for D and all distances.",
    "analysis/override_fi": "Override the frame interval read from the file's metadata.",
    "analysis/frame_interval": "Time between frames (s). Sets the time scale for diffusion (D, α).",
    "analysis/channel": "Which channel to analyse in a multi-channel CZI.",
    # Preprocessing
    "analysis/bg_method": "Background-subtraction method. Uniform Filter is ~1700× faster than Rolling Ball with similar results.",
    "analysis/bg_radius": "Background estimation radius (px). Larger keeps more low-frequency structure; smaller flattens more aggressively.",
    # Camera
    "analysis/camera_gain": "Camera gain (e⁻/ADU). With QE, converts intensities to photons for a CRLB precision estimate.",
    "analysis/camera_qe": "Quantum efficiency (0–1). Used with gain for the photon / localisation-precision estimate.",
    "analysis/camera_bg_photons": "Mean background (photons/px) for the precision (CRLB) estimate.",
    # Detection
    "analysis/diameter": "Expected spot diameter in pixels (odd) — roughly the PSF size. Too small splits spots, too large merges them.",
    "analysis/auto_minmass": "Auto-pick the detection threshold per file. Turn off to set a fixed minmass below.",
    "analysis/minmass": "Minimum integrated brightness for a detection. Higher = fewer, brighter spots. Preview it in the ROI viewer.",
    "analysis/minmass_sensitivity": "How aggressive the auto-threshold is. Strict = fewer false spots; Lenient = more detections.",
    "analysis/minmass_max_false_track_rate": "Cap the estimated false-track rate when auto-thresholding (off = no cap).",
    # Linking
    "analysis/linker": "Algorithm that connects detections into trajectories across frames.",
    "analysis/allow_merging": "Permit two tracks to merge into one (Full LAP linker only).",
    "analysis/allow_splitting": "Permit one track to split into two (Full LAP linker only).",
    "analysis/search_range": "Max distance (px) a particle can move between frames and still be linked.",
    "analysis/auto_search_range": "Estimate the search range from the data instead of setting it by hand.",
    "analysis/memory": "Frames a particle may disappear (blink/miss) and still rejoin the same track.",
    "analysis/min_track_len": "Discard trajectories shorter than this many frames.",
    "analysis/max_track_len": "Cap trajectory length (0 = no cap).",
    # Diffusion & motion
    "analysis/max_lagtime": "Largest lag (frames) used when building the MSD curve.",
    "analysis/n_fit": "Number of initial MSD points fitted for D and α.",
    "analysis/alpha_immobile": "Anomalous exponent α below this → classed Immobile.",
    "analysis/alpha_confined": "α below this (and above immobile) → Confined.",
    "analysis/alpha_directed": "α above this → Directed / super-diffusive.",
    "analysis/mobile_d": "D threshold (µm²/s) separating mobile from immobile for the mobile-fraction metric.",
    "analysis/jdd_components": "Number of diffusing populations fitted in the jump-distance distribution.",
    "analysis/filter_d_enable": "Drop trajectories whose D falls outside the range below.",
    "analysis/filter_d_min": "Minimum D (µm²/s) to keep a trajectory.",
    "analysis/filter_d_max": "Maximum D (µm²/s) to keep a trajectory.",
    # ROI
    "analysis/roi_mode": "How the region of interest is defined for this run.",
    "analysis/roi_auto_method": "Auto-threshold algorithm used to build the ROI mask.",
    "analysis/roi_threshold": "Manual intensity threshold (0–1) for the ROI mask.",
    "analysis/roi_mask_mode": "Which projection the ROI mask is thresholded on.",
    "analysis/roi_bg_sigma": "Background-flattening σ applied before thresholding the ROI mask.",
    # Drift
    "analysis/drift_correct": "Correct sample drift (redundant cross-correlation) before linking.",
    "analysis/drift_segment": "Frames per drift-estimation segment. Smaller tracks faster drift; larger is more robust.",
    # Clustering
    "analysis/cluster_eps_nm": "DBSCAN neighbourhood radius (nm) — the max gap within a cluster.",
    "analysis/cluster_min_samples": "DBSCAN minimum localisations needed to seed a cluster.",
    # Performance
    "analysis/backend": "Detection engine. Auto picks the best available (GPU when present).",
    "analysis/workers": "CPU worker processes for preprocessing / localisation.",
    "analysis/chunk_size": "Frames processed per chunk (memory vs throughput trade-off).",
    "performance/hyperfly": "Parallel multi-file batch mode on big workstations.",
    "performance/hyperfly_max_files": "Cap concurrent files in HYPER-FLY (0 = auto).",
    "performance/hyperfly_max_cores": "Cap cores used by HYPER-FLY (0 = all).",
    "performance/hyperfly_max_ram": "Cap RAM (GB) for HYPER-FLY (0 = auto).",
    "performance/hyperfly_load_slots": "Concurrent file loads in HYPER-FLY (0 = auto).",
    "performance/hyperfly_gpu_slots": "Concurrent GPU detections in HYPER-FLY (0 = auto).",
    "performance/czi_parallel_decode": "Decode CZI frames in parallel for a faster load.",
}


# enable predicates: {"key": other_key, "truthy": bool} or {"key": k, "eq": value}
_FULL_LAP = "Jaqaman LAP — TrackMate (merge/split)"

FIELDS = [
    # ── Imaging metadata ─────────────────────────────────────────────────
    # Calibration (pixel size + frame interval) is owned here — there's no
    # calibration UI on the Import tab; these write the analysis/* keys that
    # params_builder reads.
    _f("imaging", "analysis/override_px", "bool", "Override pixel size", False),
    _f("imaging", "analysis/pixel_size", "double", "Pixel size (µm)", 0.106,
       min=0.01, max=1.0, step=0.001, decimals=3,
       enable={"key": "analysis/override_px", "truthy": True}),
    _f("imaging", "analysis/override_fi", "bool", "Override frame interval", False),
    _f("imaging", "analysis/frame_interval", "double", "Frame interval (s)", 0.02,
       min=0.001, max=10.0, step=0.001, decimals=3,
       enable={"key": "analysis/override_fi", "truthy": True}),
    _f("imaging", "analysis/channel", "int", "Channel (CZI)", 0, min=0, max=8, step=1),

    # ── Preprocessing ────────────────────────────────────────────────────
    _f("preprocessing", "analysis/bg_method", "combo", "Background method", "Uniform Filter",
       items=["Uniform Filter", "Rolling Ball"]),
    _f("preprocessing", "analysis/bg_radius", "int", "Background radius (px)", 10,
       min=3, max=200, step=1),

    # ── Camera calibration (optional) ────────────────────────────────────
    _f("camera", "analysis/camera_gain", "double", "Gain (e⁻/ADU)", 0.0,
       min=0.0, max=1000.0, step=0.1, decimals=3),
    _f("camera", "analysis/camera_qe", "double", "Quantum efficiency", 0.0,
       min=0.0, max=1.0, step=0.05, decimals=2),
    _f("camera", "analysis/camera_bg_photons", "double", "Background (photons/px)", 0.0,
       min=0.0, max=1e6, step=1.0, decimals=1),

    # ── Detection ────────────────────────────────────────────────────────
    _f("detection", "analysis/diameter", "int", "Diameter (px, odd)", 7,
       min=3, max=21, step=2),
    _f("detection", "analysis/auto_minmass", "bool", "Auto threshold (per-file)", True),
    _f("detection", "analysis/minmass", "double", "Threshold (minmass)", 1.0,
       min=0.0, max=100.0, step=0.05, decimals=2,
       enable={"key": "analysis/auto_minmass", "truthy": False}),
    _f("detection", "analysis/minmass_sensitivity", "combo", "Sensitivity", "Balanced",
       items=["Strict", "Balanced", "Lenient"],
       enable={"key": "analysis/auto_minmass", "truthy": True}),
    _f("detection", "analysis/minmass_max_false_track_rate", "double",
       "Max false-track rate", 0.0, min=0.0, max=50.0, step=1.0, decimals=1,
       suffix=" %", special="off",
       enable={"key": "analysis/auto_minmass", "truthy": True}),

    # ── Linking ──────────────────────────────────────────────────────────
    _f("linking", "analysis/linker", "combo", "Linker",
       "Kalman filter — TrackMate (Linear Motion)",
       items=["Kalman filter — TrackMate (Linear Motion)",
              "Crocker–Grier — Trackpy",
              "Jaqaman LAP — TrackMate (simple)",
              "Jaqaman LAP — TrackMate (merge/split)",
              "Nearest-neighbour — greedy",
              "Simulated annealing — palmTRACER (inspired)"]),
    _f("linking", "analysis/allow_merging", "bool", "Allow merging (Full LAP)", False,
       enable={"key": "analysis/linker", "eq": _FULL_LAP}),
    _f("linking", "analysis/allow_splitting", "bool", "Allow splitting (Full LAP)", False,
       enable={"key": "analysis/linker", "eq": _FULL_LAP}),
    _f("linking", "analysis/search_range", "int", "Search range (px)", 5,
       min=1, max=30, step=1,
       enable={"key": "analysis/auto_search_range", "truthy": False}),
    _f("linking", "analysis/auto_search_range", "bool", "Auto search range", False),
    _f("linking", "analysis/memory", "int", "Memory (frames)", 3, min=0, max=10, step=1),
    _f("linking", "analysis/min_track_len", "int", "Min track length", 8, min=3, max=50, step=1),
    _f("linking", "analysis/max_track_len", "int", "Max track length (0=off)", 0,
       min=0, max=100000, step=1, special="off"),

    # ── Diffusion & motion ───────────────────────────────────────────────
    _f("diffusion", "analysis/max_lagtime", "int", "Max lag-time", 20, min=5, max=100, step=1),
    _f("diffusion", "analysis/n_fit", "int", "MSD fit points", 5, min=2, max=20, step=1),
    _f("diffusion", "analysis/alpha_immobile", "double", "α immobile", 0.5,
       min=0.0, max=2.0, step=0.01, decimals=2),
    _f("diffusion", "analysis/alpha_confined", "double", "α confined", 0.9,
       min=0.0, max=2.0, step=0.01, decimals=2),
    _f("diffusion", "analysis/alpha_directed", "double", "α directed", 1.1,
       min=0.0, max=2.0, step=0.01, decimals=2),
    _f("diffusion", "analysis/mobile_d", "double", "Mobile D threshold", 0.05,
       min=0.0, max=10.0, step=0.01, decimals=3),
    _f("diffusion", "analysis/jdd_components", "int", "JDD components", 2, min=1, max=4, step=1),
    _f("diffusion", "analysis/filter_d_enable", "bool", "Filter by D", False),
    _f("diffusion", "analysis/filter_d_min", "double", "D min", 0.0,
       min=0.0, max=10.0, step=0.000001, decimals=7,
       enable={"key": "analysis/filter_d_enable", "truthy": True}),
    _f("diffusion", "analysis/filter_d_max", "double", "D max", 1.0,
       min=0.0, max=10.0, step=0.000001, decimals=7,
       enable={"key": "analysis/filter_d_enable", "truthy": True}),

    # ── ROI ──────────────────────────────────────────────────────────────
    _f("roi", "analysis/roi_mode", "combo", "ROI mode", "Auto threshold",
       items=["None", "Auto threshold", "Manual threshold", "Manual polygon",
              "Sister TIFF", "ImageJ ROI"]),
    # Auto method only applies in Auto-threshold mode; the manual Threshold only
    # in Manual-threshold mode; Mask mode + Background σ apply to either
    # threshold mode. Mirrors the legacy _on_roi_mode_changed grey-out.
    _f("roi", "analysis/roi_auto_method", "combo", "Auto method", "Li",
       items=["Li", "Otsu", "Triangle", "Mean"],
       enable={"key": "analysis/roi_mode", "eq": "Auto threshold"}),
    _f("roi", "analysis/roi_threshold", "double", "Threshold", 0.08,
       min=0.0, max=1.0, step=0.005, decimals=3, slider=True,
       enable={"key": "analysis/roi_mode", "eq": "Manual threshold"}),
    _f("roi", "analysis/roi_mask_mode", "combo", "Mask mode", "Max",
       items=["Max", "Blink density", "Mean", "Sum"],
       enable={"key": "analysis/roi_mode", "in": ["Auto threshold", "Manual threshold"]}),
    _f("roi", "analysis/roi_bg_sigma", "double", "Background σ", 25.0,
       min=0.0, max=100.0, step=1.0, decimals=1, slider=True,
       enable={"key": "analysis/roi_mode", "in": ["Auto threshold", "Manual threshold"]}),

    # ── Drift correction ─────────────────────────────────────────────────
    _f("drift", "analysis/drift_correct", "bool", "Correct drift", True),
    _f("drift", "analysis/drift_segment", "int", "Segment (frames)", 500,
       min=50, max=5000, step=50),

    # ── Clustering (DBSCAN) ──────────────────────────────────────────────
    _f("clustering", "analysis/cluster_eps_nm", "double", "eps (nm)", 50.0,
       min=5.0, max=2000.0, step=5.0, decimals=1),
    _f("clustering", "analysis/cluster_min_samples", "int", "Min samples", 10,
       min=2, max=100, step=1),

    # ── Performance ──────────────────────────────────────────────────────
    _f("performance", "analysis/backend", "combo", "Detection backend", "Auto",
       items=["Auto", "Crocker–Grier — Trackpy (CPU)", "Crocker–Grier — PyTorch (GPU)",
              "À trous wavelet — PyTorch (GPU)", "Gaussian MLE — PyTorch (GPU)",
              "Radial symmetry — PyTorch (GPU)"]),
    _f("performance", "analysis/workers", "int", "Workers", _N_CPUS,
       min=1, max=_N_CPUS, step=1),
    _f("performance", "analysis/chunk_size", "int", "Chunk size (frames)", 500,
       min=50, max=5000, step=100),
    _f("performance", "performance/hyperfly", "combo", "HYPER-FLY", "Auto (recommended)",
       items=["Auto (recommended)", "Always on", "Off"], hyperfly=True),
    _f("performance", "performance/hyperfly_max_files", "int", "HF max files (0=auto)", 0,
       min=0, max=999, step=1, special="auto", hyperfly=True),
    _f("performance", "performance/hyperfly_max_cores", "int", "HF max cores (0=all)", 0,
       min=0, max=_N_CPUS, step=1, special="all", hyperfly=True),
    _f("performance", "performance/hyperfly_max_ram", "int", "HF max RAM GB (0=auto)", 0,
       min=0, max=100000, step=8, special="auto", hyperfly=True),
    _f("performance", "performance/hyperfly_load_slots", "int", "HF load slots (0=auto)", 0,
       min=0, max=999, step=1, special="auto", hyperfly=True),
    _f("performance", "performance/hyperfly_gpu_slots", "int", "HF GPU slots (0=auto)", 0,
       min=0, max=8, step=1, special="auto", hyperfly=True),
    _f("performance", "performance/czi_parallel_decode", "bool", "Parallel CZI decode", True),
    # Figure style/batch settings moved to the Preferences › Figures menu.
]

# index by key for O(1) lookup
BY_KEY = {f["key"]: f for f in FIELDS}


def hyperfly_machine_eligible() -> bool:
    """Big-machine gate (mirrors the Widgets HYPER-FLY visibility): ≥32 cores
    AND ≥192 GB RAM."""
    try:
        if _N_CPUS < 32:
            return False
        import psutil
        return psutil.virtual_memory().total >= 192 * (1024 ** 3)
    except Exception:
        return False
