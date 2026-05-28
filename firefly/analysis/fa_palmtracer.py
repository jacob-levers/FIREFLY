"""PALM-Tracer-format CSV export and summary loading.

Extracted from sptpalm_analysis.py (#7); re-exported there for compatibility.
"""
from __future__ import annotations

from firefly.analysis.fa_diffusion import (compute_msd_and_fit, compute_jdd,
                          compute_dwell_times, compute_turning_angles,
                          compute_mobile_fraction_over_time,
                          MOBILE_D_THRESHOLD_DEFAULT)

import json
import os
import numpy as np
import pandas as pd


def _find_stem(data_dir):
    """Find the experiment stem from filenames like {stem}_params.json or
    {stem}_diffusion_summary.csv inside an analysis output folder's data/ dir."""
    for f in sorted(os.listdir(data_dir)):
        if f.endswith("_params.json"):
            return f[:-len("_params.json")]
    for f in sorted(os.listdir(data_dir)):
        if f.endswith("_diffusion_summary.csv"):
            return f[:-len("_diffusion_summary.csv")]
    raise FileNotFoundError(f"No analysis CSVs found in {data_dir}")


def _is_palmtracer_folder(folder):
    """Return True if `folder` contains raw PALM-Tracer output."""
    try:
        names = os.listdir(folder)
    except OSError:
        return False
    # PALM-Tracer files have no stem prefix (e.g. 'locPALMTracer.txt')
    has_loc = any(n.lower() == "locpalmtracer.txt" or n.lower() == "locpalmtracer.csv"
                  for n in names)
    has_trc = any(n.lower() == "trcpalmtracer.txt" or n.lower() == "trcpalmtracer.csv"
                  for n in names)
    return has_loc and has_trc


def _read_palmtracer_table(path, header_lines):
    """Read a PALM-Tracer file (tab- or comma-separated), skipping comment /
    metadata rows.  `header_lines` is the number of non-data leading rows."""
    # PALM-Tracer's reference files are TSV; FIREFLY-emitted ones are CSV.
    # Sniff the separator from the first data line.
    with open(path, "r") as fh:
        for _ in range(header_lines):
            fh.readline()
        first = fh.readline()
    sep = "\t" if "\t" in first and first.count("\t") >= first.count(",") else ","
    return pd.read_csv(path, sep=sep, header=None, comment="#",
                       skiprows=header_lines, engine="python")


def load_summary_from_palmtracer(folder):
    """
    Read a raw PALM-Tracer output folder and return the same dict shape as
    `load_summary_from_folder` so the Compare tab can treat it identically.

    PALM-Tracer does not store FIREFLY-specific quantities (alpha, motion
    class, dwell times, turning angles, JDD, mobile fraction, Rg) — these are
    re-derived on the fly from the imported trajectories using the same
    pipeline functions FIREFLY normally runs.
    """
    # ── Locate the six PALM-Tracer files (tab or csv) ────────────────────
    def _pick(*candidates):
        for c in candidates:
            p = os.path.join(folder, c)
            if os.path.isfile(p):
                return p
        return None

    loc_path = _pick("locPALMTracer.txt", "locPALMTracer.csv")
    trc_path = _pick("trcPALMTracer.txt", "trcPALMTracer.csv")
    d_path   = _pick("trcPALMTracer-AllROI-D.txt", "trcPALMTracer-AllROI-D.csv",
                     "trcPALMTracer-1-D.txt",     "trcPALMTracer-1-D.csv")
    msd_path = _pick("trcPALMTracer-AllROI-MSD.txt", "trcPALMTracer-AllROI-MSD.csv",
                     "trcPALMTracer-1-MSD.txt",     "trcPALMTracer-1-MSD.csv")

    if not (loc_path and trc_path):
        raise FileNotFoundError(f"PALM-Tracer files not found in {folder}")

    # ── Parse loc / trc metadata header (line 2 contains values) ─────────
    pixel_size_um    = 0.106
    frame_interval_s = 0.02
    width = height = n_frames = 0
    try:
        with open(loc_path, "r") as fh:
            _hdr_names  = fh.readline().rstrip("\n").replace(",", "\t").split("\t")
            _hdr_values = fh.readline().rstrip("\n").replace(",", "\t").split("\t")
        meta = {k.strip(): v.strip() for k, v in zip(_hdr_names, _hdr_values)}
        pixel_size_um    = float(meta.get("Pixel_Size(um)", pixel_size_um))
        frame_interval_s = float(meta.get("Frame_Duration(s)", frame_interval_s))
        width    = int(float(meta.get("Width",  0) or 0))
        height   = int(float(meta.get("Height", 0) or 0))
        n_frames = int(float(meta.get("nb_Planes", 0) or 0))
    except Exception:
        pass

    # ── Localisations ────────────────────────────────────────────────────
    # Header rows in loc/trc files: metadata-names, metadata-values, column-names
    loc_df = _read_palmtracer_table(loc_path, header_lines=3)
    loc_df.columns = ["id", "Plane", "Index", "Channel", "Integrated_Intensity",
                      "CentroidX_px", "CentroidY_px", "SigmaX_px", "SigmaY_px",
                      "Angle_rad", "MSE_Gauss", "CentroidZ_um", "MSE_Z_um",
                      "Pair_Distance_px"][:loc_df.shape[1]]
    locs = pd.DataFrame({
        "x":     loc_df["CentroidX_px"].astype(float).values,
        "y":     loc_df["CentroidY_px"].astype(float).values,
        "frame": (loc_df["Plane"].astype(int).values - 1),   # 1-based → 0-based
        "mass":  loc_df["Integrated_Intensity"].astype(float).values,
    })

    # ── Trajectories ─────────────────────────────────────────────────────
    trc_df = _read_palmtracer_table(trc_path, header_lines=3)
    trc_df.columns = ["Track", "Plane", "CentroidX_px", "CentroidY_px",
                      "CentroidZ_um", "Integrated_Intensity", "id",
                      "Pair_Distance_px"][:trc_df.shape[1]]
    tracks = pd.DataFrame({
        "particle": trc_df["Track"].astype(int).values,
        "frame":    trc_df["Plane"].astype(int).values - 1,
        "x":        trc_df["CentroidX_px"].astype(float).values,
        "y":        trc_df["CentroidY_px"].astype(float).values,
        "mass":     trc_df["Integrated_Intensity"].astype(float).values,
    }).sort_values(["particle", "frame"]).reset_index(drop=True)

    # ── Re-derive D, alpha, motion via FIREFLY's own pipeline ────────────
    # This guarantees the Compare tab sees the same column names and
    # identical statistics it would for a native FIREFLY run.
    imsd_df, emsd_series, diff_df = compute_msd_and_fit(
        tracks, pixel_size_um, frame_interval_s, max_lagtime=20, n_fit=5)

    emsd_df = (emsd_series.to_frame("msd_um2")
                          .reset_index(names="lag_frame"))

    # FIREFLY-only metrics — re-derive on the fly
    try:
        jdd = compute_jdd(tracks, pixel_size_um, frame_interval_s)
    except Exception:
        jdd = None
    try:
        dwell_df, _ = compute_dwell_times(tracks, diff_df, frame_interval_s)
    except Exception:
        dwell_df = None
    try:
        ta_deg = compute_turning_angles(tracks)
    except Exception:
        ta_deg = None
    try:
        mobile_frac_df = compute_mobile_fraction_over_time(
            tracks, diff_df, frame_interval_s)
    except Exception:
        mobile_frac_df = None

    stem = os.path.basename(folder.rstrip(os.sep)) or "palmtracer_run"
    if stem.lower().endswith(".pt"):
        stem = stem[:-3]

    # ── Cache the recomputed FIREFLY-only metrics next to the PALM-Tracer
    # files so re-opening this folder in the Compare tab is instant.  The
    # cache lives in <folder>/firefly_extras/ and uses FIREFLY's native
    # CSV/JSON schema.
    try:
        import json as _json
        extras_dir = os.path.join(folder, "firefly_extras")
        os.makedirs(extras_dir, exist_ok=True)
        diff_df.to_csv(
            os.path.join(extras_dir, f"{stem}_diffusion_summary.csv"), index=False)
        tracks.to_csv(
            os.path.join(extras_dir, f"{stem}_trajectories.csv"), index=False)
        locs.to_csv(
            os.path.join(extras_dir, f"{stem}_localisations.csv"), index=False)
        emsd_df.to_csv(
            os.path.join(extras_dir, f"{stem}_ensemble_msd.csv"), index=False)
        with open(os.path.join(extras_dir, f"{stem}_params.json"), "w") as _fp:
            _json.dump({
                "stem":             stem,
                "pixel_size_um":    pixel_size_um,
                "frame_interval_s": frame_interval_s,
                "n_localisations":  int(len(locs)),
                "n_tracks":         int(diff_df.shape[0]),
                "n_frames":         int(n_frames),
                "width":            width,
                "height":           height,
                "source":           "palmtracer (re-derived)",
            }, _fp, indent=2)
        if jdd:
            with open(os.path.join(extras_dir, f"{stem}_jdd.json"), "w") as _fp:
                _json.dump(_to_jsonable(jdd) if "_to_jsonable" in globals() else jdd,
                           _fp, indent=2, default=str)
        if dwell_df is not None and len(dwell_df):
            dwell_df.to_csv(
                os.path.join(extras_dir, f"{stem}_dwell_times.csv"), index=False)
        if ta_deg is not None and len(ta_deg):
            pd.DataFrame({"turning_angle_deg": ta_deg}).to_csv(
                os.path.join(extras_dir, f"{stem}_turning_angles.csv"), index=False)
        if mobile_frac_df is not None and len(mobile_frac_df):
            mobile_frac_df.to_csv(
                os.path.join(extras_dir, f"{stem}_mobile_fraction.csv"), index=False)
    except Exception:
        # Caching is best-effort — never fail the load over a write error
        pass

    return {
        "folder":     folder,
        "stem":       stem,
        "data_dir":   folder,
        "source":     "palmtracer",
        "params": {
            "stem":             stem,
            "pixel_size_um":    pixel_size_um,
            "frame_interval_s": frame_interval_s,
            "n_localisations":  int(len(locs)),
            "n_tracks":         int(diff_df.shape[0]),
            "n_frames":         int(n_frames),
            "width":            width,
            "height":           height,
        },
        "ensemble_msd":          emsd_df,
        "diffusion":             diff_df,
        "tracks":                tracks,
        "jdd":                   jdd,
        "dwell_times":           dwell_df,
        "turning_angles":        ta_deg if ta_deg is not None else None,
        "turning_angles_signed": True,
    }


def load_summary_from_folder(folder):
    """Load all per-experiment summary data from one analysis output folder.

    Accepts any of:
      <run_dir>/                       (containing firefly_extras/ and data/)
      <run_dir>/firefly_extras/        (the FIREFLY-extras directory itself)
      <palm_tracer_folder>/            (auto-detected, re-derived on load)
      <run_dir>/data/                  (PALM-Tracer CSVs from a FIREFLY run)
    """
    import json

    # ── Resolve which directory holds the FIREFLY-native CSVs ────────────
    # 1) <folder>/firefly_extras  (folder is the run dir)
    if os.path.isdir(os.path.join(folder, "firefly_extras")):
        data_dir = os.path.join(folder, "firefly_extras")
    # 2) folder is itself the firefly_extras dir
    elif os.path.basename(folder.rstrip(os.sep)) == "firefly_extras":
        data_dir = folder
    # 3) folder is a PALM-Tracer folder (raw or FIREFLY-emitted CSV mirrors)
    elif _is_palmtracer_folder(folder):
        return load_summary_from_palmtracer(folder)
    # 4) folder is a run dir whose `data/` holds PALM-Tracer CSVs
    elif (os.path.isdir(os.path.join(folder, "data"))
          and _is_palmtracer_folder(os.path.join(folder, "data"))):
        return load_summary_from_palmtracer(os.path.join(folder, "data"))
    else:
        raise FileNotFoundError(
            f"No firefly_extras/ directory and no PALM-Tracer files in {folder}")

    stem = _find_stem(data_dir)
    s = {"folder": folder, "stem": stem, "data_dir": data_dir}

    # Params (frame interval, pixel size, ...)
    params_path = os.path.join(data_dir, f"{stem}_params.json")
    if os.path.isfile(params_path):
        with open(params_path) as f:
            s["params"] = json.load(f)
    else:
        s["params"] = {"pixel_size_um": 0.104, "frame_interval_s": 0.05}

    # Ensemble MSD
    msd_path = os.path.join(data_dir, f"{stem}_ensemble_msd.csv")
    if os.path.isfile(msd_path):
        s["ensemble_msd"] = pd.read_csv(msd_path)
    else:
        s["ensemble_msd"] = None

    # Diffusion summary (per-track D, alpha, motion_class)
    diff_path = os.path.join(data_dir, f"{stem}_diffusion_summary.csv")
    if os.path.isfile(diff_path):
        s["diffusion"] = pd.read_csv(diff_path)
    else:
        s["diffusion"] = None

    # Trajectories (for track length distribution)
    tr_path = os.path.join(data_dir, f"{stem}_trajectories.csv")
    if os.path.isfile(tr_path):
        s["tracks"] = pd.read_csv(tr_path)
    else:
        s["tracks"] = None

    # JDD
    jdd_path = os.path.join(data_dir, f"{stem}_jdd.json")
    if os.path.isfile(jdd_path):
        with open(jdd_path) as f:
            s["jdd"] = json.load(f)
    else:
        s["jdd"] = None

    # Dwell times
    dwell_path = os.path.join(data_dir, f"{stem}_dwell_times.csv")
    if os.path.isfile(dwell_path):
        s["dwell_times"] = pd.read_csv(dwell_path)
    else:
        s["dwell_times"] = None

    # Turning angles — signed degrees (-180..+180°)
    ta_path = os.path.join(data_dir, f"{stem}_turning_angles.csv")
    if os.path.isfile(ta_path):
        _ta_df = pd.read_csv(ta_path)
        s["turning_angles"]        = _ta_df["turning_angle_deg"].values
        s["turning_angles_signed"] = True
    else:
        s["turning_angles"]        = None
        s["turning_angles_signed"] = False

    return s


def save_palmtracer_csvs(out_dir, stem, locs, tracks, diff_df, imsd_df,
                         pixel_size_um, frame_interval_s,
                         width=None, height=None, n_frames=None,
                         mobile_D_threshold=None):
    """
    Emit PALM-Tracer-compatible CSV files alongside FIREFLY's native outputs.

    Files written (all comma-separated, written into `out_dir`):
        <stem>_locPALMTracer.csv              (one row per localisation)
        <stem>_trcPALMTracer.csv              (one row per trajectory plane)
        <stem>_trcPALMTracer-1-D.csv          (per-track D, MSD(0), MSE, LogD)
        <stem>_trcPALMTracer-1-MSD.csv        (per-track MSD curve, jagged)
        <stem>_trcPALMTracer-AllROI-D.csv     (per-track D summary)
        <stem>_trcPALMTracer-AllROI-MSD.csv   (per-track MSD curve, jagged)

    Column ordering, naming and unit conventions follow PALM-Tracer
    (Bordeaux Imaging Center).  ROI is hard-coded to 1 (FIREFLY does not
    sub-ROI tracks).  Fields FIREFLY does not measure (SigmaX/Y, Angle,
    MSE(Gauss), CentroidZ, MSE_Z, Pair_Distance) are filled with the
    PALM-Tracer "unused" sentinels (-1 or 0).
    """
    import csv as _csv
    import numpy as _np
    import pandas as _pd
    import os as _os

    if mobile_D_threshold is None:
        mobile_D_threshold = MOBILE_D_THRESHOLD_DEFAULT

    width    = int(width)    if width    is not None else 0
    height   = int(height)   if height   is not None else 0
    n_frames = int(n_frames) if n_frames is not None else int(
        max(locs["frame"].max() + 1, tracks["frame"].max() + 1))

    print(f"  PALM-Tracer: {len(locs):,} locs, {len(diff_df):,} tracks, "
          f"imsd_df shape {imsd_df.shape if imsd_df is not None else None}")

    # ── 1. locPALMTracer.csv ─────────────────────────────────────────────
    n_loc = len(locs)
    loc_path = _os.path.join(out_dir, f"{stem}_locPALMTracer.csv")
    with open(loc_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["Width", "Height", "nb_Planes", "nb_Points",
                    "Pixel_Size(um)", "Frame_Duration(s)",
                    "Gaussian_Fit", "Spectral"])
        w.writerow([width, height, n_frames, n_loc,
                    pixel_size_um, frame_interval_s, "None", "False"])
        w.writerow(["id", "Plane", "Index", "Channel",
                    "Integrated_Intensity",
                    "CentroidX(px)", "CentroidY(px)",
                    "SigmaX(px)", "SigmaY(px)", "Angle(rad)", "MSE(Gauss)",
                    "CentroidZ(um)", "MSE_Z(um)", "Pair_Distance(px)"])
        frames_l = locs["frame"].values
        xs       = locs["x"].values
        ys       = locs["y"].values
        mass     = (locs["mass"].values if "mass" in locs.columns
                    else _np.zeros(n_loc))
        for i in range(n_loc):
            w.writerow([i + 1, int(frames_l[i]) + 1, i + 1, -1,
                        float(mass[i]),
                        float(xs[i]), float(ys[i]),
                        0.0, 0.0, 0.0, 0.0,
                        -1.0, -1.0, 0.0])

    # ── 2. trcPALMTracer.csv ─────────────────────────────────────────────
    tr_path = _os.path.join(out_dir, f"{stem}_trcPALMTracer.csv")
    # Re-number particles 1..n in PALM-Tracer style
    pid_order  = (diff_df["particle"].values if "particle" in diff_df.columns
                  else sorted(tracks["particle"].unique()))
    pid_to_new = {int(p): i + 1 for i, p in enumerate(pid_order)}
    n_tracks   = len(pid_to_new)

    with open(tr_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["Width", "Height", "nb_Planes", "nb_Tracks",
                    "Pixel_Size(um)", "Frame_Duration(s)",
                    "Gaussian_Fit", "Spectral"])
        w.writerow([width, height, n_frames, n_tracks,
                    pixel_size_um, frame_interval_s, "None", "False"])
        w.writerow(["Track", "Plane", "CentroidX(px)", "CentroidY(px)",
                    "CentroidZ(um)", "Integrated_Intensity", "id",
                    "Pair_Distance(px)"])
        # trackpy.link sets `frame` as the index AND keeps it as a column —
        # pandas refuses to disambiguate in sort_values, so drop the index first.
        tr_sorted = tracks.reset_index(drop=True).sort_values(["particle", "frame"])
        pids      = tr_sorted["particle"].values
        frames_t  = tr_sorted["frame"].values
        xs_t      = tr_sorted["x"].values
        ys_t      = tr_sorted["y"].values
        mass_t    = (tr_sorted["mass"].values if "mass" in tr_sorted.columns
                     else _np.zeros(len(tr_sorted)))
        for k in range(len(tr_sorted)):
            new_id = pid_to_new.get(int(pids[k]))
            if new_id is None:
                continue
            w.writerow([new_id, int(frames_t[k]) + 1,
                        float(xs_t[k]), float(ys_t[k]),
                        -1, float(mass_t[k]), k + 1, 0])

    print(f"  PALM-Tracer: wrote loc + trc; starting D files")

    # ── 3 & 5. D files ───────────────────────────────────────────────────
    D_arr     = diff_df["D"].values
    msd0_arr  = (diff_df["MSD0"].values if "MSD0" in diff_df.columns
                 else _np.zeros(len(diff_df)))
    mse_arr   = (diff_df["MSE"].values  if "MSE"  in diff_df.columns
                 else _np.zeros(len(diff_df)))
    logD_arr  = _np.where(D_arr > 0, _np.log10(_np.where(D_arr > 0, D_arr, 1)),
                          _np.nan)
    mobile_n  = int(_np.sum(D_arr > mobile_D_threshold))
    immob_n   = int(_np.sum(D_arr <= mobile_D_threshold))
    mob_ratio = (mobile_n / immob_n) if immob_n else _np.nan

    d1_path = _os.path.join(out_dir, f"{stem}_trcPALMTracer-1-D.csv")
    with open(d1_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow([f"#Diffusion Coef in um2/s; Linear fit performed on the "
                    f"first points of trajectories"])
        w.writerow([f"#Pixel size= {pixel_size_um}um ; Frame rate= "
                    f"{frame_interval_s}sec"])
        w.writerow(["ROI", "Trace", "D(um2/s)", "MSD(0)", "MSE",
                    "LogD", "Mobile/Immobile", "Tracks"])
        for i, pid in enumerate(pid_order):
            new_id = pid_to_new[int(pid)]
            row = [1, new_id,
                   float(D_arr[i]) if _np.isfinite(D_arr[i]) else "",
                   float(msd0_arr[i]) if _np.isfinite(msd0_arr[i]) else "",
                   float(mse_arr[i]) if _np.isfinite(mse_arr[i]) else "",
                   float(logD_arr[i]) if _np.isfinite(logD_arr[i]) else "",
                   "", ""]
            if i == 0:
                row[6] = mob_ratio if _np.isfinite(mob_ratio) else ""
                row[7] = n_tracks
            w.writerow(row)

    dA_path = _os.path.join(out_dir, f"{stem}_trcPALMTracer-AllROI-D.csv")
    with open(dA_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow([f"#Diffusion Coef in um2/s; Linear fit performed on the "
                    f"first points of trajectories"])
        w.writerow([f"#Pixel size= {pixel_size_um}um ; Frame rate= "
                    f"{frame_interval_s}sec"])
        w.writerow(["ROI", "Trace", "D(um2/s)", "MSD(0)", "MSE"])
        for i, pid in enumerate(pid_order):
            new_id = pid_to_new[int(pid)]
            w.writerow([1, new_id,
                        float(D_arr[i]) if _np.isfinite(D_arr[i]) else "",
                        float(msd0_arr[i]) if _np.isfinite(msd0_arr[i]) else "",
                        float(mse_arr[i]) if _np.isfinite(mse_arr[i]) else ""])

    print(f"  PALM-Tracer: wrote D files; starting MSD files")

    # ── 4 & 6. MSD files (jagged: one column per surviving lag) ──────────
    def _write_msd(path):
        with open(path, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["#MSD(DeltaT) in um2"])
            w.writerow([f"#Pixel size= {pixel_size_um}um ; Frame rate= "
                        f"{frame_interval_s}sec"])
            for pid in pid_order:
                if int(pid) not in imsd_df.columns and pid not in imsd_df.columns:
                    continue
                col = imsd_df[pid] if pid in imsd_df.columns else imsd_df[int(pid)]
                vals = col.values
                finite_idx = _np.where(_np.isfinite(vals))[0]
                if len(finite_idx) == 0:
                    continue
                last = finite_idx[-1] + 1
                row = [1, pid_to_new[int(pid)]]
                row.extend(float(v) if _np.isfinite(v) else ""
                           for v in vals[:last])
                w.writerow(row)

    _write_msd(_os.path.join(out_dir, f"{stem}_trcPALMTracer-1-MSD.csv"))
    _write_msd(_os.path.join(out_dir, f"{stem}_trcPALMTracer-AllROI-MSD.csv"))
    print(f"  PALM-Tracer: all 6 files written successfully")

    return {
        "loc":           loc_path,
        "trc":           tr_path,
        "D_1":           d1_path,
        "D_AllROI":      dA_path,
        "MSD_1":         _os.path.join(out_dir, f"{stem}_trcPALMTracer-1-MSD.csv"),
        "MSD_AllROI":    _os.path.join(out_dir, f"{stem}_trcPALMTracer-AllROI-MSD.csv"),
    }
