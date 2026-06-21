"""Group-averaged single-run analysis panels for the All-panels view.

The All-panels view shows the rich per-run analysis figure (firefly.analysis.
fa_figure.make_figure — panels A..Q).  To show a *group* (a condition with
several replicate folders) we POOL each folder's loaded summary data and call
the real make_figure with the chosen panel subset, then take its ready-cropped
per-panel image.  This reuses the exact matplotlib look — no analysis-core edits.

Spatial/image panels (A Max projection, B/C trajectories, H position density,
L cluster map) are tied to each replicate's real field-of-view and can't be
pooled — they are handled per-replicate elsewhere.  This module covers the
non-spatial distribution/curve panels, which DO pool meaningfully.  The four
"dynamics" panels whose precomputed inputs aren't carried in the loaded summary
(J mobile-fraction-over-time, K jump-distance, P van Hove, Q VACF) are
recomputed here from the pooled tracks via fa_diffusion's own compute_* helpers
(calling the analysis core, never editing it).

Panel letters (fa_figure._LAYOUT):
    A Max Projection            B Trajectories            C Trajectories by D
    D MSD Curves                E Diffusion D dist         F Motion Classification
    G Anomalous Exponent α      H Position Density Map     I Turning Angle dist
    J Mobile Fraction Over Time K Jump Distance dist       L Cluster Map
    M Dwell Time dist           N Moment Scaling Spectrum  O Radial Distribution
    P van Hove                  Q Velocity Autocorrelation
"""
from __future__ import annotations

# fa_figure panels that pool meaningfully across a group's folders.  D/E/F/G/I/
# M/N/O pool directly from the loaded summary; J/K/P/Q are recomputed from the
# pooled tracks (their precomputed inputs aren't carried in the summary).
AVERAGEABLE_LETTERS = {"D", "E", "F", "G", "I", "J", "K", "M", "N", "O", "P", "Q"}
# Spatial maps tied to each replicate's real field of view — overlaying several
# distinct FOVs into one image is not a meaningful average → per-replicate only.
SPATIAL_LETTERS = {"A", "B", "C", "H", "L"}

# Panels drawn in the single accent colour → recolour to the group's colour.
# The others (E/F/G/N) are coloured by MOTION CLASS (red/orange/blue/green) which
# is meaningful, so they are left as the engine drew them.
RECOLOR_LETTERS = {"D", "I", "M", "O"}
_THEME_ACCENT = {"Dark": "#58a6ff", "AMOLED": "#58a6ff",
                 "Light": "#0969da", "Publication": "#333333"}


def _recolor_accent(pil, target_hex, accent_hex):
    """Hue-shift the engine's accent-coloured pixels (the curve/bars) to the
    group colour, preserving their lightness/edges so antialiasing survives.
    The dark panel background (low saturation) and other hues (orange/red fit
    lines, motion classes) are untouched."""
    import numpy as np
    import matplotlib.colors as mc
    from PIL import Image
    rgb = np.asarray(pil.convert("RGB"), dtype=np.float64) / 255.0
    hsv = mc.rgb_to_hsv(rgb)
    a_h = mc.rgb_to_hsv(np.array(mc.to_rgb(accent_hex)))[0]
    t_h, t_s, _ = mc.rgb_to_hsv(np.array(mc.to_rgb(target_hex)))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    dh = np.abs(h - a_h)
    dh = np.minimum(dh, 1.0 - dh)               # circular hue distance
    # the BRIGHT, saturated accent pixels (the curve) — exclude the dark panel
    # background, which is itself a saturated blue-grey (#0d1117).
    mask = (dh < 0.07) & (s > 0.30) & (v > 0.35)
    hsv[mask, 0] = t_h                          # shift hue → group colour
    if t_s > 0:
        hsv[mask, 1] = np.minimum(1.0, hsv[mask, 1] * 1.15)  # a touch more punch
    out = (mc.hsv_to_rgb(hsv) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _pool_summaries(summaries):
    """Pool a list of loaded folder summaries into make_figure inputs.

    Distributions (diffusion/turning angles/dwell) are concatenated → a pooled
    distribution; the ensemble-MSD curve is averaged across folders.  Track
    `particle` ids are offset per folder so they don't collide.
    """
    import numpy as np
    import pandas as pd

    def _concat(key, offset_particle=False):
        chunks = []
        off = 0
        for s in summaries:
            df = s.get(key)
            if not isinstance(df, pd.DataFrame) or not len(df):
                continue
            if offset_particle and "particle" in df.columns:
                df = df.copy()
                p = pd.to_numeric(df["particle"], errors="coerce")
                df["particle"] = p + off
                off = int(p.max()) + 1 + off if p.notna().any() else off
            chunks.append(df)
        return pd.concat(chunks, ignore_index=True) if chunks else None

    def _pool_tracks_diff():
        # Pool tracks AND diffusion with the SAME per-folder particle offset, so
        # a track's particle id still maps to its diffusion row after pooling.
        # (compute_mobile_fraction_over_time merges tracks↔diff on `particle`; an
        # inconsistent offset would silently misalign D values with tracks.)
        tr_chunks, df_chunks = [], []
        off = 0
        for s in summaries:
            tr, df = s.get("tracks"), s.get("diffusion")
            local_max = -1
            for d in (tr, df):
                if isinstance(d, pd.DataFrame) and len(d) and "particle" in d.columns:
                    p = pd.to_numeric(d["particle"], errors="coerce")
                    if p.notna().any():
                        local_max = max(local_max, int(p.max()))
            for d, bucket in ((tr, tr_chunks), (df, df_chunks)):
                if isinstance(d, pd.DataFrame) and len(d):
                    d = d.copy()
                    if "particle" in d.columns:
                        d["particle"] = pd.to_numeric(d["particle"], errors="coerce") + off
                    bucket.append(d)
            if local_max >= 0:
                off += local_max + 1
        tracks = pd.concat(tr_chunks, ignore_index=True) if tr_chunks else None
        diff = pd.concat(df_chunks, ignore_index=True) if df_chunks else None
        return tracks, diff

    def _avg_emsd():
        # make_figure's MSD panel reads emsd.index as lag-frames (lt = index*dt)
        # and plots emsd.values — i.e. it wants a SINGLE msd column indexed by
        # lag.  The loaded CSV has a [lag, msd] pair, so we extract the msd
        # column, average it across folders (aligned by lag), and return one
        # column indexed by sequential frame number.
        chunks = [s.get("ensemble_msd") for s in summaries
                  if isinstance(s.get("ensemble_msd"), pd.DataFrame)
                  and len(s.get("ensemble_msd"))]
        if not chunks:
            return None
        sample = chunks[0]
        msd_col = next((c for c in sample.columns if "msd" in str(c).lower()),
                       sample.columns[-1])
        series = []
        for c in chunks:
            if msd_col not in c.columns:
                continue
            # align by ROW POSITION (lag 1,2,3…) not lag value — folders share the
            # same lag sequence but float lag_times may not match exactly, which
            # would otherwise make a sparse NaN union and a broken curve.
            series.append(pd.to_numeric(c[msd_col], errors="coerce")
                          .reset_index(drop=True))
        if not series:
            return None
        avg = pd.concat(series, axis=1).mean(axis=1)   # mean over folders per lag
        out = avg.to_frame(name="msd_um2")
        out.index = range(1, len(out) + 1)             # frame numbers → lt = index*dt
        return out

    ta = []
    for s in summaries:
        v = s.get("turning_angles")
        if v is not None:
            arr = np.asarray(v, dtype=float)
            if arr.size:
                ta.append(arr)
    turning = np.concatenate(ta) if ta else None

    params = next((s.get("params") for s in summaries if s.get("params")), {}) or {}
    try:
        px = float(params.get("pixel_size_um", 0.106) or 0.106)
    except Exception:
        px = 0.106
    try:
        fi = float(params.get("frame_interval_s", 0.05) or 0.05)
    except Exception:
        fi = 0.05

    tracks, diff = _pool_tracks_diff()
    return {
        "diff": diff,
        "tracks": tracks,
        "emsd": _avg_emsd(),
        "turning_angles": turning,
        "dwell": _concat("dwell_times"),
        "pixel_size": px,
        "frame_interval": fi,
    }


def render_group_panels(folders, letters, theme="Dark", proj_cmap="Inferno",
                        traj_bg=True, group_color=None):
    """Pool `folders` and render the requested make_figure `letters` for the
    group.  Returns {letter: PIL.Image}.  Returns {} on any failure (the caller
    falls back to a placeholder / the existing renderer).

    Only the averageable distribution/curve panels are rendered.  make_figure is
    monolithic (it draws every panel and needs valid inputs even for ones it
    won't show), so we feed it the pooled distributions for the panels we want
    and lightweight stand-ins for the rest:
      * `stack`/`tracks` — tiny dummies (the averageable panels don't use them;
        only the undisplayed spatial panels do, and they just draw the dummy).
      * `imsd`/`mobile_frac` — empty DataFrames (their panels guard on length).
      * `dwell_tau` — NaN (the dwell panel then skips its fit line).
    The chosen `letters` include the polar panel O when requested, so it gets a
    polar axes (an unselected polar panel would crash on a plain scratch axes).
    """
    try:
        import numpy as np
        import pandas as pd
        from firefly.analysis.fa_palmtracer import load_summary_from_folder
        from firefly.analysis import fa_figure

        want = {l for l in letters if l in AVERAGEABLE_LETTERS}
        if not want:
            return {}

        summaries = []
        for f in folders:
            try:
                summaries.append(load_summary_from_folder(f))
            except Exception:
                continue
        if not summaries:
            return {}
        pooled = _pool_summaries(summaries)
        if pooled["diff"] is None and pooled["emsd"] is None \
                and pooled["turning_angles"] is None and pooled["dwell"] is None \
                and pooled["tracks"] is None:
            return {}

        empty = pd.DataFrame()
        stack = np.zeros((2, 8, 8), dtype=np.float32)
        # make_figure's setup needs a tracks frame with a 'particle' column for
        # its (undisplayed-here) spatial panels — a 2-track dummy is enough.
        dummy_tracks = pd.DataFrame({"particle": [0, 0, 1, 1], "frame": [0, 1, 0, 1],
                                     "x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})
        dwell = pooled["dwell"] if pooled["dwell"] is not None else empty
        ta = pooled["turning_angles"]
        if ta is None:
            ta = np.array([], dtype=float)         # O draws empty, never None

        # Dynamics panels J/K/P/Q: their precomputed inputs aren't carried in the
        # loaded summary, so recompute them from the POOLED tracks via the core's
        # own helpers (calling, never editing).  Each returns None/empty on too
        # little data → that panel is pruned from the request so it isn't drawn.
        jdd = van_hove = vacf = None
        mobile_frac = empty
        tracks = pooled["tracks"]
        if tracks is not None and len(tracks) >= 3 \
                and want & {"J", "K", "P", "Q"}:
            from firefly.analysis import fa_diffusion
            px, fi = pooled["pixel_size"], pooled["frame_interval"]
            if "K" in want:
                try:
                    jdd = fa_diffusion.compute_jdd(tracks, px, fi)
                except Exception:
                    jdd = None
            if "P" in want:
                try:
                    van_hove = fa_diffusion.compute_van_hove(tracks, px)
                except Exception:
                    van_hove = None
            if "Q" in want:
                try:
                    vacf = fa_diffusion.compute_vacf(tracks, fi, px)
                except Exception:
                    vacf = None
            if "J" in want and pooled["diff"] is not None and len(pooled["diff"]):
                try:
                    mf = fa_diffusion.compute_mobile_fraction_over_time(
                        tracks, pooled["diff"], fi)
                    if mf is not None and len(mf) >= 2:
                        mobile_frac = mf
                except Exception:
                    pass
        # Prune panels whose recompute produced nothing (their make_figure guard
        # would otherwise leave a blank axes that we'd export as an empty panel).
        if "K" in want and jdd is None:
            want.discard("K")
        if "P" in want and van_hove is None:
            want.discard("P")
        if "Q" in want and vacf is None:
            want.discard("Q")
        if "J" in want and (mobile_frac is empty or not len(mobile_frac)):
            want.discard("J")
        if not want:
            return {}

        # Always render the polar panel O so it gets a polar axes (an unselected
        # polar panel crashes on a plain scratch axes); only export what's asked.
        result = fa_figure.make_figure(
            stack, dummy_tracks, empty, pooled["emsd"], pooled["diff"],
            pooled["pixel_size"], pooled["frame_interval"],
            fig_theme=theme, proj_cmap=proj_cmap, traj_background=traj_bg,
            turning_angles=ta, jdd=jdd, van_hove=van_hove, vacf=vacf,
            mobile_frac_df=mobile_frac, dwell_df=dwell, dwell_tau=float("nan"),
            combined_panels=(want | {"O"}), want_panels=want, output_path=None)
        panels = result.get("panels", {}) or {}
        # Recolour the single-accent panels (MSD/turning/dwell/radial) to the
        # group's colour; leave the motion-class-coloured ones as drawn.
        if group_color:
            accent = _THEME_ACCENT.get(theme, "#58a6ff")
            if str(group_color).lower() != str(accent).lower():
                for l in (set(panels) & RECOLOR_LETTERS):
                    try:
                        panels[l] = _recolor_accent(panels[l], group_color, accent)
                    except Exception:
                        pass
        return panels
    except Exception as exc:                       # pragma: no cover
        print(f"  [group-panels] render failed: {exc}")
        return {}


def pil_to_qimage(pil):
    """Convert a PIL image to a detached QImage (RGBA)."""
    from PySide6.QtGui import QImage
    pil = pil.convert("RGBA")
    data = pil.tobytes("raw", "RGBA")
    return QImage(data, pil.width, pil.height,
                  QImage.Format.Format_RGBA8888).copy()
