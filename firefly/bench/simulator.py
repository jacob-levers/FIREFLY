"""Tool-agnostic sptPALM forward simulator with ground truth.

Produces a raw camera-domain image stack plus ground-truth localisation and
track tables, so any tool (FIREFLY, palmTRACER, TrackMate) can be scored against
the same known truth.  Deterministic given ``SimConfig.seed`` — one
``np.random.default_rng(seed)`` is threaded through motion → photophysics →
camera noise in a fixed order (never the global ``np.random``).

The forward model is intentionally NOT tuned to FIREFLY's detector:
  * PSF: 2D Gaussian, **pixel-integrated** via erf (the correct forward model;
    point-sampling would unfairly flatter centroid localisers).
  * Camera: Poisson shot noise + Gaussian read noise + offset/gain (EMCCD/sCMOS
    generic photon-transfer model).
  * Photophysics: per-emitter OFF↔ON→BLEACHED Markov chain → realistic gaps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import erf

from firefly.bench.config import SimConfig, DiffusionPopulation


@dataclass
class SimResult:
    stack: np.ndarray          # (T, H, W) float32, ADU (camera domain)
    gt_locs: pd.DataFrame      # frame,x,y,photons,emitter_id,population
    gt_tracks: pd.DataFrame    # frame,x,y,particle,population  (particle == emitter_id)
    meta: dict                 # pixel_size_um, frame_interval_s, psf_sigma_px, ...


def _assign_populations(cfg: SimConfig, rng) -> tuple[np.ndarray, list]:
    """Deterministically assign each emitter to a population index."""
    pops = list(cfg.populations)
    fracs = np.array([max(0.0, p.fraction) for p in pops], dtype=float)
    if fracs.sum() <= 0:
        fracs = np.ones(len(pops))
    fracs = fracs / fracs.sum()
    counts = np.floor(fracs * cfg.n_emitters).astype(int)
    # hand any rounding remainder to the largest population
    counts[int(np.argmax(fracs))] += cfg.n_emitters - int(counts.sum())
    ids = np.arange(cfg.n_emitters)
    rng.shuffle(ids)
    pop_of = np.empty(cfg.n_emitters, dtype=int)
    idx = 0
    for k, c in enumerate(counts):
        pop_of[ids[idx:idx + c]] = k
        idx += c
    return pop_of, pops


def _emitter_trajectory(cfg: SimConfig, pop: DiffusionPopulation,
                        x0: float, y0: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """Return (x[t], y[t]) in pixels for one emitter over n_frames."""
    n = cfg.n_frames
    px = cfg.pixel_size_um
    dt = cfg.frame_interval_s
    name = pop.name.lower()
    if name == "immobile" or pop.D_um2_s <= 0 and name not in ("directed",):
        return np.full(n, x0), np.full(n, y0)

    step_px = np.sqrt(2.0 * max(pop.D_um2_s, 0.0) * dt) / px   # per-axis Brownian std

    if name == "confined" and pop.confine_radius_um:
        # Reflecting (clamped) disc around the anchor → sub-diffusion, MSD plateau ≈ R².
        R = pop.confine_radius_um / px
        x = np.empty(n); y = np.empty(n)
        x[0], y[0] = x0, y0
        for t in range(1, n):
            nx = x[t - 1] + rng.normal(0.0, step_px)
            ny = y[t - 1] + rng.normal(0.0, step_px)
            dx, dy = nx - x0, ny - y0
            r = np.hypot(dx, dy)
            if r > R and r > 0:
                nx, ny = x0 + dx / r * R, y0 + dy / r * R
            x[t], y[t] = nx, ny
        return x, y

    # brownian / directed: vectorised cumulative steps (+ constant drift)
    dx = rng.normal(0.0, step_px, n)
    dy = rng.normal(0.0, step_px, n)
    dx[0] = 0.0; dy[0] = 0.0
    if name == "directed" and pop.drift_vel_um_s:
        vx, vy = pop.drift_vel_um_s
        dx[1:] += vx * dt / px
        dy[1:] += vy * dt / px
    return x0 + np.cumsum(dx), y0 + np.cumsum(dy)


def _photophysics(cfg: SimConfig, rng) -> np.ndarray:
    """(T, N) bool array: is emitter ON during frame t.

    Per-emitter 3-state Markov chain OFF(0)↔ON(1)→BLEACHED(2).  Initial state is
    drawn from the OFF/ON steady state so frame 0 is already populated.
    """
    n, N = cfg.n_frames, cfg.n_emitters
    p_on0 = cfg.k_on / max(cfg.k_on + cfg.k_off, 1e-9)
    states = (rng.random(N) < p_on0).astype(np.int8)   # 0=OFF, 1=ON
    on = np.zeros((n, N), dtype=bool)
    for t in range(n):
        on[t] = states == 1
        off_m = states == 0
        on_m = states == 1
        u1 = rng.random(N)
        u2 = rng.random(N)
        nxt = states.copy()
        nxt[off_m & (u1 < cfg.k_on)] = 1
        nxt[on_m & (u2 < cfg.bleach_prob)] = 2
        nxt[on_m & (u2 >= cfg.bleach_prob) & (u2 < cfg.bleach_prob + cfg.k_off)] = 0
        states = nxt
    return on


def _render_emitter(frame: np.ndarray, x: float, y: float, photons: float,
                    sigma: float, rad: int) -> None:
    """Add a pixel-integrated Gaussian (total ≈ `photons`) to `frame` in place.
    x = column, y = row (frame[row, col])."""
    H, W = frame.shape
    cx, cy = int(round(x)), int(round(y))
    x0, x1 = max(0, cx - rad), min(W - 1, cx + rad)
    y0, y1 = max(0, cy - rad), min(H - 1, cy + rad)
    if x0 > x1 or y0 > y1:
        return
    s2 = sigma * np.sqrt(2.0)
    jx = np.arange(x0, x1 + 1)
    jy = np.arange(y0, y1 + 1)
    # Pixel index j is the pixel CENTRE (trackpy/FIREFLY convention), so pixel j
    # spans [j-0.5, j+0.5).  Integrating the Gaussian over that interval makes a
    # spot at continuous position `x` land its centre-of-mass at `x` exactly,
    # matching what the localisers report (no 0.5 px systematic offset).
    gx = 0.5 * (erf((jx + 0.5 - x) / s2) - erf((jx - 0.5 - x) / s2))   # ∫ each column
    gy = 0.5 * (erf((jy + 0.5 - y) / s2) - erf((jy - 0.5 - y) / s2))   # ∫ each row
    frame[y0:y1 + 1, x0:x1 + 1] += photons * np.outer(gy, gx)


def simulate(cfg: SimConfig) -> SimResult:
    """Run the forward model and return the stack + ground-truth tables."""
    rng = np.random.default_rng(cfg.seed)
    H, W, T, N = cfg.height, cfg.width, cfg.n_frames, cfg.n_emitters
    sigma = cfg.psf_sigma()
    rad = max(2, int(np.ceil(4.0 * sigma)))

    pop_of, pops = _assign_populations(cfg, rng)

    # trajectories (T, N)
    margin = rad + 1
    x0 = rng.uniform(margin, W - margin, N)
    y0 = rng.uniform(margin, H - margin, N)
    # Per-emitter brightness spread (lognormal, mean ≈ 1) so the field has dim and
    # bright emitters like real data.
    if cfg.photon_cv > 0:
        s_ln = np.sqrt(np.log(1.0 + cfg.photon_cv ** 2))
        photons_e = cfg.photons_per_emitter * rng.lognormal(-0.5 * s_ln ** 2, s_ln, N)
    else:
        photons_e = np.full(N, float(cfg.photons_per_emitter))
    X = np.empty((T, N)); Y = np.empty((T, N))
    for e in range(N):
        xe, ye = _emitter_trajectory(cfg, pops[pop_of[e]], x0[e], y0[e], rng)
        X[:, e] = xe; Y[:, e] = ye

    on = _photophysics(cfg, rng)   # (T, N) bool

    # render + camera noise, and harvest GT rows for ON & in-field emitter-frames
    stack = np.empty((T, H, W), dtype=np.float32)
    rows_f, rows_e, rows_x, rows_y = [], [], [], []
    for t in range(T):
        sig = np.zeros((H, W), dtype=np.float64)
        ons = np.nonzero(on[t])[0]
        for e in ons:
            xe, ye = X[t, e], Y[t, e]
            if -rad <= xe <= W - 1 + rad and -rad <= ye <= H - 1 + rad:
                _render_emitter(sig, xe, ye, photons_e[e], sigma, rad)
                if 0 <= xe < W and 0 <= ye < H:
                    rows_f.append(t); rows_e.append(e)
                    rows_x.append(xe); rows_y.append(ye)
        rate = cfg.qe * (sig + cfg.bg_photons)
        electrons = rng.poisson(rate).astype(np.float64)
        electrons += rng.normal(0.0, cfg.read_noise_e, size=(H, W))
        adu = cfg.camera_offset + electrons / max(cfg.camera_gain, 1e-9)
        stack[t] = np.clip(adu, 0.0, 65535.0).astype(np.float32)

    emitter_id = np.asarray(rows_e, dtype=np.int64)
    pop_names = np.array([pops[pop_of[e]].name for e in emitter_id]) \
        if len(emitter_id) else np.array([], dtype=object)
    gt = pd.DataFrame({
        "frame": np.asarray(rows_f, dtype=np.int64),
        "x": np.asarray(rows_x, dtype=np.float64),
        "y": np.asarray(rows_y, dtype=np.float64),
        "photons": (photons_e[emitter_id] if len(emitter_id)
                    else np.array([], dtype=float)),
        "emitter_id": emitter_id,
        "population": pop_names,
    })
    gt_tracks = gt.rename(columns={"emitter_id": "particle"})[
        ["frame", "x", "y", "particle", "population"]].copy()

    meta = {
        "pixel_size_um": cfg.pixel_size_um,
        "frame_interval_s": cfg.frame_interval_s,
        "psf_sigma_px": sigma,
        "n_frames": T, "height": H, "width": W,
        "photons_per_emitter": cfg.photons_per_emitter,
        "bg_photons": cfg.bg_photons,
        "populations": {p.name: {"D_um2_s": p.D_um2_s, "alpha": p.alpha}
                        for p in pops},
    }
    return SimResult(stack=stack, gt_locs=gt, gt_tracks=gt_tracks, meta=meta)
