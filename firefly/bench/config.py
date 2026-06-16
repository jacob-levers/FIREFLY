"""Benchmark-harness configuration (Qt-free, stdlib + dataclasses only).

Describes a simulated sptPALM acquisition and the FIREFLY run parameters used to
analyse it.  Everything here is plain data so a config round-trips to/from JSON
without any third-party serialiser.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class DiffusionPopulation:
    """One ground-truth motion class.  Fractions across populations are
    normalised to sum to 1 at simulation time."""
    name: str                      # "immobile" | "confined" | "brownian" | "directed"
    fraction: float
    D_um2_s: float = 0.0           # diffusion coefficient (µm²/s)
    alpha: float = 1.0             # nominal anomalous exponent (informational)
    confine_radius_um: Optional[float] = None             # "confined" disc radius
    drift_vel_um_s: Optional[Tuple[float, float]] = None  # "directed" drift (vx, vy)


# Canonical 4-class mixture used by default.
_DEFAULT_POPULATIONS = (
    DiffusionPopulation("immobile", 0.25, D_um2_s=0.0),
    DiffusionPopulation("confined", 0.25, D_um2_s=0.05, confine_radius_um=0.15),
    DiffusionPopulation("brownian", 0.25, D_um2_s=0.10),
    DiffusionPopulation("directed", 0.25, D_um2_s=0.05, drift_vel_um_s=(0.5, 0.0)),
)


@dataclass(frozen=True)
class SimConfig:
    """Forward-model parameters for one simulated stack."""
    seed: int = 0
    n_frames: int = 2000
    height: int = 128
    width: int = 128
    pixel_size_um: float = 0.106
    frame_interval_s: float = 0.02
    # PSF (Gaussian approximation of the Airy disc)
    na: float = 1.49
    wavelength_nm: float = 660.0
    psf_sigma_px: Optional[float] = None   # None → derived from NA/wavelength
    # camera / photon budget
    photons_per_emitter: float = 800.0     # expected photons/frame for an ON emitter
    bg_photons: float = 20.0               # background photons/px/frame
    camera_offset: float = 100.0           # ADU baseline
    camera_gain: float = 1.0               # ADU per photo-electron
    read_noise_e: float = 1.5              # Gaussian read noise std (electrons)
    qe: float = 0.95                       # quantum efficiency
    photon_cv: float = 0.4                 # per-emitter brightness spread (CV); real
                                           # emitters vary, which is also what makes
                                           # FIREFLY's auto-threshold behave sensibly
    # density / photophysics
    n_emitters: int = 200
    k_on: float = 0.02                     # per-frame OFF→ON probability
    k_off: float = 0.2                     # per-frame ON→OFF probability
    bleach_prob: float = 0.005             # per-frame ON→BLEACHED probability
    populations: Tuple[DiffusionPopulation, ...] = _DEFAULT_POPULATIONS

    def psf_sigma(self) -> float:
        """Effective PSF sigma in PIXELS.  Derived from the Gaussian-PSF
        approximation σ_nm ≈ 0.21·λ/NA (Zhang et al. 2007) when not set."""
        if self.psf_sigma_px is not None:
            return float(self.psf_sigma_px)
        sigma_nm = 0.21 * self.wavelength_nm / self.na
        return sigma_nm / (self.pixel_size_um * 1000.0)


@dataclass(frozen=True)
class RunConfig:
    """FIREFLY analysis parameters for the in-process runner."""
    diameter: int = 7
    minmass: Optional[float] = None        # explicit value, or None → auto (below)
    auto_threshold: str = "linkability"    # when minmass is None: "linkability"
                                           # (estimate_minmass, FIREFLY's best) or
                                           # "builtin" (the lighter peak heuristic)
    percentile: float = 64.0
    search_range: float = 5.0
    memory: int = 3
    min_len: int = 4
    max_lagtime: int = 20
    n_fit: int = 5
    workers: int = 1                       # 1 = deterministic, no spawn cost
    backend: str = "auto"                  # "trackpy" | "torch" | "atrous" |
                                           # "gaussian-mle" | "radial-symmetry" | "auto"
    linker: str = "trackpy"                # "trackpy" | "kalman" | "simple_lap" |
                                           # "full_lap" | "nn" | "sa" (+ legacy
                                           # "lap"); default keeps the historical
                                           # bench behaviour; GUI default is Kalman
    link_params: dict = field(default_factory=dict)
                                           # per-linker knobs: full_lap's
                                           # allow_merging/allow_splitting/
                                           # feature_penalty; sa's seed/cooling/…


# ── JSON round-trip ───────────────────────────────────────────────────────────
def sim_config_from_dict(d: dict) -> SimConfig:
    d = dict(d)
    pops = d.pop("populations", None)
    if pops is not None:
        d["populations"] = tuple(
            DiffusionPopulation(
                **{**p, "drift_vel_um_s": (tuple(p["drift_vel_um_s"])
                                           if p.get("drift_vel_um_s") else None)})
            for p in pops)
    return SimConfig(**d)


def sim_config_to_dict(cfg: SimConfig) -> dict:
    return asdict(cfg)


def load_sim_config(path: str) -> SimConfig:
    with open(path, "r", encoding="utf-8") as fh:
        return sim_config_from_dict(json.load(fh))


def dump_sim_config(cfg: SimConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sim_config_to_dict(cfg), fh, indent=2)
