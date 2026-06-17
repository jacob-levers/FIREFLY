# FIREFLY Methods Guide — errata & corrected text

The shipped `FIREFLY_Methods_Guide-v2.pdf` has no in-repo source, so it cannot be
regenerated automatically. This file lists the sections whose wording no longer
matches the code (the README and code docstrings are the authoritative,
up-to-date statements) plus the behavioural changes made in this release. Paste
the corrected text into the PDF source when you next rebuild it.

> Convention: **strike** = wording in the current PDF that is wrong/overstated;
> **fix** = the accurate replacement (matches the code as of this release).

---

## §4 — Detection backends

**strike:** "The two backends produce the same localisations to floating-point
tolerance; the choice is purely speed."

**fix:** trackpy (CPU) and the PyTorch backend are both Crocker–Grier-family
centroid localisers, **calibrated to agree** — median centroid disagreement
≈ 0.05–0.10 px (5–10 nm at 100 nm/px), recall ≥ 0.95 vs trackpy — but they are
**not bit-identical**. The PyTorch threshold is taken over *all* bandpass pixels
(trackpy uses non-zero pixels only) and the mass is rescaled by a calibration
constant (`_TP_MASS_SCALE`), so the **total spot count differs slightly**;
PyTorch surfaces a few extra low-quality candidates that `minmass` / track-length
/ ROI filtering remove downstream. (See `TorchBackend`'s docstring in
`firefly/analysis/fa_localize_backends.py`.)

**Add:** the default backend (`Auto`) is **Torch-first** — it picks a healthy
GPU (CUDA → MPS) else the parallel PyTorch-CPU path, and **never auto-selects
trackpy**; trackpy is a deliberate manual choice in the dropdown.

## §6 — Trajectory linking

**strike:** "Detections are linked into trajectories with trackpy's recursive
subnet linker."

**fix:** the **default linker is the Kalman LAP tracker** (TrackMate "Linear
Motion"; `fa_enums.DEFAULT_LINKER = "kalman"`). FIREFLY offers six linkers:
Kalman LAP (default), Crocker–Grier (trackpy), Jaqaman simple LAP, Jaqaman full
LAP (optional merge/split), greedy nearest-neighbour, and a palmTRACER-style
simulated-annealing tracker. trackpy is selectable manually and is also the
replay fallback for pre-linker manifests (so old runs reproduce faithfully).

## §22 — Compute backends and reproducibility

**strike:** "Trackpy (CPU) and Torch (CPU/MPS/CUDA) agree to floating-point
tolerance."

**fix:** they agree to the **calibration tolerance** above (~5–10 nm median, not
float-exact); GPU batch size does not change which spots are found. Detection is
deterministic and reproducible. (Note: on Apple MPS the gaussian-MLE /
radial-symmetry refiners run their linear-algebra/convolution numerics on CPU,
because some Metal kernels mis-compute them.)

## §11 / Glossary — localisation precision

**Add caveat:** the Gaussian-MLE backend's per-spot precision columns
(`loc_sigma_x/y_px`, → `_nm`) are a Fisher-information CRLB computed on the
**per-frame min–max-normalised** preprocessed intensities, **not** photon/ADU
counts. They are therefore a *relative* precision, not a photon-calibrated one,
and are not strictly comparable across frames. The **primary, calibrated
precision estimate is the MSD localisation-error offset**
`loc_sigma_nm = √(MSD₀/4)` (`fa_diffusion`), which is in physical units.

---

## Behavioural corrections in this release (update the relevant §§)

These are **code fixes** that change documented behaviour; the guide's mechanism
descriptions should be updated to match:

- **Drift correction (§7): sign fix.** The redundant cross-correlation solver had
  an inverted shift sign, so `locs − drift` **doubled** the drift instead of
  removing it (the old test only checked the recovered range, which a sign flip
  also satisfies). Fixed and locked by a synthetic-drift regression test that
  asserts both sign and that the per-frame position trend is removed.
- **JDD (§13): localisation-error correction.** The jump-distance CDF now
  subtracts the same static offset `4σ²` the MSD fit removes (`4DΔt + 4σ²`), so
  the JDD `D` is no longer inflated by σ²/Δt and now **agrees with the
  offset-corrected MSD `D`**. (σ is taken from the MSD's median `MSD₀`; it is
  *not* fit, because D and σ are degenerate from single-lag jumps.)
- **Dwell times (§ residence-time): right-censoring.** Residence-time τ is now
  the right-censored exponential MLE (τ̂ = Σdurations / #uncompleted-at-movie-end
  events) instead of an uncensored fit, which **under-estimated** τ for tracks
  still present at the last frame. A `censored` column flags those tracks.
  (Photobleaching truncation is still not corrected — out of scope.)
- **Turning angles / MSS (§14): gap handling.** Both now use only
  frame-contiguous (single-frame) steps, consistent with JDD/van Hove/VACF; a
  memory-bridged gap is no longer mis-counted as one step.
- **Clustering (§19): caveats.** Per-cluster `area_um2` / `density_locs_per_um2`
  are convex-hull (size-biased) quantities, and are **not comparable** for runs
  that exceed the 250k-localisation DBSCAN sub-sampling cap (logged per run).
