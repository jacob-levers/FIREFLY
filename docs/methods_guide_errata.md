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

## §4 / §5 — Drosophila Quality-first `minmass`

**strike:** any statement that matching detections per frame removes the
detection threshold as a control-versus-treatment confound, or that the
Drosophila value `0.16` is a photon/ADU-calibrated detection threshold.

**fix:** the Drosophila Neurons preset pins the Crocker–Grier **PyTorch** backend
and uses `minmass = 0.16` as an **empirical FIREFLY-relative assay floor**. Mass
is the integrated brightness produced by FIREFLY's normalised preprocessing and
backend calibration. It is not photons, ADU, an estimated molecule count, or a
universal physical threshold. The floor must be re-established if the backend,
preprocessing, fluorophore, microscope, illumination, exposure, or acquisition
regime changes.

Quality-first operates on candidates inside the same static polygon ROI (or the
same full frame) that enters analysis. It never lowers the assay/model floor to
reach a target number of detections. For its ambiguity check, FIREFLY preserves
the sampled per-frame candidate counts and masses, then independently redraws
positions from a search-range-smoothed, ROI-restricted spatial density with a
small uniform component. This retains ROI geometry and coarse spatial
heterogeneity while destroying real temporal paths and exact recurrent-pixel
identity. Simple frame-order permutation is not used because immobile emitters
and hot pixels would remain spatially recurrent and make the null depend on the
sample's genuine persistence.

At each exact mass threshold, the observed and spatial-null candidates are
linked with the configured search range, memory, and minimum track length. The
policy may raise `minmass` to the lowest stable threshold whose upper null
long-track-participation fraction is below the configured ceiling (10% in the
Drosophila preset), while retaining sufficient observed linked yield. This is
an estimate of **random-link participation among candidates under that tracking
configuration**. It is explicitly **not** a candidate false-positive rate,
false-discovery rate, detection precision, or recall estimate.

After localisation and ROI application, FIREFLY reports full-run areal density,
zero-detection frames, temporal quarters, and the fraction of detections with
multiple feasible next-frame successors. These are QC outputs and do not feed
back into a count quota. A criterion that cannot be resolved is labelled
`unresolved`; excessive full-run assignment ambiguity or a failed QC is labelled
`invalid`. Such runs remain auditable but are excluded from pooled comparison.
The previous Density-matched and Linkability modes remain available for legacy
replay and sensitivity analyses.

**Validation limitation:** this is an internal assay policy, not an externally
validated detector. The supplied ELYRA material contains no blank/negative-
control acquisition and no truth-labelled synthetic-emitter injection series,
so candidate precision, recall and FDR have not been established. It also
currently contains only one treated biological replicate, which cannot validate
a treatment effect. Confirmatory use requires blank controls, realistic
injection-recovery over the observed background/SNR/motion range, and additional
independent control and treated biological replicates.

## §6 — Trajectory linking

**strike:** "Detections are linked into trajectories with trackpy's recursive
subnet linker."

**fix:** the **default linker is Crocker–Grier (trackpy)**, matching
`fa_enums.DEFAULT_LINKER = "trackpy"` and the visible sidebar default. FIREFLY
offers six linkers: Crocker–Grier (default), Kalman LAP (TrackMate "Linear
Motion"), Jaqaman simple LAP, Jaqaman full LAP (optional merge/split), greedy
nearest-neighbour, and a palmTRACER-style simulated-annealing tracker. Trackpy
is also the replay fallback for pre-linker manifests, so old runs reproduce
faithfully.

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
- **MSD / MSS (§12–14): true-frame lag handling.** The default `all_pairs`
  estimator includes every position pair whose actual frame-number difference
  equals lag `L`, including across missing intermediate observations. The
  persisted `contiguous` compatibility mode keeps pairs inside uninterrupted
  observed runs. MSD and MSS share this policy; elapsed frame span, not row
  count, determines which lags can exist. Gapless results are identical under
  both policies.
- **Turning angles / VACF (§14): timestamps.** Turning angles still require
  consecutive single-frame steps. VACF is rebuilt from single-frame velocities
  carrying their start frame and pairs velocities by actual start-frame
  difference; every exported lag includes its contributing pair count.
- **Track geometry / duration (§12): explicit definitions.** `mean_step_um`
  averages only adjacent observations exactly one frame apart.
  `mean_link_displacement_um` averages every adjacent observed link, and
  `mean_link_speed_um_s` averages each link displacement divided by its own
  `Δframe × Δt`. `track_duration_s = (max(frame)-min(frame)) × Δt`; localisation
  count and observed sampling time are distinct. Path length remains the
  observed polyline, so a gap is represented by a straight chord through an
  unknowable missing path.
- **Three time-like quantities, deliberately different (§12/§13).** They are
  easy to confuse and two of them differ by exactly one frame *by design*:
  `track_duration_s = (max(frame) − min(frame)) × Δt` is the elapsed interval
  **spanned**; observed sampling time (`n_observations × Δt`, reported per
  replicate as `mean_observed_time_s`) is the time actually **sampled** and is
  shorter whenever there are gaps; `dwell_time_total_s =
  (max(frame) − min(frame) + 1) × Δt` counts frames **occupied** and is
  therefore one frame longer than the duration of the same track. The `+1`
  belongs to dwell alone: a molecule seen in a single frame occupied that frame
  but spanned no interval. For a gapless track, duration equals the number of
  intervals between localisations × Δt (the conventional definition). All three
  are defined in the in-app glossary (Preferences → Glossary).
- **Exact zero displacement (§12): below resolution, not D=0.** If every valid
  displacement bin is exactly zero, D and α are unavailable,
  `fit_status=below_resolution`, and the α-derived motion class is unclassified.
  These tracks are counted separately and excluded from log-D and
  mobile/immobile threshold denominators.
- **Imported-table calibration (§3): visible-sidebar authority.** For external
  localisation tables, the sidebar pixel size and frame interval are the
  effective calibration. Embedded palmTRACER values are advisory provenance,
  are recorded separately, and produce a logged warning when they disagree.
- **Versioned result contracts.** New runs use manifest schema 4 and metrics
  schema 2. Missing metric metadata means legacy schema 1. Mixed schemas remain
  loadable, but schema-sensitive MSD/D/MSS/VACF and step/speed inference is
  suppressed rather than silently pooled; stable metrics can still compare.
- **Clustering (§19): caveats.** Per-cluster `area_um2` / `density_locs_per_um2`
  are convex-hull (size-biased) quantities, and are **not comparable** for runs
  that exceed the 250k-localisation DBSCAN sub-sampling cap (logged per run).
