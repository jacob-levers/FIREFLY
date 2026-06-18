# Changelog

## v2.76.0

napari removed — bespoke Qt viewers; numpy 2 / Python 3.13.

### Changed

- **Removed napari as a dependency.** The Visualise-tab viewer and both ROI
  editors are now FIREFLY's own Qt-only widgets (`QGraphicsView` / `QImage` /
  `QPainter` + numpy) — no napari, no pyqtgraph, no vispy. This drops napari
  and its tree (magicgui, npe2, superqt, qtpy, dask, lxml_html_clean) and ends
  the private-API coupling that broke across napari versions.
  - **`FireflyViewer`** (`firefly/ui/viewer.py`) — image stack with a timeline
    **playback bar** (scrub + play/pause + adjustable fps) on a unified time
    axis driven by the movie *and* the tracks, so the timeline works even when
    only tracks (no raw movie) are loaded; per-motion-class track polylines with
    moving current-frame position markers; cluster-points overlay; additive
    super-resolution overlay; click-to-inspect; wheel-zoom / drag-pan.
    (Restores the scrub/play/fps that napari's dims slider provided.)
  - **`RoiEditor`** (`firefly/ui/roi_editor.py`) — interactive draggable-vertex
    polygon editing, trackpy detection preview, bandpass "Filtered view",
    max-projection overlay, and the live auto/manual-threshold ROI mask.
- Per-class track **visibility** is now driven by colour-coded checkboxes in
  the Visualise sidebar (replacing napari's layer list); a "Max proj" checkbox
  replaces napari's per-layer eye toggle in the ROI editor.
- **Stack moved to numpy 2 / Python 3.13** (the napari pin is gone):
  `numpy>=2,<3`, `numba>=0.61`, `torch>=2.6,<3` — aligned with `pyproject.toml`
  and the CI test matrix.

### Fixed

- ROI preview no longer crashes on a `.tif` that isn't a clean single TIFF
  (OME / BigTIFF / odd-header / multi-file series) — it falls back to FIREFLY's
  canonical loader, so any file the app can analyse also previews.

## v2.75.0

Interactive track explorer.

### Added

- **Track explorer** (Visualise tab) — a new "Track explorer" sidebar section
  to slice the loaded trajectories interactively:
  - Filter by **D range**, **α range**, **motion class** (Immobile / Confined /
    Brownian / Directed), and **minimum track length**; a live count shows how
    many of the total tracks match.
  - A **sortable table** (Track, D, α, Motion, Length) lists the matches —
    click any column header to sort numerically.
  - **Selecting a row** centres the napari viewer on that track and fills the
    track inspector (start/end frame, net displacement, path length,
    straightness, mass, D, α, motion) — identical to clicking the track in the
    viewer.
  - **Export filtered tracks…** writes the current filtered subset to CSV.
  - Numeric-range filters are NaN-tolerant, so tracks without a diffusion fit
    are not silently dropped.

## v2.74.0

Super-resolution reconstruction.

### Added

- **Super-resolution reconstruction** — a high-resolution render of the
  localisation cloud (2-D histogram on a fine grid, Gaussian-blurred to the
  localisation precision), the canonical PALM/STORM image.
  - Every analysis run now saves `figures/<stem>_superres.png` (controlled by
    `superres_nm` / `superres_blur_nm`, default 20 nm), recorded in the run's
    summary metrics.
  - The **Visualise tab** gains a live "Super-resolution" control group —
    adjustable output pixel size + blur, a "Render" button that overlays the
    reconstruction on the raw image as a napari layer, and "Save PNG…". Uses the
    loaded run's localisations and pixel size.
  - The renderer (`firefly/analysis/fa_render.py`) is a shared, Qt-free,
    unit-tested function used by both paths.

## v2.73.0

Hardening and polish after the v2.72 figure overhaul.

### Added

- **QC warnings for silent caveats.** The run's QC flags now surface three
  conditions that previously only printed to the console: DBSCAN cluster
  sub-sampling (the 250k-localisation cap — cluster counts/areas reflect the
  sub-sample), a **skipped ROI** (when a polygon was drawn on a differently-sized
  movie and the whole frame was analysed instead), and a **dense field** with
  auto-threshold on (auto minmass is less reliable above ~40 emitters/frame).
- Regression tests for the figure-defaults code that the v2.72 reflow shipped
  untested — `make_figure` panel-selection edge cases (empty / invalid sets,
  height scaling), picker grid formulas matching the real renderers, preview
  asset presence, and the comparison summary band.

### Changed

- The panel-grid rule is now a single shared helper
  (`fa_compare.comparison_grid` / `fa_figure.reflow_grid`) that both the
  renderers and the UI picker's live "N → grid" count use, so they can't drift.
- The single-sample figure preview is fixed real-data example panels, so the
  single-sample theme / projection-colormap dropdowns no longer trigger a no-op
  preview re-render; the preview is now labelled "example data".

## v2.72.1

- **New FIREFLY app icon** — refreshed glossy firefly mark, regenerated for every
  platform: the Windows `.exe` icon (`assets/icon.ico`, 16–256 px), the macOS
  `.app` icon (`assets/icon.icns`), and the runtime window / dock icon
  (`assets/icon.png`).

## v2.72.0

Figure-defaults overhaul and per-figure panel selection.

### Added

- **Figure-defaults preferences reorganised into sub-tabs** — Single-sample /
  Batch / Comparison — each with its own live preview. The LogD-distribution
  style picker moved into the Comparison sub-tab (it's a comparison-figure look
  choice).
- **Panel pickers for both figures** with Select-all / Select-none buttons,
  curated presets (Essential / Diffusion / Dynamics / Spatial / All), and a live
  "N of M → R × C grid" count that mirrors the renderer's real layout.
- **The single-sample combined figure is now panel-selectable.** Previously it
  always drew all 17 panels (A–Q); now the picker chooses which appear and the
  figure reflows into a fresh grid (the default all-panels layout is unchanged).
  Panels P (van Hove) and Q (VACF) are now selectable too.
- **Real-data preview.** The Single-sample preview shows every panel rendered
  from a real example dataset; deselected panels are shown greyscale so you can
  see at a glance which panels the figure will include.
- The Preferences window now opens large enough to show the Figure-defaults page
  without scroll bars (clamped to the screen).

### Changed

- The "Use palmTRACER's own MSD/D" toggle moved out of figure settings into the
  Compare tab's input sidebar, where it belongs (it controls how `.PT` inputs are
  read, not figure styling).

### Fixed

- **Single-sample figure crash on real data.** The combined-figure reflow used a
  local `_pos` that the van Hove panel also assigns, so any analysis with van
  Hove data crashed while drawing later panels. Renamed the reflow's local and
  added a regression test.
- Comparison interaction-plot test updated for the v2.71.0 legend removal (it
  still expected the old bottom legend).

## v2.71.0

Quick-glance summary stats on the comparison figure.

### Added

- **Per-group summary band at the top of the comparison figure.** Each group now
  shows its **trajectory count, median D and median α** at a glance (colour-matched
  to the group), mirroring the individual-analysis stats panel — so condition
  differences are obvious without opening the stats CSV or reading across panels.
  D / α are the median of the per-cell medians (the same per-replicate scalars the
  across-group tests use, so the header agrees with the statistics); the track
  count is the group total. The band is sized in absolute inches with the figure
  grown to fit, so the panels keep their size from 2 up to the 12-group maximum.

### Changed

- **Removed the redundant shared bottom legend** from the comparison figure — the
  new top band already carries the colour ↔ group ↔ n key (plus the stats), so the
  legend only duplicated it. Dropping it frees vertical space for the panels. When
  the bar panels use numbered x-tick tokens (>4 groups), the band entries are
  numbered to match, so the band remains the key for those axes.

## v2.70.0

Scientific review remediation: a **critical** drift-correction sign fix plus a
set of methodological, statistical, and robustness corrections. Several changes
alter reported numbers (drift-corrected positions, JDD `D`, dwell-time τ) — see
`docs/methods_guide_errata.md` for the corresponding methods-doc updates.

### Fixed

- **CRITICAL — drift correction was doubling drift, not removing it.** The
  redundant-cross-correlation solver used `IFFT(F_i·conj(F_j))`, whose peak is
  `(driftᵢ − driftⱼ)`, so the solved per-segment drift came out negated and
  `locs − drift` *added* the motion (≈1.9× on a synthetic ramp). Swapped to
  `IFFT(F_j·conj(F_i))` so the recovered drift is the true sample drift and is
  removed. The previous unit test only checked the recovered range (`max−min`),
  which a sign inversion also satisfies; a new regression test asserts both the
  sign (correlation with the injected ramp) and that the per-frame position
  trend is removed.
- **Two-way mixed ANOVA silently dropped metrics on cross-group cell-name
  collisions.** pingouin rejects subject IDs shared across the between-groups
  factor; cells with the same base name in different groups raised a caught
  `ValueError` and the metric vanished from the report. The subject ID is now
  namespaced by group.
- **Group-comparison Prism CSV over-stated significance.** The headline "P value
  summary" / "significant?" columns were re-derived from the raw p-value,
  ignoring the engine's underpowered (n<3) blanking and the configured α — so an
  uninterpretable comparison could print `****`/"Yes". The CSV now carries the
  engine's α-gated, underpowered-aware stars and labels the α actually used.
- **uint16 background-subtraction underflow.** `_preprocess_fast` /
  `_preprocess_rolling` now cast to float32 first, so a raw integer frame can no
  longer wrap `frame − background` into a bright phantom (the loaders already
  cast, so the hot path is unchanged).
- **Simulated-annealing linker could hang / corrupt tracks.** A swap move could
  create a self-link (`succ[k]==k`) or cycle; the swap is now rejected and the
  chain trace has a hard visited-set cycle guard.

### Changed

- **JDD is now localisation-error-corrected.** The jump-distance CDF subtracts
  the same static offset `4σ²` the MSD fit removes (`4DΔt + 4σ²`, σ taken from
  the MSD median `MSD₀` — it is *not* fit, being degenerate from single-lag
  jumps), so `D_JDD` no longer carries the `σ²/Δt` inflation and now agrees with
  the offset-corrected MSD `D`. Reported as `sigma_loc_um`.
- **Dwell-time τ uses a right-censored exponential MLE.** Tracks still present at
  the final frame are right-censored (flagged via a new `censored` column);
  τ̂ = Σdurations / #completed-events instead of an uncensored CDF fit, which
  under-estimated residence times.
- **Turning angles & MSS now use frame-contiguous steps.** Both previously
  treated a memory-bridged gap as a single step (inconsistent with JDD / van
  Hove / VACF); a gap-spanning step no longer enters a turning angle, and MSS
  pairs positions by true frame separation.
- **Gap-closing memory guard.** The LAP gap-close matrix is now bounded for very
  large segment counts via a KD-tree-gated active subset (provably identical to
  the full dense solve), preventing an OOM on pathologically dense detections.
- **Drift solver hardening.** Segments with no surviving cross-correlation pair
  are interpolated from neighbours instead of being pinned to a spurious zero.
- ROI membership now rounds to the nearest pixel (was truncating, a half-pixel
  boundary bias). Per-cluster `area`/`density` and the DBSCAN sub-sample / hull
  caveats are documented. Stale default-linker docstring and the README stats
  wording corrected to match the code.

## v2.69.3

Fixes the Gaussian-MLE and radial-symmetry detection engines on Apple-Silicon
(MPS) GPUs, where they intermittently mis-localised spots.

### Fixed

- **`gaussian-mle` and `radial-symmetry` engines on MPS.** Apple's Metal backend
  intermittently returns silently-wrong results for the linear-algebra /
  small-kernel ops these refiners use (Gaussian-MLE: `linalg.solve` / `linalg.inv`
  / `einsum`; radial-symmetry: `conv2d` on tiny patches) — the
  "kernel returns garbage with no exception" failure class, which the refiners'
  own `try/except` CPU fallbacks can't catch because nothing is raised. The
  result was occasional wrong spot counts / positions on Apple GPUs while CPU
  stayed correct. The per-spot refinement now runs on **CPU when the device is
  MPS** (via `TorchBackend._refine_off_mps`); detection (bandpass + max-pool over
  full frames) stays GPU-accelerated, and CUDA is unaffected. The refinement is
  cheap (small `k×k` patches), so the cost is negligible. Verified: the MPS path
  now matches the pure-CPU localisation exactly.

## v2.69.2

Linker-dispatch correctness fixes from a full audit of the six linkers and five
detection engines, plus an MPS phantom-detection fix and doc-vs-code cleanups.

### Fixed

- **The Simulated-annealing linker crashed when selected from the GUI.** The SA
  adapter forwarded the GUI's always-present `allow_merging` / `allow_splitting`
  booleans into `link_trajectories_sa()`, which has no such parameters, raising
  `TypeError: ... unexpected keyword argument 'allow_merging'` and aborting
  linking for the whole file. The SA passthrough now only forwards the knobs the
  SA tracker actually accepts.
- MPS bandpass phantom detections — sub-noise-floor bandpass residuals on the
  Apple-GPU path are snapped to zero so they no longer surface as spurious spots.

### Changed

- **Nearest-neighbour is now canonical TrackMate NN** — strictly frame-to-frame
  (`max_gap=1`). It no longer inherits the pipeline-wide `memory` and silently
  bridges 1–3 frame gaps; use the LAP / Kalman linkers for gap-closing.
- **Unified the default linker.** A single `DEFAULT_LINKER = "kalman"` constant
  (`firefly.analysis.fa_enums`) is the forward default for `link_trajectories()`,
  matching the GUI's first-listed entry and the README. The re-ROI /
  pre-linker-manifest replay path still falls back to `trackpy` for replay
  fidelity.
- Feature-penalty docstring corrected: `penalty_weight` is a FIREFLY-specific
  knob (~3× weaker than TrackMate's same-named weight), not the canonical
  TrackMate multiplier.
- Documentation-accuracy pass: the Torch detector is described as *calibrated to
  agree* with trackpy (not step-identical — the percentile population, disk-mask
  radius and max-pool footprint differ slightly and are absorbed by the
  calibration); the LAP gap-close gate is documented as fixed (not
  "diffusion-scaled"); and the à trous / SA / MLE / registry docstrings were
  corrected to match the code.

### Added

- `tests/test_linker_dispatch.py` — regression tests: every linker runs with the
  GUI's default `link_params`; `nn` does not bridge a one-frame gap (while
  `simple_lap` does); the forward default resolves to `kalman`; the
  feature-penalty multiplier `P` is pinned.

## v2.69.1

Disambiguated the two LAP linker names in the Linker dropdown.

### Changed

- The Simple LAP linker is now labelled **"Jaqaman LAP — TrackMate (simple)"**
  (was "Jaqaman LAP — TrackMate") so it reads as clearly distinct from
  **"Jaqaman LAP — TrackMate (merge/split)"** rather than sharing the identical
  name. The stored linker value (`simple_lap`) is unchanged, and a
  settings-migration entry remaps the old label on restore so existing saved
  linker preferences are not silently reset.

## v2.69.0

Two new GPU sub-pixel refinement engines, per-localisation precision, a reworked
auto-threshold, deterministic runs, and a consistent linker naming scheme.

### Added

- **Per-localisation precision.** Every localisation now carries `loc_sigma_x_nm`
  / `loc_sigma_y_nm` (per-axis lateral precision), propagated through linking into
  the trajectory table and written to the exported CSVs. The **Gaussian MLE**
  engine derives a rigorous CRLB from its fit's Fisher information; **trackpy**
  reports its empirical `ep`; and an optional **camera calibration** (gain / QE /
  background, in the Detection panel) enables a Mortensen-2010 CRLB for the
  PyTorch Crocker–Grier / à trous / radial engines. The diffusion summary adds a
  per-track `loc_sigma_meas_nm` alongside the existing MSD-offset estimate — three
  independent precision estimates that should agree.
- **Deterministic execution mode.** Set `FIREFLY_DETERMINISTIC=1` for a
  bit-reproducible run (GPU floating-point determinism + cuBLAS workspace pinned);
  the run manifest records the determinism state and torch / CUDA / numpy versions.

- **Two new detection engines — Gaussian MLE and radial symmetry.** Both share
  the PyTorch Crocker–Grier detection (bandpass → percentile → max-pool maxima)
  and swap only the sub-pixel refiner, via a new swappable `_refine_peaks` hook on
  the Torch backend:
  - **Gaussian MLE — PyTorch (GPU)** — a maximum-likelihood 2-D Gaussian fit
    (Newton–Raphson with the Fisher/Gauss–Newton Hessian, σ self-fitted) on the
    **raw** pixels, seeded from the centroid result and falling back to it on a
    divergent fit (Smith et al., *Nat. Methods* 2010).
  - **Radial symmetry — PyTorch (GPU)** — the closed-form, fit-free radial-symmetry
    centre estimator (Parthasarathy, *Nat. Methods* 2012).

  Both keep `mass` on the trackpy scale, so `minmass` and the auto-threshold behave
  identically to the other backends, and both inherit the à trous coincident-duplicate
  guard (now a shared helper). **Honest scope:** on isolated, well-bandpassed spots
  centroid-of-mass is already near the CRLB, so these refiners track it closely —
  within ~2 % localisation RMSE on the bundled EPFL/SPT datasets, exact on noiseless
  synthetic spots, with small gains at low SNR and the occasional small loss on dense
  / overlapping fields. They are alternative refiners, not a guaranteed precision
  upgrade.

### Changed

- **Auto-threshold noise floor reworked** for robustness across detectors and
  emitter densities. The detection threshold (minmass) is now floored at a GMM
  noise/signal valley **only when a genuinely separable noise population exists** —
  gated on the mode separation *and* areal crowding. The previous unconditional
  count-knee floor over-thresholded and collapsed recall on detectors that don't
  over-detect at minmass=0 (trackpy) and on dense/low-SNR data. Validated across 33
  engine × dataset × density × SNR cases (real EPFL/SPT + a simulated grid).

- **Linkers renamed to the detection-engine style "Algorithm — Software"** for a
  consistent menu: *Kalman filter — TrackMate (Linear Motion)*, *Crocker–Grier —
  Trackpy*, *Jaqaman LAP — TrackMate* (+ a *(merge/split)* variant), *Nearest-neighbour
  — greedy*, and *Simulated annealing — palmTRACER (inspired)*. The stored linker
  values are unchanged; saved preferences for the old labels migrate automatically on
  settings restore.

## v2.68.0

New trajectory linkers and an auto search-range, a simplified GPU-backend
picker, and an internal decoupling of the PyTorch detector's auto-threshold
from trackpy.

### Added

- **Six trajectory linkers** via a new linker registry. Alongside the existing
  **Kalman** (default) and **trackpy** linkers: **Simple LAP** (TrackMate's
  Jaqaman two-step global assignment), **Full LAP** (Simple LAP plus optional
  merge/split events and intensity feature penalties — TrackMate's full tracker),
  **Nearest-neighbour** (greedy per-frame), and a **palmTRACER-style
  simulated-annealing** tracker — an independent reimplementation of the published
  Racine & Sibarita method (palmTRACER itself is closed-source). The **Linker**
  dropdown lists all six; the choice persists between sessions and is recorded in
  each run's manifest.
- **Auto search-range** — an opt-in **Auto** checkbox beside the Search-range
  field estimates the linking distance per file from the motion: it sweeps
  candidate ranges and takes the smallest at which tracks stop fragmenting, capped
  below the inter-spot spacing. Off by default; the biggest accuracy win is on
  fast motion, where a fixed default is usually too small.

### Changed

- **Detection-backend dropdown simplified.** The four PyTorch Crocker–Grier device
  options (GPU-auto / CUDA / MPS / CPU) collapse to one **"PyTorch (GPU)"** that
  auto-selects this machine's GPU — NVIDIA CUDA on Windows/Linux, Apple MPS on
  macOS — and à trous is labelled **"À trous wavelet — PyTorch (GPU)"**. The
  explicit per-device pins remain valid programmatically; saved preferences for
  them migrate automatically on settings restore.

### ⚠ Results that can change on identical data (reproducibility)

- **PyTorch / Auto detection backend with the auto-threshold.** The
  auto-threshold's candidate harvest now runs through the run's **own** backend
  instead of always trackpy, so the chosen minmass is in that backend's native
  mass units (no fixed trackpy→Torch mass-scale transfer). A torch / à trous run
  using the auto-threshold can pick a slightly different minmass than v2.67.x —
  generally **more** reliable on noisy data, where a new per-frame harvest density
  cap keeps the signal/noise split robust despite Torch's permissive `minmass=0`
  detection. Trackpy-backend runs and runs with a manual minmass are unchanged.

### Fixed

- **palmTRACER `.trc` trajectory files now load** for analysis / scoring — header
  detection is preset-aware (a palmTRACER `trc`'s metadata header and data header
  have the same column count, which the old width-only rule picked wrong), and a
  `Track → particle` mapping was added so the trajectories import directly.
- A **re-ROI** (post-process) now also reproduces the original run's **linker and
  its parameters** — previously it silently reverted to trackpy — completing the
  post-process parameter-persistence fix from v2.67.0.

## v2.67.1

### Fixed: palmTRACER comparisons showed nearly everything "Unclassified"

When a palmTRACER folder was loaded with **"use palmTRACER's native D/MSD"**
(`use_native`), FIREFLY took D from palmTRACER's own `-D` file but **blanked the
anomalous exponent and motion class** for *every* track — so a Compare's **Motion
Class Fractions** panel read 80–100% "Unclassified" (and groups whose folders all
went this way, e.g. an all-palmTRACER Propofol/Pre, showed an empty bar). The
motion class is computable from the same trajectories FIREFLY already parses (it
derives JDD / dwell / turning / mobile-fraction from them); there was no reason to
discard it.

Now the `use_native` path **keeps FIREFLY's alpha / motion / loc-σ / Rg** and lets
palmTRACER's native values override **only** the D / MSD / LogD / AUC family — so
the D graphs still reproduce palmTRACER exactly while motion classes are populated.
Existing **already-cached** palmTRACER summaries that were blanked this way
**self-heal on load**: the app re-derives motion from the cached trajectories
(keeping the native D) and rewrites the cache once. FIREFLY-localised runs are
unaffected.

## v2.67.0

A large correctness, robustness and honesty pass — the result of a multi-round
adversarial self-review spanning the figure/data pipeline, the GUI's status
reporting, the statistics, and the updater. Most changes make the app fail
*loudly* instead of silently, or recover a complete run that older versions
would have thrown away. A few change the **numbers** on the same input — always
toward the app's own canonical definitions; those are called out first.

### ⚠ Results that can change on identical data (reproducibility)

Re-running data analysed on v2.66.x can produce different values in these cases.
None is a regression — each fixes a definition that disagreed with itself — but
if you compare against archived outputs, expect:

- **Mobile fraction (headline).** Now computed over tracks with a finite,
  positive diffusion coefficient using `D ≥ threshold`, instead of over *all*
  tracks (which diluted the fraction with failed fits). The headline now agrees
  with the figure's own mobile-fraction panel. The new value is always ≥ the old
  one; per-track D values in the CSV are unchanged.
- **Post-process (re-apply ROI).** A re-ROI now faithfully reproduces the
  original run, changing only the ROI. Previously it silently reverted to default
  settings: the frame-interval default on this path was 0.03 s (vs 0.02 s
  everywhere else, rescaling every D by 1.5×), and it always used the default
  pixel size, frame interval **and linking / MSD parameters** (search range,
  memory, track-length limits, max lag time, fit points, detection diameter)
  instead of the values the original run actually used — so which tracks formed,
  and their D, could differ from the run being re-ROI'd. It now carries all of
  them across from the run's saved parameters.
- **Comparison statistics.** The cross-metric family now actually tests
  `median_D` and `median_alpha` (previously declared but never reported), and the
  parametric-vs-nonparametric choice is made once per comparison instead of
  per pair — so some pairwise tests and corrected p-values can change. Raw and
  within-metric p-values are unchanged.

Metadata-less inputs (a TIF/CZI with no embedded calibration and no override, or
a comparison folder missing its params file) now assume the unified
0.106 µm / 0.02 s default instead of older per-path values (0.104 / 0.05). Files
that carry their own calibration are unaffected.

### A failed figure no longer discards a finished run

The single-run and comparison pipelines saved the figure *before* the data CSVs,
unguarded — so an exception while rendering a multi-minute run threw the whole
result away. Core CSVs are now written first and figure rendering is guarded: a
render error logs a warning and the data still lands.

### The GUI stops painting failures green

A zero-trajectory run and an all-failed batch used to render in success-green
with empty stats. They now show a warning/danger banner pointing at detection /
ROI / calibration. A genuine crash now reports as a crash (the worker's exit
code reflects failure), while a clean finish whose summary didn't reach the UI is
reported as "completed", not a crash — the status tells the truth in every case.

### Malformed external CSVs degrade gracefully

A blank / NA / footer cell, a mis-detected delimiter, or ragged rows used to
raise an opaque error or silently produce all-NaN coordinates. Bad cells and rows
are now coerced/dropped with a reported count, and a non-finite or non-positive
pixel size is rejected with a clear message.

### Crash-free, more coherent statistics

Antipodal / degenerate angle sets no longer raise an uncaught error that voided
an entire circular comparison. Short-track motion classification no longer emits
an over-confident label from an unidentifiable 3-point fit (it returns
"Unknown"), and the JDD 3-component fit can no longer report a negative
population fraction.

### Safer updater and GPU setup

The updater now refuses to install a release asset that lacks a verifiable
`sha256` digest (instead of trusting a size + header check); install paths are
passed to the update helper without shell interpolation; and the CUDA sidecar
loader rejects non-local / world-writable locations before adding them to the
import path.

### Smaller fixes

Unified px/Δt calibration constants into one module; atomic (tmp + replace)
sidecar/JSON writes; results-panel values no longer clip under Windows
fractional scaling and the panel scrolls instead of overlapping in a small
window; auto-threshold audit legends no longer cover the plotted data; TrackMate
files with a multi-row preamble parse correctly; an all-zero projection panel no
longer renders blank.

## v2.66.2

### Separate figure settings for batch / HYPER-FLY runs

A multi-file batch (HYPER-FLY included) renders figures in a "fast" mode — 110
DPI, no vector PDF, no per-panel PNGs — to stay quick at scale. That behaviour is
now **configurable**: a new **Batch / HYPER-FLY figure** section in the figure
settings gives batch runs their own **PNG DPI**, **save-PDF** and **per-panel**
controls, independent of the single-sample figure.

Defaults are unchanged (110 DPI, no PDF, no panels), so existing batches behave
exactly as before — but if you want full-quality figures for every file in a
HYPER-FLY run, just raise the DPI and tick the PDF / per-panel boxes. Theme,
colormap and the cell-background toggle are shared with the single-sample figure.

## v2.66.1

### The Kalman linker is now the default; LAP dropped from the menu

Following v2.66.0's new selectable linker, the **Kalman** linear-motion tracker
is now the **default**. It preserves track identities through crossings and
directed / fast transport (where nearest-neighbour linking swaps tracks) and
matches trackpy on pure diffusion (validated on real data — identical
diffusion-coefficient distribution). **Trackpy** remains in the **Linker**
dropdown for the fastest, longest-tested path on Brownian-only data.

The **LAP** option has been removed from the dropdown: it was never the best
choice in any regime (trackpy wins pure diffusion, Kalman wins directed motion),
so it only added a confusing third option. It remains available programmatically
(`link_trajectories(..., linker="lap")`) for benchmarking.

## v2.66.0

### New: selectable trajectory linker — a Kalman (linear-motion) tracker for directed / crossing motion

The **Linking** panel now has a **Linker** dropdown with three choices:
**Trackpy** (the default — unchanged), **Kalman** (a constant-velocity,
TrackMate-style "linear motion" tracker), and **LAP** (a global two-step
gap-closing linker).

The **Kalman** linker predicts each particle's next position from its estimated
velocity, so it **keeps track identities through crossings and directed / fast
transport** — exactly where the nearest-neighbour linker swaps or loses tracks.
On simulated ground truth with crossing directed tracks it recovers the
trajectories near-perfectly (point-level Jaccard ≈ 0.99) where trackpy drops to
≈ 0.4. On pure Brownian diffusion it matches trackpy — validated on real
Syntaxin1a data, where it reproduces the same diffusion-coefficient distribution
(KS = 0.01) — so **trackpy stays the default** and Kalman is there for data with
genuine directed motion. It is modestly slower than trackpy at low density and
comparable at high density.

Your linker choice is saved between sessions and recorded in each run's manifest.

## v2.65.10

### Fixed: a batch could hang forever on a corrupt (zero-filled) loc file

A `locPALMTracer.txt` (or any external-localisations table) left **zero-filled**
by an interrupted copy / aborted acquisition — full size on disk but all-NUL —
made the loader fall through to pandas' `sep=None` Python sniffer, which **spins
indefinitely** on such input. The file would sit at *"Reading localisations"*
forever, and in a HYPER-FLY batch it tied up a whole worker slot permanently
(the dashboard tile never went green or red). The loader now probes the first
64 KB and **fails fast with a clear "looks corrupt" message**, so a bad file is
reported as failed and the batch keeps going instead of stalling.

## v2.65.9

### Header polish

- The **"Search parameters…"** field now has a little breathing room below the
  sidebar header's divider line instead of butting right up against it.
- The sidebar header and the tab row are now **pixel-aligned**: the last ~1 px
  offset between their bottom borders (the faint remaining "shelf") is measured
  at runtime and nudged out, so the divider is one continuous line.

## v2.65.8

### Fixed: figures failing with "module 'matplotlib.cm' has no attribute 'get_cmap'"

Figure rendering crashed on the bundled matplotlib (≥ 3.9, which **removed**
`matplotlib.cm.get_cmap`), so every affected run failed right at *"Rendering
figure"* — conspicuously in palmTRACER batches. The cluster-map panel
([fa_figure.py](firefly/analysis/fa_figure.py)) and the preview viewer's mass
colouring ([ui_widgets.py](firefly/ui/ui_widgets.py)) now use the modern
colormap registry (`matplotlib.colormaps[...]`), which works across matplotlib
versions. **This was a rendering bug, not corrupt data** — the localisations
loaded fine; only the figure step failed, so re-running the affected files on
this build produces their figures normally.

## v2.65.7

### Tab row fix (properly this time) — the selected tab no longer floats

v2.65.6's header tweak reduced but didn't remove the problem. The tab bar was
still being forced *taller* than the tabs themselves — and a `QTabBar` doesn't
stretch its tabs to fill the extra space, so the selected tab sat with an empty
gap beneath it, **detached from the content panel** (the "broken" look). The tab
bar now keeps its natural height, so every tab fills it and the selected tab
sits flush against the content; the sidebar title is matched to that same height
so their bottom borders form one continuous line, and the tabs get slightly more
padding. Verified by rendering the real themed widgets offscreen.

## v2.65.6

### Fixed: "Failed to load Python DLL python313.dll" after an in-app update

The recurring crash on the post-update restart — *"Failed to load Python DLL
…\\_MEIxxxxxx\\python313.dll. The specified module could not be found."* — is
fixed. **Root cause:** the updater's relaunch helper is spawned *by* the running
(frozen) app, so it inherited the PyInstaller one-file bootloader's `_MEIPASS2`
marker — the "I'm the re-exec'd child, skip unpacking, reuse this folder"
handshake. The relaunched copy inherited it too, so it skipped unpacking itself
and tried to load `python313.dll` from the *previous* version's temp folder,
which the helper had just deleted → "module could not be found". The helper now
strips `_MEIPASS2` / `_PYI_*` before relaunching, forcing a clean unpack. (This
is why launching FIREFLY by hand always worked — Explorer never sets that
marker.)

> **One-time note:** because the *currently installed* build is the one doing the
> install, updating **to** this version may still show the error once on the
> auto-restart. The new version is already swapped into place, though — just
> click **OK** and reopen FIREFLY. Every update **from** this version onward
> restarts cleanly with no error.

### Tidier header / tab row

The sidebar title ("Analysis Parameters", etc.) and the Import / Analysis / …
tab row are now one shared height, with their text vertically centred and their
bottom borders aligned — instead of the tab bar being force-stretched into a
tall, half-empty box with the title text sitting too high (which read as
"broken"). The tabs also get a little more breathing room.

### Faster palmTRACER loading (C-engine table reader, bit-for-bit identical)

The palmTRACER localisation/trajectory tables (the localisation file is ~45 MB)
are now parsed with pandas' compiled **C** engine instead of the pure-Python
parser, with `float_precision="round_trip"` so the parsed values stay **bit-for-bit
identical** to before. Verified on real data: `DataFrame.equals` is `True`, dtypes
match, and the max absolute difference is `0.0`. On Falcon this cut the read time
**4.9×** for the localisation table (4.11 s → 0.84 s) and **5.7×** for the
trajectory table (0.70 s → 0.12 s), so every palmTRACER open / Compare / batch run
starts faster. If the C tokenizer ever trips on an irregular file it transparently
falls back to the tolerant Python parser, so behaviour is never worse than before.
(The ragged `-D` / `-MSD` reference files use their own dedicated parsers and are
unchanged.)

## v2.65.5

### Fix: update check now distinguishes a GitHub rate-limit from "no internet"

GitHub's unauthenticated API allows only **60 update checks per hour per IP**,
which is shared by every device behind a NAT (e.g. a university network) — so the
check is hit routinely and FIREFLY used to report it as "couldn't reach GitHub.
Check your internet connection." The check now detects the rate-limit response
(HTTP 403/429 with `X-RateLimit-Remaining: 0`) and shows an accurate message —
"Update check rate-limited by GitHub … try again in about N min" — with the actual
reset time, instead of blaming the network. Normal offline/network failures still
show the connection message. ("View release page" continues to work in both cases.)

## v2.65.4

### HYPER-FLY controls hidden on machines that can't run it

The HYPER-FLY settings in the Performance section (the **HYPER-FLY batch** mode
selector plus **Max files / Max cores / Max RAM / Concurrent loads / GPU detect
slots**) are now only shown when the machine clears HYPER-FLY's hardware bar —
**≥32 CPU cores AND ≥192 GB RAM**. On smaller machines HYPER-FLY can never engage,
so those rows are hidden to avoid confusion. (The controls still exist internally
with safe defaults, so analysis is unaffected; on a capable box like the 128-core /
752 GB node they appear as before.)

## v2.65.3

### New: batch palmTRACER "Use palmTRACER's own MSD/D" now renders full outputs

The Batch tab's palmTRACER sub-mode **"Use palmTRACER's own MSD/D"** is now live
(was a placeholder). For each palmTRACER `.PT` folder it renders the standard
FIREFLY figure + PALM-Tracer CSVs + extras using palmTRACER's native
`trcPALMTracer-*-D` / `-MSD` values (no re-tracking) — verified on a real dataset
(6,989 tracks / 372,941 locs → a full 2.5 MB figure). Since palmTRACER data has no
raw image, a single-frame localisation-density projection is synthesised for the
projection panel; α / motion-class stay unclassified (palmTRACER doesn't compute
them); JDD / dwell / turning are re-derived from the palmTRACER tracks. This gives
the same "use palmTRACER's own MSD/D" option for per-folder (single) analysis that
the Compare tab already offers for groups. Outputs go to the batch output folder;
nothing is written into the source `.PT` folder.

## v2.65.2

### Fix: HYPER-FLY tiles showed only "localising" — no live preview

In a HYPER-FLY parallel batch, each file's tile never showed the live frame +
detection dots (single-file runs did). Cause: the per-worker queue adapter
`_HFQueue` implemented `put()` but not `put_nowait()`, and the preview pump emits
frames via `put_nowait()` — so every preview frame raised a silently-swallowed
`AttributeError` and was dropped before tagging/forwarding. Added `_HFQueue.put_nowait`
(delegates to the existing non-blocking `put`), so tiles now render the live
localisation preview on every backend (incl. Torch-CUDA).

### Fix: tab row "shelf"; sidebar scrollbar overflow; preview-window padding

- The Import/Analysis tab row sat ~14px lower than the sidebar "Analysis
  Parameters" header (different heights, both top-anchored), leaving a dark
  "shelf". Both are now pinned to 44px so their bottom edges form one flush line.
- The left sidebar's Performance section no longer clips under the vertical
  scrollbar: the import page now uses the viewport-clamping `_NoHScrollArea`
  (already used elsewhere), form rows wrap when too wide for the narrow sidebar,
  and section headers can shrink instead of forcing the content wider.
- The (now floating) Preview viewer's top control row got padding so it isn't
  flush against the window edges.

## v2.65.1

### New: draw FIREFLY comparison graphs from palmTRACER's own MSD/D

When a Compare group is a palmTRACER `.PT` folder, the new **"Use palmTRACER's own
MSD/D (for .PT inputs)"** checkbox on the Compare tab draws the MSD / LogD / D / AUC
panels straight from palmTRACER's native `trcPALMTracer-*-D` / `-MSD` files instead
of re-deriving them in FIREFLY — so those panels reproduce palmTRACER's numbers
exactly (verified byte-for-byte against a real `.PT` dataset). α and motion-class
aren't in palmTRACER's output, so those panels stay unclassified; JDD / dwell /
turning are still re-derived from the palmTRACER trajectories. Ignored for native
FIREFLY analysis folders. (The native parsers live in `fa_palmtracer` behind
`load_summary_from_folder(..., use_native=True)` and are unit-tested.)

## v2.65.0

### Import/Batch UX overhaul

- **Preview viewer is now a separate window.** The napari preview no longer eats
  ~⅔ of the Import tab — click **Preview viewer** (single mode) or batch
  **Open in viewer** to open it in a reusable, non-modal window. napari spins up
  lazily on first open, so the Import tab is lighter and uncluttered.
- **Batch output folder.** A new **Output folder** field (with Browse) on the
  Batch panel lets you send outputs anywhere; leave it empty for the previous
  `<input folder>/batch_results` behaviour. Each run is still wrapped in its own
  per-stem subfolder.
- **Batch input type: Raw images vs palmTRACER data.** A new **Input type**
  dropdown. "Raw images" scans for `.tif`/`.czi` (and external-loc CSV/TXT) as
  before; **palmTRACER data** descends into `.PT` / analysis folders and queues
  the `locPALMTracer` localisation tables, re-analysing them through FIREFLY's
  full tracking + diffusion pipeline. (Drawing FIREFLY graphs from palmTRACER's
  *native* MSD/D values is a follow-up.)
- **Batch series tree no longer overflows.** The series list is height-capped and
  scrolls internally instead of overlapping the Select-all / Open-in-viewer
  buttons.
- **Sidebar scrollbar gutter.** Reserved the vertical-scrollbar width so the
  Performance section's controls no longer sit under the scrollbar.

### Fix: stray grey "grid" appeared over other tabs after a comparison

Configuring/generating a comparison figure could leave an empty grey matplotlib
axes grid floating over unrelated tabs (e.g. the Analysis page). Cause: the in-GUI
figure-preview renderer called `matplotlib.use("Agg", force=False)`, which is a
no-op once the interactive QtAgg backend is active, so `plt.subplots()` built real
on-screen Qt canvas widgets. The preview now renders fully off-screen via the
object-oriented Agg API (`Figure` + `FigureCanvasAgg`, no pyplot, no global backend
change) and always releases the figure, so no canvas can leak into the UI.

### Fix: in-app updater could still fail to relaunch ("python313.dll")

The Windows relaunch helper wiped **all** `_MEI*` extraction dirs on every relaunch
attempt *and* force-killed the launched process after a timeout — both of which could
strip/interrupt the new onefile build's own extraction, reproducing the
`Failed to load Python DLL python313.dll` error. The helper now clears stale `_MEI`
dirs **exactly once, before launching** (when the old app has already exited, so
nothing live is touched), launches **once**, and **never kills** the relaunched
process; a slow Defender-scanned first extraction is left to finish. If no ready
signal arrives it simply keeps the `.bak` for manual rollback.

### macOS: new batching verified safe

The recursive sub-folder batching and the `\\?\` long-path handling are confirmed
correct on macOS — the long-path code no-ops off Windows at every call site, the
recursive scan is platform-agnostic, and the flattened folder keys stay well under
macOS's 255-byte-per-component limit.

## v2.64.4

### Fix: Compare showed empty panels for deeply-nested output folders (long paths)

The Compare tab read each cell's metric files (`firefly_extras/<stem>_*.csv` / `.json`)
with normal Windows paths. For outputs whose folder name is long — a recursive-batch key
repeated in both the per-stem subfolder **and** every filename (common on RDM trees) — those
paths exceed Windows' 260-char `MAX_PATH`, so `os.path.isfile` returned `False` and every
metric (MSD, diffusion/LogD, tracks, JDD, dwell, turning angles) was treated as missing. The
comparison still "completed" but every panel came up empty / "100% unclassified". FIREFLY now
reads those files through the `\\?\` extended-length form (in `load_summary_from_folder` and
the PALM-Tracer loader), so deep output folders load correctly. Companion to v2.64.2, which
fixed the equivalent limit on output *writes*. Verified: cells whose `firefly_extras` paths
reach ~360 chars now load full MSD/diffusion/track/JDD/dwell data.

## v2.64.3

### Fix: recursive batch scooped up palmTRACER analysis files, not just raw images

Ticking **Include subfolders** and pointing it at a whole experiment tree (e.g. an RDM
`…/Syntaxin1a` folder) queued far more than the raw acquisitions: the scan only skipped
FIREFLY's own `batch_results`/`compare_results`, so it descended into palmTRACER's `*.PT`
and `NN_Analysis…` folders and queued their derived files — `locPALMTracer.txt`/`.csv`
localisation & track tables and non-`-Tracks-` map TIFFs — as if they were raw input. On
one real tree that turned an intended ~1,485-file raw run into ~2,000 items, ~500 of them
derived data (some analysed as raw → meaningless results). Recursive mode now (a) prunes
`*.PT` and `NN_Analysis…` analysis-output folders at every level and (b) restricts the
auto-queue to raw images (`.tif`/`.tiff`/`.czi`), so a whole-tree sweep collects exactly the
raw acquisitions. One-level batch mode is unchanged — it still skips nothing extra and still
accepts external-loc `.csv`/`.txt` tables.

## v2.64.2

### Fix: batch saves failed on deep folders ("FileNotFoundError" past 260 chars)

Writing results into a deep output folder — a nested RDM path plus the sample stem
repeated in both the per-stem subfolder and the filename — could push the output path past
Windows' 260-char MAX_PATH and fail every CSV/figure/`.npy` save with `FileNotFoundError`.
Detection still ran, but the outputs were silently lost. FIREFLY now prefixes the per-file
output directory with the Windows `\\?\` extended-length form, so all writes below it
bypass the 260-char limit regardless of folder depth (paths handed back to the GUI are
stripped to normal form). Verified: a 304-char output path that previously failed now
writes successfully.

### New: "Include subfolders" checkbox for batch input

The batch **Input folder** row now has an **Include subfolders** checkbox: tick it to
recursively scan the chosen folder *and every subfolder* for raw images, so a parent
directory holding many experiment folders can be queued in one pass instead of adding each
directory separately. Our own output dirs (`batch_results` / `compare_results`) and hidden
folders are skipped at every level, and multi-file TIF series are grouped as before.
Unticked keeps the previous behaviour (the folder plus one level of subfolders).

## v2.64.1

### Fix: in-place update could fail to relaunch ("Failed to load Python DLL python313.dll")

After an update, the freshly-swapped exe could fail to start with the PyInstaller
bootloader error `Failed to load Python DLL '…\_MEIxxxxx\python313.dll'. LoadLibrary: The
specified module could not be found.` Root cause: the onefile bundle re-extracts ~1.4 GB
on every launch, and Windows Defender scans the brand-new ~586 MB exe so hard that the
first extraction can take minutes — but the relaunch helper waited only ~100 s and then
`taskkill`ed the bootloader **mid-extraction**, leaving a half-extracted `_MEI` dir whose
`python313.dll` dependency DLLs were missing.

The relaunch helper now (1) clears stale/partial `_MEI` dirs first, and (2) waits for the
app's ready signal **for as long as the launched process stays alive** (i.e. is still
extracting) — reacting immediately if it actually exits/crashes, and only force-killing if
it wedges past a ~10-minute cap. It never kills a healthy-but-slow first extraction. The
download → SHA-256 verify → size/hash-checked swap path is unchanged.

Note: because the running build performs its own update, this fix applies to updates *from*
v2.64.1 onward — the immediate update *to* v2.64.1 still uses the prior helper, so if it
doesn't reappear on its own, launch FIREFLY manually once and give the first extraction a
minute or two.

## v2.64.0

### HYPER-FLY: VRAM-aware GPU detect slots, hard-kill cleanup, faster bulk figures

On a big GPU (e.g. an L40S with ~45 GB) HYPER-FLY now **auto-sizes how many files run
GPU detection at once** from the *current* free VRAM, instead of the hardcoded
one-at-a-time. On a 6-file batch this cut wall-clock ~1.5× (179 s → 121 s) by filling an
otherwise-idle card. It stays a good neighbour: capped by `FIREFLY_HYPERFLY_GPU_SLOT_CAP`
(default 4), never more slots than files, and sized from *free* VRAM so it shares the card
with other users. The planner's RAM budget is coupled to the resolved slot count so higher
GPU concurrency can't over-commit system RAM. `FIREFLY_HYPERFLY_GPU_SLOTS` still overrides.

### Robustness: no more orphaned GPU workers on a hard kill

A forced Stop — or a crash of the batch-worker process — used to orphan its
`ProcessPoolExecutor` children, which kept holding GPU VRAM and RAM until they were killed
by hand. On Windows the worker is now placed in a Job Object flagged
`KILL_ON_JOB_CLOSE`, so the OS tears the whole process tree down with the parent (verified:
a `taskkill /F` of a worker with 15 live children left zero survivors).

### Faster bulk figures

Multi-file batches keep **one figure PNG per file** but drop the vector PDF and per-panel
PNGs and cap DPI to 110 — cheaper rasterisation for mass runs. Single-file runs are
unchanged (full quality). Override with `FIREFLY_BULK_FIGURES`.

### More accurate memory accounting

The per-file peak-RAM estimate that drives HYPER-FLY concurrency is now **float32- and
format-aware** (uncompressed TIF ≈ disk × 4, compressed CZI ≈ disk × 8) instead of a flat
× 3 that under-predicted real peak by ~25% — so the `Max RAM GB` cap is honoured rather
than optimistic. The per-file in-RAM footprint is now surfaced in the HYPER-FLY log. Minor
fixes: a benign double-release of the load semaphore is handled quietly instead of being
swallowed; a mixed-format series uses the conservative (larger) RAM multiplier.

## v2.63.1

### Build fix: Windows release exe (supersedes v2.63.0)

The v2.63.0 Windows build failed at the final rename step — PyInstaller's
onefile launcher spawns a child process that *also* holds `FIREFLY.exe` open,
so the build's post-build smoke test (which boots the exe to verify it starts)
left the file locked and `Move-Item FIREFLY.exe → FIREFLY-Windows.exe` errored
with "the process cannot access the file because it is being used by another
process." The build now kills the whole process tree after the smoke test and
retries the rename past any transient lock (Windows Defender scanning a fresh
200 MB binary). **v2.63.1 is the shippable build of the v2.63.0 changes below
— functionally identical.**

## v2.63.0

### HYPER-FLY: staggered loading — no more all-files-into-RAM-at-once surge

On a big box HYPER-FLY runs many files at once (e.g. 64 files × 2 cores). The
catch: every file in a wave *starts* by loading, so they all preallocated their
full stacks simultaneously — RAM ballooned to **hundreds of GB before any
processing started** (a reported 64 files → ~500 GB), and each file decoded on
only its small per-file core slice (2 cores), so loading dragged on while the
machine sat on a huge RAM footprint. Processing itself was quick; **loading was
the whole wait.**

HYPER-FLY now **staggers the loads**:

- Only a few files **load** at a time (auto ≈ one load per 16 cores); the rest
  wait their turn. A slot frees the moment a file's stack is in RAM, so the next
  file loads while this one moves on to detection. Because detection is quick,
  files free their RAM fast — so **resident RAM ramps up gradually and stays
  low** instead of all K files piling in at once.
- While a file holds a load slot it **decodes on a much bigger core slice**
  (total cores ÷ concurrent-loads, e.g. ~16 instead of 2) — since only a few
  load at once, the box isn't oversubscribed and each load finishes faster.
- New **"Concurrent loads (0 = auto)"** control in the Performance section.
  Lower it to be even gentler on RAM or on shared storage.

Per-file results are unchanged — this is purely scheduling. (Raw load time is
still ultimately bounded by your storage/network bandwidth, but the RAM surge is
gone and processing now overlaps loading instead of waiting for all of it.)

## v2.62.0

### HYPER-FLY can now use an NVIDIA GPU safely

When a GPU backend is selected, HYPER-FLY used to point **every** concurrent
file at the one GPU at once — which would exhaust its VRAM (OOM) and serialise
anyway. HYPER-FLY is now **GPU-aware**: a semaphore gates the **GPU detection**
stage to a few files at a time while **all the CPU stages (load/decode, linking,
MSD, diffusion) stay fully concurrent**. The result is a clean pipeline — one
file detects on the GPU while the others load and link on CPU, then hand the GPU
off.

- **GPU detect slots (0 = auto)** in the Performance section: how many files may
  run GPU detection at once. `0` = auto = **1** (safe for a single card). Raise
  it only if the GPU has VRAM to spare. Ignored entirely for CPU backends.
- CPU backends are unaffected — the gate is a no-op for them.

### Experimental: parallel CZI decode (much faster loading)

Compressed Zeiss CZI stacks (JPEG-XR / Elyra) were decoded on **one core per
file**, so on a many-core box HYPER-FLY's load phase used a fraction of the
machine while the reserved per-file cores sat idle. The new **Parallel CZI
decode** option decodes a file's subblocks across its whole core budget
(imagecodecs' JPEG-XR decoder releases the GIL), turning 1 core/file into many —
a multi-fold faster load on heavily-compressed data, with live per-file
progress in the console.

- **Off by default** (a "Parallel CZI decode (experimental)" checkbox in the
  Performance section) until you've validated it on your own data.
- **Cannot corrupt frames**: each file's parallel decode is spot-checked for
  exact equality against the reference (aicspylibczi) decoder and **silently
  falls back** to the trusted bulk read on any disagreement, odd structure, or
  decode error.

## v2.61.0

### "Auto" detection backend now prefers Torch-CPU over Trackpy

The **Auto** backend used to fall back to **Trackpy** whenever no GPU was
present — a rule from back when Torch-on-CPU ran single-process. Now that
Torch-CPU runs a parallel, multi-process localiser, that rule is obsolete.
Auto now resolves to **GPU when healthy → otherwise parallel Torch-CPU**, and
**never auto-selects Trackpy** (it only falls back to Trackpy if PyTorch isn't
installed at all). Trackpy stays available as a deliberate manual choice in the
dropdown — so on a many-core, GPU-less box (e.g. Falcon) Auto now uses the whole
machine instead of the trackpy path. A saved backend preference still wins.

### HYPER-FLY badge: a proper pill, not a banner

The green "HYPER-FLY · N at once" badge in the header now matches the
**"Update available" pill's shape and size** and the Compare tab's **"Ready to
run" pill's green** (dark text on green). It no longer stretches to the full
header height (it was reading as a chunky banner), and the pulse **animation is
removed** — it's a clean, static pill.

### Smoother collapsible sections (no more scroll-lurch / blank-box)

Expanding a collapsible section (e.g. Compare's "What these terms mean") no
longer makes the surrounding scroll area lurch or flash an empty box. The
expand used to briefly uncap the panel to its full height to measure it, which
made a `QScrollArea` jump the viewport and flash blank before the text painted.
The height is now measured without that full-height pass, so the panel grows
smoothly from the first frame. Fixes every collapsible section app-wide.

### Search range: cytosolic vs transmembrane guidance

The **Search range (px)** control — palmTRACER's "maximum distance" (how far a
particle may move between frames to still be linked) — now spells out the
practical guidance in its tooltip and glossary: **~5 px for cytosolic proteins
(e.g. Munc18), ~3 px for transmembrane proteins (e.g. Syntaxin)**.

## v2.60.1

### HYPER-FLY: visible engaged badge + the name

- Renamed **HYPERFLY → HYPER-FLY** everywhere it's shown (the badge, the
  console/log lines, the Preferences controls and tips).
- The **"HYPER-FLY" badge now actually shows** while a parallel batch runs.
  It was previously tucked into the Import tab's header, so you couldn't see it
  once you switched to the Analysis tab to watch the run. It now lives in the
  **always-visible top header strip**, so it's there on every tab for the whole
  run — a **green, gently-pulsing pill** ("HYPER-FLY · N at once") that switches
  off when the batch finishes. (No lightning emoji.)

## v2.59.1

### Updater: no leftover `FIREFLY.exe.bak` after a clean update

The updater kept a `FIREFLY.exe.bak` next to the exe as a rollback safety net —
but it left it there even after a fully successful update, cluttering the folder
(a visible `FIREFLY.exe.bak` on the Desktop). Now that the copy is SHA-256
verified *and* the relaunch confirms the app actually starts, a successful
update is provably good, so the backup is **removed automatically**. It's kept
only when something fails (a restored backup, or the new build never signals it
started) — i.e. exactly when a manual rollback might be needed.

(Any `.bak` left by an older updater is safe to delete by hand.)

## v2.59.0

### Updater: the post-update relaunch no longer shows a scary (harmless) error

After an update, the very first relaunch of the brand-new exe would sometimes
fail with "Failed to load Python DLL … python313.dll" — because antivirus was
still scanning the freshly-written file and blocked a DLL mid-extraction. The
update itself was fine (opening the app again worked), but the error looked
alarming. The swap helper now relaunches with a **ready-marker handshake**: it
pauses to let AV settle, launches with `SPTPALM_READY_MARKER` set (the app
writes that file once its window is up), and if no "ready" signal appears it
**kills the stuck process and relaunches once** — which succeeds. So the app
just opens, instead of throwing an error you had to click through.

(As with all updater changes, this takes effect for updates *from* a build that
already has it — i.e. once you're on v2.59.0+.)

## v2.58.1

### Clearer message when a download is corrupted

When the integrity check (v2.58.0) keeps failing because the download is being
corrupted in transit, the updater now says so plainly — *"its SHA-256 didn't
match GitHub's … nothing was installed … download manually"* — instead of the
misleading "unexpected format" message. (Also serves as a test build to
exercise the in-app updater's new verification.)

## v2.58.0

### Updater: verify integrity, never install a corrupt build

The in-app updater only ever checked the download's *size*, so a binary that a
flaky network or AV **corrupted while keeping the size** was installed anyway —
the real cause of the "failed to load python3xx.dll" and "decompression
resulted in return code -3" crashes after an update on managed Windows
machines. Now the whole transfer is integrity-checked end to end:

- **Download** — the file's **SHA-256 is verified against GitHub's published
  digest**. A mismatch is retried from scratch, and a persistent mismatch
  fails the update cleanly (with a "finish manually" pointer) instead of
  installing a broken exe.
- **Copy** — the Windows swap helper verifies the **copied exe's SHA-256
  matches the source** (via `certutil`); on mismatch it restores the backup and
  reveals the new exe, rather than relaunching a corrupted file.

Net effect: the updater now either installs a byte-perfect build or stops and
tells you — it can't silently ship a damaged one. (Recovery for an already-
corrupted install: rename the `FIREFLY.exe.bak` the updater left behind back to
`FIREFLY.exe`, or download fresh from the Releases page.)

## v2.57.0

### HYPERFLY: faster file loading + a live console

- **Faster load-to-RAM.** Each concurrent file was loading its TIF with a decode
  pool sized to *all* cores — so 11 files at once spun up ~11×128 ≈ 1,400 decode
  threads that thrashed instead of working (CPU sat low while loading crawled).
  The TIF loader now respects HYPERFLY's per-file core budget (the same one the
  detection pools use), so the threads match the cores and loading is much
  snappier. Single-file (non-HYPERFLY) runs are unchanged.
- **The console no longer goes silent.** During HYPERFLY it now logs a concise
  line as each file changes stage (Loading → Preprocessing → Localising →
  Linking …) plus a periodic "⚡ HYPERFLY working… N/M done · K running"
  heartbeat — so a long load/preprocess phase shows progress instead of looking
  frozen. (Still no per-chunk firehose; warnings/errors and the per-file
  done/failed ledger remain.)

## v2.56.0

### Faster downloads (in-app update + GPU installer)

Big downloads — the ~600 MB update, the CUDA wheel — now pull in **parallel
byte-range segments** instead of a single stream. GitHub's release CDN throttles
per connection, so several connections aggregate to much higher throughput; on a
throttled connection this can be several times faster. It probes for range
support first and **falls back to the single-stream path** automatically if the
server doesn't support it (or for small files). Same integrity checks
(size + format validation, atomic rename) and resume-on-retry as before. Disable
with `FIREFLY_NO_PARALLEL_DOWNLOAD=1` if ever needed.

## v2.55.0

### HYPERFLY: a RAM cap for shared machines

Added **Preferences → Performance → "Max RAM GB (0 = auto)"** — a hard ceiling on
HYPERFLY's peak memory across all concurrent files. Lower it to stay a good
neighbour when other people are on the machine: HYPERFLY simply runs fewer files
at once so the wave never exceeds the cap, and if even two files won't fit under
it, HYPERFLY stands down to ordinary one-file-at-a-time processing. 0 = auto
(bounded only by free RAM, as before). Joins the existing Max-files and
Max-cores throttles. (Env equivalent: `FIREFLY_HYPERFLY_MAX_RAM_GB`.)

## v2.54.0

### Parallel-processing review: robustness + polish

A pass over every multicore path (the methods are otherwise well-chosen):

- **trackpy detection can no longer hang on a dead worker.** Its
  multiprocessing pool was switched to `ProcessPoolExecutor` — a worker that
  fails to start now surfaces as an error and falls back to the serial path,
  instead of the silent forever-hang `Pool.imap` could produce (the same fix
  applied to the Torch-CPU path earlier). Detections are unchanged.
- **Torch detection now runs under `torch.inference_mode()`** — correct for a
  pure-inference pipeline; trims a little memory/overhead, numerically
  identical results.
- Torch CPU **inter-op threads set to 1** (the sequential pipeline never used a
  larger inter-op pool; intra-op threading is unchanged).
- Removed an unused import.

No behavioural change to results — purely robustness and hygiene.

## v2.53.1

### Fix: Performance sidebar scrolling sideways

Expanding the Performance section let the sidebar scroll left/right. Cause: the
dropdowns sized themselves to their **widest** item (e.g. "Torch — GPU (auto
device)"), pushing the form past the sidebar width. Combo boxes now request a
compact width and stretch to fill their column instead (the dropdown still
lists every option in full), and the two HYPERFLY cap labels were shortened —
so the section fits without any horizontal scroll.

## v2.53.0

### HYPERFLY live dashboard — a tile per file

When HYPERFLY runs many files at once, the Analysis cockpit now shows a **grid
of live detection tiles**, one lane per concurrent file. Each tile shows that
file's **live preview** (frame + detected spots), its **stem**, **stage /
progress**, and **spot count** — with a colour-coded border (accent while
running, green ✓ done, red ✗ failed). As a file finishes, its lane is reused by
the next file, so K tiles cycle through the whole batch.

Built to stay light at scale: previews are **downscaled and throttled
(~3 Hz/file)** before crossing to the GUI, and a single shared timer repaints
only the tiles that changed. The dashboard appears automatically when HYPERFLY
engages and is replaced by the normal single-file cockpit for ordinary runs.

## v2.52.0

### HYPERFLY: an "engaged" badge + a readable console

- A blue, gently-pulsing **"⚡ HYPERFLY"** pill now appears next to the
  run-readiness badge whenever a parallel multi-file batch is engaged, and
  switches off when it ends. (Static under Preferences → Reduce motion.)
- **The console is no longer a firehose** when many files run at once. Instead
  of interleaving every per-chunk / progress line from dozens of files, it
  shows a clean per-file **ledger** — `▶ [file] started` … `✓ [file] N locs ·
  N tracks` (or `✗ … failed`) — while still surfacing any genuine warnings or
  errors from each file. (Serial single-file runs keep their full detailed
  log.)

## v2.51.0

### Fix: "Failed to load Python DLL" on managed Windows machines

The Windows build unpacks its bundled Python + libraries to a temp folder on
every launch. On locked-down / managed profiles, `%TEMP%` is often **redirected**
to a quota'd, aggressively-cleaned location (e.g. `…\AppData\Local\Temp\14\`)
where the ~hundreds of MB of DLLs can fail to extract fully — which surfaces as
the bootloader error *"Failed to load Python DLL python313.dll. LoadLibrary: The
specified module could not be found."* (often right after an in-app update).

The build now extracts to a **stable per-user folder, `%LOCALAPPDATA%\FIREFLY\bundle`**
— the same place FIREFLY already keeps its logs / crash reports / downloaded
updates (so it's known-writable on these machines) — instead of the redirected
temp. This should let the app launch reliably on managed Windows boxes.

## v2.50.1

### Fix: Windows update can no longer strand you with a broken install

The Windows updater swapped the exe with **no backup and no verification** — so
a truncated copy or a build that won't start (e.g. "Failed to load Python DLL
python313.dll") left you with a dead install and no way back. The swap helper
now mirrors the macOS path:

- **backs up** the current exe to `FIREFLY.exe.bak` before replacing it;
- **verifies** the copied exe's byte size matches the source (a short copy is
  the usual cause of the missing-DLL bootloader error);
- **rolls back** to the backup automatically if the copy fails or is short, and
  reveals the new exe so you can finish by hand;
- on success, **keeps the `.bak`** so you can roll back manually (rename it) if
  a new build won't launch on your machine.

(A deeper fix — shipping Windows as a one-folder build with no per-launch temp
extraction — is coming next; that removes the root cause of the missing-DLL
error on managed/locked-down machines.)

## v2.50.0

### ⚡ HYPERFLY — high-throughput batch on big machines

On a workstation with lots of cores and RAM, a batch used to crawl through one
file at a time, leaving most of the machine idle during each file's I/O,
linking, and figure stages. **HYPERFLY** processes **several files at once,
RAM-resident**, so the whole box stays busy — batch wall-time drops sharply
without changing any per-file result.

- **Auto-detects** capable machines (≈≥32 cores AND ≥192 GB RAM) and engages
  automatically; a one-line banner says when it's active.
- **Fully automatic concurrency**: picks how many files to run at once and the
  per-file core budget from free RAM and cores, so the wave fits in memory and
  the cores aren't oversubscribed.
- **Manual caps for IT**: Preferences → Performance → *HYPERFLY batch*
  (Auto / Always on / Off) plus **Max concurrent files** and **Max cores**
  (0 = automatic) to throttle resource use on shared machines.
- **Identical results**: files are independent and each writes its own folder —
  only the scheduling changes, so per-file output is exactly the same as the
  serial path. Falls back to serial automatically if anything is unavailable.

(This is the engine; a live per-file preview dashboard is coming next.)

## v2.49.0

### Torch backend on CPU now uses all your cores

- The PyTorch detector, when running on CPU (no GPU), previously processed the
  movie in a **single process** and leaned on intra-op threads — which don't
  scale for the small per-frame ops on a many-core / multi-socket box (≈6% CPU
  on a 128-core machine). It now **fans the work out across processes**, one
  chunk-aligned block per worker, sized so `workers × threads ≈ cores`.
- **Results are identical to the old serial path.** The detector's per-chunk
  percentile threshold is batch-size-stable by design and the loop already
  thresholds per-chunk, so processing whole-chunk blocks in parallel reproduces
  the exact same detections (verified by a parallel-vs-serial equivalence test
  on both the in-RAM and memory-mapped paths).
- Robust by construction: uses `ProcessPoolExecutor` (a dead worker surfaces as
  an error and falls back to the serial path rather than hanging), shares the
  stack zero-copy via shared memory (in-RAM) or re-mmap (disk-backed), and can
  be tuned/disabled with `FIREFLY_TORCH_CPU_WORKERS` / `FIREFLY_TORCH_CPU_MP=0`.
- Note: for CPU-only work the **trackpy** backend (via "Auto") already scaled
  across cores; this brings the Torch-CPU path to parity. GPU paths are
  unchanged.

## v2.48.0

### Fix: crash on many-core Windows machines (≥62 logical CPUs)

- The MSD & diffusion stage built a `ProcessPoolExecutor` sized to
  `os.cpu_count()`. On **Windows that's hard-capped at 61** workers (the
  `WaitForMultipleObjects` 64-handle limit), so any file with ≥5000 tracks on a
  big box (e.g. a 128-core EPYC) raised `ValueError: max_workers must be <= 61`
  before processing a single track. Worker counts for process pools are now
  clamped to 61 on Windows via a shared `safe_process_workers` helper (applied
  at all three `ProcessPoolExecutor` sites). Thread pools / `multiprocessing.Pool`
  are unaffected and keep using every core.

### Detection backend: de-trapped for CPU-only machines

- Renamed the misleading **"Torch (auto)"** option to **"Torch — GPU (auto
  device)"** and made **"Auto"** the default. "Auto" picks a healthy GPU when
  present and the **multi-core trackpy** path when there isn't — whereas forced
  Torch on a no-GPU machine runs **single-process on CPU** and badly under-uses
  a many-core box. Existing installs that had the old default saved are migrated
  to "Auto" on restore (deliberate Torch-GPU users can re-pick it).
- When the Torch backend does land on CPU, the log now says so once and points
  to "Auto"/"Trackpy (CPU)" as the faster CPU-only choice.

## v2.47.1

### Fix: jank-free "What these terms mean" (and every collapsible)

- The Compare tab's **glossary section no longer "teleports" or glitches** when
  opening/closing. The expand animation was measuring the wrong target height:
  for word-wrapped content it read the section's *collapsed* geometry (the
  parent layout hadn't re-flowed yet), slid to a too-short height, then sprang
  to full size at the end. It now measures the true height via the layout's
  `heightForWidth`, so the slide lands exactly — smooth open and close.
- Applies to every `_CollapsibleSection` (Details expanders, sidebar groups),
  but is most visible on the long, word-wrapped glossary.

### Fix: CI Tests workflow

- `test_results_json_twoway_present` now skips when **pingouin** is absent
  (the headless Tests runner installs only the analysis core), matching the
  other two-way ANOVA tests. The release build is unaffected.

## v2.47.0

### Subtle, smooth UI animations (+ Reduce motion)

- Collapsible sections (the "Details" expanders, glossary, sidebar groups) now
  **animate open/closed** instead of snapping.
- The **Results tab cards fade in** as a comparison's results arrive.
- New **Preferences → Appearance → "Reduce motion"** turns all of it off for
  instant, static transitions (takes effect immediately).

Built for high-refresh-rate displays: animations are short, eased, and — most
importantly — each fade's `QGraphicsOpacityEffect` is **removed the moment it
finishes** (a lingering one re-buffers the widget on every repaint, which is the
usual cause of "choppy" Qt UIs). Per-frame work is kept tiny and bounded, and
the heavy bottom tables simply appear rather than animate, so nothing stutters.
All via Qt's built-in animation framework — no new dependencies.

Note: classic Qt widgets animate at ~60 fps (the vsync-native path is QML, not
widgets), so motion is smooth-60 rather than literally 144 fps — but jank-free,
which is what reads as smooth on a high-refresh monitor.

## v2.46.0

### New interactive "Results" tab

A comparison's results now appear in-app in a friendly **Results** tab (after
Compare) instead of only as CSVs on disk:

- **Per-metric verdict cards** — each metric gets a plain-language sentence
  (direction + effect-size magnitude + significance, e.g. "Iso is higher than
  Control — a large difference (g = 1.2 [0.4, 2.0]); statistically significant,
  p = 0.003") with hover-tooltips, and an **expandable** sortable pairwise table
  (raw + corrected p, significance, effect size + CI, n). Underpowered (n<3)
  comparisons are flagged as not interpretable.
- **The comparison figure is embedded** (it was previously only saved to disk) —
  click to open full size.
- **Sortable per-replicate values table** (the per-cell scalars), with
  group-coloured rows.
- **Two-way ANOVA** (paired group×time designs) and **circular-statistics**
  sections when the run produced them.
- Buttons to open the output folder / PDF report / figure / stats CSV, plus
  **"Open a previous comparison…"** to reload any past run's results.

Each comparison now also writes a small machine-readable `{stem}_results.json`
that powers the tab (auto-shown after a run, and loadable later). No new
dependencies; the figure is shown from its PNG (no matplotlib in the GUI
process). 4 new tests (results-JSON round-trip + sanitizer + two-way + an
offscreen Results-view smoke).

## v2.45.0

### LogD style picker: live preview + Overlaid is the new default

- **Overlaid KDEs is now the default** LogD distribution style (was Faceted).
- **Preferences → Figure defaults** now shows a **live preview** of the selected
  style (rendered from a small illustrative example where one “drug” immobilises
  PRE→POST), plus a one-line **“best for”** description that updates as you switch:
  - **Overlaid KDEs** — compare overall shape/peak across a few groups at a
    glance; can get crowded with many groups.
  - **Faceted (per-replicate)** — paired PRE/POST designs + honest replicate
    counts (per-cell median dots); most information-dense.
  - **Ridgeline** — many groups compactly; great for spotting multi-modality;
    exact peak heights harder to compare.
  - **Violins + points** — each group’s spread plus the replicate-level data
    (SuperPlot style).
- The Figure-defaults page is now scrollable so the picker, preview and figure
  knobs never overflow the dialog.
- Fix: the LogD render helpers referenced the UI palette key `TXT_MUTED`; they
  now use the figure palette's `MUT` (with a fallback), so themed muted colours
  are correct and the empty-data branch can't raise.

## v2.44.1

- Moved the **"LogD graph style"** picker from Preferences → Appearance to
  Preferences → **Figure defaults** (it's a figure-look choice). Same behaviour
  and persistence; just a tidier home.

## v2.44.0

### Choose your LogD distribution graph style (Preferences)

**Preferences → Appearance → "LogD graph style"** now lets you pick how the
Compare tab's LogD-distribution panel is drawn:

- **Faceted (per-replicate)** — default; one panel per group, PRE vs POST
  overlaid, with a per-cell median dot strip.
- **Ridgeline** — the classic stacked filled KDEs (now with a per-cell median
  tick on each ridge).
- **Overlaid KDEs** — every group's curve on one axes.
- **Violins + points** — per-group violins with a per-cell median dot strip
  (SuperPlot style).

The choice persists and applies to the next comparison you run.

### Faceted legend no longer overlaps the curves

The faceted panel's key now lives in its own dedicated strip above the facets
(title + a neutral PRE/POST/● per-cell-median/threshold key), so it can never
sit on top of a density curve regardless of how the distribution is shaped — the
previous in-axes placement could overlap broad distributions on real data.

## v2.43.0

### LogD distribution panel: honest, faceted redesign

Replaced the LogD ridgeline (which stacked all groups and showed only pooled
per-track density) with a per-replicate-honest view:

- **Faceted by group/drug**, with **PRE vs POST overlaid** in each facet (PRE
  solid, POST dashed, in their card colours) — so the paired shift is read
  directly instead of hunting across a stack.
- A strip of **per-cell median dots** beneath each density (filled = PRE, open =
  POST) — one dot per replicate, i.e. the level the statistics actually use. The
  pooled-per-track KDE shows shape; the dots keep it honest (a few high-track
  cells can't masquerade as the distribution).
- The mobile/immobile **D threshold** is kept as a vertical guide, and the
  legend is a neutral key (line style + the dot meaning) parked in the empty
  upper-left so it never overlaps the curves.
- Many flat groups now small-multiple (one facet each) instead of overlaying
  into spaghetti; ≤3 flat groups overlay in one facet with per-cell dots.

## v2.42.0

### Compare page: many more statistical tests & options

The Statistics panel on the Compare tab gained a much richer, still-guided set of
options (all driven by the same config that's recorded in the CSV/PDF headers, so
results stay self-describing):

- **Alternative tests** — pick the non-parametric two-group test (Mann-Whitney,
  **Brunner-Munzel** for unequal spread, or a **permutation test** that assumes no
  distribution), and a proper 3+-group **post-hoc**: **Games-Howell** (unequal
  variance), **Dunn** (after Kruskal-Wallis), or **Tukey HSD**.
- **Dunnett's test** — set a **control group** and compare every group to it
  (many-to-one), with built-in family-wise control.
- **Equivalence testing (TOST)** — ask whether two groups are *practically the
  same* within a margin (in pooled-SD units), not just whether they differ.
- **Robust effect sizes** — every pairwise comparison now also reports **Cliff's
  delta** (with bootstrap CI) and **rank-biserial**, and each omnibus test reports
  **η² / ε²** — alongside the existing Cohen's d / Hedges' g.
- **More corrections** — **Šidák** and **Hochberg** added to None / Bonferroni /
  Holm / Benjamini-Hochberg.
- **Richer "Recommended for your data"** — new advice for tiny groups (→
  permutation), unbalanced designs (→ Welch + Games-Howell), many groups (→ Dunnett
  / strong correction), and a set control group (→ Dunnett). "Apply recommended
  settings" sets the new controls too.

Self-correcting post-hocs (Games-Howell / Tukey / Dunnett) are never
double-corrected — they're labelled as family-wise in the figure/CSV and excluded
from the across-metric family. Everything round-trips through the config validator,
so existing saved settings keep working. 9 new tests cover the additions.

## v2.41.5

### Updater: clearer message during the cross-platform publish window

The macOS and Windows builds finish (and upload their installers) a few minutes
apart, so for a short window after a release one platform sees "update available"
while *its* installer isn't on the release yet. Previously that showed the update
dialog with only **View release page** and no explanation. It now says plainly:
*"The installer for your platform is still being published … try again shortly
(Preferences → Updates → Check for updates now)."*

## v2.41.4

Patch release with no functional changes — a target to verify the Windows
in-app updater end-to-end (running v2.41.3 should detect v2.41.4, download it,
close + relaunch with no visible helper window).

## v2.41.3

### Updater: fix Windows install/relaunch (hidden helper + force-exit)

Two Windows-only bugs in the self-update helper:

- **A console window popped up** during the update. The helper was spawned with
  `DETACHED_PROCESS | CREATE_NO_WINDOW`, but Windows *ignores* `CREATE_NO_WINDOW`
  in that combination — so a visible terminal appeared. Now spawned with
  `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP` (no window; the child still
  outlives the app).
- **The update could stall forever** if the app didn't fully exit. On Windows,
  napari/Qt teardown can hold the GIL during close and there's no `SIGALRM`
  fallback, so the process occasionally hung — and the helper waited for that
  PID indefinitely, so the swap never happened. The helper now waits a bounded
  ~16 s and then **force-kills** the stuck process (`taskkill /F /T`) before
  swapping and relaunching. The macOS helper gained the same `kill -9` safety
  net (after ~15 s) for parity, and the post-swap copy retries longer to ride
  out the one-file bootloader briefly holding the `.exe` lock.

## v2.41.2

### Updater: ride out transient GitHub 504s

A freshly-published GitHub release asset can have its download edge return
**HTTP 504 (Gateway Time-out)** in bursts for a minute or two — which made the
in-app update fail ("Download failed after 3 attempts… 504") if you clicked
*Update now* in that window. The downloader is now much more patient:

- Retry budget for update downloads raised to **6 attempts** with backoff
  stretched out to ~30 s (so a ~1–2 min 504 burst is ridden out automatically).
- A second-by-second **"Server busy — retrying in Ns…"** status keeps the
  progress dialog alive (and cancellable) during backoff instead of looking
  frozen.
- New `tests/test_net_download.py` covers retry-through-504, exhaustion, and
  immediate-abort on a permanent 4xx.

(Bursty 504s are server-side and intermittent; the manual **View release page**
download remains the fallback if GitHub is having a bad moment.)

## v2.41.1

Patch release with no functional changes — used to verify the new one-click
in-app updater end-to-end (running v2.41.0 should detect v2.41.1, download it,
replace itself, and relaunch).

## v2.41.0

### One-click in-app updates

The packaged app can now update itself — no more downloading a fresh DMG/EXE from
the Releases page by hand. When a newer release is found, the header "Update
available" pill (and **File → Check for Updates…** / **Preferences → Updates**)
opens a dialog with the release notes and an **Update now** button:

- **Fully automatic** — FIREFLY downloads the new build (with a progress bar you
  can cancel), replaces itself in place, and relaunches. On macOS it clears the
  Gatekeeper quarantine flag on the new bundle (the app is unsigned), so no
  manual "right-click → Open" is needed after updating.
- **Frozen builds only** — running from source the feature is hidden (use
  `git pull`). If the install lives somewhere needing admin rights, the download
  is kept and revealed so you can finish by hand.
- **Preferences → Updates** — toggle the automatic startup check, see the current
  version, or check on demand. "Skip this version" suppresses the pill until a
  newer one ships.

Under the hood: a new `firefly/net_download.py` holds the shared, hardened
downloader (atomic write, resume, stall-watchdog, throttled progress, retry —
extracted so the CUDA installer and the updater share one implementation), and
`firefly/updater.py` handles release discovery + the per-OS swap-and-relaunch.
16 new pure-logic tests (`tests/test_updater.py`) cover version compare, asset
selection, release parsing, helper-script generation, and the frozen-only guards.

## v2.40.0

### Packaging & developer onboarding (Phase 3)

- **`pyproject.toml`** — FIREFLY is now a proper installable package
  (`pip install -e ".[dev]"`), with dependencies mirroring `requirements.txt` and
  a **single-sourced version** read statically from
  `firefly/sptpalm_analysis.__version__` (the same line CI stamps), so the package
  metadata and runtime version can't drift.
- **`DEVELOPER.md`** — onboarding for contributors: setup, the
  analysis/UI/worker architecture, the layering + threading invariants (why heavy
  compute must stay off the napari thread), how to add an analysis step + test,
  and the release process/conventions.
- `.gitignore`: ignore `*.egg-info/`.

No runtime/behaviour change.

## v2.39.0

### Reliability: clearer errors on bad input (Phase 2)

Malformed input now fails with a readable message instead of a cryptic traceback
deep inside trackpy/pandas (valid runs are unchanged):

- **`link_trajectories`** — raises a clear error if the localisations are missing
  a required column (x / y / frame), and drops negative-frame rows with a warning
  (mirroring the importer) instead of crashing trackpy's frame indexing.
- **`load_external_locs`** — raises a clear "no localisations found" error when a
  file parses but yields zero rows after column-mapping/filtering (wrong preset,
  empty file, or all-bad frames), rather than returning an empty frame that
  crashes a later stage.

(The codebase's many `except: pass` sites were reviewed and found to be mostly
*intentional* graceful degradation — e.g. per-track fit fallbacks that correctly
yield NaN — so they were deliberately left silent rather than made noisy.)

## v2.38.0

### Reliability: test coverage for previously-untested science (Phase 1)

Added known-truth regression tests for analysis paths that had no coverage
(no behaviour change — purely additive safety, run automatically in CI):

- **External-format import (`load_external_locs`)** — verifies column mapping,
  unit conversion (nm→px, µm→px), per-tool frame offsets, and TrackMate
  `TRACK_ID → particle` (with unlinked spots dropped) for the
  ThunderSTORM / Picasso / TrackMate / PALM-Tracer presets. This is the import
  path most prone to silent coordinate/frame corruption.
- **MSD fit (`compute_msd_and_fit`)** — serial vs multi-worker give identical
  D/α; short/sparse tracks return finite-or-NaN (never inf), no crash.
- **`make_figure` smoke test** — the single-run master figure renders headlessly
  across all panels and writes a PNG (regression guard for the 828-LOC figure).

(`fa_drift` RCC and `fa_twoway` ANOVA were found to be already well-covered, so
no redundant tests were added there.)

## v2.37.7

### Cluster tuning: keep the view, and never freeze at large eps

- **Re-tuning no longer resets the camera.** Changing eps / min-samples / point
  size / colour rebuilds the napari Points layer, which made napari auto-fit the
  view and throw away your zoom/pan. The overlay now saves and restores the
  camera across a re-render, so you stay where you were.
- **Large eps always produces a result.** The over-large-eps guard used to
  *refuse* (returning no clusters and leaving the overlay frozen, so tuning above
  a point appeared to do nothing). It now **sub-samples to a memory-safe size and
  still clusters**, so a large eps gives a real, changing result (with the
  sub-sample noted) instead of a frozen overlay.

Note: in **Motion** colour mode, raising eps mostly *merges* clusters, which
motion colouring doesn't show (points are coloured by motion class, not by
cluster) — switch "Colour by" to **ID** to see clusters merge/split as you tune,
or watch the "N clusters | M noise" readout.

## v2.37.6

### Cluster overlay: eps changes are now visible in Motion colour mode

In Motion colour mode the overlay coloured every point by its motion class —
including noise points — so re-tuning eps changed the cluster assignments but the
view looked identical (motion class is independent of clustering). Now **noise
points (not in any cluster) are greyed in Motion mode too**, while clustered
points keep their motion colour. So as you change eps, more/fewer points drop to
grey and the effect of tuning is clearly visible (and still matches the legend).

## v2.37.5

### Wider eps range for clustering (up to 2000 nm)

The Visualise tab's eps slider was capped at 500 nm, so spread-out data — where a
larger neighbourhood is appropriate — couldn't be tuned past it, and "Suggest
eps" just pinned to the ceiling. The eps range (both the Visualise slider and the
Analysis sidebar) now goes up to **2000 nm**, and "Suggest eps" clamps to the
slider's actual range rather than a hardcoded 500. The memory guard still safely
refuses an eps so large it would exhaust memory, so the wider range can't crash.
(Tip: use the arrow keys on the slider for fine 1 nm steps.)

## v2.37.4

### Cluster overlay defaults to Motion colours

Loading a cluster map now colours the overlay **by motion class by default**
(when the run has per-localisation motion data), so the dots match the sidebar
motion legend out of the box instead of the per-cluster "ID" rainbow. The
"Colour by" dropdown still lets you switch back to ID (one colour per cluster)
whenever you want to tell individual clusters apart.

## v2.37.3

### Fix: cluster re-tuning crashed the app on macOS (background thread)

The v2.36.0 "no-freeze" background-thread re-clustering caused napari/vispy to
crash hard on macOS (a native segfault — *"Python quit unexpectedly"* — preceded
by `vispy: Cannot set parent … in a different thread`). napari's GL rendering is
not safe to drive alongside a worker thread there.

Re-clustering and "Suggest eps" now run **synchronously on the main thread**. The
eps memory guard (v2.37.2) keeps the work bounded, so this is a brief
wait-cursor pause rather than a freeze — and there's no longer any background
thread to crash the renderer. (Trade-off accepted: a momentary pause on a heavy
re-cluster instead of a crash.)

## v2.37.2

### Fix: an over-large eps could crash clustering (incl. "Suggest eps")

Re-clustering with a very large eps made DBSCAN's neighbourhood graph blow up
(O(n²) memory) and could crash the app — which is what happened when "Suggest
eps" returned an outsized value and jumped the slider to its maximum.

- **`compute_clusters` now guards against it:** it cheaply estimates the average
  neighbourhood size first and, if an eps would be enormous, skips DBSCAN and
  returns "no clusters" with a clear message instead of exhausting memory. The
  Visualise tab keeps the previous overlay and shows *"eps too large — lower it"*.
- **"Suggest eps" is more robust:** the k-distance knee now clips outlier tails
  so a few far points can't skew it toward an absurd value.

## v2.37.1

### Fix: cluster overlay failed to render on napari 0.6+

napari 0.5/0.6 renamed the Points layer's `edge_color` to `border_color` (and
removed the old name in 0.6.x), so loading the Visualise tab's DBSCAN cluster
overlay errored with *"add_points() got an unexpected keyword argument
'edge_color'"*. The overlay now uses `border_color`, falling back to
`edge_color` on older napari — so it renders across napari versions.

## v2.37.0

### Clustering: honest subsample warning + radius of gyration (Phase 3)

- **The 250k subsample is no longer silent.** DBSCAN caps at 250,000
  localisations for speed; when it kicks in, it's now surfaced everywhere the
  cluster results appear — the run log, the Results panel ("DBSCAN clusters …
  (subsampled)", with a tooltip), the figure's Cluster Map caption
  ("sub-sampled to N"), and the Visualise status — so the counts/areas are never
  silently based on a subset.
- **Radius of gyration per cluster.** A new `rg_um` column (RMS distance of a
  cluster's localisations from its centroid) is added to the cluster stats CSV
  and the click-to-inspect panel. Unlike the convex-hull area it's always
  defined, even for small or collinear clusters — a robust size measure.

## v2.36.0

### Visualise tab — DBSCAN tuning no longer freezes; eps helper (Phase 2)

- **No-freeze re-clustering.** Dragging the eps / min-samples controls now runs
  DBSCAN on a **background thread**, so the window stays responsive instead of
  hanging on large localisation sets. Only the latest tune's result is applied
  (superseded runs are dropped), with a "clustering…" status while it works.
- **"Suggest eps" button.** Estimates a sensible eps from the **k-distance knee**
  (k = min-samples) — the standard DBSCAN heuristic — sets the slider to it, and
  re-clusters. (Also runs off the GUI thread.)

## v2.35.0

### Visualise tab — the DBSCAN live tuner is now usable (Phase 1)

The interactive cluster overlay gains the pieces it was missing:

- **Save your tuning.** A new **"Export tuned clusters…"** button writes the
  current live-tuned clustering to `*_cluster_labels_tuned.csv` /
  `*_cluster_stats_tuned.csv` next to the run (the originals are untouched) and
  copies the tuned **eps / min-samples into the Analysis sidebar**, so a re-run
  reproduces it.
- **Sliders match the run.** Loading a cluster map now sets the eps / min-samples
  sliders to the run's *actual* clustering parameters (from its `params.json`),
  instead of leaving them at a generic default — so a nudge refines the loaded
  result rather than jumping away from it.
- **No more silent "Motion" fallback.** When a run has no per-localisation motion
  data, the "Colour by: Motion" option is disabled (with a tooltip) instead of
  silently showing ID colours under a "Motion" label.
- **Adjustable point size.** A point-size control resizes the overlay markers
  (live, no re-render) and is remembered across sessions.

## v2.34.0

### Circular statistics: full customization + in-app explanation (Compare tab)

The circular (turning-angle) statistics were richly computed but had no UI
customization and almost no in-app explanation. The Compare wizard now has a
**"Circular statistics" card** that brings them to parity with the scalar stats:

- **Customization.** A master toggle to include/exclude the circular CSV + PDF
  outputs (separate from the figure panels), plus individual toggles for which
  between-group tests are reported (concentration κ, resultant length R̄,
  Watson-Williams mean direction μ, circular-linear correlation). All persist in
  settings.
- **They obey your stats settings.** The per-replicate circular comparison tests
  now reuse the chosen **α + multiple-comparison correction** (and parametric
  strategy), so the circular results agree with the scalar ones — the tests CSV
  gains a `p_corrected` column and the stars honour your α.
- **Explanation.** Glossary entries (Rayleigh, V-test, Watson-Williams, κ, R̄,
  directional persistence, circular-linear correlation, …), an in-UI
  **sign-convention note** (previously only in the PDF footer), and the live
  test-plan now lists the circular tests that will run.

Pooled-angle tests stay omitted by design (pseudoreplication); the per-file
single-run circular report is unchanged.

## v2.33.0

### Distribution KDEs taper fully (even wide classes like Directed)

The v2.32.0 fix padded the KDE grid by a fixed fraction of the data range, which
still left a wide class (e.g. Directed, which spans decades of D) cut off before
its right tail reached zero. Each class's filled KDE is now evaluated on a grid
that extends a few of *its own* bandwidths past *its own* data on both sides, so
every curve — narrow or wide — tapers smoothly to ~0 regardless of how spread out
it is. Affects the single-run E (log₁₀ D), G (α) and N (MSS) panels.

## v2.32.0

### Distribution panels: KDE curves taper instead of cutting off

The filled-KDE distribution panels (single-run E log₁₀ D, G α, N MSS slope) drew
each class's curve on an x-grid clamped to exactly that data's min/max. Because a
KDE still has non-zero density at the edge data point, the curve was drawn with a
vertical "cut" at the lowest/highest value (most visible on the Immobile class,
which sits at the left edge). The KDE grid is now padded slightly on both ends so
every class tapers smoothly to ~0 at its tails. Data and counts are unchanged.

## v2.31.0

### Diffusion Coefficient panel (E): show the immobile peak + fix clipped curves

Two fixes to the single-run "Diffusion Coefficient Distribution" panel after the
move to filled KDEs:

- **The below-resolution (immobile) population is visible again.** Tracks with D
  below the resolution floor are a true delta — they're now drawn as a hatched
  bar at the floor (labelled with their %), on the same count axis as the curves,
  so the immobile peak shows up instead of only a faint dotted line + caption.
- **Curves are no longer clipped.** The y-axis now adds headroom above the
  tallest KDE peak (and the floor bar), so a sharp peak like Confined is fully
  visible instead of being cut off at the top, and the legend clears the curves.

## v2.30.0

### Single-run Motion Classification: vertical bars

Panel F (Motion Classification) is now a **simple vertical bar chart** instead of
a horizontal 100% stacked bar: one bar per class, the class names
(Immobile / Confined / Brownian / Directed) on the x-axis, and a fixed **0–100%**
y-axis ("% of classified tracks") so each proportion is read on an absolute
scale. Each bar keeps its class colour, with the % printed above it. Same data
and percentages — just an easier-to-read layout.

## v2.29.0

### SuperPlot polish + direct line-end labels (Phase 4)

The final pass of the figure-modernization work, both on the comparison figure.

- **Lighter SuperPlot bars.** In the comparison bar charts (AUC, Mobile/Immobile
  ratio, α₂, VACF persistence, …) the bar was visually competing with the data.
  The bar face is now a very pale, low-opacity wash while the **edge keeps the
  saturated group colour** and the **mean ± SEM error bars stay solid** — so the
  per-replicate dots (the actual unit of replication) read as the data, not the
  bar. The dots, error bars and all statistics are unchanged.
- **Direct line-end labels.** On the MSD overlay and the turning-angle
  distribution, each curve is now labelled in its own colour at its right-hand
  end, so you follow a line straight to its name instead of hunting the legend
  (r-graph-gallery practice). When two line-ends are too close to label without
  overlapping, it automatically falls back to the shared legend.

Presentation only — no change to any value or statistic.

## v2.28.0

### Cleaner distributions: filled KDEs + ridgeline (Phase 3)

Overlaid semi-transparent histograms turn to mud once several classes/groups
stack up. Replaced them with smooth **filled KDE** curves (Wilke /
r-graph-gallery practice) — same data, far easier to read.

- **Single-run figure — panels E (log₁₀ D), G (α), N (MSS slope).** Each motion
  class is now a translucent filled KDE with a thin saturated outline instead of
  overlaid histogram bars. The curves are **count-scaled** so the y-axis stays a
  count and every reference line/annotation is unchanged. Sparse or
  zero-variance classes fall back to a histogram automatically.
  - Panel E keeps the immobile floor honest: the KDE traces only the *resolved*
    (mobile) tracks, and the immobile fraction stays shown as the dotted floor
    line + "X%" label (it is not smeared into a fake bump).
- **Comparison figure — LogD distribution.** Up to 3 groups: overlaid filled
  KDEs. **4+ groups: a ridgeline plot** — one filled KDE per group, stacked with
  a small offset and labelled directly in the group's colour, so a many-group
  comparison stays legible instead of becoming spaghetti. The mobile-D threshold
  line and the sub-resolution clip are preserved.

Presentation only — the underlying values, classifications and statistics are
unchanged.

## v2.27.0

### Single-run figure: pie chart → 100% stacked bar (Phase 2)

The single-run master figure's **panel F (Motion Classification)** was a pie
chart. Pies are hard to read — the eye can't compare wedge angles accurately, and
small slices vanish. It is now a **100% stacked bar**, the modern replacement
(r-graph-gallery / Wilke): shares are read off a common baseline.

- Mirrors the comparison figure's Motion-Class Fractions panel — same data
  (the named Immobile / Confined / Brownian / Directed classes, renormalised to
  100%), same theme-aware, colour-blind-safe colours, on-segment **%** labels
  (shown when a class is ≥ 6% wide), a compact 2-column class legend tucked above
  the bar, and a 0–100% axis. Same percentages as the old pie.
- **Per-segment label contrast.** The on-bar **%** text now picks black or white
  per segment by WCAG contrast, so labels stay legible on every class colour and
  every theme (including the Publication colour-blind palette). This logic was
  duplicated inline in the comparison figure; it is now a single shared
  `label_text_color()` helper used by both figures.

Purely a presentation change — the classification and the numbers are unchanged.

## v2.26.0

### Figure & results readability fixes (lab feedback)

A round of fixes responding to feedback from the lab:

- **Fixed the VACF directional-persistence bar.** This metric (the lag-1 velocity
  autocorrelation) is the one comparison metric that can legitimately be
  *negative* — anti-persistent / caged motion, or the localisation-noise
  anti-correlation between consecutive steps. The bar renderer assumed every
  metric was ≥ 0 and pinned the y-axis floor to 0, which clipped the downward bar
  below the baseline and made it look "impossibly low / broken." The bar chart is
  now sign-aware: negative bars draw correctly from a **0 reference line**, with
  the y-axis spanning the real data range. The numbers were always correct — only
  the drawing was wrong.
- **Trajectory panels can now hide the cell image.** New Preferences → figure
  setting **"Show cell image behind trajectories"** (on by default). Turn it off
  to plot the tracks on a plain background — panels *B (Trajectories)* and
  *C (Trajectories by D value)* show only the trajectories, no faint
  max-projection behind them.
- **Removed the AMOLED figure theme.** The figure-theme menus (single-sample and
  comparison) now offer Dark / Light / Publication — dark is enough for figures.
  The AMOLED *app* theme is unchanged.
- **More readable section titles** on the post-run results readout (full-contrast,
  slightly larger headers instead of the faint muted labels).

## v2.25.0

### Cleaner, more modern figures (global style pass)

A consistent restyle of every panel in both the single-run figure and the
comparison figure, following modern data-viz practice (r-graph-gallery / Wilke):

- **Removed the top and right spines** from every data panel (the classic
  "chart junk") and thinned the remaining axes — panels now read as open L-shaped
  axes instead of heavy boxes. Trajectory / image / polar panels keep their frame.
- **Lighter, subtler gridlines** (thin dashed, low alpha) everywhere.
- **The Publication theme now uses a clean sans-serif font** (the journal
  standard) instead of serif. The screen themes are unchanged — Dark/AMOLED keep
  their monospace look, Light stays sans-serif.

Same data and numbers — purely a styling pass (new shared `style_axes` helper).
This is the foundation; later releases improve specific charts (the pie, the
distribution plots, and the SuperPlots).

## v2.24.0

### Presets show when you've changed something

The preset selector at the top of the Analysis sidebar gains a **"• modified"**
pill: once you apply a preset and then tweak any parameter, the pill appears so
it's obvious your current settings no longer match the saved preset (revert the
change and it disappears; save a new preset to keep them). Picking "— Current
settings —" or a fresh preset clears it. No change to how presets save/load.

## v2.23.0

### Sidebar header + title polish

- **Open sections read as "active."** An expanded section now shows a continuous
  **accent left bar** down its header and content; collapsed sections have a muted
  bar. Cleaner padding/rhythm and lighter disclosure chevrons (▾ / ▸).
- **Stronger sidebar title.** The per-tab title ("Analysis Parameters",
  "Comparison", …) is larger and bolder with a subtle divider beneath it, so it
  reads as a proper header rather than an afterthought.

Purely visual; theme-aware across Dark / AMOLED / Light.

## v2.22.0

### Live state chips on the Analysis sections

Section headers now carry a small status chip so the active config reads at a
glance — even when the section is collapsed:

- **Detection** → `auto` / `manual` (minmass mode)
- **ROI** → `none` or the active mode (e.g. `auto threshold`)
- **Diffusion** → `D-filter on` (only when the optional D-range filter is enabled)
- **Drift correction** → `RCC on` (only when drift correction is enabled)

The chips update live from the controls and reflect a preset / restored settings
automatically. (`_CollapsibleSection` gained a reusable `set_badge()`.)

## v2.21.0

### A calmer, searchable Analysis sidebar

The Analysis parameter sidebar had ~50 controls across ~10 sections, all expanded
— a wall to scroll through. It now opens scannable.

- **Advanced sections collapse by default.** Imaging / Preprocessing / Detection /
  Linking / Diffusion stay open; the rarely-touched ROI, Drift, Clustering and
  Performance sections start collapsed (click to expand).
- **"Search parameters…" box.** A filter at the top of the sidebar shows just the
  controls whose name matches what you type (e.g. "minmass", "search", "roi") and
  expands their sections; clearing it restores the default view. No setting is
  changed — it only shows/hides rows.

## v2.20.4

### Results readout: typography + spacing polish

- **Cleaner header.** The run title and the "Analysis successful" pill now share
  one header row (title left, pill right) instead of the pill sitting orphaned in
  the middle; the title is a touch larger and bolder.
- **More breathing room.** Wider row spacing and clearer separation above each
  section header, so the readout feels less cramped.
- **Tighter value column.** Verbose labels were shortened ("Mobile fraction",
  "Stuck tracks") so the values sit closer to their labels rather than far to the
  right; the dropped qualifiers moved into hover tooltips. Values are slightly
  larger for a clearer label→value hierarchy.

## v2.20.3

- **Compare sidebar can no longer be dragged sideways (for real this time).**
  Hiding the horizontal scrollbar wasn't enough — a trackpad swipe still scrolled
  the hidden overflow because the group cards were a touch wider than the panel.
  The Compare sidebar now uses a vertical-only scroll area that clamps its content
  to the panel width, and the group-card buttons use shorter labels (+ Add /
  Remove / Clear) so nothing needs to overflow. The content now compresses to fit
  instead of scrolling.

## v2.20.2

### Three UI fixes

- **Viewer tracks no longer dim when only one motion class is shown.** The
  per-class napari Tracks layers were created with `opaque` blending, whose depth
  test darkened thin antialiased lines on the black canvas until a second layer
  was toggled on. They now use napari's default `additive` blending, so a single
  visible class renders at full brightness.
- **Compare sidebar no longer drags sideways.** The group cards' folder list
  could grow a horizontal scrollbar (making the card draggable left/right) when a
  folder basename was long. It now suppresses the horizontal scrollbar and elides
  long names with "…" (the full path is still in the row's tooltip).
- **Re-process help text tightened.** The info banner is shorter and clearer (the
  output-folder detail moved to its tooltip), with a bit more breathing room above
  the preview viewer.

## v2.20.1

### Modernised the post-run results readout

- **Motion classes as a stacked proportion bar.** The four classes
  (Immobile / Confined / Brownian / Directed) are now a single coloured bar
  sized by fraction — hover a segment for its count + % — above a tidy swatch
  legend, instead of four flat coloured rows.
- **Sectioned stats.** The readout is grouped under quiet uppercase headers with
  hairline dividers (Diffusion & dynamics · Motion classes · Clustering &
  acquisition · Quality control), so it scans cleanly rather than as one long
  list. Same numbers, clearer structure.

## v2.20.0

### Guidance polish across Import, Visualise and Re-process

The same modern affordances now extend to the rest of the app.

- **Import** — a live **"Ready to analyse"** pill in the tab header turns green
  once you've picked an input (a file in single mode, a folder in batch), so
  it's obvious when you're good to go.
- **Visualise** — a **motion-class colour legend** in the sidebar (a swatch per
  Immobile / Confined / Brownian / Directed / Unknown in the active palette) so
  you can read the track colours without opening the napari layer list;
  hover-definitions on Min length / eps / min samples; and a **"No clusters
  found"** warning banner nudging you to widen eps or lower min-samples when a
  clustering run comes back empty.
- **Re-process** — a **status pill** by the source picker ("No run selected" →
  "Run loaded", or "Not a FIREFLY run" on a bad pick), and the workflow help text
  is now an info banner so the steps stand out.

All reuse the existing banner/badge/chip widgets; no analysis or settings change.
(The Windows-only GPU-status banner is deferred until it can be verified on
Windows.)

## v2.19.0

### A live pipeline map in the Analysis cockpit

The Analysis tab now shows a compact **stage map** —
Preprocess → Detect → Link → Drift → Diffuse → Classify — so you can see at a
glance where a run is. Completed stages turn green, the running stage glows in
the accent colour, and pending stages stay muted; hover any stage for a one-line
description.

- Native QPainter widget (crisp at any size, theme-aware), driven by the
  analysis worker's progress messages. Because the worker's progress percentages
  aren't strictly monotonic across stages, the map only ever advances (it never
  flickers backward).
- Resets at the start of each run (and each file in a batch) and turns fully
  green on completion. Purely additive and guarded — Compare runs (which have no
  per-stage pipeline) leave it untouched, and nothing in the run path changes.

## v2.18.0

### Post-run results: QC flags become severity banners + a readiness badge

The Results panel that appears after an analysis now reads like the Compare
wizard's guidance.

- **QC flags are severity banners**, not plain text. Each warning/info flag
  renders as a coloured callout (red/amber for warnings, blue for info) with a
  short bold lead (Low link ratio / High density / Short tracks / Stuck tracks /
  Track gaps / Drift corrected) above the full message — which still carries the
  concrete remedy from the analysis.
- **A run-readiness pill** sits under the headline: green **"Analysis successful"**
  when there are no warnings, red **"Completed with warnings"** when there are.
- No change to what's computed — purely how the existing QC results are
  presented. The stats grid and saved-file list are unchanged; the panel's API
  is untouched, so every caller keeps working.

## v2.17.1

- **Definitions now appear on hover of the label text — no ⓘ icon.** The little
  info icons are gone; instead, hovering a parameter's name shows its
  plain-English definition (a 'help' cursor hints there's an explanation). Applies
  everywhere `_label_with_info` is used (Analysis sidebar + the Compare wizard).
- **"Max lag time" now explains what a lag time is** — the gap between two
  positions on a track that you compare; the MSD curve averages squared
  displacement over all pairs at each lag.

## v2.17.0

### Plain-English ⓘ explanations for every Analysis parameter

The Analysis sidebar is powerful but jargon-heavy. Every technical parameter now
carries a small **ⓘ icon** whose tooltip gives a one-sentence, plain-English
definition — the same affordance the Compare tab's statistics wizard uses.

- **~30 parameters explained**: diameter, minmass (+ sensitivity, false-track
  rate), background method/radius, search range, memory, min/max track length,
  max lag time, n-fit lags, the α (anomalous-exponent) thresholds, mobile-D
  threshold, JDD components, filter-by-D, ROI mode/auto-method/projection/
  threshold/background-σ, RCC drift + segment size, DBSCAN eps + min-samples,
  detection backend, chunk size, pixel size, frame interval, channel.
- Backed by a new `ANALYSIS_GLOSSARY` that the shared `glossary_def()` lookup
  resolves alongside the statistics glossary, so the same ⓘ widget works
  everywhere. The detailed hover tooltips are unchanged — the ⓘ is additive.

No behaviour or settings change; the full analysis suite stays green.

## v2.16.1

- **Decision diagram: arrow no longer touches the result box.** The final arrow
  into the chosen-test box ended exactly on its left border; it now stops a
  clear gap short so the arrowhead doesn't clip into the box.

## v2.16.0

### A modern, guided "Analysis Configuration" wizard (Compare tab)

The Compare tab's centre panel was rebuilt to feel like a real stats package —
clearer guidance, on-theme visuals, and a correctness fix to the replicate count.

- **Replicate-count fix (important).** The recommendation counted *cards per
  label* instead of *folders*, so a group with 4 folders showed as "1 replicate"
  and wrongly warned "comparisons can't be interpreted". The backend treats each
  analysis-output folder as one replicate (one scalar row per folder), so the
  panel now counts folders. Your "Ready to run" status and recommendation now
  reflect the real n.
- **Guidance banners + run-readiness badge.** The recommendation is now shown as
  severity **alert banners** (red ⚠ / amber ⚠ / green ✓ with a coloured bar),
  and a **status pill** in the header reads "Ready to run" / "Need ≥3 replicates"
  / "Add 2 groups" at a glance.
- **Native, crisp decision diagram.** The matplotlib raster diagram (with its
  off-theme green result box) is replaced by a vector **QPainter** widget:
  retina-sharp at any size, fully on the accent palette, naming the chosen test,
  and hover-explained per node.
- **Richer design summary.** The plain "DMSO (1) · Ciprofol (1)" text is now
  **colour chips** matching each group's sidebar colour with a replicate count,
  plus a one-line explanation of *why* the design is paired vs unpaired (e.g.
  identical time points ⇒ compared as independent groups).
- **Polish.** Numbered **step badges** on each section; a tighter options form
  (controls no longer stretch edge-to-edge); inline glossary **definitions**
  (term — one sentence) instead of hover-only icons; a condensed intro; and the
  sidebar group cards now show readable folder **basenames** (full path on hover)
  while every consumer still gets the absolute path.

Covered by offscreen UI checks (banner severities, badge states, chips, native
diagram paint + flow, folder full-path round-trip, settings round-trip) and the
full analysis suite stays green.

## v2.15.1

- **"Generate comparison" is now the blue primary button**, matching the "Start"
  button on the Import/Analysis page (accent fill, bold). Previously it was a
  plain button, so the Compare tab's main action didn't read as the primary
  call-to-action.

## v2.15.0

### One Compare tab with a wizard-driven "Analysis Configuration" centre

The separate **Compare** and **Statistics** tabs are merged into a single
**Compare** tab that works like a modern stats package (JASP / GraphPad Prism /
jamovi): you set up the experiment on the left and read/choose the statistics in
the centre.

- **Left sidebar = the whole experiment setup.** The group/file/colour cards
  (drag-and-drop folders, per-group colour, time point) now live in the sidebar
  alongside the output folder/name — everything that *defines* the comparison in
  one place. (Figure theme, panels and the PDF toggle stay in Preferences.)
- **Centre = "Analysis Configuration", a live wizard.** A single, always-visible
  panel that updates as you edit groups: (1) your detected experimental design
  (paired/unpaired, group count, replicates), (2) a data-aware recommendation
  with one-click apply, (3) the test-choosing options, (4) the plain-English plan
  of exactly which test each metric gets, and (5) a decision diagram of how the
  test is chosen. Editing a group on the left refreshes all of this instantly.
- **Inline plain-English explanations.** Every technical term carries a small
  **ⓘ** info icon whose tooltip defines it in one sentence — Sphericity,
  Interaction effect, Welch's t-test, Holm, Benjamini–Hochberg FDR,
  Greenhouse–Geisser, Hedges' g, Family-wise, Parametric, Mann–Whitney,
  Two-way mixed ANOVA, and more — backed by a reusable `STATS_GLOSSARY`. A
  collapsible "What these terms mean" panel lists them all.
- **No behaviour change to the statistics themselves.** Same tests, same config
  keys, same saved settings; the controls just moved from a second tab into the
  Compare centre. The old "⚙ Configure statistics…" button (which jumped between
  tabs) is gone — it's all one tab now.

Covered by an offscreen UI smoke test (5-tab layout, sidebar/title alignment for
every tab, stats widgets + config, live test-plan refresh, group cards hosted in
the sidebar, info-icon tooltips, and a settings round-trip). Full analysis suite
green.

## v2.14.2

### Theme-aware motion colours in the 3-D Visualise viewer

- **"Motion colours" selector in the Visualise sidebar.** The napari viewer
  coloured each motion class (Immobile / Confined / Brownian / Directed /
  Unknown) — both the per-class Tracks layers and the motion-coloured DBSCAN
  cluster overlay — with a single fixed dark-mode palette. A new sidebar
  selector now offers **Default** (the original bright dark-mode palette,
  unchanged) and **Colour-blind safe** (the Okabe-Ito palette, the same one the
  Publication figure theme uses), so the 3-D view can be made colour-blind
  accessible and matched to an exported figure. Both options are chosen to read
  well on the viewer's dark canvas (the light figure palette is intentionally
  not offered here — its deep hues are near-invisible on dark). The choice
  recolours the loaded layers **live** (in place, no rebuild/flicker) and is
  remembered between sessions. "Default" is pixel-identical to the previous
  fixed colours, so existing views are unchanged.

## v2.14.1

### Comparison bars now respect the figure theme (no more dark bars on white)

- **Per-group "tint + outline" bars.** The comparison bar panels (AUC,
  Mobile/Immobile ratio, Tracks detected, α₂, VACF persistence) drew every bar
  with a single fixed fill from the theme palette — which on the **Publication**
  and **Light** themes was a dark grey, so the bars came out near-black on a
  white background. Each bar is now filled with a pale wash of its **own group
  colour** blended toward the figure background, with the saturated group colour
  kept as the edge. The result is theme-adaptive: a clean pastel-with-outline on
  white themes and a subtle dark tint on Dark/AMOLED, and every bar now reads as
  its own condition instead of a uniform block. (The Motion-Class Fractions panel
  already used the per-class motion colours and is unchanged.)

## v2.14.0

### Theme-aware figure colours, Motion-Class fix, and a graceful "no folders" popup

- **Per-theme, colour-blind-safe motion-class colours.** The Immobile/Confined/
  Brownian/Directed colours were a single fixed dark-mode palette used in every
  theme — so a Publication figure (white background, serif) showed dark-mode bar
  colours. Each theme now has an appropriate palette: Dark/AMOLED unchanged
  (pixel-stable), Light uses deeper hues for white-background contrast, and
  **Publication uses the Okabe-Ito colour-blind-safe palette** (distinguishable
  under deuteranopia/protanopia/tritanopia and in grayscale print). Applies to
  every motion-coloured figure panel (trajectories, log10(D), α, MSS, the motion
  pie, and the comparison Motion-Class bars). The on-bar % label colour now uses
  a proper WCAG contrast pick so labels stay legible on every palette.
- **Motion-Class Fractions graph fixed for palmTRACER data.** On dense data a
  large share of tracks are too short to fit a D/α ("Unknown"), and the bars were
  divided by *all* tracks — so the four classes summed to ~0.2 and the stacked
  bars never reached the top. The bars now renormalise over the classifiable
  tracks (matching the single-run pie, which already does), reaching 1.0, and the
  x-axis labels honestly report the unclassified % per group.
- **Inaccessible folders no longer crash — they pop up a clear message.** A
  comparison whose group folders can't be loaded (e.g. the external drive wasn't
  mounted) raised a generic error that surfaced as a *crash report*. It now raises
  a dedicated CompareInputError that the worker turns into a friendly popup naming
  each empty group, *why* each folder failed ("folder not found — is the drive
  connected?" vs "not an analysis output folder"), and how to fix it.

Covered by new headless tests (theme palettes, renormalised bars reaching 1.0
with unclassified labels, and the friendly CompareInputError → compare_error path
instead of a crash). 98 tests pass.

## v2.13.0

### Statistics transparency + a dedicated Statistics tab

A full review of the Compare-tab statistics, plus user-facing control and
transparency over every test.

- **New Statistics tab** — global, defensible controls (no per-metric test
  shopping): significance α, multiple-comparison correction
  (None / Bonferroni / Holm / Benjamini–Hochberg FDR), an optional
  family-wise correction *across* the scalar metrics, parametric strategy
  (auto-by-normality / force parametric / force non-parametric), the 3+-group
  test, effect-size CI level, and whether on-figure stars use corrected p. A
  live **"test plan" preview** shows exactly which test each metric will get for
  the current groups before anything runs.
- **Self-describing output** — figure panels now name the test *and* the
  correction (including 2-group panels); the stats CSV gains a configuration
  header block and accurate, renamed columns (`p_value_corrected` +
  `correction_method`, plus an across-metric column), replacing the previously
  hard-coded "Bonferroni" label that didn't always match what was applied.
- **Correctness fixes (from the audit):** on-figure significance stars now use
  the chosen correction (so the figure can't say "*" while the CSV says "ns");
  an across-metric family-wise correction is available; and 3+ groups use
  **Welch's ANOVA** (unequal-variance robust, consistent with the Welch's t
  already used for 2 groups) instead of equal-variance one-way ANOVA. The
  two-way mixed-ANOVA post-hoc correction now follows the global choice.
- **Confirmed sound (no change needed):** the unit of analysis is per
  cell/replicate (not per track — no pseudoreplication); the two-way mixed
  ANOVA (between=group, within=time, subject=cell, Greenhouse–Geisser) and its
  per-cell curve drill-down were already correct.

All settings are config-driven through a single `fa_stats_config` module with a
backward-compatible normaliser, so old saved settings and callers keep working.
Covered by new headless tests (config normalisation, correction math incl. Holm
/ BH / NaN / single-comparison edge cases, forced strategies, Welch's ANOVA +
fallback, and an end-to-end Compare run asserting the figure and CSV agree).

## v2.11.0

### Bulletproof auto-threshold — linkability-optimised detection
The auto detection threshold (`minmass`) no longer relies only on single-frame
spot brightness — the same information a human eyeballs. The new **primary
engine measures temporal linkability**: real emitters persist and link into
coherent ≥L-frame trajectories, whereas noise makes 1-frame blips and 2–3-frame
fragments the linker cannot stitch (Jaqaman 2008). A criterion no single-frame
inspection can reproduce.

- **Per-file threshold sweep.** Harvest every candidate at `minmass=0` over a
  few *contiguous* frame windows (with trackpy PSF features), apply a PSF
  quality pre-gate (size / eccentricity / localisation error), then sweep the
  mass threshold and re-link at each step. The operating point maximises an
  **F1 balance of track purity vs real-detection recall** — immune to the
  track-fragmentation that gamed a raw track-count objective — floored at the
  count-knee noise level.
- **Strict / Balanced / Lenient** shift the cut ±1 grid step; an optional,
  advanced **“Max false-track rate (%)”** field directly caps the *measured*
  spurious-fragment rate, overriding the selector.
- **Graceful, flagged fallback.** When real spots don’t link or there’s no
  suppressible spurious population (immobile-dominated / sparse data), the
  estimator defers to the previous static GMM-valley / mass-quantile / knee
  method and records `static_fallback:<reason>` so the audit shows why.
- **Richer audit.** `{stem}_minmass_hist.png` gains a second panel: real-track
  yield, spurious-fragment rate and good-fraction vs threshold, with the chosen
  operating point and knee marked. `{stem}_params.json` records
  `minmass_n_good`, `minmass_spurious_rate`, `minmass_score`,
  `minmass_noise_floor`.

Covered by new headless tests (linkability path selected, beats the quantile on
overlapping populations, guard fallbacks, F1 operating-point). Full suite green.

## v2.9.0

A self-contained round of analysis, performance and robustness improvements
made on a dedicated branch (main untouched). Every change is covered by the
headless test suite (`pytest -q`). Grouped by theme below.

### New analysis metrics
- **Per-track localisation precision (`loc_sigma_nm`)** — derived from the
  static offset of the MSD fit (`MSD(t) = 4·D·t^α + 4σ²`, so `σ = √(MSD₀/4)`).
  Reported per track in `diffusion_summary.csv`; no separate bead calibration
  needed.
- **van Hove displacement distribution + non-Gaussian parameter α₂** — pooled
  single-frame displacements with `α₂ = ⟨r⁴⟩/(2⟨r²⟩²) − 1` (≈0 Brownian, >0 for
  heterogeneous/mixed populations). Saved as `<stem>_van_hove.json`.
- **Velocity autocorrelation (VACF) + persistence index** — normalised ensemble
  VACF; lag-1 value summarises directionality (≈0 Brownian, >0 directed, <0
  caged). Saved as `<stem>_vacf.json`.
- **JDD goodness-of-fit (R², RMSE, AIC, BIC)** — lets you objectively justify a
  1- vs 2- vs 3-population jump-distance model. Included in `<stem>_jdd.json`
  and shown (R²) on the figure's JDD panel.

### Reporting & batch workflow
- **Figure** — two new panels: **P** (van Hove, log-scale with Gaussian
  reference + α₂) and **Q** (VACF with persistence). Now 17 panels (A–Q).
- **`<stem>_summary_metrics.json`** — one machine-readable file per run with the
  headline numbers (counts, median D/α, loc precision, α₂, persistence, mobile
  fraction) and QC flags.
- **`aggregate_run_summaries()` + CLI `--aggregate`** — fold a whole tree of run
  summaries into one `run_summaries.csv` (one row per run, condition inferred
  from the parent folder).
- **Compare** — reports per-replicate α₂ and VACF persistence, adds two new
  comparison panels (population heterogeneity α₂; directional persistence) with
  SuperPlot dots + omnibus/pairwise stats, and includes them in the summary
  CSV, stats CSV and PDF report.
- **Results panel** — the post-run stats panel now shows localisation
  precision, non-Gaussian α₂ and VACF persistence alongside median D/α.

### Performance (results numerically identical)
- **MSS ~1.8×** — displacement computed once per lag instead of once per moment
  order; direct OLS slope instead of `np.polyfit`.
- **MSD** — gapless fast path (skips per-lag frame-difference masking for tracks
  with consecutive frames) and a single-pass groupby for per-track input
  extraction.

### Robustness & tests
- `compute_msd_and_fit` validates calibration (rejects non-positive/NaN pixel
  size or frame interval, `max_lagtime < 1`) instead of silently producing
  garbage.
- New regression tests: CZI metadata parsing (Element/bytes input, FrameTime,
  ms TimeSpan); JDD/MSS/turning-angle ground truth; van Hove/VACF; aggregator;
  and the first end-to-end `compare_groups` integration test.

### Docs
- README documents the new metrics, the α identifiability behaviour (immobile
  tracks report `α = NaN` rather than pinning at the fit bound), the new figure
  panels and the `summary_metrics.json` / aggregation workflow.
