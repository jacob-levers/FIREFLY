# Drift correction

Open **Analysis parameters → Drift correction**. RCC is the default. Enable
**Adapt windows to signal** for count-based window sizes and inspect the saved
diagnostic plot after processing. Reload the Drosophila preset to use its new
explicit settings; existing custom presets keep their saved choices.

## Reference selection

- **Analysis region:** estimate translation using detections retained by the
  tracking ROI. Appropriate only when their spatial distribution is sufficiently
  stable. Common biological motion can be mistaken for stage drift.
- **Separate rectangle:** enter left, top, width and height in raw camera pixels.
  These reference detections are selected independently of the tracking ROI,
  before drift correction. Choose stationary structure and leave enough margin
  for expected drift; do not let it cross the reference boundary. This uses the
  same detector/contrast settings as the analysis. It does not expand the
  analysis ROI or add reference particles to the tracked sample.
- **Reference CSV:** use localisations from the same acquisition and frame clock,
  including a separately processed reference channel if available. Required
  columns are `frame,x,y`; frames are zero-based integers and coordinates are
  camera pixels in the same scale/orientation as the sample. No resampling,
  channel registration, unit conversion or timestamp matching is inferred.
  A CSV path can contain `{stem}` for per-movie references in batch processing;
  relative paths resolve beside the input movie. The file path and SHA-256 are
  saved, and changing the reference contents invalidates imported-table caches.

The rectangular reference is configured numerically in the sidebar. It does not
reuse or overwrite the polygon drawn for the analysis ROI.

## Stationary fiducials

Choose **Stationary fiducials** and a separate rectangle or CSV. Select known
stationary beads/markers, not the moving sample. A CSV's `particle` column can
supply bead identities. Without identities, FIREFLY links reference detections
with the configured fiducial search range and memory of two frames.

Each marker must bracket a segment's center with enough observations and no
long internal gap. Its position is interpolated at that common time. Pairwise
shifts are medians across shared markers, rejecting markers that disagree.
The default requires at least three agreeing markers per accepted pair.
Lowering this to one removes the independent agreement check. Detections cannot
establish their own biological immobility, even when markers move together.

## Estimation and support

RCC forms high-pass localization-density maps, correlates all pairs, refines
peaks below the rendering-grid spacing, and solves the quality-weighted
translation constraints. Weak, ambiguous and search-boundary peaks are rejected.
Inconsistent shifts are iteratively rejected and the graph is re-solved.
Every segment must remain connected without relying on a single bridge pair.

The method builds on [Wang et al., Optics Express (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4162368/).
The adaptive windows, quality rules and conservative skip policy are FIREFLY
implementation choices; these thresholds are not calibrated probabilities.

With adaptation enabled, windows target **Min reference spots / window**, between
approximately one quarter and four times **Segment (frames)**. There are at most
64 windows, so long acquisitions have a coarser minimum time resolution. Sparse
terminal windows may merge with their neighbour. Density maps are capped at
512 pixels per axis; effective rendering scale is recorded and is not a claim
of localization accuracy. Rendering scale, event counts and motion within a
window limit the estimate's accuracy.

Smoothing runs on a uniform time grid, preserving a linear trend. Endpoint
extrapolation is marked in the per-frame CSV. This is 2D translation correction;
it does not correct rotation, deformation, axial drift, or unresolved motion
within a segment.

If support is insufficient, **Skip correction** applies zero drift, retains the
original coordinates, and records a warning in the run log and QC output.
**Stop run** writes the drift diagnostics and then stops processing. Invalid
reference configuration/file errors stop processing rather than silently
substituting another reference. A disconnected estimate is never filled in and
presented as measured drift.

## Saved diagnostics

In `firefly_extras/`:

- `<stem>_drift.csv`: applied x/y translation in camera pixels, supporting pair
  count, applied/skipped status, and endpoint-extrapolation flag.
- `<stem>_drift_segments.csv`: window bounds, counts, support and residuals.
- `<stem>_drift_pairs.csv`: candidate shifts, quality, rejection reasons and fit
  residuals. Fiducial results include agreeing-marker counts.
- `<stem>_drift_diagnostics.json`: full configuration summary and reference provenance.

In `figures/`, `<stem>_drift_diagnostic.png` plots the applied translation and
segment support. Status/reason are also stored in run parameters and QC. These
scores diagnose consistency; they are not confidence intervals or proof that
an accepted translation was stage motion. Independent stationary controls remain
the best validation for moving-neuron recordings.
