# Changelog

## Unreleased — `improve/auto-enhance` branch

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
- **Compare** — reports per-replicate α₂ and VACF persistence and statistically
  tests them across groups (omnibus + pairwise), in the summary CSV, stats CSV
  and PDF report.

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
