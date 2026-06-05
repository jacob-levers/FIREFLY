# Changelog

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
