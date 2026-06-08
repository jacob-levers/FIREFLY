# Changelog

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
