# FIREFLY — Developer guide

Onboarding for working on FIREFLY (sptPALM single-particle-tracking PALM
analysis). User-facing docs are in `README.md` and the two methods PDFs.

## Setup

```bash
git clone <repo> && cd sptPALM
python3 -m venv sptpalm-env && source sptpalm-env/bin/activate
pip install -e ".[dev]"        # editable install + pytest  (or: pip install -r requirements.txt)
python run_firefly.py          # launch the GUI
pytest                         # run the test suite
```

`run_firefly.py` (not a console-script entry point) is the launcher: it is also
the multiprocessing **spawn** main module, so it carries the `freeze_support()`
guard that worker children re-import — keep it as the entry point.

## Architecture

```
firefly/
  analysis/         Pure analysis — NO Qt imports. Each stage is one module:
    fa_localize     spot detection (Trackpy / Torch backends)
    fa_linking      trajectory linking (trackpy)
    fa_diffusion    MSD + anomalous (D, alpha) fit
    fa_drift        RCC drift correction
    fa_clustering   DBSCAN clustering
    fa_circular     turning-angle / circular statistics
    fa_compare      multi-group comparison figure + stats
    fa_figure       single-run master figure (matplotlib, panels A–Q)
    fa_loaders      image + external-localisation (CSV) loaders
    fa_stats_config / fa_twoway   stats config + two-way mixed ANOVA
  ui/               PySide6 GUI: app_qt (MainWindow) + ui_mixin_* + ui_widgets,
                    with an embedded napari viewer (Visualise tab).
  firefly_worker.py Runs the whole pipeline in a subprocess, streaming progress
                    back to the GUI via a queue (keeps the UI responsive).
  sptpalm_analysis.py  Facade that re-exports the analysis functions AND holds
                    the single-source __version__ (line ~14).
  crash_reporter.py  Global excepthook → crash reports + rotating log under
                    ~/Library/Logs/FIREFLY/ (macOS).
```

**Layering invariant:** `firefly/ui` may import `firefly/analysis`; `firefly/analysis`
must NEVER import `firefly/ui` (keeps the analysis usable headless and avoids
import cycles).

**Threading invariant (important):** napari/vispy GL rendering is not
thread-safe — heavy compute must run **either in the worker subprocess** (the
main analysis run) **or synchronously on the GUI thread**, never on a background
QThread that then touches napari. The Visualise tab's live DBSCAN re-cluster runs
synchronously (kept bounded by an eps memory-guard in `compute_clusters`) for
exactly this reason; an earlier QThread version segfaulted vispy on macOS.

## Adding an analysis step

1. Implement it in `firefly/analysis/` (pure, no Qt).
2. Add a known-truth test in `tests/test_analysis.py` (assert recovery, not just
   "no exception" — see the MSD / drift / loader tests for the pattern).
3. Wire it into the worker (`firefly_worker.py`) and the relevant UI mixin.
4. If it writes a new output file, add it to the run manifest.

## Testing

- `pytest` (CI runs it via `.github/workflows/tests.yml`).
- Tests are headless: matplotlib uses the Agg backend; Qt logic uses
  `QT_QPA_PLATFORM=offscreen`. Do NOT pump the offscreen GL event loop in tests —
  it segfaults; build the window and assert wiring, don't drive napari rendering.
- Keep tests deterministic (seeded RNG) and fast (small synthetic inputs).

## Releasing

1. Bump `__version__` in `firefly/sptpalm_analysis.py` (single source — `pyproject.toml`
   reads it, and the CI build job stamps the same line).
2. Add a `CHANGELOG.md` entry.
3. Commit, `git tag vX.Y.Z`, `git push origin main --tags`.
4. `.github/workflows/build.yml` builds the frozen apps from the tag.
