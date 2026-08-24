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
    fa_figure       single-run master figure (matplotlib, panels A–S)
    fa_loaders      image + external-localisation (CSV) loaders
    fa_stats_config / fa_twoway   stats config + two-way mixed ANOVA
  ui/               PySide6 GUI: app_qml, QML controllers/views, and bespoke
                    QGraphicsView/QImage viewers for Visualise and ROI editing.
  firefly_worker.py Runs the whole pipeline in a subprocess, streaming progress
                    back to the GUI via a queue (keeps the UI responsive).
  release.py         Single-source version plus bundle/release version helpers.
  sptpalm_analysis.py  Facade that re-exports the analysis functions and imports
                    the version from release.py for compatibility.
  crash_reporter.py  Global excepthook → crash reports + rotating log under
                    ~/Library/Logs/FIREFLY/ (macOS).
```

**Layering invariant:** `firefly/ui` may import `firefly/analysis`; `firefly/analysis`
must NEVER import `firefly/ui` (keeps the analysis usable headless and avoids
import cycles).

**Threading invariant (important):** Qt widgets and scene objects are not
thread-safe. Heavy numerical work and image decoding may run in the analysis
subprocess or a background worker, but all QWidget/QGraphicsScene mutation must
be handed back to the GUI thread. The Visualise controller follows this split
when applying decoded stacks and rendered cluster/super-resolution images.

## Adding an analysis step

1. Implement it in `firefly/analysis/` (pure, no Qt).
2. Add a known-truth test in `tests/test_analysis.py` (assert recovery, not just
   "no exception" — see the MSD / drift / loader tests for the pattern).
3. Wire it into the worker (`firefly_worker.py`) and the relevant UI mixin.
4. If it writes a new output file, add it to the run manifest.

## Testing

- `pytest` (CI runs it via `.github/workflows/tests.yml`).
- Tests are headless: matplotlib uses the Agg backend; Qt/QML logic uses
  `QT_QPA_PLATFORM=offscreen`. Keep native Qt object lifetimes explicit in tests:
  close windows, controllers, and QML engines, then process deferred events.
- Keep tests deterministic (seeded RNG) and fast (small synthetic inputs).

## Releasing

1. Bump `__version__` in `firefly/release.py` (single source — `pyproject.toml`
   reads it, and `scripts/stamp_version.py` plus the CI build jobs stamp that
   same assignment).
2. Add a `CHANGELOG.md` entry.
3. Commit, `git tag vX.Y.Z`, `git push origin main --tags`.
4. `.github/workflows/build.yml` builds the frozen apps from the tag.
